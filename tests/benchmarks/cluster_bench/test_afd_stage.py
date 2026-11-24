# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A-Only stage wiring (pure; no GPUs): families, the A worker's launch plan,
the null-F bound's planning and analysis, --afd-f on the command line and the
report wording. The F model is in test_afd_model.py, the worker agent in
test_afd_agent.py and the whole pipeline in test_dry_run.py."""

from __future__ import annotations

from dataclasses import replace
from typing import get_args

import pytest

from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.engine_sglang import RenderError, SglangEngine
from benchmarks.cluster_bench.agents import sglang_hotfix
from benchmarks.cluster_bench.app.simulate import synthetic_inventory
from benchmarks.cluster_bench.cli import main
from benchmarks.cluster_bench.domain import afd, analysis
from benchmarks.cluster_bench.domain import profiles as prof
from benchmarks.cluster_bench.domain import search, stages
from benchmarks.cluster_bench.domain.schema import (
    AfdBound,
    AuditResult,
    Candidate,
    DeployResult,
    GridSpec,
    NoiseResult,
    PointMetrics,
    PointResult,
    RunConfig,
    Stage,
)
from benchmarks.cluster_bench.report.reasons import AFD_TERMS, localize
from benchmarks.cluster_bench.report.strings import Strings
from tests.benchmarks.cluster_bench.test_afd_model import GLM
from tests.benchmarks.cluster_bench.test_afd_model import plan as f_plan

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

RECIPES = profile_store.RecipeLoader()
GPU = profile_store.load_gpu_profiles()["rtx_pro_6000d"]
INVENTORY = synthetic_inventory("rtx_pro_6000d", nodes=2)
BIG = search.Budget(
    remaining_s=1e9, deploy_s=900, point_s={"screen": 900, "decision": 1800}
)
COSTS = search.StepCosts(deploy_s=900, point_s={"screen": 900, "decision": 1800})
LEAD = Candidate(family_id="a-only-dtp8x2", simulate_acceptance=True)
NULL = LEAD.model_copy(update={"null_f": True})
NOISE = NoiseResult(
    cand_key=LEAD.key,
    concurrency=256,
    samples=(85.0, 86.0, 84.0),
    mean=85.0,
    stdev=1.0,
    cv=0.012,
    noise_floor=1.7,
    noise_floor_pct=2.0,
)


def staged(model: str, gpu: str, stage: str = "attention"):
    base = profile_store.load_deployment_profile(model, gpu)
    return stages.stage_deployment(base, stage, RECIPES.role_config)


def fitted(model: str = "glm-5.3", gpu: str = "rtx_pro_6000d", nodes: int = 2):
    return prof.fit_deployment(staged(model, gpu), nodes * 8, 8)[0]


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, ids",
    [
        ("glm-5.3", ["a-only-dtp8x2"]),
        ("glm-5.3-flash", ["a-only-dtp4x4", "a-only-dtp8x2"]),
    ],
)
def test_rtx_pro_6000d_runs_the_profiles_own_a_only_families(model, ids) -> None:
    dep = fitted(model)
    assert [f.id for f in dep.families] == ids
    for family in dep.families:
        assert family.stage == "attention"
        assert {r.role for r in family.roles} == {"decode"}


def test_fitted_profiles_keep_stage_families_to_their_stage() -> None:
    e2e = profile_store.load_fitted_profile("glm-5.3", "rtx_pro_6000d", nodes=2)
    assert e2e.families and {f.stage for f in e2e.families} == {"e2e"}
    attention = profile_store.load_fitted_profile(
        "glm-5.3", "rtx_pro_6000d", nodes=2, stage="attention"
    )
    assert [f.id for f in attention.families] == ["a-only-dtp8x2"]


def test_a_only_counts_the_a_sides_flops() -> None:
    assert stages.flops_arch("attention", GLM) == afd.a_side_arch(GLM)
    assert stages.flops_arch("decode", GLM) is GLM
    assert stages.flops_arch("attention", None) is None


def test_an_a_worker_needs_its_instance_size() -> None:
    with pytest.raises(ValueError, match="GPU count"):
        stages.apply_stage("attention", "decode", {"tensor-parallel-size": 8}, {})


def test_derived_a_only_families_get_twice_d_onlys_concurrency() -> None:
    (d_only,) = staged("glm-5.3", "h200", "decode").families
    (a_only,) = staged("glm-5.3", "h200").families
    assert (a_only.id, a_only.source, a_only.roles) == (
        "a-only-dtp8x4",
        d_only.source,
        d_only.roles,
    )
    # no routed experts on A: twice D-Only's KV room, so twice its grid max
    assert d_only.grid == GridSpec(start=128, max=1024, min=8)
    assert a_only.grid == GridSpec(start=128, max=2048, min=8)
    assert stages.stage_lane_cap(a_only, 512, GridSpec(start=64, max=512)) == 2048


# --------------------------------------------------------------------------
# The A worker's launch plan
# --------------------------------------------------------------------------


def render(tmp_path, cand: Candidate, plan=None):
    dep = fitted()
    return SglangEngine().render(
        cand,
        config=RunConfig(
            exp_id="exp-1",
            image="img:1",
            workdir=str(tmp_path),
            stage="attention",
            checkpoint="nvfp4-nvidia",
        ),
        inventory=INVENTORY,
        gpu=GPU,
        model=profile_store.load_model_profile("glm-5.3"),
        deployment=dep,
        model_dir="/data/models/GLM-5.3-NVFP4",
        afd=plan,
    )


@pytest.mark.parametrize("cand", [LEAD, NULL], ids=["simulated-f", "null-f"])
def test_a_workers_run_dp_attention_and_wait_for_the_planned_f(tmp_path, cand):
    plan = f_plan()
    launch = render(tmp_path, cand, plan)
    assert launch.stage == "attention"
    assert sorted(launch.rendered_configs) == ["decode:0", "decode:1"]
    for cfg in launch.rendered_configs.values():
        assert (cfg["tensor-parallel-size"], cfg["data-parallel-size"]) == (8, 8)
        assert cfg["enable-dp-attention"] is True
        assert cfg["moe-a2a-backend"] == "none"
        assert cfg["disaggregation-transfer-backend"] == "fake"
    workers = [s for s in launch.containers if s.role not in ("control", "frontend")]
    assert sorted(len(spec.gpus) for spec in workers) == [8, 8]
    wait = afd.F_WAIT_OFF if cand.null_f else afd.f_wait(plan, 8).to_env()
    for spec in workers:
        assert spec.role == "decode"
        # a store-true flag SGLang's YAML loader rejects (engine CLI_ONLY_KEYS)
        assert "--enable-dp-attention" in spec.command
        assert spec.env[afd.ATTN_ONLY_ENV] == "1"
        assert spec.env[afd.TARGET_LAYERS_ENV] == str(plan.layers)
        assert spec.env[afd.F_WAIT_ENV] == wait
        assert afd.HOTFIX_ID in spec.env["CB_SGLANG_HOTFIXES"].split(",")


def test_an_a_only_family_needs_the_runs_f_plan(tmp_path) -> None:
    with pytest.raises(RenderError, match="F plan"):
        render(tmp_path, LEAD)


# --------------------------------------------------------------------------
# The null-F bound: planning and analysis
# --------------------------------------------------------------------------


def _ctx(f_bound: bool = True) -> search.SearchContext:
    families = fitted().families
    grids = {f.id: GridSpec(start=32, max=512, min=2) for f in families}
    return search.SearchContext(
        families=families,
        ladder=(),
        grids=grids,
        simulate_acceptance=True,
        stage="attention",
        f_bound=f_bound,
    )


_clock = [0.0]


def point(cand, c, tps, *, purpose="search", status="valid", reasons=()):
    _clock[0] += 1000
    return PointResult(
        key=search.RunPoint(cand, c, "screen", purpose).key,
        candidate=cand,
        gpu_count=16,
        metrics=PointMetrics(
            user_tps_p50=tps * 16 / c,
            ttft_p50_ms=900.0,
            itl_p50_ms=1000.0 * c / (tps * 16),
            request_count=10 * c,
            output_tps=tps * 16,
        ),
        audit=AuditResult(status=status, reasons=reasons),
        tps_per_gpu=tps,
        started_at=_clock[0],
        ended_at=_clock[0] + 900,
    )


def test_the_leader_is_measured_again_with_the_f_wait_off() -> None:
    ctx = _ctx()
    history = search.History(points=(point(LEAD, 128, 70.0), point(LEAD, 256, 85.0)))
    action = search.decide_f_bound(ctx, history, BIG)
    assert isinstance(action, search.RunPoint)
    assert (action.candidate, action.concurrency, action.spec, action.purpose) == (
        NULL,
        256,
        "screen",
        "f_bound",
    )
    assert NULL.key == f"{LEAD.key}|f=null" and NULL.fp != LEAD.fp
    measured = search.History(
        points=(*history.points, point(NULL, 256, 110.0, purpose="f_bound"))
    )
    failed = search.History(
        points=history.points,
        deploys=(DeployResult(candidate=NULL, cand_fp=NULL.fp, ok=False),),
    )
    cases = [
        (ctx, measured, BIG, "null-F bound measured"),
        (ctx, failed, BIG, "null-F deployment failed"),
        (
            ctx,
            history,
            replace(BIG, remaining_s=1000.0),
            "budget skipped the null-F bound",
        ),
        (ctx, search.History(), BIG, "no leader to bound"),
        (_ctx(f_bound=False), history, BIG, "null-F bound not applicable"),
    ]
    for case_ctx, case_history, budget, reason in cases:
        done = search.decide_f_bound(case_ctx, case_history, budget)
        assert isinstance(done, search.Done) and done.reason == reason
        assert localize(reason) != reason  # the report shows it in Chinese


def test_the_null_f_bound_is_reserved_until_it_is_measured() -> None:
    ctx = _ctx()
    history = search.History(points=(point(LEAD, 256, 85.0),))
    # its own deployment (the wait is a launch env) and one screen point
    assert search.reserve_plan(ctx, history, "confirm", COSTS)["f_bound"] == 1800.0
    done = search.History(
        points=(*history.points, point(NULL, 256, 110.0, purpose="f_bound"))
    )
    assert search.reserve_plan(ctx, done, "confirm", COSTS)["f_bound"] == 0.0
    off = _ctx(f_bound=False)
    assert search.reserve_plan(off, history, "confirm", COSTS)["f_bound"] == 0.0


def test_the_bound_is_the_null_f_throughput_the_wait_costs() -> None:
    sim, null = point(LEAD, 256, 85.0), point(NULL, 256, 110.0, purpose="f_bound")
    bound = analysis.afd_bound(sim, null, NOISE, "null-F bound measured")
    assert (bound.status, bound.concurrency) == ("measured", 256)
    assert (bound.sim_point_id, bound.null_point_id) == (sim.key.id, null.key.id)
    assert bound.f_cost_pct == pytest.approx(100 * 25 / 110)
    assert bound.within_noise is False
    assert (bound.sim_tps_per_gpu, bound.null_tps_per_gpu) == (85.0, 110.0)
    assert bound.null_itl_p50_ms < bound.sim_itl_p50_ms
    close = analysis.afd_bound(point(LEAD, 256, 108.5), null, NOISE)
    assert close.f_cost_pct == pytest.approx(100 * 1.5 / 110) and close.within_noise


def test_a_missing_or_broken_null_f_point_says_why() -> None:
    sim = point(LEAD, 256, 85.0)
    reason = "budget skipped the null-F bound"
    assert analysis.afd_bound(sim, None, NOISE, reason) == AfdBound(
        status="skipped", reason=reason, sim_point_id=sim.key.id
    )
    broken = point(
        NULL,
        256,
        110.0,
        purpose="f_bound",
        status="invalid",
        reasons=("error rate 2.00% >= 1%",),
    )
    # the point says why, not the planner's "measured"
    bound = analysis.afd_bound(sim, broken, NOISE, "null-F bound measured")
    assert (bound.status, bound.reason, bound.f_cost_pct) == (
        "invalid",
        "error rate 2.00% >= 1%",
        None,
    )
    no_sim = analysis.afd_bound(
        None, point(NULL, 256, 110.0, purpose="f_bound"), NOISE, "null-F bound measured"
    )
    assert (no_sim.status, no_sim.reason) == ("invalid", "no usable simulated-F point")
    for text in (bound.reason, no_sim.reason, "null-F point has no throughput"):
        assert localize(text) != text, text


# --------------------------------------------------------------------------
# Command line and wording
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--afd-f", "gpus=4"], "--afd-f applies to --stage a-only only"),
        (
            ["--stage", "d-only", "--afd-f", "gpus=4"],
            "--afd-f applies to --stage a-only only",
        ),
        (["--stage", "a-only", "--afd-f", "speed=1"], "unknown key 'speed'"),
    ],
)
def test_afd_f_is_checked_before_the_run_starts(tmp_path, extra, message) -> None:
    argv = ["dry-run", "--workdir", str(tmp_path), "--exp-id", "t1", *extra]
    with pytest.raises(SystemExit, match=message):
        main(argv)
    assert not (tmp_path / "experiments").exists()


def test_every_pd_stage_has_its_label_and_banner() -> None:
    base = Strings.load()
    for spec in stages.STAGE_SPECS:
        if spec.role is None:
            continue
        assert base._template(f"stages.label.{spec.id}").startswith(spec.label)
        assert base._template(f"stages.banner.{spec.id}")
    a_only = base.for_stage("attention")
    assert a_only._template("method.objective") != base._template("method.objective")
    assert base.label("purpose", "f_bound") != "f_bound"


def test_names_shared_across_layers_agree() -> None:
    assert stages.HOTFIX_ENV == sglang_hotfix.ENV_VAR  # the launcher reads it
    assert get_args(Stage) == tuple(stages.STAGES)
    base = Strings.load()
    for kind, text in AFD_TERMS.items():  # reasons.py and the report's table
        assert base.label("resources.afd_bound_kind", kind) == text


def test_a_only_result_texts_read_in_chinese() -> None:
    spec = stages.stage_spec("attention")
    texts = (
        *spec.conditions,
        *spec.unverified,
        spec.mfu_note,
        f"accuracy gate not applicable to {spec.label}: the stage produces no "
        "end-to-end answers",
    )
    for text in texts:
        assert localize(text) != text, text
