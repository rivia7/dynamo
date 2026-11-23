# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P-Only and D-Only stages: derived families, worker configs, objectives,
audit, search and report wording (pure; no GPUs). The dry-run tests cover the
whole pipeline and the KB's PD split."""

from __future__ import annotations

from typing import Any, Iterator

import pytest
import yaml

from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.engine_sglang import SglangEngine
from benchmarks.cluster_bench.app import measure
from benchmarks.cluster_bench.app.simulate import synthetic_inventory
from benchmarks.cluster_bench.domain import analysis
from benchmarks.cluster_bench.domain import profiles as prof
from benchmarks.cluster_bench.domain import search, stages
from benchmarks.cluster_bench.domain.profiles import ProfileError
from benchmarks.cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
    Candidate,
    GridSpec,
    PointMetrics,
    RunConfig,
)
from benchmarks.cluster_bench.report.strings import Strings

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

GPUS = ("b200", "b300", "h200", "h100", "rtx_pro_6000", "rtx_pro_6000d")
RECIPES = profile_store.RecipeLoader()
BIG = search.Budget(
    remaining_s=1e9, deploy_s=900, point_s={"screen": 900, "decision": 1800}
)


def role_config(role: Any) -> dict[str, Any]:
    return prof.merge_config(
        RECIPES.config(role.base) if role.base else {}, role.sglang
    )


def stage_profile(gpu: str, stage: str):
    base = profile_store.load_deployment_profile("glm-5.3", gpu)
    return stages.stage_deployment(base, stage, role_config)


@pytest.mark.parametrize(
    "text, stage",
    [
        ("e2e", "e2e"),
        ("PD", "e2e"),
        ("p-only", "prefill"),
        ("P_Only", "prefill"),
        ("prefill", "prefill"),
        ("d-only", "decode"),
        (" D ", "decode"),
    ],
)
def test_parse_stage(text: str, stage: str) -> None:
    assert stages.parse_stage(text) == stage


def test_parse_stage_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="use e2e, p-only or d-only"):
        stages.parse_stage("a-only")


@pytest.mark.parametrize("gpu", GPUS)
@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_stage_families_come_from_the_pd_roles(gpu: str, stage: str) -> None:
    base = profile_store.load_deployment_profile("glm-5.3", gpu)
    pd = {
        prof.reference_id(f): f
        for f in base.families
        if {r.role for r in f.roles} >= {"prefill", "decode"}
    }
    dep = stage_profile(gpu, stage)
    assert dep.families, gpu
    for family in dep.families:
        assert family.stage == stage and family.source in pd
        assert family.id.startswith(f"{stages.STAGE_PREFIX[stage]}-")
        assert {r.role for r in family.roles} == {stage}
        assert family.transport == "none"
        # the stage role fills every GPU of the family it came from
        assert family.total_gpus == pd[family.source].total_gpus
    signatures = [stages._role_signature(f.roles[0]) for f in dep.families]
    assert len(signatures) == len(set(signatures))


@pytest.mark.parametrize(
    "nodes, ids",
    [(1, ["p-only-tp8ep8x1"]), (2, ["p-only-tp8ep8x2"]), (8, ["p-only-tp8ep8x8"])],
)
def test_stage_families_fit_any_cluster_size(nodes: int, ids: list[str]) -> None:
    fitted, _dropped = prof.fit_deployment(
        stage_profile("h200", "prefill"), nodes * 8, 8
    )
    assert [f.id for f in fitted.families] == ids


def test_cross_node_stage_roles_need_their_nodes() -> None:
    dep = stage_profile("h100", "prefill")
    (family,) = dep.families
    assert family.id == "p-only-tp16ep16x2" and family.cross_node == ("collective",)
    one_node, dropped = prof.fit_deployment(dep, 8, 8)
    assert one_node.families == () and dropped == ("p-only-tp16ep16x2",)


def test_d_only_grid_and_lane_cap_widen_by_the_decode_share() -> None:
    dep = stage_profile("h200", "decode")
    (family,) = dep.families
    assert (family.id, family.source, family.roles[0].instances) == (
        "d-only-dtp8x4",
        "pd-2p2d",
        4,
    )
    # pd-2p2d decodes on half its GPUs, so every lane bound doubles
    assert family.grid == GridSpec(start=128, max=1024, min=8)
    assert stages.stage_lane_cap(family, 512, dep.default_grid) == 1024
    (p_only,) = stage_profile("h200", "prefill").families
    assert p_only.grid == dep.default_grid
    assert stages.stage_lane_cap(p_only, 512, dep.default_grid) == 512


def test_profile_stage_families_win_and_are_checked() -> None:
    base = profile_store.load_deployment_profile("glm-5.3", "b200")
    derived = stage_profile("b200", "decode").families
    with_own = base.model_copy(update={"families": base.families + derived[:1]})
    again = stages.stage_deployment(with_own, "decode", role_config)
    assert [f.id for f in again.families] == [f.id for f in derived]
    assert stages.stage_deployment(with_own, "e2e").families == base.families
    wrong = derived[0].model_copy(update={"stage": "prefill"})
    broken = base.model_copy(update={"families": base.families + (wrong,)})
    with pytest.raises(ProfileError, match="may only have prefill roles"):
        stages.stage_deployment(broken, "prefill")


def test_apply_stage_prefill_runs_the_role_aggregated() -> None:
    cfg = {
        "tp-size": 8,
        "disaggregation-mode": "prefill",
        "disaggregation-ib-device": "mlx5_0",
        "disaggregation-bootstrap-port": 8998,
        "cuda-graph-max-bs-decode": 64,
    }
    out, env = stages.apply_stage(
        "prefill", "prefill", cfg, {stages.DECODE_ONLY_ENV: "1", "X": "y"}
    )
    assert out == {
        "tp-size": 8,
        "cuda-graph-max-bs-decode": 64,
        "cuda-graph-backend-decode": "disabled",
    }
    assert env == {"X": "y"}
    assert stages.apply_stage("prefill", "prefill", out, env) == (out, env)


def test_apply_stage_decode_fakes_the_kv_transfer() -> None:
    cfg = {
        "disaggregation-mode": "decode",
        "disaggregation-ib-device": "mlx5_0",
        "disaggregation-transfer-backend": "nixl",
    }
    out, env = stages.apply_stage("decode", "decode", cfg, {})
    assert out == {
        "disaggregation-mode": "decode",
        "disaggregation-transfer-backend": "fake",
    }
    assert env == {stages.DECODE_ONLY_ENV: "1"}
    assert stages.apply_stage("e2e", "prefill", cfg, {}) == (cfg, {})
    with pytest.raises(ProfileError, match="cannot run role 'prefill'"):
        stages.apply_stage("decode", "prefill", cfg, {})
    assert stages.stage_flags("decode") == {
        "--disaggregation-mode",
        "--disaggregation-transfer-backend",
    }
    assert stages.stage_flags("prefill") == {"--cuda-graph-backend-decode"}
    assert stages.stage_flags("e2e") == set()


def render_stage(tmp_path, gpu: str, stage: str, nodes: int = 2):
    fitted, _ = prof.fit_deployment(stage_profile(gpu, stage), nodes * 8, 8)
    family = fitted.families[0]
    plan = SglangEngine().render(
        Candidate(family_id=family.id, simulate_acceptance=stage == "decode"),
        config=RunConfig(
            exp_id="exp-1", image="img:1", workdir=str(tmp_path), stage=stage
        ),
        inventory=synthetic_inventory(gpu, nodes=nodes),
        gpu=profile_store.load_gpu_profiles()[gpu],
        model=profile_store.load_model_profile("glm-5.3"),
        deployment=fitted,
        model_dir="/workspace/models/glm",
    )
    workers = []
    for spec in plan.containers:
        if spec.role in ("control", "frontend"):
            continue
        path = next(p for p in spec.files if p.endswith(".yaml"))
        workers.append((spec, yaml.safe_load(spec.files[path])))
    return plan, workers


@pytest.mark.parametrize("gpu", ["b200", "h200", "rtx_pro_6000"])
def test_p_only_workers_run_the_prefill_role_aggregated(tmp_path, gpu: str) -> None:
    plan, workers = render_stage(tmp_path, gpu, "prefill")
    assert plan.stage == "prefill" and workers
    for spec, cfg in workers:
        assert spec.role == "prefill"
        assert [k for k in cfg if k.startswith("disaggregation-")] == []
        assert cfg["cuda-graph-backend-decode"] == "disabled"
        assert stages.DECODE_ONLY_ENV not in spec.env


@pytest.mark.parametrize("gpu", ["b200", "h200", "h100"])
def test_d_only_workers_decode_with_a_fake_kv_transfer(tmp_path, gpu: str) -> None:
    plan, workers = render_stage(tmp_path, gpu, "decode")
    assert plan.stage == "decode" and workers
    for spec, cfg in workers:
        assert spec.role == "decode"
        assert cfg["disaggregation-mode"] == "decode"
        assert cfg["disaggregation-transfer-backend"] == "fake"
        assert "disaggregation-ib-device" not in cfg
        assert spec.env[stages.DECODE_ONLY_ENV] == "1"


def test_pd_split_balances_prefill_and_decode_seconds() -> None:
    # a request: 8000 / 20000 = 0.4 prefill GPU-s and 500 / 100 = 5 decode GPU-s
    ratio, tps = stages.pd_split(isl=8000, osl=500, prefill_tps=20000, decode_tps=100)
    assert ratio == pytest.approx(0.08)
    assert tps == pytest.approx(500 / 5.4)
    with pytest.raises(ValueError):
        stages.pd_split(8000, 0, 20000, 100)


def test_objective_counts_prompt_tokens_for_p_only() -> None:
    m = PointMetrics(
        output_tps=800.0, input_tps=64000.0, total_output_tokens=4000, duration_s=500.0
    )
    assert measure.objective_per_gpu(m, 16, "prefill") == 4000.0
    assert measure.objective_per_gpu(m, 16, "decode") == 50.0
    assert measure.objective_per_gpu(m, 16, "e2e") == 50.0
    assert measure.objective_per_gpu(None, 16, "prefill") is None
    no_input = m.model_copy(update={"input_tps": None})
    assert measure.objective_per_gpu(no_input, 16, "prefill") is None
    assert measure.objective_tokens(m, "prefill") == 64000.0 * 500.0
    assert measure.objective_tokens(m, "decode") == 4000
    assert stages.objective_tokens("prefill") == "input"


def test_p_only_audit_needs_input_tps_and_one_output_token() -> None:
    common: dict[str, Any] = dict(
        spec=DEFAULT_BENCH_SPECS["screen"],
        concurrency=64,
        expected_aiperf_version="0.14.0",
        engagement=None,
        server_alive=True,
        telemetry_coverage=1.0,
        nic_restricted=False,
        nic_engaged=None,
    )
    good = PointMetrics(
        aiperf_version="0.14.0",
        request_count=500,
        output_tps=50.0,
        input_tps=400000.0,
        duration_s=480,
        osl_avg=1.0,
    )
    assert analysis.audit_point(good, stage="prefill", **common).status == "valid"
    uncapped = good.model_copy(update={"osl_avg": 37.0})
    audit = analysis.audit_point(uncapped, stage="prefill", **common)
    assert audit.status == "invalid"
    assert any("one-token cap was not honoured" in r for r in audit.reasons)
    assert analysis.audit_point(uncapped, **common).status == "valid"  # fine for e2e
    no_input = good.model_copy(update={"input_tps": None})
    audit = analysis.audit_point(no_input, stage="prefill", **common)
    assert audit.status == "invalid" and any(
        "input_token_throughput" in r for r in audit.reasons
    )


def test_p_only_has_no_interactivity_floors() -> None:
    p_only = search.SearchContext(families=(), ladder=(), grids={}, stage="prefill")
    action = search.decide_floors(p_only, search.History(), BIG)
    assert isinstance(action, search.Done) and "P-Only" in action.reason
    assert search.floor_points_needed(p_only, search.History(), None) == 0
    e2e = search.SearchContext(families=(), ladder=(), grids={})
    assert (
        search.floor_points_needed(e2e, search.History(), None)
        == search.FLOOR_POINTS_UNKNOWN
    )


def _texts(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _texts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _texts(value)


def test_report_wording_follows_the_stage() -> None:
    base = Strings.load()
    assert base.for_stage("e2e") is base
    p_only = base.for_stage("prefill")
    assert [t for t in _texts(p_only._table) if "输出 tok/s/GPU" in t] == []
    assert p_only._template("method.objective") != base._template("method.objective")
    assert p_only._template("summary.best") != base._template("summary.best")
    d_only = base.for_stage("decode")
    assert d_only._template("method.objective") != base._template("method.objective")
    for stage in ("prefill", "decode"):
        assert base._template(f"stages.banner.{stage}")


def test_glm_5_3_flash_d_only_on_rtx_pro_6000d(tmp_path) -> None:
    base = profile_store.load_deployment_profile("glm-5.3-flash", "rtx_pro_6000d")
    staged = stages.stage_deployment(base, "decode", role_config)
    # lanes at profile scale (4 nodes): the D-Only climb on 2 nodes runs 32..256
    assert {f.grid for f in staged.families} == {GridSpec(start=64, max=512, min=2)}
    fitted, _ = prof.fit_deployment(staged, 16, 8)
    assert [f.id for f in fitted.families] == ["d-only-tp4x4", "d-only-tp8x2"]
    model = profile_store.load_model_profile("glm-5.3-flash")
    gpu = profile_store.load_gpu_profiles()["rtx_pro_6000d"]
    model_dir = "/data/models/GLM-5.3-Flash-NVFP4"
    for family, tp in zip(fitted.families, (4, 8)):
        for mtp in (True, False):
            plan = SglangEngine().render(
                Candidate(family_id=family.id, mtp=mtp, simulate_acceptance=mtp),
                config=RunConfig(
                    exp_id="exp-1",
                    image="img:1",
                    workdir=str(tmp_path),
                    stage="decode",
                    checkpoint="nvfp4-nvidia",
                ),
                inventory=synthetic_inventory("rtx_pro_6000d", nodes=2),
                gpu=gpu,
                model=model,
                deployment=fitted,
                model_dir=model_dir,
            )
            workers = [
                (spec, yaml.safe_load(spec.files[path]))
                for spec in plan.containers
                if spec.role not in ("control", "frontend")
                for path in spec.files
                if path.endswith(".yaml")
            ]
            assert sorted(len(s.gpus) for s, _ in workers) == [tp] * (16 // tp)
            for spec, cfg in workers:
                assert spec.role == "decode" and spec.env[stages.DECODE_ONLY_ENV] == "1"
                assert (
                    spec.env["CB_SGLANG_HOTFIXES"] == "sm120-tilelang-dsa-v1-one-stage"
                )
                assert cfg["disaggregation-transfer-backend"] == "fake"
                assert cfg["tensor-parallel-size"] == tp
                assert (cfg["kv-cache-dtype"], cfg["dsa-decode-backend"]) == (
                    "bfloat16",
                    "tilelang",
                )
                assert cfg["quantization"] == "modelopt_fp4"
                assert cfg["disable-shared-experts-fusion"] is True
                assert cfg["model-path"] == model_dir
                assert cfg["served-model-name"] == "zai-org/GLM-5.3-Flash"
                assert cfg["context-length"] == 229376
                # without a draft model there is no ReplaySSM ring either
                assert ("speculative-algorithm" in cfg) == mtp
                assert ("enable-linear-replayssm-spec" in cfg) == mtp
                assert ("SGLANG_SIMULATE_ACC_LEN" in spec.env) == mtp
