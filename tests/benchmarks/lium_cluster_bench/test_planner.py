# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure planner and analysis behaviour on synthetic histories."""

from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.lium_cluster_bench.adapters import profile_store
from benchmarks.lium_cluster_bench.domain import analysis, search
from benchmarks.lium_cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
    AuditResult,
    Candidate,
    DeployResult,
    EngagementResult,
    GridSpec,
    PointMetrics,
    PointResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

BIG = search.Budget(
    remaining_s=1e9, deploy_s=900, point_s={"screen": 900, "decision": 1800}
)


def _ctx(gpu="h200", nic_counts=(8, 4, 2, 1)):
    dep = profile_store.load_fitted_profile("glm-5.3", gpu)
    families = tuple(
        sorted((f for f in dep.families if f.enabled), key=lambda f: f.priority)
    )
    grids = {f.id: GridSpec(start=64, max=512, min=8) for f in families}
    return search.SearchContext(
        families=families, ladder=dep.ladder, grids=grids, nic_counts=nic_counts
    )


_clock = [0.0]


def point(cand, c, tps, *, user=60.0, ttft=800.0, purpose="search", spec="screen", status="valid",
          cross=0.0, rep=0, nic_engaged=None):  # fmt: skip
    _clock[0] += 1000
    key = search.RunPoint(cand, c, spec, purpose, rep=rep).key
    return PointResult(
        key=key,
        candidate=cand,
        gpu_count=32,
        metrics=PointMetrics(
            user_tps_p50=user,
            ttft_p50_ms=ttft,
            request_count=10 * c,
            output_tps=tps * 32,
        ),
        audit=AuditResult(status=status),
        tps_per_gpu=tps,
        cross_node_gbs=cross,
        nic_engaged=nic_engaged,
        started_at=_clock[0],
        ended_at=_clock[0] + 900,
    )


def ok_deploy(cand):
    return DeployResult(candidate=cand, cand_fp=cand.fp, ok=True, load_seconds=900)


def test_first_point_then_climb_then_saturate():
    ctx = _ctx()
    f0 = Candidate(family_id="agg-tp8")
    action = search.decide_search(ctx, search.History(), BIG)
    assert isinstance(action, search.RunPoint) and action.concurrency == 64
    h = search.History(points=(point(f0, 64, 50),), deploys=(ok_deploy(f0),))
    assert search.decide_search(ctx, h, BIG).concurrency == 128
    h = search.History(
        points=(point(f0, 64, 50), point(f0, 128, 70)), deploys=(ok_deploy(f0),)
    )
    assert search.decide_search(ctx, h, BIG).concurrency == 256
    h = search.History(
        points=(point(f0, 64, 50), point(f0, 128, 70), point(f0, 256, 71)),
        deploys=(ok_deploy(f0),),
    )
    closed = search.decide_search(ctx, h, BIG)
    assert isinstance(closed, search.CloseFamily) and closed.status == "measured"
    sat, _ = analysis.saturation(list(h.points), None)
    assert sat


def test_climb_stops_on_low_user_rate_and_invalid_point():
    ctx = _ctx()
    f0 = Candidate(family_id="agg-tp8")
    h = search.History(
        points=(point(f0, 64, 50), point(f0, 128, 80, user=8)), deploys=(ok_deploy(f0),)
    )
    assert isinstance(search.decide_search(ctx, h, BIG), search.CloseFamily)
    h = search.History(
        points=(
            point(f0, 64, 50),
            point(f0, 128, 80),
            point(f0, 256, 0, status="invalid"),
        ),
        deploys=(ok_deploy(f0),),
    )
    assert isinstance(search.decide_search(ctx, h, BIG), search.CloseFamily)


def test_prune_dominated_family_and_pick_alternate_by_symptom():
    ctx = _ctx()
    f0 = Candidate(family_id="agg-tp8")
    pd = Candidate(family_id="pd-2p2d")
    pts = (point(f0, 64, 100), point(f0, 128, 101))
    closed = {"agg-tp8": ("measured", "")}
    h = search.History(
        points=pts + (point(pd, 64, 40), point(pd, 128, 45)),
        deploys=(ok_deploy(f0), ok_deploy(pd)),
        closed=closed,
    )
    pruned = search.decide_search(ctx, h, BIG)
    assert (
        isinstance(pruned, search.CloseFamily)
        and pruned.status == "pruned"
        and pruned.family_id == "pd-2p2d"
    )
    # PD primary measured and decode-bound -> the decode-heavy alternate (1P3D) is chosen.
    closed = {"agg-tp8": ("measured", ""), "pd-2p2d": ("measured", "")}
    h = search.History(
        points=pts + (point(pd, 64, 99, user=30),),
        deploys=(ok_deploy(f0), ok_deploy(pd)),
        closed=closed,
    )
    action = search.decide_search(ctx, h, BIG)
    assert (
        isinstance(action, search.RunPoint) and action.candidate.family_id == "pd-1p3d"
    )
    # ...and the prefill-heavy alternate is pruned once 1P3D is closed.
    closed["pd-1p3d"] = ("measured", "")
    action = search.decide_search(
        ctx, search.History(points=h.points, deploys=h.deploys, closed=closed), BIG
    )
    assert isinstance(action, search.CloseFamily) and action.family_id == "pd-3p1d"


def test_kernel_failure_walks_ladder_then_marks_unsupported():
    dep = profile_store.load_fitted_profile("glm-5.3", "rtx_pro_6000")
    families = tuple(sorted(dep.families, key=lambda f: f.priority))
    ctx = search.SearchContext(
        families=families,
        ladder=dep.ladder,
        grids={f.id: GridSpec(start=32, max=256) for f in families},
    )
    base = Candidate(family_id="agg-dtp8")
    failed = DeployResult(
        candidate=base,
        cand_fp=base.fp,
        ok=False,
        failure="kernel_unsupported",
        detail="no kernel image",
    )
    nxt = search.decide_search(ctx, search.History(deploys=(failed,)), BIG)
    assert nxt.candidate.rungs == ("sm120-dsa-tilelang",)
    deploys = [failed]
    cand = base
    for rung in [r.id for r in dep.ladder if r.when == "kernel_unsupported"]:
        cand = cand.with_rung(rung)
        deploys.append(
            DeployResult(
                candidate=cand, cand_fp=cand.fp, ok=False, failure="kernel_unsupported"
            )
        )
    final = search.decide_search(ctx, search.History(deploys=tuple(deploys)), BIG)
    assert isinstance(final, search.CloseFamily) and final.status == "unsupported"


def test_kernel_rungs_found_once_are_inherited():
    ctx = _ctx("rtx_pro_6000")
    smoke = Candidate(family_id="agg-dtp8@smoke", rungs=("sm120-dsa-tilelang",))
    action = search.decide_search(ctx, search.History(deploys=(ok_deploy(smoke),)), BIG)
    assert action.candidate.rungs == ("sm120-dsa-tilelang",)


def test_budget_skips_family():
    ctx = _ctx()
    tiny = search.Budget(
        remaining_s=100, deploy_s=900, point_s={"screen": 900, "decision": 1800}
    )
    action = search.decide_search(ctx, search.History(), tiny)
    assert isinstance(action, search.CloseFamily) and action.status == "skipped_budget"


def test_floors_mtp_pilot_nic_confirm_sequence():
    ctx = _ctx()
    agg = Candidate(family_id="agg-tp8")
    pd = Candidate(family_id="pd-2p2d")
    pts = (
        point(agg, 64, 100, user=45), point(agg, 128, 120, user=30), point(agg, 256, 121, user=15),
        point(pd, 64, 90, user=55, cross=6.0), point(pd, 128, 110, user=40, cross=9.0),
    )  # fmt: skip
    h = search.History(points=pts, deploys=(ok_deploy(agg), ok_deploy(pd)))
    floors = search.decide_floors(ctx, h, BIG)
    assert floors.concurrency == 32  # the leader's lowest point is below 50 tok/s/user
    mtp = search.decide_mtp(ctx, h, BIG)
    assert mtp.candidate.mtp is False and mtp.concurrency == 256
    pilot = search.decide_pilot(ctx, h, BIG)
    assert pilot.purpose == "pilot" and pilot.rep == 1
    base, reason = search.nic_base(h)
    assert base.candidate.family_id == "pd-2p2d" and "no cross-node" in reason
    nic = search.decide_nic(ctx, h, BIG)
    assert nic.candidate.nic_count == 1 and nic.concurrency == 128  # extremes first
    assert search.nic_arm_order((8, 4, 2, 1)) == [8, 1, 4, 2]
    confirm = search.decide_confirm(ctx, h, BIG)
    assert (
        confirm.spec == "decision"
        and confirm.candidate == search.final_leader(h).candidate
    )


def test_audit_and_noise():
    spec = DEFAULT_BENCH_SPECS["screen"]
    good = PointMetrics(
        aiperf_version="0.13.0", request_count=500, output_tps=1.0, duration_s=480
    )
    assert analysis.audit_point(good, spec=spec, concurrency=64, expected_aiperf_version="0.13.0", engagement=None,
                                server_alive=True, telemetry_coverage=1.0, nic_restricted=False, nic_engaged=None).status == "valid"  # fmt: skip
    bad = good.model_copy(update={"error_count": 50, "aiperf_version": "0.12.0"})
    audit = analysis.audit_point(bad, spec=spec, concurrency=64, expected_aiperf_version="0.13.0",
                                 engagement=EngagementResult(ok=False, checks={"x": False}), server_alive=False,
                                 telemetry_coverage=0.5, nic_restricted=True, nic_engaged=False)  # fmt: skip
    assert audit.status == "invalid" and len(audit.reasons) >= 4
    noise = analysis.noise_from_samples("k", 64, [100.0, 102.0, 98.0])
    assert noise.noise_floor == pytest.approx(4.303 * 2.0 * (1 + 1 / 3) ** 0.5)
    assert noise.noise_floor_pct == pytest.approx(noise.noise_floor)


def test_floor_interpolation_and_nic_arms():
    cand = Candidate(family_id="agg-tp8")
    pts = [point(cand, 64, 100, user=60), point(cand, 128, 120, user=40)]
    fl = analysis.floor_result(pts, 50.0)
    assert fl.crossed and fl.tps_per_gpu == 100
    assert 100 < fl.interpolated_tps_per_gpu < 120
    assert not analysis.floor_result(pts, 80.0).crossed
    full = point(cand, 128, 120)
    arms = analysis.nic_arms(
        [
            (8, full),
            (
                1,
                point(
                    cand.model_copy(update={"nic_count": 1}), 128, 60, nic_engaged=True
                ),
            ),
            (2, None),
        ],
        None,
    )
    by_n = {a.nic_count: a for a in arms}
    assert (
        by_n[1].delta_vs_full_pct == pytest.approx(-50.0)
        and by_n[1].within_noise is False
    )
    assert by_n[2].status == "skipped"


def test_accuracy_gate_checks_leaders_in_order_and_skips_failures():
    ctx = _ctx()
    agg = Candidate(family_id="agg-tp8")
    pd = Candidate(family_id="pd-2p2d")
    pts = (point(agg, 128, 120), point(pd, 128, 110, cross=9.0))
    base = search.History(points=pts, deploys=(ok_deploy(agg), ok_deploy(pd)))
    first = search.decide_accuracy(ctx, base, BIG)
    assert isinstance(first, search.CheckAccuracy) and first.candidate == agg
    failed = replace(base, accuracy={agg.key: "fail"})
    assert search.final_leader(failed).candidate == pd  # a failed config never leads
    nxt = search.decide_accuracy(ctx, failed, BIG)
    assert isinstance(nxt, search.CheckAccuracy) and nxt.candidate == pd
    assert "1 failed" in nxt.reason
    passed = replace(base, accuracy={agg.key: "fail", pd.key: "pass"})
    assert search.decide_accuracy(ctx, passed, BIG).reason.startswith(
        "the leader passed"
    )
    # confirm and pilot follow the passing leader
    assert search.decide_confirm(ctx, passed, BIG).candidate == pd
    errored = replace(base, accuracy={agg.key: "error"})
    assert "unchecked" in search.decide_accuracy(ctx, errored, BIG).reason
    exhausted = replace(base, accuracy={agg.key: "fail"})
    assert (
        "1 accuracy checks used"
        in search.decide_accuracy(ctx, exhausted, BIG, max_checks=1).reason
    )
    poor = search.Budget(remaining_s=100.0, deploy_s=900.0, point_s=BIG.point_s)
    assert "budget" in search.decide_accuracy(ctx, base, poor).reason


# --------------------------------------------------------------------------
# Budget reservations for the steps after the current one
# --------------------------------------------------------------------------

SCREEN_S, DECISION_S, DEPLOY_S = 960.0, 1800.0, 900.0
COSTS = search.StepCosts(
    deploy_s=DEPLOY_S,
    point_s={"screen": SCREEN_S, "decision": DECISION_S},
    accuracy=True,
    finalize_s=600.0,
    margin=0.0,
)
LEAD = Candidate(family_id="agg-tp8")
OTHER = Candidate(family_id="agg-dp8")


def _led(*, user: float = 60.0, cross: float = 0.0, running: Candidate | None = LEAD):
    """agg-tp8 measured at c=64 and c=128 (its knee) and currently deployed."""
    return search.History(
        points=(
            point(LEAD, 64, 50, user=user),
            point(LEAD, 128, 70, user=user - 10, cross=cross),
        ),
        deploys=(ok_deploy(LEAD),),
        deployed_key=running.key if running else None,
    )


def test_floor_bracketing_is_reserved_only_when_the_leader_needs_it():
    ctx = _ctx()
    assert search.reserve_plan(ctx, _led(user=80.0), "search", COSTS)["floors"] == 0.0
    # 64 -> 32 -> 16 -> 8: three descents, the grid minimum and the cap agree
    slow = search.reserve_plan(ctx, _led(user=30.0), "search", COSTS)
    assert slow["floors"] == 3 * SCREEN_S
    # another configuration runs when the search ends: the leader comes back
    elsewhere = search.reserve_plan(
        ctx, _led(user=30.0, running=OTHER), "search", COSTS
    )
    assert elsewhere["floors"] == DEPLOY_S + 3 * SCREEN_S
    # nothing measured yet: the default two bracketing points
    unknown = search.reserve_plan(ctx, search.History(), "search", COSTS)
    assert unknown["floors"] == DEPLOY_S + 2 * SCREEN_S


def test_the_accuracy_check_leaves_the_leader_running_for_the_pilot():
    """Regression: costing the pilot with its own redeploy made the accuracy
    check unaffordable although the check itself redeploys the leader."""
    ctx = _ctx()
    off = LEAD.model_copy(update={"mtp": False})
    h = _led(cross=5.0)
    after_mtp = replace(
        h,
        points=h.points + (point(off, 128, 60, purpose="mtp_ablation", cross=5.0),),
        deploys=h.deploys + (ok_deploy(off),),
        deployed_key=off.key,
    )
    plan = search.reserve_plan(ctx, after_mtp, "accuracy", COSTS)
    assert plan["pilot"] == 2 * SCREEN_S  # no redeploy
    # the knee is measured at full NIC count; N=1, 4, 2 each redeploy
    assert plan["nic_sweep"] == 3 * (DEPLOY_S + SCREEN_S)
    assert plan["confirm"] == DEPLOY_S + DECISION_S  # the last NIC arm still runs
    # seen from the search, the MTP-off variant runs before the check
    during = search.reserve_plan(ctx, h, "search", COSTS)
    assert during["mtp"] == DEPLOY_S + SCREEN_S
    assert during["accuracy"] == DEPLOY_S + search.ACCURACY_CHECK_S
    assert during["pilot"] == 2 * SCREEN_S
    no_gate = search.reserve_plan(ctx, h, "search", replace(COSTS, accuracy=False))
    assert no_gate["accuracy"] == 0.0 and no_gate["pilot"] == DEPLOY_S + 2 * SCREEN_S


def test_nic_arms_stay_reserved_until_the_search_rules_out_cross_node_traffic():
    ctx = _ctx()
    h = _led(cross=0.0)
    assert search.reserve_plan(ctx, h, "search", COSTS)["nic_sweep"] == 3 * (
        DEPLOY_S + SCREEN_S
    )  # a configuration with cross-node traffic may still be measured
    assert search.reserve_plan(ctx, h, "accuracy", COSTS)["nic_sweep"] == 0.0
    two = search.reserve_plan(_ctx(nic_counts=(8, 1)), _led(cross=5.0), "pilot", COSTS)
    assert two["nic_sweep"] == DEPLOY_S + SCREEN_S


def test_reservation_uses_the_leader_familys_slowest_deploy():
    ctx = _ctx()
    h = replace(
        _led(),
        deploys=(
            ok_deploy(LEAD),
            DeployResult(candidate=LEAD, cand_fp=LEAD.fp, ok=True, load_seconds=1500),
            DeployResult(candidate=OTHER, cand_fp=OTHER.fp, ok=True, load_seconds=300),
        ),
    )
    assert search.reserve_plan(ctx, h, "search", COSTS)["mtp"] == 1500 + SCREEN_S


def test_reserve_after_adds_the_margin_but_keeps_finalize_fixed():
    ctx = _ctx()
    h = _led(cross=5.0)
    costs = replace(COSTS, margin=0.1)
    plan = search.reserve_plan(ctx, h, "nic_sweep", costs)
    steps = sum(v for k, v in plan.items() if k != "finalize")
    assert steps > 0
    reserve = search.reserve_after(ctx, h, "nic_sweep", costs)
    assert reserve == pytest.approx(steps * 1.1 + 600.0)
    assert search.reserve_after(ctx, h, "confirm", costs) == pytest.approx(600.0)
    assert search.reserve_after(ctx, h, "finalize", costs) == 0.0
    assert search.reserve_after(ctx, h, "not-a-step", costs) == 0.0
