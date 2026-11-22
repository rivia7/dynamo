# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVLink off vs on: switching, planning, rendering, auditing and comparing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.engine_sglang import SglangEngine
from benchmarks.cluster_bench.app.nvlink_ops import NvlinkControl, nvlink_events
from benchmarks.cluster_bench.app.simulate import synthetic_inventory
from benchmarks.cluster_bench.app.store import EventStore
from benchmarks.cluster_bench.domain import analysis
from benchmarks.cluster_bench.domain import nvlink as nvl
from benchmarks.cluster_bench.domain import search
from benchmarks.cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
    AuditResult,
    BandwidthStat,
    Candidate,
    DeployResult,
    GpuProfile,
    GridSpec,
    NoiseResult,
    PeakResult,
    PointMetrics,
    PointResult,
    RunConfig,
)
from benchmarks.cluster_bench.ports import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

BIG = search.Budget(
    remaining_s=1e9, deploy_s=900, point_s={"screen": 900, "decision": 1800}
)
COSTS = search.StepCosts(deploy_s=900, point_s={"screen": 900, "decision": 1800})


# --------------------------------------------------------------------------
# nvidia-smi output and method choice
# --------------------------------------------------------------------------


def test_bandwidth_mode_parsing_is_lenient_but_never_guesses():
    hopper = "GPU 0: NVIDIA H200 (UUID: GPU-1)\n    Bandwidth Mode: FULL\n"
    assert nvl.parse_bw_mode(hopper) == "FULL"
    assert nvl.parse_bw_mode("NVLink Bandwidth Mode: OFF") == "OFF"
    assert nvl.parse_bw_mode("3QUARTER") == "3QUARTER"
    assert (
        nvl.parse_bw_mode("NVLink Bandwidth Mode is DEPRECATED for Blackwell+.") is None
    )
    assert nvl.parse_bw_mode("") is None
    assert nvl.parse_bw_mode("GPU 0: OFF\nGPU 1: FULL") is None  # ambiguous
    with pytest.raises(ValueError):
        nvl.set_bw_mode("SOMETIMES")


def test_link_widths_ignore_gpu_indexes_and_bus_ids():
    text = "GPU 0: 00000000:18:00.0 Possible Link Widths: 18, 9, 6\n"
    assert nvl.parse_link_widths(text) == (6, 9, 18)  # "GPU 0" is not width 0
    assert nvl.parse_link_widths("Link Widths: 0, 6, 18") == (0, 6, 18)
    assert nvl.parse_link_widths("Not Supported") == ()


@pytest.mark.parametrize(
    "arch,mode,widths,expected",
    [
        ("sm90", "FULL", (), "bw_mode_off"),  # Hopper: the documented switch
        ("sm90", None, (), None),
        ("sm100", None, (0, 6, 18), "link_width_0"),  # Blackwell prefers width
        ("sm100", "FULL", (0, 18), "link_width_0"),
        ("sm103", None, (6, 18), None),  # no width 0: software only
        ("sm100", "FULL", (6, 18), "bw_mode_off"),  # deprecated but readable
    ],
)
def test_hardware_method(arch, mode, widths, expected):
    assert nvl.hardware_method(arch, mode, widths) == expected


def test_a_switch_counts_only_at_pcie_speed():
    assert nvl.switch_effective(41.5, 64.0)
    assert not nvl.switch_effective(380.0, 64.0)  # NVLink still in the path
    assert not nvl.switch_effective(None, 64.0)
    assert nvl.path_after_switch(56) == "pcie"
    assert nvl.path_after_switch(0) == "host"


def test_left_off_tracks_the_last_hardware_switch():
    off = {"action": "off", "method": "bw_mode_off", "path": "pcie"}
    assert nvl.left_off([]) is None
    assert nvl.left_off([off]) == off
    assert nvl.left_off([off, {"action": "restore", "ok": True}]) is None
    assert nvl.left_off([off, {"action": "restore", "ok": False}]) == off
    assert nvl.left_off([{"action": "off", "method": "nccl_no_p2p"}]) is None


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def _ctx(nvlink: bool = True) -> search.SearchContext:
    dep = profile_store.load_fitted_profile("glm-5.3", "b200")
    families = tuple(sorted(dep.families, key=lambda f: f.priority))
    grids = {f.id: GridSpec(start=64, max=512, min=8) for f in families}
    return search.SearchContext(
        families=families, ladder=dep.ladder, grids=grids, nvlink=nvlink
    )


_clock = [0.0]


def point(cand, c, tps, *, purpose="search", status="valid", stats=(), user=20.0):
    _clock[0] += 1000
    return PointResult(
        key=search.RunPoint(cand, c, "screen", purpose).key,
        candidate=cand,
        gpu_count=32,
        metrics=PointMetrics(
            user_tps_p50=user,
            ttft_p50_ms=900.0,
            itl_p50_ms=50.0,
            request_count=10 * c,
            output_tps=tps * 32,
        ),
        audit=AuditResult(status=status),
        tps_per_gpu=tps,
        bandwidth=tuple(stats),
        started_at=_clock[0],
        ended_at=_clock[0] + 900,
    )


AGG = Candidate(family_id="agg-dtp8")
DEP = Candidate(family_id="wide-ep-dep32")  # DeepEP: NVLink-only intranode


def test_the_leader_is_repeated_at_its_concurrency_with_nvlink_off():
    ctx = _ctx()
    lead = point(AGG, 256, 120.0)
    history = search.History(points=(point(AGG, 128, 100.0), lead))
    action = search.decide_nvlink(ctx, history, BIG, "pcie")
    assert isinstance(action, search.RunPoint)
    assert (action.concurrency, action.spec, action.purpose) == (
        256,
        "screen",
        "nvlink",
    )
    assert action.candidate == AGG.model_copy(update={"p2p": "pcie"})
    assert action.candidate.key.endswith("|p2p=pcie") and AGG.key.count("p2p") == 0
    off = point(action.candidate, 256, 80.0, purpose="nvlink")
    done = search.History(points=(*history.points, off))
    assert search.decide_nvlink(ctx, done, BIG, "pcie").reason == (
        "NVLink comparison measured"
    )
    # either path counts as measured (the plan does not know which one ran)
    assert search.nvlink_plan(ctx, done, BIG) == (None, "NVLink comparison measured")
    failed = search.History(
        points=history.points,
        deploys=(
            DeployResult(
                candidate=action.candidate, cand_fp=action.candidate.fp, ok=False
            ),
        ),
    )
    assert search.decide_nvlink(ctx, failed, BIG, "pcie").reason == (
        "NVLink-off deployment failed"
    )
    tiny = replace(BIG, remaining_s=1000.0)
    assert search.nvlink_plan(ctx, history, tiny)[1] == (
        "budget skipped the NVLink comparison"
    )


def test_a_deepep_leader_hands_the_comparison_to_the_best_family_without_it():
    ctx = _ctx()
    history = search.History(
        points=(point(DEP, 256, 150.0), point(AGG, 128, 90.0), point(AGG, 256, 110.0))
    )
    base, reason = search.nvlink_base(ctx, history)
    assert base is not None and base.candidate == AGG and base.key.concurrency == 256
    assert "needs NVLink (DeepEP" in reason and "(agg-dtp8)" in reason
    only = search.History(points=(point(DEP, 256, 150.0),))
    base, reason = search.nvlink_base(ctx, only)
    assert base is None and "no measured configuration runs without it" in reason


def test_nothing_to_compare_without_nvlink_or_a_leader():
    assert search.nvlink_base(_ctx(nvlink=False), search.History()) == (
        None,
        "NVLink comparison not applicable",
    )
    assert search.nvlink_base(_ctx(), search.History()) == (
        None,
        "no leader to compare",
    )


def test_reservation_counts_the_switch_and_is_soft():
    ctx = _ctx()
    history = search.History(points=(point(AGG, 256, 120.0),))
    plan = search.reserve_plan(ctx, history, "confirm", COSTS)
    assert plan["nvlink"] == nvl.TOGGLE_S + 900 + 900
    assert search.soft_reserve_after(ctx, history, "confirm", COSTS) == pytest.approx(
        plan["nvlink"] * 1.1
    )
    off = point(AGG.model_copy(update={"p2p": "host"}), 256, 70.0, purpose="nvlink")
    measured = search.History(points=(*history.points, off))
    assert search.reserve_plan(ctx, measured, "confirm", COSTS)["nvlink"] == 0.0
    assert "nvlink" not in search.reserve_plan(
        _ctx(nvlink=False), history, "confirm", COSTS
    ) or (
        search.reserve_plan(_ctx(nvlink=False), history, "confirm", COSTS)["nvlink"]
        == 0.0
    )


def test_a_competitive_climb_may_spend_the_nvlink_share_exploration_may_not():
    ctx = _ctx()
    family = next(f for f in ctx.families if f.id == "pd-4p2d")
    pd = Candidate(family_id=family.id)
    climbing = (point(pd, 64, 50.0), point(pd, 128, 70.0))  # +40%: keep climbing
    tight = search.Budget(
        remaining_s=2000.0,
        deploy_s=900,
        point_s={"screen": 900, "decision": 1800},
        reserve_s=1500.0,
        soft_reserve_s=1000.0,
    )
    history = search.History(points=climbing)
    # leading family: the next point may use the soft (NVLink) share
    act = search._decide_family(ctx, history, tight, family, climbing[-1])
    assert isinstance(act, search.RunPoint) and act.concurrency == 256
    # another family leads at equal load: the reservation holds
    rival = point(AGG, 128, 80.0)
    history = search.History(points=(*climbing, rival))
    act = search._decide_family(ctx, history, tight, family, rival)
    assert isinstance(act, search.CloseFamily)
    assert act.reason == "budget stopped the climb before saturation"


# --------------------------------------------------------------------------
# Rendering and auditing
# --------------------------------------------------------------------------


def _render(tmp_path, cand: Candidate):
    gpus = profile_store.load_gpu_profiles()
    return SglangEngine().render(
        cand,
        config=RunConfig(exp_id="exp-1", image="img:1", workdir=str(tmp_path)),
        inventory=synthetic_inventory("h200", nodes=4),
        gpu=gpus["h200"],
        model=profile_store.load_model_profile("glm-5.3"),
        deployment=profile_store.load_fitted_profile("glm-5.3", "h200", nodes=4),
        model_dir="/workspace/models/glm",
    )


def _workers(plan):
    for spec in plan.containers:
        if spec.role in ("control", "frontend"):
            continue
        path = next(p for p in spec.files if p.endswith(".yaml"))
        yield spec, yaml.safe_load(spec.files[path])


@pytest.mark.parametrize(
    "p2p,env",
    [
        ("pcie", {"NCCL_P2P_LEVEL": "SYS", "NCCL_NVLS_ENABLE": "0"}),
        ("host", {"NCCL_P2P_DISABLE": "1", "NCCL_NVLS_ENABLE": "0"}),
    ],
)
def test_nvlink_off_variant_drops_nvlink_only_fast_paths(tmp_path, p2p, env):
    base = Candidate(family_id="agg-tp8")
    plain = _render(tmp_path, base)
    off = _render(tmp_path, base.model_copy(update={"p2p": p2p}))
    assert off.cand_fp != plain.cand_fp
    for spec, cfg in _workers(off):
        assert {k: spec.env.get(k) for k in env} == env
        assert cfg["disable-custom-all-reduce"] is True
    for spec, cfg in _workers(plain):
        assert "NCCL_P2P_DISABLE" not in spec.env and "NCCL_P2P_LEVEL" not in spec.env
        assert "disable-custom-all-reduce" not in cfg


def _nvl(scope, p99, method="measured"):
    return BandwidthStat(
        plane="nvlink", scope=scope, p99_gbs=p99, peak_gbs=p99, method=method
    )


def test_nvlink_traffic_during_an_off_run_invalidates_it():
    metrics = PointMetrics(
        aiperf_version="0.14.0", request_count=600, output_tps=1000.0, duration_s=600
    )
    common = dict(
        spec=DEFAULT_BENCH_SPECS["screen"],
        concurrency=64,
        expected_aiperf_version="0.14.0",
        engagement=None,
        server_alive=True,
        telemetry_coverage=1.0,
        nic_restricted=False,
        nic_engaged=None,
    )
    leak = analysis.audit_point(metrics, nvlink_off=True, nvlink_p99_gbs=35.2, **common)
    assert leak.status == "invalid"
    assert "NVLink still carried 35.2 GB/s per GPU with NVLink off" in leak.reasons
    quiet = analysis.audit_point(metrics, nvlink_off=True, nvlink_p99_gbs=0.2, **common)
    assert quiet.status == "valid"
    blind = analysis.audit_point(
        metrics, nvlink_off=True, nvlink_p99_gbs=None, **common
    )
    assert blind.status == "valid_with_recovery"
    # with NVLink on, NVLink traffic is the normal case
    on = analysis.audit_point(metrics, nvlink_p99_gbs=300.0, **common)
    assert on.status == "valid"
    assert nvl.nvlink_p99([_nvl("node0/gpu1", 3.0), _nvl("node0", 90.0)]) == 3.0
    assert nvl.nvlink_p99([_nvl("node0/gpu1", 0.0, "unavailable")]) is None


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _pcie(scope, p99, nominal=64.0):
    return BandwidthStat(
        plane="pcie", scope=scope, p99_gbs=p99, peak_gbs=p99, nominal_gbs=nominal
    )


NOISE = NoiseResult(
    cand_key="x",
    concurrency=256,
    samples=(120.0, 121.0, 119.0),
    mean=120.0,
    stdev=1.0,
    cv=0.008,
    noise_floor=2.4,
    noise_floor_pct=2.0,
)


def test_compare_measures_the_drop_and_the_path():
    on = point(AGG, 256, 120.0, user=20.0)
    off_cand = AGG.model_copy(update={"p2p": "pcie"})
    off = point(
        off_cand,
        256,
        78.0,
        purpose="nvlink",
        user=13.0,
        stats=(
            _nvl("node0/gpu0", 0.0),
            _pcie("node0/gpu3", 48.0),
            _pcie("node0/gpu3/p2p", 60.0),
        ),
    )
    switch = {
        "method": "bw_mode_off",
        "path": "pcie",
        "p2p": {"median_gbs": 41.5, "peer_pairs": 56},
    }
    cmp = nvl.compare(
        on,
        off,
        NOISE,
        switch=switch,
        restore={"ok": True},
        reason="the throughput leader",
    )
    assert (cmp.status, cmp.delta_pct, cmp.within_noise) == ("measured", -35.0, False)
    assert (cmp.method, cmp.path, cmp.p2p_gbs, cmp.peer_pairs) == (
        "bw_mode_off",
        "pcie",
        41.5,
        56,
    )
    assert (cmp.nvlink_p99_gbs, cmp.pcie_p99_gbs, cmp.pcie_capacity_gbs) == (
        0.0,
        48.0,
        64.0,
    )
    assert cmp.restored is True and (cmp.on_user_tps, cmp.off_user_tps) == (20.0, 13.0)
    small = point(off_cand, 256, 119.0, purpose="nvlink")
    assert nvl.compare(
        on, small, NOISE, switch=switch, restore=None, reason=""
    ).within_noise
    assert (
        nvl.compare(on, small, NOISE, switch=switch, restore=None, reason="").restored
        is False
    )


def test_compare_flags_invalid_runs_skips_and_software_switches():
    on = point(AGG, 256, 120.0)
    bad = point(
        AGG.model_copy(update={"p2p": "host"}),
        256,
        60.0,
        purpose="nvlink",
        status="invalid",
    )
    software = {"method": "nccl_no_p2p", "path": "host"}
    cmp = nvl.compare(on, bad, NOISE, switch=software, restore=None, reason="r")
    assert cmp.status == "invalid" and cmp.delta_pct is None
    assert cmp.restored is None  # nothing to restore after a software switch
    skipped = nvl.compare(None, None, NOISE, switch={}, restore=None, reason="why")
    assert (skipped.status, skipped.reason, skipped.method) == ("skipped", "why", None)
    lone = point(AGG.model_copy(update={"p2p": "pcie"}), 256, 70.0, purpose="nvlink")
    orphan = nvl.compare(None, lone, NOISE, switch=software, restore=None, reason="")
    assert orphan.status == "measured" and orphan.delta_pct is None


# --------------------------------------------------------------------------
# Switching on the nodes (scripted nvidia-smi)
# --------------------------------------------------------------------------


@dataclass
class FakeNodes:
    """nvidia-smi on N nodes; ``fail`` maps a command fragment to failing nodes."""

    names: tuple[str, ...]
    hopper: bool = True
    widths: str = "Possible Link Widths: 18, 9, 6"
    fail: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mode: dict[str, str] = field(default_factory=dict)
    width: dict[str, int] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.names

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        self.calls.append((node, command))
        for fragment, nodes in self.fail.items():
            if fragment in command and node in nodes:
                return CommandResult(node, command, 3, "", "Insufficient Permissions")
        if "-gBwMode" in command:
            text = (
                f"GPU 0: X\n    Bandwidth Mode: {self.mode.get(node, 'FULL')}"
                if self.hopper
                else "NVLink Bandwidth Mode is DEPRECATED for Blackwell+."
            )
            return CommandResult(node, command, 0, text, "")
        if "-sBwMode" in command:
            self.mode[node] = command.split("-sBwMode ")[1].split()[0]
        elif "-sLWidth values" in command:
            return CommandResult(node, command, 0, f"GPU 0: {self.widths}", "")
        elif "-sLWidth" in command:
            self.width[node] = int(command.split("-sLWidth ")[1].split()[0])
        return CommandResult(node, command, 0, "", "")

    def fanout(self, nodes, fn):
        return {n: fn(n) for n in nodes}


@dataclass
class FakeCopy:
    """The GPU-GPU copy check: fast while NVLink is still in the path."""

    nodes: FakeNodes
    peer_pairs: int = 56

    def p2p_without_nvlink(self, node):
        off = self.nodes.mode.get(node) == "OFF" or self.nodes.width.get(node) == 0
        value = 41.5 if off else 380.0
        return [
            PeakResult(
                plane="pcie",
                test="p2p_copy",
                scope=node,
                value_gbs=value,
                tool="fake",
                details={"peer_pairs": self.peer_pairs},
            )
        ]


def _control(tmp_path, nodes: FakeNodes, arch="sm90", peer_pairs=56):
    gpu = GpuProfile(
        name="g",
        match=("g",),
        arch=arch,
        hbm_gb=141,
        nvlink=True,
        weights_variant="fp8",
        nominal_nvlink_gbs=450.0,
        nominal_pcie_gbs=64.0,
    )
    notes: list[str] = []
    ctx = SimpleNamespace(
        store=EventStore(tmp_path / "events.jsonl"),
        cluster=nodes,
        simulator=FakeCopy(nodes, peer_pairs),
        exp_dir=tmp_path,
        node_names=list(nodes.names),
        node0=nodes.names[0],
        require_hardware=lambda: SimpleNamespace(gpu=gpu),
        note=lambda text, **_: notes.append(text),
        notes=notes,
    )
    return NvlinkControl(ctx), ctx


NODES = ("node0", "node1")


def test_hopper_switch_is_proven_and_restored(tmp_path):
    nodes = FakeNodes(NODES)
    control, ctx = _control(tmp_path, nodes)
    switch = control.switch_off()
    assert (switch.method, switch.path) == ("bw_mode_off", "pcie")
    assert nodes.mode == {"node0": "OFF", "node1": "OFF"}
    assert not any("-sLWidth" in c for _, c in nodes.calls)  # Hopper: no width
    (off,) = nvlink_events(ctx)
    assert off["p2p"]["median_gbs"] == 41.5 and off["before"] == {
        "node0": "FULL",
        "node1": "FULL",
    }
    assert control.restore() is True
    assert nodes.mode == {"node0": "FULL", "node1": "FULL"}
    assert nvl.left_off(nvlink_events(ctx)) is None


def test_a_node_that_refuses_the_switch_is_rolled_back_to_software(tmp_path):
    nodes = FakeNodes(NODES, fail={"-sBwMode OFF": ("node1",)})
    control, ctx = _control(tmp_path, nodes)
    switch = control.switch_off()
    assert (switch.method, switch.path) == ("nccl_no_p2p", "host")
    assert nodes.mode["node0"] == "FULL"  # the node that did switch is restored
    actions = [(e["action"], e["method"]) for e in nvlink_events(ctx)]
    assert actions == [
        ("off", "bw_mode_off"),
        ("restore", "bw_mode_off"),
        ("off", "nccl_no_p2p"),
    ]
    note = nvlink_events(ctx)[-1]["notes"][0]
    assert note.startswith("nvidia-smi nvlink -sBwMode OFF failed on node1")


def test_a_switch_that_leaves_nvlink_in_the_path_is_undone(tmp_path):
    nodes = FakeNodes(NODES)
    control, ctx = _control(tmp_path, nodes)
    copy = ctx.simulator
    ctx.simulator = SimpleNamespace(
        p2p_without_nvlink=lambda node: [
            p.model_copy(update={"value_gbs": 380.0})
            for p in copy.p2p_without_nvlink(node)
        ]
    )
    switch = control.switch_off()
    assert switch.method == "nccl_no_p2p" and nodes.mode["node0"] == "FULL"
    assert "did not take NVLink out of the path (GPU-GPU copy 380.0 GB/s)" in (
        nvlink_events(ctx)[-1]["notes"][0]
    )


def test_blackwell_uses_link_width_zero_when_offered(tmp_path):
    nodes = FakeNodes(NODES, hopper=False, widths="Possible Link Widths: 0, 6, 18")
    control, ctx = _control(tmp_path, nodes, arch="sm100")
    switch = control.switch_off()
    assert (switch.method, switch.path) == ("link_width_0", "pcie")
    assert nodes.width == {"node0": 0, "node1": 0}
    assert control.restore() is True
    assert nodes.width == {"node0": 18, "node1": 18}  # back to the widest


def test_no_peer_access_over_pcie_means_host_staging(tmp_path):
    nodes = FakeNodes(NODES)
    control, ctx = _control(tmp_path, nodes, peer_pairs=0)
    switch = control.switch_off()
    assert (switch.method, switch.path) == ("bw_mode_off", "host")
    assert any(
        "NCCL stages through host memory" in n for n in nvlink_events(ctx)[0]["notes"]
    )


def test_a_crash_with_nvlink_off_is_repaired_on_resume(tmp_path):
    nodes = FakeNodes(NODES)
    control, ctx = _control(tmp_path, nodes)
    control.switch_off()  # ... and the runner dies before the restore
    resumed, _ = _control(tmp_path, nodes)
    resumed.ctx.store = ctx.store
    resumed.restore_left_off()
    assert nodes.mode == {"node0": "FULL", "node1": "FULL"}
    assert nvlink_events(ctx)[-1]["reason"] == "resumed with NVLink still off"
    # a restore that fails is reported and stays pending
    broken = FakeNodes(NODES, fail={"-sBwMode FULL": NODES})
    control, ctx = _control(tmp_path / "b", broken)
    control.switch_off()
    assert control.restore() is False
    assert nvl.left_off(nvlink_events(ctx)) is not None
    assert any("NVLink restore failed" in n for n in ctx.notes)
