# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/gpuactivity.py and the gpm loader in adapters/telemetry.py."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.lium_cluster_bench.adapters.telemetry import load_samples, parse_samples
from benchmarks.lium_cluster_bench.domain import loadstress
from benchmarks.lium_cluster_bench.domain.bandwidth import Window
from benchmarks.lium_cluster_bench.domain.gpuactivity import (
    GpmSamples,
    GpmSeries,
    activity_by_role,
    analyze_gpu_activity,
    brief_activity,
    classify_activity,
    gpu_role_map,
    percentile,
    summarize_gpu,
)
from benchmarks.lium_cluster_bench.domain.loadstress import PointStress, analyze_point_stress
from benchmarks.lium_cluster_bench.domain.schema import ContainerSpec

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures"
GPM = FIXTURES / "gpuactivity"
T0 = 1793700000.0
# The AIPerf window sits just inside the agent's ticks (t = T0 + k + 0.0213).
WINDOW = (T0 + 2.03, T0 + 12.01)
ROLES = {
    "node0:0": "prefill",
    "node0:1": "prefill",
    "node0:2": "decode",
    "node0:3": "decode",
    "node1:0": "agg",
}


def series(dt=1.0, start=1, **columns) -> GpmSeries:
    """Samples at T0 + start, T0 + start + 1, ... each lasting ``dt``."""
    count = max(len(v) for v in columns.values())
    return GpmSeries(
        t=tuple(T0 + start + i for i in range(count)),
        dt_s=(dt,) * count,
        values={name: tuple(values) for name, values in columns.items()},
    )


def flat(n=10, **values) -> GpmSeries:
    return series(**{name: [value] * n for name, value in values.items()})


COMPUTE = dict(gr_pct=99.0, sm_pct=92.0, tensor_pct=60.0, dram_pct=30.0)
MEMORY = dict(gr_pct=98.0, sm_pct=80.0, tensor_pct=15.0, dram_pct=65.0)


# ----------------------------------------------------------------- statistics


def test_summarize_gpu_overlap_weighted_mean_p90_max_and_rates() -> None:
    gpu = summarize_gpu(
        "node0:0",
        series(
            sm_pct=[10, 20, 30, 40, 50],
            pcie_tx_mibs=[1000.0] * 5,
            nvlink_rx_mibs=[None, 2000.0, 2000.0, None, None],
        ),
        (T0 + 0.5, T0 + 4.5),
    )
    # (T0, T0+1] and (T0+4, T0+5] each overlap the window by half a second.
    assert (gpu.samples, gpu.covered_s) == (5, 4.0)
    assert gpu.sm_pct_mean == 30.0  # (5 + 20 + 30 + 40 + 25) / 4
    assert (gpu.sm_pct_p90, gpu.sm_pct_max) == (46.0, 50.0)
    assert gpu.pcie_tx_gbs_mean == pytest.approx(1.0486, abs=1e-4)  # MiB/s -> GB/s
    assert gpu.nvlink_rx_gbs_mean == pytest.approx(2.0972, abs=1e-4)
    assert gpu.tensor_pct_mean is None and gpu.regime == "unknown"
    assert percentile([1.0, 2.0, 3.0, 4.0], 90) == pytest.approx(3.7)
    with pytest.raises(ValueError):
        summarize_gpu("node0:0", flat(sm_pct=1.0), (T0 + 5, T0 + 5))


def test_unknown_interval_length_falls_back_to_the_agent_tick() -> None:
    samples = GpmSeries(
        t=(T0 + 2.0, T0 + 4.0),
        dt_s=(None, None),
        values={"tensor_pct": (50.0, 70.0), "dram_pct": (10.0, 10.0)},
    )
    two_s = summarize_gpu("n:0", samples, (T0, T0 + 4.0), default_dt_s=2.0)
    one_s = summarize_gpu("n:0", samples, (T0, T0 + 4.0))
    assert (two_s.covered_s, one_s.covered_s) == (4.0, 2.0)
    assert two_s.tensor_pct_mean == one_s.tensor_pct_mean == 60.0


@pytest.mark.parametrize(
    ("gr", "sm", "tensor", "dram", "regime", "reason"),
    [
        (99.0, 92.0, 58.0, 31.0, "compute", "tensor pipes active 58% (DRAM 31%)"),
        (98.0, 81.0, 17.0, 63.0, "memory", "DRAM bandwidth 63% used (tensor 17%)"),
        (99.0, 90.0, 45.0, 60.0, "memory", "DRAM bandwidth 60% used (tensor 45%)"),
        (
            62.0,
            40.0,
            10.0,
            12.0,
            "waiting",
            "no kernel running 38% of the time (host or scheduler gaps)",
        ),
        (
            97.0,
            30.0,
            3.0,
            9.0,
            "waiting",
            "kernels active but 30% of SMs busy (communication or small kernels)",
        ),
        (
            97.0,
            70.0,
            25.0,
            35.0,
            "waiting",
            "tensor 25% and DRAM 35% below saturation "
            "(latency or synchronization bound)",
        ),
        (2.0, 1.0, 0.0, 0.5, "idle", "GPU idle (engine 2%, SMs 1%)"),
        (None, 3.0, None, None, "idle", "GPU idle (engine n/a, SMs 3%)"),
        (95.0, 85.0, None, None, "unknown", "tensor and DRAM activity unavailable"),
    ],
)
def test_classify_activity(gr, sm, tensor, dram, regime, reason) -> None:
    assert classify_activity(gr, sm, tensor, dram) == (regime, reason)


# ----------------------------------------------------------------- analysis


def test_availability_covers_probe_status_failures_and_missing_samples() -> None:
    node0 = GpmSamples(
        node="node0",
        interval_s=1.0,
        status={0: "ok", 1: "ok", 2: "ok", 3: "ok", 4: "ok"},
        series={
            0: flat(**COMPUTE),  # T0+1 .. T0+10
            1: flat(n=3, **COMPUTE),  # ends at T0+3, then failed
            2: flat(**MEMORY),  # failed long after the window
            4: series(start=20, **{k: [v] for k, v in MEMORY.items()}),
        },
        failures={1: (T0 + 4.0, "NVML_ERROR_UNKNOWN"), 2: (T0 + 30.0, "X")},
        notes=("gpu3: nvlink_tx_mibs unavailable (NVML_ERROR_NOT_SUPPORTED)",),
    )
    node1 = GpmSamples(
        node="node1",
        status={0: "NVML_ERROR_NO_PERMISSION", 1: "unsupported", 2: "no-pynvml"},
    )
    summary = analyze_gpu_activity(
        {"node1": node1, "node0": node0, "node2": None}, Window(T0 + 1.0, T0 + 9.0)
    )
    assert summary.availability == {
        "node0:0": "ok",
        "node0:1": "NVML_ERROR_UNKNOWN",
        "node0:2": "ok",
        "node0:3": "no-samples",
        "node0:4": "no-samples",
        "node1:0": "NVML_ERROR_NO_PERMISSION",
        "node1:1": "unsupported",
        "node1:2": "no-pynvml",
    }
    assert summary.gpus_expected == 8 and summary.gpus_sampled == 3
    assert [g.gpu for g in summary.gpus] == ["node0:0", "node0:1", "node0:2"]
    assert summary.gpu("node0:1").covered_s == 2.0
    assert summary.gpu("node1:0") is None
    assert summary.notes == (
        "node0: gpu3: nvlink_tx_mibs unavailable (NVML_ERROR_NOT_SUPPORTED)",
        "node0:1: GPM failed (NVML_ERROR_UNKNOWN)",
        "node2: no GPM telemetry (agent without gpm records)",
        "GPM intervals cover < 80% of the window on 1 GPU(s): node0:1",
        "GPM unavailable: no-samples on 2 GPU(s), NVML_ERROR_NO_PERMISSION on 1 "
        "GPU(s), NVML_ERROR_UNKNOWN on 1 GPU(s), no-pynvml on 1 GPU(s), "
        "unsupported on 1 GPU(s)",
    )


def test_fleet_regime_is_the_dominant_gpu_regime_or_mixed() -> None:
    def fleet(*rows) -> GpmSamples:
        return GpmSamples(
            node="node0",
            status={i: "ok" for i in range(len(rows))},
            series={i: flat(**row) for i, row in enumerate(rows)},
        )

    dominant = analyze_gpu_activity(
        {"node0": fleet(COMPUTE, COMPUTE, COMPUTE, MEMORY)}, (T0, T0 + 10)
    )
    assert dominant.regimes == {"compute": 3, "memory": 1}
    assert (dominant.regime, dominant.reason) == (
        "compute",
        "tensor pipes active 60% (DRAM 30%) on 3 of 4 GPU(s)",
    )
    assert dominant.tensor_pct_mean == 48.8 and dominant.tensor_pct_max == 60.0
    mixed = analyze_gpu_activity(
        {"node0": fleet(COMPUTE, COMPUTE, MEMORY, MEMORY)}, (T0, T0 + 10)
    )
    assert (mixed.regime, mixed.reason) == (
        "mixed",
        "compute on 2 GPU(s), memory on 2 GPU(s)",
    )
    empty = analyze_gpu_activity({"node0": GpmSamples(node="node0")}, (T0, T0 + 1))
    assert (empty.regime, empty.reason, empty.gpus_sampled) == (
        "unknown",
        "no GPM samples in the window",
        0,
    )


def test_roles_from_the_launch_plan_containers() -> None:
    containers = [
        SimpleNamespace(node="node0", role="frontend", gpus=()),
        SimpleNamespace(node="node0", role="prefill", gpus=(0, 1)),
        ContainerSpec(
            node="node1",
            name="decode-0-r0",
            role="decode",
            gpus=(0, 1, 2, 3),
            image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1",
            command=("python3", "-m", "dynamo.sglang"),
        ),
    ]
    roles = gpu_role_map(containers)
    assert roles == {
        "node0:0": "prefill",
        "node0:1": "prefill",
        "node1:0": "decode",
        "node1:1": "decode",
        "node1:2": "decode",
        "node1:3": "decode",
    }
    samples = {
        "node0": GpmSamples(
            node="node0",
            status={0: "ok", 1: "ok", 2: "ok"},
            series={0: flat(**COMPUTE), 1: flat(**COMPUTE), 2: flat(**MEMORY)},
        ),
        "node1": GpmSamples(node="node1", status={0: "ok"}, series={0: flat(**MEMORY)}),
    }
    summary = analyze_gpu_activity(samples, (T0, T0 + 10), gpu_roles=roles)
    assert sorted(summary.roles) == ["decode", "prefill", "unassigned"]
    assert summary.gpu("node0:2").role is None
    prefill = summary.roles["prefill"]
    assert (prefill.gpus_sampled, prefill.regime, prefill.tensor_pct_mean) == (
        2,
        "compute",
        60.0,
    )
    assert summary.roles["decode"].regime == "memory"
    assert summary.roles["unassigned"].gpus_sampled == 1
    assert activity_by_role(summary, roles) == summary.roles
    assert activity_by_role(summary, {}) == {}


# ----------------------------------------------------------------- recorded files


@pytest.fixture(scope="module")
def recorded():
    return {
        "node0": load_samples(GPM / "telemetry-gpm-node0.jsonl"),
        "node1": load_samples(GPM / "telemetry-gpm-node1.jsonl"),
    }


def test_loader_builds_gpm_samples_from_recorded_files(recorded) -> None:
    gpm = recorded["node0"].gpm
    assert gpm is not None and gpm.node == "node0" and gpm.interval_s == 1.0
    assert gpm.status == {0: "ok", 1: "ok", 2: "ok", 3: "ok"}
    assert gpm.metrics[:2] == ("gr_pct", "sm_pct") and len(gpm.metrics) == 9
    # 14 ticks; the torn final gpm line is dropped; GPU3 failed at tick 12.
    assert [len(gpm.series[g].t) for g in range(4)] == [14, 14, 14, 11]
    assert gpm.series[0].dt_s[0] == 1.0063
    assert gpm.series[0].values["sm_pct"][6] == 98.0  # tick 7 spike
    assert gpm.failures == {
        3: (pytest.approx(T0 + 12.0213, abs=1e-6), "NVML_ERROR_UNKNOWN")
    }
    # The other kinds of the same file are unaffected.
    assert len(recorded["node0"].gpu[3].t) == 14
    assert len(recorded["node0"].ib[("mlx5_0", 1)].t) == 14
    rtx = recorded["node1"].gpm
    assert rtx is not None and rtx.series == {} and rtx.metrics == ()
    assert set(rtx.status.values()) == {"unsupported"}


def test_point_activity_from_recorded_files(recorded) -> None:
    point = analyze_point_stress(
        {n: s.stress for n, s in recorded.items()},
        WINDOW,
        gpm_by_node={n: s.gpm for n, s in recorded.items()},
        gpu_roles=ROLES,
    )
    activity = point.activity
    assert activity is not None
    gpu0 = activity.gpu("node0:0")
    assert (gpu0.samples, gpu0.covered_s, gpu0.role) == (10, 9.98, "prefill")
    # Nine 92 % intervals and a 98 % spike, boundary intervals weighted ~0.99.
    assert (gpu0.sm_pct_mean, gpu0.sm_pct_p90, gpu0.sm_pct_max) == (92.6, 92.6, 98.0)
    assert (gpu0.tensor_pct_mean, gpu0.tensor_pct_max) == (58.8, 66.0)
    assert gpu0.nvlink_tx_gbs_mean == 43.2538  # 41250 MiB/s
    assert gpu0.pcie_rx_gbs_mean == 1.7302 and gpu0.regime == "compute"
    gpu2 = activity.gpu("node0:2")
    assert (gpu2.dram_pct_mean, gpu2.dram_pct_max, gpu2.regime) == (
        63.8,
        71.0,
        "memory",
    )
    gpu3 = activity.gpu("node0:3")
    assert (gpu3.samples, gpu3.covered_s) == (9, 8.991)
    assert activity.availability["node0:3"] == "NVML_ERROR_UNKNOWN"
    assert activity.availability["node1:2"] == "unsupported"
    assert (activity.gpus_expected, activity.gpus_sampled) == (8, 4)
    assert (activity.sm_pct_mean, activity.sm_pct_max) == (85.9, 92.6)
    assert (activity.tensor_pct_mean, activity.dram_pct_mean) == (36.8, 46.2)
    assert activity.regime == "mixed"
    assert activity.regimes == {"compute": 2, "memory": 2}
    prefill, decode = activity.roles["prefill"], activity.roles["decode"]
    assert (prefill.regime, prefill.reason) == (
        "compute",
        "tensor pipes active 57% (DRAM 30%)",
    )
    assert (decode.regime, decode.reason) == (
        "memory",
        "DRAM bandwidth 62% used (tensor 17%)",
    )
    assert decode.nvlink_tx_gbs_mean == pytest.approx(5.332, abs=1e-4)
    assert "agg" not in activity.roles  # node1 has no samples
    assert "node0:3: GPM failed (NVML_ERROR_UNKNOWN)" in activity.notes
    # The rest of the stress analysis is untouched by GPM.
    assert point.nodes == ("node0", "node1") and point.gpus == ()

    brief = brief_activity(activity)
    assert brief.availability == {"NVML_ERROR_UNKNOWN": 1, "ok": 3, "unsupported": 4}
    assert (brief.regime, brief.sm_pct_mean, brief.gpus_expected) == ("mixed", 85.9, 8)
    assert brief.roles == activity.roles
    assert len(brief.model_dump_json()) < 2500  # compact enough for the summary


def test_point_stress_with_activity_round_trips_as_json(recorded) -> None:
    point = analyze_point_stress(
        {"node0": recorded["node0"].stress},
        WINDOW,
        gpm_by_node={"node0": recorded["node0"].gpm},
    )
    restored = PointStress.model_validate_json(point.model_dump_json(indent=1))
    assert restored == point and restored.activity.roles == {}
    # stress.json written before GPM existed still validates.
    legacy = point.model_dump(exclude={"activity"})
    assert PointStress.model_validate(legacy).activity is None


def test_old_files_have_no_gpm_and_no_activity() -> None:
    old = load_samples(FIXTURES / "telemetry" / "telemetry-node0.jsonl")
    stress = load_samples(FIXTURES / "telemetry" / "telemetry-stress-node1.jsonl")
    assert old.gpm is None and stress.gpm is None
    point = analyze_point_stress(
        {"node0": old.stress, "node1": stress.stress},
        (T0 + 2.0, T0 + 12.0),
        gpm_by_node={"node0": old.gpm, "node1": stress.gpm},
    )
    assert point.activity is None
    assert analyze_point_stress({"node1": stress.stress}, WINDOW).activity is None


def test_malformed_gpm_records_never_fail_the_file() -> None:
    lines = [
        {"k": "meta", "t": 1.0, "node": "n", "gpm": {"0": "ok", "x": "ok"}},
        {"k": "gpm", "t": 2.0, "g": {"0": {"dt_s": 1.0, "sm_pct": 50.0}}},
        {"k": "gpm", "t": "soon", "g": {"0": {"sm_pct": 1.0}}},
        {"k": "gpm", "t": 3.0, "g": [1, 2]},
        {"k": "gpm", "t": 3.0, "g": {"gpu1": {}, "1": 7, "0": {"dt_s": -1, "x": 1}}},
        {"k": "gpm", "t": 4.0, "g": {"0": {"sm_pct": float("inf"), "dram_pct": 9}}},
        {"k": "note", "t": 4.0, "source": "gpm", "error": "NVML_ERROR_UNKNOWN"},
        {"k": "note", "t": 5.0, "source": "gpm", "gpu": 0, "error": "E", "metrics": []},
        {"k": "note", "t": 5.0, "source": "ethtool", "iface": "eth2", "error": "x"},
        {"k": "net", "t": 5.0, "iface": "wg0", "tx": 1, "rx": 2},
    ]
    samples = parse_samples("\n".join(json.dumps(r) for r in lines))
    gpm = samples.gpm
    assert gpm is not None and gpm.status == {0: "ok"} and gpm.metrics == ()
    assert gpm.series[0].t == (2.0, 3.0, 4.0)
    assert gpm.series[0].dt_s == (1.0, None, None)
    assert gpm.series[0].values == {
        "sm_pct": (50.0, None, None),
        "dram_pct": (None, None, 9.0),
    }
    # A gpm note with an empty metrics list is a GPU failure.
    assert gpm.failures == {0: (5.0, "E")}
    assert gpm.notes == ("5 malformed gpm record(s) skipped",)
    assert samples.stress is not None
    assert samples.stress.notes == ("ethtool: x (eth2)",)
    assert samples.net["wg0"].tx == (1,)


def test_activity_failure_costs_only_activity(
    monkeypatch: pytest.MonkeyPatch, recorded
) -> None:
    def broken(*args, **kwargs):
        raise ArithmeticError("boom")

    monkeypatch.setattr(loadstress, "analyze_gpu_activity", broken)
    point = analyze_point_stress(
        {"node0": recorded["node0"].stress},
        WINDOW,
        gpm_by_node={"node0": recorded["node0"].gpm},
    )
    assert point.activity is None
    assert "GPU activity analysis failed: boom" in point.notes
    assert point.nodes == ("node0",)
