# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVML GPM sampling in agents/telemetry_agent.py (the gpm kind and its notes)."""

from __future__ import annotations

import ctypes
import io
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.lium_cluster_bench.adapters.telemetry import parse_samples
from benchmarks.lium_cluster_bench.agents import telemetry_agent as agent
from benchmarks.lium_cluster_bench.domain.gpuactivity import GPM_FIELDS
from benchmarks.lium_cluster_bench.domain.loadstress import analyze_point_stress

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

# nvml.h / nvidia-ml-py 13.615.71 metric ids (NVLink TOTAL_RX = 60 < TX = 61).
METRIC_IDS = {
    "NVML_GPM_METRIC_GRAPHICS_UTIL": 1,
    "NVML_GPM_METRIC_SM_UTIL": 2,
    "NVML_GPM_METRIC_SM_OCCUPANCY": 3,
    "NVML_GPM_METRIC_ANY_TENSOR_UTIL": 5,
    "NVML_GPM_METRIC_DRAM_BW_UTIL": 10,
    "NVML_GPM_METRIC_PCIE_TX_PER_SEC": 20,
    "NVML_GPM_METRIC_PCIE_RX_PER_SEC": 21,
    "NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC": 60,
    "NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC": 61,
}
# Interval averages the fake reports per metric id (percent, MiB/s).
VALUES = {
    1: 99.5,
    2: 87.25,
    3: 41.5,
    5: 63.125,
    10: 55.0,
    20: 1024.0,
    21: 2048.0,
    60: 40960.0,
    61: 39936.0,
}
NVLINK_IDS = (60, 61)
SUCCESS, NOT_SUPPORTED, NO_PERMISSION, GPU_IS_LOST = 0, 3, 4, 15
T0 = 1793700000.0


class FakeNVMLError(Exception):
    """pynvml.NVMLError: carries the NVML return code as ``value``."""

    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.value = value


# The fake's samples carry the handle and a sequence number; the real
# c_nvmlGpmSample_t is an opaque pointer.
class FakeSample(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_int), ("seq", ctypes.c_int)]


class MetricInfo(ctypes.Structure):
    _fields_ = [
        ("shortName", ctypes.c_char_p),
        ("longName", ctypes.c_char_p),
        ("unit", ctypes.c_char_p),
    ]


class GpmMetric(ctypes.Structure):  # c_nvmlGpmMetric_t
    _fields_ = [
        ("metricId", ctypes.c_uint),
        ("nvmlReturn", ctypes.c_uint),
        ("value", ctypes.c_double),
        ("metricInfo", MetricInfo),
    ]


class GpmMetricsGet(ctypes.Structure):  # c_nvmlGpmMetricsGet_t (12.535.x: 98)
    _fields_ = [
        ("version", ctypes.c_uint),
        ("numMetrics", ctypes.c_uint),
        ("sample1", ctypes.POINTER(FakeSample)),
        ("sample2", ctypes.POINTER(FakeSample)),
        ("metrics", GpmMetric * 98),
    ]


class FakeGpmNvml:
    """pynvml stand-in: GPU0 supported, GPU1 not, GPU2 NO_PERMISSION on
    sampling, GPU3 (PCIe-only) refuses the NVLink metrics."""

    NVMLError = FakeNVMLError
    NVML_SUCCESS = SUCCESS
    NVML_ERROR_INVALID_ARGUMENT = 2
    NVML_ERROR_NOT_SUPPORTED = NOT_SUPPORTED
    NVML_ERROR_NO_PERMISSION = NO_PERMISSION
    NVML_ERROR_GPU_IS_LOST = GPU_IS_LOST
    NVML_ERROR_UNKNOWN = 999
    NVML_GPM_METRICS_GET_VERSION = 1
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1
    c_nvmlGpmMetricsGet_t = GpmMetricsGet

    def __init__(self, gpus: int = 4, **overrides: Any) -> None:
        for name, metric_id in METRIC_IDS.items():
            setattr(self, name, metric_id)
        self.gpus = gpus
        self.supported = {0: 1, 1: 0, 2: 1, 3: 1}
        self.denied_sample = {2}  # nvmlGpmSampleGet -> NO_PERMISSION
        self.no_nvlink = {3}
        self.refused: dict[int, int] = {}  # GPU -> per-metric error for all
        self.lost_after: dict[int, int] = {}  # GPU -> MetricsGet calls before loss
        self.events: list[str] = []
        self.pairs: list[tuple[int, int, int]] = []  # (gpu, seq1, seq2)
        self.calls = {"sample_get": 0, "metrics_get": 0}
        self.allocated: list[Any] = []
        self.freed: list[int] = []
        self._seq = 0
        for name, value in overrides.items():
            setattr(self, name, value)

    # -- device basics (for NvmlGpus / open_sources)
    def nvmlInit(self) -> None:
        self.events.append("init")

    def nvmlShutdown(self) -> None:
        self.events.append("shutdown")

    def nvmlDeviceGetCount(self) -> int:
        return self.gpus

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, handle: int) -> bytes:
        return b"NVIDIA H200"

    def nvmlDeviceGetPciInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(busId=b"00000000:%02X:00.0" % (0x18 + 0x11 * handle))

    def nvmlDeviceGetPcieThroughput(self, handle: int, counter: int) -> int:
        return 1000

    # -- GPM
    def nvmlGpmQueryDeviceSupport(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(version=1, isSupportedDevice=self.supported[handle])

    def nvmlGpmSampleAlloc(self) -> Any:
        sample = ctypes.pointer(FakeSample(-1, 0))
        self.allocated.append(sample)
        return sample

    def nvmlGpmSampleFree(self, sample: Any) -> None:
        self.freed.append(ctypes.addressof(sample.contents))
        self.events.append("free")

    def nvmlGpmSampleGet(self, handle: int, sample: Any) -> Any:
        self.calls["sample_get"] += 1
        if handle in self.denied_sample:
            raise FakeNVMLError(NO_PERMISSION)
        self._seq += 1
        sample.contents.handle = handle
        sample.contents.seq = self._seq
        return sample

    def nvmlGpmMetricsGet(self, request: GpmMetricsGet) -> GpmMetricsGet:
        self.calls["metrics_get"] += 1
        first, second = request.sample1.contents, request.sample2.contents
        gpu = second.handle
        if first.handle != gpu or request.version != 1:
            raise FakeNVMLError(self.NVML_ERROR_INVALID_ARGUMENT)
        left = self.lost_after.get(gpu)
        if left is not None:
            if left == 0:
                raise FakeNVMLError(GPU_IS_LOST)
            self.lost_after[gpu] = left - 1
        self.pairs.append((gpu, first.seq, second.seq))
        for slot in range(request.numMetrics):
            metric = request.metrics[slot]
            if gpu in self.refused:
                metric.nvmlReturn = self.refused[gpu]
            elif gpu in self.no_nvlink and metric.metricId in NVLINK_IDS:
                metric.nvmlReturn = NOT_SUPPORTED
            else:
                metric.nvmlReturn = SUCCESS
                metric.value = VALUES[metric.metricId]
        return request


class Clock:
    """A monotonic clock advanced by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def opened(nvml: FakeGpmNvml, clock: Clock) -> agent.NvmlGpm:
    gpm = agent.NvmlGpm(nvml, clock)
    gpm.open([0, 1, 2, 3][: nvml.gpus])
    return gpm


GPU0_ROW = {
    "gr_pct": 99.5,
    "sm_pct": 87.25,
    "occ_pct": 41.5,
    "tensor_pct": 63.12,  # 63.125 rounded to two decimals
    "dram_pct": 55.0,
    "pcie_tx_mibs": 1024.0,
    "pcie_rx_mibs": 2048.0,
    "nvlink_tx_mibs": 39936.0,  # TOTAL_TX is id 61
    "nvlink_rx_mibs": 40960.0,
}


# ----------------------------------------------------------------- probe


def test_gpm_metric_fields_are_the_domain_contract() -> None:
    assert tuple(key for key, _ in agent.GPM_METRICS) == GPM_FIELDS


def test_probe_records_availability_per_gpu() -> None:
    nvml = FakeGpmNvml()
    gpm = agent.NvmlGpm(nvml, Clock())
    status = gpm.open([0, 1, 2, 3])
    assert status == {
        0: "ok",
        1: "unsupported",
        2: "NVML_ERROR_NO_PERMISSION",
        3: "ok",
    }
    assert gpm.enabled == [0, 3] and gpm.metric_keys == list(GPM_FIELDS)
    # GPU1 allocated nothing; GPU2's two samples were freed at once.
    assert len(nvml.allocated) == 6 and len(nvml.freed) == 2
    gpm.close()
    assert sorted(nvml.freed) == sorted(
        ctypes.addressof(s.contents) for s in nvml.allocated
    )
    gpm.close()  # idempotent
    assert len(nvml.freed) == 6


def test_metric_ids_resolve_by_name_and_old_bindings_degrade() -> None:
    nvml = FakeGpmNvml()
    del nvml.NVML_GPM_METRIC_GRAPHICS_UTIL  # an older nvidia-ml-py
    nvml.NVML_GPM_METRIC_SM_OCCUPANCY = None  # not an id: skipped too
    assert [k for k, _ in agent.resolve_gpm_metrics(nvml)] == [
        k for k in GPM_FIELDS if k not in ("gr_pct", "occ_pct")
    ]
    assert dict(agent.resolve_gpm_metrics(nvml))["nvlink_rx_mibs"] == 60

    # nvidia-ml-py without the GPM API (or without any metric constant).
    legacy = SimpleNamespace(NVMLError=FakeNVMLError)
    assert agent.NvmlGpm(legacy, Clock()).open([0, 1]) == {
        0: "no-gpm-api",
        1: "no-gpm-api",
    }


def test_nvml_error_names() -> None:
    nvml = FakeGpmNvml()
    assert agent.nvml_return_name(nvml, 4) == "NVML_ERROR_NO_PERMISSION"
    assert agent.nvml_return_name(nvml, 77) == "NVML_ERROR_77"
    assert agent.nvml_error_name(nvml, FakeNVMLError(15)) == "NVML_ERROR_GPU_IS_LOST"
    assert agent.nvml_error_name(nvml, TypeError("bad field")) == "TypeError"


# ----------------------------------------------------------------- read


def test_read_costs_one_sample_and_one_metrics_call_per_gpu() -> None:
    nvml = FakeGpmNvml()
    clock = Clock()
    gpm = opened(nvml, clock)
    nvml.calls.update(sample_get=0, metrics_get=0)
    clock.now += 1.0
    values, notes = gpm.read()
    assert nvml.calls == {"sample_get": 2, "metrics_get": 2}
    assert values[0] == {"dt_s": 1.0, **GPU0_ROW}
    assert values[3] == {
        "dt_s": 1.0,
        **{k: v for k, v in GPU0_ROW.items() if not k.startswith("nvlink")},
    }
    # The PCIe-only GPU drops its NVLink metrics once and keeps sampling.
    assert notes == [
        {
            "source": "gpm",
            "gpu": 3,
            "error": "NVML_ERROR_NOT_SUPPORTED",
            "metrics": ["nvlink_tx_mibs", "nvlink_rx_mibs"],
        }
    ]
    clock.now += 1.5
    values, notes = gpm.read()
    assert notes == [] and values[3]["dt_s"] == 1.5
    assert "nvlink_tx_mibs" not in values[3]
    # Samples alternate: each interval starts where the previous one ended.
    assert [(g, a, b) for g, a, b in nvml.pairs if g == 0] == [(0, 1, 3), (0, 3, 5)]


def test_read_never_samples_faster_than_100_ms() -> None:
    nvml = FakeGpmNvml()
    clock = Clock()
    gpm = opened(nvml, clock)
    before = dict(nvml.calls)
    clock.now += 0.05
    assert gpm.read() == ({}, [])
    assert nvml.calls == before  # no NVML call at all
    clock.now += 0.06
    values, _ = gpm.read()
    assert values[0]["dt_s"] == pytest.approx(0.11)


def test_call_failure_disables_only_that_gpu() -> None:
    nvml = FakeGpmNvml(lost_after={3: 1})
    clock = Clock()
    gpm = opened(nvml, clock)
    clock.now += 1.0
    gpm.read()
    clock.now += 1.0
    values, notes = gpm.read()
    assert list(values) == [0]
    assert notes == [{"source": "gpm", "gpu": 3, "error": "NVML_ERROR_GPU_IS_LOST"}]
    assert gpm.enabled == [0] and gpm.status[3] == "NVML_ERROR_GPU_IS_LOST"
    assert len(nvml.freed) == 2 + 2  # GPU2 at the probe, GPU3 now
    clock.now += 1.0
    values, notes = gpm.read()
    assert list(values) == [0] and notes == []


def test_every_metric_refused_disables_the_gpu() -> None:
    nvml = FakeGpmNvml(refused={0: NO_PERMISSION})
    clock = Clock()
    gpm = opened(nvml, clock)
    clock.now += 1.0
    values, notes = gpm.read()
    assert 0 not in values and gpm.enabled == [3]
    assert notes[0] == {
        "source": "gpm",
        "gpu": 0,
        "error": "NVML_ERROR_NO_PERMISSION",
        "metrics": list(GPM_FIELDS),
    }
    assert notes[1] == {"source": "gpm", "gpu": 0, "error": "NVML_ERROR_NO_PERMISSION"}


def test_binding_mismatch_costs_gpm_not_the_sampler() -> None:
    nvml = FakeGpmNvml()

    class Broken:  # a struct whose metrics array is missing
        version = numMetrics = sample1 = sample2 = None

    nvml.c_nvmlGpmMetricsGet_t = Broken
    clock = Clock()
    gpm = opened(nvml, clock)
    clock.now += 1.0
    values, notes = gpm.read()
    assert values == {} and gpm.enabled == []
    assert {n["error"] for n in notes} == {"AttributeError"}


# ----------------------------------------------------------------- agent lines


class FakeTime:
    """Monotonic and wall clocks that only advance when the loop sleeps."""

    def __init__(self) -> None:
        self.mono = 100.0

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return T0 + (self.mono - 100.0)

    def sleep(self, seconds: float) -> None:
        self.mono += seconds


class IdleHost:
    def cpu_times(self) -> None:
        return None

    def net(self, iface: str) -> None:
        return None


def gpm_sources(nvml: FakeGpmNvml, clock: FakeTime) -> agent.Sources:
    gpm = agent.NvmlGpm(nvml, clock.monotonic)
    status = gpm.open([0, 1, 2, 3])
    return agent.Sources(
        host=IdleHost(),
        gpus=[agent.GpuDesc(i, None, "NVIDIA H200") for i in range(4)],
        nvml=True,
        gpu_stats=lambda: {},
        pcie=agent.PcieAccumulator(),
        pcie_source=None,
        nvlink=None,
        nvlink_source=None,
        ib_ports=[],
        gpm=gpm,
        gpm_status=status,
    )


def test_run_loop_writes_gpm_lines_meta_and_notes() -> None:
    nvml = FakeGpmNvml(lost_after={0: 2})
    clock = FakeTime()
    sources = gpm_sources(nvml, clock)
    cfg = agent.AgentConfig(out="unused", node="node2", duration=4.0)
    out = io.StringIO()
    ticks = agent.run_loop(
        cfg, sources, out, lambda: False, "h", clock.monotonic, clock.time, clock.sleep
    )
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    meta = records[0]
    assert meta["gpm"] == {
        "0": "ok",
        "1": "unsupported",
        "2": "NVML_ERROR_NO_PERMISSION",
        "3": "ok",
    }
    assert meta["gpm_metrics"] == list(GPM_FIELDS)
    gpm = [r for r in records if r["k"] == "gpm"]
    notes = [r for r in records if r["k"] == "note"]
    assert ticks == 4 and len(gpm) == 4  # three interval ticks + the final sample
    assert [set(r["g"]) for r in gpm] == [{"0", "3"}] * 2 + [{"3"}] * 2
    assert set(gpm[0]) == {"k", "t", "g"}
    assert gpm[0]["t"] == pytest.approx(T0 + 1.0, abs=1e-3)
    assert gpm[0]["g"]["0"] == {"dt_s": pytest.approx(1.0, abs=1e-3), **GPU0_ROW}
    assert [(n["gpu"], n["error"], "metrics" in n) for n in notes] == [
        (3, "NVML_ERROR_NOT_SUPPORTED", True),
        (0, "NVML_ERROR_GPU_IS_LOST", False),
    ]
    assert set(notes[1]) == {"k", "t", "source", "gpu", "error"}

    # The loader and the domain read what the agent writes.
    samples = parse_samples(out.getvalue())
    assert samples.gpm is not None and samples.stress is not None
    assert samples.gpm.status[2] == "NVML_ERROR_NO_PERMISSION"
    assert len(samples.gpm.series[3].t) == 4 and len(samples.gpm.series[0].t) == 2
    assert samples.gpm.failures[0][1] == "NVML_ERROR_GPU_IS_LOST"
    point = analyze_point_stress(
        {"node2": samples.stress},
        (T0 + 0.5, T0 + 3.9),
        gpm_by_node={"node2": samples.gpm},
    )
    activity = point.activity
    assert activity is not None
    assert activity.availability == {
        "node2:0": "NVML_ERROR_GPU_IS_LOST",
        "node2:1": "unsupported",
        "node2:2": "NVML_ERROR_NO_PERMISSION",
        "node2:3": "ok",
    }
    gpu3 = activity.gpu("node2:3")
    assert gpu3 is not None and gpu3.tensor_pct_mean == 63.1
    assert gpu3.nvlink_tx_gbs_mean is None and gpu3.regime == "compute"
    assert activity.gpu("node2:0").pcie_rx_gbs_mean == pytest.approx(2.1475, abs=1e-4)


def test_no_gpm_sources_write_no_gpm_lines() -> None:
    clock = FakeTime()
    sources = gpm_sources(FakeGpmNvml(), clock)
    sources.gpm, sources.gpm_status = None, {0: "disabled"}
    cfg = agent.AgentConfig(out="unused", node="n", duration=2.0)
    out = io.StringIO()
    agent.run_loop(
        cfg, sources, out, lambda: False, "h", clock.monotonic, clock.time, clock.sleep
    )
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert records[0]["gpm"] == {"0": "disabled"} and records[0]["gpm_metrics"] == []
    assert {r["k"] for r in records} == {"meta"}


# ----------------------------------------------------------------- open_sources


# A dmon binary that cannot start: open_sources must not run nvidia-smi.
NO_DMON = ["/nonexistent/nvidia-smi-dmon"]


def no_tools(argv, timeout=None):
    return None


def smi_listing(argv, timeout=None):
    if "--query-gpu=index,pci.bus_id,name" in argv:
        return agent.ToolResult(
            0, "0, 00000000:18:00.0, NVIDIA H200\n1, 00000000:29:00.0, NVIDIA H200\n"
        )
    return None


def test_open_sources_probes_gpm_and_frees_samples_before_shutdown(
    tmp_path: Path,
) -> None:
    nvml = FakeGpmNvml()
    cfg = agent.AgentConfig(out=str(tmp_path / "t.jsonl"), node="node0")
    with ExitStack() as stack:
        sources = agent.open_sources(
            cfg, stack, nvml_module=nvml, run=no_tools, root=str(tmp_path)
        )
        assert sources.gpm is not None and sources.gpm.enabled == [0, 3]
        assert sources.gpm_status[1] == "unsupported"
        meta = agent.meta_record(cfg, sources, "h", 0.0)
        assert meta["gpm"]["2"] == "NVML_ERROR_NO_PERMISSION"
    assert len(nvml.freed) == len(nvml.allocated) == 6
    assert nvml.events[-1] == "shutdown" and nvml.events.count("free") == 6


def test_open_sources_gpm_status_without_nvml(tmp_path: Path) -> None:
    cfg = agent.AgentConfig(out=str(tmp_path / "t.jsonl"), node="node0")
    with ExitStack() as stack:
        missing = agent.open_sources(
            cfg,
            stack,
            nvml_module=None,
            run=smi_listing,
            root=str(tmp_path),
            dmon_argv=NO_DMON,
        )
    assert missing.gpm is None
    assert missing.gpm_status == {0: "no-pynvml", 1: "no-pynvml"}

    class BrokenInit(FakeGpmNvml):
        NVML_ERROR_LIBRARY_NOT_FOUND = 12

        def nvmlInit(self) -> None:
            raise FakeNVMLError(12)

    with ExitStack() as stack:
        failed = agent.open_sources(
            cfg,
            stack,
            nvml_module=BrokenInit(),
            run=smi_listing,
            root=str(tmp_path),
            dmon_argv=NO_DMON,
        )
    assert failed.gpm_status == {
        0: "NVML_ERROR_LIBRARY_NOT_FOUND",
        1: "NVML_ERROR_LIBRARY_NOT_FOUND",
    }

    off = agent.parse_args(["--out", cfg.out, "--node", "node0", "--no-gpm"])
    assert off.gpm is False and agent.parse_args(["--out", "x", "--node", "n"]).gpm
    nvml = FakeGpmNvml()
    with ExitStack() as stack:
        disabled = agent.open_sources(
            off, stack, nvml_module=nvml, run=no_tools, root=str(tmp_path)
        )
    assert disabled.gpm is None and set(disabled.gpm_status.values()) == {"disabled"}
    assert nvml.allocated == [] and nvml.calls["sample_get"] == 0
