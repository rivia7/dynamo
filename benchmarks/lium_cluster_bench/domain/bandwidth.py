# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bandwidth attribution from under-load telemetry. Pure: no I/O.

Inputs are the per-node samples written by ``agents/telemetry_agent.py`` (and
loaded by ``adapters/telemetry.py``); the output is one ``PointBandwidth``
per benchmark point: ``BandwidthStat`` rows plus the NIC verdicts that
``PointResult`` carries.

Conventions
-----------
* Rates are bytes/s internally and GB/s (1 GB = 1e9 bytes) in every output.
* A series becomes piecewise-constant rate intervals clipped to the benchmark
  window, then 1-s bins aligned to the window start. ``mean_gbs`` is the
  time-weighted mean over bins with data; ``p99_gbs`` and ``peak_gbs`` are the
  99th percentile and the maximum of the 1-s bin rates. Sums over devices add
  bin rates, so an aggregate peak is a simultaneous peak.
* Cumulative IB counters are 64-bit when any raw value (bytes / 4) reaches
  2**32, else 32-bit. A negative delta on a 32-bit counter is one wrap
  (+2**32 * 4 bytes); on a 64-bit counter it is a reset and that interval is
  dropped. NVLink and network counters are always 64-bit.

Stats emitted (plane: scope)
----------------------------
* ib: "<node>/<hca>" per HCA with counters (selected or not), "<node>" and
  "cluster" aggregates; tx and rx; nominal = active port rate / 8.
* pcie: "<node>/gpu<i>" tx and rx, the GPU's whole PCIe traffic.
* nvlink: "<node>/gpu<i>" tx and rx, or a single "<node>" "bidir" row with
  method "unavailable" when the node has no NVLink counters.
* pcie_nic: "<node>/gpu<i>" tx and rx, method "derived": per 1-s bin,
  min(GPU PCIe, bytes of the paired HCA / GPUs sharing that HCA).
* PCIe-only GPU profiles add pcie "<node>/gpu<i>/p2p" tx and rx, method
  "derived": GPU PCIe minus its pcie_nic share, i.e. GPU-GPU P2P (an upper
  bound: it includes host<->GPU copies).
* A node without telemetry gets "<node>" "bidir" rows, method "unavailable".

Reference peaks (``ref_peak_gbs``)
----------------------------------
Only ``PeakResult`` rows of the stat's plane match, minus rows whose own
engagement check failed (``details["engaged"] is False``); for IB, rows whose
``nic_count`` equals the node's selected NIC count are preferred when any
exist. A device stat ("<node>/<device>") uses a row with the identical scope
as-is. Every other case takes the best per-device value and multiplies it by
the number of devices the stat covers (selected NICs for IB node and cluster
stats), preferring rows whose scope names the stat's node ("node0",
"node0/mlx5_0", "node0<->node1", "node0+node1") and, for IB, GPU-direct
("gdr") rows over collectives, whose bus bandwidth is not a per-NIC rate.
Per-device values come from ``details["per_nic_gbs" | "per_gpu_gbs" |
"per_device_gbs"]`` when it is a number; a device-scoped row is already per
device; any other IB row is a per-node figure (``details["per_node_gbs"]``
when present, else ``value_gbs``) over ``nic_count`` NICs (all active NICs
when unset); a GPU-plane test named "*aggregate*" is an aggregate over the
node's GPUs; anything else is per device. The highest candidate wins. The
derived ``pcie_nic`` plane falls back to the per-NIC GPU-direct IB reference
divided by the GPUs sharing that NIC.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from benchmarks.lium_cluster_bench.domain import nicplan
from benchmarks.lium_cluster_bench.domain.gpuactivity import GpmSamples
from benchmarks.lium_cluster_bench.domain.loadstress import StressSamples
from benchmarks.lium_cluster_bench.domain.schema import (
    CROSS_NODE_MIN_GBS,
    BandwidthStat,
    GpuProfile,
    Inventory,
    Method,
    NodeInventory,
    PeakResult,
    Plane,
)

GB = 1e9
COUNTER32_SPAN = 1 << 32
# Below this mean IB rate per node (GB/s) a point has no cross-node traffic.
# NIC engagement: HCAs outside the selection must carry less than this share
# of a node's IB bytes, and every selected HCA more than this share ("carries
# traffic", so subnet-management packets on an idle port do not count).
OUTSIDE_MAX_SHARE = 0.01
SELECTED_MIN_SHARE = 0.01
# Host-staging heuristic: the best selected HCA peaks below this fraction of
# the per-NIC GPU-direct reference.
STAGING_PEAK_FRACTION = 0.2
LOW_COVERAGE = 0.9
# Mean control-interface traffic (GB/s) that suggests bulk data on wg0.
CONTROL_NET_WARN_GBS = 0.05
DETAIL_PER_DEVICE_KEYS = ("per_nic_gbs", "per_gpu_gbs", "per_device_gbs")
_SCOPE_SPLIT_RE = re.compile(r"<->|->|/|,|\+|\s+")


# --------------------------------------------------------------------------
# Telemetry samples (built by adapters/telemetry.load_samples)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryGpu:
    index: int
    pci_bus_id: str | None
    name: str


@dataclass(frozen=True)
class TelemetryMeta:
    node: str
    hostname: str
    t: float
    ib_unit_bytes: int = 4
    nvml: bool = False
    nvlink_source: str | None = None
    gpus: tuple[TelemetryGpu, ...] = ()
    hcas: tuple[tuple[str, int], ...] = ()  # active (device, port)
    interval_s: float = 1.0
    pcie_interval_s: float = 0.1
    pcie_source: str | None = None
    control_interface: str | None = None
    tag: str | None = None


@dataclass(frozen=True)
class CounterSeries:
    """Cumulative byte counters sampled at ``t`` (unix seconds)."""

    t: tuple[float, ...]
    tx: tuple[int, ...]
    rx: tuple[int, ...]

    def points(self, direction: str) -> list[tuple[float, int]]:
        values = self.tx if direction == "tx" else self.rx
        return list(zip(self.t, values))


@dataclass(frozen=True)
class PcieSeries:
    """Per-interval PCIe aggregates; each row covers (t - interval, t]."""

    t: tuple[float, ...]
    tx_mean: tuple[float, ...]
    tx_max: tuple[float, ...]
    rx_mean: tuple[float, ...]
    rx_max: tuple[float, ...]
    n: tuple[int, ...]


@dataclass(frozen=True)
class GpuSeries:
    t: tuple[float, ...]
    util: tuple[float | None, ...]
    mem_used_mib: tuple[float | None, ...]
    power_w: tuple[float | None, ...]


@dataclass(frozen=True)
class ScalarSeries:
    t: tuple[float, ...] = ()
    v: tuple[float, ...] = ()


@dataclass(frozen=True)
class TelemetrySamples:
    """One node's telemetry file, parsed."""

    meta: TelemetryMeta
    ib: Mapping[tuple[str, int], CounterSeries] = field(default_factory=dict)
    pcie: Mapping[int, PcieSeries] = field(default_factory=dict)
    nvlink: Mapping[int, CounterSeries] = field(default_factory=dict)
    gpu: Mapping[int, GpuSeries] = field(default_factory=dict)
    cpu: ScalarSeries = ScalarSeries()
    net: Mapping[str, CounterSeries] = field(default_factory=dict)
    stress: StressSamples | None = None  # ibx/gpux/ethx records (newer agents)
    gpm: GpmSamples | None = None  # NVML GPM records (GPM-capable agents)


# --------------------------------------------------------------------------
# Rates and summaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A closed time range in unix seconds (the AIPerf profiling window)."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.end_s > self.start_s:
            raise ValueError(f"empty window [{self.start_s}, {self.end_s}]")

    @classmethod
    def from_ns(cls, start_ns: int, end_ns: int) -> Window:
        return cls(start_ns / 1e9, end_ns / 1e9)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s


def as_window(window: Window | Sequence[float]) -> Window:
    """Accept a ``Window`` or a ``(start_s, end_s)`` pair."""
    if isinstance(window, Window):
        return window
    start, end = window
    return Window(float(start), float(end))


@dataclass(frozen=True)
class RateInterval:
    t0: float
    t1: float
    bytes_per_s: float


@dataclass(frozen=True)
class CounterRates:
    intervals: tuple[RateInterval, ...]
    bits: int  # 32 or 64
    wraps: int = 0  # 32-bit wraps corrected inside the window
    resets: int = 0  # intervals dropped as counter resets
    saturated: bool = False  # a 32-bit counter pinned at 2**32 - 1


@dataclass(frozen=True)
class RateSummary:
    mean_gbs: float
    p99_gbs: float
    peak_gbs: float
    covered_s: float  # seconds of the window with data

    @property
    def available(self) -> bool:
        return self.covered_s > 0


EMPTY_SUMMARY = RateSummary(0.0, 0.0, 0.0, 0.0)
# One rate (bytes/s) per bin edge pair; None where the series has no data.
Bins = list[float | None]


def _clip(t0: float, t1: float, window: Window | None) -> tuple[float, float]:
    if window is None:
        return t0, t1
    return max(t0, window.start_s), min(t1, window.end_s)


def counter_rates(
    series: Iterable[tuple[float, int]],
    window: Window | Sequence[float] | None = None,
    *,
    unit_bytes: int = 4,
    counter_bits: int | None = None,
    scale: float = 1.0,
    max_rate_bytes_per_s: float | None = None,
) -> CounterRates:
    """Per-interval byte rates from cumulative byte counters.

    ``series`` holds (t, cumulative bytes) where bytes = raw counter x
    ``unit_bytes``. Width is detected from the raw values unless
    ``counter_bits`` is given. A corrected 32-bit wrap implying more than
    ``max_rate_bytes_per_s`` is treated as a reset instead. Intervals are
    clipped to ``window``; partially covered ones keep their rate.
    """
    win = None if window is None else as_window(window)
    points = sorted(series)
    if counter_bits is None:
        wide = any(v // unit_bytes >= COUNTER32_SPAN for _, v in points)
        bits = 64 if wide else 32
    else:
        bits = counter_bits
    saturated = bits == 32 and any(
        v // unit_bytes == COUNTER32_SPAN - 1 for _, v in points
    )
    span_bytes = COUNTER32_SPAN * unit_bytes
    intervals: list[RateInterval] = []
    wraps = resets = 0
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        dt = t1 - t0
        lo, hi = _clip(t0, t1, win)
        if dt <= 0 or hi <= lo:
            continue
        delta = v1 - v0
        if delta < 0:
            if bits != 32:
                resets += 1
                continue
            delta += span_bytes
            if max_rate_bytes_per_s is not None and delta / dt > max_rate_bytes_per_s:
                resets += 1
                continue
            wraps += 1
        intervals.append(RateInterval(lo, hi, delta / dt * scale))
    return CounterRates(tuple(intervals), bits, wraps, resets, saturated)


def pcie_intervals(
    series: PcieSeries,
    direction: str,
    interval_s: float,
    window: Window | None = None,
) -> list[RateInterval]:
    """Rate intervals of one PCIe direction (mean of the sub-samples)."""
    values = series.tx_mean if direction == "tx" else series.rx_mean
    out: list[RateInterval] = []
    prev: float | None = None
    for t, value in zip(series.t, values):
        start = t - interval_s if prev is None else max(prev, t - interval_s)
        prev = t
        lo, hi = _clip(start, t, window)
        if hi > lo and value is not None:
            out.append(RateInterval(lo, hi, float(value)))
    return out


def bin_edges(window: Window, bin_s: float = 1.0) -> tuple[tuple[float, float], ...]:
    """Consecutive bins of ``bin_s`` from the window start; the last may be short."""
    if bin_s <= 0:
        raise ValueError("bin_s must be positive")
    count = max(1, math.ceil(window.duration_s / bin_s - 1e-9))
    return tuple(
        (
            window.start_s + i * bin_s,
            min(window.start_s + (i + 1) * bin_s, window.end_s),
        )
        for i in range(count)
    )


def bin_rates(
    intervals: Sequence[RateInterval], edges: Sequence[tuple[float, float]]
) -> Bins:
    """Average rate of non-overlapping ``intervals`` in each bin (None = no data)."""
    ordered = sorted(intervals, key=lambda iv: iv.t0)
    out: Bins = []
    first = 0
    for b0, b1 in edges:
        while first < len(ordered) and ordered[first].t1 <= b0:
            first += 1
        nbytes = covered = 0.0
        i = first
        while i < len(ordered) and ordered[i].t0 < b1:
            overlap = min(ordered[i].t1, b1) - max(ordered[i].t0, b0)
            if overlap > 0:
                nbytes += ordered[i].bytes_per_s * overlap
                covered += overlap
            i += 1
        out.append(nbytes / covered if covered > 0 else None)
    return out


def sum_bins(series: Iterable[Bins]) -> Bins:
    """Element-wise sum; a bin is None only when every series lacks it."""
    total: Bins = []
    for bins in series:
        if not total:
            total = list(bins)
            continue
        total = [
            a if b is None else (b if a is None else a + b) for a, b in zip(total, bins)
        ]
    return total


def _combine(a: Bins, b: Bins, fn: Callable[[float, float], float]) -> Bins:
    return [None if x is None or y is None else fn(x, y) for x, y in zip(a, b)]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    k = (len(ordered) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize_bins(bins: Bins, edges: Sequence[tuple[float, float]]) -> RateSummary:
    """mean (time-weighted), p99 and peak of 1-s bin rates, in GB/s."""
    pairs = [(r, b1 - b0) for r, (b0, b1) in zip(bins, edges) if r is not None]
    if not pairs:
        return EMPTY_SUMMARY
    covered = sum(w for _, w in pairs)
    mean = sum(r * w for r, w in pairs) / covered
    rates = [r for r, _ in pairs]
    return RateSummary(
        mean_gbs=mean / GB,
        p99_gbs=percentile(rates, 99) / GB,
        peak_gbs=max(rates) / GB,
        covered_s=covered,
    )


def summarize(
    rates: CounterRates | Sequence[RateInterval],
    window: Window | Sequence[float] | None = None,
    bin_s: float = 1.0,
) -> RateSummary:
    """mean / p99 / peak (GB/s, peak = max over 1-s bins) of one series."""
    intervals = rates.intervals if isinstance(rates, CounterRates) else tuple(rates)
    if window is None:
        if not intervals:
            return EMPTY_SUMMARY
        win = Window(min(iv.t0 for iv in intervals), max(iv.t1 for iv in intervals))
    else:
        win = as_window(window)
    edges = bin_edges(win, bin_s)
    return summarize_bins(bin_rates(intervals, edges), edges)


def _bins_bytes(bins: Bins, edges: Sequence[tuple[float, float]]) -> float:
    return sum(r * (b1 - b0) for r, (b0, b1) in zip(bins, edges) if r is not None)


def nic_effective_count(bytes_per_hca: Iterable[float]) -> float | None:
    """(sum b)^2 / sum(b^2): 8.0 for eight equally busy NICs, 1.0 for one."""
    values = [max(0.0, float(b)) for b in bytes_per_hca]
    squares = sum(b * b for b in values)
    if squares <= 0:
        return None
    return sum(values) ** 2 / squares


def calibrate_ib_scale(known_bytes: float, measured_bytes: float) -> float:
    """Factor turning counter bytes into payload bytes.

    Compare the bytes an NCCL microbenchmark is known to have moved over the
    fabric with the IB counter delta (already x4) observed meanwhile; pass the
    result as ``analyze_point_bandwidth(..., ib_scale=...)``. Roughly 4 or 0.25
    means the 4-byte unit assumption is wrong for this HCA.
    """
    if known_bytes <= 0 or measured_bytes <= 0:
        raise ValueError(
            f"calibration needs positive byte counts, got known={known_bytes} "
            f"measured={measured_bytes}"
        )
    return known_bytes / measured_bytes


# --------------------------------------------------------------------------
# Reference peaks
# --------------------------------------------------------------------------


def _scope_nodes(scope: str) -> set[str]:
    return {token for token in _SCOPE_SPLIT_RE.split(scope) if token}


def _positive(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def per_device_gbs(peak: PeakResult, node_devices: int = 1) -> float:
    """``peak`` normalised to one device (see the module docstring)."""
    for key in DETAIL_PER_DEVICE_KEYS:
        value = _positive(peak.details.get(key))
        if value is not None:
            return value
    if "/" in peak.scope:
        return peak.value_gbs
    if peak.plane == "ib":
        per_node = _positive(peak.details.get("per_node_gbs")) or peak.value_gbs
        return per_node / max(1, peak.nic_count or node_devices)
    if "aggregate" in peak.test.lower():
        count = peak.details.get("gpu_count")
        divisor = count if isinstance(count, int) and count > 0 else node_devices
        return peak.value_gbs / max(1, divisor)
    return peak.value_gbs


def _candidates(
    peaks: Sequence[PeakResult],
    plane: Plane,
    nic_count: int | None,
    test_filter: str | None,
    *,
    prefer_gdr: bool = False,
) -> list[PeakResult]:
    pool = [
        p
        for p in peaks
        if p.plane == plane
        and p.details.get("engaged") is not False
        and (test_filter is None or test_filter in p.test.lower())
    ]
    if prefer_gdr and plane == "ib":
        # NCCL collective bus bandwidth is not a per-NIC rate: dividing it by
        # the NIC count understates the reference ~8x. Without a GPU-direct
        # row the per-HCA reference is unknown rather than wrong.
        pool = [p for p in pool if "gdr" in p.test.lower()]
    if nic_count is not None:
        same = [p for p in pool if p.nic_count == nic_count]
        if same:
            return same
    return pool


def per_device_ref(
    peaks: Sequence[PeakResult],
    plane: Plane,
    node: str | None,
    *,
    nic_count: int | None = None,
    test_filter: str | None = None,
    node_devices: int = 1,
) -> float | None:
    """Best per-device reference of ``plane``, preferring ``node``'s results."""
    pool = _candidates(peaks, plane, nic_count, test_filter, prefer_gdr=True)
    if node is not None:
        local = [p for p in pool if node in _scope_nodes(p.scope)]
        pool = local or pool
    values = [per_device_gbs(p, node_devices) for p in pool]
    values = [v for v in values if v > 0]
    return max(values) if values else None


def ref_peak_gbs(
    peaks: Sequence[PeakResult],
    plane: Plane,
    scope: str,
    *,
    devices: int = 1,
    nic_count: int | None = None,
    test_filter: str | None = None,
    node_devices: int = 1,
) -> float | None:
    """Speed-of-light reference for a stat (matching rule: module docstring)."""
    if "/" in scope:  # a device stat: an identical-scope row is used as-is
        pool = _candidates(peaks, plane, nic_count, test_filter)
        exact = [p.value_gbs for p in pool if p.scope == scope and p.value_gbs > 0]
        if exact:
            return max(exact)
    node = scope.split("/", 1)[0]
    per_device = per_device_ref(
        peaks,
        plane,
        None if node == "cluster" else node,
        nic_count=nic_count,
        test_filter=test_filter,
        node_devices=node_devices,
    )
    return None if per_device is None else per_device * max(1, devices)


# --------------------------------------------------------------------------
# GPU <-> HCA pairing
# --------------------------------------------------------------------------


def gpu_hca_pairing(
    node: NodeInventory, selected: Sequence[str]
) -> tuple[dict[int, str], bool]:
    """GPU index -> HCA carrying its RDMA traffic, and whether the NIC planner
    produced it.

    When ``selected`` is what ``nicplan.plan_node`` picks for that NIC count,
    its assignment is the one the transports were configured with. Otherwise
    each GPU goes to its nearest selected HCA (``nicplan.distance``) with the
    same ceil(GPUs / HCAs) load cap.
    """
    active = {h.name for h in node.active_hcas}
    wanted = tuple(sorted(name for name in selected if name in active))
    if not wanted or not node.gpus:
        return {}, False
    count = None if len(wanted) == len(active) else len(wanted)
    plan = nicplan.plan_node(node, count)
    if tuple(sorted(plan.hcas)) == wanted:
        return dict(plan.gpu_to_hca), True
    by_name = {h.name: h for h in node.hcas}
    return nicplan.assign(node, [by_name[name] for name in wanted]), False


# --------------------------------------------------------------------------
# Point analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PointBandwidth:
    """Bandwidth evidence of one benchmark point (feeds ``PointResult``)."""

    stats: tuple[BandwidthStat, ...] = ()
    cross_node_gbs: float | None = None
    nic_effective: float | None = None
    nic_engaged: bool | None = None
    staging_suspected: bool = False
    coverage: float = 0.0
    notes: tuple[str, ...] = ()
    # node -> GPU index -> HCA used to derive the "pcie_nic" plane.
    pairing: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def stat(
        self, plane: str, scope: str, direction: str = "tx"
    ) -> BandwidthStat | None:
        for s in self.stats:
            if s.plane == plane and s.scope == scope and s.direction == direction:
                return s
        return None


@dataclass
class _NodeIb:
    node: str
    selected: tuple[str, ...]
    active: int
    tx_bins: Bins
    rx_bins: Bins
    tx: RateSummary
    rx: RateSummary
    nominal_gbs: float | None
    hca_bins: dict[str, tuple[Bins, Bins]] = field(default_factory=dict)
    hca_bytes: dict[str, float] = field(default_factory=dict)
    hca_peak_gbs: dict[str, float] = field(default_factory=dict)
    hca_nominal: dict[str, float | None] = field(default_factory=dict)


@dataclass
class _Ctx:
    window: Window
    edges: tuple[tuple[float, float], ...]
    peaks: Sequence[PeakResult]
    gpu: GpuProfile
    ib_scale: float
    stats: list[BandwidthStat] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def add(
        self,
        plane: Plane,
        scope: str,
        direction: str,
        summary: RateSummary,
        *,
        ref: float | None,
        nominal: float | None,
        method: Method = "measured",
    ) -> None:
        if not summary.available:
            summary, method = EMPTY_SUMMARY, "unavailable"
        self.stats.append(
            BandwidthStat(
                plane=plane,
                scope=scope,
                direction=direction,
                mean_gbs=round(summary.mean_gbs, 4),
                p99_gbs=round(summary.p99_gbs, 4),
                peak_gbs=round(summary.peak_gbs, 4),
                ref_peak_gbs=None if ref is None else round(ref, 4),
                nominal_gbs=None if nominal is None else round(nominal, 4),
                method=method,
            )
        )

    def summary(self, bins: Bins) -> RateSummary:
        return summarize_bins(bins, self.edges)


def _natural(text: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def _selected(node: NodeInventory, selection: Sequence[str] | None) -> tuple[str, ...]:
    if selection:
        return tuple(selection)
    return tuple(h.name for h in node.active_hcas)


def _hca_nominal(node: NodeInventory, hca: str) -> float | None:
    for info in node.hcas:
        if info.name == hca:
            return info.nominal_gbs
    return None


def _count_in(times: Sequence[float], window: Window) -> int:
    return sum(1 for t in times if window.contains(t))


def _coverage_counts(
    node: NodeInventory, samples: TelemetrySamples | None, window: Window
) -> tuple[int, int]:
    """(expected, present) IB + PCIe samples of one node inside the window."""
    if samples is None:
        series = sum(len(h.active_ports) for h in node.hcas) + len(node.gpus)
        return series * max(1, int(window.duration_s + 1e-6)), 0
    interval = samples.meta.interval_s if samples.meta.interval_s > 0 else 1.0
    per_series = max(1, int(window.duration_s / interval + 1e-6))
    times: list[tuple[float, ...] | None] = []
    for dev, port in samples.meta.hcas:
        found = samples.ib.get((dev, port))
        times.append(found.t if found else None)
    for gpu in samples.meta.gpus:
        found_pcie = samples.pcie.get(gpu.index)
        times.append(found_pcie.t if found_pcie else None)
    present = sum(min(per_series, _count_in(t, window)) for t in times if t)
    return per_series * len(times), present


def _analyze_ib(
    ctx: _Ctx, node: NodeInventory, samples: TelemetrySamples, selected: tuple[str, ...]
) -> _NodeIb:
    ports: dict[str, list[int]] = {}
    for dev, port in samples.ib:
        ports.setdefault(dev, []).append(port)
    active = len(node.active_hcas) or len(ports) or 1
    nic_count = len(selected) or None
    wraps = resets = 0
    saturated: list[str] = []
    hca_bins: dict[str, tuple[Bins, Bins]] = {}
    for hca in sorted(ports, key=_natural):
        directions: list[Bins] = []
        for direction in ("tx", "rx"):
            parts = []
            for port in sorted(ports[hca]):
                rates = counter_rates(
                    samples.ib[(hca, port)].points(direction),
                    ctx.window,
                    unit_bytes=samples.meta.ib_unit_bytes,
                    scale=ctx.ib_scale,
                )
                wraps += rates.wraps
                resets += rates.resets
                if rates.saturated:
                    saturated.append(f"{hca}:{port}")
                parts.append(bin_rates(rates.intervals, ctx.edges))
            directions.append(sum_bins(parts))
        hca_bins[hca] = (directions[0], directions[1])
    if wraps:
        ctx.note(f"{node.name}: 32-bit IB counters wrapped {wraps}x (corrected)")
    if resets:
        ctx.note(f"{node.name}: {resets} IB counter reset(s); those intervals dropped")
    if saturated:
        ctx.note(
            f"{node.name}: 32-bit IB counters saturated on {', '.join(sorted(set(saturated)))}"
            "; IB rates there are lower bounds"
        )
    if not hca_bins:
        ctx.note(f"{node.name}: no IB port counters in telemetry")
    node_tx = sum_bins(b[0] for b in hca_bins.values()) or [None] * len(ctx.edges)
    node_rx = sum_bins(b[1] for b in hca_bins.values()) or [None] * len(ctx.edges)
    nominals = [_hca_nominal(node, h) for h in selected]
    known = [n for n in nominals if n is not None]
    result = _NodeIb(
        node=node.name,
        selected=selected,
        active=active,
        tx_bins=node_tx,
        rx_bins=node_rx,
        tx=ctx.summary(node_tx),
        rx=ctx.summary(node_rx),
        nominal_gbs=sum(known) if known else None,
    )
    for hca, (tx_bins, rx_bins) in hca_bins.items():
        scope = f"{node.name}/{hca}"
        ref = ref_peak_gbs(
            ctx.peaks, "ib", scope, nic_count=nic_count, node_devices=active
        )
        nominal = _hca_nominal(node, hca)
        tx, rx = ctx.summary(tx_bins), ctx.summary(rx_bins)
        ctx.add("ib", scope, "tx", tx, ref=ref, nominal=nominal)
        ctx.add("ib", scope, "rx", rx, ref=ref, nominal=nominal)
        result.hca_bins[hca] = (tx_bins, rx_bins)
        result.hca_bytes[hca] = _bins_bytes(tx_bins, ctx.edges) + _bins_bytes(
            rx_bins, ctx.edges
        )
        result.hca_peak_gbs[hca] = max(tx.peak_gbs, rx.peak_gbs)
        result.hca_nominal[hca] = nominal
    node_ref = ref_peak_gbs(
        ctx.peaks,
        "ib",
        node.name,
        devices=len(selected),
        nic_count=nic_count,
        node_devices=active,
    )
    for direction, summary in (("tx", result.tx), ("rx", result.rx)):
        ctx.add(
            "ib",
            node.name,
            direction,
            summary,
            ref=node_ref,
            nominal=result.nominal_gbs,
        )
    return result


def _gpu_index_map(node: NodeInventory, meta: TelemetryMeta) -> dict[int, int]:
    """Telemetry (NVML) GPU index -> inventory (nvidia-smi) index, by bus id."""
    by_bus = {g.pci_bus_id: g.index for g in node.gpus}
    mapping: dict[int, int] = {}
    for gpu in meta.gpus:
        if gpu.pci_bus_id in by_bus:
            mapping[gpu.index] = by_bus[gpu.pci_bus_id]
    return mapping


def _min_share(gpu_bins: Bins, hca_bins: Bins, sharing: int) -> Bins:
    return _combine(gpu_bins, hca_bins, lambda g, h: min(g, h / sharing))


def _analyze_gpus(
    ctx: _Ctx,
    node: NodeInventory,
    samples: TelemetrySamples,
    ib: _NodeIb,
    pairing: Mapping[int, str],
) -> None:
    gpu = ctx.gpu
    gpus = len(node.gpus) or len(samples.meta.gpus) or 1
    index_map = _gpu_index_map(node, samples.meta)
    interval = samples.meta.interval_s if samples.meta.interval_s > 0 else 1.0
    pcie: dict[int, tuple[Bins, Bins]] = {}
    for tele_index, series in sorted(samples.pcie.items()):
        index = index_map.get(tele_index, tele_index)
        scope = f"{node.name}/gpu{index}"
        tx = bin_rates(pcie_intervals(series, "tx", interval, ctx.window), ctx.edges)
        rx = bin_rates(pcie_intervals(series, "rx", interval, ctx.window), ctx.edges)
        pcie[index] = (tx, rx)
        ref = ref_peak_gbs(ctx.peaks, "pcie", scope, node_devices=gpus)
        for direction, bins in (("tx", tx), ("rx", rx)):
            ctx.add(
                "pcie",
                scope,
                direction,
                ctx.summary(bins),
                ref=ref,
                nominal=gpu.nominal_pcie_gbs,
            )
    if not pcie:
        ctx.add(
            "pcie",
            node.name,
            "bidir",
            EMPTY_SUMMARY,
            ref=None,
            nominal=gpu.nominal_pcie_gbs,
        )
        ctx.note(
            f"{node.name}: GPU PCIe throughput unavailable "
            f"(pcie_source={samples.meta.pcie_source}); pcie_nic not derived"
        )

    if samples.nvlink:
        for tele_index, series in sorted(samples.nvlink.items()):
            index = index_map.get(tele_index, tele_index)
            scope = f"{node.name}/gpu{index}"
            ref = ref_peak_gbs(ctx.peaks, "nvlink", scope, node_devices=gpus)
            for direction in ("tx", "rx"):
                rates = counter_rates(
                    series.points(direction),
                    ctx.window,
                    unit_bytes=1,
                    counter_bits=64,
                )
                if rates.resets:
                    ctx.note(f"{scope}: NVLink counter reset; interval dropped")
                ctx.add(
                    "nvlink",
                    scope,
                    direction,
                    ctx.summary(bin_rates(rates.intervals, ctx.edges)),
                    ref=ref,
                    nominal=gpu.nominal_nvlink_gbs,
                )
    else:
        ctx.add(
            "nvlink",
            node.name,
            "bidir",
            EMPTY_SUMMARY,
            ref=None,
            nominal=gpu.nominal_nvlink_gbs,
        )
        if gpu.nvlink:
            ctx.note(
                f"{node.name}: NVLink counters unavailable "
                f"(nvlink_source={samples.meta.nvlink_source}); intra-node "
                "GPU-GPU traffic not measured"
            )
        else:
            ctx.note(
                f"{node.name}: PCIe-only GPUs (no NVLink): GPU-GPU P2P is derived "
                "as GPU PCIe minus the paired-NIC share and includes host<->GPU "
                "copies (upper bound)"
            )

    sharing = Counter(pairing.values())
    gdr = per_device_ref(
        ctx.peaks,
        "ib",
        node.name,
        nic_count=len(ib.selected) or None,
        test_filter="gdr",
        node_devices=ib.active,
    )
    unpaired: list[str] = []
    for index, (gpu_tx, gpu_rx) in sorted(pcie.items()):
        scope = f"{node.name}/gpu{index}"
        hca = pairing.get(index)
        if hca is None or hca not in ib.hca_bins:
            unpaired.append(str(index))
            continue
        k = sharing[hca]
        hca_tx, hca_rx = ib.hca_bins[hca]
        nic_tx = _min_share(gpu_tx, hca_tx, k)
        nic_rx = _min_share(gpu_rx, hca_rx, k)
        hca_nominal = ib.hca_nominal.get(hca)
        nominal = gpu.nominal_pcie_gbs
        if hca_nominal:
            nominal = min(nominal, hca_nominal / k)
        ref = ref_peak_gbs(ctx.peaks, "pcie_nic", scope, node_devices=gpus)
        if ref is None and gdr is not None:
            ref = gdr / k
        for direction, bins in (("tx", nic_tx), ("rx", nic_rx)):
            ctx.add(
                "pcie_nic",
                scope,
                direction,
                ctx.summary(bins),
                ref=ref,
                nominal=nominal,
                method="derived",
            )
        if gpu.nvlink:
            continue
        p2p_scope = f"{scope}/p2p"
        p2p_ref = ref_peak_gbs(
            ctx.peaks, "pcie", p2p_scope, test_filter="p2p", node_devices=gpus
        )
        for direction, total, nic in (
            ("tx", gpu_tx, nic_tx),
            ("rx", gpu_rx, nic_rx),
        ):
            p2p = _combine(total, nic, lambda a, b: max(0.0, a - b))
            ctx.add(
                "pcie",
                p2p_scope,
                direction,
                ctx.summary(p2p),
                ref=p2p_ref,
                nominal=gpu.nominal_pcie_gbs,
                method="derived",
            )
    if unpaired:
        ctx.note(
            f"{node.name}: GPU(s) {', '.join(unpaired)} have no paired HCA with "
            "counters; pcie_nic not derived for them"
        )


def _check_control_net(
    ctx: _Ctx, node: NodeInventory, samples: TelemetrySamples
) -> None:
    for iface, series in samples.net.items():
        worst = 0.0
        for direction in ("tx", "rx"):
            rates = counter_rates(
                series.points(direction), ctx.window, unit_bytes=1, counter_bits=64
            )
            worst = max(
                worst, ctx.summary(bin_rates(rates.intervals, ctx.edges)).mean_gbs
            )
        if worst > CONTROL_NET_WARN_GBS:
            ctx.note(
                f"{node.name}: control interface {iface} carried {worst:.3f} GB/s "
                "mean; bulk data may be leaking onto the control network"
            )


def _engaged(ctx: _Ctx, ib: _NodeIb) -> bool:
    total = sum(ib.hca_bytes.values())
    if total <= 0:
        return False
    outside = {h: b for h, b in ib.hca_bytes.items() if h not in ib.selected}
    outside_share = sum(outside.values()) / total
    idle = [
        h for h in ib.selected if ib.hca_bytes.get(h, 0.0) <= SELECTED_MIN_SHARE * total
    ]
    if outside_share >= OUTSIDE_MAX_SHARE:
        busiest = max(outside, key=outside.get)
        ctx.note(
            f"{ib.node}: HCAs outside the NIC selection carried "
            f"{outside_share:.1%} of IB bytes (most: {busiest}); restriction "
            "not engaged"
        )
    if idle:
        ctx.note(f"{ib.node}: selected HCA(s) {', '.join(idle)} carried no traffic")
    return outside_share < OUTSIDE_MAX_SHARE and not idle


def _staged(ctx: _Ctx, ib: _NodeIb) -> bool:
    ref = per_device_ref(
        ctx.peaks,
        "ib",
        ib.node,
        nic_count=len(ib.selected) or None,
        test_filter="gdr",
        node_devices=ib.active,
    )
    if ref is None:
        ctx.note("no GPU-direct (gdr) IB reference peak; host-staging check skipped")
        return False
    best = max((ib.hca_peak_gbs.get(h, 0.0) for h in ib.selected), default=0.0)
    if best >= STAGING_PEAK_FRACTION * ref:
        return False
    ctx.note(
        f"{ib.node}: best selected-HCA peak {best:.2f} GB/s < "
        f"{STAGING_PEAK_FRACTION:.0%} of the {ref:.1f} GB/s per-NIC GPU-direct "
        "reference; host staging suspected (or demand is simply low)"
    )
    return True


def analyze_point_bandwidth(
    samples_by_node: Mapping[str, TelemetrySamples],
    window: Window | Sequence[float],
    inventory: Inventory,
    nic_selection: Mapping[str, Sequence[str]],
    peaks: Sequence[PeakResult],
    gpu_profile: GpuProfile,
    *,
    pairing: Mapping[str, Mapping[int, str]] | None = None,
    ib_scale: float = 1.0,
    bin_s: float = 1.0,
) -> PointBandwidth:
    """Attribute one point's traffic to IB, NVLink, PCIe and GPU<->NIC planes.

    ``window`` is the AIPerf profiling window in unix seconds (a ``Window`` or
    a ``(start, end)`` pair), ``nic_selection`` the plan's node -> HCAs, and
    ``pairing`` (optional) overrides the GPU -> HCA mapping used for the
    derived ``pcie_nic`` plane. ``ib_scale`` comes from
    ``calibrate_ib_scale``.
    """
    win = as_window(window)
    ctx = _Ctx(win, bin_edges(win, bin_s), tuple(peaks), gpu_profile, ib_scale)
    if ib_scale != 1.0:
        ctx.note(f"IB bytes scaled by calibration factor {ib_scale:.3f}")
    nodes_ib: list[_NodeIb] = []
    used_pairing: dict[str, dict[int, str]] = {}
    expected = present = 0
    for node in inventory.nodes:
        samples = samples_by_node.get(node.name)
        node_expected, node_present = _coverage_counts(node, samples, win)
        expected += node_expected
        present += node_present
        if samples is None:
            ctx.note(
                f"{node.name}: no telemetry; its planes are missing from this point"
            )
            for plane, nominal in (
                ("ib", None),
                ("pcie", gpu_profile.nominal_pcie_gbs),
                ("nvlink", gpu_profile.nominal_nvlink_gbs),
            ):
                ctx.add(
                    plane, node.name, "bidir", EMPTY_SUMMARY, ref=None, nominal=nominal
                )
            continue
        selected = _selected(node, nic_selection.get(node.name))
        known_hcas = {x.name for x in node.hcas}
        unknown = [h for h in selected if h not in known_hcas]
        if unknown:
            ctx.note(
                f"{node.name}: selected HCA(s) {', '.join(unknown)} not in inventory"
            )
        ib = _analyze_ib(ctx, node, samples, selected)
        nodes_ib.append(ib)
        if pairing is not None and node.name in pairing:
            node_pairing = dict(pairing[node.name])
        else:
            node_pairing, planned = gpu_hca_pairing(node, selected)
            if node_pairing and not planned:
                ctx.note(
                    f"{node.name}: NIC selection differs from the NIC planner's; "
                    "pcie_nic uses nearest-NIC pairing"
                )
        used_pairing[node.name] = node_pairing
        _analyze_gpus(ctx, node, samples, ib, node_pairing)
        _check_control_net(ctx, node, samples)

    if nodes_ib:
        counts = {len(n.selected) for n in nodes_ib}
        nominals = [n.nominal_gbs for n in nodes_ib if n.nominal_gbs is not None]
        cluster_ref = ref_peak_gbs(
            ctx.peaks,
            "ib",
            "cluster",
            devices=sum(len(n.selected) for n in nodes_ib),
            nic_count=counts.pop() if len(counts) == 1 else None,
            node_devices=max(n.active for n in nodes_ib),
        )
        for direction, attr in (("tx", "tx_bins"), ("rx", "rx_bins")):
            bins = sum_bins(getattr(n, attr) for n in nodes_ib)
            ctx.add(
                "ib",
                "cluster",
                direction,
                ctx.summary(bins),
                ref=cluster_ref,
                nominal=sum(nominals) if nominals else None,
            )

    measured = [n for n in nodes_ib if n.tx.available or n.rx.available]
    cross = sum(n.tx.mean_gbs for n in measured) / len(measured) if measured else None
    nic_engaged: bool | None = None
    nic_effective: float | None = None
    staging = False
    if cross is None or cross < CROSS_NODE_MIN_GBS:
        ctx.note(
            "no cross-node IB traffic (< "
            f"{CROSS_NODE_MIN_GBS} GB/s per node): NIC engagement not assessed"
        )
    else:
        busy = [
            n for n in measured if n.tx.mean_gbs + n.rx.mean_gbs >= CROSS_NODE_MIN_GBS
        ]
        verdicts = [_engaged(ctx, n) for n in busy]
        nic_engaged = all(verdicts) if verdicts else None
        effective = [nic_effective_count(n.hca_bytes.values()) for n in busy]
        known = [e for e in effective if e is not None]
        nic_effective = round(sum(known) / len(known), 3) if known else None
        # A list, not a generator: every staged node gets its note.
        staging = any([_staged(ctx, n) for n in busy])

    coverage = present / expected if expected else 0.0
    if coverage < LOW_COVERAGE:
        ctx.note(f"telemetry coverage {coverage:.0%} of expected samples")
    return PointBandwidth(
        stats=tuple(ctx.stats),
        cross_node_gbs=None if cross is None else round(cross, 4),
        nic_effective=nic_effective,
        nic_engaged=nic_engaged,
        staging_suspected=staging,
        coverage=round(min(1.0, coverage), 4),
        notes=tuple(ctx.notes),
        pairing=used_pairing,
    )
