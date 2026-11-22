# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric and GPU stress under load: congestion, errors, clocks, power. Pure.

Inputs are the ``ibx`` / ``gpux`` / ``ethx`` records of
``agents/telemetry_agent.py``, parsed by ``adapters/telemetry.py`` into one
``StressSamples`` per node (``TelemetrySamples.stress``). The output is one
``PointStress`` per benchmark point, computed over the same window as the
bandwidth analysis.

Conventions
-----------
* Counters are cumulative. An interval whose value decreases is a reset and
  is dropped (counted in ``resets``). IB PMA PortCounters fields saturate at
  ``2**width - 1`` instead of wrapping (IBTA spec widths in
  ``IB_COUNTER_BITS``); a pinned counter is listed in ``saturated`` and its
  increments are lower bounds.
* Rates (``*_per_s``) prorate each sampling interval by its overlap with the
  window. Error increments (``errors``, ``discards``) count whole intervals
  that overlap the window, so an event at most one slow interval (default
  5 s) outside the window may be attributed to the point.
* ``xmit_wait_fraction`` = port_xmit_wait ticks/s x ``XMIT_WAIT_TICK_S``,
  clamped to [0, 1]: the share of time the port had data but no credits.
  UNVERIFIED: the tick duration is device specific; the raw tick rate is
  kept so the estimate can be recomputed.
* GPU energy is the NVML total-energy counter delta over the window, with
  linear interpolation at the window edges ("counter"); otherwise the
  trapezoid integral of sampled power ("integrated"). A GPU whose samples
  cover less than ``MIN_ENERGY_COVERAGE`` of the window has no energy; one
  covering more is scaled to the full window. Energy covers GPUs only (not
  CPUs, NICs or switches) of every node that reported, idle GPUs included.
* Throttle fractions are the share of samples inside the window whose NVML
  clocks-event bitmask has the reason's bit set; ``throttled_fraction``
  counts any reason in ``LIMITING_REASONS`` (power cap, power brake, HW
  slowdown, SW/HW thermal).
* ``activity`` is the NVML GPM view of the same window (SM, tensor, DRAM,
  PCIe, NVLink activity; ``domain/gpuactivity.py``), present when the agents
  wrote GPM records or availability (``TelemetrySamples.gpm``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from benchmarks.cluster_bench.domain.base import Frozen
from benchmarks.cluster_bench.domain.gpuactivity import (
    GpmSamples,
    GpuActivitySummary,
    analyze_gpu_activity,
)

# NVML clocks-event ("throttle") reason bits, nvml.h nvmlClocksEventReason*.
REASON_BITS: dict[str, int] = {
    "gpu_idle": 0x1,
    "app_clocks": 0x2,
    "sw_power_cap": 0x4,
    "hw_slowdown": 0x8,
    "sync_boost": 0x10,
    "sw_thermal": 0x20,
    "hw_thermal": 0x40,
    "hw_power_brake": 0x80,
    "display_clocks": 0x100,
}
# Reasons that lower clocks under load; the others are informational.
LIMITING_REASONS = (
    "sw_power_cap",
    "hw_slowdown",
    "sw_thermal",
    "hw_thermal",
    "hw_power_brake",
)
LIMITING_MASK = sum(REASON_BITS[r] for r in LIMITING_REASONS)

# sysfs names the agent samples (``counters/`` and ``hw_counters/``).
IB_PORT_COUNTERS = (
    "port_xmit_wait",
    "port_xmit_discards",
    "port_rcv_errors",
    "symbol_error",
    "link_downed",
    "local_link_integrity_errors",
    "excessive_buffer_overrun_errors",
    "port_rcv_remote_physical_errors",
    "VL15_dropped",
)
IB_HW_COUNTERS = (
    "np_cnp_sent",
    "rp_cnp_handled",
    "np_ecn_marked_roce_packets",
    "out_of_buffer",
    "out_of_sequence",
    "packet_seq_err",
    "local_ack_timeout_err",
    "implied_nak_seq_err",
    "rnr_nak_retry_err",
    "req_cqe_error",
    "resp_cqe_error",
)
# Congestion signals reported as rates; every other sampled name is an error.
IB_RATE_COUNTERS = (
    "port_xmit_wait",
    "np_cnp_sent",
    "rp_cnp_handled",
    "np_ecn_marked_roce_packets",
)
IB_ERROR_COUNTERS = tuple(
    n for n in IB_PORT_COUNTERS + IB_HW_COUNTERS if n not in IB_RATE_COUNTERS
)
# PortCounters field widths (IBTA); these saturate rather than wrap.
IB_COUNTER_BITS: dict[str, int] = {
    "symbol_error": 16,
    "link_downed": 8,
    "port_rcv_errors": 16,
    "port_rcv_remote_physical_errors": 16,
    "port_xmit_discards": 16,
    "local_link_integrity_errors": 4,
    "excessive_buffer_overrun_errors": 4,
    "VL15_dropped": 16,
    "port_xmit_wait": 32,
}
# UNVERIFIED: duration of one port_xmit_wait tick (device specific).
XMIT_WAIT_TICK_S = 4e-9
MIN_ENERGY_COVERAGE = 0.8
# Ethernet (RoCE) ``ethtool -S`` counter classes (mlx5 naming, UNVERIFIED
# overlap): port-level discard/ECN counters are preferred and the
# per-priority ones are used only without them, so nothing is counted twice.
_PAUSE_FRAMES_RE = re.compile(r"^(rx|tx)_(?:prio\d+_)?pause(?:_ctrl_phy)?$")
_PAUSE_DURATION_RE = re.compile(r"^(rx|tx)_(?:prio\d+_)?pause_duration$")
ETH_PORT_DISCARDS = ("rx_discards_phy", "tx_discards_phy", "rx_out_of_buffer")
_DISCARD_RE = re.compile(r"discard|drop|out_of_buffer", re.IGNORECASE)
ETH_PORT_ECN = ("rx_ecn_mark",)
_ECN_RE = re.compile(r"ecn|_marked$", re.IGNORECASE)
_CNP_RE = re.compile(r"cnp", re.IGNORECASE)
# UNVERIFIED: mlx5 documents *_pause_duration in microseconds.
PAUSE_DURATION_UNIT_S = 1e-6


# --------------------------------------------------------------------------
# Samples (built by adapters/telemetry.parse_samples)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterTable:
    """Cumulative counters of one HCA port or netdev, sampled at ``t``.

    ``values[name][i]`` belongs to ``t[i]``; ``None`` where that sample
    lacked the counter.
    """

    t: tuple[float, ...] = ()
    values: Mapping[str, tuple[int | None, ...]] = field(default_factory=dict)

    def points(self, name: str) -> list[tuple[float, int]]:
        column = self.values.get(name, ())
        return [(t, v) for t, v in zip(self.t, column) if v is not None]


@dataclass(frozen=True)
class GpuStressSeries:
    """Per-GPU clocks, temperature, power, energy and clocks-event reasons."""

    t: tuple[float, ...] = ()
    sm_mhz: tuple[float | None, ...] = ()
    mem_mhz: tuple[float | None, ...] = ()
    temp_c: tuple[float | None, ...] = ()
    power_w: tuple[float | None, ...] = ()
    energy_mj: tuple[int | None, ...] = ()
    reasons: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class StressSamples:
    """One node's stress records (``TelemetrySamples.stress``)."""

    node: str = ""
    slow_interval_s: float | None = None
    gpux_source: str | None = None
    # (HCA, port) -> "InfiniBand" | "Ethernet".
    link_layers: Mapping[tuple[str, int], str] = field(default_factory=dict)
    ibx: Mapping[tuple[str, int], CounterTable] = field(default_factory=dict)
    gpux: Mapping[int, GpuStressSeries] = field(default_factory=dict)
    ethx: Mapping[str, CounterTable] = field(default_factory=dict)  # by netdev
    eth_hca: Mapping[str, str] = field(default_factory=dict)  # netdev -> HCA
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class HcaStress(Frozen):
    """Congestion and error counters of one HCA port over the window."""

    node: str
    hca: str
    port: int
    link_layer: str | None = None
    covered_s: float = 0.0
    xmit_wait_ticks_per_s: float | None = None
    xmit_wait_fraction: float | None = None  # UNVERIFIED tick duration
    cnp_sent_per_s: float | None = None  # np_cnp_sent (this port marked)
    cnp_handled_per_s: float | None = None  # rp_cnp_handled (we slowed down)
    ecn_marked_per_s: float | None = None  # np_ecn_marked_roce_packets
    xmit_discards: int | None = None
    errors: dict[str, int] = Field(default_factory=dict)  # nonzero increments
    saturated: tuple[str, ...] = ()
    resets: int = 0


class EthStress(Frozen):
    """Pause/PFC, discard and CNP/ECN counters of one RoCE netdev."""

    node: str
    iface: str
    hca: str | None = None
    covered_s: float = 0.0
    pause_rx_per_s: float | None = None
    pause_tx_per_s: float | None = None
    pause_duration_fraction: float | None = None  # UNVERIFIED unit
    discards: int | None = None
    cnp_per_s: float | None = None
    ecn_per_s: float | None = None
    counters: dict[str, int] = Field(default_factory=dict)  # nonzero increments


class GpuStress(Frozen):
    """Power, energy, clocks, temperature and throttling of one GPU."""

    node: str
    gpu: int
    samples: int = 0
    power_mean_w: float | None = None
    power_max_w: float | None = None
    energy_j: float | None = None
    energy_method: Literal["counter", "integrated", "unavailable"] = "unavailable"
    temp_mean_c: float | None = None
    temp_max_c: float | None = None
    sm_mhz_mean: float | None = None
    sm_mhz_min: float | None = None
    mem_mhz_mean: float | None = None
    throttle: dict[str, float] = Field(default_factory=dict)  # reason -> share
    throttled_fraction: float | None = None


class PointStress(Frozen):
    """Fabric and GPU stress of one benchmark point (feeds the KB)."""

    window_start_s: float
    window_end_s: float
    nodes: tuple[str, ...] = ()
    gpus: tuple[GpuStress, ...] = ()
    hcas: tuple[HcaStress, ...] = ()
    eths: tuple[EthStress, ...] = ()
    gpu_count: int = 0
    energy_j: float | None = None  # summed over energy_gpus GPUs
    energy_gpus: int = 0
    power_total_mean_w: float | None = None  # sum of per-GPU means
    power_mean_w: float | None = None  # mean per GPU
    power_max_w: float | None = None  # highest single-GPU sample
    temp_max_c: float | None = None
    sm_mhz_mean: float | None = None
    throttle: dict[str, float] = Field(default_factory=dict)
    throttled_fraction: float | None = None
    xmit_wait_fraction_max: float | None = None
    xmit_discards_total: int = 0
    # CNPs sent / ECN-marked packets: HCA hw_counters, plus netdevs whose HCA
    # lacks them (the same RoCE packets are never counted twice).
    cnp_per_s_total: float | None = None
    ecn_per_s_total: float | None = None
    pause_per_s_total: float | None = None  # rx + tx pause/PFC frames
    eth_discards_total: int = 0
    error_increments: dict[str, int] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    # NVML GPM activity (None for agents without GPM records).
    activity: GpuActivitySummary | None = None

    @property
    def duration_s(self) -> float:
        return self.window_end_s - self.window_start_s


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def decode_reasons(mask: int | None) -> tuple[str, ...]:
    """Names of the NVML clocks-event reason bits set in ``mask``."""
    if mask is None:
        return ()
    return tuple(name for name, bit in REASON_BITS.items() if mask & bit)


def window_bounds(window: Any) -> tuple[float, float]:
    """``(start_s, end_s)`` of a ``(start, end)`` pair or a ``Window``-like
    object with ``start_s`` / ``end_s``; raises ValueError when empty."""
    if hasattr(window, "start_s") and hasattr(window, "end_s"):
        start, end = float(window.start_s), float(window.end_s)
    else:
        start, end = (float(v) for v in window)
    if not end > start:
        raise ValueError(f"empty window [{start}, {end}]")
    return start, end


@dataclass(frozen=True)
class CounterDelta:
    """Increment of one cumulative counter over a window."""

    value: float  # prorated (rates) or whole-interval (events) increment
    covered_s: float  # seconds of the window covered by valid intervals
    resets: int = 0
    saturated: bool = False

    @property
    def rate(self) -> float | None:
        return self.value / self.covered_s if self.covered_s > 0 else None


def counter_delta(
    points: Sequence[tuple[float, int]],
    window: tuple[float, float],
    *,
    prorate: bool,
    bits: int | None = None,
) -> CounterDelta | None:
    """Increment of a cumulative counter inside ``window`` (module docstring);
    ``None`` when no sampling interval overlaps the window."""
    start, end = window
    ordered = sorted(points)
    total = covered = 0.0
    resets = 0
    overlapped = False
    for (t0, v0), (t1, v1) in zip(ordered, ordered[1:]):
        dt = t1 - t0
        lo, hi = max(t0, start), min(t1, end)
        if dt <= 0 or hi <= lo:
            continue
        overlapped = True
        delta = v1 - v0
        if delta < 0:
            resets += 1
            continue
        total += delta * (hi - lo) / dt if prorate else delta
        covered += hi - lo
    if not overlapped:
        return None
    pinned = bits is not None and any(v >= (1 << bits) - 1 for _, v in ordered)
    return CounterDelta(total, covered, resets, pinned)


def _interp(points: Sequence[tuple[float, float]], t: float) -> float:
    """Linear interpolation inside ``points`` (sorted, t within range)."""
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v1
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return points[-1][1] if t >= points[-1][0] else points[0][1]


def _clipped_span(
    points: Sequence[tuple[float, float]], window: tuple[float, float]
) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    lo, hi = max(points[0][0], window[0]), min(points[-1][0], window[1])
    return (lo, hi) if hi > lo else None


def energy_from_counter(
    points: Sequence[tuple[float, float]], window: tuple[float, float]
) -> tuple[float, float] | None:
    """(joules, covered seconds) from an NVML energy counter in mJ, or None
    when it does not cover the window or resets inside it."""
    ordered = sorted(points)
    span = _clipped_span(ordered, window)
    if span is None:
        return None
    inside = [v for t, v in ordered if span[0] <= t <= span[1]]
    if any(b < a for a, b in zip(inside, inside[1:])):
        return None
    delta = _interp(ordered, span[1]) - _interp(ordered, span[0])
    if delta < 0:
        return None
    return delta / 1000.0, span[1] - span[0]


def energy_from_power(
    points: Sequence[tuple[float, float]], window: tuple[float, float]
) -> tuple[float, float] | None:
    """(joules, covered seconds) as the trapezoid integral of power samples."""
    ordered = sorted(points)
    span = _clipped_span(ordered, window)
    if span is None:
        return None
    lo, hi = span
    knots = [(lo, _interp(ordered, lo))]
    knots += [(t, w) for t, w in ordered if lo < t < hi]
    knots.append((hi, _interp(ordered, hi)))
    joules = sum(
        (t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(knots, knots[1:])
    )
    return joules, hi - lo


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _in_window(
    t: Sequence[float], column: Sequence[Any], window: tuple[float, float]
) -> list[Any]:
    return [
        v for ts, v in zip(t, column) if v is not None and window[0] <= ts <= window[1]
    ]


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _gpu_stress(
    node: str, gpu: int, series: GpuStressSeries, window: tuple[float, float]
) -> GpuStress:
    power = [float(v) for v in _in_window(series.t, series.power_w, window)]
    temps = [float(v) for v in _in_window(series.t, series.temp_c, window)]
    sm = [float(v) for v in _in_window(series.t, series.sm_mhz, window)]
    mem = [float(v) for v in _in_window(series.t, series.mem_mhz, window)]
    reasons = [int(v) for v in _in_window(series.t, series.reasons, window)]
    duration = window[1] - window[0]
    energy: float | None = None
    method: Literal["counter", "integrated", "unavailable"] = "unavailable"
    counter = [
        (t, float(v)) for t, v in zip(series.t, series.energy_mj) if v is not None
    ]
    measured = energy_from_counter(counter, window)
    if measured is not None and measured[1] >= MIN_ENERGY_COVERAGE * duration:
        energy, method = measured[0] * duration / measured[1], "counter"
    else:
        watts = [
            (t, float(v)) for t, v in zip(series.t, series.power_w) if v is not None
        ]
        integrated = energy_from_power(watts, window)
        if integrated is not None and integrated[1] >= MIN_ENERGY_COVERAGE * duration:
            energy, method = integrated[0] * duration / integrated[1], "integrated"
    throttle: dict[str, float] = {}
    throttled: float | None = None
    if reasons:
        for name, bit in REASON_BITS.items():
            share = sum(1 for r in reasons if r & bit) / len(reasons)
            if share > 0:
                throttle[name] = round(share, 4)
        throttled = round(
            sum(1 for r in reasons if r & LIMITING_MASK) / len(reasons), 4
        )
    return GpuStress(
        node=node,
        gpu=gpu,
        samples=max(len(power), len(sm), len(reasons), len(temps)),
        power_mean_w=_round(_mean(power), 2),
        power_max_w=_round(max(power), 2) if power else None,
        energy_j=_round(energy, 1),
        energy_method=method,
        temp_mean_c=_round(_mean(temps), 2),
        temp_max_c=max(temps) if temps else None,
        sm_mhz_mean=_round(_mean(sm), 1),
        sm_mhz_min=min(sm) if sm else None,
        mem_mhz_mean=_round(_mean(mem), 1),
        throttle=throttle,
        throttled_fraction=throttled,
    )


def _hca_stress(
    node: str,
    hca: str,
    port: int,
    table: CounterTable,
    link_layer: str | None,
    window: tuple[float, float],
) -> HcaStress:
    rates: dict[str, float | None] = {}
    covered = 0.0
    resets = 0
    saturated: list[str] = []
    for name in IB_RATE_COUNTERS:
        delta = counter_delta(
            table.points(name), window, prorate=True, bits=IB_COUNTER_BITS.get(name)
        )
        if delta is None:
            continue
        rates[name] = delta.rate
        covered = max(covered, delta.covered_s)
        resets += delta.resets
        if delta.saturated:
            saturated.append(name)
    errors: dict[str, int] = {}
    discards: int | None = None
    for name in sorted(set(table.values) - set(IB_RATE_COUNTERS)):
        delta = counter_delta(
            table.points(name), window, prorate=False, bits=IB_COUNTER_BITS.get(name)
        )
        if delta is None:
            continue
        covered = max(covered, delta.covered_s)
        resets += delta.resets
        if delta.saturated:
            saturated.append(name)
        if name == "port_xmit_discards":
            discards = int(delta.value)
        if delta.value > 0:
            errors[name] = int(delta.value)
    wait = rates.get("port_xmit_wait")
    return HcaStress(
        node=node,
        hca=hca,
        port=port,
        link_layer=link_layer,
        covered_s=round(covered, 3),
        xmit_wait_ticks_per_s=_round(wait, 1),
        xmit_wait_fraction=(
            None if wait is None else round(min(1.0, wait * XMIT_WAIT_TICK_S), 6)
        ),
        cnp_sent_per_s=_round(rates.get("np_cnp_sent"), 3),
        cnp_handled_per_s=_round(rates.get("rp_cnp_handled"), 3),
        ecn_marked_per_s=_round(rates.get("np_ecn_marked_roce_packets"), 3),
        xmit_discards=discards,
        errors=errors,
        saturated=tuple(sorted(saturated)),
        resets=resets,
    )


def _preferred(names: Sequence[str], port_level: Sequence[str]) -> list[str]:
    """Port-level counters when the netdev has any, else the matching rest."""
    chosen = [n for n in names if n in port_level]
    return chosen or list(names)


def _eth_stress(
    node: str,
    iface: str,
    hca: str | None,
    table: CounterTable,
    window: tuple[float, float],
) -> EthStress:
    rates: dict[str, float] = {}
    events: dict[str, int] = {}
    covered = 0.0
    for name in sorted(table.values):
        rate_delta = counter_delta(table.points(name), window, prorate=True)
        event_delta = counter_delta(table.points(name), window, prorate=False)
        if rate_delta is None or event_delta is None:
            continue
        covered = max(covered, rate_delta.covered_s)
        rates[name] = rate_delta.rate or 0.0
        events[name] = int(event_delta.value)
    frames = {n: _PAUSE_FRAMES_RE.match(n) for n in rates}
    pause = [n for n, m in frames.items() if m]
    durations = [n for n in rates if _PAUSE_DURATION_RE.match(n)]
    discard = _preferred(
        [n for n in events if _DISCARD_RE.search(n)], ETH_PORT_DISCARDS
    )
    ecn = _preferred([n for n in rates if _ECN_RE.search(n)], ETH_PORT_ECN)
    cnp = [n for n in rates if _CNP_RE.search(n)]

    def total(names: Sequence[str], digits: int = 3) -> float | None:
        return round(sum(rates[n] for n in names), digits) if names else None

    rx = [n for n in pause if frames[n].group(1) == "rx"]  # type: ignore[union-attr]
    tx = [n for n in pause if frames[n].group(1) == "tx"]  # type: ignore[union-attr]
    stalled = max((rates[n] * PAUSE_DURATION_UNIT_S for n in durations), default=0.0)
    return EthStress(
        node=node,
        iface=iface,
        hca=hca,
        covered_s=round(covered, 3),
        pause_rx_per_s=round(sum(rates[n] for n in rx), 3) if pause else None,
        pause_tx_per_s=round(sum(rates[n] for n in tx), 3) if pause else None,
        pause_duration_fraction=round(min(1.0, stalled), 6) if durations else None,
        discards=sum(events[n] for n in discard) if discard else None,
        cnp_per_s=total(cnp),
        ecn_per_s=total(ecn),
        counters={n: v for n, v in events.items() if v > 0},
    )


def _sum_known(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known), 3) if known else None


def _port_total(
    hcas: Sequence[HcaStress], eths: Sequence[EthStress], hca_attr: str, eth_attr: str
) -> float | None:
    """Sum over HCAs; a netdev counts only when its HCA lacks the counter
    (hw_counters and ethtool see the same RoCE packets)."""
    values = [getattr(h, hca_attr) for h in hcas]
    covered = {(h.node, h.hca) for h in hcas if getattr(h, hca_attr) is not None}
    values += [getattr(e, eth_attr) for e in eths if (e.node, e.hca) not in covered]
    return _sum_known(values)


def _activity(
    gpm_by_node: Mapping[str, GpmSamples | None] | None,
    bounds: tuple[float, float],
    gpu_roles: Mapping[str, str] | None,
    notes: list[str],
) -> GpuActivitySummary | None:
    """GPM activity, or None without GPM data; its failure costs only itself."""
    if not gpm_by_node or all(s is None for s in gpm_by_node.values()):
        return None
    try:
        return analyze_gpu_activity(gpm_by_node, bounds, gpu_roles=gpu_roles)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        notes.append(f"GPU activity analysis failed: {exc}")
        return None


def analyze_point_stress(
    stress_by_node: Mapping[str, StressSamples | None],
    window: Any,
    *,
    gpm_by_node: Mapping[str, GpmSamples | None] | None = None,
    gpu_roles: Mapping[str, str] | None = None,
) -> PointStress:
    """Summarize every node's stress samples over ``window``.

    ``window`` is ``(start_s, end_s)`` in unix seconds or a
    ``bandwidth.Window``; use the same window as the bandwidth analysis.
    Nodes mapped to ``None`` (no stress records) are listed in ``notes``.
    ``gpm_by_node`` (``TelemetrySamples.gpm`` per node) adds ``activity``;
    ``gpu_roles`` (``"node:idx"`` -> role, ``gpuactivity.gpu_role_map``)
    splits it per serving role.
    """
    bounds = window_bounds(window)
    notes: list[str] = []
    activity = _activity(gpm_by_node, bounds, gpu_roles, notes)
    gpus: list[GpuStress] = []
    hcas: list[HcaStress] = []
    eths: list[EthStress] = []
    nodes: list[str] = []
    for node in sorted(stress_by_node):
        samples = stress_by_node[node]
        if samples is None:
            notes.append(f"{node}: no stress telemetry (agent without ibx/gpux)")
            continue
        nodes.append(node)
        notes.extend(f"{node}: {n}" for n in samples.notes)
        for gpu in sorted(samples.gpux):
            stats = _gpu_stress(node, gpu, samples.gpux[gpu], bounds)
            if stats.samples or stats.energy_j is not None:
                gpus.append(stats)
        for hca, port in sorted(samples.ibx):
            hcas.append(
                _hca_stress(
                    node,
                    hca,
                    port,
                    samples.ibx[(hca, port)],
                    samples.link_layers.get((hca, port)),
                    bounds,
                )
            )
        for iface in sorted(samples.ethx):
            eths.append(
                _eth_stress(
                    node, iface, samples.eth_hca.get(iface), samples.ethx[iface], bounds
                )
            )
        if not samples.gpux:
            notes.append(f"{node}: no GPU stress samples (gpux)")
    with_energy = [g for g in gpus if g.energy_j is not None]
    if gpus and len(with_energy) < len(gpus):
        notes.append(
            f"energy unavailable for {len(gpus) - len(with_energy)} of "
            f"{len(gpus)} GPU(s) (window coverage < {MIN_ENERGY_COVERAGE:.0%})"
        )
    integrated = [g for g in with_energy if g.energy_method == "integrated"]
    if integrated:
        notes.append(
            f"energy of {len(integrated)} GPU(s) integrated from sampled power "
            "(no NVML energy counter)"
        )
    means = [g.power_mean_w for g in gpus if g.power_mean_w is not None]
    maxima = [g.power_max_w for g in gpus if g.power_max_w is not None]
    temps = [g.temp_max_c for g in gpus if g.temp_max_c is not None]
    sm = [g.sm_mhz_mean for g in gpus if g.sm_mhz_mean is not None]
    throttle_counts: dict[str, float] = {}
    throttle_samples = 0
    limited = 0.0
    for g in gpus:
        if g.throttled_fraction is None:
            continue
        throttle_samples += g.samples
        limited += g.throttled_fraction * g.samples
        for name, share in g.throttle.items():
            throttle_counts[name] = throttle_counts.get(name, 0.0) + share * g.samples
    errors: dict[str, int] = {}
    for h in hcas:
        for name, count in h.errors.items():
            errors[name] = errors.get(name, 0) + count
    waits = [h.xmit_wait_fraction for h in hcas if h.xmit_wait_fraction is not None]
    return PointStress(
        window_start_s=bounds[0],
        window_end_s=bounds[1],
        nodes=tuple(nodes),
        gpus=tuple(gpus),
        hcas=tuple(hcas),
        eths=tuple(eths),
        gpu_count=len(gpus),
        energy_j=(
            round(sum(g.energy_j for g in with_energy), 1) if with_energy else None
        ),
        energy_gpus=len(with_energy),
        power_total_mean_w=round(sum(means), 2) if means else None,
        power_mean_w=_round(_mean(means), 2),
        power_max_w=max(maxima) if maxima else None,
        temp_max_c=max(temps) if temps else None,
        sm_mhz_mean=_round(_mean(sm), 1),
        throttle={
            name: round(count / throttle_samples, 4)
            for name, count in sorted(throttle_counts.items())
        }
        if throttle_samples
        else {},
        throttled_fraction=(
            round(limited / throttle_samples, 4) if throttle_samples else None
        ),
        xmit_wait_fraction_max=max(waits) if waits else None,
        xmit_discards_total=sum(h.xmit_discards or 0 for h in hcas),
        cnp_per_s_total=_port_total(hcas, eths, "cnp_sent_per_s", "cnp_per_s"),
        ecn_per_s_total=_port_total(hcas, eths, "ecn_marked_per_s", "ecn_per_s"),
        pause_per_s_total=_sum_known(
            [e.pause_rx_per_s for e in eths] + [e.pause_tx_per_s for e in eths]
        ),
        eth_discards_total=sum(e.discards or 0 for e in eths),
        error_increments=dict(sorted(errors.items())),
        notes=tuple(dict.fromkeys(notes)),
        activity=activity,
    )


def energy_per_token_j(
    stress: PointStress | None, output_tokens: float | int | None
) -> float | None:
    """GPU joules per output token over the point's window (None if unknown).

    Pass the output tokens AIPerf counted inside the same window
    (``PointMetrics.total_output_tokens``).
    """
    if stress is None or stress.energy_j is None or not output_tokens:
        return None
    if output_tokens <= 0 or stress.energy_j <= 0:
        return None
    return round(stress.energy_j / float(output_tokens), 6)
