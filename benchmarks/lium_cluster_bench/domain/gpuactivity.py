# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU hardware activity under load from NVML GPM. Pure: no I/O.

Inputs are the ``gpm`` records of ``agents/telemetry_agent.py`` (NVML GPU
Performance Monitoring, Hopper and newer), parsed by ``adapters/telemetry.py``
into one ``GpmSamples`` per node (``TelemetrySamples.gpm``). The output is one
``GpuActivitySummary`` per benchmark point (``PointStress.activity``),
computed over the same window as the bandwidth and stress analyses: were the
GPUs compute-bound, memory-bandwidth-bound or waiting?

Conventions
-----------
* A GPM value is the device average over one sampling interval
  ``(t - dt_s, t]`` between two NVML samples taken > 100 ms apart; there is
  no per-kernel attribution. Percent metrics are 0..100 as NVML returns them
  (``gr`` graphics/compute engine active, ``sm`` SMs busy, ``occ`` resident
  warps vs the maximum, ``tensor`` any tensor pipe active, ``dram`` DRAM
  bandwidth vs the theoretical peak); PCIe and NVLink rates arrive in MiB/s
  and are reported in GB/s (1e9 bytes/s).
* A GPU is keyed ``"node:idx"`` with the NVML index (the nvidia-smi and
  ``docker --gpus device=`` index). Per GPU, ``*_mean`` is the overlap-weighted
  mean of the intervals overlapping the window and ``*_p90`` / ``*_max`` the
  90th percentile (linear interpolation) / maximum of those interval values.
* Aggregates (fleet, role, brief) weight every sampled GPU equally:
  ``*_mean`` is the mean of the per-GPU means, ``*_max`` the busiest GPU's
  mean, ``regimes`` the number of GPUs per regime.
* ``regime`` of one GPU (heuristic thresholds below, UNVERIFIED tuning):
  ``idle`` when the engine and the SMs are below ``IDLE_PCT``; else
  ``compute`` (tensor >= ``COMPUTE_TENSOR_PCT``) or ``memory`` (DRAM >=
  ``MEMORY_DRAM_PCT``), whichever exceeds its threshold by more; else
  ``waiting``: neither pipe is near saturation, the GPU waits on something
  else (communication, host scheduling gaps, synchronization, latency-bound
  small kernels); ``unknown`` without tensor and DRAM metrics. An aggregate
  takes the regime of at least ``DOMINANT_SHARE`` of its GPUs, else ``mixed``.
* ``availability`` maps every GPU the agents listed to its status: the
  agent's start-time probe ("ok", "unsupported", an NVML error name such as
  "NVML_ERROR_NO_PERMISSION", "no-pynvml", "no-gpm-api", "disabled"), the
  error name when GPM failed on the GPU before the window ended, or
  "no-samples" for an "ok" GPU without an interval inside the window.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from benchmarks.lium_cluster_bench.domain.base import Frozen

# Field names of one GPU's entry in a ``gpm`` record (the agent's
# GPM_METRICS keys), percent metrics first.
GPM_PERCENT_FIELDS = ("gr_pct", "sm_pct", "occ_pct", "tensor_pct", "dram_pct")
GPM_RATE_FIELDS = ("pcie_tx_mibs", "pcie_rx_mibs", "nvlink_tx_mibs", "nvlink_rx_mibs")
GPM_FIELDS = GPM_PERCENT_FIELDS + GPM_RATE_FIELDS
# Percent metrics with p90 / max statistics (``gr`` keeps only its mean).
SPREAD_FIELDS = ("sm_pct", "occ_pct", "tensor_pct", "dram_pct")
RATE_OUTPUT = {
    "pcie_tx_mibs": "pcie_tx_gbs_mean",
    "pcie_rx_mibs": "pcie_rx_gbs_mean",
    "nvlink_tx_mibs": "nvlink_tx_gbs_mean",
    "nvlink_rx_mibs": "nvlink_rx_gbs_mean",
}
MIB_BYTES = 1024 * 1024
GB = 1e9
PCT_DIGITS = 1
GBS_DIGITS = 4

# Regime heuristics, percent (UNVERIFIED tuning; DCGM documents ~80 % as the
# practical DRAM-activity ceiling).
IDLE_PCT = 5.0
COMPUTE_TENSOR_PCT = 40.0
MEMORY_DRAM_PCT = 50.0
BUBBLE_GR_PCT = 80.0  # engine idle > 20 % of the time: host/scheduler gaps
LOW_SM_PCT = 50.0  # kernels resident but most SMs idle
DOMINANT_SHARE = 0.75
MIN_COVERAGE = 0.8  # share of the window a GPU's intervals should cover

STATUS_OK = "ok"
STATUS_NO_SAMPLES = "no-samples"
UNASSIGNED_ROLE = "unassigned"
SERVING_ROLES = ("agg", "prefill", "decode")

Regime = Literal["compute", "memory", "waiting", "idle", "mixed", "unknown"]


# --------------------------------------------------------------------------
# Samples (built by adapters/telemetry.parse_samples)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpmSeries:
    """One GPU's GPM intervals; sample ``i`` covers ``(t[i] - dt_s[i], t[i]]``.

    ``values[name][i]`` belongs to ``t[i]`` (``None`` where that sample
    lacked the metric); ``dt_s[i]`` is ``None`` when unknown.
    """

    t: tuple[float, ...] = ()
    dt_s: tuple[float | None, ...] = ()
    values: Mapping[str, tuple[float | None, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class GpmSamples:
    """One node's GPM records (``TelemetrySamples.gpm``)."""

    node: str = ""
    interval_s: float | None = None  # the agent's tick, used when dt is unknown
    status: Mapping[int, str] = field(default_factory=dict)  # meta "gpm"
    metrics: tuple[str, ...] = ()  # field names the agent requested
    series: Mapping[int, GpmSeries] = field(default_factory=dict)
    # GPU -> (unix time, NVML error) of a failure that disabled GPM there.
    failures: Mapping[int, tuple[float, str]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class GpuActivity(Frozen):
    """GPM statistics of one GPU over a window (percent 0..100, GB/s)."""

    gpu: str  # "node:idx"
    role: str | None = None
    samples: int = 0
    covered_s: float = 0.0
    gr_pct_mean: float | None = None
    sm_pct_mean: float | None = None
    sm_pct_p90: float | None = None
    sm_pct_max: float | None = None
    occ_pct_mean: float | None = None
    occ_pct_p90: float | None = None
    occ_pct_max: float | None = None
    tensor_pct_mean: float | None = None
    tensor_pct_p90: float | None = None
    tensor_pct_max: float | None = None
    dram_pct_mean: float | None = None
    dram_pct_p90: float | None = None
    dram_pct_max: float | None = None
    pcie_tx_gbs_mean: float | None = None
    pcie_rx_gbs_mean: float | None = None
    nvlink_tx_gbs_mean: float | None = None
    nvlink_rx_gbs_mean: float | None = None
    regime: Regime = "unknown"


class ActivityAggregate(Frozen):
    """Equal-weight GPM aggregate over a set of GPUs (fleet, role, brief)."""

    gpus_sampled: int = 0
    gr_pct_mean: float | None = None
    sm_pct_mean: float | None = None
    sm_pct_max: float | None = None  # the busiest GPU's mean
    occ_pct_mean: float | None = None
    occ_pct_max: float | None = None
    tensor_pct_mean: float | None = None
    tensor_pct_max: float | None = None
    dram_pct_mean: float | None = None
    dram_pct_max: float | None = None
    pcie_tx_gbs_mean: float | None = None  # per GPU
    pcie_rx_gbs_mean: float | None = None
    nvlink_tx_gbs_mean: float | None = None
    nvlink_rx_gbs_mean: float | None = None
    regimes: dict[str, int] = Field(default_factory=dict)  # regime -> GPUs
    regime: Regime = "unknown"
    reason: str = ""


class RoleActivity(ActivityAggregate):
    """GPM aggregate of the GPUs serving one role ("agg", "prefill", ...)."""

    role: str


class GpuActivitySummary(ActivityAggregate):
    """GPU hardware activity of one benchmark point (``PointStress.activity``)."""

    gpus_expected: int = 0  # GPUs listed by the agents' GPM probes
    availability: dict[str, str] = Field(default_factory=dict)  # "node:idx"
    roles: dict[str, RoleActivity] = Field(default_factory=dict)
    gpus: tuple[GpuActivity, ...] = ()
    notes: tuple[str, ...] = ()

    def gpu(self, key: str) -> GpuActivity | None:
        """The per-GPU row of ``"node:idx"``, if that GPU was sampled."""
        for row in self.gpus:
            if row.gpu == key:
                return row
        return None


class ActivityBrief(ActivityAggregate):
    """Compact copy of a summary for ``ExperimentSummary`` and the KB."""

    gpus_expected: int = 0
    availability: dict[str, int] = Field(default_factory=dict)  # status -> GPUs
    roles: dict[str, RoleActivity] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def gpu_key(node: str, index: int) -> str:
    """``"node:idx"``: the key of one GPU in every activity mapping."""
    return f"{node}:{index}"


def _natural(text: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def _window_bounds(window: Any) -> tuple[float, float]:
    """``(start_s, end_s)`` of a pair or a ``Window``-like object."""
    if hasattr(window, "start_s") and hasattr(window, "end_s"):
        start, end = float(window.start_s), float(window.end_s)
    else:
        start, end = (float(v) for v in window)
    if not end > start:
        raise ValueError(f"empty window [{start}, {end}]")
    return start, end


def _at(column: Sequence[Any], index: int) -> Any:
    return column[index] if index < len(column) else None


def _round(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _weighted_mean(pairs: Sequence[tuple[float, float]]) -> float | None:
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    return sum(v * w for v, w in pairs) / total


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    k = (len(ordered) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def classify_activity(
    gr_pct: float | None,
    sm_pct: float | None,
    tensor_pct: float | None,
    dram_pct: float | None,
) -> tuple[Regime, str]:
    """``(regime, reason)`` of GPM means (module docstring); never raises."""
    busy = [v for v in (gr_pct, sm_pct) if v is not None]
    if busy and max(busy) < IDLE_PCT:
        return "idle", f"GPU idle (engine {_pct(gr_pct)}, SMs {_pct(sm_pct)})"
    if tensor_pct is None and dram_pct is None:
        return "unknown", "tensor and DRAM activity unavailable"
    compute = (tensor_pct or 0.0) / COMPUTE_TENSOR_PCT
    memory = (dram_pct or 0.0) / MEMORY_DRAM_PCT
    if max(compute, memory) >= 1.0:
        if compute >= memory:
            return "compute", (
                f"tensor pipes active {_pct(tensor_pct)} (DRAM {_pct(dram_pct)})"
            )
        return "memory", (
            f"DRAM bandwidth {_pct(dram_pct)} used (tensor {_pct(tensor_pct)})"
        )
    if gr_pct is not None and gr_pct < BUBBLE_GR_PCT:
        return "waiting", (
            f"no kernel running {100.0 - gr_pct:.0f}% of the time "
            "(host or scheduler gaps)"
        )
    if sm_pct is not None and sm_pct < LOW_SM_PCT:
        return "waiting", (
            f"kernels active but {_pct(sm_pct)} of SMs busy "
            "(communication or small kernels)"
        )
    return "waiting", (
        f"tensor {_pct(tensor_pct)} and DRAM {_pct(dram_pct)} below saturation "
        "(latency or synchronization bound)"
    )


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def summarize_gpu(
    key: str,
    series: GpmSeries,
    window: Any,
    *,
    default_dt_s: float = 1.0,
    role: str | None = None,
) -> GpuActivity:
    """Statistics of one GPU's intervals overlapping ``window``.

    An interval without ``dt_s`` is assumed to last ``default_dt_s``.
    """
    start, end = _window_bounds(window)
    picked: dict[str, list[tuple[float, float]]] = {name: [] for name in GPM_FIELDS}
    samples = 0
    covered = 0.0
    for i, t in enumerate(series.t):
        dt = _at(series.dt_s, i)
        if dt is None or not dt > 0:
            dt = default_dt_s
        weight = min(t, end) - max(t - dt, start)
        if weight <= 0:
            continue
        present = False
        for name in GPM_FIELDS:
            value = _at(series.values.get(name, ()), i)
            if value is None or not math.isfinite(value):
                continue
            picked[name].append((float(value), weight))
            present = True
        if present:
            samples += 1
            covered += weight
    stats: dict[str, float | None] = {}
    for name in GPM_PERCENT_FIELDS:
        stats[f"{name}_mean"] = _round(_weighted_mean(picked[name]), PCT_DIGITS)
    for name in SPREAD_FIELDS:
        values = [v for v, _ in picked[name]]
        stats[f"{name}_p90"] = (
            _round(percentile(values, 90), PCT_DIGITS) if values else None
        )
        stats[f"{name}_max"] = _round(max(values), PCT_DIGITS) if values else None
    for name, output in RATE_OUTPUT.items():
        mean = _weighted_mean(picked[name])
        stats[output] = _round(
            None if mean is None else mean * MIB_BYTES / GB, GBS_DIGITS
        )
    regime: Regime = "unknown"
    if samples:
        regime, _ = classify_activity(
            stats["gr_pct_mean"],
            stats["sm_pct_mean"],
            stats["tensor_pct_mean"],
            stats["dram_pct_mean"],
        )
    return GpuActivity(
        gpu=key,
        role=role,
        samples=samples,
        covered_s=round(covered, 3),
        regime=regime,
        **stats,
    )


def _means(gpus: Sequence[GpuActivity], name: str) -> list[float]:
    return [v for v in (getattr(g, name) for g in gpus) if v is not None]


def _aggregate_regime(
    gpus: Sequence[GpuActivity], counts: Mapping[str, int]
) -> tuple[str, str]:
    """(regime, reason): the dominant GPU regime, else "mixed"."""
    if not gpus:
        return "unknown", "no GPM samples in the window"
    regime, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    if count >= DOMINANT_SHARE * len(gpus):
        members = [g for g in gpus if g.regime == regime]
        _, reason = classify_activity(
            *(
                _mean(_means(members, f"{name}_mean"))
                for name in ("gr_pct", "sm_pct", "tensor_pct", "dram_pct")
            )
        )
        if count < len(gpus):
            reason = f"{reason} on {count} of {len(gpus)} GPU(s)"
        return regime, reason
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "mixed", ", ".join(f"{name} on {n} GPU(s)" for name, n in ranked)


def _aggregate(gpus: Sequence[GpuActivity]) -> dict[str, Any]:
    """The ``ActivityAggregate`` fields of ``gpus`` (equal GPU weights)."""
    out: dict[str, Any] = {"gpus_sampled": len(gpus)}
    for name in GPM_PERCENT_FIELDS:
        out[f"{name}_mean"] = _round(_mean(_means(gpus, f"{name}_mean")), PCT_DIGITS)
    for name in SPREAD_FIELDS:
        means = _means(gpus, f"{name}_mean")
        out[f"{name}_max"] = _round(max(means), PCT_DIGITS) if means else None
    for output in RATE_OUTPUT.values():
        out[output] = _round(_mean(_means(gpus, output)), GBS_DIGITS)
    counts = Counter(g.regime for g in gpus)
    out["regimes"] = dict(sorted(counts.items()))
    out["regime"], out["reason"] = _aggregate_regime(gpus, counts)
    return out


def _by_role(
    gpus: Sequence[GpuActivity], gpu_roles: Mapping[str, str]
) -> dict[str, RoleActivity]:
    if not gpu_roles:
        return {}
    groups: dict[str, list[GpuActivity]] = {}
    for gpu in gpus:
        groups.setdefault(gpu_roles.get(gpu.gpu) or UNASSIGNED_ROLE, []).append(gpu)
    return {
        role: RoleActivity(role=role, **_aggregate(groups[role]))
        for role in sorted(groups)
    }


def activity_by_role(
    summary: GpuActivitySummary, gpu_roles: Mapping[str, str]
) -> dict[str, RoleActivity]:
    """Aggregate the summary's GPUs per serving role.

    ``gpu_roles`` maps ``"node:idx"`` -> role ("agg" | "prefill" | "decode"),
    e.g. ``gpu_role_map(plan.containers)``. Sampled GPUs missing from it are
    grouped under ``UNASSIGNED_ROLE``; an empty map gives ``{}``.
    """
    return _by_role(summary.gpus, gpu_roles)


def gpu_role_map(containers: Iterable[Any]) -> dict[str, str]:
    """``"node:idx"`` -> role of every serving container's GPUs.

    ``containers`` are ``ContainerSpec``-like objects (``node``, ``role``,
    ``gpus``), e.g. ``LaunchPlan.containers``; control and frontend
    containers are skipped.
    """
    out: dict[str, str] = {}
    for container in containers:
        role = getattr(container, "role", None)
        if role not in SERVING_ROLES:
            continue
        node = str(getattr(container, "node", ""))
        for index in getattr(container, "gpus", ()) or ():
            out[gpu_key(node, int(index))] = str(role)
    return out


def _lost_window_end(series: GpmSeries | None, failed_at: float, end: float) -> bool:
    """Whether a GPM failure at ``failed_at`` cost part of the window: the
    GPU's last interval before it ends before the window does."""
    before = [t for t in (series.t if series is not None else ()) if t <= failed_at]
    return not before or max(before) < end


def analyze_gpu_activity(
    gpm_by_node: Mapping[str, GpmSamples | None],
    window: Any,
    *,
    gpu_roles: Mapping[str, str] | None = None,
) -> GpuActivitySummary:
    """Summarize every node's GPM samples over ``window``.

    ``window`` is ``(start_s, end_s)`` in unix seconds or a
    ``bandwidth.Window``; nodes mapped to ``None`` (agents without GPM) are
    listed in ``notes``. ``gpu_roles`` (``"node:idx"`` -> role) labels the
    per-GPU rows and fills ``roles``.
    """
    start, end = _window_bounds(window)
    roles = dict(gpu_roles or {})
    availability: dict[str, str] = {}
    gpus: list[GpuActivity] = []
    notes: list[str] = []
    short: list[str] = []
    for node in sorted(gpm_by_node, key=_natural):
        samples = gpm_by_node[node]
        if samples is None:
            notes.append(f"{node}: no GPM telemetry (agent without gpm records)")
            continue
        notes.extend(f"{node}: {note}" for note in samples.notes)
        default_dt = samples.interval_s if samples.interval_s else 1.0
        indices = set(samples.status) | set(samples.series) | set(samples.failures)
        for index in sorted(indices):
            key = gpu_key(node, index)
            status = samples.status.get(index, STATUS_OK)
            activity: GpuActivity | None = None
            series = samples.series.get(index)
            if series is not None:
                activity = summarize_gpu(
                    key,
                    series,
                    (start, end),
                    default_dt_s=default_dt,
                    role=roles.get(key),
                )
                if not activity.samples:
                    activity = None
            failure = samples.failures.get(index)
            if failure is not None and _lost_window_end(series, failure[0], end):
                status = failure[1]
                notes.append(f"{key}: GPM failed ({failure[1]})")
            elif status == STATUS_OK and activity is None:
                status = STATUS_NO_SAMPLES
            availability[key] = status
            if activity is not None:
                gpus.append(activity)
                if activity.covered_s < MIN_COVERAGE * (end - start):
                    short.append(key)
    if short:
        notes.append(
            f"GPM intervals cover < {MIN_COVERAGE:.0%} of the window on "
            f"{len(short)} GPU(s): {', '.join(short)}"
        )
    missing = Counter(s for s in availability.values() if s != STATUS_OK)
    if missing:
        ranked = sorted(missing.items(), key=lambda item: (-item[1], item[0]))
        notes.append(
            "GPM unavailable: " + ", ".join(f"{s} on {n} GPU(s)" for s, n in ranked)
        )
    return GpuActivitySummary(
        gpus_expected=len(availability),
        availability=availability,
        roles=_by_role(gpus, roles),
        gpus=tuple(gpus),
        notes=tuple(dict.fromkeys(notes)),
        **_aggregate(gpus),
    )


def brief_activity(summary: GpuActivitySummary) -> ActivityBrief:
    """The summary without per-GPU rows: aggregates, status counts, roles."""
    fields = summary.model_dump(include=set(ActivityAggregate.model_fields))
    return ActivityBrief(
        **fields,
        gpus_expected=summary.gpus_expected,
        availability=dict(sorted(Counter(summary.availability.values()).items())),
        roles=dict(summary.roles),
    )
