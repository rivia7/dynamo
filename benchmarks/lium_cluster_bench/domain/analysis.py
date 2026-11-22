# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure analysis: per-point validity audit, noise floor, saturation,
interactivity floors, MTP and NIC comparisons.

Rules encode agent-docs/rules/benchmarking/*.md: one measured run per point;
a once-per-series n=3 pilot sets the noise floor; deltas inside it are noise;
saturation needs a flat curve top; invalid runs are kept, never ranked.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Iterable, Literal, Sequence

from benchmarks.lium_cluster_bench.domain.schema import (
    INTERACTIVITY_FLOORS,
    AuditResult,
    BenchSpec,
    EngagementResult,
    FloorResult,
    MtpComparison,
    NicArm,
    NoiseResult,
    PointMetrics,
    PointResult,
)

MAX_ERROR_RATE = (
    0.01  # agent-docs: error storms invalidate; AgentX: >1% overflow invalid
)
MIN_DURATION_FRACTION = 0.9
MIN_TELEMETRY_COVERAGE = 0.9
PROVISIONAL_NOISE_PCT = 5.0  # placeholder until the pilot measures the real floor
OVERFLOW_MARKERS = ("context", "length", "too long", "maximum")

# Two-sided Student t quantiles at 97.5% by degrees of freedom.
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


# Render-time notes name one container: "cb-<fp>-<role><instance>-r<rank>: <text>".
_CONTAINER_NOTE = re.compile(r"^(cb-[0-9a-f]+-[a-z]+\d+-r\d+): (.+)$")


def collapse_container_notes(notes: Iterable[str]) -> list[str]:
    """Each note once; per-container notes with the same text as one line.

    Every deployment re-renders the same clamps for every container, so a
    run repeats them dozens of times; the collapsed line counts the
    distinct containers instead. Order of first appearance is kept.
    """
    order: list[tuple[bool, str]] = []  # (per-container group?, text)
    names: dict[str, set[str]] = {}
    for note in notes:
        match = _CONTAINER_NOTE.match(note)
        key = (True, match[2]) if match else (False, note)
        if key not in order:
            order.append(key)
        if match:
            names.setdefault(match[2], set()).add(match[1])
    out = []
    for grouped, text in order:
        if not grouped:
            out.append(text)
            continue
        found = sorted(names[text])
        head = found[0] if len(found) == 1 else f"{len(found)} containers"
        out.append(f"{head}: {text}")
    return out


def audit_point(
    metrics: PointMetrics | None,
    *,
    spec: BenchSpec,
    concurrency: int,
    expected_aiperf_version: str,
    engagement: EngagementResult | None,
    server_alive: bool,
    telemetry_coverage: float | None,
    nic_restricted: bool,
    nic_engaged: bool | None,
    nvlink_off: bool = False,
    nvlink_p99_gbs: float | None = None,
    nvlink_leak_gbs: float = 1.0,
) -> AuditResult:
    """Classify one measured run as valid / valid_with_recovery / invalid."""
    invalid: list[str] = []
    recovered: list[str] = []
    if metrics is None:
        return AuditResult(status="invalid", reasons=("no AIPerf export found",))
    if metrics.aiperf_version != expected_aiperf_version:
        invalid.append(
            f"aiperf {metrics.aiperf_version} != pinned {expected_aiperf_version}"
        )
    if metrics.was_cancelled:
        invalid.append("run was cancelled")
    if metrics.output_tps is None:
        invalid.append("no output_token_throughput")
    if metrics.request_count < concurrency:
        invalid.append(
            f"{metrics.request_count} completed requests < {concurrency} lanes"
        )
    if metrics.error_rate >= MAX_ERROR_RATE:
        invalid.append(f"error rate {metrics.error_rate:.2%} >= {MAX_ERROR_RATE:.0%}")
    overflow = sum(
        count
        for label, count in metrics.error_types.items()
        if any(marker in label.lower() for marker in OVERFLOW_MARKERS)
    )
    total = metrics.request_count + metrics.error_count
    if total and overflow / total > MAX_ERROR_RATE:
        invalid.append(f"context overflow {overflow}/{total} > 1% (AgentX invalid)")
    if (
        metrics.duration_s is not None
        and metrics.duration_s < MIN_DURATION_FRACTION * spec.duration_s
    ):
        invalid.append(
            f"measured {metrics.duration_s:.0f}s < {MIN_DURATION_FRACTION:.0%} of {spec.duration_s}s"
        )
    if engagement is not None and not engagement.ok:
        failed = [k for k, ok in engagement.checks.items() if not ok]
        invalid.append(f"config not engaged: {failed}")
    if not server_alive:
        invalid.append("a serving container exited or restarted during the run")
    if nic_restricted and nic_engaged is False:
        invalid.append("NIC restriction not engaged (traffic on non-selected HCAs)")
    if nic_restricted and nic_engaged is None:
        recovered.append("NIC engagement not provable (no cross-node traffic)")
    if nvlink_off and nvlink_p99_gbs is not None and nvlink_p99_gbs > nvlink_leak_gbs:
        invalid.append(
            f"NVLink still carried {nvlink_p99_gbs:.1f} GB/s per GPU with NVLink off"
        )
    if nvlink_off and nvlink_p99_gbs is None:
        recovered.append("NVLink idleness not provable (no NVLink counters)")
    if telemetry_coverage is not None and telemetry_coverage < MIN_TELEMETRY_COVERAGE:
        recovered.append(f"telemetry coverage {telemetry_coverage:.0%}")
    if invalid:
        return AuditResult(status="invalid", reasons=tuple(invalid + recovered))
    if recovered:
        return AuditResult(status="valid_with_recovery", reasons=tuple(recovered))
    return AuditResult(status="valid")


def noise_from_samples(
    cand_key: str, concurrency: int, samples: Sequence[float]
) -> NoiseResult:
    """Pilot noise floor: half-width of a 95% prediction interval for one new
    run vs the pilot mean, ``t(0.975, n-1) * s * sqrt(1 + 1/n)``."""
    if len(samples) < 2:
        raise ValueError("noise floor needs at least two repetitions")
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    n = len(samples)
    t = T_975.get(n - 1, 1.96)
    floor = t * stdev * math.sqrt(1 + 1 / n)
    return NoiseResult(
        cand_key=cand_key,
        concurrency=concurrency,
        samples=tuple(samples),
        mean=mean,
        stdev=stdev,
        cv=stdev / mean if mean else 0.0,
        noise_floor=floor,
        noise_floor_pct=100 * floor / mean if mean else 0.0,
    )


def noise_pct(noise: NoiseResult | None) -> float:
    return noise.noise_floor_pct if noise is not None else PROVISIONAL_NOISE_PCT


def pct_delta(current: float, reference: float) -> float:
    return 100.0 * (current - reference) / reference if reference else math.inf


def valid_points(points: Iterable[PointResult]) -> list[PointResult]:
    return [p for p in points if p.audit.usable and p.tps_per_gpu is not None]


def best_point(points: Iterable[PointResult]) -> PointResult | None:
    usable = valid_points(points)
    return max(usable, key=lambda p: p.tps_per_gpu or 0.0) if usable else None


def curve(points: Iterable[PointResult]) -> list[PointResult]:
    """One valid point per concurrency (latest wins), sorted by concurrency."""
    by_c: dict[int, PointResult] = {}
    for p in valid_points(points):
        by_c[p.key.concurrency] = p
    return [by_c[c] for c in sorted(by_c)]


def saturation(
    points: Sequence[PointResult], noise: NoiseResult | None
) -> tuple[bool, str]:
    """Is the top of the throughput curve flat within the noise floor?"""
    pts = curve(points)
    if len(pts) < 2:
        return False, "fewer than two valid concurrency points"
    best = max(pts, key=lambda p: p.tps_per_gpu or 0.0)
    threshold = noise_pct(noise)
    if best is not pts[-1]:
        nxt = pts[pts.index(best) + 1]
        return True, (
            f"peak at c={best.key.concurrency}; c={nxt.key.concurrency} is "
            f"{pct_delta(nxt.tps_per_gpu or 0, best.tps_per_gpu or 1):+.1f}%"
        )
    prev = pts[-2]
    gain = pct_delta(best.tps_per_gpu or 0, prev.tps_per_gpu or 1)
    if gain <= threshold:
        return True, f"top two points within {threshold:.1f}% ({gain:+.1f}%)"
    return False, f"still rising {gain:+.1f}% at c={best.key.concurrency}"


def floor_result(points: Sequence[PointResult], user_floor: float) -> FloorResult:
    """Best tok/s/GPU among valid points with per-user p50 >= floor, plus a
    log2(concurrency) interpolation at the floor crossing of the best curve."""
    meeting = [
        p
        for p in valid_points(points)
        if (p.metrics and p.metrics.user_tps_p50 or 0) >= user_floor
    ]
    if not meeting:
        return FloorResult(
            user_tps_floor=user_floor,
            crossed=False,
            note="no valid point meets the floor",
        )
    best = max(meeting, key=lambda p: p.tps_per_gpu or 0.0)
    interp = _interpolate_floor(
        [p for p in points if p.key.cand_key == best.key.cand_key], user_floor
    )
    return FloorResult(
        user_tps_floor=user_floor,
        point_id=best.key.id,
        cand_key=best.key.cand_key,
        concurrency=best.key.concurrency,
        tps_per_gpu=best.tps_per_gpu,
        interpolated_tps_per_gpu=interp,
        crossed=True,
        note="interpolated in log2(concurrency) between straddling points"
        if interp
        else "",
    )


def _interpolate_floor(
    points: Sequence[PointResult], user_floor: float
) -> float | None:
    pts = curve(points)
    for lo, hi in zip(pts, pts[1:]):
        u_lo = lo.metrics.user_tps_p50 if lo.metrics else None
        u_hi = hi.metrics.user_tps_p50 if hi.metrics else None
        if u_lo is None or u_hi is None or not (u_lo >= user_floor > u_hi):
            continue
        x_lo, x_hi = math.log2(lo.key.concurrency), math.log2(hi.key.concurrency)
        frac = (u_lo - user_floor) / (u_lo - u_hi)
        x = x_lo + frac * (x_hi - x_lo)
        t_lo, t_hi = lo.tps_per_gpu or 0.0, hi.tps_per_gpu or 0.0
        return t_lo + (x - x_lo) / (x_hi - x_lo) * (t_hi - t_lo)
    return None


def mtp_comparison(
    on: PointResult | None, off: PointResult | None, noise: NoiseResult | None
) -> MtpComparison:
    usable = (
        on is not None
        and off is not None
        and on.audit.usable
        and off.audit.usable
        and on.tps_per_gpu is not None
        and off.tps_per_gpu is not None
    )
    if not usable:
        # An invalid run on either side is evidence of failure, not of a delta.
        return MtpComparison(
            on_point_id=on.key.id if on else None,
            off_point_id=off.key.id if off else None,
            accept_len=on.accept_len if on else None,
        )
    delta = pct_delta(on.tps_per_gpu, off.tps_per_gpu)
    return MtpComparison(
        on_point_id=on.key.id,
        off_point_id=off.key.id,
        on_tps_per_gpu=on.tps_per_gpu,
        off_tps_per_gpu=off.tps_per_gpu,
        delta_pct=delta,
        within_noise=abs(delta) <= noise_pct(noise),
        accept_len=on.accept_len,
    )


def nic_arms(
    arms: Sequence[tuple[int, PointResult | None]], noise: NoiseResult | None
) -> tuple[NicArm, ...]:
    """Compare each NIC count against the full-NIC arm."""
    if not arms:
        return ()
    full_count = max(n for n, _ in arms)
    full = dict(arms).get(full_count)
    full_tps = full.tps_per_gpu if full and full.audit.usable else None
    out: list[NicArm] = []
    for count, point in sorted(arms, key=lambda a: -a[0]):
        if point is None:
            out.append(NicArm(nic_count=count, status="skipped"))
            continue
        status = "measured"
        if not point.audit.usable:
            status = "not_engaged" if point.nic_engaged is False else "invalid"
        delta = None
        within = None
        if full_tps and point.tps_per_gpu is not None and point.audit.usable:
            delta = pct_delta(point.tps_per_gpu, full_tps)
            within = abs(delta) <= noise_pct(noise)
        out.append(
            NicArm(
                nic_count=count,
                point_id=point.key.id,
                tps_per_gpu=point.tps_per_gpu,
                delta_vs_full_pct=delta,
                within_noise=within,
                nic_effective=point.nic_effective,
                engaged=point.nic_engaged,
                staging_suspected=point.staging_suspected,
                status=status,
            )
        )
    return tuple(out)


Symptom = Literal["prefill_bound", "decode_bound", "balanced", "unknown"]
TTFT_BOUND_MS = 5000.0
USER_FLOOR = max(INTERACTIVITY_FLOORS)


def pd_symptom(point: PointResult | None) -> Symptom:
    """Which side of a PD split limits it (agent-docs rate-matching heuristics)."""
    if point is None or point.metrics is None:
        return "unknown"
    ttft = point.metrics.ttft_p50_ms
    user = point.metrics.user_tps_p50
    if ttft is None or user is None:
        return "unknown"
    if ttft > TTFT_BOUND_MS:
        return "prefill_bound"
    if user < USER_FLOOR:
        return "decode_bound"
    return "balanced"
