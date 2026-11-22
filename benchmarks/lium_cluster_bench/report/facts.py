# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure derivations from an ``ExperimentSummary``, shared by report and KB.

The summary written by the finalize phase is the single source of truth:
nothing here re-decides a leader, a floor or a noise verdict. These helpers
only regroup and look up what the summary already holds, and every one of
them tolerates missing data (no leader, no points, no noise pilot).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from benchmarks.lium_cluster_bench.domain.schema import (
    CROSS_NODE_MIN_GBS,
    BandwidthStat,
    Candidate,
    ExperimentSummary,
    FamilySummary,
    LaunchPlan,
    NicArm,
    PointResult,
)
from benchmarks.lium_cluster_bench.report.fmt import pct_delta

# Mean inter-node IB traffic below this (GB/s per node) is control noise, not
# a data plane: such a topology cannot be sensitive to the NIC count.
NEGLIGIBLE_CROSS_NODE_GBS = CROSS_NODE_MIN_GBS  # same rule as the planner

PLANE_ORDER = ("ib", "nvlink", "pcie", "pcie_nic")
DIRECTION_ORDER = ("tx", "rx", "bidir")
PURPOSE_ORDER = ("search", "mtp_ablation", "pilot", "nic", "confirm", "smoke")
AUDIT_ORDER = ("valid", "valid_with_recovery", "invalid")
SPEC_ORDER = ("smoke", "screen", "decision")
METHOD_RANK = {"measured": 2, "derived": 1, "unavailable": 0}


def _order(value: str, order: Sequence[str]) -> tuple[int, str]:
    return (order.index(value) if value in order else len(order), value)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are not ``None``, or ``None``."""
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def family_titles(summary: ExperimentSummary) -> dict[str, str]:
    """Family id -> title."""
    return {f.family_id: f.title for f in summary.families}


def family_rank(summary: ExperimentSummary) -> dict[str, int]:
    """Family id -> position in the summary."""
    return {f.family_id: i for i, f in enumerate(summary.families)}


def family_summary(
    summary: ExperimentSummary, family_id: str | None
) -> FamilySummary | None:
    """The ``FamilySummary`` of ``family_id``, if any."""
    for family in summary.families:
        if family.family_id == family_id:
            return family
    return None


def is_measured(point: PointResult) -> bool:
    """Usable audit and a throughput number."""
    return point.audit.usable and point.tps_per_gpu is not None


def error_rate(point: PointResult) -> float | None:
    """Error rate of a point, or ``None`` without metrics."""
    return point.metrics.error_rate if point.metrics is not None else None


def point_sort_key(summary: ExperimentSummary, point: PointResult) -> tuple:
    """Display order: family, purpose, candidate, concurrency, rep."""
    rank = family_rank(summary)
    family = point.candidate.family_id
    return (
        rank.get(family, len(rank)),
        family,
        _order(point.key.purpose, PURPOSE_ORDER),
        point.key.cand_key,
        point.key.concurrency,
        point.key.rep,
        point.key.id,
    )


def goodput_leaders(
    summary: ExperimentSummary, purposes: Sequence[str] = ("search", "mtp_ablation")
) -> dict[str, PointResult]:
    """Per SLO tier, the measured point with the most good output tok/s/GPU."""
    best: dict[str, tuple[float, PointResult]] = {}
    for point in summary.points:
        if not is_measured(point) or point.key.purpose not in purposes:
            continue
        for tier in point.requests.tiers if point.requests else ():
            value = tier.good_output_tps_per_gpu
            if value is None or value <= 0:
                continue
            if tier.name not in best or value > best[tier.name][0]:
                best[tier.name] = (value, point)
    return {name: point for name, (_, point) in best.items()}


def leader_point(summary: ExperimentSummary) -> PointResult | None:
    """The leader's measured point, if it exists."""
    return summary.point(summary.leader_point_id)


def leader_family_id(summary: ExperimentSummary) -> str | None:
    """Family of the leader (from the candidate or its point)."""
    if summary.leader is not None:
        return summary.leader.family_id
    point = leader_point(summary)
    return point.candidate.family_id if point is not None else None


def leader_candidate(summary: ExperimentSummary) -> Candidate | None:
    """The leader candidate (from the summary or its point)."""
    if summary.leader is not None:
        return summary.leader
    point = leader_point(summary)
    return point.candidate if point is not None else None


def within_noise(summary: ExperimentSummary, delta_pct: float | None) -> bool | None:
    """``|delta| <= noise floor`` when both are known."""
    if delta_pct is None or summary.noise is None:
        return None
    return abs(delta_pct) <= summary.noise.noise_floor_pct


def leader_cross_node(
    summary: ExperimentSummary, plan: LaunchPlan | None = None
) -> bool | None:
    """Whether the leader moves data-plane traffic between nodes.

    The rendered plan is authoritative (its declared cross-node classes);
    otherwise the measured inter-node IB rate of the leader decides.
    """
    if plan is not None:
        return bool(plan.cross_node)
    point = leader_point(summary)
    rate = point.cross_node_gbs if point is not None else None
    if rate is None:
        family = family_summary(summary, leader_family_id(summary))
        rate = family.cross_node_gbs if family is not None else None
    if rate is None:
        return None
    return rate >= NEGLIGIBLE_CROSS_NODE_GBS


def runner_up(summary: ExperimentSummary) -> FamilySummary | None:
    """Best measured family other than the leader's."""
    lead = leader_family_id(summary)
    best: FamilySummary | None = None
    for family in summary.families:
        if family.family_id == lead or family.best_tps_per_gpu is None:
            continue
        if best is None or family.best_tps_per_gpu > (best.best_tps_per_gpu or 0.0):
            best = family
    return best


def confirm_delta(
    summary: ExperimentSummary,
) -> tuple[PointResult | None, float | None]:
    """The confirmatory rerun and its delta (percent) vs the leader point."""
    confirm = summary.point(summary.confirm_point_id)
    lead = leader_point(summary)
    if confirm is None or lead is None:
        return confirm, None
    return confirm, pct_delta(confirm.tps_per_gpu, lead.tps_per_gpu)


# --------------------------------------------------------------------------
# Concurrency curves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CurvePoint:
    """One concurrency of a curve (repetitions averaged)."""

    concurrency: int
    tps_per_gpu: float
    user_tps_p50: float | None
    ttft_p50_ms: float | None
    itl_p50_ms: float | None
    error_rate: float | None
    cross_node_gbs: float | None
    point_ids: tuple[str, ...]

    @property
    def reps(self) -> int:
        return len(self.point_ids)


@dataclass(frozen=True)
class Curve:
    """The search curve of one candidate."""

    family_id: str
    candidate: Candidate
    points: tuple[CurvePoint, ...]


def family_curves(summary: ExperimentSummary) -> list[Curve]:
    """Usable search points grouped per candidate; repetitions are averaged."""
    groups: dict[str, list[PointResult]] = {}
    for point in summary.points:
        if point.key.purpose == "search" and is_measured(point):
            groups.setdefault(point.key.cand_key, []).append(point)
    rank = family_rank(summary)
    curves: list[Curve] = []
    for points in groups.values():
        by_c: dict[int, list[PointResult]] = {}
        for point in points:
            by_c.setdefault(point.key.concurrency, []).append(point)
        curve_points = []
        for concurrency in sorted(by_c):
            reps = sorted(by_c[concurrency], key=lambda p: (p.key.rep, p.key.id))
            curve_points.append(
                CurvePoint(
                    concurrency=concurrency,
                    tps_per_gpu=statistics.fmean(p.tps_per_gpu or 0.0 for p in reps),
                    user_tps_p50=mean_or_none(
                        p.metrics.user_tps_p50 if p.metrics else None for p in reps
                    ),
                    ttft_p50_ms=mean_or_none(
                        p.metrics.ttft_p50_ms if p.metrics else None for p in reps
                    ),
                    itl_p50_ms=mean_or_none(
                        p.metrics.itl_p50_ms if p.metrics else None for p in reps
                    ),
                    error_rate=mean_or_none(error_rate(p) for p in reps),
                    cross_node_gbs=mean_or_none(p.cross_node_gbs for p in reps),
                    point_ids=tuple(p.key.id for p in reps),
                )
            )
        candidate = points[0].candidate
        curves.append(Curve(candidate.family_id, candidate, tuple(curve_points)))
    curves.sort(
        key=lambda c: (rank.get(c.family_id, len(rank)), c.family_id, c.candidate.key)
    )
    return curves


# --------------------------------------------------------------------------
# MTP on/off pairs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MtpPair:
    """MTP on/off throughput under the same family, concurrency and spec."""

    family_id: str
    concurrency: int
    spec: str
    on_tps: float
    off_tps: float
    on_ids: tuple[str, ...]
    off_ids: tuple[str, ...]
    accept_len: float | None

    @property
    def delta_pct(self) -> float | None:
        return pct_delta(self.on_tps, self.off_tps)


def mtp_pairs(summary: ExperimentSummary) -> list[MtpPair]:
    """MTP on/off points measured under the same family, concurrency and spec."""
    on: dict[tuple, list[PointResult]] = {}
    off: dict[tuple, list[PointResult]] = {}
    for point in summary.points:
        if point.key.purpose == "smoke" or not is_measured(point):
            continue
        cand = point.candidate
        key = (
            cand.family_id,
            cand.nic_count,
            cand.simulate_acceptance,
            point.key.concurrency,
            point.key.spec,
        )
        (on if cand.mtp else off).setdefault(key, []).append(point)
    rank = family_rank(summary)
    pairs = []
    for key, off_points in off.items():
        on_points = on.get(key)
        if not on_points or key[1] is not None or key[2]:
            continue
        pairs.append(
            MtpPair(
                family_id=key[0],
                concurrency=key[3],
                spec=key[4],
                on_tps=statistics.fmean(p.tps_per_gpu or 0.0 for p in on_points),
                off_tps=statistics.fmean(p.tps_per_gpu or 0.0 for p in off_points),
                on_ids=tuple(sorted(p.key.id for p in on_points)),
                off_ids=tuple(sorted(p.key.id for p in off_points)),
                accept_len=mean_or_none(p.accept_len for p in on_points),
            )
        )
    pairs.sort(
        key=lambda p: (
            rank.get(p.family_id, len(rank)),
            p.family_id,
            p.concurrency,
            p.spec,
        )
    )
    return pairs


# --------------------------------------------------------------------------
# Audit and spec bookkeeping
# --------------------------------------------------------------------------


def audit_counts(points: Iterable[PointResult]) -> dict[str, int]:
    """Point count per audit status."""
    counts = Counter(p.audit.status for p in points)
    return {status: counts.get(status, 0) for status in AUDIT_ORDER}


@dataclass(frozen=True)
class SpecStat:
    """Point and audit counts of one bench spec."""

    spec: str
    total: int
    valid: int
    recovered: int
    invalid: int
    median_duration_s: float | None


def spec_stats(summary: ExperimentSummary) -> list[SpecStat]:
    """Per-spec point counts, audit outcomes and median duration."""
    by_spec: dict[str, list[PointResult]] = {}
    for point in summary.points:
        by_spec.setdefault(point.key.spec, []).append(point)
    stats = []
    for spec in sorted(by_spec, key=lambda s: _order(s, SPEC_ORDER)):
        points = by_spec[spec]
        counts = audit_counts(points)
        durations = [
            p.metrics.duration_s
            for p in points
            if p.metrics is not None and p.metrics.duration_s is not None
        ]
        stats.append(
            SpecStat(
                spec=spec,
                total=len(points),
                valid=counts["valid"],
                recovered=counts["valid_with_recovery"],
                invalid=counts["invalid"],
                median_duration_s=statistics.median(durations) if durations else None,
            )
        )
    return stats


# --------------------------------------------------------------------------
# Bandwidth and NIC arms
# --------------------------------------------------------------------------


def sort_stats(stats: Iterable[BandwidthStat]) -> list[BandwidthStat]:
    """Bandwidth stats in plane, scope, direction order."""
    return sorted(
        stats,
        key=lambda s: (
            _order(s.plane, PLANE_ORDER),
            s.scope,
            _order(s.direction, DIRECTION_ORDER),
        ),
    )


def busiest_stats(stats: Iterable[BandwidthStat]) -> list[BandwidthStat]:
    """One stat per (plane, direction): the busiest device or link.

    Measured beats derived beats unavailable; then the highest utilization
    vs the speed-of-light reference; then the highest peak.
    """
    best: dict[tuple[str, str], BandwidthStat] = {}

    def score(stat: BandwidthStat) -> tuple[int, float, float]:
        util = stat.util_vs_peak
        return (
            METHOD_RANK.get(stat.method, 0),
            util if util is not None else -1.0,
            stat.peak_gbs,
        )

    for stat in sort_stats(stats):
        key = (stat.plane, stat.direction)
        if key not in best or score(stat) > score(best[key]):
            best[key] = stat
    return [
        best[key]
        for key in sorted(
            best,
            key=lambda k: (_order(k[0], PLANE_ORDER), _order(k[1], DIRECTION_ORDER)),
        )
    ]


def nic_arms_sorted(arms: Sequence[NicArm]) -> list[NicArm]:
    """NIC arms, most NICs first."""
    return sorted(arms, key=lambda a: -a.nic_count)


def full_nic_arm(arms: Sequence[NicArm]) -> NicArm | None:
    """The reference arm: the most NICs, when it produced a number.

    Mirrors the analysis that computed ``delta_vs_full_pct``: when the arm
    with the most NICs has no throughput there is no reference at all.
    """
    if not arms:
        return None
    full = max(arms, key=lambda a: a.nic_count)
    return full if full.tps_per_gpu is not None else None
