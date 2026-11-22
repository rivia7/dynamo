# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throughput-first search planner.

Every ``decide_*`` function is pure: (context, history, budget) -> the next
action. The pipeline executes the action, appends its result to the event
log and asks again, so resuming after a crash is a replay of the log.

Search policy (agent-docs/rules/benchmarking + optimize-loop):
* families run in priority order; the recipe baseline first;
* per family a bounded powers-of-two concurrency climb that stops at the
  first gain inside the noise floor, an invalid point, a per-user rate
  below the climb floor, or the grid maximum;
* families whose best is below ``prune_ratio`` x the leader after two
  points are pruned (a wide margin is valid at n=1);
* families sharing a ``group`` (P:D ratios) are alternatives: after the
  primary, at most one alternate is chosen by the primary's symptom;
* deploy failures walk the profile's ladder for that failure class;
* after the search: interactivity-floor bracketing, an MTP-off ablation,
  an accuracy gate on the leader (a failed config is never the leader; the
  next best is checked, up to a limit), an n=3 pilot at the leader's knee,
  the NIC sweep on the best config with measured cross-node traffic, a
  decision-length confirmatory rerun, and the leader re-measured with
  NVLink switched off (GPU-GPU over PCIe; domain/nvlink.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence, Union

from benchmarks.cluster_bench.domain.analysis import (
    best_point,
    curve,
    noise_pct,
    pct_delta,
    pd_symptom,
)
from benchmarks.cluster_bench.domain.nvlink import NVLINK_ONLY_TRANSPORTS
from benchmarks.cluster_bench.domain.nvlink import TOGGLE_S as NVLINK_TOGGLE_S
from benchmarks.cluster_bench.domain.schema import (
    CROSS_NODE_MIN_GBS,
    INTERACTIVITY_FLOORS,
    Candidate,
    DeployResult,
    FamilySpec,
    FamilyStatus,
    GridSpec,
    LadderRung,
    NoiseResult,
    PointKey,
    PointResult,
    Purpose,
    Stage,
)

MAX_FLOOR_DESCENTS = 3
MAX_INVALID_FIRST_POINTS = 2


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPoint:
    candidate: Candidate
    concurrency: int
    spec: str
    purpose: Purpose
    rep: int = 0
    reason: str = ""

    @property
    def key(self) -> PointKey:
        return PointKey(
            cand_key=self.candidate.key,
            concurrency=self.concurrency,
            spec=self.spec,
            purpose=self.purpose,
            rep=self.rep,
        )


@dataclass(frozen=True)
class CloseFamily:
    family_id: str
    status: FamilyStatus
    reason: str


@dataclass(frozen=True)
class Done:
    reason: str


@dataclass(frozen=True)
class CheckAccuracy:
    """Deploy ``candidate`` (if needed) and run the accuracy probe on it."""

    candidate: Candidate
    reason: str = ""

    @property
    def key(self) -> str:
        return f"accuracy:{self.candidate.key}"


Action = Union[RunPoint, CloseFamily, Done, CheckAccuracy]
AccuracyStatus = str  # "pass" | "fail" | "error" | "skipped" (domain/accuracy.py)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    remaining_s: float
    deploy_s: float
    point_s: Mapping[str, float]
    reserve_s: float = 0.0
    # Part of ``reserve_s`` a competitive family's climb may spend: the
    # NVLink comparison yields to finding the right leader.
    soft_reserve_s: float = 0.0

    def affordable(self, cost_s: float) -> bool:
        return self.remaining_s - self.reserve_s >= cost_s

    def without_soft_reserve(self) -> Budget:
        return replace(self, reserve_s=max(0.0, self.reserve_s - self.soft_reserve_s))


@dataclass(frozen=True)
class History:
    points: tuple[PointResult, ...] = ()
    deploys: tuple[DeployResult, ...] = ()
    closed: Mapping[str, tuple[FamilyStatus, str]] = field(default_factory=dict)
    noise: NoiseResult | None = None
    deployed_key: str | None = None  # candidate currently running, if any
    # cand_key -> accuracy verdict status, in check order (ACCURACY events)
    accuracy: Mapping[str, AccuracyStatus] = field(default_factory=dict)

    def accuracy_failed(self, cand_key: str) -> bool:
        return self.accuracy.get(cand_key) == "fail"

    def points_for(
        self, cand_key: str, purposes: Sequence[str] | None = None
    ) -> list[PointResult]:
        return [
            p
            for p in self.points
            if p.key.cand_key == cand_key
            and (purposes is None or p.key.purpose in purposes)
        ]

    def deploys_for(self, cand_key: str) -> list[DeployResult]:
        return [d for d in self.deploys if d.candidate.key == cand_key]

    def deploy_failed(self, cand_key: str) -> bool:
        attempts = self.deploys_for(cand_key)
        return bool(attempts) and not attempts[-1].ok


@dataclass(frozen=True)
class SearchContext:
    families: tuple[FamilySpec, ...]  # enabled, fitted, priority-sorted
    ladder: tuple[LadderRung, ...]
    grids: Mapping[str, GridSpec]
    simulate_acceptance: bool = False
    mtp_capable: bool = True
    floors: tuple[float, ...] = INTERACTIVITY_FLOORS
    prune_ratio: float = 0.7
    climb_min_user_tps: float = 10.0
    nic_counts: tuple[int, ...] = (8, 4, 2, 1)  # feasible counts, descending
    nvlink: bool = False  # GPUs have NVLink and the NVLink comparison is on
    # e2e, or the one PD stage every family runs (domain/stages.py). P-Only
    # has no per-user decode rate, so no interactivity floors.
    stage: Stage = "e2e"


def deploy_cost(budget: Budget, history: History, cand: Candidate) -> float:
    return 0.0 if history.deployed_key == cand.key else budget.deploy_s


# --------------------------------------------------------------------------
# Candidates and the ladder
# --------------------------------------------------------------------------


def _inherited_kernel_rungs(ctx: SearchContext, history: History) -> tuple[str, ...]:
    """Kernel fallbacks found by the smoke test or an earlier family apply
    to every family on this GPU type."""
    kernel_ids = {r.id for r in ctx.ladder if r.when == "kernel_unsupported"}
    for deploy in history.deploys:
        if deploy.ok:
            return tuple(r for r in deploy.candidate.rungs if r in kernel_ids)
    return ()


def search_candidate(
    ctx: SearchContext, history: History, family: FamilySpec
) -> Candidate | CloseFamily:
    """The candidate to measure for ``family``, walking the ladder on failures."""
    base = Candidate(
        family_id=family.id,
        simulate_acceptance=ctx.simulate_acceptance,
        rungs=_inherited_kernel_rungs(ctx, history),
    )
    attempts = [
        d
        for d in history.deploys
        if d.candidate.family_id == family.id
        and d.candidate.mtp
        and d.candidate.nic_count is None
    ]
    if not attempts:
        return base
    last = attempts[-1]
    if last.ok:
        return last.candidate
    roles = {r.role for r in family.roles}
    remaining = [
        r
        for r in ctx.ladder
        if r.when == last.failure
        and r.id not in last.candidate.rungs
        and roles & set(r.roles)
    ]
    if remaining:
        return last.candidate.with_rung(remaining[0].id)
    status: FamilyStatus = (
        "unsupported" if last.failure == "kernel_unsupported" else "deploy_failed"
    )
    return CloseFamily(family.id, status, f"{last.failure}: {last.detail[:300]}")


def leader_point(
    history: History, purposes: Sequence[str] = ("search", "mtp_ablation")
) -> PointResult | None:
    """Best screen point; configurations that failed the accuracy gate never lead."""
    return best_point(
        p
        for p in history.points
        if p.key.purpose in purposes
        and p.key.spec == "screen"
        and not history.accuracy_failed(p.key.cand_key)
    )


# --------------------------------------------------------------------------
# Search phase
# --------------------------------------------------------------------------


def _prefill_fraction(family: FamilySpec) -> float:
    total = family.total_gpus or 1
    return sum(r.total_gpus for r in family.roles if r.role == "prefill") / total


def _choose_alternate(
    ctx: SearchContext, history: History, group: list[FamilySpec]
) -> str | None:
    primary, group_alternates = group[0], group[1:]
    measured = [
        f
        for f in group_alternates
        if f.id in history.closed and history.closed[f.id][0] == "measured"
    ]
    if measured:
        return measured[0].id  # already committed to one alternate
    # An alternate that could not be deployed was not a real try: choose again
    # among the others.
    alternates = [
        f
        for f in group_alternates
        if history.closed.get(f.id, ("", ""))[0] not in ("unsupported", "deploy_failed")
    ]
    primary_best = None
    for d in history.deploys:
        if d.candidate.family_id == primary.id and d.ok:
            primary_best = (
                best_point(history.points_for(d.candidate.key, ("search",)))
                or primary_best
            )
    if primary_best is None:
        # The primary never produced a valid point: measure the first alternate.
        return alternates[0].id if alternates else None
    symptom = pd_symptom(primary_best)
    base = _prefill_fraction(primary)
    if symptom == "prefill_bound":
        more = [f for f in alternates if _prefill_fraction(f) > base]
        return min(more, key=_prefill_fraction).id if more else None
    if symptom == "decode_bound":
        less = [f for f in alternates if _prefill_fraction(f) < base]
        return max(less, key=_prefill_fraction).id if less else None
    return None


def decide_search(ctx: SearchContext, history: History, budget: Budget) -> Action:
    leader = leader_point(history)
    for family in ctx.families:
        if family.id in history.closed:
            continue
        if family.group:
            group = [f for f in ctx.families if f.group == family.group]
            if family is not group[0]:
                chosen = _choose_alternate(ctx, history, group)
                if family.id != chosen:
                    primary = group[0]
                    symptom = pd_symptom(
                        best_point(
                            p
                            for p in history.points
                            if p.candidate.family_id == primary.id
                            and p.key.purpose == "search"
                        )
                    )
                    return CloseFamily(
                        family.id,
                        "pruned",
                        f"alternate not selected (primary {primary.id}: {symptom})",
                    )
        return _decide_family(ctx, history, budget, family, leader)
    return Done("all families evaluated")


def _decide_family(
    ctx: SearchContext,
    history: History,
    budget: Budget,
    family: FamilySpec,
    leader: PointResult | None,
) -> Action:
    if ctx.families and family.id == ctx.families[0].id:
        # The baseline is the minimum evidence of a run: it may spend budget
        # reserved for later steps, which then stop on their own budget checks.
        budget = replace(budget, reserve_s=0.0)
    cand = search_candidate(ctx, history, family)
    if isinstance(cand, CloseFamily):
        return cand
    grid = ctx.grids[family.id]
    measured = sorted(
        history.points_for(cand.key, ("search",)), key=lambda p: p.started_at
    )
    screen_s = budget.point_s["screen"]
    if not measured:
        if not budget.affordable(deploy_cost(budget, history, cand) + screen_s):
            return CloseFamily(
                family.id,
                "skipped_budget",
                "insufficient budget for deploy + first point",
            )
        return RunPoint(cand, grid.start, "screen", "search", reason="first point")

    valid = curve(measured)
    last = measured[-1]
    if not valid:
        invalid_count = len(measured)
        lower = last.key.concurrency // 2
        if (
            invalid_count < MAX_INVALID_FIRST_POINTS
            and lower >= grid.min
            and budget.affordable(screen_s)
        ):
            return RunPoint(
                cand, lower, "screen", "search", reason="no valid point yet; descend"
            )
        return CloseFamily(
            family.id,
            "pruned",
            f"no valid measurement ({'; '.join(last.audit.reasons)[:200]})",
        )

    best = max(valid, key=lambda p: p.tps_per_gpu or 0.0)
    if leader is not None and leader.key.cand_key != cand.key and len(valid) >= 2:
        # Compare at equal load: a family still climbing must not be pruned
        # against the leader's knee at a much higher concurrency.
        reference = _leader_tps_at(history, leader, best.key.concurrency)
        if (best.tps_per_gpu or 0.0) < ctx.prune_ratio * reference:
            return CloseFamily(
                family.id,
                "pruned",
                f"best {best.tps_per_gpu:.1f} < {ctx.prune_ratio} x leader {reference:.1f} tok/s/GPU",
            )

    top = valid[-1]
    frontier = last.key.concurrency == top.key.concurrency and last.audit.usable
    if frontier:
        prev = valid[-2] if len(valid) >= 2 else None
        gain = (
            pct_delta(top.tps_per_gpu or 0.0, prev.tps_per_gpu or 1.0)
            if prev
            else math.inf
        )
        user = top.metrics.user_tps_p50 if top.metrics else None
        nxt = top.key.concurrency * 2
        rising = gain > noise_pct(history.noise)
        if any(p.key.concurrency == nxt for p in measured):
            # Already measured (e.g. invalid before a descent): do not repeat.
            return CloseFamily(family.id, "measured", "climb complete")
        if (
            rising
            and nxt <= grid.max
            and (user is None or user >= ctx.climb_min_user_tps)
        ):
            climb = budget
            if _competitive(history, leader, cand, best):
                climb = budget.without_soft_reserve()
            if climb.affordable(screen_s):
                return RunPoint(
                    cand,
                    nxt,
                    "screen",
                    "search",
                    reason=f"climb ({gain:+.1f}% at c={top.key.concurrency})",
                )
            return CloseFamily(
                family.id, "measured", "budget stopped the climb before saturation"
            )
        if rising and nxt > grid.max:
            return CloseFamily(
                family.id, "measured", f"grid max {grid.max} reached while still rising"
            )
    return CloseFamily(family.id, "measured", "climb complete")


def _competitive(
    history: History,
    leader: PointResult | None,
    cand: Candidate,
    best: PointResult,
) -> bool:
    """The family leads, or matches the leader at equal load: finishing its
    climb decides the leader, which outranks the soft-reserved steps."""
    if leader is None or leader.key.cand_key == cand.key:
        return True
    reference = _leader_tps_at(history, leader, best.key.concurrency)
    return (best.tps_per_gpu or 0.0) >= reference


def _leader_tps_at(history: History, leader: PointResult, concurrency: int) -> float:
    """The leader's tok/s/GPU at (or just below) ``concurrency``; its best otherwise."""
    pts = [
        p
        for p in curve(history.points_for(leader.key.cand_key, ("search",)))
        if p.key.concurrency <= concurrency
    ]
    if pts:
        return pts[-1].tps_per_gpu or 0.0
    return leader.tps_per_gpu or 0.0


def decide_floors(ctx: SearchContext, history: History, budget: Budget) -> Action:
    """Bracket the interactivity floors on the leader's curve by descending."""
    if ctx.stage == "prefill":
        return Done("interactivity floors do not apply to P-Only (one output token)")
    leader = leader_point(history, ("search",))
    if leader is None:
        return Done("no valid leader")
    cand = leader.candidate
    if history.deploy_failed(cand.key):
        return Done("leader redeploy failed")
    grid = ctx.grids.get(cand.family_id)
    pts = curve(history.points_for(cand.key, ("search",)))
    lowest = pts[0]
    descents = (
        sum(
            1
            for p in history.points_for(cand.key, ("search",))
            if p.key.concurrency < grid.start
        )
        if grid
        else 0
    )
    top_floor = max(ctx.floors)
    user = lowest.metrics.user_tps_p50 if lowest.metrics else None
    if grid is None or user is None or user >= top_floor:
        return Done("floors bracketed")
    nxt = lowest.key.concurrency // 2
    if any(p.key.concurrency == nxt for p in history.points_for(cand.key, ("search",))):
        return Done(f"c={nxt} already measured without a valid result")
    if nxt < grid.min or descents >= MAX_FLOOR_DESCENTS:
        return Done(f"floor {top_floor} not reached down to c={lowest.key.concurrency}")
    if not budget.affordable(
        deploy_cost(budget, history, cand) + budget.point_s["screen"]
    ):
        return Done("budget stopped floor bracketing")
    return RunPoint(
        cand,
        nxt,
        "screen",
        "search",
        reason=f"bracket the {top_floor:g} tok/s/user floor",
    )


def mtp_disabled_by(ctx: SearchContext, cand: Candidate) -> str | None:
    """The applied ladder rung that already removed speculative decoding."""
    for rung in ctx.ladder:
        if rung.id in cand.rungs and "speculative-algorithm" in rung.sglang:
            if rung.sglang["speculative-algorithm"] is None:
                return rung.id
    return None


def decide_mtp(ctx: SearchContext, history: History, budget: Budget) -> Action:
    leader = leader_point(history, ("search",))
    if not ctx.mtp_capable or leader is None or not leader.candidate.mtp:
        return Done("MTP ablation not applicable")
    disabled = mtp_disabled_by(ctx, leader.candidate)
    if disabled is not None:
        # "On" and "off" would render the same config; measuring it twice
        # costs a redeploy and reports noise as an MTP effect.
        return Done(f"MTP already disabled by fallback rung {disabled}")
    off = leader.candidate.model_copy(update={"mtp": False})
    if history.deploy_failed(off.key):
        return Done("MTP-off deployment failed")
    if history.points_for(off.key, ("mtp_ablation",)):
        return Done("MTP ablation measured")
    if not budget.affordable(
        deploy_cost(budget, history, off) + budget.point_s["screen"]
    ):
        return Done("budget skipped the MTP ablation")
    return RunPoint(
        off,
        leader.key.concurrency,
        "screen",
        "mtp_ablation",
        reason="MTP off at the leader's knee",
    )


# --------------------------------------------------------------------------
# Pilot, NIC sweep, confirm
# --------------------------------------------------------------------------


def final_leader(history: History) -> PointResult | None:
    return leader_point(history, ("search", "mtp_ablation"))


# Seconds one accuracy check takes (dataset fetch + probe; the probe's own
# deadline is 900 s); used for the budget gate before a check.
ACCURACY_CHECK_S = 600.0
PILOT_REPS = 3  # samples at the leader's knee, the search point included


def decide_accuracy(
    ctx: SearchContext, history: History, budget: Budget, max_checks: int = 3
) -> Action:
    """Check the leader; a failure makes the next best the leader and so on.

    Stops at the first pass, at an ``error``/``skipped`` verdict (an
    infrastructure problem repeats for every config: the leader stays
    unchecked and the report says so), after ``max_checks`` checks, or when
    the budget cannot cover a (re)deploy plus a check.
    """
    leader = final_leader(history)
    if leader is None:
        return Done("no leader to check for accuracy")
    status = history.accuracy.get(leader.candidate.key)
    if status == "pass":
        return Done("the leader passed the accuracy check")
    if status is not None:
        return Done(f"accuracy check {status}: the leader stays unchecked")
    if len(history.accuracy) >= max_checks:
        return Done(f"{max_checks} accuracy checks used; the leader stays unchecked")
    if history.deploy_failed(leader.candidate.key):
        return Done("leader redeploy failed before the accuracy check")
    if not budget.affordable(
        deploy_cost(budget, history, leader.candidate) + ACCURACY_CHECK_S
    ):
        return Done("budget skipped the accuracy check")
    failed = [k for k, s in history.accuracy.items() if s == "fail"]
    reason = (
        f"accuracy gate on the leader {leader.candidate.family_id} "
        f"(after {len(failed)} failed config(s))"
        if failed
        else f"accuracy gate on the leader {leader.candidate.family_id}"
    )
    return CheckAccuracy(leader.candidate, reason=reason)


def decide_pilot(
    ctx: SearchContext, history: History, budget: Budget, reps: int = PILOT_REPS
) -> Action:
    leader = final_leader(history)
    if leader is None:
        return Done("no valid leader for the pilot")
    if history.noise is not None:
        return Done("noise floor measured")
    if history.deploy_failed(leader.candidate.key):
        return Done("leader redeploy failed")
    samples = pilot_samples(history, leader)
    if samples >= reps:
        return Done("pilot complete")
    if not budget.affordable(
        deploy_cost(budget, history, leader.candidate) + budget.point_s["screen"]
    ):
        return Done("budget stopped the pilot")
    return RunPoint(
        leader.candidate,
        leader.key.concurrency,
        "screen",
        "pilot",
        rep=samples,
        reason=f"noise pilot rep {samples + 1}/{reps}",
    )


def nic_base(history: History) -> tuple[PointResult | None, str]:
    """Best measured config that actually moves data between nodes."""
    leader = final_leader(history)
    if leader is not None and (leader.cross_node_gbs or 0.0) >= CROSS_NODE_MIN_GBS:
        return leader, "the throughput leader has cross-node data-plane traffic"
    by_cand: dict[str, list[PointResult]] = {}
    for p in history.points:
        if (
            p.key.purpose == "search"
            and p.candidate.nic_count is None
            and not history.accuracy_failed(p.key.cand_key)
        ):
            by_cand.setdefault(p.key.cand_key, []).append(p)
    cross = []
    for pts in by_cand.values():
        best = best_point(pts)
        if best is not None and (best.cross_node_gbs or 0.0) >= CROSS_NODE_MIN_GBS:
            cross.append(best)
    if not cross:
        return (
            None,
            "no measured configuration has cross-node data-plane traffic; NIC count cannot matter",
        )
    chosen = max(cross, key=lambda p: p.tps_per_gpu or 0.0)
    return chosen, (
        "the throughput leader has no cross-node data-plane traffic, so NIC count is irrelevant for it; "
        f"the sweep uses the best cross-node configuration ({chosen.candidate.family_id})"
    )


def nic_arm_order(counts: Sequence[int]) -> list[int]:
    """Full count first (often already measured), then extremes before the middle."""
    ordered = sorted(set(counts), reverse=True)
    if len(ordered) <= 2:
        return ordered
    full, rest = ordered[0], ordered[1:]
    return [full, rest[-1], *rest[:-1]]


def nic_arm_candidate(base: Candidate, count: int, full: int) -> Candidate:
    return base if count == full else base.model_copy(update={"nic_count": count})


def decide_nic(ctx: SearchContext, history: History, budget: Budget) -> Action:
    base, reason = nic_base(history)
    if base is None:
        return Done(reason)
    knee = base.key.concurrency
    full = max(ctx.nic_counts)
    for count in nic_arm_order(ctx.nic_counts):
        cand = nic_arm_candidate(base.candidate, count, full)
        purposes = ("search", "nic", "mtp_ablation") if count == full else ("nic",)
        done = [
            p
            for p in history.points_for(cand.key, purposes)
            if p.key.concurrency == knee
        ]
        if done or history.deploy_failed(cand.key):
            continue
        if not budget.affordable(
            deploy_cost(budget, history, cand) + budget.point_s["screen"]
        ):
            return Done(f"budget stopped the NIC sweep before N={count}")
        return RunPoint(cand, knee, "screen", "nic", reason=f"NIC sweep N={count}")
    return Done("NIC sweep complete")


def decide_confirm(ctx: SearchContext, history: History, budget: Budget) -> Action:
    leader = final_leader(history)
    if leader is None:
        return Done("nothing to confirm")
    if history.points_for(leader.candidate.key, ("confirm",)) or history.deploy_failed(
        leader.candidate.key
    ):
        return Done("confirmation done")
    if not budget.affordable(
        deploy_cost(budget, history, leader.candidate) + budget.point_s["decision"]
    ):
        return Done("budget skipped the confirmatory rerun")
    return RunPoint(
        leader.candidate,
        leader.key.concurrency,
        "decision",
        "confirm",
        reason="decision-length confirmatory rerun",
    )


# --------------------------------------------------------------------------
# NVLink off vs on
# --------------------------------------------------------------------------


def _family_transport(ctx: SearchContext, family_id: str) -> str | None:
    return next((f.transport for f in ctx.families if f.id == family_id), None)


def nvlink_base(ctx: SearchContext, history: History) -> tuple[PointResult | None, str]:
    """The point the NVLink-off run repeats, and why that one.

    The final leader at its concurrency, unless its family needs NVLink-only
    kernels (DeepEP intranode); then the best measured family without them.
    """
    if not ctx.nvlink:
        return None, "NVLink comparison not applicable"
    leader = final_leader(history)
    if leader is None:
        return None, "no leader to compare"
    if _family_transport(ctx, leader.candidate.family_id) not in NVLINK_ONLY_TRANSPORTS:
        return leader, "the throughput leader"
    others: dict[str, PointResult] = {}
    for p in history.points:
        family = p.candidate.family_id
        if (
            p.key.purpose != "search"
            or not p.audit.usable
            or p.candidate.p2p != "nvlink"
            or _family_transport(ctx, family) in NVLINK_ONLY_TRANSPORTS
            or history.accuracy_failed(p.candidate.key)
        ):
            continue
        if family not in others or (p.tps_per_gpu or 0.0) > (
            others[family].tps_per_gpu or 0.0
        ):
            others[family] = p
    if not others:
        return None, (
            f"the leader {leader.candidate.family_id} needs NVLink (DeepEP intranode "
            "kernels) and no measured configuration runs without it"
        )
    chosen = max(others.values(), key=lambda p: p.tps_per_gpu or 0.0)
    return chosen, (
        f"the leader {leader.candidate.family_id} needs NVLink (DeepEP intranode "
        f"kernels); the comparison uses the best configuration without DeepEP "
        f"({chosen.candidate.family_id})"
    )


def nvlink_off_points(history: History, base: PointResult) -> list[PointResult]:
    """NVLink-off runs of ``base`` at its concurrency (either path)."""
    out = []
    for path in ("pcie", "host"):
        key = base.candidate.model_copy(update={"p2p": path}).key
        out += [
            p
            for p in history.points_for(key, ("nvlink",))
            if p.key.concurrency == base.key.concurrency
        ]
    return out


def nvlink_off_failed(history: History, base: PointResult) -> bool:
    return any(
        history.deploy_failed(base.candidate.model_copy(update={"p2p": path}).key)
        for path in ("pcie", "host")
    )


def nvlink_plan(
    ctx: SearchContext, history: History, budget: Budget
) -> tuple[PointResult | None, str]:
    """The point to repeat with NVLink off, or None and why not (before any
    node is touched: switching NVLink is only worth it when a run follows)."""
    base, reason = nvlink_base(ctx, history)
    if base is None:
        return None, reason
    if nvlink_off_points(history, base):
        return None, "NVLink comparison measured"
    if nvlink_off_failed(history, base):
        return None, "NVLink-off deployment failed"
    cost = NVLINK_TOGGLE_S + budget.deploy_s + budget.point_s["screen"]
    if not budget.affordable(cost):
        return None, "budget skipped the NVLink comparison"
    return base, reason


def decide_nvlink(
    ctx: SearchContext, history: History, budget: Budget, p2p: str
) -> Action:
    """Repeat the base point with GPU-GPU traffic on ``p2p`` ("pcie" | "host")."""
    base, reason = nvlink_base(ctx, history)
    if base is None:
        return Done(reason)
    off = base.candidate.model_copy(update={"p2p": p2p})
    if history.deploy_failed(off.key):
        return Done("NVLink-off deployment failed")
    if nvlink_off_points(history, base):
        return Done("NVLink comparison measured")
    if not budget.affordable(
        deploy_cost(budget, history, off) + budget.point_s["screen"]
    ):
        return Done("budget skipped the NVLink comparison")
    return RunPoint(
        off,
        base.key.concurrency,
        "screen",
        "nvlink",
        reason=f"NVLink off: GPU-GPU over {p2p} at the leader's concurrency",
    )


def pilot_samples(history: History, leader: PointResult) -> int:
    """Samples at the leader's knee: its search point plus pilot repetitions."""
    return 1 + len(
        [
            p
            for p in history.points_for(leader.candidate.key, ("pilot",))
            if p.key.concurrency == leader.key.concurrency
        ]
    )


# --------------------------------------------------------------------------
# Budget reservations for the steps after the current one
# --------------------------------------------------------------------------

# The planner's steps in execution order; the family search is the only
# elastic one, every later step is a user-requested deliverable.
STEPS = (
    "search",
    "floors",
    "mtp",
    "accuracy",
    "pilot",
    "nic_sweep",
    "confirm",
    "nvlink",
    "diagnose",
    "finalize",
)
# Floor-bracketing points assumed while the leader's lowest point is unknown.
FLOOR_POINTS_UNKNOWN = 2
# Stand-ins for candidates that are not known yet; distinct keys so the
# deployment walk below counts one redeploy per switch between them.
_LEADER = "<leader>"
_LEADER_MTP_OFF = "<leader>/mtp-off"
_LEADER_NVLINK_OFF = "<leader>/nvlink-off"


@dataclass(frozen=True)
class StepCosts:
    """Reservation inputs that do not come from the history."""

    deploy_s: float  # deploy estimate when the leader's family has none observed
    point_s: Mapping[str, float]  # per-spec point duration estimates
    accuracy: bool = False  # the accuracy gate runs
    diagnose_s: float = 0.0  # opted-in diagnostic window, without its deploy
    finalize_s: float = 600.0
    margin: float = 0.10  # share added to the step costs for deploy-time variance


def floor_points_needed(
    ctx: SearchContext, history: History, leader: PointResult | None
) -> int:
    """Descent points ``decide_floors`` would still measure on this leader."""
    if ctx.stage == "prefill":
        return 0
    if leader is None:
        return FLOOR_POINTS_UNKNOWN
    cand = leader.candidate
    measured = history.points_for(cand.key, ("search",))
    pts = curve(measured)
    if not pts:
        return FLOOR_POINTS_UNKNOWN
    lowest = pts[0]
    user = lowest.metrics.user_tps_p50 if lowest.metrics else None
    grid = ctx.grids.get(cand.family_id)
    if grid is None or user is None or user >= max(ctx.floors, default=0.0):
        return 0
    descents = sum(1 for p in measured if p.key.concurrency < grid.start)
    needed, conc = 0, lowest.key.concurrency
    while descents + needed < MAX_FLOOR_DESCENTS and conc // 2 >= grid.min:
        conc //= 2
        needed += 1
    return needed


def family_deploy_s(history: History, family_id: str | None, fallback: float) -> float:
    """Slowest successful deploy of ``family_id``; ``fallback`` when none ran."""
    seen = [
        d.load_seconds
        for d in history.deploys
        if d.ok and d.load_seconds > 0 and d.candidate.family_id == family_id
    ]
    return max(seen) if seen else fallback


def _mtp_off_key(
    ctx: SearchContext, history: History, leader: PointResult | None
) -> str | None:
    """Key of the MTP-off variant ``decide_mtp`` would still measure."""
    if not ctx.mtp_capable:
        return None
    if leader is None:
        return _LEADER_MTP_OFF
    if not leader.candidate.mtp or mtp_disabled_by(ctx, leader.candidate) is not None:
        return None
    off = leader.candidate.model_copy(update={"mtp": False})
    if history.deploy_failed(off.key) or history.points_for(off.key, ("mtp_ablation",)):
        return None
    return off.key


def _nic_arm_keys(ctx: SearchContext, history: History, phase: str) -> list[str]:
    """Arms ``decide_nic`` would still measure, as candidate keys."""
    if not ctx.nic_counts:
        return []
    full = max(ctx.nic_counts)
    base, _ = nic_base(history)
    if base is None:
        if phase in ("search", "floors", "mtp"):
            # A configuration with cross-node traffic may still be measured.
            return [
                f"<nic-base>/{n}" for n in nic_arm_order(ctx.nic_counts) if n != full
            ]
        return []
    keys = []
    for count in nic_arm_order(ctx.nic_counts):
        cand = nic_arm_candidate(base.candidate, count, full)
        purposes = ("search", "nic", "mtp_ablation") if count == full else ("nic",)
        done = any(
            p.key.concurrency == base.key.concurrency
            for p in history.points_for(cand.key, purposes)
        )
        if not done and not history.deploy_failed(cand.key):
            keys.append(cand.key)
    return keys


def _nvlink_off_key(ctx: SearchContext, history: History) -> str | None:
    """Key of the NVLink-off run ``decide_nvlink`` would still make."""
    if not ctx.nvlink:
        return None
    base, _ = nvlink_base(ctx, history)
    if base is None:
        return _LEADER_NVLINK_OFF if final_leader(history) is None else None
    if nvlink_off_points(history, base) or nvlink_off_failed(history, base):
        return None
    return base.candidate.model_copy(update={"p2p": "pcie"}).key


def reserve_plan(
    ctx: SearchContext, history: History, phase: str, costs: StepCosts
) -> dict[str, float]:
    """Seconds each step after ``phase`` still needs, before the margin.

    Every step is costed the way its ``decide_*`` function will run it,
    given what is already measured and which candidate the previous step
    leaves running: the floor bracketing only when the leader's lowest
    point is below the top floor, a redeploy only when another candidate
    runs (the accuracy check leaves the leader running for the pilot), the
    NIC arms not measured yet. The walk starts at ``phase`` itself, whose
    pending work is not reserved but decides what runs afterwards. Deploys
    take the slowest observed deploy of the leader's family. The
    reservation follows the current leader; when a later point changes the
    leader, the next decision re-plans.
    """
    if phase not in STEPS:
        return {}
    walk = STEPS[STEPS.index(phase) :]
    screen, decision = costs.point_s["screen"], costs.point_s["decision"]
    search_lead = leader_point(history, ("search",))
    lead = final_leader(history)
    lead_key = lead.candidate.key if lead else _LEADER
    deploy_s = family_deploy_s(
        history, lead.candidate.family_id if lead else None, costs.deploy_s
    )
    running = history.deployed_key

    def run_on(key: str) -> float:
        nonlocal running
        cost = 0.0 if key == running else deploy_s
        running = key
        return cost

    plan: dict[str, float] = {}
    for step in walk:
        cost = 0.0
        if step == "floors":
            needed = floor_points_needed(ctx, history, search_lead)
            if needed:
                key = search_lead.candidate.key if search_lead else _LEADER
                cost = run_on(key) + needed * screen
        elif step == "mtp":
            off = _mtp_off_key(ctx, history, search_lead)
            if off is not None:
                cost = run_on(off) + screen
        elif step == "accuracy":
            if costs.accuracy and (lead is None or lead_key not in history.accuracy):
                cost = run_on(lead_key) + ACCURACY_CHECK_S
        elif step == "pilot":
            missing = PILOT_REPS - (pilot_samples(history, lead) if lead else 1)
            if history.noise is None and missing > 0:
                cost = run_on(lead_key) + missing * screen
        elif step == "nic_sweep":
            for key in _nic_arm_keys(ctx, history, phase):
                cost += run_on(key) + screen
        elif step == "confirm":
            if lead is None or not (
                history.points_for(lead_key, ("confirm",))
                or history.deploy_failed(lead_key)
            ):
                cost = run_on(lead_key) + decision
        elif step == "nvlink":
            off = _nvlink_off_key(ctx, history)
            if off is not None:
                cost = NVLINK_TOGGLE_S + run_on(off) + screen
        elif step == "diagnose":
            if costs.diagnose_s > 0:
                cost = deploy_s + costs.diagnose_s
        elif step == "finalize":
            cost = costs.finalize_s
        if step != phase:
            plan[step] = cost
    return plan


def soft_reserve_after(
    ctx: SearchContext, history: History, phase: str, costs: StepCosts
) -> float:
    """The share of ``reserve_after`` a competitive climb may spend: the
    NVLink comparison (a secondary result) with its margin."""
    plan = reserve_plan(ctx, history, phase, costs)
    return plan.get("nvlink", 0.0) * (1.0 + costs.margin)


def reserve_after(
    ctx: SearchContext, history: History, phase: str, costs: StepCosts
) -> float:
    """Seconds the steps after ``phase`` need, so an earlier step never spends
    them: the planned step costs plus ``margin``, finalize kept as is."""
    plan = reserve_plan(ctx, history, phase, costs)
    finalize = plan.pop("finalize", 0.0)
    return sum(plan.values()) * (1.0 + costs.margin) + finalize
