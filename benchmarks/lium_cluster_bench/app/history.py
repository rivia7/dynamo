# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct planner inputs (history, budget) from the event log."""

from __future__ import annotations

import statistics
from typing import Sequence

from benchmarks.lium_cluster_bench.app.store import EventStore
from benchmarks.lium_cluster_bench.domain.schema import (
    BenchSpec,
    BudgetConfig,
    DeployResult,
    Event,
    EventType,
    NoiseResult,
    PeakResult,
    PointResult,
)
from benchmarks.lium_cluster_bench.domain.search import (
    Budget,
    History,
    SearchContext,
    StepCosts,
    reserve_after,
    soft_reserve_after,
)

AIPERF_CONFIGURE_ESTIMATE_S = (
    240.0  # dataset load + tokenizer warm-up, refined from observations
)
DRAIN_ESTIMATE_S = 60.0


def build_history(store: EventStore, deployed_key: str | None = None) -> History:
    points: list[PointResult] = []
    deploys: list[DeployResult] = []
    closed: dict[str, tuple[str, str]] = {}
    noise: NoiseResult | None = None
    accuracy: dict[str, str] = {}
    for event in store.events:
        if event.type == EventType.POINT:
            points.append(PointResult.model_validate(event.data["result"]))
        elif event.type == EventType.DEPLOY:
            deploys.append(DeployResult.model_validate(event.data["result"]))
        elif (
            event.type == EventType.DECISION
            and event.data.get("kind") == "close_family"
        ):
            closed[event.data["family_id"]] = (
                event.data["status"],
                event.data["reason"],
            )
        elif event.type == EventType.NOISE:
            noise = NoiseResult.model_validate(event.data["result"])
        elif event.type == EventType.ACCURACY:
            cand_key = event.data.get("cand_key")
            verdict = event.data.get("verdict") or {}
            if isinstance(cand_key, str):
                accuracy[cand_key] = str(verdict.get("status") or "error")
    return History(
        points=tuple(points),
        deploys=tuple(deploys),
        closed=closed,
        noise=noise,
        deployed_key=deployed_key,
        accuracy=accuracy,
    )


def peaks(store: EventStore) -> list[PeakResult]:
    return [
        PeakResult.model_validate(p)
        for e in store.of_type(EventType.PEAK)
        for p in e.data["results"]
    ]


def spent_seconds(events: Sequence[Event], now: float) -> float:
    """Wall-clock the runner was alive, summed over process lifetimes.

    Each process start appends RUN_START; a process's lifetime ends at its last
    event (a crash leaves no end marker). The current process runs until now.
    """
    spent = 0.0
    run_start: float | None = None
    last_ts: float | None = None
    for event in events:
        if event.type == EventType.RUN_START:
            if run_start is not None and last_ts is not None:
                spent += max(0.0, last_ts - run_start)
            run_start = event.ts
        last_ts = event.ts
    if run_start is not None:
        spent += max(0.0, now - run_start)
    return spent


def _median(values: list[float], default: float) -> float:
    return statistics.median(values) if values else default


def estimates(
    history: History, budget: BudgetConfig, specs: dict[str, BenchSpec]
) -> tuple[float, dict[str, float]]:
    """Deploy and per-spec point cost estimates from observed history."""
    deploy_s = _median(
        [d.load_seconds for d in history.deploys if d.ok and d.load_seconds > 0],
        budget.deploy_estimate_min * 60,
    )
    point_s: dict[str, float] = {}
    for spec_id, spec in specs.items():
        observed = [
            p.ended_at - p.started_at
            for p in history.points
            if p.key.spec == spec_id and p.ended_at > p.started_at
        ]
        default = (
            AIPERF_CONFIGURE_ESTIMATE_S
            + spec.warmup_s
            + spec.duration_s
            + DRAIN_ESTIMATE_S
        )
        point_s[spec_id] = _median(observed, default)
    return deploy_s, point_s


def make_budget(
    store: EventStore,
    history: History,
    budget_cfg: BudgetConfig,
    specs: dict[str, BenchSpec],
    phase: str,
    plan: SearchContext | None,
    now: float,
    *,
    accuracy: bool = False,
    diagnose_s: float = 0.0,
) -> Budget:
    """Remaining budget and the reservation the steps after ``phase`` need.

    ``plan`` is the phase's search context (floors, grids, NIC counts); without
    one the defaults apply, which reserve every NIC arm.
    """
    deploy_s, point_s = estimates(history, budget_cfg, specs)
    remaining = budget_cfg.total_hours * 3600 - spent_seconds(store.events, now)
    costs = StepCosts(
        deploy_s=deploy_s,
        point_s=point_s,
        accuracy=accuracy,
        diagnose_s=diagnose_s,
        finalize_s=budget_cfg.reserve_finalize_min * 60,
        margin=budget_cfg.reserve_margin,
    )
    context = plan or SearchContext(families=(), ladder=(), grids={})
    reserve = reserve_after(context, history, phase, costs)
    soft = soft_reserve_after(context, history, phase, costs)
    if phase != "search":
        # The NVLink comparison yields to every deliverable before it; only
        # the search's exploration of new families must leave its share.
        reserve, soft = max(0.0, reserve - soft), 0.0
    return Budget(
        remaining_s=remaining,
        deploy_s=deploy_s,
        point_s=point_s,
        reserve_s=reserve,
        soft_reserve_s=soft,
    )
