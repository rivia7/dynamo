# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planner-driven phases: search (+ floors, MTP ablation), pilot, NIC sweep,
confirm. Each phase only wires a pure ``decide_*`` function into ``drive``."""

from __future__ import annotations

from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.app.history import build_history
from benchmarks.lium_cluster_bench.app.pipeline import PhaseRuntime, drive
from benchmarks.lium_cluster_bench.domain import search
from benchmarks.lium_cluster_bench.domain.analysis import noise_from_samples, valid_points
from benchmarks.lium_cluster_bench.domain.nicplan import feasible_counts
from benchmarks.lium_cluster_bench.domain.profiles import fit_family, scale_grid
from benchmarks.lium_cluster_bench.domain.schema import EventType


def search_context(ctx: RunContext) -> search.SearchContext:
    hw = ctx.require_hardware()
    inventory, gpu, model, deployment = hw.inventory, hw.gpu, hw.model, hw.deployment
    gpn = len(inventory.nodes[0].gpus)
    wanted = set(ctx.config.families)
    families = []
    for family in deployment.families:
        if wanted and not any(family.matches(ref) for ref in wanted):
            continue
        if not wanted and not family.enabled:
            continue
        fitted = fit_family(family, inventory.gpu_count, gpn)
        if fitted is not None:
            families.append(fitted)
    families.sort(key=lambda f: (f.priority, f.id))
    grids = {
        family.id: scale_grid(
            family.grid or deployment.default_grid, family.scale, gpu.lane_cap
        )
        for family in families
    }
    return search.SearchContext(
        families=tuple(families),
        ladder=deployment.ladder,
        grids=grids,
        simulate_acceptance=ctx.config.simulate_acceptance,
        mtp_capable=bool(model.spec_decode_keys),
        nic_counts=feasible_counts(inventory, ctx.config.nic_counts),
        nvlink=gpu.nvlink and ctx.config.nvlink_ablation,
    )


def unsupported_verdict(ctx: RunContext) -> str | None:
    for event in reversed(ctx.store.of_type(EventType.DECISION)):
        if (
            event.data.get("kind") == "verdict"
            and event.data.get("verdict") == "unsupported"
        ):
            return str(event.data.get("reason", "unsupported"))
    return None


class SearchPhase:
    name = "search"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        if unsupported_verdict(ctx):
            ctx.note("search skipped: the kernel smoke found this GPU type unsupported")
            return
        sc = search_context(ctx)
        rt.plan = sc
        drive(ctx, rt, "search", lambda h, b: search.decide_search(sc, h, b))
        drive(ctx, rt, "floors", lambda h, b: search.decide_floors(sc, h, b))
        drive(ctx, rt, "mtp", lambda h, b: search.decide_mtp(sc, h, b))


class AccuracyPhase:
    """Accuracy gate on the leader (and the next best after a failure)."""

    name = "accuracy"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        if unsupported_verdict(ctx):
            return
        if not ctx.config.accuracy.enabled:
            ctx.store.append(
                EventType.DECISION,
                {
                    "kind": "phase_result",
                    "phase": "accuracy",
                    "reason": "accuracy gate disabled (--no-accuracy)",
                },
            )
            return
        sc = search_context(ctx)
        rt.plan = sc
        limit = ctx.config.accuracy.max_checks
        drive(
            ctx,
            rt,
            "accuracy",
            lambda h, b: search.decide_accuracy(sc, h, b, max_checks=limit),
        )


class PilotPhase:
    name = "pilot"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        if unsupported_verdict(ctx):
            return
        sc = search_context(ctx)
        rt.plan = sc
        drive(ctx, rt, "pilot", lambda h, b: search.decide_pilot(sc, h, b))
        history = build_history(ctx.store)
        if history.noise is not None:
            return
        leader = search.final_leader(history)
        if leader is None:
            return
        reps = [
            p
            for p in valid_points(history.points_for(leader.candidate.key, ("pilot",)))
            if p.key.concurrency == leader.key.concurrency
        ]
        samples = [leader.tps_per_gpu or 0.0] + [p.tps_per_gpu or 0.0 for p in reps]
        if len(samples) < 2:
            ctx.note("noise floor unmeasured: fewer than two valid pilot repetitions")
            return
        noise = noise_from_samples(
            leader.candidate.key, leader.key.concurrency, samples
        )
        ctx.store.append(EventType.NOISE, {"result": noise.model_dump(mode="json")})


class NicSweepPhase:
    name = "nic_sweep"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        if unsupported_verdict(ctx):
            return
        sc = search_context(ctx)
        rt.plan = sc
        drive(ctx, rt, "nic_sweep", lambda h, b: search.decide_nic(sc, h, b))


class ConfirmPhase:
    name = "confirm"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        if not unsupported_verdict(ctx):
            sc = search_context(ctx)
            rt.plan = sc
            drive(ctx, rt, "confirm", lambda h, b: search.decide_confirm(sc, h, b))
        rt.session.teardown()
