# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A-Only: the run's F model, resolved once and kept in the event store, and
the summary of what the run found about it.

The plan (domain/afd.py) needs the model's config.json (MODEL_ARCH), the GPU
profile, the routed experts' precision, the inventory, the microbench peaks
and the fabric probe's latency. It is resolved before the first A-Only
deployment and stored as a DECISION (kind ``afd_plan``), so a resumed run
renders the same wait; a modelled step time that would only measure the link
stops the run before the first deployment.
"""

from __future__ import annotations

from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.fabric_ops import fabric_event, stored_fabric
from benchmarks.cluster_bench.app.history import peaks
from benchmarks.cluster_bench.app.model_ops import stored_arch
from benchmarks.cluster_bench.app.store import EventStore
from benchmarks.cluster_bench.domain import afd, analysis, search
from benchmarks.cluster_bench.domain.profiles import select_checkpoint
from benchmarks.cluster_bench.domain.schema import (
    AfdPlan,
    AfdReport,
    DeployResult,
    EventType,
    LaunchPlan,
    PointResult,
)
from benchmarks.cluster_bench.domain.stages import stage_spec

PLAN_KIND = "afd_plan"
# Simulated-F points the null-F bound compares against (same candidate and
# concurrency, screen length).
SIM_PURPOSES = ("search", "mtp_ablation", "nic", "pilot")


def stored_plan(store: EventStore) -> AfdPlan | None:
    """The F plan a run stored; None before it has one."""
    for event in reversed(store.of_type(EventType.DECISION)):
        if event.data.get("kind") == PLAN_KIND:
            return AfdPlan.model_validate(event.data["plan"])
    return None


def instance_gpus(ctx: RunContext) -> tuple[int, ...]:
    """GPUs of one instance of the stage's role, per distinct family size."""
    role = stage_spec(ctx.config.stage).role
    families = ctx.require_hardware().deployment.families
    return tuple(
        sorted(
            {r.gpus_per_instance for f in families for r in f.roles if r.role == role}
        )
    )


def plan_instance_gpus(plan: LaunchPlan) -> int:
    """GPUs of one instance of a stage launch plan (one role, equal instances)."""
    return plan.gpu_count // max(1, sum(plan.expected_instances.values()))


def ensure_plan(ctx: RunContext) -> AfdPlan | None:
    """The run's F plan; None unless the stage runs a side of AFD."""
    if stage_spec(ctx.config.stage).afd_side is None:
        return None
    if ctx.afd is None:
        plan = stored_plan(ctx.store)
        if plan is None:
            plan = resolve(ctx)
            ctx.store.append(
                EventType.DECISION,
                {"kind": PLAN_KIND, "plan": plan.model_dump(mode="json")},
            )
            for note in plan.notes:
                ctx.note(f"A-Only F model: {note}")
        ctx.afd = plan
    return ctx.afd


def resolve(ctx: RunContext) -> AfdPlan:
    """A new plan from what the run measured; AfdError when it cannot be sized."""
    hw = ctx.require_hardware()
    arch = stored_arch(ctx)
    if arch is None:
        raise afd.AfdError(
            "A-Only needs the model's config.json (no MODEL_ARCH event): the F "
            "model cannot be sized"
        )
    variant, _ = select_checkpoint(hw.model, hw.gpu, ctx.config.checkpoint)
    _topology, links, _findings = stored_fabric(fabric_event(ctx))
    plan = afd.resolve_plan(
        ctx.config.afd_f,
        arch=arch,
        gpu=hw.gpu,
        dtype=afd.routed_dtype(variant, ctx.config.afd_f.expert_dtype),
        inventory=hw.inventory,
        peaks=peaks(ctx.store),
        links=links,
    )
    warnings = [
        warning
        for n_a in instance_gpus(ctx)
        if (warning := afd.check_step_time(plan, n_a)) is not None
    ]
    return plan.model_copy(update={"notes": plan.notes + tuple(warnings)})


def afd_summary(
    ctx: RunContext,
    history: search.History,
    leader: PointResult | None,
    lead_deploy: DeployResult | None,
    lead_plan: LaunchPlan | None,
) -> AfdReport | None:
    """The F model, the null-F bound and the A<->F traffic of an A-Only run."""
    spec = stage_spec(ctx.config.stage)
    if spec.afd_side is None:
        return None
    plan = ctx.afd or stored_plan(ctx.store)
    if plan is None:
        return None  # the run stopped before the F model was sized
    hw = ctx.require_hardware()
    n_a = (
        plan_instance_gpus(lead_plan)
        if lead_plan is not None
        else min(instance_gpus(ctx), default=1)
    )
    null_points = [p for p in history.points if p.key.purpose == "f_bound"]
    null = null_points[-1] if null_points else None
    sim = None
    if null is not None:
        sim_key = null.candidate.model_copy(update={"null_f": False}).key
        same = [
            p
            for p in history.points_for(sim_key, SIM_PURPOSES)
            if p.key.concurrency == null.key.concurrency and p.key.spec == "screen"
        ]
        sim = same[0] if same else None
    skips = [
        e.data
        for e in ctx.store.of_type(EventType.DECISION)
        if e.data.get("phase") == "f_bound" and e.data.get("kind") == "phase_result"
    ]
    reason = str(skips[-1].get("reason", "")) if skips else ""
    bound = analysis.afd_bound(sim, null, history.noise, reason)
    if null is None and leader is not None:
        bound_key = leader.candidate.model_copy(update={"null_f": True}).key
        if history.deploy_failed(bound_key):
            bound = bound.model_copy(
                update={"status": "failed", "reason": "null-F deployment failed"}
            )
    rendered = lead_plan.rendered_configs if lead_plan is not None else {}
    draft_rows = max(
        (
            int(cfg.get("speculative-num-draft-tokens") or 1)
            for cfg in rendered.values()
            if cfg.get("speculative-algorithm")
        ),
        default=1,
    )
    traffic = afd.traffic(
        plan,
        draft_rows=draft_rows,
        accept_len=leader.accept_len if leader is not None else None,
        sim_tps_per_gpu=bound.sim_tps_per_gpu
        or (leader.tps_per_gpu if leader is not None else None),
        null_tps_per_gpu=bound.null_tps_per_gpu,
    )
    lifecycle = lead_deploy.lifecycle if lead_deploy is not None else None
    role = lifecycle.roles.get(spec.role or "decode") if lifecycle else None
    _, checkpoint = select_checkpoint(hw.model, hw.gpu, ctx.config.checkpoint)
    expected = afd.expected_weights(plan, checkpoint.approx_size_gb, n_a)
    return AfdReport(
        plan=plan,
        n_a=n_a,
        bound=bound,
        traffic=traffic,
        steps=afd.step_rows(plan, n_a),
        weight_gb_per_gpu=role.weight_mem_gb_max if role is not None else None,
        null_weight_gb_per_gpu=expected[0] if expected else None,
        loaded_weight_gb_per_gpu=expected[1] if expected else None,
    )
