# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fold the event log into an ``ExperimentSummary`` (report + KB input)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from benchmarks.cluster_bench.adapters.aiperf import dropped_trace_count
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.fabric_ops import fabric_event, stored_fabric
from benchmarks.cluster_bench.app.facts_ops import FactsFold, fold_facts
from benchmarks.cluster_bench.app.history import build_history, peaks, spent_seconds
from benchmarks.cluster_bench.app.model_ops import stored_arch
from benchmarks.cluster_bench.app.nvlink_ops import nvlink_events
from benchmarks.cluster_bench.domain import analysis
from benchmarks.cluster_bench.domain import nvlink as nvl
from benchmarks.cluster_bench.domain import search
from benchmarks.cluster_bench.domain.accuracy import AccuracyResult, AccuracyVerdict
from benchmarks.cluster_bench.domain.capacity import interconnect_needs, kv_capacity
from benchmarks.cluster_bench.domain.cost import cost_summary, node_hour_price
from benchmarks.cluster_bench.domain.fabric import topology_findings, topology_signature
from benchmarks.cluster_bench.domain.inventory import shape_fingerprint
from benchmarks.cluster_bench.domain.nicplan import feasible_counts
from benchmarks.cluster_bench.domain.schema import (
    INTERACTIVITY_FLOORS,
    AccuracyCheck,
    Checkpoint,
    DiagnosticsResult,
    Event,
    EventType,
    ExperimentSummary,
    FamilyStatus,
    FamilySummary,
    GpuProfile,
    LaunchPlan,
    NvlinkComparison,
    PointResult,
    ProviderInfo,
    Verdict,
)
from benchmarks.cluster_bench.domain.stages import stage_label

logger = logging.getLogger(__name__)

PROVIDER_FILE = "provider.json"

# What a P-Only / D-Only result rests on that no GPU run has checked yet.
STAGE_UNVERIFIED: dict[str, tuple[str, ...]] = {
    "prefill": (
        "P-Only: the PD prefill role config run as an aggregated worker matches "
        "a real prefill worker's cost (no KV send, no bootstrap queue)",
        "P-Only: AIPerf max_completion_tokens:1 caps every AgentX turn at one "
        "output token (audited per point: OSL <= 1.5)",
    ),
    "decode": (
        "D-Only: dynamo.sglang decode with SGLang's fake KV transfer and no "
        "prefill peer (source-checked against Dynamo 1.6.2 / SGLang 0.5.21 and 0.5.23; "
        "preflight runs the shim's --check)",
        "D-Only: MoE expert load over meaningless KV may differ from real "
        "requests (EP balance), and MTP acceptance is simulated",
    ),
}

UNVERIFIED_BASE = (
    "SGLang 0.5.21 Mooncake device flag --disaggregation-ib-device "
    "(checked by the preflight --help probe)",
    "NVML NVLink throughput counters on this driver/GPU "
    "(falls back to nvidia-smi or 'unavailable')",
    "SGLang speculative acceptance-length metric name (matched by substring)",
    "IB port counters assumed in 4-byte units, calibrated against NCCL "
    "microbenchmarks when available",
    "PortXmitWait tick length (switch and HCA stall fractions are estimates)",
    "SGLang 0.5.21 / Dynamo 1.6.2 metric names for KV usage and PD queues "
    "(absent metrics are reported as missing)",
    "SGLang startup log lines used for cold-start phases and KV capacity",
    "Fabric tools in the pinned image (infiniband-diags, perftest); probing "
    "falls back to the host tools, then to sysfs only",
    "NVML GPM activity units (percent, MiB/s) and support on this GPU and pod "
    "(availability is recorded per GPU)",
    "GSM8K accuracy floor calibrated by binomial analysis, not yet on GLM-5.3 "
    "through Dynamo /v1/completions",
    "Goodput SLO tiers (TTFT and ITL bounds) are harness defaults, not product SLOs",
    "Model FLOPs counted from config.json (MLA / MoE / DSA formula; useful "
    "tokens only, MTP drafts excluded); MFU uses the dense datasheet peak of "
    "the checkpoint's GEMM precision",
    "KV capacity per instance = max_total_num_tokens (logged by TP rank 0) x "
    "attention-DP ranks",
    "nvidia-smi nvlink -sBwMode / -sLWidth inside the pod (needs root; output "
    "format undocumented); a switch counts only when GPU-GPU copies drop to "
    "PCIe speed and NVLink counters read zero under load",
)


def profile_unverified(gpu: GpuProfile) -> tuple[str, ...]:
    """Sentences of the GPU profile's notes that are marked UNVERIFIED."""
    sentences = re.split(r"(?<=\.)\s+", gpu.notes.strip()) if gpu.notes else []
    return tuple(
        f"{gpu.name} profile: {s.rstrip('.')}" for s in sentences if "UNVERIFIED" in s
    )


def nvlink_summary(ctx: RunContext, history: search.History) -> NvlinkComparison | None:
    """The NVLink-off run against the same configuration with NVLink."""
    switches = nvlink_events(ctx)
    decisions = [
        e.data
        for e in ctx.store.of_type(EventType.DECISION)
        if e.data.get("phase") == "nvlink"
    ]
    if not switches and not decisions:
        return None
    off_points = [p for p in history.points if p.key.purpose == "nvlink"]
    off = off_points[-1] if off_points else None
    on = None
    if off is not None:
        on_key = off.candidate.model_copy(update={"p2p": "nvlink"}).key
        same = [
            p
            for p in history.points_for(on_key, ("search", "mtp_ablation", "nic"))
            if p.key.concurrency == off.key.concurrency
        ]
        on = same[0] if same else None
    base = [d for d in decisions if d.get("kind") == "nvlink_base"]
    skips = [d for d in decisions if d.get("kind") == "phase_result"]
    if off is not None and base:
        reason = str(base[-1].get("reason", ""))
    else:
        reason = str(skips[-1].get("reason", "")) if skips else ""
    switch, restore = nvl.last_switch(switches)
    return nvl.compare(
        on, off, history.noise, switch=switch, restore=restore, reason=reason
    )


def stored_diagnostics(events: Sequence[Event]) -> DiagnosticsResult | None:
    """The last DIAGNOSTICS event, if any (unreadable: None plus a warning)."""
    for event in reversed(events):
        try:
            return DiagnosticsResult.model_validate(event.data["result"])
        except (KeyError, ValueError) as exc:
            logger.warning("ignoring unreadable diagnostics event: %s", exc)
    return None


def plan_lookup_for(exp_dir: Path):
    def lookup(cand_fp: str) -> LaunchPlan | None:
        path = exp_dir / "deployments" / cand_fp / "plan.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return LaunchPlan.model_validate(json.load(fh))

    return lookup


def load_provider(exp_dir: Path) -> ProviderInfo | None:
    """provider.json written by the controller next to the event log, if any."""
    path = exp_dir / PROVIDER_FILE
    if not path.exists():
        return None
    try:
        return ProviderInfo.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        logger.warning("ignoring unreadable %s: %s", path, exc)
        return None


def _family_summaries(
    ctx: RunContext, history: search.History
) -> tuple[FamilySummary, ...]:
    deployment = ctx.require_hardware().deployment
    out = []
    for family in sorted(deployment.families, key=lambda f: (f.priority, f.id)):
        pts = [
            p
            for p in history.points
            if p.candidate.family_id == family.id
            and p.key.purpose == "search"
            and p.candidate.nic_count is None
        ]
        best = analysis.best_point(pts)
        status: FamilyStatus
        reason: str
        if family.id in history.closed:
            status, reason = history.closed[family.id]
        elif not family.enabled and not any(
            family.matches(ref) for ref in ctx.config.families
        ):
            status, reason = "disabled", "opt-in family (enable with --families)"
        elif pts:
            status, reason = "measured", ""
        else:
            status, reason = "skipped_budget", "not reached"
        saturated = None
        if best is not None:
            saturated, sat_note = analysis.saturation(
                [p for p in pts if p.key.cand_key == best.key.cand_key], history.noise
            )
            reason = reason or sat_note
        out.append(
            FamilySummary(
                family_id=family.id,
                title=family.title,
                status=status,
                reason=reason,
                best_point_id=best.key.id if best else None,
                best_tps_per_gpu=best.tps_per_gpu if best else None,
                best_concurrency=best.key.concurrency if best else None,
                saturated=saturated,
                cross_node_gbs=best.cross_node_gbs if best else None,
                cross_node=family.cross_node,
                point_ids=tuple(p.key.id for p in pts),
                accuracy=history.accuracy.get(best.key.cand_key) if best else None,
            )
        )
    return tuple(out)


def _nic_arms(ctx: RunContext, history: search.History, base: PointResult | None):
    if base is None:
        return ()
    inventory = ctx.inventory
    if inventory is None:
        return ()
    counts = feasible_counts(inventory, ctx.config.nic_counts)
    full = max(counts)
    arms = []
    for count in counts:
        cand = search.nic_arm_candidate(base.candidate, count, full)
        purposes = ("search", "nic", "mtp_ablation") if count == full else ("nic",)
        pts = [
            p
            for p in history.points_for(cand.key, purposes)
            if p.key.concurrency == base.key.concurrency
        ]
        arms.append((count, pts[-1] if pts else None))
    return analysis.nic_arms(arms, history.noise)


def _dropped_traces(points: list[PointResult]) -> int | None:
    from_logs: list[int] = []
    for p in points:
        log = Path(p.artifact_dir) / "aiperf.log"
        if log.exists():
            count = dropped_trace_count(log.read_text(errors="replace"))
            if count is not None:
                from_logs.append(count)
    return max(from_logs) if from_logs else None


def accuracy_checks(events: Sequence[Event]) -> tuple[AccuracyCheck, ...]:
    """ACCURACY events as summary records (per-item rows stay in accuracy/)."""
    out = []
    for event in events:
        try:
            result = AccuracyResult.model_validate(event.data["result"])
            out.append(
                AccuracyCheck(
                    cand_key=str(event.data["cand_key"]),
                    family_id=str(event.data.get("family_id") or ""),
                    result=result.model_copy(update={"items": (), "samples": ()}),
                    verdict=AccuracyVerdict.model_validate(event.data["verdict"]),
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("ignoring unreadable accuracy event: %s", exc)
    return tuple(out)


def accuracy_limitations(
    ctx: RunContext, leader: PointResult | None, checks: Sequence[AccuracyCheck]
) -> list[str]:
    out = [
        f"accuracy gate excluded {c.family_id}: {c.verdict.reason}"
        for c in checks
        if c.verdict.status == "fail"
    ]
    if ctx.config.stage != "e2e":
        out.append(
            f"accuracy gate not applicable to {stage_label(ctx.config.stage)}: "
            "the stage produces no end-to-end answers"
        )
    elif not ctx.config.accuracy.enabled:
        out.append("accuracy gate disabled: the leader's numerics are unchecked")
    elif leader is not None:
        last = [c for c in checks if c.cand_key == leader.candidate.key]
        if not last:
            out.append("the leader was not accuracy-checked")
        elif last[-1].verdict.status != "pass":
            out.append(
                f"leader accuracy check {last[-1].verdict.status}: "
                f"{last[-1].verdict.reason}"
            )
    return out


def build_summary(ctx: RunContext) -> ExperimentSummary:
    hw = ctx.require_hardware()
    inventory, gpu, model, deployment = hw.inventory, hw.gpu, hw.model, hw.deployment
    history = build_history(ctx.store)
    inv_event = ctx.store.of_type(EventType.INVENTORY)[-1].data
    checkpoint = Checkpoint.model_validate(inv_event["checkpoint"])
    downloads = ctx.store.of_type(EventType.DOWNLOAD)
    if downloads and downloads[-1].data.get("commit"):
        checkpoint = checkpoint.model_copy(
            update={"revision": downloads[-1].data["commit"]}
        )
    leader = search.final_leader(history)
    rankable = [
        p
        for p in history.points
        if p.key.purpose in ("search", "mtp_ablation") and p.key.spec == "screen"
    ]
    saturated, sat_note = (None, "")
    if leader is not None:
        saturated, sat_note = analysis.saturation(
            history.points_for(leader.candidate.key, ("search",)), history.noise
        )
    mtp = None
    if leader is not None:
        on_cand = leader.candidate.model_copy(update={"mtp": True})
        off_cand = leader.candidate.model_copy(update={"mtp": False})
        on = [
            p
            for p in history.points_for(on_cand.key, ("search",))
            if p.key.concurrency == leader.key.concurrency
        ]
        off = [
            p
            for p in history.points_for(off_cand.key, ("mtp_ablation", "search"))
            if p.key.concurrency == leader.key.concurrency
        ]
        mtp = analysis.mtp_comparison(
            on[-1] if on else None, off[-1] if off else None, history.noise
        )
    base, nic_reason = search.nic_base(history)
    nic_arms = _nic_arms(ctx, history, base)
    arch = stored_arch(ctx)
    lead_deploy = None
    if leader is not None:
        lead_deploy = next(
            (
                d
                for d in reversed(history.deploys)
                if d.ok and d.candidate.key == leader.candidate.key
            ),
            None,
        )
    lead_plan = plan_lookup_for(ctx.exp_dir)(leader.candidate.fp) if leader else None
    confirm_runs = [p for p in history.points if p.key.purpose == "confirm"]
    confirm = [p for p in confirm_runs if p.audit.usable]
    verdict: Verdict = "ok"
    unsupported = [
        e
        for e in ctx.store.of_type(EventType.DECISION)
        if e.data.get("kind") == "verdict"
    ]
    if unsupported:
        verdict = "unsupported"
    elif leader is None:
        verdict = "failed"
    elif any(
        "budget" in str(e.data.get("reason", ""))
        for e in ctx.store.of_type(EventType.DECISION)
    ):
        verdict = "partial"
    conditions = [
        "AIPerf client co-located on node0 (no spare CPU host)",
        "screening runs use --unsafe-override (AgentX submission_valid=false); "
        "the confirm run does not",
        f"served context {deployment.served_context} tokens; traces whose recorded "
        f"peak exceeds {deployment.served_context - 2048} tokens are dropped by AIPerf",
        "NIC counts restricted in software (per-transport env/config), "
        "verified with per-HCA counters",
    ]
    if ctx.config.simulate_acceptance:
        conditions.append(
            "simulated speculative acceptance (SGLANG_SIMULATE_ACC_*) enabled; "
            "results do not reflect measured acceptance"
        )
    if ctx.simulator is not None:
        conditions.append(
            "DRY RUN: all measurements are synthetic and not hardware evidence"
        )
    limitations = [e.data["text"] for e in ctx.store.of_type(EventType.NOTE)]
    if confirm_runs and not confirm:
        limitations.append(
            "confirmatory rerun invalid: " + "; ".join(confirm_runs[-1].audit.reasons)
        )
    if history.noise is None:
        limitations.append(
            "noise floor not measured; a provisional 5% threshold was used"
        )
    if leader is not None and saturated is False:
        limitations.append(f"throughput curve not saturated: {sat_note}")
    if leader is not None:
        for rung in deployment.ladder:
            if (
                rung.id in leader.candidate.rungs
                and "speculative-algorithm" in rung.sglang
                and rung.sglang["speculative-algorithm"] is None
            ):
                limitations.append(
                    f"MTP disabled by fallback rung {rung.id}; MTP on/off not compared"
                )
    events = ctx.store.events
    provider = load_provider(ctx.exp_dir)
    try:
        topology, links, link_findings = stored_fabric(fabric_event(ctx))
    except ValueError as exc:  # an unreadable FABRIC payload must not cost the report
        limitations.append(f"fabric topology unreadable: {exc}")
        topology, links, link_findings = None, None, ()
    try:
        fold = fold_facts(ctx)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        limitations.append(f"node facts unreadable: {exc}")
        fold = FactsFold()
    accuracy = accuracy_checks(ctx.store.of_type(EventType.ACCURACY))
    limitations.extend(accuracy_limitations(ctx, leader, accuracy))
    # P-Only emits one token per request: no per-user decode rate to floor.
    floors = (
        ()
        if ctx.config.stage == "prefill"
        else tuple(analysis.floor_result(rankable, f) for f in INTERACTIVITY_FLOORS)
    )
    spent_hours = spent_seconds(events, ctx.clock.now()) / 3600
    cost = cost_summary(
        price=node_hour_price(
            provider, ctx.config.price_per_node_hour, len(inventory.nodes)
        ),
        node_count=len(inventory.nodes),
        gpu_count=inventory.gpu_count,
        spent_hours=spent_hours,
        leader_tps_per_gpu=leader.tps_per_gpu if leader else None,
        floors=floors,
    )
    return ExperimentSummary(
        exp_id=ctx.config.exp_id,
        fingerprint=inv_event["fingerprint"],
        shape_fingerprint=shape_fingerprint(inventory),
        gpu_profile=gpu.name,
        gpu_name=inventory.nodes[0].gpus[0].name,
        node_count=len(inventory.nodes),
        gpus_per_node=len(inventory.nodes[0].gpus),
        model=model.name,
        checkpoint=checkpoint,
        served_context=deployment.served_context,
        dataset=ctx.config.dataset,
        aiperf_version=ctx.config.aiperf_version,
        image=ctx.config.image,
        stage=ctx.config.stage,
        platform=ctx.config.platform,
        started_at=events[0].ts if events else 0.0,
        ended_at=ctx.clock.now(),
        budget_hours=ctx.config.budget.total_hours,
        spent_hours=spent_hours,
        inventory=inventory,
        families=_family_summaries(ctx, history),
        points=history.points,
        deploys=history.deploys,
        peaks=tuple(peaks(ctx.store)),
        leader=leader.candidate if leader else None,
        leader_point_id=leader.key.id if leader else None,
        saturated=saturated,
        saturation_note=sat_note,
        floors=floors,
        noise=history.noise,
        mtp=mtp,
        nic_family_id=base.candidate.family_id if base else None,
        nic_concurrency=base.key.concurrency if base else None,
        nic_reason=nic_reason,
        nic_arms=nic_arms,
        confirm_point_id=confirm[-1].key.id if confirm else None,
        dropped_traces=_dropped_traces(list(history.points)),
        conditions=tuple(conditions),
        limitations=tuple(analysis.collapse_container_notes(limitations)),
        unverified=UNVERIFIED_BASE
        + STAGE_UNVERIFIED.get(ctx.config.stage, ())
        + profile_unverified(gpu),
        accuracy=accuracy,
        diagnostics=stored_diagnostics(ctx.store.of_type(EventType.DIAGNOSTICS)),
        verdict=verdict,
        run_config=ctx.config,
        provider=provider,
        cost=cost,
        fabric=topology,
        fabric_links=links,
        link_findings=link_findings,
        fabric_findings=tuple(topology_findings(topology)) if topology else (),
        fabric_signature=topology_signature(topology) if topology else {},
        software=fold.software,
        facts_signature=fold.signature,
        fact_findings=fold.findings,
        health_deltas=fold.deltas,
        clock_offsets=fold.clocks,
        disk_reads=fold.disks,
        model_arch=arch,
        nvlink=nvlink_summary(ctx, history),
        interconnect=interconnect_needs(
            leader,
            nic_arms,
            gpu,
            target_util=ctx.config.link_target_util,
            sweep_reason=nic_reason,
            sweep_family=base.candidate.family_id if base else None,
        ),
        kv_capacity=kv_capacity(
            lead_deploy.lifecycle if lead_deploy else None,
            lead_plan,
            arch=arch,
            requests=leader.requests if leader else None,
            state=leader.state if leader else None,
        ),
    )
