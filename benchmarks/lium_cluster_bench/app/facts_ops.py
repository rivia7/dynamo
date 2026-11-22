# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node facts, software stack, clocks and disk reads: collection points and
the summary fold.

Collected at preflight ("before" snapshot, software stack in the image, clock
offsets), after the download (weights-disk read speed) and at finalize
("after" snapshot). Raw documents stay under ``<exp>/facts/``; FACTS events
record what was collected so a resumed run and the summary find them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.lium_cluster_bench.adapters.docker import experiment_labels
from benchmarks.lium_cluster_bench.adapters.nodefacts import FactCollector
from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.app.fabric_ops import COLLECTION_ERRORS
from benchmarks.lium_cluster_bench.domain.nodefacts import (
    ClockOffset,
    DiskRead,
    FactFinding,
    HealthDelta,
    NodeFacts,
    SoftwareStack,
    clock_findings,
    clock_offsets,
    delta_findings,
    disk_findings,
    fact_findings,
    facts_signature,
    health_delta,
    parse_node_facts,
    software_findings,
    unavailable_facts,
)
from benchmarks.lium_cluster_bench.domain.schema import EventType

logger = logging.getLogger(__name__)

FACTS_DIR = "facts"
SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


def make_collector(ctx: RunContext) -> FactCollector:
    cfg = ctx.config
    return FactCollector(
        ctx.cluster,
        image=cfg.image,
        workdir=cfg.workdir,
        results_dir=ctx.exp_dir / FACTS_DIR,
        now=ctx.clock.now,
        src_dir=ctx.src_dir,
        python=ctx.python,
        labels=experiment_labels(cfg.exp_id),
        reference=ctx.node0,
    )


def _event(ctx: RunContext, tag: str) -> dict[str, Any] | None:
    for event in reversed(ctx.store.of_type(EventType.FACTS)):
        if event.data.get("tag") == tag:
            return event.data
    return None


def collect_host(ctx: RunContext, tag: str) -> dict[str, NodeFacts]:
    ctx.status(step=f"node facts and health snapshot ({tag})")
    try:
        facts = make_collector(ctx).host_facts(ctx.node_names, tag=tag)
    except COLLECTION_ERRORS as exc:
        ctx.note(f"node facts ({tag}) failed: {exc}")
        facts = {n: unavailable_facts(n, str(exc)) for n in ctx.node_names}
    ctx.store.append(
        EventType.FACTS,
        {"tag": tag, "available": {n: f.available for n, f in facts.items()}},
    )
    return facts


def collect_preflight(ctx: RunContext) -> None:
    """'before' snapshot, software stack in the image (GPU 0), clock offsets."""
    collect_host(ctx, "before")
    collector = make_collector(ctx)
    ctx.status(step="software stack inside the serving image")
    try:
        software = collector.software(ctx.node0)
        samples = collector.clock_samples(ctx.node_names)
    except COLLECTION_ERRORS as exc:
        ctx.note(f"software stack / clock offsets failed: {exc}")
        software, samples = None, {}
    ctx.store.append(
        EventType.FACTS,
        {
            "tag": "software",
            "software": software.model_dump(mode="json") if software else None,
        },
    )
    ctx.store.append(
        EventType.FACTS,
        {
            "tag": "clock",
            "samples": {n: [list(s) for s in v] for n, v in samples.items()},
        },
    )


def collect_disk(ctx: RunContext, model_dir: str) -> None:
    """Direct-read speed of the weights disk (cold-start and restart cost)."""
    if _event(ctx, "disk") is not None:
        return
    ctx.status(step="weights disk read speed")
    try:
        reads = make_collector(ctx).disk_read(ctx.node_names, model_dir)
    except COLLECTION_ERRORS as exc:
        ctx.note(f"weights disk read failed: {exc}")
        reads = {}
    ctx.store.append(
        EventType.FACTS,
        {
            "tag": "disk",
            "reads": {n: r.model_dump(mode="json") for n, r in reads.items()},
        },
    )


def load_host_facts(exp_dir: Path, tag: str, nodes: list[str]) -> dict[str, NodeFacts]:
    out: dict[str, NodeFacts] = {}
    for node in nodes:
        path = exp_dir / FACTS_DIR / f"host-{tag}" / f"{node}.json"
        if not path.exists():
            out[node] = unavailable_facts(node, f"no {tag} snapshot")
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                out[node] = parse_node_facts(json.load(fh), node=node)
        except (OSError, ValueError) as exc:
            out[node] = unavailable_facts(node, f"unreadable {tag} snapshot: {exc}")
    return out


@dataclass
class FactsFold:
    """The node-facts fields of the experiment summary."""

    software: SoftwareStack | None = None
    signature: dict[str, Any] = field(default_factory=dict)
    findings: tuple[FactFinding, ...] = ()
    deltas: tuple[HealthDelta, ...] = ()
    clocks: tuple[ClockOffset, ...] = ()
    disks: tuple[DiskRead, ...] = ()


def fold_facts(ctx: RunContext) -> FactsFold:
    nodes = ctx.node_names
    if _event(ctx, "before") is None:
        return FactsFold()
    before = load_host_facts(ctx.exp_dir, "before", nodes)
    after = load_host_facts(ctx.exp_dir, "after", nodes) if _event(ctx, "after") else {}
    sw_event = _event(ctx, "software") or {}
    software = (
        SoftwareStack.model_validate(sw_event["software"])
        if sw_event.get("software")
        else None
    )
    clock_event = _event(ctx, "clock") or {}
    offsets = clock_offsets(
        {n: [tuple(s) for s in v] for n, v in clock_event.get("samples", {}).items()}
    )
    disk_event = _event(ctx, "disk") or {}
    disks = {
        n: DiskRead.model_validate(r) for n, r in disk_event.get("reads", {}).items()
    }
    findings: list[FactFinding] = list(fact_findings(before))
    findings += software_findings(software, node=ctx.node0)
    findings += disk_findings(disks)
    findings += clock_findings(offsets, nodes=nodes)
    deltas: list[HealthDelta] = []
    for node in nodes:
        if node in after and before[node].available and after[node].available:
            delta = health_delta(before[node], after[node])
            deltas.append(delta)
            findings += delta_findings(delta)
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.node, f.code))
    return FactsFold(
        software=software,
        signature=facts_signature(before, software),
        findings=tuple(findings),
        deltas=tuple(deltas),
        clocks=tuple(offsets[n] for n in nodes if n in offsets),
        disks=tuple(disks[n] for n in nodes if n in disks),
    )
