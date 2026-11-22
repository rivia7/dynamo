# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric helpers shared by the fabric phase and the measurement loop.

Kept out of ``app/phases`` because ``app/measure.py`` needs them and the
phase modules import the pipeline (which imports the measurer).
"""

from __future__ import annotations

import logging
from typing import Any

from benchmarks.lium_cluster_bench.adapters.docker import experiment_labels
from benchmarks.lium_cluster_bench.adapters.fabric import FabricProber
from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.domain.fabric import (
    FabricTopology,
    LinkFinding,
    LinkMatrix,
    SwitchPortLoad,
    SwitchPortRef,
    SwitchSnapshot,
    switch_load,
)
from benchmarks.lium_cluster_bench.domain.nicplan import affine_pairs
from benchmarks.lium_cluster_bench.domain.schema import EventType, Inventory

logger = logging.getLogger(__name__)

# What a misbehaving collector may raise; data collection must never end a
# multi-hour benchmark (Interrupted is a BaseException and still propagates).
COLLECTION_ERRORS = (ValueError, KeyError, TypeError, OSError, RuntimeError)


def rail_map(inventory: Inventory) -> dict[str, dict[str, int]]:
    """{node: {hca: rail}}; a rail is the lowest GPU index an HCA serves."""
    out: dict[str, dict[str, int]] = {}
    for node in inventory.nodes:
        if not node.active_hcas:
            continue
        rails: dict[str, int] = {}
        for gpu, hca in sorted(affine_pairs(node).items()):
            rails.setdefault(hca, gpu)
        out[node.name] = rails
    return out


def make_prober(ctx: RunContext) -> FabricProber:
    cfg = ctx.config
    return FabricProber(
        ctx.cluster,
        image=cfg.image,
        workdir=cfg.workdir,
        results_dir=ctx.exp_dir / "fabric",
        run_id=cfg.exp_id,
        src_dir=ctx.src_dir,
        python=ctx.python,
        control_ips={n.name: n.address for n in ctx.nodes},
        labels=experiment_labels(cfg.exp_id),
    )


def fabric_event(ctx: RunContext) -> dict[str, Any] | None:
    events = ctx.store.of_type(EventType.FABRIC)
    return events[-1].data if events else None


def stored_fabric(
    data: dict[str, Any] | None,
) -> tuple[FabricTopology | None, LinkMatrix | None, tuple[LinkFinding, ...]]:
    """(topology, link matrix, link findings) of a FABRIC event payload."""
    if not data:
        return None, None, ()
    topology = (
        FabricTopology.model_validate(data["topology"])
        if data.get("topology")
        else None
    )
    links = LinkMatrix.model_validate(data["links"]) if data.get("links") else None
    findings = tuple(
        LinkFinding.model_validate(f) for f in data.get("link_findings", ())
    )
    return topology, links, findings


class SwitchSampler:
    """Switch-port counter snapshots around one load point (fail-soft)."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        data = fabric_event(ctx) or {}
        self.refs = tuple(SwitchPortRef.model_validate(r) for r in data.get("refs", ()))
        self._prober: FabricProber | None = None

    def snapshot(self, label: str) -> SwitchSnapshot | None:
        if not self.refs:
            return None
        if self.ctx.simulator is not None:
            return self.ctx.simulator.switch_snapshot(self.refs, label)
        if self._prober is None:
            self._prober = make_prober(self.ctx)
        return self._prober.snapshot_switch_ports(self.refs, label=label)

    @staticmethod
    def load(
        before: SwitchSnapshot | None, after: SwitchSnapshot | None
    ) -> tuple[SwitchPortLoad, ...]:
        if before is None or after is None:
            return ()
        try:
            return tuple(switch_load(before, after))
        except ValueError as exc:  # mismatched snapshots: keep the point
            logger.warning("switch load unavailable: %s", exc)
            return ()
