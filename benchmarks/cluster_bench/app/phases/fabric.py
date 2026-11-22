# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric: upstream switch topology, node-to-node paths and link tests.

identify (sysfs, MAD/SMP, LLDP tiers) -> topology -> traces -> topology with
hop counts -> perftest link matrix (round-robin pairs, every rail) ->
findings. The switch ports serving our HCAs are recorded so every load point
can snapshot their counters (``app/fabric_ops.SwitchSampler``). Runs after
the download and before the inter-node microbenchmarks; every step is
fail-soft, so a locked-down fabric only reduces coverage.
"""

from __future__ import annotations

import logging

from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.fabric_ops import (
    COLLECTION_ERRORS,
    make_prober,
    rail_map,
)
from benchmarks.cluster_bench.app.pipeline import PhaseRuntime
from benchmarks.cluster_bench.domain.fabric import (
    FabricNodeReport,
    LinkMatrix,
    TraceResult,
    analyze_link_matrix,
    build_topology,
    nominal_gbs_by_rail,
    switch_port_refs,
)
from benchmarks.cluster_bench.domain.schema import EventType

logger = logging.getLogger(__name__)


class FabricPhase:
    name = "fabric"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        try:
            self._probe(ctx)
        except COLLECTION_ERRORS as exc:
            # Fabric facts are evidence, not a precondition: never fail the run.
            logger.warning("fabric probing failed: %s", exc)
            ctx.note(f"fabric probing failed: {exc}")
            ctx.store.append(
                EventType.FABRIC,
                {"topology": None, "links": None, "link_findings": [], "refs": []},
            )

    def _probe(self, ctx: RunContext) -> None:
        cfg = ctx.config
        hw = ctx.require_hardware()
        nodes = ctx.node_names
        rails = rail_map(hw.inventory)
        sim = ctx.simulator
        prober = None if sim is not None else make_prober(ctx)

        ctx.status(step="fabric: switch identity (sysfs / MAD / LLDP)")
        reports: dict[str, FabricNodeReport] = (
            sim.fabric_reports()
            if sim is not None
            else prober.identify(nodes, lldp_wait_s=cfg.lldp_wait_s)
        )
        topology = build_topology(reports, rail_of=rails)
        ctx.status(step="fabric: paths between nodes")
        traces: list[TraceResult] = (
            sim.fabric_traces(topology)
            if sim is not None
            else prober.trace(topology, reports)
        )
        topology = build_topology(reports, traces, rail_of=rails)

        matrix: LinkMatrix | None = None
        if cfg.fabric_link_matrix and len(nodes) > 1:
            ctx.status(step="fabric: perftest link matrix (every pair, every rail)")
            matrix = (
                sim.fabric_link_matrix()
                if sim is not None
                else prober.link_matrix(reports, topology=topology)
            )
        findings = (
            analyze_link_matrix(
                matrix, nominal_gbs_by_rail(topology), hops=topology.hops
            )
            if matrix is not None and not matrix.skipped
            else []
        )
        refs = switch_port_refs(topology, via_node=ctx.node0)
        ctx.store.append(
            EventType.FABRIC,
            {
                "topology": topology.model_dump(mode="json"),
                "links": matrix.model_dump(mode="json") if matrix else None,
                "link_findings": [f.model_dump(mode="json") for f in findings],
                "refs": [r.model_dump(mode="json") for r in refs],
            },
        )
