# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speed-of-light microbenchmarks (KB evidence category 5).

Groups, each recorded once as a PEAK event so a resume skips it:
* ``p2p``         - intra-node GPU<->GPU copy matrix and host<->device PCIe
                    (usually already run during the download phase);
* ``collectives_intra`` / ``collectives_inter`` - NCCL all_reduce/all_gather bus
                    bandwidth within each node (overlaps the download) and across nodes;
* ``gdr``         - GPU-direct RDMA send/recv between two nodes per NIC count,
                    the reference for the NIC sweep's utilization numbers.

A failed group is noted and skipped: peaks make the bandwidth section
quantitative but are not needed to rank serving configurations.
"""

from __future__ import annotations

import logging
import shlex

from benchmarks.cluster_bench.adapters.microbench import MicrobenchRunner
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.domain.nicplan import feasible_counts, subsets_for_counts
from benchmarks.cluster_bench.domain.schema import EventType, PeakResult
from benchmarks.cluster_bench.ports import CommandError

logger = logging.getLogger(__name__)

GROUPS = ("p2p", "collectives_intra", "collectives_inter", "gdr")


def done_groups(ctx: RunContext) -> set[str]:
    return {str(e.data.get("group")) for e in ctx.store.of_type(EventType.PEAK)}


def record_peaks(ctx: RunContext, group: str, results: list[PeakResult]) -> None:
    ctx.store.append(
        EventType.PEAK,
        {"group": group, "results": [r.model_dump(mode="json") for r in results]},
    )


def runner_for(ctx: RunContext) -> MicrobenchRunner:
    inventory = ctx.require_hardware().inventory
    return MicrobenchRunner.from_inventory(
        ctx.cluster,
        ctx.docker,
        ctx.config.image,
        ctx.config.workdir,
        inventory,
        local_dir=ctx.exp_dir / "microbench",
    )


def run_group(ctx: RunContext, group: str) -> None:
    """Run one microbenchmark group unless it already ran."""
    if group in done_groups(ctx):
        return
    inventory = ctx.inventory
    if inventory is None:
        raise RuntimeError("microbenchmarks need the hardware inventory")
    nodes = ctx.node_names
    # Result files live in their own directory: ``local_dir`` (exp_dir/microbench)
    # holds per-node copies and container logs, so the two must not collide.
    results_dir = ctx.exp_dir / "microbench" / "results"
    remote = str(results_dir / "peak.json")
    ctx.status(step=f"microbenchmark: {group}")
    if ctx.simulator is not None:
        record_peaks(ctx, group, ctx.simulator.peaks(group))
        return
    runner = runner_for(ctx)
    try:
        mkdir = f"mkdir -p {shlex.quote(str(results_dir))}"
        ctx.cluster.fanout(nodes, lambda node: ctx.cluster.run(node, mkdir))
        if group == "p2p":
            per_node = ctx.cluster.fanout(nodes, lambda node: runner.p2p(node, remote))
            results = [peak for node in nodes for peak in per_node[node]]
        elif group == "collectives_intra":
            results = runner.collectives(nodes, remote, intra=True, inter=False)
        elif group == "collectives_inter":
            results = (
                runner.collectives(nodes, remote, intra=False, inter=True)
                if len(nodes) > 1
                else []
            )
        elif group == "gdr":
            if len(nodes) < 2:
                results = []
            else:
                counts = feasible_counts(inventory, ctx.config.nic_counts)
                subsets = subsets_for_counts(inventory, counts)
                results = runner.gdr_pairs(nodes[0], nodes[1], subsets, remote)
        else:
            raise ValueError(f"unknown microbenchmark group {group!r}")
    except (CommandError, ValueError, KeyError, FileNotFoundError) as exc:
        ctx.note(f"microbenchmark {group} failed: {exc}")
        record_peaks(ctx, group, [])
        return
    record_peaks(ctx, group, results)


class MicrobenchPhase:
    name = "microbench"

    def run(self, ctx: RunContext, rt: object) -> None:
        for group in GROUPS:
            run_group(ctx, group)
