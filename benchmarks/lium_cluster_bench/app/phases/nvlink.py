# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVLink comparison: the leader again with GPU-GPU traffic on PCIe.

Runs after the confirmatory rerun, so every ranked result is in before any
node's NVLink state changes. Order: tear down, switch NVLink off on every
node (hardware if possible, proven by a copy benchmark; else NCCL without
P2P), deploy the same configuration with NVLink-only fast paths off, measure
one point at the leader's concurrency, tear down, restore NVLink. The
restore runs in ``finally`` and again on resume after a crash.
"""

from __future__ import annotations

from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.app.history import build_history, make_budget
from benchmarks.lium_cluster_bench.app.nvlink_ops import NvlinkControl
from benchmarks.lium_cluster_bench.app.phases.planned import (
    search_context,
    unsupported_verdict,
)
from benchmarks.lium_cluster_bench.app.pipeline import PhaseRuntime, drive
from benchmarks.lium_cluster_bench.domain import search
from benchmarks.lium_cluster_bench.domain.nvlink import NVLINK_OFF_FLAGS
from benchmarks.lium_cluster_bench.domain.schema import EventType


def _skip(ctx: RunContext, reason: str) -> None:
    ctx.store.append(
        EventType.DECISION,
        {"kind": "phase_result", "phase": "nvlink", "reason": reason},
    )


def _image_flags_missing(ctx: RunContext) -> list[str]:
    path = ctx.exp_dir / "preflight" / "dynamo_sglang_help.txt"
    if not path.exists():
        return []
    help_text = path.read_text(errors="replace")
    return [f for f in NVLINK_OFF_FLAGS if f not in help_text]


class NvlinkPhase:
    name = "nvlink"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        if unsupported_verdict(ctx):
            return
        hw = ctx.require_hardware()
        if not hw.gpu.nvlink:
            _skip(ctx, "PCIe-only GPUs: there is no NVLink to switch off")
            return
        control = NvlinkControl(ctx)
        control.restore_left_off()
        if not ctx.config.nvlink_ablation:
            _skip(ctx, "NVLink comparison disabled (--no-nvlink-ablation)")
            return
        missing = _image_flags_missing(ctx)
        if missing:
            _skip(ctx, f"image lacks {', '.join(missing)}")
            return
        sc = search_context(ctx)
        rt.plan = sc
        history = build_history(ctx.store, rt.session.deployed_key)
        budget = make_budget(
            ctx.store,
            history,
            ctx.config.budget,
            ctx.config.bench_specs,
            "nvlink",
            sc,
            ctx.clock.now(),
            accuracy=False,
            diagnose_s=ctx.config.diagnose.window_s,
        )
        base, reason = search.nvlink_plan(sc, history, budget)
        if base is None:
            _skip(ctx, reason)
            return
        ctx.store.append(
            EventType.DECISION,
            {
                "kind": "nvlink_base",
                "phase": "nvlink",
                "point_id": base.key.id,
                "family_id": base.candidate.family_id,
                "reason": reason,
            },
        )
        ctx.status(step="switching NVLink off on every node")
        rt.session.teardown()
        try:
            switch = control.switch_off()
            drive(
                ctx,
                rt,
                "nvlink",
                lambda h, b: search.decide_nvlink(sc, h, b, switch.path),
            )
        finally:
            rt.session.teardown()
            ctx.status(step="restoring NVLink")
            control.restore(reason="NVLink comparison done")
