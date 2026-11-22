# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finalize: tear down, fold events into the summary, render the Chinese
report, and export the experiment into the node-local knowledge base (the
laptop controller merges it into the user's KB after pulling)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.app.facts_ops import collect_host
from benchmarks.lium_cluster_bench.app.nvlink_ops import NvlinkControl, nvlink_events
from benchmarks.lium_cluster_bench.app.pipeline import PhaseRuntime
from benchmarks.lium_cluster_bench.app.summary import build_summary, plan_lookup_for
from benchmarks.lium_cluster_bench.domain.nvlink import left_off
from benchmarks.lium_cluster_bench.domain.schema import EventType
from benchmarks.lium_cluster_bench.kb.writer import export_experiment
from benchmarks.lium_cluster_bench.report.render import render_report


class FinalizePhase:
    name = "finalize"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        rt.session.teardown()
        if left_off(nvlink_events(ctx)) is not None:
            # A crash between the NVLink switch and its restore.
            NvlinkControl(ctx).restore(reason="finalize")
        if not any(
            e.data.get("tag") == "after" for e in ctx.store.of_type(EventType.FACTS)
        ):
            # Health counters after every container is gone: what grew under load.
            collect_host(ctx, "after")
        ctx.status(step="writing summary, report and knowledge base")
        summary = build_summary(ctx)
        ctx.save_json("analysis/summary.json", summary.model_dump(mode="json"))
        lookup = plan_lookup_for(ctx.exp_dir)
        md_path, html_path = render_report(summary, ctx.exp_dir / "report", lookup)
        kb_root = Path(ctx.config.workdir) / "kb"
        export_experiment(summary, ctx.exp_dir, kb_root, lookup)
        ctx.status(
            report_md=str(md_path),
            report_html=str(html_path),
            kb_root=str(kb_root),
            verdict=summary.verdict,
        )
