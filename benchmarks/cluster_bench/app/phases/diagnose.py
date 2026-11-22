# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostic window after the confirmation (``--diagnose``).

Redeploys the leader as a diagnostic variant: SGLang's opt-in performance
metrics (``--enable-mfu-metrics``, the device timer, EPLB balancedness for
MoE models) and, when an image with Nsight Systems is available (the pinned
image, else a derived ``cb-diag/...`` image built on every node), workers
under ``nsys profile --capture-range=cudaProfilerApi``. A short AIPerf load
runs at the leader's concurrency; after its warm-up every instance leader
gets ``POST /engine/control/start_profile`` for a few forward steps. The
reports are summarized with ``nsys stats`` into a kernel-time breakdown per
role. Throughput inside the window is never an official result (nsys node
graph tracing slows the workers); the window is skipped when the budget
cannot cover it. Every step is fail-soft and leaves a note.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence

from benchmarks.cluster_bench.adapters.diagnostics import (
    DEFAULT_WINDOW,
    NsysDiagnostics,
    expected_reports,
    profile_targets,
)
from benchmarks.cluster_bench.adapters.docker import experiment_labels
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.fabric_ops import COLLECTION_ERRORS
from benchmarks.cluster_bench.app.history import build_history, make_budget
from benchmarks.cluster_bench.app.nvlink_ops import nvlink_events
from benchmarks.cluster_bench.app.phases.planned import unsupported_verdict
from benchmarks.cluster_bench.app.pipeline import PhaseRuntime
from benchmarks.cluster_bench.domain import search
from benchmarks.cluster_bench.domain.kernels import (
    KernelBreakdown,
    KernelReport,
    merge_breakdowns,
)
from benchmarks.cluster_bench.domain.nvlink import left_off
from benchmarks.cluster_bench.domain.schema import (
    DiagnosticsResult,
    EventType,
    LaunchPlan,
    PointResult,
)
from benchmarks.cluster_bench.domain.search import RunPoint

logger = logging.getLogger(__name__)

WINDOW = DEFAULT_WINDOW  # one capture window per worker start (nsys stop mode)
# Engine-internal values kept from the diagnostic point (role.<key>).
ENGINE_DERIVED = ("est_tflops_per_gpu", "est_mem_gbs_per_gpu", "forward_busy_fraction")
ENGINE_VALUES = ("eplb_balancedness",)


def _done(ctx: RunContext, reason: str) -> None:
    ctx.store.append(
        EventType.DECISION,
        {"kind": "phase_result", "phase": "diagnose", "reason": reason},
    )


def engine_values(point: PointResult | None) -> dict[str, float]:
    """MFU estimate, HBM traffic estimate, GPU-busy share, EPLB balancedness."""
    state = point.state if point else None
    out: dict[str, float] = {}
    if state is None:
        return out
    for role, brief in sorted(state.roles.items()):
        for key in ENGINE_DERIVED:
            if key in brief.derived:
                out[f"{role}.{key}"] = brief.derived[key]
        for key in ENGINE_VALUES:
            value = brief.get(key)
            if value is not None:
                out[f"{role}.{key}"] = value
    return out


def role_breakdowns(
    reports: Sequence[KernelReport], plan: LaunchPlan
) -> dict[str, KernelBreakdown]:
    """Kernel breakdown per serving role, merged over that role's reports."""
    role_of = {c.name: c.role for c in plan.containers}
    grouped: dict[str, list[KernelBreakdown]] = {}
    for report in reports:
        if report.breakdown is not None:
            role = role_of.get(report.source, "unknown")
            grouped.setdefault(role, []).append(report.breakdown)
    return {role: merge_breakdowns(items) for role, items in sorted(grouped.items())}


class DiagnosePhase:
    name = "diagnose"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        cfg = ctx.config.diagnose
        if not cfg.enabled:
            _done(ctx, "diagnostic window not requested (--diagnose)")
            return
        if unsupported_verdict(ctx):
            _done(ctx, "diagnostic window skipped: GPU type unsupported")
            return
        if left_off(nvlink_events(ctx)) is not None:
            _done(ctx, "diagnostic window skipped: NVLink could not be restored")
            return
        history = build_history(ctx.store, rt.session.deployed_key)
        leader = search.final_leader(history)
        if leader is None:
            _done(ctx, "diagnostic window skipped: no leader")
            return
        budget = make_budget(
            ctx.store,
            history,
            ctx.config.budget,
            ctx.config.bench_specs,
            "diagnose",
            rt.plan,
            ctx.clock.now(),
        )
        if not budget.affordable(budget.deploy_s + cfg.window_s):
            reason = "budget skipped the diagnostic window"
            ctx.note(reason)
            _done(ctx, reason)
            return
        result = self._window(ctx, rt, leader)
        ctx.save_json("analysis/diagnostics.json", result.model_dump(mode="json"))
        ctx.store.append(
            EventType.DIAGNOSTICS, {"result": result.model_dump(mode="json")}
        )
        _done(ctx, f"diagnostic window recorded (nsys {result.nsys})")

    # ------------------------------------------------------------------ steps
    def _window(
        self, ctx: RunContext, rt: PhaseRuntime, leader: PointResult
    ) -> DiagnosticsResult:
        cfg = ctx.config.diagnose
        notes: list[str] = []
        diag = NsysDiagnostics(
            ctx.cluster,
            workdir=ctx.config.workdir,
            results_dir=ctx.exp_dir / "diagnostics",
            labels=experiment_labels(ctx.config.exp_id),
        )
        image, nsys_state, version = None, "off", None
        if cfg.nsys:
            ctx.status(step="diagnostics: preparing an image with Nsight Systems")
            images = diag.ensure_image(ctx.node_names, ctx.config.image)
            bad = {n: i for n, i in images.items() if not i.ok}
            tags = {i.image for i in images.values() if i.ok}
            if bad or len(tags) != 1:
                nsys_state = "unavailable"
                first = next(iter(bad.values()), None)
                why = "; ".join(first.notes) if first and first.notes else "no image"
                notes.append(f"Nsight Systems unavailable on {len(bad)} node(s): {why}")
            else:
                image = tags.pop()
                nsys_state = "ok"
                info = next(iter(images.values())).nsys
                version = info.version if info else None
        cand = leader.candidate.model_copy(
            update={"diag": "nsys" if nsys_state == "ok" else "metrics"}
        )
        result = DiagnosticsResult(
            cand_key=cand.key,
            family_id=cand.family_id,
            concurrency=leader.key.concurrency,
            nsys=nsys_state,
            nsys_version=version,
        )
        deploy = rt.session.ensure(cand, image=image)
        if not deploy.ok or rt.session.plan is None:
            notes.append(f"diagnostic deployment failed: {deploy.failure or 'unknown'}")
            return result.model_copy(update={"notes": tuple(notes)})
        plan = rt.session.plan
        action = RunPoint(
            cand,
            leader.key.concurrency,
            "diagnose",
            "diagnose",
            reason="diagnostic window load (not an official result)",
        )
        point, triggered = self._load_and_capture(
            ctx, rt, diag, action, plan, cand.diag == "nsys"
        )
        notes.extend(triggered)
        reports: list[KernelReport] = []
        if cand.diag == "nsys":
            reports = self._collect(ctx, diag, plan, image or ctx.config.image, notes)
        rt.session.teardown()
        kernels = role_breakdowns(reports, plan)
        for report in reports:
            notes.extend(report.notes)
        if cand.diag == "nsys" and not kernels:
            notes.append("no kernel breakdown: no readable Nsight Systems report")
        return result.model_copy(
            update={
                "kernels": {r: b.model_dump(mode="json") for r, b in kernels.items()}
                or None,
                "engine": engine_values(point),
                "output_tps": point.metrics.output_tps
                if point and point.metrics
                else None,
                "notes": tuple(dict.fromkeys(notes)),
            }
        )

    def _load_and_capture(
        self,
        ctx: RunContext,
        rt: PhaseRuntime,
        diag: NsysDiagnostics,
        action: RunPoint,
        plan: LaunchPlan,
        capture: bool,
    ) -> tuple[PointResult | None, list[str]]:
        """Run the load; mid-way ask every instance leader to profile."""
        notes: list[str] = []
        spec = ctx.config.bench_specs[action.spec]
        targets = profile_targets(plan.containers) if capture else []
        if ctx.simulator is not None or not targets:
            point = rt.measurer.run(action)
            for c in targets:
                self._trigger(ctx, diag, c.node, c.system_port, notes)
            return point, notes
        box: dict[str, PointResult | None] = {}
        failure: list[BaseException] = []

        def load() -> None:
            try:
                box["point"] = rt.measurer.run(action)
            except COLLECTION_ERRORS as exc:  # reported; the phase goes on
                failure.append(exc)

        worker = threading.Thread(target=load, name="cb-diag-load", daemon=True)
        worker.start()
        # The measurer's AIPerf configure step plus the spec warm-up, then a
        # quarter of the measured duration: steady-state decode batches.
        ctx.clock.sleep(240.0 + spec.warmup_estimate_s + spec.duration_s / 4)
        for c in targets:
            self._trigger(ctx, diag, c.node, c.system_port, notes)
        worker.join()
        if failure:
            notes.append(f"diagnostic load failed: {failure[0]}")
        return box.get("point"), notes

    @staticmethod
    def _trigger(
        ctx: RunContext,
        diag: NsysDiagnostics,
        node: str,
        port: int | None,
        notes: list[str],
    ) -> None:
        if port is None:
            return
        trig = diag.trigger(node, port, num_steps=ctx.config.diagnose.steps)
        if not trig.ok:
            notes.append(
                f"start_profile on {node}:{port} failed: {trig.message or trig.http_status}"
            )

    def _collect(
        self,
        ctx: RunContext,
        diag: NsysDiagnostics,
        plan: LaunchPlan,
        image: str,
        notes: list[str],
    ) -> list[KernelReport]:
        expected = expected_reports(
            plan.containers,
            workdir=ctx.config.workdir,
            exp_id=ctx.config.exp_id,
            window=WINDOW,
        )
        ready = diag.wait_reports(
            {node: [path for _, path, _ in items] for node, items in expected.items()},
            clock=ctx.clock,
        )
        missing = [r.path for files in ready.values() for r in files if not r.ready]
        if missing:
            notes.append(f"{len(missing)} Nsight Systems report(s) not ready")
        gpus = {
            path: (name, n) for items in expected.values() for name, path, n in items
        }

        def summarize(node: str) -> list[KernelReport]:
            out = []
            for report in ready.get(node, ()):
                if not report.ready:
                    continue
                name, devices = gpus.get(report.path, (None, None))
                out.append(
                    diag.stats(
                        node,
                        report.path,
                        image=image,
                        source=name,
                        expected_devices=devices,
                    )
                )
            return out

        ctx.status(step="diagnostics: nsys stats")
        per_node: Mapping[str, list[KernelReport]] = ctx.cluster.fanout(
            list(expected), summarize
        )
        return [r for node in sorted(per_node) for r in per_node[node]]
