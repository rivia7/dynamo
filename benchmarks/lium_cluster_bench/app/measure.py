# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute one planned measurement point end to end.

deploy (if needed) -> telemetry on -> AIPerf -> telemetry off -> liveness
check -> parse -> bandwidth analysis -> audit -> seal -> POINT event.

An interrupted attempt leaves a point directory without a POINT event; the
next attempt renames it ``<id>.aborted-<ts>`` and records POINT_ABORTED, so
the evidence is kept but never ranked.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

from benchmarks.lium_cluster_bench.adapters import aiperf as aiperf_adapter
from benchmarks.lium_cluster_bench.adapters.telemetry import (
    TelemetryCollector,
    TelemetryFormatError,
    load_samples,
    load_service_state,
)
from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.app.deployment import DeploymentSession
from benchmarks.lium_cluster_bench.app.fabric_ops import SwitchSampler
from benchmarks.lium_cluster_bench.app.history import peaks as stored_peaks
from benchmarks.lium_cluster_bench.app.model_ops import point_mfu, stored_arch
from benchmarks.lium_cluster_bench.app.store import seal_dir
from benchmarks.lium_cluster_bench.domain.analysis import audit_point
from benchmarks.lium_cluster_bench.domain.bandwidth import (
    PointBandwidth,
    analyze_point_bandwidth,
)
from benchmarks.lium_cluster_bench.domain.gpuactivity import gpu_role_map
from benchmarks.lium_cluster_bench.domain.loadstress import (
    PointStress,
    analyze_point_stress,
    energy_per_token_j,
)
from benchmarks.lium_cluster_bench.domain.nvlink import NVLINK_LEAK_GBS, nvlink_p99
from benchmarks.lium_cluster_bench.domain.pointstate import (
    PointState,
    max_running_by_role,
    point_state,
)
from benchmarks.lium_cluster_bench.domain.requests import RequestSummary
from benchmarks.lium_cluster_bench.domain.schema import (
    EventType,
    LaunchPlan,
    PointMetrics,
    PointResult,
)
from benchmarks.lium_cluster_bench.domain.search import RunPoint
from benchmarks.lium_cluster_bench.domain.servicestate import (
    ServiceState,
    classify_bottleneck,
    endpoint_descriptors,
)
from benchmarks.lium_cluster_bench.domain.switchload import summarize_switch_load
from benchmarks.lium_cluster_bench.fsutil import atomic_write_text

logger = logging.getLogger(__name__)

ACCEPT_KEY_MARKERS = ("accept_len", "accept_length", "acceptance_length")


def pairing_json(bandwidth: PointBandwidth) -> dict[str, dict[str, str]]:
    """GPU->HCA pairing the bandwidth analysis used, JSON-ready."""
    return {
        node: {str(gpu): hca for gpu, hca in pairs.items()}
        for node, pairs in bandwidth.pairing.items()
    }


def served_context(plan: LaunchPlan, default: int) -> int:
    """Smallest context length any serving instance actually renders.

    A ladder rung may lower ``context-length`` (e.g. H100 OOM fallback);
    AIPerf must then drop the traces that no longer fit.
    """
    lengths = [
        int(cfg["context-length"])
        for cfg in plan.rendered_configs.values()
        if "context-length" in cfg
    ]
    return min(lengths) if lengths else default


def point_dir(ctx: RunContext, point_id: str) -> Path:
    return ctx.exp_dir / "points" / point_id


def _archive_aborted(ctx: RunContext, path: Path, point_id: str) -> None:
    if not path.exists():
        return
    target = path.with_name(f"{path.name}.aborted-{int(ctx.clock.now())}")
    path.rename(target)
    ctx.store.append(
        EventType.POINT_ABORTED, {"point_id": point_id, "dir": str(target)}
    )


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value > 0:
            return float(value)
    return None


def _series_averages(metric: Any) -> list[float]:
    """``{"series": [{"stats": {"avg": x}}, ...]}`` (AIPerf server-metrics JSON)."""
    if not isinstance(metric, dict):
        return []
    values = []
    for series in metric.get("series") or []:
        stats = series.get("stats") if isinstance(series, dict) else None
        avg = _numeric(stats.get("avg")) if isinstance(stats, dict) else None
        if avg is not None:
            values.append(avg)
    return values


def find_accept_len(obj: Any) -> list[float]:
    """Speculative acceptance values anywhere in a server-metrics export.

    AIPerf stores ``metrics[<name>]["series"][i]["stats"]["avg"]``; the SGLang
    metric name is UNVERIFIED for 0.5.18, so names are matched by substring
    and the report labels the value accordingly. Plain numeric values under a
    matching key are accepted too.
    """
    found: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if any(m in str(key).lower() for m in ACCEPT_KEY_MARKERS):
                number = _numeric(value)
                if number is not None:
                    found.append(number)
                else:
                    found.extend(_series_averages(value))
            else:
                found.extend(find_accept_len(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_accept_len(item))
    return found


def accept_len_from_exports(paths: Iterable[Path]) -> float | None:
    values: list[float] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                values.extend(find_accept_len(json.load(fh)))
        except (json.JSONDecodeError, OSError):
            continue
    return sum(values) / len(values) if values else None


def accept_length(state: PointState | None, pdir: Path) -> float | None:
    """Request-weighted speculative acceptance length of one point.

    Generated tokens per verify step from the worker counters (decode, else
    agg role) when both exist; otherwise the mean of the
    ``spec_accept_length`` gauge, a mean of log-interval means that weights
    every scrape equally (deep-research finding, AIPerf 0.13.0 docs).
    """
    weighted = state.derived("spec_tokens_per_verify") if state else None
    if weighted is not None and weighted > 0:
        return weighted
    return accept_len_from_exports(sorted(pdir.rglob("server_metrics_export*.json")))


def _with_note(state: PointState | None, note: str) -> PointState:
    base = state or PointState()
    return base.model_copy(update={"notes": (*base.notes, note)})


class Measurer:
    def __init__(self, ctx: RunContext, session: DeploymentSession):
        self.ctx = ctx
        self.session = session
        self.telemetry = TelemetryCollector(
            ctx.cluster,
            ctx.python,
            f"{ctx.src_dir}/benchmarks/lium_cluster_bench/agents/telemetry_agent.py",
            control_interface=ctx.config.control_interface,
        )

    def run(self, action: RunPoint) -> PointResult | None:
        """Returns the PointResult, or None when the deployment failed."""
        deploy = self.session.ensure(action.candidate)
        if not deploy.ok:
            return None
        plan = self.session.plan
        if plan is None:
            raise RuntimeError("deployment reported ok without a plan")
        key = action.key
        pdir = point_dir(self.ctx, key.id)
        _archive_aborted(self.ctx, pdir, key.id)
        pdir.mkdir(parents=True)
        spec = self.ctx.config.bench_specs[action.spec]
        self.ctx.status(
            step=f"measuring {key.cand_key} c={key.concurrency} ({action.purpose})",
            reason=action.reason,
        )
        atomic_write_text(
            pdir / "action.json",
            json.dumps({"key": key.model_dump(), "reason": action.reason}, indent=2),
        )

        switches = SwitchSampler(self.ctx)
        before = switches.snapshot(f"{key.id}-before")
        started = self.ctx.clock.now()
        state: PointState | None
        requests: RequestSummary | None
        if self.ctx.simulator is not None:
            metrics, bandwidth = self.ctx.simulator.measure(
                self.ctx, action, plan, pdir
            )
            alive = True
            state = self.ctx.simulator.point_state(action, plan, metrics)
            requests = self.ctx.simulator.requests(
                action,
                plan,
                metrics,
                self.ctx.config.slo_tiers,
                arch=stored_arch(self.ctx),
            )
        else:
            metrics, bandwidth, alive, state = self._measure_real(action, plan, pdir)
            requests = self._requests(pdir, plan.gpu_count, metrics, state)
        ended = self.ctx.clock.now()
        loads = switches.load(before, switches.snapshot(f"{key.id}-after"))
        if loads:
            atomic_write_text(
                pdir / "switch_load.json",
                json.dumps([x.model_dump(mode="json") for x in loads], indent=1),
            )

        audit = audit_point(
            metrics,
            spec=spec,
            concurrency=action.concurrency,
            expected_aiperf_version=self.ctx.config.aiperf_version,
            engagement=None,  # proven at deploy time; a failed proof never reaches here
            server_alive=alive,
            telemetry_coverage=bandwidth.coverage if bandwidth else None,
            nic_restricted=action.candidate.nic_count is not None,
            nic_engaged=bandwidth.nic_engaged if bandwidth else None,
            nvlink_off=action.candidate.p2p != "nvlink",
            nvlink_p99_gbs=nvlink_p99(bandwidth.stats) if bandwidth else None,
            nvlink_leak_gbs=NVLINK_LEAK_GBS,
        )
        tps = (
            (metrics.output_tps / plan.gpu_count)
            if metrics and metrics.output_tps
            else None
        )
        accept = accept_length(state, pdir) if action.candidate.mtp else None
        if action.candidate.mtp and accept is None and metrics is not None:
            state = _with_note(
                state,
                "speculative decoding is configured but no acceptance was "
                "measured (MTP inactive, or the spec metrics are missing)",
            )
        result = PointResult(
            key=key,
            candidate=action.candidate,
            gpu_count=plan.gpu_count,
            metrics=metrics,
            audit=audit,
            tps_per_gpu=tps,
            bandwidth=bandwidth.stats if bandwidth else (),
            cross_node_gbs=bandwidth.cross_node_gbs if bandwidth else None,
            nic_effective=bandwidth.nic_effective if bandwidth else None,
            nic_engaged=bandwidth.nic_engaged if bandwidth else None,
            staging_suspected=bandwidth.staging_suspected if bandwidth else False,
            telemetry_coverage=bandwidth.coverage if bandwidth else None,
            telemetry_notes=tuple(bandwidth.notes) if bandwidth else (),
            accept_len=accept,
            switch_load=summarize_switch_load(loads),
            state=state,
            requests=requests,
            mfu=point_mfu(
                plan=plan,
                gpu=self.ctx.require_hardware().gpu,
                variant=self.ctx.require_hardware().gpu.weights_variant,
                requests=requests,
                state=state,
            ),
            artifact_dir=str(pdir),
            started_at=started,
            ended_at=ended,
        )
        atomic_write_text(pdir / "result.json", result.model_dump_json(indent=2))
        if bandwidth is not None:
            atomic_write_text(
                pdir / "bandwidth.json",
                json.dumps(
                    {
                        "coverage": bandwidth.coverage,
                        "notes": list(bandwidth.notes),
                        "pairing": pairing_json(bandwidth),
                    },
                    indent=2,
                ),
            )
        seal_dir(
            pdir,
            {
                "point_id": key.id,
                "cand_key": key.cand_key,
                "aiperf": self.ctx.config.aiperf_version,
            },
        )
        self.ctx.store.append(
            EventType.POINT, {"result": result.model_dump(mode="json")}
        )
        self.ctx.status(
            last_point={
                "cand_key": key.cand_key,
                "concurrency": key.concurrency,
                "purpose": key.purpose,
                "tps_per_gpu": tps,
                "status": audit.status,
            }
        )
        return result

    def _requests(
        self,
        pdir: Path,
        gpus: int,
        metrics: PointMetrics | None,
        state: PointState | None = None,
    ) -> RequestSummary | None:
        """Goodput / turn / session view of the point's AIPerf records."""
        window = None
        if metrics and metrics.window_start_ns and metrics.window_end_ns:
            window = (metrics.window_start_ns / 1e9, metrics.window_end_ns / 1e9)
        try:
            summary = aiperf_adapter.load_request_summary(
                pdir / "aiperf",
                gpus=gpus,
                tiers=self.ctx.config.slo_tiers,
                window=window,
                arch=stored_arch(self.ctx),
                cached_fraction=state.derived(
                    "cached_token_fraction", roles=("prefill", "agg")
                )
                if state
                else None,
            )
        except (OSError, ValueError, TypeError) as exc:
            self.ctx.note(f"request records of {pdir.name} unreadable: {exc}")
            return None
        if summary is not None:
            atomic_write_text(pdir / "requests.json", summary.model_dump_json(indent=1))
        return summary

    def _measure_real(
        self, action: RunPoint, plan: LaunchPlan, pdir: Path
    ) -> tuple[PointMetrics | None, PointBandwidth | None, bool, PointState | None]:
        hw = self.ctx.require_hardware()
        inventory, gpu, deployment, model_dir = (
            hw.inventory,
            hw.gpu,
            hw.deployment,
            hw.model_dir,
        )
        spec = self.ctx.config.bench_specs[action.spec]
        remote_tel = str(pdir / "telemetry")
        # GPM and Nsight Systems may compete for the GPU performance counters.
        self.telemetry.gpm = action.candidate.diag != "nsys"
        try:
            handle = self.telemetry.start(
                self.ctx.node_names, remote_tel, action.key.id
            )
        except RuntimeError as exc:
            # Telemetry failure degrades the bandwidth evidence, not the
            # throughput measurement: run without it and say so.
            self.ctx.note(f"telemetry did not start for {action.key.id}: {exc}")
            handle = None
        aiperf_dir = pdir / "aiperf"
        command = aiperf_adapter.build_profile_command(
            aiperf_bin=self.ctx.aiperf_bin,
            served_model=plan.served_model_name,
            tokenizer=model_dir,
            url=f"http://127.0.0.1:{self.ctx.config.frontend_port}",
            dataset=self.ctx.config.dataset,
            spec=spec,
            concurrency=action.concurrency,
            served_context=served_context(plan, deployment.served_context),
            artifact_dir=str(aiperf_dir),
            server_metrics_urls=plan.metrics_urls,
            cpu_count=inventory.nodes[0].cpu_count,
            hf_home=self.ctx.hf_home,
        )
        atomic_write_text(pdir / "aiperf_command.sh", command + "\n")
        try:
            limit = int(aiperf_adapter.run_timeout_s(spec))
            # SIGINT lets AIPerf cancel its workers and export; KILL after 120 s.
            # The outer timeout only fires if `timeout` itself hangs.
            res = self.ctx.cluster.run(
                self.ctx.node0,
                f"timeout -s INT -k 120 {limit} {command}",
                timeout=limit + 300,
                check=False,
            )
            self.ctx.cluster.run(
                self.ctx.node0, "pkill -KILL -f '[a]iperf profile' || true", check=False
            )
        finally:
            files = self.telemetry.stop(handle, pdir / "telemetry") if handle else {}
        atomic_write_text(
            pdir / "aiperf.log", self.ctx.redactor(res.stdout + "\n" + res.stderr)
        )
        alive = self.session.alive()
        metrics = aiperf_adapter.load_point_metrics(aiperf_dir)
        if metrics is None:
            return None, None, alive, None
        window = None
        if metrics.window_start_ns and metrics.window_end_ns:
            window = (metrics.window_start_ns / 1e9, metrics.window_end_ns / 1e9)
        samples = {}
        for node, path in files.items():
            try:
                samples[node] = load_samples(path)
            except TelemetryFormatError as exc:
                self.ctx.note(f"telemetry from {node} unreadable: {exc}")
        bandwidth = None
        if window is not None and samples:
            bandwidth = analyze_point_bandwidth(
                samples,
                window,
                inventory,
                plan.nic_selection,
                stored_peaks(self.ctx.store),
                gpu,
            )
        state = self._point_state(
            plan, pdir, aiperf_dir, window, samples, bandwidth, metrics
        )
        return metrics, bandwidth, alive, state

    def _point_state(
        self,
        plan: LaunchPlan,
        pdir: Path,
        aiperf_dir: Path,
        window: tuple[float, float] | None,
        samples: dict[str, Any],
        bandwidth: PointBandwidth | None,
        metrics: PointMetrics,
    ) -> PointState | None:
        """Service state, stress and bottleneck of one point (fail-soft)."""
        notes: list[str] = []
        service: ServiceState | None = load_service_state(
            aiperf_dir, endpoint_descriptors(plan.containers), window=window
        )
        stress: PointStress | None = None
        if window is not None and samples:
            try:
                stress = analyze_point_stress(
                    {n: s.stress for n, s in samples.items()},
                    window,
                    gpm_by_node={n: s.gpm for n, s in samples.items()},
                    gpu_roles=gpu_role_map(plan.containers),
                )
            except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                notes.append(f"stress analysis failed: {exc}")
        ib_utils = [
            s.util_vs_peak
            for s in (bandwidth.stats if bandwidth else ())
            if s.plane == "ib" and s.util_vs_peak is not None
        ]
        lifecycle = self.session.lifecycle
        activity = stress.activity if stress else None
        try:
            bottleneck = classify_bottleneck(
                service,
                max_running_requests=max_running_by_role(lifecycle.roles)
                if lifecycle
                else None,
                gpu_util_mean=activity.gr_pct_mean if activity else None,
                ib_util_max=max(ib_utils) if ib_utils else None,
                throttled_fraction=stress.throttled_fraction if stress else None,
                ib_congestion=stress.xmit_wait_fraction_max if stress else None,
                activity=activity.roles if activity else None,
            )
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            notes.append(f"bottleneck classification failed: {exc}")
            bottleneck = None
        energy = (
            energy_per_token_j(stress, metrics.total_output_tokens) if stress else None
        )
        atomic_write_text(pdir / "servicestate.json", service.model_dump_json(indent=1))
        if stress is not None:
            atomic_write_text(pdir / "stress.json", stress.model_dump_json(indent=1))
        return point_state(service, stress, bottleneck, energy, tuple(notes))
