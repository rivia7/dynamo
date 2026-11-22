# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf 0.13.0: AgentX replay of the SemiAnalysis Claude Code corpus.

Command shape follows recipes/nemotron-3.5-super-vl-120b-a12b/perf/perf.yaml
(weka trace + inferencex-agentx-mvp), with facts verified in the 0.13.0
wheel:

* warmup is ``--agentic-cache-warmup-duration`` (``--warmup-requests-per-lane``
  does not exist in 0.13.0);
* every run gets a fresh uuid ``benchmark_id`` and the scenario cache-busts
  first-turn prefixes with it, so consecutive runs cannot hit each other's
  prefix cache and no worker restart is needed between concurrency points;
* ``--max-context-length`` drops whole traces whose recorded peak exceeds
  it (no truncation);
* ``output_token_throughput`` is total OSL / duration, and OSL already
  includes reasoning tokens.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.lium_cluster_bench.domain.modelarch import ModelArch
from benchmarks.lium_cluster_bench.domain.requests import (
    RequestSummary,
    SloTier,
    parse_records,
    summarize_requests,
)
from benchmarks.lium_cluster_bench.domain.schema import BenchSpec, PointMetrics

SCENARIO = "inferencex-agentx-mvp"
# AIPerf --public-dataset alias -> HF dataset repo (aiperf/plugin/plugins.yaml, 0.13.0).
PUBLIC_DATASET_REPOS = {
    "semianalysis_cc_traces_weka_062126": "semianalysisai/cc-traces-weka-062126",
    "semianalysis_cc_traces_weka_062126_256k": "semianalysisai/cc-traces-weka-062126-256k",
}
CONTEXT_MARGIN_TOKENS = 2048  # chat template + tool schema overhead
REQUEST_TIMEOUT_S = 3900

REQUIRED_FLAGS = (
    "--public-dataset",
    "--scenario",
    "--unsafe-override",
    "--agentic-cache-warmup-duration",
    "--max-context-length",
    "--trajectory-start-min-ratio",
    "--trajectory-start-max-ratio",
    "--no-fixed-schedule",
    "--use-server-token-count",
    "--benchmark-duration",
    "--workers-max",
    "--failed-request-threshold",
    "--server-metrics",
    "--server-metrics-formats",
    "--export-level",
    "--artifact-dir",
)


def missing_flags(
    help_text: str, required: Sequence[str] = REQUIRED_FLAGS
) -> list[str]:
    """Flags absent from ``aiperf profile --help`` output."""
    return [flag for flag in required if flag not in help_text]


def workers_for(concurrency: int, cpu_count: int) -> int:
    """AIPerf worker processes: enough to drive the lanes, bounded by CPUs.

    The client shares node0 with the frontend, so it gets at most a quarter
    of the cores.
    """
    return max(1, min(concurrency, max(4, cpu_count // 4)))


def build_profile_command(
    *,
    aiperf_bin: str,
    served_model: str,
    tokenizer: str,
    url: str,
    dataset: str,
    spec: BenchSpec,
    concurrency: int,
    served_context: int,
    artifact_dir: str,
    server_metrics_urls: Sequence[str],
    cpu_count: int,
    hf_home: str,
    seed: int = 42,
) -> str:
    """Shell command (env assignments + argv) for one measured point."""
    env = {
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
        # Per-trajectory session affinity for Dynamo's KV-aware router.
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "1",
        "AIPERF_HTTP_CONNECTION_LIMIT": str(max(400, 2 * concurrency)),
        "HF_HOME": hf_home,
    }
    argv = [
        aiperf_bin,
        "profile",
        "--model", served_model,
        "--tokenizer", tokenizer,
        "--tokenizer-trust-remote-code",
        "--endpoint-type", "chat",
        "--streaming",
        "--url", url,
        "--public-dataset", dataset,
        "--scenario", SCENARIO,
        "--concurrency", str(concurrency),
        "--workers-max", str(workers_for(concurrency, cpu_count)),
        "--benchmark-duration", str(spec.duration_s),
        "--max-context-length", str(served_context - CONTEXT_MARGIN_TOKENS),
        "--trajectory-start-min-ratio", "0.25",
        "--trajectory-start-max-ratio", "0.75",
        "--random-seed", str(seed),
        "--no-fixed-schedule",
        "--use-server-token-count",
        "--extra-inputs", "ignore_eos:true",
        "--failed-request-threshold", "0.1",
        "--request-timeout-seconds", str(REQUEST_TIMEOUT_S),
        "--export-level", "records",
        "--server-metrics-formats", "json",
        "--server-metrics-formats", "jsonl",
        "--artifact-dir", artifact_dir,
        "--ui", "none",
    ]  # fmt: skip
    if spec.unsafe_override:
        # Screening runs shorter than AgentX's 900 s minimum; AIPerf stamps
        # submission_valid=false, which the report states.
        argv.append("--unsafe-override")
    if spec.warmup_s > 0:
        argv += ["--agentic-cache-warmup-duration", str(spec.warmup_s)]
    if server_metrics_urls:
        argv += ["--server-metrics", *server_metrics_urls]
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    return f"env {env_prefix} {shlex.join(argv)}"


def run_timeout_s(spec: BenchSpec) -> float:
    """Upper bound for one AIPerf process: configure + warmup + run + drain."""
    return 1800 + spec.warmup_s + spec.duration_s + 600


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _metric(report: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    value = report.get(name)
    if value is None and isinstance(report.get("records"), Mapping):
        value = report["records"].get(name)  # pre-0.12 layout
    return value if isinstance(value, Mapping) else None


def _stat(report: Mapping[str, Any], name: str, stat: str) -> float | None:
    metric = _metric(report, name)
    if metric is None or metric.get(stat) is None:
        return None
    return float(metric[stat])


def parse_profile_export(report: Mapping[str, Any]) -> PointMetrics:
    """``profile_export_aiperf.json`` -> PointMetrics."""
    error_types: dict[str, int] = {}
    for item in report.get("error_summary") or []:
        if not isinstance(item, Mapping):
            continue
        details = item.get("error_details") or {}
        label = str(
            details.get("type")
            or details.get("code")
            or details.get("message")
            or "error"
        )
        error_types[label] = error_types.get(label, 0) + int(item.get("count", 0))
    request_count = _stat(report, "request_count", "avg")
    error_count = _stat(report, "error_request_count", "avg")
    if error_count is None:
        error_count = float(sum(error_types.values()))
    reasoning = _stat(report, "total_reasoning_tokens", "avg")
    total_out = _stat(report, "total_output_tokens", "avg")
    return PointMetrics(
        aiperf_version=report.get("aiperf_version"),
        was_cancelled=bool(report.get("was_cancelled", False)),
        request_count=int(request_count or 0),
        error_count=int(error_count or 0),
        output_tps=_stat(report, "output_token_throughput", "avg"),
        total_output_tokens=int(total_out) if total_out is not None else None,
        total_reasoning_tokens=int(reasoning) if reasoning is not None else None,
        user_tps_p50=_stat(report, "output_token_throughput_per_user", "p50"),
        ttft_p50_ms=_stat(report, "time_to_first_token", "p50"),
        ttft_p99_ms=_stat(report, "time_to_first_token", "p99"),
        itl_p50_ms=_stat(report, "inter_token_latency", "p50"),
        itl_p99_ms=_stat(report, "inter_token_latency", "p99"),
        request_throughput=_stat(report, "request_throughput", "avg"),
        duration_s=_stat(report, "benchmark_duration", "avg"),
        isl_avg=_stat(report, "input_sequence_length", "avg"),
        osl_avg=_stat(report, "output_sequence_length", "avg"),
        error_types=error_types,
    )


def profiling_window(records_jsonl: Path) -> tuple[int, int] | None:
    """(first request start, last request end) in ns over profiling-phase records."""
    start: int | None = None
    end: int | None = None
    if not records_jsonl.exists():
        return None
    with open(records_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line of an interrupted run
            meta = record.get("metadata") or {}
            phase = str(meta.get("benchmark_phase", "profiling")).lower()
            if phase != "profiling":
                continue
            s = meta.get("request_start_ns")
            e = meta.get("request_end_ns")
            if isinstance(s, int):
                start = s if start is None else min(start, s)
            if isinstance(e, int):
                end = e if end is None else max(end, e)
    if start is None or end is None or end <= start:
        return None
    return start, end


def load_request_summary(
    artifact_dir: Path,
    *,
    gpus: int,
    tiers: Sequence[SloTier] | None = None,
    window: tuple[float, float] | None = None,
    arch: ModelArch | None = None,
    cached_fraction: float | None = None,
) -> RequestSummary | None:
    """Per-request summary from the newest ``profile_export.jsonl`` (or None)."""
    records_files = sorted(artifact_dir.rglob("profile_export.jsonl"))
    if not records_files:
        return None
    with open(records_files[-1], "r", encoding="utf-8") as fh:
        records, cancelled, bad = parse_records(fh)
    return summarize_requests(
        records,
        gpus=gpus,
        tiers=tiers,
        window=window,
        cancelled=cancelled,
        unreadable=bad,
        arch=arch,
        cached_fraction=cached_fraction,
    )


def load_point_metrics(artifact_dir: Path) -> PointMetrics | None:
    """Locate AIPerf exports under ``artifact_dir`` (it may nest a run folder)."""
    exports = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if not exports:
        return None
    with open(exports[-1], "r", encoding="utf-8") as fh:
        metrics = parse_profile_export(json.load(fh))
    records = sorted(artifact_dir.rglob("profile_export.jsonl"))
    window = profiling_window(records[-1]) if records else None
    if window:
        metrics = metrics.model_copy(
            update={"window_start_ns": window[0], "window_end_ns": window[1]}
        )
    return metrics


def dropped_trace_count(log_text: str) -> int | None:
    """Best-effort count of traces dropped by --max-context-length (UNVERIFIED wording)."""
    for line in reversed(log_text.splitlines()):
        lowered = line.lower()
        if "max_context_length" in lowered or "max-context-length" in lowered:
            digits = [
                int(tok) for tok in lowered.replace(",", " ").split() if tok.isdigit()
            ]
            if digits:
                return digits[0]
    return None
