# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf 0.14.0 load: AgentX replay of the SemiAnalysis Claude Code corpus by
default, or any dataset domain/datasets.py accepts (public, file, synthetic).

The AgentX command is AIPerf's short command (docs: tutorials/datasets-inputs/
inference-x-agent-x-mvp-benchmark, "Short command with --scenario agentx"):
``--scenario agentx --public-dataset semianalysis_cc_traces_weka_062126`` plus
server, model, tokenizer, concurrency and artifact directory. Facts verified in
the 0.14.0 wheel (aiperf/common/scenario/inferencex_agentx_mvp.py):

* the preset supplies streaming chat, seed 42, trajectory start ratios
  0.25-0.75, 10 warmup requests per lane with an 1800 s grace period, a 0.10
  failure threshold, a 300 s trace idle cap, 393 dataset entries, server token
  counts and 1800 s configure timeouts. An explicit flag wins over the preset,
  so the harness still passes what it must own: duration and warmup per
  BenchSpec, ``--max-context-length``, ``--extra-inputs``, server metrics and
  the router's session affinity (the docs: "Router affinity and server metrics
  remain explicit");
* ``--agentic-cache-warmup-duration`` replaces the preset's request budget
  (the two are mutually exclusive); the decision spec keeps the preset's;
* runs shorter than 900 s, and local trace files (the corpus identity cannot
  be pinned), need ``--unsafe-override``, which stamps ``submission_valid=false``
  into the ``metadata`` block of profile_export_aiperf.json;
* every run gets a fresh uuid ``benchmark_id`` and the scenario cache-busts
  first-turn prefixes with it, so consecutive runs cannot hit each other's
  prefix cache and no worker restart is needed between concurrency points;
* ``--max-context-length`` drops whole traces whose recorded peak exceeds
  it (no truncation);
* ``output_token_throughput`` is total OSL over the profiling phase's wall
  time (``start_time`` to ``end_time``, grace period included; the
  ``benchmark_duration`` metric spans only the finished requests), and OSL
  already includes reasoning tokens; ``input_token_throughput`` is total ISL
  over the same time (the P-Only objective). Requests still running when the
  grace period ends never finish, so neither total counts them;
* ``--extra-inputs`` consumes several ``key:value`` pairs and is merged into
  the payload after the per-turn ``max_completion_tokens``, so P-Only's
  ``max_completion_tokens:1`` caps every request at one output token.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.cluster_bench.domain.datasets import DatasetSpec, parse_dataset
from benchmarks.cluster_bench.domain.modelarch import ModelArch
from benchmarks.cluster_bench.domain.requests import (
    RequestSummary,
    SloTier,
    parse_records,
    summarize_requests,
)
from benchmarks.cluster_bench.domain.schema import BenchSpec, PointMetrics, Stage
from benchmarks.cluster_bench.domain.stages import PREFILL_ONLY_EXTRA_INPUT

SCENARIO = "agentx"
# The scenario's floor on an explicit --benchmark-duration
# (min_benchmark_duration_seconds; the preset's own default is 3600 s).
AGENTX_MIN_DURATION_S = 900
# The agentx preset's warmup_grace_period: a request-budget warmup ends at most
# this long after its burst is sent (bounds the run timeout).
AGENTX_WARMUP_GRACE_S = 1800
# AIPerf --public-dataset alias -> HF dataset repo (aiperf/plugin/plugins.yaml, 0.14.0).
PUBLIC_DATASET_REPOS = {
    "semianalysis_cc_traces_weka_062126": "semianalysisai/cc-traces-weka-062126",
    "semianalysis_cc_traces_weka_062126_256k": "semianalysisai/cc-traces-weka-062126-256k",
}
CONTEXT_MARGIN_TOKENS = 2048  # chat template + tool schema overhead
REQUEST_TIMEOUT_S = 3900

# Flags every point uses, then the extra ones per dataset kind.
COMMON_FLAGS = (
    "--use-server-token-count",
    "--benchmark-duration",
    "--workers-max",
    "--failed-request-threshold",
    "--extra-inputs",
    "--server-metrics",
    "--server-metrics-formats",
    "--export-level",
    "--output-artifact-dir",
)
AGENTX_FLAGS = (
    "--scenario",
    "--unsafe-override",
    "--agentic-cache-warmup-duration",
    "--warmup-requests-per-lane",
    "--max-context-length",
)
KIND_FLAGS = {
    "agentx": AGENTX_FLAGS,
    "public": ("--public-dataset", "--warmup-duration"),
    "file": (
        "--input-file",
        "--custom-dataset-type",
        "--no-fixed-schedule",
        "--warmup-duration",
    ),
    "synthetic": (
        "--synthetic-input-tokens-mean",
        "--synthetic-input-tokens-stddev",
        "--output-tokens-mean",
        "--output-tokens-stddev",
        "--warmup-duration",
    ),
}
# The default (AgentX public dataset) set, kept for callers that check it.
REQUIRED_FLAGS = ("--public-dataset", *AGENTX_FLAGS, *COMMON_FLAGS)
ALL_FLAGS = tuple(
    dict.fromkeys(
        (
            "--public-dataset",
            "--input-file",
            "--custom-dataset-type",
            *COMMON_FLAGS,
            *(flag for flags in KIND_FLAGS.values() for flag in flags),
        )
    )
)
# Flags the harness owns: an --aiperf-arg may not override them, or points
# would stop being comparable (load, dataset, output length, artifacts).
MANAGED_FLAGS = frozenset(
    (
        *ALL_FLAGS,
        "--model",
        "--tokenizer",
        "--url",
        "--endpoint-type",
        "--streaming",
        "--concurrency",
        "--request-rate",
        "--random-seed",
        "--output-tokens-mean",
        "--osl",
        "--isl",
        # Left to the agentx preset, but a point may not change them either.
        "--artifact-dir",
        "--trajectory-start-min-ratio",
        "--trajectory-start-max-ratio",
        "--fixed-schedule",
        "--no-fixed-schedule",
        "--ignore-trace-delays",
    )
)


def check_extra_args(extra: Sequence[str]) -> None:
    """Refuse ``--aiperf-arg`` values that override a harness-managed flag."""
    for arg in extra:
        if arg.startswith("--"):
            flag = arg.split("=", 1)[0]
            if flag in MANAGED_FLAGS:
                raise ValueError(
                    f"--aiperf-arg {arg!r}: {flag} is set by the harness "
                    "(choose the load with --dataset / --stage instead)"
                )


def required_flags(dataset: DatasetSpec) -> tuple[str, ...]:
    """Flags the points of ``dataset`` pass to ``aiperf profile``."""
    source = ("--input-file", "--custom-dataset-type") if dataset.path else ()
    if dataset.public:
        source = ("--public-dataset",)
    flags = (*source, *KIND_FLAGS[dataset.kind], *COMMON_FLAGS)
    return tuple(dict.fromkeys(flags))


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
    dataset: DatasetSpec | str,
    spec: BenchSpec,
    concurrency: int,
    served_context: int,
    artifact_dir: str,
    server_metrics_urls: Sequence[str],
    cpu_count: int,
    hf_home: str,
    seed: int = 42,
    stage: Stage = "e2e",
    extra_args: Sequence[str] = (),
    hf_endpoint: str | None = None,
) -> str:
    """Shell command (env assignments + argv) for one measured point.

    ``stage`` "prefill" (P-Only) caps every request at one output token;
    ``extra_args`` (``--aiperf-arg``) are appended verbatim; ``hf_endpoint``
    (--hf-endpoint) is where AIPerf downloads a public dataset from.
    """
    ds = parse_dataset(dataset) if isinstance(dataset, str) else dataset
    check_extra_args(extra_args)
    env = {
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
        # Per-trajectory session affinity for Dynamo's KV-aware router.
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "1",
        "AIPERF_HTTP_CONNECTION_LIMIT": str(max(400, 2 * concurrency)),
        "HF_HOME": hf_home,
    }
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    argv = [
        aiperf_bin,
        "profile",
        "--model", served_model,
        "--tokenizer", tokenizer,
        "--tokenizer-trust-remote-code",
        "--endpoint-type", "chat",
        "--streaming",
        "--url", url,
        *_dataset_args(ds),
        "--concurrency", str(concurrency),
        "--workers-max", str(workers_for(concurrency, cpu_count)),
        "--benchmark-duration", str(spec.duration_s),
        *(
            ["--max-context-length", str(served_context - CONTEXT_MARGIN_TOKENS)]
            if ds.agentx
            else []
        ),
        "--random-seed", str(seed),
        *(["--no-fixed-schedule"] if ds.kind == "file" else []),
        "--use-server-token-count",
        "--extra-inputs", *extra_inputs(stage),
        "--failed-request-threshold", "0.1",
        "--request-timeout-seconds", str(REQUEST_TIMEOUT_S),
        "--export-level", "records",
        "--server-metrics-formats", "json",
        "--server-metrics-formats", "jsonl",
        "--output-artifact-dir", artifact_dir,
        "--ui", "none",
    ]  # fmt: skip
    if ds.agentx and (spec.unsafe_override or ds.path):
        # Screening runs shorter than AgentX's 900 s minimum, or a local trace
        # file the scenario cannot pin; AIPerf stamps submission_valid=false,
        # which the report states.
        argv.append("--unsafe-override")
    argv += warmup_args(ds, spec)
    if server_metrics_urls:
        argv += ["--server-metrics", *server_metrics_urls]
    argv += list(extra_args)
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    return f"env {env_prefix} {shlex.join(argv)}"


def warmup_args(ds: DatasetSpec, spec: BenchSpec) -> list[str]:
    """Warmup flags of one point; AgentX without any keeps the preset's warmup."""
    requests = spec.agentx_warmup_requests
    if ds.agentx and requests is not None:
        if requests == "preset":
            return []  # 10 requests per lane, 1800 s grace
        return ["--warmup-requests-per-lane", str(requests)]
    if spec.warmup_s > 0:
        flag = "--agentic-cache-warmup-duration" if ds.agentx else "--warmup-duration"
        return [flag, str(spec.warmup_s)]
    return []


def extra_inputs(stage: Stage) -> list[str]:
    """``--extra-inputs`` values: exact OSL always; P-Only one output token."""
    values = ["ignore_eos:true"]
    if stage == "prefill":
        values.append(PREFILL_ONLY_EXTRA_INPUT)
    return values


def _dataset_args(ds: DatasetSpec) -> list[str]:
    if ds.kind == "synthetic":
        return [
            "--synthetic-input-tokens-mean", str(ds.isl),
            "--synthetic-input-tokens-stddev", str(ds.isl_std),
            "--output-tokens-mean", str(ds.osl),
            "--output-tokens-stddev", str(ds.osl_std),
        ]  # fmt: skip
    if ds.path:
        args = ["--input-file", ds.path, "--custom-dataset-type", str(ds.file_type)]
    else:
        args = ["--public-dataset", str(ds.public)]
    if ds.agentx:
        args += ["--scenario", SCENARIO]
    return args


def run_timeout_s(spec: BenchSpec) -> float:
    """Upper bound for one AIPerf process: configure + warmup + run + drain.

    A request-budget warmup (AgentX with no duration warmup) ends within the
    preset's grace period after its burst.
    """
    warmup = float(spec.warmup_s)
    if spec.agentx_warmup_requests is not None or spec.warmup_s == 0:
        warmup = max(warmup, AGENTX_WARMUP_GRACE_S)
    return 1800 + warmup + spec.duration_s + 600


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


def _phase_s(report: Mapping[str, Any]) -> float | None:
    """Wall time of the profiling phase: what AIPerf divides its throughputs by.

    ``start_time`` and ``end_time`` bound the phase, its grace period included
    (``benchmark_duration`` spans only the finished requests); without them the
    output throughput and its own total give it back.
    """
    try:
        start = datetime.fromisoformat(str(report["start_time"]))
        span = (datetime.fromisoformat(str(report["end_time"])) - start).total_seconds()
    except (KeyError, TypeError, ValueError):
        span = 0.0
    if span > 0:
        return span
    osl = _stat(report, "total_osl", "avg")
    tps = _stat(report, "output_token_throughput", "avg")
    return osl / tps if osl and tps else None


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
    # --scenario runs: AIPerf's verdict on the scenario rules (0.14.0 export).
    meta = report.get("metadata")
    meta = meta if isinstance(meta, Mapping) else {}
    valid = meta.get("submission_valid")
    complete = report.get("is_complete")
    return PointMetrics(
        aiperf_version=report.get("aiperf_version"),
        was_cancelled=bool(report.get("was_cancelled", False)),
        scenario=str(meta["scenario"]) if meta.get("scenario") else None,
        submission_valid=bool(valid) if valid is not None else None,
        submission_invalid_reasons=tuple(
            str(r) for r in meta.get("submission_invalid_reasons") or ()
        ),
        is_complete=bool(complete) if complete is not None else None,
        incomplete_reason=report.get("incomplete_reason") or None,
        request_count=int(request_count or 0),
        error_count=int(error_count or 0),
        output_tps=_stat(report, "output_token_throughput", "avg"),
        input_tps=_stat(report, "input_token_throughput", "avg"),
        total_output_tokens=int(total_out) if total_out is not None else None,
        total_reasoning_tokens=int(reasoning) if reasoning is not None else None,
        user_tps_p50=_stat(report, "output_token_throughput_per_user", "p50"),
        ttft_p50_ms=_stat(report, "time_to_first_token", "p50"),
        ttft_p99_ms=_stat(report, "time_to_first_token", "p99"),
        itl_p50_ms=_stat(report, "inter_token_latency", "p50"),
        itl_p99_ms=_stat(report, "inter_token_latency", "p99"),
        request_throughput=_stat(report, "request_throughput", "avg"),
        duration_s=_stat(report, "benchmark_duration", "avg"),
        phase_s=_phase_s(report),
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
    duration_s: float | None = None,
    arch: ModelArch | None = None,
    cached_fraction: float | None = None,
) -> RequestSummary | None:
    """Per-request summary from the newest ``profile_export.jsonl`` (or None).

    ``duration_s`` is what the rates divide by: pass ``PointMetrics.phase_s``.
    """
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
        duration_s=duration_s,
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


# AIPerf's exit box: "Reason: Run aborted by 'timing_manager': warmup_failure."
# (an AgentX warmup aborts the whole run on one failed trajectory).
_ABORT_RE = re.compile(r"Run aborted by '([^']+)': ([A-Za-z0-9_-]+)")
# Record errors arrive as "SSEResponseError('Error occurred in SSE response: ...')".
_WRAPPED_RE = re.compile(r"^\w+\((['\"])(.*)\1\)$", re.DOTALL)
_SSE_PREFIX = "Error occurred in SSE response: "
MAX_ERROR_CHARS = 160


def _error_text(error: Mapping[str, Any]) -> str:
    """``HTTP 400: <message>`` without AIPerf's exception wrapper."""
    message = " ".join(
        str(error.get("message") or error.get("type") or "error").split()
    )
    wrapped = _WRAPPED_RE.match(message)
    if wrapped:
        message = wrapped[2]
    message = message.removeprefix(_SSE_PREFIX)
    code = error.get("code")
    text = f"HTTP {code}: {message}" if code else message
    if len(text) > MAX_ERROR_CHARS:
        text = text[: MAX_ERROR_CHARS - 3] + "..."
    return text


def failure_details(point_dir: Path) -> tuple[str, ...]:
    """Why AIPerf left no summary export (best effort).

    AIPerf's abort reason from the harness's copy of its output
    (``aiperf.log``) and the most common error among the per-request records
    it wrote before stopping.
    """
    details: list[str] = []
    try:
        log = (point_dir / "aiperf.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    aborted = _ABORT_RE.search(log)
    if aborted:
        details.append(f"AIPerf aborted the run ({aborted[1]}: {aborted[2]})")
    errors: Counter[str] = Counter()
    for path in sorted(point_dir.rglob("profile_export.jsonl")):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"error"' not in line:
                        continue
                    try:
                        error = json.loads(line).get("error")
                    except (ValueError, AttributeError):
                        continue
                    if isinstance(error, Mapping):
                        errors[_error_text(error)] += 1
        except OSError:
            continue
    if errors:
        text, count = errors.most_common(1)[0]
        details.append(
            f"{sum(errors.values())} failed request(s), most common ({count}x): {text}"
        )
    return tuple(details)


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
