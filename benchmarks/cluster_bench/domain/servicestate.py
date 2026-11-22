# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-service state under load, from AIPerf's server-metrics export. Pure.

AIPerf 0.14.0 (``--server-metrics <url>... --server-metrics-formats json
jsonl``) scrapes every rank-0 worker's Dynamo system port and the frontend's
``/metrics`` (also ``<--url>/metrics``) about 3x/s and writes, per point:

* ``server_metrics_export.json`` (``ServerMetricsExportData``):
  ``metrics[<family>] = {"type": gauge|counter|histogram|unknown,
  "description", "unit", "series": [{"endpoint_url", "labels", "stats",
  "buckets"?, "timeslices"?}]}`` aggregated over the profiling phase
  (``summary.phase_time_ranges.profiling``, ns). Counter families lose their
  ``_total`` suffix (prometheus_client's parser). Gauge stats: avg/min/max/
  std/p1..p99; counter stats: total/rate; histogram stats: count/sum/avg/
  count_rate/sum_rate/pN_estimate plus cumulative bucket deltas.
* ``server_metrics_export.jsonl``: one ``SlimRecord`` per non-duplicate
  scrape: ``{"endpoint_url", "timestamp_ns", "metrics": {<family>:
  [{"labels"?, "value"} | {"labels"?, "buckets", "sum", "count"}]},
  "benchmark_phase"?: "warmup" | "profiling", ...}`` (``_info`` families
  dropped, ``None`` fields omitted).

``summarize_service_state`` turns both into a ``ServiceState``: per endpoint
and per role (frontend / agg / prefill / decode) the metrics of ``CATALOG``
under stable keys, plus a compact catch-all (``RoleState.other``) of every
other family. ``classify_bottleneck`` names what saturated a point.

Combination rules (one rule set for series and for instances)
--------------------------------------------------------------
* Series that differ only in ``REPLICA_LABELS`` (TP/PP/EP ranks, pids)
  report one scheduler's state and collapse to their maximum.
* The remaining groups (DP ranks, workers, label values) combine by sum for
  additive metrics (counts, throughputs, capacities) and by mean (``avg``,
  ``last``) and maximum (``max``) for intensive ones (ratios, lengths).
  ``rank_avg_max`` is the busiest group's average (the hottest scheduler),
  the value to compare with a per-scheduler cap such as
  ``max_running_requests``.
* Gauges come from the JSONL when it has scrapes in the window (exact
  per-scrape sums, ``last`` values); otherwise from the JSON, where an
  additive ``max`` over several groups is a sum of maxima (``approx``).
  Counters and histograms come from the JSON, else from JSONL deltas.
  Histogram percentiles are AIPerf's estimates for a single series, else
  linear interpolation in the merged cumulative buckets.
* Role values combine instances the same way: an additive role ``avg`` is
  the total over the role's instances (``RoleState.endpoints`` of them).

Metric names
------------
``dynamo_*`` names are VERIFIED against this repository
(``lib/runtime/src/metrics/prometheus_names.rs``, ``lib/llm/src/http/
service/metrics.rs``, ``components/src/dynamo/common/utils/prometheus.py``);
they were checked again in the Dynamo 1.6.2 source (tag v1.6.2; missing
ones are simply absent). ``sglang:*`` names marked ``verified`` were checked
against the SGLang 0.5.21 and 0.5.23 sources
(``srt/observability/metrics_collector.py`` and ``srt/eplb/``, 2026-11-22); the export's own type wins over ``kind`` (for
example ``kv_transfer_latency_ms`` is a histogram there). Opt-in families
appear only when their switch is set: ``--enable-mfu-metrics`` (estimated
FLOPs/bytes), ``SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`` (forward execution
seconds), ``--expert-balancedness-report-mode prometheus`` (EPLB
balancedness). Note: with DP attention SGLang may export only TP0's
scheduler unless ``--enable-metrics-for-all-schedulers`` (UNVERIFIED on
hardware), so worker request counts may cover one DP rank.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

from benchmarks.cluster_bench.domain.base import Frozen

MetricKind = Literal["gauge", "counter", "histogram"]
WORKER_ROLES = ("agg", "prefill", "decode")
ROLE_ORDER = ("frontend", "agg", "prefill", "decode", "control", "unknown")
REPLICA_LABELS = frozenset(
    {"tp_rank", "pp_rank", "moe_ep_rank", "ep_rank", "attn_tp_rank", "pid"}
)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})
SIGNIFICANT_DIGITS = 6


@dataclass(frozen=True)
class MetricSpec:
    """One metric extracted under a stable key (see the module docstring)."""

    key: str
    families: tuple[str, ...]  # candidate family names, without "_total"
    kind: MetricKind  # expected; the export's own type wins
    additive: bool
    verified: bool
    labels: tuple[tuple[str, str], ...] = ()  # keep only matching series
    doc: str = ""


def _spec(
    key: str,
    families: str | tuple[str, ...],
    kind: MetricKind,
    additive: bool,
    verified: bool,
    doc: str,
    labels: tuple[tuple[str, str], ...] = (),
) -> MetricSpec:
    names = (families,) if isinstance(families, str) else families
    return MetricSpec(key, names, kind, additive, verified, labels, doc)


# ``stage`` label values of sglang:per_stage_req_latency_seconds that SGLang
# 0.5.21 and 0.5.23 observe (RequestStage entries with metrics_is_observed=True).
REQUEST_STAGES: tuple[str, ...] = (
    "request_process",
    "prefill_forward",
    "chunked_prefill",
    "prefill_bootstrap",
    "prefill_transfer_kv_cache",
    "decode_prepare",
    "decode_bootstrap",
    "decode_waiting",
    "decode_transferred",
    "fake_output",
    "quick_finish",
)
CUDA_GRAPH_MODES: tuple[str, ...] = (
    "decode_cuda_graph",
    "decode_none",
    "prefill_cuda_graph",
    "prefill_none",
)


def _stage_specs() -> tuple[MetricSpec, ...]:
    return tuple(
        _spec(
            f"stage_{stage}",
            "sglang:per_stage_req_latency_seconds",
            "histogram",
            True,
            True,
            f"request stage {stage} s",
            (("stage", stage),),
        )
        for stage in REQUEST_STAGES
    )


def _cuda_graph_specs() -> tuple[MetricSpec, ...]:
    return tuple(
        _spec(
            f"cuda_graph_{mode}",
            "sglang:cuda_graph_passes",
            "counter",
            True,
            True,
            f"forward passes, mode {mode}",
            (("mode", mode),),
        )
        for mode in CUDA_GRAPH_MODES
    )


CATALOG: tuple[MetricSpec, ...] = (
    # SGLang scheduler / tokenizer (worker system port, "sglang:" filter).
    _spec("running_reqs", "sglang:num_running_reqs", "gauge", True, True, "requests in the running batch"),
    _spec("queue_reqs", "sglang:num_queue_reqs", "gauge", True, True, "requests waiting for admission"),
    _spec("kv_usage", "sglang:token_usage", "gauge", False, True, "KV pool usage 0..1"),
    _spec("used_tokens", "sglang:num_used_tokens", "gauge", True, True, "KV tokens in use"),
    _spec("max_total_tokens", "sglang:max_total_num_tokens", "gauge", True, True, "KV pool capacity (tokens)"),
    _spec("context_len", "sglang:context_len", "gauge", False, True, "served context length"),
    _spec("cache_hit_rate", "sglang:cache_hit_rate", "gauge", False, True, "prefix cache hit rate 0..1"),
    _spec("gen_throughput", "sglang:gen_throughput", "gauge", True, True, "decode tokens/s (scheduler log interval)"),
    _spec("spec_accept_length", "sglang:spec_accept_length", "gauge", False, True, "speculative acceptance length (log-interval mean)"),
    _spec("spec_accept_rate", "sglang:spec_accept_rate", "gauge", False, True, "speculative acceptance rate 0..1 (accepted / proposed drafts)"),
    _spec("retracted_reqs", ("sglang:num_retracted_reqs", "sglang:total_retracted_reqs"), "gauge", True, True, "requests retracted for KV space"),
    _spec("retracted_input_tokens", "sglang:num_retracted_input_tokens", "counter", True, True, "input tokens of retracted requests (recomputed later)"),
    _spec("retracted_output_tokens", "sglang:num_retracted_output_tokens", "counter", True, True, "output tokens of retracted requests (recomputed later)"),
    _spec("kv_evictable_tokens", "sglang:kv_evictable_tokens", "gauge", True, True, "radix-cached token slots that can be evicted"),
    _spec("evicted_tokens", "sglang:evicted_tokens", "counter", True, True, "KV token slots freed by eviction"),
    _spec("eviction_s", "sglang:eviction_duration_seconds", "histogram", True, True, "device eviction pass s"),
    _spec("prefill_bootstrap_queue", ("sglang:num_prefill_bootstrap_queue_reqs", "sglang:num_prefill_prealloc_queue_reqs"), "gauge", True, True, "PD prefill: waiting for decode-side bootstrap"),
    _spec("prefill_inflight_queue", "sglang:num_prefill_inflight_queue_reqs", "gauge", True, True, "PD prefill: done, KV transfer in flight"),
    _spec("decode_prealloc_queue", "sglang:num_decode_prealloc_queue_reqs", "gauge", True, True, "PD decode: waiting for KV allocation"),
    _spec("decode_transfer_queue", "sglang:num_decode_transfer_queue_reqs", "gauge", True, True, "PD decode: waiting for KV transfer"),
    _spec("kv_transfer_speed_gbs", "sglang:kv_transfer_speed_gb_s", "histogram", False, True, "PD KV transfer speed GB/s"),
    _spec("kv_transfer_latency_ms", "sglang:kv_transfer_latency_ms", "histogram", False, True, "PD KV transfer latency ms"),
    _spec("kv_transfer_bootstrap_ms", "sglang:kv_transfer_bootstrap_ms", "histogram", False, True, "PD KV transfer bootstrap ms"),
    _spec("kv_transfer_alloc_ms", "sglang:kv_transfer_alloc_ms", "histogram", False, True, "PD KV transfer: waiting for decode allocation ms"),
    _spec("kv_transfer_mb", "sglang:kv_transfer_total_mb", "histogram", True, True, "PD KV MB per transfer"),
    _spec("transfer_failed_reqs", "sglang:num_transfer_failed_reqs", "counter", True, True, "PD KV transfer failures"),
    _spec("bootstrap_failed_reqs", "sglang:num_bootstrap_failed_reqs", "counter", True, True, "PD bootstrap failures"),
    _spec("prompt_tokens", "sglang:prompt_tokens", "counter", True, True, "prefill tokens"),
    _spec("generation_tokens", "sglang:generation_tokens", "counter", True, True, "generated tokens"),
    _spec("cached_tokens", "sglang:cached_tokens", "counter", True, True, "prompt tokens served from the prefix cache"),
    _spec("requests", "sglang:num_requests", "counter", True, True, "requests finished"),
    _spec("aborted_requests", "sglang:num_aborted_requests", "counter", True, True, "requests aborted"),
    _spec("spec_verify_calls", "sglang:spec_verify_calls", "counter", True, True, "speculative verify steps"),
    _spec("decode_tokens", "sglang:realtime_tokens", "counter", True, True, "decode tokens", (("mode", "decode"),)),
    _spec("ttft", "sglang:time_to_first_token_seconds", "histogram", True, True, "worker TTFT s"),
    _spec("itl", "sglang:inter_token_latency_seconds", "histogram", True, True, "worker ITL s"),
    _spec("e2e_latency", "sglang:e2e_request_latency_seconds", "histogram", True, True, "worker e2e latency s"),
    _spec("queue_time", "sglang:queue_time_seconds", "histogram", True, True, "worker queueing s"),
    *_stage_specs(),  # per-request stage latency (PD bootstrap, prefill, KV transfer, decode wait)
    *_cuda_graph_specs(),  # CUDA graph coverage of forward passes
    # SGLang opt-in families (diagnostic window switches, see the docstring).
    _spec("est_flops", "sglang:estimated_flops_per_gpu", "counter", True, True, "estimated FLOPs per GPU"),
    _spec("est_read_bytes", "sglang:estimated_read_bytes_per_gpu", "counter", True, True, "estimated bytes read per GPU"),
    _spec("est_write_bytes", "sglang:estimated_write_bytes_per_gpu", "counter", True, True, "estimated bytes written per GPU"),
    _spec("forward_s", "sglang:forward_execution_seconds", "counter", True, True, "GPU-busy forward execution s"),
    _spec("eplb_balancedness", "sglang:eplb_balancedness", "gauge", False, True, "MoE balancedness 0..1 (mean / max GPU load per layer)"),
    # Dynamo worker (work handler + LLMBackendMetrics).
    _spec("worker_inflight", "dynamo_component_inflight_requests", "gauge", True, True, "requests inside the worker process"),
    _spec("worker_requests", "dynamo_component_requests", "counter", True, True, "requests handled"),
    _spec("worker_errors", "dynamo_component_errors", "counter", True, True, "handler errors"),
    _spec("worker_request_duration", "dynamo_component_request_duration_seconds", "histogram", True, True, "handler duration s"),
    _spec("kv_total_blocks", "dynamo_component_total_blocks", "gauge", True, True, "KV blocks"),
    _spec("kv_usage_dynamo", "dynamo_component_gpu_cache_usage_percent", "gauge", False, True, "KV usage 0..1 (Dynamo)"),
    _spec("kv_hit_rate_dynamo", "dynamo_component_kv_cache_hit_rate", "gauge", False, True, "prefix hit rate 0..1 (Dynamo)"),
    _spec("model_load_time_s", "dynamo_component_model_load_time_seconds", "gauge", False, True, "model load seconds"),
    # Dynamo frontend and KV router.
    _spec("fe_inflight", "dynamo_frontend_inflight_requests", "gauge", True, True, "requests going to the engines"),
    _spec("fe_queued", "dynamo_frontend_queued_requests", "gauge", True, True, "requests before their first response"),
    _spec("fe_active", "dynamo_frontend_active_requests", "gauge", True, True, "requests in the HTTP handler"),
    _spec("fe_disconnected", "dynamo_frontend_disconnected_clients", "gauge", True, True, "disconnected clients"),
    _spec("fe_requests", "dynamo_frontend_requests", "counter", True, True, "requests completed"),
    _spec("fe_errors", "dynamo_frontend_requests", "counter", True, True, "requests failed", (("status", "error"),)),
    _spec("fe_output_tokens", "dynamo_frontend_output_tokens", "counter", True, True, "output tokens"),
    _spec("fe_ttft", "dynamo_frontend_time_to_first_token_seconds", "histogram", True, True, "frontend TTFT s"),
    _spec("fe_itl", "dynamo_frontend_inter_token_latency_seconds", "histogram", True, True, "frontend ITL s"),
    _spec("fe_isl", "dynamo_frontend_input_sequence_tokens", "histogram", True, True, "input tokens per request"),
    _spec("fe_osl", "dynamo_frontend_output_sequence_tokens", "histogram", True, True, "output tokens per request"),
    _spec("fe_request_duration", "dynamo_frontend_request_duration_seconds", "histogram", True, True, "request duration s"),
    _spec("fe_cached_tokens", "dynamo_frontend_cached_tokens", "histogram", True, True, "cached tokens per request"),
    _spec("fe_tokenizer_ms", "dynamo_frontend_tokenizer_latency_ms", "histogram", True, True, "tokenizer ms"),
    _spec("router_queue_pending", "dynamo_frontend_router_queue_pending_requests", "gauge", True, True, "requests queued in the router"),
    _spec("router_kv_hit_rate", ("dynamo_component_router_kv_hit_rate", "dynamo_frontend_kv_hit_rate"), "histogram", True, True, "predicted prefix hit rate"),
    _spec("router_overhead_ms", "dynamo_router_overhead_total_ms", "histogram", True, True, "routing overhead ms"),
    _spec("router_non_max_overlap", "dynamo_component_router_non_max_overlap_selections", "counter", True, True, "selections below the best overlap"),
    _spec("router_selected_cached_tokens", "dynamo_component_router_kv_selected_cached_prefix_tokens", "counter", True, True, "cached prefix tokens on the chosen worker"),
    _spec("router_reused_tokens", "dynamo_component_router_kv_worker_reused_tokens", "counter", True, True, "worker-reported reused tokens"),
    _spec("worker_active_decode_blocks", "dynamo_frontend_worker_active_decode_blocks", "gauge", True, True, "router view: decode blocks"),
    _spec("worker_active_prefill_tokens", "dynamo_frontend_worker_active_prefill_tokens", "gauge", True, True, "router view: prefill tokens"),
    _spec("model_total_kv_blocks", "dynamo_frontend_model_total_kv_blocks", "gauge", True, True, "KV blocks over workers"),
    _spec("model_max_num_seqs", "dynamo_frontend_model_max_num_seqs", "gauge", False, True, "per-worker request cap"),
    _spec("model_max_batched_tokens", "dynamo_frontend_model_max_num_batched_tokens", "gauge", False, True, "per-worker batched-token cap"),
)  # fmt: skip
CATALOG_FAMILIES = frozenset(name for spec in CATALOG for name in spec.families)


def _gauge_specs() -> dict[str, tuple[MetricSpec, ...]]:
    """Family name -> catalog gauges reading it (tracked per JSONL scrape)."""
    out: dict[str, list[MetricSpec]] = {}
    for spec in CATALOG:
        # Histogram specs too: other SGLang versions export some of these
        # families as plain gauges (only scalar samples are ever tracked).
        if spec.kind in ("gauge", "histogram"):
            for name in spec.families:
                out.setdefault(name, []).append(spec)
    return {name: tuple(specs) for name, specs in out.items()}


_GAUGE_SPECS = _gauge_specs()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class EndpointDescriptor(Frozen):
    """A scraped URL and the container behind it."""

    url: str
    role: str  # control | frontend | agg | prefill | decode | unknown
    node: str = ""
    container: str = ""


class MetricValue(Frozen):
    """One catalog metric reduced over the window (fields by kind).

    gauge: avg/max/p50/p90/p99/last/rank_avg_max. counter: total (increase)
    and rate_per_s. histogram: total (observations), rate_per_s, avg (mean
    observation), p50/p90/p99.
    """

    kind: MetricKind
    source: str  # family name that matched
    source_kind: Literal["json", "jsonl"] = "json"
    series: int = 1
    groups: int = 1
    avg: float | None = None
    max: float | None = None
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    last: float | None = None
    rank_avg_max: float | None = None
    total: float | None = None
    rate_per_s: float | None = None
    approx: bool = False  # max is a sum of per-group maxima


class OtherMetric(Frozen):
    """A non-catalog family, compactly: gauge mean/max/last (over series);
    counter mean = rate per second and total = increase; histogram mean =
    mean observation, max = highest p99 estimate, total = observations."""

    kind: str
    mean: float | None = None
    max: float | None = None
    last: float | None = None
    total: float | None = None
    series: int = 0
    endpoints: int = 0


class EndpointState(Frozen):
    url: str
    role: str
    node: str = ""
    container: str = ""
    scraped: bool = False
    scrapes: int = 0  # JSONL scrapes inside the window
    fetch_latency_ms: float | None = None
    metrics: dict[str, MetricValue] = Field(default_factory=dict)


class RoleState(Frozen):
    role: str
    endpoints: int
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    derived: dict[str, float] = Field(default_factory=dict)
    other: dict[str, OtherMetric] = Field(default_factory=dict)


class ServiceState(Frozen):
    """What the serving stack was doing during one point's window."""

    source: Literal["none", "json", "jsonl", "json+jsonl"] = "none"
    aiperf_version: str | None = None
    window_start_s: float | None = None
    window_end_s: float | None = None
    roles: dict[str, RoleState] = Field(default_factory=dict)
    endpoints: tuple[EndpointState, ...] = ()
    unmatched_urls: tuple[str, ...] = ()  # scraped, no descriptor
    missing_urls: tuple[str, ...] = ()  # described, never scraped
    notes: tuple[str, ...] = ()

    def metric(self, role: str, key: str) -> MetricValue | None:
        state = self.roles.get(role)
        return state.metrics.get(key) if state else None

    def value(self, role: str, key: str, attr: str = "avg") -> float | None:
        metric = self.metric(role, key)
        return getattr(metric, attr) if metric is not None else None

    @property
    def worker_roles(self) -> tuple[str, ...]:
        return tuple(r for r in WORKER_ROLES if r in self.roles)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """AIPerf's metrics-URL form (scheme, ``/metrics``), lower-case host."""
    text = (url or "").strip()
    if not text.lower().startswith(("http://", "https://")):
        text = "http://" + text
    text = text.rstrip("/")
    if not text.endswith("/metrics"):
        text += "/metrics"
    parts = urlsplit(text)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path}"


def endpoint_descriptors(containers: Iterable[Any]) -> tuple[EndpointDescriptor, ...]:
    """Descriptors of ``LaunchPlan.containers`` that expose ``metrics_url``
    (duck-typed: ``metrics_url``, ``role``, ``node``, ``name``)."""
    out: list[EndpointDescriptor] = []
    for container in containers:
        url = getattr(container, "metrics_url", None)
        if not url:
            continue
        out.append(
            EndpointDescriptor(
                url=normalize_url(str(url)),
                role=str(getattr(container, "role", "unknown")),
                node=str(getattr(container, "node", "")),
                container=str(getattr(container, "name", "")),
            )
        )
    return tuple(out)


def canonical_family(name: str) -> str:
    """Family name without the counter ``_total`` suffix."""
    return name[:-6] if name.endswith("_total") else name


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sig(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    digits = SIGNIFICANT_DIGITS - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, max(0, digits))


def _labels(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _matches(labels: Mapping[str, str], wanted: tuple[tuple[str, str], ...]) -> bool:
    return all(labels.get(k) == v for k, v in wanted)


def _group_key(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, v) for k, v in labels.items() if k not in REPLICA_LABELS))


def _series_key(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy default), ``q`` in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        return None
    k = (len(ordered) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _le(bound: Any) -> float | None:
    text = str(bound).strip()
    if text.lower() in ("+inf", "inf"):
        return math.inf
    try:
        return float(text)
    except ValueError:
        return None


def bucket_quantile(buckets: Mapping[float, float], q: float) -> float | None:
    """Prometheus-style quantile from cumulative ``le -> count`` buckets.

    Linear inside a bucket (the first starts at 0); a quantile in +Inf
    returns the highest finite bound.
    """
    ordered = sorted(buckets.items())
    if not ordered:
        return None
    total = ordered[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in ordered:
        if count >= target:
            if math.isinf(le):
                return prev_le
            if count <= prev_count:
                return le
            lower = prev_le if prev_le < le else 0.0
            return lower + (le - lower) * (target - prev_count) / (count - prev_count)
        if not math.isinf(le):
            prev_le = le
        prev_count = count
    return prev_le


def _bucket_map(raw: Any) -> dict[float, float]:
    out: dict[float, float] = {}
    if isinstance(raw, Mapping):
        for bound, count in raw.items():
            le, value = _le(bound), _num(count)
            if le is not None and value is not None:
                out[le] = out.get(le, 0.0) + value
    return out


def _merge_buckets(maps: Iterable[Mapping[float, float]]) -> dict[float, float]:
    out: dict[float, float] = {}
    for buckets in maps:
        for le, count in buckets.items():
            out[le] = out.get(le, 0.0) + count
    return out


# --------------------------------------------------------------------------
# Internal values (mergeable), converted to MetricValue at the end
# --------------------------------------------------------------------------


@dataclass
class _Value:
    kind: MetricKind
    source: str
    additive: bool
    origin: Literal["json", "jsonl"]
    series: int = 1
    groups: int = 1
    avg: float | None = None
    peak: float | None = None
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    last: float | None = None
    rank_avg_max: float | None = None
    approx: bool = False
    total: float | None = None
    rate: float | None = None
    hsum: float | None = None
    buckets: dict[float, float] | None = None

    def model(self) -> MetricValue:
        avg = self.avg
        if self.kind == "histogram" and self.total and self.hsum is not None:
            avg = self.hsum / self.total
        return MetricValue(
            kind=self.kind,
            source=self.source,
            source_kind=self.origin,
            series=self.series,
            groups=self.groups,
            avg=_sig(avg),
            max=_sig(self.peak),
            p50=_sig(self.p50),
            p90=_sig(self.p90),
            p99=_sig(self.p99),
            last=_sig(self.last),
            rank_avg_max=_sig(self.rank_avg_max),
            total=_sig(self.total),
            rate_per_s=_sig(self.rate),
            approx=self.approx,
        )


def _combine_gauges(values: Sequence[_Value], additive: bool) -> _Value:
    """Groups (or instances) of one gauge into one value."""
    first = values[0]
    if len(values) == 1:
        return first
    known = [v.avg for v in values if v.avg is not None]
    maxima = [v.peak for v in values if v.peak is not None]
    lasts = [v.last for v in values if v.last is not None]
    hottest = [
        v.rank_avg_max if v.rank_avg_max is not None else v.avg
        for v in values
        if v.rank_avg_max is not None or v.avg is not None
    ]
    if additive:
        avg = sum(known) if known else None
        peak = sum(maxima) if maxima else None
        last = sum(lasts) if len(lasts) == len(values) else None
    else:
        avg = sum(known) / len(known) if known else None
        peak = max(maxima) if maxima else None
        last = sum(lasts) / len(lasts) if lasts else None
    return _Value(
        kind="gauge",
        source=first.source,
        additive=additive,
        origin=first.origin,
        series=sum(v.series for v in values),
        groups=sum(v.groups for v in values),
        avg=avg,
        peak=peak,
        last=last,
        rank_avg_max=max(hottest) if hottest else None,
        approx=additive or any(v.approx for v in values),
    )


def _combine_counters(values: Sequence[_Value]) -> _Value:
    first = values[0]
    if len(values) == 1:
        return first
    totals = [v.total for v in values if v.total is not None]
    rates = [v.rate for v in values if v.rate is not None]
    return _Value(
        kind="counter",
        source=first.source,
        additive=True,
        origin=first.origin,
        series=sum(v.series for v in values),
        groups=sum(v.groups for v in values),
        total=sum(totals) if totals else None,
        rate=sum(rates) if rates else None,
    )


def _combine_histograms(values: Sequence[_Value]) -> _Value:
    first = values[0]
    if len(values) == 1:
        return first
    buckets = _merge_buckets(v.buckets for v in values if v.buckets)
    totals = [v.total for v in values if v.total is not None]
    sums = [v.hsum for v in values if v.hsum is not None]
    rates = [v.rate for v in values if v.rate is not None]
    return _Value(
        kind="histogram",
        source=first.source,
        additive=True,
        origin=first.origin,
        series=sum(v.series for v in values),
        groups=sum(v.groups for v in values),
        total=sum(totals) if totals else None,
        hsum=sum(sums) if sums else None,
        rate=sum(rates) if rates else None,
        p50=bucket_quantile(buckets, 0.50),
        p90=bucket_quantile(buckets, 0.90),
        p99=bucket_quantile(buckets, 0.99),
        buckets=buckets or None,
    )


def _combine(values: Sequence[_Value], additive: bool) -> _Value:
    kind = values[0].kind
    if kind == "gauge":
        return _combine_gauges(values, additive)
    if kind == "counter":
        return _combine_counters(values)
    return _combine_histograms(values)


# --------------------------------------------------------------------------
# JSON export
# --------------------------------------------------------------------------


@dataclass
class _JsonFamily:
    kind: str
    series: list[tuple[dict[str, str], Mapping[str, Any], Mapping[str, Any] | None]]


@dataclass
class _JsonExport:
    families: dict[str, dict[str, _JsonFamily]]  # url -> family -> data
    info: dict[str, Mapping[str, Any]]
    window: tuple[float, float] | None
    successful: tuple[str, ...]
    version: str | None


def _parse_export(export: Any, notes: list[str]) -> _JsonExport | None:
    if not isinstance(export, Mapping):
        return None
    metrics = export.get("metrics")
    if not isinstance(metrics, Mapping):
        notes.append("server-metrics JSON has no metrics object")
        return None
    families: dict[str, dict[str, _JsonFamily]] = {}
    for name, family in metrics.items():
        if not isinstance(family, Mapping) or str(name).endswith("_info"):
            continue
        kind = str(family.get("type", "gauge")).lower()
        for series in family.get("series") or ():
            if not isinstance(series, Mapping) or not series.get("endpoint_url"):
                continue
            url = normalize_url(str(series["endpoint_url"]))
            stats = series.get("stats")
            entry = families.setdefault(url, {}).setdefault(
                canonical_family(str(name)), _JsonFamily(kind, [])
            )
            entry.series.append(
                (
                    _labels(series.get("labels")),
                    stats if isinstance(stats, Mapping) else {},
                    series.get("buckets")
                    if isinstance(series.get("buckets"), Mapping)
                    else None,
                )
            )
    summary = export.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    raw_info = summary.get("endpoint_info")
    info = {
        normalize_url(str(url)): data
        for url, data in (raw_info.items() if isinstance(raw_info, Mapping) else ())
        if isinstance(data, Mapping)
    }
    window = None
    ranges = summary.get("phase_time_ranges")
    profiling = ranges.get("profiling") if isinstance(ranges, Mapping) else None
    if isinstance(profiling, Mapping):
        start, end = _num(profiling.get("start_ns")), _num(profiling.get("end_ns"))
        if start is not None and end is not None and end > start:
            window = (start / 1e9, end / 1e9)
    successful = tuple(
        normalize_url(str(u)) for u in summary.get("endpoints_successful") or ()
    )
    version = export.get("aiperf_version")
    return _JsonExport(
        families, info, window, successful, str(version) if version else None
    )


def _json_gauge(spec: MetricSpec, name: str, family: _JsonFamily) -> _Value | None:
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for labels, stats, _ in family.series:
        if _matches(labels, spec.labels) and _num(stats.get("avg")) is not None:
            groups.setdefault(_group_key(labels), []).append(stats)
    if not groups:
        return None
    parts: list[_Value] = []
    for members in groups.values():
        stats = max(members, key=lambda s: _num(s.get("avg")) or 0.0)
        parts.append(
            _Value(
                kind="gauge",
                source=name,
                additive=spec.additive,
                origin="json",
                series=len(members),
                groups=1,
                avg=_num(stats.get("avg")),
                peak=_num(stats.get("max")),
                p50=_num(stats.get("p50")),
                p90=_num(stats.get("p90")),
                p99=_num(stats.get("p99")),
                rank_avg_max=_num(stats.get("avg")),
            )
        )
    return _combine_gauges(parts, spec.additive)


def _json_counter(spec: MetricSpec, name: str, family: _JsonFamily) -> _Value | None:
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for labels, stats, _ in family.series:
        if _matches(labels, spec.labels) and _num(stats.get("total")) is not None:
            groups.setdefault(_group_key(labels), []).append(stats)
    if not groups:
        return None
    parts = []
    for members in groups.values():
        stats = max(members, key=lambda s: _num(s.get("total")) or 0.0)
        parts.append(
            _Value(
                kind="counter",
                source=name,
                additive=True,
                origin="json",
                series=len(members),
                total=_num(stats.get("total")),
                rate=_num(stats.get("rate")),
            )
        )
    return _combine_counters(parts)


def _json_histogram(spec: MetricSpec, name: str, family: _JsonFamily) -> _Value | None:
    groups: dict[tuple, list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]] = {}
    for labels, stats, buckets in family.series:
        if _matches(labels, spec.labels) and _num(stats.get("count")) is not None:
            groups.setdefault(_group_key(labels), []).append((stats, buckets))
    if not groups:
        return None
    parts = []
    for members in groups.values():
        stats, buckets = max(members, key=lambda m: _num(m[0].get("count")) or 0.0)
        count = _num(stats.get("count")) or 0.0
        parts.append(
            _Value(
                kind="histogram",
                source=name,
                additive=True,
                origin="json",
                series=len(members),
                total=count,
                hsum=_num(stats.get("sum")) if count else None,
                rate=_num(stats.get("count_rate")),
                p50=_num(stats.get("p50_estimate")),
                p90=_num(stats.get("p90_estimate")),
                p99=_num(stats.get("p99_estimate")),
                buckets=_bucket_map(buckets) or None,
            )
        )
    return _combine_histograms(parts)


# --------------------------------------------------------------------------
# JSONL stream
# --------------------------------------------------------------------------


@dataclass
class _GaugeTrack:
    """Per-scrape combined values of one catalog gauge on one endpoint."""

    values: list[float] = field(default_factory=list)
    groups: dict[tuple, list[float]] = field(default_factory=dict)  # [sum, n]
    series: set = field(default_factory=set)


@dataclass
class _SeriesEnds:
    """Reference (last before the window), first and last in-window samples."""

    ref: Any = None
    ref_t: float | None = None
    first: Any = None
    first_t: float | None = None
    last: Any = None
    last_t: float | None = None


@dataclass
class _FamilyStream:
    histogram: bool = False
    ends: dict[tuple, _SeriesEnds] = field(default_factory=dict)
    labels: dict[tuple, dict[str, str]] = field(default_factory=dict)
    # Catch-all gauge view: mean over series per scrape, streamed.
    n: int = 0
    total: float = 0.0
    peak: float = -math.inf
    last: float | None = None
    last_sum: float | None = None  # counters: sum over series of the last scrape


@dataclass
class _Stream:
    window: tuple[float, float] | None
    families: dict[str, dict[str, _FamilyStream]] = field(default_factory=dict)
    gauges: dict[str, dict[str, _GaugeTrack]] = field(default_factory=dict)
    scrapes: dict[str, int] = field(default_factory=dict)
    seen_profiling: bool = False
    bad: int = 0

    def position(self, t: float, phase: Any) -> int:
        """-1 before the window, 0 inside, 1 after."""
        if self.window is not None:
            if t < self.window[0]:
                return -1
            return 1 if t > self.window[1] else 0
        if phase == "profiling":
            self.seen_profiling = True
            return 0
        return 1 if self.seen_profiling else -1


def _sample_value(sample: Mapping[str, Any]) -> Any:
    if sample.get("buckets") is not None:
        count = _num(sample.get("count"))
        return (
            count if count is not None else 0.0,
            _num(sample.get("sum")) or 0.0,
            _bucket_map(sample.get("buckets")),
        )
    return _num(sample.get("value"))


def _consume(stream: _Stream, record: Mapping[str, Any]) -> None:
    url = normalize_url(str(record.get("endpoint_url") or ""))
    t_ns = _num(record.get("timestamp_ns"))
    metrics = record.get("metrics")
    if t_ns is None or not isinstance(metrics, Mapping):
        stream.bad += 1
        return
    t = t_ns / 1e9
    where = stream.position(t, record.get("benchmark_phase"))
    if where > 0:
        return
    if where == 0:
        stream.scrapes[url] = stream.scrapes.get(url, 0) + 1
    families = stream.families.setdefault(url, {})
    for raw_name, samples in metrics.items():
        if not isinstance(samples, list):
            continue
        name = canonical_family(str(raw_name))
        fam = families.setdefault(name, _FamilyStream())
        scalars: list[tuple[dict[str, str], float]] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            labels = _labels(sample.get("labels"))
            key = _series_key(labels)
            value = _sample_value(sample)
            if value is None:
                continue
            if isinstance(value, tuple):
                fam.histogram = True
            else:
                scalars.append((labels, value))
            fam.labels.setdefault(key, labels)
            ends = fam.ends.setdefault(key, _SeriesEnds())
            if where < 0:
                ends.ref, ends.ref_t = value, t
            else:
                if ends.first is None:
                    ends.first, ends.first_t = value, t
                ends.last, ends.last_t = value, t
        if where < 0 or not scalars:
            continue
        summed = sum(v for _, v in scalars)
        mean = summed / len(scalars)
        fam.n += 1
        fam.total += mean
        fam.peak = max(fam.peak, mean)
        fam.last, fam.last_sum = mean, summed
        for spec in _GAUGE_SPECS.get(name, ()):
            _track_gauge(stream, url, spec, scalars)


def _track_gauge(
    stream: _Stream,
    url: str,
    spec: MetricSpec,
    scalars: Sequence[tuple[dict[str, str], float]],
) -> None:
    grouped: dict[tuple, float] = {}
    track = stream.gauges.setdefault(url, {}).setdefault(spec.key, _GaugeTrack())
    for labels, value in scalars:
        if not _matches(labels, spec.labels):
            continue
        key = _group_key(labels)
        grouped[key] = max(grouped.get(key, -math.inf), value)
        track.series.add(_series_key(labels))
    if not grouped:
        return
    combined = sum(grouped.values())
    if not spec.additive:
        combined /= len(grouped)
    track.values.append(combined)
    for key, value in grouped.items():
        acc = track.groups.setdefault(key, [0.0, 0.0])
        acc[0] += value
        acc[1] += 1


def _stream_lines(lines: Iterable[str], stream: _Stream) -> None:
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            stream.bad += 1  # e.g. the torn last line of a killed run
            continue
        if isinstance(record, Mapping):
            _consume(stream, record)
        else:
            stream.bad += 1


def _jsonl_gauge(spec: MetricSpec, name: str, track: _GaugeTrack) -> _Value | None:
    if not track.values:
        return None
    averages = [s / n for s, n in track.groups.values() if n]
    return _Value(
        kind="gauge",
        source=name,
        additive=spec.additive,
        origin="jsonl",
        series=len(track.series),
        groups=len(track.groups),
        avg=sum(track.values) / len(track.values),
        peak=max(track.values),
        p50=percentile(track.values, 50),
        p90=percentile(track.values, 90),
        p99=percentile(track.values, 99),
        last=track.values[-1],
        rank_avg_max=max(averages) if averages else None,
    )


def _delta(ends: _SeriesEnds) -> tuple[Any, float] | None:
    """(value delta, seconds) of one cumulative series inside the window."""
    if ends.last is None:
        return None
    base, base_t = (
        (ends.ref, ends.ref_t) if ends.ref is not None else (ends.first, ends.first_t)
    )
    if base is None or base_t is None or ends.last_t is None:
        return None
    seconds = ends.last_t - base_t
    if isinstance(ends.last, tuple):
        count = max(0.0, ends.last[0] - base[0])
        total = max(0.0, ends.last[1] - base[1])
        buckets = {
            le: max(0.0, c - base[2].get(le, 0.0)) for le, c in ends.last[2].items()
        }
        return (count, total, buckets), seconds
    return max(0.0, ends.last - base), seconds


def _jsonl_cumulative(
    spec: MetricSpec, name: str, fam: _FamilyStream, kind: MetricKind
) -> _Value | None:
    groups: dict[tuple, list[tuple[Any, float]]] = {}
    for key, ends in fam.ends.items():
        if not _matches(fam.labels.get(key, {}), spec.labels):
            continue
        if (kind == "histogram") != isinstance(ends.last, tuple):
            continue  # sample shape disagrees with the kind: not comparable
        delta = _delta(ends)
        if delta is not None:
            groups.setdefault(_group_key(fam.labels.get(key, {})), []).append(delta)
    if not groups:
        return None
    parts = []
    for members in groups.values():
        if kind == "histogram":
            (count, total, buckets), seconds = max(members, key=lambda m: m[0][0])
            parts.append(
                _Value(
                    kind="histogram",
                    source=name,
                    additive=True,
                    origin="jsonl",
                    series=len(members),
                    total=count,
                    hsum=total if count else None,
                    rate=count / seconds if seconds > 0 else None,
                    p50=bucket_quantile(buckets, 0.50),
                    p90=bucket_quantile(buckets, 0.90),
                    p99=bucket_quantile(buckets, 0.99),
                    buckets=buckets or None,
                )
            )
        else:
            value, seconds = max(members, key=lambda m: m[0])
            parts.append(
                _Value(
                    kind="counter",
                    source=name,
                    additive=True,
                    origin="jsonl",
                    series=len(members),
                    total=value,
                    rate=value / seconds if seconds > 0 else None,
                )
            )
    return _combine(parts, True)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _kind_of(spec: MetricSpec, family: _JsonFamily | None) -> MetricKind:
    if family is not None:
        if family.kind in ("gauge", "unknown"):
            return "gauge"
        if family.kind in ("counter", "histogram"):
            return family.kind  # type: ignore[return-value]
    return spec.kind


def _catalog_value(
    spec: MetricSpec,
    json_fams: Mapping[str, _JsonFamily],
    stream_fams: Mapping[str, _FamilyStream],
    tracks: Mapping[str, _GaugeTrack],
) -> _Value | None:
    for name in spec.families:
        family = json_fams.get(name)
        fam_stream = stream_fams.get(name)
        if family is None and fam_stream is None:
            continue
        kind = _kind_of(spec, family)
        if fam_stream is not None and fam_stream.histogram:
            kind = "histogram"
        elif kind == "histogram" and family is None and fam_stream is not None:
            kind = "gauge"  # scalar samples only: this version exports a gauge
        value: _Value | None = None
        if kind == "gauge":
            track = tracks.get(spec.key)
            if track is not None and track.values:
                value = _jsonl_gauge(spec, name, track)
            elif family is not None:
                value = _json_gauge(spec, name, family)
                if value is not None and track is not None and track.values:
                    value.last = track.values[-1]
        elif family is not None:
            value = (
                _json_counter(spec, name, family)
                if kind == "counter"
                else _json_histogram(spec, name, family)
            )
        if value is None and kind != "gauge" and fam_stream is not None:
            value = _jsonl_cumulative(spec, name, fam_stream, kind)
        if value is not None:
            return value
    return None


def _other_from_json(family: _JsonFamily) -> OtherMetric:
    kind = family.kind
    stats = [s for _, s, _ in family.series]
    if kind == "counter":
        rates = [r for r in (_num(s.get("rate")) for s in stats) if r is not None]
        totals = [t for t in (_num(s.get("total")) for s in stats) if t is not None]
        return OtherMetric(
            kind=kind,
            mean=_sig(sum(rates)) if rates else None,
            total=_sig(sum(totals)) if totals else None,
            series=len(stats),
        )
    if kind == "histogram":
        counts = [_num(s.get("count")) or 0.0 for s in stats]
        sums = [_num(s.get("sum")) or 0.0 for s in stats]
        p99 = [p for p in (_num(s.get("p99_estimate")) for s in stats) if p is not None]
        total = sum(counts)
        return OtherMetric(
            kind=kind,
            mean=_sig(sum(sums) / total) if total else None,
            max=_sig(max(p99)) if p99 else None,
            total=_sig(total),
            series=len(stats),
        )
    avgs = [a for a in (_num(s.get("avg")) for s in stats) if a is not None]
    maxima = [m for m in (_num(s.get("max")) for s in stats) if m is not None]
    return OtherMetric(
        kind=kind,
        mean=_sig(sum(avgs) / len(avgs)) if avgs else None,
        max=_sig(max(maxima)) if maxima else None,
        series=len(stats),
    )


def _other_from_stream(fam: _FamilyStream) -> OtherMetric:
    if fam.histogram:
        deltas = [d for d in (_delta(e) for e in fam.ends.values()) if d is not None]
        count = sum(d[0][0] for d in deltas)
        total = sum(d[0][1] for d in deltas)
        return OtherMetric(
            kind="histogram",
            mean=_sig(total / count) if count else None,
            total=_sig(count),
            series=len(fam.ends),
        )
    return OtherMetric(
        kind="gauge",
        mean=_sig(fam.total / fam.n) if fam.n else None,
        max=_sig(fam.peak) if fam.n else None,
        last=_sig(fam.last),
        series=len(fam.ends),
    )


def _merge_other(items: Sequence[OtherMetric]) -> OtherMetric:
    if len(items) == 1:
        return items[0].model_copy(update={"endpoints": 1})
    kind = items[0].kind
    means = [i.mean for i in items if i.mean is not None]
    maxima = [i.max for i in items if i.max is not None]
    lasts = [i.last for i in items if i.last is not None]
    totals = [i.total for i in items if i.total is not None]
    summed = kind == "counter"

    def reduce(values: list[float]) -> float | None:
        if not values:
            return None
        return _sig(sum(values) if summed else sum(values) / len(values))

    return OtherMetric(
        kind=kind,
        mean=reduce(means),
        max=_sig(max(maxima)) if maxima else None,
        last=reduce(lasts),
        total=_sig(sum(totals)) if totals else None,
        series=sum(i.series for i in items),
        endpoints=len(items),
    )


def _loopback_alias(url: str, known: Mapping[str, EndpointDescriptor]) -> str | None:
    """The described URL a loopback scrape duplicates (same port and path)."""
    parts = urlsplit(url)
    if (parts.hostname or "") not in LOOPBACK_HOSTS:
        return None
    hits = []
    for candidate in known:
        other = urlsplit(candidate)
        if (other.port, other.path) == (parts.port, parts.path):
            hits.append(candidate)
    return hits[0] if len(hits) == 1 else None


def _total(metrics: Mapping[str, _Value], key: str) -> float | None:
    value = metrics.get(key)
    return value.total if value is not None else None


def _per_group_rate(metrics: Mapping[str, _Value], key: str) -> float | None:
    """Counter rate per scheduler group (TP ranks collapse to one group)."""
    value = metrics.get(key)
    if value is None or value.rate is None or value.groups < 1:
        return None
    return value.rate / value.groups


def _derived(metrics: Mapping[str, _Value]) -> dict[str, float]:
    """Ratios of one role's counters over the window (absent inputs: no key).

    ``spec_tokens_per_verify`` is the request-weighted acceptance length
    (generated tokens per verify step; it includes each request's first
    token, negligible for long outputs). ``decode_cuda_graph_fraction`` is
    the share of decode passes replayed from a CUDA graph (a missing mode
    series counts as zero passes). ``forward_busy_fraction`` (device timer)
    and the ``est_*`` rates (MFU metrics) are per scheduler group.
    """
    out: dict[str, float] = {}

    def put(key: str, value: float | None) -> None:
        if value is not None and math.isfinite(value):
            out[key] = _sig(value) or 0.0

    gen, verify = _total(metrics, "generation_tokens"), _total(
        metrics, "spec_verify_calls"
    )
    if gen and verify:
        put("spec_tokens_per_verify", gen / verify)
    prompt, cached = _total(metrics, "prompt_tokens"), _total(metrics, "cached_tokens")
    if prompt and cached is not None:
        put("cached_token_fraction", cached / prompt)
    graph = _total(metrics, "cuda_graph_decode_cuda_graph")
    eager = _total(metrics, "cuda_graph_decode_none")
    if graph is not None or eager is not None:
        passes = (graph or 0.0) + (eager or 0.0)
        if passes > 0:
            put("decode_cuda_graph_fraction", (graph or 0.0) / passes)
    retracted = [
        t
        for t in (
            _total(metrics, "retracted_input_tokens"),
            _total(metrics, "retracted_output_tokens"),
        )
        if t is not None
    ]
    if retracted and gen:
        put("retracted_tokens_per_generated", sum(retracted) / gen)
    evicted = _total(metrics, "evicted_tokens")
    if evicted is not None and prompt:
        put("evicted_tokens_per_prompt", evicted / prompt)
    busy = _per_group_rate(metrics, "forward_s")
    if busy is not None:
        put("forward_busy_fraction", min(busy, 1.0))
    flops = _per_group_rate(metrics, "est_flops")
    if flops is not None:
        put("est_tflops_per_gpu", flops / 1e12)
    reads = _per_group_rate(metrics, "est_read_bytes")
    writes = _per_group_rate(metrics, "est_write_bytes")
    if reads is not None or writes is not None:
        put("est_mem_gbs_per_gpu", ((reads or 0.0) + (writes or 0.0)) / 1e9)
    return out


def summarize_service_state(
    export: Mapping[str, Any] | None,
    jsonl_lines: Iterable[str] | None,
    endpoints: Sequence[EndpointDescriptor],
    *,
    window: tuple[float, float] | None = None,
    notes: Sequence[str] = (),
) -> ServiceState:
    """Summarize one point's server-metrics export (module docstring).

    ``export`` is the parsed ``server_metrics_export.json`` (or None),
    ``jsonl_lines`` the lines of ``server_metrics_export.jsonl`` (streamed
    once; or None), ``endpoints`` from ``endpoint_descriptors(plan.
    containers)``. ``window`` (unix seconds) restricts the JSONL; without it
    the JSON's profiling range is used, else the scrapes tagged
    ``benchmark_phase == "profiling"``.
    """
    out_notes = list(notes)
    parsed = _parse_export(export, out_notes)
    effective = window or (parsed.window if parsed else None)
    stream: _Stream | None = None
    if jsonl_lines is not None:
        stream = _Stream(window=effective)
        _stream_lines(jsonl_lines, stream)
        if stream.bad:
            out_notes.append(f"{stream.bad} unreadable server-metrics JSONL line(s)")
        if not stream.scrapes:
            out_notes.append("no server-metrics JSONL scrapes inside the window")
    source = {
        (False, False): "none",
        (True, False): "json",
        (False, True): "jsonl",
        (True, True): "json+jsonl",
    }[(parsed is not None and bool(parsed.families), bool(stream and stream.scrapes))]
    known = {normalize_url(d.url): d for d in endpoints}
    seen = set(parsed.families if parsed else ()) | set(
        stream.families if stream else ()
    )
    assign: dict[str, EndpointDescriptor] = {}
    unmatched: list[str] = []
    for url in sorted(seen):
        if url in known:
            assign[url] = known[url]
            continue
        alias = _loopback_alias(url, known)
        if alias is not None and alias in seen:
            out_notes.append(f"{url} duplicates {alias}; left out of the roles")
        elif alias is not None:
            assign[url] = known[alias]
        else:
            unmatched.append(url)
            assign[url] = EndpointDescriptor(url=url, role="unknown")
    described = {d.url for d in assign.values()}
    missing = sorted(u for u in known if u not in described)
    if missing:
        out_notes.append(f"{len(missing)} metrics endpoint(s) never scraped")

    values: dict[str, dict[str, _Value]] = {}
    states: list[EndpointState] = []
    others: dict[str, dict[str, list[OtherMetric]]] = {}
    for url, desc in sorted(assign.items(), key=lambda kv: (kv[1].role, kv[0])):
        json_fams = parsed.families.get(url, {}) if parsed else {}
        stream_fams = stream.families.get(url, {}) if stream else {}
        tracks = stream.gauges.get(url, {}) if stream else {}
        found: dict[str, _Value] = {}
        for spec in CATALOG:
            value = _catalog_value(spec, json_fams, stream_fams, tracks)
            if value is not None:
                found[spec.key] = value
        values[url] = found
        role_others = others.setdefault(desc.role, {})
        for name in sorted(set(json_fams) | set(stream_fams)):
            if name in CATALOG_FAMILIES or name.endswith("_info"):
                continue
            if name in json_fams:
                item = _other_from_json(json_fams[name])
                fam_stream = stream_fams.get(name)
                if fam_stream is not None and fam_stream.last is not None:
                    last = (
                        fam_stream.last_sum
                        if item.kind == "counter"
                        else fam_stream.last
                    )
                    item = item.model_copy(update={"last": _sig(last)})
            else:
                item = _other_from_stream(stream_fams[name])
            role_others.setdefault(name, []).append(item)
        info = parsed.info.get(url, {}) if parsed else {}
        states.append(
            EndpointState(
                url=url,
                role=desc.role,
                node=desc.node,
                container=desc.container,
                scraped=bool(json_fams or stream_fams),
                scrapes=stream.scrapes.get(url, 0) if stream else 0,
                fetch_latency_ms=_sig(_num(info.get("avg_fetch_latency_ms"))),
                metrics={k: v.model() for k, v in found.items()},
            )
        )
    roles: dict[str, RoleState] = {}
    role_names = sorted(
        {d.role for d in assign.values()},
        key=lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else len(ROLE_ORDER),
    )
    for role in role_names:
        urls = [u for u, d in assign.items() if d.role == role]
        merged: dict[str, _Value] = {}
        for spec in CATALOG:
            parts = [values[u][spec.key] for u in urls if spec.key in values[u]]
            if parts:
                merged[spec.key] = _combine(parts, spec.additive)
        roles[role] = RoleState(
            role=role,
            endpoints=len(urls),
            metrics={k: v.model() for k, v in merged.items()},
            derived=_derived(merged),
            other={
                name: _merge_other(items)
                for name, items in sorted(others.get(role, {}).items())
            },
        )
    return ServiceState(
        source=source,  # type: ignore[arg-type]
        aiperf_version=parsed.version if parsed else None,
        window_start_s=effective[0] if effective else None,
        window_end_s=effective[1] if effective else None,
        roles=roles,
        endpoints=tuple(states),
        unmatched_urls=tuple(unmatched),
        missing_urls=tuple(missing),
        notes=tuple(dict.fromkeys(out_notes)),
    )


# --------------------------------------------------------------------------
# Bottleneck classification
# --------------------------------------------------------------------------

BottleneckLabel = Literal[
    "kv_capacity",
    "batch_cap",
    "prefill",
    "decode",
    "pd_transfer",
    "frontend",
    "network",
    "power_thermal",
    "client",
    "unknown",
]
# Transparent thresholds of ``classify_bottleneck``.
KV_USAGE_HIGH = 0.90  # sglang:token_usage avg: the KV pool is the limit
KV_USAGE_PEAK = 0.97  # peak usage that counts as full
KV_USAGE_LOW = 0.50
BACKLOG_REQS = 1.0  # avg waiting requests per instance that form a backlog
PD_QUEUE_REQS = 1.0  # avg requests per instance in a PD transfer queue
BATCH_CAP_FRACTION = 0.90  # hottest scheduler's running avg / its cap
LOW_RUNNING_FRACTION = 0.50
FRONTEND_QUEUE_REQS = 1.0  # dynamo_frontend_queued_requests avg
FRONTEND_EXCESS = 0.25  # frontend inflight above what workers report
IB_UTIL_HIGH = 0.80  # peak IB / speed-of-light reference
IB_CONGESTION_HIGH = 0.05  # port_xmit_wait stalled fraction (UNVERIFIED tick)
THROTTLE_HIGH = 0.10  # share of GPU samples with a limiting clocks reason
GPU_UTIL_BUSY_PCT = 70.0  # gpu_util_mean in percent (nvidia-smi semantics)
GPU_UTIL_IDLE_PCT = 40.0
CUDA_GRAPH_LOW = 0.90  # decode passes replayed from a CUDA graph
RETRACTED_TOKENS_HIGH = 0.05  # retracted (recomputed) tokens per generated token
EVICTED_PER_PROMPT_HIGH = 0.50  # prefix-cache tokens evicted per prompt token
MIN_SCORE = 0.35  # below this the point is "unknown"
AMBIGUOUS_RATIO = 0.8  # runner-up within this share of the winner: confidence x 0.75
# Tie-break order when scores are equal.
LABEL_PRIORITY: tuple[BottleneckLabel, ...] = (
    "kv_capacity",
    "batch_cap",
    "pd_transfer",
    "prefill",
    "decode",
    "network",
    "power_thermal",
    "frontend",
    "client",
)


class Bottleneck(Frozen):
    """Why a point saturated: a label, a 0..1 confidence and the evidence."""

    label: BottleneckLabel
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    scores: dict[str, float] = Field(default_factory=dict)
    secondary: BottleneckLabel | None = None


@dataclass
class _Scores:
    points: dict[str, float] = field(default_factory=dict)
    lines: dict[str, list[str]] = field(default_factory=dict)

    def add(self, label: BottleneckLabel, weight: float, line: str) -> None:
        self.points[label] = self.points.get(label, 0.0) + weight
        self.lines.setdefault(label, []).append(line)


@dataclass(frozen=True)
class _RoleView:
    role: str
    instances: int
    queue: float | None  # avg waiting requests per instance
    running: float | None  # per instance
    hottest: float | None  # busiest scheduler's running avg
    kv: float | None
    kv_peak: float | None
    retracted: bool
    bootstrap: float | None
    inflight_queue: float | None
    prealloc: float | None
    transfer: float | None
    failures: float | None
    worker_inflight: float | None  # role total, Dynamo work handler


def _per_instance(service: ServiceState, role: str, key: str, n: int) -> float | None:
    total = service.value(role, key)
    return None if total is None else total / max(1, n)


def _role_view(service: ServiceState, role: str) -> _RoleView:
    n = service.roles[role].endpoints
    kv = service.value(role, "kv_usage")
    kv_peak = service.value(role, "kv_usage", "max")
    if kv is None:
        kv = service.value(role, "kv_usage_dynamo")
        kv_peak = service.value(role, "kv_usage_dynamo", "max")
    retraction = service.metric(role, "retracted_reqs")
    # A gauge that was non-zero in the window, or a counter that grew in it.
    retracted = retraction is not None and (
        (retraction.max or 0.0) > 0 or (retraction.total or 0.0) > 0
    )
    failures = [
        service.value(role, k, "total")
        for k in ("transfer_failed_reqs", "bootstrap_failed_reqs")
    ]
    known_failures = [f for f in failures if f is not None]
    return _RoleView(
        role=role,
        instances=n,
        queue=_per_instance(service, role, "queue_reqs", n),
        running=_per_instance(service, role, "running_reqs", n),
        hottest=service.value(role, "running_reqs", "rank_avg_max"),
        kv=kv,
        kv_peak=kv_peak,
        retracted=retracted,
        bootstrap=_per_instance(service, role, "prefill_bootstrap_queue", n),
        inflight_queue=_per_instance(service, role, "prefill_inflight_queue", n),
        prealloc=_per_instance(service, role, "decode_prealloc_queue", n),
        transfer=_per_instance(service, role, "decode_transfer_queue", n),
        failures=sum(known_failures) if known_failures else None,
        worker_inflight=service.value(role, "worker_inflight"),
    )


def _cap_for(
    role: str,
    max_running_requests: int | Mapping[str, int] | None,
    service: ServiceState,
) -> float | None:
    if isinstance(max_running_requests, Mapping):
        cap = max_running_requests.get(role)
    else:
        cap = max_running_requests
    if cap:
        return float(cap)
    fallback = service.value("frontend", "model_max_num_seqs", "max")
    return fallback if fallback else None


def _ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _kv_rules(scores: _Scores, view: _RoleView, decode_waits: bool) -> None:
    high = _ge(view.kv, KV_USAGE_HIGH)
    peak = _ge(view.kv_peak, KV_USAGE_PEAK)
    if view.role == "prefill" and (high or peak) and decode_waits:
        scores.add(
            "pd_transfer",
            0.3,
            f"prefill: KV usage avg {_fmt(view.kv)} held while decode-side "
            "queues wait (KV parked on prefill until transfer)",
        )
        return
    if high:
        scores.add(
            "kv_capacity",
            0.45,
            f"{view.role}: KV cache usage avg {_fmt(view.kv)} >= {KV_USAGE_HIGH}",
        )
    elif peak:
        scores.add(
            "kv_capacity",
            0.2,
            f"{view.role}: KV cache usage peaked at {_fmt(view.kv_peak)} >= "
            f"{KV_USAGE_PEAK}",
        )
    backlog = max(view.queue or 0.0, view.prealloc or 0.0)
    if (high or peak) and backlog >= BACKLOG_REQS:
        scores.add(
            "kv_capacity",
            0.25,
            f"{view.role}: {backlog:.1f} requests waiting per instance while KV is "
            "full: admission blocked by KV space",
        )
    if view.retracted:
        scores.add(
            "kv_capacity",
            0.25,
            f"{view.role}: requests were retracted (KV pool exhausted mid-decode)",
        )


def _batch_rules(scores: _Scores, view: _RoleView, cap: float | None) -> None:
    running = view.hottest if view.hottest is not None else view.running
    if cap is None or running is None:
        return
    fraction = running / cap
    if fraction >= BATCH_CAP_FRACTION:
        scores.add(
            "batch_cap",
            0.45,
            f"{view.role}: busiest scheduler runs {running:.0f} requests on average "
            f"= {fraction:.0%} of max_running_requests {cap:.0f}",
        )
        if _ge(view.queue, BACKLOG_REQS):
            scores.add(
                "batch_cap",
                0.25,
                f"{view.role}: {view.queue:.1f} requests waiting per instance "
                "behind the full batch",
            )
        if view.kv is not None and view.kv < KV_USAGE_HIGH:
            scores.add(
                "batch_cap",
                0.1,
                f"{view.role}: KV usage {_fmt(view.kv)} still has headroom",
            )


def _admission_rules(
    scores: _Scores, views: Mapping[str, _RoleView], caps: Mapping[str, float | None]
) -> None:
    agg = views.get("agg")
    if agg is not None and _ge(agg.queue, BACKLOG_REQS):
        kv_room = agg.kv is None or agg.kv < KV_USAGE_HIGH
        cap = caps.get("agg")
        running = agg.hottest if agg.hottest is not None else agg.running
        below_cap = cap is None or running is None or running / cap < BATCH_CAP_FRACTION
        if kv_room and below_cap:
            scores.add(
                "prefill",
                0.45 if cap is not None else 0.3,
                f"agg: {agg.queue:.1f} requests waiting per instance with KV usage "
                f"{_fmt(agg.kv)} and running {_fmt(running, 0)} below the cap "
                f"{_fmt(cap, 0)}: admission limited by prefill throughput",
            )
    prefill, decode = views.get("prefill"), views.get("decode")
    if prefill is not None:
        if _ge(prefill.queue, BACKLOG_REQS):
            scores.add(
                "prefill",
                0.5,
                f"prefill: {prefill.queue:.1f} requests waiting per prefill instance",
            )
        if _ge(prefill.bootstrap, BACKLOG_REQS):
            label: BottleneckLabel = (
                "kv_capacity"
                if decode is not None and _ge(decode.kv, KV_USAGE_HIGH)
                else "decode"
            )
            scores.add(
                label,
                0.3,
                f"prefill: {prefill.bootstrap:.1f} requests per instance wait for "
                "decode-side bootstrap (decode is not accepting)",
            )
        if _ge(prefill.inflight_queue, PD_QUEUE_REQS):
            scores.add(
                "pd_transfer",
                0.35,
                f"prefill: {prefill.inflight_queue:.1f} finished prefills per "
                "instance wait for their KV transfer",
            )
    if decode is not None:
        if _ge(decode.transfer, PD_QUEUE_REQS):
            scores.add(
                "pd_transfer",
                0.35,
                f"decode: {decode.transfer:.1f} requests per instance wait for KV "
                "transfer",
            )
        kv_room = decode.kv is None or decode.kv < KV_USAGE_HIGH
        if _ge(decode.prealloc, BACKLOG_REQS) and kv_room:
            scores.add(
                "decode",
                0.3,
                f"decode: {decode.prealloc:.1f} requests per instance wait for "
                "decode admission with KV headroom",
            )
        cap = caps.get("decode")
        running = decode.hottest if decode.hottest is not None else decode.running
        below_cap = cap is None or running is None or running / cap < BATCH_CAP_FRACTION
        if _ge(decode.queue, BACKLOG_REQS) and kv_room and below_cap:
            scores.add(
                "decode",
                0.4,
                f"decode: {decode.queue:.1f} requests waiting per decode instance "
                "below the batch cap",
            )
    for view in views.values():
        if view.failures:
            scores.add(
                "pd_transfer",
                0.2,
                f"{view.role}: {view.failures:.0f} KV transfer/bootstrap failures",
            )


def _frontend_rules(
    scores: _Scores,
    service: ServiceState,
    views: Mapping[str, _RoleView],
    worker_backlog: bool,
) -> None:
    queued = service.value("frontend", "fe_queued")
    if _ge(queued, FRONTEND_QUEUE_REQS):
        scores.add(
            "frontend",
            0.45 if not worker_backlog else 0.15,
            f"frontend: {queued:.1f} requests queued before their first response",
        )
    pending = service.value("frontend", "router_queue_pending")
    if _ge(pending, FRONTEND_QUEUE_REQS):
        scores.add(
            "frontend",
            0.3,
            f"router: {pending:.1f} requests pending in the KV router queue",
        )
    inflight = service.value("frontend", "fe_inflight")
    if inflight is None or worker_backlog:
        return
    handler = [v.worker_inflight for v in views.values() if v.worker_inflight]
    if handler:
        visible = sum(handler)
        weight, what = 0.25, "the workers' Dynamo handlers"
    else:
        visible = sum(
            ((v.running or 0.0) + (v.queue or 0.0)) * v.instances
            for v in views.values()
        )
        weight, what = 0.15, "SGLang schedulers (may cover one DP rank)"
    if visible > 0 and inflight > (1 + FRONTEND_EXCESS) * visible + 2:
        scores.add(
            "frontend",
            weight,
            f"frontend: {inflight:.0f} requests in flight but only {visible:.0f} "
            f"visible at {what}: tokenization, routing or streaming holds them",
        )


def _runtime_rules(
    scores: _Scores, service: ServiceState, observations: list[str]
) -> None:
    """CUDA graph coverage, retraction and prefix-cache eviction (derived)."""
    for role in service.worker_roles:
        derived = service.roles[role].derived
        graph = derived.get("decode_cuda_graph_fraction")
        if graph is not None and role in ("agg", "decode") and graph < CUDA_GRAPH_LOW:
            line = (
                f"{role}: only {graph:.0%} of decode passes replayed a CUDA graph "
                "(running batch above the largest captured size?)"
            )
            observations.append(line)
            scores.add("decode", 0.2, line)
        retracted = derived.get("retracted_tokens_per_generated")
        if retracted is not None and retracted >= RETRACTED_TOKENS_HIGH:
            scores.add(
                "kv_capacity",
                0.2,
                f"{role}: retracted requests recompute {retracted:.0%} extra tokens "
                "per generated token",
            )
        evicted = derived.get("evicted_tokens_per_prompt")
        if evicted is not None and evicted >= EVICTED_PER_PROMPT_HIGH:
            scores.add(
                "kv_capacity",
                0.15,
                f"{role}: prefix cache evicted {evicted:.2f} tokens per prompt "
                "token: the KV pool is smaller than the multi-turn working set",
            )


def _activity_rules(
    scores: _Scores,
    activity: Mapping[str, Any],
    observations: list[str],
    worker_backlog: bool,
) -> None:
    """GPU regimes (NVML GPM) per serving role as evidence."""
    for role in WORKER_ROLES:
        agg = activity.get(role)
        regime = getattr(agg, "regime", None)
        reason = str(getattr(agg, "reason", "") or "")
        if regime is None or regime == "unknown":
            continue
        observations.append(f"{role} GPUs: {regime} ({reason})")
        if role == "prefill" and regime == "compute":
            scores.add("prefill", 0.25, f"prefill GPUs compute-bound: {reason}")
        elif role in ("agg", "decode") and regime in ("memory", "compute"):
            scores.add(
                "decode",
                0.25,
                f"{role} GPUs {'memory-bandwidth' if regime == 'memory' else 'compute'}"
                f"-bound: {reason}",
            )
        elif regime in ("waiting", "idle") and not worker_backlog:
            scores.add(
                "client",
                0.15,
                f"{role} GPUs {regime} with no queue anywhere: {reason}",
            )


def classify_bottleneck(
    service: ServiceState | None,
    *,
    max_running_requests: int | Mapping[str, int] | None = None,
    gpu_util_mean: float | None = None,
    ib_util_max: float | None = None,
    throttled_fraction: float | None = None,
    ib_congestion: float | None = None,
    activity: Mapping[str, Any] | None = None,
) -> Bottleneck:
    """Name what limited a point, from its ``ServiceState`` and optional
    hardware context; every rule and threshold is a module constant.

    ``activity`` maps a role to its NVML GPM aggregate (``gpuactivity.
    RoleActivity``: ``regime`` compute | memory | waiting | idle | mixed |
    unknown, and ``reason``); ``gpu_util_mean`` is then GPM's "any kernel
    active" share.

    ``max_running_requests``: the per-scheduler cap (int, or role -> cap; the
    SGLang "max_running_requests=" startup line, see ``lifecycle``);
    ``gpu_util_mean`` in percent; ``ib_util_max`` peak IB / reference (0..1);
    ``throttled_fraction`` and ``ib_congestion`` from ``PointStress``
    (``throttled_fraction``, ``xmit_wait_fraction_max``).
    """
    scores = _Scores()
    observations: list[str] = []
    views: dict[str, _RoleView] = {}
    if service is not None:
        views = {r: _role_view(service, r) for r in service.worker_roles}
    if not views:
        observations.append("no worker metrics in the server-metrics export")
    caps = {r: _cap_for(r, max_running_requests, service) for r in views}  # type: ignore[arg-type]
    decode_waits = any(
        _ge(getattr(views.get("decode"), name, None), BACKLOG_REQS)
        for name in ("prealloc", "transfer", "queue")
    ) or _ge(getattr(views.get("prefill"), "bootstrap", None), BACKLOG_REQS)
    for view in views.values():
        observations.append(
            f"{view.role} x{view.instances}: running {_fmt(view.running, 1)}/inst "
            f"(hottest {_fmt(view.hottest, 1)}, cap {_fmt(caps.get(view.role), 0)}), "
            f"waiting {_fmt(view.queue, 1)}/inst, KV {_fmt(view.kv)} "
            f"(peak {_fmt(view.kv_peak)})"
        )
        _kv_rules(scores, view, decode_waits)
        _batch_rules(scores, view, caps.get(view.role))
    _admission_rules(scores, views, caps)
    worker_backlog = any(
        _ge(max(v.queue or 0.0, v.prealloc or 0.0, v.bootstrap or 0.0), BACKLOG_REQS)
        for v in views.values()
    )
    if service is not None:
        _frontend_rules(scores, service, views, worker_backlog)
        _runtime_rules(scores, service, observations)
    if activity:
        _activity_rules(scores, activity, observations, worker_backlog)
    transfer_symptoms = "pd_transfer" in scores.points
    if _ge(ib_util_max, IB_UTIL_HIGH):
        scores.add(
            "network",
            0.35 + (0.15 if transfer_symptoms else 0.0),
            f"IB peak at {ib_util_max:.0%} of the speed-of-light reference",
        )
    if _ge(ib_congestion, IB_CONGESTION_HIGH):
        scores.add(
            "network",
            0.2,
            f"HCA transmit stalled ~{ib_congestion:.1%} of the time "
            "(port_xmit_wait; tick duration UNVERIFIED)",
        )
    if _ge(throttled_fraction, THROTTLE_HIGH):
        scores.add(
            "power_thermal",
            0.3 + min(0.4, throttled_fraction or 0.0),
            f"GPU clocks limited (power cap / slowdown / thermal) in "
            f"{throttled_fraction:.0%} of samples",
        )
        if _ge(gpu_util_mean, GPU_UTIL_BUSY_PCT):
            scores.add(
                "power_thermal", 0.1, f"GPUs busy ({gpu_util_mean:.0f}% utilization)"
            )
    kv_values = [v.kv for v in views.values() if v.kv is not None]
    kv_low = bool(kv_values) and max(kv_values) < KV_USAGE_LOW
    queued_fe = service.value("frontend", "fe_queued") if service else None
    if views and not worker_backlog and not _ge(queued_fe, FRONTEND_QUEUE_REQS):
        if gpu_util_mean is not None and gpu_util_mean < GPU_UTIL_IDLE_PCT:
            scores.add(
                "client",
                0.5,
                f"no queue anywhere and GPUs only {gpu_util_mean:.0f}% busy: the "
                "client does not offer enough load",
            )
        elif gpu_util_mean is None and kv_low:
            scores.add(
                "client",
                0.25,
                "no queue anywhere and KV usage below "
                f"{KV_USAGE_LOW}: load likely client-limited",
            )
        if gpu_util_mean is not None and gpu_util_mean >= GPU_UTIL_BUSY_PCT:
            busy = [r for r in ("decode", "agg") if r in views]
            if busy:
                scores.add(
                    "decode",
                    0.4,
                    f"{'/'.join(busy)}: no queue while GPUs are "
                    f"{gpu_util_mean:.0f}% busy: decode step time bounds "
                    "throughput",
                )
    return _verdict(scores, observations)


def _verdict(scores: _Scores, observations: Sequence[str]) -> Bottleneck:
    ranked = sorted(
        scores.points.items(),
        key=lambda kv: (-kv[1], LABEL_PRIORITY.index(kv[0])),  # type: ignore[arg-type]
    )
    rounded = {label: round(min(1.0, s), 3) for label, s in ranked}
    if not ranked or ranked[0][1] < MIN_SCORE:
        evidence = [
            f"no rule reached {MIN_SCORE}; observed: " + line for line in observations
        ]
        for label, _ in ranked:
            evidence.extend(f"{label}: {line}" for line in scores.lines[label])
        return Bottleneck(
            label="unknown",
            confidence=round(ranked[0][1], 3) if ranked else 0.0,
            evidence=tuple(evidence),
            scores=rounded,
        )
    label, top = ranked[0]
    confidence = min(1.0, top)
    secondary = None
    if len(ranked) > 1 and ranked[1][1] >= MIN_SCORE:
        secondary = ranked[1][0]
        if ranked[1][1] >= AMBIGUOUS_RATIO * top:
            confidence *= 0.75
    evidence = list(scores.lines[label])
    if secondary is not None:
        evidence.extend(f"also {secondary}: {line}" for line in scores.lines[secondary])
    evidence.extend(observations)
    return Bottleneck(
        label=label,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        evidence=tuple(evidence),
        scores=rounded,
        secondary=secondary,  # type: ignore[arg-type]
    )
