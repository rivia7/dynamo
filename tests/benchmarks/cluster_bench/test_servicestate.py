# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/servicestate.py and its loader in adapters/telemetry.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.cluster_bench.adapters.telemetry import load_service_state
from benchmarks.cluster_bench.domain.servicestate import (
    CATALOG,
    EndpointDescriptor,
    ServiceState,
    bucket_quantile,
    canonical_family,
    classify_bottleneck,
    endpoint_descriptors,
    normalize_url,
    summarize_service_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "servicestate"
FRONTEND = "http://10.42.0.1:8000/metrics"
PREFILL0 = "http://10.42.0.1:8081/metrics"
PREFILL1 = "http://10.42.0.2:8081/metrics"
DECODE0 = "http://10.42.0.3:8081/metrics"
START_S = 1793700100.0
END_S = START_S + 600.0


def descriptors() -> list[EndpointDescriptor]:
    return [
        EndpointDescriptor(url=FRONTEND, role="frontend", node="node0"),
        EndpointDescriptor(url=PREFILL0, role="prefill", node="node0"),
        EndpointDescriptor(url=PREFILL1, role="prefill", node="node1"),
        EndpointDescriptor(url=DECODE0, role="decode", node="node2"),
    ]


def export_json() -> dict[str, Any]:
    with open(FIXTURES / "server_metrics_export.json", encoding="utf-8") as fh:
        return json.load(fh)


def jsonl_lines() -> list[str]:
    with open(FIXTURES / "server_metrics_export.jsonl", encoding="utf-8") as fh:
        return fh.readlines()


@pytest.fixture(scope="module")
def state() -> ServiceState:
    return load_service_state(FIXTURES, descriptors())


# ----------------------------------------------------------------- summary


def test_fixture_summary_sources_window_and_endpoint_matching(
    state: ServiceState,
) -> None:
    assert state.source == "json+jsonl" and state.aiperf_version == "0.14.0"
    assert (state.window_start_s, state.window_end_s) == (START_S, END_S)
    assert list(state.roles) == ["frontend", "prefill", "decode"]
    assert [r.endpoints for r in state.roles.values()] == [1, 2, 1]
    # AIPerf also scrapes <--url>/metrics: the loopback copy is dropped.
    assert any("127.0.0.1:8000" in n and "duplicates" in n for n in state.notes)
    assert "1 unreadable server-metrics JSONL line(s)" in state.notes  # torn tail
    assert state.unmatched_urls == () and state.missing_urls == ()
    by_url = {e.url: e for e in state.endpoints}
    assert by_url[DECODE0].scrapes == 5 and by_url[DECODE0].fetch_latency_ms == 6.87
    assert by_url[FRONTEND].node == "node0" and by_url[FRONTEND].scraped


def test_decode_gauges_combine_dp_ranks_and_collapse_tp_replicas(
    state: ServiceState,
) -> None:
    running = state.metric("decode", "running_reqs")
    assert running is not None and running.source_kind == "jsonl"
    # Two DP-rank groups (a TP1 copy of DP0 collapses), summed per scrape.
    assert (running.series, running.groups) == (3, 2)
    assert running.avg == 242.2 and running.max == 249.0 and running.last == 249.0
    assert running.rank_avg_max == 121.4 and not running.approx
    kv = state.metric("decode", "kv_usage")
    assert kv is not None and kv.avg == 0.957 and kv.max == 0.985  # mean of ranks
    assert state.value("decode", "decode_prealloc_queue") == 12.8
    assert state.value("decode", "retracted_reqs", "max") == 4.0
    assert state.roles["decode"].derived == {"spec_tokens_per_verify": 2.71687}


def test_prefill_role_totals_and_counters_from_json(state: ServiceState) -> None:
    running = state.metric("prefill", "running_reqs")
    assert running is not None
    assert running.avg == 5.2 and running.rank_avg_max == 2.6  # 2 instances
    assert running.approx  # role max is a sum of per-instance maxima
    prompt = state.metric("prefill", "prompt_tokens")
    assert prompt is not None and prompt.kind == "counter"
    assert prompt.source == "sglang:prompt_tokens"  # AIPerf drops "_total"
    assert prompt.rate_per_s == pytest.approx(61000.0 * 2.05, rel=1e-6)
    ttft = state.metric("prefill", "ttft")
    assert ttft is not None and ttft.kind == "histogram" and ttft.avg == 2.6
    assert ttft.total == 2338.0 and 1.5 < ttft.p50 < 2.5 and ttft.p99 <= 40.0
    assert state.roles["prefill"].derived["cached_token_fraction"] == pytest.approx(
        50100 / 61000, rel=1e-5
    )


def test_frontend_metrics_and_label_filtered_errors(state: ServiceState) -> None:
    assert state.value("frontend", "fe_queued") == 11.0
    assert state.value("frontend", "fe_inflight") == 256.4
    errors = state.metric("frontend", "fe_errors")
    requests = state.metric("frontend", "fe_requests")
    assert errors is not None and requests is not None
    assert errors.total == 3.0 and requests.total == 2278.0  # success + error
    ttft = state.metric("frontend", "fe_ttft")
    assert ttft is not None and ttft.avg == 4.8 and ttft.rate_per_s == 3.78977
    assert state.metric("frontend", "router_kv_hit_rate") is not None
    assert state.value("frontend", "model_max_num_seqs", "max") == 256.0


def test_stage_latencies_come_from_the_stage_label(state: ServiceState) -> None:
    forward = state.metric("prefill", "stage_prefill_forward")
    bootstrap = state.metric("prefill", "stage_prefill_bootstrap")
    assert forward is not None and bootstrap is not None
    assert forward.kind == "histogram" and forward.avg == pytest.approx(0.4)
    assert bootstrap.avg == pytest.approx(1.4)
    # decode_forward is not an observed stage in SGLang 0.5.21 / 0.5.23: not extracted
    assert state.metric("prefill", "stage_decode_waiting") is None


def test_catch_all_keeps_unknown_families_compactly(state: ServiceState) -> None:
    other = state.roles["prefill"].other
    assert set(other) >= {
        "sglang:num_grammar_queue_reqs",
        "dynamo_component_request_bytes",
    }
    # per-stage latency is a catalog family now (one key per observed stage)
    assert "sglang:per_stage_req_latency_seconds" not in other
    counter = other["dynamo_component_request_bytes"]
    assert counter.kind == "counter" and counter.endpoints == 2
    assert counter.mean == pytest.approx(152000 * 2.05, rel=1e-5)  # rate sum
    busy = state.roles["frontend"].other["dynamo_tokio_worker_busy_ratio"]
    assert (busy.mean, busy.max, busy.last) == (0.338, 0.36, 0.34)
    catalog_names = {name for spec in CATALOG for name in spec.families}
    assert not catalog_names & set(other)


def test_json_only_and_jsonl_only_agree_on_cumulative_metrics() -> None:
    json_only = summarize_service_state(export_json(), None, descriptors())
    jsonl_only = summarize_service_state(None, jsonl_lines(), descriptors())
    assert json_only.source == "json" and jsonl_only.source == "jsonl"
    # Without the JSON the JSONL's benchmark_phase marks the window.
    assert jsonl_only.window_start_s is None
    for role, key in (
        ("prefill", "prompt_tokens"),
        ("decode", "generation_tokens"),
        ("frontend", "fe_ttft"),
        ("prefill", "kv_transfer_mb"),
    ):
        a, b = json_only.metric(role, key), jsonl_only.metric(role, key)
        assert a is not None and b is not None, (role, key)
        assert a.total == b.total and a.source_kind != b.source_kind
    # JSON-only gauges add per-rank averages; maxima become upper bounds.
    running = json_only.metric("decode", "running_reqs")
    assert running is not None and running.avg == 242.2 and running.approx
    assert running.p50 is None and running.last is None


def test_window_argument_overrides_and_unmatched_endpoints() -> None:
    partial = [EndpointDescriptor(url=DECODE0, role="decode"), *descriptors()[:1]]
    extra = EndpointDescriptor(url="http://10.42.0.9:8081/metrics", role="agg")
    late = summarize_service_state(
        export_json(),
        jsonl_lines(),
        [*partial, extra],
        window=(START_S + 200.0, END_S),
    )
    assert late.window_start_s == START_S + 200.0
    assert set(late.unmatched_urls) == {PREFILL0, PREFILL1}
    assert late.missing_urls == ("http://10.42.0.9:8081/metrics",)
    assert late.roles["unknown"].endpoints == 2
    running = late.metric("decode", "running_reqs")
    # Only the scrapes at 300, 450 and 600 s remain (6 significant digits).
    assert running is not None
    assert running.avg == pytest.approx((245 + 239 + 249) / 3, rel=1e-5)


def test_loader_is_fail_soft(tmp_path: Path) -> None:
    empty = load_service_state(tmp_path / "missing", descriptors())
    assert empty.source == "none" and empty.roles == {}
    assert empty.missing_urls == tuple(sorted(d.url for d in descriptors()))
    (tmp_path / "server_metrics_export.json").write_text("{not json")
    broken = load_service_state(tmp_path, descriptors())
    assert any("unreadable" in n for n in broken.notes)


def test_endpoint_descriptors_from_launch_plan_containers() -> None:
    containers = [
        SimpleNamespace(
            name="cb-x-control", role="control", node="node0", metrics_url=None
        ),
        SimpleNamespace(
            name="cb-x-agg0-r0",
            role="agg",
            node="node1",
            metrics_url="http://10.42.0.2:8081/metrics/",
        ),
    ]
    (desc,) = endpoint_descriptors(containers)
    assert desc == EndpointDescriptor(
        url="http://10.42.0.2:8081/metrics",
        role="agg",
        node="node1",
        container="cb-x-agg0-r0",
    )


def test_url_and_name_helpers() -> None:
    assert normalize_url("10.42.0.1:8000") == "http://10.42.0.1:8000/metrics"
    assert normalize_url("HTTP://Node-A:8081/metrics/") == "http://node-a:8081/metrics"
    assert canonical_family("sglang:prompt_tokens_total") == "sglang:prompt_tokens"
    assert canonical_family("sglang:token_usage") == "sglang:token_usage"


def test_bucket_quantile_interpolates_like_prometheus() -> None:
    buckets = {0.1: 10.0, 0.5: 50.0, 1.0: 90.0, math.inf: 100.0}
    assert bucket_quantile(buckets, 0.5) == pytest.approx(0.5)
    assert bucket_quantile(buckets, 0.05) == pytest.approx(0.05)
    assert bucket_quantile(buckets, 0.95) == 1.0  # +Inf bucket: last finite bound
    assert bucket_quantile({}, 0.5) is None
    assert bucket_quantile({1.0: 0.0, math.inf: 0.0}, 0.5) is None


# ----------------------------------------------------------------- bottleneck


def series(url: str, stats: dict[str, Any], **labels: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"endpoint_url": url, "stats": stats}
    if labels:
        entry["labels"] = labels
    return entry


def gauge(avg: float, peak: float | None = None) -> dict[str, float]:
    return {"avg": avg, "max": avg if peak is None else peak}


def synthetic(metrics: dict[str, tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """A minimal AIPerf export: family -> (type, series)."""
    return {
        "aiperf_version": "0.14.0",
        "summary": {
            "phase_time_ranges": {
                "profiling": {
                    "start_ns": 1_000_000_000_000,
                    "end_ns": 1_600_000_000_000,
                }
            }
        },
        "metrics": {
            name: {"type": kind, "description": "", "series": items}
            for name, (kind, items) in metrics.items()
        },
    }


AGG = "http://10.42.0.2:8081/metrics"
PRE = "http://10.42.0.3:8081/metrics"
DEC = "http://10.42.0.4:8081/metrics"
FE = "http://10.42.0.1:8000/metrics"
ROLES = [
    EndpointDescriptor(url=AGG, role="agg"),
    EndpointDescriptor(url=PRE, role="prefill"),
    EndpointDescriptor(url=DEC, role="decode"),
    EndpointDescriptor(url=FE, role="frontend"),
]


def service(**families: tuple[str, list[dict[str, Any]]]) -> ServiceState:
    names = {key.replace("__", ":"): value for key, value in families.items()}
    return summarize_service_state(synthetic(names), None, ROLES)


def agg_state(running: float, queue: float, kv: float, **extra) -> ServiceState:
    return service(
        sglang__num_running_reqs=("gauge", [series(AGG, gauge(running))]),
        sglang__num_queue_reqs=("gauge", [series(AGG, gauge(queue))]),
        sglang__token_usage=("gauge", [series(AGG, gauge(kv, min(1.0, kv + 0.02)))]),
        **extra,
    )


def test_kv_capacity_wins_on_the_recorded_pd_point(state: ServiceState) -> None:
    verdict = classify_bottleneck(
        state, max_running_requests={"decode": 128, "prefill": 64}, gpu_util_mean=88.0
    )
    assert verdict.label == "kv_capacity" and verdict.confidence == 1.0
    assert verdict.secondary == "batch_cap"
    assert verdict.evidence[0] == "decode: KV cache usage avg 0.96 >= 0.9"
    assert any("retracted" in line for line in verdict.evidence)
    assert verdict.scores["kv_capacity"] == 1.0


def test_batch_cap_when_running_hits_the_cap_with_kv_headroom() -> None:
    verdict = classify_bottleneck(agg_state(250, 20, 0.6), max_running_requests=256)
    assert verdict.label == "batch_cap"
    assert "98% of max_running_requests 256" in verdict.evidence[0]


def test_prefill_when_agg_backlog_has_kv_and_batch_room() -> None:
    verdict = classify_bottleneck(agg_state(40, 30, 0.5), max_running_requests=256)
    assert verdict.label == "prefill" and verdict.confidence == 0.45
    unknown_cap = classify_bottleneck(agg_state(40, 30, 0.5))
    assert unknown_cap.label == "unknown" and unknown_cap.scores == {"prefill": 0.3}


def test_decode_when_gpus_busy_without_any_queue() -> None:
    verdict = classify_bottleneck(
        agg_state(120, 0, 0.7), max_running_requests=256, gpu_util_mean=93.0
    )
    assert verdict.label == "decode"


def test_client_when_nothing_queues_and_gpus_idle() -> None:
    verdict = classify_bottleneck(agg_state(8, 0, 0.1), gpu_util_mean=22.0)
    assert verdict.label == "client" and verdict.confidence == 0.5


def test_frontend_when_requests_wait_before_the_workers() -> None:
    state = agg_state(
        30,
        0,
        0.3,
        dynamo_frontend_queued_requests=("gauge", [series(FE, gauge(40.0))]),
        dynamo_frontend_inflight_requests=("gauge", [series(FE, gauge(200.0))]),
        dynamo_component_inflight_requests=("gauge", [series(AGG, gauge(31.0))]),
    )
    verdict = classify_bottleneck(state, gpu_util_mean=55.0)
    assert verdict.label == "frontend"
    assert any("visible at the workers' Dynamo handlers" in e for e in verdict.evidence)


def test_pd_transfer_and_network_overlay() -> None:
    state = service(
        sglang__num_prefill_inflight_queue_reqs=("gauge", [series(PRE, gauge(4.0))]),
        sglang__num_decode_transfer_queue_reqs=("gauge", [series(DEC, gauge(5.0))]),
        sglang__token_usage=(
            "gauge",
            [series(PRE, gauge(0.4)), series(DEC, gauge(0.5))],
        ),
        sglang__num_transfer_failed_reqs=(
            "counter",
            [series(DEC, {"total": 3.0, "rate": 0.005})],
        ),
    )
    verdict = classify_bottleneck(state)
    assert verdict.label == "pd_transfer"
    assert verdict.scores["pd_transfer"] == pytest.approx(0.9)
    congested = classify_bottleneck(state, ib_util_max=0.93, ib_congestion=0.12)
    assert congested.scores["network"] == pytest.approx(0.7)
    assert congested.label == "pd_transfer" and congested.secondary == "network"


def test_power_thermal_when_clocks_are_limited() -> None:
    verdict = classify_bottleneck(
        agg_state(120, 0, 0.7), gpu_util_mean=95.0, throttled_fraction=0.6
    )
    assert verdict.label == "power_thermal"
    assert verdict.scores["power_thermal"] == pytest.approx(0.8)
    assert verdict.secondary == "decode"


def test_unknown_without_service_metrics() -> None:
    verdict = classify_bottleneck(None)
    assert verdict.label == "unknown" and verdict.confidence == 0.0
    assert any("no worker metrics" in line for line in verdict.evidence)


def test_counter_typed_retractions_count_as_kv_pressure() -> None:
    state = agg_state(
        100,
        5,
        0.93,
        sglang__total_retracted_reqs=(
            "counter",
            [series(AGG, {"total": 12.0, "rate": 0.02})],
        ),
    )
    retracted = state.metric("agg", "retracted_reqs")
    assert retracted is not None and retracted.kind == "counter"
    verdict = classify_bottleneck(state, max_running_requests=256)
    assert verdict.label == "kv_capacity"
    assert any("retracted" in line for line in verdict.evidence)


def test_jsonl_only_family_names_lose_the_total_suffix() -> None:
    def record(t: float, phase: str | None, value: float) -> str:
        line: dict[str, Any] = {
            "endpoint_url": AGG,
            "timestamp_ns": int(t * 1e9),
            "metrics": {
                "sglang:prompt_tokens_total": [{"value": value}],
                "sglang:num_running_reqs": [{"value": 3.0}],
            },
        }
        if phase:
            line["benchmark_phase"] = phase
        return json.dumps(line)

    lines = [
        record(10.0, "warmup", 100.0),
        record(20.0, "profiling", 600.0),
        record(30.0, "profiling", 1100.0),
        record(40.0, None, 9999.0),  # after profiling: ignored
    ]
    state = summarize_service_state(None, lines, ROLES)
    prompt = state.metric("agg", "prompt_tokens")
    assert prompt is not None and prompt.source == "sglang:prompt_tokens"
    # Reference = last warmup scrape: (1100 - 100) / (30 - 10) s.
    assert (prompt.total, prompt.rate_per_s) == (1000.0, 50.0)
    assert state.value("agg", "running_reqs") == 3.0


def _counter(total: float, rate: float, labels: dict[str, str] | None = None) -> dict:
    return {
        "endpoint_url": DECODE0,
        "labels": labels or {},
        "stats": {"total": total, "rate": rate},
    }


def test_round3_counters_derive_cuda_graph_retraction_eviction_and_busy() -> None:
    """New SGLang 0.5.18 families: CUDA graph passes by mode, retracted and
    evicted tokens, opt-in device timer and MFU estimates (one decode rank)."""
    export = {
        "summary": {
            "phase_time_ranges": {
                "profiling": {
                    "start_ns": int(START_S * 1e9),
                    "end_ns": int(END_S * 1e9),
                }
            }
        },
        "metrics": {
            "sglang:cuda_graph_passes": {
                "type": "counter",
                "series": [
                    _counter(900.0, 1.5, {"mode": "decode_cuda_graph"}),
                    _counter(100.0, 0.2, {"mode": "decode_none"}),
                    _counter(40.0, 0.1, {"mode": "prefill_none"}),
                ],
            },
            "sglang:generation_tokens": {
                "type": "counter",
                "series": [_counter(20000.0, 33.3)],
            },
            "sglang:spec_verify_calls": {
                "type": "counter",
                "series": [_counter(8000.0, 13.3)],
            },
            "sglang:prompt_tokens": {
                "type": "counter",
                "series": [_counter(1e6, 1666.7)],
            },
            "sglang:num_retracted_input_tokens": {
                "type": "counter",
                "series": [_counter(1500.0, 2.5)],
            },
            "sglang:num_retracted_output_tokens": {
                "type": "counter",
                "series": [_counter(500.0, 0.8)],
            },
            "sglang:evicted_tokens": {
                "type": "counter",
                "series": [_counter(250000.0, 416.7)],
            },
            "sglang:forward_execution_seconds": {
                "type": "counter",
                "series": [_counter(480.0, 0.8)],
            },
            "sglang:estimated_flops_per_gpu": {
                "type": "counter",
                "series": [_counter(6e17, 1e15)],
            },
            "sglang:estimated_read_bytes_per_gpu": {
                "type": "counter",
                "series": [_counter(1.2e15, 2e12)],
            },
            "sglang:estimated_write_bytes_per_gpu": {
                "type": "counter",
                "series": [_counter(6e13, 1e11)],
            },
        },
    }
    endpoints = [EndpointDescriptor(url=DECODE0, role="decode", node="node2")]
    state = summarize_service_state(export, None, endpoints)
    decode = state.roles["decode"]
    assert decode.metrics["cuda_graph_decode_cuda_graph"].total == 900.0
    assert "cuda_graph_prefill_cuda_graph" not in decode.metrics  # never incremented
    derived = decode.derived
    assert derived["decode_cuda_graph_fraction"] == pytest.approx(0.9)
    assert derived["spec_tokens_per_verify"] == pytest.approx(2.5)
    assert derived["retracted_tokens_per_generated"] == pytest.approx(0.1)
    assert derived["evicted_tokens_per_prompt"] == pytest.approx(0.25)
    assert derived["forward_busy_fraction"] == pytest.approx(0.8)
    assert derived["est_tflops_per_gpu"] == pytest.approx(1000.0)
    assert derived["est_mem_gbs_per_gpu"] == pytest.approx(2100.0)


def test_prefill_bootstrap_queue_reads_the_0_5_18_family_name() -> None:
    spec = next(s for s in CATALOG if s.key == "prefill_bootstrap_queue")
    assert spec.families[0] == "sglang:num_prefill_bootstrap_queue_reqs"
    stages = [s for s in CATALOG if s.key.startswith("stage_")]
    assert len(stages) == 11 and all(s.labels[0][0] == "stage" for s in stages)


def test_bottleneck_uses_cuda_graph_retraction_and_eviction() -> None:
    from benchmarks.cluster_bench.domain.servicestate import RoleState

    state = ServiceState(
        source="json",
        roles={
            "decode": RoleState(
                role="decode",
                endpoints=1,
                derived={
                    "decode_cuda_graph_fraction": 0.6,
                    "retracted_tokens_per_generated": 0.12,
                    "evicted_tokens_per_prompt": 0.9,
                },
            )
        },
    )
    verdict = classify_bottleneck(state)
    assert verdict.scores.get("kv_capacity", 0) >= 0.35
    assert verdict.scores.get("decode", 0) >= 0.2
    joined = " ".join(verdict.evidence)
    assert "CUDA graph" in joined or verdict.label == "kv_capacity"


def test_bottleneck_takes_gpu_regimes_as_evidence() -> None:
    from benchmarks.cluster_bench.domain.gpuactivity import RoleActivity

    activity = {
        "prefill": RoleActivity(role="prefill", regime="compute", reason="tensor pipes active 62% (DRAM 30%)"),
        "decode": RoleActivity(role="decode", regime="memory", reason="DRAM bandwidth 78% used (tensor 21%)"),
    }  # fmt: skip
    verdict = classify_bottleneck(None, activity=activity)
    assert verdict.scores["prefill"] >= 0.25 and verdict.scores["decode"] >= 0.25
    assert any("memory-bandwidth-bound" in e for e in verdict.evidence) or (
        verdict.label == "prefill"
    )
