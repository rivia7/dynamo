# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request analysis: goodput tiers, turn profile, sessions, accept length."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.cluster_bench.adapters.aiperf import load_request_summary
from benchmarks.cluster_bench.app.measure import accept_length
from benchmarks.cluster_bench.domain.pointstate import PointState, RoleBrief
from benchmarks.cluster_bench.domain.requests import (
    DEFAULT_FLOORS,
    SloTier,
    default_tiers,
    parse_record,
    parse_records,
    parse_tier,
    summarize_requests,
    synthetic_records,
)
from benchmarks.cluster_bench.domain.schema import INTERACTIVITY_FLOORS

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

T0 = 1_790_000_000.0


def _record(
    i: int,
    *,
    ttft_ms: float,
    itl_ms: float,
    osl: int = 500,
    isl: int = 90000,
    turn: int = 0,
    session: str = "s0",
    phase: str = "profiling",
    error: object = None,
    cancelled: bool = False,
    cache_read: int | None = None,
    depth: int = 0,
) -> dict:
    start = T0 + i
    latency_ms = ttft_ms + itl_ms * osl
    metrics = {
        "time_to_first_token": {"value": ttft_ms, "unit": "ms"},
        "inter_token_latency": {"value": itl_ms, "unit": "ms"},
        "request_latency": {"value": latency_ms, "unit": "ms"},
        "input_sequence_length": {"value": isl, "unit": "tokens"},
        "output_sequence_length": {"value": osl, "unit": "tokens"},
    }
    if cache_read is not None:
        metrics["usage_prompt_cache_read_tokens"] = {
            "value": cache_read,
            "unit": "tokens",
        }
    return {
        "metadata": {
            "session_num": 0,
            "x_correlation_id": session,
            "conversation_id": f"conv-{session}",
            "turn_index": turn,
            "agent_depth": depth,
            "request_start_ns": int(start * 1e9),
            "request_end_ns": int((start + latency_ms / 1e3) * 1e9),
            "benchmark_phase": phase,
            "was_cancelled": cancelled,
        },
        "metrics": metrics,
        "error": error,
    }


def test_default_tiers_follow_the_interactivity_floors() -> None:
    assert DEFAULT_FLOORS == INTERACTIVITY_FLOORS
    tiers = {t.name: t for t in default_tiers()}
    assert set(tiers) == {"u50", "u20"}
    assert tiers["u50"].itl_ms == pytest.approx(20.0) and tiers["u50"].ttft_s == 10.0
    assert tiers["u20"].itl_ms == pytest.approx(50.0) and tiers["u20"].ttft_s == 30.0


def test_parse_tier_accepts_the_cli_form_and_rejects_garbage() -> None:
    assert parse_tier("fast:2:15") == SloTier(name="fast", ttft_s=2.0, itl_ms=15.0)
    for bad in ("fast:2", ":1:2", "x:0:5", "x:a:5"):
        with pytest.raises(ValueError):
            parse_tier(bad)


def test_parse_record_converts_units_and_keys() -> None:
    raw = _record(0, ttft_ms=1500.0, itl_ms=12.5, turn=2, session="root-1", depth=1)
    raw["metadata"]["root_correlation_id"] = "tree-9"
    rec = parse_record(raw)
    assert rec is not None and rec.ok
    assert rec.ttft_s == pytest.approx(1.5) and rec.itl_ms == pytest.approx(12.5)
    assert rec.turn == 3  # 1-based
    assert rec.session == "tree-9"  # the agentic session tree wins
    assert rec.depth == 1
    assert parse_record(_record(1, ttft_ms=1, itl_ms=1, phase="warmup")) is None
    assert parse_record({"metadata": {}, "metrics": {}}) is None
    failed = parse_record(_record(2, ttft_ms=1, itl_ms=1, error={"code": 500}))
    assert failed is not None and not failed.ok


def test_parse_records_counts_cancelled_and_torn_lines() -> None:
    lines = [
        json.dumps(_record(0, ttft_ms=900, itl_ms=10)),
        json.dumps(_record(1, ttft_ms=900, itl_ms=10, cancelled=True)),
        json.dumps(_record(2, ttft_ms=900, itl_ms=10, phase="warmup", cancelled=True)),
        "",
        '{"metadata": {"request_start_ns": 1',  # torn final line
    ]
    records, cancelled, bad = parse_records(lines)
    assert len(records) == 1 and cancelled == 1 and bad == 1


def test_goodput_tiers_apply_both_bounds() -> None:
    raws = [
        _record(0, ttft_ms=2000, itl_ms=15),  # good for u50 and u20
        _record(1, ttft_ms=2000, itl_ms=30),  # u20 only (ITL)
        _record(2, ttft_ms=20000, itl_ms=15),  # u20 only (TTFT 20 s > 10 s)
        _record(3, ttft_ms=40000, itl_ms=15),  # neither (TTFT)
        _record(4, ttft_ms=1000, itl_ms=10, error={"code": 500}),  # errors never count
    ]
    records = [parse_record(r) for r in raws]
    summary = summarize_requests(
        [r for r in records if r is not None], gpus=8, window=(T0, T0 + 100.0)
    )
    u50, u20 = summary.tier("u50"), summary.tier("u20")
    assert u50 is not None and u20 is not None
    assert (u50.good_requests, u20.good_requests) == (1, 3)
    assert u20.good_fraction == pytest.approx(0.6)
    assert u20.goodput_rps == pytest.approx(0.03)
    assert u20.good_output_tps == pytest.approx(15.0)  # 3 x 500 tokens / 100 s
    assert u20.good_output_tps_per_gpu == pytest.approx(15.0 / 8, abs=1e-3)
    assert summary.errors == 1 and summary.finished == 4


def test_goodput_divides_by_the_aiperf_throughput_duration() -> None:
    # AIPerf divides its throughput by the profiling phase, grace period
    # included: here the last request ended 90 s into a 120 s phase.
    raws = [_record(i, ttft_ms=1000, itl_ms=12, osl=600) for i in range(10)]
    records = [r for r in (parse_record(x) for x in raws) if r is not None]
    summary = summarize_requests(
        records, gpus=8, window=(T0, T0 + 90.0), duration_s=120.0
    )
    u20 = summary.tier("u20")
    assert summary.window_s == 120.0
    assert u20 is not None and u20.good_fraction == 1.0
    # every request good: goodput is AIPerf's own throughput, 6000 tokens / 120 s
    assert u20.good_output_tps == pytest.approx(50.0)
    assert u20.goodput_rps == pytest.approx(10 / 120.0, abs=1e-4)
    # without that duration: the request window, as before
    fallback = summarize_requests(records, gpus=8, window=(T0, T0 + 90.0))
    assert fallback.window_s == 90.0


def test_turn_profile_and_sessions() -> None:
    raws = []
    for s in range(3):
        for turn in range(10):
            raws.append(
                _record(
                    s * 10 + turn,
                    ttft_ms=4000 if turn == 0 else 1200,
                    itl_ms=12,
                    isl=20000 + 8000 * turn,
                    turn=turn,
                    session=f"s{s}",
                    cache_read=0 if turn == 0 else 18000 + 8000 * turn,
                    depth=1 if turn == 9 else 0,
                )
            )
    records = [r for r in (parse_record(x) for x in raws) if r is not None]
    summary = summarize_requests(records, gpus=16)
    buckets = {b.bucket: b for b in summary.turns}
    assert [b.bucket for b in summary.turns] == ["1", "2-3", "4-7", "8-15"]
    assert buckets["1"].requests == 3 and buckets["1"].ttft_p50_s == pytest.approx(4.0)
    assert buckets["8-15"].ttft_p50_s == pytest.approx(1.2)
    assert buckets["1"].cache_read_fraction == 0.0
    assert 0.8 < (buckets["8-15"].cache_read_fraction or 0.0) < 1.0
    assert summary.first_turn_ttft_p50_s == pytest.approx(4.0)
    assert summary.late_turn_ttft_p50_s == pytest.approx(1.2)
    sessions = summary.sessions
    assert sessions is not None and sessions.sessions == 3
    assert sessions.turns_p50 == 10 and sessions.turns_max == 10
    assert sessions.subagent_requests == 3
    assert sessions.span_p50_s is not None and sessions.span_p50_s > 9
    assert sessions.busy_fraction_p50 is not None and sessions.busy_fraction_p50 <= 1.0


def test_records_without_ttft_explain_why_goodput_is_missing() -> None:
    raw = _record(0, ttft_ms=1, itl_ms=1)
    del raw["metrics"]["time_to_first_token"]
    summary = summarize_requests([parse_record(raw)], gpus=1)  # type: ignore[list-item]
    assert summary.tiers == ()
    assert any("time_to_first_token" in n for n in summary.notes)
    assert summarize_requests([], gpus=1).records == 0


def test_synthetic_records_round_trip_through_the_analysis() -> None:
    raw = synthetic_records(
        requests=200,
        start_s=T0,
        duration_s=100.0,
        ttft_p50_s=1.5,
        itl_ms=15.0,
        osl=600,
        isl=90000,
        sessions=20,
    )
    records = [r for r in (parse_record(x) for x in raw) if r is not None]
    summary = summarize_requests(records, gpus=8, window=(T0, T0 + 100.0))
    assert summary.records == 200 and summary.sessions is not None
    assert summary.sessions.sessions == 20
    assert summary.tier("u20") is not None


def test_load_request_summary_reads_the_newest_records_file(tmp_path: Path) -> None:
    assert load_request_summary(tmp_path, gpus=8) is None
    run = tmp_path / "aiperf" / "run1"
    run.mkdir(parents=True)
    lines = [json.dumps(_record(i, ttft_ms=1000, itl_ms=12)) for i in range(5)]
    (run / "profile_export.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = load_request_summary(tmp_path / "aiperf", gpus=8)
    assert summary is not None and summary.records == 5
    assert summary.tier("u50") is not None and summary.tier("u50").good_requests == 5
    timed = load_request_summary(tmp_path / "aiperf", gpus=8, duration_s=50.0)
    assert timed is not None and timed.window_s == 50.0


def test_accept_length_prefers_the_request_weighted_counter_ratio(
    tmp_path: Path,
) -> None:
    export = {
        "metrics": {"sglang:spec_accept_length": {"series": [{"stats": {"avg": 3.4}}]}}
    }
    (tmp_path / "server_metrics_export.json").write_text(
        json.dumps(export), encoding="utf-8"
    )
    weighted = PointState(
        roles={
            "decode": RoleBrief(endpoints=1, derived={"spec_tokens_per_verify": 2.7})
        }
    )
    assert accept_length(weighted, tmp_path) == pytest.approx(2.7)
    # no counters: the gauge mean is the fallback
    assert accept_length(PointState(), tmp_path) == pytest.approx(3.4)
    assert accept_length(None, tmp_path / "missing") is None
