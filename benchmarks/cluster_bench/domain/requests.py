# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request analysis of one point from AIPerf's record export. Pure.

AIPerf 0.14.0 with ``--export-level records`` writes ``profile_export.jsonl``,
one ``MetricRecordInfo`` per request (verified in the 0.13.0 and 0.14.0 wheels,
``common/models/record_models.py``)::

    {"metadata": {"x_correlation_id", "conversation_id", "turn_index",
                  "root_correlation_id", "agent_depth", "request_start_ns",
                  "request_end_ns", "benchmark_phase", "was_cancelled", ...},
     "metrics": {<tag>: {"value": v, "unit": "ms" | "tokens" | ...}},
     "error": null | {...}}

Record tags used here: ``time_to_first_token``, ``inter_token_latency``,
``request_latency``, ``input_sequence_length``, ``output_sequence_length``
and, when the server reports usage, ``usage_prompt_cache_read_tokens``.

``summarize_requests`` turns the profiling-phase records into what the
aggregate export cannot show: goodput under SLO tiers (AIPerf's definition:
requests meeting every SLO per second; here also their output tokens per
second and per GPU), how TTFT behaves as an agentic session grows (prefix
reuse across turns) and session-level spans. The throughput search does not
use it; it explains results and lets the knowledge base compare clusters by
what users would experience.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from benchmarks.cluster_bench.domain.base import Frozen
from benchmarks.cluster_bench.domain.modelarch import ModelArch
from benchmarks.cluster_bench.domain.servicestate import percentile

# Per-user output tok/s floors; equal to schema.INTERACTIVITY_FLOORS (schema
# imports this module, so the values are repeated; a test keeps them equal).
DEFAULT_FLOORS: tuple[float, ...] = (20.0, 50.0)
# Turn-index buckets of an agentic session (1-based turn numbers).
TURN_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16+", 16, None),
)
_TIME_UNITS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "sec": 1.0}


class SloTier(Frozen):
    """A request is good when TTFT <= ttft_s and mean ITL <= itl_ms."""

    name: str
    ttft_s: float
    itl_ms: float


def default_tiers() -> tuple[SloTier, ...]:
    """One tier per interactivity floor (ITL = 1000 / floor ms).

    TTFT bounds suit long agentic prompts (~100k tokens): 10 s next to the
    stricter floor, 30 s next to the looser one.
    """
    floors = sorted(DEFAULT_FLOORS, reverse=True)
    bounds = (10.0, 30.0)
    return tuple(
        SloTier(name=f"u{floor:g}", ttft_s=bounds[min(i, 1)], itl_ms=1000.0 / floor)
        for i, floor in enumerate(floors)
    )


def parse_tier(text: str) -> SloTier:
    """``NAME:TTFT_S:ITL_MS`` (CLI form), e.g. ``u50:10:20``."""
    parts = text.split(":")
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"SLO tier must be NAME:TTFT_S:ITL_MS, got {text!r}")
    ttft, itl = float(parts[1]), float(parts[2])
    if ttft <= 0 or itl <= 0:
        raise ValueError(f"SLO tier bounds must be positive: {text!r}")
    return SloTier(name=parts[0], ttft_s=ttft, itl_ms=itl)


@dataclass(frozen=True)
class Record:
    """One finished profiling-phase request (internal, compact)."""

    start_ns: int
    end_ns: int
    ok: bool
    session: str | None
    turn: int | None  # 1-based
    depth: int
    ttft_s: float | None
    itl_ms: float | None
    latency_s: float | None
    isl: float | None
    osl: float | None
    cache_read: float | None


class GoodputTier(Frozen):
    name: str
    ttft_s: float
    itl_ms: float
    good_requests: int
    good_fraction: float | None = None
    goodput_rps: float | None = None  # good requests per second
    good_output_tps: float | None = None  # their output tokens per second
    good_output_tps_per_gpu: float | None = None


class TurnBucket(Frozen):
    bucket: str
    requests: int
    ttft_p50_s: float | None = None
    ttft_p90_s: float | None = None
    isl_p50: float | None = None
    ttft_s_per_100k_isl: float | None = None  # median TTFT per 100k input tokens
    cache_read_fraction: float | None = None  # usage-reported prompt cache reads


class SessionStats(Frozen):
    sessions: int
    subagent_requests: int = 0  # agent_depth > 0
    turns_p50: float | None = None
    turns_max: int | None = None
    span_p50_s: float | None = None  # first request start -> last request end
    span_p90_s: float | None = None
    busy_fraction_p50: float | None = None  # sum(request latency) / span


class RequestSummary(Frozen):
    """What the requests of one point experienced (``summarize_requests``)."""

    records: int = 0  # profiling-phase records read
    finished: int = 0  # without error
    errors: int = 0
    cancelled: int = 0
    window_s: float | None = None
    tiers: tuple[GoodputTier, ...] = ()
    turns: tuple[TurnBucket, ...] = ()
    sessions: SessionStats | None = None
    first_turn_ttft_p50_s: float | None = None
    late_turn_ttft_p50_s: float | None = None  # turns >= 8
    # Workload of the finished requests (MFU and KV-capacity inputs).
    prompt_tokens: float | None = None  # sum of ISL
    cached_tokens: float | None = None  # prompt tokens served from the prefix cache
    output_tokens: float | None = None  # sum of OSL
    cache_source: str = "none"  # "usage" (per request) | "server" (fraction) | "none"
    context_p50: float | None = None  # ISL + OSL of one request
    context_p90: float | None = None
    context_max: float | None = None
    flops_prefill: float | None = None  # model FLOPs of the uncached prompt tokens
    flops_decode: float | None = None  # model FLOPs of the generated tokens
    notes: tuple[str, ...] = ()

    def tier(self, name: str) -> GoodputTier | None:
        return next((t for t in self.tiers if t.name == name), None)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _number(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _metric(metrics: Mapping[str, object], tag: str) -> tuple[float | None, str]:
    entry = metrics.get(tag)
    if isinstance(entry, Mapping):
        return _number(entry.get("value")), str(entry.get("unit") or "")
    return _number(entry), ""


def _seconds(metrics: Mapping[str, object], tag: str) -> float | None:
    value, unit = _metric(metrics, tag)
    if value is None:
        return None
    return value * _TIME_UNITS.get(unit.lower(), 1e-3)  # AIPerf's display unit: ms


def _session_key(meta: Mapping[str, object]) -> str | None:
    for key in ("root_correlation_id", "x_correlation_id", "conversation_id"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def parse_record(obj: object) -> Record | None:
    """A profiling-phase record, or None (warmup, malformed, cancelled)."""
    if not isinstance(obj, Mapping):
        return None
    meta = obj.get("metadata")
    metrics = obj.get("metrics") or {}
    if not isinstance(meta, Mapping) or not isinstance(metrics, Mapping):
        return None
    if str(meta.get("benchmark_phase", "profiling")).lower() != "profiling":
        return None
    start, end = meta.get("request_start_ns"), meta.get("request_end_ns")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        return None
    turn_raw = meta.get("turn_index")
    turn = turn_raw + 1 if isinstance(turn_raw, int) and turn_raw >= 0 else None
    depth_raw = meta.get("agent_depth")
    itl_s = _seconds(metrics, "inter_token_latency")
    return Record(
        start_ns=start,
        end_ns=end,
        ok=obj.get("error") is None and not bool(meta.get("was_cancelled")),
        session=_session_key(meta),
        turn=turn,
        depth=depth_raw if isinstance(depth_raw, int) else 0,
        ttft_s=_seconds(metrics, "time_to_first_token"),
        itl_ms=itl_s * 1e3 if itl_s is not None else None,
        latency_s=_seconds(metrics, "request_latency"),
        isl=_metric(metrics, "input_sequence_length")[0],
        osl=_metric(metrics, "output_sequence_length")[0],
        cache_read=_metric(metrics, "usage_prompt_cache_read_tokens")[0],
    )


def parse_records(lines: Iterable[str]) -> tuple[list[Record], int, int]:
    """(records, cancelled, unreadable lines) from ``profile_export.jsonl``."""
    records: list[Record] = []
    cancelled = bad = 0
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            bad += 1  # e.g. the torn last line of an interrupted run
            continue
        meta = obj.get("metadata") if isinstance(obj, Mapping) else None
        if isinstance(meta, Mapping) and meta.get("was_cancelled"):
            if str(meta.get("benchmark_phase", "profiling")).lower() == "profiling":
                cancelled += 1
            continue
        record = parse_record(obj)
        if record is not None:
            records.append(record)
    return records, cancelled, bad


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _good(record: Record, tier: SloTier) -> bool:
    return (
        record.ok
        and record.ttft_s is not None
        and record.ttft_s <= tier.ttft_s
        and (record.itl_ms is None or record.itl_ms <= tier.itl_ms)
    )


def goodput_tiers(
    records: Sequence[Record],
    tiers: Sequence[SloTier],
    window_s: float | None,
    gpus: int,
) -> tuple[GoodputTier, ...]:
    out = []
    for tier in tiers:
        good = [r for r in records if _good(r, tier)]
        tokens = sum(r.osl or 0.0 for r in good)
        rate = (len(good) / window_s) if window_s else None
        tps = (tokens / window_s) if window_s else None
        out.append(
            GoodputTier(
                name=tier.name,
                ttft_s=tier.ttft_s,
                itl_ms=tier.itl_ms,
                good_requests=len(good),
                good_fraction=_round(len(good) / len(records)) if records else None,
                goodput_rps=_round(rate),
                good_output_tps=_round(tps, 2),
                good_output_tps_per_gpu=_round(tps / gpus, 3)
                if tps is not None and gpus
                else None,
            )
        )
    return tuple(out)


def _bucket_of(turn: int) -> str:
    for name, low, high in TURN_BUCKETS:
        if turn >= low and (high is None or turn <= high):
            return name
    return TURN_BUCKETS[-1][0]


def turn_profile(records: Sequence[Record]) -> tuple[TurnBucket, ...]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        if record.ok and record.turn is not None:
            grouped.setdefault(_bucket_of(record.turn), []).append(record)
    out = []
    for name, _, _ in TURN_BUCKETS:
        members = grouped.get(name)
        if not members:
            continue
        ttfts = [r.ttft_s for r in members if r.ttft_s is not None]
        isls = [r.isl for r in members if r.isl]
        per_100k = [
            r.ttft_s / r.isl * 1e5 for r in members if r.ttft_s is not None and r.isl
        ]
        reads = [(r.cache_read, r.isl) for r in members if r.cache_read is not None]
        prompt = sum(isl or 0.0 for _, isl in reads)
        out.append(
            TurnBucket(
                bucket=name,
                requests=len(members),
                ttft_p50_s=_round(percentile(ttfts, 50)),
                ttft_p90_s=_round(percentile(ttfts, 90)),
                isl_p50=_round(percentile(isls, 50), 1),
                ttft_s_per_100k_isl=_round(percentile(per_100k, 50)),
                cache_read_fraction=_round(sum(c for c, _ in reads) / prompt)
                if reads and prompt
                else None,
            )
        )
    return tuple(out)


def session_stats(records: Sequence[Record]) -> SessionStats | None:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        if record.session is not None:
            grouped.setdefault(record.session, []).append(record)
    if not grouped:
        return None
    turns, spans, busy = [], [], []
    for members in grouped.values():
        turns.append(len(members))
        start = min(r.start_ns for r in members)
        end = max(r.end_ns for r in members)
        span = (end - start) / 1e9
        spans.append(span)
        if span > 0:
            used = sum(r.latency_s or 0.0 for r in members)
            busy.append(min(1.0, used / span))
    return SessionStats(
        sessions=len(grouped),
        subagent_requests=sum(1 for r in records if r.depth > 0),
        turns_p50=_round(percentile(turns, 50), 1),
        turns_max=max(turns),
        span_p50_s=_round(percentile(spans, 50), 2),
        span_p90_s=_round(percentile(spans, 90), 2),
        busy_fraction_p50=_round(percentile(busy, 50)),
    )


def summarize_requests(
    records: Sequence[Record],
    *,
    gpus: int,
    tiers: Sequence[SloTier] | None = None,
    window: tuple[float, float] | None = None,
    cancelled: int = 0,
    unreadable: int = 0,
    arch: ModelArch | None = None,
    cached_fraction: float | None = None,
) -> RequestSummary:
    """Goodput tiers, turn profile, sessions and workload of one point's records.

    ``window`` (unix seconds) is the measurement window; without it the span
    from the first request start to the last request end is used. With
    ``arch`` the model FLOPs of the uncached prompt tokens and of the
    generated tokens are summed; cached prompt tokens come from the
    per-request usage, else from the server-side ``cached_fraction``.
    """
    notes: list[str] = []
    if unreadable:
        notes.append(f"{unreadable} unreadable record line(s) skipped")
    if not records:
        return RequestSummary(cancelled=cancelled, notes=tuple(notes))
    if window is not None and window[1] > window[0]:
        window_s: float | None = window[1] - window[0]
    else:
        span = (max(r.end_ns for r in records) - min(r.start_ns for r in records)) / 1e9
        window_s = span if span > 0 else None
    finished = [r for r in records if r.ok]
    if not any(r.ttft_s is not None for r in finished):
        notes.append("records carry no time_to_first_token: goodput not computed")
    profile = turn_profile(records)
    first = next((b for b in profile if b.bucket == "1"), None)
    late = [
        r.ttft_s for r in finished if r.turn and r.turn >= 8 and r.ttft_s is not None
    ]
    return RequestSummary(
        records=len(records),
        finished=len(finished),
        errors=len(records) - len(finished),
        cancelled=cancelled,
        window_s=_round(window_s, 2),
        tiers=goodput_tiers(records, tiers or default_tiers(), window_s, gpus)
        if any(r.ttft_s is not None for r in finished)
        else (),
        turns=profile,
        sessions=session_stats(records),
        first_turn_ttft_p50_s=first.ttft_p50_s if first else None,
        late_turn_ttft_p50_s=_round(percentile(late, 50)),
        notes=tuple(notes),
        **workload(finished, arch, cached_fraction),
    )


def workload(
    finished: Sequence[Record],
    arch: ModelArch | None,
    cached_fraction: float | None,
) -> dict[str, object]:
    """Token, context and model-FLOPs totals of the finished requests."""
    sized = [r for r in finished if r.isl is not None and r.osl is not None]
    if not sized:
        return {}
    usage = any(r.cache_read is not None for r in sized)
    source = "usage" if usage else ("server" if cached_fraction is not None else "none")
    prompt = cached = output = 0.0
    f_prefill = f_decode = 0.0
    contexts: list[float] = []
    for r in sized:
        isl, osl = float(r.isl or 0.0), float(r.osl or 0.0)
        if usage:
            hit = min(float(r.cache_read or 0.0), isl)
        elif cached_fraction is not None:
            hit = isl * min(max(cached_fraction, 0.0), 1.0)
        else:
            hit = 0.0
        prompt, cached, output = prompt + isl, cached + hit, output + osl
        contexts.append(isl + osl)
        if arch is not None:
            f_prefill += arch.tokens_flops(hit, isl - hit)
            f_decode += arch.tokens_flops(isl, osl)
    out: dict[str, object] = {
        "prompt_tokens": prompt,
        "cached_tokens": round(cached, 1),
        "output_tokens": output,
        "cache_source": source,
        "context_p50": _round(percentile(contexts, 50)),
        "context_p90": _round(percentile(contexts, 90)),
        "context_max": max(contexts),
    }
    if arch is not None:
        out["flops_prefill"] = f_prefill
        out["flops_decode"] = f_decode
    return out


def synthetic_records(
    *,
    requests: int,
    start_s: float,
    duration_s: float,
    ttft_p50_s: float,
    itl_ms: float,
    osl: float,
    isl: float,
    sessions: int,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Deterministic AIPerf-style record dicts for the dry-run simulator."""
    out: list[dict[str, object]] = []
    sessions = max(1, sessions)
    for i in range(max(0, requests)):
        session = i % sessions
        turn = i // sessions
        wobble = 0.6 + 0.8 * ((i * 7919 + seed * 104729) % 1000) / 1000.0
        ttft = ttft_p50_s * wobble * (1.6 if turn == 0 else 0.85)
        itl = itl_ms * (0.8 + 0.4 * ((i * 31 + seed) % 100) / 100.0)
        latency = ttft + itl * osl / 1e3
        begin = start_s + duration_s * (i / max(1, requests))
        out.append(
            {
                "metadata": {
                    "session_num": session,
                    "x_correlation_id": f"sim-{seed}-{session}",
                    "conversation_id": f"conv-{session}",
                    "turn_index": turn,
                    "agent_depth": 1 if i % 9 == 8 else 0,
                    "request_start_ns": int(begin * 1e9),
                    "request_end_ns": int((begin + latency) * 1e9),
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                },
                "metrics": {
                    "time_to_first_token": {
                        "value": round(ttft * 1e3, 3),
                        "unit": "ms",
                    },
                    "inter_token_latency": {"value": round(itl, 3), "unit": "ms"},
                    "request_latency": {"value": round(latency * 1e3, 3), "unit": "ms"},
                    "input_sequence_length": {
                        "value": round(isl * (0.5 + 0.1 * min(turn, 10))),
                        "unit": "tokens",
                    },
                    "output_sequence_length": {"value": round(osl), "unit": "tokens"},
                    # later turns reuse the previous turn's prompt from the prefix cache
                    "usage_prompt_cache_read_tokens": {
                        "value": round(isl * (0.5 + 0.1 * min(turn - 1, 10)))
                        if turn > 0
                        else 0,
                        "unit": "tokens",
                    },
                },
                "error": None,
            }
        )
    return out


__all__ = [
    "GoodputTier",
    "Record",
    "RequestSummary",
    "SessionStats",
    "SloTier",
    "TurnBucket",
    "default_tiers",
    "goodput_tiers",
    "parse_record",
    "parse_records",
    "parse_tier",
    "session_stats",
    "summarize_requests",
    "synthetic_records",
    "turn_profile",
]
