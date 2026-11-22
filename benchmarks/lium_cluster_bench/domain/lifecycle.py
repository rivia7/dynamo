# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment lifecycle from container logs: cold-start cost, KV capacity. Pure.

``parse_container_log`` reads the ``docker logs`` text of one container
(``dynamo.sglang`` worker, Dynamo frontend, or the etcd/NATS control
container) into a ``ContainerLifecycle``: when weights finished loading, the
KV pool was allocated, CUDA graphs were captured and the container became
ready, with SGLang's memory and capacity numbers and a census of warnings
and errors. ``deploy_lifecycle`` summarizes a deployment per role; the
slowest container of a role sets its time-to-ready.

Log formats (all tolerated, per line)
-------------------------------------
* ``docker logs --timestamps`` prefix ``2026-11-05T10:02:30.123456789Z ``
  (preferred when present: UTC, ns).
* Dynamo/Rust tracing (SGLang's loggers are bridged into it under
  ``dynamo.sglang``): ``2026-11-05T10:02:30.123456Z  INFO target: msg``
  (``lib/runtime/src/logging.rs``; local ``+hh:mm`` offsets accepted).
* SGLang's own format: ``[2026-11-05 10:02:30 TP0] msg`` (rank tags such as
  ``DP0 TP3 EP3`` optional). Naive timestamps are read as UTC (containers
  default to TZ=UTC; ``ContainerLifecycle.naive_utc`` flags it).
* JSON lines (``DYN_LOGGING_JSONL``) with ``time``/``timestamp`` and
  ``message``/``msg``.
Lines without a timestamp (tqdm bars, NCCL, tracebacks) inherit the previous
line's time. Per-rank lines repeat once per TP rank: "begin" events take the
first occurrence, "end" events the last (the slowest rank), memory numbers
the tightest GPU (minimum available, maximum used).

SGLang 0.5.x message strings matched (UNVERIFIED exact wording; regexes are
anchored on the leading phrase and read each ``key=value`` independently):
``Load weight begin. avail mem=X GB``; ``Load weight end. [elapsed=S s, ]
type=T, dtype=D, avail mem=X GB, mem usage=Y GB.``; ``KV Cache is allocated.
#tokens: N, K size: A GB, V size: B GB`` (MHA) or ``..., KV size: C GB``
(MLA); ``Memory pool end. avail mem=X GB``; ``Capture cuda graph begin. ...
avail mem=X GB``; ``Capture cuda graph end. Time elapsed: S s. mem usage=M
GB. avail mem=A GB.``; ``max_total_num_tokens=N, chunked_prefill_size=C,
max_prefill_tokens=P, max_running_requests=R, context_len=L,
available_gpu_mem=X GB``. Readiness: ``READY_PATTERNS`` (Dynamo strings
VERIFIED in ``components/src/dynamo/sglang/register.py`` and
``lib/llm/src/discovery/watcher.rs`` of this repo; others UNVERIFIED). A
non-leader rank never registers: its readiness is its last scheduler line.
"""

from __future__ import annotations

import calendar
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from benchmarks.lium_cluster_bench.domain.base import Frozen

MAX_ISSUE_LINES = 3
MAX_LINE_CHARS = 300

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_TS = (
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?"
    r"(Z|[+-]\d{2}:?\d{2})?"
)
_DOCKER_TS_RE = re.compile(r"^" + _TS + r"\s")
_DYNAMO_RE = re.compile(
    r"^" + _TS + r"\s+(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\s+"
    r"(?:[\w.:<>\-]+:\s+)?"
)
_SGLANG_RE = re.compile(
    r"^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?"
    r"((?:\s+[A-Z]{1,4}\d+)*)\]\s?"
)
_PLAIN_RE = re.compile(r"^" + _TS + r"(?:\s+-)?\s+")

EVENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("server_args", re.compile(r"ServerArgs\(")),
    ("dist_init_begin", re.compile(r"Init torch distributed begin")),
    ("dist_init_end", re.compile(r"Init torch distributed ends?\b")),
    ("weights_begin", re.compile(r"Load weight begin")),
    ("weights_end", re.compile(r"Load weight end")),
    ("kv_allocated", re.compile(r"KV Cache is allocated")),
    ("memory_pool_end", re.compile(r"Memory pool end")),
    ("cuda_graph_begin", re.compile(r"Capture (?:[\w-]+ )*?(?:cuda|CUDA) graph begin")),
    ("cuda_graph_end", re.compile(r"Capture (?:[\w-]+ )*?(?:cuda|CUDA) graph end")),
    ("scheduler_ready", re.compile(r"max_total_num_tokens\s*=\s*\d+")),
    ("http_started", re.compile(r"Starting HTTP(?:\(S\))? service|Uvicorn running on")),
)
READY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Successfully registered LLM"),  # dynamo.sglang (VERIFIED)
    re.compile(r"Model registration succeeded"),  # dynamo.sglang (VERIFIED)
    re.compile(r"\b(?:Chat completions|Completions) is ready"),  # frontend (VERIFIED)
    re.compile(r"The server is fired up and ready to roll"),  # SGLang HTTP server
    re.compile(r"ready to serve client requests"),  # etcd
    re.compile(r"Server is ready"),  # NATS
)
ISSUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cuda_oom",
        re.compile(
            r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|"
            r"cudaErrorMemoryAllocation",
            re.IGNORECASE,
        ),
    ),
    (
        "nccl_error",
        re.compile(
            r"NCCL error|ncclSystemError|ncclInternalError|ncclUnhandledCudaError|"
            r"ncclRemoteError|ncclInvalidUsage"
        ),
    ),
    ("nccl_warn", re.compile(r"NCCL WARN")),
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    (
        "kv_transfer",
        re.compile(
            r"(?:mooncake|transfer[ _]?engine|nixl|kv[ _]?transfer|KVTransferError|"
            r"bootstrap)\S*.*?(?:error|fail|timed? ?out)",
            re.IGNORECASE,
        ),
    ),
    (
        "rdma",
        re.compile(
            r"(?:\bibv_\w+|IBV_WC_\w+|\brdma_\w+|\bmlx5_\d+\b).*?(?:error|fail)",
            re.IGNORECASE,
        ),
    ),
    ("xid", re.compile(r"\bXid\b|NVRM:")),
    (
        "cuda_error",
        re.compile(
            r"CUDA error|illegal memory access|cudaError\w*|uncorrectable ECC",
            re.IGNORECASE,
        ),
    ),
)
_PY_WARNING_RE = re.compile(
    r"\b(?:User|Future|Deprecation|Runtime|Resource)Warning\b|^WARNING\b"
)
_SHARDS_RE = re.compile(
    r"Loading \w+ checkpoint shards:\s*100%.*?\|\s*(\d+)/\1\s*\[(\d+):(\d+)(?::(\d+))?<"
)
_GB = r"\s*=?\s*([0-9]+(?:\.[0-9]+)?)\s*GB"


def _float_field(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _int_field(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _str_field(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _epoch(groups: Sequence[str | None]) -> tuple[float, bool]:
    """(unix seconds, naive) from _TS-style groups (y, m, d, H, M, S, frac, tz)."""
    year, month, day, hour, minute, second = (int(g or 0) for g in groups[:6])
    frac = groups[6] if len(groups) > 6 else None
    tz = groups[7] if len(groups) > 7 else None
    value = float(calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0)))
    if frac:
        value += int(frac) / 10 ** len(frac)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        value -= sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
    return value, not tz


@dataclass(frozen=True)
class LogLine:
    """One log line with its parsed timestamp, level and rank tags."""

    number: int
    t: float | None  # unix seconds (inherited when the line has none)
    level: str | None
    ranks: str | None  # e.g. "TP0" or "DP0 TP3 EP3"
    message: str
    naive: bool = False  # timestamp had no zone and was read as UTC


def _json_line(text: str) -> tuple[float | None, str | None, str] | None:
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, Mapping):
        return None
    message = record.get("message", record.get("msg"))
    if not isinstance(message, str):
        return None
    stamp = record.get("time", record.get("timestamp", record.get("ts")))
    t = None
    if isinstance(stamp, str):
        match = re.match(_TS, stamp)
        if match:
            t = _epoch(match.groups())[0]
    elif isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
        t = float(stamp)  # zap-style epoch seconds
    level = record.get("level")
    return t, str(level).upper() if level else None, message


@dataclass(frozen=True)
class _Parsed:
    t: float | None = None
    naive: bool = False
    level: str | None = None
    ranks: str | None = None
    message: str = ""


def _parse_body(line: str) -> _Parsed:
    """One line without a docker prefix: Dynamo, SGLang, JSON or plain."""
    if line.lstrip().startswith("{"):
        parsed = _json_line(line.strip())
        if parsed is not None:
            return _Parsed(t=parsed[0], level=parsed[1], message=parsed[2])
    dynamo = _DYNAMO_RE.match(line)
    if dynamo is not None:
        t, naive = _epoch(dynamo.groups()[:8])
        return _Parsed(t, naive, dynamo.group(9), None, line[dynamo.end() :])
    sglang = _SGLANG_RE.match(line)
    if sglang is not None:
        t, naive = _epoch(sglang.groups()[:7])
        ranks = sglang.group(8).strip() or None
        return _Parsed(t, naive, None, ranks, line[sglang.end() :])
    plain = _PLAIN_RE.match(line)
    if plain is not None:
        t, naive = _epoch(plain.groups()[:8])
        return _Parsed(t, naive, None, None, line[plain.end() :])
    return _Parsed(message=line)


def parse_log_lines(text: str) -> list[LogLine]:
    """Split container log text into ``LogLine`` records (module docstring)."""
    out: list[LogLine] = []
    current: float | None = None
    for number, raw in enumerate((text or "").splitlines(), start=1):
        line = _ANSI_RE.sub("", raw).rstrip()
        if not line.strip():
            continue
        body = _parse_body(line)
        docker = _DOCKER_TS_RE.match(line)
        if body.level is None and docker is not None and docker.group(8):
            # A `docker logs --timestamps` prefix (always zoned): its time
            # wins; the rest of the line is parsed on its own.
            t, naive = _epoch(docker.groups())
            inner = _parse_body(line[docker.end() :])
            body = _Parsed(t, naive, inner.level, inner.ranks, inner.message)
        if body.t is not None:
            current = body.t
        out.append(
            LogLine(
                number=number,
                t=current,
                level=body.level.upper() if body.level else None,
                ranks=body.ranks,
                message=body.message.strip(),
                naive=body.naive,
            )
        )
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class IssueSummary(Frozen):
    count: int = 0
    first: tuple[str, ...] = ()  # first MAX_ISSUE_LINES lines, truncated


class ContainerLifecycle(Frozen):
    """Startup timeline and numbers of one container (module docstring)."""

    name: str
    role: str
    node: str = ""
    lines: int = 0
    naive_utc: bool = False
    # Event times, unix seconds.
    first_log_t: float | None = None
    last_log_t: float | None = None
    weights_begin_t: float | None = None
    weights_loaded_t: float | None = None
    kv_pool_t: float | None = None
    cuda_graph_t: float | None = None
    scheduler_ready_t: float | None = None
    ready_t: float | None = None
    ready_source: Literal["marker", "scheduler", "none"] = "none"
    ready_line: str | None = None
    # Seconds from the first log line.
    to_weights_loaded_s: float | None = None
    to_kv_pool_s: float | None = None
    to_cuda_graph_s: float | None = None
    to_ready_s: float | None = None
    # Seconds from the deployment start (``started_at``), when given.
    deploy_to_ready_s: float | None = None
    # Phase durations.
    weights_load_s: float | None = None
    kv_pool_s: float | None = None
    cuda_graph_capture_s: float | None = None
    post_graph_to_ready_s: float | None = None
    checkpoint_shards: int | None = None
    shard_load_s: float | None = None
    # Memory and capacity (per GPU / per scheduler).
    model_type: str | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    avail_mem_before_load_gb: float | None = None
    weight_mem_gb: float | None = None
    avail_mem_after_load_gb: float | None = None
    kv_tokens: int | None = None
    kv_cache_gb: float | None = None
    avail_mem_after_pool_gb: float | None = None
    cuda_graph_mem_gb: float | None = None
    avail_mem_after_cuda_graph_gb: float | None = None
    max_total_num_tokens: int | None = None
    chunked_prefill_size: int | None = None
    max_prefill_tokens: int | None = None
    max_running_requests: int | None = None
    context_len: int | None = None
    available_gpu_mem_gb: float | None = None
    ranks_seen: int = 0  # distinct SGLang rank tags (TP0..TPn) in the log
    # Problems.
    warnings: int = 0
    errors: int = 0
    issues: dict[str, IssueSummary] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


class RoleLifecycle(Frozen):
    """One role's containers; the slowest sets ``time_to_ready_s``."""

    role: str
    containers: int
    ready: int = 0
    time_to_ready_s: float | None = None  # from started_at, else first log line
    slowest: str | None = None
    weights_load_s_max: float | None = None
    cuda_graph_capture_s_max: float | None = None
    weight_mem_gb_max: float | None = None
    kv_tokens_min: int | None = None
    kv_cache_gb_min: float | None = None
    max_total_num_tokens_min: int | None = None
    max_running_requests_min: int | None = None
    context_len: int | None = None
    available_gpu_mem_gb_min: float | None = None
    issues: dict[str, int] = Field(default_factory=dict)


class DeployLifecycle(Frozen):
    """Cold-start cost and KV capacity of one deployment (DEPLOY record)."""

    started_at: float | None = None
    time_to_ready_s: float | None = None
    slowest: str | None = None
    roles: dict[str, RoleLifecycle] = Field(default_factory=dict)
    containers: tuple[ContainerLifecycle, ...] = ()
    issues: dict[str, int] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass
class _Events:
    first: dict[str, LogLine] = field(default_factory=dict)
    last: dict[str, LogLine] = field(default_factory=dict)
    lines: dict[str, list[LogLine]] = field(default_factory=dict)

    def add(self, name: str, line: LogLine) -> None:
        self.first.setdefault(name, line)
        self.last[name] = line
        self.lines.setdefault(name, []).append(line)

    def t_first(self, name: str) -> float | None:
        line = self.first.get(name)
        return line.t if line else None

    def t_last(self, name: str) -> float | None:
        line = self.last.get(name)
        return line.t if line else None

    def values(self, name: str, pattern: str) -> list[float]:
        found = []
        for line in self.lines.get(name, ()):
            value = _float_field(line.message, pattern)
            if value is not None:
                found.append(value)
        return found


def _min(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def _max(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def _since(t: float | None, t0: float | None) -> float | None:
    if t is None or t0 is None:
        return None
    return round(t - t0, 3)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_LINE_CHARS else text[: MAX_LINE_CHARS - 3] + "..."


def _issues(lines: Sequence[LogLine]) -> tuple[dict[str, IssueSummary], int, int]:
    counts: dict[str, list[str]] = {}
    totals: dict[str, int] = {}
    warnings = errors = 0
    for line in lines:
        if line.level in ("WARN", "WARNING") or _PY_WARNING_RE.search(line.message):
            warnings += 1
        if line.level in ("ERROR", "CRITICAL"):
            errors += 1
        for category, pattern in ISSUE_PATTERNS:
            if pattern.search(line.message):
                totals[category] = totals.get(category, 0) + 1
                first = counts.setdefault(category, [])
                if len(first) < MAX_ISSUE_LINES:
                    first.append(_clip(line.message))
    issues = {
        category: IssueSummary(count=totals[category], first=tuple(counts[category]))
        for category, _ in ISSUE_PATTERNS
        if category in totals
    }
    return issues, warnings, errors


def parse_container_log(
    text: str,
    *,
    name: str,
    role: str,
    node: str = "",
    started_at: float | None = None,
) -> ContainerLifecycle:
    """Lifecycle of one container from its ``docker logs`` text.

    ``started_at`` (unix seconds, the deployment start) adds
    ``deploy_to_ready_s``; it may precede the first log line by the image
    start and container creation time.
    """
    lines = parse_log_lines(text)
    events = _Events()
    ready: LogLine | None = None
    ranks: set[str] = set()
    kv_dtype: str | None = None
    shards: tuple[int, float] | None = None
    for line in lines:
        if line.ranks:
            ranks.add(line.ranks)
        for event, pattern in EVENT_PATTERNS:
            if pattern.search(line.message):
                events.add(event, line)
        if ready is None and any(p.search(line.message) for p in READY_PATTERNS):
            ready = line
        if kv_dtype is None:
            kv_dtype = _str_field(line.message, r"Using KV cache dtype:\s*(\S+)")
        shard = _SHARDS_RE.search(line.message)
        if shard is not None:
            a, b, c = shard.group(2), shard.group(3), shard.group(4)
            seconds = (
                int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
            )
            shards = (int(shard.group(1)), float(seconds))
    notes: list[str] = []
    times = [line.t for line in lines if line.t is not None]
    t0 = times[0] if times else None
    if not times and lines:
        notes.append("no timestamps in the log: times unavailable")
    if "weights_end" in events.first and "weights_begin" not in events.first:
        notes.append(
            "log head missing (docker logs --tail?): times count from the first "
            "kept line"
        )
    scheduler_t = events.t_last("scheduler_ready")
    ready_source: Literal["marker", "scheduler", "none"] = "none"
    ready_t: float | None = None
    if ready is not None:
        ready_t, ready_source = ready.t, "marker"
    elif scheduler_t is not None:
        ready_t, ready_source = scheduler_t, "scheduler"
    weights_end = events.first.get("weights_end")
    kv_line = events.first.get("kv_allocated")
    sched_line = events.first.get("scheduler_ready")
    sched = sched_line.message if sched_line else ""
    kv_text = kv_line.message if kv_line else ""
    k_gb = _float_field(kv_text, r"(?<![A-Za-z])K size:" + _GB)
    v_gb = _float_field(kv_text, r"(?<![A-Za-z])V size:" + _GB)
    kv_gb = _float_field(kv_text, r"\bKV size:" + _GB)
    if kv_gb is None and k_gb is not None and v_gb is not None:
        kv_gb = round(k_gb + v_gb, 3)
    elapsed = events.values("weights_end", r"elapsed=([0-9.]+)\s*s")
    graph_elapsed = events.values("cuda_graph_end", r"Time elapsed:\s*([0-9.]+)\s*s")
    weights_load_s = _max(elapsed)
    if weights_load_s is None:
        weights_load_s = _since(
            events.t_last("weights_end"), events.t_first("weights_begin")
        )
    if weights_load_s is None and shards is not None:
        weights_load_s = shards[1]
    graph_s = _max(graph_elapsed)
    if graph_s is None:
        graph_s = _since(
            events.t_last("cuda_graph_end"), events.t_first("cuda_graph_begin")
        )
    pool_t = events.t_last("memory_pool_end") or events.t_last("kv_allocated")
    graph_t = events.t_last("cuda_graph_end")
    issues, warnings, errors = _issues(lines)
    return ContainerLifecycle(
        name=name,
        role=role,
        node=node,
        lines=len(lines),
        naive_utc=any(line.naive for line in lines),
        first_log_t=t0,
        last_log_t=times[-1] if times else None,
        weights_begin_t=events.t_first("weights_begin"),
        weights_loaded_t=events.t_last("weights_end"),
        kv_pool_t=pool_t,
        cuda_graph_t=graph_t,
        scheduler_ready_t=scheduler_t,
        ready_t=ready_t,
        ready_source=ready_source,
        ready_line=_clip(ready.message) if ready is not None else None,
        to_weights_loaded_s=_since(events.t_last("weights_end"), t0),
        to_kv_pool_s=_since(pool_t, t0),
        to_cuda_graph_s=_since(graph_t, t0),
        to_ready_s=_since(ready_t, t0),
        deploy_to_ready_s=_since(ready_t, started_at),
        weights_load_s=weights_load_s,
        kv_pool_s=_since(pool_t, events.t_last("weights_end")),
        cuda_graph_capture_s=graph_s,
        post_graph_to_ready_s=_since(ready_t, graph_t) if graph_t else None,
        checkpoint_shards=shards[0] if shards else None,
        shard_load_s=shards[1] if shards else None,
        model_type=_str_field(weights_end.message, r"\btype=([^,\s]+)")
        if weights_end
        else None,
        dtype=_str_field(weights_end.message, r"\bdtype=([^,\s]+)")
        if weights_end
        else None,
        kv_cache_dtype=kv_dtype,
        avail_mem_before_load_gb=_min(
            events.values("weights_begin", r"avail mem" + _GB)
        ),
        weight_mem_gb=_max(events.values("weights_end", r"mem usage" + _GB)),
        avail_mem_after_load_gb=_min(events.values("weights_end", r"avail mem" + _GB)),
        kv_tokens=_int_field(kv_text, r"#tokens:\s*([0-9]+)"),
        kv_cache_gb=kv_gb,
        avail_mem_after_pool_gb=_min(
            events.values("memory_pool_end", r"avail mem" + _GB)
        ),
        cuda_graph_mem_gb=_max(events.values("cuda_graph_end", r"mem usage" + _GB)),
        avail_mem_after_cuda_graph_gb=_min(
            events.values("cuda_graph_end", r"avail mem" + _GB)
        ),
        max_total_num_tokens=_int_field(sched, r"max_total_num_tokens\s*=\s*([0-9]+)"),
        chunked_prefill_size=_int_field(
            sched, r"chunked_prefill_size\s*=\s*(-?[0-9]+)"
        ),
        max_prefill_tokens=_int_field(sched, r"max_prefill_tokens\s*=\s*([0-9]+)"),
        max_running_requests=_int_field(sched, r"max_running_requests\s*=\s*([0-9]+)"),
        context_len=_int_field(sched, r"context_len\s*=\s*([0-9]+)"),
        available_gpu_mem_gb=_min(
            events.values("scheduler_ready", r"available_gpu_mem" + _GB)
        ),
        ranks_seen=len(ranks),
        warnings=warnings,
        errors=errors,
        issues=issues,
        notes=tuple(notes),
    )


def _role_summary(
    role: str, items: Sequence[ContainerLifecycle], started_at: float | None
) -> RoleLifecycle:
    ready: list[tuple[float, str]] = []
    for item in items:
        value = item.deploy_to_ready_s if started_at is not None else item.to_ready_s
        if value is not None:
            ready.append((value, item.name))
    issues: dict[str, int] = {}
    for item in items:
        for category, summary in item.issues.items():
            issues[category] = issues.get(category, 0) + summary.count

    def lowest(attr: str) -> Any:
        values = [getattr(i, attr) for i in items if getattr(i, attr) is not None]
        return min(values) if values else None

    def highest(attr: str) -> Any:
        values = [getattr(i, attr) for i in items if getattr(i, attr) is not None]
        return max(values) if values else None

    slowest = max(ready) if ready else None
    return RoleLifecycle(
        role=role,
        containers=len(items),
        ready=len(ready),
        time_to_ready_s=slowest[0] if slowest else None,
        slowest=slowest[1] if slowest else None,
        weights_load_s_max=highest("weights_load_s"),
        cuda_graph_capture_s_max=highest("cuda_graph_capture_s"),
        weight_mem_gb_max=highest("weight_mem_gb"),
        kv_tokens_min=lowest("kv_tokens"),
        kv_cache_gb_min=lowest("kv_cache_gb"),
        max_total_num_tokens_min=lowest("max_total_num_tokens"),
        max_running_requests_min=lowest("max_running_requests"),
        context_len=lowest("context_len"),
        available_gpu_mem_gb_min=lowest("available_gpu_mem_gb"),
        issues=dict(sorted(issues.items())),
    )


ROLE_ORDER = ("control", "frontend", "agg", "prefill", "decode")


def deploy_lifecycle(
    containers: Sequence[tuple[str, str, str, str]],
    started_at: float | None,
) -> DeployLifecycle:
    """Summarize a deployment from ``(name, role, node, log_text)`` tuples.

    ``started_at`` is the deployment start in unix seconds (the runner's
    clock; node clocks are assumed NTP-synchronized), else times count from
    each container's first log line.
    """
    parsed = [
        parse_container_log(
            text, name=name, role=role, node=node, started_at=started_at
        )
        for name, role, node, text in containers
    ]
    by_role: dict[str, list[ContainerLifecycle]] = {}
    for item in parsed:
        by_role.setdefault(item.role, []).append(item)
    roles = {
        role: _role_summary(role, by_role[role], started_at)
        for role in sorted(
            by_role,
            key=lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else len(ROLE_ORDER),
        )
    }
    notes: list[str] = []
    for item in parsed:
        if item.ready_source == "none" and item.role not in ("control",):
            notes.append(f"{item.name}: no readiness line found")
        notes.extend(f"{item.name}: {n}" for n in item.notes)
    serving = [r for r in roles.values() if r.role != "control" and r.time_to_ready_s]
    slowest = max(serving, key=lambda r: r.time_to_ready_s or 0.0) if serving else None
    issues: dict[str, int] = {}
    for role in roles.values():
        for category, count in role.issues.items():
            issues[category] = issues.get(category, 0) + count
    return DeployLifecycle(
        started_at=started_at,
        time_to_ready_s=slowest.time_to_ready_s if slowest else None,
        slowest=slowest.slowest if slowest else None,
        roles=roles,
        containers=tuple(parsed),
        issues=dict(sorted(issues.items())),
        notes=tuple(notes),
    )
