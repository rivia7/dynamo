# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce an Nsight Systems SQLite export to per-GPU wall-time totals.

    python3 nsys_timeline.py REPORT.sqlite OUT.json --steps N

Runs inside the diagnostic image (standard library only), next to the
report, so only a small JSON leaves the node. For every CUDA device of the
capture it sweeps the kernel, memcpy and memset intervals in time order:

* ``busy_ns``: time with at least one interval running; the rest of the
  device's window (first start to last end) is idle, split into gaps by
  length (``gaps``: [lo_ns, hi_ns or null, count, total_ns]).
* ``exclusive_ns``: wall time per name, overlapping intervals splitting
  their common time evenly, so the names of a device sum to ``busy_ns``.
* ``big_gaps``: the longest idle gaps (>= 100 us) with the names around
  them ([offset from the device's start, length, name before, name after]).
* ``idle_bins``: the window cut into ``IDLE_BINS`` bins of ``bin_ns``, each
  bin's idle share in per mille: where the bubbles sit over the capture.
* ``name_stats``: per name, [name index, calls, total_ns, median_ns,
  past_median_ns]; past_median_ns sums the time calls ran past that name's
  median length (for a collective, mostly waiting for a slower rank).
* ``longest``: the ``LONGEST_KEPT`` longest intervals, longest first
  ([offset from the device's start, length, name index]).
* ``sample``: the raw intervals of the lowest device over two mean steps
  from the middle of its window ([name index, start, end] in ns from
  ``sample.start_ns``), for a timeline figure.

Names are classified by the harness (``domain/steps.py``), not here.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sqlite3
import sys

GAP_EDGES_NS = (0, 2_000, 10_000, 100_000, 1_000_000)
BIG_GAP_NS = 100_000
BIG_GAPS_KEPT = 12
IDLE_BINS = 400
LONGEST_KEPT = 16
SAMPLE_STEPS = 2
SAMPLE_MAX_ROWS = 20_000
COPY_KINDS = {
    1: "Host-to-Device",
    2: "Device-to-Host",
    8: "Device-to-Device",
    10: "Peer-to-Peer",
}


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _tables(db: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def read_intervals(
    db: sqlite3.Connection,
) -> tuple[list[str], dict[int, list[tuple[int, int, int]]], dict]:
    """(names, device -> [(start, end, name index)], table counts)."""
    tables = _tables(db)
    names: list[str] = []
    index: dict[str, int] = {}

    def name_id(text: str) -> int:
        if text not in index:
            index[text] = len(names)
            names.append(text)
        return index[text]

    strings: dict[int, str] = {}
    if "StringIds" in tables:
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
    per_device: dict[int, list[tuple[int, int, int]]] = {}
    counts: dict[str, int] = {}
    kernel = "CUPTI_ACTIVITY_KIND_KERNEL"
    if kernel in tables:
        cols = _columns(db, kernel)
        column = next(
            (c for c in ("demangledName", "shortName", "mangledName") if c in cols),
            None,
        )
        rows = db.execute(
            f"SELECT start, end, deviceId, {column or 'NULL'} FROM {kernel}"
        )
        n = 0
        for start, end, device, sid in rows:
            label = strings.get(sid) if sid is not None else None
            per_device.setdefault(int(device), []).append(
                (int(start), int(end), name_id(label or "[unknown kernel]"))
            )
            n += 1
        counts["kernels"] = n
    for table, label in (
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        if table not in tables:
            continue
        cols = _columns(db, table)
        kind = "copyKind" if "copyKind" in cols and label == "memcpy" else "NULL"
        n = 0
        for start, end, device, copy in db.execute(
            f"SELECT start, end, deviceId, {kind} FROM {table}"
        ):
            suffix = f" {COPY_KINDS[copy]}" if copy in COPY_KINDS else ""
            per_device.setdefault(int(device), []).append(
                (int(start), int(end), name_id(f"[CUDA {label}{suffix}]"))
            )
            n += 1
        counts[f"{label}s"] = n
    return names, per_device, counts


def _bucket(gap: int) -> int:
    slot = 0
    for i, edge in enumerate(GAP_EDGES_NS):
        if gap >= edge:
            slot = i
    return slot


def sweep(intervals: list[tuple[int, int, int]]) -> dict:
    """Busy time, per-name exclusive time and idle gaps of one device."""
    events: list[tuple[int, int, int]] = []
    for start, end, name in intervals:
        if end < start:
            continue
        events.append((start, 1, name))
        events.append((end, -1, name))
    events.sort()
    active: dict[int, int] = {}
    depth = 0
    busy = 0
    exclusive: dict[int, float] = {}
    gaps = [
        [edge, GAP_EDGES_NS[i + 1] if i + 1 < len(GAP_EDGES_NS) else None, 0, 0]
        for i, edge in enumerate(GAP_EDGES_NS)
    ]
    big: list[tuple[int, int, int, int]] = []  # min-heap (length, start, before, after)
    idle: list[tuple[int, int]] = []
    last_end = -1
    prev = events[0][0] if events else 0
    i = 0
    while i < len(events):
        t = events[i][0]
        span = t - prev
        if span > 0:
            if depth > 0:
                busy += span
                for name, count in active.items():
                    exclusive[name] = exclusive.get(name, 0.0) + span * count / depth
            else:
                slot = gaps[_bucket(span)]
                slot[2] += 1
                slot[3] += span
                idle.append((prev, t))
                if span >= BIG_GAP_NS:  # nothing runs: the next event is a start
                    heapq.heappush(big, (span, prev, last_end, events[i][2]))
                    if len(big) > BIG_GAPS_KEPT:
                        heapq.heappop(big)
        while i < len(events) and events[i][0] == t:
            _, step, name = events[i]
            if step < 0:
                last_end = name
            count = active.get(name, 0) + step
            if count:
                active[name] = count
            else:
                active.pop(name, None)
            depth += step
            i += 1
        prev = t
    start = events[0][0] if events else 0
    end = events[-1][0] if events else 0
    bin_ns = (end - start) / IDLE_BINS if end > start else 0.0
    bins = [0.0] * IDLE_BINS if bin_ns else []
    for lo, hi in idle:
        first = min(int((lo - start) / bin_ns), IDLE_BINS - 1)
        last = min(int((hi - start) / bin_ns), IDLE_BINS - 1)
        for b in range(first, last + 1):
            b_lo = start + b * bin_ns
            bins[b] += max(0.0, min(hi, b_lo + bin_ns) - max(lo, b_lo))
    return {
        "start_ns": start,
        "end_ns": end,
        "busy_ns": busy,
        "intervals": len(intervals),
        "exclusive_ns": {str(k): round(v) for k, v in sorted(exclusive.items())},
        "gaps": gaps,
        "big_gaps": sorted([s - start, n, a, b] for n, s, a, b in big),
        "bin_ns": round(bin_ns, 1),
        "idle_bins": [min(1000, round(v / bin_ns * 1000)) for v in bins],
    }


def name_stats(
    intervals: list[tuple[int, int, int]], start: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Per name [name, calls, total, median, time past the median]; and the
    longest intervals [offset from start, length, name], longest first."""
    lengths: dict[int, list[int]] = {}
    longest: list[tuple[int, int, int]] = []  # min-heap (length, start, name)
    for s, e, name in intervals:
        if e < s:
            continue
        lengths.setdefault(name, []).append(e - s)
        heapq.heappush(longest, (e - s, s, name))
        if len(longest) > LONGEST_KEPT:
            heapq.heappop(longest)
    stats = []
    for name in sorted(lengths):
        values = sorted(lengths[name])
        median = values[len(values) // 2]
        stats.append(
            [
                name,
                len(values),
                sum(values),
                median,
                sum(v - median for v in values if v > median),
            ]
        )
    return stats, [[s - start, n, name] for n, s, name in sorted(longest, reverse=True)]


def sample(
    intervals: list[tuple[int, int, int]], start: int, end: int, steps: int
) -> dict:
    """Raw intervals of two mean steps from the middle of the window."""
    step = (end - start) / max(steps, 1)
    lo = int(start + (end - start) / 2)
    hi = int(min(end, lo + SAMPLE_STEPS * step))
    rows = [
        [name, max(s, lo) - lo, min(e, hi) - lo]
        for s, e, name in sorted(intervals)
        if e > lo and s < hi
    ][:SAMPLE_MAX_ROWS]
    return {"start_ns": lo, "end_ns": hi, "rows": rows}


def reduce_db(db: sqlite3.Connection, steps: int) -> dict:
    names, per_device, counts = read_intervals(db)
    devices = []
    for device in sorted(per_device):
        swept = sweep(per_device[device])
        swept["name_stats"], swept["longest"] = name_stats(
            per_device[device], swept["start_ns"]
        )
        swept["device"] = device
        devices.append(swept)
    out = {
        "version": 1,
        "steps": steps,
        "names": names,
        "tables": counts,
        "devices": devices,
    }
    if devices:
        first = devices[0]
        picked = sample(
            per_device[first["device"]], first["start_ns"], first["end_ns"], steps
        )
        picked["device"] = first["device"]
        out["sample"] = picked
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite")
    parser.add_argument("out")
    parser.add_argument(
        "--steps", type=int, default=1, help="forward steps the capture holds"
    )
    args = parser.parse_args(argv)
    if not os.path.isfile(args.sqlite):  # connect() would create an empty one
        print(f"no SQLite export at {args.sqlite}", file=sys.stderr)
        return 2
    db = sqlite3.connect(args.sqlite)
    try:
        doc = reduce_db(db, args.steps)
    finally:
        db.close()
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"devices={len(doc['devices'])} names={len(doc['names'])} {doc['tables']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
