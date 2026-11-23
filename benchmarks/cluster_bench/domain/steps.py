# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-time split of the forward steps in a diagnostic window (pure).

``agents/nsys_timeline.py`` reduces each Nsight Systems report on its node:
per CUDA device, the busy time, the wall time of every kernel or memory
operation name (overlapping intervals split their common time evenly) and the
idle gaps between them. This module classifies those names with the kernel
rules of ``domain/kernels.py`` and answers "of one forward step, how much
is communication, attention, MoE and idle?":

* A device's window runs from its first to its last interval inside the
  capture; ``start_profile`` captured ``steps`` forward steps, so
  ``step_ms`` = window / steps (mean over devices).
* ``shares``: group -> share of the window (mean over devices); the
  groups and ``idle`` sum to 1. Idle is time without any kernel or copy
  on that GPU: launch gaps inside a step and host time between steps.
* Unlike ``KernelBreakdown`` (summed GPU time over every stream), these are
  shares of wall time. Kernels that wait for a peer (NCCL) count as busy:
  rank skew shows up as communication, not idle. ``wait_share`` is the
  part of it that collectives ran past their median length (mean over
  devices), mostly waiting for a slower rank rather than moving data.
* Times of gaps and calls run from the report's first interval: the GPUs of
  one report (one instance) share a clock, so they line up.
* ``steady_*``: the same with stalls cut out of every GPU's window: idle
  gaps and collectives of at least ``STALL_US``, far longer than a decode
  step's own pauses. Starting a capture can stall every process of an
  instance once; the steady split is closer to an ordinary step.
* ``--cuda-graph-trace=node`` adds work around every graph node, so the
  short gaps run longer than without the profiler.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from benchmarks.cluster_bench.domain.base import Frozen
from benchmarks.cluster_bench.domain.kernels import (
    CATEGORIES,
    KernelCategory,
    classify_kernel,
    short_kernel_name,
)

Group = Literal["communication", "attention", "moe", "dense_gemm", "other", "idle"]
GROUPS: tuple[tuple[Group, tuple[KernelCategory, ...]], ...] = (
    ("communication", ("nccl", "other_comm", "deepep")),
    ("attention", ("attention",)),
    ("moe", ("moe_gemm", "moe_routing")),
    ("dense_gemm", ("dense_gemm",)),
    (
        "other",
        (
            "kv_cache",
            "norm_act",
            "quantization",
            "sampling",
            "memory",
            "elementwise",
            "other",
        ),
    ),
)
GROUP_ORDER: tuple[Group, ...] = (*(g for g, _ in GROUPS), "idle")
GROUP_OF: dict[str, Group] = {c: g for g, cats in GROUPS for c in cats}
MERGE_GAP_US = 1.0  # sample: same-category neighbours closer than this merge
SAMPLE_MAX_SEGMENTS = 4000
TIMELINE_VERSION = 1
LONG_CALL_US = 1000.0  # a collective this long waited for a peer
STALL_US = 5000.0  # an idle gap or a collective this long is a stall, not step work
STRIP_BINS = 200  # the sample, cut into bins for a picture of it
ZOOM_US = 1800.0  # a close-up of the sample: a few layers of a decode step


class GapBucket(Frozen):
    """Idle gaps of one length range: [lo_us, hi_us), hi None = unbounded."""

    lo_us: float
    hi_us: float | None = None
    count: float = 0  # gaps (a mean per step and GPU in a StepBreakdown)
    total_us: float = 0.0


class BigGap(Frozen):
    """One long idle gap of a GPU and the work on either side of it."""

    at_ms: float  # from the report's first interval
    us: float
    before: str = ""  # short kernel names
    after: str = ""
    before_category: KernelCategory = "other"
    after_category: KernelCategory = "other"


class LongInterval(Frozen):
    """One of a GPU's longest kernels or copies."""

    at_ms: float  # from the report's first interval
    us: float
    name: str = ""  # short kernel name
    category: KernelCategory = "other"


class DeviceTimeline(Frozen):
    """One GPU of one report."""

    source: str
    device: int  # CUDA device id inside the container
    offset_ms: float = 0.0  # its first interval, from the report's first one
    window_us: float
    busy_us: float
    intervals: int = 0
    categories: dict[str, float] = Field(default_factory=dict)  # category -> us
    gaps: tuple[GapBucket, ...] = ()
    big_gaps: tuple[BigGap, ...] = ()  # the longest (>= 100 us), in time order
    bin_ms: float | None = None
    idle_bins: tuple[int, ...] = ()  # idle per mille of each bin of the window
    wait_us: float = 0.0  # collectives' time past their median length
    longest: tuple[LongInterval, ...] = ()  # longest first

    @property
    def idle_us(self) -> float:
        return max(0.0, self.window_us - self.busy_us)

    def stalls_us(self) -> tuple[float, float]:
        """(idle, communication) time in stalls: idle gaps and collectives of
        at least STALL_US (only the longest few of each are recorded)."""
        idle = sum(g.us for g in self.big_gaps if g.us >= STALL_US)
        comm = sum(
            c.us
            for c in self.longest
            if c.us >= STALL_US and GROUP_OF.get(c.category) == "communication"
        )
        return idle, comm


class Segment(Frozen):
    """A stretch of one category on the sample GPU (us from the sample start)."""

    start_us: float
    end_us: float
    category: KernelCategory


class StepBreakdown(Frozen):
    """Wall-time split of the captured forward steps of one role."""

    steps: int
    sources: tuple[str, ...] = ()
    devices: tuple[DeviceTimeline, ...] = ()
    window_us: float = 0.0  # mean over devices
    step_ms: float | None = None
    shares: dict[str, float] = Field(default_factory=dict)  # group -> share
    category_shares: dict[str, float] = Field(default_factory=dict)
    gaps: tuple[GapBucket, ...] = ()  # per step, mean over devices
    wait_share: float = 0.0  # part of the communication share (see module doc)
    # the same with every GPU's stalls (see STALL_US) cut out of its window
    stall_ms: float = 0.0  # cut per GPU, mean over devices
    steady_shares: dict[str, float] = Field(default_factory=dict)
    steady_wait_share: float = 0.0
    steady_step_ms: float | None = None
    sample_source: str | None = None
    sample_device: int | None = None
    sample_us: float = 0.0  # length of the sample
    sample: tuple[Segment, ...] = ()
    notes: tuple[str, ...] = ()

    def share(self, group: Group) -> float:
        return self.shares.get(group, 0.0)


def category_of(name: str) -> KernelCategory:
    """The reducer names copies "[CUDA memcpy ...]", which the rules know."""
    return classify_kernel(name)


def parse_timeline(text: str) -> dict[str, Any] | None:
    """The reducer's JSON, or None when it is unreadable or of another version."""
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    if not isinstance(doc, dict) or doc.get("version") != TIMELINE_VERSION:
        return None
    if not isinstance(doc.get("names"), list) or not isinstance(
        doc.get("devices"), list
    ):
        return None
    return doc


def _gaps(raw: Sequence[Sequence[Any]]) -> tuple[GapBucket, ...]:
    out = []
    for lo, hi, count, total in raw:
        out.append(
            GapBucket(
                lo_us=lo / 1000,
                hi_us=None if hi is None else hi / 1000,
                count=count,
                total_us=total / 1000,
            )
        )
    return tuple(out)


def _name(names: Sequence[Any], i: int) -> str:
    return str(names[i]) if 0 <= i < len(names) else ""


def _big_gaps(
    raw: Sequence[Sequence[Any]], names: Sequence[Any], offset_ns: float
) -> tuple[BigGap, ...]:
    def name(i: int) -> str:
        return _name(names, i)

    return tuple(
        BigGap(
            at_ms=round((at + offset_ns) / 1e6, 4),
            us=round(length / 1000, 3),
            before=short_kernel_name(name(int(a))) if name(int(a)) else "",
            after=short_kernel_name(name(int(b))) if name(int(b)) else "",
            before_category=category_of(name(int(a))),
            after_category=category_of(name(int(b))),
        )
        for at, length, a, b in raw
    )


def _longest(
    raw: Sequence[Sequence[Any]], names: Sequence[Any], offset_ns: float
) -> tuple[LongInterval, ...]:
    return tuple(
        LongInterval(
            at_ms=round((at + offset_ns) / 1e6, 4),
            us=round(length / 1000, 3),
            name=short_kernel_name(_name(names, int(i)))
            if _name(names, int(i))
            else "",
            category=category_of(_name(names, int(i))),
        )
        for at, length, i in raw
    )


def _wait_us(
    rows: Sequence[Sequence[Any]], cats: Sequence[str], comm_us: float
) -> float:
    """Time the device's collectives ran past their median length, at most
    their wall time."""
    past = sum(
        r[4]
        for r in rows
        if 0 <= int(r[0]) < len(cats)
        and GROUP_OF.get(cats[int(r[0])]) == "communication"
    )
    return round(min(past / 1000, comm_us), 3)


def device_timelines(doc: Mapping[str, Any], source: str) -> list[DeviceTimeline]:
    names = doc["names"]
    cats = [category_of(str(n)) for n in names]
    origin = min((dev["start_ns"] for dev in doc["devices"]), default=0)
    out = []
    for dev in doc["devices"]:
        per_cat: dict[str, float] = {}
        for key, ns in (dev.get("exclusive_ns") or {}).items():
            cat = cats[int(key)]
            per_cat[cat] = per_cat.get(cat, 0.0) + ns / 1000
        offset = dev["start_ns"] - origin
        comm_us = sum(
            us for c, us in per_cat.items() if GROUP_OF.get(c) == "communication"
        )
        out.append(
            DeviceTimeline(
                source=source,
                device=int(dev["device"]),
                offset_ms=round(offset / 1e6, 4),
                window_us=(dev["end_ns"] - dev["start_ns"]) / 1000,
                busy_us=dev["busy_ns"] / 1000,
                intervals=int(dev.get("intervals") or 0),
                categories={c: per_cat[c] for c in CATEGORIES if c in per_cat},
                gaps=_gaps(dev.get("gaps") or ()),
                big_gaps=_big_gaps(dev.get("big_gaps") or (), names, offset),
                bin_ms=dev["bin_ns"] / 1e6 if dev.get("bin_ns") else None,
                idle_bins=tuple(int(v) for v in dev.get("idle_bins") or ()),
                wait_us=_wait_us(dev.get("name_stats") or (), cats, comm_us),
                longest=_longest(dev.get("longest") or (), names, offset),
            )
        )
    return out


def longest_gaps(b: StepBreakdown, n: int = 6) -> list[tuple[DeviceTimeline, BigGap]]:
    """The n longest idle gaps over the role's GPUs, longest first."""
    pairs = [(d, g) for d in b.devices for g in d.big_gaps]
    return sorted(pairs, key=lambda p: -p[1].us)[:n]


def longest_calls(
    b: StepBreakdown,
    n: int = 6,
    *,
    group: Group = "communication",
    min_us: float = LONG_CALL_US,
) -> list[tuple[DeviceTimeline, LongInterval]]:
    """The n longest calls of one group (at least min_us) over the role's
    GPUs, longest first."""
    pairs = [
        (d, c)
        for d in b.devices
        for c in d.longest
        if c.us >= min_us and GROUP_OF.get(c.category, "other") == group
    ]
    return sorted(pairs, key=lambda p: -p[1].us)[:n]


def sample_segments(doc: Mapping[str, Any]) -> tuple[float, tuple[Segment, ...]]:
    """(sample length us, merged segments) of the reducer's sample rows.

    Rows overlapping in time keep the category of the earlier one; a row's
    neighbour of the same category within ``MERGE_GAP_US`` extends it.
    """
    raw = doc.get("sample") or {}
    rows = raw.get("rows") or []
    names = doc["names"]
    length = (raw.get("end_ns", 0) - raw.get("start_ns", 0)) / 1000
    out: list[list[Any]] = []
    for idx, start, end in sorted(rows, key=lambda r: (r[1], r[2])):
        cat = category_of(str(names[int(idx)]))
        s, e = start / 1000, end / 1000
        if out and s <= out[-1][1]:  # overlap: the running segment keeps it
            if cat == out[-1][2]:
                out[-1][1] = max(out[-1][1], e)
            elif e > out[-1][1]:
                out.append([out[-1][1], e, cat])
            continue
        if out and cat == out[-1][2] and s - out[-1][1] < MERGE_GAP_US:
            out[-1][1] = max(out[-1][1], e)
            continue
        out.append([s, e, cat])
    if len(out) > SAMPLE_MAX_SEGMENTS:
        # the sample ends where the kept segments do: past them, the cut-off
        # rows would read as idle
        out = out[:SAMPLE_MAX_SEGMENTS]
        length = out[-1][1]
    segs = tuple(
        Segment(start_us=round(s, 3), end_us=round(e, 3), category=c)
        for s, e, c in out
        if e > s
    )
    return length, segs


def strip_bins(
    b: StepBreakdown, bins: int = STRIP_BINS
) -> tuple[float, list[dict[str, float]]]:
    """The sample cut into ``bins`` equal bins: (bin length in us, per bin
    group -> share of the bin). A bin's groups are capped at a full bin and
    its "idle" is the rest; kernels on two streams at once can sum past it."""
    if not b.sample or b.sample_us <= 0 or bins <= 0:
        return 0.0, []
    width = b.sample_us / bins
    out = [dict.fromkeys(GROUP_ORDER, 0.0) for _ in range(bins)]
    for seg in b.sample:
        group = GROUP_OF.get(seg.category, "other")
        lo, hi = max(seg.start_us, 0.0), min(seg.end_us, b.sample_us)
        for i in range(int(lo / width), min(int(hi / width), bins - 1) + 1):
            overlap = min(hi, (i + 1) * width) - max(lo, i * width)
            if overlap > 0:
                out[i][group] += overlap / width
    for shares in out:
        busy = sum(v for g, v in shares.items() if g != "idle")
        shares["idle"] = max(0.0, 1.0 - busy)
    return width, out


def zoom_segments(
    b: StepBreakdown, length_us: float = ZOOM_US
) -> tuple[float, tuple[Segment, ...]]:
    """A close-up of the sample: ``length_us`` from just before its first
    attention segment past a quarter of it (the start of a layer, clear of the
    sample's edges); times from the close-up's start. Returns (length, segments)."""
    if not b.sample or b.sample_us <= 0:
        return 0.0, ()
    quarter = b.sample_us / 4
    start = next(
        (
            s.start_us
            for s in b.sample
            if s.category == "attention" and s.start_us >= quarter
        ),
        quarter,
    )
    lo = max(0.0, start - 5.0)
    hi = min(b.sample_us, lo + length_us)
    segs = tuple(
        Segment(
            start_us=max(s.start_us, lo) - lo,
            end_us=min(s.end_us, hi) - lo,
            category=s.category,
        )
        for s in b.sample
        if s.end_us > lo and s.start_us < hi
    )
    return hi - lo, segs


def _group_us(d: DeviceTimeline) -> dict[str, float]:
    out = {g: 0.0 for g in GROUP_ORDER}
    for cat, us in d.categories.items():
        out[GROUP_OF.get(cat, "other")] += us
    out["idle"] = d.idle_us
    return out


def _summary(devices: Sequence[DeviceTimeline], steps: int) -> dict[str, Any]:
    """The StepBreakdown fields that average over devices."""
    live = [d for d in devices if d.window_us > 0]
    if not live:
        return {}
    n = len(live)
    window = sum(d.window_us for d in live) / n
    groups = {g: 0.0 for g in GROUP_ORDER}
    cats: dict[str, float] = {}
    steady = {g: 0.0 for g in GROUP_ORDER}
    steady_window = steady_wait = stall = wait = 0.0
    for d in live:
        for cat, us in d.categories.items():
            cats[cat] = cats.get(cat, 0.0) + us / d.window_us / n
        per_group = _group_us(d)
        for g, us in per_group.items():
            groups[g] += us / d.window_us / n
        wait += d.wait_us / d.window_us / n
        # the same GPU with its stalls cut out of the window
        idle_stall, comm_stall = d.stalls_us()
        left = d.window_us - idle_stall - comm_stall
        if left <= 0:
            left, idle_stall, comm_stall = d.window_us, 0.0, 0.0
        per_group["idle"] = max(0.0, per_group["idle"] - idle_stall)
        per_group["communication"] = max(0.0, per_group["communication"] - comm_stall)
        for g, us in per_group.items():
            steady[g] += us / left / n
        steady_wait += max(0.0, d.wait_us - comm_stall) / left / n
        steady_window += left / n
        stall += (idle_stall + comm_stall) / n
    per_step = max(steps, 1) * n
    buckets: list[GapBucket] = []
    for i, first in enumerate(live[0].gaps):
        count = sum(d.gaps[i].count for d in live if i < len(d.gaps))
        total = sum(d.gaps[i].total_us for d in live if i < len(d.gaps))
        buckets.append(
            first.model_copy(
                update={
                    "count": round(count / per_step, 3),
                    "total_us": round(total / per_step, 3),
                }
            )
        )
    order = {c: i for i, c in enumerate(CATEGORIES)}

    def per_step_ms(us: float) -> float | None:
        return round(us / steps / 1000, 4) if steps > 0 else None

    return {
        "window_us": round(window, 3),
        "step_ms": per_step_ms(window),
        "shares": {g: round(v, 6) for g, v in groups.items()},
        "category_shares": {
            c: round(v, 6)
            for c, v in sorted(cats.items(), key=lambda kv: order.get(kv[0], 99))
        },
        "gaps": tuple(buckets),
        "wait_share": round(wait, 6),
        "stall_ms": round(stall / 1000, 3),
        "steady_shares": {g: round(v, 6) for g, v in steady.items()},
        "steady_wait_share": round(steady_wait, 6),
        "steady_step_ms": per_step_ms(steady_window),
    }


def step_breakdown(doc: Mapping[str, Any], *, source: str, steps: int) -> StepBreakdown:
    """Breakdown of one report (one container: one TP instance)."""
    devices = device_timelines(doc, source)
    length, segs = sample_segments(doc)
    sample = doc.get("sample") or {}
    notes = []
    if not devices:
        notes.append(f"{source}: no kernels in the capture")
    return StepBreakdown(
        steps=steps,
        sources=(source,),
        devices=tuple(devices),
        **_summary(devices, steps),
        sample_source=source if segs else None,
        sample_device=sample.get("device") if segs else None,
        sample_us=round(length, 3) if segs else 0.0,
        sample=segs,
        notes=tuple(notes),
    )


def merge_step_breakdowns(items: Sequence[StepBreakdown]) -> StepBreakdown | None:
    """One role's reports together: every device counts once."""
    items = [i for i in items if i.devices]
    if not items:
        return None
    steps = items[0].steps
    devices = tuple(d for i in items for d in i.devices)
    first = next((i for i in items if i.sample), items[0])
    notes = tuple(dict.fromkeys(n for i in items for n in i.notes))
    if any(i.steps != steps for i in items):
        notes += ("reports captured different step counts; the first one's is used",)
    return StepBreakdown(
        steps=steps,
        sources=tuple(s for i in items for s in i.sources),
        devices=devices,
        **_summary(devices, steps),
        sample_source=first.sample_source,
        sample_device=first.sample_device,
        sample_us=first.sample_us,
        sample=first.sample,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Synthetic reducer output (dry-run simulator)
# --------------------------------------------------------------------------

_SYNTHETIC_STEP: tuple[tuple[str, float], ...] = (
    ("void flashinfer::BatchMLAPagedAttentionKernel<128>()", 300.0),
    (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        120.0,
    ),
    ("void cutlass::device_kernel<flashinfer::cutlass_moe_fp4_gemm>()", 520.0),
    ("void at::native::elementwise_kernel<128, 4>()", 60.0),
)


_SYNTHETIC_WAIT_NS = 1_200_000
_SYNTHETIC_STALL_NS = 6_000_000


def synthetic_timeline_json(role: str, *, devices: int = 8, steps: int = 10) -> str:
    """Reducer output of a plausible decode capture: per step and GPU, a few
    layers of attention, all-reduce and MoE with 2 us launch gaps and a 150 us
    host gap between steps; GPU 1's first all-reduce waits 1.2 ms for a peer,
    and GPU 2 stalls 6 ms after its first step."""
    names = [n for n, _ in _SYNTHETIC_STEP]
    layers = 4 if role != "prefill" else 2
    per_step = sum(d for _, d in _SYNTHETIC_STEP) * layers
    gaps_in = len(_SYNTHETIC_STEP) * layers - 1
    step_us = per_step + gaps_in * 2.0 + 150.0
    per_layer = sum(d for _, d in _SYNTHETIC_STEP) + len(_SYNTHETIC_STEP) * 2.0
    moe_at = sum(d + 2.0 for _, d in _SYNTHETIC_STEP[:2])
    calls = layers * steps
    out_devices = []
    for dev in range(devices):
        wait = _SYNTHETIC_WAIT_NS if dev == 1 and devices > 1 else 0
        stall = _SYNTHETIC_STALL_NS if dev == 2 and devices > 2 else 0
        host_gaps = steps - (1 if stall else 0)  # the 150 us ones
        exclusive = {
            str(i): round(d * calls * 1000) for i, (_, d) in enumerate(_SYNTHETIC_STEP)
        }
        exclusive["1"] += wait
        stats = [
            [
                i,
                calls,
                round(d * calls * 1000) + (wait if i == 1 else 0),
                round(d * 1000),
                wait if i == 1 else 0,
            ]
            for i, (_, d) in enumerate(_SYNTHETIC_STEP)
        ]
        longest = (
            [[round((_SYNTHETIC_STEP[0][1] + 2.0) * 1000), 120_000 + wait, 1]]
            if wait
            else []
        )
        longest += [
            [round((layer * per_layer + moe_at) * 1000), 520_000, 2]
            for layer in range(2)
        ]
        window_ns = round(step_us * steps * 1000) + wait + stall
        out_devices.append(
            {
                "device": dev,
                "start_ns": 0,
                "end_ns": window_ns,
                "busy_ns": round(per_step * steps * 1000) + wait,
                "intervals": len(_SYNTHETIC_STEP) * layers * steps,
                "exclusive_ns": exclusive,
                "gaps": [
                    [0, 2000, 0, 0],
                    [2000, 10000, gaps_in * steps, round(gaps_in * 2000 * steps)],
                    [10000, 100000, 0, 0],
                    [100000, 1000000, host_gaps, 150000 * host_gaps],
                    [
                        1000000,
                        None,
                        steps - host_gaps,
                        (150000 + stall) if stall else 0,
                    ],
                ],
                "big_gaps": [
                    [
                        round(((s + 1) * step_us - 150.0) * 1000)
                        + wait
                        + (stall if s > 0 else 0),
                        150_000 + (stall if s == 0 else 0),
                        2,
                        0,
                    ]
                    for s in range(min(steps, 3))
                ],
                "bin_ns": round(window_ns / (4 * steps), 1),
                "idle_bins": [0, 0, 0, round(150 / (step_us / 4) * 1000)] * steps,
                "name_stats": stats,
                "longest": longest,
            }
        )
    rows = []
    t = 0.0
    for _ in range(layers):
        for i, (_, d) in enumerate(_SYNTHETIC_STEP):
            rows.append([i, round(t * 1000), round((t + d) * 1000)])
            t += d + 2.0
    doc = {
        "version": TIMELINE_VERSION,
        "steps": steps,
        "names": names,
        "tables": {"kernels": len(rows) * steps * devices},
        "devices": out_devices,
        "sample": {
            "device": 0,
            "start_ns": 0,
            "end_ns": round(step_us * 2 * 1000),
            "rows": rows,
        },
    }
    return json.dumps(doc)
