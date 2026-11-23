# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step-timeline reducer (agents/nsys_timeline.py) and domain/steps.py."""

from __future__ import annotations

import json
import sqlite3

import pytest

from benchmarks.cluster_bench.agents import nsys_timeline as reducer
from benchmarks.cluster_bench.domain import steps

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

ATTN = "void flashinfer::BatchMLAPagedAttentionKernel<128>()"
NCCL = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
MOE = "void cutlass::device_kernel<flashinfer::cutlass_moe_fp4_gemm>()"
NAMES = (ATTN, NCCL, MOE)


def capture(path: str = ":memory:") -> sqlite3.Connection:
    """Two GPUs of an Nsight Systems export (times in ns), committed.

    GPU 0: attention, MoE overlapped by an all-reduce for 50 us, a 1 us gap,
    attention, a 5 us gap, a host-to-device copy, a 150 us gap, MoE.
    GPU 1: one all-reduce.
    """
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT,"
        " streamId INT, demangledName INT, shortName INT)"
    )
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, deviceId INT,"
        " streamId INT, copyKind INT)"
    )
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(NAMES))
    kernels = [
        (0, 100_000, 0, 7, 0),
        (100_000, 300_000, 0, 7, 2),
        (250_000, 400_000, 0, 8, 1),
        (401_000, 501_000, 0, 7, 0),
        (756_000, 856_000, 0, 7, 2),
        (0, 10_000, 1, 7, 1),
    ]
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
        [(s, e, dev, stream, n, n) for s, e, dev, stream, n in kernels],
    )
    db.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (506000, 606000, 0, 7, 1)"
    )
    db.commit()
    return db


def test_the_reducer_splits_wall_time_and_buckets_the_gaps() -> None:
    doc = reducer.reduce_db(capture(), steps=2)
    assert doc["version"] == steps.TIMELINE_VERSION
    assert doc["tables"] == {"kernels": 6, "memcpys": 1}
    names = doc["names"]
    assert "[CUDA memcpy Host-to-Device]" in names
    gpu0, gpu1 = doc["devices"]
    assert (gpu0["device"], gpu0["start_ns"], gpu0["end_ns"]) == (0, 0, 856_000)
    assert gpu0["busy_ns"] == 700_000
    exclusive = {names[int(k)]: v for k, v in gpu0["exclusive_ns"].items()}
    # the 50 us of overlap is split evenly between MoE and the all-reduce
    assert exclusive == {
        ATTN: 200_000,
        MOE: 275_000,
        NCCL: 125_000,
        "[CUDA memcpy Host-to-Device]": 100_000,
    }
    assert sum(exclusive.values()) == gpu0["busy_ns"]
    assert [g[2:] for g in gpu0["gaps"]] == [
        [1, 1_000],
        [1, 5_000],
        [0, 0],
        [1, 150_000],
        [0, 0],
    ]
    # the one gap over 100 us, with the work on either side of it
    copy = names.index("[CUDA memcpy Host-to-Device]")
    assert gpu0["big_gaps"] == [[606_000, 150_000, copy, names.index(MOE)]]
    # idle per mille of 400 bins over the window: 156 us in all
    bins = gpu0["idle_bins"]
    assert len(bins) == reducer.IDLE_BINS and gpu0["bin_ns"] == 2140.0
    assert sum(bins) * gpu0["bin_ns"] / 1000 == pytest.approx(156_000, rel=0.01)
    assert bins[0] == 0 and bins[int(700_000 / 2140)] == 1000
    # per name: [name, calls, total, median, time past the median]; the longest calls
    stats = {names[row[0]]: row[1:] for row in gpu0["name_stats"]}
    assert stats[MOE] == [2, 300_000, 200_000, 0] and stats[NCCL] == [
        1,
        150_000,
        150_000,
        0,
    ]
    assert gpu0["longest"][:2] == [
        [100_000, 200_000, names.index(MOE)],
        [250_000, 150_000, names.index(NCCL)],
    ]
    assert len(gpu0["longest"]) == 6
    assert gpu1["busy_ns"] == gpu1["end_ns"] - gpu1["start_ns"] == 10_000
    assert gpu1["big_gaps"] == [] and set(gpu1["idle_bins"]) == {0}
    # two mean steps from the middle of GPU 0's window, clipped to it
    sample = doc["sample"]
    assert (sample["device"], sample["start_ns"], sample["end_ns"]) == (
        0,
        428_000,
        856_000,
    )
    assert [names[r[0]] for r in sample["rows"]] == [
        ATTN,
        "[CUDA memcpy Host-to-Device]",
        MOE,
    ]
    assert sample["rows"][0][1:] == [0, 73_000]


def test_name_stats_sum_the_time_past_each_median() -> None:
    # three all-reduces, the last one waiting about 1 us for a peer, and an attention
    intervals = [
        (1_000, 1_010, 1),
        (1_020, 1_030, 1),
        (1_040, 2_040, 1),
        (1_050, 1_060, 0),
    ]
    stats, longest = reducer.name_stats(intervals, start=1_000)
    assert stats == [[0, 1, 10, 10, 0], [1, 3, 1_020, 10, 990]]
    assert longest == [[40, 1_000, 1], [50, 10, 0], [20, 10, 1], [0, 10, 1]]


def test_the_reducer_writes_json_from_a_file(tmp_path) -> None:
    path = tmp_path / "r.sqlite"
    capture(str(path)).close()
    out = tmp_path / "r_timeline.json"
    assert reducer.main([str(path), str(out), "--steps", "2"]) == 0
    assert steps.parse_timeline(out.read_text()) is not None
    missing = tmp_path / "missing.sqlite"
    assert reducer.main([str(missing), str(out)]) == 2 and not missing.exists()


def test_a_step_breakdown_groups_categories_and_adds_idle() -> None:
    doc = reducer.reduce_db(capture(), steps=2)
    b = steps.step_breakdown(doc, source="cb-x-decode0-r0", steps=2)
    gpu0, gpu1 = b.devices
    assert gpu0.window_us == 856.0 and gpu0.idle_us == pytest.approx(156.0)
    assert gpu0.categories == {
        "attention": 200.0,
        "moe_gemm": 275.0,
        "nccl": 125.0,
        "memory": 100.0,
    }
    assert gpu1.categories == {"nccl": 10.0}
    # shares are means over the GPUs, each GPU's window counting once
    assert b.share("communication") == pytest.approx((125 / 856 + 1.0) / 2, abs=1e-6)
    assert b.share("moe") == pytest.approx(275 / 856 / 2, abs=1e-6)
    assert b.share("other") == pytest.approx(100 / 856 / 2, abs=1e-6)
    assert b.share("idle") == pytest.approx(156 / 856 / 2, abs=1e-6)
    assert sum(b.shares.values()) == pytest.approx(1.0, abs=1e-5)
    assert b.step_ms == pytest.approx((856 + 10) / 2 / 2 / 1000, abs=1e-4)
    # gaps per step and GPU: GPU 0's three gaps over 2 steps x 2 GPUs
    assert [(g.count, g.total_us) for g in b.gaps] == [
        (0.25, 0.25),
        (0.25, 1.25),
        (0.0, 0.0),
        (0.25, 37.5),
        (0.0, 0.0),
    ]
    (gap,) = gpu0.big_gaps
    assert (gap.at_ms, gap.us) == (0.606, 150.0)
    assert gap.before == "[CUDA memcpy Host-to-Device]"
    assert (gap.before_category, gap.after_category) == ("memory", "moe_gemm")
    assert steps.longest_gaps(b) == [(gpu0, gap)]
    assert gpu0.bin_ms == pytest.approx(0.00214) and len(gpu0.idle_bins) == 400
    assert [s.category for s in b.sample] == ["attention", "memory", "moe_gemm"]
    assert b.sample_us == 428.0 and b.sample_source == "cb-x-decode0-r0"


def test_merging_reports_counts_every_gpu_once() -> None:
    doc = reducer.reduce_db(capture(), steps=2)
    one = steps.step_breakdown(doc, source="a", steps=2)
    two = steps.step_breakdown(doc, source="b", steps=2)
    merged = steps.merge_step_breakdowns([one, two])
    assert merged is not None and merged.sources == ("a", "b")
    assert len(merged.devices) == 4
    assert merged.shares == one.shares and merged.gaps == one.gaps
    assert merged.wait_share == one.wait_share
    assert merged.sample_source == "a"
    assert steps.merge_step_breakdowns([]) is None


def test_waits_and_long_calls_line_up_on_the_report_clock() -> None:
    """GPU 1 starts 0.5 ms after GPU 0, then stalls for 2 ms; GPU 0's
    all-reduce waits for it."""
    gpu0_raw = {
        "device": 0,
        "start_ns": 1_000_000,
        "end_ns": 5_000_000,
        "busy_ns": 4_000_000,
        "exclusive_ns": {"0": 1_500_000, "1": 2_500_000},
        "name_stats": [
            [0, 10, 1_500_000, 150_000, 0],
            [1, 6, 2_500_000, 100_000, 2_000_000],
        ],
        "longest": [[1_000_000, 2_100_000, 1], [0, 150_000, 0]],
    }
    gpu1_raw = {
        "device": 1,
        "start_ns": 1_500_000,
        "end_ns": 5_000_000,
        "busy_ns": 1_000_000,
        "exclusive_ns": {"0": 500_000, "1": 500_000},
        "name_stats": [[0, 4, 500_000, 125_000, 0], [1, 5, 500_000, 100_000, 0]],
        "longest": [[0, 125_000, 0]],
        "big_gaps": [[500_000, 2_000_000, 0, 1]],
    }
    doc = {"names": [ATTN, NCCL], "devices": [gpu0_raw, gpu1_raw]}
    b = steps.step_breakdown(doc, source="cb-x-decode0-r0", steps=4)
    gpu0, gpu1 = b.devices
    assert (gpu0.offset_ms, gpu1.offset_ms) == (0.0, 0.5)
    assert (gpu0.wait_us, gpu1.wait_us) == (2000.0, 0.0)
    assert b.wait_share == pytest.approx((2000 / 4000 + 0) / 2, abs=1e-6)
    # GPU 1's idle gap and GPU 0's long all-reduce sit at the same time
    (gap,) = gpu1.big_gaps
    assert (gap.at_ms, gap.us) == (1.0, 2000.0)
    ((dev, call),) = steps.longest_calls(b)
    assert dev is gpu0 and (call.at_ms, call.us, call.category) == (1.0, 2100.0, "nccl")
    assert call.name.startswith("ncclDevKernel_AllReduce")
    # a wait never exceeds the communication time itself
    shorter = {**gpu0_raw, "exclusive_ns": {"0": 3_000_000, "1": 1_000_000}}
    (capped,) = steps.device_timelines({**doc, "devices": [shorter]}, "s")
    assert capped.wait_us == 1000.0


def test_stalls_are_cut_out_of_the_steady_split() -> None:
    """Two GPUs of 20 ms with a 6 ms stall each: GPU 0 waits in an
    all-reduce, GPU 1 sits idle; outside it, 12 ms attention, 1 ms all-reduce
    and 1 ms idle on both."""
    gpu0 = {
        "device": 0,
        "start_ns": 0,
        "end_ns": 20_000_000,
        "busy_ns": 19_000_000,
        "exclusive_ns": {"0": 12_000_000, "1": 7_000_000},
        "name_stats": [[1, 8, 7_000_000, 100_000, 5_900_000]],
        "longest": [[2_000_000, 6_000_000, 1]],
    }
    gpu1 = {
        "device": 1,
        "start_ns": 0,
        "end_ns": 20_000_000,
        "busy_ns": 13_000_000,
        "exclusive_ns": {"0": 12_000_000, "1": 1_000_000},
        "big_gaps": [[2_000_000, 6_000_000, 0, 0], [9_000_000, 200_000, 0, 1]],
    }
    b = steps.step_breakdown(
        {"names": [ATTN, NCCL], "devices": [gpu0, gpu1]}, source="s", steps=4
    )
    assert b.devices[0].stalls_us() == (0.0, 6000.0)
    assert b.devices[1].stalls_us() == (6000.0, 0.0)
    # the window as captured: the stall is communication on GPU 0, idle on GPU 1
    assert b.share("communication") == pytest.approx((7 / 20 + 1 / 20) / 2, abs=1e-6)
    assert b.share("idle") == pytest.approx((1 / 20 + 7 / 20) / 2, abs=1e-6)
    assert b.step_ms == 5.0
    # without it both GPUs read 12 : 1 : 1 over 14 ms
    assert b.stall_ms == 6.0 and b.steady_step_ms == 3.5
    assert b.steady_shares["attention"] == pytest.approx(12 / 14, abs=1e-6)
    assert b.steady_shares["communication"] == pytest.approx(1 / 14, abs=1e-6)
    assert b.steady_shares["idle"] == pytest.approx(1 / 14, abs=1e-6)
    assert sum(b.steady_shares.values()) == pytest.approx(1.0, abs=1e-5)
    assert b.wait_share == pytest.approx(5.9 / 20 / 2, abs=1e-6)
    assert b.steady_wait_share == 0.0


def test_sample_segments_merge_neighbours_and_resolve_overlaps() -> None:
    doc = {
        "names": [ATTN, NCCL],
        "sample": {
            "start_ns": 0,
            "end_ns": 40_000,
            # attention twice 0.5 us apart, then an all-reduce overlapping it
            "rows": [[0, 0, 10_000], [0, 10_500, 20_000], [1, 15_000, 30_000]],
        },
    }
    length, segs = steps.sample_segments(doc)
    assert length == 40.0
    assert [(s.start_us, s.end_us, s.category) for s in segs] == [
        (0.0, 20.0, "attention"),
        (20.0, 30.0, "nccl"),
    ]


def test_a_cut_sample_ends_with_its_last_segment(monkeypatch) -> None:
    # past the segments kept, the rows cut off must not read as idle
    monkeypatch.setattr(steps, "SAMPLE_MAX_SEGMENTS", 2)
    doc = {
        "names": [ATTN, NCCL],
        "sample": {
            "start_ns": 0,
            "end_ns": 90_000,
            "rows": [[0, 0, 10_000], [1, 12_000, 20_000], [0, 22_000, 30_000]],
        },
    }
    length, segs = steps.sample_segments(doc)
    assert length == 20.0
    assert [s.category for s in segs] == ["attention", "nccl"]


def test_parse_timeline_refuses_other_versions_and_garbage() -> None:
    good = json.loads(steps.synthetic_timeline_json("decode", devices=2, steps=4))
    assert steps.parse_timeline(json.dumps(good)) == good
    assert steps.parse_timeline(json.dumps({**good, "version": 99})) is None
    assert steps.parse_timeline("{not json") is None
    assert steps.parse_timeline("[]") is None


def test_the_synthetic_capture_reads_like_a_decode_step() -> None:
    doc = steps.parse_timeline(steps.synthetic_timeline_json("decode", steps=10))
    assert doc is not None
    b = steps.step_breakdown(doc, source="sim", steps=10)
    assert len(b.devices) == 8
    assert sum(b.shares.values()) == pytest.approx(1.0, abs=1e-5)
    assert b.share("moe") > b.share("attention") > b.share("communication") > 0
    assert 0 < b.share("idle") < 0.1
    assert b.sample
    # GPU 1's first all-reduce waits 1.2 ms for a peer
    assert 0 < b.wait_share < b.share("communication")
    ((dev, call),) = steps.longest_calls(b)
    assert (dev.device, call.us, call.category) == (1, 1320.0, "nccl")
    # GPU 2's 6 ms stall is cut from the steady split
    assert b.stall_ms == pytest.approx(6.15 / 8, abs=1e-3)
    assert b.steady_shares["idle"] < b.share("idle")


def _synthetic_breakdown() -> steps.StepBreakdown:
    doc = steps.parse_timeline(steps.synthetic_timeline_json("decode", steps=10))
    assert doc is not None
    return steps.step_breakdown(doc, source="sim", steps=10)


def test_strip_bins_cover_the_sample_and_add_idle() -> None:
    b = _synthetic_breakdown()
    width, bins = steps.strip_bins(b)
    assert len(bins) == steps.STRIP_BINS
    assert width * len(bins) == pytest.approx(b.sample_us)
    for shares in bins:
        assert set(shares) == set(steps.GROUP_ORDER)
        busy = sum(v for g, v in shares.items() if g != "idle")
        assert shares["idle"] == pytest.approx(max(0.0, 1.0 - busy))
    busy_us = sum(min(s.end_us, b.sample_us) - max(s.start_us, 0.0) for s in b.sample)
    assert sum(1.0 - s["idle"] for s in bins) * width == pytest.approx(busy_us)
    assert steps.strip_bins(b.model_copy(update={"sample": ()})) == (0.0, [])


def test_zoom_segments_start_just_before_an_attention_kernel() -> None:
    b = _synthetic_breakdown()
    length, segs = steps.zoom_segments(b)
    assert 0 < length <= steps.ZOOM_US
    assert all(0.0 <= s.start_us < s.end_us <= length for s in segs)
    assert [s.start_us for s in segs] == sorted(s.start_us for s in segs)
    (first,) = [s for s in segs if s.start_us == pytest.approx(5.0)]
    assert first.category == "attention"
    assert steps.zoom_segments(b.model_copy(update={"sample": ()})) == (0.0, ())
