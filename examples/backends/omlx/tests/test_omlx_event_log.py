# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from array import array

import pytest
from dynamo_omlx.bridge.event_log import KvEventLog, NewBlock
from dynamo_omlx.protocol import (
    MAX_BLOCKS_PER_FRAME,
    ResetReason,
    block_hash,
    decode_frame,
    parse_blocks,
)
from omlx_fakes import chain

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

B = 4


def events(log: KvEventLog, after: int = 0) -> list[dict]:
    reason, frames = log.read_after(after, limit=10_000)
    assert reason is None
    return [decode_frame(frame) for frame in frames]


@pytest.fixture
def log() -> KvEventLog:
    kv_log = KvEventLog()
    kv_log.set_block_size(B)
    return kv_log


def test_store_chain_announces_parent_first_with_protocol_hashes(log):
    assert log.store_chain(None, chain(1, 3, B)) == 3
    [event] = events(log)
    assert event["seq"] == 1 and event["op"] == "store"
    blocks = parse_blocks(event["blocks"], B)
    expected_parent = None
    for i, block in enumerate(blocks):
        assert block.parent == expected_parent
        assert block.ext == block_hash(B, expected_parent, [1, i, 0, 0])
        expected_parent = block.ext
    assert [b.ext for b in log.snapshot().blocks] == [b.ext for b in blocks]


def test_store_chain_skips_live_prefix_and_extends(log):
    log.store_chain(None, chain(1, 2, B))
    assert log.store_chain(None, chain(1, 2, B)) == 0
    assert log.store_chain(None, chain(1, 4, B)) == 2
    second = events(log)[1]
    first_new = parse_blocks(second["blocks"], B)[0]
    assert first_new.parent == log.snapshot().blocks[1].ext
    assert log.last_seq == 2


def test_store_chain_requires_live_parent(log):
    assert log.store_chain(b"missing", chain(1, 1, B)) == 0
    assert log.last_seq == 0


def test_long_chains_split_into_frames(log):
    log.store_chain(None, chain(1, MAX_BLOCKS_PER_FRAME + 3, B))
    frames = events(log)
    assert [len(f["blocks"]) for f in frames] == [MAX_BLOCKS_PER_FRAME, 3]
    tail = parse_blocks(frames[1]["blocks"], B)[0]
    assert tail.parent == parse_blocks(frames[0]["blocks"], B)[-1].ext


def test_remove_cascades_to_descendants_leaf_first(log):
    log.store_chain(None, chain(1, 3, B))
    exts = [b.ext for b in log.snapshot().blocks]
    branch = [NewBlock(bid=b"branch", tokens=array("I", [9] * B))]
    log.store_chain(bytes([1, 0]), branch)
    assert log.remove_bids([bytes([1, 1])]) == 2
    removed = events(log)[-1]
    assert removed["op"] == "remove" and removed["hashes"] == [exts[2], exts[1]]
    assert [b.ext for b in log.snapshot().blocks] == [
        exts[0],
        block_hash(B, exts[0], [9] * B),
    ]
    assert log.remove_bids([b"unknown"]) == 0


def test_clear_and_block_size_change(log):
    log.store_chain(None, chain(1, 2, B))
    log.set_block_size(B)  # unchanged: nothing happens
    assert len(log) == 2
    log.set_block_size(2 * B)
    assert len(log) == 0 and events(log)[-1]["op"] == "clear"
    log.clear()  # already empty: no event
    assert log.last_seq == 2


def test_read_after_reports_ahead_and_floor():
    log = KvEventLog(ring_max_events=2)
    log.set_block_size(B)
    for prefix in range(4):
        log.store_chain(None, chain(prefix, 1, B))
    assert log.floor == 3
    assert log.read_after(5)[0] == ResetReason.AHEAD
    assert log.read_after(1)[0] == ResetReason.FLOOR
    reason, frames = log.read_after(2)
    assert reason is None and [decode_frame(f)["seq"] for f in frames] == [3, 4]
    assert log.read_after(4) == (None, [])


def test_ring_trims_by_bytes():
    log = KvEventLog(ring_max_bytes=1)
    log.set_block_size(B)
    log.store_chain(None, chain(1, 1, B))
    log.store_chain(None, chain(2, 1, B))
    assert log.stats()["ring_events"] == 1  # the newest frame always stays
    assert log.floor == 2


@pytest.mark.timeout(10)
async def test_wait_beyond_wakes_on_commit_from_another_thread(log):
    log.bind_loop(asyncio.get_running_loop())
    assert await log.wait_beyond(0, timeout=0.01) is False
    timer = threading.Timer(0.05, lambda: log.store_chain(None, chain(1, 1, B)))
    timer.start()
    assert await log.wait_beyond(0, timeout=5) is True
    timer.join(timeout=5)


def test_concurrent_producers_keep_seq_and_table_consistent(log):
    def produce(prefix: int) -> None:
        for i in range(50):
            block = NewBlock(
                bid=bytes([prefix, i]), tokens=array("I", [prefix, i, 1, 1])
            )
            log.store_chain(None, [block])
            if i % 5 == 4:
                log.remove_bids([bytes([prefix, i - 1])])

    threads = [threading.Thread(target=produce, args=(t,)) for t in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    frames = events(log)
    assert [f["seq"] for f in frames] == list(range(1, log.last_seq + 1))
    assert len(log) == 8 * 40
