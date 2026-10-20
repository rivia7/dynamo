# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.kv_tracker import KvTracker
from dynamo_omlx.protocol import KvState, decode_frame
from omlx_fakes import omlx_hash

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

B = 4
MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"


def verify(parent, tokens, bid, model_name, extra_keys) -> bool:
    return omlx_hash(parent, list(tokens), model_name, extra_keys) == bid


@dataclass
class FakeBlock:
    block_hash: bytes | None
    token_count: int


@dataclass
class FakePagedCache:
    block_size: int = B
    model_name: str = MODEL
    allocated_blocks: dict[int, FakeBlock] = field(default_factory=dict)


@dataclass
class FakeTable:
    block_ids: list[int]


class FakeSsd:
    def __init__(self) -> None:
        self.blocks: set[bytes] = set()

    def has_block(self, bid: bytes) -> bool:
        return bid in self.blocks


class Harness:
    """A request's stored sequence as oMLX would hand it to store_cache."""

    def __init__(self) -> None:
        self.paged = FakePagedCache()
        self.ssd = FakeSsd()
        self.log = KvEventLog()
        self.ssd_available = True
        self.tracker = KvTracker(
            self.log,
            ssd_provider=self.provider,
            verify_block=verify,
            audit_interval_s=3600,
        )
        self._next_id = 0

    def provider(self):
        return self.ssd if self.ssd_available else None

    def store(
        self,
        tokens: list[int],
        *,
        persist: bool = True,
        tail: int = 0,
        corrupt_at: int | None = None,
    ) -> list[bytes]:
        """Lay out ``tokens`` as full blocks (+ an optional partial tail block) and call on_store."""
        bids, block_ids, parent = [], [], None
        full = len(tokens) // B
        for i in range(full):
            block_tokens = tokens[i * B : (i + 1) * B]
            bid = omlx_hash(parent, block_tokens, MODEL)
            if corrupt_at == i:
                bid = b"x" * 32
            self._next_id += 1
            self.paged.allocated_blocks[self._next_id] = FakeBlock(bid, B)
            block_ids.append(self._next_id)
            bids.append(bid)
            if persist:
                self.ssd.blocks.add(bid)
            parent = bid
        if tail:
            self._next_id += 1
            self.paged.allocated_blocks[self._next_id] = FakeBlock(b"tail", tail)
            block_ids.append(self._next_id)
        self.tracker.on_store(self.paged, FakeTable(block_ids), tokens + [0] * tail)
        self.tracker.process_pending()
        return bids


def ops(log: KvEventLog) -> list[str]:
    return [decode_frame(frame)["op"] for frame in log.read_after(0, limit=1000)[1]]


def test_announces_full_persisted_blocks_only():
    h = Harness()
    bids = h.store(list(range(3 * B)), tail=2)
    assert [h.log.contains(bid) for bid in bids] == [True, True, True]
    assert len(h.log) == 3 and h.log.block_size == B


def test_unpersisted_block_ends_the_chain():
    h = Harness()
    bids = h.store(list(range(3 * B)), persist=False)
    h.ssd.blocks.add(bids[0])
    h.store(list(range(3 * B)), persist=False)
    assert len(h.log) == 1 and h.log.contains(bids[0])


def test_second_request_extends_the_announced_prefix():
    h = Harness()
    h.store(list(range(2 * B)))
    h.store(list(range(4 * B)))
    assert len(h.log) == 4
    assert ops(h.log) == ["store", "store"]


def test_exact_terminal_block_is_not_announced():
    h = Harness()
    tokens = list(range(2 * B))
    bid0 = omlx_hash(None, tokens[:B], MODEL)
    bid1 = omlx_hash(bid0, tokens[B:], MODEL)
    h.paged.allocated_blocks.update({1: FakeBlock(bid0, B), 2: FakeBlock(bid1, B)})
    h.ssd.blocks.update({bid0, bid1})
    h.tracker.on_store(h.paged, FakeTable([1, 2]), tokens, exclude_last=True)
    h.tracker.process_pending()
    assert h.log.contains(bid0) and not h.log.contains(bid1)


def image_chain(h: Harness, keys) -> tuple[list[int], list[bytes]]:
    """Three blocks whose oMLX hashes carry ``keys[i]``, stored and persisted."""
    tokens = list(range(3 * B))
    bids, parent = [], None
    for i, key in enumerate(keys):
        parent = omlx_hash(parent, tokens[i * B : (i + 1) * B], MODEL, key)
        h.paged.allocated_blocks[10 + i] = FakeBlock(parent, B)
        h.ssd.blocks.add(parent)
        bids.append(parent)
    return tokens, bids


def test_image_blocks_verify_with_their_keys_and_announce_routing_tokens():
    h = Harness()
    keys = [None, ("img",), ("img",)]
    tokens, bids = image_chain(h, keys)
    routing = tokens[:B] + [1_000_007] * B + tokens[2 * B :]
    h.tracker.on_store(
        h.paged,
        FakeTable([10, 11, 12]),
        tokens,
        routing_tokens=routing,
        extra_keys=keys,
    )
    h.tracker.process_pending()
    assert len(h.log) == 3 and h.tracker.stats().get("faults_hash_mismatch") is None
    announced = [list(block.tokens) for block in h.log.snapshot().blocks]
    assert announced == [routing[i * B : (i + 1) * B] for i in range(3)]


def test_image_blocks_need_their_keys_and_respect_max_blocks():
    h = Harness()
    keys = [None, ("img",), ("img",)]
    tokens, _ = image_chain(h, keys)
    h.tracker.on_store(h.paged, FakeTable([10, 11, 12]), tokens)  # keys left out
    h.tracker.process_pending()
    assert len(h.log) == 1 and h.tracker.stats()["faults_hash_mismatch"] == 1
    h.tracker.on_store(
        h.paged, FakeTable([10, 11, 12]), tokens, extra_keys=keys, max_blocks=2
    )
    h.tracker.process_pending()
    assert len(h.log) == 2


def test_hash_mismatch_stops_the_chain_and_trips_the_breaker():
    h = Harness()
    for _ in range(2):
        h.store(list(range(3 * B)), corrupt_at=1)
    assert h.tracker.state == KvState.LIVE and len(h.log) == 1
    h.store(list(range(3 * B)), corrupt_at=1)
    assert h.tracker.state == KvState.DEGRADED
    assert len(h.log) == 0 and ops(h.log)[-1] == "clear"
    h.store(list(range(8, 8 + 2 * B)))
    assert len(h.log) == 0  # nothing is announced while degraded


def test_removal_waits_until_no_tier_holds_the_block():
    h = Harness()
    bids = h.store(list(range(2 * B)))
    h.tracker.on_removed([bids[1]], "evicted")  # still in the hot cache
    h.tracker.process_pending()
    assert len(h.log) == 2
    h.ssd.blocks.discard(bids[1])
    h.tracker.on_removed([bids[1], b"unrelated"], "evicted")
    h.tracker.process_pending()
    assert len(h.log) == 1


def test_evicting_a_root_withdraws_the_chain():
    h = Harness()
    bids = h.store(list(range(3 * B)))
    h.ssd.blocks.discard(bids[0])
    h.tracker.on_removed([bids[0]], "deleted")
    h.tracker.process_pending()
    assert len(h.log) == 0


def test_audit_needs_two_misses_unless_the_cache_was_cleared():
    h = Harness()
    bids = h.store(list(range(2 * B)))
    h.ssd.blocks.discard(bids[1])
    h.tracker.request_audit("periodic")
    h.tracker.process_pending()
    assert len(h.log) == 2
    h.tracker.request_audit("periodic")
    h.tracker.process_pending()
    assert len(h.log) == 1
    h.ssd.blocks.clear()
    h.tracker.on_ssd_cleared()
    h.tracker.process_pending()
    assert len(h.log) == 0


def test_unloaded_model_skips_stores_and_audits_but_trusts_removals():
    h = Harness()
    bids = h.store(list(range(2 * B)))
    h.ssd_available = False
    h.store(list(range(8, 8 + B)))
    h.tracker.request_audit("periodic")
    h.tracker.process_pending()
    assert len(h.log) == 2
    h.tracker.on_removed([bids[1]], "deleted")
    h.tracker.process_pending()
    assert len(h.log) == 1


def test_hook_errors_trip_the_breaker():
    h = Harness()
    h.store(list(range(B)))
    for _ in range(5):
        h.tracker.record_hook_error()
    assert h.tracker.state == KvState.DEGRADED and len(h.log) == 0
    assert h.tracker.stats()["faults_hook_error"] == 5


@pytest.mark.timeout(10)
def test_worker_thread_processes_jobs():
    h = Harness()
    h.tracker.start()
    tokens = list(range(2 * B))
    bid0 = omlx_hash(None, tokens[:B], MODEL)
    h.paged.allocated_blocks[1] = FakeBlock(bid0, B)
    h.ssd.blocks.add(bid0)
    h.tracker.on_store(h.paged, FakeTable([1]), tokens)
    # Jobs run in FIFO order, so the stop sentinel queued by stop() runs after
    # the store job and the join guarantees it was processed.
    h.tracker.stop(timeout=5)
    assert h.log.contains(bid0)
