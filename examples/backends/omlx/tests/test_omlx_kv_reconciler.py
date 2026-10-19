# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from array import array

import pytest
from dynamo_omlx.bridge.event_log import KvEventLog, NewBlock
from dynamo_omlx.protocol import KvBlock, decode_frame, parse_blocks
from dynamo_omlx.proxy.kv_reconciler import (
    KvReconciler,
    RemoveOp,
    ResyncRequired,
    StoreOp,
)
from omlx_fakes import Router, chain

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

B = 2


def tokens(*values):
    return array("I", values)


def block(ext, parent, value=0):
    return KvBlock(ext=ext, parent=parent, tokens=tokens(value, value))


def store_event(seq, blocks):
    return {
        "t": "ev",
        "seq": seq,
        "op": "store",
        "blocks": [b.to_wire() for b in blocks],
    }


def test_snapshot_announces_parent_first_and_skips_orphans():
    rec = KvReconciler(B)
    ops = rec.apply_snapshot(
        "inc", 5, [block(1, None), block(2, 1), block(3, 2), block(9, 8), block(4, 1)]
    )
    assert ops == [
        StoreOp(None, (1, 2, 3), (tokens(0, 0),) * 3),
        StoreOp(1, (4,), (tokens(0, 0),)),
    ]
    assert rec.announced() == [1, 2, 3, 4] and rec.skipped_blocks == 1
    assert (rec.incarnation, rec.last_seq) == ("inc", 5)


def test_snapshot_diff_only_emits_changes():
    rec = KvReconciler(B)
    rec.apply_snapshot("inc", 1, [block(1, None), block(2, 1), block(3, 2)])
    ops = rec.apply_snapshot("inc2", 7, [block(1, None), block(2, 1), block(5, 1)])
    assert ops == [RemoveOp((3,)), StoreOp(1, (5,), (tokens(0, 0),))]
    assert (
        rec.apply_snapshot("inc2", 7, [block(1, None), block(2, 1), block(5, 1)]) == []
    )


def test_events_apply_in_sequence_and_skip_replays():
    rec = KvReconciler(B)
    rec.apply_snapshot("inc", 3, [])
    assert rec.apply_event(store_event(4, [block(1, None), block(2, 1)])) == [
        StoreOp(None, (1, 2), (tokens(0, 0),) * 2)
    ]
    assert rec.apply_event(store_event(4, [block(1, None)])) == []  # replayed
    with pytest.raises(ResyncRequired, match="gap"):
        rec.apply_event(store_event(6, [block(3, 2)]))
    assert rec.apply_event({"t": "ev", "seq": 5, "op": "remove", "hashes": [1]}) == [
        RemoveOp((2, 1))
    ]
    assert rec.apply_event({"t": "ev", "seq": 6, "op": "remove", "hashes": [1]}) == []
    assert rec.last_seq == 6 and len(rec) == 0


def test_store_of_known_blocks_is_deduplicated():
    rec = KvReconciler(B)
    rec.apply_snapshot("inc", 0, [block(1, None)])
    assert rec.apply_event(store_event(1, [block(1, None), block(2, 1)])) == [
        StoreOp(1, (2,), (tokens(0, 0),))
    ]


def test_orphans_and_unknown_ops_require_a_resync():
    rec = KvReconciler(B)
    rec.apply_snapshot("inc", 0, [])
    with pytest.raises(ResyncRequired, match="unknown parent"):
        rec.apply_event(store_event(1, [block(2, 1)]))
    with pytest.raises(ResyncRequired, match="unknown KV op"):
        rec.apply_event({"t": "ev", "seq": 1, "op": "compact"})


def test_clear_withdraws_everything_leaf_first():
    rec = KvReconciler(B)
    rec.apply_snapshot("inc", 0, [block(1, None), block(2, 1), block(3, None)])
    assert rec.apply_event({"t": "ev", "seq": 1, "op": "clear"}) == [
        RemoveOp((3, 2, 1))
    ]
    assert len(rec) == 0 and rec.reset() == []


def test_long_chains_split_at_the_frame_limit():
    rec = KvReconciler(B)
    chain = [block(1, None)] + [block(i, i - 1) for i in range(2, 21)]
    ops = rec.apply_snapshot("inc", 0, chain)
    assert [len(op.hashes) for op in ops] == [16, 4] and ops[1].parent == 16


def test_randomized_runs_mirror_the_bridge_and_keep_router_invariants():
    """Drive a real KvEventLog with random stores/evictions/clears and random
    reconnects, and check the reconciler against the log and router rules."""
    rng = random.Random(1234)
    log = KvEventLog(ring_max_events=8)
    log.set_block_size(B)
    rec = KvReconciler(B)
    router = Router()

    def resync():
        snap = log.snapshot()
        router.submit(rec.apply_snapshot(snap.incarnation, snap.as_of, snap.blocks))

    def catch_up():
        reason, frames = log.read_after(rec.last_seq, limit=10_000)
        if reason is not None:
            resync()
            return
        for frame in frames:
            try:
                router.submit(rec.apply_event(decode_frame(frame)))
            except ResyncRequired:
                resync()
                return

    resync()
    for step in range(3000):
        action = rng.random()
        if action < 0.55:
            log.store_chain(None, chain(rng.randrange(6), rng.randrange(1, 6), B))
        elif action < 0.9:
            live = log.live_bids()
            if live:
                log.remove_bids([rng.choice(live)])
        elif action < 0.92:
            log.clear()
        if rng.random() < 0.3:  # the proxy is connected and reads the stream
            catch_up()
        if step % 97 == 0:
            catch_up()
            assert router.blocks == {b.ext for b in log.snapshot().blocks}
    catch_up()
    expected = {b.ext for b in log.snapshot().blocks}
    assert router.blocks == expected == set(rec.announced())


def test_store_events_from_the_log_parse_into_reconciler_ops():
    log = KvEventLog()
    log.set_block_size(B)
    log.store_chain(
        None,
        [
            NewBlock(bid=b"a", tokens=tokens(1, 2)),
            NewBlock(bid=b"b", tokens=tokens(3, 4)),
        ],
    )
    [frame] = log.read_after(0)[1]
    event = decode_frame(frame)
    blocks = parse_blocks(event["blocks"], B)
    rec = KvReconciler(B)
    rec.apply_snapshot(log.incarnation, 0, [])
    [op] = rec.apply_event(event)
    assert op == StoreOp(
        None, tuple(b.ext for b in blocks), (tokens(1, 2), tokens(3, 4))
    )
