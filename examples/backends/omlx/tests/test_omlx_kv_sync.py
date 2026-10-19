# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end KV pipeline without HTTP: bridge log and stream service on one
side, KvSync and the reconciler on the other, a recording router at the end."""

import asyncio
from contextlib import asynccontextmanager, suppress

import pytest
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.service import KvStreamService
from dynamo_omlx.protocol import KvState, decode_frame
from dynamo_omlx.proxy.kv_reconciler import KvReconciler
from dynamo_omlx.proxy.kv_sync import KvSync, KvSyncFatal, SyncState
from omlx_fakes import Router, chain, eventually

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

B = 2


class Bridge:
    """One bridge incarnation: its log and KV stream service."""

    def __init__(self, block_size: int = B, ring_max_events: int = 1000) -> None:
        self.log = KvEventLog(ring_max_events=ring_max_events)
        self.log.set_block_size(block_size)
        self.log.bind_loop(asyncio.get_running_loop())
        self.state = KvState.LIVE
        self.block_size = block_size
        self.service = KvStreamService(self.log, lambda: self.state, heartbeat_s=0.02)

    def store(self, prefix: int, length: int) -> None:
        self.log.store_chain(None, chain(prefix, length, self.block_size))

    def exts(self) -> set[int]:
        return {block.ext for block in self.log.snapshot().blocks}


class InProcessClient:
    """KvClient over a Bridge that can drop connections like a flaky network."""

    def __init__(self, bridge: Bridge) -> None:
        self.bridge = bridge
        self.connected = True
        self.snapshots = 0
        self.streams = 0

    @asynccontextmanager
    async def kv_snapshot(self):
        if not self.connected:
            raise ConnectionError("network down")
        self.snapshots += 1

        async def frames():
            async for raw in self.bridge.service.snapshot(self.bridge.block_size):
                yield decode_frame(raw)

        yield frames()

    @asynccontextmanager
    async def kv_stream(self, incarnation, after):
        if not self.connected:
            raise ConnectionError("network down")
        self.streams += 1
        stream = self.bridge.service.stream(incarnation, after, self.bridge.block_size)

        async def frames():
            async for raw in stream:
                if not self.connected:
                    raise ConnectionError("connection dropped")
                yield decode_frame(raw)

        try:
            yield frames()
        finally:
            await stream.aclose()


@asynccontextmanager
async def running_sync(client, router):
    sync = KvSync(
        client,
        KvReconciler(B),
        router.submit,
        min_stall_timeout_s=0.5,
        backoff_initial_s=0.01,
        backoff_max_s=0.05,
        disabled_poll_s=0.05,
    )
    task = asyncio.ensure_future(sync.run())
    try:
        yield sync, task
    finally:
        sync.stop()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_follows_events_and_resumes_after_a_drop_without_a_snapshot():
    bridge = Bridge()
    bridge.store(1, 3)
    client, router = InProcessClient(bridge), Router()
    async with running_sync(client, router) as (sync, _):
        await eventually(
            lambda: router.blocks == bridge.exts() and sync.state == SyncState.LIVE
        )
        bridge.store(1, 5)
        bridge.log.remove_bids([bytes([1, 2])])
        await eventually(lambda: router.blocks == bridge.exts())

        client.connected = False
        bridge.store(
            2, 2
        )  # delivered on the broken connection's next frame, which fails
        await eventually(lambda: sync.counters["reconnects"] >= 1)
        client.connected = True
        bridge.store(3, 1)
        await eventually(lambda: router.blocks == bridge.exts())
        assert client.snapshots == 1  # resumed from the replay ring


async def test_ring_overflow_triggers_a_snapshot_diff():
    bridge = Bridge(ring_max_events=2)
    bridge.store(1, 2)
    client, router = InProcessClient(bridge), Router()
    async with running_sync(client, router) as (sync, _):
        await eventually(
            lambda: router.blocks == bridge.exts() and sync.state == SyncState.LIVE
        )
        client.connected = False
        await eventually(lambda: sync.counters["reconnects"] >= 1)
        for prefix in range(3, 8):
            bridge.store(prefix, 1)
        bridge.log.remove_bids([bytes([1, 0])])
        client.connected = True
        await eventually(lambda: router.blocks == bridge.exts())
        assert client.snapshots == 2 and sync.counters["resyncs"] >= 1


async def test_restarted_bridge_is_reconciled_by_diff():
    old = Bridge()
    old.store(1, 2)
    old.store(2, 2)
    client, router = InProcessClient(old), Router()
    async with running_sync(client, router):
        await eventually(lambda: router.blocks == old.exts())
        new = Bridge()  # new incarnation; only chain 1 survived on the SSD
        new.store(1, 2)
        new.store(4, 1)
        client.bridge = new
        old.service.begin_shutdown()
        await eventually(lambda: router.blocks == new.exts())


async def test_counters_exist_before_the_first_increment():
    sync = KvSync(InProcessClient(Bridge()), KvReconciler(B), Router().submit)
    assert set(sync.counters) == {
        "events",
        "snapshots",
        "resyncs",
        "reconnects",
        "protocol_errors",
    }
    assert not any(sync.counters.values())


async def test_block_size_drift_is_fatal():
    bridge = Bridge(block_size=4)
    client, router = InProcessClient(bridge), Router()
    sync = KvSync(client, KvReconciler(B), router.submit, backoff_initial_s=0.01)
    with pytest.raises(KvSyncFatal, match="block size"):
        await asyncio.wait_for(sync.run(), 5)


async def test_disabled_bridge_withdraws_blocks_until_enabled():
    bridge = Bridge()
    bridge.store(1, 2)
    client, router = InProcessClient(bridge), Router()
    async with running_sync(client, router) as (sync, _):
        await eventually(lambda: router.blocks == bridge.exts())
        bridge.state = KvState.DISABLED
        bridge.store(
            5, 1
        )  # wakes the stream so it notices nothing new; the reset comes on reconnect
        client.connected = False
        await eventually(lambda: sync.counters["reconnects"] >= 1)
        client.connected = True
        await eventually(lambda: sync.state == SyncState.DISABLED and not router.blocks)
        bridge.state = KvState.LIVE
        await eventually(
            lambda: router.blocks == bridge.exts() and sync.state == SyncState.LIVE
        )
