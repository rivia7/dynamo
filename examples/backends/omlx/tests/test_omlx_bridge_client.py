# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BridgeClient against an aiohttp server that serves the real bridge services."""

import asyncio
from array import array
from contextlib import asynccontextmanager, suppress

import pytest

try:
    from omlx_fake_bridge import FakeBridge
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

from dynamo_omlx.bridge.event_log import NewBlock
from dynamo_omlx.protocol import PROTOCOL_VERSION, ProtocolError, error_payload
from dynamo_omlx.proxy import bridge_client
from dynamo_omlx.proxy.bridge_client import (
    BridgeClient,
    BridgeRequestError,
    BridgeUnavailable,
)
from dynamo_omlx.proxy.kv_reconciler import KvReconciler
from dynamo_omlx.proxy.kv_sync import KvSync, SyncState
from omlx_fakes import Router, eventually

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

API_KEY = "secret"


@asynccontextmanager
async def connected(url: str, api_key: str = API_KEY, **kwargs):
    client = BridgeClient(url, api_key, **kwargs)
    await client.start()
    try:
        yield client
    finally:
        await client.close()


async def test_info_and_error_mapping():
    async with FakeBridge(api_key=API_KEY) as bridge:
        async with connected(bridge.url) as client:
            assert (await client.info())["v"] == PROTOCOL_VERSION
            bridge.info_error = (503, {"error": error_payload("overloaded", "busy")})
            with pytest.raises(BridgeRequestError) as info:
                await client.info()
            assert (info.value.code, info.value.retryable, info.value.status) == (
                "overloaded",
                True,
                503,
            )
            bridge.info_error = (500, "boom")
            with pytest.raises(BridgeUnavailable):
                await client.info()
        bridge.info_error = None
        async with connected(bridge.url, api_key="wrong") as client:
            with pytest.raises(BridgeRequestError) as info:
                await client.info()
            assert info.value.code == "unauthorized" and not info.value.retryable


async def test_unreachable_bridge():
    async with FakeBridge() as bridge:
        url = bridge.url
    async with connected(url, connect_timeout_s=1) as client:
        with pytest.raises(BridgeUnavailable):
            await client.info()


async def test_generate_stream():
    async with FakeBridge() as bridge:
        async with connected(bridge.url) as client:
            body = {"v": PROTOCOL_VERSION, "request_id": "r", "token_ids": [1, 2]}
            async with client.generate(body, sock_read=5) as frames:
                kinds = [frame["t"] async for frame in frames]
    assert kinds == ["start", "tok", "end"]


async def test_snapshot_lines_larger_than_aiohttp_defaults():
    block_size = 8192  # 16 blocks of 8192 tokens is a ~700 KB NDJSON line
    async with FakeBridge(block_size=block_size) as bridge:
        blocks = [
            NewBlock(bid=bytes([i]), tokens=array("I", [i] * block_size))
            for i in range(16)
        ]
        bridge.log.store_chain(None, blocks)
        async with connected(bridge.url) as client:
            async with client.kv_snapshot() as frames:
                snapshot = [frame async for frame in frames]
    assert [f["t"] for f in snapshot] == ["snap_begin", "snap", "snap_end"]
    assert len(snapshot[1]["blocks"]) == 16


async def test_oversized_line_is_a_protocol_error(monkeypatch):
    monkeypatch.setattr(bridge_client, "_READ_BUFSIZE", 1024)
    async with FakeBridge(block_size=4096) as bridge:
        block = NewBlock(bid=b"a", tokens=array("I", [1] * 4096))
        bridge.log.store_chain(None, [block])
        async with connected(bridge.url) as client:
            with pytest.raises(ProtocolError, match="line over"):
                async with client.kv_snapshot() as frames:
                    [frame async for frame in frames]


async def test_kv_sync_over_http_converges():
    async with FakeBridge() as bridge:
        bridge.log.store_chain(None, [NewBlock(bid=b"a", tokens=array("I", [1, 2]))])
        router = Router()
        async with connected(bridge.url) as client:
            sync = KvSync(
                client,
                KvReconciler(2),
                router.submit,
                backoff_initial_s=0.01,
                backoff_max_s=0.05,
            )
            task = asyncio.ensure_future(sync.run())
            try:
                bridge.log.store_chain(
                    b"a", [NewBlock(bid=b"b", tokens=array("I", [3, 4]))]
                )
                expected = {block.ext for block in bridge.log.snapshot().blocks}
                await eventually(
                    lambda: router.blocks == expected and sync.state == SyncState.LIVE
                )
            finally:
                sync.stop()
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
