# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmlxProxyEngine against an aiohttp server that plays the Mac bridge."""

import asyncio
import json
from array import array
from types import SimpleNamespace

import pytest

try:
    from dynamo_omlx.proxy.engine import OmlxProxyEngine
    from omlx_fake_bridge import FakeBridge

    from dynamo.llm.exceptions import CannotConnect, InvalidArgument
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

from dynamo_omlx.bridge.event_log import NewBlock
from dynamo_omlx.bridge.service import BridgeError
from dynamo_omlx.protocol import ErrorCode
from omlx_fakes import FakeRequest, eventually

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(60),
]

B = 2


class Context:
    def __init__(self, ident: str) -> None:
        self.ident = ident
        self.stopped = False

    def id(self) -> str:
        return self.ident

    def is_stopped(self) -> bool:
        return self.stopped


async def drain(stream) -> list[dict]:
    return [chunk async for chunk in stream]


@pytest.fixture
async def env(tmp_path, monkeypatch):
    # RuntimeConfig.validate() exports these; record them so they are restored.
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "tcp")
    monkeypatch.setenv("DYN_TCP_TLS_INSECURE", "0")
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 64, "bos_token_id": 0})
    )
    async with FakeBridge(block_size=B) as bridge:
        argv = [
            "--bridge-url",
            bridge.url,
            "--model-path",
            str(tmp_path),
            "--discovery-backend",
            "file",
            "--stats-interval",
            "0.05",
        ]
        engine, worker_config = await OmlxProxyEngine.from_args(argv)
        try:
            config = await engine.start(worker_id=1)
            yield SimpleNamespace(
                bridge=bridge,
                engine=engine,
                worker_config=worker_config,
                config=config,
                model_path=str(tmp_path),
            )
        finally:
            await engine.cleanup()
            await engine.cleanup()  # cleanup() must be idempotent


async def test_registration(env):
    llm = env.config.llm
    assert (
        llm.context_length,
        llm.kv_cache_block_size,
        llm.max_num_seqs,
        llm.total_kv_blocks,
    ) == (64, B, 4, 128)
    worker = env.worker_config
    assert (worker.component, worker.endpoint, worker.model_name) == (
        "backend",
        "generate",
        env.model_path,
    )


async def test_generate_streams_tokens_and_usage(env):
    chunks = await drain(env.engine.generate({"token_ids": [1, 2, 3]}, Context("r")))
    assert chunks[0] == {"token_ids": [7, 8], "index": 0}
    assert chunks[-1]["finish_reason"] == "stop"
    usage = chunks[-1]["completion_usage"]
    assert usage["prompt_tokens_details"] == {"cached_tokens": 2}


async def test_unsupported_requests_are_rejected(env):
    request = {"token_ids": [1], "output_options": {"logprobs": 1}}
    with pytest.raises(InvalidArgument, match="logprobs"):
        await drain(env.engine.generate(request, Context("r")))


async def test_abort_ends_the_stream_as_cancelled(env):
    env.bridge.request = FakeRequest([], gate=asyncio.Event())
    context = Context("r")
    stream = env.engine.generate({"token_ids": [4, 5]}, context)
    pending = asyncio.ensure_future(anext(stream))
    await eventually(
        lambda: env.bridge.generate_service is not None
        and env.bridge.generate_service.active_count == 1
    )
    context.stopped = True
    await env.engine.abort(context)
    assert (await asyncio.wait_for(pending, 5))["finish_reason"] == "cancelled"
    assert await drain(stream) == []


async def test_error_end_after_client_stop_is_cancelled(env):
    # The client left during prefill and the engine reported the abort as an error.
    env.bridge.request = FakeRequest(
        [BridgeError(ErrorCode.ENGINE_ERROR, "Request aborted")]
    )
    context = Context("r")
    context.stopped = True
    chunks = await drain(env.engine.generate({"token_ids": [4]}, context))
    assert [chunk["finish_reason"] for chunk in chunks] == ["cancelled"]


async def test_pre_stream_errors_are_migratable(env):
    env.bridge.generate_error = ("model_not_ready", "loading")
    with pytest.raises(CannotConnect):
        await drain(env.engine.generate({"token_ids": [1]}, Context("r")))


async def test_kv_events_reach_the_publisher(env):
    events: list[dict] = []
    publisher = SimpleNamespace(publish_batch=events.extend)
    [source] = await env.engine.kv_event_sources()
    source.on_ready(publisher)
    blocks = [
        NewBlock(bid=b"a", tokens=array("I", [1, 2])),
        NewBlock(bid=b"b", tokens=array("I", [3, 4])),
    ]
    env.bridge.log.store_chain(None, blocks)
    await eventually(lambda: len(events) >= 1)
    stored = events[0]
    assert (
        stored["type"],
        stored["token_ids"],
        stored["num_block_tokens"],
        stored["parent_hash"],
    ) == ("stored", [1, 2, 3, 4], [B, B], None)
    env.bridge.log.remove_bids([b"b"])
    removed = {"type": "removed", "block_hashes": [stored["block_hashes"][1]]}
    await eventually(lambda: events[-1] == removed)


async def test_metrics_and_component_snapshots(env):
    snapshots: list = []
    env.engine.attach_snapshot_publisher(
        SimpleNamespace(publish=lambda rank, snap: snapshots.append((rank, snap)))
    )
    await eventually(lambda: snapshots)
    assert snapshots[-1][0] == 0 and snapshots[-1][1].kv_total_blocks == 128
    callbacks: dict = {}
    metrics = SimpleNamespace(
        auto_labels=[("worker_id", "1")],
        register_prometheus_expfmt_callback=lambda cb: callbacks.update(text=cb),
        register_prometheus_typed_callback=lambda cb: callbacks.update(typed=cb),
    )
    await env.engine.register_prometheus(metrics)
    text = callbacks["text"]()
    for name in (
        "omlx_bridge_up",
        "omlx_running_requests",
        "omlx_proxy_requests_total",
    ):
        assert name in text
