# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.service import (
    BridgeError,
    GenerateService,
    GenerateSpec,
    KvStreamService,
    ModelState,
)
from dynamo_omlx.protocol import PROTOCOL_VERSION, ErrorCode, KvState, decode_frame
from omlx_fakes import FakeFacade, FakeRequest, chain, make_runtime, output

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


async def collect(stream) -> list[dict]:
    return [decode_frame(frame) async for frame in stream]


def spec(**overrides):
    return GenerateSpec(request_id="r1", token_ids=[1, 2, 3], **overrides)


async def test_generate_streams_tokens_and_usage():
    request = FakeRequest(
        [
            output([10]),
            output([11, 12]),
            output([], finished=True, reason="stop", completion=3),
        ]
    )
    frames = await collect(
        GenerateService(FakeFacade(request), "inc", heartbeat_s=5).stream(spec())
    )
    assert [f["t"] for f in frames] == ["start", "tok", "tok", "end"]
    assert frames[0]["inc"] == "inc" and frames[1]["ids"] == [10]
    assert frames[-1] == {
        "t": "end",
        "finish": "stop",
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "cached_tokens": 2},
        "error": None,
    }


async def test_generate_sends_heartbeats_while_the_engine_is_busy():
    gate = asyncio.Event()
    request = FakeRequest([output([], finished=True, reason="length")], gate=gate)
    service = GenerateService(FakeFacade(request), "inc", heartbeat_s=0.01)
    frames = []
    async for frame in service.stream(spec()):
        frames.append(decode_frame(frame))
        if len(frames) == 3:
            gate.set()
    assert [f["t"] for f in frames[:3]] == ["start", "hb", "hb"]
    assert frames[-1]["finish"] == "length"


async def test_client_cancel_aborts_the_engine_request():
    request = FakeRequest([], gate=asyncio.Event())
    service = GenerateService(FakeFacade(request), "inc", heartbeat_s=5)
    stream = service.stream(spec())
    assert decode_frame(await anext(stream))["t"] == "start"
    pending = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    assert service.cancel("r1") is True
    end = decode_frame(await pending)
    assert end["finish"] == "abort" and request.abort_calls == 1
    assert [frame async for frame in stream] == []
    assert service.cancel("r1") is False  # finished requests are forgotten


async def test_engine_initiated_abort_is_a_retryable_error():
    request = FakeRequest([output([], finished=True, reason="abort")])
    frames = await collect(
        GenerateService(FakeFacade(request), "inc", heartbeat_s=5).stream(spec())
    )
    assert frames[-1]["finish"] == "error"
    assert (
        frames[-1]["error"]["code"] == ErrorCode.ABORTED_BY_ENGINE
        and frames[-1]["error"]["retryable"]
    )


async def test_engine_errors_end_the_stream_with_their_code():
    request = FakeRequest(
        [output([7]), BridgeError(ErrorCode.OVERLOADED, "memory pressure")]
    )
    frames = await collect(
        GenerateService(FakeFacade(request), "inc", heartbeat_s=5).stream(spec())
    )
    assert frames[-1]["error"] == {
        "code": "overloaded",
        "message": "memory pressure",
        "retryable": True,
    }


async def test_stream_ending_without_finish_is_an_engine_error():
    frames = await collect(
        GenerateService(FakeFacade(FakeRequest([output([1])])), "inc", 5).stream(spec())
    )
    assert frames[-1]["error"]["code"] == ErrorCode.ENGINE_ERROR


async def test_shutdown_ends_new_and_running_streams():
    request = FakeRequest([], gate=asyncio.Event())
    service = GenerateService(FakeFacade(request), "inc", heartbeat_s=5)
    stream = service.stream(spec())
    await anext(stream)
    pending = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    service.begin_shutdown()
    assert decode_frame(await pending)["error"]["code"] == ErrorCode.SHUTTING_DOWN
    frames = await collect(service.stream(spec(skip_cache_store=True)))
    assert frames[-1]["error"]["code"] == ErrorCode.SHUTTING_DOWN


def test_generate_spec_validation():
    body = {
        "v": PROTOCOL_VERSION,
        "request_id": "r",
        "token_ids": [1, 2],
        "temperature": 0,
        "top_k": 5,
    }
    parsed = GenerateSpec.from_body(body)
    assert (
        parsed.temperature == 0.0 and parsed.top_k == 5 and parsed.stop_token_ids == ()
    )
    assert parsed.images == () and parsed.expanded_prompt_len is None
    images = [{"data": "aW1n", "mm_hash": "00000000000000ff"}, {"data": "aW1n"}]
    parsed = GenerateSpec.from_body(
        {
            **body,
            "images": images,
            "expanded_prompt_len": 9,
            "min_pixels": 65536,
            "max_pixels": 4194304,
        }
    )
    assert [(image.data, image.mm_hash) for image in parsed.images] == [
        ("aW1n", 0xFF),
        ("aW1n", None),
    ]
    assert parsed.expanded_prompt_len == 9
    assert (parsed.min_pixels, parsed.max_pixels) == (65536, 4194304)
    for bad in (
        {**body, "v": 99},
        {**body, "request_id": ""},
        {**body, "token_ids": []},
        {**body, "token_ids": [-1]},
        {**body, "stop_token_ids": 3},
        {**body, "max_tokens": 0},
        {**body, "top_k": 1.5},
        {**body, "temperature": True},
        {**body, "skip_cache_store": "yes"},
        {**body, "images": {"data": "aW1n"}},
        {**body, "images": [{"data": ""}]},
        {**body, "images": [{"data": "aW1n", "mm_hash": 255}]},
        {**body, "expanded_prompt_len": 1},
        {**body, "max_pixels": 0},
        {**body, "min_pixels": 100, "max_pixels": 99},
    ):
        with pytest.raises(BridgeError) as info:
            GenerateSpec.from_body(bad)
        assert info.value.code == ErrorCode.INVALID_REQUEST


# -- KV stream -------------------------------------------------------------------------

B = 4


def bound_log() -> KvEventLog:
    """A log whose wake-ups go to the running test loop (call from async tests)."""
    log = KvEventLog()
    log.set_block_size(B)
    log.bind_loop(asyncio.get_running_loop())
    return log


async def test_kv_stream_rejects_unknown_incarnation():
    kv_log = bound_log()
    service = KvStreamService(kv_log, lambda: KvState.LIVE, 5)
    hello, reset = await collect(service.stream("other", 0, B))
    assert (hello["t"], hello["block_size"], hello["inc"]) == (
        "hello",
        B,
        kv_log.incarnation,
    )
    assert reset == {"t": "reset", "reason": "incarnation"}


async def test_kv_stream_resumes_and_heartbeats():
    kv_log = bound_log()
    kv_log.store_chain(None, chain(1, 1, B))
    kv_log.store_chain(None, chain(2, 1, B))
    service = KvStreamService(kv_log, lambda: KvState.LIVE, heartbeat_s=0.01)
    stream = service.stream(kv_log.incarnation, 1, B)
    assert decode_frame(await anext(stream))["t"] == "hello"
    assert decode_frame(await anext(stream))["seq"] == 2
    heartbeat = decode_frame(await anext(stream))
    assert heartbeat == {"t": "hb", "seq": 2}
    kv_log.store_chain(None, chain(3, 1, B))
    frame = decode_frame(await anext(stream))
    while frame["t"] == "hb":
        frame = decode_frame(await anext(stream))
    assert frame["seq"] == 3
    service.begin_shutdown()
    rest = [decode_frame(f) async for f in stream]
    assert rest[-1] == {"t": "reset", "reason": "shutdown"}


async def test_kv_stream_resets_when_the_ring_no_longer_covers_the_cursor():
    log = KvEventLog(ring_max_events=1)
    log.set_block_size(B)
    log.store_chain(None, chain(1, 1, B))
    log.store_chain(None, chain(2, 1, B))
    service = KvStreamService(log, lambda: KvState.LIVE, 5)
    frames = await collect(service.stream(log.incarnation, 0, B))
    assert frames[-1] == {"t": "reset", "reason": "floor"}
    frames = await collect(service.stream(log.incarnation, 9, B))
    assert frames[-1] == {"t": "reset", "reason": "ahead"}


async def test_kv_stream_and_snapshot_when_disabled():
    kv_log = bound_log()
    kv_log.store_chain(None, chain(1, 2, B))
    service = KvStreamService(kv_log, lambda: KvState.DISABLED, 5)
    frames = await collect(service.stream(kv_log.incarnation, 0, B))
    assert frames[-1]["reason"] == "disabled"
    snapshot = await collect(service.snapshot(B))
    assert snapshot[0]["count"] == 0 and snapshot[0]["state"] == "disabled"


async def test_snapshot_lists_live_blocks_parent_first():
    kv_log = bound_log()
    kv_log.store_chain(None, chain(1, 20, B))
    service = KvStreamService(kv_log, lambda: KvState.LIVE, 5)
    begin, *chunks, end = await collect(service.snapshot(B))
    assert (begin["t"], begin["as_of"], begin["count"]) == (
        "snap_begin",
        kv_log.last_seq,
        20,
    )
    blocks = [block for frame in chunks for block in frame["blocks"]]
    assert [b["p"] for b in blocks[1:]] == [b["h"] for b in blocks[:-1]]
    assert end == {"t": "snap_end", "as_of": kv_log.last_seq, "count": 20}


# -- runtime ---------------------------------------------------------------------------


async def test_check_ready_maps_model_state_and_triggers_loads():
    facade = FakeFacade(state=ModelState.UNLOADED)
    bridge = make_runtime(facade, preload=False)
    with pytest.raises(BridgeError) as info:
        bridge.check_ready()
    assert (info.value.code, info.value.http_status) == (ErrorCode.MODEL_NOT_READY, 503)
    await asyncio.sleep(0)
    assert facade.load_calls == 1
    facade.state = ModelState.MISSING
    with pytest.raises(BridgeError) as info:
        bridge.check_ready()
    assert info.value.code == ErrorCode.MODEL_NOT_FOUND
    facade.state = ModelState.READY
    bridge.check_ready()
    await bridge.stop()


def test_info_and_stats_report_the_effective_kv_state():
    bridge = make_runtime(FakeFacade(), kv_events=False)
    info = bridge.info()
    assert info["v"] == PROTOCOL_VERSION and info["model"]["block_size"] == 4
    assert info["model"]["mtp_depth"] is None
    assert info["kv"]["state"] == KvState.DISABLED
    stats = bridge.stats()
    # The tracker itself is live; the flag disables KV events.
    assert bridge.tracker.state == KvState.LIVE
    assert stats["kv"]["state"] == KvState.DISABLED
    assert stats["engine"] == {"loaded": True}
    assert stats["images"] == {"requests": 0}
    assert stats["bridge"] == {"active_requests": 0}
    assert info["model"]["vision"] is False
    assert make_runtime(FakeFacade(vision=True)).info()["model"]["vision"] is True
