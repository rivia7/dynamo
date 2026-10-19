# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""deploy/base/anthropic_fix.py, the repair of Claude Code's event streams."""

import asyncio
import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "deploy" / "base" / "anthropic_fix.py"
_SPEC = importlib.util.spec_from_file_location("omlx_anthropic_fix", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
anthropic_fix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(anthropic_fix)

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def event(name: str, **payload) -> bytes:
    """One event as Dynamo 1.5.1's frontend writes it."""
    data = json.dumps({"type": name, **payload}, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode()


def start(index: int, block: dict) -> bytes:
    return event("content_block_start", index=index, content_block=block)


def delta(index: int, **body) -> bytes:
    return event("content_block_delta", index=index, delta=body)


def stop(index: int) -> bytes:
    return event("content_block_stop", index=index)


def tool(index: int, name: str, arguments: dict) -> bytes:
    block = {"type": "tool_use", "id": f"toolu_{index}", "name": name, "input": {}}
    return (
        start(index, block)
        + delta(index, type="input_json_delta", partial_json=json.dumps(arguments))
        + stop(index)
    )


TEXT = {"type": "text", "text": ""}
READ = {"file_path": "/src/README.md"}
BASH = {"command": "ls /src", "description": "List the sources"}
HEAD = event(
    "message_start",
    message={"id": "msg_1", "type": "message", "role": "assistant", "content": []},
)
THINKING = (
    start(0, {"type": "thinking", "thinking": "", "signature": ""})
    + delta(0, type="thinking_delta", thinking="Read both.")
    + delta(0, type="signature_delta", signature="erased")
    + stop(0)
)
TAIL = (
    event(
        "message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 9}
    )
    + event("message_stop")
    + b"data: [DONE]\n\n"
)
STRAY = delta(1, type="text_delta", text="\n")
# What Dynamo 1.5.1 sends for a turn with two tool calls: the newline between
# the calls arrives after the text block closed.
TWO_CALLS = (
    HEAD
    + THINKING
    + start(1, TEXT)
    + delta(1, type="text_delta", text="\n\n")
    + stop(1)
    + STRAY
    + tool(2, "Read", READ)
    + tool(3, "Bash", BASH)
    + TAIL
)
DELTA_BLOCKS = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "signature_delta": "thinking",
    "input_json_delta": "tool_use",
}


def content(stream: bytes) -> list[dict]:
    """The message's content blocks, built the way a client builds them; an
    assertion fails where a client would reject the stream."""
    events = []
    for raw in re.split(rb"\r?\n\r?\n", stream):
        lines = raw.decode().splitlines()
        names = [line[6:].strip() for line in lines if line.startswith("event:")]
        data = [line[5:].strip() for line in lines if line.startswith("data:")]
        if names:
            events.append((names[0], json.loads(data[0])))
    assert events[0][0] == "message_start" and events[-1][0] == "message_stop"
    blocks: list[dict] = []
    open_index = None
    for name, payload in events:
        if name == "content_block_start":
            assert open_index is None and payload["index"] == len(blocks)
            open_index = payload["index"]
            blocks.append(dict(payload["content_block"]))
        elif name == "content_block_delta":
            assert payload["index"] == open_index
            block, body = blocks[open_index], payload["delta"]
            assert DELTA_BLOCKS[body["type"]] == block["type"]
            if body["type"] == "text_delta":
                block["text"] += body["text"]
            elif body["type"] == "thinking_delta":
                block["thinking"] += body["thinking"]
            elif body["type"] == "signature_delta":
                block["signature"] = body["signature"]
            else:
                block["arguments"] = block.get("arguments", "") + body["partial_json"]
        elif name == "content_block_stop":
            assert payload["index"] == open_index
            open_index = None
        elif name == "message_delta":
            assert open_index is None
    return blocks


def repair(stream: bytes, size: int = 0):
    fix = anthropic_fix.StreamRepair()
    parts = (
        [stream[i : i + size] for i in range(0, len(stream), size)]
        if size
        else [stream]
    )
    return b"".join(fix.feed(part) for part in parts) + fix.close(), fix


def test_the_frontend_stream_fails_a_client():
    with pytest.raises(AssertionError):
        content(TWO_CALLS)


def test_whitespace_between_tool_calls_is_dropped():
    out, fix = repair(TWO_CALLS)
    blocks = content(out)
    assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use", "tool_use"]
    assert [json.loads(b["arguments"]) for b in blocks[2:]] == [READ, BASH]
    assert out == TWO_CALLS.replace(STRAY, b"")
    assert (fix.dropped, fix.opened) == (1, 0)


def test_text_after_a_tool_call_gets_a_block_of_its_own():
    stream = (
        HEAD
        + THINKING
        + start(1, TEXT)
        + delta(1, type="text_delta", text="First the README.")
        + stop(1)
        + delta(1, type="text_delta", text="\nThen ")
        + delta(1, type="text_delta", text="the sources.")
        + tool(2, "Read", READ)
        + tool(3, "Bash", BASH)
        + TAIL
    )
    out, fix = repair(stream)
    blocks = content(out)
    assert [b["type"] for b in blocks] == [
        "thinking",
        "text",
        "text",
        "tool_use",
        "tool_use",
    ]
    assert blocks[2]["text"] == "\nThen the sources."
    assert [json.loads(b["arguments"]) for b in blocks[3:]] == [READ, BASH]
    assert (fix.dropped, fix.opened) == (0, 1)


def test_a_block_opened_here_closes_before_the_message_ends():
    stream = (
        HEAD
        + start(0, TEXT)
        + delta(0, type="text_delta", text="a")
        + stop(0)
        + delta(0, type="text_delta", text="b")
        + TAIL
    )
    assert [b["text"] for b in content(repair(stream)[0])] == ["a", "b"]


def test_late_thinking_gets_a_signed_block():
    stream = (
        HEAD
        + THINKING
        + start(1, TEXT)
        + delta(1, type="text_delta", text="x")
        + stop(1)
        + delta(0, type="thinking_delta", thinking="more")
        + TAIL
    )
    blocks = content(repair(stream)[0])
    assert blocks[2] == {"type": "thinking", "thinking": "more", "signature": "erased"}


def test_valid_streams_pass_unchanged():
    stream = (
        HEAD
        + THINKING
        + start(1, TEXT)
        + delta(1, type="text_delta", text="Hello")
        + stop(1)
        + tool(2, "Read", READ)
        + TAIL
    )
    out, fix = repair(stream)
    assert out == stream
    assert (fix.dropped, fix.opened) == (0, 0)


@pytest.mark.parametrize("size", [1, 7, 64])
def test_chunk_boundaries_do_not_matter(size):
    assert repair(TWO_CALLS, size)[0] == repair(TWO_CALLS)[0]


def test_crlf_separated_events_are_repaired():
    out, fix = repair(TWO_CALLS.replace(b"\n", b"\r\n"))
    assert [b["type"] for b in content(out)] == [
        "thinking",
        "text",
        "tool_use",
        "tool_use",
    ]
    assert fix.dropped == 1


def test_a_stray_delta_beside_an_open_block_passes_on_unchanged():
    # No new block may start while another is open: the stream stays as it was.
    stream = (
        HEAD
        + start(0, TEXT)
        + stop(0)
        + start(1, {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}})
        + delta(0, type="text_delta", text="late")
        + delta(1, type="input_json_delta", partial_json=json.dumps(READ))
        + stop(1)
        + TAIL
    )
    assert repair(stream)[0] == stream


def test_an_unfinished_event_is_passed_on_at_the_end():
    fix = anthropic_fix.StreamRepair()
    assert fix.feed(HEAD + b"event: ping\ndata: {}") == HEAD
    assert fix.close() == b"event: ping\ndata: {}"


def test_keep_alive_comments_pass_unchanged():
    # The frontend's heartbeat (DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS), also
    # while a block opened here is still open.
    beat = b":\n\n"
    stream = TWO_CALLS.replace(STRAY, STRAY + beat).replace(HEAD, HEAD + beat)
    out, _ = repair(stream)
    assert out == TWO_CALLS.replace(STRAY, beat).replace(HEAD, HEAD + beat)
    stray = delta(1, type="text_delta", text="Then the sources.")
    out, _ = repair(TWO_CALLS.replace(STRAY, stray + beat))
    assert beat in out
    assert [b["type"] for b in content(out)] == [
        "thinking",
        "text",
        "text",
        "tool_use",
        "tool_use",
    ]


@pytest.fixture
async def frontend():
    """A fake frontend: streams TWO_CALLS one event at a time when the request
    asks for a stream, and answers 400 otherwise."""
    seen = []

    async def messages(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        seen.append((request.method, request.rel_url.path_qs, request.headers, body))
        if not json.loads(body).get("stream"):
            error = {"type": "error", "error": {"type": "invalid_request_error"}}
            return web.json_response(error, status=400)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for part in re.split(rb"(?<=\n\n)", TWO_CALLS):
            await response.write(part)
        await response.write_eof()
        return response

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_post("/v1/messages", messages)
    async with TestServer(app) as server:
        yield server, seen


@pytest.fixture
async def client(frontend):
    server, _ = frontend
    app = anthropic_fix.make_app(str(server.make_url("/")))
    async with TestClient(TestServer(app)) as client:
        yield client


async def test_streams_are_repaired_on_the_way(frontend, client):
    _, seen = frontend
    headers = {"Authorization": "Bearer token", "anthropic-version": "2023-06-01"}
    response = await client.post(
        "/v1/messages?beta=true", json={"stream": True}, headers=headers
    )
    assert response.status == 200
    assert response.content_type == "text/event-stream"
    blocks = content(await response.read())
    assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use", "tool_use"]
    ((method, path, sent, body),) = seen
    assert (method, path, json.loads(body)) == (
        "POST",
        "/v1/messages?beta=true",
        {"stream": True},
    )
    assert sent["Authorization"] == "Bearer token"
    assert sent["anthropic-version"] == "2023-06-01"


async def test_other_answers_pass_through(client):
    response = await client.post("/v1/messages", json={"stream": False})
    assert response.status == 400
    assert (await response.json())["error"]["type"] == "invalid_request_error"


async def test_large_requests_pass(frontend, client):
    _, seen = frontend
    body = json.dumps({"stream": False, "padding": "x" * (8 * 1024 * 1024)}).encode()
    response = await client.post(
        "/v1/messages",
        data=io.BytesIO(body),
        headers={"Content-Type": "application/json"},
    )
    assert response.status == 400
    assert seen[0][3] == body


async def test_health_is_answered_here(frontend, client):
    _, seen = frontend
    response = await client.get("/healthz")
    assert response.status == 200
    assert seen == []


async def test_a_client_that_leaves_ends_the_frontend_stream():
    gone = asyncio.Event()

    async def endless(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            while True:
                await response.write(event("ping"))
                await asyncio.sleep(0.02)
        except (ConnectionError, asyncio.CancelledError):
            gone.set()
            raise

    frontend_app = web.Application()
    frontend_app.router.add_post("/v1/messages", endless)
    async with TestServer(frontend_app) as server:
        app = anthropic_fix.make_app(str(server.make_url("/")))
        async with TestClient(TestServer(app, handler_cancellation=True)) as client:
            response = await client.post("/v1/messages", json={"stream": True})
            await response.content.readany()
            response.close()
            await asyncio.wait_for(gone.wait(), 5)
