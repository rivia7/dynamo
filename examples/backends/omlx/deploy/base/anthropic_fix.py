# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repairs the Anthropic Messages API streams of Dynamo 1.5.1's frontend.

When a model calls several tools in one turn, the frontend closes the text
block at the first tool call but sends the text between the calls, such as a
newline, as deltas of that closed block. Clients that track blocks, Claude Code
among them ("The response stream was malformed"), give up on the stream.

This proxy sits between the Ingress and the frontend for /v1/messages and
passes requests and responses through unchanged, except in event streams: it
drops a whitespace-only delta of a closed block, and puts any other one into a
new block of its own, numbering the blocks after it one higher.

    python3 anthropic_fix.py [--port 8080] [--upstream http://omlx-frontend:8000]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re

import aiohttp
from aiohttp import web

# Claude Code's requests carry whole files in tool results; aiohttp's default
# limit is 1 MiB. Anthropic's own limit is 32 MB.
MAX_REQUEST_BYTES = 64 * 1024 * 1024
# Headers that belong to one connection, or that aiohttp sets itself.
HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Without accept-encoding the frontend sends its streams uncompressed.
REQUEST_SKIP = HOP_HEADERS | {"accept-encoding"}
RESPONSE_SKIP = HOP_HEADERS | {"date", "server"}
EVENT_END = re.compile(rb"\r?\n\r?\n")
# The block a stray delta of each type continues in.
STRAY_BLOCKS = {
    "text_delta": {"type": "text", "text": ""},
    "thinking_delta": {"type": "thinking", "thinking": "", "signature": ""},
}

UPSTREAM = web.AppKey("upstream", str)
SESSION = web.AppKey("session", aiohttp.ClientSession)

log = logging.getLogger("anthropic_fix")


def sse(name: str, payload: dict) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode()


def parse(event: bytes) -> tuple[str | None, str | None]:
    """The event name and data of one server-sent event."""
    name, data = None, []
    for line in event.decode("utf-8", "replace").splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    return name, "\n".join(data) if data else None


class StreamRepair:
    """Rewrites one event stream; ``feed`` takes the bytes as they arrive."""

    def __init__(self) -> None:
        self.buffer = b""
        # Block index from the frontend -> index sent on.
        self.index: dict[int, int] = {}
        self.closed: set[int] = set()
        self.next = 0
        # (frontend index, index sent on, delta type) of a block opened here.
        self.extra: tuple[int, int, str] | None = None
        self.dropped = 0
        self.opened = 0

    def feed(self, data: bytes) -> bytes:
        self.buffer += data
        out = []
        while match := EVENT_END.search(self.buffer):
            event, self.buffer = (
                self.buffer[: match.end()],
                self.buffer[match.end() :],
            )
            out.append(self.event(event))
        return b"".join(out)

    def close(self) -> bytes:
        """What is left at the end of the stream, passed on unchanged."""
        rest, self.buffer = self.buffer, b""
        return rest

    def event(self, raw: bytes) -> bytes:
        name, data = parse(raw)
        if name in ("message_delta", "message_stop"):
            return self.end_extra() + raw
        if name not in (
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ):
            return raw
        try:
            payload = json.loads(data or "")
        except ValueError:
            return raw
        upstream = payload.get("index") if isinstance(payload, dict) else None
        if not isinstance(upstream, int):
            return raw
        if name == "content_block_start":
            prefix = self.end_extra()
            self.index[upstream] = self.next
            self.next += 1
            return prefix + self.renumbered(raw, name, payload, self.index[upstream])
        if name == "content_block_stop":
            prefix = self.end_extra()
            self.closed.add(upstream)
            sent = self.index.get(upstream, upstream)
            return prefix + self.renumbered(raw, name, payload, sent)
        if upstream in self.index and upstream not in self.closed:
            sent = self.index[upstream]
            return self.end_extra() + self.renumbered(raw, name, payload, sent)
        return self.stray(raw, payload, upstream)

    def stray(self, raw: bytes, payload: dict, upstream: int) -> bytes:
        """A delta of a block that is closed or never started."""
        delta = payload.get("delta")
        kind = delta.get("type") if isinstance(delta, dict) else None
        if kind not in STRAY_BLOCKS:
            return raw
        if (
            self.extra is not None
            and self.extra[0] == upstream
            and self.extra[2] == kind
        ):
            return self.renumbered(raw, "content_block_delta", payload, self.extra[1])
        text = delta.get("text" if kind == "text_delta" else "thinking", "")
        if not str(text).strip():
            self.dropped += 1
            return b""
        if any(i not in self.closed for i in self.index):
            # Another block is open, so no new one may start: pass it on as is.
            return raw
        prefix = self.end_extra()
        sent = self.next
        self.next += 1
        self.extra = (upstream, sent, kind)
        self.opened += 1
        start = {
            "type": "content_block_start",
            "index": sent,
            "content_block": STRAY_BLOCKS[kind],
        }
        return (
            prefix
            + sse("content_block_start", start)
            + self.renumbered(raw, "content_block_delta", payload, sent)
        )

    def end_extra(self) -> bytes:
        """Close the block opened here, if any."""
        if self.extra is None:
            return b""
        _, sent, kind = self.extra
        self.extra = None
        out = b""
        if kind == "thinking_delta":
            # The frontend ends its thinking blocks this way too.
            signature = {"type": "signature_delta", "signature": "erased"}
            out = sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": sent, "delta": signature},
            )
        return out + sse(
            "content_block_stop", {"type": "content_block_stop", "index": sent}
        )

    @staticmethod
    def renumbered(raw: bytes, name: str, payload: dict, sent: int) -> bytes:
        if payload["index"] == sent:
            return raw
        return sse(name, {**payload, "index": sent})


async def health(request: web.Request) -> web.Response:
    return web.Response(text="ok\n")


async def forward(request: web.Request) -> web.StreamResponse:
    session = request.app[SESSION]
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in REQUEST_SKIP
    }
    # aiohttp sends a file object in parts; raw bytes over 1 MiB would hold
    # up every other stream while they go out.
    body = io.BytesIO(await request.read())
    url = request.app[UPSTREAM] + request.rel_url.path_qs
    async with session.request(
        request.method, url, headers=headers, data=body, allow_redirects=False
    ) as upstream:
        response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        for key, value in upstream.headers.items():
            if key.lower() not in RESPONSE_SKIP:
                response.headers.add(key, value)
        await response.prepare(request)
        repair = (
            StreamRepair() if upstream.content_type == "text/event-stream" else None
        )
        finished = False
        try:
            async for chunk in upstream.content.iter_any():
                out = repair.feed(chunk) if repair else chunk
                if out:
                    await response.write(out)
            if repair:
                await response.write(repair.close())
            finished = True
        finally:
            if not finished:
                # The client left: close the frontend's stream as well, which
                # cancels the request on the Mac.
                upstream.close()
        if repair and (repair.dropped or repair.opened):
            log.info(
                "%s %s: dropped %d whitespace deltas of closed blocks, opened %d blocks",
                request.method,
                request.rel_url.path,
                repair.dropped,
                repair.opened,
            )
        await response.write_eof()
        return response


async def session_context(app: web.Application):
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=True) as session:
        app[SESSION] = session
        yield


def make_app(upstream: str) -> web.Application:
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app[UPSTREAM] = upstream.rstrip("/")
    app.cleanup_ctx.append(session_context)
    app.router.add_get("/healthz", health)
    app.router.add_route("*", "/{tail:.*}", forward)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", default="http://omlx-frontend:8000")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log.info("repairing Anthropic event streams from %s", args.upstream)
    # handler_cancellation stops a forward, and with it the frontend's
    # request, as soon as its client disconnects.
    web.run_app(
        make_app(args.upstream),
        port=args.port,
        handler_cancellation=True,
        access_log=None,
        print=None,
    )


if __name__ == "__main__":
    main()
