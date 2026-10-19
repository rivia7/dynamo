# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An aiohttp server that plays the Mac bridge with the real bridge services."""

import asyncio
from contextlib import aclosing, suppress

from aiohttp import web
from aiohttp.test_utils import TestServer
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.service import GenerateService, GenerateSpec, KvStreamService
from dynamo_omlx.protocol import (
    API_PREFIX,
    HASH_VERSION,
    NDJSON_MEDIA_TYPE,
    PROTOCOL_VERSION,
    KvState,
    error_payload,
)
from omlx_fakes import FakeFacade, FakeRequest, output


async def ndjson_response(request: web.Request, frames) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": NDJSON_MEDIA_TYPE})
    await response.prepare(request)
    async with aclosing(frames):
        with suppress(ConnectionResetError):
            async for frame in frames:
                await response.write(frame)
    return response


class FakeBridge:
    """``async with FakeBridge() as bridge``; tests steer it through its attributes."""

    def __init__(self, block_size: int = 2, api_key: str | None = None) -> None:
        self.block_size = block_size
        self.api_key = api_key
        self.log = KvEventLog()
        self.log.set_block_size(block_size)
        self.kv = KvStreamService(self.log, lambda: KvState.LIVE, heartbeat_s=0.05)
        self.request = FakeRequest(
            [output([7, 8]), output([], finished=True, reason="stop", completion=2)]
        )
        self.generate_service: GenerateService | None = None
        self.generate_error: tuple[str, str] | None = None  # (code, message)
        self.info_error: tuple[int, dict | str] | None = None  # (status, body)
        self.model = {
            "id": "m",
            "state": "ready",
            "block_size": block_size,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "context_length": 64,
        }
        app = web.Application(middlewares=[self._auth])
        app.router.add_get(API_PREFIX + "/info", self.info)
        app.router.add_get(API_PREFIX + "/stats", self.stats)
        app.router.add_post(API_PREFIX + "/generate", self.generate)
        app.router.add_post(API_PREFIX + "/cancel", self.cancel)
        app.router.add_get(API_PREFIX + "/kv/snapshot", self.snapshot)
        app.router.add_get(API_PREFIX + "/kv/stream", self.stream)
        self.server = TestServer(app)

    @property
    def url(self) -> str:
        return str(self.server.make_url("/"))

    @web.middleware
    async def _auth(self, request: web.Request, handler):
        if (
            self.api_key
            and request.headers.get("Authorization") != f"Bearer {self.api_key}"
        ):
            return web.json_response({"detail": "bad key"}, status=401)
        return await handler(request)

    async def info(self, request: web.Request) -> web.Response:
        if self.info_error is not None:
            status, body = self.info_error
            if isinstance(body, str):
                return web.Response(status=status, text=body)
            return web.json_response(body, status=status)
        return web.json_response(
            {
                "v": PROTOCOL_VERSION,
                "hash_version": HASH_VERSION,
                "model": self.model,
                "kv": {"state": KvState.LIVE},
            }
        )

    async def stats(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "model": self.model,
                "engine": {"loaded": True, "running": 1, "waiting": 0},
                "kv": {
                    "state": KvState.LIVE,
                    "live_blocks": len(self.log),
                    "seq": self.log.last_seq,
                },
                "bridge": {"active_requests": 0},
            }
        )

    async def generate(self, request: web.Request) -> web.StreamResponse:
        if self.generate_error is not None:
            return web.json_response(
                {"error": error_payload(*self.generate_error)}, status=503
            )
        spec = GenerateSpec.from_body(await request.json())
        self.generate_service = GenerateService(
            FakeFacade(self.request), self.log.incarnation, heartbeat_s=0.05
        )
        return await ndjson_response(request, self.generate_service.stream(spec))

    async def cancel(self, request: web.Request) -> web.Response:
        body = await request.json()
        cancelled = self.generate_service.cancel(body["request_id"])
        return web.json_response({"cancelled": cancelled})

    async def snapshot(self, request: web.Request) -> web.StreamResponse:
        return await ndjson_response(request, self.kv.snapshot(self.block_size))

    async def stream(self, request: web.Request) -> web.StreamResponse:
        after = int(request.query.get("after", "0"))
        frames = self.kv.stream(request.query.get("inc"), after, self.block_size)
        return await ndjson_response(request, frames)

    async def __aenter__(self) -> "FakeBridge":
        self.log.bind_loop(asyncio.get_running_loop())
        await self.server.start_server()
        return self

    async def __aexit__(self, *exc) -> None:
        self.kv.begin_shutdown()
        await self.server.close()
