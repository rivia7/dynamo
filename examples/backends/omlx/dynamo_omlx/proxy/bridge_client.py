# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp client for the bridge API (see ``dynamo_omlx.protocol``)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from aiohttp.http_exceptions import LineTooLong
from dynamo_omlx.protocol import (
    API_PREFIX,
    MAX_LINE_BYTES,
    ErrorCode,
    ProtocolError,
    decode_frame,
)

# aiohttp rejects lines longer than twice the read buffer size.
_READ_BUFSIZE = MAX_LINE_BYTES // 2


class BridgeUnavailable(ConnectionError):
    """The bridge could not be reached, or the connection broke."""


class BridgeRequestError(Exception):
    """The bridge refused a request with a protocol error code."""

    def __init__(self, code: str, message: str, retryable: bool, status: int) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status = status


async def _raise_for_status(response: aiohttp.ClientResponse) -> None:
    if response.status == 200:
        return
    try:
        body = await response.json(content_type=None)
    except (aiohttp.ClientError, ValueError):
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        raise BridgeRequestError(
            error["code"],
            str(error.get("message", "")),
            bool(error.get("retryable")),
            response.status,
        )
    if response.status in (401, 403):
        raise BridgeRequestError(
            "unauthorized", "the bridge rejected the API key", False, response.status
        )
    if response.status >= 500:
        raise BridgeUnavailable(f"bridge returned HTTP {response.status}")
    raise BridgeRequestError(
        ErrorCode.INVALID_REQUEST,
        f"bridge returned HTTP {response.status}",
        False,
        response.status,
    )


async def _frames(response: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    while True:
        try:
            line = await response.content.readline()
        except LineTooLong as e:
            raise ProtocolError(
                f"bridge sent a line over {MAX_LINE_BYTES} bytes"
            ) from e
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise BridgeUnavailable(f"bridge stream broke: {e!r}") from e
        if not line:
            return
        if line.strip():
            yield decode_frame(line)


class BridgeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        connect_timeout_s: float = 5.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._url = self.base_url + API_PREFIX
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._connect_timeout_s = connect_timeout_s
        self._request_timeout_s = request_timeout_s
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0),
                headers=self._headers,
                read_bufsize=_READ_BUFSIZE,
            )

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("BridgeClient.start() was not called")
        return self._session

    def _timeout(self, sock_read: float | None) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=None, sock_connect=self._connect_timeout_s, sock_read=sock_read
        )

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            async with self.session.get(
                self._url + path, timeout=self._timeout(self._request_timeout_s)
            ) as response:
                await _raise_for_status(response)
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise BridgeUnavailable(f"GET {path} failed: {e!r}") from e

    async def info(self) -> dict[str, Any]:
        return await self._get_json("/info")

    async def stats(self) -> dict[str, Any]:
        return await self._get_json("/stats")

    async def cancel(self, request_id: str) -> bool:
        try:
            async with self.session.post(
                self._url + "/cancel",
                json={"request_id": request_id},
                timeout=self._timeout(self._request_timeout_s),
            ) as response:
                await _raise_for_status(response)
                return bool((await response.json()).get("cancelled"))
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise BridgeUnavailable(f"cancel failed: {e!r}") from e

    @asynccontextmanager
    async def _stream(
        self, method: str, path: str, sock_read: float | None, **kwargs: Any
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        try:
            response = await self.session.request(
                method, self._url + path, timeout=self._timeout(sock_read), **kwargs
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise BridgeUnavailable(f"{method} {path} failed: {e!r}") from e
        try:
            await _raise_for_status(response)
            yield _frames(response)
        finally:
            # close(), not release(): the bridge must see the connection drop so it
            # aborts a generation the proxy stopped reading.
            response.close()

    def generate(self, body: dict[str, Any], sock_read: float | None) -> Any:
        """Context manager yielding generate frames; ``sock_read`` bounds silence."""
        return self._stream("POST", "/generate", sock_read, json=body)

    def kv_stream(self, incarnation: str | None, after: int) -> Any:
        params = {"after": str(after)}
        if incarnation is not None:
            params["inc"] = incarnation
        # KvSync detects stalls from heartbeats itself.
        return self._stream("GET", "/kv/stream", None, params=params)

    def kv_snapshot(self) -> Any:
        return self._stream("GET", "/kv/snapshot", self._request_timeout_s)
