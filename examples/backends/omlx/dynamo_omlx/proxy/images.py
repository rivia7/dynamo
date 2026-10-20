# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turns a request's image URLs into the bridge's base64 image payloads.

The bridge takes image files only, so the proxy passes ``data:`` URLs on and
fetches ``http(s)`` URLs itself. Fetches reach public addresses only, unless
``DYN_MM_ALLOW_INTERNAL`` allows internal ones as it does for the frontend:
an image URL must not become a way into the cluster.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
from collections.abc import Sequence
from typing import Any

import aiohttp
from aiohttp.abc import AbstractResolver
from dynamo_omlx.protocol import format_mm_hash
from dynamo_omlx.proxy.request_map import ImageRef, RequestRejected
from yarl import URL

_REDIRECTS = (301, 302, 303, 307, 308)
_MAX_REDIRECTS = 5
_CHUNK_BYTES = 1 << 16


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class _PublicResolver(AbstractResolver):
    """Resolves host names to their public addresses only."""

    def __init__(self) -> None:
        self._resolver = aiohttp.DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[Any]:
        addresses = [
            address
            for address in await self._resolver.resolve(host, port, family)
            if _public(address["host"])
        ]
        if not addresses:
            raise OSError(f"{host} has no public address")
        return addresses

    async def close(self) -> None:
        await self._resolver.close()


def _data_url_payload(url: str, max_bytes: int) -> str:
    header, comma, payload = url.partition(",")
    params = [part.strip().lower() for part in header[len("data:") :].split(";")]
    if not comma or not params[0].startswith("image/") or "base64" not in params[1:]:
        raise RequestRejected(
            "image data: URLs must be base64 images (data:image/...;base64,...)"
        )
    if len(payload) // 4 * 3 > max_bytes:
        raise RequestRejected(f"an image has more than {max_bytes} bytes")
    return payload


class ImageLoader:
    def __init__(
        self, *, max_bytes: int, timeout_s: float, allow_internal: bool = False
    ) -> None:
        self._max_bytes = max_bytes
        self._timeout_s = timeout_s
        self._allow_internal = allow_internal
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None:
            # Its own session: the bridge client's carries the Mac's API key.
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    resolver=None if self._allow_internal else _PublicResolver()
                ),
                timeout=aiohttp.ClientTimeout(total=self._timeout_s),
            )

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    async def load(self, images: Sequence[ImageRef]) -> list[dict[str, Any]]:
        """The bridge's ``images`` entries; raises RequestRejected for bad images."""
        payloads = await asyncio.gather(*(self._payload(image.url) for image in images))
        return [
            {
                "data": payload,
                "mm_hash": None
                if image.mm_hash is None
                else format_mm_hash(image.mm_hash),
            }
            for image, payload in zip(images, payloads)
        ]

    async def _payload(self, url: str) -> str:
        if url[:5].lower() == "data:":
            return _data_url_payload(url, self._max_bytes)
        if url.split(":", 1)[0].lower() not in ("http", "https"):
            raise RequestRejected("image URLs must be data: or http(s) URLs")
        try:
            return base64.b64encode(await self._fetch(URL(url))).decode("ascii")
        except RequestRejected:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as e:
            raise RequestRejected(f"cannot fetch image {url[:200]}: {e}") from e

    def _check_target(self, url: URL) -> None:
        if url.scheme not in ("http", "https") or not url.host:
            raise RequestRejected(f"cannot fetch image {str(url)[:200]}")
        # The connector resolves names through _PublicResolver, but connects to
        # address literals directly.
        if not self._allow_internal:
            try:
                literal = not _public(url.host)
            except ValueError:  # a name, not an address
                literal = False
            if literal:
                raise RequestRejected(
                    f"image URL {str(url)[:200]} points to an internal address"
                )

    async def _fetch(self, url: URL) -> bytes:
        if self._session is None:
            raise RuntimeError("ImageLoader.start() was not called")
        for _ in range(_MAX_REDIRECTS + 1):
            self._check_target(url)
            async with self._session.get(url, allow_redirects=False) as response:
                location = response.headers.get("Location")
                if response.status in _REDIRECTS and location:
                    url = response.url.join(URL(location))
                    continue
                if response.status != 200:
                    raise RequestRejected(
                        f"image {str(url)[:200]} answered HTTP {response.status}"
                    )
                data = bytearray()
                async for chunk in response.content.iter_chunked(_CHUNK_BYTES):
                    data += chunk
                    if len(data) > self._max_bytes:
                        raise RequestRejected(
                            f"image {str(url)[:200]} has more than {self._max_bytes} bytes"
                        )
                return bytes(data)
        raise RequestRejected(f"image {str(url)[:200]} redirects too often")
