# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from dynamo_omlx.proxy.images import ImageLoader, _public, _PublicResolver
from dynamo_omlx.proxy.request_map import ImageRef, RequestRejected
from yarl import URL

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.fixture
async def loader():
    images = ImageLoader(max_bytes=1000, timeout_s=5)
    await images.start()
    yield images
    await images.close()


async def test_data_urls_pass_through_with_their_router_identity(loader):
    entries = await loader.load([ImageRef(PNG_URL, 0xFF), ImageRef(PNG_URL, None)])
    payload = base64.b64encode(PNG_BYTES).decode()
    assert entries == [
        {"data": payload, "mm_hash": "00000000000000ff"},
        {"data": payload, "mm_hash": None},
    ]


@pytest.mark.parametrize(
    "url, message",
    [
        ("data:text/plain;base64,aGk=", "must be base64 images"),
        ("data:image/png,raw", "must be base64 images"),
        ("data:image/png;base64," + "A" * 2000, "more than 1000 bytes"),
        ("ftp://host/image.png", "data: or http"),
        ("file:///etc/passwd", "data: or http"),
        ("http://127.0.0.1:8000/image.png", "internal address"),
        ("http://[::1]/image.png", "internal address"),
        ("http://10.43.0.1/image.png", "internal address"),
        ("http://169.254.169.254/latest/meta-data", "internal address"),
        ("http:///image.png", "cannot fetch image"),
    ],
)
async def test_rejected_image_urls(loader, url, message):
    with pytest.raises(RequestRejected, match=message):
        await loader.load([ImageRef(url, None)])


def test_only_global_unicast_addresses_are_public():
    assert _public("93.184.216.34") and _public("2606:2800:220:1::")
    for address in (
        "10.0.0.1",
        "100.64.0.1",
        "192.168.1.20",
        "127.0.0.1",
        "::1",
        "fe80::1%eth0",
        "::ffff:10.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
    ):
        assert not _public(address), address


async def test_names_resolve_to_their_public_addresses_only(monkeypatch):
    resolver = _PublicResolver()
    answers = [{"host": "10.0.0.1"}, {"host": "93.184.216.34"}]

    async def resolve(host, port=0, family=0):
        return answers

    monkeypatch.setattr(resolver._resolver, "resolve", resolve)
    assert await resolver.resolve("example.com") == [{"host": "93.184.216.34"}]
    answers.pop()
    with pytest.raises(OSError, match="no public address"):
        await resolver.resolve("example.com")
    await resolver.close()


async def test_http_images_are_fetched_through_redirects_within_limits():
    async def image(request):
        return web.Response(body=PNG_BYTES, content_type="image/png")

    async def moved(request):
        raise web.HTTPFound("/image.png")

    async def loop(request):
        raise web.HTTPFound("/loop")

    async def big(request):
        return web.Response(body=b"x" * 2000)

    app = web.Application()
    app.router.add_get("/image.png", image)
    app.router.add_get("/moved", moved)
    app.router.add_get("/loop", loop)
    app.router.add_get("/big", big)
    server = TestServer(app)
    await server.start_server()
    # The test server listens on loopback, which only an operator opt-in reaches.
    loader = ImageLoader(max_bytes=1000, timeout_s=5, allow_internal=True)
    await loader.start()
    try:
        url = str(server.make_url("/moved"))
        [entry] = await loader.load([ImageRef(url, 7)])
        assert entry == {
            "data": base64.b64encode(PNG_BYTES).decode(),
            "mm_hash": "0000000000000007",
        }
        for path, message in (
            ("/missing", "answered HTTP 404"),
            ("/big", "more than 1000 bytes"),
            ("/loop", "redirects too often"),
        ):
            with pytest.raises(RequestRejected, match=message):
                await loader.load([ImageRef(str(server.make_url(path)), None)])
    finally:
        await loader.close()
        await server.close()


def test_redirect_targets_are_checked_like_the_first_url():
    # Every hop goes through _check_target before it is fetched.
    loader = ImageLoader(max_bytes=1000, timeout_s=5)
    with pytest.raises(RequestRejected, match="internal address"):
        loader._check_target(URL("http://192.168.1.20:8000/dynamo/v1/info"))
    loader._check_target(URL("https://example.com/image.png"))
    ImageLoader(max_bytes=1000, timeout_s=5, allow_internal=True)._check_target(
        URL("http://192.168.1.20:8000/image.png")
    )
