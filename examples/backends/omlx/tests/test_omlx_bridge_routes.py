# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import httpx
    from fastapi import FastAPI, HTTPException, Request
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

from dynamo_omlx.bridge.routes import build_router
from dynamo_omlx.bridge.service import BridgeRuntime, ModelState
from dynamo_omlx.protocol import API_PREFIX, PROTOCOL_VERSION, decode_frame
from omlx_fakes import FakeFacade, FakeRequest, make_runtime, output

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

API_KEY = "secret"


async def require_key(request: Request) -> None:
    if request.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="bad key")


def make_client(facade: FakeFacade) -> tuple[httpx.AsyncClient, BridgeRuntime]:
    runtime = make_runtime(facade, preload=False)
    app = FastAPI()
    app.include_router(build_router(runtime, require_key), prefix=API_PREFIX)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bridge" + API_PREFIX,
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    return client, runtime


def frames(response: httpx.Response) -> list[dict]:
    return [decode_frame(line) for line in response.text.splitlines() if line]


async def test_every_route_requires_the_api_key():
    client, _ = make_client(FakeFacade())
    async with client:
        for method, path in (
            ("GET", "/info"),
            ("GET", "/kv/snapshot"),
            ("POST", "/generate"),
            ("POST", "/cancel"),
        ):
            response = await client.request(
                method, path, headers={"Authorization": "Bearer wrong"}
            )
            assert response.status_code == 401, path


async def test_info_and_stats():
    client, _ = make_client(FakeFacade())
    async with client:
        info = (await client.get("/info")).json()
        assert info["v"] == PROTOCOL_VERSION and info["model"]["state"] == "ready"
        stats = (await client.get("/stats")).json()
        assert stats["bridge"]["active_requests"] == 0


async def test_generate_streams_ndjson():
    request = FakeRequest(
        [output([5, 6]), output([], finished=True, reason="length", completion=2)]
    )
    client, _ = make_client(FakeFacade(request))
    async with client:
        response = await client.post(
            "/generate",
            json={"v": PROTOCOL_VERSION, "request_id": "r", "token_ids": [1]},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [f["t"] for f in frames(response)] == ["start", "tok", "end"]


async def test_generate_reports_errors_before_streaming():
    client, _ = make_client(FakeFacade(state=ModelState.LOADING))
    async with client:
        response = await client.post(
            "/generate",
            json={"v": PROTOCOL_VERSION, "request_id": "r", "token_ids": [1]},
        )
        assert response.status_code == 503
        assert response.json()["error"] == {
            "code": "model_not_ready",
            "message": "model 'm' is loading",
            "retryable": True,
        }
        response = await client.post("/generate", content=b"{not json")
        assert (
            response.status_code == 400
            and response.json()["error"]["code"] == "invalid_request"
        )
        response = await client.post(
            "/generate",
            json={"v": PROTOCOL_VERSION, "request_id": "r", "token_ids": []},
        )
        assert response.status_code == 400


async def test_cancel_unknown_request():
    client, _ = make_client(FakeFacade())
    async with client:
        assert (await client.post("/cancel", json={"request_id": "nope"})).json() == {
            "cancelled": False
        }
        assert (await client.post("/cancel", json={})).status_code == 400


async def test_kv_snapshot_and_stream_reset():
    client, runtime = make_client(FakeFacade())
    async with client:
        snapshot = frames(await client.get("/kv/snapshot"))
        assert [f["t"] for f in snapshot] == ["snap_begin", "snap_end"]
        assert snapshot[0]["inc"] == runtime.log.incarnation
        stream = frames(
            await client.get("/kv/stream", params={"inc": "stale", "after": 0})
        )
        assert [f["t"] for f in stream] == ["hello", "reset"] and stream[1][
            "reason"
        ] == "incarnation"
        assert (await client.get("/kv/stream", params={"after": -1})).status_code == 422
