# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP routes mounted on oMLX's FastAPI app under ``protocol.API_PREFIX``."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from dynamo_omlx.bridge.service import BridgeError, BridgeRuntime, GenerateSpec
from dynamo_omlx.protocol import NDJSON_MEDIA_TYPE, ErrorCode
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


def _error_response(error: BridgeError) -> JSONResponse:
    return JSONResponse(
        status_code=error.http_status, content={"error": error.payload()}
    )


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise BridgeError(
            ErrorCode.INVALID_REQUEST, f"body is not valid JSON: {e}"
        ) from e


def build_router(
    runtime: BridgeRuntime, auth_dependency: Callable[..., Any] | None = None
) -> APIRouter:
    """Routes for one served model. ``auth_dependency`` guards every route."""
    router = APIRouter(
        dependencies=[Depends(auth_dependency)] if auth_dependency is not None else []
    )

    @router.get("/info")
    async def info() -> dict[str, Any]:
        return runtime.info()

    @router.get("/stats")
    async def stats() -> dict[str, Any]:
        return runtime.stats()

    @router.post("/generate")
    async def generate(request: Request) -> Response:
        try:
            spec = GenerateSpec.from_body(await _json_body(request))
            runtime.check_ready()
        except BridgeError as e:
            return _error_response(e)
        return StreamingResponse(
            runtime.generate.stream(spec), media_type=NDJSON_MEDIA_TYPE
        )

    @router.post("/cancel")
    async def cancel(request: Request) -> Response:
        try:
            body = await _json_body(request)
        except BridgeError as e:
            return _error_response(e)
        request_id = body.get("request_id") if isinstance(body, dict) else None
        if not isinstance(request_id, str) or not request_id:
            return _error_response(
                BridgeError(
                    ErrorCode.INVALID_REQUEST, "'request_id' must be a non-empty string"
                )
            )
        return JSONResponse({"cancelled": runtime.generate.cancel(request_id)})

    @router.get("/kv/stream")
    async def kv_stream(
        inc: str | None = None, after: int = Query(0, ge=0)
    ) -> StreamingResponse:
        block_size = runtime.facade.model_info().block_size
        return StreamingResponse(
            runtime.kv.stream(inc, after, block_size), media_type=NDJSON_MEDIA_TYPE
        )

    @router.get("/kv/snapshot")
    async def kv_snapshot() -> StreamingResponse:
        block_size = runtime.facade.model_info().block_size
        return StreamingResponse(
            runtime.kv.snapshot(block_size), media_type=NDJSON_MEDIA_TYPE
        )

    return router
