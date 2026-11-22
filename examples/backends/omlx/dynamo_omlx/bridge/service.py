# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic bridge services: token streams, KV streams and the runtime.

The oMLX specifics live behind :class:`EngineFacade`, implemented by
``omlx_adapter``; tests substitute a fake.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from dynamo_omlx.bridge.config import BridgeConfig
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.kv_tracker import KvTracker
from dynamo_omlx.protocol import (
    ERROR_HTTP_STATUS,
    HASH_VERSION,
    MAX_BLOCKS_PER_FRAME,
    PROTOCOL_VERSION,
    ErrorCode,
    FinishReason,
    Frame,
    KvState,
    ProtocolError,
    ResetReason,
    encode_frame,
    error_payload,
    parse_mm_hash,
    tokens_to_array,
)

logger = logging.getLogger(__name__)


class ModelState:
    READY = "ready"
    LOADING = "loading"
    UNLOADED = "unloaded"
    MISSING = "missing"
    FAILED = "failed"


class BridgeError(Exception):
    """A failure reported to the proxy with a protocol error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS.get(self.code, 500)

    def payload(self) -> dict[str, Any]:
        return error_payload(self.code, self.message)


def _optional_number(body: dict[str, Any], key: str, kind: type) -> Any:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
        value, (int, float) if kind is float else int
    ):
        raise BridgeError(
            ErrorCode.INVALID_REQUEST, f"'{key}' must be a {kind.__name__}"
        )
    return kind(value)


@dataclass(frozen=True)
class ImageInput:
    # Base64 of the image file, decoded off the event loop.
    data: str
    # The image's identity in Dynamo's router; None when the proxy had none.
    mm_hash: int | None = None


def _images(body: dict[str, Any]) -> tuple[ImageInput, ...]:
    items = body.get("images") or []
    if not isinstance(items, list):
        raise BridgeError(ErrorCode.INVALID_REQUEST, "'images' must be a list")
    images = []
    for item in items:
        data = item.get("data") if isinstance(item, dict) else None
        if not isinstance(data, str) or not data:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "each image needs base64 'data'"
            )
        try:
            mm_hash = parse_mm_hash(item.get("mm_hash"))
        except ProtocolError as e:
            raise BridgeError(ErrorCode.INVALID_REQUEST, str(e)) from e
        images.append(ImageInput(data=data, mm_hash=mm_hash))
    return tuple(images)


@dataclass(frozen=True)
class GenerateSpec:
    request_id: str
    # One placeholder token per image in images.
    token_ids: list[int]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    skip_cache_store: bool = False
    images: tuple[ImageInput, ...] = ()
    # The prompt length Dynamo's frontend routed with after expanding images.
    expanded_prompt_len: int | None = None
    # The pixel range the frontend counted image tokens with; None leaves the
    # model's image processor defaults.
    min_pixels: int | None = None
    max_pixels: int | None = None

    @classmethod
    def from_body(cls, body: Any) -> GenerateSpec:
        if not isinstance(body, dict):
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "request body must be a JSON object"
            )
        if body.get("v") != PROTOCOL_VERSION:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST,
                f"unsupported protocol version {body.get('v')!r}; this bridge speaks {PROTOCOL_VERSION}",
            )
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "'request_id' must be a non-empty string"
            )
        token_ids = body.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "'token_ids' must be a non-empty list"
            )
        stop_token_ids = body.get("stop_token_ids") or []
        if not isinstance(stop_token_ids, list):
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "'stop_token_ids' must be a list"
            )
        try:
            tokens_to_array(token_ids)
            tokens_to_array(stop_token_ids)
        except ProtocolError as e:
            raise BridgeError(ErrorCode.INVALID_REQUEST, str(e)) from e
        max_tokens = _optional_number(body, "max_tokens", int)
        if max_tokens is not None and max_tokens <= 0:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "'max_tokens' must be positive"
            )
        skip_cache_store = body.get("skip_cache_store", False)
        if not isinstance(skip_cache_store, bool):
            raise BridgeError(
                ErrorCode.INVALID_REQUEST, "'skip_cache_store' must be a boolean"
            )
        expanded_prompt_len = _optional_number(body, "expanded_prompt_len", int)
        if expanded_prompt_len is not None and expanded_prompt_len < len(token_ids):
            raise BridgeError(
                ErrorCode.INVALID_REQUEST,
                "'expanded_prompt_len' is shorter than 'token_ids'",
            )
        min_pixels = _optional_number(body, "min_pixels", int)
        max_pixels = _optional_number(body, "max_pixels", int)
        if any(p is not None and p <= 0 for p in (min_pixels, max_pixels)) or (
            None not in (min_pixels, max_pixels) and min_pixels > max_pixels
        ):
            raise BridgeError(
                ErrorCode.INVALID_REQUEST,
                "'min_pixels' and 'max_pixels' must be positive, min at most max",
            )
        return cls(
            request_id=request_id,
            token_ids=token_ids,
            max_tokens=max_tokens,
            temperature=_optional_number(body, "temperature", float),
            top_p=_optional_number(body, "top_p", float),
            top_k=_optional_number(body, "top_k", int),
            min_p=_optional_number(body, "min_p", float),
            repetition_penalty=_optional_number(body, "repetition_penalty", float),
            presence_penalty=_optional_number(body, "presence_penalty", float),
            frequency_penalty=_optional_number(body, "frequency_penalty", float),
            seed=_optional_number(body, "seed", int),
            stop_token_ids=tuple(stop_token_ids),
            skip_cache_store=skip_cache_store,
            images=_images(body),
            expanded_prompt_len=expanded_prompt_len,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )


@dataclass(frozen=True)
class EngineOutput:
    new_token_ids: list[int]
    finished: bool
    finish_reason: str | None  # stop | length | abort when finished
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


class EngineRequest(Protocol):
    def outputs(self) -> AsyncIterator[EngineOutput]:
        """Engine outputs until the finished one. Raises BridgeError on engine errors."""
        ...

    async def abort(self) -> None:
        ...


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    state: str
    # oMLX paged cache block size; None while unloaded or without an SSD cache.
    block_size: int | None = None
    kv_available: bool = False
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    context_length: int | None = None
    # Most draft tokens per verify cycle while the loaded model runs oMLX's
    # multi-token prediction; None without it.
    mtp_depth: int | None = None
    # The loaded engine accepts images in generate requests.
    vision: bool = False


class EngineFacade(Protocol):
    def model_info(self) -> ModelInfo:
        ...

    def engine_stats(self) -> dict[str, Any]:
        ...

    def image_stats(self) -> dict[str, int]:
        ...

    def open(self, spec: GenerateSpec) -> AbstractAsyncContextManager[EngineRequest]:
        """Submit ``spec``; leaving the context aborts the request unless it finished."""
        ...

    async def ensure_loaded(self) -> None:
        ...


@dataclass
class _Active:
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    by_client: bool = False


def _usage(output: EngineOutput | None, prompt_tokens: int) -> dict[str, int]:
    if output is None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
    return {
        "prompt_tokens": output.prompt_tokens or prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "cached_tokens": output.cached_tokens,
    }


class GenerateService:
    def __init__(
        self, facade: EngineFacade, incarnation: str, heartbeat_s: float
    ) -> None:
        self._facade = facade
        self._incarnation = incarnation
        self._heartbeat_s = heartbeat_s
        self._active: dict[str, _Active] = {}
        self._closing = False

    @property
    def active_count(self) -> int:
        return len(self._active)

    def cancel(self, request_id: str) -> bool:
        active = self._active.get(request_id)
        if active is None:
            return False
        active.by_client = True
        active.cancelled.set()
        return True

    def begin_shutdown(self) -> None:
        self._closing = True
        for active in self._active.values():
            active.cancelled.set()

    async def stream(self, spec: GenerateSpec) -> AsyncIterator[bytes]:
        yield encode_frame(
            {
                "t": Frame.START,
                "v": PROTOCOL_VERSION,
                "rid": spec.request_id,
                "inc": self._incarnation,
            }
        )
        if self._closing:
            yield self._end_error(
                BridgeError(ErrorCode.SHUTTING_DOWN, "bridge is shutting down"),
                None,
                spec,
            )
            return
        superseded = self._active.get(spec.request_id)
        if (
            superseded is not None
        ):  # a retry of the same request id replaces the old one
            superseded.cancelled.set()
        active = _Active()
        self._active[spec.request_id] = active
        try:
            async for frame in self._run(spec, active):
                yield frame
        finally:
            if self._active.get(spec.request_id) is active:
                del self._active[spec.request_id]

    async def _run(self, spec: GenerateSpec, active: _Active) -> AsyncIterator[bytes]:
        last: EngineOutput | None = None
        try:
            async with self._facade.open(spec) as request:
                outputs = request.outputs().__aiter__()
                next_output: asyncio.Future[EngineOutput] = asyncio.ensure_future(
                    outputs.__anext__()
                )
                cancel_wait: asyncio.Future[Any] | None = asyncio.ensure_future(
                    active.cancelled.wait()
                )
                try:
                    while True:
                        waiting = (
                            {next_output}
                            if cancel_wait is None
                            else {next_output, cancel_wait}
                        )
                        done, _ = await asyncio.wait(
                            waiting,
                            timeout=self._heartbeat_s,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not done:
                            yield encode_frame({"t": Frame.HEARTBEAT})
                            continue
                        if cancel_wait is not None and cancel_wait in done:
                            cancel_wait = None
                            await request.abort()  # the engine then finishes the stream with "abort"
                        if next_output not in done:
                            continue
                        try:
                            output = next_output.result()
                        except StopAsyncIteration:
                            raise BridgeError(
                                ErrorCode.ENGINE_ERROR,
                                "engine stream ended without finishing",
                            ) from None
                        last = output
                        if output.new_token_ids:
                            yield encode_frame(
                                {"t": Frame.TOKENS, "ids": output.new_token_ids}
                            )
                        if output.finished:
                            yield self._end(output, active, spec)
                            return
                        next_output = asyncio.ensure_future(outputs.__anext__())
                finally:
                    for task in (next_output, cancel_wait):
                        if task is not None and not task.done():
                            task.cancel()
        except BridgeError as e:
            yield self._end_error(e, last, spec)
        except asyncio.CancelledError:
            raise
        # Reported to the client as an internal error.
        except Exception as e:  # noqa: BLE001
            logger.exception("generate %s failed", spec.request_id)
            yield self._end_error(BridgeError(ErrorCode.INTERNAL, str(e)), last, spec)

    def _end(self, output: EngineOutput, active: _Active, spec: GenerateSpec) -> bytes:
        if output.finish_reason == FinishReason.ABORT and not active.by_client:
            code = (
                ErrorCode.SHUTTING_DOWN
                if self._closing
                else ErrorCode.ABORTED_BY_ENGINE
            )
            return self._end_error(
                BridgeError(code, "oMLX aborted the request"), output, spec
            )
        finish = (
            output.finish_reason
            if output.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
            else FinishReason.ABORT
        )
        return encode_frame(
            {
                "t": Frame.END,
                "finish": finish,
                "usage": _usage(output, len(spec.token_ids)),
                "error": None,
            }
        )

    @staticmethod
    def _end_error(
        error: BridgeError, last: EngineOutput | None, spec: GenerateSpec
    ) -> bytes:
        return encode_frame(
            {
                "t": Frame.END,
                "finish": FinishReason.ERROR,
                "usage": _usage(last, len(spec.token_ids)),
                "error": error.payload(),
            }
        )


class KvStreamService:
    def __init__(
        self, log: KvEventLog, kv_state: Callable[[], str], heartbeat_s: float
    ) -> None:
        self._log = log
        self._kv_state = kv_state
        self._heartbeat_s = heartbeat_s
        self._closing = False

    def begin_shutdown(self) -> None:
        self._closing = True
        self._log.wake_all()

    def _hello(self, state: str, block_size: int | None) -> bytes:
        return encode_frame(
            {
                "t": Frame.HELLO,
                "v": PROTOCOL_VERSION,
                "inc": self._log.incarnation,
                "seq": self._log.last_seq,
                "floor": self._log.floor,
                "block_size": block_size,
                "state": state,
                "hb_s": self._heartbeat_s,
            }
        )

    async def stream(
        self, incarnation: str | None, after: int, block_size: int | None
    ) -> AsyncIterator[bytes]:
        state = self._kv_state()
        yield self._hello(state, block_size)
        if state == KvState.DISABLED:
            yield encode_frame({"t": Frame.RESET, "reason": ResetReason.DISABLED})
            return
        if incarnation != self._log.incarnation:
            yield encode_frame({"t": Frame.RESET, "reason": ResetReason.INCARNATION})
            return
        cursor = after
        while not self._closing:
            reason, frames = self._log.read_after(cursor)
            if reason is not None:
                yield encode_frame({"t": Frame.RESET, "reason": reason})
                return
            if frames:
                for frame in frames:
                    yield frame
                cursor += len(frames)
                continue
            if not await self._log.wait_beyond(cursor, self._heartbeat_s):
                yield encode_frame({"t": Frame.HEARTBEAT, "seq": self._log.last_seq})
        yield encode_frame({"t": Frame.RESET, "reason": ResetReason.SHUTDOWN})

    async def snapshot(self, block_size: int | None) -> AsyncIterator[bytes]:
        state = self._kv_state()
        snap = self._log.snapshot()
        blocks = snap.blocks if state == KvState.LIVE else []
        yield encode_frame(
            {
                "t": Frame.SNAPSHOT_BEGIN,
                "v": PROTOCOL_VERSION,
                "inc": snap.incarnation,
                "as_of": snap.as_of,
                "block_size": snap.block_size or block_size,
                "count": len(blocks),
                "state": state,
            }
        )
        for start in range(0, len(blocks), MAX_BLOCKS_PER_FRAME):
            chunk = blocks[start : start + MAX_BLOCKS_PER_FRAME]
            yield encode_frame(
                {"t": Frame.SNAPSHOT_BLOCKS, "blocks": [b.to_wire() for b in chunk]}
            )
            await asyncio.sleep(
                0
            )  # base64 encoding is CPU-bound; let other requests run
        yield encode_frame(
            {"t": Frame.SNAPSHOT_END, "as_of": snap.as_of, "count": len(blocks)}
        )


class BridgeRuntime:
    """Everything the routes need, wired for one served model."""

    def __init__(
        self,
        cfg: BridgeConfig,
        facade: EngineFacade,
        log: KvEventLog,
        tracker: KvTracker,
    ) -> None:
        self.cfg = cfg
        self.facade = facade
        self.log = log
        self.tracker = tracker
        self.generate = GenerateService(
            facade, log.incarnation, cfg.generate_heartbeat_s
        )
        self.kv = KvStreamService(log, self.kv_state, cfg.kv_heartbeat_s)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._load_task: asyncio.Task[None] | None = None

    def kv_state(self) -> str:
        if not self.cfg.kv_events:
            return KvState.DISABLED
        info = self.facade.model_info()
        if info.state == ModelState.READY and not info.kv_available:
            return (
                KvState.DISABLED
            )  # oMLX runs without an SSD cache, so nothing is reusable
        return self.tracker.state

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.log.bind_loop(self.loop)
        if self.cfg.kv_events:
            self.tracker.start()
        if self.cfg.preload:
            self.request_load()

    async def stop(self) -> None:
        self.begin_shutdown()
        task, self._load_task = self._load_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.tracker.stop()

    def begin_shutdown(self) -> None:
        self.generate.begin_shutdown()
        self.kv.begin_shutdown()

    def request_load(self) -> None:
        """Load the model in the background unless a load is already running."""
        if self._load_task is None or self._load_task.done():
            self._load_task = asyncio.ensure_future(self._load())

    async def _load(self) -> None:
        try:
            await self.facade.ensure_loaded()
        except asyncio.CancelledError:
            raise
        # Reported through /info; retried on the next request.
        except Exception:  # noqa: BLE001
            logger.exception("loading model %s failed", self.cfg.model_id)

    def check_ready(self) -> None:
        """Raise the error a generate request would hit right now, if any."""
        info = self.facade.model_info()
        if info.state == ModelState.READY:
            return
        if info.state == ModelState.MISSING:
            raise BridgeError(
                ErrorCode.MODEL_NOT_FOUND,
                f"oMLX does not serve model {self.cfg.model_id!r}",
            )
        if info.state == ModelState.UNLOADED:
            self.request_load()
        raise BridgeError(
            ErrorCode.MODEL_NOT_READY, f"model {self.cfg.model_id!r} is {info.state}"
        )

    def info(self) -> dict[str, Any]:
        info = self.facade.model_info()
        return {
            "v": PROTOCOL_VERSION,
            "hash_version": HASH_VERSION,
            "incarnation": self.log.incarnation,
            "model": {
                "id": info.model_id,
                "state": info.state,
                "block_size": info.block_size,
                "max_num_seqs": info.max_num_seqs,
                "max_num_batched_tokens": info.max_num_batched_tokens,
                "context_length": info.context_length,
                "mtp_depth": info.mtp_depth,
                "vision": info.vision,
            },
            "kv": {
                "state": self.kv_state(),
                "seq": self.log.last_seq,
                "floor": self.log.floor,
            },
        }

    def stats(self) -> dict[str, Any]:
        info = self.info()
        return {
            **info,
            "engine": self.facade.engine_stats(),
            # The effective state from info() wins over the tracker's own state,
            # which knows nothing about --no-kv-events or a missing SSD cache.
            "kv": {**self.tracker.stats(), **info["kv"]},
            "images": self.facade.image_stats(),
            "bridge": {"active_requests": self.generate.active_count},
        }
