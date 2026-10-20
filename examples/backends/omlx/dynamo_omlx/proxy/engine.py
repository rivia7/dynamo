# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo ``LLMEngine`` that serves a remote oMLX bridge.

The Dynamo frontend tokenizes and applies the chat template; this worker
forwards token ids and any image files to the Mac, streams token ids back,
mirrors oMLX's prefix cache into the KV router, and exits (deregistering)
when the Mac stays unreachable so the frontend stops routing to it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from dynamo_omlx.protocol import (
    HASH_VERSION,
    PROTOCOL_VERSION,
    ErrorCode,
    FinishReason,
    Frame,
    KvState,
)
from dynamo_omlx.proxy.args import ProxyConfig, parse_args
from dynamo_omlx.proxy.bridge_client import (
    BridgeClient,
    BridgeRequestError,
    BridgeUnavailable,
)
from dynamo_omlx.proxy.images import ImageLoader
from dynamo_omlx.proxy.kv_reconciler import KvReconciler
from dynamo_omlx.proxy.kv_sink import KvPublishSink
from dynamo_omlx.proxy.kv_sync import KvSync, KvSyncFatal
from dynamo_omlx.proxy.metrics import ProxyMetrics
from dynamo_omlx.proxy.request_map import RequestRejected, build_generate_body

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    LlmRegistration,
)
from dynamo.common.backend.health_check import build_health_check_payload, is_probe
from dynamo.common.backend.metrics import register_engine_registry
from dynamo.common.backend.publisher import ComponentSnapshot, KvEventSource, PushSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import (
    CannotConnect,
    EngineShutdown,
    InvalidArgument,
    StreamIncomplete,
    Unknown,
)

logger = logging.getLogger(__name__)


def dynamo_error(code: str, message: str) -> Exception:
    """Map a bridge error code to the Dynamo exception the frontend expects.

    CannotConnect and EngineShutdown let the frontend migrate the request to
    another worker; InvalidArgument is returned to the client.
    """
    if code in (ErrorCode.INVALID_REQUEST, ErrorCode.PREFILL_MEMORY_EXCEEDED):
        return InvalidArgument(message)
    if code in (
        ErrorCode.MODEL_NOT_FOUND,
        ErrorCode.MODEL_NOT_READY,
        ErrorCode.OVERLOADED,
    ):
        return CannotConnect(message)
    if code in (ErrorCode.ABORTED_BY_ENGINE, ErrorCode.SHUTTING_DOWN):
        return EngineShutdown(message)
    return Unknown(message)


_CANCELLED = "cancelled"


def _final_chunk(finish_reason: str, **extra: Any) -> GenerateChunk:
    return {"token_ids": [], "index": 0, "finish_reason": finish_reason, **extra}


def _end_outcome(frame: dict[str, Any], context: Context) -> str:
    """Finish reason for an ``end`` frame; raises the mapped error for failures."""
    finish = frame.get("finish")
    if finish in (FinishReason.STOP, FinishReason.LENGTH):
        return finish
    # A request the client cancelled ends as cancelled even if the engine
    # reported the abort as an error.
    if finish == FinishReason.ABORT or context.is_stopped():
        return _CANCELLED
    error = frame.get("error") or {}
    raise dynamo_error(
        error.get("code", ErrorCode.ENGINE_ERROR),
        error.get("message", "oMLX request failed"),
    )


def _completion_usage(usage: dict[str, Any], prompt_len: int) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens") or prompt_len)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": int(usage.get("cached_tokens") or 0)
        },
    }


class OmlxProxyEngine(LLMEngine):
    def __init__(self, cfg: ProxyConfig) -> None:
        self.cfg = cfg
        self._client = BridgeClient(cfg.bridge_url, cfg.bridge_api_key)
        self._images = ImageLoader(
            max_bytes=cfg.max_image_bytes,
            timeout_s=cfg.image_fetch_timeout_s,
            allow_internal=cfg.allow_internal_image_urls,
        )
        # The Mac's loaded model takes images; refreshed by every stats poll.
        self._vision = False
        self._metrics = ProxyMetrics()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._block_size: int | None = None
        self._context_length: int | None = cfg.context_length
        self._total_kv_blocks: int | None = None
        self._kv_enabled = False
        self._reconciler: KvReconciler | None = None
        self._sink: KvPublishSink | None = None
        self._sync: KvSync | None = None
        self._snapshot_publisher: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        # context id -> tokens the request occupies on the Mac (prompt + generated)
        self._inflight: dict[str, int] = {}
        self._stopping = False
        self._metrics.inflight = lambda: len(self._inflight)

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[OmlxProxyEngine, WorkerConfig]:
        cfg, runtime_cfg = parse_args(argv)
        worker_config = WorkerConfig.from_runtime_config(
            runtime_cfg,
            model_name=cfg.model_path,
            served_model_name=cfg.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return cls(cfg), worker_config

    # -- lifecycle ----------------------------------------------------------------

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id
        self._loop = asyncio.get_running_loop()
        await self._client.start()
        await self._images.start()
        info = await self._wait_for_bridge()
        model = info["model"]
        self._vision = bool(model.get("vision"))
        self._block_size = model.get("block_size")
        if (
            self.cfg.expect_block_size is not None
            and self._block_size != self.cfg.expect_block_size
        ):
            raise RuntimeError(
                f"the Mac's oMLX block size is {self._block_size}, expected {self.cfg.expect_block_size}"
            )
        if self._context_length is None:
            self._context_length = model.get("context_length")
        max_num_seqs = model.get("max_num_seqs")
        self._total_kv_blocks = self.cfg.total_kv_blocks
        if self._total_kv_blocks is None and None not in (
            max_num_seqs,
            self._context_length,
            self._block_size,
        ):
            self._total_kv_blocks = max_num_seqs * math.ceil(
                self._context_length / self._block_size
            )
        kv_state = info["kv"]["state"]
        self._kv_enabled = (
            self.cfg.kv_events
            and kv_state == KvState.LIVE
            and self._block_size is not None
        )
        if self._kv_enabled:
            self._reconciler = KvReconciler(self._block_size)
        else:
            logger.warning(
                "KV events are off (proxy flag %s, bridge KV state %s)",
                self.cfg.kv_events,
                kv_state,
            )
        self._tasks.append(asyncio.ensure_future(self._stats_loop()))
        logger.info(
            "oMLX bridge %s ready: block size %s, context %s, max_num_seqs %s, "
            "images %s (%s to %s pixels)",
            self.cfg.bridge_url,
            self._block_size,
            self._context_length,
            max_num_seqs,
            "on" if self._vision else "off",
            self.cfg.image_min_pixels,
            self.cfg.image_max_pixels,
        )
        return EngineConfig(
            model=self.cfg.model_path,
            served_model_name=self.cfg.served_model_name,
            llm=LlmRegistration(
                context_length=self._context_length,
                kv_cache_block_size=self._block_size,
                total_kv_blocks=self._total_kv_blocks,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=model.get("max_num_batched_tokens"),
            ),
        )

    async def _wait_for_bridge(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + self.cfg.startup_timeout_s
            if self.cfg.startup_timeout_s > 0
            else None
        )
        delay = 1.0
        while True:
            try:
                info = await self._client.info()
            except BridgeUnavailable as e:
                logger.info(
                    "waiting for the oMLX bridge at %s: %s", self.cfg.bridge_url, e
                )
            except BridgeRequestError as e:
                raise RuntimeError(f"the oMLX bridge refused the proxy: {e}") from e
            else:
                if (
                    info.get("v") != PROTOCOL_VERSION
                    or info.get("hash_version") != HASH_VERSION
                ):
                    raise RuntimeError(
                        f"bridge speaks protocol {info.get('v')} / hash {info.get('hash_version')}; "
                        f"this proxy needs {PROTOCOL_VERSION} / {HASH_VERSION}"
                    )
                state = info["model"]["state"]
                if state == "ready":
                    return info
                if state == "missing":
                    raise RuntimeError(
                        f"the bridge does not serve model {info['model']['id']!r}"
                    )
                logger.info(
                    "waiting for oMLX to load %s (state: %s)",
                    info["model"]["id"],
                    state,
                )
            if deadline is not None and loop.time() > deadline:
                raise RuntimeError(
                    f"timed out waiting for the oMLX bridge at {self.cfg.bridge_url}"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

    async def cleanup(self) -> None:
        self._stopping = True
        if self._sync is not None:
            self._sync.stop()
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        sink, self._sink = self._sink, None
        if sink is not None:
            sink.stop()
        await self._images.close()
        await self._client.close()

    def _shutdown_worker(self, reason: str) -> None:
        """Exit through the Rust worker's graceful path so discovery deregisters us."""
        if self._stopping:
            return
        self._stopping = True
        logger.error("shutting down the oMLX proxy: %s", reason)
        os.kill(os.getpid(), signal.SIGTERM)

    # -- generation ---------------------------------------------------------------------

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        request_id = context.id() or uuid.uuid4().hex
        probe = is_probe(request)
        try:
            body, images = build_generate_body(
                request,
                request_id=request_id,
                context_length=self._context_length,
                max_output_tokens=self.cfg.max_output_tokens,
                probe=probe,
            )
            if images:
                # A bridge without image support would take the placeholders for text.
                if not self._vision:
                    raise RequestRejected("the model on the Mac takes no images")
                body["images"] = await self._images.load(images)
                # The Mac sizes images within the range the frontend counted with.
                for key, value in (
                    ("min_pixels", self.cfg.image_min_pixels),
                    ("max_pixels", self.cfg.image_max_pixels),
                ):
                    if value is not None:
                        body[key] = value
        except RequestRejected as e:
            raise InvalidArgument(str(e)) from e
        prompt_len = body.get("expanded_prompt_len") or len(body["token_ids"])
        self._metrics.images += len(images)
        self._inflight[request_id] = prompt_len
        outcome, usage, started = "error", None, False
        try:
            async with self._client.generate(
                body, self.cfg.generate_read_timeout_s
            ) as frames:
                async for frame in frames:
                    kind = frame.get("t")
                    if kind == Frame.TOKENS:
                        started = True
                        token_ids = frame["ids"]
                        self._inflight[request_id] += len(token_ids)
                        yield {"token_ids": token_ids, "index": 0}
                        if context.is_stopped():
                            # Leaving the context closes the stream, which aborts
                            # the request on the Mac.
                            outcome = _CANCELLED
                            yield _final_chunk(_CANCELLED)
                            return
                    elif kind == Frame.END:
                        usage = frame.get("usage") or {}
                        outcome = _end_outcome(frame, context)
                        yield _final_chunk(
                            outcome,
                            completion_usage=_completion_usage(usage, prompt_len),
                        )
                        return
                raise StreamIncomplete(
                    "the oMLX bridge closed the stream before finishing"
                )
        except BridgeRequestError as e:
            raise dynamo_error(e.code, e.message) from e
        except BridgeUnavailable as e:
            if context.is_stopped():
                outcome = _CANCELLED
                yield _final_chunk(_CANCELLED)
                return
            # Before the first token the request never reached the engine.
            raise (
                StreamIncomplete(str(e)) if started else CannotConnect(str(e))
            ) from e
        finally:
            self._inflight.pop(request_id, None)
            if not probe:
                self._metrics.record_request(outcome, usage)

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if request_id is None or request_id not in self._inflight:
            return
        try:
            await self._client.cancel(request_id)
        except (BridgeUnavailable, BridgeRequestError) as e:
            # Closing the stream also aborts it on the Mac, so this is best effort.
            logger.debug("cancel of %s failed: %s", request_id, e)

    async def health_check_payload(self) -> dict[str, Any] | None:
        return build_health_check_payload(bos_token_id=self.cfg.bos_token_id)

    # -- KV events and metrics ---------------------------------------------------------------

    async def kv_event_sources(self) -> list[KvEventSource]:
        if not self._kv_enabled:
            return []
        return [PushSource(on_ready=self._on_publisher_ready, dp_rank=0)]

    def _on_publisher_ready(self, publisher: Any) -> None:
        # Runs on a Rust runtime thread; hand the rest to the event loop.
        self._sink = KvPublishSink(
            publisher, self._block_size, on_error=self._on_publish_error
        )
        self._sink.start()
        self._loop.call_soon_threadsafe(self._start_kv_sync)

    def _start_kv_sync(self) -> None:
        if self._stopping or self._sink is None:
            return
        self._sync = KvSync(self._client, self._reconciler, self._sink.submit)
        self._metrics.kv_sync_state = (
            lambda: self._sync.state if self._sync is not None else None
        )
        self._metrics.kv_counters = self._kv_counters
        self._tasks.append(asyncio.ensure_future(self._run_kv_sync()))

    async def _run_kv_sync(self) -> None:
        try:
            await self._sync.run()
        except KvSyncFatal as e:
            self._shutdown_worker(str(e))
        # A broken KV mirror must not keep serving stale routing state.
        except Exception:  # noqa: BLE001
            logger.exception("KV sync failed")
            self._shutdown_worker("KV sync failed")

    def _on_publish_error(self, error: BaseException) -> None:
        # The router's view no longer matches the reconciler; restart cleanly.
        self._loop.call_soon_threadsafe(
            self._shutdown_worker, f"publishing KV events failed: {error}"
        )

    def _kv_counters(self) -> dict[str, float]:
        counters: dict[str, float] = {}
        if self._sync is not None:
            counters.update(self._sync.counters)
        if self._reconciler is not None:
            counters["announced_blocks"] = len(self._reconciler)
            counters["skipped_blocks"] = self._reconciler.skipped_blocks
        if self._sink is not None:
            counters["published_events"] = self._sink.published_events
            counters["publish_errors"] = self._sink.errors
        return counters

    def component_metrics_dp_ranks(self) -> list[int]:
        return [0]

    def attach_snapshot_publisher(self, publisher: Any) -> None:
        self._snapshot_publisher = publisher

    async def register_prometheus(self, metrics: Any) -> None:
        register_engine_registry(metrics, self._metrics.registry)

    def _publish_component_snapshot(self) -> None:
        if (
            self._snapshot_publisher is None
            or not self._block_size
            or not self._total_kv_blocks
        ):
            return
        used = sum(
            math.ceil(tokens / self._block_size) for tokens in self._inflight.values()
        )
        self._snapshot_publisher.publish(
            0,
            ComponentSnapshot(
                kv_used_blocks=used,
                kv_total_blocks=self._total_kv_blocks,
                gpu_cache_usage=min(1.0, used / self._total_kv_blocks),
                dp_rank=0,
                kv_cache_hit_rate=self._metrics.cache_hit_rate(),
            ),
        )

    async def _stats_loop(self) -> None:
        loop = asyncio.get_running_loop()
        last_ok = loop.time()
        while not self._stopping:
            started = loop.time()
            try:
                stats = await self._client.stats()
            except (BridgeUnavailable, BridgeRequestError) as e:
                self._metrics.bridge_up = False
                down_for = loop.time() - last_ok
                if down_for > self.cfg.mac_down_grace_s:
                    self._shutdown_worker(
                        f"the oMLX bridge has been unreachable for {down_for:.0f}s: {e}"
                    )
                    return
            else:
                last_ok = loop.time()
                self._metrics.bridge_up = True
                self._metrics.bridge_rtt_s = last_ok - started
                self._metrics.bridge_stats = stats
                self._vision = bool((stats.get("model") or {}).get("vision"))
                block_size = (stats.get("model") or {}).get("block_size")
                if (
                    self._kv_enabled
                    and block_size is not None
                    and block_size != self._block_size
                ):
                    self._shutdown_worker(
                        f"the Mac's block size changed from {self._block_size} to {block_size}"
                    )
                    return
                self._publish_component_snapshot()
            await asyncio.sleep(self.cfg.stats_interval_s)
