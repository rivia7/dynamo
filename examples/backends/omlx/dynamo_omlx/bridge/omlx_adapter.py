# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All of the bridge's use of oMLX internals (``__main__`` only calls ``omlx.cli``).

It installs the bridge into an unmodified ``omlx serve`` process:

* ``omlx.server.init_server`` is wrapped to mount the routes, pin the model
  and hook the app lifespan (``omlx.cli`` imports ``init_server`` lazily, so
  the wrapper is picked up).
* ``BlockAwarePrefixCache.store_cache`` and the ``PagedSSDCacheManager``
  removal paths are wrapped at class level to feed the KV tracker.
* ``uvicorn.Server.handle_exit`` is wrapped so long-lived streams end when the
  server shuts down instead of blocking it.

Everything here relies on oMLX internals, so :func:`probe` checks each one and
``python -m dynamo_omlx.bridge --check`` reports the result. Wrappers catch
their own failures (an observer must never break oMLX's cache paths) and feed
them to the tracker's circuit breaker.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import inspect
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dynamo_omlx.bridge.config import BridgeConfig
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.kv_tracker import KvTracker
from dynamo_omlx.bridge.routes import build_router
from dynamo_omlx.bridge.service import (
    BridgeError,
    BridgeRuntime,
    EngineOutput,
    GenerateSpec,
    ModelInfo,
    ModelState,
)
from dynamo_omlx.protocol import API_PREFIX, ErrorCode
from omlx import server as omlx_server
from omlx.cache.paged_cache import compute_block_hash
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.engine_core import EngineCore
from omlx.engine_pool import EnginePool
from omlx.exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
    PrefillMemoryAbortedError,
    PrefillMemoryExceededError,
    SchedulerQueueFullError,
)
from omlx.request import SamplingParams

logger = logging.getLogger(__name__)

# The oMLX commit this adapter was written against (informational only).
TESTED_OMLX_COMMIT = "31b6b6a0"
_PATCHED = "_dynamo_omlx_patched"
_REQUEST_ID_PREFIX = "dyn-"
# RequestOutput.finish_reason of a request EngineCore.abort_request() ended.
_ABORT = "abort"

# RequestOutput.error_code -> protocol error code. Unknown codes are engine errors.
_OUTPUT_ERROR_CODES = {
    "prefill_memory_exceeded": ErrorCode.PREFILL_MEMORY_EXCEEDED,
    "prefill_memory_aborted": ErrorCode.OVERLOADED,
    "memory_admission_stalled": ErrorCode.OVERLOADED,
    "store_cache_admission_stalled": ErrorCode.OVERLOADED,
    "model_unloading": ErrorCode.ABORTED_BY_ENGINE,
    "client_disconnected": ErrorCode.ABORTED_BY_ENGINE,
}

_SAMPLING_FIELDS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "xtc_probability",
    "xtc_threshold",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "stop_token_ids",
    "seed",
)


def probe() -> list[str]:
    """Return every missing oMLX capability the bridge relies on."""
    problems: list[str] = []
    required = [
        (
            omlx_server,
            (
                "init_server",
                "app",
                "get_server_state",
                "get_sampling_params",
                "verify_api_key",
            ),
        ),
        (
            EnginePool,
            (
                "acquire",
                "get_entry",
                "set_pinned",
                "get_model_ids",
                "_resolve_scheduler_from_engine",
            ),
        ),
        (BlockAwarePrefixCache, ("store_cache",)),
        (
            PagedSSDCacheManager,
            (
                "_evict_tracked_until_size",
                "delete_block",
                "forget_block",
                "clear",
                "has_block",
                "get_stats_dict",
            ),
        ),
        (uvicorn.Server, ("handle_exit",)),
    ]
    for owner, names in required:
        for name in names:
            if not hasattr(owner, name):
                problems.append(f"{owner.__name__}.{name} is missing")
    parameters = [
        (
            BlockAwarePrefixCache.store_cache,
            ("tokens", "extra_keys", "extra_key_ranges", "_store_exact_terminal"),
        ),
        (compute_block_hash, ("parent_hash", "token_ids", "extra_keys", "model_name")),
        (
            EngineCore.add_request,
            ("prompt", "sampling_params", "request_id", "skip_cache_store"),
        ),
        (
            omlx_server.get_sampling_params,
            ("req_temperature", "req_top_p", "model_id", "req_top_k", "req_max_tokens"),
        ),
    ]
    for function, names in parameters:
        params = inspect.signature(function).parameters
        problems.extend(
            f"{function.__qualname__}() has no '{name}' parameter"
            for name in names
            if name not in params
        )
    fields = {f.name for f in dataclasses.fields(SamplingParams)}
    problems.extend(
        f"SamplingParams has no '{name}' field"
        for name in _SAMPLING_FIELDS
        if name not in fields
    )
    try:
        from omlx.model_settings import ModelSettings, ModelSettingsManager
        from omlx.patches import mlx_lm_mtp
    except ImportError as e:
        problems.append(f"oMLX's model settings or MTP patch do not import: {e}")
    else:
        for owner, names in (
            (ModelSettingsManager, ("get_settings", "set_settings")),
            (mlx_lm_mtp, ("is_mtp_active", "get_mtp_depth")),
        ):
            problems.extend(
                f"{owner.__name__}.{name} is missing"
                for name in names
                if not hasattr(owner, name)
            )
        fields = {f.name for f in dataclasses.fields(ModelSettings)}
        problems.extend(
            f"ModelSettings has no '{name}' field"
            for name in _MTP_SETTINGS
            if name not in fields
        )
    return problems


def install(cfg: BridgeConfig) -> BridgeRuntime:
    """Patch oMLX in this process; call before ``omlx.cli.main()``."""
    problems = probe()
    if problems:
        raise RuntimeError(
            "oMLX is incompatible with this bridge:\n  " + "\n  ".join(problems)
        )
    if getattr(omlx_server.init_server, _PATCHED, False):
        raise RuntimeError("the Dynamo bridge is already installed in this process")
    log = KvEventLog(
        ring_max_events=cfg.ring_max_events, ring_max_bytes=cfg.ring_max_bytes
    )
    tracker = KvTracker(
        log,
        ssd_provider=functools.partial(_current_ssd_manager, cfg.model_id),
        verify_block=_verify_block if cfg.verify_hashes else None,
        audit_interval_s=cfg.audit_interval_s,
    )
    runtime = BridgeRuntime(cfg, OmlxEngineFacade(cfg.model_id), log, tracker)
    _wrap_init_server(runtime)
    _wrap_uvicorn_exit(runtime)
    if cfg.kv_events:
        _wrap_store_cache(tracker, cfg.model_id)
        _wrap_ssd_manager(tracker)
    logger.info(
        "Dynamo bridge installed for model %s (KV events %s, tested against oMLX %s)",
        cfg.model_id,
        "on" if cfg.kv_events else "off",
        TESTED_OMLX_COMMIT,
    )
    return runtime


# -- engine access -------------------------------------------------------------------


def _pool() -> EnginePool | None:
    return omlx_server.get_server_state().engine_pool


def _entry(model_id: str) -> Any:
    """The model's EngineEntry, or None before init_server() or for unknown ids."""
    pool = _pool()
    return pool.get_entry(model_id) if pool is not None else None


def _scheduler(engine: Any) -> Any:
    return EnginePool._resolve_scheduler_from_engine(engine)


def _current_ssd_manager(model_id: str) -> Any:
    entry = _entry(model_id)
    if entry is None or entry.engine is None:
        return None
    scheduler = _scheduler(entry.engine)
    return scheduler.paged_ssd_cache_manager if scheduler is not None else None


# The per-model settings --mtp needs. oMLX added the ones --mtp-depth writes
# only in 0.7.0rc1, after Qwen3.8 support, so _apply_mtp checks for those.
_MTP_SETTINGS = ("mtp_enabled", "dflash_enabled")


def _mtp_depth() -> int | None:
    """Most draft tokens per verify cycle if the loaded model runs MTP, else None."""
    from omlx.patches import mlx_lm_mtp

    return mlx_lm_mtp.get_mtp_depth() if mlx_lm_mtp.is_mtp_active() else None


def _apply_mtp(cfg: BridgeConfig) -> None:
    """Write --mtp and --mtp-depth to oMLX's per-model settings, which the model
    load reads. They persist, and oMLX's admin page shows them."""
    changes: dict[str, Any] = {}
    if cfg.mtp is not None:
        changes["mtp_enabled"] = cfg.mtp
    if cfg.mtp_depth is not None:
        # A fixed depth would take precedence over the adaptive ceiling.
        changes.update(mtp_adaptive_max_depth=cfg.mtp_depth, mtp_fixed_depth=None)
    if not changes:
        return
    manager = omlx_server.get_server_state().settings_manager
    settings = manager.get_settings(cfg.model_id)
    missing = [name for name in changes if not hasattr(settings, name)]
    if missing:
        raise RuntimeError(
            f"this oMLX build has no {', '.join(missing)} model setting; "
            "drop --mtp-depth or upgrade oMLX"
        )
    if cfg.mtp and settings.dflash_enabled:
        raise RuntimeError(
            f"oMLX runs DFlash for {cfg.model_id}, which rules out MTP; "
            "turn DFlash off on oMLX's admin page first"
        )
    for name, value in changes.items():
        setattr(settings, name, value)
    manager.set_settings(cfg.model_id, settings)
    logger.info(
        "oMLX model settings for %s: %s",
        cfg.model_id,
        ", ".join(f"{name}={value}" for name, value in changes.items()),
    )


def _verify_block(
    parent_bid: bytes | None, tokens: Any, bid: bytes, model_name: str
) -> bool:
    return (
        compute_block_hash(
            parent_bid, list(tokens), extra_keys=None, model_name=model_name
        )
        == bid
    )


def _map_exception(exc: BaseException) -> BridgeError | None:
    # PrefillMemoryAbortedError subclasses PrefillMemoryExceededError: check it first.
    if isinstance(exc, PrefillMemoryAbortedError):
        return BridgeError(ErrorCode.OVERLOADED, str(exc))
    if isinstance(exc, PrefillMemoryExceededError):
        return BridgeError(ErrorCode.PREFILL_MEMORY_EXCEEDED, str(exc))
    if isinstance(exc, ModelNotFoundError):
        return BridgeError(ErrorCode.MODEL_NOT_FOUND, str(exc))
    if isinstance(exc, (ModelLoadingError, ModelUnavailableError)):
        return BridgeError(ErrorCode.MODEL_NOT_READY, str(exc))
    if isinstance(
        exc,
        (
            InsufficientMemoryError,
            ModelBusyError,
            ModelTooLargeError,
            SchedulerQueueFullError,
        ),
    ):
        return BridgeError(ErrorCode.OVERLOADED, str(exc))
    return None


class _OmlxRequest:
    def __init__(self, core: Any, request_id: str) -> None:
        self._core = core
        self._request_id = request_id
        self.done = False  # the engine no longer tracks the request

    async def outputs(self) -> AsyncIterator[EngineOutput]:
        async with contextlib.aclosing(
            self._core.stream_outputs(self._request_id)
        ) as stream:
            async for output in stream:
                # EngineCore.abort_request() ends the stream with an "abort" output
                # that also carries error="Request aborted" and no error code. It
                # is a cancellation, which the service classifies, not an error.
                if output.finish_reason == _ABORT and not output.error_code:
                    self.done = True
                    yield EngineOutput(
                        new_token_ids=list(output.new_token_ids),
                        finished=True,
                        finish_reason=_ABORT,
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        cached_tokens=output.cached_tokens,
                    )
                    return
                if output.error:
                    self.done = True
                    code = _OUTPUT_ERROR_CODES.get(
                        output.error_code, ErrorCode.ENGINE_ERROR
                    )
                    raise BridgeError(code, output.error)
                self.done = output.finished
                yield EngineOutput(
                    new_token_ids=list(output.new_token_ids),
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    cached_tokens=output.cached_tokens,
                )
                if output.finished:
                    return

    async def abort(self) -> None:
        await self._core.abort_request(self._request_id)


class OmlxEngineFacade:
    """:class:`dynamo_omlx.bridge.service.EngineFacade` over oMLX's engine pool."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def _require_pool(self) -> EnginePool:
        pool = _pool()
        if pool is None:
            raise BridgeError(
                ErrorCode.MODEL_NOT_READY, "oMLX has not initialized its engine pool"
            )
        return pool

    def model_info(self) -> ModelInfo:
        entry = _entry(self._model_id)
        if entry is None:
            return ModelInfo(model_id=self._model_id, state=ModelState.MISSING)
        context_length = entry.model_context_length
        if entry.engine is None:
            if entry.is_loading:
                state = ModelState.LOADING
            elif entry.load_failed:
                state = ModelState.FAILED
            else:
                state = ModelState.UNLOADED
            return ModelInfo(
                model_id=self._model_id, state=state, context_length=context_length
            )
        scheduler = _scheduler(entry.engine)
        if scheduler is None:  # e.g. the distributed engine; tokens only
            return ModelInfo(
                model_id=self._model_id,
                state=ModelState.READY,
                context_length=context_length,
                mtp_depth=_mtp_depth(),
            )
        config = scheduler.config
        kv_available = (
            scheduler.block_aware_cache is not None
            and scheduler.paged_ssd_cache_manager is not None
        )
        return ModelInfo(
            model_id=self._model_id,
            state=ModelState.READY,
            block_size=config.paged_cache_block_size if kv_available else None,
            kv_available=kv_available,
            max_num_seqs=config.max_num_seqs,
            # oMLX prefills one request at a time in prefill_step_size chunks.
            max_num_batched_tokens=config.prefill_step_size,
            context_length=context_length,
            mtp_depth=_mtp_depth(),
        )

    def engine_stats(self) -> dict[str, Any]:
        entry = _entry(self._model_id)
        if entry is None:
            return {"loaded": False}
        stats: dict[str, Any] = {
            "loaded": entry.engine is not None,
            "loading": entry.is_loading,
            "pinned": entry.is_pinned,
            "in_use": entry.in_use,
        }
        scheduler = _scheduler(entry.engine) if entry.engine is not None else None
        if scheduler is not None:
            stats["running"] = len(scheduler.running)
            stats["waiting"] = len(scheduler.waiting)
            ssd = scheduler.paged_ssd_cache_manager
            if ssd is not None:
                ssd_stats = ssd.get_stats_dict()
                for key in (
                    "total_size",
                    "max_size",
                    "num_files",
                    "hot_cache_size_bytes",
                    "hot_cache_max_bytes",
                ):
                    stats[f"ssd_{key}"] = ssd_stats.get(key)
        return stats

    async def ensure_loaded(self) -> None:
        async with self._require_pool().acquire(self._model_id):
            pass

    def _sampling_params(self, spec: GenerateSpec) -> SamplingParams:
        # oMLX resolves unset fields as request > per-model settings > server defaults.
        (
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_probability,
            xtc_threshold,
        ) = omlx_server.get_sampling_params(
            req_temperature=spec.temperature,
            req_top_p=spec.top_p,
            model_id=self._model_id,
            req_top_k=spec.top_k,
            req_repetition_penalty=spec.repetition_penalty,
            req_min_p=spec.min_p,
            req_presence_penalty=spec.presence_penalty,
            req_frequency_penalty=spec.frequency_penalty,
            req_max_tokens=spec.max_tokens,
        )
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=[],  # the Dynamo frontend owns stop strings
            stop_token_ids=list(spec.stop_token_ids),
            seed=spec.seed,
        )

    @asynccontextmanager
    async def open(self, spec: GenerateSpec) -> AsyncIterator[_OmlxRequest]:
        pool = self._require_pool()
        try:
            async with pool.acquire(self._model_id) as engine:
                core = getattr(engine, "_engine", None)
                if core is None:
                    raise BridgeError(
                        ErrorCode.ENGINE_ERROR,
                        f"{type(engine).__name__} does not accept token requests",
                    )
                request_id = await core.add_request(
                    prompt=list(spec.token_ids),
                    sampling_params=self._sampling_params(spec),
                    request_id=f"{_REQUEST_ID_PREFIX}{uuid.uuid4().hex}",
                    skip_cache_store=spec.skip_cache_store,
                )
                request = _OmlxRequest(core, request_id)
                try:
                    yield request
                finally:
                    if not request.done:
                        # Shielded: when the client disconnects this runs inside a
                        # cancelled scope, and the abort must still reach the engine.
                        await asyncio.shield(core.abort_request(request_id))
        except Exception as e:
            mapped = _map_exception(e)
            if mapped is None:
                raise
            raise mapped from e


# -- patches ------------------------------------------------------------------------


def _served(model_name: str, model_id: str) -> bool:
    # oMLX may suffix the cache namespace, e.g. "<model_id>:<ANE signature>".
    return model_name == model_id or model_name.startswith(model_id + ":")


def _wrap_init_server(runtime: BridgeRuntime) -> None:
    original = omlx_server.init_server
    cfg = runtime.cfg

    @functools.wraps(original)
    def init_server(*args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(original).bind(*args, **kwargs)
        scheduler_config = bound.arguments.get("scheduler_config")
        if cfg.kv_block_size is not None and scheduler_config is not None:
            scheduler_config.paged_cache_block_size = cfg.kv_block_size
        result = original(*args, **kwargs)
        _attach(runtime)
        return result

    setattr(init_server, _PATCHED, True)
    omlx_server.init_server = init_server


def _attach(runtime: BridgeRuntime) -> None:
    cfg = runtime.cfg
    pool = _pool()
    if pool is None or pool.get_entry(cfg.model_id) is None:
        available = ", ".join(pool.get_model_ids()) if pool is not None else "none"
        raise RuntimeError(
            f"oMLX has no model {cfg.model_id!r}; available: {available}"
        )
    # Before the model loads: preloading happens later, in the lifespan.
    _apply_mtp(cfg)
    if cfg.pin_model:
        pool.set_pinned(cfg.model_id, True)
    omlx_server.app.include_router(
        build_router(runtime, omlx_server.verify_api_key), prefix=API_PREFIX
    )

    router = omlx_server.app.router
    original_lifespan = router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[Any]:
        async with original_lifespan(app) as state:
            await runtime.start()
            try:
                yield state
            finally:
                await runtime.stop()

    router.lifespan_context = lifespan


def _wrap_uvicorn_exit(runtime: BridgeRuntime) -> None:
    original = uvicorn.Server.handle_exit

    @functools.wraps(original)
    def handle_exit(self: uvicorn.Server, *args: Any, **kwargs: Any) -> None:
        # uvicorn waits for open connections before running the lifespan shutdown;
        # ending the bridge streams first keeps a graceful stop from hanging.
        loop = runtime.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(runtime.begin_shutdown)
        original(self, *args, **kwargs)

    uvicorn.Server.handle_exit = handle_exit


def _wrap_store_cache(tracker: KvTracker, model_id: str) -> None:
    original = BlockAwarePrefixCache.store_cache
    signature = inspect.signature(original)

    @functools.wraps(original)
    def store_cache(self: BlockAwarePrefixCache, *args: Any, **kwargs: Any) -> Any:
        table = original(self, *args, **kwargs)
        try:
            if table is not None and _served(self.paged_cache.model_name, model_id):
                arguments = signature.bind(self, *args, **kwargs).arguments
                # Image-keyed chains hash differently from their tokens: skip them.
                if not arguments.get("extra_keys") and not arguments.get(
                    "extra_key_ranges"
                ):
                    tracker.on_store(
                        self.paged_cache,
                        table,
                        arguments["tokens"],
                        exclude_last=bool(arguments.get("_store_exact_terminal")),
                    )
        # An observer must never break oMLX's store path.
        except Exception:  # noqa: BLE001
            logger.exception("Dynamo bridge store_cache hook failed")
            tracker.record_hook_error()
        return table

    BlockAwarePrefixCache.store_cache = store_cache


def _wrap_ssd_manager(tracker: KvTracker) -> None:
    cls = PagedSSDCacheManager
    original_evict = cls._evict_tracked_until_size

    @functools.wraps(original_evict)
    def _evict_tracked_until_size(
        self: PagedSSDCacheManager, *args: Any, **kwargs: Any
    ) -> Any:
        evicted = original_evict(self, *args, **kwargs)
        try:
            # (source index, metadata); GDN sidecar metadata carries no block_hash.
            bids = [
                meta.block_hash
                for _, meta in evicted
                if isinstance(getattr(meta, "block_hash", None), bytes)
            ]
            if bids:
                tracker.on_removed(bids, "evicted")
        # An observer must never break oMLX's eviction path.
        except Exception:  # noqa: BLE001
            logger.exception("Dynamo bridge eviction hook failed")
            tracker.record_hook_error()
        return evicted

    cls._evict_tracked_until_size = _evict_tracked_until_size
    cls.delete_block = _removal_wrapper(cls.delete_block, tracker, "deleted")
    cls.forget_block = _removal_wrapper(cls.forget_block, tracker, "forgotten")

    original_clear = cls.clear

    @functools.wraps(original_clear)
    def clear(self: PagedSSDCacheManager, *args: Any, **kwargs: Any) -> Any:
        result = original_clear(self, *args, **kwargs)
        try:
            tracker.on_ssd_cleared()
        # An observer must never break oMLX's cache paths.
        except Exception:  # noqa: BLE001
            logger.exception("Dynamo bridge clear hook failed")
            tracker.record_hook_error()
        return result

    cls.clear = clear


def _removal_wrapper(
    original: Callable[..., Any], tracker: KvTracker, reason: str
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(
        self: PagedSSDCacheManager, block_hash: bytes, *args: Any, **kwargs: Any
    ) -> Any:
        result = original(self, block_hash, *args, **kwargs)
        try:
            if result:
                tracker.on_removed([block_hash], reason)
        # An observer must never break oMLX's cache paths.
        except Exception:  # noqa: BLE001
            logger.exception("Dynamo bridge %s hook failed", reason)
            tracker.record_hook_error()
        return result

    return wrapper
