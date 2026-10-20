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
* Image requests run the second half of oMLX's VLM preprocessing
  (``VLMBatchedEngine._prepare_vision_inputs``) on prompts Dynamo tokenized.

Everything here relies on oMLX internals, so :func:`probe` checks each one and
``python -m dynamo_omlx.bridge --check`` reports the result; :func:`probe_vision`
does the same for image input, which the bridge turns off when it fails.
Wrappers catch their own failures (an observer must never break oMLX's cache
paths) and feed them to the tracker's circuit breaker.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import inspect
import logging
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dynamo_omlx.bridge.config import BridgeConfig
from dynamo_omlx.bridge.event_log import KvEventLog
from dynamo_omlx.bridge.image_routing import (
    ImageRequests,
    ImageSpan,
    expand_placeholders,
    routing_tokens,
)
from dynamo_omlx.bridge.images import (
    cumulative_keys,
    decode_pixels,
    image_key,
    read_images,
)
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
from omlx.cache import paged_cache as omlx_paged_cache
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


def probe_vision() -> list[str]:
    """Return every missing oMLX capability that image input relies on."""
    try:
        from omlx.engine import vlm as omlx_vlm
        from omlx.utils import image as omlx_image
    except ImportError as e:
        return [f"oMLX's VLM engine does not import: {e}"]
    problems = [
        f"{owner.__name__}.{name} is missing"
        for owner, names in (
            (omlx_paged_cache, ("resolve_block_extra_keys",)),
            (omlx_vlm, ("_capture_vlm_position_state",)),
            (omlx_image, ("get_max_image_bytes",)),
        )
        for name in names
        if not hasattr(owner, name)
    ]
    parameters = [
        (
            EngineCore.add_request,
            (
                "vlm_inputs_embeds",
                "vlm_extra_kwargs",
                "vlm_image_hash",
                "vlm_cache_key_start",
                "vlm_cache_key_ranges",
            ),
        ),
        (BlockAwarePrefixCache.store_cache, ("request_id",)),
    ]
    resolve = getattr(omlx_paged_cache, "resolve_block_extra_keys", None)
    if resolve is not None:
        parameters.append(
            (
                resolve,
                (
                    "block_end",
                    "extra_keys",
                    "extra_key_token_start",
                    "extra_key_ranges",
                ),
            )
        )
    for function, names in parameters:
        params = inspect.signature(function).parameters
        problems.extend(
            f"{function.__qualname__}() has no '{name}' parameter"
            for name in names
            if name not in params
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
    vision_off = None
    if not cfg.images:
        vision_off = "the bridge runs with --no-images"
    elif vision_problems := probe_vision():
        vision_off = "this oMLX lacks " + "; ".join(vision_problems)
        logger.warning("Image input is off: %s", vision_off)
    log = KvEventLog(
        ring_max_events=cfg.ring_max_events, ring_max_bytes=cfg.ring_max_bytes
    )
    tracker = KvTracker(
        log,
        ssd_provider=functools.partial(_current_ssd_manager, cfg.model_id),
        verify_block=_verify_block if cfg.verify_hashes else None,
        audit_interval_s=cfg.audit_interval_s,
    )
    images = ImageRequests()
    facade = OmlxEngineFacade(cfg.model_id, images=images, vision_off=vision_off)
    runtime = BridgeRuntime(cfg, facade, log, tracker)
    _wrap_init_server(runtime)
    _wrap_uvicorn_exit(runtime)
    if cfg.kv_events:
        _wrap_store_cache(tracker, cfg.model_id, images)
        _wrap_ssd_manager(tracker)
    logger.info(
        "Dynamo bridge installed for model %s (KV events %s, images %s, tested against oMLX %s)",
        cfg.model_id,
        "on" if cfg.kv_events else "off",
        "off" if vision_off else "on",
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
    parent_bid: bytes | None,
    tokens: Any,
    bid: bytes,
    model_name: str,
    extra_keys: tuple[Any, ...] | None,
) -> bool:
    return (
        compute_block_hash(
            parent_bid, list(tokens), extra_keys=extra_keys, model_name=model_name
        )
        == bid
    )


# Model types whose get_input_embeddings reads pixel values only to run the
# vision tower, which cached features replace (mlx-vlm's Qwen3.5 family at the
# commit oMLX pins). A request whose images are all cached then needs no pixels;
# other models get them decoded, since they might ignore the cached features.
_FEATURES_ALONE_MODELS = frozenset({"qwen3_5", "qwen3_5_moe", "qwen4_exp"})


def _max_image_bytes() -> int:
    """oMLX's image size limit (``max_image_upload_size``); 0 means none."""
    from omlx.utils import image as omlx_image

    return omlx_image.get_max_image_bytes()


def _rows(value: Any) -> list[list[int]]:
    return [[int(v) for v in row] for row in getattr(value, "tolist", lambda: value)()]


@dataclasses.dataclass(frozen=True)
class _PreparedImages:
    token_ids: list[int]  # the prompt with every image expanded
    spans: tuple[ImageSpan, ...]
    request_kwargs: dict[str, Any]  # the vlm_* arguments of EngineCore.add_request


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

    def __init__(
        self,
        model_id: str,
        images: ImageRequests | None = None,
        vision_off: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._images = images if images is not None else ImageRequests()
        # Why image input is off for every model, or None.
        self._vision_off = vision_off
        self._image_counters: Counter[str] = Counter(
            dict.fromkeys(
                (
                    "requests",
                    "images",
                    "images_decoded",
                    "vision_cache_hits",
                    "vision_cache_misses",
                    "token_count_mismatches",
                ),
                0,
            )
        )
        # Set once the model rejects cached vision features.
        self._feature_cache_off = False

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
            vision=self._vision_problem(entry.engine) is None,
        )

    def _vision_problem(self, engine: Any) -> str | None:
        """Why ``engine`` takes no images from the bridge, or None if it does."""
        if self._vision_off is not None:
            return self._vision_off
        model = getattr(engine, "_vlm_model", None)
        processor = getattr(engine, "_processor", None)
        if model is None or processor is None:
            return "the model is not a vision-language model"
        if getattr(engine, "is_diffusion_model", False):
            return "oMLX serves the model as a diffusion model"
        # The bridge expands one placeholder per image by the processor's grid,
        # as the Qwen-VL family does; other layouts need their own expansion.
        if not hasattr(getattr(processor, "image_processor", None), "merge_size"):
            return "the model's image processor works without a patch grid"
        config = getattr(model, "config", None)
        if None in (
            getattr(config, "image_token_id", None),
            getattr(config, "vision_start_token_id", None),
        ):
            return "the model's config names no image or vision start token"
        return None

    def image_stats(self) -> dict[str, int]:
        return dict(self._image_counters)

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

    def _sampling_params(
        self, spec: GenerateSpec, max_tokens: int | None
    ) -> SamplingParams:
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
            req_max_tokens=max_tokens,
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

    def _fit_context(self, prompt_len: int, max_tokens: int | None) -> int | None:
        """Cap ``max_tokens`` by the room an expanded prompt leaves in the context.

        The proxy could only budget for the prompt Dynamo tokenized.
        """
        context = self.model_info().context_length
        if context is None:
            return max_tokens
        if prompt_len >= context:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST,
                f"the prompt has {prompt_len} tokens with its images, "
                f"but the model context is {context}",
            )
        room = context - prompt_len
        return room if max_tokens is None else min(max_tokens, room)

    def _feature_cache(self, engine: Any) -> Any:
        """oMLX's vision feature cache while the model can take its features, else None."""
        cache = getattr(engine, "_vision_cache", None)
        tower = getattr(engine._vlm_model, "vision_tower", None)
        if (
            self._feature_cache_off
            or cache is None
            or not getattr(engine, "_vision_cache_enabled", False)
            or tower is None
            # Cached features stand in for the tower's main output only.
            or getattr(tower, "deepstack_visual_indexes", None)
        ):
            return None
        return cache

    def _cached_vision(
        self, engine: Any, keys: list[str], merge: int
    ) -> list[tuple[Any, list[int]]] | None:
        """Each image's cached vision features and patch grid; None unless all are cached.

        Runs on the MLX executor: the cache loads entries from SSD.
        """
        cache = self._feature_cache(engine)
        if cache is None:
            return None
        found = []
        for key in keys:
            entry = cache.get(key, engine._model_name)
            if not isinstance(entry, list) or len(entry) != 2:
                return None
            features, grid = entry
            rows = [int(value) for value in grid.tolist()]
            if (
                len(rows) != 3
                or features.ndim != 2
                or features.shape[0] != rows[0] * rows[1] * rows[2] // merge**2
            ):
                return None
            found.append((features, rows))
        if len({features.shape[1] for features, _ in found}) != 1:
            return None
        return found

    async def _prepare_images(
        self, engine: Any, core: Any, spec: GenerateSpec
    ) -> _PreparedImages:
        """Expand the images of ``spec`` and compute what oMLX's VLM path needs.

        Mirrors ``VLMBatchedEngine._prepare_vision_inputs`` from tokenization on,
        but keeps Dynamo's tokens, so the router's view of the prompt holds.
        """
        problem = self._vision_problem(engine)
        if problem is not None:
            raise BridgeError(
                ErrorCode.INVALID_REQUEST,
                f"{self._model_id} takes no images: {problem}",
            )
        config = engine._vlm_model.config
        image_processor = engine._processor.image_processor
        merge = int(image_processor.merge_size)
        # The range the frontend counted image tokens with.
        limits = {
            name: value
            for name, value in (
                ("min_pixels", spec.min_pixels),
                ("max_pixels", spec.max_pixels),
            )
            if value is not None
        }
        # Reading files and pixels is CPU work; keep it off the event loop and
        # off the single MLX thread, which also runs generation.
        files = await asyncio.to_thread(
            read_images, [image.data for image in spec.images], _max_image_bytes()
        )
        keys = [
            image_key(
                file.digest,
                limits.get("min_pixels", getattr(image_processor, "min_pixels", None)),
                limits.get("max_pixels", getattr(image_processor, "max_pixels", None)),
            )
            for file in files
        ]
        loop = asyncio.get_running_loop()
        cached = await loop.run_in_executor(
            core._mlx_executor, self._cached_vision, engine, keys, merge
        )
        processed = None
        if cached is None or engine.model_type not in _FEATURES_ALONE_MODELS:
            pixels = await asyncio.to_thread(decode_pixels, files)
            processed = await asyncio.to_thread(
                functools.partial(image_processor, images=pixels, **limits)
            )
            self._image_counters["images_decoded"] += len(files)
        grids = (
            _rows(processed["image_grid_thw"])
            if processed is not None
            else [rows for _, rows in cached]
        )
        lengths = [t * h * w // merge**2 for t, h, w in grids]
        try:
            token_ids, starts = expand_placeholders(
                spec.token_ids, int(config.image_token_id), lengths
            )
        except ValueError as e:
            raise BridgeError(ErrorCode.INVALID_REQUEST, str(e)) from e
        # The model's mRoPE locates an image by the vision start token before it.
        if any(
            start == 0 or token_ids[start - 1] != config.vision_start_token_id
            for start in starts
        ):
            raise BridgeError(
                ErrorCode.INVALID_REQUEST,
                "an image placeholder does not follow the vision start token",
            )
        if spec.expanded_prompt_len is not None and spec.expanded_prompt_len != len(
            token_ids
        ):
            self._image_counters["token_count_mismatches"] += 1
            logger.warning(
                "request %s: Dynamo's frontend counted %d prompt tokens with the "
                "images, the model's processor %d; the KV router cannot match its "
                "blocks from the first image on",
                spec.request_id,
                spec.expanded_prompt_len,
                len(token_ids),
            )
        embeds, extra = await loop.run_in_executor(
            core._mlx_executor,
            functools.partial(
                self._embed, engine, token_ids, grids, cached, processed, keys
            ),
        )
        self._image_counters["requests"] += 1
        self._image_counters["images"] += len(spec.images)
        # oMLX keys every block that ends past an image's first token with the
        # identity of all images up to it, the boundaries its grid models use.
        ranges = list(zip(starts, cumulative_keys(keys)))
        return _PreparedImages(
            token_ids=token_ids,
            spans=tuple(
                ImageSpan(start=start, length=length, mm_hash=image.mm_hash)
                for start, length, image in zip(starts, lengths, spec.images)
            ),
            request_kwargs={
                "vlm_inputs_embeds": embeds,
                "vlm_extra_kwargs": extra,
                "vlm_image_hash": ranges[-1][1],
                "vlm_cache_key_start": ranges[0][0],
                "vlm_cache_key_ranges": ranges,
            },
        )

    def _embed(
        self,
        engine: Any,
        token_ids: list[int],
        grids: list[list[int]],
        cached: list[tuple[Any, list[int]]] | None,
        processed: dict[str, Any] | None,
        keys: list[str],
    ) -> tuple[Any, dict[str, Any]]:
        """Merged prompt embeddings and the language model's extra inputs.

        Uses the cached vision features when every image has them, encodes
        the images otherwise. Runs on oMLX's MLX executor, which serializes
        every MLX operation.
        """
        import mlx.core as mx
        from omlx.engine import vlm as omlx_vlm

        model = engine._vlm_model
        input_ids = mx.array([token_ids])
        grid = mx.array(grids)
        if processed is not None:
            pixel_values = mx.array(processed["pixel_values"])
        else:
            # The model reads pixels only to run the tower the features replace.
            pixel_values = mx.zeros((1, 1))
        if cached is not None:
            self._image_counters["vision_cache_hits"] += len(cached)
            features = (
                cached[0][0]
                if len(cached) == 1
                else mx.concatenate([features for features, _ in cached], axis=0)
            )
        else:
            features = self._encode(engine, pixel_values, grid, grids, keys)
        kwargs: dict[str, Any] = {"image_grid_thw": grid}
        if features is not None:
            kwargs["cached_image_features"] = features
        mask = mx.ones_like(input_ids)
        try:
            embedded = model.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            )
        except TypeError:
            if features is None or processed is None:
                raise
            logger.warning(
                "%s takes no cached vision features; encoding every image anew",
                self._model_id,
            )
            self._feature_cache_off = True
            del kwargs["cached_image_features"]
            embedded = model.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            )
        mx.eval(embedded.inputs_embeds)
        extra = {
            key: value
            for key, value in embedded.to_dict().items()
            if key != "inputs_embeds" and value is not None
        }
        # The language model keeps mRoPE positions as global state that the next
        # request overwrites; the request carries its own copy.
        omlx_vlm._capture_vlm_position_state(
            getattr(model, "language_model", None), extra
        )
        embeds = embedded.inputs_embeds
        # Free the vision encoder's Metal intermediates, as oMLX does.
        mx.synchronize()
        mx.clear_cache()
        return embeds, extra

    def _encode(
        self,
        engine: Any,
        pixel_values: Any,
        grid: Any,
        grids: list[list[int]],
        keys: list[str],
    ) -> Any:
        """Run the vision tower and cache each image's features with its grid.

        oMLX fills its vision feature cache only for model types it lists, which
        leave out Qwen4-Exp. Returns None to let the model run its tower itself.
        """
        import mlx.core as mx

        cache = self._feature_cache(engine)
        if cache is None:
            return None
        tower = engine._vlm_model.vision_tower
        result = tower(pixel_values.astype(tower.patch_embed.proj.weight.dtype), grid)
        features = result[0] if isinstance(result, tuple) else result
        merge = int(engine._processor.image_processor.merge_size)
        lengths = [t * h * w // merge**2 for t, h, w in grids]
        if features.ndim != 2 or features.shape[0] != sum(lengths):
            return None
        mx.eval(features)
        offset = 0
        for key, rows, length in zip(keys, grids, lengths):
            entry = [features[offset : offset + length], mx.array(rows, dtype=mx.int32)]
            cache.put(key, engine._model_name, entry)
            offset += length
        self._image_counters["vision_cache_misses"] += len(keys)
        return features

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
                request_id = f"{_REQUEST_ID_PREFIX}{uuid.uuid4().hex}"
                prompt, max_tokens, vision = list(spec.token_ids), spec.max_tokens, {}
                if spec.images:
                    prepared = await self._prepare_images(engine, core, spec)
                    prompt, vision = prepared.token_ids, prepared.request_kwargs
                    max_tokens = self._fit_context(len(prompt), max_tokens)
                    # Before add_request: oMLX may store the cache right after.
                    self._images.add(request_id, prepared.spans)
                request_id = await core.add_request(
                    prompt=prompt,
                    sampling_params=self._sampling_params(spec, max_tokens),
                    request_id=request_id,
                    skip_cache_store=spec.skip_cache_store,
                    **vision,
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


def _block_extra_keys(
    arguments: dict[str, Any], block_size: int, count: int
) -> list[tuple[Any, ...] | None]:
    """The oMLX key of each of the first ``count`` blocks of a store_cache call."""
    return [
        omlx_paged_cache.resolve_block_extra_keys(
            (index + 1) * block_size,
            extra_keys=arguments.get("extra_keys"),
            extra_key_token_start=arguments.get("extra_key_token_start"),
            extra_key_ranges=arguments.get("extra_key_ranges"),
        )
        for index in range(count)
    ]


def _announce_store(
    tracker: KvTracker,
    images: ImageRequests,
    paged_cache: Any,
    table: Any,
    arguments: dict[str, Any],
) -> None:
    tokens = arguments["tokens"]
    exclude_last = bool(arguments.get("_store_exact_terminal"))
    if not arguments.get("extra_keys") and not arguments.get("extra_key_ranges"):
        tracker.on_store(paged_cache, table, tokens, exclude_last=exclude_last)
        return
    if not hasattr(omlx_paged_cache, "resolve_block_extra_keys"):
        return  # which blocks carry an image key is unknown
    size = paged_cache.block_size
    keys = _block_extra_keys(arguments, size, len(table.block_ids))
    first_keyed = next(
        (index for index, key in enumerate(keys) if key is not None), len(keys)
    )
    spans = images.get(arguments.get("request_id"))
    routing, known = tokens, first_keyed * size
    if spans:
        # The bridge keys blocks from each image's first token on, so the first
        # keyed block is the one holding the first image token.
        first_image = min(span.start for span in spans)
        if first_keyed == min(first_image // size, len(keys)):
            routing, known = routing_tokens(tokens, spans)
        else:
            logger.warning(
                "oMLX keyed request %s from token %d, its first image starts at %d; "
                "announcing only the blocks before both",
                arguments.get("request_id"),
                first_keyed * size,
                first_image,
            )
            known = min(first_keyed * size, first_image)
    # Without spans, e.g. an image request to oMLX's own API, the router cannot
    # know the images; only the blocks before the first one are announced.
    tracker.on_store(
        paged_cache,
        table,
        tokens,
        exclude_last=exclude_last,
        routing_tokens=routing,
        extra_keys=keys,
        max_blocks=known // size,
    )


def _wrap_store_cache(tracker: KvTracker, model_id: str, images: ImageRequests) -> None:
    original = BlockAwarePrefixCache.store_cache
    signature = inspect.signature(original)

    @functools.wraps(original)
    def store_cache(self: BlockAwarePrefixCache, *args: Any, **kwargs: Any) -> Any:
        table = original(self, *args, **kwargs)
        try:
            if table is not None and _served(self.paged_cache.model_name, model_id):
                arguments = signature.bind(self, *args, **kwargs).arguments
                _announce_store(tracker, images, self.paged_cache, table, arguments)
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
