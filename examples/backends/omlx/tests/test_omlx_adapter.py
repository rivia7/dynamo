# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""omlx_adapter against fake oMLX modules shaped like the real ones.

The fakes mirror the oMLX signatures the adapter probes, so these tests
exercise the patch logic on Linux; `python -m dynamo_omlx.bridge --check` is
the equivalent check against a real oMLX install.
"""

import asyncio
import base64
import dataclasses
import importlib
import io
import signal
import sys
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

try:
    import httpx
    import uvicorn
    from fastapi import FastAPI
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

try:  # the image tests skip without them
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None

from dynamo_omlx.bridge.config import BridgeConfig
from dynamo_omlx.bridge.service import BridgeError, GenerateSpec, ImageInput, ModelState
from dynamo_omlx.protocol import API_PREFIX, ErrorCode, pad_value_for_mm_hash
from omlx_fakes import omlx_hash

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"
B = 2


def compute_block_hash(parent_hash, token_ids, extra_keys=None, model_name=None):
    return omlx_hash(parent_hash, token_ids, model_name, extra_keys)


def resolve_block_extra_keys(
    block_end, extra_keys=None, extra_key_token_start=None, extra_key_ranges=None
):
    """omlx.cache.paged_cache.resolve_block_extra_keys at oMLX 31b6b6a0."""
    if extra_key_ranges:
        selected = None
        for start, keys in extra_key_ranges:
            if block_end > start:
                selected = keys
            else:
                break
        return selected
    if extra_keys is not None and (
        extra_key_token_start is None or block_end > extra_key_token_start
    ):
        return extra_keys
    return None


def _capture_vlm_position_state(lm, extra_kwargs):
    """omlx.engine.vlm._capture_vlm_position_state, without the language model."""
    if extra_kwargs.get("rope_deltas") is not None:
        extra_kwargs["_captured_rope_deltas"] = extra_kwargs["rope_deltas"]


@dataclass
class ModelSettings:
    mtp_enabled: bool = False
    mtp_adaptive_max_depth: int | None = None
    mtp_fixed_depth: int | None = None
    dflash_enabled: bool = False


class ModelSettingsManager:
    def __init__(self):
        self.saved: dict[str, ModelSettings] = {}

    def get_settings(self, model_id):
        return dataclasses.replace(self.saved.get(model_id, ModelSettings()))

    def set_settings(self, model_id, settings):
        self.saved[model_id] = dataclasses.replace(settings)


class OMLXError(Exception):
    pass


class EnginePoolError(OMLXError):
    pass


class ModelNotFoundError(EnginePoolError):
    pass


class ModelTooLargeError(EnginePoolError):
    pass


class InsufficientMemoryError(EnginePoolError):
    pass


class ModelLoadingError(EnginePoolError):
    pass


class ModelBusyError(EnginePoolError):
    pass


class ModelUnavailableError(EnginePoolError):
    pass


class SchedulerQueueFullError(OMLXError):
    pass


class PrefillMemoryExceededError(OMLXError):
    pass


class PrefillMemoryAbortedError(PrefillMemoryExceededError):
    pass


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list | None = None
    stop_token_ids: list | None = None
    seed: int | None = None


class PagedSSDCacheManager:
    def __init__(self):
        self.blocks: set[bytes] = set()

    def has_block(self, block_hash):
        return block_hash in self.blocks

    def _evict_tracked_until_size(self, target_size, max_count=None):
        evicted = [
            (None, SimpleNamespace(block_hash=h))
            for h in sorted(self.blocks)[:max_count]
        ]
        self.blocks.difference_update(meta.block_hash for _, meta in evicted)
        return evicted

    def delete_block(self, block_hash):
        present = block_hash in self.blocks
        self.blocks.discard(block_hash)
        return present

    def forget_block(self, block_hash):
        return self.delete_block(block_hash)

    def clear(self):
        count = len(self.blocks)
        self.blocks.clear()
        return count

    def get_stats_dict(self):
        return {"total_size": 1, "max_size": 2, "num_files": len(self.blocks)}


class BlockAwarePrefixCache:
    def __init__(self, paged_cache, ssd):
        self.paged_cache = paged_cache
        self.paged_ssd_cache = ssd

    def store_cache(
        self,
        request_id,
        tokens,
        cache_data,
        model_cache_config=None,
        boundary_snapshots=None,
        extra_keys=None,
        extra_key_token_start=None,
        extra_key_ranges=None,
        hot_cache_write_back=True,
        _store_exact_terminal=False,
        _store_tail_terminal=False,
    ):
        table, parent = SimpleNamespace(block_ids=[]), None
        for i in range(len(tokens) // B):
            bid = compute_block_hash(
                parent,
                tokens[i * B : (i + 1) * B],
                resolve_block_extra_keys(
                    (i + 1) * B, extra_keys, extra_key_token_start, extra_key_ranges
                ),
                self.paged_cache.model_name,
            )
            block_id = len(self.paged_cache.allocated_blocks) + 1
            self.paged_cache.allocated_blocks[block_id] = SimpleNamespace(
                block_hash=bid, token_count=B
            )
            self.paged_ssd_cache.blocks.add(bid)
            table.block_ids.append(block_id)
            parent = bid
        return table


class FakeCore:
    def __init__(self):
        self.outputs = []
        self.added = []
        self.aborted = []
        self.add_error = None
        self._mlx_executor = ThreadPoolExecutor(max_workers=1)

    async def add_request(
        self,
        prompt,
        sampling_params=None,
        request_id=None,
        skip_cache_store=False,
        **vlm,
    ):
        if self.add_error is not None:
            raise self.add_error
        self.added.append(
            SimpleNamespace(
                prompt=prompt,
                params=sampling_params,
                rid=request_id,
                skip=skip_cache_store,
                vlm=vlm,
            )
        )
        return request_id

    async def stream_outputs(self, request_id):
        for out in self.outputs:
            yield out

    async def abort_request(self, request_id):
        self.aborted.append(request_id)
        return True


class EngineCore:
    async def add_request(
        self,
        prompt,
        sampling_params=None,
        request_id=None,
        vlm_inputs_embeds=None,
        vlm_extra_kwargs=None,
        vlm_image_hash=None,
        vlm_cache_key_start=0,
        vlm_cache_key_ranges=None,
        skip_cache_store=False,
    ):
        raise NotImplementedError


class EnginePool:
    def __init__(self, entries):
        self._entries = entries
        self.pinned: set[str] = set()

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def get_model_ids(self):
        return list(self._entries)

    def set_pinned(self, model_id, pinned):
        self.pinned.add(model_id)
        return True

    @asynccontextmanager
    async def acquire(self, model_id, force_lm=False):
        yield self._entries[model_id].engine

    @staticmethod
    def _resolve_scheduler_from_engine(engine):
        return getattr(engine, "scheduler", None)


def omlx_output(ids, finished=False, reason=None, error=None, error_code=None):
    return SimpleNamespace(
        new_token_ids=ids,
        finished=finished,
        finish_reason=reason,
        prompt_tokens=2,
        completion_tokens=len(ids),
        cached_tokens=0,
        error=error,
        error_code=error_code,
    )


def module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def fake(monkeypatch):
    ssd = PagedSSDCacheManager()
    paged = SimpleNamespace(block_size=B, model_name=MODEL, allocated_blocks={})
    scheduler = SimpleNamespace(
        config=SimpleNamespace(
            paged_cache_block_size=B, max_num_seqs=8, prefill_step_size=2048
        ),
        block_aware_cache=BlockAwarePrefixCache(paged, ssd),
        paged_ssd_cache_manager=ssd,
        running={},
        waiting=deque(),
    )
    core = FakeCore()
    engine = SimpleNamespace(scheduler=scheduler, _engine=core)
    entry = SimpleNamespace(
        engine=engine,
        is_loading=False,
        load_failed=False,
        is_pinned=False,
        in_use=0,
        model_context_length=4096,
    )
    pool = EnginePool({MODEL: entry})
    state = SimpleNamespace(engine_pool=None, settings_manager=None)
    settings = ModelSettingsManager()
    # Process-wide state of oMLX's MTP patch after the last model load.
    mtp = module("omlx.patches.mlx_lm_mtp", active=False, depth=3)
    mtp.is_mtp_active = lambda: mtp.active
    mtp.get_mtp_depth = lambda: mtp.depth

    def init_server(
        model_dirs, scheduler_config=None, api_key=None, global_settings=None
    ):
        state.settings_manager = settings
        state.engine_pool = pool

    def get_sampling_params(
        req_temperature,
        req_top_p,
        model_id=None,
        req_top_k=None,
        req_repetition_penalty=None,
        req_min_p=None,
        req_presence_penalty=None,
        req_frequency_penalty=None,
        req_max_tokens=None,
        ocr_defaults=None,
        req_xtc_probability=None,
        req_xtc_threshold=None,
    ):
        def pick(value, default):
            return default if value is None else value

        return (
            pick(req_temperature, 1.0),
            pick(req_top_p, 0.95),
            pick(req_top_k, 0),
            pick(req_repetition_penalty, 1.0),
            pick(req_min_p, 0.0),
            pick(req_presence_penalty, 0.0),
            pick(req_frequency_penalty, 0.0),
            pick(req_max_tokens, 32768),
            0.0,
            0.1,
        )

    async def verify_api_key():
        return True

    app = FastAPI()
    modules = {
        "omlx": module("omlx", __path__=[]),
        "omlx.cache": module("omlx.cache", __path__=[]),
        "omlx.server": module(
            "omlx.server",
            init_server=init_server,
            app=app,
            get_server_state=lambda: state,
            get_sampling_params=get_sampling_params,
            verify_api_key=verify_api_key,
        ),
        "omlx.cli": module("omlx.cli", main=lambda: None),
        "omlx.cache.paged_cache": module(
            "omlx.cache.paged_cache",
            compute_block_hash=compute_block_hash,
            resolve_block_extra_keys=resolve_block_extra_keys,
        ),
        "omlx.engine": module("omlx.engine", __path__=[]),
        "omlx.engine.vlm": module(
            "omlx.engine.vlm", _capture_vlm_position_state=_capture_vlm_position_state
        ),
        "omlx.utils": module("omlx.utils", __path__=[]),
        "omlx.utils.image": module(
            "omlx.utils.image", get_max_image_bytes=lambda: 50 * 1024 * 1024
        ),
        "omlx.cache.paged_ssd_cache": module(
            "omlx.cache.paged_ssd_cache", PagedSSDCacheManager=PagedSSDCacheManager
        ),
        "omlx.cache.prefix_cache": module(
            "omlx.cache.prefix_cache", BlockAwarePrefixCache=BlockAwarePrefixCache
        ),
        "omlx.engine_core": module("omlx.engine_core", EngineCore=EngineCore),
        "omlx.model_settings": module(
            "omlx.model_settings",
            ModelSettings=ModelSettings,
            ModelSettingsManager=ModelSettingsManager,
        ),
        "omlx.patches": module("omlx.patches", __path__=[], mlx_lm_mtp=mtp),
        "omlx.patches.mlx_lm_mtp": mtp,
        "omlx.engine_pool": module("omlx.engine_pool", EnginePool=EnginePool),
        "omlx.request": module("omlx.request", SamplingParams=SamplingParams),
        "omlx.exceptions": module(
            "omlx.exceptions",
            InsufficientMemoryError=InsufficientMemoryError,
            ModelBusyError=ModelBusyError,
            ModelLoadingError=ModelLoadingError,
            ModelNotFoundError=ModelNotFoundError,
            ModelTooLargeError=ModelTooLargeError,
            PrefillMemoryAbortedError=PrefillMemoryAbortedError,
            PrefillMemoryExceededError=PrefillMemoryExceededError,
            ModelUnavailableError=ModelUnavailableError,
            SchedulerQueueFullError=SchedulerQueueFullError,
        ),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # install() patches these classes; restore them after each test.
    for cls, attr in (
        (uvicorn.Server, "handle_exit"),
        (BlockAwarePrefixCache, "store_cache"),
        (PagedSSDCacheManager, "_evict_tracked_until_size"),
        (PagedSSDCacheManager, "delete_block"),
        (PagedSSDCacheManager, "forget_block"),
        (PagedSSDCacheManager, "clear"),
    ):
        monkeypatch.setattr(cls, attr, getattr(cls, attr))
    sys.modules.pop("dynamo_omlx.bridge.omlx_adapter", None)
    adapter = importlib.import_module("dynamo_omlx.bridge.omlx_adapter")
    yield SimpleNamespace(
        adapter=adapter,
        server=modules["omlx.server"],
        app=app,
        pool=pool,
        ssd=ssd,
        scheduler=scheduler,
        core=core,
        entry=entry,
        settings=settings,
        mtp=mtp,
        modules=modules,
    )
    core._mlx_executor.shutdown(wait=False)
    sys.modules.pop("dynamo_omlx.bridge.omlx_adapter", None)


def install(fake, **cfg):
    runtime = fake.adapter.install(BridgeConfig(model_id=MODEL, **cfg))
    scheduler_config = SimpleNamespace(paged_cache_block_size=256)
    fake.server.init_server(["/models"], scheduler_config=scheduler_config)
    return runtime, scheduler_config


def test_probe_accepts_the_expected_surface(fake):
    assert fake.adapter.probe() == []


def test_probe_reports_missing_capabilities(fake, monkeypatch):
    monkeypatch.delattr(PagedSSDCacheManager, "forget_block")
    assert fake.adapter.probe() == ["PagedSSDCacheManager.forget_block is missing"]


async def test_install_mounts_routes_pins_and_presets_the_block_size(fake):
    runtime, scheduler_config = install(fake, kv_block_size=2048)
    assert scheduler_config.paged_cache_block_size == 2048
    assert MODEL in fake.pool.pinned
    # Recent FastAPI keeps included routers out of app.routes, so dispatch instead.
    transport = httpx.ASGITransport(app=fake.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://mac") as client:
        response = await client.post(API_PREFIX + "/generate", json={})
    assert response.status_code != 404
    with pytest.raises(RuntimeError, match="already installed"):
        fake.adapter.install(BridgeConfig(model_id=MODEL))


def test_mtp_is_written_to_omlx_model_settings_before_the_model_loads(fake):
    fake.settings.saved[MODEL] = ModelSettings(mtp_fixed_depth=2)
    install(fake, mtp=True, mtp_depth=4)
    saved = fake.settings.saved[MODEL]
    assert (saved.mtp_enabled, saved.mtp_adaptive_max_depth, saved.mtp_fixed_depth) == (
        True,
        4,
        None,
    )


def test_mtp_settings_are_left_alone_by_default(fake):
    install(fake)
    assert fake.settings.saved == {}


def test_mtp_is_refused_while_omlx_runs_dflash(fake):
    fake.settings.saved[MODEL] = ModelSettings(dflash_enabled=True)
    with pytest.raises(RuntimeError, match="DFlash"):
        install(fake, mtp=True)


@dataclass
class SettingsBeforeMtpDepth:
    mtp_enabled: bool = False
    dflash_enabled: bool = False


def test_mtp_works_on_an_omlx_without_depth_settings(fake):
    fake.settings.saved[MODEL] = SettingsBeforeMtpDepth()
    install(fake, mtp=True)
    assert fake.settings.saved[MODEL] == SettingsBeforeMtpDepth(mtp_enabled=True)


def test_mtp_depth_needs_omlx_depth_settings(fake):
    fake.settings.saved[MODEL] = SettingsBeforeMtpDepth()
    with pytest.raises(
        RuntimeError, match="no mtp_adaptive_max_depth, mtp_fixed_depth"
    ):
        install(fake, mtp=True, mtp_depth=4)
    assert fake.settings.saved[MODEL] == SettingsBeforeMtpDepth()


def test_info_reports_the_mtp_draft_depth(fake):
    runtime, _ = install(fake)
    assert runtime.info()["model"]["mtp_depth"] is None
    fake.mtp.active = True
    assert runtime.info()["model"]["mtp_depth"] == 3


def test_probe_reports_a_missing_mtp_api(fake, monkeypatch):
    monkeypatch.delattr(fake.mtp, "get_mtp_depth")
    assert fake.adapter.probe() == ["omlx.patches.mlx_lm_mtp.get_mtp_depth is missing"]


def test_unknown_model_fails_at_startup(fake):
    fake.adapter.install(BridgeConfig(model_id="other"))
    with pytest.raises(RuntimeError, match="no model 'other'"):
        fake.server.init_server(["/models"])


def test_store_and_removal_hooks_feed_the_tracker(fake):
    runtime, _ = install(fake)
    cache = fake.scheduler.block_aware_cache
    cache.store_cache("r1", [1, 2, 3, 4, 5], cache_data=None)
    # An image request to oMLX's own API: the router cannot know its image.
    cache.store_cache("r2", [7, 8], None, extra_keys=("image",))
    runtime.tracker.process_pending()
    assert len(runtime.log) == 2 and runtime.tracker.stats()["state"] == "live"
    first = compute_block_hash(None, [1, 2], model_name=MODEL)
    assert fake.ssd.delete_block(first) is True
    runtime.tracker.process_pending()
    assert len(runtime.log) == 0  # the chain went with its root


def test_verification_catches_misaligned_tokens(fake):
    runtime, _ = install(fake)
    paged = fake.scheduler.block_aware_cache.paged_cache
    bid = compute_block_hash(None, [1, 2], model_name=MODEL)
    paged.allocated_blocks[1] = SimpleNamespace(block_hash=bid, token_count=B)
    fake.ssd.blocks.add(bid)
    runtime.tracker.on_store(paged, SimpleNamespace(block_ids=[1]), [9, 9])
    runtime.tracker.process_pending()
    assert (
        runtime.tracker.stats()["faults_hash_mismatch"] == 1 and len(runtime.log) == 0
    )


def test_eviction_hook_withdraws_blocks(fake):
    runtime, _ = install(fake)
    fake.scheduler.block_aware_cache.store_cache("r1", [1, 2], None)
    runtime.tracker.process_pending()
    fake.ssd._evict_tracked_until_size(0, max_count=1)
    runtime.tracker.process_pending()
    assert len(runtime.log) == 0


# -- images ------------------------------------------------------------------------

PAD, START, END = 9, 7, 8  # image placeholder, vision start, vision end
HIDDEN = 4


class FakeImageProcessor:
    """Qwen-VL shaped: an image w pixels wide gets a (1, 2, 2w) grid, w tokens.

    ``max_pixels`` caps that width, so tests can see the pixel range applied.
    """

    merge_size = 2
    min_pixels, max_pixels = 1, 1000  # the model's own range

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, images, **limits):
        self.calls.append(limits)
        cap = limits.get("max_pixels", self.max_pixels)
        grid = [[1, 2, 2 * min(image.size[0], cap)] for image in images]
        return {
            "pixel_values": np.zeros((sum(t * h * w for t, h, w in grid), 3)),
            "image_grid_thw": np.array(grid),
        }


class FakeVisionTower:
    deepstack_visual_indexes: list = []
    patch_embed = SimpleNamespace(
        proj=SimpleNamespace(weight=SimpleNamespace(dtype="float32"))
    )

    def __init__(self):
        self.calls = 0

    def __call__(self, pixel_values, grid):
        self.calls += 1
        tokens = sum(t * h * w // 4 for t, h, w in grid.tolist())
        return np.full((tokens, HIDDEN), float(self.calls)), []


class FakeVlmModel:
    def __init__(self):
        self.config = SimpleNamespace(image_token_id=PAD, vision_start_token_id=START)
        self.vision_tower = FakeVisionTower()
        self.language_model = SimpleNamespace()
        self.embedded: list[dict] = []
        self.pixel_shapes: list[tuple] = []

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        self.embedded.append(kwargs)
        self.pixel_shapes.append(pixel_values.shape)
        length = input_ids.shape[1]
        features = {
            "inputs_embeds": np.zeros((1, length, HIDDEN)),
            "position_ids": np.zeros((3, 1, length)),
            "rope_deltas": np.array([[-1]]),
        }
        return SimpleNamespace(
            inputs_embeds=features["inputs_embeds"], to_dict=lambda: features
        )


class FakeVisionCache:
    def __init__(self):
        self.entries = {}

    def get(self, image_hash, model_name):
        return self.entries.get((model_name, image_hash))

    def put(self, image_hash, model_name, features):
        self.entries[(model_name, image_hash)] = features


def png(width: int) -> str:
    if Image is None:
        pytest.skip("needs PIL")
    buffer = io.BytesIO()
    Image.new("RGB", (width, 2), (width, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def vlm(fake, monkeypatch):
    """``fake`` with a Qwen-VL shaped model loaded in oMLX's VLM engine; mlx is numpy."""
    if np is None or Image is None:
        pytest.skip("needs numpy and PIL")
    mx = module(
        "mlx.core",
        array=np.asarray,
        zeros=np.zeros,
        ones_like=np.ones_like,
        int32=np.int32,
        concatenate=lambda arrays, axis=0: np.concatenate(arrays, axis=axis),
        eval=lambda *arrays: None,
        synchronize=lambda: None,
        clear_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "mlx", module("mlx", __path__=[], core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    engine = fake.entry.engine
    engine.model_type = "qwen4_exp"
    engine._vlm_model = FakeVlmModel()
    engine._processor = SimpleNamespace(image_processor=FakeImageProcessor())
    engine._vision_cache = FakeVisionCache()
    engine._vision_cache_enabled = True
    engine._model_name = "/models/" + MODEL
    fake.core.outputs = [omlx_output([5], finished=True, reason="stop")]
    return fake


def image_spec(images, token_ids=(1, START, PAD, END, 2), **overrides):
    return GenerateSpec(
        request_id="client-1",
        token_ids=list(token_ids),
        images=tuple(ImageInput(data, mm_hash) for data, mm_hash in images),
        **overrides,
    )


async def run(facade, spec):
    async with facade.open(spec) as request:
        return [output async for output in request.outputs()]


def image_keys(payloads, min_pixels=1, max_pixels=1000) -> list[str]:
    """The bridge's cumulative oMLX keys of these images."""
    from dynamo_omlx.bridge.images import cumulative_keys, image_key, read_images

    files = read_images(payloads, max_bytes=0)
    return cumulative_keys(
        [image_key(file.digest, min_pixels, max_pixels) for file in files]
    )


async def test_image_requests_reach_omlx_expanded_and_keyed(vlm):
    runtime, _ = install(vlm)
    facade = runtime.facade
    assert runtime.info()["model"]["vision"] is True
    spec = image_spec([(png(3), 0xA)], max_tokens=5000, expanded_prompt_len=7)
    await run(facade, spec)
    added = vlm.core.added[-1]
    assert added.prompt == [1, START, PAD, PAD, PAD, END, 2]
    assert added.params.max_tokens == 4096 - 7  # the context minus the expanded prompt
    [image_hash] = image_keys([png(3)])
    assert added.vlm["vlm_cache_key_ranges"] == [(2, image_hash)]
    assert (added.vlm["vlm_image_hash"], added.vlm["vlm_cache_key_start"]) == (
        image_hash,
        2,
    )
    assert added.vlm["vlm_inputs_embeds"].shape == (1, 7, HIDDEN)
    extra = added.vlm["vlm_extra_kwargs"]
    assert "inputs_embeds" not in extra and "_captured_rope_deltas" in extra
    assert facade.image_stats() == {
        "requests": 1,
        "images": 1,
        "images_decoded": 1,
        "vision_cache_hits": 0,
        "vision_cache_misses": 1,
        "token_count_mismatches": 0,
    }

    # oMLX stores the finished request with the keys add_request derived.
    vlm.scheduler.block_aware_cache.store_cache(
        added.rid,
        added.prompt + [5],
        None,
        extra_keys=(image_hash,),
        extra_key_token_start=2,
        extra_key_ranges=[(2, (image_hash,))],
    )
    runtime.tracker.process_pending()
    pad = pad_value_for_mm_hash(0xA)
    assert [list(block.tokens) for block in runtime.log.snapshot().blocks] == [
        [1, START],
        [pad, pad],
        [pad, END],
        [2, 5],
    ]
    assert runtime.tracker.stats()["state"] == "live"


async def test_cached_images_need_no_pixels(vlm):
    runtime, _ = install(vlm)
    engine = vlm.entry.engine
    model, processor = engine._vlm_model, engine._processor.image_processor
    for _ in range(2):
        await run(runtime.facade, image_spec([(png(3), 0xA)]))
    # The second request found every image's features and grid in the cache.
    assert (model.vision_tower.calls, len(processor.calls)) == (1, 1)
    assert model.pixel_shapes[1] == (1, 1)
    first, second = model.embedded
    assert np.array_equal(
        first["cached_image_features"], second["cached_image_features"]
    )
    assert second["image_grid_thw"].tolist() == [[1, 2, 6]]
    one, two = vlm.core.added
    assert (one.prompt, one.vlm["vlm_cache_key_ranges"]) == (
        two.prompt,
        two.vlm["vlm_cache_key_ranges"],
    )
    stats = runtime.facade.image_stats()
    assert (
        stats["images_decoded"],
        stats["vision_cache_misses"],
        stats["vision_cache_hits"],
    ) == (1, 1, 1)


async def test_models_that_might_ignore_cached_features_still_get_pixels(vlm):
    runtime, _ = install(vlm)
    engine = vlm.entry.engine
    engine.model_type = "qwen2_vl"
    for _ in range(2):
        await run(runtime.facade, image_spec([(png(3), 0xA)]))
    model = engine._vlm_model
    assert (
        model.vision_tower.calls == 1
        and len(engine._processor.image_processor.calls) == 2
    )
    assert model.pixel_shapes[1] != (1, 1)
    assert "cached_image_features" in model.embedded[1]


async def test_the_pixel_range_reaches_the_processor_and_the_keys(vlm):
    runtime, _ = install(vlm)
    engine = vlm.entry.engine
    processor = engine._processor.image_processor
    await run(
        runtime.facade,
        image_spec([(png(3), 0xA)], min_pixels=1, max_pixels=2, expanded_prompt_len=6),
    )
    assert processor.calls == [{"min_pixels": 1, "max_pixels": 2}]
    added = vlm.core.added[-1]
    assert added.prompt == [1, START, PAD, PAD, END, 2]  # 2 tokens within the range
    assert added.vlm["vlm_image_hash"] == image_keys([png(3)], 1, 2)[0]
    assert runtime.facade.image_stats()["token_count_mismatches"] == 0
    # Another range is another image to the caches.
    await run(runtime.facade, image_spec([(png(3), 0xA)], max_pixels=3))
    assert engine._vlm_model.vision_tower.calls == 2


async def test_a_damaged_cache_entry_is_recomputed(vlm):
    runtime, _ = install(vlm)
    engine = vlm.entry.engine
    await run(runtime.facade, image_spec([(png(3), 0xA)]))
    for key in engine._vision_cache.entries:
        features, grid = engine._vision_cache.entries[key]
        engine._vision_cache.entries[key] = [features, np.array([1, 2, 8])]
    await run(runtime.facade, image_spec([(png(3), 0xA)]))
    assert engine._vlm_model.vision_tower.calls == 2
    assert vlm.core.added[-1].prompt == [1, START, PAD, PAD, PAD, END, 2]


async def test_images_without_mm_hash_are_announced_up_to_the_image(vlm):
    runtime, _ = install(vlm)
    await run(
        runtime.facade,
        image_spec([(png(1), None)], token_ids=(1, 2, 3, START, PAD, END)),
    )
    added = vlm.core.added[-1]
    vlm.scheduler.block_aware_cache.store_cache(
        added.rid,
        added.prompt,
        None,
        extra_key_ranges=[(4, (added.vlm["vlm_image_hash"],))],
    )
    runtime.tracker.process_pending()
    assert [list(b.tokens) for b in runtime.log.snapshot().blocks] == [
        [1, 2],
        [3, START],
    ]


def test_images_omlx_served_itself_announce_only_the_text_before_them(fake):
    runtime, _ = install(fake)
    fake.scheduler.block_aware_cache.store_cache(
        "native", [1, 2, 3, PAD, PAD, 4], None, extra_key_ranges=[(3, ("h",))]
    )
    runtime.tracker.process_pending()
    assert [list(b.tokens) for b in runtime.log.snapshot().blocks] == [[1, 2]]


async def test_a_token_count_the_router_disagrees_with_is_counted(vlm, caplog):
    runtime, _ = install(vlm)
    await run(runtime.facade, image_spec([(png(3), 0xA)], expanded_prompt_len=6))
    assert runtime.facade.image_stats()["token_count_mismatches"] == 1
    assert "cannot match its blocks" in caplog.text


@pytest.mark.parametrize(
    "token_ids, context, message",
    [
        ((1, PAD, 2), 4096, "vision start token"),
        ((1, START, PAD, END, START, PAD, END), 4096, "2 image placeholders for 1"),
        ((1, START, PAD, END, 2), 7, "model context is 7"),
    ],
)
async def test_image_requests_need_placeholders_and_room(
    vlm, token_ids, context, message
):
    runtime, _ = install(vlm)
    vlm.entry.model_context_length = context
    with pytest.raises(BridgeError, match=message) as info:
        await run(runtime.facade, image_spec([(png(3), 0xA)], token_ids=token_ids))
    assert info.value.code == ErrorCode.INVALID_REQUEST
    assert vlm.core.added == []


async def test_text_models_refuse_images(fake):
    runtime, _ = install(fake)
    assert runtime.info()["model"]["vision"] is False
    with pytest.raises(BridgeError, match="not a vision-language model"):
        await run(runtime.facade, image_spec([(png(3), 0xA)]))


def test_images_can_be_turned_off(vlm):
    runtime, _ = install(vlm, images=False)
    assert runtime.info()["model"]["vision"] is False


def test_an_omlx_without_the_vlm_surface_serves_text_only(vlm, monkeypatch):
    assert vlm.adapter.probe_vision() == []
    monkeypatch.delattr(vlm.modules["omlx.engine.vlm"], "_capture_vlm_position_state")
    assert vlm.adapter.probe_vision() == [
        "omlx.engine.vlm._capture_vlm_position_state is missing"
    ]
    runtime, _ = install(vlm)
    assert runtime.info()["model"]["vision"] is False


async def test_facade_streams_outputs_with_omlx_sampling_defaults(fake):
    fake.core.outputs = [
        omlx_output([5]),
        omlx_output([6], finished=True, reason="stop"),
    ]
    facade = fake.adapter.OmlxEngineFacade(MODEL)
    install(fake)
    info = facade.model_info()
    assert (info.state, info.block_size, info.max_num_seqs, info.context_length) == (
        ModelState.READY,
        B,
        8,
        4096,
    )
    spec = GenerateSpec(
        request_id="r",
        token_ids=[1, 2],
        temperature=0.5,
        stop_token_ids=(9,),
        skip_cache_store=True,
    )
    async with facade.open(spec) as request:
        outputs = [out async for out in request.outputs()]
    assert [o.new_token_ids for o in outputs] == [[5], [6]] and outputs[
        -1
    ].finish_reason == "stop"
    [added] = fake.core.added
    assert (
        added.prompt == [1, 2] and added.skip is True and added.rid.startswith("dyn-")
    )
    assert (
        added.params.temperature,
        added.params.top_p,
        added.params.stop,
        added.params.stop_token_ids,
    ) == (
        0.5,
        0.95,
        [],
        [9],
    )
    assert fake.core.aborted == []


async def test_facade_aborts_unfinished_requests_and_maps_errors(fake):
    install(fake)
    facade = fake.adapter.OmlxEngineFacade(MODEL)
    spec = GenerateSpec(request_id="r", token_ids=[1])
    fake.core.outputs = [omlx_output([5])]
    async with facade.open(spec) as request:
        pass
    assert len(fake.core.aborted) == 1

    fake.core.outputs = [
        omlx_output([], error="guard", error_code="prefill_memory_aborted")
    ]
    with pytest.raises(BridgeError) as info:
        async with facade.open(spec) as request:
            [out async for out in request.outputs()]
    assert info.value.code == ErrorCode.OVERLOADED

    for error, code in (
        (SchedulerQueueFullError("queue full"), ErrorCode.OVERLOADED),
        (ModelBusyError("busy"), ErrorCode.OVERLOADED),
        (ModelUnavailableError("unloading"), ErrorCode.MODEL_NOT_READY),
    ):
        fake.core.add_error = error
        with pytest.raises(BridgeError) as info:
            async with facade.open(spec):
                pass
        assert info.value.code == code

    fake.entry.engine = None
    fake.entry.is_loading = True
    assert facade.model_info().state == ModelState.LOADING


async def test_engine_abort_output_is_a_cancellation_not_an_error(fake):
    install(fake)
    facade = fake.adapter.OmlxEngineFacade(MODEL)
    # What EngineCore.abort_request() puts in the collector.
    fake.core.outputs = [
        omlx_output([], finished=True, reason="abort", error="Request aborted")
    ]
    async with facade.open(GenerateSpec(request_id="r", token_ids=[1])) as request:
        outputs = [out async for out in request.outputs()]
    assert [(o.finished, o.finish_reason) for o in outputs] == [(True, "abort")]
    assert fake.core.aborted == []  # the engine already dropped it


async def test_lifespan_starts_the_runtime_and_uvicorn_exit_ends_streams(fake):
    runtime, _ = install(fake, preload=False)
    async with fake.app.router.lifespan_context(fake.app):
        assert runtime.loop is asyncio.get_running_loop()
        server = uvicorn.Server(uvicorn.Config(fake.app, log_config=None))
        server.handle_exit(signal.SIGINT, None)
        await asyncio.sleep(0)
        assert server.should_exit and runtime.generate._closing
    assert runtime.tracker._thread is None
