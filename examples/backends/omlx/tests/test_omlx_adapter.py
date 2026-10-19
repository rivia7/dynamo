# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""omlx_adapter against fake oMLX modules shaped like the real ones.

The fakes mirror the oMLX signatures the adapter probes, so these tests
exercise the patch logic on Linux; `python -m dynamo_omlx.bridge --check` is
the equivalent check against a real oMLX install.
"""

import asyncio
import dataclasses
import importlib
import signal
import sys
import types
from collections import deque
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

from dynamo_omlx.bridge.config import BridgeConfig
from dynamo_omlx.bridge.service import BridgeError, GenerateSpec, ModelState
from dynamo_omlx.protocol import API_PREFIX, ErrorCode
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
                extra_keys,
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

    async def add_request(
        self, prompt, sampling_params=None, request_id=None, skip_cache_store=False
    ):
        if self.add_error is not None:
            raise self.add_error
        self.added.append(
            SimpleNamespace(
                prompt=prompt,
                params=sampling_params,
                rid=request_id,
                skip=skip_cache_store,
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
        self, prompt, sampling_params=None, request_id=None, skip_cache_store=False
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
            "omlx.cache.paged_cache", compute_block_hash=compute_block_hash
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
    )
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
    cache.store_cache(
        "r2", [7, 8], None, extra_keys=("image",)
    )  # image chains are skipped
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
