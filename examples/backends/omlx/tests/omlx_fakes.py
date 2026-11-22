# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fakes and helpers shared by the oMLX example tests (stdlib only)."""

import asyncio
import hashlib
from array import array
from contextlib import asynccontextmanager

from dynamo_omlx.bridge.config import BridgeConfig
from dynamo_omlx.bridge.event_log import KvEventLog, NewBlock
from dynamo_omlx.bridge.kv_tracker import KvTracker
from dynamo_omlx.bridge.service import (
    BridgeRuntime,
    EngineOutput,
    GenerateSpec,
    ModelInfo,
    ModelState,
)
from dynamo_omlx.proxy.kv_reconciler import StoreOp


def omlx_hash(parent, tokens, model_name, extra_keys=None) -> bytes:
    """oMLX's block hash (omlx.cache.paged_cache.compute_block_hash)."""
    hasher = hashlib.sha256()
    if model_name:
        hasher.update(model_name.encode("utf-8"))
    hasher.update(parent if parent else b"omlx-root")
    hasher.update(bytes(str(tuple(tokens)), "utf-8"))
    if extra_keys:
        hasher.update(bytes(str(extra_keys), "utf-8"))
    return hasher.digest()


def chain(prefix: int, count: int, block_size: int) -> list[NewBlock]:
    """Blocks of chain ``prefix``: block i has id (prefix, i) and tokens [prefix, i, 0...]."""
    return [
        NewBlock(
            bid=bytes([prefix, i]),
            tokens=array("I", [prefix, i] + [0] * (block_size - 2)),
        )
        for i in range(count)
    ]


def output(ids, finished=False, reason=None, completion=0):
    return EngineOutput(
        new_token_ids=ids,
        finished=finished,
        finish_reason=reason,
        prompt_tokens=3,
        completion_tokens=completion,
        cached_tokens=2,
    )


class FakeRequest:
    """Plays a scripted list of outputs (or raises scripted exceptions).

    With ``gate``, nothing is produced until the gate opens or the request is
    aborted; an abort then finishes it with an ``abort`` output, which is what
    the oMLX adapter reports for an aborted request.
    """

    def __init__(self, script, gate: asyncio.Event | None = None):
        self._script = list(script)
        self._gate = gate
        self._aborted = asyncio.Event()
        self.abort_calls = 0

    async def outputs(self):
        if self._gate is not None:
            gate = asyncio.ensure_future(self._gate.wait())
            aborted = asyncio.ensure_future(self._aborted.wait())
            await asyncio.wait({gate, aborted}, return_when=asyncio.FIRST_COMPLETED)
            gate.cancel()
            aborted.cancel()
            if self._aborted.is_set():
                yield output([], finished=True, reason="abort")
                return
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            yield item

    async def abort(self):
        self.abort_calls += 1
        self._aborted.set()


class FakeFacade:
    def __init__(
        self,
        request: FakeRequest | None = None,
        state=ModelState.READY,
        vision: bool = False,
    ):
        self.request = request
        self.state = state
        self.vision = vision
        self.opened: list[GenerateSpec] = []
        self.load_calls = 0

    def model_info(self):
        return ModelInfo(
            model_id="m",
            state=self.state,
            block_size=4,
            kv_available=True,
            max_num_seqs=8,
            vision=self.vision,
        )

    def engine_stats(self):
        return {"loaded": self.state == ModelState.READY}

    def image_stats(self):
        return {"requests": len([s for s in self.opened if s.images])}

    @asynccontextmanager
    async def open(self, spec):
        self.opened.append(spec)
        yield self.request

    async def ensure_loaded(self):
        self.load_calls += 1


def make_runtime(facade, **cfg) -> BridgeRuntime:
    """A BridgeRuntime over ``facade`` with a tracker that never sees an SSD cache."""
    log = KvEventLog()
    tracker = KvTracker(log, ssd_provider=lambda: None, verify_block=None)
    return BridgeRuntime(BridgeConfig(model_id="m", **cfg), facade, log, tracker)


class Router:
    """Applies operations as the Dynamo router sees them, asserting its rules:
    parents are stored before children, a hash is stored once until it is
    removed, and a removal never orphans an announced child."""

    def __init__(self) -> None:
        self.blocks: set[int] = set()
        self.parents: dict[int, int | None] = {}

    def submit(self, ops) -> None:
        for op in ops:
            if isinstance(op, StoreOp):
                parent = op.parent
                for ext in op.hashes:
                    assert (
                        parent is None or parent in self.blocks
                    ), "child before parent"
                    assert ext not in self.blocks, "stored twice without a removal"
                    self.blocks.add(ext)
                    self.parents[ext] = parent
                    parent = ext
            else:
                removed = set(op.hashes)
                for ext in removed:
                    orphans = [
                        child
                        for child in self.blocks - removed
                        if self.parents.get(child) == ext
                    ]
                    assert not orphans, "removal orphaned a child"
                self.blocks -= removed


async def eventually(predicate, timeout: float = 5.0) -> None:
    """Wait on the running loop until ``predicate()`` holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)
