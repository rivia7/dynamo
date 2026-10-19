# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keeps the reconciler in step with the bridge's KV event stream.

The loop resumes the stream from the last applied sequence number after a
dropped connection. Any gap, reset, incarnation change or orphan block
instead triggers a snapshot, which the reconciler diffs against what is
already announced, so recovery never withdraws blocks that are still valid.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from dynamo_omlx.protocol import (
    Frame,
    KvState,
    ProtocolError,
    ResetReason,
    parse_blocks,
)
from dynamo_omlx.proxy.kv_reconciler import KvOperation, KvReconciler, ResyncRequired

logger = logging.getLogger(__name__)


class KvClient(Protocol):
    def kv_snapshot(self) -> AbstractAsyncContextManager[AsyncIterator[dict[str, Any]]]:
        ...

    def kv_stream(
        self, incarnation: str | None, after: int
    ) -> AbstractAsyncContextManager[AsyncIterator[dict[str, Any]]]:
        ...


class KvSyncFatal(Exception):
    """A configuration change the proxy cannot absorb without re-registering."""


class SyncState:
    CONNECTING = "connecting"
    SYNCING = "syncing"
    LIVE = "live"
    DISABLED = "disabled"
    STOPPED = "stopped"


class _Disabled(Exception):
    pass


# Consecutive heartbeats announcing events the stream has not delivered.
_MAX_LAGGING_HEARTBEATS = 2


class KvSync:
    def __init__(
        self,
        client: KvClient,
        reconciler: KvReconciler,
        submit: Callable[[Sequence[KvOperation]], None],
        *,
        stall_heartbeats: float = 4.0,
        min_stall_timeout_s: float = 5.0,
        backoff_initial_s: float = 0.5,
        backoff_max_s: float = 10.0,
        disabled_poll_s: float = 30.0,
    ) -> None:
        self._client = client
        self._reconciler = reconciler
        self._submit = submit
        self._stall_heartbeats = stall_heartbeats
        self._min_stall_timeout_s = min_stall_timeout_s
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._disabled_poll_s = disabled_poll_s
        self._stopped = asyncio.Event()
        self._live_since: float | None = None
        self.state = SyncState.CONNECTING
        # Present from the start, so that rate() sees the first increment.
        self.counters: Counter[str] = Counter(
            dict.fromkeys(
                ("events", "snapshots", "resyncs", "reconnects", "protocol_errors"), 0
            )
        )

    def stop(self) -> None:
        self._stopped.set()
        self.state = SyncState.STOPPED

    async def run(self) -> None:
        """Run until :meth:`stop`; raises :class:`KvSyncFatal` on config drift.

        :meth:`_follow` only returns by raising, so every pass ends in a backoff
        that resets once a stream stayed live for ``backoff_max_s``.
        """
        loop = asyncio.get_running_loop()
        need_snapshot = True
        backoff = self._backoff_initial_s
        while not self._stopped.is_set():
            self._live_since = None
            try:
                if need_snapshot:
                    self.state = SyncState.SYNCING
                    await self._sync_snapshot()
                    need_snapshot = False
                await self._follow()
            except ResyncRequired as e:
                logger.info("KV resync required: %s", e.reason)
                self.counters["resyncs"] += 1
                need_snapshot = True
            except _Disabled:
                # The snapshot already withdrew every block; poll until re-enabled.
                self.state = SyncState.DISABLED
                need_snapshot = True
                await self._sleep(self._disabled_poll_s)
                continue
            except ProtocolError as e:
                logger.warning("KV stream protocol error, resyncing: %s", e)
                self.counters["protocol_errors"] += 1
                need_snapshot = True
            except ConnectionError as e:
                logger.warning("KV stream disconnected: %s", e)
                self.counters["reconnects"] += 1
                self.state = SyncState.CONNECTING
            if (
                self._live_since is not None
                and loop.time() - self._live_since >= self._backoff_max_s
            ):
                backoff = self._backoff_initial_s
            await self._sleep(backoff * random.uniform(0.8, 1.2))
            backoff = min(backoff * 2, self._backoff_max_s)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopped.wait(), seconds)
        except asyncio.TimeoutError:
            pass

    def _check_block_size(self, block_size: Any) -> None:
        if block_size is not None and block_size != self._reconciler.block_size:
            raise KvSyncFatal(
                f"bridge block size changed to {block_size}; this worker registered "
                f"{self._reconciler.block_size} and must restart"
            )

    async def _sync_snapshot(self) -> None:
        async with self._client.kv_snapshot() as frames:
            begin = await anext(frames, None)
            if begin is None or begin.get("t") != Frame.SNAPSHOT_BEGIN:
                raise ProtocolError("snapshot does not start with snap_begin")
            state = begin.get("state")
            if state == KvState.LIVE:
                self._check_block_size(begin.get("block_size"))
            blocks = []
            end = None
            async for frame in frames:
                kind = frame.get("t")
                if kind == Frame.SNAPSHOT_BLOCKS:
                    blocks.extend(
                        parse_blocks(frame.get("blocks"), self._reconciler.block_size)
                    )
                elif kind == Frame.SNAPSHOT_END:
                    end = frame
                    break
            if end is None:
                raise ConnectionError("snapshot ended before snap_end")
            if end.get("count") != len(blocks) or end.get("as_of") != begin.get(
                "as_of"
            ):
                raise ProtocolError(
                    "snapshot count or sequence does not match its header"
                )
        ops = self._reconciler.apply_snapshot(
            str(begin.get("inc")), int(begin.get("as_of", 0)), blocks
        )
        self.counters["snapshots"] += 1
        if ops:
            self._submit(ops)
        if state != KvState.LIVE:
            raise _Disabled()

    async def _follow(self) -> None:
        rec = self._reconciler
        async with self._client.kv_stream(rec.incarnation, rec.last_seq) as frames:
            hello = await anext(frames, None)
            if hello is None or hello.get("t") != Frame.HELLO:
                raise ProtocolError("KV stream does not start with hello")
            if hello.get("state") == KvState.LIVE:
                self._check_block_size(hello.get("block_size"))
            stall_timeout = max(
                self._min_stall_timeout_s,
                self._stall_heartbeats * float(hello.get("hb_s", 2.0)),
            )
            self.state = SyncState.LIVE
            self._live_since = asyncio.get_running_loop().time()
            lagging = 0
            while True:
                try:
                    frame = await asyncio.wait_for(anext(frames), stall_timeout)
                except StopAsyncIteration:
                    raise ConnectionError("KV stream ended") from None
                except asyncio.TimeoutError:
                    raise ConnectionError(
                        f"KV stream silent for {stall_timeout:.0f}s"
                    ) from None
                kind = frame.get("t")
                if kind == Frame.EVENT:
                    ops = rec.apply_event(frame)
                    self.counters["events"] += 1
                    if ops:
                        self._submit(ops)
                    lagging = 0
                elif kind == Frame.HEARTBEAT:
                    lagging = lagging + 1 if frame.get("seq", 0) > rec.last_seq else 0
                    if lagging >= _MAX_LAGGING_HEARTBEATS:
                        raise ConnectionError(
                            "KV stream heartbeats are ahead of delivered events"
                        )
                elif kind == Frame.RESET:
                    reason = frame.get("reason")
                    if reason == ResetReason.SHUTDOWN:
                        raise ConnectionError("bridge is shutting down")
                    # Includes "disabled": the snapshot then reports the KV state.
                    raise ResyncRequired(str(reason))
