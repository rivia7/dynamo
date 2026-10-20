# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turns oMLX cache hook callbacks into KvEventLog changes.

oMLX calls the hooks on its store thread, engine thread, event loop and
memory-enforcer workers, sometimes while holding its own cache locks. Hooks
therefore only copy what they need and enqueue a job. One worker thread then
verifies and confirms each job against the model's SSD cache manager, outside
every oMLX lock, and applies it to the log in enqueue order.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from array import array
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dynamo_omlx.bridge.event_log import KvEventLog, NewBlock
from dynamo_omlx.protocol import KvState

logger = logging.getLogger(__name__)

# (parent oMLX hash or None, block tokens, oMLX hash, model name, the block's
# oMLX extra keys or None) -> matches oMLX?
VerifyBlock = Callable[[bytes | None, array, bytes, str, tuple[Any, ...] | None], bool]
# Returns the served model's current PagedSSDCacheManager, or None while unloaded.
SsdProvider = Callable[[], Any]

# Consecutive audits that must miss a block before it is withdrawn, so a block
# that oMLX forgets and immediately re-saves does not flap.
_AUDIT_MISSES_TO_REMOVE = 2
_AUDIT_SSD_CLEARED = "ssd_cleared"


@dataclass(frozen=True)
class _StoreJob:
    model_name: str
    block_size: int
    bids: tuple[bytes, ...]
    tokens: array  # len(bids) * block_size tokens, block i at [i * B, (i + 1) * B)
    routing_tokens: array  # tokens as Dynamo's router sees them, same layout
    extra_keys: tuple[tuple[Any, ...] | None, ...]  # oMLX's keys of each block


@dataclass(frozen=True)
class _RemoveJob:
    bids: tuple[bytes, ...]
    reason: str


@dataclass(frozen=True)
class _AuditJob:
    reason: str


_STOP = object()


class KvTracker:
    def __init__(
        self,
        log: KvEventLog,
        *,
        ssd_provider: SsdProvider,
        verify_block: VerifyBlock | None,
        audit_interval_s: float = 30.0,
        mismatch_limit: int = 3,
        error_limit: int = 5,
        fault_window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log = log
        self._ssd_provider = ssd_provider
        self._verify_block = verify_block
        self._audit_interval_s = audit_interval_s
        self._fault_limits = {"hash_mismatch": mismatch_limit}
        self._error_limit = error_limit
        self._fault_window_s = fault_window_s
        self._clock = clock
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._fault_lock = threading.Lock()
        self._faults: dict[str, deque[float]] = {}
        self._missing: dict[bytes, int] = {}
        self._counters: Counter[str] = Counter()
        self.state = KvState.LIVE

    # -- hook side: any thread, cheap, never touches oMLX locks --------------------

    def on_store(
        self,
        paged_cache: Any,
        table: Any,
        tokens: Any,
        *,
        exclude_last: bool = False,
        routing_tokens: Any = None,
        extra_keys: Sequence[tuple[Any, ...] | None] | None = None,
        max_blocks: int | None = None,
    ) -> None:
        """Called after ``BlockAwarePrefixCache.store_cache`` returned ``table``.

        The table covers the whole stored sequence from its first block, restored
        prefix blocks included, and block i holds ``tokens[i * B:(i + 1) * B]``.
        Only full blocks are announceable; a partial tail ends the chain.
        ``routing_tokens`` (default ``tokens``) are what the router sees,
        ``extra_keys[i]`` is the oMLX key block i was hashed with, and only the
        first ``max_blocks`` blocks may be announced.
        """
        if self.state != KvState.LIVE:
            return
        block_size = paged_cache.block_size
        allocated = paged_cache.allocated_blocks
        limit = len(table.block_ids) if max_blocks is None else max_blocks
        bids: list[bytes] = []
        for index, block_id in enumerate(table.block_ids[:limit]):
            block = allocated.get(block_id)
            if (
                block is None
                or not block.block_hash
                or block.token_count != block_size
                or (index + 1) * block_size > len(tokens)
            ):
                break
            bids.append(bytes(block.block_hash))
        if exclude_last and bids and len(bids) == len(table.block_ids):
            # An exact-prefix terminal block is keyed in a separate hash domain.
            bids.pop()
        if bids:
            end = len(bids) * block_size
            routing = tokens if routing_tokens is None else routing_tokens
            self._queue.put(
                _StoreJob(
                    model_name=paged_cache.model_name,
                    block_size=block_size,
                    bids=tuple(bids),
                    tokens=array("I", tokens[:end]),
                    routing_tokens=array("I", routing[:end]),
                    extra_keys=(
                        tuple(extra_keys[: len(bids)])
                        if extra_keys is not None
                        else (None,) * len(bids)
                    ),
                )
            )

    def on_removed(self, bids: Iterable[bytes], reason: str) -> None:
        """Called after oMLX evicted, deleted or forgot SSD cache blocks."""
        tracked = tuple(bid for bid in bids if self._log.contains(bid))
        if tracked:
            self._queue.put(_RemoveJob(bids=tracked, reason=reason))

    def on_ssd_cleared(self) -> None:
        self.request_audit(_AUDIT_SSD_CLEARED)

    def request_audit(self, reason: str) -> None:
        self._queue.put(_AuditJob(reason=reason))

    def record_hook_error(self) -> None:
        """A hook wrapper's own code raised; counts toward the circuit breaker."""
        self._fault("hook_error")

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="dynamo-omlx-kv", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            self._queue.put(_STOP)
            thread.join(timeout)

    def stats(self) -> dict[str, Any]:
        return {"state": self.state, **self._counters, **self._log.stats()}

    # -- worker thread ----------------------------------------------------------------

    def _run(self) -> None:
        next_audit = self._clock() + self._audit_interval_s
        while True:
            try:
                job = self._queue.get(timeout=max(0.0, next_audit - self._clock()))
            except queue.Empty:
                job = _AuditJob(reason="periodic")
            if job is _STOP:
                return
            try:
                self._process(job)
            # One bad job must not stop tracking; the breaker bounds repeats.
            except Exception:  # noqa: BLE001
                logger.exception("KV tracker failed to process %s", type(job).__name__)
                self._fault("worker_error")
            if isinstance(job, _AuditJob):
                next_audit = self._clock() + self._audit_interval_s

    def process_pending(self) -> None:
        """Drain queued jobs on the calling thread instead of the worker (tests)."""
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            if job is not _STOP:
                self._process(job)

    def _process(self, job: Any) -> None:
        if isinstance(job, _StoreJob):
            self._store(job)
        elif isinstance(job, _RemoveJob):
            self._remove(job)
        elif isinstance(job, _AuditJob):
            self._audit(job.reason)

    def _store(self, job: _StoreJob) -> None:
        if self.state != KvState.LIVE:
            return
        ssd = self._ssd_provider()
        if ssd is None:
            return  # unloaded meanwhile; the next completed request re-announces the chain
        self._log.set_block_size(job.block_size)
        size = job.block_size
        parent_bid: bytes | None = None
        new_blocks: list[NewBlock] = []
        for index, bid in enumerate(job.bids):
            if not new_blocks and self._log.contains(bid):
                parent_bid = bid
                continue
            tokens = job.tokens[index * size : (index + 1) * size]
            omlx_parent = job.bids[index - 1] if index else None
            if self._verify_block is not None and not self._verify_block(
                omlx_parent, tokens, bid, job.model_name, job.extra_keys[index]
            ):
                logger.error(
                    "oMLX block %s does not hash from its tokens; not announcing it or its descendants",
                    bid.hex()[:16],
                )
                self._fault("hash_mismatch")
                break
            if not ssd.has_block(bid):
                break  # not persisted; its descendants are unreachable too
            new_blocks.append(
                NewBlock(
                    bid=bid,
                    tokens=job.routing_tokens[index * size : (index + 1) * size],
                )
            )
        if new_blocks and self.state == KvState.LIVE:
            self._counters["blocks_stored"] += self._log.store_chain(
                parent_bid, new_blocks
            )

    def _remove(self, job: _RemoveJob) -> None:
        ssd = self._ssd_provider()
        # With no loaded engine there is no index to consult; trust the hook,
        # which only reports blocks oMLX actually dropped.
        gone = [bid for bid in job.bids if ssd is None or not ssd.has_block(bid)]
        if gone:
            self._counters[f"blocks_removed_{job.reason}"] += self._log.remove_bids(
                gone
            )

    def _audit(self, reason: str) -> None:
        ssd = self._ssd_provider()
        if ssd is None:
            return  # blocks stay valid on disk while the engine is unloaded
        immediate = reason == _AUDIT_SSD_CLEARED
        missing: dict[bytes, int] = {}
        gone: list[bytes] = []
        for bid in self._log.live_bids():
            if ssd.has_block(bid):
                continue
            misses = self._missing.get(bid, 0) + 1
            if immediate or misses >= _AUDIT_MISSES_TO_REMOVE:
                gone.append(bid)
            else:
                missing[bid] = misses
        self._missing = missing
        self._counters["audits"] += 1
        if gone:
            self._counters["blocks_removed_audit"] += self._log.remove_bids(gone)

    def _fault(self, kind: str) -> None:
        limit = self._fault_limits.get(kind, self._error_limit)
        with self._fault_lock:
            now = self._clock()
            window = self._faults.setdefault(kind, deque())
            window.append(now)
            while window and now - window[0] > self._fault_window_s:
                window.popleft()
            self._counters[f"faults_{kind}"] += 1
            tripped = self.state == KvState.LIVE and len(window) >= limit
            if tripped:
                self.state = KvState.DEGRADED
        if tripped:
            logger.error(
                "KV tracking degraded after %d %s events in %.0fs; withdrawing every "
                "announced block until the bridge restarts",
                limit,
                kind,
                self._fault_window_s,
            )
            self._log.clear()
