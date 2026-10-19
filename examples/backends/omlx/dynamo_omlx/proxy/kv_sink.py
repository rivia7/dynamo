# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single producer thread in front of Dynamo's ``KvEventPublisher``.

Dynamo only orders events published from one thread, so every operation goes
through this sink's queue. The publisher is injected, which keeps this module
free of ``dynamo`` imports.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

from dynamo_omlx.proxy.kv_reconciler import KvOperation, RemoveOp, StoreOp

logger = logging.getLogger(__name__)

# Operation batches drained into one publish_batch() call.
_MAX_BATCHES_PER_PUBLISH = 64
_STOP = object()


def to_publisher_events(
    ops: Sequence[KvOperation], block_size: int
) -> list[dict[str, Any]]:
    """Encode operations for ``KvEventPublisher.publish_batch``."""
    events: list[dict[str, Any]] = []
    for op in ops:
        if isinstance(op, StoreOp):
            events.append(
                {
                    "type": "stored",
                    "token_ids": list(itertools.chain.from_iterable(op.tokens)),
                    "num_block_tokens": [block_size] * len(op.hashes),
                    "block_hashes": list(op.hashes),
                    "parent_hash": op.parent,
                }
            )
        elif isinstance(op, RemoveOp):
            events.append({"type": "removed", "block_hashes": list(op.hashes)})
    return events


class KvPublishSink:
    def __init__(
        self,
        publisher: Any,
        block_size: int,
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._publisher = publisher
        self._block_size = block_size
        self._on_error = on_error
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self.published_events = 0
        self.errors = 0

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="dynamo-omlx-kv-publish", daemon=True
            )
            self._thread.start()

    def submit(self, ops: Sequence[KvOperation]) -> None:
        """Queue operations; safe from any thread, never blocks."""
        if ops:
            self._queue.put(list(ops))

    def stop(self, timeout: float = 5.0) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            self._queue.put(_STOP)
            thread.join(timeout)

    def _run(self) -> None:
        stopping = False
        while not stopping:
            batches = [self._queue.get()]
            while len(batches) < _MAX_BATCHES_PER_PUBLISH:
                try:
                    batches.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if _STOP in batches:
                stopping = True
                batches = batches[: batches.index(_STOP)]
            ops = [op for batch in batches for op in batch]
            if ops:
                self._publish(ops)

    def _publish(self, ops: Sequence[KvOperation]) -> None:
        events = to_publisher_events(ops, self._block_size)
        try:
            self._publisher.publish_batch(events)
        # Reported to the engine, which decides whether to exit.
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            logger.exception("publishing %d KV events failed", len(events))
            if self._on_error is not None:
                self._on_error(e)
            return
        self.published_events += len(events)
