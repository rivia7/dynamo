# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Announced-block table and replayable KV event journal."""

from __future__ import annotations

import asyncio
import itertools
import threading
import uuid
from array import array
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dynamo_omlx.protocol import (
    MAX_BLOCKS_PER_FRAME,
    Frame,
    KvBlock,
    KvOp,
    ResetReason,
    block_hash,
    encode_frame,
)

# Hashes per remove event; one hash is at most 20 JSON characters.
_MAX_HASHES_PER_FRAME = 4096


@dataclass(frozen=True, slots=True)
class NewBlock:
    """A block oMLX persisted: its oMLX block hash and exactly block_size tokens,
    as Dynamo's router sees them (image tokens carry the image's pad value)."""

    bid: bytes
    tokens: array


@dataclass(frozen=True, slots=True)
class _Entry:
    block: KvBlock
    parent_bid: bytes | None


@dataclass(frozen=True)
class KvSnapshot:
    incarnation: str
    as_of: int
    block_size: int | None
    blocks: list[KvBlock]  # parent-first


class KvEventLog:
    """Prefix-closed table of announced blocks plus a bounded replay ring.

    Producers may call from any thread. Every mutation holds one lock while it
    assigns the next sequence number and appends a pre-encoded frame, so ring
    order is commit order. A block is inserted only while its parent is live
    and removals take whole subtrees, which keeps the table prefix-closed as
    the Dynamo router requires. Dict insertion order is therefore parent-first.
    """

    def __init__(
        self,
        *,
        ring_max_events: int = 100_000,
        ring_max_bytes: int = 64 * 1024 * 1024,
        incarnation: str | None = None,
    ) -> None:
        self.incarnation = incarnation or uuid.uuid4().hex
        self._ring_max_events = ring_max_events
        self._ring_max_bytes = ring_max_bytes
        self._lock = threading.Lock()
        self._block_size: int | None = None
        self._entries: dict[bytes, _Entry] = {}
        self._children: dict[bytes, set[bytes]] = {}
        self._bid_by_ext: dict[int, bytes] = {}
        self._seq = 0
        self._ring: deque[tuple[int, bytes]] = deque()
        self._ring_bytes = 0
        self._ext_collisions = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wakeup: asyncio.Event | None = None

    # -- queries --------------------------------------------------------------

    @property
    def block_size(self) -> int | None:
        return self._block_size

    @property
    def last_seq(self) -> int:
        return self._seq

    @property
    def floor(self) -> int:
        """Oldest sequence number still replayable."""
        with self._lock:
            return self._floor_locked()

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, bid: bytes) -> bool:
        return bid in self._entries

    def live_bids(self) -> list[bytes]:
        with self._lock:
            return list(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "live_blocks": len(self._entries),
                "seq": self._seq,
                "floor": self._floor_locked(),
                "ring_events": len(self._ring),
                "ring_bytes": self._ring_bytes,
                "ext_collisions": self._ext_collisions,
            }

    def snapshot(self) -> KvSnapshot:
        with self._lock:
            return KvSnapshot(
                incarnation=self.incarnation,
                as_of=self._seq,
                block_size=self._block_size,
                blocks=[entry.block for entry in self._entries.values()],
            )

    def read_after(
        self, after: int, limit: int = 256
    ) -> tuple[str | None, list[bytes]]:
        """Encoded frames with ``seq > after``, or a reset reason if impossible."""
        with self._lock:
            if after > self._seq:
                return ResetReason.AHEAD, []
            floor = self._floor_locked()
            if after < floor - 1:
                return ResetReason.FLOOR, []
            start = after - floor + 1
            return None, [
                data for _, data in itertools.islice(self._ring, start, start + limit)
            ]

    # -- mutations --------------------------------------------------------------

    def set_block_size(self, block_size: int) -> None:
        """Adopt oMLX's block size; a change invalidates every announced block."""
        with self._lock:
            if self._block_size == block_size:
                return
            changed = self._clear_locked()
            self._block_size = block_size
        if changed:
            self._notify()

    def store_chain(self, parent_bid: bytes | None, blocks: Sequence[NewBlock]) -> int:
        """Announce a chain continuing from ``parent_bid`` (None for the root).

        Blocks already live are skipped. Returns the number of blocks added; 0
        when the parent is no longer live, since its descendants cannot be
        reused without it.
        """
        with self._lock:
            block_size = self._block_size
            if block_size is None:
                raise RuntimeError("store_chain called before the block size is known")
            parent = self._entries.get(parent_bid) if parent_bid is not None else None
            if parent_bid is not None and parent is None:
                return 0
            parent_ext = parent.block.ext if parent is not None else None
            added: list[KvBlock] = []
            for new in blocks:
                if len(new.tokens) != block_size:
                    raise ValueError(
                        f"block has {len(new.tokens)} tokens, expected {block_size}"
                    )
                existing = self._entries.get(new.bid)
                if existing is not None:
                    # oMLX hashes chain on the parent, so a live block always sits
                    # at the same position. Anything else is corrupt input.
                    if added or existing.parent_bid != parent_bid:
                        break
                    parent_bid, parent_ext = new.bid, existing.block.ext
                    continue
                ext = block_hash(block_size, parent_ext, new.tokens)
                if ext in self._bid_by_ext:
                    self._ext_collisions += 1
                    break
                block = KvBlock(ext=ext, parent=parent_ext, tokens=new.tokens)
                self._entries[new.bid] = _Entry(block=block, parent_bid=parent_bid)
                self._bid_by_ext[ext] = new.bid
                if parent_bid is not None:
                    self._children.setdefault(parent_bid, set()).add(new.bid)
                added.append(block)
                parent_bid, parent_ext = new.bid, ext
            for start in range(0, len(added), MAX_BLOCKS_PER_FRAME):
                chunk = added[start : start + MAX_BLOCKS_PER_FRAME]
                self._append_locked(
                    {"op": KvOp.STORE, "blocks": [b.to_wire() for b in chunk]}
                )
        if added:
            self._notify()
        return len(added)

    def remove_bids(self, bids: Iterable[bytes]) -> int:
        """Remove blocks and all their descendants. Returns blocks removed."""
        removed: list[int] = []
        with self._lock:
            for bid in bids:
                if bid in self._entries:
                    self._remove_subtree_locked(bid, removed)
            for start in range(0, len(removed), _MAX_HASHES_PER_FRAME):
                self._append_locked(
                    {
                        "op": KvOp.REMOVE,
                        "hashes": removed[start : start + _MAX_HASHES_PER_FRAME],
                    }
                )
        if removed:
            self._notify()
        return len(removed)

    def clear(self) -> None:
        with self._lock:
            changed = self._clear_locked()
        if changed:
            self._notify()

    # -- async consumers ----------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Route wake-ups for :meth:`wait_beyond` to ``loop``."""
        self._loop = loop
        self._wakeup = asyncio.Event()

    def wake_all(self) -> None:
        """Wake every waiter, for example to let streams observe shutdown."""
        self._notify()

    async def wait_beyond(self, seq: int, timeout: float) -> bool:
        """Wait until an event newer than ``seq`` exists, or ``timeout`` passes.

        Must run on the loop given to :meth:`bind_loop`.
        """
        wakeup = self._wakeup
        if wakeup is None:
            raise RuntimeError("bind_loop() must be called before wait_beyond()")
        # Grab the event before checking the sequence so that a commit landing in
        # between still sets the event this waiter holds.
        if self._seq > seq:
            return True
        try:
            await asyncio.wait_for(wakeup.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self._seq > seq

    # -- internals ----------------------------------------------------------------

    def _floor_locked(self) -> int:
        return self._ring[0][0] if self._ring else self._seq + 1

    def _append_locked(self, payload: dict[str, Any]) -> None:
        self._seq += 1
        data = encode_frame({"t": Frame.EVENT, "seq": self._seq, **payload})
        self._ring.append((self._seq, data))
        self._ring_bytes += len(data)
        while len(self._ring) > 1 and (
            len(self._ring) > self._ring_max_events
            or self._ring_bytes > self._ring_max_bytes
        ):
            _, old = self._ring.popleft()
            self._ring_bytes -= len(old)

    def _clear_locked(self) -> bool:
        if not self._entries:
            return False
        self._entries.clear()
        self._children.clear()
        self._bid_by_ext.clear()
        self._append_locked({"op": KvOp.CLEAR})
        return True

    def _remove_subtree_locked(self, root: bytes, removed: list[int]) -> None:
        # Iterative post-order so the event lists leaves before their parents.
        stack: list[tuple[bytes, bool]] = [(root, False)]
        while stack:
            bid, children_done = stack.pop()
            if not children_done:
                stack.append((bid, True))
                stack.extend((child, False) for child in self._children.get(bid, ()))
                continue
            entry = self._entries.pop(bid)
            self._children.pop(bid, None)
            del self._bid_by_ext[entry.block.ext]
            if entry.parent_bid is not None:
                siblings = self._children.get(entry.parent_bid)
                if siblings is not None:
                    siblings.discard(bid)
                    if not siblings:
                        del self._children[entry.parent_bid]
            removed.append(entry.block.ext)

    def _notify(self) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._wake)

    def _wake(self) -> None:
        wakeup, self._wakeup = self._wakeup, asyncio.Event()
        if wakeup is not None:
            wakeup.set()
