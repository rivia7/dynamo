# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror of the bridge's announced blocks, turned into router operations.

The Dynamo router drops a stored block whose parent it does not know, and
its publisher counts repeated stores of one hash. The reconciler therefore
keeps the announced set prefix-closed, emits stores parent-first and removes
leaf-first, and never announces a hash twice between removals.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dynamo_omlx.protocol import (
    MAX_BLOCKS_PER_FRAME,
    KvBlock,
    KvOp,
    ProtocolError,
    parse_blocks,
    parse_hashes,
)


@dataclass(frozen=True)
class StoreOp:
    """A chain: ``hashes[0]``'s parent is ``parent``, each next block extends the previous."""

    parent: int | None
    hashes: tuple[int, ...]
    tokens: tuple[array, ...]


@dataclass(frozen=True)
class RemoveOp:
    hashes: tuple[int, ...]  # leaf-first


KvOperation = StoreOp | RemoveOp


class ResyncRequired(Exception):
    """The event stream no longer matches local state; a snapshot is needed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _store_ops(blocks: Sequence[KvBlock]) -> list[KvOperation]:
    """Group parent-first ``blocks`` into chains of at most MAX_BLOCKS_PER_FRAME."""
    ops: list[KvOperation] = []
    run: list[KvBlock] = []
    for block in blocks:
        if run and (block.parent != run[-1].ext or len(run) == MAX_BLOCKS_PER_FRAME):
            ops.append(
                StoreOp(
                    run[0].parent,
                    tuple(b.ext for b in run),
                    tuple(b.tokens for b in run),
                )
            )
            run = []
        run.append(block)
    if run:
        ops.append(
            StoreOp(
                run[0].parent, tuple(b.ext for b in run), tuple(b.tokens for b in run)
            )
        )
    return ops


class KvReconciler:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.incarnation: str | None = None
        self.last_seq = 0
        self._announced: dict[
            int, int | None
        ] = {}  # ext -> parent ext, parent-first order
        self._children: dict[int, set[int]] = {}
        self.skipped_blocks = 0

    def __len__(self) -> int:
        return len(self._announced)

    def announced(self) -> list[int]:
        return list(self._announced)

    def apply_snapshot(
        self, incarnation: str, as_of: int, blocks: Sequence[KvBlock]
    ) -> list[KvOperation]:
        """Converge on a snapshot by diffing it against what is announced."""
        desired: dict[int, KvBlock] = {}
        for block in blocks:
            if block.parent is None or block.parent in desired:
                desired[block.ext] = block
            else:
                self.skipped_blocks += 1  # its parent is missing, so it is unreachable
        stale = [ext for ext in self._announced if ext not in desired]
        ops: list[KvOperation] = []
        if stale:
            ops.append(RemoveOp(tuple(reversed(stale))))
            for ext in stale:
                self._forget(ext)
        additions = [
            block for block in desired.values() if block.ext not in self._announced
        ]
        for block in additions:
            self._announce(block)
        ops.extend(_store_ops(additions))
        self.incarnation = incarnation
        self.last_seq = as_of
        return ops

    def apply_event(self, frame: dict[str, Any]) -> list[KvOperation]:
        """Apply one ``ev`` frame. Raises ResyncRequired on gaps or orphans."""
        seq = frame.get("seq")
        if not isinstance(seq, int):
            raise ProtocolError("event without an integer 'seq'")
        if seq <= self.last_seq:
            return []  # already applied (replay overlap)
        if seq != self.last_seq + 1:
            raise ResyncRequired(f"gap: expected seq {self.last_seq + 1}, got {seq}")
        op = frame.get("op")
        if op == KvOp.STORE:
            ops = self._store(parse_blocks(frame.get("blocks"), self.block_size))
        elif op == KvOp.REMOVE:
            ops = self._remove(parse_hashes(frame.get("hashes")))
        elif op == KvOp.CLEAR:
            ops = self.reset()
        else:
            raise ResyncRequired(f"unknown KV op {op!r}")
        self.last_seq = seq
        return ops

    def reset(self) -> list[KvOperation]:
        """Withdraw everything announced."""
        if not self._announced:
            return []
        op = RemoveOp(tuple(reversed(self._announced)))
        self._announced.clear()
        self._children.clear()
        return [op]

    def _store(self, blocks: Sequence[KvBlock]) -> list[KvOperation]:
        additions: list[KvBlock] = []
        for block in blocks:
            if block.ext in self._announced:
                if additions:
                    raise ResyncRequired("store event mixes new and announced blocks")
                continue
            parent_known = block.parent is None or block.parent in self._announced
            if not parent_known:
                raise ResyncRequired("store event references an unknown parent")
            self._announce(block)
            additions.append(block)
        return _store_ops(additions)

    def _remove(self, hashes: Iterable[int]) -> list[KvOperation]:
        removed: list[int] = []
        for ext in hashes:
            if ext in self._announced:
                self._collect_subtree(ext, removed)
        return [RemoveOp(tuple(removed))] if removed else []

    def _collect_subtree(self, root: int, removed: list[int]) -> None:
        # Post-order: descendants the bridge did not list go first, then the root.
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            ext, children_done = stack.pop()
            if not children_done:
                stack.append((ext, True))
                stack.extend((child, False) for child in self._children.get(ext, ()))
                continue
            self._forget(ext)
            removed.append(ext)

    def _announce(self, block: KvBlock) -> None:
        self._announced[block.ext] = block.parent
        if block.parent is not None:
            self._children.setdefault(block.parent, set()).add(block.ext)

    def _forget(self, ext: int) -> None:
        parent = self._announced.pop(ext)
        self._children.pop(ext, None)
        if parent is not None:
            siblings = self._children.get(parent)
            if siblings is not None:
                siblings.discard(ext)
                if not siblings:
                    del self._children[parent]
