# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append-only JSONL files with content dedupe and conflict detection.

Existing lines are never rewritten. The only repair is dropping a torn final
line (a crash mid-append leaves an unterminated, unparseable tail); any other
unparseable line is fatal, because silently skipping evidence would corrupt
every view built from it. Callers serialize writers with ``KbLock``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from benchmarks.cluster_bench.domain.schema import stable_hash
from benchmarks.cluster_bench.fsutil import FileLock

logger = logging.getLogger(__name__)

Row = dict[str, Any]


class KbError(RuntimeError):
    """The knowledge base is in a state the writer refuses to touch."""


class KbLockTimeout(KbError):
    """Another export held the KB lock for longer than the timeout."""

    pass


def canonical_line(row: Row) -> str:
    """Compact, key-sorted JSON (the on-disk form of a record)."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path) -> list[Row]:
    """All rows of ``path`` (missing file = no rows); drops a torn final line."""
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.split(b"\n")
    rows: list[Row] = []
    last = len(lines) - 1
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == last:
                logger.warning("dropping torn final line of %s", path)
                with open(path, "r+b") as fh:
                    fh.truncate(len(data) - len(raw))
                break
            raise KbError(f"{path}: corrupt line {index + 1}") from exc
        if not isinstance(row, dict):
            raise KbError(f"{path}: line {index + 1} is not a JSON object")
        rows.append(row)
    return rows


def append_jsonl(path: Path, rows: Sequence[Row]) -> None:
    """Append ``rows`` durably (flush + fsync); never touches existing lines."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        for row in rows:
            fh.write(canonical_line(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class MergeOutcome:
    """What ``merge_rows`` decided for a batch of incoming rows."""

    appended: list[Row] = field(default_factory=list)
    duplicates: int = 0
    conflicts: list[Row] = field(default_factory=list)


def conflict_row(reason: str, existing: Row, incoming: Row, kind: str) -> Row:
    """A ``conflicts.jsonl`` row with a content-derived ``conflict_id``."""
    body = {
        "schema": "cluster_bench.conflict/v1",
        "kind": kind,
        "reason": reason,
        "key": incoming.get("key"),
        "fp": incoming.get("fp"),
        "exp_id": incoming.get("exp_id"),
        "rec_id": incoming.get("rec_id"),
        "existing_rec_id": existing.get("rec_id"),
        "existing": existing,
        "incoming": incoming,
    }
    return {**body, "conflict_id": stable_hash(body, 16)}


def merge_rows(
    existing: Sequence[Row], incoming: Sequence[Row], kind: str
) -> MergeOutcome:
    """Decide, row by row, whether ``incoming`` appends, dedupes or conflicts.

    * same ``rec_id`` and identical content -> duplicate (no-op);
    * same ``rec_id`` but different stored content -> ``content_mismatch``
      (an existing line was edited after it was written);
    * new ``rec_id`` for a ``key`` that already exists -> ``key_exists`` (the
      same logical record re-exported with different values).
    """
    by_id: dict[str, Row] = {}
    by_key: dict[str, Row] = {}
    for row in existing:
        by_id.setdefault(str(row.get("rec_id")), row)
        by_key.setdefault(str(row.get("key")), row)
    outcome = MergeOutcome()
    for row in incoming:
        rec_id, key = str(row["rec_id"]), str(row["key"])
        stored = by_id.get(rec_id)
        if stored is not None:
            if canonical_line(stored) == canonical_line(row):
                outcome.duplicates += 1
            else:
                outcome.conflicts.append(
                    conflict_row("content_mismatch", stored, row, kind)
                )
            continue
        stored = by_key.get(key)
        if stored is not None:
            outcome.conflicts.append(conflict_row("key_exists", stored, row, kind))
            continue
        outcome.appended.append(row)
        by_id[rec_id] = row
        by_key[key] = row
    return outcome


class KbLock(FileLock):
    """Exclusive ``flock`` on ``kb_root/.lock``, waiting up to ``timeout_s``."""

    timeout_error = KbLockTimeout

    def __init__(self, path: Path, timeout_s: float = 60.0, poll_s: float = 0.1):
        super().__init__(path, timeout_s=timeout_s, poll_s=poll_s)
