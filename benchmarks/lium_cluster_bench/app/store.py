# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable experiment state.

* ``EventStore``: an append-only JSONL log. Every line carries a CRC32 of its
  own canonical JSON, is flushed and fsync'ed, and a torn final line (a crash
  mid-write) is dropped on load. Any other corrupt line is fatal: silently
  skipping evidence would corrupt every later decision.
* ``seal_dir`` / ``verify_seal``: freeze a raw-artifact directory with a
  ``SHA256SUMS`` manifest so later analysis can prove it read unmodified data.
* ``RunLock``: a single-writer ``flock`` guard for one experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from benchmarks.lium_cluster_bench.domain.schema import Event, EventType
from benchmarks.lium_cluster_bench.fsutil import (
    FileLock,
    LockTimeout,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    sha256_file,
)

SEAL_MANIFEST = "SHA256SUMS"
SEAL_META = "sealed.json"


class StoreCorruption(RuntimeError):
    pass


class SealMismatch(RuntimeError):
    pass


def _crc(text: str) -> str:
    return f"{zlib.crc32(text.encode()) & 0xFFFFFFFF:08x}"


class EventStore:
    """Append-only, crash-safe event log for one experiment."""

    def __init__(self, path: Path, now: Callable[[], float] = time.time):
        self.path = path
        self._now = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[Event] = self._load()

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def of_type(self, *types: EventType) -> list[Event]:
        wanted = set(types)
        return [e for e in self._events if e.type in wanted]

    def append(self, event_type: EventType | str, data: Mapping[str, Any]) -> Event:
        event = Event(type=EventType(event_type), ts=self._now(), data=dict(data))
        record = event.model_dump(mode="json")
        line = canonical_json(record)
        record["crc"] = _crc(line)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._events.append(event)
        return event

    def _load(self) -> list[Event]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        # A trailing "" is the normal end of file; anything else after the last
        # newline is a torn write that is safe to discard.
        events: list[Event] = []
        last_index = len(lines) - 1
        for index, raw in enumerate(lines):
            if not raw:
                continue
            parsed = self._parse_line(raw)
            if parsed is None:
                if index == last_index:
                    self._truncate_torn_tail(raw)
                    break
                raise StoreCorruption(f"{self.path}: corrupt event at line {index + 1}")
            events.append(parsed)
        return events

    @staticmethod
    def _parse_line(raw: str) -> Event | None:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict) or "crc" not in record:
            return None
        crc = record.pop("crc")
        if _crc(canonical_json(record)) != crc:
            return None
        return Event.model_validate(record)

    def _truncate_torn_tail(self, torn: str) -> None:
        size = self.path.stat().st_size
        with open(self.path, "r+b") as fh:
            fh.truncate(size - len(torn.encode()))


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in (SEAL_MANIFEST, SEAL_META):
            yield path


def seal_dir(root: Path, meta: Mapping[str, Any]) -> str:
    """Write SHA256SUMS + sealed.json and drop write permission. Returns digest."""
    lines = [
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in iter_files(root)
    ]
    manifest = "\n".join(lines) + ("\n" if lines else "")
    atomic_write_text(root / SEAL_MANIFEST, manifest)
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    atomic_write_json(root / SEAL_META, {**dict(meta), "manifest_sha256": digest})
    for path in [*iter_files(root), root / SEAL_MANIFEST, root / SEAL_META]:
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return digest


def verify_seal(root: Path) -> None:
    manifest_path = root / SEAL_MANIFEST
    if not manifest_path.exists():
        raise SealMismatch(f"{root} is not sealed")
    expected: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        expected[rel] = digest
    actual = {p.relative_to(root).as_posix(): sha256_file(p) for p in iter_files(root)}
    if actual != expected:
        changed = sorted(set(actual.items()) ^ set(expected.items()))
        raise SealMismatch(f"{root}: sealed contents changed: {changed[:5]}")


class RunLock(FileLock):
    """Exclusive, non-blocking lock so only one runner drives an experiment."""

    def __enter__(self) -> RunLock:
        try:
            super().__enter__()
        except LockTimeout as exc:
            raise RuntimeError(
                f"another runner holds {self.path}; refusing to start a second writer"
            ) from exc
        if self._fh is not None:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(f"{os.getpid()}\n")
            self._fh.flush()
        return self
