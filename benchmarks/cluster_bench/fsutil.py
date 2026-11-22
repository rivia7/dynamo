# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-safe file primitives shared by every layer (stdlib only).

The node-side scripts under ``agents/`` keep their own copies on purpose:
they run with a bare interpreter and must not import ``benchmarks.*``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import IO, Any


def canonical_json(obj: Any) -> str:
    """Deterministic compact JSON (sorted keys) for hashing and CRCs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` so readers never observe a partial file.

    The temporary file is unique per call (``mkstemp``), so concurrent writers
    in different threads cannot rename each other's temp file away.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LockTimeout(RuntimeError):
    """Another process held the lock for longer than the timeout."""


class FileLock:
    """Exclusive ``flock`` on ``path``; ``timeout_s=0`` fails immediately.

    Subclasses may set ``timeout_error`` to raise a domain-specific error.
    """

    timeout_error: type[Exception] = LockTimeout

    def __init__(self, path: Path, timeout_s: float = 0.0, poll_s: float = 0.1):
        self.path = path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._fh: IO[str] | None = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        if not self._acquire(fh.fileno()):
            fh.close()
            raise self.timeout_error(
                f"another writer holds {self.path} (waited {self.timeout_s:g}s)"
            )
        self._fh = fh
        return self

    def _acquire(self, fd: int) -> bool:
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(self.poll_s)

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
