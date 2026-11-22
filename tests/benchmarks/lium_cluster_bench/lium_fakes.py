# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles for the lium CLI (used by provider and controller tests)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

TESTS_DIR = Path(__file__).parent
LIUM_FIXTURES = TESTS_DIR / "fixtures" / "lium"
CONTROLLER_FIXTURES = TESTS_DIR / "fixtures" / "controller"
API_KEY = "sk_live_FAKEKEY_0123456789abcdef"
HF_TOKEN = "hf_FAKETOKEN_abcdefghijklmnop"


@dataclass
class FakeLium:
    """Scripts the fake lium binary and reads back the calls it saw."""

    dir: Path

    def script(self, **responses: list[dict[str, Any]]) -> None:
        mapping = {key.replace("_", " "): value for key, value in responses.items()}
        (self.dir / "script.json").write_text(json.dumps(mapping))
        (self.dir / "counts.json").unlink(missing_ok=True)

    def calls(self) -> list[dict[str, Any]]:
        path = self.dir / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def keys(self) -> list[str]:
        return [c["key"] for c in self.calls()]


def install_fake_lium(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLium:
    """Put a ``lium`` wrapper running fixtures/lium/fake_lium.py first on PATH."""
    state = tmp_path / "fake_lium"
    state.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    wrapper = bindir / "lium"
    wrapper.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" -I "{LIUM_FIXTURES / "fake_lium.py"}" "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_LIUM_DIR", str(state))
    monkeypatch.setenv("LIUM_API_KEY", API_KEY)
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)
    fake = FakeLium(state)
    fake.script()
    return fake


def fixture_text(path: Path) -> str:
    """Fixture content without the repo SPDX header lines."""
    return "".join(
        line
        for line in path.read_text().splitlines(keepends=True)
        if not line.startswith("# SPDX-")
    )


def ok(name: str) -> dict[str, Any]:
    return {"rc": 0, "stdout_file": str(LIUM_FIXTURES / name)}


def fail(
    rc: int, stderr_name: str | None = None, stdout_name: str | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {"rc": rc}
    if stderr_name:
        response["stderr_file"] = str(LIUM_FIXTURES / stderr_name)
    if stdout_name:
        response["stdout_file"] = str(LIUM_FIXTURES / stdout_name)
    return response
