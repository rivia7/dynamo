# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the cluster_bench tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.cluster_bench.lium_fakes import FakeLium, install_fake_lium


@pytest.fixture
def fake_lium(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLium:
    return install_fake_lium(tmp_path, monkeypatch)
