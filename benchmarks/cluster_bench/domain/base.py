# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base model and hashing shared by every domain module.

Kept separate from ``schema.py`` so domain modules that define their own
models (fabric, node facts, service state, ...) can import the base without
importing ``schema`` (which in turn references their models).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable, strict base model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def stable_hash(payload: Any, length: int = 12) -> str:
    """Deterministic short sha256 of a JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]
