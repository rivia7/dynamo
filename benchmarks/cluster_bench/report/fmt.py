# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic number and time formatting shared by the report and the KB.

Every helper accepts ``None`` and returns the caller's placeholder instead of
raising, so a partially populated summary never crashes a renderer.
"""

from __future__ import annotations

import math
import time

DASH = "—"


def _finite(value: float | None) -> bool:
    return value is not None and not (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    )


def num(value: float | None, digits: int = 1, *, dash: str = DASH) -> str:
    """``1234.56 -> "1,234.6"`` with a fixed number of decimals."""
    if not _finite(value):
        return dash
    return f"{value:,.{digits}f}"


def integer(value: float | None, *, dash: str = DASH) -> str:
    """``1234 -> "1,234"``."""
    if not _finite(value):
        return dash
    return f"{int(round(value)):,}"


def gbs(value: float | None, *, dash: str = DASH) -> str:
    """Bandwidth in GB/s: two decimals below 10, one decimal above."""
    if not _finite(value):
        return dash
    return num(value, 2 if abs(value) < 10 else 1, dash=dash)


def ms(value: float | None, *, dash: str = DASH) -> str:
    """Latency in milliseconds: no decimals from 100 ms up."""
    if not _finite(value):
        return dash
    return num(value, 0 if abs(value) >= 100 else 1, dash=dash)


def pct(
    value: float | None, digits: int = 1, *, signed: bool = False, dash: str = DASH
) -> str:
    """A value already in percent units: ``-7.24 -> "-7.2%"``."""
    if not _finite(value):
        return dash
    sign = "+" if signed and value > 0 else ""
    text = f"{value:.{digits}f}"
    if text.lstrip("-").strip("0.") == "":
        text = text.lstrip("-")
        sign = ""
    return f"{sign}{text}%"


def ratio_pct(value: float | None, digits: int = 1, *, dash: str = DASH) -> str:
    """A ratio rendered as percent: ``0.5 -> "50.0%"``."""
    if not _finite(value):
        return dash
    return pct(value * 100.0, digits, dash=dash)


def ts_utc(epoch: float | None, *, dash: str = DASH) -> str:
    """Epoch seconds as ``YYYY-MM-DD HH:MM UTC``."""
    if not _finite(epoch) or epoch <= 0:
        return dash
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(epoch))


def iso_utc(epoch: float | None) -> str | None:
    """Epoch seconds as ISO-8601 UTC (``2026-11-03T18:00:00Z``), or ``None``."""
    if not _finite(epoch) or epoch <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def hours(value: float | None, *, dash: str = DASH) -> str:
    """Hours with one decimal."""
    return num(value, 1, dash=dash)


def gib(num_bytes: float | None, *, dash: str = DASH) -> str:
    """Bytes as GiB with no decimals (``2.0 TiB`` style is not needed here)."""
    if not _finite(num_bytes) or num_bytes <= 0:
        return dash
    return num(num_bytes / float(1 << 30), 0, dash=dash)


def pct_delta(new: float | None, base: float | None) -> float | None:
    """Relative change of ``new`` vs ``base`` in percent, or ``None``."""
    if not _finite(new) or not _finite(base) or base == 0:
        return None
    return (new - base) / base * 100.0
