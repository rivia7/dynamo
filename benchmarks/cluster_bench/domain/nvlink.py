# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVLink off vs on: switching NVLink out of the GPU-GPU path, and the comparison.

How NVLink is taken out of the path, best first:

* ``bw_mode_off`` - ``nvidia-smi nvlink -sBwMode OFF`` on every node (Hopper;
  the nvidia-smi manual: "OFF - Bandwidth is not used. P2P is via PCIe bus";
  DEPRECATED for Blackwell+). Restored with ``-sBwMode FULL``.
* ``link_width_0`` - ``nvidia-smi nvlink -sLWidth 0`` (Blackwell's link-width
  control keeps that many links active and the rest asleep), only when the
  width list (``-sLWidth values``) offers 0. Restored to the widest value.
* ``nccl_no_p2p`` - software only, when neither control works (no root in
  the pod, unsupported GPU): NCCL without P2P and NVLS, SGLang without its
  custom all-reduce. NVLink stays up; GPU-GPU data is staged through host
  memory over PCIe, a lower bound for a PCIe-only server.

A hardware switch counts only when a GPU-GPU copy benchmark afterwards runs
at PCIe speed (at most the PCIe line rate); otherwise NVLink is restored and
the software method is used. GPUs without peer access over PCIe make NCCL
stage through host memory even with NVLink off, so the path is ``host``.

The nvidia-smi output formats of these commands are not documented; the
parsers below are lenient and the copy benchmark is the proof.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.cluster_bench.domain.analysis import noise_pct, pct_delta
from benchmarks.cluster_bench.domain.schema import (
    BandwidthStat,
    NoiseResult,
    NvlinkComparison,
    NvlinkMethod,
    P2pPath,
    PointResult,
)

BW_MODES = ("3QUARTER", "HALF", "MIN", "OFF", "FULL")
GET_BW_MODE = "nvidia-smi nvlink -gBwMode"
LIST_LINK_WIDTHS = "nvidia-smi nvlink -sLWidth values"
GET_LINK_WIDTH = "nvidia-smi nvlink -gLWidth"
# DeepEP's intranode kernels move tokens over NVLink only; such families
# cannot run with NVLink off.
NVLINK_ONLY_TRANSPORTS = ("deepep",)
# SGLang flag the off variant adds (verified against --help at preflight).
NVLINK_OFF_FLAGS = ("--disable-custom-all-reduce",)
# Above this NVLink p99 per GPU during the off run, NVLink still carried
# traffic and the run does not measure a PCIe-only path.
NVLINK_LEAK_GBS = 1.0
# Switching NVLink off on every node, the copy check, and switching back.
TOGGLE_S = 300.0
# NCCL / SGLang settings per path (engine_sglang applies them).
P2P_ENV: Mapping[str, Mapping[str, str]] = {
    # NVLink off in hardware: allow P2P through the PCIe root complex too.
    "pcie": {"NCCL_NVLS_ENABLE": "0", "NCCL_P2P_LEVEL": "SYS"},
    # No P2P at all: NCCL stages GPU-GPU data through host memory (SHM).
    "host": {"NCCL_NVLS_ENABLE": "0", "NCCL_P2P_DISABLE": "1"},
}
P2P_SGLANG: Mapping[str, Any] = {"disable-custom-all-reduce": True}


def set_bw_mode(mode: str) -> str:
    if mode not in BW_MODES:
        raise ValueError(f"unknown NVLink bandwidth mode {mode!r}")
    return f"nvidia-smi nvlink -sBwMode {mode}"


def set_link_width(width: int) -> str:
    return f"nvidia-smi nvlink -sLWidth {int(width)}"


_UNSUPPORTED = re.compile(r"not supported|deprecated|invalid|unknown|error", re.I)


def parse_bw_mode(text: str) -> str | None:
    """The NVLink bandwidth mode in ``-gBwMode`` output; None if absent."""
    if not text or _UNSUPPORTED.search(text):
        return None
    match = re.search(
        r"bandwidth\s*mode\s*[:=]?\s*(3QUARTER|HALF|MIN|OFF|FULL)\b", text, re.I
    )
    if match:
        return match[1].upper()
    found = {m.upper() for m in re.findall(r"\b(3QUARTER|HALF|MIN|OFF|FULL)\b", text)}
    return found.pop() if len(found) == 1 else None


def parse_link_widths(text: str) -> tuple[int, ...]:
    """Link widths offered by ``-sLWidth values`` (GPU indexes ignored)."""
    if not text or re.search(r"not supported|invalid|unknown", text, re.I):
        return ()
    widths: set[int] = set()
    for line in text.splitlines():
        if "width" not in line.lower():
            continue
        rest = line.split(":", 1)[1] if ":" in line else line
        rest = re.sub(r"\bGPU\s+\d+\b", " ", rest)
        rest = re.sub(r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d", " ", rest)
        widths.update(int(n) for n in re.findall(r"\b\d{1,2}\b", rest))
    return tuple(sorted(widths))


def blackwell_plus(arch: str) -> bool:
    """sm100 / sm103 / sm120 and newer: -sBwMode is deprecated there."""
    return arch.startswith("sm1")


def hardware_method(
    arch: str, bw_mode: str | None, widths: Sequence[int]
) -> NvlinkMethod | None:
    """The hardware switch to try first; None when only software is left.

    The bandwidth mode is the documented switch before Blackwell; Blackwell
    and newer deprecate it in favour of the link width, usable only if a
    width of 0 is offered.
    """
    if bw_mode is not None and not blackwell_plus(arch):
        return "bw_mode_off"
    if 0 in widths and max(widths, default=0) > 0:
        return "link_width_0"
    if bw_mode is not None:
        return "bw_mode_off"
    return None


def switch_effective(p2p_gbs: float | None, pcie_gbs: float) -> bool:
    """A GPU-GPU copy no faster than the PCIe line rate: NVLink is out of the path."""
    return p2p_gbs is not None and p2p_gbs <= 1.05 * pcie_gbs


def path_after_switch(peer_pairs: int | None) -> P2pPath:
    """GPU pairs with peer access over PCIe: P2P; none: NCCL stages through host."""
    return "pcie" if peer_pairs else "host"


def nvlink_p99(stats: Sequence[BandwidthStat]) -> float | None:
    """Busiest GPU's NVLink p99 (worse direction); None without NVLink rows."""
    rows = [
        s.p99_gbs
        for s in stats
        if s.plane == "nvlink" and s.method != "unavailable" and "/gpu" in s.scope
    ]
    return max(rows) if rows else None


def _busiest_pcie(stats: Sequence[BandwidthStat]) -> tuple[float, float | None] | None:
    rows = [
        s
        for s in stats
        if s.plane == "pcie"
        and s.method != "unavailable"
        and "/gpu" in s.scope
        and not s.scope.endswith("/p2p")
    ]
    if not rows:
        return None
    top = max(rows, key=lambda s: s.p99_gbs)
    return top.p99_gbs, top.nominal_gbs


def last_switch(events: Sequence[Mapping[str, Any]]) -> tuple[dict, dict | None]:
    """(last "off" record, the "restore" record after it) of NVLINK event data."""
    off: dict = {}
    restore: dict | None = None
    for data in events:
        if data.get("action") == "off":
            off, restore = dict(data), None
        elif data.get("action") == "restore" and off:
            restore = dict(data)
    return off, restore


def left_off(events: Sequence[Mapping[str, Any]]) -> dict | None:
    """The hardware switch still in effect (crash before the restore), if any."""
    off, restore = last_switch(events)
    if not off or off.get("method") not in ("bw_mode_off", "link_width_0"):
        return None
    if restore is not None and restore.get("ok"):
        return None
    return off


def compare(
    on: PointResult | None,
    off: PointResult | None,
    noise: NoiseResult | None,
    *,
    switch: Mapping[str, Any],
    restore: Mapping[str, Any] | None,
    reason: str,
) -> NvlinkComparison:
    """The NVLink comparison from the two points and the switch records."""
    p2p = switch.get("p2p") or {}
    common: dict[str, Any] = dict(
        reason=reason,
        method=switch.get("method"),
        path=switch.get("path"),
        p2p_gbs=p2p.get("median_gbs"),
        peer_pairs=p2p.get("peer_pairs"),
        restored=None
        if switch.get("method") in (None, "nccl_no_p2p")
        else bool(restore and restore.get("ok")),
        notes=tuple(switch.get("notes") or ()),
    )
    if off is None:
        return NvlinkComparison(status="skipped", **common)
    stats = off.bandwidth
    pcie = _busiest_pcie(stats)
    measured: dict[str, Any] = dict(
        family_id=off.candidate.family_id,
        concurrency=off.key.concurrency,
        on_point_id=on.key.id if on else None,
        off_point_id=off.key.id,
        on_tps_per_gpu=on.tps_per_gpu if on else None,
        off_tps_per_gpu=off.tps_per_gpu,
        on_user_tps=on.metrics.user_tps_p50 if on and on.metrics else None,
        off_user_tps=off.metrics.user_tps_p50 if off.metrics else None,
        on_ttft_p50_ms=on.metrics.ttft_p50_ms if on and on.metrics else None,
        off_ttft_p50_ms=off.metrics.ttft_p50_ms if off.metrics else None,
        on_itl_p50_ms=on.metrics.itl_p50_ms if on and on.metrics else None,
        off_itl_p50_ms=off.metrics.itl_p50_ms if off.metrics else None,
        nvlink_p99_gbs=nvlink_p99(stats),
        pcie_p99_gbs=pcie[0] if pcie else None,
        pcie_capacity_gbs=pcie[1] if pcie else None,
    )
    if not off.audit.usable:
        return NvlinkComparison(status="invalid", **common, **measured)
    if on is None or not on.tps_per_gpu or off.tps_per_gpu is None:
        return NvlinkComparison(status="measured", **common, **measured)
    delta = pct_delta(off.tps_per_gpu, on.tps_per_gpu)
    return NvlinkComparison(
        status="measured",
        delta_pct=round(delta, 2),
        within_noise=abs(delta) <= noise_pct(noise),
        **common,
        **measured,
    )
