# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact per-point view of switch-port load (uplinks vs downlinks).

A point samples every leaf port serving our HCAs plus the leaves' uplinks
(hundreds of ports on 8 nodes); the full list goes to the point directory,
the summary and the knowledge base keep this aggregate and the hottest ports.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import Field

from benchmarks.lium_cluster_bench.domain.base import Frozen
from benchmarks.lium_cluster_bench.domain.fabric import SwitchPortLoad

HOTTEST = 5


class PortLoadBrief(Frozen):
    switch: str  # description, else GUID
    port: int
    role: str  # "uplink" | "downlink"
    rail: int | None = None
    peer: str | None = None  # our node (downlink) or the spine (uplink)
    util: float | None = None  # max(tx, rx) utilization vs the port rate
    tx_gbs: float | None = None
    rx_gbs: float | None = None
    wait_ratio: float | None = None
    stall_fraction_est: float | None = None


class LinkClassLoad(Frozen):
    ports: int
    util_max: float | None = None
    util_mean: float | None = None
    tx_gbs: float = 0.0  # summed over the ports
    rx_gbs: float = 0.0
    wait_ratio_max: float | None = None
    stall_fraction_max: float | None = None


class SwitchLoadSummary(Frozen):
    sampled: int
    measured: int  # ports with both snapshots and a usable interval
    uplink: LinkClassLoad | None = None
    downlink: LinkClassLoad | None = None
    discards: int = 0
    errors: dict[str, int] = Field(default_factory=dict)
    resets: int = 0
    hottest: tuple[PortLoadBrief, ...] = ()
    notes: tuple[str, ...] = ()


def _util(load: SwitchPortLoad) -> float | None:
    values = [v for v in (load.tx_util, load.rx_util) if v is not None]
    return max(values) if values else None


def _max(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _class(loads: Sequence[SwitchPortLoad]) -> LinkClassLoad | None:
    if not loads:
        return None
    utils = [u for u in (_util(x) for x in loads) if u is not None]
    return LinkClassLoad(
        ports=len(loads),
        util_max=round(max(utils), 4) if utils else None,
        util_mean=round(sum(utils) / len(utils), 4) if utils else None,
        tx_gbs=round(sum(x.tx_gbs or 0.0 for x in loads), 3),
        rx_gbs=round(sum(x.rx_gbs or 0.0 for x in loads), 3),
        wait_ratio_max=_max([x.wait_ratio for x in loads]),
        stall_fraction_max=_max([x.stall_fraction_est for x in loads]),
    )


def _brief(load: SwitchPortLoad) -> PortLoadBrief:
    ref = load.ref
    util = _util(load)
    return PortLoadBrief(
        switch=ref.switch_desc or ref.switch_guid,
        port=ref.port,
        role=ref.role,
        rail=ref.rail,
        peer=ref.peer_node if ref.role == "downlink" else ref.peer_guid,
        util=round(util, 4) if util is not None else None,
        tx_gbs=load.tx_gbs,
        rx_gbs=load.rx_gbs,
        wait_ratio=load.wait_ratio,
        stall_fraction_est=load.stall_fraction_est,
    )


def summarize_switch_load(
    loads: Sequence[SwitchPortLoad], *, hottest: int = HOTTEST
) -> SwitchLoadSummary | None:
    """Aggregate of one point's port loads; None when nothing was sampled."""
    if not loads:
        return None
    measured = [x for x in loads if x.interval_s and not x.reset]
    discards = 0
    errors: dict[str, int] = {}
    for load in measured:
        for name, delta in load.error_deltas.items():
            if delta <= 0:
                continue
            if "Discard" in name:
                discards += delta
            else:
                errors[name] = errors.get(name, 0) + delta
    ranked = sorted(
        (x for x in measured if _util(x) is not None),
        key=lambda x: (-(_util(x) or 0.0), x.ref.key),
    )
    notes = sorted({n for x in loads for n in x.notes})[:10]
    return SwitchLoadSummary(
        sampled=len(loads),
        measured=len(measured),
        uplink=_class([x for x in measured if x.ref.role == "uplink"]),
        downlink=_class([x for x in measured if x.ref.role == "downlink"]),
        discards=discards,
        errors=dict(sorted(errors.items())),
        resets=sum(1 for x in loads if x.reset),
        hottest=tuple(_brief(x) for x in ranked[:hottest]),
        notes=tuple(notes),
    )
