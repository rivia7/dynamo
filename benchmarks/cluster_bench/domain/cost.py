# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rental price -> cost per token at the measured operating points."""

from __future__ import annotations

from typing import Iterable

from benchmarks.cluster_bench.domain.schema import (
    CostSummary,
    FloorResult,
    ProviderInfo,
)


def usd_per_mtok(
    cluster_per_hour: float, tps_per_gpu: float, gpus: int
) -> float | None:
    """USD per 1e6 output tokens when the whole cluster serves ``tps_per_gpu``."""
    tokens_per_hour = tps_per_gpu * gpus * 3600.0
    if tokens_per_hour <= 0:
        return None
    return cluster_per_hour / tokens_per_hour * 1e6


def node_hour_price(
    provider: ProviderInfo | None, override: float | None, node_count: int
) -> tuple[float, str] | None:
    """(USD per node-hour, source); the CLI value wins over provider.json."""
    if override is not None and override > 0:
        return override, "--price-per-node-hour"
    if provider is None:
        return None
    if provider.price_per_node_hour and provider.price_per_node_hour > 0:
        return provider.price_per_node_hour, provider.provider
    if provider.price_per_hour and provider.price_per_hour > 0 and node_count > 0:
        return provider.price_per_hour / node_count, provider.provider
    return None


def cost_summary(
    *,
    price: tuple[float, str] | None,
    node_count: int,
    gpu_count: int,
    spent_hours: float,
    leader_tps_per_gpu: float | None,
    floors: Iterable[FloorResult] = (),
) -> CostSummary | None:
    if price is None:
        return None
    per_node, source = price
    cluster = per_node * node_count
    floor_cost = {}
    for floor in floors:
        tps = floor.interpolated_tps_per_gpu or floor.tps_per_gpu
        cost = usd_per_mtok(cluster, tps, gpu_count) if tps else None
        if cost is not None:
            floor_cost[f"{floor.user_tps_floor:g}"] = round(cost, 4)
    leader = (
        usd_per_mtok(cluster, leader_tps_per_gpu, gpu_count)
        if leader_tps_per_gpu
        else None
    )
    return CostSummary(
        source=source,
        price_per_node_hour=per_node,
        cluster_per_hour=cluster,
        experiment_usd=round(cluster * spent_hours, 2),
        leader_usd_per_mtok=round(leader, 4) if leader is not None else None,
        floor_usd_per_mtok=floor_cost,
    )
