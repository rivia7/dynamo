# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-plane transport plugins (NIC restriction + engagement evidence).

Adding a transport is one module implementing ``ports.TransportPlugin`` plus
one line in ``REGISTRY``; no core code changes.
"""

from __future__ import annotations

from benchmarks.lium_cluster_bench.adapters.transports.deepep import DeepEpTransport
from benchmarks.lium_cluster_bench.adapters.transports.mooncake import MooncakeTransport
from benchmarks.lium_cluster_bench.adapters.transports.nccl import NcclTransport
from benchmarks.lium_cluster_bench.adapters.transports.nixl import NixlTransport
from benchmarks.lium_cluster_bench.ports import TransportPlugin

REGISTRY: dict[str, TransportPlugin] = {
    "mooncake": MooncakeTransport(),
    "nixl": NixlTransport(),
    "nccl": NcclTransport(),
    "deepep": DeepEpTransport(),
}


def get_transport(name: str) -> TransportPlugin:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown transport {name!r}; known: {sorted(REGISTRY)}"
        ) from exc
