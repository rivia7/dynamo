# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepEP (NVSHMEM) wide expert parallelism across nodes.

``NVSHMEM_HCA_LIST`` semantics for DeepEP are UNVERIFIED; per-HCA counters
decide whether the restriction engaged. NCCL is restricted alongside because
DeepEP deployments still run NCCL collectives.
"""

from __future__ import annotations

from typing import Any, Sequence

from benchmarks.cluster_bench.adapters.transports.nccl import NcclTransport


class DeepEpTransport:
    name = "deepep"

    def restrict(self, hcas: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
        env, overrides = NcclTransport().restrict(hcas)
        env["NVSHMEM_HCA_LIST"] = ",".join(f"{h}:1" for h in sorted(hcas))
        return env, overrides

    def engagement_patterns(self, hcas: Sequence[str]) -> tuple[str, ...]:
        return NcclTransport().engagement_patterns(hcas)
