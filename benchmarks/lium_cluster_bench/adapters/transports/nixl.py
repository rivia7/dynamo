# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NIXL over UCX (ablation transport for SGLang PD).

UCX without UCX_NET_DEVICES can fail NIXL backend creation, and one affine
NIC per rank is the fast path (recipes/deepseek-v4/README.md). SGLang has
no per-rank NIC mapper, so every rank of an instance gets the instance's
affine NIC list. ``UCX_TLS=cuda_copy,rc`` matches the Beam B200/B300/H200
recipes.
"""

from __future__ import annotations

from typing import Any, Sequence


class NixlTransport:
    name = "nixl"

    def restrict(self, hcas: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
        env = {
            "UCX_NET_DEVICES": ",".join(f"{h}:1" for h in sorted(hcas)),
            "UCX_TLS": "cuda_copy,rc",
        }
        return env, {"disaggregation-transfer-backend": "nixl"}

    def engagement_patterns(self, hcas: Sequence[str]) -> tuple[str, ...]:
        return (r"Backend UCX was instantiated",)
