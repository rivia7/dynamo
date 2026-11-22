# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL: collectives of multi-node replicas (TP/EP across nodes)."""

from __future__ import annotations

import re
from typing import Any, Sequence


class NcclTransport:
    name = "nccl"

    def restrict(self, hcas: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
        # A leading "=" makes NCCL match HCA names exactly; without it
        # "mlx5_1" would also select mlx5_10 and mlx5_11.
        env = {
            "NCCL_IB_HCA": "=" + ",".join(sorted(hcas)),
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
        }
        return env, {}

    def engagement_patterns(self, hcas: Sequence[str]) -> tuple[str, ...]:
        # e.g. "NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_3:1/IB ; OOB wg0:10.42.0.1<0>"
        return tuple(rf"NET/IB : Using .*\]{re.escape(h)}:" for h in sorted(hcas))
