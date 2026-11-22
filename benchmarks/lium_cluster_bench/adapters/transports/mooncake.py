# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mooncake: SGLang PD KV transfer (the GLM-5.3 recipes' backend).

SGLang exposes the device list as ``--disaggregation-ib-device``; the flag
spelling on SGLang 0.5.18 is UNVERIFIED and checked against ``--help`` in
preflight. With several devices Mooncake picks a topology-near NIC per GPU.
Counters are the authoritative engagement proof; the log pattern is a hint.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


class MooncakeTransport:
    name = "mooncake"
    config_key = "disaggregation-ib-device"

    def restrict(self, hcas: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
        return {}, {self.config_key: ",".join(sorted(hcas))}

    def engagement_patterns(self, hcas: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            rf"(?i)(mooncake|rdma|transfer).*\b{re.escape(h)}\b" for h in sorted(hcas)
        )
