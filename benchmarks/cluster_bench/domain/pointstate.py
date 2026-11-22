# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact per-point inference-service state for the summary and the KB.

The full ``ServiceState`` (every metric family of every endpoint) and
``PointStress`` (every GPU, HCA and netdev) are written to the point
directory; ``PointState`` keeps the numbers that explain saturation: running
and queued requests, KV usage, prefix hits, PD queues, frontend and router
pressure, power/thermal throttling, fabric congestion, the bottleneck label
and energy per output token.
"""

from __future__ import annotations

from typing import Mapping

from pydantic import Field

from benchmarks.cluster_bench.domain.base import Frozen
from benchmarks.cluster_bench.domain.gpuactivity import ActivityBrief, brief_activity
from benchmarks.cluster_bench.domain.loadstress import PointStress
from benchmarks.cluster_bench.domain.servicestate import (
    REQUEST_STAGES,
    Bottleneck,
    MetricValue,
    ServiceState,
)

# catalog key -> statistics kept ("avg", "max", "p50", "p99", "last", "total")
BRIEF_STATS: dict[str, tuple[str, ...]] = {
    "running_reqs": ("avg", "max"),
    "queue_reqs": ("avg", "max"),
    "kv_usage": ("avg", "max"),
    "kv_usage_dynamo": ("avg", "max"),
    "used_tokens": ("avg", "max"),
    "max_total_tokens": ("last",),
    "cache_hit_rate": ("avg",),
    "kv_hit_rate_dynamo": ("avg",),
    "gen_throughput": ("avg", "max"),
    "spec_accept_length": ("avg",),
    "spec_accept_rate": ("avg",),
    "retracted_reqs": ("max",),
    "retracted_input_tokens": ("total",),
    "retracted_output_tokens": ("total",),
    "kv_evictable_tokens": ("avg",),
    "evicted_tokens": ("total",),
    "eviction_s": ("p99",),
    "prefill_bootstrap_queue": ("avg", "max"),
    "prefill_inflight_queue": ("avg", "max"),
    "decode_prealloc_queue": ("avg", "max"),
    "decode_transfer_queue": ("avg", "max"),
    "kv_transfer_speed_gbs": ("avg",),
    "kv_transfer_latency_ms": ("avg", "p99"),
    "kv_transfer_bootstrap_ms": ("avg", "p99"),
    "kv_transfer_alloc_ms": ("avg", "p99"),
    "transfer_failed_reqs": ("total",),
    "bootstrap_failed_reqs": ("total",),
    "worker_inflight": ("avg", "max"),
    "ttft": ("p50", "p99"),
    "itl": ("p50", "p99"),
    "queue_time": ("p50", "p99"),
    "fe_inflight": ("avg", "max"),
    "fe_queued": ("avg", "max"),
    "fe_ttft": ("p50", "p99"),
    "fe_itl": ("p50", "p99"),
    "fe_tokenizer_ms": ("p50", "p99"),
    "router_queue_pending": ("avg", "max"),
    "router_kv_hit_rate": ("avg",),
    "router_overhead_ms": ("p50", "p99"),
    "eplb_balancedness": ("avg",),
    **{f"stage_{stage}": ("avg", "p99") for stage in REQUEST_STAGES},
}


class RoleBrief(Frozen):
    endpoints: int = 0
    values: dict[str, float] = Field(default_factory=dict)  # "<metric>.<stat>"
    derived: dict[str, float] = Field(default_factory=dict)  # RoleState.derived

    def get(self, metric: str, stat: str = "avg") -> float | None:
        return self.values.get(f"{metric}.{stat}")


class StressBrief(Frozen):
    gpu_count: int = 0
    energy_j: float | None = None
    power_mean_w: float | None = None  # per GPU
    power_max_w: float | None = None
    temp_max_c: float | None = None
    sm_mhz_mean: float | None = None
    throttled_fraction: float | None = None
    throttle: dict[str, float] = Field(default_factory=dict)
    xmit_wait_fraction_max: float | None = None
    xmit_discards_total: int = 0
    cnp_per_s_total: float | None = None
    ecn_per_s_total: float | None = None
    pause_per_s_total: float | None = None
    eth_discards_total: int = 0
    error_increments: dict[str, int] = Field(default_factory=dict)
    # NVML GPM view (SM / tensor / DRAM activity, regime per role).
    activity: ActivityBrief | None = None


class PointState(Frozen):
    service_source: str = "none"
    roles: dict[str, RoleBrief] = Field(default_factory=dict)
    bottleneck: Bottleneck | None = None
    stress: StressBrief | None = None
    energy_per_token_j: float | None = None
    notes: tuple[str, ...] = ()

    def value(self, role: str, metric: str, stat: str = "avg") -> float | None:
        brief = self.roles.get(role)
        return brief.get(metric, stat) if brief else None

    def derived(
        self, key: str, roles: tuple[str, ...] = ("decode", "agg")
    ) -> float | None:
        """First role (in ``roles`` order) that has the derived ratio."""
        for role in roles:
            brief = self.roles.get(role)
            if brief is not None and key in brief.derived:
                return brief.derived[key]
        return None


def _stat(value: MetricValue, stat: str) -> float | None:
    raw = getattr(value, stat, None)
    return None if raw is None else round(float(raw), 6)


def brief_roles(service: ServiceState) -> dict[str, RoleBrief]:
    out: dict[str, RoleBrief] = {}
    for role, state in sorted(service.roles.items()):
        values: dict[str, float] = {}
        for metric, stats in BRIEF_STATS.items():
            value = state.metrics.get(metric)
            if value is None:
                continue
            for stat in stats:
                number = _stat(value, stat)
                if number is not None:
                    values[f"{metric}.{stat}"] = number
        derived = {k: round(float(v), 6) for k, v in sorted(state.derived.items())}
        if values or derived or state.endpoints:
            out[role] = RoleBrief(
                endpoints=state.endpoints, values=values, derived=derived
            )
    return out


def brief_stress(stress: PointStress) -> StressBrief:
    return StressBrief(
        gpu_count=stress.gpu_count,
        energy_j=stress.energy_j,
        power_mean_w=stress.power_mean_w,
        power_max_w=stress.power_max_w,
        temp_max_c=stress.temp_max_c,
        sm_mhz_mean=stress.sm_mhz_mean,
        throttled_fraction=stress.throttled_fraction,
        throttle=dict(stress.throttle),
        xmit_wait_fraction_max=stress.xmit_wait_fraction_max,
        xmit_discards_total=stress.xmit_discards_total,
        cnp_per_s_total=stress.cnp_per_s_total,
        ecn_per_s_total=stress.ecn_per_s_total,
        pause_per_s_total=stress.pause_per_s_total,
        eth_discards_total=stress.eth_discards_total,
        error_increments=dict(stress.error_increments),
        activity=brief_activity(stress.activity) if stress.activity else None,
    )


def point_state(
    service: ServiceState | None,
    stress: PointStress | None,
    bottleneck: Bottleneck | None,
    energy_per_token_j: float | None,
    notes: tuple[str, ...] = (),
) -> PointState:
    return PointState(
        service_source=service.source if service else "none",
        roles=brief_roles(service) if service else {},
        bottleneck=bottleneck,
        stress=brief_stress(stress) if stress else None,
        energy_per_token_j=energy_per_token_j,
        notes=tuple(notes) + (tuple(service.notes[:5]) if service else ()),
    )


def max_running_by_role(roles: Mapping[str, object]) -> dict[str, int]:
    """{role: max_running_requests_min} from a DeployLifecycle's roles."""
    out: dict[str, int] = {}
    for role, info in roles.items():
        value = getattr(info, "max_running_requests_min", None)
        if isinstance(value, int) and value > 0:
            out[role] = value
    return out
