# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focus resources of the leader: interconnect needs and KV cache capacity.

Interconnect needs (``interconnect_needs``): for each link kind (NVLink,
GPU PCIe, GPU<->NIC PCIe, IB/RoCE per node) the busiest device's p99 rate
over the 1-s bins of the leader's saturation point, the link's nominal
per-direction capacity, the p99 utilization, and the capacity a link needs
to carry that p99 at ``target_util`` (default 70 %). The smallest NIC count
is read twice: measured (the NIC sweep: fewest NICs whose throughput stays
within the noise floor, with every larger count also holding up) and
estimated (IB need per node / one NIC's rate). NVLink is "needed" when the
GPU-to-GPU need exceeds what one PCIe link per GPU offers.

KV capacity (``kv_capacity``): SGLang logs ``max_total_num_tokens`` once per
instance from TP rank 0 (scheduler.py), which with attention data
parallelism is the pool of one DP rank; every DP rank holds its own pool of
the same size, so an instance holds ``per_rank x dp`` tokens and one request
(which lives on one rank) at most ``per_rank`` tokens, capped by
``context_len``. In-flight context lives on the decode role (PD) or the agg
role; prefill pools hold transient KV and the prefix cache.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.cluster_bench.domain.lifecycle import DeployLifecycle
from benchmarks.cluster_bench.domain.modelarch import ModelArch
from benchmarks.cluster_bench.domain.pointstate import PointState
from benchmarks.cluster_bench.domain.requests import RequestSummary
from benchmarks.cluster_bench.domain.schema import (
    DEFAULT_LINK_TARGET_UTIL,
    BandwidthStat,
    GpuProfile,
    InterconnectNeeds,
    KvCapacity,
    KvRoleCapacity,
    LaunchPlan,
    LinkNeed,
    LinkStatus,
    NicArm,
    PointResult,
)

DEFAULT_TARGET_UTIL = DEFAULT_LINK_TARGET_UTIL
IDLE_FRACTION = 0.01  # below 1 % of the link's capacity the link is idle
IDLE_IB_NOTE = "the leader moves almost no data between nodes: one NIC is enough"
KV_DTYPE_BYTES = {"fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "fp4_e2m1": 0.5}
SGLANG_GB = 1 << 30  # SGLang logs memory in "GB" of 1024**3 bytes (memory_pool.GB)


# --------------------------------------------------------------------------
# Interconnect
# --------------------------------------------------------------------------


def _device_stats(stats: Sequence[BandwidthStat], plane: str) -> list[BandwidthStat]:
    """Per-GPU rows of a plane ("<node>/gpu<i>"); derived P2P rows excluded."""
    return [
        st
        for st in stats
        if st.plane == plane
        and st.method != "unavailable"
        and "/gpu" in st.scope
        and not st.scope.endswith("/p2p")
    ]


def _node_stats(stats: Sequence[BandwidthStat], plane: str) -> list[BandwidthStat]:
    return [
        st
        for st in stats
        if st.plane == plane
        and st.method != "unavailable"
        and "/" not in st.scope
        and st.scope != "cluster"
    ]


def _busiest(
    rows: Sequence[BandwidthStat],
) -> tuple[str, float, float, float | None] | None:
    """(scope, p99, peak, capacity) of the busiest scope, worse direction."""
    by_scope: dict[str, list[BandwidthStat]] = {}
    for st in rows:
        by_scope.setdefault(st.scope, []).append(st)
    best: tuple[str, float, float, float | None] | None = None
    for scope, items in by_scope.items():
        p99 = max(st.p99_gbs for st in items)
        peak = max(st.peak_gbs for st in items)
        caps = [st.nominal_gbs for st in items if st.nominal_gbs]
        if best is None or p99 > best[1]:
            best = (scope, p99, peak, max(caps) if caps else None)
    return best


def _status(p99: float, capacity: float | None, target: float) -> LinkStatus:
    if capacity is None or capacity <= 0:
        return "unavailable"
    if p99 < IDLE_FRACTION * capacity:
        return "idle"
    return "ok" if p99 / capacity <= target else "tight"


def _link(
    kind: str,
    rows: Sequence[BandwidthStat],
    target: float,
    fallback_capacity: float | None = None,
) -> LinkNeed:
    found = _busiest(rows)
    if found is None:
        return LinkNeed(link=kind)  # type: ignore[arg-type]
    scope, p99, peak, capacity = found
    capacity = capacity or fallback_capacity
    return LinkNeed(
        link=kind,  # type: ignore[arg-type]
        scope=scope,
        p99_gbs=round(p99, 3),
        peak_gbs=round(peak, 3),
        capacity_gbs=capacity,
        util_p99=round(p99 / capacity, 4) if capacity else None,
        need_gbs=round(p99 / target, 3),
        status=_status(p99, capacity, target),
    )


def min_nics_measured(arms: Sequence[NicArm]) -> tuple[int | None, str]:
    """Fewest NICs whose throughput holds up, every larger count holding too."""
    usable = [a for a in arms if a.status == "measured" and a.tps_per_gpu is not None]
    if not usable:
        return None, "no usable NIC sweep arm"
    full = max(a.nic_count for a in usable)
    if all(a.nic_count == full for a in usable):
        return None, f"only the {full}-NIC arm was measured"

    def holds(arm: NicArm) -> bool:
        if arm.nic_count == full:
            return True
        if arm.engaged is False or arm.staging_suspected:
            return False
        if arm.within_noise:
            return True
        return arm.delta_vs_full_pct is not None and arm.delta_vs_full_pct > 0

    best = full
    for arm in sorted(usable, key=lambda a: a.nic_count, reverse=True):
        if not holds(arm):
            break
        best = arm.nic_count
    if best == full:
        return best, f"fewer than {full} NICs cost throughput beyond the noise floor"
    return best, f"{best} NIC(s) per node stay within the noise floor of {full}"


def interconnect_needs(
    point: PointResult | None,
    arms: Sequence[NicArm],
    gpu: GpuProfile,
    *,
    target_util: float = DEFAULT_TARGET_UTIL,
    sweep_reason: str = "",
    sweep_family: str | None = None,
) -> InterconnectNeeds | None:
    """What the leader's point demands of each link, and the minimum NIC count.

    ``arms`` come from the NIC sweep, which runs on ``sweep_family``: the
    leader's family, or the best cross-node family when the leader has no
    cross-node traffic. Its minimum then describes that family, not the
    leader (``InterconnectNeeds.sweep_on_leader``).
    """
    if point is None or not point.bandwidth:
        return None
    stats = point.bandwidth
    nvlink = _link(
        "nvlink", _device_stats(stats, "nvlink"), target_util, gpu.nominal_nvlink_gbs
    )
    pcie = _link(
        "pcie", _device_stats(stats, "pcie"), target_util, gpu.nominal_pcie_gbs
    )
    pcie_nic = _link(
        "pcie_nic", _device_stats(stats, "pcie_nic"), target_util, gpu.nominal_pcie_gbs
    )
    ib = _link("ib", _node_stats(stats, "ib"), target_util)
    hca_rows = [st for st in stats if st.plane == "ib" and st.scope.count("/") == 1]
    nic_rate = max((st.nominal_gbs or 0.0 for st in hca_rows), default=0.0) or None
    per_node: dict[str, set[str]] = {}
    for st in hca_rows:
        node, hca = st.scope.split("/", 1)
        per_node.setdefault(node, set()).add(hca)
    nics = max((len(v) for v in per_node.values()), default=0) or None
    notes: list[str] = []
    estimated = None
    if ib.status in ("ok", "tight") and ib.need_gbs and nic_rate:
        estimated = max(1, math.ceil(ib.need_gbs / nic_rate - 1e-9))
    elif ib.status == "idle":
        notes.append(IDLE_IB_NOTE)
        estimated = 1
    measured, reason = min_nics_measured(arms) if arms else (None, sweep_reason)
    needed = None
    if nvlink.status in ("ok", "tight", "idle") and nvlink.need_gbs is not None:
        needed = nvlink.need_gbs > gpu.nominal_pcie_gbs
    return InterconnectNeeds(
        target_util=target_util,
        point_id=point.key.id,
        family_id=point.candidate.family_id,
        sweep_family_id=sweep_family if arms else None,
        links=(nvlink, pcie, pcie_nic, ib),
        nic_rate_gbs=nic_rate,
        nics_per_node=nics,
        min_nics_measured=measured,
        min_nics_reason=reason,
        min_nics_estimated=estimated,
        pcie_alt_gbs=gpu.nominal_pcie_gbs,
        nvlink_needed=needed,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# KV capacity
# --------------------------------------------------------------------------


def _dp_ranks(cfg: Mapping[str, Any]) -> int:
    if not cfg.get("enable-dp-attention"):
        return 1
    dp = cfg.get("data-parallel-size") or cfg.get("dp-size") or 1
    try:
        return max(1, int(dp))
    except (TypeError, ValueError):
        return 1


def _min(values: Sequence[Any]) -> Any:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _kv_dtype_bytes(plan: LaunchPlan) -> float:
    for cfg in plan.rendered_configs.values():
        dtype = str(cfg.get("kv-cache-dtype") or "").lower()
        if dtype in KV_DTYPE_BYTES:
            return KV_DTYPE_BYTES[dtype]
    return 2.0  # "auto": the model dtype (BF16)


def kv_capacity(
    lifecycle: DeployLifecycle | None,
    plan: LaunchPlan | None,
    *,
    arch: ModelArch | None = None,
    requests: RequestSummary | None = None,
    state: PointState | None = None,
) -> KvCapacity | None:
    """KV pools of the leader deployment and the context they can hold.

    Uses the per-role cold-start numbers (the smallest pool of the role) and
    the rendered configs (attention-DP ranks, HiCache size per rank).
    """
    if lifecycle is None or plan is None or not lifecycle.roles:
        return None
    notes: list[str] = []
    roles: list[KvRoleCapacity] = []
    for role in ("agg", "prefill", "decode"):
        rl = lifecycle.roles.get(role)
        if rl is None:
            continue
        instances = sorted({c.instance for c in plan.containers if c.role == role})
        cfgs = [plan.rendered_configs.get(f"{role}:{i}", {}) for i in instances]
        dp = min((_dp_ranks(c) for c in cfgs), default=1)
        per_rank = rl.max_total_num_tokens_min or rl.kv_tokens_min
        bpt = None
        if rl.kv_cache_gb_min and rl.kv_tokens_min:
            bpt = round(rl.kv_cache_gb_min * SGLANG_GB / rl.kv_tokens_min, 1)
        hosts = [c.get("hicache-size") for c in cfgs]
        host_gb = _min([float(h) for h in hosts if isinstance(h, (int, float))])
        host_tokens = None
        if host_gb and bpt:
            host_tokens = int(len(instances) * dp * host_gb * 1e9 // bpt)
        ctx_len = rl.context_len or _min(
            [
                c.get("context-length")
                for c in cfgs
                if isinstance(c.get("context-length"), int)
            ]
        )
        roles.append(
            KvRoleCapacity(
                role=role,
                instances=len(instances),
                dp_ranks=dp,
                per_rank_tokens=per_rank,
                instance_tokens=per_rank * dp if per_rank else None,
                total_tokens=per_rank * dp * len(instances) if per_rank else None,
                kv_gb_per_rank=rl.kv_cache_gb_min,
                bytes_per_token=bpt,
                host_gb_per_rank=host_gb,
                host_tokens_total=host_tokens,
                max_running_requests=rl.max_running_requests_min,
                context_len=ctx_len,
            )
        )
    if not roles:
        return None
    names = [r.role for r in roles]
    serving = "agg" if "agg" in names else ("decode" if "decode" in names else names[0])
    main = next(r for r in roles if r.role == serving)
    max_request = main.per_rank_tokens
    if max_request is not None and main.context_len:
        max_request = min(max_request, int(main.context_len))
    if main.total_tokens is None:
        notes.append("no KV pool size in the cold-start logs")
    p50 = requests.context_p50 if requests else None
    p90 = requests.context_p90 if requests else None
    usage_avg = usage_peak = None
    if state is not None and serving in state.roles:
        brief = state.roles[serving]
        usage_avg, usage_peak = brief.get("kv_usage", "avg"), brief.get(
            "kv_usage", "max"
        )
    total = main.total_tokens
    return KvCapacity(
        roles=tuple(roles),
        serving_role=serving,
        total_tokens=total,
        max_request_tokens=max_request,
        context_p50=p50,
        context_p90=p90,
        requests_at_p50=int(total // p50) if total and p50 else None,
        requests_at_p90=int(total // p90) if total and p90 else None,
        usage_avg=usage_avg,
        usage_peak=usage_peak,
        arch_bytes_per_token=round(arch.kv_bytes_per_token(_kv_dtype_bytes(plan)), 1)
        if arch is not None
        else None,
        notes=tuple(notes),
    )
