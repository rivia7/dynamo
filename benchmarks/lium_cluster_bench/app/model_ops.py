# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model architecture (config.json) and per-point model FLOPs utilization.

The download phase reads the checkpoint's ``config.json`` on node0 and records
the parsed architecture as a MODEL_ARCH event; every measured point then
turns its request workload (prompt tokens not served from the prefix cache,
generated tokens) into model FLOPs (domain/modelarch.py) and divides by the
GPUs of each role, the window and the GPU's dense peak at the checkpoint's
GEMM precision. Fail-soft: without a config or a peak, MFU is absent and a
note says why.
"""

from __future__ import annotations

import json
import logging
import shlex

from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.domain.modelarch import (
    MfuResult,
    ModelArch,
    arch_from_config,
    compute_mfu,
    precision_of,
)
from benchmarks.lium_cluster_bench.domain.pointstate import PointState
from benchmarks.lium_cluster_bench.domain.requests import RequestSummary
from benchmarks.lium_cluster_bench.domain.schema import EventType, GpuProfile, LaunchPlan
from benchmarks.lium_cluster_bench.ports import CommandError

logger = logging.getLogger(__name__)

WORKER_ROLES = ("agg", "prefill", "decode")
# config.json keys kept verbatim in the event (the parsed ModelArch is derived).
CONFIG_KEYS = (
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "n_routed_experts",
    "num_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "first_k_dense_replace",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "num_nextn_predict_layers",
    "max_position_embeddings",
    "torch_dtype",
    "quantization_config",
)


def record_model_arch(ctx: RunContext, model_dir: str) -> ModelArch | None:
    """Read ``<model_dir>/config.json`` on node0 and append a MODEL_ARCH event."""
    path = f"{model_dir.rstrip('/')}/config.json"
    try:
        res = ctx.cluster.run(ctx.node0, f"cat {shlex.quote(path)}", timeout=60)
        cfg = json.loads(res.stdout)
        source = "simulated" if ctx.simulator is not None else "config.json"
        arch = arch_from_config(cfg, source=source)
    except (CommandError, ValueError, TypeError, OSError) as exc:
        ctx.note(f"model config unreadable ({path}): {exc}")
        return None
    subset = {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
    ctx.store.append(
        EventType.MODEL_ARCH,
        {"arch": arch.model_dump(mode="json"), "config": subset, "path": path},
    )
    return arch


def stored_arch(ctx: RunContext) -> ModelArch | None:
    events = ctx.store.of_type(EventType.MODEL_ARCH)
    if not events:
        return None
    try:
        return ModelArch.model_validate(events[-1].data["arch"])
    except (KeyError, ValueError):
        return None


def role_gpus(plan: LaunchPlan) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in plan.containers:
        if c.role in WORKER_ROLES:
            out[c.role] = out.get(c.role, 0) + len(c.gpus)
    return out


def point_mfu(
    *,
    plan: LaunchPlan,
    gpu: GpuProfile,
    variant: str | None,
    requests: RequestSummary | None,
    state: PointState | None,
) -> MfuResult | None:
    """MFU of one point from its request workload (None without FLOPs or window)."""
    if requests is None or requests.flops_prefill is None or not requests.window_s:
        return None
    precision = precision_of(variant)
    tensor: dict[str, float] = {}
    activity = state.stress.activity if state and state.stress else None
    if activity is not None:
        for role, row in activity.roles.items():
            if row.tensor_pct_mean is not None:
                tensor[role] = round(row.tensor_pct_mean / 100.0, 4)
    gpus = role_gpus(plan)
    if not gpus:
        return None
    return compute_mfu(
        flops_prefill=requests.flops_prefill,
        flops_decode=requests.flops_decode or 0.0,
        window_s=requests.window_s,
        role_gpus=gpus,
        precision=precision,
        peak_tflops=gpu.peak_dense_tflops.get(precision),
        tensor_active=tensor,
        cache_source=requests.cache_source,
    )
