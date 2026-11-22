# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model FLOPs and KV bytes from config.json, and per-role MFU."""

from __future__ import annotations

import pytest

from benchmarks.cluster_bench.domain.modelarch import (
    SIMULATED_CONFIG,
    arch_from_config,
    attended_sum,
    compute_mfu,
    precision_of,
)
from benchmarks.cluster_bench.domain.requests import (
    parse_record,
    summarize_requests,
    synthetic_records,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

DEEPSEEK_V3 = {k: v for k, v in SIMULATED_CONFIG.items() if not k.startswith("index_")}
LLAMA3_8B = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
}


def test_deepseek_v3_activates_about_37b_parameters():
    """DeepSeek-V3 publishes 37B activated parameters per token."""
    arch = arch_from_config(DEEPSEEK_V3)
    assert arch.mla and arch.moe_layers == 58 and arch.dense_layers == 3
    assert 36.5e9 < arch.active_params() < 37.5e9
    assert arch.linear_flops_per_token() == pytest.approx(2 * arch.active_params())
    # the DSA indexer adds its projections on top
    with_dsa = arch_from_config(SIMULATED_CONFIG)
    assert with_dsa.active_params() > arch.active_params()


def test_dense_gqa_model_and_its_kv_bytes():
    arch = arch_from_config(LLAMA3_8B)
    assert not arch.mla and arch.routed_experts == 0 and arch.dense_layers == 32
    # 8.03B parameters minus the 0.53B input embedding (a lookup, no FLOPs)
    assert 7.4e9 < arch.active_params() < 7.6e9
    assert arch.kv_bytes_per_token(2.0) == 2 * 8 * 128 * 2 * 32  # 128 KiB in BF16
    mla = arch_from_config(DEEPSEEK_V3)
    assert mla.kv_bytes_per_token(2.0) == (512 + 64) * 2 * 61


def test_attended_sum_matches_brute_force():
    for start, n, cap in (
        (0, 10, None),
        (5, 7, None),
        (0, 10, 4),
        (3, 9, 6),
        (20, 5, 8),
    ):
        expected = sum(
            min(p + 1, cap) if cap else p + 1 for p in range(start, start + n)
        )
        assert attended_sum(start, n, cap) == pytest.approx(expected)


def test_sparse_attention_caps_long_context_cost():
    arch = arch_from_config(SIMULATED_CONFIG)  # index_topk 2048
    short = arch.tokens_flops(4_000, 1) - arch.tokens_flops(3_999, 1)
    long = arch.tokens_flops(200_000, 1) - arch.tokens_flops(199_999, 1)
    # beyond top-k only the cheap indexer still grows with the context
    attn = 2 * arch.heads * (arch.qk_dim + arch.v_dim) * arch.layers
    assert long - short < 0.01 * attn * 196_000
    dense = arch_from_config(DEEPSEEK_V3)
    assert dense.tokens_flops(200_000, 1) > arch.tokens_flops(200_000, 1)


def test_config_parsing_rejects_and_unwraps():
    with pytest.raises(ValueError):
        arch_from_config({"model_type": "x"})
    wrapped = {"model_type": "vl", "text_config": dict(LLAMA3_8B)}
    assert arch_from_config(wrapped).hidden == 4096
    assert precision_of("nvfp4") == "fp4" and precision_of("fp8") == "fp8"
    assert precision_of(None) == "bf16"


def test_mfu_splits_prefill_and_decode_over_their_gpus():
    pd = compute_mfu(
        flops_prefill=8e15,
        flops_decode=2e15,
        window_s=10.0,
        role_gpus={"prefill": 16, "decode": 16},
        precision="fp4",
        peak_tflops=10_000.0,
        tensor_active={"decode": 0.2},
        cache_source="usage",
    )
    assert pd is not None and pd.notes == ()
    pre, dec = pd.role("prefill"), pd.role("decode")
    assert pre.tflops_per_gpu == pytest.approx(50.0) and pre.mfu == pytest.approx(0.005)
    assert dec.tflops_per_gpu == pytest.approx(12.5) and dec.tensor_active == 0.2
    assert pd.total_tflops_per_gpu == pytest.approx(31.25)
    agg = compute_mfu(
        flops_prefill=1e15,
        flops_decode=1e15,
        window_s=10.0,
        role_gpus={"agg": 8},
        precision="fp8",
        peak_tflops=None,
    )
    assert agg is not None and [r.role for r in agg.roles] == ["agg"]
    assert agg.total_mfu is None and agg.roles[0].tflops_per_gpu == pytest.approx(25.0)
    assert any("no dense fp8 peak" in n for n in agg.notes)
    assert any("upper bound" in n for n in agg.notes)
    assert compute_mfu(
        flops_prefill=1.0, flops_decode=1.0, window_s=0.0, role_gpus={"agg": 8},
        precision="fp4", peak_tflops=1.0,
    ) is None  # fmt: skip


def test_request_workload_counts_cache_hits_and_flops():
    raw = synthetic_records(
        requests=40, start_s=0.0, duration_s=40.0, ttft_p50_s=1.0, itl_ms=20.0,
        osl=500, isl=80_000, sessions=10,
    )  # fmt: skip
    records = [r for r in (parse_record(x) for x in raw) if r is not None]
    arch = arch_from_config(SIMULATED_CONFIG)
    summary = summarize_requests(records, gpus=8, arch=arch)
    assert summary.cache_source == "usage"
    assert 0 < summary.cached_tokens < summary.prompt_tokens
    assert summary.output_tokens == 40 * 500
    assert summary.context_p50 <= summary.context_p90 <= summary.context_max
    assert summary.flops_prefill > summary.flops_decode > 0
    # the same records without the cache would cost more prefill FLOPs
    for r in records:
        object.__setattr__(r, "cache_read", None)
    cold = summarize_requests(records, gpus=8, arch=arch)
    assert cold.cache_source == "none" and cold.flops_prefill > summary.flops_prefill
    warm = summarize_requests(records, gpus=8, arch=arch, cached_fraction=0.5)
    assert warm.cache_source == "server"
    assert warm.cached_tokens == pytest.approx(0.5 * warm.prompt_tokens, rel=1e-6)
