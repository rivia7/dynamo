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
        dram_active={"prefill": 0.3, "decode": 0.6},
        peak_mem_tbs=8.0,
        cache_source="usage",
    )
    assert pd is not None and pd.notes == ()
    pre, dec = pd.role("prefill"), pd.role("decode")
    assert pre.tflops_per_gpu == pytest.approx(50.0) and pre.mfu == pytest.approx(0.005)
    assert dec.tflops_per_gpu == pytest.approx(12.5) and dec.tensor_active == 0.2
    assert pd.total_tflops_per_gpu == pytest.approx(31.25)
    # MBU: each role's measured DRAM share, GPU-weighted for the whole cluster
    assert pre.mbu == 0.3 and dec.mem_tbs_per_gpu == pytest.approx(4.8)
    assert pd.total_mbu == pytest.approx(0.45) and pd.peak_mem_tbs_per_gpu == 8.0
    assert pd.total_mem_tbs_per_gpu == pytest.approx(3.6)
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
    assert agg.total_mbu is None and any("MBU not measured" in n for n in agg.notes)
    assert compute_mfu(
        flops_prefill=1.0, flops_decode=1.0, window_s=0.0, role_gpus={"agg": 8},
        precision="fp4", peak_tflops=1.0,
    ) is None  # fmt: skip


def test_d_only_mfu_counts_only_the_decode_flops():
    # the KV arrives by a fake transfer: no GPU ran the requests' prefill
    d_only = compute_mfu(
        flops_prefill=9e15,
        flops_decode=2e15,
        window_s=10.0,
        role_gpus={"decode": 16},
        precision="fp4",
        peak_tflops=10_000.0,
        dram_active={"decode": 0.596},
        peak_mem_tbs=1.398,
    )
    assert d_only is not None and d_only.notes == ()
    assert d_only.role("decode").flops == 2e15
    assert d_only.total_tflops_per_gpu == pytest.approx(12.5)
    assert d_only.total_mfu == pytest.approx(0.00125)
    assert d_only.total_mbu == 0.596
    assert d_only.role("decode").mem_tbs_per_gpu == pytest.approx(0.8332)


def test_mbu_is_measured_and_needs_neither_flops_nor_a_peak():
    # GPM samples but no model FLOPs (config.json unreadable): MBU alone
    only = compute_mfu(
        flops_prefill=None,
        flops_decode=None,
        window_s=None,
        role_gpus={"prefill": 8, "decode": 24},
        precision="fp8",
        peak_tflops=1979.0,
        dram_active={"prefill": 0.3, "decode": 0.6, "agg": 0.9},  # agg: not deployed
        peak_mem_tbs=None,
    )
    assert only is not None and only.window_s is None and only.total_mfu is None
    dec = only.role("decode")
    assert dec.flops is None and dec.tflops_per_gpu is None and dec.mfu is None
    assert dec.mbu == 0.6 and dec.mem_tbs_per_gpu is None
    assert only.total_mbu == pytest.approx(0.525)  # (0.3 x 8 + 0.6 x 24) / 32
    assert only.notes == (
        "model FLOPs unknown: MFU not computed",
        "no memory bandwidth in the GPU profile: MBU not converted to TB/s",
    )
    # a role without samples: no cluster-wide MBU
    partial = compute_mfu(
        flops_prefill=1e15,
        flops_decode=1e15,
        window_s=10.0,
        role_gpus={"prefill": 8, "decode": 8},
        precision="fp8",
        peak_tflops=1979.0,
        dram_active={"decode": 0.6},
        peak_mem_tbs=4.8,
        cache_source="usage",
    )
    assert partial is not None and partial.total_mbu is None
    assert partial.role("prefill").mbu is None and partial.role("decode").mbu == 0.6
    # neither model FLOPs nor GPM samples: nothing to report
    assert compute_mfu(
        flops_prefill=None, flops_decode=None, window_s=None, role_gpus={"agg": 8},
        precision="fp8", peak_tflops=1979.0,
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


GLM53_FLASH = {
    "model_type": "glm5_next",
    "text_config": {
        "model_type": "glm5_next_text",
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 0,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "intermediate_size": 12288,
        "first_k_dense_replace": 3,
        "vocab_size": 154880,
        "num_nextn_predict_layers": 1,
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "linear_attn_config": {
            "full_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
            "head_dim": 128,
            "num_heads": 64,
        },
    },
}


def test_glm_5_3_flash_hybrid_attention():
    """34 KDA linear-attention layers and 11 DSA MLA layers (no RoPE tail)."""
    arch = arch_from_config(GLM53_FLASH)
    assert arch.mla and (arch.attention_layers, arch.linear_layers) == (11, 34)
    assert (arch.dense_layers, arch.moe_layers) == (3, 42)
    assert (arch.qk_dim, arch.v_dim) == (256, 256)
    # the checkpoint's KDA weights: q, k, v, o, beta, decay and output gates
    kda = 3 * 8192 * 4096 + 4096 * 8192 + 64 * 4096 + 2 * (128 * 4096 + 8192 * 128)
    assert arch.linear_attention_params() == kda
    assert 16.5e9 < arch.active_params() < 17e9
    # only the MLA layers keep KV: a BF16 latent plus the FP8 indexer key and
    # its FP32 scale, per token; SGLang's pool holds 1,576,832 tokens in 18.67 GiB
    assert arch.kv_bytes_per_token(2.0) == (512 * 2 + 128 + 4) * 11
    assert arch.kv_bytes_per_token(2.0) * 1_576_832 / 2**30 == pytest.approx(
        18.67, abs=0.005
    )
    # 1000 decode steps at 100k context: capped attention, pooled indexer
    # scores and the KDA state update, on top of the projections
    start, n = 100_000, 1000
    full = attended_sum(start, n, None)
    expected = (
        n * arch.linear_flops_per_token()
        + 2 * 64 * (256 + 256) * attended_sum(start, n, 2048) * 11
        + 2 * 32 * 128 * (full / 4) * 11
        + n * 6 * 64 * 128 * 128 * 34
    )
    assert arch.tokens_flops(start, n) == pytest.approx(expected)


def test_a_stored_arch_from_before_hybrid_attention_still_parses():
    arch = arch_from_config(DEEPSEEK_V3)
    new = {"index_kpool", "index_kpool_compress", "full_attention_layers"}
    new |= {"linear_heads", "linear_head_dim"}
    old = {k: v for k, v in arch.model_dump(mode="json").items() if k not in new}
    again = type(arch).model_validate(old)
    assert again == arch and again.attention_layers == again.layers == 61
    assert again.linear_layers == 0 and again.linear_attention_params() == 0


GLM53 = {
    "model_type": "glm_moe_dsa",
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "first_k_dense_replace": 3,
    "vocab_size": 154880,
    "index_topk": 2048,
    "index_n_heads": 32,
    "index_head_dim": 128,
}


def test_dsa_kv_bytes_match_the_sglang_pools():
    """Per-token KV bytes against SGLang 0.5.21 pools on an RTX PRO 6000D."""
    arch = arch_from_config(GLM53)
    # FP8: FlashMLA's layout (512 FP8 + 4 FP32 scales + 64 BF16 RoPE = 656 B)
    # and the FP8 indexer key with its scale (132 B), in each of 78 layers
    assert arch.kv_bytes_per_token(1.0) == (656 + 132) * 78
    # GLM-5.3-NVFP4, TP8 decode: 248,192 tokens in 14.21 GiB
    assert arch.kv_bytes_per_token(1.0) * 248_192 / 2**30 == pytest.approx(
        14.21, abs=0.005
    )
    # BF16 keeps the plain latent: 2 bytes per element, plus the same key
    assert arch.kv_bytes_per_token(2.0) == ((512 + 64) * 2 + 132) * 78
