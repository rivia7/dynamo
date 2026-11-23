# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model architecture from a Hugging Face ``config.json``: FLOPs and KV bytes.

Model FLOPs utilization (MFU) needs the model's own arithmetic, not the
engine's: SGLang 0.5.21's ``--enable-mfu-metrics`` estimator treats every
model as dense (``intermediate_size``, server dtype bytes), which does not fit
MoE + MLA models such as GLM-5.x or DeepSeek-V3. Here the forward FLOPs of one
token are counted from the architecture:

* attention projections: MLA (``q_lora_rank`` / ``kv_lora_rank`` and the
  nope / rope / v head dims) or GQA (``num_key_value_heads`` x ``head_dim``),
  plus the DSA lightning-indexer projections when ``index_n_heads`` exists;
* FFN: dense SwiGLU (3 matrices) on the first ``first_k_dense_replace``
  layers, then routed (``num_experts_per_tok``) plus shared experts of size
  ``moe_intermediate_size`` and the router;
* the LM head; embeddings are a lookup (no FLOPs); MTP draft layers are not
  counted (MFU counts the useful work of the target model);
* attention dot products: 2 x heads x (qk_dim + v_dim) per attended token and
  layer; with DSA a query attends at most ``index_topk`` tokens while the
  indexer scores every previous token (one pooled key per ``index_kpool``
  tokens when it pools them, as GLM-5.3-Flash does);
* hybrid linear attention (``linear_attn_config``, e.g. the KDA layers of
  GLM-5.3-Flash): only the ``full_attn_layers`` it lists have the attention
  above and a per-token KV cache; the other layers have linear-attention
  projections (q, k, v, o and low-rank gates) and a recurrent state update of
  about 6 x heads x head_dim^2 FLOPs per token.

Memory bandwidth utilization (MBU) is measured, not counted: the NVML GPM
DRAM bandwidth utilization of each role's GPUs over the point (telemetry
agent), and that share of the GPU profile's memory bandwidth in TB/s. The
bytes a decode step reads depend on which experts the batch routes to, on KV
replicated across TP ranks, on the DSA indexer scan and on speculative
decoding, which a formula from config.json would not get right.

All counts are per token and per model replica (the whole model), i.e. the
FLOPs one GPU would do if it held the model alone. Pure: no I/O and no
``domain.schema`` import (schema imports this module).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from pydantic import Field

from benchmarks.cluster_bench.domain.base import Frozen

# HF checkpoint variant -> GEMM precision whose peak is the MFU denominator.
PRECISION_OF_VARIANT = {"nvfp4": "fp4", "fp4": "fp4", "fp8": "fp8", "bf16": "bf16"}


class ModelArch(Frozen):
    """The architecture facts the FLOPs and KV formulas need."""

    source: str = "config.json"  # or "simulated"
    model_type: str | None = None
    hidden: int
    layers: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    # MLA (multi-head latent attention)
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_dim: int | None = None
    qk_rope_dim: int | None = None
    v_head_dim: int | None = None
    # DSA (sparse attention with a lightning indexer)
    index_topk: int | None = None
    index_heads: int | None = None
    index_head_dim: int | None = None
    # FFN
    intermediate: int = 0
    moe_intermediate: int | None = None
    routed_experts: int = 0
    experts_per_token: int = 0
    shared_experts: int = 0
    dense_layers: int = 0  # layers with a dense FFN (all of them without MoE)
    mtp_layers: int = 0
    # DSA indexer key pooling: the indexer scores one key per index_kpool tokens;
    # index_kpool_compress is recorded as configured (the KV cache keeps every
    # token's key either way, see kv_bytes_per_token)
    index_kpool: int = 1
    index_kpool_compress: bool = False
    # Hybrid linear attention: layers outside full_attention_layers are linear
    # (KDA) with linear_heads x linear_head_dim q/k/v; None = every layer is full.
    full_attention_layers: int | None = None
    linear_heads: int | None = None
    linear_head_dim: int | None = None

    @property
    def mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def moe_layers(self) -> int:
        return max(0, self.layers - self.dense_layers) if self.routed_experts else 0

    @property
    def attention_layers(self) -> int:
        """Layers with softmax attention and a per-token KV cache."""
        if self.full_attention_layers is None:
            return self.layers
        return min(self.layers, self.full_attention_layers)

    @property
    def linear_layers(self) -> int:
        return self.layers - self.attention_layers

    def _linear_shape(self) -> tuple[float, float]:
        heads = float(self.linear_heads or self.heads)
        return heads, float(self.linear_head_dim or self.head_dim)

    @property
    def qk_dim(self) -> int:
        if self.mla:
            return (self.qk_nope_dim or 0) + (self.qk_rope_dim or 0)
        return self.head_dim

    @property
    def v_dim(self) -> int:
        return (self.v_head_dim or self.head_dim) if self.mla else self.head_dim

    def attention_params(self) -> float:
        """Projection weights of one attention layer (incl. the DSA indexer)."""
        h = float(self.hidden)
        if self.mla:
            qk, rope = float(self.qk_dim), float(self.qk_rope_dim or 0)
            lora_q = float(self.q_lora_rank or 0)
            q = h * lora_q + lora_q * self.heads * qk if lora_q else h * self.heads * qk
            kv_lora = float(self.kv_lora_rank or 0)
            kv = h * (kv_lora + rope) + kv_lora * self.heads * (
                (self.qk_nope_dim or 0) + self.v_dim
            )
            out = self.heads * self.v_dim * h
            index = 0.0
            if self.index_heads and self.index_head_dim:
                q_in = lora_q or h
                index = (
                    q_in * self.index_heads * self.index_head_dim
                    + h * self.index_head_dim
                    + h * self.index_heads
                )
            return q + kv + out + index
        q = h * self.heads * self.head_dim
        kv = 2.0 * h * self.kv_heads * self.head_dim
        return q + kv + self.heads * self.head_dim * h

    def linear_attention_params(self) -> float:
        """Projection weights of one linear-attention (KDA) layer: q, k, v and o,
        the low-rank decay and output gates (hidden -> head_dim -> heads x
        head_dim) and the per-head beta gate."""
        if not self.linear_layers:
            return 0.0
        h = float(self.hidden)
        heads, dim = self._linear_shape()
        inner = heads * dim
        return 4.0 * h * inner + 2.0 * (h * dim + dim * inner) + h * heads

    def active_params(self) -> float:
        """Weights one token is multiplied with (embedding excluded, LM head in)."""
        h = float(self.hidden)
        dense_ffn = 3.0 * h * self.intermediate
        moe_ffn = (self.experts_per_token + self.shared_experts) * 3.0 * h * float(
            self.moe_intermediate or 0
        ) + h * self.routed_experts
        return (
            self.attention_layers * self.attention_params()
            + self.linear_layers * self.linear_attention_params()
            + self.dense_layers * dense_ffn
            + self.moe_layers * moe_ffn
            + h * self.vocab
        )

    def linear_flops_per_token(self) -> float:
        return 2.0 * self.active_params()

    def tokens_flops(self, start: float, n: float) -> float:
        """Forward FLOPs of ``n`` tokens at positions ``start .. start+n-1``.

        Covers prefill (``start`` = cached prefix) and decode (``start`` =
        context so far): each token attends to its causal prefix, capped at
        ``index_topk`` with DSA, and the DSA indexer scores the full prefix.
        """
        if n <= 0:
            return 0.0
        full = attended_sum(start, n, None)
        attended = attended_sum(start, n, self.index_topk) if self.index_topk else full
        flops = n * self.linear_flops_per_token()
        layers = self.attention_layers
        flops += 2.0 * self.heads * (self.qk_dim + self.v_dim) * attended * layers
        if self.index_heads and self.index_head_dim:
            scored = full / self.index_kpool if self.index_kpool > 1 else full
            flops += 2.0 * self.index_heads * self.index_head_dim * scored * layers
        if self.linear_layers:
            heads, dim = self._linear_shape()
            flops += n * 6.0 * heads * dim * dim * self.linear_layers
        return flops

    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        """KV cache bytes of one token in one model replica (all layers).

        MLA keeps one latent (kv_lora + rope) per layer; with DSA, SGLang
        stores an FP8 cache in FlashMLA's layout (the latent in FP8 with an
        FP32 scale per 128 values, the RoPE part in BF16: 656 B for 512 + 64)
        and adds the indexer key: FP8 with an FP32 scale per 128 values, kept
        per token even when the indexer scores pooled keys. On SGLang 0.5.21
        this matches the pools exactly: GLM-5.3 (FP8) 248,192 tokens in
        14.21 GiB, GLM-5.3-Flash (BF16) 1,576,832 tokens in 18.67 GiB. GQA
        keeps K and V for every KV head. Linear-attention layers keep a
        per-request state instead.
        """
        if self.mla:
            latent = float(self.kv_lora_rank or 0)
            rope = float(self.qk_rope_dim or 0)
            if self.index_head_dim and dtype_bytes == 1.0:
                per_layer = latent * (1.0 + 4.0 / 128) + rope * 2.0
            else:
                per_layer = (latent + rope) * dtype_bytes
            if self.index_head_dim:
                per_layer += self.index_head_dim * (1.0 + 4.0 / 128)
            return per_layer * self.attention_layers
        return 2.0 * self.kv_heads * self.head_dim * dtype_bytes * self.attention_layers


def attended_sum(start: float, n: float, cap: int | None) -> float:
    """Sum over positions p in [start, start+n) of min(p + 1, cap)."""
    if n <= 0:
        return 0.0
    a, b = float(start), float(start + n)

    def tri(x: float, y: float) -> float:  # sum of (p + 1) for p in [x, y)
        return (y - x) * (x + y + 1.0) / 2.0 if y > x else 0.0

    if cap is None:
        return tri(a, b)
    k = float(cap)
    return tri(a, min(b, k)) + max(0.0, b - max(a, k)) * k


def _int(cfg: Mapping[str, Any], *names: str, default: int | None = None) -> int | None:
    for name in names:
        value = cfg.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return default


def arch_from_config(
    cfg: Mapping[str, Any], *, source: str = "config.json"
) -> ModelArch:
    """Parse a HF config (or its ``text_config``); ValueError when it lacks the basics."""
    text = cfg.get("text_config")
    if isinstance(text, Mapping) and "hidden_size" in text:
        cfg = text
    hidden = _int(cfg, "hidden_size", "d_model")
    layers = _int(cfg, "num_hidden_layers", "n_layer", "num_layers")
    heads = _int(cfg, "num_attention_heads", "n_head")
    vocab = _int(cfg, "vocab_size", default=0) or 0
    if hidden is None or layers is None or heads is None:
        raise ValueError(
            "config lacks hidden_size / num_hidden_layers / num_attention_heads"
        )
    head_dim = _int(cfg, "head_dim", default=hidden // heads) or hidden // heads
    kv_heads = _int(cfg, "num_key_value_heads", default=heads) or heads
    routed = (
        _int(cfg, "n_routed_experts", "num_experts", "num_local_experts", default=0)
        or 0
    )
    per_token = _int(cfg, "num_experts_per_tok", "moe_topk", default=0) or 0
    if routed and not per_token:
        routed = 0  # an expert count without a routing width is not a MoE we can cost
    shared = _int(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0
    moe_inter = _int(cfg, "moe_intermediate_size")
    if routed and moe_inter is None:
        moe_inter = _int(cfg, "intermediate_size")
    dense_layers = layers
    if routed:
        dense_layers = _int(cfg, "first_k_dense_replace", default=0) or 0
    full_attention = linear_heads = linear_dim = None
    linear = cfg.get("linear_attn_config")
    if isinstance(linear, Mapping) and isinstance(
        linear.get("full_attn_layers"), (list, tuple)
    ):
        full_attention = len(linear["full_attn_layers"])
        linear_heads = _int(linear, "num_heads")
        linear_dim = _int(linear, "head_dim")
    return ModelArch(
        source=source,
        model_type=str(cfg.get("model_type")) if cfg.get("model_type") else None,
        hidden=hidden,
        layers=layers,
        vocab=vocab,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_lora_rank=_int(cfg, "q_lora_rank"),
        kv_lora_rank=_int(cfg, "kv_lora_rank"),
        qk_nope_dim=_int(cfg, "qk_nope_head_dim"),
        qk_rope_dim=_int(cfg, "qk_rope_head_dim"),
        v_head_dim=_int(cfg, "v_head_dim"),
        index_topk=_int(cfg, "index_topk"),
        index_heads=_int(cfg, "index_n_heads"),
        index_head_dim=_int(cfg, "index_head_dim"),
        intermediate=_int(cfg, "intermediate_size", "ffn_hidden_size", default=0) or 0,
        moe_intermediate=moe_inter if routed else None,
        routed_experts=routed,
        experts_per_token=per_token if routed else 0,
        shared_experts=shared if routed else 0,
        dense_layers=dense_layers,
        mtp_layers=_int(cfg, "num_nextn_predict_layers", default=0) or 0,
        index_kpool=_int(cfg, "index_kpool", default=1) or 1,
        index_kpool_compress=bool(cfg.get("index_kpool_compress")),
        full_attention_layers=full_attention,
        linear_heads=linear_heads,
        linear_head_dim=linear_dim,
    )


# Plausible DeepSeek-V3.2-style MoE + MLA + DSA config the dry run serves
# (GLM-5.x reads its real config.json on hardware).
SIMULATED_CONFIG: dict[str, Any] = {
    "model_type": "simulated_moe_mla_dsa",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "vocab_size": 129280,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "index_topk": 2048,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "num_nextn_predict_layers": 1,
}


# --------------------------------------------------------------------------
# MFU
# --------------------------------------------------------------------------


class RoleMfu(Frozen):
    role: str  # "agg" | "prefill" | "decode"
    gpus: int
    flops: float | None = None  # model FLOPs done in the window (None: unknown)
    tflops_per_gpu: float | None = None  # achieved
    mfu: float | None = None  # achieved / peak (0..1)
    tensor_active: float | None = None  # NVML GPM tensor pipes active, mean (0..1)
    # MBU, measured: NVML GPM DRAM bandwidth utilization of the role's GPUs,
    # window mean (0..1), and that share of the profile's memory bandwidth
    mbu: float | None = None
    mem_tbs_per_gpu: float | None = None


class MfuResult(Frozen):
    """Model FLOPs (MFU) and memory bandwidth (MBU) utilization of one point."""

    precision: str  # GEMM precision of the checkpoint ("fp4" | "fp8" | "bf16")
    peak_tflops_per_gpu: float | None = None  # dense peak at that precision
    window_s: float | None = None  # None: model FLOPs unknown (MBU only)
    roles: tuple[RoleMfu, ...] = ()
    total_mfu: float | None = None  # all GPUs together
    total_tflops_per_gpu: float | None = None
    peak_mem_tbs_per_gpu: float | None = None  # memory bandwidth, MBU denominator
    total_mbu: float | None = None  # GPU-weighted over the roles
    total_mem_tbs_per_gpu: float | None = None
    cache_source: str = "none"  # where cached prompt tokens came from
    notes: tuple[str, ...] = Field(default_factory=tuple)

    def role(self, name: str) -> RoleMfu | None:
        return next((r for r in self.roles if r.role == name), None)


def compute_mfu(
    *,
    flops_prefill: float | None,
    flops_decode: float | None,
    window_s: float | None,
    role_gpus: Mapping[str, int],
    precision: str,
    peak_tflops: float | None,
    tensor_active: Mapping[str, float] | None = None,
    dram_active: Mapping[str, float] | None = None,
    peak_mem_tbs: float | None = None,
    cache_source: str = "none",
) -> MfuResult | None:
    """Per-role MFU from model FLOPs and MBU from measured DRAM bandwidth.

    PD splits prefill and decode FLOPs over their own GPUs. MBU is NVML GPM
    ``DRAM_BW_UTIL`` (bandwidth used / the GPU's maximum), not a bytes model:
    the experts a batch routes to, KV replicated across TP ranks, the DSA
    indexer scan and speculative decoding change the bytes a step reads.
    None when neither the FLOPs (with a window) nor DRAM samples are known.
    """
    total_gpus = sum(role_gpus.values())
    if total_gpus <= 0:
        return None
    window = window_s if window_s is not None and window_s > 0 else 0.0
    known = flops_prefill is not None and window > 0
    dram = {r: v for r, v in (dram_active or {}).items() if r in role_gpus}
    if not known and not dram:
        return None
    notes: list[str] = []
    # D-Only: the KV arrives by a fake transfer, so no GPU runs the prefill
    decode_only = set(role_gpus) == {"decode"}
    if not known:
        notes.append("model FLOPs unknown: MFU not computed")
    elif peak_tflops is None:
        notes.append(f"no dense {precision} peak in the GPU profile: MFU not computed")
    if known and cache_source == "none" and not decode_only:
        notes.append(
            "cached prompt tokens unknown: prefill FLOPs (and MFU) are an upper bound"
        )
    if not dram:
        notes.append("no NVML GPM DRAM bandwidth samples: MBU not measured")
    elif not peak_mem_tbs:
        notes.append(
            "no memory bandwidth in the GPU profile: MBU not converted to TB/s"
        )
    tensor_active = tensor_active or {}
    pre, dec = flops_prefill or 0.0, flops_decode or 0.0

    def role(name: str, flops: float) -> RoleMfu:
        gpus = role_gpus[name]
        tflops = flops / window / gpus / 1e12 if known else None
        mbu = dram.get(name)
        return RoleMfu(
            role=name,
            gpus=gpus,
            flops=flops if known else None,
            tflops_per_gpu=round(tflops, 3) if tflops is not None else None,
            mfu=round(tflops / peak_tflops, 5)
            if tflops is not None and peak_tflops
            else None,
            tensor_active=tensor_active.get(name),
            mbu=mbu,
            mem_tbs_per_gpu=round(mbu * peak_mem_tbs, 4)
            if mbu is not None and peak_mem_tbs
            else None,
        )

    roles: list[RoleMfu] = []
    computed = pre + dec
    if "prefill" in role_gpus and "decode" in role_gpus:
        roles = [role("prefill", pre), role("decode", dec)]
    elif decode_only:
        roles = [role("decode", dec)]
        computed = dec
    else:
        main = "agg" if "agg" in role_gpus else next(iter(role_gpus))
        roles = [role(main, computed)]
    total = computed / window / total_gpus / 1e12 if known else None
    weighted = [(r.mbu, r.gpus) for r in roles if r.mbu is not None]
    total_mbu = None
    if weighted and len(weighted) == len(roles):
        total_mbu = round(
            sum(m * g for m, g in weighted) / sum(g for _, g in weighted), 4
        )
    return MfuResult(
        precision=precision,
        peak_tflops_per_gpu=peak_tflops,
        window_s=round(window, 3) if known else None,
        roles=tuple(roles),
        total_mfu=round(total / peak_tflops, 5)
        if total is not None and peak_tflops
        else None,
        total_tflops_per_gpu=round(total, 3) if total is not None else None,
        peak_mem_tbs_per_gpu=peak_mem_tbs,
        total_mbu=total_mbu,
        total_mem_tbs_per_gpu=round(total_mbu * peak_mem_tbs, 4)
        if total_mbu is not None and peak_mem_tbs
        else None,
        cache_source=cache_source,
        notes=tuple(notes),
    )


def precision_of(variant: str | None) -> str:
    return PRECISION_OF_VARIANT.get((variant or "").lower(), "bf16")


def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b - 1e-9)) if b > 0 else 0
