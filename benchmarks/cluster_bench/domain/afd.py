# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A-Only: the attention side (A) of attention-FFN disaggregation (AFD).

An A-Only worker is D-Only's decode worker (fake KV transfer, no prefill peer)
laid out as SGLang AFD's attention instance: DP attention with TP1 per rank,
ep 1, the shared expert as its own MLP. The target model's routed experts are
not loaded; at each of its MoE layers a GPU wait kernel stands in for "send the
batch to F, F computes, the results come back" (agents/sglang_afd.py, SGLang
hotfix afd-attn-only). The MTP draft layer keeps its own experts, as in AFD.

This module holds the F model behind that wait, the A<->F traffic estimate and
the A-side config transform. For one MoE layer with T rows, run as m
microbatches of n = T/m rows:

    f(n)   = 2 lat + n (b_in + b_out) P / BW
             + max(W hit(n) / (N_F HBM eff_mem), n 6 k h I / (N_F PEAK eff_flop))
    hit(n) = 1 - (1 - k/E)^n    share of the experts n uniformly routed rows use
    b_in   = h wire + 8 k       hidden state + top-k expert ids and weights
    b_out  = h wire
    P      = N_F                staged: every F rank gets the whole batch and
                                returns a partial sum (SGLang AFD's nccl backend)
             N_F (1 - (1 - 1/N_F)^k)   direct: only the ranks owning the top-k

with h the hidden size, I the expert intermediate size, k experts per token, E
routed experts, W one layer's routed-expert bytes and BW the A<->F bandwidth
of one A instance. The first MoE layer of a forward, and m == 1, wait f(T)
from "now"; a later layer waits until the previous layer's F result plus
m f(T/m), so it takes max(A's own work, m f(T/m)): the microbatch overlap,
credited on the GPU timeline. Simplifications (stated in the report): A runs
the whole batch per layer, routing is uniform, one F serves one A instance (no
queueing), F runs at eff_mem / eff_flop of its peaks. Pure.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, fields
from typing import Any, Sequence

from pydantic import ValidationError

from benchmarks.cluster_bench.domain.fabric import LinkMatrix
from benchmarks.cluster_bench.domain.modelarch import ModelArch
from benchmarks.cluster_bench.domain.schema import (
    AfdFSpec,
    AfdPlan,
    AfdStepRow,
    AfdTraffic,
    GpuProfile,
    Inventory,
    PeakResult,
)

# The SGLang hotfix and worker env of an A-Only worker; agents/sglang_afd.py
# reads the same names (a test keeps them equal).
HOTFIX_ID = "afd-attn-only"
ATTN_ONLY_ENV = "CB_AFD_ATTN_ONLY"
TARGET_LAYERS_ENV = "CB_AFD_TARGET_LAYERS"
F_WAIT_ENV = "CB_AFD_F_WAIT"
F_WAIT_OFF = "off"

# Bytes per routed-expert weight: NVFP4 = 4 bits + one FP8 scale per 16
# values, MXFP4 = 4 bits + one E8M0 scale per 32.
ROUTED_BYTES_PER_PARAM = {"nvfp4": 0.5625, "mxfp4": 0.53125, "fp8": 1.0, "bf16": 2.0}
# Dense peaks an F expert GEMM of that precision may use, best first.
PEAK_ORDER = {
    "nvfp4": ("fp4", "fp8", "bf16"),
    "mxfp4": ("fp4", "fp8", "bf16"),
    "fp8": ("fp8", "bf16"),
    "bf16": ("bf16",),
}
WIRE_BYTES = {"bf16": 2, "fp8": 1}
TOPK_ROW_BYTES = 8  # int32 expert id + fp32 weight per selected expert
# Link fallbacks and latency defaults (UNVERIFIED: no GPU run measured them).
NIC_EFFICIENCY = 0.8  # of the NICs' line rate, without a gdr_send_recv peak
P2P_EFFICIENCY = 0.8  # of the nominal NVLink / PCIe rate, without a p2p peak
NET_SOFTWARE_US = 10.0  # NCCL send/recv on top of a perftest RDMA write
DEFAULT_LAT_US = {"net": 20.0, "intra": 10.0}
# The report's step table, and the slow-link check before the search.
STEP_ROWS = (16, 64, 256, 1024)
SLOW_STEP_ROWS = 256
SLOW_STEP_WARN_S = 0.5
SLOW_STEP_FAIL_S = 10.0
MFU_NOTE = "A-Only: A-side FLOPs only (the routed experts run on F), BF16 peak"
# Keys an A-Only worker drops: expert parallelism and EPLB (A holds no routed
# experts; AFD requires ep 1), SGLang's own batch overlap (AFD has its own,
# simulated here) and the startup prefetch (it reads every expert from disk).
ATTENTION_DROPPED_KEYS = (
    "ep-size",
    "expert-parallel-size",
    "tp-size",
    "dp-size",
    "deepep-mode",
    "enable-eplb",
    "ep-num-redundant-experts",
    "init-expert-location",
    "enable-two-batch-overlap",
    "enable-single-batch-overlap",
    "weight-loader-prefetch-checkpoints",
)
# What attention_config sets besides the instance's TP and DP size.
ATTENTION_SETTINGS: dict[str, Any] = {
    "enable-dp-attention": True,
    "enable-dp-lm-head": True,
    "moe-a2a-backend": "none",
    "disable-shared-experts-fusion": True,
}
# CLI flags an A-Only worker adds to D-Only's (checked against --help).
ATTENTION_FLAGS = frozenset(
    f"--{key}"
    for key in (
        *ATTENTION_SETTINGS,
        "data-parallel-size",
        "speculative-moe-a2a-backend",
    )
)

_INT_KEYS = ("gpus", "microbatches")
_FLOAT_KEYS = ("bw_gbs", "lat_us", "eff_mem", "eff_flop", "mem_fraction")
_TEXT_KEYS = ("link", "layout", "wire", "expert_dtype")


class AfdError(ValueError):
    """A bad --afd-f, or an F model this run cannot resolve."""


# --------------------------------------------------------------------------
# --afd-f
# --------------------------------------------------------------------------


def parse_afd_f(text: str) -> AfdFSpec:
    """``gpus=8,link=net,layout=staged,microbatches=2,wire=bf16,bw_gbs=50,...``

    Every key is optional. ``bw_gbs`` is GB/s per A GPU, one direction;
    ``bw_gbps`` gives the same in Gbit/s.
    """
    values: dict[str, Any] = {}
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        key, sep, raw = item.partition("=")
        key, raw = key.strip().lower(), raw.strip()
        if not sep or not raw:
            raise AfdError(f"--afd-f: {item!r} is not key=value")
        scale = 1.0
        if key == "bw_gbps":
            key, scale = "bw_gbs", 1 / 8
        if key not in (*_INT_KEYS, *_FLOAT_KEYS, *_TEXT_KEYS):
            known = ", ".join((*_INT_KEYS, *_TEXT_KEYS, *_FLOAT_KEYS, "bw_gbps"))
            raise AfdError(f"--afd-f: unknown key {key!r}; use {known}")
        if key in values:
            raise AfdError(f"--afd-f: {key} given twice (bw_gbps is bw_gbs in Gbit/s)")
        try:
            if key in _INT_KEYS:
                values[key] = int(raw)
            elif key in _FLOAT_KEYS:
                values[key] = float(raw) * scale
            else:
                values[key] = raw.lower()
        except ValueError as exc:
            raise AfdError(f"--afd-f: {key}={raw!r} is not a number") from exc
    try:
        return AfdFSpec(**values)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(p) for p in error.get("loc", ())) or "--afd-f"
        raise AfdError(f"--afd-f: {where}: {error.get('msg', exc)}") from exc


def format_afd_f(spec: AfdFSpec) -> str:
    """The ``--afd-f`` text of ``spec``: its non-default keys, in field order."""
    default = AfdFSpec()
    parts = []
    for name in AfdFSpec.model_fields:
        value = getattr(spec, name)
        if value != getattr(default, name):
            parts.append(f"{name}={_number(value)}")
    return ",".join(parts)


def _number(value: Any) -> str:
    if isinstance(value, float):
        short = f"{value:g}"
        return short if float(short) == value else repr(value)
    return str(value)


# --------------------------------------------------------------------------
# The F model
# --------------------------------------------------------------------------


def routed_dtype(variant: str, explicit: str | None = None) -> str:
    """Routed-expert weight precision: ``--afd-f expert_dtype``, else the stem
    of the checkpoint's variant key (``nvfp4-nvidia`` -> nvfp4)."""
    if explicit is not None:
        return explicit
    name = variant.split("-")[0].split("_")[0].lower()
    if name not in ROUTED_BYTES_PER_PARAM:
        raise AfdError(
            f"routed-expert precision of checkpoint {variant!r} is unknown: "
            f"pass --afd-f expert_dtype={'|'.join(ROUTED_BYTES_PER_PARAM)}"
        )
    return name


def routed_layer_bytes(arch: ModelArch, dtype: str) -> float:
    """Routed-expert weight bytes of one MoE layer (gate, up and down)."""
    per_expert = 3.0 * arch.hidden * float(arch.moe_intermediate or 0)
    return arch.routed_experts * per_expert * ROUTED_BYTES_PER_PARAM[dtype]


def default_f_gpus(
    arch: ModelArch, dtype: str, hbm_gb: float, mem_fraction: float
) -> int:
    """Smallest power of two that holds every routed expert in ``mem_fraction``
    of each GPU's HBM and divides the routed experts."""
    total = routed_layer_bytes(arch, dtype) * arch.moe_layers
    room = hbm_gb * 1e9 * mem_fraction
    n = 1
    while n <= arch.routed_experts:
        if arch.routed_experts % n == 0 and total <= n * room:
            return n
        n *= 2
    raise AfdError(
        f"{total / 1e9:.0f} GB of routed experts do not fit F GPUs of "
        f"{hbm_gb:.0f} GB at {mem_fraction:.0%}: pass --afd-f gpus=..."
    )


def fan_out(layout: str, f_gpus: int, top_k: int) -> float:
    """F ranks one row reaches: all of them (staged), or the expected number
    of distinct owners of its top-k experts under uniform routing (direct)."""
    if layout == "staged":
        return float(f_gpus)
    return f_gpus * (1.0 - (1.0 - 1.0 / f_gpus) ** top_k)


def row_bytes(hidden: int, top_k: int, wire: str) -> tuple[float, float]:
    """(A->F, F->A) bytes of one row: hidden state plus top-k ids and weights
    out, the expert output back."""
    width = hidden * WIRE_BYTES[wire]
    return float(width + TOPK_ROW_BYTES * top_k), float(width)


def a_side_arch(arch: ModelArch) -> ModelArch:
    """The model A computes: no routed experts (the router and the shared
    experts stay), for A-side FLOPs and MFU."""
    return arch.model_copy(update={"experts_per_token": 0})


def select_link(
    spec: AfdFSpec,
    *,
    peaks: Sequence[PeakResult],
    inventory: Inventory,
    gpu: GpuProfile,
    links: LinkMatrix | None,
) -> tuple[float, str, float, str]:
    """(GB/s per A GPU, its source, one-way latency in us, its source)."""
    gpn = max(1, len(inventory.nodes[0].gpus)) if inventory.nodes else 1
    if spec.bw_gbs is not None:
        bw, bw_source = spec.bw_gbs, "--afd-f bw_gbs"
    elif spec.link == "net":
        bw, bw_source = _net_bandwidth(peaks, inventory, gpn)
    else:
        bw, bw_source = _intra_bandwidth(peaks, gpu, gpn)
    lats = (
        [t.lat_typical_us for t in links.tests if t.lat_typical_us is not None]
        if links is not None
        else []
    )
    if spec.lat_us is not None:
        lat, lat_source = spec.lat_us, "--afd-f lat_us"
    elif spec.link == "net" and lats:
        median = statistics.median(lats)
        lat = median + NET_SOFTWARE_US
        lat_source = (
            f"perftest ib_write_lat median {median:.1f} us + "
            f"{NET_SOFTWARE_US:.0f} us NCCL software (estimate)"
        )
    else:
        lat, lat_source = DEFAULT_LAT_US[spec.link], "default, not measured"
    return bw, bw_source, lat, lat_source


def _net_bandwidth(
    peaks: Sequence[PeakResult], inventory: Inventory, gpn: int
) -> tuple[float, str]:
    measured = [
        p
        for p in peaks
        if p.test == "gdr_send_recv"
        and p.value_gbs > 0
        and p.details.get("engaged") is not False
    ]
    if measured:
        # All NICs (nic_count None) first, then the most NICs, then the fastest.
        best = max(
            measured, key=lambda p: (p.nic_count is None, p.nic_count or 0, p.value_gbs)
        )
        nnodes = int(best.details.get("nnodes") or 2)
        per_node = float(
            best.details.get("per_node_gbs") or best.value_gbs / max(1, nnodes // 2)
        )
        count = best.nic_count
        nics = (
            "all NICs" if count is None else f"{count} NIC{'' if count == 1 else 's'}"
        )
        return per_node / gpn, (
            f"microbench gdr_send_recv ({nics}): {per_node:.1f} GB/s per node "
            f"/ {gpn} GPUs"
        )
    rate = (
        sum(h.nominal_gbs or 0.0 for h in inventory.nodes[0].active_hcas)
        if inventory.nodes
        else 0.0
    )
    if rate > 0:
        return rate * NIC_EFFICIENCY / gpn, (
            f"inventory: {rate:.1f} GB/s of active NICs x {NIC_EFFICIENCY} / "
            f"{gpn} GPUs (no gdr_send_recv peak)"
        )
    raise AfdError(
        "no inter-node bandwidth for the A<->F link (no gdr_send_recv peak, no "
        "active NIC): pass --afd-f bw_gbs=... or link=intra"
    )


def _intra_bandwidth(
    peaks: Sequence[PeakResult], gpu: GpuProfile, gpn: int
) -> tuple[float, str]:
    rings = [
        float(p.details.get("per_gpu_gbs") or p.value_gbs / gpn)
        for p in peaks
        if p.test == "p2p_ring_aggregate" and p.value_gbs > 0
    ]
    if rings:
        bw = statistics.median(rings)
        return bw, f"microbench p2p_ring_aggregate: {bw:.1f} GB/s per GPU"
    nvlink = bool(gpu.nvlink and gpu.nominal_nvlink_gbs)
    nominal = gpu.nominal_nvlink_gbs if nvlink else gpu.nominal_pcie_gbs
    assert nominal is not None
    return nominal * P2P_EFFICIENCY, (
        f"GPU profile: {nominal:.0f} GB/s {'NVLink' if nvlink else 'PCIe'} x "
        f"{P2P_EFFICIENCY} (no p2p_ring_aggregate peak)"
    )


def _f_peak(gpu: GpuProfile, dtype: str, notes: list[str]) -> tuple[str, float]:
    order = PEAK_ORDER[dtype]
    for precision in order:
        peak = gpu.peak_dense_tflops.get(precision)
        if peak:
            if precision != order[0]:
                notes.append(
                    f"no dense {order[0]} peak in the {gpu.name} profile: F's "
                    f"{dtype} expert GEMMs use the {precision} peak"
                )
            return precision, peak
    raise AfdError(f"the {gpu.name} profile has no dense peak for {dtype} experts")


def resolve_plan(
    spec: AfdFSpec,
    *,
    arch: ModelArch,
    gpu: GpuProfile,
    dtype: str,
    inventory: Inventory,
    peaks: Sequence[PeakResult],
    links: LinkMatrix | None,
) -> AfdPlan:
    """The run's F model from the model config, the GPU profile, the routed
    experts' precision and the cluster's measured links."""
    if not (arch.routed_experts and arch.experts_per_token and arch.moe_intermediate):
        raise AfdError("A-Only needs a MoE model: config.json has no routed experts")
    if not arch.moe_layers:
        raise AfdError("A-Only needs a MoE model: config.json has no MoE layers")
    if gpu.hbm_tbs is None:
        raise AfdError(
            f"the {gpu.name} profile has no hbm_tbs: F's memory time is unknown"
        )
    notes: list[str] = []
    layer_bytes = routed_layer_bytes(arch, dtype)
    total_gb = layer_bytes * arch.moe_layers / 1e9
    if spec.gpus is not None:
        if arch.routed_experts % spec.gpus:
            raise AfdError(
                f"--afd-f gpus={spec.gpus} does not divide the "
                f"{arch.routed_experts} routed experts"
            )
        f_gpus, f_source = spec.gpus, "--afd-f gpus"
        if total_gb / f_gpus > gpu.hbm_gb * spec.mem_fraction:
            notes.append(
                f"F with {f_gpus} GPUs holds {total_gb / f_gpus:.0f} GB of experts "
                f"per GPU, more than {spec.mem_fraction:.0%} of {gpu.hbm_gb:.0f} GB"
            )
    else:
        f_gpus = default_f_gpus(arch, dtype, gpu.hbm_gb, spec.mem_fraction)
        f_source = (
            f"smallest power of two holding {total_gb:.0f} GB of routed experts "
            f"at {spec.mem_fraction:.0%} of {gpu.hbm_gb:.0f} GB"
        )
    precision, peak = _f_peak(gpu, dtype, notes)
    bw, bw_source, lat, lat_source = select_link(
        spec, peaks=peaks, inventory=inventory, gpu=gpu, links=links
    )
    return AfdPlan(
        spec=spec,
        layers=arch.layers,
        moe_layers=arch.moe_layers,
        hidden=arch.hidden,
        moe_intermediate=int(arch.moe_intermediate),
        experts=arch.routed_experts,
        top_k=arch.experts_per_token,
        expert_dtype=dtype,
        layer_bytes=layer_bytes,
        f_gpus=f_gpus,
        f_gpus_source=f_source,
        f_hbm_tbs=gpu.hbm_tbs,
        f_peak_tflops=peak,
        f_peak_precision=precision,
        link_gbs_per_gpu=bw,
        link_source=bw_source,
        lat_us=lat,
        lat_source=lat_source,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class FWait:
    """Per-layer constants of one A instance's F wait (``CB_AFD_F_WAIT``).

    agents/sglang_afd.py parses the same text and computes the same budgets
    (a test pins the two against each other).
    """

    lat_ns: float  # both transfers' latency, per microbatch
    row_ns: float  # wire time of one row (both directions, fan-out included)
    wt_ns: float  # reading every routed expert of a layer on F
    flop_ns: float  # one row's expert GEMMs on F
    top_k: int
    experts: int
    m: int  # microbatches

    def to_env(self) -> str:
        return ",".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))


def f_wait(plan: AfdPlan, n_a: int) -> FWait:
    """The wait constants of an A instance of ``n_a`` GPUs."""
    spec = plan.spec
    b_in, b_out = row_bytes(plan.hidden, plan.top_k, spec.wire)
    link = plan.link_gbs_per_gpu * max(1, n_a)  # GB/s == bytes per ns
    gemm = 6.0 * plan.top_k * plan.hidden * plan.moe_intermediate
    return FWait(
        lat_ns=2.0 * plan.lat_us * 1e3,
        row_ns=(b_in + b_out) * fan_out(spec.layout, plan.f_gpus, plan.top_k) / link,
        wt_ns=plan.layer_bytes / (plan.f_gpus * plan.f_hbm_tbs * 1e3 * spec.eff_mem),
        flop_ns=gemm / (plan.f_gpus * plan.f_peak_tflops * 1e3 * spec.eff_flop),
        top_k=plan.top_k,
        experts=plan.experts,
        m=spec.microbatches,
    )


def hit(wait: FWait, rows: float) -> float:
    """Share of the routed experts ``rows`` uniformly routed rows touch."""
    return 1.0 - (1.0 - wait.top_k / wait.experts) ** rows


def f_ns(wait: FWait, rows: float) -> float:
    """f(n): one microbatch of ``rows`` rows to F and back."""
    if rows <= 0:
        return 0.0
    f_time = max(wait.wt_ns * hit(wait, rows), rows * wait.flop_ns)
    return wait.lat_ns + rows * wait.row_ns + f_time


def wait_budget(wait: FWait, rows: int, first: bool) -> tuple[bool, float]:
    """(serial, ns) of one MoE layer with ``rows`` rows. Serial: wait from now
    (the first MoE layer of a forward, m == 1, fewer rows than microbatches);
    else wait from the previous layer's F result."""
    if rows <= 0:
        return True, 0.0
    m = min(wait.m, rows)
    if first or m <= 1:
        return True, f_ns(wait, rows)
    return False, m * f_ns(wait, rows / m)


def step_rows(
    plan: AfdPlan, n_a: int, rows: Sequence[int] = STEP_ROWS
) -> tuple[AfdStepRow, ...]:
    """The F wait of one layer and one forward at each MoE row count."""
    wait = f_wait(plan, n_a)
    out = []
    for t in rows:
        m = max(1, min(wait.m, t))
        n = t / m
        terms = {
            "latency": wait.lat_ns,
            "link": n * wait.row_ns,
            "memory": wait.wt_ns * hit(wait, n),
            "compute": n * wait.flop_ns,
        }
        # what the worker waits (agents/sglang_afd.py budgets the same way)
        serial = wait_budget(wait, t, first=True)[1]
        overlap = wait_budget(wait, t, first=False)[1]
        out.append(
            AfdStepRow(
                rows=t,
                latency_us=terms["latency"] / 1e3,
                link_us=terms["link"] / 1e3,
                f_mem_us=terms["memory"] / 1e3,
                f_flop_us=terms["compute"] / 1e3,
                serial_us=serial / 1e3,
                overlap_us=overlap / 1e3,
                step_ms=(serial + (plan.moe_layers - 1) * overlap) / 1e6,
                bound=max(terms, key=lambda k: terms[k]),  # type: ignore[arg-type]
            )
        )
    return tuple(out)


def check_step_time(plan: AfdPlan, n_a: int) -> str | None:
    """A warning when the modelled F wait dominates a forward; AfdError when
    it is absurd (a step of many seconds times out the benchmark)."""
    (row,) = step_rows(plan, n_a, (SLOW_STEP_ROWS,))
    seconds = row.step_ms / 1e3
    text = (
        f"simulated F: a forward with {SLOW_STEP_ROWS} MoE rows waits "
        f"{seconds:.2f} s for F on an A instance of {n_a} GPUs ({row.bound}-bound; "
        f"A<->F {plan.link_gbs_per_gpu * n_a:.1f} GB/s, {plan.link_source})"
    )
    if seconds > SLOW_STEP_FAIL_S:
        raise AfdError(
            f"{text}: above {SLOW_STEP_FAIL_S:.0f} s; model another fabric with "
            "--afd-f bw_gbs=...,lat_us=... or link=intra"
        )
    if seconds > SLOW_STEP_WARN_S:
        return f"{text}: the simulated-F numbers will mostly measure this link"
    return None


def traffic(
    plan: AfdPlan,
    *,
    draft_rows: int,
    accept_len: float | None,
    sim_tps_per_gpu: float | None = None,
    null_tps_per_gpu: float | None = None,
) -> AfdTraffic:
    """A<->F bytes per output token and the bandwidth an A GPU needs.

    ``draft_rows``: MoE rows per decode step (MTP verify tokens, else 1);
    ``accept_len``: output tokens per step (None: one row per token).
    """
    b_in, b_out = row_bytes(plan.hidden, plan.top_k, plan.spec.wire)
    p = fan_out(plan.spec.layout, plan.f_gpus, plan.top_k)
    per_token = (
        draft_rows / accept_len if accept_len and accept_len > 0 else float(draft_rows)
    )
    to_f = b_in * p * plan.moe_layers * per_token
    from_f = b_out * p * plan.moe_layers * per_token
    link = plan.link_gbs_per_gpu

    def need(tps: float | None) -> float | None:
        return None if tps is None else max(to_f, from_f) * tps / 1e9

    sim, null = need(sim_tps_per_gpu), need(null_tps_per_gpu)
    return AfdTraffic(
        fan_out=p,
        rows_per_token=per_token,
        to_f_bytes_per_token=to_f,
        from_f_bytes_per_token=from_f,
        link_gbs_per_gpu=link,
        sim_need_gbs_per_gpu=sim,
        sim_util=None if sim is None else sim / link,
        null_need_gbs_per_gpu=null,
        null_util=None if null is None else null / link,
    )


def expected_weights(
    plan: AfdPlan, checkpoint_gb: float, n_a: int
) -> tuple[float, float] | None:
    """A's weight GB per GPU without the target's routed experts and with
    them. DP attention keeps every other weight whole on each GPU (an upper
    bound: the MTP layer's experts count whole too); loaded routed experts
    would be split over the instance (ep 1). None when the checkpoint is not
    larger than its routed experts (no estimate)."""
    rest = checkpoint_gb - plan.routed_gb
    if rest <= 0:
        return None
    return rest, rest + plan.routed_gb / max(1, n_a)


def weights_engaged(weight_gb_per_gpu: float, expected: tuple[float, float]) -> bool:
    """Nearer the size without routed experts than the size with them."""
    return weight_gb_per_gpu < sum(expected) / 2


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------


def attention_config(cfg: dict[str, Any], gpus: int) -> None:
    """SGLang AFD's attention instance over ``gpus`` GPUs: DP attention with
    TP1 per rank, ep 1, MoE all-to-all off, and the shared expert unfused (it
    stays on A while the routed experts are gone). Idempotent."""
    for key in ATTENTION_DROPPED_KEYS:
        cfg.pop(key, None)
    cfg.update(
        {"tensor-parallel-size": gpus, "data-parallel-size": gpus, **ATTENTION_SETTINGS}
    )
    if "speculative-algorithm" in cfg:  # the draft's MoE runs locally on A
        cfg["speculative-moe-a2a-backend"] = "none"


def attention_env(
    env: dict[str, str], plan: AfdPlan | None, *, gpus: int, null_f: bool
) -> None:
    """The worker env of agents/sglang_afd.py. Without a plan (a family's id
    tag) only the gate is set; a launched worker always has one."""
    env[ATTN_ONLY_ENV] = "1"
    if plan is None:
        return
    env[TARGET_LAYERS_ENV] = str(plan.layers)
    env[F_WAIT_ENV] = F_WAIT_OFF if null_f else f_wait(plan, gpus).to_env()
