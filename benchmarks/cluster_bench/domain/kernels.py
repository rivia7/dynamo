# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-time breakdown of an Nsight Systems diagnostic window (pure).

The opt-in diagnostic window (``adapters/diagnostics.py``) runs workers under
``nsys profile --capture-range=cudaProfilerApi`` and records a few forward
steps. ``nsys stats --format csv`` then summarizes each report:

* ``cuda_gpu_kern_sum``      one row per kernel name (demangled, templated)
* ``cuda_gpu_mem_time_sum``  one row per memcpy / memset kind
* ``cuda_kern_exec_sum``     one row per (process, thread, device, kernel);
                             used here only for the per-GPU coverage check

This module parses those tables (tolerant of renamed columns, time units,
console preambles and quoted C++ names), classifies kernel names with ordered
regex rules (first match wins) and builds ``KernelBreakdown`` documents that
merge across ranks and reports.

"GPU time" is the sum of kernel and memory-operation durations over every
traced device and stream. Work that overlaps on several streams (DeepEP
communication under two-batch overlap, for example) counts once per stream,
so shares are fractions of summed GPU time, not of wall time.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from benchmarks.cluster_bench.domain.base import Frozen

KernelCategory = Literal[
    "attention",
    "moe_gemm",
    "dense_gemm",
    "moe_routing",
    "deepep",
    "nccl",
    "other_comm",
    "kv_cache",
    "norm_act",
    "quantization",
    "sampling",
    "memory",
    "elementwise",
    "other",
]
RowKind = Literal["kernel", "memop"]
CudaTrace = Literal["hardware", "software", "unknown"]

# Report order (the breakdown itself is sorted by time).
CATEGORIES: tuple[KernelCategory, ...] = (
    "attention",
    "moe_gemm",
    "dense_gemm",
    "moe_routing",
    "deepep",
    "nccl",
    "other_comm",
    "kv_cache",
    "norm_act",
    "quantization",
    "sampling",
    "memory",
    "elementwise",
    "other",
)
CATEGORY_LABELS: dict[KernelCategory, str] = {
    "attention": "attention (incl. sparse-attention indexer and top-k)",
    "moe_gemm": "MoE expert GEMM",
    "dense_gemm": "dense GEMM (projections, shared experts, lm_head)",
    "moe_routing": "MoE gating, top-k and token permutation",
    "deepep": "DeepEP all-to-all (dispatch / combine)",
    "nccl": "NCCL collectives",
    "other_comm": "other communication (custom all-reduce, NVSHMEM, symm-mem)",
    "kv_cache": "KV-cache writes, page tables and transfers",
    "norm_act": "norms, activations and rotary embedding",
    "quantization": "activation quantization and scale layout",
    "sampling": "logits, sampling and speculative verification",
    "memory": "memcpy / memset and copy kernels",
    "elementwise": "generic PyTorch / Triton elementwise, reduce and sort",
    "other": "unclassified",
}
COMMUNICATION: tuple[KernelCategory, ...] = ("deepep", "nccl", "other_comm")

TOP_N = 15
NAME_MAX = 400  # stored kernel names (CUTLASS names run to thousands of chars)
SHORT_MAX = 120


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelRule:
    """One ordered classification rule; ``guess`` marks naming conventions
    inferred from library sources rather than seen in a recorded trace."""

    rule_id: str
    category: KernelCategory
    pattern: str
    guess: bool = False


# First match wins, so specific families precede generic words ("gemm",
# "attention", "quant", "at::native::"). Patterns see the name without a
# leading "void ".
RULES: tuple[KernelRule, ...] = (
    # ---- communication
    KernelRule("nccl", "nccl", r"\bnccl(?:Dev)?Kernel|ncclDevFunc"),
    KernelRule("deepep", "deepep", r"\bdeep_ep::"),
    KernelRule("nvshmem", "other_comm", r"(?i)nvshmem", guess=True),
    KernelRule(
        "custom_allreduce",
        "other_comm",
        r"cross_device_reduce|(?i:custom_?all_?reduce)",
    ),
    KernelRule(
        "fused_allreduce",
        "other_comm",
        r"(?i)allreduce_fusion|ar_fusion|lamport|mnnvl|mscclpp|quick_?reduce",
        guess=True,
    ),
    KernelRule(
        "collective",
        "other_comm",
        r"(?i)all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|multimem"
        r"|symm_mem|one_?shot_|two_?shot_",
        guess=True,
    ),
    # ---- KV cache (before attention: "PagedKVCache" names both)
    KernelRule("kv_append", "kv_cache", r"AppendPagedKVCache|append_paged_kv"),
    KernelRule(
        "kv_store",
        "kv_cache",
        r"(?i)store_?kv_?cache|set_(?:mla_)?kv_buffer|get_mla_kv_buffer"
        r"|reshape_and_cache|concat_and_cache|kv_?cache_(?:store|write|copy)"
        r"|copy_all_layer|transfer_kv|kv_?indices|req_to_token"
        r"|index_k_(?:cache|buffer)|set_index_k|store_index_k|k_quant_and_cache"
        r"|hicache|to_staging|from_staging|page_head",
        guess=True,
    ),
    # ---- attention
    KernelRule(
        "flashinfer_attention",
        "attention",
        r"Batch(?:Prefill|Decode)With|Single(?:Prefill|Decode)With|BatchMLA"
        r"|MLAPagedAttention|BatchPOD|flashinfer::mla::",
    ),
    KernelRule(
        "flash_attention",
        "attention",
        r"\bflash::|flash_fwd|flash_attn|FlashAttn|flash_mla|FlashMLA"
        r"|splitkv_mla|mla_combine",
    ),
    KernelRule(
        "fmha",
        "attention",
        r"(?i)fmha|\bxqa|mha_kernel|sparse_mla|cutlass_mla|mla_kernel",
    ),
    KernelRule(
        "sparse_indexer",
        "attention",
        # sglang::topk_main_kernel: as many calls as the indexer's
        # paged_mqa_logits in GLM-5.3 traces, far fewer than the MoE layers'
        r"mqa_logits|clean_logits|fast_topk|topk_transform|page_table_transform"
        r"|ragged_transform|TopKPageTable|indexer|(?:^|::)topk_main_kernel",
        guess=True,
    ),
    # Sequence mixing of the linear-attention / state-space layers of hybrid
    # models (KDA, gated DeltaNet, Mamba) is those layers' attention.
    KernelRule(
        "linear_attention",
        "attention",
        r"(?i)delta_rule|gated_delta|(?:^|_)kda_|replayssm|causal_conv1d"
        r"|selective_(?:state|scan)|mamba|lightning_attn|linear_attn"
        r"|fused_recurrent|chunk_(?:gla|rwkv|simple_gla)",
        guess=True,
    ),
    # TileLang names a kernel after its prim_func, "main" by default. In SGLang
    # traces that is the DSA sparse MLA attention of the tilelang backend (it
    # runs right after the indexer's top-k); other TileLang kernels can share
    # the name, hence a guess.
    KernelRule("tilelang_main", "attention", r"^main_kernel$", guess=True),
    KernelRule(
        "triton_attention",
        "attention",
        r"^_fwd_(?:grouped_)?kernel(?:_stage\d|_flash_decode\w*)?(?:_\d+d\w*)?$"
        r"|_decode_(?:grouped_)?att|decode_attention|extend_attention"
        r"|prefill_attention|context_attention|merge_state|MergeStates"
        r"|merge_attn_states|concat_mla",
        guess=True,
    ),
    KernelRule("attention", "attention", r"(?i)attention|attn_fwd", guess=True),
    # ---- MoE (routing before GEMM: "moe_fused_gate", "fused_moe_router")
    KernelRule(
        "moe_routing",
        "moe_routing",
        r"(?i)topk_softmax|topk_sigmoid|grouped_topk|topkgating|moe_fused_gate"
        r"|router|routing|moe_align|align_block_size|count_and_sort_expert"
        r"|sort_expert|expert_tokens|moe_sum|expandinputrows|finalizemoe"
        r"|ep_scatter|ep_gather|src2dst|seg_indptr"
        r"|deepep_(?:permute|post_reorder|run_moe)"
        r"|(?:^|_|::)(?:un)?permute(?:_|<|\(|$)|pad_topk|packed_topk|hash_topk"
        r"|eplb|moe::dev::finalize|buildexpertmaps|sortfirsttoken",
        guess=True,
    ),
    KernelRule(
        "deep_gemm_layout",
        "quantization",
        r"\bdeep_gemm::\w*(?:transpose|ue8m0|layout)",
        guess=True,
    ),
    KernelRule(
        "deep_gemm_grouped",
        "moe_gemm",
        r"\bdeep_gemm::.*(?:GemmType\)[1235]\b|GemmType::\w*Grouped|grouped)",
        guess=True,
    ),
    KernelRule(
        "moe_gemm",
        "moe_gemm",
        r"(?i)fused_moe|moe_gemm|moefcgemm|grouped_?gemm|groupproblemshape"
        r"|cutlass_moe|moe_mm|marlin_moe|moe_wna16|m_grouped|masked_gemm|mega_moe",
        guess=True,
    ),
    # trtllm-gen batched GEMM cubins (FlashInfer's TRT-LLM MoE path), named
    # like "bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s6_..._sm100a".
    KernelRule(
        "trtllm_bmm",
        "moe_gemm",
        r"^bmm_(?:Mx)?(?:E[245]m\d|Bfloat16|Fp16|Fp8)_",
        guess=True,
    ),
    # ---- dense GEMM
    KernelRule("deep_gemm", "dense_gemm", r"\bdeep_gemm::", guess=True),
    KernelRule(
        "cublas",
        "dense_gemm",
        r"nvjet|xmma_\w*gemm|cublas|gemv|splitKreduce|gemmSN|gemmk1"
        r"|(?:ampere|hopper|volta|turing|sm\d+)_\w*gemm",
    ),
    # CuTe DSL kernels are named "kernel_cutlass_<function>..."; FlashInfer's
    # fused add + RMSNorm is one, so the name says norm before cutlass counts
    KernelRule("cute_dsl_norm", "norm_act", r"(?i)^kernel_cutlass_\w*norm", guess=True),
    KernelRule(
        "gemm", "dense_gemm", r"(?i)gemm|matmul|scaled_mm|\bbmm|cutlass", guess=True
    ),
    # ---- quantization (after GEMMs whose names mention fp8 / scaling)
    KernelRule(
        "quantization",
        "quantization",
        r"(?i)quant|cvt_\w*fp4|nvfp4|mxfp|transform_sf|sf_layout|ue8m0",
        guess=True,
    ),
    # ---- norms, activations, rotary, hyper-connection mixing (mHC)
    KernelRule(
        "norm", "norm_act", r"(?i)rms_?norm|layer_?norm|group_?norm|norm_?kernel"
    ),
    KernelRule(
        "hyper_connection", "norm_act", r"(?i)(?:^|_)mhc_|hyper_?conn", guess=True
    ),
    KernelRule(
        "activation",
        "norm_act",
        r"(?i)_and_mul|silu|gelu|swiglu|relu|activation|softcap|tanh",
    ),
    KernelRule("rotary", "norm_act", r"(?i)rotary|(?<![a-z])rope|pos_?ids_?cos_?sin"),
    # ---- sampling and speculative verification
    KernelRule(
        "sampling",
        "sampling",
        r"(?i)sampling|top_?k_?top_?p|top_?[kp]_?renorm|topkmask|min_?p_"
        r"|softmax|argmax|penalt|token_bitmask|grammar|verify_?tree"
        r"|tree_speculative|build_tree|speculative|multinomial|gumbel|logits",
        guess=True,
    ),
    # ---- memory
    KernelRule("memop", "memory", r"^\[CUDA mem(?:cpy|set)"),
    KernelRule(
        "copy",
        "memory",
        r"direct_copy_kernel|CatArrayBatchedCopy|FillFunctor|\bcopy_kernel"
        r"|memcpy|memset|fill_kernel",
    ),
    # ---- generic PyTorch / Triton-inductor / CUB
    KernelRule(
        "elementwise",
        "elementwise",
        r"at::native::|\btriton_(?:poi|red|per|tem)_|elementwise_kernel"
        r"|reduce_kernel|\bcub::|\bthrust::|DeviceRadixSort|DeviceScan"
        r"|DeviceSegmented|radixSort|gatherTopK",
    ),
)

_COMPILED: tuple[tuple[KernelRule, re.Pattern[str]], ...] = tuple(
    (rule, re.compile(rule.pattern)) for rule in RULES
)
_VOID = re.compile(r"^\s*void\s+")


def match_rule(name: str) -> KernelRule | None:
    """The first rule matching a kernel name (None: unclassified)."""
    core = _VOID.sub("", name.strip())
    for rule, pattern in _COMPILED:
        if pattern.search(core):
            return rule
    return None


def classify_kernel(name: str) -> KernelCategory:
    """Category of a CUDA kernel or memory-operation name."""
    rule = match_rule(name)
    return rule.category if rule else "other"


_INNER_TEMPLATE = re.compile(r"<[^<>]*>")
_COLLAPSED = "\x01"


def _strip_params(text: str) -> str:
    """Drop the trailing top-level parameter list ``(...)``."""
    if not text.endswith(")"):
        return text
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return text[:index].rstrip() if index > 0 else text
    return text


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# CuTe DSL kernels spell their argument types into the name:
# kernel_cutlass_<function>_tensorptri32gmemo2624..._tensorptrbf16gmemo...
_CUTE_DSL_ARGS = re.compile(r"^(kernel_cutlass_\w+?)_[Tt]ensor")


def short_kernel_name(name: str, max_len: int = SHORT_MAX) -> str:
    """Display form: no ``void``, template arguments collapsed, no parameters
    (for a CuTe DSL kernel, no argument types)."""
    cute = _CUTE_DSL_ARGS.match(name.strip())
    if cute:
        return _truncate(cute[1] + "…", max_len)
    text = _VOID.sub("", name.strip())
    previous = None
    while previous != text:
        previous = text
        text = _INNER_TEMPLATE.sub(_COLLAPSED, text)
    text = _strip_params(text).replace(_COLLAPSED, "<…>")
    return _truncate(text, max_len) if text else _truncate(name.strip(), max_len)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class KernelRow(Frozen):
    """A ``cuda_gpu_kern_sum`` (kernel) or ``cuda_gpu_mem_time_sum`` (memop) row."""

    name: str
    kind: RowKind = "kernel"
    total_us: float
    instances: int = 0
    avg_us: float | None = None
    med_us: float | None = None
    min_us: float | None = None
    max_us: float | None = None
    stddev_us: float | None = None


class KernelExecRow(Frozen):
    """One row of ``cuda_kern_exec_sum``: a kernel on one device from one thread."""

    name: str
    pid: int | None = None
    tid: int | None = None
    device: int | None = None
    count: int = 0
    kernel_avg_us: float | None = None
    api: str | None = None

    @property
    def total_us(self) -> float:
        return self.count * (self.kernel_avg_us or 0.0)


class CategoryTime(Frozen):
    category: KernelCategory
    time_us: float
    share: float  # of the breakdown's total GPU time
    instances: int = 0
    kernels: int = 0  # distinct names (max over inputs after a merge)


class TopKernel(Frozen):
    name: str  # demangled name, truncated to NAME_MAX
    short: str  # display form (short_kernel_name)
    category: KernelCategory
    time_us: float
    share: float
    instances: int = 0
    avg_us: float | None = None


class DeviceCoverage(Frozen):
    """Which GPUs of one report have kernels inside the capture range."""

    source: str
    devices: tuple[int, ...] = ()  # CUDA device ids as seen inside the container
    device_time_us: tuple[float, ...] = ()  # aligned with ``devices``
    processes: int = 0  # distinct PIDs that launched kernels
    expected_devices: int | None = None  # GPUs given to the container
    # Kernel time attributed by cuda_kern_exec_sum / cuda_gpu_kern_sum; well
    # below 1 means the launch-to-kernel join missed kernels.
    exec_share: float | None = None

    @property
    def complete(self) -> bool | None:
        if self.expected_devices is None:
            return None
        return len(self.devices) >= self.expected_devices

    @property
    def imbalance(self) -> float | None:
        """Busiest GPU over the mean GPU (1.0 = balanced)."""
        if len(self.device_time_us) < 2:
            return None
        mean = sum(self.device_time_us) / len(self.device_time_us)
        return round(max(self.device_time_us) / mean, 4) if mean > 0 else None


class KernelBreakdown(Frozen):
    """GPU-time split of one or more reports by kernel category."""

    sources: tuple[str, ...] = ()  # report labels (container names)
    total_us: float = 0.0  # kernel_us + memop_us
    kernel_us: float = 0.0
    memop_us: float = 0.0
    instances: int = 0  # kernel launches (graph nodes included) + memory ops
    distinct_kernels: int = 0  # max over inputs after a merge
    categories: tuple[CategoryTime, ...] = ()  # by time, descending
    top: tuple[TopKernel, ...] = ()
    coverage: tuple[DeviceCoverage, ...] = ()  # one per source with exec data
    notes: tuple[str, ...] = ()

    def share(self, category: KernelCategory) -> float:
        for item in self.categories:
            if item.category == category:
                return item.share
        return 0.0

    @property
    def unclassified_share(self) -> float:
        return self.share("other")

    @property
    def comm_share(self) -> float:
        return round(sum(self.share(c) for c in COMMUNICATION), 6)

    @property
    def partial_coverage(self) -> bool:
        return any(c.complete is False for c in self.coverage)


class NsysInfo(Frozen):
    """Whether an image ships the Nsight Systems CLI, and which version."""

    image: str
    available: bool
    path: str | None = None
    version: str | None = None  # e.g. "2025.6.1.190-256162285624v0"
    version_tuple: tuple[int, int, int] | None = None
    notes: tuple[str, ...] = ()


class DiagImage(Frozen):
    """The image a node deploys for the diagnostic window."""

    node: str
    base_image: str
    image: str | None = None  # base_image itself when it already has nsys
    built: bool = False
    nsys: NsysInfo | None = None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.image is not None and self.nsys is not None and self.nsys.available


class ProfileTrigger(Frozen):
    """Outcome of POST /engine/control/start_profile on one worker."""

    node: str
    port: int
    ok: bool
    http_status: int | None = None
    message: str | None = None


class ReportFile(Frozen):
    node: str
    path: str  # host path of the .nsys-rep
    size_bytes: int | None = None
    ready: bool = False  # present, non-empty and no longer growing


class NsysLogState(Frozen):
    """What a worker's log says about its nsys session."""

    capture_started: bool = False
    capture_ended: bool = False
    generated: tuple[str, ...] = ()  # report paths nsys says it wrote
    errors: tuple[str, ...] = ()


class KernelReport(Frozen):
    """One report's ``nsys stats`` result."""

    node: str
    source: str  # container name
    report: str  # host path of the .nsys-rep
    breakdown: KernelBreakdown | None = None
    timeline: dict[str, Any] | None = None  # domain/steps.StepBreakdown as JSON
    files: tuple[str, ...] = ()  # host paths of the CSVs that were read
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# nsys version
# --------------------------------------------------------------------------

_VERSION = re.compile(r"Nsight Systems version\s+(\S+)", re.IGNORECASE)
_VERSION_NUM = re.compile(r"(\d{4})\.(\d+)\.(\d+)")
HES_DEFAULT_FROM = (2025, 6, 1)  # hardware CUDA trace default on Blackwell


def parse_nsys_version(text: str) -> tuple[str | None, tuple[int, int, int] | None]:
    """(version string, (year, minor, patch)) from ``nsys --version`` output."""
    match = _VERSION.search(text)
    version = match.group(1) if match else None
    numbers = _VERSION_NUM.search(version or text)
    if numbers is None:
        return version, None
    triple = (int(numbers.group(1)), int(numbers.group(2)), int(numbers.group(3)))
    return version or numbers.group(0), triple


def expected_cuda_trace(
    version: tuple[int, int, int] | None, compute_capability: str | None
) -> CudaTrace:
    """CUDA trace method nsys should pick by default (the report's Diagnostics
    Summary records the method actually used). Blackwell is CC >= 10; whether
    RTX PRO 6000 (CC 12.0) counts as a supported GPU is unverified."""
    if version is None or not compute_capability:
        return "unknown"
    try:
        major = int(compute_capability.split(".")[0])
    except ValueError:
        return "unknown"
    if major >= 10 and version >= HES_DEFAULT_FROM:
        return "hardware"
    return "software"


# --------------------------------------------------------------------------
# nsys stats CSV parsing
# --------------------------------------------------------------------------

_UNIT = re.compile(
    r"^(?P<base>.*?)\s*\((?P<unit>ns|nsec|us|usec|µs|ms|msec|s|sec|%)\)\s*$",
    re.IGNORECASE,
)
_US_PER_UNIT = {
    "ns": 1e-3,
    "nsec": 1e-3,
    "us": 1.0,
    "usec": 1.0,
    "µs": 1.0,
    "ms": 1e3,
    "msec": 1e3,
    "s": 1e6,
    "sec": 1e6,
}
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_FIELDS: dict[str, tuple[str, ...]] = {
    "total": ("totaltime", "total"),
    "instances": ("instances", "count", "numcalls", "calls"),
    "avg": ("avg", "average", "mean"),
    "med": ("med", "median"),
    "min": ("min", "minimum"),
    "max": ("max", "maximum"),
    "stddev": ("stddev", "stdev", "std"),
    "kernel_name": ("name", "kernelname"),
    "operation": ("operation",),
    "pid": ("pid",),
    "tid": ("tid",),
    "device": ("devid", "device", "deviceid", "gpu"),
    "kavg": ("kavg",),
    "api": ("apiname", "api"),
}


def _column_key(header: str) -> tuple[str, str | None]:
    text = header.strip()
    unit = None
    match = _UNIT.match(text)
    if match:
        text, unit = match.group("base"), match.group("unit").lower()
    return re.sub(r"[^a-z0-9]", "", text.lower()), unit


@dataclass(frozen=True)
class _Table:
    columns: dict[str, tuple[int, str | None]]  # field -> (index, unit)
    rows: tuple[tuple[str, ...], ...]


def _columns(cells: Sequence[str]) -> dict[str, tuple[int, str | None]]:
    found: dict[str, tuple[int, str | None]] = {}
    for index, cell in enumerate(cells):
        key, unit = _column_key(cell)
        for field, aliases in _FIELDS.items():
            if key in aliases and field not in found:
                found[field] = (index, unit)
    return found


def _split(line: str) -> list[str] | None:
    try:
        return next(csv.reader([line]))
    except (csv.Error, StopIteration):
        return None


def _tables(text: str) -> list[_Table]:
    """Every CSV table in ``text``: a header row with at least three known
    columns, then rows until a blank line (console output separates reports
    with blank lines and decorates them with progress messages)."""
    tables: list[_Table] = []
    columns: dict[str, tuple[int, str | None]] | None = None
    rows: list[tuple[str, ...]] = []
    for line in text.splitlines():
        if not line.strip():
            if columns is not None:
                tables.append(_Table(columns, tuple(rows)))
                columns, rows = None, []
            continue
        cells = _split(line)
        if not cells:
            continue
        if columns is None:
            found = _columns(cells)
            if len(cells) >= 3 and len(found) >= 3:
                columns = found
            continue
        rows.append(tuple(cells))
    if columns is not None:
        tables.append(_Table(columns, tuple(rows)))
    return tables


def _number(cell: str) -> float | None:
    text = cell.strip().replace(" ", "").rstrip("%")
    if not text or text.lower() in ("-", "n/a", "na", "nan", "none"):
        return None
    if _THOUSANDS.fullmatch(text):
        text = text.replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _cell(row: Sequence[str], column: tuple[int, str | None] | None) -> str | None:
    if column is None or column[0] >= len(row):
        return None
    return row[column[0]]


def _time_us(row: Sequence[str], column: tuple[int, str | None] | None) -> float | None:
    cell = _cell(row, column)
    value = _number(cell) if cell is not None else None
    if value is None or column is None:
        return None
    return round(value * _US_PER_UNIT.get(column[1] or "ns", 1e-3), 6)


def _count(row: Sequence[str], column: tuple[int, str | None] | None) -> int | None:
    cell = _cell(row, column)
    value = _number(cell) if cell is not None else None
    return int(round(value)) if value is not None else None


def _summary_rows(text: str, kind: RowKind) -> list[KernelRow]:
    name_field = "kernel_name" if kind == "kernel" else "operation"
    for table in _tables(text):
        cols = table.columns
        if "total" not in cols or name_field not in cols:
            continue
        if kind == "kernel" and "operation" in cols:
            continue  # a memory-operation table
        rows: list[KernelRow] = []
        for row in table.rows:
            name = (_cell(row, cols[name_field]) or "").strip()
            total = _time_us(row, cols["total"])
            if not name or total is None:
                continue
            rows.append(
                KernelRow(
                    name=name,
                    kind=kind,
                    total_us=total,
                    instances=_count(row, cols.get("instances")) or 0,
                    avg_us=_time_us(row, cols.get("avg")),
                    med_us=_time_us(row, cols.get("med")),
                    min_us=_time_us(row, cols.get("min")),
                    max_us=_time_us(row, cols.get("max")),
                    stddev_us=_time_us(row, cols.get("stddev")),
                )
            )
        return rows
    return []


def parse_kern_sum_csv(text: str) -> list[KernelRow]:
    """Rows of a ``cuda_gpu_kern_sum`` CSV (file or console output).

    Accepts current headers ("Time (%)", "Total Time (ns)", "Instances",
    "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)", "Name"),
    older ones ("Time(%)", "Average", "Minimum", "Maximum") and other time
    units ("(us)", "(ms)", ...). Rows without a name or total are dropped.
    """
    return _summary_rows(text, "kernel")


def parse_mem_time_sum_csv(text: str) -> list[KernelRow]:
    """Rows of a ``cuda_gpu_mem_time_sum`` CSV ("Count" / "Operation" columns)."""
    return _summary_rows(text, "memop")


def _device_id(cell: str | None) -> int | None:
    if cell is None:
        return None
    text = cell.strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"\((\d+)\)\s*$", text) or re.match(r"(\d+)\b", text)
    return int(match.group(1)) if match else None


def _int(cell: str | None) -> int | None:
    value = _number(cell) if cell is not None else None
    return int(value) if value is not None else None


def parse_kern_exec_sum_csv(text: str) -> list[KernelExecRow]:
    """Rows of a ``cuda_kern_exec_sum`` CSV ("PID", "TID", "DevId", "Count",
    ..., "KAvg (ns)", ..., "API Name", "Kernel Name")."""
    for table in _tables(text):
        cols = table.columns
        if "kernel_name" not in cols or "instances" not in cols:
            continue
        if "kavg" not in cols and "total" not in cols:
            continue
        rows: list[KernelExecRow] = []
        for row in table.rows:
            name = (_cell(row, cols["kernel_name"]) or "").strip()
            count = _count(row, cols["instances"])
            if not name or count is None:
                continue
            avg = _time_us(row, cols.get("kavg"))
            if avg is None and "total" in cols and count > 0:
                total = _time_us(row, cols["total"])
                avg = round(total / count, 6) if total is not None else None
            api = _cell(row, cols.get("api"))
            rows.append(
                KernelExecRow(
                    name=name,
                    pid=_int(_cell(row, cols.get("pid"))),
                    tid=_int(_cell(row, cols.get("tid"))),
                    device=_device_id(_cell(row, cols.get("device"))),
                    count=count,
                    kernel_avg_us=avg,
                    api=api.strip() if api else None,
                )
            )
        return rows
    return []


_STATS_PROBLEM = re.compile(
    r"SKIPPED|does not contain|ERROR|Error:|could not|No such file", re.IGNORECASE
)


def stats_problems(text: str, limit: int = 5) -> tuple[str, ...]:
    """Lines of ``nsys stats`` output reporting skipped or failed reports."""
    found = [line.strip() for line in text.splitlines() if _STATS_PROBLEM.search(line)]
    return tuple(_truncate(line, 300) for line in found[:limit])


# --------------------------------------------------------------------------
# Breakdown
# --------------------------------------------------------------------------


def merge_rows(groups: Iterable[Sequence[KernelRow]]) -> list[KernelRow]:
    """Sum rows of the same (kind, name) across reports or ranks.

    Totals, instances, min and max combine exactly; the average is recomputed;
    median and stddev survive only for names seen once.
    """
    merged: dict[tuple[str, str], KernelRow] = {}
    seen: dict[tuple[str, str], int] = {}
    for rows in groups:
        for row in rows:
            key = (row.kind, row.name)
            seen[key] = seen.get(key, 0) + 1
            prev = merged.get(key)
            if prev is None:
                merged[key] = row
                continue
            instances = prev.instances + row.instances
            total = round(prev.total_us + row.total_us, 6)
            mins = [v for v in (prev.min_us, row.min_us) if v is not None]
            maxs = [v for v in (prev.max_us, row.max_us) if v is not None]
            merged[key] = KernelRow(
                name=row.name,
                kind=row.kind,
                total_us=total,
                instances=instances,
                avg_us=round(total / instances, 6) if instances else None,
                min_us=min(mins) if mins else None,
                max_us=max(maxs) if maxs else None,
            )
    return list(merged.values())


def device_coverage(
    exec_rows: Sequence[KernelExecRow],
    *,
    source: str = "",
    expected_devices: int | None = None,
    kernel_us: float | None = None,
) -> DeviceCoverage:
    """Per-GPU kernel time of one report from its ``cuda_kern_exec_sum`` rows."""
    per_device: dict[int, float] = {}
    pids: set[int] = set()
    for row in exec_rows:
        if row.device is not None:
            per_device[row.device] = per_device.get(row.device, 0.0) + row.total_us
        if row.pid is not None:
            pids.add(row.pid)
    devices = sorted(per_device)
    attributed = sum(per_device.values())
    return DeviceCoverage(
        source=source,
        devices=tuple(devices),
        device_time_us=tuple(round(per_device[d], 3) for d in devices),
        processes=len(pids),
        expected_devices=expected_devices,
        exec_share=round(attributed / kernel_us, 4) if kernel_us else None,
    )


def _share(value: float, total: float) -> float:
    return round(value / total, 6) if total > 0 else 0.0


def _categories(
    times: dict[KernelCategory, float],
    instances: dict[KernelCategory, int],
    kernels: dict[KernelCategory, int],
    total: float,
) -> tuple[CategoryTime, ...]:
    order = {c: i for i, c in enumerate(CATEGORIES)}
    present = [c for c in CATEGORIES if times.get(c, 0.0) > 0 or instances.get(c, 0)]
    present.sort(key=lambda c: (-times.get(c, 0.0), order[c]))
    return tuple(
        CategoryTime(
            category=c,
            time_us=round(times.get(c, 0.0), 3),
            share=_share(times.get(c, 0.0), total),
            instances=instances.get(c, 0),
            kernels=kernels.get(c, 0),
        )
        for c in present
    )


def _coverage_notes(coverage: Sequence[DeviceCoverage]) -> list[str]:
    notes = []
    for item in coverage:
        if item.complete is False:
            notes.append(
                f"{item.source or 'report'}: kernels on {len(item.devices)} of "
                f"{item.expected_devices} GPUs inside the capture range"
            )
        if item.exec_share is not None and item.exec_share < 0.9:
            notes.append(
                f"{item.source or 'report'}: cuda_kern_exec_sum attributes only "
                f"{item.exec_share:.0%} of the kernel time to a GPU"
            )
    return notes


def breakdown(
    rows: Sequence[KernelRow],
    *,
    source: str = "",
    top_n: int = TOP_N,
    exec_rows: Sequence[KernelExecRow] | None = None,
    expected_devices: int | None = None,
    notes: Sequence[str] = (),
) -> KernelBreakdown:
    """Category split, top kernels and coverage of one report's rows.

    Memory-operation rows count as ``memory`` whatever their name; duplicate
    names are summed first. ``exec_rows`` (``cuda_kern_exec_sum``) add the
    per-GPU coverage check against ``expected_devices``.
    """
    merged = merge_rows([rows])
    times: dict[KernelCategory, float] = {}
    instances: dict[KernelCategory, int] = {}
    kernels: dict[KernelCategory, int] = {}
    classified: list[tuple[KernelRow, KernelCategory]] = []
    for row in merged:
        category = "memory" if row.kind == "memop" else classify_kernel(row.name)
        classified.append((row, category))
        times[category] = times.get(category, 0.0) + row.total_us
        instances[category] = instances.get(category, 0) + row.instances
        kernels[category] = kernels.get(category, 0) + 1
    kernel_us = sum(r.total_us for r in merged if r.kind == "kernel")
    memop_us = sum(r.total_us for r in merged if r.kind == "memop")
    total = kernel_us + memop_us
    ranked = sorted(classified, key=lambda item: (-item[0].total_us, item[0].name))
    top = tuple(
        TopKernel(
            name=_truncate(row.name, NAME_MAX),
            short=short_kernel_name(row.name),
            category=category,
            time_us=round(row.total_us, 3),
            share=_share(row.total_us, total),
            instances=row.instances,
            avg_us=row.avg_us,
        )
        for row, category in ranked[: max(0, top_n)]
    )
    coverage: tuple[DeviceCoverage, ...] = ()
    if exec_rows is not None:
        coverage = (
            device_coverage(
                exec_rows,
                source=source,
                expected_devices=expected_devices,
                kernel_us=kernel_us or None,
            ),
        )
    all_notes = list(notes)
    if not merged:
        all_notes.append(f"{source or 'report'}: no kernel or memory-operation rows")
    all_notes.extend(_coverage_notes(coverage))
    return KernelBreakdown(
        sources=(source,) if source else (),
        total_us=round(total, 3),
        kernel_us=round(kernel_us, 3),
        memop_us=round(memop_us, 3),
        instances=sum(r.instances for r in merged),
        distinct_kernels=len(merged),
        categories=_categories(times, instances, kernels, total),
        top=top,
        coverage=coverage,
        notes=tuple(dict.fromkeys(all_notes)),
    )


def merge_breakdowns(
    breakdowns: Sequence[KernelBreakdown], *, top_n: int = TOP_N
) -> KernelBreakdown:
    """One breakdown over several reports (ranks, containers, instances).

    Category times are exact; top kernels merge the inputs' top lists, so a
    kernel outside every input's top ``N`` is missing (``breakdown`` over
    ``merge_rows`` of the raw rows is exact). Coverage entries are kept per
    source.
    """
    if not breakdowns:
        return KernelBreakdown(notes=("no reports to merge",))
    total = sum(b.total_us for b in breakdowns)
    times: dict[KernelCategory, float] = {}
    instances: dict[KernelCategory, int] = {}
    kernels: dict[KernelCategory, int] = {}
    for item in breakdowns:
        for cat in item.categories:
            times[cat.category] = times.get(cat.category, 0.0) + cat.time_us
            instances[cat.category] = instances.get(cat.category, 0) + cat.instances
            kernels[cat.category] = max(kernels.get(cat.category, 0), cat.kernels)
    tops: dict[str, TopKernel] = {}
    for item in breakdowns:
        for kernel in item.top:
            prev = tops.get(kernel.name)
            if prev is None:
                tops[kernel.name] = kernel
                continue
            count = prev.instances + kernel.instances
            time_us = round(prev.time_us + kernel.time_us, 3)
            tops[kernel.name] = prev.model_copy(
                update={
                    "time_us": time_us,
                    "instances": count,
                    "avg_us": round(time_us / count, 6) if count else None,
                }
            )
    ranked = sorted(tops.values(), key=lambda k: (-k.time_us, k.name))[: max(0, top_n)]
    top = tuple(
        k.model_copy(update={"share": _share(k.time_us, total)}) for k in ranked
    )
    sources = tuple(dict.fromkeys(s for b in breakdowns for s in b.sources))
    notes = [n for b in breakdowns for n in b.notes]
    if len(breakdowns) > 1:
        notes.append(f"merged {len(breakdowns)} reports; top kernels approximate")
    return KernelBreakdown(
        sources=sources,
        total_us=round(total, 3),
        kernel_us=round(sum(b.kernel_us for b in breakdowns), 3),
        memop_us=round(sum(b.memop_us for b in breakdowns), 3),
        instances=sum(b.instances for b in breakdowns),
        distinct_kernels=max(b.distinct_kernels for b in breakdowns),
        categories=_categories(times, instances, kernels, total),
        top=top,
        coverage=tuple(c for b in breakdowns for c in b.coverage),
        notes=tuple(dict.fromkeys(notes)),
    )


# --------------------------------------------------------------------------
# Synthetic data for the dry-run simulator
# --------------------------------------------------------------------------

# (demangled name, share of the role's kernel time, average duration in us)
_SYNTH_DECODE: tuple[tuple[str, float, float], ...] = (
    (
        "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, "
        "(cute::UMMA::Major)0, 0u, 4096u, 6144u, 32u, 128u, 160u, 128u, 128u, 1u, "
        "7u, 128u, 128u, 1u, false, 148u, (deep_gemm::GemmType)2, false, "
        "cutlass::bfloat16_t>(int*, unsigned int, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st)",
        0.205,
        98.0,
    ),
    (
        "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, "
        "(cute::UMMA::Major)0, 0u, 6144u, 2048u, 32u, 128u, 96u, 128u, 128u, 1u, "
        "8u, 128u, 128u, 1u, false, 148u, (deep_gemm::GemmType)2, false, "
        "cutlass::bfloat16_t>(int*, unsigned int, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st)",
        0.11,
        52.0,
    ),
    (
        "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvSparseP64"
        "MultiCtasKvCgaVarSeqQ8Kv128PersistentSwapsAbForGen",
        0.145,
        70.0,
    ),
    (
        "void deep_ep::internode_ll::combine<3, false, 6144, 9>(void*, void*, int*, "
        "void*, void const*, long const*, float const*, int const*, long const*, "
        "int*, int*, void*, int, int, int, int, int, int, int, bool, bool, bool)",
        0.109,
        52.0,
    ),
    (
        "void deep_ep::internode_ll::dispatch<true, false, 6144>(void*, float*, "
        "int*, long*, void*, int*, void*, int*, long*, void const*, long const*, "
        "int*, int*, int, int, int, int, int, int, int, int, int, bool, bool)",
        0.103,
        48.0,
    ),
    (
        "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, "
        "(cute::UMMA::Major)0, 0u, 2112u, 6144u, 1u, 128u, 112u, 128u, 128u, 1u, "
        "7u, 128u, 128u, 1u, false, 148u, (deep_gemm::GemmType)0, false, "
        "cutlass::bfloat16_t>(int*, unsigned int, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st)",
        0.073,
        14.0,
    ),
    ("nvjet_tst_128x64_64x8_1x2_h_bz_TNT", 0.055, 9.0),
    (
        "void deep_gemm::sm100_fp8_paged_mqa_logits<64u, 128u, 4u, 256u, 148u, 4u, "
        "128u, 128u>(unsigned int, unsigned long, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*, unsigned int*, "
        "int*)",
        0.036,
        18.0,
    ),
    ("nvjet_hsh_64x32_64x16_4x1_v_bz_TNN", 0.024, 6.0),
    ("_silu_and_mul_post_quant_kernel", 0.019, 9.0),
    (
        "void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>"
        "(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, unsigned int, unsigned "
        "int, unsigned int, float, float)",
        0.0175,
        4.2,
    ),
    (
        "void per_token_group_quant_8bit_kernel<__nv_bfloat16, c10::Float8_e4m3fn, "
        "true, false>(__nv_bfloat16 const*, void*, float*, int, int, int, int, int, "
        "float, float, float, int, int)",
        0.0175,
        2.8,
    ),
    (
        "sm100_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x64x64_cgasize1x1x1_"
        "warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
        0.016,
        600.0,
    ),
    (
        "void (anonymous namespace)::fast_topk_transform_fused_kernel<2048>"
        "(float const*, int*, int const*, int const*, int, int, int)",
        0.018,
        9.0,
    ),
    (
        "void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, "
        "__nv_bfloat16*, __nv_bfloat16*, unsigned int, unsigned int, unsigned int, "
        "unsigned int, float, float)",
        0.0105,
        2.5,
    ),
    (
        "void dsv3_router_gemm_kernel<__nv_bfloat16, float, 128, 8, 256, 6144>"
        "(float*, __nv_bfloat16 const*, __nv_bfloat16 const*)",
        0.0085,
        4.0,
    ),
    (
        "void moe_fused_gate_kernel<float, 256, 8, 4, 32>(void*, void*, float*, "
        "int*, long, long, long, double, long, bool)",
        0.0073,
        3.5,
    ),
    (
        "void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel"
        "<false, 64u, 8u, 16u, __nv_bfloat16, long>(__nv_bfloat16*, __nv_bfloat16*, "
        "__nv_bfloat16*, __nv_bfloat16*, float*, long*, unsigned int, unsigned int, "
        "unsigned int, unsigned int, unsigned long, unsigned long, unsigned long, "
        "unsigned long, unsigned long, unsigned long, unsigned long, unsigned long)",
        0.0055,
        2.6,
    ),
    (
        "void at::native::vectorized_elementwise_kernel<4, "
        "at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, "
        "at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)",
        0.004,
        2.0,
    ),
    ("set_mla_kv_buffer_kernel", 0.0036, 1.8),
    (
        "void at::native::unrolled_elementwise_kernel<at::native::direct_copy_"
        "kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::"
        "{lambda()#7}::operator()() const::{lambda(c10::BFloat16)#1}, "
        "std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, "
        "TrivialOffsetCalculator<1, unsigned int>, at::native::memory::"
        "LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, "
        "at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::"
        "{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::"
        "{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>, "
        "TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, "
        "unsigned int>, at::native::memory::LoadWithCast<1>, "
        "at::native::memory::StoreWithCast<1>)",
        0.003,
        1.5,
    ),
    ("ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)", 0.0024, 90.0),
    (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        0.0018,
        30.0,
    ),
    (
        "void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, "
        "(cub::CUB_300001_SM_1000::BlockScanAlgorithm)2, "
        "(cub::CUB_300001_SM_1000::BlockReduceAlgorithm)2, 4u, true, float, int>"
        "(float*, int*, int*, float*, int*, float*, unsigned int, unsigned int, "
        "unsigned long, unsigned long)",
        0.0007,
        25.0,
    ),
    (
        "void sglang::cross_device_reduce_1stage<__nv_bfloat16, 8>"
        "(sglang::RankData*, sglang::RankSignals, sglang::Signal*, __nv_bfloat16*, "
        "int, int)",
        0.0007,
        12.0,
    ),
    ("compute_position_kernel", 0.0007, 1.0),
)
_SYNTH_PREFILL: tuple[tuple[str, float, float], ...] = (
    (
        "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, "
        "(cute::UMMA::Major)0, 0u, 4096u, 6144u, 32u, 128u, 192u, 128u, 128u, 1u, "
        "6u, 128u, 128u, 1u, false, 148u, (deep_gemm::GemmType)1, false, "
        "cutlass::bfloat16_t>(int*, unsigned int, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st)",
        0.25,
        2100.0,
    ),
    (
        "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, "
        "(cute::UMMA::Major)0, 0u, 6144u, 2048u, 32u, 128u, 128u, 128u, 128u, 1u, "
        "7u, 128u, 128u, 1u, false, 148u, (deep_gemm::GemmType)1, false, "
        "cutlass::bfloat16_t>(int*, unsigned int, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st)",
        0.125,
        1100.0,
    ),
    (
        "fmhaSm100fKernel_QkvBfloat16OBfloat16H192HV128PagedKvCausalP64VarSeqQ128"
        "Kv128PersistentContext",
        0.155,
        950.0,
    ),
    (
        "void deep_ep::internode::dispatch<false, 8, 9, 6144, 4, 24, 128>"
        "(int4*, float*, long*, float*, void*, int*, int*, int*, int*, int*, "
        "int4 const*, float const*, long const*, float const*, int const*, "
        "bool const*, int, int, int, int, int, int, void*, int, int, void**, int, "
        "int, int, int)",
        0.065,
        1500.0,
    ),
    (
        "void deep_ep::internode::combine<__nv_bfloat16, 8, 9, 6144, 4, 24, 128>"
        "(int4*, float*, bool const*, void const*, float const*, int const*, "
        "int const*, void const*, int*, int*, int*, int*, int, int, int, int, int, "
        "void*, int, int, void**, int, int, int)",
        0.065,
        1500.0,
    ),
    (
        "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, "
        "(cute::UMMA::Major)0, 0u, 2112u, 6144u, 1u, 128u, 128u, 128u, 128u, 1u, "
        "6u, 128u, 128u, 1u, false, 148u, (deep_gemm::GemmType)0, false, "
        "cutlass::bfloat16_t>(int*, unsigned int, unsigned int, unsigned int, "
        "CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st)",
        0.085,
        420.0,
    ),
    (
        "void deep_gemm::sm100_fp8_mqa_logits<64u, 128u, 4u, 148u, 128u, 128u>"
        "(unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, "
        "CUtensorMap_st, float*, unsigned int*, unsigned int*)",
        0.045,
        600.0,
    ),
    ("nvjet_tst_256x128_64x4_2x1_v_bz_coopB_TNT", 0.045, 380.0),
    ("_fwd_kernel_ep_scatter_2", 0.012, 160.0),
    ("_fwd_kernel_ep_gather", 0.01, 140.0),
    (
        "void per_token_group_quant_8bit_kernel<__nv_bfloat16, c10::Float8_e4m3fn, "
        "true, false>(__nv_bfloat16 const*, void*, float*, int, int, int, int, int, "
        "float, float, float, int, int)",
        0.022,
        90.0,
    ),
    (
        "void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, "
        "&silu<float> >(__nv_bfloat16*, __nv_bfloat16 const*, int)",
        0.022,
        110.0,
    ),
    (
        "void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>"
        "(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, unsigned int, unsigned "
        "int, unsigned int, float, float)",
        0.016,
        60.0,
    ),
    ("set_mla_kv_buffer_kernel", 0.006, 40.0),
    (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        0.011,
        300.0,
    ),
    (
        "void deep_ep::layout::get_dispatch_layout<256, 32, 8>(long const*, int*, "
        "int*, int*, bool*, int, int, int, int)",
        0.004,
        40.0,
    ),
    (
        "void at::native::vectorized_elementwise_kernel<4, "
        "at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, "
        "at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)",
        0.011,
        45.0,
    ),
    (
        "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::"
        "(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::"
        "(anonymous namespace)::OpaqueType<2u>*, at::native::(anonymous namespace)::"
        "CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<2u>, "
        "unsigned int, 128, 1>, at::native::(anonymous namespace)::"
        "TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
        0.01,
        30.0,
    ),
    ("compute_position_kernel", 0.002, 4.0),
)
# (operation, share of the kernel time, average duration in us)
_SYNTH_MEMOPS: tuple[tuple[str, float, float], ...] = (
    ("[CUDA memcpy Device-to-Device]", 0.003, 2.5),
    ("[CUDA memset]", 0.0008, 1.2),
    ("[CUDA memcpy Host-to-Device]", 0.0004, 3.8),
    ("[CUDA memcpy Device-to-Host]", 0.0001, 1.9),
)
# Summed kernel time per GPU and forward step (illustrative).
_SYNTH_STEP_US = {"decode": 45_000.0, "prefill": 400_000.0}


def _synthetic_rows(
    role: str, devices: int, steps: int
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """(kernels, memops) as (name, total_us, avg_us) for a role's window."""
    if role in ("decode", "prefill"):
        blend = [(role, 1.0)]
    else:  # aggregated workers both prefill and decode
        blend = [("decode", 0.7), ("prefill", 0.3)]
    kernels: dict[str, tuple[float, float]] = {}
    total = 0.0
    for part, weight in blend:
        table = _SYNTH_DECODE if part == "decode" else _SYNTH_PREFILL
        norm = sum(share for _, share, _ in table)
        budget = _SYNTH_STEP_US[part] * weight * max(1, devices) * max(1, steps)
        total += budget
        for name, share, avg in table:
            time_us, _ = kernels.get(name, (0.0, avg))
            kernels[name] = (time_us + budget * share / norm, avg)
    kernel_rows = [(n, t, a) for n, (t, a) in kernels.items()]
    memops = [(n, total * share, avg) for n, share, avg in _SYNTH_MEMOPS]
    return kernel_rows, memops


def _summary_csv(
    rows: Sequence[tuple[str, float, float]], name_header: str, count_header: str
) -> str:
    lines: list[list[str]] = [
        [
            "Time (%)",
            "Total Time (ns)",
            count_header,
            "Avg (ns)",
            "Med (ns)",
            "Min (ns)",
            "Max (ns)",
            "StdDev (ns)",
            name_header,
        ]
    ]
    total = sum(t for _, t, _ in rows) or 1.0
    for name, time_us, avg_us in sorted(rows, key=lambda r: (-r[1], r[0])):
        count = max(1, round(time_us / avg_us))
        total_ns = round(time_us * 1000)
        avg_ns = total_ns / count
        lines.append(
            [
                f"{100 * time_us / total:.1f}",
                str(total_ns),
                str(count),
                f"{avg_ns:.1f}",
                f"{avg_ns * 0.99:.1f}",
                str(round(avg_ns * 0.8)),
                str(round(avg_ns * 1.6)),
                f"{avg_ns * 0.07:.1f}",
                name,
            ]
        )
    return _csv_text(lines)


def _csv_text(lines: Sequence[Sequence[str]]) -> str:
    out: list[str] = []
    for line in lines:
        cells = []
        for cell in line:
            if any(c in cell for c in ',"\n'):
                cell = '"' + cell.replace('"', '""') + '"'
            cells.append(cell)
        out.append(",".join(cells))
    return "\n".join(out) + "\n"


def synthetic_kern_sum_csv(
    role: str = "decode", *, devices: int = 8, steps: int = 10
) -> str:
    """Illustrative ``cuda_gpu_kern_sum`` CSV of a GLM-5.3-like worker window."""
    kernels, _ = _synthetic_rows(role, devices, steps)
    return _summary_csv(kernels, "Name", "Instances")


def synthetic_mem_time_sum_csv(
    role: str = "decode", *, devices: int = 8, steps: int = 10
) -> str:
    """Illustrative ``cuda_gpu_mem_time_sum`` CSV matching the kernel table."""
    _, memops = _synthetic_rows(role, devices, steps)
    return _summary_csv(memops, "Operation", "Count")


def synthetic_kern_exec_sum_csv(
    role: str = "decode", *, devices: int = 8, steps: int = 10
) -> str:
    """Illustrative ``cuda_kern_exec_sum`` CSV: the kernels split over GPUs."""
    kernels, _ = _synthetic_rows(role, devices, steps)
    stats = ("Avg", "Med", "Min", "Max", "StdDev")
    header = ["PID", "TID", "DevId", "Count", "QCount"]
    for prefix in ("T", "A", "Q", "K"):
        header += [f"{prefix}{s} (ns)" for s in stats]
    header += ["API Name", "Kernel Name"]
    lines: list[list[str]] = [header]
    devices = max(1, devices)
    for device in range(devices):
        pid = 4100 + device
        for name, time_us, avg_us in sorted(kernels, key=lambda r: (-r[1], r[0])):
            count = max(1, round(time_us / avg_us / devices))
            avg_ns = round(avg_us * 1000, 1)
            timing = [avg_ns, avg_ns * 0.99, avg_ns * 0.8, avg_ns * 1.6, avg_ns * 0.07]
            api = [2400.0, 2300.0, 1800.0, 9000.0, 500.0]
            queue = [12000.0, 11000.0, 900.0, 60000.0, 4000.0]
            total = [a + q + k for a, q, k in zip(api, queue, timing)]
            row = [str(pid), str(pid), str(device), str(count), str(count)]
            for values in (total, api, queue, timing):
                row += [f"{v:.1f}" for v in values]
            row += ["cudaGraphLaunch_v10000", name]
            lines.append(row)
    return _csv_text(lines)


def synthetic_breakdown(
    role: str = "decode",
    *,
    devices: int = 8,
    steps: int = 10,
    source: str = "simulated",
) -> KernelBreakdown:
    """Breakdown of the synthetic tables (labelled as simulated)."""
    rows = parse_kern_sum_csv(
        synthetic_kern_sum_csv(role, devices=devices, steps=steps)
    ) + parse_mem_time_sum_csv(
        synthetic_mem_time_sum_csv(role, devices=devices, steps=steps)
    )
    exec_rows = parse_kern_exec_sum_csv(
        synthetic_kern_exec_sum_csv(role, devices=devices, steps=steps)
    )
    return breakdown(
        rows,
        source=source,
        exec_rows=exec_rows,
        expected_devices=devices,
        notes=("simulated kernel breakdown (illustrative numbers)",),
    )
