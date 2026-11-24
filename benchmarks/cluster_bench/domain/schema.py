# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed contracts shared by every layer of the cluster benchmark harness.

Everything here is pure data: no I/O, no subprocesses. Profiles (YAML) are
validated into these models, the planner and analyzers consume and emit them,
and the event store persists their JSON form.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator

from benchmarks.cluster_bench.domain.accuracy import AccuracyResult, AccuracyVerdict
from benchmarks.cluster_bench.domain.base import Frozen, stable_hash
from benchmarks.cluster_bench.domain.fabric import (
    FabricTopology,
    LinkFinding,
    LinkMatrix,
)
from benchmarks.cluster_bench.domain.lifecycle import DeployLifecycle
from benchmarks.cluster_bench.domain.modelarch import MfuResult, ModelArch
from benchmarks.cluster_bench.domain.nodefacts import (
    ClockOffset,
    DiskRead,
    FactFinding,
    HealthDelta,
    SoftwareStack,
)
from benchmarks.cluster_bench.domain.pointstate import PointState
from benchmarks.cluster_bench.domain.requests import (
    RequestSummary,
    SloTier,
    default_tiers,
)
from benchmarks.cluster_bench.domain.switchload import SwitchLoadSummary

SCHEMA_VERSION = 1
GIB = 1 << 30
# Per-user output tok/s floors the report reads off the throughput curve.
INTERACTIVITY_FLOORS: tuple[float, ...] = (20.0, 50.0)
# Mean inter-node IB GB/s per node above which a point has cross-node traffic.
CROSS_NODE_MIN_GBS = 0.5


__all__ = ["Frozen", "stable_hash"]  # re-exported from domain.base


# --------------------------------------------------------------------------
# Hardware inventory (produced by agents/probe.py, parsed by domain/inventory)
# --------------------------------------------------------------------------


class GpuInfo(Frozen):
    index: int
    name: str
    uuid: str
    pci_bus_id: str  # normalized lowercase "0000:18:00.0"
    memory_mib: int
    numa_node: int | None = None
    compute_cap: str | None = None
    pcie_gen_max: int | None = None
    pcie_width_max: int | None = None


class HcaPort(Frozen):
    port: int
    state: str  # e.g. "4: ACTIVE"
    rate: str  # e.g. "400 Gb/sec (4X NDR)"
    rate_gbps: float  # gigabits per second parsed from ``rate``
    link_layer: str  # "InfiniBand" | "Ethernet"

    @property
    def active(self) -> bool:
        return self.state.strip().startswith("4")


class HcaInfo(Frozen):
    name: str  # e.g. mlx5_0
    pci_bus_id: str
    numa_node: int | None = None
    netdev: str | None = None
    board_id: str | None = None
    fw_ver: str | None = None
    ports: tuple[HcaPort, ...] = ()

    @property
    def active_ports(self) -> tuple[HcaPort, ...]:
        return tuple(p for p in self.ports if p.active)

    @property
    def nominal_gbs(self) -> float | None:
        """Line rate of the fastest active port in GB/s (Gb/s / 8)."""
        rates = [p.rate_gbps for p in self.active_ports if p.rate_gbps > 0]
        return max(rates) / 8 if rates else None


class DiskInfo(Frozen):
    path: str
    device: str
    fs_type: str
    total_bytes: int
    free_bytes: int


class NodeInventory(Frozen):
    name: str  # logical name: node0..nodeN-1 (node0 runs the harness)
    hostname: str
    address: str  # control-plane IPv4 (Lium: the WireGuard wg0 address)
    control_interface: str = "wg0"
    gpus: tuple[GpuInfo, ...]
    hcas: tuple[HcaInfo, ...] = ()
    # PCI bus id -> resolved sysfs device path; used for GPU<->NIC affinity.
    pci_paths: dict[str, str] = Field(default_factory=dict)
    # nvidia-smi topo -m matrix: row label -> column label -> link type.
    topo: dict[str, dict[str, str]] = Field(default_factory=dict)
    driver_version: str
    cuda_version: str | None = None
    cpu_model: str = ""
    cpu_count: int = 0
    mem_total_bytes: int = 0
    cgroup_mem_limit_bytes: int | None = None
    disks: tuple[DiskInfo, ...] = ()
    docker_version: str | None = None
    kernel: str = ""
    ofed_version: str | None = None
    nvidia_peermem: bool = False
    nvlink_active: bool = False

    @property
    def active_hcas(self) -> tuple[HcaInfo, ...]:
        return tuple(h for h in self.hcas if h.active_ports)


class Inventory(Frozen):
    nodes: tuple[NodeInventory, ...]

    @property
    def gpu_count(self) -> int:
        return sum(len(n.gpus) for n in self.nodes)

    def node(self, name: str) -> NodeInventory:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"unknown node {name!r}")


# --------------------------------------------------------------------------
# Profiles (profiles/*.yaml)
# --------------------------------------------------------------------------


class GpuProfile(Frozen):
    name: str  # profile id, e.g. "b200"
    match: tuple[str, ...]  # case-insensitive substrings of the nvidia-smi name
    arch: str  # e.g. "sm100"
    hbm_gb: float
    nvlink: bool
    weights_variant: str  # key into ModelProfile.checkpoints
    nominal_nvlink_gbs: float | None = None  # per GPU, per direction, GB/s
    nominal_pcie_gbs: float = 64.0  # per GPU, per direction, GB/s
    # Dense (no sparsity) Tensor Core peak per GPU, TFLOPS, by GEMM precision
    # ("fp4" | "fp8" | "bf16"): the MFU denominator.
    peak_dense_tflops: dict[str, float] = Field(default_factory=dict)
    hbm_tbs: float | None = None  # HBM bandwidth per GPU, TB/s
    lane_cap: int = 1024  # upper bound for AgentX concurrency grids
    notes: str = ""


class Checkpoint(Frozen):
    repo: str
    revision: str = "main"
    approx_size_gb: float
    # SGLang config this checkpoint needs in every role (e.g. its quantization
    # method), merged over the recipe and the family overrides.
    sglang: dict[str, Any] = Field(default_factory=dict)


class ModelProfile(Frozen):
    name: str
    served_model_name: str
    checkpoints: dict[str, Checkpoint]
    worker_args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    # SGLang config keys that make up speculative decoding; removed for MTP-off.
    spec_decode_keys: tuple[str, ...] = ()
    simulate_acceptance_env: dict[str, str] = Field(default_factory=dict)
    trust_remote_code: bool = True
    moe: bool = False  # mixture of experts: EPLB balancedness in diagnostics


class RecipeRef(Frozen):
    """A SGLang config embedded in a recipe ConfigMap (``data.<key>``)."""

    path: str  # repo-relative path of the recipe deploy.yaml
    key: str  # ConfigMap data key, e.g. "prefill.yaml"


Role = Literal["agg", "prefill", "decode"]
# What a run measures (domain/stages.py): the whole serving pipeline, or one PD
# stage on every GPU of the cluster (P-Only: prompt tokens/s; D-Only: output
# tokens/s with SGLang's fake KV transfer; A-Only: D-Only's decode role as the
# attention side of attention-FFN disaggregation, F simulated, domain/afd.py).
Stage = Literal["e2e", "prefill", "decode", "attention"]
# Where the machines come from: any SSH-reachable hosts, or a Lium.io rental.
Platform = Literal["generic", "lium"]
Transport = Literal["none", "mooncake", "nixl", "nccl", "deepep"]
CrossNodeClass = Literal["kv", "collective"]


# SGLang 0.5.21 split the CUDA-graph batch-size flags per phase and removed the
# old spellings, which argparse then rejects as ambiguous prefixes of the
# -decode/-prefill forms. Configs written for older images still use them, so
# role overrides, ladder rungs and recipe configs are read in the new names,
# each old flag mapped to the decode form it meant (as Dynamo's profiler does in
# dynamo/profiler/utils/config_modifiers/sglang.py).
LEGACY_SGLANG_KEYS: Mapping[str, str] = {
    "cuda-graph-bs": "cuda-graph-bs-decode",
    "cuda-graph-max-bs": "cuda-graph-max-bs-decode",
}


def modernize_sglang_keys(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``cfg`` in SGLang 0.5.21+ key names; an explicit new key wins."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        new = LEGACY_SGLANG_KEYS.get(key)
        if new is None:
            out[key] = value
        elif new not in cfg:
            out[new] = value
    return out


def _modern_sglang(value: Any) -> Any:
    return modernize_sglang_keys(value) if isinstance(value, Mapping) else value


class RoleSpec(Frozen):
    role: Role
    gpus_per_instance: int
    instances: int
    nodes_per_instance: int = 1
    base: RecipeRef | None = None
    # Overrides merged over the recipe config. A ``null`` value deletes a key.
    sglang: dict[str, Any] = Field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("sglang", mode="before")
    @classmethod
    def modern_sglang_keys(cls, value: Any) -> Any:
        return _modern_sglang(value)

    @property
    def total_gpus(self) -> int:
        return self.gpus_per_instance * self.instances


class GridSpec(Frozen):
    start: int
    max: int
    min: int = 8


class FamilySpec(Frozen):
    id: str
    title: str
    roles: tuple[RoleSpec, ...]
    transport: Transport = "none"
    # Cross-node data-plane traffic this topology is expected to generate.
    cross_node: tuple[CrossNodeClass, ...] = ()
    # Families sharing a group are alternatives (e.g. P:D ratios); the search
    # measures the first and picks at most one more by symptom.
    group: str | None = None
    priority: int = 100  # lower runs first; 0 = recipe baseline
    enabled: bool = True
    grid: GridSpec | None = None
    notes: str = ""
    # Cluster sizes (node counts) this family applies to; empty = any size the
    # roles scale to exactly (see domain/profiles.fit_family).
    nodes: tuple[int, ...] = ()
    # Set by fit_family: ``id``/``title`` may hold {prefill}/{decode}/{agg}/
    # {gpus} placeholders rendered with the fitted instance counts; the
    # templates and the id at the profile's own scale are kept here.
    base_id: str | None = None
    id_template: str | None = None
    title_template: str | None = None
    # Fitted GPUs / profile-scale GPUs (2.0 on 8 nodes for a 4-node profile);
    # concurrency grids scale with it.
    scale: float = 1.0
    # "e2e" families serve requests end to end; "prefill"/"decode" families run
    # one PD stage on the whole cluster (domain/stages.py). ``source`` is the
    # profile-scale id of the PD family a derived stage family came from.
    stage: Stage = "e2e"
    source: str | None = None

    @property
    def total_gpus(self) -> int:
        return sum(r.total_gpus for r in self.roles)

    def matches(self, ref: str) -> bool:
        """``ref`` names this family: its fitted id or its profile-scale id."""
        return ref in (self.id, self.base_id)


LadderTrigger = Literal["oom", "kernel_unsupported"]


class LadderRung(Frozen):
    """A fallback applied when a deployment fails with a known failure class."""

    id: str
    when: LadderTrigger
    sglang: dict[str, Any] = Field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    drop_args: tuple[str, ...] = ()  # flag names removed (with their value)
    env: dict[str, str] = Field(default_factory=dict)
    mem_fraction_delta: float = 0.0
    halve_cuda_graph_bs: bool = False
    roles: tuple[Role, ...] = ("agg", "prefill", "decode")

    @field_validator("sglang", mode="before")
    @classmethod
    def modern_sglang_keys(cls, value: Any) -> Any:
        return _modern_sglang(value)


class DeploymentProfile(Frozen):
    """profiles/families/<model>/<gpu>.yaml"""

    gpu: str
    model: str
    served_context: int
    families: tuple[FamilySpec, ...]
    ladder: tuple[LadderRung, ...] = ()
    default_grid: GridSpec = GridSpec(start=64, max=1024)

    def family(self, family_id: str) -> FamilySpec:
        for f in self.families:
            if f.id == family_id:
                return f
        raise KeyError(f"unknown family {family_id!r}")


# The agentx preset's warmup: 10 cache-pressure requests per lane.
AGENTX_PRESET_WARMUP_REQUESTS = 10
# Planning guess per warmup request of one lane (long-context, one-token
# outputs); observed point durations replace it once a spec has run.
REQUEST_WARMUP_ESTIMATE_S = 60.0


class BenchSpec(Frozen):
    id: str  # "smoke" | "screen" | "decision"
    # Duration warmup: --warmup-duration, for AgentX datasets
    # --agentic-cache-warmup-duration (unless agentx_warmup_requests is set).
    warmup_s: int
    duration_s: int
    unsafe_override: bool
    # AgentX request-budget warmup instead of warmup_s: N requests per lane
    # (--warmup-requests-per-lane N), or "preset" for the agentx preset's 10.
    # AgentX with neither falls back to the preset's too (it has no "off").
    agentx_warmup_requests: Annotated[int, Field(gt=0)] | Literal[
        "preset"
    ] | None = None

    @property
    def warmup_estimate_s(self) -> float:
        """Planning estimate of the warmup, whichever kind the dataset gets."""
        requests = self.agentx_warmup_requests
        if requests is None:
            return float(self.warmup_s)
        budget = AGENTX_PRESET_WARMUP_REQUESTS if requests == "preset" else requests
        return max(float(self.warmup_s), REQUEST_WARMUP_ESTIMATE_S * budget)


DEFAULT_BENCH_SPECS: dict[str, BenchSpec] = {
    "smoke": BenchSpec(
        id="smoke",
        warmup_s=0,
        duration_s=60,
        unsafe_override=True,
        agentx_warmup_requests=1,
    ),
    "screen": BenchSpec(
        id="screen", warmup_s=180, duration_s=480, unsafe_override=True
    ),
    # AgentX: the agentx preset's warmup (10 requests per lane) and no
    # --unsafe-override, so submission_valid can be true. The duration is
    # explicit: AIPerf honours any value >= 900 s; 1200 s keeps the confirm run
    # inside the default budget (--confirm-duration-s 3600 = the preset's own).
    "decision": BenchSpec(
        id="decision",
        warmup_s=300,
        duration_s=1200,
        unsafe_override=False,
        agentx_warmup_requests="preset",
    ),
    # Load of the opt-in diagnostic window (never ranked).
    "diagnose": BenchSpec(
        id="diagnose", warmup_s=120, duration_s=300, unsafe_override=True
    ),
}


# --------------------------------------------------------------------------
# Candidates and rendered launch plans
# --------------------------------------------------------------------------


P2pPath = Literal["nvlink", "pcie", "host"]


class Candidate(Frozen):
    family_id: str
    mtp: bool = True
    nic_count: int | None = None  # None = every active HCA
    # How GPUs of one node exchange data: "nvlink" (default), "pcie" (NVLink
    # switched off, P2P over PCIe) or "host" (no P2P: NCCL stages through host
    # memory). Only the NVLink comparison sets it.
    p2p: P2pPath = "nvlink"
    simulate_acceptance: bool = False
    rungs: tuple[str, ...] = ()  # applied ladder rung ids, in order
    # Diagnostic variant (never ranked): "" | "metrics" | "nsys" (nsys implies
    # metrics). Only set by the diagnose phase.
    diag: str = ""
    # A-Only null-F bound (never ranked): the F wait is off. Only set by the
    # f_bound step (domain/search.py).
    null_f: bool = False

    @property
    def key(self) -> str:
        nic = "all" if self.nic_count is None else str(self.nic_count)
        rungs = "+".join(self.rungs) or "-"
        key = (
            f"{self.family_id}|mtp={int(self.mtp)}|nic={nic}"
            f"|sim={int(self.simulate_acceptance)}|rungs={rungs}"
        )
        if self.p2p != "nvlink":
            key = f"{key}|p2p={self.p2p}"
        if self.null_f:
            key = f"{key}|f=null"
        return f"{key}|diag={self.diag}" if self.diag else key

    @property
    def fp(self) -> str:
        return stable_hash(self.key, 8)

    def with_rung(self, rung_id: str) -> Candidate:
        return self.model_copy(update={"rungs": (*self.rungs, rung_id)})


ContainerRole = Literal["control", "frontend", "agg", "prefill", "decode"]


class ContainerSpec(Frozen):
    node: str  # logical node name
    name: str  # docker container name
    role: ContainerRole
    instance: int = 0  # instance index within its role
    rank: int = 0  # node rank within a multi-node instance
    gpus: tuple[int, ...] = ()
    image: str
    env: dict[str, str] = Field(default_factory=dict)
    # Files written on the node before start: absolute path -> content.
    files: dict[str, str] = Field(default_factory=dict)
    command: tuple[str, ...]  # argv executed inside the container
    docker_run: tuple[str, ...] = ()  # full `docker run` argv
    system_port: int | None = None
    health_url: str | None = None  # only for rank-0 serving containers
    metrics_url: str | None = None
    hcas: tuple[str, ...] = ()  # HCAs this container may use


class LaunchPlan(Frozen):
    candidate: Candidate
    cand_fp: str
    namespace: str
    served_model_name: str
    frontend_url: str
    containers: tuple[ContainerSpec, ...]
    gpu_count: int
    transport: Transport
    cross_node: tuple[CrossNodeClass, ...]
    # Instances that must register with the frontend, by role.
    expected_instances: dict[str, int]
    # Node -> HCAs the data plane is restricted to (all active when unrestricted).
    nic_selection: dict[str, tuple[str, ...]]
    # Rendered SGLang config per (role, instance): key "<role>:<instance>".
    rendered_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()  # render-time adjustments (e.g. HiCache clamps)
    stage: Stage = "e2e"  # the family's stage (P-Only caps output at one token)

    @property
    def metrics_urls(self) -> tuple[str, ...]:
        return tuple(c.metrics_url for c in self.containers if c.metrics_url)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------

Purpose = Literal[
    "search",
    "mtp_ablation",
    "pilot",
    "nic",
    "confirm",
    "smoke",
    "diagnose",
    "nvlink",
    "f_bound",
]
AuditStatus = Literal["valid", "valid_with_recovery", "invalid"]


class PointKey(Frozen):
    cand_key: str
    concurrency: int
    spec: str
    purpose: Purpose
    rep: int = 0

    @property
    def id(self) -> str:
        return stable_hash(
            [self.cand_key, self.concurrency, self.spec, self.purpose, self.rep], 12
        )


class PointMetrics(Frozen):
    """The subset of profile_export_aiperf.json the harness decides on."""

    aiperf_version: str | None = None
    was_cancelled: bool = False
    # --scenario runs (metadata block): AIPerf's verdict on the scenario rules.
    scenario: str | None = None
    submission_valid: bool | None = None
    submission_invalid_reasons: tuple[str, ...] = ()
    # False when AIPerf degraded before finishing (e.g. its stall watchdog).
    is_complete: bool | None = None
    incomplete_reason: str | None = None
    request_count: int = 0
    error_count: int = 0
    output_tps: float | None = None  # system output tokens/s (+ reasoning)
    input_tps: float | None = (
        None  # system prompt tokens/s (AIPerf input_token_throughput)
    )
    total_output_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    user_tps_p50: float | None = None  # output tok/s per user, p50
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p99_ms: float | None = None
    request_throughput: float | None = None
    duration_s: float | None = None  # AIPerf benchmark_duration: requests' span
    # Wall time of the profiling phase, its grace period included: AIPerf divides
    # its throughputs by it, and the goodput and MFU of the records divide by it too.
    phase_s: float | None = None
    isl_avg: float | None = None
    osl_avg: float | None = None
    error_types: dict[str, int] = Field(default_factory=dict)
    window_start_ns: int | None = None
    window_end_ns: int | None = None

    @property
    def error_rate(self) -> float:
        total = self.request_count + self.error_count
        return self.error_count / total if total else 1.0


class AuditResult(Frozen):
    status: AuditStatus
    reasons: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.status != "invalid"


Plane = Literal["ib", "nvlink", "pcie", "pcie_nic"]
Method = Literal["measured", "derived", "unavailable"]


class BandwidthStat(Frozen):
    plane: Plane
    scope: str  # "<node>/<device>" or "<node>" or "cluster"
    direction: Literal["tx", "rx", "bidir"] = "tx"
    mean_gbs: float = 0.0
    p99_gbs: float = 0.0
    peak_gbs: float = 0.0
    ref_peak_gbs: float | None = None  # measured speed-of-light reference
    nominal_gbs: float | None = None  # link line rate
    method: Method = "measured"

    @property
    def util_vs_peak(self) -> float | None:
        return self.peak_gbs / self.ref_peak_gbs if self.ref_peak_gbs else None

    @property
    def util_vs_nominal(self) -> float | None:
        return self.peak_gbs / self.nominal_gbs if self.nominal_gbs else None


class EngagementResult(Frozen):
    ok: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    evidence: tuple[str, ...] = ()


class PointResult(Frozen):
    key: PointKey
    candidate: Candidate
    gpu_count: int
    metrics: PointMetrics | None = None
    audit: AuditResult
    # The objective per GPU: output tok/s (e2e, D-Only) or input tok/s (P-Only).
    tps_per_gpu: float | None = None
    input_tps_per_gpu: float | None = None  # prompt tok/s per GPU, every stage
    bandwidth: tuple[BandwidthStat, ...] = ()
    cross_node_gbs: float | None = None  # mean inter-node IB GB/s per node
    nic_effective: float | None = None  # (sum b)^2 / sum b^2 over HCAs
    nic_engaged: bool | None = None  # only restricted HCAs carried traffic
    staging_suspected: bool = False
    telemetry_coverage: float | None = None  # fraction of expected samples present
    telemetry_notes: tuple[str, ...] = ()  # why a bandwidth metric is degraded
    accept_len: float | None = None  # measured speculative acceptance length
    # Switch ports serving our HCAs (leaf downlinks + uplinks) over the AIPerf
    # run; None when PMA queries are unavailable. Full list: switch_load.json.
    switch_load: SwitchLoadSummary | None = None
    # Inference-service state, GPU/fabric stress and the bottleneck verdict
    # (full detail: servicestate.json and stress.json in the point dir).
    state: PointState | None = None
    # Per-request view from AIPerf's records: goodput under SLO tiers, TTFT by
    # session turn, session spans (domain/requests.py).
    requests: RequestSummary | None = None
    mfu: MfuResult | None = None  # model FLOPs utilization (domain/modelarch.py)
    artifact_dir: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0


class PeakResult(Frozen):
    """A speed-of-light microbenchmark result (KB evidence category 5)."""

    plane: Plane
    test: str  # e.g. "p2p_copy", "nccl_all_reduce_inter", "gdr_send_recv"
    scope: str
    nic_count: int | None = None
    value_gbs: float
    nominal_gbs: float | None = None
    tool: str
    command: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


FailureClass = Literal[
    "oom",
    "kernel_unsupported",
    "rdma_registration",
    "nccl",
    "port_in_use",
    "timeout",
    "container_exit",
    "smoke_failed",
    "engagement",
    "unknown",
]


class DeployResult(Frozen):
    candidate: Candidate
    cand_fp: str
    ok: bool
    load_seconds: float = 0.0
    failure: FailureClass | None = None
    detail: str = ""
    engagement: EngagementResult | None = None
    log_dir: str = ""
    # Cold-start phases and KV capacity parsed from the container logs
    # (per-container detail in <log_dir>/lifecycle.json).
    lifecycle: DeployLifecycle | None = None


# --------------------------------------------------------------------------
# Experiment summary (written by the finalize phase; the only input of the
# report and knowledge-base writers)
# --------------------------------------------------------------------------


class NoiseResult(Frozen):
    cand_key: str
    concurrency: int
    samples: tuple[float, ...]  # tok/s/GPU of each repetition
    mean: float
    stdev: float
    cv: float
    # Smallest delta distinguishable from noise for one new run vs the pilot mean.
    noise_floor: float
    noise_floor_pct: float


class FloorResult(Frozen):
    """Best throughput among points whose per-user p50 meets a floor."""

    user_tps_floor: float
    point_id: str | None = None
    cand_key: str | None = None
    concurrency: int | None = None
    tps_per_gpu: float | None = None
    interpolated_tps_per_gpu: float | None = None  # log2(concurrency) interpolation
    crossed: bool = False
    note: str = ""


FamilyStatus = Literal[
    "measured", "pruned", "deploy_failed", "unsupported", "skipped_budget", "disabled"
]


class FamilySummary(Frozen):
    family_id: str
    title: str
    status: FamilyStatus
    reason: str = ""
    best_point_id: str | None = None
    best_tps_per_gpu: float | None = None
    best_concurrency: int | None = None
    saturated: bool | None = None
    cross_node_gbs: float | None = None
    # Cross-node traffic the topology is designed to generate (from the profile).
    cross_node: tuple[CrossNodeClass, ...] = ()
    point_ids: tuple[str, ...] = ()
    # Accuracy gate verdict of the best point's candidate (None: not checked).
    accuracy: str | None = None


class NicArm(Frozen):
    nic_count: int
    point_id: str | None = None
    tps_per_gpu: float | None = None
    delta_vs_full_pct: float | None = None
    within_noise: bool | None = None
    nic_effective: float | None = None
    engaged: bool | None = None
    staging_suspected: bool = False
    status: str = "measured"  # measured | invalid | not_engaged | skipped


class KvRoleCapacity(Frozen):
    """KV cache pools of one serving role (cold-start logs + rendered config)."""

    role: str
    instances: int
    dp_ranks: int = 1  # attention-DP ranks per instance; each holds its own pool
    per_rank_tokens: int | None = None  # max_total_num_tokens (smallest instance)
    instance_tokens: int | None = None  # per_rank_tokens x dp_ranks
    total_tokens: int | None = None  # over all instances of the role
    kv_gb_per_rank: float | None = None
    bytes_per_token: float | None = None  # measured: KV pool bytes / KV tokens
    host_gb_per_rank: float | None = None  # HiCache host memory
    host_tokens_total: int | None = None
    max_running_requests: int | None = None  # per instance
    context_len: int | None = None


class KvCapacity(Frozen):
    """How much context the leader deployment can hold at once."""

    roles: tuple[KvRoleCapacity, ...] = ()
    serving_role: str = "agg"  # where in-flight KV lives (agg, or decode with PD)
    total_tokens: int | None = None  # serving role, all instances
    max_request_tokens: int | None = None  # one request lives on one rank
    context_p50: float | None = None  # ISL + OSL of one request at the leader point
    context_p90: float | None = None
    requests_at_p50: int | None = None  # total_tokens // context_p50
    requests_at_p90: int | None = None
    usage_avg: float | None = None  # KV pool usage at the leader point (0..1)
    usage_peak: float | None = None
    arch_bytes_per_token: float | None = None  # from config.json, for comparison
    notes: tuple[str, ...] = ()

    def role(self, name: str) -> KvRoleCapacity | None:
        return next((r for r in self.roles if r.role == name), None)


LinkKind = Literal["nvlink", "pcie", "pcie_nic", "ib"]
LinkStatus = Literal["idle", "ok", "tight", "unavailable"]


DEFAULT_LINK_TARGET_UTIL = 0.7


class LinkNeed(Frozen):
    """Busiest link of one kind at the leader's saturation point."""

    link: LinkKind
    scope: str = ""  # busiest device (GPU / HCA) or node for "ib"
    p99_gbs: float | None = None  # per direction, worse of tx / rx
    peak_gbs: float | None = None
    capacity_gbs: float | None = None  # nominal, per direction
    util_p99: float | None = None  # p99 / capacity
    need_gbs: float | None = None  # p99 / target utilization
    status: LinkStatus = "unavailable"


class InterconnectNeeds(Frozen):
    """Interconnect the leader needs, and the smallest NIC count that holds up."""

    target_util: float = (
        DEFAULT_LINK_TARGET_UTIL  # links are sized to run at most this busy
    )
    point_id: str | None = None
    family_id: str | None = None  # the leader point's family
    sweep_family_id: str | None = None  # family the NIC sweep ran on
    links: tuple[LinkNeed, ...] = ()
    nic_rate_gbs: float | None = None  # one NIC, per direction
    nics_per_node: int | None = None  # active today
    min_nics_measured: int | None = (
        None  # NIC sweep (sweep family): fewest within noise
    )
    min_nics_reason: str = ""
    min_nics_estimated: int | None = None  # ceil(IB need per node / NIC rate)
    pcie_alt_gbs: float | None = None  # PCIe per GPU, per direction
    nvlink_needed: bool | None = None  # GPU-GPU need above what PCIe offers
    notes: tuple[str, ...] = ()

    def link(self, kind: str) -> LinkNeed | None:
        return next((n for n in self.links if n.link == kind), None)

    @property
    def sweep_on_leader(self) -> bool:
        """The NIC sweep ran on the leader's own family (or none ran)."""
        return self.sweep_family_id in (None, self.family_id)


NvlinkMethod = Literal["bw_mode_off", "link_width_0", "nccl_no_p2p"]


class NvlinkComparison(Frozen):
    """Throughput with NVLink switched off (GPU-GPU over PCIe) vs on."""

    status: str = "skipped"  # measured | invalid | failed | skipped
    reason: str = ""  # why this configuration, or why nothing was measured
    family_id: str | None = None
    concurrency: int | None = None
    on_point_id: str | None = None
    off_point_id: str | None = None
    on_tps_per_gpu: float | None = None
    off_tps_per_gpu: float | None = None
    delta_pct: float | None = None  # off vs on
    within_noise: bool | None = None
    on_user_tps: float | None = None
    off_user_tps: float | None = None
    on_ttft_p50_ms: float | None = None
    off_ttft_p50_ms: float | None = None
    on_itl_p50_ms: float | None = None
    off_itl_p50_ms: float | None = None
    method: NvlinkMethod | None = None  # how NVLink was taken out of the path
    path: P2pPath | None = None  # what carried GPU-GPU traffic in the off run
    p2p_gbs: float | None = None  # GPU-GPU copy median with NVLink off
    peer_pairs: int | None = None  # GPU pairs with peer access over PCIe
    nvlink_p99_gbs: float | None = None  # busiest GPU's NVLink p99 in the off run
    pcie_p99_gbs: float | None = None  # busiest GPU's PCIe p99 in the off run
    pcie_capacity_gbs: float | None = None
    restored: bool | None = None  # NVLink back to full after the run
    notes: tuple[str, ...] = ()


class MtpComparison(Frozen):
    on_point_id: str | None = None
    off_point_id: str | None = None
    on_tps_per_gpu: float | None = None
    off_tps_per_gpu: float | None = None
    delta_pct: float | None = None
    within_noise: bool | None = None
    accept_len: float | None = None


# --------------------------------------------------------------------------
# A-Only: the attention side of attention-FFN disaggregation (domain/afd.py)
# --------------------------------------------------------------------------

AfdLink = Literal["net", "intra"]
AfdLayout = Literal["staged", "direct"]
AfdExpertDtype = Literal["nvfp4", "mxfp4", "fp8", "bf16"]  # routed-expert weights


class AfdFSpec(Frozen):
    """The F an A-Only run simulates (``--afd-f``); every field has a default."""

    # F GPUs serving one A instance; None: the smallest power of two that
    # holds every routed expert at mem_fraction of HBM and divides the experts.
    gpus: int | None = Field(None, ge=1)
    # A<->F link: F on other nodes (measured inter-node bandwidth) or on the
    # same node (measured P2P bandwidth).
    link: AfdLink = "net"
    # staged: every F rank receives the whole batch and returns a partial sum
    # (SGLang AFD's default nccl backend); direct: a row goes only to the F
    # ranks owning its top-k experts (DeepEP / MegaMoE backends).
    layout: AfdLayout = "staged"
    microbatches: int = Field(2, ge=1, le=8)  # decode microbatches (1 = serial)
    wire: Literal["bf16", "fp8"] = "bf16"  # hidden-state precision on the wire
    bw_gbs: float | None = Field(None, gt=0)  # GB/s per A GPU, one direction
    lat_us: float | None = Field(None, ge=0)  # one-way latency of one transfer
    eff_mem: float = Field(0.7, gt=0, le=1)  # F's share of HBM bandwidth
    eff_flop: float = Field(0.5, gt=0, le=1)  # F's share of its dense peak
    mem_fraction: float = Field(0.8, gt=0, le=1)  # HBM share F's experts may use
    # Routed-expert precision; None: the checkpoint variant's (nvfp4-nvidia ->
    # nvfp4). A --model-repo checkpoint has no variant name and needs it.
    expert_dtype: AfdExpertDtype | None = None


class AfdPlan(Frozen):
    """The F model of one run, resolved once before the search (DECISION)."""

    spec: AfdFSpec = AfdFSpec()
    layers: int  # target decoder layers (the MTP layer has this index)
    moe_layers: int
    hidden: int
    moe_intermediate: int
    experts: int  # routed experts per layer
    top_k: int
    expert_dtype: str  # routed-expert weights: nvfp4 | mxfp4 | fp8 | bf16
    layer_bytes: float  # routed-expert weight bytes of one MoE layer
    f_gpus: int
    f_gpus_source: str
    f_hbm_tbs: float
    f_peak_tflops: float
    f_peak_precision: str
    link_gbs_per_gpu: float  # A<->F GB/s per A GPU, one direction
    link_source: str
    lat_us: float
    lat_source: str
    notes: tuple[str, ...] = ()

    @property
    def routed_gb(self) -> float:
        return self.layer_bytes * self.moe_layers / 1e9


class AfdStepRow(Frozen):
    """Modelled F time of one MoE layer, and of one forward, at ``rows`` rows."""

    rows: int
    latency_us: float  # both transfers of one microbatch
    link_us: float  # one microbatch's bytes on the wire
    f_mem_us: float  # one microbatch's expert weights read on F
    f_flop_us: float  # one microbatch's expert GEMMs on F
    serial_us: float  # f(T): the first MoE layer of a forward, or m == 1
    overlap_us: float  # m * f(T/m): later layers (A's own work runs meanwhile)
    step_ms: float  # F wait of one forward; A's own work not included
    bound: Literal["latency", "link", "memory", "compute"]


class AfdTraffic(Frozen):
    """A<->F bytes per output token and the link share they need, per A GPU
    (an instance's GPU count cancels out of every ratio)."""

    fan_out: float  # F ranks a row reaches
    rows_per_token: float  # MoE rows per output token (verify rows / accepted)
    to_f_bytes_per_token: float
    from_f_bytes_per_token: float
    link_gbs_per_gpu: float  # one direction
    sim_need_gbs_per_gpu: float | None = None  # at the simulated-F leader's tok/s
    sim_util: float | None = None
    null_need_gbs_per_gpu: float | None = None  # at the null-F bound's tok/s
    null_util: float | None = None


class AfdBound(Frozen):
    """Simulated F vs the null-F bound (wait off) at the leader's concurrency."""

    status: Literal["measured", "invalid", "failed", "skipped"] = "skipped"
    reason: str = ""
    concurrency: int | None = None
    sim_point_id: str | None = None
    null_point_id: str | None = None
    sim_tps_per_gpu: float | None = None
    null_tps_per_gpu: float | None = None
    f_cost_pct: float | None = None  # (null - sim) / null * 100
    within_noise: bool | None = None
    sim_user_tps: float | None = None
    null_user_tps: float | None = None
    sim_itl_p50_ms: float | None = None
    null_itl_p50_ms: float | None = None


class AfdReport(Frozen):
    """What an A-Only summary adds: the F model, the bound and the traffic."""

    plan: AfdPlan
    n_a: int  # GPUs of one A instance of the leader's family
    bound: AfdBound = AfdBound()
    traffic: AfdTraffic | None = None
    steps: tuple[AfdStepRow, ...] = ()
    weight_gb_per_gpu: float | None = None  # measured (engine log), A side
    # Expected A weights per GPU without the target's routed experts and with
    # them (domain/afd.expected_weights); None: no estimate.
    null_weight_gb_per_gpu: float | None = None
    loaded_weight_gb_per_gpu: float | None = None


Verdict = Literal["ok", "partial", "unsupported", "failed"]


class ProviderMember(Frozen):
    name: str  # node<rank>
    rank: int
    huid: str = ""  # lium pod handle (changes per rental)
    pod_id: str = ""
    gpu_type: str | None = None
    gpu_count: int | None = None


class ProviderInfo(Frozen):
    """Where the cluster came from (written by the controller as provider.json).

    Never holds credentials or public addresses.
    """

    provider: str = "lium"  # "lium" | "ssh" (machines the user brought)
    cluster_id: str | None = None
    cluster_name: str | None = None
    fabric_id: str | None = None
    fabric_type: str | None = None  # lium's label: "infiniband" | "roce"
    link_rate: str | None = None
    fabric_measured: bool | None = None
    gpu_type: str | None = None
    node_count: int | None = None
    price_per_node_hour: float | None = None  # USD
    price_per_hour: float | None = None  # USD, whole cluster as billed
    template: str | None = None
    members: tuple[ProviderMember, ...] = ()


class CostSummary(Frozen):
    """Rental cost turned into $/token at the measured operating points."""

    currency: str = "USD"
    source: str  # "lium" | "--price-per-node-hour"
    price_per_node_hour: float
    cluster_per_hour: float
    experiment_usd: float  # spent_hours x cluster_per_hour
    leader_usd_per_mtok: float | None = None  # per 1e6 output tokens at saturation
    floor_usd_per_mtok: dict[str, float] = Field(default_factory=dict)  # floor -> $


class AccuracyCheck(Frozen):
    """One accuracy gate check (result without per-item rows; see accuracy/)."""

    cand_key: str
    family_id: str
    result: AccuracyResult
    verdict: AccuracyVerdict


class DiagnosticsResult(Frozen):
    """Opt-in diagnostic window of one candidate (app/phases/diagnose.py)."""

    cand_key: str
    family_id: str
    concurrency: int
    nsys: str = "off"  # off | unavailable | ok | failed
    nsys_version: str | None = None
    kernels: Any = None  # domain/kernels.KernelBreakdown (as JSON)
    # role -> domain/steps.StepBreakdown (as JSON): wall-time split of the steps
    timeline: Any = None
    capture_steps: int | None = None  # forward steps start_profile captured
    engine: dict[str, float] = Field(default_factory=dict)  # role.metric -> value
    output_tps: float | None = None  # inside the window (overhead included)
    notes: tuple[str, ...] = ()


class ExperimentSummary(Frozen):
    exp_id: str
    schema_version: int = SCHEMA_VERSION
    fingerprint: str
    # Fingerprint without the node count: one per SKU+fabric across sizes.
    shape_fingerprint: str | None = None
    gpu_profile: str
    gpu_name: str
    node_count: int
    gpus_per_node: int
    model: str
    checkpoint: Checkpoint
    served_context: int
    dataset: str
    aiperf_version: str
    image: str
    stage: Stage = "e2e"
    platform: Platform = "generic"
    started_at: float
    ended_at: float
    budget_hours: float
    spent_hours: float
    inventory: Inventory
    families: tuple[FamilySummary, ...]
    points: tuple[PointResult, ...]
    deploys: tuple[DeployResult, ...]
    peaks: tuple[PeakResult, ...]
    leader: Candidate | None = None
    leader_point_id: str | None = None
    saturated: bool | None = None
    saturation_note: str = ""
    floors: tuple[FloorResult, ...] = ()
    noise: NoiseResult | None = None
    mtp: MtpComparison | None = None
    nic_family_id: str | None = None
    nic_concurrency: int | None = None
    nic_reason: str = ""
    nic_arms: tuple[NicArm, ...] = ()
    confirm_point_id: str | None = None
    dropped_traces: int | None = None
    conditions: tuple[str, ...] = ()  # e.g. "AIPerf client co-located on node0"
    limitations: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    verdict: Verdict = "ok"
    # The configuration the experiment ran with (for exact reproduction).
    run_config: RunConfig | None = None
    provider: ProviderInfo | None = None
    cost: CostSummary | None = None
    # Upstream fabric: switches, rails, hops, link tests (domain/fabric.py).
    fabric: FabricTopology | None = None
    fabric_links: LinkMatrix | None = None
    link_findings: tuple[LinkFinding, ...] = ()
    fabric_findings: tuple[str, ...] = ()
    fabric_signature: dict[str, Any] = Field(default_factory=dict)
    # Node facts and health (domain/nodefacts.py).
    software: SoftwareStack | None = None
    facts_signature: dict[str, Any] = Field(default_factory=dict)
    fact_findings: tuple[FactFinding, ...] = ()
    health_deltas: tuple[HealthDelta, ...] = ()
    clock_offsets: tuple[ClockOffset, ...] = ()
    disk_reads: tuple[DiskRead, ...] = ()
    # Accuracy gate checks in order (domain/accuracy.py); the leader is the
    # best configuration that did not fail (``accuracy_of``).
    accuracy: tuple[AccuracyCheck, ...] = ()
    # Opt-in diagnostic window (kernel breakdown, EPLB, MFU), never ranked.
    diagnostics: DiagnosticsResult | None = None
    # Focus resources: interconnect needs, model FLOPs utilization, KV capacity.
    model_arch: ModelArch | None = None
    interconnect: InterconnectNeeds | None = None
    kv_capacity: KvCapacity | None = None
    nvlink: NvlinkComparison | None = None
    afd: AfdReport | None = None  # A-Only (domain/afd.py)

    def accuracy_of(self, cand_key: str | None) -> AccuracyCheck | None:
        """The last check of ``cand_key`` (None: never checked)."""
        found = [c for c in self.accuracy if c.cand_key == cand_key]
        return found[-1] if found else None

    def point(self, point_id: str | None) -> PointResult | None:
        if point_id is None:
            return None
        for p in self.points:
            if p.key.id == point_id:
                return p
        return None


# --------------------------------------------------------------------------
# Events (append-only store)
# --------------------------------------------------------------------------


class EventType(str, Enum):
    RUN_START = "run_start"
    SEGMENT = "segment"  # a wall-clock span the runner was alive
    PHASE_START = "phase_start"
    PHASE_DONE = "phase_done"
    INVENTORY = "inventory"
    FACTS = "facts"  # node facts / software / clocks / disk reads (tag in data)
    FABRIC = "fabric"  # fabric topology, link matrix, switch port refs
    DOWNLOAD = "download"
    PEAK = "peak"
    DEPLOY = "deploy"
    POINT = "point"
    POINT_ABORTED = "point_aborted"
    NOISE = "noise"
    ACCURACY = "accuracy"  # accuracy gate result + verdict of one candidate
    DIAGNOSTICS = "diagnostics"  # opt-in diagnostic window (kernels, EPLB, MFU)
    NVLINK = "nvlink"  # NVLink switched off / restored on every node
    MODEL_ARCH = "model_arch"  # architecture parsed from the checkpoint's config.json
    DECISION = "decision"
    NOTE = "note"
    RUN_DONE = "run_done"


class Event(Frozen):
    type: EventType
    ts: float
    data: dict[str, Any] = Field(default_factory=dict)
    v: int = SCHEMA_VERSION


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------


class BudgetConfig(Frozen):
    total_hours: float = 9.5
    # The steps after the search keep the budget they will need, costed from
    # the run's state (domain/search.py reserve_after): this share is added
    # for deploy-time variance, and finalize keeps a fixed reserve.
    reserve_margin: float = 0.10
    reserve_finalize_min: float = 10.0
    # Deploy estimate until a deploy is observed; refined from observed costs.
    deploy_estimate_min: float = 18.0


class AccuracyConfig(Frozen):
    """Accuracy gate (GSM8K few-shot through the frontend; domain/accuracy.py)."""

    enabled: bool = True
    questions: int = 200
    shots: int = 5
    concurrency: int = 32
    max_tokens: int = 512
    # Defaults from exact binomial analysis at n=200 (domain/accuracy.py): a
    # healthy model (>= 0.90) essentially never fails; a true 0.75 fails 94%.
    floor: float = 0.80  # absolute minimum accuracy (catches broken numerics)
    max_drop: float = 0.07  # allowed drop below the best checked config
    max_error_rate: float = 0.10  # more failed requests: inconclusive
    max_checks: int = 3  # configs checked before the leader stays unchecked


class DiagnoseConfig(Frozen):
    """Opt-in diagnostic window after the confirmation (not an official result)."""

    enabled: bool = False
    nsys: bool = True  # kernel-time breakdown (needs nsys in or added to the image)
    metrics: bool = True  # --enable-mfu-metrics, device timer, EPLB balancedness
    steps: int = 10  # forward steps captured by start_profile
    load_s: float = 300.0  # AIPerf load length inside the window

    @property
    def window_s(self) -> float:
        """Budget estimate of the window without its deploy (load + capture + stats)."""
        return self.load_s + 600.0 if self.enabled else 0.0


class RunConfig(Frozen):
    exp_id: str
    model: str = "glm-5.3"
    gpu_profile: str | None = None  # auto-detected when None
    image: str
    workdir: str  # node-local harness directory, identical on every node
    model_root: str | None = None  # auto-selected local disk when None
    dataset: str = "semianalysis_cc_traces_weka_062126"
    aiperf_version: str = "0.14.0"
    budget: BudgetConfig = BudgetConfig()
    families: tuple[str, ...] = ()  # restrict to these family ids when set
    nic_counts: tuple[int, ...] = (8, 4, 2, 1)
    simulate_acceptance: bool = False
    recipe_parity: bool = False
    fallback_model: str | None = None
    # Untuned generic model (--model-repo): no tuned families, auto TP sizing.
    model_repo: str | None = None
    model_revision: str = "main"
    model_size_gb: float | None = None
    served_context: int | None = None  # overrides the profile's served context
    # Checkpoint variant (a key of the model profile's ``checkpoints``); None
    # picks the GPU profile's weights_variant.
    checkpoint: str | None = None
    # Weights already at this absolute path on every node (--model-path): no
    # Hugging Face download; the copies must agree in file names and sizes.
    model_path: str | None = None
    # HF_ENDPOINT for node-side Hugging Face downloads (model files, the AIPerf
    # dataset), e.g. a mirror where huggingface.co is unreachable.
    hf_endpoint: str | None = None
    # Control-plane NIC: "auto" finds the interface that holds each node's
    # control address (wg0 on Lium); a name applies to every node.
    control_interface: str = "auto"
    stage: Stage = "e2e"
    platform: Platform = "generic"
    # Extra ``aiperf profile`` arguments (--aiperf-arg), appended verbatim.
    aiperf_args: tuple[str, ...] = ()
    bench_specs: dict[str, BenchSpec] = Field(
        default_factory=lambda: dict(DEFAULT_BENCH_SPECS)
    )
    frontend_port: int = 8000
    dry_run: bool = False
    # Fabric phase: LLDP listen time on RoCE ports; perftest link matrix.
    lldp_wait_s: float = 35.0
    fabric_link_matrix: bool = True
    # USD per node-hour for cost metrics when no provider.json says otherwise.
    price_per_node_hour: float | None = None
    # Goodput SLO tiers applied to every point's AIPerf records.
    slo_tiers: tuple[SloTier, ...] = Field(default_factory=default_tiers)
    accuracy: AccuracyConfig = AccuracyConfig()
    diagnose: DiagnoseConfig = DiagnoseConfig()
    # Interconnect sizing: a link should run at most this busy at p99 (0..1).
    link_target_util: float = Field(DEFAULT_LINK_TARGET_UTIL, gt=0.0, le=1.0)
    # Re-measure the leader with NVLink switched off (GPU-GPU over PCIe).
    nvlink_ablation: bool = True
    # A-Only: the F the attention side waits for (--afd-f, domain/afd.py).
    afd_f: AfdFSpec = AfdFSpec()


# ExperimentSummary references RunConfig, which is defined after it.
ExperimentSummary.model_rebuild()
