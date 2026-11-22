# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Knowledge-base records: one JSONL row per fact, keyed by cluster fingerprint.

Every record carries

* ``schema`` - ``cluster_bench.<kind>/v1`` (the JSON key is ``schema``),
* ``fp`` and ``exp_id`` - the cluster fingerprint and the experiment,
* ``key`` - the natural identity of the logical record (stable across
  re-exports of the same experiment),
* ``rec_id`` - ``stable_hash(canonical payload without rec_id, 16)``.

``rec_id`` identifies exact content, so re-exporting an experiment is a
no-op; ``key`` identifies the logical record, so a re-export with different
values for the same record is reported as a conflict instead of becoming a
second truth.

``build_records`` derives every record of one experiment from its
``ExperimentSummary`` (plus launch plans, when available) and nothing else.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.cluster_bench.domain.kernels import KernelBreakdown
from benchmarks.cluster_bench.domain.lifecycle import DeployLifecycle
from benchmarks.cluster_bench.domain.pointstate import PointState
from benchmarks.cluster_bench.domain.schema import (
    BandwidthStat,
    ExperimentSummary,
    LaunchPlan,
    PointResult,
    stable_hash,
)
from benchmarks.cluster_bench.report import facts
from benchmarks.cluster_bench.report.fmt import iso_utc
from benchmarks.cluster_bench.report.plans import PlanLookup, role_views

R = TypeVar("R", bound="KbRecord")

FindingKind = Literal[
    "topology",
    "user_floor",
    "saturation",
    "noise",
    "mtp",
    "nic_count",
    "nic_irrelevant",
    "bandwidth",
    "deploy",
]
FindingTable = Literal["knob", "scenario"]


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


class KbRecord(BaseModel):
    """Base of every KB row; see the module docstring for the common fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    FILE_KIND: ClassVar[str] = ""
    SCHEMA: ClassVar[str] = ""

    record_schema: str = Field(alias="schema")
    fp: str
    exp_id: str
    key: str
    rec_id: str = ""

    @classmethod
    def create(cls: type[R], **fields: Any) -> R:
        """Build a record with its schema and content-derived ``rec_id``."""
        record = cls(record_schema=cls.SCHEMA, **fields)
        return record.model_copy(update={"rec_id": stable_hash(record.payload(), 16)})

    def payload(self) -> dict[str, Any]:
        """Canonical JSON payload without ``rec_id`` (what ``rec_id`` hashes)."""
        return self.model_dump(mode="json", by_alias=True, exclude={"rec_id"})

    def as_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def rec_id_of(row: dict[str, Any]) -> str:
    """Recompute the ``rec_id`` of a stored JSON row."""
    return stable_hash({k: v for k, v in row.items() if k != "rec_id"}, 16)


class RunRecord(KbRecord):
    """One measured point (``PointResult``) with its workload context."""

    FILE_KIND: ClassVar[str] = "run"
    SCHEMA: ClassVar[str] = "cluster_bench.run/v1"

    point_id: str
    family: str
    family_title: str
    cand_key: str
    purpose: str
    concurrency: int
    spec: str
    rep: int
    mtp: bool
    simulate_acceptance: bool
    status: str
    audit_reasons: tuple[str, ...]
    gpu_count: int
    tps_per_gpu: float | None
    output_tps: float | None
    user_tps_p50: float | None
    ttft_p50_ms: float | None
    itl_p50_ms: float | None
    error_rate: float | None
    cross_node_gbs: float | None
    nic_count: int | None
    p2p: str = "nvlink"  # GPU-GPU path: nvlink | pcie | host (NVLink off)
    nic_effective: float | None
    nic_engaged: bool | None
    staging_suspected: bool
    accept_len: float | None
    gpu_profile: str
    model: str
    checkpoint: str
    dataset: str
    served_context: int
    artifact_dir: str
    # P-Only / D-Only (domain/stages.py): tps_per_gpu is input tok/s/GPU for
    # P-Only; rows exported before stages existed are e2e.
    stage: str = "e2e"
    input_tps_per_gpu: float | None = None
    # Round 2 (optional): why the point saturated, energy, fabric pressure.
    bottleneck: str | None = None
    bottleneck_confidence: float | None = None
    kv_usage_avg: float | None = None  # serving role (decode or agg)
    queue_reqs_avg: float | None = None
    running_reqs_avg: float | None = None
    prefix_hit_rate: float | None = None
    energy_per_token_j: float | None = None
    power_mean_w: float | None = None
    throttled_fraction: float | None = None
    ib_xmit_wait_fraction_max: float | None = None
    switch_uplink_util_max: float | None = None
    switch_downlink_util_max: float | None = None
    switch_discards: int | None = None
    # Round 3 (optional): engine internals, goodput and multi-turn behaviour.
    decode_cuda_graph_fraction: float | None = None
    spec_accept_rate: float | None = None
    retracted_tokens_per_generated: float | None = None
    evicted_tokens_per_prompt: float | None = None
    stage_prefill_forward_s: float | None = None
    stage_kv_transfer_s: float | None = None
    stage_decode_wait_s: float | None = None
    goodput_tps_per_gpu: dict[str, float] = Field(default_factory=dict)  # SLO tier ->
    good_fraction: dict[str, float] = Field(default_factory=dict)
    ttft_first_turn_p50_s: float | None = None
    ttft_late_turn_p50_s: float | None = None
    # NVML GPM (serving role = decode or agg; prefill separately).
    gpu_regime: str | None = None
    sm_pct_mean: float | None = None
    tensor_pct_mean: float | None = None
    dram_pct_mean: float | None = None
    prefill_gpu_regime: str | None = None
    prefill_tensor_pct_mean: float | None = None
    gpm_availability: dict[str, int] = Field(default_factory=dict)
    # Round 4: model FLOPs utilization (config.json FLOPs / dense peak).
    mfu_total: float | None = None
    mfu_prefill: float | None = None
    mfu_decode: float | None = None
    achieved_tflops_per_gpu: float | None = None
    mfu_precision: str | None = None


class PeakRecord(KbRecord):
    """One speed-of-light microbenchmark result (evidence category 5)."""

    FILE_KIND: ClassVar[str] = "peak"
    SCHEMA: ClassVar[str] = "cluster_bench.peak/v1"

    plane: str
    test: str
    scope: str
    nic_count: int | None
    value_gbs: float
    nominal_gbs: float | None
    fraction_of_nominal: float | None
    tool: str
    command: str
    details: dict[str, Any]
    evidence_category: int = Field(default=5, ge=1, le=5)
    gpu_profile: str


class UtilRecord(KbRecord):
    """One under-load bandwidth statistic of a usable point."""

    FILE_KIND: ClassVar[str] = "util"
    SCHEMA: ClassVar[str] = "cluster_bench.util/v1"

    point_id: str
    family: str
    cand_key: str
    purpose: str
    concurrency: int
    leader: bool
    plane: str
    scope: str
    direction: str
    mean_gbs: float
    p99_gbs: float
    peak_gbs: float
    ref_peak_gbs: float | None
    nominal_gbs: float | None
    util_vs_peak: float | None
    util_vs_nominal: float | None
    method: str


class NicSweepRecord(KbRecord):
    """One arm of the NIC-count comparison."""

    FILE_KIND: ClassVar[str] = "nic"
    SCHEMA: ClassVar[str] = "cluster_bench.nic/v1"

    family: str | None
    concurrency: int | None
    nic_count: int
    point_id: str | None
    tps_per_gpu: float | None
    delta_vs_full_pct: float | None
    within_noise: bool | None
    noise_floor_pct: float | None
    nic_effective: float | None
    engaged: bool | None
    staging_suspected: bool
    status: str
    leader_cross_node: bool | None
    reason: str
    gpu_profile: str


class DeployRecord(KbRecord):
    """One deployment attempt and its outcome."""

    FILE_KIND: ClassVar[str] = "deploy"
    SCHEMA: ClassVar[str] = "cluster_bench.deploy/v1"

    cand_fp: str
    cand_key: str
    family: str
    mtp: bool
    nic_count: int | None
    simulate_acceptance: bool
    rungs: tuple[str, ...]
    attempt: int
    ok: bool
    load_seconds: float
    failure: str | None
    detail: str
    engagement_ok: bool | None
    engagement_checks: dict[str, bool]
    log_dir: str
    # Round 2 (optional): cold start and KV capacity from the container logs.
    time_to_ready_s: float | None = None
    weights_load_s: float | None = None
    cuda_graph_s: float | None = None
    kv_tokens_min: int | None = None
    max_running_requests_min: int | None = None
    log_issues: dict[str, int] = Field(default_factory=dict)


class RoleConfig(BaseModel):
    """Effective SGLang config, env and argv of one role (masked)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    instances: int
    containers: int
    nodes: tuple[str, ...]
    sglang: dict[str, Any]
    config_variants: int
    env: dict[str, str]
    command: tuple[str, ...]


class ConfigRecord(KbRecord):
    """The effective configuration of one deployed candidate (from its plan)."""

    FILE_KIND: ClassVar[str] = "config"
    SCHEMA: ClassVar[str] = "cluster_bench.config/v1"

    cand_fp: str
    cand_key: str
    family: str
    transport: str
    cross_node: tuple[str, ...]
    gpu_count: int
    nic_selection: dict[str, tuple[str, ...]]
    roles: tuple[RoleConfig, ...]


class FindingRecord(KbRecord):
    """One claim with its evidence, renderable in the agent-docs table schemas.

    ``table == "knob"`` rows render as ``Symptom | Candidate knobs | Expected
    metric effect | Validation | Caveat``; ``"scenario"`` rows as ``Scenario |
    Knobs / config surface | Evidence signal | What to try first | Caveat``
    (``symptom`` holds the scenario, ``effect`` the evidence signal).
    """

    FILE_KIND: ClassVar[str] = "finding"
    SCHEMA: ClassVar[str] = "cluster_bench.finding/v1"

    kind: FindingKind
    table: FindingTable
    claim: str
    symptom: str
    knobs: str
    effect: str
    validation: str
    try_first: str
    caveat: str
    evidence_point_ids: tuple[str, ...]
    evidence_category: int = Field(ge=1, le=5)
    delta_pct: float | None
    noise_floor_pct: float | None
    family: str | None
    concurrency: int | None
    gpu_profile: str
    model: str


class NodeSnapshot(BaseModel):
    """Per-node hardware facts of a cluster snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    gpus: int
    gpu_names: tuple[str, ...]
    hcas: int
    active_hcas: tuple[str, ...]
    hca_rates: str
    driver_version: str
    nvlink_active: bool
    nvidia_peermem: bool


class ClusterRecord(KbRecord):
    """Hardware fingerprint snapshot as observed by one experiment."""

    FILE_KIND: ClassVar[str] = "cluster"
    SCHEMA: ClassVar[str] = "cluster_bench.cluster/v1"

    gpu_profile: str
    gpu_name: str
    node_count: int
    gpus_per_node: int
    hbm_mib: int | None
    compute_cap: str | None
    nvlink_active: bool | None
    driver_version: str | None
    cuda_version: str | None
    ofed_version: str | None
    nvidia_peermem: bool | None
    kernel: str | None
    docker_version: str | None
    cpu_model: str | None
    cpu_count: int | None
    mem_total_bytes: int | None
    nodes: tuple[NodeSnapshot, ...]
    # Round 2 (optional): node facts signature and the image's software stack.
    facts_signature: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, str] = Field(default_factory=dict)
    shape_fp: str | None = None


class ExperimentRecord(KbRecord):
    """One line of ``kb_root/index.jsonl``."""

    FILE_KIND: ClassVar[str] = "experiment"
    SCHEMA: ClassVar[str] = "cluster_bench.experiment/v1"

    gpu_profile: str
    gpu_name: str
    node_count: int
    gpus_per_node: int
    model: str
    checkpoint: str
    dataset: str
    served_context: int
    aiperf_version: str
    image: str
    started_at: float
    ended_at: float
    started_utc: str | None
    budget_hours: float
    spent_hours: float
    verdict: str
    leader_family: str | None
    leader_title: str | None
    leader_cand_key: str | None
    leader_point_id: str | None
    leader_tps_per_gpu: float | None
    leader_concurrency: int | None
    saturated: bool | None
    noise_floor_pct: float | None
    points_total: int
    points_usable: int
    points_invalid: int
    # Added in round 2; absent (None) in rows exported by older harnesses.
    shape_fp: str | None = None  # fingerprint without the node count
    provider: str | None = None
    fabric_id: str | None = None
    fabric_type: str | None = None
    link_rate: str | None = None
    # P-Only / D-Only (domain/stages.py): leader_tps_per_gpu is input
    # tok/s/GPU for P-Only. The leader's mean lengths size a PD split.
    stage: str = "e2e"
    platform: str | None = None
    leader_isl_avg: float | None = None
    leader_osl_avg: float | None = None
    price_per_node_hour: float | None = None  # USD
    cluster_per_hour: float | None = None
    experiment_usd: float | None = None
    leader_usd_per_mtok: float | None = None  # per 1e6 output tokens
    floor_usd_per_mtok: dict[str, float] = Field(default_factory=dict)
    fabric_kind: str | None = None
    rail_optimized: bool | None = None
    nodes_per_leaf: int | None = None
    max_hops: int | None = None
    leader_bottleneck: str | None = None
    leader_energy_per_token_j: float | None = None
    # Round 3: accuracy gate of the leader (GSM8K few-shot).
    leader_accuracy: float | None = None
    leader_accuracy_status: str | None = None  # pass|fail|error|skipped|unchecked
    accuracy_failed_families: tuple[str, ...] = ()
    # Round 3: opt-in diagnostic window (kernel shares per role, engine metrics).
    diag_nsys: str | None = None  # off | unavailable | ok | failed
    diag_kernel_shares: dict[str, dict[str, float]] = Field(default_factory=dict)
    diag_engine: dict[str, float] = Field(default_factory=dict)
    health_errors: int | None = None
    health_warnings: int | None = None
    # Round 4: focus resources of the leader (interconnect, MFU, KV capacity).
    link_target_util: float | None = None
    min_nics_measured: int | None = None  # by the NIC sweep, on min_nics_family
    min_nics_family: str | None = None  # may differ from leader_family
    min_nics_estimated: int | None = None  # the leader's own IB traffic
    nic_rate_gbs: float | None = None
    ib_p99_gbs_per_node: float | None = None
    ib_need_gbs_per_node: float | None = None
    nvlink_p99_gbs_per_gpu: float | None = None
    nvlink_needed: bool | None = None
    gpu_nic_pcie_p99_gbs: float | None = None
    leader_mfu_total: float | None = None
    leader_mfu_prefill: float | None = None
    leader_mfu_decode: float | None = None
    leader_tflops_per_gpu: float | None = None
    mfu_precision: str | None = None
    peak_tflops_per_gpu: float | None = None
    kv_total_tokens: int | None = None
    kv_max_request_tokens: int | None = None
    kv_context_p50: float | None = None
    kv_requests_at_p50: int | None = None
    kv_usage_peak: float | None = None
    kv_bytes_per_token: float | None = None
    # NVLink switched off (GPU-GPU over PCIe) vs on, same config and C.
    nvlink_off_status: str | None = None  # measured | invalid | skipped
    nvlink_off_reason: str | None = None
    nvlink_off_family: str | None = None
    nvlink_off_concurrency: int | None = None
    nvlink_off_method: str | None = None  # bw_mode_off | link_width_0 | nccl_no_p2p
    nvlink_off_path: str | None = None  # pcie | host
    nvlink_on_tps_per_gpu: float | None = None
    nvlink_off_tps_per_gpu: float | None = None
    nvlink_off_delta_pct: float | None = None
    nvlink_off_within_noise: bool | None = None
    nvlink_off_p2p_gbs: float | None = None  # GPU-GPU copy with NVLink off
    nvlink_off_pcie_p99_gbs: float | None = None


class FabricRecord(KbRecord):
    """Upstream fabric as one experiment saw it (one row per experiment)."""

    FILE_KIND: ClassVar[str] = "fabric"
    SCHEMA: ClassVar[str] = "cluster_bench.fabric/v1"

    kind: str
    nodes: int
    rails: int
    rail_rate_gbps: Any = None  # one rate, or the sorted list when mixed
    rail_optimized: bool | None = None
    rails_isolated: bool | None = None
    nodes_per_leaf: int | None = None
    leaf_shape: tuple[str, ...] = ()
    leaves: int = 0
    spines: int = 0
    tiers: int | None = None
    max_hops: int | None = None
    max_oversubscription: float | None = None
    degraded_links: int = 0
    pfc_priorities: tuple[int, ...] = ()
    tiers_seen: tuple[str, ...] = ()
    topology_hash: str | None = None
    leaf_names: tuple[str, ...] = ()  # switch descriptions (identity across rentals)
    link_bw_median_gbs: dict[str, float] = Field(default_factory=dict)  # rail -> GB/s
    link_lat_median_us: dict[str, float] = Field(default_factory=dict)
    link_findings: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()


class AccuracyRecord(KbRecord):
    """One accuracy gate check of one configuration (GSM8K few-shot)."""

    FILE_KIND: ClassVar[str] = "accuracy"
    SCHEMA: ClassVar[str] = "cluster_bench.accuracy/v1"

    family: str
    cand_key: str
    mtp: bool
    gpu_profile: str
    model: str
    checkpoint: str
    benchmark: str
    protocol: str
    status: str  # pass | fail | error | skipped
    code: str
    reason: str
    accuracy: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0
    errors: int = 0
    floor: float | None = None
    best_accuracy: float | None = None
    wall_s: float | None = None
    answer_sources: dict[str, int] = Field(default_factory=dict)


class HealthRecord(KbRecord):
    """One node-facts / health finding (degraded link, error growth, ...)."""

    FILE_KIND: ClassVar[str] = "health"
    SCHEMA: ClassVar[str] = "cluster_bench.health/v1"

    severity: str
    code: str
    node: str
    subject: str | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


CLUSTER_RECORD_TYPES: tuple[type[KbRecord], ...] = (
    ClusterRecord,
    RunRecord,
    PeakRecord,
    UtilRecord,
    NicSweepRecord,
    DeployRecord,
    ConfigRecord,
    FindingRecord,
    FabricRecord,
    HealthRecord,
    AccuracyRecord,
)
SCHEMA_TYPES: dict[str, type[KbRecord]] = {
    cls.SCHEMA: cls for cls in (*CLUSTER_RECORD_TYPES, ExperimentRecord)
}


@dataclass(frozen=True)
class RecordSet:
    """Every record of one experiment, grouped by file kind (output order)."""

    by_kind: dict[str, tuple[KbRecord, ...]] = field(default_factory=dict)
    experiment: ExperimentRecord | None = None

    def of(self, kind: str) -> tuple[KbRecord, ...]:
        return self.by_kind.get(kind, ())

    def counts(self) -> dict[str, int]:
        return {kind: len(rows) for kind, rows in self.by_kind.items()}


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _checkpoint(summary: ExperimentSummary) -> str:
    return f"{summary.checkpoint.repo}@{summary.checkpoint.revision}"


def _ordinal_keys(parts: list[str]) -> list[str]:
    """Append an occurrence index so repeated natural keys stay distinct."""
    seen: Counter[str] = Counter()
    keys = []
    for part in parts:
        keys.append(f"{part}:{seen[part]}")
        seen[part] += 1
    return keys


def _common(summary: ExperimentSummary, key: str) -> dict[str, Any]:
    return {"fp": summary.fingerprint, "exp_id": summary.exp_id, "key": key}


def run_record(summary: ExperimentSummary, point: PointResult) -> RunRecord:
    """The ``RunRecord`` of one point."""
    metrics = point.metrics
    cand = point.candidate
    titles = facts.family_titles(summary)
    return RunRecord.create(
        **_common(summary, f"run:{summary.exp_id}:{point.key.id}"),
        point_id=point.key.id,
        family=cand.family_id,
        family_title=titles.get(cand.family_id, cand.family_id),
        cand_key=point.key.cand_key,
        purpose=point.key.purpose,
        concurrency=point.key.concurrency,
        spec=point.key.spec,
        rep=point.key.rep,
        mtp=cand.mtp,
        simulate_acceptance=cand.simulate_acceptance,
        status=point.audit.status,
        audit_reasons=point.audit.reasons,
        gpu_count=point.gpu_count,
        tps_per_gpu=point.tps_per_gpu,
        output_tps=metrics.output_tps if metrics else None,
        user_tps_p50=metrics.user_tps_p50 if metrics else None,
        ttft_p50_ms=metrics.ttft_p50_ms if metrics else None,
        itl_p50_ms=metrics.itl_p50_ms if metrics else None,
        error_rate=_round(facts.error_rate(point)),
        cross_node_gbs=point.cross_node_gbs,
        nic_count=cand.nic_count,
        p2p=cand.p2p,
        nic_effective=point.nic_effective,
        nic_engaged=point.nic_engaged,
        staging_suspected=point.staging_suspected,
        accept_len=point.accept_len,
        gpu_profile=summary.gpu_profile,
        model=summary.model,
        checkpoint=_checkpoint(summary),
        dataset=summary.dataset,
        served_context=summary.served_context,
        artifact_dir=point.artifact_dir,
        stage=summary.stage,
        input_tps_per_gpu=point.input_tps_per_gpu,
        **_point_extras(point),
    )


def _serving_role(state: PointState) -> str | None:
    for role in ("decode", "agg", "prefill"):
        if role in state.roles:
            return role
    return None


def _point_extras(point: PointResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if point.mfu is not None:
        m = point.mfu
        out["mfu_total"] = _round(m.total_mfu, 5)
        out["achieved_tflops_per_gpu"] = _round(m.total_tflops_per_gpu, 3)
        out["mfu_precision"] = m.precision
        for role in ("prefill", "decode"):
            r = m.role(role)
            if r is not None:
                out[f"mfu_{role}"] = _round(r.mfu, 5)
    state = point.state
    if state is not None:
        role = _serving_role(state)
        if state.bottleneck is not None:
            out["bottleneck"] = state.bottleneck.label
            out["bottleneck_confidence"] = _round(state.bottleneck.confidence, 3)
        if role is not None:
            kv = state.value(role, "kv_usage") or state.value(role, "kv_usage_dynamo")
            out["kv_usage_avg"] = _round(kv, 4)
            out["queue_reqs_avg"] = _round(state.value(role, "queue_reqs"), 2)
            out["running_reqs_avg"] = _round(state.value(role, "running_reqs"), 2)
            out["prefix_hit_rate"] = _round(
                state.value(role, "cache_hit_rate")
                or state.value(role, "kv_hit_rate_dynamo"),
                4,
            )
        out["energy_per_token_j"] = _round(state.energy_per_token_j, 4)
        if state.stress is not None:
            out["power_mean_w"] = _round(state.stress.power_mean_w, 1)
            out["throttled_fraction"] = _round(state.stress.throttled_fraction, 4)
            out["ib_xmit_wait_fraction_max"] = _round(
                state.stress.xmit_wait_fraction_max, 5
            )
    if state is not None:
        out["decode_cuda_graph_fraction"] = _round(
            state.derived("decode_cuda_graph_fraction"), 4
        )
        role = _serving_role(state)
        if role is not None:
            out["spec_accept_rate"] = _round(state.value(role, "spec_accept_rate"), 4)
        serving = ("agg", "prefill", "decode")
        out["retracted_tokens_per_generated"] = _round(
            state.derived("retracted_tokens_per_generated", serving), 5
        )
        out["evicted_tokens_per_prompt"] = _round(
            state.derived("evicted_tokens_per_prompt", serving), 4
        )
        out["stage_prefill_forward_s"] = _round(
            state.value("prefill", "stage_prefill_forward"), 4
        )
        out["stage_kv_transfer_s"] = _round(
            state.value("prefill", "stage_prefill_transfer_kv_cache"), 4
        )
        out["stage_decode_wait_s"] = _round(
            state.value("decode", "stage_decode_waiting"), 4
        )
    activity = state.stress.activity if state and state.stress else None
    if activity is not None:
        serving = activity.roles.get("decode") or activity.roles.get("agg")
        if serving is not None:
            out["gpu_regime"] = serving.regime
            out["sm_pct_mean"] = serving.sm_pct_mean
            out["tensor_pct_mean"] = serving.tensor_pct_mean
            out["dram_pct_mean"] = serving.dram_pct_mean
        prefill = activity.roles.get("prefill")
        if prefill is not None:
            out["prefill_gpu_regime"] = prefill.regime
            out["prefill_tensor_pct_mean"] = prefill.tensor_pct_mean
        out["gpm_availability"] = dict(activity.availability)
    req = point.requests
    if req is not None:
        out["goodput_tps_per_gpu"] = {
            t.name: t.good_output_tps_per_gpu
            for t in req.tiers
            if t.good_output_tps_per_gpu is not None
        }
        out["good_fraction"] = {
            t.name: t.good_fraction for t in req.tiers if t.good_fraction is not None
        }
        out["ttft_first_turn_p50_s"] = req.first_turn_ttft_p50_s
        out["ttft_late_turn_p50_s"] = req.late_turn_ttft_p50_s
    load = point.switch_load
    if load is not None:
        out["switch_uplink_util_max"] = (
            _round(load.uplink.util_max, 4) if load.uplink else None
        )
        out["switch_downlink_util_max"] = (
            _round(load.downlink.util_max, 4) if load.downlink else None
        )
        out["switch_discards"] = load.discards
    return out


def _util_records(summary: ExperimentSummary) -> list[UtilRecord]:
    leader_id = summary.leader_point_id
    records: list[UtilRecord] = []
    for point in sorted(summary.points, key=lambda p: facts.point_sort_key(summary, p)):
        if not point.audit.usable or not point.bandwidth:
            continue
        stats: list[BandwidthStat] = facts.sort_stats(point.bandwidth)
        base = [
            f"util:{summary.exp_id}:{point.key.id}:{s.plane}:{s.scope}:{s.direction}"
            for s in stats
        ]
        for key, stat in zip(_ordinal_keys(base), stats):
            records.append(
                UtilRecord.create(
                    **_common(summary, key),
                    point_id=point.key.id,
                    family=point.candidate.family_id,
                    cand_key=point.key.cand_key,
                    purpose=point.key.purpose,
                    concurrency=point.key.concurrency,
                    leader=point.key.id == leader_id,
                    plane=stat.plane,
                    scope=stat.scope,
                    direction=stat.direction,
                    mean_gbs=stat.mean_gbs,
                    p99_gbs=stat.p99_gbs,
                    peak_gbs=stat.peak_gbs,
                    ref_peak_gbs=stat.ref_peak_gbs,
                    nominal_gbs=stat.nominal_gbs,
                    util_vs_peak=_round(stat.util_vs_peak),
                    util_vs_nominal=_round(stat.util_vs_nominal),
                    method=stat.method,
                )
            )
    return records


def _peak_records(summary: ExperimentSummary) -> list[PeakRecord]:
    base = [
        f"peak:{summary.exp_id}:{p.plane}:{p.test}:{p.scope}:{p.nic_count}"
        for p in summary.peaks
    ]
    return [
        PeakRecord.create(
            **_common(summary, key),
            plane=peak.plane,
            test=peak.test,
            scope=peak.scope,
            nic_count=peak.nic_count,
            value_gbs=peak.value_gbs,
            nominal_gbs=peak.nominal_gbs,
            fraction_of_nominal=_round(
                peak.value_gbs / peak.nominal_gbs if peak.nominal_gbs else None
            ),
            tool=peak.tool,
            command=peak.command,
            details=dict(peak.details),
            gpu_profile=summary.gpu_profile,
        )
        for key, peak in zip(_ordinal_keys(base), summary.peaks)
    ]


def _nic_records(
    summary: ExperimentSummary, leader_plan: LaunchPlan | None
) -> list[NicSweepRecord]:
    arms = facts.nic_arms_sorted(summary.nic_arms)
    base = [f"nic:{summary.exp_id}:{arm.nic_count}" for arm in arms]
    noise = summary.noise
    cross = facts.leader_cross_node(summary, leader_plan)
    return [
        NicSweepRecord.create(
            **_common(summary, key),
            family=summary.nic_family_id,
            concurrency=summary.nic_concurrency,
            nic_count=arm.nic_count,
            point_id=arm.point_id,
            tps_per_gpu=arm.tps_per_gpu,
            delta_vs_full_pct=_round(arm.delta_vs_full_pct),
            within_noise=arm.within_noise,
            noise_floor_pct=_round(noise.noise_floor_pct) if noise else None,
            nic_effective=arm.nic_effective,
            engaged=arm.engaged,
            staging_suspected=arm.staging_suspected,
            status=arm.status,
            leader_cross_node=cross,
            reason=summary.nic_reason,
            gpu_profile=summary.gpu_profile,
        )
        for key, arm in zip(_ordinal_keys(base), arms)
    ]


def _deploy_records(summary: ExperimentSummary) -> list[DeployRecord]:
    base = [f"deploy:{summary.exp_id}:{d.cand_fp}" for d in summary.deploys]
    attempts: Counter[str] = Counter()
    records = []
    for key, deploy in zip(_ordinal_keys(base), summary.deploys):
        cand = deploy.candidate
        engagement = deploy.engagement
        records.append(
            DeployRecord.create(
                **_common(summary, key),
                cand_fp=deploy.cand_fp,
                cand_key=cand.key,
                family=cand.family_id,
                mtp=cand.mtp,
                nic_count=cand.nic_count,
                simulate_acceptance=cand.simulate_acceptance,
                rungs=cand.rungs,
                attempt=attempts[deploy.cand_fp],
                ok=deploy.ok,
                load_seconds=deploy.load_seconds,
                failure=deploy.failure,
                detail=deploy.detail[:2000],
                engagement_ok=engagement.ok if engagement else None,
                engagement_checks=dict(engagement.checks) if engagement else {},
                log_dir=deploy.log_dir,
                **_lifecycle_extras(deploy.lifecycle),
            )
        )
        attempts[deploy.cand_fp] += 1
    return records


def _lifecycle_extras(life: DeployLifecycle | None) -> dict[str, Any]:
    if life is None:
        return {}
    roles = [
        r for name, r in life.roles.items() if name in ("agg", "prefill", "decode")
    ]

    def pick(values: list[Any], fn: Any) -> Any:
        present = [v for v in values if v is not None]
        return fn(present) if present else None

    return {
        "time_to_ready_s": _round(life.time_to_ready_s, 1),
        "weights_load_s": _round(pick([r.weights_load_s_max for r in roles], max), 1),
        "cuda_graph_s": _round(
            pick([r.cuda_graph_capture_s_max for r in roles], max), 1
        ),
        "kv_tokens_min": pick(
            [r.kv_tokens_min or r.max_total_num_tokens_min for r in roles], min
        ),
        "max_running_requests_min": pick(
            [r.max_running_requests_min for r in roles], min
        ),
        "log_issues": dict(life.issues),
    }


def config_record(summary: ExperimentSummary, plan: LaunchPlan) -> ConfigRecord:
    """The ``ConfigRecord`` of one rendered launch plan."""
    roles = tuple(
        RoleConfig(
            role=view.role,
            instances=view.instances,
            containers=view.containers,
            nodes=view.nodes,
            sglang=view.sglang,
            config_variants=view.config_variants,
            env=view.env,
            command=view.command,
        )
        for view in role_views(plan)
    )
    return ConfigRecord.create(
        **_common(summary, f"config:{summary.exp_id}:{plan.cand_fp}"),
        cand_fp=plan.cand_fp,
        cand_key=plan.candidate.key,
        family=plan.candidate.family_id,
        transport=plan.transport,
        cross_node=plan.cross_node,
        gpu_count=plan.gpu_count,
        nic_selection={
            node: tuple(h) for node, h in sorted(plan.nic_selection.items())
        },
        roles=roles,
    )


def _hca_rates(hcas) -> str:
    rates: Counter[str] = Counter()
    for hca in hcas:
        port = hca.active_ports[0]
        rates[f"{port.rate_gbps:g} Gb/s {port.link_layer}"] += 1
    return ", ".join(f"{n} x {rate}" for rate, n in sorted(rates.items()))


def cluster_record(summary: ExperimentSummary) -> ClusterRecord:
    """The hardware snapshot of the experiment's inventory."""
    nodes = summary.inventory.nodes
    node0 = nodes[0] if nodes else None
    gpu0 = node0.gpus[0] if node0 and node0.gpus else None
    snapshots = tuple(
        NodeSnapshot(
            name=node.name,
            gpus=len(node.gpus),
            gpu_names=tuple(sorted({g.name for g in node.gpus})),
            hcas=len(node.hcas),
            active_hcas=tuple(h.name for h in node.active_hcas),
            hca_rates=_hca_rates(node.active_hcas),
            driver_version=node.driver_version,
            nvlink_active=node.nvlink_active,
            nvidia_peermem=node.nvidia_peermem,
        )
        for node in nodes
    )
    return ClusterRecord.create(
        **_common(summary, f"cluster:{summary.exp_id}"),
        gpu_profile=summary.gpu_profile,
        gpu_name=summary.gpu_name,
        node_count=summary.node_count,
        gpus_per_node=summary.gpus_per_node,
        hbm_mib=gpu0.memory_mib if gpu0 else None,
        compute_cap=gpu0.compute_cap if gpu0 else None,
        nvlink_active=node0.nvlink_active if node0 else None,
        driver_version=node0.driver_version if node0 else None,
        cuda_version=node0.cuda_version if node0 else None,
        ofed_version=node0.ofed_version if node0 else None,
        nvidia_peermem=node0.nvidia_peermem if node0 else None,
        kernel=(node0.kernel or None) if node0 else None,
        docker_version=node0.docker_version if node0 else None,
        cpu_model=(node0.cpu_model or None) if node0 else None,
        cpu_count=(node0.cpu_count or None) if node0 else None,
        mem_total_bytes=(node0.mem_total_bytes or None) if node0 else None,
        nodes=snapshots,
        facts_signature=dict(summary.facts_signature),
        software=dict(summary.software.packages) if summary.software else {},
        shape_fp=summary.shape_fingerprint,
    )


def experiment_record(summary: ExperimentSummary) -> ExperimentRecord:
    """The ``index.jsonl`` row of one experiment."""
    lead = facts.leader_point(summary)
    cand = facts.leader_candidate(summary)
    counts = facts.audit_counts(summary.points)
    titles = facts.family_titles(summary)
    prov, cost = summary.provider, summary.cost
    return ExperimentRecord.create(
        **_common(summary, f"experiment:{summary.exp_id}"),
        gpu_profile=summary.gpu_profile,
        gpu_name=summary.gpu_name,
        node_count=summary.node_count,
        gpus_per_node=summary.gpus_per_node,
        model=summary.model,
        checkpoint=_checkpoint(summary),
        dataset=summary.dataset,
        served_context=summary.served_context,
        aiperf_version=summary.aiperf_version,
        image=summary.image,
        started_at=summary.started_at,
        ended_at=summary.ended_at,
        started_utc=iso_utc(summary.started_at),
        budget_hours=summary.budget_hours,
        spent_hours=summary.spent_hours,
        verdict=summary.verdict,
        leader_family=cand.family_id if cand else None,
        leader_title=titles.get(cand.family_id, cand.family_id) if cand else None,
        leader_cand_key=cand.key if cand else None,
        leader_point_id=lead.key.id if lead else None,
        leader_tps_per_gpu=lead.tps_per_gpu if lead else None,
        leader_concurrency=lead.key.concurrency if lead else None,
        saturated=summary.saturated,
        noise_floor_pct=_round(summary.noise.noise_floor_pct)
        if summary.noise
        else None,
        points_total=len(summary.points),
        points_usable=counts["valid"] + counts["valid_with_recovery"],
        points_invalid=counts["invalid"],
        shape_fp=summary.shape_fingerprint,
        stage=summary.stage,
        platform=summary.platform,
        leader_isl_avg=_round(lead.metrics.isl_avg, 1)
        if lead and lead.metrics
        else None,
        leader_osl_avg=_round(lead.metrics.osl_avg, 1)
        if lead and lead.metrics
        else None,
        provider=prov.provider if prov else None,
        fabric_id=prov.fabric_id if prov else None,
        fabric_type=prov.fabric_type if prov else None,
        link_rate=prov.link_rate if prov else None,
        price_per_node_hour=cost.price_per_node_hour if cost else None,
        cluster_per_hour=cost.cluster_per_hour if cost else None,
        experiment_usd=cost.experiment_usd if cost else None,
        leader_usd_per_mtok=cost.leader_usd_per_mtok if cost else None,
        floor_usd_per_mtok=dict(cost.floor_usd_per_mtok) if cost else {},
        fabric_kind=summary.fabric.kind if summary.fabric else None,
        rail_optimized=summary.fabric.rail_optimized if summary.fabric else None,
        nodes_per_leaf=summary.fabric.nodes_per_leaf if summary.fabric else None,
        max_hops=summary.fabric.max_hops if summary.fabric else None,
        leader_bottleneck=(
            lead.state.bottleneck.label
            if lead and lead.state and lead.state.bottleneck
            else None
        ),
        leader_energy_per_token_j=_round(lead.state.energy_per_token_j, 4)
        if lead and lead.state
        else None,
        health_errors=_count(summary, "error"),
        health_warnings=_count(summary, "warn"),
        **_leader_accuracy(summary, cand),
        **_diagnostics_extras(summary),
        **_resources_extras(summary, lead),
    )


def _resources_extras(
    summary: ExperimentSummary, lead: PointResult | None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    need = summary.interconnect
    if need is not None:
        ib, nvlink, nic = need.link("ib"), need.link("nvlink"), need.link("pcie_nic")
        out.update(
            link_target_util=need.target_util,
            min_nics_measured=need.min_nics_measured,
            min_nics_family=need.sweep_family_id,
            min_nics_estimated=need.min_nics_estimated,
            nic_rate_gbs=need.nic_rate_gbs,
            ib_p99_gbs_per_node=ib.p99_gbs if ib else None,
            ib_need_gbs_per_node=ib.need_gbs if ib else None,
            nvlink_p99_gbs_per_gpu=nvlink.p99_gbs if nvlink else None,
            nvlink_needed=need.nvlink_needed,
            gpu_nic_pcie_p99_gbs=nic.p99_gbs if nic else None,
        )
    if lead is not None and lead.mfu is not None:
        m = lead.mfu
        out.update(
            leader_mfu_total=_round(m.total_mfu, 5),
            leader_mfu_prefill=_round(m.role("prefill").mfu, 5)
            if m.role("prefill")
            else None,
            leader_mfu_decode=_round(m.role("decode").mfu, 5)
            if m.role("decode")
            else None,
            leader_tflops_per_gpu=_round(m.total_tflops_per_gpu, 3),
            mfu_precision=m.precision,
            peak_tflops_per_gpu=m.peak_tflops_per_gpu,
        )
    kv = summary.kv_capacity
    if kv is not None:
        main = kv.role(kv.serving_role)
        out.update(
            kv_total_tokens=kv.total_tokens,
            kv_max_request_tokens=kv.max_request_tokens,
            kv_context_p50=kv.context_p50,
            kv_requests_at_p50=kv.requests_at_p50,
            kv_usage_peak=_round(kv.usage_peak, 4),
            kv_bytes_per_token=main.bytes_per_token if main else None,
        )
    cmp = summary.nvlink
    if cmp is not None:
        out.update(
            nvlink_off_status=cmp.status,
            nvlink_off_reason=cmp.reason or None,
            nvlink_off_family=cmp.family_id,
            nvlink_off_concurrency=cmp.concurrency,
            nvlink_off_method=cmp.method,
            nvlink_off_path=cmp.path,
            nvlink_on_tps_per_gpu=_round(cmp.on_tps_per_gpu, 3),
            nvlink_off_tps_per_gpu=_round(cmp.off_tps_per_gpu, 3),
            nvlink_off_delta_pct=cmp.delta_pct,
            nvlink_off_within_noise=cmp.within_noise,
            nvlink_off_p2p_gbs=cmp.p2p_gbs,
            nvlink_off_pcie_p99_gbs=_round(cmp.pcie_p99_gbs, 3),
        )
    return out


def _diagnostics_extras(summary: ExperimentSummary) -> dict[str, Any]:
    diag = summary.diagnostics
    if diag is None:
        return {}
    shares: dict[str, dict[str, float]] = {}
    for role, raw in (diag.kernels or {}).items():
        try:
            breakdown = KernelBreakdown.model_validate(raw)
        except ValueError:
            continue
        shares[role] = {c.category: round(c.share, 4) for c in breakdown.categories}
    return {
        "diag_nsys": diag.nsys,
        "diag_kernel_shares": shares,
        "diag_engine": {k: round(v, 4) for k, v in diag.engine.items()},
    }


def _leader_accuracy(summary: ExperimentSummary, cand: Any) -> dict[str, Any]:
    check = summary.accuracy_of(cand.key) if cand is not None else None
    if check is None:
        return {"leader_accuracy_status": "unchecked" if cand is not None else None}
    return {
        "leader_accuracy": _round(check.result.accuracy, 4),
        "leader_accuracy_status": check.verdict.status,
        "accuracy_failed_families": tuple(
            sorted(
                {c.family_id for c in summary.accuracy if c.verdict.status == "fail"}
            )
        ),
    }


def _count(summary: ExperimentSummary, severity: str) -> int | None:
    if not summary.facts_signature and not summary.fact_findings:
        return None
    return sum(1 for f in summary.fact_findings if f.severity == severity)


def fabric_records(summary: ExperimentSummary) -> list[FabricRecord]:
    topo = summary.fabric
    if topo is None:
        return []
    sig = summary.fabric_signature
    bw: dict[str, float] = {}
    lat: dict[str, float] = {}
    links = summary.fabric_links
    if links is not None:
        for rail in links.rails:
            tests = [t for t in links.tests if t.rail == rail and not t.error]
            bws = sorted(t.bw_gbs for t in tests if t.bw_gbs is not None)
            lats = sorted(
                t.lat_typical_us for t in tests if t.lat_typical_us is not None
            )
            if bws:
                bw[str(rail)] = round(bws[len(bws) // 2], 3)
            if lats:
                lat[str(rail)] = round(lats[len(lats) // 2], 3)
    leaves = sorted(
        {sw.desc or sid for sid, sw in topo.switches.items() if sw.role == "leaf"}
    )
    return [
        FabricRecord.create(
            **_common(summary, f"fabric:{summary.exp_id}"),
            kind=topo.kind,
            nodes=len(topo.nodes),
            rails=len(topo.rails),
            rail_rate_gbps=sig.get("rail_rate_gbps"),
            rail_optimized=topo.rail_optimized,
            rails_isolated=topo.rails_isolated,
            nodes_per_leaf=topo.nodes_per_leaf,
            leaf_shape=tuple(sig.get("leaf_shape") or ()),
            leaves=int(sig.get("leaves") or 0),
            spines=int(sig.get("spines") or 0),
            tiers=topo.tiers,
            max_hops=topo.max_hops,
            max_oversubscription=sig.get("max_oversubscription"),
            degraded_links=len(topo.degraded),
            pfc_priorities=tuple(sig.get("pfc_priorities") or ()),
            tiers_seen=tuple(sig.get("tiers_seen") or ()),
            topology_hash=sig.get("hash"),
            leaf_names=tuple(leaves),
            link_bw_median_gbs=bw,
            link_lat_median_us=lat,
            link_findings=tuple(f.detail for f in summary.link_findings),
            findings=tuple(summary.fabric_findings),
        )
    ]


def accuracy_records(summary: ExperimentSummary) -> list[AccuracyRecord]:
    out = []
    for i, check in enumerate(summary.accuracy):
        res, ver = check.result, check.verdict
        out.append(
            AccuracyRecord.create(
                **_common(summary, f"accuracy:{summary.exp_id}:{i}:{check.cand_key}"),
                family=check.family_id,
                cand_key=check.cand_key,
                mtp="|mtp=0|" not in check.cand_key,
                gpu_profile=summary.gpu_profile,
                model=summary.model,
                checkpoint=summary.checkpoint.repo,
                benchmark=res.benchmark,
                protocol=res.protocol,
                status=ver.status,
                code=ver.code,
                reason=ver.reason,
                accuracy=_round(res.accuracy, 4),
                ci_low=_round(res.ci_low, 4),
                ci_high=_round(res.ci_high, 4),
                n=res.n,
                errors=res.errors,
                floor=ver.floor,
                best_accuracy=_round(ver.best_accuracy, 4),
                wall_s=_round(res.wall_s, 1),
                answer_sources=dict(res.answer_sources),
            )
        )
    return out


def health_records(summary: ExperimentSummary) -> list[HealthRecord]:
    seen: set[str] = set()
    out = []
    for finding in summary.fact_findings:
        key = f"health:{summary.exp_id}:{finding.key}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            HealthRecord.create(
                **_common(summary, key),
                severity=finding.severity,
                code=finding.code,
                node=finding.node,
                subject=finding.subject,
                message=finding.message,
                data=dict(finding.data),
            )
        )
    return out


def build_records(
    summary: ExperimentSummary,
    plan_lookup: PlanLookup | None = None,
    findings: tuple[FindingRecord, ...] = (),
) -> RecordSet:
    """All records of one experiment (``findings`` come from ``kb.findings``)."""
    leader = facts.leader_candidate(summary)
    leader_plan = plan_lookup(leader.fp) if plan_lookup and leader else None
    configs: list[ConfigRecord] = []
    if plan_lookup is not None:
        seen: set[str] = set()
        for deploy in summary.deploys:
            if not deploy.ok or deploy.cand_fp in seen:
                continue
            seen.add(deploy.cand_fp)
            plan = plan_lookup(deploy.cand_fp)
            if plan is not None:
                configs.append(config_record(summary, plan))
    points = sorted(summary.points, key=lambda p: facts.point_sort_key(summary, p))
    by_kind: dict[str, tuple[KbRecord, ...]] = {
        ClusterRecord.FILE_KIND: (cluster_record(summary),),
        RunRecord.FILE_KIND: tuple(run_record(summary, p) for p in points),
        PeakRecord.FILE_KIND: tuple(_peak_records(summary)),
        UtilRecord.FILE_KIND: tuple(_util_records(summary)),
        NicSweepRecord.FILE_KIND: tuple(_nic_records(summary, leader_plan)),
        DeployRecord.FILE_KIND: tuple(_deploy_records(summary)),
        ConfigRecord.FILE_KIND: tuple(configs),
        FindingRecord.FILE_KIND: tuple(findings),
        FabricRecord.FILE_KIND: tuple(fabric_records(summary)),
        HealthRecord.FILE_KIND: tuple(health_records(summary)),
        AccuracyRecord.FILE_KIND: tuple(accuracy_records(summary)),
    }
    return RecordSet(by_kind=by_kind, experiment=experiment_record(summary))
