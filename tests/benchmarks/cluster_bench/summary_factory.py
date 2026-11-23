# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic but realistic ``ExperimentSummary`` fixtures (report + KB tests).

``h200_summary`` models a full standard run on 4 x 8 H200 with GLM-5.3 FP8:
three measured topology families (per-node aggregated TP8 wins, cross-node
disagg 1P1D, wide-EP DEP16 pruned) plus one OOM deploy failure, an MTP
on/off pair, an n=3 noise pilot, a confirmatory rerun, a 8/4/2/1 NIC sweep on
the cross-node family (the winner has no cross-node traffic), speed-of-light
peaks and under-load bandwidth statistics. Numbers are plausible for the
hardware; they are not measurements.

``rtx6000_unsupported_summary`` models a run whose preflight failed: no
points, no leader, verdict ``unsupported``.
"""

from __future__ import annotations

import math
import statistics

from benchmarks.cluster_bench.domain import nvlink as nvl
from benchmarks.cluster_bench.domain.capacity import interconnect_needs, kv_capacity
from benchmarks.cluster_bench.domain.lifecycle import DeployLifecycle, RoleLifecycle
from benchmarks.cluster_bench.domain.modelarch import (
    SIMULATED_CONFIG,
    MfuResult,
    arch_from_config,
    compute_mfu,
)
from benchmarks.cluster_bench.domain.pointstate import PointState, RoleBrief
from benchmarks.cluster_bench.domain.requests import RequestSummary
from benchmarks.cluster_bench.domain.schema import (
    AuditResult,
    BandwidthStat,
    Candidate,
    Checkpoint,
    ContainerSpec,
    DeployResult,
    EngagementResult,
    ExperimentSummary,
    FamilySummary,
    FloorResult,
    GpuInfo,
    GpuProfile,
    HcaInfo,
    HcaPort,
    Inventory,
    LaunchPlan,
    MtpComparison,
    NicArm,
    NodeInventory,
    NoiseResult,
    PeakResult,
    PointKey,
    PointMetrics,
    PointResult,
)

H200_FP = "h200-8x4-ndr400x8-5e1c2a7b"
RTX_FP = "rtxpro6000-8x4-roce200x4-91d0e3aa"
IMAGE = (
    "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1@sha256:"
    "37badffd83eb334aee9d06fb442e36fc3c963314336c92622c78ac5c037d514d"
)
DATASET = "semianalysis_cc_traces_weka_062126"
T0 = 1_793_520_000.0  # 2026-11-01 08:00 UTC

AGG = Candidate(family_id="agg-tp8")
AGG_OFF = Candidate(family_id="agg-tp8", mtp=False)
AGG_PCIE = Candidate(family_id="agg-tp8", p2p="pcie")
NVLINK_SWITCH = {
    "action": "off",
    "method": "bw_mode_off",
    "path": "pcie",
    "p2p": {"median_gbs": 41.5, "peer_pairs": 56},
}
DIS = Candidate(family_id="disagg-1p1d")
WEP = Candidate(family_id="wideep-dep16")
DIS2 = Candidate(family_id="disagg-2p1d")

LEADER_TPS = 112.4
ARCH = arch_from_config(SIMULATED_CONFIG, source="config.json")
H200_PROFILE = GpuProfile(
    name="h200",
    match=("h200",),
    arch="sm90",
    hbm_gb=141.0,
    nvlink=True,
    weights_variant="fp8",
    nominal_nvlink_gbs=450.0,
    nominal_pcie_gbs=64.0,
    peak_dense_tflops={"fp8": 1979.0, "bf16": 989.0},
    hbm_tbs=4.8,
)
CACHED_FRACTION = 0.82  # prompt tokens served from the prefix cache
SECRET_ENV_VALUE = "hf_dummy_secret_value_for_mask_test"


def dis_nic(count: int) -> Candidate:
    return Candidate(family_id="disagg-1p1d", nic_count=count)


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def _hca(index: int, rate: float, layer: str, numa: int) -> HcaInfo:
    rate_text = f"{int(rate)} Gb/sec (4X {'NDR' if rate >= 400 else 'HDR'})"
    return HcaInfo(
        name=f"mlx5_{index}",
        pci_bus_id=f"0000:{0x18 + index * 0x10:02x}:00.0",
        numa_node=numa,
        netdev=f"ibp{index}s0" if layer == "InfiniBand" else f"enp{index}s0np0",
        fw_ver="28.42.1000",
        ports=(
            HcaPort(
                port=1,
                state="4: ACTIVE",
                rate=rate_text,
                rate_gbps=rate,
                link_layer=layer,
            ),
        ),
    )


def _node(
    index: int,
    *,
    gpu_name: str,
    memory_mib: int,
    compute_cap: str,
    hcas: int,
    rate: float,
    layer: str,
    nvlink: bool,
) -> NodeInventory:
    gpus = tuple(
        GpuInfo(
            index=g,
            name=gpu_name,
            uuid=f"GPU-{index:02d}{g:02d}a1b2-c3d4-e5f6-0718-293a4b5c6d7e",
            pci_bus_id=f"0000:{0x19 + g * 0x10:02x}:00.0",
            memory_mib=memory_mib,
            numa_node=0 if g < 4 else 1,
            compute_cap=compute_cap,
            pcie_gen_max=5,
            pcie_width_max=16,
        )
        for g in range(8)
    )
    return NodeInventory(
        name=f"node{index}",
        hostname=f"lium-{index:02d}",
        address=f"10.42.0.{index + 1}",
        gpus=gpus,
        hcas=tuple(
            _hca(i, rate, layer, 0 if i < hcas // 2 else 1) for i in range(hcas)
        ),
        driver_version="580.95.05",
        cuda_version="13.0",
        cpu_model="Intel(R) Xeon(R) Platinum 8480+",
        cpu_count=224,
        mem_total_bytes=2 * (1 << 40),
        docker_version="27.5.1",
        kernel="5.15.0-140-generic",
        ofed_version="MLNX_OFED_LINUX-24.10-1.1.4.0",
        nvidia_peermem=True,
        nvlink_active=nvlink,
    )


def h200_inventory() -> Inventory:
    return Inventory(
        nodes=tuple(
            _node(
                i,
                gpu_name="NVIDIA H200",
                memory_mib=143771,
                compute_cap="9.0",
                hcas=8,
                rate=400.0,
                layer="InfiniBand",
                nvlink=True,
            )
            for i in range(4)
        )
    )


def rtx_inventory() -> Inventory:
    return Inventory(
        nodes=tuple(
            _node(
                i,
                gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                memory_mib=97887,
                compute_cap="12.0",
                hcas=4,
                rate=200.0,
                layer="Ethernet",
                nvlink=False,
            )
            for i in range(4)
        )
    )


# --------------------------------------------------------------------------
# Points
# --------------------------------------------------------------------------


def _stat(plane, scope, direction, mean, p99, peak, ref, nominal, method="measured"):
    return BandwidthStat(
        plane=plane,
        scope=scope,
        direction=direction,
        mean_gbs=mean,
        p99_gbs=p99,
        peak_gbs=peak,
        ref_peak_gbs=ref,
        nominal_gbs=nominal,
        method=method,
    )


def agg_bandwidth(scale: float = 1.0) -> tuple[BandwidthStat, ...]:
    """Per-node replicas: busy NVLink, idle IB."""
    return (
        _stat("ib", "cluster", "tx", 0.01, 0.02, 0.03, 48.3, 50.0),
        _stat("ib", "node0", "tx", 0.01, 0.02, 0.03, 386.4, 400.0),
        _stat("ib", "node0/mlx5_0", "rx", 0.0, 0.01, 0.02, 48.3, 50.0),
        _stat("nvlink", "node0/gpu0", "tx", 38.2 * scale, 96.4, 131.7, 368.2, 450.0),
        _stat("nvlink", "node1/gpu3", "tx", 41.0 * scale, 101.2, 140.9, 368.2, 450.0),
        _stat("nvlink", "node1/gpu3", "rx", 40.1 * scale, 99.0, 138.2, 368.2, 450.0),
        _stat("pcie", "node0/gpu0", "tx", 1.3, 4.1, 6.2, 53.1, 64.0),
        _stat("pcie", "node0/gpu0", "rx", 2.2, 7.9, 11.4, 53.1, 64.0),
        _stat("pcie_nic", "node0", "tx", 0.01, 0.02, 0.03, 47.6, 50.0, "derived"),
        _stat(
            "pcie_nic",
            "node0/gpu0-mlx5_0",
            "bidir",
            0.0,
            0.0,
            0.0,
            None,
            None,
            "unavailable",
        ),
    )


def nvlink_off_bandwidth() -> tuple[BandwidthStat, ...]:
    """NVLink switched off: zero NVLink, the all-reduce saturating PCIe."""
    return (
        _stat("ib", "node0", "tx", 0.01, 0.02, 0.03, 386.4, 400.0),
        _stat("nvlink", "node1/gpu3", "tx", 0.0, 0.0, 0.0, 368.2, 450.0),
        _stat("pcie", "node1/gpu3", "tx", 31.0, 47.8, 55.1, 53.1, 64.0),
        _stat("pcie", "node1/gpu3", "rx", 30.2, 46.9, 54.0, 53.1, 64.0),
    )


def disagg_bandwidth(load: float) -> tuple[BandwidthStat, ...]:
    """Prefill and decode on different nodes: KV moves over IB."""
    return (
        _stat("ib", "node0", "tx", 6.4 * load, 19.8 * load, 27.9 * load, 386.4, 400.0),
        _stat("ib", "node2", "rx", 6.1 * load, 18.9 * load, 26.6 * load, 386.4, 400.0),
        _stat(
            "ib", "node0/mlx5_0", "tx", 6.4 * load, 19.8 * load, 27.9 * load, 48.3, 50.0
        ),
        _stat(
            "ib", "node2/mlx5_3", "rx", 6.1 * load, 18.9 * load, 26.6 * load, 48.3, 50.0
        ),
        _stat("nvlink", "node2/gpu1", "tx", 22.5, 61.0, 88.3, 368.2, 450.0),
        _stat(
            "pcie", "node0/gpu0", "tx", 6.6 * load, 20.1 * load, 28.4 * load, 53.1, 64.0
        ),
        _stat(
            "pcie_nic",
            "node0/gpu0-mlx5_0",
            "tx",
            6.3 * load,
            19.6 * load,
            27.5 * load,
            47.6,
            50.0,
        ),
    )


def point_mfu(
    point: PointResult, tensor: float | None = None, dram: float | None = None
) -> MfuResult | None:
    """MFU and MBU of an aggregated point (as ``point_mfu``): requests and GPM."""
    m = point.metrics
    if m is None or not m.duration_s or not m.request_count:
        return None
    isl, osl = int(m.isl_avg or 0), int(m.osl_avg or 0)
    cached = int(isl * CACHED_FRACTION)
    n = m.request_count
    return compute_mfu(
        flops_prefill=n * ARCH.tokens_flops(cached, isl - cached),
        flops_decode=n * ARCH.tokens_flops(isl, osl),
        window_s=m.duration_s,
        role_gpus={"agg": point.gpu_count},
        precision="fp8",
        peak_tflops=H200_PROFILE.peak_dense_tflops["fp8"],
        tensor_active={"agg": tensor} if tensor is not None else {},
        dram_active={"agg": dram} if dram is not None else {},
        peak_mem_tbs=H200_PROFILE.hbm_tbs,
        cache_source="usage",
    )


def make_point(
    cand: Candidate,
    concurrency: int,
    tps_per_gpu: float,
    user_tps: float,
    *,
    spec: str = "screen",
    purpose: str = "search",
    rep: int = 0,
    gpu_count: int = 32,
    ttft_ms: float = 900.0,
    itl_ms: float = 15.0,
    cross_node_gbs: float | None = None,
    bandwidth: tuple[BandwidthStat, ...] = (),
    status: str = "valid",
    reasons: tuple[str, ...] = (),
    errors: int = 0,
    accept_len: float | None = None,
    nic_effective: float | None = None,
    nic_engaged: bool | None = None,
    staging: bool = False,
    started_at: float = T0,
) -> PointResult:
    key = PointKey(
        cand_key=cand.key, concurrency=concurrency, spec=spec, purpose=purpose, rep=rep
    )
    duration = 1200.0 if spec == "decision" else 480.0
    output_tps = tps_per_gpu * gpu_count
    requests = max(int(output_tps * duration / 900.0), 1)
    metrics = PointMetrics(
        aiperf_version="0.14.0",
        request_count=requests,
        error_count=errors,
        output_tps=output_tps,
        total_output_tokens=int(output_tps * duration),
        user_tps_p50=user_tps,
        ttft_p50_ms=ttft_ms,
        ttft_p99_ms=ttft_ms * 3.1,
        itl_p50_ms=itl_ms,
        itl_p99_ms=itl_ms * 2.4,
        request_throughput=requests / duration,
        duration_s=duration,
        isl_avg=61234.0,
        osl_avg=912.0,
    )
    return PointResult(
        key=key,
        candidate=cand,
        gpu_count=gpu_count,
        metrics=metrics,
        audit=AuditResult(status=status, reasons=reasons),
        tps_per_gpu=tps_per_gpu,
        bandwidth=bandwidth,
        cross_node_gbs=cross_node_gbs,
        nic_effective=nic_effective,
        nic_engaged=nic_engaged,
        staging_suspected=staging,
        accept_len=accept_len,
        artifact_dir=f"points/{key.id}",
        started_at=started_at,
        ended_at=started_at + duration,
    )


AGG_CURVE = (  # C, tok/s/GPU, tok/s/user, TTFT ms, ITL ms
    (8, 21.3, 86.0, 820.0, 11.2),
    (16, 39.0, 79.3, 870.0, 12.1),
    (32, 64.2, 64.0, 1010.0, 14.9),
    (64, 92.8, 46.2, 1350.0, 20.6),
    (128, 110.3, 27.6, 2400.0, 34.3),
    (256, LEADER_TPS, 14.0, 5200.0, 68.9),
    (512, 111.1, 6.9, 11800.0, 141.0),
)
DIS_CURVE = (  # C, tok/s/GPU, tok/s/user, cross-node GB/s
    (16, 30.9, 88.4, 1.9),
    (32, 57.8, 74.6, 3.4),
    (64, 88.1, 55.3, 5.6),
    (128, 101.6, 34.1, 7.9),
    (256, 104.9, 17.3, 9.1),
    (512, 103.8, 8.5, 9.2),
)
WEP_CURVE = ((64, 41.0, 21.0, 14.2), (128, 58.2, 15.2, 19.8), (256, 66.4, 8.9, 22.5))
PILOT_SAMPLES = (111.9, 113.6, 112.2)
NIC_ARMS = (  # count, tok/s/GPU, effective NICs, staging suspected
    (8, 105.6, 7.91, False),
    (4, 104.6, 3.97, False),
    (2, 97.9, 1.99, False),
    (1, 61.8, 1.0, True),
)


def _noise(samples: tuple[float, ...]) -> NoiseResult:
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    floor = 2.0 * stdev * math.sqrt(1.0 + 1.0 / len(samples))
    return NoiseResult(
        cand_key=AGG.key,
        concurrency=256,
        samples=samples,
        mean=mean,
        stdev=stdev,
        cv=stdev / mean,
        noise_floor=floor,
        noise_floor_pct=floor / mean * 100.0,
    )


def h200_points(scale: float = 1.0) -> dict[str, PointResult]:
    """Named points of the H200 run (``scale`` multiplies every throughput)."""
    pts: dict[str, PointResult] = {}
    for c, tps, user, ttft, itl in AGG_CURVE:
        pts[f"agg-{c}"] = make_point(
            AGG,
            c,
            round(tps * scale, 2),
            user,
            ttft_ms=ttft,
            itl_ms=itl,
            cross_node_gbs=0.01,
            bandwidth=agg_bandwidth(scale) if c >= 128 else (),
            accept_len=2.41,
        )
        point = pts[f"agg-{c}"]
        tensor = round(0.08 + 0.0009 * c, 3) if c >= 128 else None
        dram = round(0.35 + 0.0012 * c, 3) if c >= 128 else None
        mfu = point_mfu(point, tensor, dram)
        pts[f"agg-{c}"] = point.model_copy(update={"mfu": mfu})
    for c, tps, user, cross in DIS_CURVE:
        status = "valid_with_recovery" if c == 512 else "valid"
        reasons = ("worker restart recovered; 2 requests retried",) if c == 512 else ()
        pts[f"dis-{c}"] = make_point(
            DIS,
            c,
            round(tps * scale, 2),
            user,
            cross_node_gbs=cross,
            bandwidth=disagg_bandwidth(c / 256.0) if c >= 128 else (),
            status=status,
            reasons=reasons,
            accept_len=2.38,
        )
    for c, tps, user, cross in WEP_CURVE:
        pts[f"wep-{c}"] = make_point(
            WEP, c, round(tps * scale, 2), user, cross_node_gbs=cross, accept_len=2.36
        )
    pts["wep-512"] = make_point(
        WEP,
        512,
        52.0,
        4.1,
        cross_node_gbs=23.0,
        status="invalid",
        reasons=("error rate 3.2% > 1%", "run cancelled by aiperf"),
        errors=31,
    )
    pts["agg-off-256"] = make_point(
        AGG_OFF, 256, round(95.1 * scale, 2), 11.8, purpose="mtp_ablation"
    )
    pts["agg-off-64"] = make_point(
        AGG_OFF, 64, round(80.2 * scale, 2), 39.9, purpose="mtp_ablation"
    )
    for rep, sample in enumerate(PILOT_SAMPLES):
        pts[f"pilot-{rep}"] = make_point(
            AGG,
            256,
            round(sample * scale, 2),
            14.1,
            spec="decision",
            purpose="pilot",
            rep=rep,
            accept_len=2.42,
        )
    pts["confirm"] = make_point(
        AGG, 256, round(112.9 * scale, 2), 14.0, spec="decision", purpose="confirm"
    )
    # The leader again with NVLink switched off: TP8 all-reduce on PCIe.
    pts["nvlink-off"] = make_point(
        AGG_PCIE,
        256,
        round(74.6 * scale, 2),
        9.2,
        purpose="nvlink",
        ttft_ms=7900.0,
        itl_ms=104.0,
        cross_node_gbs=0.01,
        bandwidth=nvlink_off_bandwidth(),
    )
    for count, tps, eff, staging in NIC_ARMS:
        cand = DIS if count == 8 else dis_nic(count)
        pts[f"nic-{count}"] = make_point(
            cand,
            256,
            round(tps * scale, 2),
            17.4,
            spec="decision",
            purpose="nic",
            cross_node_gbs=9.1 * tps / 105.6,
            bandwidth=disagg_bandwidth(tps / 105.6),
            nic_effective=eff,
            nic_engaged=True,
            staging=staging,
        )
    return pts


def _deploy(cand: Candidate, ok: bool = True, **kwargs) -> DeployResult:
    engagement = None
    if ok and cand.nic_count is not None:
        engagement = EngagementResult(
            ok=True,
            checks={"hca_counters": True, "proc_environ": True},
            evidence=(f"only {cand.nic_count} HCAs carried traffic",),
        )
    return DeployResult(
        candidate=cand,
        cand_fp=cand.fp,
        ok=ok,
        engagement=engagement,
        log_dir=f"deployments/{cand.fp}/logs",
        **kwargs,
    )


def h200_peaks() -> tuple[PeakResult, ...]:
    return (
        PeakResult(
            plane="ib",
            test="gdr_write_bw",
            scope="node0/mlx5_0",
            nic_count=1,
            value_gbs=48.3,
            nominal_gbs=50.0,
            tool="ib_write_bw --use_cuda",
            command="ib_write_bw -d mlx5_0 --use_cuda=0 -s 1048576 --report_gbits",
        ),
        *(
            PeakResult(
                plane="ib",
                test="nccl_all_reduce_inter",
                scope="cluster",
                nic_count=n,
                value_gbs=value,
                nominal_gbs=50.0 * n,
                tool="nccl_bench",
            )
            for n, value in ((8, 362.5), (4, 183.0), (2, 92.4), (1, 46.8))
        ),
        PeakResult(
            plane="nvlink",
            test="p2p_copy",
            scope="node0",
            value_gbs=368.2,
            nominal_gbs=450.0,
            tool="p2p_bench",
        ),
        PeakResult(
            plane="nvlink",
            test="nccl_all_reduce_intra",
            scope="node0",
            value_gbs=471.9,
            tool="nccl_bench",
        ),
        PeakResult(
            plane="pcie",
            test="h2d_copy",
            scope="node0/gpu0",
            value_gbs=53.1,
            nominal_gbs=64.0,
            tool="p2p_bench",
        ),
        PeakResult(
            plane="pcie_nic",
            test="gdr_read_bw",
            scope="node0/gpu0-mlx5_0",
            value_gbs=47.6,
            nominal_gbs=50.0,
            tool="ib_read_bw --use_cuda",
        ),
    )


def h200_summary(
    exp_id: str = "exp-h200-001", scale: float = 1.0, started_at: float = T0
) -> ExperimentSummary:
    """A complete ``ok`` H200 experiment (see module docstring)."""
    pts = h200_points(scale)
    ids = {name: p.key.id for name, p in pts.items()}
    lead = pts["agg-256"]
    noise = _noise(tuple(round(s * scale, 2) for s in PILOT_SAMPLES))
    full_tps = pts["nic-8"].tps_per_gpu or 1.0
    arms = []
    for count, _, eff, staging in NIC_ARMS:
        point = pts[f"nic-{count}"]
        delta = ((point.tps_per_gpu or 0.0) - full_tps) / full_tps * 100.0
        arms.append(
            NicArm(
                nic_count=count,
                point_id=point.key.id,
                tps_per_gpu=point.tps_per_gpu,
                delta_vs_full_pct=delta,
                within_noise=abs(delta) <= noise.noise_floor_pct,
                nic_effective=eff,
                engaged=True,
                staging_suspected=staging,
            )
        )
    on_tps = lead.tps_per_gpu or 0.0
    off_tps = pts["agg-off-256"].tps_per_gpu or 0.0
    mtp_delta = (on_tps - off_tps) / off_tps * 100.0
    families = (
        FamilySummary(
            family_id="agg-tp8",
            title="Aggregated TP8 x4 (per-node replicas)",
            status="measured",
            best_point_id=ids["agg-256"],
            best_tps_per_gpu=lead.tps_per_gpu,
            best_concurrency=256,
            saturated=True,
            cross_node_gbs=0.01,
            point_ids=tuple(ids[f"agg-{c}"] for c, *_ in AGG_CURVE),
        ),
        FamilySummary(
            family_id="disagg-1p1d",
            title="Disagg 1P1D TP8/TP8 x2 (cross-node, NIXL)",
            status="measured",
            best_point_id=ids["dis-256"],
            best_tps_per_gpu=pts["dis-256"].tps_per_gpu,
            best_concurrency=256,
            saturated=True,
            cross_node_gbs=9.1,
            point_ids=tuple(ids[f"dis-{c}"] for c, *_ in DIS_CURVE),
        ),
        FamilySummary(
            family_id="wideep-dep16",
            title="Wide-EP DEP16 x2 (DeepEP, 2-node replicas)",
            status="pruned",
            reason="below 60% of the leader at equal concurrency after 3 points",
            best_point_id=ids["wep-256"],
            best_tps_per_gpu=pts["wep-256"].tps_per_gpu,
            best_concurrency=256,
            saturated=False,
            cross_node_gbs=22.5,
            point_ids=tuple(ids[f"wep-{c}"] for c, *_ in WEP_CURVE) + (ids["wep-512"],),
        ),
        FamilySummary(
            family_id="disagg-2p1d",
            title="Disagg 2P1D TP8 (cross-node, NIXL)",
            status="deploy_failed",
            reason="oom: decode KV pool does not fit at mem-fraction 0.88",
        ),
    )
    deploys = (
        _deploy(AGG, load_seconds=812.0),
        _deploy(AGG_OFF, load_seconds=790.0),
        _deploy(DIS, load_seconds=905.0),
        _deploy(WEP, load_seconds=1010.0),
        _deploy(
            DIS2,
            ok=False,
            load_seconds=640.0,
            failure="oom",
            detail=(
                "torch.OutOfMemoryError: CUDA out of memory. "
                "Tried to allocate 2.00 GiB"
            ),
        ),
        *(_deploy(dis_nic(n), load_seconds=880.0 + n) for n in (4, 2, 1)),
    )
    return ExperimentSummary(
        exp_id=exp_id,
        fingerprint=H200_FP,
        gpu_profile="h200",
        gpu_name="NVIDIA H200",
        node_count=4,
        gpus_per_node=8,
        model="glm-5.3",
        checkpoint=Checkpoint(
            repo="zai-org/GLM-5.3", revision="main", approx_size_gb=756.0
        ),
        served_context=262144,
        dataset=DATASET,
        aiperf_version="0.14.0",
        image=IMAGE,
        started_at=started_at,
        ended_at=started_at + 9.1 * 3600,
        budget_hours=9.5,
        spent_hours=9.1,
        inventory=h200_inventory(),
        families=families,
        points=tuple(pts.values()),
        deploys=deploys,
        peaks=h200_peaks(),
        leader=AGG,
        leader_point_id=lead.key.id,
        saturated=True,
        saturation_note="C=256 to 512: -1.2%, within the 1.86% noise floor",
        floors=(
            FloorResult(
                user_tps_floor=20.0,
                point_id=ids["agg-128"],
                cand_key=AGG.key,
                concurrency=128,
                tps_per_gpu=pts["agg-128"].tps_per_gpu,
                interpolated_tps_per_gpu=round(111.5 * scale, 2),
                crossed=True,
            ),
            FloorResult(
                user_tps_floor=50.0,
                point_id=ids["dis-64"],
                cand_key=DIS.key,
                concurrency=64,
                tps_per_gpu=pts["dis-64"].tps_per_gpu,
                interpolated_tps_per_gpu=round(90.3 * scale, 2),
                crossed=True,
            ),
        ),
        noise=noise,
        mtp=MtpComparison(
            on_point_id=lead.key.id,
            off_point_id=ids["agg-off-256"],
            on_tps_per_gpu=on_tps,
            off_tps_per_gpu=off_tps,
            delta_pct=mtp_delta,
            within_noise=abs(mtp_delta) <= noise.noise_floor_pct,
            accept_len=2.41,
        ),
        nic_family_id="disagg-1p1d",
        nic_concurrency=256,
        nic_reason=(
            "The leader agg-tp8 has no cross-node data-plane traffic; the sweep "
            "runs on the best cross-node family."
        ),
        nic_arms=tuple(arms),
        model_arch=ARCH,
        interconnect=interconnect_needs(
            lead, arms, H200_PROFILE, sweep_family="disagg-1p1d"
        ),
        kv_capacity=h200_kv_capacity(),
        nvlink=nvl.compare(
            lead,
            pts["nvlink-off"],
            noise,
            switch=NVLINK_SWITCH,
            restore={"action": "restore", "ok": True},
            reason="the throughput leader",
        ),
        confirm_point_id=ids["confirm"],
        dropped_traces=3,
        conditions=(
            "AIPerf client co-located on node0 (no spare CPU host)",
            "No worker restart between concurrency points; per-run AgentX cache-bust",
        ),
        limitations=(
            "Single cluster rental; run-to-run variance across rentals not measured.",
        ),
        unverified=("DeepEP low_latency mode was not exercised on this image.",),
        verdict="ok",
    )


def h200_kv_capacity():
    """KV pools of the H200 leader: 4 replicas, no attention DP, FP8 KV."""
    lifecycle = DeployLifecycle(
        roles={
            "agg": RoleLifecycle(
                role="agg",
                containers=4,
                ready=4,
                kv_tokens_min=381_000,
                kv_cache_gb_min=16.4,
                max_total_num_tokens_min=381_000,
                max_running_requests_min=128,
                context_len=262_144,
            )
        }
    )
    return kv_capacity(
        lifecycle,
        h200_leader_plan(),
        arch=ARCH,
        requests=RequestSummary(context_p50=62_146.0, context_p90=171_030.0),
        state=PointState(
            roles={
                "agg": RoleBrief(values={"kv_usage.avg": 0.83, "kv_usage.max": 0.97})
            }
        ),
    )


def h200_leader_plan(summary: ExperimentSummary | None = None) -> LaunchPlan:
    """The rendered plan of the H200 leader (4 per-node TP8 replicas)."""
    cand = AGG if summary is None or summary.leader is None else summary.leader
    sglang = {
        "model-path": "/root/models/zai-org--GLM-5.3@main",
        "served-model-name": "glm-5.3",
        "tp-size": 8,
        "context-length": 262144,
        "mem-fraction-static": 0.82,
        "chunked-prefill-size": 16384,
        "kv-cache-dtype": "fp8_e4m3",
        "cuda-graph-max-bs-decode": 128,
        "speculative-algorithm": "EAGLE",
        "speculative-num-steps": 3,
        "speculative-eagle-topk": 1,
        "speculative-num-draft-tokens": 4,
        "enable-metrics": True,
        "trust-remote-code": True,
        "kv-events-config": {"publisher": "zmq", "endpoint": "tcp://*:20080"},
    }
    containers = [
        ContainerSpec(
            node="node0",
            name="cb-control",
            role="control",
            image=IMAGE,
            command=("etcd", "--listen-client-urls", "http://10.42.0.1:2379"),
        ),
        ContainerSpec(
            node="node0",
            name="cb-frontend",
            role="frontend",
            image=IMAGE,
            env={"ETCD_ENDPOINTS": "http://10.42.0.1:2379", "DYN_HTTP_PORT": "8000"},
            command=("python3", "-m", "dynamo.frontend", "--router-mode", "kv"),
            health_url="http://10.42.0.1:8000/v1/models",
        ),
    ]
    for i in range(4):
        containers.append(
            ContainerSpec(
                node=f"node{i}",
                name=f"cb-agg-{i}",
                role="agg",
                instance=i,
                gpus=tuple(range(8)),
                image=IMAGE,
                env={
                    "DYN_DISCOVERY_BACKEND": "etcd",
                    "ETCD_ENDPOINTS": "http://10.42.0.1:2379",
                    "DYN_REQUEST_PLANE": "tcp",
                    "DYN_TCP_RPC_HOST": f"10.42.0.{i + 1}",
                    "DYN_SYSTEM_PORT": "8081",
                    "SGLANG_ENABLE_SPEC_V2": "1",
                    "HF_TOKEN": SECRET_ENV_VALUE,
                },
                command=(
                    "python3",
                    "-m",
                    "dynamo.sglang",
                    "--config",
                    f"/bench/experiments/exp/deployments/{cand.fp}/agg-{i}.yaml",
                    "--api-key",
                    SECRET_ENV_VALUE,
                ),
                system_port=8081,
                metrics_url=f"http://10.42.0.{i + 1}:8081/metrics",
                hcas=tuple(f"mlx5_{h}" for h in range(8)),
            )
        )
    return LaunchPlan(
        candidate=cand,
        cand_fp=cand.fp,
        namespace="cb-exp-h200-001",
        served_model_name="glm-5.3",
        frontend_url="http://10.42.0.1:8000",
        containers=tuple(containers),
        gpu_count=32,
        transport="none",
        cross_node=(),
        expected_instances={"agg": 4},
        nic_selection={
            f"node{i}": tuple(f"mlx5_{h}" for h in range(8)) for i in range(4)
        },
        rendered_configs={f"agg:{i}": dict(sglang) for i in range(4)},
    )


def rtx6000_unsupported_summary(exp_id: str = "exp-rtx6000-001") -> ExperimentSummary:
    """A run whose GLM-5.3 preflight failed on SM120: no points, no leader."""
    cand = Candidate(family_id="agg-tp8")
    return ExperimentSummary(
        exp_id=exp_id,
        fingerprint=RTX_FP,
        gpu_profile="rtx_pro_6000",
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        node_count=4,
        gpus_per_node=8,
        model="glm-5.3",
        checkpoint=Checkpoint(
            repo="RadixArk/GLM-5.3-NVFP4", revision="main", approx_size_gb=430.0
        ),
        served_context=262144,
        dataset=DATASET,
        aiperf_version="0.14.0",
        image=IMAGE,
        started_at=T0 + 86400,
        ended_at=T0 + 86400 + 1.4 * 3600,
        budget_hours=9.5,
        spent_hours=1.4,
        inventory=rtx_inventory(),
        families=(
            FamilySummary(
                family_id="agg-tp8",
                title="Aggregated TP8 x4 (per-node replicas)",
                status="unsupported",
                reason="kernel_unsupported: FlashMLA sparse decode has no SM120 build",
            ),
            FamilySummary(
                family_id="agg-tp4x2",
                title="Aggregated TP4 x8",
                status="unsupported",
                reason="same kernel gap as agg-tp8",
            ),
        ),
        points=(),
        deploys=(
            DeployResult(
                candidate=cand,
                cand_fp=cand.fp,
                ok=False,
                load_seconds=420.0,
                failure="kernel_unsupported",
                detail="RuntimeError: FlashMLA sparse attention requires SM90/SM100",
                log_dir=f"deployments/{cand.fp}/logs",
            ),
        ),
        peaks=(
            PeakResult(
                plane="pcie",
                test="p2p_copy",
                scope="node0",
                value_gbs=41.2,
                nominal_gbs=64.0,
                tool="p2p_bench",
            ),
            PeakResult(
                plane="ib",
                test="gdr_write_bw",
                scope="node0/mlx5_0",
                nic_count=1,
                value_gbs=23.1,
                nominal_gbs=25.0,
                tool="ib_write_bw --use_cuda",
            ),
        ),
        nic_reason="no candidate deployed",
        limitations=(
            "GLM-5.3 DSA attention / FlashMLA is not supported on SM120 in this image.",
        ),
        unverified=("FP8 fallback model was not attempted.",),
        verdict="unsupported",
    )
