# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-free simulation for ``--dry-run`` and end-to-end tests.

A ``RecordingExecutor`` answers every command through ``Simulator.respond``
(probe JSON, docker state, health checks, completions), so the real phase,
deployment and planner code runs unchanged while every command is recorded.
Measurements come from a deterministic synthetic throughput model; its
numbers are illustrative only and are labelled as simulated in reports.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.lium_cluster_bench.adapters.accuracy import (
    simulated_stdout as accuracy_simulated_stdout,
)
from benchmarks.lium_cluster_bench.adapters.diagnostics import (
    simulated_stdout as diagnostics_simulated_stdout,
)
from benchmarks.lium_cluster_bench.adapters.nodefacts import simulated_stdout
from benchmarks.lium_cluster_bench.app.pipeline import Interrupted
from benchmarks.lium_cluster_bench.domain.bandwidth import PointBandwidth
from benchmarks.lium_cluster_bench.domain.fabric import (
    FabricNodeReport,
    FabricTopology,
    LinkMatrix,
    SwitchPortRef,
    SwitchSnapshot,
    TraceResult,
    synthetic_link_matrix,
    synthetic_reports,
    synthetic_snapshot,
    synthetic_traces,
)
from benchmarks.lium_cluster_bench.domain.gpuactivity import (
    ActivityBrief,
    RoleActivity,
    classify_activity,
)
from benchmarks.lium_cluster_bench.domain.modelarch import SIMULATED_CONFIG, ModelArch
from benchmarks.lium_cluster_bench.domain.pointstate import (
    PointState,
    RoleBrief,
    StressBrief,
)
from benchmarks.lium_cluster_bench.domain.requests import (
    RequestSummary,
    SloTier,
    parse_record,
    summarize_requests,
    synthetic_records,
)
from benchmarks.lium_cluster_bench.domain.schema import (
    GIB,
    BandwidthStat,
    DiskInfo,
    GpuInfo,
    GpuProfile,
    HcaInfo,
    HcaPort,
    Inventory,
    LaunchPlan,
    NodeInventory,
    PeakResult,
    PointMetrics,
)
from benchmarks.lium_cluster_bench.domain.search import RunPoint
from benchmarks.lium_cluster_bench.domain.servicestate import Bottleneck
from benchmarks.lium_cluster_bench.ports import CommandResult

SIM_NAMES = {
    "b200": ("NVIDIA B200", 183359, "10.0"),
    "b300": ("NVIDIA B300 SXM6 AC", 275040, "10.3"),
    "h200": ("NVIDIA H200", 143771, "9.0"),
    "h100": ("NVIDIA H100 80GB HBM3", 81559, "9.0"),
    "rtx_pro_6000": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, "12.0"),
}


def synthetic_node(
    index: int,
    *,
    gpu_name: str,
    memory_mib: int,
    compute_cap: str = "9.0",
    gpus: int = 8,
    hcas: int = 8,
    nvlink: bool = True,
    hca_rate_gbps: float = 400.0,
    mem_total_bytes: int = 2048 * GIB,
    free_bytes: int = 6000 * GIB,
) -> NodeInventory:
    """A node where GPUs and NICs share PCIe switches evenly (two NUMA nodes)."""
    switches = max(1, min(gpus, hcas))
    gpus_per_switch = max(1, gpus // switches)
    gpu_list, hca_list = [], []
    pci_paths: dict[str, str] = {}
    for g in range(gpus):
        switch = g // gpus_per_switch
        bdf = f"0000:{0x18 + 0x10 * g:02x}:00.0"
        gpu_list.append(
            GpuInfo(
                index=g,
                name=gpu_name,
                uuid=f"GPU-sim-{index}-{g}",
                pci_bus_id=bdf,
                memory_mib=memory_mib,
                numa_node=0 if switch < switches / 2 else 1,
                compute_cap=compute_cap,
                pcie_gen_max=5,
                pcie_width_max=16,
            )
        )
        pci_paths[
            bdf
        ] = f"/sys/devices/pci0000:{switch:02x}/0000:{switch:02x}:01.0/0000:{0x40 + switch:02x}:00.0/{bdf}"
    for h in range(hcas):
        switch = h * switches // hcas
        bdf = f"0000:{0x19 + 0x10 * h:02x}:00.0"
        hca_list.append(
            HcaInfo(
                name=f"mlx5_{h}",
                pci_bus_id=bdf,
                numa_node=0 if switch < switches / 2 else 1,
                netdev=f"ibp{h}s0",
                ports=(
                    HcaPort(
                        port=1,
                        state="4: ACTIVE",
                        rate=f"{int(hca_rate_gbps)} Gb/sec (4X NDR)",
                        rate_gbps=hca_rate_gbps,
                        link_layer="InfiniBand",
                    ),
                ),
            )
        )
        pci_paths[
            bdf
        ] = f"/sys/devices/pci0000:{switch:02x}/0000:{switch:02x}:01.0/0000:{0x40 + switch:02x}:00.0/{bdf}"
    topo: dict[str, dict[str, str]] = {}
    for gpu in gpu_list:
        g_switch = gpu.index // gpus_per_switch
        row = {}
        for h, hca in enumerate(hca_list):
            h_switch = h * switches // hcas
            row[hca.name] = (
                "PIX"
                if h_switch == g_switch
                else ("NODE" if hca.numa_node == gpu.numa_node else "SYS")
            )
        topo[f"GPU{gpu.index}"] = row
    return NodeInventory(
        name=f"node{index}",
        hostname=f"sim-{index}",
        address=f"10.42.0.{index + 1}",
        gpus=tuple(gpu_list),
        hcas=tuple(hca_list),
        pci_paths=pci_paths,
        topo=topo,
        driver_version="580.65.06",
        cuda_version="13.0",
        cpu_model="Simulated CPU",
        cpu_count=224,
        mem_total_bytes=mem_total_bytes,
        disks=(
            DiskInfo(
                path="/",
                device="overlay",
                fs_type="overlay",
                total_bytes=200 * GIB,
                free_bytes=150 * GIB,
            ),
            DiskInfo(
                path="/workspace",
                device="/dev/nvme0n1",
                fs_type="ext4",
                total_bytes=7000 * GIB,
                free_bytes=free_bytes,
            ),
        ),
        docker_version="27.3.1",
        kernel="5.15.0-sim",
        ofed_version="MLNX_OFED_LINUX-24.10",
        nvidia_peermem=True,
        nvlink_active=nvlink,
    )


def synthetic_inventory(gpu_profile: str, nodes: int = 4) -> Inventory:
    name, mem, cc = SIM_NAMES[gpu_profile]
    pcie_only = gpu_profile == "rtx_pro_6000"
    kwargs: dict[str, Any] = {"gpu_name": name, "memory_mib": mem, "compute_cap": cc}
    if pcie_only:
        kwargs.update(
            hcas=4, nvlink=False, hca_rate_gbps=200.0, mem_total_bytes=1024 * GIB
        )
    return Inventory(nodes=tuple(synthetic_node(i, **kwargs) for i in range(nodes)))


_CONTAINER = re.compile(r"\b(cb-[A-Za-z0-9][A-Za-z0-9-]*)")


def _container_name(command: str) -> str:
    match = _CONTAINER.search(command)
    return match.group(1) if match else ""


class SimClock:
    """Virtual time: polls cost nothing, simulated work advances the clock."""

    def __init__(self, start: float = 1_800_000_000.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds

    def advance(self, seconds: float) -> None:
        self._t += seconds


SIM_DEPLOY_S = 15 * 60
SIM_CONFIGURE_S = 240
SIM_FABRIC_S = 180
SIM_RECORDS = (
    1500  # synthetic AIPerf records per point  # identify + traces + link matrix
)


@dataclass
class Simulator:
    inventory: Inventory
    gpu: GpuProfile
    clock: SimClock
    help_text: Callable[[], str] = field(default=lambda: "")
    unsupported_rungs_needed: tuple[str, ...] = ()  # kernel rungs required to deploy
    crash_after_points: int | None = None  # raise Interrupted after N measurements
    measured_points: int = 0
    accuracy: float = 0.94  # GSM8K accuracy the simulated probe reports
    accuracy_fail_families: tuple[str, ...] = ()  # these report 0.42 instead
    calls: list[str] = field(default_factory=list)
    # NVLink control state (nvidia-smi nvlink -sBwMode / -sLWidth).
    nvlink_bw_mode: str = "FULL"
    nvlink_width: int = 18

    # ------------------------------------------------------------- commands
    def respond(self, node: str, command: str) -> CommandResult | None:
        self.calls.append(f"{node}: {command[:200]}")
        facts = simulated_stdout(
            command,
            node,
            node_names=[n.name for n in self.inventory.nodes],
            gpu_name=self.inventory.nodes[0].gpus[0].name,
            now=self.clock.now(),
        )
        if facts is not None:
            return CommandResult(node, command, 0, facts, "")
        family = getattr(self, "_family", "")
        diag = diagnostics_simulated_stdout(
            command, node, devices=len(self.inventory.nodes[0].gpus)
        )
        if diag is not None:
            return CommandResult(node, command, 0, diag, "")
        probe = accuracy_simulated_stdout(
            command,
            accuracy=0.42 if family in self.accuracy_fail_families else self.accuracy,
        )
        if probe is not None:
            return CommandResult(node, command, 0, probe, "")
        if "agents/probe.py" in command:
            data = self.inventory.node(node).model_dump(mode="json")
            data.pop("name")
            data.pop("address")
            return CommandResult(node, command, 0, json.dumps(data), "")
        if "--help" in command:
            return CommandResult(node, command, 0, self.help_text(), "")
        if command.startswith("cat ") and command.rstrip("'\" ").endswith(
            "/config.json"
        ):
            return CommandResult(node, command, 0, json.dumps(SIMULATED_CONFIG), "")
        container = _container_name(command)
        if command.startswith("docker inspect"):
            if self._kernel_fails(container):
                state = {
                    "Running": False,
                    "ExitCode": 1,
                    "OOMKilled": False,
                    "Status": "exited",
                }
            else:
                state = {
                    "Running": True,
                    "ExitCode": 0,
                    "OOMKilled": False,
                    "Status": "running",
                }
            return CommandResult(node, command, 0, json.dumps(state) + "|0", "")
        if command.startswith("docker logs"):
            text = ""
            if self._kernel_fails(container):
                text = "RuntimeError: no kernel image is available for execution on the device"
            elif any(role in container for role in ("agg", "prefill", "decode")):
                text = self._worker_log()
            return CommandResult(node, command, 0, text, "")
        if command.startswith("docker exec") and "environ" in command:
            env = self._env.get(container, {})
            return CommandResult(
                node, command, 0, "\n".join(f"{k}={v}" for k, v in env.items()), ""
            )
        if "/v1/models" in command:
            return CommandResult(
                node, command, 0, json.dumps({"data": [{"id": self._served}]}), ""
            )
        if "/v1/chat/completions" in command:
            body = {
                "choices": [{"message": {"content": "OK", "reasoning_content": None}}]
            }
            return CommandResult(node, command, 0, json.dumps(body), "")
        nvlink = self._nvlink_smi(node, command)
        if nvlink is not None:
            return nvlink
        if "nvidia-smi --query-gpu=memory.used" in command:
            return CommandResult(
                node,
                command,
                0,
                "\n".join("0" for _ in self.inventory.nodes[0].gpus),
                "",
            )
        return None

    # ------------------------------------------------------- NVLink control
    def _nvlink_smi(self, node: str, command: str) -> CommandResult | None:
        """nvidia-smi nvlink bandwidth mode (Hopper) and link width (Blackwell).

        Hopper honours -sBwMode; Blackwell reports it deprecated and offers
        no link width of 0, so the harness falls back to NCCL without P2P.
        """
        if "nvidia-smi nvlink" not in command:
            return None
        hopper = self.gpu.arch == "sm90"
        gpus = self.inventory.nodes[0].gpus
        if "-gBwMode" in command:
            if not hopper:
                return CommandResult(
                    node,
                    command,
                    0,
                    "NVLink Bandwidth Mode is DEPRECATED for Blackwell+.",
                    "",
                )
            lines = [
                f"GPU {g.index}: {g.name}\n    Bandwidth Mode: {self.nvlink_bw_mode}"
                for g in gpus
            ]
            return CommandResult(node, command, 0, "\n".join(lines), "")
        match = re.search(r"-sBwMode (\w+)", command)
        if match:
            if not hopper:
                return CommandResult(node, command, 3, "", "Not Supported")
            self.nvlink_bw_mode = match[1]
            return CommandResult(node, command, 0, "", "")
        if "-sLWidth values" in command:
            if hopper:
                return CommandResult(node, command, 3, "", "Not Supported")
            lines = [f"GPU {g.index}: Possible Link Widths: 18, 9, 6" for g in gpus]
            return CommandResult(node, command, 0, "\n".join(lines), "")
        match = re.search(r"-sLWidth (\d+)", command)
        if match:
            if hopper or int(match[1]) not in (18, 9, 6):
                return CommandResult(node, command, 3, "", "Invalid link width")
            self.nvlink_width = int(match[1])
            return CommandResult(node, command, 0, "", "")
        return CommandResult(node, command, 0, "", "")

    def nvlink_in_path(self) -> bool:
        return (
            self.gpu.nvlink and self.nvlink_bw_mode == "FULL" and self.nvlink_width > 0
        )

    def p2p_without_nvlink(self, node: str) -> list[PeakResult]:
        """GPU-GPU copy check after a switch: PCIe speed if NVLink is out."""
        count = len(self.inventory.nodes[0].gpus)
        pairs = count * (count - 1)
        value = 380.0 if self.nvlink_in_path() else 41.5
        self.clock.advance(90)
        return [
            PeakResult(
                plane="pcie",
                test="p2p_copy",
                scope=node,
                value_gbs=value,
                tool="p2p_bench.py (simulated)",
                details={
                    "peer_pairs": pairs,
                    "pairs_measured": pairs,
                    "min_gbs": value * 0.86,
                    "max_gbs": value * 1.06,
                },
            )
        ]

    # ---------------------------------------------------------- point state
    def point_state(
        self, action: RunPoint, plan: LaunchPlan, metrics: PointMetrics | None
    ) -> PointState:
        """Deterministic service state: KV usage and queues grow with load."""
        c = action.concurrency
        instances: dict[str, set[int]] = {}
        for spec in plan.containers:
            if spec.role in ("agg", "prefill", "decode"):
                instances.setdefault(spec.role, set()).add(spec.instance)
        pd = "decode" in instances
        serving = len(instances.get("decode" if pd else "agg", ())) or 1
        cap = 256
        per_instance = c / serving
        kv = min(0.97, 0.2 + per_instance / (per_instance + 96))
        queue = max(0.0, per_instance - cap)
        tps = metrics.output_tps if metrics and metrics.output_tps else 0.0
        roles: dict[str, RoleBrief] = {}
        for role, ids in sorted(instances.items()):
            serving_role = role in ("agg", "decode")
            values = {
                "running_reqs.avg": round(min(per_instance, cap) * 0.92, 2)
                if serving_role
                else round(min(8.0, per_instance / 16), 2),
                "running_reqs.max": float(min(int(per_instance), cap))
                if serving_role
                else 8.0,
                "queue_reqs.avg": round(queue, 2) if serving_role else 0.0,
                "kv_usage.avg": round(kv, 4),
                "kv_usage.max": round(min(0.99, kv + 0.03), 4),
                "cache_hit_rate.avg": 0.62,
            }
            if serving_role and tps:
                values["gen_throughput.avg"] = round(tps / max(1, len(ids)), 1)
            if role == "decode":
                values["decode_transfer_queue.avg"] = round(per_instance * 0.02, 2)
                values["stage_decode_waiting.avg"] = round(
                    0.05 + per_instance / 2000, 4
                )
                values["stage_decode_transferred.avg"] = 0.012
            if role == "prefill":
                values["stage_prefill_bootstrap.avg"] = round(0.02 + c / 20000, 4)
                values["stage_prefill_forward.avg"] = 0.9
                values["stage_prefill_transfer_kv_cache.avg"] = 0.18
                values["kv_transfer_bootstrap_ms.avg"] = 4.0
            derived: dict[str, float] = {"cached_token_fraction": 0.62}
            if serving_role:
                # Batches past the largest captured CUDA graph decode eagerly.
                derived["decode_cuda_graph_fraction"] = (
                    1.0 if per_instance <= 160 else round(160 / per_instance, 4)
                )
                if action.candidate.mtp:
                    derived["spec_tokens_per_verify"] = 2.62
                    values["spec_accept_rate.avg"] = 0.54
            if action.candidate.diag:
                # SGLang's opt-in metrics of the diagnostic variant.
                derived["forward_busy_fraction"] = 0.83 if serving_role else 0.61
                derived["est_tflops_per_gpu"] = 310.0 if serving_role else 820.0
                derived["est_mem_gbs_per_gpu"] = 5200.0 if serving_role else 2100.0
                values["eplb_balancedness.avg"] = 0.71
            roles[role] = RoleBrief(endpoints=len(ids), values=values, derived=derived)
        roles["frontend"] = RoleBrief(
            endpoints=1,
            values={
                "fe_inflight.avg": round(c * 0.95, 1),
                "fe_queued.avg": round(queue * serving, 1),
                "router_kv_hit_rate.avg": 0.58,
            },
        )
        if kv > 0.9:
            label, evidence = "kv_capacity", f"KV usage {kv:.0%} (simulated)"
        elif queue > 0:
            label, evidence = "batch_cap", f"{queue:.0f} queued per instance"
        elif pd:
            label, evidence = "decode", "decode running batch near cap (simulated)"
        else:
            label, evidence = "client", "below saturation (simulated)"
        power = 0.62 * (1000.0 if self.gpu.hbm_gb >= 170 else 700.0)
        duration = metrics.duration_s if metrics and metrics.duration_s else 0.0
        activity = self._activity(instances, plan, kv)
        stress = StressBrief(
            gpu_count=plan.gpu_count,
            energy_j=round(power * plan.gpu_count * duration, 1) if duration else None,
            power_mean_w=round(power, 1),
            power_max_w=round(power * 1.35, 1),
            temp_max_c=78.0,
            sm_mhz_mean=1830.0,
            throttled_fraction=0.03,
            throttle={"sw_power_cap": 0.03},
            xmit_wait_fraction_max=0.012 if pd else 0.0,
            activity=activity,
        )
        return PointState(
            service_source="simulated",
            roles=roles,
            bottleneck=Bottleneck(label=label, confidence=0.6, evidence=(evidence,)),
            stress=stress,
            energy_per_token_j=round(power * plan.gpu_count / tps, 4) if tps else None,
        )

    def _activity(
        self, instances: dict[str, set[int]], plan: LaunchPlan, kv: float
    ) -> ActivityBrief:
        """Illustrative GPM view: prefill compute-bound, decode memory-bound."""
        shapes = {
            "prefill": (88.0, 58.0, 34.0),
            "decode": (70.0, 22.0, 64.0 + 20.0 * kv),
            "agg": (78.0, 36.0, 55.0),
        }
        roles: dict[str, RoleActivity] = {}
        gpus = {r: 0 for r in instances}
        for spec in plan.containers:
            if spec.role in gpus:
                gpus[spec.role] += len(spec.gpus)
        for role, n in gpus.items():
            sm, tensor, dram = shapes.get(role, (60.0, 30.0, 40.0))
            regime, reason = classify_activity(96.0, sm, tensor, dram)
            roles[role] = RoleActivity(
                role=role,
                gpus_sampled=n,
                gr_pct_mean=96.0,
                sm_pct_mean=sm,
                tensor_pct_mean=tensor,
                dram_pct_mean=round(dram, 1),
                regimes={regime: n},
                regime=regime,
                reason=reason,
            )
        return ActivityBrief(
            gpus_sampled=plan.gpu_count,
            gpus_expected=plan.gpu_count,
            availability={"ok": plan.gpu_count},
            roles=roles,
        )

    def requests(
        self,
        action: RunPoint,
        plan: LaunchPlan,
        metrics: PointMetrics | None,
        tiers: tuple[SloTier, ...],
        arch: ModelArch | None = None,
    ) -> RequestSummary | None:
        """Synthetic AIPerf records through the real per-request analysis.

        At most SIM_RECORDS records over the matching share of the window,
        so rates stay consistent with the simulated throughput."""
        if metrics is None or not metrics.request_count or not metrics.duration_s:
            return None
        n = min(metrics.request_count, SIM_RECORDS)
        start = (metrics.window_start_ns or 0) / 1e9
        span = metrics.duration_s * n / metrics.request_count
        raw = synthetic_records(
            requests=n,
            start_s=start,
            duration_s=span,
            ttft_p50_s=(metrics.ttft_p50_ms or 1000.0) / 1e3,
            itl_ms=metrics.itl_p50_ms or 20.0,
            osl=metrics.osl_avg or 700.0,
            isl=metrics.isl_avg or 95000.0,
            sessions=max(1, min(action.concurrency, n // 4 or 1)),
            seed=action.concurrency,
        )
        records = [r for r in (parse_record(x) for x in raw) if r is not None]
        return summarize_requests(
            records,
            gpus=plan.gpu_count,
            tiers=tiers,
            window=(start, start + span),
            arch=arch,
        )

    # ---------------------------------------------------------------- fabric
    def _fabric_shape(self) -> dict[str, Any]:
        node0 = self.inventory.nodes[0]
        ports = [p for h in node0.active_hcas for p in h.active_ports]
        ethernet = bool(ports) and ports[0].link_layer.lower() == "ethernet"
        names = [n.name for n in self.inventory.nodes]
        return {
            "node_names": names,
            "rails": len(node0.active_hcas) or 8,
            "rate_gbps": ports[0].rate_gbps if ports else 400.0,
            "nodes_per_leaf": min(4, len(names)),
            "kind": "roce" if ethernet else "infiniband",
        }

    def fabric_reports(self) -> dict[str, FabricNodeReport]:
        shape = self._fabric_shape()
        raw = synthetic_reports(
            shape["node_names"],
            rails=shape["rails"],
            kind=shape["kind"],
            nodes_per_leaf=shape["nodes_per_leaf"],
            rate_gbps=shape["rate_gbps"],
        )
        self.clock.advance(SIM_FABRIC_S / 3)
        return {n: FabricNodeReport.model_validate(d) for n, d in raw.items()}

    def fabric_traces(self, topology: FabricTopology) -> list[TraceResult]:
        return synthetic_traces(topology)

    def fabric_link_matrix(self) -> LinkMatrix:
        shape = self._fabric_shape()
        self.clock.advance(SIM_FABRIC_S * 2 / 3)
        return synthetic_link_matrix(
            shape["node_names"],
            rails=shape["rails"],
            rate_gbps=shape["rate_gbps"],
            nodes_per_leaf=shape["nodes_per_leaf"],
        )

    def switch_snapshot(
        self, refs: tuple[SwitchPortRef, ...], label: str
    ) -> SwitchSnapshot:
        return synthetic_snapshot(
            refs,
            node=self.inventory.nodes[0].name,
            t_s=self.clock.now(),
            # One rate per point: counters are cumulative from t=0, so the
            # before/after snapshots of a point must share it.
            utilization=0.3,
            label=label,
        )

    _deploy_started: float = 0.0
    _served: str = "zai-org/GLM-5.3"
    _env: dict[str, dict[str, str]] = field(default_factory=dict)
    _deploying_rungs: tuple[str, ...] = ()

    def _worker_log(self) -> str:
        """SGLang-format startup lines (lifecycle parsing in the dry-run)."""
        t0 = self._deploy_started

        def at(offset: float) -> str:
            stamp = datetime.fromtimestamp(t0 + offset, tz=timezone.utc)
            return stamp.strftime("[%Y-%m-%d %H:%M:%S TP0]")

        tokens = int(self.gpu.hbm_gb * 4000)
        kv_gb = round(self.gpu.hbm_gb * 0.18, 2)
        return "\n".join(
            [
                f"{at(20)} Load weight begin. avail mem={self.gpu.hbm_gb - 2:.2f} GB",
                f"{at(170)} Load weight end. type=GlmMoeDsaForCausalLM, "
                f"dtype=torch.bfloat16, avail mem=49.20 GB, mem usage=89.40 GB.",
                f"{at(172)} KV Cache is allocated. #tokens: {tokens}, KV size: {kv_gb} GB",
                f"{at(172)} Memory pool end. avail mem=24.80 GB",
                f"{at(174)} Capture cuda graph begin. This can take up to several "
                "minutes. avail mem=24.20 GB",
                f"{at(230)} Capture cuda graph end. Time elapsed: 56.00 s. "
                "mem usage=1.60 GB. avail mem=22.60 GB.",
                f"{at(231)} max_total_num_tokens={tokens}, chunked_prefill_size=16384, "
                "max_prefill_tokens=16384, max_running_requests=256, "
                "context_len=262144, available_gpu_mem=22.60 GB",
                f"{at(240)} The server is fired up and ready to roll!",
            ]
        )

    def set_plan(self, plan: LaunchPlan) -> None:
        self._deploy_started = self.clock.now()
        self._family = plan.candidate.family_id
        self._served = plan.served_model_name
        self._deploying_rungs = plan.candidate.rungs
        self._env = {c.name: dict(c.env) for c in plan.containers}
        self.clock.advance(SIM_DEPLOY_S)

    def _kernel_fails(self, container_name: str) -> bool:
        if (
            not self.unsupported_rungs_needed
            or "control" in container_name
            or "frontend" in container_name
        ):
            return False
        return not set(self.unsupported_rungs_needed) <= set(self._deploying_rungs)

    # ---------------------------------------------------------- measurements
    def measure(
        self, ctx: Any, action: RunPoint, plan: LaunchPlan, pdir: Path
    ) -> tuple[PointMetrics, PointBandwidth]:
        if (
            self.crash_after_points is not None
            and self.measured_points >= self.crash_after_points
        ):
            raise Interrupted("simulated SIGTERM")
        self.measured_points += 1
        cand = action.candidate
        family = cand.family_id
        c = action.concurrency
        gpus = plan.gpu_count
        scale = {"pd": 1.35, "agg-dtp8": 1.1, "agg-dp": 1.15}
        peak = 120.0 * self.gpu.hbm_gb / 141.0
        for prefix, factor in scale.items():
            if family.startswith(prefix):
                peak *= factor
        if not cand.mtp:
            peak /= 1.45
        if cand.p2p != "nvlink":
            # TP collectives on PCIe: ~1/7 of NVLink bandwidth; host staging
            # costs another hop through memory (illustrative).
            peak *= 0.64 if cand.p2p == "pcie" else 0.52
        cross = bool(plan.cross_node)
        if cross and cand.nic_count is not None:
            full = len(self.inventory.nodes[0].active_hcas)
            peak *= max(0.35, 1.0 - 0.12 * math.log2(full / cand.nic_count))
        sat = 96.0 if family.startswith("pd") else 64.0
        tps_gpu = peak * (1.0 - math.exp(-c / sat))
        # Deterministic +-1.5% wobble so the pilot sees non-zero noise.
        wobble = 1.0 + 0.015 * math.sin(len(ctx.store.events) * 1.7 + c)
        tps_gpu *= wobble
        total = tps_gpu * gpus
        # Per-user decode rate falls as the batch grows (illustrative).
        user = 110.0 * 64 / (64 + c)
        if cand.p2p != "nvlink":
            user *= 0.64 if cand.p2p == "pcie" else 0.52
        slow = 110.0 * 64 / (64 + c) / user  # >1 with NVLink off
        spec = ctx.config.bench_specs[action.spec]
        duration = float(spec.duration_s)
        self.clock.advance(SIM_CONFIGURE_S + spec.warmup_s)
        requests = max(c, int(total * duration / 700))
        metrics = PointMetrics(
            aiperf_version=ctx.config.aiperf_version,
            request_count=requests,
            error_count=0,
            output_tps=total,
            total_output_tokens=int(total * duration * 0.6),
            total_reasoning_tokens=int(total * duration * 0.4),
            user_tps_p50=user,
            ttft_p50_ms=(400 + 25 * c) * slow,
            ttft_p99_ms=(1200 + 60 * c) * slow,
            itl_p50_ms=1000 / max(user, 1e-6),
            itl_p99_ms=2000 / max(user, 1e-6),
            request_throughput=requests / duration,
            duration_s=duration,
            isl_avg=95000.0,
            osl_avg=700.0,
            window_start_ns=int(ctx.clock.now() * 1e9),
            window_end_ns=int((ctx.clock.now() + duration) * 1e9),
        )
        # Record the AIPerf command so --dry-run shows exactly what would run.
        ctx.cluster.run(
            ctx.node0,
            f"# simulated: aiperf profile --concurrency {c} --spec {action.spec}",
        )
        # KV handoff of every prompt (~95K tokens x ~45 KB/token for FP8 MLA)
        # for PD; TP/EP collectives for multi-node replicas.
        nodes = len(self.inventory.nodes)
        node_gbs = 0.02
        if "kv" in plan.cross_node:
            node_gbs = (total / 700.0) * 95_000 * 45e3 / 1e9 / nodes
        if "collective" in plan.cross_node:
            node_gbs += 35.0
        selected = plan.nic_selection.get(ctx.node0, ())
        stats = (
            BandwidthStat(
                plane="ib",
                scope="cluster",
                mean_gbs=node_gbs,
                p99_gbs=node_gbs * 1.6,
                peak_gbs=node_gbs * 2.0,
                ref_peak_gbs=45.0 * max(1, len(selected)),
                nominal_gbs=50.0 * max(1, len(selected)),
                method="measured",
            ),
            BandwidthStat(
                plane="nvlink" if self.gpu.nvlink else "pcie",
                scope="cluster",
                mean_gbs=40.0,
                p99_gbs=90.0,
                peak_gbs=120.0,
                ref_peak_gbs=self.gpu.nominal_nvlink_gbs or self.gpu.nominal_pcie_gbs,
                nominal_gbs=self.gpu.nominal_nvlink_gbs or self.gpu.nominal_pcie_gbs,
                method="measured" if self.gpu.nvlink else "derived",
            ),
            BandwidthStat(
                plane="pcie_nic",
                scope="cluster",
                mean_gbs=node_gbs / 8,
                p99_gbs=node_gbs / 5,
                peak_gbs=node_gbs / 4,
                ref_peak_gbs=self.gpu.nominal_pcie_gbs * 0.8,
                nominal_gbs=self.gpu.nominal_pcie_gbs,
                method="derived",
            ),
        )
        stats = stats + self._device_stats(plan, node_gbs, p2p=cand.p2p)
        self.clock.advance(duration + 60)
        bandwidth = PointBandwidth(
            stats=stats,
            cross_node_gbs=node_gbs,
            nic_effective=float(len(selected)) if cross else None,
            nic_engaged=True if (cross and cand.nic_count is not None) else None,
            staging_suspected=False,
            coverage=1.0,
            notes=("simulated",),
        )
        return metrics, bandwidth

    def _device_stats(
        self, plan: LaunchPlan, node_gbs: float, p2p: str = "nvlink"
    ) -> tuple[BandwidthStat, ...]:
        """Per-node IB, per-HCA and per-GPU rows like the real analysis emits.

        With NVLink off (``p2p`` pcie / host) the GPU-GPU traffic moves to
        PCIe, where it saturates the link, and NVLink reads zero.
        """
        rows: list[BandwidthStat] = []
        off = p2p != "nvlink"
        gpu_plane = "nvlink" if self.gpu.nvlink else "pcie"
        gpu_nominal = self.gpu.nominal_nvlink_gbs or self.gpu.nominal_pcie_gbs
        for n_idx, node in enumerate(self.inventory.nodes):
            hcas = plan.nic_selection.get(node.name, ()) or tuple(
                h.name for h in node.active_hcas
            )
            for direction, scale in (("tx", 1.0), ("rx", 0.9)):
                rows.append(
                    BandwidthStat(
                        plane="ib",
                        scope=node.name,
                        direction=direction,
                        mean_gbs=node_gbs * scale,
                        p99_gbs=node_gbs * 1.6 * scale,
                        peak_gbs=node_gbs * 2.0 * scale,
                        nominal_gbs=50.0 * max(1, len(hcas)),
                        method="measured",
                    )
                )
                for hca in hcas:
                    share = node_gbs / max(1, len(hcas))
                    rows.append(
                        BandwidthStat(
                            plane="ib",
                            scope=f"{node.name}/{hca}",
                            direction=direction,
                            mean_gbs=share * scale,
                            p99_gbs=share * 1.7 * scale,
                            peak_gbs=share * 2.1 * scale,
                            nominal_gbs=50.0,
                            method="measured",
                        )
                    )
                for g in node.gpus:
                    wobble = 1.0 + 0.05 * ((g.index + n_idx) % 4)
                    gpu_gbs = 0.0 if off and self.gpu.nvlink else 1.0
                    rows.append(
                        BandwidthStat(
                            plane=gpu_plane,
                            scope=f"{node.name}/gpu{g.index}",
                            direction=direction,
                            mean_gbs=40.0 * wobble * scale * gpu_gbs,
                            p99_gbs=90.0 * wobble * scale * gpu_gbs,
                            peak_gbs=120.0 * wobble * scale * gpu_gbs,
                            nominal_gbs=gpu_nominal,
                            method="measured" if self.gpu.nvlink else "derived",
                        )
                    )
                    nic_share = node_gbs / max(1, len(node.gpus))
                    rows.append(
                        BandwidthStat(
                            plane="pcie_nic",
                            scope=f"{node.name}/gpu{g.index}",
                            direction=direction,
                            mean_gbs=nic_share * scale,
                            p99_gbs=nic_share * 1.6 * scale,
                            peak_gbs=nic_share * 2.0 * scale,
                            nominal_gbs=self.gpu.nominal_pcie_gbs,
                            method="derived",
                        )
                    )
                    if self.gpu.nvlink:
                        moved = 34.0 * wobble if off else 0.0  # GPU-GPU on PCIe
                        rows.append(
                            BandwidthStat(
                                plane="pcie",
                                scope=f"{node.name}/gpu{g.index}",
                                direction=direction,
                                mean_gbs=(nic_share + 2.0 + moved * 0.6) * scale,
                                p99_gbs=(nic_share * 1.6 + 6.0 + moved) * scale,
                                peak_gbs=(nic_share * 2.0 + 9.0 + moved * 1.2) * scale,
                                nominal_gbs=self.gpu.nominal_pcie_gbs,
                                method="measured",
                            )
                        )
        return tuple(rows)

    def peaks(self, group: str) -> list[PeakResult]:
        nvlink_plane = "nvlink" if self.gpu.nvlink else "pcie"
        if group == "p2p":
            return [
                PeakResult(
                    plane=nvlink_plane,
                    test="p2p_ring_aggregate",
                    scope=n.name,
                    tool="simulated",
                    value_gbs=(self.gpu.nominal_nvlink_gbs or self.gpu.nominal_pcie_gbs)
                    * 0.8
                    * len(n.gpus),
                )
                for n in self.inventory.nodes
            ] + [
                PeakResult(
                    plane="pcie",
                    test="h2d",
                    scope=n.name,
                    tool="simulated",
                    value_gbs=self.gpu.nominal_pcie_gbs * 0.85,
                )
                for n in self.inventory.nodes
            ]
        if group == "collectives_intra":
            return [
                PeakResult(
                    plane=nvlink_plane,
                    test="nccl_all_reduce_intra",
                    scope=n.name,
                    tool="simulated",
                    value_gbs=(self.gpu.nominal_nvlink_gbs or self.gpu.nominal_pcie_gbs)
                    * 0.75,
                )
                for n in self.inventory.nodes
            ]
        if group == "collectives_inter":
            return [
                PeakResult(
                    plane="ib",
                    test="nccl_all_reduce_inter",
                    scope="cluster",
                    tool="simulated",
                    value_gbs=180.0,
                )
            ]
        if group == "gdr":
            full = len(self.inventory.nodes[0].active_hcas)
            out = []
            n = full
            while n >= 1:
                out.append(
                    PeakResult(
                        plane="ib",
                        test="gdr_send_recv",
                        scope="node0<->node1",
                        nic_count=n,
                        tool="simulated",
                        value_gbs=46.0 * n,
                        nominal_gbs=50.0 * n,
                    )
                )
                n //= 2
            return out
        return []
