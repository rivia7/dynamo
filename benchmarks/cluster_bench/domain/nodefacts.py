# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node facts, software stack, health snapshots and their findings (pure).

``agents/node_facts.py`` emits three JSON documents (its module docstring is
the field contract): ``node_facts`` (host mode), ``software_stack``
(container mode) and ``disk_read``. This module validates them into frozen
models and derives what the summary, report and knowledge base need:

* ``fact_findings``   - static problems on one or more nodes (degraded PCIe,
                        power caps, ECC / row-remap state, NVLink down,
                        IOMMU translation, missing GPU-direct RDMA, small
                        limits, unsynchronized clocks, heterogeneous
                        drivers / firmware / VBIOS / kernels, ...).
* ``health_of`` / ``health_delta`` / ``delta_findings`` - the counter subset
  of two snapshots ("before" / "after" a run) and what grew in between,
  with counter resets and vanished devices handled.
* ``clock_offsets`` / ``clock_findings`` - per-node clock offset to node0
  from round-trip samples.
* ``facts_signature`` - a small stable dict for the KB fingerprint.
* ``synthetic_*`` - deterministic healthy data for the dry-run simulator.

Parsing is lenient on purpose (data collection must never abort a run):
unknown keys are dropped and an invalid section or list element is replaced
by its default, each with a ``Note``. Messages are English; the report
localizes them.
"""

from __future__ import annotations

import math
import re
import types
from collections import defaultdict
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, ValidationError

from benchmarks.cluster_bench.domain.base import Frozen, stable_hash

GIB = 1 << 30
FACTS_VERSION = 1

Severity = Literal["info", "warn", "error"]
SEVERITY_RANK: dict[str, int] = {"info": 0, "warn": 1, "error": 2}

# Thresholds (documented in the finding messages).
MIN_SHM_BYTES = 1 * GIB
MIN_PIDS_MAX = 8192
CLOCK_INFO_S = 0.1
CLOCK_WARN_S = 1.0
TIME_SYNC_OFFSET_WARN_S = 0.1
DISK_SLOW_GBS = 0.5
DISK_SPREAD_RATIO = 2.0
DMABUF_MIN_KERNEL = (5, 12)
ACS_REDIRECT_BITS = 0x0C  # P2P Request Redirect | P2P Completion Redirect
MAX_LISTED = 8
# Idle GPUs park their link at Gen1; any other speed below the maximum means
# the link could not train to full speed.
GEN1_GTS = 2.5 + 0.01

PCIE_GEN_BY_GTS = (
    (2.5, 1),
    (5.0, 2),
    (8.0, 3),
    (16.0, 4),
    (32.0, 5),
    (64.0, 6),
    (128.0, 7),
)
THROTTLE_REASONS_BAD = frozenset(
    {
        "hw_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
        "sw_thermal_slowdown",
    }
)
# Port counters that measure traffic, not health.
IB_TRAFFIC_COUNTERS = frozenset(
    {
        "port_xmit_data",
        "port_rcv_data",
        "port_xmit_packets",
        "port_rcv_packets",
        "unicast_xmit_packets",
        "unicast_rcv_packets",
        "multicast_xmit_packets",
        "multicast_rcv_packets",
        "rx_write_requests",
        "rx_read_requests",
        "rx_atomic_requests",
        "rx_dct_connect",
        "lifespan",
    }
)
_HEALTH_WORDS = re.compile(
    r"err|discard|drop|ecn|cnp|marked|retrans|timeout|restart|wait|nak|seq|"
    r"buffer|downed|recovery|integrity|constraint|relay|duplicate|icrc|flush"
)
IB_ERROR_COUNTERS = frozenset({"link_downed"})
IB_WARN_COUNTERS = frozenset(
    {
        "symbol_error",
        "link_error_recovery",
        "port_rcv_errors",
        "port_rcv_remote_physical_errors",
        "port_rcv_switch_relay_errors",
        "local_link_integrity_errors",
        "excessive_buffer_overrun_errors",
        "port_xmit_discards",
        "port_xmit_constraint_errors",
        "port_rcv_constraint_errors",
        "out_of_buffer",
        "out_of_sequence",
        "packet_seq_err",
        "local_ack_timeout_err",
        "rnr_nak_retry_err",
        "implied_nak_seq_err",
        "req_cqe_error",
        "resp_cqe_error",
        "req_remote_access_errors",
        "req_remote_invalid_request",
        "resp_local_length_error",
        "resp_remote_access_errors",
        "duplicate_request",
        "rx_icrc_encapsulated",
        "roce_adp_retrans_to",
    }
)
# Static (since boot) port counters worth an info finding before a run.
IB_HISTORY_COUNTERS = (
    "link_downed",
    "symbol_error",
    "link_error_recovery",
    "port_rcv_errors",
    "port_rcv_remote_physical_errors",
    "local_link_integrity_errors",
    "excessive_buffer_overrun_errors",
    "port_xmit_discards",
)
# Xid codes that mean hardware / driver trouble (others are usually app-side).
XID_ERROR_CODES = frozenset(
    {48, 61, 62, 64, 74, 79, 94, 95, 109, 119, 120, 121, 140, 154}
)
COUNTER_GROUPS = ("gpu", "nvlink", "ib", "ethtool", "pcie_aer", "kernel_log", "cgroup")
GROUP_TITLES = {
    "gpu": "GPU",
    "nvlink": "NVLink",
    "ib": "RDMA port",
    "ethtool": "RoCE NIC (ethtool)",
    "pcie_aer": "PCIe AER",
    "pcie": "PCIe link",
    "kernel_log": "kernel log",
    "cgroup": "cgroup",
}
KEY_SOFTWARE = (
    "sglang",
    "sglang-kernel",
    "sgl-kernel",
    "flashinfer-python",
    "torch",
    "triton",
    "transformers",
    "nixl",
    "ai-dynamo",
    "ai-dynamo-runtime",
    "deep-ep",
    "sgl-deep-ep",
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-cuda13",
    "xgrammar",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
)

M = TypeVar("M", bound=BaseModel)


# --------------------------------------------------------------------------
# Models: host facts
# --------------------------------------------------------------------------


class Note(Frozen):
    """A machine-readable reason why a field is missing."""

    source: str
    error: str


class KernelFacts(Frozen):
    release: str | None = None
    uptime_s: float | None = None
    cmdline_flags: dict[str, str] = Field(default_factory=dict)
    iommu_groups: int | None = None
    iommu_devices: tuple[str, ...] = ()
    iommu_group_types: dict[str, str] = Field(default_factory=dict)

    @property
    def iommu_enabled(self) -> bool | None:
        if self.iommu_groups is None:
            return None
        return self.iommu_groups > 0

    @property
    def iommu_passthrough(self) -> bool | None:
        """True when GPUs/NICs bypass DMA translation (identity domains)."""
        kinds = set(self.iommu_group_types.values())
        if kinds:
            return kinds <= {"identity"}
        flags = self.cmdline_flags
        if flags.get("iommu") == "pt" or flags.get("iommu.passthrough") in (
            "1",
            "on",
            "y",
        ):
            return True
        if flags.get("iommu.passthrough") in ("0", "off", "n"):
            return False
        if flags.get("intel_iommu", "").startswith("on") or flags.get(
            "amd_iommu", ""
        ).startswith("on"):
            return False
        return None

    @property
    def release_tuple(self) -> tuple[int, ...]:
        match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", self.release or "")
        if not match:
            return ()
        return tuple(int(g) for g in match.groups() if g is not None)


class DriverFacts(Frozen):
    version: str | None = None
    flavor: Literal["open", "proprietary"] | None = None
    nvrm: str | None = None
    cuda_version: str | None = None


class CpuFacts(Frozen):
    online: str | None = None
    online_count: int | None = None
    governors: dict[str, int] = Field(default_factory=dict)
    scaling_driver: str | None = None
    cur_khz_min: int | None = None
    cur_khz_max: int | None = None
    cur_khz_mean: int | None = None
    max_khz: int | None = None
    boost: str | None = None
    smt_control: str | None = None
    smt_active: bool | None = None


class NumaNode(Frozen):
    node: int
    cpulist: str | None = None
    cpus: int | None = None
    mem_total_bytes: int | None = None
    distances: tuple[int, ...] = ()


class NumaFacts(Frozen):
    online: str | None = None
    balancing: int | None = None
    nodes: tuple[NumaNode, ...] = ()


class HugepagePool(Frozen):
    size_kb: int
    total: int = 0
    free: int | None = None
    reserved: int | None = None
    surplus: int | None = None


class MemoryFacts(Frozen):
    mem_total_bytes: int | None = None
    mem_available_bytes: int | None = None
    swap_total_bytes: int | None = None
    hugepages: tuple[HugepagePool, ...] = ()
    thp_enabled: str | None = None
    thp_defrag: str | None = None
    shm_total_bytes: int | None = None
    shm_free_bytes: int | None = None


class Limits(Frozen):
    """RLIMIT_* of the collecting process; -1 = unlimited."""

    memlock_soft: int | None = None
    memlock_hard: int | None = None
    nofile_soft: int | None = None
    nofile_hard: int | None = None
    nproc_soft: int | None = None
    nproc_hard: int | None = None


class CgroupFacts(Frozen):
    version: int | None = None
    path: str | None = None
    cpu_quota_us: int | None = None  # None = unlimited
    cpu_period_us: int | None = None
    cpuset_cpus: str | None = None
    cpuset_count: int | None = None
    pids_max: int | None = None  # None = unlimited
    pids_current: int | None = None
    cpu_nr_throttled: int | None = None
    cpu_throttled_us: int | None = None
    memory_events: dict[str, int] = Field(default_factory=dict)

    @property
    def cpu_quota_cores(self) -> float | None:
        if not self.cpu_quota_us or not self.cpu_period_us:
            return None
        return self.cpu_quota_us / self.cpu_period_us


def gen_of_speed(gts: float | None) -> int | None:
    """PCIe generation of a link speed in GT/s (16.0 -> 4)."""
    if gts is None:
        return None
    for speed, gen in PCIE_GEN_BY_GTS:
        if abs(gts - speed) < 0.3:
            return gen
    return None


def pcie_label(gen: int | None, width: int | None) -> str:
    """ "Gen5 x16" ("?" for an unknown part)."""
    return f"Gen{gen if gen else '?'} x{width if width else '?'}"


class PcieLink(Frozen):
    """One PCIe function: a GPU / HCA endpoint (depth 0) or a bridge above it."""

    bdf: str
    kind: Literal["gpu", "hca", "bridge"] = "bridge"
    vendor: str | None = None
    device: str | None = None
    pci_class: str | None = None
    driver: str | None = None
    numa_node: int | None = None
    current_speed_gts: float | None = None
    max_speed_gts: float | None = None
    current_width: int | None = None
    max_width: int | None = None
    parent: str | None = None
    depth: int = 0
    root_port: bool = False
    endpoints: tuple[str, ...] = ()
    aer_correctable: int | None = None
    aer_nonfatal: int | None = None
    aer_fatal: int | None = None
    acs_ctrl: int | None = None

    @property
    def gen_current(self) -> int | None:
        return gen_of_speed(self.current_speed_gts)

    @property
    def gen_max(self) -> int | None:
        return gen_of_speed(self.max_speed_gts)

    @property
    def width_degraded(self) -> bool:
        return bool(
            self.current_width
            and self.max_width
            and self.current_width < self.max_width
        )

    @property
    def speed_degraded(self) -> bool:
        return bool(
            self.current_speed_gts
            and self.max_speed_gts
            and self.current_speed_gts < self.max_speed_gts - 0.01
        )

    @property
    def current_label(self) -> str:
        return pcie_label(self.gen_current, self.current_width)

    @property
    def max_label(self) -> str:
        return pcie_label(self.gen_max, self.max_width)

    @property
    def acs_redirect(self) -> bool | None:
        if self.acs_ctrl is None:
            return None
        return bool(self.acs_ctrl & ACS_REDIRECT_BITS)


class GpuFacts(Frozen):
    """One GPU as ``nvidia-smi -q -x`` reports it."""

    index: int
    bus_id: str | None = None
    uuid: str | None = None
    serial: str | None = None
    product: str | None = None
    architecture: str | None = None
    vbios: str | None = None
    inforom_image: str | None = None
    gsp_firmware: str | None = None
    persistence_mode: str | None = None
    mig_current: str | None = None
    mig_pending: str | None = None
    compute_mode: str | None = None
    performance_state: str | None = None
    ecc_current: str | None = None
    ecc_pending: str | None = None
    ecc_volatile_corrected: int | None = None
    ecc_volatile_uncorrected: int | None = None
    ecc_aggregate_corrected: int | None = None
    ecc_aggregate_uncorrected: int | None = None
    ecc_counters: dict[str, int] = Field(default_factory=dict)
    retired_sbe: int | None = None
    retired_dbe: int | None = None
    retired_pending: bool | None = None
    remapped_correctable: int | None = None
    remapped_uncorrectable: int | None = None
    remap_pending: bool | None = None
    remap_failure: bool | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None  # enforced
    power_requested_w: float | None = None
    power_default_w: float | None = None
    power_min_w: float | None = None
    power_max_w: float | None = None
    clocks_max_graphics_mhz: int | None = None
    clocks_max_sm_mhz: int | None = None
    clocks_max_mem_mhz: int | None = None
    clocks_sm_mhz: int | None = None
    clocks_mem_mhz: int | None = None
    clocks_app_graphics_mhz: int | None = None
    clocks_app_mem_mhz: int | None = None
    clocks_event_active: tuple[str, ...] = ()
    clocks_event_us: dict[str, int] = Field(default_factory=dict)
    pcie_gen_current: int | None = None
    pcie_gen_max: int | None = None
    pcie_gen_device_max: int | None = None
    pcie_gen_host_max: int | None = None
    pcie_width_current: int | None = None
    pcie_width_max: int | None = None
    pcie_replay: int | None = None
    pcie_replay_rollover: int | None = None
    temp_c: float | None = None
    mem_temp_c: float | None = None
    temperature: dict[str, float] = Field(default_factory=dict)
    fabric_state: str | None = None
    fabric_status: str | None = None
    fabric_clique_id: str | None = None
    fabric_cluster_uuid: str | None = None
    fabric_health: dict[str, str] = Field(default_factory=dict)
    reset_required: str | None = None
    drain_and_reset_recommended: str | None = None
    recovery_action: str | None = None
    xml_fields: dict[str, str] = Field(default_factory=dict)

    @property
    def subject(self) -> str:
        return self.bus_id or f"gpu{self.index}"

    @property
    def label(self) -> str:
        return f"GPU{self.index} ({self.bus_id})" if self.bus_id else f"GPU{self.index}"


class NvlinkLink(Frozen):
    link: int
    speed_gbs: float | None = None  # None = inactive
    errors: dict[str, int] = Field(default_factory=dict)


class NvlinkGpu(Frozen):
    index: int
    uuid: str | None = None
    bus_id: str | None = None
    links: tuple[NvlinkLink, ...] = ()

    @property
    def subject(self) -> str:
        return self.bus_id or f"gpu{self.index}"

    @property
    def active(self) -> int:
        return sum(1 for link in self.links if link.speed_gbs)

    @property
    def total(self) -> int:
        return len(self.links)

    def error_totals(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for link in self.links:
            for name, value in link.errors.items():
                totals[name] += value
        return dict(sorted(totals.items()))


class HcaPortFacts(Frozen):
    port: int
    link_layer: str | None = None
    state: str | None = None
    phys_state: str | None = None
    rate: str | None = None
    netdev: str | None = None
    mtu: int | None = None
    counters: dict[str, int] = Field(default_factory=dict)
    hw_counters: dict[str, int] = Field(default_factory=dict)
    ethtool: dict[str, int] = Field(default_factory=dict)

    @property
    def active(self) -> bool:
        return "ACTIVE" in (self.state or "").upper()

    @property
    def rate_gbps(self) -> float | None:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Gb/s", self.rate or "")
        return float(match.group(1)) if match else None


class HcaFacts(Frozen):
    name: str
    bdf: str | None = None
    fw_ver: str | None = None
    board_id: str | None = None
    ports: tuple[HcaPortFacts, ...] = ()


class TimeSync(Frozen):
    source: str | None = None
    synchronized: bool | None = None
    offset_s: float | None = None
    stratum: int | None = None
    reference: str | None = None
    leap_status: str | None = None
    max_error_s: float | None = None
    est_error_s: float | None = None
    root_dispersion_s: float | None = None


class DockerFacts(Frozen):
    server_version: str | None = None
    storage_driver: str | None = None
    cgroup_driver: str | None = None
    cgroup_version: str | None = None
    default_runtime: str | None = None
    runtimes: tuple[str, ...] = ()
    nvidia_runtime: bool | None = None
    nvidia_container_runtime: str | None = None
    nvidia_ctk: str | None = None
    cdi_spec_dirs: tuple[str, ...] = ()
    docker_root_dir: str | None = None
    security_options: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class KernelEvent(Frozen):
    t_s: float | None = None  # seconds since boot
    kind: Literal["xid", "mlx5", "aer"]
    code: int | None = None  # Xid code
    pci: str | None = None
    ours: bool = True  # False: an Xid of a GPU this node does not own


class KernelSample(Frozen):
    t_s: float | None = None
    kind: str
    text: str


class KernelLog(Frozen):
    available: bool = False
    source: str | None = None
    error: str | None = None
    lines: int = 0
    events: tuple[KernelEvent, ...] = ()
    samples: tuple[KernelSample, ...] = ()


class NodeFacts(Frozen):
    """``agents/node_facts.py host`` output (one node, one moment)."""

    kind: Literal["node_facts"] = "node_facts"
    version: int = FACTS_VERSION
    node: str = ""
    hostname: str = ""
    available: bool = True  # False: the agent did not produce a document
    started_at: float | None = None
    collected_at: float | None = None
    duration_s: float | None = None
    kernel: KernelFacts = Field(default_factory=KernelFacts)
    driver: DriverFacts = Field(default_factory=DriverFacts)
    modules: dict[str, bool] = Field(default_factory=dict)
    cpu: CpuFacts = Field(default_factory=CpuFacts)
    numa: NumaFacts = Field(default_factory=NumaFacts)
    memory: MemoryFacts = Field(default_factory=MemoryFacts)
    limits: Limits = Field(default_factory=Limits)
    cgroup: CgroupFacts = Field(default_factory=CgroupFacts)
    pcie: tuple[PcieLink, ...] = ()
    gpus: tuple[GpuFacts, ...] = ()
    nvlink: tuple[NvlinkGpu, ...] = ()
    p2p: dict[str, dict[str, dict[str, str]]] = Field(default_factory=dict)
    hcas: tuple[HcaFacts, ...] = ()
    time_sync: TimeSync = Field(default_factory=TimeSync)
    docker: DockerFacts | None = None
    kernel_log: KernelLog = Field(default_factory=KernelLog)
    notes: tuple[Note, ...] = ()

    @property
    def dmabuf_capable(self) -> bool:
        """Open kernel module on Linux >= 5.12: GPU-direct RDMA via DMA-BUF."""
        release = self.kernel.release_tuple
        return (
            self.driver.flavor == "open"
            and bool(release)
            and (release[:2] >= DMABUF_MIN_KERNEL)
        )


# --------------------------------------------------------------------------
# Models: software stack, disk reads
# --------------------------------------------------------------------------


class TorchInfo(Frozen):
    version: str | None = None
    git_version: str | None = None
    cuda: str | None = None
    nccl: str | None = None
    cudnn: str | None = None
    arch_list: tuple[str, ...] = ()
    cuda_available: bool | None = None
    device_count: int | None = None
    device_name: str | None = None
    capability: str | None = None
    errors: tuple[str, ...] = ()


class UcxInfo(Frozen):
    version: str | None = None
    revision: str | None = None
    configured_with: tuple[str, ...] = ()


class SoftwareStack(Frozen):
    """``agents/node_facts.py container`` output (inside the serving image)."""

    kind: Literal["software_stack"] = "software_stack"
    version: int = FACTS_VERSION
    node: str = ""
    hostname: str = ""
    image: str | None = None  # set by the collector; the container cannot know
    collected_at: float | None = None
    python: str | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    missing_packages: tuple[str, ...] = ()
    torch: TorchInfo | None = None
    ucx: UcxInfo | None = None
    nvcc: str | None = None
    cuda_toolkit: str | None = None
    ibv_devices: tuple[str, ...] = ()
    os_release: dict[str, str] = Field(default_factory=dict)
    glibc: str | None = None
    nccl_libraries: tuple[str, ...] = ()
    nccl_plugins: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    notes: tuple[Note, ...] = ()


class DiskRead(Frozen):
    """Sequential read throughput of one weights file on one node."""

    kind: Literal["disk_read"] = "disk_read"
    version: int = FACTS_VERSION
    node: str = ""
    hostname: str = ""
    requested_path: str | None = None
    path: str | None = None
    file_bytes: int | None = None
    bytes: int = 0
    seconds: float | None = None
    gbs: float | None = None  # 1e9 bytes/s
    method: Literal["dd_direct", "o_direct", "buffered"] | None = None
    direct: bool = False
    cache: Literal["bypassed", "dropped", "unknown"] = "unknown"
    flushed: bool = False  # fsync'ed before timing (no writeback in the read)
    block_bytes: int | None = None
    error: str | None = None
    notes: tuple[Note, ...] = ()


# --------------------------------------------------------------------------
# Models: findings, health, clocks
# --------------------------------------------------------------------------


class FactFinding(Frozen):
    severity: Severity
    code: str  # stable machine code, e.g. "pcie_width_degraded"
    node: str  # logical node, or "cluster"
    subject: str = ""  # device(s) concerned
    message: str  # English
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Dedup key for the knowledge base."""
        return f"{self.code}|{self.node}|{self.subject}"


class HealthSnapshot(Frozen):
    """The counters subset of ``NodeFacts`` (plus states that can change).

    Counter groups map subject -> counter -> value: ``gpu`` and ``nvlink`` by
    GPU bus id, ``ib`` / ``ethtool`` by "<hca>/<port>", ``pcie_aer`` by BDF,
    ``kernel_log`` ("xid/<bdf>", "mlx5", "aer"; "<kind>/other" for PCI
    functions this node does not own) and
    ``cgroup`` ("cpu", "memory_events"). ``states`` maps group -> subject ->
    key -> value for non-counter facts compared verbatim (port state, link
    rate, active NVLink count, remap pending, ...).
    """

    node: str
    available: bool = True
    collected_at: float | None = None
    uptime_s: float | None = None
    gpu: dict[str, dict[str, int]] = Field(default_factory=dict)
    nvlink: dict[str, dict[str, int]] = Field(default_factory=dict)
    ib: dict[str, dict[str, int]] = Field(default_factory=dict)
    ethtool: dict[str, dict[str, int]] = Field(default_factory=dict)
    pcie_aer: dict[str, dict[str, int]] = Field(default_factory=dict)
    kernel_log: dict[str, dict[str, int]] = Field(default_factory=dict)
    cgroup: dict[str, dict[str, int]] = Field(default_factory=dict)
    states: dict[str, dict[str, dict[str, str]]] = Field(default_factory=dict)
    kernel_log_available: bool = False
    kernel_events: tuple[KernelEvent, ...] = ()

    def groups(self) -> dict[str, dict[str, dict[str, int]]]:
        return {
            "gpu": self.gpu,
            "nvlink": self.nvlink,
            "ib": self.ib,
            "ethtool": self.ethtool,
            "pcie_aer": self.pcie_aer,
            "kernel_log": self.kernel_log,
            "cgroup": self.cgroup,
        }


class CounterDelta(Frozen):
    group: str
    subject: str
    counter: str
    before: int | None = None
    after: int
    delta: int
    reset: bool = False  # after < before: the counter restarted; delta = after


class StateChange(Frozen):
    group: str
    subject: str
    key: str
    before: str | None = None
    after: str | None = None


class HealthDelta(Frozen):
    node: str
    complete: bool  # both snapshots were available
    before_at: float | None = None
    after_at: float | None = None
    elapsed_s: float | None = None
    increments: tuple[CounterDelta, ...] = ()
    changes: tuple[StateChange, ...] = ()
    missing: tuple[str, ...] = ()  # "<group>:<subject>" present before only
    notes: tuple[str, ...] = ()


class ClockOffset(Frozen):
    node: str
    offset_s: float  # remote - local midpoint; > 0: the node's clock is ahead
    uncertainty_s: float  # rtt / 2 of the chosen sample
    rtt_s: float
    samples: int


# --------------------------------------------------------------------------
# Lenient parsing
# --------------------------------------------------------------------------


def _model_of(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        for arg in get_args(annotation):
            model = _model_of(arg)
            if model is not None:
                return model
    return None


def _prune_value(annotation: Any, value: Any, path: str, dropped: list[str]) -> Any:
    model = _model_of(annotation)
    if model is not None and isinstance(value, Mapping):
        return _prune(model, value, path, dropped)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        for arg in args:
            if arg is not type(None):
                return _prune_value(arg, value, path, dropped)
    if origin in (tuple, list) and args and isinstance(value, (list, tuple)):
        return [_prune_value(args[0], item, path, dropped) for item in value]
    if origin is dict and len(args) == 2 and isinstance(value, Mapping):
        return {k: _prune_value(args[1], v, path, dropped) for k, v in value.items()}
    return value


def _prune(
    model: type[BaseModel], data: Mapping[str, Any], path: str, dropped: list[str]
) -> dict[str, Any]:
    """``data`` restricted (recursively) to the fields ``model`` declares."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        info = model.model_fields.get(key)
        if info is None:
            dropped.append(f"{path}{key}")
            continue
        out[key] = _prune_value(info.annotation, value, f"{path}{key}.", dropped)
    return out


def _error_text(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", ()))
    return f"{loc}: {first.get('msg', 'invalid')}"[:200]


def _element_model(model: type[BaseModel], name: str) -> type[BaseModel] | None:
    annotation = model.model_fields[name].annotation
    if get_origin(annotation) in (tuple, list):
        args = get_args(annotation)
        return _model_of(args[0]) if args else None
    return None


def _lenient(model: type[M], data: Mapping[str, Any]) -> tuple[M, list[Note]]:
    """Validate ``data``; on failure keep every field / element that is valid.

    Every field of ``model`` must have a default (true for the top-level
    document models here) so a single field validates on its own.
    """
    dropped: list[str] = []
    pruned = _prune(model, data, "", dropped)
    notes = [
        Note(source=f"parse:{path}", error="unknown field dropped")
        for path in dropped[:20]
    ]
    try:
        return model.model_validate(pruned), notes
    except ValidationError:
        pass
    good: dict[str, Any] = {}
    for name, value in pruned.items():
        try:
            model.model_validate({name: value})
            good[name] = value
            continue
        except ValidationError as exc:
            error = _error_text(exc)
        element = _element_model(model, name)
        if element is not None and isinstance(value, list):
            kept = []
            for item in value:
                try:
                    kept.append(element.model_validate(item))
                except ValidationError as item_exc:
                    error = _error_text(item_exc)
            good[name] = kept
            notes.append(
                Note(
                    source=f"parse:{name}",
                    error=f"{len(value) - len(kept)} invalid element(s) dropped: "
                    f"{error}",
                )
            )
        else:
            notes.append(
                Note(source=f"parse:{name}", error=f"section dropped: {error}")
            )
    return model.model_validate(good), notes


def _document(obj: Any, kind: str) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"expected a {kind!r} JSON object, got {type(obj).__name__}")
    if obj.get("kind") != kind:
        raise ValueError(f"expected kind={kind!r}, got kind={obj.get('kind')!r}")
    return dict(obj)


def parse_node_facts(obj: Any, *, node: str | None = None) -> NodeFacts:
    """Agent ``host`` JSON -> NodeFacts (raises ValueError on a wrong kind)."""
    data = _document(obj, "node_facts")
    if node:
        data["node"] = node
    data.pop("available", None)
    facts, notes = _lenient(NodeFacts, data)
    if notes:
        facts = facts.model_copy(update={"notes": (*facts.notes, *notes)})
    return facts


def unavailable_facts(node: str, reason: str) -> NodeFacts:
    """Placeholder for a node whose agent produced no usable document."""
    return NodeFacts(
        node=node,
        available=False,
        notes=(Note(source="collector", error=reason[:300]),),
    )


def parse_software(
    obj: Any, *, node: str | None = None, image: str | None = None
) -> SoftwareStack:
    """Agent ``container`` JSON -> SoftwareStack."""
    data = _document(obj, "software_stack")
    if node:
        data["node"] = node
    if image:
        data["image"] = image
    stack, notes = _lenient(SoftwareStack, data)
    if notes:
        stack = stack.model_copy(update={"notes": (*stack.notes, *notes)})
    return stack


def parse_disk_read(obj: Any, *, node: str | None = None) -> DiskRead:
    """Agent ``disk-read`` JSON -> DiskRead."""
    data = _document(obj, "disk_read")
    if node:
        data["node"] = node
    read, notes = _lenient(DiskRead, data)
    if notes:
        read = read.model_copy(update={"notes": (*read.notes, *notes)})
    return read


def failed_disk_read(node: str, path: str, reason: str) -> DiskRead:
    return DiskRead(node=node, requested_path=path, error=reason[:300])


# --------------------------------------------------------------------------
# Health snapshots and deltas
# --------------------------------------------------------------------------


def is_health_counter(name: str) -> bool:
    """Error / congestion counters (not traffic) of an RDMA port."""
    if name in IB_TRAFFIC_COUNTERS:
        return False
    if _HEALTH_WORDS.search(name):
        return True
    return not name.endswith(("_packets", "_bytes", "_data", "_requests"))


def _yes_no(value: bool | None) -> str | None:
    if value is None:
        return None
    return "yes" if value else "no"


def _states(**values: Any) -> dict[str, str]:
    return {k: str(v) for k, v in values.items() if v is not None}


def kernel_counts(log: KernelLog) -> dict[str, dict[str, int]]:
    """Kernel log events as subject -> counter -> count."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in log.events:
        subject, counter = _event_key(event)
        counts[subject][counter] += 1
    return {s: dict(c) for s, c in sorted(counts.items())}


def _event_key(event: KernelEvent) -> tuple[str, str]:
    """(subject, counter); devices of other tenants go to "<kind>/other"."""
    if event.kind == "xid":
        subject = f"xid/{event.pci}" if event.ours and event.pci else "xid/other"
        return subject, f"xid_{event.code}"
    suffix = "" if event.ours else "/other"
    if event.kind == "mlx5":
        return f"mlx5{suffix}", "mlx5_errors"
    return f"aer{suffix}", "aer_events"


def health_of(facts: NodeFacts) -> HealthSnapshot:
    """The counters (and comparable states) of one facts document."""
    if not facts.available:
        return HealthSnapshot(node=facts.node, available=False)
    gpu: dict[str, dict[str, int]] = {}
    states: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for g in facts.gpus:
        counters: dict[str, int] = {}
        volatile = {
            k: v for k, v in g.ecc_counters.items() if k.startswith("volatile.")
        }
        for name, value in (volatile or g.ecc_counters).items():
            counters[f"ecc.{name}"] = value
        for name, value in (
            ("retired_sbe", g.retired_sbe),
            ("retired_dbe", g.retired_dbe),
            ("remapped_correctable", g.remapped_correctable),
            ("remapped_uncorrectable", g.remapped_uncorrectable),
            ("pcie_replay", g.pcie_replay),
            ("pcie_replay_rollover", g.pcie_replay_rollover),
        ):
            if value is not None:
                counters[name] = value
        for name, value in g.clocks_event_us.items():
            counters[f"throttle_us.{name}"] = value
        gpu[g.subject] = counters
        states["gpu"][g.subject] = _states(
            remap_pending=_yes_no(g.remap_pending),
            remap_failure=_yes_no(g.remap_failure),
            retired_pending=_yes_no(g.retired_pending),
            fabric_state=g.fabric_state,
            fabric_status=g.fabric_status,
            reset_required=g.reset_required,
            drain_and_reset_recommended=g.drain_and_reset_recommended,
            recovery_action=g.recovery_action,
            ecc_current=g.ecc_current,
            mig_current=g.mig_current,
            persistence_mode=g.persistence_mode,
            pcie_width_current=g.pcie_width_current,
            power_limit_w=g.power_limit_w,
        )
    nvlink: dict[str, dict[str, int]] = {}
    for entry in facts.nvlink:
        if not entry.links:
            continue
        nvlink[entry.subject] = entry.error_totals()
        states["nvlink"][entry.subject] = _states(
            active_links=entry.active, total_links=entry.total
        )
    ib: dict[str, dict[str, int]] = {}
    ethtool: dict[str, dict[str, int]] = {}
    for hca in facts.hcas:
        for port in hca.ports:
            subject = f"{hca.name}/{port.port}"
            merged = {k: v for k, v in port.counters.items() if is_health_counter(k)}
            for name, value in port.hw_counters.items():
                if is_health_counter(name):
                    merged[name if name not in merged else f"hw.{name}"] = value
            ib[subject] = dict(sorted(merged.items()))
            if port.ethtool:
                ethtool[subject] = dict(port.ethtool)
            states["ib"][subject] = _states(
                state=port.state, phys_state=port.phys_state, rate=port.rate
            )
    aer: dict[str, dict[str, int]] = {}
    for link in facts.pcie:
        values = {
            name: value
            for name, value in (
                ("correctable", link.aer_correctable),
                ("nonfatal", link.aer_nonfatal),
                ("fatal", link.aer_fatal),
            )
            if value is not None
        }
        if values:
            aer[link.bdf] = values
        states["pcie"][link.bdf] = _states(
            width=link.current_width,
            speed_gts=link.current_speed_gts if link.kind != "gpu" else None,
        )
    cgroup: dict[str, dict[str, int]] = {}
    cpu = {
        name: value
        for name, value in (
            ("nr_throttled", facts.cgroup.cpu_nr_throttled),
            ("throttled_us", facts.cgroup.cpu_throttled_us),
        )
        if value is not None
    }
    if cpu:
        cgroup["cpu"] = cpu
    if facts.cgroup.memory_events:
        cgroup["memory_events"] = dict(facts.cgroup.memory_events)
    log = facts.kernel_log
    return HealthSnapshot(
        node=facts.node,
        available=True,
        collected_at=facts.collected_at,
        uptime_s=facts.kernel.uptime_s,
        gpu=gpu,
        nvlink=nvlink,
        ib=ib,
        ethtool=ethtool,
        pcie_aer=aer,
        kernel_log=kernel_counts(log) if log.available else {},
        cgroup=cgroup,
        states={g: dict(s) for g, s in states.items() if s},
        kernel_log_available=log.available,
        kernel_events=log.events,
    )


def _snapshot(value: HealthSnapshot | NodeFacts) -> HealthSnapshot:
    return value if isinstance(value, HealthSnapshot) else health_of(value)


def _counter_increments(
    group: str,
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
) -> tuple[list[CounterDelta], list[str]]:
    increments: list[CounterDelta] = []
    missing: list[str] = []
    for subject in sorted(set(before) | set(after)):
        if subject not in after:
            missing.append(f"{group}:{subject}")
            continue
        old = before.get(subject, {})
        for counter, value in sorted(after[subject].items()):
            prior = old.get(counter)
            if prior is None:
                if value > 0:
                    increments.append(
                        CounterDelta(
                            group=group,
                            subject=subject,
                            counter=counter,
                            after=value,
                            delta=value,
                        )
                    )
            elif value < prior:
                increments.append(
                    CounterDelta(
                        group=group,
                        subject=subject,
                        counter=counter,
                        before=prior,
                        after=value,
                        delta=value,
                        reset=True,
                    )
                )
            elif value > prior:
                increments.append(
                    CounterDelta(
                        group=group,
                        subject=subject,
                        counter=counter,
                        before=prior,
                        after=value,
                        delta=value - prior,
                    )
                )
    return increments, missing


def _kernel_increments(
    before: HealthSnapshot, after: HealthSnapshot, notes: list[str]
) -> list[CounterDelta]:
    if not (before.kernel_log_available and after.kernel_log_available):
        notes.append("kernel log unreadable in a snapshot: Xid/mlx5/AER not compared")
        return []
    if before.uptime_s is None or after.uptime_s is None:
        increments, _ = _counter_increments(
            "kernel_log", before.kernel_log, after.kernel_log
        )
        return increments
    rebooted = after.uptime_s < before.uptime_s
    # Events logged while the "before" agent ran belong to "before".
    threshold = max(
        [before.uptime_s] + [e.t_s for e in before.kernel_events if e.t_s is not None]
    )
    if rebooted:
        notes.append("node rebooted between snapshots (uptime decreased)")
        fresh = list(after.kernel_events)
    else:
        fresh = [
            e for e in after.kernel_events if e.t_s is not None and e.t_s > threshold
        ]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for event in fresh:
        counts[_event_key(event)] += 1
    increments = []
    for (subject, counter), count in sorted(counts.items()):
        increments.append(
            CounterDelta(
                group="kernel_log",
                subject=subject,
                counter=counter,
                before=before.kernel_log.get(subject, {}).get(counter, 0),
                after=after.kernel_log.get(subject, {}).get(counter, count),
                delta=count,
                reset=rebooted,
            )
        )
    return increments


def _state_changes(before: HealthSnapshot, after: HealthSnapshot) -> list[StateChange]:
    changes: list[StateChange] = []
    for group in sorted(set(before.states) | set(after.states)):
        old_group = before.states.get(group, {})
        new_group = after.states.get(group, {})
        for subject in sorted(set(old_group) & set(new_group)):
            old, new = old_group[subject], new_group[subject]
            for key in sorted(set(old) | set(new)):
                if old.get(key) != new.get(key):
                    changes.append(
                        StateChange(
                            group=group,
                            subject=subject,
                            key=key,
                            before=old.get(key),
                            after=new.get(key),
                        )
                    )
    return changes


def health_delta(
    before: HealthSnapshot | NodeFacts, after: HealthSnapshot | NodeFacts
) -> HealthDelta:
    """What changed on one node between two snapshots.

    Counters that went down were reset (driver reload, link reset, reboot):
    their delta is the new value and ``reset`` is set. Subjects present only
    before (a GPU fell off the bus, an HCA vanished) are listed in
    ``missing``. Kernel log events are compared by timestamp so a wrapped
    ring buffer does not hide new Xids.
    """
    old = _snapshot(before)
    new = _snapshot(after)
    node = new.node or old.node
    elapsed = None
    if old.collected_at is not None and new.collected_at is not None:
        elapsed = new.collected_at - old.collected_at
    if not (old.available and new.available):
        which = "before" if not old.available else "after"
        return HealthDelta(
            node=node,
            complete=False,
            before_at=old.collected_at,
            after_at=new.collected_at,
            elapsed_s=elapsed,
            notes=(f"{which} snapshot unavailable",),
        )
    notes: list[str] = []
    increments: list[CounterDelta] = []
    missing: list[str] = []
    old_groups, new_groups = old.groups(), new.groups()
    for group in COUNTER_GROUPS:
        if group == "kernel_log":
            continue
        found, gone = _counter_increments(group, old_groups[group], new_groups[group])
        increments += found
        missing += gone
    increments += _kernel_increments(old, new, notes)
    return HealthDelta(
        node=node,
        complete=True,
        before_at=old.collected_at,
        after_at=new.collected_at,
        elapsed_s=elapsed,
        increments=tuple(increments),
        changes=tuple(_state_changes(old, new)),
        missing=tuple(missing),
        notes=tuple(notes),
    )


def counter_severity(group: str, counter: str) -> Severity:
    """How alarming an increment of ``counter`` is during a run."""
    if group == "ib":
        name = counter[3:] if counter.startswith("hw.") else counter
        if name in IB_ERROR_COUNTERS:
            return "error"
        if name in IB_WARN_COUNTERS:
            return "warn"
        return "info"
    if group == "ethtool":
        if "storm_error" in counter:
            return "error"
        if re.search(r"discard|drop|err|crc|symbol|out_of_buffer|storm", counter):
            return "warn"
        return "info"
    if group == "gpu":
        if counter.startswith("ecc."):
            bad = "uncorrectable" in counter or "double_bit" in counter
            return "error" if bad else "info"
        if counter in ("remapped_uncorrectable", "retired_dbe"):
            return "error"
        if counter.startswith("throttle_us."):
            name = counter.split(".", 1)[1]
            benign = ("power_cap", "sync_boost", "application", "idle", "display")
            return "info" if any(word in name for word in benign) else "warn"
        return "warn"
    if group == "nvlink":
        return "error" if "fail" in counter else "warn"
    if group == "pcie_aer":
        return "warn" if counter == "correctable" else "error"
    if group == "kernel_log":
        match = re.match(r"^xid_(\d+)$", counter)
        if match:
            return "error" if int(match.group(1)) in XID_ERROR_CODES else "warn"
        return "warn"
    if group == "cgroup":
        if counter == "oom_kill":
            return "error"
        if counter in ("oom", "nr_throttled", "throttled_us"):
            return "warn"
        return "info"
    return "info"


def _state_severity(change: StateChange) -> Severity:
    if change.group == "ib" and change.key in ("state", "phys_state"):
        return "error"
    if change.group == "nvlink" and change.key == "active_links":
        try:
            dropped = int(change.after or 0) < int(change.before or 0)
        except ValueError:
            dropped = True
        return "error" if dropped else "info"
    if change.group == "gpu" and change.key in (
        "fabric_state",
        "fabric_status",
        "reset_required",
        "drain_and_reset_recommended",
        "recovery_action",
        "remap_failure",
    ):
        return "error"
    return "warn"


def _snake(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def _listing(parts: Sequence[str]) -> str:
    shown = ", ".join(parts[:MAX_LISTED])
    if len(parts) > MAX_LISTED:
        shown += f" and {len(parts) - MAX_LISTED} more"
    return shown


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return ""
    if seconds >= 3600:
        return f" ({seconds / 3600:.1f} h)"
    return f" ({seconds / 60:.0f} min)"


def delta_findings(delta: HealthDelta) -> list[FactFinding]:
    """Findings for counters that grew, states that changed, devices lost."""
    node = delta.node
    if not delta.complete:
        return [
            FactFinding(
                severity="info",
                code="delta_incomplete",
                node=node,
                message="health counters not compared: " + "; ".join(delta.notes),
            )
        ]
    span = _duration(delta.elapsed_s)
    findings: list[FactFinding] = []
    grouped: dict[tuple[str, str], list[CounterDelta]] = defaultdict(list)
    for item in delta.increments:
        grouped[(item.group, item.counter)].append(item)
    for (group, counter), items in sorted(grouped.items()):
        total = sum(i.delta for i in items)
        severity = counter_severity(group, counter)
        if total == 0 or all(i.subject.endswith("/other") for i in items):
            severity = "info"  # only resets, or devices this node does not own
        parts = [
            f"{i.subject} reset (now {i.after})"
            if i.reset
            else f"{i.subject} +{i.delta}"
            for i in items
        ]
        title = GROUP_TITLES.get(group, group)
        # After a reset only the count since the reset is known: a lower bound.
        bound = "at least " if any(i.reset for i in items) else ""
        verb = f"increased by {bound}{total}" if total else "was reset"
        findings.append(
            FactFinding(
                severity=severity,
                code=f"delta_{group}_{_snake(counter)}",
                node=node,
                subject=",".join(i.subject for i in items),
                message=f"{title} counter {counter} {verb} during the run{span}: "
                + _listing(parts),
                data={
                    "total": total,
                    "by_subject": {i.subject: i.delta for i in items},
                    "reset": sorted(i.subject for i in items if i.reset),
                },
            )
        )
    for change in delta.changes:
        findings.append(
            FactFinding(
                severity=_state_severity(change),
                code=f"delta_state_{change.group}_{_snake(change.key)}",
                node=node,
                subject=change.subject,
                message=(
                    f"{GROUP_TITLES.get(change.group, change.group)} {change.subject} "
                    f"{change.key} changed from {change.before} to {change.after} "
                    f"during the run{span}"
                ),
                data={"before": change.before, "after": change.after},
            )
        )
    for item in delta.missing:
        group, _, subject = item.partition(":")
        findings.append(
            FactFinding(
                severity="error",
                code=f"delta_missing_{group}",
                node=node,
                subject=subject,
                message=f"{GROUP_TITLES.get(group, group)} {subject} present before "
                f"the run is missing afterwards{span}",
            )
        )
    for note in delta.notes:
        findings.append(
            FactFinding(severity="info", code="delta_note", node=node, message=note)
        )
    return _sorted(findings)


# --------------------------------------------------------------------------
# Static findings
# --------------------------------------------------------------------------


class _Findings:
    """Collects per-device problems and folds them into one finding per code."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        self._meta: dict[tuple[str, str], tuple[Severity, str]] = {}
        self.plain: list[FactFinding] = []

    def add(
        self,
        severity: Severity,
        code: str,
        node: str,
        subject: str,
        detail: str,
        title: str,
    ) -> None:
        key = (node, code)
        self._items[key].append((subject, detail))
        current = self._meta.get(key)
        if current is None or SEVERITY_RANK[severity] > SEVERITY_RANK[current[0]]:
            self._meta[key] = (severity, title)

    def one(
        self,
        severity: Severity,
        code: str,
        node: str,
        message: str,
        subject: str = "",
        **data: Any,
    ) -> None:
        self.plain.append(
            FactFinding(
                severity=severity,
                code=code,
                node=node,
                subject=subject,
                message=message,
                data=data,
            )
        )

    def findings(self) -> list[FactFinding]:
        out = list(self.plain)
        for (node, code), entries in self._items.items():
            severity, title = self._meta[(node, code)]
            parts = [f"{s} {d}".strip() for s, d in entries]
            out.append(
                FactFinding(
                    severity=severity,
                    code=code,
                    node=node,
                    subject=",".join(s for s, _ in entries),
                    message=f"{title}: {_listing(parts)}",
                    data={"count": len(entries)},
                )
            )
        return _sorted(out)


def _sorted(findings: Iterable[FactFinding]) -> list[FactFinding]:
    return sorted(
        findings,
        key=lambda f: (-SEVERITY_RANK[f.severity], f.node, f.code, f.subject),
    )


def _device_labels(facts: NodeFacts) -> dict[str, str]:
    labels = {g.bus_id: g.label for g in facts.gpus if g.bus_id}
    for hca in facts.hcas:
        if hca.bdf:
            labels[hca.bdf] = f"{hca.name} ({hca.bdf})"
    return labels


def _pcie_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    labels = _device_labels(facts)
    by_bdf = {link.bdf: link for link in facts.pcie}
    endpoints = [link for link in facts.pcie if link.depth == 0]
    for ep in endpoints:
        name = labels.get(ep.bdf, f"{ep.kind} {ep.bdf}")
        detail = f"at {ep.current_label}, capable of {ep.max_label}"
        if ep.width_degraded:
            out.add(
                "warn",
                "pcie_width_degraded",
                node,
                name,
                detail,
                "PCIe link trained narrower than its maximum width",
            )
        idle = (
            ep.kind == "gpu"
            and ep.current_speed_gts is not None
            and ep.current_speed_gts <= GEN1_GTS
        )
        if ep.speed_degraded and idle:
            out.add(
                "info",
                "pcie_speed_idle",
                node,
                name,
                detail,
                "GPU PCIe link at Gen1 at collection time (idle GPUs drop to "
                "2.5 GT/s, so the trained speed was not verified)",
            )
        elif ep.speed_degraded:
            out.add(
                "warn",
                "pcie_speed_degraded",
                node,
                name,
                detail,
                "PCIe link trained below its maximum speed",
            )
    reported: set[str] = set()
    for ep in endpoints:
        # The first bridge shares the endpoint's own link: start one above.
        first = by_bdf.get(ep.parent or "")
        bridge = by_bdf.get(first.parent or "") if first else None
        while bridge is not None:
            if bridge.width_degraded or bridge.speed_degraded:
                if bridge.bdf not in reported:
                    reported.add(bridge.bdf)
                    out.add(
                        "warn",
                        "pcie_upstream_degraded",
                        node,
                        bridge.bdf,
                        f"at {bridge.current_label}, capable of {bridge.max_label} "
                        f"(above {labels.get(ep.bdf, ep.bdf)})",
                        "Upstream PCIe link (switch / root port) below its maximum",
                    )
                break
            bridge = by_bdf.get(bridge.parent or "")
    for link in facts.pcie:
        if link.kind == "bridge" and not link.root_port and link.acs_redirect:
            below = [labels.get(b, b) for b in link.endpoints]
            out.add(
                "warn",
                "pcie_acs_redirect",
                node,
                link.bdf,
                f"(ACS ctrl 0x{link.acs_ctrl:04x}) above {_listing(below)}",
                "PCIe ACS peer-to-peer redirect enabled on a switch port: GPU<->NIC "
                "and GPU<->GPU traffic below it detours through the root complex",
            )
        bad = (link.aer_fatal or 0) + (link.aer_nonfatal or 0)
        if bad:
            out.add(
                "warn",
                "pcie_aer_uncorrectable",
                node,
                labels.get(link.bdf, link.bdf),
                f"nonfatal={link.aer_nonfatal or 0} fatal={link.aer_fatal or 0}",
                "PCIe uncorrectable errors recorded since boot (AER)",
            )
        elif link.aer_correctable:
            out.add(
                "info",
                "pcie_aer_correctable",
                node,
                labels.get(link.bdf, link.bdf),
                f"{link.aer_correctable}",
                "PCIe correctable errors recorded since boot (AER)",
            )
    in_sysfs = {link.bdf for link in endpoints}
    for gpu in facts.gpus:
        if gpu.bus_id in in_sysfs:
            continue
        if (
            gpu.pcie_width_current
            and gpu.pcie_width_max
            and gpu.pcie_width_current < gpu.pcie_width_max
        ):
            out.add(
                "warn",
                "pcie_width_degraded",
                node,
                gpu.label,
                f"at x{gpu.pcie_width_current}, capable of x{gpu.pcie_width_max} "
                "(nvidia-smi)",
                "PCIe link trained narrower than its maximum width",
            )
    for gpu in facts.gpus:
        if (
            gpu.pcie_gen_device_max
            and gpu.pcie_gen_host_max
            and gpu.pcie_gen_host_max < gpu.pcie_gen_device_max
        ):
            out.add(
                "warn",
                "pcie_slot_limited",
                node,
                gpu.label,
                f"host side supports Gen{gpu.pcie_gen_host_max}, GPU Gen"
                f"{gpu.pcie_gen_device_max}",
                "GPU sits behind a slower PCIe generation than it supports",
            )


def _gpu_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    for g in facts.gpus:
        name = g.label
        if (
            g.power_limit_w is not None
            and g.power_default_w is not None
            and g.power_limit_w < g.power_default_w - 1.0
        ):
            out.add(
                "warn",
                "gpu_power_capped",
                node,
                name,
                f"{g.power_limit_w:.0f} W of default {g.power_default_w:.0f} W",
                "GPU power limit below its default (throughput is power-capped)",
            )
        if (
            g.power_limit_w is not None
            and g.power_requested_w is not None
            and g.power_limit_w < g.power_requested_w - 1.0
        ):
            out.add(
                "warn",
                "gpu_power_enforced_below_requested",
                node,
                name,
                f"enforced {g.power_limit_w:.0f} W, "
                f"requested {g.power_requested_w:.0f} W",
                "Enforced GPU power limit below the requested one (system power "
                "brake or chassis cap)",
            )
        if (g.persistence_mode or "").lower() == "disabled":
            out.add(
                "info",
                "gpu_persistence_off",
                node,
                name,
                "",
                "Persistence mode off (driver state is torn down between "
                "processes; slower CUDA init)",
            )
        if "enabled" in ((g.mig_current or "") + (g.mig_pending or "")).lower():
            out.add(
                "error",
                "gpu_mig_enabled",
                node,
                name,
                f"current={g.mig_current} pending={g.mig_pending}",
                "MIG enabled (full-GPU serving is impossible)",
            )
        if g.compute_mode and g.compute_mode.lower() != "default":
            severity: Severity = (
                "error" if g.compute_mode.lower() == "prohibited" else "warn"
            )
            out.add(
                severity,
                "gpu_compute_mode",
                node,
                name,
                g.compute_mode,
                "GPU compute mode is not Default",
            )
        if (g.ecc_current or "").lower() == "disabled":
            out.add(
                "warn",
                "gpu_ecc_disabled",
                node,
                name,
                "",
                "ECC disabled (memory errors go undetected)",
            )
        if g.ecc_current and g.ecc_pending and g.ecc_current != g.ecc_pending:
            out.add(
                "warn",
                "gpu_ecc_pending_change",
                node,
                name,
                f"{g.ecc_current} -> {g.ecc_pending}",
                "ECC mode change pending a GPU reset",
            )
        if g.ecc_volatile_uncorrected:
            out.add(
                "error",
                "gpu_ecc_uncorrectable_volatile",
                node,
                name,
                f"{g.ecc_volatile_uncorrected}",
                "Uncorrectable ECC errors since the driver loaded",
            )
        elif g.ecc_aggregate_uncorrected:
            out.add(
                "warn",
                "gpu_ecc_uncorrectable_aggregate",
                node,
                name,
                f"{g.ecc_aggregate_uncorrected}",
                "Uncorrectable ECC errors in the GPU's lifetime record",
            )
        if g.remap_failure:
            out.add(
                "error",
                "gpu_remap_failure",
                node,
                name,
                "",
                "HBM row remapping failed (the GPU needs service)",
            )
        if g.remap_pending:
            out.add(
                "warn",
                "gpu_remap_pending",
                node,
                name,
                "",
                "HBM row remap pending (applied at the next GPU reset)",
            )
        if g.remapped_uncorrectable:
            out.add(
                "info",
                "gpu_rows_remapped_uncorrectable",
                node,
                name,
                f"{g.remapped_uncorrectable}",
                "HBM rows remapped after uncorrectable errors",
            )
        retired = (g.retired_sbe or 0) + (g.retired_dbe or 0)
        if retired or g.retired_pending:
            out.add(
                "warn",
                "gpu_retired_pages",
                node,
                name,
                f"sbe={g.retired_sbe or 0} dbe={g.retired_dbe or 0} "
                f"pending={_yes_no(g.retired_pending)}",
                "Retired GPU memory pages",
            )
        if g.pcie_replay:
            out.add(
                "warn",
                "gpu_pcie_replays",
                node,
                name,
                f"{g.pcie_replay} (rollovers {g.pcie_replay_rollover or 0})",
                "PCIe replays since the driver loaded (link-level retransmissions)",
            )
        bad = sorted(set(g.clocks_event_active) & THROTTLE_REASONS_BAD)
        if bad:
            out.add(
                "warn",
                "gpu_throttled",
                node,
                name,
                ",".join(bad),
                "GPU clocks throttled by hardware / thermal reasons",
            )
        if (g.reset_required or "").lower() == "yes" or (
            g.drain_and_reset_recommended or ""
        ).lower() == "yes":
            out.add(
                "error",
                "gpu_reset_required",
                node,
                name,
                f"reset_required={g.reset_required} "
                f"drain_and_reset={g.drain_and_reset_recommended}",
                "GPU reports that it needs a reset",
            )
        if g.recovery_action and g.recovery_action.lower() != "none":
            out.add(
                "error",
                "gpu_recovery_action",
                node,
                name,
                g.recovery_action,
                "GPU requests a recovery action",
            )
        state_bad = g.fabric_state and g.fabric_state.lower() != "completed"
        status_bad = g.fabric_status and g.fabric_status.lower() != "success"
        if state_bad or status_bad:
            out.add(
                "error",
                "nvswitch_fabric_not_ready",
                node,
                name,
                f"state={g.fabric_state} status={g.fabric_status}",
                "NVSwitch fabric not initialised (fabric manager down? NVLink "
                "peer access unavailable)",
            )
        health_bad = [
            f"{k}={v}"
            for k, v in sorted(g.fabric_health.items())
            if (k == "bandwidth" and v.lower() != "full")
            or (k == "summary" and v.lower() != "healthy")
        ]
        if health_bad:
            out.add(
                "warn",
                "nvswitch_fabric_degraded",
                node,
                name,
                ",".join(health_bad),
                "NVSwitch fabric health degraded",
            )
    vbios = {g.vbios for g in facts.gpus if g.vbios}
    if len(vbios) > 1:
        out.one(
            "warn",
            "vbios_mixed_in_node",
            node,
            f"GPUs run different VBIOS versions: {', '.join(sorted(vbios))}",
            versions=sorted(vbios),
        )


def _nvlink_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    labels = {g.index: g.label for g in facts.gpus}
    for entry in facts.nvlink:
        if not entry.total:
            continue
        name = labels.get(entry.index, f"GPU{entry.index}")
        down = [str(link.link) for link in entry.links if not link.speed_gbs]
        if down and entry.active:
            out.add(
                "error",
                "nvlink_down",
                node,
                name,
                f"{entry.active}/{entry.total} links active (down: {','.join(down)})",
                "NVLink links down",
            )
        elif down:
            out.add(
                "warn",
                "nvlink_inactive",
                node,
                name,
                f"0/{entry.total} links active",
                "No active NVLink on a GPU that lists links",
            )
        speeds = {link.speed_gbs for link in entry.links if link.speed_gbs}
        if len(speeds) > 1:
            out.add(
                "warn",
                "nvlink_speed_mismatch",
                node,
                name,
                "GB/s: " + ",".join(f"{s:g}" for s in sorted(speeds)),
                "NVLink links run at different speeds",
            )
        totals = {k: v for k, v in entry.error_totals().items() if v}
        if totals:
            out.add(
                "info",
                "nvlink_errors",
                node,
                name,
                ",".join(f"{k}={v}" for k, v in totals.items()),
                "NVLink error counters non-zero since the driver loaded",
            )


def _p2p_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    for mode in ("read", "write"):
        matrix = facts.p2p.get(mode) or {}
        pairs = [
            (row, col, status)
            for row, cells in matrix.items()
            for col, status in cells.items()
            if row != col and status not in ("OK", "X")
        ]
        if not pairs:
            continue
        total = sum(1 for row, cells in matrix.items() for col in cells if row != col)
        statuses = sorted({status for _, _, status in pairs})
        sample = [f"{r}->{c}={s}" for r, c, s in pairs[:4]]
        out.one(
            "warn",
            "p2p_unsupported",
            node,
            f"GPU peer-to-peer {mode} unsupported for {len(pairs)} of {total} GPU "
            f"pairs ({','.join(statuses)}; e.g. {', '.join(sample)}): those "
            "transfers stage through host memory",
            subject=mode,
            pairs=len(pairs),
            total=total,
            statuses=statuses,
        )


def _hca_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    rates: dict[float, list[str]] = defaultdict(list)
    for hca in facts.hcas:
        for port in hca.ports:
            subject = f"{hca.name}/{port.port}"
            if not port.active:
                out.add(
                    "warn",
                    "hca_port_not_active",
                    node,
                    subject,
                    f"state={port.state} phys={port.phys_state}",
                    "RDMA port not ACTIVE",
                )
            elif port.rate_gbps:
                rates[port.rate_gbps].append(subject)
            history = {
                k: port.counters[k] for k in IB_HISTORY_COUNTERS if port.counters.get(k)
            }
            if history:
                out.add(
                    "info",
                    "hca_error_counters",
                    node,
                    subject,
                    ",".join(f"{k}={v}" for k, v in history.items()),
                    "RDMA port error counters non-zero before the run",
                )
    if len(rates) > 1:
        parts = [
            f"{rate:g} Gb/s on {','.join(ports)}"
            for rate, ports in sorted(rates.items())
        ]
        out.one(
            "warn",
            "hca_rate_mismatch",
            node,
            "Active RDMA ports run at different rates: " + "; ".join(parts),
            rates={f"{rate:g}": ports for rate, ports in sorted(rates.items())},
        )
    firmware = {h.fw_ver for h in facts.hcas if h.fw_ver}
    if len(firmware) > 1:
        out.one(
            "warn",
            "hca_firmware_mixed_in_node",
            node,
            f"HCAs run different firmware: {', '.join(sorted(firmware))}",
            versions=sorted(firmware),
        )


def _platform_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    kernel = facts.kernel
    if kernel.iommu_enabled and kernel.iommu_passthrough is False:
        kinds = sorted(set(kernel.iommu_group_types.values())) or ["translated"]
        out.one(
            "warn",
            "iommu_translated",
            node,
            f"IOMMU translates GPU/NIC DMA ({','.join(kinds)}): GPU-direct RDMA "
            "and P2P may be slower or fail; iommu=pt avoids it",
            domains=kinds,
        )
    modules = facts.modules
    if modules and facts.gpus:
        if not modules.get("nvidia", False):
            out.one(
                "info",
                "kernel_modules_unknown",
                node,
                "Kernel module list unreadable (nvidia not listed although GPUs "
                "are visible); GPU-direct RDMA support not assessed",
            )
        elif not modules.get("nvidia_peermem", False):
            if facts.dmabuf_capable:
                out.one(
                    "info",
                    "gpudirect_rdma_dmabuf_only",
                    node,
                    "nvidia_peermem not loaded; GPU-direct RDMA relies on DMA-BUF "
                    "(open kernel module)",
                )
            else:
                out.one(
                    "warn",
                    "gpudirect_rdma_unavailable",
                    node,
                    "Neither nvidia_peermem nor DMA-BUF (open kernel module on "
                    "Linux >= 5.12) is available: NCCL/NIXL stage RDMA through "
                    "host memory",
                    flavor=facts.driver.flavor,
                    kernel=kernel.release,
                )
    limits = facts.limits
    if limits.memlock_hard not in (None, -1):
        out.one(
            "warn",
            "memlock_limited",
            node,
            f"RLIMIT_MEMLOCK hard limit is {limits.memlock_hard} bytes; RDMA "
            "memory registration needs unlimited (containers request "
            "memlock=-1, which a limited hard limit can refuse)",
            soft=limits.memlock_soft,
            hard=limits.memlock_hard,
        )
    elif limits.memlock_soft not in (None, -1):
        out.one(
            "info",
            "memlock_limited",
            node,
            f"RLIMIT_MEMLOCK soft limit is {limits.memlock_soft} bytes (hard "
            "unlimited): host-side RDMA tools must raise it",
            soft=limits.memlock_soft,
            hard=limits.memlock_hard,
        )
    shm = facts.memory.shm_total_bytes
    if shm is not None and shm < MIN_SHM_BYTES:
        out.one(
            "warn",
            "shm_small",
            node,
            f"/dev/shm is {shm / GIB:.2f} GiB; containers share it (--ipc host) "
            "for NCCL and tokenizer IPC",
            shm_total_bytes=shm,
        )
    if (facts.memory.thp_enabled or "").lower() == "always":
        out.one(
            "info",
            "thp_always",
            node,
            "Transparent huge pages enabled=always (compaction stalls are "
            "possible; madvise is the usual choice)",
        )
    cgroup = facts.cgroup
    cores = cgroup.cpu_quota_cores
    cpus = cgroup.cpuset_count or facts.cpu.online_count
    if cores is not None and cpus and cores < cpus - 0.5:
        out.one(
            "warn",
            "cpu_quota",
            node,
            f"cgroup CPU quota allows {cores:.1f} cores of {cpus} CPUs "
            "(tokenizers, schedulers and AIPerf share it)",
            quota_cores=round(cores, 2),
            cpus=cpus,
        )
    if cgroup.pids_max is not None and cgroup.pids_max < MIN_PIDS_MAX:
        out.one(
            "warn",
            "pids_limited",
            node,
            f"cgroup pids.max={cgroup.pids_max} (threads count; worker pools "
            "can exhaust it)",
            pids_max=cgroup.pids_max,
        )
    oom_kills = cgroup.memory_events.get("oom_kill", 0)
    if oom_kills:
        out.one(
            "info",
            "cgroup_oom_kills_before",
            node,
            f"{oom_kills} OOM kill(s) recorded in this cgroup before the snapshot",
            oom_kill=oom_kills,
        )
    governors = facts.cpu.governors
    if governors and set(governors) != {"performance"}:
        out.one(
            "info",
            "cpu_governor",
            node,
            "CPU frequency governor is not 'performance': "
            + ", ".join(f"{k}={v}" for k, v in sorted(governors.items())),
            governors=dict(governors),
        )
    if facts.numa.balancing:
        out.one(
            "info",
            "numa_balancing",
            node,
            "Automatic NUMA balancing is enabled (page migrations can add "
            "latency jitter)",
        )
    sync = facts.time_sync
    if sync.synchronized is False:
        out.one(
            "warn",
            "clock_unsynchronized",
            node,
            f"System clock not synchronized ({sync.source}"
            + (f", {sync.leap_status}" if sync.leap_status else "")
            + "): cross-node timestamps may drift",
            source=sync.source,
        )
    elif sync.offset_s is not None and abs(sync.offset_s) > TIME_SYNC_OFFSET_WARN_S:
        out.one(
            "warn",
            "clock_offset_large",
            node,
            f"System clock is {sync.offset_s:+.3f} s off its time source "
            f"({sync.source})",
            offset_s=sync.offset_s,
        )


def _kernel_log_findings(node: str, facts: NodeFacts, out: _Findings) -> None:
    log = facts.kernel_log
    if not log.available:
        return
    uptime = facts.kernel.uptime_s
    xids: dict[str, int] = defaultdict(int)
    last: float | None = None
    for event in log.events:
        if event.kind == "xid" and event.ours:
            xids[f"{event.code}@{event.pci}"] += 1
            if event.t_s is not None:
                last = max(last or 0.0, event.t_s)
    if xids:
        age = ""
        if uptime is not None and last is not None and uptime >= last:
            age = f"; last {(uptime - last) / 3600:.1f} h before collection"
        out.one(
            "warn",
            "kernel_log_xid",
            node,
            "NVIDIA Xid errors in the kernel log: "
            + ", ".join(f"Xid {k} x{v}" for k, v in sorted(xids.items()))
            + age,
            xids=dict(xids),
        )
    for kind, code, text in (
        ("mlx5", "kernel_log_mlx5", "mlx5 errors/warnings in the kernel log"),
        ("aer", "kernel_log_aer", "PCIe AER messages in the kernel log"),
    ):
        count = sum(1 for e in log.events if e.kind == kind and e.ours)
        if count:
            out.one("info", code, node, f"{count} {text}", count=count)


def _groups_by_value(values: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node, value in values.items():
        grouped[str(value)].append(node)
    return sorted(grouped.items())


def _heterogeneity(facts_by_node: Mapping[str, NodeFacts], out: _Findings) -> None:
    present = {n: f for n, f in facts_by_node.items() if f.available}
    if len(present) < 2:
        return

    def joined(items: Iterable[Any]) -> str | None:
        values = sorted({str(v) for v in items if v not in (None, "")})
        return "+".join(values) if values else None

    checks: list[tuple[Severity, str, str, Any]] = [
        ("warn", "driver", "NVIDIA driver version", lambda f: f.driver.version),
        (
            "warn",
            "driver_flavor",
            "NVIDIA kernel module flavor",
            lambda f: f.driver.flavor,
        ),
        ("warn", "kernel", "kernel release", lambda f: f.kernel.release),
        ("warn", "vbios", "GPU VBIOS", lambda f: joined(g.vbios for g in f.gpus)),
        (
            "warn",
            "hca_firmware",
            "HCA firmware",
            lambda f: joined(h.fw_ver for h in f.hcas),
        ),
        (
            "warn",
            "gpu_product",
            "GPU product",
            lambda f: joined(g.product for g in f.gpus),
        ),
        (
            "warn",
            "power_limit",
            "GPU power limit (W)",
            lambda f: joined(
                f"{g.power_limit_w:g}" for g in f.gpus if g.power_limit_w is not None
            ),
        ),
        ("info", "thp", "THP mode", lambda f: f.memory.thp_enabled),
        (
            "info",
            "iommu_passthrough",
            "IOMMU passthrough",
            lambda f: f.kernel.iommu_passthrough,
        ),
        ("info", "cpu_governor", "CPU governor", lambda f: joined(f.cpu.governors)),
    ]
    for severity, code, what, getter in checks:
        values = {n: getter(f) for n, f in present.items()}
        values = {n: v for n, v in values.items() if v is not None}
        groups = _groups_by_value(values)
        if len(groups) < 2:
            continue
        parts = [f"{value} on {','.join(nodes)}" for value, nodes in groups]
        out.one(
            severity,
            f"heterogeneous_{code}",
            "cluster",
            f"Nodes differ in {what}: " + "; ".join(parts),
            values={value: nodes for value, nodes in groups},
        )


def fact_findings(facts_by_node: Mapping[str, NodeFacts]) -> list[FactFinding]:
    """Static findings over one snapshot of every node (sorted, worst first)."""
    out = _Findings()
    unknown_sync: list[str] = []
    no_log: list[str] = []
    for node, facts in facts_by_node.items():
        if not facts.available:
            reason = "; ".join(n.error for n in facts.notes) or "no document"
            out.one(
                "warn",
                "facts_unavailable",
                node,
                f"Node facts unavailable: {reason}",
            )
            continue
        _pcie_findings(node, facts, out)
        _gpu_findings(node, facts, out)
        _nvlink_findings(node, facts, out)
        _p2p_findings(node, facts, out)
        _hca_findings(node, facts, out)
        _platform_findings(node, facts, out)
        _kernel_log_findings(node, facts, out)
        if facts.time_sync.synchronized is None:
            unknown_sync.append(node)
        if not facts.kernel_log.available:
            no_log.append(node)
    if unknown_sync:
        out.one(
            "info",
            "time_sync_unknown",
            "cluster",
            "Time synchronization status unknown (no chronyc/timedatectl/adjtimex) "
            f"on {','.join(unknown_sync)}; rely on measured clock offsets",
            nodes=unknown_sync,
        )
    if no_log:
        out.one(
            "info",
            "kernel_log_unavailable",
            "cluster",
            "Kernel log unreadable (dmesg / /dev/kmsg denied) on "
            f"{','.join(no_log)}: Xid and mlx5 errors are not visible",
            nodes=no_log,
        )
    _heterogeneity(facts_by_node, out)
    return out.findings()


def software_findings(
    software: SoftwareStack | None, *, node: str = ""
) -> list[FactFinding]:
    """Problems visible from inside the serving image."""
    if software is None:
        return [
            FactFinding(
                severity="info",
                code="software_unavailable",
                node=node or "cluster",
                message="Software stack of the serving image was not collected",
            )
        ]
    out = _Findings()
    where = software.node or node or "cluster"
    torch = software.torch
    if torch is None or torch.version is None:
        errors = "; ".join(torch.errors) if torch else "no probe output"
        out.one(
            "warn",
            "container_torch_unavailable",
            where,
            f"torch not importable in the serving image: {errors}",
        )
    elif torch.cuda_available is False:
        out.one(
            "error",
            "container_no_gpu",
            where,
            "CUDA unavailable inside the serving image (driver / toolkit mismatch "
            "or no GPU passed)",
            errors=list(torch.errors),
        )
    if not software.ibv_devices:
        out.one(
            "warn",
            "container_no_rdma_devices",
            where,
            "No RDMA device visible inside the serving image (ibv_devinfo -l): "
            "NIXL/NCCL fall back to TCP",
        )
    if software.missing_packages:
        out.one(
            "info",
            "container_packages_missing",
            where,
            "Packages not installed in the serving image: "
            + ", ".join(software.missing_packages),
            missing=list(software.missing_packages),
        )
    wheel = software.packages.get("nvidia-nccl-cu13") or software.packages.get(
        "nvidia-nccl-cu12"
    )
    if torch is not None and torch.nccl and wheel and not wheel.startswith(torch.nccl):
        out.one(
            "info",
            "nccl_version_mismatch",
            where,
            f"torch loads NCCL {torch.nccl} but the nvidia-nccl wheel is {wheel} "
            "(an override library is on the path)",
            torch_nccl=torch.nccl,
            wheel=wheel,
        )
    return out.findings()


def disk_findings(reads: Mapping[str, DiskRead]) -> list[FactFinding]:
    """Slow, failed, possibly cached or uneven weight-disk reads."""
    out = _Findings()
    speeds: dict[str, float] = {}
    for node, read in reads.items():
        if read.error or not read.gbs:
            out.one(
                "warn",
                "disk_read_failed",
                node,
                f"Weights disk read not measured: {read.error or 'no throughput'}",
                path=read.requested_path,
            )
            continue
        speeds[node] = read.gbs
        if read.gbs < DISK_SLOW_GBS:
            out.one(
                "warn",
                "disk_read_slow",
                node,
                f"Weights disk reads at {read.gbs:.2f} GB/s ({read.method}): model "
                "loads and restarts are slow",
                gbs=read.gbs,
            )
        if not read.direct and read.cache != "dropped":
            out.one(
                "info",
                "disk_read_maybe_cached",
                node,
                f"Disk read was buffered (cache {read.cache}): {read.gbs:.2f} GB/s "
                "may come from the page cache",
                gbs=read.gbs,
            )
    if len(speeds) > 1 and min(speeds.values()) > 0:
        ratio = max(speeds.values()) / min(speeds.values())
        if ratio > DISK_SPREAD_RATIO:
            parts = [f"{n}={v:.2f}" for n, v in sorted(speeds.items())]
            out.one(
                "info",
                "disk_read_heterogeneous",
                "cluster",
                f"Weights disk speed differs {ratio:.1f}x across nodes (GB/s): "
                + ", ".join(parts),
                gbs=speeds,
            )
    return out.findings()


# --------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------


def clock_offsets(
    samples: Mapping[str, Sequence[tuple[float, float, float]]]
) -> dict[str, ClockOffset]:
    """Per-node clock offset from (t_send, remote_time, t_recv) round trips.

    The minimum-RTT sample wins: offset = remote - (t_send + t_recv) / 2,
    uncertainty = rtt / 2 (the remote reading happened somewhere inside the
    round trip). Samples with a negative RTT (local clock jump) are dropped;
    nodes without a valid sample are omitted.
    """
    out: dict[str, ClockOffset] = {}
    for node, rounds in samples.items():
        valid = [
            (t_send, remote, t_recv)
            for t_send, remote, t_recv in rounds
            if t_recv >= t_send
            and all(math.isfinite(x) for x in (t_send, remote, t_recv))
        ]
        if not valid:
            continue
        t_send, remote, t_recv = min(valid, key=lambda s: s[2] - s[0])
        rtt = t_recv - t_send
        out[node] = ClockOffset(
            node=node,
            offset_s=remote - (t_send + t_recv) / 2.0,
            uncertainty_s=rtt / 2.0,
            rtt_s=rtt,
            samples=len(valid),
        )
    return out


def clock_findings(
    offsets: Mapping[str, ClockOffset],
    *,
    nodes: Sequence[str] = (),
    info_s: float = CLOCK_INFO_S,
    warn_s: float = CLOCK_WARN_S,
) -> list[FactFinding]:
    """Nodes whose clock is measurably off node0's (beyond the uncertainty)."""
    out = _Findings()
    for node, offset in sorted(offsets.items()):
        skew = abs(offset.offset_s) - offset.uncertainty_s
        if skew <= info_s:
            continue
        severity: Severity = "warn" if skew > warn_s else "info"
        direction = "ahead of" if offset.offset_s > 0 else "behind"
        out.one(
            severity,
            "clock_offset",
            node,
            f"Clock is {abs(offset.offset_s):.3f} s {direction} node0 "
            f"(+/- {offset.uncertainty_s:.3f} s): align cross-node timestamps",
            offset_s=offset.offset_s,
            uncertainty_s=offset.uncertainty_s,
        )
    unmeasured = [n for n in nodes if n not in offsets]
    if unmeasured:
        out.one(
            "info",
            "clock_offset_unmeasured",
            "cluster",
            f"Clock offset not measured on {','.join(unmeasured)}",
            nodes=unmeasured,
        )
    return out.findings()


# --------------------------------------------------------------------------
# Knowledge-base signature
# --------------------------------------------------------------------------


def _set_of(values: Iterable[Any]) -> list[str]:
    return sorted({str(v) for v in values if v not in (None, "")})


def facts_signature(
    facts_by_node: Mapping[str, NodeFacts], software: SoftwareStack | None
) -> dict[str, Any]:
    """Small, stable description of the platform for the KB.

    Per-node attributes are sorted sets of observed values (one element
    when homogeneous); hash it with ``stable_hash`` for a fingerprint.
    """
    nodes = [f for f in facts_by_node.values() if f.available]

    def each(getter: Any) -> list[str]:
        return _set_of(v for f in nodes for v in getter(f))

    def endpoint_labels(kind: str, current: bool) -> list[str]:
        return each(
            lambda f: (
                link.current_label if current else link.max_label
                for link in f.pcie
                if link.depth == 0 and link.kind == kind
            )
        )

    signature: dict[str, Any] = {
        "nodes": len(facts_by_node),
        "nodes_with_facts": len(nodes),
        "driver": each(lambda f: [f.driver.version]),
        "driver_flavor": each(lambda f: [f.driver.flavor]),
        "cuda_driver": each(lambda f: [f.driver.cuda_version]),
        "kernel": each(lambda f: [f.kernel.release]),
        "gpu_products": each(lambda f: (g.product for g in f.gpus)),
        "vbios": each(lambda f: (g.vbios for g in f.gpus)),
        "hca_firmware": each(lambda f: (h.fw_ver for h in f.hcas)),
        "hca_boards": each(lambda f: (h.board_id for h in f.hcas)),
        "hca_link_layers": each(
            lambda f: (p.link_layer for h in f.hcas for p in h.ports)
        ),
        "hca_rates": each(lambda f: (p.rate for h in f.hcas for p in h.ports)),
        "gpu_pcie_max": endpoint_labels("gpu", current=False),
        "gpu_pcie_degraded": sum(
            1
            for f in nodes
            for link in f.pcie
            if link.depth == 0 and link.kind == "gpu" and link.width_degraded
        ),
        "hca_pcie": endpoint_labels("hca", current=True),
        "gpu_power_limit_w": each(
            lambda f: (
                f"{g.power_limit_w:g}" for g in f.gpus if g.power_limit_w is not None
            )
        ),
        "gpu_power_default_w": each(
            lambda f: (
                f"{g.power_default_w:g}"
                for g in f.gpus
                if g.power_default_w is not None
            )
        ),
        "nvlink_active_links": each(
            lambda f: (str(e.active) for e in f.nvlink if e.total)
        ),
        "fabric_state": each(lambda f: (g.fabric_state for g in f.gpus)),
        "nvidia_peermem": each(lambda f: [f.modules.get("nvidia_peermem")]),
        "iommu_passthrough": each(lambda f: [f.kernel.iommu_passthrough]),
        "thp": each(lambda f: [f.memory.thp_enabled]),
        "software": None,
    }
    if software is not None:
        torch = software.torch
        signature["software"] = {
            "image": software.image,
            "packages": {
                name: software.packages[name]
                for name in KEY_SOFTWARE
                if name in software.packages
            },
            "torch_cuda": torch.cuda if torch else None,
            "nccl": torch.nccl if torch else None,
            "ucx": software.ucx.version if software.ucx else None,
            "cuda_toolkit": software.cuda_toolkit,
            "os": software.os_release.get("PRETTY_NAME"),
            "glibc": software.glibc,
            "nccl_plugins": sorted(p.rsplit("/", 1)[-1] for p in software.nccl_plugins),
        }
    return signature


# --------------------------------------------------------------------------
# Synthetic data for the dry-run simulator
# --------------------------------------------------------------------------

_SYNTH_GPU = {
    "h100": {
        "product": "NVIDIA H100 80GB HBM3",
        "architecture": "Hopper",
        "vbios": "96.00.99.00.01",
        "power": 700.0,
        "nvlink_gbs": 26.562,
        "pcie_gts": 32.0,
        "fabric": True,
        "sm_mhz": 1980,
        "mem_mhz": 2619,
        "hca_rate": "400 Gb/sec (4X NDR)",
        "hca_fw": "28.39.2048",
        "board": "MT_0000000838",
    },
    "h200": {
        "product": "NVIDIA H200",
        "architecture": "Hopper",
        "vbios": "96.00.AF.00.01",
        "power": 700.0,
        "nvlink_gbs": 26.562,
        "pcie_gts": 32.0,
        "fabric": True,
        "sm_mhz": 1980,
        "mem_mhz": 3201,
        "hca_rate": "400 Gb/sec (4X NDR)",
        "hca_fw": "28.39.2048",
        "board": "MT_0000000838",
    },
    "b200": {
        "product": "NVIDIA B200",
        "architecture": "Blackwell",
        "vbios": "97.00.88.00.0C",
        "power": 1000.0,
        "nvlink_gbs": 50.0,
        "pcie_gts": 32.0,
        "fabric": True,
        "sm_mhz": 1965,
        "mem_mhz": 3996,
        "hca_rate": "400 Gb/sec (4X NDR)",
        "hca_fw": "28.43.1014",
        "board": "MT_0000000838",
    },
    "b300": {
        "product": "NVIDIA B300 SXM6 AC",
        "architecture": "Blackwell",
        "vbios": "97.10.4A.00.04",
        "power": 1100.0,
        "nvlink_gbs": 50.0,
        "pcie_gts": 64.0,
        "fabric": True,
        "sm_mhz": 2032,
        "mem_mhz": 3996,
        "hca_rate": "800 Gb/sec (4X XDR)",
        "hca_fw": "40.45.1200",
        "board": "MT_0000001222",
    },
    "rtx": {
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "architecture": "Blackwell",
        "vbios": "98.02.2E.00.06",
        "power": 600.0,
        "nvlink_gbs": None,
        "pcie_gts": 32.0,
        "fabric": False,
        "sm_mhz": 2617,
        "mem_mhz": 14001,
        "hca_rate": "400 Gb/sec (4X NDR)",
        "hca_fw": "28.39.2048",
        "board": "MT_0000000838",
    },
}


def _synth_profile(gpu_name: str) -> dict[str, Any]:
    lowered = gpu_name.lower()
    key = "h100"
    for candidate in ("b300", "b200", "h200", "h100"):
        if candidate in lowered:
            key = candidate
            break
    else:
        if "rtx" in lowered or "6000" in lowered:
            key = "rtx"
    profile = dict(_SYNTH_GPU[key])
    if lowered.startswith("nvidia "):
        profile["product"] = gpu_name  # keep the caller's exact nvidia-smi name
    return profile


# GPU buses of the HGX layout used by the probe fixtures; NIC i sits on bus+1.
_SYNTH_BUSES = (0x18, 0x29, 0x3A, 0x4B, 0x9A, 0xAB, 0xBC, 0xCD)


def _synth_bdf(index: int, offset: int = 0) -> str:
    bus = _SYNTH_BUSES[index % len(_SYNTH_BUSES)] + offset
    return f"{index // len(_SYNTH_BUSES):04x}:{bus:02x}:00.0"


def synthetic_facts(
    node_names: Sequence[str], gpu_name: str, gpus: int = 8, hcas: int = 8
) -> dict[str, NodeFacts]:
    """Deterministic, healthy facts for every node (dry-run simulator)."""
    profile = _synth_profile(gpu_name)
    out: dict[str, NodeFacts] = {}
    for rank, node in enumerate(node_names):
        gpu_list: list[GpuFacts] = []
        nvlink: list[NvlinkGpu] = []
        pcie: list[PcieLink] = []
        hca_list: list[HcaFacts] = []
        for i in range(gpus):
            bdf = _synth_bdf(i)
            h = stable_hash([node, i], 32)
            gpu_list.append(
                GpuFacts(
                    index=i,
                    bus_id=bdf,
                    uuid=f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}",
                    serial=str(1650000000000 + rank * 100 + i),
                    product=profile["product"],
                    architecture=profile["architecture"],
                    vbios=profile["vbios"],
                    persistence_mode="Enabled",
                    mig_current="Disabled",
                    mig_pending="Disabled",
                    compute_mode="Default",
                    ecc_current="Enabled",
                    ecc_pending="Enabled",
                    ecc_volatile_corrected=0,
                    ecc_volatile_uncorrected=0,
                    ecc_aggregate_corrected=0,
                    ecc_aggregate_uncorrected=0,
                    ecc_counters={
                        "volatile.sram_correctable": 0,
                        "volatile.dram_uncorrectable": 0,
                    },
                    remapped_correctable=0,
                    remapped_uncorrectable=0,
                    remap_pending=False,
                    remap_failure=False,
                    power_limit_w=profile["power"],
                    power_requested_w=profile["power"],
                    power_default_w=profile["power"],
                    power_min_w=200.0,
                    power_max_w=profile["power"],
                    clocks_max_graphics_mhz=profile["sm_mhz"],
                    clocks_max_sm_mhz=profile["sm_mhz"],
                    clocks_max_mem_mhz=profile["mem_mhz"],
                    clocks_event_active=("gpu_idle",),
                    pcie_gen_max=gen_of_speed(profile["pcie_gts"]),
                    pcie_width_current=16,
                    pcie_width_max=16,
                    pcie_replay=0,
                    fabric_state="Completed" if profile["fabric"] else None,
                    fabric_status="Success" if profile["fabric"] else None,
                    recovery_action="None",
                )
            )
            if profile["nvlink_gbs"]:
                nvlink.append(
                    NvlinkGpu(
                        index=i,
                        uuid=gpu_list[-1].uuid,
                        bus_id=bdf,
                        links=tuple(
                            NvlinkLink(link=k, speed_gbs=profile["nvlink_gbs"])
                            for k in range(18)
                        ),
                    )
                )
            pcie.append(
                PcieLink(
                    bdf=bdf,
                    kind="gpu",
                    vendor="0x10de",
                    current_speed_gts=profile["pcie_gts"],
                    max_speed_gts=profile["pcie_gts"],
                    current_width=16,
                    max_width=16,
                    numa_node=0 if i < gpus // 2 else 1,
                )
            )
        for h in range(hcas):
            bdf = _synth_bdf(h, offset=1)
            pcie.append(
                PcieLink(
                    bdf=bdf,
                    kind="hca",
                    vendor="0x15b3",
                    current_speed_gts=min(profile["pcie_gts"], 32.0),
                    max_speed_gts=min(profile["pcie_gts"], 32.0),
                    current_width=16,
                    max_width=16,
                    numa_node=0 if h < hcas // 2 else 1,
                )
            )
            hca_list.append(
                HcaFacts(
                    name=f"mlx5_{h}",
                    bdf=bdf,
                    fw_ver=profile["hca_fw"],
                    board_id=profile["board"],
                    ports=(
                        HcaPortFacts(
                            port=1,
                            link_layer="InfiniBand",
                            state="4: ACTIVE",
                            phys_state="5: LinkUp",
                            rate=profile["hca_rate"],
                            counters={
                                "link_downed": 0,
                                "symbol_error": 0,
                                "port_rcv_errors": 0,
                                "port_xmit_discards": 0,
                                "port_xmit_wait": 0,
                            },
                            hw_counters={
                                "out_of_sequence": 0,
                                "local_ack_timeout_err": 0,
                            },
                        ),
                    ),
                )
            )
        labels = [f"GPU{i}" for i in range(gpus)]
        matrix = {r: {c: ("X" if r == c else "OK") for c in labels} for r in labels}
        out[node] = NodeFacts(
            node=node,
            hostname=f"synthetic-{node}",
            started_at=1_800_000_000.0,
            collected_at=1_800_000_010.0,
            duration_s=10.0,
            kernel=KernelFacts(
                release="6.8.0-60-generic",
                uptime_s=86400.0,
                cmdline_flags={"iommu": "pt"},
                iommu_groups=0,
            ),
            driver=DriverFacts(
                version="580.95.05",
                flavor="open",
                nvrm="NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 "
                "580.95.05",
                cuda_version="13.0",
            ),
            modules={
                "nvidia": True,
                "nvidia_uvm": True,
                "nvidia_peermem": True,
                "mlx5_core": True,
                "mlx5_ib": True,
                "ib_uverbs": True,
                "ib_umad": True,
                "rdma_ucm": True,
            },
            cpu=CpuFacts(
                online="0-223", online_count=224, governors={"performance": 224}
            ),
            numa=NumaFacts(
                online="0-1",
                balancing=0,
                nodes=(
                    NumaNode(node=0, cpulist="0-55,112-167", cpus=112),
                    NumaNode(node=1, cpulist="56-111,168-223", cpus=112),
                ),
            ),
            memory=MemoryFacts(
                mem_total_bytes=2048 * GIB,
                thp_enabled="madvise",
                thp_defrag="madvise",
                shm_total_bytes=1024 * GIB,
            ),
            limits=Limits(
                memlock_soft=-1,
                memlock_hard=-1,
                nofile_soft=1048576,
                nofile_hard=1048576,
            ),
            cgroup=CgroupFacts(version=2, cpuset_cpus="0-223", cpuset_count=224),
            pcie=tuple(pcie),
            gpus=tuple(gpu_list),
            nvlink=tuple(nvlink),
            p2p={"read": matrix, "write": matrix} if gpus > 1 else {},
            hcas=tuple(hca_list),
            time_sync=TimeSync(
                source="chronyc",
                synchronized=True,
                offset_s=0.000012,
                stratum=3,
                reference="169.254.169.123",
                leap_status="Normal",
            ),
            docker=DockerFacts(
                server_version="27.5.1",
                storage_driver="overlay2",
                cgroup_driver="cgroupfs",
                cgroup_version="2",
                default_runtime="runc",
                runtimes=("io.containerd.runc.v2", "nvidia", "runc"),
                nvidia_runtime=True,
            ),
            kernel_log=KernelLog(available=True, source="dmesg", lines=0),
        )
    return out


def synthetic_software(image: str | None = None) -> SoftwareStack:
    """Plausible software stack of the pinned serving image (dry-run only).

    Versions follow SGLang 0.5.21's wheel pins and the lmsysorg/sglang v0.5.21
    Dockerfile that sglang-runtime:1.6.2 builds on (nixl is unpinned there).
    """
    return SoftwareStack(
        node="node0",
        hostname="synthetic-container",
        image=image,
        python="3.12.3",
        packages={
            "ai-dynamo": "1.6.2",
            "ai-dynamo-runtime": "1.6.2",
            "flashinfer-python": "0.6.18",
            "mooncake-transfer-engine-cuda13": "0.3.13",
            "nixl": "1.5.0",
            "nvidia-nccl-cu13": "2.30.7",
            "sgl-deep-ep": "0.1.2",
            "sglang": "0.5.21",
            "sglang-kernel": "0.4.7",
            "torch": "2.13.0+cu130",
            "transformers": "5.12.1",
            "triton": "3.7.1",
            "xgrammar": "0.2.7",
        },
        torch=TorchInfo(
            version="2.13.0+cu130",
            cuda="13.0",
            nccl="2.30.7",
            cudnn="91300",
            cuda_available=True,
            device_count=1,
        ),
        ucx=UcxInfo(version="1.19.0", revision="0000000"),
        cuda_toolkit="13.0.3",
        ibv_devices=tuple(f"mlx5_{i}" for i in range(8)),
        os_release={
            "ID": "ubuntu",
            "VERSION_ID": "24.04",
            "PRETTY_NAME": "Ubuntu 24.04.3 LTS",
        },
        glibc="2.39",
        notes=(Note(source="synthetic", error="dry-run: not measured"),),
    )


def synthetic_disk_read(node: str, path: str, gbs: float = 5.0) -> DiskRead:
    """A plausible NVMe O_DIRECT read (dry-run only)."""
    size = 5_000_000_000
    return DiskRead(
        node=node,
        hostname=f"synthetic-{node}",
        requested_path=path,
        path=f"{path.rstrip('/')}/model-00001-of-00160.safetensors",
        file_bytes=size,
        bytes=size,
        seconds=round(size / (gbs * 1e9), 6),
        gbs=gbs,
        method="dd_direct",
        direct=True,
        cache="bypassed",
        flushed=True,
        block_bytes=16 << 20,
    )
