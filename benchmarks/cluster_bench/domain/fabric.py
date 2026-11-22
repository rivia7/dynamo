# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream switch / fabric topology for the knowledge base. Pure: no I/O.

Inputs are the JSON documents of ``agents/fabric_probe.py`` (its module
docstring is the contract), validated with ``model_validate`` into the
``Fabric*`` / ``Trace*`` / ``Perf*`` report models below. Outputs:

* ``build_topology`` -> ``FabricTopology``: fabric kind, rails (rate, subnets,
  SMs), leaf/spine switches, which leaf every rail port hangs off, rail
  optimization, nodes per leaf, switch hops between our nodes per rail,
  leaf oversubscription, degraded links, LLDP/DCBX facts, plus ``evidence``
  (fact -> collection tier) and ``coverage`` (what could not be observed).
* ``round_robin_rounds`` + ``link_tests_from_perftest`` -> ``LinkMatrix``;
  ``analyze_link_matrix`` -> ``LinkFinding`` (weak links/rails, asymmetry,
  latency groups that expose same-leaf vs cross-spine pairs).
* ``switch_port_refs`` -> ``SwitchPortRef``; two ``SwitchSnapshot`` ->
  ``switch_load`` -> ``SwitchPortLoad`` (GB/s each way, utilization,
  PortXmitWait stall estimate, error deltas).
* ``topology_signature`` / ``topology_findings`` for the KB, and
  deterministic ``synthetic_*`` generators for the dry-run simulator.

Tiers: S sysfs (unprivileged), M MAD/SMP tools (smpquery, sminfo,
ibtracert), L LLDP, T unprivileged UDP traceroute, P perfquery/perftest,
D derived from other facts. Units: link rates Gb/s (``*_gbps``), bandwidth
GB/s = 1e9 bytes/s (``*_gbs``), seconds (``*_s``), microseconds (``*_us``).
Switch counters PortXmitData/PortRcvData count 4-byte words.

Imports only ``domain/base.py``, the stdlib and pydantic (``schema.py`` may
import this module).
"""

from __future__ import annotations

import ipaddress
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import Field

from benchmarks.cluster_bench.domain.base import Frozen, stable_hash

FABRIC_REPORT_VERSION = 1
Tier = Literal["S", "M", "L", "T", "P", "D"]
FabricKind = Literal["infiniband", "roce", "mixed", "unknown"]
SwitchRole = Literal["leaf", "spine", "unknown"]
CensusState = Literal["complete", "partial", "none"]
IB = "InfiniBand"
ETHERNET = "Ethernet"

LANE_GBPS = {
    "SDR": 2.5,
    "DDR": 5.0,
    "QDR": 10.0,
    "FDR10": 10.0,
    "FDR": 14.0,
    "EDR": 25.0,
    "HDR": 50.0,
    "NDR": 100.0,
    "XDR": 200.0,
}
WIDTH_LANES = {"1X": 1, "2X": 2, "4X": 4, "8X": 8, "12X": 12}
CENSUS_RANK = {"none": 0, "partial": 1, "complete": 2}
ROCE_V2_UDP_PORT = 4791
ROCE_V1_ETHERTYPE = 0x8915

# analyze_link_matrix thresholds.
WEAK_FRACTION = 0.8  # of the rail median or of line rate
ASYMMETRY_FRACTION = 0.2  # |a->b - b->a| / max
LAT_GAP_MIN_US = 0.15  # latency groups split at gaps at least this wide ...
LAT_GAP_REL = 0.05  # ... and at least this fraction of the lower latency

# PortCounters (non -x) widths; PortCountersExtended fields are all 64-bit.
BASIC_COUNTER_BITS = {
    "PortXmitData": 32,
    "PortRcvData": 32,
    "PortXmitPkts": 32,
    "PortRcvPkts": 32,
    "PortXmitWait": 32,
    "SymbolErrorCounter": 16,
    "LinkErrorRecoveryCounter": 8,
    "LinkDownedCounter": 8,
    "PortRcvErrors": 16,
    "PortRcvRemotePhysicalErrors": 16,
    "PortRcvSwitchRelayErrors": 16,
    "PortXmitDiscards": 16,
    "PortXmitConstraintErrors": 8,
    "PortRcvConstraintErrors": 8,
    "LocalLinkIntegrityErrors": 4,
    "ExcessiveBufferOverrunErrors": 4,
    "VL15Dropped": 16,
}
ERROR_COUNTERS = (
    "SymbolErrorCounter",
    "LinkErrorRecoveryCounter",
    "LinkDownedCounter",
    "PortRcvErrors",
    "PortRcvRemotePhysicalErrors",
    "PortRcvSwitchRelayErrors",
    "PortXmitDiscards",
    "PortXmitConstraintErrors",
    "PortRcvConstraintErrors",
    "LocalLinkIntegrityErrors",
    "ExcessiveBufferOverrunErrors",
    "VL15Dropped",
)
IB_WORD_BYTES = 4
# UNVERIFIED: the IB spec leaves the PortXmitWait tick implementation
# defined. ``stall_fraction_est`` assumes one tick = 1 ns; compare
# ``xmit_wait_per_s`` across load points rather than trusting the absolute.
XMIT_WAIT_TICK_NS_UNVERIFIED = 1.0
# A wrapped counter delta is plausible up to this multiple of line rate.
WRAP_RATE_SLACK = 1.1


def natural_key(text: str) -> list[Any]:
    """Sort key ordering node2 before node10 and mlx5_2 before mlx5_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def lanes(width: str | None) -> int | None:
    return WIDTH_LANES.get(width or "")


# --------------------------------------------------------------------------
# Agent report models (agents/fabric_probe.py JSON, version 1)
# --------------------------------------------------------------------------


class FabricNote(Frozen):
    source: str
    error: str
    target: str | None = None


class FabricLink(Frozen):
    """SMP PortInfo of one port (``Link`` in the agent contract)."""

    port: int
    state: str | None = None  # "Active" | "Down" | "Initialize" | "Armed"
    phys_state: str | None = None
    lid: int | None = None
    width_active: str | None = None
    width_supported: tuple[str, ...] = ()
    speed_active: str | None = None  # IB generation: "NDR", "XDR", ...
    speed_supported: tuple[str, ...] = ()
    rate_gbps: float | None = None
    max_rate_gbps: float | None = None

    @property
    def active(self) -> bool:
        return self.state == "Active"

    @property
    def max_lanes(self) -> int | None:
        counts = [lanes(w) for w in self.width_supported]
        known = [c for c in counts if c]
        return max(known) if known else None


class FabricPeer(Frozen):
    """SMP NodeInfo (+NodeDescription) of the node behind a port."""

    node_type: str | None = None  # "Switch" | "Channel Adapter" | "Router"
    node_guid: str | None = None
    port_guid: str | None = None
    system_guid: str | None = None
    num_ports: int | None = None
    local_port: int | None = None  # that node's port the SMP arrived on
    desc: str | None = None
    dev_id: str | None = None
    vendor_id: str | None = None

    @property
    def is_switch(self) -> bool:
        return self.node_type == "Switch"


class FabricNeighbor(FabricPeer):
    """The node behind our port; for a switch also its LID and the switch
    side of our cable."""

    lid: int | None = None
    link: FabricLink | None = None


class CensusPort(FabricLink):
    peer: FabricPeer | None = None


class FabricSwitchRecord(Frozen):
    guid: str
    desc: str | None = None
    lid: int | None = None
    num_ports: int | None = None
    via: str | None = None  # "mlx5_0:1": local port the census went through
    census: CensusState = "none"
    ports: tuple[CensusPort, ...] = ()


class FabricGid(Frozen):
    index: int
    gid: str
    type: str | None = None  # "IB/RoCE v1" | "RoCE v2"
    ndev: str | None = None


class FabricIpv4(Frozen):
    address: str
    prefixlen: int | None = None


class FabricPcie(Frozen):
    current_speed_gts: float | None = None
    current_width: int | None = None
    max_speed_gts: float | None = None
    max_width: int | None = None

    @property
    def degraded(self) -> bool | None:
        """Trained below the function's own maximum speed or width."""
        pairs = (
            (self.current_speed_gts, self.max_speed_gts),
            (self.current_width, self.max_width),
        )
        known = [(c, m) for c, m in pairs if c is not None and m is not None]
        if not known:
            return None
        return any(c < m for c, m in known)


class FabricPortReport(Frozen):
    """One HCA port as ``identify`` reports it."""

    hca: str
    port: int
    pci_bus_id: str | None = None
    link_layer: str = ""
    state: str = ""
    phys_state: str = ""
    rate: str = ""
    rate_gbps: float = 0.0
    width: str | None = None
    speed: str | None = None
    lid: int | None = None
    sm_lid: int | None = None
    sm_sl: int | None = None
    lmc: int | None = None
    cap_mask: str | None = None
    node_guid: str | None = None
    sys_image_guid: str | None = None
    node_desc: str | None = None
    fw_ver: str | None = None
    board_id: str | None = None
    port_guid: str | None = None
    subnet_prefix: str | None = None
    gids: tuple[FabricGid, ...] = ()
    roce_v2_gid_index: int | None = None
    roce_v2_ipv4: str | None = None
    netdev: str | None = None
    mtu: int | None = None
    operstate: str | None = None
    mac: str | None = None
    ipv4: tuple[FabricIpv4, ...] = ()
    active_mtu: int | None = None
    max_mtu: int | None = None
    pcie: FabricPcie | None = None
    neighbor: FabricNeighbor | None = None

    @property
    def key(self) -> str:
        return f"{self.hca}:{self.port}"

    @property
    def active(self) -> bool:
        return self.state.strip().startswith("4")

    @property
    def ip(self) -> FabricIpv4 | None:
        """The RoCE v2 GID's IPv4, else the netdev's first address."""
        if self.roce_v2_ipv4:
            for addr in self.ipv4:
                if addr.address == self.roce_v2_ipv4:
                    return addr
            return FabricIpv4(address=self.roce_v2_ipv4)
        return self.ipv4[0] if self.ipv4 else None


class FabricSm(Frozen):
    guid: str | None = None
    lid: int | None = None
    priority: int | None = None
    state: int | None = None
    state_name: str | None = None
    activity_count: int | None = None
    ports: tuple[str, ...] = ()


class SmpStats(Frozen):
    budget: int = 0
    used: int = 0
    failed: int = 0
    blocked: bool = False
    exhausted: bool = False


class FabricPfc(Frozen):
    willing: bool = False
    mbc: bool = False
    cap: int = 0
    enabled: tuple[int, ...] = ()


class FabricEts(Frozen):
    willing: bool = False
    cbs: bool = False
    max_tcs: int = 8
    prio_tc: tuple[int, ...] = ()
    tc_bw: tuple[int, ...] = ()
    tsa: tuple[int, ...] = ()


class FabricAppPriority(Frozen):
    priority: int
    selector: int  # 1 ethertype, 2 TCP, 3 UDP, 4 TCP/UDP, 5 DSCP
    protocol: int


class FabricVlan(Frozen):
    vid: int
    name: str | None = None


class LldpNeighbor(Frozen):
    source: str  # "lldpctl" | "raw"
    chassis_id: str | None = None
    chassis_id_subtype: str | None = None
    port_id: str | None = None
    port_id_subtype: str | None = None
    ttl_s: int | None = None
    port_descr: str | None = None
    system_name: str | None = None
    system_descr: str | None = None
    mgmt_addresses: tuple[str, ...] = ()
    port_vlan_id: int | None = None
    vlans: tuple[FabricVlan, ...] = ()
    max_frame_size: int | None = None
    pfc: FabricPfc | None = None
    ets: FabricEts | None = None
    app_priority: tuple[FabricAppPriority, ...] = ()
    unknown_tlvs: tuple[str, ...] = ()


class HostQos(Frozen):
    source: str = "mlnx_qos"
    trust: str | None = None
    pfc_enabled: tuple[int, ...] | None = None


class FabricNodeReport(Frozen):
    """``fabric_probe.py identify`` output.

    ``runner`` ("container", "host" or "container+host") and ``runner_tools``
    (mode -> tool -> installed) are added runner-side, not by the agent.
    """

    kind: Literal["fabric_identify"] = "fabric_identify"
    version: int = FABRIC_REPORT_VERSION
    hostname: str = ""
    tools: dict[str, bool] = Field(default_factory=dict)
    umad: bool = False
    smp: SmpStats = SmpStats()
    ports: tuple[FabricPortReport, ...] = ()
    sm: tuple[FabricSm, ...] = ()
    switches: dict[str, FabricSwitchRecord] = Field(default_factory=dict)
    lldp: dict[str, LldpNeighbor] = Field(default_factory=dict)
    qos: dict[str, HostQos] = Field(default_factory=dict)
    notes: tuple[FabricNote, ...] = ()
    elapsed_s: float = 0.0
    runner: str | None = None
    runner_tools: dict[str, dict[str, bool]] = Field(default_factory=dict)

    def port(self, key: str) -> FabricPortReport | None:
        for p in self.ports:
            if p.key == key:
                return p
        return None

    @property
    def mad_ok(self) -> bool:
        """MAD tools ran and the fabric answered at least one SMP."""
        return (
            self.umad
            and bool(self.tools.get("smpquery"))
            and not self.smp.blocked
            and any(p.neighbor is not None for p in self.ports)
        )


class TraceTarget(Frozen):
    rail: int
    hca: str
    port: int
    src_lid: int | None = None
    dst_lid: int | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    netdev: str | None = None
    src_node: str | None = None
    dst_node: str | None = None


class TraceHop(Frozen):
    ttl: int | None = None
    node_type: str | None = None
    guid: str | None = None
    desc: str | None = None
    lid: int | None = None
    in_port: int | None = None
    out_port: int | None = None
    ip: str | None = None
    rtt_ms: float | None = None
    icmp_type: int | None = None
    icmp_code: int | None = None


class TraceResult(Frozen):
    """One traced path (``fabric_trace`` results entry)."""

    target: TraceTarget
    method: Literal["ibtracert", "udp", "none"]
    hops: tuple[TraceHop, ...] = ()
    hop_count: int | None = None  # switches (IB) or routers (UDP)
    reached: bool = False
    error: str | None = None
    elapsed_s: float = 0.0


class TraceReport(Frozen):
    kind: Literal["fabric_trace"] = "fabric_trace"
    version: int = FABRIC_REPORT_VERSION
    hostname: str = ""
    results: tuple[TraceResult, ...] = ()
    notes: tuple[FabricNote, ...] = ()
    elapsed_s: float = 0.0


class PortCounterReading(Frozen):
    """One switch port as ``fabric_probe.py perfquery`` read it."""

    key: str | None = None
    hca: str
    port: int
    lid: int
    switch_port: int
    extended: bool = False
    wide: tuple[str, ...] = ()  # counters that are 64-bit
    counters: dict[str, int] = Field(default_factory=dict)
    mono_s: float | None = None
    wall_s: float | None = None
    error: str | None = None


class PerfqueryReport(Frozen):
    kind: Literal["fabric_perfquery"] = "fabric_perfquery"
    version: int = FABRIC_REPORT_VERSION
    hostname: str = ""
    ports: tuple[PortCounterReading, ...] = ()
    notes: tuple[FabricNote, ...] = ()
    elapsed_s: float = 0.0


class PerftestBw(Frozen):
    msg_bytes: int | None = None
    iterations: int | None = None
    unit: str | None = None  # as printed: "Gb/sec" (decimal) or "MB/sec" (MiB)
    bw_peak: float | None = None  # in ``unit``; 0.00 in -D mode
    bw_avg: float | None = None
    msg_rate_mpps: float | None = None
    bw_peak_gbs: float | None = None
    bw_avg_gbs: float | None = None


class PerftestLat(Frozen):
    msg_bytes: int | None = None
    iterations: int | None = None
    t_min_us: float | None = None
    t_max_us: float | None = None
    t_typical_us: float | None = None
    t_avg_us: float | None = None
    t_stdev_us: float | None = None
    p99_us: float | None = None
    p999_us: float | None = None


class PerftestRun(Frozen):
    rail: int
    hca: str
    port: int
    test: Literal["bw", "lat"]
    rc: int | None = None
    attempts: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    argv: tuple[str, ...] = ()
    bw: PerftestBw | None = None
    lat: PerftestLat | None = None


class PerftestReport(Frozen):
    kind: Literal["fabric_perftest"] = "fabric_perftest"
    version: int = FABRIC_REPORT_VERSION
    hostname: str = ""
    role: Literal["server", "client"]
    peer: str | None = None
    results: tuple[PerftestRun, ...] = ()
    notes: tuple[FabricNote, ...] = ()
    elapsed_s: float = 0.0


# --------------------------------------------------------------------------
# Derived models
# --------------------------------------------------------------------------


class FabricEvidence(Frozen):
    tier: Tier
    source: str


class RailPort(Frozen):
    """One node's HCA port on one rail, with where it plugs in upstream."""

    node: str
    rail: int
    hca: str
    port: int
    link_layer: str
    active: bool
    rate_gbps: float
    width: str | None = None
    speed: str | None = None
    lid: int | None = None
    subnet: str | None = None  # IB: "sm:<guid>" or "<prefix>/sm<lid>"; RoCE: CIDR
    subnet_prefix: str | None = None
    sm_lid: int | None = None
    sm_guid: str | None = None
    node_guid: str | None = None
    port_guid: str | None = None
    netdev: str | None = None
    ip: str | None = None
    ip_prefixlen: int | None = None
    gid_index: int | None = None
    mtu: int | None = None
    active_mtu: int | None = None
    switch_id: str | None = None  # IB switch GUID or LLDP chassis id
    switch_lid: int | None = None
    switch_port: int | None = None
    switch_port_name: str | None = None  # LLDP port id
    switch_desc: str | None = None
    attach_tier: Tier | None = None
    direct_peer: str | None = None  # CA node GUID when cabled back-to-back
    pcie: FabricPcie | None = None


class LeafAttachment(Frozen):
    switch_id: str
    port: int | None = None
    port_name: str | None = None
    rail: int
    tier: Tier


class RailSummary(Frozen):
    rail: int
    link_layer: str  # "InfiniBand" | "Ethernet" | "mixed" | ""
    ports: int
    active: int
    rate_gbps: float | None = None  # fastest active port: the rail's line rate
    min_rate_gbps: float | None = None
    subnets: tuple[str, ...] = ()
    sm_guids: tuple[str, ...] = ()
    leaves: tuple[str, ...] = ()


class SwitchLink(Frozen):
    port: int
    rate_gbps: float | None = None
    width: str | None = None
    speed: str | None = None
    peer_id: str | None = None
    peer_port: int | None = None


class SwitchSummary(Frozen):
    id: str
    role: SwitchRole
    desc: str | None = None
    lid: int | None = None
    num_ports: int | None = None
    census: CensusState = "none"
    tier: Tier
    rails: tuple[int, ...] = ()  # rails of our HCAs plugged in (leaves)
    our_nodes: tuple[str, ...] = ()
    our_ports: int = 0
    host_ports_active: int | None = None  # census: active ports facing CAs
    other_hosts: int | None = None  # ... of which not ours
    uplinks_active: int | None = None  # census: active ports facing switches
    unknown_active: int | None = None  # active ports whose peer is unknown
    host_capacity_gbps: float | None = None
    uplink_capacity_gbps: float | None = None
    oversubscription: float | None = None  # host-facing / uplink capacity
    peers: tuple[str, ...] = ()  # leaf: its spines; spine: our leaves
    uplinks: tuple[SwitchLink, ...] = ()
    degraded_ports: tuple[int, ...] = ()


DegradedKind = Literal[
    "port_down",
    "rate_below_peers",
    "width_below_supported",
    "pcie_degraded",
    "uplink_below_peers",
]


class DegradedLink(Frozen):
    kind: DegradedKind
    node: str | None = None
    hca: str | None = None
    rail: int | None = None
    switch_id: str | None = None
    switch_port: int | None = None
    value: float | None = None
    reference: float | None = None
    detail: str
    tier: Tier


class SmSummary(Frozen):
    guid: str | None = None
    lid: int | None = None
    priority: int | None = None
    state_name: str | None = None
    rails: tuple[int, ...] = ()
    nodes: tuple[str, ...] = ()


class LldpSummary(Frozen):
    node: str
    rail: int
    hca: str
    netdev: str | None = None
    source: str | None = None
    system_name: str | None = None
    chassis_id: str | None = None
    port_id: str | None = None
    port_descr: str | None = None
    mgmt_address: str | None = None
    port_vlan_id: int | None = None
    max_frame_size: int | None = None
    pfc_enabled: tuple[int, ...] | None = None
    pfc_willing: bool | None = None
    ets_tc_bw: tuple[int, ...] | None = None
    roce_app_priority: int | None = None
    host_trust: str | None = None
    host_pfc_enabled: tuple[int, ...] | None = None


class FabricTopology(Frozen):
    kind: FabricKind
    nodes: tuple[str, ...]
    rails: tuple[RailSummary, ...] = ()
    ports: tuple[RailPort, ...] = ()
    switches: dict[str, SwitchSummary] = Field(default_factory=dict)
    leaf_of: dict[str, dict[str, LeafAttachment]] = Field(default_factory=dict)
    leaf_groups: dict[int, tuple[tuple[str, ...], ...]] = Field(default_factory=dict)
    nodes_per_leaf: int | None = None
    rail_optimized: bool | None = None
    rails_isolated: bool | None = None
    tiers: int | None = None  # switch tiers between our nodes ((hops+1)//2)
    max_hops: int | None = None
    hops: dict[int, dict[str, dict[str, int]]] = Field(default_factory=dict)
    hop_method: dict[int, str] = Field(default_factory=dict)
    sms: tuple[SmSummary, ...] = ()
    degraded: tuple[DegradedLink, ...] = ()
    lldp: tuple[LldpSummary, ...] = ()
    evidence: dict[str, FabricEvidence] = Field(default_factory=dict)
    coverage: tuple[str, ...] = ()

    def port(self, node: str, rail: int) -> RailPort | None:
        for p in self.ports:
            if p.node == node and p.rail == rail:
                return p
        return None

    def rail_ports(self, rail: int) -> tuple[RailPort, ...]:
        return tuple(p for p in self.ports if p.rail == rail)

    @property
    def rail_indices(self) -> tuple[int, ...]:
        return tuple(r.rail for r in self.rails)

    def by_role(self, role: SwitchRole) -> tuple[SwitchSummary, ...]:
        return tuple(s for _, s in sorted(self.switches.items()) if s.role == role)


# --------------------------------------------------------------------------
# Merging reports (container + host runs, census passes)
# --------------------------------------------------------------------------


def _port_richness(port: FabricPortReport) -> tuple[int, int, int, int]:
    neighbor = port.neighbor
    return (
        int(neighbor is not None),
        int(neighbor is not None and neighbor.link is not None),
        int(bool(port.ipv4)),
        int(port.active_mtu is not None),
    )


def merge_switch(
    a: FabricSwitchRecord | None, b: FabricSwitchRecord | None
) -> FabricSwitchRecord:
    """The better census of one switch (complete > partial > none, then more
    ports), with identity fields filled from the other record."""
    if a is None or b is None:
        record = a or b
        if record is None:
            raise ValueError("merge_switch needs at least one record")
        return record
    best, other = (a, b)
    if (CENSUS_RANK[b.census], len(b.ports)) > (CENSUS_RANK[a.census], len(a.ports)):
        best, other = b, a
    return best.model_copy(
        update={
            "desc": best.desc or other.desc,
            "lid": best.lid if best.lid is not None else other.lid,
            "num_ports": best.num_ports
            if best.num_ports is not None
            else other.num_ports,
        }
    )


def merge_reports(
    primary: FabricNodeReport,
    secondary: FabricNodeReport,
    *,
    runner: str | None = None,
) -> FabricNodeReport:
    """Union of two ``identify`` runs on the same node.

    Per port the richer record wins (neighbor known, switch side known, IPs,
    MTU), ties keep ``primary``; tools and umad are OR-ed; switches keep the
    better census; LLDP/QoS from ``primary`` win; SMs and notes are united.
    """
    ports: list[FabricPortReport] = []
    by_key = {p.key: p for p in secondary.ports}
    seen = set()
    for port in primary.ports:
        other = by_key.get(port.key)
        if other is not None and _port_richness(other) > _port_richness(port):
            port = other
        ports.append(port)
        seen.add(port.key)
    ports += [p for p in secondary.ports if p.key not in seen]
    switches = dict(primary.switches)
    for guid, record in secondary.switches.items():
        switches[guid] = merge_switch(switches.get(guid), record)
    sms: dict[tuple[str | None, int | None], FabricSm] = {}
    for sm in (*primary.sm, *secondary.sm):
        key = (sm.guid, sm.lid)
        if key in sms:
            merged = tuple(sorted(set(sms[key].ports) | set(sm.ports), key=natural_key))
            sms[key] = sms[key].model_copy(update={"ports": merged})
        else:
            sms[key] = sm
    notes = list(primary.notes)
    notes += [n for n in secondary.notes if n not in notes]
    tools = dict(primary.tools)
    for name, present in secondary.tools.items():
        tools[name] = tools.get(name, False) or present
    smp = SmpStats(
        budget=primary.smp.budget + secondary.smp.budget,
        used=primary.smp.used + secondary.smp.used,
        failed=primary.smp.failed + secondary.smp.failed,
        blocked=primary.smp.blocked and secondary.smp.blocked,
        exhausted=primary.smp.exhausted or secondary.smp.exhausted,
    )
    return primary.model_copy(
        update={
            "hostname": primary.hostname or secondary.hostname,
            "tools": tools,
            "umad": primary.umad or secondary.umad,
            "smp": smp,
            "ports": tuple(ports),
            "sm": tuple(sms.values()),
            "switches": switches,
            "lldp": {**secondary.lldp, **primary.lldp},
            "qos": {**secondary.qos, **primary.qos},
            "notes": tuple(notes),
            "elapsed_s": primary.elapsed_s + secondary.elapsed_s,
            "runner": runner if runner is not None else primary.runner,
            "runner_tools": {**secondary.runner_tools, **primary.runner_tools},
        }
    )


def census_plan(reports: Mapping[str, FabricNodeReport]) -> dict[str, tuple[str, ...]]:
    """Which node censuses which neighbor switch, through which local port.

    Every distinct neighbor switch (IB) not yet censused anywhere is assigned
    to exactly one node that sees it and has working MAD tools, balancing the
    number of switches per node (ties: lower node). Returns node -> local
    port keys ("mlx5_0:1") to pass as ``--census-only``.
    """
    nodes = sorted(reports, key=natural_key)
    done = {
        guid
        for r in reports.values()
        for guid, rec in r.switches.items()
        if rec.census == "complete"
    }
    seen_by: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for node in nodes:
        report = reports[node]
        if not report.mad_ok:
            continue
        for port in report.ports:
            nb = port.neighbor
            if nb is None or not nb.is_switch or not nb.node_guid:
                continue
            if nb.node_guid in done:
                continue
            if nb.node_guid not in seen_by:
                order.append(nb.node_guid)
            seen = seen_by.setdefault(nb.node_guid, [])
            if all(n != node for n, _ in seen):
                seen.append((node, port.key))
    load: Counter[str] = Counter()
    plan: dict[str, list[str]] = defaultdict(list)
    for guid in order:
        node, key = min(seen_by[guid], key=lambda c: (load[c[0]], nodes.index(c[0])))
        load[node] += 1
        plan[node].append(key)
    return {node: tuple(plan[node]) for node in nodes if node in plan}


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------


def default_rail_map(
    reports: Mapping[str, FabricNodeReport],
) -> dict[str, dict[str, int]]:
    """node -> HCA -> rail: the natural order of every HCA name that has an
    active port on some node, so one HCA name is one rail on every node and
    a port that is down on one node still shows up (as not active) there."""
    names = sorted(
        {p.hca for r in reports.values() for p in r.ports if p.active}, key=natural_key
    )
    index = {hca: rail for rail, hca in enumerate(names)}
    return {
        node: {p.hca: index[p.hca] for p in report.ports if p.hca in index}
        for node, report in reports.items()
    }


def _hca_port(report: FabricNodeReport, hca: str) -> FabricPortReport | None:
    ports = sorted((p for p in report.ports if p.hca == hca), key=lambda p: p.port)
    active = [p for p in ports if p.active]
    if active:
        return active[0]
    return ports[0] if ports else None


def _sm_by_port(report: FabricNodeReport) -> dict[str, FabricSm]:
    return {key: sm for sm in report.sm for key in sm.ports}


def _ip_network(ip: str | None, prefixlen: int | None) -> str | None:
    if not ip or prefixlen is None:
        return None
    try:
        return str(ipaddress.ip_interface(f"{ip}/{prefixlen}").network)
    except ValueError:
        return None


def _rail_port(
    node: str, rail: int, port: FabricPortReport, report: FabricNodeReport
) -> RailPort:
    sm = _sm_by_port(report).get(port.key)
    ip = port.ip
    fields: dict[str, Any] = {
        "node": node,
        "rail": rail,
        "hca": port.hca,
        "port": port.port,
        "link_layer": port.link_layer,
        "active": port.active,
        "rate_gbps": port.rate_gbps,
        "width": port.width,
        "speed": port.speed,
        "lid": port.lid,
        "subnet_prefix": port.subnet_prefix,
        "sm_lid": port.sm_lid,
        "sm_guid": sm.guid if sm else None,
        "node_guid": port.node_guid,
        "port_guid": port.port_guid,
        "netdev": port.netdev,
        "ip": ip.address if ip else None,
        "ip_prefixlen": ip.prefixlen if ip else None,
        "gid_index": port.roce_v2_gid_index,
        "mtu": port.mtu,
        "active_mtu": port.active_mtu,
        "pcie": port.pcie,
    }
    if port.link_layer == IB:
        if sm and sm.guid:
            fields["subnet"] = f"sm:{sm.guid}"
        elif port.subnet_prefix:
            fields["subnet"] = f"{port.subnet_prefix}/sm{port.sm_lid}"
        nb = port.neighbor
        if nb is not None and nb.is_switch and nb.node_guid:
            fields.update(
                switch_id=nb.node_guid,
                switch_lid=nb.lid,
                switch_port=nb.local_port,
                switch_desc=nb.desc,
                attach_tier="M",
            )
        elif nb is not None and nb.node_guid:
            fields["direct_peer"] = nb.node_guid
    elif port.link_layer == ETHERNET:
        fields["subnet"] = _ip_network(fields["ip"], fields["ip_prefixlen"])
        lldp = report.lldp.get(port.netdev or "")
        if lldp is not None and lldp.chassis_id:
            fields.update(
                switch_id=lldp.chassis_id,
                switch_port_name=lldp.port_id,
                switch_desc=lldp.system_name,
                attach_tier="L",
            )
    return RailPort(**fields)


def _rail_ports(
    reports: Mapping[str, FabricNodeReport],
    nodes: Sequence[str],
    rail_of: Mapping[str, Mapping[str, int]] | None,
    coverage: list[str],
) -> list[RailPort]:
    ports: list[RailPort] = []
    defaults = default_rail_map(reports)
    for node in nodes:
        report = reports[node]
        mapping = rail_of.get(node) if rail_of is not None else None
        if mapping is None:
            mapping = defaults.get(node, {})
        if not any(p.active for p in report.ports):
            coverage.append(f"{node}: no active RDMA port in the fabric report")
        for hca, rail in sorted(mapping.items(), key=lambda kv: kv[1]):
            port = _hca_port(report, hca)
            if port is None:
                coverage.append(f"{node}: rail {rail} HCA {hca} missing from report")
                continue
            ports.append(_rail_port(node, rail, port, report))
    return ports


def _fabric_kind(ports: Sequence[RailPort]) -> FabricKind:
    layers = {p.link_layer for p in ports if p.link_layer in (IB, ETHERNET)}
    if layers == {IB}:
        return "infiniband"
    if layers == {ETHERNET}:
        return "roce"
    return "mixed" if layers else "unknown"


def _guid_aliases(reports: Iterable[FabricNodeReport]) -> dict[str, str]:
    """Switch port GUID -> node GUID (ibtracert prints port GUIDs)."""
    aliases: dict[str, str] = {}

    def add(peer: FabricPeer | None) -> None:
        if peer is not None and peer.node_guid:
            aliases[peer.node_guid] = peer.node_guid
            if peer.port_guid:
                aliases.setdefault(peer.port_guid, peer.node_guid)

    for report in reports:
        for port in report.ports:
            add(port.neighbor)
        for record in report.switches.values():
            for census_port in record.ports:
                add(census_port.peer)
    return aliases


def _census_records(
    reports: Iterable[FabricNodeReport],
) -> dict[str, FabricSwitchRecord]:
    records: dict[str, FabricSwitchRecord] = {}
    for report in reports:
        for guid, record in report.switches.items():
            records[guid] = merge_switch(records.get(guid), record)
    return records


def _our_guids(reports: Iterable[FabricNodeReport]) -> set[str]:
    return {p.node_guid for r in reports for p in r.ports if p.node_guid}


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _rail_summaries(
    ports: Sequence[RailPort], leaves_by_rail: Mapping[int, set[str]]
) -> tuple[RailSummary, ...]:
    summaries = []
    for rail in sorted({p.rail for p in ports}):
        mine = [p for p in ports if p.rail == rail]
        layers = {p.link_layer for p in mine if p.link_layer}
        rates = [p.rate_gbps for p in mine if p.active and p.rate_gbps > 0]
        summaries.append(
            RailSummary(
                rail=rail,
                link_layer=layers.pop()
                if len(layers) == 1
                else ("mixed" if layers else ""),
                ports=len(mine),
                active=sum(1 for p in mine if p.active),
                rate_gbps=max(rates) if rates else None,
                min_rate_gbps=min(rates) if rates else None,
                subnets=tuple(sorted({p.subnet for p in mine if p.subnet})),
                sm_guids=tuple(sorted({p.sm_guid for p in mine if p.sm_guid})),
                leaves=tuple(sorted(leaves_by_rail.get(rail, set()))),
            )
        )
    return tuple(summaries)


def _rails_isolated(ports: Sequence[RailPort]) -> bool | None:
    """Do rails live in separate subnets? ``None`` when it cannot be told."""
    by_rail: dict[int, set[str]] = defaultdict(set)
    for p in ports:
        if p.subnet:
            by_rail[p.rail].add(p.subnet)
    if len(by_rail) < 2:
        return None
    rails = sorted(by_rail)
    shared = any(
        by_rail[a] & by_rail[b] for i, a in enumerate(rails) for b in rails[i + 1 :]
    )
    if not shared:
        return True
    ib = [p for p in ports if p.link_layer == IB]
    if ib and all(p.sm_guid for p in ib):
        return False  # the same master SM serves several rails: one subnet
    lids: dict[int, set[int]] = defaultdict(set)
    for p in ib:
        if p.lid is not None:
            lids[p.lid].add(p.rail)
    if any(len(r) > 1 for r in lids.values()):
        return True  # one LID on two rails: only possible in separate subnets
    if not ib:
        return False  # RoCE rails sharing an IP subnet
    return None


def _census_peer_id(peer: FabricPeer | None, aliases: Mapping[str, str]) -> str | None:
    if peer is None or not peer.node_guid:
        return None
    return aliases.get(peer.node_guid, peer.node_guid)


def _summarize_census(
    record: FabricSwitchRecord,
    our: set[str],
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    host = uplink = unknown = other = 0
    host_cap = uplink_cap = 0.0
    links: list[SwitchLink] = []
    for port in record.ports:
        if not port.active:
            continue
        rate = port.rate_gbps or 0.0
        peer = port.peer
        if peer is None or peer.node_type is None:
            unknown += 1
        elif peer.is_switch:
            uplink += 1
            uplink_cap += rate
            links.append(
                SwitchLink(
                    port=port.port,
                    rate_gbps=port.rate_gbps,
                    width=port.width_active,
                    speed=port.speed_active,
                    peer_id=_census_peer_id(peer, aliases),
                    peer_port=peer.local_port,
                )
            )
        else:
            host += 1
            host_cap += rate
            if peer.node_guid not in our:
                other += 1
    rates = [u.rate_gbps for u in links if u.rate_gbps]
    best = max(rates) if rates else None
    degraded = tuple(
        u.port for u in links if best and u.rate_gbps and u.rate_gbps < best
    )
    return {
        "host_ports_active": host,
        "other_hosts": other,
        "uplinks_active": uplink,
        "unknown_active": unknown,
        "host_capacity_gbps": host_cap,
        "uplink_capacity_gbps": uplink_cap,
        "oversubscription": (host_cap / uplink_cap) if uplink_cap > 0 else None,
        "uplinks": tuple(links),
        "degraded_ports": degraded,
    }


def _switches(
    ports: Sequence[RailPort],
    records: Mapping[str, FabricSwitchRecord],
    traces: Sequence[TraceResult],
    our: set[str],
    aliases: Mapping[str, str],
) -> dict[str, SwitchSummary]:
    leaves: dict[str, list[RailPort]] = defaultdict(list)
    for p in ports:
        if p.switch_id:
            leaves[p.switch_id].append(p)
    switches: dict[str, SwitchSummary] = {}
    spine_leaves: dict[str, set[str]] = defaultdict(set)
    for leaf_id, attached in leaves.items():
        record = records.get(leaf_id)
        first = attached[0]
        fields: dict[str, Any] = {
            "id": leaf_id,
            "role": "leaf",
            "desc": first.switch_desc,
            "lid": next((p.switch_lid for p in attached if p.switch_lid), None),
            "tier": first.attach_tier or "D",
            "rails": tuple(sorted({p.rail for p in attached})),
            "our_nodes": tuple(sorted({p.node for p in attached}, key=natural_key)),
            "our_ports": len(attached),
        }
        if record is not None:
            fields.update(
                desc=record.desc or first.switch_desc,
                lid=record.lid if record.lid is not None else fields["lid"],
                num_ports=record.num_ports,
                census=record.census,
            )
            if record.ports:
                fields.update(_summarize_census(record, our, aliases))
                fields["peers"] = tuple(
                    sorted({u.peer_id for u in fields["uplinks"] if u.peer_id})
                )
                for spine in fields["peers"]:
                    spine_leaves[spine].add(leaf_id)
        switches[leaf_id] = SwitchSummary(**fields)
    for result in traces:
        for hop in result.hops:
            if hop.guid is None:
                continue
            guid = aliases.get(hop.guid, hop.guid)
            if guid in leaves:
                continue
            spine_leaves.setdefault(guid, set())
            if guid not in switches:
                switches[guid] = SwitchSummary(
                    id=guid, role="spine", desc=hop.desc, lid=hop.lid, tier="M"
                )
    for spine, leaf_ids in spine_leaves.items():
        if spine in leaves:
            continue  # two of our leaves cabled to each other
        record = records.get(spine)
        current = switches.get(spine)
        switches[spine] = SwitchSummary(
            id=spine,
            role="spine",
            desc=(current.desc if current else None) or _peer_desc(records, spine),
            lid=current.lid if current else (record.lid if record else None),
            num_ports=record.num_ports if record else None,
            census=record.census if record else "none",
            tier="M",
            peers=tuple(sorted(leaf_ids)),
        )
    return dict(sorted(switches.items()))


def _peer_desc(records: Mapping[str, FabricSwitchRecord], guid: str) -> str | None:
    for record in records.values():
        for port in record.ports:
            if port.peer is not None and port.peer.node_guid == guid and port.peer.desc:
                return port.peer.desc
    return None


def _leaf_groups(
    ports: Sequence[RailPort], nodes: Sequence[str]
) -> dict[int, tuple[tuple[str, ...], ...]]:
    groups: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for p in ports:
        if p.switch_id:
            groups[p.rail][p.switch_id].add(p.node)
    order = {n: i for i, n in enumerate(nodes)}
    result = {}
    for rail, by_leaf in sorted(groups.items()):
        tuples = [
            tuple(sorted(members, key=order.__getitem__))
            for members in by_leaf.values()
        ]
        result[rail] = tuple(sorted(tuples, key=lambda t: order[t[0]]))
    return result


def _rail_optimized(
    ports: Sequence[RailPort], switches: Mapping[str, SwitchSummary]
) -> bool | None:
    rails = {p.rail for p in ports}
    leaves = [s for s in switches.values() if s.role == "leaf"]
    if len(rails) < 2 or not leaves:
        return None
    return all(len(s.rails) == 1 for s in leaves)


def _trace_hops(
    traces: Sequence[TraceResult],
    coverage: list[str],
) -> tuple[dict[int, dict[tuple[str, str], int]], dict[int, set[str]]]:
    hops: dict[int, dict[tuple[str, str], int]] = defaultdict(dict)
    methods: dict[int, set[str]] = defaultdict(set)
    unnamed = failed = 0
    for result in traces:
        target = result.target
        if not target.src_node or not target.dst_node:
            unnamed += 1
            continue
        if not result.reached or result.hop_count is None:
            failed += 1
            continue
        if result.method == "udp" and result.hop_count == 0:
            continue  # same L2 segment: switches are invisible to L3
        pair = (target.src_node, target.dst_node)
        hops[target.rail][pair] = result.hop_count
        methods[target.rail].add(result.method)
    if unnamed:
        coverage.append(f"traces: {unnamed} result(s) without node names ignored")
    if failed:
        coverage.append(f"traces: {failed} of {len(traces)} path(s) not traced")
    return hops, methods


def _inferred_hops(
    ports: Sequence[RailPort], switches: Mapping[str, SwitchSummary]
) -> dict[int, dict[tuple[str, str], int]]:
    hops: dict[int, dict[tuple[str, str], int]] = defaultdict(dict)
    by_rail: dict[int, list[RailPort]] = defaultdict(list)
    guid_node = {p.node_guid: p.node for p in ports if p.node_guid}
    for p in ports:
        by_rail[p.rail].append(p)
    for rail, members in by_rail.items():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                value = None
                if a.switch_id and a.switch_id == b.switch_id:
                    value = 1
                elif a.direct_peer and guid_node.get(a.direct_peer) == b.node:
                    value = 0
                elif a.switch_id and b.switch_id:
                    sa, sb = switches.get(a.switch_id), switches.get(b.switch_id)
                    if sa and sb and set(sa.peers) & set(sb.peers):
                        value = 3
                if value is not None:
                    hops[rail][(a.node, b.node)] = value
    return hops


def _hops_matrix(
    traced: Mapping[int, Mapping[tuple[str, str], int]],
    inferred: Mapping[int, Mapping[tuple[str, str], int]],
    methods: Mapping[int, set[str]],
) -> tuple[dict[int, dict[str, dict[str, int]]], dict[int, str], set[str]]:
    """(rail -> a -> b -> hops, rail -> method label, every method used);
    traced pairs win over inferred ones; the matrix is filled symmetrically."""
    matrix: dict[int, dict[str, dict[str, int]]] = {}
    method_of: dict[int, str] = {}
    every: set[str] = set()
    for rail in sorted(set(traced) | set(inferred)):
        cells: dict[str, dict[str, int]] = defaultdict(dict)
        used: set[str] = set(methods.get(rail, set()))
        pairs = dict(inferred.get(rail, {}))
        if pairs and not set(pairs) <= set(traced.get(rail, {})):
            used.add("inferred")
        pairs.update(traced.get(rail, {}))
        for (a, b), value in pairs.items():
            cells[a][b] = value
            cells[b].setdefault(a, value)
        matrix[rail] = {
            a: dict(sorted(row.items())) for a, row in sorted(cells.items())
        }
        every |= used
        method_of[rail] = next(iter(used)) if len(used) == 1 else "mixed"
    return matrix, method_of, every


def _unknown_pairs(
    ports: Sequence[RailPort], hops: Mapping[int, Mapping[str, Mapping[str, int]]]
) -> int:
    """Pairs of active ports on one rail with no hop count."""
    missing = 0
    by_rail: dict[int, list[RailPort]] = defaultdict(list)
    for p in ports:
        if p.active:
            by_rail[p.rail].append(p)
    for rail, members in by_rail.items():
        row = hops.get(rail, {})
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if b.node not in row.get(a.node, {}):
                    missing += 1
    return missing


def _degraded(
    ports: Sequence[RailPort],
    reports: Mapping[str, FabricNodeReport],
    switches: Mapping[str, SwitchSummary],
) -> tuple[DegradedLink, ...]:
    found: list[DegradedLink] = []
    best: dict[int, float] = {}
    for p in ports:
        if p.active and p.rate_gbps > 0:
            best[p.rail] = max(best.get(p.rail, 0.0), p.rate_gbps)
    for p in ports:
        where = f"{p.node}/{p.hca}"
        if not p.active:
            found.append(
                DegradedLink(
                    kind="port_down",
                    node=p.node,
                    hca=p.hca,
                    rail=p.rail,
                    detail=f"{where} (rail {p.rail}) is not ACTIVE",
                    tier="S",
                )
            )
            continue
        top = best.get(p.rail)
        if top and p.rate_gbps < top:
            found.append(
                DegradedLink(
                    kind="rate_below_peers",
                    node=p.node,
                    hca=p.hca,
                    rail=p.rail,
                    value=p.rate_gbps,
                    reference=top,
                    detail=(
                        f"{where} runs at {p.rate_gbps:g} Gb/s while rail {p.rail} "
                        f"peers run at {top:g} Gb/s"
                    ),
                    tier="S",
                )
            )
        report_port = reports[p.node].port(f"{p.hca}:{p.port}")
        link = (
            report_port.neighbor.link if report_port and report_port.neighbor else None
        )
        if link is not None and link.active:
            active_lanes, max_lanes = lanes(link.width_active), link.max_lanes
            if active_lanes and max_lanes and active_lanes < max_lanes:
                found.append(
                    DegradedLink(
                        kind="width_below_supported",
                        node=p.node,
                        hca=p.hca,
                        rail=p.rail,
                        switch_id=p.switch_id,
                        switch_port=p.switch_port,
                        value=float(active_lanes),
                        reference=float(max_lanes),
                        detail=(
                            f"{where}: switch port {p.switch_port} trained "
                            f"{link.width_active} of {max_lanes}X supported"
                        ),
                        tier="M",
                    )
                )
        if p.pcie is not None and p.pcie.degraded:
            pcie = p.pcie
            found.append(
                DegradedLink(
                    kind="pcie_degraded",
                    node=p.node,
                    hca=p.hca,
                    rail=p.rail,
                    value=pcie.current_width,
                    reference=pcie.max_width,
                    detail=(
                        f"{where} PCIe at {_pcie_text(pcie.current_speed_gts, pcie.current_width)}"
                        f" of {_pcie_text(pcie.max_speed_gts, pcie.max_width)}"
                    ),
                    tier="S",
                )
            )
    for switch in switches.values():
        if switch.role != "leaf" or not switch.degraded_ports:
            continue
        rates = [u.rate_gbps for u in switch.uplinks if u.rate_gbps]
        top = max(rates) if rates else None
        for link in switch.uplinks:
            if link.port in switch.degraded_ports:
                found.append(
                    DegradedLink(
                        kind="uplink_below_peers",
                        switch_id=switch.id,
                        switch_port=link.port,
                        value=link.rate_gbps,
                        reference=top,
                        detail=(
                            f"leaf {switch.desc or switch.id} uplink port {link.port} "
                            f"at {link.rate_gbps:g} Gb/s vs {top:g} Gb/s on other uplinks"
                        ),
                        tier="M",
                    )
                )
    return tuple(found)


def _pcie_text(speed: float | None, width: int | None) -> str:
    speed_text = "?" if speed is None else f"{speed:g}"
    return f"{speed_text} GT/s x{'?' if width is None else width}"


def _roce_app_priority(lldp: LldpNeighbor) -> int | None:
    for app in lldp.app_priority:
        if app.selector in (3, 4) and app.protocol == ROCE_V2_UDP_PORT:
            return app.priority
    for app in lldp.app_priority:
        if app.selector == 1 and app.protocol == ROCE_V1_ETHERTYPE:
            return app.priority
    return None


def _lldp_summaries(
    ports: Sequence[RailPort], reports: Mapping[str, FabricNodeReport]
) -> tuple[LldpSummary, ...]:
    out = []
    for p in ports:
        if p.link_layer != ETHERNET or not p.netdev:
            continue
        report = reports[p.node]
        lldp = report.lldp.get(p.netdev)
        qos = report.qos.get(p.netdev)
        if lldp is None and qos is None:
            continue
        fields: dict[str, Any] = {
            "node": p.node,
            "rail": p.rail,
            "hca": p.hca,
            "netdev": p.netdev,
        }
        if lldp is not None:
            fields.update(
                source=lldp.source,
                system_name=lldp.system_name,
                chassis_id=lldp.chassis_id,
                port_id=lldp.port_id,
                port_descr=lldp.port_descr,
                mgmt_address=lldp.mgmt_addresses[0] if lldp.mgmt_addresses else None,
                port_vlan_id=lldp.port_vlan_id,
                max_frame_size=lldp.max_frame_size,
                pfc_enabled=lldp.pfc.enabled if lldp.pfc else None,
                pfc_willing=lldp.pfc.willing if lldp.pfc else None,
                ets_tc_bw=lldp.ets.tc_bw if lldp.ets else None,
                roce_app_priority=_roce_app_priority(lldp),
            )
        if qos is not None:
            fields.update(host_trust=qos.trust, host_pfc_enabled=qos.pfc_enabled)
        out.append(LldpSummary(**fields))
    return tuple(out)


def _sm_summaries(
    ports: Sequence[RailPort], reports: Mapping[str, FabricNodeReport]
) -> tuple[SmSummary, ...]:
    rail_of = {(p.node, f"{p.hca}:{p.port}"): p.rail for p in ports}
    merged: dict[tuple[str | None, int | None], dict[str, Any]] = {}
    for node, report in reports.items():
        for sm in report.sm:
            entry = merged.setdefault(
                (sm.guid, sm.lid),
                {
                    "guid": sm.guid,
                    "lid": sm.lid,
                    "priority": sm.priority,
                    "state_name": sm.state_name,
                    "rails": set(),
                    "nodes": set(),
                },
            )
            entry["nodes"].add(node)
            for key in sm.ports:
                if (node, key) in rail_of:
                    entry["rails"].add(rail_of[(node, key)])
    return tuple(
        SmSummary(
            **{
                **e,
                "rails": tuple(sorted(e["rails"])),
                "nodes": tuple(sorted(e["nodes"], key=natural_key)),
            }
        )
        for _, e in sorted(merged.items(), key=lambda kv: str(kv[0]))
    )


def _coverage_notes(
    reports: Mapping[str, FabricNodeReport],
    nodes: Sequence[str],
    ports: Sequence[RailPort],
    switches: Mapping[str, SwitchSummary],
) -> list[str]:
    notes: list[str] = []
    for node in nodes:
        report = reports[node]
        sources = {(n.source, n.error) for n in report.notes}
        if not report.ports:
            notes.append(f"{node}: no fabric data (agent failed or no RDMA devices)")
            continue
        if any(s == "umad" for s, _ in sources):
            notes.append(f"{node}: MAD tier unavailable (no umad device)")
        elif report.smp.blocked:
            notes.append(f"{node}: MAD tier unavailable (SMPs blocked)")
        elif any(s == "smpquery" and e == "not installed" for s, e in sources):
            notes.append(f"{node}: MAD tier unavailable (infiniband-diags missing)")
    unknown = [
        p for p in ports if p.active and p.switch_id is None and not p.direct_peer
    ]
    if unknown:
        notes.append(
            f"leaf identity unknown for {len(unknown)} of {len(ports)} rail ports"
        )
    leaves = [s for s in switches.values() if s.role == "leaf" and s.tier == "M"]
    if leaves:
        complete = sum(1 for s in leaves if s.census == "complete")
        partial = sum(1 for s in leaves if s.census == "partial")
        if complete < len(leaves):
            notes.append(
                f"census: {complete} of {len(leaves)} leaves complete, {partial} partial"
            )
    eth = [p for p in ports if p.link_layer == ETHERNET and p.active]
    if eth:
        heard = sum(1 for p in eth if p.attach_tier == "L")
        if heard < len(eth):
            notes.append(f"LLDP heard on {heard} of {len(eth)} RoCE rail ports")
    return notes


def _evidence(
    kind: FabricKind,
    ports: Sequence[RailPort],
    switches: Mapping[str, SwitchSummary],
    hop_sources: set[str],
    sms: Sequence[SmSummary],
    lldp: Sequence[LldpSummary],
) -> dict[str, FabricEvidence]:
    ev: dict[str, FabricEvidence] = {}
    if kind != "unknown":
        ev["kind"] = FabricEvidence(tier="S", source="sysfs link_layer")
        ev["rail_rates"] = FabricEvidence(tier="S", source="sysfs ports/*/rate")
    if any(p.sm_guid for p in ports):
        ev["subnets"] = FabricEvidence(tier="M", source="sminfo SM GUID per port")
    elif any(p.subnet for p in ports if p.link_layer == IB):
        ev["subnets"] = FabricEvidence(tier="S", source="sysfs gids/0 prefix + sm_lid")
    elif any(p.subnet for p in ports):
        ev["subnets"] = FabricEvidence(tier="S", source="netdev IPv4 subnets")
    tiers = {p.attach_tier for p in ports if p.attach_tier}
    if tiers:
        source = " + ".join(
            s
            for t, s in (
                ("M", "smpquery nodeinfo DR 0,<port>"),
                ("L", "LLDP chassis/port id"),
            )
            if t in tiers
        )
        tier: Tier = "M" if "M" in tiers else "L"
        ev["leaf_of"] = FabricEvidence(tier=tier, source=source)
        ev["rail_optimized"] = FabricEvidence(tier=tier, source="derived from leaf_of")
        ev["nodes_per_leaf"] = FabricEvidence(tier=tier, source="derived from leaf_of")
    if hop_sources:
        if "ibtracert" in hop_sources:
            ev["hops"] = FabricEvidence(tier="M", source="ibtracert LID routes")
        elif "udp" in hop_sources:
            ev["hops"] = FabricEvidence(tier="T", source="UDP traceroute (IP_RECVERR)")
        else:
            ev["hops"] = FabricEvidence(tier="D", source="leaf_of + census spines")
        ev["tiers"] = ev["hops"]
    if any(s.census != "none" and s.role == "leaf" for s in switches.values()):
        ev["oversubscription"] = FabricEvidence(
            tier="M", source="smpquery switch census"
        )
    if sms:
        ev["sm"] = FabricEvidence(tier="M", source="sminfo")
    if any(item.source for item in lldp):
        ev["lldp"] = FabricEvidence(tier="L", source="lldpctl / raw LLDP frames")
    if any(p.pcie is not None for p in ports):
        ev["pcie"] = FabricEvidence(tier="S", source="sysfs current/max_link_*")
    return ev


def build_topology(
    node_reports: Mapping[str, FabricNodeReport],
    traces: Sequence[TraceResult] | None = None,
    *,
    rail_of: Mapping[str, Mapping[str, int]] | None = None,
) -> FabricTopology:
    """Derive the fabric shape from per-node ``identify`` reports (+ traces).

    ``rail_of`` (node -> HCA -> rail index, from GPU affinity) selects the
    data rails; without it every HCA with an active port is a rail, in
    natural name order. Missing data never raises: it becomes ``None``
    fields and ``coverage`` notes.
    """
    nodes = sorted(node_reports, key=natural_key)
    traces = list(traces or ())
    coverage: list[str] = []
    ports = _rail_ports(node_reports, nodes, rail_of, coverage)
    reports = list(node_reports.values())
    aliases = _guid_aliases(reports)
    records = _census_records(reports)
    our = _our_guids(reports)
    switches = _switches(ports, records, traces, our, aliases)
    leaves_by_rail: dict[int, set[str]] = defaultdict(set)
    leaf_of: dict[str, dict[str, LeafAttachment]] = defaultdict(dict)
    for p in ports:
        if p.switch_id:
            leaves_by_rail[p.rail].add(p.switch_id)
            leaf_of[p.node][p.hca] = LeafAttachment(
                switch_id=p.switch_id,
                port=p.switch_port,
                port_name=p.switch_port_name,
                rail=p.rail,
                tier=p.attach_tier or "D",
            )
    groups = _leaf_groups(ports, nodes)
    sizes = [len(g) for rail_groups in groups.values() for g in rail_groups]
    traced, methods = _trace_hops(traces, coverage)
    inferred = _inferred_hops(ports, switches)
    hops, hop_method, hop_sources = _hops_matrix(traced, inferred, methods)
    values = [
        v for rows in hops.values() for row in rows.values() for v in row.values()
    ]
    max_hops = max(values) if values else None
    unknown_pairs = _unknown_pairs(ports, hops)
    if unknown_pairs:
        coverage.append(
            f"switch hops unknown for {unknown_pairs} rail pair(s); tiers not estimated"
        )
    sms = _sm_summaries(ports, node_reports)
    lldp = _lldp_summaries(ports, node_reports)
    kind = _fabric_kind(ports)
    coverage += _coverage_notes(node_reports, nodes, ports, switches)
    return FabricTopology(
        kind=kind,
        nodes=tuple(nodes),
        rails=_rail_summaries(ports, leaves_by_rail),
        ports=tuple(ports),
        switches=switches,
        leaf_of={
            n: dict(sorted(v.items(), key=lambda kv: natural_key(kv[0])))
            for n, v in sorted(leaf_of.items(), key=lambda kv: natural_key(kv[0]))
        },
        leaf_groups=groups,
        nodes_per_leaf=max(sizes) if sizes else None,
        rail_optimized=_rail_optimized(ports, switches),
        rails_isolated=_rails_isolated(ports),
        tiers=(max_hops + 1) // 2 if max_hops and not unknown_pairs else None,
        max_hops=max_hops,
        hops=hops,
        hop_method=hop_method,
        sms=sms,
        degraded=_degraded(ports, node_reports, switches),
        lldp=lldp,
        evidence=_evidence(kind, ports, switches, hop_sources, sms, lldp),
        coverage=tuple(coverage),
    )


def nominal_gbs_by_rail(topology: FabricTopology) -> dict[int, float]:
    """Line rate of each rail in GB/s (fastest active port / 8)."""
    return {r.rail: r.rate_gbps / 8.0 for r in topology.rails if r.rate_gbps}


# --------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------


def trace_targets(topology: FabricTopology) -> dict[str, tuple[TraceTarget, ...]]:
    """One target per unordered node pair per rail, run from the lower node.

    IB rails trace LID routes (ibtracert); Ethernet rails trace from the
    rail's source IP to the peer's (UDP). IPoIB addresses are never used:
    IB switches are invisible to L3.
    """
    order = {n: i for i, n in enumerate(topology.nodes)}
    out: dict[str, list[TraceTarget]] = defaultdict(list)
    for rail in topology.rail_indices:
        members = sorted(
            (p for p in topology.rail_ports(rail) if p.active),
            key=lambda p: order[p.node],
        )
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if a.link_layer == IB and a.lid and b.lid:
                    target = TraceTarget(
                        rail=rail, hca=a.hca, port=a.port, src_lid=a.lid,
                        dst_lid=b.lid, src_node=a.node, dst_node=b.node,
                    )  # fmt: skip
                elif a.link_layer == ETHERNET and a.ip and b.ip:
                    target = TraceTarget(
                        rail=rail, hca=a.hca, port=a.port, src_ip=a.ip,
                        dst_ip=b.ip, netdev=a.netdev, src_node=a.node,
                        dst_node=b.node,
                    )  # fmt: skip
                else:
                    continue
                out[a.node].append(target)
    return {node: tuple(out[node]) for node in topology.nodes if out.get(node)}


# --------------------------------------------------------------------------
# Link matrix (perftest)
# --------------------------------------------------------------------------


def round_robin_rounds(nodes: Sequence[str]) -> list[list[tuple[str, str]]]:
    """Every unordered pair exactly once, in rounds of disjoint pairs.

    Circle method: N-1 rounds for even N, N rounds (one bye each) for odd N.
    Each pair is ordered by position in ``nodes`` (lower first).
    """
    names = list(nodes)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate node names in {names}")
    if len(names) < 2:
        return []
    order = {n: i for i, n in enumerate(names)}
    slots: list[str | None] = list(names)
    if len(slots) % 2:
        slots.append(None)
    count = len(slots)
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(count - 1):
        pairs = []
        for i in range(count // 2):
            a, b = slots[i], slots[count - 1 - i]
            if a is None or b is None:
                continue
            pairs.append((a, b) if order[a] < order[b] else (b, a))
        rounds.append(sorted(pairs, key=lambda p: order[p[0]]))
        slots = [slots[0], slots[-1], *slots[1:-1]]
    return rounds


class LinkTest(Frozen):
    """perftest between one rail port of ``src`` (client) and ``dst``."""

    src: str
    dst: str
    rail: int
    src_hca: str | None = None
    dst_hca: str | None = None
    round_index: int = 0
    bw_gbs: float | None = None  # ib_write_bw average, GB/s
    bw_peak_gbs: float | None = None
    msg_rate_mpps: float | None = None
    bw_msg_bytes: int | None = None
    lat_typical_us: float | None = None  # ib_write_lat (half round trip)
    lat_avg_us: float | None = None
    lat_min_us: float | None = None
    lat_max_us: float | None = None
    lat_stdev_us: float | None = None
    lat_p99_us: float | None = None
    lat_p999_us: float | None = None
    error: str | None = None


class LinkMatrix(Frozen):
    nodes: tuple[str, ...]
    rails: tuple[int, ...] = ()
    tests: tuple[LinkTest, ...] = ()
    rounds: int = 0
    bidirectional: bool = True
    bw_seconds: int = 3
    bw_msg_bytes: int = 65536
    bw_qps: int = 2
    lat_iters: int = 2000
    lat_msg_bytes: int = 2
    tool: str = "perftest ib_write_bw/ib_write_lat"
    skipped: bool = False
    notes: tuple[str, ...] = ()
    elapsed_s: float = 0.0

    def get(self, src: str, dst: str, rail: int) -> LinkTest | None:
        for test in self.tests:
            if (test.src, test.dst, test.rail) == (src, dst, rail):
                return test
        return None


def link_tests_from_perftest(
    client: PerftestReport,
    *,
    src: str,
    dst: str,
    round_index: int,
    dst_hcas: Mapping[int, str] | None = None,
    server: PerftestReport | None = None,
) -> list[LinkTest]:
    """One ``LinkTest`` per rail from a client run (bw and lat merged)."""
    by_rail: dict[int, dict[str, Any]] = {}
    server_errors = {
        (r.rail, r.test): r.error for r in (server.results if server else ()) if r.error
    }
    for run in client.results:
        entry = by_rail.setdefault(
            run.rail,
            {
                "src": src,
                "dst": dst,
                "rail": run.rail,
                "src_hca": run.hca,
                "dst_hca": (dst_hcas or {}).get(run.rail),
                "round_index": round_index,
                "errors": [],
            },
        )
        if run.error:
            detail = run.error
            if server_errors.get((run.rail, run.test)):
                detail += f"; server: {server_errors[(run.rail, run.test)]}"
            entry["errors"].append(f"{run.test}: {detail}")
        if run.bw is not None:
            entry.update(
                bw_gbs=run.bw.bw_avg_gbs,
                bw_peak_gbs=run.bw.bw_peak_gbs,
                msg_rate_mpps=run.bw.msg_rate_mpps,
                bw_msg_bytes=run.bw.msg_bytes,
            )
        if run.lat is not None:
            entry.update(
                lat_typical_us=run.lat.t_typical_us,
                lat_avg_us=run.lat.t_avg_us,
                lat_min_us=run.lat.t_min_us,
                lat_max_us=run.lat.t_max_us,
                lat_stdev_us=run.lat.t_stdev_us,
                lat_p99_us=run.lat.p99_us,
                lat_p999_us=run.lat.p999_us,
            )
    tests = []
    for rail in sorted(by_rail):
        entry = by_rail[rail]
        errors = entry.pop("errors")
        tests.append(LinkTest(**entry, error="; ".join(errors) or None))
    return tests


LinkFindingKind = Literal[
    "weak_link", "weak_rail", "asymmetry", "latency_groups", "failed"
]


class LinkFinding(Frozen):
    kind: LinkFindingKind
    rail: int | None = None
    src: str | None = None
    dst: str | None = None
    value: float | None = None
    reference: float | None = None
    detail: str
    groups: tuple[tuple[str, ...], ...] = ()  # latency groups of "a->b" labels


def _latency_groups(
    items: Sequence[tuple[str, float]],
) -> list[list[tuple[str, float]]]:
    ordered = sorted(items, key=lambda kv: kv[1])
    groups: list[list[tuple[str, float]]] = []
    for label, value in ordered:
        if groups:
            low = groups[-1][-1][1]
            gap = value - low
            if gap < max(LAT_GAP_MIN_US, LAT_GAP_REL * low):
                groups[-1].append((label, value))
                continue
        groups.append([(label, value)])
    return groups


def analyze_link_matrix(
    matrix: LinkMatrix,
    nominal_gbs_by_rail: Mapping[int, float],
    *,
    hops: Mapping[int, Mapping[str, Mapping[str, int]]] | None = None,
) -> list[LinkFinding]:
    """Weak links (< 80 % of the rail median or of line rate), weak rails,
    direction asymmetry (> 20 %), failed tests, and per-rail latency groups.

    Latency groups split sorted typical latencies at gaps of at least
    max(0.15 us, 5 %): with no MAD hop data, the fast group is likely the
    same-leaf pairs and the slow one crosses spines; when ``hops`` are
    given, each group's hop counts are reported alongside.
    """
    findings: list[LinkFinding] = []
    for test in matrix.tests:
        if test.error and test.bw_gbs is None and test.lat_typical_us is None:
            findings.append(
                LinkFinding(
                    kind="failed",
                    rail=test.rail,
                    src=test.src,
                    dst=test.dst,
                    detail=f"rail {test.rail} {test.src}->{test.dst}: {test.error}",
                )
            )
    for rail in sorted({t.rail for t in matrix.tests}):
        tests = [t for t in matrix.tests if t.rail == rail]
        bws = [t.bw_gbs for t in tests if t.bw_gbs is not None]
        median = _median(bws)
        nominal = nominal_gbs_by_rail.get(rail)
        if median is not None and nominal and median < WEAK_FRACTION * nominal:
            findings.append(
                LinkFinding(
                    kind="weak_rail",
                    rail=rail,
                    value=median,
                    reference=nominal,
                    detail=(
                        f"rail {rail} median {median:.1f} GB/s is "
                        f"{100 * median / nominal:.0f}% of line rate {nominal:.1f} GB/s"
                    ),
                )
            )
        for test in tests:
            if test.bw_gbs is None or median is None:
                continue
            refs = [WEAK_FRACTION * median]
            if nominal:
                refs.append(WEAK_FRACTION * nominal)
            if test.bw_gbs < max(refs):
                ref = (
                    nominal
                    if nominal and test.bw_gbs < WEAK_FRACTION * nominal
                    else median
                )
                findings.append(
                    LinkFinding(
                        kind="weak_link",
                        rail=rail,
                        src=test.src,
                        dst=test.dst,
                        value=test.bw_gbs,
                        reference=ref,
                        detail=(
                            f"rail {rail} {test.src}->{test.dst} {test.bw_gbs:.1f} GB/s "
                            f"(rail median {median:.1f}"
                            + (f", line rate {nominal:.1f}" if nominal else "")
                            + " GB/s)"
                        ),
                    )
                )
        seen: set[tuple[str, str]] = set()
        for test in tests:
            pair = tuple(sorted((test.src, test.dst), key=natural_key))
            back = next(
                (t for t in tests if (t.src, t.dst) == (test.dst, test.src)), None
            )
            if pair in seen or back is None:
                continue
            seen.add(pair)
            if test.bw_gbs is None or back.bw_gbs is None:
                continue
            high = max(test.bw_gbs, back.bw_gbs)
            if high > 0 and abs(test.bw_gbs - back.bw_gbs) / high > ASYMMETRY_FRACTION:
                findings.append(
                    LinkFinding(
                        kind="asymmetry",
                        rail=rail,
                        src=test.src,
                        dst=test.dst,
                        value=test.bw_gbs,
                        reference=back.bw_gbs,
                        detail=(
                            f"rail {rail} {test.src}->{test.dst} {test.bw_gbs:.1f} GB/s vs "
                            f"{test.dst}->{test.src} {back.bw_gbs:.1f} GB/s"
                        ),
                    )
                )
        lat = [
            (f"{t.src}->{t.dst}", t.lat_typical_us)
            for t in tests
            if t.lat_typical_us is not None
        ]
        groups = _latency_groups(lat)
        if len(groups) > 1:
            parts = []
            for group in groups:
                values = [v for _, v in group]
                text = f"{len(group)} pair(s) ~{statistics.median(values):.2f} us"
                rail_hops = (hops or {}).get(rail, {})
                counts = sorted(
                    {
                        rail_hops.get(a, {}).get(b)
                        for a, b in (label.split("->") for label, _ in group)
                    }
                    - {None}
                )
                if counts:
                    text += " (hops " + "/".join(str(c) for c in counts) + ")"
                parts.append(text)
            findings.append(
                LinkFinding(
                    kind="latency_groups",
                    rail=rail,
                    value=groups[0][0][1],
                    reference=groups[-1][-1][1],
                    detail=f"rail {rail} latency groups: " + "; ".join(parts),
                    groups=tuple(tuple(label for label, _ in g) for g in groups),
                )
            )
    return findings


# --------------------------------------------------------------------------
# Switch port load (perfquery snapshots)
# --------------------------------------------------------------------------


class SwitchPortRef(Frozen):
    """A switch port to sample, and the local port to query it through."""

    switch_guid: str
    switch_desc: str | None = None
    switch_lid: int
    port: int
    role: Literal["downlink", "uplink"]
    rate_gbps: float | None = None
    rail: int | None = None
    peer_node: str | None = None  # downlinks: our node behind the port
    peer_hca: str | None = None
    peer_guid: str | None = None  # uplinks: the spine
    via_node: str
    via_hca: str
    via_port: int

    @property
    def key(self) -> str:
        return f"{self.switch_guid}:{self.port}"


def switch_port_refs(
    topology: FabricTopology, *, via_node: str | None = None
) -> list[SwitchPortRef]:
    """Each leaf's downlinks to our HCAs plus its uplinks (deduplicated).

    PMA queries are LID-routed, so each ref goes through a ``via_node``
    (default: the first node) port on the leaf's rail, else on the leaf's
    subnet. Leaves without a LID (RoCE) or without such a port are skipped.
    """
    if not topology.nodes:
        return []
    via = via_node or topology.nodes[0]
    local = [
        p for p in topology.ports if p.node == via and p.active and p.link_layer == IB
    ]
    refs: dict[str, SwitchPortRef] = {}
    for leaf in topology.by_role("leaf"):
        if leaf.lid is None:
            continue
        attached = [p for p in topology.ports if p.switch_id == leaf.id]
        subnets = {p.subnet for p in attached if p.subnet}
        through = next(
            (
                p
                for p in local
                if p.rail in leaf.rails and (not subnets or p.subnet in subnets)
            ),
            None,
        )
        if through is None:
            through = next((p for p in local if p.subnet and p.subnet in subnets), None)
        if through is None:
            continue
        common = {
            "switch_guid": leaf.id,
            "switch_desc": leaf.desc,
            "switch_lid": leaf.lid,
            "via_node": via,
            "via_hca": through.hca,
            "via_port": through.port,
        }
        for p in sorted(
            attached, key=lambda q: (q.switch_port or 0, natural_key(q.node))
        ):
            if p.switch_port is None:
                continue
            ref = SwitchPortRef(
                **common,
                port=p.switch_port,
                role="downlink",
                rate_gbps=p.rate_gbps or None,
                rail=p.rail,
                peer_node=p.node,
                peer_hca=p.hca,
            )
            refs.setdefault(ref.key, ref)
        for link in leaf.uplinks:
            ref = SwitchPortRef(
                **common,
                port=link.port,
                role="uplink",
                rate_gbps=link.rate_gbps,
                rail=leaf.rails[0] if len(leaf.rails) == 1 else None,
                peer_guid=link.peer_id,
            )
            refs.setdefault(ref.key, ref)
    return list(refs.values())


def perfquery_ports(refs: Sequence[SwitchPortRef]) -> list[dict[str, Any]]:
    """The ``perfquery --ports`` input of the agent for ``refs``."""
    return [
        {
            "key": ref.key,
            "hca": ref.via_hca,
            "port": ref.via_port,
            "lid": ref.switch_lid,
            "switch_port": ref.port,
        }
        for ref in refs
    ]


class PortCounterSample(Frozen):
    ref: SwitchPortRef
    counters: dict[str, int] = Field(default_factory=dict)
    wide: tuple[str, ...] = ()
    extended: bool = False
    mono_s: float | None = None
    wall_s: float | None = None
    error: str | None = None


class SwitchSnapshot(Frozen):
    node: str  # where perfquery ran (monotonic times are comparable there)
    label: str = ""
    taken_at_s: float | None = None  # median wall time of the samples
    samples: tuple[PortCounterSample, ...] = ()
    notes: tuple[str, ...] = ()
    elapsed_s: float = 0.0


def snapshot_from_perfquery(
    refs: Sequence[SwitchPortRef],
    report: PerfqueryReport | None,
    *,
    node: str,
    label: str = "",
    notes: Sequence[str] = (),
) -> SwitchSnapshot:
    """Join agent readings back to their refs (missing ones get an error)."""
    readings = {r.key: r for r in (report.ports if report else ())}
    samples = []
    for ref in refs:
        reading = readings.get(ref.key)
        if reading is None:
            samples.append(PortCounterSample(ref=ref, error="not sampled"))
            continue
        samples.append(
            PortCounterSample(
                ref=ref,
                counters=reading.counters,
                wide=reading.wide,
                extended=reading.extended,
                mono_s=reading.mono_s,
                wall_s=reading.wall_s,
                error=reading.error,
            )
        )
    walls = [s.wall_s for s in samples if s.wall_s is not None]
    agent_notes = [f"{n.source}: {n.error}" for n in (report.notes if report else ())]
    return SwitchSnapshot(
        node=node,
        label=label,
        taken_at_s=statistics.median(walls) if walls else None,
        samples=tuple(samples),
        notes=tuple([*notes, *agent_notes]),
        elapsed_s=report.elapsed_s if report else 0.0,
    )


class SwitchPortLoad(Frozen):
    """Load on one switch port between two snapshots.

    ``tx`` is what the switch port transmitted (towards the peer: our HCA for
    downlinks, the spine for uplinks), ``rx`` what it received.
    """

    ref: SwitchPortRef
    interval_s: float | None = None
    tx_gbs: float | None = None
    rx_gbs: float | None = None
    tx_util: float | None = None  # vs rate_gbps / 8
    rx_util: float | None = None
    xmit_wait_delta: int | None = None
    xmit_wait_per_s: float | None = None
    stall_fraction_est: float | None = None  # UNVERIFIED tick unit
    wait_ratio: float | None = None  # xmit_wait / (xmit_wait + xmit words)
    error_deltas: dict[str, int] = Field(default_factory=dict)
    reset: bool = False
    saturated: bool = False
    wrapped: bool = False
    notes: tuple[str, ...] = ()


def _counter_bits(name: str, a: PortCounterSample, b: PortCounterSample) -> int:
    if name in a.wide and name in b.wide:
        return 64
    return BASIC_COUNTER_BITS.get(name, 32)


def _counter_delta(
    name: str,
    a: PortCounterSample,
    b: PortCounterSample,
    max_delta: float | None = None,
) -> tuple[int | None, str | None]:
    """(delta, flag) with flag in {None, "missing", "saturated", "wrapped",
    "reset"}. Narrow counters saturate per the IB spec; a negative delta is
    a wrap when that is plausible (<= ``max_delta``), else a reset."""
    if name not in a.counters or name not in b.counters:
        return None, "missing"
    bits = _counter_bits(name, a, b)
    top = (1 << bits) - 1
    v0, v1 = a.counters[name], b.counters[name]
    if bits < 64 and (v0 >= top or v1 >= top):
        return None, "saturated"
    delta = v1 - v0
    if delta >= 0:
        return delta, None
    wrapped = delta + (1 << bits)
    if max_delta is not None and wrapped <= max_delta:
        return wrapped, "wrapped"
    if max_delta is None and bits < 64:
        return wrapped, "wrapped"
    return None, "reset"


def _interval(
    a: PortCounterSample, b: PortCounterSample, same_node: bool
) -> float | None:
    if same_node and a.mono_s is not None and b.mono_s is not None:
        return b.mono_s - a.mono_s
    if a.wall_s is not None and b.wall_s is not None:
        return b.wall_s - a.wall_s
    return None


def switch_load(
    before: SwitchSnapshot,
    after: SwitchSnapshot,
    *,
    xmit_wait_tick_ns: float = XMIT_WAIT_TICK_NS_UNVERIFIED,
) -> list[SwitchPortLoad]:
    """Per-port load between two snapshots of the same refs."""
    first = {s.ref.key: s for s in before.samples}
    same_node = before.node == after.node
    loads = []
    for b in after.samples:
        a = first.get(b.ref.key)
        notes: list[str] = []
        if a is None or a.error or b.error:
            notes.append(
                "not in the first snapshot" if a is None else (a.error or b.error or "")
            )
            loads.append(SwitchPortLoad(ref=b.ref, notes=tuple(notes)))
            continue
        dt = _interval(a, b, same_node)
        if dt is None or dt <= 0:
            loads.append(
                SwitchPortLoad(
                    ref=b.ref, interval_s=dt, notes=("no positive interval",)
                )
            )
            continue
        rate = b.ref.rate_gbps
        max_words = (
            rate * 1e9 / 8 / IB_WORD_BYTES * dt * WRAP_RATE_SLACK if rate else None
        )
        flags: set[str] = set()
        rates: dict[str, float | None] = {}
        words: dict[str, int | None] = {}
        for name, key in (("PortXmitData", "tx"), ("PortRcvData", "rx")):
            delta, flag = _counter_delta(name, a, b, max_words)
            if flag:
                flags.add(flag)
                notes.append(f"{name} {flag}")
            words[key] = delta
            rates[key] = None if delta is None else delta * IB_WORD_BYTES / dt / 1e9
        line_gbs = rate / 8.0 if rate else None
        wait, wait_flag = _counter_delta("PortXmitWait", a, b)
        if wait_flag and wait_flag != "missing":
            flags.add(wait_flag)
            notes.append(f"PortXmitWait {wait_flag}")
        errors: dict[str, int] = {}
        for name in ERROR_COUNTERS:
            delta, flag = _counter_delta(name, a, b, 0)
            if (
                delta is None
                and flag == "saturated"
                and name in b.counters
                and name in a.counters
            ):
                delta = max(0, b.counters[name] - a.counters[name])
                notes.append(f"{name} saturated (lower bound)")
            if flag == "reset":
                flags.add("reset")
            if delta:
                errors[name] = delta
        wait_ratio = None
        if wait is not None and words["tx"] is not None and (wait + words["tx"]) > 0:
            wait_ratio = wait / (wait + words["tx"])
        loads.append(
            SwitchPortLoad(
                ref=b.ref,
                interval_s=dt,
                tx_gbs=rates["tx"],
                rx_gbs=rates["rx"],
                tx_util=rates["tx"] / line_gbs
                if rates["tx"] is not None and line_gbs
                else None,
                rx_util=rates["rx"] / line_gbs
                if rates["rx"] is not None and line_gbs
                else None,
                xmit_wait_delta=wait,
                xmit_wait_per_s=wait / dt if wait is not None else None,
                stall_fraction_est=min(1.0, wait * xmit_wait_tick_ns * 1e-9 / dt)
                if wait is not None
                else None,
                wait_ratio=wait_ratio,
                error_deltas=errors,
                reset="reset" in flags,
                saturated="saturated" in flags,
                wrapped="wrapped" in flags,
                notes=tuple(notes),
            )
        )
    return loads


# --------------------------------------------------------------------------
# Knowledge-base views
# --------------------------------------------------------------------------


def _rounded(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def topology_signature(topology: FabricTopology) -> dict[str, Any]:
    """Small, stable, identifier-free summary (KB key material)."""
    leaves = topology.by_role("leaf")
    spines = topology.by_role("spine")
    ratios = [s.oversubscription for s in leaves if s.oversubscription is not None]
    rates = sorted({r.rate_gbps for r in topology.rails if r.rate_gbps})
    pfc = sorted({p for s in topology.lldp for p in (s.pfc_enabled or ())})
    shape = {
        str(rail): "+".join(str(len(g)) for g in groups)
        for rail, groups in sorted(topology.leaf_groups.items())
    }
    return {
        "kind": topology.kind,
        "nodes": len(topology.nodes),
        "rails": len(topology.rails),
        "rail_rate_gbps": rates[0] if len(rates) == 1 else rates,
        "rail_optimized": topology.rail_optimized,
        "rails_isolated": topology.rails_isolated,
        "nodes_per_leaf": topology.nodes_per_leaf,
        "leaf_shape": sorted(set(shape.values())) if shape else [],
        "leaves": len(leaves),
        "spines": len(spines),
        "tiers": topology.tiers,
        "max_hops": topology.max_hops,
        "max_oversubscription": _rounded(max(ratios)) if ratios else None,
        "degraded_links": len(topology.degraded),
        "pfc_priorities": pfc,
        "tiers_seen": sorted({e.tier for e in topology.evidence.values()}),
        "hash": stable_hash(
            [topology.kind, len(topology.rails), rates, topology.rail_optimized,
             topology.nodes_per_leaf, topology.tiers, sorted(shape.items())],
            12,
        ),  # fmt: skip
    }


def _leaf_findings(leaves: Sequence[SwitchSummary]) -> list[str]:
    out = []
    censused = [s for s in leaves if s.oversubscription is not None]
    if censused:
        worst = max(censused, key=lambda s: s.oversubscription or 0.0)
        over = [s for s in censused if (s.oversubscription or 0.0) > 1.0 + 1e-9]
        if over:
            out.append(
                f"{len(over)} of {len(censused)} censused leaves oversubscribed, "
                f"worst {worst.desc or worst.id} {worst.oversubscription:.2f}:1 "
                f"({worst.host_ports_active} host ports vs {worst.uplinks_active} "
                "uplinks)"
            )
        else:
            out.append(
                f"non-blocking leaves: host-facing capacity <= uplink capacity on "
                f"{len(censused)} censused leaves"
            )
    shared = [s for s in leaves if s.other_hosts]
    if shared:
        most = max(s.other_hosts or 0 for s in shared)
        out.append(
            f"{len(shared)} of {len(leaves)} leaves also serve other tenants' hosts "
            f"(up to {most} host ports per leaf)"
        )
    return out


def topology_findings(topology: FabricTopology) -> list[str]:
    """English one-liners for the report and the KB."""
    out: list[str] = []
    rates = sorted({r.rate_gbps for r in topology.rails if r.rate_gbps})
    rate_text = "/".join(f"{r:g}" for r in rates) or "?"
    out.append(
        f"{topology.kind} fabric: {len(topology.rails)} rail(s) x {rate_text} Gb/s "
        f"per node ({len(topology.nodes)} nodes)"
    )
    leaves = topology.by_role("leaf")
    if topology.rail_optimized is True:
        out.append(
            f"rail-optimized: {len(topology.rails)} rails, "
            f"{topology.nodes_per_leaf} nodes per leaf"
        )
    elif topology.rail_optimized is False:
        widest = max((len(s.rails) for s in leaves), default=0)
        out.append(
            f"not rail-optimized: a leaf carries up to {widest} rails of our HCAs "
            f"({len(leaves)} leaves, {topology.nodes_per_leaf} nodes per leaf)"
        )
    elif not leaves:
        out.append("leaf switches unknown (no MAD access and no LLDP)")
    if topology.rails_isolated is True:
        sms = f" ({len(topology.sms)} SMs)" if topology.sms else ""
        out.append(f"rails are isolated subnets{sms}")
    elif topology.rails_isolated is False:
        out.append("rails share one subnet")
    if topology.max_hops is not None:
        method = "/".join(sorted(set(topology.hop_method.values())))
        if topology.tiers is not None:
            out.append(
                f"{topology.tiers}-tier path between our nodes: up to "
                f"{topology.max_hops} switch hop(s) ({method})"
            )
        else:
            out.append(
                f"up to {topology.max_hops} switch hop(s) between the node pairs "
                f"with known paths ({method}); other pairs unknown"
            )
    out += _leaf_findings(leaves)
    for item in topology.degraded:
        out.append(f"degraded link: {item.detail}")
    pfc_sets = {s.pfc_enabled for s in topology.lldp if s.pfc_enabled is not None}
    if len(pfc_sets) == 1:
        (prios,) = pfc_sets
        text = ",".join(str(p) for p in prios) or "none"
        out.append(f"switch DCBX advertises PFC on priority {text}")
    elif len(pfc_sets) > 1:
        out.append("switches advertise inconsistent PFC priorities across RoCE ports")
    for item in topology.lldp:
        if (
            item.pfc_enabled is not None
            and item.host_pfc_enabled is not None
            and set(item.pfc_enabled) != set(item.host_pfc_enabled)
        ):
            out.append(
                f"PFC mismatch on {item.node}/{item.netdev}: switch {list(item.pfc_enabled)} "
                f"vs host {list(item.host_pfc_enabled)}"
            )
    out += [f"coverage: {c}" for c in topology.coverage]
    return out


# --------------------------------------------------------------------------
# Synthetic data for the dry-run simulator (deterministic)
# --------------------------------------------------------------------------


def _guid(*parts: Any) -> str:
    return "0x" + stable_hash(["guid", *parts], 16)


def _mac_of(*parts: Any) -> str:
    raw = stable_hash(["mac", *parts], 10)
    return "02:" + ":".join(raw[i : i + 2] for i in range(0, 10, 2))


def _gid_text(prefix: str, guid: str) -> str:
    raw = prefix[2:] + guid[2:]
    return ":".join(raw[i : i + 4] for i in range(0, 32, 4))


def _speed_for(rate_gbps: float) -> str:
    per_lane = rate_gbps / 4
    for name, value in LANE_GBPS.items():
        if abs(value - per_lane) < 1e-6 and name != "FDR10":
            return name
    return "NDR"


def _synthetic_layout(
    node_names: Sequence[str],
    rails: int,
    nodes_per_leaf: int,
    rail_optimized: bool,
) -> dict[tuple[str, int], tuple[str, int]]:
    """(node, rail) -> (leaf key, leaf port)."""
    layout = {}
    for index, node in enumerate(node_names):
        group = index // max(1, nodes_per_leaf)
        slot = index % max(1, nodes_per_leaf)
        for rail in range(rails):
            if rail_optimized:
                layout[(node, rail)] = (f"r{rail}g{group}", slot + 1)
            else:
                layout[(node, rail)] = (f"g{group}", slot * rails + rail + 1)
    return layout


def _synthetic_port(
    node: str, rail: int, is_ib: bool, rate_gbps: float, speed: str
) -> dict[str, Any]:
    hca = f"mlx5_{rail}"
    guid = _guid("hca", node, rail)
    bus = 0x18 + 0x10 * rail
    return {
        "hca": hca,
        "port": 1,
        "pci_bus_id": f"0000:{bus:02x}:00.0",
        "link_layer": IB if is_ib else ETHERNET,
        "state": "4: ACTIVE",
        "phys_state": "5: LinkUp",
        "rate": f"{rate_gbps:g} Gb/sec (4X {speed})",
        "rate_gbps": rate_gbps,
        "width": "4X",
        "speed": speed,
        "lid": None,
        "sm_lid": None,
        "sm_sl": 0,
        "lmc": 0,
        "cap_mask": "0xa751e848" if is_ib else "0x04010000",
        "node_guid": guid,
        "sys_image_guid": guid,
        "node_desc": f"{node} {hca}",
        "fw_ver": "28.43.1014",
        "board_id": "MT_0000000838",
        "port_guid": guid,
        "subnet_prefix": "0xfe80000000000000",
        "gids": [],
        "roce_v2_gid_index": None,
        "roce_v2_ipv4": None,
        "netdev": f"ibp{bus}s0" if is_ib else f"enp{bus}s0np0",
        "mtu": 4092 if is_ib else 9000,
        "operstate": "up",
        "mac": None if is_ib else _mac_of("nic", node, rail),
        "ipv4": [],
        "active_mtu": 4096,
        "max_mtu": 4096,
        "pcie": {
            "current_speed_gts": 32.0,
            "current_width": 16,
            "max_speed_gts": 32.0,
            "max_width": 16,
        },
        "neighbor": None,
    }


def _synthetic_lldp(leaf: str, leaf_port: int, leaf_index: int) -> dict[str, Any]:
    return {
        "source": "raw",
        "chassis_id": _mac_of("leaf", leaf),
        "chassis_id_subtype": "mac",
        "port_id": f"swp{leaf_port}",
        "port_id_subtype": "ifname",
        "ttl_s": 120,
        "port_descr": f"swp{leaf_port}",
        "system_name": f"leaf-{leaf}",
        "system_descr": "Cumulus Linux 5.9 on SN5600",
        "mgmt_addresses": [f"192.168.0.{10 + leaf_index}"],
        "port_vlan_id": None,
        "vlans": [],
        "max_frame_size": 9216,
        "pfc": {"willing": False, "mbc": False, "cap": 8, "enabled": [3]},
        "ets": {
            "willing": False,
            "cbs": False,
            "max_tcs": 8,
            "prio_tc": [0, 0, 0, 3, 0, 0, 6, 0],
            "tc_bw": [50, 0, 0, 50, 0, 0, 0, 0],
            "tsa": [2, 0, 0, 2, 0, 0, 0, 0],
        },
        "app_priority": [{"priority": 3, "selector": 3, "protocol": ROCE_V2_UDP_PORT}],
        "unknown_tlvs": [],
    }


def _ipv4_gid(ip: str) -> str:
    octets = tuple(int(o) for o in ip.split("."))
    return "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x" % octets


def synthetic_reports(
    node_names: Sequence[str],
    *,
    rails: int = 8,
    kind: str = "infiniband",
    nodes_per_leaf: int = 4,
    rail_optimized: bool = True,
    rate_gbps: float = 400.0,
    leaf_ports: int = 64,
) -> dict[str, dict[str, Any]]:
    """Agent ``identify`` JSON per node for a synthetic fabric.

    IB: one subnet per rail when rail-optimized (else one subnet), leaves
    with ``leaf_ports`` ports (our nodes first, other tenants' hosts next,
    then non-blocking uplinks to 2 spines per subnet), census complete in
    the report of each leaf's first node. RoCE: per-rail /24, LLDP with PFC
    on priority 3. "mixed": the first half of the rails IB, the rest RoCE.
    """
    if kind not in ("infiniband", "roce", "mixed"):
        raise ValueError(f"unknown synthetic fabric kind {kind!r}")
    layout = _synthetic_layout(node_names, rails, nodes_per_leaf, rail_optimized)
    speed = _speed_for(rate_gbps)
    ib_rails = {
        "infiniband": set(range(rails)),
        "mixed": set(range(rails // 2)),
        "roce": set(),
    }[kind]
    leaf_members: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (node, rail), (leaf, port) in layout.items():
        leaf_members[leaf].append((node, rail, port))
    leaf_index = {leaf: i for i, leaf in enumerate(sorted(leaf_members))}
    censused: set[str] = set()
    reports: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(node_names):
        ports = []
        lldp: dict[str, Any] = {}
        sm_ports: dict[str, list[str]] = defaultdict(list)
        switches: dict[str, Any] = {}
        for rail in range(rails):
            leaf, leaf_port = layout[(node, rail)]
            port = _synthetic_port(node, rail, rail in ib_rails, rate_gbps, speed)
            ports.append(port)
            if rail not in ib_rails:
                ip = f"10.{100 + rail}.0.{index + 1}"
                port["ipv4"] = [{"address": ip, "prefixlen": 24}]
                port["roce_v2_gid_index"] = 3
                port["roce_v2_ipv4"] = ip
                port["gids"] = [
                    {"index": 3, "gid": _ipv4_gid(ip), "type": "RoCE v2",
                     "ndev": port["netdev"]},
                ]  # fmt: skip
                lldp[port["netdev"]] = _synthetic_lldp(
                    leaf, leaf_port, leaf_index[leaf]
                )
                continue
            subnet_key = f"rail{rail}" if rail_optimized else "all"
            # One subnet per rail when rail-optimized: LIDs repeat per rail.
            port["lid"] = 100 + index if rail_optimized else 100 + index * rails + rail
            port["sm_lid"] = 1
            port["gids"] = [
                {"index": 0, "gid": _gid_text("0xfe80000000000000", port["node_guid"]),
                 "type": "IB/RoCE v1", "ndev": None},
            ]  # fmt: skip
            leaf_guid = _guid("leaf", leaf)
            port["neighbor"] = {
                "node_type": "Switch",
                "node_guid": leaf_guid,
                "port_guid": leaf_guid,
                "system_guid": leaf_guid,
                "num_ports": leaf_ports,
                "local_port": leaf_port,
                "desc": f"MF0;leaf-{leaf}:MQM9700/U1",
                "dev_id": "0xd2f0",
                "vendor_id": "0x0002c9",
                "lid": 2 + leaf_index[leaf],
                "link": _synthetic_link(leaf_port, rate_gbps, speed),
            }
            sm_ports[subnet_key].append(f"{port['hca']}:1")
            if leaf not in censused:
                censused.add(leaf)
                switches[leaf_guid] = _synthetic_census(
                    leaf, leaf_guid, port, leaf_members[leaf], subnet_key,
                    leaf_ports, rate_gbps, speed, leaf_index[leaf],
                )  # fmt: skip
        has_ib = bool(sm_ports)
        reports[node] = {
            "kind": "fabric_identify",
            "version": FABRIC_REPORT_VERSION,
            "hostname": f"synthetic-{node}",
            "tools": {
                **dict.fromkeys(
                    ("smpquery", "sminfo", "ibtracert", "perfquery"), has_ib
                ),
                **dict.fromkeys(
                    ("ibv_devinfo", "ib_write_bw", "ib_write_lat", "ip"), True
                ),
                **dict.fromkeys(("lldpctl", "mlnx_qos"), False),
            },
            "umad": has_ib,
            "smp": {
                "budget": 400,
                "used": 6 * sum(len(keys) for keys in sm_ports.values()),
                "failed": 0,
                "blocked": False,
                "exhausted": False,
            },
            "ports": ports,
            "sm": [
                {"guid": _guid("sm", key), "lid": 1, "priority": 15, "state": 3,
                 "state_name": "SMINFO_MASTER", "activity_count": 1000, "ports": keys}
                for key, keys in sorted(sm_ports.items())
            ],  # fmt: skip
            "switches": switches,
            "lldp": lldp,
            "qos": {},
            "notes": [],
            "elapsed_s": 2.5,
        }
    return reports


def _synthetic_link(port: int, rate_gbps: float, speed: str) -> dict[str, Any]:
    return {
        "port": port,
        "state": "Active",
        "phys_state": "LinkUp",
        "lid": None,
        "width_active": "4X",
        "width_supported": ["1X", "2X", "4X"],
        "speed_active": speed,
        "speed_supported": [
            s
            for s in ("SDR", "DDR", "QDR", "FDR", "EDR", "HDR", "NDR")
            if LANE_GBPS[s] <= LANE_GBPS[speed]
        ],
        "rate_gbps": rate_gbps,
        "max_rate_gbps": rate_gbps,
    }


def _synthetic_census(
    leaf: str,
    leaf_guid: str,
    via_port: dict[str, Any],
    members: Sequence[tuple[str, int, int]],
    subnet_key: str,
    leaf_ports: int,
    rate_gbps: float,
    speed: str,
    leaf_index: int,
) -> dict[str, Any]:
    ours = {port: (node, rail) for node, rail, port in members}
    downlinks = leaf_ports // 2
    entries = []
    for port in range(1, leaf_ports + 1):
        link = _synthetic_link(port, rate_gbps, speed)
        if port in ours:
            node, rail = ours[port]
            guid = _guid("hca", node, rail)
            peer = {
                "node_type": "Channel Adapter",
                "node_guid": guid,
                "port_guid": guid,
                "system_guid": guid,
                "num_ports": 1,
                "local_port": 1,
                "desc": None,
                "dev_id": "0x1021",
                "vendor_id": "0x0002c9",
            }
        elif port <= downlinks:
            guid = _guid("tenant", leaf, port)
            peer = {
                "node_type": "Channel Adapter",
                "node_guid": guid,
                "port_guid": guid,
                "system_guid": guid,
                "num_ports": 1,
                "local_port": 1,
                "desc": None,
                "dev_id": "0x1021",
                "vendor_id": "0x0002c9",
            }
        else:
            spine = (port - downlinks - 1) % 2
            guid = _guid("spine", subnet_key, spine)
            peer = {
                "node_type": "Switch",
                "node_guid": guid,
                "port_guid": guid,
                "system_guid": guid,
                "num_ports": leaf_ports,
                "local_port": leaf_index * (leaf_ports // 4)
                + (port - downlinks + 1) // 2,
                "desc": f"MF0;spine-{subnet_key}-{spine}:MQM9700/U1",
                "dev_id": "0xd2f0",
                "vendor_id": "0x0002c9",
            }
        entries.append({**link, "peer": peer})
    return {
        "guid": leaf_guid,
        "desc": f"MF0;leaf-{leaf}:MQM9700/U1",
        "lid": 2 + leaf_index,
        "num_ports": leaf_ports,
        "via": f"{via_port['hca']}:1",
        "census": "complete",
        "ports": entries,
    }


def synthetic_link_matrix(
    node_names: Sequence[str],
    *,
    rails: int = 8,
    rate_gbps: float = 400.0,
    nodes_per_leaf: int = 4,
    rail_optimized: bool = True,
    bidirectional: bool = True,
) -> LinkMatrix:
    """Deterministic perftest results: ~96 % of line rate, 1.8 us within a
    leaf and 2.35 us across spines (+-1 % pseudo-random jitter)."""
    layout = _synthetic_layout(node_names, rails, nodes_per_leaf, rail_optimized)
    tests = []
    for index, pair_round in enumerate(round_robin_rounds(node_names)):
        for a, b in pair_round:
            for src, dst in ((a, b), (b, a)) if bidirectional else ((a, b),):
                for rail in range(rails):
                    jitter = int(stable_hash([src, dst, rail], 4), 16) / 0xFFFF - 0.5
                    same = layout[(src, rail)][0] == layout[(dst, rail)][0]
                    tests.append(
                        LinkTest(
                            src=src,
                            dst=dst,
                            rail=rail,
                            src_hca=f"mlx5_{rail}",
                            dst_hca=f"mlx5_{rail}",
                            round_index=index,
                            bw_gbs=round(rate_gbps / 8 * (0.96 + 0.02 * jitter), 3),
                            bw_peak_gbs=0.0,
                            msg_rate_mpps=round(rate_gbps / 8 * 0.96e3 / 65.536, 3),
                            bw_msg_bytes=65536,
                            lat_typical_us=round(
                                (1.8 if same else 2.35) * (1 + 0.02 * jitter), 3
                            ),
                            lat_avg_us=round(
                                (1.82 if same else 2.37) * (1 + 0.02 * jitter), 3
                            ),
                            lat_min_us=1.7 if same else 2.25,
                            lat_max_us=5.0,
                            lat_stdev_us=0.05,
                            lat_p99_us=1.95 if same else 2.5,
                            lat_p999_us=4.5,
                        )
                    )
    rounds = len(round_robin_rounds(node_names))
    return LinkMatrix(
        nodes=tuple(node_names),
        rails=tuple(range(rails)),
        tests=tuple(tests),
        rounds=rounds * (2 if bidirectional else 1),
        bidirectional=bidirectional,
        notes=("synthetic",),
    )


def synthetic_traces(topology: FabricTopology) -> list[TraceResult]:
    """Traces consistent with ``topology`` (inferred hops; leaf GUIDs)."""
    results = []
    for node, targets in trace_targets(topology).items():
        for target in targets:
            a = topology.port(target.src_node or node, target.rail)
            b = topology.port(target.dst_node or "", target.rail)
            if a is None or b is None:
                continue
            count = topology.hops.get(target.rail, {}).get(a.node, {}).get(b.node, 3)
            if a.link_layer == IB:
                leafs = [s for s in (a.switch_id, b.switch_id) if s]
                guids = (
                    leafs[:1]
                    if count == 1
                    else [leafs[0], _guid("spine-path", target.rail), leafs[-1]]
                )
                hops = tuple(
                    TraceHop(node_type="Switch", guid=g) for g in guids[:count]
                )
                results.append(
                    TraceResult(
                        target=target,
                        method="ibtracert",
                        hops=hops,
                        hop_count=len(hops),
                        reached=True,
                    )
                )
            else:
                results.append(
                    TraceResult(target=target, method="udp", hop_count=0, reached=True)
                )
    return results


def synthetic_snapshot(
    refs: Sequence[SwitchPortRef],
    *,
    node: str,
    t_s: float,
    utilization: float = 0.3,
    label: str = "",
) -> SwitchSnapshot:
    """Counters that grow at ``utilization`` x line rate (64-bit counters)."""
    samples = []
    for ref in refs:
        words_per_s = (ref.rate_gbps or 400.0) * 1e9 / 8 / IB_WORD_BYTES * utilization
        counters = {
            "PortXmitData": int(1e12 + words_per_s * t_s),
            "PortRcvData": int(2e12 + words_per_s * 0.9 * t_s),
            "PortXmitWait": int(1e9 + 1e6 * utilization * t_s),
            "PortXmitDiscards": 0,
            "SymbolErrorCounter": 0,
            "LinkDownedCounter": 0,
            "PortRcvErrors": 0,
        }
        samples.append(
            PortCounterSample(
                ref=ref,
                counters=counters,
                wide=tuple(sorted(counters)),
                extended=True,
                mono_s=t_s,
                wall_s=1_800_000_000.0 + t_s,
            )
        )
    return SwitchSnapshot(
        node=node,
        label=label,
        taken_at_s=1_800_000_000.0 + t_s,
        samples=tuple(samples),
        notes=("synthetic",),
    )
