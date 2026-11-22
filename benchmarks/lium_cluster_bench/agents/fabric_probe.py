# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric probe: upstream switch topology, paths, counters and link tests.

Run as ``<python> fabric_probe.py <subcommand> [options] [--out PATH]``. Every
subcommand prints ONE JSON document on stdout (or writes it to ``--out``) and
exits 0 whenever JSON was produced; a missing tool, file, device or permission
never raises: the field becomes null/empty and a note is added.

Standalone on purpose: stdlib only, Python >= 3.8 syntax, no ``benchmarks.*``
imports. Tool output is parsed by pure ``parse_*`` functions over text; the
injectable ``Host`` (sysfs root, tool runner, clocks, sockets) lets tests feed
recorded output and fake sysfs trees.

Collection tiers (the domain records which tier produced each fact)
-------------------------------------------------------------------
* S  sysfs under /sys/class/infiniband and /sys/class/net: no privileges.
* M  MAD tools (smpquery, sminfo, ibtracert): need /dev/infiniband/umad* and a
     fabric that answers SMPs. SMPs use ``-t 200`` ms and a run-wide budget
     (``--max-smp``); when the very first SMP fails, every later MAD query is
     skipped and the note ``smp_blocked`` is recorded.
* L  LLDP (Ethernet ports): ``lldpctl -f json`` when lldpd runs, else an
     AF_PACKET socket per RoCE netdev (needs CAP_NET_RAW).
* T  unprivileged UDP traceroute (IP_RECVERR + MSG_ERRQUEUE, like tracepath).
* P  performance: perfquery (PMA counters, needs umad) and perftest.

Subcommands and JSON contract (``version`` 1, parsed by
``benchmarks/lium_cluster_bench/domain/fabric.py``; keep both in sync)
-----------------------------------------------------------------
Shared shapes::

    Note  = {source, error, target}           # target: "mlx5_0:1", netdev, ...
    Link  = {port, state, phys_state, lid, width_active, width_supported[],
             speed_active, speed_supported[], rate_gbps, max_rate_gbps}
            # SMP PortInfo. state "Active"|"Down"|...; widths "4X"...; speeds
            # are IB generations "SDR".."NDR","XDR" (from LinkSpeed[Ext[2]]*);
            # rate_gbps = lanes x per-lane data rate (null unless Active).
    Peer  = {node_type, node_guid, port_guid, system_guid, num_ports,
             local_port, desc, dev_id, vendor_id}
            # SMP NodeInfo (+NodeDescription). node_type "Switch" |
            # "Channel Adapter" | "Router"; local_port = the port of THAT node
            # the SMP arrived on; GUIDs "0x" + 16 lowercase hex digits.

``identify [--lldp-wait S=35] [--max-smp N=400] [--no-census]
[--census-only HCA:PORT,...]``::

    {kind: "fabric_identify", version, hostname,
     tools: {smpquery, sminfo, ibtracert, perfquery, ibv_devinfo, ib_write_bw,
             ib_write_lat, lldpctl, ip, mlnx_qos: bool},
     umad: bool,                               # /dev/infiniband/umad* exists
     smp: {budget, used, failed, blocked, exhausted},
     ports: [{hca, port, pci_bus_id, link_layer, state, phys_state, rate,
              rate_gbps, width, speed,                   # from sysfs "rate"
              lid, sm_lid, sm_sl, lmc, cap_mask,         # lid/sm_lid null if 0
              node_guid, sys_image_guid, node_desc, fw_ver, board_id,
              port_guid, subnet_prefix,                  # from gids/0
              gids: [{index, gid, type, ndev}],          # non-zero entries
              roce_v2_gid_index, roce_v2_ipv4,           # IPv4-mapped RoCE v2
              netdev, mtu, operstate, mac, ipv4: [{address, prefixlen}],
              active_mtu, max_mtu,                       # ibv_devinfo
              pcie: {current_speed_gts, current_width, max_speed_gts,
                     max_width} | null,                  # the HCA function
              neighbor: Peer + {lid, link: Link|null} | null}],
              # neighbor = node behind DR path "0,<port>"; lid = its port-0
              # LID (switch LID); link = its PortInfo for neighbor.local_port,
              # i.e. the switch side of our cable.
     sm: [{guid, lid, priority, state, state_name, activity_count,
           ports: ["mlx5_0:1", ...]}],                  # sminfo, deduplicated
     switches: {guid: {guid, desc, lid, num_ports, via: "HCA:PORT",
                       census: "complete"|"partial"|"none",
                       ports: [Link + {peer: Peer|null}]}},
              # bounded census of each neighbor switch: PortInfo of every
              # port, NodeInfo behind every active port, NodeDescription of
              # switch peers only (other tenants' hosts stay anonymous).
     lldp: {netdev: {source: "lldpctl"|"raw", chassis_id, chassis_id_subtype,
                     port_id, port_id_subtype, ttl_s, port_descr, system_name,
                     system_descr, mgmt_addresses[], port_vlan_id,
                     vlans: [{vid, name}], max_frame_size,
                     pfc: {willing, mbc, cap, enabled[]} | null,
                     ets: {willing, cbs, max_tcs, prio_tc[8], tc_bw[8],
                           tsa[8]} | null,
                     app_priority: [{priority, selector, protocol}],
                     unknown_tlvs: ["OUI/subtype", ...]}},
     qos: {netdev: {source: "mlnx_qos", trust, pfc_enabled[] | null}},
     notes: [Note], elapsed_s}

``trace --targets PATH|-`` (targets: [{rail, hca, port, src_lid, dst_lid,
src_ip, dst_ip, netdev?, src_node?, dst_node?}])::

    {kind: "fabric_trace", version, hostname, notes, elapsed_s,
     results: [{target, method: "ibtracert"|"udp"|"none", hop_count, reached,
                error, elapsed_s,
                hops: [{ttl, node_type, guid, desc, lid, in_port, out_port,
                        ip, rtt_ms, icmp_type, icmp_code}]}]}

IB (LIDs given and ibtracert present): ``hops`` lists the switches in path
order (in_port = where the trace entered, out_port = where it left) and
``hop_count`` counts them. Otherwise (IPs given) an unprivileged UDP
traceroute bound to ``src_ip``: ``hops`` are the routers that answered TTL
1..8 (``ip`` null for silent hops) and ``hop_count`` = routers before the
destination (0 = same L2 segment); null when the destination never answered.

``perfquery --ports PATH|-`` (ports: [{hca, port, lid, switch_port, key?}])::

    {kind: "fabric_perfquery", version, hostname, notes, elapsed_s,
     ports: [{key, hca, port, lid, switch_port, extended, wide[],
              counters: {PortXmitData, PortRcvData, PortXmitWait,
                         PortXmitDiscards, SymbolErrorCounter, ...: int},
              mono_s, wall_s, error}]}

Counters are the raw values perfquery prints: PortXmitData / PortRcvData
count 4-byte words. ``-x`` (64-bit PortCountersExtended) is tried first, then
the 32-bit saturating PortCounters; ``wide`` names the counters that are
64-bit. ``mono_s``/``wall_s`` are taken around each query (midpoint).

``perftest --role server|client --rails PATH|- [--peer IP]`` (rails:
[{rail, hca, port, gid_index, bw_tcp_port, lat_tcp_port}])::

    {kind: "fabric_perftest", version, hostname, role, peer, notes, elapsed_s,
     results: [{rail, hca, port, test: "bw"|"lat", rc, attempts, elapsed_s,
                error, argv[],
                bw: {msg_bytes, iterations, unit, bw_peak, bw_avg, msg_rate_mpps,
                     bw_peak_gbs, bw_avg_gbs} | null,
                lat: {msg_bytes, iterations, t_min_us, t_max_us, t_typical_us,
                      t_avg_us, t_stdev_us, p99_us, p999_us} | null}]}

``ib_write_bw`` runs with ``--report_gbits`` so ``unit`` is "Gb/sec" (decimal
gigabits); "MB/sec" output (perftest MB = 2**20 bytes) is also understood.
``*_gbs`` are GB/s (1e9 bytes/s). In ``-D`` (duration) mode perftest prints a
peak of 0.00, kept as reported. Latencies are perftest's (half round trip for
write_lat) in microseconds. The server runs every rail's bw and lat servers at
once on distinct TCP ports and exits when they finish or after ``--timeout``;
the client retries the TCP connection for up to ``--connect-wait`` seconds,
then runs bw on every rail concurrently, then lat. When a client test fails
it connects to that server port once and hangs up, so the server's
handshake fails and it exits instead of waiting for ``--timeout``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

VERSION = 1
TOOL_TIMEOUT_S = 10.0
SMP_TIMEOUT_MS = 200
SMP_PROC_TIMEOUT_S = 5.0
DEFAULT_MAX_SMP = 400
DEFAULT_LLDP_WAIT_S = 35.0
MAD_THREADS = 8
TRACE_THREADS = 8
PERFQUERY_THREADS = 32
UDP_MAX_TTL = 8
UDP_HOP_WAIT_S = 0.4
UDP_PORT_BASE = 33434
DEFAULT_BW_SECONDS = 3
DEFAULT_BW_SIZE = 65536
DEFAULT_BW_QPS = 2
DEFAULT_LAT_ITERS = 2000
DEFAULT_LAT_SIZE = 2
DEFAULT_PERFTEST_TIMEOUT_S = 60.0
DEFAULT_CONNECT_WAIT_S = 20.0
CONNECT_RETRY_S = 0.5
POKE_TIMEOUT_S = 2.0
PERFTEST_SLACK_S = 30.0

TOOLS = (
    "smpquery",
    "sminfo",
    "ibtracert",
    "perfquery",
    "ibv_devinfo",
    "ib_write_bw",
    "ib_write_lat",
    "lldpctl",
    "ip",
    "mlnx_qos",
)

# Per-lane data rate (Gb/s) of each IB generation, as the kernel's sysfs
# "rate" reports it (4X NDR = 400 Gb/sec).
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
# PortInfo LinkSpeed{,Ext,Ext2}* print the per-lane signalling rate.
SIGNAL_GBPS_TO_SPEED = (
    (2.5, "SDR"),
    (5.0, "DDR"),
    (10.0, "QDR"),
    (14.0625, "FDR"),
    (25.78125, "EDR"),
    (53.125, "HDR"),
    (106.25, "NDR"),
    (212.5, "XDR"),
)
WIDTH_LANES = {"1X": 1, "2X": 2, "4X": 4, "8X": 8, "12X": 12}

# LLDP
ETH_P_LLDP = 0x88CC
ETH_P_8021Q = 0x8100
LLDP_MULTICAST = (
    b"\x01\x80\xc2\x00\x00\x0e",  # nearest bridge
    b"\x01\x80\xc2\x00\x00\x03",  # nearest non-TPMR bridge
    b"\x01\x80\xc2\x00\x00\x00",  # nearest customer bridge
)
# Linux <linux/if_packet.h>; Python's socket module does not export them.
SOL_PACKET = 263
PACKET_ADD_MEMBERSHIP = 1
PACKET_MR_MULTICAST = 0
OUI_8021 = b"\x00\x80\xc2"
OUI_8023 = b"\x00\x12\x0f"
CHASSIS_SUBTYPES = {
    1: "chassis_component",
    2: "ifalias",
    3: "port_component",
    4: "mac",
    5: "ip",
    6: "ifname",
    7: "local",
}
PORT_SUBTYPES = {
    1: "ifalias",
    2: "port_component",
    3: "mac",
    4: "ip",
    5: "ifname",
    6: "agent_circuit_id",
    7: "local",
}

# UDP traceroute (<linux/errqueue.h>, <linux/in.h>). Python exposes
# IP_RECVERR only on some builds, hence the numeric fallbacks.
IP_RECVERR = getattr(socket, "IP_RECVERR", 11)
SOL_IP = getattr(socket, "SOL_IP", 0)
MSG_ERRQUEUE = getattr(socket, "MSG_ERRQUEUE", 0x2000)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
SO_EE_ORIGIN_ICMP = 2
ICMP_DEST_UNREACH = 3
ICMP_PORT_UNREACH = 3  # code of ICMP_DEST_UNREACH
SOCK_EE_FORMAT = "=IBBBBII"
SOCK_EE_SIZE = struct.calcsize(SOCK_EE_FORMAT)
SOCKADDR_IN_FORMAT = "=HH4s8x"
SOCKADDR_IN_SIZE = struct.calcsize(SOCKADDR_IN_FORMAT)
TRACE_MAGIC = b"CBFAB1"

# SIOCGIFADDR / SIOCGIFNETMASK (<linux/sockios.h>).
SIOCGIFADDR = 0x8915
SIOCGIFNETMASK = 0x891B

# (quoted payload, [(cmsg_level, cmsg_type, cmsg_data)], original destination)
ErrQueueEntry = Tuple[bytes, List[Tuple[int, int, bytes]], Any]

_SYSFS_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_RATE_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*Gb/s(?:ec)?(?:\s*\((\d+X)\s+(\w+)\))?", re.IGNORECASE
)
_GT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s")
_DUMP_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\.*(.*)$")
_GBPS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*Gbps")
_NODEDESC_RE = re.compile(r"^Node Description:\.+(.*)$")
_SMINFO_RE = re.compile(
    r"sm lid (?P<lid>\d+) sm guid (?P<guid>0x[0-9a-fA-F]+), activity count "
    r"(?P<act>\d+) priority (?P<prio>\d+) state (?P<state>\d+)(?:\s+(?P<name>\S+))?"
)
_DEVINFO_MTU_RE = re.compile(r"^\s*(active_mtu|max_mtu):\s*(\d+)", re.MULTILINE)
_TRACE_END_RE = re.compile(
    r"^(?P<dir>From|To)\s+(?P<type>.+?)\s+\{(?P<guid>0x[0-9a-fA-F]+)\}\s+portnum\s+"
    r"(?P<port>\d+)\s+lid\s+(?P<lid>\d+)(?:-\d+)?\s+\"(?P<desc>.*)\"\s*$"
)
_TRACE_HOP_RE = re.compile(
    r"^\[(?P<out>\d+)\]\s*->\s*(?P<type>.+?)\s+port\s+\{(?P<guid>0x[0-9a-fA-F]+)\}"
    r"\[(?P<in>\d+)\]\s+lid\s+(?P<lid>\d+)(?:-\d+)?\s+\"(?P<desc>.*)\"\s*$"
)
_BW_UNIT_RE = re.compile(r"BW average\[([^\]]+)\]")
_LAT_COLUMNS_RE = re.compile(
    r"#bytes|#iterations|t_min|t_max|t_typical|t_avg|t_stdev|99\.9%|99%"
)
_NUMBER_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")
_QOS_TRUST_RE = re.compile(r"Priority trust state:\s*(\S+)", re.IGNORECASE)
_PERFTEST_CONNECT_RE = re.compile(
    r"Couldn't connect|Unable to init the socket connection|Failed to connect",
    re.IGNORECASE,
)

LAT_KEYS = {
    "#bytes": "msg_bytes",
    "#iterations": "iterations",
    "t_min": "t_min_us",
    "t_max": "t_max_us",
    "t_typical": "t_typical_us",
    "t_avg": "t_avg_us",
    "t_stdev": "t_stdev_us",
    "99%": "p99_us",
    "99.9%": "p999_us",
}
# PortCounters (non -x) field widths; every PortCountersExtended field is 64.
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
NOT_COUNTERS = frozenset({"PortSelect", "CounterSelect", "CounterSelect2"})


# --------------------------------------------------------------------------
# Tool runner and injectable host
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def run_tool(
    argv: Sequence[str], timeout: float = TOOL_TIMEOUT_S
) -> Optional[ToolResult]:
    """Run an external tool; ``None`` when it is not installed."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return ToolResult(124, out, f"timed out after {timeout:.1f}s")
    return ToolResult(proc.returncode, proc.stdout, proc.stderr)


def listen_lldp(
    netdevs: Sequence[str], wait_s: float
) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """First LLDP frame per netdev within ``wait_s`` (raw AF_PACKET sockets).

    Returns (frames, errors): netdev -> raw Ethernet frame, netdev -> error.
    All netdevs are listened to concurrently; a netdev stops at its first frame.
    """
    frames: Dict[str, bytes] = {}
    errors: Dict[str, str] = {}
    family = getattr(socket, "AF_PACKET", None)  # Linux only
    if family is None:
        return frames, {nd: "AF_PACKET unsupported on this OS" for nd in netdevs}
    socks: Dict[Any, str] = {}
    try:
        for netdev in netdevs:
            try:
                sock = socket.socket(family, socket.SOCK_RAW, socket.htons(ETH_P_LLDP))
            except OSError as exc:
                errors[netdev] = f"raw socket: {exc}"
                continue
            try:
                sock.bind((netdev, 0))
                sock.setblocking(False)
            except OSError as exc:
                sock.close()
                errors[netdev] = f"bind: {exc}"
                continue
            _join_lldp_multicast(sock, netdev)
            socks[sock] = netdev
        deadline = time.monotonic() + wait_s
        while socks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select(list(socks), [], [], remaining)
            for sock in ready:
                try:
                    frame = sock.recv(65535)
                except OSError:
                    continue
                if parse_lldp_frame(frame) is None:
                    continue
                frames[socks.pop(sock)] = frame
                sock.close()
    finally:
        for sock in socks:
            sock.close()
    return frames, errors


def _join_lldp_multicast(sock: Any, netdev: str) -> None:
    """Ask the NIC to accept the LLDP multicast MACs (best effort)."""
    try:
        index = socket.if_nametoindex(netdev)
    except OSError:
        return
    for mac in LLDP_MULTICAST:
        mreq = struct.pack("iHH8s", index, PACKET_MR_MULTICAST, 6, mac)
        try:
            sock.setsockopt(SOL_PACKET, PACKET_ADD_MEMBERSHIP, mreq)
        except OSError:
            return  # unprivileged or unsupported: rely on the default filter


def ioctl_ipv4(netdev: str) -> List[Dict[str, Any]]:
    """Primary IPv4 address of ``netdev`` via SIOCGIFADDR (no ``ip`` tool)."""
    name = struct.pack("256s", netdev.encode()[:15])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            addr = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, name)[20:24]
            mask = fcntl.ioctl(sock.fileno(), SIOCGIFNETMASK, name)[20:24]
    except OSError:
        return []
    prefix = bin(struct.unpack("!I", mask)[0]).count("1")
    return [{"address": socket.inet_ntoa(addr), "prefixlen": prefix}]


class UdpProbeSocket:
    """UDP socket for unprivileged traceroute: TTL control + error queue."""

    def __init__(self, src_ip: Optional[str], netdev: Optional[str]) -> None:
        self.notes: List[str] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(SOL_IP, IP_RECVERR, 1)
            if netdev:
                try:
                    self._sock.setsockopt(
                        socket.SOL_SOCKET, SO_BINDTODEVICE, netdev.encode()
                    )
                except OSError as exc:
                    self.notes.append(f"SO_BINDTODEVICE {netdev}: {exc}")
            if src_ip:
                self._sock.bind((src_ip, 0))
        except OSError:
            self._sock.close()
            raise

    def send(self, dst_ip: str, port: int, ttl: int, payload: bytes) -> None:
        self._sock.setsockopt(SOL_IP, socket.IP_TTL, ttl)
        self._sock.sendto(payload, (dst_ip, port))

    def wait_error(self, timeout_s: float) -> Optional[ErrQueueEntry]:
        """The next queued error: (quoted payload, ancillary data, original
        destination (ip, port)), or ``None`` on timeout.

        A pending error makes the socket readable (POLLERR is in select's
        read set); ``recvmsg(MSG_ERRQUEUE)`` never blocks.
        """
        ready, _, _ = select.select([self._sock], [], [], max(0.0, timeout_s))
        if not ready:
            return None
        try:
            data, ancdata, _, address = self._sock.recvmsg(512, 512, MSG_ERRQUEUE)
        except BlockingIOError:
            # Readable without a queued error: drop the regular datagram so
            # the next select() does not spin on it.
            try:
                self._sock.recv(512, socket.MSG_DONTWAIT)
            except OSError:
                pass
            return None
        except OSError:
            return None
        return data, list(ancdata), address

    def close(self) -> None:
        self._sock.close()


def open_udp_probe(src_ip: Optional[str], netdev: Optional[str]) -> UdpProbeSocket:
    return UdpProbeSocket(src_ip, netdev)


def poke_tcp(address: str, port: int) -> bool:
    """Connect and hang up at once: a perftest server whose client gave up
    fails its handshake and exits instead of waiting for ``--timeout``."""
    try:
        with socket.create_connection((address, port), timeout=POKE_TIMEOUT_S):
            return True
    except OSError:
        return False


@dataclass
class Host:
    """Everything the probe touches; tests substitute fakes.

    ``root`` prefixes every /sys and /dev path; ``run`` executes tools.
    """

    root: str = "/"
    run: Callable[..., Optional[ToolResult]] = run_tool
    which: Callable[[str], Optional[str]] = shutil.which
    monotonic: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    hostname: Callable[[], str] = socket.gethostname
    lldp_listener: Callable[
        [Sequence[str], float], Tuple[Dict[str, bytes], Dict[str, str]]
    ] = listen_lldp
    ipv4_lookup: Callable[[str], List[Dict[str, Any]]] = ioctl_ipv4
    udp_probe: Callable[[Optional[str], Optional[str]], Any] = open_udp_probe
    poke: Callable[[str, int], bool] = poke_tcp

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *(p.lstrip("/") for p in parts))


def note(source: str, error: str, target: Optional[str] = None) -> Dict[str, Any]:
    return {"source": source, "error": error, "target": target}


def _tail(result: Optional[ToolResult], limit: int = 300) -> str:
    if result is None:
        return "not installed"
    text = (result.stderr.strip() or result.stdout.strip()).splitlines()
    last = text[-1] if text else ""
    return f"exit {result.rc}: {last}"[:limit]


# --------------------------------------------------------------------------
# Pure parsers: sysfs values
# --------------------------------------------------------------------------


def natural_key(text: str) -> List[Any]:
    """Sort key ordering mlx5_2 before mlx5_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def parse_int(text: Optional[str]) -> Optional[int]:
    """Decimal or 0x-prefixed integer; ``None`` when unparseable."""
    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(value, 16) if re.match(r"^[0-9a-fA-F]+$", value) else None
        except ValueError:
            return None


def parse_lid(text: Optional[str]) -> Optional[int]:
    """sysfs "0x1a" -> 26; LID 0 (RoCE, unconfigured) -> None."""
    value = parse_int(text)
    return value if value else None


def normalize_guid(text: Optional[str]) -> Optional[str]:
    """sysfs "b83f:d203:00a5:1b2c" or "0xB83F..." -> "0xb83fd20300a51b2c"."""
    if text is None:
        return None
    raw = text.strip().lower()
    if raw.startswith("0x"):
        raw = raw[2:]
    raw = raw.replace(":", "")
    if not raw or not re.match(r"^[0-9a-f]+$", raw) or len(raw) > 16:
        return None
    return "0x" + raw.zfill(16)


def parse_rate(rate: Optional[str]) -> Tuple[float, Optional[str], Optional[str]]:
    """sysfs "400 Gb/sec (4X NDR)" -> (400.0, "4X", "NDR")."""
    match = _RATE_RE.search(rate or "")
    if match is None:
        return 0.0, None, None
    return float(match.group(1)), match.group(2), match.group(3)


def normalize_gid(text: Optional[str]) -> Optional[str]:
    """A GID as 8 lowercase 4-digit groups; ``None`` when empty or zero."""
    if text is None:
        return None
    groups = text.strip().lower().split(":")
    if len(groups) != 8 or not all(re.match(r"^[0-9a-f]{1,4}$", g) for g in groups):
        return None
    groups = [g.zfill(4) for g in groups]
    if all(g == "0000" for g in groups):
        return None
    return ":".join(groups)


def gid_prefix(gid: str) -> str:
    """Subnet prefix (upper 64 bits) as "0xfe80000000000000"."""
    return "0x" + "".join(gid.split(":")[:4])


def gid_interface_id(gid: str) -> str:
    """Interface id (lower 64 bits; the port GUID on IB) as "0x..."."""
    return "0x" + "".join(gid.split(":")[4:])


def gid_ipv4(gid: str) -> Optional[str]:
    """IPv4 of an IPv4-mapped GID (::ffff:a.b.c.d), else ``None``."""
    groups = gid.split(":")
    if groups[:6] != ["0000"] * 5 + ["ffff"]:
        return None
    raw = bytes.fromhex(groups[6] + groups[7])
    return socket.inet_ntoa(raw)


def parse_pci_speed(text: Optional[str]) -> Optional[float]:
    """ "32.0 GT/s PCIe" / "8 GT/s" -> GT/s; "Unknown" -> None."""
    match = _GT_RE.search(text or "")
    return float(match.group(1)) if match else None


def parse_pci_width(text: Optional[str]) -> Optional[int]:
    value = parse_int(text)
    return value if value else None


# --------------------------------------------------------------------------
# Pure parsers: MAD tools
# --------------------------------------------------------------------------


def parse_mad_dump(text: str) -> Dict[str, str]:
    """``Name:.....value`` lines of libibmad dumps (``#`` headers skipped)."""
    fields: Dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith(("#", "\t", " ")):
            continue
        match = _DUMP_RE.match(line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def parse_nodeinfo(text: str) -> Optional[Dict[str, Any]]:
    """``smpquery nodeinfo`` -> Peer fields (``desc`` left null)."""
    fields = parse_mad_dump(text)
    if "NodeType" not in fields:
        return None
    return {
        "node_type": fields.get("NodeType") or None,
        "node_guid": normalize_guid(fields.get("Guid")),
        "port_guid": normalize_guid(fields.get("PortGuid")),
        "system_guid": normalize_guid(fields.get("SystemGuid")),
        "num_ports": parse_int(fields.get("NumPorts")),
        "local_port": parse_int(fields.get("LocalPort")),
        "desc": None,
        "dev_id": fields.get("DevId") or None,
        "vendor_id": fields.get("VendorId") or None,
    }


def parse_nodedesc(text: str) -> Optional[str]:
    """``smpquery nodedesc``: "Node Description:......<name>"."""
    for line in text.splitlines():
        match = _NODEDESC_RE.match(line.strip())
        if match:
            return match.group(1).strip() or None
    return None


def _speed_of_signal(gbps: float) -> Optional[str]:
    for signal, name in SIGNAL_GBPS_TO_SPEED:
        if abs(signal - gbps) < 1e-3:
            return name
    return None


def parse_speeds(text: Optional[str]) -> List[str]:
    """ "2.5 Gbps or 5.0 Gbps or 10.0 Gbps" -> ["SDR", "DDR", "QDR"]."""
    speeds: List[str] = []
    for match in _GBPS_RE.finditer(text or ""):
        name = _speed_of_signal(float(match.group(1)))
        if name and name not in speeds:
            speeds.append(name)
    return speeds


def parse_widths(text: Optional[str]) -> List[str]:
    """ "1X or 2X or 4X" -> ["1X", "2X", "4X"]."""
    found = re.findall(r"\b(1|2|4|8|12)X\b", text or "")
    widths: List[str] = []
    for lanes in found:
        width = f"{lanes}X"
        if width not in widths:
            widths.append(width)
    return widths


def _lane_rate(speed: Optional[str]) -> Optional[float]:
    return LANE_GBPS.get(speed or "")


def link_rate_gbps(width: Optional[str], speed: Optional[str]) -> Optional[float]:
    lanes = WIDTH_LANES.get(width or "")
    lane = _lane_rate(speed)
    if lanes is None or lane is None:
        return None
    return lanes * lane


def parse_portinfo(text: str, port: int) -> Optional[Dict[str, Any]]:
    """``smpquery portinfo`` -> Link (effective speed = Ext2 > Ext > legacy)."""
    fields = parse_mad_dump(text)
    if "LinkState" not in fields and "LinkWidthActive" not in fields:
        return None
    active = None
    for key in ("LinkSpeedExt2Active", "LinkSpeedExtActive", "LinkSpeedActive"):
        speeds = parse_speeds(fields.get(key))
        if speeds:
            active = speeds[0]
            break
    supported: List[str] = []
    for key in (
        "LinkSpeedSupported",
        "LinkSpeedExtSupported",
        "LinkSpeedExt2Supported",
    ):
        for speed in parse_speeds(fields.get(key)):
            if speed not in supported:
                supported.append(speed)
    supported.sort(key=lambda s: LANE_GBPS.get(s, 0.0))
    widths_active = parse_widths(fields.get("LinkWidthActive"))
    width_active = widths_active[0] if widths_active else None
    widths_supported = parse_widths(fields.get("LinkWidthSupported"))
    state = fields.get("LinkState") or None
    rate = link_rate_gbps(width_active, active) if state == "Active" else None
    max_rate = None
    if widths_supported and supported:
        max_lanes = max(WIDTH_LANES[w] for w in widths_supported)
        max_rate = max_lanes * max(LANE_GBPS[s] for s in supported)
    return {
        "port": port,
        "state": state,
        "phys_state": fields.get("PhysLinkState") or None,
        "lid": parse_int(fields.get("Lid")),
        "width_active": width_active,
        "width_supported": widths_supported,
        "speed_active": active,
        "speed_supported": supported,
        "rate_gbps": rate,
        "max_rate_gbps": max_rate,
    }


def parse_sminfo(text: str) -> Optional[Dict[str, Any]]:
    """ "sminfo: sm lid 1 sm guid 0x..., activity count N priority P state S NAME"."""
    match = _SMINFO_RE.search(text or "")
    if match is None:
        return None
    return {
        "guid": normalize_guid(match.group("guid")),
        "lid": int(match.group("lid")),
        "priority": int(match.group("prio")),
        "state": int(match.group("state")),
        "state_name": match.group("name"),
        "activity_count": int(match.group("act")),
    }


def parse_ibv_devinfo(text: str) -> Dict[str, Optional[int]]:
    """active_mtu / max_mtu from ``ibv_devinfo -d HCA -i PORT``."""
    values: Dict[str, Optional[int]] = {"active_mtu": None, "max_mtu": None}
    for key, value in _DEVINFO_MTU_RE.findall(text or ""):
        values[key] = int(value)
    return values


def parse_ibtracert(text: str) -> Dict[str, Any]:
    """``ibtracert`` -> switch hops in order, the endpoints, and whether the
    destination CA was reached."""
    src: Optional[Dict[str, Any]] = None
    dst: Optional[Dict[str, Any]] = None
    entries: List[Dict[str, Any]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        end = _TRACE_END_RE.match(line)
        if end:
            info = {
                "node_type": end.group("type"),
                "guid": normalize_guid(end.group("guid")),
                "port": int(end.group("port")),
                "lid": int(end.group("lid")),
                "desc": end.group("desc"),
            }
            if end.group("dir") == "From":
                src = info
            else:
                dst = info
            continue
        hop = _TRACE_HOP_RE.match(line)
        if hop:
            entries.append(
                {
                    "node_type": hop.group("type"),
                    "guid": normalize_guid(hop.group("guid")),
                    "desc": hop.group("desc"),
                    "lid": int(hop.group("lid")),
                    "in_port": int(hop.group("in")),
                    "arrived_via": int(hop.group("out")),
                }
            )
    hops: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if entry["node_type"].lower() != "switch":
            continue
        out_port = (
            entries[index + 1]["arrived_via"] if index + 1 < len(entries) else None
        )
        hops.append(
            {
                "ttl": None,
                "node_type": "Switch",
                "guid": entry["guid"],
                "desc": entry["desc"],
                "lid": entry["lid"],
                "in_port": entry["in_port"],
                "out_port": out_port,
                "ip": None,
                "rtt_ms": None,
                "icmp_type": None,
                "icmp_code": None,
            }
        )
    reached = dst is not None or bool(
        entries and entries[-1]["node_type"].lower() != "switch"
    )
    return {"src": src, "dst": dst, "hops": hops, "reached": reached}


def parse_perfquery(text: str) -> Dict[str, int]:
    """``perfquery [-x]`` -> counter name -> raw value."""
    counters: Dict[str, int] = {}
    for key, value in parse_mad_dump(text).items():
        if key in NOT_COUNTERS:
            continue
        number = parse_int(value)
        if number is not None:
            counters[key] = number
    return counters


# --------------------------------------------------------------------------
# Pure parsers: host tools, LLDP, perftest, errqueue
# --------------------------------------------------------------------------


def parse_ip_addr_json(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """``ip -j -4 addr show`` -> ifname -> [{address, prefixlen}]."""
    try:
        data = json.loads(text or "[]")
    except json.JSONDecodeError:
        return {}
    result: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(data, list):
        return result
    for link in data:
        if not isinstance(link, dict) or not link.get("ifname"):
            continue
        addrs = []
        for info in link.get("addr_info") or []:
            if isinstance(info, dict) and info.get("family", "inet") == "inet":
                if info.get("local"):
                    addrs.append(
                        {
                            "address": str(info["local"]),
                            "prefixlen": parse_int(str(info.get("prefixlen", ""))),
                        }
                    )
        result[str(link["ifname"])] = addrs
    return result


def parse_mlnx_qos(text: str) -> Dict[str, Any]:
    """Trust mode and PFC-enabled priorities from ``mlnx_qos -i NETDEV``."""
    trust_match = _QOS_TRUST_RE.search(text or "")
    pfc: Optional[List[int]] = None
    for line in (text or "").splitlines():
        tokens = line.split()
        if len(tokens) == 9 and tokens[0] == "enabled":
            if all(t in ("0", "1") for t in tokens[1:]):
                pfc = [i for i, t in enumerate(tokens[1:]) if t == "1"]
    return {
        "source": "mlnx_qos",
        "trust": trust_match.group(1) if trust_match else None,
        "pfc_enabled": pfc,
    }


def empty_lldp(source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "chassis_id": None,
        "chassis_id_subtype": None,
        "port_id": None,
        "port_id_subtype": None,
        "ttl_s": None,
        "port_descr": None,
        "system_name": None,
        "system_descr": None,
        "mgmt_addresses": [],
        "port_vlan_id": None,
        "vlans": [],
        "max_frame_size": None,
        "pfc": None,
        "ets": None,
        "app_priority": [],
        "unknown_tlvs": [],
    }


def _first(value: Any) -> Any:
    """lldpctl JSON turns repeated keys into lists; take the first element."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text_value(value: Any) -> Optional[str]:
    value = _first(value)
    if isinstance(value, dict):
        value = value.get("value")
    return None if value is None else str(value)


def _lldpctl_interfaces(data: Any) -> List[Tuple[str, Dict[str, Any]]]:
    lldp = _first(data.get("lldp")) if isinstance(data, dict) else None
    if not isinstance(lldp, dict):
        return []
    interfaces = lldp.get("interface")
    items: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(interfaces, dict):
        if "name" in interfaces:  # json0 single entry
            interfaces = [interfaces]
        else:
            return [(k, v) for k, v in interfaces.items() if isinstance(v, dict)]
    for entry in _as_list(interfaces):
        if not isinstance(entry, dict):
            continue
        if "name" in entry:
            items.append((str(_text_value(entry["name"])), entry))
        else:
            items.extend((k, v) for k, v in entry.items() if isinstance(v, dict))
    return items


def parse_lldpctl_json(text: str) -> Dict[str, Dict[str, Any]]:
    """``lldpctl -f json`` (dict-per-name or list form) -> netdev -> LLDP."""
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for name, body in _lldpctl_interfaces(data):
        record = empty_lldp("lldpctl")
        chassis = _first(body.get("chassis"))
        if isinstance(chassis, dict) and "id" not in chassis and len(chassis) == 1:
            sysname, chassis = next(iter(chassis.items()))
            record["system_name"] = str(sysname)
            chassis = _first(chassis)
        if isinstance(chassis, dict):
            ident = _first(chassis.get("id"))
            if isinstance(ident, dict):
                record["chassis_id"] = _text_value(ident.get("value"))
                record["chassis_id_subtype"] = _text_value(ident.get("type"))
            if record["system_name"] is None and chassis.get("name") is not None:
                record["system_name"] = _text_value(chassis.get("name"))
            record["system_descr"] = _text_value(chassis.get("descr"))
            record["mgmt_addresses"] = [
                str(_text_value(a)) for a in _as_list(chassis.get("mgmt-ip"))
            ]
        port = _first(body.get("port"))
        if isinstance(port, dict):
            ident = _first(port.get("id"))
            if isinstance(ident, dict):
                record["port_id"] = _text_value(ident.get("value"))
                record["port_id_subtype"] = _text_value(ident.get("type"))
            record["port_descr"] = _text_value(port.get("descr"))
            record["ttl_s"] = parse_int(_text_value(port.get("ttl")))
            record["max_frame_size"] = parse_int(_text_value(port.get("mfs")))
        if record["ttl_s"] is None:
            record["ttl_s"] = parse_int(_text_value(body.get("ttl")))
        for vlan in _as_list(body.get("vlan")):
            if not isinstance(vlan, dict):
                continue
            vid = parse_int(_text_value(vlan.get("vlan-id")))
            if vid is None:
                continue
            record["vlans"].append({"vid": vid, "name": _text_value(vlan.get("value"))})
            if vlan.get("pvid") in (True, "yes", "true"):
                record["port_vlan_id"] = vid
        result[name] = record
    return result


def _mac(raw: bytes) -> str:
    return ":".join("%02x" % b for b in raw)


def _id_value(subtype: int, data: bytes, mac_subtype: int, addr_subtype: int) -> str:
    if subtype == mac_subtype and len(data) == 6:
        return _mac(data)
    if subtype == addr_subtype and len(data) == 5 and data[0] == 1:
        return socket.inet_ntoa(data[1:])
    return data.decode("utf-8", errors="replace")


def _parse_ets(info: bytes) -> Optional[Dict[str, Any]]:
    if len(info) < 21:
        return None
    first = info[0]
    prio_tc: List[int] = []
    for byte in info[1:5]:
        prio_tc += [byte >> 4, byte & 0x0F]
    return {
        "willing": bool(first & 0x80),
        "cbs": bool(first & 0x40),
        "max_tcs": (first & 0x07) or 8,
        "prio_tc": prio_tc,
        "tc_bw": list(info[5:13]),
        "tsa": list(info[13:21]),
    }


def _parse_pfc(info: bytes) -> Optional[Dict[str, Any]]:
    if len(info) < 2:
        return None
    first, enable = info[0], info[1]
    return {
        "willing": bool(first & 0x80),
        "mbc": bool(first & 0x40),
        "cap": first & 0x0F,
        "enabled": [p for p in range(8) if enable & (1 << p)],
    }


def _parse_app(info: bytes) -> List[Dict[str, int]]:
    entries = []
    body = info[1:]  # first octet reserved
    for offset in range(0, len(body) - 2, 3):
        head = body[offset]
        entries.append(
            {
                "priority": head >> 5,
                "selector": head & 0x07,
                "protocol": struct.unpack("!H", body[offset + 1 : offset + 3])[0],
            }
        )
    return entries


def _parse_org_tlv(record: Dict[str, Any], value: bytes) -> None:
    if len(value) < 4:
        return
    oui, subtype, info = value[:3], value[3], value[4:]
    if oui == OUI_8021:
        if subtype == 1 and len(info) >= 2:
            record["port_vlan_id"] = struct.unpack("!H", info[:2])[0]
            return
        if subtype == 3 and len(info) >= 3:
            vid = struct.unpack("!H", info[:2])[0]
            name = info[3 : 3 + info[2]].decode("utf-8", errors="replace")
            record["vlans"].append({"vid": vid, "name": name})
            return
        if subtype == 9:
            record["ets"] = _parse_ets(info)
            return
        if subtype == 11:
            record["pfc"] = _parse_pfc(info)
            return
        if subtype == 12:
            record["app_priority"] = _parse_app(info)
            return
    if oui == OUI_8023 and subtype == 4 and len(info) >= 2:
        record["max_frame_size"] = struct.unpack("!H", info[:2])[0]
        return
    record["unknown_tlvs"].append("%s/%d" % (_mac(oui).replace(":", "-"), subtype))


def parse_lldp_frame(frame: bytes) -> Optional[Dict[str, Any]]:
    """Decode a raw Ethernet LLDP frame (optionally 802.1Q tagged)."""
    if len(frame) < 16:
        return None
    ethertype = struct.unpack("!H", frame[12:14])[0]
    pos = 14
    if ethertype == ETH_P_8021Q and len(frame) >= 18:
        ethertype = struct.unpack("!H", frame[16:18])[0]
        pos = 18
    if ethertype != ETH_P_LLDP:
        return None
    record = empty_lldp("raw")
    while pos + 2 <= len(frame):
        header = struct.unpack("!H", frame[pos : pos + 2])[0]
        tlv_type, length = header >> 9, header & 0x1FF
        value = frame[pos + 2 : pos + 2 + length]
        pos += 2 + length
        if tlv_type == 0 or len(value) < length:
            break
        if tlv_type == 1 and value:
            record["chassis_id_subtype"] = CHASSIS_SUBTYPES.get(value[0], str(value[0]))
            record["chassis_id"] = _id_value(value[0], value[1:], 4, 5)
        elif tlv_type == 2 and value:
            record["port_id_subtype"] = PORT_SUBTYPES.get(value[0], str(value[0]))
            record["port_id"] = _id_value(value[0], value[1:], 3, 4)
        elif tlv_type == 3 and len(value) >= 2:
            record["ttl_s"] = struct.unpack("!H", value[:2])[0]
        elif tlv_type in (4, 5, 6):
            key = {4: "port_descr", 5: "system_name", 6: "system_descr"}[tlv_type]
            record[key] = value.decode("utf-8", errors="replace").rstrip("\x00")
        elif tlv_type == 8 and len(value) >= 2:
            addr_len, subtype = value[0], value[1]
            addr = value[2 : 1 + addr_len]
            if subtype == 1 and len(addr) == 4:
                record["mgmt_addresses"].append(socket.inet_ntoa(addr))
            elif subtype == 2 and len(addr) == 16:
                record["mgmt_addresses"].append(socket.inet_ntop(socket.AF_INET6, addr))
            elif subtype == 6 and len(addr) == 6:
                record["mgmt_addresses"].append(_mac(addr))
        elif tlv_type == 127:
            _parse_org_tlv(record, value)
    if record["chassis_id"] is None and record["port_id"] is None:
        return None
    return record


def _perftest_table(text: str, marker: str) -> Tuple[Optional[str], List[List[str]]]:
    """Header line containing ``marker`` and the numeric rows below it."""
    header: Optional[str] = None
    rows: List[List[str]] = []
    for line in (text or "").splitlines():
        if marker in line and "#bytes" in line:
            header = line
            continue
        if header is None:
            continue
        tokens = line.split()
        if tokens and all(_NUMBER_RE.match(t) for t in tokens):
            rows.append(tokens)
        elif tokens and tokens[0].startswith("---") and rows:
            break
    return header, rows


def _unit_to_gbs(unit: str) -> Optional[float]:
    """Factor turning one ``unit`` into GB/s (1e9 bytes/s)."""
    lowered = unit.lower()
    if lowered.startswith("gb/") and unit[1] == "b":  # Gb/sec: gigabits
        return 1.0 / 8.0
    if lowered.startswith(("mb/", "mib/")):  # perftest MB = 2**20 bytes
        return 1048576.0 / 1e9
    if unit.startswith(("GB/", "GiB/")):
        return 1073741824.0 / 1e9
    return None


def parse_perftest_bw(text: str) -> Optional[Dict[str, Any]]:
    """``ib_write_bw`` result table (best row by average bandwidth)."""
    header, rows = _perftest_table(text, "BW average")
    if header is None or not rows:
        return None
    unit_match = _BW_UNIT_RE.search(header)
    unit = unit_match.group(1) if unit_match else "Gb/sec"
    factor = _unit_to_gbs(unit)
    best = max(rows, key=lambda r: float(r[3]) if len(r) > 3 else 0.0)
    if len(best) < 4:
        return None
    peak, avg = float(best[2]), float(best[3])
    return {
        "msg_bytes": int(float(best[0])),
        "iterations": int(float(best[1])),
        "unit": unit,
        "bw_peak": peak,
        "bw_avg": avg,
        "msg_rate_mpps": float(best[4]) if len(best) > 4 else None,
        "bw_peak_gbs": None if factor is None else peak * factor,
        "bw_avg_gbs": None if factor is None else avg * factor,
    }


def parse_perftest_lat(text: str) -> Optional[Dict[str, Any]]:
    """``ib_write_lat`` result table; columns mapped by header names."""
    header, rows = _perftest_table(text, "t_min")
    if header is None or not rows:
        return None
    names = [LAT_KEYS[m] for m in _LAT_COLUMNS_RE.findall(header)]
    row = rows[-1]
    result: Dict[str, Any] = {key: None for key in LAT_KEYS.values()}
    for name, value in zip(names, row):
        number = float(value)
        result[name] = int(number) if name in ("msg_bytes", "iterations") else number
    return result


def parse_sock_extended_err(cmsg_data: bytes) -> Optional[Dict[str, Any]]:
    """``struct sock_extended_err`` + offender ``sockaddr_in`` (IP_RECVERR)."""
    if len(cmsg_data) < SOCK_EE_SIZE:
        return None
    errno_, origin, icmp_type, icmp_code, _, info, data = struct.unpack(
        SOCK_EE_FORMAT, cmsg_data[:SOCK_EE_SIZE]
    )
    offender: Optional[str] = None
    tail = cmsg_data[SOCK_EE_SIZE : SOCK_EE_SIZE + SOCKADDR_IN_SIZE]
    if len(tail) == SOCKADDR_IN_SIZE:
        family, _, addr = struct.unpack(SOCKADDR_IN_FORMAT, tail)
        if family == socket.AF_INET:
            offender = socket.inet_ntoa(addr)
    return {
        "errno": errno_,
        "origin": origin,
        "type": icmp_type,
        "code": icmp_code,
        "info": info,
        "data": data,
        "offender": offender,
    }


# --------------------------------------------------------------------------
# Readers: sysfs (tier S)
# --------------------------------------------------------------------------


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:  # missing, permission denied, EINVAL on empty GID attrs
        return None


def read_stripped(path: str) -> Optional[str]:
    text = read_text(path)
    if text is None:
        return None
    return text.strip() or None


def _listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def read_pcie(device_dir: str) -> Optional[Dict[str, Any]]:
    """PCIe link of the HCA function (current vs maximum speed and width)."""
    values = {
        "current_speed_gts": parse_pci_speed(
            read_stripped(os.path.join(device_dir, "current_link_speed"))
        ),
        "current_width": parse_pci_width(
            read_stripped(os.path.join(device_dir, "current_link_width"))
        ),
        "max_speed_gts": parse_pci_speed(
            read_stripped(os.path.join(device_dir, "max_link_speed"))
        ),
        "max_width": parse_pci_width(
            read_stripped(os.path.join(device_dir, "max_link_width"))
        ),
    }
    return values if any(v is not None for v in values.values()) else None


def read_gids(port_dir: str) -> List[Dict[str, Any]]:
    gids = []
    names = [n for n in _listdir(os.path.join(port_dir, "gids")) if n.isdigit()]
    for name in sorted(names, key=int):
        gid = normalize_gid(read_stripped(os.path.join(port_dir, "gids", name)))
        if gid is None:
            continue
        gids.append(
            {
                "index": int(name),
                "gid": gid,
                "type": read_stripped(os.path.join(port_dir, "gid_attrs/types", name)),
                "ndev": read_stripped(os.path.join(port_dir, "gid_attrs/ndevs", name)),
            }
        )
    return gids


def pick_roce_v2(
    gids: Sequence[Dict[str, Any]], netdev: Optional[str]
) -> Tuple[Optional[int], Optional[str]]:
    """Index and IPv4 of the RoCE v2 IPv4-mapped GID (the netdev's first)."""
    candidates = []
    for entry in gids:
        ipv4 = gid_ipv4(entry["gid"])
        if ipv4 is None or (entry["type"] or "").lower() != "roce v2":
            continue
        candidates.append((entry["ndev"] != netdev, entry["index"], ipv4))
    if not candidates:
        return None, None
    _, index, ipv4 = min(candidates)
    return index, ipv4


def read_netdev(host: Host, netdev: Optional[str]) -> Dict[str, Any]:
    if not netdev:
        return {"mtu": None, "operstate": None, "mac": None}
    base = host.path("sys/class/net", netdev)
    return {
        "mtu": parse_int(read_stripped(os.path.join(base, "mtu"))),
        "operstate": read_stripped(os.path.join(base, "operstate")),
        "mac": read_stripped(os.path.join(base, "address")),
    }


def read_ports(host: Host) -> List[Dict[str, Any]]:
    """Every port of every PCI RDMA device (soft devices such as rxe skipped)."""
    ib_root = host.path("sys/class/infiniband")
    ports: List[Dict[str, Any]] = []
    for name in sorted(_listdir(ib_root), key=natural_key):
        dev_dir = os.path.join(ib_root, name)
        device = os.path.realpath(os.path.join(dev_dir, "device"))
        bdf = os.path.basename(device)
        if not _SYSFS_BDF_RE.match(bdf):
            continue
        netdevs = sorted(_listdir(os.path.join(device, "net")), key=natural_key)
        hca = {
            "node_guid": normalize_guid(
                read_stripped(os.path.join(dev_dir, "node_guid"))
            ),
            "sys_image_guid": normalize_guid(
                read_stripped(os.path.join(dev_dir, "sys_image_guid"))
            ),
            "node_desc": read_stripped(os.path.join(dev_dir, "node_desc")),
            "fw_ver": read_stripped(os.path.join(dev_dir, "fw_ver")),
            "board_id": read_stripped(os.path.join(dev_dir, "board_id")),
        }
        pcie = read_pcie(device)
        ports_dir = os.path.join(dev_dir, "ports")
        for port in sorted(_listdir(ports_dir), key=natural_key):
            if port.isdigit():
                ports.append(
                    _read_port(
                        host, name, int(port), ports_dir, bdf, netdevs, hca, pcie
                    )
                )
    return ports


def _read_port(
    host: Host,
    hca: str,
    port: int,
    ports_dir: str,
    bdf: str,
    netdevs: Sequence[str],
    hca_info: Dict[str, Any],
    pcie: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    port_dir = os.path.join(ports_dir, str(port))
    rate = read_stripped(os.path.join(port_dir, "rate")) or ""
    rate_gbps, width, speed = parse_rate(rate)
    gids = read_gids(port_dir)
    gid0 = next((g for g in gids if g["index"] == 0), None)
    link_layer = read_stripped(os.path.join(port_dir, "link_layer")) or ""
    netdev = None
    if link_layer == "Ethernet" and gid0 and gid0["ndev"]:
        netdev = gid0["ndev"]
    elif netdevs:
        netdev = netdevs[0]
    gid_index, gid_ip = pick_roce_v2(gids, netdev)
    record = {
        "hca": hca,
        "port": port,
        "pci_bus_id": bdf,
        "link_layer": link_layer,
        "state": read_stripped(os.path.join(port_dir, "state")) or "",
        "phys_state": read_stripped(os.path.join(port_dir, "phys_state")) or "",
        "rate": rate,
        "rate_gbps": rate_gbps,
        "width": width,
        "speed": speed,
        "lid": parse_lid(read_stripped(os.path.join(port_dir, "lid"))),
        "sm_lid": parse_lid(read_stripped(os.path.join(port_dir, "sm_lid"))),
        "sm_sl": parse_int(read_stripped(os.path.join(port_dir, "sm_sl"))),
        "lmc": parse_int(read_stripped(os.path.join(port_dir, "lid_mask_count"))),
        "cap_mask": read_stripped(os.path.join(port_dir, "cap_mask")),
        "port_guid": gid_interface_id(gid0["gid"]) if gid0 else None,
        "subnet_prefix": gid_prefix(gid0["gid"]) if gid0 else None,
        "gids": gids,
        "roce_v2_gid_index": gid_index,
        "roce_v2_ipv4": gid_ip,
        "netdev": netdev,
        "ipv4": [],
        "active_mtu": None,
        "max_mtu": None,
        "pcie": pcie,
        "neighbor": None,
    }
    record.update(hca_info)
    record.update(read_netdev(host, netdev))
    return record


def port_key(port: Dict[str, Any]) -> str:
    return f"{port['hca']}:{port['port']}"


def is_active(port: Dict[str, Any]) -> bool:
    return str(port.get("state", "")).strip().startswith("4")


def fill_ipv4(
    host: Host, ports: Sequence[Dict[str, Any]], tools: Dict[str, bool]
) -> None:
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    if tools.get("ip"):
        result = host.run(["ip", "-j", "-4", "addr", "show"], timeout=TOOL_TIMEOUT_S)
        if result is not None and result.ok:
            by_name = parse_ip_addr_json(result.stdout)
    for port in ports:
        netdev = port["netdev"]
        if not netdev:
            continue
        if netdev in by_name:
            port["ipv4"] = by_name[netdev]
        elif not by_name:
            port["ipv4"] = host.ipv4_lookup(netdev)


def fill_devinfo(
    host: Host, ports: Sequence[Dict[str, Any]], tools: Dict[str, bool]
) -> None:
    if not tools.get("ibv_devinfo"):
        return
    for port in ports:
        result = host.run(
            ["ibv_devinfo", "-d", port["hca"], "-i", str(port["port"])],
            timeout=TOOL_TIMEOUT_S,
        )
        if result is not None and result.ok:
            port.update(parse_ibv_devinfo(result.stdout))


# --------------------------------------------------------------------------
# MAD queries (tier M)
# --------------------------------------------------------------------------


class SmpBudget:
    """Run-wide SMP budget (thread-safe); ``limit`` caps every SMP sent."""

    def __init__(self, limit: int, parent: Optional[SmpBudget] = None) -> None:
        self.limit = max(0, limit)
        self.parent = parent
        self.used = 0
        self.failed = 0
        self.blocked = False
        self.exhausted = False
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        """SMPs still allowed here and in every enclosing budget."""
        own = max(0, self.limit - self.used)
        if self.parent is not None:
            own = min(own, self.parent.remaining)
        return 0 if self.blocked else own

    def mark_exhausted(self) -> None:
        self.exhausted = True
        if self.parent is not None:
            self.parent.mark_exhausted()

    def take(self, count: int = 1) -> bool:
        with self._lock:
            if self.blocked:
                return False
            if self.used + count > self.limit:
                self.exhausted = True
                return False
            if self.parent is not None and not self.parent.take(count):
                self.exhausted = True
                return False
            self.used += count
            return True

    def fail(self) -> None:
        with self._lock:
            self.failed += 1
        if self.parent is not None:
            self.parent.fail()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.limit,
            "used": self.used,
            "failed": self.failed,
            "blocked": self.blocked,
            "exhausted": self.exhausted,
        }


class Mad:
    """SMP queries by directed route through one local HCA port.

    PortInfo is queried with ``-x`` (SMSupportsExtendedSpeeds, attribute
    modifier bit 31) so switch ports report FDR..XDR in LinkSpeedExt*; an
    smpquery that rejects ``-x`` is retried once without it.
    """

    def __init__(
        self, host: Host, budget: SmpBudget, timeout_ms: int = SMP_TIMEOUT_MS
    ) -> None:
        self.host = host
        self.budget = budget
        self.timeout_ms = timeout_ms
        self.extended_speeds = True

    def _smpquery(
        self,
        budget: SmpBudget,
        hca: str,
        port: int,
        op: str,
        path: str,
        params: Sequence[str] = (),
        cost: int = 1,
        options: Sequence[str] = (),
    ) -> Tuple[Optional[str], Optional[ToolResult]]:
        """(stdout, result); stdout is None when the query failed."""
        if not budget.take(cost):
            return None, None
        argv = [
            "smpquery", "-C", hca, "-P", str(port), "-t", str(self.timeout_ms),
            *options, "-D", op, path, *params,
        ]  # fmt: skip
        result = self.host.run(argv, timeout=SMP_PROC_TIMEOUT_S)
        if result is None or not result.ok:
            budget.fail()
            return None, result
        return result.stdout, result

    def nodeinfo(
        self, hca: str, port: int, path: str, budget: Optional[SmpBudget] = None
    ) -> Optional[Dict[str, Any]]:
        text, _ = self._smpquery(budget or self.budget, hca, port, "nodeinfo", path)
        return parse_nodeinfo(text) if text is not None else None

    def nodedesc(
        self, hca: str, port: int, path: str, budget: Optional[SmpBudget] = None
    ) -> Optional[str]:
        # smpquery sends NodeInfo then NodeDescription: two SMPs.
        text, _ = self._smpquery(
            budget or self.budget, hca, port, "nodedesc", path, cost=2
        )
        return parse_nodedesc(text) if text is not None else None

    def portinfo(
        self,
        hca: str,
        port: int,
        path: str,
        portnum: int,
        budget: Optional[SmpBudget] = None,
    ) -> Optional[Dict[str, Any]]:
        budget = budget or self.budget
        params = (str(portnum),)
        options = ("-x",) if self.extended_speeds else ()
        text, result = self._smpquery(
            budget, hca, port, "portinfo", path, params, options=options
        )
        if text is None and options and _rejects_option(result):
            self.extended_speeds = False
            text, _ = self._smpquery(budget, hca, port, "portinfo", path, params)
        return parse_portinfo(text, portnum) if text is not None else None

    def sminfo(self, hca: str, port: int) -> Optional[Dict[str, Any]]:
        if not self.budget.take(1):
            return None
        argv = ["sminfo", "-C", hca, "-P", str(port), "-t", str(self.timeout_ms)]
        result = self.host.run(argv, timeout=SMP_PROC_TIMEOUT_S)
        if result is None or not result.ok:
            self.budget.fail()
            return None
        return parse_sminfo(result.stdout)


def _rejects_option(result: Optional[ToolResult]) -> bool:
    if result is None:
        return False
    text = (result.stderr + result.stdout).lower()
    return "invalid option" in text or "unrecognized option" in text


def identify_neighbor(mad: Mad, port: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The node behind DR path "0,<port>" plus, for a switch, its LID and the
    PortInfo of the switch port our cable plugs into."""
    hca, num = port["hca"], port["port"]
    path = f"0,{num}"
    info = mad.nodeinfo(hca, num, path)
    if info is None:
        return None
    neighbor = dict(info)
    neighbor["desc"] = mad.nodedesc(hca, num, path)
    neighbor["lid"] = None
    neighbor["link"] = None
    if info["node_type"] == "Switch":
        port0 = mad.portinfo(hca, num, path, 0)
        neighbor["lid"] = port0["lid"] if port0 else None
        if info["local_port"]:
            neighbor["link"] = mad.portinfo(hca, num, path, info["local_port"])
    return neighbor


def _pool_map(
    fn: Callable[[Any], Any], items: Sequence[Any], workers: int
) -> List[Any]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        return list(pool.map(fn, items))


def collect_mad(
    host: Host,
    ports: Sequence[Dict[str, Any]],
    budget: SmpBudget,
    tools: Dict[str, bool],
    notes: List[Dict[str, Any]],
    *,
    census: bool,
    census_only: Optional[Sequence[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Neighbors (filled into ``ports``), SMs and switch censuses.

    The first port's neighbor query runs alone: when that SMP fails the
    fabric is treated as blocking SMPs and nothing else is sent.
    """
    if budget.limit <= 0:
        notes.append(note("smpquery", "smp budget is 0"))
        return [], {}
    mad = Mad(host, budget)
    first, rest = ports[0], list(ports[1:])
    first["neighbor"] = identify_neighbor(mad, first)
    if first["neighbor"] is None:
        budget.blocked = True
        notes.append(note("smpquery", "smp_blocked", port_key(first)))
        return [], {}
    neighbors = _pool_map(lambda p: identify_neighbor(mad, p), rest, MAD_THREADS)
    for port, neighbor in zip(rest, neighbors):
        port["neighbor"] = neighbor
        if neighbor is None:
            notes.append(note("smpquery", "neighbor query failed", port_key(port)))
    sms = collect_sms(mad, ports, tools, notes)
    switches: Dict[str, Dict[str, Any]] = {}
    for port in ports:
        neighbor = port["neighbor"]
        if neighbor and neighbor["node_type"] == "Switch" and neighbor["node_guid"]:
            switches.setdefault(
                neighbor["node_guid"],
                {
                    "guid": neighbor["node_guid"],
                    "desc": neighbor["desc"],
                    "lid": neighbor["lid"],
                    "num_ports": neighbor["num_ports"],
                    "via": port_key(port),
                    "census": "none",
                    "ports": [],
                },
            )
    if census:
        census_switches(mad, ports, switches, census_only, notes)
    return sms, switches


def collect_sms(
    mad: Mad,
    ports: Sequence[Dict[str, Any]],
    tools: Dict[str, bool],
    notes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """``sminfo`` through every port, deduplicated by (SM GUID, SM LID)."""
    if not tools.get("sminfo"):
        notes.append(note("sminfo", "not installed"))
        return []
    infos = _pool_map(
        lambda p: mad.sminfo(p["hca"], p["port"]), list(ports), MAD_THREADS
    )
    by_guid: Dict[Any, Dict[str, Any]] = {}
    for port, info in zip(ports, infos):
        if info is None:
            notes.append(note("sminfo", "query failed", port_key(port)))
            continue
        entry = by_guid.setdefault((info["guid"], info["lid"]), dict(info, ports=[]))
        entry["ports"].append(port_key(port))
    return list(by_guid.values())


def census_switches(
    mad: Mad,
    ports: Sequence[Dict[str, Any]],
    switches: Dict[str, Dict[str, Any]],
    census_only: Optional[Sequence[str]],
    notes: List[Dict[str, Any]],
) -> None:
    """Bounded census of each distinct neighbor switch (budget split evenly)."""
    wanted = set(census_only) if census_only is not None else None
    targets: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen = set()
    for port in ports:
        neighbor = port["neighbor"]
        if not neighbor or neighbor["node_type"] != "Switch":
            continue
        guid = neighbor["node_guid"]
        if guid in seen or (wanted is not None and port_key(port) not in wanted):
            continue
        seen.add(guid)
        targets.append((port, neighbor))
    for index, (port, neighbor) in enumerate(targets):
        share = mad.budget.remaining // (len(targets) - index)
        record = switches[neighbor["node_guid"]]
        record["via"] = port_key(port)
        census_one(mad, port, neighbor, record, SmpBudget(share, mad.budget))
        if record["census"] != "complete":
            notes.append(
                note(
                    "census", f"{record['census']} (smp budget)", neighbor["node_guid"]
                )
            )


def _self_peer(port: Dict[str, Any]) -> Dict[str, Any]:
    """Peer record of our own HCA port (known from sysfs, no SMP needed)."""
    return {
        "node_type": "Channel Adapter",
        "node_guid": port["node_guid"],
        "port_guid": port["port_guid"],
        "system_guid": port["sys_image_guid"],
        "num_ports": None,
        "local_port": port["port"],
        "desc": port["node_desc"],
        "dev_id": None,
        "vendor_id": None,
    }


def _affordable(budget: SmpBudget, items: Sequence[Any], cost: int = 1) -> List[Any]:
    """The prefix of ``items`` the budget can pay for (marks exhaustion)."""
    count = budget.remaining // cost
    if count < len(items):
        budget.mark_exhausted()
    return list(items[:count])


def census_one(
    mad: Mad,
    port: Dict[str, Any],
    neighbor: Dict[str, Any],
    record: Dict[str, Any],
    budget: SmpBudget,
) -> None:
    """PortInfo of every switch port, NodeInfo behind active ports,
    NodeDescription of switch peers, in that priority order."""
    hca, num = port["hca"], port["port"]
    path = f"0,{num}"
    total = neighbor["num_ports"] or 0
    chosen = _affordable(budget, list(range(1, total + 1)))
    links = _pool_map(
        lambda p: mad.portinfo(hca, num, path, p, budget), chosen, MAD_THREADS
    )
    by_port = {p: link for p, link in zip(chosen, links) if link is not None}
    ours = neighbor["local_port"]
    active = [p for p, link in sorted(by_port.items()) if link["state"] == "Active"]
    peers: Dict[int, Optional[Dict[str, Any]]] = {}
    if ours in by_port:
        peers[ours] = _self_peer(port)
    others = _affordable(budget, [p for p in active if p != ours])
    infos = _pool_map(
        lambda p: mad.nodeinfo(hca, num, f"{path},{p}", budget), others, MAD_THREADS
    )
    peers.update(zip(others, infos))
    switch_peers: Dict[str, int] = {}
    for p, peer in sorted(peers.items()):
        if peer and peer["node_type"] == "Switch" and peer["node_guid"]:
            switch_peers.setdefault(peer["node_guid"], p)
    named = _affordable(budget, list(switch_peers.items()), cost=2)
    descs = _pool_map(
        lambda gp: mad.nodedesc(hca, num, f"{path},{gp[1]}", budget),
        named,
        MAD_THREADS,
    )
    desc_by_guid = {g: d for (g, _), d in zip(named, descs)}
    entries = []
    for p, link in sorted(by_port.items()):
        peer = peers.get(p)
        if peer is not None and peer["node_type"] == "Switch":
            peer = dict(peer, desc=desc_by_guid.get(peer["node_guid"]))
        entries.append(dict(link, peer=peer))
    record["ports"] = entries
    complete = (
        len(by_port) == total
        and all(peers.get(p) is not None for p in active)
        and all(desc_by_guid.get(g) is not None for g in switch_peers)
    )
    record["census"] = "complete" if complete else ("partial" if entries else "none")


# --------------------------------------------------------------------------
# LLDP / QoS (tier L)
# --------------------------------------------------------------------------


def collect_lldp(
    host: Host,
    netdevs: Sequence[str],
    wait_s: float,
    tools: Dict[str, bool],
    notes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not netdevs:
        return result
    if tools.get("lldpctl"):
        run = host.run(["lldpctl", "-f", "json"], timeout=TOOL_TIMEOUT_S)
        if run is not None and run.ok:
            for netdev, record in parse_lldpctl_json(run.stdout).items():
                if netdev in netdevs:
                    result[netdev] = record
        else:
            notes.append(note("lldpctl", _tail(run)))
    pending = [nd for nd in netdevs if nd not in result]
    if not pending:
        return result
    if wait_s <= 0:
        notes.append(note("lldp", "listening disabled (--lldp-wait 0)"))
        return result
    frames, errors = host.lldp_listener(pending, wait_s)
    for netdev in pending:
        frame = frames.get(netdev)
        parsed = parse_lldp_frame(frame) if frame is not None else None
        if parsed is not None:
            result[netdev] = parsed
        elif netdev in errors:
            notes.append(note("lldp", errors[netdev], netdev))
        else:
            notes.append(note("lldp", f"no LLDP frame within {wait_s:g}s", netdev))
    return result


def collect_qos(
    host: Host,
    netdevs: Sequence[str],
    tools: Dict[str, bool],
    notes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not tools.get("mlnx_qos"):
        return result
    for netdev in netdevs:
        run = host.run(["mlnx_qos", "-i", netdev], timeout=TOOL_TIMEOUT_S)
        if run is not None and run.ok:
            result[netdev] = parse_mlnx_qos(run.stdout)
        else:
            notes.append(note("mlnx_qos", _tail(run), netdev))
    return result


# --------------------------------------------------------------------------
# identify
# --------------------------------------------------------------------------


def has_umad(host: Host) -> bool:
    return any(n.startswith("umad") for n in _listdir(host.path("dev/infiniband")))


def collect_identify(
    host: Host,
    *,
    lldp_wait_s: float = DEFAULT_LLDP_WAIT_S,
    max_smp: int = DEFAULT_MAX_SMP,
    census: bool = True,
    census_only: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The ``identify`` document (see the module docstring)."""
    start = host.monotonic()
    notes: List[Dict[str, Any]] = []
    tools = {name: host.which(name) is not None for name in TOOLS}
    ports = read_ports(host)
    if not ports:
        notes.append(note("sysfs", "no PCI RDMA devices under /sys/class/infiniband"))
    fill_ipv4(host, ports, tools)
    fill_devinfo(host, ports, tools)
    umad = has_umad(host)
    budget = SmpBudget(max_smp)
    sms: List[Dict[str, Any]] = []
    switches: Dict[str, Dict[str, Any]] = {}
    ib_ports = [p for p in ports if p["link_layer"] == "InfiniBand" and is_active(p)]
    if ib_ports:
        if not umad:
            notes.append(note("umad", "no /dev/infiniband/umad* device"))
        elif not tools["smpquery"]:
            notes.append(note("smpquery", "not installed"))
        else:
            sms, switches = collect_mad(
                host,
                ib_ports,
                budget,
                tools,
                notes,
                census=census,
                census_only=census_only,
            )
            if budget.exhausted:
                notes.append(note("smpquery", f"smp budget of {max_smp} exhausted"))
    eth = [p["netdev"] for p in ports if p["link_layer"] == "Ethernet" and p["netdev"]]
    eth = sorted(set(eth), key=natural_key)
    return {
        "kind": "fabric_identify",
        "version": VERSION,
        "hostname": host.hostname(),
        "tools": tools,
        "umad": umad,
        "smp": budget.as_dict(),
        "ports": ports,
        "sm": sms,
        "switches": switches,
        "lldp": collect_lldp(host, eth, lldp_wait_s, tools, notes),
        "qos": collect_qos(host, eth, tools, notes),
        "notes": notes,
        "elapsed_s": round(host.monotonic() - start, 3),
    }


# --------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------


def _hop(
    ttl: int, ip: Optional[str], rtt_ms: Optional[float], ee: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "ttl": ttl,
        "node_type": None,
        "guid": None,
        "desc": None,
        "lid": None,
        "in_port": None,
        "out_port": None,
        "ip": ip,
        "rtt_ms": rtt_ms,
        "icmp_type": ee["type"] if ee else None,
        "icmp_code": ee["code"] if ee else None,
    }


def udp_trace(
    host: Host,
    src_ip: Optional[str],
    dst_ip: str,
    netdev: Optional[str],
    *,
    max_ttl: int = UDP_MAX_TTL,
    hop_wait_s: float = UDP_HOP_WAIT_S,
) -> Dict[str, Any]:
    """Unprivileged UDP traceroute (IP_RECVERR), one probe per TTL."""
    result: Dict[str, Any] = {
        "method": "udp",
        "hops": [],
        "hop_count": None,
        "reached": False,
        "error": None,
    }
    try:
        sock = host.udp_probe(src_ip, netdev)
    except OSError as exc:
        result["error"] = f"socket: {exc}"
        return result
    try:
        for ttl in range(1, max_ttl + 1):
            payload = TRACE_MAGIC + bytes([ttl])
            sent = host.monotonic()
            try:
                sock.send(dst_ip, UDP_PORT_BASE + ttl, ttl, payload)
            except OSError as exc:
                result["error"] = f"send ttl {ttl}: {exc}"
                break
            ee, rtt_ms = _wait_for_ttl(host, sock, ttl, sent, hop_wait_s)
            if ee is None:
                result["hops"].append(_hop(ttl, None, None, None))
                continue
            if ee["origin"] != SO_EE_ORIGIN_ICMP:
                result["error"] = f"local error errno {ee['errno']} at ttl {ttl}"
                break
            # Port unreachable comes from the destination host itself (its
            # source address may be another of its interfaces).
            if ee["type"] == ICMP_DEST_UNREACH and (
                ee["code"] == ICMP_PORT_UNREACH or ee["offender"] == dst_ip
            ):
                result["reached"] = True
                result["hop_count"] = ttl - 1
                break
            result["hops"].append(_hop(ttl, ee["offender"], rtt_ms, ee))
            if ee["type"] == ICMP_DEST_UNREACH:
                result[
                    "error"
                ] = f"unreachable from {ee['offender']} (code {ee['code']})"
                break
    finally:
        sock.close()
    if not result["reached"] and result["error"] is None:
        result["error"] = f"destination silent up to ttl {max_ttl}"
    return result


def _probe_matches(entry: ErrQueueEntry, ttl: int) -> bool:
    """Is a queued error the answer to the probe sent with ``ttl``?

    The kernel returns the probe's original destination (whose port encodes
    the TTL); routers that quote only 8 bytes leave no payload to check.
    """
    data, _, address = entry
    if isinstance(address, tuple) and len(address) >= 2:
        return address[1] == UDP_PORT_BASE + ttl
    return data[: len(TRACE_MAGIC) + 1] == TRACE_MAGIC + bytes([ttl])


def _wait_for_ttl(
    host: Host, sock: Any, ttl: int, sent: float, hop_wait_s: float
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    deadline = sent + hop_wait_s
    while True:
        remaining = deadline - host.monotonic()
        if remaining <= 0:
            return None, None
        entry = sock.wait_error(remaining)
        if entry is None or not _probe_matches(entry, ttl):
            continue  # timeout slice, or a late answer to an earlier TTL
        for level, kind, cmsg in entry[1]:
            if level == SOL_IP and kind == IP_RECVERR:
                ee = parse_sock_extended_err(cmsg)
                if ee is not None:
                    rtt = (host.monotonic() - sent) * 1000.0
                    return ee, round(rtt, 3)
        return None, None


def ib_trace(host: Host, target: Dict[str, Any]) -> Dict[str, Any]:
    argv = [
        "ibtracert", "-C", str(target["hca"]), "-P", str(target["port"]),
        "-t", str(SMP_TIMEOUT_MS), str(target["src_lid"]), str(target["dst_lid"]),
    ]  # fmt: skip
    run = host.run(argv, timeout=TOOL_TIMEOUT_S)
    if run is None or not run.ok:
        return {
            "method": "ibtracert",
            "hops": [],
            "hop_count": None,
            "reached": False,
            "error": _tail(run),
        }
    parsed = parse_ibtracert(run.stdout)
    return {
        "method": "ibtracert",
        "hops": parsed["hops"],
        "hop_count": len(parsed["hops"]) if parsed["reached"] else None,
        "reached": parsed["reached"],
        "error": None if parsed["reached"] else "destination not reached",
    }


def trace_one(
    host: Host, target: Dict[str, Any], tools: Dict[str, bool]
) -> Dict[str, Any]:
    start = host.monotonic()
    if target.get("src_lid") and target.get("dst_lid") and tools.get("ibtracert"):
        result = ib_trace(host, target)
    elif target.get("dst_ip"):
        result = udp_trace(
            host, target.get("src_ip"), str(target["dst_ip"]), target.get("netdev")
        )
    else:
        reason = "no LIDs with ibtracert and no IPs"
        if target.get("src_lid") and not tools.get("ibtracert"):
            reason = "ibtracert not installed and no IPs"
        result = {
            "method": "none",
            "hops": [],
            "hop_count": None,
            "reached": False,
            "error": reason,
        }
    result["target"] = target
    result["elapsed_s"] = round(host.monotonic() - start, 3)
    return result


def collect_trace(host: Host, targets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    start = host.monotonic()
    tools = {"ibtracert": host.which("ibtracert") is not None}
    notes: List[Dict[str, Any]] = []
    if any(t.get("src_lid") for t in targets) and not has_umad(host):
        notes.append(note("umad", "no /dev/infiniband/umad* device"))
        tools["ibtracert"] = False
    results = _pool_map(
        lambda t: trace_one(host, t, tools), list(targets), TRACE_THREADS
    )
    return {
        "kind": "fabric_trace",
        "version": VERSION,
        "hostname": host.hostname(),
        "results": results,
        "notes": notes,
        "elapsed_s": round(host.monotonic() - start, 3),
    }


# --------------------------------------------------------------------------
# perfquery
# --------------------------------------------------------------------------


def _perfquery_argv(spec: Dict[str, Any], extended: bool) -> List[str]:
    argv = ["perfquery"]
    if extended:
        argv.append("-x")
    argv += [
        "-C", str(spec["hca"]), "-P", str(spec["port"]), "-t", str(SMP_TIMEOUT_MS),
        str(spec["lid"]), str(spec["switch_port"]),
    ]  # fmt: skip
    return argv


def _reading(
    spec: Dict[str, Any],
    counters: Dict[str, int],
    extended: bool,
    times: Tuple[float, float, float, float],
    error: Optional[str],
) -> Dict[str, Any]:
    t0m, t0w, t1m, t1w = times
    if extended:
        wide = sorted(counters)
    else:
        wide = sorted(k for k in counters if k not in BASIC_COUNTER_BITS)
    return {
        "key": spec.get("key"),
        "hca": spec["hca"],
        "port": spec["port"],
        "lid": spec["lid"],
        "switch_port": spec["switch_port"],
        "extended": extended,
        "wide": wide,
        "counters": counters,
        "mono_s": (t0m + t1m) / 2.0,
        "wall_s": (t0w + t1w) / 2.0,
        "error": error,
    }


def query_port_counters(
    host: Host, spec: Dict[str, Any], try_extended: bool = True
) -> Dict[str, Any]:
    """One switch port: ``perfquery -x`` first (unless the switch already
    refused it), then the 32-bit PortCounters."""
    errors: List[str] = []
    counters: Dict[str, int] = {}
    extended = False
    t0m, t0w = host.monotonic(), host.wall()
    if try_extended:
        run = host.run(_perfquery_argv(spec, True), timeout=SMP_PROC_TIMEOUT_S)
        counters = parse_perfquery(run.stdout) if run is not None and run.ok else {}
        extended = "PortXmitData" in counters
        if not extended:
            errors.append(f"-x: {_tail(run)}")
    if not extended:
        t0m, t0w = host.monotonic(), host.wall()
        run = host.run(_perfquery_argv(spec, False), timeout=SMP_PROC_TIMEOUT_S)
        counters = parse_perfquery(run.stdout) if run is not None and run.ok else {}
        if not counters:
            errors.append(f"basic: {_tail(run)}")
    times = (t0m, t0w, host.monotonic(), host.wall())
    error = "; ".join(errors) if not counters else None
    return _reading(spec, counters, extended, times, error)


def collect_perfquery(host: Host, specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counters of every requested switch port, fast and bounded.

    Ports are grouped per switch (local HCA port + LID). The first port of
    every switch is queried first, in parallel; it decides whether the rest
    of that switch uses ``-x``, and when it fails entirely (PMA unreachable
    or blocked) the switch's other ports are skipped instead of each waiting
    for its own timeouts.
    """
    start = host.monotonic()
    notes: List[Dict[str, Any]] = []
    readings: Dict[int, Dict[str, Any]] = {}
    if specs and host.which("perfquery") is None:
        notes.append(note("perfquery", "not installed"))
    elif specs and not has_umad(host):
        notes.append(note("umad", "no /dev/infiniband/umad* device"))
    elif specs:
        groups: Dict[Tuple[Any, Any, Any], List[int]] = {}
        for index, spec in enumerate(specs):
            group = (spec["hca"], spec["port"], spec["lid"])
            groups.setdefault(group, []).append(index)
        heads = [members[0] for members in groups.values()]
        for index, reading in zip(
            heads,
            _pool_map(
                lambda i: query_port_counters(host, specs[i]), heads, PERFQUERY_THREADS
            ),
        ):
            readings[index] = reading
        rest: List[Tuple[int, bool]] = []
        for group, members in groups.items():
            head = readings[members[0]]
            if head["error"] is not None:
                notes.append(note("perfquery", "pma_blocked", "lid %s" % group[2]))
                stamp = (host.monotonic(), host.wall()) * 2
                for index in members[1:]:
                    readings[index] = _reading(
                        specs[index], {}, False, stamp, "skipped: switch PMA failed"
                    )
                continue
            rest += [(index, head["extended"]) for index in members[1:]]
        for (index, _), reading in zip(
            rest,
            _pool_map(
                lambda item: query_port_counters(host, specs[item[0]], item[1]),
                rest,
                PERFQUERY_THREADS,
            ),
        ):
            readings[index] = reading
    return {
        "kind": "fabric_perfquery",
        "version": VERSION,
        "hostname": host.hostname(),
        "ports": [readings[i] for i in sorted(readings)],
        "notes": notes,
        "elapsed_s": round(host.monotonic() - start, 3),
    }


# --------------------------------------------------------------------------
# perftest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerftestConfig:
    role: str
    peer: Optional[str]
    tests: Tuple[str, ...]
    bw_seconds: int = DEFAULT_BW_SECONDS
    bw_size: int = DEFAULT_BW_SIZE
    bw_qps: int = DEFAULT_BW_QPS
    lat_iters: int = DEFAULT_LAT_ITERS
    lat_size: int = DEFAULT_LAT_SIZE
    timeout_s: float = DEFAULT_PERFTEST_TIMEOUT_S
    connect_wait_s: float = DEFAULT_CONNECT_WAIT_S


def perftest_argv(cfg: PerftestConfig, rail: Dict[str, Any], test: str) -> List[str]:
    """``ib_write_bw`` / ``ib_write_lat`` argv; the client appends the peer."""
    tool = "ib_write_bw" if test == "bw" else "ib_write_lat"
    tcp_port = rail["bw_tcp_port"] if test == "bw" else rail["lat_tcp_port"]
    argv = [tool, "-d", str(rail["hca"]), "-i", str(rail["port"]), "-p", str(tcp_port)]
    if test == "bw":
        argv += [
            "-D", str(cfg.bw_seconds), "-s", str(cfg.bw_size), "-q", str(cfg.bw_qps),
            "--report_gbits",
        ]  # fmt: skip
    else:
        argv += ["-n", str(cfg.lat_iters), "-s", str(cfg.lat_size)]
    argv.append("-F")
    if rail.get("gid_index") is not None:
        argv += ["-x", str(rail["gid_index"])]
    if cfg.role == "client" and cfg.peer:
        argv.append(cfg.peer)
    return argv


def _perftest_result(
    cfg: PerftestConfig,
    rail: Dict[str, Any],
    test: str,
    argv: List[str],
    run: Optional[ToolResult],
    attempts: int,
    elapsed: float,
) -> Dict[str, Any]:
    text = "" if run is None else run.stdout
    bw = parse_perftest_bw(text) if test == "bw" else None
    lat = parse_perftest_lat(text) if test == "lat" else None
    error = None
    if run is None:
        error = f"{argv[0]} not installed"
    elif (test == "bw" and bw is None) or (test == "lat" and lat is None):
        error = _tail(run)
    return {
        "rail": rail["rail"],
        "hca": rail["hca"],
        "port": rail["port"],
        "test": test,
        "rc": None if run is None else run.rc,
        "attempts": attempts,
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "argv": argv,
        "bw": bw,
        "lat": lat,
    }


def _serve(
    host: Host, cfg: PerftestConfig, rail: Dict[str, Any], test: str, deadline: float
) -> Dict[str, Any]:
    argv = perftest_argv(cfg, rail, test)
    start = host.monotonic()
    run = host.run(argv, timeout=max(1.0, deadline - start))
    return _perftest_result(cfg, rail, test, argv, run, 1, host.monotonic() - start)


def _connect(
    host: Host, cfg: PerftestConfig, rail: Dict[str, Any], test: str
) -> Dict[str, Any]:
    argv = perftest_argv(cfg, rail, test)
    start = host.monotonic()
    budget = cfg.bw_seconds + PERFTEST_SLACK_S if test == "bw" else PERFTEST_SLACK_S
    attempts = 0
    while True:
        attempts += 1
        run = host.run(argv, timeout=budget)
        if run is None or run.ok:
            break
        waited = host.monotonic() - start
        if (
            not _PERFTEST_CONNECT_RE.search(run.stdout + run.stderr)
            or waited >= cfg.connect_wait_s
        ):
            break
        host.sleep(CONNECT_RETRY_S)
    if run is not None and not run.ok and cfg.peer:
        port = rail["bw_tcp_port"] if test == "bw" else rail["lat_tcp_port"]
        host.poke(cfg.peer, int(port))
    return _perftest_result(
        cfg, rail, test, argv, run, attempts, host.monotonic() - start
    )


def collect_perftest(
    host: Host, cfg: PerftestConfig, rails: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    start = host.monotonic()
    notes: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for test in cfg.tests:
        tool = "ib_write_bw" if test == "bw" else "ib_write_lat"
        if host.which(tool) is None:
            notes.append(note(tool, "not installed"))
    tests = [
        t
        for t in cfg.tests
        if host.which("ib_write_bw" if t == "bw" else "ib_write_lat")
    ]
    if cfg.role == "server":
        deadline = start + cfg.timeout_s
        jobs = [(rail, test) for test in tests for rail in rails]
        results = _pool_map(
            lambda j: _serve(host, cfg, j[0], j[1], deadline), jobs, max(1, len(jobs))
        )
    else:
        for test in tests:
            results += _pool_map(
                lambda r, t=test: _connect(host, cfg, r, t),
                list(rails),
                max(1, len(rails)),
            )
    return {
        "kind": "fabric_perftest",
        "version": VERSION,
        "hostname": host.hostname(),
        "role": cfg.role,
        "peer": cfg.peer,
        "results": results,
        "notes": notes,
        "elapsed_s": round(host.monotonic() - start, 3),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _read_json_input(source: str) -> Any:
    if source == "-":
        return json.loads(sys.stdin.read() or "[]")
    with open(source, encoding="utf-8") as fh:
        return json.load(fh)


def _split_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="write the JSON here instead of stdout")
    sub = parser.add_subparsers(dest="command", required=True)
    ident = sub.add_parser("identify", help="ports, neighbors, SMs, switches, LLDP")
    ident.add_argument("--lldp-wait", type=float, default=DEFAULT_LLDP_WAIT_S)
    ident.add_argument("--max-smp", type=int, default=DEFAULT_MAX_SMP)
    ident.add_argument("--no-census", action="store_true")
    ident.add_argument(
        "--census-only",
        help="comma-separated HCA:PORT whose neighbor switch to census",
    )
    trace = sub.add_parser("trace", help="switch/router path per target")
    trace.add_argument("--targets", required=True, help="JSON file or - for stdin")
    perf = sub.add_parser("perfquery", help="switch port counters")
    perf.add_argument("--ports", required=True, help="JSON file or - for stdin")
    test = sub.add_parser("perftest", help="ib_write_bw / ib_write_lat per rail")
    test.add_argument("--role", choices=("server", "client"), required=True)
    test.add_argument("--rails", required=True, help="JSON file or - for stdin")
    test.add_argument("--peer", help="server control address (client role)")
    test.add_argument("--tests", default="bw,lat")
    test.add_argument("--bw-seconds", type=int, default=DEFAULT_BW_SECONDS)
    test.add_argument("--bw-size", type=int, default=DEFAULT_BW_SIZE)
    test.add_argument("--bw-qps", type=int, default=DEFAULT_BW_QPS)
    test.add_argument("--lat-iters", type=int, default=DEFAULT_LAT_ITERS)
    test.add_argument("--lat-size", type=int, default=DEFAULT_LAT_SIZE)
    test.add_argument("--timeout", type=float, default=DEFAULT_PERFTEST_TIMEOUT_S)
    test.add_argument("--connect-wait", type=float, default=DEFAULT_CONNECT_WAIT_S)
    return parser


INPUT_ARG = {"trace": "targets", "perfquery": "ports", "perftest": "rails"}


def dispatch(args: argparse.Namespace, host: Host) -> Dict[str, Any]:
    if args.command == "identify":
        return collect_identify(
            host,
            lldp_wait_s=args.lldp_wait,
            max_smp=args.max_smp,
            census=not args.no_census,
            census_only=_split_list(args.census_only),
        )
    input_note: Optional[Dict[str, Any]] = None
    try:
        items = _read_json_input(getattr(args, INPUT_ARG[args.command]))
        if not isinstance(items, list):
            raise ValueError("expected a JSON list")
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        items = []
        input_note = note("input", f"unreadable input: {exc}")
    doc = _run_command(args, host, items)
    if input_note is not None:
        doc["notes"].insert(0, input_note)
    return doc


def _run_command(
    args: argparse.Namespace, host: Host, items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if args.command == "trace":
        return collect_trace(host, items)
    if args.command == "perfquery":
        return collect_perfquery(host, items)
    cfg = PerftestConfig(
        role=args.role,
        peer=args.peer,
        tests=tuple(t for t in _split_list(args.tests) or [] if t in ("bw", "lat")),
        bw_seconds=args.bw_seconds,
        bw_size=args.bw_size,
        bw_qps=args.bw_qps,
        lat_iters=args.lat_iters,
        lat_size=args.lat_size,
        timeout_s=args.timeout,
        connect_wait_s=args.connect_wait,
    )
    return collect_perftest(host, cfg, items)


def emit(doc: Dict[str, Any], out: Optional[str]) -> None:
    text = json.dumps(doc, sort_keys=False) + "\n"
    if out:
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, out)
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def main(argv: Optional[Sequence[str]] = None, host: Optional[Host] = None) -> int:
    args = build_parser().parse_args(argv)
    emit(dispatch(args, host or Host()), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
