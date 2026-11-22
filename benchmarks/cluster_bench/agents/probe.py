# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe one cluster node and print its hardware inventory as JSON.

Run as ``<python> probe.py [--control-interface wg0|auto --control-address IP]``. Prints ONE JSON object
on stdout whose keys are the ``NodeInventory`` fields of
``benchmarks/cluster_bench/domain/schema.py`` except ``name`` and ``address``,
which the runner adds.

Standalone on purpose: stdlib only, Python >= 3.8 syntax and no
``benchmarks.*`` imports, so whatever interpreter a node has can run it.
``nvidia-smi`` is required (exit code 2 without a working one); every other
source is optional and becomes ``null`` / empty when missing. Parsing lives in
pure ``parse_*`` functions over text so tests can feed recorded output; the
thin readers below them only fetch that text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

EXIT_PROBE_FAILED = 2
TOOL_TIMEOUT_S = 20.0
OPTIONAL_TOOL_TIMEOUT_S = 10.0
GIB = 1 << 30
MIN_DISK_BYTES = 50 * GIB
# cgroup v1 reports "no limit" as ~2**63; anything this large is no limit.
CGROUP_UNLIMITED_BYTES = 1 << 60

GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "memory.total",
    "compute_cap",
    "pcie.link.gen.max",
    "pcie.link.width.max",
)
# Accepted by every driver; used when an old driver rejects an optional field.
GPU_QUERY_CORE_FIELDS = GPU_QUERY_FIELDS[:5]

KEEP_FS_TYPES = frozenset({"ext4", "xfs", "btrfs", "zfs", "overlay"})
EXCLUDED_FS_TYPES = frozenset(
    {"cifs", "9p", "tmpfs", "proc", "sysfs", "devtmpfs", "squashfs", "s3fs", "ceph"}
)
EXCLUDED_FS_PREFIXES = ("fuse", "nfs")

_NA_VALUES = frozenset(
    {
        "",
        "n/a",
        "[n/a]",
        "not supported",
        "[not supported]",
        "[unknown error]",
        "[insufficient permissions]",
    }
)
_BDF_RE = re.compile(
    r"^(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])$"
)
_SYSFS_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_RATE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*Gb/s", re.IGNORECASE)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Device labels in `nvidia-smi topo -m`: GPU0, NIC3, mlx5_0, mlx5_bond_0, ...
# The extra columns ("CPU Affinity", "NUMA Affinity", "GPU NUMA ID") never
# end in a digit, so they never match.
_TOPO_LABEL_RE = re.compile(r"^[A-Za-z]\w*\d$")
_NIC_LEGEND_RE = re.compile(r"^\s*(NIC\d+)\s*:\s*(\S+)\s*$")
_CUDA_RE = re.compile(r"CUDA Version\s*:\s*([0-9][0-9.]*)")
_DRIVER_RE = re.compile(r"Driver Version\s*:\s*([0-9][0-9.]*)")
_NVLINK_UP_RE = re.compile(r"Link\s+\d+\s*:\s*[0-9]+(?:\.[0-9]+)?\s*GB/s")
_MOUNT_ESCAPE_RE = re.compile(r"\\([0-7]{3})")


class ProbeError(RuntimeError):
    """A mandatory source (nvidia-smi) is missing or broken."""


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
    except subprocess.TimeoutExpired:
        return ToolResult(124, "", f"timed out after {timeout:.0f}s")
    return ToolResult(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class Host:
    """Everything the probe reads from the machine; tests substitute fakes.

    ``root`` prefixes every /sys and /proc path, ``run`` executes tools.
    """

    root: str = "/"
    run: Callable[..., Optional[ToolResult]] = run_tool
    statvfs: Callable[[str], Any] = os.statvfs
    isdir: Callable[[str], bool] = os.path.isdir

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *(p.lstrip("/") for p in parts))

    def display_path(self, real: str) -> str:
        """``real`` as it would read on the host (strips a test root)."""
        root = os.path.realpath(self.root)
        if root == os.sep:
            return real
        return os.sep + os.path.relpath(real, root)


# --------------------------------------------------------------------------
# Pure parsers (text in, data out)
# --------------------------------------------------------------------------


def natural_key(text: str) -> List[Any]:
    """Sort key ordering mlx5_2 before mlx5_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def normalize_bdf(raw: str) -> str:
    """nvidia-smi "00000000:1B:00.0" -> sysfs "0000:1b:00.0"."""
    match = _BDF_RE.match(raw.strip())
    if match is None:
        raise ValueError(f"not a PCI bus id: {raw!r}")
    domain, bus, device, function = match.groups()
    return "%04x:%02x:%02x.%s" % (
        int(domain or "0", 16),
        int(bus, 16),
        int(device, 16),
        function,
    )


def _na(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return None if text.lower() in _NA_VALUES else text


def _na_int(value: Optional[str]) -> Optional[int]:
    text = _na(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_gpu_csv(
    text: str, fields: Sequence[str] = GPU_QUERY_FIELDS
) -> List[Dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits``."""
    gpus: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        extra = len(parts) - len(fields)
        if extra < 0:
            raise ValueError(
                f"nvidia-smi line has {len(parts)} fields, want {len(fields)}: "
                f"{line!r}"
            )
        if extra:  # a comma inside the GPU name (field 1)
            parts = [parts[0], ", ".join(parts[1 : 2 + extra])] + parts[2 + extra :]
        row = dict(zip(fields, parts))
        gpus.append(
            {
                "index": int(row["index"]),
                "name": row["name"],
                "uuid": row["uuid"],
                "pci_bus_id": normalize_bdf(row["pci.bus_id"]),
                "memory_mib": int(float(row["memory.total"])),
                "numa_node": None,
                "compute_cap": _na(row.get("compute_cap")),
                "pcie_gen_max": _na_int(row.get("pcie.link.gen.max")),
                "pcie_width_max": _na_int(row.get("pcie.link.width.max")),
            }
        )
    return gpus


def parse_first_line(text: Optional[str]) -> Optional[str]:
    """First non-empty line, stripped (``None`` when there is none)."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return None


def parse_rate_gbps(rate: str) -> float:
    """Gigabits/s from a sysfs IB rate such as "400 Gb/sec (4X NDR)"."""
    match = _RATE_RE.search(rate or "")
    return float(match.group(1)) if match else 0.0


def parse_nic_legend(text: str) -> Dict[str, str]:
    """``NIC Legend`` section of ``nvidia-smi topo -m``: NIC0 -> mlx5_0."""
    legend: Dict[str, str] = {}
    for line in _ANSI_RE.sub("", text).splitlines():
        match = _NIC_LEGEND_RE.match(line)
        if match:
            legend[match.group(1)] = match.group(2)
    return legend


def _topo_header(lines: Sequence[str]) -> Tuple[int, List[str]]:
    """(line index, device column labels) of the matrix header, or (-1, [])."""
    fallback: Tuple[int, List[str]] = (-1, [])
    for index, line in enumerate(lines):
        tokens = line.split()
        labels: List[str] = []
        for token in tokens:
            if not _TOPO_LABEL_RE.match(token):
                break
            labels.append(token)
        if not labels:
            continue
        # The header row is indented (it has no row label); data rows are not.
        if line[:1].isspace():
            return index, labels
        if len(labels) >= 2 and fallback[0] < 0:
            fallback = (index, labels)
    return fallback


def parse_topo(text: str) -> Dict[str, Dict[str, str]]:
    """``nvidia-smi topo -m`` as row label -> column label -> link type.

    Tolerates ANSI escapes, tab or space alignment and the trailing affinity
    columns. NICk labels are renamed to their HCA (mlx5_*) through the "NIC
    Legend" section when present.
    """
    clean = _ANSI_RE.sub("", text)
    lines = clean.splitlines()
    header_index, columns = _topo_header(lines)
    if header_index < 0:
        return {}
    known = set(columns)
    matrix: Dict[str, Dict[str, str]] = {}
    for line in lines[header_index + 1 :]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].rstrip(":").lower() == "legend":
            break
        if tokens[0] not in known:
            continue
        cells = tokens[1 : 1 + len(columns)]
        matrix[tokens[0]] = dict(zip(columns, cells))
    legend = parse_nic_legend(clean)
    if not legend:
        return matrix
    return {
        legend.get(row, row): {legend.get(col, col): cell for col, cell in r.items()}
        for row, r in matrix.items()
    }


def parse_meminfo_total(text: Optional[str]) -> int:
    """MemTotal from /proc/meminfo in bytes (0 when absent)."""
    for line in (text or "").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return 0


def parse_cpu_model(text: Optional[str]) -> str:
    """CPU model from /proc/cpuinfo ("model name") or lscpu ("Model name")."""
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "model name" and value.strip():
            return value.strip()
    return ""


def parse_cgroup_limit(text: Optional[str]) -> Optional[int]:
    """cgroup memory limit in bytes; ``None`` for no limit or unreadable."""
    value = (text or "").strip()
    if not value.isdigit():
        return None  # "max" (cgroup v2 unlimited), empty or garbage
    limit = int(value)
    if limit <= 0 or limit >= CGROUP_UNLIMITED_BYTES:
        return None
    return limit


@dataclass(frozen=True)
class MountEntry:
    device: str
    mountpoint: str
    fs_type: str


def _unescape_mount(field: str) -> str:
    return _MOUNT_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 8)), field)


def parse_mounts(text: Optional[str]) -> List[MountEntry]:
    """/proc/mounts rows; octal escapes (``\\040`` = space) are decoded."""
    entries: List[MountEntry] = []
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        entries.append(
            MountEntry(
                device=_unescape_mount(parts[0]),
                mountpoint=_unescape_mount(parts[1]),
                fs_type=parts[2],
            )
        )
    return entries


def keep_fs_type(fs_type: str) -> bool:
    """Local block-device filesystems that can hold model weights."""
    if fs_type in EXCLUDED_FS_TYPES or fs_type.startswith(EXCLUDED_FS_PREFIXES):
        return False
    return fs_type in KEEP_FS_TYPES


def parse_cuda_version(text: Optional[str]) -> Optional[str]:
    """ "CUDA Version: 13.0" from the plain ``nvidia-smi`` banner."""
    match = _CUDA_RE.search(text or "")
    return match.group(1) if match else None


def parse_banner_driver(text: Optional[str]) -> Optional[str]:
    """ "Driver Version: 580.95.05" from the plain ``nvidia-smi`` banner."""
    match = _DRIVER_RE.search(text or "")
    return match.group(1) if match else None


def parse_ofed_version(text: Optional[str]) -> Optional[str]:
    """``ofed_info -s`` prints e.g. "MLNX_OFED_LINUX-24.10-1.1.4.0:"."""
    line = parse_first_line(text)
    if line is None:
        return None
    return line.rstrip(":").strip() or None


def parse_nvlink_active(text: Optional[str]) -> bool:
    """True when ``nvidia-smi nvlink -s`` reports any link with a GB/s rate."""
    return bool(_NVLINK_UP_RE.search(text or ""))


# --------------------------------------------------------------------------
# Readers (fetch text from the host, then parse)
# --------------------------------------------------------------------------


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:  # missing, permission denied, or an unreadable sysfs attr
        return None


def read_stripped(path: str) -> Optional[str]:
    text = read_text(path)
    if text is None:
        return None
    return text.strip() or None


def read_numa_node(device_dir: str) -> Optional[int]:
    text = read_stripped(os.path.join(device_dir, "numa_node"))
    if text is None:
        return None
    try:
        node = int(text)
    except ValueError:
        return None
    return node if node >= 0 else None


def _listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _smi(host: Host, *args: str) -> Optional[ToolResult]:
    return host.run(["nvidia-smi", *args])


def probe_gpus(host: Host) -> List[Dict[str, Any]]:
    """GPUs from nvidia-smi (mandatory) plus their NUMA node from sysfs."""
    query = "--query-gpu=" + ",".join(GPU_QUERY_FIELDS)
    result = _smi(host, query, "--format=csv,noheader,nounits")
    if result is None:
        raise ProbeError(
            "nvidia-smi not found on PATH: the NVIDIA driver and its userspace "
            "tools are required on every node"
        )
    fields: Sequence[str] = GPU_QUERY_FIELDS
    if not result.ok:
        core = "--query-gpu=" + ",".join(GPU_QUERY_CORE_FIELDS)
        retry = _smi(host, core, "--format=csv,noheader,nounits")
        if retry is None or not retry.ok:
            detail = (result.stderr.strip() or result.stdout.strip())[-500:]
            raise ProbeError(f"nvidia-smi failed (exit {result.rc}): {detail}")
        result, fields = retry, GPU_QUERY_CORE_FIELDS
    gpus = parse_gpu_csv(result.stdout, fields)
    if not gpus:
        raise ProbeError("nvidia-smi reported no GPUs")
    for gpu in gpus:
        gpu["numa_node"] = read_numa_node(
            host.path("sys/bus/pci/devices", gpu["pci_bus_id"])
        )
    return gpus


def probe_hcas(host: Host) -> List[Dict[str, Any]]:
    """RDMA HCAs from /sys/class/infiniband (empty when there are none)."""
    ib_root = host.path("sys/class/infiniband")
    hcas: List[Dict[str, Any]] = []
    for name in sorted(_listdir(ib_root), key=natural_key):
        dev_dir = os.path.join(ib_root, name)
        device = os.path.join(dev_dir, "device")
        bdf = os.path.basename(os.path.realpath(device))
        if not _SYSFS_BDF_RE.match(bdf):
            continue  # e.g. soft-RoCE (rxe) has no PCI function
        ports = []
        ports_dir = os.path.join(dev_dir, "ports")
        for port in sorted(_listdir(ports_dir), key=natural_key):
            if not port.isdigit():
                continue
            port_dir = os.path.join(ports_dir, port)
            rate = read_stripped(os.path.join(port_dir, "rate")) or ""
            ports.append(
                {
                    "port": int(port),
                    "state": read_stripped(os.path.join(port_dir, "state")) or "",
                    "rate": rate,
                    "rate_gbps": parse_rate_gbps(rate),
                    "link_layer": read_stripped(os.path.join(port_dir, "link_layer"))
                    or "",
                }
            )
        netdevs = sorted(_listdir(os.path.join(device, "net")), key=natural_key)
        hcas.append(
            {
                "name": name,
                "pci_bus_id": bdf,
                "numa_node": read_numa_node(device),
                "netdev": netdevs[0] if netdevs else None,
                "board_id": read_stripped(os.path.join(dev_dir, "board_id")),
                "fw_ver": read_stripped(os.path.join(dev_dir, "fw_ver")),
                "ports": ports,
            }
        )
    return hcas


def probe_pci_paths(host: Host, bdfs: Sequence[str]) -> Dict[str, str]:
    """BDF -> resolved sysfs device path (shared prefixes = shared switches)."""
    paths: Dict[str, str] = {}
    for bdf in bdfs:
        link = host.path("sys/bus/pci/devices", bdf)
        if os.path.exists(link):
            paths[bdf] = host.display_path(os.path.realpath(link))
    return paths


def probe_disks(host: Host) -> List[Dict[str, Any]]:
    """Local filesystems of at least MIN_DISK_BYTES, one per mount point."""
    text = read_text(host.path("proc/mounts"))
    by_mount: Dict[str, MountEntry] = {}
    for entry in parse_mounts(text):
        by_mount[entry.mountpoint] = entry  # later mounts shadow earlier ones
    disks: List[Dict[str, Any]] = []
    for mountpoint, entry in by_mount.items():
        # Bind-mounted files (/etc/hosts, /etc/resolv.conf) are not disks.
        if not keep_fs_type(entry.fs_type) or not host.isdir(mountpoint):
            continue
        try:
            st = host.statvfs(mountpoint)
        except OSError:
            continue
        total = st.f_blocks * st.f_frsize
        if total < MIN_DISK_BYTES:
            continue
        disks.append(
            {
                "path": mountpoint,
                "device": entry.device,
                "fs_type": entry.fs_type,
                "total_bytes": total,
                "free_bytes": st.f_bavail * st.f_frsize,
            }
        )
    return sorted(disks, key=lambda d: d["path"])


def _tool_text(host: Host, argv: Sequence[str], timeout: float) -> Optional[str]:
    result = host.run(list(argv), timeout=timeout)
    if result is None or not result.ok:
        return None
    return result.stdout


def parse_interface_for(ip_addr_text: str, address: str) -> Optional[str]:
    """The interface holding ``address`` in ``ip -o -4 addr show`` output."""
    for line in ip_addr_text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "inet" and parts[3].split("/")[0] == address:
            return parts[1].rstrip(":").split("@", 1)[0]
    return None


def resolve_control_interface(
    host: Host, control_interface: str, control_address: Optional[str]
) -> str:
    """``auto`` -> the NIC that holds this node's control address."""
    if control_interface != "auto":
        return control_interface
    if not control_address:
        raise ProbeError("--control-interface auto needs --control-address")
    text = _tool_text(host, ["ip", "-o", "-4", "addr", "show"], TOOL_TIMEOUT_S)
    iface = parse_interface_for(text or "", control_address)
    if iface is None:
        raise ProbeError(
            f"control address {control_address} is not assigned to any interface of "
            "this host (a NAT or public address?); give the node a private address "
            "it owns (machines file control_ip=, hosts file address)"
        )
    return iface


def collect(
    host: Host, control_interface: str = "wg0", control_address: Optional[str] = None
) -> Dict[str, Any]:
    """The probe payload (NodeInventory minus ``name`` and ``address``)."""
    control_interface = resolve_control_interface(
        host, control_interface, control_address
    )
    gpus = probe_gpus(host)
    hcas = probe_hcas(host)
    topo_text = _tool_text(host, ["nvidia-smi", "topo", "-m"], TOOL_TIMEOUT_S)
    driver_text = _tool_text(
        host,
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        TOOL_TIMEOUT_S,
    )
    banner = _tool_text(host, ["nvidia-smi"], TOOL_TIMEOUT_S)
    nvlink_text = _tool_text(host, ["nvidia-smi", "nvlink", "-s"], TOOL_TIMEOUT_S)
    cpu_model = parse_cpu_model(read_text(host.path("proc/cpuinfo")))
    if not cpu_model:  # aarch64 /proc/cpuinfo has no "model name"
        cpu_model = parse_cpu_model(
            _tool_text(host, ["lscpu"], OPTIONAL_TOOL_TIMEOUT_S)
        )
    cgroup = read_text(host.path("sys/fs/cgroup/memory.max"))
    if cgroup is None:
        cgroup = read_text(host.path("sys/fs/cgroup/memory/memory.limit_in_bytes"))
    docker = _tool_text(
        host,
        ["docker", "version", "--format", "{{.Server.Version}}"],
        OPTIONAL_TOOL_TIMEOUT_S,
    )
    ofed = _tool_text(host, ["ofed_info", "-s"], OPTIONAL_TOOL_TIMEOUT_S)
    bdfs = [g["pci_bus_id"] for g in gpus] + [h["pci_bus_id"] for h in hcas]
    return {
        "hostname": socket.gethostname(),
        "control_interface": control_interface,
        "gpus": gpus,
        "hcas": hcas,
        "pci_paths": probe_pci_paths(host, bdfs),
        "topo": parse_topo(topo_text) if topo_text else {},
        "driver_version": parse_first_line(driver_text)
        or parse_banner_driver(banner)
        or "unknown",
        "cuda_version": parse_cuda_version(banner),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count() or 0,
        "mem_total_bytes": parse_meminfo_total(read_text(host.path("proc/meminfo"))),
        "cgroup_mem_limit_bytes": parse_cgroup_limit(cgroup),
        "disks": probe_disks(host),
        "docker_version": parse_first_line(docker),
        "kernel": os.uname().release,
        "ofed_version": parse_ofed_version(ofed),
        "nvidia_peermem": os.path.exists(host.path("sys/module/nvidia_peermem")),
        "nvlink_active": parse_nvlink_active(nvlink_text),
    }


def main(argv: Optional[Sequence[str]] = None, host: Optional[Host] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--control-interface",
        default="wg0",
        help="control-plane network interface recorded in the inventory; "
        "'auto' = the interface that holds --control-address",
    )
    parser.add_argument(
        "--control-address",
        default=None,
        help="this node's control-plane IPv4 (for --control-interface auto)",
    )
    args = parser.parse_args(argv)
    try:
        payload = collect(host or Host(), args.control_interface, args.control_address)
    except ProbeError as exc:
        print(f"probe: {exc}", file=sys.stderr)
        return EXIT_PROBE_FAILED
    sys.stdout.write(json.dumps(payload, sort_keys=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
