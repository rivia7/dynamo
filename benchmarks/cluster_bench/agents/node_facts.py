# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node facts, health counters, software stack and disk read speed as JSON.

Run as::

    <python> node_facts.py [host] [--node NAME] [--out PATH] [--deadline-s 150]
    python3  node_facts.py container [--node NAME] [--out PATH]
    <python> node_facts.py disk-read --path P [--max-bytes N] [--block-mib 16]
        [--max-seconds S] [--node NAME] [--out PATH]

Complements ``probe.py`` (GPU/HCA inventory, ``topo -m``, versions, disks):
this agent records what explains performance differences between clusters -
degraded PCIe links, power caps, ECC / row-remap state, error and congestion
counters (snapshot "before" and "after" a run), versions, time sync and
storage speed. A few identity fields (kernel release, driver version, HCA
firmware) repeat the probe on purpose so a facts document stands alone.

Standalone: stdlib only, Python >= 3.8 syntax, no ``benchmarks.*`` imports.
Every source is optional: a missing tool, file or permission leaves the field
null / empty and appends ``{"source": ..., "error": ...}`` to ``notes``. One
JSON document goes to stdout (or atomically to ``--out``); exit 0 whenever it
was produced (2 on bad arguments). Parsing lives in pure ``parse_*`` functions
over text; ``Host`` injects every read so tests use a fake tree under
``tmp_path`` and recorded tool output.

JSON contract (parsed by ``benchmarks/cluster_bench/domain/nodefacts.py``)
=========================================================================
Integers are counts / bytes; ``-1`` means "unlimited" in ``limits``; times are
seconds since the epoch (``*_at``) or since boot (``t_s``).

host (``kind="node_facts"``, ``version=1``)
  node, hostname, started_at, collected_at (``time.time()`` when done),
  duration_s, notes [{source, error}]
  kernel   {release, uptime_s, cmdline_flags {flag: value} (perf-relevant
           allowlist only), iommu_groups, iommu_devices [..],
           iommu_group_types {bdf: "identity"|"DMA"|"DMA-FQ"|..}}
  driver   {version, flavor "open"|"proprietary"|null, nvrm (first line of
           /proc/driver/nvidia/version), cuda_version}
  modules  {name: loaded} for nvidia_peermem, nvidia_fs, ib_umad, ib_uverbs,
           rdma_ucm, mlx5_ib, mlx5_core, nvidia_uvm, gdrdrv, nvidia,
           nvidia_drm, nvidia_modeset
  cpu      {online, online_count, governors {name: cpus}, scaling_driver,
           cur_khz_min, cur_khz_max, cur_khz_mean, max_khz, boost,
           smt_control, smt_active}
  numa     {online, balancing, nodes [{node, cpulist, cpus, mem_total_bytes,
           distances [..]}]}
  memory   {mem_total_bytes, mem_available_bytes, swap_total_bytes,
           hugepages [{size_kb, total, free, reserved, surplus}],
           thp_enabled, thp_defrag, shm_total_bytes, shm_free_bytes}
  limits   {memlock_soft, memlock_hard, nofile_soft, nofile_hard, nproc_soft,
           nproc_hard} (RLIMIT_*, bytes for memlock, -1 = unlimited)
  cgroup   {version 1|2, path, cpu_quota_us (null = unlimited), cpu_period_us,
           cpuset_cpus, cpuset_count, pids_max (null = unlimited),
           pids_current, cpu_nr_throttled, cpu_throttled_us,
           memory_events {oom, oom_kill, max, high, ..}}
  pcie     [{bdf, kind "gpu"|"hca"|"bridge", vendor, device, pci_class,
           driver, numa_node, current_speed_gts, max_speed_gts,
           current_width, max_width, parent, depth (0 = endpoint, 1 = its
           parent bridge, ..), root_port, endpoints [bdf below a bridge],
           aer_correctable, aer_nonfatal, aer_fatal, acs_ctrl}]
           NVIDIA GPUs (class 0x0300/0x0302) and Mellanox functions (class
           0x02xx) from sysfs plus every upstream bridge of their paths.
  gpus     [{index, bus_id, uuid, serial, product, architecture, vbios,
           inforom_image, gsp_firmware, persistence_mode, mig_current,
           mig_pending, compute_mode, performance_state, ecc_current,
           ecc_pending, ecc_volatile_corrected, ecc_volatile_uncorrected,
           ecc_aggregate_corrected, ecc_aggregate_uncorrected,
           ecc_counters {"volatile.<tag>": n}, retired_sbe, retired_dbe,
           retired_pending, remapped_correctable, remapped_uncorrectable,
           remap_pending, remap_failure, power_draw_w, power_limit_w
           (enforced), power_requested_w, power_default_w, power_min_w,
           power_max_w, clocks_max_graphics_mhz, clocks_max_sm_mhz,
           clocks_max_mem_mhz, clocks_sm_mhz, clocks_mem_mhz,
           clocks_app_graphics_mhz, clocks_app_mem_mhz, clocks_event_active
           [reason], clocks_event_us {reason: us}, pcie_gen_current,
           pcie_gen_max, pcie_gen_device_max, pcie_gen_host_max,
           pcie_width_current, pcie_width_max, pcie_replay,
           pcie_replay_rollover, temp_c, mem_temp_c, temperature {tag: C},
           fabric_state, fabric_status, fabric_clique_id,
           fabric_cluster_uuid, fabric_health {key: value}, reset_required,
           drain_and_reset_recommended, recovery_action,
           xml_fields {"section/leaf": text}}] from ``nvidia-smi -q -x``
  nvlink   [{index, uuid, bus_id, links [{link, speed_gbs (null = inactive),
           errors {counter: n}}]}] from ``nvidia-smi nvlink -s`` / ``-e``
  p2p      {"read"|"write"|"nvlink"|"atomics": {GPUi: {GPUj: "OK"|"CNS"|..}}}
  hcas     [{name, bdf, fw_ver, board_id, ports [{port, link_layer, state,
           phys_state, rate, netdev, mtu, counters {name: n}, hw_counters
           {name: n}, ethtool {name: n}}]}] (ethtool -S only for Ethernet
           ports, filtered to pause/pfc/discard/drop/error/cnp/ecn names)
  time_sync {source "chronyc"|"timedatectl"|"adjtimex"|null, synchronized,
           offset_s, stratum, reference, leap_status, max_error_s,
           est_error_s, root_dispersion_s}
  docker   {server_version, storage_driver, cgroup_driver, cgroup_version,
           default_runtime, runtimes [..], nvidia_runtime,
           nvidia_container_runtime, nvidia_ctk, cdi_spec_dirs [..],
           docker_root_dir, security_options [..], warnings [..]}
  kernel_log {available, source "dmesg"|"kmsg"|null, error, lines,
           events [{t_s, kind "xid"|"mlx5"|"aer", code, pci, ours}] (most
           recent, oldest first; ``ours`` is False for PCI functions of other
           tenants), samples [{t_s, kind, text}]}

container (``kind="software_stack"``, ``version=1``)
  node, hostname, collected_at, python, packages {dist: version},
  missing_packages [..], torch {version, git_version, cuda, nccl, cudnn,
  arch_list, cuda_available, device_count, device_name, capability, errors}
  | null, ucx {version, revision, configured_with [..]} | null, nvcc,
  cuda_toolkit, ibv_devices [..], os_release {ID, VERSION_ID, ..}, glibc,
  nccl_libraries [path], nccl_plugins [path], env {NCCL_*|UCX_*|..: value},
  notes

disk-read (``kind="disk_read"``, ``version=1``)
  node, hostname, requested_path, path (file read), file_bytes, bytes,
  seconds, gbs (1e9 bytes/s), method "dd_direct"|"o_direct"|"buffered",
  direct, cache "bypassed"|"dropped"|"unknown", flushed (fsync'ed before
  timing), block_bytes, error, notes
"""

from __future__ import annotations

import argparse
import ctypes
import fnmatch
import importlib.metadata as importlib_metadata
import json
import math
import mmap
import os
import platform
import re
import resource
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

VERSION = 1
KIND_HOST = "node_facts"
KIND_SOFTWARE = "software_stack"
KIND_DISK = "disk_read"

DEFAULT_DEADLINE_S = 150.0
SMI_XML_TIMEOUT_S = 60.0
SMI_TIMEOUT_S = 30.0
TOPO_TIMEOUT_S = 20.0
SHORT_TIMEOUT_S = 10.0
DOCKER_TIMEOUT_S = 20.0
TORCH_TIMEOUT_S = 180.0
KMSG_TIMEOUT_S = 3.0
KMSG_MAX_BYTES = 32 << 20
MAX_EVENTS = 2000
SAMPLES_PER_KIND = 3
SAMPLE_TEXT_CHARS = 300
NOTE_CHARS = 300

MIB = 1 << 20
DEFAULT_DISK_MAX_BYTES = 8 << 30
DEFAULT_DISK_MAX_SECONDS = 120.0
DEFAULT_BLOCK_MIB = 16

GPU_VENDOR = "0x10de"
MELLANOX_VENDOR = "0x15b3"
GPU_CLASS_PREFIXES = ("0x0300", "0x0302")
NIC_CLASS_PREFIX = "0x02"
# Generation by link speed in GT/s (PCIe 1.0 .. 7.0).
PCIE_GEN_BY_GTS = (
    (2.5, 1),
    (5.0, 2),
    (8.0, 3),
    (16.0, 4),
    (32.0, 5),
    (64.0, 6),
    (128.0, 7),
)
ACS_CAP_ID = 0x000D
EXT_CAP_START = 0x100

MODULES = (
    "nvidia",
    "nvidia_uvm",
    "nvidia_drm",
    "nvidia_modeset",
    "nvidia_peermem",
    "nvidia_fs",
    "gdrdrv",
    "mlx5_core",
    "mlx5_ib",
    "ib_uverbs",
    "ib_umad",
    "rdma_ucm",
)
# Kernel command-line flags that matter for performance; nothing else is kept
# (boot lines may carry provider metadata URLs).
CMDLINE_PREFIXES = (
    "iommu",
    "intel_iommu",
    "amd_iommu",
    "intremap",
    "pci",
    "pcie_",
    "numa",
    "transparent_hugepage",
    "hugepage",
    "default_hugepagesz",
    "isolcpus",
    "nohz",
    "rcu_nocbs",
    "irqaffinity",
    "mitigations",
    "nosmt",
    "processor.max_cstate",
    "intel_idle",
    "idle",
    "intel_pstate",
    "amd_pstate",
    "cpufreq",
    "nvidia",
    "nouveau",
    "rd.driver.blacklist",
    "modprobe.blacklist",
    "init_on_alloc",
    "init_on_free",
    "mem_encrypt",
    "swiotlb",
    "skew_tick",
    "nmi_watchdog",
    "tsc",
    "clocksource",
    "systemd.unified_cgroup_hierarchy",
    "cgroup",
)
# ethtool -S names worth keeping for RoCE ports (per-ring counters excluded).
ETHTOOL_KEEP_RE = re.compile(
    r"pause|pfc|discard|drop|err|crc|symbol|cnp|ecn|marked|congest|"
    r"out_of_buffer|buf|storm|timeout|lost|corrected|fec",
    re.IGNORECASE,
)
ETHTOOL_PER_RING_RE = re.compile(r"^(?:rx|tx|ch|xdp|ptp)\d+_|^(?:rx|tx)_xsk_")

EXPECTED_PACKAGES = (
    "sglang",
    "sglang-kernel",
    "flashinfer-python",
    "torch",
    "triton",
    "transformers",
    "nixl",
    "ai-dynamo",
    "ai-dynamo-runtime",
    "xgrammar",
)
# Former names of expected packages, accepted in their place: SGLang's kernels
# were published as sgl-kernel before sglang-kernel (0.5.18+ pin sglang-kernel).
FORMER_NAMES: Dict[str, Tuple[str, ...]] = {"sglang-kernel": ("sgl-kernel",)}
OPTIONAL_PACKAGES = (
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
    "nixl-cu12",
    "nixl-cu13",
    "deep-ep",
    "deep-gemm",
    "mooncake-transfer-engine",
    "cuda-python",
    "cuda-bindings",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu12",
    "nvidia-cudnn-cu13",
    "nvidia-nvshmem-cu12",
    "nvidia-nvshmem-cu13",
    "nvidia-cutlass-dsl",
    "flash-attn",
    "aiperf",
    "numpy",
    "pydantic",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
    "uvloop",
)
PACKAGE_PREFIXES = (
    "nvidia-",
    "cuda-",
    "sgl",
    "flashinfer",
    "nixl",
    "deep-",
    "mooncake",
    "ai-dynamo",
    "triton",
    "torch",
    "nccl",
    "ucx",
)
ENV_PREFIXES = (
    "NCCL_",
    "UCX_",
    "NVSHMEM_",
    "NIXL_",
    "SGLANG_",
    "SGL_",
    "DYN_",
    "CUDA_",
    "NVIDIA_",
    "TORCH_",
    "PYTORCH_",
    "TRITON_",
    "FLASHINFER_",
    "MC_",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)
ENV_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSW", "KEY", "CREDENTIAL", "AUTH")
NCCL_LIB_DIRS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
    "/usr/local/nccl/lib",
    "/usr/local/cuda/lib64",
    "/opt/hpcx/nccl_rdma_sharp_plugin/lib",
    "/opt/amazon/ofi-nccl/lib",
)
NCCL_LIB_PATTERNS = ("libnccl.so*",)
NCCL_PLUGIN_PATTERNS = ("libnccl-net*.so*", "libnccl-tuner*.so*", "libnccl-env*.so*")
UCX_INFO_PATHS = (
    "ucx_info",
    "/usr/local/ucx/bin/ucx_info",
    "/opt/hpcx/ucx/bin/ucx_info",
)
NVCC_PATHS = ("nvcc", "/usr/local/cuda/bin/nvcc")

_NA_VALUES = frozenset(
    {
        "",
        "n/a",
        "[n/a]",
        "none",
        "not supported",
        "[not supported]",
        "unknown",
        "[unknown error]",
        "[insufficient permissions]",
        "requested functionality has been deprecated",
    }
)
_BDF_RE = re.compile(
    r"^(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])$"
)
_SYSFS_BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_HOST_BRIDGE_RE = re.compile(r"^pci[0-9a-f]{4}:[0-9a-f]{2}$")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_GTS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s", re.IGNORECASE)
_SMI_GPU_RE = re.compile(r"^GPU\s+(\d+):\s*(.*?)\s*(?:\(UUID:\s*([^)\s]+)\s*\))?\s*$")
_SMI_LINK_RE = re.compile(r"^\s*Link\s+(\d+)\s*:\s*(.*?)\s*$")
_SMI_LINK_SPEED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GB/s")
_SMI_LINK_COUNTER_RE = re.compile(r"^\s*Link\s+(\d+)\s*:\s*(.+?)\s*:\s*([-+]?\d+)\s*$")
_SMI_BARE_COUNTER_RE = re.compile(r"^\s*([A-Za-z][^:]*?)\s*:\s*([-+]?\d+)\s*$")
_P2P_LABEL_RE = re.compile(r"^GPU\d+$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_XID_RE = re.compile(
    r"NVRM: Xid \(PCI:([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2})(?:\.[0-7])?\):"
    r"\s*(\d+)"
)
_DMESG_RE = re.compile(r"^\[\s*(\d+\.\d+)\]\s?(.*)$")
_DD_GNU_RE = re.compile(
    r"(\d+)\s+bytes\b[^\n]*?copied,\s*([0-9]+(?:[.,][0-9]+)?)\s*s", re.IGNORECASE
)
_DD_BSD_RE = re.compile(r"(\d+)\s+bytes transferred in\s+([0-9]+(?:[.,][0-9]+)?)\s*sec")
_NVCC_RE = re.compile(r"release\s+(\d+\.\d+)(?:,\s*V(\d+(?:\.\d+)+))?")
_SNAKE_RE = re.compile(r"[^0-9a-zA-Z]+")

# adjtimex(2): modes, offset, freq, maxerror, esterror, status (glibc layout on
# LP64; the kernel writes ~208 bytes, the buffer is larger on purpose).
_TIMEX_HEAD = struct.Struct("@Illlli")
TIMEX_BUFFER_BYTES = 512
STA_UNSYNC = 0x0040
NTP_UNSYNC_MAXERROR_US = 16_000_000  # systemd's ntp_synced() threshold

# Platform-dependent os attributes (Linux only); None elsewhere.
_O_DIRECT = getattr(os, "O_DIRECT", None)
_FADVISE = getattr(os, "posix_fadvise", None)
_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)

_TORCH_MARK = "NODE_FACTS_TORCH="
# Runs in a child interpreter: a torch import that hangs or crashes never
# takes the agent down, and host mode never pays for importing torch.
_TORCH_SNIPPET = r"""
import json
import sys

MARK = "NODE_FACTS_TORCH="
errors = []

def safe(name, fn):
    try:
        return fn()
    except (AssertionError, AttributeError, ImportError, OSError, RuntimeError,
            TypeError, ValueError) as exc:
        errors.append("%s: %s" % (name, exc))
        return None

try:
    import torch
except (ImportError, OSError) as exc:
    print(MARK + json.dumps({"errors": ["import torch: %s" % exc]}))
    sys.exit(0)

nccl = safe("nccl", lambda: torch.cuda.nccl.version())
if isinstance(nccl, (tuple, list)):
    nccl = ".".join(str(p) for p in nccl)
elif isinstance(nccl, int):
    nccl = "%d.%d.%d" % (nccl // 10000, (nccl % 10000) // 100, nccl % 100)
cudnn = safe("cudnn", lambda: torch.backends.cudnn.version())
available = safe("cuda_available", lambda: bool(torch.cuda.is_available()))
count = safe("device_count", lambda: int(torch.cuda.device_count()))
name = cap = None
if available and count:
    name = safe("device_name", lambda: torch.cuda.get_device_name(0))
    pair = safe("capability", lambda: torch.cuda.get_device_capability(0))
    if pair:
        cap = "%d.%d" % tuple(pair)
out = {
    "version": str(torch.__version__),
    "git_version": safe("git_version", lambda: str(torch.version.git_version)),
    "cuda": safe("cuda", lambda: torch.version.cuda),
    "nccl": nccl,
    "cudnn": None if cudnn is None else str(cudnn),
    "arch_list": safe("arch_list", lambda: list(torch.cuda.get_arch_list())) or [],
    "cuda_available": available,
    "device_count": count,
    "device_name": name,
    "capability": cap,
    "errors": errors,
}
print(MARK + json.dumps(out))
"""


# --------------------------------------------------------------------------
# Tool execution and the injectable host
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def error(self) -> str:
        """Short "exit N: last stderr line" description."""
        lines = (self.stderr.strip() or self.stdout.strip()).splitlines()
        tail = lines[-1].strip() if lines else ""
        return f"exit {self.rc}: {tail}" if tail else f"exit {self.rc}"


def run_tool(
    argv: Sequence[str], timeout: float = SHORT_TIMEOUT_S
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
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return ToolResult(124, "", f"timed out after {timeout:.0f}s")
    except OSError as exc:  # not executable, exec format error, ...
        return ToolResult(126, "", str(exc))
    return ToolResult(proc.returncode, proc.stdout, proc.stderr)


def read_adjtimex() -> Optional[Tuple[int, int, int, int]]:
    """(clock state, status flags, maxerror us, esterror us) from adjtimex(2).

    ``modes`` is 0, so the call only reads (no CAP_SYS_TIME needed). ``None``
    where the symbol is missing (non-Linux) or the call fails (seccomp).
    """
    try:
        func = ctypes.CDLL(None, use_errno=True).adjtimex
    except (OSError, AttributeError):
        return None
    func.argtypes = [ctypes.c_void_p]
    func.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(TIMEX_BUFFER_BYTES)
    state = func(buf)
    if state < 0:
        return None
    _, _, _, maxerror, esterror, status = _TIMEX_HEAD.unpack_from(buf.raw)
    return state, status, maxerror, esterror


def read_kmsg(
    path: str, timeout_s: float = KMSG_TIMEOUT_S, max_bytes: int = KMSG_MAX_BYTES
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Every record of ``/dev/kmsg`` without blocking: (lines, error)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        return None, _os_error(exc)
    chunks: List[bytes] = []
    total = 0
    overruns = 0
    deadline = time.monotonic() + timeout_s
    try:
        while total < max_bytes and time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 8192)
            except BlockingIOError:
                break  # /dev/kmsg: no further records
            except BrokenPipeError:
                overruns += 1  # records overwritten under us; the next read resyncs
                if overruns > 100:
                    break
                continue
            except OSError as exc:
                if not chunks:
                    return None, _os_error(exc)
                break
            if not chunk:
                break  # a regular file (tests) ends with EOF
            chunks.append(chunk)
            total += len(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8", errors="replace").splitlines(), None


def installed_distributions() -> List[Tuple[str, str]]:
    """(name, version) of every installed Python distribution."""
    found: List[Tuple[str, str]] = []
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if name and dist.version:
            found.append((str(name), str(dist.version)))
    return found


def glibc_version() -> Optional[str]:
    """ "2.39" for glibc; ``None`` for other C libraries."""
    try:
        text = os.confstr("CS_GNU_LIBC_VERSION")
    except (ValueError, OSError):
        text = None
    if text and text.startswith("glibc "):
        return text.split(" ", 1)[1].strip() or None
    lib, version = platform.libc_ver()
    return version if lib == "glibc" and version else None


def _uname_release() -> str:
    return os.uname().release


@dataclass
class Host:
    """Everything the agent reads from the machine; tests substitute fakes.

    ``root`` prefixes every /sys, /proc, /dev and /etc path; the callables
    replace tools, syscalls and clocks.
    """

    root: str = "/"
    run: Callable[..., Optional[ToolResult]] = run_tool
    statvfs: Callable[[str], Any] = os.statvfs
    getrlimit: Callable[[int], Tuple[int, int]] = resource.getrlimit
    which: Callable[[str], Optional[str]] = shutil.which
    adjtimex: Callable[[], Optional[Tuple[int, int, int, int]]] = read_adjtimex
    clock: Callable[[], float] = time.time
    kernel_release: Callable[[], str] = _uname_release
    hostname: Callable[[], str] = socket.gethostname
    distributions: Callable[[], Iterable[Tuple[str, str]]] = installed_distributions
    libc_version: Callable[[], Optional[str]] = glibc_version
    python: str = sys.executable
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    search_path: Sequence[str] = field(default_factory=lambda: list(sys.path))

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *(p.lstrip("/") for p in parts))

    def display_path(self, real: str) -> str:
        """``real`` as it would read on the host (strips a test root)."""
        root = os.path.realpath(self.root)
        if root == os.sep:
            return real
        return os.sep + os.path.relpath(real, root)


# What an unexpected file or tool format can raise inside one collector.
SECTION_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    struct.error,
)


class Session:
    """Notes and the collection deadline shared by every collector."""

    def __init__(self, host: Host, deadline_s: float = DEFAULT_DEADLINE_S):
        self.host = host
        self.notes: List[Dict[str, str]] = []
        self._deadline = time.monotonic() + deadline_s

    def note(self, source: str, error: str) -> None:
        self.notes.append({"source": source, "error": error[:NOTE_CHARS]})

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def result(
        self, argv: Sequence[str], timeout: float, source: Optional[str] = None
    ) -> Optional[ToolResult]:
        """Run a tool within the deadline; ``None`` (with a note) when absent."""
        name = source or " ".join(argv[:3])
        remaining = self.remaining()
        if remaining <= 1.0:
            self.note(name, "skipped: collection deadline reached")
            return None
        res = self.host.run(list(argv), timeout=min(timeout, remaining))
        if res is None:
            self.note(name, "not installed")
        return res

    def guarded(self, name: str, fn: Callable[[], Any], default: Any) -> Any:
        """``fn()``, or ``default`` plus a note: one odd section never costs
        the whole document."""
        try:
            return fn()
        except SECTION_ERRORS as exc:
            self.note(f"collector:{name}", f"{type(exc).__name__}: {exc}")
            return default

    def tool(
        self, argv: Sequence[str], timeout: float, source: Optional[str] = None
    ) -> Optional[str]:
        """stdout of a successful run; ``None`` (with a note) otherwise."""
        res = self.result(argv, timeout, source)
        if res is None:
            return None
        if not res.ok:
            self.note(source or " ".join(argv[:3]), res.error())
            return None
        return res.stdout


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def _os_error(exc: OSError) -> str:
    return f"{type(exc).__name__}: {exc.strerror or exc}"


def natural_key(text: str) -> List[Any]:
    """Sort key ordering mlx5_2 before mlx5_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def normalize_bdf(raw: str) -> Optional[str]:
    """nvidia-smi "00000000:1B:00.0" -> sysfs "0000:1b:00.0" (None if invalid)."""
    match = _BDF_RE.match(raw.strip())
    if match is None:
        return None
    domain, bus, device, function = match.groups()
    return "%04x:%02x:%02x.%s" % (
        int(domain or "0", 16),
        int(bus, 16),
        int(device, 16),
        function,
    )


def snake(text: str) -> str:
    """ "FEC Errors - 0" -> "fec_errors_0"."""
    return _SNAKE_RE.sub("_", text.strip()).strip("_").lower()


def na(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(value.split())
    return None if text.lower() in _NA_VALUES else text


def parse_number(text: Optional[str]) -> Optional[float]:
    """Leading number of "700.00 W", "16x", "-5 C" or "1980 MHz"."""
    text = na(text)
    if text is None:
        return None
    match = _NUM_RE.search(text)
    return float(match.group(0)) if match else None


def parse_int(text: Optional[str]) -> Optional[int]:
    value = parse_number(text)
    return int(value) if value is not None else None


def parse_strict_int(text: Optional[str]) -> Optional[int]:
    """An integer only when the whole stripped text is one (sysfs counters)."""
    if text is None:
        return None
    value = text.strip()
    if re.match(r"^[-+]?\d+$", value):
        return int(value)
    if re.match(r"^0x[0-9a-fA-F]+$", value):
        return int(value, 16)
    return None


def parse_yes_no(text: Optional[str]) -> Optional[bool]:
    value = na(text)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("yes", "true", "enabled", "active", "1"):
        return True
    if lowered in ("no", "false", "disabled", "not active", "0"):
        return False
    return None


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:  # missing, permission denied, or an unreadable sysfs attr
        return None


def read_bytes(path: str, limit: int = 4096) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit)
    except OSError:
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


def _realpath_basename(path: str) -> Optional[str]:
    if not os.path.lexists(path):
        return None
    return os.path.basename(os.path.realpath(path))


def _first_line(text: Optional[str]) -> Optional[str]:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return None


# --------------------------------------------------------------------------
# Pure parsers: sysfs / procfs
# --------------------------------------------------------------------------


def parse_link_speed_gts(text: Optional[str]) -> Optional[float]:
    """GT/s from a sysfs link speed such as "32.0 GT/s PCIe" (None: unknown)."""
    match = _GTS_RE.search(text or "")
    return float(match.group(1)) if match else None


def gen_of_speed(gts: Optional[float]) -> Optional[int]:
    """PCIe generation of a link speed in GT/s (16.0 -> 4)."""
    if gts is None:
        return None
    for speed, gen in PCIE_GEN_BY_GTS:
        if abs(gts - speed) < 0.3:
            return gen
    return None


def parse_cpulist(text: Optional[str]) -> List[int]:
    """ "0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11] (empty when unparsable)."""
    cpus: List[int] = []
    for part in (text or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            if sep:
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(lo))
        except ValueError:
            return []
    return cpus


def parse_bracket_choice(text: Optional[str]) -> Optional[str]:
    """The selected value of "always [madvise] never"."""
    match = re.search(r"\[([^\]]+)\]", text or "")
    if match:
        return match.group(1)
    value = (text or "").strip()
    return value or None


def parse_meminfo(text: Optional[str]) -> Dict[str, int]:
    """/proc/meminfo (or a node meminfo) as name -> bytes (kB lines x 1024)."""
    values: Dict[str, int] = {}
    for line in (text or "").splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.split()
        if not parts or not parts[0].isdigit():
            continue
        unit = parts[1].lower() if len(parts) > 1 else ""
        number = int(parts[0])
        # Node meminfo keys read "Node 0 MemTotal"; keep the last word.
        name = key.split()[-1] if key.split() else key
        values[name] = number * 1024 if unit == "kb" else number
    return values


def parse_kv_lines(text: Optional[str], sep: str = " ") -> Dict[str, str]:
    """ "key value" (cpu.stat, memory.events) or "KEY=VALUE" lines."""
    values: Dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, found, value = line.partition(sep)
        if found:
            values[key.strip()] = value.strip()
    return values


def parse_int_map(text: Optional[str]) -> Dict[str, int]:
    """ "key 123" lines -> {key: 123} (non-integer values dropped)."""
    out: Dict[str, int] = {}
    for key, value in parse_kv_lines(text).items():
        number = parse_strict_int(value)
        if number is not None:
            out[key] = number
    return out


def parse_cgroup_cpu_max(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """cgroup v2 cpu.max "max 100000" / "800000 100000" -> (quota, period)."""
    parts = (text or "").split()
    if not parts:
        return None, None
    period = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    quota = int(parts[0]) if parts[0].isdigit() else None
    return quota, period


def parse_pids_max(text: Optional[str]) -> Optional[int]:
    value = (text or "").strip()
    return int(value) if value.isdigit() else None


def parse_aer(text: Optional[str]) -> Optional[int]:
    """Total of an ``aer_dev_*`` file ("TOTAL_ERR_COR 3" or the sum of lines)."""
    if text is None:
        return None
    total = None
    summed = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if parts[0].startswith("TOTAL_"):
            total = int(parts[1])
        else:
            summed += int(parts[1])
    return total if total is not None else summed


def parse_acs_ctrl(config: Optional[bytes]) -> Optional[int]:
    """ACS Control register of a bridge from its PCI config space bytes.

    ``None`` when the extended config space is not readable (non-root or no
    CAP_SYS_ADMIN gives only the first 64 bytes) or there is no ACS
    capability. Bit 2 (P2P Request Redirect) and bit 3 (P2P Completion
    Redirect) force peer-to-peer traffic up to the root complex.
    """
    if config is None or len(config) < EXT_CAP_START + 8:
        return None
    offset = EXT_CAP_START
    for _ in range(64):  # the chain is short; guard against loops
        if offset < EXT_CAP_START or offset + 8 > len(config):
            return None
        header = struct.unpack_from("<I", config, offset)[0]
        if header in (0, 0xFFFFFFFF):
            return None
        if header & 0xFFFF == ACS_CAP_ID:
            return struct.unpack_from("<H", config, offset + 6)[0]
        offset = (header >> 20) & 0xFFC
        if offset == 0:
            return None
    return None


def parse_cmdline(text: Optional[str]) -> Dict[str, str]:
    """Allowlisted kernel command-line flags as flag -> value ("" if bare)."""
    flags: Dict[str, str] = {}
    for token in (text or "").split():
        key, _, value = token.partition("=")
        if any(key == p or key.startswith(p) for p in CMDLINE_PREFIXES):
            flags[key] = value
    return flags


def parse_nvidia_version(text: Optional[str]) -> Dict[str, Optional[str]]:
    """/proc/driver/nvidia/version -> {version, flavor, nvrm}."""
    nvrm = None
    for line in (text or "").splitlines():
        if line.startswith("NVRM version:"):
            nvrm = " ".join(line.split())
            break
    if nvrm is None:
        return {"version": None, "flavor": None, "nvrm": None}
    match = re.search(r"\b(\d{3,}\.\d+(?:\.\d+)*)\b", nvrm)
    flavor = None
    if "Open Kernel Module" in nvrm:
        flavor = "open"
    elif "Kernel Module" in nvrm:
        flavor = "proprietary"
    return {
        "version": match.group(1) if match else None,
        "flavor": flavor,
        "nvrm": nvrm,
    }


def parse_proc_modules(text: Optional[str]) -> List[str]:
    """Loaded module names from /proc/modules."""
    return [line.split()[0] for line in (text or "").splitlines() if line.split()]


# --------------------------------------------------------------------------
# Pure parsers: nvidia-smi
# --------------------------------------------------------------------------


def _leaf(elem: Optional[ET.Element], path: str) -> Optional[str]:
    if elem is None:
        return None
    node = elem.find(path)
    if node is None or node.text is None:
        return None
    return na(node.text)


def _first_leaf(elem: Optional[ET.Element], *paths: str) -> Optional[str]:
    for path in paths:
        value = _leaf(elem, path)
        if value is not None:
            return value
    return None


def _flatten(elem: ET.Element, prefix: str, out: Dict[str, str]) -> None:
    children = list(elem)
    if not children:
        value = na(elem.text)
        if value is not None:
            out[prefix] = value
        return
    for child in children:
        _flatten(child, f"{prefix}/{child.tag}", out)


# Sections copied verbatim (flattened) into ``xml_fields`` so a format change
# never loses data even when the typed fields below miss it.
XML_SECTIONS = (
    "fabric",
    "ecc_mode",
    "ecc_errors",
    "retired_pages",
    "remapped_rows",
    "clocks_event_reasons",
    "clocks_throttle_reasons",
    "clocks_event_reasons_counters",
    "gpu_power_readings",
    "power_readings",
    "module_power_readings",
    "temperature",
    "pci/pci_gpu_link_info",
    "gpu_reset_status",
    "mig_mode",
    "max_clocks",
    "applications_clocks",
    "inforom_version",
    "platformInfo",
)
_REASON_PREFIXES = (
    "clocks_event_reasons_counters_",
    "clocks_event_reason_",
    "clocks_throttle_reason_",
    "clocks_event_reasons_",
    "counters_",
)


def _ecc_totals(
    section: Optional[ET.Element], scope: str, counters: Dict[str, int]
) -> Tuple[Optional[int], Optional[int]]:
    """(corrected, uncorrected) of one ``volatile``/``aggregate`` block.

    Newer drivers list ``sram_correctable`` / ``dram_uncorrectable`` / ...;
    older ones nest ``single_bit`` / ``double_bit`` blocks with a ``total``.
    """
    if section is None:
        return None, None
    single = section.find("single_bit")
    double = section.find("double_bit")
    if single is not None or double is not None:
        for name, block in (("single_bit", single), ("double_bit", double)):
            for leaf in list(block) if block is not None else []:
                value = parse_strict_int(na(leaf.text))
                if value is not None:
                    counters[f"{scope}.{name}.{leaf.tag}"] = value
        return parse_int(_leaf(single, "total")), parse_int(_leaf(double, "total"))
    corrected: Optional[int] = None
    uncorrected: Optional[int] = None
    for leaf in section:
        value = parse_strict_int(na(leaf.text))
        if value is None:
            continue
        counters[f"{scope}.{leaf.tag}"] = value
        tag = leaf.tag.lower()
        if "uncorrectable" in tag:
            uncorrected = (uncorrected or 0) + value
        elif "correctable" in tag:
            corrected = (corrected or 0) + value
    return corrected, uncorrected


def _reason_name(tag: str) -> str:
    for prefix in _REASON_PREFIXES:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return tag


def _clock_reasons(gpu: ET.Element) -> Tuple[List[str], Dict[str, int]]:
    active: List[str] = []
    section = gpu.find("clocks_event_reasons")
    if section is None:
        section = gpu.find("clocks_throttle_reasons")
    for leaf in list(section) if section is not None else []:
        if (na(leaf.text) or "").lower() == "active":
            active.append(_reason_name(leaf.tag))
    counters: Dict[str, int] = {}
    # UNVERIFIED tag names: newer drivers add cumulative "Clocks Event Reasons
    # Counters" in microseconds; parse any child holding "<n> us".
    for name in ("clocks_event_reasons_counters", "clocks_throttle_reasons_counters"):
        block = gpu.find(name)
        for leaf in list(block) if block is not None else []:
            value = parse_int(leaf.text)
            if value is not None:
                counters[_reason_name(leaf.tag)] = value
    return active, counters


def _fabric(gpu: ET.Element) -> Dict[str, Any]:
    section = gpu.find("fabric")
    out: Dict[str, Any] = {
        "fabric_state": None,
        "fabric_status": None,
        "fabric_clique_id": None,
        "fabric_cluster_uuid": None,
        "fabric_health": {},
    }
    if section is None:
        return out
    for leaf in section:
        key = leaf.tag.replace("_", "").lower()
        if key == "state":
            out["fabric_state"] = na(leaf.text)
        elif key == "status":
            out["fabric_status"] = na(leaf.text)
        elif key == "cliqueid":
            out["fabric_clique_id"] = na(leaf.text)
        elif key == "clusteruuid":
            out["fabric_cluster_uuid"] = na(leaf.text)
        elif key == "health":
            health: Dict[str, str] = {}
            _flatten(leaf, "", health)
            out["fabric_health"] = {k.strip("/"): v for k, v in health.items()}
    return out


def _temperatures(gpu: ET.Element) -> Dict[str, float]:
    temps: Dict[str, float] = {}
    section = gpu.find("temperature")
    for leaf in list(section) if section is not None else []:
        value = parse_number(leaf.text)
        if value is not None:
            temps[leaf.tag] = value
    return temps


def parse_gpu_element(gpu: ET.Element, index: int) -> Dict[str, Any]:
    """One ``<gpu>`` element of ``nvidia-smi -q -x`` as a ``gpus`` record."""
    bus = normalize_bdf(gpu.get("id") or "") or normalize_bdf(
        _leaf(gpu, "pci/pci_bus_id") or ""
    )
    ecc_counters: Dict[str, int] = {}
    errors = gpu.find("ecc_errors")
    vol_c, vol_u = _ecc_totals(
        errors.find("volatile") if errors is not None else None,
        "volatile",
        ecc_counters,
    )
    agg_c, agg_u = _ecc_totals(
        errors.find("aggregate") if errors is not None else None,
        "aggregate",
        ecc_counters,
    )
    power = gpu.find("gpu_power_readings")
    legacy_power = power is None
    if legacy_power:
        power = gpu.find("power_readings")
    if legacy_power:
        enforced = _first_leaf(power, "enforced_power_limit", "power_limit")
        requested = _leaf(power, "power_limit")
    else:
        enforced = _first_leaf(power, "current_power_limit", "enforced_power_limit")
        requested = _first_leaf(power, "requested_power_limit", "power_limit")
    active, reason_us = _clock_reasons(gpu)
    temps = _temperatures(gpu)
    link = "pci/pci_gpu_link_info"
    xml_fields: Dict[str, str] = {}
    for section in XML_SECTIONS:
        node = gpu.find(section)
        if node is not None:
            _flatten(node, section, xml_fields)
    for leaf in ("gpu_recovery_action", "addressing_mode", "c2c_mode"):
        value = _leaf(gpu, leaf)
        if value is not None:
            xml_fields[leaf] = value
    record: Dict[str, Any] = {
        "index": index,
        "bus_id": bus,
        "uuid": _leaf(gpu, "uuid"),
        "serial": _leaf(gpu, "serial"),
        "product": _leaf(gpu, "product_name"),
        "architecture": _leaf(gpu, "product_architecture"),
        "vbios": _leaf(gpu, "vbios_version"),
        "inforom_image": _leaf(gpu, "inforom_version/img_version"),
        "gsp_firmware": _leaf(gpu, "gsp_firmware_version"),
        "persistence_mode": _leaf(gpu, "persistence_mode"),
        "mig_current": _leaf(gpu, "mig_mode/current_mig"),
        "mig_pending": _leaf(gpu, "mig_mode/pending_mig"),
        "compute_mode": _leaf(gpu, "compute_mode"),
        "performance_state": _leaf(gpu, "performance_state"),
        "ecc_current": _leaf(gpu, "ecc_mode/current_ecc"),
        "ecc_pending": _leaf(gpu, "ecc_mode/pending_ecc"),
        "ecc_volatile_corrected": vol_c,
        "ecc_volatile_uncorrected": vol_u,
        "ecc_aggregate_corrected": agg_c,
        "ecc_aggregate_uncorrected": agg_u,
        "ecc_counters": ecc_counters,
        "retired_sbe": parse_int(
            _leaf(gpu, "retired_pages/multiple_single_bit_retirement/retired_count")
        ),
        "retired_dbe": parse_int(
            _leaf(gpu, "retired_pages/double_bit_retirement/retired_count")
        ),
        "retired_pending": parse_yes_no(
            _first_leaf(
                gpu,
                "retired_pages/pending_retirement",
                "retired_pages/pending_blacklist",
            )
        ),
        "remapped_correctable": parse_int(
            _leaf(gpu, "remapped_rows/remapped_row_corr")
        ),
        "remapped_uncorrectable": parse_int(
            _leaf(gpu, "remapped_rows/remapped_row_unc")
        ),
        "remap_pending": parse_yes_no(_leaf(gpu, "remapped_rows/remapped_row_pending")),
        "remap_failure": parse_yes_no(_leaf(gpu, "remapped_rows/remapped_row_failure")),
        "power_draw_w": parse_number(
            _first_leaf(power, "average_power_draw", "power_draw", "instant_power_draw")
        ),
        "power_limit_w": parse_number(enforced),
        "power_requested_w": parse_number(requested),
        "power_default_w": parse_number(_leaf(power, "default_power_limit")),
        "power_min_w": parse_number(_leaf(power, "min_power_limit")),
        "power_max_w": parse_number(_leaf(power, "max_power_limit")),
        "clocks_max_graphics_mhz": parse_int(_leaf(gpu, "max_clocks/graphics_clock")),
        "clocks_max_sm_mhz": parse_int(_leaf(gpu, "max_clocks/sm_clock")),
        "clocks_max_mem_mhz": parse_int(_leaf(gpu, "max_clocks/mem_clock")),
        "clocks_sm_mhz": parse_int(_leaf(gpu, "clocks/sm_clock")),
        "clocks_mem_mhz": parse_int(_leaf(gpu, "clocks/mem_clock")),
        "clocks_app_graphics_mhz": parse_int(
            _leaf(gpu, "applications_clocks/graphics_clock")
        ),
        "clocks_app_mem_mhz": parse_int(_leaf(gpu, "applications_clocks/mem_clock")),
        "clocks_event_active": active,
        "clocks_event_us": reason_us,
        "pcie_gen_current": parse_int(_leaf(gpu, f"{link}/pcie_gen/current_link_gen")),
        "pcie_gen_max": parse_int(_leaf(gpu, f"{link}/pcie_gen/max_link_gen")),
        "pcie_gen_device_max": parse_int(
            _leaf(gpu, f"{link}/pcie_gen/max_device_link_gen")
        ),
        "pcie_gen_host_max": parse_int(
            _leaf(gpu, f"{link}/pcie_gen/max_host_link_gen")
        ),
        "pcie_width_current": parse_int(
            _leaf(gpu, f"{link}/link_widths/current_link_width")
        ),
        "pcie_width_max": parse_int(_leaf(gpu, f"{link}/link_widths/max_link_width")),
        "pcie_replay": parse_int(_leaf(gpu, "pci/replay_counter")),
        "pcie_replay_rollover": parse_int(_leaf(gpu, "pci/replay_rollover_counter")),
        "temp_c": temps.get("gpu_temp"),
        "mem_temp_c": temps.get("memory_temp"),
        "temperature": temps,
        "reset_required": _leaf(gpu, "gpu_reset_status/reset_required"),
        "drain_and_reset_recommended": _leaf(
            gpu, "gpu_reset_status/drain_and_reset_recommended"
        ),
        "recovery_action": _leaf(gpu, "gpu_recovery_action"),
        "xml_fields": xml_fields,
    }
    record.update(_fabric(gpu))
    return record


def parse_smi_xml(text: str) -> Dict[str, Any]:
    """``nvidia-smi -q -x`` -> {driver_version, cuda_version, gpus [..]}.

    Raises ``ValueError`` when the text is not an nvidia_smi_log document.
    """
    try:
        root = ET.fromstring(text.strip())
    except ET.ParseError as exc:
        raise ValueError(f"nvidia-smi XML does not parse: {exc}") from exc
    if root.tag != "nvidia_smi_log":
        raise ValueError(f"unexpected nvidia-smi XML root {root.tag!r}")
    gpus = [parse_gpu_element(gpu, i) for i, gpu in enumerate(root.findall("gpu"))]
    return {
        "driver_version": _leaf(root, "driver_version"),
        "cuda_version": _leaf(root, "cuda_version"),
        "gpus": gpus,
    }


def parse_nvlink_status(text: Optional[str]) -> List[Dict[str, Any]]:
    """``nvidia-smi nvlink -s``: per GPU, every link and its speed (GB/s)."""
    gpus: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in _ANSI_RE.sub("", text or "").splitlines():
        gpu_match = _SMI_GPU_RE.match(line.strip())
        if gpu_match and line[:1] != " " and line[:1] != "\t":
            current = {
                "index": int(gpu_match.group(1)),
                "uuid": gpu_match.group(3),
                "bus_id": None,
                "links": [],
            }
            gpus.append(current)
            continue
        link_match = _SMI_LINK_RE.match(line)
        if link_match and current is not None:
            speed = _SMI_LINK_SPEED_RE.search(link_match.group(2))
            current["links"].append(
                {
                    "link": int(link_match.group(1)),
                    "speed_gbs": float(speed.group(1)) if speed else None,
                    "errors": {},
                }
            )
    return gpus


def parse_nvlink_errors(text: Optional[str]) -> Dict[int, Dict[int, Dict[str, int]]]:
    """``nvidia-smi nvlink -e``: GPU index -> link -> counter -> value.

    UNVERIFIED across driver branches: accepts "Link N: <name>: <int>" lines
    and "Link N:" headers followed by "<name>: <int>" lines; non-integer
    values (BER floats) are skipped.
    """
    result: Dict[int, Dict[int, Dict[str, int]]] = {}
    gpu: Optional[int] = None
    link: Optional[int] = None
    for line in _ANSI_RE.sub("", text or "").splitlines():
        gpu_match = _SMI_GPU_RE.match(line.strip())
        if gpu_match and line[:1] not in (" ", "\t"):
            gpu = int(gpu_match.group(1))
            link = None
            result.setdefault(gpu, {})
            continue
        if gpu is None:
            continue
        counter = _SMI_LINK_COUNTER_RE.match(line)
        if counter:
            link = int(counter.group(1))
            name = snake(counter.group(2))
            result[gpu].setdefault(link, {})[name] = int(counter.group(3))
            continue
        header = _SMI_LINK_RE.match(line)
        if header and not header.group(2):
            link = int(header.group(1))
            continue
        bare = _SMI_BARE_COUNTER_RE.match(line)
        if bare and link is not None:
            result[gpu].setdefault(link, {})[snake(bare.group(1))] = int(bare.group(2))
    return result


def parse_p2p_matrix(text: Optional[str]) -> Dict[str, Dict[str, str]]:
    """``nvidia-smi topo -p2p r|w|n|a`` as GPUi -> GPUj -> status."""
    lines = _ANSI_RE.sub("", text or "").splitlines()
    columns: List[str] = []
    matrix: Dict[str, Dict[str, str]] = {}
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].rstrip(":").lower() == "legend":
            break
        if not columns:
            if all(_P2P_LABEL_RE.match(t) for t in tokens):
                columns = tokens
            continue
        if _P2P_LABEL_RE.match(tokens[0]):
            matrix[tokens[0]] = dict(zip(columns, tokens[1 : 1 + len(columns)]))
    return matrix


# --------------------------------------------------------------------------
# Pure parsers: networking, time, docker, kernel log
# --------------------------------------------------------------------------


def parse_ethtool_stats(text: Optional[str]) -> Dict[str, int]:
    """``ethtool -S`` -> name -> value (every integer counter)."""
    stats: Dict[str, int] = {}
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        number = parse_strict_int(value)
        name = key.strip()
        if number is not None and name and " " not in name:
            stats[name] = number
    return stats


def filter_ethtool(stats: Mapping[str, int]) -> Dict[str, int]:
    """Pause / PFC / discard / error / congestion counters, no per-ring ones."""
    return {
        name: value
        for name, value in sorted(stats.items())
        if ETHTOOL_KEEP_RE.search(name) and not ETHTOOL_PER_RING_RE.match(name)
    }


def parse_chronyc_tracking(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """``chronyc -c tracking`` CSV -> time_sync fields (None if unparsable).

    Fields: ref id, ref name, stratum, ref time, system time offset (s), last
    offset, RMS offset, frequency, residual frequency, skew, root delay, root
    dispersion, update interval, leap status.
    """
    line = _first_line(text)
    if line is None:
        return None
    parts = line.split(",")
    if len(parts) < 14:
        return None
    try:
        stratum = int(parts[2])
        offset = float(parts[4])
        dispersion = float(parts[11])
    except ValueError:
        return None
    leap = parts[13].strip()
    synchronized = leap.lower() != "not synchronised" and parts[0] != "00000000"
    return {
        "source": "chronyc",
        "synchronized": synchronized,
        "offset_s": offset,
        "stratum": stratum,
        "reference": parts[1].strip() or parts[0].strip(),
        "leap_status": leap,
        "max_error_s": None,
        "est_error_s": None,
        "root_dispersion_s": dispersion,
    }


def parse_timedatectl(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """``timedatectl show`` -> time_sync fields (None without NTPSynchronized)."""
    values = parse_kv_lines(text, "=")
    synced = parse_yes_no(values.get("NTPSynchronized"))
    if synced is None:
        return None
    return {
        "source": "timedatectl",
        "synchronized": synced,
        "offset_s": None,
        "stratum": None,
        "reference": None,
        "leap_status": None,
        "max_error_s": None,
        "est_error_s": None,
        "root_dispersion_s": None,
    }


def interpret_adjtimex(
    state: int, status: int, maxerror_us: int, esterror_us: int
) -> Dict[str, Any]:
    """Kernel clock discipline (adjtimex) -> time_sync fields.

    Same rule as systemd's ``ntp_synced()``: synchronized unless STA_UNSYNC is
    set or the maximum error reached 16 s.
    """
    synced = not (status & STA_UNSYNC) and maxerror_us < NTP_UNSYNC_MAXERROR_US
    return {
        "source": "adjtimex",
        "synchronized": bool(synced),
        "offset_s": None,
        "stratum": None,
        "reference": None,
        "leap_status": f"state={state}",
        "max_error_s": maxerror_us / 1e6,
        "est_error_s": esterror_us / 1e6,
        "root_dispersion_s": None,
    }


def parse_docker_info(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """``docker info --format '{{json .}}'`` -> docker fields."""
    line = None
    for candidate in reversed((text or "").splitlines()):
        if candidate.strip().startswith("{"):
            line = candidate.strip()
            break
    if line is None:
        return None
    try:
        info = json.loads(line)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    runtimes = info.get("Runtimes") if isinstance(info.get("Runtimes"), dict) else {}
    nvidia = any(
        "nvidia" in name.lower() or "nvidia" in json.dumps(spec).lower()
        for name, spec in runtimes.items()
    )

    def strings(key: str) -> List[str]:
        value = info.get(key)
        return [str(v) for v in value] if isinstance(value, list) else []

    return {
        "server_version": info.get("ServerVersion") or None,
        "storage_driver": info.get("Driver") or None,
        "cgroup_driver": info.get("CgroupDriver") or None,
        "cgroup_version": str(info["CgroupVersion"])
        if info.get("CgroupVersion")
        else None,
        "default_runtime": info.get("DefaultRuntime") or None,
        "runtimes": sorted(runtimes),
        "nvidia_runtime": nvidia,
        "nvidia_container_runtime": None,
        "nvidia_ctk": None,
        "cdi_spec_dirs": strings("CDISpecDirs"),
        "docker_root_dir": info.get("DockerRootDir") or None,
        "security_options": strings("SecurityOptions"),
        "warnings": strings("Warnings")[:10],
        "server_errors": strings("ServerErrors")[:3],
    }


def parse_kmsg_line(line: str) -> Optional[Tuple[int, Optional[float], str]]:
    """A /dev/kmsg record "6,1234,5678901,-;text" -> (level, t_s, text)."""
    head, sep, text = line.partition(";")
    if not sep:
        return None  # continuation lines (" KEY=VALUE") carry no message
    fields = head.split(",")
    if len(fields) < 3 or not fields[0].isdigit():
        return None
    level = int(fields[0]) & 7
    t_s = int(fields[2]) / 1e6 if fields[2].isdigit() else None
    return level, t_s, text


def parse_dmesg_line(line: str) -> Tuple[Optional[float], str]:
    """ "[ 1234.567890] text" -> (seconds since boot, text)."""
    match = _DMESG_RE.match(line)
    if match:
        return float(match.group(1)), match.group(2)
    return None, line


def classify_kernel_line(
    text: str, gpu_pcis: Mapping[str, str], our_bdfs: Iterable[str] = ()
) -> Optional[Dict[str, Any]]:
    """An Xid / mlx5 / AER event from one kernel log message (else None).

    ``gpu_pcis`` maps "0000:18:00" (the Xid form) -> our GPU's full BDF;
    ``our_bdfs`` are this node's PCI functions (empty: everything is ours).
    The kernel log of a shared host also shows other tenants' devices.
    """
    xid = _XID_RE.search(text)
    if xid:
        pci = xid.group(1).lower()
        ours = pci in gpu_pcis
        return {
            "kind": "xid",
            "code": int(xid.group(2)),
            "pci": gpu_pcis.get(pci, pci),
            "ours": ours,
        }
    if "mlx5" in text:
        kind = "mlx5"
    elif "AER:" in text or "PCIe Bus Error" in text:
        kind = "aer"
    else:
        return None
    pci = _pci_in(text)
    known = set(our_bdfs)
    ours = not known or pci is None or pci in known
    return {"kind": kind, "code": None, "pci": pci, "ours": ours}


def _pci_in(text: str) -> Optional[str]:
    match = re.search(r"\b([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\b", text)
    return match.group(1) if match else None


def scan_kernel_log(
    entries: Iterable[Tuple[Optional[float], str]],
    gpu_pcis: Mapping[str, str],
    our_bdfs: Iterable[str] = (),
) -> Dict[str, Any]:
    """Events and a few sample lines from (t_s, text) kernel log entries."""
    events: List[Dict[str, Any]] = []
    samples: Dict[str, List[Dict[str, Any]]] = {}
    known = sorted(set(our_bdfs))
    lines = 0
    for t_s, text in entries:
        lines += 1
        event = classify_kernel_line(text, gpu_pcis, known)
        if event is None:
            continue
        event["t_s"] = t_s
        events.append(event)
        kept = samples.setdefault(event["kind"], [])
        kept.append(
            {"t_s": t_s, "kind": event["kind"], "text": text[:SAMPLE_TEXT_CHARS]}
        )
        del kept[:-SAMPLES_PER_KIND]
    flat = [s for kind in sorted(samples) for s in samples[kind]]
    return {"lines": lines, "events": events[-MAX_EVENTS:], "samples": flat}


# --------------------------------------------------------------------------
# Pure parsers: container mode and disk reads
# --------------------------------------------------------------------------


def normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def select_packages(
    dists: Iterable[Tuple[str, str]]
) -> Tuple[Dict[str, str], List[str]]:
    """(reported packages, missing expected packages) from installed dists."""
    installed: Dict[str, str] = {}
    for name, version in dists:
        installed.setdefault(normalize_dist(name), version)
    wanted = set(EXPECTED_PACKAGES) | set(OPTIONAL_PACKAGES)
    packages = {
        name: version
        for name, version in installed.items()
        if name in wanted or name.startswith(PACKAGE_PREFIXES)
    }
    missing = [
        name
        for name in EXPECTED_PACKAGES
        if name not in installed
        and not any(old in installed for old in FORMER_NAMES.get(name, ()))
    ]
    return dict(sorted(packages.items())), missing


def parse_torch_probe(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The JSON line the torch snippet prints after ``NODE_FACTS_TORCH=``."""
    for line in reversed((text or "").splitlines()):
        if line.startswith(_TORCH_MARK):
            try:
                value = json.loads(line[len(_TORCH_MARK) :])
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


def parse_ucx_info(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """``ucx_info -v`` -> {version, revision, configured_with}."""
    version = revision = None
    configured: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.lstrip("#").strip()
        lowered = line.lower()
        if lowered.startswith("library version:"):
            version = line.split(":", 1)[1].strip() or None
        elif lowered.startswith("uct version=") or lowered.startswith("ucx version="):
            match = re.search(r"version=(\S+)(?:\s+revision\s+(\S+))?", line)
            if match:
                version = version or match.group(1)
                revision = revision or match.group(2)
        elif "revision" in lowered and lowered.startswith("git branch"):
            revision = line.rsplit("revision", 1)[1].strip() or None
        elif lowered.startswith("configured with:"):
            flags = line.split(":", 1)[1].split()
            configured = [f for f in flags if f.startswith(("--with-", "--enable-"))]
    if version is None:
        return None
    return {
        "version": version,
        "revision": revision,
        "configured_with": configured[:40],
    }


def parse_nvcc_version(text: Optional[str]) -> Optional[str]:
    """ "release 13.0, V13.0.88" -> "13.0.88"."""
    match = _NVCC_RE.search(text or "")
    if not match:
        return None
    return match.group(2) or match.group(1)


def parse_ibv_devinfo_list(text: Optional[str]) -> List[str]:
    """``ibv_devinfo -l`` -> device names."""
    names: List[str] = []
    for line in (text or "").splitlines():
        token = line.strip()
        if not token or token.endswith(":") or " " in token:
            continue  # the "8 HCAs found:" header, "No IB devices found"
        names.append(token)
    return names


def parse_os_release(text: Optional[str]) -> Dict[str, str]:
    keep = ("ID", "ID_LIKE", "VERSION_ID", "VERSION_CODENAME", "PRETTY_NAME")
    values = parse_kv_lines(text, "=")
    return {k: values[k].strip().strip('"').strip("'") for k in keep if k in values}


def parse_dd_output(text: Optional[str]) -> Optional[Tuple[int, float]]:
    """(bytes, seconds) from GNU / BusyBox / BSD ``dd`` statistics."""
    for regex in (_DD_GNU_RE, _DD_BSD_RE):
        match = regex.search(text or "")
        if match:
            return int(match.group(1)), float(match.group(2).replace(",", "."))
    return None


# --------------------------------------------------------------------------
# Host collectors
# --------------------------------------------------------------------------


def _pci_kind(vendor: Optional[str], pci_class: Optional[str]) -> Optional[str]:
    if vendor == GPU_VENDOR and (pci_class or "").startswith(GPU_CLASS_PREFIXES):
        return "gpu"
    if vendor == MELLANOX_VENDOR and (pci_class or "").startswith(NIC_CLASS_PREFIX):
        return "hca"
    return None


def _positive(value: Optional[int]) -> Optional[int]:
    return value if value is not None and value > 0 else None


def _pci_record(
    dev: str, bdf: str, kind: str, parent: Optional[str], depth: int
) -> Dict[str, Any]:
    numa = parse_strict_int(read_stripped(os.path.join(dev, "numa_node")))
    return {
        "bdf": bdf,
        "kind": kind,
        "vendor": read_stripped(os.path.join(dev, "vendor")),
        "device": read_stripped(os.path.join(dev, "device")),
        "pci_class": read_stripped(os.path.join(dev, "class")),
        "driver": _realpath_basename(os.path.join(dev, "driver")),
        "numa_node": numa if numa is not None and numa >= 0 else None,
        "current_speed_gts": parse_link_speed_gts(
            read_stripped(os.path.join(dev, "current_link_speed"))
        ),
        "max_speed_gts": parse_link_speed_gts(
            read_stripped(os.path.join(dev, "max_link_speed"))
        ),
        "current_width": _positive(
            parse_strict_int(read_stripped(os.path.join(dev, "current_link_width")))
        ),
        "max_width": _positive(
            parse_strict_int(read_stripped(os.path.join(dev, "max_link_width")))
        ),
        "parent": parent,
        "depth": depth,
        "root_port": False,
        "endpoints": [],
        "aer_correctable": parse_aer(
            read_text(os.path.join(dev, "aer_dev_correctable"))
        ),
        "aer_nonfatal": parse_aer(read_text(os.path.join(dev, "aer_dev_nonfatal"))),
        "aer_fatal": parse_aer(read_text(os.path.join(dev, "aer_dev_fatal"))),
        "acs_ctrl": None,
    }


def collect_pcie(session: Session) -> List[Dict[str, Any]]:
    """Link state of GPUs, Mellanox functions and every bridge above them."""
    host = session.host
    devices = host.path("sys/bus/pci/devices")
    # Fixed-width lowercase hex BDFs sort correctly as plain strings.
    names = sorted(_listdir(devices))
    if not names:
        session.note("sysfs:pci", "no /sys/bus/pci/devices entries")
        return []
    endpoints: List[Dict[str, Any]] = []
    bridges: Dict[str, Dict[str, Any]] = {}
    for bdf in names:
        if not _SYSFS_BDF_RE.match(bdf):
            continue
        link = os.path.join(devices, bdf)
        kind = _pci_kind(
            read_stripped(os.path.join(link, "vendor")),
            read_stripped(os.path.join(link, "class")),
        )
        if kind is None or os.path.lexists(os.path.join(link, "physfn")):
            continue  # not ours, or an SR-IOV virtual function
        real = os.path.realpath(link)
        parts = real.split(os.sep)
        chain: List[Tuple[str, str]] = []  # (bdf, path), nearest first
        top_is_host_bridge = False
        for i in range(len(parts) - 2, -1, -1):
            if _SYSFS_BDF_RE.match(parts[i]):
                chain.append((parts[i], os.sep.join(parts[: i + 1])))
                continue
            top_is_host_bridge = bool(_HOST_BRIDGE_RE.match(parts[i]))
            break
        endpoints.append(
            _pci_record(real, bdf, kind, chain[0][0] if chain else None, 0)
        )
        for depth, (bridge_bdf, path) in enumerate(chain, start=1):
            record = bridges.get(bridge_bdf)
            if record is None:
                parent = chain[depth][0] if depth < len(chain) else None
                record = _pci_record(path, bridge_bdf, "bridge", parent, depth)
                record["root_port"] = depth == len(chain) and top_is_host_bridge
                record["acs_ctrl"] = parse_acs_ctrl(
                    read_bytes(os.path.join(path, "config"))
                )
                bridges[bridge_bdf] = record
            record["depth"] = min(record["depth"], depth)
            record["endpoints"].append(bdf)
    if not endpoints:
        session.note("sysfs:pci", "no NVIDIA GPU or Mellanox function found")
    ordered = [bridges[bdf] for bdf in sorted(bridges)]
    return endpoints + ordered


def collect_numa(session: Session) -> Dict[str, Any]:
    host = session.host
    base = host.path("sys/devices/system/node")
    nodes = []
    for name in sorted(_listdir(base), key=natural_key):
        match = re.match(r"^node(\d+)$", name)
        if not match:
            continue
        node_dir = os.path.join(base, name)
        cpulist = read_stripped(os.path.join(node_dir, "cpulist"))
        meminfo = parse_meminfo(read_text(os.path.join(node_dir, "meminfo")))
        distances = [
            int(d)
            for d in (read_stripped(os.path.join(node_dir, "distance")) or "").split()
            if d.isdigit()
        ]
        nodes.append(
            {
                "node": int(match.group(1)),
                "cpulist": cpulist,
                "cpus": len(parse_cpulist(cpulist)) if cpulist else None,
                "mem_total_bytes": meminfo.get("MemTotal"),
                "distances": distances,
            }
        )
    if not nodes:
        session.note("sysfs:numa", "no /sys/devices/system/node/node* entries")
    return {
        "online": read_stripped(os.path.join(base, "online")),
        "balancing": parse_strict_int(
            read_stripped(host.path("proc/sys/kernel/numa_balancing"))
        ),
        "nodes": nodes,
    }


def _hugepage_pool(pool_dir: str, size_kb: int) -> Dict[str, Any]:
    def count(attr: str) -> Optional[int]:
        return parse_strict_int(read_stripped(os.path.join(pool_dir, attr)))

    return {
        "size_kb": size_kb,
        "total": count("nr_hugepages") or 0,
        "free": count("free_hugepages"),
        "reserved": count("resv_hugepages"),
        "surplus": count("surplus_hugepages"),
    }


def collect_memory(session: Session) -> Dict[str, Any]:
    host = session.host
    meminfo = parse_meminfo(read_text(host.path("proc/meminfo")))
    pools = []
    hp_dir = host.path("sys/kernel/mm/hugepages")
    for name in sorted(_listdir(hp_dir), key=natural_key):
        match = re.match(r"^hugepages-(\d+)kB$", name)
        if match:
            pools.append(
                _hugepage_pool(os.path.join(hp_dir, name), int(match.group(1)))
            )
    thp = host.path("sys/kernel/mm/transparent_hugepage")
    shm_total = shm_free = None
    try:
        st = host.statvfs("/dev/shm")
        shm_total = st.f_blocks * st.f_frsize
        shm_free = st.f_bavail * st.f_frsize
    except OSError as exc:
        session.note("statvfs:/dev/shm", _os_error(exc))
    return {
        "mem_total_bytes": meminfo.get("MemTotal"),
        "mem_available_bytes": meminfo.get("MemAvailable"),
        "swap_total_bytes": meminfo.get("SwapTotal"),
        "hugepages": pools,
        "thp_enabled": parse_bracket_choice(read_text(os.path.join(thp, "enabled"))),
        "thp_defrag": parse_bracket_choice(read_text(os.path.join(thp, "defrag"))),
        "shm_total_bytes": shm_total,
        "shm_free_bytes": shm_free,
    }


def collect_limits(session: Session) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, which in (
        ("memlock", resource.RLIMIT_MEMLOCK),
        ("nofile", resource.RLIMIT_NOFILE),
        ("nproc", resource.RLIMIT_NPROC),
    ):
        soft = hard = None
        try:
            soft, hard = session.host.getrlimit(which)
        except (ValueError, OSError) as exc:
            session.note(f"getrlimit:{key}", str(exc))
        out[f"{key}_soft"] = _rlimit(soft)
        out[f"{key}_hard"] = _rlimit(hard)
    return out


def _rlimit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return -1 if value == resource.RLIM_INFINITY or value < 0 else int(value)


def collect_cgroup(session: Session) -> Dict[str, Any]:
    host = session.host
    root = host.path("sys/fs/cgroup")
    proc_cgroup = read_text(host.path("proc/self/cgroup"))
    path = None
    for line in (proc_cgroup or "").splitlines():
        if line.startswith("0::"):
            path = line[3:].strip() or None
    out: Dict[str, Any] = {
        "version": None,
        "path": path,
        "cpu_quota_us": None,
        "cpu_period_us": None,
        "cpuset_cpus": None,
        "cpuset_count": None,
        "pids_max": None,
        "pids_current": None,
        "cpu_nr_throttled": None,
        "cpu_throttled_us": None,
        "memory_events": {},
    }
    if os.path.exists(os.path.join(root, "cgroup.controllers")):
        out["version"] = 2
        quota, period = parse_cgroup_cpu_max(read_text(os.path.join(root, "cpu.max")))
        cpuset = read_stripped(os.path.join(root, "cpuset.cpus.effective"))
        pids_max = read_text(os.path.join(root, "pids.max"))
        pids_current = read_stripped(os.path.join(root, "pids.current"))
        stat = parse_int_map(read_text(os.path.join(root, "cpu.stat")))
        throttled = stat.get("throttled_usec")
        events = parse_int_map(read_text(os.path.join(root, "memory.events")))
    elif os.path.isdir(os.path.join(root, "cpu")) or os.path.isdir(
        os.path.join(root, "cpu,cpuacct")
    ):
        out["version"] = 1
        cpu_dir = os.path.join(root, "cpu")
        if not os.path.isdir(cpu_dir):
            cpu_dir = os.path.join(root, "cpu,cpuacct")
        raw_quota = parse_strict_int(
            read_stripped(os.path.join(cpu_dir, "cpu.cfs_quota_us"))
        )
        quota = raw_quota if raw_quota is not None and raw_quota > 0 else None
        period = parse_strict_int(
            read_stripped(os.path.join(cpu_dir, "cpu.cfs_period_us"))
        )
        cpuset = read_stripped(os.path.join(root, "cpuset", "cpuset.effective_cpus"))
        cpuset = cpuset or read_stripped(os.path.join(root, "cpuset", "cpuset.cpus"))
        pids_max = read_text(os.path.join(root, "pids", "pids.max"))
        pids_current = read_stripped(os.path.join(root, "pids", "pids.current"))
        stat = parse_int_map(read_text(os.path.join(cpu_dir, "cpu.stat")))
        throttled_ns = stat.get("throttled_time")
        throttled = throttled_ns // 1000 if throttled_ns is not None else None
        events = {}
        oom = parse_kv_lines(
            read_text(os.path.join(root, "memory", "memory.oom_control"))
        )
        if parse_strict_int(oom.get("oom_kill")) is not None:
            events["oom_kill"] = int(oom["oom_kill"])
    else:
        session.note("cgroup", "no cgroup v1 or v2 hierarchy at /sys/fs/cgroup")
        return out
    out.update(
        {
            "cpu_quota_us": quota,
            "cpu_period_us": period,
            "cpuset_cpus": cpuset,
            "cpuset_count": len(parse_cpulist(cpuset)) if cpuset else None,
            "pids_max": parse_pids_max(pids_max),
            "pids_current": parse_strict_int(pids_current),
            "cpu_nr_throttled": stat.get("nr_throttled"),
            "cpu_throttled_us": throttled,
            "memory_events": events,
        }
    )
    return out


def collect_kernel(session: Session, bdfs: Sequence[str]) -> Dict[str, Any]:
    host = session.host
    try:
        release = host.kernel_release()
    except OSError as exc:
        release = None
        session.note("uname", str(exc))
    uptime_text = read_stripped(host.path("proc/uptime"))
    uptime = None
    if uptime_text:
        try:
            uptime = float(uptime_text.split()[0])
        except ValueError:
            uptime = None
    groups_dir = host.path("sys/kernel/iommu_groups")
    groups = None
    if os.path.isdir(groups_dir):
        groups = sum(1 for name in _listdir(groups_dir) if name.isdigit())
    types: Dict[str, str] = {}
    for bdf in bdfs:
        group = host.path("sys/bus/pci/devices", bdf, "iommu_group")
        kind = read_stripped(os.path.join(group, "type"))
        if kind:
            types[bdf] = kind
    return {
        "release": release,
        "uptime_s": uptime,
        "cmdline_flags": parse_cmdline(read_text(host.path("proc/cmdline"))),
        "iommu_groups": groups,
        "iommu_devices": sorted(
            _listdir(host.path("sys/class/iommu")), key=natural_key
        ),
        "iommu_group_types": types,
    }


def collect_modules(session: Session) -> Dict[str, bool]:
    host = session.host
    loaded = set(parse_proc_modules(read_text(host.path("proc/modules"))))
    return {
        name: name in loaded or os.path.isdir(host.path("sys/module", name))
        for name in MODULES
    }


def collect_cpu(session: Session) -> Dict[str, Any]:
    host = session.host
    base = host.path("sys/devices/system/cpu")
    online = read_stripped(os.path.join(base, "online"))
    governors: Dict[str, int] = {}
    cur: List[int] = []
    max_khz: Optional[int] = None
    driver = None
    for name in sorted(_listdir(base), key=natural_key):
        if not re.match(r"^cpu\d+$", name):
            continue
        freq = os.path.join(base, name, "cpufreq")
        governor = read_stripped(os.path.join(freq, "scaling_governor"))
        if governor:
            governors[governor] = governors.get(governor, 0) + 1
        value = parse_strict_int(read_stripped(os.path.join(freq, "scaling_cur_freq")))
        if value is not None:
            cur.append(value)
        top = parse_strict_int(read_stripped(os.path.join(freq, "cpuinfo_max_freq")))
        if top is not None:
            max_khz = max(max_khz or 0, top)
        driver = driver or read_stripped(os.path.join(freq, "scaling_driver"))
    if not governors:
        session.note("sysfs:cpufreq", "no cpufreq governors (VM or container mask)")
    boost = read_stripped(os.path.join(base, "cpufreq", "boost"))
    if boost is None:
        no_turbo = read_stripped(os.path.join(base, "intel_pstate", "no_turbo"))
        if no_turbo is not None:
            boost = "0" if no_turbo == "1" else "1"
    smt_active = read_stripped(os.path.join(base, "smt", "active"))
    return {
        "online": online,
        "online_count": len(parse_cpulist(online)) if online else None,
        "governors": governors,
        "scaling_driver": driver,
        "cur_khz_min": min(cur) if cur else None,
        "cur_khz_max": max(cur) if cur else None,
        "cur_khz_mean": int(sum(cur) / len(cur)) if cur else None,
        "max_khz": max_khz,
        "boost": boost,
        "smt_control": read_stripped(os.path.join(base, "smt", "control")),
        "smt_active": None if smt_active is None else smt_active == "1",
    }


def collect_driver(session: Session, smi: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    host = session.host
    parsed = parse_nvidia_version(read_text(host.path("proc/driver/nvidia/version")))
    version = parsed["version"] or (smi or {}).get("driver_version")
    if version is None:
        version = read_stripped(host.path("sys/module/nvidia/version"))
    if parsed["nvrm"] is None:
        session.note("/proc/driver/nvidia/version", "unreadable")
    return {
        "version": version,
        "flavor": parsed["flavor"],
        "nvrm": parsed["nvrm"],
        "cuda_version": (smi or {}).get("cuda_version"),
    }


def collect_smi(session: Session) -> Optional[Dict[str, Any]]:
    text = session.tool(
        ["nvidia-smi", "-q", "-x"], SMI_XML_TIMEOUT_S, "nvidia-smi -q -x"
    )
    if text is None:
        return None
    try:
        return parse_smi_xml(text)
    except ValueError as exc:
        session.note("nvidia-smi -q -x", str(exc))
        return None


def collect_nvlink(
    session: Session, gpus: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    text = session.tool(["nvidia-smi", "nvlink", "-s"], SMI_TIMEOUT_S)
    entries = parse_nvlink_status(text)
    by_uuid = {g["uuid"]: g["bus_id"] for g in gpus if g.get("uuid")}
    by_index = {g["index"]: g["bus_id"] for g in gpus}
    for entry in entries:
        entry["bus_id"] = by_uuid.get(entry["uuid"]) or by_index.get(entry["index"])
    if not any(link["speed_gbs"] for e in entries for link in e["links"]):
        return entries  # no active NVLink: nothing to count errors on
    errors = parse_nvlink_errors(
        session.tool(["nvidia-smi", "nvlink", "-e"], SMI_TIMEOUT_S)
    )
    for entry in entries:
        per_link = errors.get(entry["index"], {})
        known = {link["link"] for link in entry["links"]}
        for link in entry["links"]:
            link["errors"] = per_link.get(link["link"], {})
        for number in sorted(set(per_link) - known):
            entry["links"].append(
                {"link": number, "speed_gbs": None, "errors": per_link[number]}
            )
    return entries


def collect_p2p(
    session: Session, gpu_count: int
) -> Dict[str, Dict[str, Dict[str, str]]]:
    if gpu_count < 2:
        return {}
    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    for flag, name in (
        ("r", "read"),
        ("w", "write"),
        ("n", "nvlink"),
        ("a", "atomics"),
    ):
        text = session.tool(["nvidia-smi", "topo", "-p2p", flag], TOPO_TIMEOUT_S)
        matrix = parse_p2p_matrix(text)
        if matrix:
            result[name] = matrix
    return result


def _int_dir(path: str, skip: Sequence[str] = ()) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for name in sorted(_listdir(path)):
        if name in skip:
            continue
        value = parse_strict_int(read_stripped(os.path.join(path, name)))
        if value is not None:
            values[name] = value
    return values


def collect_hcas(session: Session) -> List[Dict[str, Any]]:
    host = session.host
    ib_root = host.path("sys/class/infiniband")
    hcas = []
    for name in sorted(_listdir(ib_root), key=natural_key):
        dev_dir = os.path.join(ib_root, name)
        bdf = _realpath_basename(os.path.join(dev_dir, "device"))
        if bdf is None or not _SYSFS_BDF_RE.match(bdf):
            continue  # soft-RoCE (rxe) and friends have no PCI function
        device_netdevs = sorted(
            _listdir(os.path.join(dev_dir, "device", "net")), key=natural_key
        )
        ports = []
        for port in sorted(_listdir(os.path.join(dev_dir, "ports")), key=natural_key):
            if not port.isdigit():
                continue
            port_dir = os.path.join(dev_dir, "ports", port)
            layer = read_stripped(os.path.join(port_dir, "link_layer"))
            netdev = read_stripped(os.path.join(port_dir, "gid_attrs", "ndevs", "0"))
            netdev = netdev or (device_netdevs[0] if device_netdevs else None)
            counters = _int_dir(os.path.join(port_dir, "counters"))
            if not counters:
                session.note(f"sysfs:{name}/{port}/counters", "no readable counters")
            ethtool: Dict[str, int] = {}
            mtu = None
            if netdev:
                mtu_text = read_stripped(host.path("sys/class/net", netdev, "mtu"))
                mtu = parse_strict_int(mtu_text)
            if layer == "Ethernet" and netdev:
                ethtool = filter_ethtool(
                    parse_ethtool_stats(
                        session.tool(["ethtool", "-S", netdev], SHORT_TIMEOUT_S)
                    )
                )
            ports.append(
                {
                    "port": int(port),
                    "link_layer": layer,
                    "state": read_stripped(os.path.join(port_dir, "state")),
                    "phys_state": read_stripped(os.path.join(port_dir, "phys_state")),
                    "rate": read_stripped(os.path.join(port_dir, "rate")),
                    "netdev": netdev,
                    "mtu": mtu,
                    "counters": counters,
                    # "lifespan" is a cache setting (ms), not a counter.
                    "hw_counters": _int_dir(
                        os.path.join(port_dir, "hw_counters"), skip=("lifespan",)
                    ),
                    "ethtool": ethtool,
                }
            )
        hcas.append(
            {
                "name": name,
                "bdf": bdf,
                "fw_ver": read_stripped(os.path.join(dev_dir, "fw_ver")),
                "board_id": read_stripped(os.path.join(dev_dir, "board_id")),
                "ports": ports,
            }
        )
    if not hcas:
        session.note("sysfs:infiniband", "no RDMA devices under /sys/class/infiniband")
    return hcas


def collect_time_sync(session: Session) -> Dict[str, Any]:
    text = session.tool(["chronyc", "-c", "tracking"], 5.0)
    parsed = parse_chronyc_tracking(text) if text else None
    if parsed is None:
        parsed = parse_timedatectl(session.tool(["timedatectl", "show"], 5.0))
    if parsed is None:
        raw = session.host.adjtimex()
        if raw is not None:
            parsed = interpret_adjtimex(*raw)
        else:
            session.note("adjtimex", "unavailable")
    return parsed or {
        "source": None,
        "synchronized": None,
        "offset_s": None,
        "stratum": None,
        "reference": None,
        "leap_status": None,
        "max_error_s": None,
        "est_error_s": None,
        "root_dispersion_s": None,
    }


def collect_docker(session: Session) -> Optional[Dict[str, Any]]:
    res = session.result(
        ["docker", "info", "--format", "{{json .}}"], DOCKER_TIMEOUT_S, "docker info"
    )
    parsed = parse_docker_info(res.stdout) if res is not None else None
    if res is not None and parsed is None:
        session.note("docker info", res.error() if not res.ok else "unparsable JSON")
    if parsed is None:
        return None
    errors = parsed.pop("server_errors")
    if errors:
        session.note("docker info", "; ".join(errors))
    parsed["nvidia_container_runtime"] = session.host.which("nvidia-container-runtime")
    parsed["nvidia_ctk"] = session.host.which("nvidia-ctk")
    return parsed


def collect_kernel_log(
    session: Session, gpu_bdfs: Sequence[str], our_bdfs: Sequence[str] = ()
) -> Dict[str, Any]:
    """Xid / mlx5 / AER events from ``dmesg``, else ``/dev/kmsg`` (if allowed)."""
    gpu_pcis = {bdf.rsplit(".", 1)[0]: bdf for bdf in gpu_bdfs}
    levels = "emerg,alert,crit,err,warn"
    res = session.result(["dmesg", f"--level={levels}"], SHORT_TIMEOUT_S, "dmesg")
    errors: List[str] = []
    if res is not None and res.ok:
        entries = [parse_dmesg_line(line) for line in res.stdout.splitlines() if line]
        scanned = scan_kernel_log(entries, gpu_pcis, our_bdfs)
        return {"available": True, "source": "dmesg", "error": None, **scanned}
    if res is not None:
        errors.append(f"dmesg {res.error()}")
        session.note("dmesg", res.error())
    lines, error = read_kmsg(session.host.path("dev/kmsg"))
    if lines is not None:
        records = [parse_kmsg_line(line) for line in lines]
        entries = [(r[1], r[2]) for r in records if r is not None and r[0] <= 4]
        scanned = scan_kernel_log(entries, gpu_pcis, our_bdfs)
        # Xid lines are kept whatever their level: re-scan the other levels.
        extra = [
            (r[1], r[2])
            for r in records
            if r is not None and r[0] > 4 and "Xid" in r[2]
        ]
        scanned["events"] += scan_kernel_log(extra, gpu_pcis, our_bdfs)["events"]
        scanned["events"].sort(key=lambda e: e["t_s"] or 0.0)
        return {"available": True, "source": "kmsg", "error": None, **scanned}
    errors.append(f"/dev/kmsg {error}")
    return {
        "available": False,
        "source": None,
        "error": "; ".join(errors),
        "lines": 0,
        "events": [],
        "samples": [],
    }


def collect_host(
    host: Host, node: str = "", deadline_s: float = DEFAULT_DEADLINE_S
) -> Dict[str, Any]:
    """The ``node_facts`` document (see the module docstring)."""
    session = Session(host, deadline_s)
    guard = session.guarded
    started = host.clock()
    pcie = guard("pcie", lambda: collect_pcie(session), [])
    smi = guard("gpus", lambda: collect_smi(session), None)
    gpus = (smi or {}).get("gpus") or []
    gpu_bdfs = [g["bus_id"] for g in gpus if g.get("bus_id")] or [
        r["bdf"] for r in pcie if r["kind"] == "gpu"
    ]
    endpoint_bdfs = [r["bdf"] for r in pcie if r["depth"] == 0]
    our_bdfs = [r["bdf"] for r in pcie]
    payload: Dict[str, Any] = {
        "kind": KIND_HOST,
        "version": VERSION,
        "node": node,
        "hostname": host.hostname(),
        "started_at": started,
        "collected_at": None,
        "duration_s": None,
        "kernel": guard("kernel", lambda: collect_kernel(session, endpoint_bdfs), {}),
        "driver": guard("driver", lambda: collect_driver(session, smi), {}),
        "modules": guard("modules", lambda: collect_modules(session), {}),
        "cpu": guard("cpu", lambda: collect_cpu(session), {}),
        "numa": guard("numa", lambda: collect_numa(session), {}),
        "memory": guard("memory", lambda: collect_memory(session), {}),
        "limits": guard("limits", lambda: collect_limits(session), {}),
        "cgroup": guard("cgroup", lambda: collect_cgroup(session), {}),
        "pcie": pcie,
        "gpus": gpus,
        "nvlink": guard("nvlink", lambda: collect_nvlink(session, gpus), [])
        if gpus
        else [],
        "p2p": guard("p2p", lambda: collect_p2p(session, len(gpus)), {}),
        "hcas": guard("hcas", lambda: collect_hcas(session), []),
        "time_sync": guard("time_sync", lambda: collect_time_sync(session), {}),
        "docker": guard("docker", lambda: collect_docker(session), None),
        "kernel_log": guard(
            "kernel_log",
            lambda: collect_kernel_log(session, gpu_bdfs, our_bdfs),
            {},
        ),
    }
    finished = host.clock()
    payload["collected_at"] = finished
    payload["duration_s"] = round(max(0.0, finished - started), 3)
    payload["notes"] = session.notes
    return payload


# --------------------------------------------------------------------------
# Container (software stack) collector
# --------------------------------------------------------------------------


def _find_tool(host: Host, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if os.sep in candidate:
            if os.access(host.path(candidate), os.X_OK):
                return candidate
        else:
            found = host.which(candidate)
            if found:
                return found
    return None


def find_nccl_files(host: Host, dirs: Iterable[str]) -> Tuple[List[str], List[str]]:
    """(libnccl.so* paths, NCCL plugin paths) found directly in ``dirs``.

    Paths are reported as on the host (a test root is stripped); symlinked
    library names are kept, not resolved.
    """
    libs: List[str] = []
    plugins: List[str] = []
    seen = set()
    for directory in dirs:
        real_dir = os.path.realpath(directory) if directory else ""
        if not real_dir or real_dir in seen:
            continue
        seen.add(real_dir)
        shown = host.display_path(real_dir)
        for name in sorted(_listdir(real_dir)):
            full = os.path.join(shown, name)
            if any(fnmatch.fnmatch(name, p) for p in NCCL_PLUGIN_PATTERNS):
                plugins.append(full)
            elif any(fnmatch.fnmatch(name, p) for p in NCCL_LIB_PATTERNS):
                libs.append(full)
    return libs, plugins


def filter_env(environ: Mapping[str, str]) -> Dict[str, str]:
    """Allowlisted, non-secret environment of the image (NCCL_*, UCX_*, ...)."""
    out: Dict[str, str] = {}
    for key, value in sorted(environ.items()):
        upper = key.upper()
        if any(marker in upper for marker in ENV_SECRET_MARKERS):
            continue
        if upper.startswith(ENV_PREFIXES):
            out[key] = value
    return out


def collect_software(host: Host, node: str = "") -> Dict[str, Any]:
    """The ``software_stack`` document (see the module docstring)."""
    session = Session(host, deadline_s=TORCH_TIMEOUT_S + 120.0)
    packages, missing = session.guarded(
        "packages", lambda: select_packages(host.distributions()), ({}, [])
    )
    torch_res = session.result(
        [host.python, "-c", _TORCH_SNIPPET], TORCH_TIMEOUT_S, "torch probe"
    )
    torch_info = parse_torch_probe(torch_res.stdout) if torch_res is not None else None
    if torch_res is not None and torch_info is None:
        session.note("torch probe", torch_res.error())
    ucx = None
    ucx_tool = _find_tool(host, UCX_INFO_PATHS)
    if ucx_tool:
        ucx = parse_ucx_info(
            session.tool([ucx_tool, "-v"], SHORT_TIMEOUT_S, "ucx_info -v")
        )
    else:
        session.note("ucx_info", "not installed")
    nvcc = None
    nvcc_tool = _find_tool(host, NVCC_PATHS)
    if nvcc_tool:
        nvcc = parse_nvcc_version(
            session.tool([nvcc_tool, "--version"], SHORT_TIMEOUT_S, "nvcc --version")
        )
    ibv = parse_ibv_devinfo_list(session.tool(["ibv_devinfo", "-l"], SHORT_TIMEOUT_S))
    toolkit = None
    version_json = read_text(host.path("usr/local/cuda/version.json"))
    if version_json:
        try:
            toolkit = (json.loads(version_json).get("cuda") or {}).get("version")
        except (ValueError, AttributeError):
            session.note("/usr/local/cuda/version.json", "unparsable")
    lib_dirs = [p for p in host.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    lib_dirs += [host.path(d) for d in NCCL_LIB_DIRS]
    lib_dirs += [
        os.path.join(p, "nvidia", "nccl", "lib") for p in host.search_path if p
    ]
    libs, plugins = find_nccl_files(host, lib_dirs)
    if torch_info is not None:
        torch_info.setdefault("errors", [])
    return {
        "kind": KIND_SOFTWARE,
        "version": VERSION,
        "node": node,
        "hostname": host.hostname(),
        "collected_at": host.clock(),
        "python": platform.python_version(),
        "packages": packages,
        "missing_packages": missing,
        "torch": torch_info,
        "ucx": ucx,
        "nvcc": nvcc,
        "cuda_toolkit": toolkit,
        "ibv_devices": ibv,
        "os_release": parse_os_release(read_text(host.path("etc/os-release"))),
        "glibc": host.libc_version(),
        "nccl_libraries": libs,
        "nccl_plugins": plugins,
        "env": filter_env(host.environ),
        "notes": session.notes,
    }


# --------------------------------------------------------------------------
# Disk read speed
# --------------------------------------------------------------------------


def resolve_read_target(
    path: str,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """(file, size, error): ``path`` itself or its largest ``*.safetensors``."""
    if os.path.isfile(path):
        return path, os.path.getsize(path), None
    if not os.path.isdir(path):
        return None, None, f"no such file or directory: {path}"
    best: Optional[Tuple[int, str]] = None
    top_depth = path.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(path):
        if dirpath.count(os.sep) - top_depth >= 4:
            dirnames[:] = []
        for name in filenames:
            if not name.endswith(".safetensors"):
                continue
            full = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if best is None or size > best[0] or (size == best[0] and full < best[1]):
                best = (size, full)
    if best is None:
        return None, None, f"no *.safetensors under {path}"
    return best[1], best[0], None


def read_file_timed(
    path: str, limit: int, block: int, flags: int, max_seconds: float
) -> Tuple[int, float]:
    """Sequentially read up to ``limit`` bytes: (bytes, seconds).

    The anonymous mmap buffer is page aligned, which O_DIRECT requires.
    """
    fd = os.open(path, flags)
    try:
        with mmap.mmap(-1, block) as buf:
            total = 0
            start = time.perf_counter()
            deadline = start + max_seconds
            while total < limit:
                count = os.readv(fd, [buf])
                if count <= 0:
                    break
                total += count
                if time.perf_counter() > deadline:
                    break
            elapsed = time.perf_counter() - start
    finally:
        os.close(fd)
    return total, elapsed


def _flush_file(path: str) -> bool:
    """fsync ``path``: dirty pages of a just-written shard would otherwise be
    written back inside the timed direct read."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(fd)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _drop_cache(path: str) -> bool:
    """Best effort: evict the file's clean pages so a buffered read hits disk."""
    if _FADVISE is None or _FADV_DONTNEED is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        _FADVISE(fd, 0, 0, _FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def disk_read(
    host: Host,
    path: str,
    *,
    max_bytes: int = DEFAULT_DISK_MAX_BYTES,
    block_mib: int = DEFAULT_BLOCK_MIB,
    max_seconds: float = DEFAULT_DISK_MAX_SECONDS,
    node: str = "",
) -> Dict[str, Any]:
    """The ``disk_read`` document: dd iflag=direct, else O_DIRECT, else buffered."""
    session = Session(host, deadline_s=max_seconds * 3 + 60.0)
    block = max(1, block_mib) * MIB
    out: Dict[str, Any] = {
        "kind": KIND_DISK,
        "version": VERSION,
        "node": node,
        "hostname": host.hostname(),
        "requested_path": path,
        "path": None,
        "file_bytes": None,
        "bytes": 0,
        "seconds": None,
        "gbs": None,
        "method": None,
        "direct": False,
        "cache": "unknown",
        "flushed": False,
        "block_bytes": block,
        "error": None,
        "notes": session.notes,
    }
    target, size, error = resolve_read_target(path)
    if target is None:
        out["error"] = error
        return out
    out["path"] = target
    out["file_bytes"] = size
    out["flushed"] = _flush_file(target)
    want = min(max_bytes, size) if size else max_bytes
    measured: Optional[Tuple[int, float, str, bool, str]] = None
    if host.which("dd"):
        count = max(1, int(math.ceil(want / block)))
        argv = ["env", "LC_ALL=C", "dd", f"if={target}", "of=/dev/null"]
        argv += [f"bs={block}", f"count={count}", "iflag=direct"]
        res = session.result(argv, max_seconds + 30.0, "dd iflag=direct")
        if res is not None and res.ok:
            stats = parse_dd_output(res.stderr + "\n" + res.stdout)
            if stats is not None and stats[0] > 0 and stats[1] > 0:
                measured = (stats[0], stats[1], "dd_direct", True, "bypassed")
            else:
                session.note("dd iflag=direct", "no statistics in dd output")
        elif res is not None:
            session.note("dd iflag=direct", res.error())
    else:
        session.note("dd", "not installed")
    if measured is None and _O_DIRECT is not None:
        try:
            got, secs = read_file_timed(
                target, want, block, os.O_RDONLY | _O_DIRECT, max_seconds
            )
            if got > 0 and secs > 0:
                measured = (got, secs, "o_direct", True, "bypassed")
        except OSError as exc:
            session.note("O_DIRECT read", _os_error(exc))
    if measured is None:
        dropped = _drop_cache(target)
        try:
            got, secs = read_file_timed(target, want, block, os.O_RDONLY, max_seconds)
        except OSError as exc:
            out["error"] = f"read failed: {_os_error(exc)}"
            return out
        measured = (got, secs, "buffered", False, "dropped" if dropped else "unknown")
    got, secs, method, direct, cache = measured
    out.update(
        {
            "bytes": got,
            "seconds": round(secs, 6),
            "gbs": round(got / secs / 1e9, 4) if secs > 0 else None,
            "method": method,
            "direct": direct,
            "cache": cache,
        }
    )
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def write_document(payload: Mapping[str, Any], out: Optional[str]) -> None:
    """Print one JSON line, or write it atomically to ``out``."""
    text = json.dumps(payload, sort_keys=False, default=str) + "\n"
    if not out:
        sys.stdout.write(text)
        sys.stdout.flush()
        return
    directory = os.path.dirname(os.path.abspath(out))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".node_facts.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, out)
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="mode")
    for mode in ("host", "container", "disk-read"):
        child = sub.add_parser(mode)
        child.add_argument("--node", default="", help="logical node name to record")
        child.add_argument("--out", default=None, help="write JSON here, not stdout")
        if mode == "host":
            child.add_argument("--deadline-s", type=float, default=DEFAULT_DEADLINE_S)
        if mode == "disk-read":
            child.add_argument("--path", required=True)
            child.add_argument("--max-bytes", type=int, default=DEFAULT_DISK_MAX_BYTES)
            child.add_argument("--block-mib", type=int, default=DEFAULT_BLOCK_MIB)
            child.add_argument(
                "--max-seconds", type=float, default=DEFAULT_DISK_MAX_SECONDS
            )
    return parser


def main(argv: Optional[Sequence[str]] = None, host: Optional[Host] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    implicit_host = not args_list or (
        args_list[0].startswith("-") and args_list[0] not in ("-h", "--help")
    )
    if implicit_host:
        args_list.insert(0, "host")
    args = build_parser().parse_args(args_list)
    host = host or Host()
    if args.mode == "container":
        payload = collect_software(host, args.node)
    elif args.mode == "disk-read":
        payload = disk_read(
            host,
            args.path,
            max_bytes=args.max_bytes,
            block_mib=args.block_mib,
            max_seconds=args.max_seconds,
            node=args.node,
        )
    else:
        payload = collect_host(host, args.node, args.deadline_s)
    write_document(payload, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
