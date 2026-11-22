# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Under-load telemetry sampler for one cluster node.

Run as::

    <python> telemetry_agent.py --out PATH --node NAME [--interval 1.0]
        [--pcie-interval 0.1] [--slow-interval 5.0] [--duration S]
        [--control-interface wg0] [--tag T] [--no-gpm]

Writes JSON lines to ``--out`` (truncated at start), flushed once per
``--interval``. SIGTERM or SIGINT takes one final sample, flushes and exits 0.

Standalone: stdlib plus the optional ``pynvml`` module (nvidia-ml-py), Python
>= 3.8 syntax, no ``benchmarks.*`` imports. Without pynvml, or when NVML fails
to initialise, GPU metrics come from ``nvidia-smi`` instead (no GPM).

Line kinds; the field names are a contract with
``benchmarks/cluster_bench/adapters/telemetry.py``:

* ``meta``   first line: node, hostname, ib_unit_bytes, nvml, nvlink_source,
  gpus [{index, pci_bus_id, name}], hcas [{name, port}], plus interval,
  pcie_interval, pcie_source, control_interface, tag and version. Stress
  keys: slow_interval, slow_every and gpux_every (ticks; 0 = off),
  gpux_source ("nvml" | "nvidia-smi" | null), link_layers [{name, port,
  link_layer}], ibx_ports [{name, port, counters}], ethx_ifaces [{name,
  port, iface}], notes [{source, error, ...}]. GPM keys: gpm {"<gpu>":
  "ok" | "unsupported" | "<NVML error name>" | "no-pynvml" | "no-gpm-api" |
  "disabled"} and gpm_metrics (the gpm field names requested).
* ``ib``     per active HCA port and interval: cumulative tx/rx BYTES
  (port_xmit_data / port_rcv_data x 4). Never unwrapped here.
* ``pcie``   per GPU and interval: mean and max tx/rx bytes/s over the
  sub-samples taken every --pcie-interval, and their count ``n``.
* ``nvlink`` per GPU: cumulative tx/rx bytes summed over its links.
* ``gpu``    per GPU: util (%), mem_used_mib, power_w.
* ``cpu``    host CPU utilisation 0..1 from /proc/stat deltas.
* ``net``    control interface cumulative tx/rx bytes.
* ``gpux``   per GPU: sm_mhz, mem_mhz, temp_c, power_w, energy_mj (NVML
  total energy since driver load, cumulative) and reasons (NVML
  clocks-event bitmask). Every tick with NVML; every slow tick with the
  nvidia-smi fallback (no energy there).
* ``ibx``    per active HCA port, every slow tick: ``c`` = raw cumulative
  congestion/error counters that exist under ports/<p>/counters and
  ports/<p>/hw_counters (IBX_PORT_COUNTERS, IBX_HW_COUNTERS).
* ``ethx``   per Ethernet link-layer (RoCE) HCA netdev, every slow tick:
  iface, dev and ``c`` = ``ethtool -S`` pause/PFC/discard/CNP/ECN counters.
* ``gpm``    every tick with NVML GPM (Hopper and newer): ``g`` = {"<gpu>":
  {dt_s, gr_pct, sm_pct, occ_pct, tensor_pct, dram_pct, pcie_tx_mibs,
  pcie_rx_mibs, nvlink_tx_mibs, nvlink_rx_mibs}}, each value the average
  over the dt_s seconds before ``t`` (percent 0..100, MiB/s); a field is
  absent when NVML did not return it. A GPU is skipped on a tick less than
  0.1 s after its previous sample.
* ``note``   a stress source that failed after start (logged once):
  source, error and the device or iface concerned. GPM notes: source
  "gpm", gpu, error and, when only some metrics failed (they are dropped,
  the GPU keeps sampling the rest), ``metrics``; without ``metrics`` GPM is
  disabled on that GPU.

A slow tick is every ``slow_every`` = round(slow_interval / interval) ticks,
the first tick and the final sample; ``--slow-interval 0`` disables the
gpux, ibx and ethx kinds; ``--no-gpm`` disables the gpm kind.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

try:  # nvidia-ml-py is optional: without it GPU metrics come from nvidia-smi.
    import pynvml
except ImportError:
    pynvml = None

logger = logging.getLogger("telemetry_agent")

VERSION = 1
IB_UNIT_BYTES = 4  # port_xmit_data / port_rcv_data count 4-byte words
# UNVERIFIED: NVML documents nvmlDeviceGetPcieThroughput in "KB/s"; this
# assumes KB = 1024 bytes (as NVIDIA's own samples treat it).
NVML_PCIE_KB_BYTES = 1024
# UNVERIFIED: `nvidia-smi dmon -s t` prints rxpci/txpci in "MB/s"; assumed MiB.
DMON_MB_BYTES = 1024 * 1024
# (tx constant, rx constant, bytes per unit), resolved with getattr on the
# pynvml module because older nvidia-ml-py releases lack the newer names. The
# NVLink5 byte counters go first: Blackwell drivers may stop serving the
# legacy throughput fields. UNVERIFIED: THROUGHPUT_DATA_* are documented in
# KiB and COUNT_*_BYTES in bytes.
NVLINK_FIELD_CANDIDATES = (
    ("NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES", "NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES", 1),
    (
        "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX",
        "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX",
        1024,
    ),
)
NVML_SUCCESS = 0
NVML_SCOPE_ALL_LINKS = 0xFFFFFFFF  # scopeId that sums every link (NVML docs)
DEFAULT_MAX_NVLINKS = 18
# nvmlValueType_t -> c_nvmlValue_t union member.
NVML_VALUE_ATTRS = {
    0: "dVal",
    1: "uiVal",
    2: "ulVal",
    3: "ullVal",
    4: "sllVal",
    5: "siVal",
    6: "usVal",
}
STOP_POLL_S = 0.05
TOOL_TIMEOUT_S = 30.0
DMON_ARGV = ("nvidia-smi", "dmon", "-s", "t", "-d", "1")
DEFAULT_SLOW_INTERVAL = 5.0
ETHTOOL_TIMEOUT_S = 5.0
# Congestion and error counters under /sys/class/infiniband/<dev>/ports/<p>/.
IBX_PORT_COUNTERS = (  # counters/
    "port_xmit_wait",
    "port_xmit_discards",
    "port_rcv_errors",
    "symbol_error",
    "link_downed",
    "local_link_integrity_errors",
    "excessive_buffer_overrun_errors",
    "port_rcv_remote_physical_errors",
    "VL15_dropped",
)
IBX_HW_COUNTERS = (  # hw_counters/ (mlx5)
    "np_cnp_sent",
    "rp_cnp_handled",
    "np_ecn_marked_roce_packets",
    "out_of_buffer",
    "out_of_sequence",
    "packet_seq_err",
    "local_ack_timeout_err",
    "implied_nak_seq_err",
    "rnr_nak_retry_err",
    "req_cqe_error",
    "resp_cqe_error",
)
# ``ethtool -S`` counters kept for RoCE netdevs; per-queue counters
# ("rx3_...", "ch0_...") are dropped to keep records small.
ETHX_KEEP_RE = re.compile(
    r"pause|pfc|discard|drop|cnp|ecn|marked|out_of_buffer|storm", re.IGNORECASE
)
ETHX_PER_QUEUE_RE = re.compile(r"^(?:ch|rx|tx|xdpsq|xsk|ptp|qos)\d+_")
ETHX_MAX_COUNTERS = 96
# nvidia-smi fallback for gpux: base fields, then the clocks-event reasons
# field under its current and pre-R535 names (probed in that order).
SMI_GPUX_FIELDS = ("index", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw")
SMI_REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
NVML_CLOCK_SM_DEFAULT = 1
NVML_CLOCK_MEM_DEFAULT = 2
NVML_TEMPERATURE_GPU_DEFAULT = 0
# Newer pynvml first; the throttle name is the pre-R535 spelling.
NVML_REASON_FUNCS = (
    "nvmlDeviceGetCurrentClocksEventReasons",
    "nvmlDeviceGetCurrentClocksThrottleReasons",
)
# NVML GPM metrics sampled every tick: (gpm field, pynvml constant). Names an
# older nvidia-ml-py lacks are skipped. Per nvidia-ml-py 13.615.71 and
# 12.535.161 (pynvml.py comments): utilisations are percent 0.0-100.0, the
# *_PER_SEC rates MiB/s; PCIe TX is traffic from the GPU, NVLink TOTAL_RX /
# TOTAL_TX are ids 60 / 61, summed over all links.
GPM_METRICS = (
    ("gr_pct", "NVML_GPM_METRIC_GRAPHICS_UTIL"),
    ("sm_pct", "NVML_GPM_METRIC_SM_UTIL"),
    ("occ_pct", "NVML_GPM_METRIC_SM_OCCUPANCY"),
    ("tensor_pct", "NVML_GPM_METRIC_ANY_TENSOR_UTIL"),
    ("dram_pct", "NVML_GPM_METRIC_DRAM_BW_UTIL"),
    ("pcie_tx_mibs", "NVML_GPM_METRIC_PCIE_TX_PER_SEC"),
    ("pcie_rx_mibs", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
    ("nvlink_tx_mibs", "NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC"),
    ("nvlink_rx_mibs", "NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC"),
)
# pynvml names the GPM flow uses; without them the status is "no-gpm-api".
GPM_API = (
    "nvmlGpmQueryDeviceSupport",
    "nvmlGpmSampleAlloc",
    "nvmlGpmSampleGet",
    "nvmlGpmSampleFree",
    "nvmlGpmMetricsGet",
    "c_nvmlGpmMetricsGet_t",
)
GPM_METRICS_GET_VERSION = 1  # NVML_GPM_METRICS_GET_VERSION in nvml.h
GPM_MIN_INTERVAL_S = 0.1  # NVML: two GPM samples must be > 100 ms apart
GPM_PCT_DIGITS = 2
GPM_RATE_DIGITS = 1
# Binding mismatches (a struct field or constant an nvidia-ml-py release
# lacks) must cost GPM on that GPU only, never the sampler.
GPM_BINDING_ERRORS = (AttributeError, TypeError, ValueError)

_NVLINK_GPU_RE = re.compile(r"^\s*GPU\s+(\d+)\s*:")
_NVLINK_DATA_RE = re.compile(
    r"Link\s+\d+\s*:\s*Data\s+(Tx|Rx)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)?",
    re.IGNORECASE,
)
_SIZE_UNITS = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}
_BDF_RE = re.compile(
    r"^(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])$"
)
_NA_VALUES = frozenset({"", "n/a", "[n/a]", "[not supported]", "[unknown error]"})


@dataclass(frozen=True)
class ToolResult:
    rc: int
    stdout: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def run_tool(
    argv: Sequence[str], timeout: float = TOOL_TIMEOUT_S
) -> Optional[ToolResult]:
    """Run an external tool; ``None`` when it is not installed or hangs."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return ToolResult(proc.returncode, proc.stdout)


@dataclass(frozen=True)
class GpuDesc:
    index: int
    pci_bus_id: Optional[str]
    name: str


@dataclass(frozen=True)
class GpuStat:
    util: Optional[float]
    mem_used_mib: Optional[float]
    power_w: Optional[float]


@dataclass(frozen=True)
class GpuStress:
    """One ``gpux`` sample: clocks, temperature, power, energy, reasons."""

    sm_mhz: Optional[float] = None
    mem_mhz: Optional[float] = None
    temp_c: Optional[float] = None
    power_w: Optional[float] = None
    energy_mj: Optional[int] = None
    reasons: Optional[int] = None  # NVML clocks-event bitmask

    def known(self) -> bool:
        return any(
            v is not None
            for v in (
                self.sm_mhz,
                self.mem_mhz,
                self.temp_c,
                self.power_w,
                self.energy_mj,
                self.reasons,
            )
        )


# --------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------


def natural_key(text: str) -> List[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def normalize_bdf(raw: str) -> Optional[str]:
    """nvidia-smi / NVML "00000000:1B:00.0" -> sysfs "0000:1b:00.0"."""
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


def finite(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when missing, N/A or not finite."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in _NA_VALUES:
            return None
        try:
            value = float(value)
        except ValueError:
            return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_proc_stat(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """(busy, total) jiffies from the aggregate "cpu" line of /proc/stat."""
    for line in (text or "").splitlines():
        if not line.startswith("cpu "):
            continue
        fields = [int(v) for v in line.split()[1:9] if v.isdigit()]
        if len(fields) < 5:
            return None
        total = sum(fields)  # user..steal; guest time is already in user/nice
        idle = fields[3] + fields[4]  # idle + iowait
        return total - idle, total
    return None


def cpu_util(
    prev: Optional[Tuple[int, int]], cur: Optional[Tuple[int, int]]
) -> Optional[float]:
    if prev is None or cur is None:
        return None
    total = cur[1] - prev[1]
    if total <= 0:
        return None
    return min(1.0, max(0.0, (cur[0] - prev[0]) / total))


def parse_nvlink_counters(text: Optional[str]) -> Dict[int, Tuple[int, int]]:
    """``nvidia-smi nvlink -gt d``: GPU -> (tx bytes, rx bytes) summed over links.

    Lines look like "Link 0: Data Tx: 123456 KiB" under a "GPU 0: ..." header.
    GPUs without any link line are omitted.
    """
    totals: Dict[int, List[int]] = {}
    gpu: Optional[int] = None
    for line in (text or "").splitlines():
        header = _NVLINK_GPU_RE.match(line)
        if header:
            gpu = int(header.group(1))
            continue
        match = _NVLINK_DATA_RE.search(line)
        if match is None or gpu is None:
            continue
        unit = _SIZE_UNITS.get((match.group(3) or "KiB").lower(), 1024)
        value = int(float(match.group(2)) * unit)
        slot = totals.setdefault(gpu, [0, 0])
        slot[0 if match.group(1).lower() == "tx" else 1] += value
    return {g: (v[0], v[1]) for g, v in totals.items()}


def parse_gpu_list(text: Optional[str]) -> List[GpuDesc]:
    """``nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader``."""
    gpus: List[GpuDesc] = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        gpus.append(GpuDesc(int(parts[0]), normalize_bdf(parts[1]), parts[2]))
    return gpus


def parse_gpu_stats(text: Optional[str]) -> Dict[int, GpuStat]:
    """``--query-gpu=index,utilization.gpu,memory.used,power.draw`` (nounits)."""
    stats: Dict[int, GpuStat] = {}
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        stats[int(parts[0])] = GpuStat(
            util=finite(parts[1]),
            mem_used_mib=finite(parts[2]),
            power_w=finite(parts[3]),
        )
    return stats


def parse_dmon_header(line: str) -> Optional[List[str]]:
    """Column names of a ``nvidia-smi dmon`` header ("# gpu rxpci txpci")."""
    tokens = line.lstrip("#").split()
    if tokens and tokens[0].lower() == "gpu":
        return [t.lower() for t in tokens]
    return None


def parse_dmon_sample(
    line: str, columns: Sequence[str]
) -> Optional[Tuple[int, float, float]]:
    """(gpu, tx bytes/s, rx bytes/s) from one ``dmon -s t`` data row."""
    tokens = line.split()
    if line.lstrip().startswith("#") or len(tokens) < len(columns):
        return None
    row = dict(zip(columns, tokens))
    tx = finite(row.get("txpci", row.get("pcitx")))
    rx = finite(row.get("rxpci", row.get("pcirx")))
    gpu = row.get("gpu", "")
    if tx is None or rx is None or not gpu.isdigit():
        return None
    return int(gpu), tx * DMON_MB_BYTES, rx * DMON_MB_BYTES


def parse_mask(text: Optional[str]) -> Optional[int]:
    """A bitmask printed as hex ("0x0000000000000004") or decimal."""
    raw = (text or "").strip().lower()
    if raw in _NA_VALUES:
        return None
    try:
        return int(raw, 16) if raw.startswith("0x") else int(raw)
    except ValueError:
        return None


def parse_smi_gpux(text: Optional[str], with_reasons: bool) -> Dict[int, GpuStress]:
    """``--query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw
    [,clocks_event_reasons.active] --format=csv,noheader,nounits``."""
    expected = len(SMI_GPUX_FIELDS) + (1 if with_reasons else 0)
    out: Dict[int, GpuStress] = {}
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != expected or not parts[0].isdigit():
            continue
        out[int(parts[0])] = GpuStress(
            sm_mhz=finite(parts[1]),
            mem_mhz=finite(parts[2]),
            temp_c=finite(parts[3]),
            power_w=finite(parts[4]),
            reasons=parse_mask(parts[5]) if with_reasons else None,
        )
    return out


def parse_ethtool_stats(text: Optional[str]) -> Dict[str, int]:
    """Every integer counter of ``ethtool -S <iface>`` ("     name: 123")."""
    counters: Dict[str, int] = {}
    for line in (text or "").splitlines():
        name, sep, value = line.strip().rpartition(":")
        name, value = name.strip(), value.strip()
        if not sep or not name or " " in name:
            continue
        try:
            counters[name] = int(value)
        except ValueError:
            continue
    return counters


def filter_ethx_counters(counters: Dict[str, int]) -> Dict[str, int]:
    """The pause/PFC/discard/CNP/ECN subset of ``ethtool -S`` counters."""
    kept = {
        name: value
        for name, value in sorted(counters.items())
        if ETHX_KEEP_RE.search(name) and not ETHX_PER_QUEUE_RE.match(name)
    }
    if len(kept) > ETHX_MAX_COUNTERS:
        kept = dict(list(kept.items())[:ETHX_MAX_COUNTERS])
    return kept


# --------------------------------------------------------------------------
# PCIe sub-sample aggregation
# --------------------------------------------------------------------------


@dataclass
class _PcieAgg:
    tx_sum: float = 0.0
    tx_max: float = 0.0
    rx_sum: float = 0.0
    rx_max: float = 0.0
    n: int = 0


class PcieAccumulator:
    """Thread-safe per-GPU aggregation of PCIe throughput sub-samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._aggs: Dict[int, _PcieAgg] = {}

    def add(self, gpu: int, tx_bps: float, rx_bps: float) -> None:
        with self._lock:
            agg = self._aggs.setdefault(gpu, _PcieAgg())
            agg.tx_sum += tx_bps
            agg.rx_sum += rx_bps
            agg.tx_max = max(agg.tx_max, tx_bps)
            agg.rx_max = max(agg.rx_max, rx_bps)
            agg.n += 1

    def drain(self) -> Dict[int, Dict[str, Any]]:
        """Per-GPU mean/max since the previous drain; resets the counters."""
        with self._lock:
            aggs, self._aggs = self._aggs, {}
        return {
            gpu: {
                "tx_mean_Bps": round(a.tx_sum / a.n),
                "tx_max_Bps": round(a.tx_max),
                "rx_mean_Bps": round(a.rx_sum / a.n),
                "rx_max_Bps": round(a.rx_max),
                "n": a.n,
            }
            for gpu, a in aggs.items()
            if a.n
        }


# --------------------------------------------------------------------------
# NVML (pynvml) source
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NvlinkFields:
    """The NVML field pair that answered on GPU0, and how to scope it."""

    tx_name: str
    rx_name: str
    tx_id: int
    rx_id: int
    unit_bytes: int
    scopes: Tuple[int, ...]  # (NVML_SCOPE_ALL_LINKS,) or per-link ids

    @property
    def source(self) -> str:
        return "nvml:" + self.tx_name + "," + self.rx_name

    @property
    def scope(self) -> str:
        return "per_link" if self.scopes != (NVML_SCOPE_ALL_LINKS,) else "aggregate"


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def nvml_field_int(value: Any) -> Optional[int]:
    """Integer payload of one c_nvmlFieldValue_t, ``None`` when invalid."""
    if value.nvmlReturn != NVML_SUCCESS:
        return None
    attr = NVML_VALUE_ATTRS.get(value.valueType)
    if attr is None:
        return None
    try:
        raw = getattr(value.value, attr)
    except AttributeError:
        return None
    number = finite(raw)
    if number is None or number < 0:
        return None
    return int(number)


def read_nvlink_pair(
    nvml: Any, handle: Any, fields: NvlinkFields
) -> Optional[Tuple[int, int]]:
    """Cumulative (tx, rx) bytes of one GPU summed over ``fields.scopes``."""
    request = []
    for scope in fields.scopes:
        request.append((fields.tx_id, scope))
        request.append((fields.rx_id, scope))
    try:
        values = nvml.nvmlDeviceGetFieldValues(handle, request)
    except (nvml.NVMLError, TypeError):  # TypeError: pynvml without scope ids
        return None
    tx_total = rx_total = 0
    valid = False
    for k in range(len(fields.scopes)):
        tx = nvml_field_int(values[2 * k])
        rx = nvml_field_int(values[2 * k + 1])
        if tx is None or rx is None:
            continue
        tx_total += tx
        rx_total += rx
        valid = True
    if not valid:
        return None
    return tx_total * fields.unit_bytes, rx_total * fields.unit_bytes


def resolve_nvlink_fields(nvml: Any, handle: Any) -> Optional[NvlinkFields]:
    """First NVLink counter pair this pynvml/driver serves on ``handle``."""
    max_links = int(getattr(nvml, "NVML_NVLINK_MAX_LINKS", DEFAULT_MAX_NVLINKS))
    for tx_name, rx_name, unit in NVLINK_FIELD_CANDIDATES:
        tx_id = getattr(nvml, tx_name, None)
        rx_id = getattr(nvml, rx_name, None)
        if tx_id is None or rx_id is None:
            continue
        for scopes in ((NVML_SCOPE_ALL_LINKS,), tuple(range(max_links))):
            fields = NvlinkFields(tx_name, rx_name, tx_id, rx_id, unit, scopes)
            if read_nvlink_pair(nvml, handle, fields) is not None:
                return fields
    return None


def _nvml_try(nvml: Any, fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except nvml.NVMLError:
        return None


class NvmlGpus:
    """GPU metrics through NVML; ``nvml`` is the pynvml module or a fake."""

    def __init__(self, nvml: Any) -> None:
        self._nvml = nvml
        nvml.nvmlInit()
        try:
            count = nvml.nvmlDeviceGetCount()
            self._handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self.gpus = [
                GpuDesc(i, self._bus_id(h), _text(nvml.nvmlDeviceGetName(h)))
                for i, h in enumerate(self._handles)
            ]
            self.nvlink_fields = (
                resolve_nvlink_fields(nvml, self._handles[0]) if self._handles else None
            )
        except nvml.NVMLError:
            nvml.nvmlShutdown()
            raise

    def _bus_id(self, handle: Any) -> Optional[str]:
        info = _nvml_try(self._nvml, self._nvml.nvmlDeviceGetPciInfo, handle)
        if info is None:
            return None
        return normalize_bdf(_text(info.busId))

    @property
    def handles(self) -> List[Any]:
        """NVML device handles in index order."""
        return list(self._handles)

    def close(self) -> None:
        self._nvml.nvmlShutdown()

    def pcie(self, gpu: int) -> Optional[Tuple[float, float]]:
        """(tx, rx) bytes/s over NVML's ~20 ms window; blocks for that long."""
        nvml = self._nvml
        handle = self._handles[gpu]
        tx = _nvml_try(
            nvml, nvml.nvmlDeviceGetPcieThroughput, handle, nvml.NVML_PCIE_UTIL_TX_BYTES
        )
        rx = _nvml_try(
            nvml, nvml.nvmlDeviceGetPcieThroughput, handle, nvml.NVML_PCIE_UTIL_RX_BYTES
        )
        if tx is None or rx is None:
            return None
        return float(tx) * NVML_PCIE_KB_BYTES, float(rx) * NVML_PCIE_KB_BYTES

    def gpu_stats(self) -> Dict[int, GpuStat]:
        nvml = self._nvml
        stats: Dict[int, GpuStat] = {}
        for index, handle in enumerate(self._handles):
            util = _nvml_try(nvml, nvml.nvmlDeviceGetUtilizationRates, handle)
            mem = _nvml_try(nvml, nvml.nvmlDeviceGetMemoryInfo, handle)
            power_mw = _nvml_try(nvml, nvml.nvmlDeviceGetPowerUsage, handle)
            stats[index] = GpuStat(
                util=finite(util.gpu) if util is not None else None,
                mem_used_mib=(
                    round(mem.used / (1024 * 1024), 1) if mem is not None else None
                ),
                power_w=(round(power_mw / 1000.0, 2) if power_mw is not None else None),
            )
        return stats

    def nvlink(self) -> Dict[int, Tuple[int, int]]:
        fields = self.nvlink_fields
        if fields is None:
            return {}
        out: Dict[int, Tuple[int, int]] = {}
        for index, handle in enumerate(self._handles):
            pair = read_nvlink_pair(self._nvml, handle, fields)
            if pair is not None:
                out[index] = pair
        return out

    def _optional(self, name: str, *args: Any) -> Any:
        """``nvml.<name>(*args)``; None when unsupported or absent in pynvml."""
        fn = getattr(self._nvml, name, None)
        if fn is None:
            return None
        return _nvml_try(self._nvml, fn, *args)

    def stress(self) -> Dict[int, GpuStress]:
        """Clocks, temperature, power, total energy and clocks-event reasons."""
        nvml = self._nvml
        sm_clock = getattr(nvml, "NVML_CLOCK_SM", NVML_CLOCK_SM_DEFAULT)
        mem_clock = getattr(nvml, "NVML_CLOCK_MEM", NVML_CLOCK_MEM_DEFAULT)
        sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", NVML_TEMPERATURE_GPU_DEFAULT)
        reason_fn = next(
            (name for name in NVML_REASON_FUNCS if hasattr(nvml, name)), None
        )
        out: Dict[int, GpuStress] = {}
        for index, handle in enumerate(self._handles):
            power_mw = self._optional("nvmlDeviceGetPowerUsage", handle)
            energy = self._optional("nvmlDeviceGetTotalEnergyConsumption", handle)
            reasons = self._optional(reason_fn, handle) if reason_fn else None
            sample = GpuStress(
                sm_mhz=finite(
                    self._optional("nvmlDeviceGetClockInfo", handle, sm_clock)
                ),
                mem_mhz=finite(
                    self._optional("nvmlDeviceGetClockInfo", handle, mem_clock)
                ),
                temp_c=finite(
                    self._optional("nvmlDeviceGetTemperature", handle, sensor)
                ),
                power_w=(
                    round(power_mw / 1000.0, 2)
                    if finite(power_mw) is not None
                    else None
                ),
                energy_mj=int(energy) if finite(energy) is not None else None,
                reasons=int(reasons) if finite(reasons) is not None else None,
            )
            if sample.known():
                out[index] = sample
        return out


class NvmlPcieSampler:
    """One thread per GPU sampling PCIe every ``interval`` into ``acc``.

    NVML is thread-safe and each throughput query blocks ~20 ms, so a single
    thread could not sample eight GPUs at 10 Hz.
    """

    def __init__(self, gpus: NvmlGpus, acc: PcieAccumulator, interval: float):
        self._gpus = gpus
        self._acc = acc
        self._interval = interval
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    def __enter__(self) -> NvmlPcieSampler:
        for desc in self._gpus.gpus:
            thread = threading.Thread(
                target=self._loop, args=(desc.index,), name=f"pcie-{desc.index}"
            )
            thread.daemon = True
            thread.start()
            self._threads.append(thread)
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _loop(self, gpu: int) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            sample = self._gpus.pcie(gpu)
            if sample is not None:
                self._acc.add(gpu, sample[0], sample[1])
            next_at += self._interval
            delay = next_at - time.monotonic()
            if delay <= 0:  # fell behind: resynchronise instead of bursting
                next_at = time.monotonic()
                delay = self._interval / 10
            self._stop.wait(delay)


# --------------------------------------------------------------------------
# NVML GPM (GPU performance monitoring, Hopper and newer)
# --------------------------------------------------------------------------


def nvml_return_name(nvml: Any, code: int) -> str:
    """The pynvml constant naming NVML return ``code``, e.g.
    "NVML_ERROR_NO_PERMISSION"; "NVML_ERROR_<code>" when unnamed."""
    for name in sorted(n for n in dir(nvml) if n.startswith("NVML_ERROR_")):
        if getattr(nvml, name, None) == code:
            return name
    return "NVML_ERROR_%d" % code


def nvml_error_name(nvml: Any, exc: BaseException) -> str:
    """Name of an NVMLError (by its return code), else the exception type."""
    code = getattr(exc, "value", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return nvml_return_name(nvml, code)
    return type(exc).__name__


def resolve_gpm_metrics(nvml: Any) -> List[Tuple[str, int]]:
    """(gpm field, metric id) of every GPM_METRICS constant ``nvml`` defines."""
    out: List[Tuple[str, int]] = []
    for key, name in GPM_METRICS:
        metric_id = getattr(nvml, name, None)
        if isinstance(metric_id, int) and not isinstance(metric_id, bool):
            out.append((key, metric_id))
    return out


@dataclass
class GpmGpu:
    """GPM state of one sampling GPU: two NVML samples, swapped every read."""

    index: int
    handle: Any
    prev: Any  # the sample taken at ``taken_at``
    cur: Any  # the buffer the next sample goes into
    taken_at: float  # clock() of ``prev``
    metrics: List[Tuple[str, int]]  # (gpm field, metric id) still requested


class NvmlGpm:
    """NVML GPM interval metrics of every GPU that supports them.

    ``open`` probes each GPU (nvmlGpmQueryDeviceSupport), allocates two
    samples and takes the first; every ``read`` costs one nvmlGpmSampleGet
    and one nvmlGpmMetricsGet per GPU, the values being averages since that
    GPU's previous sample. A call that fails disables GPM on that GPU only;
    a metric NVML refuses is dropped for that GPU. ``close`` frees the
    samples (before nvmlShutdown). ``nvml`` is the pynvml module or a fake.
    """

    def __init__(self, nvml: Any, clock: Callable[[], float] = time.monotonic):
        self._nvml = nvml
        self._clock = clock
        self._errors: Tuple[type, ...] = (nvml.NVMLError,) + GPM_BINDING_ERRORS
        self.metrics = resolve_gpm_metrics(nvml)
        self.status: Dict[int, str] = {}
        self._gpus: Dict[int, GpmGpu] = {}

    @property
    def metric_keys(self) -> List[str]:
        return [key for key, _ in self.metrics]

    @property
    def enabled(self) -> List[int]:
        """Indices of the GPUs still sampling."""
        return sorted(self._gpus)

    def open(self, handles: Sequence[Any]) -> Dict[int, str]:
        """Probe every GPU; returns GPU index -> status (module docstring)."""
        api = all(hasattr(self._nvml, name) for name in GPM_API)
        for index, handle in enumerate(handles):
            if api and self.metrics:
                self.status[index] = self._probe(index, handle)
            else:
                self.status[index] = "no-gpm-api"
        return dict(self.status)

    def _probe(self, index: int, handle: Any) -> str:
        nvml = self._nvml
        samples: List[Any] = []
        try:
            support = nvml.nvmlGpmQueryDeviceSupport(handle)
            if not getattr(support, "isSupportedDevice", 0):
                return "unsupported"
            samples.append(nvml.nvmlGpmSampleAlloc())
            samples.append(nvml.nvmlGpmSampleAlloc())
            nvml.nvmlGpmSampleGet(handle, samples[0])
        except self._errors as exc:
            self._free(samples)
            return nvml_error_name(nvml, exc)
        self._gpus[index] = GpmGpu(
            index, handle, samples[0], samples[1], self._clock(), list(self.metrics)
        )
        return "ok"

    def _metrics_get(self, gpu: GpmGpu) -> List[Tuple[int, Any]]:
        """(nvmlReturn, value) per requested metric between prev and cur."""
        nvml = self._nvml
        request = nvml.c_nvmlGpmMetricsGet_t()
        request.version = getattr(
            nvml, "NVML_GPM_METRICS_GET_VERSION", GPM_METRICS_GET_VERSION
        )
        request.numMetrics = len(gpu.metrics)
        request.sample1 = gpu.prev
        request.sample2 = gpu.cur
        for slot, (_, metric_id) in enumerate(gpu.metrics):
            request.metrics[slot].metricId = metric_id
        nvml.nvmlGpmMetricsGet(request)
        return [
            (int(request.metrics[slot].nvmlReturn), request.metrics[slot].value)
            for slot in range(len(gpu.metrics))
        ]

    def read(self) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, Any]]]:
        """GPU -> gpm fields of the interval since its previous sample, and
        the failures as note payloads (source, gpu, error[, metrics])."""
        values: Dict[int, Dict[str, float]] = {}
        notes: List[Dict[str, Any]] = []
        for index in sorted(self._gpus):
            gpu = self._gpus[index]
            if self._clock() - gpu.taken_at < GPM_MIN_INTERVAL_S:
                continue  # too soon for NVML: the interval simply grows
            try:
                self._nvml.nvmlGpmSampleGet(gpu.handle, gpu.cur)
                taken = self._clock()
                results = self._metrics_get(gpu)
            except self._errors as exc:
                error = nvml_error_name(self._nvml, exc)
                self._disable(index, error)
                notes.append({"source": "gpm", "gpu": index, "error": error})
                continue
            row: Dict[str, float] = {"dt_s": round(taken - gpu.taken_at, 4)}
            gpu.prev, gpu.cur, gpu.taken_at = gpu.cur, gpu.prev, taken
            failed: Dict[str, List[str]] = {}
            kept: List[Tuple[str, int]] = []
            for (key, metric_id), (ret, value) in zip(gpu.metrics, results):
                if ret != NVML_SUCCESS:
                    name = nvml_return_name(self._nvml, ret)
                    failed.setdefault(name, []).append(key)
                    continue
                kept.append((key, metric_id))
                number = finite(value)
                if number is not None and number >= 0:
                    digits = (
                        GPM_RATE_DIGITS if key.endswith("_mibs") else GPM_PCT_DIGITS
                    )
                    row[key] = round(number, digits)
            for error, keys in sorted(failed.items()):
                notes.append(
                    {"source": "gpm", "gpu": index, "error": error, "metrics": keys}
                )
            gpu.metrics = kept
            if not kept:  # NVML refused every metric: stop asking
                error = sorted(failed)[0]
                self._disable(index, error)
                notes.append({"source": "gpm", "gpu": index, "error": error})
            if len(row) > 1:
                values[index] = row
        return values, notes

    def _free(self, samples: Sequence[Any]) -> None:
        for sample in samples:
            try:
                self._nvml.nvmlGpmSampleFree(sample)
            except self._errors as exc:
                logger.debug("nvmlGpmSampleFree failed: %s", exc)

    def _disable(self, index: int, status: str) -> None:
        gpu = self._gpus.pop(index, None)
        if gpu is not None:
            self._free((gpu.prev, gpu.cur))
        self.status[index] = status

    def close(self) -> None:
        """Free every sample; idempotent."""
        for index in sorted(self._gpus):
            gpu = self._gpus.pop(index)
            self._free((gpu.prev, gpu.cur))


# --------------------------------------------------------------------------
# nvidia-smi fallbacks
# --------------------------------------------------------------------------


class SmiGpus:
    """GPU list and stats via nvidia-smi when NVML is unavailable."""

    def __init__(self, run: Callable[..., Optional[ToolResult]] = run_tool):
        self._run = run
        listing = run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id,name", "--format=csv,noheader"]
        )
        self.gpus = parse_gpu_list(listing.stdout) if listing and listing.ok else []

    def gpu_stats(self) -> Dict[int, GpuStat]:
        result = self._run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
        return parse_gpu_stats(result.stdout) if result and result.ok else {}


def smi_gpux_argv(reason_field: Optional[str]) -> List[str]:
    fields = list(SMI_GPUX_FIELDS) + ([reason_field] if reason_field else [])
    return [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]


class SmiGpuStress:
    """``gpux`` samples from nvidia-smi; probes which reasons field exists.

    Older drivers reject ``clocks_event_reasons.active`` (non-zero exit), so
    the pre-R535 name and then no reasons field are tried in turn.
    """

    def __init__(self, run: Callable[..., Optional[ToolResult]] = run_tool):
        self._run = run
        self.argv: Optional[List[str]] = None
        self.with_reasons = False
        for reason_field in SMI_REASON_FIELDS + (None,):
            argv = smi_gpux_argv(reason_field)
            result = run(argv)
            if result is not None and result.ok:
                if parse_smi_gpux(result.stdout, reason_field is not None):
                    self.argv = argv
                    self.with_reasons = reason_field is not None
                    break

    @property
    def available(self) -> bool:
        return self.argv is not None

    def read(self) -> Dict[int, GpuStress]:
        if self.argv is None:
            return {}
        result = self._run(self.argv)
        if result is None or not result.ok:
            return {}
        return parse_smi_gpux(result.stdout, self.with_reasons)


class SmiNvlink:
    """Cumulative NVLink data counters from ``nvidia-smi nvlink -gt d``."""

    def __init__(self, run: Callable[..., Optional[ToolResult]] = run_tool):
        self._run = run

    def read(self) -> Dict[int, Tuple[int, int]]:
        result = self._run(["nvidia-smi", "nvlink", "-gt", "d"])
        return parse_nvlink_counters(result.stdout) if result and result.ok else {}


class DmonPcie:
    """Streams ``nvidia-smi dmon -s t`` (1 sample/s/GPU) into ``acc``."""

    def __init__(self, acc: PcieAccumulator, argv: Sequence[str] = DMON_ARGV):
        self._acc = acc
        self._argv = list(argv)
        self._stack = ExitStack()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> DmonPcie:
        try:
            proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            logger.warning("nvidia-smi dmon unavailable: %s", exc)
            return self
        self._proc = self._stack.enter_context(proc)
        self._thread = threading.Thread(target=self._pump, name="dmon-pcie")
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:  # drains to EOF once the child is gone
            self._thread.join(timeout=5.0)
        self._stack.close()  # Popen.__exit__: close pipes, reap the child

    def _pump(self) -> None:
        if self._proc is not None and self._proc.stdout is not None:
            pump_dmon(self._proc.stdout, self._acc)


def pump_dmon(lines: Iterable[str], acc: PcieAccumulator) -> int:
    """Feed ``nvidia-smi dmon -s t`` output into ``acc``; returns samples added.

    dmon repeats its two header lines periodically; the column layout is
    re-read from every header.
    """
    columns: Optional[List[str]] = None
    added = 0
    for line in lines:
        header = parse_dmon_header(line) if line.startswith("#") else None
        if header:
            columns = header
            continue
        sample = parse_dmon_sample(line, columns) if columns else None
        if sample is not None:
            acc.add(*sample)
            added += 1
    return added


# --------------------------------------------------------------------------
# sysfs / procfs counters
# --------------------------------------------------------------------------


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def read_int(path: str) -> Optional[int]:
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


class HostCounters:
    """IB, CPU and network counters; ``root`` is "/" except in tests."""

    def __init__(self, root: str = "/"):
        self.root = root

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *(p.lstrip("/") for p in parts))

    def ib_ports(self) -> List[Tuple[str, int]]:
        """Active (state "4: ACTIVE") HCA ports as (device, port)."""
        ib_root = self.path("sys/class/infiniband")
        try:
            devices = sorted(os.listdir(ib_root), key=natural_key)
        except OSError:
            return []
        active: List[Tuple[str, int]] = []
        for device in devices:
            ports_dir = os.path.join(ib_root, device, "ports")
            try:
                ports = sorted(
                    (p for p in os.listdir(ports_dir) if p.isdigit()), key=int
                )
            except OSError:
                continue
            for port in ports:
                state = read_text(os.path.join(ports_dir, port, "state"))
                if state is not None and state.strip().startswith("4"):
                    active.append((device, int(port)))
        return active

    def ib(self, device: str, port: int) -> Optional[Tuple[int, int]]:
        """Cumulative (tx, rx) bytes of one port (raw counter x 4)."""
        base = self.path("sys/class/infiniband", device, "ports", str(port), "counters")
        tx = read_int(os.path.join(base, "port_xmit_data"))
        rx = read_int(os.path.join(base, "port_rcv_data"))
        if tx is None or rx is None:
            return None
        return tx * IB_UNIT_BYTES, rx * IB_UNIT_BYTES

    def cpu_times(self) -> Optional[Tuple[int, int]]:
        return parse_proc_stat(read_text(self.path("proc/stat")))

    def net(self, iface: str) -> Optional[Tuple[int, int]]:
        base = self.path("sys/class/net", iface, "statistics")
        tx = read_int(os.path.join(base, "tx_bytes"))
        rx = read_int(os.path.join(base, "rx_bytes"))
        if tx is None or rx is None:
            return None
        return tx, rx

    def port_dir(self, device: str, port: int) -> str:
        return self.path("sys/class/infiniband", device, "ports", str(port))

    def link_layer(self, device: str, port: int) -> Optional[str]:
        """ "InfiniBand" or "Ethernet" (RoCE); None when unreadable."""
        text = read_text(os.path.join(self.port_dir(device, port), "link_layer"))
        return text.strip() if text and text.strip() else None

    def ibx_paths(self, device: str, port: int) -> List[Tuple[str, str]]:
        """(name, path) of the IBX_* counters this port exposes."""
        base = self.port_dir(device, port)
        found: List[Tuple[str, str]] = []
        for sub, names in (
            ("counters", IBX_PORT_COUNTERS),
            ("hw_counters", IBX_HW_COUNTERS),
        ):
            for name in names:
                path = os.path.join(base, sub, name)
                if os.path.isfile(path):
                    found.append((name, path))
        return found

    def roce_netdev(self, device: str, port: int) -> Optional[str]:
        """The netdev of a RoCE port: gid_attrs/ndevs/0, else device/net/*."""
        ndev = read_text(os.path.join(self.port_dir(device, port), "gid_attrs/ndevs/0"))
        if ndev and ndev.strip():
            return ndev.strip()
        net_dir = self.path("sys/class/infiniband", device, "device", "net")
        try:
            names = sorted(os.listdir(net_dir), key=natural_key)
        except OSError:
            return None
        for name in names:  # multi-port functions: dev_port is 0-based
            if read_int(os.path.join(net_dir, name, "dev_port")) == port - 1:
                return name
        return names[0] if names else None


def read_counters(paths: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    """Integer value of every readable (name, path); unreadable ones omitted."""
    out: Dict[str, int] = {}
    for name, path in paths:
        value = read_int(path)
        if value is not None:
            out[name] = value
    return out


# --------------------------------------------------------------------------
# Sampling loop
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    out: str
    node: str
    interval: float = 1.0
    pcie_interval: float = 0.1
    duration: Optional[float] = None
    control_interface: Optional[str] = "wg0"
    tag: Optional[str] = None
    slow_interval: float = DEFAULT_SLOW_INTERVAL  # 0 disables gpux/ibx/ethx
    gpm: bool = True  # NVML GPM interval metrics (the gpm kind)

    @property
    def slow_every(self) -> int:
        """Ticks between slow samples; 0 when stress sampling is disabled."""
        if self.slow_interval <= 0:
            return 0
        return max(1, int(round(self.slow_interval / self.interval)))


@dataclass(frozen=True)
class IbxPort:
    device: str
    port: int
    paths: Tuple[Tuple[str, str], ...]  # (counter name, sysfs path)


@dataclass(frozen=True)
class EthxPort:
    device: str
    port: int
    iface: str


@dataclass
class Sources:
    """Everything the loop samples; ``open_sources`` builds the real ones."""

    host: HostCounters
    gpus: List[GpuDesc]
    nvml: bool
    gpu_stats: Callable[[], Dict[int, GpuStat]]
    pcie: PcieAccumulator
    pcie_source: Optional[str]
    nvlink: Optional[Callable[[], Dict[int, Tuple[int, int]]]]
    nvlink_source: Optional[str]
    ib_ports: List[Tuple[str, int]]
    # Stress sources; the defaults disable them.
    slow_every: int = 0
    gpux: Optional[Callable[[], Dict[int, GpuStress]]] = None
    gpux_source: Optional[str] = None
    gpux_every: int = 0  # 1 = every tick, else only on slow ticks
    link_layers: Dict[Tuple[str, int], str] = field(default_factory=dict)
    ibx: List[IbxPort] = field(default_factory=list)
    ethx: List[EthxPort] = field(default_factory=list)
    ethtool: Callable[..., Optional[ToolResult]] = run_tool
    notes: List[Dict[str, Any]] = field(default_factory=list)
    failed: Set[str] = field(default_factory=set)  # ifaces disabled after errors
    # NVML GPM; None writes no gpm records. gpm_status: GPU -> start status.
    gpm: Optional[NvmlGpm] = None
    gpm_status: Dict[int, str] = field(default_factory=dict)


def _ts(value: float) -> float:
    return round(value, 4)


def _clean(value: Any) -> Any:
    """Replace non-finite floats so every line stays strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def write_records(out: IO[str], records: Sequence[Dict[str, Any]]) -> None:
    out.write(
        "".join(
            json.dumps(_clean(r), separators=(",", ":"), allow_nan=False) + "\n"
            for r in records
        )
    )
    out.flush()


def meta_record(
    cfg: AgentConfig, sources: Sources, hostname: str, now: float
) -> Dict[str, Any]:
    return {
        "k": "meta",
        "t": _ts(now),
        "node": cfg.node,
        "hostname": hostname,
        "ib_unit_bytes": IB_UNIT_BYTES,
        "nvml": sources.nvml,
        "nvlink_source": sources.nvlink_source,
        "gpus": [
            {"index": g.index, "pci_bus_id": g.pci_bus_id, "name": g.name}
            for g in sources.gpus
        ],
        "hcas": [{"name": dev, "port": port} for dev, port in sources.ib_ports],
        "interval": cfg.interval,
        "pcie_interval": cfg.pcie_interval,
        "pcie_source": sources.pcie_source,
        "control_interface": cfg.control_interface,
        "tag": cfg.tag,
        "version": VERSION,
        "slow_interval": cfg.slow_interval,
        "slow_every": sources.slow_every,
        "gpux_source": sources.gpux_source if sources.gpux else None,
        "gpux_every": sources.gpux_every if sources.gpux else 0,
        "link_layers": [
            {"name": dev, "port": port, "link_layer": layer}
            for (dev, port), layer in sorted(sources.link_layers.items())
        ],
        "ibx_ports": [
            {"name": p.device, "port": p.port, "counters": [n for n, _ in p.paths]}
            for p in sources.ibx
        ],
        "ethx_ifaces": [
            {"name": e.device, "port": e.port, "iface": e.iface} for e in sources.ethx
        ],
        "notes": list(sources.notes),
        "gpm": {str(gpu): status for gpu, status in sorted(sources.gpm_status.items())},
        "gpm_metrics": sources.gpm.metric_keys if sources.gpm is not None else [],
    }


def _note(sources: Sources, key: str, record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A one-time ``note`` record for a stress source that failed after start."""
    if key in sources.failed:
        return []
    sources.failed.add(key)
    logger.warning("stress source disabled: %s", record)
    note: Dict[str, Any] = {"k": "note"}
    note.update(record)
    return [note]


def stress_records(
    sources: Sources, wall: Callable[[], float], slow: bool
) -> List[Dict[str, Any]]:
    """``gpux`` (every tick with NVML, else slow ticks), ``ibx`` and ``ethx``
    (slow ticks) records; failing ethtool ifaces are disabled with a note."""
    records: List[Dict[str, Any]] = []
    if not sources.slow_every:
        return records
    if sources.gpux is not None and (slow or sources.gpux_every == 1):
        before = wall()
        samples = sources.gpux()
        stamp = _ts((before + wall()) / 2)
        for gpu, s in sorted(samples.items()):
            records.append(
                {
                    "k": "gpux",
                    "t": stamp,
                    "gpu": gpu,
                    "sm_mhz": s.sm_mhz,
                    "mem_mhz": s.mem_mhz,
                    "temp_c": s.temp_c,
                    "power_w": s.power_w,
                    "energy_mj": s.energy_mj,
                    "reasons": s.reasons,
                }
            )
    if not slow:
        return records
    for port in sources.ibx:
        counters = read_counters(port.paths)
        if counters:
            records.append(
                {
                    "k": "ibx",
                    "t": _ts(wall()),
                    "dev": port.device,
                    "port": port.port,
                    "c": counters,
                }
            )
    for eth in sources.ethx:
        if eth.iface in sources.failed:
            continue
        before = wall()
        result = sources.ethtool(["ethtool", "-S", eth.iface], ETHTOOL_TIMEOUT_S)
        stamp = _ts((before + wall()) / 2)
        if result is None or not result.ok:
            if result is None:
                error = "not installed or timed out"
            else:
                error = "exit %d" % result.rc
            records.extend(
                _note(
                    sources,
                    eth.iface,
                    {
                        "t": stamp,
                        "source": "ethtool",
                        "iface": eth.iface,
                        "dev": eth.device,
                        "error": error,
                    },
                )
            )
            continue
        counters = filter_ethx_counters(parse_ethtool_stats(result.stdout))
        if counters:
            records.append(
                {
                    "k": "ethx",
                    "t": stamp,
                    "iface": eth.iface,
                    "dev": eth.device,
                    "c": counters,
                }
            )
    return records


def gpm_records(sources: Sources, wall: Callable[[], float]) -> List[Dict[str, Any]]:
    """One ``gpm`` record (GPUs with a fresh interval) and a ``note`` per
    GPM failure; nothing when GPM is off."""
    if sources.gpm is None:
        return []
    before = wall()
    values, failures = sources.gpm.read()
    stamp = _ts((before + wall()) / 2)  # every GPU was sampled meanwhile
    records: List[Dict[str, Any]] = []
    if values:
        records.append(
            {
                "k": "gpm",
                "t": stamp,
                "g": {str(gpu): row for gpu, row in sorted(values.items())},
            }
        )
    for failure in failures:
        logger.warning("GPM source reduced: %s", failure)
        note: Dict[str, Any] = {"k": "note", "t": stamp}
        note.update(failure)
        records.append(note)
    return records


def sample_tick(
    cfg: AgentConfig,
    sources: Sources,
    prev_cpu: Optional[Tuple[int, int]],
    wall: Callable[[], float],
    slow: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int]]]:
    """One interval's records, and the CPU counters for the next delta.

    ``slow`` adds the slow-tick stress kinds (ibx, ethx; gpux on nvidia-smi).
    """
    records: List[Dict[str, Any]] = []
    for device, port in sources.ib_ports:
        counters = sources.host.ib(device, port)
        if counters is not None:
            records.append(
                {
                    "k": "ib",
                    "t": _ts(wall()),
                    "dev": device,
                    "port": port,
                    "tx": counters[0],
                    "rx": counters[1],
                }
            )
    now = _ts(wall())
    for gpu, agg in sorted(sources.pcie.drain().items()):
        record: Dict[str, Any] = {"k": "pcie", "t": now, "gpu": gpu}
        record.update(agg)
        records.append(record)
    if sources.nvlink is not None:
        before = wall()
        counters_by_gpu = sources.nvlink()
        stamp = _ts((before + wall()) / 2)  # the read may take a while
        for gpu, (tx, rx) in sorted(counters_by_gpu.items()):
            records.append({"k": "nvlink", "t": stamp, "gpu": gpu, "tx": tx, "rx": rx})
    stats = sources.gpu_stats()
    now = _ts(wall())
    for gpu, stat in sorted(stats.items()):
        records.append(
            {
                "k": "gpu",
                "t": now,
                "gpu": gpu,
                "util": stat.util,
                "mem_used_mib": stat.mem_used_mib,
                "power_w": stat.power_w,
            }
        )
    records.extend(gpm_records(sources, wall))
    cpu = sources.host.cpu_times()
    util = cpu_util(prev_cpu, cpu)
    if util is not None:
        records.append({"k": "cpu", "t": _ts(wall()), "util": round(util, 4)})
    if cfg.control_interface:
        net = sources.host.net(cfg.control_interface)
        if net is not None:
            records.append(
                {
                    "k": "net",
                    "t": _ts(wall()),
                    "iface": cfg.control_interface,
                    "tx": net[0],
                    "rx": net[1],
                }
            )
    records.extend(stress_records(sources, wall, slow))
    return records, (cpu if cpu is not None else prev_cpu)


def run_loop(
    cfg: AgentConfig,
    sources: Sources,
    out: IO[str],
    should_stop: Callable[[], bool],
    hostname: str,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Write meta, then one batch per interval until stopped. Returns ticks.

    The wait polls ``should_stop`` every STOP_POLL_S, so a signal handler only
    has to set a flag (no locks inside the handler, no deadlock).
    """
    write_records(out, [meta_record(cfg, sources, hostname, wall())])
    start = clock()
    deadline = None if cfg.duration is None else start + cfg.duration
    next_tick = start + cfg.interval
    prev_cpu = sources.host.cpu_times()
    ticks = 0
    while not should_stop():
        now = clock()
        if deadline is not None and now >= deadline:
            break
        target = next_tick if deadline is None else min(next_tick, deadline)
        if now < target:
            sleep(min(STOP_POLL_S, target - now))
            continue
        slow = bool(sources.slow_every) and ticks % sources.slow_every == 0
        records, prev_cpu = sample_tick(cfg, sources, prev_cpu, wall, slow)
        write_records(out, records)
        ticks += 1
        next_tick += cfg.interval
        now = clock()
        if next_tick <= now:  # a slow tick: skip missed slots, never burst
            next_tick = now + cfg.interval
    # Final sample so counters cover everything up to the stop request.
    records, _ = sample_tick(cfg, sources, prev_cpu, wall, slow=True)
    write_records(out, records)
    return ticks + 1


def open_sources(
    cfg: AgentConfig,
    stack: ExitStack,
    nvml_module: Any = pynvml,
    run: Callable[..., Optional[ToolResult]] = run_tool,
    root: str = "/",
    dmon_argv: Sequence[str] = DMON_ARGV,
) -> Sources:
    """Build the real sources; cleanups are registered on ``stack``."""
    host = HostCounters(root)
    acc = PcieAccumulator()
    nvml_gpus: Optional[NvmlGpus] = None
    nvml_error: Optional[str] = None
    if nvml_module is not None:
        try:
            nvml_gpus = NvmlGpus(nvml_module)
        except nvml_module.NVMLError as exc:
            logger.warning("NVML unavailable (%s); falling back to nvidia-smi", exc)
            nvml_error = nvml_error_name(nvml_module, exc)
    nvlink: Optional[Callable[[], Dict[int, Tuple[int, int]]]] = None
    nvlink_source: Optional[str] = None
    pcie_source: Optional[str] = None
    if nvml_gpus is not None:
        stack.callback(nvml_gpus.close)
        gpus = nvml_gpus.gpus
        gpu_stats: Callable[[], Dict[int, GpuStat]] = nvml_gpus.gpu_stats
        if gpus:
            stack.enter_context(NvmlPcieSampler(nvml_gpus, acc, cfg.pcie_interval))
            pcie_source = "nvml"
        if nvml_gpus.nvlink_fields is not None:
            nvlink = nvml_gpus.nvlink
            nvlink_source = nvml_gpus.nvlink_fields.source
            logger.info(
                "NVLink counters: %s (%s)", nvlink_source, nvml_gpus.nvlink_fields.scope
            )
    else:
        smi = SmiGpus(run)
        gpus, gpu_stats = smi.gpus, smi.gpu_stats
        if gpus:
            dmon = stack.enter_context(DmonPcie(acc, dmon_argv))
            pcie_source = "nvidia-smi dmon" if dmon.running else None
    if nvlink is None and gpus:
        reader = SmiNvlink(run)
        if reader.read():
            nvlink, nvlink_source = reader.read, "nvidia-smi"
    if not gpus:
        logger.warning("no GPUs visible: GPU, PCIe and NVLink lines are disabled")
    elif nvlink is None:
        logger.warning("no NVLink counters available: nvlink lines are disabled")
    ib_ports = host.ib_ports()
    if not ib_ports:
        logger.warning("no active RDMA ports under /sys/class/infiniband")
    sources = Sources(
        host=host,
        gpus=list(gpus),
        nvml=nvml_gpus is not None,
        gpu_stats=gpu_stats,
        pcie=acc,
        pcie_source=pcie_source,
        nvlink=nvlink,
        nvlink_source=nvlink_source,
        ib_ports=ib_ports,
        ethtool=run,
    )
    sources.gpm, sources.gpm_status = open_gpm(
        cfg, stack, nvml_module, nvml_gpus, sources.gpus, nvml_error
    )
    if cfg.slow_every:
        open_stress_sources(sources, cfg, nvml_gpus, run)
    return sources


def open_gpm(
    cfg: AgentConfig,
    stack: ExitStack,
    nvml_module: Any,
    nvml_gpus: Optional[NvmlGpus],
    gpus: Sequence[GpuDesc],
    nvml_error: Optional[str] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[Optional[NvmlGpm], Dict[int, str]]:
    """Probe NVML GPM on every GPU: (sampler or None, GPU -> status).

    Register this after ``nvml_gpus.close`` so that ``stack`` frees the GPM
    samples before nvmlShutdown. Unavailability is data, never an error.
    """
    if not cfg.gpm:
        return None, {g.index: "disabled" for g in gpus}
    if nvml_gpus is None:
        reason = "no-pynvml" if nvml_module is None else nvml_error or "no-nvml"
        return None, {g.index: reason for g in gpus}
    gpm = NvmlGpm(nvml_module, clock)
    status = gpm.open(nvml_gpus.handles)
    stack.callback(gpm.close)
    logger.info("GPM: %s", ", ".join("%d=%s" % item for item in sorted(status.items())))
    return (gpm if gpm.enabled else None), status


def open_stress_sources(
    sources: Sources,
    cfg: AgentConfig,
    nvml_gpus: Optional[NvmlGpus],
    run: Callable[..., Optional[ToolResult]] = run_tool,
) -> None:
    """Probe the gpux / ibx / ethx sources once; failures become meta notes."""
    sources.slow_every = cfg.slow_every
    if sources.gpus:
        if nvml_gpus is not None and nvml_gpus.stress():
            sources.gpux, sources.gpux_source = nvml_gpus.stress, "nvml"
            sources.gpux_every = 1
        else:
            smi = SmiGpuStress(run)
            if smi.available:
                sources.gpux, sources.gpux_source = smi.read, "nvidia-smi"
                sources.gpux_every = cfg.slow_every
            else:
                sources.notes.append(
                    {"source": "gpux", "error": "no NVML or nvidia-smi stress fields"}
                )
    host = sources.host
    roce: List[Tuple[str, int, str]] = []
    for device, port in sources.ib_ports:
        layer = host.link_layer(device, port)
        if layer is not None:
            sources.link_layers[(device, port)] = layer
        paths = host.ibx_paths(device, port)
        if paths:
            sources.ibx.append(IbxPort(device, port, tuple(paths)))
        else:
            sources.notes.append(
                {"source": "ibx", "dev": device, "port": port, "error": "no counters"}
            )
        if layer is not None and layer.lower() == "ethernet":
            iface = host.roce_netdev(device, port)
            if iface is None:
                sources.notes.append(
                    {"source": "ethx", "dev": device, "error": "no netdev"}
                )
            else:
                roce.append((device, port, iface))
    for device, port, iface in roce:
        probe = run(["ethtool", "-S", iface], ETHTOOL_TIMEOUT_S)
        if probe is None:
            sources.notes.append({"source": "ethtool", "error": "not installed"})
            break
        if not probe.ok:
            sources.notes.append(
                {"source": "ethtool", "iface": iface, "error": "exit %d" % probe.rc}
            )
            continue
        sources.ethx.append(EthxPort(device, port, iface))


class Stopper:
    """Signal handler target: only sets a flag (async-signal safe)."""

    def __init__(self) -> None:
        self.requested = False

    def handle(self, signum: int, frame: Any) -> None:
        self.requested = True

    def stopped(self) -> bool:
        return self.requested


def parse_args(argv: Optional[Sequence[str]]) -> AgentConfig:
    parser = argparse.ArgumentParser(description="cluster_bench telemetry sampler")
    parser.add_argument("--out", required=True, help="JSONL output path")
    parser.add_argument("--node", required=True, help="logical node name")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--pcie-interval", type=float, default=0.1)
    parser.add_argument(
        "--slow-interval",
        type=float,
        default=DEFAULT_SLOW_INTERVAL,
        help="seconds between ibx/ethx samples (gpux on nvidia-smi); 0 = off",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--control-interface", default="wg0")
    parser.add_argument("--tag", default=None, help="provenance label for meta")
    parser.add_argument(
        "--no-gpm",
        action="store_true",
        help="do not sample NVML GPM metrics (e.g. while a profiler runs)",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.pcie_interval <= 0:
        parser.error("--interval and --pcie-interval must be positive")
    if args.pcie_interval > args.interval:
        parser.error("--pcie-interval must not exceed --interval")
    if args.slow_interval < 0 or 0 < args.slow_interval < args.interval:
        parser.error("--slow-interval must be 0 (off) or >= --interval")
    if args.duration is not None and args.duration < 0:
        parser.error("--duration must be >= 0")
    return AgentConfig(
        out=args.out,
        node=args.node,
        interval=args.interval,
        pcie_interval=args.pcie_interval,
        duration=args.duration,
        control_interface=args.control_interface or None,
        tag=args.tag,
        slow_interval=args.slow_interval,
        gpm=not args.no_gpm,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s telemetry_agent %(levelname)s %(message)s",
    )
    stopper = Stopper()
    signal.signal(signal.SIGTERM, stopper.handle)
    signal.signal(signal.SIGINT, stopper.handle)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    with ExitStack() as stack:
        out = stack.enter_context(open(cfg.out, "w", encoding="utf-8"))
        sources = open_sources(cfg, stack)
        ticks = run_loop(cfg, sources, out, stopper.stopped, socket.gethostname())
    logger.info("stopped after %d samples", ticks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
