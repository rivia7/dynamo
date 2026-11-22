# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ``agents/telemetry_agent.py`` on every node around a benchmark point.

``TelemetryCollector.start`` launches one detached agent per node (``nohup``,
pid file, output under ``remote_dir``); ``stop`` sends SIGTERM, waits up to
``stop_timeout_s`` for the agent's final flush, falls back to SIGKILL, then
copies each node's JSONL into ``local_dir``. ``load_samples`` parses one
JSONL file into the ``TelemetrySamples`` dataclasses consumed by
``domain/bandwidth.py``; the stress kinds (``gpux``, ``ibx``, ``ethx``,
``note``) become ``TelemetrySamples.stress`` for ``domain/loadstress.py``
and the NVML GPM kind (``gpm``, plus ``note`` records whose source is
"gpm") becomes ``TelemetrySamples.gpm`` for ``domain/gpuactivity.py``
(both ``None`` for files of older agents). ``load_service_state``
summarizes the AIPerf server-metrics export of a point for
``domain/servicestate.py``.

A missing or unreachable node never aborts the point: its file is simply
absent from ``stop``'s result, and the bandwidth analysis reports the gap as
reduced coverage. Malformed stress and GPM records are skipped (and counted
in ``StressSamples.notes`` / ``GpmSamples.notes``) so they can never cost
the bandwidth evidence.
"""

from __future__ import annotations

import json
import logging
import math
import posixpath
import shlex
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from benchmarks.lium_cluster_bench.domain.bandwidth import (
    CounterSeries,
    GpuSeries,
    PcieSeries,
    ScalarSeries,
    TelemetryGpu,
    TelemetryMeta,
    TelemetrySamples,
)
from benchmarks.lium_cluster_bench.domain.gpuactivity import (
    GPM_FIELDS,
    GpmSamples,
    GpmSeries,
)
from benchmarks.lium_cluster_bench.domain.loadstress import (
    CounterTable,
    GpuStressSeries,
    StressSamples,
)
from benchmarks.lium_cluster_bench.domain.servicestate import (
    EndpointDescriptor,
    ServiceState,
    summarize_service_state,
)
from benchmarks.lium_cluster_bench.ports import Cluster, CommandError

logger = logging.getLogger(__name__)

__all__ = [
    "TelemetryCollector",
    "TelemetryFormatError",
    "TelemetryHandle",
    "TelemetrySamples",
    "load_samples",
    "load_service_state",
    "parse_samples",
]

POLL_S = 0.1
DEFAULT_SLOW_INTERVAL_S = 5.0
STRESS_KINDS = frozenset({"gpux", "ibx", "ethx", "note"})
# Meta keys written only by agents that know the stress kinds.
STRESS_META_KEYS = ("slow_interval", "gpux_source", "ibx_ports", "ethx_ifaces")
GPM_KIND = "gpm"
GPM_SOURCE = "gpm"  # ``note`` records about GPM carry this source
# Meta keys written only by agents that know the gpm kind.
GPM_META_KEYS = ("gpm", "gpm_metrics")
SERVER_METRICS_JSON = "server_metrics_export.json"
SERVER_METRICS_JSONL = "server_metrics_export.jsonl"

# Bash helpers shared by the start and stop scripts. ``cb_alive`` treats a
# zombie as gone (a container PID 1 may never reap it); ``cb_stop`` only
# signals a pid whose command line still contains the agent's output path,
# so a recycled pid is never killed, and never signals a process group (the
# agent may share the runner's).
_SHELL_FUNCTIONS = r"""
cb_alive() {
  kill -0 "$1" 2>/dev/null || return 1
  local state
  state=$(sed -n 's/^.*) \([A-Za-z]\) .*$/\1/p' "/proc/$1/stat" 2>/dev/null \
    || ps -o state= -p "$1" 2>/dev/null || true)
  case "$state" in Z*) return 1 ;; esac
  return 0
}
cb_stop() {
  local pid cmd i
  pid=$(cat "$1" 2>/dev/null || true)
  case "$pid" in '' | *[!0-9]*) echo absent; return 0 ;; esac
  if ! cb_alive "$pid"; then echo exited; return 0; fi
  cmd=$({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null \
    || ps -ww -o command= -p "$pid" 2>/dev/null || true)
  case "$cmd" in *"$2"*) ;; *) echo foreign; return 0 ;; esac
  kill -TERM "$pid" 2>/dev/null || true
  i=0
  while cb_alive "$pid" && [ "$i" -lt "$3" ]; do sleep 0.1; i=$((i + 1)); done
  if cb_alive "$pid"; then
    pkill -KILL -P "$pid" 2>/dev/null || true  # e.g. its nvidia-smi dmon child
    kill -KILL "$pid" 2>/dev/null || true
    echo killed
  else echo stopped; fi
}
"""


class TelemetryFormatError(ValueError):
    """A telemetry JSONL file is corrupt beyond a torn final line."""


@dataclass(frozen=True)
class TelemetryHandle:
    """Where each node's agent writes; enough to stop it after a crash."""

    tag: str
    remote_dir: str
    nodes: tuple[str, ...]
    pids: Mapping[str, int] = field(default_factory=dict)

    def out_path(self, node: str) -> str:
        return posixpath.join(self.remote_dir, f"telemetry-{node}.jsonl")

    def pid_path(self, node: str) -> str:
        return posixpath.join(self.remote_dir, f"telemetry-{node}.pid")

    def log_path(self, node: str) -> str:
        return posixpath.join(self.remote_dir, f"telemetry-{node}.log")


class TelemetryCollector:
    """Starts and stops telemetry agents on cluster nodes."""

    def __init__(
        self,
        cluster: Cluster,
        python: str,
        agent_path: str,
        *,
        interval_s: float = 1.0,
        pcie_interval_s: float = 0.1,
        control_interface: str | None = "wg0",
        start_timeout_s: float = 30.0,
        stop_timeout_s: float = 15.0,
        slow_interval_s: float = DEFAULT_SLOW_INTERVAL_S,
    ):
        if interval_s <= 0 or pcie_interval_s <= 0 or pcie_interval_s > interval_s:
            raise ValueError("need 0 < pcie_interval_s <= interval_s")
        if slow_interval_s < 0 or 0 < slow_interval_s < interval_s:
            raise ValueError("need slow_interval_s == 0 (off) or >= interval_s")
        self._cluster = cluster
        self._python = python
        self.gpm = True  # NVML GPM sampling in the agent (False: --no-gpm)
        self._agent = agent_path
        self._interval = interval_s
        self._pcie_interval = pcie_interval_s
        self._control_interface = control_interface
        self._start_timeout = start_timeout_s
        self._stop_timeout = stop_timeout_s
        self._slow_interval = slow_interval_s

    def handle(
        self, nodes: Sequence[str], remote_dir: str, tag: str
    ) -> TelemetryHandle:
        """The deterministic paths for ``(nodes, remote_dir)``, without pids."""
        return TelemetryHandle(tag=tag, remote_dir=remote_dir, nodes=tuple(nodes))

    def agent_argv(self, handle: TelemetryHandle, node: str) -> list[str]:
        argv = [
            self._python,
            self._agent,
            "--out",
            handle.out_path(node),
            "--node",
            node,
            "--interval",
            f"{self._interval:g}",
            "--pcie-interval",
            f"{self._pcie_interval:g}",
            "--tag",
            handle.tag,
            "--slow-interval",
            f"{self._slow_interval:g}",
        ]
        if self._control_interface:
            argv += ["--control-interface", self._control_interface]
        if not self.gpm:
            argv.append("--no-gpm")  # e.g. while Nsight Systems owns the counters
        return argv

    def start_script(self, handle: TelemetryHandle, node: str) -> str:
        """Bash that replaces a stale agent, launches one and prints its pid.

        Exits non-zero (with the agent log on stderr) only when the agent died
        before writing its first line.
        """
        q = shlex.quote
        out, pid, log = (
            handle.out_path(node),
            handle.pid_path(node),
            handle.log_path(node),
        )
        ticks = max(1, round(self._start_timeout / POLL_S))
        return "\n".join(
            [
                "set -e",
                _SHELL_FUNCTIONS,
                f"mkdir -p {q(handle.remote_dir)}",
                f"prev=$(cb_stop {q(pid)} {q(out)} 50)",
                'case "$prev" in stopped|killed) '
                'echo "replaced a running telemetry agent ($prev)" >&2 ;; esac',
                f"rm -f {q(pid)}",
                f"nohup {shlex.join(self.agent_argv(handle, node))} "
                f"> {q(log)} 2>&1 < /dev/null &",
                "agent=$!",
                f'echo "$agent" > {q(pid)}',
                "i=0",
                f'while [ ! -s {q(out)} ] && cb_alive "$agent" && [ "$i" -lt {ticks} ]; '
                "do sleep 0.1; i=$((i + 1)); done",
                f"if [ ! -s {q(out)} ]; then",
                '  if ! cb_alive "$agent"; then',
                f'    echo "telemetry agent exited before writing {out}" >&2',
                f"    tail -n 40 {q(log)} >&2 || true",
                "    exit 1",
                "  fi",
                '  echo "telemetry agent still initialising" >&2',
                "fi",
                'echo "$agent"',
            ]
        )

    def stop_script(self, handle: TelemetryHandle, node: str) -> str:
        """Bash that stops the node's agent and prints what happened."""
        q = shlex.quote
        ticks = max(1, round(self._stop_timeout / POLL_S))
        return "\n".join(
            [
                _SHELL_FUNCTIONS,
                f"cb_stop {q(handle.pid_path(node))} {q(handle.out_path(node))} {ticks}",
            ]
        )

    def start(self, nodes: Sequence[str], remote_dir: str, tag: str) -> TelemetryHandle:
        """Launch one agent per node in parallel.

        Raises ``RuntimeError`` when any agent fails to start, after stopping
        the agents that did start (none is left running unattended).
        """
        handle = self.handle(nodes, remote_dir, tag)

        def launch(node: str) -> int | str:
            try:
                result = self._cluster.run(
                    node,
                    self.start_script(handle, node),
                    timeout=self._start_timeout + 60,
                )
            except CommandError as exc:
                return f"exit {exc.rc}: {exc.stderr.strip()[-1500:]}"
            words = result.stdout.split()
            if not words or not words[-1].isdigit():
                return f"no pid printed: {result.stdout!r}"
            if result.stderr.strip():
                logger.info("[%s] telemetry start: %s", node, result.stderr.strip())
            return int(words[-1])

        outcomes = self._cluster.fanout(list(handle.nodes), launch)
        pids = {n: v for n, v in outcomes.items() if isinstance(v, int)}
        failed = {n: v for n, v in outcomes.items() if not isinstance(v, int)}
        if failed:
            started = replace(handle, nodes=tuple(pids), pids=pids)
            self.stop(started, None)
            detail = "; ".join(f"[{n}] {e}" for n, e in sorted(failed.items()))
            raise RuntimeError(f"telemetry agent failed to start: {detail}")
        logger.info("telemetry %s started on %s", tag, ", ".join(handle.nodes))
        return replace(handle, pids=pids)

    def stop(self, handle: TelemetryHandle, local_dir: Path | None) -> dict[str, Path]:
        """Stop every agent and fetch its JSONL; returns node -> local file.

        Nodes whose agent cannot be reached or whose file cannot be copied are
        logged and left out (the analysis reports them as missing coverage).
        ``local_dir=None`` only stops the agents.
        """

        def halt(node: str) -> str:
            try:
                result = self._cluster.run(
                    node,
                    self.stop_script(handle, node),
                    timeout=self._stop_timeout + 60,
                )
            except CommandError as exc:
                logger.warning(
                    "[%s] telemetry stop failed (exit %s): %s",
                    node,
                    exc.rc,
                    exc.stderr.strip()[-500:],
                )
                return "unreachable"
            state = (result.stdout.split() or ["?"])[-1]
            if state not in ("stopped", "absent"):
                logger.warning("[%s] telemetry agent state at stop: %s", node, state)
            return state

        def fetch(node: str) -> Path | None:
            if local_dir is None:
                return None
            target = local_dir / f"telemetry-{node}.jsonl"
            remote = handle.out_path(node)
            if str(target) == remote and target.exists():
                return target  # the runner's own node wrote it in place
            try:
                self._cluster.get_file(node, remote, target)
            except (CommandError, FileNotFoundError) as exc:
                logger.warning("[%s] telemetry fetch failed: %s", node, exc)
                return None
            log_target = local_dir / f"telemetry-{node}.log"
            if str(log_target) != handle.log_path(node):
                try:
                    self._cluster.get_file(node, handle.log_path(node), log_target)
                except (CommandError, FileNotFoundError) as exc:
                    logger.info("[%s] telemetry log not fetched: %s", node, exc)
            return target

        self._cluster.fanout(list(handle.nodes), halt)
        if local_dir is None:
            return {}
        local_dir.mkdir(parents=True, exist_ok=True)
        fetched = self._cluster.fanout(list(handle.nodes), fetch)
        return {node: path for node, path in fetched.items() if path is not None}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _meta(record: Mapping[str, Any]) -> TelemetryMeta:
    gpus = tuple(
        TelemetryGpu(
            index=int(g["index"]),
            pci_bus_id=g.get("pci_bus_id"),
            name=str(g.get("name", "")),
        )
        for g in record.get("gpus", ())
    )
    hcas = tuple((str(h["name"]), int(h["port"])) for h in record.get("hcas", ()))
    return TelemetryMeta(
        node=str(record["node"]),
        hostname=str(record.get("hostname", "")),
        t=float(record["t"]),
        ib_unit_bytes=int(record.get("ib_unit_bytes", 4)),
        nvml=bool(record.get("nvml", False)),
        nvlink_source=record.get("nvlink_source"),
        gpus=gpus,
        hcas=hcas,
        interval_s=float(record.get("interval", 1.0)),
        pcie_interval_s=float(record.get("pcie_interval", 0.1)),
        pcie_source=record.get("pcie_source"),
        control_interface=record.get("control_interface"),
        tag=record.get("tag"),
    )


def _sorted_columns(rows: list[tuple]) -> list[tuple]:
    rows.sort(key=lambda r: r[0])
    return list(zip(*rows)) if rows else []


def _counter_series(rows: list[tuple[float, int, int]]) -> CounterSeries:
    t, tx, rx = _sorted_columns(rows)
    return CounterSeries(t=tuple(t), tx=tuple(tx), rx=tuple(rx))


@dataclass
class _Rows:
    ib: dict[tuple[str, int], list] = field(default_factory=lambda: defaultdict(list))
    pcie: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    nvlink: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    gpu: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    net: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    cpu: list[tuple[float, float]] = field(default_factory=list)

    def add(self, kind: Any, record: Mapping[str, Any]) -> None:
        """Append one data record; raises KeyError/TypeError/ValueError."""
        t = float(record["t"])
        if kind == "ib":
            self.ib[(str(record["dev"]), int(record["port"]))].append(
                (t, int(record["tx"]), int(record["rx"]))
            )
        elif kind == "pcie":
            self.pcie[int(record["gpu"])].append(
                (
                    t,
                    float(record["tx_mean_Bps"]),
                    float(record["tx_max_Bps"]),
                    float(record["rx_mean_Bps"]),
                    float(record["rx_max_Bps"]),
                    int(record["n"]),
                )
            )
        elif kind == "nvlink":
            self.nvlink[int(record["gpu"])].append(
                (t, int(record["tx"]), int(record["rx"]))
            )
        elif kind == "gpu":
            self.gpu[int(record["gpu"])].append(
                (
                    t,
                    record.get("util"),
                    record.get("mem_used_mib"),
                    record.get("power_w"),
                )
            )
        elif kind == "cpu":
            self.cpu.append((t, float(record["util"])))
        elif kind == "net":
            self.net[str(record["iface"])].append(
                (t, int(record["tx"]), int(record["rx"]))
            )
        # Unknown kinds come from newer agents and are ignored.


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _counter_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"counters must be an object, got {type(raw).__name__}")
    out: dict[str, int] = {}
    for name, value in raw.items():
        number = _opt_int(value)
        if number is not None:
            out[str(name)] = number
    return out


def _note_text(record: Mapping[str, Any]) -> str:
    where = record.get("iface") or record.get("dev")
    text = f"{record.get('source', 'stress')}: {record.get('error', 'failed')}"
    return f"{text} ({where})" if where else text


@dataclass
class _StressRows:
    """Rows of the stress kinds; lenient (a bad record is skipped, counted)."""

    ibx: dict[tuple[str, int], list] = field(default_factory=lambda: defaultdict(list))
    gpux: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    ethx: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    eth_hca: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: int = 0
    seen: bool = False

    def add(self, kind: str, record: Mapping[str, Any]) -> None:
        try:
            t = float(record["t"])
            if kind == "ibx":
                key = (str(record["dev"]), int(record["port"]))
                self.ibx[key].append((t, _counter_map(record["c"])))
            elif kind == "ethx":
                iface = str(record["iface"])
                self.ethx[iface].append((t, _counter_map(record["c"])))
                if record.get("dev"):
                    self.eth_hca[iface] = str(record["dev"])
            elif kind == "gpux":
                self.gpux[int(record["gpu"])].append(
                    (
                        t,
                        _opt_float(record.get("sm_mhz")),
                        _opt_float(record.get("mem_mhz")),
                        _opt_float(record.get("temp_c")),
                        _opt_float(record.get("power_w")),
                        _opt_int(record.get("energy_mj")),
                        _opt_int(record.get("reasons")),
                    )
                )
            elif kind == "note":
                self.notes.append(_note_text(record))
        except (KeyError, TypeError, ValueError):
            self.skipped += 1
            return
        self.seen = True


def _meta_items(meta: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """The object entries of a meta list; anything malformed is ignored."""
    value = meta.get(key)
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _counter_table(rows: list[tuple[float, dict[str, int]]]) -> CounterTable:
    rows = sorted(rows, key=lambda r: r[0])
    names = sorted({name for _, counters in rows for name in counters})
    return CounterTable(
        t=tuple(t for t, _ in rows),
        values={n: tuple(c.get(n) for _, c in rows) for n in names},
    )


def _stress_samples(meta: Mapping[str, Any], rows: _StressRows) -> StressSamples | None:
    """``None`` for files of agents that predate the stress kinds."""
    if not rows.seen and not any(key in meta for key in STRESS_META_KEYS):
        return None
    link_layers: dict[tuple[str, int], str] = {}
    for item in _meta_items(meta, "link_layers"):
        try:
            link_layers[(str(item["name"]), int(item["port"]))] = str(
                item["link_layer"]
            )
        except (KeyError, TypeError, ValueError):
            continue
    eth_hca: dict[str, str] = {}
    for item in _meta_items(meta, "ethx_ifaces"):
        if item.get("iface") and item.get("name"):
            eth_hca[str(item["iface"])] = str(item["name"])
    eth_hca.update(rows.eth_hca)
    notes = [_note_text(n) for n in _meta_items(meta, "notes")]
    notes += rows.notes
    if rows.skipped:
        notes.append(f"{rows.skipped} malformed stress record(s) skipped")
    gpux: dict[int, GpuStressSeries] = {}
    for gpu, items in rows.gpux.items():
        t, sm, mem, temp, power, energy, reasons = zip(
            *sorted(items, key=lambda r: r[0])
        )
        gpux[gpu] = GpuStressSeries(
            t=t,
            sm_mhz=sm,
            mem_mhz=mem,
            temp_c=temp,
            power_w=power,
            energy_mj=energy,
            reasons=reasons,
        )
    source = meta.get("gpux_source")
    return StressSamples(
        node=str(meta.get("node", "")),
        slow_interval_s=_opt_float(meta.get("slow_interval")),
        gpux_source=source if isinstance(source, str) else None,
        link_layers=link_layers,
        ibx={key: _counter_table(items) for key, items in rows.ibx.items()},
        gpux=gpux,
        ethx={key: _counter_table(items) for key, items in rows.ethx.items()},
        eth_hca=eth_hca,
        notes=tuple(dict.fromkeys(notes)),
    )


def _finite(value: Any) -> float:
    number = _opt_float(value)
    if number is None:
        raise ValueError(f"not a finite number: {value!r}")
    return number


@dataclass
class _GpmRows:
    """Rows of the gpm kind and gpm notes; lenient (bad entries skipped)."""

    rows: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    failures: dict[int, tuple[float, str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: int = 0
    seen: bool = False

    def add(self, kind: str, record: Mapping[str, Any]) -> None:
        try:
            t = _finite(record["t"])
            if kind == GPM_KIND:
                entries = record["g"]
                if not isinstance(entries, Mapping):
                    raise TypeError("g must be an object")
            else:
                self._note(t, record)
                entries = {}
        except (KeyError, TypeError, ValueError):
            self.skipped += 1
            return
        self.seen = True
        for key, entry in entries.items():
            try:
                self._entry(t, key, entry)
            except (TypeError, ValueError):
                self.skipped += 1

    def _entry(self, t: float, key: Any, entry: Any) -> None:
        gpu = int(key)
        if not isinstance(entry, Mapping):
            raise TypeError("a GPU entry must be an object")
        dt = _opt_float(entry.get("dt_s"))
        values: dict[str, float] = {}
        for name in GPM_FIELDS:
            value = _opt_float(entry.get(name))
            if value is not None:
                values[name] = value
        self.rows[gpu].append((t, dt if dt is not None and dt > 0 else None, values))

    def _note(self, t: float, record: Mapping[str, Any]) -> None:
        gpu = _opt_int(record.get("gpu"))
        if gpu is None:
            raise ValueError("gpm note without a GPU index")
        error = str(record.get("error", "failed"))
        metrics = record.get("metrics")
        if isinstance(metrics, (list, tuple)) and metrics:
            names = ", ".join(str(m) for m in metrics)
            self.notes.append(f"gpu{gpu}: {names} unavailable ({error})")
        else:
            self.failures.setdefault(gpu, (t, error))


def _gpm_series(items: list[tuple[float, float | None, dict[str, float]]]) -> GpmSeries:
    items = sorted(items, key=lambda r: r[0])
    names = [n for n in GPM_FIELDS if any(n in values for _, _, values in items)]
    return GpmSeries(
        t=tuple(t for t, _, _ in items),
        dt_s=tuple(dt for _, dt, _ in items),
        values={n: tuple(values.get(n) for _, _, values in items) for n in names},
    )


def _gpm_samples(meta: Mapping[str, Any], rows: _GpmRows) -> GpmSamples | None:
    """``None`` for files of agents that predate the gpm kind."""
    if not rows.seen and not any(key in meta for key in GPM_META_KEYS):
        return None
    status: dict[int, str] = {}
    raw_status = meta.get("gpm")
    if isinstance(raw_status, Mapping):
        for key, value in raw_status.items():
            try:
                status[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    raw_metrics = meta.get("gpm_metrics")
    metrics = (
        tuple(m for m in raw_metrics if isinstance(m, str))
        if isinstance(raw_metrics, (list, tuple))
        else ()
    )
    notes = list(rows.notes)
    if rows.skipped:
        notes.append(f"{rows.skipped} malformed gpm record(s) skipped")
    interval = _opt_float(meta.get("interval"))
    return GpmSamples(
        node=str(meta.get("node", "")),
        interval_s=interval if interval is not None and interval > 0 else None,
        status=status,
        metrics=metrics,
        series={gpu: _gpm_series(items) for gpu, items in sorted(rows.rows.items())},
        failures=dict(rows.failures),
        notes=tuple(dict.fromkeys(notes)),
    )


def parse_samples(text: str, source: str = "<telemetry>") -> TelemetrySamples:
    """Parse agent JSONL. Only a torn final line is tolerated (and skipped).

    Stress kinds are parsed leniently into ``TelemetrySamples.stress`` and
    the gpm kind (with its notes) into ``TelemetrySamples.gpm``.
    """
    lines = [(i + 1, line) for i, line in enumerate(text.split("\n")) if line.strip()]
    meta: TelemetryMeta | None = None
    meta_record: Mapping[str, Any] = {}
    rows = _Rows()
    stress_rows = _StressRows()
    gpm_rows = _GpmRows()
    for position, (number, line) in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if position == len(lines) - 1:
                logger.info("%s: skipping torn final line %d", source, number)
                break
            raise TelemetryFormatError(f"{source}:{number}: not JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise TelemetryFormatError(f"{source}:{number}: not a JSON object")
        kind = record.get("k")
        if (meta is None) != (kind == "meta"):
            problem = "second meta record" if meta else f"first record is {kind!r}"
            raise TelemetryFormatError(f"{source}:{number}: {problem}, not one meta")
        try:
            if meta is None:
                meta = _meta(record)
                meta_record = record
            elif kind == GPM_KIND or (
                kind == "note" and record.get("source") == GPM_SOURCE
            ):
                gpm_rows.add(kind, record)
            elif kind in STRESS_KINDS:
                stress_rows.add(kind, record)
            else:
                rows.add(kind, record)
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetryFormatError(
                f"{source}:{number}: bad {kind!r} record: {exc!r}"
            ) from exc
    if meta is None:
        raise TelemetryFormatError(f"{source}: no meta record")
    stress = _stress_samples(meta_record, stress_rows)
    gpm = _gpm_samples(meta_record, gpm_rows)
    ib, pcie, nvlink, gpu, net, cpu = (
        rows.ib,
        rows.pcie,
        rows.nvlink,
        rows.gpu,
        rows.net,
        rows.cpu,
    )
    pcie_series = {}
    for index, rows in pcie.items():
        t, tx_mean, tx_max, rx_mean, rx_max, n = _sorted_columns(rows)
        pcie_series[index] = PcieSeries(
            t=tuple(t),
            tx_mean=tuple(tx_mean),
            tx_max=tuple(tx_max),
            rx_mean=tuple(rx_mean),
            rx_max=tuple(rx_max),
            n=tuple(n),
        )
    gpu_series = {}
    for index, rows in gpu.items():
        t, util, mem, power = _sorted_columns(rows)
        gpu_series[index] = GpuSeries(
            t=tuple(t), util=tuple(util), mem_used_mib=tuple(mem), power_w=tuple(power)
        )
    cpu_columns = _sorted_columns(cpu)
    return TelemetrySamples(
        meta=meta,
        ib={key: _counter_series(rows) for key, rows in ib.items()},
        pcie=pcie_series,
        nvlink={key: _counter_series(rows) for key, rows in nvlink.items()},
        gpu=gpu_series,
        cpu=(
            ScalarSeries(t=tuple(cpu_columns[0]), v=tuple(cpu_columns[1]))
            if cpu_columns
            else ScalarSeries()
        ),
        net={key: _counter_series(rows) for key, rows in net.items()},
        stress=stress,
        gpm=gpm,
    )


def load_samples(path: Path) -> TelemetrySamples:
    """Read and parse one node's telemetry JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return parse_samples(text, source=str(path))


# --------------------------------------------------------------------------
# Server metrics (AIPerf --server-metrics export)
# --------------------------------------------------------------------------


def _newest(root: Path, name: str) -> Path | None:
    found = sorted(root.rglob(name)) if root.is_dir() else []
    return found[-1] if found else None


def _lines(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        yield from fh


def load_service_state(
    aiperf_dir: Path,
    endpoints: Sequence[EndpointDescriptor],
    *,
    window: tuple[float, float] | None = None,
) -> ServiceState:
    """Summarize one point's ``server_metrics_export.json`` / ``.jsonl``.

    ``aiperf_dir`` is the point's AIPerf artifact dir (searched recursively),
    ``endpoints`` come from ``servicestate.endpoint_descriptors(plan.containers)``
    and ``window`` is the (start_s, end_s) used for the bandwidth analysis.
    Never raises on missing or corrupt files: the state carries notes. The
    JSONL is streamed (it can reach hundreds of MB at 3 scrapes/s).
    """
    notes: list[str] = []
    export: Any = None
    json_path = _newest(aiperf_dir, SERVER_METRICS_JSON)
    if json_path is not None:
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                export = json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            notes.append(f"{json_path.name} unreadable: {exc}")
    jsonl_path = _newest(aiperf_dir, SERVER_METRICS_JSONL)
    lines = _lines(jsonl_path) if jsonl_path is not None else None
    try:
        return summarize_service_state(
            export, lines, endpoints, window=window, notes=notes
        )
    except OSError as exc:
        name = jsonl_path.name if jsonl_path is not None else SERVER_METRICS_JSONL
        notes.append(f"{name} unreadable: {exc}")
        return summarize_service_state(
            export, None, endpoints, window=window, notes=notes
        )
