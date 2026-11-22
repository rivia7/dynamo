# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for agents/telemetry_agent.py and adapters/telemetry.py."""

from __future__ import annotations

import io
import json
import logging
import shlex
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.lium_cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.lium_cluster_bench.adapters.telemetry import (
    TelemetryCollector,
    TelemetryFormatError,
    load_samples,
    parse_samples,
)
from benchmarks.lium_cluster_bench.agents import telemetry_agent as agent
from benchmarks.lium_cluster_bench.domain.bandwidth import counter_rates, summarize
from benchmarks.lium_cluster_bench.ports import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry"
KIB = 1024
MIB = 1024 * 1024
NVML_SCOPE_ALL = agent.NVML_SCOPE_ALL_LINKS


def fixture(name: str) -> str:
    """Recorded command output, minus the SPDX header the repo requires."""
    text = (FIXTURES / name).read_text()
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not line.startswith("# SPDX-")
    )


# ----------------------------------------------------------------- parsers


def test_proc_stat_cpu_util() -> None:
    t0 = agent.parse_proc_stat(fixture("proc_stat_t0.txt"))
    t1 = agent.parse_proc_stat(fixture("proc_stat_t1.txt"))
    assert t0 is not None and t1 is not None
    assert t1[1] - t0[1] == 6000  # user +2000, system +1000, idle +3000
    assert agent.cpu_util(t0, t1) == pytest.approx(0.5)
    assert agent.cpu_util(None, t1) is None
    assert agent.cpu_util(t1, t1) is None
    assert agent.parse_proc_stat("intr 1 2 3\n") is None


def test_parse_nvlink_counters_sums_links() -> None:
    counters = agent.parse_nvlink_counters(fixture("nvlink_gt_h200.txt"))
    link_sum = sum(range(18))
    assert counters == {
        0: ((18 * 1_000_000 + link_sum) * KIB, (18 * 2_000_000 + link_sum) * KIB),
        1: ((18 * 2_000_000 + link_sum) * KIB, (18 * 4_000_000 + link_sum) * KIB),
    }


def test_parse_nvlink_counters_units_and_inactive_gpus() -> None:
    text = (
        "GPU 0: NVIDIA B200 (UUID: GPU-1)\n"
        "\t Link 0: Data Tx: 2 MiB\n"
        "\t Link 0: Data Rx: 3 GiB\n"
        "GPU 1: NVIDIA B200 (UUID: GPU-2)\n"
        "\t Link 0: <inactive>\n"
    )
    assert agent.parse_nvlink_counters(text) == {0: (2 * MIB, 3 * 1024 * MIB)}
    assert agent.parse_nvlink_counters("") == {}


def test_pump_dmon_reads_repeated_headers() -> None:
    acc = agent.PcieAccumulator()
    added = agent.pump_dmon(io.StringIO(fixture("dmon_pcie.txt")), acc)
    assert added == 3  # GPU 2 reported "-" (not supported)
    drained = acc.drain()
    assert drained[0]["n"] == 2
    assert drained[0]["tx_mean_Bps"] == round((9876 + 11000) / 2 * MIB)
    assert drained[0]["rx_max_Bps"] == 15000 * MIB
    assert drained[1]["tx_max_Bps"] == 0
    assert 2 not in drained


def test_dmon_sample_needs_header_columns() -> None:
    columns = agent.parse_dmon_header("# gpu  rxpci  txpci\n")
    assert columns == ["gpu", "rxpci", "txpci"]
    assert agent.parse_dmon_header("# Idx   MB/s   MB/s\n") is None
    assert agent.parse_dmon_sample("    3     10     20\n", columns) == (
        3,
        20 * MIB,
        10 * MIB,
    )
    assert agent.parse_dmon_sample("    3     10\n", columns) is None


def test_parse_smi_gpu_list_and_stats() -> None:
    gpus = agent.parse_gpu_list(
        "0, 00000000:18:00.0, NVIDIA H200\n1, 00000000:2A:00.0, NVIDIA H200\n"
    )
    assert gpus == [
        agent.GpuDesc(0, "0000:18:00.0", "NVIDIA H200"),
        agent.GpuDesc(1, "0000:2a:00.0", "NVIDIA H200"),
    ]
    stats = agent.parse_gpu_stats("0, 87, 131072, 612.40\n1, [N/A], 0, [N/A]\nbad\n")
    assert stats == {
        0: agent.GpuStat(util=87.0, mem_used_mib=131072.0, power_w=612.4),
        1: agent.GpuStat(util=None, mem_used_mib=0.0, power_w=None),
    }


def test_pcie_accumulator_aggregates_and_resets() -> None:
    acc = agent.PcieAccumulator()
    for tx, rx in ((10e9, 1e9), (30e9, 3e9), (20e9, 2e9)):
        acc.add(4, tx, rx)
    assert acc.drain() == {
        4: {
            "tx_mean_Bps": 20_000_000_000,
            "tx_max_Bps": 30_000_000_000,
            "rx_mean_Bps": 2_000_000_000,
            "rx_max_Bps": 3_000_000_000,
            "n": 3,
        }
    }
    assert acc.drain() == {}


# ----------------------------------------------------------------- NVML


class FakeNVMLError(Exception):
    pass


def fake_nvml(*, init_error: bool = False, tuple_scopes: bool = True, fields=None):
    """A pynvml stand-in; ``fields`` maps constant name -> (id, value_fn).

    ``value_fn(handle, scope)`` returns (nvmlReturn, value).
    """
    fields = fields or {}
    by_id = dict(fields.values())

    def init() -> None:
        if init_error:
            raise FakeNVMLError("NVML Shared Library Not Found")

    def field_values(handle, request):
        out = []
        for item in request:
            if not tuple_scopes and isinstance(item, tuple):
                raise TypeError("an integer is required")
            field_id, scope = item
            ret, value = by_id[field_id](handle, scope)
            out.append(
                SimpleNamespace(
                    nvmlReturn=ret, valueType=3, value=SimpleNamespace(ullVal=value)
                )
            )
        return out

    def power(handle):
        if handle == 1:
            raise FakeNVMLError("Not Supported")
        return 612400

    module = SimpleNamespace(
        NVMLError=FakeNVMLError,
        NVML_PCIE_UTIL_TX_BYTES=0,
        NVML_PCIE_UTIL_RX_BYTES=1,
        NVML_NVLINK_MAX_LINKS=18,
        nvmlInit=init,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: b"NVIDIA H200",
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(
            busId=b"00000000:%02X:00.0" % (0x18 + 0x11 * handle)
        ),
        nvmlDeviceGetPcieThroughput=lambda handle, counter: (
            (1000 + handle) if counter == 0 else (2000 + handle)
        ),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=87, memory=40),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=131072 * MIB),
        nvmlDeviceGetPowerUsage=power,
        nvmlDeviceGetFieldValues=field_values,
    )
    for name, (field_id, _) in fields.items():
        setattr(module, name, field_id)
    return module


def legacy_throughput_fields():
    def tx(handle, scope):
        return (0, (handle + 1) * 1000) if scope == NVML_SCOPE_ALL else (2, 0)

    def rx(handle, scope):
        return (0, (handle + 1) * 2000) if scope == NVML_SCOPE_ALL else (2, 0)

    return {
        "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX": (138, tx),
        "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX": (139, rx),
    }


def test_nvml_backend_reads_pcie_stats_and_legacy_nvlink() -> None:
    gpus = agent.NvmlGpus(fake_nvml(fields=legacy_throughput_fields()))
    assert gpus.gpus == [
        agent.GpuDesc(0, "0000:18:00.0", "NVIDIA H200"),
        agent.GpuDesc(1, "0000:29:00.0", "NVIDIA H200"),
    ]
    # NVML reports KB/s; the agent emits bytes/s (KB = 1024, UNVERIFIED).
    assert gpus.pcie(1) == (1001 * 1024.0, 2001 * 1024.0)
    stats = gpus.gpu_stats()
    assert stats[0] == agent.GpuStat(util=87.0, mem_used_mib=131072.0, power_w=612.4)
    assert stats[1].power_w is None  # NVMLError on that GPU only
    fields = gpus.nvlink_fields
    assert fields is not None and fields.scope == "aggregate"
    assert fields.source == (
        "nvml:NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX"
    )
    assert gpus.nvlink() == {0: (1000 * KIB, 2000 * KIB), 1: (2000 * KIB, 4000 * KIB)}


def test_resolve_nvlink_prefers_nvlink5_byte_counters_per_link() -> None:
    def per_link(scale):
        def fn(handle, scope):
            if scope == NVML_SCOPE_ALL:
                return 2, 0  # NVML_ERROR_INVALID_ARGUMENT
            return (0, scale * (scope + 1)) if scope < 4 else (3, 0)

        return fn

    fields = dict(legacy_throughput_fields())
    fields["NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES"] = (226, per_link(100))
    fields["NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES"] = (227, per_link(200))
    nvml = fake_nvml(fields=fields)
    resolved = agent.resolve_nvlink_fields(nvml, 0)
    assert resolved is not None
    assert resolved.tx_name == "NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES"
    assert resolved.scope == "per_link" and resolved.unit_bytes == 1
    assert agent.read_nvlink_pair(nvml, 0, resolved) == (1000, 2000)


@pytest.mark.parametrize("tuple_scopes", [True, False])
def test_resolve_nvlink_none_when_unsupported(tuple_scopes: bool) -> None:
    unsupported = {
        "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX": (138, lambda h, s: (3, 0)),
        "NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX": (139, lambda h, s: (3, 0)),
    }
    nvml = fake_nvml(fields=unsupported, tuple_scopes=tuple_scopes)
    assert agent.resolve_nvlink_fields(nvml, 0) is None
    assert agent.resolve_nvlink_fields(fake_nvml(), 0) is None  # constants missing


def smi_tools(extra=None):
    tools = {
        ("nvidia-smi", "--query-gpu=index,pci.bus_id,name", "--format=csv,noheader"): (
            agent.ToolResult(
                0,
                "0, 00000000:18:00.0, NVIDIA H200\n1, 00000000:29:00.0, NVIDIA H200\n",
            )
        ),
        (
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ): agent.ToolResult(0, "0, 87, 131072, 612.40\n1, 85, 131000, 600.00\n"),
    }
    tools.update(extra or {})
    return lambda argv, timeout=None: tools.get(tuple(argv))


# A stand-in for `nvidia-smi dmon` that prints a header and waits for SIGTERM.
FAKE_DMON = [
    sys.executable,
    "-c",
    "import signal; print('# gpu rxpci txpci', flush=True); signal.pause()",
]


def test_open_sources_falls_back_to_nvidia_smi(tmp_path: Path) -> None:
    run = smi_tools(
        {
            ("nvidia-smi", "nvlink", "-gt", "d"): agent.ToolResult(
                0, fixture("nvlink_gt_h200.txt")
            )
        }
    )
    cfg = agent.AgentConfig(out=str(tmp_path / "t.jsonl"), node="node0")
    with ExitStack() as stack:
        sources = agent.open_sources(
            cfg,
            stack,
            nvml_module=fake_nvml(init_error=True),
            run=run,
            root=str(tmp_path),
            dmon_argv=FAKE_DMON,
        )
        assert sources.nvml is False
        assert [g.pci_bus_id for g in sources.gpus] == ["0000:18:00.0", "0000:29:00.0"]
        assert sources.pcie_source == "nvidia-smi dmon"
        assert sources.nvlink_source == "nvidia-smi"
        assert sources.nvlink is not None and set(sources.nvlink()) == {0, 1}
        assert sources.gpu_stats()[1].power_w == 600.0
        assert sources.ib_ports == []


def test_open_sources_without_any_gpu_tooling(tmp_path: Path) -> None:
    cfg = agent.AgentConfig(out=str(tmp_path / "t.jsonl"), node="node0")
    with ExitStack() as stack:
        sources = agent.open_sources(
            cfg,
            stack,
            nvml_module=None,
            run=lambda argv, timeout=None: None,
            root=str(tmp_path),
            dmon_argv=FAKE_DMON,
        )
    assert sources.gpus == [] and sources.pcie_source is None
    assert sources.nvlink is None and sources.nvlink_source is None


# ----------------------------------------------------------------- sysfs


def write_ib_port(
    root: Path, dev: str, port: int, state: str, tx: int, rx: int
) -> None:
    base = root / "sys/class/infiniband" / dev / "ports" / str(port)
    (base / "counters").mkdir(parents=True)
    (base / "state").write_text(state + "\n")
    (base / "counters/port_xmit_data").write_text(f"{tx}\n")
    (base / "counters/port_rcv_data").write_text(f"{rx}\n")


def test_host_counters_read_fake_sysfs(tmp_path: Path) -> None:
    write_ib_port(tmp_path, "mlx5_10", 1, "4: ACTIVE", 7, 8)
    write_ib_port(tmp_path, "mlx5_2", 1, "4: ACTIVE", 20402189312000, 5)
    write_ib_port(tmp_path, "mlx5_2", 2, "1: DOWN", 0, 0)
    net = tmp_path / "sys/class/net/wg0/statistics"
    net.mkdir(parents=True)
    (net / "tx_bytes").write_text("12345\n")
    (net / "rx_bytes").write_text("67890\n")
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/stat").write_text(fixture("proc_stat_t0.txt"))
    host = agent.HostCounters(str(tmp_path))
    assert host.ib_ports() == [("mlx5_2", 1), ("mlx5_10", 1)]
    assert host.ib("mlx5_2", 1) == (20402189312000 * 4, 20)  # raw words x 4
    assert host.ib("mlx5_9", 1) is None
    assert host.net("wg0") == (12345, 67890)
    assert host.net("eth9") is None
    assert host.cpu_times() == agent.parse_proc_stat(fixture("proc_stat_t0.txt"))


# ----------------------------------------------------------------- sampling loop


class FakeTime:
    """Monotonic and wall clocks that only advance when the loop sleeps."""

    def __init__(self) -> None:
        self.mono = 100.0
        self.sleeps = 0
        self.on_sleep = None

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return 1793700000.0 + (self.mono - 100.0)

    def sleep(self, seconds: float) -> None:
        self.mono += seconds
        self.sleeps += 1
        if self.on_sleep is not None:
            self.on_sleep(self)


class FakeHost:
    """Counters growing at fixed rates with the fake clock."""

    def __init__(self, clock: FakeTime) -> None:
        self.clock = clock

    def _elapsed(self) -> float:
        return self.clock.mono - 100.0

    def ib(self, dev: str, port: int) -> tuple[int, int]:
        words = int(5e9 * self._elapsed())  # 20 GB/s in 4-byte words
        base = (1 << 40) if dev == "mlx5_0" else (1 << 41)
        return (base + words) * 4, (base + words // 2) * 4

    def cpu_times(self) -> tuple[int, int]:
        ticks = int(self._elapsed() * 1000)
        return ticks // 4, ticks

    def net(self, iface: str) -> tuple[int, int] | None:
        if iface != "wg0":
            return None
        return 1000 + int(self._elapsed() * 1e6), 2000


def fake_sources(clock: FakeTime) -> agent.Sources:
    acc = agent.PcieAccumulator()

    def sampler(fake: FakeTime) -> None:  # stands in for the per-GPU threads
        acc.add(0, 21e9, 19e9)
        acc.add(1, 1e9, 2e9)

    clock.on_sleep = sampler
    return agent.Sources(
        host=FakeHost(clock),
        gpus=[
            agent.GpuDesc(0, "0000:18:00.0", "NVIDIA H200"),
            agent.GpuDesc(1, "0000:29:00.0", "NVIDIA H200"),
        ],
        nvml=True,
        gpu_stats=lambda: {
            0: agent.GpuStat(87.0, 131072.0, 612.4),
            1: agent.GpuStat(float("nan"), None, None),
        },
        pcie=acc,
        pcie_source="nvml",
        nvlink=lambda: {0: (100 * KIB, 200 * KIB), 1: (300 * KIB, 400 * KIB)},
        nvlink_source="nvml:TX,RX",
        ib_ports=[("mlx5_0", 1), ("mlx5_1", 1)],
    )


CONTRACT_KEYS = {
    "ib": {"k", "t", "dev", "port", "tx", "rx"},
    "pcie": {
        "k",
        "t",
        "gpu",
        "tx_mean_Bps",
        "tx_max_Bps",
        "rx_mean_Bps",
        "rx_max_Bps",
        "n",
    },
    "nvlink": {"k", "t", "gpu", "tx", "rx"},
    "gpu": {"k", "t", "gpu", "util", "mem_used_mib", "power_w"},
    "cpu": {"k", "t", "util"},
    "net": {"k", "t", "iface", "tx", "rx"},
}
META_KEYS = {
    "k",
    "t",
    "node",
    "hostname",
    "ib_unit_bytes",
    "nvml",
    "nvlink_source",
    "gpus",
    "hcas",
}


def test_run_loop_writes_the_line_contract() -> None:
    clock = FakeTime()
    sources = fake_sources(clock)
    cfg = agent.AgentConfig(out="unused", node="node3", duration=3.0, tag="p1")
    out = io.StringIO()
    ticks = agent.run_loop(
        cfg,
        sources,
        out,
        lambda: False,
        "host-3",
        clock.monotonic,
        clock.time,
        clock.sleep,
    )
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    meta = records[0]
    assert ticks == 3  # two interval ticks plus the final sample at the deadline
    assert META_KEYS <= set(meta)
    assert meta["k"] == "meta" and meta["node"] == "node3" and meta["tag"] == "p1"
    assert meta["ib_unit_bytes"] == 4 and meta["nvml"] is True
    assert meta["hcas"] == [
        {"name": "mlx5_0", "port": 1},
        {"name": "mlx5_1", "port": 1},
    ]
    assert meta["gpus"][1] == {
        "index": 1,
        "pci_bus_id": "0000:29:00.0",
        "name": "NVIDIA H200",
    }
    by_kind: dict[str, list[dict]] = {}
    for record in records[1:]:
        assert set(record) == CONTRACT_KEYS[record["k"]], record
        by_kind.setdefault(record["k"], []).append(record)
    assert len(by_kind["ib"]) == 6 and len(by_kind["nvlink"]) == 6
    assert len(by_kind["gpu"]) == 6 and len(by_kind["net"]) == 3
    assert {r["gpu"] for r in by_kind["pcie"]} == {0, 1}
    assert by_kind["pcie"][0]["tx_mean_Bps"] == 21_000_000_000
    # One sub-sample per 50 ms sleep; float rounding of 20 x 0.05 s may add one.
    assert by_kind["pcie"][0]["n"] in (20, 21)
    assert by_kind["gpu"][1]["util"] is None  # NaN never reaches the file
    assert by_kind["cpu"][0]["util"] == pytest.approx(0.25, abs=0.01)

    # The loader reads what the agent writes, and the rates come out right.
    samples = parse_samples(out.getvalue())
    series = samples.ib[("mlx5_0", 1)]
    assert len(series.t) == 3
    rates = counter_rates(series.points("tx"), unit_bytes=samples.meta.ib_unit_bytes)
    assert summarize(rates).mean_gbs == pytest.approx(20.0, rel=1e-6)
    assert samples.nvlink[1].tx == (300 * KIB,) * 3
    assert samples.net["wg0"].rx == (2000,) * 3


def test_run_loop_stops_promptly_on_signal_flag() -> None:
    clock = FakeTime()
    sources = fake_sources(clock)
    stopper = agent.Stopper()
    feed = clock.on_sleep

    def on_sleep(fake: FakeTime) -> None:
        feed(fake)
        if fake.sleeps == 30:  # 1.5 s in: SIGTERM arrives
            stopper.handle(signal.SIGTERM, None)

    clock.on_sleep = on_sleep
    cfg = agent.AgentConfig(out="unused", node="node0")  # no duration
    out = io.StringIO()
    ticks = agent.run_loop(
        cfg,
        sources,
        out,
        stopper.stopped,
        "h",
        clock.monotonic,
        clock.time,
        clock.sleep,
    )
    assert ticks == 2  # the 1 s tick and the final sample
    assert clock.sleeps == 30
    kinds = [json.loads(line)["k"] for line in out.getvalue().splitlines()]
    assert kinds[0] == "meta" and kinds.count("ib") == 4


def test_agent_cli_rejects_bad_intervals() -> None:
    with pytest.raises(SystemExit):
        agent.parse_args(["--out", "x", "--node", "n", "--pcie-interval", "2"])
    with pytest.raises(SystemExit):
        agent.parse_args(["--out", "x", "--node", "n", "--interval", "0"])
    cfg = agent.parse_args(["--out", "x", "--node", "n", "--control-interface", ""])
    assert cfg.control_interface is None and cfg.duration is None


def test_agent_script_runs_standalone(tmp_path: Path) -> None:
    """Isolated interpreter, no repo on sys.path; --duration 0 exits at once."""
    out = tmp_path / "sub" / "telemetry-n0.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(agent.__file__)),
            "--out",
            str(out),
            "--node",
            "n0",
            "--duration",
            "0",
            "--tag",
            "smoke",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    samples = load_samples(out)
    assert samples.meta.node == "n0" and samples.meta.tag == "smoke"
    assert samples.meta.interval_s == 1.0


# ----------------------------------------------------------------- collector

PIDS = {"node0": 4242, "node1": 5151}


def responder(node: str, command: str) -> CommandResult:
    if "nohup" in command:
        return CommandResult(node, command, 0, f"{PIDS[node]}\n", "")
    if "cb_stop" in command:
        return CommandResult(node, command, 0, "stopped\n", "")
    return CommandResult(node, command, 0, "", "")


def collector(cluster: RecordingExecutor) -> TelemetryCollector:
    return TelemetryCollector(
        cluster,
        "/root/cluster_bench/venv/bin/python",
        "/root/cb/src/agents/telemetry_agent.py",
    )


def test_start_launches_quoted_detached_agents() -> None:
    cluster = RecordingExecutor(("node0", "node1"), responder=responder)
    remote = "/root/cluster_bench/experiments/e 1/points/abc/telemetry"
    handle = collector(cluster).start(["node0", "node1"], remote, "abc")
    assert handle.pids == PIDS and handle.nodes == ("node0", "node1")
    assert handle.out_path("node1") == f"{remote}/telemetry-node1.jsonl"
    (script,) = cluster.commands_for("node1")
    assert f"mkdir -p {shlex.quote(remote)}" in script
    assert (
        "nohup /root/cluster_bench/venv/bin/python /root/cb/src/agents/telemetry_agent.py"
        in script
    )
    assert f"--out {shlex.quote(handle.out_path('node1'))}" in script
    assert "--node node1 --interval 1 --pcie-interval 0.1 --tag abc" in script
    assert "--control-interface wg0" in script
    assert f"> {shlex.quote(handle.log_path('node1'))} 2>&1 < /dev/null &" in script
    assert f"> {shlex.quote(handle.pid_path('node1'))}" in script


def test_start_failure_stops_started_agents_and_raises() -> None:
    def failing(node: str, command: str) -> CommandResult:
        if node == "node1" and "nohup" in command:
            return CommandResult(
                node, command, 1, "", "telemetry agent exited before writing"
            )
        return responder(node, command)

    cluster = RecordingExecutor(("node0", "node1"), responder=failing)
    with pytest.raises(RuntimeError, match=r"\[node1\] exit 1: telemetry agent exited"):
        collector(cluster).start(["node0", "node1"], "/tmp-not-used/tel", "abc")
    node0 = cluster.commands_for("node0")
    assert len(node0) == 2 and "nohup" not in node0[1] and "cb_stop" in node0[1]
    assert len(cluster.commands_for("node1")) == 1


def test_stop_fetches_files_and_skips_missing_nodes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cluster = RecordingExecutor(("node0", "node1"), responder=responder)
    col = collector(cluster)
    handle = col.handle(["node0", "node1"], "/remote/tel", "abc")
    cluster.files[("node0", handle.out_path("node0"))] = fixture(
        "telemetry-node0.jsonl"
    )
    cluster.files[("node0", handle.log_path("node0"))] = "agent log\n"
    with caplog.at_level(logging.WARNING):
        files = col.stop(handle, tmp_path / "local")
    assert list(files) == ["node0"]
    assert files["node0"].read_text() == fixture("telemetry-node0.jsonl")
    assert (tmp_path / "local/telemetry-node0.log").read_text() == "agent log\n"
    assert "[node1] telemetry fetch failed" in caplog.text
    stop_script = cluster.commands_for("node1")[0]
    assert f"cb_stop {shlex.quote(handle.pid_path('node1'))}" in stop_script
    assert stop_script.rstrip().endswith(" 150")  # 15 s of 0.1 s polls


def test_stop_keeps_in_place_file_on_the_runner_node(tmp_path: Path) -> None:
    cluster = RecordingExecutor(("node0",), responder=responder)
    col = collector(cluster)
    local = tmp_path / "points" / "abc" / "telemetry"
    handle = col.handle(["node0"], str(local), "abc")
    local.mkdir(parents=True)
    (local / "telemetry-node0.jsonl").write_text(fixture("telemetry-node0.jsonl"))
    files = col.stop(handle, local)  # get_file would raise: nothing recorded
    assert files == {"node0": local / "telemetry-node0.jsonl"}


@pytest.mark.parametrize("which", ["start", "stop"])
def test_generated_scripts_are_valid_bash(which: str) -> None:
    col = collector(RecordingExecutor(("node0",)))
    handle = col.handle(["node0"], "/root/x y/telemetry", "t'1")
    script = getattr(col, f"{which}_script")(handle, "node0")
    proc = subprocess.run(
        ["bash", "-n", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_collector_rejects_bad_intervals() -> None:
    with pytest.raises(ValueError):
        TelemetryCollector(
            RecordingExecutor(("node0",)),
            "py",
            "agent.py",
            interval_s=0.1,
            pcie_interval_s=1,
        )


# ----------------------------------------------------------------- loader


def test_load_samples_fixture_skips_torn_final_line() -> None:
    samples = load_samples(FIXTURES / "telemetry-node0.jsonl")
    meta = samples.meta
    assert (meta.node, meta.hostname, meta.tag) == (
        "node0",
        "lium-h200-00",
        "3f9a1c2b7d4e",
    )
    assert meta.nvml is True and meta.ib_unit_bytes == 4 and meta.pcie_source == "nvml"
    assert meta.nvlink_source.startswith("nvml:NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX")
    assert meta.hcas == (("mlx5_0", 1), ("mlx5_1", 1))
    assert [g.pci_bus_id for g in meta.gpus] == ["0000:18:00.0", "0000:29:00.0"]
    assert (meta.interval_s, meta.pcie_interval_s) == (1.0, 0.1)
    ib = samples.ib[("mlx5_0", 1)]
    assert len(ib.t) == 4  # the torn fifth ib line is dropped
    assert ib.tx[1] - ib.tx[0] == 20_000_000_000
    assert samples.pcie[1].tx_mean == (21_000_000_001.0,) * 4
    assert samples.pcie[0].n == (10,) * 4
    assert samples.nvlink[1].rx[3] - samples.nvlink[1].rx[0] == 3 * 118_000_000_000
    assert samples.gpu[1].power_w == (None,) * 4
    assert samples.cpu.v == (0.4123,) * 3
    assert samples.net["wg0"].tx[-1] == 98_765_432 + 3 * 1_200_000
    assert summarize(counter_rates(ib.points("tx"))).mean_gbs == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"k":"ib","t":1,"dev":"mlx5_0","port":1,"tx":0,"rx":0}\n', "first record"),
        (
            '{"k":"meta","t":1,"node":"n"}\nnot json\n{"k":"cpu","t":2,"util":0.1}\n',
            ":2: not JSON",
        ),
        (
            '{"k":"meta","t":1,"node":"n"}\n{"k":"meta","t":2,"node":"n"}\n',
            "second meta",
        ),
        (
            '{"k":"meta","t":1,"node":"n"}\n{"k":"ib","t":2,"dev":"mlx5_0"}\n',
            ":2: bad 'ib'",
        ),
        ('{"k":"meta","t":1,"node":"n"}\n[1, 2]\n', "not a JSON object"),
        ("", "no meta"),
    ],
)
def test_parse_samples_rejects_corruption(text: str, message: str) -> None:
    with pytest.raises(TelemetryFormatError, match=message):
        parse_samples(text, source="x.jsonl")


def test_parse_samples_ignores_unknown_kinds_and_sorts_by_time() -> None:
    text = (
        '{"k":"meta","t":1,"node":"n"}\n'
        '{"k":"dcgm","t":2,"field":1011}\n'
        '{"k":"net","t":3,"iface":"wg0","tx":30,"rx":3}\n'
        '{"k":"net","t":2,"iface":"wg0","tx":20,"rx":2}\n'
    )
    samples = parse_samples(text)
    assert samples.net["wg0"].t == (2.0, 3.0) and samples.net["wg0"].tx == (20, 30)
    assert samples.meta.interval_s == 1.0 and samples.meta.ib_unit_bytes == 4
