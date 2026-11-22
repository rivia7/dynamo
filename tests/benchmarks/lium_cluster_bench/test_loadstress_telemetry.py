# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The telemetry agent's stress kinds (gpux, ibx, ethx, note) and their loader."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.lium_cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.lium_cluster_bench.adapters.telemetry import (
    TelemetryCollector,
    load_samples,
    parse_samples,
)
from benchmarks.lium_cluster_bench.agents import telemetry_agent as agent
from benchmarks.lium_cluster_bench.domain.loadstress import analyze_point_stress
from tests.benchmarks.lium_cluster_bench.lium_fakes import fixture_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry"
GPUX_FIELDS = ("sm_mhz", "mem_mhz", "temp_c", "power_w", "energy_mj", "reasons")
STRESS_KEYS = {
    "gpux": {"k", "t", "gpu", *GPUX_FIELDS},
    "ibx": {"k", "t", "dev", "port", "c"},
    "ethx": {"k", "t", "iface", "dev", "c"},
    "note": {"k", "t", "source", "iface", "dev", "error"},
}
ETHX_KEPT = {
    "rx_xdp_drop",
    "rx_out_of_buffer",
    "tx_queue_dropped",
    "rx_discards_phy",
    "tx_discards_phy",
    "tx_pause_ctrl_phy",
    "rx_pause_ctrl_phy",
    "rx_prio3_discards",
    "rx_prio3_pause",
    "rx_prio3_pause_duration",
    "tx_prio3_pause",
    "tx_prio3_pause_duration",
    "rx_prio3_pause_transition",
    "rx_prio3_buf_discard",
    "rx_prio3_cong_discard",
    "rx_prio3_marked",
    "rx_ecn_mark",
    "tx_pause_storm_warning_events",
    "tx_pause_storm_error_events",
}


# ----------------------------------------------------------------- parsers


def test_ethtool_stats_filtered_to_port_level_congestion_counters() -> None:
    counters = agent.parse_ethtool_stats(
        fixture_text(FIXTURES / "ethtool_s_mlx5_roce.txt")
    )
    assert counters["rx_bytes"] == 37284719312251 and "NIC statistics" not in counters
    kept = agent.filter_ethx_counters(counters)
    assert set(kept) == ETHX_KEPT  # no per-queue rx0_*/tx0_*/ch0_* counters
    assert kept["rx_prio3_pause"] == 44512 and kept["rx_ecn_mark"] == 1290011
    assert agent.parse_ethtool_stats("garbage\n  name with space: 3\n") == {}


def test_smi_gpux_parses_reasons_bitmask_and_na() -> None:
    stress = agent.parse_smi_gpux(fixture_text(FIXTURES / "smi_gpux_b200.txt"), True)
    assert len(stress) == 8
    assert stress[3] == agent.GpuStress(1590.0, 3996.0, 87.0, 994.81, None, 0x24)
    assert stress[7].sm_mhz is None and stress[7].reasons == 0x1
    assert agent.parse_smi_gpux("0, 1965, 3996, 61, 987.52\n", False)[0].reasons is None
    assert agent.parse_smi_gpux("0, 1965, 3996, 61, 987.52\n", True) == {}
    assert agent.parse_mask("0x0000000000000004") == 4
    assert agent.parse_mask("[N/A]") is None and agent.parse_mask("12") == 12


# ----------------------------------------------------------------- NVML / smi


class FakeNVMLError(Exception):
    pass


def stress_nvml(*, events: bool = True, energy: bool = True) -> SimpleNamespace:
    """A pynvml stand-in with the gpux functions (GPU1 lacks energy)."""

    def total_energy(handle: int) -> int:
        if handle == 1:
            raise FakeNVMLError("Not Supported")
        return 5_000_000_000

    module = SimpleNamespace(
        NVMLError=FakeNVMLError,
        NVML_CLOCK_SM=1,
        NVML_CLOCK_MEM=2,
        NVML_TEMPERATURE_GPU=0,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: b"NVIDIA B200",
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(
            busId=b"00000000:%02X:00.0" % (0x1B + 0x28 * handle)
        ),
        nvmlDeviceGetClockInfo=lambda handle, kind: 1965 if kind == 1 else 3996,
        nvmlDeviceGetTemperature=lambda handle, sensor: 61 + handle,
        nvmlDeviceGetPowerUsage=lambda handle: 987521,
    )
    if energy:
        module.nvmlDeviceGetTotalEnergyConsumption = total_energy
    if events:
        module.nvmlDeviceGetCurrentClocksEventReasons = lambda handle: 0x4 * handle
    else:
        module.nvmlDeviceGetCurrentClocksThrottleReasons = lambda handle: 0x20
    return module


def test_nvml_stress_reads_clocks_energy_and_reasons() -> None:
    gpus = agent.NvmlGpus(stress_nvml())
    assert gpus.stress() == {
        0: agent.GpuStress(1965.0, 3996.0, 61.0, 987.52, 5_000_000_000, 0),
        1: agent.GpuStress(1965.0, 3996.0, 62.0, 987.52, None, 4),
    }
    legacy = agent.NvmlGpus(stress_nvml(events=False, energy=False)).stress()
    assert legacy[0].reasons == 0x20 and legacy[0].energy_mj is None


def test_smi_gpu_stress_probes_the_reasons_field_name() -> None:
    calls: list[str] = []

    def run(argv, timeout=None):
        calls.append(argv[1])
        if "clocks_event_reasons" in argv[1]:
            return agent.ToolResult(2, "")  # older driver: unknown field
        return agent.ToolResult(0, "0, 1965, 3996, 61, 987.52, 0x0000000000000004\n")

    smi = agent.SmiGpuStress(run)
    assert smi.available and smi.with_reasons
    assert "clocks_throttle_reasons.active" in smi.argv[1]
    assert smi.read()[0].reasons == 4
    assert len(calls) == 3  # two probes, one read
    assert not agent.SmiGpuStress(lambda argv, timeout=None: None).available


# ----------------------------------------------------------------- sysfs


def write_port(
    root: Path,
    dev: str,
    layer: str,
    counters: dict[str, int],
    hw: dict[str, int],
    ndev: str | None = None,
) -> None:
    base = root / "sys/class/infiniband" / dev / "ports" / "1"
    (base / "counters").mkdir(parents=True)
    (base / "hw_counters").mkdir()
    (base / "state").write_text("4: ACTIVE\n")
    (base / "link_layer").write_text(layer + "\n")
    for name, value in {"port_xmit_data": 10, "port_rcv_data": 20, **counters}.items():
        (base / "counters" / name).write_text(f"{value}\n")
    for name, value in hw.items():
        (base / "hw_counters" / name).write_text(f"{value}\n")
    if ndev is not None:
        (base / "gid_attrs/ndevs").mkdir(parents=True)
        (base / "gid_attrs/ndevs/0").write_text(ndev + "\n")


def fake_fabric(root: Path) -> None:
    write_port(
        root,
        "mlx5_0",
        "InfiniBand",
        {"port_xmit_wait": 1000, "port_xmit_discards": 1, "symbol_error": 0},
        {"out_of_sequence": 7},
    )
    write_port(
        root,
        "mlx5_4",
        "Ethernet",
        {"port_xmit_wait": 50},
        {"np_cnp_sent": 900, "rp_cnp_handled": 400, "np_ecn_marked_roce_packets": 950},
        ndev="ens5f0np0",
    )
    write_port(root, "mlx5_5", "Ethernet", {}, {})
    net = root / "sys/class/infiniband/mlx5_5/device/net"
    for name, dev_port in (("ens6f0np0", "0"), ("ens6f1np1", "1")):
        (net / name).mkdir(parents=True)
        (net / name / "dev_port").write_text(dev_port + "\n")


def test_host_counters_find_stress_counters_and_roce_netdevs(tmp_path: Path) -> None:
    fake_fabric(tmp_path)
    host = agent.HostCounters(str(tmp_path))
    assert host.link_layer("mlx5_0", 1) == "InfiniBand"
    assert host.link_layer("mlx5_9", 1) is None
    paths = host.ibx_paths("mlx5_4", 1)
    assert [name for name, _ in paths] == [
        "port_xmit_wait",
        "np_cnp_sent",
        "rp_cnp_handled",
        "np_ecn_marked_roce_packets",
    ]
    assert agent.read_counters(paths) == {
        "port_xmit_wait": 50,
        "np_cnp_sent": 900,
        "rp_cnp_handled": 400,
        "np_ecn_marked_roce_packets": 950,
    }
    assert agent.read_counters([("gone", str(tmp_path / "nope"))]) == {}
    assert host.roce_netdev("mlx5_4", 1) == "ens5f0np0"  # gid_attrs/ndevs/0
    assert host.roce_netdev("mlx5_5", 1) == "ens6f0np0"  # device/net + dev_port
    assert host.roce_netdev("mlx5_0", 1) is None


# ----------------------------------------------------------------- sampling


class FakeTime:
    """Monotonic and wall clocks that only advance when the loop sleeps."""

    def __init__(self) -> None:
        self.mono = 100.0

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return 1793700000.0 + (self.mono - 100.0)

    def sleep(self, seconds: float) -> None:
        self.mono += seconds


class FakeEthtool:
    """ethtool -S: counters grow with each call; ens6f0np0 fails."""

    def __init__(self, clock: FakeTime) -> None:
        self.clock = clock
        self.calls: list[str] = []
        self.broken = False

    def __call__(self, argv, timeout=None):
        iface = argv[-1]
        self.calls.append(iface)
        if iface == "ens6f0np0" or self.broken:
            return agent.ToolResult(76, "")
        grown = int(400 * (self.clock.mono - 100.0))
        text = fixture_text(FIXTURES / "ethtool_s_mlx5_roce.txt").replace(
            "rx_prio3_pause: 44512", f"rx_prio3_pause: {44512 + grown}"
        )
        return agent.ToolResult(0, text)


def stress_sources(tmp_path: Path, clock: FakeTime, ethtool: FakeEthtool):
    fake_fabric(tmp_path)
    host = agent.HostCounters(str(tmp_path))
    cfg = agent.AgentConfig(out="unused", node="node1", duration=7.0)
    gpus = agent.NvmlGpus(stress_nvml())
    sources = agent.Sources(
        host=host,
        gpus=gpus.gpus,
        nvml=True,
        gpu_stats=lambda: {},
        pcie=agent.PcieAccumulator(),
        pcie_source=None,
        nvlink=None,
        nvlink_source=None,
        ib_ports=host.ib_ports(),
        ethtool=ethtool,
    )
    agent.open_stress_sources(sources, cfg, gpus, ethtool)
    return cfg, sources


def test_open_stress_sources_probes_once_and_notes_failures(tmp_path: Path) -> None:
    clock = FakeTime()
    ethtool = FakeEthtool(clock)
    cfg, sources = stress_sources(tmp_path, clock, ethtool)
    assert sources.slow_every == 5 and cfg.slow_every == 5
    assert (sources.gpux_source, sources.gpux_every) == ("nvml", 1)
    assert [p.device for p in sources.ibx] == ["mlx5_0", "mlx5_4"]
    assert sources.link_layers[("mlx5_5", 1)] == "Ethernet"
    assert [e.iface for e in sources.ethx] == ["ens5f0np0"]
    assert sources.notes == [
        {"source": "ibx", "dev": "mlx5_5", "port": 1, "error": "no counters"},
        {"source": "ethtool", "iface": "ens6f0np0", "error": "exit 76"},
    ]
    meta = agent.meta_record(cfg, sources, "lium-b200-01", 0.0)
    assert meta["slow_interval"] == 5.0 and meta["slow_every"] == 5
    assert meta["gpux_source"] == "nvml" and meta["gpux_every"] == 1
    assert meta["ethx_ifaces"] == [{"name": "mlx5_4", "port": 1, "iface": "ens5f0np0"}]
    assert {"name": "mlx5_0", "port": 1, "link_layer": "InfiniBand"} in meta[
        "link_layers"
    ]


def test_run_loop_writes_stress_kinds_on_their_cadence(tmp_path: Path) -> None:
    clock = FakeTime()
    ethtool = FakeEthtool(clock)
    cfg, sources = stress_sources(tmp_path, clock, ethtool)
    ethtool.calls.clear()

    def break_ethtool_later(seconds: float) -> None:
        clock.sleep(seconds)
        ethtool.broken = clock.mono - 100.0 > 6.5

    out = io.StringIO()
    ticks = agent.run_loop(
        cfg,
        sources,
        out,
        lambda: False,
        "lium-b200-01",
        clock.monotonic,
        clock.time,
        break_ethtool_later,
    )
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    by_kind: dict[str, list[dict]] = {}
    for record in records[1:]:
        by_kind.setdefault(record["k"], []).append(record)
        if record["k"] in STRESS_KEYS:
            assert set(record) == STRESS_KEYS[record["k"]], record
    assert ticks == 7  # six interval ticks plus the final sample
    assert len(by_kind["gpux"]) == 2 * ticks  # every tick with NVML
    # Slow ticks: tick 0, tick 5 and the final sample.
    assert len(by_kind["ibx"]) == 2 * 3
    assert [r["dev"] for r in by_kind["ibx"][:2]] == ["mlx5_0", "mlx5_4"]
    assert by_kind["ibx"][1]["c"]["np_cnp_sent"] == 900
    assert len(by_kind["ethx"]) == 2  # the final ethtool call failed
    assert set(by_kind["ethx"][0]["c"]) == ETHX_KEPT
    (note,) = by_kind["note"]  # logged once, then the iface is skipped
    assert note["iface"] == "ens5f0np0" and note["error"] == "exit 76"
    assert ethtool.calls == ["ens5f0np0"] * 3

    samples = parse_samples(out.getvalue())
    stress = samples.stress
    assert stress is not None and stress.gpux_source == "nvml"
    assert stress.gpux[1].energy_mj == (None,) * ticks
    assert "ethtool: exit 76 (ens5f0np0)" in stress.notes
    first, last = stress.ethx["ens5f0np0"].t[0], stress.ethx["ens5f0np0"].t[-1]
    point = analyze_point_stress({"node1": stress}, (first, last))
    assert point.eths[0].pause_rx_per_s == pytest.approx(400.0, rel=0.01)


def test_slow_interval_zero_disables_stress_kinds(tmp_path: Path) -> None:
    clock = FakeTime()
    fake_fabric(tmp_path)
    host = agent.HostCounters(str(tmp_path))
    cfg = agent.parse_args(
        ["--out", "x", "--node", "n", "--slow-interval", "0", "--duration", "2"]
    )
    assert cfg.slow_every == 0
    sources = agent.Sources(
        host=host,
        gpus=[],
        nvml=False,
        gpu_stats=lambda: {},
        pcie=agent.PcieAccumulator(),
        pcie_source=None,
        nvlink=None,
        nvlink_source=None,
        ib_ports=host.ib_ports(),
    )
    out = io.StringIO()
    agent.run_loop(
        cfg, sources, out, lambda: False, "h", clock.monotonic, clock.time, clock.sleep
    )
    kinds = {json.loads(line)["k"] for line in out.getvalue().splitlines()}
    assert kinds == {"meta", "ib"}


def test_agent_cli_slow_interval_validation() -> None:
    base = ["--out", "x", "--node", "n"]
    assert agent.parse_args(base).slow_every == 5
    assert agent.parse_args(base + ["--interval", "2"]).slow_every == 2
    with pytest.raises(SystemExit):
        agent.parse_args(base + ["--slow-interval", "0.5"])
    with pytest.raises(SystemExit):
        agent.parse_args(base + ["--slow-interval", "-1"])


def test_collector_passes_slow_interval() -> None:
    cluster = RecordingExecutor(("node0",))
    collector = TelemetryCollector(cluster, "py", "agent.py", slow_interval_s=10.0)
    handle = collector.handle(["node0"], "/r/tel", "abc")
    argv = collector.agent_argv(handle, "node0")
    assert argv[argv.index("--slow-interval") + 1] == "10"
    off = TelemetryCollector(cluster, "py", "agent.py", slow_interval_s=0.0)
    assert "--slow-interval 0" in " ".join(off.agent_argv(handle, "node0"))
    with pytest.raises(ValueError):
        TelemetryCollector(cluster, "py", "agent.py", slow_interval_s=0.5)


# ----------------------------------------------------------------- loader


def test_loader_builds_stress_samples_from_recorded_file() -> None:
    samples = load_samples(FIXTURES / "telemetry-stress-node1.jsonl")
    stress = samples.stress
    assert stress is not None
    assert (stress.node, stress.slow_interval_s, stress.gpux_source) == (
        "node1",
        5.0,
        "nvml",
    )
    assert stress.link_layers[("mlx5_4", 1)] == "Ethernet"
    assert stress.eth_hca == {"ens5f0np0": "mlx5_4", "ens6f0np0": "mlx5_5"}
    assert len(stress.ibx[("mlx5_0", 1)].t) == 4  # slow ticks 0, 5, 10, final
    assert len(stress.gpux[0].t) == 13  # the torn final line is dropped
    assert stress.gpux[0].reasons[3] == 0x4
    assert stress.notes == (
        "ethtool: exit 76 (ens6f0np0)",
        "2 malformed stress record(s) skipped",
    )
    # The bandwidth view of the same file is unaffected by the new kinds.
    assert samples.meta.hcas == (("mlx5_0", 1), ("mlx5_4", 1), ("mlx5_5", 1))
    assert len(samples.ib[("mlx5_0", 1)].t) == 13 and len(samples.gpu[0].t) == 13


def test_old_files_have_no_stress_and_meta_only_files_do() -> None:
    old = load_samples(FIXTURES / "telemetry-node0.jsonl")
    assert old.stress is None
    meta = {"k": "meta", "t": 1.0, "node": "n3", "slow_interval": 5.0, "notes": []}
    fresh = parse_samples(json.dumps(meta) + "\n")
    assert fresh.stress is not None and fresh.stress.node == "n3"
    assert fresh.stress.ibx == {} and fresh.stress.gpux == {}


def test_malformed_stress_records_never_fail_the_file() -> None:
    lines = [
        {"k": "meta", "t": 1.0, "node": "n"},
        {"k": "ibx", "t": 2.0, "dev": "mlx5_0"},
        {"k": "gpux", "t": "soon", "gpu": 0},
        {"k": "ethx", "t": 2.0, "iface": "eth2", "c": {"rx_prio3_pause": "x"}},
        {"k": "gpux", "t": 3.0, "gpu": 0, "power_w": float("inf"), "reasons": 4.5},
        {"k": "net", "t": 3.0, "iface": "wg0", "tx": 1, "rx": 2},
    ]
    text = "\n".join(json.dumps(r) for r in lines)
    samples = parse_samples(text)
    stress = samples.stress
    assert stress is not None
    assert stress.notes == ("2 malformed stress record(s) skipped",)
    assert stress.ethx["eth2"].values == {}  # non-integer counters dropped
    assert stress.gpux[0].power_w == (None,) and stress.gpux[0].reasons == (None,)
    assert samples.net["wg0"].tx == (1,)


def test_malformed_stress_meta_is_ignored() -> None:
    meta = {
        "k": "meta",
        "t": 1.0,
        "node": "n",
        "slow_interval": "five",
        "link_layers": 5,
        "ethx_ifaces": [1, {"iface": "ens5f0np0", "name": "mlx5_4"}],
        "notes": "ethtool missing",
    }
    stress = parse_samples(json.dumps(meta)).stress
    assert stress is not None and stress.slow_interval_s is None
    assert stress.link_layers == {} and stress.notes == ()
    assert stress.eth_hca == {"ens5f0np0": "mlx5_4"}
