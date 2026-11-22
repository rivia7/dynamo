# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for agents/node_facts.py (parsers, collectors, entry point).

``build_h100_node`` materialises a deliberately unhealthy 2-GPU H100 node
under ``tmp_path``: GPU1 trained at Gen4 x8, the GPU0 switch uplink at Gen4,
ACS redirect on a switch port, a 500 W power cap, one NVLink down, IB + RoCE
HCAs at different rates, a 64 MiB /dev/shm, cgroup v2 with pids.max=4096.
The host-mode document it yields is committed as
``fixtures/nodefacts/node_facts_h100.json`` (used by the domain and adapter
tests); ``test_fixture_document_matches_agent_output`` keeps the two in sync.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from benchmarks.cluster_bench.agents import node_facts as nf
from tests.benchmarks.cluster_bench.lium_fakes import fixture_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "nodefacts"
AGENT = Path(nf.__file__)
MIB = 1 << 20
GIB = 1 << 30
DMESG_ARGV = ("dmesg", "--level=emerg,alert,crit,err,warn")
DOCKER_ARGV = ("docker", "info", "--format", "{{json .}}")
FIXTURE_DOC = FIXTURES / "node_facts_h100.json"


def fx(name: str) -> str:
    return fixture_text(FIXTURES / name)


def ok(stdout: str, stderr: str = "") -> nf.ToolResult:
    return nf.ToolResult(0, stdout, stderr)


def fake_run(tools: dict[tuple[str, ...], nf.ToolResult]) -> Callable[..., Any]:
    """A ``Host.run`` answering from ``tools``; anything else is not installed."""

    def run(argv, timeout=None):
        return tools.get(tuple(argv))

    return run


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")


def h100_tools() -> dict[tuple[str, ...], nf.ToolResult]:
    p2p = fx("topo_p2p_r_h100.txt")
    return {
        ("nvidia-smi", "-q", "-x"): ok(fx("nvidia_smi_q_x_h100_r525.txt")),
        ("nvidia-smi", "nvlink", "-s"): ok(fx("nvlink_s_h100.txt")),
        ("nvidia-smi", "nvlink", "-e"): ok(fx("nvlink_e_h100.txt")),
        ("nvidia-smi", "topo", "-p2p", "r"): ok(p2p),
        ("nvidia-smi", "topo", "-p2p", "w"): ok(p2p),
        ("nvidia-smi", "topo", "-p2p", "n"): ok(p2p),
        ("nvidia-smi", "topo", "-p2p", "a"): ok(p2p.replace("OK", "NS")),
        ("ethtool", "-S", "enp59s0f0np0"): ok(fx("ethtool_S_cx7_roce.txt")),
        ("chronyc", "-c", "tracking"): ok(fx("chronyc_tracking.txt")),
        DOCKER_ARGV: ok(json.dumps(json.loads(fx("docker_info.json"))) + "\n"),
        DMESG_ARGV: ok(fx("dmesg_err_warn.txt")),
    }


# (bdf, kind, real path under /sys/devices, current GT/s, width, max GT/s, width)
PCI_LAYOUT = (
    ("0000:15:01.0", "bridge", "pci0000:15/0000:15:01.0", 16.0, 16, 32.0, 16),
    (
        "0000:16:00.0",
        "bridge",
        "pci0000:15/0000:15:01.0/0000:16:00.0",
        16.0,
        16,
        32.0,
        16,
    ),
    (
        "0000:17:00.0",
        "bridge",
        "pci0000:15/0000:15:01.0/0000:16:00.0/0000:17:00.0",
        32.0,
        16,
        32.0,
        16,
    ),
    (
        "0000:17:01.0",
        "bridge",
        "pci0000:15/0000:15:01.0/0000:16:00.0/0000:17:01.0",
        32.0,
        16,
        32.0,
        16,
    ),
    (
        "0000:18:00.0",
        "gpu",
        "pci0000:15/0000:15:01.0/0000:16:00.0/0000:17:00.0/0000:18:00.0",
        32.0,
        16,
        32.0,
        16,
    ),
    (
        "0000:19:00.0",
        "hca",
        "pci0000:15/0000:15:01.0/0000:16:00.0/0000:17:01.0/0000:19:00.0",
        32.0,
        16,
        32.0,
        16,
    ),
    ("0000:37:01.0", "bridge", "pci0000:37/0000:37:01.0", 32.0, 16, 32.0, 16),
    (
        "0000:38:00.0",
        "bridge",
        "pci0000:37/0000:37:01.0/0000:38:00.0",
        32.0,
        16,
        32.0,
        16,
    ),
    (
        "0000:39:00.0",
        "bridge",
        "pci0000:37/0000:37:01.0/0000:38:00.0/0000:39:00.0",
        16.0,
        8,
        32.0,
        16,
    ),
    (
        "0000:39:01.0",
        "bridge",
        "pci0000:37/0000:37:01.0/0000:38:00.0/0000:39:01.0",
        32.0,
        16,
        32.0,
        16,
    ),
    (
        "0000:3a:00.0",
        "gpu",
        "pci0000:37/0000:37:01.0/0000:38:00.0/0000:39:00.0/0000:3a:00.0",
        16.0,
        8,
        32.0,
        16,
    ),
    (
        "0000:3b:00.0",
        "hca",
        "pci0000:37/0000:37:01.0/0000:38:00.0/0000:39:01.0/0000:3b:00.0",
        32.0,
        16,
        32.0,
        16,
    ),
)
VENDOR_CLASS = {
    "gpu": ("0x10de", "0x2330", "0x030200", "nvidia"),
    "hca": ("0x15b3", "0x1021", "0x020700", "mlx5_core"),
    "bridge": ("0x1000", "0xc030", "0x060400", "pcieport"),
}
NUMA = {"0000:18:00.0": 0, "0000:19:00.0": 0, "0000:3a:00.0": 1, "0000:3b:00.0": 1}
AER_CORRECTABLE = {"0000:3a:00.0": 7}
ACS_CONFIG = {
    "0000:15:01.0": "pci_config_bridge_acs_redirect.txt",  # root port: ignored
    "0000:17:00.0": "pci_config_bridge_acs_redirect.txt",
    "0000:39:00.0": "pci_config_bridge_acs_off.txt",
}


def _aer(total: int, kind: str) -> str:
    names = {
        "correctable": ("RxErr", "BadTLP", "BadDLLP", "Rollover", "Timeout"),
        "nonfatal": ("Undefined", "DLP", "SDES", "TLP", "FCP"),
    }[kind]
    lines = [f"{names[0]} {total}"] + [f"{n} 0" for n in names[1:]]
    tag = "TOTAL_ERR_COR" if kind == "correctable" else "TOTAL_ERR_NONFATAL"
    return "\n".join(lines + [f"{tag} {total}"]) + "\n"


def make_pci(root: Path) -> None:
    devices = root / "sys/bus/pci/devices"
    devices.mkdir(parents=True, exist_ok=True)
    drivers = root / "sys/bus/pci/drivers"
    for bdf, kind, rel, cur_gts, cur_w, max_gts, max_w in PCI_LAYOUT:
        dev = root / "sys/devices" / rel
        dev.mkdir(parents=True, exist_ok=True)
        vendor, device, pci_class, driver = VENDOR_CLASS[kind]
        write(dev / "vendor", vendor)
        write(dev / "device", device)
        write(dev / "class", pci_class)
        write(dev / "current_link_speed", f"{cur_gts:.1f} GT/s PCIe")
        write(dev / "max_link_speed", f"{max_gts:.1f} GT/s PCIe")
        write(dev / "current_link_width", str(cur_w))
        write(dev / "max_link_width", str(max_w))
        write(dev / "numa_node", str(NUMA.get(bdf, -1)))
        write(
            dev / "aer_dev_correctable",
            _aer(AER_CORRECTABLE.get(bdf, 0), "correctable"),
        )
        write(dev / "aer_dev_nonfatal", _aer(0, "nonfatal"))
        (drivers / driver).mkdir(parents=True, exist_ok=True)
        (dev / "driver").symlink_to(drivers / driver)
        if bdf in ACS_CONFIG:
            (dev / "config").write_bytes(
                bytes.fromhex("".join(fx(ACS_CONFIG[bdf]).split()))
            )
        (devices / bdf).symlink_to(dev)
        if kind != "bridge":
            group = root / "sys/kernel/iommu_groups" / str(len(list(devices.iterdir())))
            write(group / "type", "DMA-FQ")
            (dev / "iommu_group").symlink_to(group)
    write(root / "sys/kernel/iommu_groups/90/type", "DMA")
    (root / "sys/class/iommu/dmar0").mkdir(parents=True)
    (root / "sys/class/iommu/dmar1").mkdir(parents=True)


def make_hca(
    root: Path, name: str, bdf: str, netdev: str, counters: str, rate: str
) -> None:
    rel = next(r for b, _, r, *_ in PCI_LAYOUT if b == bdf)
    dev = root / "sys/devices" / rel
    ib = root / "sys/class/infiniband" / name
    ib.mkdir(parents=True)
    (ib / "device").symlink_to(dev)
    write(ib / "fw_ver", "28.39.2048")
    write(ib / "board_id", "MT_0000000838" if name == "mlx5_0" else "MT_0000000844")
    data = json.loads((FIXTURES / counters).read_text())
    port = ib / "ports/1"
    write(port / "state", "4: ACTIVE")
    write(port / "phys_state", "5: LinkUp")
    write(port / "rate", rate)
    write(port / "link_layer", data["link_layer"])
    for group in ("counters", "hw_counters"):
        for key, value in data[group].items():
            write(port / group / key, str(value))
    (dev / "net" / netdev).mkdir(parents=True)
    if data["link_layer"] == "Ethernet":
        write(port / "gid_attrs/ndevs/0", netdev)
    write(
        root / "sys/class/net" / netdev / "mtu",
        "9000" if data["link_layer"] == "Ethernet" else "4092",
    )


def make_host_files(root: Path) -> None:
    proc = root / "proc"
    write(proc / "cmdline", fx("proc_cmdline.txt"))
    write(proc / "modules", fx("proc_modules.txt"))
    write(proc / "uptime", "12345.67 2701234.11")
    write(proc / "self/cgroup", "0::/")
    write(proc / "driver/nvidia/version", fx("nvidia_version_proprietary.txt"))
    write(proc / "sys/kernel/numa_balancing", "1")
    write(
        proc / "meminfo",
        "MemTotal:       2113348948 kB\nMemFree:        2001234567 kB\n"
        "MemAvailable:   2056789012 kB\nSwapTotal:              0 kB\n"
        "HugePages_Total:      16\nHugepagesize:       2048 kB\n",
    )
    node_dir = root / "sys/devices/system/node"
    write(node_dir / "online", "0-1")
    for node, cpus, mem, dist in (
        (0, "0-55,112-167", 1056687404, "10 21"),
        (1, "56-111,168-223", 1056661544, "21 10"),
    ):
        write(node_dir / f"node{node}/cpulist", cpus)
        write(
            node_dir / f"node{node}/meminfo",
            f"Node {node} MemTotal:       {mem} kB\n"
            f"Node {node} MemFree:        {mem - 9000} kB\n",
        )
        write(node_dir / f"node{node}/distance", dist)
    cpu_dir = root / "sys/devices/system/cpu"
    write(cpu_dir / "online", "0-223")
    write(cpu_dir / "smt/control", "on")
    write(cpu_dir / "smt/active", "1")
    write(cpu_dir / "intel_pstate/no_turbo", "0")
    for cpu, governor, cur in (
        (0, "performance", 3800000),
        (1, "performance", 3799000),
        (2, "powersave", 2000000),
        (3, "powersave", 2001000),
    ):
        freq = cpu_dir / f"cpu{cpu}/cpufreq"
        write(freq / "scaling_governor", governor)
        write(freq / "scaling_cur_freq", str(cur))
        write(freq / "cpuinfo_max_freq", "3800000")
        write(freq / "scaling_driver", "intel_pstate")
    mm = root / "sys/kernel/mm"
    for size, total in (("2048", 0), ("1048576", 16)):
        pool = mm / f"hugepages/hugepages-{size}kB"
        write(pool / "nr_hugepages", str(total))
        write(pool / "free_hugepages", str(total))
        write(pool / "resv_hugepages", "0")
        write(pool / "surplus_hugepages", "0")
    write(mm / "transparent_hugepage/enabled", "[always] madvise never")
    write(
        mm / "transparent_hugepage/defrag", "always defer defer+madvise [madvise] never"
    )
    cg = root / "sys/fs/cgroup"
    write(cg / "cgroup.controllers", "cpuset cpu io memory hugetlb pids rdma misc")
    write(cg / "cpu.max", "max 100000")
    write(cg / "cpuset.cpus.effective", "0-223")
    write(cg / "pids.max", "4096")
    write(cg / "pids.current", "312")
    write(
        cg / "cpu.stat",
        "usage_usec 81234567890\nuser_usec 61234567890\nsystem_usec 20000000000\n"
        "nr_periods 0\nnr_throttled 0\nthrottled_usec 0\n",
    )
    write(
        cg / "memory.events",
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
    )


def fake_statvfs(sizes: dict[str, tuple[int, int]]):
    def statvfs(path: str):
        if path not in sizes:
            raise FileNotFoundError(2, "No such file or directory", path)
        total, free = sizes[path]
        return SimpleNamespace(
            f_blocks=total // 4096, f_bavail=free // 4096, f_frsize=4096
        )

    return statvfs


def ticking_clock(start: float = 1793871100.0, step: float = 6.25):
    state = {"t": start - step}

    def clock() -> float:
        state["t"] += step
        return state["t"]

    return clock


def build_h100_node(root: Path, tools=None, **overrides: Any) -> nf.Host:
    """The unhealthy H100 node described in the module docstring."""
    make_pci(root)
    make_hca(
        root,
        "mlx5_0",
        "0000:19:00.0",
        "ibp25s0",
        "ib_counters_ndr.json",
        "400 Gb/sec (4X NDR)",
    )
    make_hca(
        root,
        "mlx5_1",
        "0000:3b:00.0",
        "enp59s0f0np0",
        "ib_counters_roce.json",
        "200 Gb/sec (4X HDR)",
    )
    make_host_files(root)
    which = {
        "dd": "/usr/bin/dd",
        "nvidia-container-runtime": "/usr/bin/nvidia-container-runtime",
    }
    options: dict[str, Any] = {
        "root": str(root),
        "run": fake_run(h100_tools() if tools is None else tools),
        "statvfs": fake_statvfs({"/dev/shm": (64 * MIB, 64 * MIB - 4096)}),
        "getrlimit": lambda which_limit: (64 * MIB, 64 * MIB)
        if which_limit == nf.resource.RLIMIT_MEMLOCK
        else (1048576, 1048576),
        "which": which.get,
        "adjtimex": lambda: None,
        "clock": ticking_clock(),
        "kernel_release": lambda: "5.15.0-122-generic",
        "hostname": lambda: "lium-h100-07",
    }
    options.update(overrides)
    return nf.Host(**options)


def normalized(doc: dict[str, Any]) -> dict[str, Any]:
    """A host document without run-dependent fields (for fixture comparison)."""
    out = dict(doc)
    for key in ("started_at", "collected_at", "duration_s"):
        out.pop(key, None)
    return json.loads(json.dumps(out, sort_keys=True))


# --------------------------------------------------------------------------
# Standalone script
# --------------------------------------------------------------------------


def test_agent_is_standalone_python38() -> None:
    tree = ast.parse(AGENT.read_text(), feature_version=(3, 8))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
    first = next(n for n in tree.body if isinstance(n, ast.ImportFrom))
    assert first.module == "__future__"


def test_script_runs_isolated_without_tools(tmp_path: Path) -> None:
    """No repo on sys.path and an empty PATH: still one JSON document, exit 0."""
    out = tmp_path / "facts.json"
    proc = subprocess.run(
        [sys.executable, "-I", str(AGENT), "--node", "node3", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": str(tmp_path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    doc = json.loads(out.read_text())
    assert doc["kind"] == "node_facts" and doc["node"] == "node3"
    assert any(n["source"] == "nvidia-smi -q -x" for n in doc["notes"])


# --------------------------------------------------------------------------
# Pure parsers: sysfs / procfs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "gts", "gen"),
    [
        ("32.0 GT/s PCIe", 32.0, 5),
        ("16.0 GT/s PCIe", 16.0, 4),
        ("8 GT/s", 8.0, 3),
        ("2.5 GT/s PCIe", 2.5, 1),
        ("64.0 GT/s PCIe", 64.0, 6),
        ("Unknown", None, None),
        (None, None, None),
    ],
)
def test_link_speed_and_generation(text, gts, gen) -> None:
    assert nf.parse_link_speed_gts(text) == gts
    assert nf.gen_of_speed(gts) == gen


def test_small_sysfs_parsers() -> None:
    assert nf.parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert nf.parse_cpulist("garbage") == []
    assert nf.parse_bracket_choice("always [madvise] never") == "madvise"
    assert nf.parse_bracket_choice("never\n") == "never"
    assert nf.parse_meminfo("Node 1 MemTotal:  10 kB\nHugePages_Total:  4\n") == {
        "MemTotal": 10240,
        "HugePages_Total": 4,
    }
    assert nf.parse_cgroup_cpu_max("max 100000") == (None, 100000)
    assert nf.parse_cgroup_cpu_max("800000 100000\n") == (800000, 100000)
    assert nf.parse_pids_max("max") is None and nf.parse_pids_max("4096\n") == 4096
    assert nf.parse_aer("RxErr 2\nBadTLP 1\nTOTAL_ERR_COR 3\n") == 3
    assert nf.parse_aer("RxErr 2\nBadTLP 1\n") == 3  # older kernels: no TOTAL line
    assert nf.parse_aer(None) is None
    assert nf.parse_int_map("nr_throttled 4\nthrottled_usec 12\nbad x\n") == {
        "nr_throttled": 4,
        "throttled_usec": 12,
    }


def test_parse_acs_ctrl_from_config_space() -> None:
    redirect = bytes.fromhex("".join(fx("pci_config_bridge_acs_redirect.txt").split()))
    off = bytes.fromhex("".join(fx("pci_config_bridge_acs_off.txt").split()))
    assert nf.parse_acs_ctrl(redirect) == 0x001D  # SV | RR | CR | UF
    assert nf.parse_acs_ctrl(off) == 0
    assert nf.parse_acs_ctrl(redirect[:64]) is None  # unprivileged read: header only
    no_acs = bytearray(redirect)
    no_acs[0x100:0x104] = (0x0001 | (2 << 16)).to_bytes(4, "little")  # AER, next=0
    assert nf.parse_acs_ctrl(bytes(no_acs)) is None
    assert nf.parse_acs_ctrl(None) is None


def test_parse_cmdline_keeps_only_perf_flags() -> None:
    flags = nf.parse_cmdline(fx("proc_cmdline.txt"))
    assert flags == {
        "intel_iommu": "on",
        "pci": "realloc=off",
        "transparent_hugepage": "always",
        "nvidia.NVreg_EnableGpuFirmware": "0",
        "mitigations": "off",
    }
    assert "ds" not in flags and "root" not in flags  # metadata URL, disk UUID


@pytest.mark.parametrize(
    ("name", "version", "flavor"),
    [
        ("nvidia_version_open.txt", "580.95.05", "open"),
        ("nvidia_version_proprietary.txt", "525.147.05", "proprietary"),
    ],
)
def test_parse_nvidia_version(name: str, version: str, flavor: str) -> None:
    parsed = nf.parse_nvidia_version(fx(name))
    assert parsed["version"] == version and parsed["flavor"] == flavor
    assert parsed["nvrm"].startswith("NVRM version:")
    assert nf.parse_nvidia_version(None) == {
        "version": None,
        "flavor": None,
        "nvrm": None,
    }


def test_parse_proc_modules() -> None:
    names = nf.parse_proc_modules(fx("proc_modules.txt"))
    assert {"nvidia", "nvidia_peermem", "mlx5_ib", "ib_umad"} <= set(names)
    assert "gdrdrv" not in names


# --------------------------------------------------------------------------
# Pure parsers: nvidia-smi
# --------------------------------------------------------------------------


def test_parse_smi_xml_h100_legacy_tags() -> None:
    doc = nf.parse_smi_xml(fx("nvidia_smi_q_x_h100_r525.txt"))
    assert doc["driver_version"] == "525.147.05" and doc["cuda_version"] == "12.0"
    gpu0, gpu1 = doc["gpus"]
    assert gpu0["bus_id"] == "0000:18:00.0" and gpu1["bus_id"] == "0000:3a:00.0"
    assert gpu1["index"] == 1 and gpu1["uuid"].startswith("GPU-c1e7")
    assert (
        gpu0["product"] == "NVIDIA H100 80GB HBM3" and gpu0["vbios"] == "96.00.74.00.0D"
    )
    assert gpu0["gsp_firmware"] is None  # "N/A" on a proprietary-module driver
    # Legacy <power_readings>: power_limit is requested, enforced_power_limit enforced.
    assert gpu1["power_limit_w"] == 500.0 and gpu1["power_requested_w"] == 500.0
    assert gpu1["power_default_w"] == 700.0 and gpu1["power_draw_w"] == 71.62
    assert gpu1["clocks_event_active"] == [
        "gpu_idle",
        "hw_slowdown",
        "hw_power_brake_slowdown",
    ]
    assert gpu0["clocks_event_active"] == ["gpu_idle"]
    assert gpu1["pcie_gen_current"] == 4 and gpu1["pcie_width_current"] == 8
    assert gpu1["pcie_width_max"] == 16 and gpu1["pcie_replay"] == 37
    assert (
        gpu1["ecc_aggregate_corrected"] == 12 and gpu1["ecc_aggregate_uncorrected"] == 0
    )
    assert gpu1["ecc_counters"]["aggregate.dram_correctable"] == 12
    assert gpu1["remapped_correctable"] == 1 and gpu1["remap_pending"] is False
    assert gpu1["retired_sbe"] is None and gpu1["retired_pending"] is None
    assert gpu0["fabric_state"] == "Completed" and gpu0["fabric_clique_id"] is None
    assert gpu0["temperature"]["gpu_temp_slow_threshold"] == 89.0
    assert gpu0["clocks_max_sm_mhz"] == 1980 and gpu0["clocks_app_graphics_mhz"] == 1755
    assert gpu0["xml_fields"]["power_readings/enforced_power_limit"] == "700.00 W"
    assert "fb_memory_usage/total" not in gpu0["xml_fields"]  # sections are curated


def test_parse_smi_xml_b200_current_tags() -> None:
    (gpu,) = nf.parse_smi_xml(fx("nvidia_smi_q_x_b200_r580.txt"))["gpus"]
    assert gpu["bus_id"] == "0000:1b:00.0" and gpu["architecture"] == "Blackwell"
    assert gpu["power_limit_w"] == 1000.0 and gpu["power_requested_w"] == 1000.0
    assert gpu["power_draw_w"] == 191.27
    assert gpu["clocks_event_us"] == {
        "sw_power_cap": 13377212,
        "sync_boost": 0,
        "sw_thermal_slowdown": 0,
        "hw_thermal_slowdown": 0,
        "hw_power_braking": 0,
    }
    # sram_uncorrectable_parity + _secded + dram_uncorrectable all count.
    assert gpu["ecc_volatile_corrected"] == 3 and gpu["ecc_volatile_uncorrected"] == 0
    assert (
        gpu["ecc_aggregate_corrected"] == 41 and gpu["ecc_aggregate_uncorrected"] == 1
    )
    assert "aggregate.sram_threshold_exceeded" not in gpu["ecc_counters"]
    assert gpu["remapped_uncorrectable"] == 1 and gpu["remap_pending"] is True
    assert gpu["remap_failure"] is False
    assert gpu["fabric_state"] == "Completed" and gpu["fabric_status"] == "Success"
    assert gpu["fabric_clique_id"] == "32766"
    assert gpu["fabric_cluster_uuid"] == "3f9c2b6e-8a41-4d7f-9e0c-5b1a2d3c4e5f"
    assert gpu["fabric_health"]["bandwidth"] == "Full"
    assert gpu["fabric_health"]["summary"] == "Healthy"
    # "None" (no recovery action pending) reads as N/A.
    assert gpu["recovery_action"] is None and gpu["reset_required"] == "No"
    assert gpu["pcie_gen_current"] == 1 and gpu["pcie_gen_max"] == 5  # idle downshift
    assert (
        gpu["temp_c"] == 34.0
        and gpu["temperature"]["gpu_temp_max_tlimit_threshold"] == -5.0
    )
    assert gpu["xml_fields"]["platformInfo/module_id"] == "1"
    assert gpu["xml_fields"]["addressing_mode"] == "HMM"


@pytest.mark.parametrize("text", ["", "not xml", "<other/>"])
def test_parse_smi_xml_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        nf.parse_smi_xml(text)


def test_parse_nvlink_status() -> None:
    gpus = nf.parse_nvlink_status(fx("nvlink_s_h100.txt"))
    assert [g["index"] for g in gpus] == [0, 1]
    assert gpus[0]["uuid"] == "GPU-5ad0a3c6-7e4c-11b2-2e2b-6cb1f1ad7b8e"
    assert len(gpus[0]["links"]) == 18 and gpus[0]["links"][0]["speed_gbs"] == 26.562
    assert gpus[1]["links"][7]["speed_gbs"] is None  # <inactive>
    rtx = "GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: GPU-0008)\n"
    assert nf.parse_nvlink_status(rtx) == [
        {"index": 0, "uuid": "GPU-0008", "bus_id": None, "links": []}
    ]
    assert nf.parse_nvlink_status(None) == []


def test_parse_nvlink_errors_legacy_and_nvlink5() -> None:
    legacy = nf.parse_nvlink_errors(fx("nvlink_e_h100.txt"))
    assert legacy[1][3] == {"replay_errors": 3, "recovery_errors": 0, "crc_errors": 12}
    assert len(legacy[0]) == 18
    current = nf.parse_nvlink_errors(fx("nvlink_e_b200.txt"))
    link5 = current[0][5]
    assert link5["symbol_errors"] == 2 and link5["fec_errors_0"] == 9312
    assert link5["link_recovery_successful_events"] == 1
    assert "effective_ber" not in link5 and "symbol_ber" not in link5  # floats skipped
    block = "GPU 0: NVIDIA B200 (UUID: GPU-1)\n\t Link 2:\n\t\t Rx Errors: 4\n"
    assert nf.parse_nvlink_errors(block) == {0: {2: {"rx_errors": 4}}}


def test_parse_p2p_matrix() -> None:
    h100 = nf.parse_p2p_matrix(fx("topo_p2p_r_h100.txt"))
    assert h100 == {
        "GPU0": {"GPU0": "X", "GPU1": "OK"},
        "GPU1": {"GPU0": "OK", "GPU1": "X"},
    }
    rtx = nf.parse_p2p_matrix(fx("topo_p2p_r_rtx6000.txt"))
    assert (
        len(rtx) == 8 and rtx["GPU0"]["GPU4"] == "CNS" and rtx["GPU5"]["GPU6"] == "OK"
    )
    assert nf.parse_p2p_matrix("Legend:\n X = Self\n") == {}


# --------------------------------------------------------------------------
# Pure parsers: network, time, docker, kernel log
# --------------------------------------------------------------------------


def test_parse_and_filter_ethtool() -> None:
    stats = nf.parse_ethtool_stats(fx("ethtool_S_cx7_roce.txt"))
    assert stats["rx_prio3_pause"] == 77105 and stats["rx0_packets"] == 1032
    kept = nf.filter_ethtool(stats)
    assert kept["rx_prio3_pause"] == 77105 and kept["tx_prio3_pause_duration"] == 3020
    assert kept["rx_prio3_marked"] == 1290 and kept["rx_discards_phy"] == 0
    for name in (
        "rx_out_of_buffer",
        "tx_pause_storm_error_events",
        "rx_crc_errors_phy",
        "rx_corrected_bits_phy",
    ):
        assert name in kept
    for name in (
        "rx_bytes",
        "tx_packets_phy",
        "rx0_buff_alloc_err",
        "tx0_dropped",
        "ch_events",
    ):
        assert name not in kept


def test_time_sync_parsers() -> None:
    chrony = nf.parse_chronyc_tracking(fx("chronyc_tracking.txt"))
    assert chrony["synchronized"] is True and chrony["stratum"] == 4
    assert chrony["offset_s"] == pytest.approx(-0.000012845)
    assert (
        chrony["reference"] == "169.254.169.123" and chrony["leap_status"] == "Normal"
    )
    unsynced = (
        "00000000,,0,0.000000000,0.000000000,0.0,0.0,0.000,0.000,0.000,"
        "1.0,1.0,0.0,Not synchronised\n"
    )
    assert nf.parse_chronyc_tracking(unsynced)["synchronized"] is False
    assert nf.parse_chronyc_tracking("506 Cannot talk to daemon") is None
    timedatectl = nf.parse_timedatectl(fx("timedatectl_show.txt"))
    assert (
        timedatectl["source"] == "timedatectl" and timedatectl["synchronized"] is False
    )
    assert nf.parse_timedatectl("Timezone=UTC\n") is None
    synced = nf.interpret_adjtimex(0, 0x2001, 1234, 56)
    assert synced["synchronized"] is True and synced["max_error_s"] == pytest.approx(
        0.001234
    )
    assert (
        nf.interpret_adjtimex(5, nf.STA_UNSYNC, 16_000_000, 0)["synchronized"] is False
    )
    assert nf.interpret_adjtimex(0, 0, 16_000_000, 0)["synchronized"] is False


def test_parse_docker_info() -> None:
    line = json.dumps(json.loads(fx("docker_info.json")))
    info = nf.parse_docker_info("WARNING: something\n" + line + "\n")
    assert info["server_version"] == "27.5.1" and info["storage_driver"] == "overlay2"
    assert info["cgroup_driver"] == "cgroupfs" and info["cgroup_version"] == "2"
    assert info["runtimes"] == ["io.containerd.runc.v2", "nvidia", "runc"]
    assert info["nvidia_runtime"] is True and info["default_runtime"] == "runc"
    assert info["cdi_spec_dirs"] == ["/etc/cdi", "/var/run/cdi"]
    assert info["server_errors"] == []
    down = nf.parse_docker_info(
        json.dumps({"ServerErrors": ["Cannot connect to the Docker daemon"]})
    )
    assert down["server_errors"] == ["Cannot connect to the Docker daemon"]
    assert down["nvidia_runtime"] is False
    assert nf.parse_docker_info("garbage") is None


def test_scan_dmesg_classifies_xid_mlx5_aer() -> None:
    entries = [
        nf.parse_dmesg_line(line) for line in fx("dmesg_err_warn.txt").splitlines()
    ]
    scanned = nf.scan_kernel_log(entries, {"0000:3a:00": "0000:3a:00.0"})
    xids = [
        (e["code"], e["pci"], e["ours"])
        for e in scanned["events"]
        if e["kind"] == "xid"
    ]
    assert xids == [
        (94, "0000:3a:00.0", True),
        (63, "0000:3a:00.0", True),
        (13, "0000:c1:00", False),
    ]
    kinds = [e["kind"] for e in scanned["events"]]
    assert kinds.count("mlx5") == 2 and kinds.count("aer") == 2
    assert scanned["events"][0]["t_s"] == pytest.approx(19.873305)
    assert scanned["lines"] == 10
    assert {s["kind"] for s in scanned["samples"]} == {"xid", "mlx5", "aer"}
    assert len([s for s in scanned["samples"] if s["kind"] == "xid"]) == 3


def test_foreign_mlx5_and_aer_lines_are_not_ours() -> None:
    entries = [
        (1.0, "mlx5_core 0000:d1:00.0: poll_health:801: Fatal error 1 detected"),
        (2.0, "mlx5_core 0000:19:00.0 ibp25s0: Link down"),
        (3.0, "pcieport 0000:d0:01.0: AER: Corrected error message received"),
        (4.0, "mlx5_core: module loaded"),
    ]
    events = nf.scan_kernel_log(entries, {}, ["0000:19:00.0", "0000:39:00.0"])["events"]
    assert [(e["kind"], e["pci"], e["ours"]) for e in events] == [
        ("mlx5", "0000:d1:00.0", False),
        ("mlx5", "0000:19:00.0", True),
        ("aer", "0000:d0:01.0", False),
        ("mlx5", None, True),
    ]
    assert all(e["ours"] for e in nf.scan_kernel_log(entries, {})["events"])


def test_parse_kmsg_records() -> None:
    records = [nf.parse_kmsg_line(line) for line in fx("kmsg_records.txt").splitlines()]
    assert records[0] == (4, pytest.approx(0.451802), records[0][2])
    assert records[3] is None  # " SUBSYSTEM=net" continuation line
    xid = next(r for r in records if r and "Xid" in r[2])
    assert xid[0] == 6 and xid[1] == pytest.approx(211.004512)


# --------------------------------------------------------------------------
# Pure parsers: container mode, dd
# --------------------------------------------------------------------------


def test_container_tool_parsers() -> None:
    ucx = nf.parse_ucx_info(fx("ucx_info_v.txt"))
    assert ucx["version"] == "1.19.0" and ucx["revision"] == "2a0a4e4"
    assert "--with-cuda=/usr/local/cuda" in ucx["configured_with"]
    assert "--prefix=/usr/local/ucx" not in ucx["configured_with"]
    legacy = nf.parse_ucx_info(fx("ucx_info_v_legacy.txt"))
    assert legacy["version"] == "1.12.1" and legacy["revision"] == "dc92435"
    assert nf.parse_ucx_info("") is None
    assert nf.parse_nvcc_version(fx("nvcc_version.txt")) == "13.0.88"
    assert nf.parse_ibv_devinfo_list(fx("ibv_devinfo_l.txt")) == [
        f"mlx5_{i}" for i in range(8)
    ]
    assert nf.parse_ibv_devinfo_list("No IB devices found\n") == []
    assert nf.parse_os_release(fx("os_release.txt")) == {
        "PRETTY_NAME": "Ubuntu 24.04.3 LTS",
        "VERSION_ID": "24.04",
        "VERSION_CODENAME": "noble",
        "ID": "ubuntu",
        "ID_LIKE": "debian",
    }
    marker = 'noise\nNODE_FACTS_TORCH={"version": "2.9.1", "nccl": "2.28.3"}\n'
    assert nf.parse_torch_probe(marker) == {"version": "2.9.1", "nccl": "2.28.3"}
    assert nf.parse_torch_probe("no marker") is None


def test_select_packages_and_filter_env() -> None:
    packages, missing = nf.select_packages(
        [
            ("sglang", "0.5.22"),
            ("sgl_kernel", "0.3.21"),
            ("torch", "2.9.1+cu130"),
            ("nvidia-nccl-cu13", "2.28.3"),
            ("Deep_EP", "1.2.1"),
            ("requests", "2.32.3"),
        ]
    )
    assert packages == {
        "deep-ep": "1.2.1",
        "nvidia-nccl-cu13": "2.28.3",
        "sgl-kernel": "0.3.21",
        "sglang": "0.5.22",
        "torch": "2.9.1+cu130",
    }
    assert "flashinfer-python" in missing and "sglang" not in missing
    # sgl-kernel is the former name of sglang-kernel (SGLang 0.5.18+).
    assert "sglang-kernel" not in missing
    packages, missing = nf.select_packages([("sglang_kernel", "0.4.7")])
    assert packages == {"sglang-kernel": "0.4.7"} and "sglang-kernel" not in missing
    assert "sglang-kernel" in nf.select_packages([])[1]
    env = nf.filter_env(
        {
            "NCCL_IB_DISABLE": "0",
            "UCX_TLS": "rc,cuda_copy",
            "HF_TOKEN": "hf_secret",
            "NCCL_API_KEY": "nope",
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/usr/local/ucx/lib",
        }
    )
    assert env == {
        "LD_LIBRARY_PATH": "/usr/local/ucx/lib",
        "NCCL_IB_DISABLE": "0",
        "UCX_TLS": "rc,cuda_copy",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, None),
        ("1024+0 records in\n1024+0 records out\n", None),
        (
            "1024+0 records in\n1024+0 records out\n"
            "17179869184 bytes (16.0GB) copied, 3.209823 seconds, 5.0GB/s\n",
            (17179869184, 3.209823),
        ),
        (
            "17179869184 bytes (17 GB) copied, 3,21 s, 5,3 GB/s\n",
            (17179869184, 3.21),
        ),
        (
            "17179869184 bytes transferred in 3.214001 secs (5345259163 bytes/sec)\n",
            (17179869184, 3.214001),
        ),
    ],
)
def test_parse_dd_output(text, expected) -> None:
    assert nf.parse_dd_output(text) == expected
    assert nf.parse_dd_output(fx("dd_gnu.txt")) == (17179869184, 3.21357)


# --------------------------------------------------------------------------
# Collectors
# --------------------------------------------------------------------------


def test_collect_host_from_fake_tree(tmp_path: Path) -> None:
    doc = nf.collect_host(build_h100_node(tmp_path / "root"), node="node0")
    assert (
        doc["kind"] == "node_facts" and doc["version"] == 1 and doc["node"] == "node0"
    )
    assert doc["hostname"] == "lium-h100-07"
    assert doc["started_at"] == 1793871100.0 and doc["duration_s"] == 6.25
    pcie = {r["bdf"]: r for r in doc["pcie"]}
    gpu1 = pcie["0000:3a:00.0"]
    assert (gpu1["kind"], gpu1["current_speed_gts"], gpu1["current_width"]) == (
        "gpu",
        16.0,
        8,
    )
    assert (gpu1["max_speed_gts"], gpu1["max_width"], gpu1["depth"]) == (32.0, 16, 0)
    assert gpu1["parent"] == "0000:39:00.0" and gpu1["driver"] == "nvidia"
    assert gpu1["aer_correctable"] == 7 and gpu1["numa_node"] == 1
    assert [r["bdf"] for r in doc["pcie"] if r["depth"] == 0] == [
        "0000:18:00.0",
        "0000:19:00.0",
        "0000:3a:00.0",
        "0000:3b:00.0",
    ]
    usp = pcie["0000:16:00.0"]
    assert (
        usp["kind"] == "bridge"
        and usp["depth"] == 2
        and usp["parent"] == "0000:15:01.0"
    )
    assert usp["endpoints"] == ["0000:18:00.0", "0000:19:00.0"]
    assert pcie["0000:15:01.0"]["root_port"] is True and usp["root_port"] is False
    assert (
        pcie["0000:17:00.0"]["acs_ctrl"] == 0x1D
        and pcie["0000:39:00.0"]["acs_ctrl"] == 0
    )
    assert pcie["0000:38:00.0"]["acs_ctrl"] is None
    assert doc["kernel"]["release"] == "5.15.0-122-generic"
    assert doc["kernel"]["uptime_s"] == 12345.67
    assert doc["kernel"]["iommu_devices"] == ["dmar0", "dmar1"]
    assert set(doc["kernel"]["iommu_group_types"].values()) == {"DMA-FQ"}
    assert doc["kernel"]["iommu_groups"] == 5
    assert doc["driver"] == {
        "version": "525.147.05",
        "flavor": "proprietary",
        "nvrm": "NVRM version: NVIDIA UNIX x86_64 Kernel Module 525.147.05 "
        "Wed Oct 25 20:27:35 UTC 2023",
        "cuda_version": "12.0",
    }
    assert (
        doc["modules"]["nvidia_peermem"] is True and doc["modules"]["gdrdrv"] is False
    )
    assert doc["cpu"]["governors"] == {"performance": 2, "powersave": 2}
    assert doc["cpu"]["online_count"] == 224 and doc["cpu"]["boost"] == "1"
    assert doc["cpu"]["smt_active"] is True and doc["cpu"]["cur_khz_max"] == 3800000
    assert [n["cpus"] for n in doc["numa"]["nodes"]] == [112, 112]
    assert doc["numa"]["nodes"][1]["mem_total_bytes"] == 1056661544 * 1024
    assert (
        doc["numa"]["nodes"][0]["distances"] == [10, 21]
        and doc["numa"]["balancing"] == 1
    )
    memory = doc["memory"]
    assert memory["thp_enabled"] == "always" and memory["thp_defrag"] == "madvise"
    assert memory["shm_total_bytes"] == 64 * MIB
    assert [p["size_kb"] for p in memory["hugepages"]] == [2048, 1048576]
    assert memory["hugepages"][1]["total"] == 16
    assert (
        doc["limits"]["memlock_hard"] == 64 * MIB
        and doc["limits"]["nofile_soft"] == 1048576
    )
    cgroup = doc["cgroup"]
    assert cgroup["version"] == 2 and cgroup["cpu_quota_us"] is None
    assert cgroup["cpuset_count"] == 224 and cgroup["pids_max"] == 4096
    assert cgroup["memory_events"]["oom_kill"] == 0 and cgroup["cpu_throttled_us"] == 0
    assert [g["bus_id"] for g in doc["gpus"]] == ["0000:18:00.0", "0000:3a:00.0"]
    assert [e["bus_id"] for e in doc["nvlink"]] == ["0000:18:00.0", "0000:3a:00.0"]
    assert doc["nvlink"][1]["links"][3]["errors"]["crc_errors"] == 12
    assert set(doc["p2p"]) == {"read", "write", "nvlink", "atomics"}
    assert doc["p2p"]["atomics"]["GPU0"]["GPU1"] == "NS"
    hcas = {h["name"]: h for h in doc["hcas"]}
    ib_port = hcas["mlx5_0"]["ports"][0]
    assert ib_port["link_layer"] == "InfiniBand" and ib_port["netdev"] == "ibp25s0"
    assert ib_port["counters"]["link_downed"] == 1 and ib_port["ethtool"] == {}
    assert "lifespan" not in ib_port["hw_counters"] and ib_port["mtu"] == 4092
    roce = hcas["mlx5_1"]["ports"][0]
    assert roce["netdev"] == "enp59s0f0np0" and roce["mtu"] == 9000
    assert roce["ethtool"]["rx_prio3_pause"] == 77105
    assert roce["hw_counters"]["np_cnp_sent"] == 18273
    assert doc["time_sync"]["source"] == "chronyc" and doc["time_sync"]["synchronized"]
    assert (
        doc["docker"]["nvidia_container_runtime"] == "/usr/bin/nvidia-container-runtime"
    )
    assert doc["docker"]["nvidia_ctk"] is None
    log = doc["kernel_log"]
    assert log["available"] and log["source"] == "dmesg" and log["error"] is None
    assert len(log["events"]) == 7
    json.dumps(doc)  # serializable as is


def test_fixture_document_matches_agent_output(tmp_path: Path) -> None:
    """``node_facts_h100.json`` is this agent's output for the fake node."""
    doc = nf.collect_host(build_h100_node(tmp_path / "root"), node="node0")
    fixture = json.loads(FIXTURE_DOC.read_text())
    assert normalized(fixture) == normalized(doc)


def test_collect_host_without_any_source(tmp_path: Path) -> None:
    host = nf.Host(
        root=str(tmp_path),
        run=fake_run({}),
        statvfs=fake_statvfs({}),
        which=lambda name: None,
        adjtimex=lambda: None,
        kernel_release=lambda: "6.8.0-60-generic",
        hostname=lambda: "bare",
    )
    doc = nf.collect_host(host, node="node1")
    assert doc["gpus"] == [] and doc["pcie"] == [] and doc["hcas"] == []
    assert doc["nvlink"] == [] and doc["p2p"] == {} and doc["docker"] is None
    assert doc["kernel_log"]["available"] is False
    assert doc["time_sync"]["source"] is None
    sources = {n["source"] for n in doc["notes"]}
    for expected in (
        "nvidia-smi -q -x",
        "sysfs:pci",
        "sysfs:infiniband",
        "cgroup",
        "statvfs:/dev/shm",
        "adjtimex",
        "docker info",
        "dmesg",
    ):
        assert expected in sources, expected


def test_collect_host_respects_deadline(tmp_path: Path) -> None:
    host = build_h100_node(tmp_path / "root")
    doc = nf.collect_host(host, node="node0", deadline_s=0.0)
    assert doc["gpus"] == []
    skipped = [n for n in doc["notes"] if "deadline" in n["error"]]
    assert {n["source"] for n in skipped} >= {"nvidia-smi -q -x", "chronyc -c tracking"}
    assert doc["pcie"]  # sysfs reads need no tool and still happen


def test_one_broken_section_does_not_lose_the_document(tmp_path: Path) -> None:
    def broken_getrlimit(which: int):
        raise TypeError("unexpected rlimit tuple")

    host = build_h100_node(tmp_path / "root", getrlimit=broken_getrlimit)
    doc = nf.collect_host(host, node="node0")
    assert doc["limits"] == {} and len(doc["gpus"]) == 2
    assert {
        "source": "collector:limits",
        "error": "TypeError: unexpected rlimit tuple",
    } in doc["notes"]


def test_kernel_log_falls_back_to_kmsg(tmp_path: Path) -> None:
    tools = h100_tools()
    tools[DMESG_ARGV] = nf.ToolResult(
        1, "", "dmesg: read kernel buffer failed: Operation not permitted"
    )
    host = build_h100_node(tmp_path / "root", tools=tools)
    write(tmp_path / "root/dev/kmsg", fx("kmsg_records.txt"))
    log = nf.collect_kernel_log(nf.Session(host), ["0000:3a:00.0"])
    assert log["available"] and log["source"] == "kmsg"
    kinds = [(e["kind"], e["code"]) for e in log["events"]]
    # The level-6 Xid records are kept; the level-6 EXT4 line is not an event.
    assert kinds == [("mlx5", None), ("xid", 94), ("xid", 63), ("mlx5", None)]


def test_kernel_log_unavailable_when_denied(tmp_path: Path) -> None:
    host = build_h100_node(
        tmp_path / "root",
        tools={DMESG_ARGV: nf.ToolResult(1, "", "Operation not permitted")},
    )
    session = nf.Session(host)
    log = nf.collect_kernel_log(session, [])
    assert log["available"] is False and log["events"] == []
    assert "Operation not permitted" in log["error"] and "/dev/kmsg" in log["error"]


def test_time_sync_fallbacks(tmp_path: Path) -> None:
    tools = {("timedatectl", "show"): ok(fx("timedatectl_show.txt"))}
    host = nf.Host(
        root=str(tmp_path),
        run=fake_run(tools),
        adjtimex=lambda: (5, 0x41, 16_000_000, 0),
    )
    sync = nf.collect_time_sync(nf.Session(host))
    assert sync["source"] == "timedatectl" and sync["synchronized"] is False
    host = nf.Host(
        root=str(tmp_path), run=fake_run({}), adjtimex=lambda: (0, 0x2001, 512, 16)
    )
    sync = nf.collect_time_sync(nf.Session(host))
    assert sync["source"] == "adjtimex" and sync["synchronized"] is True


def test_cgroup_v1(tmp_path: Path) -> None:
    cg = tmp_path / "sys/fs/cgroup"
    write(cg / "cpu,cpuacct/cpu.cfs_quota_us", "1600000")
    write(cg / "cpu,cpuacct/cpu.cfs_period_us", "100000")
    write(
        cg / "cpu,cpuacct/cpu.stat",
        "nr_periods 10\nnr_throttled 3\nthrottled_time 5000000\n",
    )
    write(cg / "cpuset/cpuset.cpus", "0-31")
    write(cg / "pids/pids.max", "max")
    write(
        cg / "memory/memory.oom_control",
        "oom_kill_disable 0\nunder_oom 0\noom_kill 2\n",
    )
    out = nf.collect_cgroup(nf.Session(nf.Host(root=str(tmp_path))))
    assert out["version"] == 1 and out["cpu_quota_us"] == 1600000
    assert out["cpu_period_us"] == 100000 and out["cpuset_count"] == 32
    assert out["pids_max"] is None and out["cpu_nr_throttled"] == 3
    assert out["cpu_throttled_us"] == 5000 and out["memory_events"] == {"oom_kill": 2}


def test_collect_software_with_fakes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write(root / "etc/os-release", fx("os_release.txt"))
    write(
        root / "usr/local/cuda/version.json",
        json.dumps({"cuda": {"version": "13.0.1"}}),
    )
    for name in (
        "libnccl.so.2",
        "libnccl.so.2.28.3",
        "libnccl-net.so",
        "libnccl-tuner-x.so",
    ):
        write(root / "usr/lib/x86_64-linux-gnu" / name, "")
    ucx_bin = root / "usr/local/ucx/bin/ucx_info"
    write(ucx_bin, "#!/bin/sh\n")
    ucx_bin.chmod(0o755)
    site = tmp_path / "site-packages"
    write(site / "nvidia/nccl/lib/libnccl.so.2", "")
    torch_out = nf._TORCH_MARK + json.dumps(
        {
            "version": "2.9.1+cu130",
            "cuda": "13.0",
            "nccl": "2.28.3",
            "cuda_available": True,
            "device_count": 1,
            "errors": [],
        }
    )
    python = "/usr/bin/python3"
    tools = {
        (python, "-c", nf._TORCH_SNIPPET): ok("warning: noise\n" + torch_out + "\n"),
        ("/usr/local/ucx/bin/ucx_info", "-v"): ok(fx("ucx_info_v.txt")),
        ("/usr/local/cuda/bin/nvcc", "--version"): ok(fx("nvcc_version.txt")),
        ("ibv_devinfo", "-l"): ok(fx("ibv_devinfo_l.txt")),
    }
    which = {"/usr/local/cuda/bin/nvcc": None}
    nvcc = root / "usr/local/cuda/bin/nvcc"
    write(nvcc, "#!/bin/sh\n")
    nvcc.chmod(0o755)
    host = nf.Host(
        root=str(root),
        run=fake_run(tools),
        which=which.get,
        clock=lambda: 1793871200.0,
        hostname=lambda: "container-1",
        distributions=lambda: [("sglang", "0.5.22"), ("torch", "2.9.1+cu130")],
        libc_version=lambda: "2.39",
        python=python,
        environ={"NCCL_DEBUG": "WARN", "HF_TOKEN": "hf_x", "LD_LIBRARY_PATH": ""},
        search_path=[str(site)],
    )
    doc = nf.collect_software(host, node="node0")
    assert doc["kind"] == "software_stack" and doc["node"] == "node0"
    assert doc["packages"] == {"sglang": "0.5.22", "torch": "2.9.1+cu130"}
    assert doc["torch"]["nccl"] == "2.28.3" and doc["torch"]["cuda_available"] is True
    assert doc["ucx"]["version"] == "1.19.0" and doc["nvcc"] == "13.0.88"
    assert doc["cuda_toolkit"] == "13.0.1" and doc["glibc"] == "2.39"
    assert doc["ibv_devices"] == [f"mlx5_{i}" for i in range(8)]
    assert doc["os_release"]["VERSION_ID"] == "24.04"
    assert doc["nccl_plugins"] == [
        "/usr/lib/x86_64-linux-gnu/libnccl-net.so",
        "/usr/lib/x86_64-linux-gnu/libnccl-tuner-x.so",
    ]
    assert "/usr/lib/x86_64-linux-gnu/libnccl.so.2" in doc["nccl_libraries"]
    assert any(
        p.endswith("nvidia/nccl/lib/libnccl.so.2") for p in doc["nccl_libraries"]
    )
    assert doc["env"] == {"LD_LIBRARY_PATH": "", "NCCL_DEBUG": "WARN"}


def test_collect_software_without_torch_or_tools(tmp_path: Path) -> None:
    tools = {
        ("python3", "-c", nf._TORCH_SNIPPET): ok(
            nf._TORCH_MARK
            + json.dumps({"errors": ["import torch: No module named 'torch'"]})
        )
    }
    host = nf.Host(
        root=str(tmp_path),
        run=fake_run(tools),
        which=lambda name: None,
        distributions=lambda: [],
        python="python3",
        environ={},
        search_path=[],
    )
    doc = nf.collect_software(host)
    assert doc["torch"] == {"errors": ["import torch: No module named 'torch'"]}
    assert doc["ucx"] is None and doc["nvcc"] is None and doc["ibv_devices"] == []
    assert doc["missing_packages"] == list(nf.EXPECTED_PACKAGES)
    assert {n["source"] for n in doc["notes"]} >= {"ucx_info", "ibv_devinfo -l"}


# --------------------------------------------------------------------------
# Disk reads
# --------------------------------------------------------------------------


def make_weights(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model-00001-of-00003.safetensors").write_bytes(b"\0" * 3 * MIB)
    big = directory / "model-00002-of-00003.safetensors"
    big.write_bytes(os.urandom(5 * MIB))
    (directory / "model-00003-of-00003.safetensors").write_bytes(b"\0" * MIB)
    return big


def test_disk_read_prefers_dd_direct(tmp_path: Path) -> None:
    big = make_weights(tmp_path / "weights")
    argv = (
        "env",
        "LC_ALL=C",
        "dd",
        f"if={big}",
        "of=/dev/null",
        f"bs={16 * MIB}",
        "count=1",
        "iflag=direct",
    )
    host = nf.Host(
        run=fake_run({argv: ok("", fx("dd_gnu.txt"))}),
        which={"dd": "/usr/bin/dd"}.get,
        hostname=lambda: "h",
    )
    doc = nf.disk_read(host, str(tmp_path / "weights"), max_bytes=8 * GIB, node="node2")
    assert doc["path"] == str(big) and doc["file_bytes"] == 5 * MIB
    assert (doc["method"], doc["direct"], doc["cache"]) == (
        "dd_direct",
        True,
        "bypassed",
    )
    assert doc["bytes"] == 17179869184 and doc["seconds"] == 3.21357
    assert doc["gbs"] == pytest.approx(17179869184 / 3.21357 / 1e9, abs=1e-4)
    assert doc["error"] is None and doc["node"] == "node2"
    assert doc["flushed"] is True  # fsync'ed before the timed read


def test_disk_read_falls_back_to_python_reads(tmp_path: Path) -> None:
    big = make_weights(tmp_path / "weights")
    host = nf.Host(
        run=fake_run({}),
        which={"dd": "/usr/bin/dd"}.get,  # listed on PATH but "not installed" when run
        hostname=lambda: "h",
    )
    doc = nf.disk_read(host, str(big), max_bytes=2 * MIB, block_mib=1)
    assert doc["method"] in ("o_direct", "buffered")
    assert doc["bytes"] >= 2 * MIB and doc["gbs"] > 0 and doc["seconds"] > 0
    if doc["method"] == "buffered":  # e.g. macOS or tmpfs without O_DIRECT
        assert doc["direct"] is False and doc["cache"] in ("dropped", "unknown")
    assert any(n["source"] == "dd iflag=direct" for n in doc["notes"])


@pytest.mark.parametrize("make", ["missing", "empty_dir"])
def test_disk_read_reports_unusable_paths(tmp_path: Path, make: str) -> None:
    target = tmp_path / "nothing"
    if make == "empty_dir":
        target.mkdir()
    host = nf.Host(run=fake_run({}), which=lambda name: None, hostname=lambda: "h")
    doc = nf.disk_read(host, str(target))
    assert doc["bytes"] == 0 and doc["gbs"] is None and doc["method"] is None
    assert doc["error"] and str(target) in doc["error"]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def test_main_defaults_to_host_and_prints_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert nf.main(["--node", "node0"], host=build_h100_node(tmp_path / "root")) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert json.loads(out)["kind"] == "node_facts"


def test_main_modes_write_out_files(tmp_path: Path) -> None:
    out = tmp_path / "out/disk.json"
    big = make_weights(tmp_path / "weights")
    host = nf.Host(run=fake_run({}), which=lambda name: None, hostname=lambda: "h")
    args = ["disk-read", "--path", str(big), "--max-bytes", str(MIB), "--out", str(out)]
    assert nf.main(args, host=host) == 0
    assert json.loads(out.read_text())["kind"] == "disk_read"
    with pytest.raises(SystemExit) as exc:
        nf.main(["disk-read"], host=host)  # --path is required
    assert exc.value.code == 2
