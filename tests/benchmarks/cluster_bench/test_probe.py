# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for agents/probe.py and the shared inventory fixtures."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.cluster_bench.agents import probe
from benchmarks.cluster_bench.domain.schema import Inventory, NodeInventory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "probe"
AGENTS = Path(probe.__file__).parent
GIB = 1 << 30
TIB = 1 << 40


def fixture(name: str) -> str:
    """Recorded command output, minus the SPDX header the repo requires."""
    text = (FIXTURES / name).read_text()
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not line.startswith("# SPDX-")
    )


def inventory_payload(name: str) -> dict:
    return json.loads(fixture(name))


def ok(stdout: str) -> probe.ToolResult:
    return probe.ToolResult(0, stdout, "")


GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=" + ",".join(probe.GPU_QUERY_FIELDS),
    "--format=csv,noheader,nounits",
)
GPU_CORE_QUERY = (
    "nvidia-smi",
    "--query-gpu=" + ",".join(probe.GPU_QUERY_CORE_FIELDS),
    "--format=csv,noheader,nounits",
)


def h200_tools() -> dict[tuple[str, ...], probe.ToolResult]:
    return {
        GPU_QUERY: ok(fixture("gpu_query_h200.csv")),
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"): ok(
            "580.95.05\n" * 8
        ),
        ("nvidia-smi",): ok(fixture("nvidia_smi_banner_h200.txt")),
        ("nvidia-smi", "topo", "-m"): ok(fixture("topo_h200_8nic.txt")),
        ("nvidia-smi", "nvlink", "-s"): ok(fixture("nvlink_status_h200.txt")),
        ("docker", "version", "--format", "{{.Server.Version}}"): ok("27.5.1\n"),
        ("ofed_info", "-s"): ok("MLNX_OFED_LINUX-24.10-1.1.4.0:\n"),
    }


def fake_run(tools: dict[tuple[str, ...], probe.ToolResult]):
    """A ``Host.run`` that answers from ``tools``; unknown tools are missing."""

    def run(argv, timeout=None):
        return tools.get(tuple(argv))

    return run


def fake_statvfs(sizes: dict[str, tuple[int, int]]):
    def statvfs(path: str):
        if path not in sizes:
            raise FileNotFoundError(path)
        total, free = sizes[path]
        return SimpleNamespace(
            f_blocks=total // 4096, f_bavail=free // 4096, f_frsize=4096
        )

    return statvfs


DISK_SIZES = {
    "/": (200 * GIB, 151 * GIB),
    "/root": (7 * TIB, 6 * TIB),
    "/workspace": (3 * TIB, 3 * TIB - 50 * GIB),
    "/scratch disk": (1 * TIB, 1 * TIB),
    "/small": (20 * GIB, 19 * GIB),
    "/tank": (10 * TIB, 9 * TIB),
    "/boot/efi": (512 << 20, 500 << 20),
    "/mnt/volume": (100 * TIB, 100 * TIB),
    "/mnt/nfs": (100 * TIB, 100 * TIB),
    "/etc/hosts": (7 * TIB, 6 * TIB),
}
FILE_MOUNTS = {"/etc/hosts", "/etc/hostname", "/etc/resolv.conf"}


def make_sysfs(root: Path, node: dict) -> None:
    """Materialise /sys entries of one inventory-fixture node under ``root``."""
    for bdf, real in node["pci_paths"].items():
        device = root / real.lstrip("/")
        device.mkdir(parents=True, exist_ok=True)
        link = root / "sys/bus/pci/devices" / bdf
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(device)
    for gpu in node["gpus"]:
        device = root / node["pci_paths"][gpu["pci_bus_id"]].lstrip("/")
        (device / "numa_node").write_text(f"{gpu['numa_node']}\n")
    for hca in node["hcas"]:
        device = root / node["pci_paths"][hca["pci_bus_id"]].lstrip("/")
        (device / "numa_node").write_text(f"{hca['numa_node']}\n")
        (device / "net" / hca["netdev"]).mkdir(parents=True)
        ib = root / "sys/class/infiniband" / hca["name"]
        ib.mkdir(parents=True)
        (ib / "device").symlink_to(device)
        (ib / "board_id").write_text(hca["board_id"] + "\n")
        (ib / "fw_ver").write_text(hca["fw_ver"] + "\n")
        for port in hca["ports"]:
            port_dir = ib / "ports" / str(port["port"])
            port_dir.mkdir(parents=True)
            (port_dir / "state").write_text(port["state"] + "\n")
            (port_dir / "rate").write_text(port["rate"] + "\n")
            (port_dir / "link_layer").write_text(port["link_layer"] + "\n")


def fake_host(root: Path, node: dict, tools=None) -> probe.Host:
    make_sysfs(root, node)
    (root / "proc").mkdir()
    (root / "proc/mounts").write_text(fixture("proc_mounts.txt"))
    (root / "proc/meminfo").write_text(fixture("meminfo.txt"))
    (root / "proc/cpuinfo").write_text(fixture("cpuinfo_x86.txt"))
    (root / "sys/fs/cgroup").mkdir(parents=True)
    (root / "sys/fs/cgroup/memory.max").write_text("max\n")
    (root / "sys/module/nvidia_peermem").mkdir(parents=True)
    return probe.Host(
        root=str(root),
        run=fake_run(h200_tools() if tools is None else tools),
        statvfs=fake_statvfs(DISK_SIZES),
        isdir=lambda path: path not in FILE_MOUNTS,
    )


# --------------------------------------------------------------- parsers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("00000000:18:00.0", "0000:18:00.0"),
        ("00000000:1B:00.0", "0000:1b:00.0"),
        ("00000009:01:00.1", "0009:01:00.1"),
        ("0000:9a:00.0", "0000:9a:00.0"),
        ("  00000000:DB:00.0\n", "0000:db:00.0"),
    ],
)
def test_normalize_bdf(raw: str, expected: str) -> None:
    assert probe.normalize_bdf(raw) == expected


def test_normalize_bdf_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not a PCI bus id"):
        probe.normalize_bdf("GPU-1234")


def test_parse_gpu_csv_h200() -> None:
    gpus = probe.parse_gpu_csv(fixture("gpu_query_h200.csv"))
    assert [g["index"] for g in gpus] == list(range(8))
    first = gpus[0]
    assert first == {
        "index": 0,
        "name": "NVIDIA H200",
        "uuid": "GPU-c08f2d3a-b4f5-6053-e160-6d15d5adb0f5",
        "pci_bus_id": "0000:18:00.0",
        "memory_mib": 143771,
        "numa_node": None,
        "compute_cap": "9.0",
        "pcie_gen_max": 5,
        "pcie_width_max": 16,
    }
    assert gpus[2]["pci_bus_id"] == "0000:3a:00.0"  # "3A" lower-cased


def test_parse_gpu_csv_na_fields_and_core_query() -> None:
    line = "0, NVIDIA H100 80GB HBM3, GPU-0b1c, 00000000:1B:00.0, 81559, [N/A], [N/A], [Not Supported]\n"
    (gpu,) = probe.parse_gpu_csv(line)
    assert gpu["compute_cap"] is None
    assert gpu["pcie_gen_max"] is None and gpu["pcie_width_max"] is None
    (core,) = probe.parse_gpu_csv(
        "3, NVIDIA B200, GPU-77, 00000000:9A:00.0, 183359\n",
        probe.GPU_QUERY_CORE_FIELDS,
    )
    assert core["memory_mib"] == 183359 and core["compute_cap"] is None


def test_parse_gpu_csv_comma_in_name_and_short_line() -> None:
    (gpu,) = probe.parse_gpu_csv(
        "0, Odd, Name, GPU-1, 00000000:18:00.0, 1000, 9.0, 5, 16\n"
    )
    assert gpu["name"] == "Odd, Name" and gpu["pcie_width_max"] == 16
    with pytest.raises(ValueError, match="fields"):
        probe.parse_gpu_csv("0, NVIDIA H200, GPU-1\n")


@pytest.mark.parametrize(
    ("rate", "gbps"),
    [
        ("400 Gb/sec (4X NDR)", 400.0),
        ("200 Gb/sec (4X HDR)", 200.0),
        ("100 Gb/sec (2X HDR)", 100.0),
        ("100 Gb/sec (2X EDR)", 100.0),
        ("2.5 Gb/sec (1X SDR)", 2.5),
        ("800 Gb/sec (4X XDR)", 800.0),
        ("", 0.0),
        ("invalid", 0.0),
    ],
)
def test_parse_rate_gbps(rate: str, gbps: float) -> None:
    assert probe.parse_rate_gbps(rate) == gbps


def test_parse_topo_h200_strips_ansi_and_maps_nic_legend() -> None:
    topo = probe.parse_topo(fixture("topo_h200_8nic.txt"))
    devices = [f"GPU{i}" for i in range(8)] + [f"mlx5_{i}" for i in range(8)]
    assert sorted(topo) == sorted(devices)
    assert all(sorted(row) == sorted(devices) for row in topo.values())
    assert topo["GPU0"]["GPU0"] == "X"
    assert topo["GPU0"]["GPU7"] == "NV18"
    assert topo["GPU3"]["mlx5_3"] == "PIX"
    assert topo["GPU3"]["mlx5_0"] == "NODE"
    assert topo["GPU3"]["mlx5_4"] == "SYS"
    assert topo["mlx5_5"]["GPU5"] == "PIX"
    assert topo["mlx5_0"]["mlx5_7"] == "SYS"
    flat = json.dumps(topo)
    assert "NIC" not in flat and "Affinity" not in flat and "\x1b" not in flat


def test_parse_topo_space_aligned_output() -> None:
    topo = probe.parse_topo(fixture("topo_h100_8nic_spaces.txt"))
    assert len(topo) == 16 and len(topo["GPU0"]) == 16
    assert topo["GPU0"]["mlx5_0"] == "PIX"
    assert topo["GPU0"]["mlx5_1"] == "PXB"
    assert topo["GPU1"]["mlx5_0"] == "PXB"
    assert topo["GPU6"]["mlx5_2"] == "SYS"


def test_parse_topo_pcie_only_rtx6000() -> None:
    topo = probe.parse_topo(fixture("topo_rtx6000_4nic.txt"))
    assert len(topo) == 12 and len(topo["GPU0"]) == 12
    assert topo["GPU0"]["GPU1"] == "PIX"
    assert topo["GPU0"]["GPU2"] == "NODE"
    assert topo["GPU0"]["GPU4"] == "SYS"
    assert topo["GPU5"]["mlx5_2"] == "PXB"
    assert topo["GPU5"]["mlx5_3"] == "NODE"
    links = {cell for row in topo.values() for cell in row.values()}
    assert {"PIX", "PXB", "NODE", "SYS"} <= links
    assert not any(cell.startswith("NV") for cell in links)


def test_parse_topo_legacy_hca_columns() -> None:
    topo = probe.parse_topo(fixture("topo_legacy_mlx_columns.txt"))
    assert sorted(topo) == ["GPU0", "GPU1", "GPU2", "GPU3", "mlx5_0", "mlx5_1"]
    assert topo["GPU2"]["mlx5_1"] == "PIX"
    assert topo["mlx5_1"]["mlx5_1"] == "X"


@pytest.mark.parametrize(
    "text",
    ["", "Failed to run topology matrix: Not Supported\n", "Legend:\n  X = Self\n"],
)
def test_parse_topo_without_matrix(text: str) -> None:
    assert probe.parse_topo(text) == {}


def test_parse_nic_legend() -> None:
    legend = probe.parse_nic_legend(fixture("topo_rtx6000_4nic.txt"))
    assert legend == {f"NIC{i}": f"mlx5_{i}" for i in range(4)}


def test_parse_host_files() -> None:
    assert probe.parse_meminfo_total(fixture("meminfo.txt")) == 2113348948 * 1024
    assert probe.parse_meminfo_total(None) == 0
    assert probe.parse_cpu_model(fixture("cpuinfo_x86.txt")) == (
        "Intel(R) Xeon(R) Platinum 8480+"
    )
    assert probe.parse_cpu_model(fixture("cpuinfo_aarch64.txt")) == ""
    assert probe.parse_cpu_model(fixture("lscpu_aarch64.txt")) == "Neoverse-V2"


@pytest.mark.parametrize(
    ("text", "limit"),
    [
        ("max\n", None),
        ("549755813888\n", 549755813888),
        ("9223372036854771712\n", None),  # cgroup v1 "unlimited"
        ("0\n", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_cgroup_limit(text: str | None, limit: int | None) -> None:
    assert probe.parse_cgroup_limit(text) == limit


def test_parse_mounts_decodes_octal_escapes() -> None:
    entries = probe.parse_mounts(fixture("proc_mounts.txt"))
    by_path = {e.mountpoint: e for e in entries}
    assert by_path["/scratch disk"].device == "/dev/nvme9n1p1"
    assert by_path["/mnt/volume"].fs_type == "fuse.s3fs"
    assert by_path["/"].device == "overlay"


@pytest.mark.parametrize(
    ("fs_type", "keep"),
    [
        ("ext4", True),
        ("xfs", True),
        ("btrfs", True),
        ("zfs", True),
        ("overlay", True),
        ("fuse.s3fs", False),
        ("fuseblk", False),
        ("nfs4", False),
        ("cifs", False),
        ("9p", False),
        ("tmpfs", False),
        ("ceph", False),
        ("squashfs", False),
        ("vfat", False),
    ],
)
def test_keep_fs_type(fs_type: str, keep: bool) -> None:
    assert probe.keep_fs_type(fs_type) is keep


def test_parse_tool_versions() -> None:
    banner = fixture("nvidia_smi_banner_h200.txt")
    assert probe.parse_cuda_version(banner) == "13.0"
    assert probe.parse_banner_driver(banner) == "580.95.05"
    assert probe.parse_cuda_version("no banner") is None
    assert probe.parse_ofed_version("MLNX_OFED_LINUX-24.10-1.1.4.0:\n") == (
        "MLNX_OFED_LINUX-24.10-1.1.4.0"
    )
    assert probe.parse_ofed_version("") is None
    assert probe.parse_nvlink_active(fixture("nvlink_status_h200.txt")) is True
    assert probe.parse_nvlink_active(fixture("nvlink_status_rtx6000.txt")) is False
    assert probe.parse_nvlink_active("GPU 0: X\n\t Link 0: <inactive>\n") is False


# --------------------------------------------------------------- readers


def test_probe_hcas_from_fake_sysfs(tmp_path: Path) -> None:
    node = inventory_payload("inventory_h200_4node.json")["nodes"][0]
    make_sysfs(tmp_path, node)
    # A down Ethernet port and a soft-RoCE device without a PCI function.
    down = tmp_path / "sys/class/infiniband/mlx5_10"
    (down / "ports/1").mkdir(parents=True)
    device = tmp_path / "sys/devices/pci0000:d0/0000:d0:01.0/0000:d1:00.0"
    device.mkdir(parents=True)
    (down / "device").symlink_to(device)
    (down / "ports/1/state").write_text("1: DOWN\n")
    (down / "ports/1/rate").write_text("40 Gb/sec (4X QDR)\n")
    (down / "ports/1/link_layer").write_text("Ethernet\n")
    (tmp_path / "sys/class/infiniband/rxe0/ports/1").mkdir(parents=True)

    hcas = probe.probe_hcas(probe.Host(root=str(tmp_path)))

    assert [h["name"] for h in hcas] == [f"mlx5_{i}" for i in range(8)] + ["mlx5_10"]
    first = hcas[0]
    assert first["pci_bus_id"] == "0000:19:00.0"
    assert first["numa_node"] == 0
    assert first["netdev"] == "ibp25s0"
    assert first["board_id"] == "MT_0000000838"
    assert first["ports"] == [
        {
            "port": 1,
            "state": "4: ACTIVE",
            "rate": "400 Gb/sec (4X NDR)",
            "rate_gbps": 400.0,
            "link_layer": "InfiniBand",
        }
    ]
    last = hcas[-1]
    assert last["numa_node"] is None and last["netdev"] is None
    assert last["ports"][0]["rate_gbps"] == 40.0


def test_probe_disks_keeps_large_local_filesystems(tmp_path: Path) -> None:
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/mounts").write_text(fixture("proc_mounts.txt"))
    host = probe.Host(
        root=str(tmp_path),
        statvfs=fake_statvfs(DISK_SIZES),
        isdir=lambda path: path not in FILE_MOUNTS,
    )
    disks = probe.probe_disks(host)
    assert [d["path"] for d in disks] == [
        "/",
        "/root",
        "/scratch disk",
        "/tank",
        "/workspace",
    ]
    workspace = disks[-1]
    assert workspace == {
        "path": "/workspace",
        "device": "/dev/nvme8n1p1",
        "fs_type": "xfs",
        "total_bytes": 3 * TIB,
        "free_bytes": 3 * TIB - 50 * GIB,
    }


def test_collect_payload_validates_as_node_inventory(tmp_path: Path) -> None:
    node = inventory_payload("inventory_h200_4node.json")["nodes"][0]
    payload = probe.collect(fake_host(tmp_path, node), control_interface="wg0")

    expected_keys = set(NodeInventory.model_fields) - {"name", "address"}
    assert set(payload) == expected_keys
    inv = NodeInventory.model_validate(
        {**payload, "name": "node0", "address": "10.42.0.1"}
    )
    assert [g.pci_bus_id for g in inv.gpus] == [g["pci_bus_id"] for g in node["gpus"]]
    assert [g.numa_node for g in inv.gpus] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [h.name for h in inv.active_hcas] == [f"mlx5_{i}" for i in range(8)]
    assert inv.pci_paths == node["pci_paths"]  # test root stripped from paths
    assert inv.topo == node["topo"]
    assert inv.driver_version == "580.95.05"
    assert inv.cuda_version == "13.0"
    assert inv.cpu_model == "Intel(R) Xeon(R) Platinum 8480+"
    assert inv.mem_total_bytes == 2113348948 * 1024
    assert inv.cgroup_mem_limit_bytes is None
    assert inv.docker_version == "27.5.1"
    assert inv.ofed_version == "MLNX_OFED_LINUX-24.10-1.1.4.0"
    assert inv.nvidia_peermem is True
    assert inv.nvlink_active is True
    assert inv.control_interface == "wg0"
    assert "/etc/hosts" not in {d.path for d in inv.disks}


def test_collect_tolerates_missing_optional_tools(tmp_path: Path) -> None:
    node = inventory_payload("inventory_h200_4node.json")["nodes"][0]
    tools = {GPU_QUERY: ok(fixture("gpu_query_h200.csv"))}  # nothing else installed
    payload = probe.collect(fake_host(tmp_path, node, tools), control_interface="ib0")
    assert payload["topo"] == {}
    assert payload["driver_version"] == "unknown"
    assert payload["cuda_version"] is None
    assert payload["docker_version"] is None
    assert payload["ofed_version"] is None
    assert payload["nvlink_active"] is False
    assert payload["control_interface"] == "ib0"
    NodeInventory.model_validate({**payload, "name": "node0", "address": "10.42.0.1"})


def test_collect_retries_core_fields_on_old_driver(tmp_path: Path) -> None:
    node = inventory_payload("inventory_h200_4node.json")["nodes"][0]
    core = "".join(
        ", ".join(line.split(", ")[:5]) + "\n"
        for line in fixture("gpu_query_h200.csv").splitlines()
    )
    tools = h200_tools()
    tools[GPU_QUERY] = probe.ToolResult(
        2, 'Field "compute_cap" is not a valid field to query.\n', ""
    )
    tools[GPU_CORE_QUERY] = ok(core)
    payload = probe.collect(fake_host(tmp_path, node, tools))
    assert len(payload["gpus"]) == 8
    assert payload["gpus"][0]["compute_cap"] is None
    assert payload["gpus"][0]["memory_mib"] == 143771


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ({}, "nvidia-smi not found"),
        (
            {
                GPU_QUERY: probe.ToolResult(
                    9,
                    "NVIDIA-SMI has failed because it couldn't communicate with the "
                    "NVIDIA driver.\n",
                    "",
                )
            },
            "couldn't communicate",
        ),
        ({GPU_QUERY: ok("")}, "no GPUs"),
    ],
)
def test_main_fails_fast_without_working_nvidia_smi(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tools, message: str
) -> None:
    host = probe.Host(root=str(tmp_path), run=fake_run(tools))
    assert probe.main([], host=host) == probe.EXIT_PROBE_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


def test_main_prints_exactly_one_json_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    node = inventory_payload("inventory_h200_4node.json")["nodes"][0]
    assert (
        probe.main(["--control-interface", "wg0"], host=fake_host(tmp_path, node)) == 0
    )
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert json.loads(out)["gpus"][7]["pci_bus_id"] == "0000:cd:00.0"


def test_probe_script_runs_standalone(tmp_path: Path) -> None:
    """Isolated interpreter, no repo on sys.path, no nvidia-smi on PATH."""
    proc = subprocess.run(
        [sys.executable, "-I", str(AGENTS / "probe.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": str(tmp_path)},
        check=False,
    )
    assert proc.returncode == probe.EXIT_PROBE_FAILED
    assert proc.stdout == ""
    assert "nvidia-smi not found" in proc.stderr


@pytest.mark.parametrize("script", ["probe.py", "telemetry_agent.py"])
def test_agent_scripts_are_standalone_python38(script: str) -> None:
    source = (AGENTS / script).read_text()
    tree = ast.parse(source, feature_version=(3, 8))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"__future__", "pynvml"}
    assert imported <= allowed, imported - allowed
    future = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    assert future and any(a.name == "annotations" for a in future[0].names)


# --------------------------------------------------------------- shared fixtures


def _common_depth(a: str, b: str) -> int:
    depth = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        depth += 1
    return depth


# "/sys/devices/pci0000:RR/<root port>/<switch upstream port>" = 6 components.
SAME_SWITCH_DEPTH = 6


def _switch_peers(node) -> dict[int, list[str]]:
    peers: dict[int, list[str]] = {}
    for gpu in node.gpus:
        gpu_path = node.pci_paths[gpu.pci_bus_id]
        peers[gpu.index] = [
            h.name
            for h in node.hcas
            if _common_depth(gpu_path, node.pci_paths[h.pci_bus_id])
            >= SAME_SWITCH_DEPTH
        ]
    return peers


def test_inventory_fixture_h200_4node() -> None:
    inventory = Inventory.model_validate_json(fixture("inventory_h200_4node.json"))
    assert [n.name for n in inventory.nodes] == ["node0", "node1", "node2", "node3"]
    assert inventory.gpu_count == 32
    uuids = {g.uuid for n in inventory.nodes for g in n.gpus}
    assert len(uuids) == 32
    for node in inventory.nodes:
        assert {g.name for g in node.gpus} == {"NVIDIA H200"}
        assert len(node.active_hcas) == 8
        assert {p.rate for h in node.hcas for p in h.ports} == {"400 Gb/sec (4X NDR)"}
        assert node.nvlink_active and node.nvidia_peermem
        for i in range(8):
            assert node.topo[f"GPU{i}"][f"mlx5_{i}"] == "PIX"
            assert node.gpus[i].numa_node == node.hcas[i].numa_node
        assert _switch_peers(node) == {i: [f"mlx5_{i}"] for i in range(8)}


def test_inventory_fixture_rtx6000_4node() -> None:
    inventory = Inventory.model_validate_json(fixture("inventory_rtx6000_4node.json"))
    assert len(inventory.nodes) == 4 and inventory.gpu_count == 32
    for node in inventory.nodes:
        assert {g.compute_cap for g in node.gpus} == {"12.0"}
        assert len(node.active_hcas) == 4
        assert not node.nvlink_active
        links = {cell for row in node.topo.values() for cell in row.values()}
        assert {"PIX", "PXB", "NODE", "SYS"} <= links
        assert _switch_peers(node) == {g: [f"mlx5_{g // 2}"] for g in range(8)}
        for g in range(8):
            assert node.topo[f"GPU{g}"][f"mlx5_{g // 2}"] == "PXB"


IP_ADDR_SHOW = (
    "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"
    "3: ens5f0    inet 10.0.0.11/24 brd 10.0.0.255 scope global ens5f0\\  x\n"
    "7: wg0    inet 10.42.0.1/24 scope global wg0\\       valid_lft forever\n"
)


def test_auto_control_interface_is_the_nic_that_holds_the_address(
    tmp_path: Path,
) -> None:
    assert probe.parse_interface_for(IP_ADDR_SHOW, "10.42.0.1") == "wg0"
    assert probe.parse_interface_for(IP_ADDR_SHOW, "10.0.0.11") == "ens5f0"
    assert probe.parse_interface_for(IP_ADDR_SHOW, "10.0.0.1") is None
    host = probe.Host(
        root=str(tmp_path),
        run=fake_run({("ip", "-o", "-4", "addr", "show"): ok(IP_ADDR_SHOW)}),
    )
    assert probe.resolve_control_interface(host, "auto", "10.0.0.11") == "ens5f0"
    assert probe.resolve_control_interface(host, "bond0", None) == "bond0"
    with pytest.raises(probe.ProbeError, match="not assigned to any interface"):
        probe.resolve_control_interface(host, "auto", "203.0.113.10")
    with pytest.raises(probe.ProbeError, match="needs --control-address"):
        probe.resolve_control_interface(host, "auto", None)
