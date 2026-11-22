# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agents/fabric_probe.py: parsers, sysfs readers and every subcommand.

Every collected document is validated with the domain models of
``domain/fabric.py``: that is the contract between the agent and its parser.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmarks.lium_cluster_bench.agents import fabric_probe as fp
from benchmarks.lium_cluster_bench.domain.fabric import (
    FabricNodeReport,
    PerfqueryReport,
    PerftestReport,
    TraceReport,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "fabric"
AGENT = Path(fp.__file__)
LEAF1 = "0xfc6a1c0300e7a1b0"
LEAF2 = "0xfc6a1c0300e7a2c8"
SPINE = "0xfc6a1c0300e9c000"


def text(name: str) -> str:
    """Fixture content without the repo SPDX header lines."""
    return "".join(
        line
        for line in (FIXTURES / name).read_text().splitlines(keepends=True)
        if not line.startswith("# SPDX-")
    )


def hex_bytes(name: str) -> bytes:
    return bytes.fromhex("".join(text(name).split()))


def ok(stdout: str) -> fp.ToolResult:
    return fp.ToolResult(0, stdout, "")


def fail(
    stderr: str = "ibwarn: mad_rpc: _do_madrpc failed", rc: int = 1
) -> fp.ToolResult:
    return fp.ToolResult(rc, "", stderr)


def materialize(root: Path, spec_name: str) -> None:
    """Write a recorded sysfs/dev tree (files, symlinks, empty dirs)."""
    spec = json.loads((FIXTURES / spec_name).read_text())
    for rel in spec["dirs"]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for rel, content in spec["files"].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for rel, target in spec["symlinks"].items():
        link = root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        (root / target).mkdir(parents=True, exist_ok=True)
        link.symlink_to(root / target)


class FakeFabric:
    """Answers smpquery/sminfo/ibtracert/perfquery like a small IB fabric.

    mlx5_0 hangs off LEAF1 port 17, mlx5_1 off LEAF2 port 17. On each leaf
    ports 5 and 18 lead to hosts, 33 and 34 to one spine; the rest are down.
    """

    ACTIVE = {5, 17, 18, 33, 34}

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.blocked = False
        self.rejects_x = False

    def leaf_of(self, hca: str) -> str:
        return LEAF1 if hca == "mlx5_0" else LEAF2

    def smpquery(self, argv: list[str]) -> fp.ToolResult:
        if self.blocked:
            return fail()
        if self.rejects_x and "-x" in argv:
            return fail("smpquery: invalid option -- 'x'")
        hca = argv[argv.index("-C") + 1]
        at = argv.index("-D")
        op, path, params = argv[at + 1], argv[at + 2], argv[at + 3 :]
        hops = path.split(",")
        leaf = self.leaf_of(hca)
        if op == "nodeinfo" and len(hops) == 2:
            return ok(text("smpquery_nodeinfo_switch.txt").replace(LEAF1, leaf))
        if op == "nodedesc" and len(hops) == 2:
            desc = text("smpquery_nodedesc_switch.txt")
            return ok(desc if leaf == LEAF1 else desc.replace("r0-01", "r1-01"))
        if op == "portinfo":
            port = int(params[0])
            if port == 0:
                port0 = text("smpquery_portinfo_port0.txt")
                return ok(port0 if leaf == LEAF1 else port0.replace("...3\n", "...4\n"))
            if port in self.ACTIVE:
                return ok(text("smpquery_portinfo_ndr.txt"))
            return ok(text("smpquery_portinfo_down.txt"))
        if op == "nodeinfo" and len(hops) == 3:
            port = int(hops[2])
            if port in (33, 34):
                return ok(text("smpquery_nodeinfo_spine.txt"))
            guid = "0xb83fd20300a51e44" if port == 18 else f"0x0002c90300{port:06x}"
            return ok(
                text("smpquery_nodeinfo_ca.txt").replace("0xb83fd20300a51e44", guid)
            )
        if op == "nodedesc" and len(hops) == 3:
            return ok(text("smpquery_nodedesc_spine.txt"))
        return fail()

    def __call__(
        self, argv: list[str], timeout: float | None = None
    ) -> fp.ToolResult | None:
        self.calls.append(list(argv))
        tool = argv[0]
        if tool == "smpquery":
            return self.smpquery(argv)
        if tool == "sminfo":
            return fail() if self.blocked else ok(text("sminfo.txt"))
        if tool == "ibv_devinfo":
            return ok(text("ibv_devinfo_mlx5_0.txt"))
        return None  # not installed

    def count(self, tool: str, op: str | None = None) -> int:
        return sum(1 for c in self.calls if c[0] == tool and (op is None or op in c))


def ib_host(
    tmp_path: Path, run: Callable[..., Any], tools: set[str] | None = None
) -> fp.Host:
    materialize(tmp_path, "sysfs_ib_ndr.json")
    installed = {"smpquery", "sminfo", "ibv_devinfo"} if tools is None else tools
    return fp.Host(
        root=str(tmp_path),
        run=run,
        which=lambda name: f"/usr/sbin/{name}" if name in installed else None,
        monotonic=lambda: 100.0,
        wall=lambda: 1_800_000_000.0,
        hostname=lambda: "gpu-h100-07",
        lldp_listener=lambda netdevs, wait: ({}, {}),
        ipv4_lookup=lambda netdev: [],
    )


# --------------------------------------------------------------- parsers


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        ("400 Gb/sec (4X NDR)", (400.0, "4X", "NDR")),
        ("800 Gb/sec (4X XDR)", (800.0, "4X", "XDR")),
        ("200 Gb/sec (2X NDR)", (200.0, "2X", "NDR")),
        ("100 Gb/sec (4X EDR)", (100.0, "4X", "EDR")),
        ("", (0.0, None, None)),
    ],
)
def test_parse_rate(rate: str, expected: tuple) -> None:
    assert fp.parse_rate(rate) == expected


def test_guid_gid_and_pci_helpers() -> None:
    assert fp.normalize_guid("b83f:d203:00a5:1b2c") == "0xb83fd20300a51b2c"
    assert fp.normalize_guid("0xFC6A1C0300E7A1B0") == LEAF1
    assert fp.normalize_guid("not-a-guid") is None
    gid = fp.normalize_gid("fe80:0000:0000:0000:b83f:d203:00a5:1b2c")
    assert fp.gid_prefix(gid) == "0xfe80000000000000"
    assert fp.gid_interface_id(gid) == "0xb83fd20300a51b2c"
    assert fp.normalize_gid("0000:0000:0000:0000:0000:0000:0000:0000") is None
    mapped = fp.normalize_gid("0000:0000:0000:0000:0000:ffff:0a64:000b")
    assert fp.gid_ipv4(mapped) == "10.100.0.11"
    assert fp.gid_ipv4(gid) is None
    assert fp.parse_pci_speed("32.0 GT/s PCIe") == 32.0
    assert fp.parse_pci_speed("8 GT/s") == 8.0
    assert fp.parse_pci_speed("Unknown") is None
    assert fp.parse_lid("0x1a") == 26 and fp.parse_lid("0x0") is None


def test_parse_nodeinfo_switch_and_ca() -> None:
    switch = fp.parse_nodeinfo(text("smpquery_nodeinfo_switch.txt"))
    assert switch == {
        "node_type": "Switch",
        "node_guid": LEAF1,
        "port_guid": LEAF1,
        "system_guid": LEAF1,
        "num_ports": 64,
        "local_port": 17,
        "desc": None,
        "dev_id": "0xd2f2",
        "vendor_id": "0x0002c9",
    }
    ca = fp.parse_nodeinfo(text("smpquery_nodeinfo_ca.txt"))
    assert ca["node_type"] == "Channel Adapter" and ca["num_ports"] == 1
    assert fp.parse_nodeinfo("ibwarn: [1234] mad_rpc: _do_madrpc failed\n") is None


def test_parse_nodedesc() -> None:
    assert fp.parse_nodedesc(text("smpquery_nodedesc_switch.txt")) == (
        "MF0;ib-leaf-r0-01:MQM9700/U1"
    )
    long_name = "Node Description:." + "x" * 40 + "\n"
    assert fp.parse_nodedesc(long_name) == "x" * 40
    assert fp.parse_nodedesc("") is None


def test_parse_portinfo_ndr_2x_down_xdr_and_port0() -> None:
    ndr = fp.parse_portinfo(text("smpquery_portinfo_ndr.txt"), 17)
    assert ndr["state"] == "Active" and ndr["phys_state"] == "LinkUp"
    assert ndr["width_active"] == "4X" and ndr["speed_active"] == "NDR"
    assert ndr["width_supported"] == ["1X", "2X", "4X"]
    assert ndr["speed_supported"] == ["SDR", "DDR", "QDR", "FDR", "EDR", "HDR", "NDR"]
    assert ndr["rate_gbps"] == 400.0 and ndr["max_rate_gbps"] == 400.0
    narrow = fp.parse_portinfo(text("smpquery_portinfo_2x.txt"), 18)
    assert narrow["width_active"] == "2X" and narrow["rate_gbps"] == 200.0
    down = fp.parse_portinfo(text("smpquery_portinfo_down.txt"), 40)
    assert down["state"] == "Down" and down["rate_gbps"] is None
    xdr = fp.parse_portinfo(text("smpquery_portinfo_xdr.txt"), 7)
    assert xdr["speed_active"] == "XDR" and xdr["rate_gbps"] == 800.0
    port0 = fp.parse_portinfo(text("smpquery_portinfo_port0.txt"), 0)
    assert port0["lid"] == 3  # capability lines (tab-indented) are skipped


def test_parse_sminfo_and_devinfo() -> None:
    assert fp.parse_sminfo(text("sminfo.txt")) == {
        "guid": LEAF1,
        "lid": 1,
        "priority": 15,
        "state": 3,
        "state_name": "SMINFO_MASTER",
        "activity_count": 3401542,
    }
    assert fp.parse_sminfo("sminfo: iberror: failed: query") is None
    assert fp.parse_ibv_devinfo(text("ibv_devinfo_mlx5_0.txt")) == {
        "active_mtu": 4096,
        "max_mtu": 4096,
    }


def test_parse_ibtracert_cross_spine_and_same_leaf() -> None:
    cross = fp.parse_ibtracert(text("ibtracert_cross_spine.txt"))
    assert cross["reached"] is True
    assert [h["guid"] for h in cross["hops"]] == [LEAF1, SPINE, LEAF2]
    assert [(h["in_port"], h["out_port"]) for h in cross["hops"]] == [
        (17, 33),
        (9, 21),
        (5, 5),
    ]
    assert cross["hops"][1]["desc"] == "MF0;ib-spine-01:MQM9700/U1"
    assert cross["src"]["lid"] == 26 and cross["dst"]["lid"] == 31
    same = fp.parse_ibtracert(text("ibtracert_same_leaf.txt"))
    assert len(same["hops"]) == 1 and same["hops"][0]["out_port"] == 18
    broken = fp.parse_ibtracert("ibtracert: iberror: failed: dest lid 31 not found\n")
    assert broken["reached"] is False and broken["hops"] == []


def test_parse_perfquery_extended_and_basic() -> None:
    ext = fp.parse_perfquery(text("perfquery_x.txt"))
    assert ext["PortXmitData"] == 183516234511870
    assert ext["PortXmitWait"] == 9123456789
    assert ext["PortXmitDiscards"] == 12 and ext["LinkDownedCounter"] == 1
    assert "PortSelect" not in ext and "CounterSelect" not in ext
    basic = fp.parse_perfquery(text("perfquery_basic.txt"))
    assert basic["PortXmitData"] == 0xFFFFFFFF  # saturated 32-bit counter
    assert "CounterSelect2" not in basic


def test_parse_perftest_bw_units() -> None:
    gbits = fp.parse_perftest_bw(text("ib_write_bw_client.txt"))
    assert gbits["unit"] == "Gb/sec" and gbits["msg_bytes"] == 65536
    assert gbits["bw_peak"] == 0.0  # -D mode prints no peak
    assert gbits["bw_avg_gbs"] == pytest.approx(381.85 / 8)
    assert gbits["msg_rate_mpps"] == pytest.approx(0.728335)
    mib = fp.parse_perftest_bw(text("ib_write_bw_mib_roce.txt"))
    assert mib["unit"] == "MB/sec" and mib["iterations"] == 5000
    assert mib["bw_avg_gbs"] == pytest.approx(23051.77 * 1048576 / 1e9)
    assert mib["bw_peak_gbs"] == pytest.approx(23154.28 * 1048576 / 1e9)
    assert fp.parse_perftest_bw(text("ib_write_bw_connect_failed.txt")) is None


def test_parse_perftest_lat_columns() -> None:
    lat = fp.parse_perftest_lat(text("ib_write_lat_client.txt"))
    assert lat == {
        "msg_bytes": 2,
        "iterations": 2000,
        "t_min_us": 1.71,
        "t_max_us": 5.02,
        "t_typical_us": 1.75,
        "t_avg_us": 1.76,
        "t_stdev_us": 0.05,
        "p99_us": 1.89,
        "p999_us": 5.02,
    }
    old = (
        " #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]\n"
        " 2       1000          0.98           3.10         1.02\n"
    )
    assert fp.parse_perftest_lat(old)["t_typical_us"] == 1.02
    assert fp.parse_perftest_lat(old)["p99_us"] is None


def test_parse_ip_json_and_mlnx_qos() -> None:
    addrs = fp.parse_ip_addr_json((FIXTURES / "ip_addr_roce.json").read_text())
    assert addrs["ens1f0np0"] == [{"address": "10.100.0.11", "prefixlen": 24}]
    assert addrs["wg0"][0]["address"] == "10.42.0.3"
    assert fp.parse_ip_addr_json("not json") == {}
    qos = fp.parse_mlnx_qos(text("mlnx_qos_ens1f0np0.txt"))
    assert qos == {"source": "mlnx_qos", "trust": "dscp", "pfc_enabled": [3]}


def test_parse_lldpctl_list_and_dict_forms() -> None:
    multi = fp.parse_lldpctl_json((FIXTURES / "lldpctl_multi.json").read_text())
    assert sorted(multi) == ["ens1f0np0", "ens2f0np0"]
    first = multi["ens1f0np0"]
    assert first["system_name"] == "roce-leaf-r0-01"
    assert first["chassis_id"] == "1c:34:da:a5:b0:00"
    assert first["chassis_id_subtype"] == "mac"
    assert first["port_id"] == "swp17" and first["ttl_s"] == 120
    assert first["max_frame_size"] == 9216
    assert first["mgmt_addresses"] == ["192.168.10.21", "fe80::1e34:daff:fea5:b000"]
    assert first["port_vlan_id"] == 100
    assert first["vlans"] == [{"vid": 100, "name": "vlan100"}]
    assert first["pfc"] is None  # lldpd does not decode DCBX
    assert multi["ens2f0np0"]["mgmt_addresses"] == ["192.168.10.22"]
    single = fp.parse_lldpctl_json((FIXTURES / "lldpctl_single.json").read_text())
    entry = single["ens1f0np0"]
    assert entry["system_name"] is None and entry["chassis_id"] == "b8:59:9f:12:34:00"
    assert entry["port_id"] == "Ethernet8" and entry["port_id_subtype"] == "local"
    assert entry["port_descr"] == "etp3" and entry["max_frame_size"] == 9100
    assert fp.parse_lldpctl_json("") == {}


def test_parse_raw_lldp_frame_with_dcbx() -> None:
    frame = hex_bytes("lldp_frame_dcbx.txt")
    record = fp.parse_lldp_frame(frame)
    assert record["source"] == "raw"
    assert record["chassis_id"] == "1c:34:da:a5:b0:00"
    assert record["chassis_id_subtype"] == "mac"
    assert record["port_id"] == "swp17" and record["port_id_subtype"] == "ifname"
    assert record["ttl_s"] == 120
    assert record["system_name"] == "roce-leaf-r0-01"
    assert record["system_descr"].startswith("Cumulus Linux version 5.9.1")
    assert record["mgmt_addresses"] == ["192.168.10.21"]
    assert record["port_vlan_id"] == 100
    assert record["vlans"] == [{"vid": 100, "name": "vlan100"}]
    assert record["max_frame_size"] == 9216
    assert record["pfc"] == {"willing": False, "mbc": False, "cap": 8, "enabled": [3]}
    assert record["ets"] == {
        "willing": False,
        "cbs": False,
        "max_tcs": 8,
        "prio_tc": [0, 0, 0, 3, 0, 0, 6, 0],
        "tc_bw": [50, 0, 0, 50, 0, 0, 0, 0],
        "tsa": [2, 0, 0, 2, 0, 0, 0, 0],
    }
    assert record["app_priority"] == [
        {"priority": 3, "selector": 3, "protocol": 4791},
        {"priority": 6, "selector": 5, "protocol": 48},
    ]
    assert record["unknown_tlvs"] == ["00-12-0f/1", "00-1b-21/2"]
    tagged = frame[:12] + b"\x81\x00\x00\x64" + frame[12:]
    assert fp.parse_lldp_frame(tagged)["system_name"] == "roce-leaf-r0-01"
    assert fp.parse_lldp_frame(frame[:12] + b"\x08\x00" + frame[14:]) is None
    assert fp.parse_lldp_frame(b"\x00" * 10) is None


def test_parse_sock_extended_err() -> None:
    ttl = fp.parse_sock_extended_err(hex_bytes("errqueue_ttl_exceeded.txt"))
    assert ttl == {
        "errno": 113,
        "origin": fp.SO_EE_ORIGIN_ICMP,
        "type": 11,
        "code": 0,
        "info": 0,
        "data": 0,
        "offender": "10.100.0.1",
    }
    unreach = fp.parse_sock_extended_err(hex_bytes("errqueue_port_unreach.txt"))
    assert (unreach["type"], unreach["code"]) == (3, 3)
    assert unreach["offender"] == "10.100.1.12"
    short = fp.parse_sock_extended_err(hex_bytes("errqueue_ttl_exceeded.txt")[:16])
    assert short is not None and short["offender"] is None
    assert fp.parse_sock_extended_err(b"\x00" * 8) is None


# --------------------------------------------------------------- sysfs readers


def test_read_ports_ib_tree(tmp_path: Path) -> None:
    materialize(tmp_path, "sysfs_ib_ndr.json")
    ports = fp.read_ports(fp.Host(root=str(tmp_path)))
    assert [p["hca"] for p in ports] == ["mlx5_0", "mlx5_1", "mlx5_2"]  # rxe0 skipped
    first = ports[0]
    assert first["pci_bus_id"] == "0000:19:00.0"
    assert first["link_layer"] == "InfiniBand" and fp.is_active(first)
    assert (first["rate_gbps"], first["width"], first["speed"]) == (400.0, "4X", "NDR")
    assert (first["lid"], first["sm_lid"], first["sm_sl"], first["lmc"]) == (
        26,
        1,
        0,
        0,
    )
    assert first["node_guid"] == "0xb83fd20300a51b2c"
    assert first["port_guid"] == "0xb83fd20300a51b2c"
    assert first["subnet_prefix"] == "0xfe80000000000000"
    assert first["gids"] == [
        {
            "index": 0,
            "gid": "fe80:0000:0000:0000:b83f:d203:00a5:1b2c",
            "type": "IB/RoCE v1",
            "ndev": None,
        }
    ]
    assert first["roce_v2_gid_index"] is None
    assert first["netdev"] == "ibp25s0" and first["mtu"] == 4092
    assert first["node_desc"] == "gpu-h100-07 mlx5_0"
    assert first["pcie"] == {
        "current_speed_gts": 32.0,
        "current_width": 16,
        "max_speed_gts": 32.0,
        "max_width": 16,
    }
    assert ports[1]["pcie"]["current_width"] == 8  # trained x8 in an x16 slot
    down = ports[2]
    assert not fp.is_active(down) and down["lid"] is None
    assert down["phys_state"] == "3: Disabled"


def test_read_ports_roce_tree(tmp_path: Path) -> None:
    materialize(tmp_path, "sysfs_roce.json")
    ports = fp.read_ports(fp.Host(root=str(tmp_path)))
    assert [(p["hca"], p["netdev"]) for p in ports] == [
        ("mlx5_0", "ens1f0np0"),
        ("mlx5_1", "ens2f0np0"),
    ]
    first = ports[0]
    assert first["link_layer"] == "Ethernet"
    assert first["lid"] is None and first["sm_lid"] is None
    assert [g["index"] for g in first["gids"]] == [0, 1, 2, 3]  # zero GIDs skipped
    assert first["roce_v2_gid_index"] == 3
    assert first["roce_v2_ipv4"] == "10.100.0.11"
    assert first["mtu"] == 9000 and first["mac"] == "5c:25:73:4b:0a:10"


# --------------------------------------------------------------- identify


def test_identify_ib_with_mad_census(tmp_path: Path) -> None:
    fabric = FakeFabric()
    doc = fp.collect_identify(ib_host(tmp_path, fabric), lldp_wait_s=0)
    report = FabricNodeReport.model_validate(doc)  # the agent <-> domain contract

    assert report.hostname == "gpu-h100-07" and report.umad is True
    assert report.tools["smpquery"] and not report.tools["perfquery"]
    first = report.ports[0]
    assert first.active_mtu == 4096
    neighbor = first.neighbor
    assert neighbor.node_guid == LEAF1 and neighbor.is_switch
    assert (neighbor.local_port, neighbor.num_ports, neighbor.lid) == (17, 64, 3)
    assert neighbor.desc == "MF0;ib-leaf-r0-01:MQM9700/U1"
    assert neighbor.link.rate_gbps == 400.0 and neighbor.link.port == 17
    assert report.ports[1].neighbor.node_guid == LEAF2
    assert report.ports[1].neighbor.lid == 4
    assert report.ports[2].neighbor is None  # down port: never queried

    (sm,) = report.sm
    assert sm.state_name == "SMINFO_MASTER" and sm.ports == ("mlx5_0:1", "mlx5_1:1")

    leaf = report.switches[LEAF1]
    assert leaf.census == "complete" and leaf.via == "mlx5_0:1"
    assert len(leaf.ports) == 64
    by_port = {p.port: p for p in leaf.ports}
    assert by_port[17].peer.node_guid == first.node_guid  # ourselves, no SMP
    assert by_port[18].peer.node_type == "Channel Adapter"
    assert by_port[33].peer.is_switch
    assert by_port[33].peer.desc == "MF0;ib-spine-01:MQM9700/U1"
    assert by_port[40].peer is None and by_port[40].state == "Down"
    assert report.switches[LEAF2].census == "complete"

    # Budget: 6 identity SMPs per port, then per switch 64 PortInfo,
    # 4 NodeInfo (active ports but ours) and 2 for one spine NodeDescription.
    assert report.smp.used == 2 * 6 + 2 * (64 + 4 + 2)
    assert report.smp.blocked is False and report.smp.exhausted is False
    assert fabric.count("smpquery", "portinfo") == 2 * (2 + 64)
    assert all("-x" in c for c in fabric.calls if "portinfo" in c)
    assert not report.lldp and not report.qos
    assert all(n.source != "smpquery" for n in report.notes)


def test_identify_smp_budget_makes_census_partial(tmp_path: Path) -> None:
    fabric = FakeFabric()
    doc = fp.collect_identify(ib_host(tmp_path, fabric), lldp_wait_s=0, max_smp=60)
    report = FabricNodeReport.model_validate(doc)
    assert report.smp.used <= 60 and report.smp.exhausted is True
    leaf = report.switches[LEAF1]
    assert leaf.census == "partial" and 0 < len(leaf.ports) < 64
    assert ("smpquery", "smp budget of 60 exhausted") in {
        (n.source, n.error) for n in report.notes
    }
    assert any(n.source == "census" and n.target == LEAF1 for n in report.notes)


def test_identify_census_only_and_no_census(tmp_path: Path) -> None:
    fabric = FakeFabric()
    host = ib_host(tmp_path, fabric)
    only = FabricNodeReport.model_validate(
        fp.collect_identify(host, lldp_wait_s=0, census_only=["mlx5_1:1"])
    )
    assert only.switches[LEAF1].census == "none"  # identity only
    assert only.switches[LEAF2].census == "complete"
    none = FabricNodeReport.model_validate(
        fp.collect_identify(host, lldp_wait_s=0, census=False)
    )
    assert {g: s.census for g, s in none.switches.items()} == {
        LEAF1: "none",
        LEAF2: "none",
    }
    assert none.smp.used == 12


def test_identify_stops_when_the_first_smp_fails(tmp_path: Path) -> None:
    fabric = FakeFabric()
    fabric.blocked = True
    doc = fp.collect_identify(ib_host(tmp_path, fabric), lldp_wait_s=0)
    report = FabricNodeReport.model_validate(doc)
    assert report.smp.blocked is True
    assert fabric.count("smpquery") == 1 and fabric.count("sminfo") == 0
    assert all(p.neighbor is None for p in report.ports)
    assert not report.sm and not report.switches
    assert ("smpquery", "smp_blocked", "mlx5_0:1") in {
        (n.source, n.error, n.target) for n in report.notes
    }
    assert report.mad_ok is False


def test_identify_retries_portinfo_without_extended_speeds(tmp_path: Path) -> None:
    fabric = FakeFabric()
    fabric.rejects_x = True
    doc = fp.collect_identify(ib_host(tmp_path, fabric), lldp_wait_s=0, census=False)
    report = FabricNodeReport.model_validate(doc)
    assert report.ports[0].neighbor.lid == 3
    portinfo = [c for c in fabric.calls if "portinfo" in c]
    assert "-x" in portinfo[0] and "-x" not in portinfo[-1]


def test_identify_without_umad_or_tools_is_sysfs_only(tmp_path: Path) -> None:
    fabric = FakeFabric()
    host = ib_host(tmp_path, fabric)
    for umad in (tmp_path / "dev/infiniband").glob("umad*"):
        umad.unlink()
    report = FabricNodeReport.model_validate(fp.collect_identify(host, lldp_wait_s=0))
    assert report.umad is False and fabric.count("smpquery") == 0
    assert ("umad", "no /dev/infiniband/umad* device") in {
        (n.source, n.error) for n in report.notes
    }
    assert report.ports[0].lid == 26  # tier S still complete

    bare = ib_host(tmp_path / "bare", FakeFabric(), tools=set())
    report = FabricNodeReport.model_validate(fp.collect_identify(bare, lldp_wait_s=0))
    assert ("smpquery", "not installed") in {(n.source, n.error) for n in report.notes}
    assert report.ports[0].active_mtu is None  # no ibv_devinfo


def roce_host(
    tmp_path: Path,
    *,
    tools: set[str],
    frames: dict[str, bytes],
    runs: dict[str, fp.ToolResult],
) -> tuple[fp.Host, list[tuple[tuple[str, ...], float]]]:
    materialize(tmp_path, "sysfs_roce.json")
    listened: list[tuple[tuple[str, ...], float]] = []

    def listener(netdevs, wait):
        listened.append((tuple(netdevs), wait))
        return dict(frames), {}

    def run(argv, timeout=None):
        return runs.get(argv[0])

    host = fp.Host(
        root=str(tmp_path),
        run=run,
        which=lambda name: f"/usr/bin/{name}" if name in tools else None,
        monotonic=lambda: 5.0,
        hostname=lambda: "gpu-b200-03",
        lldp_listener=listener,
        ipv4_lookup=lambda netdev: [{"address": "192.0.2.1", "prefixlen": 32}],
    )
    return host, listened


def test_identify_roce_raw_lldp_and_qos(tmp_path: Path) -> None:
    host, listened = roce_host(
        tmp_path,
        tools={"ip", "mlnx_qos"},
        frames={"ens1f0np0": hex_bytes("lldp_frame_dcbx.txt")},
        runs={
            "ip": ok((FIXTURES / "ip_addr_roce.json").read_text()),
            "mlnx_qos": ok(text("mlnx_qos_ens1f0np0.txt")),
        },
    )
    report = FabricNodeReport.model_validate(fp.collect_identify(host, lldp_wait_s=7))
    assert listened == [(("ens1f0np0", "ens2f0np0"), 7)]
    assert [p.ipv4[0].address for p in report.ports] == ["10.100.0.11", "10.101.0.11"]
    assert report.ports[0].ip.address == "10.100.0.11"
    lldp = report.lldp["ens1f0np0"]
    assert lldp.source == "raw" and lldp.pfc.enabled == (3,)
    assert "ens2f0np0" not in report.lldp
    assert ("lldp", "no LLDP frame within 7s", "ens2f0np0") in {
        (n.source, n.error, n.target) for n in report.notes
    }
    assert report.qos["ens1f0np0"].trust == "dscp"
    assert report.qos["ens1f0np0"].pfc_enabled == (3,)
    assert all(n.source != "umad" for n in report.notes)  # no IB port: no MAD


def test_identify_roce_prefers_lldpctl_and_falls_back_to_ioctl(tmp_path: Path) -> None:
    host, listened = roce_host(
        tmp_path,
        tools={"lldpctl"},
        frames={},
        runs={"lldpctl": ok((FIXTURES / "lldpctl_multi.json").read_text())},
    )
    report = FabricNodeReport.model_validate(fp.collect_identify(host))
    assert listened == []  # lldpctl answered for every RoCE netdev
    assert report.lldp["ens2f0np0"].system_name == "roce-leaf-r1-01"
    assert report.ports[0].ipv4[0].address == "192.0.2.1"  # SIOCGIFADDR path

    host, listened = roce_host(tmp_path / "x", tools=set(), frames={}, runs={})
    report = FabricNodeReport.model_validate(fp.collect_identify(host, lldp_wait_s=0))
    assert listened == []
    assert ("lldp", "listening disabled (--lldp-wait 0)") in {
        (n.source, n.error) for n in report.notes
    }


# --------------------------------------------------------------- trace


class FakeUdpProbe:
    """Error-queue answers per TTL: {ttl: (cmsg fixture, dst port shift)}."""

    def __init__(self, answers: dict[int, tuple[str, int]], clock: list[float]):
        self.answers = answers
        self.clock = clock
        self.sent: list[tuple[str, int, int]] = []
        self.closed = False

    def send(self, dst_ip: str, port: int, ttl: int, payload: bytes) -> None:
        self.sent.append((dst_ip, port, ttl))
        assert payload.startswith(fp.TRACE_MAGIC)

    def wait_error(self, timeout_s: float):
        self.clock[0] += min(timeout_s, 0.01)
        _, port, ttl = self.sent[-1]
        answer = self.answers.pop(ttl, None)
        if answer is None:
            self.clock[0] += timeout_s
            return None
        name, shift = answer
        cmsg = (fp.SOL_IP, fp.IP_RECVERR, hex_bytes(name))
        return b"", [cmsg], ("10.100.1.12", port + shift)

    def close(self) -> None:
        self.closed = True


def test_udp_trace_counts_routers_and_ignores_late_answers() -> None:
    clock = [0.0]
    probe = FakeUdpProbe(
        {
            1: ("errqueue_ttl_exceeded.txt", 0),
            # TTL 2: first a late duplicate for TTL 1 (port mismatch), dropped.
            2: ("errqueue_ttl_exceeded.txt", -1),
            3: ("errqueue_port_unreach.txt", 0),
        },
        clock,
    )
    host = fp.Host(monotonic=lambda: clock[0], udp_probe=lambda src, dev: probe)
    result = fp.udp_trace(host, "10.100.0.11", "10.100.1.12", "ens1f0np0")
    assert result["reached"] is True and result["hop_count"] == 2
    assert [h["ip"] for h in result["hops"]] == ["10.100.0.1", None]
    assert result["hops"][0]["icmp_type"] == 11 and result["hops"][0]["ttl"] == 1
    assert [s[2] for s in probe.sent] == [1, 2, 3]
    assert probe.sent[0][1] == fp.UDP_PORT_BASE + 1
    assert probe.closed and result["error"] is None


def test_udp_trace_same_segment_and_silent_destination() -> None:
    clock = [0.0]
    probe = FakeUdpProbe({1: ("errqueue_port_unreach.txt", 0)}, clock)
    host = fp.Host(monotonic=lambda: clock[0], udp_probe=lambda src, dev: probe)
    direct = fp.udp_trace(host, "10.100.0.11", "10.100.1.12", None)
    assert direct["reached"] is True and direct["hop_count"] == 0
    assert direct["hops"] == []

    silent = FakeUdpProbe({}, clock)
    host = fp.Host(monotonic=lambda: clock[0], udp_probe=lambda src, dev: silent)
    result = fp.udp_trace(host, "10.100.0.11", "10.100.9.9", None, max_ttl=3)
    assert result["reached"] is False and result["hop_count"] is None
    assert len(result["hops"]) == 3 and result["error"] == (
        "destination silent up to ttl 3"
    )

    def refuse(src, dev):
        raise OSError(99, "Cannot assign requested address")

    host = fp.Host(monotonic=lambda: 0.0, udp_probe=refuse)
    broken = fp.udp_trace(host, "10.9.9.9", "10.100.1.12", None)
    assert broken["error"].startswith("socket:") and broken["reached"] is False


def test_collect_trace_ib_and_roce_targets(tmp_path: Path) -> None:
    (tmp_path / "dev/infiniband").mkdir(parents=True)
    (tmp_path / "dev/infiniband/umad0").write_text("")
    calls: list[list[str]] = []

    def run(argv, timeout=None):
        calls.append(list(argv))
        dst = argv[-1]
        if dst == "31":
            return ok(text("ibtracert_cross_spine.txt"))
        if dst == "27":
            return ok(text("ibtracert_same_leaf.txt"))
        return fail("ibtracert: iberror: failed: dest lid not found")

    clock = [0.0]
    probe = FakeUdpProbe({1: ("errqueue_port_unreach.txt", 0)}, clock)
    host = fp.Host(
        root=str(tmp_path),
        run=run,
        which=lambda name: "/usr/sbin/ibtracert" if name == "ibtracert" else None,
        monotonic=lambda: clock[0],
        hostname=lambda: "gpu-h100-07",
        udp_probe=lambda src, dev: probe,
    )
    targets = [
        {"rail": 0, "hca": "mlx5_0", "port": 1, "src_lid": 26, "dst_lid": 31,
         "src_node": "node0", "dst_node": "node4"},
        {"rail": 0, "hca": "mlx5_0", "port": 1, "src_lid": 26, "dst_lid": 27,
         "src_node": "node0", "dst_node": "node1"},
        {"rail": 1, "hca": "mlx5_1", "port": 1, "src_lid": 27, "dst_lid": 99},
        {"rail": 2, "hca": "mlx5_2", "port": 1, "src_ip": "10.100.0.11",
         "dst_ip": "10.100.1.12", "netdev": "ens1f0np0"},
        {"rail": 3, "hca": "mlx5_3", "port": 1},
    ]  # fmt: skip
    report = TraceReport.model_validate(fp.collect_trace(host, targets))
    cross, same, missing, udp, nothing = report.results
    assert cross.method == "ibtracert" and cross.hop_count == 3
    assert cross.target.dst_node == "node4"
    assert [h.guid for h in cross.hops] == [LEAF1, SPINE, LEAF2]
    assert same.hop_count == 1 and same.reached
    assert missing.reached is False and "dest lid not found" in missing.error
    assert udp.method == "udp" and udp.hop_count == 0
    assert nothing.method == "none" and nothing.error
    assert calls[0][:5] == ["ibtracert", "-C", "mlx5_0", "-P", "1"]
    assert calls[0][-2:] == ["26", "31"]


def test_collect_trace_without_umad_falls_back_to_none(tmp_path: Path) -> None:
    host = fp.Host(root=str(tmp_path), which=lambda name: "/usr/sbin/ibtracert")
    targets = [{"rail": 0, "hca": "mlx5_0", "port": 1, "src_lid": 26, "dst_lid": 31}]
    report = TraceReport.model_validate(fp.collect_trace(host, targets))
    assert report.results[0].method == "none"
    assert report.results[0].error == "ibtracert not installed and no IPs"
    assert report.notes[0].source == "umad"


# --------------------------------------------------------------- perfquery


def test_collect_perfquery_modes_and_blocked_switch(tmp_path: Path) -> None:
    (tmp_path / "dev/infiniband").mkdir(parents=True)
    (tmp_path / "dev/infiniband/umad0").write_text("")
    calls: list[list[str]] = []

    def run(argv, timeout=None):
        calls.append(list(argv))
        lid = argv[-2]
        extended = "-x" in argv
        if lid == "3":
            return ok(text("perfquery_x.txt" if extended else "perfquery_basic.txt"))
        if lid == "4":  # an old switch: no PortCountersExtended
            if extended:
                return fail("perfquery: iberror: failed: perfextquery")
            return ok(text("perfquery_basic.txt").replace("4294967295", "123456"))
        return fail("ibwarn: mad_rpc: _do_madrpc failed; dport (Lid 5)")

    ticks = iter(range(10_000))
    host = fp.Host(
        root=str(tmp_path),
        run=run,
        which=lambda name: "/usr/sbin/perfquery",
        monotonic=lambda: float(next(ticks)),
        wall=lambda: 1_800_000_000.0,
    )
    specs = [
        {"key": f"{lid}:{p}", "hca": "mlx5_0", "port": 1, "lid": lid, "switch_port": p}
        for lid in (3, 4, 5)
        for p in (17, 18, 33)
    ]
    report = PerfqueryReport.model_validate(fp.collect_perfquery(host, specs))
    readings = {r.key: r for r in report.ports}
    assert [r.key for r in report.ports] == [s["key"] for s in specs]
    good = readings["3:17"]
    assert good.extended and good.error is None
    assert good.counters["PortXmitData"] == 183516234511870
    assert "PortXmitData" in good.wide and "PortXmitWait" in good.wide
    old = readings["4:18"]
    assert not old.extended and old.counters["PortXmitData"] == 123456
    assert "PortXmitData" not in old.wide
    assert readings["5:17"].error.startswith("-x: exit 1")
    assert readings["5:33"].error == "skipped: switch PMA failed"
    assert ("perfquery", "pma_blocked", "lid 5") in {
        (n.source, n.error, n.target) for n in report.notes
    }
    # The old switch's first port taught the rest not to try -x.
    lid4 = [c for c in calls if c[-2] == "4"]
    assert sum("-x" in c for c in lid4) == 1 and len(lid4) == 4
    assert len([c for c in calls if c[-2] == "5"]) == 2


def test_collect_perfquery_needs_tool_and_umad(tmp_path: Path) -> None:
    spec = [{"hca": "mlx5_0", "port": 1, "lid": 3, "switch_port": 17}]
    none = fp.collect_perfquery(fp.Host(root=str(tmp_path), which=lambda n: None), spec)
    assert none["ports"] == [] and none["notes"][0]["error"] == "not installed"
    no_umad = fp.collect_perfquery(
        fp.Host(root=str(tmp_path), which=lambda n: "/x"), spec
    )
    assert no_umad["notes"][0]["source"] == "umad"


# --------------------------------------------------------------- perftest


RAILS = [
    {"rail": 0, "hca": "mlx5_0", "port": 1, "gid_index": None, "bw_tcp_port": 18600,
     "lat_tcp_port": 18616},
    {"rail": 1, "hca": "mlx5_1", "port": 1, "gid_index": 3, "bw_tcp_port": 18601,
     "lat_tcp_port": 18617},
]  # fmt: skip


def perftest_host(
    run: Callable[..., Any],
    tools: set[str],
    pokes: list[tuple[str, int]] | None = None,
) -> fp.Host:
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def poke(address: str, port: int) -> bool:
        if pokes is not None:
            pokes.append((address, port))
        return True

    return fp.Host(
        run=run,
        which=lambda name: f"/usr/bin/{name}" if name in tools else None,
        monotonic=lambda: clock[0],
        sleep=sleep,
        hostname=lambda: "gpu-h100-08",
        poke=poke,
    )


def test_perftest_client_retries_until_the_server_listens() -> None:
    attempts: dict[tuple[str, str], int] = {}

    def run(argv, timeout=None):
        key = (argv[0], argv[2])
        attempts[key] = attempts.get(key, 0) + 1
        if argv[0] == "ib_write_bw" and argv[2] == "mlx5_0" and attempts[key] == 1:
            return fp.ToolResult(1, text("ib_write_bw_connect_failed.txt"), "")
        if argv[0] == "ib_write_bw":
            name = (
                "ib_write_bw_client.txt"
                if argv[2] == "mlx5_0"
                else "ib_write_bw_mib_roce.txt"
            )
            return ok(text(name))
        return ok(text("ib_write_lat_client.txt"))

    cfg = fp.PerftestConfig(role="client", peer="10.42.0.2", tests=("bw", "lat"))
    doc = fp.collect_perftest(
        perftest_host(run, {"ib_write_bw", "ib_write_lat"}), cfg, RAILS
    )
    report = PerftestReport.model_validate(doc)
    assert report.role == "client" and report.peer == "10.42.0.2"
    assert [(r.test, r.rail) for r in report.results] == [
        ("bw", 0), ("bw", 1), ("lat", 0), ("lat", 1),
    ]  # fmt: skip
    bw0, bw1, lat0, _ = report.results
    assert bw0.attempts == 2 and bw0.error is None
    assert bw0.bw.bw_avg_gbs == pytest.approx(381.85 / 8)
    assert bw1.bw.unit == "MB/sec"
    assert lat0.lat.t_typical_us == 1.75
    assert list(bw0.argv) == [
        "ib_write_bw", "-d", "mlx5_0", "-i", "1", "-p", "18600", "-D", "3",
        "-s", "65536", "-q", "2", "--report_gbits", "-F", "10.42.0.2",
    ]  # fmt: skip
    assert bw1.argv[-3:] == ("-x", "3", "10.42.0.2")  # RoCE GID index
    assert list(lat0.argv[:9]) == [
        "ib_write_lat", "-d", "mlx5_0", "-i", "1", "-p", "18616", "-n", "2000",
    ]  # fmt: skip


def test_perftest_client_failure_releases_the_waiting_server() -> None:
    def run(argv, timeout=None):
        if argv[2] == "mlx5_1":
            return fp.ToolResult(1, "", "Couldn't get device context for mlx5_1")
        return ok(text("ib_write_bw_client.txt"))

    pokes: list[tuple[str, int]] = []
    cfg = fp.PerftestConfig(role="client", peer="10.42.0.2", tests=("bw",))
    doc = fp.collect_perftest(perftest_host(run, {"ib_write_bw"}, pokes), cfg, RAILS)
    good, bad = PerftestReport.model_validate(doc).results
    assert good.error is None and bad.attempts == 1  # not a connect error: no retry
    assert bad.error == "exit 1: Couldn't get device context for mlx5_1"
    assert pokes == [("10.42.0.2", 18601)]


def test_perftest_server_runs_every_rail_and_reports_failures() -> None:
    seen: list[tuple[str, ...]] = []

    def run(argv, timeout=None):
        seen.append(tuple(argv))
        if argv[0] == "ib_write_bw" and argv[2] == "mlx5_1":
            return fp.ToolResult(124, "", "timed out after 60.0s")
        return ok(text("ib_write_bw_client.txt"))

    cfg = fp.PerftestConfig(role="server", peer=None, tests=("bw", "lat"), timeout_s=60)
    doc = fp.collect_perftest(perftest_host(run, {"ib_write_bw"}), cfg, RAILS)
    report = PerftestReport.model_validate(doc)
    assert [r.test for r in report.results] == ["bw", "bw"]  # lat tool missing
    assert ("ib_write_lat", "not installed") in {
        (n.source, n.error) for n in report.notes
    }
    assert report.results[1].error == "exit 124: timed out after 60.0s"
    assert all("10.42.0.2" not in argv for argv in seen)  # servers get no peer
    assert {argv[6] for argv in seen} == {"18600", "18601"}


# --------------------------------------------------------------- CLI


def test_main_prints_one_document_and_honours_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = ib_host(tmp_path, FakeFabric())
    assert fp.main(["identify", "--lldp-wait", "0", "--no-census"], host=host) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and json.loads(out)["kind"] == "fabric_identify"
    target = tmp_path / "out.json"
    assert (
        fp.main(["--out", str(target), "identify", "--lldp-wait", "0"], host=host) == 0
    )
    assert json.loads(target.read_text())["switches"]
    assert capsys.readouterr().out == ""


def test_main_reports_unreadable_input_as_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = fp.Host(root=str(tmp_path), which=lambda n: None, hostname=lambda: "h")
    bad = tmp_path / "targets.json"
    bad.write_text("{not json")
    assert fp.main(["trace", "--targets", str(bad)], host=host) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["results"] == [] and doc["notes"][0]["source"] == "input"
    TraceReport.model_validate(doc)
    missing = str(tmp_path / "nope.json")
    assert fp.main(["perftest", "--role", "server", "--rails", missing], host=host) == 0
    PerftestReport.model_validate(json.loads(capsys.readouterr().out))
    assert fp.main(["perfquery", "--ports", missing], host=host) == 0
    PerfqueryReport.model_validate(json.loads(capsys.readouterr().out))


def test_agent_is_standalone_python38() -> None:
    tree = ast.parse(AGENT.read_text(), feature_version=(3, 8))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    future = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    assert future and any(a.name == "annotations" for a in future[0].names)


def test_agent_script_runs_with_an_isolated_interpreter(tmp_path: Path) -> None:
    """No repo on sys.path, no tools on PATH: identify still prints JSON."""
    proc = subprocess.run(
        [sys.executable, "-I", str(AGENT), "identify", "--lldp-wait", "0"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": str(tmp_path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["kind"] == "fabric_identify" and doc["tools"]["smpquery"] is False
    FabricNodeReport.model_validate(doc)
