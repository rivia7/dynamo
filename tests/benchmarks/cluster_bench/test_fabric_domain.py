# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""domain/fabric.py: topology derivation, link matrix, switch load, KB views."""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any

import pytest

from benchmarks.cluster_bench.domain import fabric as F

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

NODES8 = tuple(f"node{i}" for i in range(8))
NODES4 = NODES8[:4]


def reports_of(raw: dict[str, dict[str, Any]]) -> dict[str, F.FabricNodeReport]:
    return {node: F.FabricNodeReport.model_validate(doc) for node, doc in raw.items()}


def synthetic(nodes=NODES8, **kwargs: Any) -> dict[str, F.FabricNodeReport]:
    return reports_of(F.synthetic_reports(nodes, **kwargs))


def leaf_id(topo: F.FabricTopology, node: str, rail: int) -> str:
    return topo.leaf_of[node][f"mlx5_{rail}"].switch_id


def sysfs_only(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What a node without umad reports: tier S only."""
    out = copy.deepcopy(raw)
    for doc in out.values():
        doc["umad"] = False
        doc["sm"] = []
        doc["switches"] = {}
        doc["smp"]["used"] = 0
        doc["notes"] = [{"source": "umad", "error": "no /dev/infiniband/umad* device"}]
        for port in doc["ports"]:
            port["neighbor"] = None
    return out


# --------------------------------------------------------------- topology


def test_rail_optimized_eight_nodes_four_per_leaf() -> None:
    topo = F.build_topology(synthetic())
    assert topo.kind == "infiniband" and topo.nodes == NODES8
    assert topo.rail_optimized is True and topo.nodes_per_leaf == 4
    assert topo.rails_isolated is True
    assert len(topo.rails) == 8 and {r.rate_gbps for r in topo.rails} == {400.0}
    assert len(topo.by_role("leaf")) == 16 and len(topo.by_role("spine")) == 16
    for rail in range(8):
        assert topo.leaf_groups[rail] == (NODES8[:4], NODES8[4:])
        assert len(topo.rails[rail].leaves) == 2
        assert len(topo.rails[rail].sm_guids) == 1  # one SM per rail subnet
    attach = topo.leaf_of["node1"]["mlx5_3"]
    assert attach.port == 2 and attach.rail == 3 and attach.tier == "M"
    leaf = topo.switches[attach.switch_id]
    assert leaf.role == "leaf" and leaf.rails == (3,)
    assert leaf.our_nodes == NODES8[:4] and leaf.census == "complete"
    assert leaf.host_ports_active == 32 and leaf.uplinks_active == 32
    assert leaf.other_hosts == 28 and leaf.oversubscription == pytest.approx(1.0)
    assert len(leaf.peers) == 2 and all(
        topo.switches[s].role == "spine" for s in leaf.peers
    )
    # Inferred from leaf identity + shared spines: 1 hop in a leaf, 3 across.
    assert topo.hops[0]["node0"]["node1"] == 1
    assert topo.hops[0]["node0"]["node4"] == 3 == topo.hops[0]["node4"]["node0"]
    assert topo.hop_method == {r: "inferred" for r in range(8)}
    assert topo.max_hops == 3 and topo.tiers == 2
    assert topo.evidence["leaf_of"].tier == "M"
    assert topo.evidence["hops"].tier == "D"
    assert topo.evidence["subnets"].tier == "M"
    assert topo.degraded == () and topo.coverage == ()


def test_non_rail_optimized_two_leaves() -> None:
    topo = F.build_topology(synthetic(rail_optimized=False))
    assert topo.rail_optimized is False and topo.nodes_per_leaf == 4
    assert topo.rails_isolated is False  # one SM serves every rail
    leaves = topo.by_role("leaf")
    assert len(leaves) == 2 and all(s.rails == tuple(range(8)) for s in leaves)
    assert {s.our_ports for s in leaves} == {32}
    assert leaf_id(topo, "node0", 0) == leaf_id(topo, "node3", 7)
    assert leaf_id(topo, "node0", 0) != leaf_id(topo, "node4", 0)
    assert topo.hops[5]["node2"]["node3"] == 1 and topo.hops[5]["node2"]["node6"] == 3
    findings = F.topology_findings(topo)
    assert any(
        f.startswith("not rail-optimized: a leaf carries up to 8 rails")
        for f in findings
    )
    assert "rails share one subnet" in findings


def test_mixed_infiniband_and_roce_rails() -> None:
    topo = F.build_topology(synthetic(NODES4, kind="mixed", rails=4))
    assert topo.kind == "mixed"
    assert [r.link_layer for r in topo.rails] == [
        "InfiniBand",
        "InfiniBand",
        "Ethernet",
        "Ethernet",
    ]
    assert topo.leaf_of["node0"]["mlx5_0"].tier == "M"
    roce = topo.leaf_of["node0"]["mlx5_2"]
    assert roce.tier == "L" and roce.port_name == "swp1"
    assert topo.switches[roce.switch_id].desc == "leaf-r2g0"
    assert topo.rails[2].subnets == ("10.102.0.0/24",)
    assert len(topo.lldp) == 8
    item = topo.lldp[0]
    assert item.pfc_enabled == (3,) and item.roce_app_priority == 3
    assert item.max_frame_size == 9216 and item.ets_tc_bw[3] == 50
    assert topo.evidence["leaf_of"].source.endswith("LLDP chassis/port id")
    assert "switch DCBX advertises PFC on priority 3" in F.topology_findings(topo)
    targets = F.trace_targets(topo)
    ib = [t for t in targets["node0"] if t.rail == 0]
    eth = [t for t in targets["node0"] if t.rail == 2]
    assert ib[0].src_lid and ib[0].src_ip is None
    assert eth[0].src_ip == "10.102.0.1" and eth[0].dst_ip == "10.102.0.2"
    assert eth[0].netdev == "enp56s0np0" and eth[0].src_lid is None


def test_sysfs_only_ib_has_rates_and_subnets_but_no_switches() -> None:
    raw = sysfs_only(F.synthetic_reports(NODES8))
    topo = F.build_topology(reports_of(raw))
    assert topo.kind == "infiniband" and {r.rate_gbps for r in topo.rails} == {400.0}
    assert topo.switches == {} and topo.leaf_of == {}
    assert topo.rail_optimized is None and topo.nodes_per_leaf is None
    assert topo.hops == {} and topo.tiers is None and topo.max_hops is None
    # Same default prefix and SM LID everywhere, but one LID on two rails can
    # only happen in separate subnets.
    assert topo.rails_isolated is True
    assert topo.evidence["subnets"].tier == "S"
    assert "leaf_of" not in topo.evidence
    assert "node3: MAD tier unavailable (no umad device)" in topo.coverage
    assert "leaf identity unknown for 64 of 64 rail ports" in topo.coverage
    assert any(c.startswith("switch hops unknown for 224") for c in topo.coverage)
    assert "leaf switches unknown (no MAD access and no LLDP)" in F.topology_findings(
        topo
    )
    assert F.switch_port_refs(topo) == []


def test_leaf_lid_comes_from_the_neighbor_without_a_census_record() -> None:
    raw = F.synthetic_reports(NODES4, rails=1)
    for doc in raw.values():
        doc["switches"] = {}
    topo = F.build_topology(reports_of(raw))
    (leaf,) = topo.by_role("leaf")
    assert leaf.census == "none" and leaf.lid == 2
    refs = F.switch_port_refs(topo)
    assert [(r.role, r.port, r.switch_lid) for r in refs] == [
        ("downlink", port, 2) for port in (1, 2, 3, 4)
    ]


def test_rail_of_restricts_and_renumbers_rails() -> None:
    reports = synthetic(NODES4, rails=4)
    rail_of = {node: {"mlx5_3": 0, "mlx5_1": 1} for node in NODES4}
    rail_of["node2"] = {"mlx5_3": 0, "mlx5_9": 1}
    topo = F.build_topology(reports, rail_of=rail_of)
    assert topo.rail_indices == (0, 1)
    assert topo.port("node0", 0).hca == "mlx5_3"
    assert topo.port("node2", 1) is None
    assert "node2: rail 1 HCA mlx5_9 missing from report" in topo.coverage
    assert F.default_rail_map(reports)["node0"] == {f"mlx5_{i}": i for i in range(4)}


def test_degraded_links() -> None:
    raw = F.synthetic_reports(NODES4, rails=4)
    node1 = raw["node1"]["ports"]
    node1[1]["rate_gbps"] = 200.0  # trained down
    node1[1]["rate"] = "200 Gb/sec (2X NDR)"
    node1[1]["neighbor"]["link"]["width_active"] = "2X"
    node1[2]["pcie"]["current_width"] = 8
    node1[3]["state"] = "1: DOWN"
    leaf = next(iter(raw["node0"]["switches"].values()))
    uplink = next(p for p in leaf["ports"] if p["peer"]["node_type"] == "Switch")
    uplink["rate_gbps"] = 100.0
    topo = F.build_topology(reports_of(raw))
    kinds = {(d.kind, d.node, d.hca) for d in topo.degraded}
    assert ("rate_below_peers", "node1", "mlx5_1") in kinds
    assert ("width_below_supported", "node1", "mlx5_1") in kinds
    assert ("pcie_degraded", "node1", "mlx5_2") in kinds
    assert ("port_down", "node1", "mlx5_3") in kinds
    low = next(d for d in topo.degraded if d.kind == "rate_below_peers")
    assert (low.value, low.reference, low.tier) == (200.0, 400.0, "S")
    assert (
        low.detail == "node1/mlx5_1 runs at 200 Gb/s while rail 1 peers run at 400 Gb/s"
    )
    pcie = next(d for d in topo.degraded if d.kind == "pcie_degraded")
    assert pcie.detail == "node1/mlx5_2 PCIe at 32 GT/s x8 of 32 GT/s x16"
    up = next(d for d in topo.degraded if d.kind == "uplink_below_peers")
    assert up.switch_port == uplink["port"] and up.value == 100.0
    oversub = topo.switches[leaf["guid"]].oversubscription
    assert oversub == pytest.approx(32 * 400 / (31 * 400 + 100))
    assert F.topology_signature(topo)["degraded_links"] == len(topo.degraded)


def test_partial_census_and_missing_node_are_coverage_notes() -> None:
    raw = F.synthetic_reports(NODES4, rails=2)
    for doc in raw.values():
        for record in doc["switches"].values():
            record["census"] = "partial"
            record["ports"] = record["ports"][:20]
    reports = reports_of(raw)
    reports["node3"] = F.FabricNodeReport(
        notes=(F.FabricNote(source="adapter", error="identify on node3: exit 125"),)
    )
    topo = F.build_topology(reports)
    assert "census: 0 of 2 leaves complete, 2 partial" in topo.coverage
    assert "node3: no fabric data (agent failed or no RDMA devices)" in topo.coverage
    assert "node3: no active RDMA port in the fabric report" in topo.coverage
    leaf = topo.by_role("leaf")[0]
    assert leaf.uplinks_active == 0 and leaf.oversubscription is None


def test_traces_override_inference_and_reveal_spines() -> None:
    raw = F.synthetic_reports(NODES4, rails=1, nodes_per_leaf=2)
    for doc in raw.values():
        doc["switches"] = {}  # no census: spines only visible in traces
    reports = reports_of(raw)
    base = F.build_topology(reports)
    assert base.hops[0]["node0"]["node1"] == 1 and "node2" not in base.hops[0]["node0"]
    assert base.tiers is None  # cross-leaf pairs unknown
    leaf_a, leaf_b = leaf_id(base, "node0", 0), leaf_id(base, "node2", 0)
    spine = "0xfc6a1c0300e9c000"
    traces = []
    for src, dst in itertools.combinations(NODES4, 2):
        same = leaf_id(base, src, 0) == leaf_id(base, dst, 0)
        guids = [leaf_id(base, src, 0)] if same else [leaf_a, spine, leaf_b]
        traces.append(
            F.TraceResult(
                target=F.TraceTarget(
                    rail=0, hca="mlx5_0", port=1, src_node=src, dst_node=dst
                ),
                method="ibtracert",
                hops=tuple(F.TraceHop(guid=g, node_type="Switch") for g in guids),
                hop_count=len(guids),
                reached=True,
            )
        )
    traces.append(
        F.TraceResult(target=F.TraceTarget(rail=0, hca="mlx5_0", port=1), method="none")
    )
    topo = F.build_topology(reports, traces)
    assert topo.hops[0]["node1"]["node3"] == 3 and topo.hop_method[0] == "ibtracert"
    assert topo.tiers == 2 and topo.evidence["hops"].tier == "M"
    assert topo.switches[spine].role == "spine" and topo.switches[spine].tier == "M"
    assert "traces: 1 result(s) without node names ignored" in topo.coverage


def test_udp_traces_count_routers_and_ignore_l2() -> None:
    reports = synthetic(NODES4, kind="roce", rails=1, nodes_per_leaf=2)
    target = F.TraceTarget(
        rail=0, hca="mlx5_0", port=1, src_node="node0", dst_node="node2"
    )
    routed = F.TraceResult(target=target, method="udp", hop_count=3, reached=True)
    l2 = F.TraceResult(
        target=target.model_copy(update={"dst_node": "node1"}),
        method="udp",
        hop_count=0,
        reached=True,
    )
    lost = F.TraceResult(
        target=target.model_copy(update={"dst_node": "node3"}),
        method="udp",
        error="destination silent up to ttl 8",
    )
    topo = F.build_topology(reports, [routed, l2, lost])
    assert topo.hops[0]["node0"]["node2"] == 3  # routers
    assert topo.hops[0]["node0"]["node1"] == 1  # L2: inferred from LLDP leaf
    assert topo.hop_method[0] == "mixed"
    assert "traces: 1 of 3 path(s) not traced" in topo.coverage
    assert topo.evidence["hops"].tier == "T"


def test_trace_targets_one_direction_per_pair_per_rail() -> None:
    topo = F.build_topology(synthetic(NODES4, rails=2))
    targets = F.trace_targets(topo)
    assert sorted(targets) == ["node0", "node1", "node2"]  # the last node never sources
    flat = [t for batch in targets.values() for t in batch]
    assert len(flat) == 2 * 6
    assert {(t.src_node, t.dst_node) for t in flat if t.rail == 0} == set(
        itertools.combinations(NODES4, 2)
    )
    first = targets["node0"][0]
    assert (first.src_lid, first.dst_lid) == (100, 101)
    assert first.hca == "mlx5_0" and first.src_ip is None


# --------------------------------------------------------------- rounds


@pytest.mark.parametrize(("count", "rounds"), [(2, 1), (3, 3), (4, 3), (8, 7)])
def test_round_robin_rounds(count: int, rounds: int) -> None:
    nodes = [f"node{i}" for i in range(count)]
    schedule = F.round_robin_rounds(nodes)
    assert len(schedule) == rounds
    pairs = [pair for rnd in schedule for pair in rnd]
    assert sorted(pairs) == sorted(itertools.combinations(nodes, 2))  # each once
    for rnd in schedule:
        members = [n for pair in rnd for n in pair]
        assert len(members) == len(set(members))  # disjoint pairs
        assert len(rnd) == count // 2
        assert all(nodes.index(a) < nodes.index(b) for a, b in rnd)


def test_round_robin_edge_cases() -> None:
    assert F.round_robin_rounds(["node0"]) == []
    with pytest.raises(ValueError, match="duplicate"):
        F.round_robin_rounds(["a", "a"])


# --------------------------------------------------------------- link matrix


def link(
    src: str, dst: str, rail: int, bw: float | None, lat: float | None, **kw: Any
) -> F.LinkTest:
    return F.LinkTest(src=src, dst=dst, rail=rail, bw_gbs=bw, lat_typical_us=lat, **kw)


def test_analyze_link_matrix_findings() -> None:
    tests = [
        link("node0", "node1", 0, 47.5, 1.80),
        link("node1", "node0", 0, 47.8, 1.80),
        link("node0", "node2", 0, 47.0, 2.36),
        link("node2", "node0", 0, 30.0, 2.38),  # weak and asymmetric
        link("node1", "node2", 0, 47.2, 2.35),
        link("node2", "node1", 0, 47.4, 2.36),
        link("node0", "node1", 1, 33.0, 1.80),
        link("node1", "node0", 1, 33.5, 1.81),
        link("node0", "node2", 1, None, None, error="bw: exit 1: Couldn't connect"),
    ]
    matrix = F.LinkMatrix(nodes=NODES4[:3], rails=(0, 1), tests=tuple(tests))
    hops = {0: {"node0": {"node1": 1, "node2": 3}, "node1": {"node2": 3}}}
    findings = F.analyze_link_matrix(matrix, {0: 50.0, 1: 50.0}, hops=hops)
    kinds = [(f.kind, f.rail, f.src, f.dst) for f in findings]
    assert ("failed", 1, "node0", "node2") in kinds
    assert ("weak_link", 0, "node2", "node0") in kinds
    assert ("asymmetry", 0, "node0", "node2") in kinds
    assert ("weak_rail", 1, None, None) in kinds
    assert ("weak_link", 1, "node0", "node1") in kinds  # below 80 % of line rate
    assert sum(1 for k in kinds if k[0] == "weak_link" and k[1] == 0) == 1
    groups = next(f for f in findings if f.kind == "latency_groups")
    assert groups.rail == 0
    assert groups.groups == (
        ("node0->node1", "node1->node0"),
        ("node1->node2", "node0->node2", "node2->node1", "node2->node0"),
    )
    assert groups.detail == (
        "rail 0 latency groups: 2 pair(s) ~1.80 us (hops 1); "
        "4 pair(s) ~2.36 us (hops 3)"
    )
    weak = next(f for f in findings if f.kind == "weak_link" and f.rail == 0)
    assert weak.value == 30.0 and weak.reference == 50.0  # below 80 % of both
    assert weak.detail == (
        "rail 0 node2->node0 30.0 GB/s (rail median 47.3, line rate 50.0 GB/s)"
    )


def test_analyze_uniform_matrix_has_no_findings() -> None:
    matrix = F.synthetic_link_matrix(NODES4, rails=2, nodes_per_leaf=4)
    assert F.analyze_link_matrix(matrix, {0: 50.0, 1: 50.0}) == []
    split = F.synthetic_link_matrix(NODES8, rails=1)
    (finding,) = F.analyze_link_matrix(split, {0: 50.0})
    assert finding.kind == "latency_groups" and len(finding.groups) == 2
    assert len(finding.groups[0]) == 24 and len(finding.groups[1]) == 32


def perftest_report(role: str, results: list[dict[str, Any]]) -> F.PerftestReport:
    return F.PerftestReport.model_validate(
        {"kind": "fabric_perftest", "role": role, "results": results}
    )


def test_link_tests_from_perftest_merges_bw_and_lat() -> None:
    client = perftest_report(
        "client",
        [
            {"rail": 0, "hca": "mlx5_0", "port": 1, "test": "bw", "rc": 0,
             "bw": {"msg_bytes": 65536, "unit": "Gb/sec", "bw_peak": 0.0,
                    "bw_avg": 381.85, "msg_rate_mpps": 0.73,
                    "bw_peak_gbs": 0.0, "bw_avg_gbs": 47.73}},
            {"rail": 0, "hca": "mlx5_0", "port": 1, "test": "lat", "rc": 0,
             "lat": {"msg_bytes": 2, "t_typical_us": 1.75, "t_avg_us": 1.76,
                     "p99_us": 1.89}},
            {"rail": 1, "hca": "mlx5_1", "port": 1, "test": "bw", "rc": 1,
             "error": "exit 1: Couldn't connect to 10.42.0.2:18601"},
        ],
    )  # fmt: skip
    server = perftest_report(
        "server",
        [{"rail": 1, "hca": "mlx5_1", "port": 1, "test": "bw", "rc": 124,
          "error": "exit 124: timed out after 60.0s"}],
    )  # fmt: skip
    tests = F.link_tests_from_perftest(
        client,
        src="node0",
        dst="node1",
        round_index=2,
        dst_hcas={0: "mlx5_4"},
        server=server,
    )
    good, bad = tests
    assert (good.rail, good.src_hca, good.dst_hca, good.round_index) == (
        0,
        "mlx5_0",
        "mlx5_4",
        2,
    )
    assert (
        good.bw_gbs == 47.73 and good.lat_typical_us == 1.75 and good.lat_p99_us == 1.89
    )
    assert good.error is None and good.bw_msg_bytes == 65536
    assert bad.bw_gbs is None and bad.error == (
        "bw: exit 1: Couldn't connect to 10.42.0.2:18601; "
        "server: exit 124: timed out after 60.0s"
    )


# --------------------------------------------------------------- switch load


def ref(
    port: int = 17, rate: float | None = 400.0, role: str = "downlink"
) -> F.SwitchPortRef:
    return F.SwitchPortRef(
        switch_guid="0xfc6a1c0300e7a1b0",
        switch_lid=3,
        port=port,
        role=role,
        rate_gbps=rate,
        via_node="node0",
        via_hca="mlx5_0",
        via_port=1,
    )


def sample(
    port_ref: F.SwitchPortRef, t: float, wide: bool = True, **counters: int
) -> F.PortCounterSample:
    return F.PortCounterSample(
        ref=port_ref,
        counters=counters,
        wide=tuple(sorted(counters)) if wide else (),
        extended=wide,
        mono_s=t,
        wall_s=1_800_000_000.0 + t,
    )


def snap(*samples: F.PortCounterSample, node: str = "node0") -> F.SwitchSnapshot:
    return F.SwitchSnapshot(node=node, samples=samples)


def test_switch_load_rates_utilization_and_stall_estimate() -> None:
    r = ref()
    words = int(25e9 / 4 * 10)  # 25 GB/s for 10 s
    before = snap(
        sample(
            r,
            100.0,
            PortXmitData=10**12,
            PortRcvData=5 * 10**11,
            PortXmitWait=1000,
            PortXmitDiscards=3,
            SymbolErrorCounter=0,
        )
    )
    after = snap(sample(r, 110.0, PortXmitData=10**12 + words,
                        PortRcvData=5 * 10**11 + words // 2,
                        PortXmitWait=1000 + 2 * 10**9, PortXmitDiscards=10,
                        SymbolErrorCounter=0))  # fmt: skip
    (load,) = F.switch_load(before, after)
    assert load.interval_s == 10.0
    assert load.tx_gbs == pytest.approx(25.0) and load.rx_gbs == pytest.approx(12.5)
    assert load.tx_util == pytest.approx(0.5) and load.rx_util == pytest.approx(0.25)
    assert load.xmit_wait_delta == 2 * 10**9
    assert load.xmit_wait_per_s == pytest.approx(2e8)
    assert load.stall_fraction_est == pytest.approx(0.2)  # 1 ns tick (UNVERIFIED)
    assert load.wait_ratio == pytest.approx(2e9 / (2e9 + words))
    assert load.error_deltas == {"PortXmitDiscards": 7}
    assert not (load.reset or load.saturated or load.wrapped)
    slower = F.switch_load(before, after, xmit_wait_tick_ns=0.25)[0]
    assert slower.stall_fraction_est == pytest.approx(0.05)


def test_switch_load_32bit_wrap_saturation_and_64bit_reset() -> None:
    r = ref(rate=400.0)
    wrap_before = snap(
        sample(r, 0.0, wide=False, PortXmitData=2**32 - 1000, PortRcvData=10)
    )
    wrap_after = snap(sample(r, 1.0, wide=False, PortXmitData=500, PortRcvData=20))
    (load,) = F.switch_load(wrap_before, wrap_after)
    assert load.wrapped and load.tx_gbs == pytest.approx(1500 * 4 / 1e9)
    sat_after = snap(
        sample(r, 1.0, wide=False, PortXmitData=2**32 - 1, PortRcvData=20)
    )
    (sat,) = F.switch_load(wrap_before, sat_after)
    assert sat.saturated and sat.tx_gbs is None and sat.rx_gbs is not None
    reset_before = snap(sample(r, 0.0, PortXmitData=10**15, PortRcvData=10**15))
    reset_after = snap(sample(r, 1.0, PortXmitData=10**6, PortRcvData=10**15 + 4))
    (reset,) = F.switch_load(reset_before, reset_after)
    assert reset.reset and reset.tx_gbs is None and reset.rx_gbs == pytest.approx(16e-9)
    # A "wrap" implying more than line rate is a reset of a 32-bit counter.
    # (4 * 2**32 bytes in 0.1 s would exceed 400 Gb/s.)
    big = snap(sample(r, 0.1, wide=False, PortXmitData=2**31, PortRcvData=20))
    huge_before = snap(
        sample(r, 0.0, wide=False, PortXmitData=2**31 + 5, PortRcvData=10)
    )
    assert F.switch_load(huge_before, big)[0].reset is True


def test_switch_load_error_counter_saturation_and_bad_samples() -> None:
    r = ref()
    before = snap(
        sample(
            r, 0.0, wide=False, PortXmitData=1, PortRcvData=1, SymbolErrorCounter=65530
        )
    )
    after = snap(
        sample(
            r, 2.0, wide=False, PortXmitData=2, PortRcvData=2, SymbolErrorCounter=65535
        )
    )
    (load,) = F.switch_load(before, after)
    assert load.error_deltas == {"SymbolErrorCounter": 5}
    assert "SymbolErrorCounter saturated (lower bound)" in load.notes
    other = ref(port=18)
    failed = F.PortCounterSample(ref=other, error="skipped: switch PMA failed")
    (missing, broken) = F.switch_load(
        snap(sample(r, 0.0, PortXmitData=1)), snap(sample(ref(port=19), 1.0), failed)
    )
    assert missing.notes == ("not in the first snapshot",)
    assert broken.notes == ("not in the first snapshot",)
    (same_time,) = F.switch_load(
        snap(sample(r, 5.0, PortXmitData=1)), snap(sample(r, 5.0, PortXmitData=2))
    )
    assert same_time.notes == ("no positive interval",) and same_time.tx_gbs is None
    cross = F.switch_load(
        snap(sample(r, 5.0, PortXmitData=1)),
        snap(sample(r, 99.0, PortXmitData=1), node="node1"),
    )[0]
    assert cross.interval_s == pytest.approx(94.0)  # wall clock across nodes


def test_switch_port_refs_and_snapshot_join() -> None:
    topo = F.build_topology(synthetic())
    refs = F.switch_port_refs(topo)
    assert len(refs) == 16 * (4 + 32) and len({r.key for r in refs}) == len(refs)
    assert {r.via_node for r in refs} == {"node0"}
    down = [r for r in refs if r.role == "downlink"]
    assert len(down) == 64 and {r.peer_node for r in down} == set(NODES8)
    node5 = next(r for r in down if r.peer_node == "node5" and r.rail == 6)
    assert node5.via_hca == "mlx5_6"  # node0's port on the same rail subnet
    assert node5.port == 2 and node5.switch_lid == topo.switches[node5.switch_guid].lid
    up = [r for r in refs if r.role == "uplink"]
    assert all(r.peer_guid and topo.switches[r.peer_guid].role == "spine" for r in up)
    ports = F.perfquery_ports(refs[:2])
    assert ports[0] == {
        "key": refs[0].key,
        "hca": refs[0].via_hca,
        "port": 1,
        "lid": refs[0].switch_lid,
        "switch_port": refs[0].port,
    }
    report = F.PerfqueryReport.model_validate(
        {
            "kind": "fabric_perfquery",
            "ports": [
                {"key": refs[0].key, "hca": "mlx5_0", "port": 1, "lid": refs[0].switch_lid,
                 "switch_port": refs[0].port, "extended": True, "wide": ["PortXmitData"],
                 "counters": {"PortXmitData": 7}, "mono_s": 3.0, "wall_s": 9.0},
            ],
            "notes": [{"source": "perfquery", "error": "pma_blocked", "target": "lid 9"}],
        }
    )  # fmt: skip
    snapshot = F.snapshot_from_perfquery(
        refs[:2], report, node="node0", label="p1-before"
    )
    assert snapshot.samples[0].counters == {"PortXmitData": 7}
    assert snapshot.samples[1].error == "not sampled"
    assert snapshot.taken_at_s == 9.0 and snapshot.notes == ("perfquery: pma_blocked",)


def test_synthetic_snapshot_load() -> None:
    topo = F.build_topology(synthetic(NODES4, rails=2))
    refs = F.switch_port_refs(topo)
    loads = F.switch_load(
        F.synthetic_snapshot(refs, node="node0", t_s=10.0, utilization=0.4),
        F.synthetic_snapshot(refs, node="node0", t_s=40.0, utilization=0.4),
    )
    assert len(loads) == len(refs)
    assert all(load.tx_util == pytest.approx(0.4) for load in loads)
    assert all(load.rx_util == pytest.approx(0.36) for load in loads)


# --------------------------------------------------------------- merges


def test_merge_reports_prefers_richer_ports_and_better_census() -> None:
    raw = F.synthetic_reports(NODES4, rails=2)
    full = F.FabricNodeReport.model_validate(raw["node0"])
    poor_doc = sysfs_only({"node0": raw["node0"]})["node0"]
    poor_doc["tools"]["smpquery"] = False
    poor_doc["lldp"] = {"x": {"source": "raw", "chassis_id": "aa"}}
    poor = F.FabricNodeReport.model_validate(poor_doc)
    merged = F.merge_reports(poor, full, runner="container+host")
    assert merged.runner == "container+host"
    assert all(p.neighbor is not None for p in merged.ports)
    assert merged.tools["smpquery"] is True and merged.umad is True
    assert merged.switches == full.switches and merged.sm == full.sm
    assert merged.lldp["x"].chassis_id == "aa"
    assert [n.source for n in merged.notes] == ["umad"]
    assert merged.smp.used == full.smp.used and merged.smp.blocked is False
    partial = full.switches[next(iter(full.switches))].model_copy(
        update={"census": "partial", "desc": None}
    )
    better = F.merge_switch(partial, full.switches[partial.guid])
    assert better.census == "complete"
    worse = F.merge_switch(
        partial.model_copy(update={"desc": "x"}),
        F.FabricSwitchRecord(guid=partial.guid),
    )
    assert worse.census == "partial" and worse.desc == "x"


def test_census_plan_balances_switches_over_nodes() -> None:
    raw = F.synthetic_reports(NODES8, rails=8)
    for doc in raw.values():
        doc["switches"] = {}  # identity pass only
    raw["node1"]["smp"]["blocked"] = True
    plan = F.census_plan(reports_of(raw))
    assignments = [(node, key) for node, keys in plan.items() for key in keys]
    assert len(assignments) == 16  # every leaf exactly once
    assert "node1" not in plan  # SMPs blocked there
    assert {len(keys) for keys in plan.values()} == {2, 3}
    reports = reports_of(raw)
    seen = set()
    for node, key in assignments:
        guid = reports[node].port(key).neighbor.node_guid
        assert guid not in seen
        seen.add(guid)
    done = F.synthetic_reports(NODES4, rails=2)
    assert F.census_plan(reports_of(done)) == {}  # already censused


# --------------------------------------------------------------- KB views


def test_topology_signature_is_small_stable_and_identifier_free() -> None:
    topo = F.build_topology(synthetic())
    signature = F.topology_signature(topo)
    assert signature == F.topology_signature(F.build_topology(synthetic()))
    assert signature["kind"] == "infiniband" and signature["rails"] == 8
    assert signature["rail_rate_gbps"] == 400.0 and signature["leaf_shape"] == ["4+4"]
    assert signature["rail_optimized"] is True and signature["nodes_per_leaf"] == 4
    assert signature["tiers"] == 2 and signature["max_oversubscription"] == 1.0
    assert signature["tiers_seen"] == ["D", "M", "S"]
    text = json.dumps(signature)
    assert "0x" not in text and "node0" not in text and len(text) < 600
    other = F.topology_signature(F.build_topology(synthetic(rail_optimized=False)))
    assert other["hash"] != signature["hash"]


def test_topology_findings_one_liners() -> None:
    findings = F.topology_findings(F.build_topology(synthetic()))
    assert findings[:4] == [
        "infiniband fabric: 8 rail(s) x 400 Gb/s per node (8 nodes)",
        "rail-optimized: 8 rails, 4 nodes per leaf",
        "rails are isolated subnets (8 SMs)",
        "2-tier path between our nodes: up to 3 switch hop(s) (inferred)",
    ]
    assert (
        "non-blocking leaves: host-facing capacity <= uplink capacity on 16 censused leaves"
        in findings
    )
    assert (
        "16 of 16 leaves also serve other tenants' hosts (up to 28 host ports per leaf)"
        in findings
    )
    raw = F.synthetic_reports(NODES4, kind="roce", rails=1)
    raw["node2"]["qos"] = {
        "enp24s0np0": {"source": "mlnx_qos", "trust": "pcp", "pfc_enabled": []}
    }
    roce = F.topology_findings(F.build_topology(reports_of(raw)))
    assert "PFC mismatch on node2/enp24s0np0: switch [3] vs host []" in roce
    # Four nodes on one leaf per rail: every pair is known to be one hop.
    assert "1-tier path between our nodes: up to 1 switch hop(s) (inferred)" in roce


def test_synthetic_generators_are_deterministic_and_valid() -> None:
    assert F.synthetic_reports(NODES4) == F.synthetic_reports(NODES4)
    assert F.synthetic_link_matrix(NODES4) == F.synthetic_link_matrix(NODES4)
    with pytest.raises(ValueError, match="unknown synthetic fabric kind"):
        F.synthetic_reports(NODES4, kind="omnipath")
    topo = F.build_topology(synthetic(NODES4, rails=2, nodes_per_leaf=2))
    traces = F.synthetic_traces(topo)
    assert len(traces) == 2 * 6 and all(t.reached for t in traces)
    rebuilt = F.build_topology(synthetic(NODES4, rails=2, nodes_per_leaf=2), traces)
    assert rebuilt.hop_method == {0: "ibtracert", 1: "ibtracert"} and rebuilt.tiers == 2
    matrix = F.synthetic_link_matrix(NODES4, rails=2)
    assert len(matrix.tests) == 2 * 12 and matrix.rounds == 6
    assert F.nominal_gbs_by_rail(topo) == {0: 50.0, 1: 50.0}
