# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""adapters/fabric.py: command construction, tool routing, merging and the
fail-soft paths, against a recording cluster that plays the agent."""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from benchmarks.lium_cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.lium_cluster_bench.adapters.fabric import (
    AGENT_REL,
    FabricProber,
    extract_document,
)
from benchmarks.lium_cluster_bench.domain import fabric as F
from benchmarks.lium_cluster_bench.ports import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

NODES = ("node0", "node1", "node2", "node3")
IPS = {node: f"10.42.0.{i + 1}" for i, node in enumerate(NODES)}
WORKDIR = "/root/cluster_bench"
IMAGE = "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
REMOTE = f"{WORKDIR}/fabric/exp1"
ALL_TOOLS = {
    "smpquery": True,
    "sminfo": True,
    "ibtracert": True,
    "perfquery": True,
    "ibv_devinfo": True,
    "ib_write_bw": True,
    "ib_write_lat": True,
    "lldpctl": False,
    "ip": True,
    "mlnx_qos": False,
}
NO_FABRIC_TOOLS = {**ALL_TOOLS, **{t: False for t in ("smpquery", "sminfo", "ibtracert", "perfquery", "ib_write_bw", "ib_write_lat")}}  # fmt: skip


def agent_call(command: str) -> tuple[str, str, list[str]] | None:
    """(mode, subcommand, args) of an agent invocation, else None."""
    argv = shlex.split(command)
    at = next((i for i, a in enumerate(argv) if a.endswith("fabric_probe.py")), None)
    if at is None:
        return None
    return ("container" if "docker" in argv else "host"), argv[at + 1], argv[at + 2 :]


def flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


class FabricCluster:
    """A RecordingExecutor responder playing fabric_probe.py on 4 nodes:
    2 rails, two leaves per rail (2 nodes per leaf), complete censuses."""

    def __init__(self, rails: int = 2) -> None:
        self.raw = F.synthetic_reports(NODES, rails=rails, nodes_per_leaf=2)
        self.records = {
            guid: record
            for doc in self.raw.values()
            for guid, record in doc["switches"].items()
        }
        self.tools = {"container": dict(ALL_TOOLS), "host": dict(ALL_TOOLS)}
        self.fail: dict[tuple[str, str, str], CommandResult] = {}
        self.perftest_ok = True
        self.rec = RecordingExecutor(node_names=NODES, responder=self)

    def __call__(self, node: str, command: str) -> CommandResult | None:
        call = agent_call(command)
        if call is None:
            return None
        mode, sub, args = call
        if (node, mode, sub) in self.fail:
            return self.fail[(node, mode, sub)]
        doc = getattr(self, f"_{sub}")(node, mode, args)
        banner = "\n==========\n== CUDA ==\n==========\n"  # image entrypoint noise
        return CommandResult(node, command, 0, banner + json.dumps(doc) + "\n", "")

    def payload(self, node: str, args: list[str], name: str) -> Any:
        return json.loads(self.rec.files[(node, flag(args, name))])

    def _identify(self, node: str, mode: str, args: list[str]) -> dict[str, Any]:
        doc = copy.deepcopy(self.raw[node])
        doc["tools"] = dict(self.tools[mode])
        if not doc["tools"]["smpquery"]:
            doc["sm"] = []
            for port in doc["ports"]:
                port["neighbor"] = None
            doc["notes"] = [{"source": "smpquery", "error": "not installed"}]
        doc["switches"] = {}
        if "--census-only" in args:
            for key in flag(args, "--census-only").split(","):
                port = next(p for p in doc["ports"] if f"{p['hca']}:{p['port']}" == key)
                guid = port["neighbor"]["node_guid"]
                doc["switches"][guid] = dict(self.records[guid], via=key)
        return doc

    def _trace(self, node: str, mode: str, args: list[str]) -> dict[str, Any]:
        results = []
        for target in self.payload(node, args, "--targets"):
            group = lambda n: NODES.index(n) // 2  # noqa: E731
            count = 1 if group(target["src_node"]) == group(target["dst_node"]) else 3
            hops = [
                {"node_type": "Switch", "guid": f"0x{i:016x}"} for i in range(count)
            ]
            results.append(
                {"target": target, "method": "ibtracert", "hops": hops,
                 "hop_count": count, "reached": True}
            )  # fmt: skip
        return {"kind": "fabric_trace", "hostname": node, "results": results}

    def _perftest(self, node: str, mode: str, args: list[str]) -> dict[str, Any]:
        role = flag(args, "--role")
        results = []
        for rail in self.payload(node, args, "--rails"):
            for test in ("bw", "lat"):
                entry = {
                    "rail": rail["rail"],
                    "hca": rail["hca"],
                    "port": 1,
                    "test": test,
                }
                if role == "client" and not self.perftest_ok:
                    entry.update(rc=1, error="exit 1: Couldn't connect to x:1")
                elif role == "client" and test == "bw":
                    entry.update(
                        rc=0,
                        bw={
                            "msg_bytes": 65536,
                            "unit": "Gb/sec",
                            "bw_peak": 0.0,
                            "bw_avg": 384.0,
                            "bw_peak_gbs": 0.0,
                            "bw_avg_gbs": 48.0,
                        },
                    )
                elif role == "client":
                    entry.update(rc=0, lat={"msg_bytes": 2, "t_typical_us": 1.9})
                else:
                    entry.update(rc=0)
                results.append(entry)
        return {
            "kind": "fabric_perftest",
            "role": role,
            "peer": flag(args, "--peer") if role == "client" else None,
            "results": results,
        }  # fmt: skip

    def _perfquery(self, node: str, mode: str, args: list[str]) -> dict[str, Any]:
        ports = []
        for spec in self.payload(node, args, "--ports"):
            ports.append(
                {**spec, "extended": True, "wide": ["PortRcvData", "PortXmitData"],
                 "counters": {"PortXmitData": 1000, "PortRcvData": 2000},
                 "mono_s": 50.0, "wall_s": 1_800_000_050.0}
            )  # fmt: skip
        return {"kind": "fabric_perfquery", "hostname": node, "ports": ports}

    def calls(
        self, node: str, sub: str | None = None
    ) -> list[tuple[str, str, list[str]]]:
        found = []
        for command in self.rec.commands_for(node):
            call = agent_call(command)
            if call is not None and (sub is None or call[1] == sub):
                found.append(call)
        return found


def prober(cluster: FabricCluster, tmp_path: Path, **kwargs: Any) -> FabricProber:
    options = {
        "image": IMAGE,
        "workdir": WORKDIR,
        "results_dir": tmp_path / "fabric",
        "run_id": "exp1",
        "control_ips": IPS,
    }
    options.update(kwargs)
    return FabricProber(cluster.rec, **options)


@pytest.fixture
def fabric() -> FabricCluster:
    return FabricCluster()


# --------------------------------------------------------------- identify


def test_identify_runs_the_pinned_image_and_censuses_each_leaf_once(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    reports = prober(fabric, tmp_path).identify(list(NODES), lldp_wait_s=35)

    assert list(reports) == list(NODES)
    (first,) = [c for c in fabric.rec.commands_for("node0") if "identify" in c][:1]
    argv = shlex.split(first.split(" && ", 1)[1])
    assert first.startswith(f"mkdir -p {REMOTE} && docker run --rm ")
    assert argv[argv.index("--name") + 1] == "cb-fabric-exp1-identify-1"
    assert "--gpus" not in argv and "-d" not in argv
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--device") + 1] == "/dev/infiniband"
    assert argv[argv.index("--cap-add", argv.index("SYS_RESOURCE")) + 1] == "NET_ADMIN"
    assert argv[argv.index("--user") + 1] == "0:0"
    assert "cb.role=fabric" in argv
    assert f"{WORKDIR}/src:/bench/src:ro" in argv and f"{REMOTE}:{REMOTE}:ro" in argv
    assert argv[argv.index(IMAGE) + 1 :] == [
        "python3", f"/bench/src/{AGENT_REL}", "identify", "--no-census",
        "--lldp-wait", "35",
    ]  # fmt: skip

    assert all(
        c[0] == "container" for n in NODES for c in fabric.calls(n)
    )  # no host pass
    census = {n: fabric.calls(n, "identify")[1:] for n in NODES}
    keys = [flag(c[2], "--census-only") for calls in census.values() for c in calls]
    assert sorted(len(k.split(",")) for k in keys) == [1, 1, 1, 1]  # 4 leaves, 4 nodes
    for calls in census.values():
        for _, _, args in calls:
            assert (
                flag(args, "--max-smp") == "1000" and flag(args, "--lldp-wait") == "0"
            )
    guids = {
        guid
        for report in reports.values()
        for guid, record in report.switches.items()
        if record.census == "complete"
    }
    assert len(guids) == 4
    assert {r.runner for r in reports.values()} == {"container"}
    assert (tmp_path / "fabric" / "identify-node0-container.json").exists()
    assert any((tmp_path / "fabric").glob("census-*-container.json"))
    topo = F.build_topology(reports)
    assert topo.rail_optimized is True and topo.nodes_per_leaf == 2
    assert all(s.census == "complete" for s in topo.by_role("leaf"))


def test_identify_uses_host_tools_when_the_image_lacks_them(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    fabric.tools["container"] = dict(NO_FABRIC_TOOLS)
    reports = prober(fabric, tmp_path).identify(list(NODES), lldp_wait_s=10)

    host_calls = [
        c
        for c in fabric.rec.commands_for("node1")
        if "identify" in c and "docker" not in c
    ]
    assert host_calls[0] == shlex.join(
        [f"{WORKDIR}/venv/bin/python", f"{WORKDIR}/src/{AGENT_REL}", "identify",
         "--no-census", "--lldp-wait", "0"]
    )  # fmt: skip
    report = reports["node1"]
    assert report.runner == "container+host"
    assert all(p.neighbor is not None for p in report.ports)  # richer host ports win
    assert report.tools["smpquery"] is True
    assert ("smpquery", "not installed") in {(n.source, n.error) for n in report.notes}
    census_modes = {c[0] for n in NODES for c in fabric.calls(n, "identify")[2:]}
    assert census_modes == {"host"}
    assert F.build_topology(reports).rail_optimized is True


def test_identify_failures_become_placeholders_and_notes(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    fabric.fail[("node2", "container", "identify")] = CommandResult(
        "node2", "x", 124, "", "partial\n[timeout]"
    )
    fabric.fail[("node2", "host", "identify")] = CommandResult(
        "node2", "x", 127, "", "bash: /root/cluster_bench/venv/bin/python: No such file"
    )
    fabric.fail[("node3", "container", "identify")] = CommandResult(
        "node3", "x", 0, "not json at all\n", ""
    )
    reports = prober(fabric, tmp_path).identify(list(NODES))
    failed = reports["node2"]
    assert failed.ports == () and failed.runner is None
    errors = [n.error for n in failed.notes]
    assert errors[0].startswith("identify on node2 (container) exit 124: [timeout]")
    assert "exit 127" in errors[0]  # host attempt joined in the same note
    assert "docker rm -f cb-fabric-exp1-identify-" in "\n".join(
        fabric.rec.commands_for("node2")
    )
    assert reports["node3"].runner == "host"  # unparseable container output
    assert any("no JSON document" in n.error for n in reports["node3"].notes)
    topo = F.build_topology(reports)
    assert "node2: no fabric data (agent failed or no RDMA devices)" in topo.coverage


def test_extract_document_skips_banners() -> None:
    assert extract_document('banner\n{"a": 1}\n') == {"a": 1}
    assert extract_document('{"a": 1}\n{"b": 2}\n') == {"b": 2}
    with pytest.raises(ValueError, match="no JSON"):
        extract_document("[1, 2]\n{broken\n")


# --------------------------------------------------------------- trace


def test_trace_batches_targets_per_lower_node(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    topo = F.build_topology(reports)
    traces = probe.trace(topo, reports)

    assert len(traces) == 2 * 6 and all(t.reached for t in traces)
    assert fabric.calls("node3", "trace") == []  # never the lower node of a pair
    (call,) = fabric.calls("node0", "trace")
    sent = fabric.payload("node0", call[2], "--targets")
    assert flag(call[2], "--targets").startswith(f"{REMOTE}/trace-node0-")
    assert {(t["dst_node"], t["rail"]) for t in sent} == {
        (n, r) for n in NODES[1:] for r in (0, 1)
    }
    assert sent[0]["src_lid"] == topo.port("node0", 0).lid
    rebuilt = F.build_topology(reports, traces)
    assert rebuilt.hop_method == {0: "ibtracert", 1: "ibtracert"}
    assert rebuilt.hops[1]["node0"]["node3"] == 3 and rebuilt.tiers == 2


def test_trace_routes_to_host_tools_and_fails_soft(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    fabric.tools["container"]["ibtracert"] = False
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    fabric.fail[("node1", "host", "trace")] = CommandResult("node1", "x", 1, "", "boom")
    traces = probe.trace(F.build_topology(reports), reports)
    assert fabric.calls("node0", "trace")[0][0] == "host"
    broken = [t for t in traces if t.target.src_node == "node1"]
    assert broken and all(t.method == "none" for t in broken)
    assert broken[0].error == "trace on node1 (host) exit 1: boom"
    assert sum(t.reached for t in traces) == len(traces) - len(broken)


def test_trace_with_a_fresh_prober_learns_tools_from_reports(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    fabric.tools["container"]["ibtracert"] = False
    reports = prober(fabric, tmp_path).identify(list(NODES))
    assert reports["node0"].runner_tools["container"]["ibtracert"] is False
    assert reports["node0"].runner_tools["host"]["ibtracert"] is True
    assert reports["node0"].tools["ibtracert"] is True  # OR of both modes
    fresh = prober(fabric, tmp_path)  # e.g. after a resume
    traces = fresh.trace(F.build_topology(reports), reports)
    assert len(traces) == 12 and all(t.method == "ibtracert" for t in traces)
    assert {c[0] for c in fabric.calls("node0", "trace")} == {"host"}


# --------------------------------------------------------------- link matrix


def test_link_matrix_rounds_roles_ports_and_results(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    topo = F.build_topology(reports)
    matrix = probe.link_matrix(reports, topology=topo, bw_seconds=2, lat_iters=500)

    assert matrix.rounds == 6 and matrix.nodes == NODES and matrix.rails == (0, 1)
    assert len(matrix.tests) == 6 * 2 * 2 and not matrix.skipped
    assert {(t.src, t.dst) for t in matrix.tests} == {
        (a, b) for a in NODES for b in NODES if a != b
    }
    test = matrix.get("node3", "node1", 1)
    assert test.bw_gbs == 48.0 and test.lat_typical_us == 1.9 and test.error is None
    assert test.src_hca == "mlx5_1" and test.dst_hca == "mlx5_1"
    for node in NODES:
        roles = [flag(c[2], "--role") for c in fabric.calls(node, "perftest")]
        assert len(roles) == 6 and roles.count("server") == 3  # one role per round
    client = next(c for c in fabric.calls("node0", "perftest") if "client" in c[2])
    args = client[2]
    assert flag(args, "--peer") in {IPS[n] for n in NODES[1:]}
    assert flag(args, "--bw-seconds") == "2" and flag(args, "--lat-iters") == "500"
    assert flag(args, "--timeout") == "60"
    rails = fabric.payload("node0", args, "--rails")
    assert [(r["rail"], r["hca"], r["gid_index"]) for r in rails] == [
        (0, "mlx5_0", None),
        (1, "mlx5_1", None),
    ]
    assert (
        len({r["bw_tcp_port"] for r in rails} | {r["lat_tcp_port"] for r in rails}) == 4
    )
    assert all(18600 <= r["bw_tcp_port"] < 18728 for r in rails)
    assert (tmp_path / "fabric" / "perftest-00-client-node0.json").exists()
    findings = F.analyze_link_matrix(
        matrix, F.nominal_gbs_by_rail(topo), hops=topo.hops
    )
    assert findings == []


def test_link_matrix_one_direction_and_rail_subset(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    matrix = probe.link_matrix(reports, rails=[1], bidirectional=False)
    assert matrix.rounds == 3 and matrix.rails == (1,) and len(matrix.tests) == 6
    assert {(t.src, t.dst) for t in matrix.tests} == {
        (a, b) for i, a in enumerate(NODES) for b in NODES[i + 1 :]
    }


def test_link_matrix_abandons_a_dead_fabric(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    fabric.perftest_ok = False
    matrix = probe.link_matrix(reports)
    assert matrix.rounds == 1 and len(matrix.tests) == 4
    assert all(t.error and t.bw_gbs is None for t in matrix.tests)
    assert matrix.notes == ("every test of round 1 failed; link matrix abandoned",)


def test_link_matrix_budget_skips_and_client_failures(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    ticks = iter([0.0, 0.0, 100.0, 500.0, 500.0])
    probe = prober(fabric, tmp_path, clock=lambda: next(ticks))
    reports = probe.identify(list(NODES))
    fabric.fail[("node1", "container", "perftest")] = CommandResult(
        "node1", "x", 1, "", "oops"
    )
    matrix = probe.link_matrix(reports, budget_s=360)
    assert matrix.rounds == 2
    assert matrix.notes == ("budget 360s exhausted after 2 of 6 rounds",)
    failed = [t for t in matrix.tests if t.error]
    assert failed and all("oops" in t.error for t in failed)

    fabric.tools["container"]["ib_write_bw"] = False
    fabric.tools["host"]["ib_write_bw"] = False
    bare = prober(fabric, tmp_path)
    reports = bare.identify(list(NODES))
    skipped = bare.link_matrix(reports)
    assert skipped.skipped and skipped.tests == ()
    assert skipped.notes[0].startswith("perftest (ib_write_bw) missing")
    no_ips = prober(fabric, tmp_path, control_ips={})
    no_ips.identify(list(NODES))
    assert no_ips.link_matrix(reports).skipped


# --------------------------------------------------------------- snapshots


def test_snapshot_switch_ports_runs_once_on_node0(
    fabric: FabricCluster, tmp_path: Path
) -> None:
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    refs = F.switch_port_refs(F.build_topology(reports))
    snapshot = probe.snapshot_switch_ports(refs, label="p1 before")

    (call,) = fabric.calls("node0", "perfquery")
    assert all(fabric.calls(n, "perfquery") == [] for n in NODES[1:])
    assert fabric.payload("node0", call[2], "--ports") == F.perfquery_ports(refs)
    assert snapshot.node == "node0" and snapshot.label == "p1 before"
    assert len(snapshot.samples) == len(refs)
    assert all(s.counters["PortXmitData"] == 1000 for s in snapshot.samples)
    assert (tmp_path / "fabric" / "perfquery-p1-before-node0.json").exists()
    later = snapshot.model_copy(
        update={
            "samples": tuple(
                s.model_copy(update={"mono_s": 60.0, "counters": {"PortXmitData": 1000 + 25 * 10**8, "PortRcvData": 2000}})
                for s in snapshot.samples
            )
        }
    )  # fmt: skip
    load = F.switch_load(snapshot, later)[0]
    assert load.tx_gbs == pytest.approx(1.0) and load.rx_gbs == 0.0


def test_snapshot_fails_soft(fabric: FabricCluster, tmp_path: Path) -> None:
    fabric.tools["container"]["perfquery"] = False
    probe = prober(fabric, tmp_path)
    reports = probe.identify(list(NODES))
    refs = F.switch_port_refs(F.build_topology(reports))
    assert probe.snapshot_switch_ports(refs).samples[0].counters  # host mode
    assert fabric.calls("node0", "perfquery")[-1][0] == "host"
    fabric.fail[("node0", "host", "perfquery")] = CommandResult(
        "node0", "x", 124, "", "[timeout]"
    )
    broken = probe.snapshot_switch_ports(refs)
    assert all(s.error == "not sampled" for s in broken.samples)
    assert broken.notes == ("perfquery on node0 (host) exit 124: [timeout]",)
    empty = probe.snapshot_switch_ports([])
    assert empty.samples == () and empty.notes == ("no switch ports to sample",)
    fabric.tools["host"]["perfquery"] = False
    blind = prober(fabric, tmp_path)
    blind.identify(list(NODES))
    assert blind.snapshot_switch_ports(refs).notes == (
        "perfquery unavailable on node0",
    )
