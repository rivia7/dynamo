# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-2 data collection end to end (simulated 8-node cluster): fabric
topology, link matrix, switch-port load, service state, deploy lifecycle,
node facts and health, report sections and knowledge-base records."""

from __future__ import annotations

import json

import pytest

from benchmarks.lium_cluster_bench.app.store import EventStore
from benchmarks.lium_cluster_bench.cli import main
from benchmarks.lium_cluster_bench.domain.schema import EventType, ExperimentSummary

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture(scope="module")
def run8(tmp_path_factory):
    work = tmp_path_factory.mktemp("run8")
    args = ["dry-run", "--workdir", str(work), "--exp-id", "t8", "--log-level",
            "WARNING", "--fake-gpu", "b200", "--fake-nodes", "8",
            "--price-per-node-hour", "40"]  # fmt: skip
    assert main(args) == 0
    exp = work / "experiments" / "t8"
    summary = ExperimentSummary.model_validate(
        json.loads((exp / "analysis" / "summary.json").read_text())
    )
    return work, exp, summary


def test_fabric_topology_links_and_switch_load(run8):
    _, exp, sm = run8
    topo = sm.fabric
    assert topo is not None and topo.kind == "infiniband"
    assert topo.rail_optimized is True and topo.nodes_per_leaf == 4
    assert topo.max_hops == 3 and len(topo.rails) == 8
    assert sm.fabric_signature["leaves"] == 16
    # every unordered pair x every rail, both directions
    assert sm.fabric_links is not None and len(sm.fabric_links.tests) == 28 * 8 * 2
    lead = sm.point(sm.leader_point_id)
    assert lead.switch_load is not None and lead.switch_load.uplink is not None
    assert 0 < lead.switch_load.uplink.util_max <= 1
    assert (exp / "points" / lead.key.id / "switch_load.json").exists()


def test_service_state_bottleneck_energy_and_lifecycle(run8):
    _, _, sm = run8
    lead = sm.point(sm.leader_point_id)
    assert lead.state is not None and lead.state.bottleneck is not None
    assert lead.state.energy_per_token_j and lead.state.energy_per_token_j > 0
    assert lead.state.value("decode", "kv_usage") is not None
    assert all(d.lifecycle is not None for d in sm.deploys if d.ok)


def test_node_facts_software_clocks_disks_and_health(run8):
    _, exp, sm = run8
    tags = {
        e.data["tag"] for e in EventStore(exp / "events.jsonl").of_type(EventType.FACTS)
    }
    assert tags == {"before", "software", "clock", "disk", "after"}
    assert sm.software is not None and sm.software.packages
    assert {c.node for c in sm.clock_offsets} == {f"node{i}" for i in range(8)}
    assert len(sm.disk_reads) == 8 and all(d.gbs for d in sm.disk_reads)
    assert len(sm.health_deltas) == 8
    assert sm.facts_signature


def test_report_has_the_new_sections(run8):
    _, exp, _ = run8
    report = (exp / "report" / "report.zh-CN.md").read_text()
    for heading in ("## 2 带宽、算力与 KV 容量", "## 9 网络拓扑与上游交换机",
                    "## 10 部署与运行状态", "## 11 节点健康与环境", "## 12 推荐配置",
                    "## 14 复现方式"):  # fmt: skip
        assert heading in report
    assert "满载时交换机端口负载" in report and "瓶颈判断" in report
    assert "rail-optimized：8 条 rail，每台 leaf 下 4 个节点" in report
    assert "J/token" in report


def test_kb_records_carry_the_new_data(run8):
    work, _, sm = run8
    cluster = work / "kb" / "clusters" / sm.fingerprint
    fabric = [
        json.loads(x)
        for x in (cluster / "records" / "fabric.jsonl").read_text().splitlines()
    ]
    assert fabric[0]["rail_optimized"] is True and fabric[0]["leaf_names"]
    runs = [
        json.loads(x)
        for x in (cluster / "records" / "run.jsonl").read_text().splitlines()
    ]
    lead = next(r for r in runs if r["point_id"] == sm.leader_point_id)
    assert lead["bottleneck"] and lead["energy_per_token_j"] > 0
    assert lead["switch_uplink_util_max"] is not None
    deploys = [
        json.loads(x)
        for x in (cluster / "records" / "deploy.jsonl").read_text().splitlines()
    ]
    assert any(d["time_to_ready_s"] for d in deploys)
    index = [
        json.loads(x) for x in (work / "kb" / "index.jsonl").read_text().splitlines()
    ]
    assert index[0]["fabric_kind"] == "infiniband" and index[0]["leader_bottleneck"]
    assert (cluster / "fabric-and-health.md").exists()


def test_link_matrix_can_be_skipped(tmp_path):
    args = ["dry-run", "--workdir", str(tmp_path), "--exp-id", "t1", "--log-level",
            "WARNING", "--fake-gpu", "rtx_pro_6000", "--fake-nodes", "2",
            "--no-link-matrix"]  # fmt: skip
    assert main(args) == 0
    sm = ExperimentSummary.model_validate(
        json.loads((tmp_path / "experiments/t1/analysis/summary.json").read_text())
    )
    assert sm.fabric is not None and sm.fabric_links is None
