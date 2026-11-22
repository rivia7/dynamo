# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact per-point views (switch load, service state), rail mapping and the
fail-soft guards around the new collectors."""

from __future__ import annotations

import json

import pytest

from benchmarks.cluster_bench.app.fabric_ops import rail_map
from benchmarks.cluster_bench.app.simulate import Simulator, synthetic_inventory
from benchmarks.cluster_bench.cli import main
from benchmarks.cluster_bench.domain.fabric import SwitchPortLoad, SwitchPortRef
from benchmarks.cluster_bench.domain.lifecycle import RoleLifecycle
from benchmarks.cluster_bench.domain.pointstate import (
    brief_roles,
    max_running_by_role,
    point_state,
)
from benchmarks.cluster_bench.domain.schema import ExperimentSummary
from benchmarks.cluster_bench.domain.servicestate import (
    Bottleneck,
    MetricValue,
    RoleState,
    ServiceState,
)
from benchmarks.cluster_bench.domain.switchload import summarize_switch_load

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _ref(port: int, role: str, peer: str | None = None) -> SwitchPortRef:
    return SwitchPortRef(
        switch_guid="0xleaf",
        switch_desc="leaf-1",
        switch_lid=7,
        port=port,
        role=role,
        rate_gbps=400.0,
        rail=0,
        peer_node=peer if role == "downlink" else None,
        peer_guid="0xspine" if role == "uplink" else None,
        via_node="node0",
        via_hca="mlx5_0",
        via_port=1,
    )


def test_switch_load_splits_uplinks_and_downlinks_and_ranks_hot_ports():
    loads = [
        SwitchPortLoad(ref=_ref(1, "downlink", "node0"), interval_s=10, tx_gbs=20,
                       rx_gbs=10, tx_util=0.4, rx_util=0.2),
        SwitchPortLoad(ref=_ref(33, "uplink"), interval_s=10, tx_gbs=45, rx_gbs=5,
                       tx_util=0.9, rx_util=0.1, wait_ratio=0.02,
                       error_deltas={"PortXmitDiscards": 3, "SymbolErrorCounter": 1}),
        SwitchPortLoad(ref=_ref(34, "uplink"), interval_s=10, reset=True),
    ]  # fmt: skip
    summary = summarize_switch_load(loads, hottest=2)
    assert (summary.sampled, summary.measured, summary.resets) == (3, 2, 1)
    assert summary.uplink.ports == 1 and summary.uplink.util_max == 0.9
    assert summary.downlink.util_max == 0.4 and summary.downlink.tx_gbs == 20
    assert summary.discards == 3 and summary.errors == {"SymbolErrorCounter": 1}
    assert [h.port for h in summary.hottest] == [33, 1]
    assert summarize_switch_load([]) is None


def test_point_state_keeps_the_saturation_signals():
    service = ServiceState(
        source="json",
        roles={
            "decode": RoleState(
                role="decode",
                endpoints=2,
                metrics={
                    "kv_usage": MetricValue(kind="gauge", source="sglang:token_usage",
                                            avg=0.93, max=0.99),
                    "running_reqs": MetricValue(kind="gauge", source="x", avg=240, max=256),
                },
            )
        },
    )  # fmt: skip
    state = point_state(service, None, Bottleneck(label="kv_capacity"), 2.5)
    assert state.value("decode", "kv_usage") == 0.93
    assert state.value("decode", "running_reqs", "max") == 256
    assert state.bottleneck.label == "kv_capacity" and state.energy_per_token_j == 2.5
    assert brief_roles(ServiceState()) == {}
    roles = {
        "decode": RoleLifecycle(
            role="decode", containers=2, max_running_requests_min=128
        ),
        "frontend": RoleLifecycle(role="frontend", containers=1),
    }
    assert max_running_by_role(roles) == {"decode": 128}


def test_rail_map_uses_the_lowest_gpu_an_hca_serves():
    inventory = synthetic_inventory("b200", nodes=2)
    rails = rail_map(inventory)
    assert set(rails) == {"node0", "node1"}
    assert sorted(rails["node0"].values()) == list(range(8))


def test_a_failing_fabric_probe_never_ends_the_run(tmp_path, monkeypatch):
    def boom(self):
        raise ValueError("smp storm")

    monkeypatch.setattr(Simulator, "fabric_reports", boom)
    args = ["dry-run", "--workdir", str(tmp_path), "--exp-id", "t1", "--log-level",
            "WARNING", "--fake-gpu", "h200", "--fake-nodes", "2"]  # fmt: skip
    assert main(args) == 0
    summary = ExperimentSummary.model_validate(
        json.loads((tmp_path / "experiments/t1/analysis/summary.json").read_text())
    )
    assert summary.fabric is None and summary.leader is not None
    assert any("fabric probing failed: smp storm" in x for x in summary.limitations)
    report = (tmp_path / "experiments/t1/report/report.zh-CN.md").read_text()
    assert "没有采集到网络拓扑" in report and "网络拓扑探测失败：smp storm" in report
