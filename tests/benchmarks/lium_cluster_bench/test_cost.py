# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rental price -> $/token."""

from __future__ import annotations

import json

import pytest

from benchmarks.lium_cluster_bench.cli import main
from benchmarks.lium_cluster_bench.domain.cost import (
    cost_summary,
    node_hour_price,
    usd_per_mtok,
)
from benchmarks.lium_cluster_bench.domain.schema import (
    ExperimentSummary,
    FloorResult,
    ProviderInfo,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_usd_per_million_output_tokens():
    # 4 nodes x $40/h = $160/h; 32 GPUs x 250 tok/s = 28.8 M tok/h -> $5.56/M
    assert usd_per_mtok(160.0, 250.0, 32) == pytest.approx(160 / 28.8)
    assert usd_per_mtok(160.0, 0.0, 32) is None


def test_price_precedence_cli_then_node_price_then_cluster_price():
    provider = ProviderInfo(price_per_node_hour=38.4, price_per_hour=160.0)
    assert node_hour_price(provider, 50.0, 4) == (50.0, "--price-per-node-hour")
    assert node_hour_price(provider, None, 4) == (38.4, "lium")
    assert node_hour_price(ProviderInfo(price_per_hour=160.0), None, 4) == (
        40.0,
        "lium",
    )
    assert node_hour_price(None, None, 4) is None


def test_cost_summary_covers_leader_and_floors():
    floors = (
        FloorResult(user_tps_floor=20.0, tps_per_gpu=200.0),
        FloorResult(user_tps_floor=50.0, interpolated_tps_per_gpu=100.0),
        FloorResult(user_tps_floor=80.0),
    )
    cost = cost_summary(
        price=(40.0, "lium"),
        node_count=8,
        gpu_count=64,
        spent_hours=9.0,
        leader_tps_per_gpu=250.0,
        floors=floors,
    )
    assert cost.cluster_per_hour == 320.0 and cost.experiment_usd == 2880.0
    assert cost.leader_usd_per_mtok == pytest.approx(
        320 / (250 * 64 * 3600) * 1e6, 1e-3
    )
    assert set(cost.floor_usd_per_mtok) == {"20", "50"}
    assert (
        cost_summary(
            price=None, node_count=8, gpu_count=64, spent_hours=1, leader_tps_per_gpu=1
        )
        is None
    )


def test_dry_run_reads_the_controller_provider_json(tmp_path):
    exp = tmp_path / "experiments" / "t1"
    exp.mkdir(parents=True)
    provider = ProviderInfo(
        cluster_id="c1",
        fabric_id="ib-x",
        fabric_type="infiniband",
        price_per_node_hour=30.0,
        node_count=2,
    )
    (exp / "provider.json").write_text(provider.model_dump_json())
    args = [
        "dry-run",
        "--workdir",
        str(tmp_path),
        "--exp-id",
        "t1",
        "--log-level",
        "WARNING",
        "--fake-gpu",
        "h200",
        "--fake-nodes",
        "2",
    ]
    assert main(args) == 0
    summary = ExperimentSummary.model_validate(
        json.loads((exp / "analysis" / "summary.json").read_text())
    )
    assert summary.provider is not None and summary.provider.fabric_id == "ib-x"
    assert summary.cost is not None and summary.cost.source == "lium"
    assert summary.cost.cluster_per_hour == 60.0
    assert summary.cost.leader_usd_per_mtok is not None


def test_kb_index_lines_up_one_shape_across_cluster_sizes(tmp_path):
    kb_root = tmp_path / "kb-shared"
    for nodes in (2, 4):
        work = tmp_path / f"w{nodes}"
        args = [
            "dry-run",
            "--workdir",
            str(work),
            "--exp-id",
            f"s{nodes}",
            "--log-level",
            "WARNING",
            "--fake-gpu",
            "rtx_pro_6000",
            "--fake-nodes",
            str(nodes),
            "--price-per-node-hour",
            "20",
        ]
        assert main(args) == 0
        exp_dir = work / "experiments" / f"s{nodes}"
        assert main(["kb", "--exp-dir", str(exp_dir), "--kb-root", str(kb_root)]) == 0
    index = (kb_root / "INDEX.md").read_text()
    assert "## Scaling across cluster sizes" in index
    assert "2 x 8" in index and "4 x 8" in index
    rows = [
        json.loads(line) for line in (kb_root / "index.jsonl").read_text().splitlines()
    ]
    assert {r["node_count"] for r in rows} == {2, 4}
    assert len({r["shape_fp"] for r in rows}) == 1 and len({r["fp"] for r in rows}) == 2
    assert all(r["leader_usd_per_mtok"] for r in rows)
