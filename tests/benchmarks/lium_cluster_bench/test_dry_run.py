# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline against simulated hardware (no GPUs, SSH or network)."""

from __future__ import annotations

import json

import pytest

from benchmarks.lium_cluster_bench.app.store import EventStore
from benchmarks.lium_cluster_bench.cli import main
from benchmarks.lium_cluster_bench.domain.schema import EventType, ExperimentSummary

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _run(tmp_path, *extra: str) -> int:
    return main(
        [
            "dry-run",
            "--workdir",
            str(tmp_path),
            "--exp-id",
            "t1",
            "--log-level",
            "WARNING",
            *extra,
        ]
    )


def _summary(tmp_path) -> ExperimentSummary:
    path = tmp_path / "experiments" / "t1" / "analysis" / "summary.json"
    return ExperimentSummary.model_validate(json.loads(path.read_text()))


@pytest.mark.parametrize("gpu", ["h200", "b200", "h100", "rtx_pro_6000"])
def test_full_pipeline_simulated(tmp_path, gpu):
    assert _run(tmp_path, "--fake-gpu", gpu) == 0
    summary = _summary(tmp_path)
    assert summary.verdict in ("ok", "partial")
    assert summary.leader is not None
    assert summary.noise is not None
    assert {f.user_tps_floor for f in summary.floors} == {20.0, 50.0}
    measured = [f for f in summary.families if f.status == "measured"]
    assert measured, [f.model_dump() for f in summary.families]
    assert summary.nic_arms, summary.nic_reason
    assert summary.confirm_point_id is not None
    assert any("DRY RUN" in c for c in summary.conditions)
    exp = tmp_path / "experiments" / "t1"
    assert (exp / "report" / "report.zh-CN.md").exists()
    assert (exp / "report" / "report.zh-CN.html").exists()
    assert (tmp_path / "kb").exists()
    commands = (exp / "dry_run_commands.txt").read_text()
    assert "docker run" in commands and "aiperf" in commands
    # the three focus resources of the leader
    assert summary.model_arch is not None and summary.model_arch.mla
    need = summary.interconnect
    assert need is not None and need.point_id == summary.leader_point_id
    assert {n.link for n in need.links} == {"nvlink", "pcie", "pcie_nic", "ib"}
    assert need.link("ib").status != "unavailable"
    assert need.min_nics_estimated is not None and need.min_nics_reason
    kv = summary.kv_capacity
    assert kv is not None and kv.total_tokens and kv.max_request_tokens
    assert kv.requests_at_p50 is not None and kv.usage_peak is not None
    lead = summary.point(summary.leader_point_id)
    assert lead is not None and lead.mfu is not None
    assert lead.mfu.total_tflops_per_gpu and lead.mfu.total_tflops_per_gpu > 0
    report = (exp / "report" / "report.zh-CN.md").read_text()
    assert "## 2 带宽、算力与 KV 容量" in report
    assert "### 2.4 KV cache 容量" in report and "可同时放下的上下文" in report
    # NVLink off vs on: hardware switch on Hopper, NCCL without P2P on
    # Blackwell (no -sBwMode, no width 0), nothing on PCIe-only GPUs
    nvl = summary.nvlink
    assert nvl is not None and "### 2.2 禁用 NVLink 改走 PCIe 的对比" in report
    if gpu == "rtx_pro_6000":
        assert nvl.status == "skipped" and "PCIe-only" in nvl.reason
        assert "nvidia-smi nvlink" not in commands
    else:
        assert nvl.status == "measured" and nvl.delta_pct is not None
        assert nvl.on_point_id == summary.leader_point_id
        assert nvl.nvlink_p99_gbs == 0.0
        off = summary.point(nvl.off_point_id)
        assert off is not None and off.key.purpose == "nvlink"
        assert off.candidate.p2p == nvl.path
        if gpu in ("h200", "h100"):
            assert (nvl.method, nvl.path, nvl.restored) == ("bw_mode_off", "pcie", True)
            assert "nvidia-smi nvlink -sBwMode OFF" in commands
            assert "nvidia-smi nvlink -sBwMode FULL" in commands
            assert "-sLWidth" not in commands  # Hopper never gets the width command
        else:
            assert (nvl.method, nvl.path) == ("nccl_no_p2p", "host")
            assert "-sBwMode OFF" not in commands
    resources = tmp_path / "kb" / "clusters" / summary.fingerprint / "resources.md"
    assert "## KV cache capacity" in resources.read_text()


@pytest.mark.parametrize(
    ("gpu", "nodes", "pd_id"),
    [
        ("b200", 2, "pd-2p1d"),
        ("b200", 8, "pd-8p4d"),
        ("h100", 2, None),
        ("h200", 8, "pd-4p4d"),
    ],
)
def test_two_and_eight_node_clusters(tmp_path, gpu, nodes, pd_id):
    assert _run(tmp_path, "--fake-gpu", gpu, "--fake-nodes", str(nodes)) == 0
    summary = _summary(tmp_path)
    assert summary.node_count == nodes
    assert summary.leader is not None and summary.nic_arms
    ids = {f.family_id for f in summary.families}
    if pd_id:
        assert pd_id in ids  # ids are rendered with the fitted instance counts
    assert not any("{" in i for i in ids)
    assert all(
        p.candidate.family_id in ids for p in summary.points if p.key.purpose != "smoke"
    )
    assert (
        summary.shape_fingerprint and summary.fingerprint != summary.shape_fingerprint
    )
    report = (
        tmp_path / "experiments" / "t1" / "report" / "report.zh-CN.md"
    ).read_text()
    assert f"{nodes} 节点" in report


def test_resume_is_a_no_op_after_completion(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "h200") == 0
    store = EventStore(tmp_path / "experiments" / "t1" / "events.jsonl")
    points_before = len(store.of_type(EventType.POINT))
    assert _run(tmp_path, "--fake-gpu", "h200") == 0
    store = EventStore(tmp_path / "experiments" / "t1" / "events.jsonl")
    assert len(store.of_type(EventType.POINT)) == points_before


def test_resume_after_crash_mid_search(tmp_path):
    exp = tmp_path / "experiments" / "t1"
    # SIGTERM after 5 measured points: the runner exits "resumable" (75).
    assert _run(tmp_path, "--fake-gpu", "h200", "--fake-crash-after", "5") == 75
    store = EventStore(exp / "events.jsonl")
    before = {
        e.data["result"]["key"]["cand_key"]
        + str(e.data["result"]["key"]["concurrency"])
        for e in store.of_type(EventType.POINT)
    }
    assert len(before) == 5
    assert json.loads((exp / "status.json").read_text())["state"] == "interrupted"
    # Resume with the same experiment id: nothing measured twice, run completes.
    assert _run(tmp_path, "--fake-gpu", "h200") == 0
    store = EventStore(exp / "events.jsonl")
    ids = [e.data["result"]["key"] for e in store.of_type(EventType.POINT)]
    keys = [
        (k["cand_key"], k["concurrency"], k["spec"], k["purpose"], k["rep"])
        for k in ids
    ]
    assert len(keys) == len(set(keys)), "a point was measured twice"
    assert store.of_type(EventType.POINT_ABORTED), "the interrupted point is archived"
    phases = [e.data["phase"] for e in store.of_type(EventType.PHASE_DONE)]
    assert phases.count("preflight") == 1 and phases.count("search") == 1
    assert _summary(tmp_path).leader is not None


def test_kernel_ladder_finds_working_rung(tmp_path):
    # Every family needs the tilelang DSA rung: the smoke finds it, search inherits it.
    assert (
        _run(
            tmp_path,
            "--fake-gpu",
            "rtx_pro_6000",
            "--fake-needs-rungs",
            "sm120-dsa-tilelang",
        )
        == 0
    )
    summary = _summary(tmp_path)
    assert summary.leader is not None and "sm120-dsa-tilelang" in summary.leader.rungs


def test_unsupported_verdict_when_every_kernel_fallback_fails(tmp_path):
    assert (
        _run(
            tmp_path, "--fake-gpu", "rtx_pro_6000", "--fake-needs-rungs", "no-such-rung"
        )
        == 0
    )
    summary = _summary(tmp_path)
    assert summary.verdict == "unsupported"
    assert summary.leader is None
    assert not [p for p in summary.points if p.key.purpose == "search"]
    assert summary.peaks, "speed-of-light is still reported for an unsupported model"


def test_budget_exhaustion_yields_partial_report(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "h200", "--budget-hours", "4") == 0
    summary = _summary(tmp_path)
    assert summary.verdict == "partial"
    assert summary.leader is not None  # the baseline family is always measured
    assert summary.spent_hours <= 4.5


def test_accuracy_gate_replaces_a_broken_leader(tmp_path):
    """The throughput leader fails GSM8K: the next best is checked, passes and
    becomes the recommendation; the failure is recorded and reported."""
    assert (
        _run(tmp_path, "--fake-gpu", "b200", "--fake-accuracy-fail", "pd-4p2d")
        == 0
    )  # fmt: skip
    sm = _summary(tmp_path)
    statuses = [(c.family_id, c.verdict.status) for c in sm.accuracy]
    assert statuses[0] == ("pd-4p2d", "fail")
    assert statuses[-1][1] == "pass"
    lead = sm.point(sm.leader_point_id)
    assert lead is not None and lead.candidate.family_id != "pd-4p2d"
    assert sm.accuracy_of(lead.candidate.key).verdict.status == "pass"
    # the pass is judged against passing configs only, not the broken one
    assert sm.accuracy_of(lead.candidate.key).verdict.best_accuracy is None
    assert any(x.startswith("accuracy gate excluded pd-4p2d") for x in sm.limitations)
    family = next(f for f in sm.families if f.family_id == "pd-4p2d")
    assert family.accuracy == "fail"
    report = (
        tmp_path / "experiments" / "t1" / "report" / "report.zh-CN.md"
    ).read_text()
    assert "精度检查淘汰了 pd-4p2d" in report and "精度未通过" in report
    # the recommended config states the check it passed, not "unverified"
    assert "计算正确性：GSM8K 精度检查通过" in report
    kb = list((tmp_path / "kb").rglob("accuracy.jsonl"))
    assert kb and '"status":"fail"' in kb[0].read_text().replace(" ", "")


def test_accuracy_gate_can_be_disabled(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "h200", "--no-accuracy") == 0
    sm = _summary(tmp_path)
    assert sm.accuracy == ()
    assert any("accuracy gate disabled" in x for x in sm.limitations)
    report = (
        tmp_path / "experiments" / "t1" / "report" / "report.zh-CN.md"
    ).read_text()
    assert "计算正确性：没有做精度检查" in report


def test_diagnostic_window_records_kernels_without_touching_the_ranking(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "b200", "--diagnose") == 0
    sm = _summary(tmp_path)
    diag = sm.diagnostics
    assert (
        diag is not None and diag.nsys == "ok" and diag.cand_key.endswith("|diag=nsys")
    )
    assert set(diag.kernels) >= {"decode", "prefill"}
    assert any(k.endswith(".est_tflops_per_gpu") for k in diag.engine)
    # the window's load is a POINT with purpose "diagnose", never the leader
    lead = sm.point(sm.leader_point_id)
    assert lead is not None and not lead.candidate.diag
    assert any(p.key.purpose == "diagnose" for p in sm.points)
    report = (
        tmp_path / "experiments" / "t1" / "report" / "report.zh-CN.md"
    ).read_text()
    assert "诊断时段（不计入正式结果）" in report and "DeepEP all-to-all" in report
    index = next((tmp_path / "kb").rglob("index.jsonl")).read_text()
    assert '"diag_nsys":"ok"' in index.replace(" ", "")


@pytest.mark.parametrize("extra", [(), ("--diagnose",)])
def test_default_budget_keeps_every_later_step(tmp_path, extra):
    """The search spends only what the later steps will not need: the leader's
    climb is not cut by the budget and no later step is skipped for it."""
    assert _run(tmp_path, "--fake-gpu", "b200", *extra) == 0
    store = EventStore(tmp_path / "experiments" / "t1" / "events.jsonl")
    decisions = [e.data for e in store.of_type(EventType.DECISION)]
    results = {
        d["phase"]: d["reason"] for d in decisions if d.get("kind") == "phase_result"
    }
    later = ["mtp", "accuracy", "pilot", "nic_sweep", "confirm", "nvlink"]
    later += ["diagnose"] if extra else []
    for phase in later:
        assert "budget" not in results[phase], (phase, results[phase])
    closed = {
        d["family_id"]: d["reason"]
        for d in decisions
        if d.get("kind") == "close_family"
    }
    sm = _summary(tmp_path)
    lead = sm.point(sm.leader_point_id)
    assert lead is not None
    assert "budget" not in closed[lead.candidate.family_id], closed
    assert sm.spent_hours <= 9.5


def test_nvlink_comparison_can_be_turned_off(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "h200", "--no-nvlink-ablation") == 0
    sm = _summary(tmp_path)
    assert sm.nvlink is not None and sm.nvlink.status == "skipped"
    assert sm.nvlink.reason == "NVLink comparison disabled (--no-nvlink-ablation)"
    assert not any(p.key.purpose == "nvlink" for p in sm.points)
    exp = tmp_path / "experiments" / "t1"
    assert "nvidia-smi nvlink" not in (exp / "dry_run_commands.txt").read_text()
    report = (exp / "report" / "report.zh-CN.md").read_text()
    assert "用 --no-nvlink-ablation 关掉了这一步" in report
    assert "--no-nvlink-ablation" in report.split("## 14 ", 1)[1]  # repro keeps it


def test_link_target_util_sizes_the_interconnect(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "h200", "--link-target-util", "0.5") == 0
    need = _summary(tmp_path).interconnect
    assert need is not None and need.target_util == 0.5
    nvlink = need.link("nvlink")
    assert nvlink.need_gbs == pytest.approx(nvlink.p99_gbs / 0.5, abs=0.01)
    report = (
        tmp_path / "experiments" / "t1" / "report" / "report.zh-CN.md"
    ).read_text()
    assert "p99 占用不超过 50%" in report
    assert "--link-target-util 0.5" in report  # the repro command keeps it
