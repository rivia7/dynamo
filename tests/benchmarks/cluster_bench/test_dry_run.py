# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline against simulated hardware (no GPUs, SSH or network)."""

from __future__ import annotations

import json

import pytest
import yaml

from benchmarks.cluster_bench.app.phases.preflight import PreflightError
from benchmarks.cluster_bench.app.store import EventStore
from benchmarks.cluster_bench.cli import main
from benchmarks.cluster_bench.domain import afd
from benchmarks.cluster_bench.domain.schema import EventType, ExperimentSummary

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
    # MBU: the measured GPM DRAM bandwidth, in the same result as the MFU
    assert lead.mfu.total_mbu is not None and 0 < lead.mfu.total_mbu < 1
    assert lead.mfu.total_mem_tbs_per_gpu
    report = (exp / "report" / "report.zh-CN.md").read_text()
    assert "## 2 带宽、算力与 KV 容量" in report
    assert "MFU / MBU" in report and "DRAM_BW_UTIL" in report
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
            "sm120-dsa-tilelang-bf16-kv",
        )
        == 0
    )
    summary = _summary(tmp_path)
    assert (
        summary.leader is not None
        and "sm120-dsa-tilelang-bf16-kv" in summary.leader.rungs
    )


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


def test_confirm_duration_is_configurable_above_the_agentx_floor(tmp_path):
    with pytest.raises(SystemExit, match="--confirm-duration-s must be >= 900"):
        _run(tmp_path, "--fake-gpu", "h200", "--confirm-duration-s", "600")
    assert _run(tmp_path, "--fake-gpu", "h200", "--confirm-duration-s", "3600") == 0
    confirm = [p for p in _summary(tmp_path).points if p.key.purpose == "confirm"]
    assert confirm and all(p.metrics.duration_s == 3600 for p in confirm)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--model-path", "models/glm"), "--model-path must be an absolute path"),
        (("--hf-endpoint", "hf-mirror.com"), r"--hf-endpoint must be an http\(s\) URL"),
        (
            ("--checkpoint", "nvfp4", "--model-repo", "org/m", "--model-size-gb", "10"),
            "--checkpoint picks a tuned profile's checkpoint",
        ),
    ],
)
def test_weight_source_options_are_checked(tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        _run(tmp_path, "--fake-gpu", "h200", *extra)


def test_d_only_on_rtx_pro_6000d_with_weights_on_every_node(tmp_path):
    model_path = "/data/models/GLM-5.3-NVFP4"
    assert (
        _run(
            tmp_path,
            "--fake-gpu",
            "rtx_pro_6000d",
            "--fake-nodes",
            "2",
            "--stage",
            "d-only",
            "--checkpoint",
            "nvfp4-nvidia",
            "--model-path",
            model_path,
            "--hf-endpoint",
            "https://hf-mirror.com",
        )
        == 0
    )
    summary = _summary(tmp_path)
    assert summary.verdict in ("ok", "partial")
    measured = [f.family_id for f in summary.families if f.status == "measured"]
    assert measured == ["d-only-tp8x2"]
    exp = tmp_path / "experiments" / "t1"
    decode = [
        yaml.safe_load(content)
        for plan in (exp / "deployments").glob("*/plan.json")
        for spec in json.loads(plan.read_text())["containers"]
        if spec["role"] == "decode"
        for path, content in spec["files"].items()
        if path.endswith(".yaml")
    ]
    assert decode
    for cfg in decode:
        assert cfg["model-path"] == model_path
        assert cfg["quantization"] == "modelopt_mixed"
        assert cfg["disable-shared-experts-fusion"] is True
        assert cfg["disaggregation-transfer-backend"] == "fake"
        assert "data-parallel-size" not in cfg


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


def test_redo_adds_a_diagnostic_window_to_a_finished_run(tmp_path):
    assert _run(tmp_path, "--fake-gpu", "b200") == 0
    exp = tmp_path / "experiments" / "t1"
    before = _summary(tmp_path)
    assert before.diagnostics is None
    store = EventStore(exp / "events.jsonl")
    measured = len(store.of_type(EventType.POINT))
    # without --diagnose, or on an experiment that never ran: refused
    with pytest.raises(SystemExit, match="needs --diagnose"):
        _run(tmp_path, "--fake-gpu", "b200", "--redo", "diagnose")
    with pytest.raises(SystemExit, match="cannot redo"):
        _run(tmp_path, "--fake-gpu", "b200", "--redo", "search")
    with pytest.raises(SystemExit, match="has not run"):
        main(
            [
                "dry-run",
                "--workdir",
                str(tmp_path),
                "--exp-id",
                "t2",
                "--diagnose",
                "--redo",
                "diagnose",
                "--log-level",
                "WARNING",
            ]
        )
    assert _run(tmp_path, "--fake-gpu", "b200", "--diagnose", "--redo", "diagnose") == 0
    after = _summary(tmp_path)
    diag = after.diagnostics
    assert diag is not None and diag.nsys == "ok" and diag.kernels
    lead = before.point(before.leader_point_id)
    assert lead is not None and diag.family_id == lead.candidate.family_id
    store = EventStore(exp / "events.jsonl")
    points = store.of_type(EventType.POINT)
    # one more point, the window's own load; nothing else is measured again
    assert len(points) == measured + 1
    assert points[-1].data["result"]["key"]["purpose"] == "diagnose"
    phases = [e.data["phase"] for e in store.of_type(EventType.PHASE_DONE)]
    assert phases.count("search") == 1 and phases.count("confirm") == 1
    assert phases.count("diagnose") == 2 and phases.count("finalize") == 2
    assert after.leader_point_id == before.leader_point_id
    assert any("(--redo)" in n.data["text"] for n in store.of_type(EventType.NOTE))
    # a resumed run gives the peers this runner's code again
    pushed = (exp / "dry_run_commands.txt").read_text()
    assert "<push " in pushed
    report = (exp / "report" / "report.zh-CN.md").read_text()
    assert "诊断时段（不计入正式结果）" in report
    # the wall-time split of the captured steps, per role
    assert diag.capture_steps == 10 and diag.timeline
    for raw in diag.timeline.values():
        assert sum(raw["shares"].values()) == pytest.approx(1.0, abs=1e-4)
        assert raw["step_ms"] > 0 and raw["sample"]
    assert "一个 forward step 的墙钟时间去向" in report
    assert "最长的几段空闲" in report
    assert "其中超出中位耗时的部分" in report and "最长的几次通信 kernel" in report
    assert "去掉停顿之后的时间去向" in report
    # the same split drawn: bars per role, then per role every GPU's window,
    # a stretch of the timeline and one layer of it
    assert "](figures/fig-diag-steps.svg)" in report
    for role in diag.timeline:
        for chart in ("idle", "strip", "zoom"):
            assert f"](figures/fig-diag-{chart}-{role}.svg)" in report
    # a second redo reads the window's reports again: no deployment, no load
    assert _run(tmp_path, "--fake-gpu", "b200", "--diagnose", "--redo", "diagnose") == 0
    again = _summary(tmp_path).diagnostics
    assert again is not None and again.kernels == diag.kernels
    assert again.timeline == diag.timeline
    assert any("re-analyzed" in n for n in again.notes)
    # a third one replaces the second one's note instead of adding to it
    assert _run(tmp_path, "--fake-gpu", "b200", "--diagnose", "--redo", "diagnose") == 0
    third = _summary(tmp_path).diagnostics
    assert third is not None and third.notes == again.notes
    store = EventStore(exp / "events.jsonl")
    assert len(store.of_type(EventType.POINT)) == measured + 1
    assert _run(tmp_path, "--fake-gpu", "b200", "--redo", "finalize") == 0
    assert _summary(tmp_path).diagnostics == again


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


def _summary_of(tmp_path, exp_id: str) -> ExperimentSummary:
    path = tmp_path / "experiments" / exp_id / "analysis" / "summary.json"
    return ExperimentSummary.model_validate(json.loads(path.read_text()))


def test_p_only_and_d_only_dry_runs_give_the_kb_a_pd_split(tmp_path):
    common = [
        "dry-run", "--workdir", str(tmp_path), "--log-level", "WARNING",
        "--fake-gpu", "h200", "--fake-nodes", "2", "--budget-hours", "4",
    ]  # fmt: skip
    assert main([*common, "--exp-id", "p1", "--stage", "p-only"]) == 0
    p = _summary_of(tmp_path, "p1")
    assert (p.stage, p.platform, p.floors) == ("prefill", "generic", ())
    assert [f.family_id for f in p.families] == ["p-only-tp8ep8x2"]
    lead = p.point(p.leader_point_id)
    assert lead is not None and lead.metrics is not None
    assert lead.metrics.osl_avg is not None and lead.metrics.osl_avg <= 1.5
    assert lead.tps_per_gpu == pytest.approx(lead.input_tps_per_gpu)
    report = (
        tmp_path / "experiments" / "p1" / "report" / "report.zh-CN.md"
    ).read_text()
    assert "P-Only" in report and "输入 tok/s/GPU" in report
    assert main([*common, "--exp-id", "d1", "--stage", "d-only"]) == 0
    d = _summary_of(tmp_path, "d1")
    assert d.stage == "decode"
    assert [f.family_id for f in d.families] == ["d-only-dtp8x2"]
    assert d.leader is not None and d.leader.simulate_acceptance
    index = (tmp_path / "kb" / "INDEX.md").read_text()
    assert "## PD split from P-Only and D-Only" in index
    assert "`p1` / `d1`" in index and " : 1 |" in index


def test_glm_5_3_flash_d_only_checks_its_sglang_hotfix(tmp_path):
    model_path = "/data/models/GLM-5.3-Flash-NVFP4"
    assert (
        _run(
            tmp_path,
            "--fake-gpu",
            "rtx_pro_6000d",
            "--fake-nodes",
            "2",
            "--stage",
            "d-only",
            "--model",
            "glm-5.3-flash",
            "--checkpoint",
            "nvfp4-nvidia",
            "--model-path",
            model_path,
            "--hf-endpoint",
            "https://hf-mirror.com",
        )
        == 0
    )
    summary = _summary(tmp_path)
    assert summary.verdict in ("ok", "partial")
    measured = {f.family_id for f in summary.families if f.status == "measured"}
    assert measured and measured <= {"d-only-tp4x4", "d-only-tp8x2"}
    exp = tmp_path / "experiments" / "t1"
    assert (exp / "preflight" / "sglang_hotfixes.txt").exists()
    commands = (exp / "dry_run_commands.txt").read_text()
    assert "sglang_hotfix.py check sm120-tilelang-dsa-v1-one-stage" in commands


def _a_only(tmp_path, *extra: str) -> int:
    return _run(
        tmp_path,
        "--fake-gpu",
        "rtx_pro_6000d",
        "--fake-nodes",
        "2",
        "--stage",
        "a-only",
        "--checkpoint",
        "nvfp4-nvidia",
        *extra,
    )


def _f_waits(exp) -> list[str]:
    """CB_AFD_F_WAIT of every A worker launched, checking its config."""
    waits = []
    for path in sorted((exp / "deployments").glob("*/plan.json")):
        plan = json.loads(path.read_text())
        assert plan["stage"] == "attention"
        for cfg in plan["rendered_configs"].values():
            assert cfg["enable-dp-attention"] is True
            assert cfg["data-parallel-size"] == cfg["tensor-parallel-size"]
            assert cfg["disaggregation-transfer-backend"] == "fake"
        for spec in plan["containers"]:
            if spec["role"] == "decode":
                env = spec["env"]
                assert env[afd.ATTN_ONLY_ENV] == "1"
                assert afd.HOTFIX_ID in env["CB_SGLANG_HOTFIXES"].split(",")
                waits.append(env[afd.F_WAIT_ENV])
    return waits


def _hotfixes(exp) -> set[str]:
    """SGLang hotfix ids of the A workers launched."""
    return {
        hotfix
        for path in (exp / "deployments").glob("*/plan.json")
        for spec in json.loads(path.read_text())["containers"]
        if spec["role"] == "decode"
        for hotfix in spec["env"]["CB_SGLANG_HOTFIXES"].split(",")
    }


@pytest.mark.parametrize(
    "model, families, hotfixes",
    [
        ("glm-5.3", {"a-only-dtp8x2"}, {afd.HOTFIX_ID}),
        (
            "glm-5.3-flash",
            {"a-only-dtp4x4", "a-only-dtp8x2"},
            {afd.HOTFIX_ID, "sm120-tilelang-dsa-v1-one-stage"},  # the profile's too
        ),
    ],
)
def test_a_only_waits_for_a_simulated_f_and_measures_the_null_f_bound(
    tmp_path, model, families, hotfixes
):
    assert _a_only(tmp_path, "--model", model) == 0
    summary = _summary(tmp_path)
    assert summary.stage == "attention" and summary.verdict in ("ok", "partial")
    measured = {f.family_id for f in summary.families if f.status == "measured"}
    assert measured == families
    lead = summary.point(summary.leader_point_id)
    assert lead is not None and lead.candidate.simulate_acceptance
    assert lead.mfu is not None and lead.mfu.precision == "bf16"
    assert afd.MFU_NOTE in lead.mfu.notes
    report = summary.afd
    assert report is not None and report.n_a in (4, 8)
    bound = report.bound
    assert bound.status == "measured", bound.reason
    null = summary.point(bound.null_point_id)
    assert null is not None and null.key.purpose == "f_bound"
    assert null.candidate == lead.candidate.model_copy(update={"null_f": True})
    assert null.key.concurrency == lead.key.concurrency == bound.concurrency
    assert bound.f_cost_pct is not None and bound.f_cost_pct > 0
    traffic = report.traffic
    assert traffic is not None and traffic.to_f_bytes_per_token > 0
    assert traffic.sim_need_gbs_per_gpu and traffic.null_need_gbs_per_gpu
    assert traffic.null_need_gbs_per_gpu > traffic.sim_need_gbs_per_gpu
    assert report.steps and report.weight_gb_per_gpu is not None
    expected = (report.null_weight_gb_per_gpu, report.loaded_weight_gb_per_gpu)
    if model == "glm-5.3":  # the simulated config.json's experts fit its checkpoint
        assert afd.weights_engaged(report.weight_gb_per_gpu, expected)
    else:  # they outweigh Flash's 203 GB: no expected size, so no check
        assert expected == (None, None)
    exp = tmp_path / "experiments" / "t1"
    assert _hotfixes(exp) == hotfixes
    waits = _f_waits(exp)
    assert afd.F_WAIT_OFF in waits and len(set(waits)) >= 2
    assert (exp / "preflight" / "afd_wait_kernel.txt").exists()
    assert "sglang_afd.py --check" in (exp / "dry_run_commands.txt").read_text()
    md = (exp / "report" / "report.zh-CN.md").read_text()
    assert "### 2.5 A-Only" in md and "--stage a-only" in md
    a_side = afd.a_side_arch(summary.model_arch)  # MFU counts A's FLOPs only
    assert f"激活参数约 {a_side.active_params() / 1e9:.1f}B" in md
    records = [
        json.loads(line)
        for line in (tmp_path / "kb" / "index.jsonl").read_text().splitlines()
    ]
    (record,) = [r for r in records if r["exp_id"] == "t1"]
    assert (record["stage"], record["afd_bound_status"]) == ("attention", "measured")
    assert record["afd_f_cost_pct"] == pytest.approx(bound.f_cost_pct, abs=0.01)
    need = record["afd_need_gbs_per_gpu"]
    assert need == pytest.approx(traffic.sim_need_gbs_per_gpu, abs=1e-3)
    (resources,) = (tmp_path / "kb" / "clusters").glob("*/resources.md")
    assert "## A-Only: simulated F" in resources.read_text()


def test_a_only_takes_its_f_from_the_command_line(tmp_path):
    text = "bw_gbps=400,lat_us=8,microbatches=1"
    assert _a_only(tmp_path, "--afd-f", text) == 0
    summary = _summary(tmp_path)
    assert summary.afd is not None
    plan = summary.afd.plan
    assert plan.spec == afd.parse_afd_f(text)
    assert (plan.link_gbs_per_gpu, plan.lat_us) == (50.0, 8.0)
    assert (plan.link_source, plan.lat_source) == ("--afd-f bw_gbs", "--afd-f lat_us")
    # one microbatch: every MoE layer waits for F serially
    assert all(row.overlap_us == row.serial_us for row in summary.afd.steps)
    md = (tmp_path / "experiments" / "t1" / "report" / "report.zh-CN.md").read_text()
    assert f"--afd-f {afd.format_afd_f(plan.spec)}" in md


def test_a_only_takes_a_new_f_after_its_plan_failed(tmp_path):
    exp = tmp_path / "experiments" / "t1"
    # 0.01 GB/s: a forward would wait 43 s for F, so the plan stops the run
    with pytest.raises(afd.AfdError, match="above 10 s"):
        _a_only(tmp_path, "--afd-f", "bw_gbs=0.01")
    assert not list((exp / "deployments").glob("*/plan.json"))
    good = "bw_gbs=50,lat_us=14"
    assert _a_only(tmp_path, "--afd-f", good) == 0
    summary = _summary(tmp_path)
    assert summary.afd is not None and summary.afd.plan.spec == afd.parse_afd_f(good)
    md = (exp / "report" / "report.zh-CN.md").read_text()
    assert f"--afd-f {afd.format_afd_f(summary.afd.plan.spec)}" in md


def test_a_only_runs_a_model_repo_moe_given_its_expert_precision(tmp_path):
    a_only = ("--fake-gpu", "rtx_pro_6000d", "--fake-nodes", "2", "--stage", "a-only")
    repo = ("--model-repo", "org/moe", "--model-size-gb", "400")
    # no profile, no variant name: the routed experts' precision is unknown
    with pytest.raises(PreflightError, match="pass --afd-f expert_dtype="):
        _run(tmp_path, *a_only, *repo)
    assert _run(tmp_path, *a_only, *repo, "--afd-f", "expert_dtype=nvfp4") == 0
    summary = _summary(tmp_path)
    assert summary.afd is not None and summary.afd.plan.expert_dtype == "nvfp4"
    assert summary.afd.bound.status == "measured"


def test_a_only_resumes_with_the_f_it_planned(tmp_path):
    exp = tmp_path / "experiments" / "t1"
    assert _a_only(tmp_path, "--fake-crash-after", "3") == 75
    assert _a_only(tmp_path) == 0
    store = EventStore(exp / "events.jsonl")
    plans = [
        e for e in store.of_type(EventType.DECISION) if e.data.get("kind") == "afd_plan"
    ]
    assert len(plans) == 1  # resolved once, before the first deployment
    waits = {w for w in _f_waits(exp) if w != afd.F_WAIT_OFF}
    assert len(waits) == 1  # one A instance size: the same wait before and after
