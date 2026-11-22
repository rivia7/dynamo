# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Chinese report renderer and its SVG charts."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from benchmarks.cluster_bench.domain.capacity import interconnect_needs
from benchmarks.cluster_bench.domain.schema import (
    FloorResult,
    Inventory,
    MtpComparison,
    NicArm,
)
from benchmarks.cluster_bench.report import svg
from benchmarks.cluster_bench.report.doc import Span, esc, md_table, strong
from benchmarks.cluster_bench.report.render import render_report
from benchmarks.cluster_bench.report.strings import Strings
from tests.benchmarks.cluster_bench import summary_factory as factory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

SECTION_KEYS = (
    "summary",
    "resources",
    "env",
    "method",
    "topology",
    "mtp",
    "bandwidth",
    "nic",
    "config",
    "limits",
    "repro",
)
FIGURES = {
    "fig-tps-concurrency.svg",
    "fig-tps-user.svg",
    "fig-mtp.svg",
    "fig-bw-util.svg",
    "fig-nic.svg",
}


@pytest.fixture
def strings():
    return Strings.load("zh_CN")


@pytest.fixture
def h200():
    return factory.h200_summary()


@pytest.fixture
def plan_lookup(h200):
    plan = factory.h200_leader_plan(h200)
    return lambda cand_fp: plan if cand_fp == plan.cand_fp else None


def _render(summary, out_dir, plan_lookup=None) -> tuple[str, str]:
    md_path, html_path = render_report(summary, out_dir, plan_lookup)
    assert md_path == out_dir / "report.zh-CN.md"
    assert html_path == out_dir / "report.zh-CN.html"
    return md_path.read_text(encoding="utf-8"), html_path.read_text(encoding="utf-8")


def _inline_svgs(page: str) -> list[str]:
    return re.findall(r"<svg\b.*?</svg>", page, re.S)


def _confirm_verdict(summary, **verdict):
    """The confirm point as AIPerf 0.14.0 exports it under --scenario agentx."""
    points = tuple(
        p.model_copy(
            update={
                "metrics": p.metrics.model_copy(
                    update={"scenario": "agentx", **verdict}
                )
            }
        )
        if p.key.purpose == "confirm"
        else p
        for p in summary.points
    )
    return summary.model_copy(update={"points": points})


def _unescaped(md: str) -> str:
    return md.replace("\\_", "_")


def test_method_section_states_aiperfs_agentx_verdict(tmp_path, h200):
    verdict = "AIPerf 对长时复测的场景判定"
    md, _ = _render(h200, tmp_path / "none")  # no metadata block: no verdict
    assert verdict not in md
    md, _ = _render(_confirm_verdict(h200, submission_valid=True), tmp_path / "ok")
    md = _unescaped(md)
    assert f"{verdict}（`agentx`）：submission_valid=true" in md
    assert "不能当作 AgentX 的正式提交结果" not in md  # e2e keeps the AgentX load
    bad = _confirm_verdict(
        h200,
        submission_valid=False,
        submission_invalid_reasons=("--unsafe-override was set",),
    )
    md, _ = _render(bad, tmp_path / "bad")
    md = _unescaped(md)
    assert f"{verdict}（`agentx`）：submission_valid=false" in md
    assert "--unsafe-override was set" in md
    d_only = _confirm_verdict(h200, submission_valid=True).model_copy(
        update={"stage": "decode"}
    )
    md, _ = _render(d_only, tmp_path / "d-only")
    assert "不能当作 AgentX 的正式提交结果" in md


def test_full_report_has_every_section_and_key_number(tmp_path, h200, plan_lookup):
    md, page = _render(h200, tmp_path / "report", plan_lookup)
    s = Strings.load("zh_CN")
    for key in SECTION_KEYS:
        heading = s.t(f"sections.{key}")
        assert f"## {heading}" in md
        assert heading in page
    for number in (
        "112.4",  # leader tok/s/GPU at saturation
        "110.3",  # >= 20 tok/s/user
        "88.1",  # >= 50 tok/s/user
        "+18.2%",  # MTP on vs off
        "-41.5%",  # 1 NIC vs 8
        "-0.9%",  # 4 NICs, within noise
        "1.86%",  # noise floor
        "362.5",  # inter-node NCCL speed of light
        "140.9",  # busiest NVLink peak under load
    ):
        assert number in md, number
        assert number in page, number
    assert "agg-tp8|mtp=1|nic=all|sim=0|rungs=-" in md
    assert "tp-size: 8" in md
    assert "Disagg 2P1D TP8 (cross-node, NIXL)" in md
    assert "error rate 3.2% \\> 1%" in md


def test_nic_section_explains_irrelevance_for_per_node_leader(
    tmp_path, h200, plan_lookup
):
    md, page = _render(h200, tmp_path, plan_lookup)
    assert "最优配置没有跨节点流量" in md
    assert "的每个副本都在单台节点内，没有跨节点流量" in page
    assert "疑似绕道 CPU 内存" in md
    assert "| 8 | 已测量 | 105.6 | 基准 | 基准 |" in md


def test_nic_section_for_cross_node_leader(tmp_path, h200, strings):
    leader_point = next(
        p
        for p in h200.points
        if p.candidate == factory.DIS
        and p.key.concurrency == 256
        and p.key.purpose == "search"
    )
    needs = interconnect_needs(
        leader_point,
        h200.nic_arms,
        factory.H200_PROFILE,
        sweep_family=h200.nic_family_id,
    )
    disagg_led = h200.model_copy(
        update={
            "leader": factory.DIS,
            "leader_point_id": leader_point.key.id,
            "interconnect": needs,
        }
    )
    md, _ = _render(disagg_led, tmp_path)
    assert strings.t("nic.leader_cross") in md
    assert "没有跨节点流量" not in md
    assert strings.t("config.no_plan", fp=factory.DIS.fp) in md
    # KV moves over IB: the traffic, not idleness, sizes the NIC count
    assert needs.link("ib").status == "ok" and needs.min_nics_estimated == 1
    assert needs.sweep_on_leader
    assert strings.t("resources.nic_est_idle") not in md
    assert "实测最少 4 张/节点" in md
    # the sweep ran on this leader: its traffic estimate can be compared
    assert strings.t("resources.nic_gap", estimated=1, measured=4) in md
    assert f"| **{strings.t('resources.kpi_nics')}** | 4 |" in md


def test_resources_section_reports_needs_mfu_and_kv(
    tmp_path, h200, plan_lookup, strings
):
    md, page = _render(h200, tmp_path, plan_lookup)
    sec = md.split("## 2 ", 1)[1].split("## 3 ", 1)[0]
    # 2.1: the agg leader has idle IB; the sweep's 4 NICs belong to disagg
    assert "| `node1/gpu3` | 101.20 | 140.90 | 450 | 22.5% | 144.57 | 有余量 |" in sec
    assert strings.t("resources.nic_est_idle") in sec
    assert "在有跨节点流量的 Disagg 1P1D" in sec and "它实测最少 4 张/节点" in sec
    assert "瓶颈很可能不在带宽总量" not in sec  # no cross-family comparison
    assert "这个部署需要 NVLink" in sec
    # 2.2: MFU of the one agg role, its curve, the dense FP8 peak
    assert "| 聚合（agg） | 32 | 210.6 | 10.6% | 31% |" in sec
    assert "稠密峰值 1,979 TFLOPS" in sec and "| 256 | 112.4 | 10.6% |" in sec
    assert strings.t("resources.mfu_total") not in sec  # single role
    # 2.3: 4 x 381k tokens; 24 requests at the p50 context; a near-full pool
    assert "可同时放下的上下文：152.4 万 tokens" in sec
    assert "约 24 / 8 个请求" in sec and strings.t("resources.kv_full") in sec
    # section 1 tiles: the leader's own NIC need, not the other family's
    est = strings.t("resources.kpi_nics_est")
    assert f"| **{est}** | 1 |" in md
    assert f"| **{strings.t('resources.kpi_nics')}** |" not in md
    assert "| **MFU（agg）** | 10.6% |" in md
    assert "152.4 万 tokens" in page


def test_nvlink_comparison_section_and_tile(tmp_path, h200, plan_lookup, strings):
    md, _ = _render(h200, tmp_path, plan_lookup)
    sec = md.split("### 2.2 ", 1)[1].split("### 2.3 ", 1)[0]
    assert sec.startswith("禁用 NVLink 改走 PCIe 的对比")
    assert "nvidia-smi nvlink -sBwMode OFF" in sec and "测完恢复为 FULL" in sec
    assert "只有 41.5 GB/s（56 对 GPU 能经 PCIe 直接 P2P；PCIe 单向上限 64 GB/s）" in sec
    assert "| NVLink（默认） | 112.4 | 基准 | 14.0 | 5,200 | 68.9 |" in sec
    assert "| PCIe P2P | 74.6 | -33.6% | 9.2 | 7,900 | 104.0 |" in sec
    assert strings.t("resources.nvl_needed", delta="-33.6%") in sec
    assert strings.t("resources.nvl_host_caveat") not in sec  # real P2P here
    assert "实测（见 2.2）：禁用 NVLink 后吞吐 -33.6%，确实需要 NVLink。" in md
    assert "| **禁用 NVLink（改走 PCIe）** | -33.6% | 112.4 → 74.6 tok/s/GPU |" in md
    # the off run is evidence, never a ranked configuration
    assert h200.leader == factory.AGG


def test_nvlink_section_explains_a_skip_and_a_failed_restore(tmp_path, h200, strings):
    skipped = h200.nvlink.model_copy(
        update={
            "status": "skipped",
            "off_point_id": None,
            "reason": "PCIe-only GPUs: there is no NVLink to switch off",
        }
    )
    md, _ = _render(h200.model_copy(update={"nvlink": skipped}), tmp_path / "a")
    assert "没有做这个对比：这款 GPU 只有 PCIe，没有 NVLink 可关。" in md
    assert "禁用 NVLink（改走 PCIe）" not in md  # no tile without a measurement
    broken = h200.nvlink.model_copy(update={"restored": False})
    md, _ = _render(h200.model_copy(update={"nvlink": broken}), tmp_path / "b")
    assert strings.t("resources.nvl_restore_failed") in md


def test_resources_section_without_data_says_so(tmp_path, h200, strings):
    lead = next(p for p in h200.points if p.key.id == h200.leader_point_id)
    bare = h200.model_copy(
        update={
            "interconnect": None,
            "kv_capacity": None,
            "model_arch": None,
            "points": tuple(
                p.model_copy(update={"mfu": None}) if p is lead else p
                for p in h200.points
            ),
        }
    )
    md, _ = _render(bare, tmp_path)
    sec = md.split("## 2 ", 1)[1].split("## 3 ", 1)[0]
    for key in ("no_bw", "no_mfu", "no_kv"):
        assert strings.t(f"resources.{key}") in sec
    assert strings.t("resources.kpi_kv") not in md


def test_html_is_self_contained_and_print_friendly(tmp_path, h200, plan_lookup):
    _, page = _render(h200, tmp_path, plan_lookup)
    assert page.startswith("<!DOCTYPE html>")
    assert '<html lang="zh-CN">' in page
    assert '<meta charset="utf-8">' in page
    for forbidden in ("http://", "https://", "<script", "<link", "@import", "url("):
        assert forbidden not in page, forbidden
    assert "@media print" in page
    assert "prefers-color-scheme: dark" in page
    # The etcd endpoint is shown, but with entity-encoded slashes.
    assert "http:&#47;&#47;10.42.0.1:2379" in page


def test_charts_are_well_formed_svg(tmp_path, h200, plan_lookup):
    out = tmp_path / "report"
    md, page = _render(h200, out, plan_lookup)
    inline = _inline_svgs(page)
    assert len(inline) == len(FIGURES)
    for chart in inline:
        root = ET.fromstring(chart)
        assert root.tag == "svg"
        assert root.find("title") is not None
    files = {p.name for p in (out / "figures").glob("*.svg")}
    assert files == FIGURES
    for name in sorted(files):
        root = ET.parse(out / "figures" / name).getroot()
        assert root.tag == f"{{{svg.SVG_NS}}}svg"
        assert f"](figures/{name})" in md


def test_secrets_are_masked(tmp_path, h200, plan_lookup):
    md, page = _render(h200, tmp_path, plan_lookup)
    assert factory.SECRET_ENV_VALUE not in md
    assert factory.SECRET_ENV_VALUE not in page
    assert "HF_TOKEN='***'" in md
    assert "--api-key '***'" in md


def test_render_is_deterministic(tmp_path, h200, plan_lookup):
    first = _render(h200, tmp_path / "a", plan_lookup)
    second = _render(h200, tmp_path / "b", plan_lookup)
    assert first == second
    for name in sorted(FIGURES):
        a = (tmp_path / "a" / "figures" / name).read_bytes()
        assert a == (tmp_path / "b" / "figures" / name).read_bytes()


def test_rerender_removes_stale_figures(tmp_path, h200, plan_lookup):
    _render(h200, tmp_path, plan_lookup)
    _render(factory.rtx6000_unsupported_summary(), tmp_path)
    assert not list((tmp_path / "figures").glob("*.svg"))


def test_unsupported_run_renders_explicit_text(tmp_path, strings):
    md, page = _render(factory.rtx6000_unsupported_summary(), tmp_path)
    for key in (
        "summary.no_leader",
        "topology.no_curves",
        "mtp.none",
        "bandwidth.no_leader",
        "config.none",
        "appendix.none",
    ):
        assert strings.t(key) in md, key
    assert strings.t("verdict.unsupported") in md
    assert "kernel 不支持：FlashMLA" in page  # localized deploy-failure reason
    assert strings.t("failure.kernel_unsupported") in md
    assert "41.2" in md  # PCIe speed of light still reported
    assert _inline_svgs(page) == []
    assert "http://" not in page and "<script" not in page


def test_dangling_references_are_none_safe(tmp_path, h200, strings):
    broken = h200.model_copy(
        update={
            "leader": None,
            "leader_point_id": "missing",
            "confirm_point_id": "gone",
            "noise": None,
            "saturated": None,
            "mtp": MtpComparison(on_point_id="nowhere"),
            "floors": (FloorResult(user_tps_floor=20.0, note="curve never crossed"),),
            "nic_arms": (NicArm(nic_count=8, status="skipped"),),
            "nic_family_id": None,
            "nic_concurrency": None,
        }
    )
    md, page = _render(broken, tmp_path, plan_lookup=lambda fp: None)
    assert strings.t("summary.no_leader") in md
    assert strings.t("method.no_noise") in md
    assert strings.t("config.none") in md
    assert "curve never crossed" in md
    for chart in _inline_svgs(page):
        ET.fromstring(chart)


def test_nic_reference_requires_the_full_arm(tmp_path, h200):
    arms = tuple(
        a.model_copy(update={"tps_per_gpu": None, "status": "skipped"})
        if a.nic_count == 8
        else a.model_copy(update={"delta_vs_full_pct": None, "within_noise": None})
        for a in h200.nic_arms
    )
    md, page = _render(h200.model_copy(update={"nic_arms": arms}), tmp_path)
    nic = md.split("## 8 ", 1)[1].split("## 9 ", 1)[0]
    assert "| 基准 |" not in nic  # no reference row without a full-NIC number
    assert "| 8 | 跳过 | — |" in md
    assert "| 4 | 已测量 | 104.6 | — |" in md
    for chart in _inline_svgs(page):
        ET.fromstring(chart)


def test_bare_summary_is_none_safe(tmp_path, strings):
    bare = factory.rtx6000_unsupported_summary().model_copy(
        update={
            "inventory": Inventory(nodes=()),
            "families": (),
            "deploys": (),
            "peaks": (),
            "limitations": (),
            "unverified": (),
            "nic_reason": "",
            "verdict": "failed",
            "started_at": 0.0,
            "ended_at": 0.0,
        }
    )
    md, page = _render(bare, tmp_path)
    assert strings.t("env.no_inventory") in md
    assert strings.t("bandwidth.no_peaks") in md
    assert strings.t("verdict.failed") in page


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def test_nice_ticks_and_labels():
    assert svg.nice_ticks(0.0, 121.4) == (0.0, 25.0, 50.0, 75.0, 100.0, 125.0)
    ticks = svg.nice_ticks(0.0, 1.08)
    assert ticks[-1] >= 1.08
    assert svg.tick_label(0.25, ticks) == "0.25"
    assert svg.tick_label(1000.0, (0.0, 500.0, 1000.0)) == "1,000"


def test_line_chart_log2_axis_is_deterministic():
    series = [
        svg.Series("agg", (svg.Point(8, 21.3, "a"), svg.Point(1024, 110.0, "b", "最佳"))),
        svg.Series("disagg", (svg.Point(16, 30.0), svg.Point(512, 104.0))),
    ]
    kwargs = {"title": "t", "x_label": "并发", "y_label": "tok/s/GPU", "log2_x": True}
    chart = svg.line_chart("c1", series, **kwargs)
    assert chart == svg.line_chart("c1", series, **kwargs)
    root = ET.fromstring(chart)
    labels = {t.text for t in root.iter("text")}
    assert {"8", "16", "32", "64", "128", "256", "512", "1024"} <= labels
    assert {"agg", "disagg", "最佳"} <= labels
    assert len(root.findall(".//path[@class='ln s1']")) == 1


def test_charts_handle_empty_input():
    for chart in (
        svg.line_chart("e1", [], title="t", x_label="x", y_label="y", empty_text="无"),
        svg.hbar_chart("e2", [], title="t", x_label="x", empty_text="无"),
        svg.grouped_bar_chart("e3", [], [], [], title="t", y_label="y"),
        svg.grouped_bar_chart("e4", ["g"], ["a"], [[None]], title="t", y_label="y"),
    ):
        ET.fromstring(chart)


def test_chart_text_is_escaped():
    chart = svg.hbar_chart(
        "x",
        [svg.Bar("<a&b>", 3.0, "3 (http://x)", tip="'q'"), svg.Bar("none", None, "无")],
        title="t",
        x_label="x",
    )
    root = ET.fromstring(chart)
    assert "<a&b>" in {t.text for t in root.iter("text")}
    assert "http://" not in chart


def test_standalone_svg_has_namespace():
    chart = svg.grouped_bar_chart(
        "g", ["A\nB"], ["on", "off"], [[2.0, 1.0]], title="t", y_label="y"
    )
    root = ET.fromstring(svg.standalone(chart))
    assert root.tag == f"{{{svg.SVG_NS}}}svg"


def test_strings_fail_fast_and_labels_fall_back(strings):
    with pytest.raises(KeyError):
        strings.t("no.such.key")
    assert strings.label("status", "brand_new_status") == "brand_new_status"
    rich = strings.r("summary.mtp", delta=strong("+1.0%"), noise="噪声内", accept="2.00")
    assert Span("+1.0%", "strong") in rich


def test_markdown_and_html_escaping():
    table = md_table(("a", "b"), [("x|y", "`z`")])
    assert "x\\|y" in table
    assert "\\`z\\`" in table
    assert esc("ETCD=http://10.0.0.1:2379 <b>") == (
        "ETCD=http:&#47;&#47;10.0.0.1:2379 &lt;b&gt;"
    )


def test_pd_leader_headlines_the_whole_cluster_mfu(tmp_path, h200, strings):
    from benchmarks.cluster_bench.domain.modelarch import compute_mfu

    pd = compute_mfu(
        flops_prefill=8e18,
        flops_decode=2e17,
        window_s=480.0,
        role_gpus={"prefill": 16, "decode": 16},
        precision="fp8",
        peak_tflops=1979.0,
        cache_source="usage",
    )
    points = tuple(
        p.model_copy(update={"mfu": pd}) if p.key.id == h200.leader_point_id else p
        for p in h200.points
    )
    md, _ = _render(h200.model_copy(update={"points": points}), tmp_path)
    total = f"{pd.total_mfu * 100:.1f}%"
    tile = f"| **{strings.t('resources.kpi_mfu_total')}** | {total} | prefill "
    assert tile in md and " · decode " in md
    assert "| **MFU（decode）** |" not in md
