# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report localization of planner, audit and analysis reason templates."""

from __future__ import annotations

import pytest

from benchmarks.cluster_bench.domain import analysis
from benchmarks.cluster_bench.domain.schema import DEFAULT_BENCH_SPECS, PointMetrics
from benchmarks.cluster_bench.report.reasons import localize

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("climb complete", "加并发结束"),
        ("grid max 512 reached while still rising", "到并发上限 512 时吞吐仍在上升"),
        (
            "alternate not selected (primary pd-2p2d: prefill_bound)",
            "没选这个备选：主拓扑 pd-2p2d 瓶颈在 prefill",
        ),
        ("still rising +4.5% at c=512", "C=512 时仍在上升（+4.5%）"),
        ("top two points within 3.2% (+1.0%)", "顶部两点相差 +1.0%，在 3.2% 的测量噪声以内"),
        ("kernel_unsupported: no kernel image", "kernel 不支持：no kernel image"),
        ("error rate 2.00% >= 1%", "错误率 2.00% ≥ 1%"),
        (
            "cb-1-agg0-r0: HiCache clamped 200 -> 164 GB/rank to fit host memory",
            "cb-1-agg0-r0：HiCache 由 200 收紧为 164 GB/rank 以适配主机内存",
        ),
        (
            "48 containers: HiCache clamped 200 -> 164 GB/rank to fit host memory",
            "48 个容器：HiCache 由 200 收紧为 164 GB/rank 以适配主机内存",
        ),
        ("3 containers: a new note", "3 个容器：a new note"),
        (
            "NVLink still carried 35.2 GB/s per GPU with NVLink off",
            "关掉 NVLink 后每块 GPU 仍有 35.2 GB/s 的 NVLink 流量",
        ),
        (
            "the leader wide-ep-dep32 needs NVLink (DeepEP intranode kernels); the "
            "comparison uses the best configuration without DeepEP (pd-4p2d)",
            "最优配置 wide-ep-dep32 依赖 NVLink（DeepEP 的机内 kernel 只走 NVLink），"
            "所以改测不用 DeepEP 的最好配置 pd-4p2d",
        ),
        (
            "nvidia-smi nvlink -sBwMode OFF failed on node1: Insufficient Permissions",
            "nvidia-smi nvlink -sBwMode OFF 在 node1 上执行失败：Insufficient Permissions",
        ),
        (
            "only the 8-NIC arm was measured",
            "只有 8 张网卡这一组可用，没有更少网卡的对比",
        ),
    ],
)
def test_known_templates(text, expected):
    assert localize(text) == expected


def test_unknown_text_passes_through():
    assert localize("something new happened") == "something new happened"


def test_audit_reasons_are_all_localized():
    bad = PointMetrics(
        aiperf_version="0.12.0",
        request_count=1,
        error_count=5,
        duration_s=10,
        was_cancelled=True,
        is_complete=False,
        incomplete_reason="cancelled at the benchmark deadline",
    )
    audit = analysis.audit_point(
        bad, spec=DEFAULT_BENCH_SPECS["screen"], concurrency=64, expected_aiperf_version="0.14.0",
        engagement=None, server_alive=False, telemetry_coverage=0.5, nic_restricted=True, nic_engaged=False,
    )  # fmt: skip
    assert any(r.startswith("AIPerf marked the run incomplete") for r in audit.reasons)
    for reason in audit.reasons:
        assert localize(reason) != reason, reason


def test_repeated_container_notes_collapse_to_one_line():
    clamp = "HiCache clamped 200 -> 164 GB/rank to fit host memory"
    notes = [
        f"cb-abc12608-agg0-r0: {clamp}",
        "noise floor not measured; a provisional 5% threshold was used",
        f"cb-abc12608-agg1-r0: {clamp}",
        f"cb-abc12608-agg0-r0: {clamp}",  # a redeploy renders it again
        f"cb-4594a056-prefill3-r1: {clamp}",
        "cb-4594a056-decode0-r0: HiCache disabled: 10 GB/rank host budget < 16 GB",
        "noise floor not measured; a provisional 5% threshold was used",
    ]
    assert analysis.collapse_container_notes(notes) == [
        f"3 containers: {clamp}",
        "noise floor not measured; a provisional 5% threshold was used",
        "cb-4594a056-decode0-r0: HiCache disabled: 10 GB/rank host budget < 16 GB",
    ]


def test_unverified_profile_notes_reach_the_summary_and_read_in_chinese():
    from benchmarks.cluster_bench.adapters.profile_store import load_gpu_profiles
    from benchmarks.cluster_bench.app.summary import profile_unverified

    profiles = load_gpu_profiles()
    rtx = profile_unverified(profiles["rtx_pro_6000"])
    assert any("peaks are half the product page figures" in s for s in rtx)
    assert profile_unverified(profiles["h200"]) == ()  # datasheet values
    for name in profiles:
        for sentence in profile_unverified(profiles[name]):
            assert localize(sentence) != sentence, sentence
