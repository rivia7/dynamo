# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why a point has no AIPerf summary export: abort reason and request errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.cluster_bench.adapters.aiperf import failure_details
from benchmarks.cluster_bench.domain import analysis
from benchmarks.cluster_bench.domain.schema import DEFAULT_BENCH_SPECS
from benchmarks.cluster_bench.report.reasons import localize

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

TOO_LONG = (
    "SSEResponseError('Error occurred in SSE response: Input length (248675 "
    "tokens) exceeds the maximum allowed length (248186 tokens).')"
)
# The tail of AIPerf's output when the AgentX warmup gives up (box drawn in ASCII).
EXIT_BOX = """\
01:47:26.761 NOTICE   Phase agentic.warmup (warmup) sending complete
+- AIPerf System Exit Errors -------------------------------+
|  * Service: timing_manager                                |
|    Reason: Run aborted by 'timing_manager': warmup_failure. |
"""


def point_dir(tmp_path: Path, log: str, records: list) -> Path:
    (tmp_path / "aiperf").mkdir()
    (tmp_path / "aiperf.log").write_text(log, encoding="utf-8")
    with open(tmp_path / "aiperf" / "profile_export.jsonl", "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return tmp_path


def test_an_aborted_run_names_the_abort_and_the_commonest_error(tmp_path):
    ok = {"metadata": {"turn_index": 1}, "metrics": {}}
    too_long = {"metadata": {}, "error": {"code": 400, "message": TOO_LONG}}
    busy = {"metadata": {}, "error": {"code": 503, "type": "Unavailable"}}
    point = point_dir(tmp_path, EXIT_BOX, [ok, too_long, busy, too_long, ok])
    assert failure_details(point) == (
        "AIPerf aborted the run (timing_manager: warmup_failure)",
        "3 failed request(s), most common (2x): HTTP 400: Input length (248675 "
        "tokens) exceeds the maximum allowed length (248186 tokens).",
    )


def test_long_error_texts_are_cut_and_missing_files_give_nothing(tmp_path):
    assert failure_details(tmp_path) == ()
    point = point_dir(
        tmp_path, "", [{"error": {"message": "x" * 500}}, ["not", "a", "record"]]
    )
    (detail,) = failure_details(point)
    assert detail.endswith("...") and len(detail) < 220


def test_the_audit_keeps_the_details_after_the_missing_export():
    aborted = "AIPerf aborted the run (timing_manager: warmup_failure)"
    audit = analysis.audit_point(
        None, spec=DEFAULT_BENCH_SPECS["screen"], concurrency=8, expected_aiperf_version="0.14.0",
        engagement=None, server_alive=True, telemetry_coverage=None, nic_restricted=False,
        nic_engaged=None, failure_details=(aborted,),
    )  # fmt: skip
    assert audit.status == "invalid"
    assert audit.reasons == ("no AIPerf export found", aborted)
    closed = f"no valid measurement ({'; '.join(audit.reasons)})"
    assert localize(closed) == (
        "没有有效的测量结果（找不到 AIPerf 的导出结果；AIPerf 中止了这次测量（timing_manager：warmup_failure））"
    )
    assert localize("2 failed request(s), most common (2x): HTTP 400: too long") == (
        "2 个请求失败，最常见的错误（2 次）：HTTP 400: too long"
    )
