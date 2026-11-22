# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the incremental knowledge-base export."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from benchmarks.lium_cluster_bench.kb import views
from benchmarks.lium_cluster_bench.kb.findings import derive_findings
from benchmarks.lium_cluster_bench.kb.jsonl import (
    KbError,
    KbLockTimeout,
    canonical_line,
    merge_rows,
    read_jsonl,
)
from benchmarks.lium_cluster_bench.kb.records import FindingRecord, build_records, rec_id_of
from benchmarks.lium_cluster_bench.kb.writer import export_experiment
from benchmarks.lium_cluster_bench.kb.writer import main as kb_main
from benchmarks.lium_cluster_bench.report.render import main as render_main
from tests.benchmarks.lium_cluster_bench import summary_factory as factory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

SPDX_FIRST_LINES = (
    "<!--\nSPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\nSPDX-License-Identifier: Apache-2.0\n-->\n"
)
CLUSTER_VIEWS = ("cluster.md", "speed-of-light.md", "topology-ranking.md")


@pytest.fixture
def h200():
    return factory.h200_summary()


@pytest.fixture
def plan_lookup(h200):
    plan = factory.h200_leader_plan(h200)
    return lambda cand_fp: plan if cand_fp == plan.cand_fp else None


@pytest.fixture
def kb(tmp_path) -> Path:
    return tmp_path / "kb"


def _second_h200():
    return factory.h200_summary(
        exp_id="exp-h200-002", scale=1.03, started_at=factory.T0 + 3 * 86400
    )


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _generated(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split(views.BEGIN, 1)[1].split(views.END, 1)[0]


def test_export_writes_records_and_views(tmp_path, kb, h200, plan_lookup):
    result = export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    assert result.conflicts == ()
    assert result.appended["run"] == len(h200.points)
    assert result.appended["peak"] == len(h200.peaks)
    assert result.appended["nic"] == len(h200.nic_arms)
    assert result.appended["deploy"] == len(h200.deploys)
    assert result.appended["config"] == 1
    assert result.appended["experiment"] == 1
    records = kb / "clusters" / h200.fingerprint / "records"
    runs = read_jsonl(records / "run.jsonl")
    assert len(runs) == len(h200.points)
    for row in runs:
        assert row["schema"] == "cluster_bench.run/v1"
        assert row["fp"] == h200.fingerprint and row["exp_id"] == h200.exp_id
        assert len(row["rec_id"]) == 16 and row["rec_id"] == rec_id_of(row)
    utils = read_jsonl(records / "util.jsonl")
    usable = sum(len(p.bandwidth) for p in h200.points if p.audit.usable)
    assert len(utils) == usable
    assert any(u["leader"] and u["plane"] == "nvlink" for u in utils)
    peaks = read_jsonl(records / "peak.jsonl")
    assert {p["evidence_category"] for p in peaks} == {5}
    index = read_jsonl(kb / "index.jsonl")
    assert [row["exp_id"] for row in index] == [h200.exp_id]
    assert index[0]["leader_tps_per_gpu"] == factory.LEADER_TPS
    exp = index[0]  # the experiment record
    assert (exp["min_nics_measured"], exp["min_nics_family"]) == (4, "disagg-1p1d")
    assert exp["min_nics_estimated"] == 1 and exp["nvlink_needed"] is True
    assert exp["kv_total_tokens"] == 4 * 381_000 and exp["kv_requests_at_p50"] == 24
    assert exp["leader_mfu_total"] == pytest.approx(0.106, abs=1e-3)
    assert exp["mfu_precision"] == "fp8" and exp["peak_tflops_per_gpu"] == 1979.0
    resources = _generated(kb / "clusters" / h200.fingerprint / "resources.md")
    assert "| 4 (on disagg-1p1d) | 1 |" in resources
    assert "| 1.52M | 262K |" in resources and "| 10.6% |" in resources
    assert (exp["nvlink_off_method"], exp["nvlink_off_path"]) == ("bw_mode_off", "pcie")
    assert (
        exp["nvlink_off_delta_pct"] == -33.63
        and exp["nvlink_off_within_noise"] is False
    )
    assert (
        "| agg-tp8 | 256 | bw\\_mode\\_off | pcie | 112.4 | 74.6 | -33.6% |"
        in resources
    )
    runs = read_jsonl(records / "run.jsonl")
    assert {r["p2p"] for r in runs} == {"nvlink", "pcie"}


def test_every_markdown_file_has_spdx_and_title(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    markdown = sorted(kb.rglob("*.md"))
    assert len(markdown) == 1 + 6 + 2  # INDEX + cluster views + final/
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        assert text.startswith(SPDX_FIRST_LINES), path
        assert text[len(SPDX_FIRST_LINES) :].lstrip("\n").startswith("# "), path
        assert text.count(views.BEGIN) == 1 and text.count(views.END) == 1


def test_run_artifacts_layout(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    root = kb / "experiments" / h200.exp_id
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["exp_id"] == h200.exp_id
    assert manifest["fingerprint"] == h200.fingerprint
    assert manifest["leader"]["cand_key"] == factory.AGG.key
    findings = read_jsonl(root / "analysis" / "performance_findings.jsonl")
    assert findings and {f["schema"] for f in findings} == {FindingRecord.SCHEMA}
    recommended = (root / "final" / "recommended_config.md").read_text(encoding="utf-8")
    assert views.CORRECTNESS_LINE in recommended.splitlines()
    assert views.CORRECTNESS_LINE == (
        "Correctness status: unverified (waived: throughput benchmark; smoke "
        "completions only)"
    )
    assert "tp-size: 8" in recommended
    assert factory.SECRET_ENV_VALUE not in recommended
    limitations = (root / "final" / "known_limitations.md").read_text(encoding="utf-8")
    assert "UNVERIFIED: DeepEP low_latency mode" in limitations
    assert "Condition: AIPerf client co-located on node0" in limitations
    script = root / "final" / "reproduced_commands.sh"
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n# SPDX-FileCopyrightText")
    assert "-m benchmarks.lium_cluster_bench run" in text
    assert "--exp-id exp-h200-001" in text
    assert "--gpu-profile h200" in text and "--served-context 262144" in text
    assert "--dataset" not in text  # not a CLI flag
    assert "-m benchmarks.lium_cluster_bench kb" in text
    assert os.access(script, os.X_OK)


def test_reexport_is_a_noop(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    before = _snapshot(kb)
    again = export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    assert again.files_written == ()
    assert not again.changed
    assert sum(again.appended.values()) == 0
    assert again.duplicates["run"] == len(h200.points)
    assert again.conflicts == ()
    assert _snapshot(kb) == before


def test_second_experiment_of_same_cluster_merges(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp1", kb, plan_lookup)
    second = _second_h200()
    result = export_experiment(second, tmp_path / "exp2", kb, lambda fp: None)
    assert result.conflicts == ()
    assert result.appended["run"] == len(second.points)
    cluster = kb / "clusters" / h200.fingerprint
    assert len(read_jsonl(cluster / "records" / "run.jsonl")) == 2 * len(h200.points)
    assert [r["exp_id"] for r in read_jsonl(kb / "index.jsonl")] == [
        h200.exp_id,
        second.exp_id,
    ]
    index = _generated(kb / "INDEX.md")
    assert h200.exp_id in index and second.exp_id in index
    assert "| 2 |" in index  # two experiments on this cluster
    assert "None: every snapshot matches the latest." in _generated(
        cluster / "cluster.md"
    )
    sol = _generated(cluster / "speed-of-light.md")
    assert "| ib | `gdr_write_bw` | 1 | 48.3 | 48.3 | 50.0 | 96.6% | 2 | 2 |" in sol
    ranking = _generated(cluster / "topology-ranking.md").splitlines()
    first = next(line for line in ranking if line.startswith("| 1 |"))
    assert "`agg-tp8`" in first and f"`{second.exp_id}`" in first
    assert any(line.startswith("| `disagg-2p1d` | 2 | `oom` x2 |") for line in ranking)
    findings = _generated(cluster / "knob-findings.md")
    assert f"exp `{h200.exp_id}`" in findings and f"exp `{second.exp_id}`" in findings
    assert (kb / "experiments" / second.exp_id / "manifest.yaml").is_file()


def test_rankings_average_repetitions(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    ranking = _generated(kb / "clusters" / h200.fingerprint / "topology-ranking.md")
    best = next(line for line in ranking.splitlines() if line.startswith("| 1 |"))
    # 3 pilot reps + 1 confirm at C=256 decision spec average to 112.65; the
    # single best repetition (113.6) must not be reported as the family best.
    assert "| 112.7 | 256 | `decision` |" in best
    assert "113.6" not in best


def test_changed_reexport_goes_to_conflicts(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    limitations = kb / "experiments" / h200.exp_id / "final" / "known_limitations.md"
    before = limitations.read_bytes()
    points = list(h200.points)
    original = points[0]
    points[0] = original.model_copy(update={"tps_per_gpu": 99.9})
    changed = h200.model_copy(update={"points": tuple(points), "dropped_traces": 9})
    result = export_experiment(changed, tmp_path / "exp", kb, plan_lookup)
    assert limitations.read_bytes() == before  # experiment folder kept as first
    run_conflicts = [c for c in result.conflicts if c.kind == "run"]
    assert len(run_conflicts) == 1
    assert run_conflicts[0].reason == "key_exists"
    records = kb / "clusters" / h200.fingerprint / "records"
    rows = {r["point_id"]: r for r in read_jsonl(records / "run.jsonl")}
    assert len(rows) == len(h200.points)
    assert rows[original.key.id]["tps_per_gpu"] == original.tps_per_gpu
    conflicts = read_jsonl(records / "conflicts.jsonl")
    assert any(c["incoming"].get("tps_per_gpu") == 99.9 for c in conflicts)
    # Re-exporting the same conflicting summary does not duplicate conflicts.
    again = export_experiment(changed, tmp_path / "exp", kb, plan_lookup)
    assert len(read_jsonl(records / "conflicts.jsonl")) == len(conflicts)
    assert records / "conflicts.jsonl" not in again.files_written


def test_edited_line_is_a_content_mismatch(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    path = kb / "clusters" / h200.fingerprint / "records" / "peak.jsonl"
    rows = read_jsonl(path)
    rows[0]["value_gbs"] = 1.0  # hand edit that keeps the old rec_id
    path.write_text("".join(canonical_line(r) + "\n" for r in rows), encoding="utf-8")
    result = export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    assert [(c.kind, c.reason) for c in result.conflicts] == [
        ("peak", "content_mismatch")
    ]
    assert read_jsonl(path)[0]["value_gbs"] == 1.0  # existing lines never rewritten


def test_markers_preserve_hand_written_text(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp1", kb, plan_lookup)
    path = kb / "clusters" / h200.fingerprint / "knob-findings.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(views.BEGIN, "Operator note above.\n\n" + views.BEGIN)
    path.write_text(text + "\nOperator note below.\n", encoding="utf-8")
    second = _second_h200()
    export_experiment(second, tmp_path / "exp2", kb, lambda fp: None)
    merged = path.read_text(encoding="utf-8")
    assert "Operator note above." in merged and "Operator note below." in merged
    assert merged.index("Operator note above.") < merged.index(views.BEGIN)
    assert merged.index(views.END) < merged.index("Operator note below.")
    assert f"exp `{second.exp_id}`" in _generated(path)


def test_file_without_markers_gets_a_generated_block(tmp_path, kb, h200, plan_lookup):
    path = kb / "clusters" / h200.fingerprint / "cluster.md"
    path.parent.mkdir(parents=True)
    manual = f"{SPDX_FIRST_LINES}\n# Cluster notes\n\nRack 7, rented via lium.\n"
    path.write_text(manual, encoding="utf-8")
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    text = path.read_text(encoding="utf-8")
    assert text.startswith(manual.rstrip("\n"))
    assert "8 x 400 Gb/s InfiniBand" in _generated(path)


def test_malformed_markers_fail_before_any_write(tmp_path, kb, h200, plan_lookup):
    kb.mkdir()
    (kb / "INDEX.md").write_text(
        f"# KB\n{views.BEGIN}\n{views.BEGIN}\n{views.END}\n", encoding="utf-8"
    )
    with pytest.raises(KbError):
        export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    assert not (kb / "clusters").exists()
    assert not (kb / "index.jsonl").exists()


def test_lock_contention_times_out(tmp_path, kb, h200, plan_lookup):
    kb.mkdir()
    with open(kb / ".lock", "a+", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        with pytest.raises(KbLockTimeout):
            export_experiment(h200, tmp_path / "exp", kb, plan_lookup, lock_timeout_s=0)
    assert not (kb / "index.jsonl").exists()


def test_plans_are_read_from_the_experiment_dir(tmp_path, kb, h200):
    exp_dir = tmp_path / "exp"
    plan = factory.h200_leader_plan(h200)
    plan_file = exp_dir / "deployments" / plan.cand_fp / "plan.json"
    plan_file.parent.mkdir(parents=True)
    plan_file.write_text(plan.model_dump_json(), encoding="utf-8")
    result = export_experiment(h200, exp_dir, kb)
    assert result.appended["config"] == 1
    config = read_jsonl(kb / "clusters" / h200.fingerprint / "records" / "config.jsonl")
    assert config[0]["cand_fp"] == plan.cand_fp
    assert config[0]["roles"][0]["env"]["HF_TOKEN"] == "***"


def test_cli_entry_points_read_the_experiment_dir(tmp_path, kb, h200):
    exp_dir = tmp_path / "exp"
    summary_file = exp_dir / "analysis" / "summary.json"
    summary_file.parent.mkdir(parents=True)
    summary_file.write_text(h200.model_dump_json(indent=2), encoding="utf-8")
    plan = factory.h200_leader_plan(h200)
    plan_file = exp_dir / "deployments" / plan.cand_fp / "plan.json"
    plan_file.parent.mkdir(parents=True)
    plan_file.write_text(plan.model_dump_json(), encoding="utf-8")
    assert render_main(["--exp-dir", str(exp_dir)]) == 0
    report = (exp_dir / "report" / "report.zh-CN.md").read_text(encoding="utf-8")
    assert "tp-size: 8" in report
    assert kb_main(["--exp-dir", str(exp_dir), "--kb-root", str(kb)]) == 0
    manifest = yaml.safe_load(
        (kb / "experiments" / h200.exp_id / "manifest.yaml").read_text()
    )
    digest = hashlib.sha256(summary_file.read_bytes()).hexdigest()
    assert manifest["source"]["summary_sha256"] == digest


def test_unsupported_run_exports(tmp_path, kb):
    rtx = factory.rtx6000_unsupported_summary()
    result = export_experiment(rtx, tmp_path / "exp", kb, lambda fp: None)
    assert result.appended["run"] == 0 and result.appended["peak"] == 2
    root = kb / "experiments" / rtx.exp_id
    recommended = (root / "final" / "recommended_config.md").read_text(encoding="utf-8")
    assert views.CORRECTNESS_LINE in recommended.splitlines()
    assert "No recommendation" in recommended
    ranking = _generated(kb / "clusters" / rtx.fingerprint / "topology-ranking.md")
    assert "No valid runs recorded yet." in ranking
    assert "`kernel_unsupported` x1" in ranking
    assert yaml.safe_load((root / "manifest.yaml").read_text())["leader"] is None


def test_exp_id_of_another_cluster_is_refused(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    clash = factory.rtx6000_unsupported_summary(exp_id=h200.exp_id)
    with pytest.raises(KbError):
        export_experiment(clash, tmp_path / "exp2", kb, lambda fp: None)
    assert not (kb / "clusters" / clash.fingerprint).exists()


def test_unsafe_identifiers_are_rejected(tmp_path, kb, h200):
    with pytest.raises(ValueError):
        export_experiment(
            h200.model_copy(update={"exp_id": "../escape"}), tmp_path, kb, None
        )


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def test_findings_cover_every_kind_with_valid_categories(h200, plan_lookup):
    findings = derive_findings(h200, plan_lookup)
    assert {f.kind for f in findings} == {
        "topology",
        "user_floor",
        "saturation",
        "noise",
        "nic_irrelevant",
        "bandwidth",
        "mtp",
        "nic_count",
        "deploy",
    }
    for finding in findings:
        assert 1 <= finding.evidence_category <= 5
        assert finding.rec_id == rec_id_of(finding.as_json())
    nic = {f.key.rsplit(":", 1)[1]: f for f in findings if f.kind == "nic_count"}
    assert sorted(nic) == ["1", "2", "4"]
    assert nic["2"].symptom.startswith("NIC count reduced to 2 of 8 per node")
    assert "host staging" in nic["1"].caveat
    assert nic["4"].delta_pct == pytest.approx(-0.947, abs=1e-3)
    assert all(f.evidence_category == 5 for f in findings if f.kind == "bandwidth")
    topology = next(f for f in findings if f.kind == "topology")
    assert topology.table == "scenario"
    assert topology.delta_pct == pytest.approx(7.15, abs=0.01)


def test_records_are_deterministic(h200, plan_lookup):
    first = build_records(h200, plan_lookup, derive_findings(h200, plan_lookup))
    second = build_records(h200, plan_lookup, derive_findings(h200, plan_lookup))
    assert first == second
    keys = [r.key for rows in first.by_kind.values() for r in rows]
    assert len(keys) == len(set(keys))


def test_knob_findings_use_agent_docs_schemas(tmp_path, kb, h200, plan_lookup):
    export_experiment(h200, tmp_path / "exp", kb, plan_lookup)
    text = _generated(kb / "clusters" / h200.fingerprint / "knob-findings.md")
    assert (
        "| Symptom | Candidate knobs | Expected metric effect | Validation | " in text
    )
    assert (
        "| Scenario | Knobs / config surface | Evidence signal | What to try " in text
    )
    assert "NIC count reduced to 1 of 8 per node (`disagg-1p1d`, C=256)" in text


def test_read_jsonl_drops_torn_tail_and_rejects_corruption(tmp_path):
    path = tmp_path / "x.jsonl"
    good = canonical_line({"rec_id": "a", "key": "k1"}) + "\n"
    path.write_text(good + good.replace('"a"', '"b"') + '{"rec_id": "c", "ke')
    assert [r["rec_id"] for r in read_jsonl(path)] == ["a", "b"]
    assert path.read_text().endswith("\n")
    path.write_text(good + "not json\n" + good)
    with pytest.raises(KbError):
        read_jsonl(path)


def test_merge_rows_semantics():
    old = {"rec_id": "r1", "key": "k1", "v": 1}
    outcome = merge_rows(
        [old],
        [
            dict(old),  # duplicate
            {"rec_id": "r1", "key": "k1", "v": 2},  # same id, edited content
            {"rec_id": "r2", "key": "k1", "v": 3},  # same key, new content
            {"rec_id": "r3", "key": "k3", "v": 4},  # new
        ],
        "run",
    )
    assert outcome.duplicates == 1
    assert [c["reason"] for c in outcome.conflicts] == [
        "content_mismatch",
        "key_exists",
    ]
    assert [r["rec_id"] for r in outcome.appended] == ["r3"]
    assert json.loads(canonical_line(outcome.conflicts[0]))["existing"] == old
