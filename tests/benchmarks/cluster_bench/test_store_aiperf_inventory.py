# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event store durability, AIPerf command/parse, inventory helpers."""

from __future__ import annotations

import json
import shlex

import pytest

from benchmarks.cluster_bench.adapters import aiperf
from benchmarks.cluster_bench.adapters.executors import Redactor
from benchmarks.cluster_bench.app.simulate import synthetic_inventory
from benchmarks.cluster_bench.app.store import (
    EventStore,
    RunLock,
    SealMismatch,
    StoreCorruption,
    seal_dir,
    verify_seal,
)
from benchmarks.cluster_bench.domain.datasets import DEFAULT_DATASET
from benchmarks.cluster_bench.domain.inventory import (
    check_uniform,
    fingerprint,
    forbidden_weight_prefixes,
    pick_model_root,
    usable_weight_disks,
)
from benchmarks.cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
    DiskInfo,
    EventType,
    Inventory,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


# ------------------------------------------------------------------ store
def test_event_store_round_trip_and_torn_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    store = EventStore(path)
    store.append(EventType.NOTE, {"text": "a"})
    store.append(EventType.NOTE, {"text": "b"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "note", "ts": 1, "da')  # crash mid-write
    reloaded = EventStore(path)
    assert [e.data["text"] for e in reloaded.events] == ["a", "b"]
    assert path.read_text().endswith("\n")  # torn tail truncated
    reloaded.append(EventType.NOTE, {"text": "c"})
    assert len(EventStore(path).events) == 3


def test_event_store_rejects_mid_file_corruption(tmp_path):
    path = tmp_path / "events.jsonl"
    store = EventStore(path)
    store.append(EventType.NOTE, {"text": "a"})
    store.append(EventType.NOTE, {"text": "b"})
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"a"', '"x"')  # CRC no longer matches
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(StoreCorruption):
        EventStore(path)


def test_seal_detects_tampering(tmp_path):
    root = tmp_path / "point"
    root.mkdir()
    (root / "a.json").write_text("{}")
    seal_dir(root, {"id": 1})
    verify_seal(root)
    (root / "a.json").chmod(0o644)
    (root / "a.json").write_text('{"x": 1}')
    with pytest.raises(SealMismatch):
        verify_seal(root)


def test_run_lock_is_exclusive(tmp_path):
    with RunLock(tmp_path / ".lock"):
        with pytest.raises(RuntimeError):
            with RunLock(tmp_path / ".lock"):
                pass


def test_redactor_masks_secrets():
    redact = Redactor(["hf_abcdefSECRET"])
    assert redact("token=hf_abcdefSECRET end") == "token=*** end"


# ------------------------------------------------------------------ aiperf
def test_profile_command_has_agentx_shape():
    cmd = aiperf.build_profile_command(
        aiperf_bin="/v/bin/aiperf",
        served_model="zai-org/GLM-5.3",
        tokenizer="/models/glm",
        url="http://127.0.0.1:8000",
        dataset="semianalysis_cc_traces_weka_062126",
        spec=DEFAULT_BENCH_SPECS["screen"],
        concurrency=128,
        served_context=262144,
        artifact_dir="/x/aiperf",
        server_metrics_urls=["http://10.42.0.2:8081/metrics"],
        cpu_count=224,
        hf_home="/w/hf",
    )
    argv = shlex.split(cmd)
    assert argv[argv.index("--scenario") + 1] == "agentx"
    assert (
        argv[argv.index("--public-dataset") + 1] == "semianalysis_cc_traces_weka_062126"
    )
    assert "--unsafe-override" in argv  # 480 s screen < the 900 s AgentX floor
    # An explicit duration warmup replaces the preset's request budget.
    assert "--agentic-cache-warmup-duration 180" in cmd
    assert "--warmup-requests-per-lane" not in argv
    # Trajectory ratios and the schedule are the preset's.
    assert not [a for a in argv if a.startswith("--trajectory-start")]
    assert "--fixed-schedule" not in argv and "--no-fixed-schedule" not in argv
    assert argv[argv.index("--output-artifact-dir") + 1] == "/x/aiperf"
    assert f"--max-context-length {262144 - aiperf.CONTEXT_MARGIN_TOKENS}" in cmd
    assert "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1" in cmd
    common = dict(
        aiperf_bin="a", served_model="m", tokenizer="t", url="u", concurrency=4,
        served_context=500000, artifact_dir="x", server_metrics_urls=[],
        cpu_count=8, hf_home="h",
    )  # fmt: skip
    decision = shlex.split(
        aiperf.build_profile_command(
            dataset="semianalysis_cc_traces_weka_062126",
            spec=DEFAULT_BENCH_SPECS["decision"],
            **common,
        )
    )
    # The confirm run is a valid AgentX run: the preset's own warmup
    # (10 requests per lane) and an explicit duration above the 900 s floor.
    assert "--scenario" in decision and "--unsafe-override" not in decision
    assert "--agentic-cache-warmup-duration" not in decision
    assert "--warmup-requests-per-lane" not in decision
    assert decision[decision.index("--benchmark-duration") + 1] == "1200"
    assert "--server-metrics" not in decision
    smoke = aiperf.build_profile_command(
        dataset=DEFAULT_DATASET, spec=DEFAULT_BENCH_SPECS["smoke"], **common
    )
    assert "--warmup-requests-per-lane 1" in smoke and "--unsafe-override" in smoke
    local = aiperf.build_profile_command(
        dataset="weka_trace:/root/in/cc.jsonl",
        spec=DEFAULT_BENCH_SPECS["decision"],
        **common,
    )
    # A local trace file cannot pin the corpus: AIPerf needs the override.
    assert "--scenario agentx" in local and "--unsafe-override" in local
    plain = aiperf.build_profile_command(
        dataset="d", spec=DEFAULT_BENCH_SPECS["decision"], **common
    )
    assert "--scenario" not in plain and "--unsafe-override" not in plain
    assert aiperf.missing_flags(" ".join(aiperf.REQUIRED_FLAGS)) == []
    assert aiperf.missing_flags("--scenario") != []


def test_parse_profile_export_and_window(tmp_path):
    report = {
        "aiperf_version": "0.14.0",
        "was_cancelled": False,
        "request_count": {"avg": 990},
        "error_request_count": {"avg": 10},
        "output_token_throughput": {"avg": 25600.0},
        "output_token_throughput_per_user": {"p50": 41.5},
        "time_to_first_token": {"p50": 900.0, "p99": 4000.0},
        "inter_token_latency": {"p50": 22.0, "p99": 60.0},
        "benchmark_duration": {"avg": 481.0},
        "error_summary": [
            {"count": 10, "error_details": {"type": "ContextLengthExceeded"}}
        ],
    }
    run = tmp_path / "aiperf" / "run"
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.json").write_text(json.dumps(report))
    records = [
        {
            "metadata": {
                "benchmark_phase": "warmup",
                "request_start_ns": 1,
                "request_end_ns": 5,
            }
        },
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "request_start_ns": 100,
                "request_end_ns": 200,
            }
        },
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "request_start_ns": 150,
                "request_end_ns": 900,
            }
        },
    ]
    (run / "profile_export.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n{torn"
    )
    metrics = aiperf.load_point_metrics(tmp_path / "aiperf")
    assert metrics.output_tps == 25600.0
    assert metrics.error_rate == pytest.approx(0.01)
    assert metrics.error_types == {"ContextLengthExceeded": 10}
    assert (metrics.window_start_ns, metrics.window_end_ns) == (100, 900)
    # No --scenario, no metadata block: no verdict either way.
    assert metrics.scenario is None and metrics.submission_valid is None
    assert metrics.is_complete is None


def test_parse_profile_export_reads_the_scenario_verdict(tmp_path):
    report = {
        "aiperf_version": "0.14.0",
        "was_cancelled": False,
        "is_complete": False,
        "incomplete_reason": "duration elapsed with requests in flight",
        "request_count": {"avg": 400},
        "output_token_throughput": {"avg": 9000.0},
        "benchmark_duration": {"avg": 480.0},
        "metadata": {
            "scenario": "agentx",
            "submission_valid": False,
            "submission_invalid_reasons": ["--unsafe-override was set"],
        },
    }
    run = tmp_path / "aiperf" / "run"
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.json").write_text(json.dumps(report))
    metrics = aiperf.load_point_metrics(tmp_path / "aiperf")
    assert (metrics.scenario, metrics.submission_valid) == ("agentx", False)
    assert metrics.submission_invalid_reasons == ("--unsafe-override was set",)
    assert metrics.is_complete is False
    assert metrics.incomplete_reason == "duration elapsed with requests in flight"


# ------------------------------------------------------------------ inventory
def test_fingerprint_ignores_hostnames_and_addresses():
    a = synthetic_inventory("h200")
    renamed = Inventory(
        nodes=tuple(
            n.model_copy(update={"hostname": f"other-{i}", "address": f"10.0.0.{i}"})
            for i, n in enumerate(a.nodes)
        )
    )
    assert fingerprint(a) == fingerprint(renamed)
    assert fingerprint(a) != fingerprint(synthetic_inventory("b200"))
    assert check_uniform(a) == []


def test_model_root_prefers_local_disk_and_never_mnt_on_lium():
    inv = synthetic_inventory("h200")
    assert pick_model_root(inv.nodes, 800 * 10**9) == "/workspace"
    assert pick_model_root(inv.nodes, 800 * 10**9, platform="lium") == "/workspace"
    with pytest.raises(ValueError):
        pick_model_root(inv.nodes, 10**15)


def test_mnt_may_hold_weights_except_on_lium():
    small_root = DiskInfo(
        path="/", device="/dev/sda1", fs_type="ext4",
        total_bytes=200 * 10**9, free_bytes=100 * 10**9,
    )  # fmt: skip
    mnt = DiskInfo(
        path="/mnt/nvme", device="/dev/nvme1n1", fs_type="xfs",
        total_bytes=8 * 10**12, free_bytes=7 * 10**12,
    )  # fmt: skip
    nodes = [
        n.model_copy(update={"disks": (small_root, mnt)})
        for n in synthetic_inventory("h200").nodes
    ]
    assert "/mnt" in forbidden_weight_prefixes("lium")
    assert "/mnt" not in forbidden_weight_prefixes("generic")
    assert [d.path for d in usable_weight_disks(nodes[0], "lium")] == ["/"]
    assert pick_model_root(nodes, 800 * 10**9) == "/mnt/nvme"
    with pytest.raises(ValueError):
        pick_model_root(nodes, 800 * 10**9, platform="lium")
