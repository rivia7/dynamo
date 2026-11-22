# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event store durability, AIPerf command/parse, inventory helpers."""

from __future__ import annotations

import json

import pytest

from benchmarks.lium_cluster_bench.adapters import aiperf
from benchmarks.lium_cluster_bench.adapters.executors import Redactor
from benchmarks.lium_cluster_bench.app.simulate import synthetic_inventory
from benchmarks.lium_cluster_bench.app.store import (
    EventStore,
    RunLock,
    SealMismatch,
    StoreCorruption,
    seal_dir,
    verify_seal,
)
from benchmarks.lium_cluster_bench.domain.inventory import (
    check_uniform,
    fingerprint,
    pick_model_root,
)
from benchmarks.lium_cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
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
        dataset="semianalysis_cc_traces_weka_062126_256k",
        spec=DEFAULT_BENCH_SPECS["screen"],
        concurrency=128,
        served_context=262144,
        artifact_dir="/x/aiperf",
        server_metrics_urls=["http://10.42.0.2:8081/metrics"],
        cpu_count=224,
        hf_home="/w/hf",
    )
    assert "--scenario inferencex-agentx-mvp" in cmd
    assert "--public-dataset semianalysis_cc_traces_weka_062126_256k" in cmd
    assert "--unsafe-override" in cmd
    assert "--agentic-cache-warmup-duration 180" in cmd
    assert "--warmup-requests-per-lane" not in cmd  # absent in 0.13.0
    assert f"--max-context-length {262144 - aiperf.CONTEXT_MARGIN_TOKENS}" in cmd
    assert "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1" in cmd
    decision = aiperf.build_profile_command(
        aiperf_bin="a", served_model="m", tokenizer="t", url="u", dataset="d",
        spec=DEFAULT_BENCH_SPECS["decision"], concurrency=4, served_context=500000,
        artifact_dir="x", server_metrics_urls=[], cpu_count=8, hf_home="h",
    )  # fmt: skip
    assert "--unsafe-override" not in decision and "--server-metrics " not in decision
    assert aiperf.missing_flags(" ".join(aiperf.REQUIRED_FLAGS)) == []
    assert aiperf.missing_flags("--scenario") != []


def test_parse_profile_export_and_window(tmp_path):
    report = {
        "aiperf_version": "0.13.0",
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


def test_model_root_prefers_local_disk_and_never_mnt():
    inv = synthetic_inventory("h200")
    assert pick_model_root(inv.nodes, 800 * 10**9) == "/workspace"
    with pytest.raises(ValueError):
        pick_model_root(inv.nodes, 10**15)
