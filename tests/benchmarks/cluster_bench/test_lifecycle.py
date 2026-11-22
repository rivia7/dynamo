# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/lifecycle.py (container cold start and KV capacity)."""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import pytest

from benchmarks.cluster_bench.domain.lifecycle import (
    EVENT_PATTERNS,
    DeployLifecycle,
    deploy_lifecycle,
    parse_container_log,
    parse_log_lines,
)
from tests.benchmarks.cluster_bench.lium_fakes import fixture_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "lifecycle"


def utc(*fields: int) -> float:
    return float(calendar.timegm((*fields, 0, 0, 0)))


STARTED = utc(2026, 11, 5, 9, 11, 50)


def log(name: str) -> str:
    return fixture_text(FIXTURES / name)


def test_decode_leader_in_dynamo_format() -> None:
    c = parse_container_log(
        log("decode_r0_dynamo_fmt.txt"),
        name="cb-x-decode0-r0",
        role="decode",
        node="node2",
        started_at=STARTED,
    )
    assert c.first_log_t == pytest.approx(utc(2026, 11, 5, 9, 12, 3) + 0.418224)
    # "end" events take the slowest TP rank's line, "begin" the first.
    assert c.weights_loaded_t == pytest.approx(utc(2026, 11, 5, 9, 14, 38) + 0.950442)
    assert c.weights_load_s == 133.83  # SGLang's own elapsed=, slowest rank
    assert (c.checkpoint_shards, c.shard_load_s) == (141, 131.0)
    assert c.cuda_graph_capture_s == 56.76
    assert (c.ready_source, c.ready_line) == (
        "marker",
        "Successfully registered LLM with runtime config",
    )
    assert c.to_ready_s == pytest.approx(239.134)
    assert c.deploy_to_ready_s == pytest.approx(252.552)
    assert c.post_graph_to_ready_s == pytest.approx(24.531)
    assert (c.model_type, c.dtype) == ("GlmMoeDsaForCausalLM", "torch.bfloat16")
    assert c.kv_cache_dtype == "torch.float8_e4m3fn"
    # Tightest GPU: lowest available, highest used.
    assert (c.weight_mem_gb, c.avail_mem_after_load_gb) == (89.33, 49.27)
    assert (c.kv_tokens, c.kv_cache_gb) == (731264, 23.71)  # MLA "KV size"
    assert (c.cuda_graph_mem_gb, c.avail_mem_after_cuda_graph_gb) == (1.64, 22.65)
    assert (c.max_total_num_tokens, c.max_running_requests) == (731264, 128)
    assert (c.chunked_prefill_size, c.max_prefill_tokens) == (16384, 16384)
    assert (c.context_len, c.available_gpu_mem_gb) == (262144, 22.65)
    assert c.warnings == 2 and c.errors == 0 and not c.naive_utc
    assert c.issues["nccl_warn"].count == 1
    assert c.issues["kv_transfer"].first == (
        "KV transfer for bootstrap_room=8813721 timed out after 30 s; retrying",
    )


def test_non_leader_rank_in_sglang_format_is_ready_at_its_scheduler() -> None:
    c = parse_container_log(
        log("decode_r1_sglang_fmt.txt"), name="r1", role="decode", started_at=STARTED
    )
    assert c.naive_utc  # "[2026-11-05 09:12:20 DP8 TP8 EP8]" read as UTC
    assert c.ranks_seen == 2  # DP8.. and DP9.. tags
    assert c.ready_source == "scheduler" and c.ready_line is None
    assert c.ready_t == utc(2026, 11, 5, 9, 15, 53)  # the last scheduler line
    assert c.deploy_to_ready_s == 243.0
    assert c.weights_load_s == 147.0  # no elapsed=: first begin to last end
    assert c.available_gpu_mem_gb == 22.55


def test_docker_timestamps_mha_pool_and_transfer_errors() -> None:
    c = parse_container_log(log("prefill_mha_docker_ts.txt"), name="p0", role="prefill")
    # `docker logs --timestamps` times win over the inner SGLang seconds.
    assert c.first_log_t == pytest.approx(utc(2026, 11, 5, 9, 12, 2) + 0.99355124)
    assert c.kv_cache_gb == 36.88  # K 18.44 + V 18.44 GB
    assert c.kv_tokens == 402816 and c.max_running_requests == 64
    assert c.cuda_graph_capture_s is None and c.cuda_graph_t is None
    assert c.ready_source == "marker" and c.deploy_to_ready_s is None
    assert c.errors == 1  # the ERROR-level KVTransferError line
    assert c.issues["traceback"].count == 1
    assert c.issues["kv_transfer"].count == 4
    assert "rdma" not in c.issues


def test_frontend_ready_when_chat_completions_ready() -> None:
    c = parse_container_log(log("frontend.txt"), name="fe", role="frontend")
    assert c.ready_line == "Chat completions is ready"
    assert c.ready_t == pytest.approx(utc(2026, 11, 5, 9, 16, 4) + 0.012003)
    assert c.weights_loaded_t is None and c.warnings == 1


def test_oom_during_cuda_graph_capture() -> None:
    c = parse_container_log(log("agg_oom.txt"), name="a0", role="agg")
    assert c.ready_source == "none" and c.ready_t is None
    assert c.kv_tokens == 61440 and c.avail_mem_after_pool_gb == 3.95
    assert c.cuda_graph_t is None and c.max_total_num_tokens is None
    assert set(c.issues) == {"cuda_oom", "traceback", "xid"}
    assert c.issues["cuda_oom"].first[0].startswith("torch.OutOfMemoryError")
    assert c.errors == 2


def test_deploy_summary_per_role_and_slowest_container() -> None:
    containers = [
        ("cb-x-decode0-r0", "decode", "node2", log("decode_r0_dynamo_fmt.txt")),
        ("cb-x-decode0-r1", "decode", "node3", log("decode_r1_sglang_fmt.txt")),
        ("cb-x-prefill0-r0", "prefill", "node0", log("prefill_mha_docker_ts.txt")),
        ("cb-x-frontend", "frontend", "node0", log("frontend.txt")),
        ("cb-x-agg0-r0", "agg", "node1", log("agg_oom.txt")),
    ]
    deploy = deploy_lifecycle(containers, STARTED)
    assert list(deploy.roles) == ["frontend", "agg", "prefill", "decode"]
    decode = deploy.roles["decode"]
    assert (decode.containers, decode.ready) == (2, 2)
    assert decode.slowest == "cb-x-decode0-r0"
    assert decode.time_to_ready_s == pytest.approx(252.552)
    assert decode.weights_load_s_max == 147.0
    assert decode.kv_tokens_min == 731264 and decode.max_running_requests_min == 128
    assert decode.available_gpu_mem_gb_min == 22.55
    assert deploy.roles["prefill"].issues == {"kv_transfer": 4, "traceback": 1}
    assert deploy.roles["agg"].ready == 0
    # Readiness of the deployment is the frontend listing the model.
    assert deploy.slowest == "cb-x-frontend"
    assert deploy.time_to_ready_s == pytest.approx(254.012)
    assert deploy.issues["traceback"] == 2 and deploy.issues["cuda_oom"] == 1
    assert deploy.notes == ("cb-x-agg0-r0: no readiness line found",)
    assert DeployLifecycle.model_validate_json(deploy.model_dump_json()) == deploy


def test_without_started_at_times_count_from_first_log_line() -> None:
    deploy = deploy_lifecycle(
        [("r1", "decode", "node3", log("decode_r1_sglang_fmt.txt"))], None
    )
    assert deploy.time_to_ready_s == 213.0 and deploy.started_at is None


def test_log_line_formats_and_inherited_timestamps() -> None:
    record = {"time": "2026-11-05T10:00:00.250+01:00", "level": "warn", "message": "x"}
    text = "\n".join(
        [
            "\x1b[2m2026-11-05T09:00:00.500000Z\x1b[0m \x1b[32m INFO\x1b[0m mod: hello",
            "Traceback (most recent call last):",
            json.dumps(record),
            "2026-11-05 09:00:02,125 - INFO - plain python logging",
            "",
        ]
    )
    lines = parse_log_lines(text)
    assert [line.level for line in lines] == ["INFO", None, "WARN", None]
    assert lines[0].message == "hello"
    assert lines[1].t == lines[0].t == utc(2026, 11, 5, 9, 0, 0) + 0.5
    assert lines[2].t == utc(2026, 11, 5, 9, 0, 0) + 0.25  # +01:00 offset
    assert lines[3].naive and lines[3].message == "INFO - plain python logging"


def test_truncated_and_empty_logs() -> None:
    tail = "\n".join(
        [
            "[2026-11-05 09:14:50 TP0] Load weight end. type=X, dtype=torch.bfloat16, "
            "avail mem=49.22 GB, mem usage=89.39 GB.",
            "[2026-11-05 09:15:52 TP0] max_total_num_tokens=1000, "
            "chunked_prefill_size=-1, max_prefill_tokens=16384, "
            "max_running_requests=8, context_len=4096, available_gpu_mem=2.00 GB",
        ]
    )
    c = parse_container_log(tail, name="t", role="agg")
    assert c.chunked_prefill_size == -1
    assert any("log head missing" in note for note in c.notes)
    empty = parse_container_log("", name="e", role="agg")
    assert empty.lines == 0 and empty.first_log_t is None and empty.issues == {}
    untimed = parse_container_log("starting\nNCCL WARN x\n", name="u", role="agg")
    assert untimed.notes == ("no timestamps in the log: times unavailable",)
    assert untimed.issues["nccl_warn"].count == 1


def test_control_container_ready_markers() -> None:
    etcd = (
        '{"level":"info","ts":"2026-11-05T09:11:40.120Z","caller":"embed/serve.go:250",'
        '"msg":"ready to serve client requests"}'
    )
    c = parse_container_log(etcd, name="ctl", role="control")
    assert c.ready_line == "ready to serve client requests"
    assert c.ready_t == pytest.approx(utc(2026, 11, 5, 9, 11, 40) + 0.12)
    zap = parse_log_lines('{"ts": 1793869900.5, "msg": "Server is ready"}')
    assert zap[0].t == 1793869900.5 and zap[0].message == "Server is ready"


def test_sglang_0_5_21_startup_lines() -> None:
    text = "\n".join(
        [
            "[2026-11-22 10:00:00 TP0] server_args={'model_path': '/m', 'tp_size': 8}",
            "[2026-11-22 10:00:01 TP0] Init parallel begin.",
            "[2026-11-22 10:00:04 TP0] Init parallel ends. elapsed=3.10 s",
            "[2026-11-22 10:00:05 TP0] Load weight begin. avail mem=170.00 GB",
            "[2026-11-22 10:01:05 TP0] Load weight end. elapsed=60.00 s, "
            "type=GlmMoeDsaForCausalLM, avail mem=80.00 GB, mem usage=90.00 GB.",
            "[2026-11-22 10:01:06 TP0] KV Cache is allocated. dtype: "
            "torch.float8_e4m3fn, #tokens: 1000000, KV size: 40.00 GB",
            "[2026-11-22 10:01:07 TP0] Memory pool end. avail mem=38.00 GB",
            "[2026-11-22 10:01:08 TP0] Capture decode CUDA graph begin. "
            "backend=full, avail mem=38.00 GB",
            "[2026-11-22 10:01:20 TP0] Capture decode CUDA graph end. "
            "elapsed=12.00 s, mem usage=1.50 GB, avail mem=36.50 GB.",
            "[2026-11-22 10:01:21 TP0] max_total_num_tokens=1000000, "
            "chunked_prefill_size=16384, max_prefill_tokens=16384, "
            "max_running_requests=256, context_len=262144, "
            "available_gpu_mem=36.50 GB",
        ]
    )
    patterns = dict(EVENT_PATTERNS)
    messages = [line.message for line in parse_log_lines(text)]
    for event in ("server_args", "dist_init_begin", "dist_init_end"):
        assert any(patterns[event].search(m) for m in messages), event
    c = parse_container_log(text, name="a0", role="agg")
    assert c.weights_load_s == 60.0
    assert c.cuda_graph_capture_s == 12.0  # "elapsed=", not "Time elapsed:"
    assert (c.weight_mem_gb, c.avail_mem_after_load_gb) == (90.0, 80.0)
    assert (c.kv_tokens, c.kv_cache_gb) == (1000000, 40.0)
    assert (c.cuda_graph_mem_gb, c.avail_mem_after_cuda_graph_gb) == (1.5, 36.5)
    assert (c.max_total_num_tokens, c.max_running_requests) == (1000000, 256)
