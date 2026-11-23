# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""--dataset kinds and the AIPerf command each one gets (P-Only included)."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from typing import Any

import pytest

from benchmarks.cluster_bench.adapters import aiperf
from benchmarks.cluster_bench.app.phases.download import prefetch_dataset_command
from benchmarks.cluster_bench.domain.datasets import (
    DEFAULT_DATASET,
    DatasetError,
    parse_dataset,
)
from benchmarks.cluster_bench.domain.schema import DEFAULT_BENCH_SPECS

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

COMMON: dict[str, Any] = dict(
    aiperf_bin="/v/bin/aiperf",
    served_model="zai-org/GLM-5.3",
    tokenizer="/models/glm",
    url="http://127.0.0.1:8000",
    spec=DEFAULT_BENCH_SPECS["screen"],
    concurrency=64,
    served_context=131072,
    artifact_dir="/x/aiperf",
    server_metrics_urls=["http://10.0.0.12:8081/metrics"],
    cpu_count=64,
    hf_home="/w/hf",
)


def test_hf_endpoint_reaches_aiperf_and_the_dataset_prefetch():
    dataset = parse_dataset(DEFAULT_DATASET)
    mirror = "https://hf-mirror.com"
    cmd = aiperf.build_profile_command(dataset=dataset, hf_endpoint=mirror, **COMMON)
    assert f"HF_ENDPOINT={mirror}" in cmd
    assert "HF_ENDPOINT" not in aiperf.build_profile_command(dataset=dataset, **COMMON)
    prefetch = prefetch_dataset_command(
        "org/ds",
        python="/v/bin/python",
        hf_home="/w/hf",
        log_dir="/w/exp/download",
        hf_endpoint=mirror,
    )
    lexer = shlex.shlex(prefetch, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    argv = list(lexer)
    assert f"HF_ENDPOINT={mirror}" in argv and "HF_HOME=/w/hf" in argv
    assert "repo_id='org/ds'" in argv[argv.index("-c") + 1]
    plain = prefetch_dataset_command("org/ds", python="p", hf_home="h", log_dir="/l")
    assert "HF_ENDPOINT" not in plain


def test_dataset_prefetch_returns_while_the_download_runs(tmp_path):
    # Stand-ins: a "python" that downloads for 3 s, and setsid (absent on macOS).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("setsid", 'exec "$@"'), ("python", "sleep 3")):
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    cmd = prefetch_dataset_command(
        "org/ds",
        python=str(bin_dir / "python"),
        hf_home=str(tmp_path / "hf"),
        log_dir=str(tmp_path / "download"),
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    started = time.monotonic()
    subprocess.run(["bash", "-c", cmd], capture_output=True, env=env, check=True)
    assert time.monotonic() - started < 2, "the caller waited for the download"


def argv_of(dataset: str, **kwargs: Any) -> list[str]:
    return shlex.split(
        aiperf.build_profile_command(
            dataset=parse_dataset(dataset), **{**COMMON, **kwargs}
        )
    )


def value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def extra_inputs(argv: list[str]) -> list[str]:
    out = []
    for tok in argv[argv.index("--extra-inputs") + 1 :]:
        if tok.startswith("--"):
            break
        out.append(tok)
    return out


@pytest.mark.parametrize(
    "text, kind, label",
    [
        (DEFAULT_DATASET, "agentx", DEFAULT_DATASET),
        ("sharegpt", "public", "sharegpt"),
        (
            "mooncake_trace:/root/in/t.jsonl",
            "file",
            "mooncake_trace file /root/in/t.jsonl",
        ),
        ("file:/root/in/t.jsonl", "file", "mooncake_trace file /root/in/t.jsonl"),
        ("weka_trace:/root/in/cc.jsonl", "agentx", "weka_trace file /root/in/cc.jsonl"),
        ("synthetic:isl=8192,osl=1024", "synthetic", "synthetic ISL 8192 / OSL 1024"),
    ],
)
def test_parse_dataset(text: str, kind: str, label: str) -> None:
    ds = parse_dataset(text)
    assert (ds.kind, ds.label, ds.text) == (kind, label, text)


@pytest.mark.parametrize(
    "text, message",
    [
        ("  ", "empty --dataset"),
        ("synthetic:isl=8192", "isl >= 1 and osl >= 1"),
        ("synthetic:isl=8k,osl=1", "must be an integer"),
        ("synthetic:foo=1", "expected synthetic:"),
        ("synthetic:isl=10,osl=10,isl_std=-1", "must be >= 0"),
        ("mooncake_trace:relative.jsonl", "absolute path"),
        ("Bad-Name", "not an AIPerf public dataset name"),
        ("BAD:/x", "invalid dataset type"),
    ],
)
def test_parse_dataset_rejects(text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        parse_dataset(text)


@pytest.mark.parametrize(
    "dataset",
    [
        DEFAULT_DATASET,
        "sharegpt",
        "mooncake_trace:/root/in/t.jsonl",
        "synthetic:isl=8192,osl=1024",
    ],
)
def test_every_required_flag_is_on_the_command(dataset: str) -> None:
    # Warmup and override flags depend on the spec: each one is on the command
    # of at least one spec.
    passed = {
        flag
        for spec in DEFAULT_BENCH_SPECS.values()
        for flag in argv_of(dataset, spec=spec)
    }
    assert [
        f for f in aiperf.required_flags(parse_dataset(dataset)) if f not in passed
    ] == []


def test_warmup_follows_the_spec_and_the_dataset() -> None:
    agentx, plain = parse_dataset(DEFAULT_DATASET), parse_dataset("sharegpt")
    specs = DEFAULT_BENCH_SPECS
    assert aiperf.warmup_args(agentx, specs["decision"]) == []  # preset: 10/lane
    assert aiperf.warmup_args(agentx, specs["smoke"]) == [
        "--warmup-requests-per-lane",
        "1",
    ]
    assert aiperf.warmup_args(agentx, specs["screen"]) == [
        "--agentic-cache-warmup-duration",
        "180",
    ]
    assert aiperf.warmup_args(plain, specs["decision"]) == ["--warmup-duration", "300"]
    assert aiperf.warmup_args(plain, specs["smoke"]) == []
    # Budget and process timeout plan the request warmup, not warmup_s.
    assert specs["decision"].warmup_estimate_s == 600.0
    assert specs["screen"].warmup_estimate_s == 180.0
    grace = aiperf.AGENTX_WARMUP_GRACE_S
    assert aiperf.run_timeout_s(specs["decision"]) == 1800 + grace + 1200 + 600
    assert aiperf.run_timeout_s(specs["screen"]) == 1800 + 180 + 480 + 600


def test_confirm_run_meets_the_agentx_duration_floor() -> None:
    decision = DEFAULT_BENCH_SPECS["decision"]
    assert decision.duration_s >= aiperf.AGENTX_MIN_DURATION_S
    assert not decision.unsafe_override
    assert all(
        spec.unsafe_override
        for spec in DEFAULT_BENCH_SPECS.values()
        if spec.duration_s < aiperf.AGENTX_MIN_DURATION_S
    )


def test_default_dataset_keeps_the_agentx_flag_set() -> None:
    assert (
        aiperf.required_flags(parse_dataset(DEFAULT_DATASET)) == aiperf.REQUIRED_FLAGS
    )


def test_plain_datasets_run_closed_loop_without_the_agentx_scenario() -> None:
    public = argv_of("sharegpt")
    assert value(public, "--public-dataset") == "sharegpt"
    assert (
        "--scenario" not in public and "--agentic-cache-warmup-duration" not in public
    )
    assert "--max-context-length" not in public and "--unsafe-override" not in public
    assert value(public, "--warmup-duration") == str(
        DEFAULT_BENCH_SPECS["screen"].warmup_s
    )
    trace = argv_of("mooncake_trace:/root/in/t.jsonl")
    assert value(trace, "--input-file") == "/root/in/t.jsonl"
    assert value(trace, "--custom-dataset-type") == "mooncake_trace"
    assert "--no-fixed-schedule" in trace and "--public-dataset" not in trace
    synthetic = argv_of("synthetic:isl=8192,osl=1024,isl_std=100")
    assert value(synthetic, "--synthetic-input-tokens-mean") == "8192"
    assert value(synthetic, "--synthetic-input-tokens-stddev") == "100"
    assert value(synthetic, "--output-tokens-mean") == "1024"
    assert "--public-dataset" not in synthetic and "--input-file" not in synthetic


def test_p_only_caps_every_request_at_one_output_token() -> None:
    assert extra_inputs(argv_of("sharegpt")) == ["ignore_eos:true"]
    assert extra_inputs(argv_of("sharegpt", stage="prefill")) == [
        "ignore_eos:true",
        "max_completion_tokens:1",
    ]
    assert extra_inputs(argv_of(DEFAULT_DATASET, stage="decode")) == ["ignore_eos:true"]


def test_aiperf_args_are_appended_but_may_not_override_the_load() -> None:
    argv = argv_of("sharegpt", extra_args=["--request-rate-ramp-duration", "30"])
    assert argv[-2:] == ["--request-rate-ramp-duration", "30"]
    aiperf.check_extra_args(["--goodput", "ttft:2000"])
    for bad in (
        ["--concurrency", "8"],
        ["--osl=10"],
        ["--model", "x"],
        ["--public-dataset", "y"],
    ):
        with pytest.raises(ValueError, match="is set by the harness"):
            aiperf.check_extra_args(bad)


def test_parse_profile_export_reads_input_throughput() -> None:
    metrics = aiperf.parse_profile_export(
        {
            "aiperf_version": "0.14.0",
            "request_count": {"avg": 100},
            "output_token_throughput": {"avg": 50.0},
            "input_token_throughput": {"avg": 320000.0},
            "benchmark_duration": {"avg": 300.0},
        }
    )
    assert metrics.input_tps == 320000.0 and metrics.output_tps == 50.0
