# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""adapters/accuracy.py: command rendering, timeouts, fail-soft runs through a
recording fake cluster, and the dry-run wiring (``simulated_stdout``)."""

from __future__ import annotations

import io
import json
import shlex
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pytest

from benchmarks.cluster_bench.adapters import accuracy as adapter
from benchmarks.cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.cluster_bench.agents import accuracy_probe as agent
from benchmarks.cluster_bench.domain import accuracy as acc
from benchmarks.cluster_bench.ports import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "accuracy"
NODES = ("node0", "node1")
WORKDIR = "/root/cluster_bench"
PYTHON = f"{WORKDIR}/venv/bin/python"
AGENT = adapter.agent_path(f"{WORKDIR}/src")
CACHE = adapter.dataset_cache_dir(WORKDIR)
DATASET = f"{CACHE}/gsm8k-test.jsonl"
URL = "http://127.0.0.1:8000"
OUT = f"{WORKDIR}/experiments/exp1/deployments/abc123/accuracy-gsm8k.json"
PARAMS = acc.AccuracyParams(questions=120, concurrency=16, deadline_s=600.0)


@dataclass
class TimedRecorder(RecordingExecutor):
    """RecordingExecutor that also keeps each command's timeout."""

    timeouts: list[tuple[str, str, float | None]] = field(default_factory=list)

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        self.timeouts.append((node, command, timeout))
        return super().run(
            node, command, input_text=input_text, timeout=timeout, check=check
        )


class ExplodingCluster:
    """A cluster whose transport raises (ssh died mid-run)."""

    nodes = NODES

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        raise OSError("ssh: connect to host 10.42.0.1 port 22: Connection refused")

    def fanout(self, nodes: Sequence[str], fn):
        return {node: fn(node) for node in nodes}


def result(node: str, command: str, rc: int = 0, out: str = "", err: str = ""):
    return CommandResult(node, command, rc, out, err)


def probe_line() -> str:
    return json.dumps(json.loads((FIXTURES / "probe_output.json").read_text())) + "\n"


def run_probe(cluster, **overrides) -> acc.AccuracyResult:
    kwargs = dict(
        python=PYTHON,
        agent=AGENT,
        url=URL,
        model="glm-5.3",
        dataset=DATASET,
        out=OUT,
        params=PARAMS,
    )
    kwargs.update(overrides)
    return adapter.run_accuracy(cluster, "node0", **kwargs)


def fetch(cluster, **overrides) -> acc.DatasetStatus:
    kwargs = dict(python=PYTHON, agent=AGENT, cache_dir=CACHE)
    kwargs.update(overrides)
    return adapter.ensure_dataset(cluster, "node0", **kwargs)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_paths_and_command_builders() -> None:
    assert AGENT == f"{WORKDIR}/src/benchmarks/cluster_bench/agents/accuracy_probe.py"
    assert CACHE == f"{WORKDIR}/datasets"
    assert adapter.dataset_path(CACHE + "/") == DATASET
    assert shlex.split(adapter.fetch_command(PYTHON, AGENT, dest=DATASET)) == [
        PYTHON,
        AGENT,
        "fetch",
        "--url",
        acc.GSM8K_URL,
        "--sha256",
        acc.GSM8K_SHA256,
        "--dest",
        DATASET,
        "--attempts",
        "3",
        "--timeout-s",
        "60",
    ]
    command = adapter.probe_command(
        PYTHON,
        AGENT,
        url=URL,
        model="zai-org/GLM-5.3 FP8",
        dataset=DATASET,
        out=OUT,
        params=PARAMS,
    )
    assert shlex.split(command) == [
        PYTHON,
        AGENT,
        "gsm8k",
        "--url",
        URL,
        "--model",
        "zai-org/GLM-5.3 FP8",
        "--dataset",
        DATASET,
        "--shots",
        "5",
        "--questions",
        "120",
        "--concurrency",
        "16",
        "--max-tokens",
        "512",
        "--request-timeout-s",
        "120",
        "--deadline-s",
        "600",
        "--retries",
        "2",
        "--out",
        OUT,
        "--expect-sha256",
        acc.GSM8K_SHA256,
    ]
    unchecked = adapter.probe_command(
        PYTHON,
        AGENT,
        url=URL,
        model="m",
        dataset=DATASET,
        out=OUT,
        params=PARAMS,
        expected_sha256=None,
    )
    assert "--expect-sha256" not in unchecked


def test_rendered_commands_parse_with_the_agent_cli() -> None:
    """Every flag the adapter renders is one the agent accepts."""
    for command in (
        adapter.fetch_command(PYTHON, AGENT, dest=DATASET, urls=("u1", "u2")),
        adapter.probe_command(
            PYTHON, AGENT, url=URL, model="m", dataset=DATASET, out=OUT, params=PARAMS
        ),
    ):
        args = agent.build_parser().parse_args(shlex.split(command)[2:])
        assert args.mode in ("fetch", "gsm8k")
    probe = agent.build_parser().parse_args(
        shlex.split(
            adapter.probe_command(
                PYTHON,
                AGENT,
                url=URL,
                model="m",
                dataset=DATASET,
                out=OUT,
                params=PARAMS,
            )
        )[2:]
    )
    assert (probe.questions, probe.concurrency, probe.deadline_s) == (120, 16, 600.0)
    assert probe.expect_sha256 == acc.GSM8K_SHA256


def test_timeouts_bound_the_agent() -> None:
    assert adapter.probe_timeout_s(PARAMS) == 600.0 + adapter.PROBE_MARGIN_S
    worst = adapter.fetch_timeout_s(urls=1, attempts=3, attempt_timeout_s=60.0)
    # 3 attempts x (curl + urllib) x (60 s + 10 s grace) + 2 s + 4 s backoff.
    assert worst == 3 * 2 * 70.0 + 6.0 + adapter.FETCH_MARGIN_S


# --------------------------------------------------------------------------
# ensure_dataset
# --------------------------------------------------------------------------


def test_ensure_dataset_returns_the_verified_copy() -> None:
    doc = acc.synthetic_fetch_document(DATASET, reused=True)
    rec = TimedRecorder(
        node_names=NODES, responder=lambda n, c: result(n, c, out=json.dumps(doc))
    )
    status = fetch(rec)
    assert status.ok and status.reused and status.path == DATASET
    [(node, command, timeout)] = rec.timeouts
    assert node == "node0" and command == adapter.fetch_command(
        PYTHON, AGENT, dest=DATASET
    )
    assert timeout == adapter.fetch_timeout_s(urls=1, attempts=3)


@pytest.mark.parametrize(
    ("rc", "out", "err", "message"),
    [
        (1, "", "Traceback ...\nPermissionError: [Errno 13]", "exit 1"),
        (0, "bash: line 1: python: not found\n", "", "no dataset_fetch JSON"),
        (124, "", "\n[timeout]", "[timeout]"),
    ],
)
def test_ensure_dataset_is_fail_soft(rc: int, out: str, err: str, message: str) -> None:
    rec = RecordingExecutor(
        node_names=NODES, responder=lambda n, c: result(n, c, rc, out, err)
    )
    status = fetch(rec)
    assert not status.ok and message in (status.error or "")
    assert status.path == DATASET and status.expected_sha256 == acc.GSM8K_SHA256


def test_ensure_dataset_reports_agent_failures_and_transport_errors() -> None:
    failed = {
        **acc.synthetic_fetch_document(DATASET),
        "ok": False,
        "sha256": None,
        "error": "no verified copy after 3 attempt(s) from 1 URL(s)",
    }
    rec = RecordingExecutor(
        node_names=NODES, responder=lambda n, c: result(n, c, out=json.dumps(failed))
    )
    status = fetch(rec, attempts=1, urls=(acc.GSM8K_URL, "https://mirror/x.jsonl"))
    assert not status.ok and "no verified copy" in (status.error or "")
    assert rec.commands[0].command.count("--url") == 2
    exploded = fetch(ExplodingCluster())
    assert not exploded.ok and "Connection refused" in (exploded.error or "")


# --------------------------------------------------------------------------
# run_accuracy
# --------------------------------------------------------------------------


def test_run_accuracy_parses_the_agent_document() -> None:
    rec = TimedRecorder(
        node_names=NODES,
        responder=lambda n, c: result(n, c, out="warming up\n" + probe_line()),
    )
    res = run_probe(rec)
    assert res.status == "ok" and (res.n, res.correct) == (10, 8)
    assert res.params == PARAMS  # the adapter's request, not the agent's echo
    assert [t for _, _, t in rec.timeouts] == [adapter.probe_timeout_s(PARAMS)]
    assert rec.commands_for("node0") == [
        adapter.probe_command(
            PYTHON,
            AGENT,
            url=URL,
            model="glm-5.3",
            dataset=DATASET,
            out=OUT,
            params=PARAMS,
        )
    ]
    assert acc.accuracy_verdict(res, floor=0.5).status == "pass"


def test_run_accuracy_reads_the_out_file_when_stdout_is_empty() -> None:
    def responder(node: str, command: str) -> CommandResult:
        if command.startswith("cat "):
            return result(node, command, out=probe_line())
        return result(node, command, out="")

    rec = TimedRecorder(node_names=NODES, responder=responder)
    res = run_probe(rec)
    assert res.status == "ok" and res.n == 10
    assert rec.commands_for("node0")[1] == f"cat {OUT}"
    assert rec.timeouts[1][2] == adapter.READ_TIMEOUT_S


def test_run_accuracy_keeps_a_document_from_a_nonzero_exit() -> None:
    rec = RecordingExecutor(
        node_names=NODES,
        responder=lambda n, c: result(n, c, 1, probe_line(), "Segmentation fault"),
    )
    res = run_probe(rec)
    assert res.status == "ok" and res.n == 10
    assert res.notes[-1].source == "collector"
    assert "Segmentation fault" in res.notes[-1].error


@pytest.mark.parametrize(
    ("rc", "out", "err", "cat", "message"),
    [
        (124, "", "\n[timeout]", "", "exit 124"),
        (0, "", "", "cat: no such file", "no accuracy_probe JSON document"),
        (2, "", "usage: accuracy_probe.py ...", "", "exit 2"),
    ],
)
def test_run_accuracy_is_fail_soft(
    rc: int, out: str, err: str, cat: str, message: str
) -> None:
    def responder(node: str, command: str) -> CommandResult:
        if command.startswith("cat "):
            return result(node, command, 1, "", cat)
        return result(node, command, rc, out, err)

    rec = RecordingExecutor(node_names=NODES, responder=responder)
    res = run_probe(rec)
    assert res.status == "error" and message in (res.reason or "")
    assert res.params == PARAMS and res.endpoint == URL and res.model == "glm-5.3"
    verdict = acc.accuracy_verdict(res, floor=acc.DEFAULT_FLOOR)
    assert verdict.status == "error" and verdict.code == "probe_error"


def test_run_accuracy_survives_transport_errors() -> None:
    res = run_probe(ExplodingCluster())
    assert res.status == "error" and "Connection refused" in (res.reason or "")


def test_run_accuracy_reports_agent_errors() -> None:
    doc = json.loads(probe_line())
    doc.update(error="dataset sha256 abc does not match the expected def", n=0)
    rec = RecordingExecutor(
        node_names=NODES, responder=lambda n, c: result(n, c, out=json.dumps(doc))
    )
    res = run_probe(rec)
    assert res.status == "error" and "does not match" in (res.reason or "")


def test_agent_output_flows_through_the_adapter(tmp_path: Path) -> None:
    """Contract: what the real agent prints is what the adapter parses."""
    rows = [json.loads(line) for line in (FIXTURES / "gsm8k_synthetic.jsonl").open()]

    def post(url: str, body: bytes, timeout: float) -> tuple[int, str]:
        prompt = json.loads(body)["prompt"]
        asked = prompt.rsplit("Question: ", 1)[1][: -len("\nAnswer:")]
        gold = next(r["answer"] for r in rows if r["question"] == asked)
        text = " Let me compute.\n" + gold.rsplit("####", 1)[0] + "#### 1\n"
        payload = {"choices": [{"index": 0, "text": text, "finish_reason": "stop"}]}
        return 200, json.dumps(payload)

    printed = io.StringIO()
    with redirect_stdout(printed):
        code = agent.main(
            [
                "gsm8k",
                "--url",
                URL,
                "--model",
                "glm-5.3",
                "--dataset",
                str(FIXTURES / "gsm8k_synthetic.jsonl"),
                "--out",
                str(tmp_path / "out.json"),
            ],
            host=agent.Host(post=post),
        )
    assert code == 0
    rec = RecordingExecutor(
        node_names=NODES, responder=lambda n, c: result(n, c, out=printed.getvalue())
    )
    res = run_probe(rec)
    assert res.status == "ok" and (res.questions, res.n) == (11, 11)
    assert res.correct == 0 and res.invalid == 0  # every answer says "#### 1"
    verdict = acc.accuracy_verdict(res, floor=acc.DEFAULT_FLOOR)
    assert verdict.status == "fail" and verdict.code == "below_floor"


# --------------------------------------------------------------------------
# Dry-run wiring
# --------------------------------------------------------------------------


def test_simulated_stdout_drives_both_steps() -> None:
    def responder(node: str, command: str) -> CommandResult | None:
        out = adapter.simulated_stdout(command)
        return None if out is None else result(node, command, out=out)

    rec = RecordingExecutor(node_names=NODES, responder=responder)
    status = fetch(rec)
    assert status.ok and status.path == DATASET
    res = run_probe(rec, dataset=status.path)
    assert res.status == "ok" and res.questions == 120 and res.model == "glm-5.3"
    assert res.endpoint == URL + "/v1/completions"
    assert res.correct == round(acc.SYNTH_ACCURACY * 120)
    assert acc.accuracy_verdict(res, floor=acc.DEFAULT_FLOOR).passed
    low = adapter.simulated_stdout(rec.commands[-1].command, accuracy=0.2)
    assert low is not None and acc.parse_probe_output(low).accuracy == pytest.approx(
        0.2, abs=0.01
    )
    assert adapter.simulated_stdout("echo hi") is None
    assert adapter.simulated_stdout(f"{PYTHON} {AGENT}") is None
    assert adapter.simulated_stdout(f"{PYTHON} {AGENT} other --x 1") is None
