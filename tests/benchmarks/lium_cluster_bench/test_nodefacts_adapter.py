# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adapters/nodefacts.py: command rendering, fail-soft
collection through a recording fake cluster, raw evidence and the dry-run
wiring (``simulated_stdout``)."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pytest

from benchmarks.lium_cluster_bench.adapters import nodefacts as adapter
from benchmarks.lium_cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.lium_cluster_bench.domain import nodefacts as nf
from benchmarks.lium_cluster_bench.ports import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "nodefacts"
NODES = ("node0", "node1", "node2", "node3")
WORKDIR = "/root/cluster_bench"
PYTHON = f"{WORKDIR}/venv/bin/python"
SRC = f"{WORKDIR}/src"
AGENT = f"{SRC}/benchmarks/lium_cluster_bench/agents/node_facts.py"
IMAGE = "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
CUDA_BANNER = (
    "\n==========\n== CUDA ==\n==========\n\nCUDA Version 13.0.1\n\n"
    "Container image Copyright (c) 2016-2025, NVIDIA CORPORATION & AFFILIATES.\n"
)


@dataclass
class TimedRecorder(RecordingExecutor):
    """RecordingExecutor that also keeps each command's timeout."""

    timeouts: list[tuple[str, str, float | None]] = field(default_factory=list)

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        self.timeouts.append((node, command, timeout))
        return super().run(
            node, command, input_text=input_text, timeout=timeout, check=check
        )


def compact(name: str) -> str:
    return json.dumps(json.loads((FIXTURES / name).read_text())) + "\n"


def result(node: str, command: str, rc: int = 0, out: str = "", err: str = ""):
    return CommandResult(node, command, rc, out, err)


def collector(
    rec: RecordingExecutor, tmp_path: Path, now: Callable[[], float] = lambda: 0.0
) -> adapter.FactCollector:
    return adapter.FactCollector(
        rec,
        image=IMAGE,
        workdir=WORKDIR,
        results_dir=tmp_path / "facts",
        now=now,
        labels={"cb.exp": "exp1"},
    )


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_command_builders() -> None:
    assert adapter.host_facts_command(PYTHON, SRC, "node1") == (
        f"{PYTHON} {AGENT} host --node node1 --deadline-s 150"
    )
    disk = adapter.disk_read_command(
        PYTHON, SRC + "/", "node2", "/models/GLM 5.3", max_bytes=1 << 30, max_seconds=60
    )
    assert shlex.split(disk) == [
        PYTHON,
        AGENT,
        "disk-read",
        "--node",
        "node2",
        "--path",
        "/models/GLM 5.3",
        "--max-bytes",
        str(1 << 30),
        "--block-mib",
        "16",
        "--max-seconds",
        "60",
    ]
    assert shlex.split(adapter.clock_command(PYTHON)) == [
        PYTHON,
        "-I",
        "-S",
        "-c",
        adapter.CLOCK_SNIPPET,
    ]


def test_software_argv() -> None:
    argv = adapter.software_argv(
        name="cb-facts-sw-node0",
        image=IMAGE,
        src_dir=SRC,
        node="node0",
        labels={"cb.role": "facts"},
    )
    assert argv[:3] == ["docker", "run", "--rm"] and "-d" not in argv
    pairs = set(zip(argv, argv[1:]))
    for pair in (
        ("--name", "cb-facts-sw-node0"),
        ("--network", "host"),
        ("--user", "0:0"),
        ("--gpus", '"device=0"'),
        ("--device", "/dev/infiniband"),
        ("-v", f"{SRC}:/bench/src:ro"),
        ("--label", "cb.role=facts"),
        ("--ulimit", "memlock=-1"),
    ):
        assert pair in pairs, pair
    tail = argv[argv.index(IMAGE) :]
    assert tail == [
        IMAGE,
        "python3",
        adapter.CONTAINER_AGENT,
        "container",
        "--node",
        "node0",
    ]


def test_extract_json_document_and_remote_time() -> None:
    doc = adapter.extract_json_document(
        CUDA_BANNER + '{"kind": "x"}\n[1, 2]\n{broken\n'
    )
    assert doc == {"kind": "x"}
    with pytest.raises(ValueError):
        adapter.extract_json_document("Traceback (most recent call last):\n")
    assert adapter.parse_remote_time("1793871100.125\n") == 1793871100.125
    assert adapter.parse_remote_time("") is None
    assert adapter.parse_remote_time("bash: python: not found\n") is None


# --------------------------------------------------------------------------
# Host facts
# --------------------------------------------------------------------------


def test_host_facts_is_fail_soft_and_keeps_raw_output(tmp_path: Path) -> None:
    good = compact("node_facts_h100.json")
    wrong_kind = json.dumps({"kind": "disk_read"}) + "\n"

    def responder(node: str, command: str) -> CommandResult:
        if node == "node0":
            return result(node, command, out=good)
        if node == "node1":
            return result(node, command, out="Traceback (most recent call last):\n")
        if node == "node2":
            return result(node, command, 124, err="\n[timeout]")
        return result(node, command, out=wrong_kind)

    rec = TimedRecorder(node_names=NODES, responder=responder)
    facts = collector(rec, tmp_path).host_facts(NODES, tag="before")
    assert list(facts) == list(NODES)
    assert facts["node0"].available and facts["node0"].node == "node0"
    assert len(facts["node0"].gpus) == 2
    for node in ("node1", "node2", "node3"):
        assert facts[node].available is False and facts[node].node == node
    assert "no JSON object" in facts["node1"].notes[0].error
    assert facts["node2"].notes[0].error.startswith("exit 124")
    assert "unparsable" in facts["node3"].notes[0].error
    raw = tmp_path / "facts/host-before"
    assert json.loads((raw / "node0.json").read_text()) == json.loads(good)
    assert (raw / "node3.json").exists()  # kept even though it did not parse
    assert "no JSON object" in (raw / "node1.error.txt").read_text()
    assert "[timeout]" in (raw / "node2.error.txt").read_text()
    expected = {(n, adapter.host_facts_command(PYTHON, SRC, n)) for n in NODES}
    assert {(n, c) for n, c, _ in rec.timeouts} == expected
    assert {t for _, _, t in rec.timeouts} == {adapter.HOST_TIMEOUT_S}
    codes = {(f.code, f.node) for f in nf.fact_findings(facts)}
    assert {("facts_unavailable", n) for n in ("node1", "node2", "node3")} <= codes


@pytest.mark.parametrize("tag", ["", "../escape", "a/b", "-x"])
def test_host_facts_rejects_unsafe_tags(tmp_path: Path, tag: str) -> None:
    rec = RecordingExecutor(node_names=NODES)
    with pytest.raises(ValueError):
        collector(rec, tmp_path).host_facts(NODES, tag=tag)


def test_unwritable_results_dir_never_fails_collection(tmp_path: Path) -> None:
    blocker = tmp_path / "facts"
    blocker.write_text("not a directory")
    good = compact("node_facts_h100.json")
    rec = RecordingExecutor(
        node_names=NODES, responder=lambda n, c: result(n, c, out=good)
    )
    facts = collector(rec, tmp_path).host_facts(["node0"], tag="after")
    assert facts["node0"].available


class ExplodingCluster:
    """A cluster whose transport raises (ssh died mid-run)."""

    nodes = NODES

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        raise OSError("ssh: connect to host 10.42.0.2 port 22: Connection refused")

    def fanout(self, nodes: Sequence[str], fn):
        return {node: fn(node) for node in nodes}


def test_transport_errors_become_unavailable_documents(tmp_path: Path) -> None:
    facts_collector = adapter.FactCollector(
        ExplodingCluster(),
        image=IMAGE,
        workdir=WORKDIR,
        results_dir=tmp_path,
        now=lambda: 0.0,
    )
    facts = facts_collector.host_facts(["node1"], tag="before")
    assert "Connection refused" in facts["node1"].notes[0].error
    assert facts_collector.software("node1") is None
    assert facts_collector.clock_samples(["node0", "node1"]) == {
        "node0": [(0.0, 0.0, 0.0)],
        "node1": [],
    }
    read = facts_collector.disk_read(["node1"], "/models")["node1"]
    assert read.gbs is None and "Connection refused" in read.error


# --------------------------------------------------------------------------
# Software stack
# --------------------------------------------------------------------------


def test_software_runs_the_container_agent(tmp_path: Path) -> None:
    payload = compact("software_stack.json")

    def responder(node: str, command: str) -> CommandResult:
        if command.startswith("docker run"):
            return result(node, command, out=CUDA_BANNER + payload)
        return result(node, command)

    rec = TimedRecorder(node_names=NODES, responder=responder)
    stack = collector(rec, tmp_path).software("node0")
    assert stack is not None and stack.image == IMAGE and stack.node == "node0"
    assert stack.torch.nccl == "2.28.3"
    commands = rec.commands_for("node0")
    assert commands[0] == "docker rm -f cb-facts-sw-node0 >/dev/null 2>&1 || true"
    run = shlex.split(commands[1])
    assert run[:3] == ["docker", "run", "--rm"] and "cb.exp=exp1" in run
    assert "cb.role=facts" in run and len(commands) == 2
    assert rec.timeouts[1][2] == adapter.SOFTWARE_TIMEOUT_S
    saved = json.loads((tmp_path / "facts/software-node0.json").read_text())
    assert saved["kind"] == "software_stack"


@pytest.mark.parametrize(
    ("rc", "err"),
    [
        (124, "\n[timeout]"),
        (125, "docker: Error response from daemon: could not select device driver"),
    ],
)
def test_software_failure_cleans_up_and_returns_none(
    tmp_path: Path, rc: int, err: str
) -> None:
    def responder(node: str, command: str) -> CommandResult:
        if command.startswith("docker run"):
            return result(node, command, rc, err=err)
        return result(node, command)

    rec = RecordingExecutor(node_names=NODES, responder=responder)
    assert collector(rec, tmp_path).software("node1") is None
    commands = rec.commands_for("node1")
    assert [c.split()[1] for c in commands] == ["rm", "run", "rm"]
    error = (tmp_path / "facts/software-node1.error.txt").read_text()
    assert f"exit {rc}" in error and err.strip() in error


# --------------------------------------------------------------------------
# Clock samples
# --------------------------------------------------------------------------


def test_clock_samples_round_trips(tmp_path: Path) -> None:
    times = iter(
        [1000.0, 1000.10, 1000.13, 1000.20, 1000.21, 1000.30, 1000.36, 1000.4, 1000.41]
    )
    remote = iter(["1000.615\n", "1000.705\n", "1000.83\n"])

    def responder(node: str, command: str) -> CommandResult:
        assert command == adapter.clock_command(PYTHON)
        if node == "node1":
            return result(node, command, out=next(remote))
        return result(node, command, 255, err="ssh: Could not resolve hostname")

    rec = TimedRecorder(node_names=NODES, responder=responder)
    samples = collector(rec, tmp_path, now=lambda: next(times)).clock_samples(
        ["node0", "node1", "node2"], rounds=3
    )
    assert samples["node0"] == [(1000.0, 1000.0, 1000.0)]
    assert samples["node1"] == [
        (1000.10, 1000.615, 1000.13),
        (1000.20, 1000.705, 1000.21),
        (1000.30, 1000.83, 1000.36),
    ]
    assert samples["node2"] == []  # one failed attempt, no retries
    assert rec.commands_for("node0") == []  # the reference is never contacted
    assert {t for _, _, t in rec.timeouts} == {adapter.CLOCK_TIMEOUT_S}
    offsets = nf.clock_offsets(samples)
    assert offsets["node0"].offset_s == 0.0
    assert offsets["node1"].offset_s == pytest.approx(0.5)
    assert offsets["node1"].uncertainty_s == pytest.approx(0.005)
    saved = json.loads((tmp_path / "facts/clock-samples.json").read_text())
    assert saved["reference"] == "node0" and saved["samples"]["node2"] == []


# --------------------------------------------------------------------------
# Disk reads
# --------------------------------------------------------------------------


def test_disk_read_collects_every_node(tmp_path: Path) -> None:
    def responder(node: str, command: str) -> CommandResult:
        if node == "node1":
            return result(node, command, 1, err="PermissionError: [Errno 13] /models")
        read = nf.synthetic_disk_read("", "/models/glm", gbs=3.5)
        return result(node, command, out=read.model_dump_json() + "\n")

    rec = TimedRecorder(node_names=NODES, responder=responder)
    reads = collector(rec, tmp_path).disk_read(
        ["node0", "node1"], "/models/glm", 1 << 30, max_seconds=60.0
    )
    assert reads["node0"].gbs == 3.5 and reads["node0"].node == "node0"
    assert reads["node1"].gbs is None and "Errno 13" in reads["node1"].error
    command = rec.commands_for("node0")[0]
    assert command == adapter.disk_read_command(
        PYTHON, SRC, "node0", "/models/glm", max_bytes=1 << 30, max_seconds=60.0
    )
    assert {t for _, _, t in rec.timeouts} == {3 * 60.0 + adapter.DISK_TIMEOUT_MARGIN_S}
    assert (tmp_path / "facts/disk-read/node0.json").exists()
    assert (tmp_path / "facts/disk-read/node1.error.txt").exists()
    findings = {(f.code, f.node) for f in nf.disk_findings(reads)}
    assert findings == {("disk_read_failed", "node1")}


# --------------------------------------------------------------------------
# Dry-run wiring
# --------------------------------------------------------------------------


def test_simulated_stdout_drives_the_whole_collector(tmp_path: Path) -> None:
    clock = {"t": 1_800_000_000.0}

    def responder(node: str, command: str) -> CommandResult | None:
        out = adapter.simulated_stdout(
            command, node, node_names=NODES, gpu_name="NVIDIA B200", now=clock["t"]
        )
        return None if out is None else result(node, command, out=out)

    rec = RecordingExecutor(node_names=NODES, responder=responder)
    facts_collector = collector(rec, tmp_path, now=lambda: clock["t"])
    before = facts_collector.host_facts(NODES, tag="before")
    after = facts_collector.host_facts(NODES, tag="after")
    assert all(f.available for f in before.values())
    assert before["node2"].gpus[0].product == "NVIDIA B200"
    assert nf.fact_findings(before) == []
    for node in NODES:
        delta = nf.health_delta(before[node], after[node])
        assert delta.complete and nf.delta_findings(delta) == []
    stack = facts_collector.software("node0")
    assert stack is not None and stack.image == IMAGE
    assert nf.software_findings(stack) == []
    offsets = nf.clock_offsets(facts_collector.clock_samples(list(NODES)))
    assert {n: o.offset_s for n, o in offsets.items()} == {n: 0.0 for n in NODES}
    reads = facts_collector.disk_read(NODES, "/models/glm")
    assert {r.method for r in reads.values()} == {"dd_direct"}
    assert (
        adapter.simulated_stdout("echo hi", "node0", node_names=NODES, gpu_name="x")
        is None
    )
