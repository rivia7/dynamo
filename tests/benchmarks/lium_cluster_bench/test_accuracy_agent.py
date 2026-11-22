# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agents/accuracy_probe.py: prompts, answer extraction, scoring, the threaded
evaluation against a scripted fake server (no sockets), the dataset fetch and
the command line.

Every probe document is parsed with ``domain/accuracy.py``: that is the
contract between the agent and its parser.
"""

from __future__ import annotations

import ast
import hashlib
import http.client
import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmarks.lium_cluster_bench.agents import accuracy_probe as ap
from benchmarks.lium_cluster_bench.domain import accuracy as acc

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "accuracy"
DATASET = FIXTURES / "gsm8k_synthetic.jsonl"
AGENT = Path(ap.__file__)
URL = "http://127.0.0.1:8000"
ENDPOINT = URL + "/v1/completions"
MODEL = "glm-5.3"
ROWS = [json.loads(line) for line in DATASET.read_text().splitlines()]
QUESTIONS = [row["question"] for row in ROWS]
DATASET_SHA = hashlib.sha256(DATASET.read_bytes()).hexdigest()
STOP = ["Question", "Assistant:", "<|separator|>"]


def completion(
    text: str | None,
    *,
    finish: str = "stop",
    usage: tuple[int, int] | None = (612, 41),
    reasoning: str | None = None,
) -> tuple[int, str]:
    """A Dynamo-style non-streaming /v1/completions answer."""
    choice: dict[str, Any] = {
        "index": 0,
        "text": text,
        "finish_reason": finish,
        "logprobs": None,
    }
    if reasoning is not None:
        choice["reasoning_content"] = reasoning
    body: dict[str, Any] = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1793871100,
        "model": MODEL,
        "choices": [choice],
    }
    if usage is not None:
        body["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return 200, json.dumps(body)


@dataclass
class FakeServer:
    """Scripted /v1/completions keyed by dataset index of the asked question.

    Each question consumes its steps in order (the last repeats): a
    ``(status, body)`` tuple is returned, an exception raised, an Event
    waited on (a hung request).
    """

    script: dict[int, list[Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def post(self, url: str, body: bytes, timeout: float) -> tuple[int, str]:
        payload = json.loads(body)
        asked = payload["prompt"].rsplit("Question: ", 1)[1]
        index = QUESTIONS.index(asked[: -len("\nAnswer:")])
        with self.lock:
            self.calls.append(
                {"url": url, "index": index, "timeout": timeout, "payload": payload}
            )
            steps = self.script.get(index, [(404, '{"error": "unscripted"}')])
            step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, threading.Event):
            step.wait(timeout=10)
            return completion(" #### 0")
        if isinstance(step, BaseException):
            raise step
        return step


def make_host(server: FakeServer, sleeps: list[float] | None = None) -> ap.Host:
    record = sleeps if sleeps is not None else []
    return ap.Host(post=server.post, sleep=record.append, now=lambda: 1.8e9)


def probe_args(*extra: str, dataset: Path = DATASET) -> list[str]:
    return ["gsm8k", "--url", URL, "--model", MODEL, "--dataset", str(dataset), *extra]


def run_main(
    capsys: pytest.CaptureFixture[str], argv: list[str], host: ap.Host
) -> dict[str, Any]:
    assert ap.main(argv, host=host) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1, out  # exactly one JSON line
    return json.loads(out)


# --------------------------------------------------------------------------
# Prompts and answers
# --------------------------------------------------------------------------


def test_prompt_follows_sglang_few_shot_format() -> None:
    items, notes = ap.parse_dataset(DATASET.read_text())
    assert notes == [] and len(items) == 16
    assert [i.gold for i in items] == [
        4,
        20,
        180,
        7,
        48,
        350,
        1200,
        -3,
        8,
        18,
        163,
        6000,
        63,
        32,
        180,
        7500,
    ]
    prefix = ap.few_shot_prefix(items[:2])
    assert prefix == (
        f"Question: {ROWS[0]['question']}\nAnswer: {ROWS[0]['answer']}\n\n"
        f"Question: {ROWS[1]['question']}\nAnswer: {ROWS[1]['answer']}\n\n"
    )
    prompt = ap.build_prompt(prefix, items[5].question)
    assert prompt == prefix + f"Question: {ROWS[5]['question']}\nAnswer:"
    assert ap.build_prompt("", "Q?") == "Question: Q?\nAnswer:"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" She pays $1,200 for the bikes.", 1200.0),
        ("The temperature is -3 now.", -3.0),
        (" Each costs 3.50 dollars.", 3.5),
        (" 3 * 6 = <<3*6=18>>18 eggs.\n#### 18", 18.0),
        ("#### $1,450,000.", 1450000.0),
        ("#### -3", -3.0),
        ("####-3", -3.0),
        (" It fell to −5 degrees.", -5.0),
        ("Total: 12,345.67 units", 12345.67),
        ("Bad grouping 1,2345", 2345.0),
        # "-" between operands is a subtraction, never a sign.
        ("16-3 = 13 left, then 16 - 3 more", 3.0),
        ("(5)-2", 2.0),
        # The first number after the marker wins; text after a stop is cut.
        ("#### 18 dollars, i.e. 1800 cents", 18.0),
        (" 2 + 2 = 4\n#### 4\n\nQuestion: Next? #### 7", 4.0),
        (" The total is 42.\n#### unknown", 42.0),
        ("I am not sure.", None),
        ("####", None),
        ("", None),
    ],
)
def test_extract_answer(text: str, expected: float | None) -> None:
    assert ap.extract_answer(text) == expected


def test_gold_and_match() -> None:
    assert ap.gold_answer("a 3\nb 4\n#### 1,200") == 1200.0
    assert ap.gold_answer("Steps 1 and 2, no marker.") is None
    assert ap.answers_match(18.0, 18.0) and ap.answers_match(18.000001, 18.0)
    assert not ap.answers_match(18.01, 18.0)
    assert not ap.answers_match(None, 18.0) and not ap.answers_match(3.0, None)
    assert ap.compact_number(18.0) == 18 and ap.compact_number(3.5) == 3.5
    assert ap.compact_number(None) is None


def test_think_blocks_and_answer_sources() -> None:
    assert ap.split_think("plain 1") == ("plain 1", "")
    assert ap.split_think("<think>a 1</think> b 2") == (" b 2", "a 1")
    assert ap.split_think("pre 1</think> post 2") == (" post 2", "pre 1")
    assert ap.split_think("x <think>never done 7") == ("x ", "never done 7")
    assert ap.split_think("<think>a</think>b<think>c</think>d") == ("bd", "a\nc")

    def source(text: str, reasoning: str | None = None) -> tuple[Any, str]:
        return ap.answer_of(ap.Completion(text, "stop", None, None, reasoning))

    assert source("<think>2 + 2 = 4</think> So the answer is\n#### 5") == (5, "text")
    assert source("", reasoning="45 * 4 = 180\n#### 180") == (180, "reasoning")
    assert source(" no digits", reasoning="answer 9") == (9, "reasoning")
    assert source("<think>three fields make 7,500") == (7500, "think")
    assert source(" none here") == (None, "none")


def test_parse_completion() -> None:
    body = (FIXTURES / "completion_response.json").read_text()
    parsed = ap.parse_completion(body)
    assert parsed.text.endswith("#### 350\n\n") and parsed.finish_reason == "stop"
    assert (parsed.prompt_tokens, parsed.completion_tokens) == (612, 41)
    assert parsed.reasoning is None
    assert ap.answer_of(parsed) == (350, "text")
    obj = json.loads(body)
    obj["choices"][0]["text"] = None  # everything in a reasoning field
    obj["choices"][0]["reasoning"] = "so #### 350"
    del obj["usage"]
    parsed = ap.parse_completion(json.dumps(obj))
    assert parsed.text == "" and parsed.reasoning == "so #### 350"
    assert parsed.prompt_tokens is None and parsed.completion_tokens is None


@pytest.mark.parametrize(
    "body",
    [
        "<html>502 Bad Gateway</html>",
        "[]",
        '{"choices": []}',
        '{"choices": ["text"]}',
        '{"choices": [{"text": null, "finish_reason": "stop"}]}',
        '{"error": {"message": "Model not found", "code": 404}}',
    ],
)
def test_parse_completion_rejects_malformed_bodies(body: str) -> None:
    with pytest.raises(ValueError):
        ap.parse_completion(body)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (socket.timeout("timed out"), "timeout"),
        (TimeoutError(), "timeout"),
        (urllib.error.URLError(socket.timeout("timed out")), "timeout"),
        (urllib.error.URLError("timed out"), "timeout"),
        (ConnectionRefusedError(61, "Connection refused"), "connection"),
        (urllib.error.URLError(ConnectionRefusedError(61, "refused")), "connection"),
        (http.client.RemoteDisconnected("Remote end closed connection"), "connection"),
        (http.client.IncompleteRead(b"par"), "connection"),
        (OSError(101, "Network is unreachable"), "connection"),
    ],
)
def test_classify_transport_error(exc: BaseException, kind: str) -> None:
    assert ap.classify_transport_error(exc) == kind


def test_endpoint_and_payload() -> None:
    for url in ("http://h:8000", "http://h:8000/", "http://h:8000/v1"):
        assert ap.completions_endpoint(url) == "http://h:8000/v1/completions"
    assert ap.completions_endpoint(ENDPOINT) == ENDPOINT
    assert ap.request_payload("m", "P", 512) == {
        "model": "m",
        "prompt": "P",
        "max_tokens": 512,
        "temperature": 0.0,
        "stop": STOP,
        "stream": False,
    }


def test_small_helpers() -> None:
    assert ap.percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert ap.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert ap.percentile([5.0], 100) == 5.0 and ap.percentile([], 50) is None
    long_text = "a" * 600 + "END"
    clipped = ap.clip_text(long_text)
    assert clipped.startswith("a" * ap.SAMPLE_HEAD_CHARS) and clipped.endswith("END")
    assert " ... " in clipped and len(clipped) < len(long_text)
    assert ap.clip_text("short") == "short"


# --------------------------------------------------------------------------
# One request: retries and failures
# --------------------------------------------------------------------------


def scripted(steps: list[Any]) -> tuple[Callable[..., tuple[int, str]], list[float]]:
    timeouts: list[float] = []

    def post(url: str, body: bytes, timeout: float) -> tuple[int, str]:
        timeouts.append(timeout)
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, BaseException):
            raise step
        return step

    return post, timeouts


@pytest.mark.parametrize(
    ("steps", "retries", "kind", "calls"),
    [
        ([ConnectionResetError(54, "reset")], 2, "connection", 3),
        ([ConnectionResetError(54, "reset")], 0, "connection", 1),
        ([(503, "busy")], 2, "http_503", 3),
        ([(429, "slow down"), (502, "bad gateway")], 1, "http_502", 2),
        ([socket.timeout("timed out")], 2, "timeout", 1),  # never retried
        ([(400, '{"error": "context too long"}')], 2, "http_400", 1),
        ([(200, "not json")], 2, "malformed", 1),
        ([ConnectionRefusedError(61, "refused"), completion(" #### 1")], 2, None, 2),
    ],
)
def test_complete_one(
    steps: list[Any], retries: int, kind: str | None, calls: int
) -> None:
    post, timeouts = scripted(list(steps))
    sleeps: list[float] = []
    host = ap.Host(post=post, sleep=sleeps.append)
    job = ap.Job(5, 1.0, b"{}")
    outcome = ap.complete_one(
        host,
        ENDPOINT,
        job,
        timeout_s=120.0,
        retries=retries,
        deadline=time.monotonic() + 600,
    )
    assert outcome.error == kind and outcome.attempts == calls
    assert len(timeouts) == calls and all(1.0 <= t <= 120.0 for t in timeouts)
    assert sleeps == [0.5, 1.0][: calls - 1]
    assert (outcome.completion is not None) == (kind is None)


def test_complete_one_respects_the_deadline() -> None:
    post, timeouts = scripted([completion(" #### 1")])
    host = ap.Host(post=post)
    job = ap.Job(5, 1.0, b"{}")
    late = ap.complete_one(
        host, ENDPOINT, job, timeout_s=120.0, retries=2, deadline=time.monotonic()
    )
    assert late.error == "deadline" and timeouts == []
    soon = ap.complete_one(
        host, ENDPOINT, job, timeout_s=120.0, retries=2, deadline=time.monotonic() + 5
    )
    assert soon.completion is not None and timeouts[0] <= 5.0
    # A request whose timeout the deadline shortened was cut by the deadline.
    hung, _ = scripted([socket.timeout("timed out")])
    cut = ap.complete_one(
        ap.Host(post=hung),
        ENDPOINT,
        job,
        timeout_s=120.0,
        retries=2,
        deadline=time.monotonic() + 5,
    )
    assert cut.error == "deadline" and cut.attempts == 1
    outcomes = {5: ap.Outcome(5, error="deadline")}
    summary = ap.summarize([job], outcomes, deadline_hit=False)
    assert summary["deadline_hit"] is True and summary["error_kinds"] == {"deadline": 1}


# --------------------------------------------------------------------------
# Whole runs through main()
# --------------------------------------------------------------------------


def test_end_to_end_scores_every_kind_of_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeServer(
        {
            5: [completion(" Two weeks have 14 days: 25 * 14 = 350.\n#### 350")],
            6: [completion(" 40 * 30 = $1,200.\n#### $1,200")],
            7: [completion(" It is 8 - 5 = 3 degrees.\n#### 3")],  # sign lost
            8: [(200, "<html>bad gateway</html>")],
            9: [
                ConnectionRefusedError(61, "Connection refused"),
                completion(" 3 * 6 = 18 eggs.\n#### 18"),
            ],
            10: [(500, '{"error": {"message": "engine died", "code": 500}}')],
            11: [socket.timeout("timed out")],
            12: [completion(" I am not sure.")],
            13: [(503, '{"error": "busy"}'), completion(" 50 - 18 = 32.\n#### 32")],
            14: [completion("", reasoning="45 * 4 = 180.\n#### 180", usage=None)],
            15: [
                completion(
                    "<think>Three fields of 2,500 each make 7,500",
                    finish="length",
                    usage=(612, 512),
                )
            ],
        }
    )
    sleeps: list[float] = []
    out = tmp_path / "evidence" / "accuracy-gsm8k.json"
    doc = run_main(
        capsys,
        probe_args("--questions", "50", "--concurrency", "4", "--out", str(out)),
        make_host(server, sleeps),
    )
    assert json.loads(out.read_text()) == doc
    assert doc["kind"] == "accuracy_probe" and doc["error"] is None
    assert doc["dataset_sha256"] == DATASET_SHA and doc["dataset_items"] == 16
    assert (doc["shots"], doc["first_index"], doc["questions"]) == (5, 5, 11)
    assert (doc["n"], doc["correct"], doc["invalid"], doc["errors"]) == (8, 6, 1, 3)
    assert doc["accuracy"] == 0.75
    assert doc["error_kinds"] == {"http_500": 1, "malformed": 1, "timeout": 1}
    assert doc["answer_sources"] == {"none": 1, "reasoning": 1, "text": 5, "think": 1}
    assert doc["finish_reasons"] == {"length": 1, "stop": 7}
    assert doc["usage_items"] == 7 and doc["prompt_tokens"] == 7 * 612
    assert doc["completion_tokens"] == 6 * 41 + 512
    assert doc["deadline_hit"] is False and doc["started_at"] == 1.8e9
    assert doc["latency_p50_s"] is not None and doc["wall_s"] >= 0
    items = {item[0]: item for item in doc["items"]}
    assert sorted(items) == list(range(5, 16))
    assert items[6] == [6, True, 1200, 1200]
    assert items[7] == [7, False, 3, -3]
    assert items[8] == [8, None, None, 8]
    assert items[12] == [12, False, None, 63]
    assert items[14] == [14, True, 180, 180]
    assert [s["index"] for s in doc["samples"]] == [5, 7, 12]
    assert doc["samples"][1]["source"] == "text"
    calls = Counter(c["index"] for c in server.calls)
    assert calls[9] == 2 and calls[13] == 2 and sum(calls.values()) == 13
    assert sleeps == [0.5, 0.5]
    for call in server.calls:
        payload = call["payload"]
        assert call["url"] == ENDPOINT and payload["model"] == MODEL
        assert payload["temperature"] == 0.0 and payload["stop"] == STOP
        assert payload["max_tokens"] == 512 and payload["stream"] is False
        # Five exemplars from items [0, 5), then the evaluated question once.
        assert payload["prompt"].count("Question: ") == 6
        assert payload["prompt"].count(QUESTIONS[call["index"]]) == 1
    notes = {n["source"]: n["error"] for n in doc["notes"]}
    assert "only 11 item(s) after the exemplars" in notes["dataset"]
    assert "reasoning=1, think=1" in notes["answers"]
    request_notes = [n["error"] for n in doc["notes"] if n["source"] == "request"]
    assert any("engine died" in n for n in request_notes)
    assert all(len(n) <= ap.NOTE_CHARS for n in request_notes)

    result = acc.parse_probe_document(doc)
    assert result.status == "ok" and (result.n, result.correct) == (8, 6)
    assert result.answer_sources == doc["answer_sources"]
    assert result.params == acc.AccuracyParams(questions=50, concurrency=4)
    assert result.protocol == f"gsm8k:{DATASET_SHA[:12]}:5shot:items5+11:greedy:max512"
    verdict = acc.accuracy_verdict(result, floor=0.5)
    assert verdict.status == "error" and verdict.code == "too_many_errors"


def test_deadline_still_writes_the_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = threading.Event()
    server = FakeServer({5: [completion(" #### 350")], 6: [gate], 7: [gate]})
    out = tmp_path / "accuracy.json"
    started = time.monotonic()
    try:
        doc = run_main(
            capsys,
            probe_args("--questions", "3", "--deadline-s", "0.6", "--out", str(out)),
            make_host(server),
        )
    finally:
        gate.set()  # release the hung fake requests
    assert time.monotonic() - started < 5.0  # did not wait for them
    assert doc["deadline_hit"] is True and doc["error"] is None
    assert (doc["questions"], doc["n"], doc["correct"], doc["errors"]) == (3, 1, 1, 2)
    assert doc["error_kinds"] == {"deadline": 2}
    assert json.loads(out.read_text())["deadline_hit"] is True
    assert acc.parse_probe_document(doc).deadline_hit


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_crashed_workers_do_not_wait_for_the_deadline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = FakeServer({i: [RuntimeError("bug in post")] for i in range(5, 16)})
    started = time.monotonic()
    doc = run_main(
        capsys,
        probe_args("--concurrency", "3", "--deadline-s", "60"),
        make_host(server),
    )
    assert time.monotonic() - started < 10.0
    assert doc["errors"] == 11 and doc["error_kinds"] == {"aborted": 11}
    assert doc["deadline_hit"] is False and doc["n"] == 0 and doc["accuracy"] is None


def test_fatal_problems_become_json_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeServer({})
    host = make_host(server)
    small = tmp_path / "small.jsonl"
    small.write_text("".join(DATASET.read_text().splitlines(keepends=True)[:5]))
    cases = [
        (probe_args(dataset=tmp_path / "missing.jsonl"), "dataset unreadable"),
        (probe_args("--expect-sha256", "0" * 64), "does not match the expected"),
        (probe_args(dataset=small), "need more than 5"),
        (
            ["gsm8k", "--url", "127.0.0.1:8000", "--model", MODEL, "--dataset", "x"],
            "invalid --url",
        ),
    ]
    for argv, message in cases:
        doc = run_main(capsys, argv, host)
        assert message in doc["error"], doc["error"]
        assert doc["n"] == 0 and doc["items"] == [] and doc["accuracy"] is None
        result = acc.parse_probe_document(doc)
        assert result.status == "error" and message in (result.reason or "")
    assert server.calls == []
    ok = run_main(capsys, probe_args("--expect-sha256", DATASET_SHA.upper()), host)
    assert ok["error"] is None and ok["questions"] == 11


def test_dataset_problems_are_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = DATASET.read_text().splitlines()
    mixed = [
        lines[0],
        "{broken",
        "",
        json.dumps({"question": "No answer field"}),
        *lines[1:6],
        json.dumps({"question": "No reference?", "answer": "No marker here."}),
    ]
    items, notes = ap.parse_dataset("\n".join(mixed))
    assert [i.index for i in items] == list(range(7))
    assert items[6].gold is None
    assert "2 unreadable line(s)" in notes[0]["error"] and "line 2" in notes[0]["error"]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(mixed) + "\n")
    server = FakeServer({5: [completion(" #### 350")]})
    doc = run_main(capsys, probe_args(dataset=path), make_host(server))
    assert doc["questions"] == 1 and doc["correct"] == 1
    joined = " ".join(n["error"] for n in doc["notes"])
    assert "reference skipped (first 6)" in joined and "unreadable" in joined


def test_bad_arguments_exit_2() -> None:
    for argv in (
        probe_args("--questions", "0"),
        probe_args("--deadline-s", "-1"),
        ["gsm8k", "--url", URL],
        ["fetch", "--sha256", "x", "--dest", "y"],
        [],
    ):
        with pytest.raises(SystemExit) as exc:
            ap.main(argv, host=ap.Host())
        assert exc.value.code == 2


# --------------------------------------------------------------------------
# Dataset fetch
# --------------------------------------------------------------------------


@dataclass
class FakeNet:
    """curl (via Host.run) and urllib (Host.get) serving fixed bytes."""

    payloads: dict[str, bytes]
    curl_rc: int = 0
    curl_installed: bool = True
    runs: list[list[str]] = field(default_factory=list)
    gets: list[str] = field(default_factory=list)

    def run(self, argv: list[str], timeout: float) -> ap.ToolResult:
        self.runs.append(list(argv))
        url = argv[-1]
        if self.curl_rc or url not in self.payloads:
            return ap.ToolResult(self.curl_rc or 22, "", "curl: (6) Could not resolve")
        Path(argv[argv.index("-o") + 1]).write_bytes(self.payloads[url])
        return ap.ToolResult(0, "", "")

    def get(self, url: str, timeout: float) -> bytes:
        self.gets.append(url)
        if url not in self.payloads:
            raise urllib.error.URLError("Name or service not known")
        return self.payloads[url]

    def host(self, sleeps: list[float]) -> ap.Host:
        return ap.Host(
            run=self.run,
            get=self.get,
            which=lambda name: "/usr/bin/curl" if self.curl_installed else None,
            sleep=sleeps.append,
        )


GOOD = DATASET.read_bytes()
MIRROR = "https://mirror.example/gsm8k/test.jsonl"


def fetch_args(dest: Path, *urls: str, sha: str = DATASET_SHA) -> list[str]:
    args = ["fetch", "--sha256", sha, "--dest", str(dest)]
    for url in urls or (acc.GSM8K_URL,):
        args += ["--url", url]
    return args


def leftovers(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".part")]


def test_fetch_reuses_a_verified_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "datasets" / "gsm8k-test.jsonl"
    dest.parent.mkdir()
    dest.write_bytes(GOOD)
    net = FakeNet({})
    doc = run_main(capsys, fetch_args(dest, sha=DATASET_SHA.upper()), net.host([]))
    assert doc["ok"] is True and doc["reused"] is True and doc["attempts"] == 0
    assert doc["size_bytes"] == len(GOOD) and net.runs == [] and net.gets == []
    status = acc.parse_fetch_document(doc, path=str(dest), expected_sha256=DATASET_SHA)
    assert status.ok and status.reused


def test_fetch_downloads_with_curl_then_urllib(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "datasets" / "gsm8k-test.jsonl"
    net = FakeNet({acc.GSM8K_URL: GOOD})
    out = tmp_path / "fetch.json"
    doc = run_main(capsys, [*fetch_args(dest), "--out", str(out)], net.host([]))
    assert doc["ok"] and doc["url"] == acc.GSM8K_URL and doc["attempts"] == 1
    assert dest.read_bytes() == GOOD and net.gets == []
    assert net.runs[0][:2] == ["curl", "-fsSL"] and "--max-time" in net.runs[0]
    assert json.loads(out.read_text()) == doc and leftovers(dest.parent) == []
    # curl fails: urllib, same attempt.
    dest.unlink()
    broken = FakeNet({acc.GSM8K_URL: GOOD}, curl_rc=6)
    doc = run_main(capsys, fetch_args(dest), broken.host([]))
    assert doc["ok"] and doc["attempts"] == 2 and broken.gets == [acc.GSM8K_URL]
    assert doc["notes"][0]["source"] == "curl"
    # No curl at all.
    dest.unlink()
    plain = FakeNet({acc.GSM8K_URL: GOOD}, curl_installed=False)
    doc = run_main(capsys, fetch_args(dest), plain.host([]))
    assert doc["ok"] and plain.runs == [] and dest.read_bytes() == GOOD


def test_fetch_falls_back_to_a_mirror_and_replaces_a_bad_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "gsm8k-test.jsonl"
    dest.write_bytes(b"stale partial download")
    net = FakeNet({MIRROR: GOOD})
    doc = run_main(capsys, fetch_args(dest, acc.GSM8K_URL, MIRROR), net.host([]))
    assert doc["ok"] and doc["url"] == MIRROR and dest.read_bytes() == GOOD
    sources = [n["source"] for n in doc["notes"]]
    assert sources == ["cache", "curl", "urllib"]


def test_fetch_never_installs_unverified_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "datasets" / "gsm8k-test.jsonl"
    net = FakeNet({acc.GSM8K_URL: b'{"question": "tampered", "answer": "#### 1"}\n'})
    sleeps: list[float] = []
    doc = run_main(capsys, [*fetch_args(dest), "--attempts", "3"], net.host(sleeps))
    assert doc["ok"] is False and "no verified copy after 3" in doc["error"]
    assert not dest.exists() and leftovers(dest.parent) == []
    assert sleeps == [2.0, 4.0] and len(net.runs) == 3  # wrong bytes: no urllib
    assert {n["source"] for n in doc["notes"]} == {"verify"}
    status = acc.parse_fetch_document(doc, path=str(dest), expected_sha256=DATASET_SHA)
    assert not status.ok and status.error


# --------------------------------------------------------------------------
# Real HTTP and tool wrappers (patched transports, no sockets)
# --------------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    status = 200


def test_http_post_sends_json_and_returns_error_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[urllib.request.Request, float]] = []

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float) -> Any:
            seen.append((request, timeout))
            if len(seen) == 1:
                return FakeResponse(b'{"ok": true}')
            raise urllib.error.HTTPError(
                request.full_url, 503, "busy", {}, io.BytesIO(b'{"error": "busy"}')
            )

    monkeypatch.setattr(ap, "_DIRECT_OPENER", Opener())
    assert ap.http_post(ENDPOINT, b'{"a": 1}', 7.0) == (200, '{"ok": true}')
    request, timeout = seen[0]
    assert request.get_method() == "POST" and request.data == b'{"a": 1}'
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 7.0 and request.full_url == ENDPOINT
    assert ap.http_post(ENDPOINT, b"{}", 1.0) == (503, '{"error": "busy"}')


def test_requests_bypass_http_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    def proxied(opener: urllib.request.OpenerDirector) -> bool:
        return any(isinstance(h, urllib.request.ProxyHandler) for h in opener.handlers)

    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    assert proxied(urllib.request.build_opener())  # the default honours it
    assert not proxied(ap._DIRECT_OPENER)


def test_http_get_and_run_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        assert request.get_header("User-agent") == ap.USER_AGENT
        return FakeResponse(b"payload")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert ap.http_get(acc.GSM8K_URL, 5.0) == b"payload"
    assert ap.run_tool(["cb-no-such-tool-xyz"], timeout=5.0) is None
    done = ap.run_tool([sys.executable, "-c", "print('hi')"], timeout=30.0)
    assert done is not None and done.ok and done.stdout == "hi\n"
    slow = ap.run_tool([sys.executable, "-c", "import time; time.sleep(5)"], 0.2)
    assert slow is not None and slow.rc == 124


# --------------------------------------------------------------------------
# Standalone script
# --------------------------------------------------------------------------


def test_agent_is_standalone_python38() -> None:
    tree = ast.parse(AGENT.read_text(), feature_version=(3, 8))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
    first = next(n for n in tree.body if isinstance(n, ast.ImportFrom))
    assert first.module == "__future__"


def test_script_runs_isolated(tmp_path: Path) -> None:
    """No repo on sys.path and an empty PATH: still one JSON document, exit 0."""
    out = tmp_path / "accuracy.json"
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            str(AGENT),
            *probe_args("--out", str(out), dataset=tmp_path / "missing.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": str(tmp_path)},
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    doc = json.loads(probe.stdout)
    assert "dataset unreadable" in doc["error"] and json.loads(out.read_text()) == doc
    dest = tmp_path / "gsm8k-test.jsonl"
    dest.write_bytes(GOOD)
    fetch = subprocess.run(
        [sys.executable, "-I", str(AGENT), *fetch_args(dest)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": str(tmp_path)},
        check=False,
    )
    assert fetch.returncode == 0, fetch.stderr
    assert json.loads(fetch.stdout)["reused"] is True
