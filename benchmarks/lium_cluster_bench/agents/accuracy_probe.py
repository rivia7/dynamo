# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Few-shot GSM8K accuracy probe against an OpenAI-compatible server.

Run as::

    <python> accuracy_probe.py gsm8k --url URL --model NAME --dataset PATH
        [--shots 5] [--questions 200] [--concurrency 32] [--max-tokens 512]
        [--request-timeout-s 120] [--deadline-s 900] [--retries 2]
        [--expect-sha256 HEX] [--out PATH]
    <python> accuracy_probe.py fetch --url URL [--url URL ...] --sha256 HEX
        --dest PATH [--attempts 3] [--timeout-s 60] [--out PATH]

Throughput runs send ``ignore_eos``, so a numerically broken configuration
(bad low-precision kernels, a wrong attention backend, a broken speculative
draft) emits as many tokens as a healthy one and looks just as fast. This
probe asks real questions instead: GSM8K test items, greedy, through the
frontend's raw ``/v1/completions`` (no chat template, so no reasoning
preamble). ``--url`` is the server base URL; ``/v1/completions`` is appended
unless already present. Requests bypass any HTTP proxy.

Protocol (mirrors SGLang's few-shot GSM8K, ``benchmark/gsm8k/bench_sglang.py``):
the prompt is ``k`` exemplars ``"Question: {q}\\nAnswer: {a}\\n\\n"`` (the full
reference answer, calculator annotations and ``#### N`` included) followed by
``"Question: {q}\\nAnswer:"``; temperature 0, ``max_tokens`` 512, stop at
``"Question"``, ``"Assistant:"`` or ``"<|separator|>"``. Unlike SGLang, the
exemplars are items ``[0, k)`` and the evaluated items ``[k, k+n)``, so no
evaluated question appears in its own prompt. The prediction is the number
after ``####`` when present, else the last number of the completion (cut at
the first stop string in case the server did not apply them); thousands
separators, signs and decimals are handled and values compare numerically.

Reasoning output (the GLM workers run a reasoning parser; where it lands in a
``/v1/completions`` response is UNVERIFIED, so every place is tolerated): the
answer comes from the first source holding a number - ``text`` without its
``<think>`` blocks, then the choice's ``reasoning_content`` / ``reasoning``
field, then the think-block content itself. ``answer_sources`` counts which
source answered each question (``none`` = no number anywhere).

``fetch`` downloads the dataset (curl when present, else urllib, retried),
verifies its sha256 and atomically installs it; a verified copy is reused.

Standalone: stdlib only, Python >= 3.8 syntax, no ``benchmarks.*`` imports.
One JSON line goes to stdout (and atomically to ``--out`` when given); exit 0
whenever it was produced (2 on bad arguments). Problems never raise: they
become ``error`` (fatal, nothing was evaluated) or ``notes``. ``Host``
injects HTTP, tools and clocks, so tests need no sockets.

JSON contract (parsed by ``benchmarks/lium_cluster_bench/domain/accuracy.py``)
=========================================================================
gsm8k (``kind="accuracy_probe"``, ``version=1``)
  benchmark "gsm8k", model, endpoint, dataset (path), dataset_sha256,
  dataset_items, shots, first_index (= shots), questions (evaluated), n
  (answered), correct, accuracy (correct / n, null when n = 0), invalid
  (answered without a number; counted wrong), errors (not answered),
  error_kinds {timeout|connection|http_<status>|malformed|deadline|aborted:
  count}, finish_reasons {stop|length|..: count}, deadline_hit, wall_s,
  latency_p50_s, latency_max_s, prompt_tokens, completion_tokens (sums over
  answers that carried ``usage``; null when none did), usage_items,
  answer_sources {text|reasoning|think|none: count} (over answered
  questions), params {shots, questions (requested), concurrency, max_tokens,
  request_timeout_s, deadline_s, retries, temperature, stop [..]}, items
  [[index, ok, pred, gold]] (ok null = not answered, pred null = no number),
  samples [{index, ok, pred, gold, finish_reason, source, text, reasoning}]
  (up to 3 wrong and 1 right completion, clipped), started_at, error, notes
  [{source, error}]
fetch (``kind="dataset_fetch"``, ``version=1``)
  ok, path, url, sha256, expected_sha256, size_bytes, reused, attempts, seconds,
  error, notes
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

VERSION = 1
KIND_PROBE = "accuracy_probe"
KIND_FETCH = "dataset_fetch"
BENCHMARK = "gsm8k"
STOP_SEQUENCES = ("Question", "Assistant:", "<|separator|>")
TEMPERATURE = 0.0

DEFAULT_SHOTS = 5
DEFAULT_QUESTIONS = 200
DEFAULT_CONCURRENCY = 32
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_DEADLINE_S = 900.0
DEFAULT_RETRIES = 2
RETRY_BACKOFF_S = 0.5
# Transient availability answers worth another attempt (503 = model not ready).
RETRY_STATUSES = frozenset({429, 502, 503, 504})
MIN_REQUEST_TIMEOUT_S = 1.0
POLL_S = 0.25

DEFAULT_FETCH_ATTEMPTS = 3
DEFAULT_FETCH_TIMEOUT_S = 60.0
FETCH_BACKOFF_S = 2.0
CURL_CONNECT_TIMEOUT_S = 20

WRONG_SAMPLES = 3
RIGHT_SAMPLES = 1
SAMPLE_HEAD_CHARS = 400
SAMPLE_TAIL_CHARS = 120
NOTE_CHARS = 300
MATCH_RTOL = 1e-6
USER_AGENT = "cluster-bench-accuracy-probe/1"

# A number: thousands groups ("1,450,000") or plain digits, optional decimals.
# "-" is a sign only when attached to the digits and not preceded by a word
# character or a closing bracket, so "16-3" and "16 - 3" are subtractions.
_NUMBER_RE = re.compile(r"(?:(?<![\w)\]])-)?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?")
_UNICODE_MINUS = chr(0x2212)  # U+2212 MINUS SIGN
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Choice fields that may carry parsed-out reasoning (UNVERIFIED names).
REASONING_FIELDS = ("reasoning_content", "reasoning")
ANSWER_SOURCES = ("text", "reasoning", "think")


# --------------------------------------------------------------------------
# Pure helpers: prompts, answers, responses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One dataset record; ``index`` is its position among the valid records."""

    index: int
    question: str
    answer: str
    gold: Optional[float]


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning: Optional[str] = None  # a parsed-out reasoning field, if any


def find_numbers(text: str) -> List[float]:
    """Every number in ``text`` in order (thousands separators, signs, decimals)."""
    clean = text.replace(_UNICODE_MINUS, "-")
    return [float(m.group(0).replace(",", "")) for m in _NUMBER_RE.finditer(clean)]


def truncate_at_stop(text: str, stops: Sequence[str] = STOP_SEQUENCES) -> str:
    """``text`` up to the first stop string (the server may not have cut it)."""
    cut = len(text)
    for stop in stops:
        at = text.find(stop)
        if 0 <= at < cut:
            cut = at
    return text[:cut]


def extract_answer(text: Optional[str]) -> Optional[float]:
    """Predicted value: first number after ``####``, else the last number."""
    if not text:
        return None
    body = truncate_at_stop(text)
    marker = body.find("####")
    if marker >= 0:
        after = find_numbers(body[marker + 4 :])
        if after:
            return after[0]
    numbers = find_numbers(body)
    return numbers[-1] if numbers else None


def split_think(text: str) -> Tuple[str, str]:
    """(visible text, reasoning) of a completion that may hold think blocks.

    Handles closed ``<think>..</think>`` blocks, a leading ``..</think>``
    whose opener was in the prompt, and an unclosed trailing ``<think>..``
    (the reasoning never finished).
    """
    visible: List[str] = []
    thinking: List[str] = []
    rest = text
    close = rest.find(THINK_CLOSE)
    opener = rest.find(THINK_OPEN)
    if close >= 0 and (opener < 0 or close < opener):
        thinking.append(rest[:close])
        rest = rest[close + len(THINK_CLOSE) :]
    while True:
        opener = rest.find(THINK_OPEN)
        if opener < 0:
            visible.append(rest)
            break
        visible.append(rest[:opener])
        rest = rest[opener + len(THINK_OPEN) :]
        close = rest.find(THINK_CLOSE)
        if close < 0:
            thinking.append(rest)
            break
        thinking.append(rest[:close])
        rest = rest[close + len(THINK_CLOSE) :]
    return "".join(visible), "\n".join(thinking)


def answer_of(completion: Completion) -> Tuple[Optional[float], str]:
    """(prediction, source): the first of visible text, reasoning field and
    think blocks that holds a number; ``(None, "none")`` when none does."""
    visible, thinking = split_think(completion.text)
    bodies = (visible, completion.reasoning or "", thinking)
    for source, body in zip(ANSWER_SOURCES, bodies):
        pred = extract_answer(body)
        if pred is not None:
            return pred, source
    return None, "none"


def gold_answer(answer: str) -> Optional[float]:
    """Reference value: the number after the final ``####`` (None without one)."""
    marker = answer.rfind("####")
    if marker < 0:
        return None
    numbers = find_numbers(answer[marker + 4 :])
    return numbers[0] if numbers else None


def answers_match(pred: Optional[float], gold: Optional[float]) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= MATCH_RTOL * max(1.0, abs(gold))


def format_example(question: str, answer: Optional[str] = None) -> str:
    """One exemplar (with ``answer``) or the open question (without)."""
    if answer is None:
        return "Question: " + question + "\nAnswer:"
    return "Question: " + question + "\nAnswer: " + answer + "\n\n"


def few_shot_prefix(exemplars: Sequence[Item]) -> str:
    return "".join(format_example(item.question, item.answer) for item in exemplars)


def build_prompt(prefix: str, question: str) -> str:
    return prefix + format_example(question)


def completions_endpoint(url: str) -> str:
    """``http://host:port`` -> ``http://host:port/v1/completions``."""
    base = url.rstrip("/")
    if base.endswith("/v1/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/completions"
    return base + "/v1/completions"


def request_payload(model: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    # No top_p: the Dynamo frontend rejects temperature and top_p together.
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stop": list(STOP_SEQUENCES),
        "stream": False,
    }


def _note(source: str, error: str) -> Dict[str, str]:
    return {"source": source, "error": error[:NOTE_CHARS]}


def parse_dataset(text: str) -> Tuple[List[Item], List[Dict[str, str]]]:
    """GSM8K JSONL ``{"question", "answer"}`` lines -> items (+ notes)."""
    items: List[Item] = []
    bad = 0
    first_problem = ""
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            bad += 1
            first_problem = first_problem or "line %d: %s" % (lineno, exc)
            continue
        fields = obj if isinstance(obj, dict) else {}
        texts = [fields.get("question"), fields.get("answer")]
        if not all(isinstance(value, str) for value in texts):
            bad += 1
            first_problem = first_problem or "line %d: no question/answer" % lineno
            continue
        answer = fields["answer"]
        items.append(Item(len(items), fields["question"], answer, gold_answer(answer)))
    notes = []
    if bad:
        notes.append(
            _note("dataset", "%d unreadable line(s) skipped; %s" % (bad, first_problem))
        )
    return items, notes


def _count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def parse_completion(body: str) -> Completion:
    """Non-streaming ``/v1/completions`` response -> first choice (ValueError)."""
    try:
        obj = json.loads(body)
    except ValueError as exc:
        raise ValueError("response is not JSON: %s" % exc) from exc
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choices")
    choice = choices[0]
    reasoning = next(
        (
            choice[name]
            for name in REASONING_FIELDS
            if isinstance(choice.get(name), str) and choice[name]
        ),
        None,
    )
    text = choice.get("text")
    if not isinstance(text, str):
        if reasoning is None:
            raise ValueError("first choice has no text")
        text = ""  # everything went to the reasoning field
    finish = choice.get("finish_reason")
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return Completion(
        text=text,
        finish_reason=finish if isinstance(finish, str) else None,
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
        reasoning=reasoning,
    )


def classify_transport_error(exc: BaseException) -> str:
    """``"timeout"`` or ``"connection"`` for an exception raised by a request."""
    reason: Any = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "timeout"
    if "timed out" in str(reason).lower():
        return "timeout"
    return "connection"


def compact_number(value: Optional[float]) -> Any:
    """18.0 -> 18 for compact JSON; other values unchanged."""
    if value is None:
        return None
    if value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


def clip_text(text: str) -> str:
    if len(text) <= SAMPLE_HEAD_CHARS + SAMPLE_TAIL_CHARS + 5:
        return text
    return text[:SAMPLE_HEAD_CHARS] + " ... " + text[-SAMPLE_TAIL_CHARS:]


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (``q`` in [0, 100]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered), max(1, math.ceil(q * len(ordered) / 100.0)))
    return ordered[rank - 1]


# --------------------------------------------------------------------------
# Host: HTTP, tools and clocks (tests substitute fakes)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def run_tool(argv: Sequence[str], timeout: float) -> Optional[ToolResult]:
    """Run an external tool; ``None`` when it is not installed."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return ToolResult(124, "", "timed out after %.0fs" % timeout)
    except OSError as exc:  # e.g. not executable
        return ToolResult(126, "", str(exc))
    return ToolResult(proc.returncode, proc.stdout, proc.stderr)


# The probe talks to a server on the cluster: never through an HTTP proxy.
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _error_body(exc: urllib.error.HTTPError) -> str:
    if exc.fp is None:
        return ""
    try:
        return exc.read().decode("utf-8", "replace")
    except (OSError, http.client.HTTPException):
        return ""
    finally:
        exc.close()


def http_post(url: str, body: bytes, timeout: float) -> Tuple[int, str]:
    """POST JSON; (status, body text). Transport failures raise OSError."""
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, _error_body(exc)


def http_get(url: str, timeout: float) -> bytes:
    """GET ``url`` (honours proxy settings); raises OSError on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass
class Host:
    """Everything the probe touches outside the process."""

    post: Callable[[str, bytes, float], Tuple[int, str]] = http_post
    get: Callable[[str, float], bytes] = http_get
    run: Callable[..., Optional[ToolResult]] = run_tool
    which: Callable[[str], Optional[str]] = shutil.which
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.time


# --------------------------------------------------------------------------
# Requests and the concurrent evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    index: int
    gold: float
    body: bytes


@dataclass(frozen=True)
class Outcome:
    index: int
    completion: Optional[Completion] = None
    error: Optional[str] = None  # error kind when there is no completion
    detail: str = ""
    latency_s: Optional[float] = None
    attempts: int = 0


def _backoff(host: Host, attempt: int, deadline: float) -> None:
    wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
    host.sleep(max(0.0, min(wait, deadline - host.clock())))


def complete_one(
    host: Host,
    endpoint: str,
    job: Job,
    *,
    timeout_s: float,
    retries: int,
    deadline: float,
) -> Outcome:
    """One question; retries connection errors and 429/502/503/504."""
    attempt = 0
    while True:
        remaining = deadline - host.clock()
        if remaining <= 0:
            return Outcome(job.index, error="deadline", attempts=attempt)
        attempt += 1
        started = host.clock()
        timeout = max(MIN_REQUEST_TIMEOUT_S, min(timeout_s, remaining))
        try:
            status, body = host.post(endpoint, job.body, timeout)
        except (OSError, http.client.HTTPException) as exc:
            kind = classify_transport_error(exc)
            if kind == "timeout" and timeout < timeout_s:
                kind = "deadline"  # cut short by the overall deadline
            if kind == "connection" and attempt <= retries:
                _backoff(host, attempt, deadline)
                continue
            return Outcome(job.index, error=kind, detail=repr(exc), attempts=attempt)
        if status in RETRY_STATUSES and attempt <= retries:
            _backoff(host, attempt, deadline)
            continue
        if not 200 <= status < 300:
            return Outcome(
                job.index, error="http_%d" % status, detail=body, attempts=attempt
            )
        try:
            completion = parse_completion(body)
        except ValueError as exc:
            return Outcome(
                job.index, error="malformed", detail=str(exc), attempts=attempt
            )
        return Outcome(
            job.index,
            completion=completion,
            latency_s=host.clock() - started,
            attempts=attempt,
        )


def evaluate(
    host: Host,
    endpoint: str,
    jobs: Sequence[Job],
    *,
    concurrency: int,
    timeout_s: float,
    retries: int,
    deadline: float,
) -> Tuple[Dict[int, Outcome], bool]:
    """Send ``jobs`` with ``concurrency`` daemon threads until done or deadline.

    Returns (outcomes by item index, deadline_hit). Threads still blocked in
    a request when the deadline passes are abandoned (daemon threads end with
    the process), so the probe never outlives ``deadline`` by much.
    """
    if not jobs:
        return {}, False
    todo: queue.Queue = queue.Queue()
    for job in jobs:
        todo.put(job)
    done: queue.Queue = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            done.put(
                complete_one(
                    host,
                    endpoint,
                    job,
                    timeout_s=timeout_s,
                    retries=retries,
                    deadline=deadline,
                )
            )

    threads = [
        threading.Thread(target=worker, name="gsm8k-%d" % i, daemon=True)
        for i in range(max(1, min(concurrency, len(jobs))))
    ]
    for thread in threads:
        thread.start()
    outcomes: Dict[int, Outcome] = {}
    deadline_hit = False
    while len(outcomes) < len(jobs):
        remaining = deadline - host.clock()
        if remaining <= 0:
            deadline_hit = True
            break
        try:
            outcome = done.get(timeout=min(POLL_S, remaining))
        except queue.Empty:
            if done.empty() and not any(t.is_alive() for t in threads):
                break  # every worker died (a bug): stop waiting
            continue
        outcomes[outcome.index] = outcome
    stop.set()
    while True:  # answers that landed while we stopped waiting still count
        try:
            outcome = done.get_nowait()
        except queue.Empty:
            break
        outcomes.setdefault(outcome.index, outcome)
    return outcomes, deadline_hit


def summarize(
    jobs: Sequence[Job], outcomes: Dict[int, Outcome], deadline_hit: bool
) -> Dict[str, Any]:
    """Scores, compact items, samples, usage and latency (pure)."""
    items: List[List[Any]] = []
    samples: List[Dict[str, Any]] = []
    error_kinds: Counter = Counter()
    finish_reasons: Counter = Counter()
    sources: Counter = Counter()
    error_details: Dict[str, str] = {}
    latencies: List[float] = []
    correct = invalid = errors = 0
    prompt_tokens = completion_tokens = 0
    usage_items = wrong_kept = right_kept = 0
    for job in sorted(jobs, key=lambda j: j.index):
        outcome = outcomes.get(job.index)
        gold = compact_number(job.gold)
        if outcome is None or outcome.completion is None:
            if outcome is None:
                kind = "deadline" if deadline_hit else "aborted"
            else:
                kind = outcome.error or "unknown"
            errors += 1
            error_kinds[kind] += 1
            if outcome is not None and outcome.detail:
                error_details.setdefault(kind, outcome.detail)
            items.append([job.index, None, None, gold])
            continue
        completion = outcome.completion
        pred, source = answer_of(completion)
        ok = answers_match(pred, job.gold)
        correct += int(ok)
        invalid += int(pred is None)
        sources[source] += 1
        finish_reasons[completion.finish_reason or "unknown"] += 1
        if outcome.latency_s is not None:
            latencies.append(outcome.latency_s)
        if completion.prompt_tokens is not None:
            usage_items += 1
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens or 0
        items.append([job.index, ok, compact_number(pred), gold])
        if (not ok and wrong_kept < WRONG_SAMPLES) or (
            ok and right_kept < RIGHT_SAMPLES
        ):
            wrong_kept += int(not ok)
            right_kept += int(ok)
            samples.append(
                {
                    "index": job.index,
                    "ok": ok,
                    "pred": compact_number(pred),
                    "gold": gold,
                    "finish_reason": completion.finish_reason,
                    "source": source,
                    "text": clip_text(completion.text),
                    "reasoning": (
                        clip_text(completion.reasoning)
                        if completion.reasoning
                        else None
                    ),
                }
            )
    answered = len(jobs) - errors
    deadline_hit = deadline_hit or error_kinds["deadline"] > 0
    p50 = percentile(latencies, 50)
    notes = [
        _note("request", "%s: %s" % (kind, " ".join(detail.split())))
        for kind, detail in sorted(error_details.items())
    ]
    indirect = [(s, sources[s]) for s in ANSWER_SOURCES[1:] if sources[s]]
    if indirect:
        listed = ", ".join("%s=%d" % pair for pair in indirect)
        notes.append(_note("answers", "answers not from the visible text: " + listed))
    return {
        "questions": len(jobs),
        "n": answered,
        "correct": correct,
        "accuracy": round(correct / answered, 6) if answered else None,
        "invalid": invalid,
        "errors": errors,
        "error_kinds": dict(sorted(error_kinds.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "deadline_hit": deadline_hit,
        "latency_p50_s": round(p50, 3) if p50 is not None else None,
        "latency_max_s": round(max(latencies), 3) if latencies else None,
        "prompt_tokens": prompt_tokens if usage_items else None,
        "completion_tokens": completion_tokens if usage_items else None,
        "usage_items": usage_items,
        "answer_sources": dict(sorted(sources.items())),
        "items": items,
        "samples": samples,
        "notes": notes,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_gsm8k(
    host: Host,
    *,
    url: str,
    model: str,
    dataset: str,
    shots: int = DEFAULT_SHOTS,
    questions: int = DEFAULT_QUESTIONS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    deadline_s: float = DEFAULT_DEADLINE_S,
    retries: int = DEFAULT_RETRIES,
    expect_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``gsm8k`` document; a fatal problem sets ``error`` instead of raising."""
    started = host.clock()
    deadline = started + deadline_s
    endpoint = completions_endpoint(url)
    notes: List[Dict[str, str]] = []
    doc: Dict[str, Any] = {
        "kind": KIND_PROBE,
        "version": VERSION,
        "benchmark": BENCHMARK,
        "model": model,
        "endpoint": endpoint,
        "dataset": dataset,
        "dataset_sha256": None,
        "dataset_items": None,
        "shots": shots,
        "first_index": shots,
        "params": {
            "shots": shots,
            "questions": questions,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "request_timeout_s": request_timeout_s,
            "deadline_s": deadline_s,
            "retries": retries,
            "temperature": TEMPERATURE,
            "stop": list(STOP_SEQUENCES),
        },
        "started_at": host.now(),
        "error": None,
    }
    doc.update(summarize([], {}, False))

    def finish(error: Optional[str] = None) -> Dict[str, Any]:
        doc["error"] = error
        doc["wall_s"] = round(host.clock() - started, 3)
        doc["notes"] = notes + list(doc["notes"])
        return doc

    if not endpoint.startswith(("http://", "https://")):
        return finish("invalid --url %r: want http(s)://host:port" % url)
    try:
        with open(dataset, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return finish("dataset unreadable: %s" % exc)
    digest = _sha256_bytes(raw)
    doc["dataset_sha256"] = digest
    if expect_sha256 and digest != expect_sha256.strip().lower():
        return finish(
            "dataset sha256 %s does not match the expected %s"
            % (digest, expect_sha256.strip().lower())
        )
    items, dataset_notes = parse_dataset(raw.decode("utf-8", "replace"))
    notes.extend(dataset_notes)
    doc["dataset_items"] = len(items)
    if len(items) <= shots:
        return finish(
            "dataset has %d item(s): need more than %d (the exemplars)"
            % (len(items), shots)
        )
    if questions > len(items) - shots:
        notes.append(
            _note(
                "dataset",
                "only %d item(s) after the exemplars; evaluating all of them"
                % (len(items) - shots),
            )
        )
    evaluated = items[shots : shots + questions]
    no_gold = [item.index for item in evaluated if item.gold is None]
    if no_gold:
        notes.append(
            _note(
                "dataset",
                "%d item(s) without a '#### <number>' reference skipped (first %d)"
                % (len(no_gold), no_gold[0]),
            )
        )
    prefix = few_shot_prefix(items[:shots])
    jobs = [
        Job(
            item.index,
            item.gold,
            json.dumps(
                request_payload(model, build_prompt(prefix, item.question), max_tokens)
            ).encode("utf-8"),
        )
        for item in evaluated
        if item.gold is not None
    ]
    outcomes, deadline_hit = evaluate(
        host,
        endpoint,
        jobs,
        concurrency=concurrency,
        timeout_s=request_timeout_s,
        retries=retries,
        deadline=deadline,
    )
    doc.update(summarize(jobs, outcomes, deadline_hit))
    return finish()


# --------------------------------------------------------------------------
# Dataset download
# --------------------------------------------------------------------------


def file_digest(path: str) -> Optional[Tuple[str, int]]:
    """(hex sha256, size in bytes) of ``path``; None when missing or unreadable."""
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return None
    return digest.hexdigest(), size


def remove_file(path: str) -> str:
    """Delete ``path`` if present; an error message ("" when fine)."""
    try:
        if os.path.lexists(path):
            os.unlink(path)
    except OSError as exc:
        return "cannot remove %s: %s" % (path, exc)
    return ""


def _download(host: Host, method: str, url: str, path: str, timeout_s: float) -> str:
    """Fetch ``url`` into ``path``; returns an error message ("" on success)."""
    if method == "curl":
        result = host.run(
            [
                "curl",
                "-fsSL",
                "--connect-timeout",
                str(CURL_CONNECT_TIMEOUT_S),
                "--max-time",
                "%d" % max(1, int(timeout_s)),
                "-o",
                path,
                url,
            ],
            timeout=timeout_s + 10.0,
        )
        if result is None:
            return "curl not found"
        if not result.ok:
            detail = " ".join((result.stderr or result.stdout).split())
            return "curl exit %d: %s" % (result.rc, detail)
        return ""
    try:
        data = host.get(url, timeout_s)
    except (OSError, http.client.HTTPException) as exc:
        return "urllib: %s" % exc
    try:
        with open(path, "wb") as fh:
            fh.write(data)
    except OSError as exc:
        return "cannot write %s: %s" % (path, exc)
    return ""


def fetch_dataset(
    host: Host,
    *,
    urls: Sequence[str],
    sha256: str,
    dest: str,
    attempts: int = DEFAULT_FETCH_ATTEMPTS,
    timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
) -> Dict[str, Any]:
    """Make ``dest`` a verified copy of the dataset (reused when it already is)."""
    started = host.clock()
    expected = sha256.strip().lower()
    notes: List[Dict[str, str]] = []
    doc: Dict[str, Any] = {
        "kind": KIND_FETCH,
        "version": VERSION,
        "ok": False,
        "path": dest,
        "url": None,
        "sha256": None,
        "expected_sha256": expected,
        "size_bytes": None,
        "reused": False,
        "attempts": 0,
        "seconds": None,
        "error": None,
        "notes": notes,
    }

    def finish(error: Optional[str] = None) -> Dict[str, Any]:
        doc["ok"] = error is None
        doc["error"] = error
        doc["seconds"] = round(host.clock() - started, 3)
        return doc

    existing = file_digest(dest)
    if existing is not None and existing[0] == expected:
        doc.update(sha256=existing[0], reused=True, size_bytes=existing[1])
        return finish()
    if existing is not None:
        notes.append(
            _note("cache", "cached copy has sha256 %s; refetching" % existing[0])
        )
    directory = os.path.dirname(os.path.abspath(dest))
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".dataset.", suffix=".part")
        os.close(fd)
    except OSError as exc:
        return finish("cannot create %s: %s" % (directory, exc))
    methods = (["curl"] if host.which("curl") else []) + ["urllib"]
    try:
        for attempt in range(1, attempts + 1):
            for url in urls:
                for method in methods:
                    doc["attempts"] += 1
                    problem = _download(host, method, url, tmp, timeout_s)
                    if problem:
                        notes.append(_note(method, "%s: %s" % (url, problem)))
                        continue
                    got = file_digest(tmp)
                    if got is None or got[0] != expected:
                        digest = got[0] if got else None
                        notes.append(_note("verify", "%s: sha256 %s" % (url, digest)))
                        break  # another method would fetch the same bytes
                    os.replace(tmp, dest)
                    doc.update(url=url, sha256=got[0], size_bytes=got[1])
                    return finish()
            if attempt < attempts:
                host.sleep(FETCH_BACKOFF_S * attempt)
    except OSError as exc:
        return finish("cannot install %s: %s" % (dest, exc))
    finally:
        problem = remove_file(tmp)
        if problem:
            notes.append(_note("cleanup", problem))
    return finish(
        "no verified copy after %d attempt(s) from %d URL(s)"
        % (doc["attempts"], len(urls))
    )


# --------------------------------------------------------------------------
# Output and command line
# --------------------------------------------------------------------------


def write_document(payload: Dict[str, Any], out: Optional[str]) -> None:
    """Atomically write ``out`` (when given), then print the JSON line."""
    if out:
        directory = os.path.dirname(os.path.abspath(out))
        tmp = None
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=directory, prefix=".accuracy.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
            os.replace(tmp, out)
        except OSError as exc:
            notes = payload.setdefault("notes", [])
            notes.append(_note("out", "cannot write %s: %s" % (out, exc)))
            problem = remove_file(tmp) if tmp is not None else ""
            if problem:
                notes.append(_note("cleanup", problem))
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    probe = sub.add_parser("gsm8k", help="few-shot GSM8K through /v1/completions")
    probe.add_argument("--url", required=True, help="server base URL")
    probe.add_argument("--model", required=True, help="served model name")
    probe.add_argument("--dataset", required=True, help="GSM8K test JSONL")
    probe.add_argument("--shots", type=_non_negative_int, default=DEFAULT_SHOTS)
    probe.add_argument("--questions", type=_positive_int, default=DEFAULT_QUESTIONS)
    probe.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY)
    probe.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS)
    probe.add_argument(
        "--request-timeout-s", type=_positive_float, default=DEFAULT_REQUEST_TIMEOUT_S
    )
    probe.add_argument("--deadline-s", type=_positive_float, default=DEFAULT_DEADLINE_S)
    probe.add_argument("--retries", type=_non_negative_int, default=DEFAULT_RETRIES)
    probe.add_argument("--expect-sha256", default=None, help="refuse other data")
    probe.add_argument("--out", default=None, help="also write the JSON here")
    fetch = sub.add_parser("fetch", help="download and verify the dataset")
    fetch.add_argument("--url", dest="urls", action="append", required=True)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--dest", required=True)
    fetch.add_argument("--attempts", type=_positive_int, default=DEFAULT_FETCH_ATTEMPTS)
    fetch.add_argument(
        "--timeout-s", type=_positive_float, default=DEFAULT_FETCH_TIMEOUT_S
    )
    fetch.add_argument("--out", default=None, help="also write the JSON here")
    return parser


def main(argv: Optional[Sequence[str]] = None, host: Optional[Host] = None) -> int:
    args = build_parser().parse_args(argv)
    host = host or Host()
    if args.mode == "fetch":
        payload = fetch_dataset(
            host,
            urls=args.urls,
            sha256=args.sha256,
            dest=args.dest,
            attempts=args.attempts,
            timeout_s=args.timeout_s,
        )
    else:
        payload = run_gsm8k(
            host,
            url=args.url,
            model=args.model,
            dataset=args.dataset,
            shots=args.shots,
            questions=args.questions,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            request_timeout_s=args.request_timeout_s,
            deadline_s=args.deadline_s,
            retries=args.retries,
            expect_sha256=args.expect_sha256,
        )
    write_document(payload, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
