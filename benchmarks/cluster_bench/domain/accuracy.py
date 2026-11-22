# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate: few-shot GSM8K through the Dynamo frontend. Pure.

Throughput runs pass ``ignore_eos``, so a numerically broken configuration
emits as many tokens as a healthy one and looks just as fast.
``agents/accuracy_probe.py`` asks GSM8K test questions through
``/v1/completions`` while a deployment is up (its module docstring is the JSON
contract); this module validates that JSON, attaches a Wilson 95% interval and
decides a verdict with ``accuracy_verdict``:

* ``error``   no usable evidence: the probe failed, answered nothing, or more
              than ``max_error_rate`` of its requests failed - unless the
              result is below the floor even if every failed request had been
              answered correctly, which is a ``fail``.
* ``fail``    accuracy below the absolute ``floor``; or more than ``max_drop``
              below the best checked configuration AND the interval excludes
              that best value (a drop that sampling noise explains never
              fails).
* ``pass``    otherwise.
* ``skipped`` the check was not attempted (``skipped_result``).

Accuracy is ``correct / n`` over the answered questions; an answer without a
number ("invalid") counts as wrong, a request that got no answer counts as an
error. Results are comparable only within one ``protocol`` (dataset, shots,
evaluated items, decoding). Reasons are English templates the report
localizes. ``synthetic_*`` give deterministic data for the dry-run simulator.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import Field, ValidationError

from benchmarks.cluster_bench.domain.base import Frozen

PROBE_KIND = "accuracy_probe"
FETCH_KIND = "dataset_fetch"
BENCHMARK = "gsm8k"

# GSM8K test split (openai/grade-school-math); sha256 verified 2026-11-06.
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GSM8K_ITEMS = 1319
GSM8K_BYTES = 749_738
GSM8K_FILENAME = "gsm8k-test.jsonl"
# SGLang few_shot_gsm8k decoding: greedy, these stop strings, 512 new tokens.
STOP_SEQUENCES = ("Question", "Assistant:", "<|separator|>")

DEFAULT_SHOTS = 5
DEFAULT_QUESTIONS = 200
DEFAULT_CONCURRENCY = 32
DEFAULT_MAX_TOKENS = 512
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_DEADLINE_S = 900.0
DEFAULT_RETRIES = 2

# Suggested gate defaults (reasoning in the report): a healthy frontier model
# scores >= 0.90 on this protocol; broken numerics typically score < 0.5.
DEFAULT_FLOOR = 0.80
DEFAULT_MAX_DROP = 0.07
DEFAULT_MAX_ERROR_RATE = 0.10

Z_95 = 1.959963984540054
# Accuracies are k/n fractions; compare with a margin so 0.94 - 0.87 is 0.07.
_EPS = 1e-9
_CHARS = 300

ResultStatus = Literal["ok", "error", "skipped"]
VerdictStatus = Literal["pass", "fail", "error", "skipped"]
# (dataset index, ok, predicted, gold); ok None = no answer, predicted None =
# no number in the answer.
Item = tuple[int, bool | None, float | None, float | None]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class ProbeNote(Frozen):
    source: str
    error: str


class AccuracyParams(Frozen):
    """Probe knobs; decoding (greedy, ``STOP_SEQUENCES``) is fixed."""

    benchmark: Literal["gsm8k"] = BENCHMARK
    shots: int = Field(DEFAULT_SHOTS, ge=0, le=16)
    questions: int = Field(DEFAULT_QUESTIONS, ge=1)
    concurrency: int = Field(DEFAULT_CONCURRENCY, ge=1, le=512)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)
    request_timeout_s: float = Field(DEFAULT_REQUEST_TIMEOUT_S, gt=0)
    deadline_s: float = Field(DEFAULT_DEADLINE_S, gt=0)
    retries: int = Field(DEFAULT_RETRIES, ge=0, le=5)


class AccuracySample(Frozen):
    """One completion kept for diagnosis (clipped text)."""

    index: int
    ok: bool
    pred: float | None = None
    gold: float | None = None
    finish_reason: str | None = None
    source: str | None = None  # text | reasoning | think | none
    text: str = ""
    reasoning: str | None = None


class AccuracyResult(Frozen):
    """One probe run against one deployment."""

    status: ResultStatus
    reason: str | None = None  # why status is "error" or "skipped"
    benchmark: str = BENCHMARK
    model: str | None = None
    endpoint: str | None = None
    dataset_sha256: str | None = None
    dataset_items: int | None = None
    shots: int = 0
    first_index: int = 0
    questions: int = 0  # evaluated (sent) questions
    n: int = 0  # answered questions
    correct: int = 0
    accuracy: float | None = None  # correct / n
    ci_low: float | None = None  # Wilson 95% interval of accuracy
    ci_high: float | None = None
    invalid: int = 0  # answered without a number (counted wrong)
    errors: int = 0  # no answer (request error, timeout, deadline)
    error_kinds: dict[str, int] = Field(default_factory=dict)
    finish_reasons: dict[str, int] = Field(default_factory=dict)
    deadline_hit: bool = False
    wall_s: float | None = None
    latency_p50_s: float | None = None
    latency_max_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usage_items: int = 0
    # Which part of the completion held the number (text | reasoning |
    # think | none), over answered questions.
    answer_sources: dict[str, int] = Field(default_factory=dict)
    started_at: float | None = None
    params: AccuracyParams | None = None
    items: tuple[Item, ...] = ()
    samples: tuple[AccuracySample, ...] = ()
    notes: tuple[ProbeNote, ...] = ()

    @property
    def error_rate(self) -> float | None:
        return self.errors / self.questions if self.questions else None

    @property
    def protocol(self) -> str:
        """Key of comparable runs: dataset, shots, evaluated items, decoding."""
        sha = (self.dataset_sha256 or "unknown")[:12]
        max_tokens = self.params.max_tokens if self.params else DEFAULT_MAX_TOKENS
        return (
            f"{self.benchmark}:{sha}:{self.shots}shot:"
            f"items{self.first_index}+{self.questions}:greedy:max{max_tokens}"
        )


class AccuracyVerdict(Frozen):
    status: VerdictStatus
    code: str  # pass | pass_within_noise | below_floor | drop_vs_best | ...
    reason: str  # English template, localized by the report
    benchmark: str = BENCHMARK
    accuracy: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0
    errors: int = 0
    floor: float | None = None
    best_accuracy: float | None = None
    max_drop: float | None = None
    drop: float | None = None  # best_accuracy - accuracy

    @property
    def passed(self) -> bool:
        return self.status == "pass"


class DatasetStatus(Frozen):
    """Outcome of ``accuracy_probe.py fetch`` (download + sha256 check)."""

    ok: bool = False
    path: str
    url: str | None = None  # where the copy came from (None when reused)
    sha256: str | None = None
    expected_sha256: str
    size_bytes: int | None = None
    reused: bool = False
    attempts: int = 0
    seconds: float | None = None
    error: str | None = None
    notes: tuple[ProbeNote, ...] = ()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval of ``k`` successes in ``n`` trials (95% default).

    ``n == 0`` carries no information: (0, 1).
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# --------------------------------------------------------------------------
# Parsing (lenient: a bad field becomes a default plus a note)
# --------------------------------------------------------------------------


def _clip(text: str) -> str:
    return text[:_CHARS]


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _opt_count(value: Any) -> int | None:
    number = _opt_int(value)
    return number if number is not None and number >= 0 else None


def _count(value: Any) -> int:
    return _opt_count(value) or 0


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts = {str(k): _opt_count(v) for k, v in value.items()}
    return {k: v for k, v in counts.items() if v is not None}


def _notes(value: Any) -> list[ProbeNote]:
    notes: list[ProbeNote] = []
    for entry in value if isinstance(value, list) else ():
        if isinstance(entry, Mapping):
            notes.append(
                ProbeNote(
                    source=str(entry.get("source", "")),
                    error=_clip(str(entry.get("error", ""))),
                )
            )
    return notes


def _params(value: Any, notes: list[ProbeNote]) -> AccuracyParams | None:
    if not isinstance(value, Mapping):
        return None
    known = {k: v for k, v in value.items() if k in AccuracyParams.model_fields}
    try:
        return AccuracyParams.model_validate(known)
    except ValidationError as exc:
        notes.append(ProbeNote(source="parse:params", error=_clip(str(exc))))
        return None


def _items(value: Any, notes: list[ProbeNote]) -> tuple[Item, ...]:
    if not isinstance(value, list):
        return ()
    items: list[Item] = []
    dropped = 0
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            dropped += 1
            continue
        index, ok, pred, gold = entry
        if _opt_int(index) is None or not (ok is None or isinstance(ok, bool)):
            dropped += 1
            continue
        items.append((int(index), ok, _opt_float(pred), _opt_float(gold)))
    if dropped:
        notes.append(
            ProbeNote(
                source="parse:items", error=f"{dropped} malformed item(s) dropped"
            )
        )
    return tuple(items)


def _samples(value: Any, notes: list[ProbeNote]) -> tuple[AccuracySample, ...]:
    if not isinstance(value, list):
        return ()
    samples: list[AccuracySample] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        known = {k: v for k, v in entry.items() if k in AccuracySample.model_fields}
        try:
            samples.append(AccuracySample.model_validate(known))
        except ValidationError as exc:
            notes.append(ProbeNote(source="parse:samples", error=_clip(str(exc))))
    return tuple(samples)


def parse_probe_document(obj: Any) -> AccuracyResult:
    """Agent ``gsm8k`` JSON -> AccuracyResult (ValueError on a wrong kind).

    Accuracy and its interval are recomputed from ``correct`` and ``n``; an
    agent-reported ``error`` makes the result ``status="error"``.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    if obj.get("kind") != PROBE_KIND:
        raise ValueError(f"expected kind={PROBE_KIND!r}, got {obj.get('kind')!r}")
    notes = _notes(obj.get("notes"))
    questions = _count(obj.get("questions"))
    n = _count(obj.get("n"))
    correct = _count(obj.get("correct"))
    errors = _count(obj.get("errors"))
    if correct > n:
        notes.append(
            ProbeNote(source="parse:counts", error=f"correct={correct} > n={n}")
        )
        correct = n
    if questions < n + errors:
        notes.append(
            ProbeNote(
                source="parse:counts",
                error=f"questions={questions} < n + errors={n + errors}",
            )
        )
        questions = n + errors
    accuracy = ci_low = ci_high = None
    if n:
        accuracy = correct / n
        ci_low, ci_high = wilson_interval(correct, n)
    reason = _opt_str(obj.get("error"))
    params = _params(obj.get("params"), notes)
    items = _items(obj.get("items"), notes)
    samples = _samples(obj.get("samples"), notes)
    return AccuracyResult(
        status="error" if reason else "ok",
        reason=_clip(reason) if reason else None,
        benchmark=_opt_str(obj.get("benchmark")) or BENCHMARK,
        model=_opt_str(obj.get("model")),
        endpoint=_opt_str(obj.get("endpoint")),
        dataset_sha256=_opt_str(obj.get("dataset_sha256")),
        dataset_items=_opt_count(obj.get("dataset_items")),
        shots=_count(obj.get("shots")),
        first_index=_count(obj.get("first_index")),
        questions=questions,
        n=n,
        correct=correct,
        accuracy=accuracy,
        ci_low=ci_low,
        ci_high=ci_high,
        invalid=_count(obj.get("invalid")),
        errors=errors,
        error_kinds=_counts(obj.get("error_kinds")),
        finish_reasons=_counts(obj.get("finish_reasons")),
        deadline_hit=obj.get("deadline_hit") is True,
        wall_s=_opt_float(obj.get("wall_s")),
        latency_p50_s=_opt_float(obj.get("latency_p50_s")),
        latency_max_s=_opt_float(obj.get("latency_max_s")),
        prompt_tokens=_opt_count(obj.get("prompt_tokens")),
        completion_tokens=_opt_count(obj.get("completion_tokens")),
        usage_items=_count(obj.get("usage_items")),
        answer_sources=_counts(obj.get("answer_sources")),
        started_at=_opt_float(obj.get("started_at")),
        params=params,
        items=items,
        samples=samples,
        notes=tuple(notes),
    )


def find_document(text: str, kind: str) -> dict[str, Any] | None:
    """The last stdout line that is a JSON object of ``kind`` (None if none)."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("kind") == kind:
            return value
    return None


def error_result(
    reason: str,
    *,
    params: AccuracyParams | None = None,
    model: str | None = None,
    endpoint: str | None = None,
) -> AccuracyResult:
    """The probe produced no usable document."""
    return AccuracyResult(
        status="error",
        reason=_clip(reason),
        model=model,
        endpoint=endpoint,
        params=params,
        notes=(ProbeNote(source="collector", error=_clip(reason)),),
    )


def skipped_result(
    reason: str, *, params: AccuracyParams | None = None, model: str | None = None
) -> AccuracyResult:
    """The check was not attempted (disabled, no dataset, no deployment, ...)."""
    return AccuracyResult(
        status="skipped", reason=_clip(reason), params=params, model=model
    )


def parse_probe_output(text: str) -> AccuracyResult:
    """Probe stdout (or the ``--out`` file) -> AccuracyResult. Never raises."""
    doc = find_document(text, PROBE_KIND)
    if doc is None:
        return error_result(f"no {PROBE_KIND} JSON document in the probe output")
    try:
        return parse_probe_document(doc)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        return error_result(f"unparsable probe output: {exc}")


def failed_dataset(path: str, expected_sha256: str, reason: str) -> DatasetStatus:
    return DatasetStatus(
        path=path, expected_sha256=expected_sha256.strip().lower(), error=_clip(reason)
    )


def parse_fetch_document(obj: Any, *, path: str, expected_sha256: str) -> DatasetStatus:
    """Agent ``fetch`` JSON -> DatasetStatus; ok only with the expected sha256."""
    if not isinstance(obj, Mapping) or obj.get("kind") != FETCH_KIND:
        raise ValueError(f"expected a {FETCH_KIND!r} JSON object")
    expected = expected_sha256.strip().lower()
    sha256 = _opt_str(obj.get("sha256"))
    error = _opt_str(obj.get("error"))
    ok = obj.get("ok") is True and sha256 == expected and error is None
    if obj.get("ok") is True and not ok and error is None:
        error = f"agent reported sha256 {sha256}, expected {expected}"
    return DatasetStatus(
        ok=ok,
        path=_opt_str(obj.get("path")) or path,
        url=_opt_str(obj.get("url")),
        sha256=sha256,
        expected_sha256=expected,
        size_bytes=_opt_count(obj.get("size_bytes")),
        reused=obj.get("reused") is True,
        attempts=_count(obj.get("attempts")),
        seconds=_opt_float(obj.get("seconds")),
        error=_clip(error) if error else None,
        notes=tuple(_notes(obj.get("notes"))),
    )


def parse_fetch_output(text: str, *, path: str, expected_sha256: str) -> DatasetStatus:
    """``fetch`` stdout -> DatasetStatus. Never raises."""
    doc = find_document(text, FETCH_KIND)
    if doc is None:
        return failed_dataset(
            path, expected_sha256, f"no {FETCH_KIND} JSON document in the output"
        )
    try:
        return parse_fetch_document(doc, path=path, expected_sha256=expected_sha256)
    except ValueError as exc:
        return failed_dataset(path, expected_sha256, f"unparsable: {exc}")


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def _check_fraction(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")


def _score_text(result: AccuracyResult, acc: float, lo: float, hi: float) -> str:
    return (
        f"{result.benchmark} {result.shots}-shot accuracy {acc:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}, n={result.n})"
    )


def accuracy_verdict(
    result: AccuracyResult,
    *,
    floor: float | None,
    best_accuracy: float | None = None,
    max_drop: float | None = None,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> AccuracyVerdict:
    """pass / fail / error / skipped for one probe result (see module doc).

    ``floor`` is an absolute minimum accuracy (None: no floor). The relative
    rule applies only when both ``best_accuracy`` (the best accuracy among
    the configurations checked so far, same ``protocol``) and ``max_drop``
    (accuracy points, 0.07 = 7 points) are given.
    """
    _check_fraction("floor", floor)
    _check_fraction("best_accuracy", best_accuracy)
    _check_fraction("max_drop", max_drop)
    _check_fraction("max_error_rate", max_error_rate)
    base: dict[str, Any] = {
        "benchmark": result.benchmark,
        "n": result.n,
        "errors": result.errors,
        "floor": floor,
        "best_accuracy": best_accuracy,
        "max_drop": max_drop,
    }
    if result.status == "skipped":
        reason = f"accuracy check skipped: {result.reason or 'not attempted'}"
        return AccuracyVerdict(status="skipped", code="skipped", reason=reason, **base)
    if result.status == "error":
        reason = f"accuracy probe failed: {result.reason or 'unknown error'}"
        return AccuracyVerdict(
            status="error", code="probe_error", reason=reason, **base
        )
    if result.n <= 0 or result.questions <= 0:
        reason = (
            f"accuracy probe: no answered questions "
            f"({result.errors} of {result.questions} requests failed)"
        )
        return AccuracyVerdict(status="error", code="no_answers", reason=reason, **base)
    correct = min(result.correct, result.n)
    acc = correct / result.n
    lo, hi = wilson_interval(correct, result.n)
    base.update(accuracy=acc, ci_low=lo, ci_high=hi)
    score = _score_text(result, acc, lo, hi)
    rate = result.errors / result.questions
    if rate > max_error_rate + _EPS:
        best_case = (correct + result.errors) / result.questions
        if floor is not None and best_case < floor - _EPS:
            reason = (
                f"{score} below floor {floor:.2f} even if all "
                f"{result.errors} failed requests were correct"
            )
            return AccuracyVerdict(
                status="fail", code="below_floor", reason=reason, **base
            )
        reason = (
            f"accuracy probe: {result.errors} of {result.questions} requests "
            f"failed (error rate {rate:.2f} > {max_error_rate:.2f})"
        )
        return AccuracyVerdict(
            status="error", code="too_many_errors", reason=reason, **base
        )
    if floor is not None and acc < floor - _EPS:
        reason = f"{score} below floor {floor:.2f}"
        return AccuracyVerdict(status="fail", code="below_floor", reason=reason, **base)
    floor_text = f"passes floor {floor:.2f}" if floor is not None else "has no floor"
    if best_accuracy is None or max_drop is None:
        return AccuracyVerdict(
            status="pass", code="pass", reason=f"{score} {floor_text}", **base
        )
    drop = best_accuracy - acc
    base["drop"] = drop
    if drop > max_drop + _EPS and best_accuracy > hi + _EPS:
        reason = (
            f"{score} is {drop:.3f} below the best checked {best_accuracy:.3f} "
            f"(max drop {max_drop:.2f})"
        )
        return AccuracyVerdict(
            status="fail", code="drop_vs_best", reason=reason, **base
        )
    if drop > max_drop + _EPS:
        reason = (
            f"{score} {floor_text}; {drop:.3f} below the best checked "
            f"{best_accuracy:.3f} but within the 95% CI"
        )
        return AccuracyVerdict(
            status="pass", code="pass_within_noise", reason=reason, **base
        )
    reason = (
        f"{score} {floor_text}; within {max_drop:.2f} of the best checked "
        f"{best_accuracy:.3f}"
    )
    return AccuracyVerdict(status="pass", code="pass", reason=reason, **base)


def best_accuracy_of(
    results: Iterable[AccuracyResult], *, protocol: str | None = None
) -> float | None:
    """Highest accuracy among usable results (optionally of one ``protocol``)."""
    usable = [r for r in results if r.status == "ok" and r.accuracy is not None]
    values = [
        r.accuracy
        for r in usable
        if r.accuracy is not None and (protocol is None or r.protocol == protocol)
    ]
    return max(values) if values else None


# --------------------------------------------------------------------------
# Synthetic data (dry-run simulator)
# --------------------------------------------------------------------------

SYNTH_MODEL = "glm-5.3"
SYNTH_ACCURACY = 0.94
SYNTH_ENDPOINT = "http://127.0.0.1:8000/v1/completions"
SYNTH_DATASET = f"/root/cluster_bench/datasets/{GSM8K_FILENAME}"
SYNTH_PROMPT_TOKENS = 720
SYNTH_COMPLETION_TOKENS = 115


def _synth_gold(index: int) -> int:
    return (index * 37) % 500 + 3


def synthetic_probe_document(
    *,
    accuracy: float = SYNTH_ACCURACY,
    params: AccuracyParams | None = None,
    model: str = SYNTH_MODEL,
    errors: int = 0,
    endpoint: str = SYNTH_ENDPOINT,
    dataset: str = SYNTH_DATASET,
) -> dict[str, Any]:
    """A deterministic agent ``gsm8k`` document (wrong answers spread evenly)."""
    _check_fraction("accuracy", accuracy)
    params = params or AccuracyParams()
    shots = params.shots
    questions = max(0, min(params.questions, GSM8K_ITEMS - shots))
    errors = max(0, min(errors, questions))
    answered = questions - errors
    correct = max(0, min(answered, round(accuracy * answered)))
    wrong = answered - correct
    items: list[list[Any]] = []
    for j in range(questions):
        index = shots + j
        gold = _synth_gold(index)
        if j >= answered:
            items.append([index, None, None, gold])
            continue
        is_wrong = (j + 1) * wrong // answered > j * wrong // answered
        items.append([index, not is_wrong, gold + 1 if is_wrong else gold, gold])
    wrong_items = [i for i in items if i[1] is False][:3]
    kept = wrong_items + [i for i in items if i[1] is True][:1]
    samples = [
        {
            "index": item[0],
            "ok": item[1],
            "pred": item[2],
            "gold": item[3],
            "finish_reason": "stop",
            "source": "text",
            "text": f" So the answer is {item[2]}.\n#### {item[2]}\n\n",
            "reasoning": None,
        }
        for item in kept
    ]
    return {
        "kind": PROBE_KIND,
        "version": 1,
        "benchmark": BENCHMARK,
        "model": model,
        "endpoint": endpoint,
        "dataset": dataset,
        "dataset_sha256": GSM8K_SHA256,
        "dataset_items": GSM8K_ITEMS,
        "shots": shots,
        "first_index": shots,
        "questions": questions,
        "n": answered,
        "correct": correct,
        "accuracy": round(correct / answered, 6) if answered else None,
        "invalid": 0,
        "errors": errors,
        "error_kinds": {"timeout": errors} if errors else {},
        "finish_reasons": {"stop": answered} if answered else {},
        "deadline_hit": False,
        "wall_s": round(4.5 * math.ceil(questions / params.concurrency) + 3.0, 3),
        "latency_p50_s": 4.2 if answered else None,
        "latency_max_s": 11.8 if answered else None,
        "prompt_tokens": SYNTH_PROMPT_TOKENS * answered if answered else None,
        "completion_tokens": SYNTH_COMPLETION_TOKENS * answered if answered else None,
        "usage_items": answered,
        "answer_sources": {"text": answered} if answered else {},
        "params": {
            **params.model_dump(exclude={"benchmark"}),
            "temperature": 0.0,
            "stop": list(STOP_SEQUENCES),
        },
        "items": items,
        "samples": samples,
        "started_at": 1_800_000_000.0,
        "error": None,
        "notes": [],
    }


def synthetic_result(
    *,
    accuracy: float = SYNTH_ACCURACY,
    params: AccuracyParams | None = None,
    model: str = SYNTH_MODEL,
    errors: int = 0,
) -> AccuracyResult:
    """``synthetic_probe_document`` parsed like a real one."""
    return parse_probe_document(
        synthetic_probe_document(
            accuracy=accuracy, params=params, model=model, errors=errors
        )
    )


def synthetic_fetch_document(
    path: str, *, sha256: str = GSM8K_SHA256, reused: bool = False
) -> dict[str, Any]:
    """A successful agent ``fetch`` document."""
    return {
        "kind": FETCH_KIND,
        "version": 1,
        "ok": True,
        "path": path,
        "url": None if reused else GSM8K_URL,
        "sha256": sha256,
        "expected_sha256": sha256,
        "size_bytes": GSM8K_BYTES,
        "reused": reused,
        "attempts": 0 if reused else 1,
        "seconds": 0.002 if reused else 0.8,
        "error": None,
        "notes": [],
    }
