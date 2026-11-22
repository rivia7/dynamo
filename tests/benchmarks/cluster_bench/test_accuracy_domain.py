# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""domain/accuracy.py: Wilson intervals, lenient parsing of the probe and
fetch documents, the verdict rules and the synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.cluster_bench.agents import accuracy_probe as agent
from benchmarks.cluster_bench.domain import accuracy as acc

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "accuracy"
PATH = "/root/cluster_bench/datasets/gsm8k-test.jsonl"
FIXTURE_SHA = "f756f993b1827e3afd2668372c2dbd7abda68b1cc43ca7e8d773bbb2e48c08f1"


def probe_doc() -> dict[str, Any]:
    """A real agent document (11 synthetic questions, see the fixture)."""
    return json.loads((FIXTURES / "probe_output.json").read_text())


def scored(correct: int, n: int, errors: int = 0) -> acc.AccuracyResult:
    return acc.parse_probe_document(
        {
            "kind": "accuracy_probe",
            "shots": 5,
            "first_index": 5,
            "questions": n + errors,
            "n": n,
            "correct": correct,
            "errors": errors,
        }
    )


# --------------------------------------------------------------------------
# Constants shared with the agent
# --------------------------------------------------------------------------


def test_domain_and_agent_agree_on_the_protocol() -> None:
    assert acc.STOP_SEQUENCES == agent.STOP_SEQUENCES
    assert (acc.PROBE_KIND, acc.FETCH_KIND) == (agent.KIND_PROBE, agent.KIND_FETCH)
    params = acc.AccuracyParams()
    assert params.shots == agent.DEFAULT_SHOTS
    assert params.questions == agent.DEFAULT_QUESTIONS
    assert params.concurrency == agent.DEFAULT_CONCURRENCY
    assert params.max_tokens == agent.DEFAULT_MAX_TOKENS
    assert params.request_timeout_s == agent.DEFAULT_REQUEST_TIMEOUT_S
    assert params.deadline_s == agent.DEFAULT_DEADLINE_S
    assert params.retries == agent.DEFAULT_RETRIES
    assert acc.GSM8K_URL.endswith("/grade_school_math/data/test.jsonl")
    assert len(acc.GSM8K_SHA256) == 64


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (5, 10, 0.2366, 0.7634),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (188, 200, 0.8981, 0.9653),
        (0, 0, 0.0, 1.0),
    ],
)
def test_wilson_interval_reference_values(
    k: int, n: int, low: float, high: float
) -> None:
    lo, hi = acc.wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=1e-4) and hi == pytest.approx(high, abs=1e-4)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_narrows_and_validates() -> None:
    small = acc.wilson_interval(93, 100)
    large = acc.wilson_interval(930, 1000)
    assert small[0] < large[0] < 0.93 < large[1] < small[1]
    for k, n in ((3, 2), (-1, 5), (1, -1)):
        with pytest.raises(ValueError):
            acc.wilson_interval(k, n)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_probe_document_from_the_agent() -> None:
    result = acc.parse_probe_document(probe_doc())
    assert result.status == "ok" and result.reason is None
    assert (result.questions, result.n, result.correct) == (11, 10, 8)
    assert (result.invalid, result.errors) == (0, 1)
    assert result.accuracy == 0.8
    assert (result.ci_low, result.ci_high) == acc.wilson_interval(8, 10)
    assert result.error_kinds == {"http_503": 1}
    assert result.answer_sources == {"reasoning": 1, "text": 9}
    assert result.finish_reasons == {"length": 1, "stop": 9}
    assert result.params == acc.AccuracyParams(questions=11, concurrency=4, retries=1)
    assert result.items[0] == (5, True, 350.0, 350.0)
    assert result.items[2] == (7, False, 3.0, -3.0)
    assert result.items[5] == (10, None, None, 163.0)
    assert [s.index for s in result.samples] == [5, 7, 12]
    assert result.samples[2].text.startswith(" 9 9 9") and not result.samples[2].ok
    assert (result.prompt_tokens, result.completion_tokens) == (8 * 612 + 588, 840)
    assert result.usage_items == 9 and result.model == "glm-5.3"
    assert result.dataset_sha256 == FIXTURE_SHA and result.dataset_items == 16
    assert result.protocol == f"gsm8k:{FIXTURE_SHA[:12]}:5shot:items5+11:greedy:max512"
    assert result.error_rate == pytest.approx(1 / 11)


def test_results_and_verdicts_round_trip_through_json() -> None:
    result = acc.parse_probe_document(probe_doc())
    assert acc.AccuracyResult.model_validate_json(result.model_dump_json()) == result
    verdict = acc.accuracy_verdict(result, floor=0.5, best_accuracy=0.9, max_drop=0.07)
    assert acc.AccuracyVerdict.model_validate_json(verdict.model_dump_json()) == verdict
    status = acc.parse_fetch_document(
        acc.synthetic_fetch_document(PATH), path=PATH, expected_sha256=acc.GSM8K_SHA256
    )
    assert acc.DatasetStatus.model_validate_json(status.model_dump_json()) == status


def test_parse_probe_document_is_lenient() -> None:
    doc = probe_doc()
    doc.update(
        items=[[5, True, 350, 350], "junk", [6, "yes", 1, 1], [7, None, None, 3], [8]],
        samples=[{"index": "x"}, {"index": 9, "ok": True, "text": "t", "extra": 1}],
        params={"shots": -1},
        error_kinds={"timeout": "3", "http_500": 2, "deadline": -1},
        correct=99,
        questions=4,
        notes=[{"source": "dataset", "error": "x" * 1000}, "junk"],
        wall_s="fast",
        prompt_tokens=-5,
        deadline_hit="yes",
    )
    result = acc.parse_probe_document(doc)
    assert result.items == ((5, True, 350.0, 350.0), (7, None, None, 3.0))
    assert [s.index for s in result.samples] == [9]
    assert result.params is None and result.correct == result.n == 10
    assert result.questions == 11  # n + errors
    assert result.error_kinds == {"http_500": 2}
    assert result.wall_s is None and result.prompt_tokens is None
    assert result.deadline_hit is False
    sources = {note.source for note in result.notes}
    assert {"parse:items", "parse:samples", "parse:params", "parse:counts"} <= sources
    assert all(len(note.error) <= 300 for note in result.notes)


@pytest.mark.parametrize("obj", [{"kind": "dataset_fetch"}, [], "text", None])
def test_parse_probe_document_rejects_other_documents(obj: Any) -> None:
    with pytest.raises(ValueError):
        acc.parse_probe_document(obj)


def test_parse_probe_output_never_raises() -> None:
    doc = probe_doc()
    text = "\n".join(
        ["CUDA banner", json.dumps({"kind": "other"}), json.dumps(doc), "{broken"]
    )
    assert acc.parse_probe_output(text).n == 10
    garbage = acc.parse_probe_output("Traceback (most recent call last):\n  ...\n")
    assert garbage.status == "error" and garbage.n == 0
    assert "no accuracy_probe JSON document" in (garbage.reason or "")
    assert garbage.notes[0].source == "collector"
    failed = acc.parse_probe_output(json.dumps({**doc, "error": "dataset unreadable"}))
    assert failed.status == "error" and failed.reason == "dataset unreadable"
    odd = acc.parse_probe_output(json.dumps({**doc, "params": {"retries": 99}}))
    assert odd.status == "ok" and odd.params is None


def test_error_and_skipped_results() -> None:
    params = acc.AccuracyParams(questions=50)
    failed = acc.error_result("exit 124: [timeout]", params=params, model="m")
    assert failed.status == "error" and failed.params == params and failed.n == 0
    skipped = acc.skipped_result("dataset unavailable")
    assert skipped.status == "skipped" and skipped.reason == "dataset unavailable"
    assert acc.error_result("x" * 1000).reason == "x" * 300


def test_dataset_status_parsing() -> None:
    doc = acc.synthetic_fetch_document(PATH)
    status = acc.parse_fetch_document(doc, path=PATH, expected_sha256=acc.GSM8K_SHA256)
    assert status.ok and status.size_bytes == acc.GSM8K_BYTES and not status.reused
    assert status.url == acc.GSM8K_URL and status.path == PATH
    wrong = acc.parse_fetch_document(
        {**doc, "sha256": "0" * 64}, path=PATH, expected_sha256=acc.GSM8K_SHA256
    )
    assert not wrong.ok and "agent reported sha256" in (wrong.error or "")
    failed = acc.parse_fetch_document(
        {**doc, "ok": False, "error": "no verified copy after 3 attempt(s)"},
        path=PATH,
        expected_sha256=acc.GSM8K_SHA256.upper(),
    )
    assert not failed.ok and failed.expected_sha256 == acc.GSM8K_SHA256
    assert acc.parse_fetch_output(
        json.dumps(doc), path=PATH, expected_sha256=acc.GSM8K_SHA256
    ).ok
    garbage = acc.parse_fetch_output(
        "curl: not found", path=PATH, expected_sha256=acc.GSM8K_SHA256
    )
    assert not garbage.ok and garbage.path == PATH
    with pytest.raises(ValueError):
        acc.parse_fetch_document(
            {"kind": "accuracy_probe"}, path=PATH, expected_sha256=acc.GSM8K_SHA256
        )


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

GATE = {"floor": 0.80, "best_accuracy": 0.95, "max_drop": 0.07}


@pytest.mark.parametrize(
    ("correct", "n", "errors", "kwargs", "status", "code"),
    [
        (188, 200, 0, {"floor": 0.80}, "pass", "pass"),
        (188, 200, 0, {"floor": None}, "pass", "pass"),
        (160, 200, 0, {"floor": 0.80}, "pass", "pass"),  # exactly the floor
        (150, 200, 0, {"floor": 0.80}, "fail", "below_floor"),
        (40, 200, 0, {"floor": 0.80}, "fail", "below_floor"),  # garbage output
        (180, 200, 0, GATE, "pass", "pass"),  # 0.05 below the best
        (166, 200, 0, GATE, "fail", "drop_vs_best"),  # 0.12 below, CI < 0.95
        (174, 200, 0, {**GATE, "best_accuracy": 0.94}, "pass", "pass"),  # == 0.07
        # 0.09 below, but n=20 is too few to exclude 0.94: noise, not a fail.
        (17, 20, 0, {**GATE, "best_accuracy": 0.94}, "pass", "pass_within_noise"),
        (150, 150, 50, {"floor": 0.80}, "error", "too_many_errors"),
        # Below the floor even if every failed request had been right.
        (10, 150, 50, {"floor": 0.80}, "fail", "below_floor"),
        (180, 190, 10, {"floor": 0.80}, "pass", "pass"),  # 5% errors
        (180, 180, 20, {"floor": 0.80}, "pass", "pass"),  # exactly 10% errors
        (0, 0, 200, {"floor": 0.80}, "error", "no_answers"),
    ],
)
def test_accuracy_verdict(
    correct: int, n: int, errors: int, kwargs: dict[str, Any], status: str, code: str
) -> None:
    verdict = acc.accuracy_verdict(scored(correct, n, errors), **kwargs)
    assert (verdict.status, verdict.code) == (status, code), verdict.reason
    assert verdict.passed == (status == "pass")
    if n:
        assert verdict.accuracy == pytest.approx(correct / n)
        assert verdict.ci_low is not None and verdict.ci_low <= verdict.accuracy
    if kwargs.get("best_accuracy") is not None and status != "error":
        assert verdict.drop == pytest.approx(kwargs["best_accuracy"] - correct / n)


def test_verdict_reasons_are_stable_templates() -> None:
    def reason(result: acc.AccuracyResult, **kwargs: Any) -> str:
        return acc.accuracy_verdict(result, **{"floor": 0.80, **kwargs}).reason

    assert reason(scored(188, 200)) == (
        "gsm8k 5-shot accuracy 0.940 (95% CI 0.898-0.965, n=200) passes floor 0.80"
    )
    assert reason(scored(150, 200)) == (
        "gsm8k 5-shot accuracy 0.750 (95% CI 0.686-0.805, n=200) below floor 0.80"
    )
    assert reason(scored(166, 200), best_accuracy=0.95, max_drop=0.07) == (
        "gsm8k 5-shot accuracy 0.830 (95% CI 0.772-0.876, n=200) is 0.120 below "
        "the best checked 0.950 (max drop 0.07)"
    )
    assert reason(scored(17, 20), best_accuracy=0.94, max_drop=0.07) == (
        "gsm8k 5-shot accuracy 0.850 (95% CI 0.640-0.948, n=20) passes floor 0.80; "
        "0.090 below the best checked 0.940 but within the 95% CI"
    )
    assert reason(scored(180, 200), best_accuracy=0.95, max_drop=0.07) == (
        "gsm8k 5-shot accuracy 0.900 (95% CI 0.851-0.934, n=200) passes floor 0.80; "
        "within 0.07 of the best checked 0.950"
    )
    assert reason(scored(10, 150, 50)) == (
        "gsm8k 5-shot accuracy 0.067 (95% CI 0.037-0.118, n=150) below floor 0.80 "
        "even if all 50 failed requests were correct"
    )
    assert reason(scored(150, 150, 50)) == (
        "accuracy probe: 50 of 200 requests failed (error rate 0.25 > 0.10)"
    )
    assert reason(scored(0, 0, 200)) == (
        "accuracy probe: no answered questions (200 of 200 requests failed)"
    )
    assert reason(acc.error_result("exit 124: [timeout]")) == (
        "accuracy probe failed: exit 124: [timeout]"
    )
    assert reason(acc.skipped_result("dataset unavailable")) == (
        "accuracy check skipped: dataset unavailable"
    )
    assert reason(scored(188, 200), floor=None) == (
        "gsm8k 5-shot accuracy 0.940 (95% CI 0.898-0.965, n=200) has no floor"
    )


def test_verdict_uses_counts_not_stale_derived_fields() -> None:
    stale = scored(100, 200).model_copy(update={"accuracy": 0.99, "ci_high": 1.0})
    verdict = acc.accuracy_verdict(stale, floor=0.80)
    assert verdict.status == "fail" and verdict.accuracy == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"floor": 1.5},
        {"floor": 0.8, "best_accuracy": -0.1, "max_drop": 0.07},
        {"floor": 0.8, "max_drop": 2.0, "best_accuracy": 0.9},
        {"floor": 0.8, "max_error_rate": -0.5},
    ],
)
def test_verdict_rejects_bad_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        acc.accuracy_verdict(scored(188, 200), **kwargs)


def test_best_accuracy_of_comparable_results() -> None:
    five_shot = scored(180, 200)
    eight_shot = acc.parse_probe_document(
        {
            "kind": "accuracy_probe",
            "shots": 8,
            "questions": 200,
            "n": 200,
            "correct": 190,
        }
    )
    others = [acc.error_result("boom"), acc.skipped_result("off"), scored(0, 0, 5)]
    assert acc.best_accuracy_of([five_shot, eight_shot, *others]) == 0.95
    same_protocol = acc.best_accuracy_of(
        [five_shot, eight_shot], protocol=five_shot.protocol
    )
    assert same_protocol == 0.9
    assert acc.best_accuracy_of(others) is None and acc.best_accuracy_of([]) is None


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------


def test_synthetic_result_is_deterministic_and_parsed_like_a_real_one() -> None:
    assert acc.synthetic_probe_document() == acc.synthetic_probe_document()
    result = acc.synthetic_result()
    assert result.status == "ok" and (result.questions, result.n) == (200, 200)
    assert result.correct == 188 and result.dataset_sha256 == acc.GSM8K_SHA256
    assert sum(1 for item in result.items if item[1] is True) == 188
    # Wrong answers are spread evenly, not bunched at one end.
    assert sum(1 for item in result.items[:100] if item[1] is False) == 6
    assert [s.ok for s in result.samples] == [False, False, False, True]
    assert result.answer_sources == {"text": 200}
    assert acc.accuracy_verdict(result, floor=acc.DEFAULT_FLOOR).passed
    broken = acc.synthetic_result(
        accuracy=0.3, params=acc.AccuracyParams(questions=50), errors=5
    )
    assert (broken.questions, broken.n, broken.errors) == (50, 45, 5)
    assert broken.correct == round(0.3 * 45) and broken.error_kinds == {"timeout": 5}
    assert acc.accuracy_verdict(broken, floor=acc.DEFAULT_FLOOR).status == "fail"
    tiny = acc.synthetic_result(params=acc.AccuracyParams(shots=5, questions=5000))
    assert tiny.questions == acc.GSM8K_ITEMS - 5
