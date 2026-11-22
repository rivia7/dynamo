# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planner-driven phase loop guards and launch-script syntax."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from benchmarks.lium_cluster_bench.app.pipeline import drive
from benchmarks.lium_cluster_bench.app.simulate import SimClock
from benchmarks.lium_cluster_bench.app.store import EventStore
from benchmarks.lium_cluster_bench.domain.schema import Candidate, EventType, RunConfig
from benchmarks.lium_cluster_bench.domain.search import RunPoint

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

LAUNCH_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/lium_cluster_bench/scripts/launch_node.sh"
)


@dataclass
class _Session:
    deployed_key: str | None = None


@dataclass
class _Measurer:
    calls: list[str] = field(default_factory=list)

    def run(self, action: RunPoint) -> None:
        self.calls.append(action.key.id)  # e.g. a deploy that keeps failing


@dataclass
class _Runtime:
    session: _Session = field(default_factory=_Session)
    measurer: _Measurer = field(default_factory=_Measurer)
    plan: object = None
    checker: object = None


@dataclass
class _Checker:
    store: EventStore
    calls: list = field(default_factory=list)

    def check(self, action) -> None:
        self.calls.append(action.candidate.key)
        self.store.append(
            EventType.ACCURACY,
            {
                "cand_key": action.candidate.key,
                "result": {},
                "verdict": {"status": "fail", "reason": "below the floor"},
            },
        )


@dataclass
class _Ctx:
    store: EventStore
    exp_dir: Path
    clock: SimClock = field(default_factory=SimClock)
    config: RunConfig = field(
        default_factory=lambda: RunConfig(exp_id="t", image="i", workdir="/w")
    )
    statuses: list[dict] = field(default_factory=list)

    def status(self, **fields) -> None:
        self.statuses.append(fields)

    def note(self, text: str, **data) -> None:
        self.store.append(EventType.NOTE, {"text": text, **data})


def test_drive_ends_a_phase_whose_planner_repeats_one_point(tmp_path):
    ctx = _Ctx(
        store=EventStore(tmp_path / "events.jsonl", now=SimClock().now),
        exp_dir=tmp_path,
    )
    rt = _Runtime()
    same = RunPoint(Candidate(family_id="agg-tp8"), 64, "screen", "search")
    reason = drive(ctx, rt, "search", lambda h, b: same)
    assert reason.startswith("planner stalled")
    assert len(rt.measurer.calls) == 2  # one retry, then the phase ends
    notes = [e.data["text"] for e in ctx.store.of_type(EventType.NOTE)]
    assert any("planner stalled" in n for n in notes)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_launch_node_script_parses():
    proc = subprocess.run(
        ["bash", "-n", str(LAUNCH_SCRIPT)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr


def test_drive_dispatches_accuracy_checks_and_history_sees_the_verdict(tmp_path):
    from benchmarks.lium_cluster_bench.app.history import build_history
    from benchmarks.lium_cluster_bench.domain.search import CheckAccuracy, Done

    ctx = _Ctx(
        store=EventStore(tmp_path / "events.jsonl", now=SimClock().now),
        exp_dir=tmp_path,
    )
    cand = Candidate(family_id="agg-tp8")
    rt = _Runtime(checker=_Checker(ctx.store))

    def planner(history, _budget):
        if cand.key not in history.accuracy:
            return CheckAccuracy(cand, reason="gate")
        return Done("checked")

    assert drive(ctx, rt, "accuracy", planner) == "checked"
    assert rt.checker.calls == [cand.key]
    history = build_history(ctx.store)
    assert history.accuracy == {cand.key: "fail"}
    assert history.accuracy_failed(cand.key)
    # without a checker the phase ends instead of looping
    rt2 = _Runtime()
    ctx2 = _Ctx(
        store=EventStore(tmp_path / "events2.jsonl", now=SimClock().now),
        exp_dir=tmp_path,
    )
    assert drive(ctx2, rt2, "accuracy", planner) == "accuracy checks disabled"
