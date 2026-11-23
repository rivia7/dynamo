# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase pipeline with crash-safe resume.

Phases run in a fixed order; a phase that recorded PHASE_DONE is skipped on
resume. Planner-driven phases (search, pilot, NIC sweep, confirm) re-derive
their next action from the event log every iteration, so a restart picks up
exactly where the log ends. SIGTERM/SIGINT tear the deployment down and exit
with 75 (EX_TEMPFAIL) so the controller knows the run can be resumed.
"""

from __future__ import annotations

import logging
import signal
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from benchmarks.cluster_bench.app.accuracy_ops import AccuracyGate
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.deployment import DeploymentSession
from benchmarks.cluster_bench.app.history import build_history, make_budget
from benchmarks.cluster_bench.app.measure import Measurer
from benchmarks.cluster_bench.app.store import RunLock
from benchmarks.cluster_bench.domain.schema import EventType
from benchmarks.cluster_bench.domain.search import (
    Action,
    Budget,
    CheckAccuracy,
    CloseFamily,
    Done,
    History,
    RunPoint,
    SearchContext,
)

logger = logging.getLogger(__name__)

PHASES = (
    "preflight",
    "download",
    "fabric",
    "microbench",
    "search",
    "accuracy",
    "pilot",
    "nic_sweep",
    "confirm",
    "nvlink",
    "diagnose",
    "finalize",
)
# Phases a finished experiment can run again (``run --redo``): the redo also
# re-runs every phase after the first one named.
REDO_PHASES = ("diagnose", "finalize")
EXIT_RESUMABLE = 75
HEARTBEAT_S = 30.0
MAX_ISSUES_PER_POINT = 2


def rerun_phases(redo: Sequence[str]) -> frozenset[str]:
    """Phases a redo runs although they are done: the named ones and every
    phase after the first of them."""
    unknown = sorted(set(redo) - set(REDO_PHASES))
    if unknown:
        raise ValueError(f"cannot redo {unknown}: only {list(REDO_PHASES)}")
    if not redo:
        return frozenset()
    first = min(PHASES.index(p) for p in redo)
    return frozenset(PHASES[first:])


class Interrupted(BaseException):
    """SIGTERM/SIGINT; a BaseException so ``except Exception`` blocks
    (e.g. inside fan-outs) cannot swallow it."""


class AccuracyChecker(Protocol):
    """Runs one accuracy check and appends its ACCURACY event."""

    def check(self, action: CheckAccuracy) -> None:
        ...


@dataclass
class PhaseRuntime:
    session: DeploymentSession
    measurer: Measurer
    plan: SearchContext | None = None  # the running phase's search context
    checker: AccuracyChecker | None = None


class Phase(Protocol):
    name: str

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        ...


Decider = Callable[[History, Budget], Action]


def drive(ctx: RunContext, rt: PhaseRuntime, phase: str, decide: Decider) -> str:
    """Execute planner actions until the planner says Done; returns its reason.

    A correct planner issues each point once (twice when a deploy failure is
    retried). A point issued more often means the planner is stuck - e.g. it
    keeps asking for a deployment that keeps failing - so the phase ends
    instead of spending the remaining budget on identical retries.
    """
    issued: dict[str, int] = {}
    while True:
        history = build_history(ctx.store, rt.session.deployed_key)
        budget = make_budget(
            ctx.store,
            history,
            ctx.config.budget,
            ctx.config.bench_specs,
            phase,
            rt.plan,
            ctx.clock.now(),
            accuracy=ctx.config.accuracy.enabled
            and ctx.config.stage == "e2e"
            and rt.checker is not None,
            diagnose_s=ctx.config.diagnose.window_s,
        )
        ctx.status(
            phase=phase,
            budget={
                "total_s": round(ctx.config.budget.total_hours * 3600),
                "spent_s": round(
                    ctx.config.budget.total_hours * 3600 - budget.remaining_s
                ),
                "remaining_s": round(budget.remaining_s),
                "reserve_s": round(budget.reserve_s),
                "deploy_estimate_s": round(budget.deploy_s),
            },
            points_done=len(history.points),
        )
        action = decide(history, budget)
        if isinstance(action, Done):
            ctx.store.append(
                EventType.DECISION,
                {"kind": "phase_result", "phase": phase, "reason": action.reason},
            )
            return action.reason
        if isinstance(action, CloseFamily):
            ctx.store.append(
                EventType.DECISION,
                {
                    "kind": "close_family",
                    "family_id": action.family_id,
                    "status": action.status,
                    "reason": action.reason,
                },
            )
            continue
        if isinstance(action, CheckAccuracy):
            issued[action.key] = issued.get(action.key, 0) + 1
            if rt.checker is None or issued[action.key] > MAX_ISSUES_PER_POINT:
                reason = (
                    "accuracy checks disabled"
                    if rt.checker is None
                    else f"planner stalled: accuracy check of {action.candidate.key} repeated"
                )
                ctx.store.append(
                    EventType.DECISION,
                    {"kind": "phase_result", "phase": phase, "reason": reason},
                )
                return reason
            ctx.store.append(
                EventType.DECISION,
                {
                    "kind": "check_accuracy",
                    "phase": phase,
                    "cand_key": action.candidate.key,
                    "reason": action.reason,
                },
            )
            rt.checker.check(action)
            continue
        if not isinstance(action, RunPoint):
            raise TypeError(f"unknown planner action {action!r}")
        issued[action.key.id] = issued.get(action.key.id, 0) + 1
        if issued[action.key.id] > MAX_ISSUES_PER_POINT:
            reason = f"planner stalled: {action.key.cand_key} c={action.key.concurrency} issued {issued[action.key.id] - 1} times"
            ctx.note(reason)
            ctx.store.append(
                EventType.DECISION,
                {"kind": "phase_result", "phase": phase, "reason": reason},
            )
            return reason
        ctx.store.append(
            EventType.DECISION,
            {
                "kind": "run_point",
                "phase": phase,
                "point_id": action.key.id,
                "key": action.key.model_dump(),
                "reason": action.reason,
            },
        )
        rt.measurer.run(action)


class Pipeline:
    def __init__(
        self,
        ctx: RunContext,
        phases: dict[str, Phase],
        restore: Callable[[RunContext], None],
    ):
        missing = [p for p in PHASES if p not in phases]
        if missing:
            raise ValueError(f"pipeline is missing phases {missing}")
        self.ctx = ctx
        self.phases = phases
        self.restore = restore
        self.session = DeploymentSession(ctx)
        self.runtime = PhaseRuntime(
            session=self.session,
            measurer=Measurer(ctx, self.session),
            checker=AccuracyGate(ctx, self.session),
        )
        self._stop = threading.Event()

    def done_phases(self) -> set[str]:
        return {e.data["phase"] for e in self.ctx.store.of_type(EventType.PHASE_DONE)}

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self.ctx.status()

    def _on_signal(self, signum: int, _frame: object) -> None:
        raise Interrupted(f"signal {signum}")

    def run(self, only: tuple[str, ...] = (), redo: tuple[str, ...] = ()) -> int:
        ctx = self.ctx
        rerun = rerun_phases(redo)
        with RunLock(ctx.exp_dir / ".runner.lock"):
            ctx.store.append(
                EventType.RUN_START, {"config": ctx.config.model_dump(mode="json")}
            )
            previous = {
                sig: signal.signal(sig, self._on_signal)
                for sig in (signal.SIGTERM, signal.SIGINT)
            }
            beat = threading.Thread(
                target=self._heartbeat, name="heartbeat", daemon=True
            )
            beat.start()
            try:
                self.restore(ctx)
                done = self.done_phases()
                if rerun & done:
                    ctx.note(
                        "re-ran the finished phase(s) "
                        + ", ".join(p for p in PHASES if p in rerun & done)
                        + " (--redo)"
                    )
                for name in PHASES:
                    if (name in done and name not in rerun) or (
                        only and name not in only
                    ):
                        continue
                    ctx.store.append(EventType.PHASE_START, {"phase": name})
                    ctx.status(state="running", phase=name, step="")
                    self.phases[name].run(ctx, self.runtime)
                    ctx.store.append(EventType.PHASE_DONE, {"phase": name})
                ctx.store.append(EventType.RUN_DONE, {})
                ctx.status(state="done", step="")
                return 0
            except Interrupted as exc:
                logger.warning("interrupted (%s); tearing down, resumable", exc)
                self.session.teardown()
                ctx.status(state="interrupted", error=str(exc))
                return EXIT_RESUMABLE
            except Exception as exc:
                logger.error("run failed: %s", exc)
                ctx.status(
                    state="failed",
                    error=ctx.redactor(f"{exc}\n{traceback.format_exc()[-4000:]}"),
                )
                self.session.teardown()
                raise
            finally:
                self._stop.set()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
