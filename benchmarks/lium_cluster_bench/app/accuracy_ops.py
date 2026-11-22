# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate execution: the ``CheckAccuracy`` planner action.

Deploys the candidate (usually already running), makes sure the sha256-pinned
GSM8K file is on node0, runs ``agents/accuracy_probe.py`` against the Dynamo
frontend and appends an ACCURACY event with the result and its verdict. An
inconclusive probe (``error``) is retried once. The planner reads the verdict
back from the event log, so resume replays exactly.
"""

from __future__ import annotations

import json
import logging

from benchmarks.lium_cluster_bench.adapters.accuracy import (
    agent_path,
    dataset_cache_dir,
    ensure_dataset,
    run_accuracy,
)
from benchmarks.lium_cluster_bench.app.context import RunContext
from benchmarks.lium_cluster_bench.app.deployment import DeploymentSession
from benchmarks.lium_cluster_bench.domain.accuracy import (
    AccuracyParams,
    AccuracyResult,
    AccuracyVerdict,
    DatasetStatus,
    accuracy_verdict,
    best_accuracy_of,
    skipped_result,
)
from benchmarks.lium_cluster_bench.domain.schema import EventType
from benchmarks.lium_cluster_bench.domain.search import CheckAccuracy
from benchmarks.lium_cluster_bench.fsutil import atomic_write_text

logger = logging.getLogger(__name__)


def stored_checks(
    ctx: RunContext,
) -> list[tuple[str, AccuracyResult, AccuracyVerdict]]:
    """(cand_key, result, verdict) of every recorded check, in order."""
    out = []
    for event in ctx.store.of_type(EventType.ACCURACY):
        try:
            out.append(
                (
                    str(event.data["cand_key"]),
                    AccuracyResult.model_validate(event.data["result"]),
                    AccuracyVerdict.model_validate(event.data["verdict"]),
                )
            )
        except (KeyError, ValueError):
            continue  # a malformed event never blocks the run
    return out


class AccuracyGate:
    """``PhaseRuntime.checker``: one accuracy check per ``CheckAccuracy``."""

    def __init__(self, ctx: RunContext, session: DeploymentSession):
        self.ctx = ctx
        self.session = session
        self._dataset: DatasetStatus | None = None

    def _params(self) -> AccuracyParams:
        cfg = self.ctx.config.accuracy
        return AccuracyParams(
            questions=cfg.questions,
            shots=cfg.shots,
            concurrency=cfg.concurrency,
            max_tokens=cfg.max_tokens,
        )

    def _ensure_dataset(self) -> DatasetStatus:
        if self._dataset is None or not self._dataset.ok:
            self._dataset = ensure_dataset(
                self.ctx.cluster,
                self.ctx.node0,
                python=self.ctx.python,
                agent=agent_path(self.ctx.src_dir),
                cache_dir=dataset_cache_dir(self.ctx.config.workdir),
            )
        return self._dataset

    def _probe(self, action: CheckAccuracy, attempt: int) -> AccuracyResult:
        dataset = self._ensure_dataset()
        if not dataset.ok:
            return skipped_result(f"dataset unavailable: {dataset.error or 'unknown'}")
        plan = self.session.plan
        out = self.ctx.exp_dir / "accuracy" / f"{action.candidate.fp}-{attempt}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        return run_accuracy(
            self.ctx.cluster,
            self.ctx.node0,
            python=self.ctx.python,
            agent=agent_path(self.ctx.src_dir),
            url=f"http://127.0.0.1:{self.ctx.config.frontend_port}",
            model=plan.served_model_name if plan else "",
            dataset=dataset.path,
            out=str(out),
            params=self._params(),
        )

    def check(self, action: CheckAccuracy) -> None:
        deploy = self.session.ensure(action.candidate)
        if not deploy.ok:
            return  # the DEPLOY event tells the planner; no verdict to record
        cfg = self.ctx.config.accuracy
        # The relative rule compares with configurations that passed: a broken
        # one must not lower the bar for the next.
        previous = [
            result
            for _, result, verdict in stored_checks(self.ctx)
            if verdict.status == "pass"
        ]
        self.ctx.status(step=f"accuracy check of {action.candidate.key}")
        result = self._probe(action, 1)
        verdict = self._verdict(result, previous)
        if verdict.status == "error":
            logger.warning(
                "accuracy probe inconclusive (%s); retrying once", verdict.reason
            )
            result = self._probe(action, 2)
            verdict = self._verdict(result, previous)
        record = {
            "cand_key": action.candidate.key,
            "family_id": action.candidate.family_id,
            "result": result.model_dump(mode="json"),
            "verdict": verdict.model_dump(mode="json"),
        }
        atomic_write_text(
            self.ctx.exp_dir / "accuracy" / f"{action.candidate.fp}.verdict.json",
            json.dumps(record, indent=1),
        )
        self.ctx.store.append(EventType.ACCURACY, record)
        if cfg.enabled and verdict.status != "pass":
            logger.warning(
                "accuracy gate %s for %s: %s",
                verdict.status,
                action.candidate.key,
                verdict.reason,
            )

    def _verdict(
        self, result: AccuracyResult, previous: list[AccuracyResult]
    ) -> AccuracyVerdict:
        cfg = self.ctx.config.accuracy
        best = best_accuracy_of(previous, protocol=result.protocol)
        return accuracy_verdict(
            result,
            floor=cfg.floor,
            best_accuracy=best,
            max_drop=cfg.max_drop,
            max_error_rate=cfg.max_error_rate,
        )
