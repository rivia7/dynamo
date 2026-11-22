# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental, idempotent export of one experiment into the knowledge base.

Layout under ``kb_root``::

    .lock                                   exclusive flock held during export
    index.jsonl                             one ExperimentRecord per experiment
    INDEX.md                                experiments and clusters
    clusters/<fp>/records/<kind>.jsonl      append-only records (see records.py)
    clusters/<fp>/records/conflicts.jsonl   rejected re-exports, for a human
    clusters/<fp>/cluster.md                hardware fingerprint
    clusters/<fp>/speed-of-light.md         every peak of this fingerprint
    clusters/<fp>/topology-ranking.md       families by best valid tok/s/GPU
    clusters/<fp>/knob-findings.md          findings in the agent-docs schemas
    clusters/<fp>/fabric-and-health.md      upstream fabric and node health
    clusters/<fp>/resources.md              interconnect needs, MFU, KV capacity
    experiments/<exp_id>/manifest.yaml
    experiments/<exp_id>/analysis/performance_findings.jsonl
    experiments/<exp_id>/final/{recommended_config,known_limitations}.md
    experiments/<exp_id>/final/reproduced_commands.sh

Records are deduplicated by ``rec_id`` and existing lines are never rewritten;
a changed record for an existing ``key`` goes to ``conflicts.jsonl`` and the
first version stays authoritative. Views are rebuilt from every record of
the fingerprint, only between the generated-content markers. Every file is
written only when its bytes change, so a repeated export is a no-op.

Run standalone (repo root or ``$WORKDIR/src``)::

    python -m benchmarks.cluster_bench.kb.writer --exp-dir <dir> --kb-root <kb>
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from benchmarks.cluster_bench.domain.schema import ExperimentSummary
from benchmarks.cluster_bench.fsutil import atomic_write_text, sha256_file
from benchmarks.cluster_bench.kb import views
from benchmarks.cluster_bench.kb.findings import derive_findings
from benchmarks.cluster_bench.kb.jsonl import (
    KbError,
    KbLock,
    Row,
    append_jsonl,
    merge_rows,
    read_jsonl,
)
from benchmarks.cluster_bench.kb.records import (
    CLUSTER_RECORD_TYPES,
    SCHEMA_TYPES,
    AccuracyRecord,
    ClusterRecord,
    DeployRecord,
    ExperimentRecord,
    FabricRecord,
    FindingRecord,
    HealthRecord,
    KbRecord,
    PeakRecord,
    R,
    RunRecord,
    build_records,
)
from benchmarks.cluster_bench.report import facts
from benchmarks.cluster_bench.report.plans import PlanLookup, plan_lookup_from_dir
from benchmarks.cluster_bench.report.render import load_summary, summary_path

logger = logging.getLogger(__name__)

LOCK_NAME = ".lock"
INDEX_JSONL = "index.jsonl"
INDEX_MD = "INDEX.md"
CONFLICTS = "conflicts.jsonl"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class Conflict:
    """A record that was not appended because its key or content clashed."""

    kind: str
    key: str
    reason: str  # "content_mismatch" | "key_exists"
    rec_id: str
    existing_rec_id: str


@dataclass(frozen=True)
class KbExportResult:
    """What one export appended, deduplicated, rejected and wrote."""

    kb_root: Path
    fp: str
    exp_id: str
    appended: dict[str, int] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)
    conflicts: tuple[Conflict, ...] = ()
    files_written: tuple[Path, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.files_written)


def _safe_id(value: str, what: str) -> str:
    if not _SAFE_ID.match(value) or ".." in value:
        raise ValueError(f"{what} {value!r} is not a safe path component")
    return value


def _parse(rows: Sequence[Row], wanted: type[R]) -> list[R]:
    """Typed records of one schema; rows of other (e.g. newer) schemas are skipped."""
    parsed: list[R] = []
    for row in rows:
        cls = SCHEMA_TYPES.get(str(row.get("schema")))
        if cls is not wanted:
            continue
        try:
            parsed.append(cls.model_validate(row))
        except ValidationError as exc:
            raise KbError(f"invalid {cls.SCHEMA} record {row.get('key')!r}") from exc
    return parsed


class _Export:
    """One export run; holds the paths and the write log."""

    def __init__(self, summary: ExperimentSummary, exp_dir: Path, kb_root: Path):
        self.summary = summary
        self.exp_dir = exp_dir
        self.kb_root = kb_root
        self.fp = _safe_id(summary.fingerprint, "fingerprint")
        self.exp_id = _safe_id(summary.exp_id, "exp_id")
        self.cluster_dir = kb_root / "clusters" / self.fp
        self.records_dir = self.cluster_dir / "records"
        self.exp_root = kb_root / "experiments" / self.exp_id
        self.written: list[Path] = []
        self.appended: dict[str, int] = {}
        self.duplicates: dict[str, int] = {}
        self.conflicts: list[Conflict] = []
        self.conflict_rows: list[Row] = []

    # -- files ---------------------------------------------------------------

    def write_if_changed(self, path: Path, text: str, mode: int | None = None) -> None:
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            atomic_write_text(path, text)
            self.written.append(path)
        if mode is not None and (path.stat().st_mode & 0o777) != mode:
            path.chmod(mode)

    def check_exp_id_owner(self) -> None:
        """Refuse an ``exp_id`` that the KB already holds for another cluster."""
        for row in read_jsonl(self.kb_root / INDEX_JSONL):
            if row.get("exp_id") == self.exp_id and row.get("fp") != self.fp:
                raise KbError(
                    f"exp_id {self.exp_id!r} is already exported for cluster "
                    f"{row.get('fp')!r}; refusing to mix clusters"
                )

    def markdown_targets(self) -> dict[Path, str]:
        targets = {self.kb_root / INDEX_MD: "Cluster Bench Knowledge Base"}
        for name, title in views.cluster_views(self.fp).items():
            targets[self.cluster_dir / name] = title
        final = self.exp_root / "final"
        recommended = f"Recommended Configuration: {self.exp_id}"
        targets[final / "recommended_config.md"] = recommended
        targets[final / "known_limitations.md"] = f"Known Limitations: {self.exp_id}"
        return targets

    def read_markdown(self, targets: dict[Path, str]) -> dict[Path, str | None]:
        """Existing Markdown, validated before anything is written."""
        existing: dict[Path, str | None] = {}
        for path in targets:
            text = path.read_text(encoding="utf-8") if path.exists() else None
            if text is not None:
                views.check_markers(text, str(path))
            existing[path] = text
        return existing

    # -- records -------------------------------------------------------------

    def merge(
        self,
        path: Path,
        kind: str,
        incoming: Sequence[KbRecord],
        *,
        report_conflicts: bool = True,
    ) -> None:
        rows = [record.as_json() for record in incoming]
        outcome = merge_rows(read_jsonl(path), rows, kind)
        if outcome.appended:
            append_jsonl(path, outcome.appended)
            self.written.append(path)
        self.appended[kind] = self.appended.get(kind, 0) + len(outcome.appended)
        self.duplicates[kind] = self.duplicates.get(kind, 0) + outcome.duplicates
        if not report_conflicts:
            return
        for row in outcome.conflicts:
            self.conflicts.append(
                Conflict(
                    kind=kind,
                    key=str(row["key"]),
                    reason=str(row["reason"]),
                    rec_id=str(row["rec_id"]),
                    existing_rec_id=str(row["existing_rec_id"]),
                )
            )
            self.conflict_rows.append(row)

    def flush_conflicts(self) -> None:
        if not self.conflict_rows:
            return
        path = self.records_dir / CONFLICTS
        known = {str(row.get("conflict_id")) for row in read_jsonl(path)}
        fresh = []
        for row in self.conflict_rows:
            if str(row["conflict_id"]) not in known:
                known.add(str(row["conflict_id"]))
                fresh.append(row)
        if fresh:
            append_jsonl(path, fresh)
            self.written.append(path)
            for row in fresh:
                logger.warning(
                    "KB conflict (%s) for %s: kept the existing record",
                    row["reason"],
                    row["key"],
                )

    def load_cluster(self, cls: type[R]) -> list[R]:
        return _parse(read_jsonl(self.records_dir / f"{cls.FILE_KIND}.jsonl"), cls)


def _sha256(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def export_experiment(
    summary: ExperimentSummary,
    exp_dir: Path,
    kb_root: Path,
    plan_lookup: PlanLookup | None = None,
    *,
    lock_timeout_s: float = 60.0,
) -> KbExportResult:
    """Append one experiment to the KB at ``kb_root`` and refresh its views.

    ``exp_dir`` is the experiment directory the summary came from (used for
    provenance and, when ``plan_lookup`` is ``None``, to read
    ``deployments/<cand_fp>/plan.json``). Idempotent: exporting the same
    summary again writes nothing. Raises ``KbLockTimeout`` when another
    writer holds the lock longer than ``lock_timeout_s``.
    """
    run = _Export(summary, exp_dir, kb_root)
    lookup = plan_lookup if plan_lookup is not None else plan_lookup_from_dir(exp_dir)
    findings = derive_findings(summary, lookup)
    records = build_records(summary, lookup, findings)
    leader = facts.leader_candidate(summary)
    leader_plan = lookup(leader.fp) if leader is not None else None
    summary_sha = _sha256(summary_path(exp_dir))
    exp_dir_label = str(exp_dir)

    with KbLock(kb_root / LOCK_NAME, lock_timeout_s):
        run.check_exp_id_owner()
        targets = run.markdown_targets()
        existing_md = run.read_markdown(targets)

        for cls in CLUSTER_RECORD_TYPES:
            kind = cls.FILE_KIND
            run.merge(run.records_dir / f"{kind}.jsonl", kind, records.of(kind))
        if records.experiment is not None:
            run.merge(kb_root / INDEX_JSONL, "experiment", (records.experiment,))
        # Per-experiment mirror of the findings (run-artifacts layout); its
        # conflicts are the same as the cluster finding.jsonl ones.
        run.merge(
            run.exp_root / "analysis" / "performance_findings.jsonl",
            "performance_finding",
            records.of(FindingRecord.FILE_KIND),
            report_conflicts=False,
        )
        run.flush_conflicts()

        experiments = _parse(read_jsonl(kb_root / INDEX_JSONL), ExperimentRecord)
        cluster_exps = [e for e in experiments if e.fp == run.fp]
        cluster, final = run.cluster_dir, run.exp_root / "final"
        bodies: dict[Path, str] = {}
        bodies[kb_root / INDEX_MD] = views.render_index(experiments)
        bodies[cluster / "cluster.md"] = views.render_cluster(
            run.fp, run.load_cluster(ClusterRecord), cluster_exps
        )
        bodies[cluster / "speed-of-light.md"] = views.render_speed_of_light(
            run.load_cluster(PeakRecord), cluster_exps
        )
        bodies[cluster / "topology-ranking.md"] = views.render_topology_ranking(
            run.load_cluster(RunRecord),
            run.load_cluster(DeployRecord),
            cluster_exps,
            run.load_cluster(AccuracyRecord),
        )
        bodies[cluster / "knob-findings.md"] = views.render_knob_findings(
            run.load_cluster(FindingRecord), cluster_exps
        )
        bodies[cluster / "fabric-and-health.md"] = views.render_fabric_health(
            run.load_cluster(FabricRecord), run.load_cluster(HealthRecord), cluster_exps
        )
        bodies[cluster / "resources.md"] = views.render_resources(cluster_exps)
        bodies[final / "recommended_config.md"] = views.render_recommended_config(
            summary, leader_plan, exp_dir_label, summary_sha
        )
        bodies[final / "known_limitations.md"] = views.render_known_limitations(summary)
        # A conflicting re-export keeps the first version everywhere: records
        # are first-writer-wins, so the experiment folder is left as it was.
        keep_experiment = bool(run.conflicts)
        if keep_experiment:
            logger.warning(
                "KB export of %s had conflicts; experiments/%s left unchanged",
                run.exp_id,
                run.exp_id,
            )
        for path, body in bodies.items():
            if keep_experiment and path.is_relative_to(run.exp_root):
                continue
            text = views.merge_generated(existing_md[path], targets[path], body)
            run.write_if_changed(path, text)
        if not keep_experiment:
            run.write_if_changed(
                run.exp_root / "manifest.yaml",
                views.render_manifest(
                    summary, exp_dir_label, summary_sha, records.counts()
                ),
            )
            run.write_if_changed(
                run.exp_root / "final" / "reproduced_commands.sh",
                views.render_repro_script(summary, str(kb_root)),
                mode=0o755,
            )

    result = KbExportResult(
        kb_root=kb_root,
        fp=run.fp,
        exp_id=run.exp_id,
        appended=run.appended,
        duplicates=run.duplicates,
        conflicts=tuple(run.conflicts),
        files_written=tuple(run.written),
    )
    logger.info(
        "KB export %s -> %s: appended %s, %d conflict(s), %d file(s) written",
        run.exp_id,
        kb_root,
        sum(run.appended.values()),
        len(run.conflicts),
        len(run.written),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: export ``<exp-dir>/analysis/summary.json`` into ``--kb-root``."""
    parser = argparse.ArgumentParser(
        description="Export one cluster_bench experiment into the knowledge base."
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--kb-root", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = export_experiment(load_summary(args.exp_dir), args.exp_dir, args.kb_root)
    return 1 if result.conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
