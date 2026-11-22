# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writes the Chinese report of one experiment.

``render_report`` turns an ``ExperimentSummary`` into ``report.zh-CN.md``
(with its charts as ``figures/*.svg``) and a single-file
``report.zh-CN.html`` (inline CSS and SVG; no scripts, fonts or URLs).

Run standalone from the repo root (or ``$WORKDIR/src`` on a node)::

    python -m benchmarks.cluster_bench.report.render --exp-dir <exp_dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from benchmarks.cluster_bench.domain.schema import ExperimentSummary
from benchmarks.cluster_bench.fsutil import atomic_write_text
from benchmarks.cluster_bench.report import svg
from benchmarks.cluster_bench.report.doc import render_html, render_markdown
from benchmarks.cluster_bench.report.plans import PlanLookup, plan_lookup_from_dir
from benchmarks.cluster_bench.report.sections import build_document
from benchmarks.cluster_bench.report.strings import Strings

logger = logging.getLogger(__name__)

REPORT_MD = "report.zh-CN.md"
REPORT_HTML = "report.zh-CN.html"
FIGURES_DIR = "figures"
LOCALE = "zh_CN"


def summary_path(exp_dir: Path) -> Path:
    """``<exp_dir>/analysis/summary.json``."""
    return exp_dir / "analysis" / "summary.json"


def load_summary(exp_dir: Path) -> ExperimentSummary:
    """Read ``<exp_dir>/analysis/summary.json``."""
    text = summary_path(exp_dir).read_text(encoding="utf-8")
    return ExperimentSummary.model_validate_json(text)


def render_report(
    summary: ExperimentSummary,
    out_dir: Path,
    plan_lookup: PlanLookup | None = None,
) -> tuple[Path, Path]:
    """Render the report into ``out_dir``; returns ``(markdown, html)`` paths.

    ``plan_lookup`` maps a candidate fingerprint to its ``LaunchPlan``; it
    supplies the effective SGLang config, env and argv of the recommended
    configuration (section 8). Without it that section says so explicitly.
    """
    strings = Strings.load(LOCALE).for_stage(summary.stage)
    doc = build_document(summary, strings, plan_lookup)
    figures_dir = out_dir / FIGURES_DIR
    wanted = set()
    for figure in doc.figures:
        name = f"{figure.chart_id}.svg"
        wanted.add(name)
        atomic_write_text(figures_dir / name, svg.standalone(figure.svg))
    if figures_dir.is_dir():
        for stale in figures_dir.glob("fig-*.svg"):
            if stale.name not in wanted:
                stale.unlink()
    md_path = out_dir / REPORT_MD
    html_path = out_dir / REPORT_HTML
    atomic_write_text(
        md_path, render_markdown(doc, lambda chart_id: f"{FIGURES_DIR}/{chart_id}.svg")
    )
    atomic_write_text(html_path, render_html(doc, extra_css=svg.theme_css()))
    logger.info("report written: %s, %s", md_path, html_path)
    return md_path, html_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: render the report of ``--exp-dir``."""
    parser = argparse.ArgumentParser(
        description="Render the cluster_bench Chinese report from summary.json."
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: <exp-dir>/report"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = load_summary(args.exp_dir)
    out_dir = args.out_dir or args.exp_dir / "report"
    render_report(summary, out_dir, plan_lookup_from_dir(args.exp_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
