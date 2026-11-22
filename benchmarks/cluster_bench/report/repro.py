# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduction command lines shared by the report and the KB export.

The run command is rebuilt from the summary fields (and its recorded
``RunConfig`` when present) that map one-to-one onto ``cli.py run`` flags;
the re-render and KB commands are the ``report`` and ``kb`` subcommands.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Sequence

from benchmarks.cluster_bench.domain.requests import default_tiers
from benchmarks.cluster_bench.domain.schema import (
    DEFAULT_LINK_TARGET_UTIL,
    AccuracyConfig,
    ExperimentSummary,
)

HARNESS = "benchmarks.cluster_bench"
WORKDIR = '"$WORKDIR"'
PYTHON = '"$WORKDIR/venv/bin/python"'
SRC_DIR = '"$WORKDIR/src"'
EXP_DIR_TEMPLATE = '"$WORKDIR/experiments/{exp_id}"'


@dataclass(frozen=True)
class ReproCommand:
    """One reproduction step as a shell command line."""

    purpose: str  # "run" | "render" | "kb"
    shell: str


def pretty_argv(argv: Sequence[str], *, quote: bool = True) -> str:
    """One ``--flag value`` group per line, joined by shell continuations."""
    lines: list[list[str]] = []
    for arg in argv:
        if not lines or arg.startswith("--"):
            lines.append([])
        lines[-1].append(shlex.quote(arg) if quote else arg)
    return " \\\n  ".join(" ".join(line) for line in lines)


def _py(*args: str) -> str:
    """``$PY -m benchmarks.cluster_bench args`` (``args`` are shell words)."""
    return pretty_argv((PYTHON, "-m", HARNESS, *args), quote=False)


def exp_dir_arg(summary: ExperimentSummary) -> str:
    """The experiment directory on a node, as a shell word."""
    return EXP_DIR_TEMPLATE.format(exp_id=summary.exp_id)


def run_args(summary: ExperimentSummary) -> tuple[str, ...]:
    """Shell words of ``python -m benchmarks.cluster_bench run`` for ``summary``.

    Only flags whose values the summary records are emitted; the dataset and
    the bench specs are RunConfig defaults, not CLI flags.
    """
    q = shlex.quote
    args: list[str] = [
        "run",
        "--exp-id",
        q(summary.exp_id),
        "--workdir",
        WORKDIR,
        "--image",
        q(summary.image),
        "--model",
        q(summary.model),
        "--model-revision",
        q(summary.checkpoint.revision),
        "--gpu-profile",
        q(summary.gpu_profile),
        "--served-context",
        str(summary.served_context),
        "--budget-hours",
        f"{summary.budget_hours:g}",
        "--nodes",
        "auto",
        "--expect-nodes",
        str(summary.node_count),
    ]
    cfg = summary.run_config
    if cfg is not None:
        if cfg.model_repo:
            args += ["--model-repo", q(cfg.model_repo)]
            if cfg.model_size_gb is not None:
                args += ["--model-size-gb", f"{cfg.model_size_gb:g}"]
        if cfg.families:
            args += ["--families", q(",".join(cfg.families))]
        args += ["--nic-counts", ",".join(str(n) for n in cfg.nic_counts)]
        if cfg.simulate_acceptance:
            args.append("--simulate-acceptance")
        args += ["--control-interface", q(cfg.control_interface)]
        if cfg.price_per_node_hour is not None:
            args += ["--price-per-node-hour", f"{cfg.price_per_node_hour:g}"]
        if tuple(cfg.slo_tiers) != default_tiers():
            for tier in cfg.slo_tiers:
                args += [
                    "--goodput-slo",
                    q(f"{tier.name}:{tier.ttft_s:g}:{tier.itl_ms:g}"),
                ]
        acc, base = cfg.accuracy, AccuracyConfig()
        if not acc.enabled:
            args.append("--no-accuracy")
        else:
            for flag, value, default in (
                ("--accuracy-questions", acc.questions, base.questions),
                ("--accuracy-floor", acc.floor, base.floor),
                ("--accuracy-max-drop", acc.max_drop, base.max_drop),
                ("--accuracy-max-checks", acc.max_checks, base.max_checks),
            ):
                if value != default:
                    args += [flag, f"{value:g}"]
        if not cfg.nvlink_ablation:
            args.append("--no-nvlink-ablation")
        if cfg.link_target_util != DEFAULT_LINK_TARGET_UTIL:
            args += ["--link-target-util", f"{cfg.link_target_util:g}"]
        diag = cfg.diagnose
        if diag.enabled:
            args.append("--diagnose")
            if not diag.nsys:
                args.append("--diagnose-no-nsys")
    return tuple(args)


def repro_commands(
    summary: ExperimentSummary, kb_root: str = '"$KB_ROOT"'
) -> tuple[ReproCommand, ...]:
    """The run, re-render and KB export commands of one experiment."""
    exp_dir = exp_dir_arg(summary)
    return (
        ReproCommand("run", f"cd {SRC_DIR} && " + _py(*run_args(summary))),
        ReproCommand(
            "render", f"cd {SRC_DIR} && " + _py("report", "--exp-dir", exp_dir)
        ),
        ReproCommand(
            "kb",
            f"cd {SRC_DIR} && " + _py("kb", "--exp-dir", exp_dir, "--kb-root", kb_root),
        ),
    )
