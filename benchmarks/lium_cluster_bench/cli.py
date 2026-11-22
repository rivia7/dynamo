# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line: ``python -m benchmarks.lium_cluster_bench <command>``.

Commands
  run         run (or resume) an experiment from node0 of the cluster
  dry-run     the full pipeline against simulated hardware (no GPUs, no SSH)
  status      print an experiment's status.json
  report      re-render the Chinese report from analysis/summary.json
  kb          export/merge an experiment into a knowledge-base directory
  controller  laptop-side lium workflow (rent, sync, launch, watch, pull, down)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from benchmarks.lium_cluster_bench import controller
from benchmarks.lium_cluster_bench.adapters import profile_store
from benchmarks.lium_cluster_bench.adapters.aiperf import REQUIRED_FLAGS
from benchmarks.lium_cluster_bench.adapters.engine_sglang import SglangEngine
from benchmarks.lium_cluster_bench.adapters.executors import (
    ClusterExecutor,
    RecordingExecutor,
    Redactor,
)
from benchmarks.lium_cluster_bench.adapters.providers import ExistingClusterProvider
from benchmarks.lium_cluster_bench.app.context import RunContext, Secrets, SystemClock
from benchmarks.lium_cluster_bench.app.phases.diagnose import DiagnosePhase
from benchmarks.lium_cluster_bench.app.phases.download import DownloadPhase
from benchmarks.lium_cluster_bench.app.phases.fabric import FabricPhase
from benchmarks.lium_cluster_bench.app.phases.finalize import FinalizePhase
from benchmarks.lium_cluster_bench.app.phases.microbench import MicrobenchPhase
from benchmarks.lium_cluster_bench.app.phases.nvlink import NvlinkPhase
from benchmarks.lium_cluster_bench.app.phases.planned import (
    AccuracyPhase,
    ConfirmPhase,
    NicSweepPhase,
    PilotPhase,
    SearchPhase,
)
from benchmarks.lium_cluster_bench.app.phases.preflight import (
    PreflightPhase,
    collect_family_flags,
    restore,
)
from benchmarks.lium_cluster_bench.app.pipeline import Pipeline
from benchmarks.lium_cluster_bench.app.simulate import (
    SimClock,
    Simulator,
    synthetic_inventory,
)
from benchmarks.lium_cluster_bench.app.store import EventStore
from benchmarks.lium_cluster_bench.app.summary import plan_lookup_for
from benchmarks.lium_cluster_bench.domain.nvlink import NVLINK_OFF_FLAGS
from benchmarks.lium_cluster_bench.domain.requests import default_tiers, parse_tier
from benchmarks.lium_cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
    DEFAULT_LINK_TARGET_UTIL,
    AccuracyConfig,
    BudgetConfig,
    DiagnoseConfig,
    EventType,
    ExperimentSummary,
    RunConfig,
)
from benchmarks.lium_cluster_bench.kb.writer import export_experiment
from benchmarks.lium_cluster_bench.ports import NodeAddress, Provider
from benchmarks.lium_cluster_bench.report.render import render_report

logger = logging.getLogger("cluster_bench")

DEFAULT_IMAGE = (
    "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
    "@sha256:37badffd83eb334aee9d06fb442e36fc3c963314336c92622c78ac5c037d514d"
)
DEFAULT_WORKDIR = "/root/cluster_bench"


def _csv(text: str | None) -> tuple[str, ...]:
    return tuple(t.strip() for t in (text or "").split(",") if t.strip())


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exp-id",
        default=None,
        help="experiment id; reuse one to resume (default: timestamped)",
    )
    p.add_argument(
        "--workdir",
        default=DEFAULT_WORKDIR,
        help="node-local harness directory (same on every node)",
    )
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument(
        "--nodes",
        default="auto",
        help="'auto' (probe wg0 10.42.0.1-16) or comma-separated wg0 IPs",
    )
    p.add_argument(
        "--hosts",
        default=None,
        help="hosts file: 'name address [ssh_target] [port]' per line",
    )
    p.add_argument(
        "--expect-nodes",
        type=int,
        default=4,
        help="node count `--nodes auto` must discover (2, 4 or 8)",
    )
    p.add_argument("--model", default="glm-5.3", help="tuned model profile")
    p.add_argument(
        "--model-repo",
        default=None,
        help="untuned generic HF model instead of a tuned profile",
    )
    p.add_argument("--model-revision", default="main")
    p.add_argument("--model-size-gb", type=float, default=None)
    p.add_argument("--served-context", type=int, default=None)
    p.add_argument(
        "--gpu-profile",
        default=None,
        help="override GPU auto-detection (b200, b300, h200, h100, rtx_pro_6000)",
    )
    p.add_argument(
        "--model-root",
        default=None,
        help="local directory for weights (default: largest local disk)",
    )
    p.add_argument("--budget-hours", type=float, default=9.5)
    p.add_argument(
        "--families",
        default="",
        help="comma list of family ids (opt-in families included)",
    )
    p.add_argument("--nic-counts", default="8,4,2,1")
    p.add_argument(
        "--simulate-acceptance",
        action="store_true",
        help="SGLANG_SIMULATE_ACC_* (labelled in the report)",
    )
    p.add_argument(
        "--price-per-node-hour",
        type=float,
        default=None,
        help="USD per node-hour for $/token metrics (the controller's "
        "provider.json supplies lium prices automatically)",
    )
    p.add_argument(
        "--lldp-wait",
        type=float,
        default=35.0,
        help="seconds to listen for switch LLDP frames on RoCE ports (0 = off)",
    )
    p.add_argument(
        "--no-link-matrix",
        action="store_true",
        help="skip the perftest node-pair x rail link matrix (~2-4 min)",
    )
    p.add_argument(
        "--goodput-slo",
        action="append",
        default=None,
        metavar="NAME:TTFT_S:ITL_MS",
        help="goodput SLO tier (repeatable; replaces the defaults "
        "u50:10:20 and u20:30:50)",
    )
    p.add_argument(
        "--no-accuracy",
        action="store_true",
        help="skip the GSM8K accuracy gate on the leader (not recommended: "
        "throughput runs cannot see broken numerics)",
    )
    p.add_argument("--accuracy-questions", type=int, default=200)
    p.add_argument(
        "--accuracy-floor",
        type=float,
        default=AccuracyConfig().floor,
        help="minimum GSM8K accuracy of a recommendable configuration",
    )
    p.add_argument(
        "--accuracy-max-drop",
        type=float,
        default=AccuracyConfig().max_drop,
        help="allowed accuracy drop below the best checked configuration",
    )
    p.add_argument(
        "--accuracy-max-checks",
        type=int,
        default=AccuracyConfig().max_checks,
        help="configurations checked before the leader stays unchecked",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="opt-in diagnostic window after the confirmation: Nsight Systems "
        "kernel breakdown, SGLang MFU/EPLB metrics (not an official result)",
    )
    p.add_argument(
        "--diagnose-no-nsys",
        action="store_true",
        help="diagnostic window without Nsight Systems (metrics only)",
    )
    p.add_argument("--diagnose-steps", type=int, default=DiagnoseConfig().steps)
    p.add_argument(
        "--no-nvlink-ablation",
        action="store_true",
        help="skip re-measuring the leader with NVLink switched off "
        "(GPU-GPU traffic over PCIe)",
    )
    p.add_argument(
        "--link-target-util",
        type=float,
        default=DEFAULT_LINK_TARGET_UTIL,
        help="highest p99 utilization a link may run at when the report sizes the minimum interconnect (0-1)",
    )
    p.add_argument("--diagnose-load-s", type=float, default=DiagnoseConfig().load_s)
    p.add_argument("--control-interface", default="wg0")
    p.add_argument("--frontend-port", type=int, default=8000)
    p.add_argument("--only", default="", help="debug: run only these phases")
    p.add_argument("--log-level", default="INFO")


def _config(args: argparse.Namespace, dry_run: bool) -> RunConfig:
    return RunConfig(
        exp_id=args.exp_id or time.strftime("cb-%Y%m%d-%H%M%S"),
        model=args.model,
        gpu_profile=args.gpu_profile,
        image=args.image,
        workdir=args.workdir,
        model_root=args.model_root,
        budget=BudgetConfig(total_hours=args.budget_hours),
        families=_csv(args.families),
        nic_counts=tuple(int(n) for n in _csv(args.nic_counts)),
        simulate_acceptance=args.simulate_acceptance,
        model_repo=args.model_repo,
        model_revision=args.model_revision,
        model_size_gb=args.model_size_gb,
        served_context=args.served_context,
        control_interface=args.control_interface,
        frontend_port=args.frontend_port,
        dry_run=dry_run,
        price_per_node_hour=args.price_per_node_hour,
        lldp_wait_s=args.lldp_wait,
        fabric_link_matrix=not args.no_link_matrix,
        slo_tiers=tuple(parse_tier(x) for x in args.goodput_slo)
        if args.goodput_slo
        else default_tiers(),
        link_target_util=args.link_target_util,
        nvlink_ablation=not args.no_nvlink_ablation,
        accuracy=AccuracyConfig(
            enabled=not args.no_accuracy,
            questions=args.accuracy_questions,
            floor=args.accuracy_floor,
            max_drop=args.accuracy_max_drop,
            max_checks=args.accuracy_max_checks,
        ),
        diagnose=DiagnoseConfig(
            enabled=args.diagnose,
            nsys=not args.diagnose_no_nsys,
            steps=args.diagnose_steps,
            load_s=args.diagnose_load_s,
        ),
        bench_specs={
            **DEFAULT_BENCH_SPECS,
            "diagnose": DEFAULT_BENCH_SPECS["diagnose"].model_copy(
                update={"duration_s": int(args.diagnose_load_s)}
            ),
        },
    )


def _resume_config(store: EventStore, fresh: RunConfig) -> RunConfig:
    """A resumed experiment keeps its original configuration."""
    starts = store.of_type(EventType.RUN_START)
    if not starts:
        return fresh
    original = RunConfig.model_validate(starts[0].data["config"])
    if original.model_dump(exclude={"exp_id"}) != fresh.model_dump(exclude={"exp_id"}):
        logger.warning(
            "resuming %s with its original configuration; new CLI options are ignored",
            original.exp_id,
        )
    return original


def _phases() -> dict:
    phases = [
        PreflightPhase(),
        DownloadPhase(),
        FabricPhase(),
        MicrobenchPhase(),
        SearchPhase(),
        AccuracyPhase(),
        PilotPhase(),
        NicSweepPhase(),
        ConfirmPhase(),
        NvlinkPhase(),
        DiagnosePhase(),
        FinalizePhase(),
    ]
    return {p.name: p for p in phases}


def _provider(args: argparse.Namespace) -> Provider:
    """The runner on node0 always attaches to an existing cluster; renting is
    the laptop controller's job."""
    if args.hosts:
        return ExistingClusterProvider.from_hosts_file(Path(args.hosts))
    if args.nodes == "auto":
        return ExistingClusterProvider.auto(expect_nodes=args.expect_nodes)
    return ExistingClusterProvider.from_ip_list(_csv(args.nodes))


def _real_nodes(args: argparse.Namespace) -> list[NodeAddress]:
    return _provider(args).nodes()


def cmd_run(args: argparse.Namespace, dry_run: bool = False) -> int:
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    fresh = _config(args, dry_run)
    exp_dir = Path(fresh.workdir) / "experiments" / fresh.exp_id
    redactor = Redactor.from_env()
    if dry_run:
        clock = SimClock()
        store = EventStore(exp_dir / "events.jsonl", now=clock.now)
        config = _resume_config(store, fresh)
        gpu_name = args.fake_gpu
        inventory = synthetic_inventory(gpu_name, nodes=args.fake_nodes)
        gpu = profile_store.load_gpu_profiles()[gpu_name]
        simulator = Simulator(
            inventory=inventory,
            gpu=gpu,
            clock=clock,
            unsupported_rungs_needed=_csv(args.fake_needs_rungs),
            crash_after_points=args.fake_crash_after,
            accuracy_fail_families=_csv(args.fake_accuracy_fail),
        )
        nodes = [
            NodeAddress(n.name, n.address, None if i == 0 else f"root@{n.address}")
            for i, n in enumerate(inventory.nodes)
        ]
        cluster = RecordingExecutor(
            tuple(n.name for n in nodes), responder=simulator.respond, redactor=redactor
        )
    else:
        clock = SystemClock()
        store = EventStore(exp_dir / "events.jsonl", now=clock.now)
        config = _resume_config(store, fresh)
        nodes = _real_nodes(args)
        cluster = ClusterExecutor.from_addresses(nodes, exp_dir / ".ssh", redactor)
        simulator = None
    ctx = RunContext(
        config=config,
        exp_dir=exp_dir,
        store=store,
        cluster=cluster,
        clock=clock,
        redactor=redactor,
        nodes=nodes,
        engine=SglangEngine(),
        secrets=Secrets.from_env(),
        simulator=simulator,
    )
    if simulator is not None:
        simulator.help_text = lambda: _simulated_help(ctx)
    print(f"experiment: {config.exp_id}  dir: {exp_dir}", flush=True)
    rc = Pipeline(ctx, _phases(), restore).run(only=_csv(args.only))
    if dry_run:
        dump = exp_dir / "dry_run_commands.txt"
        dump.write_text(
            "\n".join(f"[{c.node}] {c.command}" for c in cluster.commands) + "\n"
        )
        print(f"recorded {len(cluster.commands)} commands -> {dump}")
    report = exp_dir / "report" / "report.zh-CN.md"
    if report.exists():
        print(f"report: {report}")
    return rc


def _simulated_help(ctx: RunContext) -> str:
    """Help text exposing every flag the profiles use (simulation only)."""
    from_profiles: set[str] = set()
    if ctx.deployment is not None and ctx.model is not None:
        for flags in collect_family_flags(
            ctx.deployment, ctx.model, profile_store.RecipeLoader()
        ).values():
            from_profiles |= flags
    return " ".join(sorted(from_profiles | set(REQUIRED_FLAGS) | set(NVLINK_OFF_FLAGS)))


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.workdir) / "experiments" / args.exp_id / "status.json"
    if not path.exists():
        print(f"no status at {path}", file=sys.stderr)
        return 1
    print(path.read_text())
    return 0


def _load_summary(exp_dir: Path) -> ExperimentSummary:
    with open(exp_dir / "analysis" / "summary.json", "r", encoding="utf-8") as fh:
        return ExperimentSummary.model_validate(json.load(fh))


def cmd_report(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp_dir)
    md, html = render_report(
        _load_summary(exp_dir), exp_dir / "report", plan_lookup_for(exp_dir)
    )
    print(f"{md}\n{html}")
    return 0


def cmd_kb(args: argparse.Namespace) -> int:
    exp_dir = Path(args.exp_dir)
    result = export_experiment(
        _load_summary(exp_dir), exp_dir, Path(args.kb_root), plan_lookup_for(exp_dir)
    )
    print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster_bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume an experiment (on node0)")
    _add_run_args(run)
    dry = sub.add_parser("dry-run", help="simulate the whole pipeline without GPUs")
    _add_run_args(dry)
    dry.add_argument(
        "--fake-gpu", default="h200", choices=sorted(profile_store.load_gpu_profiles())
    )
    dry.add_argument("--fake-nodes", type=int, default=4)
    dry.add_argument(
        "--fake-needs-rungs",
        default="",
        help="simulate kernel failures until these rungs apply",
    )
    dry.add_argument(
        "--fake-crash-after",
        type=int,
        default=None,
        help="simulate a SIGTERM after this many measured points (resume testing)",
    )
    dry.add_argument(
        "--fake-accuracy-fail",
        default="",
        help="comma-separated family ids whose simulated GSM8K accuracy is 0.42",
    )
    dry.set_defaults(workdir=str(Path.home() / ".cluster_bench" / "dryrun"))
    status = sub.add_parser("status", help="print status.json")
    status.add_argument("--exp-id", required=True)
    status.add_argument("--workdir", default=DEFAULT_WORKDIR)
    report = sub.add_parser("report", help="re-render the report from summary.json")
    report.add_argument("--exp-dir", required=True)
    kb = sub.add_parser("kb", help="merge an experiment into a knowledge base")
    kb.add_argument("--exp-dir", required=True)
    kb.add_argument("--kb-root", required=True)
    controller.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "dry-run":
        return cmd_run(args, dry_run=True)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "kb":
        return cmd_kb(args)
    if args.command == "controller":
        return controller.main(args)
    raise SystemExit(f"unknown command {args.command}")
