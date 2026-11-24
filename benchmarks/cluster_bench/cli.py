# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line: ``python -m benchmarks.cluster_bench <command>``.

Commands
  run         run (or resume) an experiment from node0 of the cluster
  dry-run     the full pipeline against simulated hardware (no GPUs, no SSH)
  status      print an experiment's status.json
  report      re-render the Chinese report from analysis/summary.json
  kb          export/merge an experiment into a knowledge-base directory
  controller  laptop-side workflow: SSH machines or a lium rental (up, sync,
              launch, watch, pull, down)

``--stage`` picks what is measured: e2e (default; aggregated and PD
deployments end to end), p-only (the PD prefill stage alone: input tok/s/GPU),
d-only (the PD decode stage alone: output tok/s/GPU) or a-only (the attention
side of attention-FFN disaggregation, F simulated as described by
``--afd-f``: output tok/s per A GPU); see domain/stages.py and domain/afd.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from benchmarks.cluster_bench import controller
from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.aiperf import (
    AGENTX_MIN_DURATION_S,
    ALL_FLAGS,
    check_extra_args,
)
from benchmarks.cluster_bench.adapters.engine_sglang import SglangEngine
from benchmarks.cluster_bench.adapters.executors import (
    ClusterExecutor,
    RecordingExecutor,
    Redactor,
)
from benchmarks.cluster_bench.adapters.providers import ExistingClusterProvider
from benchmarks.cluster_bench.app.afd_ops import stored_plan
from benchmarks.cluster_bench.app.context import RunContext, Secrets, SystemClock
from benchmarks.cluster_bench.app.phases.diagnose import DiagnosePhase
from benchmarks.cluster_bench.app.phases.download import DownloadPhase
from benchmarks.cluster_bench.app.phases.fabric import FabricPhase
from benchmarks.cluster_bench.app.phases.finalize import FinalizePhase
from benchmarks.cluster_bench.app.phases.microbench import MicrobenchPhase
from benchmarks.cluster_bench.app.phases.nvlink import NvlinkPhase
from benchmarks.cluster_bench.app.phases.planned import (
    AccuracyPhase,
    ConfirmPhase,
    NicSweepPhase,
    PilotPhase,
    SearchPhase,
)
from benchmarks.cluster_bench.app.phases.preflight import (
    PreflightPhase,
    collect_family_flags,
    resume,
)
from benchmarks.cluster_bench.app.pipeline import Pipeline, rerun_phases
from benchmarks.cluster_bench.app.simulate import (
    SimClock,
    Simulator,
    synthetic_inventory,
)
from benchmarks.cluster_bench.app.store import EventStore
from benchmarks.cluster_bench.app.summary import plan_lookup_for
from benchmarks.cluster_bench.domain.afd import parse_afd_f
from benchmarks.cluster_bench.domain.datasets import (
    DEFAULT_DATASET,
    DatasetError,
    parse_dataset,
)
from benchmarks.cluster_bench.domain.nvlink import NVLINK_OFF_FLAGS
from benchmarks.cluster_bench.domain.requests import default_tiers, parse_tier
from benchmarks.cluster_bench.domain.schema import (
    DEFAULT_BENCH_SPECS,
    DEFAULT_LINK_TARGET_UTIL,
    AccuracyConfig,
    AfdFSpec,
    BudgetConfig,
    DiagnoseConfig,
    EventType,
    ExperimentSummary,
    RunConfig,
)
from benchmarks.cluster_bench.domain.stages import STAGE_PREFIX, parse_stage, stage_spec
from benchmarks.cluster_bench.kb.writer import export_experiment
from benchmarks.cluster_bench.ports import NodeAddress, Provider
from benchmarks.cluster_bench.report.render import render_report

logger = logging.getLogger("cluster_bench")

# Dynamo 1.6.2 (SGLang 0.5.21, CUDA 13.0.3): the multi-arch index digest.
DEFAULT_IMAGE = (
    "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.6.2"
    "@sha256:d12d31eff2b5896dd5cdca0e9c10981535572f0bedbbf98ab2bb5248bdede43b"
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
        "--platform",
        choices=("generic", "lium"),
        default="generic",
        help="generic SSH machines (default) or a lium.io rental (wg0 control "
        "network, /mnt volumes excluded from weights, auto node discovery)",
    )
    p.add_argument(
        "--stage",
        default="e2e",
        metavar="{e2e,p-only,d-only,a-only}",
        help="e2e: whole deployments; p-only: the PD prefill stage alone "
        "(max input tok/s/GPU); d-only: the PD decode stage alone (max output "
        "tok/s/GPU); a-only: the attention side of attention-FFN "
        "disaggregation, routed experts on a simulated F (max output tok/s "
        "per A GPU)",
    )
    p.add_argument(
        "--afd-f",
        default="",
        metavar="KEY=VALUE,...",
        help="a-only: the F the attention side waits for. Keys: gpus (F GPUs "
        "per A instance; default the fewest that hold the routed experts), "
        "link=net|intra, layout=staged|direct, microbatches (default 2), "
        "wire=bf16|fp8, bw_gbs or bw_gbps (per A GPU, one direction; default "
        "measured), lat_us, eff_mem, eff_flop, mem_fraction",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="AIPerf public dataset (an AgentX trace replays closed-loop), "
        "'<custom-dataset-type>:/abs/path' on node0 (file: = mooncake_trace), "
        "or 'synthetic:isl=N,osl=M[,isl_std=S,osl_std=S]'",
    )
    p.add_argument(
        "--aiperf-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="extra 'aiperf profile' argument, appended verbatim (repeatable; "
        "flags the harness sets are refused)",
    )
    p.add_argument(
        "--nodes",
        default=None,
        help="comma-separated control IPs (root@IP over ssh), or 'auto' on "
        "lium (probe wg0 10.42.0.1-16); default: --hosts, or auto on lium",
    )
    p.add_argument(
        "--hosts",
        default=None,
        help="hosts file: 'name address [ssh_target] [port] [identity=PATH] "
        "[iface=NIC] [known_hosts=PATH]' per line",
    )
    p.add_argument(
        "--expect-nodes",
        type=int,
        default=4,
        help="node count `--nodes auto` must discover (lium: 2, 4 or 8)",
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
        help="override GPU auto-detection (b200, b300, h200, h100, rtx_pro_6000, "
        "rtx_pro_6000d)",
    )
    p.add_argument(
        "--model-root",
        default=None,
        help="local directory for weights (default: largest local disk)",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="weights already at this absolute path on every node (the same "
        "path everywhere): no Hugging Face download; the copies must match in "
        "file names and sizes. Pin --model-revision to record their commit",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint variant of the tuned model profile, e.g. nvfp4-nvidia "
        "(default: the GPU profile's weights variant)",
    )
    p.add_argument(
        "--hf-endpoint",
        default=None,
        help="Hugging Face endpoint (HF_ENDPOINT) for node-side downloads, i.e. "
        "model files and the AIPerf dataset; e.g. https://hf-mirror.com where "
        "huggingface.co is unreachable",
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
    p.add_argument(
        "--confirm-duration-s",
        type=int,
        default=DEFAULT_BENCH_SPECS["decision"].duration_s,
        help="measured seconds of the confirmatory rerun (decision spec); an "
        f"AgentX dataset needs >= {AGENTX_MIN_DURATION_S} (the agentx preset's "
        "own default is 3600)",
    )
    p.add_argument(
        "--control-interface",
        default="auto",
        help="control-plane NIC: 'auto' = the NIC holding each node's control "
        "address (wg0 on lium), or one name for every node",
    )
    p.add_argument("--frontend-port", type=int, default=8000)
    p.add_argument("--only", default="", help="debug: run only these phases")
    p.add_argument(
        "--redo",
        default="",
        help="run these phases of a finished experiment again, with every "
        "phase after them: 'diagnose' adds the diagnostic window (with the "
        "--diagnose options given now) to a run that had none, or summarizes "
        "the reports of its window again (no new capture); 'finalize' "
        "rebuilds its summary, report and KB export",
    )
    p.add_argument("--log-level", default="INFO")


def _config(args: argparse.Namespace, dry_run: bool) -> RunConfig:
    try:
        stage = parse_stage(args.stage)
        dataset = parse_dataset(args.dataset)
        afd_f = parse_afd_f(args.afd_f)
    except (ValueError, DatasetError) as exc:
        raise SystemExit(f"cluster_bench: {exc}") from exc
    if args.afd_f and stage_spec(stage).afd_side is None:
        raise SystemExit("cluster_bench: --afd-f applies to --stage a-only only")
    floor = AGENTX_MIN_DURATION_S if dataset.agentx else 1
    if args.confirm_duration_s < floor:
        raise SystemExit(
            f"cluster_bench: --confirm-duration-s must be >= {floor}"
            + (
                " for an AgentX dataset (agentx scenario floor)"
                if dataset.agentx
                else ""
            )
        )
    aiperf_args = tuple(args.aiperf_arg or ())
    try:
        check_extra_args(aiperf_args)
    except ValueError as exc:
        raise SystemExit(f"cluster_bench: {exc}") from exc
    if args.model_path and not args.model_path.startswith("/"):
        raise SystemExit("cluster_bench: --model-path must be an absolute path")
    if args.hf_endpoint and not args.hf_endpoint.startswith(("http://", "https://")):
        raise SystemExit("cluster_bench: --hf-endpoint must be an http(s) URL")
    if args.checkpoint and args.model_repo:
        raise SystemExit(
            "cluster_bench: --checkpoint picks a tuned profile's checkpoint; "
            "--model-repo already names the weights"
        )
    prefix = f"{STAGE_PREFIX[stage]}-" if STAGE_PREFIX[stage] else ""
    return RunConfig(
        exp_id=args.exp_id or time.strftime(f"cb-{prefix}%Y%m%d-%H%M%S"),
        stage=stage,
        platform=args.platform,
        dataset=dataset.text,
        aiperf_args=aiperf_args,
        model=args.model,
        gpu_profile=args.gpu_profile,
        image=args.image,
        workdir=args.workdir,
        model_root=args.model_root,
        model_path=(args.model_path.rstrip("/") or "/") if args.model_path else None,
        checkpoint=args.checkpoint,
        hf_endpoint=args.hf_endpoint.rstrip("/") if args.hf_endpoint else None,
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
        afd_f=afd_f,
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
            "decision": DEFAULT_BENCH_SPECS["decision"].model_copy(
                update={"duration_s": args.confirm_duration_s}
            ),
            "diagnose": DEFAULT_BENCH_SPECS["diagnose"].model_copy(
                update={"duration_s": int(args.diagnose_load_s)}
            ),
        },
    )


def _resume_config(
    store: EventStore, fresh: RunConfig, redo: tuple[str, ...] = ()
) -> RunConfig:
    """A resumed experiment keeps its original configuration; ``--redo
    diagnose`` takes the diagnostic window's options given now. A-Only keeps
    the F it planned; until it has one (the plan failed), a new ``--afd-f``
    applies."""
    starts = store.of_type(EventType.RUN_START)
    if not starts:
        if redo:
            raise SystemExit(
                f"cluster_bench: --redo needs a finished experiment; {fresh.exp_id} "
                "has not run"
            )
        return fresh
    original = RunConfig.model_validate(starts[0].data["config"])
    if "diagnose" in redo:
        if not fresh.diagnose.enabled:
            raise SystemExit("cluster_bench: --redo diagnose needs --diagnose")
        original = original.model_copy(
            update={
                "diagnose": fresh.diagnose,
                "bench_specs": {
                    **original.bench_specs,
                    "diagnose": fresh.bench_specs["diagnose"],
                },
            }
        )
    planned = stored_plan(store)
    if planned is not None:
        original = original.model_copy(update={"afd_f": planned.spec})
    elif fresh.afd_f != AfdFSpec():
        original = original.model_copy(update={"afd_f": fresh.afd_f})
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
    nodes = args.nodes or ("auto" if args.platform == "lium" else None)
    if nodes == "auto":
        if args.platform != "lium":
            raise SystemExit(
                "cluster_bench: --nodes auto probes lium's wg0 network; on other "
                "machines pass --hosts FILE or --nodes IP,IP,..."
            )
        return ExistingClusterProvider.auto(expect_nodes=args.expect_nodes)
    if not nodes:
        raise SystemExit("cluster_bench: pass --hosts FILE or --nodes IP,IP,...")
    return ExistingClusterProvider.from_ip_list(_csv(nodes))


def _real_nodes(args: argparse.Namespace) -> list[NodeAddress]:
    return _provider(args).nodes()


def cmd_run(args: argparse.Namespace, dry_run: bool = False) -> int:
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    fresh = _config(args, dry_run)
    redo = _csv(args.redo)
    try:
        rerun_phases(redo)
    except ValueError as exc:
        raise SystemExit(f"cluster_bench: --redo: {exc}") from exc
    exp_dir = Path(fresh.workdir) / "experiments" / fresh.exp_id
    redactor = Redactor.from_env()
    if dry_run:
        clock = SimClock()
        store = EventStore(exp_dir / "events.jsonl", now=clock.now)
        config = _resume_config(store, fresh, redo)
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
        config = _resume_config(store, fresh, redo)
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
    rc = Pipeline(ctx, _phases(), resume).run(only=_csv(args.only), redo=redo)
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
    return " ".join(sorted(from_profiles | set(ALL_FLAGS) | set(NVLINK_OFF_FLAGS)))


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
