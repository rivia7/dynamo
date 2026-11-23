# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model download with the GPUs kept busy meanwhile.

1. resolve the HF commit and shard the files across nodes (persisted plan);
2. fetch the small files (config, tokenizer, index) on every node;
3. start each node's shard download in the background;
4. while that runs: intra-node P2P/NCCL microbenchmarks and a dummy-weight
   kernel smoke that walks the GPU profile's kernel ladder (SM103/SM120
   coverage is unverified) - a GPU type that cannot run the model is found
   in minutes, not after a 700 GB download;
5. wait, NCCL-broadcast the missing shards over the RDMA fabric, verify
   sha256 everywhere and write the completion marker.

With --model-path the weights are already on every node: steps 1-3 and 5
become one check that every node holds the same files with the same sizes
(no hashing); the GPU work of step 4 still runs.
"""

from __future__ import annotations

import json
import logging
import re
import shlex

from benchmarks.cluster_bench.adapters.aiperf import PUBLIC_DATASET_REPOS
from benchmarks.cluster_bench.adapters.hf_fetch import (
    FetchPlan,
    ModelFetcher,
    ModelFetchError,
    completion_probe_command,
    local_copy_problems,
    local_file_sizes,
    model_dir_for,
)
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.facts_ops import collect_disk
from benchmarks.cluster_bench.app.history import build_history
from benchmarks.cluster_bench.app.model_ops import record_model_arch
from benchmarks.cluster_bench.app.phases.microbench import run_group
from benchmarks.cluster_bench.app.phases.planned import search_context
from benchmarks.cluster_bench.app.pipeline import PhaseRuntime
from benchmarks.cluster_bench.domain.profiles import merge_config
from benchmarks.cluster_bench.domain.schema import (
    Candidate,
    Checkpoint,
    DeploymentProfile,
    EventType,
    FamilySpec,
    Inventory,
)
from benchmarks.cluster_bench.fsutil import atomic_write_json

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_S = 4 * 3600
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SMOKE_SUFFIX = "@smoke"


def smoke_profile(
    deployment: DeploymentProfile, family: FamilySpec, inventory: Inventory
) -> tuple[DeploymentProfile, Inventory] | None:
    """One dummy-weight instance of an aggregated family on as few nodes as it needs."""
    if any(role.role != "agg" for role in family.roles):
        return None
    role = family.roles[0]
    smoke_role = role.model_copy(
        update={
            "instances": 1,
            "sglang": merge_config(role.sglang, {"load-format": "dummy"}),
        }
    )
    smoke_family = family.model_copy(
        update={
            "id": family.id + SMOKE_SUFFIX,
            "roles": (smoke_role,),
            "transport": "none",
            "cross_node": (),
            # The smoke runs on fewer nodes than the cluster: no size filter,
            # and the id is final (not re-rendered from the family template).
            "nodes": (),
            "base_id": None,
            "id_template": None,
            "title_template": None,
        }
    )
    smoke_inventory = Inventory(nodes=inventory.nodes[: role.nodes_per_instance])
    return deployment.model_copy(update={"families": (smoke_family,)}), smoke_inventory


def smoke_done(ctx: RunContext) -> bool:
    return any(
        e.data.get("kind") == "smoke_done"
        for e in ctx.store.of_type(EventType.DECISION)
    )


def kernel_smoke(ctx: RunContext, rt: PhaseRuntime) -> None:
    """Walk the kernel ladder with dummy weights; record the verdict."""
    if smoke_done(ctx):
        return
    hw = ctx.require_hardware()
    inventory, deployment = hw.inventory, hw.deployment
    baseline = next(
        (
            f
            for f in search_context(ctx).families
            if all(r.role == "agg" for r in f.roles)
        ),
        None,
    )
    smoke = smoke_profile(deployment, baseline, inventory) if baseline else None
    if smoke is None:
        ctx.note("kernel smoke skipped: no aggregated family to smoke-test")
        ctx.store.append(EventType.DECISION, {"kind": "smoke_done", "ok": None})
        return
    smoke_dep, smoke_inv = smoke
    family = smoke_dep.families[0]
    kernel_rungs = [
        r.id
        for r in deployment.ladder
        if r.when == "kernel_unsupported" and "agg" in r.roles
    ]
    history = build_history(ctx.store)
    attempts = [d for d in history.deploys if d.candidate.family_id == family.id]
    cand = Candidate(family_id=family.id)
    if attempts and attempts[-1].failure == "kernel_unsupported":
        remaining = [r for r in kernel_rungs if r not in attempts[-1].candidate.rungs]
        cand = (
            attempts[-1].candidate.with_rung(remaining[0])
            if remaining
            else attempts[-1].candidate
        )
    while True:
        ctx.status(step=f"dummy-weight kernel smoke {cand.key}")
        result = rt.session.ensure(
            cand, deployment=smoke_dep, inventory=smoke_inv, require_text=False
        )
        rt.session.teardown()
        if result.ok:
            ctx.store.append(
                EventType.DECISION,
                {"kind": "smoke_done", "ok": True, "rungs": list(cand.rungs)},
            )
            return
        if result.failure != "kernel_unsupported":
            ctx.note(
                f"kernel smoke inconclusive ({result.failure}); the real deployments decide"
            )
            ctx.store.append(EventType.DECISION, {"kind": "smoke_done", "ok": None})
            return
        remaining = [r for r in kernel_rungs if r not in cand.rungs]
        if not remaining:
            reason = f"every kernel fallback failed on {ctx.gpu.name if ctx.gpu else 'this GPU'}: {result.detail[:400]}"
            ctx.store.append(
                EventType.DECISION,
                {"kind": "verdict", "verdict": "unsupported", "reason": reason},
            )
            ctx.store.append(EventType.DECISION, {"kind": "smoke_done", "ok": False})
            return
        cand = cand.with_rung(remaining[0])


def early_gpu_work(ctx: RunContext, rt: PhaseRuntime) -> None:
    """GPU work that needs no weights and no inter-node fabric."""
    run_group(ctx, "p2p")
    run_group(ctx, "collectives_intra")
    kernel_smoke(ctx, rt)


def prefetch_dataset_command(
    repo: str,
    *,
    python: str,
    hf_home: str,
    log_dir: str,
    hf_endpoint: str | None = None,
) -> str:
    """Background ``snapshot_download`` of an AIPerf dataset repo into ``hf_home``."""
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo!r}, repo_type='dataset')"
    )
    env = f"HF_HOME={shlex.quote(hf_home)}"
    if hf_endpoint:
        env += f" HF_ENDPOINT={shlex.quote(hf_endpoint)}"
    log = shlex.quote(f"{log_dir}/dataset_prefetch.log")
    # The subshell makes "&" background the download alone: "mkdir && cmd &"
    # backgrounds the whole list, whose shell keeps the caller's stdout open
    # until the download ends (node0 runs this as a local subprocess that
    # reads its output to EOF).
    return (
        f"mkdir -p {shlex.quote(log_dir)} && "
        f"({env} setsid nohup {shlex.quote(python)} -c "
        f"{shlex.quote(code)} > {log} 2>&1 < /dev/null &)"
    )


def prefetch_dataset(ctx: RunContext) -> None:
    """Warm node0's HF cache with the AIPerf corpus (best effort, background)."""
    repo = PUBLIC_DATASET_REPOS.get(ctx.config.dataset)
    if repo is None:
        return
    cmd = prefetch_dataset_command(
        repo,
        python=ctx.python,
        hf_home=ctx.hf_home,
        log_dir=str(ctx.exp_dir / "download"),
        hf_endpoint=ctx.config.hf_endpoint,
    )
    ctx.cluster.run(ctx.node0, cmd, check=False)


def index_shards(ctx: RunContext, model_dir: str) -> list[str]:
    """Shard files named by ``model.safetensors.index.json`` on node0 ([] if none)."""
    index = shlex.quote(f"{model_dir.rstrip('/')}/model.safetensors.index.json")
    res = ctx.cluster.run(ctx.node0, f"cat {index} 2>/dev/null", check=False)
    try:
        weight_map = json.loads(res.stdout or "{}").get("weight_map") or {}
    except (json.JSONDecodeError, AttributeError):
        return []
    return sorted({str(name) for name in weight_map.values()})


class DownloadPhase:
    name = "download"

    def run(self, ctx: RunContext, rt: PhaseRuntime) -> None:
        cfg = ctx.config
        inv_event = ctx.store.of_type(EventType.INVENTORY)[-1].data
        checkpoint = Checkpoint.model_validate(inv_event["checkpoint"])
        model_root = inv_event["model_root"]
        revision = (
            cfg.model_revision if cfg.model_revision != "main" else checkpoint.revision
        )
        nodes = ctx.node_names
        if ctx.simulator is not None:
            ctx.model_dir = cfg.model_path or model_dir_for(
                model_root, checkpoint.repo, "0" * 40
            )
            early_gpu_work(ctx, rt)
            collect_disk(ctx, ctx.model_dir)
            ctx.store.append(
                EventType.DOWNLOAD,
                {
                    "repo": checkpoint.repo,
                    "commit": "simulated",
                    "model_dir": ctx.model_dir,
                    "report": {},
                },
            )
            record_model_arch(ctx, ctx.model_dir)
            return
        if cfg.model_path:
            self._local_weights(ctx, rt, cfg.model_path, checkpoint.repo, revision)
            return

        fetcher = ModelFetcher(
            ctx.cluster,
            ctx.python,
            cfg.workdir,
            hf_cache=ctx.hf_home,
            control_ips={n.name: n.address for n in ctx.nodes},
            control_iface={
                n.name: n.control_interface
                for n in ctx.require_hardware().inventory.nodes
            },
            clock=ctx.clock,
            hf_endpoint=cfg.hf_endpoint,
        )
        plan_path = ctx.exp_dir / "download" / "plan.json"
        token = ctx.secrets.hf_token
        if plan_path.exists():
            with open(plan_path, "r", encoding="utf-8") as fh:
                plan = FetchPlan.model_validate(json.load(fh))
        else:
            ctx.status(step=f"resolving {checkpoint.repo}@{revision}")
            plan = fetcher.plan(checkpoint.repo, revision, token, nodes=nodes)
            atomic_write_json(plan_path, plan.model_dump(mode="json"))
        model_dir = model_dir_for(model_root, checkpoint.repo, plan.commit)
        ctx.model_dir = model_dir
        report: dict[str, object] = {
            "total_bytes": plan.total_bytes,
            "files": len(plan.files),
        }

        if fetcher.is_complete(plan, model_dir, nodes):
            early_gpu_work(ctx, rt)
            report["skipped"] = True
        else:
            ctx.status(step="downloading small files on every node")
            small = fetcher.fetch_small_files(plan, model_dir, nodes, token)
            handles = fetcher.start_downloads(plan, model_dir, nodes, token)
            prefetch_dataset(ctx)
            early_gpu_work(ctx, rt)
            ctx.status(
                step=f"waiting for shard downloads ({plan.total_bytes / 1e9:.0f} GB)"
            )
            shards = fetcher.wait_downloads(handles, timeout_s=DOWNLOAD_TIMEOUT_S)
            ctx.status(step="broadcasting shards over the RDMA fabric")
            gather = fetcher.gather(plan, model_dir, nodes, cfg.image)
            if not gather.ok:
                raise ModelFetchError(
                    "model gather failed",
                    problems=[
                        *gather.mismatches,
                        *gather.missing,
                        *(
                            f"{n}: exit {rc}"
                            for n, rc in gather.exit_codes.items()
                            if rc
                        ),
                    ],
                )
            fetcher.verify_and_mark(plan, model_dir, nodes)
            report.update(
                {
                    "small": {n: r.model_dump(mode="json") for n, r in small.items()},
                    "shards": {n: r.model_dump(mode="json") for n, r in shards.items()},
                    "gather_seconds": gather.seconds,
                    "gather_gbs": gather.gbs,
                    "bytes_broadcast": gather.bytes_broadcast,
                }
            )
        collect_disk(ctx, model_dir)
        ctx.store.append(
            EventType.DOWNLOAD,
            {
                "repo": plan.repo_id,
                "commit": plan.commit,
                "model_dir": model_dir,
                "report": report,
            },
        )
        record_model_arch(ctx, model_dir)

    def _local_weights(
        self,
        ctx: RunContext,
        rt: PhaseRuntime,
        model_dir: str,
        repo: str,
        revision: str,
    ) -> None:
        """--model-path: the weights are on every node already; check the copies."""
        nodes = ctx.node_names
        ctx.status(step=f"checking the weights in {model_dir} on {len(nodes)} nodes")
        listings = ctx.cluster.fanout(
            nodes,
            lambda node: ctx.cluster.run(
                node, completion_probe_command(model_dir), check=False
            ),
        )
        sizes = {node: local_file_sizes(res.stdout) for node, res in listings.items()}
        problems = local_copy_problems(
            sizes, ctx.node0, index_shards(ctx, model_dir), model_dir
        )
        if problems:
            raise ModelFetchError(
                f"--model-path {model_dir} is not the same complete model on every node",
                problems=problems,
            )
        ctx.model_dir = model_dir
        prefetch_dataset(ctx)
        early_gpu_work(ctx, rt)
        collect_disk(ctx, model_dir)
        files = sizes[ctx.node0] or {}
        ctx.store.append(
            EventType.DOWNLOAD,
            {
                "repo": repo,
                # The user's copy: its commit is known only when pinned.
                "commit": revision if COMMIT_RE.fullmatch(revision) else "local",
                "model_dir": model_dir,
                "report": {
                    "local": True,
                    "files": len(files),
                    "total_bytes": sum(files.values()),
                },
            },
        )
        record_model_arch(ctx, model_dir)
