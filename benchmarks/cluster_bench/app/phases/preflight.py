# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight: replicate code to peers, bootstrap their venvs, probe hardware,
pick profiles and the weights disk, pull the image, and verify that every
flag the run will use exists in the pinned image and AIPerf version.

Everything that can fail cheaply fails here, before any GPU-hour is spent.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path

from benchmarks.cluster_bench.adapters import aiperf as aiperf_adapter
from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.docker import Mount, docker_run_argv
from benchmarks.cluster_bench.adapters.engine_sglang import CONTAINER_SRC
from benchmarks.cluster_bench.agents.sglang_hotfix import ENV_VAR as HOTFIX_ENV
from benchmarks.cluster_bench.agents.sglang_hotfix import parse_ids as parse_hotfixes
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.app.facts_ops import collect_preflight
from benchmarks.cluster_bench.domain import afd
from benchmarks.cluster_bench.domain import profiles as prof
from benchmarks.cluster_bench.domain import stages
from benchmarks.cluster_bench.domain.datasets import parse_dataset
from benchmarks.cluster_bench.domain.inventory import (
    check_uniform,
    driver_major,
    fingerprint,
    node_from_probe,
    pick_model_root,
)
from benchmarks.cluster_bench.domain.schema import (
    DeploymentProfile,
    EventType,
    GpuProfile,
    Inventory,
    ModelProfile,
)
from benchmarks.cluster_bench.fsutil import atomic_write_text
from benchmarks.cluster_bench.ports import NodeAddress

logger = logging.getLogger(__name__)

MIN_DRIVER_MAJOR = 580  # sglang-runtime:1.6.2 is built on CUDA 13.0
WEIGHTS_HEADROOM = 1.15
# Keys the renderer adds to every SGLang config regardless of the profile.
HARNESS_SGLANG_KEYS = (
    "model-path",
    "served-model-name",
    "trust-remote-code",
    "context-length",
    "enable-metrics",
    "port",
    "nccl-port",
    "load-format",  # dummy-weight kernel smoke
)
MULTINODE_KEYS = ("nnodes", "node-rank", "dist-init-addr")
CONTAINER_AGENTS = f"{CONTAINER_SRC}/benchmarks/cluster_bench/agents"


class PreflightError(RuntimeError):
    pass


def profile_hotfixes(
    model: ModelProfile, deployment: DeploymentProfile
) -> tuple[str, ...]:
    """SGLang hotfix ids the model env and the families' role envs name, in order."""
    envs = [model.env] + [r.env for f in deployment.families for r in f.roles]
    return parse_hotfixes(",".join(e.get(HOTFIX_ENV, "") for e in envs))


def collect_family_flags(
    deployment: DeploymentProfile,
    model: ModelProfile,
    recipes: profile_store.RecipeLoader,
) -> dict[str, set[str]]:
    """Every CLI flag each enabled family would pass to dynamo.sglang."""
    flags: dict[str, set[str]] = {}
    for family in deployment.families:
        used: set[str] = {f"--{k}" for k in HARNESS_SGLANG_KEYS}
        used |= stages.stage_flags(family.stage)
        used |= {a for a in model.worker_args if a.startswith("--")}
        if family.transport == "mooncake":
            used.add("--disaggregation-ib-device")
        for role in family.roles:
            cfg = recipes.role_config(role)
            used |= {f"--{k}" for k in cfg}
            used |= {a for a in role.extra_args if a.startswith("--")}
            if role.nodes_per_instance > 1:
                used |= {f"--{k}" for k in MULTINODE_KEYS}
            disaggregated = stages.stage_spec(family.stage).disaggregated
            if role.role in ("prefill", "decode") and disaggregated:
                used.add("--disaggregation-bootstrap-port")
        for rung in deployment.ladder:
            used |= {f"--{k}" for k, v in rung.sglang.items() if v is not None}
            used |= {a for a in rung.extra_args if a.startswith("--")}
        flags[family.id] = used
    return flags


def missing_in_help(flags: set[str], help_text: str) -> list[str]:
    present = set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))
    return sorted(f for f in flags if f not in present)


def probe_command(ctx: RunContext, addr: NodeAddress) -> str:
    """probe.py for one node; "auto" resolves the NIC that holds its address."""
    agent = f"{ctx.src_dir}/benchmarks/cluster_bench/agents/probe.py"
    iface = addr.control_interface or ctx.config.control_interface
    argv = [ctx.python, agent, "--control-interface", iface]
    if iface == "auto":
        argv += ["--control-address", addr.address]
    return shlex.join(argv)


def probe_inventory(ctx: RunContext) -> Inventory:
    commands = {addr.name: probe_command(ctx, addr) for addr in ctx.nodes}
    outputs = ctx.cluster.fanout(
        ctx.node_names,
        lambda node: ctx.cluster.run(node, commands[node], timeout=180).stdout,
    )
    nodes = []
    for addr in ctx.nodes:
        nodes.append(
            node_from_probe(addr.name, addr.address, json.loads(outputs[addr.name]))
        )
    return Inventory(nodes=tuple(nodes))


def resolve_profiles(
    ctx: RunContext, inventory: Inventory
) -> tuple[GpuProfile, ModelProfile, DeploymentProfile]:
    cfg = ctx.config
    gpus = profile_store.load_gpu_profiles()
    if cfg.gpu_profile:
        if cfg.gpu_profile not in gpus:
            raise PreflightError(
                f"unknown --gpu-profile {cfg.gpu_profile!r}; known {sorted(gpus)}"
            )
        gpu = gpus[cfg.gpu_profile]
    else:
        gpu = prof.detect_gpu_profile(inventory.nodes[0].gpus[0].name, gpus)
    gpn = len(inventory.nodes[0].gpus)
    recipes = profile_store.RecipeLoader()
    if cfg.model_repo:
        if cfg.model_size_gb is None:
            raise PreflightError("--model-repo needs --model-size-gb")
        model = prof.generic_model_profile(
            cfg.model_repo, cfg.model_revision, cfg.model_size_gb
        )
        deployment = prof.generic_deployment_profile(
            gpu,
            model,
            nodes=len(inventory.nodes),
            gpus_per_node=gpn,
            served_context=cfg.served_context or 131072,
            # Single-role stages derive their family from the PD split, which
            # a single node can also hold.
            pd_min_nodes=2 if stages.stage_role(cfg.stage) is None else 1,
        )
    else:
        model = profile_store.load_model_profile(cfg.model)
        deployment = profile_store.load_deployment_profile(cfg.model, gpu.name)
        if cfg.served_context:
            deployment = deployment.model_copy(
                update={"served_context": cfg.served_context}
            )
    deployment = stages.stage_deployment(deployment, cfg.stage, recipes.role_config)
    if not deployment.families:
        raise PreflightError(
            f"{deployment.model}/{deployment.gpu} has no "
            f"{stages.stage_label(cfg.stage)} family (P-Only, D-Only and A-Only "
            "derive theirs from PD families; the profile has none)"
        )
    fitted, dropped = prof.fit_deployment(deployment, inventory.gpu_count, gpn)
    if not fitted.families:
        raise PreflightError(
            f"no {stages.stage_label(cfg.stage)} family of "
            f"{deployment.model}/{deployment.gpu} fits "
            f"{len(inventory.nodes)} nodes x {gpn} GPUs"
            + (
                f" (too large or not for this size: {', '.join(dropped)})"
                if dropped
                else ""
            )
        )
    if dropped:
        ctx.note(
            f"families not applicable to {len(inventory.nodes)} nodes x {gpn} GPUs: "
            + ", ".join(dropped)
        )
    return gpu, model, fitted


def restore(ctx: RunContext) -> None:
    """Rebuild in-memory hardware/profile state from the log (resume)."""
    events = ctx.store.of_type(EventType.INVENTORY)
    if not events:
        return
    data = events[-1].data
    ctx.inventory = Inventory.model_validate(data["inventory"])
    ctx.gpu = GpuProfile.model_validate(data["gpu"])
    ctx.model = ModelProfile.model_validate(data["model"])
    ctx.deployment = DeploymentProfile.model_validate(data["deployment"])
    downloads = ctx.store.of_type(EventType.DOWNLOAD)
    if downloads:
        ctx.model_dir = downloads[-1].data["model_dir"]


def replicate_code(ctx: RunContext) -> list[str]:
    """Copy node0's harness code (agents, launch script) to every peer."""
    peers = ctx.node_names[1:]
    if peers:
        ctx.status(step="replicating harness code to peers")
        ctx.cluster.push_tree(Path(ctx.src_dir), peers, ctx.config.workdir)
    return peers


def resume(ctx: RunContext) -> None:
    """Start of every run: the state the log holds; on a resumed run (past
    preflight) the code is replicated again, since the runner may now carry
    newer code (agents, launch script) than the peers got at preflight."""
    restore(ctx)
    if ctx.store.of_type(EventType.INVENTORY):
        replicate_code(ctx)


class PreflightPhase:
    name = "preflight"

    def run(self, ctx: RunContext, rt: object) -> None:
        cfg = ctx.config
        peers = replicate_code(ctx)
        if peers:
            boot = (
                f"bash {shlex.quote(ctx.src_dir)}/benchmarks/cluster_bench/scripts/bootstrap_node.sh "
                f"--workdir {shlex.quote(cfg.workdir)} --minimal"
            )
            ctx.status(step="bootstrapping peer venvs")
            ctx.cluster.fanout(
                peers, lambda node: ctx.cluster.run(node, boot, timeout=1800)
            )

        ctx.status(step="probing hardware")
        inventory = probe_inventory(ctx)
        problems = check_uniform(inventory)
        for node in inventory.nodes:
            if driver_major(node.driver_version) < MIN_DRIVER_MAJOR:
                problems.append(
                    f"{node.name}: driver {node.driver_version} < {MIN_DRIVER_MAJOR} (CUDA 13 image)"
                )
            if not node.active_hcas:
                problems.append(
                    f"{node.name}: no ACTIVE RDMA HCA; the fabric is unusable"
                )
            if node.docker_version is None:
                problems.append(f"{node.name}: docker is not available")
        if problems:
            raise PreflightError("preflight failed:\n  " + "\n  ".join(problems))
        for node in inventory.nodes:
            if not node.nvidia_peermem:
                ctx.note(
                    f"{node.name}: nvidia_peermem not loaded; GPU-direct RDMA may fall back to dmabuf or host staging"
                )

        gpu, model, deployment = resolve_profiles(ctx, inventory)
        spec = stages.stage_spec(cfg.stage)
        # a --model-repo profile does not know: the F plan reads config.json
        if spec.needs_moe and not model.moe and model.name != prof.GENERIC_MODEL:
            raise PreflightError(
                f"{spec.label} needs a MoE model; {model.name} has no routed experts"
            )
        try:
            variant, checkpoint = prof.select_checkpoint(model, gpu, cfg.checkpoint)
            if spec.afd_side:
                afd.routed_dtype(variant, cfg.afd_f.expert_dtype)
        except (prof.ProfileError, afd.AfdError) as exc:
            raise PreflightError(str(exc)) from exc
        model_root: str | None = None
        if not cfg.model_path:
            # --model-path weights are already on every node: no space needed;
            # the download phase checks that the copies agree.
            need = int(checkpoint.approx_size_gb * 1e9 * WEIGHTS_HEADROOM)
            model_root = pick_model_root(
                inventory.nodes, need, cfg.model_root, platform=cfg.platform
            )
        ctx.store.append(
            EventType.INVENTORY,
            {
                "inventory": inventory.model_dump(mode="json"),
                "fingerprint": fingerprint(inventory),
                "gpu": gpu.model_dump(mode="json"),
                "model": model.model_dump(mode="json"),
                "deployment": deployment.model_dump(mode="json"),
                "checkpoint": checkpoint.model_dump(mode="json"),
                "model_root": model_root,
            },
        )
        restore(ctx)

        ctx.status(step=f"pulling {cfg.image} on {len(ctx.nodes)} nodes")
        ctx.cluster.fanout(
            ctx.node_names, lambda node: ctx.docker.pull(node, cfg.image)
        )
        # Before any deployment: the software probe opens a CUDA context on GPU 0.
        collect_preflight(ctx)
        self._check_flags(ctx, model, deployment)

    def _check_in_image(
        self,
        ctx: RunContext,
        *,
        name: str,
        command: list[str],
        gpus: list[int] | None,
        timeout: int,
        artifact: str,
        failure: str,
    ) -> None:
        """Run a check in the image on node0 with the harness sources mounted;
        its output goes to preflight/<artifact>, a non-zero exit fails."""
        argv = docker_run_argv(
            name=f"{name}-{ctx.config.exp_id[:8]}",
            image=ctx.config.image,
            command=command,
            gpus=gpus,
            mounts=(Mount(f"{ctx.config.workdir}/src", CONTAINER_SRC, read_only=True),),
            detach=False,
            remove=True,
            rdma=False,
        )
        res = ctx.cluster.run(
            ctx.node0, shlex.join(argv) + " 2>&1", timeout=timeout, check=False
        )
        output = res.stdout + res.stderr
        atomic_write_text(ctx.exp_dir / "preflight" / artifact, output)
        if res.rc != 0:
            raise PreflightError(f"{failure} (rc {res.rc}): {output.strip()[-600:]}")

    def _check_decode_only_shim(self, ctx: RunContext) -> None:
        """The D-Only shim patches dynamo.sglang internals: prove they exist."""
        self._check_in_image(
            ctx,
            name="cb-donly",
            command=[
                "python3",
                "-P",
                f"{CONTAINER_AGENTS}/sglang_decode_only.py",
                "--check",
            ],
            gpus=[0],
            timeout=600,
            artifact="decode_only_shim.txt",
            failure="D-Only needs the decode-only shim, which does not fit this image",
        )

    def _check_afd_kernel(self, ctx: RunContext) -> None:
        """A-Only's F wait is a Triton kernel: compile and time it on GPU 0."""
        self._check_in_image(
            ctx,
            name="cb-afd",
            command=["python3", "-P", f"{CONTAINER_AGENTS}/sglang_afd.py", "--check"],
            gpus=[0],
            timeout=900,
            artifact="afd_wait_kernel.txt",
            failure="A-Only's F wait kernel does not work in this image",
        )

    def _check_sglang_hotfixes(
        self, ctx: RunContext, model: ModelProfile, deployment: DeploymentProfile
    ) -> None:
        """Hotfixes edit SGLang sources at launch: prove each fits this image."""
        ids = parse_hotfixes(
            ",".join(
                (
                    *profile_hotfixes(model, deployment),
                    *stages.stage_spec(ctx.config.stage).hotfixes,
                )
            )
        )
        if ids:
            self._check_in_image(
                ctx,
                name="cb-hotfix",
                command=[
                    "python3",
                    "-P",
                    f"{CONTAINER_AGENTS}/sglang_hotfix.py",
                    "check",
                    ",".join(ids),
                ],
                gpus=None,
                timeout=600,
                artifact="sglang_hotfixes.txt",
                failure=f"SGLang hotfixes {','.join(ids)} do not fit this image",
            )

    def _check_flags(
        self, ctx: RunContext, model: ModelProfile, deployment: DeploymentProfile
    ) -> None:
        out_dir = ctx.exp_dir / "preflight"
        res = ctx.cluster.run(
            ctx.node0,
            f"{shlex.quote(ctx.aiperf_bin)} profile --help",
            timeout=300,
            check=False,
        )
        atomic_write_text(out_dir / "aiperf_help.txt", res.stdout + res.stderr)
        missing = aiperf_adapter.missing_flags(
            res.stdout + res.stderr,
            aiperf_adapter.required_flags(parse_dataset(ctx.config.dataset)),
        )
        if missing:
            raise PreflightError(
                f"aiperf {ctx.config.aiperf_version} lacks required flags: {missing}"
            )

        argv = docker_run_argv(
            name=f"cb-help-{ctx.config.exp_id[:8]}",
            image=ctx.config.image,
            command=["python3", "-m", "dynamo.sglang", "--help"],
            gpus=[0],
            detach=False,
            remove=True,
            rdma=False,
        )
        res = ctx.cluster.run(
            ctx.node0, shlex.join(argv) + " 2>&1", timeout=600, check=False
        )
        help_text = res.stdout + res.stderr
        atomic_write_text(out_dir / "dynamo_sglang_help.txt", help_text)
        spec = stages.stage_spec(ctx.config.stage)
        if spec.fake_kv:
            self._check_decode_only_shim(ctx)
        if spec.afd_side:
            self._check_afd_kernel(ctx)
        self._check_sglang_hotfixes(ctx, model, deployment)
        flags = collect_family_flags(deployment, model, profile_store.RecipeLoader())
        baseline = min(deployment.families, key=lambda f: f.priority)
        for family in deployment.families:
            absent = missing_in_help(flags[family.id], help_text)
            if not absent:
                continue
            if family.id == baseline.id:
                raise PreflightError(
                    f"image lacks flags used by the baseline {family.id}: {absent}"
                )
            ctx.store.append(
                EventType.DECISION,
                {
                    "kind": "close_family",
                    "family_id": family.id,
                    "status": "unsupported",
                    "reason": f"image lacks flags {absent}",
                },
            )
