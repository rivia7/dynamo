# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo + SGLang engine adapter.

``render`` turns a ``Candidate`` into a ``LaunchPlan``: GPU placement,
per-instance SGLang configs (recipe base + overrides + candidate variant +
ladder rungs), port plan, NIC restriction, Dynamo control-plane env, and
the exact ``docker run`` argv for every container. It is pure: the
deployment session writes the files and starts the containers.

Placement fills nodes in role order (prefill before decode), so prefill and
decode instances land on different nodes and KV transfer crosses the fabric.

P-Only, D-Only and A-Only families (domain/stages.py) get their stage
overrides last: P-Only workers run the prefill config as aggregated workers,
D-Only workers the decode config with the fake KV-transfer backend and the
decode-only shim, A-Only workers that config as DP attention without routed
experts, waiting for the simulated F the run's plan describes (domain/afd.py).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml

from benchmarks.cluster_bench.adapters.diagnostics import DEFAULT_WINDOW, nsys_env
from benchmarks.cluster_bench.adapters.docker import (
    LABEL_CAND,
    LABEL_EXP,
    LABEL_ROLE,
    Mount,
    docker_run_argv,
    render_env_file,
    sanitize_label,
)
from benchmarks.cluster_bench.adapters.profile_store import RecipeLoader
from benchmarks.cluster_bench.adapters.transports import get_transport
from benchmarks.cluster_bench.domain.nicplan import NodeNicPlan, plan_cluster
from benchmarks.cluster_bench.domain.nvlink import P2P_ENV, P2P_SGLANG
from benchmarks.cluster_bench.domain.profiles import (
    drop_flags,
    fit_family,
    merge_config,
    select_checkpoint,
)
from benchmarks.cluster_bench.domain.schema import (
    AfdPlan,
    Candidate,
    ContainerSpec,
    DeploymentProfile,
    FailureClass,
    FamilySpec,
    GpuProfile,
    Inventory,
    LadderRung,
    LaunchPlan,
    ModelProfile,
    RoleSpec,
    RunConfig,
    Stage,
)
from benchmarks.cluster_bench.domain.stages import (
    WorkerContext,
    apply_stage,
    stage_spec,
)

LAUNCH_SCRIPT = "/bench/src/benchmarks/cluster_bench/scripts/launch_node.sh"
# Store-true keys passed on the worker command line, not in its --config YAML:
# SGLang 0.5.23 made --enable-dp-attention a deprecated alias of --attn-dp-size
# that its YAML config loader rejects, and 0.5.21 has no --attn-dp-size. The
# flag works on both. rendered_configs keep these keys.
CLI_ONLY_KEYS = ("enable-dp-attention",)
CONTAINER_SRC = "/bench/src"
CONTAINER_EXP = "/bench/experiments"

ETCD_PORT = 2379
NATS_PORT = 4222
SYSTEM_PORT_BASE = 8081
SGLANG_PORT_BASE = 30000
DIST_INIT_BASE = 29500
KV_EVENTS_BASE = 20000
BOOTSTRAP_BASE = 12345
NIXL_TELEMETRY_BASE = 19090
HICACHE_HOST_FRACTION = 0.6
HICACHE_MIN_GB = 16

# SGLang ServerArgs field names for config keys whose names differ.
SERVER_ARG_ALIASES = {
    "tensor_parallel_size": "tp_size",
    "data_parallel_size": "dp_size",
    "pipeline_parallel_size": "pp_size",
    "expert_parallel_size": "ep_size",
}
# Keys whose live ServerArgs value must equal the rendered one. DP-attention
# sizes and max-running-requests are excluded: SGLang rewrites them at startup
# (attn_dp_size since sgl-project/sglang#41818, request caps by memory), so a
# literal comparison would fail healthy deployments.
ENGAGEMENT_KEYS = (
    "tensor-parallel-size",
    "ep-size",
    "context-length",
    "mem-fraction-static",
    "speculative-algorithm",
    "disaggregation-mode",
    "disaggregation-transfer-backend",
    "disaggregation-ib-device",
    "enable-hierarchical-cache",
    "hicache-size",
    "kv-cache-dtype",
)

FAILURE_PATTERNS: tuple[tuple[FailureClass, re.Pattern[str]], ...] = (
    (
        "oom",
        re.compile(
            r"CUDA out of memory|OutOfMemoryError|Not enough memory|"
            r"increase --mem-fraction-static|Cannot allocate memory|"
            r"max_total_num_tokens.*(too small|<)|OOMKilled",
            re.IGNORECASE,
        ),
    ),
    (
        "kernel_unsupported",
        re.compile(
            r"no kernel image is available|invalid device function|"
            r"not supported on (this )?(gpu|device|architecture)|unsupported (gpu|arch|sm)|"
            r"is not supported (for|on) (sm|compute capability|this)|"
            r"requires (sm ?90|sm ?100|hopper|blackwell)|only supports? sm",
            re.IGNORECASE,
        ),
    ),
    (
        "rdma_registration",
        re.compile(
            r"ibv_reg_(dmabuf_)?mr|Failed to register memory|NIXL_ERR_BACKEND|"
            r"transfer engine.*(fail|error)|Failed to (open|query) (rdma|ib) device",
            re.IGNORECASE,
        ),
    ),
    (
        "port_in_use",
        re.compile(r"Address already in use|EADDRINUSE", re.IGNORECASE),
    ),
    (
        "nccl",
        re.compile(
            r"NCCL error|ncclSystemError|ncclInternalError|ncclUnhandledCudaError|"
            r"NCCL WARN .*(fail|error|timeout)",
            re.IGNORECASE,
        ),
    ),
)


class RenderError(ValueError):
    pass


@dataclass
class _Placement:
    role: RoleSpec
    instance: int
    # (node name, GPU indices on that node), one entry per node rank.
    ranks: list[tuple[str, tuple[int, ...]]]


@dataclass
class _NodePorts:
    """Per-node port slots; one slot per container that needs ports."""

    next_slot: int = 0
    slots: dict[str, int] = field(default_factory=dict)

    def slot(self, container: str) -> int:
        if container not in self.slots:
            self.slots[container] = self.next_slot
            self.next_slot += 1
        return self.slots[container]


def _place(family: FamilySpec, inventory: Inventory) -> list[_Placement]:
    free: dict[str, list[int]] = {
        n.name: sorted(g.index for g in n.gpus) for n in inventory.nodes
    }
    full = {n.name: len(n.gpus) for n in inventory.nodes}
    order = [n.name for n in inventory.nodes]
    placements: list[_Placement] = []
    for role in family.roles:
        for instance in range(role.instances):
            if role.nodes_per_instance > 1:
                ranks = _take_whole_nodes(
                    free, full, order, role.nodes_per_instance, family.id
                )
            else:
                ranks = [_take_block(free, order, role.gpus_per_instance, family.id)]
            placements.append(_Placement(role=role, instance=instance, ranks=ranks))
    return placements


def _take_whole_nodes(
    free: dict[str, list[int]],
    full: dict[str, int],
    order: list[str],
    count: int,
    family_id: str,
) -> list[tuple[str, tuple[int, ...]]]:
    for start in range(len(order) - count + 1):
        window = order[start : start + count]
        if all(len(free[n]) == full[n] for n in window):
            taken = [(n, tuple(free[n])) for n in window]
            for n in window:
                free[n] = []
            return taken
    raise RenderError(
        f"{family_id}: no {count} consecutive free nodes for a multi-node instance"
    )


def _take_block(
    free: dict[str, list[int]], order: list[str], size: int, family_id: str
) -> tuple[str, tuple[int, ...]]:
    for node in order:
        gpus = free[node]
        # Prefer an aligned contiguous block (0-3, 4-7) so an instance stays
        # within one NUMA node / PCIe switch group.
        for start in range(0, len(gpus) - size + 1):
            block = gpus[start : start + size]
            if block[0] % size == 0 and block == list(range(block[0], block[0] + size)):
                free[node] = [g for g in gpus if g not in block]
                return node, tuple(block)
        if len(gpus) >= size:
            block = gpus[:size]
            free[node] = gpus[size:]
            return node, tuple(block)
    raise RenderError(f"{family_id}: not enough free GPUs for a {size}-GPU instance")


def _apply_rung(
    cfg: dict[str, Any], args: tuple[str, ...], env: dict[str, str], rung: LadderRung
):
    cfg = merge_config(cfg, rung.sglang)
    if rung.mem_fraction_delta and "mem-fraction-static" in cfg:
        cfg["mem-fraction-static"] = round(
            float(cfg["mem-fraction-static"]) + rung.mem_fraction_delta, 3
        )
    if rung.halve_cuda_graph_bs:
        # Configs arrive in SGLang 0.5.21+ names (schema.modernize_sglang_keys).
        if "cuda-graph-max-bs-decode" in cfg:
            limit = max(1, int(cfg["cuda-graph-max-bs-decode"]) // 2)
            cfg["cuda-graph-max-bs-decode"] = limit
            if isinstance(cfg.get("cuda-graph-bs-decode"), list):
                cfg["cuda-graph-bs-decode"] = [
                    b for b in cfg["cuda-graph-bs-decode"] if int(b) <= limit
                ]
    args = drop_flags(args, rung.drop_args) + rung.extra_args
    env = {**env, **rung.env}
    return cfg, args, env


def _hicache_clamp(
    cfg: dict[str, Any], node_mem_bytes: int, ranks_on_node: int
) -> tuple[dict[str, Any], str | None]:
    """Fit the per-rank host KV pool into the node's (cgroup) memory."""
    if not cfg.get("enable-hierarchical-cache") or "hicache-size" not in cfg:
        return cfg, None
    budget = int(HICACHE_HOST_FRACTION * node_mem_bytes / 1e9 / max(1, ranks_on_node))
    wanted = int(cfg["hicache-size"])
    if budget < HICACHE_MIN_GB:
        cfg = merge_config(
            cfg, {"enable-hierarchical-cache": None, "hicache-size": None}
        )
        return (
            cfg,
            f"HiCache disabled: {budget} GB/rank host budget < {HICACHE_MIN_GB} GB",
        )
    if budget < wanted:
        cfg = {**cfg, "hicache-size": budget}
        return cfg, f"HiCache clamped {wanted} -> {budget} GB/rank to fit host memory"
    return cfg, None


class SglangEngine:
    """Engine adapter for ``python3 -m dynamo.sglang`` workers."""

    def __init__(self, recipes: RecipeLoader | None = None):
        self.recipes = recipes or RecipeLoader()
        self.notes: list[str] = []

    # ------------------------------------------------------------------ render
    def render(
        self,
        candidate: Candidate,
        *,
        config: RunConfig,
        inventory: Inventory,
        gpu: GpuProfile,
        model: ModelProfile,
        deployment: DeploymentProfile,
        model_dir: str,
        afd: AfdPlan | None = None,
    ) -> LaunchPlan:
        self.notes = []
        gpn = len(inventory.nodes[0].gpus)
        family = fit_family(
            deployment.family(candidate.family_id), inventory.gpu_count, gpn
        )
        if family is None:
            raise RenderError(
                f"{candidate.family_id} does not fit {inventory.gpu_count} GPUs"
            )
        if stage_spec(family.stage).afd_side and afd is None:
            raise RenderError(
                f"{family.id}: an {stage_spec(family.stage).label} family needs the "
                "run's F plan (app/afd_ops.py)"
            )
        rungs = {r.id: r for r in deployment.ladder}
        unknown = [r for r in candidate.rungs if r not in rungs]
        if unknown:
            raise RenderError(f"unknown ladder rungs {unknown}")

        node0 = inventory.nodes[0]
        namespace = f"cb-{sanitize_label(config.exp_id)[:16]}-{candidate.fp}"
        exp_host = f"{config.workdir}/experiments/{config.exp_id}"
        deploy_host_dir = f"{exp_host}/deployments/{candidate.fp}"
        deploy_ctr_dir = f"{CONTAINER_EXP}/{config.exp_id}/deployments/{candidate.fp}"
        nic_plans = plan_cluster(inventory, candidate.nic_count)
        placements = _place(family, inventory)
        ports = {n.name: _NodePorts() for n in inventory.nodes}
        labels = {LABEL_EXP: sanitize_label(config.exp_id), LABEL_CAND: candidate.fp}
        mounts = [
            Mount(model_dir, model_dir, read_only=True),
            Mount(f"{config.workdir}/src", CONTAINER_SRC, read_only=True),
            Mount(f"{config.workdir}/experiments", CONTAINER_EXP),
            Mount(f"{config.workdir}/cache", "/root/.cache"),
        ]
        dynamo_common = {
            "DYN_NAMESPACE": namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "ETCD_ENDPOINTS": f"http://{node0.address}:{ETCD_PORT}",
            "NATS_SERVER": f"nats://{node0.address}:{NATS_PORT}",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

        containers: list[ContainerSpec] = []
        containers.append(
            self._control(
                config,
                node0.name,
                node0.address,
                labels,
                deploy_host_dir,
                deploy_ctr_dir,
            )
        )
        rendered: dict[str, dict[str, Any]] = {}
        ranks_per_node = self._hicache_ranks(placements)

        _, checkpoint = select_checkpoint(model, gpu, config.checkpoint)
        for gidx, placement in enumerate(placements):
            role = placement.role
            cfg, args, env = self._role_config(
                candidate,
                role,
                model,
                deployment,
                model_dir,
                rungs,
                family.stage,
                checkpoint.sglang,
                afd,
            )
            leader_node, _ = placement.ranks[0]
            leader_addr = inventory.node(leader_node).address
            nnodes = len(placement.ranks)
            for rank, (node_name, gpus) in enumerate(placement.ranks):
                node = inventory.node(node_name)
                name = f"cb-{candidate.fp}-{role.role}{placement.instance}-r{rank}"
                slot = ports[node_name].slot(name)
                rank_cfg, note = _hicache_clamp(
                    dict(cfg),
                    node.cgroup_mem_limit_bytes or node.mem_total_bytes,
                    ranks_per_node[node_name],
                )
                if note:
                    self.notes.append(f"{name}: {note}")
                rank_cfg["port"] = SGLANG_PORT_BASE + 100 * slot
                rank_cfg["nccl-port"] = SGLANG_PORT_BASE + 100 * slot + 50
                if "kv-events-config" in rank_cfg:
                    rank_cfg["kv-events-config"] = {
                        **dict(rank_cfg["kv-events-config"]),
                        "endpoint": f"tcp://*:{KV_EVENTS_BASE + 64 * slot}",
                    }
                if rank_cfg.get("disaggregation-mode") in ("prefill", "decode"):
                    rank_cfg["disaggregation-bootstrap-port"] = BOOTSTRAP_BASE + slot
                if nnodes > 1:
                    rank_cfg["nnodes"] = nnodes
                    rank_cfg["node-rank"] = rank
                    rank_cfg[
                        "dist-init-addr"
                    ] = f"{leader_addr}:{DIST_INIT_BASE + 20 * gidx}"
                hcas = nic_plans[node_name].hcas_for(gpus)
                rank_env = dict(env)
                if candidate.nic_count is not None or family.transport == "nixl":
                    restrict_env, restrict_cfg = self._restriction(
                        family, hcas, nic_plans[node_name]
                    )
                    rank_env.update(restrict_env)
                    rank_cfg = merge_config(rank_cfg, restrict_cfg)
                if candidate.p2p != "nvlink":
                    # NVLink comparison: no NVLS, no SGLang custom all-reduce
                    # (NVLink-only fast paths); NCCL P2P over PCIe or none.
                    rank_env.update(P2P_ENV[candidate.p2p])
                    rank_cfg = merge_config(rank_cfg, dict(P2P_SGLANG))
                system_port = SYSTEM_PORT_BASE + slot
                rank_env.update(dynamo_common)
                rank_env.update(
                    {
                        "DYN_SYSTEM_PORT": str(system_port),
                        "DYN_SYSTEM_ENABLED": "true",
                        "DYN_TCP_RPC_HOST": node.address,
                        "DYN_TCP_RESPONSE_STREAM_HOST": node.address,
                        "DYN_EVENT_PLANE_HOST": node.address,
                        "SGLANG_HOST_IP": node.address,
                        "NCCL_SOCKET_IFNAME": node.control_interface,
                        "GLOO_SOCKET_IFNAME": node.control_interface,
                    }
                )
                if candidate.diag == "nsys":
                    # scripts/launch_node.sh runs the worker under Nsight
                    # Systems when CB_NSYS_OUTPUT is set and nsys exists.
                    rank_env.update(nsys_env(config.exp_id, DEFAULT_WINDOW, name))
                if family.transport == "nixl":
                    rank_env.update(
                        {
                            "NIXL_TELEMETRY_ENABLE": "y",
                            "NIXL_TELEMETRY_EXPORTER": "prometheus",
                            "NIXL_TELEMETRY_PROMETHEUS_PORT": str(
                                NIXL_TELEMETRY_BASE + 16 * slot
                            ),
                        }
                    )
                cfg_host = f"{deploy_host_dir}/{name}.yaml"
                env_host = f"{deploy_host_dir}/{name}.env"
                yaml_cfg, cli_flags = split_cli_only(rank_cfg)
                command = (
                    "bash",
                    LAUNCH_SCRIPT,
                    "worker",
                    "--config",
                    f"{deploy_ctr_dir}/{name}.yaml",
                    *cli_flags,
                    *args,
                    *model.worker_args,
                )
                serving = rank == 0
                containers.append(
                    ContainerSpec(
                        node=node_name,
                        name=name,
                        role=role.role,
                        instance=placement.instance,
                        rank=rank,
                        gpus=gpus,
                        image=config.image,
                        env=rank_env,
                        files={
                            cfg_host: yaml.safe_dump(yaml_cfg, sort_keys=True),
                            env_host: render_env_file(rank_env),
                        },
                        command=command,
                        docker_run=tuple(
                            docker_run_argv(
                                name=name,
                                image=config.image,
                                command=command,
                                gpus=gpus,
                                env_file=env_host,
                                mounts=mounts,
                                labels={**labels, LABEL_ROLE: role.role},
                                rdma=bool(node.active_hcas),
                            )
                        ),
                        system_port=system_port,
                        health_url=f"http://{node.address}:{system_port}/health"
                        if serving
                        else None,
                        metrics_url=f"http://{node.address}:{system_port}/metrics"
                        if serving
                        else None,
                        hcas=hcas,
                    )
                )
                if serving:
                    rendered[f"{role.role}:{placement.instance}"] = rank_cfg

        frontend = self._frontend(
            config,
            node0.name,
            node0.address,
            labels,
            mounts,
            dynamo_common,
            deploy_host_dir,
        )
        containers.append(frontend)
        expected: dict[str, int] = {}
        for placement in placements:
            expected[placement.role.role] = expected.get(placement.role.role, 0) + 1
        return LaunchPlan(
            candidate=candidate,
            cand_fp=candidate.fp,
            namespace=namespace,
            served_model_name=model.served_model_name,
            frontend_url=f"http://{node0.address}:{config.frontend_port}",
            containers=tuple(containers),
            gpu_count=sum(len(gpus) for p in placements for _, gpus in p.ranks),
            transport=family.transport,
            cross_node=family.cross_node,
            expected_instances=expected,
            nic_selection={name: plan.hcas for name, plan in nic_plans.items()},
            rendered_configs=rendered,
            notes=tuple(self.notes),
            stage=family.stage,
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _hicache_ranks(placements: list[_Placement]) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for placement in placements:
            for node_name, gpus in placement.ranks:
                ranks[node_name] = ranks.get(node_name, 0) + len(gpus)
        return ranks

    def _role_config(
        self,
        candidate: Candidate,
        role: RoleSpec,
        model: ModelProfile,
        deployment: DeploymentProfile,
        model_dir: str,
        rungs: dict[str, LadderRung],
        stage: Stage = "e2e",
        checkpoint_sglang: Mapping[str, Any] | None = None,
        afd: AfdPlan | None = None,
    ) -> tuple[dict[str, Any], tuple[str, ...], dict[str, str]]:
        cfg = self.recipes.role_config(role)
        # What the served checkpoint needs (e.g. its quantization method).
        cfg = merge_config(cfg, checkpoint_sglang or {})
        cfg.update(
            {
                "model-path": model_dir,
                "served-model-name": model.served_model_name,
                "trust-remote-code": model.trust_remote_code,
                "context-length": deployment.served_context,
                "enable-metrics": True,
            }
        )
        cfg.pop("host", None)  # host networking; SGLang's own HTTP server is unused
        if not candidate.mtp:
            cfg = merge_config(cfg, {k: None for k in model.spec_decode_keys})
        if candidate.diag:
            # Diagnostic variant: SGLang's opt-in performance metrics
            # (verified in the SGLang 0.5.21 and 0.5.23 server args / environ).
            cfg["enable-mfu-metrics"] = True
            if model.moe:
                cfg["expert-balancedness-report-mode"] = "prometheus"
        args = tuple(role.extra_args)
        env = {**model.env, **role.env}
        if candidate.diag:
            env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "1"
        if (
            candidate.simulate_acceptance
            and candidate.mtp
            and role.role in ("agg", "decode")
        ):
            env.update(model.simulate_acceptance_env)
        for rung_id in candidate.rungs:
            rung = rungs[rung_id]
            if role.role in rung.roles:
                cfg, args, env = _apply_rung(cfg, args, env, rung)
        worker = WorkerContext(
            gpus=role.gpus_per_instance, afd=afd, null_f=candidate.null_f
        )
        cfg, env = apply_stage(stage, role.role, cfg, env, worker)
        return cfg, args, env

    @staticmethod
    def _restriction(
        family: FamilySpec, hcas: tuple[str, ...], plan: NodeNicPlan
    ) -> tuple[dict[str, str], dict[str, Any]]:
        env: dict[str, str] = {}
        cfg: dict[str, Any] = {}
        # NCCL is restricted for every family: multi-node replicas run their
        # TP/EP collectives over it, and single-node ones never touch IB.
        nccl_env, _ = get_transport("nccl").restrict(plan.hcas)
        env.update(nccl_env)
        if family.transport not in ("none", "nccl"):
            t_env, t_cfg = get_transport(family.transport).restrict(hcas)
            env.update(t_env)
            cfg.update(t_cfg)
        return env, cfg

    @staticmethod
    def _control(
        config: RunConfig,
        node: str,
        address: str,
        labels: dict[str, str],
        host_dir: str,
        ctr_dir: str,
    ) -> ContainerSpec:
        name = f"cb-{labels[LABEL_CAND]}-control"
        env = {
            "CB_ADVERTISE_IP": address,
            "CB_ETCD_PORT": str(ETCD_PORT),
            "CB_NATS_PORT": str(NATS_PORT),
        }
        env_host = f"{host_dir}/{name}.env"
        command = ("bash", LAUNCH_SCRIPT, "control")
        return ContainerSpec(
            node=node,
            name=name,
            role="control",
            image=config.image,
            env=env,
            files={env_host: render_env_file(env)},
            command=command,
            docker_run=tuple(
                docker_run_argv(
                    name=name,
                    image=config.image,
                    command=command,
                    gpus=None,
                    env_file=env_host,
                    mounts=[
                        Mount(f"{config.workdir}/src", CONTAINER_SRC, read_only=True)
                    ],
                    labels={**labels, LABEL_ROLE: "control"},
                    rdma=False,
                    shm_size="1g",
                )
            ),
            health_url=f"http://{address}:{ETCD_PORT}/health",
        )

    @staticmethod
    def _frontend(
        config: RunConfig,
        node: str,
        address: str,
        labels: dict[str, str],
        mounts: list[Mount],
        dynamo_common: dict[str, str],
        host_dir: str,
    ) -> ContainerSpec:
        name = f"cb-{labels[LABEL_CAND]}-frontend"
        env = {
            **dynamo_common,
            "DYN_HTTP_PORT": str(config.frontend_port),
            "DYN_ROUTER_MODE": "kv",
            "DYN_TCP_RPC_HOST": address,
            "DYN_TCP_RESPONSE_STREAM_HOST": address,
            "DYN_EVENT_PLANE_HOST": address,
        }
        env_host = f"{host_dir}/{name}.env"
        command = (
            "bash",
            LAUNCH_SCRIPT,
            "frontend",
            "--router-mode",
            "kv",
            "--trust-remote-code",
        )
        return ContainerSpec(
            node=node,
            name=name,
            role="frontend",
            image=config.image,
            env=env,
            files={env_host: render_env_file(env)},
            command=command,
            docker_run=tuple(
                docker_run_argv(
                    name=name,
                    image=config.image,
                    command=command,
                    gpus=None,
                    env_file=env_host,
                    mounts=mounts,
                    labels={**labels, LABEL_ROLE: "frontend"},
                    rdma=False,
                    shm_size="8g",
                )
            ),
            health_url=f"http://{address}:{config.frontend_port}/health",
            metrics_url=f"http://{address}:{config.frontend_port}/metrics",
        )

    # ------------------------------------------------------------ log readers
    def engagement_checks(
        self, rendered: dict[str, Any], server_args: dict[str, str]
    ) -> dict[str, bool]:
        return config_engagement(rendered, server_args)

    def classify_failure(self, log_text: str) -> FailureClass:
        for failure, pattern in FAILURE_PATTERNS:
            if pattern.search(log_text):
                return failure
        return "unknown"

    def parse_server_args(self, log_text: str) -> dict[str, str]:
        """Parse the last server-args dump SGLang logs at startup.

        SGLang 0.5.21 and newer log ``server_args={'key': value, ...}``
        (``ServerArgs.resolved_dict()``), older releases ``ServerArgs(key=value,
        ...)``. Values stay Python reprs either way.
        """
        dict_at = log_text.rfind(SERVER_ARGS_DICT)
        repr_at = log_text.rfind(SERVER_ARGS_REPR)
        if dict_at < 0 and repr_at < 0:
            return {}
        if dict_at > repr_at:
            body, sep = log_text[dict_at + len(SERVER_ARGS_DICT) :], ":"
        else:
            body, sep = log_text[repr_at + len(SERVER_ARGS_REPR) :], "="
        parsed: dict[str, str] = {}
        for item in _top_level_items(body):
            if sep in item:
                key, value = item.split(sep, 1)
                parsed[key.strip().strip("'\"")] = value.strip()
        return parsed


def normalize_value(value: Any) -> str:
    """Canonical string for comparing a rendered config value with a ServerArgs repr."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    if text.lower() in ("true", "false"):
        return text.capitalize()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number == int(number) and "." not in text:
        return str(int(number))
    return repr(number)


SERVER_ARGS_DICT = "server_args={"
SERVER_ARGS_REPR = "ServerArgs("


def split_cli_only(cfg: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The YAML part of a worker config and the ``CLI_ONLY_KEYS`` flags it held."""
    yaml_cfg = dict(cfg)
    flags: list[str] = []
    for key in CLI_ONLY_KEYS:
        if key not in yaml_cfg:
            continue
        value = yaml_cfg.pop(key)
        if value is True or str(value).strip().lower() in ("true", "1", "yes"):
            flags.append(f"--{key}")
    return yaml_cfg, tuple(flags)


def _top_level_items(body: str) -> list[str]:
    """Comma-separated items of a repr body up to its closing bracket."""
    depth = 0
    quote: str | None = None
    items: list[str] = []
    current: list[str] = []
    for ch in body:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            if depth == 0:
                items.append("".join(current))
                break
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    return items


def config_engagement(
    rendered: dict[str, Any], server_args: dict[str, str]
) -> dict[str, bool]:
    """Per critical key: did the live ServerArgs take the rendered value?

    Keys absent from ServerArgs (renamed across SGLang versions) are skipped
    rather than failed; a present key with a different value fails.
    """
    checks: dict[str, bool] = {}
    for key in ENGAGEMENT_KEYS:
        if key not in rendered:
            continue
        field_name = key.replace("-", "_")
        field_name = SERVER_ARG_ALIASES.get(field_name, field_name)
        if field_name not in server_args:
            continue
        checks[key] = normalize_value(rendered[key]) == normalize_value(
            server_args[field_name]
        )
    return checks
