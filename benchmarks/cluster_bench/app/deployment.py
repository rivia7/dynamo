# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment lifecycle: render -> write -> start -> ready -> engage -> teardown.

Readiness is polled on observable conditions (container state, worker
``/health``, frontend ``/v1/models``, a real completion), never assumed after
a fixed sleep. Any container exit during bring-up fails fast with the log
tail classified into a ``FailureClass`` that drives the profile's ladder.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.cluster_bench.adapters.docker import experiment_labels
from benchmarks.cluster_bench.adapters.transports import get_transport
from benchmarks.cluster_bench.agents.sglang_afd import MARKER as AFD_MARKER
from benchmarks.cluster_bench.app.afd_ops import ensure_plan, plan_instance_gpus
from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.domain import afd
from benchmarks.cluster_bench.domain.lifecycle import DeployLifecycle, deploy_lifecycle
from benchmarks.cluster_bench.domain.profiles import select_checkpoint
from benchmarks.cluster_bench.domain.schema import (
    Candidate,
    ContainerSpec,
    DeploymentProfile,
    DeployResult,
    EngagementResult,
    EventType,
    FailureClass,
    Inventory,
    LaunchPlan,
)
from benchmarks.cluster_bench.domain.stages import DECODE_ONLY_ENV, stage_spec
from benchmarks.cluster_bench.fsutil import atomic_write_text
from benchmarks.cluster_bench.ports import CommandError

logger = logging.getLogger(__name__)

READY_TIMEOUT_S = 35 * 60
POLL_S = 10.0
GPU_FREE_MIB = 2048
GPU_FREE_TIMEOUT_S = 180.0
ENV_KEYS_TO_VERIFY = (
    "NCCL_IB_HCA",
    "UCX_NET_DEVICES",
    "NVSHMEM_HCA_LIST",
    "MC_FORCE_HCA",
    "DYN_TCP_RPC_HOST",
    "SGLANG_HOST_IP",
    "SGLANG_SIMULATE_ACC_LEN",
    DECODE_ONLY_ENV,
    afd.ATTN_ONLY_ENV,
    afd.TARGET_LAYERS_ENV,
    afd.F_WAIT_ENV,
)
# The worker process whose environment proves the rendered env (pgrep -f, ERE).
# Under nsys the launcher execs it by absolute path (/opt/sglang/bin/python3).
WORKER_PROCESS = (
    r"^(/[^ ]*/)?python3 (-m dynamo[.]sglang|-P [^ ]*sglang_decode_only[.]py)"
)
SMOKE_PROMPT = "Reply with the single word OK."


class DeployFailed(RuntimeError):
    def __init__(self, failure: FailureClass, detail: str):
        super().__init__(f"{failure}: {detail[:500]}")
        self.failure = failure
        self.detail = detail


@dataclass
class DeploymentSession:
    ctx: RunContext
    plan: LaunchPlan | None = None
    # Lifecycle of the running deployment (KV capacity, max running requests).
    lifecycle: DeployLifecycle | None = None
    _restarts: dict[str, int] = field(default_factory=dict)
    # Containers started during the current bring-up; only these can "exit".
    _started: set[str] = field(default_factory=set)

    @property
    def deployed_key(self) -> str | None:
        return self.plan.candidate.key if self.plan else None

    # ----------------------------------------------------------------- public
    def ensure(
        self,
        candidate: Candidate,
        *,
        deployment: DeploymentProfile | None = None,
        inventory: Inventory | None = None,
        require_text: bool = True,
        image: str | None = None,
    ) -> DeployResult:
        """Make ``candidate`` the running deployment; records a DEPLOY event.

        ``deployment``/``inventory`` override the run's profile and cluster for
        the dummy-weight kernel smoke, which runs one instance on a node subset;
        ``image`` overrides the container image (diagnostic image with nsys).
        """
        if (
            self.plan is not None
            and self.plan.candidate.key == candidate.key
            and self.alive()
        ):
            return DeployResult(candidate=candidate, cand_fp=candidate.fp, ok=True)
        self.teardown()
        hw = self.ctx.require_hardware()
        inventory = inventory or hw.inventory
        deployment = deployment or hw.deployment
        config = (
            self.ctx.config.model_copy(update={"image": image})
            if image
            else self.ctx.config
        )
        plan = self.ctx.engine.render(
            candidate,
            config=config,
            inventory=inventory,
            gpu=hw.gpu,
            model=hw.model,
            deployment=deployment,
            model_dir=hw.model_dir,
            afd=ensure_plan(self.ctx),
        )
        log_dir = self.ctx.exp_dir / "deployments" / plan.cand_fp
        atomic_write_text(log_dir / "plan.json", plan.model_dump_json(indent=2))
        started = self.ctx.clock.now()
        self.ctx.status(step=f"deploying {candidate.key}")
        if self.ctx.simulator is not None:
            self.ctx.simulator.set_plan(plan)
        try:
            self._start(plan)
            self._wait_ready(plan, started)
            # Decode over fake KV: its tokens are meaningless, possibly an
            # immediate EOS, so only the HTTP round trip is checked.
            self._smoke(plan, require_text and not stage_spec(plan.stage).fake_kv)
            engagement = self._engagement(plan)
        except DeployFailed as exc:
            texts = self._save_logs(plan, log_dir)
            self.teardown_plan(plan)
            result = DeployResult(
                candidate=candidate,
                cand_fp=plan.cand_fp,
                ok=False,
                load_seconds=self.ctx.clock.now() - started,
                failure=exc.failure,
                detail=self.ctx.redactor(exc.detail[-4000:]),
                log_dir=str(log_dir),
                lifecycle=self._lifecycle(plan, texts, started, log_dir),
            )
            self.plan = None
            self.ctx.store.append(
                EventType.DEPLOY, {"result": result.model_dump(mode="json")}
            )
            return result
        texts = self._save_logs(plan, log_dir)
        self.plan = plan
        self.lifecycle = self._lifecycle(plan, texts, started, log_dir)
        self._restarts = self._restart_counts(plan)
        if stage_spec(plan.stage).afd_side:
            engagement = self._afd_engagement(plan, self.lifecycle, engagement)
        ok = engagement.ok
        result = DeployResult(
            candidate=candidate,
            cand_fp=plan.cand_fp,
            ok=ok,
            load_seconds=self.ctx.clock.now() - started,
            failure=None if ok else "engagement",
            detail=""
            if ok
            else f"not engaged: {[k for k, v in engagement.checks.items() if not v]}",
            engagement=engagement,
            log_dir=str(log_dir),
            lifecycle=self.lifecycle,
        )
        if not ok:
            self.teardown()
        self.ctx.store.append(
            EventType.DEPLOY, {"result": result.model_dump(mode="json")}
        )
        for note in plan.notes:
            self.ctx.note(note, cand_key=candidate.key)
        return result

    def alive(self) -> bool:
        """All containers of the current plan running with unchanged restart counts."""
        if self.plan is None:
            return False
        for spec in self.plan.containers:
            state = self.ctx.docker.state(spec.node, spec.name)
            if state is None or not state.running:
                return False
            if state.restart_count != self._restarts.get(
                spec.name, state.restart_count
            ):
                return False
        return True

    def teardown(self) -> None:
        if self.plan is not None:
            self.teardown_plan(self.plan)
        else:
            self._remove_labeled()
        self.plan = None
        self.lifecycle = None

    def teardown_plan(self, plan: LaunchPlan) -> None:
        self._remove_labeled()
        self._wait_gpus_free()

    # ---------------------------------------------------------------- internals
    def _remove_labeled(self) -> None:
        label = experiment_labels(self.ctx.config.exp_id)
        self.ctx.cluster.fanout(
            self.ctx.node_names, lambda node: self.ctx.docker.remove(node, label)
        )

    def _wait_gpus_free(self) -> None:
        deadline = self.ctx.clock.now() + GPU_FREE_TIMEOUT_S

        def busy(node: str) -> list[int]:
            res = self.ctx.cluster.run(
                node,
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                check=False,
            )
            values = [int(v) for v in res.stdout.split() if v.strip().isdigit()]
            return [i for i, used in enumerate(values) if used > GPU_FREE_MIB]

        while True:
            pending = {
                n: b
                for n, b in self.ctx.cluster.fanout(self.ctx.node_names, busy).items()
                if b
            }
            if not pending:
                return
            if self.ctx.clock.now() > deadline:
                # Kill only leftover engine processes (zombie SGLang schedulers),
                # never arbitrary GPU users.
                for node in pending:
                    self.ctx.cluster.run(node, _KILL_ENGINE_ZOMBIES, check=False)
                self.ctx.note(
                    "GPU memory not released after teardown; killed leftover engine processes",
                    nodes=sorted(pending),
                )
                return
            self.ctx.clock.sleep(POLL_S)

    def _start(self, plan: LaunchPlan) -> None:
        """Write files and start containers; command failures become DeployFailed
        so the planner sees a failed deployment instead of the run aborting."""
        self._started = set()
        try:
            self._start_containers(plan)
        except CommandError as exc:
            raise DeployFailed("container_exit", str(exc)) from exc

    def _start_containers(self, plan: LaunchPlan) -> None:
        for spec in plan.containers:
            for path, content in spec.files.items():
                self.ctx.cluster.put_text(spec.node, path, content)
        control = [c for c in plan.containers if c.role == "control"]
        frontend = [c for c in plan.containers if c.role == "frontend"]
        workers = [c for c in plan.containers if c.role not in ("control", "frontend")]
        for spec in control:
            self._run(spec)
            self._poll(
                lambda s=spec: self._http_ok(s.health_url),
                plan,
                120,
                f"etcd at {spec.health_url}",
            )
        by_node: dict[str, list[ContainerSpec]] = {}
        for spec in workers:
            by_node.setdefault(spec.node, []).append(spec)
        self.ctx.cluster.fanout(
            list(by_node), lambda node: [self._run(s) for s in by_node[node]]
        )
        for spec in frontend:
            self._run(spec)

    def _run(self, spec: ContainerSpec) -> None:
        self.ctx.cluster.run(spec.node, shlex.join(spec.docker_run), timeout=300)
        self._started.add(spec.name)

    def _http_ok(self, url: str | None) -> bool:
        if url is None:
            return True
        res = self.ctx.cluster.run(
            self.ctx.node0,
            f"curl -sf --max-time 5 -o /dev/null {shlex.quote(url)}",
            check=False,
        )
        return res.ok

    def _models_ready(self, plan: LaunchPlan) -> bool:
        res = self.ctx.cluster.run(
            self.ctx.node0,
            f"curl -sf --max-time 5 {shlex.quote(plan.frontend_url + '/v1/models')}",
            check=False,
        )
        if not res.ok:
            return False
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return False
        return any(m.get("id") == plan.served_model_name for m in data.get("data", []))

    def _check_exits(self, plan: LaunchPlan) -> None:
        for spec in plan.containers:
            if spec.name not in self._started:
                continue  # not created yet (e.g. workers while etcd starts)
            state = self.ctx.docker.state(spec.node, spec.name)
            if state is None:
                raise DeployFailed(
                    "container_exit", f"{spec.name} on {spec.node} disappeared"
                )
            if not state.running:
                logs = self.ctx.docker.logs(spec.node, spec.name, tail=400)
                failure: FailureClass = (
                    "oom"
                    if state.oom_killed
                    else self.ctx.engine.classify_failure(logs)
                )
                if failure == "unknown":
                    failure = "container_exit"
                raise DeployFailed(
                    failure, f"{spec.name} exited {state.exit_code}:\n{logs[-3000:]}"
                )

    def _poll(self, ready, plan: LaunchPlan, timeout_s: float, what: str) -> None:
        """Poll ``ready`` until true; any container exit fails immediately.

        Exits are checked before readiness: a crashed rank of a multi-node
        instance has no health endpoint, so health alone cannot see it.
        """
        deadline = self.ctx.clock.now() + timeout_s
        while True:
            self._check_exits(plan)
            if ready():
                return
            if self.ctx.clock.now() > deadline:
                raise DeployFailed(
                    "timeout", f"timed out after {timeout_s:.0f}s waiting for {what}"
                )
            self.ctx.clock.sleep(POLL_S)

    def _wait_ready(self, plan: LaunchPlan, started: float) -> None:
        serving = [
            c
            for c in plan.containers
            if c.role not in ("control", "frontend") and c.health_url
        ]
        remaining = READY_TIMEOUT_S - (self.ctx.clock.now() - started)

        def all_healthy() -> bool:
            return all(
                self._http_ok(c.health_url) for c in serving
            ) and self._models_ready(plan)

        self._poll(
            all_healthy,
            plan,
            remaining,
            f"{len(serving)} workers healthy and the model listed",
        )

    def _smoke(self, plan: LaunchPlan, require_text: bool) -> None:
        body = json.dumps(
            {
                "model": plan.served_model_name,
                "messages": [{"role": "user", "content": SMOKE_PROMPT}],
                "max_tokens": 64,
                "stream": False,
            }
        )
        url = shlex.quote(plan.frontend_url + "/v1/chat/completions")
        res = self.ctx.cluster.run(
            self.ctx.node0,
            f"curl -sS --max-time 600 -H 'Content-Type: application/json' -d @- {url}",
            input_text=body,
            check=False,
        )
        if not res.ok:
            raise DeployFailed(
                "smoke_failed", f"completion request failed: {res.stderr[-1000:]}"
            )
        try:
            message = json.loads(res.stdout)["choices"][0]["message"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise DeployFailed(
                "smoke_failed", f"unparseable completion: {res.stdout[:1000]}"
            ) from exc
        text = (message.get("content") or "") + (message.get("reasoning_content") or "")
        if require_text and not text.strip():
            # PD can answer HTTP 200 with null content after a failed KV transfer.
            raise DeployFailed("smoke_failed", f"empty completion: {res.stdout[:1000]}")

    def _engagement(self, plan: LaunchPlan) -> EngagementResult:
        checks: dict[str, bool] = {}
        evidence: list[str] = []
        for spec in plan.containers:
            if spec.role in ("control", "frontend"):
                continue
            if spec.rank == 0:
                rendered = plan.rendered_configs.get(f"{spec.role}:{spec.instance}", {})
                log = self.ctx.cluster.run(
                    spec.node,
                    f"docker logs {shlex.quote(spec.name)} 2>&1 | grep -m 3 'ServerArgs('",
                    check=False,
                ).stdout
                server_args = self.ctx.engine.parse_server_args(log)
                if not server_args:
                    evidence.append(
                        f"{spec.name}: ServerArgs line not found (config engagement unproven)"
                    )
                checks_for = self.ctx.engine.engagement_checks(rendered, server_args)
                for key, ok in checks_for.items():
                    checks[f"{spec.name}:{key}"] = ok
                evidence.extend(self._transport_evidence(plan, spec))
            live_env = self._live_env(spec)
            for key in ENV_KEYS_TO_VERIFY:
                if key in spec.env:
                    checks[f"{spec.name}:env:{key}"] = (
                        live_env.get(key) == spec.env[key]
                    )
        ok = all(checks.values()) if checks else True
        return EngagementResult(ok=ok, checks=checks, evidence=tuple(evidence))

    def _afd_engagement(
        self,
        plan: LaunchPlan,
        lifecycle: DeployLifecycle | None,
        engagement: EngagementResult,
    ) -> EngagementResult:
        """A-Only: each worker built the null experts (its log marker), and the
        A GPUs hold about what they would without the target's routed experts
        (those stayed on disk)."""
        checks = dict(engagement.checks)
        evidence = list(engagement.evidence)
        for spec in plan.containers:
            if spec.role in ("control", "frontend") or spec.rank != 0:
                continue
            out = self.ctx.cluster.run(
                spec.node,
                f"docker logs {shlex.quote(spec.name)} 2>&1 "
                f"| grep -m 1 -F {shlex.quote(AFD_MARKER)}",
                check=False,
            ).stdout
            line = next((x.strip() for x in out.splitlines() if AFD_MARKER in x), "")
            checks[f"{spec.name}:afd_marker"] = bool(line)
            if line:
                evidence.append(f"{spec.name}: {line[:300]}")
        role = (
            lifecycle.roles.get(stage_spec(plan.stage).role or "decode")
            if lifecycle is not None
            else None
        )
        weights = role.weight_mem_gb_max if role is not None else None
        n_a = plan_instance_gpus(plan)
        hw = self.ctx.require_hardware()
        _, checkpoint = select_checkpoint(hw.model, hw.gpu, self.ctx.config.checkpoint)
        f_plan = ensure_plan(self.ctx)
        expected = (
            afd.expected_weights(f_plan, checkpoint.approx_size_gb, n_a)
            if f_plan is not None
            else None
        )
        if weights is None:
            evidence.append("A-side weights per GPU not in the worker logs (unproven)")
        elif expected is None:
            evidence.append(
                f"A-side weights {weights:.1f} GB per GPU; no expected size (the "
                "checkpoint is not larger than its routed experts)"
            )
        else:
            checks["afd_routed_not_loaded"] = afd.weights_engaged(weights, expected)
            evidence.append(
                f"A-side weights {weights:.1f} GB per GPU vs at most "
                f"{expected[0]:.1f} GB without the target's routed experts and "
                f"{expected[1]:.1f} GB with them over {n_a} GPUs"
            )
        return engagement.model_copy(
            update={
                "ok": all(checks.values()) if checks else True,
                "checks": checks,
                "evidence": tuple(evidence),
            }
        )

    def _transport_evidence(self, plan: LaunchPlan, spec: ContainerSpec) -> list[str]:
        """Log lines showing which HCAs the data plane opened (evidence only;
        per-HCA counters during the run are the authoritative proof)."""
        if plan.candidate.nic_count is None:
            return []
        names = ["nccl"] + (
            [plan.transport] if plan.transport not in ("none", "nccl") else []
        )
        patterns = [
            re.compile(p)
            for n in names
            for p in get_transport(n).engagement_patterns(spec.hcas)
        ]
        log = self.ctx.docker.logs(spec.node, spec.name, tail=5000)
        lines = [
            line for line in log.splitlines() if any(p.search(line) for p in patterns)
        ]
        return [f"{spec.name}: {line.strip()[:300]}" for line in lines[:8]]

    def _live_env(self, spec: ContainerSpec) -> dict[str, str]:
        cmd = (
            f"pid=$(pgrep -f {shlex.quote(WORKER_PROCESS)} | head -1); "
            '[ -n "$pid" ] && tr "\\0" "\\n" < /proc/$pid/environ'
        )
        out = self.ctx.docker.exec(spec.node, spec.name, cmd, check=False)
        env: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                env[key] = value
        return env

    def _restart_counts(self, plan: LaunchPlan) -> dict[str, int]:
        counts: dict[str, int] = {}
        for spec in plan.containers:
            state = self.ctx.docker.state(spec.node, spec.name)
            if state is not None:
                counts[spec.name] = state.restart_count
        return counts

    def _save_logs(self, plan: LaunchPlan, log_dir: Path) -> dict[str, str]:
        texts: dict[str, str] = {}
        for spec in plan.containers:
            text = self.ctx.redactor(
                self.ctx.docker.logs(spec.node, spec.name, tail=5000)
            )
            atomic_write_text(log_dir / "logs" / f"{spec.name}.log", text)
            texts[spec.name] = text
        return texts

    def _lifecycle(
        self, plan: LaunchPlan, texts: dict[str, str], started: float, log_dir: Path
    ) -> DeployLifecycle | None:
        """Cold-start phases and KV capacity from the logs (compact copy)."""
        rows = [
            (s.name, s.role, s.node, texts.get(s.name, "")) for s in plan.containers
        ]
        try:
            full = deploy_lifecycle(rows, started)
        except (ValueError, TypeError, KeyError) as exc:
            self.ctx.note(f"deploy lifecycle unavailable for {plan.cand_fp}: {exc}")
            return None
        atomic_write_text(log_dir / "lifecycle.json", full.model_dump_json(indent=2))
        return full.model_copy(update={"containers": ()})


_KILL_ENGINE_ZOMBIES = (
    "for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
    '  if ps -o args= -p "$pid" 2>/dev/null | grep -qE \'sglang|dynamo|torchrun|torch.distributed.run\'; then kill -9 "$pid"; fi; '
    "done"
)
