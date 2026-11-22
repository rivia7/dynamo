# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docker on cluster nodes (Lium pods run Docker-in-Docker).

``docker_run_argv`` is pure and shared by every adapter that starts a
container (workers, microbenchmarks, the model gather). ``DockerHost``
operates on containers by label through an ``Executor`` so cleanup never
depends on in-memory state that a crash could lose.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Mapping, Sequence

from benchmarks.cluster_bench.ports import Executor

LABEL_EXP = "cb.exp"
LABEL_CAND = "cb.cand"
LABEL_ROLE = "cb.role"

# The sglang-runtime image ends with ``USER dynamo``; the recipes run workers as
# UID 0 (FlashInfer JITs into root-owned site-packages; caches live in /root).
CONTAINER_USER = "0:0"


def sanitize_label(text: str) -> str:
    """Docker-label / Dynamo-namespace safe form of an id."""
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def experiment_labels(exp_id: str) -> dict[str, str]:
    """Label selector matching every container of one experiment."""
    return {LABEL_EXP: sanitize_label(exp_id)}


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    read_only: bool = False

    def arg(self) -> str:
        return f"{self.source}:{self.target}{':ro' if self.read_only else ''}"


def docker_run_argv(
    *,
    name: str,
    image: str,
    command: Sequence[str],
    gpus: Sequence[int] | None,
    env: Mapping[str, str] | None = None,
    env_file: str | None = None,
    mounts: Sequence[Mount] = (),
    labels: Mapping[str, str] | None = None,
    rdma: bool = True,
    detach: bool = True,
    remove: bool = False,
    interactive: bool = False,
    shm_size: str = "64g",
    workdir: str | None = None,
    user: str | None = CONTAINER_USER,
    cap_add: Sequence[str] = (),
) -> list[str]:
    """Build a `docker run` argv for a host-networked GPU/RDMA workload.

    Secrets must not be passed via ``env``; pass a valueless ``-e NAME`` by
    listing NAME in ``env`` with an empty value is NOT supported here on
    purpose. Use ``interactive=True`` and feed the secret through stdin.
    """
    argv = ["docker", "run"]
    if detach:
        argv.append("-d")
    if remove:
        argv.append("--rm")
    if interactive:
        argv.append("-i")
    argv += [
        "--name",
        name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        shm_size,
    ]
    if user:
        argv += ["--user", user]
    argv += [
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "nofile=1048576:1048576",
        "--ulimit",
        "stack=67108864",
    ]
    argv += [
        "--cap-add",
        "IPC_LOCK",
        "--cap-add",
        "SYS_PTRACE",
        "--cap-add",
        "SYS_RESOURCE",
    ]
    for cap in cap_add:  # e.g. NET_ADMIN for fabric diagnostics
        argv += ["--cap-add", cap]
    if gpus is None:
        pass
    elif len(gpus) == 0:
        raise ValueError("gpus=() is ambiguous; pass None for a CPU-only container")
    else:
        # The inner double quotes are required by docker for comma lists.
        argv += ["--gpus", '"device=' + ",".join(str(g) for g in gpus) + '"']
    if rdma:
        argv += ["--device", "/dev/infiniband"]
    for key, value in sorted((labels or {}).items()):
        argv += ["--label", f"{key}={value}"]
    if env_file:
        argv += ["--env-file", env_file]
    for key, value in sorted((env or {}).items()):
        argv += ["-e", f"{key}={value}"]
    for mount in mounts:
        argv += ["-v", mount.arg()]
    if workdir:
        argv += ["-w", workdir]
    argv.append(image)
    argv += list(command)
    return argv


def render_env_file(env: Mapping[str, str]) -> str:
    """docker --env-file format: one KEY=VALUE per line, no quoting, no newlines."""
    lines = []
    for key, value in sorted(env.items()):
        if "\n" in value:
            raise ValueError(f"env value for {key} contains a newline")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ContainerState:
    name: str
    running: bool
    exit_code: int
    oom_killed: bool
    restart_count: int
    status: str


class DockerHost:
    """Label-scoped container operations on cluster nodes."""

    def __init__(self, executor: Executor):
        self._exec = executor

    def pull(self, node: str, image: str, timeout: float = 3600) -> None:
        self._exec.run(
            node, f"docker pull -q {shlex.quote(image)} >/dev/null", timeout=timeout
        )

    def image_present(self, node: str, image: str) -> bool:
        res = self._exec.run(
            node,
            f"docker image inspect {shlex.quote(image)} >/dev/null 2>&1",
            check=False,
        )
        return res.ok

    def run(
        self,
        node: str,
        argv: Sequence[str],
        input_text: str | None = None,
        timeout: float = 600,
    ) -> str:
        res = self._exec.run(
            node, shlex.join(argv), input_text=input_text, timeout=timeout
        )
        return res.stdout.strip()

    def names(
        self, node: str, labels: Mapping[str, str], include_stopped: bool = True
    ) -> list[str]:
        filters = " ".join(
            f"--filter label={shlex.quote(f'{k}={v}')}"
            for k, v in sorted(labels.items())
        )
        flag = "-a " if include_stopped else ""
        res = self._exec.run(
            node, f"docker ps {flag}{filters} --format '{{{{.Names}}}}'"
        )
        return [line for line in res.stdout.splitlines() if line.strip()]

    def remove(self, node: str, labels: Mapping[str, str]) -> list[str]:
        names = self.names(node, labels)
        if names:
            self._exec.run(
                node,
                "docker rm -f "
                + " ".join(shlex.quote(n) for n in names)
                + " >/dev/null",
                timeout=300,
            )
        return names

    def state(self, node: str, name: str) -> ContainerState | None:
        fmt = "{{json .State}}|{{.RestartCount}}"
        res = self._exec.run(
            node, f"docker inspect --format '{fmt}' {shlex.quote(name)}", check=False
        )
        if not res.ok or "|" not in res.stdout:
            return None
        state_json, restarts = res.stdout.strip().rsplit("|", 1)
        state = json.loads(state_json)
        return ContainerState(
            name=name,
            running=bool(state.get("Running")),
            exit_code=int(state.get("ExitCode", 0)),
            oom_killed=bool(state.get("OOMKilled")),
            restart_count=int(restarts or 0),
            status=str(state.get("Status", "")),
        )

    def logs(self, node: str, name: str, tail: int | None = 400) -> str:
        tail_arg = f"--tail {int(tail)} " if tail is not None else ""
        res = self._exec.run(
            node,
            f"docker logs {tail_arg}{shlex.quote(name)} 2>&1",
            check=False,
            timeout=120,
        )
        return res.stdout

    def exec(
        self,
        node: str,
        name: str,
        command: str,
        timeout: float = 60,
        check: bool = True,
    ) -> str:
        res = self._exec.run(
            node,
            f"docker exec {shlex.quote(name)} bash -c {shlex.quote(command)}",
            timeout=timeout,
            check=check,
        )
        return res.stdout
