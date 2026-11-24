# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ports: the interfaces the application layer depends on.

Adapters implement these; tests and ``dry-run`` substitute recording or fake
implementations. Keeping the protocols here lets ``app/`` stay independent of
SSH, Docker, the lium CLI, SGLang, or AIPerf specifics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypeVar

from benchmarks.cluster_bench.domain.schema import (
    AfdPlan,
    Candidate,
    DeploymentProfile,
    FailureClass,
    GpuProfile,
    Inventory,
    LaunchPlan,
    ModelProfile,
    RunConfig,
    Transport,
)

T = TypeVar("T")


class CommandError(RuntimeError):
    """A remote or local command exited non-zero while ``check`` was requested."""

    def __init__(self, node: str, command: str, rc: int, stderr: str):
        super().__init__(f"[{node}] exit {rc}: {command}\n{stderr[-2000:]}")
        self.node = node
        self.command = command
        self.rc = rc
        self.stderr = stderr


@dataclass(frozen=True)
class CommandResult:
    node: str
    command: str
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def check(self) -> CommandResult:
        if self.rc != 0:
            raise CommandError(self.node, self.command, self.rc, self.stderr)
        return self


@dataclass(frozen=True)
class NodeAddress:
    name: str  # logical name, node0 first
    address: str  # control-plane IPv4 (Lium: the wg0 address)
    ssh_target: str | None  # None means "this host"
    ssh_port: int = 22
    # Private key node0 uses for this node (None: ssh's defaults).
    identity: str | None = None
    # Control-plane NIC of this node (None: the run's --control-interface).
    control_interface: str | None = None
    # known_hosts file node0 checks this node's host key against (None: ssh's
    # default, ~/.ssh/known_hosts).
    known_hosts: str | None = None


class Executor(Protocol):
    """Runs shell commands and moves files on cluster nodes."""

    @property
    def nodes(self) -> tuple[str, ...]:
        ...

    def run(
        self,
        node: str,
        command: str,
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        ...

    def put_text(self, node: str, path: str, content: str, mode: int = 0o644) -> None:
        ...

    def put_file(self, node: str, local: Path, remote: str) -> None:
        ...

    def get_file(self, node: str, remote: str, local: Path) -> None:
        ...


class Cluster(Executor, Protocol):
    """An executor spanning every node, with concurrent fan-out."""

    def fanout(self, nodes: Sequence[str], fn: Callable[[str], T]) -> dict[str, T]:
        ...

    def push_tree(
        self, local_dir: Path, nodes: Sequence[str], remote_parent: str
    ) -> None:
        """Replicate a small directory tree (code) from this host to ``nodes``."""
        ...


class Clock(Protocol):
    def now(self) -> float:
        ...

    def sleep(self, seconds: float) -> None:
        """Pause between polls of an observable condition (never as readiness)."""
        ...


class Provider(Protocol):
    """Where the cluster comes from (SSH machines, a hosts file or a lium rental)."""

    @property
    def created_by_harness(self) -> bool:
        ...

    def nodes(self) -> list[NodeAddress]:
        ...

    def destroy(self) -> None:
        ...


class TransportPlugin(Protocol):
    """Restricts one data-plane transport to a subset of HCAs."""

    name: Transport

    def restrict(self, hcas: Sequence[str]) -> tuple[dict[str, str], dict[str, Any]]:
        """Return (env, sglang_config_overrides) limiting traffic to ``hcas``."""
        ...

    def engagement_patterns(self, hcas: Sequence[str]) -> tuple[str, ...]:
        """Regexes whose presence in worker logs proves the restriction applied."""
        ...


class Engine(Protocol):
    """Turns a candidate into concrete containers, and reads engine logs."""

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
        ...

    def classify_failure(self, log_text: str) -> FailureClass:
        ...

    def engagement_checks(
        self, rendered: dict[str, Any], server_args: dict[str, str]
    ) -> dict[str, bool]:
        """Per rendered key: did the live process take the rendered value?"""
        ...

    def parse_server_args(self, log_text: str) -> dict[str, str]:
        ...
