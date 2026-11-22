# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command execution on cluster nodes.

* ``LocalExecutor``   - the host the runner lives on (node0).
* ``SshExecutor``     - peers over the WireGuard control network, multiplexed
                        through one ControlMaster connection per node.
* ``ClusterExecutor`` - routes each logical node to one of the above.
* ``RecordingExecutor`` - records commands and answers from a responder; used
                        by ``dry-run`` and unit tests (no network, no GPUs).

Secrets never appear in argv: callers pass them through ``input_text`` (stdin)
and every string that is logged or persisted goes through ``Redactor``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TypeVar

from benchmarks.cluster_bench.ports import (
    CommandError,
    CommandResult,
    Executor,
    NodeAddress,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

SECRET_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "LIUM_API_KEY",
    "LIUM_API_API_KEY",
    "LIUM_SESSION_TOKEN",
    "NGC_API_KEY",
)


class Redactor:
    """Masks known secret values in any text before it is logged or persisted."""

    def __init__(self, secrets: Iterable[str] = ()):
        self._secrets = tuple(
            sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Redactor:
        source = os.environ if env is None else env
        return cls(source.get(name, "") for name in SECRET_ENV_VARS)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def _run_subprocess(
    node: str,
    argv: Sequence[str],
    display: str,
    input_text: str | None,
    timeout: float | None,
    check: bool,
) -> CommandResult:
    try:
        proc = subprocess.run(
            list(argv),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = CommandResult(node, display, 124, stdout, stderr + "\n[timeout]")
        if check:
            raise CommandError(node, display, 124, result.stderr) from exc
        return result
    result = CommandResult(node, display, proc.returncode, proc.stdout, proc.stderr)
    return result.check() if check else result


class LocalExecutor:
    """Executes on the current host. ``node`` is accepted for interface parity."""

    def __init__(self, name: str, redactor: Redactor | None = None):
        self.name = name
        self._redact = redactor or Redactor()

    @property
    def nodes(self) -> tuple[str, ...]:
        return (self.name,)

    def run(
        self,
        node: str,
        command: str,
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        display = self._redact(command)
        logger.debug("[%s] $ %s", node, display)
        return _run_subprocess(
            node, ["bash", "-c", command], display, input_text, timeout, check
        )

    def put_text(self, node: str, path: str, content: str, mode: int = 0o644) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        target.chmod(mode)

    def put_file(self, node: str, local: Path, remote: str) -> None:
        self.run(
            node,
            f"mkdir -p {shlex.quote(str(Path(remote).parent))} && cp -a {shlex.quote(str(local))} {shlex.quote(remote)}",
        )

    def get_file(self, node: str, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.run(node, f"cp -a {shlex.quote(remote)} {shlex.quote(str(local))}")


# ssh binds the multiplexing socket at "<ControlPath>.<16 random chars>"; with
# %C (40 hex chars) the directory must stay well under the 107-byte sun_path
# limit, which a path under the experiment directory does not.
CONTROL_SOCKET_ROOT = Path("/tmp")


def control_socket_dir(key: Path, root: Path = CONTROL_SOCKET_ROOT) -> Path:
    """Short private directory for ssh ControlMaster sockets, one per ``key``."""
    digest = hashlib.sha256(str(key).encode()).hexdigest()[:8]
    path = root / f"cbssh-{digest}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


class SshExecutor:
    """Executes on one remote node over SSH with connection multiplexing."""

    def __init__(
        self, address: NodeAddress, control_dir: Path, redactor: Redactor | None = None
    ):
        if address.ssh_target is None:
            raise ValueError(f"{address.name} has no ssh target")
        self.address = address
        self._redact = redactor or Redactor()
        socket_dir = control_socket_dir(control_dir)
        self._ssh_opts = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={socket_dir}/%C",
            "-o", "ControlPersist=900",
            "-p", str(address.ssh_port),
        ]  # fmt: skip
        if address.identity:
            self._ssh_opts += ["-i", address.identity, "-o", "IdentitiesOnly=yes"]
        if address.known_hosts:
            self._ssh_opts += ["-o", f"UserKnownHostsFile={address.known_hosts}"]

    @property
    def nodes(self) -> tuple[str, ...]:
        return (self.address.name,)

    def ssh_argv(self) -> list[str]:
        return ["ssh", *self._ssh_opts, self.address.ssh_target or ""]

    def run(
        self,
        node: str,
        command: str,
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        display = self._redact(command)
        logger.debug("[%s] $ %s", node, display)
        argv = [*self.ssh_argv(), "bash", "-c", shlex.quote(command)]
        return _run_subprocess(node, argv, display, input_text, timeout, check)

    def put_text(self, node: str, path: str, content: str, mode: int = 0o644) -> None:
        parent = shlex.quote(str(Path(path).parent))
        quoted = shlex.quote(path)
        self.run(
            node,
            f"mkdir -p {parent} && cat > {quoted}.tmp && chmod {mode:o} {quoted}.tmp && mv {quoted}.tmp {quoted}",
            input_text=content,
        )

    def _scp_opts(self) -> list[str]:
        opts = list(self._ssh_opts)
        port_index = opts.index("-p")
        opts[port_index] = "-P"  # scp spells the port flag differently
        return opts

    def put_file(self, node: str, local: Path, remote: str) -> None:
        self.run(node, f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
        argv = [
            "scp",
            "-q",
            *self._scp_opts(),
            str(local),
            f"{self.address.ssh_target}:{remote}",
        ]
        _run_subprocess(node, argv, f"scp {local} -> {remote}", None, None, True)

    def get_file(self, node: str, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "scp",
            "-q",
            "-r",
            *self._scp_opts(),
            f"{self.address.ssh_target}:{remote}",
            str(local),
        ]
        _run_subprocess(node, argv, f"scp {remote} -> {local}", None, None, True)


class ClusterExecutor:
    """Routes logical node names to per-node executors and fans out work."""

    def __init__(self, executors: Mapping[str, Executor], max_workers: int = 16):
        if not executors:
            raise ValueError("ClusterExecutor needs at least one node")
        self._executors = dict(executors)
        self._max_workers = max_workers

    @classmethod
    def from_addresses(
        cls, addresses: Sequence[NodeAddress], control_dir: Path, redactor: Redactor
    ) -> ClusterExecutor:
        executors: dict[str, Executor] = {}
        for addr in addresses:
            if addr.ssh_target is None:
                executors[addr.name] = LocalExecutor(addr.name, redactor)
            else:
                executors[addr.name] = SshExecutor(addr, control_dir, redactor)
        return cls(executors)

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._executors)

    def _for(self, node: str) -> Executor:
        try:
            return self._executors[node]
        except KeyError as exc:
            raise KeyError(
                f"unknown node {node!r}; known: {sorted(self._executors)}"
            ) from exc

    def run(
        self,
        node: str,
        command: str,
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        return self._for(node).run(
            node, command, input_text=input_text, timeout=timeout, check=check
        )

    def put_text(self, node: str, path: str, content: str, mode: int = 0o644) -> None:
        self._for(node).put_text(node, path, content, mode)

    def put_file(self, node: str, local: Path, remote: str) -> None:
        self._for(node).put_file(node, local, remote)

    def get_file(self, node: str, remote: str, local: Path) -> None:
        self._for(node).get_file(node, remote, local)

    def push_tree(
        self, local_dir: Path, nodes: Sequence[str], remote_parent: str
    ) -> None:
        """Copy ``local_dir`` to ``remote_parent/<name>`` on ``nodes`` (tar over ssh).

        Peers are reached over the control network; the tree is small (code).
        """

        def push(node: str) -> None:
            executor = self._for(node)
            if not isinstance(executor, SshExecutor):
                return  # the local node already has it
            remote = shlex.quote(remote_parent)
            remote_cmd = f"mkdir -p {remote} && tar -C {remote} -xzf -"
            local_cmd = (
                f"tar -C {shlex.quote(str(local_dir.parent))} -czf - {shlex.quote(local_dir.name)} | "
                f"{shlex.join(executor.ssh_argv())} {shlex.quote(remote_cmd)}"
            )
            _run_subprocess(
                node,
                ["bash", "-c", local_cmd],
                f"push {local_dir} -> {node}:{remote_parent}",
                None,
                600,
                True,
            )

        self.fanout(list(nodes), push)

    def fanout(self, nodes: Sequence[str], fn: Callable[[str], T]) -> dict[str, T]:
        """Run ``fn(node)`` concurrently; re-raise the first failure after all finish."""
        results: dict[str, T] = {}
        errors: list[BaseException] = []
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, max(1, len(nodes)))
        ) as pool:
            futures = {node: pool.submit(fn, node) for node in nodes}
            for node, future in futures.items():
                try:
                    results[node] = future.result()
                except Exception as exc:
                    logger.error("[%s] parallel task failed: %s", node, exc)
                    errors.append(exc)
        if errors:
            raise errors[0]
        return results


@dataclass
class RecordedCommand:
    node: str
    command: str
    had_input: bool


Responder = Callable[[str, str], "CommandResult | None"]


@dataclass
class RecordingExecutor:
    """In-memory executor: records every command, serves files from a dict.

    ``responder(node, command)`` may return a ``CommandResult`` to script
    outputs; returning ``None`` yields an empty successful result.
    """

    node_names: tuple[str, ...]
    responder: Responder | None = None
    redactor: Redactor = field(default_factory=Redactor)
    commands: list[RecordedCommand] = field(default_factory=list)
    files: dict[tuple[str, str], str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.node_names

    def run(
        self,
        node: str,
        command: str,
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        if node not in self.node_names:
            raise KeyError(f"unknown node {node!r}")
        display = self.redactor(command)
        with self._lock:
            self.commands.append(RecordedCommand(node, display, input_text is not None))
        result = self.responder(node, command) if self.responder else None
        if result is None:
            result = CommandResult(node, display, 0, "", "")
        return result.check() if check else result

    def put_text(self, node: str, path: str, content: str, mode: int = 0o644) -> None:
        with self._lock:
            self.files[(node, path)] = content
            self.commands.append(RecordedCommand(node, f"<write {path}>", True))

    def put_file(self, node: str, local: Path, remote: str) -> None:
        self.put_text(node, remote, local.read_text())

    def get_file(self, node: str, remote: str, local: Path) -> None:
        key = (node, remote)
        if key not in self.files:
            raise FileNotFoundError(f"[{node}] {remote}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(self.files[key])

    def fanout(self, nodes: Sequence[str], fn: Callable[[str], T]) -> dict[str, T]:
        return {node: fn(node) for node in nodes}

    def push_tree(
        self, local_dir: Path, nodes: Sequence[str], remote_parent: str
    ) -> None:
        with self._lock:
            for node in nodes:
                self.commands.append(
                    RecordedCommand(
                        node, f"<push {local_dir} -> {remote_parent}>", False
                    )
                )

    def commands_for(self, node: str) -> list[str]:
        return [c.command for c in self.commands if c.node == node]
