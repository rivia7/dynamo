# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Laptop-side controller: drive one experiment on SSH machines or a Lium rental.

python -m benchmarks.cluster_bench controller ACTION --exp-id ID [options] [-- runner args]

  up       --machines FILE / --machine 'ssh ...': record the machines you
           brought (node0 first), check them, let node0 ssh to the others with a
           per-experiment key; or rent N lium nodes (lium clusters up), or
           --adopt NAME an existing lium cluster
  sync     stream the harness subset of the repo to node0, verify it, bootstrap
  launch   start the runner detached on node0 (HF_TOKEN travels over ssh stdin)
  watch    poll status.json, pull every 30 min, relaunch a dead runner (max 2)
  pull     copy <workdir>/experiments/<exp>/ back (rsync, scp -r fallback)
  down     lium: remove the cluster (only if the harness created it and the
           results are in); SSH machines: remove this experiment's containers
           and trust key, never the machines
  status   print the local state and the remote heartbeat
  run-all  up -> sync -> launch -> watch -> pull -> KB merge -> down

The machines file format is in adapters/machines.py.

State: ~/.cluster_bench/<exp_id>/controller.json (never holds secrets).
Arguments after "--" are passed to the runner (cluster_bench.sh run ...).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shlex
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_serializer

from benchmarks.cluster_bench.adapters import machines as mach
from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.docker import sanitize_label
from benchmarks.cluster_bench.adapters.providers import (
    GUARDED_SECRET_VARS,
    ArgvRunner,
    LiumBillingUncertainError,
    LiumCli,
    LiumCliError,
    LiumCluster,
    LiumFabric,
    LiumNotFoundError,
    LiumProvider,
    LiumRetryableError,
    ProviderError,
    assert_no_secrets,
    format_ttl,
    resolve_lium_ssh_key,
    run_argv,
)
from benchmarks.cluster_bench.domain.profiles import ProfileError, detect_gpu_profile
from benchmarks.cluster_bench.domain.schema import ProviderInfo, ProviderMember
from benchmarks.cluster_bench.fsutil import atomic_write_json
from benchmarks.cluster_bench.ports import Clock, CommandResult

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_VERSION = 1
STATE_FILE = "controller.json"
DEFAULT_WORKDIR = "/root/cluster_bench"
DEFAULT_NODES = 4
# Cluster sizes the tuned profiles are written and tested for.
TUNED_NODE_COUNTS = (2, 4, 8)
DEFAULT_MODEL = "glm-5.3"
GPUS_PER_NODE = 8
DEFAULT_TTL_HOURS = 12.0
SYNC_MANIFEST = ".cluster_bench_sync.sha256"
RUNNER_SCRIPT = "benchmarks/cluster_bench/cluster_bench.sh"
BOOTSTRAP_SCRIPT = "benchmarks/cluster_bench/scripts/bootstrap_node.sh"
# git pathspecs of the synced subset; ``is_sync_path`` narrows them.
SYNC_PATHSPECS = (
    "benchmarks/__init__.py",
    "benchmarks/cluster_bench",
    "examples/common",
    "recipes/glm-5.3",
)
# Flags the controller sets itself; passthrough runner args may not repeat them.
MANAGED_RUNNER_FLAGS = (
    "--exp-id",
    "--workdir",
    "--nodes",
    "--hosts",
    "--expect-nodes",
    "--platform",
)
# Resuming is implicit: the runner resumes from its event log when it is
# started again with the same --exp-id (cli.py ``run`` has no --resume flag).
RESUME_ARGS: tuple[str, ...] = ()
# Names under the state dir that are not experiments (cluster_bench.sh's venv,
# cli.py's dry-run workdir).
RESERVED_EXP_IDS = frozenset({"venv", "dryrun", "kb"})
WATCH_LOST = 3
# Runner states after which the process exits (app/pipeline.py).
FINISHED_STATES = frozenset({"done", "failed", "interrupted"})

_EXP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_WORKDIR_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
_GPU_ALIASES = {
    "PRO6000": "RTXPRO6000",
    "RTX6000PRO": "RTXPRO6000",
    "6000PRO": "RTXPRO6000",
}


class ControllerError(RuntimeError):
    """A controller action cannot proceed; the message says what to do."""


def _exp_id_problem(exp_id: str) -> str | None:
    if not _EXP_ID_RE.match(exp_id) or exp_id in RESERVED_EXP_IDS:
        return (
            f"invalid --exp-id {exp_id!r}: use letters, digits, '.', '_' or '-' "
            f"(not {sorted(RESERVED_EXP_IDS)})"
        )
    return None


def _workdir_problem(workdir: str) -> str | None:
    if not _WORKDIR_RE.match(workdir) or "/../" in f"{workdir}/":
        return f"invalid --workdir {workdir!r}: an absolute path of [A-Za-z0-9._/-]"
    return None


def check_exp_id(exp_id: str) -> str:
    problem = _exp_id_problem(exp_id)
    if problem:
        raise ControllerError(problem)
    return exp_id


def check_workdir(workdir: str) -> str:
    problem = _workdir_problem(workdir)
    if problem:
        raise ControllerError(problem)
    return workdir


# --------------------------------------------------------------------------
# State file
# --------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Compact(_Model):
    """Serialized without the SSH-machine fields a lium record leaves unset."""

    _ALWAYS: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _compact(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        return {
            k: v
            for k, v in data.items()
            if k in self._ALWAYS or v not in (None, (), [])
        }


class SshEndpoint(_Compact):
    _ALWAYS: ClassVar[frozenset[str]] = frozenset({"target", "port"})
    target: str  # "user@host" (public address, for laptop access)
    port: int = 22
    # SSH machines: the user's own key, jump host and -o options for this host.
    identity: str | None = None
    jump: str | None = None
    options: tuple[str, ...] = ()  # "Key=Value", allow-listed (adapters/machines.py)


class MemberRecord(_Compact):
    _ALWAYS: ClassVar[frozenset[str]] = frozenset(
        {"name", "rank", "huid", "pod_id", "ssh", "control_ip", "gpu_type", "gpu_count"}
    )
    name: str
    rank: int
    huid: str = ""
    pod_id: str = ""
    ssh: SshEndpoint | None = None
    control_ip: str | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    # SSH machines: label, hostname, how node0 reaches it, control NIC.
    label: str | None = None
    hostname: str | None = None
    peer_ssh: SshEndpoint | None = None
    control_interface: str | None = None


class ControllerState(_Model):
    """``~/.cluster_bench/<exp_id>/controller.json``."""

    version: int = STATE_VERSION
    exp_id: str
    # "lium": a rental (cluster_id once rented); "ssh": machines the user
    # brought, never created or removed by the harness.
    provider: Literal["lium", "ssh"] = "lium"
    cluster_name: str
    cluster_id: str | None = None
    created_by_harness: bool
    fabric: str | None = None
    gpu: str | None = None
    node_count: int = DEFAULT_NODES
    ttl_hours: float | None = None
    # Offer metadata from `lium clusters list` (copied into provider.json).
    fabric_type: str | None = None
    link_rate: str | None = None
    fabric_measured: bool | None = None
    price_per_node_hour: float | None = None
    cluster_price_per_hour: float | None = None
    template: str | None = None
    node0_ssh: SshEndpoint | None = None
    control_ips: tuple[str, ...] = ()
    members: tuple[MemberRecord, ...] = ()
    workdir: str = DEFAULT_WORKDIR
    created_at: float
    synced_at: float | None = None
    sync_manifest_sha256: str | None = None
    venv_python: str | None = None
    runner_args: tuple[str, ...] = ()
    runner_nodes: str | None = None
    runner_pid: int | None = None
    launched_at: float | None = None
    launched_remote_ts: float | None = None  # node0 clock, `date +%s` at launch
    launches: int = 0
    relaunches: int = 0
    results_dir: str | None = None
    last_pull_at: float | None = None
    pulled_ok: bool = False
    destroyed_at: float | None = None
    released_at: float | None = None  # ssh: containers and trust key removed

    @field_validator("exp_id")
    @classmethod
    def _exp_id(cls, value: str) -> str:
        problem = _exp_id_problem(value)
        if problem:
            raise ValueError(problem)
        return value

    @field_validator("workdir")
    @classmethod
    def _workdir(cls, value: str) -> str:
        problem = _workdir_problem(value)
        if problem:
            raise ValueError(problem)
        return value

    def evolve(self, **updates: Any) -> ControllerState:
        """A validated copy with ``updates`` applied."""
        return type(self).model_validate({**self.model_dump(), **updates})

    @property
    def cluster_ref(self) -> str:
        return self.cluster_id or self.cluster_name

    @property
    def exp_dir(self) -> str:
        return f"{self.workdir}/experiments/{self.exp_id}"

    @property
    def status_path(self) -> str:
        return f"{self.exp_dir}/status.json"


# --------------------------------------------------------------------------
# Code sync
# --------------------------------------------------------------------------


def is_sync_path(path: str) -> bool:
    """Whether a repo-relative path belongs to the subset synced to node0."""
    p = PurePosixPath(path)
    if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
        return False
    if path == "benchmarks/__init__.py":
        return True
    if p.parts[:2] == ("benchmarks", "cluster_bench") and len(p.parts) > 2:
        return True
    if p.parent == PurePosixPath("examples/common") and p.suffix == ".sh":
        return True
    return p.parts[:2] == ("recipes", "glm-5.3") and p.name == "deploy.yaml"


def list_sync_files(repo_root: Path, runner: ArgvRunner) -> list[str]:
    """Tracked and untracked-but-not-ignored files of the synced subset."""
    argv = [
        "git", "-C", str(repo_root), "ls-files", "-z",
        "--cached", "--others", "--exclude-standard", "--", *SYNC_PATHSPECS,
    ]  # fmt: skip
    res = runner(argv, timeout=120)
    if res.rc != 0:
        raise ControllerError(f"git ls-files failed (exit {res.rc}): {res.stderr}")
    paths = sorted(
        {
            p
            for p in res.stdout.split("\0")
            if p and is_sync_path(p) and (repo_root / p).is_file()
        }
    )
    if not any(p.startswith("benchmarks/cluster_bench/") for p in paths):
        raise ControllerError(
            f"no benchmarks/cluster_bench files found under {repo_root}; "
            f"git ls-files printed:\n{res.stdout[-2000:]}"
        )
    return paths


@dataclass(frozen=True)
class SyncBundle:
    data: bytes  # tar archive of the files plus SYNC_MANIFEST
    manifest: str  # sha256sum-format lines: "<hex>  <path>"
    digest: str  # sha256 of the manifest text
    count: int


def build_sync_bundle(repo_root: Path, paths: Sequence[str]) -> SyncBundle:
    """An in-memory tar of ``paths`` with a sha256 manifest, owned by root."""
    lines: list[str] = []
    buf = io.BytesIO()
    newest = 0
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in paths:
            full = repo_root / rel
            data = full.read_bytes()
            st = full.stat()
            newest = max(newest, int(st.st_mtime))
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
            tar.addfile(
                _tar_info(rel, len(data), st.st_mode, int(st.st_mtime)),
                io.BytesIO(data),
            )
        manifest = "\n".join(lines) + "\n"
        blob = manifest.encode()
        tar.addfile(
            _tar_info(SYNC_MANIFEST, len(blob), 0o644, newest), io.BytesIO(blob)
        )
    return SyncBundle(
        buf.getvalue(), manifest, hashlib.sha256(blob).hexdigest(), len(paths)
    )


def _tar_info(name: str, size: int, mode: int, mtime: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o755 if mode & 0o111 else 0o644
    info.mtime = mtime
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


# --------------------------------------------------------------------------
# Remote shell snippets (pure; executed with `bash -c` on node0)
# --------------------------------------------------------------------------


def extract_script(workdir: str) -> str:
    """Untar stdin into ``<workdir>/src.incoming``, verify, then swap it in."""
    return "\n".join(
        [
            "set -euo pipefail",
            f"W={shlex.quote(workdir)}",
            'mkdir -p "$W"',
            'rm -rf "$W/src.incoming" "$W/src.previous"',
            'mkdir -p "$W/src.incoming"',
            'tar -x -f - -C "$W/src.incoming"',
            f'(cd "$W/src.incoming" && sha256sum --check --quiet --strict {SYNC_MANIFEST})',
            f'count=$(grep -c . "$W/src.incoming/{SYNC_MANIFEST}")',
            'if [ -d "$W/src" ]; then mv "$W/src" "$W/src.previous"; fi',
            'mv "$W/src.incoming" "$W/src"',
            'rm -rf "$W/src.previous"',
            'echo "@@SYNC_OK $count"',
        ]
    )


def bootstrap_command(workdir: str) -> str:
    return shlex.join(
        ["bash", f"{workdir}/src/{BOOTSTRAP_SCRIPT}", "--workdir", workdir]
    )


def runner_argv(
    exp_id: str,
    workdir: str,
    nodes: str | None,
    expect_nodes: int,
    extra: Sequence[str] = (),
    *,
    hosts: str | None = None,
    platform: str = "lium",
) -> list[str]:
    """Arguments of ``cluster_bench.sh run`` for this experiment."""
    argv = ["--exp-id", exp_id, "--workdir", workdir, "--platform", platform]
    if hosts is not None:
        argv += ["--hosts", hosts]
    else:
        argv += ["--nodes", nodes or "auto"]
        if nodes in (None, "auto"):
            argv += ["--expect-nodes", str(expect_nodes)]
    return [*argv, *RESUME_ARGS, *extra]


def hosts_path(state: ControllerState) -> str:
    return f"{state.exp_dir}/hosts.txt"


def hosts_text(state: ControllerState) -> str:
    """The runner's --hosts file for SSH machines (node0 = this host)."""
    members = sorted(state.members, key=lambda m: m.rank)
    if len(state.control_ips) != len(members):
        raise ControllerError("the machines' control addresses are not recorded")
    rows: list[tuple[str, int | None, str | None]] = []
    for m in members:
        if m.rank > 0 and m.peer_ssh is None:
            raise ControllerError(
                f"node0 has no checked ssh route to {m.label or m.name}; rerun "
                "`controller up --machines ...` with a new --exp-id"
            )
        peer = m.peer_ssh
        rows.append((peer.target if peer else "-", peer.port if peer else None, None))
    control = [
        mach.Address(m.control_interface or "auto", ip, 32)
        for m, ip in zip(members, state.control_ips)
    ]
    return mach.hosts_file_text(
        rows,
        control,
        mach.trust_key_path(state.workdir, state.exp_id),
        known_hosts=mach.known_hosts_path(state.workdir, state.exp_id),
    )


PROVIDER_EOF = "CB_PROVIDER_JSON_EOF"


def provider_record(state: ControllerState) -> ProviderInfo:
    """What the knowledge base may keep about the rental: no secrets, no IPs."""
    return ProviderInfo(
        provider=state.provider,
        cluster_id=state.cluster_id,
        cluster_name=state.cluster_name,
        fabric_id=state.fabric,
        fabric_type=state.fabric_type,
        link_rate=state.link_rate,
        fabric_measured=state.fabric_measured,
        gpu_type=state.gpu,
        node_count=state.node_count,
        price_per_node_hour=state.price_per_node_hour,
        price_per_hour=state.cluster_price_per_hour,
        template=state.template,
        members=tuple(
            ProviderMember(
                name=m.name,
                rank=m.rank,
                huid=m.huid,
                pod_id=m.pod_id,
                gpu_type=m.gpu_type,
                gpu_count=m.gpu_count,
            )
            for m in state.members
        ),
    )


def _write_file_lines(exp_dir: str, name: str, content: str) -> list[str]:
    """Shell lines writing ``<exp_dir>/<name>`` atomically from a heredoc."""
    if PROVIDER_EOF in content:
        raise ControllerError(f"{name} contains the heredoc delimiter")
    target = shlex.quote(f"{exp_dir}/{name}")
    tmp = shlex.quote(f"{exp_dir}/.{name}.tmp")
    return [
        f"cat > {tmp} <<'{PROVIDER_EOF}'",
        content.rstrip("\n"),
        PROVIDER_EOF,
        f"mv -f {tmp} {target}",
    ]


_INPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def input_path(workdir: str, name: str) -> str:
    """Where ``controller sync --upload`` puts a local file on node0."""
    if not _INPUT_NAME_RE.match(name):
        raise ControllerError(
            f"{name!r}: an uploaded file name may only use letters, digits, "
            "'.', '_' and '-'"
        )
    return f"{workdir}/inputs/{name}"


def upload_script(dest: str) -> str:
    """Write stdin to ``dest`` atomically, then print its sha256."""
    tmp, final = shlex.quote(f"{dest}.part"), shlex.quote(dest)
    return "\n".join(
        [
            "set -eu",
            f"mkdir -p {shlex.quote(str(PurePosixPath(dest).parent))}",
            f"cat > {tmp}",
            f"mv -f {tmp} {final}",
            f"sum=$( (sha256sum {final} 2>/dev/null || shasum -a 256 {final})"
            " | cut -d' ' -f1)",
            'echo "@@SHA256 $sum"',
        ]
    )


def launch_script(
    workdir: str,
    exp_id: str,
    run_args: Sequence[str],
    provider: ProviderInfo | None = None,
    files: Mapping[str, str] | None = None,
) -> str:
    """Read HF_TOKEN from stdin, then start the runner detached (setsid nohup).

    ``provider`` is written to ``<exp_dir>/provider.json`` first (atomically)
    so the summary can turn the rental price into $/token; ``files`` (name ->
    content, e.g. the SSH machines' hosts.txt) are written the same way.
    """
    exp_dir = f"{workdir}/experiments/{exp_id}"
    log = shlex.quote(f"{exp_dir}/runner.log")
    command = shlex.join(["bash", RUNNER_SCRIPT, "run", *run_args])
    provider_lines: list[str] = []
    if provider is not None:
        provider_lines = _write_file_lines(
            exp_dir, "provider.json", provider.model_dump_json()
        )
    for name, content in sorted((files or {}).items()):
        provider_lines += _write_file_lines(exp_dir, name, content)
    return "\n".join(
        [
            "set -eu",
            "IFS= read -r HF_TOKEN || true",
            'if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; else unset HF_TOKEN; fi',
            f"mkdir -p {shlex.quote(exp_dir)}",
            *provider_lines,
            f"cd {shlex.quote(workdir + '/src')}",
            f"export CLUSTER_BENCH_WORKDIR={shlex.quote(workdir)}",
            'echo "@@NOW $(date +%s)"',
            f'echo "=== controller launch $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> {log}',
            f"setsid nohup {command} >> {log} 2>&1 < /dev/null &",
            'echo "@@PID $!"',
        ]
    )


def _alive_function(exp_id: str) -> str:
    """``alive PID``: true while PID is this experiment's runner (bash or python)."""
    return "\n".join(
        [
            f"EXP={shlex.quote(exp_id)}",
            "alive() {",
            '  [ -n "$1" ] && [ -r "/proc/$1/cmdline" ] || return 1',
            "  c=$(tr '\\000' ' ' < \"/proc/$1/cmdline\" 2>/dev/null) || return 1",
            '  case "$c" in *cluster_bench*" run "*"--exp-id $EXP "*) return 0 ;; esac',
            "  return 1",
            "}",
        ]
    )


def snapshot_script(status_path: str, exp_id: str, pids: Sequence[int]) -> str:
    """Print status.json, the liveness of the runner pids and node0's clock."""
    extra = " ".join(str(int(p)) for p in pids)
    return "\n".join(
        [
            f"f={shlex.quote(status_path)}",
            'if [ -f "$f" ]; then echo @@STATUS-BEGIN; cat "$f"; echo; echo @@STATUS-END; fi',
            _alive_function(exp_id),
            r"""spid=$(sed -n 's/^[[:space:]]*"pid":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -n 1)""",
            f'for p in $spid {extra}; do if alive "$p"; then echo "@@ALIVE $p 1"; '
            'else echo "@@ALIVE $p 0"; fi; done',
            'echo "@@NOW $(date +%s)"',
        ]
    )


def stop_script(exp_id: str, pids: Sequence[int], grace_s: int = 90) -> str:
    """SIGTERM the runner pids (it tears its deployment down), SIGKILL on timeout."""
    plist = " ".join(str(int(p)) for p in pids)
    return "\n".join(
        [
            _alive_function(exp_id),
            f'for p in {plist}; do if alive "$p"; then kill -TERM "$p" 2>/dev/null || true; fi; done',
            "i=0",
            f"while [ $i -lt {int(grace_s)} ]; do",
            f'  any=0; for p in {plist}; do if alive "$p"; then any=1; fi; done',
            '  [ "$any" -eq 0 ] && break',
            "  sleep 1; i=$((i+1))",
            "done",
            f'for p in {plist}; do if alive "$p"; then kill -KILL "$p" 2>/dev/null || true; fi; done',
            "echo @@STOPPED",
        ]
    )


def marker_values(stdout: str, marker: str) -> list[str]:
    """Values printed as ``<marker> <value>`` lines by a remote snippet."""
    prefix = marker + " "
    return [
        line[len(prefix) :].strip()
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]


def _marker_number(stdout: str, marker: str) -> float:
    values = marker_values(stdout, marker)
    try:
        return float(values[-1])
    except (IndexError, ValueError) as exc:
        raise ControllerError(
            f"no '{marker} <number>' line in remote output:\n{stdout[-2000:]}"
        ) from exc


@dataclass(frozen=True)
class RemoteSnapshot:
    status: dict[str, Any] | None  # status.json, None when absent
    remote_now: float  # node0 clock (epoch seconds)
    alive: dict[int, bool]  # runner pid -> process alive


def parse_snapshot(stdout: str) -> RemoteSnapshot:
    """Parse :func:`snapshot_script` output; raise ControllerError on garbage."""
    lines = stdout.splitlines()
    status: dict[str, Any] | None = None
    if "@@STATUS-BEGIN" in lines:
        start = lines.index("@@STATUS-BEGIN")
        if "@@STATUS-END" not in lines[start:]:
            raise ControllerError(
                f"truncated status.json in remote output:\n{stdout[-2000:]}"
            )
        end = start + lines[start:].index("@@STATUS-END")
        text = "\n".join(lines[start + 1 : end]).strip()
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise ControllerError(
                f"status.json is not JSON ({exc}):\n{text[-2000:]}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ControllerError(f"status.json is not a JSON object:\n{text[-2000:]}")
        status = parsed
    alive: dict[int, bool] = {}
    for value in marker_values(stdout, "@@ALIVE"):
        fields = value.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1] in ("0", "1"):
            alive[int(fields[0])] = fields[1] == "1"
    return RemoteSnapshot(status, _marker_number(stdout, "@@NOW"), alive)


# --------------------------------------------------------------------------
# Watch decisions and progress lines (pure)
# --------------------------------------------------------------------------

WatchAction = Literal["wait", "done", "failed", "relaunch"]


@dataclass(frozen=True)
class WatchDecision:
    action: WatchAction
    reason: str


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first_line(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip().splitlines()
    return (text[0] if text else "")[:limit]


def decide_watch(
    snap: RemoteSnapshot,
    *,
    launched_pid: int | None,
    launched_remote_ts: float | None,
    stale_after_s: float,
) -> WatchDecision:
    """What the watcher does next, from one remote snapshot.

    status.json older than the last launch belongs to the previous runner:
    while the new one has not written yet, its pid is what counts.
    """
    status = snap.status
    heartbeat = _number(status.get("heartbeat_ts")) if status is not None else None
    fresh = status is not None and (
        launched_remote_ts is None
        or (heartbeat is not None and heartbeat >= launched_remote_ts)
    )
    if status is None or not fresh:
        if launched_pid is not None and snap.alive.get(launched_pid):
            return WatchDecision("wait", f"runner pid {launched_pid} is starting")
        if launched_pid is None:
            return WatchDecision(
                "failed",
                "no status.json on node0 and no runner launched by this controller; "
                "run `controller launch`",
            )
        return WatchDecision(
            "failed",
            f"runner pid {launched_pid} exited before writing status.json "
            "(a startup error; see runner.log)",
        )
    state = status.get("state")
    if state == "done":
        return WatchDecision("done", "runner finished")
    if state == "failed":
        return WatchDecision(
            "failed", f"runner failed: {_first_line(status.get('error'))}"
        )
    if state == "interrupted":
        return WatchDecision(
            "relaunch",
            f"runner was interrupted ({_first_line(status.get('error'))}); resumable",
        )
    pid = _number(status.get("pid"))
    if pid is not None and snap.alive.get(int(pid)) is False:
        return WatchDecision(
            "relaunch", f"runner pid {int(pid)} is gone (state={state!r})"
        )
    if heartbeat is not None and snap.remote_now - heartbeat > stale_after_s:
        return WatchDecision(
            "relaunch", f"heartbeat is {snap.remote_now - heartbeat:.0f}s old"
        )
    return WatchDecision("wait", f"state={state}")


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def format_progress(status: Mapping[str, Any] | None, remote_now: float) -> str:
    """One line: state, phase, step, budget, points, last tok/s/GPU, heartbeat age."""
    if status is None:
        return "no status.json yet"
    parts = [str(status.get("state") or "?"), f"phase={status.get('phase') or '?'}"]
    step = _first_line(status.get("step"), 70)
    if step:
        parts.append(f"step={step!r}")
    budget = status.get("budget")
    if isinstance(budget, Mapping):
        spent = _number(budget.get("spent_s"))
        left = _number(budget.get("remaining_s"))
        if spent is not None:
            parts.append(f"spent={_hours(spent)}")
        if left is not None:
            parts.append(f"left={_hours(left)}")
    parts.append(f"points={status.get('points_done', 0)}")
    last = status.get("last_point")
    if isinstance(last, Mapping) and _number(last.get("tps_per_gpu")) is not None:
        parts.append(
            f"last={float(last['tps_per_gpu']):.1f} tok/s/GPU "
            f"(c={last.get('concurrency')}, {last.get('status')})"
        )
    heartbeat = _number(status.get("heartbeat_ts"))
    if heartbeat is not None:
        parts.append(f"hb={max(0.0, remote_now - heartbeat):.0f}s ago")
    return " | ".join(parts)


# --------------------------------------------------------------------------
# Fabric choice, names, banner
# --------------------------------------------------------------------------


def normalize_gpu(name: str) -> str:
    """``b200`` -> ``B200``, ``rtx_pro_6000`` -> ``RTXPRO6000`` (lium's types)."""
    key = re.sub(r"[\s_\-]+", "", name).upper()
    return _GPU_ALIASES.get(key, key)


# Raw Nsight Systems captures of the diagnostic window (hundreds of MB each)
# stay on the nodes; their `nsys stats` CSVs and the summary are pulled.
PULL_EXCLUDES = ("*.nsys-rep", "*.sqlite", "*.qdstrm")


def profile_name(gpu: str) -> str | None:
    """lium/CLI GPU spelling -> GPU profile name (``RTX PRO 6000`` -> rtx_pro_6000)."""
    wanted = normalize_gpu(gpu)
    for name in profile_store.load_gpu_profiles():
        if normalize_gpu(name) == wanted:
            return name
    return None


def size_problems(gpu: str | None, nodes: int, model: str = DEFAULT_MODEL) -> list[str]:
    """Why renting ``nodes`` nodes of ``gpu`` cannot run the tuned profiles.

    Checked before renting so an unusable size never bills. Unknown GPUs or
    models without tuned families are left to the runner (generic profile).
    """
    if nodes < 2:
        return [f"--nodes {nodes}: the harness needs at least 2 nodes"]
    name = profile_name(gpu) if gpu else None
    if name is None:
        return []
    try:
        fitted = profile_store.load_fitted_profile(
            model, name, nodes=nodes, gpus_per_node=GPUS_PER_NODE
        )
    except ProfileError:
        return []
    if not any(f.enabled for f in fitted.families):
        return [
            f"no tuned {model} family fits {nodes} x {GPUS_PER_NODE} {name}; "
            f"tuned sizes are {', '.join(map(str, TUNED_NODE_COUNTS))} nodes "
            "(or run an untuned model with -- --model-repo ...)"
        ]
    return []


def find_fabric(fabrics: Sequence[LiumFabric], ref: str) -> LiumFabric | None:
    exact = [f for f in fabrics if f.fabric_id == ref]
    if exact:
        return exact[0]
    prefix = [f for f in fabrics if f.fabric_id.startswith(ref)]
    return prefix[0] if len(prefix) == 1 else None


def pick_fabric(fabrics: Sequence[LiumFabric], gpu: str, nodes: int) -> LiumFabric:
    """The fabric to rent: ``nodes`` free ``gpu`` nodes; measured, IB, cheapest."""
    want = normalize_gpu(gpu)
    fits = [
        f
        for f in fabrics
        if normalize_gpu(f.gpu_type) == want and f.free_count >= nodes
    ]
    if not fits:
        offer = sorted(
            {f"{f.gpu_type} x{f.gpus_per_node} ({f.free_count} free)" for f in fabrics}
        )
        raise ControllerError(
            f"no fabric has {nodes} free {want} nodes; on offer: {offer or 'nothing'}"
        )
    return min(
        fits,
        key=lambda f: (
            not f.fabric_measured,
            f.fabric_type.lower() != "infiniband",
            f.price_per_node_hour,
            f.fabric_id,
        ),
    )


def default_cluster_name(gpu: str, exp_id: str) -> str:
    """``cb-<gpu>-<exp_id>``, lower-case, safe as a lium pod name."""
    name = re.sub(r"[^a-z0-9-]+", "-", f"cb-{gpu}-{exp_id}".lower())
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name[:63].rstrip("-")


def left_running_banner(state: ControllerState, reason: str) -> str:
    """A loud multi-line warning that a cluster is still up (and billing)."""
    lines = [
        f"CLUSTER LEFT RUNNING - IT IS STILL BILLING: {state.cluster_name} "
        f"({state.cluster_ref})",
        f"reason: {reason}",
    ]
    if state.ttl_hours:
        lines.append(
            f"lium TTL: removed automatically {state.ttl_hours:g} h after creation"
        )
    if state.created_by_harness:
        lines.append(
            "remove: bash benchmarks/cluster_bench/cluster_bench.sh controller down "
            f"--exp-id {state.exp_id} [--force]"
        )
    else:
        lines.append(
            "adopted cluster: the harness never removes it; remove it yourself"
        )
    lines.append(f"    or: lium clusters rm {state.cluster_ref} -y")
    width = max(len(line) for line in lines)
    bar = "!" * (width + 6)
    return "\n".join([bar, *(f"!! {line.ljust(width)} !!" for line in lines), bar])


@dataclass(frozen=True)
class UpOptions:
    # SSH machines (node0 first); empty = rent or adopt on lium.
    machines: tuple[mach.MachineSpec, ...] = ()
    fabric: str = "auto"
    gpu: str | None = None
    nodes: int = DEFAULT_NODES
    name: str | None = None
    ttl_hours: float = DEFAULT_TTL_HOURS
    template: str | None = None
    adopt: str | None = None
    workdir: str = DEFAULT_WORKDIR
    up_timeout_s: int = 1500
    ssh_wait_s: float = 600.0
    backups: bool = False


@dataclass(frozen=True)
class WatchOptions:
    interval_s: float = 60.0
    pull_every_s: float = 1800.0
    stale_after_s: float = 300.0
    max_relaunch: int = 2
    max_poll_failures: int = 30


class WallClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _say(message: str) -> None:
    print(message, flush=True)


def _warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _tail(text: str, limit: int = 1500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


class Controller:
    """Drives one experiment on one cluster; every step persists its state."""

    def __init__(
        self,
        exp_id: str,
        *,
        state_dir: Path,
        lium: LiumCli,
        runner: ArgvRunner | None = None,
        clock: Clock | None = None,
        repo_root: Path = REPO_ROOT,
        ssh_identity: Path | None = None,
        env: Mapping[str, str] | None = None,
    ):
        self.exp_id = check_exp_id(exp_id)
        self.dir = state_dir / exp_id
        self.lium = lium
        self.runner = runner or run_argv
        self.clock = clock or WallClock()
        self.repo_root = repo_root
        self.ssh_identity = ssh_identity
        self._env = env

    # -- state ----------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.dir / STATE_FILE

    @property
    def known_hosts(self) -> Path:
        return self.dir / "known_hosts"

    def load_state_optional(self) -> ControllerState | None:
        if not self.state_path.exists():
            return None
        with open(self.state_path, "r", encoding="utf-8") as fh:
            return ControllerState.model_validate(json.load(fh))

    def load_state(self) -> ControllerState:
        state = self.load_state_optional()
        if state is None:
            raise ControllerError(
                f"no controller state at {self.state_path}; run `controller up` first"
            )
        return state

    def save_state(self, state: ControllerState) -> None:
        if state.exp_id != self.exp_id:
            raise ValueError(f"state of {state.exp_id} saved under {self.exp_id}")
        atomic_write_json(self.state_path, state.model_dump(mode="json"))
        os.chmod(self.state_path, 0o600)

    # -- processes --------------------------------------------------------------

    def _source_env(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def _child_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """The inherited environment without secrets (ssh, rsync, git, kb)."""
        env = {
            k: v for k, v in self._source_env().items() if k not in GUARDED_SECRET_VARS
        }
        env.update(extra or {})
        return env

    def _run(
        self,
        argv: Sequence[str],
        *,
        input_data: bytes | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        assert_no_secrets(argv, self._source_env())
        return self.runner(
            argv,
            input_data=input_data,
            timeout=timeout,
            env=self._child_env() if env is None else env,
        )

    def _ssh_options(
        self, port_flag: str, port: int, endpoint: SshEndpoint | None = None
    ) -> list[str]:
        opts = [
            port_flag, str(port),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f'UserKnownHostsFile="{self.known_hosts}"',
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            "-o", "LogLevel=ERROR",
        ]  # fmt: skip
        identity = endpoint.identity if endpoint and endpoint.identity else None
        if identity is None and self.ssh_identity is not None:
            identity = str(self.ssh_identity)
        if identity is not None:
            opts += ["-i", identity, "-o", "IdentitiesOnly=yes"]
        if endpoint is not None and endpoint.jump:
            opts += ["-J", endpoint.jump]
        for option in endpoint.options if endpoint else ():
            opts += ["-o", option]
        return opts

    def ssh_argv(self, endpoint: SshEndpoint, script: str) -> list[str]:
        return [
            "ssh",
            *self._ssh_options("-p", endpoint.port, endpoint),
            endpoint.target,
            "bash",
            "-c",
            shlex.quote(script),
        ]

    def _ssh(
        self,
        endpoint: SshEndpoint,
        script: str,
        *,
        input_data: bytes | None = None,
        timeout: float = 300.0,
        check: bool = True,
    ) -> CommandResult:
        self.dir.mkdir(parents=True, exist_ok=True)  # holds known_hosts
        res = self._run(
            self.ssh_argv(endpoint, script), input_data=input_data, timeout=timeout
        )
        if check and res.rc != 0:
            raise ControllerError(
                f"ssh {endpoint.target} -p {endpoint.port} failed (exit {res.rc}):\n"
                f"{_tail(res.stderr)}\n{_tail(res.stdout)}".rstrip()
            )
        return res

    @staticmethod
    def _node0(state: ControllerState) -> SshEndpoint:
        if state.node0_ssh is None:
            raise ControllerError(
                "node0's ssh endpoint is unknown; run `controller up` (or `up --adopt`)"
            )
        return state.node0_ssh

    def hf_token(self) -> str:
        env = self._source_env()
        token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or ""
        if "\n" in token or "\r" in token:
            raise ControllerError("HF_TOKEN contains a newline")
        return token

    # -- up ---------------------------------------------------------------------

    def _record(self, state: ControllerState, cluster: LiumCluster) -> ControllerState:
        members = tuple(
            MemberRecord(
                name=n.name,
                rank=n.rank,
                huid=n.huid,
                pod_id=n.pod_id,
                ssh=SshEndpoint(target=n.ssh_target, port=n.ssh_port)
                if n.ssh_target
                else None,
                control_ip=n.control_ip,
                gpu_type=n.gpu_type,
                gpu_count=n.gpu_count,
            )
            for n in cluster.nodes
        )
        node0 = members[0] if members else None
        ips = tuple(m.control_ip for m in members if m.control_ip)
        return state.evolve(
            cluster_id=cluster.id,
            cluster_name=cluster.name or state.cluster_name,
            node0_ssh=node0.ssh.model_dump() if node0 and node0.ssh else None,
            control_ips=ips if len(ips) == len(members) else (),
            members=[m.model_dump() for m in members],
            node_count=len(members) or state.node_count,
            cluster_price_per_hour=cluster.price_per_hour
            if cluster.price_per_hour is not None
            else state.cluster_price_per_hour,
        )

    def _lookup(self, name: str) -> LiumCluster | None:
        found = self.lium.find_clusters(name)
        if len(found) > 1:
            raise ControllerError(
                f"{len(found)} clusters are named {name!r} ({[c.id for c in found]}); "
                "resolve it with `lium clusters ps`"
            )
        return found[0] if found else None

    def _lookup_quietly(self, name: str) -> LiumCluster | None:
        try:
            return self._lookup(name)
        except (LiumCliError, ControllerError) as exc:
            _warn(f"could not check whether cluster {name!r} exists: {exc}")
            return None

    def up(self, opts: UpOptions) -> ControllerState:
        """Rent the cluster (or adopt one) and record it before anything else."""
        check_workdir(opts.workdir)
        existing = self.load_state_optional()
        if (
            existing is not None
            and existing.provider == "ssh"
            and existing.released_at is None
        ):
            raise ControllerError(
                f"experiment {self.exp_id} already has {existing.node_count} SSH "
                "machine(s) recorded; run `controller down` first or use a new --exp-id"
            )
        if opts.machines:
            if (
                existing is not None
                and existing.provider == "lium"
                and existing.destroyed_at is None
            ):
                raise ControllerError(
                    f"experiment {self.exp_id} already has lium cluster "
                    f"{existing.cluster_name} recorded; run `controller down` first "
                    "or use a new --exp-id"
                )
            return self._up_machines(opts)
        if existing is not None and existing.destroyed_at is None:
            if existing.cluster_id is not None:
                raise ControllerError(
                    f"experiment {self.exp_id} already has cluster "
                    f"{existing.cluster_name} ({existing.cluster_id}); run "
                    "`controller down` first or use a new --exp-id"
                )
            found = self._lookup(existing.cluster_name)
            if found is not None:
                self.save_state(self._record(existing, found))
                raise ControllerError(
                    f"an earlier `up` rented {existing.cluster_name} ({found.id}); it "
                    f"is now recorded in {self.state_path}: continue with "
                    "`controller sync`, or remove it with `controller down --force`"
                )
        if opts.adopt:
            return self._adopt(opts)
        problems = size_problems(opts.gpu, opts.nodes)
        if problems:
            raise ControllerError("; ".join(problems))
        if opts.nodes not in TUNED_NODE_COUNTS:
            _warn(
                f"{opts.nodes} nodes: tuned profiles are written for "
                f"{'/'.join(map(str, TUNED_NODE_COUNTS))} nodes; families that do "
                "not scale exactly to this size are skipped"
            )
        gpu = opts.gpu
        fabric_id = opts.fabric
        offer: LiumFabric | None = None
        if fabric_id == "auto":
            if not gpu:
                raise ControllerError("--fabric auto needs --gpu (e.g. --gpu h200)")
            chosen = pick_fabric(self.lium.list_fabrics(), gpu, opts.nodes)
            offer = chosen
            fabric_id = chosen.fabric_id
            _say(
                f"fabric {chosen.fabric_id}: {chosen.fabric_type} {chosen.link_rate or ''}"
                f" {chosen.gpus_per_node}x{chosen.gpu_type}, {chosen.free_count}/"
                f"{chosen.node_count} free, ${chosen.price_per_node_hour:.2f}/node-hour, "
                f"measured={chosen.fabric_measured}"
            )
        else:
            offer = find_fabric(self.lium.list_fabrics(), fabric_id)
            if offer is not None:
                if gpu and normalize_gpu(offer.gpu_type) != normalize_gpu(gpu):
                    raise ControllerError(
                        f"fabric {offer.fabric_id} has {offer.gpu_type} nodes, not {gpu}"
                    )
                gpu = gpu or offer.gpu_type.lower()
            elif not gpu and opts.name is None:
                raise ControllerError(
                    f"fabric {fabric_id!r} is not listed by `lium clusters list`; "
                    "pass --gpu or --name"
                )
        name = opts.name or default_cluster_name(gpu or "lium", self.exp_id)
        self._refuse_name_in_use(name)
        state = ControllerState(
            exp_id=self.exp_id,
            cluster_name=name,
            created_by_harness=True,
            fabric=fabric_id,
            gpu=gpu,
            node_count=opts.nodes,
            ttl_hours=opts.ttl_hours,
            workdir=opts.workdir,
            created_at=self.clock.now(),
            fabric_type=offer.fabric_type if offer else None,
            link_rate=offer.link_rate if offer else None,
            fabric_measured=offer.fabric_measured if offer else None,
            price_per_node_hour=offer.price_per_node_hour if offer else None,
            template=opts.template,
        )
        # Recorded before renting: a crash mid-rent still leaves the name on disk.
        self.save_state(state)
        _say(
            f"renting {opts.nodes} nodes of {fabric_id} as {name!r} "
            f"(TTL {format_ttl(opts.ttl_hours)}); this bills until `controller down`"
        )
        try:
            cluster = self.lium.cluster_up(
                fabric_id,
                opts.nodes,
                name,
                opts.ttl_hours,
                opts.template,
                wait_timeout_s=opts.up_timeout_s,
                backups=opts.backups,
            )
        except LiumCliError as exc:
            found = self._lookup_quietly(name)
            if found is not None:
                state = self._record(state, found)
                self.save_state(state)
            if found is not None or isinstance(
                exc, (LiumBillingUncertainError, LiumRetryableError)
            ):
                _warn(
                    left_running_banner(
                        state, f"`lium clusters up` failed: {exc.summary}"
                    )
                )
            raise
        state = self._record(state, cluster)
        self.save_state(state)
        _say(
            f"cluster {name} ({cluster.id}): {len(cluster.nodes)} nodes, {cluster.status}"
        )
        for member in state.members:
            ssh = f"{member.ssh.target} -p {member.ssh.port}" if member.ssh else "-"
            _say(
                f"  {member.name}  wg0={member.control_ip}  ssh={ssh}  {member.gpu_type}"
            )
        self._wait_ssh(state, opts.ssh_wait_s)
        return state

    # -- up: SSH machines --------------------------------------------------------

    @staticmethod
    def _endpoint(access: mach.SshAccess) -> SshEndpoint:
        return SshEndpoint(
            target=access.target,
            port=access.port,
            identity=access.identity,
            jump=access.jump,
            options=tuple(f"{k}={v}" for k, v in access.options),
        )

    def _up_machines(self, opts: UpOptions) -> ControllerState:
        """Check the machines the user brought and record them (node0 first).

        Every machine must answer over the given ssh access, list GPUs (the
        same model and count), run docker as the ssh user and have a writable
        workdir. node0 then gets a per-experiment key that the other machines
        trust (``authorized_keys`` comment ``cluster_bench:<exp_id>``; ``down``
        removes it), and every node0 -> machine route is checked by hostname.
        """
        specs = list(opts.machines)
        mach.check_unique(specs)
        endpoints = [self._endpoint(s.access) for s in specs]
        facts: list[mach.MachineFacts] = []
        for i, (spec, endpoint) in enumerate(zip(specs, endpoints)):
            res = self._ssh(
                endpoint, mach.discovery_script(opts.workdir), timeout=120, check=False
            )
            if res.rc != 0:
                raise ControllerError(
                    f"machine {i} ({endpoint.target} -p {endpoint.port}) refused ssh "
                    f"(exit {res.rc}): {_tail(res.stderr, 400)}"
                )
            facts.append(mach.parse_discovery(res.stdout))
        labels = [s.label or f.hostname for s, f in zip(specs, facts)]
        if len(set(labels)) != len(labels):
            labels = [f"{label}#{i}" for i, label in enumerate(labels)]
        problems = [
            p
            for f, label in zip(facts, labels)
            for p in mach.facts_problems(f, label, opts.workdir)
        ]
        shapes = {(len(f.gpus), f.gpus[0] if f.gpus else "") for f in facts}
        if len(shapes) > 1:
            problems.append(
                "the machines differ in GPUs ("
                + ", ".join(
                    f"{label}: {len(f.gpus)} x {f.gpus[0] if f.gpus else '-'}"
                    for label, f in zip(labels, facts)
                )
                + "); the harness needs identical nodes"
            )
        if problems:
            raise ControllerError("; ".join(problems))
        control = mach.choose_control(specs, facts)
        if any(f.uid != 0 for f in facts):
            _warn(
                "some ssh users are not root: the NVLink comparison (nvidia-smi "
                "nvlink -sBwMode) needs root, so pass `-- --no-nvlink-ablation` to "
                "`launch` if it fails"
            )
        gpu_name = facts[0].gpus[0]
        try:
            gpu = detect_gpu_profile(gpu_name, profile_store.load_gpu_profiles()).name
        except ProfileError:
            gpu = gpu_name  # no tuned profile: the runner falls back to the generic one
        members = tuple(
            MemberRecord(
                name=f"node{i}",
                rank=i,
                ssh=endpoints[i],
                control_ip=addr.ip,
                gpu_type=gpu_name,
                gpu_count=len(f.gpus),
                label=labels[i],
                hostname=f.hostname,
                control_interface=spec.iface or addr.iface,
            )
            for i, (spec, f, addr) in enumerate(zip(specs, facts, control))
        )
        state = ControllerState(
            exp_id=self.exp_id,
            provider="ssh",
            cluster_name=opts.name or f"ssh-{self.exp_id}",
            created_by_harness=False,
            gpu=gpu,
            node_count=len(specs),
            workdir=opts.workdir,
            created_at=self.clock.now(),
            node0_ssh=endpoints[0].model_dump(),
            control_ips=tuple(a.ip for a in control),
            members=[m.model_dump() for m in members],
        )
        self.save_state(state)  # recorded before any machine is changed
        if len(specs) > 1:
            state = self._trust_peers(state, specs, facts)
            self.save_state(state)
        for m in state.members:
            route = (
                f"{m.peer_ssh.target} -p {m.peer_ssh.port}"
                if m.peer_ssh
                else "this host"
            )
            _say(
                f"  {m.name} {m.label}: ssh {m.ssh.target if m.ssh else '-'}  "
                f"control {m.control_ip} ({m.control_interface})  from node0: {route}"
            )
        return state

    def _trust_peers(
        self,
        state: ControllerState,
        specs: Sequence[mach.MachineSpec],
        facts: Sequence[mach.MachineFacts],
    ) -> ControllerState:
        node0 = self._node0(state)
        res = self._ssh(
            node0, mach.keygen_script(state.workdir, self.exp_id), timeout=60
        )
        pubkey = (marker_values(res.stdout, "@@PUBKEY") or [""])[-1]
        members = list(sorted(state.members, key=lambda m: m.rank))
        for i in range(1, len(members)):
            member = members[i]
            assert member.ssh is not None and member.control_ip is not None
            done = self._ssh(
                member.ssh, mach.authorize_script(pubkey, self.exp_id), timeout=60
            )
            if "@@AUTHORIZED" not in done.stdout:
                raise ControllerError(
                    f"{member.label}: the trust key was not installed"
                )
            tried = []
            for target, port in mach.peer_candidates(specs[i], member.control_ip):
                check = self._ssh(
                    node0,
                    mach.peer_check_script(state.workdir, self.exp_id, target, port),
                    timeout=60,
                    check=False,
                )
                seen = (marker_values(check.stdout, "@@PEER") or [""])[-1]
                if seen == facts[i].hostname:
                    members[i] = member.model_copy(
                        update={"peer_ssh": SshEndpoint(target=target, port=port)}
                    )
                    break
                tried.append(f"{target} -p {port}: {seen or 'no answer'}")
            else:
                raise ControllerError(
                    f"node0 cannot ssh to {member.label} ({facts[i].hostname}) with the "
                    f"experiment key; tried {'; '.join(tried)}. Give the machine "
                    "peer_ssh=USER@HOST[:PORT] (how node0 reaches it)"
                )
        return state.evolve(members=[m.model_dump() for m in members])

    def _release_machines(self, state: ControllerState, *, force: bool) -> int:
        """SSH machines: remove this experiment's containers and trust key."""
        if not state.pulled_ok and not force:
            _warn(
                "results were not pulled after the runner stopped: run "
                "`controller pull` first, or `down --force` to clean up anyway"
            )
            return 1
        if force and state.runner_pid is not None:
            snap = self.snapshot(state)
            if snap is not None and any(snap.alive.values()):
                self._stop_runner(state, snap)
        label = sanitize_label(self.exp_id)
        failed = []
        for m in sorted(state.members, key=lambda m: -m.rank):  # node0 last
            if m.ssh is None:
                continue
            res = self._ssh(
                m.ssh,
                mach.cleanup_script(self.exp_id, label, m.rank == 0, state.workdir),
                timeout=300,
                check=False,
            )
            if "@@CLEANED" not in res.stdout:
                failed.append(f"{m.label or m.name} (exit {res.rc})")
        if failed:
            _warn(f"cleanup incomplete on {', '.join(failed)}; rerun `controller down`")
            return 1
        self.save_state(state.evolve(released_at=self.clock.now()))
        _say(
            f"removed this experiment's containers and trust key from "
            f"{len(state.members)} machine(s); the machines and "
            f"{state.workdir}/experiments/{self.exp_id} stay"
        )
        return 0

    def _adopt(self, opts: UpOptions) -> ControllerState:
        found = self.lium.find_clusters(opts.adopt or "")
        if len(found) != 1:
            raise ControllerError(
                f"--adopt {opts.adopt!r} matches {len(found)} clusters "
                f"{[c.id for c in found]}; pass a unique cluster name or id"
            )
        cluster = self.lium.cluster_info(found[0].id)
        state = ControllerState(
            exp_id=self.exp_id,
            cluster_name=cluster.name or opts.adopt or cluster.id,
            created_by_harness=False,
            gpu=opts.gpu,
            node_count=len(cluster.nodes),
            workdir=opts.workdir,
            created_at=self.clock.now(),
        )
        state = self._record(state, cluster)
        self.save_state(state)
        _say(
            f"adopted cluster {state.cluster_name} ({cluster.id}); created_by_harness="
            "false, so `controller down` will never remove it"
        )
        self._wait_ssh(state, opts.ssh_wait_s)
        return state

    def _refuse_name_in_use(self, name: str) -> None:
        clusters = self.lium.find_clusters(name)
        pods = [p for p in self.lium.list_pods() if p.get("name") == name]
        if clusters or pods:
            raise ControllerError(
                f"a cluster or pod named {name!r} already exists "
                f"(clusters {[c.id for c in clusters]}, pods "
                f"{[p.get('huid') for p in pods]}); choose another --name, or "
                f"`--adopt {name}` to use it without letting the harness remove it"
            )

    def _wait_ssh(self, state: ControllerState, timeout_s: float) -> None:
        endpoint = self._node0(state)
        deadline = self.clock.now() + timeout_s
        while True:
            res = self._ssh(endpoint, "true", timeout=60, check=False)
            if res.rc == 0:
                return
            if self.clock.now() >= deadline:
                raise ControllerError(
                    f"node0 ({endpoint.target} -p {endpoint.port}) refused ssh for "
                    f"{timeout_s:.0f}s: {_tail(res.stderr, 500)} (key: --ssh-identity)"
                )
            self.clock.sleep(10)

    # -- sync / launch ------------------------------------------------------------

    def sync(
        self, *, force: bool = False, uploads: Sequence[Path] = ()
    ) -> ControllerState:
        """Stream the repo subset to ``<workdir>/src`` on node0, verify, bootstrap.

        ``uploads`` (local files, e.g. a dataset) go to ``<workdir>/inputs/``.
        """
        state = self.load_state()
        endpoint = self._node0(state)
        if not force and state.runner_pid is not None:
            snap = self.snapshot(state)
            if snap is not None and any(snap.alive.values()):
                raise ControllerError(
                    f"the runner (pid {state.runner_pid}) is alive on node0; syncing "
                    "would swap its code underneath it (use --force to override)"
                )
        files = list_sync_files(self.repo_root, self._run)
        bundle = build_sync_bundle(self.repo_root, files)
        _say(
            f"syncing {bundle.count} files ({len(bundle.data) / 1e6:.1f} MB) to {state.workdir}/src"
        )
        res = self._ssh(
            endpoint, extract_script(state.workdir), input_data=bundle.data, timeout=900
        )
        if marker_values(res.stdout, "@@SYNC_OK") != [str(bundle.count)]:
            raise ControllerError(
                f"sync verification failed: expected '@@SYNC_OK {bundle.count}', got:\n"
                f"{_tail(res.stdout)}\n{_tail(res.stderr)}"
            )
        _say("bootstrapping the node0 venv (uv, Python 3.12, aiperf==0.14.0) ...")
        boot = self._ssh(endpoint, bootstrap_command(state.workdir), timeout=3600)
        lines = [line.strip() for line in boot.stdout.splitlines() if line.strip()]
        venv_python = lines[-1] if lines else ""
        if not venv_python.endswith("/venv/bin/python"):
            raise ControllerError(
                f"bootstrap_node.sh did not print the venv python on its last line:\n"
                f"{_tail(boot.stdout)}\n{_tail(boot.stderr)}"
            )
        state = state.evolve(
            synced_at=self.clock.now(),
            sync_manifest_sha256=bundle.digest,
            venv_python=venv_python,
        )
        self.save_state(state)
        _say(f"node0 ready: {venv_python} (manifest {bundle.digest[:12]})")
        self.upload(uploads)
        return state

    def upload(self, paths: Sequence[Path]) -> list[str]:
        """Copy local files to ``<workdir>/inputs/`` on node0, sha256-checked."""
        if not paths:
            return []
        state = self.load_state()
        endpoint = self._node0(state)
        out: list[str] = []
        for path in paths:
            local = Path(path).expanduser()
            if not local.is_file():
                raise ControllerError(f"--upload {path}: not a file")
            dest = input_path(state.workdir, local.name)
            data = local.read_bytes()
            res = self._ssh(
                endpoint, upload_script(dest), input_data=data, timeout=1800
            )
            digest = hashlib.sha256(data).hexdigest()
            if marker_values(res.stdout, "@@SHA256") != [digest]:
                raise ControllerError(
                    f"upload of {local} to node0:{dest} failed its sha256 check:\n"
                    f"{_tail(res.stdout)}\n{_tail(res.stderr)}"
                )
            _say(f"uploaded {local} -> node0:{dest} ({len(data) / 1e6:.1f} MB)")
            out.append(dest)
        return out

    def launch(
        self,
        *,
        runner_args: Sequence[str] | None = None,
        runner_nodes: str | None = None,
        force: bool = False,
    ) -> ControllerState:
        """Start the runner detached on node0; HF_TOKEN is written to ssh stdin."""
        state = self.load_state()
        endpoint = self._node0(state)
        extra = tuple(state.runner_args if runner_args is None else runner_args)
        check_runner_args(extra)
        if not force and state.runner_pid is not None:
            snap = self.snapshot(state)
            if snap is not None and any(snap.alive.values()):
                raise ControllerError(
                    f"a runner is already alive on node0 (pid {state.runner_pid}); "
                    "use `controller watch`"
                )
        files: dict[str, str] = {}
        if state.provider == "ssh":
            if runner_nodes:
                raise ControllerError(
                    "--runner-nodes does not apply to SSH machines (the controller "
                    "writes the runner's hosts file)"
                )
            nodes = None
            files["hosts.txt"] = hosts_text(state)
            run_args = runner_argv(
                self.exp_id,
                state.workdir,
                None,
                state.node_count,
                extra,
                hosts=hosts_path(state),
                platform="generic",
            )
        else:
            nodes = (
                runner_nodes
                or state.runner_nodes
                or (",".join(state.control_ips) if state.control_ips else "auto")
            )
            run_args = runner_argv(
                self.exp_id, state.workdir, nodes, state.node_count, extra
            )
        token = self.hf_token()
        if not token:
            _warn("HF_TOKEN is not set: gated or private model downloads will fail")
        res = self._ssh(
            endpoint,
            launch_script(
                state.workdir,
                self.exp_id,
                run_args,
                provider_record(state)
                if state.cluster_id or state.provider == "ssh"
                else None,
                files,
            ),
            input_data=(token + "\n").encode(),
            timeout=120,
        )
        pid = int(_marker_number(res.stdout, "@@PID"))
        state = state.evolve(
            runner_pid=pid,
            runner_args=extra,
            runner_nodes=nodes,
            launched_at=self.clock.now(),
            launched_remote_ts=_marker_number(res.stdout, "@@NOW"),
            launches=state.launches + 1,
            pulled_ok=False,
        )
        self.save_state(state)
        _say(f"runner started on node0: pid {pid}, log {state.exp_dir}/runner.log")
        return state

    # -- watch --------------------------------------------------------------------

    def snapshot(self, state: ControllerState) -> RemoteSnapshot | None:
        """status.json + runner liveness + node0 clock; None if unreachable."""
        pids = [state.runner_pid] if state.runner_pid is not None else []
        script = snapshot_script(state.status_path, self.exp_id, pids)
        res = self._ssh(self._node0(state), script, timeout=90, check=False)
        if res.rc != 0:
            _warn(f"node0 unreachable (ssh exit {res.rc}): {_tail(res.stderr, 300)}")
            return None
        try:
            return parse_snapshot(res.stdout)
        except ControllerError as exc:
            _warn(f"unreadable remote status: {exc}")
            return None

    def _stop_runner(self, state: ControllerState, snap: RemoteSnapshot) -> None:
        pids = sorted(pid for pid, alive in snap.alive.items() if alive)
        if pids:
            _warn(f"stopping runner pid(s) {pids} on node0 before relaunching")
            self._ssh(self._node0(state), stop_script(self.exp_id, pids), timeout=240)

    def _log_tail(self, state: ControllerState, lines: int = 40) -> str:
        script = f"tail -n {int(lines)} {shlex.quote(state.exp_dir + '/runner.log')}"
        res = self._ssh(self._node0(state), script, timeout=60, check=False)
        return (
            res.stdout if res.rc == 0 else f"(no runner.log: {_tail(res.stderr, 200)})"
        )

    def watch(
        self, opts: WatchOptions | None = None, *, results_dir: Path | None = None
    ) -> int:
        """Poll until the runner is done (0), failed (1) or unreachable (3)."""
        opts = opts or WatchOptions()
        state = self.load_state()
        last_pull = self.clock.now()
        failures = 0
        while True:
            snap = self.snapshot(state)
            if snap is None:
                failures += 1
                if failures >= opts.max_poll_failures:
                    _warn(
                        f"gave up after {failures} failed polls; the runner may still "
                        "be going: re-run `controller watch`"
                    )
                    return WATCH_LOST
            else:
                failures = 0
                decision = decide_watch(
                    snap,
                    launched_pid=state.runner_pid,
                    launched_remote_ts=state.launched_remote_ts,
                    stale_after_s=opts.stale_after_s,
                )
                stamp = time.strftime("%H:%M:%S", time.localtime(self.clock.now()))
                _say(f"[{stamp}] {format_progress(snap.status, snap.remote_now)}")
                if decision.action == "done":
                    _say(f"experiment {self.exp_id}: {decision.reason}")
                    return 0
                if decision.action == "failed":
                    _warn(f"experiment {self.exp_id}: {decision.reason}")
                    _warn(self._log_tail(state))
                    return 1
                if decision.action == "relaunch":
                    if state.relaunches >= opts.max_relaunch:
                        _warn(
                            f"{decision.reason}; already relaunched {state.relaunches} "
                            "time(s), giving up"
                        )
                        _warn(self._log_tail(state))
                        return 1
                    _warn(
                        f"{decision.reason}: relaunching with the same --exp-id (resume)"
                    )
                    self._stop_runner(state, snap)
                    relaunched = self.launch(force=True)
                    state = relaunched.evolve(relaunches=state.relaunches + 1)
                    self.save_state(state)
            if self.clock.now() - last_pull >= opts.pull_every_s:
                try:
                    self.pull(results_dir, final=False)
                except ControllerError as exc:
                    _warn(f"incremental pull failed (will retry): {exc}")
                last_pull = self.clock.now()
                state = self.load_state()
            self.clock.sleep(opts.interval_s)

    # -- pull / kb / down ---------------------------------------------------------

    def _results_dir(self, state: ControllerState, override: Path | None) -> Path:
        chosen = override or (Path(state.results_dir) if state.results_dir else None)
        if chosen is None:
            chosen = Path.cwd() / "cluster_bench_results" / self.exp_id
        return chosen.expanduser().resolve()

    def _runner_stopped(self, state: ControllerState, grace_s: float = 120.0) -> bool:
        """True once no runner pid is alive; waits out a runner that is exiting.

        The runner writes its terminal state just before the process exits, so
        a poll right after ``done`` can still see it alive for a moment.
        """
        deadline = self.clock.now() + grace_s
        while True:
            snap = self.snapshot(state)
            if snap is None:
                return False
            if not any(snap.alive.values()):
                return True
            exiting = (snap.status or {}).get("state") in FINISHED_STATES
            if not exiting or self.clock.now() >= deadline:
                return False
            self.clock.sleep(5)

    def pull(self, results_dir: Path | None = None, *, final: bool = True) -> Path:
        """Copy the experiment directory back; a final pull may set ``pulled_ok``."""
        state = self.load_state()
        endpoint = self._node0(state)
        dest = self._results_dir(state, results_dir)
        dest.mkdir(parents=True, exist_ok=True)
        finished = final and self._runner_stopped(state)
        self.dir.mkdir(parents=True, exist_ok=True)
        remote = f"{endpoint.target}:{state.exp_dir}"
        rsync = [
            "rsync", "-a", "--exclude=/.ssh/",
            *(f"--exclude={pattern}" for pattern in PULL_EXCLUDES),
            "-e", shlex.join(["ssh", *self._ssh_options("-p", endpoint.port, endpoint)]),
            f"{remote}/", f"{dest}/",
        ]  # fmt: skip
        res = self._run(rsync, timeout=3600)
        if res.rc != 0:
            _warn(
                f"rsync failed (exit {res.rc}): {_tail(res.stderr, 400)}; trying scp -r"
            )
            scp = [
                "scp",
                "-r",
                "-p",
                *self._ssh_options("-P", endpoint.port, endpoint),
                f"{remote}/*",
                f"{dest}/",
            ]
            res2 = self._run(scp, timeout=3600)
            if res2.rc != 0:
                raise ControllerError(
                    f"pull of {state.exp_dir} failed: rsync exit {res.rc} "
                    f"({_tail(res.stderr, 300)}); scp exit {res2.rc} ({_tail(res2.stderr, 300)})"
                )
        updates: dict[str, Any] = {
            "results_dir": str(dest),
            "last_pull_at": self.clock.now(),
        }
        if final:
            updates["pulled_ok"] = finished
        self.save_state(self.load_state().evolve(**updates))
        _say(f"pulled {state.exp_dir} -> {dest}")
        if final and not finished:
            _warn(
                "the runner is still alive (or node0 did not answer): this pull is a "
                "snapshot and `down` will refuse until a pull after the run ends"
            )
        return dest

    def kb_merge(self, results_dir: Path, kb_root: Path) -> None:
        """Merge the pulled experiment into a local KB (``cluster_bench kb``)."""
        if not (results_dir / "analysis" / "summary.json").is_file():
            raise ControllerError(
                f"{results_dir}/analysis/summary.json is missing (the run did not "
                "finalize); nothing to merge"
            )
        argv = [
            sys.executable, "-m", "benchmarks.cluster_bench", "kb",
            "--exp-dir", str(results_dir), "--kb-root", str(kb_root),
        ]  # fmt: skip
        pythonpath = os.pathsep.join(
            p
            for p in (str(self.repo_root), self._source_env().get("PYTHONPATH", ""))
            if p
        )
        env = self._child_env({"PYTHONPATH": pythonpath, "PYTHONSAFEPATH": "1"})
        res = self._run(argv, env=env, timeout=1800)
        if res.rc != 0:
            raise ControllerError(
                f"KB merge failed (exit {res.rc}):\n{_tail(res.stderr)}"
            )
        _say(f"KB merged into {kb_root}: {_tail(res.stdout, 500)}")

    def down(self, *, keep: bool = False, force: bool = False) -> int:
        """Remove the cluster when the harness created it and the results are in."""
        state = self.load_state()
        if state.provider == "ssh":
            if state.released_at is not None:
                _say("this experiment's containers and trust key were already removed")
                return 0
            if keep:
                _say("--keep: containers and trust key left on the machines")
                return 0
            return self._release_machines(state, force=force)
        if state.destroyed_at is not None:
            _say(f"cluster {state.cluster_name} was already removed")
            return 0
        if not state.created_by_harness:
            _warn(
                left_running_banner(state, "adopted cluster (created_by_harness=false)")
            )
            return 0
        if keep:
            _warn(left_running_banner(state, "--keep"))
            return 0
        if not state.pulled_ok and not force:
            _warn(
                left_running_banner(
                    state,
                    "results were not pulled after the runner stopped (run "
                    "`controller pull`, or `down --force` to discard them)",
                )
            )
            return 1
        provider = LiumProvider(self.lium, state.cluster_ref, created_by_harness=True)
        try:
            provider.destroy()
        except LiumNotFoundError:
            if self._lookup(state.cluster_name) is not None:
                raise
            _say(f"cluster {state.cluster_ref} no longer exists")
        except (LiumCliError, ProviderError) as exc:
            _warn(left_running_banner(state, f"`lium clusters rm` failed: {exc}"))
            raise
        self.save_state(state.evolve(destroyed_at=self.clock.now()))
        _say(f"removed cluster {state.cluster_name} ({state.cluster_ref})")
        return 0

    def banner_if_running(self, reason: str) -> None:
        state = self.load_state_optional()
        if state is None:
            return
        if state.provider == "ssh":
            if state.released_at is None:
                _warn(
                    f"{reason}: containers of experiment {state.exp_id} may still run "
                    f"on the machines; `controller down --exp-id {state.exp_id} "
                    "[--force]` removes them and the trust key"
                )
            return
        if state.destroyed_at is None:
            _warn(left_running_banner(state, reason))

    # -- status / run-all -----------------------------------------------------------

    def status(self) -> int:
        state = self.load_state()
        _say(f"experiment {state.exp_id}  state file {self.state_path}")
        _say(
            f"cluster    {state.cluster_name} ({state.cluster_ref}) created_by_harness="
            f"{state.created_by_harness} fabric={state.fabric} ttl={state.ttl_hours}"
        )
        if state.provider == "ssh":
            for m in sorted(state.members, key=lambda m: m.rank):
                route = (
                    f"{m.peer_ssh.target} -p {m.peer_ssh.port}" if m.peer_ssh else "-"
                )
                _say(
                    f"{m.name:<10} {m.label}  ssh {m.ssh.target if m.ssh else '-'} "
                    f"-p {m.ssh.port if m.ssh else '-'}  control {m.control_ip} "
                    f"({m.control_interface})  from node0 {route}"
                )
        elif state.node0_ssh is not None:
            _say(
                f"node0      {state.node0_ssh.target} -p {state.node0_ssh.port}  wg0 "
                f"{','.join(state.control_ips) or '?'}"
            )
        _say(
            f"runner     pid={state.runner_pid} launches={state.launches} "
            f"relaunches={state.relaunches} args={list(state.runner_args)}"
        )
        _say(f"results    {state.results_dir} pulled_ok={state.pulled_ok}")
        if state.destroyed_at is not None or state.released_at is not None:
            _say("cluster removed" if state.destroyed_at else "machines released")
            return 0
        if state.node0_ssh is not None:
            snap = self.snapshot(state)
            if snap is not None:
                _say(f"remote     {format_progress(snap.status, snap.remote_now)}")
        return 0

    def run_all(
        self,
        up: UpOptions,
        *,
        runner_args: Sequence[str] = (),
        runner_nodes: str | None = None,
        watch: WatchOptions | None = None,
        results_dir: Path | None = None,
        kb_root: Path | None = None,
        keep: bool = False,
        uploads: Sequence[Path] = (),
    ) -> int:
        """up -> sync -> launch -> watch -> pull -> KB merge -> down."""
        state = self.load_state_optional()
        reusable = state is not None and (
            state.released_at is None and state.members
            if state.provider == "ssh"
            else state.destroyed_at is None and state.cluster_id is not None
        )
        if not reusable:
            state = self.up(up)
        else:
            assert state is not None
            _say(f"reusing {state.cluster_name} ({state.cluster_id or state.provider})")
        snap = self.snapshot(state)
        if snap is None:
            raise ControllerError("node0 is unreachable over ssh")
        if snap.status is None and not any(snap.alive.values()):
            self.sync(uploads=uploads)
            self.launch(runner_args=runner_args, runner_nodes=runner_nodes)
        else:
            _say(
                "a runner already started for this experiment; skipping sync and launch"
            )
        rc = self.watch(watch, results_dir=results_dir)
        if rc == WATCH_LOST:
            self.banner_if_running("lost contact with node0 while watching")
            return rc
        dest = self.pull(results_dir, final=True)
        if kb_root is not None:
            try:
                self.kb_merge(dest, kb_root)
            except ControllerError as exc:
                _warn(f"KB merge skipped: {exc}")
        down_rc = self.down(keep=keep)
        return rc or down_rc


def check_runner_args(args: Sequence[str]) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in MANAGED_RUNNER_FLAGS:
            raise ControllerError(
                f"runner argument {flag} is set by the controller; use "
                "--runner-nodes / the controller's --workdir instead"
            )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _common_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--exp-id",
        default=None,
        help="experiment id (default for up/run-all: timestamp)",
    )
    p.add_argument(
        "--state-dir", type=Path, default=None, help="default: ~/.cluster_bench"
    )
    p.add_argument("--lium-bin", default="lium", help="lium executable")
    p.add_argument(
        "--ssh-identity",
        type=Path,
        default=None,
        help="private key for node ssh (default: lium's LIUM_SSH_KEY_PATH / ~/.lium/config.ini)",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return p


def _up_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--machines",
        type=Path,
        default=None,
        help="machines file: one SSH access per line, node0 first "
        "(format: adapters/machines.py); no lium rental",
    )
    p.add_argument(
        "--machine",
        action="append",
        default=None,
        metavar="SSH",
        help="one machine, e.g. 'ssh -p 2222 root@203.0.113.10 control_ip=10.0.0.11' "
        "(repeatable, after the --machines lines)",
    )
    p.add_argument(
        "--gpu", default=None, help="GPU type: b300, b200, h200, h100, rtx_pro_6000"
    )
    p.add_argument(
        "--fabric",
        default="auto",
        help="lium fabric id/prefix, or 'auto' (needs --gpu)",
    )
    p.add_argument(
        "--nodes",
        type=int,
        default=DEFAULT_NODES,
        help="cluster size: 2, 4 or 8 (tuned); other sizes >= 2 keep the families "
        "that scale exactly",
    )
    p.add_argument(
        "--name", default=None, help="cluster name (default cb-<gpu>-<exp_id>)"
    )
    p.add_argument(
        "--ttl-hours", type=float, default=DEFAULT_TTL_HOURS, help="lium auto-removal"
    )
    p.add_argument(
        "--template",
        default=None,
        help="lium --template_id (default: newest lium-cluster)",
    )
    p.add_argument(
        "--adopt",
        default=None,
        metavar="NAME",
        help="use an existing cluster; never removed",
    )
    p.add_argument(
        "--workdir", default=DEFAULT_WORKDIR, help="node-local harness directory"
    )
    p.add_argument(
        "--up-timeout",
        type=int,
        default=1500,
        help="seconds lium waits for the members",
    )
    p.add_argument(
        "--ssh-wait", type=float, default=600.0, help="seconds to wait for node0 ssh"
    )
    p.add_argument(
        "--lium-backup",
        action="store_true",
        help="keep lium's volume backups (off by default)",
    )
    return p


def _runner_args_parser(force: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--runner-nodes",
        default=None,
        help="runner --nodes (default: the wg0 IPs, else auto)",
    )
    if force:
        p.add_argument(
            "--force", action="store_true", help="launch even if a runner looks alive"
        )
    return p


def _watch_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
    p.add_argument(
        "--pull-every",
        type=float,
        default=1800.0,
        help="seconds between incremental pulls",
    )
    p.add_argument(
        "--stale-after",
        type=float,
        default=300.0,
        help="heartbeat age that triggers a relaunch",
    )
    p.add_argument("--max-relaunch", type=int, default=2)
    return p


def _results_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="default ./cluster_bench_results/<exp_id>/",
    )
    return p


def _upload_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--upload",
        action="append",
        type=Path,
        default=None,
        metavar="FILE",
        help="also copy a local file (e.g. a dataset) to <workdir>/inputs/ on node0; "
        "give it to the runner as --dataset <type>:<workdir>/inputs/<name> "
        "(repeatable)",
    )


def _remainder(p: argparse.ArgumentParser) -> None:
    p.add_argument("runner_args", nargs=argparse.REMAINDER, help="-- runner arguments")


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Add the ``controller`` subcommand (actions: see the module docstring)."""
    parser = subparsers.add_parser(
        "controller",
        help="laptop-side workflow on SSH machines or a lium rental "
        "(up, sync, launch, watch, pull, down)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(
        dest="controller_action", required=True, metavar="ACTION"
    )
    common = _common_args()
    actions.add_parser(
        "up", parents=[common, _up_args()], help="rent or adopt a cluster"
    )
    sync = actions.add_parser(
        "sync", parents=[common], help="sync code to node0 and bootstrap"
    )
    sync.add_argument(
        "--force", action="store_true", help="sync even if the runner is alive"
    )
    _upload_arg(sync)
    launch = actions.add_parser(
        "launch", parents=[common, _runner_args_parser(True)], help="start the runner"
    )
    _remainder(launch)
    actions.add_parser(
        "watch", parents=[common, _watch_args(), _results_args()], help="poll progress"
    )
    actions.add_parser(
        "pull", parents=[common, _results_args()], help="copy results back"
    )
    down = actions.add_parser("down", parents=[common], help="remove the cluster")
    down.add_argument("--keep", action="store_true", help="leave the cluster running")
    down.add_argument(
        "--force", action="store_true", help="remove even without a final pull"
    )
    actions.add_parser(
        "status", parents=[common], help="local state and remote heartbeat"
    )
    run_all = actions.add_parser(
        "run-all",
        parents=[
            common,
            _up_args(),
            _runner_args_parser(False),
            _watch_args(),
            _results_args(),
        ],
        help="up, sync, launch, watch, pull, KB merge, down",
    )
    run_all.add_argument(
        "--kb-root",
        type=Path,
        default=Path("cluster_bench_kb"),
        help="local KB directory",
    )
    run_all.add_argument("--no-kb", action="store_true", help="skip the KB merge")
    _upload_arg(run_all)
    run_all.add_argument(
        "--keep", action="store_true", help="leave the cluster running at the end"
    )
    _remainder(run_all)
    return parser


def _passthrough(raw: Sequence[str] | None) -> list[str]:
    args = list(raw or [])
    if not args:
        return []
    if args[0] != "--":
        raise ControllerError(f"runner arguments must follow '--' (got {args[0]!r})")
    return args[1:]


def machine_specs(
    machines_file: Path | None, machines: Sequence[str] | None
) -> tuple[mach.MachineSpec, ...]:
    """``--machines FILE`` lines, then each ``--machine``; node0 first."""
    specs: list[mach.MachineSpec] = []
    if machines_file is not None:
        path = machines_file.expanduser()
        specs += mach.parse_machines_text(path.read_text(encoding="utf-8"), str(path))
    for i, text in enumerate(machines or ()):
        specs.append(mach.parse_machine(text, f"--machine #{i + 1}"))
    mach.check_unique(specs)
    return tuple(specs)


def _up_options(args: argparse.Namespace) -> UpOptions:
    return UpOptions(
        machines=machine_specs(args.machines, args.machine),
        fabric=args.fabric,
        gpu=args.gpu,
        nodes=args.nodes,
        name=args.name,
        ttl_hours=args.ttl_hours,
        template=args.template,
        adopt=args.adopt,
        workdir=args.workdir,
        up_timeout_s=args.up_timeout,
        ssh_wait_s=args.ssh_wait,
        backups=args.lium_backup,
    )


def _watch_options(args: argparse.Namespace) -> WatchOptions:
    return WatchOptions(
        interval_s=args.interval,
        pull_every_s=args.pull_every,
        stale_after_s=args.stale_after,
        max_relaunch=args.max_relaunch,
    )


def _uses_lium_key(args: argparse.Namespace, state_dir: Path, exp_id: str) -> bool:
    """Whether ssh should default to lium's key: not for SSH machines.

    Their accesses name their own keys (or leave it to ssh's defaults, agent
    and config); with IdentitiesOnly a borrowed lium key would lock them out.
    """
    if getattr(args, "machines", None) or getattr(args, "machine", None):
        return False
    try:
        data = json.loads((state_dir / exp_id / "controller.json").read_text())
    except (OSError, ValueError):
        return True
    return not isinstance(data, dict) or data.get("provider", "lium") == "lium"


def build_controller(args: argparse.Namespace, exp_id: str) -> Controller:
    state_dir = (args.state_dir or Path.home() / ".cluster_bench").expanduser()
    identity = args.ssh_identity or (
        resolve_lium_ssh_key() if _uses_lium_key(args, state_dir, exp_id) else None
    )
    if identity is not None and not identity.expanduser().is_file():
        _warn(f"ssh key {identity} does not exist; using ssh's defaults")
        identity = None
    return Controller(
        exp_id,
        state_dir=state_dir,
        lium=LiumCli(args.lium_bin),
        ssh_identity=identity.expanduser() if identity else None,
    )


def main(args: argparse.Namespace) -> int:
    """Run one controller action; returns the process exit code."""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    action = args.controller_action
    ctl: Controller | None = None
    try:
        runner_args = _passthrough(vars(args).get("runner_args"))
        exp_id = args.exp_id
        if exp_id is None and action in ("up", "run-all"):
            exp_id = time.strftime("%Y%m%d-%H%M%S")
            _say(f"experiment id: {exp_id}")
        if exp_id is None:
            raise ControllerError(f"`controller {action}` needs --exp-id")
        ctl = build_controller(args, exp_id)
        if action == "up":
            ctl.up(_up_options(args))
            return 0
        if action == "sync":
            ctl.sync(force=args.force, uploads=tuple(args.upload or ()))
            return 0
        if action == "launch":
            ctl.launch(
                runner_args=runner_args or None,
                runner_nodes=args.runner_nodes,
                force=args.force,
            )
            return 0
        if action == "watch":
            return ctl.watch(_watch_options(args), results_dir=args.results_dir)
        if action == "pull":
            ctl.pull(args.results_dir, final=True)
            return 0
        if action == "down":
            return ctl.down(keep=args.keep, force=args.force)
        if action == "status":
            return ctl.status()
        if action == "run-all":
            return ctl.run_all(
                _up_options(args),
                runner_args=runner_args,
                runner_nodes=args.runner_nodes,
                watch=_watch_options(args),
                results_dir=args.results_dir,
                kb_root=None if args.no_kb else args.kb_root,
                keep=args.keep,
                uploads=tuple(args.upload or ()),
            )
        raise ControllerError(f"unknown controller action {action!r}")
    except KeyboardInterrupt:
        _warn("interrupted: whatever already runs on node0 keeps running")
        if ctl is not None:
            ctl.banner_if_running("controller interrupted")
        return 130
    except (ControllerError, ProviderError, LiumCliError, ValueError) as exc:
        _warn(f"controller {action}: {exc}")
        if ctl is not None and action == "run-all":
            ctl.banner_if_running(f"run-all stopped: {_first_line(exc)}")
        return 1
    except Exception as exc:
        # Anything unexpected after a rental must still tell the user that a
        # paid cluster may be running; then surface the real error.
        if ctl is not None:
            ctl.banner_if_running(f"controller {action} crashed: {_first_line(exc)}")
        raise
