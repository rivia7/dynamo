# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner-side model download: sharded Hugging Face download + fabric gather.

The steps are separate so the runner can overlap GPU work with the long shard
download and resume after a crash; ``ModelFetcher.fetch`` chains them:

1. ``plan``               resolve the commit, list files (size, sha256) and
                          LPT-shard the large files across nodes. One HfApi
                          call, then pure; ``FetchPlan`` is JSON-serializable.
2. ``fetch_small_files``  blocking: every node downloads the small files
                          (config, tokenizer, index, code), enough for a
                          dummy-weight smoke test.
3. ``start_downloads``    every node downloads its shard (plus the small files,
                          a no-op when present) in a detached background
                          process with a pid, log and status file.
4. ``wait_downloads``     poll the status files until every node finishes.
5. ``gather``             ``agents/file_gather.py`` on every node (one GPU per
                          node, torchrun): NCCL broadcasts over the RDMA fabric
                          fill in what a node lacks; each rank sha256-verifies
                          everything it holds and writes a report.
6. ``verify_and_mark``    check the rank reports and on-disk sizes, then write
                          ``<model_dir>/.cb_complete.json`` on every node.

``is_complete`` checks that marker (and the on-disk sizes) for resume.

The HF token only travels on stdin: a tiny ``python -c`` bootstrap reads the
first stdin line as the token and executes the rest of stdin as the download
script, passing the token as a variable. It is never in argv, the environment,
a file or a log.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from huggingface_hub import HfApi
from pydantic import Field

from benchmarks.lium_cluster_bench.adapters.docker import LABEL_ROLE, Mount, docker_run_argv
from benchmarks.lium_cluster_bench.adapters.microbench import (
    CONTAINER_SRC,
    DEFAULT_PORT_BASE,
    GATHER_SCRIPT,
    build_torchrun_command,
    rendezvous_port,
)
from benchmarks.lium_cluster_bench.domain.schema import Frozen, stable_hash
from benchmarks.lium_cluster_bench.ports import Clock, Cluster, CommandError

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
SMALL_FILE_BYTES = 64 * MIB
MARKER_NAME = ".cb_complete.json"
DOWNLOAD_MAX_WORKERS = 16
CHUNK_MIB = 256
GATHER_CONTAINER = "cb-model-gather"
GATHER_ROLE = "model-gather"
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
RESULT_PREFIX = "CB_DOWNLOAD_RESULT "
STARTED_PREFIX = "CB_STARTED "
FILES_SEPARATOR = "__CB_FILES__"
SHARD_PID = "shards.pid"
SHARD_LOG = "shards.log"
SHARD_STATUS = "shards.status.json"
SMALL_TIMEOUT_S = 3600.0
START_TIMEOUT_S = 300.0
GATHER_TIMEOUT_S = 4 * 3600.0
PROBE_TIMEOUT_S = 60.0
LOG_TAIL_LINES = 40

# Reads the token (first stdin line), then runs the rest of stdin as a script.
# The compile() filename doubles as the tag that identifies our process in
# /proc/<pid>/cmdline.
BOOTSTRAP_TAG = "cb-hf-download"
BOOTSTRAP = (
    "import sys\n"
    "token = sys.stdin.readline().rstrip('\\n')\n"
    f"code = compile(sys.stdin.read(), '<{BOOTSTRAP_TAG}>', 'exec')\n"
    "exec(code, {'__name__': '__cb_download__', 'CB_TOKEN': token or None})\n"
)

# Executed by BOOTSTRAP after a ``PARAMS = '<json>'`` line. Foreground mode
# prints one RESULT_PREFIX line; background mode detaches (double fork,
# setsid, stdio to the log file), records its pid and finally writes a status
# file {"rc": 0, "result": {...}} or {"rc": 1, "error": "..."}.
DOWNLOAD_SCRIPT = """
import json
import os
import signal
import sys
import time

from huggingface_hub import snapshot_download

P = json.loads(PARAMS)


def write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def daemonize(pid_path, log_path):
    sys.stdout.flush()
    sys.stderr.flush()
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child > 0:
        os.close(write_fd)
        with os.fdopen(read_fd) as pipe:
            pid = pipe.read().strip()
        os.waitpid(child, 0)
        print(("CB_STARTED " + pid) if pid else "CB_START_FAILED", flush=True)
        os._exit(0 if pid else 1)
    os.close(read_fd)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(null_fd)
    os.close(log_fd)
    with open(pid_path + ".tmp", "w") as fh:
        fh.write(str(os.getpid()))
    os.replace(pid_path + ".tmp", pid_path)
    os.write(write_fd, str(os.getpid()).encode())
    os.close(write_fd)


def download():
    started = time.monotonic()
    snapshot_download(
        repo_id=P["repo_id"],
        revision=P["commit"],
        local_dir=P["local_dir"],
        allow_patterns=P["allow_patterns"],
        max_workers=P["max_workers"],
        token=CB_TOKEN,
    )
    wrong = []
    for rel, size in sorted(P["sizes"].items()):
        path = os.path.join(P["local_dir"], rel)
        if not os.path.isfile(path) or os.path.getsize(path) != size:
            wrong.append(rel)
    if wrong:
        listed = ", ".join(wrong[:20])
        raise RuntimeError("missing or wrong-size after download: " + listed)
    return {
        "seconds": time.monotonic() - started,
        "files": len(P["sizes"]),
        "bytes": sum(P["sizes"].values()),
    }


if P["background"]:
    daemonize(P["pid_path"], P["log_path"])
    print("download", P["repo_id"], P["commit"], len(P["sizes"]), "files", flush=True)
    try:
        result = download()
    except BaseException as exc:
        write_json_atomic(P["status_path"], {"rc": 1, "error": repr(exc)[-2000:]})
        raise
    write_json_atomic(P["status_path"], {"rc": 0, "result": result})
    print("done", json.dumps(result), flush=True)
else:
    print("CB_DOWNLOAD_RESULT " + json.dumps(download()), flush=True)
"""


class ModelFetchError(RuntimeError):
    """Download, gather or verification failed; ``report`` is set by ``fetch``."""

    def __init__(
        self,
        message: str,
        *,
        problems: Sequence[str] = (),
        report: DownloadReport | None = None,
    ):
        super().__init__(message)
        self.problems = tuple(problems)
        self.report = report


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int
    sha256: str | None = None  # LFS sha256; None for plain git blobs


@dataclass(frozen=True)
class ShardPlan:
    """LPT assignment of large files; small files go to every node."""

    n_nodes: int
    owners: Mapping[str, int]  # large file path -> owner rank
    small: tuple[str, ...]
    rank_bytes: tuple[int, ...]  # large-file bytes per rank

    def files_for(self, rank: int) -> tuple[str, ...]:
        owned = sorted(path for path, owner in self.owners.items() if owner == rank)
        return (*self.small, *owned)


class PlannedFile(Frozen):
    path: str
    size: int
    sha256: str | None = None
    owner: int | None = None  # None: small file, every node downloads it


class FetchPlan(Frozen):
    """Everything later steps need; persist with ``model_dump(mode="json")``."""

    repo_id: str
    revision: str  # as requested
    commit: str  # resolved
    nodes: tuple[str, ...]  # rank order; nodes[0] hosts the gather rendezvous
    small_file_bytes: int
    files: tuple[PlannedFile, ...]  # sorted by path

    @property
    def plan_id(self) -> str:
        layout = [[f.path, f.size, f.owner] for f in self.files]
        return stable_hash(
            [self.repo_id, self.commit, list(self.nodes), self.small_file_bytes, layout]
        )

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def rank_of(self, node: str) -> int:
        try:
            return self.nodes.index(node)
        except ValueError as exc:
            raise KeyError(f"{node!r} is not in the plan {self.nodes}") from exc

    def small_files(self) -> tuple[PlannedFile, ...]:
        return tuple(f for f in self.files if f.owner is None)

    def shard_files(self, rank: int) -> tuple[PlannedFile, ...]:
        return tuple(f for f in self.files if f.owner == rank)

    def files_for(self, rank: int) -> tuple[PlannedFile, ...]:
        """What node ``rank`` downloads itself: the small files plus its shard."""
        return tuple(f for f in self.files if f.owner is None or f.owner == rank)


class DownloadHandle(Frozen):
    node: str
    rank: int
    pid: int | None  # None when an earlier attempt already finished
    pid_path: str
    log_path: str
    status_path: str
    files: int
    bytes: int
    reused: bool = False  # attached to a running or finished earlier attempt


class NodeDownloadResult(Frozen):
    node: str
    rank: int
    phase: Literal["small", "shards"]
    files: int
    bytes: int
    seconds: float
    mbps: float | None  # MB/s, 1 MB = 1e6 bytes


class RankReport(Frozen):
    """The parts of a ``file_gather.py`` rank report the runner checks."""

    rank: int
    node: str
    hostname: str
    bytes_received: int
    bytes_sent: int
    bytes_discarded: int
    bytes_broadcast: int
    seconds: float
    verify_seconds: float
    gbs: float | None  # GB/s (1e9 bytes/s)
    mismatches: tuple[str, ...]
    missing: tuple[str, ...]
    hashes: dict[str, str | None]  # path -> sha256 the rank computed
    failed: dict[str, str]  # path -> reason, for files that are not ok


class GatherResult(Frozen):
    ranks: tuple[RankReport, ...]
    seconds: float  # slowest rank's broadcast phase
    wall_seconds: float  # including container start and verification
    bytes_broadcast: int
    gbs: float | None
    exit_codes: dict[str, int]
    mismatches: tuple[str, ...]  # "<node>:<path>"
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.mismatches
            and not self.missing
            and not any(self.exit_codes.values())
        )


class VerifyResult(Frozen):
    nodes: tuple[str, ...]
    files: int
    total_bytes: int
    marker: str  # marker path on every node


class DownloadReport(Frozen):
    repo: str
    commit: str
    model_dir: str
    total_bytes: int
    file_count: int
    skipped: bool  # already complete on every node
    verified: bool
    small: dict[str, NodeDownloadResult] = Field(default_factory=dict)
    shards: dict[str, NodeDownloadResult] = Field(default_factory=dict)
    gather_seconds: float | None = None
    gather_gbs: float | None = None
    mismatches: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeState:
    state: Literal["done", "running", "absent"]
    pid: int | None = None
    status: Mapping[str, Any] | None = None


# --------------------------------------------------------------------------
# Pure planning, rendering and parsing
# --------------------------------------------------------------------------


def model_dir_for(model_root: str, repo_id: str, commit: str) -> str:
    """``<root>/<org>--<name>@<commit[:12]>``."""
    if not commit:
        raise ValueError("commit is required")
    return f"{model_root.rstrip('/')}/{repo_id.replace('/', '--')}@{commit[:12]}"


def list_repo_files(
    repo_id: str, revision: str, token: str | None
) -> tuple[str, list[RepoFile]]:
    """Resolve ``revision`` to a commit and list every file with size and sha256."""
    info = HfApi().model_info(
        repo_id, revision=revision, files_metadata=True, token=token or None
    )
    if not info.sha:
        raise ValueError(f"{repo_id}@{revision}: the hub returned no commit sha")
    files = []
    for sibling in info.siblings or ():
        lfs = sibling.lfs
        size = (
            sibling.size
            if sibling.size is not None
            else (lfs.size if lfs is not None else None)
        )
        if size is None:
            raise ValueError(f"{repo_id}@{info.sha}: no size for {sibling.rfilename}")
        files.append(
            RepoFile(
                sibling.rfilename, int(size), lfs.sha256 if lfs is not None else None
            )
        )
    return info.sha, sorted(files, key=_repo_path)


def _repo_path(item: RepoFile) -> str:
    return item.path


def _lpt_key(item: RepoFile) -> tuple[int, str]:
    return (-item.size, item.path)


def plan_shards(
    files: Sequence[RepoFile], n_nodes: int, small_file_bytes: int = SMALL_FILE_BYTES
) -> ShardPlan:
    """Longest-processing-time bin packing of the large files across nodes.

    Files under ``small_file_bytes`` are downloaded by every node. Large files
    are placed largest first on the least-loaded rank (lowest rank on ties),
    so the result depends only on the file set, never on input order.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    paths = [f.path for f in files]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate paths in the file list")
    small = tuple(sorted(f.path for f in files if f.size < small_file_bytes))
    loads = [0] * n_nodes
    owners: dict[str, int] = {}
    for item in sorted((f for f in files if f.size >= small_file_bytes), key=_lpt_key):
        rank = min(range(n_nodes), key=loads.__getitem__)
        owners[item.path] = rank
        loads[rank] += item.size
    return ShardPlan(
        n_nodes=n_nodes, owners=owners, small=small, rank_bytes=tuple(loads)
    )


def build_fetch_plan(
    repo_id: str,
    revision: str,
    commit: str,
    files: Sequence[RepoFile],
    nodes: Sequence[str],
    small_file_bytes: int = SMALL_FILE_BYTES,
) -> FetchPlan:
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError(f"nodes must be non-empty and unique: {list(nodes)}")
    shards = plan_shards(files, len(nodes), small_file_bytes)
    planned = tuple(
        PlannedFile(
            path=f.path, size=f.size, sha256=f.sha256, owner=shards.owners.get(f.path)
        )
        for f in sorted(files, key=_repo_path)
    )
    return FetchPlan(
        repo_id=repo_id,
        revision=revision,
        commit=commit,
        nodes=tuple(nodes),
        small_file_bytes=small_file_bytes,
        files=planned,
    )


def build_manifest(plan: FetchPlan, model_dir: str) -> dict[str, Any]:
    """The ``file_gather.py`` manifest; small files are nominally owned by rank 0."""
    return {
        "root": model_dir,
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "sha256": f.sha256,
                "owner": 0 if f.owner is None else f.owner,
            }
            for f in plan.files
        ],
    }


def completion_marker(plan: FetchPlan, written_at: float) -> dict[str, Any]:
    return {
        "repo": plan.repo_id,
        "commit": plan.commit,
        "files": {f.path: f.size for f in plan.files},
        "total_bytes": plan.total_bytes,
        "written_at": written_at,
    }


def marker_matches(marker: Mapping[str, Any] | None, plan: FetchPlan) -> bool:
    """Same repo, commit and file sizes (independent of how it was sharded)."""
    if not isinstance(marker, Mapping):
        return False
    return (
        marker.get("repo") == plan.repo_id
        and marker.get("commit") == plan.commit
        and marker.get("files") == {f.path: f.size for f in plan.files}
    )


def size_problems(plan: FetchPlan, sizes: Mapping[str, int]) -> list[str]:
    """Planned files absent from ``sizes`` (path -> bytes on disk) or wrong-sized."""
    problems = []
    for f in plan.files:
        actual = sizes.get(f.path)
        if actual != f.size:
            found = "missing" if actual is None else f"{actual} bytes"
            problems.append(f"{f.path}: {found}, expected {f.size}")
    return problems


def glob_escape(path: str) -> str:
    """Escape fnmatch metacharacters so an allow pattern matches one exact path."""
    return re.sub(r"([*?\[])", r"[\1]", path)


def download_params(
    plan: FetchPlan,
    model_dir: str,
    files: Sequence[PlannedFile],
    *,
    state_dir: str,
    background: bool,
    rank: int,
    max_workers: int = DOWNLOAD_MAX_WORKERS,
) -> dict[str, Any]:
    """``PARAMS`` of DOWNLOAD_SCRIPT for one node (no secrets in here)."""
    return {
        "repo_id": plan.repo_id,
        "commit": plan.commit,
        "local_dir": model_dir,
        "allow_patterns": [glob_escape(f.path) for f in files],
        "sizes": {f.path: f.size for f in files},
        "max_workers": max_workers,
        "background": background,
        "pid_path": f"{state_dir}/{SHARD_PID}",
        "log_path": f"{state_dir}/{SHARD_LOG}",
        "status_path": f"{state_dir}/{SHARD_STATUS}",
        "rank": rank,
    }


def render_download_script(params: Mapping[str, Any]) -> str:
    return (
        f"PARAMS = {json.dumps(json.dumps(params, sort_keys=True))}\n{DOWNLOAD_SCRIPT}"
    )


def download_stdin(token: str | None, script: str) -> str:
    """stdin for BOOTSTRAP: the token line (may be empty), then the script."""
    token = token or ""
    if "\n" in token or "\r" in token:
        raise ValueError("token must be a single line")
    return f"{token}\n{script}"


def parse_download_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise ValueError(
        f"no {RESULT_PREFIX.strip()} line in download output: {stdout[-500:]!r}"
    )


def parse_started(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        if line.startswith(STARTED_PREFIX):
            return int(line[len(STARTED_PREFIX) :])
    raise ValueError(f"background download did not report a pid: {stdout[-500:]!r}")


def alive_test(pid_path: str) -> str:
    """Shell condition: the pid file names a live download process of ours."""
    pid = shlex.quote(pid_path)
    cmdline = f'"/proc/$(cat {pid})/cmdline"'
    return f"[ -f {pid} ] && grep -qa {BOOTSTRAP_TAG} {cmdline} 2>/dev/null"


def probe_command(pid_path: str, status_path: str) -> str:
    """One line: CB_STATUS <json> | CB_RUNNING <pid> | CB_ABSENT."""
    status = shlex.quote(status_path)
    pid = shlex.quote(pid_path)
    return (
        f"if [ -f {status} ]; then printf 'CB_STATUS '; tr -d '\\n' < {status}; echo; "
        f"elif {alive_test(pid_path)}; "
        f'then echo "CB_RUNNING $(cat {pid})"; else echo CB_ABSENT; fi'
    )


def parse_probe(stdout: str) -> ProbeState:
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("CB_STATUS "):
            return ProbeState("done", status=json.loads(line[len("CB_STATUS ") :]))
        if line.startswith("CB_RUNNING "):
            return ProbeState("running", pid=int(line[len("CB_RUNNING ") :]))
        if line.strip() == "CB_ABSENT":
            return ProbeState("absent")
    raise ValueError(f"unexpected download probe output: {stdout[-300:]!r}")


def completion_probe_command(model_dir: str) -> str:
    """Marker text, a separator line, then ``<size>\\t<relpath>`` per file."""
    root = shlex.quote(model_dir)
    marker = shlex.quote(f"{model_dir}/{MARKER_NAME}")
    return (
        f"cat {marker} 2>/dev/null; echo; echo {FILES_SEPARATOR}; "
        f"find {root} -type f -printf '%s\\t%P\\n' 2>/dev/null; true"
    )


def parse_completion_probe(stdout: str) -> tuple[dict[str, Any] | None, dict[str, int]]:
    head, sep, tail = stdout.partition(f"\n{FILES_SEPARATOR}\n")
    if not sep:
        raise ValueError("completion probe output has no file listing")
    marker: dict[str, Any] | None = None
    if head.strip():
        try:
            parsed = json.loads(head)
        except json.JSONDecodeError:
            parsed = None  # a torn or foreign marker means "not complete"
        marker = parsed if isinstance(parsed, dict) else None
    sizes = {}
    for line in tail.splitlines():
        size, tab, rel = line.partition("\t")
        if tab and size.isdigit():
            sizes[rel] = int(size)
    return marker, sizes


def parse_ipv4(stdout: str) -> str:
    """First address in ``ip -4 -o addr show`` output."""
    match = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)/", stdout)
    if not match:
        raise ValueError(f"no IPv4 address in {stdout!r}")
    return match.group(1)


def parse_rank_report(raw: Mapping[str, Any], node: str) -> RankReport:
    hashes: dict[str, str | None] = {}
    failed: dict[str, str] = {}
    for item in raw["files"]:
        hashes[item["path"]] = item.get("sha256")
        if not item["ok"]:
            failed[item["path"]] = str(item.get("reason"))
    return RankReport(
        rank=int(raw["rank"]),
        node=node,
        hostname=str(raw["hostname"]),
        bytes_received=int(raw["bytes_received"]),
        bytes_sent=int(raw["bytes_sent"]),
        bytes_discarded=int(raw["bytes_discarded"]),
        bytes_broadcast=int(raw["bytes_broadcast"]),
        seconds=float(raw["seconds"]),
        verify_seconds=float(raw["verify_seconds"]),
        gbs=None if raw.get("gbs") is None else float(raw["gbs"]),
        mismatches=tuple(raw["mismatches"]),
        missing=tuple(raw["missing"]),
        hashes=hashes,
        failed=failed,
    )


def check_reports(plan: FetchPlan, reports: Mapping[int, RankReport]) -> list[str]:
    """Every rank verified every file, and all ranks agree on every sha256."""
    problems = []
    for rank, node in enumerate(plan.nodes):
        report = reports.get(rank)
        if report is None:
            problems.append(f"{node}: no gather report")
            continue
        problems += [
            f"{node}: {path}: {reason}"
            for path, reason in sorted(report.failed.items())
        ]
    for f in plan.files:
        digests = {
            r.hashes.get(f.path) for r in reports.values() if f.path not in r.failed
        }
        if None in digests or len(digests) > 1:
            seen = sorted(map(str, digests))
            problems.append(f"{f.path}: ranks disagree or did not hash it: {seen}")
        elif f.sha256 is not None and digests and digests != {f.sha256}:
            problems.append(f"{f.path}: sha256 differs from the hub")
    return problems


def _mbps(nbytes: int, seconds: float) -> float | None:
    return nbytes / seconds / 1e6 if seconds > 0 else None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class _MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class ModelFetcher:
    """Downloads a model to the same local directory on every node.

    ``python`` is the node-local venv interpreter (huggingface_hub[hf_xet]);
    ``workdir`` the node-local harness directory, where per-plan state lives in
    ``downloads/<plan_id>/`` (manifest, rank reports, shard pid/log/status).
    ``control_ips`` maps node -> control address; when a node is missing it is
    read from ``ip -4 addr show <control_iface>`` on that node.
    """

    def __init__(
        self,
        cluster: Cluster,
        python: str,
        workdir: str,
        *,
        hf_cache: str | None = None,
        control_ips: Mapping[str, str] | None = None,
        control_iface: str = "wg0",
        clock: Clock | None = None,
        port_base: int = DEFAULT_PORT_BASE,
        small_file_bytes: int = SMALL_FILE_BYTES,
        max_workers: int = DOWNLOAD_MAX_WORKERS,
        labels: Mapping[str, str] | None = None,
    ):
        self._cluster = cluster
        self._python = python
        self._workdir = workdir.rstrip("/")
        self._hf_cache = hf_cache or f"{self._workdir}/hf-cache"
        self._control_ips = dict(control_ips or {})
        self._iface = control_iface
        self._clock: Clock = clock or _MonotonicClock()
        self._port_base = port_base
        self._small_file_bytes = small_file_bytes
        self._max_workers = max_workers
        self._labels = {LABEL_ROLE: GATHER_ROLE, **(labels or {})}

    # --------------------------------------------------------------- paths

    def state_dir(self, plan: FetchPlan) -> str:
        return f"{self._workdir}/downloads/{plan.plan_id}"

    def _manifest_path(self, plan: FetchPlan) -> str:
        return f"{self.state_dir(plan)}/manifest.json"

    def _report_path(self, plan: FetchPlan, rank: int | str) -> str:
        return f"{self.state_dir(plan)}/report-{rank}.json"

    @staticmethod
    def _check_nodes(plan: FetchPlan, nodes: Sequence[str], *, exact: bool) -> None:
        unknown = [node for node in nodes if node not in plan.nodes]
        if unknown:
            raise ValueError(f"nodes {unknown} are not in the plan {plan.nodes}")
        if exact and set(nodes) != set(plan.nodes):
            raise ValueError(
                f"this step needs every planned node {plan.nodes}, got {list(nodes)}"
            )

    def _control_ip(self, node: str) -> str:
        if node not in self._control_ips:
            res = self._cluster.run(
                node, f"ip -4 -o addr show dev {shlex.quote(self._iface)}"
            )
            self._control_ips[node] = parse_ipv4(res.stdout)
        return self._control_ips[node]

    # ------------------------------------------------------------ download

    def _download_command(self, model_dir: str, state_dir: str) -> str:
        env = {
            "HF_HOME": self._hf_cache,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_ETAG_TIMEOUT": "60",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        }
        assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
        unset = " ".join(f"-u {name}" for name in TOKEN_ENV_VARS)
        dirs = " ".join(shlex.quote(d) for d in (state_dir, model_dir, self._hf_cache))
        return (
            f"mkdir -p {dirs} && env {unset} {assignments} "
            f"{shlex.quote(self._python)} -c {shlex.quote(BOOTSTRAP)}"
        )

    def _params(
        self,
        plan: FetchPlan,
        model_dir: str,
        files: Sequence[PlannedFile],
        *,
        background: bool,
        rank: int,
    ) -> dict[str, Any]:
        return download_params(
            plan,
            model_dir,
            files,
            state_dir=self.state_dir(plan),
            background=background,
            rank=rank,
            max_workers=self._max_workers,
        )

    def _handle(
        self, plan: FetchPlan, node: str, pid: int | None, reused: bool
    ) -> DownloadHandle:
        rank = plan.rank_of(node)
        files = plan.files_for(rank)
        state = self.state_dir(plan)
        return DownloadHandle(
            node=node,
            rank=rank,
            pid=pid,
            pid_path=f"{state}/{SHARD_PID}",
            log_path=f"{state}/{SHARD_LOG}",
            status_path=f"{state}/{SHARD_STATUS}",
            files=len(files),
            bytes=sum(f.size for f in files),
            reused=reused,
        )

    def _probe(self, handle: DownloadHandle) -> ProbeState:
        res = self._cluster.run(
            handle.node,
            probe_command(handle.pid_path, handle.status_path),
            timeout=PROBE_TIMEOUT_S,
        )
        return parse_probe(res.stdout)

    def _log_tail(self, node: str, log_path: str) -> str:
        res = self._cluster.run(
            node, f"tail -n {LOG_TAIL_LINES} {shlex.quote(log_path)} 2>&1", check=False
        )
        return res.stdout

    def fetch_small_files(
        self, plan: FetchPlan, model_dir: str, nodes: Sequence[str], token: str | None
    ) -> dict[str, NodeDownloadResult]:
        """Blocking: every node downloads the files under ``small_file_bytes``."""
        self._check_nodes(plan, nodes, exact=False)
        files = plan.small_files()

        def one(node: str) -> NodeDownloadResult:
            rank = plan.rank_of(node)
            script = render_download_script(
                self._params(plan, model_dir, files, background=False, rank=rank)
            )
            res = self._cluster.run(
                node,
                self._download_command(model_dir, self.state_dir(plan)),
                input_text=download_stdin(token, script),
                timeout=SMALL_TIMEOUT_S,
            )
            result = parse_download_result(res.stdout)
            seconds = float(result["seconds"])
            return NodeDownloadResult(
                node=node,
                rank=rank,
                phase="small",
                files=int(result["files"]),
                bytes=int(result["bytes"]),
                seconds=seconds,
                mbps=_mbps(int(result["bytes"]), seconds),
            )

        return self._cluster.fanout(list(nodes), one)

    def start_downloads(
        self, plan: FetchPlan, model_dir: str, nodes: Sequence[str], token: str | None
    ) -> dict[str, DownloadHandle]:
        """Launch each node's shard download in the background; returns at once.

        Idempotent: a node whose download for this plan is still running, or
        already finished successfully, is attached to instead of restarted.
        """
        self._check_nodes(plan, nodes, exact=False)

        def one(node: str) -> DownloadHandle:
            handle = self._handle(plan, node, None, False)
            state = self._probe(handle)
            if state.state == "running":
                logger.info(
                    "[%s] attaching to running shard download pid %s", node, state.pid
                )
                return handle.model_copy(update={"pid": state.pid, "reused": True})
            if (
                state.state == "done"
                and state.status is not None
                and state.status.get("rc") == 0
                and self._shard_intact(plan, node, model_dir, handle.rank)
            ):
                logger.info("[%s] shard download already finished", node)
                return handle.model_copy(update={"reused": True})
            stale = (shlex.quote(handle.status_path), shlex.quote(handle.pid_path))
            self._cluster.run(node, "rm -f " + " ".join(stale))
            files = plan.files_for(handle.rank)
            script = render_download_script(
                self._params(plan, model_dir, files, background=True, rank=handle.rank)
            )
            res = self._cluster.run(
                node,
                self._download_command(model_dir, self.state_dir(plan)),
                input_text=download_stdin(token, script),
                timeout=START_TIMEOUT_S,
            )
            pid = parse_started(res.stdout)
            logger.info(
                "[%s] shard download started: %d files, %d bytes, pid %d",
                node,
                handle.files,
                handle.bytes,
                pid,
            )
            return handle.model_copy(update={"pid": pid})

        return self._cluster.fanout(list(nodes), one)

    def wait_downloads(
        self,
        handles: Mapping[str, DownloadHandle],
        timeout_s: float,
        poll: float = 15.0,
    ) -> dict[str, NodeDownloadResult]:
        """Poll every node's status file until all downloads finish.

        Raises ``ModelFetchError`` (with the log tail) when a download fails,
        its process vanishes without a status file, or ``timeout_s`` passes.
        """
        deadline = self._clock.now() + timeout_s
        pending = dict(handles)
        results: dict[str, NodeDownloadResult] = {}
        while pending:
            finished = []
            for node, handle in pending.items():
                state = self._probe(handle)
                if state.state == "running":
                    continue
                status = state.status or {}
                if state.state == "done" and status.get("rc") == 0:
                    result = status["result"]
                    seconds = float(result["seconds"])
                    results[node] = NodeDownloadResult(
                        node=node,
                        rank=handle.rank,
                        phase="shards",
                        files=int(result["files"]),
                        bytes=int(result["bytes"]),
                        seconds=seconds,
                        mbps=_mbps(int(result["bytes"]), seconds),
                    )
                    finished.append(node)
                    continue
                reason = (
                    status.get("error")
                    or "the process exited without writing a status file"
                )
                tail = self._log_tail(node, handle.log_path)
                raise ModelFetchError(
                    f"[{node}] shard download failed: {reason}\n{tail}"
                )
            for node in finished:
                del pending[node]
            if not pending:
                break
            if self._clock.now() >= deadline:
                raise ModelFetchError(
                    f"shard downloads still running after {timeout_s:.0f} s "
                    f"on {sorted(pending)}"
                )
            self._clock.sleep(poll)
        return results

    def stop_downloads(self, handles: Mapping[str, DownloadHandle]) -> None:
        """Terminate background downloads that are still running (best effort)."""

        def one(node: str) -> None:
            path = handles[node].pid_path
            pid = shlex.quote(path)
            command = f'if {alive_test(path)}; then kill "$(cat {pid})"; fi'
            self._cluster.run(node, command, check=False)

        self._cluster.fanout(list(handles), one)

    # -------------------------------------------------------------- gather

    def gather_jobs(
        self, plan: FetchPlan, model_dir: str, image: str, *, chunk_mib: int = CHUNK_MIB
    ) -> dict[str, list[str]]:
        """node -> ``docker run`` argv of the gather (pure apart from IP lookup)."""
        state = self.state_dir(plan)
        master = self._control_ip(plan.nodes[0])
        port = rendezvous_port(self._port_base, f"gather|{plan.plan_id}")
        args = [
            "--manifest", self._manifest_path(plan),
            "--report", self._report_path(plan, "{rank}"),
            "--chunk-mib", str(chunk_mib),
        ]  # fmt: skip
        env = {
            "NCCL_SOCKET_IFNAME": self._iface,
            "NCCL_NET": "IB",
            "NCCL_DEBUG": "WARN",
        }
        mounts = (
            Mount(f"{self._workdir}/src", CONTAINER_SRC, read_only=True),
            Mount(model_dir, model_dir),
            Mount(state, state),
        )
        jobs = {}
        for rank, node in enumerate(plan.nodes):
            command = build_torchrun_command(
                GATHER_SCRIPT,
                args,
                nproc_per_node=1,
                nnodes=len(plan.nodes),
                node_rank=rank,
                master_addr=master,
                master_port=port,
            )
            jobs[node] = docker_run_argv(
                name=GATHER_CONTAINER,
                image=image,
                command=command,
                gpus=[0],
                env=env,
                mounts=mounts,
                labels=self._labels,
                detach=False,
                remove=True,
            )
        return jobs

    def _read_reports(self, plan: FetchPlan) -> dict[int, RankReport]:
        reports = {}
        with tempfile.TemporaryDirectory(prefix="cb-gather-") as tmp:
            for rank, node in enumerate(plan.nodes):
                local = Path(tmp) / f"report-{rank}.json"
                try:
                    self._cluster.get_file(node, self._report_path(plan, rank), local)
                except (FileNotFoundError, CommandError):
                    continue
                reports[rank] = parse_rank_report(
                    json.loads(local.read_text(encoding="utf-8")), node
                )
        return reports

    def gather(
        self,
        plan: FetchPlan,
        model_dir: str,
        nodes: Sequence[str],
        image: str,
        *,
        chunk_mib: int = CHUNK_MIB,
        timeout_s: float = GATHER_TIMEOUT_S,
    ) -> GatherResult:
        """Broadcast missing files over the fabric and verify every file on every node.

        Returns the per-rank results (mismatches included); raises when a rank
        produced no report.
        """
        self._check_nodes(plan, nodes, exact=True)
        state = shlex.quote(self.state_dir(plan))
        manifest = (
            json.dumps(build_manifest(plan, model_dir), indent=1, sort_keys=True) + "\n"
        )
        jobs = self.gather_jobs(plan, model_dir, image, chunk_mib=chunk_mib)

        def prepare(node: str) -> None:
            self._cluster.run(
                node,
                f"mkdir -p {state} && rm -f {state}/report-*.json && "
                f"{{ docker rm -f {GATHER_CONTAINER} >/dev/null 2>&1 || true; }}",
            )
            self._cluster.put_text(node, self._manifest_path(plan), manifest)

        def launch(node: str) -> tuple[int, str]:
            res = self._cluster.run(
                node, shlex.join(jobs[node]), timeout=timeout_s, check=False
            )
            if not res.ok:  # a timed-out client leaves the container running
                cleanup = f"docker rm -f {GATHER_CONTAINER} >/dev/null 2>&1 || true"
                self._cluster.run(node, cleanup, check=False)
            return res.rc, (res.stdout + res.stderr)[-3000:]

        self._cluster.fanout(list(plan.nodes), prepare)
        started = self._clock.now()
        outcomes = self._cluster.fanout(list(plan.nodes), launch)
        wall = self._clock.now() - started
        reports = self._read_reports(plan)
        absent = [node for rank, node in enumerate(plan.nodes) if rank not in reports]
        if absent:
            detail = "\n".join(
                f"[{node}] exit {outcomes[node][0]}: {outcomes[node][1]}"
                for node in absent
            )
            raise ModelFetchError(
                f"model gather produced no report on {absent}\n{detail}"
            )
        ranks = tuple(reports[rank] for rank in range(len(plan.nodes)))
        seconds = max(r.seconds for r in ranks)
        broadcast = max(r.bytes_broadcast for r in ranks)
        return GatherResult(
            ranks=ranks,
            seconds=seconds,
            wall_seconds=wall,
            bytes_broadcast=broadcast,
            gbs=broadcast / seconds / 1e9 if broadcast and seconds > 0 else None,
            exit_codes={node: outcomes[node][0] for node in plan.nodes},
            mismatches=tuple(f"{r.node}:{p}" for r in ranks for p in r.mismatches),
            missing=tuple(f"{r.node}:{p}" for r in ranks for p in r.missing),
        )

    # -------------------------------------------------------- verification

    def _shard_intact(
        self, plan: FetchPlan, node: str, model_dir: str, rank: int
    ) -> bool:
        """A finished download is reusable only if its files are still whole.

        A shard renamed to ``.cb_corrupt`` by the gather (or deleted) must be
        downloaded again; HF resumes cheaply, so re-running is safe.
        """
        _, sizes = self._completion_state(node, model_dir)
        wanted = {f.path: f.size for f in plan.files_for(rank)}
        intact = all(sizes.get(path) == size for path, size in wanted.items())
        if not intact:
            logger.info(
                "[%s] finished shard download is incomplete; restarting it", node
            )
        return intact

    def _completion_state(
        self, node: str, model_dir: str
    ) -> tuple[dict[str, Any] | None, dict[str, int]]:
        res = self._cluster.run(
            node, completion_probe_command(model_dir), timeout=PROBE_TIMEOUT_S
        )
        return parse_completion_probe(res.stdout)

    def verify_and_mark(
        self, plan: FetchPlan, model_dir: str, nodes: Sequence[str]
    ) -> VerifyResult:
        """Check the gather reports and on-disk sizes, then mark every node."""
        self._check_nodes(plan, nodes, exact=True)
        problems = check_reports(plan, self._read_reports(plan))
        states = self._cluster.fanout(
            list(plan.nodes), lambda node: self._completion_state(node, model_dir)
        )
        for node in plan.nodes:
            problems += [f"{node}: {p}" for p in size_problems(plan, states[node][1])]
        if problems:
            shown = "\n".join(problems[:30])
            raise ModelFetchError(
                f"model verification failed ({len(problems)} problems):\n{shown}",
                problems=problems,
            )
        marker_path = f"{model_dir}/{MARKER_NAME}"
        marker = (
            json.dumps(completion_marker(plan, time.time()), indent=1, sort_keys=True)
            + "\n"
        )
        self._cluster.fanout(
            list(plan.nodes),
            lambda node: self._cluster.put_text(node, marker_path, marker),
        )
        return VerifyResult(
            nodes=plan.nodes,
            files=len(plan.files),
            total_bytes=plan.total_bytes,
            marker=marker_path,
        )

    def is_complete(
        self, plan: FetchPlan, model_dir: str, nodes: Sequence[str]
    ) -> bool:
        """True when every node has a matching marker and every planned file on disk."""
        self._check_nodes(plan, nodes, exact=False)
        states = self._cluster.fanout(
            list(nodes), lambda node: self._completion_state(node, model_dir)
        )
        return all(
            marker_matches(marker, plan) and not size_problems(plan, sizes)
            for marker, sizes in states.values()
        )

    # ------------------------------------------------------------- planning

    def plan(
        self,
        repo_id: str,
        revision: str,
        token: str | None,
        *,
        nodes: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] = (),
    ) -> FetchPlan:
        """Resolve the commit, list files and shard them (default: all nodes)."""
        commit, files = list_repo_files(repo_id, revision, token)
        kept = [
            f
            for f in files
            if not any(
                fnmatch.fnmatchcase(f.path, pattern) for pattern in ignore_patterns
            )
        ]
        targets = tuple(nodes) if nodes is not None else tuple(self._cluster.nodes)
        return build_fetch_plan(
            repo_id, revision, commit, kept, targets, self._small_file_bytes
        )

    def fetch(
        self,
        repo_id: str,
        revision: str,
        model_dir: str,
        nodes: Sequence[str],
        token: str | None,
        image: str,
        *,
        ignore_patterns: Sequence[str] = (),
        download_timeout_s: float = 6 * 3600.0,
        poll: float = 15.0,
    ) -> DownloadReport:
        """Every step in order; skips everything when the model is already complete."""
        plan = self.plan(
            repo_id, revision, token, nodes=nodes, ignore_patterns=ignore_patterns
        )
        base = {
            "repo": plan.repo_id,
            "commit": plan.commit,
            "model_dir": model_dir,
            "total_bytes": plan.total_bytes,
            "file_count": len(plan.files),
        }
        if self.is_complete(plan, model_dir, plan.nodes):
            return DownloadReport(**base, skipped=True, verified=True)
        small = self.fetch_small_files(plan, model_dir, plan.nodes, token)
        handles = self.start_downloads(plan, model_dir, plan.nodes, token)
        try:
            shards = self.wait_downloads(handles, download_timeout_s, poll)
        except ModelFetchError:
            self.stop_downloads(handles)
            raise
        gathered = self.gather(plan, model_dir, plan.nodes, image)
        report = DownloadReport(
            **base,
            skipped=False,
            small=small,
            shards=shards,
            gather_seconds=gathered.seconds,
            gather_gbs=gathered.gbs,
            verified=False,
            mismatches=gathered.mismatches + gathered.missing,
        )
        try:
            self.verify_and_mark(plan, model_dir, plan.nodes)
        except ModelFetchError as exc:
            raise ModelFetchError(
                str(exc), problems=exc.problems, report=report
            ) from exc
        return report.model_copy(update={"verified": True})
