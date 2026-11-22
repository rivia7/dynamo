# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Nsight Systems diagnostic window: image, trigger, reports, stats.

Once per topology x GPU generation, outside the official measurements, the
leader is redeployed with its workers under ``nsys profile``:
``scripts/launch_node.sh`` wraps the worker when its env carries
``CB_NSYS_OUTPUT`` (rendered by ``nsys_env``). Under load, every instance
leader is asked to profile a few forward steps
(``POST /engine/control/start_profile`` on its Dynamo system port; Dynamo
1.5.1 forwards the body to SGLang 0.5.18's ``ProfileReq(**body)``, SGLang
calls ``cudaProfilerStart`` and stops by itself after ``num_steps``). Each
worker container writes one ``.nsys-rep``; ``nsys stats`` in a container of
the diagnostic image turns it into CSVs that ``domain/kernels.py`` parses.

Pure builders render every command (``--dry-run`` prints them); each remote
command is bounded by coreutils ``timeout``. ``NsysDiagnostics`` runs them
through an injected ``ports.Cluster`` and is fail-soft: a failing node, tool
or file becomes a note in the returned model, never an exception.

Coverage: SGLang calls ``cudaProfilerStart``/``Stop`` only in the scheduler
whose GPU is the container's first (``gpu_id == base_gpu_id``), once per node
of an instance. Whether nsys then traces the other schedulers of the same
session is not documented; ``cuda_kern_exec_sum`` is requested so every
breakdown says which GPUs actually have kernels in the capture range.
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.lium_cluster_bench.adapters.docker import (
    Mount,
    docker_run_argv,
    sanitize_label,
)
from benchmarks.lium_cluster_bench.domain.kernels import (
    DiagImage,
    KernelReport,
    NsysInfo,
    NsysLogState,
    ProfileTrigger,
    ReportFile,
    breakdown,
    parse_kern_exec_sum_csv,
    parse_kern_sum_csv,
    parse_mem_time_sum_csv,
    parse_nsys_version,
    stats_problems,
    synthetic_kern_exec_sum_csv,
    synthetic_kern_sum_csv,
    synthetic_mem_time_sum_csv,
)
from benchmarks.lium_cluster_bench.domain.schema import ContainerSpec
from benchmarks.lium_cluster_bench.fsutil import atomic_write_text
from benchmarks.lium_cluster_bench.ports import Clock, Cluster, CommandError, CommandResult

logger = logging.getLogger(__name__)

# Worker env read by scripts/launch_node.sh.
ENV_OUTPUT = "CB_NSYS_OUTPUT"  # report path prefix inside the container
ENV_FLAGS = "CB_NSYS_FLAGS"  # whitespace-separated replacement of the defaults
ENV_BIN = "CB_NSYS_BIN"  # nsys binary when not on PATH

# Keep in sync with the default list in scripts/launch_node.sh (tested).
# --trace-fork-before-exec, --cuda-graph-trace, --capture-range and
# --capture-range-end are the verified recipe (SGLang docs, v0.5.18); node-level
# CUDA-graph tracing exposes the kernels inside decode graphs at a "possibly
# significant" runtime cost, so throughput in the window is never official.
# --trace / --sample / --cpuctxsw drop OS-runtime tracing and CPU sampling
# (overhead; SGLang workers exhaust CPU-sampling resources anyway), and
# --force-overwrite keeps a restarted worker from failing on its old report.
NSYS_DEFAULT_FLAGS: tuple[str, ...] = (
    "--trace=cuda,nvtx",
    "--sample=none",
    "--cpuctxsw=none",
    "--trace-fork-before-exec=true",
    "--cuda-graph-trace=node",
    "--capture-range=cudaProfilerApi",
    "--capture-range-end=stop",
    "--force-overwrite=true",
)
# --trace-fork-before-exec relies on undefined behavior and may crash or
# deadlock a worker: retry a window whose worker died under nsys with this.
NSYS_NO_FORK_FLAGS: tuple[str, ...] = tuple(
    "--trace-fork-before-exec=false" if f.startswith("--trace-fork-before-exec") else f
    for f in NSYS_DEFAULT_FLAGS
)

CONTAINER_EXP = "/bench/experiments"  # == engine_sglang.CONTAINER_EXP
DIAG_SUBDIR = "diagnostics"
DEFAULT_WINDOW = "w1"  # the harness opens one capture window per experiment
REPORT_SUFFIX = ".nsys-rep"
STATS_REPORTS: tuple[str, ...] = (
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_kern_exec_sum",
)
START_PROFILE_ROUTE = "/engine/control/start_profile"
DEFAULT_STEPS = 10
WORKER_ROLES = ("agg", "prefill", "decode")

# Derived image: SGLang v0.5.18 docker/Dockerfile, framework stage.
NSYS_PACKAGE = "nsight-systems-cli"
DEVTOOLS_DISTRO = "ubuntu2004"  # as in that recipe, whatever the base Ubuntu
DEVTOOLS_REPOS = "http://developer.download.nvidia.com/devtools/repos"
CUDA_KEY_BASE = "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804"
CUDA_KEY_FILE = "7fa2af80.pub"
IMAGE_USER = "dynamo"  # sglang-runtime ends with ``USER dynamo``
DIAG_REPOSITORY = "cb-diag"
LABEL_DIAG = "cb.diag"

# Inner (coreutils ``timeout``) limits; Cluster.run gets OUTER_MARGIN_S more.
PROBE_TIMEOUT_S = 180.0  # container start + nsys --version (~5-20 s)
INSPECT_TIMEOUT_S = 60.0
BUILD_TIMEOUT_S = 2400.0  # apt update + ~0.3-0.5 GB download (~2-6 min)
TRIGGER_TIMEOUT_S = 30.0  # returns once the schedulers accepted the request
STAT_TIMEOUT_S = 60.0
STATS_TIMEOUT_S = 1800.0  # SQLite export + reports (~1-10 min per report)
CAT_TIMEOUT_S = 120.0
CLEANUP_TIMEOUT_S = 60.0
OUTER_MARGIN_S = 60.0
REPORT_WAIT_S = 900.0  # capture of a few steps + report generation
REPORT_POLL_S = 15.0
ERROR_TAIL_CHARS = 2000

PROBE_MARKER = "CB_NSYS_PATH="
PROBE_SCRIPT = (
    'p="$(command -v nsys 2>/dev/null || true)"; '
    'if [ -z "$p" ]; then for c in /opt/nvidia/nsight-systems*/*/bin/nsys '
    "/opt/nvidia/nsight-systems*/bin/nsys /usr/local/cuda/bin/nsys; do "
    'if [ -x "$c" ]; then p="$c"; break; fi; done; fi; '
    f'echo "{PROBE_MARKER}$p"; '
    'if [ -n "$p" ]; then "$p" --version 2>&1 | head -n 3; fi'
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DIGEST = re.compile(r"@sha256:([0-9a-f]{64})$")
_IMAGE_ID = re.compile(r"sha256:([0-9a-f]{64})")
_PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+:=~-]*$")
_NETWORK_ERROR = re.compile(r"(?i)network|entitlement")
_ROLE_IN_NAME = re.compile(r"-(agg|prefill|decode)\d*-r\d+")


# --------------------------------------------------------------------------
# Paths and worker env
# --------------------------------------------------------------------------


def _check_name(kind: str, value: str) -> str:
    if not _SAFE_NAME.match(value) or ".." in value:
        raise ValueError(f"unsafe {kind} {value!r}")
    return value


def host_experiments(workdir: str) -> str:
    """Host directory mounted at ``/bench/experiments`` in every container."""
    return f"{workdir.rstrip('/')}/experiments"


def window_dir(exp_id: str, window: str, *, root: str = CONTAINER_EXP) -> str:
    """Directory of one diagnostic window below an experiments root."""
    exp = _check_name("exp_id", exp_id)
    name = _check_name("window", window)
    return f"{root.rstrip('/')}/{exp}/{DIAG_SUBDIR}/{name}"


def output_prefix(exp_id: str, window: str, container: str) -> str:
    """``CB_NSYS_OUTPUT`` of one worker container (container path, no suffix)."""
    return f"{window_dir(exp_id, window)}/{_check_name('container', container)}"


def report_path(workdir: str, exp_id: str, window: str, container: str) -> str:
    """Host path of the ``.nsys-rep`` that container writes."""
    root = host_experiments(workdir)
    name = _check_name("container", container)
    return f"{window_dir(exp_id, window, root=root)}/{name}{REPORT_SUFFIX}"


def to_container_path(host_path: str, workdir: str) -> str | None:
    """The same file as seen inside a container (None: outside the mount)."""
    root = host_experiments(workdir) + "/"
    if not host_path.startswith(root) or ".." in host_path:
        return None
    return f"{CONTAINER_EXP}/{host_path[len(root):]}"


def flags_value(flags: Sequence[str]) -> str:
    """``CB_NSYS_FLAGS`` for a flag list (the hook appends ``-o`` itself)."""
    if not flags:
        raise ValueError("empty nsys flag list (the hook would use its defaults)")
    for flag in flags:
        if not flag or any(c.isspace() for c in flag):
            raise ValueError(f"nsys flag {flag!r} is empty or contains whitespace")
        if flag in ("-o", "--output") or flag.startswith(("-o=", "--output=")):
            raise ValueError("pass the report path as CB_NSYS_OUTPUT, not -o")
    return " ".join(flags)


def nsys_env(
    exp_id: str,
    window: str,
    container: str,
    *,
    flags: Sequence[str] | None = None,
    nsys_bin: str | None = None,
) -> dict[str, str]:
    """Env additions that make ``launch_node.sh worker`` run under nsys."""
    env = {ENV_OUTPUT: output_prefix(exp_id, window, container)}
    if flags is not None:
        env[ENV_FLAGS] = flags_value(flags)
    if nsys_bin is not None:
        if not nsys_bin.startswith("/") or any(c.isspace() for c in nsys_bin):
            raise ValueError(f"nsys binary must be an absolute path: {nsys_bin!r}")
        env[ENV_BIN] = nsys_bin
    return env


def profile_targets(containers: Sequence[ContainerSpec]) -> list[ContainerSpec]:
    """Containers that receive start_profile: rank 0 of every worker instance.

    Non-leader ranks of a multi-node instance never register ``/engine``
    routes (dynamo.sglang returns early for ``node_rank >= 1``); SGLang
    broadcasts the request to every scheduler of the instance, so each
    node's first-GPU scheduler still starts its own capture.
    """
    return [
        c
        for c in containers
        if c.role in WORKER_ROLES and c.rank == 0 and c.system_port is not None
    ]


def expected_reports(
    containers: Sequence[ContainerSpec], *, workdir: str, exp_id: str, window: str
) -> dict[str, list[tuple[str, str, int]]]:
    """node -> [(container, host report path, GPUs)] for every worker container."""
    out: dict[str, list[tuple[str, str, int]]] = {}
    for spec in containers:
        if spec.role not in WORKER_ROLES:
            continue
        path = report_path(workdir, exp_id, window, spec.name)
        out.setdefault(spec.node, []).append((spec.name, path, len(spec.gpus)))
    return out


# --------------------------------------------------------------------------
# Command builders
# --------------------------------------------------------------------------


def bounded(command: str, timeout_s: float, kill_after_s: float = 10.0) -> str:
    """``command`` (one simple command) under coreutils ``timeout``."""
    return f"timeout -k {kill_after_s:g} {timeout_s:g} {command}"


def cleanup_command(name: str) -> str:
    """Remove a leftover helper container (a timed-out client leaves it)."""
    return (
        bounded(shlex.join(["docker", "rm", "-f", name]), CLEANUP_TIMEOUT_S)
        + " >/dev/null 2>&1 || true"
    )


def probe_argv(
    image: str, *, name: str, labels: Mapping[str, str] | None = None
) -> list[str]:
    """Throw-away container that prints ``CB_NSYS_PATH=...`` and the version."""
    argv = ["docker", "run", "--rm", "--name", name, "--network", "none"]
    for key, value in sorted((labels or {}).items()):
        argv += ["--label", f"{key}={value}"]
    return argv + ["--entrypoint", "bash", image, "-lc", PROBE_SCRIPT]


def probe_command(
    image: str,
    *,
    name: str,
    labels: Mapping[str, str] | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> str:
    return bounded(shlex.join(probe_argv(image, name=name, labels=labels)), timeout_s)


def image_digest(image: str) -> str | None:
    """The sha256 hex of a ``repo:tag@sha256:...`` reference."""
    match = _DIGEST.search(image.strip())
    return match.group(1) if match else None


def image_id_command(image: str) -> str:
    """Prints the local image id (``sha256:...``); prints nothing when absent."""
    argv = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    return bounded(shlex.join(argv), INSPECT_TIMEOUT_S)


def diag_dockerfile(
    base_image: str,
    *,
    package: str = NSYS_PACKAGE,
    distro: str = DEVTOOLS_DISTRO,
    user: str = IMAGE_USER,
) -> str:
    """Dockerfile of the diagnostic image (fed to ``docker build -`` on stdin).

    Same steps as SGLang's framework stage: gnupg, NVIDIA's devtools apt
    repository for the machine's architecture (amd64 / arm64), the CUDA repo
    signing key through ``apt-key`` (deprecated, still present on Ubuntu
    24.04), then ``nsight-systems-cli``. The arch-specific key URL falls back
    to the x86_64 one (same key). The build fails unless ``nsys --version``
    runs.
    """
    if not base_image.strip() or any(c.isspace() for c in base_image):
        raise ValueError(f"invalid base image {base_image!r}")
    if not _PACKAGE.match(package):
        raise ValueError(f"invalid package spec {package!r}")
    if not re.fullmatch(r"[a-z0-9]+", distro):
        raise ValueError(f"invalid devtools distro {distro!r}")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise ValueError(f"invalid user {user!r}")
    key = f"{CUDA_KEY_BASE}/$keyarch/{CUDA_KEY_FILE}"
    fallback = f"{CUDA_KEY_BASE}/x86_64/{CUDA_KEY_FILE}"
    lines = [
        "# cluster_bench diagnostic image: the pinned serving image plus the",
        "# Nsight Systems CLI, installed as in SGLang v0.5.18 docker/Dockerfile",
        "# (framework stage) for amd64 and arm64.",
        f"FROM {base_image}",
        "USER root",
        "ARG DEBIAN_FRONTEND=noninteractive",
        "RUN set -eux; \\",
        "    apt-get update -y; \\",
        "    apt-get install -y --no-install-recommends gnupg; \\",
        '    if [ "$(uname -m)" = "aarch64" ]; then repoarch=arm64; keyarch=arm64; \\',
        "    else repoarch=amd64; keyarch=x86_64; fi; \\",
        f'    echo "deb {DEVTOOLS_REPOS}/{distro}/$repoarch /" \\',
        "      > /etc/apt/sources.list.d/nvidia-devtools.list; \\",
        f'    apt-key adv --fetch-keys "{key}" \\',
        f'      || apt-key adv --fetch-keys "{fallback}"; \\',
        "    apt-get update -y; \\",
        f"    apt-get install -y --no-install-recommends {package}; \\",
        "    rm -rf /var/lib/apt/lists/*; \\",
        "    if ! command -v nsys >/dev/null 2>&1; then \\",
        '      ln -s "$(ls -d /opt/nvidia/nsight-systems*/*/bin/nsys | head -n 1)" \\',
        "        /usr/local/bin/nsys; \\",
        "    fi; \\",
        "    nsys --version",
        f"USER {user}",
        "",
    ]
    return "\n".join(lines)


def _repo_component(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "image"


def diag_image_tag(
    base_image: str, *, base_id: str | None = None, dockerfile: str | None = None
) -> str:
    """``cb-diag/<name>-nsys:<tag>-<base digest 12>-<recipe hash 8>``.

    The base digest comes from the pinned reference, else from ``base_id``
    (``docker image inspect`` Id), else from a hash of the reference string.
    """
    reference = base_image.strip()
    digest = image_digest(reference)
    if digest is None and base_id:
        match = _IMAGE_ID.search(base_id)
        digest = match.group(1) if match else None
    if digest is None:
        digest = hashlib.sha256(reference.encode()).hexdigest()
    last = reference.split("@", 1)[0].rsplit("/", 1)[-1]
    name, _, tag = last.partition(":")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip(".-") or "latest"
    recipe = hashlib.sha256(
        (dockerfile if dockerfile is not None else diag_dockerfile(reference)).encode()
    ).hexdigest()[:8]
    return (
        f"{DIAG_REPOSITORY}/{_repo_component(name)}-nsys:"
        f"{tag[:96]}-{digest[:12]}-{recipe}"
    )


def build_argv(tag: str, *, base_image: str, host_network: bool = True) -> list[str]:
    """``docker build`` reading the Dockerfile from stdin (no build context)."""
    argv = ["docker", "build"]
    if host_network:
        argv += ["--network", "host"]
    argv += [
        "--label",
        f"{LABEL_DIAG}=nsys",
        "--label",
        f"{LABEL_DIAG}.base={base_image}",
        "-t",
        tag,
        "-",
    ]
    return argv


def build_command(
    tag: str,
    *,
    base_image: str,
    host_network: bool = True,
    timeout_s: float = BUILD_TIMEOUT_S,
) -> str:
    argv = build_argv(tag, base_image=base_image, host_network=host_network)
    return bounded(shlex.join(argv), timeout_s, kill_after_s=30)


def start_profile_body(
    num_steps: int = DEFAULT_STEPS, start_step: int | None = 1
) -> dict[str, Any]:
    """Body of ``/engine/control/start_profile`` (SGLang ``ProfileReq``).

    ``start_step`` is an absolute forward count: SGLang starts at
    ``max(start_step, current + 1)``, i.e. the next step of a running server,
    and stops after ``num_steps`` (no stop request). ``None`` omits it and
    starts at once; 0 would do the same, so it is rejected for clarity.
    """
    if int(num_steps) < 1:
        raise ValueError("num_steps must be >= 1")
    body: dict[str, Any] = {
        "num_steps": int(num_steps),
        "activities": ["CUDA_PROFILER"],
    }
    if start_step is not None:
        if int(start_step) < 1:
            raise ValueError("start_step must be >= 1 (None starts immediately)")
        body["start_step"] = int(start_step)
    return body


def start_profile_argv(
    system_port: int,
    *,
    num_steps: int = DEFAULT_STEPS,
    start_step: int | None = 1,
    host: str = "127.0.0.1",
    max_time_s: float = 20.0,
) -> list[str]:
    """curl run on the worker's own node (host networking, system port)."""
    if not 0 < int(system_port) < 65536:
        raise ValueError(f"invalid system port {system_port!r}")
    body = json.dumps(
        start_profile_body(num_steps, start_step), sort_keys=True, separators=(",", ":")
    )
    return [
        "curl",
        "-sS",
        "--connect-timeout",
        "5",
        "--max-time",
        f"{max_time_s:g}",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        body,
        "-w",
        "\\n%{http_code}",
        f"http://{host}:{int(system_port)}{START_PROFILE_ROUTE}",
    ]


def start_profile_command(
    system_port: int,
    *,
    num_steps: int = DEFAULT_STEPS,
    start_step: int | None = 1,
    timeout_s: float = TRIGGER_TIMEOUT_S,
) -> str:
    argv = start_profile_argv(system_port, num_steps=num_steps, start_step=start_step)
    return bounded(shlex.join(argv), timeout_s)


def stat_command(paths: Sequence[str], timeout_s: float = STAT_TIMEOUT_S) -> str:
    """``size mtime path`` of each existing file (missing ones are skipped)."""
    argv = ["stat", "--printf", "%s %Y %n\\n", "--", *paths]
    return bounded(shlex.join(argv), timeout_s) + " 2>/dev/null || true"


def find_reports_command(directory: str, timeout_s: float = STAT_TIMEOUT_S) -> str:
    """Every ``.nsys-rep`` below a window directory, as ``size mtime path``."""
    argv = [
        "find",
        directory,
        "-maxdepth",
        "2",
        "-type",
        "f",
        "-name",
        f"*{REPORT_SUFFIX}",
        "-printf",
        "%s %T@ %p\\n",
    ]
    return bounded(shlex.join(argv), timeout_s) + " 2>/dev/null || true"


def stats_argv(
    report: str, *, output_base: str, reports: Sequence[str] = STATS_REPORTS
) -> list[str]:
    """``nsys stats`` writing ``<output_base>_<report>.csv`` per report.

    One ``--format``/``--output`` applies to every ``--report`` (nsys repeats
    the last element). The SQLite export lands next to the report.
    """
    return [
        "nsys",
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=csv",
        f"--output={output_base}",
        *(f"--report={name}" for name in reports),
        report,
    ]


def stats_container_argv(
    *,
    name: str,
    image: str,
    workdir: str,
    report: str,
    output_base: str,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """Foreground, self-removing diagnostic-image container running nsys stats
    with the experiment tree mounted (container paths in ``report``)."""
    return docker_run_argv(
        name=name,
        image=image,
        command=stats_argv(report, output_base=output_base),
        gpus=None,
        mounts=(Mount(host_experiments(workdir), CONTAINER_EXP),),
        labels=labels,
        rdma=False,
        detach=False,
        remove=True,
        shm_size="1g",
    )


def stats_command(argv: Sequence[str], timeout_s: float = STATS_TIMEOUT_S) -> str:
    return bounded(shlex.join(list(argv)), timeout_s, kill_after_s=30)


def csv_path(output_base: str, report: str) -> str:
    """File ``nsys stats --output=<base>`` writes for one report."""
    return f"{output_base}_{report}.csv"


def cat_command(path: str, timeout_s: float = CAT_TIMEOUT_S) -> str:
    return bounded(shlex.join(["cat", "--", path]), timeout_s)


def remove_command(path: str) -> str:
    return bounded(shlex.join(["rm", "-f", "--", path]), CLEANUP_TIMEOUT_S) + " || true"


# --------------------------------------------------------------------------
# Output parsers
# --------------------------------------------------------------------------


def _tail(text: str, limit: int = ERROR_TAIL_CHARS) -> str:
    return text.strip()[-limit:]


def parse_probe(stdout: str, *, image: str, rc: int = 0, stderr: str = "") -> NsysInfo:
    """``NsysInfo`` from the probe container's output."""
    marker = False
    path: str | None = None
    for line in stdout.splitlines():
        if line.startswith(PROBE_MARKER):
            marker = True
            path = line[len(PROBE_MARKER) :].strip() or None
    if not marker:
        detail = _tail(stderr, 300) or _tail(stdout, 300) or f"exit {rc}"
        return NsysInfo(
            image=image, available=False, notes=(f"probe failed: {detail}",)
        )
    if path is None:
        return NsysInfo(image=image, available=False, notes=("nsys not in the image",))
    version, triple = parse_nsys_version(stdout)
    if version is None:
        return NsysInfo(
            image=image,
            available=False,
            path=path,
            notes=(f"{path} --version printed no version",),
        )
    return NsysInfo(
        image=image, available=True, path=path, version=version, version_tuple=triple
    )


def parse_image_id(stdout: str) -> str | None:
    match = _IMAGE_ID.search(stdout)
    return f"sha256:{match.group(1)}" if match else None


def parse_trigger(
    stdout: str, *, node: str, port: int, rc: int = 0, stderr: str = ""
) -> ProfileTrigger:
    """curl output (body, then the ``-w`` status line) -> ``ProfileTrigger``."""
    lines = stdout.rstrip("\n").splitlines()
    status: int | None = None
    body = stdout
    if lines and lines[-1].strip().isdigit():
        status = int(lines[-1].strip()) or None
        body = "\n".join(lines[:-1])
    obj: Any = None
    if body.strip():
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            obj = None
    message: str | None = None
    state = None
    if isinstance(obj, dict):
        message = obj.get("message") or obj.get("error")
        state = obj.get("status")
    ok = rc == 0 and status == 200 and state in (None, "ok", "success")
    if not ok and not message:
        message = _tail(stderr, 300) or _tail(body, 300) or f"exit {rc}"
    return ProfileTrigger(
        node=node,
        port=port,
        ok=ok,
        http_status=status,
        message=str(message) if message else None,
    )


def parse_stat(stdout: str) -> dict[str, tuple[int, int]]:
    """path -> (size bytes, mtime) from ``stat_command`` output."""
    found: dict[str, tuple[int, int]] = {}
    for line in stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            found[parts[2].strip()] = (int(parts[0]), int(parts[1]))
    return found


def parse_find(stdout: str) -> dict[str, tuple[int, float]]:
    """path -> (size bytes, mtime) from ``find_reports_command`` output."""
    found: dict[str, tuple[int, float]] = {}
    for line in stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        try:
            found[parts[2].strip()] = (int(parts[0]), float(parts[1]))
        except ValueError:
            continue
    return found


_CAPTURE_START = re.compile(r"(?i)capture range started")
_CAPTURE_END = re.compile(r"(?i)capture range ended")
_GENERATED_INLINE = re.compile(
    r"(?:Generated:|Saved report file to|Report file moved to)\s*\"?"
    r"(/\S+?\.(?:nsys-rep|qdrep))\"?\s*$"
)
_REPORT_LINE = re.compile(r"^\s+(/\S+\.(?:nsys-rep|qdrep))\s*$")
_NSYS_ERROR = re.compile(
    r"^\s*(?:Error|ERROR|FATAL ERROR)\b[: ].*|.*\bnsys\b.*\b(?:error|failed)\b.*",
    re.IGNORECASE,
)


def parse_nsys_log(text: str, limit: int = 5) -> NsysLogState:
    """Capture-range and report lines nsys printed into a worker's log.

    nsys announces ``Generated:`` followed by indented report paths (older
    versions: ``Report file moved to "..."``); the exact wording varies by
    version, so callers also check the file itself (``wait_reports``).
    """
    generated: list[str] = []
    errors: list[str] = []
    started = ended = after_generated = False
    for line in text.splitlines():
        if after_generated:
            match = _REPORT_LINE.match(line)
            if match:
                generated.append(match.group(1))
                continue
            after_generated = False
        stripped = line.strip()
        if stripped == "Generated:":
            after_generated = True
            continue
        if _CAPTURE_START.search(line):
            started = True
        if _CAPTURE_END.search(line):
            ended = True
        inline = _GENERATED_INLINE.search(stripped)
        if inline:
            generated.append(inline.group(1))
        elif _NSYS_ERROR.match(line) and len(errors) < limit:
            errors.append(stripped[:300])
    return NsysLogState(
        capture_started=started,
        capture_ended=ended,
        generated=tuple(dict.fromkeys(generated)),
        errors=tuple(errors),
    )


# --------------------------------------------------------------------------
# Dry-run simulator
# --------------------------------------------------------------------------

SIM_NSYS_VERSION = "NVIDIA Nsight Systems version 2025.6.1.190-256162285624v0"
SIM_REPORT_BYTES = 187_654_321


def simulated_stdout(
    command: str,
    node: str,
    *,
    devices: int = 8,
    steps: int = DEFAULT_STEPS,
    now: float = 1_800_000_000.0,
) -> str | None:
    """Synthetic output of a diagnostics command (dry-run simulator), else None.

    Like the pinned image, a probed serving image has no nsys and a derived
    ``cb-diag/...`` image has one. ``docker build`` / ``image inspect`` /
    ``nsys stats`` need no output: the simulator's empty success suffices.
    """
    if PROBE_MARKER in command:
        if f" {DIAG_REPOSITORY}/" not in command:
            return f"{PROBE_MARKER}\n"
        return f"{PROBE_MARKER}/usr/local/bin/nsys\n{SIM_NSYS_VERSION}\n"
    if START_PROFILE_ROUTE in command:
        return '{"message":"Profiling started","status":"ok"}\n200'
    if command.startswith("timeout") and " stat --printf " in command:
        paths = [a for a in shlex.split(command) if a.endswith(REPORT_SUFFIX)]
        return "".join(f"{SIM_REPORT_BYTES} {int(now)} {p}\n" for p in paths)
    if " cat -- " in command:
        match = _ROLE_IN_NAME.search(command)
        role = match.group(1) if match else "decode"
        for report, make in (
            ("cuda_gpu_kern_sum", synthetic_kern_sum_csv),
            ("cuda_gpu_mem_time_sum", synthetic_mem_time_sum_csv),
            ("cuda_kern_exec_sum", synthetic_kern_exec_sum_csv),
        ):
            if f"_{report}.csv" in command:
                return make(role, devices=devices, steps=steps)
    return None


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class NsysDiagnostics:
    """Fail-soft runner of the diagnostic-window commands on cluster nodes.

    ``workdir`` is the harness directory on every node (``<workdir>/
    experiments`` is mounted at ``/bench/experiments``). Raw evidence lands
    in the node0-local ``results_dir``: probe and build logs, trigger
    responses, and the CSVs of every ``nsys stats`` run (``stats/``).
    """

    def __init__(
        self,
        cluster: Cluster,
        *,
        workdir: str,
        results_dir: Path,
        labels: Mapping[str, str] | None = None,
        name_prefix: str = "cb-diag",
        probe_timeout_s: float = PROBE_TIMEOUT_S,
        build_timeout_s: float = BUILD_TIMEOUT_S,
        stats_timeout_s: float = STATS_TIMEOUT_S,
    ):
        self._cluster = cluster
        self._workdir = workdir.rstrip("/")
        self._results = results_dir
        self._labels = dict(labels or {})
        self._prefix = sanitize_label(name_prefix) or "cb-diag"
        self._probe_timeout = probe_timeout_s
        self._build_timeout = build_timeout_s
        self._stats_timeout = stats_timeout_s

    # ------------------------------------------------------------ helpers

    def _save(self, relpath: str, text: str) -> None:
        try:
            atomic_write_text(self._results / relpath, text)
        except OSError as exc:
            # Keeping raw evidence is best effort; never fail the window on it.
            logger.warning("could not save %s: %s", relpath, exc)

    def _run(
        self, node: str, command: str, timeout_s: float, input_text: str | None = None
    ) -> CommandResult | None:
        try:
            return self._cluster.run(
                node,
                command,
                input_text=input_text,
                timeout=timeout_s + OUTER_MARGIN_S,
                check=False,
            )
        except (CommandError, OSError) as exc:
            logger.warning("[%s] diagnostics command could not run: %s", node, exc)
            return None

    @staticmethod
    def _transcript(command: str, res: CommandResult | None) -> str:
        if res is None:
            return f"$ {command}\n(could not run)\n"
        return f"$ {command}\n[exit {res.rc}]\n{res.stdout}\n{res.stderr}\n"

    # -------------------------------------------------------------- image

    def probe(self, node: str, image: str) -> NsysInfo:
        """Does ``image`` ship a working nsys? (~5-20 s on a pulled image)."""
        name = f"{self._prefix}-probe-{sanitize_label(node)}"
        command = probe_command(
            image, name=name, labels=self._labels, timeout_s=self._probe_timeout
        )
        self._run(node, cleanup_command(name), CLEANUP_TIMEOUT_S)
        res = self._run(node, command, self._probe_timeout)
        self._save(f"probe-{sanitize_label(node)}.txt", self._transcript(command, res))
        if res is None:
            return NsysInfo(
                image=image, available=False, notes=("probe could not run",)
            )
        if not res.ok:
            self._run(node, cleanup_command(name), CLEANUP_TIMEOUT_S)
        return parse_probe(res.stdout, image=image, rc=res.rc, stderr=res.stderr)

    def ensure_image(
        self, nodes: Sequence[str], base_image: str, *, rebuild: bool = False
    ) -> dict[str, DiagImage]:
        """Per node: an image with nsys for the diagnostic workers.

        The serving image itself when it already has nsys; else an existing
        or freshly built ``diag_image_tag`` (nodes build concurrently, each on
        its own Docker daemon; ~2-6 min with a warm base image).
        """
        try:
            dockerfile = diag_dockerfile(base_image)
        except ValueError as exc:
            return {
                n: DiagImage(node=n, base_image=base_image, notes=(str(exc),))
                for n in nodes
            }
        return self._cluster.fanout(
            list(nodes),
            lambda node: self._ensure_one(node, base_image, dockerfile, rebuild),
        )

    def _ensure_one(
        self, node: str, base_image: str, dockerfile: str, rebuild: bool
    ) -> DiagImage:
        base = self.probe(node, base_image)
        if base.available:
            return DiagImage(
                node=node,
                base_image=base_image,
                image=base_image,
                nsys=base,
                notes=("the serving image already ships nsys",),
            )
        notes: list[str] = []
        base_id = None
        if image_digest(base_image) is None:
            res = self._run(node, image_id_command(base_image), INSPECT_TIMEOUT_S)
            base_id = parse_image_id(res.stdout) if res is not None and res.ok else None
            if base_id is None:
                notes.append("base image id unknown; tag hashed from the reference")
        tag = diag_image_tag(base_image, base_id=base_id, dockerfile=dockerfile)
        if not rebuild:
            res = self._run(node, image_id_command(tag), INSPECT_TIMEOUT_S)
            if res is not None and res.ok and parse_image_id(res.stdout):
                info = self.probe(node, tag)
                if info.available:
                    return DiagImage(
                        node=node,
                        base_image=base_image,
                        image=tag,
                        nsys=info,
                        notes=tuple(notes),
                    )
                notes.append("existing diagnostic image has no working nsys")
        built, log = self._build(node, tag, base_image, dockerfile)
        self._save(f"build-{sanitize_label(node)}.log", log)
        if not built:
            notes.append(f"diagnostic image build failed: {_tail(log, 400)}")
            return DiagImage(node=node, base_image=base_image, notes=tuple(notes))
        info = self.probe(node, tag)
        if not info.available:
            notes.append("built diagnostic image has no working nsys")
        return DiagImage(
            node=node,
            base_image=base_image,
            image=tag,
            built=True,
            nsys=info,
            notes=tuple(notes),
        )

    def _build(
        self, node: str, tag: str, base_image: str, dockerfile: str
    ) -> tuple[bool, str]:
        """Build with host networking (like every harness container); retry
        once on the default network when the builder rejects that mode."""
        transcript: list[str] = []
        for host_network in (True, False):
            command = build_command(
                tag,
                base_image=base_image,
                host_network=host_network,
                timeout_s=self._build_timeout,
            )
            res = self._run(node, command, self._build_timeout, input_text=dockerfile)
            transcript.append(self._transcript(command, res))
            if res is not None and res.ok:
                return True, "".join(transcript)
            if res is None or not _NETWORK_ERROR.search(res.stderr + res.stdout):
                break
        return False, "".join(transcript)

    # ------------------------------------------------------------ capture

    def trigger(
        self,
        node: str,
        system_port: int,
        *,
        num_steps: int = DEFAULT_STEPS,
        start_step: int | None = 1,
    ) -> ProfileTrigger:
        """Ask one instance leader to profile its next ``num_steps`` steps."""
        try:
            command = start_profile_command(
                system_port, num_steps=num_steps, start_step=start_step
            )
        except ValueError as exc:
            return ProfileTrigger(
                node=node, port=int(system_port), ok=False, message=str(exc)
            )
        res = self._run(node, command, TRIGGER_TIMEOUT_S)
        self._save(
            f"trigger-{sanitize_label(node)}-{int(system_port)}.txt",
            self._transcript(command, res),
        )
        if res is None:
            return ProfileTrigger(
                node=node, port=int(system_port), ok=False, message="could not run"
            )
        return parse_trigger(
            res.stdout, node=node, port=int(system_port), rc=res.rc, stderr=res.stderr
        )

    def wait_reports(
        self,
        expected: Mapping[str, Sequence[str]],
        *,
        clock: Clock,
        timeout_s: float = REPORT_WAIT_S,
        poll_s: float = REPORT_POLL_S,
        settle_polls: int = 2,
    ) -> dict[str, tuple[ReportFile, ...]]:
        """Poll until every expected report exists and stopped growing.

        ``expected`` maps node -> host report paths. A report is ready when
        it is non-empty and its size was identical in ``settle_polls``
        consecutive polls; the rest come back with ``ready=False``.
        """
        settle = max(1, settle_polls)
        sizes: dict[tuple[str, str], list[int]] = {}
        ready: set[tuple[str, str]] = set()
        wanted = {(n, p) for n, paths in expected.items() for p in paths}
        deadline = clock.now() + timeout_s
        while True:
            pending: dict[str, list[str]] = {}
            for node, path in sorted(wanted - ready):
                pending.setdefault(node, []).append(path)
            if not pending:
                break

            def look(
                node: str, paths: Mapping[str, list[str]] = pending
            ) -> dict[str, tuple[int, int]]:
                res = self._run(node, stat_command(paths[node]), STAT_TIMEOUT_S)
                return parse_stat(res.stdout) if res is not None else {}

            seen = self._cluster.fanout(list(pending), look)
            for node, found in seen.items():
                for path in pending[node]:
                    if path not in found:
                        continue
                    history = sizes.setdefault((node, path), [])
                    history.append(found[path][0])
                    recent = history[-settle:]
                    if (
                        recent[-1] > 0
                        and len(recent) == settle
                        and len(set(recent)) == 1
                    ):
                        ready.add((node, path))
            if wanted <= ready or clock.now() >= deadline:
                break
            clock.sleep(poll_s)
        return {
            node: tuple(
                ReportFile(
                    node=node,
                    path=path,
                    size_bytes=sizes[(node, path)][-1]
                    if (node, path) in sizes
                    else None,
                    ready=(node, path) in ready,
                )
                for path in paths
            )
            for node, paths in expected.items()
        }

    # -------------------------------------------------------------- stats

    def stats(
        self,
        node: str,
        report: str,
        *,
        image: str,
        source: str | None = None,
        expected_devices: int | None = None,
        keep_sqlite: bool = False,
    ) -> KernelReport:
        """``nsys stats`` of one report on its node -> ``KernelReport``.

        Runs in a foreground container of ``image`` (the diagnostic image)
        with ``<workdir>/experiments`` mounted, then reads the three CSVs.
        ``expected_devices`` (the container's GPU count) drives the coverage
        check. The SQLite export (hundreds of MB) is removed unless kept.
        """
        name = posixpath.basename(report)
        label = source or (
            name[: -len(REPORT_SUFFIX)] if name.endswith(REPORT_SUFFIX) else name
        )
        notes: list[str] = []
        inside = to_container_path(report, self._workdir)
        if inside is None:
            return KernelReport(
                node=node,
                source=label,
                report=report,
                notes=(f"report is outside {host_experiments(self._workdir)}",),
            )
        base_host = (
            report[: -len(REPORT_SUFFIX)] if report.endswith(REPORT_SUFFIX) else report
        )
        base_inside = (
            inside[: -len(REPORT_SUFFIX)] if inside.endswith(REPORT_SUFFIX) else inside
        )
        container = (
            f"{self._prefix}-stats-{sanitize_label(node)}-{sanitize_label(label)}"[:120]
        )
        argv = stats_container_argv(
            name=container,
            image=image,
            workdir=self._workdir,
            report=inside,
            output_base=base_inside,
            labels=self._labels,
        )
        command = stats_command(argv, self._stats_timeout)
        self._run(node, cleanup_command(container), CLEANUP_TIMEOUT_S)
        res = self._run(node, command, self._stats_timeout)
        safe = sanitize_label(label) or "report"
        self._save(f"stats/{safe}.log", self._transcript(command, res))
        if res is None or not res.ok:
            detail = "could not run" if res is None else f"exit {res.rc}"
            notes.append(f"{label}: nsys stats failed ({detail})")
            self._run(node, cleanup_command(container), CLEANUP_TIMEOUT_S)
        if res is not None:
            notes.extend(
                f"{label}: {p}" for p in stats_problems(res.stdout + "\n" + res.stderr)
            )
        texts: dict[str, str] = {}
        files: list[str] = []
        for kind in STATS_REPORTS:
            path = csv_path(base_host, kind)
            out = self._run(node, cat_command(path), CAT_TIMEOUT_S)
            if out is None or not out.ok:
                notes.append(f"{label}: {kind} CSV missing")
                continue
            texts[kind] = out.stdout
            files.append(path)
            self._save(f"stats/{safe}_{kind}.csv", out.stdout)
        if not keep_sqlite:
            self._run(node, remove_command(f"{base_host}.sqlite"), CLEANUP_TIMEOUT_S)
        rows = parse_kern_sum_csv(texts.get("cuda_gpu_kern_sum", ""))
        rows += parse_mem_time_sum_csv(texts.get("cuda_gpu_mem_time_sum", ""))
        exec_rows = None
        if "cuda_kern_exec_sum" in texts:
            exec_rows = parse_kern_exec_sum_csv(texts["cuda_kern_exec_sum"])
        else:
            notes.append(f"{label}: GPU coverage unknown")
        result = None
        if rows:
            result = breakdown(
                rows,
                source=label,
                exec_rows=exec_rows,
                expected_devices=expected_devices,
                notes=notes,
            )
        elif "cuda_gpu_kern_sum" in texts:
            notes.append(f"{label}: no kernels in the capture range")
        return KernelReport(
            node=node,
            source=label,
            report=report,
            breakdown=result,
            files=tuple(files),
            # The breakdown's notes add the coverage findings to these.
            notes=result.notes if result is not None else tuple(dict.fromkeys(notes)),
        )
