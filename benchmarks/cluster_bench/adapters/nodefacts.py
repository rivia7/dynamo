# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect node facts, the software stack, clock offsets and disk speed.

``FactCollector`` runs ``agents/node_facts.py`` on cluster nodes through an
injected ``ports.Cluster`` and turns its JSON into the models of
``domain/nodefacts.py``:

* ``host_facts``    host-side agent on many nodes at once ("before" / "after")
* ``software``      the agent inside the pinned serving image (one node)
* ``clock_samples`` sequential round trips from node0 for ``clock_offsets``
* ``disk_read``     O_DIRECT read speed of the weights disk on many nodes

Everything is fail-soft: a node whose command fails, times out or prints
garbage yields an "unavailable" document (or ``None`` / an empty sample list)
and a log line, never an exception. Raw agent output is kept under
``results_dir`` for the knowledge base. Commands are rendered by pure
``*_command`` / ``*_argv`` helpers so ``--dry-run`` can print them, and
``simulated_stdout`` answers them with synthetic data for the simulator.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from benchmarks.cluster_bench.adapters.docker import (
    LABEL_ROLE,
    Mount,
    docker_run_argv,
    sanitize_label,
)
from benchmarks.cluster_bench.domain.nodefacts import (
    DiskRead,
    NodeFacts,
    SoftwareStack,
    failed_disk_read,
    parse_disk_read,
    parse_node_facts,
    parse_software,
    synthetic_disk_read,
    synthetic_facts,
    synthetic_software,
    unavailable_facts,
)
from benchmarks.cluster_bench.fsutil import atomic_write_text
from benchmarks.cluster_bench.ports import Cluster, CommandError

logger = logging.getLogger(__name__)

AGENT_REL = "benchmarks/cluster_bench/agents/node_facts.py"
CONTAINER_SRC = "/bench/src"
CONTAINER_AGENT = f"{CONTAINER_SRC}/{AGENT_REL}"
CONTAINER_PYTHON = "python3"
FACTS_ROLE = "facts"

# The agent bounds itself (--deadline-s); the remote timeout adds margin.
HOST_DEADLINE_S = 150.0
HOST_TIMEOUT_S = 240.0
# docker run: container start, torch import and CUDA init (~30-90 s).
SOFTWARE_TIMEOUT_S = 600.0
CLOCK_TIMEOUT_S = 20.0
DEFAULT_CLOCK_ROUNDS = 5
DISK_MAX_BYTES = 8 << 30
DISK_MAX_SECONDS = 120.0
DISK_BLOCK_MIB = 16
DISK_TIMEOUT_MARGIN_S = 120.0
CLEANUP_TIMEOUT_S = 60.0
ERROR_TAIL_CHARS = 2000

_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CLOCK_SNIPPET = "import time; print(repr(time.time()))"


# --------------------------------------------------------------------------
# Pure command builders and parsers
# --------------------------------------------------------------------------


def agent_path(src_dir: str) -> str:
    return f"{src_dir.rstrip('/')}/{AGENT_REL}"


def host_facts_command(
    python: str, src_dir: str, node: str, deadline_s: float = HOST_DEADLINE_S
) -> str:
    """Host-mode agent command (prints one JSON line)."""
    return shlex.join(
        [
            python,
            agent_path(src_dir),
            "host",
            "--node",
            node,
            "--deadline-s",
            f"{deadline_s:g}",
        ]
    )


def disk_read_command(
    python: str,
    src_dir: str,
    node: str,
    path: str,
    *,
    max_bytes: int = DISK_MAX_BYTES,
    block_mib: int = DISK_BLOCK_MIB,
    max_seconds: float = DISK_MAX_SECONDS,
) -> str:
    return shlex.join(
        [
            python,
            agent_path(src_dir),
            "disk-read",
            "--node",
            node,
            "--path",
            path,
            "--max-bytes",
            str(int(max_bytes)),
            "--block-mib",
            str(int(block_mib)),
            "--max-seconds",
            f"{max_seconds:g}",
        ]
    )


def clock_command(python: str) -> str:
    """Prints the node's ``time.time()``; -I -S keeps interpreter start short."""
    return shlex.join([python, "-I", "-S", "-c", CLOCK_SNIPPET])


def software_argv(
    *,
    name: str,
    image: str,
    src_dir: str,
    node: str,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """Foreground, self-removing container running the agent in container mode.

    GPU 0 is passed so torch can report CUDA / NCCL; ``/dev/infiniband`` so
    ``ibv_devinfo -l`` sees the HCAs the workers will see.
    """
    return docker_run_argv(
        name=name,
        image=image,
        command=[CONTAINER_PYTHON, CONTAINER_AGENT, "container", "--node", node],
        gpus=[0],
        mounts=(Mount(src_dir, CONTAINER_SRC, read_only=True),),
        labels=labels,
        detach=False,
        remove=True,
    )


def extract_json_document(text: str) -> dict[str, Any]:
    """The agent's JSON object: the last stdout line that parses as one.

    Image entrypoints may print banners first; the agent prints one line.
    """
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in agent output")


def parse_remote_time(text: str) -> float | None:
    """The float the clock command printed (None when absent or garbage)."""
    for line in reversed(text.splitlines()):
        try:
            value = float(line.strip())
        except ValueError:
            continue
        return value
    return None


def simulated_stdout(
    command: str,
    node: str,
    *,
    node_names: Sequence[str],
    gpu_name: str,
    now: float | None = None,
) -> str | None:
    """Synthetic agent output for ``command`` (dry-run simulator), else None."""
    if CLOCK_SNIPPET in command:
        return repr(now if now is not None else 1_800_000_000.0)
    if AGENT_REL not in command:
        return None
    argv = shlex.split(command)
    if "disk-read" in argv:
        path = argv[argv.index("--path") + 1] if "--path" in argv else "/models"
        return synthetic_disk_read(node, path).model_dump_json()
    if "container" in argv:
        return synthetic_software().model_dump_json()
    facts = synthetic_facts(node_names, gpu_name)
    if node in facts:
        return facts[node].model_dump_json()
    return None


def _tail(text: str) -> str:
    return text.strip()[-ERROR_TAIL_CHARS:]


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------


class FactCollector:
    """Runs the node-facts agent and keeps its raw output.

    ``src_dir`` / ``python`` default to ``<workdir>/src`` and
    ``<workdir>/venv/bin/python`` (the harness layout on every node).
    ``now`` is node0's clock (the runner's). Raw files land in
    ``results_dir``: ``host-<tag>/<node>.json``, ``software-<node>.json``,
    ``clock-samples.json`` and ``disk-read/<node>.json`` (``*.error.txt`` when
    a node failed).
    """

    def __init__(
        self,
        cluster: Cluster,
        *,
        image: str,
        workdir: str,
        results_dir: Path,
        now: Callable[[], float],
        src_dir: str | None = None,
        python: str | None = None,
        name_prefix: str = "cb-facts",
        labels: Mapping[str, str] | None = None,
        reference: str | None = None,
        host_timeout_s: float = HOST_TIMEOUT_S,
        software_timeout_s: float = SOFTWARE_TIMEOUT_S,
        clock_timeout_s: float = CLOCK_TIMEOUT_S,
    ):
        self._cluster = cluster
        self._image = image
        self._workdir = workdir.rstrip("/")
        self._src_dir = src_dir or f"{self._workdir}/src"
        self._python = python or f"{self._workdir}/venv/bin/python"
        self._results = results_dir
        self._now = now
        self._prefix = sanitize_label(name_prefix) or "cb-facts"
        self._labels = {LABEL_ROLE: FACTS_ROLE, **(labels or {})}
        self._reference = reference
        self._host_timeout_s = host_timeout_s
        self._software_timeout_s = software_timeout_s
        self._clock_timeout_s = clock_timeout_s

    # ------------------------------------------------------------ helpers

    def _save(self, relpath: str, text: str) -> None:
        try:
            atomic_write_text(self._results / relpath, text)
        except OSError as exc:
            # Keeping raw evidence is best effort; never fail collection on it.
            logger.warning("could not save %s: %s", relpath, exc)

    def _save_json(self, relpath: str, obj: Any) -> None:
        self._save(
            relpath, json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n"
        )

    def _run(self, node: str, command: str, timeout: float) -> tuple[str | None, str]:
        """(stdout, error): stdout is None when the command did not succeed."""
        try:
            res = self._cluster.run(node, command, timeout=timeout, check=False)
        except (CommandError, OSError) as exc:
            return None, f"could not run: {exc}"
        if not res.ok:
            detail = _tail(res.stderr) or _tail(res.stdout)
            return res.stdout, f"exit {res.rc}: {detail}"
        return res.stdout, ""

    def _document(
        self, node: str, command: str, timeout: float, relpath: str
    ) -> tuple[dict[str, Any] | None, str]:
        stdout, error = self._run(node, command, timeout)
        if stdout is not None:
            try:
                obj = extract_json_document(stdout)
            except ValueError:
                error = error or "no JSON object in agent output"
            else:
                self._save_json(relpath, obj)
                return obj, error
        stem = relpath[: -len(".json")] if relpath.endswith(".json") else relpath
        self._save(f"{stem}.error.txt", f"$ {command}\n{error}\n")
        logger.warning("[%s] node facts: %s", node, error)
        return None, error

    # ------------------------------------------------------------ host facts

    def host_facts(self, nodes: Sequence[str], *, tag: str) -> dict[str, NodeFacts]:
        """Host-mode facts of ``nodes``, concurrently (~10-30 s per node).

        ``tag`` names the snapshot ("before", "after", ...). A node without a
        usable document maps to ``unavailable_facts`` (never missing).
        """
        if not _TAG_RE.match(tag):
            raise ValueError(f"invalid snapshot tag {tag!r}")

        def one(node: str) -> NodeFacts:
            command = host_facts_command(self._python, self._src_dir, node)
            obj, error = self._document(
                node, command, self._host_timeout_s, f"host-{tag}/{node}.json"
            )
            if obj is None:
                return unavailable_facts(node, error)
            try:
                return parse_node_facts(obj, node=node)
            except ValueError as exc:
                logger.warning("[%s] node facts unparsable: %s", node, exc)
                return unavailable_facts(node, f"unparsable: {exc}")

        return self._cluster.fanout(list(nodes), one)

    # -------------------------------------------------------------- software

    def software(self, node: str) -> SoftwareStack | None:
        """Software stack inside the serving image on ``node`` (~30-90 s)."""
        name = f"{self._prefix}-sw-{sanitize_label(node)}"
        remove = f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true"
        argv = software_argv(
            name=name,
            image=self._image,
            src_dir=self._src_dir,
            node=node,
            labels=self._labels,
        )
        self._run(node, remove, CLEANUP_TIMEOUT_S)
        obj, error = self._document(
            node, shlex.join(argv), self._software_timeout_s, f"software-{node}.json"
        )
        if error:
            # A timed-out client leaves the container (and GPU 0) busy.
            self._run(node, remove, CLEANUP_TIMEOUT_S)
        if obj is None:
            return None
        try:
            return parse_software(obj, node=node, image=self._image)
        except ValueError as exc:
            logger.warning("[%s] software stack unparsable: %s", node, exc)
            return None

    # ---------------------------------------------------------------- clocks

    def clock_samples(
        self, nodes: Sequence[str], rounds: int = DEFAULT_CLOCK_ROUNDS
    ) -> dict[str, list[tuple[float, float, float]]]:
        """(t_send, remote_time, t_recv) round trips from node0, sequentially.

        The reference node (``reference`` or the cluster's first node) gets
        one zero-RTT sample, i.e. offset 0. A node that fails stops being
        sampled and keeps whatever samples it had (possibly none).
        """
        reference = self._reference or (
            self._cluster.nodes[0] if self._cluster.nodes else None
        )
        command = clock_command(self._python)
        out: dict[str, list[tuple[float, float, float]]] = {}
        for node in nodes:
            if node == reference:
                t = self._now()
                out[node] = [(t, t, t)]
                continue
            samples: list[tuple[float, float, float]] = []
            for _ in range(max(1, rounds)):
                t_send = self._now()
                stdout, error = self._run(node, command, self._clock_timeout_s)
                t_recv = self._now()
                remote = parse_remote_time(stdout or "") if not error else None
                if remote is None:
                    logger.warning(
                        "[%s] clock sample failed: %s", node, error or "no time"
                    )
                    break
                samples.append((t_send, remote, t_recv))
            out[node] = samples
        self._save_json(
            "clock-samples.json",
            {
                "reference": reference,
                "samples": {n: [list(s) for s in v] for n, v in out.items()},
            },
        )
        return out

    # ------------------------------------------------------------ disk reads

    def disk_read(
        self,
        nodes: Sequence[str],
        path: str,
        max_bytes: int = DISK_MAX_BYTES,
        *,
        block_mib: int = DISK_BLOCK_MIB,
        max_seconds: float = DISK_MAX_SECONDS,
    ) -> dict[str, DiskRead]:
        """Sequential direct read of ``path`` (a file, or the largest
        ``*.safetensors`` below a directory) on every node, concurrently.

        Each node reads at most ``max_bytes`` for at most ``max_seconds``
        (the agent may try dd, O_DIRECT and buffered reads in turn).
        """
        timeout = 3 * max_seconds + DISK_TIMEOUT_MARGIN_S

        def one(node: str) -> DiskRead:
            command = disk_read_command(
                self._python,
                self._src_dir,
                node,
                path,
                max_bytes=max_bytes,
                block_mib=block_mib,
                max_seconds=max_seconds,
            )
            obj, error = self._document(
                node, command, timeout, f"disk-read/{node}.json"
            )
            if obj is None:
                return failed_disk_read(node, path, error)
            try:
                return parse_disk_read(obj, node=node)
            except ValueError as exc:
                return failed_disk_read(node, path, f"unparsable: {exc}")

        return self._cluster.fanout(list(nodes), one)
