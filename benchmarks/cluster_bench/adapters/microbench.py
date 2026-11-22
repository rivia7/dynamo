# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speed-of-light microbenchmarks: runner-side orchestration and parsing.

The node agents ``agents/p2p_bench.py`` and ``agents/nccl_bench.py`` run inside
the serving image (torch + NCCL). This module renders their ``docker run``
argv (``*_job`` methods and pure ``build_*`` helpers, so ``--dry-run`` can
print them), runs them through ``DockerHost``, fetches the JSON they write
with ``Cluster.get_file`` and turns it into ``PeakResult`` records (knowledge
base evidence category 5: hardware speed of light).

Planes: intra-node GPU<->GPU copies and collectives are ``nvlink`` when the
node has NVLink, else ``pcie``; host<->device copies are ``pcie``; anything
that crosses nodes is ``ib``. 1 GB = 1e9 bytes everywhere.

The shared torchrun and rendezvous-port helpers here are also used by the
model gather (``adapters/hf_fetch.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import re
import shlex
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.cluster_bench.adapters.docker import (
    LABEL_ROLE,
    DockerHost,
    Mount,
    docker_run_argv,
)
from benchmarks.cluster_bench.domain.schema import Inventory, PeakResult, Plane
from benchmarks.cluster_bench.ports import Cluster, CommandError

logger = logging.getLogger(__name__)

CONTAINER_SRC = "/bench/src"
AGENTS_DIR = f"{CONTAINER_SRC}/benchmarks/cluster_bench/agents"
P2P_SCRIPT = f"{AGENTS_DIR}/p2p_bench.py"
NCCL_SCRIPT = f"{AGENTS_DIR}/nccl_bench.py"
GATHER_SCRIPT = f"{AGENTS_DIR}/file_gather.py"
CONTAINER_PYTHON = "python3"

P2P_TOOL = "p2p_bench(torch)"
NCCL_TOOL = "nccl_bench(torch.distributed)"
MICROBENCH_ROLE = "microbench"

DEFAULT_PORT_BASE = 29600
PORT_SPAN = 200
DEFAULT_GPUS_PER_NODE = 8
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_OPS = ("all_reduce", "all_gather")
DEFAULT_SIZES = ("64M", "256M", "1G")
# A measured GPU-GPU copy above this is NVLink: PCIe Gen5 x16 tops out near 64.
NVLINK_INFER_THRESHOLD_GBS = 100.0


# --------------------------------------------------------------------------
# Pure command builders
# --------------------------------------------------------------------------


def rendezvous_port(base: int, key: str, span: int = PORT_SPAN) -> int:
    """A deterministic port in ``[base, base + span)`` for one rendezvous.

    Distinct keys spread over the range so back-to-back jobs do not reuse a
    port that may still be in TIME_WAIT.
    """
    if span <= 0 or base <= 0 or base + span > 65536:
        raise ValueError(f"invalid port range base={base} span={span}")
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % span
    return base + offset


def build_torchrun_command(
    script: str,
    script_args: Sequence[str],
    *,
    nproc_per_node: int,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> list[str]:
    """``torchrun`` argv: standalone on one node, static rendezvous otherwise."""
    if nproc_per_node <= 0 or nnodes <= 0 or not 0 <= node_rank < nnodes:
        raise ValueError(
            f"bad torchrun layout: nnodes={nnodes} node_rank={node_rank} "
            f"nproc_per_node={nproc_per_node}"
        )
    argv = ["torchrun"]
    if master_addr is None:
        if nnodes != 1:
            raise ValueError("multi-node torchrun needs master_addr and master_port")
        argv += ["--standalone", "--nnodes", "1"]
    else:
        if master_port is None:
            raise ValueError("master_port is required with master_addr")
        argv += [
            "--nnodes", str(nnodes),
            "--node-rank", str(node_rank),
            "--master-addr", master_addr,
            "--master-port", str(master_port),
        ]  # fmt: skip
    argv += ["--nproc-per-node", str(nproc_per_node), script, *script_args]
    return argv


def nccl_ib_hca_value(hcas: Sequence[str]) -> str:
    """``NCCL_IB_HCA`` exact-match list: ``=mlx5_0,mlx5_3`` (no prefix matching)."""
    names = [h.strip() for h in hcas]
    if not names or any(not n or "," in n or n.startswith(("=", "^")) for n in names):
        raise ValueError(f"invalid HCA subset {list(hcas)!r}")
    return "=" + ",".join(names)


def build_p2p_command(
    *, out: str, node: str, size_mib: int = 1024, iters: int = 10, warmup: int = 3
) -> list[str]:
    """In-container argv of the intra-node copy benchmark."""
    return [
        CONTAINER_PYTHON, P2P_SCRIPT,
        "--out", out,
        "--node", node,
        "--size-mib", str(size_mib),
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]  # fmt: skip


def build_nccl_collective_args(
    *,
    out: str,
    ops: Sequence[str] = DEFAULT_OPS,
    sizes: Sequence[str] = DEFAULT_SIZES,
    iters: int = 20,
    warmup: int = 5,
) -> list[str]:
    """``nccl_bench.py --mode collective`` arguments."""
    return [
        "--mode", "collective",
        "--ops", ",".join(ops),
        "--sizes", ",".join(sizes),
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--out", out,
    ]  # fmt: skip


def build_nccl_pairs_args(
    *, out: str, size: str = "1G", iters: int = 20, warmup: int = 5
) -> list[str]:
    """``nccl_bench.py --mode pairs`` arguments."""
    return [
        "--mode", "pairs",
        "--size", size,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--out", out,
    ]  # fmt: skip


def build_microbench_docker(
    *,
    name: str,
    image: str,
    command: Sequence[str],
    gpus: Sequence[int],
    src_dir: str,
    out_dir: str,
    env: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
) -> list[str]:
    """Foreground, self-removing container with the harness source read-only."""
    return docker_run_argv(
        name=name,
        image=image,
        command=command,
        gpus=gpus,
        env=env,
        mounts=(Mount(src_dir, CONTAINER_SRC, read_only=True), Mount(out_dir, out_dir)),
        labels=labels,
        detach=False,
        remove=True,
    )


def output_path(remote_out: str, suffix: str) -> str:
    """``/x/nccl.json`` + ``intra-node1`` -> ``/x/nccl-intra-node1.json``."""
    stem, ext = posixpath.splitext(remote_out)
    return f"{stem}-{suffix}{ext or '.json'}"


# --------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------


def _require_kind(obj: Mapping[str, Any], kind: str) -> None:
    if obj.get("kind") != kind:
        raise ValueError(f"expected a {kind!r} result, got kind={obj.get('kind')!r}")


def _measured_pairs(
    obj: Mapping[str, Any], peer_only: bool
) -> list[tuple[int, int, float]]:
    copy = obj["copy_gbs"]
    peer = obj["can_access_peer"]
    return [
        (i, j, float(value))
        for i, row in enumerate(copy)
        for j, value in enumerate(row)
        if i != j and value is not None and (peer[i][j] or not peer_only)
    ]


def infer_nvlink(obj: Mapping[str, Any]) -> bool:
    """Fallback NVLink detection from measured peer copies (no inventory)."""
    _require_kind(obj, "p2p")
    values = [v for _, _, v in _measured_pairs(obj, peer_only=True)]
    return bool(values) and statistics.median(values) > NVLINK_INFER_THRESHOLD_GBS


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "min_gbs": min(values),
        "median_gbs": statistics.median(values),
        "max_gbs": max(values),
    }


def parse_p2p(
    obj: Mapping[str, Any],
    nvlink: bool,
    *,
    command: str = "",
    details: Mapping[str, Any] | None = None,
) -> list[PeakResult]:
    """``p2p_bench.py`` JSON -> p2p_copy, p2p_ring_aggregate, h2d, d2h peaks.

    ``p2p_copy`` is the median over measured ordered pairs (min/median/max and
    the full matrix in details); h2d/d2h are medians over GPUs.
    """
    _require_kind(obj, "p2p")
    node = str(obj["node"])
    count = int(obj["gpu_count"])
    plane: Plane = "nvlink" if nvlink else "pcie"
    common = {
        "gpu_count": count,
        "size_bytes": int(obj["size_bytes"]),
        "torch": obj.get("torch"),
        "cuda": obj.get("cuda"),
        "errors": list(obj.get("errors") or []),
        **(details or {}),
    }
    results: list[PeakResult] = []
    measured = _measured_pairs(obj, peer_only=False)
    if measured:
        values = [v for _, _, v in measured]
        slow_src, slow_dst, _ = min(measured, key=_third)
        peer = obj["can_access_peer"]
        results.append(
            PeakResult(
                plane=plane,
                test="p2p_copy",
                scope=node,
                value_gbs=statistics.median(values),
                tool=P2P_TOOL,
                command=command,
                details={
                    **common,
                    "statistic": "median_over_ordered_pairs",
                    **_stats(values),
                    "slowest_pair": [slow_src, slow_dst],
                    "pairs_measured": len(values),
                    "pairs_failed": count * (count - 1) - len(values),
                    "peer_pairs": sum(
                        1
                        for i in range(count)
                        for j in range(count)
                        if i != j and peer[i][j]
                    ),
                    "copy_gbs": obj["copy_gbs"],
                    "can_access_peer": peer,
                },
            )
        )
    ring = obj.get("ring_aggregate_gbs")
    if ring is not None:
        results.append(
            PeakResult(
                plane=plane,
                test="p2p_ring_aggregate",
                scope=node,
                value_gbs=float(ring),
                tool=P2P_TOOL,
                command=command,
                details={**common, "per_gpu_gbs": float(ring) / count},
            )
        )
    for test, key in (("h2d", "h2d_gbs"), ("d2h", "d2h_gbs")):
        per_gpu = list(obj.get(key) or [])
        values = [float(v) for v in per_gpu if v is not None]
        if values:
            results.append(
                PeakResult(
                    plane="pcie",
                    test=test,
                    scope=node,
                    value_gbs=statistics.median(values),
                    tool=P2P_TOOL,
                    command=command,
                    details={
                        **common,
                        "statistic": "median_over_gpus",
                        **_stats(values),
                        "per_gpu_gbs": per_gpu,
                    },
                )
            )
    return results


def _third(item: tuple[int, int, float]) -> float:
    return item[2]


def _busbw(row: Mapping[str, Any]) -> float:
    return float(row["busbw_gbs"])


def parse_nccl(
    obj: Mapping[str, Any],
    scope: str,
    *,
    intra_plane: Plane = "nvlink",
    nic_count: int | None = None,
    nominal_gbs: float | None = None,
    command: str = "",
    details: Mapping[str, Any] | None = None,
) -> list[PeakResult]:
    """``nccl_bench.py`` JSON -> peaks.

    collective mode: one ``nccl_<op>_<intra|inter>`` result per op, valued at
    the best bus bandwidth over sizes (every size in details). pairs mode: one
    ``gdr_send_recv`` result valued at the aggregate unidirectional GB/s.
    """
    _require_kind(obj, "nccl")
    nnodes = int(obj["nnodes"])
    plane: Plane = "ib" if nnodes > 1 else intra_plane
    common = {
        "world_size": int(obj["world_size"]),
        "nnodes": nnodes,
        "env": dict(obj.get("env") or {}),
        "torch": obj.get("torch"),
        "nccl": obj.get("nccl"),
        **(details or {}),
    }
    mode = obj.get("mode")
    if mode == "pairs":
        pairs = obj["pairs"]
        return [
            PeakResult(
                plane=plane,
                test="gdr_send_recv",
                scope=scope,
                nic_count=nic_count,
                value_gbs=float(pairs["aggregate_gbs"]),
                nominal_gbs=nominal_gbs,
                tool=NCCL_TOOL,
                command=command,
                details={
                    **common,
                    "statistic": "sum_over_pairs_unidirectional",
                    "size_bytes": pairs.get("size_bytes"),
                    "per_rank_gbs": list(pairs["per_rank_gbs"]),
                    "pair_gbs": list(pairs.get("pair_gbs") or []),
                    "bidirectional_gbs": pairs.get("bidirectional_gbs"),
                    "per_node_gbs": pairs.get("per_node_gbs"),
                },
            )
        ]
    if mode != "collective":
        raise ValueError(f"unknown nccl_bench mode {mode!r}")
    where = "inter" if nnodes > 1 else "intra"
    by_op: dict[str, list[dict[str, Any]]] = {}
    for row in obj["results"]:
        by_op.setdefault(str(row["op"]), []).append(dict(row))
    results = []
    for op, rows in by_op.items():
        best = max(rows, key=_busbw)
        results.append(
            PeakResult(
                plane=plane,
                test=f"nccl_{op}_{where}",
                scope=scope,
                nic_count=nic_count,
                value_gbs=float(best["busbw_gbs"]),
                nominal_gbs=nominal_gbs,
                tool=NCCL_TOOL,
                command=command,
                details={
                    **common,
                    "op": op,
                    "statistic": "max_busbw_over_sizes",
                    "best_size_bytes": int(best["size_bytes"]),
                    "algbw_gbs": float(best["algbw_gbs"]),
                    "by_size": rows,
                },
            )
        )
    return results


_NCCL_USING_RE = re.compile(r"NET/IB\s*:\s*Using\s+(?P<devs>[^;\n]*)")
_NCCL_DEV_RE = re.compile(r"\[\d+\](?P<name>[\w.+-]+?):\d+/(?:IB|RoCE)")
_NCCL_CHANNEL_RE = re.compile(r"via NET/(?P<net>\w+)/\d+(?:\(\d+\))?(?P<gdr>/GDRDMA)?")


@dataclass(frozen=True)
class NcclNetEvidence:
    """What NCCL_DEBUG=INFO says a process actually used."""

    devices: tuple[str, ...]  # HCAs from "NET/IB : Using ..."
    gdr_connections: int  # "via NET/IB/<n>/GDRDMA"
    staged_connections: int  # IB connections without GPU-direct RDMA
    socket_connections: int  # fell back to NET/Socket

    def as_dict(self) -> dict[str, Any]:
        return {
            "devices": list(self.devices),
            "gdr_connections": self.gdr_connections,
            "staged_connections": self.staged_connections,
            "socket_connections": self.socket_connections,
        }


def parse_nccl_net_log(text: str) -> NcclNetEvidence:
    """Extract the HCAs and GPU-direct use from NCCL INFO output."""
    devices: set[str] = set()
    for match in _NCCL_USING_RE.finditer(text):
        for dev in _NCCL_DEV_RE.finditer(match.group("devs")):
            devices.update(dev.group("name").split("+"))
    gdr = staged = socket_count = 0
    for match in _NCCL_CHANNEL_RE.finditer(text):
        if match.group("net") == "Socket":
            socket_count += 1
        elif match.group("gdr"):
            gdr += 1
        else:
            staged += 1
    return NcclNetEvidence(tuple(sorted(devices)), gdr, staged, socket_count)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerJob:
    """One foreground benchmark container on one node."""

    node: str
    name: str
    argv: tuple[str, ...]
    out_dir: str

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


class MicrobenchRunner:
    """Runs the microbenchmark agents in containers and parses their output.

    ``control_ips`` maps node -> control-network (wg0) address; rank 0 of a
    multi-node job rendezvous on the first node's address. Results are read
    back with ``Cluster.get_file``; with ``local_dir`` set, the raw JSON and
    container logs are kept there as ``<node>/<file>``.
    """

    def __init__(
        self,
        cluster: Cluster,
        docker: DockerHost,
        image: str,
        workdir: str,
        control_ips: Mapping[str, str],
        control_iface: str = "wg0",
        *,
        gpu_counts: Mapping[str, int] | None = None,
        nvlink: Mapping[str, bool] | None = None,
        hca_rates_gbps: Mapping[str, Mapping[str, float]] | None = None,
        local_dir: Path | None = None,
        port_base: int = DEFAULT_PORT_BASE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        labels: Mapping[str, str] | None = None,
    ):
        self._cluster = cluster
        self._docker = docker
        self._image = image
        self._src_dir = f"{workdir.rstrip('/')}/src"
        self._ips = dict(control_ips)
        self._iface = control_iface
        self._gpu_counts = dict(gpu_counts or {})
        self._nvlink = dict(nvlink or {})
        self._inferred_nvlink: dict[str, bool] = {}
        self._hca_rates = {
            node: dict(rates) for node, rates in (hca_rates_gbps or {}).items()
        }
        self._local_dir = local_dir
        self._port_base = port_base
        self._timeout_s = timeout_s
        self._labels = {LABEL_ROLE: MICROBENCH_ROLE, **(labels or {})}

    @classmethod
    def from_inventory(
        cls,
        cluster: Cluster,
        docker: DockerHost,
        image: str,
        workdir: str,
        inventory: Inventory,
        **kwargs: Any,
    ) -> MicrobenchRunner:
        """Fill addresses, GPU counts, NVLink presence and HCA rates from a probe."""
        ifaces = {node.control_interface for node in inventory.nodes}
        if len(ifaces) != 1:
            raise ValueError(
                f"nodes disagree on the control interface: {sorted(ifaces)}"
            )
        rates = {
            node.name: {
                hca.name: max(p.rate_gbps for p in hca.active_ports)
                for hca in node.active_hcas
            }
            for node in inventory.nodes
        }
        return cls(
            cluster,
            docker,
            image,
            workdir,
            {node.name: node.address for node in inventory.nodes},
            ifaces.pop(),
            gpu_counts={node.name: len(node.gpus) for node in inventory.nodes},
            nvlink={node.name: node.nvlink_active for node in inventory.nodes},
            hca_rates_gbps=rates,
            **kwargs,
        )

    # ---------------------------------------------------------------- jobs

    def _gpus(self, node: str) -> list[int]:
        return list(range(self._gpu_counts.get(node, DEFAULT_GPUS_PER_NODE)))

    def _ip(self, node: str) -> str:
        try:
            return self._ips[node]
        except KeyError as exc:
            raise KeyError(f"no control address for {node!r}") from exc

    def _job(
        self,
        node: str,
        name: str,
        command: Sequence[str],
        out: str,
        env: Mapping[str, str],
    ) -> ContainerJob:
        out_dir = posixpath.dirname(out)
        argv = build_microbench_docker(
            name=name,
            image=self._image,
            command=command,
            gpus=self._gpus(node),
            src_dir=self._src_dir,
            out_dir=out_dir,
            env=env,
            labels=self._labels,
        )
        return ContainerJob(node=node, name=name, argv=tuple(argv), out_dir=out_dir)

    def _nccl_env(self, *, cross_node: bool) -> dict[str, str]:
        env = {"NCCL_SOCKET_IFNAME": self._iface, "NCCL_DEBUG": "WARN"}
        if cross_node:
            env["NCCL_NET"] = "IB"
        return env

    def p2p_job(
        self,
        node: str,
        remote_out: str,
        *,
        size_mib: int = 1024,
        iters: int = 10,
        warmup: int = 3,
    ) -> ContainerJob:
        command = build_p2p_command(
            out=remote_out, node=node, size_mib=size_mib, iters=iters, warmup=warmup
        )
        return self._job(node, "cb-mb-p2p", command, remote_out, {})

    def intra_job(
        self,
        node: str,
        remote_out: str,
        *,
        ops: Sequence[str] = DEFAULT_OPS,
        sizes: Sequence[str] = DEFAULT_SIZES,
        iters: int = 20,
        warmup: int = 5,
    ) -> ContainerJob:
        args = build_nccl_collective_args(
            out=remote_out, ops=ops, sizes=sizes, iters=iters, warmup=warmup
        )
        command = build_torchrun_command(
            NCCL_SCRIPT, args, nproc_per_node=len(self._gpus(node))
        )
        return self._job(
            node,
            "cb-mb-nccl-intra",
            command,
            remote_out,
            self._nccl_env(cross_node=False),
        )

    def _multinode_jobs(
        self,
        nodes: Sequence[str],
        name: str,
        script_args: Sequence[str],
        out: str,
        port_key: str,
        env_for: Mapping[str, Mapping[str, str]],
    ) -> dict[str, ContainerJob]:
        counts = {len(self._gpus(node)) for node in nodes}
        if len(counts) != 1:
            raise ValueError(
                f"nodes {list(nodes)} have different GPU counts {sorted(counts)}"
            )
        nproc = counts.pop()
        port = rendezvous_port(self._port_base, port_key)
        master = self._ip(nodes[0])
        jobs = {}
        for rank, node in enumerate(nodes):
            command = build_torchrun_command(
                NCCL_SCRIPT,
                script_args,
                nproc_per_node=nproc,
                nnodes=len(nodes),
                node_rank=rank,
                master_addr=master,
                master_port=port,
            )
            jobs[node] = self._job(node, name, command, out, env_for[node])
        return jobs

    def inter_jobs(
        self,
        nodes: Sequence[str],
        remote_out: str,
        *,
        ops: Sequence[str] = DEFAULT_OPS,
        sizes: Sequence[str] = DEFAULT_SIZES,
        iters: int = 20,
        warmup: int = 5,
    ) -> dict[str, ContainerJob]:
        """One container per node; rank 0 (first node) writes ``remote_out``."""
        if len(nodes) < 2:
            raise ValueError("inter-node collectives need at least two nodes")
        args = build_nccl_collective_args(
            out=remote_out, ops=ops, sizes=sizes, iters=iters, warmup=warmup
        )
        env = self._nccl_env(cross_node=True)
        return self._multinode_jobs(
            nodes,
            "cb-mb-nccl-inter",
            args,
            remote_out,
            "inter|" + ",".join(nodes),
            {n: env for n in nodes},
        )

    def gdr_jobs(
        self,
        node_a: str,
        node_b: str,
        hcas: Mapping[str, Sequence[str]],
        remote_out: str,
        *,
        size: str = "1G",
        iters: int = 20,
        warmup: int = 5,
    ) -> dict[str, ContainerJob]:
        """Pairs-mode containers with ``NCCL_IB_HCA`` restricted per node."""
        count = len(hcas[node_a])
        env_for = {}
        for node in (node_a, node_b):
            env_for[node] = {
                **self._nccl_env(cross_node=True),
                "NCCL_IB_HCA": nccl_ib_hca_value(hcas[node]),
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,NET",
            }
        args = build_nccl_pairs_args(
            out=remote_out, size=size, iters=iters, warmup=warmup
        )
        key = f"gdr|{node_a}|{node_b}|{count}"
        return self._multinode_jobs(
            (node_a, node_b), f"cb-mb-gdr-{count}nic", args, remote_out, key, env_for
        )

    # ------------------------------------------------------------ execution

    def _keep(self, node: str, name: str, text: str) -> None:
        if self._local_dir is None:
            return
        target = self._local_dir / node / name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            # Keeping a log copy is best effort; never fail a measurement on it.
            logger.warning("could not keep %s for %s: %s", name, node, exc)

    def _run_job(self, job: ContainerJob) -> str:
        """Run a foreground container; returns its stdout (raises on failure)."""
        remove = f"docker rm -f {shlex.quote(job.name)} >/dev/null 2>&1 || true"
        self._cluster.run(
            job.node, f"mkdir -p {shlex.quote(job.out_dir)} && {{ {remove}; }}"
        )
        logger.info("[%s] microbenchmark %s", job.node, job.name)
        try:
            output = self._docker.run(job.node, job.argv, timeout=self._timeout_s)
        except CommandError as exc:
            # A timed-out client leaves the container (and its GPUs) busy.
            self._keep(job.node, f"{job.name}.log", exc.stderr)
            self._cluster.run(job.node, remove, check=False)
            raise
        self._keep(job.node, f"{job.name}.log", output)
        return output

    def _run_group(self, jobs: Mapping[str, ContainerJob]) -> dict[str, str]:
        return self._cluster.fanout(list(jobs), lambda node: self._run_job(jobs[node]))

    def _fetch_json(self, node: str, remote: str) -> dict[str, Any]:
        name = posixpath.basename(remote)
        with tempfile.TemporaryDirectory(prefix="cb-mb-") as tmp:
            local = Path(tmp) / name
            self._cluster.get_file(node, remote, local)
            text = local.read_text(encoding="utf-8")
        self._keep(node, name, text)
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"[{node}] {remote}: expected a JSON object")
        return obj

    def _intra_plane(self, node: str) -> Plane:
        nvlink = self._nvlink.get(node, self._inferred_nvlink.get(node))
        if nvlink is None:
            raise ValueError(
                f"NVLink presence of {node!r} is unknown: "
                "pass nvlink= or run p2p() first"
            )
        return "nvlink" if nvlink else "pcie"

    def _scope(self, nodes: Sequence[str]) -> str:
        return "cluster" if set(nodes) == set(self._cluster.nodes) else "+".join(nodes)

    def _nominal_ib_gbs(self, node: str, hcas: Sequence[str]) -> float | None:
        rates = self._hca_rates.get(node)
        if not rates or any(h not in rates for h in hcas):
            return None
        return sum(rates[h] for h in hcas) / 8.0

    def p2p(
        self,
        node: str,
        remote_out: str,
        *,
        size_mib: int = 1024,
        iters: int = 10,
        warmup: int = 3,
        nvlink: bool | None = None,
    ) -> list[PeakResult]:
        """GPU<->GPU copy matrix, ring aggregate and pinned H2D/D2H on one node.

        ``nvlink`` overrides the inventory's plane (False: NVLink switched off).
        """
        job = self.p2p_job(
            node, remote_out, size_mib=size_mib, iters=iters, warmup=warmup
        )
        self._run_job(job)
        obj = self._fetch_json(node, remote_out)
        source = "caller"
        if nvlink is None:
            nvlink = self._nvlink.get(node)
            source = "inventory"
        if nvlink is None:
            nvlink = infer_nvlink(obj)
            self._inferred_nvlink[node] = nvlink
            source = "inferred_from_bandwidth"
        return parse_p2p(
            obj, nvlink, command=job.command, details={"plane_source": source}
        )

    def collectives(
        self,
        nodes: Sequence[str],
        remote_out: str,
        *,
        intra: bool = True,
        inter: bool = True,
        ops: Sequence[str] = DEFAULT_OPS,
        sizes: Sequence[str] = DEFAULT_SIZES,
        iters: int = 20,
        warmup: int = 5,
    ) -> list[PeakResult]:
        """all_reduce / all_gather bus bandwidth within each node and across nodes.

        Intra-node jobs run on every node at once and write
        ``output_path(remote_out, "intra-<node>")``; the all-GPU inter-node job
        writes ``output_path(remote_out, "inter")`` on the first node.
        """
        results: list[PeakResult] = []
        if intra:
            planes = {node: self._intra_plane(node) for node in nodes}  # fail fast
            jobs = {
                node: self.intra_job(
                    node,
                    output_path(remote_out, f"intra-{node}"),
                    ops=ops,
                    sizes=sizes,
                    iters=iters,
                    warmup=warmup,
                )
                for node in nodes
            }
            self._run_group(jobs)
            for node in nodes:
                obj = self._fetch_json(node, output_path(remote_out, f"intra-{node}"))
                results += parse_nccl(
                    obj, node, intra_plane=planes[node], command=jobs[node].command
                )
        if inter and len(nodes) > 1:
            out = output_path(remote_out, "inter")
            jobs = self.inter_jobs(
                nodes, out, ops=ops, sizes=sizes, iters=iters, warmup=warmup
            )
            self._run_group(jobs)
            obj = self._fetch_json(nodes[0], out)
            results += parse_nccl(
                obj, self._scope(nodes), command=jobs[nodes[0]].command
            )
        return results

    def gdr_pairs(
        self,
        node_a: str,
        node_b: str,
        nic_subsets: Mapping[int, Mapping[str, Sequence[str]]],
        remote_out: str,
        *,
        size: str = "1G",
        iters: int = 20,
        warmup: int = 5,
    ) -> list[PeakResult]:
        """GPU-direct RDMA send/recv between two nodes for each NIC count.

        ``nic_subsets[n][node]`` lists the HCAs node may use with n NICs.
        Engagement is checked from NCCL INFO logs: the HCAs NCCL reports using
        must equal the subset, with no socket fallback.
        """
        for count in nic_subsets:  # programming errors fail fast, before any run
            for node in (node_a, node_b):
                subset = tuple(nic_subsets[count][node])
                if len(subset) != count:
                    raise ValueError(
                        f"{node}: {count}-NIC subset lists {len(subset)} HCAs: {subset}"
                    )
        results: list[PeakResult] = []
        for count in sorted(nic_subsets, reverse=True):
            try:
                results += self._gdr_count(
                    node_a, node_b, count, nic_subsets, remote_out, size, iters, warmup
                )
            except (CommandError, ValueError, KeyError, FileNotFoundError) as exc:
                # One failing NIC count must not discard the counts that worked:
                # each is the speed-of-light reference of its own NIC-sweep arm.
                logger.warning("GPU-direct %d-NIC benchmark failed: %s", count, exc)
        return results

    def _gdr_count(
        self,
        node_a: str,
        node_b: str,
        count: int,
        nic_subsets: Mapping[int, Mapping[str, Sequence[str]]],
        remote_out: str,
        size: str,
        iters: int,
        warmup: int,
    ) -> list[PeakResult]:
        hcas = {node: tuple(nic_subsets[count][node]) for node in (node_a, node_b)}
        out = output_path(remote_out, f"{count}nic")
        jobs = self.gdr_jobs(
            node_a, node_b, hcas, out, size=size, iters=iters, warmup=warmup
        )
        logs = self._run_group(jobs)
        evidence = {node: parse_nccl_net_log(logs[node]) for node in (node_a, node_b)}
        engaged: bool | None = None
        if all(ev.devices for ev in evidence.values()):
            engaged = all(
                set(ev.devices) == set(hcas[node]) and ev.socket_connections == 0
                for node, ev in evidence.items()
            )
        obj = self._fetch_json(node_a, out)
        return parse_nccl(
            obj,
            f"{node_a}<->{node_b}",
            nic_count=count,
            nominal_gbs=self._nominal_ib_gbs(node_a, hcas[node_a]),
            command=jobs[node_a].command,
            details={
                "hcas": {node: list(subset) for node, subset in hcas.items()},
                "nccl_net": {node: ev.as_dict() for node, ev in evidence.items()},
                "engaged": engaged,
            },
        )
