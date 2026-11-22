#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL speed of light: collectives and cross-node send/recv, one rank per GPU.

Launched by torchrun (env:// rendezvous, NCCL backend, device = LOCAL_RANK)::

    torchrun ... nccl_bench.py --mode collective --ops all_reduce,all_gather \\
        --sizes 64M,256M,1G --iters 20 --warmup 5 --out PATH
    torchrun ... nccl_bench.py --mode pairs --size 1G --iters 20 --out PATH

``collective``: each op and size is timed with CUDA events after a barrier;
the slowest rank's time counts. algbw = bytes / time and busbw follows
nccl-tests: all_reduce algbw * 2(n-1)/n; all_gather algbw * (n-1)/n where
bytes is the total output size.

``pairs``: the world is split into two halves by node (torchrun numbers ranks
node-major). Rank r of the first half exchanges ``--size`` bytes with rank
r + n/2 in both directions at once (``batch_isend_irecv``). Reported: per-rank
GB/s, per-pair unidirectional GB/s and their sum, the aggregate unidirectional
bandwidth across the node boundary. The launcher restricts ``NCCL_IB_HCA`` to
measure GPU-direct RDMA with 1/2/4/8 NICs.

Rank 0 writes ``--out``::

    {"kind": "nccl", "mode", "world_size", "nnodes", "local_world_size",
     "results": [{"op", "size_bytes", "time_us", "algbw_gbs", "busbw_gbs"}],
     "pairs": {"aggregate_gbs", "per_rank_gbs", "pair_gbs", ...} | null,
     "env": {"NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "NCCL_NET", ...},
     "torch", "nccl", "hostname", "iters", "warmup"}

Sizes take binary suffixes like nccl-tests (K/M/G = 2^10/2^20/2^30); 1 GB/s =
1e9 bytes/s.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

GB = 1e9
SIZE_SUFFIXES = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
SUPPORTED_OPS = ("all_reduce", "all_gather")
REPORTED_ENV = ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "NCCL_NET")
ELEMENT_BYTES = 4  # float32 payloads


def parse_size(text: str) -> int:
    """``"64M"`` -> 67108864; plain integers are bytes."""
    value = text.strip().upper()
    multiplier = 1
    if value and value[-1] in SIZE_SUFFIXES:
        multiplier = SIZE_SUFFIXES[value[-1]]
        value = value[:-1]
    number = int(value)
    if number <= 0:
        raise ValueError(f"size must be positive: {text!r}")
    return number * multiplier


def bus_factor(op: str, world_size: int) -> float:
    """nccl-tests bus-bandwidth correction factor."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if op == "all_reduce":
        return 2.0 * (world_size - 1) / world_size
    if op == "all_gather":
        return (world_size - 1) / world_size
    raise ValueError(f"unsupported op {op!r}; expected one of {SUPPORTED_OPS}")


def collective_elements(op: str, size_bytes: int, world_size: int) -> int:
    """float32 elements in each rank's input tensor for ``size_bytes``."""
    if op == "all_reduce":
        numel = size_bytes // ELEMENT_BYTES
    elif op == "all_gather":
        numel = size_bytes // (ELEMENT_BYTES * world_size)
    else:
        raise ValueError(f"unsupported op {op!r}; expected one of {SUPPORTED_OPS}")
    if numel <= 0:
        raise ValueError(
            f"{size_bytes} bytes is too small for {op} on {world_size} ranks"
        )
    return numel


def collective_row(op: str, size_bytes: int, seconds: float, world_size: int) -> dict:
    """One ``results`` entry; ``size_bytes`` is the op's nccl-tests size."""
    algbw = size_bytes / seconds / GB
    return {
        "op": op,
        "size_bytes": size_bytes,
        "time_us": seconds * 1e6,
        "algbw_gbs": algbw,
        "busbw_gbs": algbw * bus_factor(op, world_size),
    }


def node_count(
    rank: int, local_rank: int, group_rank: int, local_world_size: int, world_size: int
) -> int:
    """Validate torchrun's node-major rank layout and return the node count."""
    if local_world_size <= 0 or world_size % local_world_size:
        raise ValueError(
            f"world {world_size} is not a multiple of {local_world_size} ranks per node"
        )
    if rank != group_rank * local_world_size + local_rank:
        raise ValueError(
            f"rank {rank} is not node-major (node {group_rank}, local {local_rank}, "
            f"{local_world_size} per node)"
        )
    return world_size // local_world_size


def pair_peer(rank: int, world_size: int) -> int:
    """Partner of ``rank`` in the other half of the world."""
    if world_size < 2 or world_size % 2:
        raise ValueError(f"pairs mode needs an even world size, got {world_size}")
    half = world_size // 2
    return rank + half if rank < half else rank - half


def pairs_summary(rank_seconds: Sequence[float], size_bytes: int, nnodes: int) -> dict:
    """Per-rank, per-pair and aggregate GB/s from each rank's iteration time.

    A pair's unidirectional rate uses the slower of its two ranks; the
    aggregate is the sum over pairs (one direction across the node boundary).
    """
    world = len(rank_seconds)
    half = world // 2
    if world < 2 or world % 2:
        raise ValueError(f"pairs mode needs an even world size, got {world}")
    pairs = [[rank, rank + half] for rank in range(half)]
    pair_gbs = [
        size_bytes / max(rank_seconds[a], rank_seconds[b]) / GB for a, b in pairs
    ]
    aggregate = sum(pair_gbs)
    node_pairs = max(1, nnodes // 2)
    return {
        "size_bytes": size_bytes,
        "pairs": pairs,
        "time_us": [seconds * 1e6 for seconds in rank_seconds],
        "per_rank_gbs": [size_bytes / seconds / GB for seconds in rank_seconds],
        "pair_gbs": pair_gbs,
        "aggregate_gbs": aggregate,
        "bidirectional_gbs": 2.0 * aggregate,
        "per_node_gbs": aggregate / node_pairs,
    }


def env_snapshot(environ: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """The NCCL settings that shaped the measurement (always the three keys)."""
    snap: Dict[str, Optional[str]] = {key: environ.get(key) for key in REPORTED_ENV}
    for key, value in environ.items():
        if key.startswith("NCCL_"):
            snap[key] = value
    return dict(sorted(snap.items()))


def nccl_version() -> str:
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def timed(
    fn: Callable[[], None], iters: int, warmup: int, device: torch.device
) -> float:
    """This rank's seconds per call of ``fn``, by CUDA events after a barrier."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def gather_floats(value: float, world_size: int, device: torch.device) -> List[float]:
    local = torch.tensor([value], dtype=torch.float64, device=device)
    parts = [
        torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(parts, local)
    return [float(part.item()) for part in parts]


def make_collective(
    op: str, size_bytes: int, world_size: int, device: torch.device
) -> Tuple[Callable[[], None], int]:
    """A closure running one ``op`` and the nccl-tests byte count it moves."""
    numel = collective_elements(op, size_bytes, world_size)
    if op == "all_reduce":
        buf = torch.zeros(numel, dtype=torch.float32, device=device)

        def run_all_reduce() -> None:
            dist.all_reduce(buf)

        return run_all_reduce, numel * ELEMENT_BYTES
    inp = torch.zeros(numel, dtype=torch.float32, device=device)
    out = torch.empty(numel * world_size, dtype=torch.float32, device=device)

    def run_all_gather() -> None:
        dist.all_gather_into_tensor(out, inp)

    return run_all_gather, numel * world_size * ELEMENT_BYTES


def make_exchange(
    peer: int, size_bytes: int, device: torch.device
) -> Callable[[], None]:
    """A closure sending and receiving ``size_bytes`` with ``peer`` at once."""
    send = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
    recv = torch.empty_like(send)
    ops = [dist.P2POp(dist.isend, send, peer), dist.P2POp(dist.irecv, recv, peer)]

    def run() -> None:
        for request in dist.batch_isend_irecv(ops):
            request.wait()

    return run


def write_json_atomic(path: str, payload: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("collective", "pairs"), required=True)
    parser.add_argument("--out", required=True, help="JSON result path (rank 0)")
    parser.add_argument("--ops", default="all_reduce,all_gather")
    parser.add_argument("--sizes", default="64M,256M,1G")
    parser.add_argument("--size", default="1G", help="bytes per direction (pairs)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout-min", type=float, default=20.0)
    args = parser.parse_args(argv)
    args.op_list = [op.strip() for op in args.ops.split(",") if op.strip()]
    for op in args.op_list:
        if op not in SUPPORTED_OPS:
            parser.error(f"unsupported op {op!r}; expected one of {SUPPORTED_OPS}")
    args.size_list = [
        parse_size(size) for size in args.sizes.split(",") if size.strip()
    ]
    args.size_bytes = parse_size(args.size)
    if args.iters <= 0 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup >= 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    group_rank = int(os.environ.get("GROUP_RANK", "0"))
    nnodes = node_count(rank, local_rank, group_rank, local_world, world)
    if args.mode == "pairs" and (nnodes < 2 or nnodes % 2):
        raise ValueError(
            f"pairs mode needs an even number of nodes (>= 2), got {nnodes}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=args.timeout_min),
        device_id=device,
    )
    results: List[dict] = []
    pairs: Optional[dict] = None
    if args.mode == "collective":
        for op in args.op_list:
            for size in args.size_list:
                run, nbytes = make_collective(op, size, world, device)
                local = timed(run, args.iters, args.warmup, device)
                slowest = max(gather_floats(local, world, device))
                results.append(collective_row(op, nbytes, slowest, world))
                del run
                torch.cuda.empty_cache()
    else:
        run = make_exchange(pair_peer(rank, world), args.size_bytes, device)
        local = timed(run, args.iters, args.warmup, device)
        pairs = pairs_summary(
            gather_floats(local, world, device), args.size_bytes, nnodes
        )

    if rank == 0:
        payload = {
            "kind": "nccl",
            "mode": args.mode,
            "world_size": world,
            "nnodes": nnodes,
            "local_world_size": local_world,
            "results": results,
            "pairs": pairs,
            "env": env_snapshot(os.environ),
            "torch": torch.__version__,
            "nccl": nccl_version(),
            "hostname": socket.gethostname(),
            "iters": args.iters,
            "warmup": args.warmup,
        }
        write_json_atomic(args.out, payload)
        for row in results:
            print(
                f"nccl_bench {row['op']} {row['size_bytes']} B: "
                f"{row['time_us']:.0f} us algbw {row['algbw_gbs']:.1f} GB/s "
                f"busbw {row['busbw_gbs']:.1f} GB/s"
            )
        if pairs is not None:
            print(
                f"nccl_bench pairs: aggregate {pairs['aggregate_gbs']:.1f} GB/s "
                "unidirectional"
            )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
