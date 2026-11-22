#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Intra-node GPU copy bandwidth: P2P matrix, ring aggregate, host<->device.

Runs inside the serving container (torch + CUDA) with every GPU of one node
visible::

    python3 p2p_bench.py --out PATH --node NAME --size-mib 1024 --iters 10 --warmup 3

* ``copy_gbs[i][j]``: unidirectional ``dst.copy_(src, non_blocking=True)``
  from GPU i to GPU j on a stream of the source device. PyTorch enables peer
  access itself when ``can_access_peer`` allows it; otherwise the driver stages
  the copy through host memory. ``null`` when the copy fails.
* ``ring_aggregate_gbs``: every GPU i copies to GPU (i+1) % N at the same time;
  total bytes over the slowest stream's elapsed time.
* ``h2d_gbs`` / ``d2h_gbs``: pinned host memory <-> each GPU, one GPU at a time.

All timings use CUDA events; 1 GB = 1e9 bytes. Output JSON (``--out``)::

    {"kind": "p2p", "node", "gpu_count", "size_bytes", "iters", "warmup",
     "can_access_peer": [[bool]], "copy_gbs": [[float|null]],
     "ring_aggregate_gbs": float|null, "h2d_gbs": [float|null],
     "d2h_gbs": [float|null], "torch": str, "cuda": str, "errors": [str]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

import torch

GB = 1e9
MIB = 1 << 20


def gbs(nbytes: int, seconds: float) -> float:
    """Bandwidth in GB/s (1 GB = 1e9 bytes)."""
    if seconds <= 0:
        raise ValueError(f"non-positive duration {seconds!r}")
    return nbytes / seconds / GB


def ring_peer(index: int, count: int) -> int:
    """Destination GPU of ``index`` in the ring pattern."""
    return (index + 1) % count


def ordered_pairs(count: int) -> List[tuple]:
    """Every ordered (src, dst) pair of distinct GPUs, row-major."""
    return [(i, j) for i in range(count) for j in range(count) if i != j]


def timed_copy(
    stream: torch.cuda.Stream,
    dst: torch.Tensor,
    src: torch.Tensor,
    iters: int,
    warmup: int,
) -> float:
    """Seconds per ``dst.copy_(src)`` enqueued on ``stream``, by CUDA events."""
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            dst.copy_(src, non_blocking=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def ring_aggregate(
    srcs: Sequence[torch.Tensor],
    dsts: Sequence[torch.Tensor],
    streams: Sequence[torch.cuda.Stream],
    iters: int,
    warmup: int,
) -> float:
    """GB/s with every GPU copying to its ring successor simultaneously."""
    count = len(srcs)

    def launch() -> None:
        for index in range(count):
            with torch.cuda.stream(streams[index]):
                dsts[ring_peer(index, count)].copy_(srcs[index], non_blocking=True)

    for _ in range(warmup):
        launch()
    for index in range(count):
        torch.cuda.synchronize(index)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for index in range(count):
        starts[index].record(streams[index])
    for _ in range(iters):
        launch()
    for index in range(count):
        ends[index].record(streams[index])
    for event in ends:
        event.synchronize()
    slowest = max(starts[i].elapsed_time(ends[i]) for i in range(count)) / 1000.0
    nbytes = srcs[0].numel() * srcs[0].element_size()
    return gbs(nbytes * iters * count, slowest)


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


def measure(node: str, size_mib: int, iters: int, warmup: int) -> dict:
    """Run every measurement on the visible GPUs and return the JSON payload."""
    count = torch.cuda.device_count()
    nbytes = size_mib * MIB
    srcs = [
        torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{i}") for i in range(count)
    ]
    dsts = [
        torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{i}") for i in range(count)
    ]
    streams = [torch.cuda.Stream(device=i) for i in range(count)]
    errors: List[str] = []

    peer = [[False] * count for _ in range(count)]
    copy: List[List[Optional[float]]] = [[None] * count for _ in range(count)]
    for src, dst in ordered_pairs(count):
        peer[src][dst] = bool(torch.cuda.can_device_access_peer(src, dst))
        try:
            seconds = timed_copy(streams[src], dsts[dst], srcs[src], iters, warmup)
        except RuntimeError as exc:  # recorded as null; the pair is reported failed
            errors.append(f"copy {src}->{dst}: {exc}")
            continue
        copy[src][dst] = gbs(nbytes, seconds)

    ring: Optional[float] = None
    if count > 1:
        try:
            ring = ring_aggregate(srcs, dsts, streams, iters, warmup)
        except RuntimeError as exc:
            errors.append(f"ring: {exc}")

    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    h2d: List[Optional[float]] = [None] * count
    d2h: List[Optional[float]] = [None] * count
    for index in range(count):
        try:
            h2d[index] = gbs(
                nbytes, timed_copy(streams[index], srcs[index], host, iters, warmup)
            )
            d2h[index] = gbs(
                nbytes, timed_copy(streams[index], host, srcs[index], iters, warmup)
            )
        except RuntimeError as exc:
            errors.append(f"host copy gpu{index}: {exc}")

    return {
        "kind": "p2p",
        "node": node,
        "gpu_count": count,
        "size_bytes": nbytes,
        "iters": iters,
        "warmup": warmup,
        "can_access_peer": peer,
        "copy_gbs": copy,
        "ring_aggregate_gbs": ring,
        "h2d_gbs": h2d,
        "d2h_gbs": d2h,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "errors": errors,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--node", required=True, help="logical node name")
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args(argv)
    if args.size_mib <= 0 or args.iters <= 0 or args.warmup < 0:
        parser.error("--size-mib and --iters must be positive, --warmup >= 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("p2p_bench: CUDA is not available", file=sys.stderr)
        return 2
    payload = measure(args.node, args.size_mib, args.iters, args.warmup)
    write_json_atomic(args.out, payload)
    copies = [v for row in payload["copy_gbs"] for v in row if v is not None]
    print(
        f"p2p_bench {args.node}: {payload['gpu_count']} GPUs, "
        f"p2p {min(copies, default=0):.1f}-{max(copies, default=0):.1f} GB/s, "
        f"ring {payload['ring_aggregate_gbs']}, errors {len(payload['errors'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
