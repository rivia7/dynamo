#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fill in model files across nodes with NCCL broadcasts over the RDMA fabric.

Launched by torchrun with one GPU per node::

    torchrun --nnodes N --nproc-per-node 1 --node-rank R --master-addr A \\
        --master-port P file_gather.py --manifest M --report 'R-{rank}.json' \\
        --chunk-mib 256

Manifest (identical on every node)::

    {"root": "<model dir>",
     "files": [{"path": "<rel>", "size": int, "sha256": str|null, "owner": rank}]}

Ranks first agree on which files each of them already holds (present with the
manifest size). For every file in manifest order that some rank lacks, all
ranks then call ``dist.broadcast`` chunk by chunk through a device staging
buffer: the source (the owner, or the lowest rank holding the file when the
owner does not) streams disk -> pinned -> GPU; ranks lacking the file stream
GPU -> pinned -> ``<path>.cb_partial`` (fsync, then rename); ranks that
already hold it take part in the collective and discard the bytes. A reader
thread (source side) and a writer thread (receiver side), each with two
alternating pinned buffers, overlap disk I/O with the transfers.

Every rank then sha256-hashes every file it holds - a thread pool re-reads the
fsync'ed files, received ones while the transfers continue - and checks the
size and, when the manifest has one, the sha256. A corrupt file is renamed to
``<path>.cb_corrupt`` so a rerun fetches it again. Each rank writes::

    {"rank", "hostname", "world_size", "root", "files": [{"path", "size",
     "action": "sent"|"received"|"present"|"missing", "sha256",
     "expected_sha256", "ok", "reason"}], "bytes_received", "bytes_sent",
     "bytes_discarded", "bytes_broadcast", "seconds", "verify_seconds",
     "total_seconds", "gbs", "mismatches": [path], "missing": [path]}

``gbs`` is GB/s (1e9 bytes/s) of the broadcast phase. The exit status is 1
when any file is missing or mismatched. ``--device cpu`` (gloo, no GPU) only
exists for local tests.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import dataclasses
import datetime
import hashlib
import json
import os
import socket
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

MIB = 1 << 20
GB = 1e9
PARTIAL_SUFFIX = ".cb_partial"
CORRUPT_SUFFIX = ".cb_corrupt"
HASH_BLOCK = 16 * MIB
SLOTS = 2  # alternating pinned buffers per direction


@dataclasses.dataclass(frozen=True)
class FileEntry:
    path: str  # relative to the model root, "/"-separated
    size: int
    sha256: Optional[str]
    owner: int


@dataclasses.dataclass(frozen=True)
class Transfer:
    index: int  # position in the manifest
    src: Optional[int]  # broadcasting rank; None when no rank holds the file
    receivers: Tuple[int, ...]  # ranks lacking the file


@dataclasses.dataclass
class Stats:
    bytes_sent: int = 0
    bytes_received: int = 0
    bytes_discarded: int = 0
    bytes_broadcast: int = 0


# --------------------------------------------------------------------------
# Pure helpers (no torch)
# --------------------------------------------------------------------------


def safe_join(root: str, rel: str) -> str:
    """``root/rel`` for a manifest path; rejects absolute or escaping paths."""
    if not rel or rel.startswith("/") or "\\" in rel:
        raise ValueError(f"unsafe manifest path {rel!r}")
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe manifest path {rel!r}")
    return os.path.join(root, *parts)


def iter_chunks(size: int, chunk: int) -> Iterator[Tuple[int, int]]:
    """(offset, length) pieces covering ``size`` bytes; none for an empty file."""
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    offset = 0
    while offset < size:
        length = min(chunk, size - offset)
        yield offset, length
        offset += length


def parse_manifest(raw: Any, world_size: int) -> Tuple[str, List[FileEntry]]:
    """Validate a manifest object; owners must be ranks of this world."""
    root = raw["root"]
    if not isinstance(root, str) or not root:
        raise ValueError("manifest root must be a non-empty string")
    entries: List[FileEntry] = []
    seen = set()
    for item in raw["files"]:
        sha = item.get("sha256")
        entry = FileEntry(
            path=str(item["path"]),
            size=int(item["size"]),
            sha256=str(sha).lower() if sha else None,
            owner=int(item["owner"]),
        )
        safe_join(root, entry.path)
        if entry.size < 0:
            raise ValueError(f"{entry.path}: negative size")
        if not 0 <= entry.owner < world_size:
            raise ValueError(
                f"{entry.path}: owner {entry.owner} outside world {world_size}"
            )
        if entry.path in seen:
            raise ValueError(f"{entry.path}: listed twice")
        seen.add(entry.path)
        entries.append(entry)
    return root, entries


def manifest_digest(entries: Sequence[FileEntry]) -> int:
    """Signed 64-bit digest of the file list, compared across ranks."""
    blob = json.dumps(
        [[e.path, e.size, e.sha256, e.owner] for e in entries], separators=(",", ":")
    )
    return int.from_bytes(
        hashlib.sha256(blob.encode()).digest()[:8], "big", signed=True
    )


def holds(root: str, entry: FileEntry) -> bool:
    """True when the file exists with the manifest size."""
    try:
        return os.stat(safe_join(root, entry.path)).st_size == entry.size
    except FileNotFoundError:
        return False


def plan_transfers(
    entries: Sequence[FileEntry], have: Sequence[Sequence[bool]]
) -> List[Transfer]:
    """Source and receivers of every file; identical on every rank."""
    world = len(have)
    transfers = []
    for index, entry in enumerate(entries):
        holders = [rank for rank in range(world) if have[rank][index]]
        receivers = tuple(rank for rank in range(world) if not have[rank][index])
        if not holders:
            src = None
        elif entry.owner in holders:
            src = entry.owner
        else:
            src = holders[0]
        transfers.append(Transfer(index=index, src=src, receivers=receivers))
    return transfers


def sha256_file(path: str, block: int = HASH_BLOCK) -> str:
    digest = hashlib.sha256()
    buf = memoryview(bytearray(block))
    with open(path, "rb", buffering=0) as fh:
        while True:
            got = fh.readinto(buf)
            if not got:
                break
            digest.update(buf[:got])
    return digest.hexdigest()


def verify_file(path: str, entry: FileEntry) -> Dict[str, Any]:
    """Size and sha256 check of a held file; a corrupt file is moved aside."""
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        return {"ok": False, "sha256": None, "reason": "missing"}
    digest: Optional[str] = None
    if size != entry.size:
        reason: Optional[str] = f"size {size} != {entry.size}"
    else:
        digest = sha256_file(path)
        reason = None
        if entry.sha256 is not None and digest != entry.sha256:
            reason = "sha256 mismatch"
    if reason is not None:
        os.replace(path, path + CORRUPT_SUFFIX)
        return {"ok": False, "sha256": digest, "reason": reason}
    return {"ok": True, "sha256": digest, "reason": None}


def build_report(
    *,
    rank: int,
    world_size: int,
    hostname: str,
    root: str,
    entries: Sequence[FileEntry],
    actions: Dict[int, str],
    outcomes: Dict[int, Dict[str, Any]],
    stats: Stats,
    seconds: float,
    verify_seconds: float,
    total_seconds: float,
) -> Dict[str, Any]:
    """The rank-local report (see the module docstring)."""
    files = []
    mismatches: List[str] = []
    missing: List[str] = []
    for index, entry in enumerate(entries):
        action = actions[index]
        if action == "missing":
            outcome: Dict[str, Any] = {
                "ok": False,
                "sha256": None,
                "reason": "no rank holds it",
            }
        else:
            outcome = outcomes[index]
        if not outcome["ok"]:
            if action == "missing" or outcome["reason"] == "missing":
                missing.append(entry.path)
            else:
                mismatches.append(entry.path)
        files.append(
            {
                "path": entry.path,
                "size": entry.size,
                "action": action,
                "sha256": outcome["sha256"],
                "expected_sha256": entry.sha256,
                "ok": outcome["ok"],
                "reason": outcome["reason"],
            }
        )
    broadcast = stats.bytes_broadcast
    return {
        "rank": rank,
        "hostname": hostname,
        "world_size": world_size,
        "root": root,
        "files": files,
        "bytes_received": stats.bytes_received,
        "bytes_sent": stats.bytes_sent,
        "bytes_discarded": stats.bytes_discarded,
        "bytes_broadcast": broadcast,
        "seconds": seconds,
        "verify_seconds": verify_seconds,
        "total_seconds": total_seconds,
        "gbs": broadcast / seconds / GB if broadcast and seconds > 0 else None,
        "mismatches": mismatches,
        "missing": missing,
    }


def write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Disk I/O (runs on the reader / writer threads)
# --------------------------------------------------------------------------


def read_into(path: str, offset: int, view: memoryview) -> None:
    with open(path, "rb", buffering=0) as fh:
        fh.seek(offset)
        filled = 0
        while filled < len(view):
            got = fh.readinto(view[filled:])
            if not got:
                raise EOFError(f"{path}: file ended at byte {offset + filled}")
            filled += got


def write_all(fd: int, view: memoryview) -> None:
    while len(view):
        written = os.write(fd, view)
        view = view[written:]


def fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def finalize(
    fd: int, partial: str, final: str, pool: ThreadPoolExecutor, entry: FileEntry
) -> Future:
    """fsync + rename a received file, then queue its verification."""
    os.fsync(fd)
    os.close(fd)
    os.replace(partial, final)
    fsync_dir(os.path.dirname(final))
    return pool.submit(verify_file, final, entry)


# --------------------------------------------------------------------------
# Transfers
# --------------------------------------------------------------------------


def host_view(tensor: torch.Tensor) -> memoryview:
    """A writable byte view of a CPU tensor's memory (no numpy needed)."""
    nbytes = tensor.numel() * tensor.element_size()
    array = (ctypes.c_ubyte * nbytes).from_address(tensor.data_ptr())
    return memoryview(array).cast("B")


class Staging:
    """The device staging buffer plus pinned host buffers per direction."""

    def __init__(self, device: torch.device, chunk_bytes: int):
        pin = device.type == "cuda"
        self.device = device
        self.buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
        self.send = [
            torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=pin)
            for _ in range(SLOTS)
        ]
        self.recv = [
            torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=pin)
            for _ in range(SLOTS)
        ]
        self.send_views = [host_view(t) for t in self.send]
        self.recv_views = [host_view(t) for t in self.recv]

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()


class ChunkReader:
    """Reads this rank's outgoing chunks ahead of the broadcasts."""

    def __init__(
        self,
        jobs: Sequence[Tuple[str, int, int]],
        views: Sequence[memoryview],
        pool: ThreadPoolExecutor,
    ):
        self._jobs = iter(jobs)
        self._views = views
        self._pool = pool
        self._pending: Deque[Tuple[int, int, Future]] = collections.deque()
        for slot in range(len(views)):
            self._schedule(slot)

    def _schedule(self, slot: int) -> None:
        job = next(self._jobs, None)
        if job is None:
            return
        path, offset, length = job
        future = self._pool.submit(read_into, path, offset, self._views[slot][:length])
        self._pending.append((slot, length, future))

    def take(self, length: int) -> int:
        """Slot holding the next outgoing chunk (which must be ``length`` bytes)."""
        slot, queued, future = self._pending.popleft()
        if queued != length:
            raise RuntimeError(
                f"reader out of step: queued {queued} bytes, expected {length}"
            )
        future.result()
        return slot

    def release(self, slot: int) -> None:
        self._schedule(slot)


def outgoing_jobs(
    root: str,
    entries: Sequence[FileEntry],
    transfers: Sequence[Transfer],
    rank: int,
    chunk: int,
) -> List[Tuple[str, int, int]]:
    jobs = []
    for transfer in transfers:
        if transfer.src == rank and transfer.receivers:
            entry = entries[transfer.index]
            path = safe_join(root, entry.path)
            jobs.extend(
                (path, offset, length)
                for offset, length in iter_chunks(entry.size, chunk)
            )
    return jobs


def transfer_all(
    root: str,
    entries: Sequence[FileEntry],
    transfers: Sequence[Transfer],
    rank: int,
    staging: Staging,
    verify_pool: ThreadPoolExecutor,
) -> Tuple[Dict[int, str], Dict[int, Future], Stats]:
    """Run every broadcast; returns actions, verification futures and byte counts."""
    chunk = staging.buf.numel()
    actions: Dict[int, str] = {}
    finals: Dict[int, Future] = {}
    stats = Stats()
    with ThreadPoolExecutor(
        1, thread_name_prefix="cb-read"
    ) as read_pool, ThreadPoolExecutor(1, thread_name_prefix="cb-write") as write_pool:
        reader = ChunkReader(
            outgoing_jobs(root, entries, transfers, rank, chunk),
            staging.send_views,
            read_pool,
        )
        pending_writes: List[Optional[Future]] = [None] * SLOTS
        received_chunks = 0
        for transfer in transfers:
            entry = entries[transfer.index]
            if transfer.src is None:
                actions[transfer.index] = "missing"
                continue
            if not transfer.receivers:
                actions[transfer.index] = "present"
                continue
            stats.bytes_broadcast += entry.size
            if transfer.src == rank:
                for _, length in iter_chunks(entry.size, chunk):
                    slot = reader.take(length)
                    staging.buf[:length].copy_(staging.send[slot][:length])
                    reader.release(slot)
                    dist.broadcast(staging.buf[:length], src=rank)
                stats.bytes_sent += entry.size
                actions[transfer.index] = "sent"
            elif rank in transfer.receivers:
                final = safe_join(root, entry.path)
                os.makedirs(os.path.dirname(final), exist_ok=True)
                partial = final + PARTIAL_SUFFIX
                fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                for _, length in iter_chunks(entry.size, chunk):
                    slot = received_chunks % SLOTS
                    received_chunks += 1
                    previous = pending_writes[slot]
                    if previous is not None:
                        previous.result()
                    dist.broadcast(staging.buf[:length], src=transfer.src)
                    staging.recv[slot][:length].copy_(staging.buf[:length])
                    pending_writes[slot] = write_pool.submit(
                        write_all, fd, staging.recv_views[slot][:length]
                    )
                finals[transfer.index] = write_pool.submit(
                    finalize, fd, partial, final, verify_pool, entry
                )
                stats.bytes_received += entry.size
                actions[transfer.index] = "received"
            else:
                for _, length in iter_chunks(entry.size, chunk):
                    dist.broadcast(staging.buf[:length], src=transfer.src)
                    staging.sync()
                stats.bytes_discarded += entry.size
                actions[transfer.index] = "present"
        staging.sync()
        verifications = {index: future.result() for index, future in finals.items()}
    return actions, verifications, stats


# --------------------------------------------------------------------------
# Distributed setup
# --------------------------------------------------------------------------


def init_distributed(
    device_kind: str, timeout_min: float
) -> Tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    timeout = datetime.timedelta(minutes=timeout_min)
    if device_kind == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timeout, device_id=device
        )
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", init_method="env://", timeout=timeout)
    return rank, world, device


def all_gather_ints(
    values: Sequence[int], world: int, device: torch.device, dtype: torch.dtype
) -> List[List[int]]:
    local = torch.tensor(list(values), dtype=dtype, device=device)
    parts = [torch.zeros(len(values), dtype=dtype, device=device) for _ in range(world)]
    dist.all_gather(parts, local)
    return [[int(v) for v in part.cpu().tolist()] for part in parts]


def exchange_have(
    have: Sequence[bool], world: int, device: torch.device
) -> List[List[bool]]:
    if not have:
        return [[] for _ in range(world)]
    rows = all_gather_ints([1 if h else 0 for h in have], world, device, torch.uint8)
    return [[bool(v) for v in row] for row in rows]


def check_same_manifest(
    entries: Sequence[FileEntry], world: int, device: torch.device
) -> None:
    digests = [
        row[0]
        for row in all_gather_ints(
            [manifest_digest(entries)], world, device, torch.int64
        )
    ]
    if len(set(digests)) != 1:
        raise RuntimeError(f"manifests differ across ranks: {digests}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--report", required=True, help="report path; '{rank}' is substituted"
    )
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--root", default=None, help="override the manifest root")
    parser.add_argument("--verify-workers", type=int, default=8)
    parser.add_argument("--timeout-min", type=float, default=15.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.chunk_mib <= 0 or args.verify_workers <= 0:
        parser.error("--chunk-mib and --verify-workers must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    with open(args.manifest, encoding="utf-8") as fh:
        root, entries = parse_manifest(json.load(fh), int(os.environ["WORLD_SIZE"]))
    if args.root:
        root = args.root
    rank, world, device = init_distributed(args.device, args.timeout_min)
    check_same_manifest(entries, world, device)
    have = exchange_have([holds(root, entry) for entry in entries], world, device)
    transfers = plan_transfers(entries, have)
    staging = Staging(device, args.chunk_mib * MIB)
    with ThreadPoolExecutor(
        args.verify_workers, thread_name_prefix="cb-verify"
    ) as verify_pool:
        transfer_start = time.monotonic()
        actions, verifications, stats = transfer_all(
            root, entries, transfers, rank, staging, verify_pool
        )
        seconds = time.monotonic() - transfer_start
        dist.barrier()
        for index, action in actions.items():
            if action in ("sent", "present"):
                path = safe_join(root, entries[index].path)
                verifications[index] = verify_pool.submit(
                    verify_file, path, entries[index]
                )
        outcomes = {index: future.result() for index, future in verifications.items()}
    verify_seconds = time.monotonic() - transfer_start - seconds
    report = build_report(
        rank=rank,
        world_size=world,
        hostname=socket.gethostname(),
        root=root,
        entries=entries,
        actions=actions,
        outcomes=outcomes,
        stats=stats,
        seconds=seconds,
        verify_seconds=verify_seconds,
        total_seconds=time.monotonic() - started,
    )
    write_json_atomic(args.report.format(rank=rank), report)
    dist.destroy_process_group()
    rate = f"{report['gbs']:.2f} GB/s" if report["gbs"] else "n/a"
    print(
        f"file_gather rank {rank}: received {stats.bytes_received} B, "
        f"sent {stats.bytes_sent} B in {seconds:.1f} s ({rate}); "
        f"mismatches {len(report['mismatches'])}, "
        f"missing {len(report['missing'])}"
    )
    return 1 if report["mismatches"] or report["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
