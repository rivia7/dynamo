# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers of the torch agents, and a CPU (gloo) end-to-end model gather.

The agents import torch at module top, so this module is skipped where torch is
not installed. Nothing here needs a GPU: ``file_gather.py --device cpu`` runs
the real gather over gloo with one local process per simulated node.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.port_utils import reserved_ports

try:
    import torch

    from benchmarks.cluster_bench.agents import file_gather, nccl_bench, p2p_bench
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

MIB = 1 << 20
GATHER_AGENT = str(Path(file_gather.__file__).resolve())


# --------------------------------------------------------------------------
# nccl_bench.py
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("64M", 64 * MIB), ("1G", 1 << 30), ("512k", 512 << 10), ("4096", 4096)],
)
def test_parse_size(text, expected):
    assert nccl_bench.parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "0", "-1M", "1T", "M", "1.5G"])
def test_parse_size_rejects_garbage(text):
    with pytest.raises(ValueError):
        nccl_bench.parse_size(text)


def test_bus_factors_follow_nccl_tests():
    assert nccl_bench.bus_factor("all_reduce", 8) == pytest.approx(1.75)
    assert nccl_bench.bus_factor("all_gather", 8) == pytest.approx(0.875)
    assert nccl_bench.bus_factor("all_reduce", 32) == pytest.approx(62 / 32)
    assert nccl_bench.bus_factor("all_reduce", 1) == 0.0
    with pytest.raises(ValueError):
        nccl_bench.bus_factor("broadcast", 8)


def test_collective_row_and_element_counts():
    row = nccl_bench.collective_row("all_reduce", 1 << 30, 0.004, 32)
    algbw = (1 << 30) / 0.004 / 1e9
    assert row["algbw_gbs"] == pytest.approx(algbw)
    assert row["busbw_gbs"] == pytest.approx(algbw * 62 / 32)
    assert row["time_us"] == pytest.approx(4000.0)
    # all_gather sizes are the total output: each rank contributes 1/n.
    assert nccl_bench.collective_elements("all_gather", 1 << 30, 32) == (1 << 30) // 128
    assert nccl_bench.collective_elements("all_reduce", 1 << 30, 32) == (1 << 30) // 4
    with pytest.raises(ValueError):
        nccl_bench.collective_elements("all_gather", 64, 32)


def test_pair_peer_maps_each_half_onto_the_other():
    world = 16
    peers = [nccl_bench.pair_peer(rank, world) for rank in range(world)]
    assert peers[:8] == list(range(8, 16)) and peers[8:] == list(range(8))
    assert all(nccl_bench.pair_peer(p, world) == r for r, p in enumerate(peers))
    with pytest.raises(ValueError):
        nccl_bench.pair_peer(0, 7)


def test_node_count_requires_node_major_ranks():
    assert nccl_bench.node_count(9, 1, 1, 8, 16) == 2
    with pytest.raises(ValueError):
        nccl_bench.node_count(9, 1, 0, 8, 16)  # rank 9 cannot live on node 0
    with pytest.raises(ValueError):
        nccl_bench.node_count(0, 0, 0, 8, 12)


def test_pairs_summary_uses_the_slower_rank_of_each_pair():
    size = 10**9
    summary = nccl_bench.pairs_summary([0.010, 0.020, 0.010, 0.010], size, 2)
    assert summary["pairs"] == [[0, 2], [1, 3]]
    assert summary["per_rank_gbs"] == pytest.approx([100.0, 50.0, 100.0, 100.0])
    assert summary["pair_gbs"] == pytest.approx([100.0, 50.0])
    assert summary["aggregate_gbs"] == pytest.approx(150.0)
    assert summary["bidirectional_gbs"] == pytest.approx(300.0)
    assert summary["per_node_gbs"] == pytest.approx(150.0)
    four_nodes = nccl_bench.pairs_summary([0.01] * 8, size, 4)
    assert four_nodes["per_node_gbs"] == pytest.approx(four_nodes["aggregate_gbs"] / 2)


def test_env_snapshot_always_reports_the_three_keys():
    snap = nccl_bench.env_snapshot(
        {"NCCL_IB_HCA": "=mlx5_0", "NCCL_DEBUG": "INFO", "PATH": "/usr/bin"}
    )
    assert snap == {
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_HCA": "=mlx5_0",
        "NCCL_NET": None,
        "NCCL_SOCKET_IFNAME": None,
    }


# --------------------------------------------------------------------------
# p2p_bench.py
# --------------------------------------------------------------------------


def test_p2p_helpers():
    assert p2p_bench.gbs(2 * 10**9, 0.5) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        p2p_bench.gbs(1, 0.0)
    assert [p2p_bench.ring_peer(i, 4) for i in range(4)] == [1, 2, 3, 0]
    pairs = p2p_bench.ordered_pairs(8)
    assert len(pairs) == len(set(pairs)) == 56
    assert all(src != dst for src, dst in pairs)


# --------------------------------------------------------------------------
# file_gather.py pure helpers
# --------------------------------------------------------------------------


def entry(path: str, size: int, owner: int = 0, sha: str | None = None):
    return file_gather.FileEntry(path=path, size=size, sha256=sha, owner=owner)


def test_safe_join_rejects_escaping_paths():
    assert file_gather.safe_join("/m", "sub/a.bin") == "/m/sub/a.bin"
    for bad in ("", "/etc/passwd", "../x", "a/../../x", "a//b", "./a", "a\\b"):
        with pytest.raises(ValueError):
            file_gather.safe_join("/m", bad)


def test_iter_chunks_cover_files_exactly():
    assert list(file_gather.iter_chunks(10, 4)) == [(0, 4), (4, 4), (8, 2)]
    assert list(file_gather.iter_chunks(8, 4)) == [(0, 4), (4, 4)]
    assert list(file_gather.iter_chunks(0, 4)) == []
    with pytest.raises(ValueError):
        list(file_gather.iter_chunks(1, 0))


def test_parse_manifest_validates_entries():
    raw = {
        "root": "/m",
        "files": [
            {"path": "a.bin", "size": 3, "sha256": "AB" * 32, "owner": 1},
            {"path": "b.json", "size": 0, "sha256": None, "owner": 0},
        ],
    }
    root, entries = file_gather.parse_manifest(raw, 2)
    assert root == "/m"
    assert entries[0] == entry("a.bin", 3, 1, "ab" * 32)  # sha256 lower-cased
    bad_files = (
        {"path": "a.bin", "size": 3, "owner": 2},
        {"path": "a.bin", "size": -1, "owner": 0},
        {"path": "../a.bin", "size": 3, "owner": 0},
    )
    for bad in bad_files:
        with pytest.raises(ValueError):
            file_gather.parse_manifest({"root": "/m", "files": [bad]}, 2)
    with pytest.raises(ValueError):
        file_gather.parse_manifest({"root": "/m", "files": raw["files"] * 2}, 2)


def test_plan_transfers_prefers_the_owner_then_the_lowest_holder():
    entries = [
        entry("a", 1, owner=1),
        entry("b", 1, owner=0),
        entry("c", 1, owner=2),
        entry("d", 1, owner=0),
        entry("e", 1, owner=2),
    ]
    have = [
        [False, True, False, True, True],  # rank 0
        [True, False, False, True, True],  # rank 1
        [True, False, False, True, False],  # rank 2
    ]
    transfers = file_gather.plan_transfers(entries, have)
    assert [(t.src, t.receivers) for t in transfers] == [
        (1, (0,)),
        (0, (1, 2)),
        (None, (0, 1, 2)),  # nobody holds it
        (0, ()),  # everyone holds it: no broadcast
        (0, (2,)),  # the owner lacks it: the lowest holder sends
    ]


def test_manifest_digest_changes_with_any_field():
    base = [entry("a", 1), entry("b", 2, owner=1)]
    digest = file_gather.manifest_digest(base)
    assert digest == file_gather.manifest_digest(list(base))
    assert -(2**63) <= digest < 2**63
    for changed in (
        [entry("a", 1), entry("b", 3, owner=1)],
        [entry("a", 1), entry("b", 2, owner=0)],
        [entry("a", 1, sha="0" * 64), entry("b", 2, owner=1)],
    ):
        assert file_gather.manifest_digest(changed) != digest


def test_verify_file_checks_size_and_hash_and_moves_corrupt_files(tmp_path):
    blob = b"model weights"
    sha = hashlib.sha256(blob).hexdigest()
    for name in ("good", "nohash", "badhash", "short"):
        (tmp_path / name).write_bytes(blob)

    good = file_gather.verify_file(
        str(tmp_path / "good"), entry("good", len(blob), sha=sha)
    )
    assert good == {"ok": True, "sha256": sha, "reason": None}
    nohash = file_gather.verify_file(str(tmp_path / "nohash"), entry("n", len(blob)))
    assert nohash["ok"] and nohash["sha256"] == sha

    bad = file_gather.verify_file(
        str(tmp_path / "badhash"), entry("b", len(blob), sha="0" * 64)
    )
    assert bad == {"ok": False, "sha256": sha, "reason": "sha256 mismatch"}
    assert not (tmp_path / "badhash").exists()
    assert (tmp_path / "badhash.cb_corrupt").exists()

    short = file_gather.verify_file(str(tmp_path / "short"), entry("s", len(blob) + 1))
    assert not short["ok"] and short["reason"].startswith("size ")
    assert (tmp_path / "short.cb_corrupt").exists()

    gone = file_gather.verify_file(str(tmp_path / "absent"), entry("x", 1))
    assert gone == {"ok": False, "sha256": None, "reason": "missing"}


def test_build_report_classifies_missing_and_mismatched_files():
    entries = [entry("a", 4), entry("b", 4), entry("c", 4), entry("d", 4)]
    stats = file_gather.Stats(bytes_received=4, bytes_broadcast=8)
    report = file_gather.build_report(
        rank=1,
        world_size=2,
        hostname="h",
        root="/m",
        entries=entries,
        actions={0: "received", 1: "missing", 2: "present", 3: "sent"},
        outcomes={
            0: {"ok": True, "sha256": "x", "reason": None},
            2: {"ok": False, "sha256": "y", "reason": "sha256 mismatch"},
            3: {"ok": False, "sha256": None, "reason": "missing"},
        },
        stats=stats,
        seconds=2.0,
        verify_seconds=1.0,
        total_seconds=3.5,
    )
    assert report["mismatches"] == ["c"]
    assert report["missing"] == ["b", "d"]
    assert report["gbs"] == pytest.approx(8 / 2.0 / 1e9)
    assert [f["action"] for f in report["files"]] == [
        "received",
        "missing",
        "present",
        "sent",
    ]
    assert report["files"][1]["reason"] == "no rank holds it"


def test_host_view_aliases_tensor_memory():
    tensor = torch.zeros(16, dtype=torch.uint8)
    view = file_gather.host_view(tensor)
    assert len(view) == 16 and not view.readonly
    view[3] = 7
    tensor[5] = 9
    assert int(tensor[3]) == 7 and view[5] == 9


# --------------------------------------------------------------------------
# file_gather.py end to end on CPU (gloo), one process per simulated node
# --------------------------------------------------------------------------


@pytest.fixture
def master_port():
    with reserved_ports(1, 29000) as ports:
        yield ports[0]


def pattern(size: int, seed: int) -> bytes:
    return bytes((i * 31 + seed) % 251 for i in range(size))


def place(roots: list[Path], rel: str, blob: bytes, holders) -> None:
    for rank in holders:
        path = roots[rank] / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def run_gather(tmp_path: Path, manifest: dict, roots: list[Path], port: int):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = str(tmp_path / "report-{rank}.json")
    loopback = "lo0" if sys.platform == "darwin" else "lo"
    with contextlib.ExitStack() as stack:
        procs = []
        for rank, root in enumerate(roots):
            env = {
                **os.environ,
                "RANK": str(rank),
                "WORLD_SIZE": str(len(roots)),
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "GLOO_SOCKET_IFNAME": loopback,
            }
            argv = [
                sys.executable, GATHER_AGENT,
                "--manifest", str(manifest_path),
                "--report", report,
                "--chunk-mib", "1",
                "--root", str(root),
                "--device", "cpu",
                "--verify-workers", "2",
                "--timeout-min", "2",
            ]  # fmt: skip
            proc = subprocess.Popen(
                argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            procs.append(stack.enter_context(proc))
        try:
            outputs = [proc.communicate(timeout=150) for proc in procs]
        except subprocess.TimeoutExpired:
            for proc in procs:
                proc.kill()
            raise
    codes = [proc.returncode for proc in procs]
    reports = [
        json.loads((tmp_path / f"report-{rank}.json").read_text())
        for rank in range(len(roots))
    ]
    return codes, reports, outputs


@pytest.mark.timeout(180)
def test_gather_fills_every_node_over_gloo(tmp_path, master_port):
    roots = [tmp_path / f"node{rank}" for rank in range(3)]
    for root in roots:
        root.mkdir()
    files = {
        # path: (bytes, owner, ranks holding it before the gather)
        "a.bin": (pattern(3 * MIB + 12345, 1), 0, {0}),
        "b.bin": (pattern(2 * MIB + 3, 2), 1, {1}),
        "sub/c.bin": (pattern(1_500_000, 3), 2, {2}),
        "empty.bin": (b"", 2, {2}),
        "config.json": (b'{"layers": 92}', 0, {0, 1, 2}),
        "fallback.bin": (pattern(MIB + 1, 4), 1, {2}),  # the owner lacks it
        "shared.bin": (pattern(MIB // 2, 5), 0, {0, 1}),  # rank 1 discards
    }
    for rel, (blob, _, holders) in files.items():
        place(roots, rel, blob, holders)
    (roots[1] / "a.bin").write_bytes(b"stale partial copy")  # wrong size: refetched
    manifest = {
        "root": "/models/unused-with-root-override",
        "files": [
            {
                "path": rel,
                "size": len(blob),
                "sha256": None if rel == "b.bin" else hashlib.sha256(blob).hexdigest(),
                "owner": owner,
            }
            for rel, (blob, owner, _) in sorted(files.items())
        ],
    }
    codes, reports, outputs = run_gather(tmp_path, manifest, roots, master_port)
    assert codes == [0, 0, 0], outputs

    for rank, root in enumerate(roots):
        for rel, (blob, _, _) in files.items():
            assert (root / rel).read_bytes() == blob, (rank, rel)
        assert not list(root.rglob("*.cb_partial"))
        assert not list(root.rglob("*.cb_corrupt"))
        assert reports[rank]["mismatches"] == [] and reports[rank]["missing"] == []
    actions = [{f["path"]: f["action"] for f in r["files"]} for r in reports]
    assert actions[0]["a.bin"] == "sent" and actions[1]["a.bin"] == "received"
    assert (
        actions[2]["fallback.bin"] == "sent"
        and actions[1]["fallback.bin"] == "received"
    )
    assert (
        actions[1]["shared.bin"] == "present" and actions[2]["shared.bin"] == "received"
    )
    assert {a["config.json"] for a in actions} == {"present"}
    assert reports[1]["bytes_discarded"] == len(files["shared.bin"][0])
    broadcast = sum(
        len(blob) for rel, (blob, _, _) in files.items() if rel != "config.json"
    )
    assert {r["bytes_broadcast"] for r in reports} == {broadcast}
    # Every rank hashed every file, including the one without a hub sha256.
    b_hashes = {
        f["sha256"] for r in reports for f in r["files"] if f["path"] == "b.bin"
    }
    assert b_hashes == {hashlib.sha256(files["b.bin"][0]).hexdigest()}


@pytest.mark.timeout(180)
def test_gather_flags_a_corrupt_source_on_every_node(tmp_path, master_port):
    roots = [tmp_path / f"node{rank}" for rank in range(2)]
    good = pattern(MIB + 7, 1)
    place(roots, "ok.bin", good, {0})
    place(roots, "bad.bin", pattern(4096, 2), {1})
    manifest = {
        "root": "/unused",
        "files": [
            {"path": "bad.bin", "size": 4096, "sha256": "0" * 64, "owner": 1},
            {"path": "ok.bin", "size": len(good), "sha256": None, "owner": 0},
        ],
    }
    codes, reports, outputs = run_gather(tmp_path, manifest, roots, master_port)
    assert codes == [1, 1], outputs
    for rank, root in enumerate(roots):
        assert reports[rank]["mismatches"] == ["bad.bin"]
        assert not (root / "bad.bin").exists()  # moved aside so a rerun refetches it
        assert (root / "bad.bin.cb_corrupt").exists()
        assert (root / "ok.bin").read_bytes() == good
