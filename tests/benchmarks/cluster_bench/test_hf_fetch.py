# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model download: shard planning, the stdin-only token rule, step orchestration."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from huggingface_hub import ModelInfo

from benchmarks.cluster_bench.adapters import hf_fetch
from benchmarks.cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.cluster_bench.adapters.hf_fetch import (
    BOOTSTRAP,
    BOOTSTRAP_TAG,
    FILES_SEPARATOR,
    MARKER_NAME,
    RESULT_PREFIX,
    SMALL_FILE_BYTES,
    DownloadHandle,
    FetchPlan,
    ModelFetcher,
    ModelFetchError,
    RepoFile,
    build_fetch_plan,
    build_manifest,
    check_reports,
    completion_marker,
    completion_probe_command,
    download_params,
    download_stdin,
    glob_escape,
    list_repo_files,
    marker_matches,
    model_dir_for,
    parse_completion_probe,
    parse_download_result,
    parse_ipv4,
    parse_probe,
    parse_rank_report,
    parse_started,
    plan_shards,
    probe_command,
    render_download_script,
)
from benchmarks.cluster_bench.adapters.microbench import (
    DEFAULT_PORT_BASE,
    GATHER_SCRIPT,
    rendezvous_port,
)
from benchmarks.cluster_bench.ports import CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "hf_fetch"
TOKEN = "hf_TESTtokenNOTreal1234567890abcdef"
NODES = ("node0", "node1", "node2", "node3")
IPS = {node: f"10.42.0.{i + 1}" for i, node in enumerate(NODES)}
WORKDIR = "/root/cluster_bench"
PYTHON = f"{WORKDIR}/venv/bin/python"
IMAGE = "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
MIB = 1 << 20


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class FakeApi:
    def __init__(self, info: ModelInfo):
        self.info = info
        self.calls: list[dict] = []

    def model_info(self, repo_id, *, revision=None, files_metadata=False, token=None):
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "files_metadata": files_metadata,
                "token": token,
            }
        )
        return self.info


def install_api(monkeypatch, info: ModelInfo) -> FakeApi:
    api = FakeApi(info)
    monkeypatch.setattr(hf_fetch, "HfApi", lambda: api)
    return api


def glm_info() -> ModelInfo:
    return ModelInfo(**json.loads((FIXTURES / "glm53_model_info.json").read_text()))


def glm_files() -> list[RepoFile]:
    info = glm_info()
    return [
        RepoFile(s.rfilename, s.size, s.lfs.sha256 if s.lfs else None)
        for s in info.siblings
    ]


def tiny_info(commit: str = COMMIT) -> ModelInfo:
    """A small repo: three small files and six shards above the threshold."""
    siblings = [
        {"rfilename": "config.json", "size": 2048},
        {"rfilename": "model.safetensors.index.json", "size": 300_000},
        {
            "rfilename": "tokenizer.json",
            "size": 9 * MIB,
            "lfs": {"sha256": "ab" * 32, "size": 9 * MIB, "pointerSize": 133},
        },
    ]
    for i in range(1, 7):
        size = (100 + 10 * i) * MIB
        siblings.append(
            {
                "rfilename": f"model-{i:05d}-of-00006.safetensors",
                "size": size,
                "lfs": {"sha256": f"{i:02x}" * 32, "size": size, "pointerSize": 135},
            }
        )
    return ModelInfo(id="org/tiny-model", sha=commit, siblings=siblings)


def tiny_plan(nodes=NODES) -> FetchPlan:
    info = tiny_info()
    files = [
        RepoFile(s.rfilename, s.size, s.lfs.sha256 if s.lfs else None)
        for s in info.siblings
    ]
    return build_fetch_plan("org/tiny-model", "main", COMMIT, files, nodes)


def ok(node: str, command: str, stdout: str = "") -> CommandResult:
    return CommandResult(node, command, 0, stdout, "")


def params_of(stdin: str) -> dict:
    """PARAMS of a download script sent on stdin (line 2, after the token)."""
    line = stdin.split("\n")[1]
    assert line.startswith("PARAMS = ")
    return json.loads(json.loads(line[len("PARAMS = ") :]))


def git_sha(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()


def rank_report(plan: FetchPlan, rank: int, *, bad: dict | None = None) -> dict:
    """A file_gather.py report in which every file verified, unless in ``bad``."""
    bad = bad or {}
    files = []
    for f in plan.files:
        reason = bad.get(f.path)
        files.append(
            {
                "path": f.path,
                "size": f.size,
                "action": "sent" if f.owner == rank else "received",
                "sha256": f.sha256 or git_sha(f.path),
                "expected_sha256": f.sha256,
                "ok": reason is None,
                "reason": reason,
            }
        )
    received = sum(f.size for f in plan.files if f.owner not in (None, rank))
    return {
        "rank": rank,
        "hostname": f"host-{rank}",
        "world_size": len(plan.nodes),
        "root": "/models/x",
        "files": files,
        "bytes_received": received,
        "bytes_sent": sum(f.size for f in plan.shard_files(rank)),
        "bytes_discarded": 0,
        "bytes_broadcast": sum(f.size for f in plan.files if f.owner is not None),
        "seconds": 2.5 + rank * 0.1,
        "verify_seconds": 1.0,
        "total_seconds": 4.0,
        "gbs": 1.0,
        "mismatches": sorted(p for p, r in bad.items() if r != "missing"),
        "missing": sorted(p for p, r in bad.items() if r == "missing"),
    }


@dataclass
class StdinRecorder(RecordingExecutor):
    """RecordingExecutor that also keeps every stdin payload, for assertions."""

    stdins: list[tuple[str, str]] = field(default_factory=list)

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        if input_text is not None:
            self.stdins.append((node, input_text))
        return super().run(
            node, command, input_text=input_text, timeout=timeout, check=check
        )


class FakeNodes:
    """Plays the node side: download launches, status probes, the gather."""

    def __init__(self, rec: StdinRecorder, plan: FetchPlan, model_dir: str):
        self.rec = rec
        self.plan = plan
        self.model_dir = model_dir
        self.complete: set[str] = set()
        self.probes: Counter = Counter()
        self.bad_reports: dict[int, dict] = {}
        self.failing_gather: set[str] = set()

    def listing(self, node: str) -> str:
        if node not in self.complete:
            return ""
        return "".join(f"{f.size}\t{f.path}\n" for f in self.plan.files)

    def __call__(self, node: str, command: str) -> CommandResult | None:
        if FILES_SEPARATOR in command:
            marker = self.rec.files.get((node, f"{self.model_dir}/{MARKER_NAME}"), "")
            return ok(
                node, command, f"{marker}\n{FILES_SEPARATOR}\n{self.listing(node)}"
            )
        if "CB_RUNNING" in command:  # shard status probe
            self.probes[node] += 1
            if self.probes[node] == 1:
                return ok(node, command, "CB_ABSENT\n")
            files = self.plan.files_for(self.plan.rank_of(node))
            result = {
                "seconds": 100.0,
                "files": len(files),
                "bytes": sum(f.size for f in files),
            }
            status = json.dumps({"rc": 0, "result": result})
            return ok(node, command, f"CB_STATUS {status}\n")
        if BOOTSTRAP_TAG in command:  # a download launch; stdin says which kind
            params = params_of(self.rec.stdins[-1][1])
            if params["background"]:
                return ok(node, command, "CB_STARTED 4242\n")
            result = {
                "seconds": 2.0,
                "files": len(params["sizes"]),
                "bytes": sum(params["sizes"].values()),
            }
            return ok(node, command, f"log line\n{RESULT_PREFIX}{json.dumps(result)}\n")
        if GATHER_SCRIPT in command:
            argv = command.split()
            rank = int(argv[argv.index("--node-rank") + 1])
            if node in self.failing_gather:
                return CommandResult(node, command, 1, "", "NCCL error: remote hung up")
            report = self.bad_reports.get(rank) or rank_report(self.plan, rank)
            state = f"{WORKDIR}/downloads/{self.plan.plan_id}"
            self.rec.files[(node, f"{state}/report-{rank}.json")] = json.dumps(report)
            self.complete.add(node)
            return ok(node, command)
        return None


def make_cluster(plan: FetchPlan, model_dir: str) -> tuple[StdinRecorder, FakeNodes]:
    rec = StdinRecorder(node_names=NODES)
    fake = FakeNodes(rec, plan, model_dir)
    rec.responder = fake
    return rec, fake


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def fetcher(rec, **kwargs) -> ModelFetcher:
    options = {"control_ips": IPS, "clock": FakeClock()}
    options.update(kwargs)
    return ModelFetcher(rec, PYTHON, WORKDIR, **options)


# --------------------------------------------------------------------------
# Listing and planning
# --------------------------------------------------------------------------


def test_list_repo_files_reads_sizes_and_lfs_sha256(monkeypatch):
    api = install_api(monkeypatch, glm_info())
    commit, files = list_repo_files("zai-org/GLM-5.3", "main", "")
    assert commit == "ba32f8c19c8738e2c57c71798c601d477a7bf844"
    assert len(files) == 156
    assert [f.path for f in files] == sorted(f.path for f in files)
    by_path = {f.path: f for f in files}
    shard = by_path["model-00001-of-00141.safetensors"]
    assert shard.size == 5363940952 and len(shard.sha256) == 64
    assert by_path["config.json"].sha256 is None
    assert by_path["tokenizer.json"].sha256 is not None  # small but LFS
    assert sum(f.size for f in files) == pytest.approx(755.66e9, rel=1e-4)
    assert api.calls == [
        {
            "repo_id": "zai-org/GLM-5.3",
            "revision": "main",
            "files_metadata": True,
            "token": None,
        }
    ]


def test_plan_shards_balances_the_real_glm_listing():
    files = glm_files()
    plan = plan_shards(files, 4)
    large = [f for f in files if f.size >= SMALL_FILE_BYTES]
    assert len(plan.owners) == len(large) == 141
    assert len(plan.small) == 15 and "config.json" in plan.small
    assert sum(plan.rank_bytes) == sum(f.size for f in large)
    # LPT keeps every node within one file of the others.
    assert max(plan.rank_bytes) - min(plan.rank_bytes) <= max(f.size for f in large)
    for rank in range(4):
        assert set(plan.small) <= set(plan.files_for(rank))
    owned = [
        p for rank in range(4) for p in plan.files_for(rank) if p not in plan.small
    ]
    assert sorted(owned) == sorted(f.path for f in large)  # each exactly once


def test_plan_shards_is_independent_of_input_order():
    files = glm_files()
    shuffled = list(files)
    random.Random(7).shuffle(shuffled)
    assert plan_shards(shuffled, 4) == plan_shards(files, 4)
    assert plan_shards(files, 3) != plan_shards(files, 4)


def test_plan_shards_edge_cases():
    files = [
        RepoFile("a", 10 * MIB),
        RepoFile("b", SMALL_FILE_BYTES),  # at the threshold: large
        RepoFile("c", SMALL_FILE_BYTES + 1),
    ]
    single = plan_shards(files, 1)
    assert single.small == ("a",) and set(single.owners.values()) == {0}
    two = plan_shards(files, 2)
    assert two.owners == {"c": 0, "b": 1}
    assert plan_shards([RepoFile("x", 1)], 3).owners == {}
    with pytest.raises(ValueError):
        plan_shards(files, 0)
    with pytest.raises(ValueError):
        plan_shards([RepoFile("a", 1), RepoFile("a", 2)], 2)


def test_fetch_plan_is_json_serializable_and_stable():
    plan = tiny_plan()
    restored = FetchPlan.model_validate(json.loads(plan.model_dump_json()))
    assert restored == plan and restored.plan_id == plan.plan_id
    assert tiny_plan(nodes=NODES[::-1]).plan_id != plan.plan_id
    assert len(plan.small_files()) == 3
    assert {f.owner for f in plan.files if f.owner is not None} == {0, 1, 2, 3}
    for rank in range(4):
        assert set(plan.small_files()) <= set(plan.files_for(rank))
        assert set(plan.shard_files(rank)) <= set(plan.files_for(rank))
    assert plan.rank_of("node2") == 2
    with pytest.raises(KeyError):
        plan.rank_of("node9")
    with pytest.raises(ValueError):
        build_fetch_plan("r", "main", COMMIT, [], ("node0", "node0"))


def test_manifest_and_completion_marker():
    plan = tiny_plan()
    manifest = build_manifest(plan, "/models/m")
    assert manifest["root"] == "/models/m"
    entries = {e["path"]: e for e in manifest["files"]}
    assert (
        entries["config.json"]["owner"] == 0
        and entries["config.json"]["sha256"] is None
    )
    shard = next(f for f in plan.files if f.owner == 3)
    assert entries[shard.path] == {
        "path": shard.path,
        "size": shard.size,
        "sha256": shard.sha256,
        "owner": 3,
    }
    marker = completion_marker(plan, 1.0)
    assert marker_matches(marker, plan)
    assert marker_matches(json.loads(json.dumps(marker)), tiny_plan(nodes=NODES[:2]))
    assert not marker_matches({**marker, "commit": "f" * 40}, plan)
    assert not marker_matches({**marker, "files": {"config.json": 2048}}, plan)
    assert not marker_matches(None, plan)


def test_model_dir_for_and_glob_escape():
    assert (
        model_dir_for("/root/models/", "RadixArk/GLM-5.3-NVFP4", "abcdef0123456789ff")
        == "/root/models/RadixArk--GLM-5.3-NVFP4@abcdef012345"
    )
    with pytest.raises(ValueError):
        model_dir_for("/m", "a/b", "")
    odd = "weights/part[1]*?.bin"
    assert fnmatch.fnmatchcase(odd, glob_escape(odd))
    assert not fnmatch.fnmatchcase("weights/part1xy.bin", glob_escape(odd))
    assert (
        glob_escape("model-00001-of-00141.safetensors")
        == "model-00001-of-00141.safetensors"
    )


# --------------------------------------------------------------------------
# Token on stdin only, and the download script itself
# --------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_bootstrap_reads_the_token_from_the_first_stdin_line():
    script = "print(repr(CB_TOKEN))\nprint(__name__)\n"
    out = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP],
        input=download_stdin(TOKEN, script),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.split()
    assert out == [repr(TOKEN), "__cb_download__"]
    empty = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP],
        input=download_stdin(None, script),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.split()
    assert empty[0] == "None"
    assert TOKEN not in BOOTSTRAP
    with pytest.raises(ValueError):
        download_stdin("two\nlines", script)


def fake_hub(sizes: dict[str, int], calls: list, shrink: str | None = None):
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        for rel, size in sizes.items():
            path = Path(kwargs["local_dir"]) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (size - 1 if rel == shrink else size))
        return kwargs["local_dir"]

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    return module


def small_params(tmp_path: Path, background: bool = False) -> dict:
    plan = build_fetch_plan(
        "org/tiny-model",
        "main",
        COMMIT,
        [RepoFile("config.json", 64), RepoFile("sub/tok[1].json", 32, "cd" * 32)],
        ("node0",),
    )
    return download_params(
        plan,
        str(tmp_path / "model"),
        plan.small_files(),
        state_dir=str(tmp_path / "state"),
        background=background,
        rank=0,
    )


def test_download_script_foreground_calls_snapshot_download(
    monkeypatch, tmp_path, capsys
):
    params = small_params(tmp_path)
    calls: list = []
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", fake_hub(params["sizes"], calls)
    )
    code = compile(render_download_script(params), f"<{BOOTSTRAP_TAG}>", "exec")
    exec(code, {"__name__": "__cb_download__", "CB_TOKEN": TOKEN})

    result = parse_download_result(capsys.readouterr().out)
    assert result["files"] == 2 and result["bytes"] == 96
    (call,) = calls
    assert call["revision"] == COMMIT and call["token"] == TOKEN
    assert call["repo_id"] == "org/tiny-model"
    assert call["local_dir"] == str(tmp_path / "model")
    assert call["allow_patterns"] == ["config.json", "sub/tok[[]1].json"]
    assert call["max_workers"] == 16


def test_download_script_rejects_wrong_sizes(monkeypatch, tmp_path):
    params = small_params(tmp_path)
    fake = fake_hub(params["sizes"], [], shrink="config.json")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    code = compile(render_download_script(params), f"<{BOOTSTRAP_TAG}>", "exec")
    with pytest.raises(RuntimeError, match="config.json"):
        exec(code, {"__name__": "__cb_download__", "CB_TOKEN": None})


FAKE_HUB_PACKAGE = """
import json
import os


def snapshot_download(*, repo_id, revision, local_dir, allow_patterns, max_workers, token):
    sizes = json.loads(os.environ["FAKE_HF_SIZES"])
    for rel, size in sizes.items():
        path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x" * size)
    with open(os.environ["FAKE_HF_CALLS"], "w") as fh:
        json.dump({"revision": revision, "token_matches": token == os.environ["FAKE_HF_EXPECT"]}, fh)
    return local_dir
"""


@pytest.mark.timeout(60)
def test_download_script_background_detaches_and_writes_status(tmp_path):
    params = small_params(tmp_path, background=True)
    os.makedirs(tmp_path / "state")
    package = tmp_path / "fakehub" / "huggingface_hub"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(FAKE_HUB_PACKAGE)
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path / "fakehub"),
        "FAKE_HF_SIZES": json.dumps(params["sizes"]),
        "FAKE_HF_CALLS": str(tmp_path / "calls.json"),
        "FAKE_HF_EXPECT": TOKEN,
    }
    launched = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP],
        input=download_stdin(TOKEN, render_download_script(params)),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=True,
    )
    pid = parse_started(launched.stdout)
    status_path = Path(params["status_path"])
    deadline = time.monotonic() + 30
    while not status_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)  # polling the daemon's status file, not a readiness sleep
    status = json.loads(status_path.read_text())
    assert status["rc"] == 0 and status["result"]["files"] == 2
    assert Path(params["pid_path"]).read_text() == str(pid)
    assert "done" in Path(params["log_path"]).read_text()
    assert json.loads((tmp_path / "calls.json").read_text()) == {
        "revision": COMMIT,
        "token_matches": True,
    }
    probe = subprocess.run(
        ["bash", "-c", probe_command(params["pid_path"], params["status_path"])],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert parse_probe(probe.stdout).state == "done"


# --------------------------------------------------------------------------
# Orchestration on a recorded cluster (no network, no GPUs)
# --------------------------------------------------------------------------


def test_fetch_runs_every_step_and_never_exposes_the_token(monkeypatch):
    install_api(monkeypatch, tiny_info())
    plan = tiny_plan()
    model_dir = model_dir_for("/root/models", "org/tiny-model", COMMIT)
    rec, fake = make_cluster(plan, model_dir)
    report = fetcher(rec).fetch(
        "org/tiny-model", "main", model_dir, NODES, TOKEN, IMAGE
    )

    # The token travels on stdin only: never in a command, file or argv.
    assert all(TOKEN not in c.command for c in rec.commands)
    assert all(TOKEN not in content for content in rec.files.values())
    assert len(rec.stdins) == 2 * len(NODES)
    assert all(stdin.split("\n", 1)[0] == TOKEN for _, stdin in rec.stdins)
    launches = [c for c in rec.commands if "HF_XET_HIGH_PERFORMANCE=1" in c.command]
    assert len(launches) == 2 * len(NODES) and all(c.had_input for c in launches)
    unset = "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "
    assert all(unset in c.command for c in launches)

    state = f"{WORKDIR}/downloads/{plan.plan_id}"
    small = {f.path for f in plan.small_files()}
    for rank, node in enumerate(NODES):
        kinds = [params_of(s) for n, s in rec.stdins if n == node]
        assert [p["background"] for p in kinds] == [False, True]
        assert set(kinds[0]["sizes"]) == small
        assert set(kinds[1]["sizes"]) == small | {
            f.path for f in plan.shard_files(rank)
        }
        manifest = json.loads(rec.files[(node, f"{state}/manifest.json")])
        assert manifest == json.loads(json.dumps(build_manifest(plan, model_dir)))
        marker = json.loads(rec.files[(node, f"{model_dir}/{MARKER_NAME}")])
        assert marker_matches(marker, plan)

    port = rendezvous_port(DEFAULT_PORT_BASE, f"gather|{plan.plan_id}")
    for rank, node in enumerate(NODES):
        (gather,) = [c for c in rec.commands_for(node) if GATHER_SCRIPT in c]
        argv = gather.split()
        assert argv[:3] == ["docker", "run", "--rm"] and "-d" not in argv
        assert argv[argv.index("--gpus") + 1] == "'\"device=0\"'"
        inner = argv[argv.index(IMAGE) + 1 :]
        assert inner[:9] == [
            "torchrun",
            "--nnodes",
            "4",
            "--node-rank",
            str(rank),
            "--master-addr",
            IPS["node0"],
            "--master-port",
            str(port),
        ]
        assert inner[inner.index("--nproc-per-node") + 1] == "1"
        assert "NCCL_SOCKET_IFNAME=wg0" in argv and "NCCL_NET=IB" in argv
        assert f"{model_dir}:{model_dir}" in argv
        assert f"{WORKDIR}/src:/bench/src:ro" in argv
        assert f"'{state}/report-{{rank}}.json'" in gather

    assert report.verified and not report.skipped
    assert set(report.small) == set(report.shards) == set(NODES)
    assert report.shards["node1"].mbps == pytest.approx(
        report.shards["node1"].bytes / 100.0 / 1e6
    )
    shard_bytes = sum(f.size for f in plan.files if f.owner is not None)
    assert report.gather_gbs == pytest.approx(shard_bytes / 2.8 / 1e9)
    assert report.total_bytes == plan.total_bytes and report.file_count == 9


def test_fetch_skips_everything_when_complete_on_every_node(monkeypatch):
    install_api(monkeypatch, tiny_info())
    plan = tiny_plan()
    model_dir = "/root/models/m"
    rec, fake = make_cluster(plan, model_dir)
    marker = json.dumps(completion_marker(plan, 5.0))
    for node in NODES:
        rec.files[(node, f"{model_dir}/{MARKER_NAME}")] = marker
        fake.complete.add(node)
    report = fetcher(rec).fetch(
        "org/tiny-model", "main", model_dir, NODES, TOKEN, IMAGE
    )
    assert report.skipped and report.verified
    assert not rec.stdins
    assert not [c for c in rec.commands if GATHER_SCRIPT in c.command]


def test_is_complete_rejects_stale_marker_or_missing_file():
    plan = tiny_plan()
    model_dir = "/root/models/m"
    rec, fake = make_cluster(plan, model_dir)
    stale = completion_marker(plan, 5.0) | {"commit": "f" * 40}
    for node in NODES:
        rec.files[(node, f"{model_dir}/{MARKER_NAME}")] = json.dumps(stale)
        fake.complete.add(node)
    assert not fetcher(rec).is_complete(plan, model_dir, NODES)
    good = json.dumps(completion_marker(plan, 5.0))
    for node in NODES:
        rec.files[(node, f"{model_dir}/{MARKER_NAME}")] = good
    assert fetcher(rec).is_complete(plan, model_dir, NODES)
    fake.complete.discard("node2")  # marker present but the files are gone
    assert not fetcher(rec).is_complete(plan, model_dir, NODES)
    assert fetcher(rec).is_complete(plan, model_dir, ["node0", "node1"])


def test_start_downloads_attaches_restarts_or_launches():
    plan = tiny_plan()
    rec = StdinRecorder(node_names=NODES)
    probes = {
        "node0": "CB_RUNNING 777",
        "node1": 'CB_STATUS {"rc": 0, "result": {"seconds": 1, "files": 1, "bytes": 1}}',
        "node2": 'CB_STATUS {"rc": 1, "error": "HTTPError 502"}',
        "node3": "CB_ABSENT",
    }

    def responder(node, command):
        if "CB_RUNNING" in command:
            return ok(node, command, probes[node] + "\n")
        if BOOTSTRAP_TAG in command:
            return ok(node, command, "CB_STARTED 4242\n")
        if FILES_SEPARATOR in command:
            # node1 finished earlier and its shard is still whole on disk.
            listing = "".join(f"{f.size}\t{f.path}\n" for f in plan.files_for(1))
            return ok(node, command, f"\n{FILES_SEPARATOR}\n{listing}")
        return None

    rec.responder = responder
    handles = fetcher(rec).start_downloads(plan, "/root/models/m", NODES, TOKEN)
    assert (handles["node0"].pid, handles["node0"].reused) == (777, True)
    assert (handles["node1"].pid, handles["node1"].reused) == (None, True)
    for node in ("node2", "node3"):
        assert (handles[node].pid, handles[node].reused) == (4242, False)
        assert any(c.startswith("rm -f ") for c in rec.commands_for(node))
    assert [n for n, _ in rec.stdins] == ["node2", "node3"]
    assert handles["node3"].files == len(plan.files_for(3))
    assert handles["node3"].status_path.endswith("/shards.status.json")


def test_start_downloads_restarts_a_finished_but_damaged_shard():
    plan = tiny_plan()
    rec = StdinRecorder(node_names=NODES)
    done = 'CB_STATUS {"rc": 0, "result": {"seconds": 1, "files": 1, "bytes": 1}}'

    def responder(node, command):
        if "CB_RUNNING" in command:
            return ok(node, command, done + "\n")
        if BOOTSTRAP_TAG in command:
            return ok(node, command, "CB_STARTED 4242\n")
        if FILES_SEPARATOR in command:
            return ok(node, command, f"\n{FILES_SEPARATOR}\n")  # shard files gone
        return None

    rec.responder = responder
    handles = fetcher(rec).start_downloads(plan, "/root/models/m", NODES, TOKEN)
    for node, handle in handles.items():
        if plan.files_for(handle.rank):
            assert (handle.pid, handle.reused) == (4242, False), node


def handles_for(plan: FetchPlan) -> dict[str, DownloadHandle]:
    state = f"{WORKDIR}/downloads/{plan.plan_id}"
    return {
        node: DownloadHandle(
            node=node,
            rank=rank,
            pid=4242,
            pid_path=f"{state}/shards.pid",
            log_path=f"{state}/shards.log",
            status_path=f"{state}/shards.status.json",
            files=1,
            bytes=1,
        )
        for rank, node in enumerate(NODES[:2])
    }


def test_wait_downloads_polls_until_every_node_finishes():
    plan = tiny_plan()
    rec = StdinRecorder(node_names=NODES)
    calls: Counter = Counter()
    done = json.dumps({"rc": 0, "result": {"seconds": 50.0, "files": 4, "bytes": 5e8}})

    def responder(node, command):
        calls[node] += 1
        finished = calls[node] >= (2 if node == "node0" else 3)
        return ok(node, command, f"CB_STATUS {done}" if finished else "CB_RUNNING 4242")

    rec.responder = responder
    clock = FakeClock()
    results = fetcher(rec, clock=clock).wait_downloads(handles_for(plan), 3600, poll=30)
    assert set(results) == {"node0", "node1"}
    assert (
        results["node1"].mbps == pytest.approx(10.0)
        and results["node1"].phase == "shards"
    )
    assert clock.sleeps == [30, 30]
    assert calls == {"node0": 2, "node1": 3}


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        ('CB_STATUS {"rc": 1, "error": "HTTPError 403 gated repo"}', "HTTPError 403"),
        ("CB_ABSENT", "without writing a status file"),
    ],
)
def test_wait_downloads_raises_with_the_log_tail(probe, expected):
    plan = tiny_plan()
    rec = StdinRecorder(node_names=NODES)

    def responder(node, command):
        if command.startswith("tail -n"):
            return ok(node, command, "Traceback ...\nlast log line\n")
        return ok(node, command, probe)

    rec.responder = responder
    with pytest.raises(ModelFetchError, match=expected) as info:
        fetcher(rec).wait_downloads(handles_for(plan), 3600)
    assert "last log line" in str(info.value)


def test_wait_downloads_times_out():
    plan = tiny_plan()
    rec = StdinRecorder(
        node_names=NODES, responder=lambda n, c: ok(n, c, "CB_RUNNING 1")
    )
    clock = FakeClock()
    with pytest.raises(ModelFetchError, match="still running"):
        fetcher(rec, clock=clock).wait_downloads(handles_for(plan), 100, poll=40)
    assert clock.sleeps == [40, 40, 40]


def test_gather_resolves_the_rendezvous_address_when_not_given():
    plan = tiny_plan()
    rec = StdinRecorder(node_names=NODES)
    ip_out = (
        "5: wg0    inet 10.42.0.1/24 scope global wg0\\       "
        "valid_lft forever preferred_lft forever\n"
    )
    rec.responder = lambda n, c: ok(n, c, ip_out) if c.startswith("ip -4") else None
    jobs = ModelFetcher(rec, PYTHON, WORKDIR).gather_jobs(plan, "/root/models/m", IMAGE)
    assert rec.commands_for("node0") == ["ip -4 -o addr show dev wg0"]
    assert all(rec.commands_for(n) == [] for n in NODES[1:])
    inner = jobs["node3"][jobs["node3"].index(IMAGE) + 1 :]
    assert inner[inner.index("--master-addr") + 1] == "10.42.0.1"


def test_gather_raises_when_a_rank_wrote_no_report():
    plan = tiny_plan()
    rec, fake = make_cluster(plan, "/root/models/m")
    fake.failing_gather.add("node2")
    with pytest.raises(ModelFetchError, match="node2") as info:
        fetcher(rec).gather(plan, "/root/models/m", NODES, IMAGE)
    assert "NCCL error: remote hung up" in str(info.value)
    # The failed client's container is removed; healthy nodes are left alone.
    cleanup = "docker rm -f cb-model-gather >/dev/null 2>&1 || true"
    assert rec.commands_for("node2")[-1] == cleanup
    assert all(rec.commands_for(n)[-1] != cleanup for n in ("node0", "node1"))
    with pytest.raises(ValueError):
        fetcher(rec).gather(plan, "/root/models/m", NODES[:3], IMAGE)


def test_gather_reports_mismatches_and_verify_refuses_to_mark():
    plan = tiny_plan()
    model_dir = "/root/models/m"
    rec, fake = make_cluster(plan, model_dir)
    shard = next(f for f in plan.files if f.owner == 0)
    fake.bad_reports[2] = rank_report(plan, 2, bad={shard.path: "sha256 mismatch"})
    result = fetcher(rec).gather(plan, model_dir, NODES, IMAGE)
    assert not result.ok
    assert result.mismatches == (f"node2:{shard.path}",)
    with pytest.raises(ModelFetchError, match="sha256 mismatch") as info:
        fetcher(rec).verify_and_mark(plan, model_dir, NODES)
    assert any(shard.path in p for p in info.value.problems)
    assert not [k for k in rec.files if k[1].endswith(MARKER_NAME)]


def test_verify_and_mark_writes_markers_after_a_clean_gather():
    plan = tiny_plan()
    model_dir = "/root/models/m"
    rec, _ = make_cluster(plan, model_dir)
    gathered = fetcher(rec).gather(plan, model_dir, NODES, IMAGE)
    assert gathered.ok and gathered.exit_codes == {n: 0 for n in NODES}
    verified = fetcher(rec).verify_and_mark(plan, model_dir, NODES)
    assert verified.files == len(plan.files)
    for node in NODES:
        marker = json.loads(rec.files[(node, f"{model_dir}/{MARKER_NAME}")])
        assert marker_matches(marker, plan)
    assert fetcher(rec).is_complete(plan, model_dir, NODES)


# --------------------------------------------------------------------------
# Pure parsers
# --------------------------------------------------------------------------


def test_check_reports_cross_checks_hashes():
    plan = tiny_plan()
    reports = {r: parse_rank_report(rank_report(plan, r), NODES[r]) for r in range(4)}
    assert check_reports(plan, reports) == []
    git_file = "config.json"
    raw = rank_report(plan, 1)
    for item in raw["files"]:
        if item["path"] == git_file:
            item["sha256"] = "0" * 64
    reports[1] = parse_rank_report(raw, "node1")
    problems = check_reports(plan, reports)
    assert problems and git_file in problems[0]
    del reports[3]
    assert "node3: no gather report" in check_reports(plan, reports)


def test_probe_and_listing_parsers():
    assert parse_probe("CB_RUNNING 4242\n").pid == 4242
    assert parse_probe('CB_STATUS {"rc": 0}\n').status == {"rc": 0}
    assert parse_probe("noise\nCB_ABSENT\n").state == "absent"
    with pytest.raises(ValueError):
        parse_probe("bash: line 1: syntax error")
    assert parse_started("x\nCB_STARTED 99\n") == 99
    with pytest.raises(ValueError):
        parse_started("CB_START_FAILED\n")
    with pytest.raises(ValueError):
        parse_download_result("Traceback (most recent call last):\n")

    command = completion_probe_command("/root/models/m")
    assert f"/root/models/m/{MARKER_NAME}" in command and "-printf" in command
    marker, sizes = parse_completion_probe(
        '{"repo": "r"}\n' + f"{FILES_SEPARATOR}\n" + "12\tconfig.json\n7\tsub/a b.bin\n"
    )
    assert marker == {"repo": "r"}
    assert sizes == {"config.json": 12, "sub/a b.bin": 7}
    torn, _ = parse_completion_probe('{"repo": \n' + f"{FILES_SEPARATOR}\n")
    assert torn is None
    assert parse_ipv4("3: wg0    inet 10.42.0.7/24 scope global wg0") == "10.42.0.7"
    with pytest.raises(ValueError):
        parse_ipv4('Device "wg0" does not exist.')
