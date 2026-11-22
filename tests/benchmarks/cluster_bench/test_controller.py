# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for controller.py and the harness shell scripts.

``FakeNode`` stands in for every local subprocess the controller starts (ssh
to node0, rsync, scp, git, the KB merge); ``lium`` is the fake binary from
fixtures/lium on PATH. Nothing touches the network, ssh or a real cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from benchmarks.cluster_bench import controller as ctl_mod
from benchmarks.cluster_bench.adapters.providers import (
    LiumBillingUncertainError,
    LiumCli,
    LiumFabric,
    LiumGeneralError,
)
from benchmarks.cluster_bench.controller import (
    SYNC_MANIFEST,
    Controller,
    ControllerError,
    ControllerState,
    RemoteSnapshot,
    UpOptions,
    WatchOptions,
)
from benchmarks.cluster_bench.ports import CommandResult
from tests.benchmarks.cluster_bench.lium_fakes import (
    API_KEY,
    CONTROLLER_FIXTURES,
    HF_TOKEN,
    LIUM_FIXTURES,
    FakeLium,
    fail,
    ok,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parents[2]
CLUSTER_BENCH_SH = REPO_ROOT / "benchmarks" / "cluster_bench" / "cluster_bench.sh"
BOOTSTRAP_SH = (
    REPO_ROOT / "benchmarks" / "cluster_bench" / "scripts" / "bootstrap_node.sh"
)

B200_ID = "4c1f9a2e-7b3d-4e8a-9f61-2d5c8b0e7a13"
CLOCK_START = 1_762_190_000.0
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not installed")


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative defaults (./cluster_bench_results, ./cluster_bench_kb) land in tmp."""
    monkeypatch.chdir(tmp_path)


class FakeClock:
    def __init__(self, start: float = CLOCK_START):
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@dataclass
class Call:
    argv: list[str]
    input_data: bytes | None
    env: dict[str, str] | None

    @property
    def tool(self) -> str:
        return Path(self.argv[0]).name

    @property
    def script(self) -> str:
        """The remote script of an ``ssh ... bash -c <quoted>`` call."""
        return shlex.split(self.argv[-1])[0] if self.tool == "ssh" else ""


StatusEntry = Callable[["FakeNode"], "dict[str, Any] | None"]


@dataclass
class FakeNode:
    """node0 and the laptop's tools, as seen through the controller's runner."""

    clock: FakeClock
    git_files: list[str] = field(default_factory=list)
    statuses: list[StatusEntry] = field(default_factory=list)
    alive: set[int] = field(default_factory=set)
    next_pid: int = 4242
    current_pid: int | None = None
    ssh_ok: bool = True
    rsync_rc: int = 0
    scp_rc: int = 0
    kb_rc: int = 0
    sync_reply: str | None = None
    synced_names: list[str] = field(default_factory=list)
    launches: list[Call] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    snapshots: int = 0

    def __call__(
        self, argv, *, input_data=None, timeout=None, env=None
    ) -> CommandResult:
        call = Call(list(argv), input_data, None if env is None else dict(env))
        self.calls.append(call)
        if call.tool == "git":
            return self._ok("\0".join(self.git_files) + "\0")
        if call.tool == "ssh":
            return self._ssh(call)
        if call.tool == "rsync":
            return self._copy(
                call, self.rsync_rc, "rsync: connection unexpectedly closed"
            )
        if call.tool == "scp":
            return self._copy(call, self.scp_rc, "scp: no such file")
        if call.argv[1:4] == ["-m", "benchmarks.cluster_bench", "kb"]:
            return CommandResult("local", "kb", self.kb_rc, "merged 3 records", "")
        raise AssertionError(f"unexpected command {call.argv}")

    @staticmethod
    def _ok(stdout: str) -> CommandResult:
        return CommandResult("local", "fake", 0, stdout, "")

    def _copy(self, call: Call, rc: int, error: str) -> CommandResult:
        if rc != 0:
            return CommandResult("local", call.tool, rc, "", error)
        dest = Path(call.argv[-1])
        (dest / "analysis").mkdir(parents=True, exist_ok=True)
        (dest / "analysis" / "summary.json").write_text("{}")
        return self._ok("")

    def _ssh(self, call: Call) -> CommandResult:
        if not self.ssh_ok:
            return CommandResult(
                "local",
                "ssh",
                255,
                "",
                "ssh: connect to host 203.0.113.10: Connection refused",
            )
        script = call.script
        if script == "true":
            return self._ok("")
        if "@@SYNC_OK" in script:
            return self._extract(call)
        if "bootstrap_node.sh" in script:
            return self._ok(
                "[bootstrap_node] venv ready\n/root/cluster_bench/venv/bin/python\n"
            )
        if "@@PID" in script:
            pid = self.next_pid
            self.next_pid += 1
            self.alive.add(pid)
            self.current_pid = pid
            self.launches.append(call)
            return self._ok(f"@@NOW {int(self.clock.now())}\n@@PID {pid}\n")
        if "@@STATUS-BEGIN" in script:
            return self._snapshot(script)
        if "@@STOPPED" in script:
            for pid in re.findall(r"for p in ([0-9 ]+); do", script)[0].split():
                self.alive.discard(int(pid))
            return self._ok("@@STOPPED\n")
        if script.startswith("tail -n"):
            return self._ok(
                "Traceback (most recent call last):\nPreflightError: no HCA\n"
            )
        raise AssertionError(f"unexpected remote script:\n{script}")

    def _extract(self, call: Call) -> CommandResult:
        if self.sync_reply is not None:
            return self._ok(self.sync_reply)
        assert call.input_data is not None
        with tarfile.open(fileobj=io.BytesIO(call.input_data)) as tar:
            names = tar.getnames()
            manifest_file = tar.extractfile(SYNC_MANIFEST)
            assert manifest_file is not None
            manifest = manifest_file.read().decode()
            for line in manifest.splitlines():
                digest, rel = line.split("  ", 1)
                member = tar.extractfile(rel)
                assert member is not None
                assert hashlib.sha256(member.read()).hexdigest() == digest
        self.synced_names = names
        return self._ok(f"@@SYNC_OK {len(manifest.splitlines())}\n")

    def _snapshot(self, script: str) -> CommandResult:
        self.snapshots += 1
        status = None
        if self.statuses:
            entry = self.statuses[min(self.snapshots - 1, len(self.statuses) - 1)]
            status = entry(self)
        lines: list[str] = []
        pids: list[int] = []
        if status is not None:
            lines += ["@@STATUS-BEGIN", json.dumps(status, indent=2), "@@STATUS-END"]
            if isinstance(status.get("pid"), int):
                pids.append(status["pid"])
        extra = re.search(r"for p in \$spid ([0-9 ]*); do", script)
        pids += [int(p) for p in extra.group(1).split()] if extra else []
        lines += [f"@@ALIVE {p} {int(p in self.alive)}" for p in pids]
        lines.append(f"@@NOW {int(self.clock.now())}")
        return self._ok("\n".join(lines) + "\n")

    def ssh_scripts(self, marker: str) -> list[Call]:
        return [c for c in self.calls if c.tool == "ssh" and marker in c.script]


def status_from(name: str, **updates: Any) -> dict[str, Any]:
    data = json.loads((CONTROLLER_FIXTURES / name).read_text())
    data.update(updates)
    return data


def running(age: float = 10.0) -> StatusEntry:
    return lambda node: status_from(
        "status_running.json", pid=node.current_pid, heartbeat_ts=node.clock.now() - age
    )


def stale_old_runner(age: float) -> StatusEntry:
    return lambda node: status_from(
        "status_running.json", pid=4242, heartbeat_ts=node.clock.now() - age
    )


def finished(name: str = "status_done.json") -> StatusEntry:
    def entry(node: FakeNode) -> dict[str, Any]:
        node.alive.discard(node.current_pid or -1)
        return status_from(name, pid=node.current_pid, heartbeat_ts=node.clock.now())

    return entry


def seeded_state(**overrides: Any) -> ControllerState:
    base: dict[str, Any] = {
        "exp_id": "exp42",
        "cluster_name": "cb-b200-exp42",
        "cluster_id": B200_ID,
        "created_by_harness": True,
        "fabric": "ib-sjc1-b200-a7f3",
        "gpu": "b200",
        "node_count": 4,
        "ttl_hours": 12.0,
        "node0_ssh": {"target": "root@203.0.113.10", "port": 40122},
        "control_ips": ["10.42.0.1", "10.42.0.2", "10.42.0.3", "10.42.0.4"],
        "workdir": "/root/cluster_bench",
        "created_at": CLOCK_START - 3600,
    }
    base.update(overrides)
    return ControllerState.model_validate(base)


def make(
    tmp_path: Path,
    node: FakeNode,
    *,
    repo_root: Path | None = None,
    exp_id: str = "exp42",
) -> Controller:
    return Controller(
        exp_id,
        state_dir=tmp_path / "state",
        lium=LiumCli(sleep=lambda _s: None),
        runner=node,
        clock=node.clock,
        repo_root=repo_root or tmp_path / "repo",
        ssh_identity=tmp_path / "id_ed25519",
    )


def assert_no_secret_anywhere(node: FakeNode, lium: FakeLium | None = None) -> None:
    for call in node.calls:
        joined = "\0".join(call.argv)
        assert HF_TOKEN not in joined and API_KEY not in joined, call.argv
        if call.env is not None:
            assert "HF_TOKEN" not in call.env and "LIUM_API_KEY" not in call.env
    for call in lium.calls() if lium else []:
        joined = "\0".join(call["argv"])
        assert HF_TOKEN not in joined and API_KEY not in joined, call["argv"]


def make_repo(root: Path) -> list[str]:
    files = {
        "benchmarks/__init__.py": "",
        "benchmarks/cluster_bench/__init__.py": "",
        "benchmarks/cluster_bench/cli.py": "print('cli')\n",
        "benchmarks/cluster_bench/cluster_bench.sh": "#!/bin/bash\n",
        "benchmarks/cluster_bench/__pycache__/cli.cpython-312.pyc": "junk",
        "benchmarks/other/tool.py": "x = 1\n",
        "examples/common/launch_utils.sh": "#!/bin/bash\n",
        "examples/common/gpu_utils.md": "# doc\n",
        "recipes/glm-5.3/sglang/agg-b200-agentic/deploy.yaml": "kind: DGD\n",
        "recipes/glm-5.3/README.md": "# GLM\n",
        "recipes/deepseek-r1/sglang/deploy.yaml": "kind: DGD\n",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "benchmarks/cluster_bench/cluster_bench.sh").chmod(0o755)
    return sorted(files)


SYNCED = [
    "benchmarks/__init__.py",
    "benchmarks/cluster_bench/__init__.py",
    "benchmarks/cluster_bench/cli.py",
    "benchmarks/cluster_bench/cluster_bench.sh",
    "examples/common/launch_utils.sh",
    "recipes/glm-5.3/sglang/agg-b200-agentic/deploy.yaml",
]


# --------------------------------------------------------------------------
# State file
# --------------------------------------------------------------------------


def test_state_round_trip_permissions_and_strictness(tmp_path: Path) -> None:
    ctl = make(tmp_path, FakeNode(FakeClock()))
    state = seeded_state(runner_args=["--budget-hours", "8"])
    ctl.save_state(state)
    assert ctl.load_state() == state
    assert stat.S_IMODE(ctl.state_path.stat().st_mode) == 0o600
    assert ctl.state_path == tmp_path / "state" / "exp42" / "controller.json"
    data = json.loads(ctl.state_path.read_text())
    for key in (
        "cluster_name",
        "created_by_harness",
        "fabric",
        "node0_ssh",
        "control_ips",
        "workdir",
        "exp_id",
        "created_at",
        "pulled_ok",
    ):
        assert key in data
    assert data["node0_ssh"] == {"target": "root@203.0.113.10", "port": 40122}
    data["surprise"] = 1
    ctl.state_path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        ctl.load_state()
    with pytest.raises(ValidationError):
        state.evolve(workdir="relative/dir")
    with pytest.raises(ControllerError):
        make(tmp_path, FakeNode(FakeClock()), exp_id="venv")
    with pytest.raises(ControllerError):
        make(tmp_path, FakeNode(FakeClock()), exp_id="../escape")


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, synced",
    [
        ("benchmarks/__init__.py", True),
        ("benchmarks/cluster_bench/adapters/providers.py", True),
        ("benchmarks/cluster_bench/__pycache__/x.cpython-312.pyc", False),
        ("benchmarks/cluster_bench", False),
        ("benchmarks/other/tool.py", False),
        ("examples/common/launch_utils.sh", True),
        ("examples/common/gpu_utils.md", False),
        ("examples/common/sub/x.sh", False),
        ("recipes/glm-5.3/sglang/disagg-h200-agentic/deploy.yaml", True),
        ("recipes/glm-5.3/README.md", False),
        ("recipes/deepseek-r1/sglang/deploy.yaml", False),
    ],
)
def test_is_sync_path(path: str, synced: bool) -> None:
    assert ctl_mod.is_sync_path(path) is synced


def test_sync_bundle_is_a_verifiable_root_owned_tar(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    make_repo(repo)
    bundle = ctl_mod.build_sync_bundle(repo, SYNCED)
    assert bundle.count == len(SYNCED)
    assert bundle.digest == hashlib.sha256(bundle.manifest.encode()).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(bundle.data)) as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert sorted(members) == sorted([*SYNCED, SYNC_MANIFEST])
        assert all(m.uid == 0 and m.gid == 0 for m in members.values())
        assert members["benchmarks/cluster_bench/cluster_bench.sh"].mode == 0o755
        assert members["benchmarks/cluster_bench/cli.py"].mode == 0o644
    for line in bundle.manifest.splitlines():
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((repo / rel).read_bytes()).hexdigest() == digest
    again = ctl_mod.build_sync_bundle(repo, SYNCED)
    assert again.data == bundle.data  # deterministic


def test_list_sync_files_filters_git_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    listed = make_repo(repo) + ["benchmarks/cluster_bench/deleted.py"]
    node = FakeNode(FakeClock(), git_files=listed)
    assert ctl_mod.list_sync_files(repo, node) == SYNCED
    git = node.calls[0].argv
    assert git[:3] == ["git", "-C", str(repo)]
    assert "--others" in git and "--exclude-standard" in git and "-z" in git
    with pytest.raises(ControllerError, match="no benchmarks/cluster_bench files"):
        ctl_mod.list_sync_files(repo, FakeNode(FakeClock(), git_files=["README.md"]))


def fabric(
    fid: str, kind: str, gpu: str, free: int, price: float, measured: bool
) -> LiumFabric:
    return LiumFabric(
        fid, kind, "400 Gb/sec (4X NDR)", measured, 8, free, gpu, 8, price
    )


def test_pick_fabric_prefers_measured_infiniband_then_price() -> None:
    fabrics = [
        fabric("roce-cheap", "roce", "B200", 6, 31.5, True),
        fabric("ib-pricey", "infiniband", "B200", 5, 38.4, True),
        fabric("ib-unmeasured", "infiniband", "B200", 8, 30.0, False),
        fabric("ib-h200", "infiniband", "H200", 2, 25.6, False),
    ]
    assert ctl_mod.pick_fabric(fabrics, "b200", 4).fabric_id == "ib-pricey"
    assert ctl_mod.pick_fabric(fabrics, "B200", 6).fabric_id == "roce-cheap"
    assert ctl_mod.pick_fabric(fabrics, "b200", 7).fabric_id == "ib-unmeasured"
    with pytest.raises(ControllerError, match="no fabric has 4 free H200"):
        ctl_mod.pick_fabric(fabrics, "h200", 4)
    assert ctl_mod.find_fabric(fabrics, "ib-p").fabric_id == "ib-pricey"
    assert ctl_mod.find_fabric(fabrics, "ib-") is None  # ambiguous prefix


def test_names_and_gpu_normalization() -> None:
    assert ctl_mod.default_cluster_name("b200", "exp42") == "cb-b200-exp42"
    assert (
        ctl_mod.default_cluster_name("RTX PRO 6000", "Exp_42.a")
        == "cb-rtx-pro-6000-exp-42-a"
    )
    assert len(ctl_mod.default_cluster_name("h200", "x" * 90)) <= 63
    assert ctl_mod.normalize_gpu("rtx_pro_6000") == "RTXPRO6000"
    assert ctl_mod.normalize_gpu("pro6000") == "RTXPRO6000"
    assert ctl_mod.normalize_gpu("h100") == "H100"


def test_left_running_banner() -> None:
    banner = ctl_mod.left_running_banner(seeded_state(), "--keep")
    assert "CLUSTER LEFT RUNNING" in banner and "--keep" in banner
    assert f"lium clusters rm {B200_ID} -y" in banner
    assert "controller down --exp-id exp42" in banner
    adopted = ctl_mod.left_running_banner(
        seeded_state(created_by_harness=False), "adopted"
    )
    assert "never removes it" in adopted and "controller down" not in adopted
    lines = banner.splitlines()
    assert lines[0] == lines[-1] and set(lines[0]) == {"!"}


def test_runner_argv_and_managed_flags() -> None:
    assert ctl_mod.runner_argv("e1", "/w", "10.42.0.1,10.42.0.2", 2, ["--budget-hours", "8"]) == [
        "--exp-id", "e1", "--workdir", "/w", "--platform", "lium",
        "--nodes", "10.42.0.1,10.42.0.2", "--budget-hours", "8",
    ]  # fmt: skip
    assert ctl_mod.runner_argv("e1", "/w", "auto", 4)[-2:] == ["--expect-nodes", "4"]
    assert ctl_mod.runner_argv(
        "e1", "/w", None, 2, hosts="/w/experiments/e1/hosts.txt", platform="generic"
    ) == [
        "--exp-id", "e1", "--workdir", "/w", "--platform", "generic",
        "--hosts", "/w/experiments/e1/hosts.txt",
    ]  # fmt: skip
    for bad in (
        ["--exp-id", "x"],
        ["--workdir=/tmp"],
        ["--nodes", "auto"],
        ["--platform", "generic"],
    ):
        with pytest.raises(ControllerError, match="set by the controller"):
            ctl_mod.check_runner_args(bad)
    ctl_mod.check_runner_args(["--families", "agg-tp8", "--budget-hours", "9.5"])


def snap(status: dict[str, Any] | None, now: float, **alive: bool) -> RemoteSnapshot:
    return RemoteSnapshot(status, now, {int(k[1:]): v for k, v in alive.items()})


@pytest.mark.parametrize(
    "status, alive, launched_ts, action, reason",
    [
        (None, {"p4242": True}, 900.0, "wait", "starting"),
        (None, {"p4242": False}, 900.0, "failed", "exited before writing"),
        (
            {"state": "running", "pid": 4242, "heartbeat_ts": 990.0},
            {"p4242": True},
            900.0,
            "wait",
            "running",
        ),
        (
            {"state": "running", "pid": 4242, "heartbeat_ts": 600.0},
            {"p4242": True},
            500.0,
            "relaunch",
            "400s old",
        ),
        (
            {"state": "running", "pid": 4242, "heartbeat_ts": 990.0},
            {"p4242": False},
            900.0,
            "relaunch",
            "is gone",
        ),
        (
            {"state": "done", "pid": 4242, "heartbeat_ts": 990.0},
            {"p4242": False},
            900.0,
            "done",
            "finished",
        ),
        (
            {"state": "failed", "error": "boom\ntrace", "heartbeat_ts": 990.0},
            {},
            900.0,
            "failed",
            "boom",
        ),
        (
            {"state": "interrupted", "error": "signal 15", "heartbeat_ts": 990.0},
            {},
            900.0,
            "relaunch",
            "signal 15",
        ),
        # status.json of the previous runner while the new one starts up
        (
            {"state": "failed", "pid": 4100, "heartbeat_ts": 800.0},
            {"p4242": True},
            900.0,
            "wait",
            "starting",
        ),
        (
            {"state": "running", "pid": 4242, "heartbeat_ts": 990.0},
            {"p4242": True},
            None,
            "wait",
            "running",
        ),
    ],
)
def test_decide_watch(
    status: dict[str, Any] | None,
    alive: dict[str, bool],
    launched_ts: float | None,
    action: str,
    reason: str,
) -> None:
    decision = ctl_mod.decide_watch(
        snap(status, 1000.0, **alive),
        launched_pid=4242,
        launched_remote_ts=launched_ts,
        stale_after_s=300,
    )
    assert decision.action == action and reason in decision.reason


def test_decide_watch_without_any_launch_fails_fast() -> None:
    decision = ctl_mod.decide_watch(
        snap(None, 1000.0),
        launched_pid=None,
        launched_remote_ts=None,
        stale_after_s=300,
    )
    assert decision.action == "failed" and "controller launch" in decision.reason


def test_format_progress() -> None:
    status = status_from("status_running.json", heartbeat_ts=1000.0)
    status["budget"]["spent_s"] = 3 * 3600
    line = ctl_mod.format_progress(status, 1025.0)
    for part in (
        "running",
        "phase=search",
        "spent=3.0h",
        "left=6.2h",
        "points=12",
        "last=312.4 tok/s/GPU (c=64, valid)",
        "hb=25s ago",
    ):
        assert part in line
    assert ctl_mod.format_progress(None, 0) == "no status.json yet"


def test_parse_snapshot_fails_loudly_on_garbage() -> None:
    parsed = ctl_mod.parse_snapshot(
        "@@STATUS-BEGIN\n{}\n@@STATUS-END\n@@ALIVE 7 1\n@@NOW 5\n"
    )
    assert parsed == RemoteSnapshot({}, 5.0, {7: True})
    with pytest.raises(ControllerError, match="not JSON"):
        ctl_mod.parse_snapshot("@@STATUS-BEGIN\n{oops\n@@STATUS-END\n@@NOW 5\n")
    with pytest.raises(ControllerError, match="@@NOW"):
        ctl_mod.parse_snapshot("motd banner only\n")


@needs_bash
def test_remote_snippets_are_valid_bash() -> None:
    snippets = [
        ctl_mod.extract_script("/root/cluster_bench"),
        ctl_mod.bootstrap_command("/root/cluster_bench"),
        ctl_mod.launch_script("/root/cluster_bench", "exp42", ["--exp-id", "exp42"]),
        ctl_mod.snapshot_script(
            "/root/cluster_bench/experiments/exp42/status.json", "exp42", []
        ),
        ctl_mod.snapshot_script("/x/status.json", "exp42", [4242]),
        ctl_mod.stop_script("exp42", [4242, 4243]),
    ]
    for script in snippets:
        res = subprocess.run(
            [BASH, "-n", "-c", script], capture_output=True, text=True, timeout=60
        )
        assert res.returncode == 0, f"{res.stderr}\n{script}"


@needs_bash
def test_snapshot_script_output_parses(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text((CONTROLLER_FIXTURES / "status_running.json").read_text())
    script = ctl_mod.snapshot_script(str(status_path), "exp42", [999999])
    res = subprocess.run(
        [BASH, "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    parsed = ctl_mod.parse_snapshot(res.stdout)
    assert parsed.status == json.loads(status_path.read_text())
    assert parsed.alive == {4242: False, 999999: False}
    absent = subprocess.run(
        [BASH, "-c", ctl_mod.snapshot_script(str(tmp_path / "none.json"), "exp42", [])],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert ctl_mod.parse_snapshot(absent.stdout).status is None


# --------------------------------------------------------------------------
# up / adopt
# --------------------------------------------------------------------------


def test_up_picks_a_fabric_rents_records_and_waits_for_ssh(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    fake_lium.script(
        clusters_list=[ok("clusters_list.json")],
        clusters_ps=[ok("clusters_ps.json")],
        ps=[ok("ps.json")],
        clusters_up=[ok("clusters_up_b200x4.json")],
    )
    node = FakeNode(FakeClock())
    ctl = make(tmp_path, node)
    state = ctl.up(UpOptions(gpu="b200"))
    assert state.created_by_harness is True
    assert (state.cluster_name, state.cluster_id) == ("cb-b200-exp42", B200_ID)
    assert state.fabric == "ib-sjc1-b200-a7f3"
    assert state.node0_ssh is not None
    assert (state.node0_ssh.target, state.node0_ssh.port) == (
        "root@203.0.113.10",
        40122,
    )
    assert state.control_ips == ("10.42.0.1", "10.42.0.2", "10.42.0.3", "10.42.0.4")
    assert [m.ssh.port for m in state.members if m.ssh] == [40122, 40222, 40322, 40422]
    assert ctl.load_state() == state
    # The offer's metadata is kept for provider.json ($/token in the report).
    assert (state.fabric_type, state.price_per_node_hour) == ("infiniband", 38.4)
    assert state.link_rate == "400 Gb/sec (4X NDR)"
    up = next(c for c in fake_lium.calls() if c["key"] == "clusters up")["argv"]
    assert (
        up[2] == "ib-sjc1-b200-a7f3" and up[up.index("--name") + 1] == "cb-b200-exp42"
    )
    assert up[up.index("--ttl") + 1] == "12h" and "--no-backup" in up
    ssh = node.calls[-1].argv
    assert ssh[ssh.index("-p") + 1] == "40122" and "root@203.0.113.10" in ssh
    assert "BatchMode=yes" in ssh and "StrictHostKeyChecking=accept-new" in ssh
    assert f'UserKnownHostsFile="{ctl.known_hosts}"' in ssh
    assert_no_secret_anywhere(node, fake_lium)


def test_up_refuses_a_cluster_size_no_tuned_family_fits(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    ctl = make(tmp_path, FakeNode(FakeClock()))
    with pytest.raises(ControllerError, match="no tuned glm-5.3 family fits 3 x 8"):
        ctl.up(UpOptions(gpu="h100", nodes=3))
    with pytest.raises(ControllerError, match="at least 2 nodes"):
        ctl.up(UpOptions(gpu="b200", nodes=1))
    assert not [c for c in fake_lium.calls() if c["key"] == "clusters up"]
    assert ctl_mod.size_problems("b200", 8) == ctl_mod.size_problems("h200", 2) == []


@needs_bash
def test_launch_writes_provider_json_without_secrets_or_addresses(
    tmp_path: Path,
) -> None:
    state = ctl_mod.ControllerState(
        exp_id="exp42",
        cluster_name="cb-b200-exp42",
        cluster_id=B200_ID,
        created_by_harness=True,
        fabric="ib-sjc1-b200-a7f3",
        fabric_type="infiniband",
        link_rate="400 Gb/sec (4X NDR)",
        price_per_node_hour=38.4,
        cluster_price_per_hour=153.6,
        gpu="b200",
        node_count=2,
        workdir="/root/cluster_bench",
        created_at=CLOCK_START,
        node0_ssh={"target": "root@203.0.113.10", "port": 40122},
        members=[
            {
                "name": "node0",
                "rank": 0,
                "huid": "brave-otter-3c",
                "pod_id": "p0",
                "ssh": {"target": "root@203.0.113.10", "port": 40122},
                "control_ip": "10.42.0.1",
            }
        ],
    )
    record = ctl_mod.provider_record(state)
    workdir = tmp_path / "w"
    script = ctl_mod.launch_script(str(workdir), "exp42", ["--exp-id", "exp42"], record)
    assert "203.0.113.10" not in script and HF_TOKEN not in script
    # Run only the provider.json part (everything before the runner starts).
    head = script.split("cd ", 1)[0]
    res = subprocess.run(
        [BASH, "-c", head], input="\n", capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0, res.stderr
    written = json.loads((workdir / "experiments/exp42/provider.json").read_text())
    assert written["price_per_node_hour"] == 38.4
    assert written["fabric_type"] == "infiniband"
    assert written["members"][0]["huid"] == "brave-otter-3c"
    assert "ssh" not in written["members"][0]


@pytest.mark.parametrize("name", ["dev-box", "train-job"])
def test_up_refuses_a_name_that_exists(
    tmp_path: Path, fake_lium: FakeLium, name: str
) -> None:
    fake_lium.script(
        clusters_list=[ok("clusters_list.json")],
        clusters_ps=[ok("clusters_ps.json")],
        ps=[ok("ps.json")],
        clusters_up=[ok("clusters_up_b200x4.json")],
    )
    ctl = make(tmp_path, FakeNode(FakeClock()))
    with pytest.raises(ControllerError, match="already exists"):
        ctl.up(UpOptions(gpu="b200", name=name))
    assert "clusters up" not in fake_lium.keys()
    assert not ctl.state_path.exists()


def test_up_rejects_a_gpu_that_does_not_match_the_fabric(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    fake_lium.script(clusters_list=[ok("clusters_list.json")])
    ctl = make(tmp_path, FakeNode(FakeClock()))
    with pytest.raises(ControllerError, match="has B200 nodes, not h200"):
        ctl.up(UpOptions(gpu="h200", fabric="ib-sjc1-b200-a7f3"))
    with pytest.raises(ControllerError, match="needs --gpu"):
        ctl.up(UpOptions())


def test_adopt_records_a_cluster_the_harness_must_not_remove(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    train = json.loads((LIUM_FIXTURES / "clusters_ps.json").read_text())[0]
    show = tmp_path / "train_show.json"
    show.write_text(json.dumps(train))
    fake_lium.script(
        clusters_ps=[ok("clusters_ps.json")],
        clusters_show=[{"rc": 0, "stdout_file": str(show)}],
        clusters_rm=[ok("clusters_rm_ok.json")],
    )
    node = FakeNode(FakeClock())
    ctl = make(tmp_path, node)
    state = ctl.up(UpOptions(adopt="train-job"))
    assert state.created_by_harness is False
    assert state.cluster_name == "train-job" and state.node0_ssh is not None
    assert state.node0_ssh.target == "root@198.51.100.21"
    assert ctl.down(force=True) == 0
    assert "clusters up" not in fake_lium.keys()
    assert "clusters rm" not in fake_lium.keys()
    assert ctl.load_state().destroyed_at is None


def test_up_failure_after_renting_records_the_cluster_and_warns(
    tmp_path: Path, fake_lium: FakeLium, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_lium.script(
        clusters_list=[ok("clusters_list.json")],
        clusters_ps=[ok("clusters_ps.json"), ok("clusters_ps_with_b200.json")],
        ps=[ok("ps.json")],
        clusters_up=[fail(3, "err_rented_not_listed.txt")],
    )
    ctl = make(tmp_path, FakeNode(FakeClock()))
    with pytest.raises(LiumBillingUncertainError):
        ctl.up(UpOptions(gpu="b200"))
    assert fake_lium.keys().count("clusters up") == 1
    state = ctl.load_state()
    assert state.cluster_id == B200_ID and state.created_by_harness is True
    assert "CLUSTER LEFT RUNNING" in capsys.readouterr().err
    with pytest.raises(ControllerError, match="already has cluster"):
        ctl.up(UpOptions(gpu="b200"))
    assert fake_lium.keys().count("clusters up") == 1


def test_up_failure_before_renting_allows_a_clean_retry(
    tmp_path: Path, fake_lium: FakeLium, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_lium.script(
        clusters_list=[ok("clusters_list.json")],
        clusters_ps=[ok("clusters_ps.json")],
        ps=[ok("ps.json")],
        clusters_up=[
            fail(1, "err_not_enough_nodes.txt"),
            ok("clusters_up_b200x4.json"),
        ],
    )
    ctl = make(tmp_path, FakeNode(FakeClock()))
    with pytest.raises(LiumGeneralError):
        ctl.up(UpOptions(gpu="b200"))
    assert "CLUSTER LEFT RUNNING" not in capsys.readouterr().err
    assert ctl.load_state().cluster_id is None
    assert ctl.up(UpOptions(gpu="b200")).cluster_id == B200_ID


# --------------------------------------------------------------------------
# down
# --------------------------------------------------------------------------


def test_down_safety_rules(
    tmp_path: Path, fake_lium: FakeLium, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_lium.script(clusters_rm=[ok("clusters_rm_ok.json")])
    ctl = make(tmp_path, FakeNode(FakeClock()))

    ctl.save_state(seeded_state(created_by_harness=False, pulled_ok=True))
    assert ctl.down(force=True) == 0
    assert "adopted" in capsys.readouterr().err

    ctl.save_state(seeded_state(pulled_ok=False))
    assert ctl.down() == 1
    assert "were not pulled" in capsys.readouterr().err

    ctl.save_state(seeded_state(pulled_ok=True))
    assert ctl.down(keep=True) == 0
    assert "CLUSTER LEFT RUNNING" in capsys.readouterr().err
    assert "clusters rm" not in fake_lium.keys()

    assert ctl.down() == 0
    assert fake_lium.calls()[-1]["argv"][:3] == ["clusters", "rm", B200_ID]
    assert ctl.load_state().destroyed_at is not None
    assert ctl.down() == 0
    assert fake_lium.keys().count("clusters rm") == 1


def test_down_force_without_pull(tmp_path: Path, fake_lium: FakeLium) -> None:
    fake_lium.script(clusters_rm=[ok("clusters_rm_ok.json")])
    ctl = make(tmp_path, FakeNode(FakeClock()))
    ctl.save_state(seeded_state(pulled_ok=False))
    assert ctl.down(force=True) == 0
    assert fake_lium.keys() == ["clusters rm"]


def test_down_of_a_cluster_that_is_already_gone(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    fake_lium.script(
        clusters_rm=[fail(5, "err_cluster_not_found.txt")],
        clusters_ps=[ok("clusters_ps.json")],
    )
    ctl = make(tmp_path, FakeNode(FakeClock()))
    ctl.save_state(seeded_state(pulled_ok=True))
    assert ctl.down() == 0
    assert ctl.load_state().destroyed_at is not None


# --------------------------------------------------------------------------
# sync / launch
# --------------------------------------------------------------------------


def test_sync_streams_a_verified_bundle_then_bootstraps(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    node = FakeNode(FakeClock(), git_files=make_repo(repo))
    ctl = make(tmp_path, node, repo_root=repo)
    ctl.save_state(seeded_state())
    state = ctl.sync()
    assert sorted(node.synced_names) == sorted([*SYNCED, SYNC_MANIFEST])
    extract = node.ssh_scripts("@@SYNC_OK")[0]
    assert (
        "sha256sum --check" in extract.script
        and 'mv "$W/src.incoming" "$W/src"' in extract.script
    )
    boot = node.ssh_scripts("bootstrap_node.sh")[0].script
    assert boot == (
        "bash /root/cluster_bench/src/benchmarks/cluster_bench/scripts/bootstrap_node.sh "
        "--workdir /root/cluster_bench"
    )
    assert state.venv_python == "/root/cluster_bench/venv/bin/python"
    assert state.sync_manifest_sha256 and len(state.sync_manifest_sha256) == 64


def test_sync_fails_with_raw_output_when_verification_is_missing(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    node = FakeNode(
        FakeClock(), git_files=make_repo(repo), sync_reply="tar: unexpected EOF\n"
    )
    ctl = make(tmp_path, node, repo_root=repo)
    ctl.save_state(seeded_state())
    with pytest.raises(ControllerError, match="tar: unexpected EOF"):
        ctl.sync()
    assert not node.ssh_scripts("bootstrap_node.sh")


def test_sync_refuses_while_the_runner_is_alive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    node = FakeNode(FakeClock(), git_files=make_repo(repo), alive={4242})
    ctl = make(tmp_path, node, repo_root=repo)
    ctl.save_state(seeded_state(runner_pid=4242))
    with pytest.raises(ControllerError, match="is alive"):
        ctl.sync()
    ctl.sync(force=True)


def test_launch_sends_hf_token_on_stdin_only(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    node = FakeNode(FakeClock())
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(pulled_ok=True))
    state = ctl.launch(runner_args=["--budget-hours", "8"])
    launch = node.launches[0]
    assert launch.input_data == f"{HF_TOKEN}\n".encode()
    script = launch.script
    assert "IFS= read -r HF_TOKEN" in script and "export HF_TOKEN" in script
    assert "setsid nohup bash benchmarks/cluster_bench/cluster_bench.sh run" in script
    assert (
        "--exp-id exp42 --workdir /root/cluster_bench --platform lium "
        "--nodes 10.42.0.1,10.42.0.2,10.42.0.3,10.42.0.4 --budget-hours 8"
    ) in script
    assert (
        ">> /root/cluster_bench/experiments/exp42/runner.log 2>&1 < /dev/null &"
        in script
    )
    assert "cd /root/cluster_bench/src" in script
    assert state.runner_pid == 4242 and state.launches == 1
    assert state.launched_remote_ts == int(CLOCK_START)
    assert state.runner_args == ("--budget-hours", "8") and state.pulled_ok is False
    assert_no_secret_anywhere(node, fake_lium)
    with pytest.raises(ControllerError, match="already alive"):
        ctl.launch()
    with pytest.raises(ControllerError, match="set by the controller"):
        ctl.launch(runner_args=["--exp-id", "other"], force=True)


def test_a_secret_on_any_controller_argv_is_refused(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    node = FakeNode(FakeClock())
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state())
    with pytest.raises(ValueError, match="secret"):
        ctl.launch(runner_args=["--note", f"token={HF_TOKEN}"])
    assert node.launches == [] and node.calls == []
    with pytest.raises(ValueError, match="secret"):
        ctl.pull(tmp_path / API_KEY)
    assert not any(c.tool == "rsync" for c in node.calls)


def test_launch_uses_auto_discovery_without_known_ips(tmp_path: Path) -> None:
    node = FakeNode(FakeClock())
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(control_ips=[]))
    ctl.launch()
    assert "--nodes auto --expect-nodes 4" in node.launches[0].script


def test_launch_without_hf_token_sends_an_empty_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    node = FakeNode(FakeClock())
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state())
    ctl.launch()
    assert node.launches[0].input_data == b"\n"
    assert "HF_TOKEN is not set" in capsys.readouterr().err


# --------------------------------------------------------------------------
# watch / pull
# --------------------------------------------------------------------------


def test_watch_relaunches_a_stale_runner_then_finishes(tmp_path: Path) -> None:
    clock = FakeClock()
    node = FakeNode(clock, alive={4242}, next_pid=4243, current_pid=4242)
    node.statuses = [
        running(age=10),
        running(age=400),  # stale heartbeat -> stop pid 4242, relaunch
        stale_old_runner(age=460),  # previous runner's status; pid 4243 starting
        running(age=10),
        finished(),
    ]
    ctl = make(tmp_path, node)
    ctl.save_state(
        seeded_state(
            runner_pid=4242,
            launched_remote_ts=CLOCK_START - 600,
            launches=1,
            runner_args=["--budget-hours", "8"],
        )
    )
    opts = WatchOptions(interval_s=60, pull_every_s=120, stale_after_s=300)
    assert ctl.watch(opts, results_dir=tmp_path / "results") == 0
    state = ctl.load_state()
    assert (state.relaunches, state.launches, state.runner_pid) == (1, 2, 4243)
    assert "for p in 4242; do" in node.ssh_scripts("@@STOPPED")[0].script
    assert "--budget-hours 8" in node.launches[0].script
    assert 4242 not in node.alive
    assert any(c.tool == "rsync" for c in node.calls)  # incremental pull
    assert state.pulled_ok is False  # an incremental pull never sets it
    assert set(clock.sleeps) == {60}


def test_watch_gives_up_after_max_relaunches(tmp_path: Path) -> None:
    node = FakeNode(FakeClock(), alive={4242}, next_pid=4243, current_pid=4242)
    node.statuses = [
        lambda n: status_from(
            "status_interrupted.json", pid=n.current_pid, heartbeat_ts=n.clock.now()
        )
    ]
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242, launched_remote_ts=CLOCK_START - 600))
    assert ctl.watch(WatchOptions(max_relaunch=2)) == 1
    state = ctl.load_state()
    assert state.relaunches == 2 and len(node.launches) == 2


def test_watch_reports_a_failed_run_with_the_log_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    node = FakeNode(FakeClock(), current_pid=4242)
    node.statuses = [
        lambda n: status_from("status_failed.json", heartbeat_ts=n.clock.now())
    ]
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242, launched_remote_ts=CLOCK_START - 600))
    assert ctl.watch() == 1
    err = capsys.readouterr().err
    assert "runner failed: preflight failed:" in err and "PreflightError: no HCA" in err


def test_watch_gives_up_when_node0_stays_unreachable(tmp_path: Path) -> None:
    node = FakeNode(FakeClock(), ssh_ok=False)
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242))
    opts = WatchOptions(max_poll_failures=3, pull_every_s=1e9)
    assert ctl.watch(opts) == ctl_mod.WATCH_LOST
    assert len(node.calls) == 3


def test_pull_uses_rsync_and_marks_a_finished_run_pulled(tmp_path: Path) -> None:
    node = FakeNode(FakeClock(), current_pid=4242, statuses=[finished()])
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242))
    dest = ctl.pull(tmp_path / "results")
    assert dest == (tmp_path / "results").resolve()
    rsync = next(c for c in node.calls if c.tool == "rsync").argv
    assert rsync[:3] == ["rsync", "-a", "--exclude=/.ssh/"]
    # raw Nsight Systems captures stay on the nodes (their CSVs are pulled)
    assert "--exclude=*.nsys-rep" in rsync and "--exclude=*.sqlite" in rsync
    assert rsync[-2:] == [
        "root@203.0.113.10:/root/cluster_bench/experiments/exp42/",
        f"{dest}/",
    ]
    ssh_e = shlex.split(rsync[rsync.index("-e") + 1])
    assert ssh_e[:3] == ["ssh", "-p", "40122"] and "BatchMode=yes" in ssh_e
    state = ctl.load_state()
    assert state.pulled_ok is True and state.results_dir == str(dest)


def test_pull_while_running_is_only_a_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    node = FakeNode(FakeClock(), alive={4242}, current_pid=4242, statuses=[running()])
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242, pulled_ok=True))
    ctl.pull(tmp_path / "results")
    assert ctl.load_state().pulled_ok is False
    assert "snapshot" in capsys.readouterr().err
    assert node.clock.sleeps == []  # a running runner is not waited for


def test_final_pull_waits_for_a_finished_runner_to_exit(tmp_path: Path) -> None:
    def done_but_exiting(node: FakeNode) -> dict[str, Any]:
        return status_from("status_done.json", heartbeat_ts=node.clock.now())

    node = FakeNode(FakeClock(), alive={4242}, current_pid=4242)
    node.statuses = [done_but_exiting, done_but_exiting, finished()]
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242))
    ctl.pull(tmp_path / "results")
    assert ctl.load_state().pulled_ok is True
    assert node.clock.sleeps == [5, 5]


def test_pull_falls_back_to_scp(tmp_path: Path) -> None:
    node = FakeNode(FakeClock(), current_pid=4242, statuses=[finished()], rsync_rc=12)
    ctl = make(tmp_path, node)
    ctl.save_state(seeded_state(runner_pid=4242))
    dest = ctl.pull(tmp_path / "results")
    scp = next(c for c in node.calls if c.tool == "scp").argv
    assert scp[:3] == ["scp", "-r", "-p"] and scp[scp.index("-P") + 1] == "40122"
    assert scp[-2:] == [
        "root@203.0.113.10:/root/cluster_bench/experiments/exp42/*",
        f"{dest}/",
    ]
    assert ctl.load_state().pulled_ok is True
    node.scp_rc = 1
    with pytest.raises(ControllerError, match="rsync exit 12"):
        ctl.pull(tmp_path / "results")


# --------------------------------------------------------------------------
# run-all and the command line
# --------------------------------------------------------------------------


def run_all_lium(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_list=[ok("clusters_list.json")],
        clusters_ps=[ok("clusters_ps.json")],
        ps=[ok("ps.json")],
        clusters_up=[ok("clusters_up_b200x4.json")],
        clusters_rm=[ok("clusters_rm_ok.json")],
    )


def test_run_all_end_to_end(tmp_path: Path, fake_lium: FakeLium) -> None:
    run_all_lium(fake_lium)
    repo = tmp_path / "repo"
    node = FakeNode(FakeClock(), git_files=make_repo(repo))
    node.statuses = [lambda n: None, running(), finished()]
    ctl = make(tmp_path, node, repo_root=repo)
    rc = ctl.run_all(
        UpOptions(gpu="b200"),
        runner_args=["--budget-hours", "8"],
        watch=WatchOptions(interval_s=60),
        results_dir=tmp_path / "results",
        kb_root=tmp_path / "kb",
    )
    assert rc == 0
    assert fake_lium.keys() == [
        "clusters list",
        "clusters ps",
        "ps",
        "clusters up",
        "clusters rm",
    ]
    state = ctl.load_state()
    assert state.pulled_ok is True and state.destroyed_at is not None
    assert node.launches[0].input_data == f"{HF_TOKEN}\n".encode()
    kb = next(
        c for c in node.calls if c.argv[1:4] == ["-m", "benchmarks.cluster_bench", "kb"]
    )
    dest = str((tmp_path / "results").resolve())
    assert kb.argv == [
        sys.executable, "-m", "benchmarks.cluster_bench", "kb",
        "--exp-dir", dest, "--kb-root", str(tmp_path / "kb"),
    ]  # fmt: skip
    assert kb.env is not None and kb.env["PYTHONSAFEPATH"] == "1"
    assert kb.env["PYTHONPATH"].split(os.pathsep)[0] == str(repo)
    order = [c.tool for c in node.calls if c.tool in ("git", "rsync")]
    assert order[0] == "git" and order[-1] == "rsync"
    assert_no_secret_anywhere(node, fake_lium)


def test_run_all_keep_leaves_the_cluster_with_a_banner(
    tmp_path: Path, fake_lium: FakeLium, capsys: pytest.CaptureFixture[str]
) -> None:
    run_all_lium(fake_lium)
    repo = tmp_path / "repo"
    node = FakeNode(FakeClock(), git_files=make_repo(repo))
    node.statuses = [lambda n: None, finished()]
    ctl = make(tmp_path, node, repo_root=repo)
    assert (
        ctl.run_all(UpOptions(gpu="b200"), keep=True, results_dir=tmp_path / "r") == 0
    )
    assert "clusters rm" not in fake_lium.keys()
    assert "CLUSTER LEFT RUNNING" in capsys.readouterr().err
    assert ctl.load_state().destroyed_at is None


def test_run_all_failed_run_still_pulls_and_tears_down(
    tmp_path: Path, fake_lium: FakeLium
) -> None:
    run_all_lium(fake_lium)
    repo = tmp_path / "repo"
    node = FakeNode(FakeClock(), git_files=make_repo(repo))
    node.statuses = [lambda n: None, finished("status_failed.json")]
    ctl = make(tmp_path, node, repo_root=repo)
    assert (
        ctl.run_all(UpOptions(gpu="b200"), kb_root=None, results_dir=tmp_path / "r")
        == 1
    )
    assert "clusters rm" in fake_lium.keys()
    assert not any(
        c.argv[1:4] == ["-m", "benchmarks.cluster_bench", "kb"] for c in node.calls
    )


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluster_bench")
    sub = parser.add_subparsers(dest="command", required=True)
    ctl_mod.register(sub)
    return parser


def test_register_parses_every_action_and_passthrough() -> None:
    parser = cli_parser()
    for action in (
        "up",
        "sync",
        "launch",
        "watch",
        "pull",
        "down",
        "status",
        "run-all",
    ):
        args = parser.parse_args(["controller", action, "--exp-id", "exp42"])
        assert args.command == "controller" and args.controller_action == action
    args = parser.parse_args(
        ["controller", "launch", "--exp-id", "exp42", "--", "--budget-hours", "8"]
    )
    assert ctl_mod._passthrough(args.runner_args) == ["--budget-hours", "8"]
    args = parser.parse_args(
        [
            "controller", "run-all", "--gpu", "h200", "--fabric", "ib-ams3", "--nodes", "2",
            "--ttl-hours", "10", "--keep", "--no-kb", "--", "--families", "agg-tp8",
        ]
    )  # fmt: skip
    assert (args.gpu, args.fabric, args.nodes, args.ttl_hours) == (
        "h200",
        "ib-ams3",
        2,
        10.0,
    )
    assert args.keep and args.no_kb
    assert ctl_mod._passthrough(args.runner_args) == ["--families", "agg-tp8"]
    down = parser.parse_args(
        ["controller", "down", "--exp-id", "e", "--keep", "--force"]
    )
    assert down.keep and down.force


def test_main_reports_errors_as_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = cli_parser()
    state_dir = [
        "--state-dir",
        str(tmp_path / "state"),
        "--ssh-identity",
        str(tmp_path / "k"),
    ]
    stray = parser.parse_args(
        ["controller", "launch", "--exp-id", "exp42", *state_dir, "stray"]
    )
    assert ctl_mod.main(stray) == 1
    assert "must follow '--'" in capsys.readouterr().err
    missing = parser.parse_args(
        ["controller", "status", "--exp-id", "exp42", *state_dir]
    )
    assert ctl_mod.main(missing) == 1
    assert "no controller state" in capsys.readouterr().err
    no_id = parser.parse_args(["controller", "watch", *state_dir])
    assert ctl_mod.main(no_id) == 1
    assert "needs --exp-id" in capsys.readouterr().err


def test_main_status_prints_the_local_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_dir = tmp_path / "state"
    ctl = make(tmp_path, FakeNode(FakeClock()))
    ctl.save_state(seeded_state(destroyed_at=CLOCK_START))
    args = cli_parser().parse_args(
        [
            "controller",
            "status",
            "--exp-id",
            "exp42",
            "--state-dir",
            str(state_dir),
            "--ssh-identity",
            str(tmp_path / "k"),
        ]
    )
    assert ctl_mod.main(args) == 0
    out = capsys.readouterr().out
    assert "cb-b200-exp42" in out and "root@203.0.113.10 -p 40122" in out
    assert "cluster removed" in out


# --------------------------------------------------------------------------
# Shell scripts
# --------------------------------------------------------------------------

FAKE_PYTHON = '#!/bin/sh\necho "PY=$0"\necho "ARGS=$*"\necho "PYTHONPATH=$PYTHONPATH"\necho "SAFE=$PYTHONSAFEPATH"\n'

FAKE_UV = """#!/bin/sh
echo "uv $*" >> "$FAKE_UV_LOG"
case "$1" in
  --version) echo "uv 0.0.0-fake" ;;
  venv)
    dir="$4"
    mkdir -p "$dir/bin"
    cat > "$dir/bin/python" <<'PY'
#!/bin/sh
case "$2" in *version_info*) echo "3.12" ; exit 0 ;; esac
case "$1" in -c) exit 0 ;; esac
echo "PY=$0"
echo "ARGS=$*"
echo "PYTHONPATH=$PYTHONPATH"
PY
    chmod +x "$dir/bin/python"
    ;;
esac
exit 0
"""


def write_exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def script_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    fakebin = tmp_path / "fakebin"
    write_exe(fakebin / "uv", FAKE_UV)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fakebin}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "FAKE_UV_LOG": str(tmp_path / "uv.log"),
        # Never let a test run as root reach the real apt-get.
        "CB_FABRIC_TOOLS": "0",
    }
    env.update(extra)
    return env


@needs_bash
@pytest.mark.parametrize("script", [CLUSTER_BENCH_SH, BOOTSTRAP_SH])
def test_shell_scripts_parse(script: Path) -> None:
    res = subprocess.run(
        [BASH, "-n", str(script)], capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0, res.stderr
    text = script.read_text()
    assert text.startswith("#!/bin/bash\n# SPDX-FileCopyrightText:")


@needs_bash
def test_cluster_bench_sh_python_choice(tmp_path: Path) -> None:
    fake = write_exe(tmp_path / "py" / "python", FAKE_PYTHON)
    res = subprocess.run(
        [BASH, str(CLUSTER_BENCH_SH), "status", "--exp-id", "x"],
        env=script_env(tmp_path, CLUSTER_BENCH_PYTHON=str(fake)),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert f"PY={fake}" in res.stdout
    assert "ARGS=-m benchmarks.cluster_bench status --exp-id x" in res.stdout
    assert f"PYTHONPATH={REPO_ROOT.resolve()}" in res.stdout and "SAFE=1" in res.stdout

    workdir = tmp_path / "wd"
    node_py = write_exe(workdir / "venv" / "bin" / "python", FAKE_PYTHON)
    res = subprocess.run(
        [
            BASH,
            str(CLUSTER_BENCH_SH),
            "status",
            "--exp-id",
            "x",
            "--workdir",
            str(workdir),
        ],
        env=script_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert f"PY={node_py}" in res.stdout

    env = script_env(tmp_path, CLUSTER_BENCH_WORKDIR=str(workdir))
    for _ in range(2):
        res = subprocess.run(
            [BASH, str(CLUSTER_BENCH_SH), "controller", "status", "--exp-id", "x"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        laptop_py = tmp_path / "home" / ".cluster_bench" / "venv" / "bin" / "python"
        assert f"PY={laptop_py}" in res.stdout
    uv_log = (tmp_path / "uv.log").read_text().splitlines()
    assert any(line.startswith("uv venv --python 3.12 ") for line in uv_log)
    installs = [line for line in uv_log if line.startswith("uv pip install")]
    assert len(installs) == 1 and installs[0].endswith(
        "pydantic>=2 pyyaml huggingface_hub"
    )


@needs_bash
def test_bootstrap_installs_fabric_tools_once_and_never_fails(tmp_path: Path) -> None:
    fakebin = tmp_path / "fakebin"
    log = tmp_path / "apt.log"
    write_exe(
        fakebin / "id", '#!/bin/sh\n[ "$1" = "-u" ] && echo 0 || /usr/bin/id "$@"\n'
    )
    write_exe(fakebin / "timeout", '#!/bin/sh\nshift\nexec "$@"\n')
    write_exe(fakebin / "apt-get", f'#!/bin/sh\necho "$*" >> "{log}"\nexit 100\n')
    env = script_env(tmp_path, CB_FABRIC_TOOLS="1")
    workdir = tmp_path / "wd"
    for _ in range(2):
        res = subprocess.run(
            [BASH, str(BOOTSTRAP_SH), "--workdir", str(workdir), "--minimal"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert res.returncode == 0, res.stderr  # apt failure is not fatal
        assert res.stdout.splitlines()[-1].endswith("/venv/bin/python")
    calls = log.read_text().splitlines()
    assert calls and calls[0].startswith("update"), calls
    assert len(calls) == 1, "attempted once per node (marker file)"
    assert (
        "fabric probing falls back" in res.stderr
        or (workdir / ".fabric_tools_attempted").exists()
    )


@needs_bash
def test_bootstrap_node_is_idempotent(tmp_path: Path) -> None:
    env = script_env(tmp_path)
    workdir = tmp_path / "wd"

    def bootstrap(*extra: str) -> list[str]:
        res = subprocess.run(
            [BASH, str(BOOTSTRAP_SH), "--workdir", str(workdir), *extra],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return res.stdout.splitlines()

    venv_python = str(workdir / "venv" / "bin" / "python")
    assert bootstrap() == [venv_python]
    assert bootstrap() == [venv_python]
    installs = [
        line
        for line in (tmp_path / "uv.log").read_text().splitlines()
        if line.startswith("uv pip install")
    ]
    assert len(installs) == 1
    assert installs[0].endswith(
        "aiperf==0.14.0 huggingface_hub[hf_xet] nvidia-ml-py pyyaml pydantic>=2"
    )
    assert bootstrap("--minimal")[-1] == venv_python
    installs = [
        line
        for line in (tmp_path / "uv.log").read_text().splitlines()
        if line.startswith("uv pip install")
    ]
    assert len(installs) == 2 and "aiperf" not in installs[1]
    res = subprocess.run(
        [BASH, str(BOOTSTRAP_SH), "--workdir", "relative"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 2 and "absolute path" in res.stderr
