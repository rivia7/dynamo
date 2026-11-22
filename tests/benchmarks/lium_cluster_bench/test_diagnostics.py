# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adapters/diagnostics.py (commands, parsers, the fail-soft
runner, dry-run answers) and the CB_NSYS_OUTPUT hook of launch_node.sh."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from benchmarks.lium_cluster_bench.adapters import diagnostics as d
from benchmarks.lium_cluster_bench.adapters.docker import render_env_file
from benchmarks.lium_cluster_bench.adapters.engine_sglang import CONTAINER_EXP
from benchmarks.lium_cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.lium_cluster_bench.domain.schema import ContainerSpec
from benchmarks.lium_cluster_bench.ports import CommandResult
from tests.benchmarks.lium_cluster_bench.lium_fakes import fixture_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "benchmarks/lium_cluster_bench/scripts/launch_node.sh"
FIXTURES = Path(__file__).parent / "fixtures" / "kernels"
IMAGE = (
    "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
    "@sha256:37badffd83eb334aee9d06fb442e36fc3c963314336c92622c78ac5c037d514d"
)
WORKDIR = "/root/cluster_bench"
EXP = "h200-007"
WINDOW = "w1"
NODES = ("node0", "node1")
VERSION_LINE = "NVIDIA Nsight Systems version 2025.6.1.190-256162285624v0"
DOCKER_REF = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r":[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$"
)


def result(node: str, command: str, rc: int = 0, out: str = "", err: str = ""):
    return CommandResult(node, command, rc, out, err)


@dataclass
class Recorder(RecordingExecutor):
    """RecordingExecutor that also keeps timeouts and stdin texts."""

    timeouts: list[tuple[str, str, float | None]] = field(default_factory=list)
    inputs: list[tuple[str, str, str]] = field(default_factory=list)

    def run(self, node, command, *, input_text=None, timeout=None, check=True):
        self.timeouts.append((node, command, timeout))
        if input_text is not None:
            self.inputs.append((node, command, input_text))
        return super().run(
            node, command, input_text=input_text, timeout=timeout, check=check
        )


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def runner(rec: RecordingExecutor, tmp_path: Path) -> d.NsysDiagnostics:
    return d.NsysDiagnostics(
        rec, workdir=WORKDIR, results_dir=tmp_path / "diag", labels={"cb.exp": EXP}
    )


# --------------------------------------------------------------------------
# Flags, paths, worker env
# --------------------------------------------------------------------------


def test_default_flags_carry_the_verified_recipe() -> None:
    for flag in (
        "--trace-fork-before-exec=true",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
    ):
        assert flag in d.NSYS_DEFAULT_FLAGS
    assert all(" " not in f and not f.startswith("-o") for f in d.NSYS_DEFAULT_FLAGS)
    changed = [
        (a, b) for a, b in zip(d.NSYS_DEFAULT_FLAGS, d.NSYS_NO_FORK_FLAGS) if a != b
    ]
    assert changed == [
        ("--trace-fork-before-exec=true", "--trace-fork-before-exec=false")
    ]


def test_paths_and_container_mapping() -> None:
    assert d.CONTAINER_EXP == CONTAINER_EXP
    assert d.window_dir(EXP, WINDOW) == "/bench/experiments/h200-007/diagnostics/w1"
    prefix = d.output_prefix(EXP, WINDOW, "cb-5d1c-decode0-r0")
    assert prefix == "/bench/experiments/h200-007/diagnostics/w1/cb-5d1c-decode0-r0"
    host = d.report_path(WORKDIR + "/", EXP, WINDOW, "cb-5d1c-decode0-r0")
    assert host == (
        "/root/cluster_bench/experiments/h200-007/diagnostics/w1/"
        "cb-5d1c-decode0-r0.nsys-rep"
    )
    assert d.to_container_path(host, WORKDIR) == prefix + ".nsys-rep"
    assert d.to_container_path("/tmp/x.nsys-rep", WORKDIR) is None
    assert d.to_container_path(f"{WORKDIR}/experiments/../x", WORKDIR) is None


@pytest.mark.parametrize("bad", ["", "../up", "a/b", "-x", ".hidden", "w 1", "a..b"])
def test_unsafe_names_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        d.output_prefix(EXP, bad, "cb-x")
    with pytest.raises(ValueError):
        d.report_path(WORKDIR, EXP, WINDOW, bad)


def test_nsys_env_renders_into_an_env_file() -> None:
    env = d.nsys_env(EXP, WINDOW, "cb-5d1c-decode0-r0")
    assert env == {d.ENV_OUTPUT: d.output_prefix(EXP, WINDOW, "cb-5d1c-decode0-r0")}
    full = d.nsys_env(
        EXP,
        WINDOW,
        "cb-5d1c-decode0-r0",
        flags=d.NSYS_NO_FORK_FLAGS,
        nsys_bin="/opt/nvidia/nsight-systems-cli/2025.6.1/bin/nsys",
    )
    assert full[d.ENV_FLAGS].split() == list(d.NSYS_NO_FORK_FLAGS)
    assert full[d.ENV_BIN].endswith("/bin/nsys")
    rendered = render_env_file(full)
    assert f"{d.ENV_FLAGS}={full[d.ENV_FLAGS]}\n" in rendered


@pytest.mark.parametrize(
    "flags",
    [(), ("--trace=cuda", ""), ("--trace=cuda nvtx",), ("-o",), ("--output=/x",)],
)
def test_nsys_env_rejects_bad_flags(flags: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        d.nsys_env(EXP, WINDOW, "cb-x", flags=flags)


def test_nsys_env_rejects_relative_binary() -> None:
    with pytest.raises(ValueError):
        d.nsys_env(EXP, WINDOW, "cb-x", nsys_bin="nsys")


def spec(name: str, node: str, role: str, rank: int = 0, port: int | None = 8083):
    return ContainerSpec(
        node=node,
        name=name,
        role=role,
        rank=rank,
        gpus=(0, 1, 2, 3) if role not in ("control", "frontend") else (),
        image=IMAGE,
        command=("bash",),
        system_port=port if role not in ("control", "frontend") else None,
    )


def test_profile_targets_and_expected_reports() -> None:
    containers = [
        spec("cb-5d1c-control", "node0", "control"),
        spec("cb-5d1c-prefill0-r0", "node0", "prefill"),
        spec("cb-5d1c-decode0-r0", "node0", "decode", port=8084),
        spec("cb-5d1c-decode0-r1", "node1", "decode", rank=1),
        spec("cb-5d1c-frontend", "node0", "frontend"),
    ]
    targets = d.profile_targets(containers)
    assert [c.name for c in targets] == ["cb-5d1c-prefill0-r0", "cb-5d1c-decode0-r0"]
    reports = d.expected_reports(containers, workdir=WORKDIR, exp_id=EXP, window=WINDOW)
    assert sorted(reports) == ["node0", "node1"]
    assert [name for name, _, _ in reports["node0"]] == [
        "cb-5d1c-prefill0-r0",
        "cb-5d1c-decode0-r0",
    ]
    ((name, path, gpus),) = reports["node1"]
    assert name == "cb-5d1c-decode0-r1" and gpus == 4
    assert path == d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r1")


# --------------------------------------------------------------------------
# Probe and diagnostic image
# --------------------------------------------------------------------------


def test_probe_command_round_trip() -> None:
    command = d.probe_command(IMAGE, name="cb-diag-probe-node0", labels={"cb.exp": EXP})
    assert shlex.split(command) == [
        "timeout",
        "-k",
        "10",
        "180",
        "docker",
        "run",
        "--rm",
        "--name",
        "cb-diag-probe-node0",
        "--network",
        "none",
        "--label",
        f"cb.exp={EXP}",
        "--entrypoint",
        "bash",
        IMAGE,
        "-lc",
        d.PROBE_SCRIPT,
    ]


def _write_exe(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def test_probe_script_finds_nsys_and_its_version(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_exe(bindir / "nsys", f"#!/bin/sh\necho '{VERSION_LINE}'\n")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    found = subprocess.run(
        [bash, "-c", d.PROBE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    info = d.parse_probe(found.stdout, image=IMAGE, rc=found.returncode)
    assert info.available and info.path == str(bindir / "nsys")
    assert info.version == "2025.6.1.190-256162285624v0"
    assert info.version_tuple == (2025, 6, 1)
    (bindir / "nsys").unlink()
    missing = subprocess.run(
        [bash, "-c", d.PROBE_SCRIPT],
        env={"PATH": str(bindir), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert missing.stdout.strip() == d.PROBE_MARKER
    absent = d.parse_probe(missing.stdout, image=IMAGE)
    assert not absent.available and absent.notes == ("nsys not in the image",)


def test_parse_probe_failures() -> None:
    docker_error = d.parse_probe(
        "",
        image=IMAGE,
        rc=125,
        stderr="docker: Error response from daemon: No such image: x.\n",
    )
    assert not docker_error.available
    assert docker_error.notes[0].startswith("probe failed: docker: Error response")
    broken = d.parse_probe(
        "CB_NSYS_PATH=/usr/local/bin/nsys\n/usr/local/bin/nsys: error while loading "
        "shared libraries: libQt6Core.so.6\n",
        image=IMAGE,
    )
    assert not broken.available and broken.path == "/usr/local/bin/nsys"
    assert "printed no version" in broken.notes[0]


def test_diag_dockerfile_follows_the_sglang_recipe() -> None:
    text = d.diag_dockerfile(IMAGE)
    lines = text.splitlines()
    assert f"FROM {IMAGE}" in lines
    assert lines.index("USER root") < lines.index("USER dynamo")
    assert lines[-1] == "USER dynamo"
    assert "apt-get install -y --no-install-recommends gnupg" in text
    assert (
        "http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$repoarch /"
    ) in text
    assert "repoarch=arm64; keyarch=arm64" in text
    assert "repoarch=amd64; keyarch=x86_64" in text
    assert (
        "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/"
        "$keyarch/7fa2af80.pub"
    ) in text
    assert "ubuntu1804/x86_64/7fa2af80.pub" in text  # fallback, same key
    assert "apt-get install -y --no-install-recommends nsight-systems-cli" in text
    assert "    nsys --version" in lines
    assert text.endswith("\n")


def test_diag_dockerfile_run_step_is_valid_posix_shell() -> None:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("sh not available")
    text = d.diag_dockerfile(IMAGE)
    run = text.split("RUN ", 1)[1].split("\nUSER ", 1)[0].replace("\\\n", " ")
    check = subprocess.run(
        [sh, "-n", "-c", run], capture_output=True, text=True, timeout=30, check=False
    )
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_image": "bad image"},
        {"base_image": ""},
        {"base_image": IMAGE, "package": "nsight; rm -rf /"},
        {"base_image": IMAGE, "distro": "Ubuntu 24.04"},
        {"base_image": IMAGE, "user": "root\nRUN x"},
    ],
)
def test_diag_dockerfile_rejects_bad_inputs(kwargs: dict) -> None:
    base = kwargs.pop("base_image")
    with pytest.raises(ValueError):
        d.diag_dockerfile(base, **kwargs)


def test_diag_image_tag_is_derived_from_the_base_digest() -> None:
    tag = d.diag_image_tag(IMAGE)
    assert tag.startswith("cb-diag/sglang-runtime-nsys:1.5.1-37badffd83eb-")
    assert DOCKER_REF.match(tag), tag
    assert tag == d.diag_image_tag(IMAGE)
    other_recipe = d.diag_dockerfile(IMAGE, package="nsight-systems-cli-2025.6.1")
    assert d.diag_image_tag(IMAGE, dockerfile=other_recipe) != tag
    plain = "lmsysorg/sglang:v0.5.18-cu130-runtime"
    image_id = "sha256:" + "ab" * 32
    by_id = d.diag_image_tag(plain, base_id=image_id)
    assert by_id.startswith("cb-diag/sglang-nsys:v0.5.18-cu130-runtime-abababababab-")
    hashed = d.diag_image_tag("localhost:5000/My_Image")
    assert hashed.startswith("cb-diag/my-image-nsys:latest-")
    assert DOCKER_REF.match(by_id) and DOCKER_REF.match(hashed)
    assert d.image_digest(IMAGE) == IMAGE.rsplit(":", 1)[1]
    assert d.image_digest(plain) is None
    assert d.parse_image_id(f"{image_id}\n") == image_id
    assert d.parse_image_id("Error: No such image") is None


def test_build_and_inspect_commands_round_trip() -> None:
    tag = d.diag_image_tag(IMAGE)
    argv = shlex.split(d.build_command(tag, base_image=IMAGE))
    assert argv[:4] == ["timeout", "-k", "30", "2400"]
    assert argv[4:] == [
        "docker",
        "build",
        "--network",
        "host",
        "--label",
        "cb.diag=nsys",
        "--label",
        f"cb.diag.base={IMAGE}",
        "-t",
        tag,
        "-",
    ]
    plain = shlex.split(d.build_command(tag, base_image=IMAGE, host_network=False))
    assert "--network" not in plain and plain[-1] == "-"
    assert shlex.split(d.image_id_command(IMAGE)) == [
        "timeout",
        "-k",
        "10",
        "60",
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        IMAGE,
    ]


# --------------------------------------------------------------------------
# Trigger
# --------------------------------------------------------------------------


def test_start_profile_body() -> None:
    assert d.start_profile_body() == {
        "num_steps": 10,
        "activities": ["CUDA_PROFILER"],
        "start_step": 1,
    }
    assert d.start_profile_body(5, None) == {
        "num_steps": 5,
        "activities": ["CUDA_PROFILER"],
    }
    for bad in ((0, 1), (10, 0), (-1, None)):
        with pytest.raises(ValueError):
            d.start_profile_body(*bad)


def test_start_profile_command_round_trip() -> None:
    argv = shlex.split(d.start_profile_command(8083, num_steps=10))
    assert argv[:4] == ["timeout", "-k", "10", "30"]
    curl = argv[4:]
    assert curl[0] == "curl" and curl[-1] == (
        "http://127.0.0.1:8083/engine/control/start_profile"
    )
    body = json.loads(curl[curl.index("--data-binary") + 1])
    assert body == {"activities": ["CUDA_PROFILER"], "num_steps": 10, "start_step": 1}
    assert curl[curl.index("-w") + 1] == "\\n%{http_code}"
    assert ("-X", "POST") in set(zip(curl, curl[1:]))
    with pytest.raises(ValueError):
        d.start_profile_command(70000)


def test_trigger_route_matches_the_dynamo_sources() -> None:
    """The route is Dynamo's generic /engine/{*path} plus SGLang's handler."""
    server = (REPO / "lib/runtime/src/system_status_server.rs").read_text()
    handler = (
        REPO / "components/src/dynamo/sglang/request_handlers/handler_base.py"
    ).read_text()
    assert '"/engine/{*path}"' in server
    assert '"control/start_profile": self.start_profile' in handler
    assert "ProfileReq(**body)" in handler
    assert d.START_PROFILE_ROUTE == "/engine/" + "control/start_profile"


@pytest.mark.parametrize(
    ("stdout", "rc", "stderr", "ok", "status", "message"),
    [
        (
            '{"message":"Profiling started","status":"ok"}\n200',
            0,
            "",
            True,
            200,
            "Profiling started",
        ),
        (
            '{"error":"Handler error","message":"Profiling is already in progress. '
            'Call /stop_profile first."}\n500',
            0,
            "",
            False,
            500,
            "Profiling is already in progress. Call /stop_profile first.",
        ),
        (
            '{"error":"Route not found","message":"Route /engine/control/'
            'start_profile not found"}\n404',
            0,
            "",
            False,
            404,
            "Route /engine/control/start_profile not found",
        ),
        (
            "\n000",
            7,
            "curl: (7) Failed to connect to 127.0.0.1 port 8083 after 0 ms: "
            "Couldn't connect to server\n",
            False,
            None,
            "curl: (7) Failed to connect to 127.0.0.1 port 8083 after 0 ms: "
            "Couldn't connect to server",
        ),
        (
            "<html>bad gateway</html>\n502",
            0,
            "",
            False,
            502,
            "<html>bad gateway</html>",
        ),
        ("\n200", 0, "", True, 200, None),
    ],
)
def test_parse_trigger(stdout, rc, stderr, ok, status, message) -> None:
    out = d.parse_trigger(stdout, node="node1", port=8083, rc=rc, stderr=stderr)
    assert (out.ok, out.http_status, out.message) == (ok, status, message)
    assert out.node == "node1" and out.port == 8083


def test_runner_trigger_is_fail_soft(tmp_path: Path) -> None:
    def responder(node: str, command: str) -> CommandResult:
        if ":8083/" in command:
            return result(
                node, command, out='{"message":"Profiling started","status":"ok"}\n200'
            )
        return result(node, command, 7, "\n000", "curl: (7) Couldn't connect\n")

    rec = Recorder(node_names=NODES, responder=responder)
    diag = runner(rec, tmp_path)
    ok = diag.trigger("node1", 8083, num_steps=12, start_step=None)
    assert ok.ok and ok.message == "Profiling started"
    bad = diag.trigger("node1", 8090)
    assert not bad.ok and "Couldn't connect" in bad.message
    invalid = diag.trigger("node1", 0)
    assert not invalid.ok and "invalid system port" in invalid.message
    sent = [c for n, c, _ in rec.timeouts if "curl" in c]
    assert len(sent) == 2
    body = json.loads(
        shlex.split(sent[0])[shlex.split(sent[0]).index("--data-binary") + 1]
    )
    assert body == {"activities": ["CUDA_PROFILER"], "num_steps": 12}
    assert {t for _, _, t in rec.timeouts} == {d.TRIGGER_TIMEOUT_S + d.OUTER_MARGIN_S}
    saved = (tmp_path / "diag" / "trigger-node1-8083.txt").read_text()
    assert "Profiling started" in saved


# --------------------------------------------------------------------------
# Reports and nsys log
# --------------------------------------------------------------------------


def test_stat_and_find_commands() -> None:
    paths = ["/r/a.nsys-rep", "/r/b c.nsys-rep"]
    command = d.stat_command(paths)
    assert command.endswith(" 2>/dev/null || true")
    head = command[: -len(" 2>/dev/null || true")]
    assert shlex.split(head)[4:] == ["stat", "--printf", "%s %Y %n\\n", "--", *paths]
    parsed = d.parse_stat(
        "187654321 1793961100 /r/a.nsys-rep\n0 1793961101 /r/b c.nsys-rep\nstat: x\n"
    )
    assert parsed == {
        "/r/a.nsys-rep": (187654321, 1793961100),
        "/r/b c.nsys-rep": (0, 1793961101),
    }
    find = shlex.split(d.find_reports_command("/r/w1")[: -len(" 2>/dev/null || true")])
    assert find[4:] == [
        "find",
        "/r/w1",
        "-maxdepth",
        "2",
        "-type",
        "f",
        "-name",
        "*.nsys-rep",
        "-printf",
        "%s %T@ %p\\n",
    ]
    assert d.parse_find("12 1793961100.5 /r/w1/a.nsys-rep\nbad line\n") == {
        "/r/w1/a.nsys-rep": (12, 1793961100.5)
    }


def test_wait_reports_waits_for_stable_non_empty_files(tmp_path: Path) -> None:
    a = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r0")
    b = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode1-r0")
    c = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r1")
    growth = {a: [100, 200, 200, 200, 200], c: [0, 0, 0, 0, 0]}
    seen = {a: 0, c: 0}

    def responder(node: str, command: str) -> CommandResult | None:
        if " stat --printf " not in command:
            return None
        lines = []
        for path in shlex.split(command):
            if path in growth:
                size = growth[path][min(seen[path], 4)]
                seen[path] += 1
                lines.append(f"{size} 1793961100 {path}")
        return result(node, command, out="\n".join(lines) + "\n")

    rec = Recorder(node_names=NODES, responder=responder)
    clock = FakeClock()
    found = runner(rec, tmp_path).wait_reports(
        {"node0": [a, b], "node1": [c]}, clock=clock, timeout_s=60, poll_s=15
    )
    (ra, rb) = found["node0"]
    (rc_,) = found["node1"]
    assert (ra.path, ra.ready, ra.size_bytes) == (a, True, 200)
    assert (rb.ready, rb.size_bytes) == (False, None)
    assert (rc_.ready, rc_.size_bytes) == (False, 0)
    assert seen[a] == 3  # no longer polled once ready
    assert clock.sleeps == [15, 15, 15, 15]


def test_wait_reports_returns_as_soon_as_everything_is_ready(tmp_path: Path) -> None:
    path = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-agg0-r0")

    def responder(node: str, command: str) -> CommandResult | None:
        if " stat --printf " in command:
            return result(node, command, out=f"4096 1793961100 {path}\n")
        return None

    rec = Recorder(node_names=NODES, responder=responder)
    clock = FakeClock()
    found = runner(rec, tmp_path).wait_reports({"node1": [path]}, clock=clock)
    assert found["node1"][0].ready and clock.sleeps == [d.REPORT_POLL_S]
    assert {t for _, _, t in rec.timeouts} == {d.STAT_TIMEOUT_S + d.OUTER_MARGIN_S}
    nothing = runner(rec, tmp_path).wait_reports({}, clock=clock)
    assert nothing == {}


def test_parse_nsys_log_of_a_worker() -> None:
    state = d.parse_nsys_log(fixture_text(FIXTURES / "worker_nsys.log.txt"))
    assert state.capture_started and state.capture_ended
    assert state.generated == (
        "/bench/experiments/h200-007/diagnostics/w1/cb-5d1c-decode0-r0.nsys-rep",
    )
    assert state.errors == ()


def test_parse_nsys_log_variants() -> None:
    old = (
        'Saved report file to "/tmp/nsys-report-ab12.qdrep"\n'
        'Report file moved to "/root/report1.qdrep"\n'
        "Generated: /bench/experiments/e/diagnostics/w/x.nsys-rep\n"
    )
    assert d.parse_nsys_log(old).generated == (
        "/tmp/nsys-report-ab12.qdrep",
        "/root/report1.qdrep",
        "/bench/experiments/e/diagnostics/w/x.nsys-rep",
    )
    failed = d.parse_nsys_log(
        "Error: Failed to create /bench/x.nsys-rep: Permission denied\n"
        "[2026-11-06 10:51:13 TP0] Scheduler hit an exception: ValueError\n"
        "nsys: CUDA injection failed to initialize\n"
    )
    assert not failed.capture_started and failed.generated == ()
    assert len(failed.errors) == 2 and failed.errors[0].startswith("Error: Failed")
    assert d.parse_nsys_log("").generated == ()


# --------------------------------------------------------------------------
# Image lifecycle through the runner
# --------------------------------------------------------------------------


def image_responder(
    tag: str,
    *,
    base_has_nsys: bool = False,
    tag_present: bool = False,
    build: Callable[[str, str], CommandResult | None] | None = None,
):
    def responder(node: str, command: str) -> CommandResult | None:
        if d.PROBE_MARKER in command:
            if f" {tag} " in command or base_has_nsys:
                out = f"{d.PROBE_MARKER}/usr/local/bin/nsys\n{VERSION_LINE}\n"
                return result(node, command, out=out)
            return result(node, command, out=f"{d.PROBE_MARKER}\n")
        if " docker image inspect " in command and command.endswith(f" {tag}"):
            if tag_present:
                return result(node, command, out="sha256:" + "ef" * 32 + "\n")
            return result(node, command, 1, err=f"Error: No such image: {tag}\n")
        if " docker build " in command and build is not None:
            return build(node, command)
        return None

    return responder


def test_ensure_image_builds_on_every_node(tmp_path: Path) -> None:
    tag = d.diag_image_tag(IMAGE)
    rec = Recorder(node_names=NODES, responder=image_responder(tag))
    images = runner(rec, tmp_path).ensure_image(NODES, IMAGE)
    assert sorted(images) == list(NODES)
    for node in NODES:
        image = images[node]
        assert image.ok and image.built and image.image == tag
        assert image.nsys.version == "2025.6.1.190-256162285624v0"
        assert image.nsys.version_tuple == (2025, 6, 1)
        builds = [
            (c, text)
            for n, c, text in rec.inputs
            if n == node and " docker build " in c
        ]
        assert len(builds) == 1
        command, dockerfile = builds[0]
        assert "--network host" in command and command.endswith(" -")
        assert dockerfile == d.diag_dockerfile(IMAGE)
        log = (tmp_path / "diag" / f"build-{node}.log").read_text()
        assert "docker build" in log
    timeouts = {t for _, c, t in rec.timeouts if " docker build " in c}
    assert timeouts == {d.BUILD_TIMEOUT_S + d.OUTER_MARGIN_S}
    probe = [c for _, c, _ in rec.timeouts if d.PROBE_MARKER in c]
    assert len(probe) == 4  # base and derived image on each node
    assert all("--label cb.exp=h200-007" in c for c in probe)


def test_ensure_image_uses_an_image_that_already_ships_nsys(tmp_path: Path) -> None:
    tag = d.diag_image_tag(IMAGE)
    rec = Recorder(node_names=NODES, responder=image_responder(tag, base_has_nsys=True))
    image = runner(rec, tmp_path).ensure_image(["node1"], IMAGE)["node1"]
    assert image.ok and not image.built and image.image == IMAGE
    assert image.notes == ("the serving image already ships nsys",)
    assert not any(" docker build " in c.command for c in rec.commands)


def test_ensure_image_reuses_a_present_tag(tmp_path: Path) -> None:
    tag = d.diag_image_tag(IMAGE)
    rec = Recorder(node_names=NODES, responder=image_responder(tag, tag_present=True))
    image = runner(rec, tmp_path).ensure_image(["node0"], IMAGE)["node0"]
    assert image.ok and not image.built and image.image == tag
    assert not any(" docker build " in c.command for c in rec.commands)
    rebuilt = runner(rec, tmp_path).ensure_image(["node0"], IMAGE, rebuild=True)
    assert rebuilt["node0"].built


def test_ensure_image_retries_without_host_network(tmp_path: Path) -> None:
    tag = d.diag_image_tag(IMAGE)

    def build(node: str, command: str) -> CommandResult | None:
        if "--network host" in command:
            return result(
                node,
                command,
                1,
                err="ERROR: failed to solve: granting entitlement network.host is "
                "not allowed\n",
            )
        return None

    rec = Recorder(node_names=NODES, responder=image_responder(tag, build=build))
    image = runner(rec, tmp_path).ensure_image(["node1"], IMAGE)["node1"]
    assert image.ok and image.built
    builds = [c.command for c in rec.commands if " docker build " in c.command]
    assert len(builds) == 2 and "--network" not in builds[1]


def test_ensure_image_build_failure_is_a_note(tmp_path: Path) -> None:
    tag = d.diag_image_tag(IMAGE)

    def build(node: str, command: str) -> CommandResult:
        return result(
            node,
            command,
            100,
            out="#7 3.112 E: Unable to locate package nsight-systems-cli\n",
        )

    rec = Recorder(node_names=NODES, responder=image_responder(tag, build=build))
    image = runner(rec, tmp_path).ensure_image(["node1"], IMAGE)["node1"]
    assert not image.ok and image.image is None and not image.built
    assert any("Unable to locate package" in n for n in image.notes)
    assert len([c for c in rec.commands if " docker build " in c.command]) == 1
    assert (
        "Unable to locate package"
        in (tmp_path / "diag" / "build-node1.log").read_text()
    )


def test_ensure_image_without_digest_uses_the_local_image_id(tmp_path: Path) -> None:
    plain = "lmsysorg/sglang:v0.5.18-cu130-runtime"
    image_id = "sha256:" + "cd" * 32
    tag = d.diag_image_tag(plain, base_id=image_id)

    base = image_responder(tag)

    def responder(node: str, command: str) -> CommandResult | None:
        if "--format '{{.Id}}'" in command and command.endswith(f" {plain}"):
            return result(node, command, out=image_id + "\n")
        return base(node, command)

    rec = Recorder(node_names=NODES, responder=responder)
    image = runner(rec, tmp_path).ensure_image(["node0"], plain)["node0"]
    assert image.ok and image.image == tag and "-cdcdcdcdcdcd-" in tag


def test_ensure_image_with_an_invalid_base_never_runs_anything(tmp_path: Path) -> None:
    rec = Recorder(node_names=NODES)
    images = runner(rec, tmp_path).ensure_image(NODES, "bad image")
    assert all(not i.ok and "invalid base image" in i.notes[0] for i in images.values())
    assert rec.commands == []


def test_probe_cleans_up_after_a_failure(tmp_path: Path) -> None:
    def responder(node: str, command: str) -> CommandResult | None:
        if d.PROBE_MARKER in command:
            return result(node, command, 124, err="\n[timeout]")
        return None

    rec = Recorder(node_names=NODES, responder=responder)
    info = runner(rec, tmp_path).probe("node0", IMAGE)
    assert not info.available and "[timeout]" in info.notes[0]
    cleanups = [c for c in rec.commands_for("node0") if "docker rm -f" in c]
    assert len(cleanups) == 2 and "cb-diag-probe-node0" in cleanups[0]


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def test_stats_argv_and_container() -> None:
    report = "/bench/experiments/e/diagnostics/w1/cb-x-agg0-r0.nsys-rep"
    base = report[: -len(".nsys-rep")]
    assert d.stats_argv(report, output_base=base) == [
        "nsys",
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=csv",
        f"--output={base}",
        "--report=cuda_gpu_kern_sum",
        "--report=cuda_gpu_mem_time_sum",
        "--report=cuda_kern_exec_sum",
        report,
    ]
    assert d.csv_path(base, "cuda_gpu_kern_sum") == f"{base}_cuda_gpu_kern_sum.csv"
    argv = d.stats_container_argv(
        name="cb-diag-stats",
        image="cb-diag/sglang-runtime-nsys:1.5.1-37badffd83eb-12345678",
        workdir=WORKDIR,
        report=report,
        output_base=base,
        labels={"cb.exp": "e"},
    )
    assert argv[:3] == ["docker", "run", "--rm"] and "-d" not in argv
    pairs = set(zip(argv, argv[1:]))
    assert ("-v", f"{WORKDIR}/experiments:/bench/experiments") in pairs
    assert ("--label", "cb.exp=e") in pairs
    assert "--gpus" not in argv and "/dev/infiniband" not in argv
    image_at = argv.index("cb-diag/sglang-runtime-nsys:1.5.1-37badffd83eb-12345678")
    assert argv[image_at + 1 :] == d.stats_argv(report, output_base=base)
    command = d.stats_command(argv)
    assert shlex.split(command)[:4] == ["timeout", "-k", "30", "1800"]
    assert shlex.split(command)[4:] == argv
    assert shlex.split(d.cat_command(f"{base}_x.csv")) == [
        "timeout",
        "-k",
        "10",
        "120",
        "cat",
        "--",
        f"{base}_x.csv",
    ]


def stats_fixture_texts(base: str) -> dict[str, str]:
    return {
        d.csv_path(base, "cuda_gpu_kern_sum"): fixture_text(
            FIXTURES / "kern_sum_decode_b200.csv"
        ),
        d.csv_path(base, "cuda_gpu_mem_time_sum"): fixture_text(
            FIXTURES / "mem_time_sum_decode_b200.csv"
        ),
        d.csv_path(base, "cuda_kern_exec_sum"): fixture_text(
            FIXTURES / "kern_exec_sum_decode_b200.csv"
        ),
    }


def stats_responder(files: dict[str, str], stats_rc: int = 0, stats_out: str = ""):
    def responder(node: str, command: str) -> CommandResult | None:
        if " cat -- " in command:
            path = shlex.split(command)[-1]
            if path in files:
                return result(node, command, out=files[path])
            return result(node, command, 1, err=f"cat: {path}: No such file\n")
        if " nsys stats " in command:
            return result(node, command, stats_rc, out=stats_out)
        return None

    return responder


def test_stats_parses_the_three_reports(tmp_path: Path) -> None:
    report = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r0")
    base = report[: -len(".nsys-rep")]
    files = stats_fixture_texts(base)
    console = (
        "Generating SQLite file "
        "/bench/experiments/h200-007/diagnostics/w1/cb-5d1c-decode0-r0.sqlite\n"
    )
    rec = Recorder(node_names=NODES, responder=stats_responder(files, 0, console))
    tag = d.diag_image_tag(IMAGE)
    rep = runner(rec, tmp_path).stats("node1", report, image=tag, expected_devices=8)
    assert rep.node == "node1" and rep.source == "cb-5d1c-decode0-r0"
    assert rep.report == report and rep.files == tuple(files)
    assert rep.notes == ()
    bd = rep.breakdown
    assert bd is not None and bd.sources == ("cb-5d1c-decode0-r0",)
    assert bd.categories[0].category == "moe_gemm"
    assert bd.coverage[0].complete is True and bd.coverage[0].processes == 8
    commands = rec.commands_for("node1")
    run = next(c for c in commands if " nsys stats " in c)
    argv = shlex.split(run)
    assert argv[argv.index(tag) + 1 :] == d.stats_argv(
        "/bench/experiments/h200-007/diagnostics/w1/cb-5d1c-decode0-r0.nsys-rep",
        output_base="/bench/experiments/h200-007/diagnostics/w1/cb-5d1c-decode0-r0",
    )
    assert any(c.endswith(" || true") and f"{base}.sqlite" in c for c in commands)
    timeouts = {t for _, c, t in rec.timeouts if " nsys stats " in c}
    assert timeouts == {d.STATS_TIMEOUT_S + d.OUTER_MARGIN_S}
    saved = tmp_path / "diag" / "stats"
    assert (saved / "cb-5d1c-decode0-r0_cuda_kern_exec_sum.csv").exists()
    assert "Generating SQLite" in (saved / "cb-5d1c-decode0-r0.log").read_text()


def test_stats_reports_partial_gpu_coverage(tmp_path: Path) -> None:
    """Only the first GPU's scheduler calls cudaProfilerStart; if nsys traced
    that process alone, the report says so."""
    report = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r0")
    base = report[: -len(".nsys-rep")]
    files = stats_fixture_texts(base)
    exec_path = d.csv_path(base, "cuda_kern_exec_sum")
    lines = files[exec_path].splitlines(keepends=True)
    files[exec_path] = "".join(
        [lines[0], *(line for line in lines[1:] if line.split(",")[2] == "0")]
    )
    rec = Recorder(node_names=NODES, responder=stats_responder(files))
    rep = runner(rec, tmp_path).stats("node0", report, image="img", expected_devices=8)
    assert rep.breakdown is not None and rep.breakdown.partial_coverage
    assert rep.breakdown.coverage[0].devices == (0,)
    assert "cb-5d1c-decode0-r0: kernels on 1 of 8 GPUs inside the capture range" in (
        rep.notes
    )
    assert rep.notes == rep.breakdown.notes


def test_stats_failure_and_missing_csvs_become_notes(tmp_path: Path) -> None:
    report = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r1")
    rec = Recorder(
        node_names=NODES,
        responder=stats_responder(
            {}, 1, "ERROR: Report file is corrupted or incomplete.\n"
        ),
    )
    rep = runner(rec, tmp_path).stats(
        "node0", report, image="img", source="r1", keep_sqlite=True
    )
    assert rep.breakdown is None and rep.files == ()
    assert "r1: nsys stats failed (exit 1)" in rep.notes
    assert "r1: ERROR: Report file is corrupted or incomplete." in rep.notes
    assert "r1: cuda_gpu_kern_sum CSV missing" in rep.notes
    assert "r1: GPU coverage unknown" in rep.notes
    commands = rec.commands_for("node0")
    assert len([c for c in commands if "docker rm -f" in c]) == 2
    assert not any(".sqlite" in c for c in commands)


def test_stats_without_kernels_in_the_window(tmp_path: Path) -> None:
    report = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-agg0-r0")
    base = report[: -len(".nsys-rep")]
    header = (
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),"
        "StdDev (ns),Name\n"
    )
    files = {d.csv_path(base, "cuda_gpu_kern_sum"): header}
    rec = Recorder(node_names=NODES, responder=stats_responder(files))
    rep = runner(rec, tmp_path).stats("node0", report, image="img")
    assert rep.breakdown is None
    assert "cb-5d1c-agg0-r0: no kernels in the capture range" in rep.notes


def test_stats_refuses_reports_outside_the_experiment_mount(tmp_path: Path) -> None:
    rec = Recorder(node_names=NODES)
    rep = runner(rec, tmp_path).stats("node0", "/tmp/report1.nsys-rep", image="img")
    assert rep.breakdown is None and rep.source == "report1"
    assert rep.notes == ("report is outside /root/cluster_bench/experiments",)
    assert rec.commands == []


def test_unwritable_results_dir_never_fails(tmp_path: Path) -> None:
    blocker = tmp_path / "diag"
    blocker.write_text("not a directory")
    tag = d.diag_image_tag(IMAGE)
    rec = Recorder(node_names=NODES, responder=image_responder(tag))
    images = runner(rec, tmp_path).ensure_image(["node0"], IMAGE)
    assert images["node0"].ok


# --------------------------------------------------------------------------
# Dry-run answers
# --------------------------------------------------------------------------


def test_simulated_stdout_answers_the_window_commands() -> None:
    tag = d.diag_image_tag(IMAGE)
    base_probe = d.simulated_stdout(d.probe_command(IMAGE, name="p"), "node0")
    assert d.parse_probe(base_probe, image=IMAGE).available is False
    tag_probe = d.simulated_stdout(d.probe_command(tag, name="p"), "node0")
    assert d.parse_probe(tag_probe, image=tag).available is True
    trigger = d.simulated_stdout(d.start_profile_command(8083), "node0")
    assert d.parse_trigger(trigger, node="node0", port=8083).ok
    path = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-prefill0-r0")
    stat = d.simulated_stdout(d.stat_command([path]), "node0", now=1793961100.0)
    assert d.parse_stat(stat) == {path: (d.SIM_REPORT_BYTES, 1793961100)}
    base = path[: -len(".nsys-rep")]
    kern = d.simulated_stdout(
        d.cat_command(d.csv_path(base, "cuda_gpu_kern_sum")), "node0"
    )
    assert "fmhaSm100fKernel_QkvBfloat16OBfloat16H192" in kern  # prefill table
    assert d.simulated_stdout(d.build_command(tag, base_image=IMAGE), "node0") is None
    assert d.simulated_stdout("docker ps", "node0") is None


def test_dry_run_window_end_to_end(tmp_path: Path) -> None:
    def responder(node: str, command: str) -> CommandResult | None:
        out = d.simulated_stdout(command, node, devices=4)
        return None if out is None else result(node, command, out=out)

    rec = Recorder(node_names=NODES, responder=responder)
    diag = runner(rec, tmp_path)
    images = diag.ensure_image(NODES, IMAGE)
    assert all(i.ok and i.built for i in images.values())
    assert diag.trigger("node0", 8083).ok
    path = d.report_path(WORKDIR, EXP, WINDOW, "cb-5d1c-decode0-r0")
    ready = diag.wait_reports({"node0": [path]}, clock=FakeClock())
    assert ready["node0"][0].ready
    rep = diag.stats("node0", path, image=images["node0"].image, expected_devices=4)
    assert rep.breakdown is not None and rep.breakdown.coverage[0].complete
    assert rep.breakdown.share("moe_gemm") > 0.2


# --------------------------------------------------------------------------
# launch_node.sh worker hook
# --------------------------------------------------------------------------

STUB_GPU_UTILS = 'build_sglang_gpu_mem_args() { echo ""; }\n'
STUB_LAUNCH_UTILS = """dyn_port() { local value="${!1:-}"; echo "${value:-$3}"; }
print_launch_banner() { :; }
dynamo_exit_trap() { :; }
wait_any_exit() { wait; }
"""
NSYS_STUB = """#!/bin/sh
printf '%s\\n' "$@" > "$CB_TEST_LOG/nsys.args"
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then shift 2; break; fi
    shift
done
exec "$@"
"""
PYTHON_STUB = """#!/bin/sh
printf '%s\\n' "$@" > "$CB_TEST_LOG/python3.args"
"""
WORKER_ARGS = ["--config", "/bench/experiments/x/cb-1.yaml", "--trust-remote-code"]
PYTHON_ARGV = ["-m", "dynamo.sglang", *WORKER_ARGS]


@pytest.fixture(scope="module")
def bash() -> str:
    path = shutil.which("bash")
    if path is None:
        pytest.skip("bash not available")
    return path


def _bash_version(bash: str) -> tuple[int, int]:
    out = subprocess.run(
        [bash, "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.split()
    return int(out[0]), int(out[1])


@dataclass
class Launch:
    proc: subprocess.CompletedProcess
    log: Path

    def args(self, tool: str) -> list[str] | None:
        path = self.log / f"{tool}.args"
        return path.read_text().splitlines() if path.exists() else None


def launch_worker(
    bash: str,
    tmp_path: Path,
    env_extra: dict[str, str],
    *,
    with_nsys: bool = True,
    real_helpers: bool = False,
    system_path: bool = True,
) -> Launch:
    log = tmp_path / "log"
    bindir = tmp_path / "bin"
    log.mkdir()
    bindir.mkdir()
    _write_exe(bindir / "python3", PYTHON_STUB)
    if with_nsys:
        _write_exe(bindir / "nsys", NSYS_STUB)
    if real_helpers:
        home = REPO
    else:
        home = tmp_path / "home"
        common = home / "examples" / "common"
        common.mkdir(parents=True)
        (common / "gpu_utils.sh").write_text(STUB_GPU_UTILS)
        (common / "launch_utils.sh").write_text(STUB_LAUNCH_UTILS)
    path = f"{bindir}:/usr/bin:/bin" if system_path else str(bindir)
    env = {
        "PATH": path,
        "HOME": str(tmp_path),
        "DYNAMO_HOME": str(home),
        "CB_TEST_LOG": str(log),
        **env_extra,
    }
    proc = subprocess.run(
        [bash, str(SCRIPT), "worker", *WORKER_ARGS],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        start_new_session=True,  # the real helpers signal their process group
        check=False,
    )
    return Launch(proc, log)


def test_launch_script_parses(bash: str) -> None:
    check = subprocess.run(
        [bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30
    )
    assert check.returncode == 0, check.stderr


def test_worker_without_nsys_output_is_unchanged(bash: str, tmp_path: Path) -> None:
    run = launch_worker(bash, tmp_path, {})
    assert run.proc.returncode == 0, run.proc.stderr
    assert run.args("python3") == PYTHON_ARGV
    assert run.args("nsys") is None
    assert "WARNING" not in run.proc.stderr


def test_worker_runs_under_nsys_with_the_default_flags(
    bash: str, tmp_path: Path
) -> None:
    prefix = tmp_path / "exp dir" / "diagnostics" / "w1" / "cb-5d1c-decode0-r0"
    run = launch_worker(bash, tmp_path, {d.ENV_OUTPUT: str(prefix)})
    assert run.proc.returncode == 0, run.proc.stderr
    assert run.args("nsys") == [
        "profile",
        *d.NSYS_DEFAULT_FLAGS,
        "-o",
        str(prefix),
        "python3",
        *PYTHON_ARGV,
    ]
    assert run.args("python3") == PYTHON_ARGV
    assert prefix.parent.is_dir()
    assert "worker runs under Nsight Systems" in run.proc.stdout
    assert "WARNING" not in run.proc.stderr


def test_worker_flag_override_is_split_without_globbing(
    bash: str, tmp_path: Path
) -> None:
    (tmp_path / "--foo=bar").write_text("")  # what "--foo=*" would glob to
    prefix = tmp_path / "diag" / "w"
    flags = "--trace=cuda   --capture-range=cudaProfilerApi --foo=*"
    run = launch_worker(bash, tmp_path, {d.ENV_OUTPUT: str(prefix), d.ENV_FLAGS: flags})
    assert run.proc.returncode == 0, run.proc.stderr
    assert run.args("nsys") == [
        "profile",
        "--trace=cuda",
        "--capture-range=cudaProfilerApi",
        "--foo=*",
        "-o",
        str(prefix),
        "python3",
        *PYTHON_ARGV,
    ]


def test_worker_blank_flag_override_keeps_the_defaults(
    bash: str, tmp_path: Path
) -> None:
    prefix = tmp_path / "diag" / "w"
    run = launch_worker(bash, tmp_path, {d.ENV_OUTPUT: str(prefix), d.ENV_FLAGS: "  "})
    assert run.args("nsys")[1 : 1 + len(d.NSYS_DEFAULT_FLAGS)] == list(
        d.NSYS_DEFAULT_FLAGS
    )


def test_worker_without_nsys_warns_once_and_starts(bash: str, tmp_path: Path) -> None:
    run = launch_worker(
        bash,
        tmp_path,
        {d.ENV_OUTPUT: str(tmp_path / "diag" / "w")},
        with_nsys=False,
        system_path=False,  # nothing but the python3 stub on PATH
    )
    assert run.proc.returncode == 0, run.proc.stderr
    warnings = [line for line in run.proc.stderr.splitlines() if "WARNING" in line]
    assert len(warnings) == 1 and "nsys is not available" in warnings[0]
    assert run.args("python3") == PYTHON_ARGV and run.args("nsys") is None
    assert not (tmp_path / "diag").exists()


def test_worker_nsys_bin_override(bash: str, tmp_path: Path) -> None:
    other = tmp_path / "opt" / "nsys-2025"
    other.mkdir(parents=True)
    nsys = _write_exe(other / "nsys", NSYS_STUB)
    prefix = tmp_path / "diag" / "w"
    run = launch_worker(
        bash,
        tmp_path,
        {d.ENV_OUTPUT: str(prefix), d.ENV_BIN: str(nsys)},
        with_nsys=False,
    )
    assert run.proc.returncode == 0, run.proc.stderr
    assert run.args("nsys")[0] == "profile"
    assert f"{nsys} profile" in run.proc.stdout


def test_worker_with_an_unusable_nsys_bin_starts_unprofiled(
    bash: str, tmp_path: Path
) -> None:
    run = launch_worker(
        bash,
        tmp_path,
        {
            d.ENV_OUTPUT: str(tmp_path / "diag" / "w"),
            d.ENV_BIN: str(tmp_path / "missing" / "nsys"),
        },
    )
    assert run.proc.returncode == 0, run.proc.stderr
    assert "nsys is not available" in run.proc.stderr
    assert run.args("nsys") is None and run.args("python3") == PYTHON_ARGV


def test_worker_with_an_uncreatable_output_dir_starts_unprofiled(
    bash: str, tmp_path: Path
) -> None:
    (tmp_path / "file").write_text("")
    run = launch_worker(bash, tmp_path, {d.ENV_OUTPUT: str(tmp_path / "file" / "w")})
    assert run.proc.returncode == 0, run.proc.stderr
    assert "cannot create the directory of CB_NSYS_OUTPUT" in run.proc.stderr
    assert run.args("nsys") is None and run.args("python3") == PYTHON_ARGV


def test_worker_hook_with_the_real_launch_utils(bash: str, tmp_path: Path) -> None:
    if _bash_version(bash) < (4, 3):
        pytest.skip("examples/common/launch_utils.sh needs bash >= 4.3 (wait -n)")
    prefix = tmp_path / "diag" / "cb-5d1c-agg0-r0"
    run = launch_worker(
        bash,
        tmp_path,
        {d.ENV_OUTPUT: str(prefix)},
        real_helpers=True,
    )
    assert run.proc.returncode == 0, run.proc.stdout + run.proc.stderr
    assert run.args("nsys") == [
        "profile",
        *d.NSYS_DEFAULT_FLAGS,
        "-o",
        str(prefix),
        "python3",
        *PYTHON_ARGV,
    ]
    assert "cluster_bench SGLang worker" in run.proc.stdout


def test_launch_hook_defaults_match_the_adapter() -> None:
    """The shell default list is the adapter's NSYS_DEFAULT_FLAGS, in order."""
    script = SCRIPT.read_text()
    block = script.split("nsys_flags=(\n", 1)[1].split(")", 1)[0]
    assert [line.strip() for line in block.splitlines() if line.strip()] == list(
        d.NSYS_DEFAULT_FLAGS
    )
    assert os.access(SCRIPT, os.R_OK)
