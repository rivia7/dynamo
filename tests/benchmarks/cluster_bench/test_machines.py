# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSH machines: the machines file, discovery, control addresses, the trust
key scripts and the controller's up -> launch -> down flow on them.

``FakeMachines`` answers the controller's ssh calls per machine and hands the
runner's own scripts to ``FakeNode``. Nothing touches the network or a real
machine; the shell scripts run in a local bash against a temporary HOME with
fake docker, ssh, ip and nvidia-smi first on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from benchmarks.cluster_bench import controller as ctl_mod
from benchmarks.cluster_bench.adapters import machines as mach
from benchmarks.cluster_bench.adapters import providers as prov
from benchmarks.cluster_bench.adapters.providers import LiumCli
from benchmarks.cluster_bench.controller import (
    Controller,
    ControllerError,
    ControllerState,
    SshEndpoint,
    UpOptions,
)
from benchmarks.cluster_bench.ports import CommandResult, NodeAddress
from tests.benchmarks.cluster_bench.lium_fakes import HF_TOKEN
from tests.benchmarks.cluster_bench.test_controller import Call, FakeClock, FakeNode

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

BASH = shutil.which("bash")
SSH_KEYGEN = shutil.which("ssh-keygen")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not installed")
EXP = "exp42"
WORKDIR = "/root/cluster_bench"
KEY = f"{WORKDIR}/ssh/{EXP}_ed25519"
KH = mach.known_hosts_path(WORKDIR, EXP)
PUBKEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOnlyAFakeKeyBodyForTheseTests0000000000 "
    "cluster_bench:exp42"
)
NODE0 = ("root@203.0.113.10", 2222)
NODE1 = ("root@203.0.113.11", 2223)
NODE2 = ("root@203.0.113.12", 2200)
MACHINES = """\
# node0 runs the harness
ssh -p 2222 -i /keys/lab root@203.0.113.10
root@203.0.113.11:2223 name=gpu-b   # node0 reaches it on its private address
ssh://root@203.0.113.12:2200
"""


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


def discovery(
    host: str,
    *addrs: str,
    gpus: int = 8,
    gpu: str = "NVIDIA H200",
    uid: int = 0,
    docker: str = "ok",
    workdir: str = "ok",
) -> str:
    """What ``machines.discovery_script`` prints on a machine."""
    lines = [f"@@HOST {host}", f"@@UID {uid}", "@@ADDR lo 127.0.0.1/8"]
    lines += ["@@ADDR docker0 172.17.0.1/16", *(f"@@ADDR {a}" for a in addrs)]
    lines += [f"@@GPU {gpu}"] * gpus
    lines += [f"@@DOCKER {docker}", f"@@WORKDIR {workdir}"]
    return "\n".join(lines) + "\n"


@dataclass
class Machine:
    hostname: str
    facts: str  # discovery output
    routes: set[tuple[str, int]] = field(default_factory=set)  # node0 -> it
    authorized: list[str] = field(default_factory=list)
    cleanups: list[str] = field(default_factory=list)


def destination(argv: list[str]) -> tuple[str, int]:
    return argv[argv.index("bash") - 1], int(argv[argv.index("-p") + 1])


@dataclass
class FakeMachines(FakeNode):
    """The laptop's ssh to each machine; the runner's scripts go to FakeNode."""

    machines: dict[tuple[str, int], Machine] = field(default_factory=dict)
    log: list[tuple[tuple[str, int], str]] = field(default_factory=list)
    upload_noise: bytes = b""  # non-empty: node0 stores something else

    def _ssh(self, call: Call) -> CommandResult:
        if not self.ssh_ok:
            return super()._ssh(call)
        where = destination(call.argv)
        machine = self.machines[where]
        script = call.script
        if "@@HOST" in script:
            kind, out = "discover", machine.facts
        elif "ssh-keygen" in script:
            kind, out = "keygen", f"@@PUBKEY {PUBKEY}\n"
        elif "@@AUTHORIZED" in script:
            quoted = re.search(r'grep -qxF (.+?) "\$f"', script)
            assert quoted, script
            line = shlex.split(quoted.group(1))[0]
            if line not in machine.authorized:
                machine.authorized.append(line)
            kind, out = "authorize", "@@AUTHORIZED\n"
        elif "@@PEER" in script:
            inner = re.search(r"\$\((.*) 2>&1 \| tail -1\)", script)
            assert inner, script
            argv = shlex.split(inner.group(1))
            route = (argv[-2], int(argv[argv.index("-p") + 1]))
            peer = next((m for m in self.machines.values() if route in m.routes), None)
            answer = (
                peer.hostname if peer else "ssh: connect to host: Connection timed out"
            )
            kind, out = "peer", f"@@PEER {answer}\n"
        elif "@@CLEANED" in script:
            machine.cleanups.append(script)
            kind, out = "cleanup", "@@CLEANED\n"
        elif "@@SHA256" in script:
            data = call.input_data or b""
            digest = hashlib.sha256(data + self.upload_noise).hexdigest()
            kind, out = "upload", f"@@SHA256 {digest}\n"
        else:
            self.log.append((where, "runner"))
            return super()._ssh(call)
        self.log.append((where, kind))
        return self._ok(out)

    def kinds(self) -> list[str]:
        return [kind for _where, kind in self.log]


def three_machines() -> FakeMachines:
    node = FakeMachines(FakeClock())
    node.machines = {
        NODE0: Machine(
            "gpu-a", discovery("gpu-a", "eth0 52.1.2.10/24", "ens5f0 10.0.0.11/24")
        ),
        NODE1: Machine(
            "gpu-b",
            discovery("gpu-b", "eth0 52.1.2.11/24", "ens5f0 10.0.0.12/24"),
            routes={("root@10.0.0.12", 22)},
        ),
        # sshd only on the forwarded port: node0 must use the laptop's route
        NODE2: Machine(
            "gpu-c", discovery("gpu-c", "eth1 10.0.0.13/24"), routes={NODE2}
        ),
    }
    return node


def make(tmp_path: Path, node: FakeNode, exp_id: str = EXP) -> Controller:
    return Controller(
        exp_id,
        state_dir=tmp_path / "state",
        lium=LiumCli(sleep=lambda _s: None),
        runner=node,
        clock=node.clock,
        repo_root=tmp_path / "repo",
        ssh_identity=tmp_path / "id_ed25519",
    )


def up_three(tmp_path: Path) -> tuple[FakeMachines, Controller, ControllerState]:
    node = three_machines()
    ctl = make(tmp_path, node)
    path = tmp_path / "machines.txt"
    path.write_text(MACHINES)
    state = ctl.up(UpOptions(machines=ctl_mod.machine_specs(path, None)))
    return node, ctl, state


def run_bash(script: str, tmp_path: Path, **tools: str) -> str:
    """Run ``script`` with HOME=tmp/home and the given fake tools on PATH."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in tools.items():
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body)
        tool.chmod(0o755)
    extra = f":{Path(SSH_KEYGEN).parent}" if SSH_KEYGEN else ""
    env = {"HOME": str(home), "PATH": f"{bin_dir}{extra}:/usr/bin:/bin"}
    res = subprocess.run(
        [BASH or "bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


def heredoc(script: str, name: str) -> str:
    """The content ``launch_script`` writes to ``<exp_dir>/<name>``."""
    match = re.search(
        rf"cat > \S*/\.{re.escape(name)}\.tmp <<'(\w+)'\n(.*?)\n\1\n", script, re.S
    )
    assert match, script
    return match.group(2)


# --------------------------------------------------------------------------
# The machines file
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, access",
    [
        ("root@203.0.113.10", mach.SshAccess("root", "203.0.113.10")),
        ("gpu-a.example.com", mach.SshAccess("root", "gpu-a.example.com")),
        ("ubuntu@gpu-b:2200", mach.SshAccess("ubuntu", "gpu-b", 2200)),
        (
            "ssh://admin@[2001:db8::7]:2222",
            mach.SshAccess("admin", "2001:db8::7", 2222),
        ),
        (
            "ssh -p 2222 -i ~/.ssh/lab root@203.0.113.10",
            mach.SshAccess("root", "203.0.113.10", 2222, "/home/u/.ssh/lab"),
        ),
        (
            "ssh -l ubuntu -o ConnectTimeout=5 -J bastion@198.51.100.1 10.0.0.12",
            mach.SshAccess(
                "ubuntu",
                "10.0.0.12",
                22,
                None,
                "bastion@198.51.100.1",
                (("ConnectTimeout", "5"),),
            ),
        ),
        (
            "ssh -oPort=2200 -oUser=admin -oIdentityFile=/k gpu-c",
            mach.SshAccess("admin", "gpu-c", 2200, "/k"),
        ),
    ],
)
def test_parse_access_forms(text: str, access: mach.SshAccess) -> None:
    assert mach.parse_access(text, "m", home="/home/u") == access


@pytest.mark.parametrize(
    "text, message",
    [
        ("ssh -o ProxyCommand=nc root@x", "is not accepted"),
        ("ssh -o 'ProxyCommand nc %h %p' root@x", "is not accepted"),
        ("ssh -o StrictHostKeyChecking=no root@x", "is not accepted"),
        ("ssh -A root@x", "is not accepted"),
        ("ssh -L 8000:localhost:8000 root@x", "is not accepted"),
        ("ssh --help", "is not accepted"),
        ("ssh root@x nvidia-smi", "remote command"),
        ("ssh root@x:2222", "write the port as -p"),
        ("ssh -p", "needs a value"),
        ("ssh -p 99999 root@x", "invalid ssh port"),
        ("ssh -o ConnectTimeout=5;id root@x", "invalid value"),
        ("ssh", "names no destination"),
        ("ssh -l 'bad user' x", "invalid ssh user"),
        ("root@bad_host!", "invalid host"),
        ("root@x extra", "expected 'ssh"),
        ("", "empty machine"),
    ],
)
def test_parse_access_rejects(text: str, message: str) -> None:
    with pytest.raises(mach.MachineError, match=message):
        mach.parse_access(text, "m")


def test_parse_machine_options() -> None:
    spec = mach.parse_machine(
        "ssh -p 2222 root@203.0.113.10 name=gpu-a control_ip=10.0.0.11 "
        "iface=ens5f0 peer_ssh=10.0.0.11:22",
        "m:1",
    )
    assert spec.access == mach.SshAccess("root", "203.0.113.10", 2222)
    assert (spec.label, spec.control_ip, spec.iface, spec.peer) == (
        "gpu-a",
        "10.0.0.11",
        "ens5f0",
        ("root@10.0.0.11", 22),
    )
    # a lower-case -o value is the ssh option's, not a machine option
    lower = mach.parse_machine("ssh -o connecttimeout=5 root@x")
    assert lower.access.options == (("connecttimeout", "5"),)


@pytest.mark.parametrize(
    "line, message",
    [
        ("root@x color=red", "unknown option"),
        ("root@x control_ip=10.0.0", "not an IPv4 address"),
        ("ubuntu@x peer_ssh=root@10.0.0.2", "must be the access user"),
        ("root@x name=-bad", "invalid name"),
        ("root@x iface=eth0;x", "invalid interface name"),
    ],
)
def test_parse_machine_rejects(line: str, message: str) -> None:
    with pytest.raises(mach.MachineError, match=message):
        mach.parse_machine(line, "m:1")


def test_machines_file_then_flags_node0_first(tmp_path: Path) -> None:
    path = tmp_path / "machines.txt"
    path.write_text(MACHINES)
    specs = ctl_mod.machine_specs(
        path, ["ssh -J bastion@198.51.100.1 root@10.0.0.14 name=gpu-d"]
    )
    assert [s.access.target for s in specs] == [
        "root@203.0.113.10",
        "root@203.0.113.11",
        "root@203.0.113.12",
        "root@10.0.0.14",
    ]
    assert specs[0].access.identity == "/keys/lab" and specs[1].label == "gpu-b"
    assert specs[3].access.jump == "bastion@198.51.100.1"
    with pytest.raises(mach.MachineError, match="listed twice"):
        ctl_mod.machine_specs(path, ["ssh -p 2222 root@203.0.113.10"])
    args = ctl_mod._up_args().parse_args(
        ["--machines", "m.txt", "--machine", "root@a", "--machine", "root@b"]
    )
    assert args.machines == Path("m.txt") and args.machine == ["root@a", "root@b"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("# nothing\n", "no machines listed"),
        ("root@a\nroot@a:22\n", "listed twice"),
        ("root@a name=x\nroot@b name=x\n", "duplicate machine name"),
        (
            "root@a control_ip=10.0.0.1\nroot@b control_ip=10.0.0.1\n",
            "duplicate control_ip",
        ),
    ],
)
def test_machines_file_rejects(text: str, message: str) -> None:
    with pytest.raises(mach.MachineError, match=message):
        mach.parse_machines_text(text, "machines.txt")


# --------------------------------------------------------------------------
# Discovery and control addresses
# --------------------------------------------------------------------------


def test_discovery_facts_and_candidates() -> None:
    facts = mach.parse_discovery(
        discovery(
            "gpu-a",
            "eth0 52.1.2.10/24",
            "ens5f0 10.0.0.11/24",
            "wg0 10.42.0.1/24",
            "tailscale0 100.101.0.7/32",
        )
    )
    assert (facts.hostname, facts.uid, facts.docker, facts.workdir_ok) == (
        "gpu-a",
        0,
        "ok",
        True,
    )
    assert facts.gpus == ("NVIDIA H200",) * 8
    # private (and CGNAT) first, wg/bond/eth/en/ib first; never lo or docker0
    assert [a.ip for a in mach.candidates(facts)] == [
        "10.42.0.1",
        "10.0.0.11",
        "100.101.0.7",
        "52.1.2.10",
    ]
    assert mach.facts_problems(facts, "gpu-a", WORKDIR) == []


def test_facts_problems_say_how_to_fix_them() -> None:
    facts = mach.parse_discovery(
        discovery("gpu-b", gpus=0, docker="denied", workdir="denied")
    )
    assert mach.facts_problems(facts, "gpu-b", WORKDIR) == [
        "gpu-b: nvidia-smi lists no GPU",
        "gpu-b: this ssh user cannot run docker (add it to the docker group or log "
        "in as root)",
        "gpu-b: /root/cluster_bench is not writable (pass --workdir)",
    ]


def test_choose_control_takes_the_first_subnet_every_machine_shares() -> None:
    facts = [
        mach.parse_discovery(discovery("a", "wg0 10.42.0.1/24", "ens5f0 10.0.0.11/24")),
        mach.parse_discovery(discovery("b", "ens5f0 10.0.0.12/24")),
        mach.parse_discovery(discovery("c", "eth1 10.0.0.13/24", "wg0 10.42.0.3/24")),
    ]

    def specs(*lines: str) -> list[mach.MachineSpec]:
        return [mach.parse_machine(line) for line in lines]

    picks = mach.choose_control(specs("root@a", "root@b", "root@c"), facts)
    assert [(p.iface, p.ip) for p in picks] == [
        ("ens5f0", "10.0.0.11"),
        ("ens5f0", "10.0.0.12"),
        ("eth1", "10.0.0.13"),
    ]
    # an iface= limits the machine to that NIC; a control_ip= is taken as given
    pinned = mach.choose_control(
        specs("root@a iface=wg0", "root@b control_ip=10.0.0.12", "root@c"), facts
    )
    assert [p.ip for p in pinned] == ["10.42.0.1", "10.0.0.12", "10.42.0.3"]
    with pytest.raises(mach.MachineError, match="control_ip 10.9.9.9 is not assigned"):
        mach.choose_control(
            specs("root@a", "root@b control_ip=10.9.9.9", "root@c"), facts
        )
    apart = [facts[0], mach.parse_discovery(discovery("b", "eth0 192.168.7.2/24"))]
    with pytest.raises(mach.MachineError, match="no subnet that every machine shares"):
        mach.choose_control(specs("root@a", "root@b"), apart)


def test_peer_candidates() -> None:
    plain = mach.parse_machine("ssh -p 2223 root@203.0.113.11")
    assert mach.peer_candidates(plain, "10.0.0.12") == [
        ("root@10.0.0.12", 22),
        ("root@203.0.113.11", 2223),
    ]
    jumped = mach.parse_machine("ssh -J bastion@198.51.100.1 root@10.0.0.12")
    assert mach.peer_candidates(jumped, "10.0.0.12") == [("root@10.0.0.12", 22)]
    pinned = mach.parse_machine("root@203.0.113.11 peer_ssh=10.0.0.12:2201")
    assert mach.peer_candidates(pinned, "10.0.0.12") == [("root@10.0.0.12", 2201)]


def test_hosts_file_text_is_what_the_runner_parses() -> None:
    control = [
        mach.Address("ens5f0", "10.0.0.11", 24),
        mach.Address("eth1", "10.0.0.12", 24),
    ]
    text = mach.hosts_file_text(
        [("-", None, None), ("root@10.0.0.12", 2201, "bond0")],
        control,
        KEY,
        known_hosts=KH,
    )
    assert prov.parse_hosts_text(text) == [
        NodeAddress("node0", "10.0.0.11", None, 22, control_interface="ens5f0"),
        NodeAddress(
            "node1",
            "10.0.0.12",
            "root@10.0.0.12",
            2201,
            identity=KEY,
            known_hosts=KH,
            control_interface="bond0",
        ),
    ]


# --------------------------------------------------------------------------
# The scripts, in a real shell
# --------------------------------------------------------------------------


@needs_bash
def test_discovery_script_prints_what_parse_discovery_reads(tmp_path: Path) -> None:
    (tmp_path / "ip.txt").write_text(
        "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"
        "2: ens5f0    inet 10.0.0.11/24 brd 10.0.0.255 scope global ens5f0\\  x\n"
        "5: wg0    inet 10.42.0.1/24 scope global wg0\\       valid_lft forever\n"
    )
    out = run_bash(
        mach.discovery_script(str(tmp_path / "w")),
        tmp_path,
        hostname="echo gpu-a\n",
        id="echo 1000\n",
        ip=f"cat '{tmp_path / 'ip.txt'}'\n",
        docker="exit 1\n",
        **{"nvidia-smi": "echo 'NVIDIA H200'\necho 'NVIDIA H200'\n"},
    )
    facts = mach.parse_discovery(out)
    assert (facts.hostname, facts.uid, facts.docker, facts.workdir_ok) == (
        "gpu-a",
        1000,
        "denied",
        True,
    )
    assert [(a.iface, a.ip, a.prefix) for a in facts.addresses] == [
        ("lo", "127.0.0.1", 8),
        ("ens5f0", "10.0.0.11", 24),
        ("wg0", "10.42.0.1", 24),
    ]
    assert facts.gpus == ("NVIDIA H200", "NVIDIA H200")


@needs_bash
@pytest.mark.skipif(SSH_KEYGEN is None, reason="ssh-keygen is not installed")
def test_keygen_makes_one_key_that_authorize_accepts(tmp_path: Path) -> None:
    workdir = str(tmp_path / "w")
    first = ctl_mod.marker_values(
        run_bash(mach.keygen_script(workdir, EXP), tmp_path), "@@PUBKEY"
    )
    again = ctl_mod.marker_values(
        run_bash(mach.keygen_script(workdir, EXP), tmp_path), "@@PUBKEY"
    )
    assert first == again and len(first) == 1
    assert first[0].startswith("ssh-ed25519 ") and first[0].endswith(
        " cluster_bench:exp42"
    )
    mach.authorize_script(first[0], EXP)  # a real key passes the checks
    assert (tmp_path / "w" / "ssh" / f"{EXP}_ed25519").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "key",
    [
        "ssh-rsa AAAAB3Nza cluster_bench:exp42",
        "ssh-ed25519 AAAAC3Nza cluster_bench:exp7",
        "ssh-ed25519 AAAA$(id) cluster_bench:exp42",
        "ssh-ed25519 AAAAC3Nza",
    ],
)
def test_authorize_refuses_unexpected_keys(key: str) -> None:
    with pytest.raises(mach.MachineError):
        mach.authorize_script(key, EXP)


@needs_bash
def test_authorize_then_cleanup_touch_only_the_trust_key(tmp_path: Path) -> None:
    keys = tmp_path / "home" / ".ssh" / "authorized_keys"
    keys.parent.mkdir(parents=True)
    keys.write_text("ssh-ed25519 AAAAuserkey me@laptop\n")
    for _ in range(2):  # idempotent
        assert "@@AUTHORIZED" in run_bash(mach.authorize_script(PUBKEY, EXP), tmp_path)
    assert keys.read_text().splitlines() == [
        "ssh-ed25519 AAAAuserkey me@laptop",
        PUBKEY,
    ]
    assert keys.stat().st_mode & 0o777 == 0o600
    log = tmp_path / "docker.log"
    docker = f"echo \"$@\" >> '{log}'\n[ \"$1\" = ps ] && printf 'c1\\nc2\\n'\nexit 0\n"
    script = mach.cleanup_script(EXP, "exp42", False, str(tmp_path / "w"))
    assert "@@CLEANED" in run_bash(script, tmp_path, docker=docker)
    assert keys.read_text().splitlines() == ["ssh-ed25519 AAAAuserkey me@laptop"]
    assert log.read_text().splitlines() == [
        "ps -aq --filter label=cb.exp=exp42",
        "rm -f c1 c2",
    ]


@needs_bash
def test_cleanup_never_truncates_authorized_keys(tmp_path: Path) -> None:
    keys = tmp_path / "home" / ".ssh" / "authorized_keys"
    keys.parent.mkdir(parents=True)
    before = f"ssh-ed25519 AAAAuserkey me@laptop\n{PUBKEY}\n"
    keys.write_text(before)
    script = mach.cleanup_script(EXP, "exp42", False, "/w")
    run_bash(script, tmp_path, awk="echo partial\nexit 2\n", docker="exit 0\n")
    assert keys.read_text() == before
    assert not (keys.parent / "authorized_keys.cb-tmp").exists()


@needs_bash
def test_cleanup_on_node0_removes_the_experiment_key(tmp_path: Path) -> None:
    ssh_dir = tmp_path / "w" / "ssh"
    ssh_dir.mkdir(parents=True)
    for name in (
        f"{EXP}_ed25519",
        f"{EXP}_ed25519.pub",
        f"{EXP}_known_hosts",
        "exp7_ed25519",
    ):
        (ssh_dir / name).write_text("x")
    run_bash(
        mach.cleanup_script(EXP, "exp42", True, str(tmp_path / "w")),
        tmp_path,
        docker="exit 0\n",
    )
    assert sorted(p.name for p in ssh_dir.iterdir()) == ["exp7_ed25519"]


@needs_bash
def test_peer_check_uses_the_experiment_key(tmp_path: Path) -> None:
    log = tmp_path / "ssh.log"
    ssh = f"printf '%s\\n' \"$@\" > '{log}'\necho gpu-b\n"
    script = mach.peer_check_script(WORKDIR, EXP, "root@10.0.0.12", 2201)
    assert ctl_mod.marker_values(run_bash(script, tmp_path, ssh=ssh), "@@PEER") == [
        "gpu-b"
    ]
    argv = log.read_text().splitlines()
    assert argv[argv.index("-i") + 1] == KEY and argv[argv.index("-p") + 1] == "2201"
    assert argv[-2:] == ["root@10.0.0.12", "hostname"]
    assert "BatchMode=yes" in argv
    assert f"UserKnownHostsFile={WORKDIR}/ssh/{EXP}_known_hosts" in argv


# --------------------------------------------------------------------------
# Controller: up -> launch -> down on SSH machines
# --------------------------------------------------------------------------


def test_up_checks_and_records_the_machines(tmp_path: Path) -> None:
    node, ctl, state = up_three(tmp_path)
    assert state.provider == "ssh" and state.created_by_harness is False
    assert (state.cluster_name, state.node_count, state.gpu) == ("ssh-exp42", 3, "h200")
    assert state.control_ips == ("10.0.0.11", "10.0.0.12", "10.0.0.13")
    members = sorted(state.members, key=lambda m: m.rank)
    assert [(m.name, m.label, m.hostname, m.control_interface) for m in members] == [
        ("node0", "gpu-a", "gpu-a", "ens5f0"),
        ("node1", "gpu-b", "gpu-b", "ens5f0"),
        ("node2", "gpu-c", "gpu-c", "eth1"),
    ]
    assert [m.peer_ssh for m in members] == [
        None,
        SshEndpoint(target="root@10.0.0.12", port=22),
        SshEndpoint(target="root@203.0.113.12", port=2200),  # port 22 refused
    ]
    assert state.node0_ssh == SshEndpoint(
        target="root@203.0.113.10", port=2222, identity="/keys/lab"
    )
    assert node.machines[NODE1].authorized == [PUBKEY]
    assert node.machines[NODE2].authorized == [PUBKEY]
    assert node.machines[NODE0].authorized == []
    assert node.kinds() == ["discover"] * 3 + [
        "keygen",
        "authorize",
        "peer",
        "authorize",
        "peer",
        "peer",
    ]
    first, second = node.calls[0].argv, node.calls[1].argv
    assert first[first.index("-i") + 1] == "/keys/lab"  # the machine's own key
    assert second[second.index("-i") + 1] == str(tmp_path / "id_ed25519")
    assert ctl.load_state() == state
    saved = json.loads(ctl.state_path.read_text())
    assert saved["node0_ssh"] == {
        "target": "root@203.0.113.10",
        "port": 2222,
        "identity": "/keys/lab",
    }


def test_launch_writes_the_runner_hosts_file(tmp_path: Path) -> None:
    node, ctl, _state = up_three(tmp_path)
    ctl.launch(runner_args=["--stage", "p-only", "--budget-hours", "4"])
    launch = node.launches[-1]
    assert launch.input_data == f"{HF_TOKEN}\n".encode()
    assert HF_TOKEN not in "\0".join(launch.argv)
    script = launch.script
    assert prov.parse_hosts_text(heredoc(script, "hosts.txt"), "hosts.txt") == [
        NodeAddress("node0", "10.0.0.11", None, 22, control_interface="ens5f0"),
        NodeAddress(
            "node1",
            "10.0.0.12",
            "root@10.0.0.12",
            22,
            identity=KEY,
            known_hosts=KH,
            control_interface="ens5f0",
        ),
        NodeAddress(
            "node2",
            "10.0.0.13",
            "root@203.0.113.12",
            2200,
            identity=KEY,
            known_hosts=KH,
            control_interface="eth1",
        ),
    ]
    provider = json.loads(heredoc(script, "provider.json"))
    assert provider["provider"] == "ssh" and provider["gpu_type"] == "h200"
    assert (
        f"run --exp-id {EXP} --workdir {WORKDIR} --platform generic --hosts "
        f"{WORKDIR}/experiments/{EXP}/hosts.txt --stage p-only --budget-hours 4"
    ) in script
    assert " --nodes " not in script
    with pytest.raises(ControllerError, match="--runner-nodes does not apply"):
        ctl.launch(runner_nodes="10.0.0.11,10.0.0.12", force=True)


def test_down_cleans_up_only_after_the_results_are_pulled(tmp_path: Path) -> None:
    node, ctl, _state = up_three(tmp_path)
    assert ctl.down() == 1  # nothing pulled yet: nothing removed
    assert "cleanup" not in node.kinds()
    ctl.save_state(ctl.load_state().evolve(pulled_ok=True))
    assert ctl.down() == 0
    assert [where for where, kind in node.log if kind == "cleanup"] == [
        NODE2,
        NODE1,
        NODE0,  # node0 holds the key: last
    ]
    node0_script = node.machines[NODE0].cleanups[0]
    assert f"rm -f {KEY} {KEY}.pub" in node0_script
    assert "label=cb.exp=exp42" in node0_script
    assert KEY not in node.machines[NODE1].cleanups[0]
    state = ctl.load_state()
    assert state.released_at is not None and state.destroyed_at is None
    assert ctl.down() == 0 and len(node.machines[NODE0].cleanups) == 1
    # released machines can be brought up again under the same experiment
    again = ctl.up(
        UpOptions(machines=ctl_mod.machine_specs(tmp_path / "machines.txt", None))
    )
    assert again.released_at is None


def test_up_refuses_a_second_recording(tmp_path: Path) -> None:
    _node, ctl, _state = up_three(tmp_path)
    one = (mach.parse_machine("ssh -p 2222 root@203.0.113.10"),)
    with pytest.raises(ControllerError, match="already has 3 SSH machine"):
        ctl.up(UpOptions(machines=one))
    lium = make(tmp_path / "lium", three_machines())
    lium.save_state(
        ControllerState(
            exp_id=EXP,
            cluster_name="cb-b200-exp42",
            cluster_id="4c1f9a2e",
            created_by_harness=True,
            created_at=1.0,
        )
    )
    with pytest.raises(ControllerError, match="already has lium cluster"):
        lium.up(UpOptions(machines=one))


def test_up_reports_every_problem_before_changing_a_machine(tmp_path: Path) -> None:
    node = FakeMachines(FakeClock())
    node.machines = {
        ("root@203.0.113.10", 22): Machine(
            "gpu-a", discovery("gpu-a", "ens5f0 10.0.0.11/24", docker="denied")
        ),
        ("root@203.0.113.11", 22): Machine(
            "gpu-b",
            discovery("gpu-b", "ens5f0 10.0.0.12/24", gpu="NVIDIA H100 80GB HBM3"),
        ),
    }
    ctl = make(tmp_path, node)
    specs = tuple(mach.parse_machine(f"root@203.0.113.1{i}") for i in range(2))
    with pytest.raises(ControllerError) as err:
        ctl.up(UpOptions(machines=specs))
    assert "gpu-a: this ssh user cannot run docker" in str(err.value)
    assert "the machines differ in GPUs" in str(err.value)
    assert node.kinds() == ["discover", "discover"]
    assert ctl.load_state_optional() is None


def test_up_names_the_fix_for_an_unreachable_peer(tmp_path: Path) -> None:
    node = FakeMachines(FakeClock())
    node.machines = {
        ("root@203.0.113.10", 22): Machine(
            "gpu-a", discovery("gpu-a", "ens5f0 10.0.0.11/24")
        ),
        ("root@203.0.113.11", 22): Machine(
            "gpu-b", discovery("gpu-b", "ens5f0 10.0.0.12/24")
        ),
    }
    ctl = make(tmp_path, node)
    specs = tuple(mach.parse_machine(f"root@203.0.113.1{i}") for i in range(2))
    with pytest.raises(ControllerError, match="peer_ssh=USER@HOST"):
        ctl.up(UpOptions(machines=specs))
    # recorded before the key went in, so down --force can take it back out
    assert ctl.load_state().provider == "ssh"
    assert ctl.down(force=True) == 0
    assert [where for where, kind in node.log if kind == "cleanup"] == [
        ("root@203.0.113.11", 22),
        ("root@203.0.113.10", 22),
    ]


def test_up_reports_a_refused_ssh(tmp_path: Path) -> None:
    node = three_machines()
    node.ssh_ok = False
    ctl = make(tmp_path, node)
    path = tmp_path / "machines.txt"
    path.write_text(MACHINES)
    with pytest.raises(
        ControllerError, match=r"machine 0 \(root@203\.0\.113\.10 -p 2222\) refused ssh"
    ):
        ctl.up(UpOptions(machines=ctl_mod.machine_specs(path, None)))


def test_one_machine_needs_no_trust_key(tmp_path: Path) -> None:
    node = FakeMachines(FakeClock())
    node.machines = {
        ("root@203.0.113.10", 22): Machine(
            "gpu-a", discovery("gpu-a", "ens5f0 10.0.0.11/24", "bond0 10.1.0.11/16")
        )
    }
    ctl = make(tmp_path, node)
    state = ctl.up(
        UpOptions(machines=(mach.parse_machine("root@203.0.113.10 iface=bond0"),))
    )
    assert node.kinds() == ["discover"] and state.control_ips == ("10.1.0.11",)
    assert ctl_mod.hosts_text(state).splitlines()[1:] == [
        "node0 10.1.0.11 - iface=bond0"
    ]


def test_status_lists_the_machines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _node, ctl, _state = up_three(tmp_path)
    capsys.readouterr()
    ctl.status()
    out = capsys.readouterr().out
    assert "gpu-b" in out and "from node0 root@10.0.0.12 -p 22" in out
    assert "from node0 root@203.0.113.12 -p 2200" in out


def test_upload_puts_a_dataset_on_node0(tmp_path: Path) -> None:
    node, ctl, _state = up_three(tmp_path)
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"timestamp": 0, "input_length": 10, "output_length": 5}\n')
    assert ctl.upload([trace]) == [f"{WORKDIR}/inputs/trace.jsonl"]
    call = next(c for c in node.calls if "@@SHA256" in c.script)
    assert destination(call.argv) == NODE0 and call.input_data == trace.read_bytes()
    bad = tmp_path / "my trace.jsonl"
    bad.write_text("x")
    with pytest.raises(ControllerError, match="may only use letters"):
        ctl.upload([bad])
    node.upload_noise = b"!"
    with pytest.raises(ControllerError, match="failed its sha256 check"):
        ctl.upload([trace])


@needs_bash
def test_upload_script_writes_atomically_and_prints_the_digest(tmp_path: Path) -> None:
    dest = tmp_path / "w" / "inputs" / "t.jsonl"
    res = subprocess.run(
        [BASH or "bash", "-c", ctl_mod.upload_script(str(dest))],
        input=b"hello\n",
        capture_output=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert dest.read_bytes() == b"hello\n"
    assert not dest.with_name("t.jsonl.part").exists()
    assert ctl_mod.marker_values(res.stdout.decode(), "@@SHA256") == [
        hashlib.sha256(b"hello\n").hexdigest()
    ]


def test_ssh_machines_do_not_borrow_the_lium_key(tmp_path: Path) -> None:
    later = argparse.Namespace()  # sync, launch, ...: no machines on the line
    assert (
        ctl_mod._uses_lium_key(argparse.Namespace(machine=["root@a"]), tmp_path, EXP)
        is False
    )
    assert (
        ctl_mod._uses_lium_key(later, tmp_path / "state", EXP) is True
    )  # no state yet
    up_three(tmp_path)
    assert ctl_mod._uses_lium_key(later, tmp_path / "state", EXP) is False
