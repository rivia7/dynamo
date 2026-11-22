# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for adapters/providers.py: existing clusters and the lium adapter.

The lium adapter runs a fake ``lium`` executable (fixtures/lium/fake_lium.py)
placed on PATH; it replays outputs shaped like lium.io 0.41.1's JSON.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from benchmarks.lium_cluster_bench.adapters import providers as prov
from benchmarks.lium_cluster_bench.adapters.providers import (
    ExistingClusterProvider,
    LiumApiError,
    LiumAuthError,
    LiumBillingUncertainError,
    LiumCli,
    LiumConfigError,
    LiumGeneralError,
    LiumNotFoundError,
    LiumNotInstalledError,
    LiumParseError,
    LiumPermissionError,
    LiumProvider,
    LiumSshError,
    LiumSshHostKeyError,
    LiumTimeoutError,
    ProviderError,
)
from benchmarks.lium_cluster_bench.ports import CommandResult, NodeAddress
from tests.benchmarks.lium_cluster_bench.lium_fakes import (
    API_KEY,
    CONTROLLER_FIXTURES,
    HF_TOKEN,
    LIUM_FIXTURES,
    FakeLium,
    fail,
    fixture_text,
    ok,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@dataclass
class ScriptedRunner:
    """ArgvRunner double: records argv, answers from a function."""

    answer: Any = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv, *, input_data=None, timeout=None, env=None
    ) -> CommandResult:
        self.calls.append(list(argv))
        rc, out = self.answer(list(argv)) if self.answer else (0, "")
        return CommandResult("local", " ".join(argv), rc, out, "")


def no_sleep(_seconds: float) -> None:
    return None


# --------------------------------------------------------------------------
# Existing clusters
# --------------------------------------------------------------------------


def test_hosts_file_with_comments_defaults_and_ports(tmp_path: Path) -> None:
    hosts = tmp_path / "hosts.txt"
    hosts.write_text(
        "# name address [ssh_target] [ssh_port]\n"
        "\n"
        "node0 10.42.0.1            # this host\n"
        "node1 10.42.0.2 -\n"
        "node2 10.42.0.3 admin@gpu-b 2222\n"
        "node3 10.42.0.4 - 2200\n"
    )
    provider = ExistingClusterProvider.from_hosts_file(hosts)
    assert provider.nodes() == [
        NodeAddress("node0", "10.42.0.1", None, 22),
        NodeAddress("node1", "10.42.0.2", "root@10.42.0.2", 22),
        NodeAddress("node2", "10.42.0.3", "admin@gpu-b", 2222),
        NodeAddress("node3", "10.42.0.4", "root@10.42.0.4", 2200),
    ]
    assert provider.created_by_harness is False


def test_hosts_file_node0_may_be_explicitly_local_or_remote() -> None:
    local = prov.parse_hosts_text("node0 10.42.0.1 -\nnode1 10.42.0.2\n")
    assert local[0].ssh_target is None
    remote = prov.parse_hosts_text("node0 10.42.0.1 root@jump 2022\n")
    assert remote == [NodeAddress("node0", "10.42.0.1", "root@jump", 2022)]


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "no hosts listed"),
        ("node1 10.42.0.2\n", "first host must be node0"),
        ("node0 10.42.0.1\nnode0 10.42.0.2\n", "duplicate node name"),
        ("node0 10.42.0.1\nnode1 10.42.0.1\n", "duplicate node address"),
        ("node0 gpu-a.local\n", "not an IPv4 address"),
        ("node0 10.42.0.1 - 22\n", "takes no ssh port"),
        ("node0 10.42.0.1\nnode1 10.42.0.2 root@x 70000\n", "invalid ssh port"),
        ("node0 10.42.0.1\nnode1 10.42.0.2 -oProxyCommand=x\n", "invalid ssh target"),
        ("node0\n", "expected 'name address"),
        ("node0 10.42.0.1 a b c\n", "expected 'name address"),
    ],
)
def test_hosts_file_rejects_bad_lines(text: str, message: str) -> None:
    with pytest.raises(ProviderError, match=message):
        prov.parse_hosts_text(text, "hosts.txt")


def test_parse_ip_addr_fixture() -> None:
    text = fixture_text(CONTROLLER_FIXTURES / "ip_addr_node0.txt")
    assert prov.parse_ip_addr(text) == (
        "127.0.0.1",
        "172.17.0.2",
        "192.168.10.11",
        "10.42.0.1",
    )


def test_local_ipv4_addresses_runs_ip_and_fails_loudly() -> None:
    text = fixture_text(CONTROLLER_FIXTURES / "ip_addr_node0.txt")
    runner = ScriptedRunner(lambda argv: (0, text))
    assert "10.42.0.1" in prov.local_ipv4_addresses(runner)
    assert runner.calls == [["ip", "-4", "-o", "addr", "show"]]
    with pytest.raises(ProviderError, match="exit 1"):
        prov.local_ipv4_addresses(ScriptedRunner(lambda argv: (1, "")))
    with pytest.raises(ProviderError, match="garbage"):
        prov.local_ipv4_addresses(ScriptedRunner(lambda argv: (0, "garbage\n")))


def test_ip_list_local_entry_becomes_node0() -> None:
    provider = ExistingClusterProvider.from_ip_list(
        "10.42.0.3, 10.42.0.1,10.42.0.2", local_addresses={"10.42.0.1", "127.0.0.1"}
    )
    assert provider.nodes() == [
        NodeAddress("node0", "10.42.0.1", None),
        NodeAddress("node1", "10.42.0.3", "root@10.42.0.3"),
        NodeAddress("node2", "10.42.0.2", "root@10.42.0.2"),
    ]
    as_sequence = ExistingClusterProvider.from_ip_list(
        ("10.42.0.1", "10.42.0.2"), local_addresses=["10.42.0.1"]
    )
    assert [n.name for n in as_sequence.nodes()] == ["node0", "node1"]


@pytest.mark.parametrize(
    "ips, local, message",
    [
        ("10.42.0.2,10.42.0.3", {"10.42.0.1"}, "exactly one of"),
        ("10.42.0.1,10.42.0.1", {"10.42.0.1"}, "duplicate"),
        ("10.42.0.1,node-b", {"10.42.0.1"}, "not an IPv4 address"),
        ("", {"10.42.0.1"}, "empty node IP list"),
    ],
)
def test_ip_list_errors(ips: str, local: set[str], message: str) -> None:
    with pytest.raises(ProviderError, match=message):
        ExistingClusterProvider.from_ip_list(ips, local_addresses=local)


def test_ip_list_uses_ip_command_when_no_addresses_given() -> None:
    text = fixture_text(CONTROLLER_FIXTURES / "ip_addr_node0.txt")
    provider = ExistingClusterProvider.from_ip_list(
        ["10.42.0.2", "10.42.0.1"], runner=ScriptedRunner(lambda argv: (0, text))
    )
    assert provider.nodes()[0] == NodeAddress("node0", "10.42.0.1", None)


def test_auto_discovers_reachable_peers_local_first() -> None:
    reachable = {"10.42.0.2", "10.42.0.4", "10.42.0.3"}
    probed: list[str] = []

    def probe(ip: str) -> bool:
        probed.append(ip)
        return ip in reachable

    provider = ExistingClusterProvider.auto(
        4, local_addresses={"10.42.0.1", "172.17.0.2"}, probe=probe
    )
    assert [(n.name, n.address, n.ssh_target) for n in provider.nodes()] == [
        ("node0", "10.42.0.1", None),
        ("node1", "10.42.0.2", "root@10.42.0.2"),
        ("node2", "10.42.0.3", "root@10.42.0.3"),
        ("node3", "10.42.0.4", "root@10.42.0.4"),
    ]
    assert sorted(probed) == sorted(f"10.42.0.{i}" for i in range(2, 17))


def test_auto_node0_need_not_be_dot_one() -> None:
    provider = ExistingClusterProvider.auto(
        2, local_addresses={"10.42.0.3"}, probe=lambda ip: ip == "10.42.0.1"
    )
    assert [n.address for n in provider.nodes()] == ["10.42.0.3", "10.42.0.1"]


def test_auto_requires_the_expected_count_and_a_local_address() -> None:
    with pytest.raises(ProviderError, match="found 2 node"):
        ExistingClusterProvider.auto(
            4, local_addresses={"10.42.0.1"}, probe=lambda ip: ip == "10.42.0.2"
        )
    with pytest.raises(ProviderError, match="exactly one local address"):
        ExistingClusterProvider.auto(4, local_addresses={"172.17.0.2"}, probe=bool)
    with pytest.raises(ValueError):
        ExistingClusterProvider.auto(0, local_addresses={"10.42.0.1"}, probe=bool)


def test_auto_default_probe_is_batch_mode_root_ssh() -> None:
    runner = ScriptedRunner(
        lambda argv: (0 if argv[-2] == "root@10.42.0.2" else 255, "")
    )
    provider = ExistingClusterProvider.auto(
        2, local_addresses={"10.42.0.1"}, runner=runner
    )
    assert [n.address for n in provider.nodes()] == ["10.42.0.1", "10.42.0.2"]
    probe = next(c for c in runner.calls if c[-2] == "root@10.42.0.2")
    assert probe[0] == "ssh" and probe[-1] == "true"
    assert "BatchMode=yes" in probe and "ConnectTimeout=3" in probe
    assert len(runner.calls) == 15


def test_existing_cluster_is_never_destroyed() -> None:
    provider = ExistingClusterProvider([NodeAddress("node0", "10.42.0.1", None)])
    with pytest.raises(ProviderError, match="did not create"):
        provider.destroy()
    with pytest.raises(ProviderError, match="only the first node"):
        ExistingClusterProvider(
            [
                NodeAddress("node0", "10.42.0.1", "root@10.42.0.1"),
                NodeAddress("node1", "10.42.0.2", None),
            ]
        )


# --------------------------------------------------------------------------
# lium output parsing (pure)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ssh_cmd, expected",
    [
        ("ssh root@203.0.113.10 -p 40122", ("root@203.0.113.10", 40122)),
        ("ssh ubuntu@gpu-7.example.net", ("ubuntu@gpu-7.example.net", 22)),
        ("ssh root@2001:db8::7 -p 2222", ("root@2001:db8::7", 2222)),
    ],
)
def test_parse_ssh_cmd(ssh_cmd: str, expected: tuple[str, int]) -> None:
    assert prov.parse_ssh_cmd(ssh_cmd) == expected


@pytest.mark.parametrize(
    "ssh_cmd",
    [
        "ssh root@203.0.113.10 -p 0",
        "ssh root@203.0.113.10 -o ProxyCommand=evil",
        "ssh 203.0.113.10",
        "scp root@203.0.113.10",
        "ssh root@host;rm -p 22",
        "ssh -oFoo@host",
        "ssh 'root@host",
    ],
)
def test_parse_ssh_cmd_rejects_unexpected_shapes(ssh_cmd: str) -> None:
    with pytest.raises(ValueError):
        prov.parse_ssh_cmd(ssh_cmd)


def test_parse_cluster_replacement_keeps_the_running_member() -> None:
    raw = (LIUM_FIXTURES / "clusters_show_replacing.json").read_text()
    cluster = prov.parse_cluster(json.loads(raw), raw)
    assert [n.rank for n in cluster.nodes] == [0, 1, 2, 3]
    assert cluster.nodes[2].huid == "lucky-badger-0f"
    assert cluster.ready


def test_parse_cluster_ambiguous_rank_is_an_error_when_strict() -> None:
    data = json.loads((LIUM_FIXTURES / "clusters_show_replacing.json").read_text())
    data["pods"][2]["status"] = "STOPPING"
    with pytest.raises(ProviderError, match="being replaced"):
        prov.parse_cluster(data, json.dumps(data))
    lenient = prov.parse_cluster(data, json.dumps(data), strict=False)
    assert len(lenient.nodes) == 4


def test_parse_cluster_rejects_garbage_with_raw_output() -> None:
    with pytest.raises(LiumParseError, match="no node_rank") as info:
        prov.parse_cluster({"id": "c1", "pods": [{"huid": "x"}]}, "RAW-OUTPUT-1")
    assert "RAW-OUTPUT-1" in str(info.value)
    with pytest.raises(LiumParseError, match="'id' and 'pods'"):
        prov.parse_cluster(["not", "a", "cluster"], "RAW-OUTPUT-2")
    with pytest.raises(LiumParseError, match="not IPv4"):
        prov.parse_cluster(
            {"id": "c1", "pods": [{"node_rank": 0, "overlay_ip": "fd00::1"}]}, "raw"
        )


def test_node_addresses_need_a_ready_cluster() -> None:
    raw = (LIUM_FIXTURES / "clusters_show_b200x4.json").read_text()
    cluster = prov.parse_cluster(json.loads(raw), raw)
    addresses = prov.node_addresses(cluster)
    assert addresses[0] == NodeAddress("node0", "10.42.0.1", "root@203.0.113.10", 40122)
    data = json.loads(raw)
    data["pods"][3].update(status="PENDING", ssh_cmd=None, overlay_ip=None)
    with pytest.raises(ProviderError, match="node3 is PENDING"):
        prov.node_addresses(prov.parse_cluster(data, raw))


@pytest.mark.parametrize(
    "rc, code, cls, retryable",
    [
        (1, None, LiumGeneralError, False),
        (1, "not_enough_nodes", LiumGeneralError, False),
        (2, "no_api_key", LiumConfigError, False),
        (2, "confirmation_required", LiumConfigError, False),
        (3, "rate_limited", LiumApiError, True),
        (3, "server_error", LiumApiError, True),
        (3, None, LiumApiError, True),
        (3, "invalid_api_key", LiumAuthError, False),
        (3, "not_found", LiumNotFoundError, False),
        (3, "cluster_rented_not_listed", LiumBillingUncertainError, False),
        (3, "cluster_rent_uncertain", LiumBillingUncertainError, False),
        (3, "cluster_ttl_not_scheduled", LiumBillingUncertainError, False),
        (4, "ssh_unavailable", LiumSshError, True),
        (4, "ssh_host_key_changed", LiumSshHostKeyError, False),
        (5, "cluster_not_found", LiumNotFoundError, False),
        (6, "insufficient_balance", LiumPermissionError, False),
        (42, None, LiumGeneralError, False),
    ],
)
def test_exit_code_mapping(
    rc: int, code: str | None, cls: type, retryable: bool
) -> None:
    assert prov.classify_failure(rc, code) is cls
    assert cls.retryable is retryable


def test_error_envelope_is_found_after_narration() -> None:
    stderr = fixture_text(LIUM_FIXTURES / "err_server_error.txt")
    code, message, hint = prov.parse_error_envelope(stderr)
    assert (code, message) == ("server_error", "502 Bad Gateway")
    assert hint and hint.startswith("Retry")
    assert prov.parse_error_envelope("Traceback: boom\n") == (None, None, None)


def test_format_ttl() -> None:
    assert prov.format_ttl(12) == "12h"
    assert prov.format_ttl(1.5) == "1.5h"
    assert prov.format_ttl(0.25) == "0.25h"
    with pytest.raises(ValueError):
        prov.format_ttl(0)
    with pytest.raises(ValueError):
        prov.format_ttl(10_000)


def test_resolve_lium_ssh_key(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".lium").mkdir(parents=True)
    assert prov.resolve_lium_ssh_key(env={}, home=home) is None
    (home / ".lium" / "config.ini").write_text(
        "[api]\napi_key = sk_never_read\n\n[ssh]\nkey_path = ~/.ssh/lium_ed25519\n"
    )
    assert prov.resolve_lium_ssh_key(env={}, home=home) == (
        home / ".ssh" / "lium_ed25519"
    )
    env = {"LIUM_SSH_KEY_PATH": str(tmp_path / "k")}
    assert prov.resolve_lium_ssh_key(env=env, home=home) == tmp_path / "k"


def test_assert_no_secrets() -> None:
    env = {"LIUM_API_KEY": API_KEY, "HF_TOKEN": HF_TOKEN, "NGC_API_KEY": "abc"}
    prov.assert_no_secrets(["lium", "ps", "--format", "json"], env)
    with pytest.raises(ValueError, match="secret"):
        prov.assert_no_secrets(["curl", f"-HAuthorization: {HF_TOKEN}"], env)
    # values shorter than 6 characters are too common to guard
    prov.assert_no_secrets(["echo", "abc"], env)


def test_run_argv_conventions(tmp_path: Path) -> None:
    res = prov.run_argv([sys.executable, "-c", "import sys; print(sys.stdin.read())"])
    assert res.rc == 0 and res.stdout == "\n"  # stdin is /dev/null
    res = prov.run_argv(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        input_data=b"abc",
    )
    assert res.stdout.strip() == "ABC"
    missing = prov.run_argv([str(tmp_path / "no-such-binary")])
    assert missing.rc == prov.MISSING_BINARY_RC
    slow = prov.run_argv(
        [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5
    )
    assert slow.rc == prov.TIMEOUT_RC and slow.stderr.endswith("[timeout]")


# --------------------------------------------------------------------------
# LiumCli against the fake binary
# --------------------------------------------------------------------------


def test_cluster_show_parses_members(fake_lium: FakeLium) -> None:
    fake_lium.script(clusters_show=[ok("clusters_show_b200x4.json")])
    nodes = LiumCli().cluster_show("cb-b200-exp42")
    assert [n.name for n in nodes] == ["node0", "node1", "node2", "node3"]
    assert [n.control_ip for n in nodes] == [f"10.42.0.{i}" for i in range(1, 5)]
    assert nodes[0].ssh_target == "root@203.0.113.10" and nodes[0].ssh_port == 40122
    assert nodes[3].ssh_target == "root@203.0.113.13" and nodes[3].ssh_port == 40422
    assert {n.gpu_type for n in nodes} == {"B200"} and nodes[0].gpu_count == 8
    call = fake_lium.calls()[0]
    assert call["argv"] == ["clusters", "show", "cb-b200-exp42", "--format", "json"]


def test_lium_environment_and_no_secret_on_argv(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_up=[ok("clusters_up_b200x4.json")],
        clusters_ps=[ok("clusters_ps.json")],
        ps=[ok("ps.json")],
        clusters_list=[ok("clusters_list.json")],
        clusters_rm=[ok("clusters_rm_ok.json")],
    )
    cli = LiumCli()
    cli.cluster_up("ib-sjc1-b200-a7f3", 4, "cb-b200-exp42", 12)
    cli.list_clusters()
    cli.list_pods()
    cli.list_fabrics()
    cli.cluster_rm("4c1f9a2e-7b3d-4e8a-9f61-2d5c8b0e7a13")
    calls = fake_lium.calls()
    assert len(calls) == 5
    for call in calls:
        assert call["noninteractive"] == "1"
        assert call["output"] == "json"
        assert call["self_update_disable"] == "1"
        assert call["api_key_set"] is True  # inherited through the environment
        assert call["hf_token_set"] is False  # lium never sees the HF token
        joined = " ".join(call["argv"])
        assert API_KEY not in joined and HF_TOKEN not in joined


def test_secret_on_lium_argv_is_refused_before_running(fake_lium: FakeLium) -> None:
    fake_lium.script(ps=[ok("ps.json")])
    with pytest.raises(ValueError, match="secret"):
        LiumCli().run(["ps", "--filter", f"name={API_KEY}"])
    assert fake_lium.calls() == []


def test_cluster_up_argv_and_result(fake_lium: FakeLium) -> None:
    fake_lium.script(clusters_up=[ok("clusters_up_b200x4.json")])
    cluster = LiumCli().cluster_up(
        "ib-sjc1-b200-a7f3", 4, "cb-b200-exp42", 12, "tmpl-123", wait_timeout_s=1200
    )
    assert cluster.id == "4c1f9a2e-7b3d-4e8a-9f61-2d5c8b0e7a13"
    assert cluster.removal_scheduled_at == "2026-11-04T06:12:45.118204+00:00"
    assert cluster.ready and len(cluster.nodes) == 4
    argv = fake_lium.calls()[0]["argv"]
    assert argv[:3] == ["clusters", "up", "ib-sjc1-b200-a7f3"]
    for flag, value in (
        ("--nodes", "4"),
        ("--name", "cb-b200-exp42"),
        ("--ttl", "12h"),
        ("--timeout", "1200"),
        ("--template_id", "tmpl-123"),
        ("--format", "json"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert "--yes" in argv and "--wait" in argv and "--no-backup" in argv


def test_cluster_up_validates_before_renting(fake_lium: FakeLium) -> None:
    fake_lium.script(clusters_up=[ok("clusters_up_b200x4.json")])
    cli = LiumCli()
    with pytest.raises(ValueError, match="at least 2"):
        cli.cluster_up("f", 1, "cb-x", 12)
    with pytest.raises(ValueError, match="invalid cluster name"):
        cli.cluster_up("f", 4, "bad name; rm -rf /", 12)
    with pytest.raises(ValueError, match="TTL"):
        cli.cluster_up("f", 4, "cb-x", 0)
    assert fake_lium.calls() == []


def test_cluster_up_is_never_retried(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_up=[
            fail(3, "err_rented_not_listed.txt"),
            ok("clusters_up_b200x4.json"),
        ]
    )
    cli = LiumCli(sleep=no_sleep, read_retries=5)
    with pytest.raises(LiumBillingUncertainError) as info:
        cli.cluster_up("ib-sjc1-b200-a7f3", 4, "cb-b200-exp42", 12)
    assert info.value.code == "cluster_rented_not_listed"
    assert "Do not re-run" in (info.value.hint or "")
    fake_lium.script(
        clusters_up=[fail(3, "err_rate_limited.txt"), ok("clusters_up_b200x4.json")]
    )
    with pytest.raises(LiumApiError):
        cli.cluster_up("ib-sjc1-b200-a7f3", 4, "cb-b200-exp42", 12)
    assert [c["key"] for c in fake_lium.calls()] == ["clusters up", "clusters up"]


def test_ttl_not_scheduled_reported_on_stdout_is_billing_uncertain(
    fake_lium: FakeLium, tmp_path: Path
) -> None:
    data = json.loads((LIUM_FIXTURES / "clusters_up_b200x4.json").read_text())
    data.update(
        ok=False,
        removal_scheduled_at=None,
        unscheduled=[{"node_rank": 3, "huid": "swift-lynx-d4", "id": "2a3b"}],
        error="Cluster 4c1f9a2e is rented and billing; auto-termination was NOT scheduled",
    )
    out = tmp_path / "ttl_failed.json"
    out.write_text(json.dumps(data))
    fake_lium.script(clusters_up=[{"rc": 3, "stdout_file": str(out)}])
    with pytest.raises(LiumBillingUncertainError, match="NOT scheduled"):
        LiumCli().cluster_up("ib-sjc1-b200-a7f3", 4, "cb-b200-exp42", 12)


def test_reads_retry_retryable_errors_with_backoff(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_show=[
            fail(3, "err_rate_limited.txt"),
            fail(4, "err_ssh_unavailable.txt"),
            ok("clusters_show_b200x4.json"),
        ]
    )
    delays: list[float] = []
    cli = LiumCli(sleep=delays.append, read_retries=2, backoff_s=5)
    assert len(cli.cluster_show("cb-b200-exp42")) == 4
    assert delays == [5, 10]
    assert len(fake_lium.calls()) == 3


def test_retries_exhausted_raise_the_last_error(fake_lium: FakeLium) -> None:
    fake_lium.script(clusters_ps=[fail(3, "err_rate_limited.txt")])
    with pytest.raises(LiumApiError, match="Rate limit") as info:
        LiumCli(sleep=no_sleep, read_retries=2).list_clusters()
    assert info.value.code == "rate_limited" and info.value.retryable
    assert len(fake_lium.calls()) == 3


@pytest.mark.parametrize(
    "rc, stderr_name, cls",
    [
        (2, "err_no_api_key.txt", LiumConfigError),
        (3, "err_invalid_api_key.txt", LiumAuthError),
        (5, "err_cluster_not_found.txt", LiumNotFoundError),
        (6, "err_insufficient_balance.txt", LiumPermissionError),
        (4, "err_ssh_host_key.txt", LiumSshHostKeyError),
        (1, "err_not_enough_nodes.txt", LiumGeneralError),
    ],
)
def test_fatal_errors_are_typed_and_not_retried(
    fake_lium: FakeLium, rc: int, stderr_name: str, cls: type
) -> None:
    fake_lium.script(
        clusters_show=[fail(rc, stderr_name), ok("clusters_show_b200x4.json")]
    )
    with pytest.raises(cls) as info:
        LiumCli(sleep=no_sleep, read_retries=3).cluster_show("cb-b200-gone")
    assert info.value.rc == rc and not info.value.retryable
    assert len(fake_lium.calls()) == 1


def test_non_json_output_fails_with_the_raw_text(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_show=[{"rc": 0, "stdout": "Cluster 4c1f (4 nodes) RUNNING\n"}]
    )
    with pytest.raises(LiumParseError) as info:
        LiumCli().cluster_show("cb-b200-exp42")
    assert "Cluster 4c1f (4 nodes) RUNNING" in str(info.value)


def test_unexpected_json_shape_fails_with_the_raw_text(fake_lium: FakeLium) -> None:
    fake_lium.script(clusters_ps=[{"rc": 0, "stdout": '{"clusters": []}'}])
    with pytest.raises(LiumParseError, match="list of clusters") as info:
        LiumCli().list_clusters()
    assert '{"clusters": []}' in str(info.value)


def test_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(LiumNotInstalledError, match="lium.io"):
        LiumCli(str(tmp_path / "lium")).list_clusters()


def test_process_timeout_is_retryable() -> None:
    def answer(argv, **kwargs):
        return CommandResult("local", "lium clusters ps", 124, "", "\n[timeout]")

    with pytest.raises(LiumTimeoutError):
        LiumCli(runner=answer, sleep=no_sleep, read_retries=0).list_clusters()


def test_cluster_rm_partial_failure_is_a_retryable_api_error(
    fake_lium: FakeLium,
) -> None:
    fake_lium.script(clusters_rm=[fail(3, stdout_name="clusters_rm_partial.json")])
    with pytest.raises(LiumApiError, match="1 of 4 members were not removed"):
        LiumCli(sleep=no_sleep, read_retries=1).cluster_rm("4c1f9a2e")
    assert len(fake_lium.calls()) == 2  # re-running retries the failed members
    fake_lium.script(clusters_rm=[ok("clusters_rm_ok.json")])
    results = LiumCli().cluster_rm("4c1f9a2e")
    assert len(results) == 4 and all(r["success"] for r in results)
    assert fake_lium.calls()[-1]["argv"] == [
        "clusters",
        "rm",
        "4c1f9a2e",
        "--yes",
        "--format",
        "json",
    ]


def test_list_fabrics_clusters_and_pods(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_list=[ok("clusters_list.json")],
        clusters_ps=[ok("clusters_ps.json")],
        ps=[ok("ps.json")],
    )
    cli = LiumCli()
    fabrics = cli.list_fabrics()
    assert [(f.fabric_id, f.gpu_type, f.free_count) for f in fabrics] == [
        ("ib-sjc1-b200-a7f3", "B200", 5),
        ("roce-dfw2-b200-11c0", "B200", 6),
        ("ib-ams3-h200-5e2d", "H200", 2),
    ]
    assert fabrics[0].fabric_measured and fabrics[1].fabric_type == "roce"
    assert [c.name for c in cli.find_clusters("train-job")] == ["train-job"]
    assert cli.find_clusters("cb-b200-exp42") == []
    assert {p["name"] for p in cli.list_pods()} == {"train-job", "eval-pool", "dev-box"}


def test_lium_provider_nodes_and_destroy_guard(fake_lium: FakeLium) -> None:
    fake_lium.script(
        clusters_show=[ok("clusters_show_b200x4.json")],
        clusters_rm=[ok("clusters_rm_ok.json")],
    )
    adopted = LiumProvider(LiumCli(), "cb-b200-exp42", created_by_harness=False)
    assert adopted.created_by_harness is False
    assert [n.ssh_target for n in adopted.nodes()][:2] == [
        "root@203.0.113.10",
        "root@203.0.113.11",
    ]
    with pytest.raises(ProviderError, match="did not create"):
        adopted.destroy()
    assert all(c["key"] != "clusters rm" for c in fake_lium.calls())
    rented = LiumProvider(LiumCli(), "4c1f9a2e", created_by_harness=True)
    rented.destroy()
    assert fake_lium.calls()[-1]["key"] == "clusters rm"
