# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster providers: where the node list comes from, and who may remove it.

* ``ExistingClusterProvider`` - a cluster the harness did not create, described
  by a hosts file, a comma list of control (wg0) IPs, or ``auto`` discovery on
  the WireGuard subnet. ``destroy()`` always refuses.
* ``LiumCli`` - a thin adapter around the ``lium`` binary.
* ``LiumProvider`` - a Lium multi-node rental, used by the laptop controller.

The ``lium`` syntax below was verified against the source of the ``lium.io``
0.41.1 wheel (``lium/cli/clusters/{command,display}.py``, ``lium/cli/utils.py``,
``lium/cli/interactive.py``, ``lium/sdk/{models,client,config}.py``) and the
reference at https://docs.lium.io/developers/cli/reference/clusters:

* ``lium clusters up FABRIC --nodes N --name NAME [--template_id ID]
  [--ttl 12h] [--wait] [--timeout S] [--no-backup] --yes --format json``;
  FABRIC is a fabric id, a unique id prefix, or a row of the last listing.
* ``lium clusters show|rm CLUSTER`` (a cluster id, a unique id prefix, or the
  pod name every member shares), ``lium clusters ps|list`` and ``lium ps``;
  each takes ``--format json``.
* Exit codes: 0 ok, 1 general, 2 configuration or a refused prompt, 3 API,
  4 SSH, 5 not found, 6 permission. ``LIUM_NONINTERACTIVE=1`` turns every
  prompt into exit 2; ``LIUM_OUTPUT=json`` makes every failure one
  ``{"ok": false, "error": {"code", "message", "hint", "exit_code"}}`` line on
  stderr. A failed ``clusters up`` may leave nodes rented and billing; lium's
  own hint is "Do not re-run 'lium clusters up'", so it is never retried.
* ``LIUM_API_KEY`` is read from the inherited environment; it never goes on
  argv. Every argv this module runs is checked for secret values first.
"""

from __future__ import annotations

import configparser
import functools
import ipaddress
import json
import logging
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol, Sequence

from benchmarks.cluster_bench.adapters.executors import SECRET_ENV_VARS, Redactor
from benchmarks.cluster_bench.ports import CommandResult, NodeAddress

logger = logging.getLogger(__name__)

LIUM_VERIFIED_VERSION = "0.41.1"

# Secret variables: never on argv. lium also reads LIUM_API_API_KEY and
# LIUM_SESSION_TOKEN (lium/sdk/config.py, lium/cli/utils.py).
GUARDED_SECRET_VARS = (*SECRET_ENV_VARS, "LIUM_API_API_KEY", "LIUM_SESSION_TOKEN")
# Variables lium never needs; they are dropped from its environment.
NOT_FOR_LIUM_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_API_KEY")

WG_PREFIX = "10.42.0."  # Lium overlay: 10.42.0.<rank+1> on wg0
AUTO_PROBE_HOSTS = tuple(f"{WG_PREFIX}{i}" for i in range(1, 17))
LOCAL_TARGET = "-"  # hosts file: "this host" for node0, "root@<address>" otherwise

TIMEOUT_RC = 124
MISSING_BINARY_RC = 127

EXIT_GENERAL = 1
EXIT_CONFIG = 2
EXIT_API = 3
EXIT_SSH = 4
EXIT_NOT_FOUND = 5
EXIT_PERMISSION = 6

# Error codes after which nodes are, or may be, rented and billing
# (lium/cli/clusters/command.py, lium/cli/utils.py "rent_outcome_unknown").
BILLING_UNCERTAIN_CODES = frozenset(
    {
        "cluster_rented_not_listed",
        "cluster_rent_uncertain",
        "cluster_ttl_not_scheduled",
        "cluster_not_ready",
        "cluster_member_failed",
        "rent_outcome_unknown",
    }
)
AUTH_CODES = frozenset({"invalid_api_key", "session_required"})

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# UNVERIFIED: lium documents no pod-name rule; this conservative pattern is ours.
_POD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_SSH_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,31}$")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$")
_SSH_TARGET_RE = re.compile(r"^[^\s@-][^\s]*$")
_OPTION_RE = re.compile(r"^[a-z_]+=")
_IFACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,14}$|^auto$")
_IP_ADDR_RE = re.compile(
    r"^\d+:\s+(?P<dev>\S+)\s+inet\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})/\d{1,2}\b"
)


class ProviderError(RuntimeError):
    """A provider cannot produce, describe or remove the requested cluster."""


# --------------------------------------------------------------------------
# Local command runner (injectable; tests substitute a fake)
# --------------------------------------------------------------------------


class ArgvRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        input_data: bytes | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        ...


def _decode(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


def run_argv(
    argv: Sequence[str],
    *,
    input_data: bytes | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run ``argv`` on this host and capture its output; never raises on exit != 0.

    A missing executable yields rc 127 and a timeout rc 124 with ``[timeout]``
    appended to stderr (the conventions of ``adapters.executors``). Without
    ``input_data`` stdin is ``/dev/null`` so ssh never reads the terminal.
    """
    display = Redactor.from_env()(shlex.join(argv))
    stdin_kwargs: dict[str, Any] = (
        {"stdin": subprocess.DEVNULL} if input_data is None else {"input": input_data}
    )
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            timeout=timeout,
            env=None if env is None else dict(env),
            check=False,
            **stdin_kwargs,
        )
    except FileNotFoundError as exc:
        return CommandResult("local", display, MISSING_BINARY_RC, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stderr = _decode(exc.stderr) + "\n[timeout]"
        return CommandResult("local", display, TIMEOUT_RC, _decode(exc.stdout), stderr)
    return CommandResult(
        "local", display, proc.returncode, _decode(proc.stdout), _decode(proc.stderr)
    )


def secret_values(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Values of the guarded secret variables (6+ characters) in ``env``."""
    source = os.environ if env is None else env
    values = (source.get(name, "") for name in GUARDED_SECRET_VARS)
    return tuple(v for v in values if v and len(v) >= 6)


def assert_no_secrets(
    argv: Sequence[str], env: Mapping[str, str] | None = None
) -> None:
    """Raise ``ValueError`` when a secret value from ``env`` appears in ``argv``."""
    secrets = secret_values(env)
    for arg in argv:
        if any(secret in arg for secret in secrets):
            raise ValueError(
                "refusing to run a command with a secret on its command line; "
                "secrets travel through stdin or the inherited environment only"
            )


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


# --------------------------------------------------------------------------
# Existing clusters (hosts file, IP list, auto discovery)
# --------------------------------------------------------------------------


def parse_ip_addr(text: str) -> tuple[str, ...]:
    """IPv4 addresses from ``ip -4 -o addr show`` output, in listing order."""
    found: list[str] = []
    for line in text.splitlines():
        match = _IP_ADDR_RE.match(line.strip())
        if match:
            found.append(match["ip"])
    return tuple(found)


def local_ipv4_addresses(runner: ArgvRunner | None = None) -> tuple[str, ...]:
    """IPv4 addresses of this host's interfaces (``ip -4 -o addr show``)."""
    res = (runner or run_argv)(["ip", "-4", "-o", "addr", "show"], timeout=15)
    if res.rc != 0:
        raise ProviderError(
            f"`ip -4 -o addr show` failed (exit {res.rc}): {_tail(res.stderr, 500)}"
        )
    addresses = parse_ip_addr(res.stdout)
    if not addresses:
        raise ProviderError(
            "no IPv4 address found in `ip -4 -o addr show` output:\n"
            + _tail(res.stdout)
        )
    return addresses


def ssh_probe(ip: str, runner: ArgvRunner | None = None, *, user: str = "root") -> bool:
    """True when ``user@ip`` accepts a non-interactive SSH login.

    ``StrictHostKeyChecking=accept-new`` is added to the probe the brief
    specifies: on a fresh cluster the peers' host keys are unknown, and with
    ``BatchMode`` an unknown key would fail the login. UNVERIFIED whether the
    Lium cluster image pre-trusts its peers' host keys.
    """
    argv = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{ip}",
        "true",
    ]  # fmt: skip
    return (runner or run_argv)(argv, timeout=20).rc == 0


def _check_ipv4(text: str, where: str) -> None:
    try:
        ipaddress.IPv4Address(text)
    except ValueError as exc:
        raise ProviderError(f"{where}: {text!r} is not an IPv4 address") from exc


def _check_unique(nodes: Sequence[NodeAddress], where: str) -> None:
    for label, values in (
        ("name", [n.name for n in nodes]),
        ("address", [n.address for n in nodes]),
    ):
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise ProviderError(f"{where}: duplicate node {label}(s) {dupes}")


HOSTS_OPTIONS = ("identity", "iface", "known_hosts")


def parse_hosts_text(text: str, source: str = "<hosts>") -> list[NodeAddress]:
    """Parse ``name address [ssh_target] [ssh_port] [key=value ...]`` lines.

    ``#`` starts a comment; blank lines are ignored. The first host must be
    ``node0`` (the host running the harness); it is this host when its
    ssh_target is ``-`` or omitted. On any other line ``-`` or an omitted
    target means ``root@<address>``. ``address`` is the node's control-plane
    IPv4 (wg0 on Lium). ``identity`` is the private key node0 uses for that
    host; ``iface`` its control-plane NIC (default: the run's
    ``--control-interface``, "auto" = the NIC that holds ``address``);
    ``known_hosts`` the file node0 checks that host's key against (default:
    ssh's own).
    """
    nodes: list[NodeAddress] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{source}:{lineno}"
        tokens = line.split()
        fields = [t for t in tokens if not _OPTION_RE.match(t)]
        options: dict[str, str] = {}
        for token in tokens:
            if not _OPTION_RE.match(token):
                continue
            key, _, value = token.partition("=")
            if key not in HOSTS_OPTIONS or not value:
                raise ProviderError(
                    f"{where}: unknown option {token!r} (known: "
                    f"{', '.join(k + '=' for k in HOSTS_OPTIONS)})"
                )
            options[key] = value
        if not 2 <= len(fields) <= 4:
            raise ProviderError(
                f"{where}: expected 'name address [ssh_target] [ssh_port] "
                f"[identity=PATH] [iface=NIC] [known_hosts=PATH]', got {raw.strip()!r}"
            )
        name, address = fields[0], fields[1]
        target = fields[2] if len(fields) >= 3 else LOCAL_TARGET
        port_text = fields[3] if len(fields) == 4 else None
        if not _NAME_RE.match(name):
            raise ProviderError(f"{where}: invalid node name {name!r}")
        _check_ipv4(address, where)
        first = not nodes
        if first and name != "node0":
            raise ProviderError(
                f"{where}: the first host must be node0 (the host that runs the "
                f"harness), got {name!r}"
            )
        ssh_target: str | None
        if target == LOCAL_TARGET:
            if first and port_text is not None:
                raise ProviderError(
                    f"{where}: node0 is this host ('-'); it takes no ssh port"
                )
            ssh_target = None if first else f"root@{address}"
        else:
            if not _SSH_TARGET_RE.match(target):
                raise ProviderError(f"{where}: invalid ssh target {target!r}")
            ssh_target = target
        port = 22
        if port_text is not None:
            if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
                raise ProviderError(f"{where}: invalid ssh port {port_text!r}")
            port = int(port_text)
        iface = options.get("iface")
        if iface is not None and not _IFACE_RE.match(iface):
            raise ProviderError(f"{where}: invalid interface name {iface!r}")
        for key in ("identity", "known_hosts"):
            path = options.get(key)
            if path is not None and not path.startswith("/"):
                raise ProviderError(
                    f"{where}: {key}= must be an absolute path on node0, got {path!r}"
                )
        nodes.append(
            NodeAddress(
                name,
                address,
                ssh_target,
                port,
                identity=options.get("identity"),
                control_interface=iface,
                known_hosts=options.get("known_hosts"),
            )
        )
    if not nodes:
        raise ProviderError(f"{source}: no hosts listed")
    _check_unique(nodes, source)
    return nodes


def _peer_nodes(node0_ip: str, peers: Sequence[str]) -> list[NodeAddress]:
    nodes = [NodeAddress("node0", node0_ip, None)]
    nodes.extend(
        NodeAddress(f"node{i}", ip, f"root@{ip}") for i, ip in enumerate(peers, start=1)
    )
    return nodes


class ExistingClusterProvider:
    """A cluster that already exists; the harness never removes it."""

    def __init__(self, nodes: Sequence[NodeAddress]):
        if not nodes:
            raise ProviderError("an existing cluster needs at least one node")
        _check_unique(nodes, "cluster")
        local = [i for i, n in enumerate(nodes) if n.ssh_target is None]
        if local and local != [0]:
            raise ProviderError("only the first node (node0) may be this host")
        self._nodes = tuple(nodes)

    @property
    def created_by_harness(self) -> bool:
        return False

    def nodes(self) -> list[NodeAddress]:
        return list(self._nodes)

    def destroy(self) -> None:
        raise ProviderError(
            "refusing to destroy an existing cluster: the harness did not create it"
        )

    @classmethod
    def from_hosts_file(cls, path: Path) -> ExistingClusterProvider:
        """Nodes from a hosts file (see :func:`parse_hosts_text`)."""
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        return cls(parse_hosts_text(text, str(path)))

    @classmethod
    def from_ip_list(
        cls,
        ips: str | Sequence[str],
        *,
        local_addresses: Iterable[str] | None = None,
        runner: ArgvRunner | None = None,
    ) -> ExistingClusterProvider:
        """Nodes from control (wg0) IPs; the one that is local becomes node0.

        ``ips`` is a comma-separated string or a sequence of addresses. Peers
        keep their order and are reached as ``root@<ip>``.
        """
        raw = ips.split(",") if isinstance(ips, str) else list(ips)
        items = [str(ip).strip() for ip in raw if str(ip).strip()]
        if not items:
            raise ProviderError("empty node IP list")
        for ip in items:
            _check_ipv4(ip, "node list")
        dupes = sorted({ip for ip in items if items.count(ip) > 1})
        if dupes:
            raise ProviderError(f"node list: duplicate address(es) {dupes}")
        local = set(
            local_ipv4_addresses(runner) if local_addresses is None else local_addresses
        )
        mine = [ip for ip in items if ip in local]
        if len(mine) != 1:
            raise ProviderError(
                f"exactly one of {items} must be an address of this host (node0); "
                f"this host has {sorted(local)}"
            )
        return cls(_peer_nodes(mine[0], [ip for ip in items if ip != mine[0]]))

    @classmethod
    def auto(
        cls,
        expect_nodes: int,
        *,
        candidates: Sequence[str] = AUTO_PROBE_HOSTS,
        local_addresses: Iterable[str] | None = None,
        probe: Callable[[str], bool] | None = None,
        runner: ArgvRunner | None = None,
        max_workers: int = 16,
    ) -> ExistingClusterProvider:
        """Discover peers on the wg0 subnet by probing root SSH concurrently.

        This host's address among ``candidates`` becomes node0; every other
        candidate is probed with ``ssh -o BatchMode=yes -o ConnectTimeout=3
        root@<ip> true``. Exactly ``expect_nodes`` nodes must be found.
        """
        if expect_nodes < 1:
            raise ValueError("expect_nodes must be >= 1")
        local = set(
            local_ipv4_addresses(runner) if local_addresses is None else local_addresses
        )
        mine = [ip for ip in candidates if ip in local]
        if len(mine) != 1:
            raise ProviderError(
                f"auto discovery needs exactly one local address among "
                f"{candidates[0]}..{candidates[-1]}; this host has {sorted(local)}"
            )
        others = [ip for ip in candidates if ip != mine[0]]
        check = probe or functools.partial(ssh_probe, runner=runner)
        if others:
            workers = min(max_workers, len(others))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                reachable = dict(zip(others, pool.map(check, others)))
        else:
            reachable = {}
        peers = [ip for ip in others if reachable[ip]]
        found = [mine[0], *peers]
        if len(found) != expect_nodes:
            raise ProviderError(
                f"auto discovery found {len(found)} node(s) {found}, expected "
                f"{expect_nodes}; pass the wg0 IPs explicitly or a hosts file"
            )
        logger.info("auto discovery: node0=%s peers=%s", mine[0], peers)
        return cls(_peer_nodes(mine[0], peers))


# --------------------------------------------------------------------------
# lium CLI adapter
# --------------------------------------------------------------------------


class LiumCliError(RuntimeError):
    """``lium`` failed; ``retryable`` says whether re-running the call is safe."""

    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        command: str = "",
        rc: int | None = None,
        stdout: str = "",
        stderr: str = "",
        code: str | None = None,
        hint: str | None = None,
    ):
        redact = Redactor.from_env()
        parts = [message]
        if code:
            parts.append(f"  lium error code: {code}")
        if hint:
            parts.append(f"  hint: {hint}")
        if command:
            parts.append(f"  command: {command}")
        if rc is not None:
            parts.append(f"  exit code: {rc}")
        if stdout.strip():
            parts.append(f"  stdout: {_tail(redact(stdout))}")
        if stderr.strip():
            parts.append(f"  stderr: {_tail(redact(stderr))}")
        super().__init__("\n".join(parts))
        self.summary = f"{code or 'exit ' + str(rc)}: {message.splitlines()[0]}"
        self.command = command
        self.rc = rc
        self.stdout = stdout
        self.stderr = stderr
        self.code = code
        self.hint = hint


class LiumRetryableError(LiumCliError):
    retryable = True


class LiumFatalError(LiumCliError):
    retryable = False


class LiumApiError(LiumRetryableError):
    """Exit 3: the API refused or failed the call (rate limit, server error)."""


class LiumSshError(LiumRetryableError):
    """Exit 4: lium could not reach a pod over SSH."""


class LiumTimeoutError(LiumRetryableError):
    """The lium process did not finish within our timeout."""


class LiumGeneralError(LiumFatalError):
    """Exit 1 (or an unknown exit code)."""


class LiumConfigError(LiumFatalError):
    """Exit 2: bad arguments, no API key, or a prompt that could not be shown."""


class LiumAuthError(LiumFatalError):
    """Exit 3 with an authentication code: the key or session was refused."""


class LiumNotFoundError(LiumFatalError):
    """Exit 5 (or API "not_found"): no such pod, cluster or fabric."""


class LiumPermissionError(LiumFatalError):
    """Exit 6: the account may not do this (balance, budget, scope)."""


class LiumSshHostKeyError(LiumFatalError):
    """Exit 4 ``ssh_host_key_changed``: lium refuses to retry blindly."""


class LiumBillingUncertainError(LiumFatalError):
    """The nodes are, or may be, rented and billing; never re-run the rent."""


class LiumParseError(LiumFatalError):
    """lium succeeded but printed something this adapter cannot read."""


class LiumNotInstalledError(LiumFatalError):
    """The lium binary was not found."""


def classify_failure(rc: int, code: str | None) -> type[LiumCliError]:
    """Map a lium exit code (and JSON error code) to an error class."""
    if code in BILLING_UNCERTAIN_CODES:
        return LiumBillingUncertainError
    if rc == EXIT_API:
        if code in AUTH_CODES:
            return LiumAuthError
        if code == "not_found":
            return LiumNotFoundError
        return LiumApiError
    if rc == EXIT_SSH:
        return LiumSshHostKeyError if code == "ssh_host_key_changed" else LiumSshError
    by_rc: dict[int, type[LiumCliError]] = {
        EXIT_CONFIG: LiumConfigError,
        EXIT_NOT_FOUND: LiumNotFoundError,
        EXIT_PERMISSION: LiumPermissionError,
    }
    return by_rc.get(rc, LiumGeneralError)


def parse_error_envelope(stderr: str) -> tuple[str | None, str | None, str | None]:
    """``(code, message, hint)`` of lium's JSON error envelope on stderr.

    lium prints the envelope as one ``sort_keys`` JSON line, possibly after
    narration lines, so stderr is scanned from the end.
    """
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
            err = obj["error"]
            return (
                _opt_str(err.get("code")),
                _opt_str(err.get("message")),
                _opt_str(err.get("hint")),
            )
    return None, None, None


def _stdout_failure(stdout: str) -> tuple[str | None, str | None]:
    """``(code, message)`` from a failure lium reports as JSON on stdout.

    ``clusters up`` prints the cluster with ``ok: false`` when the TTL could
    not be scheduled, and ``clusters rm`` prints ``ok: false`` when a member
    could not be removed; both exit 3 without a stderr envelope.
    """
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict) or obj.get("ok") is not False:
        return None, None
    message = obj.get("error")
    code = "cluster_ttl_not_scheduled" if "unscheduled" in obj else None
    return code, message if isinstance(message, str) else None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_ssh_cmd(ssh_cmd: str) -> tuple[str, int]:
    """``("user@host", port)`` from lium's ``ssh <user>@<host> [-p <port>]``.

    Only that shape is accepted (as ``lium.sdk.client.ssh_target`` does), so
    an unexpected value is reported instead of being handed to ssh.
    """
    try:
        tokens = shlex.split(ssh_cmd)
    except ValueError as exc:
        raise ValueError(f"unexpected ssh command {ssh_cmd!r}: {exc}") from exc
    if len(tokens) not in (2, 4) or tokens[0] != "ssh":
        raise ValueError(f"unexpected ssh command {ssh_cmd!r}")
    user, sep, host = tokens[1].partition("@")
    host_ok = bool(_HOSTNAME_RE.match(host))
    if not host_ok:
        try:
            ipaddress.ip_address(host)
            host_ok = "%" not in host
        except ValueError:
            host_ok = False
    if not sep or not _SSH_USER_RE.match(user) or not host_ok:
        raise ValueError(f"unexpected ssh command {ssh_cmd!r}")
    port = 22
    if len(tokens) == 4:
        if tokens[2] != "-p" or not tokens[3].isdigit():
            raise ValueError(f"unexpected ssh command {ssh_cmd!r}")
        port = int(tokens[3])
        if not 1 <= port <= 65535:
            raise ValueError(f"unexpected ssh port in {ssh_cmd!r}")
    return f"{user}@{host}", port


@dataclass(frozen=True)
class LiumNode:
    """One member of a Lium cluster (``lium clusters show --format json``)."""

    name: str  # logical node name: node<rank>
    rank: int
    pod_id: str
    huid: str
    pod_name: str
    status: str
    ssh_target: str | None  # public "user@host" for laptop access; None until ready
    ssh_port: int
    control_ip: str | None  # wg0 overlay address 10.42.0.<rank+1>
    gpu_type: str | None
    gpu_count: int | None

    @property
    def ready(self) -> bool:
        return (
            self.status.upper() == "RUNNING"
            and self.ssh_target is not None
            and self.control_ip is not None
        )


@dataclass(frozen=True)
class LiumCluster:
    """A Lium cluster rental; ``nodes`` holds one member per rank, rank order."""

    id: str
    name: str | None
    status: str
    price_per_hour: float | None
    nodes: tuple[LiumNode, ...]
    removal_scheduled_at: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.nodes) and all(n.ready for n in self.nodes)


@dataclass(frozen=True)
class LiumFabric:
    """A fabric with free nodes (``lium clusters list --format json``)."""

    fabric_id: str
    fabric_type: str  # "infiniband" | "roce"
    link_rate: str | None
    fabric_measured: bool
    node_count: int
    free_count: int
    gpu_type: str
    gpus_per_node: int
    price_per_node_hour: float


def _parse_member(pod: Any, raw: str) -> LiumNode:
    if not isinstance(pod, Mapping):
        raise LiumParseError("a cluster member is not a JSON object", stdout=raw)
    rank = _opt_int(pod.get("node_rank"))
    if rank is None or rank < 0:
        raise LiumParseError(
            f"cluster member {pod.get('huid')!r} has no node_rank", stdout=raw
        )
    target: str | None = None
    port = 22
    ssh_cmd = pod.get("ssh_cmd")
    if ssh_cmd:
        try:
            target, port = parse_ssh_cmd(str(ssh_cmd))
        except ValueError as exc:
            raise LiumParseError(
                f"cluster member rank {rank}: {exc}", stdout=raw
            ) from exc
    overlay = _opt_str(pod.get("overlay_ip"))
    if overlay is not None:
        try:
            ipaddress.IPv4Address(overlay)
        except ValueError as exc:
            raise LiumParseError(
                f"cluster member rank {rank}: overlay_ip {overlay!r} is not IPv4",
                stdout=raw,
            ) from exc
    return LiumNode(
        name=f"node{rank}",
        rank=rank,
        pod_id=str(pod.get("id") or ""),
        huid=str(pod.get("huid") or ""),
        pod_name=str(pod.get("name") or ""),
        status=str(pod.get("status") or "UNKNOWN"),
        ssh_target=target,
        ssh_port=port,
        control_ip=overlay,
        gpu_type=_opt_str(pod.get("gpu_type")),
        gpu_count=_opt_int(pod.get("gpu_count")),
    )


def _one_member_per_rank(
    members: Sequence[LiumNode], cluster_id: str, strict: bool
) -> tuple[LiumNode, ...]:
    """While a node is replaced its rank holds two pods; keep the RUNNING one."""
    by_rank: dict[int, list[LiumNode]] = {}
    for member in members:
        by_rank.setdefault(member.rank, []).append(member)
    chosen: list[LiumNode] = []
    for rank in sorted(by_rank):
        pods = by_rank[rank]
        running = [p for p in pods if p.status.upper() == "RUNNING"]
        if len(pods) == 1:
            chosen.append(pods[0])
        elif len(running) == 1:
            chosen.append(running[0])
        elif strict:
            raise ProviderError(
                f"cluster {cluster_id}: rank {rank} is held by {len(pods)} pods "
                f"({', '.join(p.huid for p in pods)}) and not exactly one is RUNNING; "
                "a member is being replaced - retry later"
            )
        else:
            chosen.append(pods[0])
    ranks = [n.rank for n in chosen]
    if strict and ranks != list(range(len(chosen))):
        raise ProviderError(
            f"cluster {cluster_id}: member ranks {ranks} are not 0..{len(chosen) - 1}"
        )
    return tuple(chosen)


def parse_cluster(obj: Any, raw: str, *, strict: bool = True) -> LiumCluster:
    """A :class:`LiumCluster` from lium's ``cluster_to_dict`` JSON."""
    if (
        not isinstance(obj, Mapping)
        or not isinstance(obj.get("id"), str)
        or not isinstance(obj.get("pods"), list)
    ):
        raise LiumParseError(
            "expected a cluster object with 'id' and 'pods'", stdout=raw
        )
    members = [_parse_member(p, raw) for p in obj["pods"]]
    return LiumCluster(
        id=obj["id"],
        name=_opt_str(obj.get("name")),
        status=str(obj.get("status") or "UNKNOWN"),
        price_per_hour=_opt_float(obj.get("price_per_hour")),
        nodes=_one_member_per_rank(members, obj["id"], strict),
        removal_scheduled_at=_opt_str(obj.get("removal_scheduled_at")),
    )


def parse_fabric(obj: Any, raw: str) -> LiumFabric:
    """A :class:`LiumFabric` from lium's ``offer_to_dict`` JSON."""
    if not isinstance(obj, Mapping) or not isinstance(obj.get("fabric_id"), str):
        raise LiumParseError("expected a fabric object with 'fabric_id'", stdout=raw)
    try:
        return LiumFabric(
            fabric_id=obj["fabric_id"],
            fabric_type=str(obj.get("fabric_type") or "infiniband"),
            link_rate=_opt_str(obj.get("link_rate")),
            fabric_measured=bool(obj.get("fabric_measured")),
            node_count=int(obj.get("node_count") or 0),
            free_count=int(obj.get("free_count") or 0),
            gpu_type=str(obj.get("gpu_type") or ""),
            gpus_per_node=int(obj.get("gpus_per_node") or 0),
            price_per_node_hour=float(obj.get("price_per_node_hour") or 0.0),
        )
    except (TypeError, ValueError) as exc:
        raise LiumParseError(f"bad fabric field: {exc}", stdout=raw) from exc


def node_addresses(cluster: LiumCluster) -> list[NodeAddress]:
    """``NodeAddress`` per member: wg0 control IP plus the public SSH endpoint."""
    addresses: list[NodeAddress] = []
    for node in cluster.nodes:
        if not node.ready or node.control_ip is None or node.ssh_target is None:
            raise ProviderError(
                f"cluster {cluster.id} is not ready: {node.name} is {node.status} "
                f"(ssh={node.ssh_target}, wg0={node.control_ip})"
            )
        addresses.append(
            NodeAddress(node.name, node.control_ip, node.ssh_target, node.ssh_port)
        )
    if not addresses:
        raise ProviderError(f"cluster {cluster.id} has no members yet")
    return addresses


def format_ttl(hours: float) -> str:
    """lium's TTL syntax (``^\\d+(\\.\\d+)?(m|h|d)$``), e.g. ``12h`` or ``1.5h``."""
    if not 0 < hours <= 24 * 30:
        raise ValueError(f"TTL must be within (0, 720] hours, got {hours}")
    return f"{round(hours, 2):g}h"


def resolve_lium_ssh_key(
    env: Mapping[str, str] | None = None, home: Path | None = None
) -> Path | None:
    """The private key lium logs in with, in lium's own order.

    ``LIUM_SSH_KEY_PATH``, then ``[ssh] key_path`` in ``~/.lium/config.ini``
    (``lium.sdk.config.resolve_ssh_key_path``). ``None`` leaves the choice to
    ssh, whose defaults (``id_ed25519``, ``id_rsa``, ``id_ecdsa``) match
    lium's fallbacks.
    """
    source = os.environ if env is None else env
    base = home or Path.home()
    configured = source.get("LIUM_SSH_KEY_PATH")
    if configured:
        return _expand_home(configured, base)
    config_file = base / ".lium" / "config.ini"
    if not config_file.is_file():
        return None
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(config_file, encoding="utf-8")
    except configparser.Error as exc:
        raise ProviderError(f"cannot read {config_file}: {exc}") from exc
    value = parser.get("ssh", "key_path", fallback=None)
    return _expand_home(value, base) if value else None


def _expand_home(value: str, home: Path) -> Path:
    """``~`` against ``home`` (the injected one in tests), not the process $HOME."""
    if value == "~" or value.startswith("~/"):
        return home / value[2:] if value != "~" else home
    return Path(value).expanduser()


@dataclass(frozen=True)
class LiumResult:
    command: str  # redacted display of the argv
    rc: int
    stdout: str
    stderr: str
    data: Any = None  # parsed JSON when requested


class LiumCli:
    """Thin adapter around the ``lium`` binary (syntax: see module docstring).

    Every call runs with ``LIUM_NONINTERACTIVE=1``, ``LIUM_OUTPUT=json`` and
    ``LIUM_SELF_UPDATE_DISABLE=1`` added to the inherited environment, so
    ``LIUM_API_KEY`` comes from the caller's environment and never from argv.
    """

    def __init__(
        self,
        binary: str = "lium",
        *,
        runner: ArgvRunner | None = None,
        env: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = 300.0,
        read_retries: int = 2,
        backoff_s: float = 5.0,
    ):
        self.binary = binary
        self._runner = runner or run_argv
        self._env = env
        self._sleep = sleep
        self.timeout_s = timeout_s
        self.read_retries = read_retries
        self.backoff_s = backoff_s

    def _source_env(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def environment(self) -> dict[str, str]:
        """The environment lium runs with (inherited, minus unrelated secrets)."""
        env = {
            k: v for k, v in self._source_env().items() if k not in NOT_FOR_LIUM_VARS
        }
        env["LIUM_NONINTERACTIVE"] = "1"
        env["LIUM_OUTPUT"] = "json"
        env["LIUM_SELF_UPDATE_DISABLE"] = "1"
        return env

    def run(
        self,
        args: Sequence[str],
        json_out: bool = False,
        *,
        timeout: float | None = None,
        retries: int = 0,
    ) -> LiumResult:
        """Run ``lium <args>``; raise a :class:`LiumCliError` subclass on failure.

        ``json_out`` appends ``--format json`` and parses stdout; output that
        is not JSON raises :class:`LiumParseError` carrying the raw output.
        Only :class:`LiumRetryableError` failures are retried, ``retries``
        times with exponential backoff.
        """
        argv = [self.binary, *args]
        if json_out and "--format" not in args and "--json" not in args:
            argv += ["--format", "json"]
        assert_no_secrets(argv, self._source_env())
        env = self.environment()
        attempt = 0
        while True:
            res = self._runner(argv, timeout=timeout or self.timeout_s, env=env)
            try:
                return self._interpret(res, json_out)
            except LiumRetryableError as exc:
                if attempt >= retries:
                    raise
                delay = self.backoff_s * (2**attempt)
                attempt += 1
                logger.warning(
                    "%s failed (%s); retry %d/%d in %.0fs",
                    res.command,
                    exc.summary,
                    attempt,
                    retries,
                    delay,
                )
                self._sleep(delay)

    def _interpret(self, res: CommandResult, json_out: bool) -> LiumResult:
        if res.rc != 0:
            raise self._failure(res)
        if not json_out:
            return LiumResult(res.command, res.rc, res.stdout, res.stderr)
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError as exc:
            raise LiumParseError(
                f"lium printed output that is not JSON ({exc})",
                command=res.command,
                rc=res.rc,
                stdout=res.stdout,
                stderr=res.stderr,
            ) from exc
        return LiumResult(res.command, res.rc, res.stdout, res.stderr, data)

    def _failure(self, res: CommandResult) -> LiumCliError:
        context = {
            "command": res.command,
            "rc": res.rc,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
        if res.rc == MISSING_BINARY_RC and not res.stdout:
            return LiumNotInstalledError(
                f"{self.binary!r} not found: install the lium CLI "
                "(`pip install lium.io`) and export LIUM_API_KEY",
                **context,
            )
        if res.rc == TIMEOUT_RC and res.stderr.endswith("[timeout]"):
            return LiumTimeoutError("lium did not finish in time", **context)
        code, message, hint = parse_error_envelope(res.stderr)
        if code is None and message is None:
            code, message = _stdout_failure(res.stdout)
        cls = classify_failure(res.rc, code)
        return cls(message or "lium failed", code=code, hint=hint, **context)

    # -- clusters -----------------------------------------------------------

    def cluster_up(
        self,
        fabric: str,
        nodes: int,
        name: str,
        ttl_hours: float,
        template: str | None = None,
        *,
        wait_timeout_s: int = 1500,
        backups: bool = False,
    ) -> LiumCluster:
        """Rent ``nodes`` whole nodes of ``fabric`` as one cluster; never retried.

        Any failure may leave nodes rented and billing, so callers must look
        the cluster up by name (:meth:`find_clusters`) before trying again.
        lium backs member volumes up by default; ``backups=False`` passes
        ``--no-backup`` so backup traffic cannot perturb the measurements.
        """
        if nodes < 2:
            raise ValueError("a lium cluster needs at least 2 nodes")
        if not _POD_NAME_RE.match(name):
            raise ValueError(f"invalid cluster name {name!r}")
        args = [
            "clusters", "up", fabric,
            "--nodes", str(nodes),
            "--name", name,
            "--ttl", format_ttl(ttl_hours),
            "--wait",
            "--timeout", str(int(wait_timeout_s)),
            "--yes",
        ]  # fmt: skip
        if template:
            args += ["--template_id", template]
        if not backups:
            args.append("--no-backup")
        res = self.run(args, json_out=True, timeout=wait_timeout_s + 300, retries=0)
        return parse_cluster(res.data, res.stdout)

    def cluster_info(self, cluster: str) -> LiumCluster:
        """``lium clusters show CLUSTER --format json``."""
        res = self.run(
            ["clusters", "show", cluster], json_out=True, retries=self.read_retries
        )
        return parse_cluster(res.data, res.stdout)

    def cluster_show(self, cluster: str) -> list[LiumNode]:
        """Members of ``cluster`` (id, id prefix or shared pod name), rank order."""
        return list(self.cluster_info(cluster).nodes)

    def cluster_rm(self, cluster: str) -> list[dict[str, Any]]:
        """``lium clusters rm CLUSTER --yes``; re-running retries failed members."""
        res = self.run(
            ["clusters", "rm", cluster, "--yes"],
            json_out=True,
            retries=self.read_retries,
        )
        data = res.data
        if (
            not isinstance(data, Mapping)
            or data.get("ok") is not True
            or not isinstance(data.get("results"), list)
        ):
            raise LiumParseError(
                "unexpected `clusters rm` answer",
                command=res.command,
                rc=res.rc,
                stdout=res.stdout,
            )
        return [dict(r) for r in data["results"] if isinstance(r, Mapping)]

    def list_clusters(self) -> list[LiumCluster]:
        """Your clusters (``lium clusters ps --format json``)."""
        res = self.run(["clusters", "ps"], json_out=True, retries=self.read_retries)
        if not isinstance(res.data, list):
            raise LiumParseError(
                "expected a JSON list of clusters",
                command=res.command,
                stdout=res.stdout,
            )
        return [parse_cluster(c, res.stdout, strict=False) for c in res.data]

    def find_clusters(self, name: str) -> list[LiumCluster]:
        """Clusters whose shared pod name (or id) is ``name``."""
        return [c for c in self.list_clusters() if name in (c.name, c.id)]

    def list_fabrics(self) -> list[LiumFabric]:
        """Fabrics with free nodes (``lium clusters list --format json``)."""
        res = self.run(["clusters", "list"], json_out=True, retries=self.read_retries)
        if not isinstance(res.data, list):
            raise LiumParseError(
                "expected a JSON list of fabrics",
                command=res.command,
                stdout=res.stdout,
            )
        return [parse_fabric(f, res.stdout) for f in res.data]

    def list_pods(self) -> list[dict[str, Any]]:
        """Every pod of the account (``lium ps --format json``), raw rows."""
        res = self.run(["ps"], json_out=True, retries=self.read_retries)
        if not isinstance(res.data, list) or not all(
            isinstance(p, Mapping) for p in res.data
        ):
            raise LiumParseError(
                "expected a JSON list of pods", command=res.command, stdout=res.stdout
            )
        return [dict(p) for p in res.data]


class LiumProvider:
    """A Lium cluster rental (laptop side).

    ``created_by_harness`` comes from the controller state: an adopted cluster
    is never destroyed, whatever the caller asks.
    """

    def __init__(self, cli: LiumCli, cluster: str, *, created_by_harness: bool):
        self._cli = cli
        self.cluster_ref = cluster
        self._created = created_by_harness

    @property
    def created_by_harness(self) -> bool:
        return self._created

    def describe(self) -> LiumCluster:
        return self._cli.cluster_info(self.cluster_ref)

    def nodes(self) -> list[NodeAddress]:
        return node_addresses(self.describe())

    def destroy(self) -> None:
        if not self._created:
            raise ProviderError(
                f"refusing to destroy {self.cluster_ref}: the harness did not create "
                "it (adopted or existing cluster)"
            )
        results = self._cli.cluster_rm(self.cluster_ref)
        failed = [r for r in results if not r.get("success")]
        if failed:
            raise ProviderError(
                f"{len(failed)} member(s) of {self.cluster_ref} were not removed: "
                f"{failed}"
            )
