# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Machines the user brings: one SSH access per machine (``controller --machines``).

A machines file lists the cluster with node0 (the machine that runs the
harness) first. ``#`` starts a comment. Each line is one machine: how the
laptop reaches it, then optional ``key=value`` options::

    ssh -p 2222 -i ~/.ssh/lab root@203.0.113.10    control_ip=10.0.0.11
    ubuntu@gpu-b.example.com                         name=gpu-b
    ssh://root@203.0.113.12:2200                     peer_ssh=root@10.0.0.13:22

Access forms: an ``ssh`` command (only -p, -i, -l, -J, and -o with a key from
SSH_OPTION_KEYS: nothing that runs a local command, such as ProxyCommand),
``user@host[:port]`` or ``ssh://user@host[:port]``. The user defaults to root.

Options:
  name=LABEL       label shown by ``controller status`` (default: the hostname)
  control_ip=IPv4  private address the cluster talks over (default: the address
                   on a subnet the machine shares with node0)
  peer_ssh=T[:P]   how node0 reaches the machine (default: <user>@<control_ip>
                   port 22, then the laptop's own target and port)
  iface=NIC        control-plane NIC (default: the one holding control_ip)

Everything here is pure: the controller runs the scripts over ssh and feeds
their output back (controller.py).
"""

from __future__ import annotations

import ipaddress
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Iterable, Sequence

MACHINE_OPTIONS = ("name", "control_ip", "peer_ssh", "iface")
# -o keys accepted from a pasted ssh command (lower case). Port, User,
# IdentityFile and ProxyJump are folded into SshAccess fields.
SSH_OPTION_KEYS = frozenset(
    {
        "port",
        "user",
        "identityfile",
        "proxyjump",
        "identitiesonly",
        "connecttimeout",
        "serveraliveinterval",
        "serveralivecountmax",
    }
)
DEFAULT_USER = "root"
TRUST_KEY_PREFIX = "cluster_bench:"  # authorized_keys comment of a trust key
# Interfaces that never carry the cluster's control plane.
VIRTUAL_IFACE_PREFIXES = (
    "lo",
    "docker",
    "br-",
    "veth",
    "virbr",
    "cni",
    "flannel",
    "cali",
    "tunl",
    "vxlan",
    "kube",
    "cilium",
    "weave",
    "lxc",
    "podman",
)
# Preferred control-plane NICs, most preferred first (prefix match).
IFACE_PREFERENCE = ("wg", "bond", "eth", "en", "ib")
_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,31}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$")
_JUMP_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._@:,\[\]-]*$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_IFACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,14}$")
_OPTION_RE = re.compile(r"^[a-z_]+=")
_SIMPLE_VALUE_RE = re.compile(r"^[A-Za-z0-9]+$")


class MachineError(ValueError):
    pass


@dataclass(frozen=True)
class SshAccess:
    """How one ssh client reaches a machine (never a shell string)."""

    user: str
    host: str
    port: int = 22
    identity: str | None = None  # private key path on the client
    jump: str | None = None  # ProxyJump
    options: tuple[tuple[str, str], ...] = ()  # allow-listed -o options

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_options(self) -> list[str]:
        """Client options for this access (no port: ssh and scp spell it apart)."""
        out: list[str] = []
        if self.identity:
            out += ["-i", self.identity, "-o", "IdentitiesOnly=yes"]
        if self.jump:
            out += ["-J", self.jump]
        for key, value in self.options:
            out += ["-o", f"{key}={value}"]
        return out


@dataclass(frozen=True)
class MachineSpec:
    access: SshAccess
    label: str | None = None
    control_ip: str | None = None
    peer: tuple[str, int] | None = None  # (target, port) node0 uses
    iface: str | None = None
    line: str = ""  # where it came from, for messages


def _check_host(host: str, where: str) -> None:
    if _HOST_RE.match(host):
        return
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise MachineError(f"{where}: invalid host {host!r}") from None


def _check_user(user: str, where: str) -> None:
    if not _USER_RE.match(user):
        raise MachineError(f"{where}: invalid ssh user {user!r}")


def _port(text: str, where: str) -> int:
    if not text.isdigit() or not 1 <= int(text) <= 65535:
        raise MachineError(f"{where}: invalid ssh port {text!r}")
    return int(text)


def _identity(path: str, where: str, home: str | None) -> str:
    if not path or path.startswith("-") or any(c.isspace() for c in path):
        raise MachineError(f"{where}: invalid identity file {path!r}")
    if path == "~" or path.startswith("~/"):
        base = home if home is not None else os.path.expanduser("~")
        path = base.rstrip("/") + path[1:]
    return path


def _split_target(text: str, where: str) -> tuple[str | None, str, int | None]:
    """``[user@]host[:port]`` (IPv6 only as ``[addr]`` with a port)."""
    user, sep, rest = text.rpartition("@")
    if not sep:
        user, rest = "", text
    port: int | None = None
    if rest.startswith("["):
        host, _, tail = rest[1:].partition("]")
        if tail:
            if not tail.startswith(":"):
                raise MachineError(f"{where}: invalid host {rest!r}")
            port = _port(tail[1:], where)
    elif rest.count(":") == 1:
        host, _, port_text = rest.partition(":")
        port = _port(port_text, where)
    else:
        host = rest
    _check_host(host, where)
    return user or None, host, port


def _ssh_command(tokens: Sequence[str], where: str, home: str | None) -> SshAccess:
    user: str | None = None
    port: int | None = None
    identity: str | None = None
    jump: str | None = None
    options: list[tuple[str, str]] = []
    dest: str | None = None
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            if dest is not None:
                raise MachineError(
                    f"{where}: one destination per machine (a remote command after "
                    f"it is not allowed): {tok!r}"
                )
            dest = tok
            i += 1
            continue
        flag, inline = tok[:2], tok[2:]
        if flag not in ("-p", "-i", "-l", "-J", "-o") or tok.startswith("--"):
            raise MachineError(
                f"{where}: ssh flag {tok!r} is not accepted (allowed: -p -i -l -J -o)"
            )
        if inline:
            value = inline
            i += 1
        else:
            if i + 1 >= len(tokens):
                raise MachineError(f"{where}: {flag} needs a value")
            value = tokens[i + 1]
            i += 2
        if flag == "-p":
            port = _port(value, where)
        elif flag == "-i":
            identity = _identity(value, where, home)
        elif flag == "-l":
            user = value
        elif flag == "-J":
            jump = value
        else:
            key, sep, val = value.partition("=")
            if not sep:
                key, _, val = value.partition(" ")
            key, val = key.strip(), val.strip()
            lowered = key.lower()
            if lowered not in SSH_OPTION_KEYS or not val:
                raise MachineError(
                    f"{where}: ssh option {value!r} is not accepted (allowed: "
                    f"{', '.join(sorted(SSH_OPTION_KEYS))})"
                )
            if lowered == "port":
                port = _port(val, where)
            elif lowered == "user":
                user = val
            elif lowered == "identityfile":
                identity = _identity(val, where, home)
            elif lowered == "proxyjump":
                jump = val
            else:
                if not _SIMPLE_VALUE_RE.match(val):
                    raise MachineError(f"{where}: invalid value in {value!r}")
                options.append((key, val))
    if dest is None:
        raise MachineError(f"{where}: the ssh command names no destination")
    dest_user, host, dest_port = _split_target(dest, where)
    if dest_port is not None:
        raise MachineError(f"{where}: write the port as -p, not host:port")
    user = dest_user or user or DEFAULT_USER
    _check_user(user, where)
    if jump is not None and not _JUMP_RE.match(jump):
        raise MachineError(f"{where}: invalid ProxyJump {jump!r}")
    return SshAccess(user, host, port or 22, identity, jump, tuple(options))


def parse_access(
    text: str, where: str = "<machine>", home: str | None = None
) -> SshAccess:
    """One machine's access: an ssh command, ``user@host[:port]`` or ssh:// URL."""
    raw = text.strip()
    if raw.startswith("ssh://"):
        rest = raw[len("ssh://") :].rstrip("/")
        user, host, port = _split_target(rest, where)
        user = user or DEFAULT_USER
        _check_user(user, where)
        return SshAccess(user, host, port or 22)
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise MachineError(f"{where}: {exc}") from None
    if not tokens:
        raise MachineError(f"{where}: empty machine")
    if tokens[0] == "ssh":
        return _ssh_command(tokens, where, home)
    if len(tokens) != 1:
        raise MachineError(f"{where}: expected 'ssh ...', user@host[:port] or ssh://")
    user, host, port = _split_target(tokens[0], where)
    user = user or DEFAULT_USER
    _check_user(user, where)
    return SshAccess(user, host, port or 22)


def _check_ipv4(text: str, where: str) -> str:
    try:
        return str(ipaddress.IPv4Address(text))
    except ValueError:
        raise MachineError(f"{where}: {text!r} is not an IPv4 address") from None


def parse_machine(
    line: str, where: str = "<machine>", home: str | None = None
) -> MachineSpec:
    """One machines-file line (access, then key=value options)."""
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        raise MachineError(f"{where}: {exc}") from None
    options: dict[str, str] = {}
    access_tokens: list[str] = []
    flag_value = False  # the token after -p/-i/-l/-J/-o belongs to ssh
    for tok in tokens:
        if flag_value or tok in ("-p", "-i", "-l", "-J", "-o"):
            access_tokens.append(tok)
            flag_value = not flag_value and tok in ("-p", "-i", "-l", "-J", "-o")
            continue
        if _OPTION_RE.match(tok):
            key, _, value = tok.partition("=")
            if key not in MACHINE_OPTIONS or not value:
                raise MachineError(
                    f"{where}: unknown option {tok!r} (known: "
                    f"{', '.join(k + '=' for k in MACHINE_OPTIONS)})"
                )
            options[key] = value
        else:
            access_tokens.append(tok)
    access = parse_access(shlex.join(access_tokens), where, home)
    label = options.get("name")
    if label is not None and not _LABEL_RE.match(label):
        raise MachineError(f"{where}: invalid name {label!r}")
    control_ip = options.get("control_ip")
    if control_ip is not None:
        control_ip = _check_ipv4(control_ip, where)
    peer: tuple[str, int] | None = None
    if "peer_ssh" in options:
        user, host, port = _split_target(options["peer_ssh"], where)
        user = user or access.user
        _check_user(user, where)
        if user != access.user:
            raise MachineError(
                f"{where}: peer_ssh user {user!r} must be the access user "
                f"{access.user!r} (node0's key is installed for that user)"
            )
        peer = (f"{user}@{host}", port or 22)
    iface = options.get("iface")
    if iface is not None and not _IFACE_RE.match(iface):
        raise MachineError(f"{where}: invalid interface name {iface!r}")
    return MachineSpec(access, label, control_ip, peer, iface, line.strip())


def parse_machines_text(
    text: str, source: str = "<machines>", home: str | None = None
) -> list[MachineSpec]:
    """A machines file (see the module docstring); node0 first."""
    specs: list[MachineSpec] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if line:
            specs.append(parse_machine(line, f"{source}:{lineno}", home))
    if not specs:
        raise MachineError(f"{source}: no machines listed")
    check_unique(specs, source)
    return specs


def check_unique(specs: Sequence[MachineSpec], source: str = "<machines>") -> None:
    seen: set[tuple[str, int]] = set()
    for spec in specs:
        key = (spec.access.host, spec.access.port)
        if key in seen and spec.access.jump is None:
            raise MachineError(
                f"{source}: {spec.access.target} -p {key[1]} listed twice"
            )
        seen.add(key)
    labels = [s.label for s in specs if s.label]
    if len(labels) != len(set(labels)):
        raise MachineError(f"{source}: duplicate machine name in {labels}")
    ips = [s.control_ip for s in specs if s.control_ip]
    if len(ips) != len(set(ips)):
        raise MachineError(f"{source}: duplicate control_ip in {ips}")


# --------------------------------------------------------------------------
# Discovery: what each machine reports about itself
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Address:
    iface: str
    ip: str
    prefix: int

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(f"{self.ip}/{self.prefix}", strict=False)


@dataclass(frozen=True)
class MachineFacts:
    hostname: str
    uid: int | None
    addresses: tuple[Address, ...] = ()
    gpus: tuple[str, ...] = ()
    docker: str = "missing"  # ok | denied | missing
    workdir_ok: bool = False
    raw: str = field(default="", repr=False)


def discovery_script(workdir: str) -> str:
    """Prints @@ markers: hostname, uid, IPv4 addresses, GPUs, docker, workdir."""
    w = shlex.quote(workdir)
    return "\n".join(
        [
            "set -u",
            'echo "@@HOST $(hostname 2>/dev/null || echo unknown)"',
            'echo "@@UID $(id -u)"',
            "ip -o -4 addr show 2>/dev/null | awk '{print \"@@ADDR\", $2, $4}'",
            "if command -v nvidia-smi >/dev/null 2>&1; then",
            "  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null"
            " | sed 's/^/@@GPU /'",
            "fi",
            "if ! command -v docker >/dev/null 2>&1; then echo '@@DOCKER missing'",
            "elif docker info >/dev/null 2>&1; then echo '@@DOCKER ok'",
            "else echo '@@DOCKER denied'; fi",
            f"if mkdir -p {w} 2>/dev/null && [ -w {w} ]; then echo '@@WORKDIR ok';"
            " else echo '@@WORKDIR denied'; fi",
        ]
    )


def parse_discovery(stdout: str) -> MachineFacts:
    hostname, uid, docker, workdir_ok = "unknown", None, "missing", False
    addresses: list[Address] = []
    gpus: list[str] = []
    for line in stdout.splitlines():
        parts = line.strip().split(None, 1)
        if not parts or not parts[0].startswith("@@"):
            continue
        tag, rest = parts[0], (parts[1] if len(parts) > 1 else "").strip()
        if tag == "@@HOST" and rest:
            hostname = rest.split()[0]
        elif tag == "@@UID" and rest.isdigit():
            uid = int(rest)
        elif tag == "@@ADDR":
            fields = rest.split()
            if len(fields) == 2 and "/" in fields[1]:
                iface = fields[0].rstrip(":").split("@", 1)[0]
                ip, _, prefix = fields[1].partition("/")
                try:
                    ipaddress.IPv4Address(ip)
                except ValueError:
                    continue
                if prefix.isdigit() and 0 <= int(prefix) <= 32:
                    addresses.append(Address(iface, ip, int(prefix)))
        elif tag == "@@GPU" and rest:
            gpus.append(rest)
        elif tag == "@@DOCKER" and rest in ("ok", "denied", "missing"):
            docker = rest
        elif tag == "@@WORKDIR":
            workdir_ok = rest == "ok"
    return MachineFacts(
        hostname, uid, tuple(addresses), tuple(gpus), docker, workdir_ok, stdout
    )


def facts_problems(facts: MachineFacts, label: str, workdir: str) -> list[str]:
    """Why the harness cannot run on this machine ([] when it can)."""
    problems = []
    if not facts.gpus:
        problems.append(f"{label}: nvidia-smi lists no GPU")
    if facts.docker == "missing":
        problems.append(f"{label}: docker is not installed")
    elif facts.docker == "denied":
        problems.append(
            f"{label}: this ssh user cannot run docker (add it to the docker group "
            "or log in as root)"
        )
    if not facts.workdir_ok:
        problems.append(f"{label}: {workdir} is not writable (pass --workdir)")
    return problems


# --------------------------------------------------------------------------
# Control-plane addresses
# --------------------------------------------------------------------------


def _usable(addr: Address) -> bool:
    ip = ipaddress.IPv4Address(addr.ip)
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return False
    return not addr.iface.startswith(VIRTUAL_IFACE_PREFIXES)


def _rank(addr: Address) -> tuple[int, int, str]:
    ip = ipaddress.IPv4Address(addr.ip)
    private = ip.is_private or ip in ipaddress.IPv4Network("100.64.0.0/10")
    pref = next(
        (i for i, p in enumerate(IFACE_PREFERENCE) if addr.iface.startswith(p)),
        len(IFACE_PREFERENCE),
    )
    return (0 if private else 1, pref, addr.iface)


def candidates(facts: MachineFacts) -> list[Address]:
    """Usable addresses, best control-plane candidate first."""
    return sorted((a for a in facts.addresses if _usable(a)), key=_rank)


def _address(facts: MachineFacts, ip: str) -> Address | None:
    return next((a for a in facts.addresses if a.ip == ip), None)


def choose_control(
    specs: Sequence[MachineSpec], facts: Sequence[MachineFacts]
) -> list[Address]:
    """One control-plane address per machine (node0 first).

    An explicit ``control_ip`` must be assigned on its machine. The others
    take their address on the first node0 subnet that every one of them has
    an address on (private before public; wg*, bond*, eth*, en*, ib* first);
    a machine's ``iface`` limits it to that NIC's addresses.
    """
    labels = [s.label or f.hostname for s, f in zip(specs, facts)]

    def options(i: int) -> list[Address]:
        found = candidates(facts[i])
        iface = specs[i].iface
        return [a for a in found if a.iface == iface] if iface else found

    chosen: list[Address | None] = []
    for spec, fact, label in zip(specs, facts, labels):
        if spec.control_ip is None:
            chosen.append(None)
            continue
        addr = _address(fact, spec.control_ip)
        if addr is None:
            raise MachineError(
                f"{label}: control_ip {spec.control_ip} is not assigned on the machine "
                f"(its addresses: {', '.join(a.ip for a in fact.addresses) or 'none'})"
            )
        chosen.append(addr)
    if chosen[0] is None:
        node0_options = options(0)
    else:
        node0_options = [chosen[0]]
    for base in node0_options:
        net = base.network
        picks: list[Address | None] = [base]
        for i in range(1, len(specs)):
            if chosen[i] is not None:
                picks.append(chosen[i])
                continue
            picks.append(
                next(
                    (a for a in options(i) if ipaddress.IPv4Address(a.ip) in net),
                    None,
                )
            )
        if all(p is not None for p in picks):
            return [p for p in picks if p is not None]
    listing = "; ".join(
        f"{label}: {', '.join(f'{a.iface}={a.ip}/{a.prefix}' for a in candidates(f)) or 'none'}"
        for label, f in zip(labels, facts)
    )
    raise MachineError(
        "no subnet that every machine shares with node0; pass control_ip= for each "
        f"machine (addresses: {listing})"
    )


def peer_candidates(spec: MachineSpec, control_ip: str) -> list[tuple[str, int]]:
    """(target, port) pairs node0 tries for a peer, in order."""
    if spec.peer is not None:
        return [spec.peer]
    out = [(f"{spec.access.user}@{control_ip}", 22)]
    laptop = (spec.access.target, spec.access.port)
    if spec.access.jump is None and laptop not in out:
        out.append(laptop)
    return out


# --------------------------------------------------------------------------
# node0 -> peer trust (one key per experiment) and the runner's hosts file
# --------------------------------------------------------------------------


def trust_marker(exp_id: str) -> str:
    return f"{TRUST_KEY_PREFIX}{exp_id}"


def trust_key_path(workdir: str, exp_id: str) -> str:
    return f"{workdir}/ssh/{exp_id}_ed25519"


def known_hosts_path(workdir: str, exp_id: str) -> str:
    """Host keys of the peers, as node0 first saw them for this experiment."""
    return f"{workdir}/ssh/{exp_id}_known_hosts"


def keygen_script(workdir: str, exp_id: str) -> str:
    """On node0: create the experiment's key once; print the public key."""
    key = trust_key_path(workdir, exp_id)
    d = shlex.quote(f"{workdir}/ssh")
    return "\n".join(
        [
            "set -eu",
            f"mkdir -p {d}",
            f"chmod 700 {d}",
            f"[ -f {shlex.quote(key)} ] || ssh-keygen -q -t ed25519 -N '' "
            f"-C {shlex.quote(trust_marker(exp_id))} -f {shlex.quote(key)}",
            f'echo "@@PUBKEY $(cat {shlex.quote(key + ".pub")})"',
        ]
    )


def authorize_script(pubkey: str, exp_id: str) -> str:
    """On a peer: add node0's key to the ssh user's authorized_keys once."""
    parts = pubkey.split()
    if len(parts) != 3 or parts[0] != "ssh-ed25519" or parts[2] != trust_marker(exp_id):
        raise MachineError(f"unexpected trust key {pubkey[:60]!r}")
    if not re.match(r"^[A-Za-z0-9+/=]+$", parts[1]):
        raise MachineError("unexpected trust key encoding")
    line = shlex.quote(" ".join(parts))
    return "\n".join(
        [
            "set -eu",
            'mkdir -p "$HOME/.ssh"',
            'chmod 700 "$HOME/.ssh"',
            'f="$HOME/.ssh/authorized_keys"',
            'touch "$f"',
            'chmod 600 "$f"',
            f'grep -qxF {line} "$f" || printf "%s\\n" {line} >> "$f"',
            "echo @@AUTHORIZED",
        ]
    )


def peer_check_script(workdir: str, exp_id: str, target: str, port: int) -> str:
    """On node0: ssh to a peer with the experiment key; print its hostname."""
    argv = [
        "ssh",
        "-i", trust_key_path(workdir, exp_id),
        "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts_path(workdir, exp_id)}",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        target,
        "hostname",
    ]  # fmt: skip
    return f'echo "@@PEER $({shlex.join(argv)} 2>&1 | tail -1)"'


def cleanup_script(exp_id: str, label_value: str, node0: bool, workdir: str) -> str:
    """Remove this experiment's containers and its trust key (machines stay)."""
    marker = shlex.quote(trust_marker(exp_id))
    lines = [
        "set -u",
        f"ids=$(docker ps -aq --filter label=cb.exp={shlex.quote(label_value)} "
        "2>/dev/null || true)",
        'if [ -n "$ids" ]; then docker rm -f $ids >/dev/null 2>&1 || true; fi',
        'f="$HOME/.ssh/authorized_keys"',
        # Replace the file only from a complete awk pass: a failed one must
        # never truncate the user's authorized_keys.
        'if [ -f "$f" ]; then',
        f'  if awk -v m={marker} \'$NF != m\' "$f" > "$f.cb-tmp"; then cat "$f.cb-tmp" > "$f"; fi',
        '  rm -f "$f.cb-tmp"',
        "fi",
    ]
    if node0:
        key = shlex.quote(trust_key_path(workdir, exp_id))
        hosts = shlex.quote(known_hosts_path(workdir, exp_id))
        lines.append(f"rm -f {key} {key}.pub {hosts}")
    lines.append("echo @@CLEANED")
    return "\n".join(lines)


def hosts_file_text(
    members: Iterable[tuple[str, int | None, str | None]],
    control: Sequence[Address],
    key_path: str,
    *,
    known_hosts: str | None = None,
) -> str:
    """The runner's --hosts file: node0 is this host, peers use the trust key.

    ``members``: (peer target, port, iface override) per machine, node0 first
    (its target is ignored). ``known_hosts``: the file the peer check filled,
    so the runner trusts the host keys ``controller up`` saw, not a stale
    ``~/.ssh/known_hosts`` entry left by a reinstalled machine.
    """
    lines = ["# written by `cluster_bench controller launch`; node0 runs the harness"]
    for i, ((target, port, iface), addr) in enumerate(zip(members, control)):
        nic = f" iface={iface or addr.iface}"
        if i == 0:
            lines.append(f"node0 {addr.ip} -{nic}")
        else:
            kh = f" known_hosts={known_hosts}" if known_hosts else ""
            lines.append(
                f"node{i} {addr.ip} {target} {port or 22} identity={key_path}{kh}{nic}"
            )
    return "\n".join(lines) + "\n"
