# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner-side fabric probing: switch identity, paths, link tests, counters.

``FabricProber`` runs ``agents/fabric_probe.py`` on cluster nodes and turns
its JSON into the models of ``domain/fabric.py``. The agent runs in the
pinned image (``docker run --rm``, no GPUs, ``/dev/infiniband``, host
network, ``NET_ADMIN``, repo read-only at ``/bench/src``); when the image
lacks a fabric tool (infiniband-diags, perftest) the node's host venv python
runs the same agent against the host's tools. Inputs (trace targets, rails,
ports) are written to ``remote_dir`` on the node and mounted read-only.

Everything is fail-soft: every remote command has a timeout, and a node or
tool failure becomes a note, an ``error`` field or a placeholder report;
nothing raises into the benchmark. Raw agent JSON is kept in ``results_dir``.

Coordinator flow (fabric phase, after the model download)::

    reports = prober.identify(nodes, lldp_wait_s=35)
    topo = build_topology(reports, rail_of=rail_of)
    topo = build_topology(reports, prober.trace(topo, reports), rail_of=rail_of)
    matrix = prober.link_matrix(reports, topology=topo)
    findings = analyze_link_matrix(matrix, nominal_gbs_by_rail(topo), hops=topo.hops)
    refs = switch_port_refs(topo)          # per load point:
    before = prober.snapshot_switch_ports(refs, label=f"{point}-before")
    after = prober.snapshot_switch_ports(refs, label=f"{point}-after")
    loads = switch_load(before, after)

Expected durations, 8 nodes x 8 rails, defaults (worst case in brackets):

* ``identify``: container start ~2 s + sysfs + ~6 SMPs per IB port (~1 s);
  RoCE netdevs listen for LLDP for up to ``lldp_wait_s`` (35 s). A host pass
  (only when the image lacks a tool) ~3 s; a census pass on the nodes chosen
  by ``census_plan`` (<= ``census_max_smp`` SMPs each, ~2-5 s).
  [2 x (lldp_wait_s + identify_timeout_s) + census_timeout_s = 250 s.]
* ``trace``: ibtracert ~50 ms per path, UDP traceroute <= 8 x 0.4 s per
  path, 8 paths in parallel per node: < 10 s. [20 s + 10 s per 8 paths of
  the busiest node: 90 s for node0 with 56 paths.]
* ``link_matrix``: N-1 rounds per direction (N rounds for odd N), each about
  ``bw_seconds`` + 4 s (agent and container start, bw then lat): 8 nodes
  bidirectional = 14 rounds ~ 2 min. [``budget_s`` = 240 s, plus the round
  in flight (``round_timeout_s`` + 30 s); a fully failed first round, or two
  in a row, abandon the matrix.]
* ``snapshot_switch_ports``: one agent run on the via node (node0), ~1-3 s
  for 100-600 ports. [``snapshot_timeout_s`` = 60 s.]

Typical total ~3 min (IB) or ~3.5 min (RoCE, LLDP wait); worst case ~11 min.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from benchmarks.cluster_bench.adapters.docker import (
    LABEL_ROLE,
    Mount,
    docker_run_argv,
    sanitize_label,
)
from benchmarks.cluster_bench.domain.fabric import (
    FabricNodeReport,
    FabricNote,
    FabricTopology,
    LinkMatrix,
    LinkTest,
    PerfqueryReport,
    PerftestReport,
    SwitchPortRef,
    SwitchSnapshot,
    TraceReport,
    TraceResult,
    TraceTarget,
    build_topology,
    census_plan,
    link_tests_from_perftest,
    merge_reports,
    perfquery_ports,
    round_robin_rounds,
    snapshot_from_perfquery,
    trace_targets,
)
from benchmarks.cluster_bench.fsutil import atomic_write_text
from benchmarks.cluster_bench.ports import Cluster, CommandError

logger = logging.getLogger(__name__)

T = TypeVar("T")

CONTAINER_SRC = "/bench/src"
AGENT_REL = "benchmarks/cluster_bench/agents/fabric_probe.py"
CONTAINER_PYTHON = "python3"
FABRIC_ROLE = "fabric"
CONTAINER = "container"
HOST = "host"
# Tools whose absence from the image is worth a host-mode identify pass.
HOST_FALLBACK_TOOLS = (
    "smpquery",
    "sminfo",
    "ibtracert",
    "perfquery",
    "ib_write_bw",
    "ib_write_lat",
)
DEFAULT_LLDP_WAIT_S = 35.0
IDENTIFY_TIMEOUT_S = 60.0
CENSUS_TIMEOUT_S = 60.0
CENSUS_MAX_SMP = 1000
TRACE_BASE_TIMEOUT_S = 20.0
TRACE_TIMEOUT_PER_WAVE_S = 10.0
TRACE_WAVE = 8  # agent TRACE_THREADS
SNAPSHOT_TIMEOUT_S = 60.0
PERFTEST_PORT_BASE = 18600
PERFTEST_PORT_SLOTS = 4  # consecutive rounds rotate through distinct ports
PERFTEST_SLOT_SPAN = 32  # bw ports base+0..15, lat ports base+16..31
PERFTEST_SLACK_S = 30.0
MATRIX_BUDGET_S = 240.0
NODE_ERRORS = (CommandError, OSError, ValueError, KeyError)


def _tail(text: str, limit: int = 300) -> str:
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1][:limit] if lines else ""


def extract_document(stdout: str) -> dict[str, Any]:
    """The agent's JSON document: the last stdout line that parses as an
    object (image entrypoints may print banners first)."""
    for line in reversed((stdout or "").splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict):
            return doc
    raise ValueError("no JSON document on stdout")


def failed_report(error: str) -> FabricNodeReport:
    """Placeholder for a node whose agent never produced a report."""
    return FabricNodeReport(notes=(FabricNote(source="adapter", error=error),))


class FabricProber:
    """Runs the fabric agent on nodes; all methods are fail-soft.

    ``control_ips`` (node -> wg0 address) carries perftest's TCP handshake;
    ``clock`` only bounds the link-matrix budget.
    """

    def __init__(
        self,
        cluster: Cluster,
        *,
        image: str,
        workdir: str,
        results_dir: Path,
        run_id: str,
        src_dir: str | None = None,
        python: str | None = None,
        control_ips: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
        remote_dir: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        identify_timeout_s: float = IDENTIFY_TIMEOUT_S,
        census_max_smp: int = CENSUS_MAX_SMP,
        census_timeout_s: float = CENSUS_TIMEOUT_S,
        snapshot_timeout_s: float = SNAPSHOT_TIMEOUT_S,
        perftest_port_base: int = PERFTEST_PORT_BASE,
    ):
        self._cluster = cluster
        self._image = image
        self._workdir = workdir.rstrip("/")
        self._src_dir = src_dir or f"{self._workdir}/src"
        self._python = python or f"{self._workdir}/venv/bin/python"
        self._results = results_dir
        self._run_id = sanitize_label(run_id) or "run"
        self._remote_dir = remote_dir or f"{self._workdir}/fabric/{self._run_id}"
        self._ips = dict(control_ips or {})
        self._labels = {LABEL_ROLE: FABRIC_ROLE, **(labels or {})}
        self._clock = clock
        self._identify_timeout_s = identify_timeout_s
        self._census_max_smp = census_max_smp
        self._census_timeout_s = census_timeout_s
        self._snapshot_timeout_s = snapshot_timeout_s
        self._port_base = perftest_port_base
        self._seq = itertools.count(1)
        # node -> mode ("container" | "host") -> tool -> installed
        self._tools: dict[str, dict[str, dict[str, bool]]] = {}

    # ------------------------------------------------------------ commands

    def container_name(self, op: str, seq: int) -> str:
        return f"cb-fabric-{self._run_id[:24]}-{sanitize_label(op)}-{seq}"

    def agent_command(
        self, mode: str, args: Sequence[str], *, name: str
    ) -> tuple[str, str | None]:
        """(shell command, container name or None) running the agent."""
        if mode == HOST:
            argv = [self._python, f"{self._src_dir}/{AGENT_REL}", *args]
            return shlex.join(argv), None
        argv = docker_run_argv(
            name=name,
            image=self._image,
            command=[CONTAINER_PYTHON, f"{CONTAINER_SRC}/{AGENT_REL}", *args],
            gpus=None,
            mounts=(
                Mount(self._src_dir, CONTAINER_SRC, read_only=True),
                Mount(self._remote_dir, self._remote_dir, read_only=True),
            ),
            labels=self._labels,
            detach=False,
            remove=True,
            cap_add=("NET_ADMIN",),
        )
        prep = f"mkdir -p {shlex.quote(self._remote_dir)}"
        return f"{prep} && {shlex.join(argv)}", name

    def _save(self, name: str, text: str) -> None:
        try:
            atomic_write_text(self._results / name, text)
        except OSError as exc:
            # Raw evidence is best effort; never fail collection on it.
            logger.warning("could not save %s: %s", name, exc)

    def _run_agent(
        self,
        node: str,
        mode: str,
        args: Sequence[str],
        *,
        op: str,
        timeout: float,
        payload: Any = None,
        input_flag: str | None = None,
        save_as: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """(document, error): the parsed agent JSON or why there is none."""
        seq = next(self._seq)
        argv = list(args)
        try:
            if payload is not None and input_flag is not None:
                path = f"{self._remote_dir}/{sanitize_label(op)}-{node}-{seq}.json"
                self._cluster.put_text(node, path, json.dumps(payload) + "\n")
                argv += [input_flag, path]
            command, container = self.agent_command(
                mode, argv, name=self.container_name(op, seq)
            )
            result = self._cluster.run(node, command, timeout=timeout, check=False)
        except NODE_ERRORS as exc:
            return None, f"{op} on {node} ({mode}) could not run: {exc}"
        if not result.ok:
            if container is not None:
                self._cleanup(node, container)
            detail = _tail(result.stderr) or _tail(result.stdout)
            return None, f"{op} on {node} ({mode}) exit {result.rc}: {detail}"
        self._save(save_as or f"{sanitize_label(op)}-{node}-{mode}.json", result.stdout)
        try:
            return extract_document(result.stdout), None
        except ValueError as exc:
            return None, f"{op} on {node} ({mode}): {exc}"

    def _cleanup(self, node: str, container: str) -> None:
        """A timed-out ``docker run`` may leave its container running."""
        try:
            self._cluster.run(
                node,
                f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1 || true",
                timeout=60,
                check=False,
            )
        except NODE_ERRORS as exc:
            logger.warning("[%s] could not remove %s: %s", node, container, exc)

    def _fanout(
        self,
        nodes: Sequence[str],
        fn: Callable[[str], T],
        fallback: Callable[[str, str], T],
    ) -> dict[str, T]:
        """Concurrent ``fn`` per node; an exception becomes ``fallback``."""

        def safe(node: str) -> T:
            try:
                return fn(node)
            except NODE_ERRORS as exc:
                logger.warning("[%s] fabric probe failed: %s", node, exc)
                return fallback(node, str(exc))

        if not nodes:
            return {}
        return self._cluster.fanout(list(nodes), safe)

    # --------------------------------------------------------- tool routing

    def _learn(self, node: str, mode: str, report: FabricNodeReport) -> None:
        self._tools.setdefault(node, {})[mode] = dict(report.tools)

    def mode_for(self, node: str, tool: str | None) -> str | None:
        """Where to run an agent needing ``tool`` (None: any working mode);
        the pinned image is preferred."""
        modes = self._tools.get(node, {})
        for mode in (CONTAINER, HOST):
            tools = modes.get(mode)
            if tools is not None and (tool is None or tools.get(tool)):
                return mode
        return None

    def _seed_tools(self, reports: Mapping[str, FabricNodeReport] | None) -> None:
        """Tool knowledge for a prober that did not run ``identify`` (resume)."""
        for node, report in (reports or {}).items():
            if node in self._tools:
                continue
            if report.runner_tools:
                self._tools[node] = {m: dict(t) for m, t in report.runner_tools.items()}
            elif report.tools:
                mode = HOST if report.runner == HOST else CONTAINER
                self._tools[node] = {mode: dict(report.tools)}

    # ------------------------------------------------------------- identify

    def _identify(
        self, node: str, mode: str, args: Sequence[str], timeout: float, op: str
    ) -> tuple[FabricNodeReport | None, str | None]:
        doc, error = self._run_agent(
            node, mode, ["identify", *args], op=op, timeout=timeout
        )
        if doc is None:
            return None, error
        try:
            report = FabricNodeReport.model_validate(doc)
        except ValueError as exc:  # pydantic.ValidationError is a ValueError
            return None, f"{op} on {node} ({mode}): bad report: {exc}"
        update = {"runner": mode, "runner_tools": {mode: dict(report.tools)}}
        return report.model_copy(update=update), None

    def _identify_node(self, node: str, lldp_wait_s: float) -> FabricNodeReport:
        wait = f"{lldp_wait_s:g}"
        timeout = lldp_wait_s + self._identify_timeout_s
        report, error = self._identify(
            node,
            CONTAINER,
            ["--no-census", "--lldp-wait", wait],
            timeout,
            "identify",
        )
        errors = [error] if error else []
        if report is not None:
            self._learn(node, CONTAINER, report)
            lacking = [t for t in HOST_FALLBACK_TOOLS if not report.tools.get(t)]
            if not lacking:
                return report
        host_wait = "0" if report is not None else wait
        host, host_error = self._identify(
            node,
            HOST,
            ["--no-census", "--lldp-wait", host_wait],
            timeout,
            "identify",
        )
        if host_error:
            errors.append(host_error)
        if host is not None:
            self._learn(node, HOST, host)
        if report is not None and host is not None:
            report = merge_reports(report, host, runner=f"{CONTAINER}+{HOST}")
        report = report or host
        if report is None:
            return failed_report("; ".join(errors) or "identify failed")
        notes = tuple(FabricNote(source="adapter", error=e) for e in errors)
        return report.model_copy(update={"notes": (*report.notes, *notes)})

    def _census(
        self, node: str, keys: Sequence[str], report: FabricNodeReport
    ) -> FabricNodeReport:
        mode = self.mode_for(node, "smpquery")
        if mode is None:
            return report
        census, error = self._identify(
            node,
            mode,
            [
                "--lldp-wait", "0",
                "--max-smp", str(self._census_max_smp),
                "--census-only", ",".join(keys),
            ],
            self._census_timeout_s,
            "census",
        )  # fmt: skip
        if census is None:
            note = FabricNote(source="adapter", error=error or "census failed")
            return report.model_copy(update={"notes": (*report.notes, note)})
        return merge_reports(report, census, runner=report.runner)

    def identify(
        self, nodes: Sequence[str], *, lldp_wait_s: float = DEFAULT_LLDP_WAIT_S
    ) -> dict[str, FabricNodeReport]:
        """Identity of every node's fabric ports, neighbors, SMs and LLDP,
        then one bounded census per distinct neighbor switch.

        Returns a report for every requested node (a placeholder carrying the
        error for nodes whose agent failed), in node order.
        """
        reports = self._fanout(
            nodes, lambda n: self._identify_node(n, lldp_wait_s), _failed_for
        )
        plan = census_plan(reports)
        censused = self._fanout(
            list(plan), lambda n: self._census(n, plan[n], reports[n]), _keep(reports)
        )
        reports.update(censused)
        return {node: reports[node] for node in nodes}

    # ---------------------------------------------------------------- trace

    def trace(
        self,
        topology: FabricTopology,
        reports: Mapping[str, FabricNodeReport] | None = None,
    ) -> list[TraceResult]:
        """One path per unordered node pair per rail, traced from the lower
        node (ibtracert on IB, UDP traceroute on RoCE); batched per node."""
        self._seed_tools(reports)
        targets = trace_targets(topology)

        def run(node: str) -> list[TraceResult]:
            batch = targets[node]
            needs_ib = any(t.src_lid for t in batch)
            mode = self.mode_for(node, "ibtracert" if needs_ib else None)
            if mode is None:
                mode = self.mode_for(node, None)
            if mode is None:
                return _untraced(batch, f"no working fabric agent on {node}")
            waves = math.ceil(len(batch) / TRACE_WAVE)
            doc, error = self._run_agent(
                node,
                mode,
                ["trace"],
                op="trace",
                timeout=TRACE_BASE_TIMEOUT_S + TRACE_TIMEOUT_PER_WAVE_S * waves,
                payload=[t.model_dump() for t in batch],
                input_flag="--targets",
            )
            if doc is None:
                return _untraced(batch, error or "trace failed")
            try:
                return list(TraceReport.model_validate(doc).results)
            except ValueError as exc:
                return _untraced(batch, f"bad trace report from {node}: {exc}")

        results = self._fanout(
            list(targets), run, lambda n, e: _untraced(targets[n], e)
        )
        return [r for node in targets for r in results.get(node, [])]

    # ---------------------------------------------------------- link matrix

    def link_matrix(
        self,
        reports: Mapping[str, FabricNodeReport],
        *,
        topology: FabricTopology | None = None,
        rails: Sequence[int] | None = None,
        bw_seconds: int = 3,
        lat_iters: int = 2000,
        round_timeout_s: float = 60.0,
        bidirectional: bool = True,
        budget_s: float = MATRIX_BUDGET_S,
        bw_size: int = 65536,
        bw_qps: int = 2,
        lat_size: int = 2,
    ) -> LinkMatrix:
        """perftest bandwidth + latency between every node pair on every rail.

        Rounds of disjoint pairs (``round_robin_rounds``) run their pairs
        concurrently and every rail of a pair concurrently: server on the
        second node, client on the first (then swapped when
        ``bidirectional``), TCP handshake over the control network.
        """
        self._seed_tools(reports)
        topo = topology or build_topology(reports)
        settings = {
            "bidirectional": bidirectional,
            "bw_seconds": bw_seconds,
            "bw_msg_bytes": bw_size,
            "bw_qps": bw_qps,
            "lat_iters": lat_iters,
            "lat_msg_bytes": lat_size,
        }
        notes: list[str] = []
        nodes = [n for n in topo.nodes if self.mode_for(n, "ib_write_bw")]
        missing = [n for n in topo.nodes if n not in nodes]
        if missing:
            notes.append(
                "perftest (ib_write_bw) missing in the image and on the host of "
                + ", ".join(missing)
            )
        no_ip = [n for n in nodes if n not in self._ips]
        if no_ip:
            notes.append("no control address for " + ", ".join(no_ip))
            nodes = [n for n in nodes if n not in no_ip]
        wanted = set(rails) if rails is not None else set(topo.rail_indices)
        rail_ids = tuple(r for r in topo.rail_indices if r in wanted)
        if len(nodes) < 2 or not rail_ids:
            notes.append(
                "link matrix skipped: fewer than two testable nodes or no rails"
            )
            return LinkMatrix(
                nodes=topo.nodes,
                rails=rail_ids,
                skipped=True,
                notes=tuple(notes),
                **settings,
            )
        schedule: list[tuple[int, list[tuple[str, str]]]] = []
        for index, pairs in enumerate(round_robin_rounds(nodes)):
            schedule.append((index, pairs))
            if bidirectional:
                schedule.append((index, [(b, a) for a, b in pairs]))
        start = self._clock()
        tests: list[LinkTest] = []
        dead_rounds = 0
        done = 0
        for seq, (index, pairs) in enumerate(schedule):
            if self._clock() - start > budget_s:
                notes.append(
                    f"budget {budget_s:g}s exhausted after {seq} of {len(schedule)} rounds"
                )
                break
            round_tests = self._perftest_round(
                topo, pairs, rail_ids, seq, index, round_timeout_s, settings
            )
            tests += round_tests
            done += 1
            if round_tests and all(
                t.bw_gbs is None and t.lat_typical_us is None for t in round_tests
            ):
                dead_rounds += 1
                if seq == 0 or dead_rounds >= 2:
                    notes.append(
                        f"every test of round {seq + 1} failed; link matrix abandoned"
                    )
                    break
            else:
                dead_rounds = 0
        return LinkMatrix(
            nodes=tuple(nodes),
            rails=rail_ids,
            tests=tuple(tests),
            rounds=done,
            notes=tuple(notes),
            elapsed_s=round(self._clock() - start, 3),
            **settings,
        )

    def _perftest_round(
        self,
        topo: FabricTopology,
        pairs: Sequence[tuple[str, str]],
        rail_ids: Sequence[int],
        seq: int,
        round_index: int,
        round_timeout_s: float,
        settings: Mapping[str, Any],
    ) -> list[LinkTest]:
        slot = self._port_base + (seq % PERFTEST_PORT_SLOTS) * PERFTEST_SLOT_SPAN
        jobs: dict[str, tuple[str, str, list[dict[str, Any]]]] = {}
        planned: dict[tuple[str, str], list[int]] = {}
        for client, server in pairs:
            common = [
                r
                for r in rail_ids
                if _active(topo, client, r) and _active(topo, server, r)
            ]
            planned[(client, server)] = common
            for node, role, peer in (
                (client, "client", server),
                (server, "server", client),
            ):
                entries = []
                for position, rail in enumerate(common):
                    port = topo.port(node, rail)
                    if port is None:
                        continue
                    entries.append(
                        {
                            "rail": rail,
                            "hca": port.hca,
                            "port": port.port,
                            "gid_index": port.gid_index,
                            "bw_tcp_port": slot + position % 16,
                            "lat_tcp_port": slot + 16 + position % 16,
                        }
                    )
                jobs[node] = (role, peer, entries)
        args_common = [
            "--tests", "bw,lat",
            "--bw-seconds", str(settings["bw_seconds"]),
            "--bw-size", str(settings["bw_msg_bytes"]),
            "--bw-qps", str(settings["bw_qps"]),
            "--lat-iters", str(settings["lat_iters"]),
            "--lat-size", str(settings["lat_msg_bytes"]),
            "--timeout", f"{round_timeout_s:g}",
        ]  # fmt: skip

        def run(node: str) -> tuple[PerftestReport | None, str | None]:
            role, peer, entries = jobs[node]
            if not entries:
                return None, "no common active rails"
            mode = self.mode_for(node, "ib_write_bw")
            if mode is None:
                return None, f"perftest unavailable on {node}"
            args = ["perftest", "--role", role, *args_common]
            if role == "client":
                args += ["--peer", self._ips[peer]]
            doc, error = self._run_agent(
                node,
                mode,
                args,
                op=f"perftest-{role}",
                timeout=round_timeout_s + PERFTEST_SLACK_S,
                payload=entries,
                input_flag="--rails",
                save_as=f"perftest-{seq:02d}-{role}-{node}.json",
            )
            if doc is None:
                return None, error
            try:
                return PerftestReport.model_validate(doc), None
            except ValueError as exc:
                return None, f"bad perftest report from {node}: {exc}"

        outcomes = self._fanout(list(jobs), run, lambda n, e: (None, e))
        tests: list[LinkTest] = []
        for (client, server), common in planned.items():
            report, error = outcomes.get(client, (None, "not run"))
            server_report, server_error = outcomes.get(server, (None, "not run"))
            dst_hcas = {}
            for rail in common:
                port = topo.port(server, rail)
                if port is not None:
                    dst_hcas[rail] = port.hca
            if report is None:
                detail = error or "client failed"
                if server_error:
                    detail += f"; server: {server_error}"
                tests += [
                    LinkTest(
                        src=client,
                        dst=server,
                        rail=rail,
                        src_hca=_hca(topo, client, rail),
                        dst_hca=dst_hcas.get(rail),
                        round_index=round_index,
                        error=detail,
                    )
                    for rail in common
                ]
                continue
            tests += link_tests_from_perftest(
                report,
                src=client,
                dst=server,
                round_index=round_index,
                dst_hcas=dst_hcas,
                server=server_report,
            )
        return tests

    # ------------------------------------------------------------ snapshots

    def snapshot_switch_ports(
        self, refs: Sequence[SwitchPortRef], *, label: str = ""
    ) -> SwitchSnapshot:
        """PMA counters of every ref, read in one agent run on the refs'
        ``via_node`` (node0). Refs routed through other nodes are reported as
        not sampled."""
        if not refs:
            return SwitchSnapshot(
                node="", label=label, notes=("no switch ports to sample",)
            )
        node = refs[0].via_node
        mine = [r for r in refs if r.via_node == node]
        notes: list[str] = []
        if len(mine) < len(refs):
            notes.append(
                f"{len(refs) - len(mine)} ref(s) routed via other nodes skipped"
            )
        mode = self.mode_for(node, "perfquery")
        if mode is None:
            notes.append(f"perfquery unavailable on {node}")
            return snapshot_from_perfquery(
                refs, None, node=node, label=label, notes=notes
            )
        stem = sanitize_label(label) or "snapshot"
        doc, error = self._run_agent(
            node,
            mode,
            ["perfquery"],
            op="perfquery",
            timeout=self._snapshot_timeout_s,
            payload=perfquery_ports(mine),
            input_flag="--ports",
            save_as=f"perfquery-{stem}-{node}.json",
        )
        report = None
        if doc is None:
            notes.append(error or "perfquery failed")
        else:
            try:
                report = PerfqueryReport.model_validate(doc)
            except ValueError as exc:
                notes.append(f"bad perfquery report: {exc}")
        return snapshot_from_perfquery(
            refs, report, node=node, label=label, notes=notes
        )


def _failed_for(node: str, error: str) -> FabricNodeReport:
    return failed_report(f"identify on {node}: {error}")


def _keep(
    reports: Mapping[str, FabricNodeReport],
) -> Callable[[str, str], FabricNodeReport]:
    def fallback(node: str, error: str) -> FabricNodeReport:
        note = FabricNote(source="adapter", error=f"census on {node}: {error}")
        report = reports[node]
        return report.model_copy(update={"notes": (*report.notes, note)})

    return fallback


def _untraced(targets: Sequence[TraceTarget], error: str) -> list[TraceResult]:
    return [TraceResult(target=t, method="none", error=error) for t in targets]


def _active(topo: FabricTopology, node: str, rail: int) -> bool:
    port = topo.port(node, rail)
    return port is not None and port.active


def _hca(topo: FabricTopology, node: str, rail: int) -> str | None:
    port = topo.port(node, rail)
    return port.hca if port is not None else None
