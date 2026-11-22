# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Markdown views of the knowledge base and of one exported experiment.

Views follow the agent-docs conventions (SPDX HTML comment, ``# Title``, no
frontmatter) and only the part between the generated-content markers is
rewritten on export, so hand-written notes above or below survive. Every
view is a pure function of the records (or the summary), so re-exporting
unchanged data reproduces identical bytes.
"""

from __future__ import annotations

import json
import shlex
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import yaml

from benchmarks.lium_cluster_bench.domain.schema import (
    INTERACTIVITY_FLOORS,
    ExperimentSummary,
    LaunchPlan,
)
from benchmarks.lium_cluster_bench.kb.jsonl import KbError
from benchmarks.lium_cluster_bench.kb.records import (
    AccuracyRecord,
    ClusterRecord,
    DeployRecord,
    ExperimentRecord,
    FabricRecord,
    FindingRecord,
    HealthRecord,
    PeakRecord,
    RunRecord,
)
from benchmarks.lium_cluster_bench.report import facts
from benchmarks.lium_cluster_bench.report.doc import (
    SPDX_MD_HEADER,
    Cell,
    code,
    inline,
    md_escape,
    md_table,
    raw,
)
from benchmarks.lium_cluster_bench.report.fmt import (
    gbs,
    gib,
    iso_utc,
    ms,
    num,
    pct,
    ratio_pct,
)
from benchmarks.lium_cluster_bench.report.plans import role_views
from benchmarks.lium_cluster_bench.report.repro import pretty_argv, repro_commands

BEGIN = "<!-- cluster_bench:generated:begin -->"
END = "<!-- cluster_bench:generated:end -->"
NA = "n/a"
CORRECTNESS_LINE = (
    "Correctness status: unverified (waived: throughput benchmark; smoke "
    "completions only)"
)
USER_FLOORS = INTERACTIVITY_FLOORS
USABLE = ("valid", "valid_with_recovery")
KIND_ORDER = (
    "topology",
    "user_floor",
    "saturation",
    "noise",
    "nic_irrelevant",
    "bandwidth",
    "mtp",
    "nic_count",
    "deploy",
)


# --------------------------------------------------------------------------
# Generated-content markers
# --------------------------------------------------------------------------


def check_markers(text: str, label: str) -> None:
    """Raise ``KbError`` unless ``text`` has no markers or one ordered pair."""
    begins, ends = text.count(BEGIN), text.count(END)
    if begins == 0 and ends == 0:
        return
    if begins != 1 or ends != 1 or text.index(BEGIN) > text.index(END):
        raise KbError(f"{label}: malformed cluster_bench generated-content markers")


def merge_generated(existing: str | None, title: str, body: str) -> str:
    """Replace the generated block of ``existing`` (or create the document)."""
    block = f"{BEGIN}\n{body.strip()}\n{END}"
    if existing is None:
        return f"{SPDX_MD_HEADER}\n# {title}\n\n{block}\n"
    check_markers(existing, title)
    if BEGIN not in existing:
        return existing.rstrip("\n") + f"\n\n{block}\n"
    head, rest = existing.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{block}{tail}"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _yes_no(value: bool | None) -> str:
    return NA if value is None else ("yes" if value else "no")


def _link(text: str, href: str) -> Cell:
    return raw(f"[{md_escape(text)}]({href})")


def _text(value: str) -> Cell:
    """Generated finding text: backticks render as code, pipes stay escaped."""
    return raw(value)


def _date(epoch: float | None) -> str:
    text = iso_utc(epoch)
    return text.replace("T", " ").replace("Z", "") if text else NA


def _exp_order(experiments: Sequence[ExperimentRecord]) -> dict[str, tuple]:
    return {e.exp_id: (e.started_at, e.exp_id) for e in experiments}


def cluster_views(fp: str) -> dict[str, str]:
    """View file name -> title for one cluster directory."""
    return {
        "cluster.md": f"Cluster {fp}",
        "speed-of-light.md": f"Speed of Light: {fp}",
        "topology-ranking.md": f"Topology Ranking: {fp}",
        "knob-findings.md": f"Knob Findings: {fp}",
        "fabric-and-health.md": f"Fabric and Health: {fp}",
        "resources.md": f"Resources: {fp}",
    }


# --------------------------------------------------------------------------
# kb_root/INDEX.md
# --------------------------------------------------------------------------


def render_index(experiments: Sequence[ExperimentRecord]) -> str:
    """Body of ``kb_root/INDEX.md``: every experiment and cluster."""
    exps = sorted(experiments, key=lambda e: (e.started_at, e.exp_id))
    rows = [
        (
            code(e.exp_id),
            _link(e.fp, f"clusters/{e.fp}/cluster.md"),
            e.gpu_name,
            f"{e.node_count} x {e.gpus_per_node}",
            f"{e.model} ({e.checkpoint})",
            code(e.dataset),
            e.verdict,
            code(e.leader_family) if e.leader_family else NA,
            num(e.leader_tps_per_gpu, dash=NA),
            str(e.leader_concurrency) if e.leader_concurrency else NA,
            _yes_no(e.saturated),
            _date(e.started_at),
        )
        for e in exps
    ]
    by_fp: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for e in exps:
        by_fp[e.fp].append(e)
    cluster_rows = []
    for fp in sorted(by_fp):
        members = by_fp[fp]
        views = ", ".join(
            f"[{name.removesuffix('.md')}](clusters/{fp}/{name})"
            for name in list(cluster_views(fp))[1:]
        )
        cluster_rows.append(
            (
                _link(fp, f"clusters/{fp}/cluster.md"),
                members[-1].gpu_name,
                str(len(members)),
                code(members[-1].exp_id),
                raw(views),
            )
        )
    return "\n\n".join(
        [
            "Experiments exported by `benchmarks/lium_cluster_bench`. Records are "
            "append-only JSONL under `clusters/<fingerprint>/records/` and "
            "`index.jsonl`; the tables below are regenerated on every export.",
            "## Experiments",
            md_table(
                (
                    "Experiment",
                    "Cluster",
                    "GPU",
                    "Nodes x GPUs",
                    "Model",
                    "Dataset",
                    "Verdict",
                    "Best family",
                    "Best tok/s/GPU",
                    "C",
                    "Saturated",
                    "Started (UTC)",
                ),
                rows,
                numeric=(8, 9),
            ),
            "## Clusters",
            md_table(
                ("Cluster", "GPU", "Experiments", "Latest", "Views"), cluster_rows, (2,)
            ),
            *_scaling_section(exps),
        ]
    )


def _scaling_section(exps: Sequence[ExperimentRecord]) -> list[str]:
    """Same SKU and fabric (shape fingerprint) measured at several sizes."""
    by_shape: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for e in exps:
        if e.shape_fp:
            by_shape[e.shape_fp].append(e)
    rows = []
    for shape in sorted(by_shape):
        members = by_shape[shape]
        if len({e.node_count for e in members}) < 2:
            continue
        for e in sorted(members, key=lambda m: (m.node_count, m.started_at)):
            rows.append(
                (
                    code(shape),
                    e.gpu_name,
                    f"{e.node_count} x {e.gpus_per_node}",
                    code(e.exp_id),
                    code(e.leader_family) if e.leader_family else NA,
                    num(e.leader_tps_per_gpu, dash=NA),
                    num(e.leader_usd_per_mtok, 2, dash=NA),
                    e.fabric_type or NA,
                )
            )
    if not rows:
        return []
    return [
        "## Scaling across cluster sizes",
        "Experiments whose nodes share one hardware shape (shape fingerprint: "
        "the cluster fingerprint without the node count), by cluster size.",
        md_table(
            (
                "Shape",
                "GPU",
                "Nodes x GPUs",
                "Experiment",
                "Best family",
                "Best tok/s/GPU",
                "$/M output tok",
                "Fabric",
            ),
            rows,
            (5, 6),
        ),
    ]


# --------------------------------------------------------------------------
# clusters/<fp>/cluster.md
# --------------------------------------------------------------------------

SNAPSHOT_FIELDS = (
    "gpu_name",
    "node_count",
    "gpus_per_node",
    "hbm_mib",
    "compute_cap",
    "nvlink_active",
    "driver_version",
    "cuda_version",
    "ofed_version",
    "nvidia_peermem",
    "kernel",
    "docker_version",
    "cpu_model",
    "cpu_count",
    "mem_total_bytes",
)


def _nic_summary(snapshot: ClusterRecord) -> str:
    if not snapshot.nodes:
        return NA
    node = snapshot.nodes[0]
    return f"{len(node.active_hcas)}/{node.hcas} active: {node.hca_rates or NA}"


def render_cluster(
    fp: str, snapshots: Sequence[ClusterRecord], experiments: Sequence[ExperimentRecord]
) -> str:
    """Body of ``clusters/<fp>/cluster.md`` from its hardware snapshots."""
    order = _exp_order(experiments)
    snaps = sorted(snapshots, key=lambda c: order.get(c.exp_id, (0.0, c.exp_id)))
    if not snaps:
        return "No hardware snapshot recorded yet."
    latest = snaps[-1]
    props: list[tuple[Cell, ...]] = [
        ("Fingerprint", code(fp)),
        (
            "GPU",
            f"{latest.gpu_name} x {latest.gpus_per_node} per node x "
            f"{latest.node_count} nodes",
        ),
        ("GPU profile", code(latest.gpu_profile)),
        (
            "HBM per GPU",
            f"{num(latest.hbm_mib / 1024, 0)} GiB" if latest.hbm_mib else NA,
        ),
        ("Compute capability", latest.compute_cap or NA),
        ("NVLink active", _yes_no(latest.nvlink_active)),
        ("RDMA NICs per node", _nic_summary(latest)),
        (
            "Driver / CUDA",
            f"{latest.driver_version or NA} / {latest.cuda_version or NA}",
        ),
        (
            "OFED / nvidia-peermem",
            f"{latest.ofed_version or NA} / {_yes_no(latest.nvidia_peermem)}",
        ),
        ("Kernel / Docker", f"{latest.kernel or NA} / {latest.docker_version or NA}"),
        (
            "CPU / memory",
            f"{latest.cpu_model or NA} ({latest.cpu_count or NA} logical CPUs) / "
            f"{gib(latest.mem_total_bytes, dash=NA)} GiB",
        ),
    ]
    node_rows = [
        (
            n.name,
            str(n.gpus),
            " / ".join(n.gpu_names) or NA,
            f"{len(n.active_hcas)}/{n.hcas}",
            n.hca_rates or NA,
            n.driver_version,
            _yes_no(n.nvlink_active),
            _yes_no(n.nvidia_peermem),
        )
        for n in latest.nodes
    ]
    exps = sorted(
        (e for e in experiments if e.fp == fp), key=lambda e: (e.started_at, e.exp_id)
    )
    exp_rows = [
        (
            code(e.exp_id),
            _date(e.started_at),
            e.verdict,
            f"{e.model} ({e.checkpoint})",
            code(e.leader_family) if e.leader_family else NA,
            num(e.leader_tps_per_gpu, dash=NA),
        )
        for e in exps
    ]
    diffs = []
    for snap in snaps[:-1]:
        changed = [
            f"{name} {getattr(snap, name)!s} -> {getattr(latest, name)!s}"
            for name in SNAPSHOT_FIELDS
            if getattr(snap, name) != getattr(latest, name)
        ]
        if _nic_summary(snap) != _nic_summary(latest):
            changed.append(f"NICs {_nic_summary(snap)} -> {_nic_summary(latest)}")
        if changed:
            diffs.append(f"- `{snap.exp_id}`: " + "; ".join(changed))
    parts = [
        f"Hardware fingerprint as observed by `{latest.exp_id}` (latest of "
        f"{len(snaps)} snapshot(s)).",
        md_table(("Property", "Value"), props),
        "## Nodes",
        md_table(
            (
                "Node",
                "GPUs",
                "GPU model",
                "Active NICs",
                "NIC rates",
                "Driver",
                "NVLink",
                "peermem",
            ),
            node_rows,
            (1,),
        ),
        "## Experiments",
        md_table(
            (
                "Experiment",
                "Started (UTC)",
                "Verdict",
                "Model",
                "Best family",
                "Best tok/s/GPU",
            ),
            exp_rows,
            (5,),
        ),
        "## Snapshot differences",
        "\n".join(diffs) if diffs else "None: every snapshot matches the latest.",
        "## Views",
        "\n".join(
            f"- [{title}]({name})"
            for name, title in list(cluster_views(fp).items())[1:]
        ),
    ]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# clusters/<fp>/resources.md
# --------------------------------------------------------------------------


def _pct(value: float | None, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%" if value is not None else NA


def _tokens(value: float | None) -> str:
    if value is None:
        return NA
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{int(value)}"


def _min_nics(e: ExperimentRecord) -> str:
    """Measured minimum, naming the swept family when it is not the leader."""
    if e.min_nics_measured is None:
        return NA
    text = str(e.min_nics_measured)
    if e.min_nics_family and e.min_nics_family != e.leader_family:
        text += f" (on {e.min_nics_family})"
    return text


def _nvlink_row(e: ExperimentRecord, shape: str) -> tuple[str, ...]:
    delta = NA
    if e.nvlink_off_delta_pct is not None:
        delta = f"{e.nvlink_off_delta_pct:+.1f}%"
        if e.nvlink_off_within_noise is not None:
            delta += " (noise)" if e.nvlink_off_within_noise else ""
    return (
        code(e.exp_id),
        shape,
        e.nvlink_off_family or e.leader_family or NA,
        str(e.nvlink_off_concurrency) if e.nvlink_off_concurrency else NA,
        e.nvlink_off_method or NA,
        e.nvlink_off_path or NA,
        num(e.nvlink_on_tps_per_gpu, 1, dash=NA),
        num(e.nvlink_off_tps_per_gpu, 1, dash=NA),
        delta,
        num(e.nvlink_off_p2p_gbs, 1, dash=NA),
        num(e.nvlink_off_pcie_p99_gbs, 1, dash=NA),
        e.nvlink_off_status or NA,
        (e.nvlink_off_reason or "").replace("|", "/") or NA,
    )


def render_resources(experiments: Sequence[ExperimentRecord]) -> str:
    """Interconnect needs, NVLink off vs on, MFU and KV capacity per experiment."""
    order = _exp_order(experiments)
    exps = sorted(experiments, key=lambda e: order.get(e.exp_id, (0.0, e.exp_id)))
    links, nvlink, compute, kv = [], [], [], []
    for e in exps:
        shape = f"{e.node_count} x {e.gpus_per_node}"
        if e.nvlink_off_status is not None:
            nvlink.append(_nvlink_row(e, shape))
        links.append(
            (
                code(e.exp_id),
                shape,
                e.leader_family or NA,
                _min_nics(e),
                str(e.min_nics_estimated) if e.min_nics_estimated is not None else NA,
                num(e.ib_p99_gbs_per_node, 1, dash=NA),
                num(e.ib_need_gbs_per_node, 1, dash=NA),
                num(e.nvlink_p99_gbs_per_gpu, 1, dash=NA),
                _yes_no(e.nvlink_needed),
                num(e.gpu_nic_pcie_p99_gbs, 1, dash=NA),
            )
        )
        compute.append(
            (
                code(e.exp_id),
                shape,
                e.leader_family or NA,
                (e.mfu_precision or NA).upper(),
                num(e.peak_tflops_per_gpu, 0, dash=NA),
                num(e.leader_tflops_per_gpu, 1, dash=NA),
                _pct(e.leader_mfu_prefill),
                _pct(e.leader_mfu_decode),
                _pct(e.leader_mfu_total),
            )
        )
        kv.append(
            (
                code(e.exp_id),
                shape,
                e.leader_family or NA,
                _tokens(e.kv_total_tokens),
                _tokens(e.kv_max_request_tokens),
                _tokens(e.kv_context_p50),
                str(e.kv_requests_at_p50) if e.kv_requests_at_p50 is not None else NA,
                _pct(e.kv_usage_peak, 0),
                num(e.kv_bytes_per_token, 0, dash=NA),
            )
        )
    parts = [
        "Focus resources of each experiment's leader at saturation: the interconnect "
        "it needs (busiest link p99; need = p99 / target utilization; fewest NICs "
        "measured by the NIC sweep, which runs on the best cross-node family when "
        "the leader has no cross-node traffic, and estimated from the leader's "
        "own IB traffic), model FLOPs "
        "utilization (config.json FLOPs of useful tokens / dense peak at the "
        "checkpoint's GEMM precision) and KV cache capacity (per-rank pool x "
        "attention-DP ranks x instances of the serving role).",
        "## Interconnect",
        md_table(
            (
                "Experiment",
                "Nodes x GPUs",
                "Leader",
                "Fewest NICs (measured)",
                "Fewest NICs (traffic)",
                "IB p99 GB/s/node",
                "IB need GB/s/node",
                "NVLink p99 GB/s/GPU",
                "NVLink needed",
                "GPU-NIC PCIe p99 GB/s",
            ),
            links,
        ),
        "## NVLink off vs on",
        "The same configuration at the same concurrency with NVLink taken out of the "
        "GPU-GPU path: method bw_mode_off (nvidia-smi nvlink -sBwMode OFF, P2P over "
        "PCIe), link_width_0 (-sLWidth 0) or nccl_no_p2p (software: NCCL without P2P, "
        "staged through host memory, a lower bound); path is what carried the traffic.",
        md_table(
            (
                "Experiment",
                "Nodes x GPUs",
                "Config",
                "C",
                "Method",
                "Path",
                "tok/s/GPU NVLink",
                "tok/s/GPU off",
                "Delta",
                "GPU-GPU copy GB/s (off)",
                "PCIe p99 GB/s (off)",
                "Status",
                "Why",
            ),
            nvlink,
        ),
        "## Model FLOPs utilization",
        md_table(
            (
                "Experiment",
                "Nodes x GPUs",
                "Leader",
                "Precision",
                "Peak TFLOPS/GPU",
                "Achieved TFLOP/s/GPU",
                "MFU prefill",
                "MFU decode",
                "MFU total",
            ),
            compute,
        ),
        "## KV cache capacity",
        md_table(
            (
                "Experiment",
                "Nodes x GPUs",
                "Leader",
                "Context held (tokens)",
                "Longest request",
                "Context p50",
                "Requests at p50",
                "Peak KV usage",
                "Bytes/token",
            ),
            kv,
        ),
    ]
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# clusters/<fp>/fabric-and-health.md
# --------------------------------------------------------------------------

SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}


def render_fabric_health(
    fabric: Sequence[FabricRecord],
    health: Sequence[HealthRecord],
    experiments: Sequence[ExperimentRecord],
) -> str:
    """Upstream switch topology per rental and node-health findings."""
    order = _exp_order(experiments)
    rows = []
    for f in sorted(fabric, key=lambda r: order.get(r.exp_id, (0.0, r.exp_id))):
        rate = f.rail_rate_gbps
        rows.append(
            (
                code(f.exp_id),
                f.kind,
                f"{f.rails} x {rate if not isinstance(rate, list) else '/'.join(map(str, rate))} Gb/s",
                _yes_no(f.rail_optimized),
                str(f.nodes_per_leaf) if f.nodes_per_leaf else NA,
                f"{f.leaves}/{f.spines}",
                str(f.max_hops) if f.max_hops is not None else NA,
                num(f.max_oversubscription, 2, dash=NA),
                str(f.degraded_links),
                ",".join(f.tiers_seen) or NA,
                code(f.topology_hash) if f.topology_hash else NA,
            )
        )
    parts = [
        "Upstream fabric as each experiment observed it (tiers: S sysfs, M MAD/SMP, "
        "L LLDP, T traceroute, P perftest/perfquery, D derived) and node-health "
        "findings. Leaf names identify the switches across rentals.",
        "## Fabric",
        md_table(
            (
                "Experiment",
                "Kind",
                "Rails",
                "Rail-optimized",
                "Nodes/leaf",
                "Leaves/spines",
                "Max hops",
                "Max oversub",
                "Degraded",
                "Tiers",
                "Topology hash",
            ),
            rows,
            (4, 6, 7, 8),
        )
        if rows
        else "No fabric topology recorded.",
    ]
    if fabric:
        latest = max(fabric, key=lambda r: order.get(r.exp_id, (0.0, r.exp_id)))
        if latest.leaf_names:
            parts.append("Leaf switches (latest): " + ", ".join(latest.leaf_names))
        if latest.findings or latest.link_findings:
            parts.append(
                "\n".join(f"- {t}" for t in (*latest.link_findings, *latest.findings))
            )
    issues = sorted(
        (h for h in health if h.severity in ("error", "warn")),
        key=lambda h: (SEVERITY_RANK.get(h.severity, 3), h.code, h.node, h.exp_id),
    )
    parts.append("## Health findings (errors and warnings)")
    parts.append(
        md_table(
            ("Severity", "Code", "Node", "Experiment", "Message"),
            [
                (h.severity, code(h.code), h.node, code(h.exp_id), _text(h.message))
                for h in issues[:200]
            ],
        )
        if issues
        else "None recorded."
    )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# clusters/<fp>/speed-of-light.md
# --------------------------------------------------------------------------


def _plane_rank(plane: str) -> int:
    order = facts.PLANE_ORDER
    return order.index(plane) if plane in order else len(order)


def render_speed_of_light(
    peaks: Sequence[PeakRecord], experiments: Sequence[ExperimentRecord]
) -> str:
    """Body of ``speed-of-light.md``: every peak of the fingerprint."""
    if not peaks:
        return "No microbenchmark peaks recorded yet."
    order = _exp_order(experiments)
    groups: dict[tuple, list[PeakRecord]] = defaultdict(list)
    for peak in peaks:
        groups[(peak.plane, peak.test, peak.nic_count)].append(peak)
    best_rows = []
    for plane, test, nics in sorted(
        groups, key=lambda k: (_plane_rank(k[0]), k[1], -(k[2] or 0))
    ):
        members = groups[(plane, test, nics)]
        values = [p.value_gbs for p in members]
        best = max(values)
        nominal = max((p.nominal_gbs for p in members if p.nominal_gbs), default=None)
        best_rows.append(
            (
                plane,
                code(test),
                NA if nics is None else str(nics),
                gbs(best),
                gbs(statistics.median(values)),
                gbs(nominal, dash=NA),
                ratio_pct(best / nominal if nominal else None, dash=NA),
                str(len(members)),
                str(len({p.exp_id for p in members})),
            )
        )
    all_rows = [
        (
            code(p.exp_id),
            p.plane,
            code(p.test),
            code(p.scope),
            NA if p.nic_count is None else str(p.nic_count),
            gbs(p.value_gbs),
            gbs(p.nominal_gbs, dash=NA),
            ratio_pct(p.fraction_of_nominal, dash=NA),
            p.tool,
        )
        for p in sorted(
            peaks,
            key=lambda p: (
                order.get(p.exp_id, (0.0, p.exp_id)),
                _plane_rank(p.plane),
                p.test,
                -(p.nic_count or 0),
                p.scope,
                p.key,
            ),
        )
    ]
    return "\n\n".join(
        [
            "Speed-of-light microbenchmarks measured in the serving container "
            "(consult-perf-knowledge evidence category 5). Under-load utilization "
            "in reports is the sampled peak divided by these values.",
            "## Best per test",
            md_table(
                (
                    "Plane",
                    "Test",
                    "NICs",
                    "Best GB/s",
                    "Median GB/s",
                    "Nominal GB/s",
                    "Best / nominal",
                    "Samples",
                    "Experiments",
                ),
                best_rows,
                (2, 3, 4, 5, 6, 7, 8),
            ),
            "## All measurements",
            md_table(
                (
                    "Experiment",
                    "Plane",
                    "Test",
                    "Scope",
                    "NICs",
                    "GB/s",
                    "Nominal GB/s",
                    "/ nominal",
                    "Tool",
                ),
                all_rows,
                (4, 5, 6, 7),
            ),
        ]
    )


# --------------------------------------------------------------------------
# clusters/<fp>/topology-ranking.md
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Measurement:
    """Repetitions of one (experiment, candidate, concurrency, spec), averaged."""

    exp_id: str
    family: str
    family_title: str
    concurrency: int
    spec: str
    cand_key: str
    mtp: bool
    nic_count: int | None
    tps: float
    user_tps: float | None
    point_ids: tuple[str, ...]
    goodput: tuple[tuple[str, float], ...] = ()  # (SLO tier, good tok/s/GPU)

    def sort_key(self) -> tuple[Any, ...]:
        return (self.tps, self.exp_id, self.family, self.concurrency, self.spec)


def _measurements(runs: Sequence[RunRecord]) -> list[_Measurement]:
    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for run in runs:
        groups[(run.exp_id, run.cand_key, run.concurrency, run.spec)].append(run)
    out = []
    for key in sorted(groups):
        members = groups[key]
        first = members[0]
        tiers = sorted({name for r in members for name in r.goodput_tps_per_gpu})
        goodput = tuple(
            (
                name,
                statistics.fmean(
                    r.goodput_tps_per_gpu[name]
                    for r in members
                    if name in r.goodput_tps_per_gpu
                ),
            )
            for name in tiers
        )
        out.append(
            _Measurement(
                exp_id=first.exp_id,
                family=first.family,
                family_title=first.family_title,
                concurrency=first.concurrency,
                spec=first.spec,
                cand_key=first.cand_key,
                mtp=first.mtp,
                nic_count=first.nic_count,
                tps=statistics.fmean(r.tps_per_gpu or 0.0 for r in members),
                user_tps=facts.mean_or_none(r.user_tps_p50 for r in members),
                point_ids=tuple(sorted(r.point_id for r in members)),
                goodput=goodput,
            )
        )
    return out


def _best(items: Sequence[_Measurement]) -> _Measurement | None:
    return max(items, key=_Measurement.sort_key, default=None)


def _accuracy_cell(record: AccuracyRecord | None) -> str:
    if record is None:
        return NA
    if record.accuracy is None:
        return record.status
    return f"{record.status} {record.accuracy:.3f}"


def _best_goodput(items: Sequence[_Measurement], tier: str) -> float | None:
    """Highest good output tok/s/GPU of one SLO tier over a family's items."""
    values = [dict(m.goodput)[tier] for m in items if tier in dict(m.goodput)]
    return max(values) if values else None


def render_topology_ranking(
    runs: Sequence[RunRecord],
    deploys: Sequence[DeployRecord],
    experiments: Sequence[ExperimentRecord],
    accuracy: Sequence[AccuracyRecord] = (),
) -> str:
    """Body of ``topology-ranking.md``: families by best tok/s/GPU."""
    verdicts = {(a.exp_id, a.cand_key): a for a in accuracy}
    usable = [
        r
        for r in runs
        if r.status in USABLE
        and r.tps_per_gpu is not None
        and r.purpose not in ("smoke", "diagnose")
        and not r.simulate_acceptance
    ]
    parts = [
        "Families ranked by their best valid output tok/s/GPU across experiments "
        "(Accuracy: the GSM8K gate verdict of that measurement's configuration; a "
        "`fail` never becomes an experiment's recommendation); "
        "goodput columns give the best output tok/s/GPU of requests meeting each "
        "SLO tier (TTFT and ITL bounds, from AIPerf per-request records). "
        "A measurement is one experiment, candidate, concurrency and spec, with its "
        "repetitions averaged; valid and valid-with-recovery runs of every purpose "
        "count, smoke, diagnostic-window and simulated-acceptance runs do not. Rankings compare only "
        "within one workload: model, checkpoint, dataset and served context."
    ]
    workloads: dict[tuple, list[RunRecord]] = defaultdict(list)
    for run in usable:
        workloads[(run.model, run.checkpoint, run.dataset, run.served_context)].append(
            run
        )
    if not workloads:
        parts.append("No valid runs recorded yet.")
    for workload in sorted(workloads):
        model, checkpoint, dataset, context = workload
        tiers = sorted(
            {name for r in workloads[workload] for name in r.goodput_tps_per_gpu}
        )
        by_family: dict[str, list[_Measurement]] = defaultdict(list)
        for item in _measurements(workloads[workload]):
            by_family[item.family].append(item)
        ranked = []
        for items in by_family.values():
            best = _best(items)
            if best is None:
                continue
            at_floor = [
                _best([m for m in items if (m.user_tps or 0.0) >= floor])
                for floor in USER_FLOORS
            ]
            ranked.append((best, items, at_floor))
        ranked.sort(key=lambda entry: (-entry[0].tps, entry[0].family))
        rows = []
        for rank, (best, items, at_floor) in enumerate(ranked, start=1):
            rows.append(
                (
                    str(rank),
                    inline(f"{best.family_title} ", code(best.family)),
                    num(best.tps),
                    str(best.concurrency),
                    code(best.spec),
                    "on" if best.mtp else "off",
                    "all" if best.nic_count is None else str(best.nic_count),
                    _accuracy_cell(verdicts.get((best.exp_id, best.cand_key))),
                    num(best.user_tps, dash=NA),
                    *(num(m.tps if m else None, dash=NA) for m in at_floor),
                    *(num(_best_goodput(items, tier), dash=NA) for tier in tiers),
                    str(len(items)),
                    str(len({m.exp_id for m in items})),
                    inline(
                        code(best.exp_id),
                        " / ",
                        code(", ".join(best.point_ids)),
                    ),
                )
            )
        parts.append(f"## {model} ({checkpoint}), {dataset}, context {context}")
        parts.append(
            md_table(
                (
                    "Rank",
                    "Family",
                    "Best tok/s/GPU",
                    "C",
                    "Spec",
                    "MTP",
                    "NICs",
                    "Accuracy",
                    "User tok/s p50",
                    *(f"Best at >= {f:g} tok/s/user" for f in USER_FLOORS),
                    *(f"Goodput {tier} tok/s/GPU" for tier in tiers),
                    "Measurements",
                    "Experiments",
                    "Best measurement (exp / points)",
                ),
                rows,
                (0, 2, 3, *range(8, 11 + len(USER_FLOORS) + len(tiers))),
            )
        )
    ranked_families = {r.family for r in usable}
    failed: dict[str, list[DeployRecord]] = defaultdict(list)
    for deploy in deploys:
        if deploy.family not in ranked_families:
            failed[deploy.family].append(deploy)
    if failed:
        rows = []
        for family in sorted(failed):
            members = failed[family]
            reasons = Counter(d.failure or "no valid run" for d in members if not d.ok)
            rows.append(
                (
                    code(family),
                    str(len(members)),
                    raw(", ".join(f"`{k}` x{v}" for k, v in sorted(reasons.items())))
                    if reasons
                    else NA,
                    ", ".join(sorted({d.exp_id for d in members})),
                )
            )
        parts.append("## Families without a valid run")
        parts.append(
            md_table(
                ("Family", "Deploy attempts", "Failures", "Experiments"), rows, (1,)
            )
        )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# clusters/<fp>/knob-findings.md
# --------------------------------------------------------------------------


def _finding_order(finding: FindingRecord, order: dict[str, tuple]) -> tuple[Any, ...]:
    kind = KIND_ORDER.index(finding.kind) if finding.kind in KIND_ORDER else 99
    return (order.get(finding.exp_id, (0.0, finding.exp_id)), kind, finding.key)


def render_knob_findings(
    findings: Sequence[FindingRecord], experiments: Sequence[ExperimentRecord]
) -> str:
    """Body of ``knob-findings.md`` in the two agent-docs schemas."""
    order = _exp_order(experiments)
    ordered = sorted(findings, key=lambda f: _finding_order(f, order))
    knob_rows = [
        (
            _text(f.symptom),
            _text(f.knobs),
            _text(f.effect),
            _text(f"{f.validation}; evidence category {f.evidence_category}"),
            _text(f.caveat),
        )
        for f in ordered
        if f.table == "knob"
    ]
    scenario_rows = [
        (
            _text(f.symptom),
            _text(f.knobs),
            _text(
                f"{f.effect} ({f.validation}; evidence category {f.evidence_category})"
            ),
            _text(f.try_first or NA),
            _text(f.caveat),
        )
        for f in ordered
        if f.table == "scenario"
    ]
    parts = [
        "Findings derived from audited runs, written in the two table schemas of "
        "`agent-docs/guides/knob-tuning/sglang.md`. Evidence categories follow "
        "`consult-perf-knowledge`: 1 AIPerf data, 3 deployment history, "
        "5 hardware speed-of-light. Deltas at or below the noise floor are noise.",
        "## Knob findings",
        md_table(
            (
                "Symptom",
                "Candidate knobs",
                "Expected metric effect",
                "Validation",
                "Caveat",
            ),
            knob_rows,
        )
        if knob_rows
        else "No knob findings yet.",
        "## Evidence-backed entries",
        md_table(
            (
                "Scenario",
                "Knobs / config surface",
                "Evidence signal",
                "What to try first",
                "Caveat",
            ),
            scenario_rows,
        )
        if scenario_rows
        else "No evidence-backed entries yet.",
    ]
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# experiments/<exp_id>/ (run-artifacts layout)
# --------------------------------------------------------------------------


def _yaml_block(data: dict[str, Any]) -> str:
    plain = json.loads(json.dumps(data, default=str))
    return yaml.safe_dump(
        plain, sort_keys=True, allow_unicode=True, default_flow_style=False
    ).rstrip()


def _plan_sections(plan: LaunchPlan) -> list[str]:
    parts = [
        f"Transport `{plan.transport}`; cross-node classes "
        f"{', '.join(plan.cross_node) or 'none'}; {plan.gpu_count} GPUs.",
    ]
    for view in role_views(plan):
        if view.role == "control":
            continue
        parts.append(
            f"### {view.role} ({view.instances} instance(s), "
            f"{view.containers} container(s))"
        )
        if view.sglang:
            parts.append("SGLang config (instance 0):")
            parts.append(f"```yaml\n{_yaml_block(view.sglang)}\n```")
            if view.config_variants > 1:
                parts.append(
                    f"This role has {view.config_variants} distinct rendered configs."
                )
        env = "\n".join(f"{k}={shlex.quote(v)}" for k, v in view.env.items())
        parts.append(f"Env of `{view.representative}` (secrets masked):")
        parts.append(f"```bash\n{env or '# none'}\n```")
        parts.append("Command:")
        parts.append(f"```bash\n{pretty_argv(view.command) or '# none'}\n```")
    return parts


def render_recommended_config(
    summary: ExperimentSummary,
    plan: LaunchPlan | None,
    exp_dir: str,
    summary_sha256: str | None,
) -> str:
    """Body of ``final/recommended_config.md``."""
    lines = [CORRECTNESS_LINE]
    lead = facts.leader_point(summary)
    cand = facts.leader_candidate(summary)
    titles = facts.family_titles(summary)
    if cand is None:
        lines.append(
            f"No recommendation: the experiment verdict is `{summary.verdict}` and "
            "no valid leader was measured. See `known_limitations.md`."
        )
    else:
        metrics = lead.metrics if lead else None
        confirm, confirm_delta = facts.confirm_delta(summary)
        noise = summary.noise
        bullets = [
            f"- Candidate: `{cand.key}` (candidate fingerprint `{cand.fp}`), "
            f"{titles.get(cand.family_id, cand.family_id)}",
            f"- Cluster: `{summary.fingerprint}`, {summary.node_count} x "
            f"{summary.gpus_per_node} {summary.gpu_name}",
            f"- Workload: `{summary.dataset}` (AIPerf {summary.aiperf_version}, AgentX "
            f"closed loop), served context {summary.served_context}",
            f"- Throughput: {num(lead.tps_per_gpu if lead else None, dash=NA)} "
            f"tok/s/GPU at C={lead.key.concurrency if lead else NA} (user p50 "
            f"{num(metrics.user_tps_p50 if metrics else None, dash=NA)} tok/s, TTFT "
            f"p50 {ms(metrics.ttft_p50_ms if metrics else None, dash=NA)} ms, ITL p50 "
            f"{ms(metrics.itl_p50_ms if metrics else None, dash=NA)} ms); saturated: "
            f"{_yes_no(summary.saturated)}",
            "- Confirmatory rerun: "
            + (
                f"{num(confirm.tps_per_gpu, dash=NA)} tok/s/GPU "
                f"({pct(confirm_delta, signed=True, dash=NA)})"
                if confirm
                else "not run"
            ),
            "- Noise floor: "
            + (
                f"{noise.noise_floor_pct:.2f}% (n={len(noise.samples)})"
                if noise
                else "not measured"
            ),
        ]
        if summary.mtp is not None:
            bullets.append(
                f"- MTP on vs off: {pct(summary.mtp.delta_pct, signed=True, dash=NA)} "
                f"(accept length {num(summary.mtp.accept_len, 2, dash=NA)})"
            )
        lines.append("\n".join(bullets))
        lines.append("## Effective configuration")
        if plan is None:
            lines.append(
                f"LaunchPlan not available (`deployments/{cand.fp}/plan.json` was not "
                "found), so the effective flags cannot be listed."
            )
        else:
            lines.extend(_plan_sections(plan))
    evidence = [
        f"- Summary: `{exp_dir}/analysis/summary.json`"
        + (f" (sha256 `{summary_sha256}`)" if summary_sha256 else ""),
        f"- Report: `{exp_dir}/report/report.zh-CN.html`",
        f"- KB records: `clusters/{summary.fingerprint}/records/`",
    ]
    if lead is not None and lead.artifact_dir:
        evidence.insert(1, f"- Leader point artifacts: `{exp_dir}/{lead.artifact_dir}`")
    lines.append("## Evidence")
    lines.append("\n".join(evidence))
    return "\n\n".join(lines)


def known_limitations(summary: ExperimentSummary) -> list[str]:
    """Limitations of one experiment, most important first (English)."""
    items: list[str] = []
    lead = facts.leader_point(summary)
    if lead is None:
        items.append(f"No valid leader (verdict `{summary.verdict}`).")
    if lead is not None and summary.saturated is False:
        items.append(f"Leader not saturated: {summary.saturation_note or NA}.")
    if summary.points and summary.noise is None:
        items.append("No noise pilot: deltas cannot be separated from noise.")
    if lead is not None and summary.confirm_point_id is None:
        items.append("No confirmatory rerun of the leader.")
    if any(p.key.spec == "screen" for p in summary.points):
        items.append(
            "Search points use the screen spec (`--unsafe-override`, "
            "submission_valid=false) and only rank candidates."
        )
    if summary.dropped_traces:
        items.append(f"{summary.dropped_traces} trace(s) dropped for context overflow.")
    for family in summary.families:
        if family.status not in ("measured", "disabled"):
            items.append(
                f"Family `{family.family_id}`: {family.status} ({family.reason or NA})."
            )
    if not summary.nic_arms and summary.nic_reason:
        items.append(f"No NIC-count comparison: {summary.nic_reason}.")
    items.extend(f"Condition: {c}" for c in summary.conditions)
    items.extend(summary.limitations)
    items.extend(f"UNVERIFIED: {u}" for u in summary.unverified)
    items.append(
        "Correctness was not verified beyond smoke completions (throughput benchmark)."
    )
    return items


def render_known_limitations(summary: ExperimentSummary) -> str:
    """Body of ``final/known_limitations.md``."""
    return "\n".join(f"- {item}" for item in known_limitations(summary))


def render_repro_script(summary: ExperimentSummary, kb_root: str) -> str:
    """``final/reproduced_commands.sh``: harness and re-export commands."""
    lines = [
        "#!/bin/bash",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
        "AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        f"# Reproduction commands of experiment {summary.exp_id} on cluster",
        f"# {summary.fingerprint}. Generated by the cluster_bench KB export and",
        "# rewritten on every export: do not edit. Run on node0 of a cluster with",
        "# the same fingerprint; the full run takes several hours.",
        "set -e",
        "",
        'WORKDIR="${WORKDIR:-/root/cluster_bench}"',
        f'KB_ROOT="${{KB_ROOT:-{kb_root}}}"',
        "",
    ]
    titles = {
        "run": "# Full experiment (harness command line)",
        "render": "# Re-render the report from analysis/summary.json",
        "kb": "# Export the experiment into the knowledge base",
    }
    for command in repro_commands(summary):
        lines.append(titles[command.purpose])
        lines.append(command.shell)
        lines.append("")
    return "\n".join(lines)


def render_manifest(
    summary: ExperimentSummary,
    exp_dir: str,
    summary_sha256: str | None,
    record_counts: dict[str, int],
) -> str:
    """``manifest.yaml`` of one exported experiment."""
    cand = facts.leader_candidate(summary)
    lead = facts.leader_point(summary)
    manifest: dict[str, Any] = {
        "exp_id": summary.exp_id,
        "fingerprint": summary.fingerprint,
        "harness": "benchmarks/lium_cluster_bench",
        "summary_schema_version": summary.schema_version,
        "started_at": iso_utc(summary.started_at),
        "ended_at": iso_utc(summary.ended_at),
        "verdict": summary.verdict,
        "cluster": {
            "gpu_profile": summary.gpu_profile,
            "gpu_name": summary.gpu_name,
            "nodes": summary.node_count,
            "gpus_per_node": summary.gpus_per_node,
        },
        "model": {
            "name": summary.model,
            "checkpoint": summary.checkpoint.repo,
            "revision": summary.checkpoint.revision,
            "served_context": summary.served_context,
        },
        "workload": {"dataset": summary.dataset, "aiperf": summary.aiperf_version},
        "image": summary.image,
        "budget_hours": summary.budget_hours,
        "spent_hours": summary.spent_hours,
        "leader": (
            {
                "cand_key": cand.key,
                "cand_fp": cand.fp,
                "family": cand.family_id,
                "point_id": lead.key.id if lead else None,
                "tps_per_gpu": lead.tps_per_gpu if lead else None,
                "concurrency": lead.key.concurrency if lead else None,
            }
            if cand
            else None
        ),
        "source": {
            "exp_dir": exp_dir,
            "summary": "analysis/summary.json",
            "summary_sha256": summary_sha256,
        },
        "records": dict(sorted(record_counts.items())),
    }
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
        "AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# Generated by the cluster_bench KB export; rewritten on every export.\n"
    )
    body = yaml.safe_dump(
        manifest, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    return header + body
