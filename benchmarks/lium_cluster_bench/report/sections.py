# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds the Chinese report as document blocks: sections 1-13 plus appendix
(8-10 live in ``sections_cluster.py``).

Every builder reads only the ``ExperimentSummary`` (and the leader's
``LaunchPlan`` when one is available) and degrades to an explicit sentence
when the data it needs is missing.
"""

from __future__ import annotations

import json
import shlex
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import yaml

from benchmarks.lium_cluster_bench.domain.schema import (
    BandwidthStat,
    Candidate,
    ExperimentSummary,
    FloorResult,
    LaunchPlan,
    NicArm,
    NodeInventory,
    PointResult,
)
from benchmarks.lium_cluster_bench.report import facts, svg
from benchmarks.lium_cluster_bench.report.doc import (
    Block,
    Bullets,
    Cell,
    Code,
    Document,
    Figure,
    Heading,
    Inline,
    Kpi,
    KpiRow,
    Paragraph,
    Table,
    code,
    inline,
    strong,
)
from benchmarks.lium_cluster_bench.report.fmt import (
    DASH,
    gbs,
    gib,
    hours,
    integer,
    ms,
    num,
    pct,
    pct_delta,
    ratio_pct,
    ts_utc,
)
from benchmarks.lium_cluster_bench.report.plans import (
    PlanLookup,
    role_views,
    sorted_containers,
)
from benchmarks.lium_cluster_bench.report.reasons import localize
from benchmarks.lium_cluster_bench.report.repro import pretty_argv, repro_commands
from benchmarks.lium_cluster_bench.report.sections_cluster import (
    fabric_section,
    health_section,
    runtime_kpi_lines,
    runtime_section,
)
from benchmarks.lium_cluster_bench.report.sections_resources import (
    resource_kpis,
    resources_section,
)
from benchmarks.lium_cluster_bench.report.strings import Strings

MAX_DETAIL_ROWS = 40
SERVING_ROLES = ("agg", "prefill", "decode")


def conc(value: int | None) -> str:
    """Concurrency, or a dash when unknown."""
    return DASH if value is None else str(value)


def truncate(text: str, limit: int = 160) -> str:
    """Single-line text cut to ``limit`` characters."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class Ctx:
    """Shared state and formatting helpers of the section builders."""

    summary: ExperimentSummary
    s: Strings
    plan: LaunchPlan | None
    titles: dict[str, str]
    curves: tuple[facts.Curve, ...]

    def family(self, family_id: str | None) -> str:
        if family_id is None:
            return DASH
        return self.titles.get(family_id, family_id)

    def variant(self, cand: Candidate) -> str:
        """Family title plus every knob that differs from the default."""
        parts = [self.family(cand.family_id)]
        if not cand.mtp:
            parts.append(self.s.t("variant.mtp_off"))
        if cand.nic_count is not None:
            parts.append(self.s.t("variant.nics", n=cand.nic_count))
        if cand.simulate_acceptance:
            parts.append(self.s.t("variant.sim"))
        if cand.rungs:
            parts.append(self.s.t("variant.rungs", rungs="+".join(cand.rungs)))
        return " · ".join(parts)

    def family_of(self, cand_key: str | None, point_id: str | None = None) -> str:
        point = self.summary.point(point_id)
        if point is not None:
            return self.variant(point.candidate)
        if cand_key:
            return self.family(cand_key.split("|", 1)[0])
        return DASH

    def tri(self, value: bool | None) -> str:
        if value is None:
            return self.s.t("common.unknown")
        return self.s.t("common.bool_true" if value else "common.bool_false")

    def noise(self, within: bool | None) -> str:
        if within is None:
            return self.s.t("common.noise_unknown")
        return self.s.t("common.within_noise" if within else "common.beyond_noise")

    def basis(self, note: str) -> str:
        return self.s.t("common.basis", note=localize(note)) if note else ""

    def tps(self, value: float | None) -> str:
        return DASH if value is None else self.s.t("common.tps_gpu", v=num(value))

    def mtp(self, enabled: bool) -> str:
        return self.s.t("common.mtp_on" if enabled else "common.mtp_off")

    def nics(self, count: int | None) -> str:
        return self.s.t("common.all_nics") if count is None else str(count)


def build_document(
    summary: ExperimentSummary,
    strings: Strings,
    plan_lookup: PlanLookup | None = None,
) -> Document:
    """The whole report as one ``Document``."""
    leader = facts.leader_candidate(summary)
    plan = plan_lookup(leader.fp) if plan_lookup is not None and leader else None
    ctx = Ctx(
        summary=summary,
        s=strings,
        plan=plan,
        titles=facts.family_titles(summary),
        curves=tuple(facts.family_curves(summary)),
    )
    blocks: list[Block] = []
    for builder in (
        summary_section,
        resources_section,
        env_section,
        method_section,
        topology_section,
        mtp_section,
        bandwidth_section,
        nic_section,
        fabric_section,
        runtime_section,
        health_section,
        config_section,
        limits_section,
        repro_section,
        appendix_section,
    ):
        blocks.extend(builder(ctx))
    return Document(
        lang=strings.t("meta.lang"),
        title=strings.t(
            "meta.title",
            model=summary.model,
            nodes=summary.node_count,
            gpus=summary.gpus_per_node,
            gpu=summary.gpu_name,
        ),
        subtitle=strings.t(
            "meta.subtitle",
            exp_id=summary.exp_id,
            fp=summary.fingerprint,
            start=ts_utc(summary.started_at),
            end=ts_utc(summary.ended_at),
        ),
        blocks=tuple(blocks),
        toc_label=strings.t("meta.toc"),
        footer=strings.t("meta.footer", schema=summary.schema_version),
        figure_label=strings.raw("meta.figure"),
    )


# --------------------------------------------------------------------------
# 1 Summary
# --------------------------------------------------------------------------


def _sorted_floors(summary: ExperimentSummary) -> list[FloorResult]:
    return sorted(summary.floors, key=lambda f: f.user_tps_floor)


def _nic_reduced_arms(ctx: Ctx) -> tuple[NicArm | None, list[NicArm]]:
    arms = ctx.summary.nic_arms
    full = facts.full_nic_arm(arms)
    if full is None:
        return None, []
    others = [a for a in facts.nic_arms_sorted(arms) if a.nic_count < full.nic_count]
    return full, others


def summary_section(ctx: Ctx) -> list[Block]:
    """Section 1: verdict, headline numbers and conclusions."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.summary"), 2, "sec-1")]
    out.append(
        Paragraph(
            s.r(
                "summary.verdict",
                verdict=strong(s.label("verdict", sm.verdict)),
                detail=s.label("verdict_detail", sm.verdict),
            ),
            tone="normal" if sm.verdict == "ok" else "warn",
        )
    )
    lead = facts.leader_point(sm)
    if lead is None:
        out.append(Paragraph(s.r("summary.no_leader"), tone="warn"))
    else:
        headers = (
            s.t("summary.kpi_metric"),
            s.t("summary.kpi_value"),
            s.t("summary.kpi_detail"),
        )
        out.append(KpiRow(tuple(_kpis(ctx, lead)), headers))
    out.append(Bullets(tuple(_summary_bullets(ctx, lead))))
    return out


def _kpis(ctx: Ctx, lead: PointResult) -> list[Kpi]:
    s, sm = ctx.s, ctx.summary
    unit = s.t("common.tps_unit")
    tiles = [
        Kpi(
            s.t("summary.kpi_best"),
            num(lead.tps_per_gpu),
            s.t(
                "common.family_at",
                family=ctx.family(lead.candidate.family_id),
                c=lead.key.concurrency,
            ),
            unit,
        )
    ]
    for floor in _sorted_floors(sm):
        label = s.t("summary.kpi_floor", floor=num(floor.user_tps_floor, 0))
        if floor.tps_per_gpu is None:
            tiles.append(Kpi(label, s.t("summary.kpi_not_met"), floor.note))
            continue
        tiles.append(
            Kpi(
                label,
                num(floor.tps_per_gpu),
                s.t(
                    "common.family_at",
                    family=ctx.family_of(floor.cand_key, floor.point_id),
                    c=conc(floor.concurrency),
                ),
                unit,
            )
        )
    if sm.mtp is not None and sm.mtp.delta_pct is not None:
        tiles.append(
            Kpi(
                s.t("summary.kpi_mtp"),
                pct(sm.mtp.delta_pct, signed=True),
                s.t("summary.kpi_accept", accept=num(sm.mtp.accept_len, 2))
                + " · "
                + ctx.noise(sm.mtp.within_noise),
            )
        )
    cost = sm.cost
    if cost is not None and cost.leader_usd_per_mtok is not None:
        tiles.append(
            Kpi(
                s.t("summary.kpi_cost"),
                num(cost.leader_usd_per_mtok, 2),
                s.t(
                    "summary.kpi_cost_detail",
                    cluster=num(cost.cluster_per_hour, 2),
                    source=cost.source,
                ),
                s.t("summary.kpi_cost_unit"),
            )
        )
    full, others = _nic_reduced_arms(ctx)
    measured = [a for a in others if a.delta_vs_full_pct is not None]
    if full is not None and measured:
        smallest = measured[-1]
        tiles.append(
            Kpi(
                s.t("summary.kpi_nic", n=smallest.nic_count, full=full.nic_count),
                pct(smallest.delta_vs_full_pct, signed=True),
                ctx.family(sm.nic_family_id),
            )
        )
    tiles.extend(resource_kpis(ctx, lead))
    return tiles


def _floor_line(ctx: Ctx, floor: FloorResult) -> Inline:
    s = ctx.s
    if floor.tps_per_gpu is None:
        return s.r(
            "summary.floor_missing",
            floor=num(floor.user_tps_floor, 0),
            note=floor.note,
        )
    where = s.t(
        "common.family_at",
        family=ctx.family_of(floor.cand_key, floor.point_id),
        c=conc(floor.concurrency),
    )
    return s.r(
        "summary.floor",
        floor=num(floor.user_tps_floor, 0),
        tps=strong(ctx.tps(floor.tps_per_gpu)),
        where=where,
        interp=ctx.tps(floor.interpolated_tps_per_gpu),
    )


def _nic_arms_text(ctx: Ctx) -> str:
    s = ctx.s
    full, others = _nic_reduced_arms(ctx)
    parts = []
    for arm in others:
        text = s.t(
            "summary.nic_arm",
            full=full.nic_count if full else DASH,
            n=arm.nic_count,
            delta=pct(arm.delta_vs_full_pct, signed=True),
            noise=ctx.noise(arm.within_noise),
        )
        if arm.staging_suspected:
            text += s.t("common.clause_sep") + s.t("summary.nic_staging")
        parts.append(text)
    return s.t("common.list_sep").join(parts) or DASH


def _nic_line(ctx: Ctx) -> Inline:
    s, sm = ctx.s, ctx.summary
    if not sm.nic_arms:
        return s.r("summary.nic_none", reason=localize(sm.nic_reason))
    arms = _nic_arms_text(ctx)
    if facts.leader_cross_node(sm, ctx.plan) is False:
        lead = facts.leader_point(sm)
        return s.r(
            "summary.nic_irrelevant",
            rate=gbs(lead.cross_node_gbs if lead else None),
            family=strong(ctx.family(sm.nic_family_id)),
            arms=arms,
        )
    return s.r(
        "summary.nic",
        family=ctx.family(sm.nic_family_id),
        c=conc(sm.nic_concurrency),
        arms=arms,
    )


def _mtp_line(ctx: Ctx) -> Inline:
    s, mtp = ctx.s, ctx.summary.mtp
    if mtp is None or mtp.delta_pct is None:
        return s.r("summary.mtp_none")
    return s.r(
        "summary.mtp",
        delta=strong(pct(mtp.delta_pct, signed=True)),
        noise=ctx.noise(mtp.within_noise),
        accept=num(mtp.accept_len, 2),
    )


def _summary_bullets(ctx: Ctx, lead: PointResult | None) -> list[Inline]:
    s, sm = ctx.s, ctx.summary
    items: list[Inline] = []
    if lead is not None:
        metrics = lead.metrics
        items.append(
            s.r(
                "summary.best",
                family=strong(ctx.variant(lead.candidate)),
                key=code(lead.candidate.key),
                c=lead.key.concurrency,
                tps=strong(ctx.tps(lead.tps_per_gpu)),
                user=num(metrics.user_tps_p50 if metrics else None),
                ttft=ms(metrics.ttft_p50_ms if metrics else None),
                itl=ms(metrics.itl_p50_ms if metrics else None),
            )
        )
        saturation_key = {
            True: "summary.saturated",
            False: "summary.not_saturated",
            None: "summary.saturation_unknown",
        }[sm.saturated]
        items.append(s.r(saturation_key, note=ctx.basis(sm.saturation_note)))
        confirm, delta = facts.confirm_delta(sm)
        if confirm is None:
            items.append(s.r("summary.no_confirm"))
        else:
            items.append(
                s.r(
                    "summary.confirm",
                    tps=strong(ctx.tps(confirm.tps_per_gpu)),
                    delta=pct(delta, signed=True),
                    noise=ctx.noise(facts.within_noise(sm, delta)),
                )
            )
        second = facts.runner_up(sm)
        if second is not None:
            delta = pct_delta(second.best_tps_per_gpu, lead.tps_per_gpu)
            items.append(
                s.r(
                    "summary.runner_up",
                    family=ctx.family(second.family_id),
                    tps=ctx.tps(second.best_tps_per_gpu),
                    c=conc(second.best_concurrency),
                    delta=pct(delta, signed=True),
                    noise=ctx.noise(facts.within_noise(sm, delta)),
                )
            )
    items.extend(_floor_line(ctx, floor) for floor in _sorted_floors(sm))
    items.append(_nic_line(ctx))
    items.append(_mtp_line(ctx))
    items.extend(runtime_kpi_lines(ctx))
    return items


# --------------------------------------------------------------------------
# 2 Environment
# --------------------------------------------------------------------------


def _hca_rates(node: NodeInventory, sep: str) -> str:
    rates: Counter[str] = Counter()
    for hca in node.active_hcas:
        port = hca.active_ports[0]
        rates[f"{num(port.rate_gbps, 0)} Gb/s {port.link_layer}"] += 1
    return sep.join(f"{count} × {rate}" for rate, count in sorted(rates.items()))


def _hca_summary(ctx: Ctx, node: NodeInventory) -> str:
    if not node.hcas:
        return ctx.s.t("common.none")
    return ctx.s.t(
        "env.hcas_value",
        active=len(node.active_hcas),
        total=len(node.hcas),
        rates=_hca_rates(node, ctx.s.t("common.clause_sep")) or DASH,
    )


def _provider_rows(ctx: Ctx) -> list[tuple[Cell, ...]]:
    """Rental facts from provider.json and the price-derived cost."""
    s, sm = ctx.s, ctx.summary
    rows: list[tuple[Cell, ...]] = []
    prov = sm.provider
    if prov is not None:
        rows.append(
            (
                s.t("env.provider"),
                s.t(
                    "env.provider_value",
                    provider=prov.provider,
                    cluster=prov.cluster_name or prov.cluster_id or DASH,
                    fabric=prov.fabric_id or DASH,
                    kind=prov.fabric_type or DASH,
                    rate=prov.link_rate or DASH,
                ),
            )
        )
    cost = sm.cost
    if cost is not None:
        rows.append(
            (
                s.t("env.cost"),
                s.t(
                    "env.cost_value",
                    node=num(cost.price_per_node_hour, 2),
                    cluster=num(cost.cluster_per_hour, 2),
                    total=num(cost.experiment_usd, 0),
                    source=cost.source,
                ),
            )
        )
    return rows


def env_section(ctx: Ctx) -> list[Block]:
    """Section 2: test environment and cluster fingerprint."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.env"), 2, "sec-3")]
    nodes = sm.inventory.nodes
    node0 = nodes[0] if nodes else None
    rows: list[tuple[Cell, ...]] = [
        (s.t("env.fingerprint"), code(sm.fingerprint)),
        (
            s.t("env.gpu"),
            s.t(
                "env.gpu_value",
                name=sm.gpu_name,
                per_node=sm.gpus_per_node,
                nodes=sm.node_count,
                total=sm.gpus_per_node * sm.node_count,
            ),
        ),
        (s.t("env.profile"), code(sm.gpu_profile)),
    ]
    if node0 is not None:
        gpu0 = node0.gpus[0] if node0.gpus else None
        rows += [
            (
                s.t("env.hbm"),
                s.t("env.hbm_value", gib=num(gpu0.memory_mib / 1024, 0))
                if gpu0
                else DASH,
            ),
            (s.t("env.compute_cap"), (gpu0.compute_cap if gpu0 else None) or DASH),
            (
                s.t("env.interconnect"),
                s.t("env.nvlink_active" if node0.nvlink_active else "env.pcie_only"),
            ),
            (s.t("env.hcas"), _hca_summary(ctx, node0)),
            (
                s.t("env.driver"),
                f"{node0.driver_version} / {node0.cuda_version or DASH}",
            ),
            (
                s.t("env.ofed"),
                f"{node0.ofed_version or DASH} / {ctx.tri(node0.nvidia_peermem)}",
            ),
            (
                s.t("env.kernel"),
                f"{node0.kernel or DASH} / {node0.docker_version or DASH}",
            ),
            (
                s.t("env.cpu"),
                s.t(
                    "env.cpu_value",
                    model=node0.cpu_model or DASH,
                    cores=node0.cpu_count or DASH,
                    mem=gib(node0.mem_total_bytes),
                ),
            ),
        ]
    rows += [
        (s.t("env.image"), code(sm.image)),
        (s.t("env.aiperf"), sm.aiperf_version),
        (s.t("env.dataset"), code(sm.dataset)),
        (
            s.t("env.context"),
            s.t("env.context_value", tokens=integer(sm.served_context)),
        ),
        (
            s.t("env.model"),
            s.t(
                "env.model_value",
                model=sm.model,
                repo=sm.checkpoint.repo,
                rev=sm.checkpoint.revision,
                size=num(sm.checkpoint.approx_size_gb, 0),
            ),
        ),
        (
            s.t("env.budget"),
            s.t(
                "env.budget_value",
                spent=hours(sm.spent_hours),
                budget=hours(sm.budget_hours),
            ),
        ),
        (s.t("env.window"), f"{ts_utc(sm.started_at)} → {ts_utc(sm.ended_at)}"),
    ]
    rows += _provider_rows(ctx)
    out.append(Paragraph(s.r("env.intro")))
    out.append(Table((s.t("env.key"), s.t("env.value")), tuple(rows)))
    if not nodes:
        out.append(Paragraph(s.r("env.no_inventory"), tone="note"))
        return out
    node_rows = []
    for node in nodes:
        names = sorted({g.name for g in node.gpus})
        node_rows.append(
            (
                node.name,
                str(len(node.gpus)),
                " / ".join(names) or DASH,
                f"{len(node.active_hcas)}/{len(node.hcas)}",
                _hca_rates(node, s.t("common.clause_sep")) or DASH,
                node.driver_version,
                ctx.tri(node.nvlink_active),
                ctx.tri(node.nvidia_peermem),
            )
        )
    out.append(
        Table(
            (
                s.t("env.node"),
                s.t("env.gpu_count"),
                s.t("env.gpu_name"),
                s.t("env.active_hcas"),
                s.t("env.hca_rate"),
                s.t("env.driver"),
                "NVLink",
                s.t("env.peermem"),
            ),
            tuple(node_rows),
            numeric=(1,),
            caption=s.t("env.nodes_caption"),
        )
    )
    return out


# --------------------------------------------------------------------------
# 3 Method and validity
# --------------------------------------------------------------------------


def method_section(ctx: Ctx) -> list[Block]:
    """Section 3: method, specs, noise floor and validity audit."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.method"), 2, "sec-4")]
    out.append(
        Paragraph(
            s.r(
                "method.agentx",
                aiperf=sm.aiperf_version,
                dataset=code(sm.dataset),
                ctx=integer(sm.served_context),
            )
        )
    )
    out.append(Paragraph(s.r("method.objective")))
    out.append(Paragraph(s.r("method.cache")))
    stats = facts.spec_stats(sm)
    if stats:
        out.append(
            Table(
                (
                    s.t("method.spec"),
                    s.t("method.spec_desc"),
                    s.t("method.points"),
                    s.t("audit.valid"),
                    s.t("audit.valid_with_recovery"),
                    s.t("audit.invalid"),
                    s.t("method.median_duration"),
                ),
                tuple(
                    (
                        code(st.spec),
                        s.label("spec", st.spec),
                        str(st.total),
                        str(st.valid),
                        str(st.recovered),
                        str(st.invalid),
                        num(st.median_duration_s, 0),
                    )
                    for st in stats
                ),
                numeric=(2, 3, 4, 5, 6),
                caption=s.t("method.specs_caption"),
            )
        )
    noise = sm.noise
    if noise is None:
        out.append(Paragraph(s.r("method.no_noise"), tone="warn"))
    else:
        cv = noise.stdev / noise.mean * 100.0 if noise.mean else None
        out.append(
            Paragraph(
                s.r(
                    "method.noise",
                    family=ctx.family_of(noise.cand_key),
                    c=noise.concurrency,
                    n=len(noise.samples),
                    samples=s.t("common.enum_sep").join(num(x) for x in noise.samples)
                    or DASH,
                    mean=num(noise.mean),
                    stdev=num(noise.stdev, 2),
                    cv=pct(cv, 2),
                    floor=strong(num(noise.noise_floor, 2)),
                    floor_pct=strong(pct(noise.noise_floor_pct, 2)),
                )
            )
        )
    out.append(Paragraph(s.r("method.saturation", note=ctx.basis(sm.saturation_note))))
    counts = facts.audit_counts(sm.points)
    out.append(
        Paragraph(
            s.r(
                "method.audit",
                total=len(sm.points),
                valid=counts["valid"],
                recovered=counts["valid_with_recovery"],
                invalid=counts["invalid"],
            )
        )
    )
    invalid = sorted(
        (p for p in sm.points if p.audit.status == "invalid"),
        key=lambda p: facts.point_sort_key(sm, p),
    )
    if invalid:
        out.append(
            Table(
                (
                    s.t("appendix.point"),
                    s.t("appendix.family"),
                    s.t("appendix.purpose"),
                    s.t("appendix.c"),
                    s.t("method.reasons"),
                ),
                tuple(
                    (
                        code(p.key.id),
                        ctx.variant(p.candidate),
                        s.label("purpose", p.key.purpose),
                        str(p.key.concurrency),
                        s.t("common.list_sep").join(
                            localize(r) for r in p.audit.reasons
                        )
                        or DASH,
                    )
                    for p in invalid
                ),
                numeric=(3,),
                caption=s.t("method.invalid_caption"),
            )
        )
    out.extend(_deploy_blocks(ctx))
    items: list[Inline] = [inline(localize(c)) for c in sm.conditions]
    if sm.dropped_traces is None:
        items.append(s.r("method.dropped_unknown"))
    else:
        items.append(s.r("method.dropped", n=sm.dropped_traces))
    out.append(Paragraph(s.r("method.conditions")))
    out.append(Bullets(tuple(items)))
    return out


def _deploy_blocks(ctx: Ctx) -> list[Block]:
    s, sm = ctx.s, ctx.summary
    ok = [d for d in sm.deploys if d.ok]
    failed = [d for d in sm.deploys if not d.ok]
    load = facts.mean_or_none(d.load_seconds for d in ok)
    out: list[Block] = [
        Paragraph(
            s.r(
                "method.deploys",
                total=len(sm.deploys),
                ok=len(ok),
                failed=len(failed),
                load=integer(load),
            )
        )
    ]
    if failed:
        out.append(
            Table(
                (
                    s.t("method.candidate"),
                    s.t("appendix.family"),
                    s.t("method.failure"),
                    s.t("method.detail"),
                ),
                tuple(
                    (
                        code(d.cand_fp),
                        ctx.variant(d.candidate),
                        s.label("failure", d.failure or "unknown"),
                        truncate(d.detail) or DASH,
                    )
                    for d in failed
                ),
                caption=s.t("method.deploy_caption"),
            )
        )
    return out


# --------------------------------------------------------------------------
# 4 Topology comparison
# --------------------------------------------------------------------------


def _curve_series(ctx: Ctx) -> tuple[list[svg.Series], list[svg.Series]]:
    s, sm = ctx.s, ctx.summary
    curves = ctx.curves[: svg.MAX_SERIES]
    lead = facts.leader_point(sm)
    target: tuple[int, int] | None = None
    for ci, curve in enumerate(curves):
        for pi, cp in enumerate(curve.points):
            if lead is not None and lead.key.id in cp.point_ids:
                target = (ci, pi)
    if target is None:
        best = -1.0
        for ci, curve in enumerate(curves):
            for pi, cp in enumerate(curve.points):
                if cp.tps_per_gpu > best:
                    best, target = cp.tps_per_gpu, (ci, pi)
    by_conc: list[svg.Series] = []
    by_user: list[svg.Series] = []
    for ci, curve in enumerate(curves):
        name = ctx.variant(curve.candidate)
        conc_points, user_points = [], []
        for pi, cp in enumerate(curve.points):
            tip = s.t(
                "topology.tip",
                family=name,
                c=cp.concurrency,
                tps=num(cp.tps_per_gpu),
                user=num(cp.user_tps_p50),
            )
            label = (
                s.t("topology.best_label", v=num(cp.tps_per_gpu))
                if target == (ci, pi)
                else ""
            )
            conc_points.append(svg.Point(cp.concurrency, cp.tps_per_gpu, tip, label))
            if cp.user_tps_p50 is not None:
                user_points.append(
                    svg.Point(cp.user_tps_p50, cp.tps_per_gpu, tip, label)
                )
        by_conc.append(svg.Series(name, tuple(conc_points)))
        by_user.append(svg.Series(name, tuple(user_points)))
    return by_conc, by_user


def topology_section(ctx: Ctx) -> list[Block]:
    """Section 4: topology families, curves and curve data."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [
        Heading(s.t("sections.topology"), 2, "sec-5"),
        Paragraph(s.r("topology.intro")),
    ]
    if sm.families:
        out.append(
            Table(
                (
                    s.t("topology.family"),
                    s.t("topology.status"),
                    s.t("topology.best"),
                    s.t("topology.concurrency"),
                    s.t("topology.saturated"),
                    s.t("topology.cross_node"),
                    s.t("topology.note"),
                ),
                tuple(
                    (
                        inline(strong(f.title), " ", code(f.family_id)),
                        s.label("status", f.status)
                        + (
                            s.t("topology.accuracy_fail")
                            if f.accuracy == "fail"
                            else ""
                        ),
                        num(f.best_tps_per_gpu),
                        conc(f.best_concurrency),
                        DASH if f.saturated is None else ctx.tri(f.saturated),
                        gbs(f.cross_node_gbs),
                        truncate(localize(f.reason)) or DASH,
                    )
                    for f in sm.families
                ),
                numeric=(2, 3, 5),
            )
        )
    if not ctx.curves:
        out.append(Paragraph(s.r("topology.no_curves"), tone="note"))
        return out
    by_conc, by_user = _curve_series(ctx)
    out.append(
        Figure(
            "fig-tps-concurrency",
            svg.line_chart(
                "fig-tps-concurrency",
                by_conc,
                title=s.t("topology.chart_c_title"),
                x_label=s.t("topology.x_concurrency"),
                y_label=s.t("topology.y_tps"),
                log2_x=True,
                empty_text=s.t("topology.no_curves"),
            ),
            s.t("topology.chart_c_caption"),
        )
    )
    refs = [
        svg.RefLine(
            f.user_tps_floor, s.t("topology.floor_ref", floor=num(f.user_tps_floor, 0))
        )
        for f in _sorted_floors(sm)
    ]
    out.append(
        Figure(
            "fig-tps-user",
            svg.line_chart(
                "fig-tps-user",
                by_user,
                title=s.t("topology.chart_u_title"),
                x_label=s.t("topology.x_user"),
                y_label=s.t("topology.y_tps"),
                x_refs=refs,
                empty_text=s.t("topology.no_curves"),
            ),
            s.t("topology.chart_u_caption"),
        )
    )
    rows = []
    for curve in ctx.curves:
        name = ctx.variant(curve.candidate)
        for cp in curve.points:
            rows.append(
                (
                    name,
                    str(cp.concurrency),
                    num(cp.tps_per_gpu),
                    num(cp.user_tps_p50),
                    ms(cp.ttft_p50_ms),
                    ms(cp.itl_p50_ms),
                    ratio_pct(cp.error_rate, 2),
                    gbs(cp.cross_node_gbs),
                    str(cp.reps),
                )
            )
    out.append(
        Table(
            (
                s.t("topology.family"),
                s.t("topology.concurrency"),
                s.t("topology.y_tps"),
                s.t("topology.user"),
                s.t("topology.ttft"),
                s.t("topology.itl"),
                s.t("topology.errors"),
                s.t("topology.cross_node"),
                s.t("topology.reps"),
            ),
            tuple(rows),
            numeric=(1, 2, 3, 4, 5, 6, 7, 8),
            caption=s.t("topology.data_caption"),
        )
    )
    return out


# --------------------------------------------------------------------------
# 5 MTP on/off
# --------------------------------------------------------------------------


def mtp_section(ctx: Ctx) -> list[Block]:
    """Section 5: MTP on/off."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [
        Heading(s.t("sections.mtp"), 2, "sec-6"),
        Paragraph(s.r("mtp.intro")),
    ]
    pairs = facts.mtp_pairs(sm)
    mtp = sm.mtp
    if mtp is None and not pairs:
        out.append(Paragraph(s.r("mtp.none"), tone="note"))
    if mtp is not None:
        anchor = sm.point(mtp.on_point_id) or sm.point(mtp.off_point_id)
        out.append(
            Paragraph(
                s.r(
                    "mtp.headline",
                    family=ctx.family(anchor.candidate.family_id) if anchor else DASH,
                    on=strong(ctx.tps(mtp.on_tps_per_gpu)),
                    off=strong(ctx.tps(mtp.off_tps_per_gpu)),
                    delta=strong(pct(mtp.delta_pct, signed=True)),
                    noise=ctx.noise(mtp.within_noise),
                    accept=num(mtp.accept_len, 2),
                )
            )
        )
    if pairs:
        out.append(
            Table(
                (
                    s.t("mtp.family"),
                    s.t("topology.concurrency"),
                    s.t("mtp.spec"),
                    s.t("mtp.col_on"),
                    s.t("mtp.col_off"),
                    s.t("mtp.delta"),
                    s.t("mtp.noise"),
                    s.t("mtp.accept"),
                ),
                tuple(
                    (
                        ctx.family(p.family_id),
                        str(p.concurrency),
                        code(p.spec),
                        num(p.on_tps),
                        num(p.off_tps),
                        pct(p.delta_pct, signed=True),
                        ctx.noise(facts.within_noise(sm, p.delta_pct)),
                        num(p.accept_len, 2),
                    )
                    for p in pairs
                ),
                numeric=(1, 3, 4, 5, 7),
                caption=s.t("mtp.pairs_caption"),
            )
        )
        names = (s.t("mtp.col_on"), s.t("mtp.col_off"))
        groups = [
            s.t("mtp.group", family=ctx.family(p.family_id), c=p.concurrency)
            for p in pairs
        ]
        tips = [
            [
                s.t(
                    "mtp.tip",
                    family=ctx.family(p.family_id),
                    c=p.concurrency,
                    variant=name,
                    tps=num(value),
                )
                for name, value in zip(names, (p.on_tps, p.off_tps))
            ]
            for p in pairs
        ]
        labels = (
            [[num(p.on_tps), num(p.off_tps)] for p in pairs]
            if len(pairs) <= 4
            else None
        )
        out.append(
            Figure(
                "fig-mtp",
                svg.grouped_bar_chart(
                    "fig-mtp",
                    groups,
                    names,
                    [[p.on_tps, p.off_tps] for p in pairs],
                    title=s.t("mtp.chart_title"),
                    y_label=s.t("mtp.y"),
                    tips=tips,
                    value_labels=labels,
                    empty_text=s.t("mtp.none"),
                ),
                s.t("mtp.chart_caption"),
            )
        )
    simulated = [p for p in sm.points if p.candidate.simulate_acceptance]
    if simulated:
        out.append(Paragraph(s.r("mtp.simulated", n=len(simulated)), tone="note"))
    return out


# --------------------------------------------------------------------------
# 6 Bandwidth
# --------------------------------------------------------------------------


def _stat_value(stat: BandwidthStat, value: float) -> str:
    return DASH if stat.method == "unavailable" else gbs(value)


def _stats_table(ctx: Ctx, stats: Sequence[BandwidthStat], caption: str) -> Table:
    s = ctx.s
    return Table(
        (
            s.t("bandwidth.plane"),
            s.t("bandwidth.direction"),
            s.t("bandwidth.scope"),
            s.t("bandwidth.mean"),
            s.t("bandwidth.p99"),
            s.t("bandwidth.peak"),
            s.t("bandwidth.sol"),
            s.t("bandwidth.util_peak"),
            s.t("bandwidth.util_nominal"),
            s.t("bandwidth.method"),
        ),
        tuple(
            (
                s.label("plane", st.plane),
                s.label("direction", st.direction),
                code(st.scope),
                _stat_value(st, st.mean_gbs),
                _stat_value(st, st.p99_gbs),
                _stat_value(st, st.peak_gbs),
                gbs(st.ref_peak_gbs),
                DASH if st.method == "unavailable" else ratio_pct(st.util_vs_peak),
                DASH if st.method == "unavailable" else ratio_pct(st.util_vs_nominal),
                s.label("method_label", st.method),
            )
            for st in stats
        ),
        numeric=(3, 4, 5, 6, 7, 8),
        caption=caption,
    )


def _method_notes(ctx: Ctx, stats: Sequence[BandwidthStat]) -> list[Block]:
    s = ctx.s
    out: list[Block] = []
    for method, key in (
        ("derived", "bandwidth.derived_note"),
        ("unavailable", "bandwidth.unavailable_note"),
    ):
        planes = sorted(
            {s.label("plane", st.plane) for st in stats if st.method == method}
        )
        if planes:
            text = s.t("common.enum_sep").join(planes)
            out.append(Paragraph(s.r(key, planes=text), tone="note"))
    return out


def _util_chart(ctx: Ctx, stats: Sequence[BandwidthStat]) -> Block:
    s = ctx.s
    usable = [
        st for st in stats if st.ref_peak_gbs and st.method in ("measured", "derived")
    ]
    if not usable:
        return Paragraph(s.r("bandwidth.no_chart"), tone="note")
    names = (s.t("bandwidth.mean"), s.t("bandwidth.p99"), s.t("bandwidth.peak"))
    groups, values, tips, labels = [], [], [], []
    for st in usable:
        ref = st.ref_peak_gbs or 1.0
        plane = s.label("plane", st.plane)
        direction = s.label("direction", st.direction)
        groups.append(s.t("bandwidth.group", plane=plane, direction=direction))
        triple = (st.mean_gbs, st.p99_gbs, st.peak_gbs)
        values.append([v / ref * 100.0 for v in triple])
        tips.append(
            [
                s.t(
                    "bandwidth.tip",
                    plane=plane,
                    direction=direction,
                    stat=name,
                    gbs=gbs(v),
                    sol=gbs(ref),
                )
                for name, v in zip(names, triple)
            ]
        )
        labels.append(["", "", pct(st.peak_gbs / ref * 100.0, 0)])
    return Figure(
        "fig-bw-util",
        svg.grouped_bar_chart(
            "fig-bw-util",
            groups,
            names,
            values,
            title=s.t("bandwidth.chart_title"),
            y_label=s.t("bandwidth.y"),
            tips=tips,
            value_labels=labels,
            ref=svg.RefLine(100.0, s.t("bandwidth.sol_ref")),
        ),
        s.t("bandwidth.chart_caption"),
    )


def bandwidth_section(ctx: Ctx) -> list[Block]:
    """Section 6: speed-of-light peaks and under-load occupancy."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [
        Heading(s.t("sections.bandwidth"), 2, "sec-7"),
        Paragraph(s.r("bandwidth.intro")),
    ]
    if sm.peaks:
        peaks = sorted(
            sm.peaks,
            key=lambda p: (
                facts.PLANE_ORDER.index(p.plane),
                p.test,
                -(p.nic_count or 0),
                p.scope,
            ),
        )
        out.append(
            Table(
                (
                    s.t("bandwidth.plane"),
                    s.t("bandwidth.test"),
                    s.t("bandwidth.scope"),
                    s.t("bandwidth.nic_count"),
                    s.t("bandwidth.measured_gbs"),
                    s.t("bandwidth.nominal_gbs"),
                    s.t("bandwidth.of_nominal"),
                    s.t("bandwidth.tool"),
                ),
                tuple(
                    (
                        s.label("plane", p.plane),
                        code(p.test),
                        code(p.scope),
                        DASH if p.nic_count is None else str(p.nic_count),
                        gbs(p.value_gbs),
                        gbs(p.nominal_gbs),
                        ratio_pct(
                            p.value_gbs / p.nominal_gbs if p.nominal_gbs else None
                        ),
                        p.tool,
                    )
                    for p in peaks
                ),
                numeric=(3, 4, 5, 6),
                caption=s.t("bandwidth.peaks_caption"),
            )
        )
    else:
        out.append(Paragraph(s.r("bandwidth.no_peaks"), tone="note"))
    lead = facts.leader_point(sm)
    if lead is None:
        out.append(Paragraph(s.r("bandwidth.no_leader"), tone="note"))
        return out
    if lead.cross_node_gbs is not None:
        out.append(
            Paragraph(
                s.r("bandwidth.cross_node", rate=strong(gbs(lead.cross_node_gbs)))
            )
        )
    if not lead.bandwidth:
        out.append(Paragraph(s.r("bandwidth.no_stats"), tone="note"))
    else:
        busiest = facts.busiest_stats(lead.bandwidth)
        out.append(_stats_table(ctx, busiest, s.t("bandwidth.leader_caption")))
        out.extend(_method_notes(ctx, lead.bandwidth))
        out.append(_util_chart(ctx, busiest))
    if facts.leader_cross_node(sm, ctx.plan) is False:
        full = facts.full_nic_arm(sm.nic_arms)
        point = sm.point(full.point_id) if full is not None else None
        if point is not None and point.bandwidth:
            caption = s.t(
                "bandwidth.cross_caption",
                family=ctx.family(point.candidate.family_id),
                c=point.key.concurrency,
            )
            out.append(_stats_table(ctx, facts.busiest_stats(point.bandwidth), caption))
    if lead.bandwidth:
        stats = facts.sort_stats(lead.bandwidth)
        out.append(
            _stats_table(ctx, stats[:MAX_DETAIL_ROWS], s.t("bandwidth.details_caption"))
        )
        if len(stats) > MAX_DETAIL_ROWS:
            out.append(
                Paragraph(
                    s.r(
                        "meta.rows_omitted",
                        n=len(stats) - MAX_DETAIL_ROWS,
                        file=code("records/util.jsonl"),
                    ),
                    tone="note",
                )
            )
    return out


# --------------------------------------------------------------------------
# 7 NIC count
# --------------------------------------------------------------------------


def nic_section(ctx: Ctx) -> list[Block]:
    """Section 7: NIC-count comparison."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [
        Heading(s.t("sections.nic"), 2, "sec-8"),
        Paragraph(s.r("nic.intro")),
    ]
    arms = facts.nic_arms_sorted(sm.nic_arms)
    if not arms:
        out.append(
            Paragraph(s.r("nic.none", reason=localize(sm.nic_reason)), tone="note")
        )
        return out
    cross = facts.leader_cross_node(sm, ctx.plan)
    leader = facts.leader_candidate(sm)
    leader_swept = (
        bool(cross) and leader is not None and leader.family_id == sm.nic_family_id
    )
    family, c = strong(ctx.family(sm.nic_family_id)), conc(sm.nic_concurrency)
    if cross is False or leader_swept:
        # the note below explains the choice; the planner reason would repeat it
        out.append(Paragraph(s.r("nic.sweep_plain", family=family, c=c)))
    else:
        out.append(
            Paragraph(
                s.r("nic.sweep", family=family, c=c, reason=localize(sm.nic_reason))
            )
        )
    if cross is False:
        out.append(
            Paragraph(
                s.r(
                    "nic.irrelevant",
                    leader=strong(ctx.variant(leader)) if leader else DASH,
                    family=strong(ctx.family(sm.nic_family_id)),
                ),
                tone="note",
            )
        )
    elif leader_swept:
        out.append(Paragraph(s.r("nic.leader_cross")))
    full = facts.full_nic_arm(arms)
    reference = s.t("nic.reference")
    out.append(
        Table(
            (
                s.t("nic.count"),
                s.t("nic.status"),
                s.t("nic.tps"),
                s.t("nic.delta"),
                s.t("nic.noise"),
                s.t("nic.effective"),
                s.t("nic.engaged"),
                s.t("nic.staging"),
            ),
            tuple(
                (
                    str(a.nic_count),
                    s.label("nic_status", a.status),
                    num(a.tps_per_gpu),
                    reference if a is full else pct(a.delta_vs_full_pct, signed=True),
                    reference if a is full else ctx.tri(a.within_noise),
                    num(a.nic_effective, 2),
                    ctx.tri(a.engaged),
                    ctx.tri(a.staging_suspected),
                )
                for a in arms
            ),
            numeric=(0, 2, 3, 5),
        )
    )
    bars = []
    for arm in arms:
        is_full = full is not None and arm.nic_count == full.nic_count
        value_label = num(arm.tps_per_gpu)
        if arm.tps_per_gpu is None:
            value_label = s.label("nic_status", arm.status)
        elif not is_full and arm.delta_vs_full_pct is not None:
            value_label += f"（{pct(arm.delta_vs_full_pct, signed=True)}）"
        bars.append(
            svg.Bar(
                label=s.t("nic.bar", n=arm.nic_count),
                value=arm.tps_per_gpu,
                value_label=value_label,
                tip=s.t(
                    "nic.tip",
                    n=arm.nic_count,
                    tps=num(arm.tps_per_gpu),
                    delta=pct(arm.delta_vs_full_pct, signed=True),
                ),
                muted=arm.status != "measured" or arm.engaged is False,
            )
        )
    out.append(
        Figure(
            "fig-nic",
            svg.hbar_chart(
                "fig-nic",
                bars,
                title=s.t("nic.chart_title"),
                x_label=s.t("nic.x"),
            ),
            s.t(
                "nic.chart_caption",
                family=ctx.family(sm.nic_family_id),
                c=conc(sm.nic_concurrency),
            ),
        )
    )
    if any(a.staging_suspected for a in arms):
        out.append(Paragraph(s.r("nic.staging_note"), tone="warn"))
    if any(a.status == "not_engaged" or a.engaged is False for a in arms):
        out.append(Paragraph(s.r("nic.not_engaged_note"), tone="warn"))
    out.append(Paragraph(s.r("nic.effective_note")))
    return out


# --------------------------------------------------------------------------
# 8 Recommended configuration
# --------------------------------------------------------------------------


def _yaml_text(config: dict) -> str:
    plain = json.loads(json.dumps(config, default=str))
    return yaml.safe_dump(
        plain, sort_keys=True, allow_unicode=True, default_flow_style=False
    )


def config_section(ctx: Ctx) -> list[Block]:
    """Section 8: effective configuration of the leader."""
    s, sm, plan = ctx.s, ctx.summary, ctx.plan
    out: list[Block] = [Heading(s.t("sections.config"), 2, "sec-12")]
    cand = facts.leader_candidate(sm)
    if cand is None:
        out.append(Paragraph(s.r("config.none"), tone="note"))
        return out
    lead = facts.leader_point(sm)
    if plan is not None:
        gpus = str(plan.gpu_count)
        transport: Cell = code(plan.transport)
        cross = ", ".join(plan.cross_node) or s.t("common.none")
    else:
        gpus = str(lead.gpu_count) if lead else DASH
        transport, cross = DASH, DASH
    out.append(
        Paragraph(
            s.r(
                "config.candidate",
                key=code(cand.key),
                fp=code(cand.fp),
                family=strong(ctx.family(cand.family_id)),
                mtp=ctx.mtp(cand.mtp),
                nics=ctx.nics(cand.nic_count),
                gpus=gpus,
                transport=transport,
                cross=cross,
            )
        )
    )
    check = sm.accuracy_of(cand.key)
    if check is not None and check.result.accuracy is not None:
        out.append(
            Paragraph(
                s.r(
                    "config.correctness_checked",
                    status=s.label("accuracy_status", check.verdict.status),
                    acc=ratio_pct(check.result.accuracy, 1),
                    n=str(check.result.n),
                )
            )
        )
    else:
        out.append(Paragraph(s.r("config.correctness")))
    if plan is None:
        out.append(Paragraph(s.r("config.no_plan", fp=cand.fp), tone="warn"))
        return out
    out.append(Paragraph(s.r("config.intro", fp=cand.fp)))
    out.append(
        Table(
            (
                s.t("config.node"),
                s.t("config.container"),
                s.t("config.role"),
                s.t("config.instance"),
                s.t("config.rank"),
                s.t("config.gpus"),
                s.t("config.hcas"),
            ),
            tuple(
                (
                    c.node,
                    code(c.name),
                    s.label("role", c.role),
                    str(c.instance),
                    str(c.rank),
                    ",".join(str(g) for g in c.gpus) or DASH,
                    ",".join(c.hcas) or DASH,
                )
                for c in sorted_containers(plan)
            ),
            numeric=(3, 4),
            caption=s.t("config.placement_caption"),
        )
    )
    for view in role_views(plan):
        if view.role == "control":
            continue
        out.append(
            Heading(
                s.t(
                    "config.role_heading",
                    role=s.label("role", view.role),
                    instances=view.instances,
                    containers=view.containers,
                ),
                3,
            )
        )
        if view.sglang:
            out.append(Paragraph(s.r("config.sglang")))
            out.append(Code(_yaml_text(view.sglang), "yaml"))
            if view.config_variants > 1:
                out.append(
                    Paragraph(
                        s.r("config.variants", n=view.config_variants), tone="note"
                    )
                )
        elif view.role in SERVING_ROLES:
            out.append(Paragraph(s.r("config.no_config"), tone="note"))
        out.append(Paragraph(s.r("config.env", name=code(view.representative))))
        env_text = "\n".join(f"{k}={shlex.quote(v)}" for k, v in view.env.items())
        out.append(Code(env_text or DASH, "bash"))
        out.append(Paragraph(s.r("config.command")))
        out.append(Code(pretty_argv(view.command) or DASH, "bash"))
    return out


# --------------------------------------------------------------------------
# 9 Limitations, 10 Reproduction, appendix
# --------------------------------------------------------------------------


def limits_section(ctx: Ctx) -> list[Block]:
    """Section 9: limitations and unverified items."""
    s, sm = ctx.s, ctx.summary
    items: list[Inline] = []
    lead = facts.leader_point(sm)
    if lead is None:
        items.append(s.r("limits.no_leader", verdict=s.label("verdict", sm.verdict)))
    if lead is not None and sm.saturated is False:
        items.append(
            s.r("limits.not_saturated", note=localize(sm.saturation_note) or DASH)
        )
    if sm.points and sm.noise is None:
        items.append(s.r("limits.no_noise"))
    if lead is not None and sm.confirm_point_id is None:
        items.append(s.r("limits.no_confirm"))
    if any(p.key.spec == "screen" for p in sm.points):
        items.append(s.r("limits.screen"))
    if sm.mtp is not None:
        items.append(s.r("limits.measured_mtp"))
    if sm.dropped_traces:
        items.append(s.r("limits.dropped", n=sm.dropped_traces))
    for family in sm.families:
        if family.status not in ("measured", "disabled"):
            items.append(
                s.r(
                    "limits.family_status",
                    family=family.title,
                    status=s.label("status", family.status),
                    reason=localize(family.reason) or DASH,
                )
            )
    if not sm.nic_arms and sm.nic_reason:
        items.append(s.r("limits.nic_skipped", reason=localize(sm.nic_reason)))
    items.extend(inline(localize(text)) for text in sm.limitations)
    items.extend(
        s.r("limits.unverified", item=localize(text)) for text in sm.unverified
    )
    if not items:
        items.append(s.r("limits.none"))
    return [
        Heading(s.t("sections.limits"), 2, "sec-13"),
        Paragraph(s.r("limits.intro")),
        Bullets(tuple(items)),
    ]


def repro_section(ctx: Ctx) -> list[Block]:
    """Section 10: how to reproduce."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [
        Heading(s.t("sections.repro"), 2, "sec-14"),
        Paragraph(s.r("repro.intro")),
    ]
    for command in repro_commands(sm):
        out.append(Paragraph(s.r(f"repro.{command.purpose}")))
        out.append(Code(command.shell, "bash"))
    out.append(
        Paragraph(
            s.r(
                "repro.artifacts",
                events=code("events.jsonl"),
                summary=code("analysis/summary.json"),
                plans=code("deployments/<cand_fp>/plan.json"),
                points=code("points/<point_id>/"),
            )
        )
    )
    lead = facts.leader_point(sm)
    if lead is not None and lead.artifact_dir:
        out.append(
            Paragraph(s.r("repro.leader_artifacts", dir=code(lead.artifact_dir)))
        )
    return out


def appendix_section(ctx: Ctx) -> list[Block]:
    """Appendix A: every measured point."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [
        Heading(s.t("sections.appendix"), 2, "sec-a"),
        Paragraph(s.r("appendix.intro")),
    ]
    if not sm.points:
        out.append(Paragraph(s.r("appendix.none"), tone="note"))
        return out
    rows = []
    for p in sorted(sm.points, key=lambda point: facts.point_sort_key(sm, point)):
        metrics = p.metrics
        rows.append(
            (
                code(p.key.id),
                ctx.family(p.candidate.family_id),
                s.label("purpose", p.key.purpose),
                code(p.key.spec),
                str(p.key.concurrency),
                str(p.key.rep),
                ctx.mtp(p.candidate.mtp),
                ctx.nics(p.candidate.nic_count),
                num(p.tps_per_gpu),
                num(metrics.user_tps_p50 if metrics else None),
                ms(metrics.ttft_p50_ms if metrics else None),
                ms(metrics.itl_p50_ms if metrics else None),
                ratio_pct(facts.error_rate(p), 2),
                gbs(p.cross_node_gbs),
                s.label("audit", p.audit.status),
            )
        )
    out.append(
        Table(
            (
                s.t("appendix.point"),
                s.t("appendix.family"),
                s.t("appendix.purpose"),
                s.t("appendix.spec"),
                s.t("appendix.c"),
                s.t("appendix.rep"),
                s.t("appendix.mtp"),
                s.t("appendix.nics"),
                s.t("appendix.tps"),
                s.t("appendix.user"),
                s.t("appendix.ttft"),
                s.t("appendix.itl"),
                s.t("appendix.errors"),
                s.t("appendix.cross"),
                s.t("appendix.audit"),
            ),
            tuple(rows),
            numeric=(4, 5, 8, 9, 10, 11, 12, 13),
        )
    )
    return out
