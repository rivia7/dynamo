# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report sections 8-10: network topology and switches, deployment and
runtime state, node health and environment.

Like ``sections.py`` every builder reads only the ``ExperimentSummary`` and
says so explicitly when a collector produced nothing (locked-down fabrics,
missing tools and old summaries are normal).
"""

from __future__ import annotations

import re
from statistics import median
from typing import TYPE_CHECKING, Sequence

from benchmarks.lium_cluster_bench.domain.fabric import (
    FabricTopology,
    LinkFinding,
    LinkMatrix,
)
from benchmarks.lium_cluster_bench.domain.kernels import KernelBreakdown
from benchmarks.lium_cluster_bench.domain.pointstate import PointState
from benchmarks.lium_cluster_bench.domain.schema import PointResult
from benchmarks.lium_cluster_bench.domain.switchload import LinkClassLoad
from benchmarks.lium_cluster_bench.report import facts
from benchmarks.lium_cluster_bench.report.doc import (
    Block,
    Bullets,
    Cell,
    Heading,
    Inline,
    Paragraph,
    Table,
    code,
    inline,
)
from benchmarks.lium_cluster_bench.report.fmt import DASH, gbs, integer, num, ratio_pct
from benchmarks.lium_cluster_bench.report.reasons import REGIMES, localize, localize_activity

if TYPE_CHECKING:
    from benchmarks.lium_cluster_bench.report.sections import Ctx

MAX_ROWS = 40
SERVING = ("agg", "prefill", "decode")
KEY_PACKAGES = (
    "sglang",
    "sgl-kernel",
    "flashinfer-python",
    "torch",
    "ai-dynamo",
    "ai-dynamo-runtime",
    "nixl",
    "mooncake-transfer-engine",
    "deep-ep",
    "triton",
    "transformers",
)


def _rate_text(topo: FabricTopology) -> str:
    rates = sorted({r.rate_gbps for r in topo.rails if r.rate_gbps})
    return "/".join(f"{r:g}" for r in rates) + " Gb/s" if rates else DASH


# --------------------------------------------------------------------------
# 8 Network topology and switches
# --------------------------------------------------------------------------


def _topology_rows(ctx: Ctx, topo: FabricTopology) -> list[tuple[Cell, ...]]:
    s, sig = ctx.s, ctx.summary.fabric_signature
    over = sig.get("max_oversubscription")
    pfc = sig.get("pfc_priorities") or []
    rows: list[tuple[Cell, ...]] = [
        (s.t("fabric.kind"), s.label("fabric_kind", topo.kind)),
        (
            s.t("fabric.rails"),
            s.t("fabric.rails_value", n=len(topo.rails), rate=_rate_text(topo)),
        ),
        (s.t("fabric.rail_optimized"), ctx.tri(topo.rail_optimized)),
        (s.t("fabric.nodes_per_leaf"), integer(topo.nodes_per_leaf)),
        (
            s.t("fabric.switches"),
            s.t(
                "fabric.switches_value",
                leaves=sig.get("leaves", 0),
                spines=sig.get("spines", 0),
            ),
        ),
        (
            s.t("fabric.hops"),
            s.t(
                "fabric.hops_value",
                hops=integer(topo.max_hops),
                tiers=integer(topo.tiers),
                method="/".join(sorted(set(topo.hop_method.values()))) or DASH,
            ),
        ),
        (
            s.t("fabric.oversub"),
            s.t("fabric.oversub_value", ratio=num(over, 2)) if over else DASH,
        ),
        (
            s.t("fabric.subnets"),
            s.t(
                "fabric.subnets_value",
                isolated=ctx.tri(topo.rails_isolated),
                sms=len(topo.sms),
            ),
        ),
        (s.t("fabric.degraded"), integer(len(topo.degraded))),
    ]
    if pfc:
        rows.append((s.t("fabric.pfc"), ",".join(str(p) for p in pfc)))
    tiers = sig.get("tiers_seen") or []
    rows.append(
        (
            s.t("fabric.tiers_seen"),
            ", ".join(s.label("fabric_tier", t) for t in tiers) or DASH,
        )
    )
    return rows


def _leaf_rows(topo: FabricTopology) -> list[tuple[Cell, ...]]:
    names = {sid: sw.desc or sid for sid, sw in topo.switches.items()}
    rows: list[tuple[Cell, ...]] = []
    for rail in topo.rails:
        groups = topo.leaf_groups.get(rail.rail, ())
        leaves = ", ".join(names.get(x, x) for x in rail.leaves) or DASH
        rows.append(
            (
                str(rail.rail),
                f"{rail.rate_gbps:g} Gb/s" if rail.rate_gbps else DASH,
                " | ".join(",".join(g) for g in groups) or DASH,
                leaves,
            )
        )
    return rows


def _hop_table(ctx: Ctx, topo: FabricTopology) -> Table | None:
    if not topo.hops:
        return None
    rail = min(topo.hops)
    matrix = topo.hops[rail]
    nodes = list(topo.nodes)
    rows = []
    for a in nodes:
        row: list[Cell] = [a]
        for b in nodes:
            if a == b:
                row.append("·")
                continue
            hops = matrix.get(a, {}).get(b, matrix.get(b, {}).get(a))
            row.append(DASH if hops is None else str(hops))
        rows.append(tuple(row))
    same = all(topo.hops[r] == matrix for r in topo.hops)
    caption = ctx.s.t(
        "fabric.hops_caption_same" if same else "fabric.hops_caption",
        rail=rail,
        method=topo.hop_method.get(rail, DASH),
    )
    return Table(("", *nodes), tuple(rows), tuple(range(1, len(nodes) + 1)), caption)


def _link_rows(ctx: Ctx, links: LinkMatrix, topo: FabricTopology) -> list[tuple]:
    nominal = {r.rail: r.rate_gbps / 8 for r in topo.rails if r.rate_gbps}
    rows = []
    for rail in links.rails:
        tests = [t for t in links.tests if t.rail == rail and not t.error]
        bws = [t.bw_gbs for t in tests if t.bw_gbs is not None]
        lats = [t.lat_typical_us for t in tests if t.lat_typical_us is not None]
        line = nominal.get(rail)
        rows.append(
            (
                str(rail),
                str(len(tests)),
                gbs(median(bws)) if bws else DASH,
                gbs(min(bws)) if bws else DASH,
                ratio_pct(median(bws) / line) if bws and line else DASH,
                num(median(lats), 2) if lats else DASH,
                num(max(lats), 2) if lats else DASH,
            )
        )
    return rows


def _class_row(ctx: Ctx, label: str, load: LinkClassLoad | None) -> tuple[Cell, ...]:
    if load is None:
        return (label, "0", DASH, DASH, DASH, DASH)
    return (
        label,
        str(load.ports),
        ratio_pct(load.util_max),
        ratio_pct(load.util_mean),
        gbs(load.tx_gbs + load.rx_gbs),
        ratio_pct(load.stall_fraction_max, 2),
    )


def _switch_load_blocks(ctx: Ctx, lead: PointResult | None) -> list[Block]:
    s = ctx.s
    load = lead.switch_load if lead is not None else None
    if load is None:
        return [Paragraph(s.r("fabric.no_switch_load"), tone="note")]
    out: list[Block] = [
        Paragraph(
            s.r(
                "fabric.switch_load_intro",
                c=lead.key.concurrency,
                family=ctx.family(lead.candidate.family_id),
                n=load.measured,
            )
        ),
        Table(
            (
                s.t("fabric.port_class"),
                s.t("fabric.ports"),
                s.t("fabric.util_max"),
                s.t("fabric.util_mean"),
                s.t("fabric.traffic"),
                s.t("fabric.stall"),
            ),
            (
                _class_row(ctx, s.t("fabric.uplinks"), load.uplink),
                _class_row(ctx, s.t("fabric.downlinks"), load.downlink),
            ),
            (1, 2, 3, 4, 5),
        ),
    ]
    if load.hottest:
        out.append(
            Table(
                (
                    s.t("fabric.switch"),
                    s.t("fabric.port"),
                    s.t("fabric.role"),
                    s.t("fabric.peer"),
                    s.t("fabric.util_max"),
                    "TX/RX GB/s",
                ),
                tuple(
                    (
                        h.switch,
                        str(h.port),
                        s.label("port_role", h.role),
                        h.peer or DASH,
                        ratio_pct(h.util),
                        f"{gbs(h.tx_gbs)} / {gbs(h.rx_gbs)}",
                    )
                    for h in load.hottest
                ),
                (1, 4),
                s.t("fabric.hottest_caption"),
            )
        )
    if load.discards or load.errors:
        errors = ", ".join(f"{k}={v}" for k, v in load.errors.items()) or DASH
        out.append(
            Paragraph(
                s.r("fabric.switch_errors", discards=load.discards, errors=errors),
                tone="warn",
            )
        )
    return out


def _distinct_link_findings(findings: Sequence[LinkFinding]) -> list[LinkFinding]:
    """Problems first; one latency-group line stands for identical rails."""
    problems = [f for f in findings if f.kind != "latency_groups"]
    groups = [f for f in findings if f.kind == "latency_groups"]
    shown: list[LinkFinding] = []
    seen: set[str] = set()
    for finding in groups:
        shape = finding.detail.split(":", 1)[-1]
        shape = re.sub(r"~[\d.]+ us", "", shape)
        if shape not in seen:
            seen.add(shape)
            shown.append(finding)
    return problems + shown


def fabric_section(ctx: Ctx) -> list[Block]:
    """Section 8: upstream switches, rails, paths, link tests, port load."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.fabric"), 2, "sec-9")]
    topo = sm.fabric
    if topo is None:
        out.append(Paragraph(s.r("fabric.none"), tone="note"))
        return out
    out.append(Paragraph(s.r("fabric.intro")))
    out.append(
        Table(
            (s.t("fabric.key"), s.t("fabric.value")), tuple(_topology_rows(ctx, topo))
        )
    )
    out.append(
        Table(
            (
                s.t("fabric.rail"),
                s.t("fabric.rate"),
                s.t("fabric.leaf_groups"),
                s.t("fabric.leaf_names"),
            ),
            tuple(_leaf_rows(topo)),
            (0,),
            s.t("fabric.leaf_caption"),
        )
    )
    hop_table = _hop_table(ctx, topo)
    if hop_table is not None:
        out.append(hop_table)
    links = sm.fabric_links
    if links is not None and links.tests:
        out.append(Paragraph(s.r("fabric.links_intro", rounds=links.rounds)))
        out.append(
            Table(
                (
                    s.t("fabric.rail"),
                    s.t("fabric.tests"),
                    s.t("fabric.bw_median"),
                    s.t("fabric.bw_min"),
                    s.t("fabric.bw_vs_line"),
                    s.t("fabric.lat_median"),
                    s.t("fabric.lat_max"),
                ),
                tuple(_link_rows(ctx, links, topo)),
                (0, 1, 2, 3, 4, 5, 6),
            )
        )
    elif links is not None and links.skipped:
        out.append(Paragraph(s.r("fabric.links_skipped"), tone="note"))
    findings: list[Inline] = [
        inline(localize(f.detail)) for f in _distinct_link_findings(sm.link_findings)
    ]
    findings += [
        inline(localize(text))
        for text in sm.fabric_findings
        if not text.startswith("coverage:")
    ]
    if findings:
        out.append(Bullets(tuple(findings[:MAX_ROWS])))
    coverage = [
        t.removeprefix("coverage: ")
        for t in sm.fabric_findings
        if t.startswith("coverage:")
    ]
    if coverage:
        out.append(
            Paragraph(
                s.r("fabric.coverage", items="；".join(localize(c) for c in coverage)),
                tone="note",
            )
        )
    out.append(Heading(s.t("fabric.load_heading"), 3))
    out.extend(_switch_load_blocks(ctx, facts.leader_point(sm)))
    return out


# --------------------------------------------------------------------------
# 9 Deployment and runtime state
# --------------------------------------------------------------------------


def _lifecycle_rows(ctx: Ctx) -> list[tuple[Cell, ...]]:
    sm = ctx.summary
    rows: list[tuple[Cell, ...]] = []
    seen: set[str] = set()
    for deploy in sm.deploys:
        life = deploy.lifecycle
        if life is None or deploy.cand_fp in seen:
            continue
        seen.add(deploy.cand_fp)
        roles = [life.roles[r] for r in SERVING if r in life.roles]
        kv = [r.kv_tokens_min or r.max_total_num_tokens_min for r in roles]
        kv_known = [k for k in kv if k]
        mrr = [r.max_running_requests_min for r in roles if r.max_running_requests_min]
        weights = [r.weights_load_s_max for r in roles if r.weights_load_s_max]
        graphs = [
            r.cuda_graph_capture_s_max for r in roles if r.cuda_graph_capture_s_max
        ]
        issues = sum(life.issues.values())
        rows.append(
            (
                ctx.variant(deploy.candidate),
                ctx.s.label("deploy_ok", deploy.ok),
                num(life.time_to_ready_s or deploy.load_seconds, 0),
                num(max(weights), 0) if weights else DASH,
                num(max(graphs), 0) if graphs else DASH,
                integer(min(kv_known)) if kv_known else DASH,
                integer(min(mrr)) if mrr else DASH,
                integer(issues) if issues else "0",
            )
        )
        if len(rows) >= MAX_ROWS:
            break
    return rows


def _role_rows(ctx: Ctx, state: PointState) -> list[tuple[Cell, ...]]:
    rows: list[tuple[Cell, ...]] = []
    for role in (*SERVING, "frontend"):
        brief = state.roles.get(role)
        if brief is None:
            continue
        if role == "frontend":
            rows.append(
                (
                    ctx.s.label("role", role),
                    num(brief.get("fe_inflight"), 0),
                    num(brief.get("fe_queued"), 0),
                    DASH,
                    ratio_pct(brief.get("router_kv_hit_rate")),
                    DASH,
                )
            )
            continue
        kv = brief.get("kv_usage") or brief.get("kv_usage_dynamo")
        kv_max = brief.get("kv_usage", "max") or brief.get("kv_usage_dynamo", "max")
        rows.append(
            (
                f"{ctx.s.label('role', role)} ×{brief.endpoints}",
                f"{num(brief.get('running_reqs'), 0)} / {num(brief.get('running_reqs', 'max'), 0)}",
                f"{num(brief.get('queue_reqs'), 0)} / {num(brief.get('queue_reqs', 'max'), 0)}",
                f"{ratio_pct(kv)} / {ratio_pct(kv_max)}",
                ratio_pct(
                    brief.get("cache_hit_rate") or brief.get("kv_hit_rate_dynamo")
                ),
                num(brief.get("gen_throughput"), 0),
            )
        )
    return rows


def _bottleneck_rows(ctx: Ctx) -> list[tuple[Cell, ...]]:
    sm = ctx.summary
    rows: list[tuple[Cell, ...]] = []
    for family in sm.families:
        point = sm.point(family.best_point_id)
        if point is None or point.state is None or point.state.bottleneck is None:
            continue
        b = point.state.bottleneck
        rows.append(
            (
                family.title,
                str(point.key.concurrency),
                ctx.tps(point.tps_per_gpu),
                ctx.s.label("bottleneck", b.label),
                ratio_pct(b.confidence, 0),
                localize(b.evidence[0]) if b.evidence else DASH,
            )
        )
    return rows


def _stress_rows(ctx: Ctx, point: PointResult) -> list[tuple[Cell, ...]]:
    s = ctx.s
    st = point.state.stress if point.state else None
    if st is None:
        return []
    throttle = ", ".join(
        f"{s.label('throttle', k)} {ratio_pct(v)}"
        for k, v in sorted(st.throttle.items())
    )
    return [
        (s.t("runtime.power"), s.t("runtime.power_value", mean=num(st.power_mean_w, 0), peak=num(st.power_max_w, 0))),
        (s.t("runtime.energy_token"), s.t("runtime.energy_value", j=num(point.state.energy_per_token_j, 3))),
        (s.t("runtime.thermal"), s.t("runtime.thermal_value", temp=num(st.temp_max_c, 0), mhz=num(st.sm_mhz_mean, 0))),
        (s.t("runtime.throttle"), f"{ratio_pct(st.throttled_fraction)}（{throttle or DASH}）"),
        (s.t("runtime.ib_wait"), ratio_pct(st.xmit_wait_fraction_max, 2)),
        (
            s.t("runtime.roce"),
            s.t(
                "runtime.roce_value",
                cnp=num(st.cnp_per_s_total, 0),
                ecn=num(st.ecn_per_s_total, 0),
                pause=num(st.pause_per_s_total, 0),
            ),
        ),
        (
            s.t("runtime.fabric_errors"),
            ", ".join(f"{k}={v}" for k, v in st.error_increments.items()) or "0",
        ),
    ]  # fmt: skip


# (catalog key, i18n key) of request-stage latencies shown for PD points.
STAGE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("prefill", "stage_prefill_bootstrap", "runtime.stage_bootstrap"),
    ("prefill", "stage_prefill_forward", "runtime.stage_prefill"),
    ("prefill", "stage_prefill_transfer_kv_cache", "runtime.stage_transfer"),
    ("decode", "stage_decode_waiting", "runtime.stage_decode_wait"),
)


def _pct_text(value: float | None) -> str:
    return DASH if value is None else f"{value:.0f}%"


def _activity_rows(ctx: Ctx, point: PointResult) -> list[tuple[Cell, ...]]:
    """NVML GPM regime per serving role, and GPM availability."""
    s = ctx.s
    st = point.state.stress if point.state else None
    act = st.activity if st else None
    if act is None:
        return []
    rows: list[tuple[Cell, ...]] = []
    for role in (*SERVING, "unassigned"):
        agg = act.roles.get(role)
        if agg is None:
            continue
        rows.append(
            (
                s.t("runtime.activity_role", role=s.label("role", role)),
                s.t(
                    "runtime.activity_value",
                    sm=_pct_text(agg.sm_pct_mean),
                    tensor=_pct_text(agg.tensor_pct_mean),
                    dram=_pct_text(agg.dram_pct_mean),
                    regime=REGIMES.get(agg.regime, agg.regime),
                    reason=localize_activity(agg.reason) if agg.reason else DASH,
                ),
            )
        )
    missing = {k: v for k, v in act.availability.items() if k != "ok"}
    if missing:
        rows.append(
            (
                s.t("runtime.gpm_availability"),
                ", ".join(f"{k} ×{v}" for k, v in sorted(missing.items()))
                + (
                    f"（ok ×{act.availability['ok']}）"
                    if act.availability.get("ok")
                    else ""
                ),
            )
        )
    return rows


def _engine_rows(ctx: Ctx, point: PointResult) -> list[tuple[Cell, ...]]:
    """Engine-internal ratios of the leader (only rows that were measured)."""
    s, state = ctx.s, point.state
    if state is None:
        return []
    rows: list[tuple[Cell, ...]] = []
    graph = state.derived("decode_cuda_graph_fraction")
    if graph is not None:
        rows.append((s.t("runtime.cuda_graph"), ratio_pct(graph)))
    if point.accept_len is not None:
        rate = state.value("decode", "spec_accept_rate") or state.value(
            "agg", "spec_accept_rate"
        )
        rows.append(
            (
                s.t("runtime.accept"),
                s.t(
                    "runtime.accept_value",
                    length=num(point.accept_len, 2),
                    rate=ratio_pct(rate) if rate is not None else DASH,
                ),
            )
        )
    retracted = state.derived("retracted_tokens_per_generated", SERVING)
    if retracted is not None:
        rows.append((s.t("runtime.retracted"), ratio_pct(retracted, 1)))
    evicted = state.derived("evicted_tokens_per_prompt", SERVING)
    if evicted is not None:
        rows.append((s.t("runtime.evicted"), num(evicted, 2)))
    for role, key, label in STAGE_ROWS:
        value = state.value(role, key)
        if value is not None:
            p99 = state.value(role, key, "p99")
            rows.append(
                (
                    s.t(label),
                    s.t(
                        "runtime.stage_value",
                        avg=num(value, 3),
                        p99=num(p99, 3) if p99 is not None else DASH,
                    ),
                )
            )
    busy = state.derived("forward_busy_fraction", SERVING)
    if busy is not None:
        rows.append((s.t("runtime.forward_busy"), ratio_pct(busy)))
    return rows


def _goodput_blocks(ctx: Ctx, point: PointResult) -> list[Block]:
    s, req = ctx.s, point.requests
    if req is None or not req.records:
        return []
    out: list[Block] = []
    if req.tiers:
        out.append(
            Table(
                (
                    s.t("runtime.tier"),
                    s.t("runtime.tier_slo"),
                    s.t("runtime.good_fraction"),
                    s.t("runtime.goodput_rps"),
                    s.t("runtime.good_tps_gpu"),
                ),
                tuple(
                    (
                        tier.name,
                        s.t(
                            "runtime.tier_slo_value",
                            ttft=num(tier.ttft_s, 0),
                            itl=num(tier.itl_ms, 0),
                        ),
                        ratio_pct(tier.good_fraction),
                        num(tier.goodput_rps, 2),
                        num(tier.good_output_tps_per_gpu, 1),
                    )
                    for tier in req.tiers
                ),
                (2, 3, 4),
                s.t("runtime.goodput_caption", n=req.records, errors=req.errors),
            )
        )
    leaders = facts.goodput_leaders(ctx.summary)
    if leaders:
        rows = []
        for name, best in leaders.items():
            tier = best.requests.tier(name) if best.requests else None
            if tier is None:
                continue
            rows.append(
                (
                    name,
                    ctx.family(best.candidate.family_id),
                    str(best.key.concurrency),
                    num(tier.good_output_tps_per_gpu, 1),
                    ratio_pct(tier.good_fraction),
                    ctx.tps(best.tps_per_gpu),
                )
            )
        out.append(
            Table(
                (
                    s.t("runtime.tier"),
                    s.t("runtime.family"),
                    "C",
                    s.t("runtime.good_tps_gpu"),
                    s.t("runtime.good_fraction"),
                    s.t("runtime.tps"),
                ),
                tuple(rows),
                (2, 3, 4, 5),
                s.t("runtime.goodput_best_caption"),
            )
        )
    if req.turns:
        out.append(
            Table(
                (
                    s.t("runtime.turn"),
                    s.t("runtime.requests"),
                    s.t("runtime.ttft_p50"),
                    s.t("runtime.ttft_p90"),
                    s.t("runtime.isl_p50"),
                    s.t("runtime.cache_read"),
                ),
                tuple(
                    (
                        b.bucket,
                        integer(b.requests),
                        num(b.ttft_p50_s, 2),
                        num(b.ttft_p90_s, 2),
                        integer(b.isl_p50),
                        ratio_pct(b.cache_read_fraction),
                    )
                    for b in req.turns
                ),
                (1, 2, 3, 4, 5),
                s.t("runtime.turn_caption"),
            )
        )
    if req.sessions is not None:
        sess = req.sessions
        out.append(
            Paragraph(
                s.r(
                    "runtime.sessions",
                    n=sess.sessions,
                    turns=num(sess.turns_p50, 0),
                    span=num(sess.span_p50_s, 0),
                    busy=ratio_pct(sess.busy_fraction_p50),
                    sub=sess.subagent_requests,
                )
            )
        )
    return out


def runtime_section(ctx: Ctx) -> list[Block]:
    """Section 9: cold start, KV capacity, service state, bottleneck, energy."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.runtime"), 2, "sec-10")]
    life_rows = _lifecycle_rows(ctx)
    if life_rows:
        out.append(Paragraph(s.r("runtime.lifecycle_intro")))
        out.append(
            Table(
                (
                    s.t("runtime.config"),
                    s.t("runtime.deploy_ok"),
                    s.t("runtime.ready_s"),
                    s.t("runtime.weights_s"),
                    s.t("runtime.graph_s"),
                    s.t("runtime.kv_tokens"),
                    s.t("runtime.max_running"),
                    s.t("runtime.issues"),
                ),
                tuple(life_rows),
                (2, 3, 4, 5, 6, 7),
            )
        )
    lead = facts.leader_point(sm)
    if lead is None or lead.state is None:
        out.append(Paragraph(s.r("runtime.no_state"), tone="note"))
        return out
    state = lead.state
    out.append(
        Paragraph(
            s.r(
                "runtime.state_intro",
                family=ctx.family(lead.candidate.family_id),
                c=lead.key.concurrency,
                source=state.service_source,
            )
        )
    )
    role_rows = _role_rows(ctx, state)
    if role_rows:
        out.append(
            Table(
                (
                    s.t("runtime.role"),
                    s.t("runtime.running"),
                    s.t("runtime.queued"),
                    s.t("runtime.kv_usage"),
                    s.t("runtime.prefix_hit"),
                    s.t("runtime.gen_tps"),
                ),
                tuple(role_rows),
                (1, 2, 3, 4, 5),
            )
        )
    if state.bottleneck is not None:
        b = state.bottleneck
        out.append(
            Paragraph(
                s.r(
                    "runtime.bottleneck",
                    label=s.label("bottleneck", b.label),
                    confidence=ratio_pct(b.confidence, 0),
                    evidence="；".join(localize(e) for e in b.evidence[:3]) or DASH,
                )
            )
        )
    rows = _bottleneck_rows(ctx)
    if len(rows) > 1:
        out.append(
            Table(
                (
                    s.t("runtime.family"),
                    "C",
                    s.t("runtime.tps"),
                    s.t("runtime.bottleneck_label"),
                    s.t("runtime.confidence"),
                    s.t("runtime.evidence"),
                ),
                tuple(rows),
                (1, 2, 4),
                s.t("runtime.bottleneck_caption"),
            )
        )
    stress = (
        _stress_rows(ctx, lead) + _activity_rows(ctx, lead) + _engine_rows(ctx, lead)
    )
    if stress:
        out.append(Table((s.t("runtime.metric"), s.t("runtime.value")), tuple(stress)))
    out.extend(_goodput_blocks(ctx, lead))
    out.extend(_accuracy_blocks(ctx))
    out.extend(_diagnostics_blocks(ctx))
    return out


ENGINE_LABELS = (
    ("est_tflops_per_gpu", "runtime.diag_tflops", 0),
    ("est_mem_gbs_per_gpu", "runtime.diag_mem", 0),
    ("forward_busy_fraction", "runtime.diag_busy", None),
    ("eplb_balancedness", "runtime.diag_eplb", 2),
)


def _diag_steps(ctx: Ctx) -> int:
    cfg = ctx.summary.run_config
    return cfg.diagnose.steps if cfg is not None else 10


def _diagnostics_blocks(ctx: Ctx) -> list[Block]:
    """Opt-in diagnostic window: kernel-time shares per role, engine metrics."""
    s, diag = ctx.s, ctx.summary.diagnostics
    if diag is None:
        return []
    out: list[Block] = [
        Paragraph(
            s.r(
                "runtime.diag_intro",
                family=ctx.family(diag.family_id),
                c=diag.concurrency,
                nsys=s.label("nsys_state", diag.nsys),
                version=diag.nsys_version or DASH,
            )
        )
    ]
    kernels = {
        role: KernelBreakdown.model_validate(raw)
        for role, raw in (diag.kernels or {}).items()
    }
    if kernels:
        roles = [r for r in (*SERVING, "unknown") if r in kernels]
        cats: list[str] = []
        for role in roles:
            for cat in kernels[role].categories:
                if cat.category not in cats:
                    cats.append(cat.category)
        rows = []
        for cat in cats:
            rows.append(
                (
                    s.label("kernel_category", cat),
                    *(ratio_pct(kernels[r].share(cat)) for r in roles),  # type: ignore[arg-type]
                )
            )
        out.append(
            Table(
                (s.t("runtime.diag_category"), *(s.label("role", r) for r in roles)),
                tuple(rows),
                tuple(range(1, len(roles) + 1)),
                s.t("runtime.diag_caption", steps=_diag_steps(ctx)),
            )
        )
        top_rows = []
        for role in roles:
            for kern in kernels[role].top[:5]:
                top_rows.append(
                    (
                        s.label("role", role),
                        code(kern.short),
                        s.label("kernel_category", kern.category),
                        ratio_pct(kern.share),
                    )
                )
        if top_rows:
            out.append(
                Table(
                    (
                        s.t("runtime.role"),
                        s.t("runtime.diag_kernel"),
                        s.t("runtime.diag_category"),
                        s.t("runtime.diag_share"),
                    ),
                    tuple(top_rows),
                    (3,),
                    s.t("runtime.diag_top_caption"),
                )
            )
    engine_rows = []
    for key, label, digits in ENGINE_LABELS:
        for role in (*SERVING,):
            value = diag.engine.get(f"{role}.{key}")
            if value is None:
                continue
            text = ratio_pct(value) if digits is None else num(value, digits)
            engine_rows.append((f"{s.t(label)}（{s.label('role', role)}）", text))
    if engine_rows:
        out.append(
            Table((s.t("runtime.metric"), s.t("runtime.value")), tuple(engine_rows))
        )
    for note in diag.notes[:6]:
        out.append(Paragraph(inline(localize(note)), tone="note"))
    return out


def _accuracy_blocks(ctx: Ctx) -> list[Block]:
    """Accuracy gate checks (GSM8K few-shot through the frontend)."""
    s, sm = ctx.s, ctx.summary
    if not sm.accuracy:
        return []
    rows = []
    for check in sm.accuracy:
        res, ver = check.result, check.verdict
        mtp = "|mtp=0|" not in check.cand_key
        rows.append(
            (
                ctx.family(check.family_id) + ("" if mtp else " · MTP off"),
                s.t(
                    "runtime.accuracy_value",
                    acc=ratio_pct(res.accuracy, 1),
                    lo=ratio_pct(res.ci_low, 1),
                    hi=ratio_pct(res.ci_high, 1),
                )
                if res.accuracy is not None
                else DASH,
                f"{res.n} / {res.errors}",
                s.label("accuracy_status", ver.status),
                localize(ver.reason),
            )
        )
    first = sm.accuracy[0].result
    return [
        Table(
            (
                s.t("runtime.config"),
                s.t("runtime.accuracy"),
                s.t("runtime.accuracy_n"),
                s.t("runtime.accuracy_verdict"),
                s.t("runtime.evidence"),
            ),
            tuple(rows),
            (2,),
            s.t(
                "runtime.accuracy_caption",
                shots=first.shots,
                questions=first.questions,
            ),
        )
    ]


# --------------------------------------------------------------------------
# 10 Node health and environment
# --------------------------------------------------------------------------


def _software_rows(ctx: Ctx) -> list[tuple[Cell, ...]]:
    sw = ctx.summary.software
    if sw is None:
        return []
    rows: list[tuple[Cell, ...]] = [
        (name, code(sw.packages[name])) for name in KEY_PACKAGES if name in sw.packages
    ]
    if sw.torch is not None:
        rows.append(
            ("torch CUDA / NCCL", f"{sw.torch.cuda or DASH} / {sw.torch.nccl or DASH}")
        )
    if sw.ucx is not None and sw.ucx.version:
        rows.append(("UCX", sw.ucx.version))
    if sw.os_release.get("PRETTY_NAME"):
        rows.append(("OS", sw.os_release["PRETTY_NAME"]))
    return rows


def _finding_rows(ctx: Ctx) -> list[tuple[Cell, ...]]:
    return [
        (
            ctx.s.label("severity", f.severity),
            f.node or DASH,
            code(f.code),
            localize(f.message),
        )
        for f in ctx.summary.fact_findings[:MAX_ROWS]
    ]


def health_section(ctx: Ctx) -> list[Block]:
    """Section 10: findings, counters that grew under load, versions, clocks."""
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.health"), 2, "sec-11")]
    if not sm.facts_signature and not sm.fact_findings and sm.software is None:
        out.append(Paragraph(s.r("health.none"), tone="note"))
        return out
    counts = {sev: 0 for sev in ("error", "warn", "info")}
    for finding in sm.fact_findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    grown = sum(1 for d in sm.health_deltas if d.increments)
    out.append(
        Paragraph(
            s.r(
                "health.intro",
                errors=counts["error"],
                warns=counts["warn"],
                infos=counts["info"],
                grown=grown,
                nodes=len(sm.health_deltas),
            ),
            tone="warn" if counts["error"] else "normal",
        )
    )
    rows = _finding_rows(ctx)
    if rows:
        out.append(
            Table(
                (
                    s.t("health.severity"),
                    s.t("health.node"),
                    s.t("health.code"),
                    s.t("health.message"),
                ),
                tuple(rows),
                (),
                s.t(
                    "health.findings_caption",
                    shown=len(rows),
                    total=len(sm.fact_findings),
                ),
            )
        )
    clock_rows = [
        (c.node, num(c.offset_s * 1000, 1), num(c.uncertainty_s * 1000, 1))
        for c in sm.clock_offsets
    ]
    disk_rows = [
        (d.node, gbs(d.gbs), d.method or DASH, d.error or DASH) for d in sm.disk_reads
    ]
    if clock_rows:
        out.append(
            Table(
                (
                    s.t("health.node"),
                    s.t("health.offset_ms"),
                    s.t("health.uncertainty_ms"),
                ),
                tuple(clock_rows),
                (1, 2),
                s.t("health.clock_caption"),
            )
        )
    if disk_rows:
        out.append(
            Table(
                (s.t("health.node"), "GB/s", s.t("health.method"), s.t("health.error")),
                tuple(disk_rows),
                (1,),
                s.t("health.disk_caption"),
            )
        )
    sw_rows = _software_rows(ctx)
    if sw_rows:
        out.append(
            Table(
                (s.t("health.package"), s.t("health.version")),
                tuple(sw_rows),
                (),
                s.t("health.software_caption", image=sm.software.image or sm.image),
            )
        )
    return out


def runtime_kpi_lines(ctx: Ctx) -> Sequence[Inline]:
    """Summary bullets: bottleneck, energy, fabric shape, health warnings."""
    s, sm = ctx.s, ctx.summary
    out: list[Inline] = []
    lead = facts.leader_point(sm)
    if lead is not None and lead.state is not None and lead.state.bottleneck:
        out.append(
            s.r(
                "summary.bottleneck",
                label=s.label("bottleneck", lead.state.bottleneck.label),
                energy=num(lead.state.energy_per_token_j, 3),
            )
        )
    if sm.fabric is not None:
        out.append(
            s.r(
                "summary.fabric",
                kind=s.label("fabric_kind", sm.fabric.kind),
                rails=len(sm.fabric.rails),
                rate=_rate_text(sm.fabric),
                rail_opt=ctx.tri(sm.fabric.rail_optimized),
                per_leaf=integer(sm.fabric.nodes_per_leaf),
                hops=integer(sm.fabric.max_hops),
            )
        )
    check = sm.accuracy_of(lead.candidate.key) if lead is not None else None
    if check is not None and check.result.accuracy is not None:
        out.append(
            s.r(
                "summary.accuracy",
                status=s.label("accuracy_status", check.verdict.status),
                acc=ratio_pct(check.result.accuracy, 1),
                lo=ratio_pct(check.result.ci_low, 1),
                hi=ratio_pct(check.result.ci_high, 1),
                n=check.result.n,
            )
        )
    elif lead is not None and sm.run_config is not None:
        out.append(
            s.r(
                "summary.accuracy_none",
                why=s.label("accuracy_status", check.verdict.status)
                if check is not None
                else s.t("runtime.accuracy_unchecked"),
            )
        )
    bad = [f for f in sm.fact_findings if f.severity in ("error", "warn")]
    if bad:
        out.append(s.r("summary.health", n=len(bad)))
    return out
