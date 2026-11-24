# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report section 2: interconnect needs, NVLink off vs on, MFU / MBU, KV capacity,
and for A-Only the simulated F (its model, the null-F bound, A<->F traffic).

The three focus resources of the leader at saturation: how much interconnect
it needs (and the smallest NIC count that holds up), how much of the GPUs'
floating-point peak (MFU from config.json FLOPs) and memory bandwidth (MBU,
measured by NVML GPM) it uses, and how much context its KV cache pools hold.
Reads only the ``ExperimentSummary``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.cluster_bench.domain import afd
from benchmarks.cluster_bench.domain.capacity import IDLE_IB_NOTE
from benchmarks.cluster_bench.domain.schema import (
    AfdBound,
    AfdFSpec,
    AfdReport,
    InterconnectNeeds,
    KvCapacity,
    NvlinkComparison,
    PointResult,
)
from benchmarks.cluster_bench.domain.stages import flops_arch
from benchmarks.cluster_bench.report import facts
from benchmarks.cluster_bench.report.doc import (
    Block,
    Bullets,
    Cell,
    Heading,
    Inline,
    Kpi,
    Paragraph,
    Table,
    code,
)
from benchmarks.cluster_bench.report.fmt import DASH, num, pct, ratio_pct
from benchmarks.cluster_bench.report.reasons import localize

if TYPE_CHECKING:
    from benchmarks.cluster_bench.report.sections import Ctx

KV_FULL = 0.9  # a pool this full at the leader point limits throughput


def tokens(value: float | None) -> str:
    """Token counts the way Chinese readers count: 万 above ten thousand."""
    if value is None:
        return DASH
    if value >= 1e4:
        text = f"{value / 1e4:.1f}".rstrip("0").rstrip(".")
        return f"{text} 万"
    return f"{int(round(value))}"


def _spaced(label: str) -> str:
    """A Latin-ending label gets a space before the Chinese that follows it."""
    return f"{label} " if label and label[-1].isascii() else label


def resources_section(ctx: Ctx) -> list[Block]:
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("sections.resources"), 2, "sec-2")]
    lead = facts.leader_point(sm)
    if lead is None:
        out.append(Paragraph(s.r("summary.no_leader"), tone="note"))
        return out
    out.append(
        Paragraph(
            s.r(
                "resources.intro",
                family=ctx.family(lead.candidate.family_id),
                c=lead.key.concurrency,
            )
        )
    )
    out.extend(_interconnect(ctx, sm.interconnect))
    out.extend(_nccl_hops(ctx))
    out.extend(_nvlink(ctx, sm.nvlink))
    out.extend(_mfu(ctx, lead))
    out.extend(_kv(ctx, sm.kv_capacity))
    out.extend(_afd(ctx, sm.afd))
    return out


# --------------------------------------------------------------------------
# 2.1 interconnect
# --------------------------------------------------------------------------


def _interconnect(ctx: Ctx, need: InterconnectNeeds | None) -> list[Block]:
    s = ctx.s
    out: list[Block] = [Heading(s.t("resources.bw_heading"), 3)]
    if need is None or all(n.status == "unavailable" for n in need.links):
        out.append(Paragraph(s.r("resources.no_bw"), tone="note"))
        return out
    out.append(
        Paragraph(s.r("resources.bw_intro", target=ratio_pct(need.target_util, 0)))
    )
    rows: list[tuple[Cell, ...]] = []
    for link in need.links:
        rows.append(
            (
                s.label("link_kind", link.link),
                code(link.scope) if link.scope else DASH,
                num(link.p99_gbs, 2),
                num(link.peak_gbs, 2),
                num(link.capacity_gbs, 0),
                ratio_pct(link.util_p99),
                num(link.need_gbs, 2),
                s.label("link_status", link.status),
            )
        )
    out.append(
        Table(
            (
                s.t("resources.link"),
                s.t("resources.scope"),
                s.t("resources.p99"),
                s.t("resources.peak"),
                s.t("resources.capacity"),
                s.t("resources.util"),
                s.t("resources.need"),
                s.t("resources.status"),
            ),
            tuple(rows),
            numeric=(2, 3, 4, 5, 6),
            caption=s.t("resources.bw_caption"),
        )
    )
    out.append(Bullets(tuple(_interconnect_bullets(ctx, need))))
    return out


HOPS_MAX = 32


def _nccl_hops(ctx: Ctx) -> list[Block]:
    """How NCCL connected the GPUs of one node: per GPU pair the transports
    the intra-node microbenchmark's INFO log names, next to the path
    ``nvidia-smi topo -m`` reports."""
    s, sm = ctx.s, ctx.summary
    peak = next((p for p in sm.peaks if p.details.get("nccl_hops")), None)
    if peak is None:
        return []
    node = next((n for n in sm.inventory.nodes if n.name == peak.scope), None)
    topo = node.topo if node is not None else {}
    pairs: dict[tuple[int, int], set[str]] = {}
    for hop, vias in dict(peak.details["nccl_hops"]).items():
        src, _, dst = str(hop).partition("->")
        if not (src.isdigit() and dst.isdigit()):
            continue
        a, b = sorted((int(src), int(dst)))
        pairs.setdefault((a, b), set()).update(str(v) for v in vias)
    rows: list[tuple[Cell, ...]] = [
        (
            f"GPU {a} ↔ GPU {b}",
            (topo.get(f"GPU{a}") or {}).get(f"GPU{b}") or DASH,
            " / ".join(sorted(vias)),
        )
        for (a, b), vias in sorted(pairs.items())
    ]
    if not rows:
        return []
    out: list[Block] = [
        Table(
            (s.t("resources.hop"), s.t("resources.hop_topo"), s.t("resources.hop_via")),
            tuple(rows[:HOPS_MAX]),
            caption=s.t("resources.hops_caption", node=peak.scope),
        )
    ]
    if any(v.startswith("SHM") for vias in pairs.values() for v in vias):
        out.append(Paragraph(s.r("resources.hops_shm"), tone="note"))
    return out


def _nic_estimate(ctx: Ctx, need: InterconnectNeeds) -> Inline:
    """The leader's NIC count from its own IB traffic."""
    s, ib = ctx.s, need.link("ib")
    if ib is not None and ib.status == "idle":
        return s.r("resources.nic_est_idle")
    if need.min_nics_estimated is None or ib is None:
        return s.r("resources.nic_est_none")
    return s.r(
        "resources.nic_est",
        estimated=str(need.min_nics_estimated),
        need=num(ib.need_gbs, 1),
        rate=num(need.nic_rate_gbs, 0),
    )


def _nic_measured(ctx: Ctx, need: InterconnectNeeds) -> Inline:
    """The NIC sweep's minimum, attributed to the family it ran on."""
    s = ctx.s
    reason = localize(need.min_nics_reason) or DASH
    if need.min_nics_measured is None:
        return s.r("resources.nic_meas_none", reason=reason)
    if need.sweep_on_leader:
        return s.r(
            "resources.nic_meas", measured=str(need.min_nics_measured), reason=reason
        )
    return s.r(
        "resources.nic_meas_other",
        family=ctx.family(need.sweep_family_id),
        measured=str(need.min_nics_measured),
        reason=reason,
    )


def _interconnect_bullets(ctx: Ctx, need: InterconnectNeeds) -> list[Inline]:
    s = ctx.s
    items: list[Inline] = [
        s.r(
            "resources.nic_min",
            estimate=_nic_estimate(ctx, need),
            measured=_nic_measured(ctx, need),
        )
    ]
    # Only a sweep of the leader itself can show that its traffic estimate
    # is too optimistic; another family's minimum says nothing about it.
    if (
        need.sweep_on_leader
        and need.min_nics_measured is not None
        and need.min_nics_estimated is not None
        and need.min_nics_estimated < need.min_nics_measured
    ):
        items.append(
            s.r(
                "resources.nic_gap",
                estimated=str(need.min_nics_estimated),
                measured=str(need.min_nics_measured),
            )
        )
    nvlink = need.link("nvlink")
    pcie = num(need.pcie_alt_gbs, 0)
    if need.nvlink_needed is True and nvlink is not None:
        items.append(
            s.r("resources.nvlink_needed", need=num(nvlink.need_gbs, 1), pcie=pcie)
        )
    elif need.nvlink_needed is False and nvlink is not None:
        items.append(
            s.r("resources.nvlink_optional", need=num(nvlink.need_gbs, 1), pcie=pcie)
        )
    else:
        items.append(s.r("resources.nvlink_none"))
    measured = ctx.summary.nvlink
    if measured is not None and measured.delta_pct is not None:
        verdict = ""
        if measured.status == "measured" and measured.within_noise is not None:
            verdict = s.t(
                "resources.nvl_verdict_not_needed"
                if measured.within_noise
                else "resources.nvl_verdict_needed"
            )
        items.append(
            s.r(
                "resources.nvl_see_measured",
                delta=pct(measured.delta_pct, 1, signed=True),
                verdict=verdict
                if measured.delta_pct < 0 or measured.within_noise
                else "",
            )
        )
    # the idle-IB note is already part of the NIC bullet
    items.extend(localize(n) for n in need.notes if n != IDLE_IB_NOTE)
    if ctx.summary.afd is not None:
        items.append(s.r("resources.afd_no_traffic_ib"))
    return items


# --------------------------------------------------------------------------
# 2.2 NVLink off vs on
# --------------------------------------------------------------------------

HARDWARE_METHODS = ("bw_mode_off", "link_width_0")


def _nvlink(ctx: Ctx, cmp: NvlinkComparison | None) -> list[Block]:
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("resources.nvl_heading"), 3)]
    if cmp is None or cmp.off_point_id is None:
        reason = localize(cmp.reason) if cmp is not None and cmp.reason else DASH
        out.append(Paragraph(s.r("resources.nvl_skipped", reason=reason), tone="note"))
        out.extend(_nvlink_tail(ctx, cmp))
        return out
    out.append(
        Paragraph(
            s.r(
                "resources.nvl_intro",
                c=str(cmp.concurrency),
                family=ctx.family(cmp.family_id),
                reason=localize(cmp.reason) or DASH,
            )
        )
    )
    facts: list[Inline] = []
    if cmp.method:
        facts.append(s.r(f"resources.nvl_method_{cmp.method}"))
    facts.append(s.r("resources.nvl_fast_paths"))
    if cmp.method in HARDWARE_METHODS:
        if cmp.path == "host":
            facts.append(s.r("resources.nvl_proof_host"))
        elif cmp.p2p_gbs is not None:
            gpu = ctx.summary.interconnect
            facts.append(
                s.r(
                    "resources.nvl_proof",
                    p2p=num(cmp.p2p_gbs, 1),
                    pairs=str(cmp.peer_pairs) if cmp.peer_pairs is not None else DASH,
                    pcie=num(gpu.pcie_alt_gbs if gpu else cmp.pcie_capacity_gbs, 0),
                )
            )
    if cmp.nvlink_p99_gbs is not None or cmp.pcie_p99_gbs is not None:
        util = (
            cmp.pcie_p99_gbs / cmp.pcie_capacity_gbs
            if cmp.pcie_p99_gbs is not None and cmp.pcie_capacity_gbs
            else None
        )
        facts.append(
            s.r(
                "resources.nvl_load",
                nvl=num(cmp.nvlink_p99_gbs, 1),
                pcie=num(cmp.pcie_p99_gbs, 1),
                util=ratio_pct(util, 0),
            )
        )
    out.append(Bullets(tuple(facts)))
    rows: list[tuple[Cell, ...]] = [
        (
            s.label("p2p_path", "nvlink"),
            num(cmp.on_tps_per_gpu, 1),
            s.t("resources.nvl_baseline"),
            num(cmp.on_user_tps, 1),
            num(cmp.on_ttft_p50_ms, 0),
            num(cmp.on_itl_p50_ms, 1),
        ),
        (
            s.label("p2p_path", cmp.path or "pcie"),
            num(cmp.off_tps_per_gpu, 1),
            pct(cmp.delta_pct, 1, signed=True),
            num(cmp.off_user_tps, 1),
            num(cmp.off_ttft_p50_ms, 0),
            num(cmp.off_itl_p50_ms, 1),
        ),
    ]
    out.append(
        Table(
            (
                s.t("resources.nvl_link"),
                s.t("resources.nvl_tps"),
                s.t("resources.nvl_delta"),
                s.t("resources.nvl_user"),
                s.t("resources.nvl_ttft"),
                s.t("resources.nvl_itl"),
            ),
            tuple(rows),
            numeric=(1, 2, 3, 4, 5),
            caption=s.t("resources.nvl_caption"),
        )
    )
    off = sm.point(cmp.off_point_id)
    if cmp.status == "invalid":
        reasons = "；".join(localize(r) for r in (off.audit.reasons if off else ()))
        out.append(
            Paragraph(
                s.r("resources.nvl_invalid", reasons=reasons or DASH), tone="warn"
            )
        )
    elif cmp.delta_pct is None:
        out.append(Paragraph(s.r("resources.nvl_no_reference"), tone="note"))
    else:
        delta = pct(cmp.delta_pct, 1, signed=True)
        if cmp.within_noise:
            key = "resources.nvl_not_needed"
        elif cmp.delta_pct < 0:
            key = "resources.nvl_needed"
        else:
            key = "resources.nvl_odd"
        out.append(Paragraph(s.r(key, delta=delta)))
        if cmp.path == "host":
            out.append(Paragraph(s.r("resources.nvl_host_caveat"), tone="note"))
    out.extend(_nvlink_tail(ctx, cmp))
    return out


def _nvlink_tail(ctx: Ctx, cmp: NvlinkComparison | None) -> list[Block]:
    """Restore warning and the switch's notes."""
    s = ctx.s
    out: list[Block] = []
    if cmp is None:
        return out
    if cmp.restored is False:
        out.append(Paragraph(s.r("resources.nvl_restore_failed"), tone="warn"))
    notes = [localize(n) for n in cmp.notes]
    if notes:
        out.append(Bullets(tuple(notes)))
    return out


# --------------------------------------------------------------------------
# 2.3 MFU
# --------------------------------------------------------------------------


def _tbs(value: float) -> str:
    """A memory bandwidth in TB/s at its own precision: 1.398, 4.8, 8."""
    return f"{value:g}"


def _mfu(ctx: Ctx, lead: PointResult) -> list[Block]:
    s, sm = ctx.s, ctx.summary
    out: list[Block] = [Heading(s.t("resources.mfu_heading"), 3)]
    mfu, arch = lead.mfu, flops_arch(sm.stage, sm.model_arch)
    if mfu is None:
        out.append(Paragraph(s.r("resources.no_mfu"), tone="note"))
        return out
    if arch is not None and mfu.window_s is not None:
        gflops = num(arch.linear_flops_per_token() / 1e9, 1)
        if mfu.peak_tflops_per_gpu:
            out.append(
                Paragraph(
                    s.r(
                        "resources.mfu_intro",
                        gflops=gflops,
                        params=num(arch.active_params() / 1e9, 1),
                        gpu=sm.gpu_name,
                        precision=mfu.precision.upper(),
                        peak=num(mfu.peak_tflops_per_gpu, 0),
                    )
                )
            )
        else:
            out.append(
                Paragraph(
                    s.r(
                        "resources.mfu_intro_nopeak",
                        gflops=gflops,
                        precision=mfu.precision.upper(),
                    ),
                    tone="note",
                )
            )
    # MBU is measured by NVML GPM: it needs neither config.json nor a peak
    has_mbu = any(r.mbu is not None for r in mfu.roles)
    has_tbs = any(r.mem_tbs_per_gpu is not None for r in mfu.roles)
    metric = code("DRAM_BW_UTIL")
    if has_mbu and mfu.peak_mem_tbs_per_gpu:
        peak = _tbs(mfu.peak_mem_tbs_per_gpu)
        out.append(Paragraph(s.r("resources.mbu_intro", metric=metric, peak=peak)))
    elif has_mbu:
        intro = s.r("resources.mbu_intro_nopeak", metric=metric)
        out.append(Paragraph(intro, tone="note"))
    else:
        out.append(Paragraph(s.r("resources.no_mbu"), tone="note"))
    headers = (
        s.t("resources.role"),
        s.t("resources.gpus"),
        s.t("resources.tflops"),
        s.t("resources.mfu"),
        s.t("resources.tensor"),
        *((s.t("resources.mbu"),) if has_mbu else ()),
        *((s.t("resources.mem_tbs"),) if has_tbs else ()),
    )
    rows = [
        (
            s.label("role", r.role),
            str(r.gpus),
            num(r.tflops_per_gpu, 1),
            ratio_pct(r.mfu, 1),
            ratio_pct(r.tensor_active, 0),
            *((ratio_pct(r.mbu, 1),) if has_mbu else ()),
            *((num(r.mem_tbs_per_gpu, 2),) if has_tbs else ()),
        )
        for r in mfu.roles
    ]
    out.append(
        Table(
            headers,
            tuple(rows),
            numeric=tuple(range(1, len(headers))),
            caption=s.t("resources.mfu_caption"),
        )
    )
    curve = sorted(
        (
            p
            for p in sm.points
            if p.key.cand_key == lead.key.cand_key
            and p.key.purpose == "search"
            and p.mfu is not None
        ),
        key=lambda p: p.key.concurrency,
    )
    if len(curve) >= 2:
        out.append(_mfu_curve(ctx, lead, curve))
    notes = [localize(n) for n in mfu.notes]
    if notes:
        out.append(Bullets(tuple(notes)))
    return out


def _mfu_curve(ctx: Ctx, lead: PointResult, curve: list[PointResult]) -> Table:
    """MFU, and MBU where GPM measured it, of the leader's search points."""
    s = ctx.s
    results = [p.mfu for p in curve if p.mfu is not None]
    roles = [r.role for r in results[-1].roles]
    split = len(roles) > 1  # PD: one column per role plus the total
    mbu = any(r.mbu is not None for m in results for r in m.roles)
    headers = [s.t("resources.concurrency"), s.t("resources.tps")]
    if split:
        headers += [s.t("resources.kpi_mfu", role=r) for r in roles]
    headers.append(s.t("resources.mfu_total") if split else s.t("resources.mfu"))
    if mbu:
        headers += (
            [s.t("resources.kpi_mbu", role=r) for r in roles]
            if split
            else [s.t("resources.mbu")]
        )
    rows = []
    for p in curve:
        m = p.mfu
        assert m is not None
        per_role = [m.role(r) for r in roles]
        row = [str(p.key.concurrency), num(p.tps_per_gpu, 1)]
        if split:
            row += [ratio_pct(x.mfu if x else None, 1) for x in per_role]
        row.append(ratio_pct(m.total_mfu, 1))
        if mbu:
            row += [ratio_pct(x.mbu if x else None, 1) for x in per_role]
        rows.append(tuple(row))
    return Table(
        tuple(headers),
        tuple(rows),
        numeric=tuple(range(len(headers))),
        caption=s.t(
            "resources.mfu_curve_caption",
            family=ctx.family(lead.candidate.family_id),
        ),
    )


# --------------------------------------------------------------------------
# 2.4 KV capacity
# --------------------------------------------------------------------------


def _kv(ctx: Ctx, kv: KvCapacity | None) -> list[Block]:
    s = ctx.s
    out: list[Block] = [Heading(s.t("resources.kv_heading"), 3)]
    if kv is None or kv.total_tokens is None:
        out.append(Paragraph(s.r("resources.no_kv"), tone="note"))
        return out
    out.append(Paragraph(s.r("resources.kv_intro")))
    rows = [
        (
            s.label("role", r.role),
            str(r.instances),
            str(r.dp_ranks),
            tokens(r.per_rank_tokens),
            tokens(r.instance_tokens),
            tokens(r.total_tokens),
            num(r.bytes_per_token, 0),
            tokens(r.host_tokens_total),
        )
        for r in kv.roles
    ]
    out.append(
        Table(
            (
                s.t("resources.kv_role"),
                s.t("resources.instances"),
                s.t("resources.dp"),
                s.t("resources.per_rank"),
                s.t("resources.per_instance"),
                s.t("resources.total"),
                s.t("resources.bpt"),
                s.t("resources.host"),
            ),
            tuple(rows),
            numeric=(1, 2, 3, 4, 5, 6, 7),
            caption=s.t("resources.kv_caption"),
        )
    )
    items: list[Inline] = [
        s.r(
            "resources.kv_total",
            total=tokens(kv.total_tokens),
            role=_spaced(s.label("role", kv.serving_role)),
        )
    ]
    if kv.max_request_tokens is not None:
        items.append(
            s.r("resources.kv_max_request", tokens=tokens(kv.max_request_tokens))
        )
    if kv.requests_at_p50 is not None:
        items.append(
            s.r(
                "resources.kv_requests",
                p50=tokens(kv.context_p50),
                p90=tokens(kv.context_p90),
                r50=str(kv.requests_at_p50),
                r90=str(kv.requests_at_p90) if kv.requests_at_p90 is not None else DASH,
            )
        )
    if kv.usage_avg is not None or kv.usage_peak is not None:
        items.append(
            s.r(
                "resources.kv_usage",
                avg=ratio_pct(kv.usage_avg, 0),
                peak=ratio_pct(kv.usage_peak, 0),
            )
        )
        if (kv.usage_peak or 0.0) >= KV_FULL:
            items.append(s.r("resources.kv_full"))
    main = kv.role(kv.serving_role)
    if main is not None and (main.bytes_per_token or kv.arch_bytes_per_token):
        items.append(
            s.r(
                "resources.kv_bpt",
                measured=num(main.bytes_per_token, 0),
                arch=num(kv.arch_bytes_per_token, 0),
            )
        )
    items.extend(localize(n) for n in kv.notes)
    out.append(Bullets(tuple(items)))
    return out


# --------------------------------------------------------------------------
# section 1 tiles
# --------------------------------------------------------------------------


def _nic_tile(ctx: Ctx, need: InterconnectNeeds) -> Kpi | None:
    """Fewest NICs per node for the leader.

    Measured when the NIC sweep ran on the leader's family; otherwise the
    leader's own traffic estimate, with the other family's measurement as
    context (it does not describe the leader).
    """
    s = ctx.s
    measured, estimated = need.min_nics_measured, need.min_nics_estimated
    if measured is not None and need.sweep_on_leader:
        return Kpi(
            s.t("resources.kpi_nics"),
            str(measured),
            s.t(
                "resources.kpi_nics_detail",
                estimated=str(estimated) if estimated else DASH,
            ),
        )
    if estimated is None:
        return None
    detail = (
        s.t("resources.kpi_nics_detail_other", measured=str(measured))
        if measured is not None
        else s.t("resources.kpi_nics_detail_none")
    )
    return Kpi(s.t("resources.kpi_nics_est"), str(estimated), detail)


def resource_kpis(ctx: Ctx, lead: PointResult) -> list[Kpi]:
    """Headline tiles for the summary: fewest NICs, NVLink, MFU, MBU, KV capacity."""
    s, sm = ctx.s, ctx.summary
    tiles: list[Kpi] = []
    need = sm.interconnect
    tile = _nic_tile(ctx, need) if need is not None else None
    if tile is not None:
        tiles.append(tile)
    cmp = sm.nvlink
    if cmp is not None and cmp.status == "measured" and cmp.delta_pct is not None:
        detail = (
            "resources.kpi_nvl_detail_host"
            if cmp.path == "host"
            else "resources.kpi_nvl_detail"
        )
        tiles.append(
            Kpi(
                s.t("resources.kpi_nvl"),
                pct(cmp.delta_pct, 1, signed=True),
                s.t(
                    detail,
                    on=num(cmp.on_tps_per_gpu, 1),
                    off=num(cmp.off_tps_per_gpu, 1),
                ),
            )
        )
    mfu = lead.mfu
    if mfu is not None and mfu.total_mfu is not None and len(mfu.roles) > 1:
        # PD: the whole cluster's MFU, with each role's share in the detail
        tiles.append(
            Kpi(
                s.t("resources.kpi_mfu_total"),
                ratio_pct(mfu.total_mfu, 1),
                " · ".join(f"{r.role} {ratio_pct(r.mfu, 1)}" for r in mfu.roles),
            )
        )
    elif mfu is not None and mfu.roles and mfu.roles[0].mfu is not None:
        main = mfu.roles[0]
        tiles.append(
            Kpi(
                s.t("resources.kpi_mfu", role=main.role),
                ratio_pct(main.mfu, 1),
                s.t("resources.kpi_mfu_detail", tflops=num(main.tflops_per_gpu, 0)),
            )
        )
    if mfu is not None and mfu.total_mbu is not None and len(mfu.roles) > 1:
        # PD: the whole cluster's MBU, with each role's in the detail
        tiles.append(
            Kpi(
                s.t("resources.kpi_mbu_total"),
                ratio_pct(mfu.total_mbu, 1),
                " · ".join(f"{r.role} {ratio_pct(r.mbu, 1)}" for r in mfu.roles),
            )
        )
    elif mfu is not None and mfu.roles and mfu.roles[0].mbu is not None:
        main = mfu.roles[0]
        peak = mfu.peak_mem_tbs_per_gpu
        detail = (
            s.t(
                "resources.kpi_mbu_detail",
                tbs=num(main.mem_tbs_per_gpu, 2),
                peak=_tbs(peak),
            )
            if main.mem_tbs_per_gpu is not None and peak
            else s.t("resources.kpi_mbu_detail_nopeak")
        )
        tiles.append(
            Kpi(
                s.t("resources.kpi_mbu", role=main.role), ratio_pct(main.mbu, 1), detail
            )
        )
    kv = sm.kv_capacity
    if kv is not None and kv.total_tokens:
        tiles.append(
            Kpi(
                s.t("resources.kpi_kv"),
                tokens(kv.total_tokens),
                s.t(
                    "resources.kpi_kv_detail",
                    max=tokens(kv.max_request_tokens),
                    n=str(kv.requests_at_p50)
                    if kv.requests_at_p50 is not None
                    else DASH,
                ),
                "tokens",
            )
        )
    return tiles


# --------------------------------------------------------------------------
# 2.5 A-Only: the simulated F
# --------------------------------------------------------------------------


def _afd(ctx: Ctx, report: AfdReport | None) -> list[Block]:
    """A-Only: the F model behind the wait, the null-F bound, A<->F traffic."""
    if report is None:
        return []
    s = ctx.s
    out: list[Block] = [
        Heading(s.t("resources.afd_heading"), 3),
        Paragraph(s.r("resources.afd_intro")),
        _afd_model(ctx, report),
    ]
    if report.steps:
        out.append(_afd_steps(ctx, report))
    out.extend(_afd_bound(ctx, report.bound))
    items = _afd_bullets(ctx, report)
    items.extend(localize(n) for n in report.plan.notes)
    items.append(s.r("resources.afd_simplified"))
    out.append(Bullets(tuple(items)))
    return out


def _short(value: float) -> str:
    """Up to two decimals, trailing zeros dropped: 8.0 -> "8", 3.476 -> "3.48"."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _afd_flag(ctx: Ctx, spec: AfdFSpec, name: str) -> Inline:
    """Where an --afd-f setting came from: the flag, or the default."""
    value = getattr(spec, name)
    if value == AfdFSpec.model_fields[name].default:
        return ctx.s.t("resources.afd_default")
    return code(f"--afd-f {name}={value}")


def _afd_model(ctx: Ctx, report: AfdReport) -> Table:
    s, plan = ctx.s, report.plan
    spec = plan.spec
    b_in, b_out = afd.row_bytes(plan.hidden, plan.top_k, spec.wire)
    rows: tuple[tuple[Cell, ...], ...] = (
        (
            s.t("resources.afd_f_gpus"),
            s.r(
                "resources.afd_f_gpus_value",
                gpus=str(plan.f_gpus),
                per_gpu=num(plan.routed_gb / plan.f_gpus, 0),
            ),
            localize(plan.f_gpus_source),
        ),
        (
            s.t("resources.afd_experts"),
            s.r(
                "resources.afd_experts_value",
                layers=str(plan.moe_layers),
                layer=num(plan.layer_bytes / 1e9, 2),
                total=num(plan.routed_gb, 0),
                dtype=plan.expert_dtype.upper(),
                experts=str(plan.experts),
                top_k=str(plan.top_k),
            ),
            s.t("resources.afd_from_config"),
        ),
        (
            s.t("resources.afd_link"),
            s.r(
                "resources.afd_link_value",
                bw=num(plan.link_gbs_per_gpu, 1),
                kind=s.label("resources.afd_link_kind", spec.link),
            ),
            localize(plan.link_source),
        ),
        (
            s.t("resources.afd_lat"),
            s.r("resources.afd_lat_value", lat=num(plan.lat_us, 1)),
            localize(plan.lat_source),
        ),
        (
            s.t("resources.afd_layout"),
            s.r(
                f"resources.afd_layout_{spec.layout}",
                fan=_short(afd.fan_out(spec.layout, plan.f_gpus, plan.top_k)),
                b_in=f"{b_in:,.0f}",
                b_out=f"{b_out:,.0f}",
                wire=spec.wire.upper(),
            ),
            _afd_flag(ctx, spec, "layout"),
        ),
        (
            s.t("resources.afd_microbatches"),
            s.r(
                "resources.afd_mb_overlap"
                if spec.microbatches > 1
                else "resources.afd_mb_serial",
                m=str(spec.microbatches),
            ),
            _afd_flag(ctx, spec, "microbatches"),
        ),
        (
            s.t("resources.afd_f_speed"),
            s.r(
                "resources.afd_f_speed_value",
                hbm=_tbs(plan.f_hbm_tbs),
                eff_mem=ratio_pct(spec.eff_mem, 0),
                precision=plan.f_peak_precision.upper(),
                peak=num(plan.f_peak_tflops, 0),
                eff_flop=ratio_pct(spec.eff_flop, 0),
            ),
            s.t("resources.afd_from_profile"),
        ),
    )
    return Table(
        (
            s.t("resources.afd_item"),
            s.t("resources.afd_value"),
            s.t("resources.afd_source"),
        ),
        rows,
        caption=s.t("resources.afd_model_caption", n_a=report.n_a),
    )


def _afd_steps(ctx: Ctx, report: AfdReport) -> Table:
    s = ctx.s
    rows = tuple(
        (
            f"{row.rows:,}",
            num(row.latency_us, 1),
            num(row.link_us, 1),
            num(row.f_mem_us, 1),
            num(row.f_flop_us, 1),
            num(row.serial_us, 1),
            num(row.overlap_us, 1),
            num(row.step_ms, 2),
            s.label("resources.afd_bound_kind", row.bound),
        )
        for row in report.steps
    )
    return Table(
        (
            s.t("resources.afd_rows"),
            s.t("resources.afd_latency"),
            s.t("resources.afd_wire"),
            s.t("resources.afd_f_mem"),
            s.t("resources.afd_f_flop"),
            s.t("resources.afd_first"),
            s.t("resources.afd_later"),
            s.t("resources.afd_step"),
            s.t("resources.afd_bound"),
        ),
        rows,
        numeric=(0, 1, 2, 3, 4, 5, 6, 7),
        caption=s.t(
            "resources.afd_steps_caption",
            m=report.plan.spec.microbatches,
            rest=report.plan.moe_layers - 1,
        ),
    )


def _afd_bound(ctx: Ctx, bound: AfdBound) -> list[Block]:
    """The simulated F against the same point with the wait off."""
    s = ctx.s
    if bound.status == "skipped":
        reason = localize(bound.reason) or DASH
        return [
            Paragraph(s.r("resources.afd_bound_skipped", reason=reason), tone="note")
        ]
    if bound.status == "failed":
        return [Paragraph(s.r("resources.afd_bound_failed"), tone="warn")]
    out: list[Block] = [
        Paragraph(s.r("resources.afd_bound_intro", c=str(bound.concurrency or DASH)))
    ]
    if bound.status == "invalid":
        reason = "；".join(localize(r) for r in bound.reason.split("; ")) or DASH
        out.append(
            Paragraph(s.r("resources.afd_bound_invalid", reason=reason), tone="warn")
        )
        return out
    out.append(
        Table(
            (
                s.t("resources.afd_f"),
                s.t("resources.nvl_tps"),
                s.t("resources.afd_vs_null"),
                s.t("resources.nvl_user"),
                s.t("resources.nvl_itl"),
            ),
            (
                (
                    s.t("resources.afd_sim"),
                    num(bound.sim_tps_per_gpu, 1),
                    pct(
                        -bound.f_cost_pct if bound.f_cost_pct is not None else None,
                        1,
                        signed=True,
                    ),
                    num(bound.sim_user_tps, 1),
                    num(bound.sim_itl_p50_ms, 1),
                ),
                (
                    s.t("resources.afd_null"),
                    num(bound.null_tps_per_gpu, 1),
                    s.t("resources.afd_baseline"),
                    num(bound.null_user_tps, 1),
                    num(bound.null_itl_p50_ms, 1),
                ),
            ),
            numeric=(1, 2, 3, 4),
            caption=s.t("resources.afd_bound_caption", c=bound.concurrency),
        )
    )
    verdict = ""
    if bound.within_noise is not None:
        verdict = s.t(
            "resources.afd_cost_noise"
            if bound.within_noise
            else "resources.afd_cost_real"
        )
    out.append(
        Paragraph(
            s.r("resources.afd_cost", cost=pct(bound.f_cost_pct, 1), verdict=verdict)
        )
    )
    return out


def _afd_bullets(ctx: Ctx, report: AfdReport) -> list[Inline]:
    """Estimated A<->F traffic and the measured A-side weights."""
    s, plan = ctx.s, report.plan
    items: list[Inline] = []
    traffic = report.traffic
    if traffic is not None:
        items.append(
            s.r(
                "resources.afd_traffic",
                to_f=num(traffic.to_f_bytes_per_token / 1e6, 2),
                from_f=num(traffic.from_f_bytes_per_token / 1e6, 2),
                layers=str(plan.moe_layers),
                rows=num(traffic.rows_per_token, 2),
                fan=_short(traffic.fan_out),
            )
        )
        if traffic.sim_need_gbs_per_gpu is not None:
            null = traffic.null_need_gbs_per_gpu
            items.append(
                s.r(
                    "resources.afd_traffic_need"
                    if null is None
                    else "resources.afd_traffic_need_null",
                    need=num(traffic.sim_need_gbs_per_gpu, 2),
                    link=num(traffic.link_gbs_per_gpu, 1),
                    util=ratio_pct(traffic.sim_util, 0),
                    null_need=num(null, 2),
                    null_util=ratio_pct(traffic.null_util, 0),
                )
            )
    items.append(_afd_weights(ctx, report))
    return items


def _afd_weights(ctx: Ctx, report: AfdReport) -> Inline:
    """The A-side weights against their size without and with the target's
    routed experts (what deploy checked)."""
    s, weights = ctx.s, report.weight_gb_per_gpu
    null, loaded = report.null_weight_gb_per_gpu, report.loaded_weight_gb_per_gpu
    if weights is None:
        return s.r("resources.afd_weights_unknown")
    if null is None or loaded is None:
        return s.r("resources.afd_weights_unchecked", w=num(weights, 1))
    return s.r(
        "resources.afd_weights"
        if afd.weights_engaged(weights, (null, loaded))
        else "resources.afd_weights_high",
        w=num(weights, 1),
        null=num(null, 1),
        loaded=num(loaded, 1),
        n_a=str(report.n_a),
    )
