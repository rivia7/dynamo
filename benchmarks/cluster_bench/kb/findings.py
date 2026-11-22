# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turns one ``ExperimentSummary`` into ``FindingRecord`` rows.

Each finding states one claim, the evidence point ids behind it, a
consult-perf-knowledge evidence category (1 AIPerf data, 3 same-series
deployment history, 5 hardware speed-of-light) and the measured noise floor,
and carries the cells of one of the two agent-docs knob-tuning table schemas
(see ``FindingRecord``). Findings only restate what the summary measured; a
mechanism is never asserted beyond what the evidence supports.
"""

from __future__ import annotations

from typing import Any

from benchmarks.cluster_bench.domain.schema import (
    BandwidthStat,
    DeployResult,
    ExperimentSummary,
    LaunchPlan,
    NicArm,
    PointResult,
)
from benchmarks.cluster_bench.kb.records import FindingRecord
from benchmarks.cluster_bench.report import facts
from benchmarks.cluster_bench.report.fmt import gbs, num, pct, pct_delta, ratio_pct
from benchmarks.cluster_bench.report.plans import PlanLookup

CAT_AIPERF = 1
CAT_HISTORY = 3
CAT_SOL = 5

NIC_KNOBS = {
    "nccl": "`NCCL_IB_HCA`",
    "nixl": "`UCX_NET_DEVICES`",
    "mooncake": "Mooncake `--disaggregation-ib-device`",
    "deepep": "`NVSHMEM_HCA_LIST`",
}
ALL_NIC_KNOBS = (
    "`NCCL_IB_HCA`, `UCX_NET_DEVICES`, Mooncake `--disaggregation-ib-device`, "
    "`NVSHMEM_HCA_LIST`"
)
PLANE_NAMES = {
    "ib": "inter-node IB/RDMA",
    "nvlink": "intra-node NVLink",
    "pcie": "GPU PCIe",
    "pcie_nic": "GPU<->NIC PCIe",
}
PLANE_KNOBS = {
    "ib": "topology (cross-node KV transfer or collectives), NIC count",
    "nvlink": "TP/EP size, all-reduce path (`--enable-symm-mem`, NCCL NVLS)",
    "pcie": "host offload and CPU staging, GPU placement",
    "pcie_nic": "GPU-direct RDMA path (nvidia-peermem), GPU-to-NIC affinity",
}
FAILURE_KNOBS = {
    "oom": "`--mem-fraction-static`, `--cuda-graph-max-bs-decode`, `--max-running-requests`",
    "kernel_unsupported": "`--attention-backend`, checkpoint precision, image",
    "rdma_registration": "`UCX_NET_DEVICES`/`NCCL_IB_HCA`, nvidia-peermem, memlock",
    "nccl": "`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, NCCL version",
    "port_in_use": "port allocation of co-located instances",
    "timeout": "load timeout, model cache warmth",
    "engagement": "transport restriction env (NIC selection)",
}
SPEC_KNOBS = (
    "`--speculative-algorithm EAGLE`, `--speculative-num-steps`, "
    "`--speculative-eagle-topk`, `--speculative-num-draft-tokens`, "
    "`SGLANG_ENABLE_SPEC_V2`"
)


def _yes_no(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _noise_text(within: bool | None, floor_pct: float | None) -> str:
    if within is None or floor_pct is None:
        return "no noise floor measured"
    side = "within" if within else "beyond"
    return f"{side} the {floor_pct:.2f}% noise floor"


def _ids(*ids: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(i for i in ids if i))


class _Builder:
    def __init__(self, summary: ExperimentSummary, plan_lookup: PlanLookup | None):
        self.sm = summary
        self.lookup = plan_lookup
        leader = facts.leader_candidate(summary)
        self.leader_plan = plan_lookup(leader.fp) if plan_lookup and leader else None
        self.titles = facts.family_titles(summary)
        noise = summary.noise
        self.floor_pct = noise.noise_floor_pct if noise else None
        self.out: list[FindingRecord] = []
        self.workload = (
            f"{summary.model} ({summary.checkpoint.repo}) on {summary.node_count}x"
            f"{summary.gpus_per_node} {summary.gpu_name}; {summary.dataset}; "
            f"served context {summary.served_context}"
        )
        conditions = "; ".join(summary.conditions)
        self.context_caveat = (
            f"AgentX closed-loop replay of {summary.dataset}; throughput benchmark, "
            "correctness not verified" + (f"; {conditions}" if conditions else "")
        )

    def title(self, family_id: str | None) -> str:
        if family_id is None:
            return "n/a"
        return self.titles.get(family_id, family_id)

    def add(self, kind: str, subject: str, **fields: Any) -> None:
        self.out.append(
            FindingRecord.create(
                fp=self.sm.fingerprint,
                exp_id=self.sm.exp_id,
                key=f"finding:{self.sm.exp_id}:{kind}:{subject}",
                kind=kind,
                gpu_profile=self.sm.gpu_profile,
                model=self.sm.model,
                noise_floor_pct=self.floor_pct,
                **fields,
            )
        )

    def validation(self, *point_ids: str | None, extra: str = "") -> str:
        ids = ", ".join(f"`{i}`" for i in _ids(*point_ids)) or "n/a"
        text = f"exp `{self.sm.exp_id}`; points {ids}"
        return f"{text}; {extra}" if extra else text

    # -- scenario findings ---------------------------------------------------

    def topology(self) -> None:
        sm = self.sm
        lead = facts.leader_point(sm)
        cand = facts.leader_candidate(sm)
        if lead is None or cand is None:
            return
        second = facts.runner_up(sm)
        delta = pct_delta(lead.tps_per_gpu, second.best_tps_per_gpu if second else None)
        confirm, confirm_delta = facts.confirm_delta(sm)
        signal = (
            f"{num(lead.tps_per_gpu, dash='n/a')} tok/s/GPU at C={lead.key.concurrency}"
            f" (saturated: {_yes_no(sm.saturated)})"
        )
        if second is not None:
            signal += (
                f"; runner-up `{second.family_id}` "
                f"{num(second.best_tps_per_gpu, dash='n/a')} tok/s/GPU, leader "
                f"{pct(delta, signed=True, dash='n/a')} "
                f"({_noise_text(facts.within_noise(sm, delta), self.floor_pct)})"
            )
        knobs = (
            f"family `{cand.family_id}` ({self.title(cand.family_id)}); "
            f"MTP {'on' if cand.mtp else 'off'}"
        )
        if self.leader_plan is not None:
            knobs += f"; transport `{self.leader_plan.transport}`"
        extra = ""
        if confirm is not None:
            extra = (
                f"confirmatory rerun {num(confirm.tps_per_gpu, dash='n/a')} tok/s/GPU "
                f"({pct(confirm_delta, signed=True, dash='n/a')})"
            )
        self.add(
            "topology",
            "best",
            table="scenario",
            claim=(
                f"Best topology for {sm.model} on {sm.node_count}x{sm.gpus_per_node} "
                f"{sm.gpu_name}: {self.title(cand.family_id)} at "
                f"{num(lead.tps_per_gpu, dash='n/a')} tok/s/GPU "
                f"(C={lead.key.concurrency})."
            ),
            symptom=self.workload,
            knobs=knobs,
            effect=signal,
            validation=self.validation(
                lead.key.id,
                second.best_point_id if second else None,
                confirm.key.id if confirm else None,
                extra=extra,
            ),
            try_first=(
                f"Start from `{cand.family_id}` at C={lead.key.concurrency}; "
                "re-establish the noise floor before tuning knobs."
            ),
            caveat=self.context_caveat,
            evidence_point_ids=_ids(
                lead.key.id,
                second.best_point_id if second else None,
                confirm.key.id if confirm else None,
            ),
            evidence_category=CAT_AIPERF,
            delta_pct=delta,
            family=cand.family_id,
            concurrency=lead.key.concurrency,
        )

    def floors(self) -> None:
        for floor in sorted(self.sm.floors, key=lambda f: f.user_tps_floor):
            label = f">= {floor.user_tps_floor:g} output tok/s/user (p50)"
            family = floor.cand_key.split("|", 1)[0] if floor.cand_key else None
            if floor.tps_per_gpu is None:
                self.add(
                    "user_floor",
                    f"{floor.user_tps_floor:g}",
                    table="scenario",
                    claim=f"No measured point met {label}.",
                    symptom=f"Interactivity floor {label}; {self.workload}",
                    knobs="concurrency (AgentX lanes), topology family",
                    effect="no point at or above the floor",
                    validation=self.validation(extra=floor.note),
                    try_first="Measure lower concurrency or a latency-oriented family.",
                    caveat=self.context_caveat,
                    evidence_point_ids=(),
                    evidence_category=CAT_AIPERF,
                    delta_pct=None,
                    family=None,
                    concurrency=None,
                )
                continue
            self.add(
                "user_floor",
                f"{floor.user_tps_floor:g}",
                table="scenario",
                claim=(
                    f"At {label} the best measured throughput is "
                    f"{num(floor.tps_per_gpu)} tok/s/GPU ({self.title(family)}, "
                    f"C={floor.concurrency})."
                ),
                symptom=f"Interactivity floor {label}; {self.workload}",
                knobs=f"family `{family}`, concurrency <= {floor.concurrency}",
                effect=(
                    f"{num(floor.tps_per_gpu)} tok/s/GPU measured; "
                    f"{num(floor.interpolated_tps_per_gpu, dash='n/a')} tok/s/GPU "
                    "interpolated on log2(concurrency)"
                ),
                validation=self.validation(floor.point_id, extra=floor.note),
                try_first=f"Cap concurrency at {floor.concurrency} on `{family}`.",
                caveat="Read from the concurrency curve; " + self.context_caveat,
                evidence_point_ids=_ids(floor.point_id),
                evidence_category=CAT_AIPERF,
                delta_pct=None,
                family=family,
                concurrency=floor.concurrency,
            )

    def saturation(self) -> None:
        sm = self.sm
        lead = facts.leader_point(sm)
        if lead is None or sm.saturated is None:
            return
        family = lead.candidate.family_id
        c = lead.key.concurrency
        try_first = (
            f"Do not extend the grid beyond C={c} for `{family}`; the top is flat."
            if sm.saturated
            else f"Extend the concurrency grid above C={c} for `{family}`."
        )
        self.add(
            "saturation",
            "leader",
            table="scenario",
            claim=(
                f"The throughput curve of `{family}` is "
                f"{'saturated' if sm.saturated else 'not saturated'} at C={c}."
            ),
            symptom=f"Saturation of `{family}`; {self.workload}",
            knobs="concurrency (AgentX lanes)",
            effect=sm.saturation_note or "n/a",
            validation=self.validation(lead.key.id),
            try_first=try_first,
            caveat="Flatness is judged against the pilot noise floor.",
            evidence_point_ids=_ids(lead.key.id),
            evidence_category=CAT_AIPERF,
            delta_pct=None,
            family=family,
            concurrency=c,
        )

    def noise(self) -> None:
        sm = self.sm
        noise = sm.noise
        if noise is None:
            return
        family = noise.cand_key.split("|", 1)[0]
        pilot_ids = tuple(
            p.key.id
            for p in sm.points
            if p.key.purpose == "pilot" and p.key.cand_key == noise.cand_key
        )
        cv = noise.stdev / noise.mean * 100.0 if noise.mean else None
        self.add(
            "noise",
            "pilot",
            table="scenario",
            claim=(
                f"Run-to-run noise of `{family}` at C={noise.concurrency}: CV "
                f"{pct(cv, 2, dash='n/a')}, noise floor {noise.noise_floor_pct:.2f}% "
                f"(n={len(noise.samples)})."
            ),
            symptom=f"Noise floor / minimum detectable effect; {self.workload}",
            knobs=f"n={len(noise.samples)} repetitions at the decision spec",
            effect=(
                f"floor {noise.noise_floor:.2f} tok/s/GPU "
                f"({noise.noise_floor_pct:.2f}%), mean {noise.mean:.1f}"
            ),
            validation=self.validation(*pilot_ids),
            try_first=(
                f"Treat deltas below {noise.noise_floor_pct:.2f}% as noise on this "
                "cluster and workload."
            ),
            caveat="Measured at one concurrency; noise can differ elsewhere.",
            evidence_point_ids=pilot_ids,
            evidence_category=CAT_AIPERF,
            delta_pct=None,
            family=family,
            concurrency=noise.concurrency,
        )

    def nic_irrelevant(self) -> None:
        sm = self.sm
        if facts.leader_cross_node(sm, self.leader_plan) is not False:
            return
        lead = facts.leader_point(sm)
        cand = facts.leader_candidate(sm)
        if cand is None:
            return
        rate = lead.cross_node_gbs if lead else None
        self.add(
            "nic_irrelevant",
            "leader",
            table="scenario",
            claim=(
                f"The leader `{cand.family_id}` has no cross-node data-plane traffic "
                f"({gbs(rate, dash='n/a')} GB/s per node), so the NIC count cannot "
                "change its throughput."
            ),
            symptom=f"NIC-count tuning for per-node replicas; {self.workload}",
            knobs=f"NIC selection env ({ALL_NIC_KNOBS})",
            effect=f"mean inter-node IB {gbs(rate, dash='n/a')} GB/s per node",
            validation=self.validation(
                lead.key.id if lead else None,
                extra=f"NIC sweep ran on `{sm.nic_family_id}`"
                if sm.nic_family_id
                else "",
            ),
            try_first=(
                "Skip NIC-count tuning for per-node replicas; sweep NICs only on "
                "topologies with cross-node traffic."
            ),
            caveat="Holds only while every replica stays within one node.",
            evidence_point_ids=_ids(lead.key.id if lead else None),
            evidence_category=CAT_SOL,
            delta_pct=None,
            family=cand.family_id,
            concurrency=lead.key.concurrency if lead else None,
        )

    def bandwidth(self) -> None:
        """One finding per plane: its busiest direction at the leader point."""
        lead = facts.leader_point(self.sm)
        if lead is None:
            return
        per_plane: dict[str, BandwidthStat] = {}
        for stat in facts.busiest_stats(lead.bandwidth):
            if stat.method == "unavailable" or not stat.ref_peak_gbs:
                continue
            current = per_plane.get(stat.plane)
            if current is None or (stat.util_vs_peak or 0.0) > (
                current.util_vs_peak or 0.0
            ):
                per_plane[stat.plane] = stat
        for stat in per_plane.values():
            self._bandwidth_finding(lead, stat)

    def _bandwidth_finding(self, lead: PointResult, stat: BandwidthStat) -> None:
        point_id = lead.key.id
        util = stat.util_vs_peak or 0.0
        plane = PLANE_NAMES.get(stat.plane, stat.plane)
        family = lead.candidate.family_id
        busy = util >= 0.5
        self.add(
            "bandwidth",
            stat.plane,
            table="scenario",
            claim=(
                f"Under full load the busiest {plane} {stat.direction} link "
                f"({stat.scope}) peaked at {stat.peak_gbs:.1f} GB/s = "
                f"{ratio_pct(stat.util_vs_peak, 0)} of measured speed-of-light "
                f"({stat.ref_peak_gbs:.1f} GB/s); p99 {stat.p99_gbs:.1f}, mean "
                f"{stat.mean_gbs:.1f} GB/s."
            ),
            symptom=(
                f"{plane} occupancy ({stat.direction}) at the leader point "
                f"(`{family}`, C={lead.key.concurrency})"
            ),
            knobs=PLANE_KNOBS.get(stat.plane, "n/a"),
            effect=(
                f"peak {ratio_pct(stat.util_vs_peak, dash='n/a')} of speed-of-light, "
                f"{ratio_pct(stat.util_vs_nominal, dash='n/a')} of nominal "
                f"({stat.method})"
            ),
            validation=self.validation(point_id, extra=f"scope `{stat.scope}`"),
            try_first=(
                f"{plane} is close to its limit; reduce its traffic first."
                if busy
                else f"{plane} is not the bottleneck at this load; look elsewhere."
            ),
            caveat=(
                "Derived from other counters, not measured directly."
                if stat.method == "derived"
                else "Sampled counters miss bursts shorter than the sample period."
            ),
            evidence_point_ids=(point_id,),
            evidence_category=CAT_SOL,
            delta_pct=None,
            family=family,
            concurrency=lead.key.concurrency,
        )

    # -- knob findings -------------------------------------------------------

    def mtp(self) -> None:
        sm = self.sm
        mtp = sm.mtp
        spec_keys = SPEC_KNOBS
        if self.leader_plan is not None:
            keys = sorted(
                {
                    k
                    for cfg in self.leader_plan.rendered_configs.values()
                    for k in cfg
                    if k.startswith("speculative")
                }
            )
            if keys:
                spec_keys = ", ".join(f"`--{k}`" for k in keys)
        headline_ids: tuple[str, ...] = ()
        if mtp is not None and mtp.delta_pct is not None:
            anchor = sm.point(mtp.on_point_id) or sm.point(mtp.off_point_id)
            family = anchor.candidate.family_id if anchor else None
            c = anchor.key.concurrency if anchor else None
            headline_ids = _ids(mtp.on_point_id, mtp.off_point_id)
            self._mtp_finding(
                "headline",
                family,
                c,
                mtp.on_tps_per_gpu,
                mtp.off_tps_per_gpu,
                mtp.delta_pct,
                mtp.within_noise,
                mtp.accept_len,
                headline_ids,
                spec_keys,
            )
        for pair in facts.mtp_pairs(sm):
            ids = _ids(*pair.on_ids, *pair.off_ids)
            if set(headline_ids) <= set(ids) and headline_ids:
                continue
            self._mtp_finding(
                f"{pair.family_id}@{pair.concurrency}/{pair.spec}",
                pair.family_id,
                pair.concurrency,
                pair.on_tps,
                pair.off_tps,
                pair.delta_pct,
                facts.within_noise(sm, pair.delta_pct),
                pair.accept_len,
                ids,
                spec_keys,
            )

    def _mtp_finding(
        self,
        subject: str,
        family: str | None,
        concurrency: int | None,
        on_tps: float | None,
        off_tps: float | None,
        delta: float | None,
        within: bool | None,
        accept_len: float | None,
        ids: tuple[str, ...],
        spec_keys: str,
    ) -> None:
        direction = "raises" if (delta or 0.0) > 0 else "lowers"
        self.add(
            "mtp",
            subject,
            table="knob",
            claim=(
                f"MTP (EAGLE) {direction} output tok/s/GPU by "
                f"{pct(abs(delta) if delta is not None else None, dash='n/a')} on "
                f"`{family}` at C={concurrency} (measured accept length "
                f"{num(accept_len, 2, dash='n/a')})."
            ),
            symptom=f"Speculative decoding on vs off (`{family}`, C={concurrency})",
            knobs=spec_keys,
            effect=(
                f"{pct(delta, signed=True, dash='n/a')} tok/s/GPU "
                f"({num(on_tps, dash='n/a')} on vs {num(off_tps, dash='n/a')} off)"
            ),
            validation=self.validation(*ids, extra=_noise_text(within, self.floor_pct)),
            try_first="",
            caveat=(
                f"Measured acceptance on {self.sm.dataset}; MTP can lose at higher "
                "concurrency when draft state reduces batch capacity."
            ),
            evidence_point_ids=ids,
            evidence_category=CAT_AIPERF,
            delta_pct=delta,
            family=family,
            concurrency=concurrency,
        )

    def nic_arms(self) -> None:
        sm = self.sm
        full = facts.full_nic_arm(sm.nic_arms)
        if full is None:
            return
        for arm in facts.nic_arms_sorted(sm.nic_arms):
            if arm.nic_count >= full.nic_count:
                continue
            self._nic_finding(arm, full)

    def _nic_knobs(self, arm: NicArm) -> str:
        point = self.sm.point(arm.point_id)
        plan: LaunchPlan | None = None
        if point is not None and self.lookup is not None:
            plan = self.lookup(point.candidate.fp)
        if plan is not None and plan.transport in NIC_KNOBS:
            return f"{NIC_KNOBS[plan.transport]} (transport `{plan.transport}`)"
        return ALL_NIC_KNOBS

    def _nic_finding(self, arm: NicArm, full: NicArm) -> None:
        sm = self.sm
        family = sm.nic_family_id
        staging = (
            "GPU-direct likely collapsed to host staging (ranks share a NIC). "
            if arm.staging_suspected
            else ""
        )
        self.add(
            "nic_count",
            str(arm.nic_count),
            table="knob",
            claim=(
                f"Restricting `{family}` to {arm.nic_count} of {full.nic_count} NICs "
                f"per node changes output tok/s/GPU by "
                f"{pct(arm.delta_vs_full_pct, signed=True, dash='n/a')} "
                f"({_noise_text(arm.within_noise, self.floor_pct)})."
            ),
            symptom=(
                f"NIC count reduced to {arm.nic_count} of {full.nic_count} per node "
                f"(`{family}`, C={sm.nic_concurrency})"
            ),
            knobs=self._nic_knobs(arm),
            effect=(
                f"{pct(arm.delta_vs_full_pct, signed=True, dash='n/a')} tok/s/GPU "
                f"({num(arm.tps_per_gpu, dash='n/a')} vs "
                f"{num(full.tps_per_gpu, dash='n/a')}); effective NICs "
                f"{num(arm.nic_effective, 2, dash='n/a')}"
            ),
            validation=self.validation(
                arm.point_id,
                full.point_id,
                extra=(
                    f"restriction engaged: {_yes_no(arm.engaged)}; staging "
                    f"suspected: {_yes_no(arm.staging_suspected)}; status {arm.status}"
                ),
            ),
            try_first="",
            caveat=(
                f"{staging}Only topologies with cross-node data-plane traffic are "
                "sensitive to the NIC count."
            ),
            evidence_point_ids=_ids(arm.point_id, full.point_id),
            evidence_category=CAT_AIPERF,
            delta_pct=arm.delta_vs_full_pct,
            family=family,
            concurrency=sm.nic_concurrency,
        )

    def deploys(self) -> None:
        seen: dict[str, int] = {}
        for deploy in self.sm.deploys:
            ordinal = seen.get(deploy.cand_fp, 0)
            seen[deploy.cand_fp] = ordinal + 1
            if not deploy.ok:
                self._deploy_failure(deploy, ordinal)
            elif deploy.candidate.rungs:
                self._deploy_rungs(deploy, ordinal)

    def _deploy_failure(self, deploy: DeployResult, ordinal: int) -> None:
        cand = deploy.candidate
        failure = deploy.failure or "unknown"
        detail = " ".join(deploy.detail.split())[:200]
        self.add(
            "deploy",
            f"{deploy.cand_fp}-{ordinal}",
            table="knob",
            claim=f"`{cand.key}` failed to deploy ({failure}).",
            symptom=f"Deploy failure `{failure}` of `{cand.family_id}`",
            knobs=FAILURE_KNOBS.get(failure, "see the deploy log"),
            effect="no serving capacity; the family has no throughput number",
            validation=(
                f"exp `{self.sm.exp_id}`; log `{deploy.log_dir or 'n/a'}`"
                + (f"; {detail}" if detail else "")
            ),
            try_first="",
            caveat="Runtime log evidence; the failure may be image or driver specific.",
            evidence_point_ids=(),
            evidence_category=CAT_HISTORY,
            delta_pct=None,
            family=cand.family_id,
            concurrency=None,
        )

    def _deploy_rungs(self, deploy: DeployResult, ordinal: int) -> None:
        cand = deploy.candidate
        rungs = "+".join(cand.rungs)
        self.add(
            "deploy",
            f"{deploy.cand_fp}-{ordinal}",
            table="knob",
            claim=f"`{cand.family_id}` deployed only after fallback rungs {rungs}.",
            symptom=f"Default config of `{cand.family_id}` did not deploy",
            knobs=f"ladder rungs `{rungs}`",
            effect="deployment succeeded with the fallback applied",
            validation=f"exp `{self.sm.exp_id}`; log `{deploy.log_dir or 'n/a'}`",
            try_first="",
            caveat="Fallbacks usually trade memory or graph coverage for fit.",
            evidence_point_ids=(),
            evidence_category=CAT_HISTORY,
            delta_pct=None,
            family=cand.family_id,
            concurrency=None,
        )


def derive_findings(
    summary: ExperimentSummary, plan_lookup: PlanLookup | None = None
) -> tuple[FindingRecord, ...]:
    """Every finding of one experiment, in a deterministic order."""
    builder = _Builder(summary, plan_lookup)
    builder.topology()
    builder.floors()
    builder.saturation()
    builder.noise()
    builder.nic_irrelevant()
    builder.bandwidth()
    builder.mtp()
    builder.nic_arms()
    builder.deploys()
    return tuple(builder.out)
