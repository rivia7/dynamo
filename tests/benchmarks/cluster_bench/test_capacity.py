# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interconnect needs, the minimum NIC count, and KV cache capacity."""

from __future__ import annotations

import pytest

from benchmarks.cluster_bench.domain.capacity import (
    interconnect_needs,
    kv_capacity,
    min_nics_measured,
)
from benchmarks.cluster_bench.domain.lifecycle import DeployLifecycle, RoleLifecycle
from benchmarks.cluster_bench.domain.modelarch import SIMULATED_CONFIG, arch_from_config
from benchmarks.cluster_bench.domain.pointstate import PointState, RoleBrief
from benchmarks.cluster_bench.domain.requests import RequestSummary
from benchmarks.cluster_bench.domain.schema import (
    Candidate,
    ContainerSpec,
    GpuProfile,
    LaunchPlan,
    NicArm,
)
from tests.benchmarks.cluster_bench import summary_factory as factory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

H200 = GpuProfile(
    name="h200",
    match=("h200",),
    arch="sm90",
    hbm_gb=141.0,
    nvlink=True,
    weights_variant="fp8",
    nominal_nvlink_gbs=450.0,
    nominal_pcie_gbs=64.0,
)
RTX = GpuProfile(
    name="rtx6000",
    match=("rtx pro 6000",),
    arch="sm120",
    hbm_gb=96.0,
    nvlink=False,
    weights_variant="nvfp4",
    nominal_pcie_gbs=64.0,
)
stat = factory._stat


def hca_rows(nodes=("node0", "node1"), hcas=8, p99=4.0):
    return tuple(
        stat("ib", f"{node}/mlx5_{h}", "tx", 1.0, p99, p99 * 1.5, 48.3, 50.0)
        for node in nodes
        for h in range(hcas)
    )


def point(*rows):
    return factory.make_point(factory.DIS, 256, 104.9, 17.3, bandwidth=tuple(rows))


def busy_point():
    """Disagg traffic: IB carries KV, one GPU's PCIe runs hot."""
    return point(
        stat("ib", "cluster", "tx", 21.0, 80.0, 100.0, None, 800.0),
        stat("ib", "node0", "tx", 10.0, 30.0, 40.0, None, 400.0),
        stat("ib", "node0", "rx", 9.0, 20.0, 30.0, None, 400.0),
        stat("ib", "node1", "tx", 11.0, 35.0, 45.0, None, 400.0),
        stat("ib", "node1", "rx", 15.0, 50.0, 60.0, None, 400.0),
        *hca_rows(),
        stat("nvlink", "node0/gpu0", "tx", 40.0, 100.0, 130.0, 368.2, 450.0),
        stat("nvlink", "node0/gpu0", "rx", 41.0, 120.0, 140.0, 368.2, 450.0),
        # a higher burst but a lower p99: not the busiest
        stat("nvlink", "node1/gpu5", "tx", 38.0, 90.0, 150.0, 368.2, 450.0),
        stat("pcie", "node0/gpu0", "tx", 5.0, 10.0, 12.0, 53.1, 64.0),
        stat("pcie", "node0/gpu0", "rx", 20.0, 50.0, 58.0, 53.1, 64.0),
        # derived P2P share and unavailable rows never count
        stat("pcie", "node0/gpu0/p2p", "tx", 30.0, 63.0, 64.0, None, 64.0, "derived"),
        stat("pcie", "node1", "bidir", 0.0, 0.0, 0.0, None, 64.0, "unavailable"),
        stat("pcie_nic", "node0/gpu0", "tx", 2.0, 5.0, 7.0, 47.6, 50.0, "derived"),
    )


def test_busiest_device_sets_each_link_need():
    needs = interconnect_needs(busy_point(), (), H200, sweep_reason="no sweep")
    assert needs is not None and needs.target_util == 0.7
    ib = needs.link("ib")
    assert (ib.scope, ib.p99_gbs, ib.peak_gbs) == ("node1", 50.0, 60.0)  # worse dir
    assert ib.capacity_gbs == 400.0 and ib.util_p99 == pytest.approx(0.125)
    assert ib.need_gbs == pytest.approx(50.0 / 0.7, abs=1e-3) and ib.status == "ok"
    nvl = needs.link("nvlink")
    assert (nvl.scope, nvl.p99_gbs, nvl.peak_gbs) == ("node0/gpu0", 120.0, 140.0)
    assert nvl.util_p99 == pytest.approx(120.0 / 450.0, abs=1e-4)
    pcie = needs.link("pcie")
    assert pcie.p99_gbs == 50.0 and pcie.status == "tight"  # 78 % > 70 %
    nic = needs.link("pcie_nic")
    assert nic.status == "ok" and nic.capacity_gbs == 50.0
    # one NIC is 50 GB/s; the busiest node needs 71.4 GB/s -> 2 NICs
    assert (needs.nic_rate_gbs, needs.nics_per_node) == (50.0, 8)
    assert needs.min_nics_estimated == 2
    # GPU-GPU need 171 GB/s is beyond one PCIe link (64 GB/s)
    assert needs.nvlink_needed is True and needs.pcie_alt_gbs == 64.0
    assert needs.min_nics_measured is None and needs.min_nics_reason == "no sweep"
    assert needs.notes == ()


def test_target_utilization_scales_the_need():
    strict = interconnect_needs(busy_point(), (), H200, target_util=0.5)
    assert strict.link("ib").need_gbs == pytest.approx(100.0)
    assert strict.min_nics_estimated == 2  # 100 / 50, not rounded up past 2
    assert strict.link("pcie").status == "tight"
    loose = interconnect_needs(busy_point(), (), H200, target_util=0.9)
    assert loose.link("pcie").status == "ok"


def test_idle_ib_needs_one_nic_and_pcie_can_carry_small_gpu_traffic():
    needs = interconnect_needs(
        point(
            stat("ib", "node0", "tx", 0.01, 0.02, 0.05, None, 400.0),
            stat("ib", "node0", "rx", 0.01, 0.01, 0.03, None, 400.0),
            *hca_rows(("node0",), p99=0.0),
            stat("nvlink", "node0/gpu0", "tx", 12.0, 30.0, 41.0, 368.2, 450.0),
        ),
        (),
        H200,
    )
    assert needs.link("ib").status == "idle" and needs.min_nics_estimated == 1
    assert any("one NIC is enough" in n for n in needs.notes)
    assert needs.nvlink_needed is False  # 30 / 0.7 = 43 GB/s < 64 GB/s
    assert needs.link("pcie").status == "unavailable"  # no rows at all


def test_pcie_only_gpus_have_no_nvlink_verdict():
    needs = interconnect_needs(
        point(
            stat("nvlink", "node0", "bidir", 0.0, 0.0, 0.0, None, None, "unavailable"),
            stat("pcie", "node0/gpu0", "tx", 10.0, 30.0, 40.0, None, 64.0),
        ),
        (),
        RTX,
    )
    assert needs.link("nvlink").status == "unavailable" and needs.nvlink_needed is None
    assert needs.link("pcie").capacity_gbs == 64.0
    assert needs.min_nics_estimated is None  # no IB rows: nothing to size from


def test_no_bandwidth_means_no_needs():
    assert interconnect_needs(None, (), H200) is None
    assert interconnect_needs(point(), (), H200) is None


def arm(count, *, within=True, delta=None, engaged=True, staging=False, status=None):
    return NicArm(
        nic_count=count,
        tps_per_gpu=100.0,
        within_noise=within,
        delta_vs_full_pct=delta,
        engaged=engaged,
        staging_suspected=staging,
        status=status or "measured",
    )


def test_min_nics_is_the_fewest_that_hold_with_every_larger_count():
    full = arm(8, within=None)
    assert min_nics_measured(
        (full, arm(4), arm(2, within=False, delta=-12.0), arm(1, within=False))
    ) == (4, "4 NIC(s) per node stay within the noise floor of 8")
    # 2 NICs held, but 4 did not: 2 is not a safe minimum
    assert min_nics_measured((full, arm(4, within=False, delta=-10.0), arm(2))) == (
        8,
        "fewer than 8 NICs cost throughput beyond the noise floor",
    )
    # faster than the full count beyond the noise floor still holds
    assert min_nics_measured((full, arm(4, within=False, delta=3.0)))[0] == 4


def test_min_nics_ignores_arms_that_do_not_test_the_nics():
    full = arm(8, within=None)
    assert min_nics_measured((full, arm(4, engaged=False)))[0] == 8  # not engaged
    assert min_nics_measured((full, arm(4, staging=True)))[0] == 8  # host staging
    assert min_nics_measured((full, arm(4, status="invalid"))) == (
        None,
        "only the 8-NIC arm was measured",
    )
    assert min_nics_measured(()) == (None, "no usable NIC sweep arm")


def test_sweep_arms_feed_the_measured_minimum():
    arms = (arm(8, within=None), arm(4), arm(2, within=False, delta=-9.0))
    needs = interconnect_needs(
        busy_point(), arms, H200, sweep_reason="ignored", sweep_family="disagg-1p1d"
    )
    assert needs.min_nics_measured == 4 and "noise floor of 8" in needs.min_nics_reason
    assert needs.family_id == needs.sweep_family_id == "disagg-1p1d"
    assert needs.sweep_on_leader


def test_a_sweep_of_another_family_does_not_describe_the_leader():
    arms = (arm(8, within=None), arm(4))
    other = interconnect_needs(busy_point(), arms, H200, sweep_family="agg-tp8")
    assert other.min_nics_measured == 4 and not other.sweep_on_leader
    # without arms nothing was measured, whatever family was named
    none = interconnect_needs(busy_point(), (), H200, sweep_family="agg-tp8")
    assert none.sweep_family_id is None and none.sweep_on_leader


# --------------------------------------------------------------------------
# KV capacity
# --------------------------------------------------------------------------


def agg_lifecycle(**kwargs) -> DeployLifecycle:
    fields = dict(
        role="agg",
        containers=4,
        ready=4,
        kv_tokens_min=360_000,
        kv_cache_gb_min=25.0,
        max_total_num_tokens_min=360_000,
        max_running_requests_min=512,
        context_len=262_144,
    )
    fields.update(kwargs)
    return DeployLifecycle(roles={"agg": RoleLifecycle(**fields)})


def agg_plan(**extra) -> LaunchPlan:
    plan = factory.h200_leader_plan()
    configs = {k: {**v, **extra} for k, v in plan.rendered_configs.items()}
    return plan.model_copy(update={"rendered_configs": configs})


def test_dp_attention_multiplies_the_pool_of_each_instance():
    arch = arch_from_config(SIMULATED_CONFIG)
    cap = kv_capacity(
        agg_lifecycle(),
        agg_plan(**{"enable-dp-attention": True, "data-parallel-size": 8}),
        arch=arch,
        requests=RequestSummary(context_p50=62_000.0, context_p90=150_000.0),
        state=PointState(
            roles={
                "agg": RoleBrief(values={"kv_usage.avg": 0.71, "kv_usage.max": 0.96})
            }
        ),
    )
    assert cap is not None and cap.serving_role == "agg" and cap.notes == ()
    agg = cap.role("agg")
    assert (agg.instances, agg.dp_ranks, agg.per_rank_tokens) == (4, 8, 360_000)
    assert agg.instance_tokens == 2_880_000 and agg.total_tokens == 11_520_000
    assert cap.total_tokens == 11_520_000
    # one request lives on one rank, and context_len caps it
    assert cap.max_request_tokens == 262_144
    assert (cap.requests_at_p50, cap.requests_at_p90) == (185, 76)
    assert (cap.usage_avg, cap.usage_peak) == (0.71, 0.96)
    assert agg.bytes_per_token == pytest.approx(25e9 / 360_000, abs=0.1)
    # FP8 KV (kv-cache-dtype fp8_e4m3): 1 byte per element
    assert cap.arch_bytes_per_token == pytest.approx(
        arch.kv_bytes_per_token(1.0), abs=0.1
    )


def test_without_dp_attention_an_instance_is_one_pool_and_hicache_adds_host_tokens():
    cap = kv_capacity(
        agg_lifecycle(max_total_num_tokens_min=200_000, context_len=None),
        agg_plan(**{"data-parallel-size": 8, "hicache-size": 100}),
    )
    agg = cap.role("agg")
    assert agg.dp_ranks == 1  # data-parallel-size alone does not split attention
    assert agg.instance_tokens == 200_000 and cap.total_tokens == 800_000
    # context_len from the rendered config when the logs lack it
    assert agg.context_len == 262_144 and cap.max_request_tokens == 200_000
    assert agg.host_gb_per_rank == 100.0
    assert agg.host_tokens_total == int(4 * 100e9 // agg.bytes_per_token)
    assert cap.requests_at_p50 is None and cap.arch_bytes_per_token is None


def pd_plan() -> LaunchPlan:
    def worker(role, instance, node):
        return ContainerSpec(
            node=node,
            name=f"cb-{role}-{instance}",
            role=role,
            instance=instance,
            gpus=tuple(range(8)),
            image=factory.IMAGE,
            command=("python3", "-m", "dynamo.sglang"),
        )

    cand = Candidate(family_id="disagg-2p1d")
    containers = (
        worker("prefill", 0, "node0"),
        worker("prefill", 1, "node1"),
        # one decode instance across two nodes (two ranks)
        worker("decode", 0, "node2"),
        worker("decode", 0, "node3").model_copy(update={"rank": 1}),
    )
    return LaunchPlan(
        candidate=cand,
        cand_fp=cand.fp,
        namespace="cb-test",
        served_model_name="glm-5.3",
        frontend_url="http://10.42.0.1:8000",
        containers=containers,
        gpu_count=32,
        transport="nixl",
        cross_node=(),
        expected_instances={"prefill": 2, "decode": 1},
        nic_selection={},
        rendered_configs={
            "prefill:0": {"context-length": 131072},
            "prefill:1": {"context-length": 131072},
            "decode:0": {
                "context-length": 131072,
                "enable-dp-attention": True,
                "dp-size": 16,
            },
        },
    )


def test_pd_counts_in_flight_context_on_decode():
    lifecycle = DeployLifecycle(
        roles={
            "prefill": RoleLifecycle(
                role="prefill", containers=2, kv_tokens_min=500_000
            ),
            "decode": RoleLifecycle(role="decode", containers=2, kv_tokens_min=150_000),
        }
    )
    cap = kv_capacity(
        lifecycle, pd_plan(), requests=RequestSummary(context_p50=60_000.0)
    )
    assert cap.serving_role == "decode" and [r.role for r in cap.roles] == [
        "prefill",
        "decode",
    ]
    dec = cap.role("decode")
    assert (dec.instances, dec.dp_ranks) == (1, 16)  # ranks of one instance count once
    assert cap.total_tokens == 150_000 * 16 and cap.requests_at_p50 == 40
    assert cap.max_request_tokens == 131_072  # context-length from the config
    assert cap.role("prefill").total_tokens == 1_000_000
    assert cap.usage_avg is None  # no point state


def test_missing_pool_size_is_reported_not_guessed():
    cap = kv_capacity(
        agg_lifecycle(max_total_num_tokens_min=None, kv_tokens_min=None),
        agg_plan(),
        requests=RequestSummary(context_p50=62_000.0),
    )
    assert cap.total_tokens is None and cap.requests_at_p50 is None
    assert cap.notes == ("no KV pool size in the cold-start logs",)
    assert kv_capacity(None, agg_plan()) is None
    assert kv_capacity(agg_lifecycle(), None) is None
    assert kv_capacity(DeployLifecycle(), agg_plan()) is None
