# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A-Only's F model, A<->F traffic and A-side config (domain/afd.py; pure)."""

from __future__ import annotations

from typing import get_args

import pytest

from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.app.simulate import synthetic_inventory
from benchmarks.cluster_bench.domain import afd
from benchmarks.cluster_bench.domain.fabric import LinkMatrix, LinkTest
from benchmarks.cluster_bench.domain.modelarch import arch_from_config
from benchmarks.cluster_bench.domain.schema import AfdExpertDtype, AfdFSpec, PeakResult
from benchmarks.cluster_bench.report.reasons import localize
from tests.benchmarks.cluster_bench.test_modelarch import GLM53, GLM53_FLASH

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

GPU = profile_store.load_gpu_profiles()["rtx_pro_6000d"]  # 84 GB, 1.398 TB/s
INVENTORY = synthetic_inventory("rtx_pro_6000d", nodes=2)  # one 25 GbE port per node
GLM = arch_from_config(GLM53)
FLASH = arch_from_config(GLM53_FLASH)


def plan(spec: AfdFSpec = AfdFSpec(bw_gbs=50.0, lat_us=14.0), **kw):
    args = dict(
        arch=GLM, gpu=GPU, dtype="nvfp4", inventory=INVENTORY, peaks=(), links=None
    )
    return afd.resolve_plan(spec, **{**args, **kw})


def gdr_peak(per_node_gbs: float, nic_count: int | None = None, **details):
    return PeakResult(
        plane="ib",
        test="gdr_send_recv",
        scope="inter",
        nic_count=nic_count,
        value_gbs=per_node_gbs,
        tool="nccl",
        details={"per_node_gbs": per_node_gbs, "nnodes": 2, **details},
    )


def test_afd_f_text_reads_every_key_and_round_trips() -> None:
    spec = afd.parse_afd_f(
        "gpus=16, link=intra,layout=DIRECT,microbatches=3,wire=fp8,"
        "bw_gbps=400,lat_us=5,eff_mem=0.6,eff_flop=0.4,mem_fraction=0.9,"
        "expert_dtype=FP8"
    )
    assert spec == AfdFSpec(
        gpus=16,
        link="intra",
        layout="direct",
        microbatches=3,
        wire="fp8",
        bw_gbs=50.0,
        lat_us=5.0,
        eff_mem=0.6,
        eff_flop=0.4,
        mem_fraction=0.9,
        expert_dtype="fp8",
    )
    assert afd.parse_afd_f(afd.format_afd_f(spec)) == spec
    keys = (*afd._INT_KEYS, *afd._FLOAT_KEYS, *afd._TEXT_KEYS)
    assert sorted(keys) == sorted(AfdFSpec.model_fields)  # every field has a key
    assert afd.parse_afd_f("") == AfdFSpec()
    assert afd.format_afd_f(AfdFSpec()) == ""
    assert afd.format_afd_f(AfdFSpec(bw_gbs=0.1 + 0.2)) == "bw_gbs=0.30000000000000004"


@pytest.mark.parametrize(
    "text, match",
    [
        ("gpus", "not key=value"),
        ("speed=1", "unknown key 'speed'"),
        ("bw_gbs=1,bw_gbps=8", "given twice"),
        ("gpus=x", "not a number"),
        ("microbatches=9", "microbatches"),
        ("link=ib", "link"),
        ("eff_mem=0", "eff_mem"),
    ],
)
def test_bad_afd_f_text_is_refused(text: str, match: str) -> None:
    with pytest.raises(afd.AfdError, match=match):
        afd.parse_afd_f(text)


def test_routed_expert_precision_comes_from_afd_f_or_the_variant() -> None:
    assert afd.routed_dtype("nvfp4-nvidia") == "nvfp4"
    assert afd.routed_dtype("fp8") == "fp8"
    assert afd.routed_dtype("generic", "bf16") == "bf16"  # --model-repo
    with pytest.raises(afd.AfdError, match="--afd-f expert_dtype=nvfp4"):
        afd.routed_dtype("generic")
    # --afd-f offers exactly the precisions the F model knows
    assert set(get_args(AfdExpertDtype)) == set(afd.ROUTED_BYTES_PER_PARAM)


def test_glm_5_3_experts_fill_eight_f_gpus_and_flash_four() -> None:
    layer = afd.routed_layer_bytes(GLM, "nvfp4")
    assert layer == 256 * 3 * 6144 * 2048 * 0.5625  # 5.44 GB per MoE layer
    assert layer * GLM.moe_layers / 1e9 == pytest.approx(407.7, abs=0.1)
    # 51 GB per GPU fits 80% of 84 GB; 102 GB (4 GPUs) does not
    assert afd.default_f_gpus(GLM, "nvfp4", GPU.hbm_gb, 0.8) == 8
    # Flash: 171 GB over 288 experts (powers of two up to 32 divide 288)
    assert afd.default_f_gpus(FLASH, "nvfp4", GPU.hbm_gb, 0.8) == 4
    assert afd.default_f_gpus(GLM, "bf16", GPU.hbm_gb, 0.8) == 32
    with pytest.raises(afd.AfdError, match="do not fit"):
        afd.default_f_gpus(GLM, "bf16", 1.0, 0.8)


def test_fan_out_and_row_bytes() -> None:
    assert afd.fan_out("staged", 8, 8) == 8
    assert afd.fan_out("direct", 8, 8) == pytest.approx(8 * (1 - (7 / 8) ** 8))
    assert afd.fan_out("direct", 1, 8) == 1
    # BF16 hidden state plus an int32 id and an fp32 weight per selected expert
    assert afd.row_bytes(6144, 8, "bf16") == (6144 * 2 + 64, 6144 * 2)
    assert afd.row_bytes(6144, 8, "fp8") == (6144 + 64, 6144)


def test_wait_constants_of_glm_5_3_on_rtx_pro_6000d() -> None:
    p = plan()
    assert (p.f_gpus, p.f_peak_precision, p.f_peak_tflops) == (8, "fp4", 1660.0)
    wait = afd.f_wait(p, n_a=8)
    assert wait.lat_ns == 28_000  # two 14 us transfers
    # 24,640 B per row to all 8 F ranks over 8 A GPUs x 50 GB/s
    assert wait.row_ns == pytest.approx((12352 + 12288) * 8 / 400)
    # 5.44 GB of experts from 8 GPUs' HBM at 70%: 694 us
    assert wait.wt_ns == pytest.approx(5.436e9 / (8 * 1398 * 0.7), rel=1e-3)
    # 6 k h I FLOPs per row on 8 x 1660 TFLOPS at 50%: 91 ns
    assert wait.flop_ns == pytest.approx(6 * 8 * 6144 * 2048 / (8 * 1660e3 * 0.5))
    assert (wait.top_k, wait.experts, wait.m) == (8, 256, 2)


def test_first_layer_waits_serially_later_layers_from_the_last_result() -> None:
    wait = afd.f_wait(plan(), n_a=8)
    rows = 64
    hit = 1 - (1 - 8 / 256) ** rows
    f_64 = wait.lat_ns + rows * wait.row_ns + max(wait.wt_ns * hit, rows * wait.flop_ns)
    assert afd.f_ns(wait, rows) == pytest.approx(f_64)
    assert afd.wait_budget(wait, rows, first=True) == (True, pytest.approx(f_64))
    serial, ns = afd.wait_budget(wait, rows, first=False)
    assert not serial and ns == pytest.approx(2 * afd.f_ns(wait, 32))
    # one row cannot be split; one microbatch is serial everywhere
    assert afd.wait_budget(wait, 1, first=False)[0]
    one = afd.f_wait(plan(AfdFSpec(bw_gbs=50.0, lat_us=14.0, microbatches=1)), 8)
    assert afd.wait_budget(one, rows, first=False) == (True, pytest.approx(f_64))
    assert afd.wait_budget(wait, 0, first=True) == (True, 0.0)


def test_step_rows_split_the_layer_time_and_name_the_bound() -> None:
    p = plan()
    rows = afd.step_rows(p, 8)
    assert [r.rows for r in rows] == list(afd.STEP_ROWS)
    for row in rows:
        assert row.step_ms * 1e3 == pytest.approx(
            row.serial_us + (GLM.moe_layers - 1) * row.overlap_us
        )
    # 64 rows: F reads 87% of the experts once per microbatch
    assert rows[1].bound == "memory"
    slow = plan(AfdFSpec(bw_gbs=0.5, lat_us=14.0))
    assert afd.step_rows(slow, 8)[-1].bound == "link"


def test_slow_links_warn_and_absurd_ones_stop_the_run() -> None:
    assert afd.check_step_time(plan(), 8) is None
    # the synthetic hosts' one 25 GbE port per node: ~1.6 s per forward at 256 rows
    default = plan(AfdFSpec())
    assert default.link_gbs_per_gpu == pytest.approx(3.125 * 0.8 / 8)
    warning = afd.check_step_time(default, 8)
    assert warning is not None and "mostly measure this link" in warning
    with pytest.raises(afd.AfdError, match="above 10 s"):
        afd.check_step_time(plan(AfdFSpec(bw_gbs=0.01)), 8)


def test_link_choice_prefers_overrides_then_measurements_then_nominals() -> None:
    links = LinkMatrix(
        nodes=("n0", "n1"),
        tests=(
            LinkTest(src="n0", dst="n1", rail=0, lat_typical_us=3.0),
            LinkTest(src="n1", dst="n0", rail=0, lat_typical_us=5.0),
        ),
    )
    measured = plan(
        AfdFSpec(),
        peaks=[
            gdr_peak(10.0, nic_count=1),
            gdr_peak(40.0),
            gdr_peak(99.0, engaged=False),
        ],
        links=links,
    )
    assert measured.link_gbs_per_gpu == pytest.approx(40.0 / 8)
    assert "gdr_send_recv (all NICs)" in measured.link_source
    assert measured.lat_us == pytest.approx(4.0 + afd.NET_SOFTWARE_US)
    nic = plan(AfdFSpec(), peaks=[])
    assert nic.link_gbs_per_gpu == pytest.approx(3.125 * afd.NIC_EFFICIENCY / 8)
    assert "no gdr_send_recv peak" in nic.link_source
    assert (nic.lat_us, nic.lat_source) == (20.0, "default, not measured")
    intra = plan(AfdFSpec(link="intra"))
    assert intra.link_gbs_per_gpu == pytest.approx(64.0 * afd.P2P_EFFICIENCY)
    assert "PCIe" in intra.link_source and intra.lat_us == 10.0
    ring = PeakResult(
        plane="pcie",
        test="p2p_ring_aggregate",
        scope="intra",
        value_gbs=200.0,
        tool="p2p_bench",
        details={"per_gpu_gbs": 25.0},
    )
    assert plan(AfdFSpec(link="intra"), peaks=[ring]).link_gbs_per_gpu == 25.0
    override = plan(AfdFSpec(bw_gbs=7.0, lat_us=2.0), peaks=[gdr_peak(40.0)])
    assert (override.link_gbs_per_gpu, override.lat_us) == (7.0, 2.0)


def test_plans_refuse_dense_models_and_explain_oversized_f() -> None:
    dense = GLM.model_copy(update={"routed_experts": 0, "experts_per_token": 0})
    with pytest.raises(afd.AfdError, match="MoE model"):
        plan(arch=dense)
    with pytest.raises(afd.AfdError, match="does not divide"):
        plan(AfdFSpec(gpus=3, bw_gbs=50.0))
    small = plan(AfdFSpec(gpus=2, bw_gbs=50.0))
    assert small.f_gpus == 2 and any("more than 80%" in n for n in small.notes)


def test_plan_sources_and_notes_read_in_chinese() -> None:
    links = LinkMatrix(
        nodes=("n0", "n1"),
        tests=(LinkTest(src="n0", dst="n1", rail=0, lat_typical_us=3.0),),
    )
    ring = PeakResult(
        plane="pcie",
        test="p2p_ring_aggregate",
        scope="intra",
        value_gbs=200.0,
        tool="p2p_bench",
        details={"per_gpu_gbs": 25.0},
    )
    peaks = {k: v for k, v in GPU.peak_dense_tflops.items() if k != "fp4"}
    no_fp4 = GPU.model_copy(update={"peak_dense_tflops": peaks})
    plans = (
        plan(AfdFSpec(), peaks=[gdr_peak(40.0)], links=links),
        plan(AfdFSpec(), peaks=[gdr_peak(10.0, nic_count=1)]),
        plan(AfdFSpec()),
        plan(AfdFSpec(link="intra")),
        plan(AfdFSpec(link="intra"), peaks=[ring]),
        plan(AfdFSpec(gpus=2, bw_gbs=50.0)),
        plan(gpu=no_fp4),
    )
    texts = {afd.check_step_time(plan(AfdFSpec()), 8)}
    for p in plans:
        texts |= {p.f_gpus_source, p.link_source, p.lat_source, *p.notes}
    flags = {t for t in texts if t.startswith("--afd-f ")}  # code spans in the report
    assert flags == {"--afd-f gpus", "--afd-f bw_gbs", "--afd-f lat_us"}
    for text in sorted(texts - flags):
        chinese = localize(text)
        assert chinese != text, text
        # run notes carry the plan's notes behind a prefix (app/afd_ops.py)
        assert chinese in localize(f"A-Only F model: {text}")


def test_traffic_per_output_token_of_glm_5_3() -> None:
    p = plan()
    t = afd.traffic(p, draft_rows=4, accept_len=2.69, sim_tps_per_gpu=100.0)
    # staged: every row to all 8 F ranks, in each of 75 MoE layers, 4/2.69 rows
    # per output token
    assert t.to_f_bytes_per_token == pytest.approx(12352 * 8 * 75 * 4 / 2.69)
    assert t.to_f_bytes_per_token / 1e6 == pytest.approx(11.02, abs=0.01)
    assert t.from_f_bytes_per_token / 1e6 == pytest.approx(10.96, abs=0.01)
    # per A GPU: 100 output tok/s of 11 MB each over a 50 GB/s link
    assert t.link_gbs_per_gpu == 50.0
    assert t.sim_need_gbs_per_gpu == pytest.approx(t.to_f_bytes_per_token * 100 / 1e9)
    assert t.sim_util == pytest.approx(t.sim_need_gbs_per_gpu / 50.0)
    assert t.null_need_gbs_per_gpu is None and t.null_util is None
    direct = plan(AfdFSpec(layout="direct", bw_gbs=50.0, lat_us=14.0))
    no_mtp = afd.traffic(direct, draft_rows=1, accept_len=None)
    assert no_mtp.rows_per_token == 1.0
    assert no_mtp.fan_out == pytest.approx(afd.fan_out("direct", 8, 8))


def test_attention_config_is_dp_attention_with_tp1_ranks() -> None:
    cfg = {
        "tensor-parallel-size": 4,
        "ep-size": 4,
        "enable-eplb": True,
        "moe-a2a-backend": "deepep",
        "deepep-mode": "low_latency",
        "enable-two-batch-overlap": True,
        "weight-loader-prefetch-checkpoints": True,
        "speculative-algorithm": "EAGLE",
        "moe-dense-tp-size": 1,
        "mem-fraction-static": 0.8,
    }
    afd.attention_config(cfg, 8)
    assert cfg == {
        "tensor-parallel-size": 8,
        "data-parallel-size": 8,
        "enable-dp-attention": True,
        "enable-dp-lm-head": True,
        "moe-a2a-backend": "none",
        "disable-shared-experts-fusion": True,
        "speculative-algorithm": "EAGLE",
        "speculative-moe-a2a-backend": "none",
        "moe-dense-tp-size": 1,
        "mem-fraction-static": 0.8,
    }
    again = dict(cfg)
    afd.attention_config(again, 8)
    assert again == cfg
    no_mtp = {"mem-fraction-static": 0.8}
    afd.attention_config(no_mtp, 4)
    assert no_mtp["tensor-parallel-size"] == no_mtp["data-parallel-size"] == 4
    assert "speculative-moe-a2a-backend" not in no_mtp


def test_attention_env_carries_the_wait_or_switches_it_off() -> None:
    env: dict[str, str] = {}
    afd.attention_env(env, None, gpus=8, null_f=False)
    assert env == {afd.ATTN_ONLY_ENV: "1"}
    p = plan()
    afd.attention_env(env, p, gpus=8, null_f=False)
    assert env[afd.TARGET_LAYERS_ENV] == "78"
    assert env[afd.F_WAIT_ENV] == afd.f_wait(p, 8).to_env()
    assert " " not in env[afd.F_WAIT_ENV]
    afd.attention_env(env, p, gpus=8, null_f=True)
    assert env[afd.F_WAIT_ENV] == afd.F_WAIT_OFF


def test_a_side_keeps_router_and_shared_expert_flops_only() -> None:
    a_side = afd.a_side_arch(GLM)
    assert a_side.experts_per_token == 0 and a_side.routed_experts == 256
    routed = GLM.moe_layers * 8 * 3 * 6144 * 2048
    assert GLM.active_params() - a_side.active_params() == pytest.approx(routed)


def test_a_side_weights_show_whether_the_routed_experts_loaded() -> None:
    p = plan()  # GLM-5.3 NVFP4: 407.7 GB of target routed experts
    expected = afd.expected_weights(p, 464.0, 8)
    assert expected is not None
    null_gb, loaded_gb = expected
    # DP attention keeps the rest whole on each A GPU; ep 1 splits the experts
    assert null_gb == pytest.approx(464.0 - p.routed_gb)
    assert loaded_gb == pytest.approx(null_gb + p.routed_gb / 8)
    assert afd.weights_engaged(50.0, expected)
    assert not afd.weights_engaged(100.0, expected)
    assert afd.expected_weights(p, 400.0, 8) is None  # smaller than its experts
