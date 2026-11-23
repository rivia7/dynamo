# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/kernels.py: nsys stats CSV parsing, kernel-name
classification, breakdowns, merging, coverage and the synthetic tables."""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

import pytest

from benchmarks.cluster_bench.domain import kernels as k
from tests.benchmarks.cluster_bench.lium_fakes import fixture_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "kernels"


def text(name: str) -> str:
    return fixture_text(FIXTURES / name)


def decode_rows() -> list[k.KernelRow]:
    return k.parse_kern_sum_csv(text("kern_sum_decode_b200.csv")) + (
        k.parse_mem_time_sum_csv(text("mem_time_sum_decode_b200.csv"))
    )


def decode_exec() -> list[k.KernelExecRow]:
    return k.parse_kern_exec_sum_csv(text("kern_exec_sum_decode_b200.csv"))


# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------


def test_parse_kern_sum_keeps_quoted_templated_names() -> None:
    rows = k.parse_kern_sum_csv(text("kern_sum_decode_b200.csv"))
    assert len(rows) == 32 and all(r.kind == "kernel" for r in rows)
    first = rows[0]
    assert first.name.startswith("void deep_gemm::sm100_fp8_gemm_1d1d_impl<")
    assert "(deep_gemm::GemmType)2, false, cutlass::bfloat16_t>(int*," in first.name
    assert first.name.endswith("CUtensorMap_st, CUtensorMap_st)")
    assert first.total_us == pytest.approx(742_118.4)
    assert first.instances == 7680
    assert first.avg_us == pytest.approx(96.63)
    assert first.med_us is not None and first.stddev_us is not None
    assert first.min_us is not None and first.max_us is not None
    assert first.min_us < first.avg_us < first.max_us
    names = {r.name for r in rows}
    lambda_copy = next(n for n in names if "direct_copy_kernel_cuda" in n)
    assert "{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>, 4," in lambda_copy
    assert "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)" in names
    assert sum(r.total_us for r in rows) == pytest.approx(3_345_397.76, rel=1e-9)


def test_parse_mem_time_sum() -> None:
    rows = k.parse_mem_time_sum_csv(text("mem_time_sum_decode_b200.csv"))
    assert [r.name for r in rows] == [
        "[CUDA memcpy Device-to-Device]",
        "[CUDA memset]",
        "[CUDA memcpy Host-to-Device]",
        "[CUDA memcpy Device-to-Host]",
    ]
    assert all(r.kind == "memop" for r in rows)
    assert rows[0].total_us == pytest.approx(8123.456) and rows[0].instances == 3200
    # A kernel table is never read as memory operations and vice versa.
    assert k.parse_kern_sum_csv(text("mem_time_sum_decode_b200.csv")) == []
    assert k.parse_mem_time_sum_csv(text("kern_sum_decode_b200.csv")) == []


def test_parse_console_output_with_preamble_and_several_tables() -> None:
    console = text("stats_console.txt")
    kernels = k.parse_kern_sum_csv(console)
    memops = k.parse_mem_time_sum_csv(console)
    assert len(kernels) == 5 and len(memops) == 2
    assert kernels[0].name.startswith("void deep_gemm::sm90_fp8_gemm_1d2d_impl<")
    assert kernels[1].name.startswith("void cutlass::device_kernel<flash::")
    assert memops[1].name == "[CUDA memset]"
    problems = k.stats_problems(console)
    assert len(problems) == 1 and problems[0].startswith("SKIPPED:")
    assert "NVTX" in problems[0]


def test_parse_old_header_format_quoted_everywhere() -> None:
    rows = k.parse_kern_sum_csv(text("kern_sum_v2021.csv"))
    assert len(rows) == 6
    assert rows[0].total_us == pytest.approx(3_114_201.6)
    assert rows[0].avg_us == pytest.approx(2544.2824)
    assert rows[0].min_us is not None and rows[0].max_us is not None
    assert rows[0].med_us is None and rows[0].stddev_us is None


def test_parse_units_thousands_separators_and_doubled_quotes() -> None:
    csv_text = (
        "Time(%),Total Time (us),Instances,Avg (us),Med (us),Min (us),Max (us),"
        "StdDev (us),Name\n"
        '61.0,"4,170.393",10,417.0393,416.7525,415.296,420.864,1.7704,'
        '"void k<""tag"", 2>(int, float)"\n'
        "39.0,2666.0,4,666.5,666.5,600,733,n/a,plain_kernel\n"
        ",,,,,,,,missing_total\n"
        "12.0,100.0,1,100.0,100.0,100.0,100.0,0.0,\n"
    )
    rows = k.parse_kern_sum_csv(csv_text)
    assert [r.name for r in rows] == ['void k<"tag", 2>(int, float)', "plain_kernel"]
    assert rows[0].total_us == pytest.approx(4170.393)
    assert rows[0].min_us == pytest.approx(415.296)
    assert rows[1].stddev_us is None
    seconds = "Total Time (s),Instances,Name\n1.5,3,slow_kernel\n"
    assert k.parse_kern_sum_csv(seconds)[0].total_us == pytest.approx(1.5e6)


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "\n\n",
        "SKIPPED: x.sqlite does not contain CUDA kernel data.\n",
        "Error: unrecognised option '--report=cuda_gpu_kern_sum'\n",
        "Time (%),Name\n1.0,k\n",
    ],
)
def test_parse_tolerates_missing_tables(garbage: str) -> None:
    assert k.parse_kern_sum_csv(garbage) == []
    assert k.parse_mem_time_sum_csv(garbage) == []
    assert k.parse_kern_exec_sum_csv(garbage) == []


def test_parse_kern_exec_sum() -> None:
    rows = decode_exec()
    assert len(rows) == 32 * 8
    assert {r.device for r in rows} == set(range(8))
    assert {r.pid for r in rows} == set(range(48211, 48219))
    first = rows[0]
    assert first.api == "cudaGraphLaunch_v10000" and first.count == 960
    assert first.kernel_avg_us == pytest.approx(96.63)
    assert first.total_us == pytest.approx(960 * 96.63)
    nccl = [r for r in rows if r.name.startswith("ncclDevKernel")]
    assert nccl and all(r.api == "cudaLaunchKernelExC_v11060" for r in nccl)
    # Total time instead of KAvg; a device column like cuda_gpu_trace's.
    alt = "PID,DevId,Count,Total Time (ns),Kernel Name\n7,NVIDIA H200 (3),4,4000,k\n"
    (row,) = k.parse_kern_exec_sum_csv(alt)
    assert row.device == 3 and row.kernel_avg_us == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

DG_GROUPED = (
    "void deep_gemm::sm100_fp8_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::"
    "Major)0, 0u, 4096u, 6144u, 32u, 128u, 160u, (deep_gemm::GemmType)2, false, "
    "cutlass::bfloat16_t>(int*, unsigned int)"
)

# (name, category); "guess" marks names inferred from library sources.
CLASSIFICATION = [
    # GLM-5.3-Flash on SM120 (SGLang 0.5.21): DSA, KDA layers, mHC
    ("main_kernel", "attention"),
    ("fused_sigmoid_gating_delta_rule_update_kernel", "attention"),
    ("kda_replayssm_exact_fold_kernel", "attention"),
    ("_causal_conv1d_update_kernel", "attention"),
    ("layer_norm_gated_fwd_kernel", "norm_act"),
    ("mhc_pre_big_fuse_with_norm_tilelang_kernel", "norm_act"),
    ("mhc_post_tilelang_kernel", "norm_act"),
    (
        "void tensorrt_llm::kernels::cutlass_kernels::"
        "fusedBuildExpertMapsSortFirstTokenKernel<(int)64, (int)8, (int)9>(const int *)",
        "moe_routing",
    ),
    (
        "void deep_gemm::sm120_tf32_hc_prenorm_gemm_impl<(unsigned int)24>(float *)",
        "dense_gemm",
    ),
    (
        "void sglang::<unnamed>::kpool_topk_transform_kernel<(int)512>(int *)",
        "attention",
    ),
    # GLM-5.3 (DSA indexer top-k; a CuTe DSL norm and GEMM)
    (
        "void sglang::topk_main_kernel<(int)2048>(const float *, int *, int)",
        "attention",
    ),
    (
        "kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm"
        "FusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign16o1",
        "norm_act",
    ),
    (
        "kernel_cutlass__dsv3_fused_a_gemm_kernel_tensorptri32gmemo2624307230721",
        "dense_gemm",
    ),
    (DG_GROUPED, "moe_gemm"),
    (DG_GROUPED.replace("GemmType)2", "GemmType)1"), "moe_gemm"),
    (DG_GROUPED.replace("GemmType)2", "GemmType)0"), "dense_gemm"),
    (
        "void deep_gemm::fp8_gemm_kernel<4096u, 7168u, 128u, 128u, 128u, 1u, 8u, "
        "128u, 8u, deep_gemm::GemmType::GroupedMasked>(__nv_bfloat16*, float*, int*)",
        "moe_gemm",
    ),
    ("void deep_gemm::transpose_fp32<128u, 32u>(float const*, float*)", "quantization"),
    (
        "void deep_gemm::sm90_fp8_paged_mqa_logits<64u, 128u>(unsigned int, float*)",
        "attention",
    ),
    ("void deep_gemm::smxx_clean_logits<128u>(unsigned int, float*)", "attention"),
    ("fused_moe_kernel", "moe_gemm"),
    ("fused_moe_kernel_gptq_awq", "moe_gemm"),
    (
        "void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cutlass::"
        "gemm::GroupProblemShape<cute::tuple<long, long, long> >, cutlass::gemm::"
        "collective::CollectiveMma<> > >(Params)",
        "moe_gemm",
    ),
    (
        "bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_cga1x1x1_16dp256b_TN_"
        "transOut_schedS_bN_dsFp8_tma_tmaSf_sm100a",
        "moe_gemm",
    ),
    ("bmm_kernel_persistent", "dense_gemm"),
    ("nvjet_tst_128x64_64x8_1x2_h_bz_TNT", "dense_gemm"),
    (
        "sm90_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize128x128x128_warpgroupsize1x1x1"
        "_execute_segment_k_off_kernel__5x_cublas",
        "dense_gemm",
    ),
    ("ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn", "dense_gemm"),
    (
        "void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, float, float, "
        "128, 16, 4, 4, false, false, true, false, false, cublasGemvParamsEx<int> >"
        "(cublasGemvParamsEx<int>, float, float)",
        "dense_gemm",
    ),
    (
        "void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, "
        "__nv_bfloat16, true, false, false>(cublasLt::cublasSplitKParams<float>)",
        "dense_gemm",
    ),
    (
        "void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<"
        "int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::"
        "MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<4, 2>>, cutlass::gemm::"
        "PersistentScheduler> >(Params)",
        "dense_gemm",
    ),
    ("_w8a8_block_fp8_matmul", "dense_gemm"),
    (
        "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90"
        "<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1> >, 192, "
        "cutlass::bfloat16_t> > > >(Params)",
        "attention",
    ),
    (
        "void flash::flash_fwd_splitkv_mla_kernel<Flash_fwd_kernel_traits_mla<576, 64, "
        "64, 8, cutlass::bfloat16_t, 512>, true>(Flash_fwd_mla_params)",
        "attention",
    ),
    (
        "void flash::flash_fwd_mla_combine_kernel<cutlass::bfloat16_t, float, 512, 64>"
        "(Flash_fwd_mla_params)",
        "attention",
    ),
    (
        "void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::"
        "PosEncodingMode)0, 2u, 4u, 8u, 16u, 1u, 8u>(Params)",
        "attention",
    ),
    (
        "void flashinfer::mla::BatchMLAPagedAttentionKernel<flashinfer::mla::"
        "KernelTraits<true, 2u, 1u, 512u, 64u, 128u>>(Params)",
        "attention",
    ),
    (
        "void cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<cute::tuple<"
        "cute::C<128>, cute::C<128> >, cutlass::bfloat16_t>(Params)",
        "attention",
    ),
    (
        "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvSparseP64Multi"
        "CtasKvCgaVarSeqQ8Kv128PersistentSwapsAbForGen",
        "attention",
    ),
    ("void xqa_kernel_mla<128, 512>(float*, int)", "attention"),
    ("_fwd_kernel_stage1", "attention"),
    ("_fwd_grouped_kernel_stage1", "attention"),
    ("_fwd_kernel", "attention"),
    ("merge_state_kernel", "attention"),
    (
        "void (anonymous namespace)::fast_topk_transform_fused_kernel<2048>(float "
        "const*, int*)",
        "attention",
    ),  # guess
    ("_fwd_kernel_ep_scatter_1", "moe_routing"),
    ("_fwd_kernel_ep_gather", "moe_routing"),
    (
        "void topkGatingSoftmax<4, 256, 4, 16, 64, int, float, 2>(float const*, bool "
        "const*, float*, int)",
        "moe_routing",
    ),
    ("void topk_softmax_kernel<256>(float const*, float*, int*)", "moe_routing"),
    (
        "void moe_fused_gate_kernel<float, 256, 8, 4, 32>(void*, void*, float*, int*)",
        "moe_routing",
    ),
    ("void moe_align_block_size_kernel<int>(int const*, int*, int*)", "moe_routing"),
    ("void moe_sum_reduce_kernel<__nv_bfloat16, 8>(__nv_bfloat16*)", "moe_routing"),
    ("deepep_permute_triton_kernel", "moe_routing"),
    (
        "void dsv3_router_gemm_kernel<__nv_bfloat16, float, 128, 8, 256, 6144>(float*)",
        "moe_routing",
    ),  # guess
    (
        "void moe::dev::routing::routingDeepSeek::routingMainKernel<moe::dev::routing::"
        "KernelParams<__nv_bfloat16> >(KernelParams)",
        "moe_routing",
    ),
    (
        "void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<"
        "__nv_bfloat16, float, false> >(moe::dev::finalize::KernelParams<"
        "__nv_bfloat16, float, false>)",
        "moe_routing",
    ),  # guess
    (
        "void deep_ep::internode_ll::dispatch<true, false, 6144>(void*, float*, int*, "
        "long*, void*, int*)",
        "deepep",
    ),
    ("void deep_ep::intranode::combine<__nv_bfloat16, 8, 768, 4096>(int4*)", "deepep"),
    ("void deep_ep::layout::get_dispatch_layout<256, 32, 8>(long const*)", "deepep"),
    (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        "nccl",
    ),
    ("ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)", "nccl"),
    ("ncclKernel_AllReduce_RING_LL_Sum_half(ncclDevComm*, ncclWork*)", "nccl"),
    (
        "void sglang::cross_device_reduce_1stage<__nv_bfloat16, 8>(sglang::RankData*, "
        "sglang::RankSignals, sglang::Signal*, __nv_bfloat16*, int, int)",
        "other_comm",
    ),
    ("void vllm::cross_device_reduce_2stage<half, 8>(vllm::RankData*)", "other_comm"),
    (
        "void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_"
        "lamport<true, __nv_bfloat16, 8>(Params)",
        "other_comm",
    ),  # guess
    ("void nvshmemi_proxy_progress_kernel(int)", "other_comm"),  # guess
    ("_all_gather_kernel_inner", "other_comm"),
    (
        "void flashinfer::AppendPagedKVCacheKernel<128u, 8u, __nv_bfloat16, int>"
        "(flashinfer::paged_kv_t<__nv_bfloat16, int>, __nv_bfloat16*)",
        "kv_cache",
    ),
    ("set_mla_kv_buffer_kernel", "kv_cache"),
    ("masked_set_kv_buffer_kernel", "kv_cache"),
    ("create_flashinfer_kv_indices_triton", "kv_cache"),
    ("assign_req_to_token_pool", "kv_cache"),
    ("_fused_gather_to_staging_kernel", "kv_cache"),
    (
        "void reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, "
        "(vllm::Fp8KVCacheDataType)0>(__nv_bfloat16 const*)",
        "kv_cache",
    ),
    (
        "void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*,"
        " __nv_bfloat16*)",
        "norm_act",
    ),
    ("_layer_norm_fwd_1pass_kernel", "norm_act"),
    (
        "void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &silu<float> >"
        "(__nv_bfloat16*, __nv_bfloat16 const*, int)",
        "norm_act",
    ),
    (
        "void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<"
        "false, 64u, 8u, 16u, __nv_bfloat16, long>(__nv_bfloat16*)",
        "norm_act",
    ),
    ("_silu_and_mul_post_quant_kernel", "quantization"),
    (
        "void per_token_group_quant_8bit_kernel<__nv_bfloat16, c10::Float8_e4m3fn, "
        "true, false>(__nv_bfloat16 const*, void*, float*)",
        "quantization",
    ),
    ("_per_token_group_quant_fp8", "quantization"),
    (
        "void cvt_fp16_to_fp4<__nv_bfloat16, false>(int, int, __nv_bfloat16 const*)",
        "quantization",
    ),
    (
        "void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (cub::"
        "CUB_300001_SM_1000::BlockScanAlgorithm)2, 4u, true, float, int>(float*, int*)",
        "sampling",
    ),
    (
        "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, "
        "float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*)",
        "sampling",
    ),
    (
        "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, "
        "at::native::ArgMaxOps<float>, unsigned int, long, 4> >(Params)",
        "sampling",
    ),
    ("void VerifyTreeGreedy<int, long>(int*, int*, int*)", "sampling"),  # guess
    ("[CUDA memcpy Device-to-Device]", "memory"),
    ("[CUDA memset]", "memory"),
    (
        "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, "
        "std::array<char*, 1ul> >(int, at::native::FillFunctor<int>)",
        "memory",
    ),
    (
        "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::"
        "(anonymous namespace)::OpaqueType<2u>, unsigned int, 2, 128, 1>(Params)",
        "memory",
    ),
    (
        "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<"
        "c10::BFloat16>, std::array<char*, 3ul> >(int)",
        "elementwise",
    ),
    ("triton_poi_fused_add_mul_0", "elementwise"),
    (
        "void cub::CUB_200700_900_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_900"
        "_NS::DeviceRadixSortPolicy<long, long, unsigned int>::Policy900>(int*)",
        "elementwise",
    ),
    ("compute_position_kernel", "other"),
    ("", "other"),
]


@pytest.mark.parametrize(("name", "category"), CLASSIFICATION)
def test_classification_table(name: str, category: str) -> None:
    assert k.classify_kernel(name) == category


def test_fixture_kernels_are_all_classified_as_expected() -> None:
    by_category: dict[str, set[str]] = {}
    for row in decode_rows():
        cat = "memory" if row.kind == "memop" else k.classify_kernel(row.name)
        by_category.setdefault(cat, set()).add(k.short_kernel_name(row.name, 60))
    assert by_category["other"] == {"compute_position_kernel"}
    assert "deep_ep::internode_ll::dispatch<…>" in by_category["deepep"]
    assert "set_mla_kv_buffer_kernel" in by_category["kv_cache"]
    prefill = k.parse_kern_sum_csv(text("kern_sum_prefill_h200.csv"))
    assert {k.classify_kernel(r.name) for r in prefill} >= {
        "moe_gemm",
        "attention",
        "deepep",
        "dense_gemm",
        "moe_routing",
        "kv_cache",
        "nccl",
        "other_comm",
        "elementwise",
        "memory",
        "quantization",
    }
    assert all(k.classify_kernel(r.name) != "other" for r in prefill)


def test_rules_are_well_formed() -> None:
    ids = [rule.rule_id for rule in k.RULES]
    assert len(ids) == len(set(ids))
    assert {rule.category for rule in k.RULES} <= set(k.CATEGORIES)
    assert set(k.CATEGORY_LABELS) == set(k.CATEGORIES)
    assert len(k.CATEGORIES) == len(set(k.CATEGORIES))
    for rule in k.RULES:
        re.compile(rule.pattern)
    assert k.match_rule("compute_position_kernel") is None
    assert k.match_rule(DG_GROUPED).rule_id == "deep_gemm_grouped"
    # A leading "void " never matters, anchored rules included.
    assert k.classify_kernel("void _fwd_kernel_stage2") == "attention"


@pytest.mark.parametrize(
    ("name", "short"),
    [
        (DG_GROUPED, "deep_gemm::sm100_fp8_gemm_1d1d_impl<…>"),
        (
            "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
            "ncclDevKernel_AllGather_RING_LL",
        ),
        (
            "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::"
            "(anonymous namespace)::OpaqueType<2u>, unsigned int>(at::native::"
            "(anonymous namespace)::OpaqueType<2u>*, int)",
            "at::native::(anonymous namespace)::CatArrayBatchedCopy<…>",
        ),
        ("fused_moe_kernel", "fused_moe_kernel"),
        ("[CUDA memset]", "[CUDA memset]"),
        (  # a CuTe DSL kernel spells its argument types into the name
            "kernel_cutlass__dsv3_fused_a_gemm_kernel_tensorptri32gmemo26243072307"
            "21_tensorptri32gmemo16307230721_tensorptrbf16gmemoalign16o71686144",
            "kernel_cutlass__dsv3_fused_a_gemm_kernel…",
        ),
        (
            "kernel_cutlass_rmsnorm_kernel_Tensorgmemoalign16_1024_Float32_0",
            "kernel_cutlass_rmsnorm_kernel…",
        ),
    ],
)
def test_short_kernel_name(name: str, short: str) -> None:
    assert k.short_kernel_name(name) == short


def test_short_kernel_name_of_lambda_copy_and_truncation() -> None:
    row = next(
        r
        for r in k.parse_kern_sum_csv(text("kern_sum_decode_b200.csv"))
        if "direct_copy_kernel_cuda" in r.name
    )
    assert k.short_kernel_name(row.name) == (
        "at::native::unrolled_elementwise_kernel<…>"
    )
    long_name = "x" * 500
    assert len(k.short_kernel_name(long_name)) == k.SHORT_MAX
    assert k.short_kernel_name(long_name).endswith("…")


# --------------------------------------------------------------------------
# Breakdown, coverage, merge
# --------------------------------------------------------------------------


def test_breakdown_of_a_decode_window() -> None:
    rows = decode_rows()
    bd = k.breakdown(
        rows,
        source="cb-5d1c-decode0-r0",
        exec_rows=decode_exec(),
        expected_devices=8,
    )
    kernels = sum(r.total_us for r in rows if r.kind == "kernel")
    memops = sum(r.total_us for r in rows if r.kind == "memop")
    assert bd.sources == ("cb-5d1c-decode0-r0",)
    assert bd.kernel_us == pytest.approx(kernels, abs=1e-3)
    assert bd.memop_us == pytest.approx(memops, abs=1e-3)
    assert bd.total_us == pytest.approx(kernels + memops, abs=1e-3)
    assert bd.distinct_kernels == 36
    assert bd.instances == sum(r.instances for r in rows)
    times = [c.time_us for c in bd.categories]
    assert times == sorted(times, reverse=True)
    assert sum(c.share for c in bd.categories) == pytest.approx(1.0, abs=1e-5)
    assert [c.category for c in bd.categories[:4]] == [
        "moe_gemm",
        "deepep",
        "attention",
        "dense_gemm",
    ]
    assert bd.share("moe_gemm") == pytest.approx(
        (742_118.4 + 398_622.72) / bd.total_us, abs=1e-5
    )
    assert bd.share("missing") == 0.0
    assert bd.comm_share == pytest.approx(
        bd.share("deepep") + bd.share("nccl") + bd.share("other_comm"), abs=1e-6
    )
    assert 0 < bd.unclassified_share < 1e-4
    memory = next(c for c in bd.categories if c.category == "memory")
    assert memory.kernels == 6  # 4 memory operations + copy + fill kernels
    assert len(bd.top) == k.TOP_N
    assert bd.top[0].category == "moe_gemm" and bd.top[0].instances == 7680
    assert bd.top[0].short == "deep_gemm::sm100_fp8_gemm_1d1d_impl<…>"
    assert [t.time_us for t in bd.top] == sorted(
        (t.time_us for t in bd.top), reverse=True
    )
    assert all(len(t.name) <= k.NAME_MAX for t in bd.top)
    (cov,) = bd.coverage
    assert cov.devices == tuple(range(8)) and cov.processes == 8
    assert cov.complete is True and not bd.partial_coverage
    assert cov.exec_share == pytest.approx(1.0, abs=0.002)
    assert cov.imbalance == pytest.approx(
        max(cov.device_time_us) / (sum(cov.device_time_us) / 8), abs=1e-4
    )
    assert cov.imbalance > 1.02  # EP rank 3 is hot in the fixture
    assert bd.notes == ()


def test_breakdown_flags_partial_rank_coverage() -> None:
    first_gpu = [r for r in decode_exec() if r.device == 0]
    bd = k.breakdown(
        decode_rows(), source="r0", exec_rows=first_gpu, expected_devices=8
    )
    (cov,) = bd.coverage
    assert cov.devices == (0,) and cov.complete is False and bd.partial_coverage
    assert cov.imbalance is None
    assert "r0: kernels on 1 of 8 GPUs inside the capture range" in bd.notes
    assert any("attributes only 12%" in n for n in bd.notes)
    unknown = k.breakdown(decode_rows(), exec_rows=None)
    assert unknown.coverage == () and not unknown.partial_coverage
    no_expectation = k.breakdown(decode_rows(), exec_rows=first_gpu)
    assert no_expectation.coverage[0].complete is None


def test_breakdown_of_nothing_and_duplicates() -> None:
    empty = k.breakdown([], source="w")
    assert empty.total_us == 0 and empty.categories == () and empty.top == ()
    assert empty.notes == ("w: no kernel or memory-operation rows",)
    row = k.KernelRow(name="fused_moe_kernel", total_us=10.0, instances=2)
    twice = k.breakdown([row, row], top_n=1)
    assert twice.distinct_kernels == 1 and twice.top[0].time_us == 20.0
    assert twice.categories[0].share == 1.0
    assert k.breakdown([row], top_n=0).top == ()


def test_merge_rows() -> None:
    a = k.KernelRow(
        name="x",
        total_us=10.0,
        instances=2,
        avg_us=5.0,
        med_us=5.0,
        min_us=4.0,
        max_us=6.0,
        stddev_us=1.0,
    )
    b = k.KernelRow(
        name="x", total_us=30.0, instances=3, avg_us=10.0, min_us=8.0, max_us=14.0
    )
    memop = k.KernelRow(name="x", kind="memop", total_us=1.0, instances=1)
    merged = {(r.kind, r.name): r for r in k.merge_rows([[a, memop], [b]])}
    row = merged[("kernel", "x")]
    assert row.total_us == 40.0 and row.instances == 5 and row.avg_us == 8.0
    assert (row.min_us, row.max_us) == (4.0, 14.0)
    assert row.med_us is None and row.stddev_us is None
    assert merged[("memop", "x")] == memop
    assert k.merge_rows([[a]]) == [a]


def test_merge_breakdowns_matches_exact_categories() -> None:
    rows = decode_rows()
    half = len(rows) // 2
    left = k.breakdown(rows[:half], source="r0")
    right = k.breakdown(rows[half:], source="r1")
    exact = k.breakdown(rows, source="all")
    merged = k.merge_breakdowns([left, right])
    assert merged.sources == ("r0", "r1")
    assert merged.total_us == pytest.approx(exact.total_us, abs=1e-2)
    assert {c.category: c.time_us for c in merged.categories} == pytest.approx(
        {c.category: c.time_us for c in exact.categories}, abs=1e-2
    )
    assert {c.category: c.share for c in merged.categories} == pytest.approx(
        {c.category: c.share for c in exact.categories}, abs=1e-5
    )
    assert merged.top[0].name == exact.top[0].name
    assert merged.top[0].share == pytest.approx(exact.top[0].share, abs=1e-6)
    assert "merged 2 reports; top kernels approximate" in merged.notes


def test_merge_breakdowns_across_reports_and_ranks() -> None:
    decode = k.breakdown(
        decode_rows(), source="decode0-r0", exec_rows=decode_exec(), expected_devices=8
    )
    prefill = k.breakdown(
        k.parse_kern_sum_csv(text("kern_sum_prefill_h200.csv")), source="prefill0-r0"
    )
    merged = k.merge_breakdowns([decode, prefill, decode], top_n=5)
    assert merged.sources == ("decode0-r0", "prefill0-r0")
    assert merged.total_us == pytest.approx(2 * decode.total_us + prefill.total_us)
    assert len(merged.top) == 5
    first = merged.top[0]
    assert first.name.startswith("void deep_gemm::sm90_fp8_gemm_1d2d_impl")
    dg = next(t for t in merged.top if t.name == decode.top[0].name)
    assert dg.instances == 2 * decode.top[0].instances
    assert dg.time_us == pytest.approx(2 * decode.top[0].time_us)
    assert dg.avg_us == pytest.approx(dg.time_us / dg.instances)
    assert len(merged.coverage) == 2
    assert merged.distinct_kernels == max(
        decode.distinct_kernels, prefill.distinct_kernels
    )
    empty = k.merge_breakdowns([])
    assert empty.total_us == 0 and empty.notes == ("no reports to merge",)
    assert k.merge_breakdowns([decode]).notes == decode.notes


def test_breakdown_json_round_trip() -> None:
    bd = k.breakdown(
        decode_rows(), source="s", exec_rows=decode_exec(), expected_devices=8
    )
    again = k.KernelBreakdown.model_validate_json(bd.model_dump_json())
    assert again == bd
    assert again.coverage[0].complete is True


# --------------------------------------------------------------------------
# nsys version and trace method
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "version", "triple"),
    [
        (
            "NVIDIA Nsight Systems version 2025.6.1.190-256162285624v0\n",
            "2025.6.1.190-256162285624v0",
            (2025, 6, 1),
        ),
        (
            "NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0",
            "2024.6.2.225-246235244400v0",
            (2024, 6, 2),
        ),
        ("bash: nsys: command not found", None, None),
    ],
)
def test_parse_nsys_version(output: str, version: str | None, triple) -> None:
    assert k.parse_nsys_version(output) == (version, triple)


@pytest.mark.parametrize(
    ("version", "cc", "expected"),
    [
        ((2025, 6, 1), "10.0", "hardware"),
        ((2026, 1, 0), "10.3", "hardware"),
        ((2026, 1, 0), "12.0", "hardware"),
        ((2025, 5, 1), "10.0", "software"),
        ((2025, 6, 1), "9.0", "software"),
        (None, "10.0", "unknown"),
        ((2025, 6, 1), None, "unknown"),
        ((2025, 6, 1), "sm_100", "unknown"),
    ],
)
def test_expected_cuda_trace(version, cc, expected) -> None:
    assert k.expected_cuda_trace(version, cc) == expected


# --------------------------------------------------------------------------
# Synthetic tables (dry-run simulator)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["decode", "prefill", "agg"])
def test_synthetic_tables_round_trip(role: str) -> None:
    kern = k.synthetic_kern_sum_csv(role, devices=4, steps=5)
    assert kern == k.synthetic_kern_sum_csv(role, devices=4, steps=5)
    rows = k.parse_kern_sum_csv(kern)
    memops = k.parse_mem_time_sum_csv(
        k.synthetic_mem_time_sum_csv(role, devices=4, steps=5)
    )
    exec_rows = k.parse_kern_exec_sum_csv(
        k.synthetic_kern_exec_sum_csv(role, devices=4, steps=5)
    )
    assert rows and memops and exec_rows
    assert {r.device for r in exec_rows} == {0, 1, 2, 3}
    bd = k.synthetic_breakdown(role, devices=4, steps=5, source="sim")
    assert bd.sources == ("sim",)
    assert sum(c.share for c in bd.categories) == pytest.approx(1.0, abs=1e-5)
    assert bd.coverage[0].complete is True
    assert bd.coverage[0].exec_share == pytest.approx(1.0, abs=0.01)
    assert any("simulated" in n for n in bd.notes)
    assert bd.share("moe_gemm") > 0.2 and bd.share("attention") > 0.1
    assert bd.unclassified_share < 0.01
    expected_total = sum(r.total_us for r in rows) + sum(r.total_us for r in memops)
    assert math.isclose(bd.total_us, expected_total, rel_tol=1e-6)


def test_synthetic_window_scales_with_devices_and_steps() -> None:
    small = k.synthetic_breakdown("decode", devices=1, steps=1)
    big = k.synthetic_breakdown("decode", devices=8, steps=10)
    assert big.kernel_us == pytest.approx(80 * small.kernel_us, rel=0.01)
    assert small.kernel_us == pytest.approx(45_000.0, rel=0.01)


# --------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------


def test_domain_module_is_pure() -> None:
    tree = ast.parse(Path(k.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "benchmarks.cluster_bench.domain.schema" not in imported
    local = {m for m in imported if m.startswith("benchmarks")}
    assert local == {"benchmarks.cluster_bench.domain.base"}
    for module in imported - local:
        root = module.split(".")[0]
        assert root in sys.stdlib_module_names or root == "pydantic", module
