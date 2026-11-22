# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiles, NIC planning and the SGLang renderer (pure; no GPUs)."""

from __future__ import annotations

import collections

import pytest
import yaml

from benchmarks.lium_cluster_bench.adapters import profile_store
from benchmarks.lium_cluster_bench.adapters.engine_sglang import (
    SglangEngine,
    config_engagement,
)
from benchmarks.lium_cluster_bench.app.simulate import synthetic_inventory
from benchmarks.lium_cluster_bench.domain import nicplan
from benchmarks.lium_cluster_bench.domain import profiles as prof
from benchmarks.lium_cluster_bench.domain.schema import Candidate, GridSpec, RunConfig

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

GPUS = ("b200", "b300", "h200", "h100", "rtx_pro_6000")
MODEL_DIR = "/workspace/models/glm"


def _config(tmp_path) -> RunConfig:
    return RunConfig(exp_id="exp-1", image="img:1", workdir=str(tmp_path))


def _render(tmp_path, gpu_name: str, cand: Candidate, nodes: int = 4):
    gpus = profile_store.load_gpu_profiles()
    deployment = profile_store.load_fitted_profile("glm-5.3", gpu_name, nodes=nodes)
    return SglangEngine().render(
        cand,
        config=_config(tmp_path),
        inventory=synthetic_inventory(gpu_name, nodes=nodes),
        gpu=gpus[gpu_name],
        model=profile_store.load_model_profile("glm-5.3"),
        deployment=deployment,
        model_dir=MODEL_DIR,
    )


def _worker_configs(plan):
    out = []
    for spec in plan.containers:
        if spec.role in ("control", "frontend"):
            continue
        cfg_path = next(p for p in spec.files if p.endswith(".yaml"))
        out.append((spec, yaml.safe_load(spec.files[cfg_path])))
    return out


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA B200", "b200"),
        ("NVIDIA B300 SXM6 AC", "b300"),
        ("NVIDIA H200", "h200"),
        ("NVIDIA H100 80GB HBM3", "h100"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "rtx_pro_6000"),
    ],
)
def test_gpu_detection(name, expected):
    assert (
        prof.detect_gpu_profile(name, profile_store.load_gpu_profiles()).name
        == expected
    )


def test_gb200_is_not_mistaken_for_b200():
    with pytest.raises(prof.ProfileError):
        prof.detect_gpu_profile("NVIDIA GB200", profile_store.load_gpu_profiles())


@pytest.mark.parametrize("gpu", GPUS)
def test_every_profile_family_resolves_recipes_and_fits(gpu):
    deployment = profile_store.load_deployment_profile("glm-5.3", gpu)
    recipes = profile_store.RecipeLoader()
    for family in deployment.families:
        sizes = family.nodes or (4,)
        for nodes in sizes:
            fitted = prof.fit_family(family, nodes * 8, 8)
            assert fitted is not None, (prof.reference_id(family), nodes)
            assert "{" not in fitted.id and "{" not in fitted.title
        for role in family.roles:
            if role.base:
                assert "model-path" in recipes.config(role.base)


@pytest.mark.parametrize("gpu", GPUS)
@pytest.mark.parametrize("nodes", [2, 4, 8])
def test_every_cluster_size_has_a_baseline_and_a_cross_node_family(gpu, nodes):
    fitted = profile_store.load_fitted_profile("glm-5.3", gpu, nodes=nodes)
    enabled = [f for f in fitted.families if f.enabled]
    assert enabled, (gpu, nodes)
    assert any(f.cross_node for f in enabled), "the NIC sweep needs fabric traffic"
    ids = [f.id for f in fitted.families]
    assert len(ids) == len(set(ids))
    assert all(f.total_gpus == nodes * 8 for f in fitted.families)


def test_b300_inherits_b200_families_but_has_its_own_ladder():
    b200 = profile_store.load_deployment_profile("glm-5.3", "b200")
    b300 = profile_store.load_deployment_profile("glm-5.3", "b300")
    assert [f.id for f in b200.families] == [f.id for f in b300.families]
    assert b300.gpu == "b300"
    assert "kernel-flashmla-dsa" in {r.id for r in b300.ladder}


def test_hopper_serves_256k_corpus_context():
    for gpu in ("h200", "h100", "rtx_pro_6000"):
        assert (
            profile_store.load_deployment_profile("glm-5.3", gpu).served_context
            >= 262144
        )


def test_merge_config_deletes_on_none_and_drop_flags():
    merged = prof.merge_config({"a": 1, "b": 2}, {"b": None, "c": 3})
    assert merged == {"a": 1, "c": 3}
    args = ("--x", "1", "--dsa-prefill-backend", "trtllm", "--flag", "--y=2")
    assert prof.drop_flags(args, ("--dsa-prefill-backend", "--y")) == (
        "--x",
        "1",
        "--flag",
    )


def _raw_family(gpu: str, ref: str):
    deployment = profile_store.load_deployment_profile("glm-5.3", gpu)
    return next(f for f in deployment.families if prof.reference_id(f) == ref)


def test_fit_family_scales_and_renames_for_two_and_eight_nodes():
    family = _raw_family("b200", "pd-4p2d")
    two = prof.fit_family(family, 16, 8)
    assert [r.instances for r in two.roles] == [2, 1]
    assert (two.id, two.base_id, two.scale) == ("pd-2p1d", "pd-4p2d", 0.5)
    assert two.title == "PD 分离 2P(DEP4) + 1D(DTP8)"
    eight = prof.fit_family(family, 64, 8)
    assert (eight.id, eight.scale) == ("pd-8p4d", 2.0)
    assert eight.matches("pd-4p2d") and eight.matches("pd-8p4d")
    # Refitting a fitted family for the same cluster changes nothing.
    assert prof.fit_family(eight, 64, 8) == eight
    assert prof.fit_family(_raw_family("b200", "pd-6p1d"), 16, 8) is None


def test_node_restricted_families_only_fit_their_sizes():
    dep16 = _raw_family("b200", "wide-ep-dep16")
    assert prof.fit_family(dep16, 16, 8) is not None
    assert prof.fit_family(dep16, 32, 8) is None
    fitted, dropped = prof.fit_deployment(
        profile_store.load_deployment_profile("glm-5.3", "b200"), 16, 8
    )
    assert "pd-6p1d" in dropped and "wide-ep-dep64" in dropped
    assert {"pd-2p1d", "pd-3p1d-dtp4", "pd-1p3d-dtp4"} <= {
        f.id for f in fitted.families
    }


def test_grids_scale_with_the_cluster():
    grid = GridSpec(start=64, max=1024, min=8)
    assert prof.scale_grid(grid, 2.0, 1024) == GridSpec(start=128, max=2048, min=8)
    assert prof.scale_grid(grid, 0.5, 1024) == GridSpec(start=32, max=512, min=8)
    assert prof.scale_grid(grid, 1.0, 256) == GridSpec(start=64, max=256, min=8)


def test_generic_profile_sizes_tp_from_weights():
    gpu = profile_store.load_gpu_profiles()["h100"]
    model = prof.generic_model_profile("org/m", "main", 140.0)
    deployment = prof.generic_deployment_profile(
        gpu, model, nodes=2, gpus_per_node=8, served_context=32768
    )
    assert deployment.families[0].id == "agg-tp4"
    assert any(f.id.startswith("pd-") for f in deployment.families)


def test_nic_plan_spreads_across_numa_and_balances():
    node = synthetic_inventory("h200").nodes[0]
    assert nicplan.affine_pairs(node) == {g: f"mlx5_{g}" for g in range(8)}
    plan2 = nicplan.plan_node(node, 2)
    assert plan2.hcas == ("mlx5_0", "mlx5_4")
    assert collections.Counter(plan2.gpu_to_hca.values()) == {"mlx5_0": 4, "mlx5_4": 4}
    assert plan2.shared
    full = nicplan.plan_node(node, None)
    assert not full.shared and full.worst_link == "PIX"


def test_nic_plan_pcie_only_four_nics():
    node = synthetic_inventory("rtx_pro_6000").nodes[0]
    plan = nicplan.plan_node(node, 2)
    assert plan.hcas == ("mlx5_0", "mlx5_2")
    inv = synthetic_inventory("rtx_pro_6000")
    assert nicplan.feasible_counts(inv, (8, 4, 2, 1)) == (4, 2, 1)


@pytest.mark.parametrize("gpu", GPUS)
@pytest.mark.parametrize("nodes", [2, 4, 8])
def test_render_all_families(tmp_path, gpu, nodes):
    deployment = profile_store.load_fitted_profile("glm-5.3", gpu, nodes=nodes)
    for family in deployment.families:
        plan = _render(tmp_path, gpu, Candidate(family_id=family.id), nodes=nodes)
        workers = _worker_configs(plan)
        assert plan.gpu_count == nodes * 8
        used = collections.defaultdict(set)
        ports = collections.defaultdict(set)
        for spec, cfg in workers:
            assert not used[spec.node] & set(spec.gpus), "GPU assigned twice"
            used[spec.node] |= set(spec.gpus)
            for key in ("port", "nccl-port"):
                assert cfg[key] not in ports[spec.node]
                ports[spec.node].add(cfg[key])
            assert spec.system_port not in ports[spec.node]
            ports[spec.node].add(spec.system_port)
            assert cfg["model-path"] == MODEL_DIR
            assert cfg["context-length"] == deployment.served_context
            assert spec.env["DYN_TCP_RPC_HOST"] == spec.env["SGLANG_HOST_IP"]
        prefill_nodes = {s.node for s, _ in workers if s.role == "prefill"}
        decode_nodes = {s.node for s, _ in workers if s.role == "decode"}
        if not decode_nodes:
            continue
        prefill_gpus = sum(len(s.gpus) for s, _ in workers if s.role == "prefill")
        if prefill_gpus % 8 == 0:
            assert not prefill_nodes & decode_nodes, "PD must cross nodes"
        else:  # e.g. 3P(DEP4)+1D(DTP4) on 2 nodes: one P shares D's node
            assert any(
                p != d for p in prefill_nodes for d in decode_nodes
            ), "some KV must cross nodes"


def test_multinode_instance_ranks_share_dist_init(tmp_path):
    plan = _render(tmp_path, "h100", Candidate(family_id="agg-tp16"))
    by_instance = collections.defaultdict(list)
    for spec, cfg in _worker_configs(plan):
        by_instance[spec.instance].append((spec, cfg))
    for ranks in by_instance.values():
        assert sorted(cfg["node-rank"] for _, cfg in ranks) == [0, 1]
        assert len({cfg["dist-init-addr"] for _, cfg in ranks}) == 1
        assert [
            s.health_url is not None
            for s, _ in sorted(ranks, key=lambda r: r[1]["node-rank"])
        ] == [True, False]


def test_nic_restriction_is_exact_and_per_instance(tmp_path):
    plan = _render(tmp_path, "b200", Candidate(family_id="pd-4p2d", nic_count=2))
    for spec, cfg in _worker_configs(plan):
        assert spec.env["NCCL_IB_HCA"] == "=mlx5_0,mlx5_4"
        assert cfg["disaggregation-ib-device"] == ",".join(spec.hcas)
        assert set(spec.hcas) <= {"mlx5_0", "mlx5_4"}
    unrestricted = _render(tmp_path, "b200", Candidate(family_id="pd-4p2d"))
    for spec, cfg in _worker_configs(unrestricted):
        assert "NCCL_IB_HCA" not in spec.env
        assert "disaggregation-ib-device" not in cfg


def test_mtp_off_and_simulated_acceptance(tmp_path):
    off = _render(tmp_path, "h200", Candidate(family_id="agg-tp8", mtp=False))
    for _, cfg in _worker_configs(off):
        assert not any(k.startswith("speculative-") for k in cfg)
    sim = _render(
        tmp_path, "b200", Candidate(family_id="pd-4p2d", simulate_acceptance=True)
    )
    for spec, _ in _worker_configs(sim):
        assert ("SGLANG_SIMULATE_ACC_LEN" in spec.env) == (spec.role == "decode")


def test_ladder_rungs_apply(tmp_path):
    plan = _render(
        tmp_path,
        "b300",
        Candidate(family_id="pd-4p2d", rungs=("oom-mem-0.03", "kernel-flashmla-dsa")),
    )
    for spec, cfg in _worker_configs(plan):
        if spec.role == "prefill":
            assert cfg["mem-fraction-static"] == pytest.approx(0.83)
            args = list(spec.command)
            assert args.count("--dsa-prefill-backend") == 1
            assert args[args.index("--dsa-prefill-backend") + 1] == "flashmla_kv"
        if spec.role == "decode":
            assert cfg["cuda-graph-max-bs"] == 128


def test_recipe_parity_of_baseline(tmp_path):
    recipe = profile_store.RecipeLoader().config(
        profile_store.load_deployment_profile("glm-5.3", "h200")
        .family("agg-tp8")
        .roles[0]
        .base
    )
    plan = _render(tmp_path, "h200", Candidate(family_id="agg-tp8"))
    harness_keys = {
        "model-path",
        "context-length",
        "port",
        "nccl-port",
        "kv-events-config",
        "enable-metrics",
    }
    _, cfg = _worker_configs(plan)[0]
    for key, value in recipe.items():
        if key not in harness_keys:
            assert cfg[key] == value, key


def test_hicache_is_clamped_to_host_memory(tmp_path):
    plan = _render(tmp_path, "b200", Candidate(family_id="agg-dtp4"))
    _, cfg = _worker_configs(plan)[0]
    assert cfg["hicache-size"] < 200
    assert any("HiCache clamped" in n for n in plan.notes)


def test_failure_classification_and_server_args():
    engine = SglangEngine()
    assert (
        engine.classify_failure("torch.OutOfMemoryError: CUDA out of memory") == "oom"
    )
    assert (
        engine.classify_failure(
            "no kernel image is available for execution on the device"
        )
        == "kernel_unsupported"
    )
    assert engine.classify_failure("ibv_reg_dmabuf_mr failed") == "rdma_registration"
    assert (
        engine.classify_failure("OSError: [Errno 98] Address already in use")
        == "port_in_use"
    )
    assert engine.classify_failure("all good") == "unknown"
    log = "INFO server_args=ServerArgs(model_path='/m', tp_size=8, dp_size=8, mem_fraction_static=0.84, cuda_graph_bs=[1, 2], speculative_algorithm='EAGLE', disaggregation_ib_device='mlx5_0,mlx5_4')\n"
    args = engine.parse_server_args(log)
    assert args["tp_size"] == "8" and args["cuda_graph_bs"] == "[1, 2]"
    rendered = {
        "tensor-parallel-size": 8,
        "mem-fraction-static": 0.84,
        "speculative-algorithm": "EAGLE",
        "disaggregation-ib-device": "mlx5_0,mlx5_4",
        "context-length": 262144,
    }
    checks = config_engagement(rendered, args)
    assert checks == {
        "tensor-parallel-size": True,
        "mem-fraction-static": True,
        "speculative-algorithm": True,
        "disaggregation-ib-device": True,
    }
    assert config_engagement({"tensor-parallel-size": 4}, args) == {
        "tensor-parallel-size": False
    }


def test_diagnostic_variant_adds_metrics_switches_and_nsys_output(tmp_path):
    dep = profile_store.load_fitted_profile("glm-5.3", "b200")
    family = next(f for f in dep.families if f.enabled)
    base = Candidate(family_id=family.id)
    diag = base.model_copy(update={"diag": "nsys"})
    assert diag.key == f"{base.key}|diag=nsys" and diag.fp != base.fp
    plain = _worker_configs(_render(tmp_path, "b200", base))
    assert all("enable-mfu-metrics" not in cfg for _, cfg in plain)
    workers = _worker_configs(_render(tmp_path, "b200", diag))
    assert workers
    for spec, cfg in workers:
        assert cfg["enable-mfu-metrics"] is True
        assert cfg["expert-balancedness-report-mode"] == "prometheus"  # GLM-5.3 is MoE
        assert spec.env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] == "1"
        assert spec.env["CB_NSYS_OUTPUT"] == (
            f"/bench/experiments/exp-1/diagnostics/w1/{spec.name}"
        )
    metrics_only = base.model_copy(update={"diag": "metrics"})
    for spec, _ in _worker_configs(_render(tmp_path, "b200", metrics_only)):
        assert "CB_NSYS_OUTPUT" not in spec.env
