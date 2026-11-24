# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A-Only worker side (agents/sglang_afd.py) and its SGLang hotfix, on a fake
sglang package; the null experts module where torch is installed (no GPUs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.cluster_bench.agents import sglang_afd as agent
from benchmarks.cluster_bench.agents import sglang_hotfix as hotfix
from benchmarks.cluster_bench.domain import afd, stages
from tests.benchmarks.cluster_bench.test_afd_model import plan

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

PLAN = plan()
GATE = {agent.ATTN_ONLY_ENV: "1", agent.TARGET_LAYERS_ENV: "78"}


def test_the_agent_and_the_harness_share_the_env_contract() -> None:
    assert (agent.ATTN_ONLY_ENV, agent.TARGET_LAYERS_ENV, agent.F_WAIT_ENV) == (
        afd.ATTN_ONLY_ENV,
        afd.TARGET_LAYERS_ENV,
        afd.F_WAIT_ENV,
    )
    assert agent.F_WAIT_OFF == afd.F_WAIT_OFF
    assert afd.HOTFIX_ID in hotfix.HOTFIXES
    assert stages.HOTFIX_ENV == hotfix.ENV_VAR


@pytest.mark.parametrize("n_a", [4, 8])
@pytest.mark.parametrize("rows", [1, 2, 3, 64, 1024, 4096])
@pytest.mark.parametrize("first", [True, False])
def test_the_worker_waits_what_the_harness_modelled(n_a: int, rows: int, first: bool):
    modelled = afd.f_wait(PLAN, n_a)
    parsed = agent.parse_f_wait(modelled.to_env())
    assert parsed is not None
    assert parsed.budget(rows, first) == afd.wait_budget(modelled, rows, first)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "lat_ns=1",
        "lat_ns=1,lat_ns=1,row_ns=1,wt_ns=1,flop_ns=1,top_k=8,experts=256,m=2",
        "lat_ns=-1,row_ns=1,wt_ns=1,flop_ns=1,top_k=8,experts=256,m=2",
        "lat_ns=nan,row_ns=1,wt_ns=1,flop_ns=1,top_k=8,experts=256,m=2",
        "lat_ns=1,row_ns=1,wt_ns=1,flop_ns=1,top_k=9,experts=8,m=2",
        "lat_ns=1,row_ns=1,wt_ns=1,flop_ns=1,top_k=8,experts=256,m=0",
        "lat_ns=1,row_ns=1,wt_ns=1,flop_ns=1,top_k=8.5,experts=256,m=2",
    ],
)
def test_a_bad_wait_fails_the_worker_at_startup(text: str) -> None:
    with pytest.raises(ValueError, match=agent.F_WAIT_ENV):
        agent.parse_f_wait(text)


def test_off_is_the_null_f_bound() -> None:
    assert agent.parse_f_wait(" off ") is None


def test_only_target_layers_lose_their_routed_experts(monkeypatch) -> None:
    target = (
        "model.layers.3.mlp.experts.17.gate_proj.weight",
        "model.layers.77.mlp.experts.255.down_proj.weight_scale_2",
        "model.layers.5.mlp.experts.w13_weight",
    )
    kept = (
        "model.layers.78.mlp.experts.0.down_proj.weight",  # the MTP layer
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.layers.1.mlp.gate_proj.weight",  # a dense layer
        "model.layers.3.self_attn.q_a_proj.weight",
        "lm_head.weight",
    )
    for name in target + kept:
        assert not agent.skip_tensor(name)  # no gate: nothing changes
    assert not agent.nulls_experts(False)
    for key, value in GATE.items():
        monkeypatch.setenv(key, value)
    assert [agent.skip_tensor(n) for n in target] == [True] * len(target)
    assert [agent.skip_tensor(n) for n in kept] == [False] * len(kept)
    assert agent.nulls_experts(False) and not agent.nulls_experts(True)
    monkeypatch.delenv(agent.TARGET_LAYERS_ENV)
    with pytest.raises(RuntimeError, match=agent.TARGET_LAYERS_ENV):
        agent.skip_tensor(target[0])


# The hotfix on a fake sglang package: SGLang 0.5.21's anchors in modules
# that compile.
DSV2 = (
    "from torch import nn\n\n\n"
    "def get_moe_impl_class(quant_config):\n    return nn.Identity\n\n\n"
    + hotfix._DSV2_MOE_OLD
    + "        self, quant_config=None, is_nextn=False\n    ):\n"
    + "        num_experts_for_moe = 256\n"
    + hotfix._DSV2_EXPERTS_OLD
    + "        )\n"
    + "        topk_kwargs = dict(\n"
    + hotfix._DSV2_TOPK_OLD
    + "        )\n"
)
WEIGHTS = (
    "import os\n\n"
    + hotfix._WEIGHTS_SKIP_OLD
    + "\n\ndef safetensors_weights_iterator(files, safe_open):\n"
    "    for st_file in files:\n"
    "        if st_file:\n"
    "            with safe_open(st_file) as f:\n"
    + hotfix._WEIGHTS_ITER_OLD
    + "\n\ndef multi_thread_safetensors_weights_iterator(files, safe_open):\n"
    "    def _load_file(st_file):\n"
    "        if st_file:\n"
    "            with safe_open(st_file) as f:\n"
    + hotfix._WEIGHTS_THREADS_OLD
    + "        return result\n"
)


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "sglang"
    for rel, text in ((hotfix._DSV2, DSV2), (hotfix._WEIGHTS, WEIGHTS)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_the_hotfix_installs_the_agent_and_routes_sglang_to_it(tmp_path) -> None:
    root = _package(tmp_path)
    pending = hotfix.plan([afd.HOTFIX_ID], root)
    assert sorted(str(p.relative_to(root)) for p in pending) == [
        "srt/cb_afd.py",
        "srt/cb_afd_kernel.py",
        hotfix._WEIGHTS,
        hotfix._DSV2,
    ]
    hotfix.write(pending)
    installed = root / "srt/cb_afd.py"
    assert installed.read_text() == Path(agent.__file__).read_text()
    # the agent finds its kernel beside itself under either name
    assert installed.with_name(f"{installed.stem}_kernel.py").exists()
    assert Path(agent.__file__).with_name("sglang_afd_kernel.py").exists()
    dsv2 = (root / hotfix._DSV2).read_text()
    assert "self.experts = _cb_afd_experts_class(quant_config, is_nextn)(" in dsv2
    assert "if _cb_afd_standard_topk(is_nextn)\n" in dsv2  # A computes the top-k
    weights = (root / hotfix._WEIGHTS).read_text()
    assert weights.count("_cb_afd_skip_tensor(") == 3  # the helper and two iterators
    assert hotfix.plan([afd.HOTFIX_ID], root) == {}  # a restarted container


def test_the_hotfix_refuses_an_sglang_without_its_anchors(tmp_path) -> None:
    root = _package(tmp_path)
    (root / hotfix._WEIGHTS).write_text("import os\n")
    with pytest.raises(hotfix.HotfixError, match="occurs 0 times"):
        hotfix.plan([afd.HOTFIX_ID], root)
    with pytest.raises(hotfix.HotfixError, match="does not exist"):
        hotfix.plan([afd.HOTFIX_ID], tmp_path / "empty")


def test_an_edit_that_breaks_the_syntax_is_refused(tmp_path, monkeypatch) -> None:
    root = _package(tmp_path)
    bad = hotfix.Edit(
        hotfix._DSV2, hotfix._DSV2_EXPERTS_OLD, "        self.experts = )(\n"
    )
    monkeypatch.setitem(hotfix.HOTFIXES, "bad", (bad,))
    with pytest.raises(hotfix.HotfixError, match="would not compile"):
        hotfix.plan(["bad"], root)


def test_check_without_cuda_reports_unsupported(monkeypatch, capsys) -> None:
    def no_gpu() -> str:
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(agent, "check", no_gpu)
    assert agent.main(["--check"]) == agent.EXIT_UNSUPPORTED
    assert "no CUDA device" in capsys.readouterr().err
    assert agent.main([]) == 2


class _Launches:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def __call__(self, stamp, budget_ns: int, first: bool) -> None:
        self.calls.append((budget_ns, first))


def test_null_experts_wait_then_return_zeros(monkeypatch, capsys) -> None:
    torch = pytest.importorskip("torch")
    for key, value in GATE.items():
        monkeypatch.setenv(key, value)
    wait = afd.f_wait(PLAN, 8)
    monkeypatch.setenv(agent.F_WAIT_ENV, wait.to_env())
    launches = _Launches()
    process = agent._Process()
    monkeypatch.setattr(agent, "PROCESS", process)
    monkeypatch.setattr(agent, "_launch", launches)
    monkeypatch.setattr(process, "stamp", lambda device: None)
    monkeypatch.setattr(process, "in_dummy_run", lambda: False)
    null = agent.null_experts_class()
    first, later = null(layer_id=3), null(layer_id=4)
    assert launches.calls == [(0, True), (0, False)] * 2  # warm-up, per module
    assert list(first.parameters()) == [] and first.num_local_experts == 0
    marker = agent.marker_line(78, 3, wait.to_env())
    assert capsys.readouterr().out.splitlines() == [marker]
    assert marker.startswith(agent.MARKER) and wait.to_env() in marker
    launches.calls.clear()
    hidden = torch.ones(64, 16)
    out = first(hidden, object())
    assert torch.equal(out, torch.zeros_like(hidden))
    later.forward_impl(hidden, None)
    later(hidden[:0], None)  # no rows: no wait
    expected = [afd.wait_budget(wait, 64, first=f) for f in (True, False)]
    assert launches.calls == [(int(ns), serial) for serial, ns in expected]
    with pytest.raises(RuntimeError, match="disable-shared-experts-fusion"):
        null(layer_id=5, num_fused_shared_experts=1)


def test_null_f_bound_registers_without_a_kernel(monkeypatch, capsys) -> None:
    pytest.importorskip("torch")
    for key, value in GATE.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(agent.F_WAIT_ENV, agent.F_WAIT_OFF)
    launches = _Launches()
    monkeypatch.setattr(agent, "PROCESS", agent._Process())
    monkeypatch.setattr(agent, "_launch", launches)
    agent.null_experts_class()(layer_id=3)
    assert launches.calls == []
    marker = agent.marker_line(78, 3, agent.F_WAIT_OFF)
    assert capsys.readouterr().out.splitlines() == [marker]
    assert marker.endswith("F wait off (null-F bound)")
