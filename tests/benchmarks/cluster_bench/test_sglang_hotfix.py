# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang source hotfixes (agents/sglang_hotfix.py) on a fake sglang package,
the launcher hook and the profiles that name them (pure; no GPUs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.agents import sglang_hotfix as hotfix
from benchmarks.cluster_bench.app.phases.preflight import profile_hotfixes

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

TILELANG = "sm120-tilelang-dsa-v1-one-stage"
(EDIT,) = hotfix.HOTFIXES[TILELANG]
# The end of SGLang 0.5.21's tilelang_sparse_fwd, in a module that compiles.
UPSTREAM = (
    "import torch\n\n\n"
    "def tilelang_sparse_fwd(q, kv, indices, num_heads, d_v, tail_dim, topk, sm_scale):\n"
    "    if _is_hip:\n"
    "        out = None\n"
    "    else:\n"
    "        kernel_factory = pick_kernel(tail_dim)\n" + EDIT.old
)
LAUNCH_SCRIPT = Path(hotfix.__file__).resolve().parents[1] / "scripts/launch_node.sh"


def _package(tmp_path: Path, text: str = UPSTREAM) -> Path:
    root = tmp_path / "sglang"
    path = root / EDIT.path
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return root


def test_plan_edits_the_kernel_once_and_write_applies_it(tmp_path):
    root = _package(tmp_path)
    path = root / EDIT.path
    pending = hotfix.plan([TILELANG], root)
    assert list(pending) == [path]
    new = pending[path]
    assert "get_device_capability(q.device)[0] == 12" in new and "num_stages=1" in new
    compile(new, str(path), "exec")
    assert path.read_text() == UPSTREAM  # planning writes nothing
    hotfix.write(pending)
    assert path.read_text() == new
    assert hotfix.plan([TILELANG], root) == {}  # applied: nothing left


def test_write_drops_the_bytecode_of_edited_files_only(tmp_path):
    root = _package(tmp_path)
    cache = (root / EDIT.path).parent / "__pycache__"
    cache.mkdir()
    stale = cache / "tilelang_kernel.cpython-312.pyc"
    other = cache / "tilelang_kernel_v2.cpython-312.pyc"
    stale.write_bytes(b"pyc")
    other.write_bytes(b"pyc")
    hotfix.write(hotfix.plan([TILELANG], root))
    assert not stale.exists() and other.exists()


@pytest.mark.parametrize("copies", [0, 2])
def test_a_hotfix_that_does_not_fit_fails_and_writes_nothing(tmp_path, copies):
    text = "def f():\n    pass\n" + EDIT.old * copies
    root = _package(tmp_path, text)
    with pytest.raises(hotfix.HotfixError, match=f"occurs {copies} times"):
        hotfix.plan([TILELANG], root)
    assert (root / EDIT.path).read_text() == text


def test_unknown_ids_and_missing_files_fail(tmp_path):
    with pytest.raises(hotfix.HotfixError, match="unknown hotfix 'nope'"):
        hotfix.plan(["nope"], tmp_path)
    with pytest.raises(hotfix.HotfixError, match="cannot read"):
        hotfix.plan([TILELANG], tmp_path)


def test_ids_are_a_comma_list_without_repeats():
    assert hotfix.parse_ids(" a, b,,a ") == ("a", "b")
    assert hotfix.parse_ids("") == ()


def test_cli_check_reports_and_apply_writes(tmp_path, monkeypatch, capsys):
    root = _package(tmp_path)
    monkeypatch.setattr(hotfix, "sglang_dir", lambda: root)
    assert hotfix.main(["check", TILELANG]) == 0
    assert (root / EDIT.path).read_text() == UPSTREAM
    assert f"fits; would edit {EDIT.path}" in capsys.readouterr().out
    assert hotfix.main(["apply", f"{TILELANG},{TILELANG}"]) == 0
    assert EDIT.new in (root / EDIT.path).read_text()
    assert hotfix.main(["apply", TILELANG]) == 0  # a restarted container
    assert "edited nothing (applied)" in capsys.readouterr().out
    assert hotfix.main(["check", "nope"]) == 1
    assert "unknown hotfix" in capsys.readouterr().err


def test_the_launcher_applies_hotfixes_before_the_worker_starts():
    text = LAUNCH_SCRIPT.read_text()
    applied = text.index('apply "$CB_SGLANG_HOTFIXES"')
    assert "sglang_hotfix.py" in text[applied - 200 : applied]
    assert applied < text.index('python3 "${worker_entry[@]}"')


def test_profiles_name_only_known_hotfixes():
    named = {}
    for path in sorted((profile_store.PROFILES_DIR / "families").glob("*/*.yaml")):
        model, gpu = path.parent.name, path.stem
        deployment = profile_store.load_deployment_profile(model, gpu)
        named[model, gpu] = profile_hotfixes(
            profile_store.load_model_profile(model), deployment
        )
        assert set(named[model, gpu]) <= set(hotfix.HOTFIXES), path
    assert named["glm-5.3-flash", "rtx_pro_6000d"] == (TILELANG,)
    assert named["glm-5.3", "rtx_pro_6000d"] == ()
