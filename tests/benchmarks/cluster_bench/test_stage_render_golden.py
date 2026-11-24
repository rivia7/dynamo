# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E, P-Only and D-Only render exactly as before the A-Only stage was added.

One hash per (model, GPU, stage, node count, family, MTP) of everything a
launch plan carries except the candidate itself (containers with their env,
files and argv, rendered SGLang configs, notes), against
fixtures/stages/render_hashes.json. Regenerate it only for an intended
rendering change:

    CB_UPDATE_RENDER_HASHES=1 python3 -m pytest \\
        tests/benchmarks/cluster_bench/test_stage_render_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from benchmarks.cluster_bench.adapters import profile_store
from benchmarks.cluster_bench.adapters.engine_sglang import SglangEngine
from benchmarks.cluster_bench.app.simulate import synthetic_inventory
from benchmarks.cluster_bench.domain import profiles as prof
from benchmarks.cluster_bench.domain import stages
from benchmarks.cluster_bench.domain.schema import Candidate, RunConfig

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURE = Path(__file__).parent / "fixtures" / "stages" / "render_hashes.json"
UPDATE_ENV = "CB_UPDATE_RENDER_HASHES"
MODELS = {
    "glm-5.3": ("b200", "b300", "h100", "h200", "rtx_pro_6000", "rtx_pro_6000d"),
    "glm-5.3-flash": ("rtx_pro_6000d",),
}
PINNED_STAGES = ("e2e", "prefill", "decode")
NODE_COUNTS = (2, 4)
RECIPES = profile_store.RecipeLoader()


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def render_hashes() -> dict[str, str]:
    gpus = profile_store.load_gpu_profiles()
    out: dict[str, str] = {}
    for model_name, gpu_names in MODELS.items():
        model = profile_store.load_model_profile(model_name)
        for gpu_name in gpu_names:
            base = profile_store.load_deployment_profile(model_name, gpu_name)
            for stage in PINNED_STAGES:
                staged = stages.stage_deployment(base, stage, RECIPES.role_config)
                for nodes in NODE_COUNTS:
                    fitted, _dropped = prof.fit_deployment(staged, nodes * 8, 8)
                    inventory = synthetic_inventory(gpu_name, nodes=nodes)
                    for family in fitted.families:
                        for mtp in (True, False):
                            key = (
                                f"{model_name}/{gpu_name}/{stage}/{nodes}n/"
                                f"{family.id}/mtp{int(mtp)}"
                            )
                            try:
                                plan = SglangEngine().render(
                                    Candidate(
                                        family_id=family.id,
                                        mtp=mtp,
                                        simulate_acceptance=stage == "decode",
                                    ),
                                    config=RunConfig(
                                        exp_id="exp-golden",
                                        image="img:golden",
                                        workdir="/wd",
                                        stage=stage,
                                    ),
                                    inventory=inventory,
                                    gpu=gpus[gpu_name],
                                    model=model,
                                    deployment=fitted,
                                    model_dir="/workspace/models/m",
                                )
                            except Exception as exc:  # pinned as well
                                out[key] = f"error:{type(exc).__name__}:{exc}"
                                continue
                            out[key] = _digest(
                                plan.model_dump(mode="json", exclude={"candidate"})
                            )
    return out


def test_existing_stages_render_byte_identical() -> None:
    actual = render_hashes()
    if os.environ.get(UPDATE_ENV) == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"wrote {len(actual)} render hashes to {FIXTURE.name}")
    expected = json.loads(FIXTURE.read_text())
    changed = sorted(
        k for k in expected.keys() & actual.keys() if expected[k] != actual[k]
    )
    missing = sorted(expected.keys() - actual.keys())
    added = sorted(actual.keys() - expected.keys())
    assert (changed, missing, added) == (
        [],
        [],
        [],
    ), f"{len(changed)} changed, {len(missing)} missing, {len(added)} new renders"
