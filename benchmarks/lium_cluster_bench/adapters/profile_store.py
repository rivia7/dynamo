# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load and validate GPU, model and deployment (topology family) profiles
and the recipe ConfigMaps they reference.

Deployment profiles may ``extends:`` another file in the same directory; the
child replaces top-level keys (``families`` and ``ladder`` lists wholesale).
Engine configs reference recipe ConfigMaps (``base: {path, key}``) so a
family stays byte-for-byte aligned with the recipe it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from benchmarks.lium_cluster_bench.domain import profiles as prof
from benchmarks.lium_cluster_bench.domain.profiles import ProfileError
from benchmarks.lium_cluster_bench.domain.schema import (
    DeploymentProfile,
    GpuProfile,
    ModelProfile,
    RecipeRef,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROFILES_DIR = PACKAGE_DIR / "profiles"
REPO_ROOT = PACKAGE_DIR.parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: expected a mapping at top level")
    return data


def load_gpu_profiles(profiles_dir: Path = PROFILES_DIR) -> dict[str, GpuProfile]:
    profiles = {}
    for path in sorted((profiles_dir / "gpus").glob("*.yaml")):
        profile = GpuProfile.model_validate(load_yaml(path))
        profiles[profile.name] = profile
    if not profiles:
        raise ProfileError(f"no GPU profiles under {profiles_dir / 'gpus'}")
    return profiles


def load_model_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> ModelProfile:
    path = profiles_dir / "models" / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(f"no model profile {path}")
    return ModelProfile.model_validate(load_yaml(path))


def _resolve_extends(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in seen:
        raise ProfileError(f"circular extends: {[str(p) for p in (*seen, path)]}")
    data = load_yaml(path)
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data
    parent = _resolve_extends(path.parent / parent_name, (*seen, path))
    parent.update(data)
    return parent


def load_deployment_profile(
    model: str, gpu: str, profiles_dir: Path = PROFILES_DIR
) -> DeploymentProfile:
    path = profiles_dir / "families" / model / f"{gpu}.yaml"
    if not path.exists():
        raise ProfileError(
            f"no tuned families for model {model!r} on {gpu!r} ({path}); "
            "use --model-repo for an untuned generic profile"
        )
    profile = DeploymentProfile.model_validate(_resolve_extends(path))
    ids = [prof.reference_id(f) for f in profile.families]
    if len(ids) != len(set(ids)):
        raise ProfileError(f"{path}: duplicate family ids {ids}")
    return profile


def load_fitted_profile(
    model: str,
    gpu: str,
    *,
    nodes: int = 4,
    gpus_per_node: int = 8,
    profiles_dir: Path = PROFILES_DIR,
) -> DeploymentProfile:
    """The deployment profile fitted to ``nodes`` x ``gpus_per_node`` GPUs."""
    fitted, _ = prof.fit_deployment(
        load_deployment_profile(model, gpu, profiles_dir),
        nodes * gpus_per_node,
        gpus_per_node,
    )
    return fitted


class RecipeLoader:
    """Reads SGLang configs embedded in recipe ConfigMaps (``data.<key>``)."""

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def config(self, ref: RecipeRef) -> dict[str, Any]:
        cache_key = (ref.path, ref.key)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load(ref)
        return dict(self._cache[cache_key])

    def _load(self, ref: RecipeRef) -> dict[str, Any]:
        path = self.repo_root / ref.path
        if not path.exists():
            raise ProfileError(f"recipe {path} not found (sync recipes/ to the node?)")
        with open(path, "r", encoding="utf-8") as fh:
            docs = [d for d in yaml.safe_load_all(fh) if isinstance(d, dict)]
        for doc in docs:
            if doc.get("kind") != "ConfigMap":
                continue
            data = doc.get("data") or {}
            if ref.key in data:
                parsed = yaml.safe_load(data[ref.key])
                if not isinstance(parsed, dict):
                    raise ProfileError(f"{path}#{ref.key} is not a mapping")
                return parsed
        raise ProfileError(f"{path}: no ConfigMap data key {ref.key!r}")
