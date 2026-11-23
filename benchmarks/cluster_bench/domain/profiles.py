# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure profile logic: GPU detection, config merging, fitting families to a
cluster, and generic (untuned) profiles. Loading YAML profiles and recipe
ConfigMaps from disk lives in ``adapters/profile_store.py``.

Overrides are merged over recipe configs; an override value of ``null``
deletes the key.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from benchmarks.cluster_bench.domain.schema import (
    Checkpoint,
    DeploymentProfile,
    FamilySpec,
    GpuProfile,
    GridSpec,
    ModelProfile,
    RoleSpec,
)


class ProfileError(ValueError):
    pass


# --------------------------------------------------------------------------
# GPU profiles
# --------------------------------------------------------------------------


def detect_gpu_profile(gpu_name: str, profiles: Mapping[str, GpuProfile]) -> GpuProfile:
    """Pick the single profile whose ``match`` regexes hit the nvidia-smi name."""
    hits = [
        p
        for p in profiles.values()
        if any(re.search(pattern, gpu_name, re.IGNORECASE) for pattern in p.match)
    ]
    if len(hits) != 1:
        names = [p.name for p in hits]
        raise ProfileError(
            f"GPU {gpu_name!r} matched {names or 'no'} profile(s); pass --gpu-profile"
        )
    return hits[0]


# --------------------------------------------------------------------------
# Model and deployment profiles
# --------------------------------------------------------------------------


def select_checkpoint(
    model: ModelProfile, gpu: GpuProfile, requested: str | None = None
) -> tuple[str, Checkpoint]:
    """The checkpoint variant to serve and its entry.

    ``requested`` (--checkpoint) wins; else the GPU profile's weights_variant
    when the model has it; else the model's first checkpoint.
    """
    if requested is not None:
        if requested not in model.checkpoints:
            raise ProfileError(
                f"model {model.name} has no checkpoint {requested!r}; "
                f"choose one of {sorted(model.checkpoints)}"
            )
        return requested, model.checkpoints[requested]
    if gpu.weights_variant in model.checkpoints:
        return gpu.weights_variant, model.checkpoints[gpu.weights_variant]
    variant = next(iter(model.checkpoints))
    return variant, model.checkpoints[variant]


# --------------------------------------------------------------------------
# Recipe ConfigMaps and config merging
# --------------------------------------------------------------------------


def merge_config(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Shallow merge; ``None`` deletes a key, nested mappings are replaced."""
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def drop_flags(args: tuple[str, ...], flags: tuple[str, ...]) -> tuple[str, ...]:
    """Remove ``--flag value`` pairs (and bare ``--flag=value``) from an argv."""
    if not flags:
        return args
    out: list[str] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        name = arg.split("=", 1)[0]
        if name in flags:
            has_value = (
                "=" not in arg
                and index + 1 < len(args)
                and not args[index + 1].startswith("--")
            )
            skip_next = has_value
            continue
        out.append(arg)
    return tuple(out)


# --------------------------------------------------------------------------
# Fitting families to the actual cluster
# --------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _template_values(family: FamilySpec) -> dict[str, int]:
    values = {"prefill": 0, "decode": 0, "agg": 0, "gpus": family.total_gpus}
    for role in family.roles:
        values[role.role] += role.instances
    return values


def render_template(template: str, family: FamilySpec) -> str:
    """Fill {prefill}/{decode}/{agg}/{gpus} with ``family``'s instance counts."""

    def substitute(match: re.Match[str]) -> str:
        values = _template_values(family)
        key = match.group(1)
        if key not in values:
            raise ProfileError(
                f"unknown placeholder {{{key}}} in {template!r}; "
                f"use one of {sorted(values)}"
            )
        return str(values[key])

    return _PLACEHOLDER_RE.sub(substitute, template)


def reference_id(family: FamilySpec) -> str:
    """The family's id at the scale its profile was written for."""
    if family.base_id is not None:
        return family.base_id
    return render_template(family.id_template or family.id, family)


def _scaled_roles(
    family: FamilySpec, total_gpus: int, gpus_per_node: int
) -> tuple[RoleSpec, ...] | None:
    for role in family.roles:
        if role.gpus_per_instance > gpus_per_node * role.nodes_per_instance:
            return None
        if (
            role.nodes_per_instance > 1
            and role.gpus_per_instance != gpus_per_node * role.nodes_per_instance
        ):
            return None
    if family.total_gpus == total_gpus:
        return family.roles
    factor = total_gpus / family.total_gpus
    scaled = []
    for role in family.roles:
        count = role.instances * factor
        if count < 1 or not math.isclose(count, round(count)):
            return None
        scaled.append(role.model_copy(update={"instances": int(round(count))}))
    return tuple(scaled)


def fit_family(
    family: FamilySpec, total_gpus: int, gpus_per_node: int
) -> FamilySpec | None:
    """Scale ``family`` so it uses every GPU of the cluster, or return None.

    Profiles are written for one cluster size (4 x 8 GPUs); on 2 or 8 nodes
    every role's instance count scales by the same exact factor, and families
    restricted with ``nodes`` apply only to those sizes. The id and title are
    rendered from their templates with the fitted counts (``pd-{prefill}p
    {decode}d`` is pd-4p2d on 4 nodes, pd-8p4d on 8). Fitting an already
    fitted family for the same cluster returns an equal family.
    """
    if gpus_per_node <= 0 or total_gpus % gpus_per_node:
        return None
    if family.nodes and total_gpus // gpus_per_node not in family.nodes:
        return None
    roles = _scaled_roles(family, total_gpus, gpus_per_node)
    if roles is None:
        return None
    id_template = family.id_template or family.id
    title_template = family.title_template or family.title
    base_id = reference_id(family)
    scaled = family.model_copy(update={"roles": roles})
    return scaled.model_copy(
        update={
            "id": render_template(id_template, scaled),
            "title": render_template(title_template, scaled),
            "base_id": base_id,
            "id_template": id_template,
            "title_template": title_template,
            "scale": family.scale * total_gpus / family.total_gpus,
        }
    )


def scale_grid(grid: GridSpec, scale: float, lane_cap: int) -> GridSpec:
    """A concurrency grid written for the profile's cluster size, resized.

    ``lane_cap`` is the profile-scale ceiling; both it and the grid scale
    with the cluster (8 nodes double a 4-node profile's start and max).
    """
    cap = max(grid.min, int(min(grid.max, lane_cap) * scale))
    start = max(grid.min, min(int(grid.start * scale), cap))
    return GridSpec(start=start, max=cap, min=grid.min)


def fit_deployment(
    deployment: DeploymentProfile, total_gpus: int, gpus_per_node: int
) -> tuple[DeploymentProfile, tuple[str, ...]]:
    """The profile with only the families that fit, fitted; plus the dropped ids."""
    fitted: list[FamilySpec] = []
    dropped: list[str] = []
    for family in deployment.families:
        result = fit_family(family, total_gpus, gpus_per_node)
        if result is None:
            dropped.append(reference_id(family))
        else:
            fitted.append(result)
    ids = [f.id for f in fitted]
    if len(ids) != len(set(ids)):
        raise ProfileError(
            f"{deployment.model}/{deployment.gpu}: fitted family ids collide {ids}"
        )
    return deployment.model_copy(update={"families": tuple(fitted)}), tuple(dropped)


# --------------------------------------------------------------------------
# Generic (untuned) profiles for --model-repo
# --------------------------------------------------------------------------


def generic_model_profile(
    repo: str, revision: str, size_gb: float, served_name: str | None = None
) -> ModelProfile:
    return ModelProfile(
        name="generic",
        served_model_name=served_name or repo,
        checkpoints={
            "generic": Checkpoint(repo=repo, revision=revision, approx_size_gb=size_gb)
        },
    )


def generic_deployment_profile(
    gpu: GpuProfile,
    model: ModelProfile,
    *,
    nodes: int,
    gpus_per_node: int,
    served_context: int,
    pd_min_nodes: int = 2,
) -> DeploymentProfile:
    """Aggregated min-TP replicas plus a half/half PD split; no tuned flags."""
    size_gb = next(iter(model.checkpoints.values())).approx_size_gb
    tp = 1
    while tp < gpus_per_node and size_gb * 1.3 > gpu.hbm_gb * tp:
        tp *= 2
    if size_gb * 1.3 > gpu.hbm_gb * tp:
        raise ProfileError(f"{size_gb} GB does not fit one {gpus_per_node}-GPU node")
    total = nodes * gpus_per_node
    base_cfg: dict[str, Any] = {
        "tensor-parallel-size": tp,
        "context-length": served_context,
        "mem-fraction-static": 0.85,
        "enable-metrics": True,
        "kv-events-config": {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:5557",
        },
    }
    families = [
        FamilySpec(
            id=f"agg-tp{tp}",
            title=f"聚合 TP{tp} x{total // tp}（通用）",
            priority=0,
            roles=(
                RoleSpec(
                    role="agg",
                    gpus_per_instance=tp,
                    instances=total // tp,
                    sglang=base_cfg,
                ),
            ),
        )
    ]
    if nodes >= pd_min_nodes and total // tp >= 2 and (total // 2) % tp == 0:
        half = total // 2 // tp
        families.append(
            FamilySpec(
                id=f"pd-{half}p{half}d",
                title=f"PD 分离 {half}P + {half}D（通用）",
                priority=10,
                transport="mooncake",
                cross_node=("kv",),
                roles=(
                    RoleSpec(
                        role="prefill",
                        gpus_per_instance=tp,
                        instances=half,
                        sglang={
                            **base_cfg,
                            "disaggregation-mode": "prefill",
                            "disaggregation-transfer-backend": "mooncake",
                        },
                    ),
                    RoleSpec(
                        role="decode",
                        gpus_per_instance=tp,
                        instances=half,
                        sglang={
                            **{
                                k: v
                                for k, v in base_cfg.items()
                                if k != "kv-events-config"
                            },
                            "disaggregation-mode": "decode",
                            "disaggregation-transfer-backend": "mooncake",
                        },
                    ),
                ),
            )
        )
    return DeploymentProfile(
        gpu=gpu.name,
        model=model.name,
        served_context=served_context,
        families=tuple(families),
        # lane_cap is written for 4 nodes; the generic profile is built for
        # the actual cluster, so its grid is resized here (fit scale is 1).
        default_grid=scale_grid(
            GridSpec(start=32, max=gpu.lane_cap, min=4), nodes / 4, gpu.lane_cap
        ),
    )
