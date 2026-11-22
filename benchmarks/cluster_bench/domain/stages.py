# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage modes: the whole serving pipeline, or one PD stage on every GPU.

* ``e2e`` (default): the profile's agg and PD families serve requests end to
  end; the objective is output tok/s/GPU.
* P-Only (``prefill``): every GPU runs the *prefill* role of a PD family, but
  as an aggregated worker (no ``disaggregation-*`` keys, decode CUDA graphs
  off as in prefill mode) and AIPerf caps every request at one output token.
  A request is then one prefill plus the sampling of its first token, and the
  objective is prompt (input) tok/s/GPU on the chosen dataset, prefix-cache
  hits included (the cache hit rate is reported next to it).
* D-Only (``decode``): every GPU runs the *decode* role with SGLang's
  ``fake`` KV-transfer backend. The worker shim agents/sglang_decode_only.py
  hands each request SGLang's fake bootstrap address (Dynamo's own decode
  health check does the same) and registers the worker without a prefill
  peer, so the frontend routes straight to decode. The engine allocates the
  full prompt's KV, skips the transfer and decodes; the objective is output
  tok/s/GPU. The KV contents are not real, so speculative decoding runs with
  simulated acceptance and the accuracy gate does not apply.

Stage families are derived from the profile's PD families (one per distinct
role config) unless the profile defines families with ``stage:`` itself.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from benchmarks.cluster_bench.domain.base import stable_hash
from benchmarks.cluster_bench.domain.profiles import (
    ProfileError,
    reference_id,
    render_template,
)
from benchmarks.cluster_bench.domain.schema import (
    DeploymentProfile,
    FamilySpec,
    GridSpec,
    RoleSpec,
    Stage,
)

STAGES: tuple[Stage, ...] = ("e2e", "prefill", "decode")
# Family-id prefix and display label per stage.
STAGE_PREFIX: dict[Stage, str] = {"e2e": "", "prefill": "p-only", "decode": "d-only"}
STAGE_LABEL: dict[Stage, str] = {"e2e": "E2E", "prefill": "P-Only", "decode": "D-Only"}
_ALIASES: dict[str, Stage] = {
    "e2e": "e2e",
    "pd": "e2e",
    "full": "e2e",
    "p-only": "prefill",
    "p_only": "prefill",
    "ponly": "prefill",
    "prefill": "prefill",
    "p": "prefill",
    "d-only": "decode",
    "d_only": "decode",
    "donly": "decode",
    "decode": "decode",
    "d": "decode",
}
# scripts/launch_node.sh starts agents/sglang_decode_only.py instead of
# ``python3 -m dynamo.sglang`` when this is "1".
DECODE_ONLY_ENV = "CB_DECODE_ONLY"
FAKE_TRANSFER_BACKEND = "fake"
# P-Only: AIPerf --extra-inputs value that caps every request at one token
# (it overrides the per-turn max_completion_tokens AIPerf sets from the trace).
PREFILL_ONLY_EXTRA_INPUT = "max_completion_tokens:1"
# A P-Only point whose mean OSL exceeds this did not honour the cap.
PREFILL_ONLY_MAX_OSL = 1.5


def parse_stage(text: str) -> Stage:
    """``e2e`` | ``p-only`` | ``d-only`` (also prefill/decode, p/d, pd)."""
    stage = _ALIASES.get(text.strip().lower())
    if stage is None:
        raise ValueError(f"unknown stage {text!r}; use e2e, p-only or d-only")
    return stage


def stage_label(stage: Stage) -> str:
    return STAGE_LABEL[stage]


def objective_tokens(stage: Stage) -> str:
    """Which tokens the objective counts: "input" for P-Only, else "output"."""
    return "input" if stage == "prefill" else "output"


def stage_role(stage: Stage) -> str | None:
    """The PD role a stage runs on every GPU (None for e2e)."""
    return None if stage == "e2e" else stage


def apply_stage(
    stage: Stage, role: str, cfg: Mapping[str, Any], env: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """SGLang config and env of one worker of a ``stage`` family.

    Idempotent, and applied after the recipe, the family overrides and the
    ladder rungs, so no rung can bring disaggregation back to P-Only.
    """
    out_cfg = dict(cfg)
    out_env = dict(env)
    if stage == "e2e":
        return out_cfg, out_env
    if role != stage:
        raise ProfileError(f"a {stage_label(stage)} family cannot run role {role!r}")
    if stage == "prefill":
        for key in [k for k in out_cfg if k.startswith("disaggregation-")]:
            del out_cfg[key]
        # Prefill-mode SGLang never captures decode graphs; neither do we.
        out_cfg["cuda-graph-backend-decode"] = "disabled"
        out_env.pop(DECODE_ONLY_ENV, None)
    else:
        out_cfg["disaggregation-mode"] = "decode"
        out_cfg["disaggregation-transfer-backend"] = FAKE_TRANSFER_BACKEND
        out_cfg.pop("disaggregation-ib-device", None)
        out_env[DECODE_ONLY_ENV] = "1"
    return out_cfg, out_env


def stage_flags(stage: Stage) -> set[str]:
    """CLI flags a stage adds to every worker (checked against --help)."""
    if stage == "prefill":
        return {"--cuda-graph-backend-decode"}
    if stage == "decode":
        return {"--disaggregation-mode", "--disaggregation-transfer-backend"}
    return set()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parallel_tag(cfg: Mapping[str, Any], gpus: int) -> str:
    """Short parallelism label of an SGLang config: dep4, dtp8, tp8ep8, tp4."""
    tp = _int(cfg.get("tensor-parallel-size", cfg.get("tp-size")), gpus)
    ep = _int(cfg.get("ep-size", cfg.get("expert-parallel-size")), 1)
    dp_attention = bool(cfg.get("enable-dp-attention"))
    if dp_attention and ep > 1:
        return f"dep{tp}"
    if dp_attention:
        return f"dtp{tp}"
    if ep > 1:
        return f"tp{tp}ep{ep}"
    return f"tp{tp}"


def _role_signature(role: RoleSpec) -> str:
    return stable_hash(role.model_dump(mode="json", exclude={"instances"}))


def _stage_grid(source: FamilySpec, stage: Stage, default: GridSpec) -> GridSpec:
    grid = source.grid or default
    if stage != "decode":
        return grid
    # Every GPU decodes: lanes scale with the decode share the grid was sized for.
    decode_gpus = sum(r.total_gpus for r in source.roles if r.role == "decode")
    ratio = source.total_gpus / decode_gpus if decode_gpus else 1.0
    return GridSpec(
        start=max(grid.min, int(grid.start * ratio)),
        max=max(grid.min, int(grid.max * ratio)),
        min=grid.min,
    )


def pd_split(
    isl: float, osl: float, prefill_tps: float, decode_tps: float
) -> tuple[float, float]:
    """Ideal prefill:decode GPU ratio and the PD output tok/s/GPU it implies.

    ``prefill_tps`` is P-Only input tok/s/GPU and ``decode_tps`` D-Only output
    tok/s/GPU, both at saturation; ``isl``/``osl`` are the workload's mean
    prompt and output lengths. A request costs isl/P prefill GPU-seconds and
    osl/D decode GPU-seconds, so GPUs_P / GPUs_D = (isl * D) / (osl * P), and
    that split delivers osl / (isl/P + osl/D) output tok/s/GPU. KV transfer
    cost and latency SLOs are not included: it is an upper bound.
    """
    if min(isl, osl, prefill_tps, decode_tps) <= 0:
        raise ValueError("pd_split needs positive lengths and throughputs")
    prefill_s, decode_s = isl / prefill_tps, osl / decode_tps
    return prefill_s / decode_s, osl / (prefill_s + decode_s)


def stage_lane_cap(family: FamilySpec, lane_cap: int, default: GridSpec) -> int:
    """The GPU profile's lane ceiling for ``family`` (both at profile scale).

    A D-Only family's grid was widened by total/decode GPUs; the ceiling
    widens with it, so each decode GPU can see the load it carries in PD.
    """
    if family.stage != "decode":
        return lane_cap
    return max(lane_cap, (family.grid or default).max)


def derive_stage_families(
    deployment: DeploymentProfile,
    stage: Stage,
    config_for: Callable[[RoleSpec], Mapping[str, Any]] | None = None,
) -> tuple[FamilySpec, ...]:
    """One stage family per distinct ``stage`` role config of the PD families.

    Families are written for the profile's cluster size; the stage family
    fills that size with the role (``fit_family`` scales it later).
    ``config_for`` returns a role's merged SGLang config (for the id's
    parallelism tag); without it the tag is the GPU count.
    """
    role_name = stage_role(stage)
    if role_name is None:
        return ()
    families: list[FamilySpec] = []
    seen: set[str] = set()
    used_ids: set[str] = set()
    sources = sorted(
        (f for f in deployment.families if f.stage == "e2e"),
        key=lambda f: (f.priority, reference_id(f)),
    )
    for source in sources:
        kinds = {r.role for r in source.roles}
        if not {"prefill", "decode"} <= kinds:
            continue
        for role in source.roles:
            if role.role != role_name:
                continue
            signature = _role_signature(role)
            if signature in seen:
                continue
            total = source.total_gpus
            if total % role.gpus_per_instance:
                continue
            seen.add(signature)
            cfg = config_for(role) if config_for else {}
            tag = (
                parallel_tag(cfg, role.gpus_per_instance)
                if cfg
                else f"{role.gpus_per_instance}g"
            )
            prefix = STAGE_PREFIX[stage]
            base = f"{prefix}-{tag}"
            suffix = ""
            n = 2
            while f"{base}{suffix}" in used_ids:
                suffix = f"-{n}"
                n += 1
            used_ids.add(f"{base}{suffix}")
            id_template = f"{base}{suffix}x{{{role_name}}}"
            source_id = reference_id(source)
            title_template = (
                f"{stage_label(stage)}: {{{role_name}}} x {tag} {role_name} "
                f"(from {source_id})"
            )
            draft = FamilySpec(
                id=id_template,
                title=title_template,
                roles=(
                    role.model_copy(
                        update={"instances": total // role.gpus_per_instance}
                    ),
                ),
                transport="none",
                cross_node=("collective",) if role.nodes_per_instance > 1 else (),
                group=None,
                priority=source.priority,
                enabled=source.enabled,
                grid=_stage_grid(source, stage, deployment.default_grid),
                notes=f"{stage_label(stage)} of {source_id}'s {role_name} role",
                nodes=source.nodes,
                stage=stage,
                source=source_id,
            )
            families.append(
                draft.model_copy(
                    update={
                        "id": render_template(id_template, draft),
                        "title": render_template(title_template, draft),
                        "id_template": id_template,
                        "title_template": title_template,
                    }
                )
            )
    return tuple(families)


def _check_stage_family(family: FamilySpec) -> None:
    if family.stage == "e2e":
        return
    roles = {r.role for r in family.roles}
    if roles != {family.stage}:
        raise ProfileError(
            f"{family.id}: a {stage_label(family.stage)} family may only have "
            f"{family.stage} roles, got {sorted(roles)}"
        )


def stage_deployment(
    deployment: DeploymentProfile,
    stage: Stage,
    config_for: Callable[[RoleSpec], Mapping[str, Any]] | None = None,
) -> DeploymentProfile:
    """The profile restricted to ``stage``: its own families of that stage,
    plus (P-Only/D-Only) families derived from the PD families."""
    for family in deployment.families:
        _check_stage_family(family)
    own = tuple(f for f in deployment.families if f.stage == stage)
    if stage == "e2e":
        return deployment.model_copy(update={"families": own})
    own_signatures = {_role_signature(r) for f in own for r in f.roles}
    derived = tuple(
        f
        for f in derive_stage_families(deployment, stage, config_for)
        if _role_signature(f.roles[0]) not in own_signatures
        and f.id not in {o.id for o in own}
    )
    return deployment.model_copy(update={"families": own + derived})
