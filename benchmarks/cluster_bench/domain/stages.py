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
* A-Only (``attention``): D-Only's decode worker as the attention side of
  attention-FFN disaggregation: DP attention with TP1 per rank, ep 1, and no
  routed experts on the target model; each MoE layer waits for a simulated F
  instead (domain/afd.py, agents/sglang_afd.py). The objective is output
  tok/s per A GPU; the leader is re-measured with the wait off (null-F bound).

Stage families are derived from the profile's PD families (one per distinct
role config) unless the profile defines families with ``stage:`` itself.
Everything that differs between stages is one row of ``STAGES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from benchmarks.cluster_bench.domain import afd
from benchmarks.cluster_bench.domain.base import stable_hash
from benchmarks.cluster_bench.domain.modelarch import ModelArch
from benchmarks.cluster_bench.domain.profiles import (
    ProfileError,
    reference_id,
    render_template,
)
from benchmarks.cluster_bench.domain.schema import (
    AfdPlan,
    DeploymentProfile,
    FamilySpec,
    GridSpec,
    RoleSpec,
    Stage,
)

# scripts/launch_node.sh starts agents/sglang_decode_only.py instead of
# ``python3 -m dynamo.sglang`` when this is "1".
DECODE_ONLY_ENV = "CB_DECODE_ONLY"
FAKE_TRANSFER_BACKEND = "fake"
# SGLang source hotfix ids a worker applies before it starts
# (agents/sglang_hotfix.py ENV_VAR; a test keeps the two equal).
HOTFIX_ENV = "CB_SGLANG_HOTFIXES"
# P-Only: AIPerf --extra-inputs value that caps every request at one token
# (it overrides the per-turn max_completion_tokens AIPerf sets from the trace).
PREFILL_ONLY_EXTRA_INPUT = "max_completion_tokens:1"
# A P-Only point whose mean OSL exceeds this did not honour the cap.
PREFILL_ONLY_MAX_OSL = 1.5


@dataclass(frozen=True)
class WorkerContext:
    """What a stage's worker transform may use besides the SGLang config."""

    gpus: int = 0  # GPUs of one instance of the stage's role
    afd: AfdPlan | None = None  # A-Only: the run's F model (None: config only)
    null_f: bool = False  # A-Only: the null-F bound (the F wait is off)


# Mutates one worker's SGLang config and env in place (apply_stage copies first).
WorkerTransform = Callable[[dict[str, Any], dict[str, str], WorkerContext], None]


def _unchanged(cfg: dict[str, Any], env: dict[str, str], ctx: WorkerContext) -> None:
    return None


def _prefill_worker(
    cfg: dict[str, Any], env: dict[str, str], ctx: WorkerContext
) -> None:
    for key in [k for k in cfg if k.startswith("disaggregation-")]:
        del cfg[key]
    # Prefill-mode SGLang never captures decode graphs; neither do we.
    cfg["cuda-graph-backend-decode"] = "disabled"
    env.pop(DECODE_ONLY_ENV, None)


def _decode_worker(
    cfg: dict[str, Any], env: dict[str, str], ctx: WorkerContext
) -> None:
    cfg["disaggregation-mode"] = "decode"
    cfg["disaggregation-transfer-backend"] = FAKE_TRANSFER_BACKEND
    cfg.pop("disaggregation-ib-device", None)
    env[DECODE_ONLY_ENV] = "1"


def _attention_worker(
    cfg: dict[str, Any], env: dict[str, str], ctx: WorkerContext
) -> None:
    if ctx.gpus < 1:
        raise ValueError("an A-Only worker needs its instance's GPU count")
    _decode_worker(cfg, env, ctx)
    afd.attention_config(cfg, ctx.gpus)
    afd.attention_env(env, ctx.afd, gpus=ctx.gpus, null_f=ctx.null_f)


@dataclass(frozen=True)
class StageSpec:
    """Everything that differs between stages, in one row of ``STAGES``."""

    id: Stage  # the persisted value (RunConfig.stage, LaunchPlan.stage, the KB)
    label: str  # E2E | P-Only | D-Only | A-Only
    cli: str  # canonical --stage value; also the family and experiment id prefix
    aliases: tuple[str, ...]  # --stage spellings (lower case)
    # The PD role every GPU runs; None = the profile's own (agg and PD) families.
    role: str | None = None
    objective: Literal["input", "output"] = "output"  # tokens the objective counts
    one_token: bool = False  # AIPerf caps every request at one output token
    # Decode over SGLang's fake KV transfer (the D-Only worker shim): no prefill
    # peer, meaningless tokens (no smoke-text check), simulated MTP acceptance.
    fake_kv: bool = False
    disaggregated: bool = True  # PD roles keep a disaggregation mode (bootstrap port)
    speculative: bool = True  # MTP and its acceptance length apply
    floors: bool = True  # a per-user decode rate exists (interactivity floors)
    accuracy: bool = False  # the GSM8K accuracy gate applies
    # Derived grids and lane caps scale by the family's total / role GPUs, and
    # the grid's max by grid_max_scale on top (more KV room per GPU).
    widen_grid: bool = False
    grid_max_scale: float = 1.0
    needs_moe: bool = False  # preflight refuses a model without routed experts
    # Side of attention-FFN disaggregation the stage runs ("attention"): the
    # run resolves an F model, measures the null-F bound and reports traffic.
    afd_side: str | None = None
    worker: WorkerTransform = _unchanged
    flags: frozenset[str] = frozenset()  # CLI flags the stage adds (vs --help)
    hotfixes: tuple[str, ...] = ()  # SGLang hotfix ids merged into HOTFIX_ENV
    # MFU: GEMM precision of the peak (None: the checkpoint's) and the model
    # the GPUs compute (None: the whole model), with a note on the result.
    mfu_precision: str | None = None
    mfu_arch: Callable[[ModelArch], ModelArch] | None = None
    mfu_note: str = ""
    conditions: tuple[str, ...] = ()  # summary conditions every result carries
    # What the result rests on that no GPU run has checked yet (summary).
    unverified: tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        """Family and experiment id prefix ("" for e2e)."""
        return "" if self.role is None else self.cli


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        id="e2e", label="E2E", cli="e2e", aliases=("e2e", "pd", "full"), accuracy=True
    ),
    StageSpec(
        id="prefill",
        label="P-Only",
        cli="p-only",
        aliases=("p-only", "p_only", "ponly", "prefill", "p"),
        role="prefill",
        objective="input",
        one_token=True,
        disaggregated=False,
        speculative=False,
        floors=False,
        worker=_prefill_worker,
        flags=frozenset({"--cuda-graph-backend-decode"}),
        unverified=(
            "P-Only: the PD prefill role config run as an aggregated worker matches "
            "a real prefill worker's cost (no KV send, no bootstrap queue)",
            "P-Only: AIPerf max_completion_tokens:1 caps every AgentX turn at one "
            "output token (audited per point: OSL <= 1.5)",
        ),
    ),
    StageSpec(
        id="decode",
        label="D-Only",
        cli="d-only",
        aliases=("d-only", "d_only", "donly", "decode", "d"),
        role="decode",
        fake_kv=True,
        widen_grid=True,
        worker=_decode_worker,
        flags=frozenset({"--disaggregation-mode", "--disaggregation-transfer-backend"}),
        unverified=(
            "D-Only: dynamo.sglang decode with SGLang's fake KV transfer and no "
            "prefill peer (source-checked against Dynamo 1.6.2 / SGLang 0.5.21 and "
            "0.5.23; preflight runs the shim's --check)",
            "D-Only: MoE expert load over meaningless KV may differ from real "
            "requests (EP balance), and MTP acceptance is simulated",
        ),
    ),
    StageSpec(
        id="attention",
        label="A-Only",
        cli="a-only",
        aliases=("a-only", "a_only", "aonly", "attention", "a"),
        role="decode",
        fake_kv=True,
        widen_grid=True,
        # No routed experts on A: an A GPU holds ~10x D-Only's KV.
        grid_max_scale=2.0,
        needs_moe=True,
        afd_side="attention",
        worker=_attention_worker,
        flags=frozenset({"--disaggregation-mode", "--disaggregation-transfer-backend"})
        | afd.ATTENTION_FLAGS,
        hotfixes=(afd.HOTFIX_ID,),
        mfu_precision="bf16",
        mfu_arch=afd.a_side_arch,
        mfu_note=afd.MFU_NOTE,
        conditions=(
            "A-Only: F is a simulated GPU wait at every MoE layer (domain/afd.py); "
            "tok/s/GPU and cost count the A GPUs only",
        ),
        unverified=(
            "A-Only: SGLang hotfix afd-attn-only (target routed experts null and "
            "not read from disk, the F wait kernel) source-checked against SGLang "
            "0.5.21 only; preflight runs its check and the wait kernel's GPU "
            "self-test, deploy checks the worker's log marker and A-side weights",
            "A-Only: the F wait is a model (F roofline at eff_mem/eff_flop, "
            "measured link bandwidth, default or perftest-based latency, "
            "microbatch overlap credit, uniform routing, one F per A instance); "
            "no real F and no real A<->F transfer",
            "A-Only: DP attention (TP1 per rank) over fake KV with GLM-5.3's DSA "
            "and GLM-5.3-Flash's KDA + DSA (Flash is not on SGLang AFD's "
            "supported list) has not run on these GPUs",
            "A-Only: the wait kernel (Triton, %globaltimer) under CUDA graph "
            "replay is self-tested on GPU 0 only; eager batches above the graph "
            "sizes run the shared expert after the wait (no overlap)",
            "A-Only: MTP acceptance is simulated (fake KV), as in D-Only",
        ),
    ),
)
STAGES: dict[Stage, StageSpec] = {spec.id: spec for spec in STAGE_SPECS}
# Family-id prefix per stage.
STAGE_PREFIX: dict[Stage, str] = {spec.id: spec.prefix for spec in STAGE_SPECS}
_ALIASES: dict[str, Stage] = {
    alias: spec.id for spec in STAGE_SPECS for alias in spec.aliases
}


def parse_stage(text: str) -> Stage:
    """``e2e`` | ``p-only`` | ``d-only`` | ``a-only`` (also prefill, decode,
    attention, p/d/a, pd)."""
    stage = _ALIASES.get(text.strip().lower())
    if stage is None:
        names = [spec.cli for spec in STAGE_SPECS]
        listed = f"{', '.join(names[:-1])} or {names[-1]}"
        raise ValueError(f"unknown stage {text!r}; use {listed}")
    return stage


def stage_spec(stage: Stage) -> StageSpec:
    return STAGES[stage]


def flops_arch(stage: Stage, arch: ModelArch | None) -> ModelArch | None:
    """The model whose FLOPs the stage's GPUs compute (A-Only: no routed
    experts); the MFU and the report's FLOPs per token both use it."""
    transform = STAGES[stage].mfu_arch
    return transform(arch) if arch is not None and transform is not None else arch


def stage_label(stage: str) -> str:
    """Display label; a stage this harness does not know shows as itself."""
    spec = STAGES.get(stage)  # type: ignore[call-overload]
    return spec.label if spec is not None else str(stage)


def objective_tokens(stage: Stage) -> str:
    """Which tokens the objective counts: "input" for P-Only, else "output"."""
    return STAGES[stage].objective


def stage_role(stage: Stage) -> str | None:
    """The PD role a stage runs on every GPU (None for e2e)."""
    return STAGES[stage].role


def merge_csv(existing: str, ids: tuple[str, ...]) -> str:
    """``existing`` (comma-separated) plus ``ids``, in order, without repeats."""
    items = [i for i in existing.replace(" ", "").split(",") if i]
    return ",".join(dict.fromkeys([*items, *ids]))


def apply_stage(
    stage: Stage,
    role: str,
    cfg: Mapping[str, Any],
    env: Mapping[str, str],
    worker: WorkerContext | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """SGLang config and env of one worker of a ``stage`` family.

    Idempotent, and applied after the recipe, the family overrides and the
    ladder rungs, so no rung can bring disaggregation back to P-Only.
    """
    spec = STAGES[stage]
    out_cfg = dict(cfg)
    out_env = dict(env)
    if spec.role is None:
        return out_cfg, out_env
    if role != spec.role:
        raise ProfileError(f"a {spec.label} family cannot run role {role!r}")
    spec.worker(out_cfg, out_env, worker or WorkerContext())
    if spec.hotfixes:
        out_env[HOTFIX_ENV] = merge_csv(out_env.get(HOTFIX_ENV, ""), spec.hotfixes)
    return out_cfg, out_env


def stage_flags(stage: Stage) -> set[str]:
    """CLI flags a stage adds to every worker (checked against --help)."""
    return set(STAGES[stage].flags)


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
    spec = STAGES[stage]
    if not spec.widen_grid:
        return grid
    # Every GPU runs the role: lanes scale with the share the grid was sized for.
    role_gpus = sum(r.total_gpus for r in source.roles if r.role == spec.role)
    ratio = source.total_gpus / role_gpus if role_gpus else 1.0
    return GridSpec(
        start=max(grid.min, int(grid.start * ratio)),
        max=max(grid.min, int(grid.max * ratio * spec.grid_max_scale)),
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

    A D-Only or A-Only family's grid was widened by total/decode GPUs; the
    ceiling widens with it, so each GPU can see the load it carries in PD.
    """
    if not STAGES[family.stage].widen_grid:
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
    spec = STAGES[stage]
    role_name = spec.role
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
            if cfg:
                # The tag names the parallelism the stage's worker runs.
                staged, _env = apply_stage(
                    stage,
                    role_name,
                    cfg,
                    {},
                    WorkerContext(gpus=role.gpus_per_instance),
                )
                tag = parallel_tag(staged, role.gpus_per_instance)
            else:
                tag = f"{role.gpus_per_instance}g"
            base = f"{spec.prefix}-{tag}"
            suffix = ""
            n = 2
            while f"{base}{suffix}" in used_ids:
                suffix = f"-{n}"
                n += 1
            used_ids.add(f"{base}{suffix}")
            id_template = f"{base}{suffix}x{{{role_name}}}"
            source_id = reference_id(source)
            title_template = (
                f"{spec.label}: {{{role_name}}} x {tag} {role_name} "
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
                notes=f"{spec.label} of {source_id}'s {role_name} role",
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
    spec = STAGES[family.stage]
    if spec.role is None:
        return
    roles = {r.role for r in family.roles}
    if roles != {spec.role}:
        raise ProfileError(
            f"{family.id}: a {spec.label} family may only have "
            f"{spec.role} roles, got {sorted(roles)}"
        )


def stage_deployment(
    deployment: DeploymentProfile,
    stage: Stage,
    config_for: Callable[[RoleSpec], Mapping[str, Any]] | None = None,
) -> DeploymentProfile:
    """The profile restricted to ``stage``: its own families of that stage,
    plus (stages with a role) families derived from the PD families. A profile
    family replaces a derived one with the same role config or the same id
    (compared at profile scale, so ``d-only-dtp8x{decode}`` counts)."""
    for family in deployment.families:
        _check_stage_family(family)
    own = tuple(f for f in deployment.families if f.stage == stage)
    if STAGES[stage].role is None:
        return deployment.model_copy(update={"families": own})
    own_signatures = {_role_signature(r) for f in own for r in f.roles}
    own_ids = {reference_id(f) for f in own}
    derived = tuple(
        f
        for f in derive_stage_families(deployment, stage, config_for)
        if _role_signature(f.roles[0]) not in own_signatures
        and reference_id(f) not in own_ids
    )
    return deployment.model_copy(update={"families": own + derived})
