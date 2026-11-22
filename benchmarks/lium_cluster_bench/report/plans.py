# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read access to rendered launch plans and a compact per-role view of them.

A plan lives at ``<exp_dir>/deployments/<cand_fp>/plan.json`` (a
``LaunchPlan`` JSON). The report and the KB only need the effective SGLang
config, env and argv per role, so ``role_views`` folds the container list
into one representative per role. Env values whose name looks like a
credential, and argv values that follow a credential-looking flag, are
masked defensively even though the harness never puts secrets there.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from benchmarks.lium_cluster_bench.domain.schema import ContainerSpec, LaunchPlan

PlanLookup = Callable[[str], "LaunchPlan | None"]

MASK = "***"
ROLE_ORDER = ("agg", "prefill", "decode", "frontend", "control")
_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL|PRIVATE[_-]?KEY)",
    re.IGNORECASE,
)
_CAND_FP = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def plan_path(exp_dir: Path, cand_fp: str) -> Path:
    """Where the plan of ``cand_fp`` lives inside an experiment directory."""
    return exp_dir / "deployments" / cand_fp / "plan.json"


def plan_lookup_from_dir(exp_dir: Path) -> PlanLookup:
    """A lookup reading ``<exp_dir>/deployments/<cand_fp>/plan.json``.

    A missing file yields ``None``; a present but invalid plan raises, since
    a corrupt plan must not silently disappear from the evidence.
    """

    def lookup(cand_fp: str) -> LaunchPlan | None:
        if not _CAND_FP.match(cand_fp):
            raise ValueError(f"invalid candidate fingerprint {cand_fp!r}")
        path = plan_path(exp_dir, cand_fp)
        if not path.is_file():
            return None
        return LaunchPlan.model_validate_json(path.read_text(encoding="utf-8"))

    return lookup


def mask_env(env: Mapping[str, str]) -> dict[str, str]:
    """Sorted copy of ``env`` with credential-looking values masked."""
    return {
        key: (MASK if _SECRET_NAME.search(key) else value)
        for key, value in sorted(env.items())
    }


def mask_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Mask the value of ``--flag value`` / ``--flag=value`` credential flags."""
    out: list[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            out.append(MASK)
            hide_next = False
            continue
        if arg.startswith("-") and _SECRET_NAME.search(arg):
            if "=" in arg:
                out.append(arg.split("=", 1)[0] + "=" + MASK)
            else:
                out.append(arg)
                hide_next = True
            continue
        out.append(arg)
    return tuple(out)


@dataclass(frozen=True)
class RoleView:
    """One role of a plan: rendered config and representative env/argv."""

    role: str
    instances: int
    containers: int
    nodes: tuple[str, ...]
    sglang: dict[str, Any]
    config_variants: int
    representative: str  # container name the env/command come from
    env: dict[str, str]
    command: tuple[str, ...]


def _role_rank(role: str) -> tuple[int, str]:
    return (ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER), role)


def sorted_containers(plan: LaunchPlan) -> list[ContainerSpec]:
    """Containers in role, instance, rank order."""
    return sorted(
        plan.containers,
        key=lambda c: (_role_rank(c.role), c.instance, c.rank, c.node, c.name),
    )


def role_views(plan: LaunchPlan) -> list[RoleView]:
    """One representative (instance 0, rank 0) per role, in role order."""
    by_role: dict[str, list[ContainerSpec]] = {}
    for container in sorted_containers(plan):
        by_role.setdefault(container.role, []).append(container)
    views = []
    for role, containers in by_role.items():
        configs = {
            key: value
            for key, value in sorted(plan.rendered_configs.items())
            if key.split(":", 1)[0] == role
        }
        first = configs.get(f"{role}:0")
        if first is None and configs:
            first = next(iter(configs.values()))
        variants = {
            json.dumps(v, sort_keys=True, default=str) for v in configs.values()
        }
        rep = containers[0]
        views.append(
            RoleView(
                role=role,
                instances=len({c.instance for c in containers}),
                containers=len(containers),
                nodes=tuple(sorted({c.node for c in containers})),
                sglang=dict(first or {}),
                config_variants=len(variants),
                representative=rep.name,
                env=mask_env(rep.env),
                command=mask_argv(rep.command),
            )
        )
    return views
