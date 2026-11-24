#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named edits of the image's SGLang sources, applied inside a worker container.

A GPU/model pair can need a small SGLang change that the pinned image does not
have yet. A profile names the edits in a role's ``env`` as
``CB_SGLANG_HOTFIXES`` (comma-separated ids). scripts/launch_node.sh applies
them in the container before the worker starts, so every process the worker
spawns imports the edited sources, and nothing outlives the container. Each
edit replaces one exact text that must occur once in the installed file (or
finds its result already there); anything else fails the launch instead of
serving an unpatched engine. Preflight runs ``check`` in the image first.

    python3 -P sglang_hotfix.py apply ID[,ID...]
    python3 -P sglang_hotfix.py check ID[,ID...]

Hotfixes:

``sm120-tilelang-dsa-v1-one-stage``
    SGLang 0.5.21 serves DSA models whose MLA has no RoPE tail and whose
    indexer pools keys (GLM-5.3-Flash: ``index_kpool`` 4) only with the FA3,
    TRTLLM or TileLang DSA kernels. On SM12x FA3 (SM90 only) and TRTLLM
    (trtllm-gen, SM100 only) do not run, and the TileLang sparse-MLA v1 kernel
    double-buffers 64-row BF16 KV tiles: 148 KiB of shared memory at d_v=512,
    over the 99 KiB one block may use. On SM12x the edit runs that kernel with
    one pipeline stage (about 82 KiB). On an RTX PRO 6000D it matches a
    float32 torch reference within 0.0011 (8 and 16 heads, topk 2048), and
    GLM-5.3-Flash answers correctly with it at TP4 and TP8. Upstream #50752
    fixes the same overflow only for the v2 kernel's 512+64 geometry (GLM-5.3,
    DeepSeek). Other GPUs and attention geometries are unchanged.

``afd-attn-only``
    The A-Only stage (attention side of attention-FFN disaggregation, F
    simulated). Copies agents/sglang_afd.py and its kernel into the package
    as ``srt/cb_afd.py`` and ``srt/cb_afd_kernel.py``; ``DeepseekV2MoE``
    builds that module's null experts in target layers and gives them the
    full top-k (the routing A sends to F, which SM100's flashinfer_trtllm
    runner would leave to its fused kernel), and the safetensors iterators
    skip target layers' routed-expert tensors. Inert unless the worker env
    sets ``CB_AFD_ATTN_ONLY=1`` (domain/afd.py renders it).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ENV_VAR = "CB_SGLANG_HOTFIXES"


AGENTS_DIR = Path(__file__).resolve().parent


class HotfixError(RuntimeError):
    pass


@dataclass(frozen=True)
class Edit:
    path: str  # relative to the installed sglang package directory
    old: str
    new: str


@dataclass(frozen=True)
class Copy:
    """Install a file of this directory into the sglang package."""

    src: str  # file name in this agents directory
    path: str  # destination, relative to the installed sglang package directory


_TILELANG_OLD = """\
        kernel = kernel_factory(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale)
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
    return out
"""
_TILELANG_NEW = """\
        if tail_dim == 0 and torch.cuda.get_device_capability(q.device)[0] == 12:
            # cluster_bench hotfix sm120-tilelang-dsa-v1-one-stage: one stage
            # keeps the 64-row BF16 KV tile within SM12x's 99 KiB per block.
            kernel = sparse_attention_fwd_kernel_v1(
                num_heads, d_v, tail_dim, topk, sm_scale=sm_scale, num_stages=1
            )
        else:
            kernel = kernel_factory(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale)
        out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # type: ignore
    return out
"""

_DSV2 = "srt/models/deepseek_v2.py"
_DSV2_MOE_OLD = """\
class DeepseekV2MoE(nn.Module):
    def __init__(
"""
_DSV2_MOE_NEW = """\
def _cb_afd_experts_class(quant_config, is_nextn):
    # cluster_bench hotfix afd-attn-only: an A-Only worker's target MoE layers
    # hold no routed experts (benchmarks/cluster_bench/agents/sglang_afd.py).
    from sglang.srt import cb_afd

    if cb_afd.nulls_experts(is_nextn):
        return cb_afd.null_experts_class()
    return get_moe_impl_class(quant_config)


def _cb_afd_standard_topk(is_nextn):
    # cluster_bench hotfix afd-attn-only: A computes the routing it sends to F,
    # so a null layer always takes the full top-k.
    from sglang.srt import cb_afd

    return cb_afd.nulls_experts(is_nextn)


class DeepseekV2MoE(nn.Module):
    def __init__(
"""
_DSV2_EXPERTS_OLD = """\
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=num_experts_for_moe + get_exec().moe.ep_num_redundant_experts,
"""
_DSV2_EXPERTS_NEW = """\
        self.experts = _cb_afd_experts_class(quant_config, is_nextn)(
            num_experts=num_experts_for_moe + get_exec().moe.ep_num_redundant_experts,
"""
_DSV2_TOPK_OLD = """\
                output_format=(
                    TopKOutputFormat.STANDARD
                    if (quant_config is None)
                    and (not get_moe_runner_backend().is_flashinfer_trtllm())
                    else None
                ),
"""
_DSV2_TOPK_NEW = """\
                output_format=(
                    TopKOutputFormat.STANDARD
                    if _cb_afd_standard_topk(is_nextn)
                    or (
                        (quant_config is None)
                        and (not get_moe_runner_backend().is_flashinfer_trtllm())
                    )
                    else None
                ),
"""
_WEIGHTS = "srt/model_loader/weight_utils.py"
_WEIGHTS_SKIP_OLD = "CAPTURE_SAFE_WEIGHT_SENTINEL = 1e-3\n"
_WEIGHTS_SKIP_NEW = """\
CAPTURE_SAFE_WEIGHT_SENTINEL = 1e-3


def _cb_afd_skip_tensor(name):
    # cluster_bench hotfix afd-attn-only: an A-Only worker never reads the
    # target's routed experts (benchmarks/cluster_bench/agents/sglang_afd.py).
    from sglang.srt import cb_afd

    return cb_afd.skip_tensor(name)
"""
_WEIGHTS_ITER_OLD = """\
                for name in f.keys():
                    yield name, f.get_tensor(name)
"""
_WEIGHTS_ITER_NEW = """\
                for name in f.keys():
                    if _cb_afd_skip_tensor(name):
                        continue
                    yield name, f.get_tensor(name)
"""
_WEIGHTS_THREADS_OLD = (
    "                result = {k: f.get_tensor(k) for k in f.keys()}\n"
)
_WEIGHTS_THREADS_NEW = """\
                result = {
                    k: f.get_tensor(k) for k in f.keys() if not _cb_afd_skip_tensor(k)
                }
"""

HOTFIXES: dict[str, tuple[Edit | Copy, ...]] = {
    "sm120-tilelang-dsa-v1-one-stage": (
        Edit(
            "kernels/ops/attention/dsa/tilelang_kernel.py",
            _TILELANG_OLD,
            _TILELANG_NEW,
        ),
    ),
    "afd-attn-only": (
        Copy("sglang_afd.py", "srt/cb_afd.py"),
        Copy("sglang_afd_kernel.py", "srt/cb_afd_kernel.py"),
        Edit(_DSV2, _DSV2_MOE_OLD, _DSV2_MOE_NEW),
        Edit(_DSV2, _DSV2_EXPERTS_OLD, _DSV2_EXPERTS_NEW),
        Edit(_DSV2, _DSV2_TOPK_OLD, _DSV2_TOPK_NEW),
        Edit(_WEIGHTS, _WEIGHTS_SKIP_OLD, _WEIGHTS_SKIP_NEW),
        Edit(_WEIGHTS, _WEIGHTS_ITER_OLD, _WEIGHTS_ITER_NEW),
        Edit(_WEIGHTS, _WEIGHTS_THREADS_OLD, _WEIGHTS_THREADS_NEW),
    ),
}


def parse_ids(text: str) -> tuple[str, ...]:
    """``"a, b,,a"`` -> ``("a", "b")``: comma-separated ids, in order, once each."""
    ids: list[str] = []
    for part in text.split(","):
        hotfix_id = part.strip()
        if hotfix_id and hotfix_id not in ids:
            ids.append(hotfix_id)
    return tuple(ids)


def sglang_dir() -> Path:
    """The installed sglang package directory, found without importing sglang."""
    spec = importlib.util.find_spec("sglang")
    locations = list(spec.submodule_search_locations or []) if spec else []
    if not locations:
        raise HotfixError("sglang is not installed in this Python")
    return Path(locations[0])


def plan(ids: Iterable[str], root: Path) -> dict[Path, str]:
    """The new text of every file the hotfixes change; empty once all are applied.

    Raises HotfixError for an unknown id, an unreadable file, an edit whose
    text does not occur exactly once (the hotfix does not fit this SGLang), or
    a changed Python file that no longer compiles.
    """
    pending: dict[Path, str] = {}
    for hotfix_id in ids:
        edits = HOTFIXES.get(hotfix_id)
        if edits is None:
            known = ", ".join(sorted(HOTFIXES))
            raise HotfixError(f"unknown hotfix {hotfix_id!r} (known: {known})")
        for edit in edits:
            path = root / edit.path
            if isinstance(edit, Copy):
                _plan_copy(hotfix_id, edit, path, pending)
                continue
            text = pending.get(path)
            if text is None:
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise HotfixError(
                        f"{hotfix_id}: cannot read {path}: {exc}"
                    ) from exc
            if edit.new in text:
                continue  # applied already
            count = text.count(edit.old)
            if count != 1:
                raise HotfixError(
                    f"{hotfix_id}: the text to replace occurs {count} times in "
                    f"{path}, not once: the hotfix does not fit this SGLang"
                )
            pending[path] = text.replace(edit.old, edit.new)
    for path, text in pending.items():
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            raise HotfixError(f"{path} would not compile: {exc}") from exc
    return pending


def _plan_copy(
    hotfix_id: str, copy: Copy, path: Path, pending: dict[Path, str]
) -> None:
    try:
        text = (AGENTS_DIR / copy.src).read_text(encoding="utf-8")
    except OSError as exc:
        raise HotfixError(f"{hotfix_id}: cannot read {copy.src}: {exc}") from exc
    if not path.parent.is_dir():
        raise HotfixError(
            f"{hotfix_id}: {path.parent} does not exist: the hotfix does not fit "
            "this SGLang"
        )
    try:
        installed = path.read_text(encoding="utf-8")
    except OSError:
        installed = None
    if installed != text:
        pending[path] = text


def write(pending: dict[Path, str]) -> None:
    """Write the edited files and drop their cached bytecode."""
    for path, text in pending.items():
        path.write_text(text, encoding="utf-8")
        for cached in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
            cached.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply or check named edits of the installed SGLang sources."
    )
    parser.add_argument("action", choices=("apply", "check"))
    parser.add_argument("ids", help="comma-separated hotfix ids")
    args = parser.parse_args(argv)
    ids = parse_ids(args.ids)
    if not ids:
        parser.error("no hotfix ids given")
    try:
        root = sglang_dir()
        pending = plan(ids, root)
        if args.action == "apply":
            write(pending)
    except (HotfixError, OSError) as exc:
        print(f"cluster_bench hotfix: {exc}", file=sys.stderr)
        return 1
    files = ", ".join(str(p.relative_to(root)) for p in pending) or "nothing (applied)"
    verb = "edited" if args.action == "apply" else "fits; would edit"
    print(f"cluster_bench hotfix {','.join(ids)}: {verb} {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
