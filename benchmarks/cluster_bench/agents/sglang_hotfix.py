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
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ENV_VAR = "CB_SGLANG_HOTFIXES"


class HotfixError(RuntimeError):
    pass


@dataclass(frozen=True)
class Edit:
    path: str  # relative to the installed sglang package directory
    old: str
    new: str


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

HOTFIXES: dict[str, tuple[Edit, ...]] = {
    "sm120-tilelang-dsa-v1-one-stage": (
        Edit(
            "kernels/ops/attention/dsa/tilelang_kernel.py",
            _TILELANG_OLD,
            _TILELANG_NEW,
        ),
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

    Raises HotfixError for an unknown id, an unreadable file, or an edit whose
    text does not occur exactly once (the hotfix does not fit this SGLang).
    """
    pending: dict[Path, str] = {}
    for hotfix_id in ids:
        edits = HOTFIXES.get(hotfix_id)
        if edits is None:
            known = ", ".join(sorted(HOTFIXES))
            raise HotfixError(f"unknown hotfix {hotfix_id!r} (known: {known})")
        for edit in edits:
            path = root / edit.path
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
    return pending


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
