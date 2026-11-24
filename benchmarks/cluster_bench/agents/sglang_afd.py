#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A-Only worker side: the target model's routed experts live on a simulated F.

An A-Only worker (domain/stages.py, ``a-only``) is the attention side (A) of
attention-FFN disaggregation. SGLang hotfix ``afd-attn-only``
(agents/sglang_hotfix.py) copies this file into the installed sglang package
as ``sglang/srt/cb_afd.py`` (sglang_afd_kernel.py beside it, as
``cb_afd_kernel.py``) and edits two SGLang files to call it, so every process
the worker spawns uses it:

* ``DeepseekV2MoE`` builds ``null_experts_class()`` instead of its routed
  experts in every target layer (``is_nextn`` False); the MTP draft layer
  keeps its real experts.
* The safetensors iterators skip a target layer's routed-expert tensors
  (``skip_tensor``): they are neither read nor allocated, and SGLang's KV pool
  grows into the memory they would have taken.

The null module holds no weights. Its forward launches one GPU kernel that
waits for the simulated F (``CB_AFD_F_WAIT``, from domain/afd.py) and returns
zeros; the router, top-k, shared expert and the DP gather and reduce around
the MoE layer run as before. The first MoE layer of a forward waits f(T) from
the kernel's start; a later layer waits until the previous layer's F result
plus m f(T/m), so A's own work in between overlaps F (the microbatch credit).
The kernel reads %globaltimer and keeps the last F result's time in a per-GPU
stamp, so CUDA graphs capture and replay it.

Env (rendered by domain/afd.py):

    CB_AFD_ATTN_ONLY=1        gate: without it nothing changes
    CB_AFD_TARGET_LAYERS=78   target decoder layers (the MTP layer has this index)
    CB_AFD_F_WAIT=off | lat_ns=..,row_ns=..,wt_ns=..,flop_ns=..,top_k=..,experts=..,m=..

``python3 -P sglang_afd.py --check`` (preflight, on GPU 0) compiles the kernel
and times it eagerly, against a stamp in the past, and in a CUDA graph.
The standard library only at import; torch and triton load when used.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

ATTN_ONLY_ENV = "CB_AFD_ATTN_ONLY"
TARGET_LAYERS_ENV = "CB_AFD_TARGET_LAYERS"
F_WAIT_ENV = "CB_AFD_F_WAIT"
F_WAIT_OFF = "off"
# Every worker process prints it once; deploy looks for it in the worker log.
MARKER = "cluster_bench afd-attn-only: target routed experts nulled"
MAX_WAIT_NS = 2_000_000_000  # keeps the kernel's budget argument an int32
EXIT_UNSUPPORTED = 3
CHECK_WAIT_NS = 300_000
CHECK_LAUNCHES = 3
# A routed-expert tensor of a decoder layer: per-expert names
# (model.layers.3.mlp.experts.17.gate_proj.weight) or fused ones (experts.w13_weight).
_EXPERT_TENSOR = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(?:\d+\.|w13_|w2_|gate_up_proj|down_proj)"
)
_INT_FIELDS = ("top_k", "experts", "m")


@dataclass(frozen=True)
class FWait:
    """Per-layer constants of the F model (domain/afd.py ``FWait``, same fields)."""

    lat_ns: float  # both transfers' latency, per microbatch
    row_ns: float  # wire time of one row
    wt_ns: float  # reading every routed expert of a layer on F
    flop_ns: float  # one row's expert GEMMs on F
    top_k: int
    experts: int
    m: int  # microbatches

    def f_ns(self, rows: float) -> float:
        """One microbatch of ``rows`` rows to F and back."""
        if rows <= 0:
            return 0.0
        hit = 1.0 - (1.0 - self.top_k / self.experts) ** rows
        f_time = max(self.wt_ns * hit, rows * self.flop_ns)
        return self.lat_ns + rows * self.row_ns + f_time

    def budget(self, rows: int, first: bool) -> tuple[bool, float]:
        """(serial, ns) of one MoE layer. Serial: wait from now (the first MoE
        layer of a forward, or one microbatch); else from the previous layer's
        F result."""
        if rows <= 0:
            return True, 0.0
        m = min(self.m, rows)
        if first or m <= 1:
            return True, self.f_ns(rows)
        return False, m * self.f_ns(rows / m)


def parse_f_wait(text: str) -> FWait | None:
    """``CB_AFD_F_WAIT``; None for ``off`` (the null-F bound)."""
    text = text.strip()
    if text == F_WAIT_OFF:
        return None
    names = [f.name for f in fields(FWait)]
    values: dict[str, str] = {}
    for item in text.split(","):
        key, sep, raw = item.partition("=")
        if not sep or key in values:
            raise ValueError(f"{F_WAIT_ENV}: bad or repeated item {item!r}")
        values[key] = raw
    if sorted(values) != sorted(names):
        raise ValueError(
            f"{F_WAIT_ENV} needs {', '.join(names)} (or {F_WAIT_OFF!r}); "
            f"got {', '.join(values) or 'nothing'}"
        )
    try:
        parsed = {
            name: int(values[name]) if name in _INT_FIELDS else float(values[name])
            for name in names
        }
    except ValueError as exc:
        raise ValueError(f"{F_WAIT_ENV}: {exc}") from None
    wait = FWait(**parsed)  # type: ignore[arg-type]
    numbers = (wait.lat_ns, wait.row_ns, wait.wt_ns, wait.flop_ns)
    if not all(math.isfinite(v) and v >= 0 for v in numbers):
        raise ValueError(f"{F_WAIT_ENV}: times must be finite and >= 0: {text!r}")
    if not (1 <= wait.top_k <= wait.experts) or wait.m < 1:
        raise ValueError(f"{F_WAIT_ENV}: need 1 <= top_k <= experts and m >= 1")
    return wait


def enabled() -> bool:
    return os.environ.get(ATTN_ONLY_ENV) == "1"


def target_layers() -> int:
    raw = os.environ.get(TARGET_LAYERS_ENV, "")
    try:
        layers = int(raw)
    except ValueError:
        layers = 0
    if layers <= 0:
        raise RuntimeError(
            f"{ATTN_ONLY_ENV}=1 needs {TARGET_LAYERS_ENV} (the target model's "
            f"decoder layers); got {raw!r}"
        )
    return layers


def marker_line(layers: int | str, first_layer: int, wait: str) -> str:
    """The line a worker prints when it builds its first null module; ``wait``
    is its CB_AFD_F_WAIT (deploy greps MARKER; app/simulate.py prints it too)."""
    shown = "off (null-F bound)" if wait == F_WAIT_OFF else wait
    return (
        f"{MARKER}; layers below {layers} hold no routed experts; "
        f"first MoE layer {first_layer}; F wait {shown}"
    )


def nulls_experts(is_nextn: bool) -> bool:
    """A target MoE layer of an A-Only worker; the MTP draft keeps its experts."""
    return enabled() and not is_nextn


def skip_tensor(name: str) -> bool:
    """A target layer's routed-expert tensor, which an A-Only worker never reads."""
    if not enabled():
        return False
    match = _EXPERT_TENSOR.search(name)
    return match is not None and int(match.group(1)) < target_layers()


class _Process:
    """Per-process state: target MoE layers, the wait, one stamp per GPU."""

    def __init__(self) -> None:
        self.layers: set[int] = set()
        self.first = -1
        self.stamps: dict[int, Any] = {}
        self._wait: tuple[FWait | None] | None = None
        self._dummy_run: Callable[[], bool] | None = None

    @property
    def wait(self) -> FWait | None:
        if self._wait is None:
            raw = os.environ.get(F_WAIT_ENV)
            if raw is None:
                raise RuntimeError(f"{ATTN_ONLY_ENV}=1 needs {F_WAIT_ENV}")
            self._wait = (parse_f_wait(raw),)
        return self._wait[0]

    def register(self, layer_id: int) -> None:
        """A null module was built (model construction, outside any capture)."""
        announce = not self.layers
        self.layers.add(layer_id)
        self.first = min(self.layers)
        wait = self.wait
        if wait is not None:
            _warm_up(self.stamp(None))
        if announce:
            text = F_WAIT_OFF if wait is None else os.environ[F_WAIT_ENV]
            print(marker_line(target_layers(), layer_id, text), flush=True)

    def stamp(self, device: Any) -> Any:
        import torch

        index = device.index if device is not None else None
        if index is None:
            index = torch.cuda.current_device()
        stamp = self.stamps.get(index)
        if stamp is None:
            stamp = torch.zeros(1, dtype=torch.int64, device=f"cuda:{index}")
            self.stamps[index] = stamp
        return stamp

    def in_dummy_run(self) -> bool:
        """SGLang's FlashInfer autotune runs dummy forwards: no F wait there."""
        if self._dummy_run is None:
            try:
                from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
            except ImportError:
                get_in_autotune_dummy_run = lambda: False  # noqa: E731
            self._dummy_run = get_in_autotune_dummy_run
        return bool(self._dummy_run())


PROCESS = _Process()
_KERNEL: Any = None
_NULL_CLASS: Any = None


def _kernel() -> Any:
    """The Triton kernel from <this file's stem>_kernel.py beside this file."""
    global _KERNEL
    if _KERNEL is None:
        path = Path(__file__).with_name(f"{Path(__file__).stem}_kernel.py")
        spec = importlib.util.spec_from_file_location("cb_afd_kernel", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load the F wait kernel from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _KERNEL = module.f_wait_kernel
    return _KERNEL


def _launch(stamp: Any, budget_ns: int, first: bool) -> None:
    _kernel()[(1,)](stamp, budget_ns, FIRST=first, num_warps=1)


def _warm_up(stamp: Any) -> None:
    """Compile both kernel variants before SGLang captures CUDA graphs."""
    _launch(stamp, 0, True)
    _launch(stamp, 0, False)


def launch_wait(hidden_states: Any, layer_id: int) -> None:
    """The F wait of one target MoE layer, on the current stream."""
    wait = PROCESS.wait
    rows = int(hidden_states.shape[0])
    if wait is None or rows <= 0 or PROCESS.in_dummy_run():
        return
    serial, ns = wait.budget(rows, first=layer_id == PROCESS.first)
    budget = int(min(ns, MAX_WAIT_NS))
    _launch(PROCESS.stamp(hidden_states.device), budget, serial)


def null_experts_class() -> type:
    global _NULL_CLASS
    if _NULL_CLASS is None:
        _NULL_CLASS = _build_null_class()
    return _NULL_CLASS


def _build_null_class() -> type:
    from types import SimpleNamespace

    import torch

    class NullRoutedExperts(torch.nn.Module):
        """A target layer's routed experts, on the simulated F: no weights,
        the F wait, zeros out. Carries the FusedMoE attributes DeepseekV2MoE
        reads; any top-k output format is accepted and ignored."""

        num_local_experts = 0
        quant_method = None
        dispatcher = None
        should_fuse_routed_scaling_factor_in_topk = False
        supports_deferred_finalize = False
        use_flashinfer_trtllm_moe = False

        def __init__(
            self, *, layer_id: int, num_fused_shared_experts: int = 0, **_: Any
        ) -> None:
            super().__init__()
            if num_fused_shared_experts:
                raise RuntimeError(
                    "afd-attn-only: shared experts fused into the routed experts "
                    "would vanish with them (run with --disable-shared-experts-fusion)"
                )
            self.layer_id = layer_id
            self.moe_runner_config = SimpleNamespace(
                inplace=False, apply_router_weight_on_input=False
            )
            PROCESS.register(layer_id)

        def forward(self, hidden_states: Any, topk_output: Any = None, **_: Any) -> Any:
            launch_wait(hidden_states, self.layer_id)
            return torch.zeros_like(hidden_states)

        def forward_impl(
            self, hidden_states: Any, topk_output: Any = None, **kwargs: Any
        ) -> Any:
            return self.forward(hidden_states, topk_output, **kwargs)

    return NullRoutedExperts


# --------------------------------------------------------------------------
# --check (preflight)
# --------------------------------------------------------------------------


def _elapsed_ms(run: Callable[[], None]) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def check() -> str:
    """Prove the kernel's three behaviours on GPU 0; raise RuntimeError if not."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    torch.cuda.set_device(0)
    stamp = torch.zeros(1, dtype=torch.int64, device="cuda:0")

    def chain() -> None:  # one forward: the first layer, then from the stamp
        for i in range(CHECK_LAUNCHES):
            _launch(stamp, CHECK_WAIT_NS, i == 0)

    chain()  # compiles both variants
    torch.cuda.synchronize()
    expect_ms = CHECK_LAUNCHES * CHECK_WAIT_NS / 1e6
    eager_ms = _elapsed_ms(chain)
    # A stamp far in the past: A's own work already covered the F time.
    _launch(stamp, CHECK_WAIT_NS, True)
    torch.cuda.synchronize()
    time.sleep(0.005)
    credit_ms = _elapsed_ms(lambda: _launch(stamp, CHECK_WAIT_NS, False))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    graph_ms = _elapsed_ms(lambda: (graph.replay(), graph.replay())) / 2
    for label, ms in (("eager", eager_ms), ("CUDA graph replay", graph_ms)):
        if not expect_ms * 0.95 <= ms <= expect_ms * 2:
            raise RuntimeError(
                f"{label}: {CHECK_LAUNCHES} chained waits of "
                f"{CHECK_WAIT_NS / 1e3:.0f} us took {ms:.3f} ms, not {expect_ms:.2f}"
            )
    if credit_ms > CHECK_WAIT_NS / 1e6 / 3:
        raise RuntimeError(
            f"a wait whose target already passed took {credit_ms:.3f} ms "
            "(it must return at once)"
        )
    return (
        f"{torch.cuda.get_device_name(0)}: {CHECK_LAUNCHES} chained waits of "
        f"{CHECK_WAIT_NS / 1e3:.0f} us took {eager_ms:.3f} ms eager and "
        f"{graph_ms:.3f} ms per CUDA graph replay (expected {expect_ms:.2f}); "
        f"a passed target returned in {credit_ms:.3f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["--check"]:
        print("usage: sglang_afd.py --check", file=sys.stderr)
        return 2
    try:
        report = check()
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"cluster_bench afd-attn-only check: {exc}", file=sys.stderr, flush=True)
        return EXIT_UNSUPPORTED
    print(f"cluster_bench afd-attn-only check: {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
