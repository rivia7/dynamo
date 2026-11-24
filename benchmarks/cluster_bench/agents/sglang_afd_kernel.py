# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The F wait kernel of an A-Only worker; agents/sglang_afd.py loads it by path.

One program of one warp spins on %globaltimer until a target time, then
stores the time in a per-GPU stamp. FIRST: the target is the kernel's start
plus the budget (the first MoE layer of a forward). Otherwise it is the stamp
(the previous layer's F result) plus the budget, so A's work since then counts
against it. Launches are CUDA-graph capturable: the budget is an argument
fixed per captured batch size, the stamp a persistent tensor.
"""

import triton
import triton.language as tl


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit(do_not_specialize=["budget_ns"])
def f_wait_kernel(stamp_ptr, budget_ns, FIRST: tl.constexpr):
    now = _globaltimer()
    if FIRST:
        target = now + budget_ns
    else:
        target = tl.load(stamp_ptr) + budget_ns
    while now < target:
        now = _globaltimer()
    tl.store(stamp_ptr, now)
