# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switch NVLink out of the GPU-GPU path on every node, and back.

Each switch is one NVLINK event ("off", then "restore") so a resumed run
knows whether a node was left with NVLink off (``restore_left_off``). The
methods and the proof are described in domain/nvlink.py.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Any

from benchmarks.cluster_bench.app.context import RunContext
from benchmarks.cluster_bench.domain import nvlink as nvl
from benchmarks.cluster_bench.domain.schema import EventType, PeakResult
from benchmarks.cluster_bench.ports import CommandError

logger = logging.getLogger(__name__)

CMD_TIMEOUT_S = 120.0
P2P_CHECK_MIB = 256  # copy size of the check; enough to reach line rate


@dataclass(frozen=True)
class Switch:
    """How NVLink was taken out of the path and what carries GPU-GPU data."""

    method: str  # bw_mode_off | link_width_0 | nccl_no_p2p
    path: str  # pcie | host


def nvlink_events(ctx: RunContext) -> list[dict[str, Any]]:
    return [dict(e.data) for e in ctx.store.of_type(EventType.NVLINK)]


class NvlinkControl:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        hw = ctx.require_hardware()
        self.arch = hw.gpu.arch
        self.pcie_gbs = hw.gpu.nominal_pcie_gbs

    # ------------------------------------------------------------- commands
    def _run_all(self, command: str) -> dict[str, dict[str, Any]]:
        def one(node: str) -> dict[str, Any]:
            res = self.ctx.cluster.run(
                node, f"{command} 2>&1", timeout=CMD_TIMEOUT_S, check=False
            )
            return {"rc": res.rc, "out": (res.stdout + res.stderr).strip()[-500:]}

        return self.ctx.cluster.fanout(self.ctx.node_names, one)

    def _p2p_check(self) -> dict[str, Any]:
        """GPU-GPU copy benchmark on node0, NVLink counted as off."""
        node = self.ctx.node0
        if self.ctx.simulator is not None:
            peaks = self.ctx.simulator.p2p_without_nvlink(node)
        else:
            from benchmarks.cluster_bench.app.phases.microbench import runner_for

            out_dir = self.ctx.exp_dir / "microbench" / "results"
            remote = str(out_dir / "p2p-nvlink-off.json")
            self.ctx.cluster.run(node, f"mkdir -p {shlex.quote(str(out_dir))}")
            peaks = runner_for(self.ctx).p2p(
                node, remote, size_mib=P2P_CHECK_MIB, iters=5, warmup=2, nvlink=False
            )
        copy = next((p for p in peaks if p.test == "p2p_copy"), None)
        return _copy_summary(node, copy)

    # ---------------------------------------------------------------- off
    def switch_off(self) -> Switch:
        """Take NVLink out of the path on every node (hardware if possible)."""
        notes: list[str] = []
        probe = self._run_all(nvl.GET_BW_MODE)
        modes = {
            n: nvl.parse_bw_mode(r["out"]) if r["rc"] == 0 else None
            for n, r in probe.items()
        }
        common_mode = (
            None if None in modes.values() else next(iter(modes.values()), None)
        )
        widths_out: dict[str, dict[str, Any]] = {}
        if nvl.blackwell_plus(self.arch) or common_mode is None:
            # Blackwell+ prefers the link width; Hopper with a readable
            # bandwidth mode never receives the width command.
            widths_out = self._run_all(nvl.LIST_LINK_WIDTHS)
        widths = {
            n: nvl.parse_link_widths(r["out"]) if r["rc"] == 0 else ()
            for n, r in widths_out.items()
        }
        common_widths = (
            tuple(sorted(set.intersection(*(set(w) for w in widths.values()))))
            if widths
            else ()
        )
        method = nvl.hardware_method(self.arch, common_mode, common_widths)
        if method is not None:
            switch = self._try_hardware(method, modes, widths, notes)
            if switch is not None:
                return switch
        else:
            notes.append(_why_no_hardware(probe, widths_out))
        switch = Switch("nccl_no_p2p", "host")
        self.ctx.store.append(
            EventType.NVLINK,
            {
                "action": "off",
                "method": switch.method,
                "path": switch.path,
                "notes": notes,
            },
        )
        return switch

    def _try_hardware(
        self,
        method: str,
        modes: dict[str, str | None],
        widths: dict[str, tuple[int, ...]],
        notes: list[str],
    ) -> Switch | None:
        if method == "bw_mode_off":
            off_cmd, before = nvl.set_bw_mode("OFF"), {
                n: m or "FULL" for n, m in modes.items()
            }
        else:
            off_cmd = nvl.set_link_width(0)
            before = {n: str(max(w, default=0)) for n, w in widths.items()}
        results = self._run_all(off_cmd)
        failed = sorted(n for n, r in results.items() if r["rc"] != 0)
        record: dict[str, Any] = {
            "action": "off",
            "method": method,
            "before": before,
            "nodes": results,
            "notes": list(notes),
        }
        if failed:
            # Restore the nodes that did switch before falling back.
            record["path"] = None
            self.ctx.store.append(EventType.NVLINK, record)
            self.restore(reason=f"{off_cmd} failed on {', '.join(failed)}")
            notes.append(
                f"{off_cmd} failed on {', '.join(failed)}: {results[failed[0]]['out'][:160]}"
            )
            return None
        try:
            check = self._p2p_check()
        except (CommandError, ValueError, KeyError, FileNotFoundError) as exc:
            check = {"error": str(exc)[:200]}
        record["p2p"] = check
        if not nvl.switch_effective(check.get("median_gbs"), self.pcie_gbs):
            record["path"] = None
            self.ctx.store.append(EventType.NVLINK, record)
            self.restore(reason="GPU-GPU copies still faster than PCIe")
            measured = check.get("median_gbs")
            notes.append(
                f"{off_cmd} did not take NVLink out of the path (GPU-GPU copy "
                f"{measured if measured is not None else 'unmeasured'} GB/s)"
            )
            return None
        path = nvl.path_after_switch(check.get("peer_pairs"))
        record["path"] = path
        if path == "host":
            record["notes"].append(
                "no GPU pair has peer access over PCIe: NCCL stages through host memory"
            )
        self.ctx.store.append(EventType.NVLINK, record)
        return Switch(method, path)

    # ------------------------------------------------------------- restore
    def restore(self, reason: str = "") -> bool:
        """Put NVLink back the way it was before the last hardware switch."""
        off = nvl.left_off([*nvlink_events(self.ctx)])
        if off is None:
            return True
        before = dict(off.get("before") or {})
        if off["method"] == "bw_mode_off":
            results = self._run_all(nvl.set_bw_mode("FULL"))
            check = self._run_all(nvl.GET_BW_MODE)
            ok = all(
                results[n]["rc"] == 0
                and nvl.parse_bw_mode(check[n]["out"]) in (None, "FULL")
                for n in results
            )
        else:
            width = max(
                (int(v) for v in before.values() if str(v).isdigit()), default=0
            )
            results = self._run_all(nvl.set_link_width(width)) if width else {}
            ok = bool(results) and all(r["rc"] == 0 for r in results.values())
        self.ctx.store.append(
            EventType.NVLINK,
            {
                "action": "restore",
                "method": off["method"],
                "ok": ok,
                "nodes": results,
                "reason": reason,
            },
        )
        if not ok:
            self.ctx.note(
                "NVLink restore failed: run 'nvidia-smi nvlink -sBwMode FULL' "
                "(Hopper) or reset the link width on every node"
            )
        return ok

    def restore_left_off(self) -> None:
        """A run that died with NVLink off puts it back before anything else."""
        if nvl.left_off(nvlink_events(self.ctx)) is not None:
            self.restore(reason="resumed with NVLink still off")


def _copy_summary(node: str, copy: PeakResult | None) -> dict[str, Any]:
    if copy is None:
        return {"node": node, "median_gbs": None, "peer_pairs": None}
    details = copy.details or {}
    return {
        "node": node,
        "median_gbs": round(copy.value_gbs, 2),
        "peer_pairs": details.get("peer_pairs"),
        "pairs_measured": details.get("pairs_measured"),
        "min_gbs": details.get("min_gbs"),
        "max_gbs": details.get("max_gbs"),
    }


def _why_no_hardware(
    probe: dict[str, dict[str, Any]], widths: dict[str, dict[str, Any]]
) -> str:
    node, res = next(iter(probe.items()))
    detail = res["out"].splitlines()[0][:120] if res["out"] else f"exit {res['rc']}"
    return (
        "NVLink cannot be switched off on this host (nvidia-smi nvlink -gBwMode on "
        f"{node}: {detail}; no link width of 0 offered)"
    )
