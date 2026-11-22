# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware inventory: probe JSON -> typed inventory, uniformity checks,
cluster fingerprint, and local-disk selection for model weights."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from benchmarks.cluster_bench.domain.schema import (
    GIB,
    DiskInfo,
    Inventory,
    NodeInventory,
    stable_hash,
)

# Mount points that are never acceptable for model weights: pseudo or RAM
# backed filesystems, plus per-platform extras (Lium volumes under /mnt are
# object-storage backed and bill egress; elsewhere /mnt is often local NVMe).
FORBIDDEN_WEIGHT_PREFIXES = ("/proc", "/sys", "/dev", "/run", "/boot")
PLATFORM_FORBIDDEN_PREFIXES: dict[str, tuple[str, ...]] = {"lium": ("/mnt",)}
PREFERRED_WEIGHT_ROOTS = ("/workspace", "/root", "/data", "/ephemeral", "/scratch", "/")


class InventoryError(ValueError):
    pass


def node_from_probe(name: str, address: str, probe: Mapping[str, Any]) -> NodeInventory:
    """Validate a probe.py JSON payload into a NodeInventory."""
    payload = dict(probe)
    payload["name"] = name
    payload["address"] = address
    return NodeInventory.model_validate(payload)


def check_uniform(inventory: Inventory) -> list[str]:
    """Problems that make tok/s/GPU comparisons or placements invalid."""
    problems: list[str] = []
    if not inventory.nodes:
        return ["no nodes"]
    ref = inventory.nodes[0]
    ref_gpu = ref.gpus[0].name if ref.gpus else None
    for node in inventory.nodes:
        if not node.gpus:
            problems.append(f"{node.name}: no GPUs visible")
            continue
        names = {g.name for g in node.gpus}
        if len(names) != 1:
            problems.append(f"{node.name}: mixed GPU models {sorted(names)}")
        if node.gpus[0].name != ref_gpu:
            problems.append(
                f"{node.name}: GPU {node.gpus[0].name!r} != node0 {ref_gpu!r}"
            )
        if len(node.gpus) != len(ref.gpus):
            problems.append(
                f"{node.name}: {len(node.gpus)} GPUs != node0 {len(ref.gpus)}"
            )
        if len(node.active_hcas) != len(ref.active_hcas):
            problems.append(
                f"{node.name}: {len(node.active_hcas)} active HCAs != node0 {len(ref.active_hcas)}"
            )
    return problems


def gpus_per_node(inventory: Inventory) -> int:
    counts = {len(n.gpus) for n in inventory.nodes}
    if len(counts) != 1:
        raise InventoryError(f"non-uniform GPU counts per node: {sorted(counts)}")
    return counts.pop()


def driver_major(version: str) -> int:
    head = version.strip().split(".", 1)[0]
    if not head.isdigit():
        raise InventoryError(f"unparseable driver version {version!r}")
    return int(head)


def _node_descriptor(node: NodeInventory) -> dict[str, Any]:
    active = node.active_hcas
    port_rates = sorted({p.rate_gbps for h in active for p in h.active_ports})
    link_layers = sorted({p.link_layer for h in active for p in h.active_ports})
    return {
        "gpu": node.gpus[0].name if node.gpus else None,
        "gpus": len(node.gpus),
        "gpu_mem_gib": round(node.gpus[0].memory_mib / 1024) if node.gpus else 0,
        "cc": node.gpus[0].compute_cap if node.gpus else None,
        "pcie": [node.gpus[0].pcie_gen_max, node.gpus[0].pcie_width_max]
        if node.gpus
        else None,
        "driver_major": driver_major(node.driver_version),
        "hcas": len(active),
        "hca_rate_gbps": port_rates,
        "link_layer": link_layers,
        "nvlink": node.nvlink_active,
        "cpu": node.cpu_model,
        "cpus": node.cpu_count,
        # Round RAM to 64 GiB so reboots/firmware reservations do not change it.
        "ram_64gib": round(node.mem_total_bytes / (64 * GIB)),
    }


def fingerprint(inventory: Inventory) -> str:
    """Stable id of the hardware shape; excludes hostnames, IPs, UUIDs, BDFs.

    Two rentals of the same SKU and fabric get the same fingerprint, so the
    knowledge base accumulates evidence per hardware shape.
    """
    descriptors = sorted(
        (_node_descriptor(n) for n in inventory.nodes),
        key=lambda d: repr(sorted(d.items())),
    )
    return stable_hash({"nodes": len(inventory.nodes), "shape": descriptors}, 16)


def shape_fingerprint(inventory: Inventory) -> str:
    """Like ``fingerprint`` but without the node count.

    2-, 4- and 8-node rentals of one SKU and fabric share it, so the
    knowledge base can line them up as a scaling series.
    """
    descriptors = sorted(
        {
            repr(sorted(_node_descriptor(n).items())): _node_descriptor(n)
            for n in inventory.nodes
        }.values(),
        key=lambda d: repr(sorted(d.items())),
    )
    return stable_hash({"shape": descriptors}, 16)


def hardware_descriptor(inventory: Inventory) -> dict[str, Any]:
    """Human-readable summary of node0 plus the node count (for reports)."""
    node0 = inventory.nodes[0]
    desc = _node_descriptor(node0)
    desc["nodes"] = len(inventory.nodes)
    desc["driver"] = node0.driver_version
    desc["cuda"] = node0.cuda_version
    desc["ofed"] = node0.ofed_version
    desc["kernel"] = node0.kernel
    return desc


def _weight_root_rank(path: str) -> int:
    for index, prefix in enumerate(PREFERRED_WEIGHT_ROOTS):
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return index
    return len(PREFERRED_WEIGHT_ROOTS)


def forbidden_weight_prefixes(platform: str = "generic") -> tuple[str, ...]:
    return FORBIDDEN_WEIGHT_PREFIXES + PLATFORM_FORBIDDEN_PREFIXES.get(platform, ())


def usable_weight_disks(
    node: NodeInventory, platform: str = "generic"
) -> list[DiskInfo]:
    forbidden = forbidden_weight_prefixes(platform)
    return [
        d
        for d in node.disks
        if not any(d.path == p or d.path.startswith(p + "/") for p in forbidden)
    ]


def pick_model_root(
    nodes: Sequence[NodeInventory],
    need_bytes: int,
    preferred: str | None = None,
    platform: str = "generic",
) -> str:
    """Choose one local path that has ``need_bytes`` free on every node."""
    if preferred:
        for node in nodes:
            disk = _disk_for(node, preferred, platform)
            if disk is None or disk.free_bytes < need_bytes:
                free = 0 if disk is None else disk.free_bytes
                raise InventoryError(
                    f"{node.name}: {preferred} has {free / GIB:.0f} GiB free, need {need_bytes / GIB:.0f} GiB"
                )
        return preferred
    candidates: dict[str, int] = {}
    for disk in usable_weight_disks(nodes[0], platform):
        candidates[disk.path] = _weight_root_rank(disk.path)
    for path in sorted(candidates, key=lambda p: (candidates[p], p)):
        root = (
            "/workspace"
            if path == "/" and _all_have(nodes, "/workspace", platform)
            else path
        )
        if all(
            (d := _disk_for(n, root, platform)) is not None
            and d.free_bytes >= need_bytes
            for n in nodes
        ):
            return root
    raise InventoryError(
        f"no local disk with {need_bytes / GIB:.0f} GiB free on every node; "
        "free space or pass --model-root"
    )


def _all_have(
    nodes: Sequence[NodeInventory], path: str, platform: str = "generic"
) -> bool:
    return all(_disk_for(n, path, platform) is not None for n in nodes)


def _disk_for(
    node: NodeInventory, path: str, platform: str = "generic"
) -> DiskInfo | None:
    """The disk whose mount point is the longest prefix of ``path``."""
    best: DiskInfo | None = None
    for disk in usable_weight_disks(node, platform):
        mount = disk.path.rstrip("/") or "/"
        if path == mount or path.startswith(mount.rstrip("/") + "/") or mount == "/":
            if best is None or len(mount) > len(best.path.rstrip("/") or "/"):
                best = disk
    return best
