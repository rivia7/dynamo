# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU <-> RDMA NIC affinity and NIC-count restriction plans.

The fast path for GPU-direct RDMA is one PCIe-affine NIC per GPU: forcing
several ranks through one non-affine NIC can collapse GPU-direct to host
staging (recipes/deepseek-v4/README.md: ~0.76 vs 15-25 GB/s). For a sweep
with N < GPUs NICs we therefore (1) keep NICs spread across NUMA nodes and
PCIe switches, and (2) send each GPU to the nearest selected NIC, balancing
load so no NIC serves more than ceil(GPUs / N) GPUs.

Distances use ``nvidia-smi topo -m`` link classes when the probe captured
them, else the depth of the common sysfs PCI ancestor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from benchmarks.cluster_bench.domain.schema import HcaInfo, Inventory, NodeInventory

# nvidia-smi topo -m link classes, nearest first.
TOPO_RANK = {"PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}
UNKNOWN_DISTANCE = 6


@dataclass(frozen=True)
class NodeNicPlan:
    node: str
    hcas: tuple[str, ...]  # selected HCAs, sorted by name
    gpu_to_hca: Mapping[int, str]  # GPU index -> HCA serving it
    worst_link: str  # worst topo class among GPU->HCA pairs ("PIX".."SYS"/"?")
    shared: bool  # some HCA serves more than one GPU

    def hcas_for(self, gpus: Sequence[int]) -> tuple[str, ...]:
        return tuple(sorted({self.gpu_to_hca[g] for g in gpus}))


def _sysfs_distance(node: NodeInventory, gpu_bdf: str, hca_bdf: str) -> int:
    a = node.pci_paths.get(gpu_bdf)
    b = node.pci_paths.get(hca_bdf)
    if not a or not b:
        return UNKNOWN_DISTANCE * 10
    pa = [p for p in a.split("/") if p]
    pb = [p for p in b.split("/") if p]
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    return (len(pa) - common) + (len(pb) - common)


def link_class(node: NodeInventory, gpu_index: int, hca: HcaInfo) -> str:
    row = node.topo.get(f"GPU{gpu_index}", {})
    value = row.get(hca.name)
    return value if value in TOPO_RANK else "?"


def distance(node: NodeInventory, gpu_index: int, hca: HcaInfo) -> tuple[int, int, int]:
    """Sort key: topo class, then sysfs path distance, then NUMA mismatch."""
    gpu = next(g for g in node.gpus if g.index == gpu_index)
    cls = TOPO_RANK.get(link_class(node, gpu_index, hca), UNKNOWN_DISTANCE)
    numa_miss = int(
        gpu.numa_node is not None
        and hca.numa_node is not None
        and gpu.numa_node != hca.numa_node
    )
    return (cls, _sysfs_distance(node, gpu.pci_bus_id, hca.pci_bus_id), numa_miss)


def affine_pairs(node: NodeInventory) -> dict[int, str]:
    """GPU -> nearest active HCA with every HCA used at most ceil(G/H) times."""
    hcas = list(node.active_hcas)
    if not hcas:
        raise ValueError(f"{node.name}: no active HCAs")
    return _assign(node, [g.index for g in node.gpus], hcas)


def _assign(
    node: NodeInventory, gpu_indices: Sequence[int], hcas: Sequence[HcaInfo]
) -> dict[int, str]:
    capacity = math.ceil(len(gpu_indices) / len(hcas))
    load = {h.name: 0 for h in hcas}

    # Assign the most constrained GPU (biggest gap between best and second-best
    # NIC) first so a tie elsewhere cannot steal its only good NIC.
    def regret(gpu: int) -> int:
        ranked = sorted(distance(node, gpu, h) for h in hcas)
        if len(ranked) < 2:
            return 0
        best, second = ranked[0], ranked[1]
        # Negative so that larger regret sorts first.
        return -((second[0] - best[0]) * 1000 + (second[1] - best[1]))

    mapping: dict[int, str] = {}
    for gpu in sorted(gpu_indices, key=lambda g: (regret(g), g)):
        for hca in sorted(
            hcas, key=lambda h: (distance(node, gpu, h), load[h.name], h.name)
        ):
            if load[hca.name] < capacity:
                mapping[gpu] = hca.name
                load[hca.name] += 1
                break
    return dict(sorted(mapping.items()))


def assign(node: NodeInventory, hcas: Sequence[HcaInfo]) -> dict[int, str]:
    """GPU -> HCA for an arbitrary HCA selection (same rule the renderer uses)."""
    if not hcas:
        raise ValueError(f"{node.name}: cannot assign GPUs to zero HCAs")
    return _assign(node, [g.index for g in node.gpus], hcas)


def select_nics(node: NodeInventory, count: int) -> tuple[str, ...]:
    """Pick ``count`` active HCAs spread evenly over the node's GPUs.

    Takes the affine NIC of every (G/count)-th GPU, so with 8 GPUs on two NUMA
    nodes N=2 keeps one NIC per socket and N=4 one per GPU pair.
    """
    active = sorted(node.active_hcas, key=lambda h: h.name)
    if count <= 0 or count > len(active):
        raise ValueError(
            f"{node.name}: cannot select {count} of {len(active)} active HCAs"
        )
    if count == len(active):
        return tuple(h.name for h in active)
    pairs = affine_pairs(node)
    gpus = sorted(pairs)
    stride = len(gpus) / count
    chosen: list[str] = []
    for slot in range(count):
        gpu = gpus[int(slot * stride)]
        name = pairs[gpu]
        if name not in chosen:
            chosen.append(name)
    for hca in active:  # fill gaps when two strided GPUs shared one NIC
        if len(chosen) == count:
            break
        if hca.name not in chosen:
            chosen.append(hca.name)
    return tuple(sorted(chosen))


def plan_node(node: NodeInventory, count: int | None) -> NodeNicPlan:
    active = node.active_hcas
    if count is None or count >= len(active):
        selected = tuple(sorted(h.name for h in active))
    else:
        selected = select_nics(node, count)
    hcas = [h for h in active if h.name in selected]
    mapping = _assign(node, [g.index for g in node.gpus], hcas)
    worst = "?"
    worst_rank = 0
    by_name = {h.name: h for h in hcas}
    for gpu, hca_name in mapping.items():
        cls = link_class(node, gpu, by_name[hca_name])
        rank = TOPO_RANK.get(cls, UNKNOWN_DISTANCE)
        if rank > worst_rank:
            worst, worst_rank = cls, rank
    shared = len(set(mapping.values())) < len(mapping)
    return NodeNicPlan(
        node=node.name,
        hcas=selected,
        gpu_to_hca=mapping,
        worst_link=worst,
        shared=shared,
    )


def plan_cluster(inventory: Inventory, count: int | None) -> dict[str, NodeNicPlan]:
    return {node.name: plan_node(node, count) for node in inventory.nodes}


def feasible_counts(inventory: Inventory, requested: Sequence[int]) -> tuple[int, ...]:
    """Requested NIC counts that every node can satisfy, plus the full count."""
    available = min(len(n.active_hcas) for n in inventory.nodes)
    counts = {c for c in requested if 1 <= c <= available}
    counts.add(available)
    return tuple(sorted(counts, reverse=True))


def subsets_for_counts(
    inventory: Inventory, counts: Sequence[int]
) -> dict[int, dict[str, tuple[str, ...]]]:
    """NIC subsets per node for each count (consumed by GPU-direct microbenchmarks)."""
    return {
        c: {name: plan.hcas for name, plan in plan_cluster(inventory, c).items()}
        for c in counts
    }
