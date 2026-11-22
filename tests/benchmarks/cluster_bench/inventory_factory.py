# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic but realistic cluster inventories for unit tests."""

from __future__ import annotations

from benchmarks.cluster_bench.domain.schema import (
    DiskInfo,
    GpuInfo,
    HcaInfo,
    HcaPort,
    Inventory,
    NodeInventory,
)

GIB = 1 << 30


def make_node(
    index: int,
    *,
    gpu_name: str = "NVIDIA H200",
    gpus: int = 8,
    hcas: int = 8,
    memory_mib: int = 143771,
    nvlink: bool = True,
    hca_rate_gbps: float = 400.0,
    mem_total_bytes: int = 2048 * GIB,
    free_bytes: int = 6000 * GIB,
    gpus_per_switch: int | None = None,
) -> NodeInventory:
    """A node where GPU i shares a PCIe switch with mlx5_(i // ratio).

    Two NUMA nodes: the first half of GPUs and NICs on socket 0.
    """
    per_switch = gpus_per_switch or max(1, gpus // max(1, hcas))
    gpu_list = []
    hca_list = []
    pci_paths: dict[str, str] = {}
    topo: dict[str, dict[str, str]] = {}
    switches = max(1, gpus // per_switch)
    for g in range(gpus):
        switch = g // per_switch
        numa = 0 if switch < switches / 2 else 1
        bdf = f"0000:{0x18 + 0x10 * g:02x}:00.0"
        gpu_list.append(
            GpuInfo(
                index=g,
                name=gpu_name,
                uuid=f"GPU-{index:02d}{g:02d}",
                pci_bus_id=bdf,
                memory_mib=memory_mib,
                numa_node=numa,
                compute_cap="9.0",
                pcie_gen_max=5,
                pcie_width_max=16,
            )
        )
        pci_paths[
            bdf
        ] = f"/sys/devices/pci0000:{switch:02x}/0000:{switch:02x}:01.0/0000:{0x40 + switch:02x}:00.0/{bdf}"
    for h in range(hcas):
        switch = h * switches // hcas
        numa = 0 if switch < switches / 2 else 1
        bdf = f"0000:{0x19 + 0x10 * h:02x}:00.0"
        hca_list.append(
            HcaInfo(
                name=f"mlx5_{h}",
                pci_bus_id=bdf,
                numa_node=numa,
                netdev=f"ibp{h}s0",
                ports=(
                    HcaPort(
                        port=1,
                        state="4: ACTIVE",
                        rate=f"{int(hca_rate_gbps)} Gb/sec (4X NDR)",
                        rate_gbps=hca_rate_gbps,
                        link_layer="InfiniBand",
                    ),
                ),
            )
        )
        pci_paths[
            bdf
        ] = f"/sys/devices/pci0000:{switch:02x}/0000:{switch:02x}:01.0/0000:{0x40 + switch:02x}:00.0/{bdf}"
    for g in range(gpus):
        row: dict[str, str] = {}
        g_switch = g // per_switch
        g_numa = gpu_list[g].numa_node
        for h, hca in enumerate(hca_list):
            h_switch = h * switches // hcas
            if h_switch == g_switch:
                row[hca.name] = "PIX"
            elif hca.numa_node == g_numa:
                row[hca.name] = "NODE"
            else:
                row[hca.name] = "SYS"
        topo[f"GPU{g}"] = row
    return NodeInventory(
        name=f"node{index}",
        hostname=f"host-{index}",
        address=f"10.42.0.{index + 1}",
        gpus=tuple(gpu_list),
        hcas=tuple(hca_list),
        pci_paths=pci_paths,
        topo=topo,
        driver_version="580.65.06",
        cuda_version="13.0",
        cpu_model="Intel(R) Xeon(R) Platinum 8480+",
        cpu_count=224,
        mem_total_bytes=mem_total_bytes,
        disks=(
            DiskInfo(
                path="/",
                device="overlay",
                fs_type="overlay",
                total_bytes=200 * GIB,
                free_bytes=150 * GIB,
            ),
            DiskInfo(
                path="/workspace",
                device="/dev/nvme0n1",
                fs_type="ext4",
                total_bytes=7000 * GIB,
                free_bytes=free_bytes,
            ),
            DiskInfo(
                path="/mnt/volume",
                device="s3fs",
                fs_type="ext4",
                total_bytes=10000 * GIB,
                free_bytes=10000 * GIB,
            ),
        ),
        docker_version="27.3.1",
        kernel="5.15.0-119-generic",
        ofed_version="MLNX_OFED_LINUX-24.10-1.1.4.0",
        nvidia_peermem=True,
        nvlink_active=nvlink,
    )


def make_inventory(nodes: int = 4, **kwargs) -> Inventory:
    return Inventory(nodes=tuple(make_node(i, **kwargs) for i in range(nodes)))


def b200_inventory(nodes: int = 4) -> Inventory:
    return make_inventory(nodes, gpu_name="NVIDIA B200", memory_mib=183359)


def h100_inventory(nodes: int = 4) -> Inventory:
    return make_inventory(nodes, gpu_name="NVIDIA H100 80GB HBM3", memory_mib=81559)


def rtx6000_inventory(nodes: int = 4) -> Inventory:
    return make_inventory(
        nodes,
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        memory_mib=97887,
        hcas=4,
        nvlink=False,
        gpus_per_switch=2,
        hca_rate_gbps=200.0,
        mem_total_bytes=1024 * GIB,
    )
