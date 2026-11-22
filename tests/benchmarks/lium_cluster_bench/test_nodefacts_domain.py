# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/nodefacts.py: parsing, findings, health deltas,
clock offsets, the KB signature and the synthetic dry-run data.

``fixtures/nodefacts/node_facts_h100.json`` is real agent output for the
unhealthy 2-GPU H100 node built in ``test_nodefacts_agent.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.lium_cluster_bench.agents import node_facts as agent
from benchmarks.lium_cluster_bench.domain import nodefacts as nf
from benchmarks.lium_cluster_bench.domain.base import stable_hash
from tests.benchmarks.lium_cluster_bench.lium_fakes import fixture_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "nodefacts"
GIB = 1 << 30
HOUR = 3600.0


def h100_doc() -> dict[str, Any]:
    return json.loads((FIXTURES / "node_facts_h100.json").read_text())


def h100_facts(node: str = "node0") -> nf.NodeFacts:
    return nf.parse_node_facts(h100_doc(), node=node)


def software_doc() -> dict[str, Any]:
    return json.loads((FIXTURES / "software_stack.json").read_text())


def by_code(findings: list[nf.FactFinding]) -> dict[str, nf.FactFinding]:
    out: dict[str, nf.FactFinding] = {}
    for finding in findings:
        assert finding.code not in out, f"duplicate code {finding.code}"
        out[finding.code] = finding
    return out


def synthetic(nodes: int = 2, gpu: str = "NVIDIA H200") -> dict[str, nf.NodeFacts]:
    return nf.synthetic_facts([f"node{i}" for i in range(nodes)], gpu)


def with_gpu(facts: nf.NodeFacts, index: int, **update: Any) -> nf.NodeFacts:
    gpus = list(facts.gpus)
    gpus[index] = gpus[index].model_copy(update=update)
    return facts.model_copy(update={"gpus": tuple(gpus)})


def nested(facts: nf.NodeFacts, section: str, **update: Any) -> nf.NodeFacts:
    current = getattr(facts, section)
    return facts.model_copy(update={section: current.model_copy(update=update)})


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_fixture_document() -> None:
    facts = h100_facts()
    assert facts.available and facts.node == "node0" and facts.notes == ()
    assert facts.hostname == "lium-h100-07" and facts.duration_s == 6.25
    assert [g.label for g in facts.gpus] == [
        "GPU0 (0000:18:00.0)",
        "GPU1 (0000:3a:00.0)",
    ]
    gpu1 = next(link for link in facts.pcie if link.bdf == "0000:3a:00.0")
    assert gpu1.current_label == "Gen4 x8" and gpu1.max_label == "Gen5 x16"
    assert gpu1.width_degraded and gpu1.speed_degraded
    acs = next(link for link in facts.pcie if link.bdf == "0000:17:00.0")
    assert acs.acs_redirect is True and not acs.root_port
    assert facts.kernel.iommu_enabled is True
    assert facts.kernel.iommu_passthrough is False
    assert facts.kernel.release_tuple == (5, 15, 0)
    assert facts.dmabuf_capable is False  # proprietary module
    assert facts.cgroup.cpu_quota_cores is None
    nvlink = facts.nvlink[1]
    assert (nvlink.active, nvlink.total) == (17, 18)
    assert nvlink.error_totals() == {
        "crc_errors": 12,
        "recovery_errors": 0,
        "replay_errors": 3,
    }
    port = facts.hcas[1].ports[0]
    assert port.active and port.rate_gbps == 200.0 and port.link_layer == "Ethernet"
    assert facts.kernel_log.events[0].kind == "mlx5"


def test_parse_round_trip_and_node_override() -> None:
    facts = h100_facts()
    again = nf.parse_node_facts(facts.model_dump(mode="json"))
    assert again == facts
    assert nf.parse_node_facts(h100_doc(), node="node7").node == "node7"


def test_parse_is_lenient() -> None:
    doc = h100_doc()
    doc["future_section"] = {"x": 1}
    doc["cpu"]["new_counter"] = 3
    doc["gpus"][0]["index"] = "not-a-number"
    doc["memory"] = "garbage"
    doc["available"] = False  # the agent cannot declare itself unavailable
    facts = nf.parse_node_facts(doc)
    assert facts.available is True
    assert [g.bus_id for g in facts.gpus] == ["0000:3a:00.0"]
    assert facts.memory == nf.MemoryFacts()
    assert facts.cpu.governors == {"performance": 2, "powersave": 2}
    sources = {n.source: n.error for n in facts.notes}
    assert sources["parse:future_section"] == "unknown field dropped"
    assert sources["parse:cpu.new_counter"] == "unknown field dropped"
    assert sources["parse:gpus"].startswith("1 invalid element(s) dropped")
    assert sources["parse:memory"].startswith("section dropped")


@pytest.mark.parametrize("obj", [[], "x", {"kind": "software_stack"}, {}])
def test_parse_rejects_wrong_documents(obj: Any) -> None:
    with pytest.raises(ValueError):
        nf.parse_node_facts(obj)


def test_b200_r580_gpu_record_findings() -> None:
    """The current-driver XML flows through the agent parser into findings."""
    xml = fixture_text(FIXTURES / "nvidia_smi_q_x_b200_r580.txt")
    doc = {"kind": "node_facts", "gpus": agent.parse_smi_xml(xml)["gpus"]}
    facts = nf.parse_node_facts(doc, node="node0")
    assert facts.notes == ()  # every agent field is a model field
    (gpu,) = facts.gpus
    assert gpu.fabric_clique_id == "32766" and gpu.clocks_event_us["sw_power_cap"]
    findings = nf.fact_findings({"node0": facts})
    codes = {f.code: f.severity for f in findings if f.node == "node0"}
    assert codes == {
        "gpu_remap_pending": "warn",
        "gpu_ecc_uncorrectable_aggregate": "warn",
        "gpu_rows_remapped_uncorrectable": "info",
    }
    snap = nf.health_of(facts)
    assert snap.gpu["0000:1b:00.0"]["throttle_us.sw_power_cap"] == 13377212
    assert snap.gpu["0000:1b:00.0"]["ecc.volatile.sram_uncorrectable_parity"] == 0


def test_agent_documents_match_the_models(tmp_path: Path) -> None:
    host = agent.Host(
        root=str(tmp_path),
        run=lambda argv, timeout=None: None,
        which=lambda name: None,
        distributions=lambda: [("sglang", "0.5.22")],
        environ={"NCCL_DEBUG": "INFO"},
        search_path=[],
    )
    stack = nf.parse_software(agent.collect_software(host, node="node0"))
    assert stack.packages == {"sglang": "0.5.22"} and stack.torch is None
    assert {n.source for n in stack.notes} >= {"torch probe", "ucx_info"}
    assert not any(n.source.startswith("parse:") for n in stack.notes)
    weights = tmp_path / "model-00001-of-00001.safetensors"
    weights.write_bytes(b"\0" * (1 << 20))
    read = nf.parse_disk_read(agent.disk_read(host, str(tmp_path), node="node1"))
    assert read.node == "node1" and read.bytes == 1 << 20 and read.gbs
    assert not any(n.source.startswith("parse:") for n in read.notes)
    facts = nf.parse_node_facts(agent.collect_host(host, node="node2"))
    assert facts.available and facts.gpus == ()
    assert not any(n.source.startswith("parse:") for n in facts.notes)


def test_unavailable_facts() -> None:
    facts = nf.unavailable_facts("node3", "exit 124: timed out")
    assert facts.available is False and facts.notes[0].error == "exit 124: timed out"
    (finding,) = nf.fact_findings({"node3": facts})
    assert (finding.code, finding.severity, finding.node) == (
        "facts_unavailable",
        "warn",
        "node3",
    )
    assert "timed out" in finding.message
    assert nf.health_of(facts).available is False


def test_parse_software_and_disk_documents() -> None:
    stack = nf.parse_software(software_doc(), image="nvcr.io/x:1")
    assert stack.image == "nvcr.io/x:1" and stack.torch.nccl == "2.28.3"
    assert stack.notes == ()
    assert stack.ucx.version == "1.19.0" and len(stack.ibv_devices) == 8
    with pytest.raises(ValueError):
        nf.parse_software({"kind": "node_facts"})
    read = nf.parse_disk_read(
        {
            "kind": "disk_read",
            "version": 1,
            "node": "",
            "hostname": "h",
            "requested_path": "/w",
            "path": "/w/a.safetensors",
            "file_bytes": 5,
            "bytes": 5,
            "seconds": 1.0,
            "gbs": 5e-9,
            "method": "o_direct",
            "direct": True,
            "cache": "bypassed",
            "block_bytes": 16,
            "error": None,
            "notes": [],
        },
        node="node1",
    )
    assert read.node == "node1" and read.method == "o_direct" and read.bytes == 5
    failed = nf.failed_disk_read("node2", "/w", "exit 1: boom")
    assert failed.error == "exit 1: boom" and failed.gbs is None


# --------------------------------------------------------------------------
# Static findings
# --------------------------------------------------------------------------


EXPECTED_H100 = {
    "nvlink_down": "error",
    "pcie_width_degraded": "warn",
    "pcie_upstream_degraded": "warn",
    "pcie_acs_redirect": "warn",
    "gpu_power_capped": "warn",
    "gpu_pcie_replays": "warn",
    "gpu_throttled": "warn",
    "hca_rate_mismatch": "warn",
    "iommu_translated": "warn",
    "memlock_limited": "warn",
    "shm_small": "warn",
    "pids_limited": "warn",
    "kernel_log_xid": "warn",
    "pcie_speed_degraded": "warn",
    "pcie_aer_correctable": "info",
    "nvlink_errors": "info",
    "hca_error_counters": "info",
    "thp_always": "info",
    "cpu_governor": "info",
    "numa_balancing": "info",
    "kernel_log_mlx5": "info",
    "kernel_log_aer": "info",
}


def test_fact_findings_on_unhealthy_h100_node() -> None:
    findings = nf.fact_findings({"node0": h100_facts()})
    codes = by_code(findings)
    assert {c: f.severity for c, f in codes.items()} == EXPECTED_H100
    ranks = [nf.SEVERITY_RANK[f.severity] for f in findings]
    assert ranks == sorted(ranks, reverse=True)  # worst first
    assert codes["pcie_width_degraded"].subject == "GPU1 (0000:3a:00.0)"
    assert "at Gen4 x8, capable of Gen5 x16" in codes["pcie_width_degraded"].message
    upstream = codes["pcie_upstream_degraded"]
    assert (
        upstream.subject == "0000:16:00.0"
        and "above GPU0 (0000:18:00.0)" in upstream.message
    )
    assert codes["pcie_acs_redirect"].subject == "0000:17:00.0"
    assert "0x001d" in codes["pcie_acs_redirect"].message
    assert "500 W of default 700 W" in codes["gpu_power_capped"].message
    assert "17/18 links active (down: 7)" in codes["nvlink_down"].message
    assert "200 Gb/s on mlx5_1/1" in codes["hca_rate_mismatch"].message
    assert "DMA-FQ" in codes["iommu_translated"].message
    xid = codes["kernel_log_xid"]
    assert xid.data["xids"] == {"63@0000:3a:00.0": 1, "94@0000:3a:00.0": 1}
    assert "3.4 h before collection" in xid.message
    assert "13@" not in xid.message  # another tenant's GPU
    for finding in findings:
        assert finding.node == "node0" and finding.message
        json.dumps(finding.model_dump(mode="json"))


@pytest.mark.parametrize(
    "gpu_name",
    [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H200",
        "NVIDIA B200",
        "NVIDIA B300 SXM6 AC",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    ],
)
def test_synthetic_facts_are_healthy_and_deterministic(gpu_name: str) -> None:
    names = ["node0", "node1", "node2", "node3"]
    facts = nf.synthetic_facts(names, gpu_name)
    assert list(facts) == names
    assert facts == nf.synthetic_facts(names, gpu_name)
    assert nf.fact_findings(facts) == []
    node0 = facts["node0"]
    assert len(node0.gpus) == 8 and len(node0.hcas) == 8
    assert {g.product for g in node0.gpus} == {gpu_name}
    assert len({g.uuid for f in facts.values() for g in f.gpus}) == 32
    assert nf.parse_node_facts(json.loads(node0.model_dump_json())) == node0
    delta = nf.health_delta(node0, node0)
    assert delta.complete and not delta.increments and not delta.changes
    rtx = "RTX" in gpu_name
    assert bool(node0.nvlink) is not rtx
    assert (node0.gpus[0].fabric_state is None) is rtx


def test_heterogeneous_cluster_findings() -> None:
    facts = synthetic(3)
    facts["node1"] = with_gpu(
        nested(facts["node1"], "kernel", release="6.8.0-48-generic"),
        3,
        vbios="96.00.B0.00.01",
    )
    hcas = list(facts["node2"].hcas)
    hcas[0] = hcas[0].model_copy(update={"fw_ver": "28.41.1000"})
    facts["node2"] = nested(facts["node2"], "driver", version="575.57.08").model_copy(
        update={"hcas": tuple(hcas)}
    )
    codes = by_code(nf.fact_findings(facts))
    assert set(codes) == {
        "heterogeneous_driver",
        "heterogeneous_kernel",
        "heterogeneous_vbios",
        "heterogeneous_hca_firmware",
        "vbios_mixed_in_node",
        "hca_firmware_mixed_in_node",
    }
    driver = codes["heterogeneous_driver"]
    assert driver.node == "cluster" and driver.severity == "warn"
    assert driver.message == (
        "Nodes differ in NVIDIA driver version: 575.57.08 on node2; "
        "580.95.05 on node0,node1"
    )
    assert driver.data["values"] == {
        "575.57.08": ["node2"],
        "580.95.05": ["node0", "node1"],
    }
    assert codes["vbios_mixed_in_node"].node == "node1"


@pytest.mark.parametrize(
    ("update", "code", "severity"),
    [
        ({"power_limit_w": 450.0}, "gpu_power_capped", "warn"),
        (
            {
                "power_limit_w": 650.0,
                "power_requested_w": 700.0,
                "power_default_w": 650.0,
            },
            "gpu_power_enforced_below_requested",
            "warn",
        ),
        ({"persistence_mode": "Disabled"}, "gpu_persistence_off", "info"),
        ({"mig_pending": "Enabled"}, "gpu_mig_enabled", "error"),
        ({"compute_mode": "Exclusive_Process"}, "gpu_compute_mode", "warn"),
        ({"compute_mode": "Prohibited"}, "gpu_compute_mode", "error"),
        (
            {"ecc_current": "Disabled", "ecc_pending": "Disabled"},
            "gpu_ecc_disabled",
            "warn",
        ),
        ({"ecc_pending": "Disabled"}, "gpu_ecc_pending_change", "warn"),
        ({"ecc_volatile_uncorrected": 2}, "gpu_ecc_uncorrectable_volatile", "error"),
        ({"ecc_aggregate_uncorrected": 1}, "gpu_ecc_uncorrectable_aggregate", "warn"),
        ({"remap_failure": True}, "gpu_remap_failure", "error"),
        ({"remap_pending": True}, "gpu_remap_pending", "warn"),
        ({"remapped_uncorrectable": 2}, "gpu_rows_remapped_uncorrectable", "info"),
        ({"retired_dbe": 2}, "gpu_retired_pages", "warn"),
        ({"pcie_replay": 5}, "gpu_pcie_replays", "warn"),
        ({"reset_required": "Yes"}, "gpu_reset_required", "error"),
        ({"recovery_action": "Drain and Reset"}, "gpu_recovery_action", "error"),
        ({"fabric_state": "In Progress"}, "nvswitch_fabric_not_ready", "error"),
        ({"fabric_status": "Failure"}, "nvswitch_fabric_not_ready", "error"),
        (
            {"fabric_health": {"bandwidth": "Degraded"}},
            "nvswitch_fabric_degraded",
            "warn",
        ),
        (
            {"clocks_event_active": ("gpu_idle", "sw_thermal_slowdown")},
            "gpu_throttled",
            "warn",
        ),
        (
            {"pcie_gen_device_max": 6, "pcie_gen_host_max": 5},
            "pcie_slot_limited",
            "warn",
        ),
    ],
)
def test_gpu_findings(update: dict[str, Any], code: str, severity: str) -> None:
    facts = synthetic(1)
    facts["node0"] = with_gpu(facts["node0"], 2, **update)
    codes = by_code(nf.fact_findings(facts))
    assert code in codes, sorted(codes)
    assert codes[code].severity == severity
    assert codes[code].subject.startswith("GPU2 (")


@pytest.mark.parametrize(
    ("flavor", "release", "code", "severity"),
    [
        ("proprietary", "6.8.0-48-generic", "gpudirect_rdma_unavailable", "warn"),
        ("open", "5.4.0-200-generic", "gpudirect_rdma_unavailable", "warn"),
        ("open", "6.8.0-48-generic", "gpudirect_rdma_dmabuf_only", "info"),
    ],
)
def test_gpudirect_rdma_findings(
    flavor: str, release: str, code: str, severity: str
) -> None:
    facts = synthetic(1)["node0"]
    facts = nested(facts, "driver", flavor=flavor)
    facts = nested(facts, "kernel", release=release)
    facts = facts.model_copy(
        update={"modules": {**facts.modules, "nvidia_peermem": False}}
    )
    codes = by_code(nf.fact_findings({"node0": facts}))
    assert set(codes) == {code} and codes[code].severity == severity


def test_module_list_unreadable_is_not_a_missing_peermem() -> None:
    facts = synthetic(1)["node0"]
    facts = facts.model_copy(
        update={"modules": {"nvidia_peermem": False, "nvidia": False}}
    )
    assert set(by_code(nf.fact_findings({"node0": facts}))) == {
        "kernel_modules_unknown"
    }


def test_platform_findings() -> None:
    facts = synthetic(2)
    node0 = nested(facts["node0"], "limits", memlock_soft=65536)
    node0 = nested(node0, "cgroup", cpu_quota_us=1600000, cpu_period_us=100000)
    node0 = nested(
        node0, "time_sync", synchronized=False, leap_status="Not synchronised"
    )
    node0 = nested(node0, "kernel_log", available=False)
    node1 = nested(facts["node1"], "time_sync", synchronized=None, source=None)
    node1 = nested(node1, "cgroup", memory_events={"oom_kill": 2})
    codes = by_code(nf.fact_findings({"node0": node0, "node1": node1}))
    assert codes["memlock_limited"].severity == "info"
    assert codes["cpu_quota"].data == {"quota_cores": 16.0, "cpus": 224}
    assert codes["clock_unsynchronized"].node == "node0"
    assert "Not synchronised" in codes["clock_unsynchronized"].message
    assert codes["time_sync_unknown"].data["nodes"] == ["node1"]
    assert codes["kernel_log_unavailable"].data["nodes"] == ["node0"]
    assert codes["cgroup_oom_kills_before"].node == "node1"


def test_link_findings() -> None:
    facts = synthetic(1)["node0"]
    nvlink = list(facts.nvlink)
    links = list(nvlink[0].links)
    links[2] = links[2].model_copy(update={"speed_gbs": 13.281})
    nvlink[0] = nvlink[0].model_copy(update={"links": tuple(links)})
    nvlink[1] = nvlink[1].model_copy(
        update={
            "links": tuple(
                link.model_copy(update={"speed_gbs": None}) for link in nvlink[1].links
            )
        }
    )
    hcas = list(facts.hcas)
    port = (
        hcas[4]
        .ports[0]
        .model_copy(update={"state": "1: DOWN", "phys_state": "3: Disabled"})
    )
    hcas[4] = hcas[4].model_copy(update={"ports": (port,)})
    labels = [f"GPU{i}" for i in range(8)]
    read = {
        r: {
            c: "X" if r == c else ("OK" if (r < "GPU4") == (c < "GPU4") else "CNS")
            for c in labels
        }
        for r in labels
    }
    facts = facts.model_copy(
        update={"nvlink": tuple(nvlink), "hcas": tuple(hcas), "p2p": {"read": read}}
    )
    codes = by_code(nf.fact_findings({"node0": facts}))
    assert codes["nvlink_speed_mismatch"].subject.startswith("GPU0")
    assert codes["nvlink_inactive"].subject.startswith("GPU1")
    assert codes["hca_port_not_active"].subject == "mlx5_4/1"
    p2p = codes["p2p_unsupported"]
    assert (
        p2p.data["pairs"] == 32
        and p2p.data["total"] == 56
        and p2p.data["statuses"] == ["CNS"]
    )


def test_gpu_link_speed_idle_versus_degraded() -> None:
    facts = synthetic(1)["node0"]
    pcie = list(facts.pcie)
    pcie[0] = pcie[0].model_copy(update={"current_speed_gts": 2.5})  # idle: Gen1
    pcie[1] = pcie[1].model_copy(update={"current_speed_gts": 16.0})  # trained Gen4
    pcie[8] = pcie[8].model_copy(update={"current_speed_gts": 16.0})  # HCA mlx5_0
    codes = by_code(
        nf.fact_findings({"node0": facts.model_copy(update={"pcie": tuple(pcie)})})
    )
    assert codes["pcie_speed_idle"].severity == "info"
    assert codes["pcie_speed_idle"].subject == "GPU0 (0000:18:00.0)"
    degraded = codes["pcie_speed_degraded"]
    assert degraded.severity == "warn"
    assert degraded.subject == "GPU1 (0000:29:00.0),mlx5_0 (0000:19:00.0)"
    assert "at Gen4 x16, capable of Gen5 x16" in degraded.message


def test_iommu_inference_without_group_types() -> None:
    assert (
        nf.KernelFacts(
            iommu_groups=10, cmdline_flags={"intel_iommu": "on"}
        ).iommu_passthrough
        is False
    )
    assert (
        nf.KernelFacts(
            cmdline_flags={"intel_iommu": "on", "iommu": "pt"}
        ).iommu_passthrough
        is True
    )
    assert (
        nf.KernelFacts(cmdline_flags={"iommu.passthrough": "0"}).iommu_passthrough
        is False
    )
    assert nf.KernelFacts().iommu_passthrough is None
    assert nf.KernelFacts(iommu_groups=0).iommu_enabled is False
    mixed = nf.KernelFacts(iommu_group_types={"a": "identity", "b": "DMA"})
    assert mixed.iommu_passthrough is False


def test_software_findings() -> None:
    stack = nf.parse_software(software_doc())
    assert nf.software_findings(stack) == []
    no_rdma = stack.model_copy(update={"ibv_devices": ()})
    no_gpu = stack.model_copy(
        update={"torch": stack.torch.model_copy(update={"cuda_available": False})}
    )
    mismatch = stack.model_copy(
        update={"torch": stack.torch.model_copy(update={"nccl": "2.27.7"})}
    )
    assert {f.code for f in nf.software_findings(no_rdma)} == {
        "container_no_rdma_devices"
    }
    assert [f.severity for f in nf.software_findings(no_gpu)] == ["error"]
    assert {f.code for f in nf.software_findings(mismatch)} == {"nccl_version_mismatch"}
    no_torch = stack.model_copy(
        update={"torch": nf.TorchInfo(errors=("import torch: boom",))}
    )
    (finding,) = nf.software_findings(no_torch)
    assert finding.code == "container_torch_unavailable" and "boom" in finding.message
    (missing,) = nf.software_findings(None, node="node0")
    assert missing.code == "software_unavailable"


def test_disk_findings() -> None:
    reads = {
        "node0": nf.synthetic_disk_read("node0", "/w", gbs=6.0),
        "node1": nf.synthetic_disk_read("node1", "/w", gbs=0.3),
        "node2": nf.failed_disk_read("node2", "/w", "exit 124: timed out"),
        "node3": nf.synthetic_disk_read("node3", "/w", gbs=2.0).model_copy(
            update={"method": "buffered", "direct": False, "cache": "unknown"}
        ),
    }
    codes = {(f.code, f.node) for f in nf.disk_findings(reads)}
    assert codes == {
        ("disk_read_slow", "node1"),
        ("disk_read_failed", "node2"),
        ("disk_read_maybe_cached", "node3"),
        ("disk_read_heterogeneous", "cluster"),
    }


# --------------------------------------------------------------------------
# Health snapshots and deltas
# --------------------------------------------------------------------------


def test_health_of_fixture() -> None:
    snap = nf.health_of(h100_facts())
    assert snap.available and snap.uptime_s == 12345.67 and snap.kernel_log_available
    gpu1 = snap.gpu["0000:3a:00.0"]
    assert gpu1["pcie_replay"] == 37 and gpu1["remapped_correctable"] == 1
    assert gpu1["ecc.volatile.dram_correctable"] == 0
    assert not any(k.startswith("ecc.aggregate") for k in gpu1)  # volatile only
    assert snap.nvlink["0000:3a:00.0"] == {
        "crc_errors": 12,
        "recovery_errors": 0,
        "replay_errors": 3,
    }
    ib = snap.ib["mlx5_0/1"]
    assert ib["link_downed"] == 1 and ib["port_xmit_wait"] == 1288231
    for traffic in (
        "port_xmit_data",
        "port_rcv_packets",
        "rx_write_requests",
        "lifespan",
    ):
        assert traffic not in ib
    assert snap.ib["mlx5_1/1"]["np_cnp_sent"] == 18273
    assert snap.ethtool["mlx5_1/1"]["rx_prio3_pause"] == 77105
    assert snap.pcie_aer["0000:3a:00.0"] == {"correctable": 7, "nonfatal": 0}
    assert snap.kernel_log == {
        "aer": {"aer_events": 2},
        "mlx5": {"mlx5_errors": 2},
        "xid/0000:3a:00.0": {"xid_63": 1, "xid_94": 1},
        "xid/other": {"xid_13": 1},
    }
    assert snap.cgroup["cpu"] == {"nr_throttled": 0, "throttled_us": 0}
    assert snap.states["ib"]["mlx5_1/1"]["rate"] == "200 Gb/sec (4X HDR)"
    assert snap.states["nvlink"]["0000:3a:00.0"] == {
        "active_links": "17",
        "total_links": "18",
    }
    assert snap.states["gpu"]["0000:3a:00.0"]["power_limit_w"] == "500.0"
    assert "speed_gts" not in snap.states["pcie"]["0000:3a:00.0"]  # idle GPUs downshift
    assert snap.states["pcie"]["0000:19:00.0"]["speed_gts"] == "32.0"


def after_run_doc() -> dict[str, Any]:
    """The H100 node one hour later, after a rough run."""
    doc = copy.deepcopy(h100_doc())
    doc["started_at"] += HOUR
    doc["collected_at"] += HOUR
    doc["kernel"]["uptime_s"] += HOUR
    hca0 = doc["hcas"][0]["ports"][0]
    hca0["counters"]["symbol_error"] = 12
    hca0["counters"]["link_downed"] = 2
    hca0["counters"]["port_xmit_wait"] = 100  # reset (port re-initialised)
    hca0["counters"]["port_rcv_data"] += 10**12  # traffic: never a finding
    hca1 = doc["hcas"][1]["ports"][0]
    hca1["state"] = "1: DOWN"
    hca1["hw_counters"]["out_of_sequence"] += 100
    hca1["hw_counters"]["np_cnp_sent"] += 1000
    hca1["ethtool"]["rx_prio3_pause"] += 500
    hca1["ethtool"]["rx_discards_phy"] = 3
    gpu1 = doc["gpus"][1]
    gpu1["pcie_replay"] = 42
    gpu1["ecc_counters"]["volatile.dram_correctable"] = 2
    doc["nvlink"][1]["links"][3]["errors"]["crc_errors"] = 20
    doc["cgroup"]["memory_events"]["oom_kill"] = 1
    doc["kernel_log"]["events"].append(
        {
            "t_s": 12445.67,
            "kind": "xid",
            "code": 79,
            "pci": "0000:3a:00.0",
            "ours": True,
        }
    )
    # GPU0 fell off the bus: gone from nvidia-smi and nvlink.
    doc["gpus"] = doc["gpus"][1:]
    doc["nvlink"] = doc["nvlink"][1:]
    return doc


def test_health_delta_and_findings() -> None:
    before = h100_facts()
    after = nf.parse_node_facts(after_run_doc(), node="node0")
    delta = nf.health_delta(before, after)
    assert delta.complete and delta.elapsed_s == HOUR and delta.node == "node0"
    found = {(i.group, i.subject, i.counter): i for i in delta.increments}
    assert found[("ib", "mlx5_0/1", "symbol_error")].delta == 12
    assert found[("ib", "mlx5_0/1", "link_downed")].delta == 1
    reset = found[("ib", "mlx5_0/1", "port_xmit_wait")]
    assert reset.reset and reset.delta == 100 and reset.before == 1288231
    assert found[("ethtool", "mlx5_1/1", "rx_discards_phy")].delta == 3
    assert found[("gpu", "0000:3a:00.0", "pcie_replay")].delta == 5
    assert found[("nvlink", "0000:3a:00.0", "crc_errors")].delta == 8
    xid = found[("kernel_log", "xid/0000:3a:00.0", "xid_79")]
    assert xid.delta == 1 and not xid.reset
    assert ("kernel_log", "xid/0000:3a:00.0", "xid_94") not in found  # already there
    assert found[("cgroup", "memory_events", "oom_kill")].delta == 1
    assert not any(i.counter == "port_rcv_data" for i in delta.increments)
    assert delta.missing == ("gpu:0000:18:00.0", "nvlink:0000:18:00.0")
    (change,) = [c for c in delta.changes if c.group == "ib"]
    assert (change.subject, change.key, change.before, change.after) == (
        "mlx5_1/1",
        "state",
        "4: ACTIVE",
        "1: DOWN",
    )
    findings = nf.delta_findings(delta)
    severities = {f.code: f.severity for f in findings}
    assert severities == {
        "delta_ib_link_downed": "error",
        "delta_kernel_log_xid_79": "error",
        "delta_cgroup_oom_kill": "error",
        "delta_state_ib_state": "error",
        "delta_missing_gpu": "error",
        "delta_missing_nvlink": "error",
        "delta_ib_symbol_error": "warn",
        "delta_ib_out_of_sequence": "warn",
        "delta_ethtool_rx_discards_phy": "warn",
        "delta_gpu_pcie_replay": "warn",
        "delta_nvlink_crc_errors": "warn",
        "delta_ib_port_xmit_wait": "info",
        "delta_ib_np_cnp_sent": "info",
        "delta_ethtool_rx_prio3_pause": "info",
        "delta_gpu_ecc_volatile_dram_correctable": "info",
    }
    messages = {f.code: f.message for f in findings}
    assert messages["delta_ib_symbol_error"] == (
        "RDMA port counter symbol_error increased by 12 during the run (1.0 h): "
        "mlx5_0/1 +12"
    )
    assert messages["delta_ib_port_xmit_wait"].startswith(
        "RDMA port counter port_xmit_wait increased by at least 100 during the run"
    )
    assert "mlx5_0/1 reset (now 100)" in messages["delta_ib_port_xmit_wait"]
    assert messages["delta_missing_gpu"].startswith("GPU 0000:18:00.0 present before")
    assert [nf.SEVERITY_RANK[f.severity] for f in findings] == sorted(
        (nf.SEVERITY_RANK[f.severity] for f in findings), reverse=True
    )


def test_health_delta_after_reboot_counts_every_new_event() -> None:
    doc = after_run_doc()
    doc["kernel"]["uptime_s"] = 600.0
    doc["kernel_log"]["events"] = [
        {"t_s": 300.0, "kind": "xid", "code": 48, "pci": "0000:3a:00.0", "ours": True},
        {"t_s": 310.0, "kind": "xid", "code": 13, "pci": "0000:c1:00", "ours": False},
    ]
    delta = nf.health_delta(h100_facts(), nf.parse_node_facts(doc))
    kernel = [i for i in delta.increments if i.group == "kernel_log"]
    assert [(i.subject, i.counter, i.delta, i.reset) for i in kernel] == [
        ("xid/0000:3a:00.0", "xid_48", 1, True),
        ("xid/other", "xid_13", 1, True),
    ]
    assert "node rebooted between snapshots (uptime decreased)" in delta.notes
    findings = {f.code: f for f in nf.delta_findings(delta)}
    assert findings["delta_kernel_log_xid_48"].severity == "error"
    assert findings["delta_kernel_log_xid_13"].severity == "info"  # not our GPU


def test_foreign_kernel_events_are_informational() -> None:
    doc = after_run_doc()
    doc["kernel_log"]["events"].append(
        {
            "t_s": 12500.0,
            "kind": "mlx5",
            "code": None,
            "pci": "0000:d1:00.0",
            "ours": False,
        }
    )
    facts = nf.parse_node_facts(doc)
    assert nf.health_of(facts).kernel_log["mlx5/other"] == {"mlx5_errors": 1}
    findings = {
        f.code: f for f in nf.delta_findings(nf.health_delta(h100_facts(), facts))
    }
    other = findings["delta_kernel_log_mlx5_errors"]
    assert other.severity == "info" and other.subject == "mlx5/other"
    static = by_code(nf.fact_findings({"node0": facts}))
    assert static["kernel_log_mlx5"].data["count"] == 2  # only this node's HCAs


def test_health_delta_with_unreadable_kernel_log() -> None:
    doc = h100_doc()
    doc["kernel_log"] = {
        "available": False,
        "error": "denied",
        "lines": 0,
        "events": [],
        "samples": [],
    }
    delta = nf.health_delta(h100_facts(), nf.parse_node_facts(doc))
    assert delta.complete and not delta.increments
    assert delta.notes == (
        "kernel log unreadable in a snapshot: Xid/mlx5/AER not compared",
    )


def test_health_delta_incomplete() -> None:
    delta = nf.health_delta(h100_facts(), nf.unavailable_facts("node0", "timeout"))
    assert delta.complete is False and delta.notes == ("after snapshot unavailable",)
    (finding,) = nf.delta_findings(delta)
    assert finding.code == "delta_incomplete" and finding.severity == "info"
    snapshot = nf.health_of(h100_facts())
    assert nf.health_delta(snapshot, snapshot).increments == ()


@pytest.mark.parametrize(
    ("group", "counter", "severity"),
    [
        ("ib", "link_downed", "error"),
        ("ib", "symbol_error", "warn"),
        ("ib", "hw.out_of_sequence", "warn"),
        ("ib", "port_xmit_wait", "info"),
        ("ib", "VL15_dropped", "info"),
        ("ethtool", "tx_pause_storm_error_events", "error"),
        ("ethtool", "rx_prio3_buf_discard", "warn"),
        ("ethtool", "rx_prio3_pause", "info"),
        ("ethtool", "rx_corrected_bits_phy", "info"),
        ("gpu", "ecc.volatile.sram_uncorrectable_parity", "error"),
        ("gpu", "ecc.volatile.dram_correctable", "info"),
        ("gpu", "remapped_uncorrectable", "error"),
        ("gpu", "remapped_correctable", "warn"),
        ("gpu", "throttle_us.sw_power_cap", "info"),
        ("gpu", "throttle_us.hw_thermal_slowdown", "warn"),
        ("nvlink", "link_recovery_failed_events", "error"),
        ("nvlink", "replay_errors", "warn"),
        ("pcie_aer", "correctable", "warn"),
        ("pcie_aer", "fatal", "error"),
        ("kernel_log", "xid_79", "error"),
        ("kernel_log", "xid_43", "warn"),
        ("kernel_log", "mlx5_errors", "warn"),
        ("cgroup", "oom_kill", "error"),
        ("cgroup", "throttled_us", "warn"),
        ("cgroup", "high", "info"),
    ],
)
def test_counter_severity(group: str, counter: str, severity: str) -> None:
    assert nf.counter_severity(group, counter) == severity


@pytest.mark.parametrize(
    ("name", "health"),
    [
        ("symbol_error", True),
        ("np_ecn_marked_roce_packets", True),
        ("rx_icrc_encapsulated", True),
        ("port_xmit_wait", True),
        ("port_xmit_data", False),
        ("unicast_rcv_packets", False),
        ("rx_write_requests", False),
        ("some_new_bytes", False),
        ("some_new_thing", True),
    ],
)
def test_is_health_counter(name: str, health: bool) -> None:
    assert nf.is_health_counter(name) is health


# --------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------


def test_clock_offsets_pick_the_min_rtt_sample() -> None:
    samples = {
        "node0": [(10.0, 10.0, 10.0)],
        "node1": [
            (100.0, 100.62, 100.06),
            (101.0, 101.52, 101.01),
            (102.0, 99.0, 102.4),
        ],
        "node2": [(5.0, 1.0, 4.0)],  # local clock stepped back: discarded
        "node3": [],
    }
    offsets = nf.clock_offsets(samples)
    assert set(offsets) == {"node0", "node1"}
    assert offsets["node0"].offset_s == 0.0 and offsets["node0"].uncertainty_s == 0.0
    node1 = offsets["node1"]
    assert node1.offset_s == pytest.approx(0.515)
    assert node1.uncertainty_s == pytest.approx(0.005) and node1.rtt_s == pytest.approx(
        0.01
    )
    assert node1.samples == 3
    findings = nf.clock_findings(offsets, nodes=["node0", "node1", "node2", "node3"])
    codes = {(f.code, f.node, f.severity) for f in findings}
    assert codes == {
        ("clock_offset", "node1", "info"),
        ("clock_offset_unmeasured", "cluster", "info"),
    }
    far = {
        "node1": nf.ClockOffset(
            node="node1", offset_s=-2.5, uncertainty_s=0.01, rtt_s=0.02, samples=5
        )
    }
    (finding,) = nf.clock_findings(far)
    assert finding.severity == "warn" and "2.500 s behind node0" in finding.message


# --------------------------------------------------------------------------
# Knowledge-base signature
# --------------------------------------------------------------------------


def test_facts_signature() -> None:
    facts = {"node0": h100_facts(), "node1": nf.unavailable_facts("node1", "x")}
    software = nf.parse_software(
        software_doc(), image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
    )
    sig = nf.facts_signature(facts, software)
    assert sig["nodes"] == 2 and sig["nodes_with_facts"] == 1
    assert sig["driver"] == ["525.147.05"] and sig["driver_flavor"] == ["proprietary"]
    assert sig["kernel"] == ["5.15.0-122-generic"]
    assert sig["gpu_products"] == ["NVIDIA H100 80GB HBM3"]
    assert sig["vbios"] == ["96.00.74.00.0D"] and sig["hca_firmware"] == ["28.39.2048"]
    assert sig["hca_link_layers"] == ["Ethernet", "InfiniBand"]
    assert sig["hca_rates"] == ["200 Gb/sec (4X HDR)", "400 Gb/sec (4X NDR)"]
    assert sig["gpu_pcie_max"] == ["Gen5 x16"] and sig["gpu_pcie_degraded"] == 1
    assert sig["hca_pcie"] == ["Gen5 x16"]
    assert sig["gpu_power_limit_w"] == ["500", "700"] and sig[
        "gpu_power_default_w"
    ] == ["700"]
    assert sig["nvlink_active_links"] == ["17", "18"]
    assert sig["iommu_passthrough"] == ["False"] and sig["thp"] == ["always"]
    soft = sig["software"]
    assert soft["image"].endswith("sglang-runtime:1.5.1")
    assert soft["packages"]["sglang"] == "0.5.22" and "numpy" not in soft["packages"]
    assert soft["nccl"] == "2.28.3" and soft["ucx"] == "1.19.0"
    assert soft["os"] == "Ubuntu 24.04.3 LTS" and soft["glibc"] == "2.39"
    assert stable_hash(sig) == stable_hash(nf.facts_signature(facts, software))
    assert nf.facts_signature(facts, None)["software"] is None
    json.dumps(sig)


def test_synthetic_software_and_disk() -> None:
    stack = nf.synthetic_software("img:1")
    assert stack.image == "img:1" and nf.software_findings(stack) == []
    read = nf.synthetic_disk_read("node1", "/models/glm")
    assert read.gbs == 5.0 and read.direct and nf.disk_findings({"node1": read}) == []
    assert nf.parse_disk_read(json.loads(read.model_dump_json())) == read
