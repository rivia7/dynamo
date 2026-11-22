# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark command rendering, orchestration and result parsing."""

from __future__ import annotations

import json
import re
import shlex
import statistics
from pathlib import Path

import pytest

from benchmarks.cluster_bench.adapters.docker import LABEL_ROLE, DockerHost
from benchmarks.cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.cluster_bench.adapters.microbench import (
    DEFAULT_PORT_BASE,
    GATHER_SCRIPT,
    MICROBENCH_ROLE,
    NCCL_SCRIPT,
    P2P_SCRIPT,
    PORT_SPAN,
    MicrobenchRunner,
    build_microbench_docker,
    build_nccl_collective_args,
    build_nccl_pairs_args,
    build_p2p_command,
    build_torchrun_command,
    infer_nvlink,
    nccl_ib_hca_value,
    output_path,
    parse_nccl,
    parse_nccl_net_log,
    parse_p2p,
    rendezvous_port,
)
from benchmarks.cluster_bench.domain.schema import (
    GpuInfo,
    HcaInfo,
    HcaPort,
    Inventory,
    NodeInventory,
)
from benchmarks.cluster_bench.ports import CommandError, CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "microbench"
NODES = ("node0", "node1", "node2", "node3")
IPS = {node: f"10.42.0.{i + 1}" for i, node in enumerate(NODES)}
WORKDIR = "/root/cluster_bench"
IMAGE = "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1"
OUT_DIR = f"{WORKDIR}/experiments/exp1/peaks"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
def recorder() -> RecordingExecutor:
    return RecordingExecutor(node_names=NODES)


def make_runner(rec: RecordingExecutor, tmp_path: Path, **kwargs) -> MicrobenchRunner:
    options = {"nvlink": {node: True for node in NODES}, "local_dir": tmp_path / "raw"}
    options.update(kwargs)
    return MicrobenchRunner(rec, DockerHost(rec), IMAGE, WORKDIR, IPS, "wg0", **options)


def docker_runs(rec: RecordingExecutor, node: str) -> list[list[str]]:
    return [
        shlex.split(c) for c in rec.commands_for(node) if c.startswith("docker run ")
    ]


def env_of(argv: list[str]) -> dict[str, str]:
    env = {}
    for flag, value in zip(argv, argv[1:]):
        if flag == "-e":
            key, _, val = value.partition("=")
            env[key] = val
    return env


def flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --------------------------------------------------------------------------
# Command builders
# --------------------------------------------------------------------------


def test_rendezvous_port_is_deterministic_and_in_range():
    base = DEFAULT_PORT_BASE
    ports = {rendezvous_port(base, f"job-{i}") for i in range(50)}
    assert all(base <= p < base + PORT_SPAN for p in ports)
    assert len(ports) > 10  # keys spread over the range
    assert rendezvous_port(base, "gather|abc") == rendezvous_port(base, "gather|abc")
    with pytest.raises(ValueError):
        rendezvous_port(65500, "x", span=100)


def test_torchrun_standalone_and_static_rendezvous():
    single = build_torchrun_command(
        NCCL_SCRIPT, ["--mode", "collective"], nproc_per_node=8
    )
    assert single[:4] == ["torchrun", "--standalone", "--nnodes", "1"]
    assert single[-3:] == [NCCL_SCRIPT, "--mode", "collective"]
    assert flag_value(single, "--nproc-per-node") == "8"

    multi = build_torchrun_command(
        GATHER_SCRIPT,
        ["--chunk-mib", "256"],
        nproc_per_node=1,
        nnodes=4,
        node_rank=2,
        master_addr=IPS["node0"],
        master_port=DEFAULT_PORT_BASE + 7,
    )
    assert "--standalone" not in multi
    assert flag_value(multi, "--nnodes") == "4"
    assert flag_value(multi, "--node-rank") == "2"
    assert flag_value(multi, "--master-addr") == IPS["node0"]
    assert flag_value(multi, "--master-port") == str(DEFAULT_PORT_BASE + 7)
    assert multi.index("--nproc-per-node") < multi.index(GATHER_SCRIPT)

    with pytest.raises(ValueError):
        build_torchrun_command(NCCL_SCRIPT, [], nproc_per_node=8, nnodes=2)
    with pytest.raises(ValueError):
        build_torchrun_command(
            NCCL_SCRIPT, [], nproc_per_node=8, nnodes=2, node_rank=2, master_addr="a"
        )


def test_nccl_ib_hca_uses_exact_match_syntax():
    assert nccl_ib_hca_value(["mlx5_0", "mlx5_3"]) == "=mlx5_0,mlx5_3"
    assert nccl_ib_hca_value(("mlx5_1",)) == "=mlx5_1"
    for bad in ([], ["mlx5_0", ""], ["^mlx5_0"], ["=mlx5_0"], ["mlx5_0,mlx5_1"]):
        with pytest.raises(ValueError):
            nccl_ib_hca_value(bad)


def test_agent_argv_builders():
    p2p = build_p2p_command(out="/o/p2p.json", node="node1", size_mib=512)
    assert p2p[:2] == ["python3", P2P_SCRIPT]
    assert flag_value(p2p, "--size-mib") == "512"
    assert flag_value(p2p, "--node") == "node1"
    coll = build_nccl_collective_args(out="/o/n.json", sizes=("64M", "1G"))
    assert flag_value(coll, "--mode") == "collective"
    assert flag_value(coll, "--ops") == "all_reduce,all_gather"
    assert flag_value(coll, "--sizes") == "64M,1G"
    pairs = build_nccl_pairs_args(out="/o/g.json", size="512M", iters=7)
    assert flag_value(pairs, "--mode") == "pairs"
    assert flag_value(pairs, "--size") == "512M"
    assert flag_value(pairs, "--iters") == "7"
    assert output_path("/x/nccl.json", "intra-node1") == "/x/nccl-intra-node1.json"
    assert output_path("/x/gdr", "8nic") == "/x/gdr-8nic.json"


def test_microbench_docker_is_foreground_and_self_removing():
    argv = build_microbench_docker(
        name="cb-mb-p2p",
        image=IMAGE,
        command=["python3", P2P_SCRIPT],
        gpus=range(8),
        src_dir=f"{WORKDIR}/src",
        out_dir=OUT_DIR,
        env={"NCCL_DEBUG": "WARN"},
        labels={LABEL_ROLE: MICROBENCH_ROLE},
    )
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "-d" not in argv
    assert flag_value(argv, "--gpus") == '"device=0,1,2,3,4,5,6,7"'
    assert f"{WORKDIR}/src:/bench/src:ro" in argv
    assert f"{OUT_DIR}:{OUT_DIR}" in argv
    assert "/dev/infiniband" in argv
    assert argv[argv.index(IMAGE) + 1 :] == ["python3", P2P_SCRIPT]


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------


def test_parse_p2p_nvlink_node():
    obj = load("p2p_h100_nvlink.json")
    peaks = {p.test: p for p in parse_p2p(obj, nvlink=True, command="cmd")}
    assert set(peaks) == {"p2p_copy", "p2p_ring_aggregate", "h2d", "d2h"}
    copy = peaks["p2p_copy"]
    values = [
        v for i, row in enumerate(obj["copy_gbs"]) for j, v in enumerate(row) if i != j
    ]
    assert copy.plane == "nvlink" and copy.scope == "node1"
    assert copy.value_gbs == pytest.approx(statistics.median(values))
    assert copy.details["min_gbs"] == min(values)
    assert copy.details["max_gbs"] == max(values)
    assert copy.details["pairs_measured"] == 56 and copy.details["pairs_failed"] == 0
    assert copy.details["peer_pairs"] == 56
    assert copy.command == "cmd" and copy.tool.startswith("p2p_bench")
    ring = peaks["p2p_ring_aggregate"]
    assert ring.value_gbs == pytest.approx(3046.2)
    assert ring.details["per_gpu_gbs"] == pytest.approx(3046.2 / 8)
    assert peaks["h2d"].plane == "pcie" and peaks["d2h"].plane == "pcie"
    assert peaks["h2d"].value_gbs == pytest.approx(statistics.median(obj["h2d_gbs"]))
    assert peaks["d2h"].details["min_gbs"] == pytest.approx(55.1)
    assert infer_nvlink(obj) is True


def test_parse_p2p_pcie_node_with_failed_pair_and_staged_copies():
    obj = load("p2p_rtxpro_pcie.json")
    peaks = {
        p.test: p for p in parse_p2p(obj, nvlink=False, details={"plane_source": "x"})
    }
    copy = peaks["p2p_copy"]
    assert copy.plane == "pcie" and peaks["p2p_ring_aggregate"].plane == "pcie"
    assert copy.details["pairs_measured"] == 55
    assert copy.details["pairs_failed"] == 1
    assert copy.details["peer_pairs"] == 24  # two PCIe switches of four GPUs
    assert copy.details["slowest_pair"] == [0, 6]  # host-staged, across switches
    assert copy.details["errors"] == obj["errors"]
    assert copy.details["plane_source"] == "x"
    # The slow host-staged GPU (h2d 41.7) shows up as the minimum, not the median.
    assert peaks["h2d"].details["min_gbs"] == pytest.approx(41.7)
    assert peaks["h2d"].value_gbs > 52
    assert infer_nvlink(obj) is False


def test_parse_p2p_rejects_other_kinds():
    with pytest.raises(ValueError):
        parse_p2p(load("nccl_intra_h100.json"), nvlink=True)


def test_parse_nccl_collectives_intra_and_inter():
    intra = {p.test: p for p in parse_nccl(load("nccl_intra_h100.json"), "node1")}
    assert set(intra) == {"nccl_all_reduce_intra", "nccl_all_gather_intra"}
    ar = intra["nccl_all_reduce_intra"]
    assert ar.plane == "nvlink" and ar.scope == "node1"
    assert ar.value_gbs == pytest.approx(476.9)
    assert ar.details["best_size_bytes"] == 1 << 30
    assert len(ar.details["by_size"]) == 3
    assert ar.details["algbw_gbs"] == pytest.approx(476.9 / 1.75, rel=1e-4)
    assert ar.details["nccl"] == "2.27.7"

    pcie = parse_nccl(load("nccl_intra_h100.json"), "node0", intra_plane="pcie")
    assert {p.plane for p in pcie} == {"pcie"}

    inter = {p.test: p for p in parse_nccl(load("nccl_inter_h100.json"), "cluster")}
    assert set(inter) == {"nccl_all_reduce_inter", "nccl_all_gather_inter"}
    assert {p.plane for p in inter.values()} == {"ib"}
    assert inter["nccl_all_gather_inter"].value_gbs == pytest.approx(45.6)
    assert inter["nccl_all_reduce_inter"].details["world_size"] == 32
    assert inter["nccl_all_reduce_inter"].details["env"]["NCCL_NET"] == "IB"


def test_parse_nccl_pairs():
    obj = load("nccl_pairs_8nic.json")
    (peak,) = parse_nccl(obj, "node0<->node1", nic_count=8, nominal_gbs=400.0)
    assert peak.test == "gdr_send_recv" and peak.plane == "ib"
    assert peak.nic_count == 8 and peak.nominal_gbs == 400.0
    assert peak.value_gbs == pytest.approx(obj["pairs"]["aggregate_gbs"])
    assert len(peak.details["per_rank_gbs"]) == 16
    assert peak.details["bidirectional_gbs"] == pytest.approx(2 * peak.value_gbs)
    assert peak.details["env"]["NCCL_IB_HCA"].startswith("=")


def test_parse_nccl_net_log_reports_hcas_and_gpu_direct():
    evidence = parse_nccl_net_log(fixture_text("nccl_info_pairs_node0.log.txt"))
    assert evidence.devices == tuple(
        sorted(
            [
                "mlx5_0",
                "mlx5_3",
                "mlx5_4",
                "mlx5_5",
                "mlx5_6",
                "mlx5_9",
                "mlx5_10",
                "mlx5_11",
            ]
        )
    )
    assert evidence.gdr_connections == 32
    assert evidence.staged_connections == 0 and evidence.socket_connections == 0

    staged = parse_nccl_net_log(
        "h:1:2 [0] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [RO]; OOB eth0:10.0.0.2<0>\n"
        "h:1:3 [0] NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0\n"
        "h:1:3 [0] NCCL INFO Channel 01/0 : 8[0] -> 0[0] [receive] via NET/IB/0(0)/GDRDMA\n"
        "h:1:3 [0] NCCL INFO Channel 02/0 : 0[0] -> 8[0] [send] via NET/Socket/0\n"
    )
    assert staged.devices == ("mlx5_2",)
    assert (
        staged.gdr_connections,
        staged.staged_connections,
        staged.socket_connections,
    ) == (
        1,
        1,
        1,
    )
    assert parse_nccl_net_log("no nccl output").devices == ()


# --------------------------------------------------------------------------
# Runner orchestration (RecordingExecutor: no Docker, no GPUs)
# --------------------------------------------------------------------------


def test_runner_p2p_runs_container_and_parses_result(recorder, tmp_path):
    remote = f"{OUT_DIR}/p2p-node1.json"
    recorder.files[("node1", remote)] = fixture_text("p2p_h100_nvlink.json")
    runner = make_runner(recorder, tmp_path, gpu_counts={"node1": 8})
    peaks = runner.p2p("node1", remote)

    assert {p.test for p in peaks} == {"p2p_copy", "p2p_ring_aggregate", "h2d", "d2h"}
    assert peaks[0].details["plane_source"] == "inventory"
    (argv,) = docker_runs(recorder, "node1")
    assert "--rm" in argv and "-d" not in argv
    assert flag_value(argv, "--name") == "cb-mb-p2p"
    assert flag_value(argv, "--gpus") == '"device=0,1,2,3,4,5,6,7"'
    assert f"{OUT_DIR}:{OUT_DIR}" in argv
    assert flag_value(argv, "--label") == f"{LABEL_ROLE}={MICROBENCH_ROLE}"
    inner = argv[argv.index(IMAGE) + 1 :]
    assert inner[:2] == ["python3", P2P_SCRIPT]
    assert flag_value(inner, "--out") == remote
    prep = [c for c in recorder.commands_for("node1") if c.startswith("mkdir -p")]
    assert prep and "docker rm -f cb-mb-p2p" in prep[0]
    assert (tmp_path / "raw" / "node1" / "p2p-node1.json").exists()
    assert peaks[0].command == shlex.join(argv)


def test_runner_p2p_infers_plane_without_inventory(recorder, tmp_path):
    remote = f"{OUT_DIR}/p2p-node0.json"
    recorder.files[("node0", remote)] = fixture_text("p2p_rtxpro_pcie.json")
    runner = make_runner(recorder, tmp_path, nvlink=None)
    peaks = runner.p2p("node0", remote)
    copy = next(p for p in peaks if p.test == "p2p_copy")
    assert copy.plane == "pcie"
    assert copy.details["plane_source"] == "inferred_from_bandwidth"


def test_runner_removes_the_container_when_the_run_fails(recorder, tmp_path):
    def responder(node, command):
        if command.startswith("docker run "):
            return CommandResult(node, command, 124, "", "partial log\n[timeout]")
        return None

    recorder.responder = responder
    with pytest.raises(CommandError):
        make_runner(recorder, tmp_path).p2p("node1", f"{OUT_DIR}/p2p.json")
    assert recorder.commands_for("node1")[-1] == (
        "docker rm -f cb-mb-p2p >/dev/null 2>&1 || true"
    )
    assert "[timeout]" in (tmp_path / "raw" / "node1" / "cb-mb-p2p.log").read_text()


def test_runner_collectives_intra_then_inter(recorder, tmp_path):
    remote = f"{OUT_DIR}/nccl.json"
    for node in NODES:
        recorder.files[(node, output_path(remote, f"intra-{node}"))] = fixture_text(
            "nccl_intra_h100.json"
        )
    recorder.files[("node0", output_path(remote, "inter"))] = fixture_text(
        "nccl_inter_h100.json"
    )
    runner = make_runner(recorder, tmp_path, nvlink={n: n != "node3" for n in NODES})
    peaks = runner.collectives(list(NODES), remote)

    intra = [p for p in peaks if p.test.endswith("_intra")]
    inter = [p for p in peaks if p.test.endswith("_inter")]
    assert len(intra) == 2 * len(NODES) and len(inter) == 2
    assert {p.plane for p in intra if p.scope == "node3"} == {"pcie"}
    assert {p.plane for p in intra if p.scope == "node0"} == {"nvlink"}
    assert {p.scope for p in inter} == {"cluster"} and {p.plane for p in inter} == {
        "ib"
    }

    port = rendezvous_port(DEFAULT_PORT_BASE, "inter|" + ",".join(NODES))
    for rank, node in enumerate(NODES):
        intra_argv, inter_argv = docker_runs(recorder, node)
        inner = intra_argv[intra_argv.index(IMAGE) + 1 :]
        assert inner[:2] == ["torchrun", "--standalone"]
        assert "NCCL_NET" not in env_of(intra_argv)
        inner = inter_argv[inter_argv.index(IMAGE) + 1 :]
        assert flag_value(inner, "--nnodes") == "4"
        assert flag_value(inner, "--node-rank") == str(rank)
        assert flag_value(inner, "--master-addr") == IPS["node0"]
        assert flag_value(inner, "--master-port") == str(port)
        assert flag_value(inner, "--nproc-per-node") == "8"
        assert flag_value(inner, "--out") == output_path(remote, "inter")
        assert env_of(inter_argv)["NCCL_NET"] == "IB"
        assert env_of(inter_argv)["NCCL_SOCKET_IFNAME"] == "wg0"


def test_runner_collectives_needs_nvlink_knowledge_before_running(recorder, tmp_path):
    runner = make_runner(recorder, tmp_path, nvlink=None)
    with pytest.raises(ValueError, match="NVLink"):
        runner.collectives(["node0"], f"{OUT_DIR}/nccl.json")
    assert recorder.commands == []


def test_runner_gdr_pairs_restricts_hcas_per_node_and_checks_engagement(
    recorder, tmp_path
):
    subsets = {
        8: {
            "node0": (
                "mlx5_0",
                "mlx5_3",
                "mlx5_4",
                "mlx5_5",
                "mlx5_6",
                "mlx5_9",
                "mlx5_10",
                "mlx5_11",
            ),
            "node1": (
                "mlx5_0",
                "mlx5_1",
                "mlx5_2",
                "mlx5_3",
                "mlx5_4",
                "mlx5_5",
                "mlx5_6",
                "mlx5_7",
            ),
        },
        2: {"node0": ("mlx5_0", "mlx5_6"), "node1": ("mlx5_0", "mlx5_4")},
    }
    remote = f"{OUT_DIR}/gdr.json"
    hca_re = re.compile(r"NCCL_IB_HCA==(\S+)")

    def responder(node, command):
        if not command.startswith("docker run "):
            return None
        # Emulate NCCL: report exactly the HCAs the restriction allows.
        names = hca_re.search(command).group(1).split(",")
        using = " ".join(f"[{i}]{n}:1/IB" for i, n in enumerate(names))
        log = (
            f"{node}:1:2 [0] NCCL INFO NET/IB : Using {using} [RO]; OOB wg0:x<0>\n"
            f"{node}:1:3 [0] NCCL INFO Channel 00/1 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA\n"
        )
        return recorder_result(node, command, log)

    recorder.responder = responder
    pairs = load("nccl_pairs_8nic.json")
    for count in subsets:
        recorder.files[("node0", output_path(remote, f"{count}nic"))] = json.dumps(
            pairs
        )
    rates = {
        node: {f"mlx5_{i}": 400.0 for i in range(12)} for node in ("node0", "node1")
    }
    runner = make_runner(recorder, tmp_path, hca_rates_gbps=rates)
    peaks = runner.gdr_pairs("node0", "node1", subsets, remote)

    assert [p.nic_count for p in peaks] == [8, 2]
    assert all(p.test == "gdr_send_recv" and p.plane == "ib" for p in peaks)
    assert {p.scope for p in peaks} == {"node0<->node1"}
    assert peaks[0].nominal_gbs == pytest.approx(8 * 50.0)
    assert peaks[1].nominal_gbs == pytest.approx(2 * 50.0)
    assert all(p.details["engaged"] is True for p in peaks)
    assert peaks[1].details["nccl_net"]["node1"]["devices"] == ["mlx5_0", "mlx5_4"]

    node0_runs = docker_runs(recorder, "node0")
    node1_runs = docker_runs(recorder, "node1")
    assert env_of(node0_runs[1])["NCCL_IB_HCA"] == "=mlx5_0,mlx5_6"
    assert env_of(node1_runs[1])["NCCL_IB_HCA"] == "=mlx5_0,mlx5_4"
    assert env_of(node0_runs[0])["NCCL_DEBUG"] == "INFO"
    for rank, argv in enumerate((node0_runs[1], node1_runs[1])):
        inner = argv[argv.index(IMAGE) + 1 :]
        assert flag_value(inner, "--node-rank") == str(rank)
        assert flag_value(inner, "--nnodes") == "2"
        assert flag_value(inner, "--nproc-per-node") == "8"
        assert flag_value(inner, "--mode") == "pairs"
        assert flag_value(argv, "--name") == "cb-mb-gdr-2nic"


def test_runner_gdr_pairs_flags_unengaged_restriction(recorder, tmp_path):
    subsets = {1: {"node0": ("mlx5_0",), "node1": ("mlx5_0",)}}
    remote = f"{OUT_DIR}/gdr.json"

    def responder(node, command):
        if command.startswith("docker run "):
            # NCCL ignored the restriction and used two HCAs.
            log = "x NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [RO]; OOB\n"
            return recorder_result(node, command, log)
        return None

    recorder.responder = responder
    recorder.files[("node0", output_path(remote, "1nic"))] = fixture_text(
        "nccl_pairs_8nic.json"
    )
    (peak,) = make_runner(recorder, tmp_path).gdr_pairs(
        "node0", "node1", subsets, remote
    )
    assert peak.details["engaged"] is False
    assert peak.nominal_gbs is None  # no HCA rates known

    with pytest.raises(ValueError, match="subset"):
        make_runner(recorder, tmp_path).gdr_pairs(
            "node0",
            "node1",
            {2: {"node0": ("mlx5_0",), "node1": ("mlx5_0", "mlx5_1")}},
            remote,
        )


def inventory_node(name: str, index: int, nvlink: bool) -> NodeInventory:
    gpus = tuple(
        GpuInfo(
            index=i,
            name="NVIDIA H100 80GB HBM3",
            uuid=f"GPU-{name}-{i}",
            pci_bus_id=f"0000:{0x18 + i:02x}:00.0",
            memory_mib=81559,
        )
        for i in range(4)
    )
    port = HcaPort(
        port=1,
        state="4: ACTIVE",
        rate="400 Gb/sec (4X NDR)",
        rate_gbps=400.0,
        link_layer="InfiniBand",
    )
    hcas = tuple(
        HcaInfo(name=f"mlx5_{i}", pci_bus_id=f"0000:{0x1a + i:02x}:00.0", ports=(port,))
        for i in range(2)
    )
    return NodeInventory(
        name=name,
        hostname=f"h-{name}",
        address=f"10.42.0.{index + 1}",
        gpus=gpus,
        hcas=hcas,
        driver_version="580.95",
        nvlink_active=nvlink,
    )


def test_from_inventory_fills_addresses_counts_planes_and_rates(recorder, tmp_path):
    inventory = Inventory(
        nodes=(inventory_node("node0", 0, True), inventory_node("node1", 1, False))
    )
    runner = MicrobenchRunner.from_inventory(
        recorder, DockerHost(recorder), IMAGE, WORKDIR, inventory, local_dir=tmp_path
    )
    remote = f"{OUT_DIR}/nccl.json"
    recorder.files[("node1", output_path(remote, "intra-node1"))] = fixture_text(
        "nccl_intra_h100.json"
    )
    peaks = runner.collectives(["node1"], remote, inter=False)
    assert {p.plane for p in peaks} == {"pcie"}  # node1 has no NVLink
    (argv,) = docker_runs(recorder, "node1")
    assert flag_value(argv, "--gpus") == '"device=0,1,2,3"'

    gdr = f"{OUT_DIR}/gdr.json"
    recorder.files[("node0", output_path(gdr, "2nic"))] = fixture_text(
        "nccl_pairs_8nic.json"
    )
    both = ("mlx5_0", "mlx5_1")
    (peak,) = runner.gdr_pairs(
        "node0", "node1", {2: {"node0": both, "node1": both}}, gdr
    )
    assert peak.nominal_gbs == pytest.approx(2 * 400.0 / 8)
    assert peak.details["engaged"] is None  # no NCCL INFO output to check against
    node1_gdr = docker_runs(recorder, "node1")[-1]
    inner = node1_gdr[node1_gdr.index(IMAGE) + 1 :]
    assert flag_value(inner, "--master-addr") == "10.42.0.1"
    assert flag_value(inner, "--nproc-per-node") == "4"


def recorder_result(node: str, command: str, stdout: str) -> CommandResult:
    """A successful CommandResult carrying ``stdout``."""
    return CommandResult(node, command, 0, stdout, "")
