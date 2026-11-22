# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/bandwidth.py (pure bandwidth attribution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.cluster_bench.domain import bandwidth as bw
from benchmarks.cluster_bench.domain.bandwidth import (
    CounterSeries,
    PcieSeries,
    TelemetryGpu,
    TelemetryMeta,
    TelemetrySamples,
    Window,
    analyze_point_bandwidth,
    counter_rates,
    summarize,
)
from benchmarks.cluster_bench.domain.schema import (
    GpuProfile,
    Inventory,
    NodeInventory,
    PeakResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "probe"
T0 = 1_793_700_000.0
WINDOW = (T0 + 5.0, T0 + 55.0)  # 50 s inside 60 s of telemetry
SPAN32 = 1 << 32
GB = 1e9


def load_inventory(name: str) -> Inventory:
    return Inventory.model_validate_json((FIXTURES / name).read_text())


H200 = load_inventory("inventory_h200_4node.json")
RTX = load_inventory("inventory_rtx6000_4node.json")
H200_PROFILE = GpuProfile(
    name="h200",
    match=("h200",),
    arch="sm90",
    hbm_gb=141.0,
    nvlink=True,
    weights_variant="fp8",
    nominal_nvlink_gbs=450.0,
    nominal_pcie_gbs=64.0,
)
RTX_PROFILE = GpuProfile(
    name="rtx6000",
    match=("rtx pro 6000",),
    arch="sm120",
    hbm_gb=96.0,
    nvlink=False,
    weights_variant="nvfp4",
    nominal_pcie_gbs=64.0,
)


def peak(plane: str, test: str, scope: str, value: float, nic_count=None, **details):
    return PeakResult(
        plane=plane,
        test=test,
        scope=scope,
        nic_count=nic_count,
        value_gbs=value,
        tool="test",
        details=details,
    )


# Shaped like the coordinator's simulator output (app/simulate.py).
H200_PEAKS = (
    peak("ib", "gdr_send_recv", "node0<->node1", 368.0, nic_count=8),
    peak("ib", "gdr_send_recv", "node0<->node1", 94.0, nic_count=2),
    peak("ib", "gdr_send_recv", "node0<->node1", 47.0, nic_count=1),
    peak("ib", "nccl_all_reduce_inter", "cluster", 180.0),
    peak("nvlink", "p2p_ring_aggregate", "node0", 8 * 360.0),
    peak("pcie", "h2d", "node0", 54.0),
)
RTX_PEAKS = (
    peak("ib", "gdr_send_recv", "node0<->node1", 184.0, nic_count=4),
    peak("pcie", "p2p_ring_aggregate", "node0", 8 * 40.0),
    peak("pcie", "h2d", "node0", 53.0),
)


def cumulative(rate: float, times, unit: int, base_words: int) -> tuple[int, ...]:
    return tuple((base_words + int(rate * (t - T0) / unit)) * unit for t in times)


def node_samples(
    node: NodeInventory,
    *,
    ib: dict[str, tuple[float, float]] | None = None,
    pcie: dict[int, tuple[float, float]] | None = None,
    nvlink: tuple[float, float] | None = (100 * GB, 100 * GB),
    net_tx: float | None = None,
    start: float = T0,
    seconds: int = 60,
) -> TelemetrySamples:
    """Synthetic telemetry at constant rates (bytes/s), one sample per second."""
    times = [start + i for i in range(seconds + 1)]
    ib = ib or {}
    ib_series = {
        (h.name, 1): CounterSeries(
            t=tuple(times),
            tx=cumulative(ib.get(h.name, (0.0, 0.0))[0], times, 4, 1 << 42),
            rx=cumulative(ib.get(h.name, (0.0, 0.0))[1], times, 4, 1 << 41),
        )
        for h in node.active_hcas
    }
    pcie = {g.index: (22 * GB, 21 * GB) for g in node.gpus} if pcie is None else pcie
    n = len(times) - 1
    pcie_series = {
        g: PcieSeries(
            t=tuple(times[1:]),
            tx_mean=(tx,) * n,
            tx_max=(tx * 1.5,) * n,
            rx_mean=(rx,) * n,
            rx_max=(rx * 1.5,) * n,
            n=(10,) * n,
        )
        for g, (tx, rx) in pcie.items()
    }
    nvlink_series = {}
    if nvlink is not None:
        nvlink_series = {
            g.index: CounterSeries(
                t=tuple(times),
                tx=cumulative(nvlink[0], times, 1, 1 << 50),
                rx=cumulative(nvlink[1], times, 1, 1 << 50),
            )
            for g in node.gpus
        }
    net = {}
    if net_tx is not None:
        net = {
            "wg0": CounterSeries(
                t=tuple(times),
                tx=cumulative(net_tx, times, 1, 0),
                rx=cumulative(1e6, times, 1, 0),
            )
        }
    meta = TelemetryMeta(
        node=node.name,
        hostname=node.hostname,
        t=start,
        nvml=True,
        nvlink_source="nvml:TX,RX" if nvlink is not None else None,
        gpus=tuple(TelemetryGpu(g.index, g.pci_bus_id, g.name) for g in node.gpus),
        hcas=tuple((h.name, 1) for h in node.active_hcas),
        pcie_source="nvml" if pcie else None,
    )
    return TelemetrySamples(
        meta=meta, ib=ib_series, pcie=pcie_series, nvlink=nvlink_series, net=net
    )


def uniform(rate: float, hcas) -> dict[str, tuple[float, float]]:
    return {name: (rate, rate) for name in hcas}


ALL8 = [f"mlx5_{i}" for i in range(8)]


def all_nodes(inventory: Inventory, **kwargs) -> dict[str, TelemetrySamples]:
    return {n.name: node_samples(n, **kwargs) for n in inventory.nodes}


def selection(inventory: Inventory, hcas) -> dict[str, tuple[str, ...]]:
    return {n.name: tuple(hcas) for n in inventory.nodes}


# ----------------------------------------------------------------- counters


def test_counter_rates_64bit() -> None:
    base = 1 << 36  # raw words >= 2**32 -> 64-bit
    series = [(T0 + i, (base + i * 1_000_000_000) * 4) for i in range(4)]
    rates = counter_rates(series)
    assert rates.bits == 64 and rates.wraps == 0 and rates.resets == 0
    assert [iv.bytes_per_s for iv in rates.intervals] == [4e9, 4e9, 4e9]


def test_counter_rates_32bit_wrap_adds_one_span() -> None:
    raw = [SPAN32 - 3000, SPAN32 - 1000, 1000, 3000]
    rates = counter_rates([(T0 + i, w * 4) for i, w in enumerate(raw)])
    assert rates.bits == 32 and rates.wraps == 1 and rates.resets == 0
    assert [iv.bytes_per_s for iv in rates.intervals] == [8000.0, 8000.0, 8000.0]
    assert not rates.saturated


def test_counter_rates_32bit_saturation_flag() -> None:
    raw = [SPAN32 - 1, SPAN32 - 1, SPAN32 - 1]
    rates = counter_rates([(T0 + i, w * 4) for i, w in enumerate(raw)])
    assert rates.bits == 32 and rates.saturated
    assert summarize(rates).mean_gbs == 0.0


def test_counter_rates_64bit_reset_drops_interval() -> None:
    raw = [1 << 33, (1 << 33) + 1000, 500, 1500]
    rates = counter_rates([(T0 + i, w * 4) for i, w in enumerate(raw)])
    assert rates.bits == 64 and rates.resets == 1 and rates.wraps == 0
    assert [(iv.t0 - T0, iv.bytes_per_s) for iv in rates.intervals] == [
        (0.0, 4000.0),
        (2.0, 4000.0),
    ]


def test_counter_rates_implausible_32bit_wrap_is_a_reset() -> None:
    series = [(T0, 1000 * 4), (T0 + 1, 10 * 4)]
    assert counter_rates(series).wraps == 1
    guarded = counter_rates(series, max_rate_bytes_per_s=1e9)
    assert guarded.resets == 1 and guarded.intervals == ()


def test_counter_rates_forced_64bit_for_nvlink() -> None:
    series = [(T0, 5000), (T0 + 1, 100), (T0 + 2, 300)]
    rates = counter_rates(series, unit_bytes=1, counter_bits=64)
    assert rates.resets == 1 and [iv.bytes_per_s for iv in rates.intervals] == [200.0]


def test_counter_rates_clip_to_window() -> None:
    series = [(T0 + i, (1 << 40) + i * 250_000_000 * 4) for i in range(11)]  # 1 GB/s
    window = Window(T0 + 2.5, T0 + 7.5)
    rates = counter_rates(series, window)
    assert (rates.intervals[0].t0, rates.intervals[-1].t1) == (T0 + 2.5, T0 + 7.5)
    assert len(rates.intervals) == 6
    summary = summarize(rates, window)
    assert summary.mean_gbs == pytest.approx(
        1.0
    ) and summary.covered_s == pytest.approx(5.0)


def test_counter_rates_sorted_and_duplicate_timestamps() -> None:
    series = [(T0 + 2, 3000), (T0, 1000), (T0 + 1, 2000), (T0 + 1, 2000)]
    rates = counter_rates(series, unit_bytes=1, counter_bits=64)
    assert [iv.bytes_per_s for iv in rates.intervals] == [1000.0, 1000.0]


def test_bins_split_partial_intervals() -> None:
    # 1, 3, 1 GB/s over [0,1], [1,2], [2,3]; bins [0.5,1.5] and [1.5,2.5] see 2 GB/s.
    words = [0, 250_000_000, 1_000_000_000, 1_250_000_000]
    series = [(T0 + i, ((1 << 40) + w) * 4) for i, w in enumerate(words)]
    window = Window(T0 + 0.5, T0 + 2.5)
    summary = summarize(counter_rates(series, window), window)
    assert summary.mean_gbs == pytest.approx(2.0)
    assert summary.peak_gbs == pytest.approx(2.0)


def test_summarize_burst_peak_and_p99() -> None:
    rates = [1, 1, 1, 10, 1]
    intervals = [
        bw.RateInterval(T0 + i, T0 + i + 1, r * GB) for i, r in enumerate(rates)
    ]
    summary = summarize(intervals)
    assert summary.mean_gbs == pytest.approx(2.8)
    assert summary.peak_gbs == pytest.approx(10.0)
    assert summary.p99_gbs == pytest.approx(9.64)
    assert summarize([]) == bw.EMPTY_SUMMARY and not bw.EMPTY_SUMMARY.available


def test_window_validation_and_ns() -> None:
    assert Window.from_ns(2_000_000_000, 5_000_000_000).duration_s == 3.0
    assert bw.as_window((1.0, 2.0)) == Window(1.0, 2.0)
    with pytest.raises(ValueError, match="empty window"):
        Window(5.0, 5.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0] * 8, 8.0),
        ([5.0] + [0.0] * 7, 1.0),
        ([3.0, 1.0], 1.6),
        ([], None),
        ([0.0, 0.0], None),
    ],
)
def test_nic_effective_count(values, expected) -> None:
    result = bw.nic_effective_count(values)
    assert result == (pytest.approx(expected) if expected is not None else None)


def test_calibrate_ib_scale() -> None:
    assert bw.calibrate_ib_scale(100e9, 25e9) == 4.0
    with pytest.raises(ValueError):
        bw.calibrate_ib_scale(0, 1)
    with pytest.raises(ValueError):
        bw.calibrate_ib_scale(1, 0)


# ----------------------------------------------------------------- peaks


def test_ref_peak_rules_for_microbench_rows() -> None:
    """Rows shaped like adapters/microbench.py output."""
    gdr = peak("ib", "gdr_send_recv", "node2<->node3", 360.0, 8, per_node_gbs=360.0)
    four_nodes = peak("ib", "gdr_send_recv", "cluster", 720.0, 4, per_node_gbs=180.0)
    collective = peak("ib", "nccl_all_reduce_inter", "cluster", 390.0, 8)
    failed = peak("ib", "gdr_send_recv", "node2+node3", 999.0, 8, engaged=False)
    assert bw.per_device_gbs(four_nodes) == 45.0  # per node / NICs, not total
    peaks = (gdr, four_nodes, collective, failed)
    # GPU-direct beats the collective's bus bandwidth; the failed row is ignored.
    assert bw.ref_peak_gbs(peaks, "ib", "node3/mlx5_1", nic_count=8) == 45.0
    assert bw.ref_peak_gbs(peaks, "ib", "node3", devices=8, nic_count=8) == 360.0
    # A collective's bus bandwidth is not a per-NIC rate: without a GPU-direct
    # row the per-HCA reference is unknown instead of ~8x too small.
    assert bw.per_device_ref((collective,), "ib", "node0", nic_count=8) is None
    ring = peak("nvlink", "p2p_ring_aggregate", "node0", 2900.0, per_gpu_gbs=362.5)
    h2d = peak("pcie", "h2d", "node0", 54.0, per_gpu_gbs=[54.0, 53.5, None])
    assert bw.ref_peak_gbs((ring,), "nvlink", "node0/gpu4") == 362.5
    assert bw.ref_peak_gbs((h2d,), "pcie", "node0/gpu1") == 54.0  # list ignored


def test_per_device_normalisation() -> None:
    assert (
        bw.per_device_gbs(peak("ib", "gdr_send_recv", "node0<->node1", 184.0, 4))
        == 46.0
    )
    assert (
        bw.per_device_gbs(peak("ib", "nccl_all_reduce_inter", "cluster", 180.0), 8)
        == 22.5
    )
    assert bw.per_device_gbs(peak("ib", "ib_write_bw", "node0/mlx5_3", 48.0, 4)) == 48.0
    assert (
        bw.per_device_gbs(peak("ib", "x", "cluster", 99.0, 8, per_nic_gbs=45.5)) == 45.5
    )
    assert (
        bw.per_device_gbs(peak("nvlink", "p2p_ring_aggregate", "node1", 2880.0), 8)
        == 360
    )
    assert (
        bw.per_device_gbs(peak("nvlink", "p2p_ring_aggregate", "n", 900.0, gpu_count=4))
        == 225
    )
    assert bw.per_device_gbs(peak("pcie", "h2d", "node0", 54.0), 8) == 54.0


def test_ref_peak_matching_rule() -> None:
    peaks = (
        peak("ib", "ib_write_bw", "node1/mlx5_2", 49.0, 1),
        peak("ib", "gdr_send_recv", "node0<->node1", 368.0, 8),
        peak("ib", "gdr_send_recv", "node2<->node3", 344.0, 8),
        peak("ib", "gdr_send_recv", "node0<->node1", 47.0, 1),
    )
    # A device stat uses its identical-scope row as-is.
    assert bw.ref_peak_gbs(peaks, "ib", "node1/mlx5_2", nic_count=1) == 49.0
    # Otherwise per device from the stat's node, preferring the NIC count.
    assert bw.ref_peak_gbs(peaks, "ib", "node3/mlx5_0", nic_count=8) == 43.0
    assert bw.ref_peak_gbs(peaks, "ib", "node0", devices=8, nic_count=8) == 368.0
    assert bw.ref_peak_gbs(peaks, "ib", "node0", devices=1, nic_count=1) == 47.0
    assert bw.ref_peak_gbs(peaks, "ib", "cluster", devices=32, nic_count=8) == 46.0 * 32
    # Unmeasured NIC count: the best per-NIC GPU-direct value of any count
    # (the 49 GB/s row is a different device's, and not a gdr test).
    assert bw.ref_peak_gbs(peaks, "ib", "node0/mlx5_0", nic_count=4) == 47.0
    assert bw.ref_peak_gbs(peaks, "ib", "node1/mlx5_0", nic_count=4) == 47.0
    assert bw.ref_peak_gbs(peaks, "nvlink", "node0/gpu0") is None
    assert (
        bw.per_device_ref(peaks, "ib", "node2", nic_count=8, test_filter="gdr") == 43.0
    )


# ----------------------------------------------------------------- pairing


def test_gpu_hca_pairing_follows_the_nic_planner() -> None:
    node = H200.nodes[0]
    assert bw.gpu_hca_pairing(node, ALL8) == ({i: f"mlx5_{i}" for i in range(8)}, True)
    two, planned = bw.gpu_hca_pairing(node, ["mlx5_0", "mlx5_4"])
    assert planned and two == {g: ("mlx5_0" if g < 4 else "mlx5_4") for g in range(8)}
    rtx = RTX.nodes[0]
    assert bw.gpu_hca_pairing(rtx, [f"mlx5_{i}" for i in range(4)]) == (
        {g: f"mlx5_{g // 2}" for g in range(8)},
        True,
    )


def test_gpu_hca_pairing_nearest_nic_for_unplanned_selection() -> None:
    mapping, planned = bw.gpu_hca_pairing(H200.nodes[0], ["mlx5_1", "mlx5_6"])
    assert not planned
    assert mapping == {g: ("mlx5_1" if g < 4 else "mlx5_6") for g in range(8)}
    assert bw.gpu_hca_pairing(H200.nodes[0], []) == ({}, False)


# ----------------------------------------------------------------- analysis


def analyze(inventory, samples, nics, *, profile=H200_PROFILE, peaks=H200_PEAKS, **kw):
    return analyze_point_bandwidth(
        samples, WINDOW, inventory, selection(inventory, nics), peaks, profile, **kw
    )


def test_full_nic_engagement_h200() -> None:
    samples = all_nodes(H200, ib=uniform(20 * GB, ALL8))
    result = analyze(H200, samples, ALL8)

    assert result.cross_node_gbs == pytest.approx(160.0)
    assert result.nic_effective == pytest.approx(8.0)
    assert result.nic_engaged is True
    assert result.staging_suspected is False
    assert result.coverage == 1.0
    assert result.notes == ()
    assert result.pairing["node2"] == {i: f"mlx5_{i}" for i in range(8)}

    hca = result.stat("ib", "node0/mlx5_0", "tx")
    assert (hca.mean_gbs, hca.p99_gbs, hca.peak_gbs) == (20.0, 20.0, 20.0)
    assert hca.ref_peak_gbs == 46.0 and hca.nominal_gbs == 50.0
    assert hca.util_vs_peak == pytest.approx(20 / 46)
    assert result.stat("ib", "node0/mlx5_0", "rx").mean_gbs == 20.0
    node = result.stat("ib", "node0", "tx")
    assert (node.mean_gbs, node.ref_peak_gbs, node.nominal_gbs) == (160.0, 368.0, 400.0)
    cluster = result.stat("ib", "cluster", "tx")
    assert (cluster.mean_gbs, cluster.ref_peak_gbs, cluster.nominal_gbs) == (
        640.0,
        46.0 * 32,
        1600.0,
    )
    pcie = result.stat("pcie", "node3/gpu7", "tx")
    assert (pcie.mean_gbs, pcie.ref_peak_gbs, pcie.nominal_gbs) == (22.0, 54.0, 64.0)
    nvlink = result.stat("nvlink", "node1/gpu3", "rx")
    assert (nvlink.mean_gbs, nvlink.ref_peak_gbs, nvlink.nominal_gbs) == (
        100.0,
        360.0,
        450.0,
    )
    nic = result.stat("pcie_nic", "node0/gpu5", "tx")
    assert nic.method == "derived" and nic.mean_gbs == 20.0  # min(22 PCIe, 20 HCA / 1)
    assert (nic.ref_peak_gbs, nic.nominal_gbs) == (46.0, 50.0)
    assert result.stat("pcie_nic", "node0/gpu5", "rx").mean_gbs == 20.0
    planes = {s.plane for s in result.stats}
    assert planes == {"ib", "pcie", "nvlink", "pcie_nic"}
    assert all(s.method != "unavailable" for s in result.stats)


def test_nic_not_engaged_when_outside_hca_carries_traffic() -> None:
    nics = ["mlx5_0", "mlx5_4"]
    rates = {"mlx5_0": (20 * GB, 20 * GB), "mlx5_4": (20 * GB, 20 * GB)}
    rates["mlx5_1"] = (2 * GB, 2 * GB)  # the restriction leaked
    result = analyze(H200, all_nodes(H200, ib=rates), nics)
    assert result.nic_engaged is False
    assert result.nic_effective == pytest.approx(
        84**2 / (40**2 + 40**2 + 4**2), rel=1e-4
    )
    assert any("outside the NIC selection" in n and "mlx5_1" in n for n in result.notes)
    # Four GPUs share each selected NIC: pcie_nic = min(22, 20 / 4).
    assert result.stat("pcie_nic", "node0/gpu1", "tx").mean_gbs == 5.0
    assert result.stat("ib", "node0", "tx").ref_peak_gbs == 94.0  # nic_count=2 row


def test_nic_not_engaged_when_a_selected_hca_is_idle() -> None:
    nics = ["mlx5_0", "mlx5_4"]
    result = analyze(H200, all_nodes(H200, ib={"mlx5_0": (20 * GB, 20 * GB)}), nics)
    assert result.nic_engaged is False
    assert any("mlx5_4 carried no traffic" in n for n in result.notes)
    assert result.nic_effective == pytest.approx(1.0)


def test_nic_engaged_with_single_nic_restriction() -> None:
    result = analyze(
        H200, all_nodes(H200, ib={"mlx5_0": (15 * GB, 15 * GB)}), ["mlx5_0"]
    )
    assert result.nic_engaged is True
    assert result.staging_suspected is False  # 15 GB/s >= 20% of 47
    assert result.stat("ib", "node0/mlx5_0", "tx").ref_peak_gbs == 47.0


def test_staging_suspected_on_collapsed_single_nic() -> None:
    result = analyze(
        H200, all_nodes(H200, ib={"mlx5_0": (0.8 * GB, 0.8 * GB)}), ["mlx5_0"]
    )
    assert result.cross_node_gbs == pytest.approx(0.8)
    assert result.nic_engaged is True
    assert result.staging_suspected is True
    assert any("host staging suspected" in n for n in result.notes)


def test_staging_check_skipped_without_gdr_reference() -> None:
    peaks = tuple(p for p in H200_PEAKS if "gdr" not in p.test)
    result = analyze(
        H200,
        all_nodes(H200, ib={"mlx5_0": (0.8 * GB, 0.8 * GB)}),
        ["mlx5_0"],
        peaks=peaks,
    )
    assert result.staging_suspected is False
    assert any("host-staging check skipped" in n for n in result.notes)


def test_no_cross_node_traffic_leaves_verdicts_undefined() -> None:
    result = analyze(H200, all_nodes(H200, ib=uniform(0.001 * GB, ALL8)), ALL8)
    assert result.cross_node_gbs == pytest.approx(0.008)
    assert result.nic_engaged is None
    assert result.nic_effective is None
    assert result.staging_suspected is False
    assert any("no cross-node IB traffic" in n for n in result.notes)


def test_pcie_only_profile_derives_p2p_from_pcie() -> None:
    rtx4 = [f"mlx5_{i}" for i in range(4)]
    samples = all_nodes(
        RTX,
        ib={name: (12 * GB, 4 * GB) for name in rtx4},
        pcie={g: (10 * GB, 8 * GB) for g in range(8)},
        nvlink=None,
    )
    result = analyze(RTX, samples, rtx4, profile=RTX_PROFILE, peaks=RTX_PEAKS)

    assert result.pairing["node1"] == {g: f"mlx5_{g // 2}" for g in range(8)}
    nic_tx = result.stat("pcie_nic", "node1/gpu2", "tx")
    assert nic_tx.mean_gbs == 6.0  # min(10, 12 / 2 GPUs per NIC)
    assert nic_tx.ref_peak_gbs == 23.0  # 184 / 4 NICs / 2 GPUs per NIC
    assert result.stat("pcie_nic", "node1/gpu2", "rx").mean_gbs == 2.0
    p2p_tx = result.stat("pcie", "node1/gpu2/p2p", "tx")
    assert (p2p_tx.method, p2p_tx.mean_gbs, p2p_tx.ref_peak_gbs) == (
        "derived",
        4.0,
        40.0,
    )
    assert result.stat("pcie", "node1/gpu2/p2p", "rx").mean_gbs == 6.0
    assert result.stat("pcie", "node1/gpu2", "tx").ref_peak_gbs == 53.0  # h2d > ring
    nvlink = [s for s in result.stats if s.plane == "nvlink"]
    assert [(s.scope, s.method) for s in nvlink] == [
        (f"node{i}", "unavailable") for i in range(4)
    ]
    assert any("PCIe-only GPUs" in n for n in result.notes)
    assert not any("NVLink counters unavailable" in n for n in result.notes)
    assert result.nic_engaged is True and result.nic_effective == pytest.approx(4.0)


def test_missing_nvlink_counters_on_an_nvlink_profile() -> None:
    samples = all_nodes(H200, ib=uniform(20 * GB, ALL8), nvlink=None)
    result = analyze(H200, samples, ALL8)
    stat = result.stat("nvlink", "node0", "bidir")
    assert (
        stat is not None and stat.method == "unavailable" and stat.nominal_gbs == 450.0
    )
    assert any("node0: NVLink counters unavailable" in n for n in result.notes)
    assert not any(s.scope.endswith("/p2p") for s in result.stats)


def test_missing_pcie_samples_disable_pcie_nic() -> None:
    samples = all_nodes(H200, ib=uniform(20 * GB, ALL8), pcie={})
    result = analyze(H200, samples, ALL8)
    assert result.stat("pcie", "node0", "bidir").method == "unavailable"
    assert not any(s.plane == "pcie_nic" for s in result.stats)
    assert any("GPU PCIe throughput unavailable" in n for n in result.notes)
    assert result.coverage == pytest.approx(0.5)  # IB present, PCIe absent


def test_missing_node_and_late_start_reduce_coverage() -> None:
    samples = all_nodes(H200, ib=uniform(20 * GB, ALL8))
    del samples["node3"]
    result = analyze(H200, samples, ALL8)
    assert result.coverage == pytest.approx(0.75)
    assert any("node3: no telemetry" in n for n in result.notes)
    assert result.stat("ib", "node3", "bidir").method == "unavailable"
    assert result.cross_node_gbs == pytest.approx(160.0)  # mean over measured nodes
    assert any("telemetry coverage 75%" in n for n in result.notes)

    late = all_nodes(H200, ib=uniform(20 * GB, ALL8), start=T0 + 30)
    late_result = analyze(H200, late, ALL8)
    # IB samples at t = 30..55 (26, edges inclusive) and PCIe at 31..55 (25).
    assert late_result.coverage == pytest.approx((26 + 25) / 2 / 50)
    assert late_result.stat("ib", "node0/mlx5_0", "tx").mean_gbs == 20.0


def test_window_tuple_and_object_agree_and_ib_scale_applies() -> None:
    samples = all_nodes(H200, ib=uniform(20 * GB, ALL8))
    as_tuple = analyze(H200, samples, ALL8)
    as_window = analyze_point_bandwidth(
        samples, Window(*WINDOW), H200, selection(H200, ALL8), H200_PEAKS, H200_PROFILE
    )
    assert as_tuple == as_window
    scaled = analyze(H200, samples, ALL8, ib_scale=0.5)
    assert scaled.cross_node_gbs == pytest.approx(80.0)
    assert any("calibration factor 0.500" in n for n in scaled.notes)


def test_pairing_override_is_used_for_pcie_nic() -> None:
    samples = all_nodes(H200, ib={"mlx5_7": (20 * GB, 20 * GB)})
    override = {n.name: {g: "mlx5_7" for g in range(8)} for n in H200.nodes}
    result = analyze(H200, samples, ALL8, pairing=override)
    assert result.pairing["node0"][0] == "mlx5_7"
    assert result.stat("pcie_nic", "node0/gpu0", "tx").mean_gbs == 2.5  # 20 GB/s / 8


def test_control_interface_leak_is_noted() -> None:
    samples = all_nodes(H200, ib=uniform(20 * GB, ALL8), net_tx=0.2 * GB)
    result = analyze(H200, samples, ALL8)
    assert any("control interface wg0 carried 0.200 GB/s" in n for n in result.notes)
    quiet = analyze(H200, all_nodes(H200, ib=uniform(20 * GB, ALL8), net_tx=1e6), ALL8)
    assert not any("control interface" in n for n in quiet.notes)


def test_wrapped_32bit_counters_are_noted_and_corrected() -> None:
    node = H200.nodes[0]
    samples = node_samples(node, ib=uniform(20 * GB, ALL8))
    times = samples.ib[("mlx5_0", 1)].t
    words = [(SPAN32 - 1_000_000 + int(0.25 * GB * (t - T0))) % SPAN32 for t in times]
    ib = dict(samples.ib)
    ib[("mlx5_0", 1)] = CounterSeries(
        t=times, tx=tuple(w * 4 for w in words), rx=tuple(w * 4 for w in words)
    )
    wrapped = TelemetrySamples(
        meta=samples.meta, ib=ib, pcie=samples.pcie, nvlink=samples.nvlink
    )
    inventory = Inventory(nodes=(node,))
    result = analyze_point_bandwidth(
        {"node0": wrapped},
        WINDOW,
        inventory,
        {"node0": tuple(ALL8)},
        H200_PEAKS,
        H200_PROFILE,
    )
    assert result.stat("ib", "node0/mlx5_0", "tx").mean_gbs == pytest.approx(1.0)
    assert any("32-bit IB counters wrapped" in n for n in result.notes)
