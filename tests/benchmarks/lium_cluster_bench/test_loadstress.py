# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for domain/loadstress.py (fabric and GPU stress under load)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.lium_cluster_bench.adapters.telemetry import load_samples
from benchmarks.lium_cluster_bench.domain.bandwidth import Window
from benchmarks.lium_cluster_bench.domain.loadstress import (
    CounterTable,
    GpuStressSeries,
    PointStress,
    StressSamples,
    analyze_point_stress,
    counter_delta,
    decode_reasons,
    energy_from_counter,
    energy_from_power,
    energy_per_token_j,
    window_bounds,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES = Path(__file__).parent / "fixtures" / "telemetry"
T0 = 1793700000.0
WINDOW = (T0 + 2.0, T0 + 12.0)


@pytest.fixture(scope="module")
def point() -> PointStress:
    samples = load_samples(FIXTURES / "telemetry-stress-node1.jsonl")
    return analyze_point_stress({"node1": samples.stress, "node2": None}, WINDOW)


def test_gpu_power_energy_and_throttle_from_recorded_samples(
    point: PointStress,
) -> None:
    gpu0, gpu1 = point.gpus
    # NVML energy counter: exactly 1000 W over the 10 s window.
    assert (gpu0.energy_method, gpu0.energy_j) == ("counter", 10000.0)
    # No counter on GPU1: trapezoid integral of a constant 700 W.
    assert (gpu1.energy_method, gpu1.energy_j) == ("integrated", 7000.0)
    assert gpu0.samples == 10  # ticks at 2.02 .. 11.02 s
    assert gpu0.power_mean_w == 990.0 and gpu0.power_max_w == 1000.0
    assert gpu0.sm_mhz_min == 1890.0 and gpu0.temp_max_c == 66.0
    # SW power cap on 4 of 10 samples, SW thermal on 1 (bitmask 0x24).
    assert gpu0.throttle == {"sw_power_cap": 0.4, "sw_thermal": 0.1}
    assert gpu0.throttled_fraction == 0.4
    assert gpu1.throttle == {} and gpu1.throttled_fraction == 0.0
    assert point.energy_j == 17000.0 and point.energy_gpus == 2
    assert point.power_total_mean_w == 1690.0 and point.power_mean_w == 845.0
    assert point.throttle == {"sw_power_cap": 0.2, "sw_thermal": 0.05}
    assert point.throttled_fraction == 0.2


def test_hca_congestion_rates_and_error_increments(point: PointStress) -> None:
    ib = {h.hca: h for h in point.hcas}
    assert ib["mlx5_0"].link_layer == "InfiniBand"
    assert ib["mlx5_0"].xmit_wait_ticks_per_s == pytest.approx(5e6)
    assert ib["mlx5_0"].xmit_wait_fraction == pytest.approx(0.02)  # 4 ns ticks
    assert ib["mlx5_0"].xmit_discards == 3
    # link_downed stays at 1: a constant counter is not an increment.
    assert ib["mlx5_0"].errors == {"out_of_sequence": 2, "port_xmit_discards": 3}
    roce = ib["mlx5_4"]
    assert (roce.cnp_sent_per_s, roce.cnp_handled_per_s) == (1000.0, 500.0)
    assert roce.ecn_marked_per_s == 1200.0
    assert roce.errors == {"local_ack_timeout_err": 1}
    assert point.xmit_wait_fraction_max == pytest.approx(0.02)
    assert point.error_increments == {
        "local_ack_timeout_err": 1,
        "out_of_sequence": 2,
        "port_xmit_discards": 3,
    }


def test_roce_netdev_pause_discards_without_double_counting(point: PointStress) -> None:
    (eth,) = point.eths
    assert (eth.iface, eth.hca) == ("ens5f0np0", "mlx5_4")
    assert (eth.pause_rx_per_s, eth.pause_tx_per_s) == (400.0, 200.0)
    assert eth.pause_duration_fraction == pytest.approx(0.1)  # 1e5 us/s
    # rx_prio3_cong_discard repeats rx_discards_phy: the port counter wins.
    assert eth.discards == 7 and eth.counters["rx_prio3_cong_discard"] == 7
    assert eth.ecn_per_s == 1200.0  # rx_ecn_mark, not + rx_prio3_marked
    assert point.pause_per_s_total == 600.0 and point.eth_discards_total == 7
    # The same ECN marks are on mlx5_4's hw_counters: counted once.
    assert point.ecn_per_s_total == 1200.0
    assert point.cnp_per_s_total == 1000.0


def test_notes_carry_agent_and_coverage_gaps(point: PointStress) -> None:
    assert point.nodes == ("node1",)
    assert "node1: ethtool: exit 76 (ens6f0np0)" in point.notes
    assert "node1: 2 malformed stress record(s) skipped" in point.notes
    assert "node2: no stress telemetry (agent without ibx/gpux)" in point.notes
    assert any("integrated from sampled power" in n for n in point.notes)


def test_point_stress_round_trips_as_json(point: PointStress) -> None:
    assert PointStress.model_validate_json(point.model_dump_json()) == point
    assert point.duration_s == 10.0


def test_counter_delta_resets_saturation_and_prorating() -> None:
    window = (10.0, 20.0)
    points = [(5.0, 100), (15.0, 200), (25.0, 300)]
    rate = counter_delta(points, window, prorate=True)
    assert rate is not None and rate.value == pytest.approx(100.0)
    assert rate.covered_s == 10.0 and rate.rate == pytest.approx(10.0)
    events = counter_delta(points, window, prorate=False)
    assert events is not None and events.value == 200  # whole intervals
    reset = counter_delta([(10.0, 50), (15.0, 3), (20.0, 13)], window, prorate=True)
    assert reset is not None and reset.resets == 1 and reset.value == 10
    pinned = counter_delta(
        [(10.0, 65535), (20.0, 65535)], window, prorate=False, bits=16
    )
    assert pinned is not None and pinned.saturated and pinned.value == 0
    assert counter_delta([(30.0, 1), (40.0, 2)], window, prorate=True) is None


def test_energy_falls_back_to_power_when_counter_resets() -> None:
    t = tuple(float(x) for x in range(0, 21))
    energy = tuple([1000 * x for x in range(0, 11)] + [5 * x for x in range(11, 21)])
    series = GpuStressSeries(t=t, power_w=(500.0,) * len(t), energy_mj=energy)
    stress = StressSamples(node="n", gpux={0: series})
    result = analyze_point_stress({"n": stress}, (2.0, 18.0))
    (gpu,) = result.gpus
    assert (gpu.energy_method, gpu.energy_j) == ("integrated", 8000.0)
    assert energy_from_counter(list(zip(t, energy)), (2.0, 18.0)) is None


def test_energy_needs_window_coverage() -> None:
    series = GpuStressSeries(
        t=(0.0, 1.0, 2.0), power_w=(100.0, 100.0, 100.0), energy_mj=(0, 100, 200)
    )
    result = analyze_point_stress({"n": StressSamples(gpux={0: series})}, (0.0, 10.0))
    assert result.gpus[0].energy_j is None and result.energy_j is None
    assert any("coverage" in note for note in result.notes)
    # 80 % coverage is enough; the energy is scaled to the whole window.
    joules, covered = energy_from_power([(0.0, 100.0), (8.0, 100.0)], (0.0, 10.0))
    assert (joules, covered) == (800.0, 8.0)
    wide = GpuStressSeries(t=(0.0, 8.0), power_w=(100.0, 100.0))
    scaled = analyze_point_stress({"n": StressSamples(gpux={0: wide})}, (0.0, 10.0))
    assert scaled.gpus[0].energy_j == 1000.0


def test_decode_reasons_and_window_forms() -> None:
    assert decode_reasons(0x24) == ("sw_power_cap", "sw_thermal")
    assert decode_reasons(0x1C8) == (
        "hw_slowdown",
        "hw_thermal",
        "hw_power_brake",
        "display_clocks",
    )
    assert decode_reasons(None) == ()
    assert window_bounds(Window(5.0, 9.0)) == (5.0, 9.0)
    assert window_bounds([1, 2]) == (1.0, 2.0)
    with pytest.raises(ValueError):
        window_bounds((3.0, 3.0))


def test_energy_per_token() -> None:
    point = PointStress(window_start_s=0.0, window_end_s=10.0, energy_j=4500.0)
    assert energy_per_token_j(point, 15000) == 0.3
    assert energy_per_token_j(point, 0) is None
    assert energy_per_token_j(point, None) is None
    assert energy_per_token_j(None, 15000) is None
    no_energy = PointStress(window_start_s=0.0, window_end_s=10.0)
    assert energy_per_token_j(no_energy, 15000) is None


def test_empty_inputs_give_an_empty_point() -> None:
    empty = analyze_point_stress({}, (0.0, 1.0))
    assert empty.gpus == () and empty.energy_j is None and empty.throttle == {}
    table = CounterTable(t=(0.0, 1.0), values={"np_cnp_sent": (None, 5)})
    assert table.points("np_cnp_sent") == [(1.0, 5)]
    assert table.points("missing") == []
