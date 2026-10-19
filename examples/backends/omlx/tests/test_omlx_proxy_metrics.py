# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("prometheus_client")

from dynamo_omlx.proxy.metrics import ProxyMetrics  # noqa: E402

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

STATS = {
    "engine": {"loaded": True, "running": 2, "waiting": 1},
    "kv": {"state": "live", "live_blocks": 3, "faults_hash_mismatch": 0},
}


def samples(metrics: ProxyMetrics) -> dict[tuple[str, tuple], float]:
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in metrics.registry.collect()
        for sample in family.samples
    }


def by_label(metrics: ProxyMetrics, name: str, label: str) -> dict[str, float]:
    return {
        dict(labels)[label]: value
        for (sample, labels), value in samples(metrics).items()
        if sample == name
    }


def test_mac_values_drop_out_while_the_bridge_is_down():
    metrics = ProxyMetrics()
    metrics.bridge_stats, metrics.bridge_up = STATS, True
    up = samples(metrics)
    assert up[("omlx_model_loaded", ())] == 1.0
    assert up[("omlx_running_requests", ())] == 2.0
    assert by_label(metrics, "omlx_bridge_kv_state", "state")["live"] == 1.0

    # A failed poll keeps the last stats; they must not pass for current.
    metrics.bridge_up = False
    down = samples(metrics)
    assert down[("omlx_bridge_up", ())] == 0.0
    assert down[("omlx_model_loaded", ())] == 0.0
    assert ("omlx_running_requests", ()) not in down
    assert not by_label(metrics, "omlx_bridge_kv_state", "state")


def test_kv_sync_state_reports_every_state():
    metrics = ProxyMetrics()
    metrics.kv_sync_state = lambda: "connecting"
    assert by_label(metrics, "omlx_proxy_kv_sync_state", "state") == {
        "connecting": 1.0,
        "syncing": 0.0,
        "live": 0.0,
        "disabled": 0.0,
        "stopped": 0.0,
    }


def test_request_outcomes_exist_before_the_first_request():
    metrics = ProxyMetrics()
    assert by_label(metrics, "omlx_proxy_requests_total", "outcome") == {
        "stop": 0.0,
        "length": 0.0,
        "cancelled": 0.0,
        "error": 0.0,
    }
    metrics.record_request("length", {"prompt_tokens": 5, "cached_tokens": 4})
    assert by_label(metrics, "omlx_proxy_requests_total", "outcome")["length"] == 1.0
