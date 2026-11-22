# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""oMLX and proxy metrics, exposed through the Dynamo worker's ``/metrics``.

oMLX has no Prometheus endpoint, so the proxy polls the bridge's ``/stats``
and renders the latest poll at scrape time. Values that originate on the Mac
are gauges because they reset when the bridge restarts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

from dynamo_omlx.proxy.kv_sync import SyncState
from prometheus_client import CollectorRegistry
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

# (metric name, help, path into the bridge /stats document)
_BRIDGE_GAUGES = (
    ("omlx_running_requests", "Requests oMLX is decoding", ("engine", "running")),
    ("omlx_waiting_requests", "Requests queued in oMLX", ("engine", "waiting")),
    (
        "omlx_ssd_cache_bytes",
        "Bytes in oMLX's SSD prefix cache",
        ("engine", "ssd_total_size"),
    ),
    (
        "omlx_ssd_cache_capacity_bytes",
        "oMLX SSD prefix cache budget",
        ("engine", "ssd_max_size"),
    ),
    (
        "omlx_ssd_cache_files",
        "Files in oMLX's SSD prefix cache",
        ("engine", "ssd_num_files"),
    ),
    (
        "omlx_bridge_active_requests",
        "Generate streams open on the bridge",
        ("bridge", "active_requests"),
    ),
    (
        "omlx_bridge_kv_live_blocks",
        "Blocks the bridge announces",
        ("kv", "live_blocks"),
    ),
    (
        "omlx_bridge_kv_seq",
        "Last KV event sequence number on the bridge",
        ("kv", "seq"),
    ),
    (
        "omlx_model_vision",
        "1 when the loaded model takes images",
        ("model", "vision"),
    ),
    (
        "omlx_bridge_image_requests",
        "Image requests the bridge prepared since it started",
        ("images", "requests"),
    ),
    (
        "omlx_bridge_images",
        "Images the bridge prepared since it started",
        ("images", "images"),
    ),
    (
        "omlx_bridge_images_decoded",
        "Images whose pixels the bridge decoded, for want of cached vision features",
        ("images", "images_decoded"),
    ),
    (
        "omlx_bridge_vision_cache_hits",
        "Images whose vision features came from oMLX's vision feature cache",
        ("images", "vision_cache_hits"),
    ),
    (
        "omlx_bridge_vision_cache_misses",
        "Images the vision tower encoded",
        ("images", "vision_cache_misses"),
    ),
    (
        "omlx_bridge_image_token_mismatches",
        "Image requests whose token count differed from the frontend's",
        ("images", "token_count_mismatches"),
    ),
)
_KV_STATES = ("live", "degraded", "disabled")
_SYNC_STATES = (
    SyncState.CONNECTING,
    SyncState.SYNCING,
    SyncState.LIVE,
    SyncState.DISABLED,
    SyncState.STOPPED,
)
_OUTCOMES = ("stop", "length", "cancelled", "error")


def _lookup(document: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    value: Any = document
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else None


class ProxyMetrics:
    """Prometheus collector over the proxy's state; register ``registry`` with Dynamo."""

    def __init__(self) -> None:
        self.bridge_stats: dict[str, Any] | None = None
        self.bridge_up = False
        self.bridge_rtt_s: float | None = None
        # Present from the start, so that rate() sees the first increment.
        self.requests: Counter[str] = Counter(dict.fromkeys(_OUTCOMES, 0))
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.images = 0
        # Filled in by the engine once the KV pipeline exists.
        self.kv_counters: Callable[[], dict[str, float]] = dict
        self.kv_sync_state: Callable[[], str | None] = lambda: None
        self.inflight: Callable[[], int] = lambda: 0
        self.registry = CollectorRegistry()
        self.registry.register(self)

    def record_request(self, outcome: str, usage: dict[str, Any] | None) -> None:
        self.requests[outcome] += 1
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.cached_tokens += int(usage.get("cached_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def cache_hit_rate(self) -> float | None:
        """Share of prompt tokens served from oMLX's prefix cache; None before any request."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else None

    def collect(self) -> Iterator[Metric]:
        up = GaugeMetricFamily(
            "omlx_bridge_up", "1 when the last bridge stats poll succeeded"
        )
        up.add_metric([], 1.0 if self.bridge_up else 0.0)
        yield up
        if self.bridge_rtt_s is not None:
            rtt = GaugeMetricFamily(
                "omlx_bridge_rtt_seconds", "Latency of the last bridge stats poll"
            )
            rtt.add_metric([], self.bridge_rtt_s)
            yield rtt
        # The last poll's values go stale while the bridge is unreachable, and
        # the model then counts as not loaded.
        stats = self.bridge_stats if self.bridge_up else None
        loaded = GaugeMetricFamily(
            "omlx_model_loaded",
            "1 when oMLX has the model loaded and the bridge answers",
        )
        loaded.add_metric([], _lookup(stats, ("engine", "loaded")) or 0.0)
        yield loaded
        for name, help_text, path in _BRIDGE_GAUGES:
            value = _lookup(stats, path)
            if value is not None:
                gauge = GaugeMetricFamily(name, help_text)
                gauge.add_metric([], value)
                yield gauge
        kv = (stats or {}).get("kv") or {}
        if kv.get("state") is not None:
            state = GaugeMetricFamily(
                "omlx_bridge_kv_state",
                "1 for the bridge's current KV state",
                labels=["state"],
            )
            for name in _KV_STATES:
                state.add_metric([name], 1.0 if kv.get("state") == name else 0.0)
            yield state
        faults = GaugeMetricFamily(
            "omlx_bridge_kv_faults",
            "KV tracking faults counted by the bridge since it started",
            labels=["kind"],
        )
        for key, value in kv.items():
            if key.startswith("faults_") and isinstance(value, (int, float)):
                faults.add_metric([key.removeprefix("faults_")], float(value))
        yield faults

        requests = CounterMetricFamily(
            "omlx_proxy_requests", "Requests finished by outcome", labels=["outcome"]
        )
        for outcome, count in self.requests.items():
            requests.add_metric([outcome], count)
        yield requests
        for name, help_text, value in (
            (
                "omlx_proxy_prompt_tokens",
                "Prompt tokens sent to oMLX",
                self.prompt_tokens,
            ),
            (
                "omlx_proxy_cached_tokens",
                "Prompt tokens oMLX served from its prefix cache",
                self.cached_tokens,
            ),
            (
                "omlx_proxy_completion_tokens",
                "Tokens generated by oMLX",
                self.completion_tokens,
            ),
            ("omlx_proxy_images", "Images sent to oMLX", self.images),
        ):
            counter = CounterMetricFamily(name, help_text)
            counter.add_metric([], value)
            yield counter
        inflight = GaugeMetricFamily(
            "omlx_proxy_inflight_requests", "Requests the proxy is streaming"
        )
        inflight.add_metric([], self.inflight())
        yield inflight

        sync_state = self.kv_sync_state()
        if sync_state is not None:
            family = GaugeMetricFamily(
                "omlx_proxy_kv_sync_state",
                "1 for the KV sync loop's state",
                labels=["state"],
            )
            for name in _SYNC_STATES:
                family.add_metric([name], 1.0 if sync_state == name else 0.0)
            yield family
        for key, value in sorted(self.kv_counters().items()):
            if key == "announced_blocks":
                gauge = GaugeMetricFamily(
                    "omlx_proxy_kv_announced_blocks",
                    "Blocks announced to the Dynamo router",
                )
                gauge.add_metric([], value)
                yield gauge
            else:
                counter = CounterMetricFamily(
                    f"omlx_proxy_kv_{key}", f"KV pipeline {key.replace('_', ' ')}"
                )
                counter.add_metric([], value)
                yield counter
