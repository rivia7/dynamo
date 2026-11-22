# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run context: everything a phase needs, wired once by the CLI."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.lium_cluster_bench.adapters.docker import DockerHost
from benchmarks.lium_cluster_bench.adapters.executors import Redactor
from benchmarks.lium_cluster_bench.app.store import EventStore
from benchmarks.lium_cluster_bench.domain.schema import (
    DeploymentProfile,
    EventType,
    GpuProfile,
    Inventory,
    ModelProfile,
    RunConfig,
)
from benchmarks.lium_cluster_bench.fsutil import atomic_write_json
from benchmarks.lium_cluster_bench.ports import Clock, Cluster, Engine, NodeAddress


class SystemClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class Hardware:
    """What preflight + download established about the cluster and the model."""

    inventory: Inventory
    gpu: GpuProfile
    model: ModelProfile
    deployment: DeploymentProfile
    model_dir: str


@dataclass
class Secrets:
    """Held in memory only; handed to remote processes through stdin."""

    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> Secrets:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return cls(hf_token=token or None)


@dataclass
class RunContext:
    config: RunConfig
    exp_dir: Path  # on node0; identical path layout on every node
    store: EventStore
    cluster: Cluster
    clock: Clock
    redactor: Redactor
    nodes: list[NodeAddress]
    engine: Engine
    secrets: Secrets = field(default_factory=Secrets)
    inventory: Inventory | None = None
    gpu: GpuProfile | None = None
    model: ModelProfile | None = None
    deployment: DeploymentProfile | None = None
    model_dir: str | None = None
    simulator: Any = None  # app.simulate.Simulator in --dry-run, else None
    _status: dict[str, Any] = field(default_factory=dict)
    _status_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def docker(self) -> DockerHost:
        return DockerHost(self.cluster)

    @property
    def node0(self) -> str:
        return self.nodes[0].name

    @property
    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    @property
    def python(self) -> str:
        return f"{self.config.workdir}/venv/bin/python"

    @property
    def aiperf_bin(self) -> str:
        return f"{self.config.workdir}/venv/bin/aiperf"

    @property
    def src_dir(self) -> str:
        return f"{self.config.workdir}/src"

    @property
    def hf_home(self) -> str:
        return f"{self.config.workdir}/hf-cache"

    def require_hardware(self) -> Hardware:
        if not (
            self.inventory
            and self.gpu
            and self.model
            and self.deployment
            and self.model_dir
        ):
            raise RuntimeError("preflight has not completed: hardware/profiles unknown")
        return Hardware(
            inventory=self.inventory,
            gpu=self.gpu,
            model=self.model,
            deployment=self.deployment,
            model_dir=self.model_dir,
        )

    def status(self, **fields: Any) -> None:
        """Merge ``fields`` into status.json (polled by the laptop controller).

        Called from the main thread and the heartbeat thread, hence the lock.
        """
        with self._status_lock:
            self._status.update(fields)
            self._status["exp_id"] = self.config.exp_id
            self._status["pid"] = os.getpid()
            self._status["heartbeat_ts"] = self.clock.now()
            atomic_write_json(self.exp_dir / "status.json", dict(self._status))

    def note(self, text: str, **data: Any) -> None:
        self.store.append(EventType.NOTE, {"text": self.redactor(text), **data})

    def save_json(self, rel: str, obj: Any) -> Path:
        path = self.exp_dir / rel
        atomic_write_json(path, obj)
        return path

    def load_json(self, rel: str) -> Any:
        with open(self.exp_dir / rel, "r", encoding="utf-8") as fh:
            return json.load(fh)
