# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DockerHost: removing containers survives a kill that has not finished."""

from __future__ import annotations

import pytest

from benchmarks.cluster_bench.adapters.docker import REMOVE_ATTEMPTS, DockerHost
from benchmarks.cluster_bench.adapters.executors import RecordingExecutor
from benchmarks.cluster_bench.ports import CommandError, CommandResult

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

LABELS = {"cb.exp": "e1"}
NO_EXIT_EVENT = (
    'Error response from daemon: cannot remove container "c1": could not kill '
    "container: tried to kill container, but did not receive an exit event"
)


def docker_host(rm_failures: int, listings: list[list[str]]):
    """A node whose ``docker rm -f`` fails ``rm_failures`` times.

    ``docker ps`` answers with ``listings`` in turn (the last one repeats).
    """
    calls = {"ps": 0, "rm": 0}

    def responder(node: str, command: str) -> CommandResult | None:
        if command.startswith("docker ps"):
            names = listings[min(calls["ps"], len(listings) - 1)]
            calls["ps"] += 1
            return CommandResult(node, command, 0, "".join(f"{n}\n" for n in names), "")
        if command.startswith("docker rm -f"):
            calls["rm"] += 1
            if calls["rm"] <= rm_failures:
                return CommandResult(node, command, 1, "", NO_EXIT_EVENT)
        return None

    rec = RecordingExecutor(("node0",), responder=responder)
    sleeps: list[float] = []
    return DockerHost(rec, sleep=sleeps.append), rec, sleeps


def removes(rec: RecordingExecutor) -> list[str]:
    return [c.command for c in rec.commands if c.command.startswith("docker rm -f")]


def test_remove_retries_a_kill_that_has_not_finished():
    host, rec, sleeps = docker_host(2, [["c1", "c2"]])
    assert host.remove("node0", LABELS) == ["c1", "c2"]
    assert len(removes(rec)) == 3 and len(sleeps) == 2


def test_remove_retries_only_what_is_left_and_stops_when_it_is_gone():
    host, rec, sleeps = docker_host(1, [["c1", "c2"], ["c2"]])
    assert host.remove("node0", LABELS) == ["c1", "c2"]
    assert removes(rec) == [
        "docker rm -f c1 c2 >/dev/null",
        "docker rm -f c2 >/dev/null",
    ]
    gone, rec, sleeps = docker_host(1, [["c1"], []])
    assert gone.remove("node0", LABELS) == ["c1"]
    assert len(removes(rec)) == 1 and sleeps == []


def test_remove_gives_up_after_the_last_attempt():
    host, rec, sleeps = docker_host(REMOVE_ATTEMPTS, [["c1"]])
    with pytest.raises(CommandError, match="did not receive an exit event"):
        host.remove("node0", LABELS)
    assert len(removes(rec)) == REMOVE_ATTEMPTS
    assert len(sleeps) == REMOVE_ATTEMPTS - 1
