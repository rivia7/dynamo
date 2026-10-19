#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the state of the oMLX example's Kubernetes deployment.

    python3 status.py [--namespace dynamo-omlx] [--name omlx] [--context <context>]

Prints Markdown: the nodes, the Dynamo platform and monitoring releases, the
DynamoGraphDeployment with its pods, the model it serves, each Mac as its proxy
sees it, the frontend's traffic, the addresses to reach them, and findings that
need attention. It changes nothing: it runs kubectl get and top, helm list, and
curl inside the pods against their own metrics ports. Of a Mac's Secret it reads
OMLX_BRIDGE_URL only, never OMLX_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This file is <example>/.agents/skills/omlx-status/scripts/status.py.
EXAMPLE = Path(__file__).resolve().parents[4]
TIMEOUT_S = 30
GIB = 1024**3
_UNITS = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}
_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_DEFAULT_PORTS = {"system": 9090, "http": 8000}
CURL = ("curl", "-s", "-m", "5")

Metrics = dict[str, list[tuple[dict[str, str], float]]]
Obj = dict[str, Any]


class Kube:
    """Read-only kubectl and helm calls; a failed call returns None."""

    def __init__(self, context: str | None) -> None:
        self.context = context

    def run(self, tool: str, *args: str) -> str | None:
        flag = "--kube-context" if tool == "helm" else "--context"
        cmd = [tool, *((flag, self.context) if self.context else ()), *args]
        try:
            done = subprocess.run(
                cmd, capture_output=True, text=True, timeout=TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout if done.returncode == 0 else None

    def items(self, *args: str) -> list[Obj] | None:
        """The items of ``kubectl get <args> -o json``."""
        out = self.run("kubectl", "get", *args, "-o", "json")
        try:
            return None if out is None else json.loads(out).get("items", [])
        except ValueError:
            return None

    def scrape(self, pod: Obj, port_name: str) -> Metrics:
        """The pod's metrics, fetched with curl inside its main container."""
        container = main_container(pod)
        port = next(
            (
                p["containerPort"]
                for p in container.get("ports", [])
                if p.get("name") == port_name
            ),
            _DEFAULT_PORTS[port_name],
        )
        meta = pod["metadata"]
        cmd = f"exec -n {meta['namespace']} {meta['name']} -c {container['name']}"
        out = self.run(
            "kubectl", *cmd.split(), "--", *CURL, f"localhost:{port}/metrics"
        )
        return parse_metrics(out or "")

    def top(self, *args: str) -> dict[str, list[str]]:
        """``kubectl top`` columns by name; empty without metrics-server."""
        out = self.run("kubectl", "top", *args, "--no-headers") or ""
        rows = (line.split() for line in out.splitlines())
        return {cols[0]: cols[1:] for cols in rows if cols}

    def secret_keys(self, namespace: str, secret: str) -> list[str] | None:
        """The key names of a Secret; its values stay unprinted."""
        keys = "go-template={{range $k, $v := .data}}{{$k}} {{end}}"
        out = self.run("kubectl", "get", "secret", secret, "-n", namespace, "-o", keys)
        return None if out is None else out.split()

    def bridge_url(self, namespace: str, secret: str) -> str | None:
        """OMLX_BRIDGE_URL of a Mac's Secret; its other keys stay unread."""
        query = "jsonpath={.data.OMLX_BRIDGE_URL}"
        out = self.run("kubectl", "get", "secret", secret, "-n", namespace, "-o", query)
        if out is None:
            return None
        try:
            return base64.b64decode(out).decode() or "(no OMLX_BRIDGE_URL)"
        except (binascii.Error, UnicodeDecodeError):
            return "(unreadable OMLX_BRIDGE_URL)"


def parse_metrics(text: str) -> Metrics:
    """The samples of a Prometheus text exposition, by metric name."""
    metrics: Metrics = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, labels, number = match.groups()
        try:
            sample = float(number)
        except ValueError:
            continue
        metrics.setdefault(name, []).append(
            (dict(_LABEL.findall(labels or "")), sample)
        )
    return metrics


def value(metrics: Metrics, name: str, **labels: str) -> float | None:
    """The sum of the samples of ``name`` that carry ``labels``."""
    found = [v for lbl, v in metrics.get(name, []) if labels.items() <= lbl.items()]
    return sum(found) if found else None


def by_label(metrics: Metrics, name: str, label: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for labels, sample in metrics.get(name, []):
        key = labels.get(label, "")
        totals[key] = totals.get(key, 0.0) + sample
    return totals


def one_hot(metrics: Metrics, name: str) -> str:
    """The ``state`` whose sample is 1 in a one-hot state gauge."""
    states = [lbl.get("state", "") for lbl, v in metrics.get(name, []) if v == 1]
    return "/".join(states) or "-"


def quantity(text: str) -> float:
    """A Kubernetes quantity such as 1908m, 128, 13129Mi or 527901388Ki."""
    if text.endswith("m"):
        return float(text[:-1]) / 1000
    for suffix, factor in _UNITS.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * factor
    return float(text)


def parse_time(stamp: str | None) -> datetime | None:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None


def ago(stamp: str | None, now: datetime) -> str:
    then = parse_time(stamp)
    if then is None:
        return "-"
    seconds = max(int((now - then).total_seconds()), 0)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def gib(number: float) -> str:
    return f"{number / GIB:.1f} GiB"


def plural(count: float, noun: str) -> str:
    return f"{count:,.0f} {noun}{'' if count == 1 else 's'}"


def counts(totals: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:,.0f}" for k, v in sorted(totals.items()))


def table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def read_env(path: Path) -> dict[str, str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    pairs = (line.split("=", 1) for line in lines if not line.startswith("#"))
    return dict(pair for pair in pairs if len(pair) == 2)


def overlay_of(profile: dict[str, str]) -> str | None:
    """The overlay in this checkout whose profile.env the ConfigMap holds."""
    for env in sorted((EXAMPLE / "deploy" / "overlays").glob("*/profile.env")):
        if read_env(env) == profile:
            return env.parent.name
    return None


def main_container(pod: Obj) -> Obj:
    containers = pod["spec"]["containers"]
    return next((c for c in containers if c["name"] == "main"), containers[0])


def pod_status(pod: Obj) -> str:
    """The status word of ``kubectl get pods``."""
    if pod["metadata"].get("deletionTimestamp"):
        return "Terminating"
    status = pod.get("status", {})
    for container in status.get("initContainerStatuses", []):
        state = container.get("state", {})
        if "waiting" in state:
            return "Init:" + state["waiting"].get("reason", "Waiting")
        if "running" in state:
            return "Init:Running"
        if state.get("terminated", {}).get("exitCode", 0) != 0:
            return "Init:" + state["terminated"].get("reason", "Error")
    for container in status.get("containerStatuses", []):
        state = container.get("state", {})
        if "waiting" in state:
            return state["waiting"].get("reason", "Waiting")
        if "terminated" in state:
            return state["terminated"].get("reason", "Terminated")
    return status.get("phase", "Unknown")


def pod_ready(pod: Obj) -> bool:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    return bool(statuses) and all(c.get("ready") for c in statuses)


def restarts(pod: Obj, now: datetime) -> str:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    count = sum(c.get("restartCount", 0) for c in statuses)
    exits = [
        c["lastState"]["terminated"]
        for c in statuses
        if "terminated" in c.get("lastState", {})
    ]
    if not count or not exits:
        return str(count)
    last = max(exits, key=lambda t: t.get("finishedAt") or "")
    return (
        f"{count} (last {ago(last.get('finishedAt'), now)} ago: "
        f"{last.get('reason', 'exited')}, exit {last.get('exitCode', '?')})"
    )


def refs(spec: Obj) -> tuple[set[str], list[str]]:
    """The ConfigMaps a pod template uses, and the Secrets its containers need."""
    configmaps = {
        v["configMap"]["name"] for v in spec.get("volumes", []) if "configMap" in v
    }
    secrets: list[str] = []
    for kind in ("containers", "initContainers"):
        for container in spec.get(kind, []):
            for source in container.get("envFrom", []):
                if "configMapRef" in source:
                    configmaps.add(source["configMapRef"]["name"])
                secret = source.get("secretRef") or {}
                if kind == "containers" and secret and not secret.get("optional"):
                    secrets.append(secret["name"])
    return configmaps, secrets


def anthropic_api(dgd: Obj) -> bool:
    """Whether the frontend serves the Anthropic Messages API, which Claude Code calls."""
    for component in dgd["spec"].get("components", []):
        if component.get("type") != "frontend":
            continue
        container = main_container(component.get("podTemplate", {}))
        env = {e.get("name"): str(e.get("value", "")) for e in container.get("env", [])}
        if env.get("DYN_ENABLE_ANTHROPIC_API", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return True
        if "--enable-anthropic-api" in container.get("args", []):
            return True
    return False


def started_at(pod: Obj | None) -> str | None:
    """When the pod's main container last started."""
    if pod is None:
        return None
    name = main_container(pod)["name"]
    for container in pod.get("status", {}).get("containerStatuses", []):
        if container["name"] == name:
            return container.get("state", {}).get("running", {}).get("startedAt")
    return None


class Report:
    def __init__(self, kube: Kube, namespace: str, name: str) -> None:
        self.kube = kube
        self.ns = namespace
        self.name = name
        self.now = datetime.now(timezone.utc)
        self.out: list[str] = []
        self.findings: list[str] = []
        self.node_address = "<node>"

    def add(self, *lines: str) -> None:
        self.out.extend(lines)

    def cluster(self, nodes: list[Obj]) -> None:
        usage = self.kube.top("nodes")
        rows = []
        for node in nodes:
            name, status = node["metadata"]["name"], node["status"]
            ready = any(
                c["type"] == "Ready" and c["status"] == "True"
                for c in status.get("conditions", [])
            )
            if not ready:
                self.findings.append(f"Node `{name}` is not Ready.")
            address = next(
                (
                    a["address"]
                    for a in status.get("addresses", [])
                    if a["type"] == "InternalIP"
                ),
                "-",
            )
            if self.node_address == "<node>" and address != "-":
                self.node_address = address
            cpu = f"{quantity(status['capacity']['cpu']):g} cores"
            memory = gib(quantity(status["capacity"]["memory"]))
            used = usage.get(name, [])
            if len(used) >= 3:
                cpu = f"{quantity(used[0]):.1f} of {cpu}"
                memory = f"{gib(quantity(used[2]))} of {memory}"
            version = status.get("nodeInfo", {}).get("kubeletVersion", "-")
            rows.append(
                [name, "yes" if ready else "**no**", address, version, cpu, memory]
            )
        self.add("## Cluster", "")
        self.add(*table(["Node", "Ready", "Address", "Version", "CPU", "Memory"], rows))
        self.add("")
        self.releases()

    def releases(self) -> None:
        try:
            releases = json.loads(self.kube.run("helm", "list", "-A", "-o", "json"))
        except (TypeError, ValueError):
            self.add("`helm list` failed or helm is missing: releases not checked.", "")
            return
        rows = []
        for chart, what in (
            ("dynamo-platform-", "the Dynamo platform"),
            ("kube-prometheus-stack-", "monitoring"),
        ):
            found = [r for r in releases if r.get("chart", "").startswith(chart)]
            if not found:
                self.findings.append(
                    f"No `{chart[:-1]}` release: {what} is not installed."
                )
            for release in found:
                pods = self.kube.items("pods", "-n", release["namespace"]) or []
                running = [p for p in pods if p["status"].get("phase") != "Succeeded"]
                ready = sum(pod_ready(p) for p in running)
                state = release.get("status", "-")
                if ready < len(running) or state != "deployed":
                    self.findings.append(
                        f"Release `{release['name']}` is {state} with {ready} of "
                        f"{len(running)} pods ready in `{release['namespace']}`."
                    )
                rows.append(
                    [
                        release["name"],
                        release["namespace"],
                        release.get("chart", "-"),
                        state,
                        f"{ready}/{len(running)}",
                    ]
                )
        if rows:
            headers = ["Release", "Namespace", "Chart", "Status", "Pods ready"]
            self.add(*table(headers, rows), "")

    def deployment(self) -> Obj | None:
        dgds = self.kube.items("dynamographdeployments", "-n", self.ns) or []
        dgd = next((d for d in dgds if d["metadata"]["name"] == self.name), None)
        self.add(f"## Deployment `{self.ns}/{self.name}`", "")
        if dgd is None:
            self.add(f"No DynamoGraphDeployment `{self.name}` in `{self.ns}`.", "")
            self.findings.append(
                f"No DynamoGraphDeployment `{self.name}` in `{self.ns}`."
            )
            return None
        status = dgd.get("status", {})
        conditions = status.get("conditions", [])
        ready = next((c for c in conditions if c.get("type") == "Ready"), {})
        self.add(
            f"State `{status.get('state', '-')}`, Ready={ready.get('status', '-')} "
            f"({ready.get('reason') or '-'}), created "
            f"{ago(dgd['metadata'].get('creationTimestamp'), self.now)} ago.",
            "",
        )
        if ready.get("status") != "True":
            self.findings.append(
                f"The DGD is not Ready: {ready.get('message') or status.get('state')}."
            )
        return dgd

    def pods(self, pods: list[Obj], registered: set[str]) -> None:
        usage = self.kube.top("pods", "-n", self.ns)
        rows = []
        for pod in sorted(pods, key=lambda p: p["metadata"]["name"]):
            name = pod["metadata"]["name"]
            labels = pod["metadata"].get("labels", {})
            component = labels.get("nvidia.com/dynamo-component")
            role = component or (
                "job " + labels["job-name"]
                if "job-name" in labels
                else labels.get("app.kubernetes.io/name", "-")
            )
            containers = pod.get("status", {}).get("containerStatuses", [])
            ready_count = sum(bool(c.get("ready")) for c in containers)
            cpu, memory = (usage.get(name, []) + ["-", "-"])[:2]
            known = "-" if component is None else "yes" if name in registered else "no"
            rows.append(
                [
                    name,
                    role,
                    pod_status(pod),
                    f"{ready_count}/{len(pod['spec']['containers'])}",
                    restarts(pod, self.now),
                    ago(pod["metadata"].get("creationTimestamp"), self.now),
                    cpu,
                    memory,
                    known,
                ]
            )
            # A Job's finished pod is reported with its Job.
            if pod["status"].get("phase") in ("Succeeded", "Failed") or pod_ready(pod):
                continue
            note = "" if component is None or name in registered else ", unregistered"
            self.findings.append(
                f"Pod `{name}` ({role}) is not ready: {pod_status(pod)}{note}."
            )
        headers = ["Pod", "Component", "Status", "Ready", "Restarts", "Age"]
        headers += ["CPU", "Memory", "Registered"]
        self.add("### Pods", "", *table(headers, rows), "")

    def model(self, profiles: list[Obj], frontend: Metrics) -> list[str]:
        """Reports the model; returns the names the frontend serves."""
        self.add("## Model", "")
        for configmap in profiles:
            data = configmap.get("data", {})
            overlay = overlay_of(data)
            source = (
                f"overlay `{overlay}`" if overlay else "no overlay in this checkout"
            )
            self.add(
                f"Profile `{configmap['metadata']['name']}` ({source}): "
                f"`{data.get('OMLX_MODEL_REPO', '?')}` at revision "
                f"`{data.get('OMLX_MODEL_REVISION', '?')[:12]}`, served as "
                f"`{data.get('SERVED_MODEL_NAME', '?')}`; block size "
                f"{data.get('EXPECT_BLOCK_SIZE', '?')}; tool-call parser "
                f"`{data.get('DYN_TOOL_CALL_PARSER', '-')}`, reasoning parser "
                f"`{data.get('DYN_REASONING_PARSER', '-')}`.",
                "",
            )
        ready = by_label(frontend, "dynamo_frontend_model_ready", "model")
        if frontend and not ready:
            self.add("The frontend serves no model.", "")
            self.findings.append(
                "The frontend serves no model: no proxy is registered and ready."
            )
        for model, up in sorted(ready.items()):

            def gauge(name: str, model: str = model) -> str:
                number = value(frontend, f"dynamo_frontend_model_{name}", model=model)
                return "-" if number is None else f"{number:g}"

            self.add(
                f"The frontend serves `{model}` (ready {up:g}): context "
                f"{gauge('context_length')} tokens, block size "
                f"{gauge('kv_cache_block_size')}, max_num_seqs "
                f"{gauge('max_num_seqs')} per Mac.",
                "",
            )
            if up != 1:
                self.findings.append(f"The frontend reports `{model}` as not ready.")
        return sorted(ready)

    def macs(self, dgd: Obj, pods: list[Obj]) -> tuple[int, int]:
        """Reports each Mac; returns how many Macs serve and how many there are."""
        self.add("## Macs", "")
        workers = [
            c for c in dgd["spec"].get("components", []) if c.get("type") == "worker"
        ]
        serving = 0
        for component in workers:
            name = component["name"]
            _, secrets = refs(component.get("podTemplate", {}).get("spec", {}))
            urls = []
            for secret in secrets:
                url = self.kube.bridge_url(self.ns, secret)
                if url is None:
                    url = f"(Secret `{secret}` missing)"
                    self.findings.append(f"`{name}`: its Secret `{secret}` is missing.")
                urls.append(url)
            self.add(f"### {name}: {', '.join(urls) or 'no Secret'}", "")
            label = "nvidia.com/dynamo-component"
            mine = [
                p for p in pods if p["metadata"].get("labels", {}).get(label) == name
            ]
            if not mine:
                self.add("- No proxy pod.", "")
                self.findings.append(f"`{name}` has no proxy pod.")
            # Report every proxy of the Mac before counting it as serving.
            serving += any([self.mac(name, pod) for pod in mine])
        return serving, len(workers)

    def mac(self, name: str, pod: Obj) -> bool:
        """Reports one proxy; returns whether its Mac serves through it."""
        pod_name = pod["metadata"]["name"]
        if pod["status"].get("phase") != "Running":
            self.add(f"- Proxy `{pod_name}` is {pod_status(pod)}.", "")
            return False
        m = self.kube.scrape(pod, "system")
        if not m:
            self.add(f"- Proxy `{pod_name}`: its metrics did not answer.", "")
            self.findings.append(f"`{name}`: the proxy's metrics did not answer.")
            return False
        up = value(m, "omlx_bridge_up") == 1
        loaded = value(m, "omlx_model_loaded") == 1
        rtt = value(m, "omlx_bridge_rtt_seconds")
        bridge = "**down**"
        if up:
            bridge = "up" if rtt is None else f"up, stats poll {rtt * 1000:.0f} ms"
        self.add(
            f"- Proxy `{pod_name}`; bridge {bridge}; model "
            f"{'loaded' if loaded else '**not loaded**'}."
        )
        if not up:
            self.findings.append(f"`{name}`: the proxy cannot reach the Mac's bridge.")
        elif not loaded:
            self.findings.append(f"`{name}`: oMLX on the Mac has not loaded the model.")
        running = value(m, "omlx_running_requests")
        if running is not None:
            waiting = value(m, "omlx_waiting_requests") or 0
            self.add(f"- oMLX: {running:g} running, {waiting:g} waiting.")
        cache = value(m, "omlx_ssd_cache_bytes")
        if cache is not None:
            capacity = value(m, "omlx_ssd_cache_capacity_bytes")
            files = value(m, "omlx_ssd_cache_files") or 0
            budget = f" of {gib(capacity)}" if capacity else ""
            self.add(f"- SSD prefix cache: {gib(cache)}{budget} in {files:g} files.")
        bridge_kv = one_hot(m, "omlx_bridge_kv_state")
        sync = one_hot(m, "omlx_proxy_kv_sync_state")
        announced = value(m, "omlx_proxy_kv_announced_blocks") or 0
        faults = by_label(m, "omlx_bridge_kv_faults", "kind")
        faults = {k: v for k, v in faults.items() if v}
        self.add(
            f"- KV events: bridge {bridge_kv}, proxy sync {sync}; "
            f"{plural(announced, 'block')} announced to the router"
            + (f"; faults: {counts(faults)}" if faults else "")
            + "."
        )
        if up and (bridge_kv != "live" or sync != "live"):
            self.findings.append(
                f"`{name}`: KV events are not live (bridge {bridge_kv}, proxy sync "
                f"{sync}), so the router routes to this Mac by load only."
            )
        outcomes = by_label(m, "omlx_proxy_requests_total", "outcome")
        prompt = value(m, "omlx_proxy_prompt_tokens_total") or 0
        cached = value(m, "omlx_proxy_cached_tokens_total") or 0
        since = f"Since the proxy started {ago(started_at(pod), self.now)} ago"
        if sum(outcomes.values()):
            self.add(
                f"- {since}: {plural(sum(outcomes.values()), 'request')} "
                f"({counts(outcomes)}); {cached / max(prompt, 1):.1%} of "
                f"{prompt:,.0f} prompt tokens came from the cache."
            )
        else:
            self.add(f"- {since}: no requests.")
        if outcomes.get("error"):
            self.findings.append(
                f"`{name}`: {plural(outcomes['error'], 'request')} failed on the Mac "
                "since the proxy started; its log says why."
            )
        self.add("")
        return up and loaded and pod_ready(pod)

    def traffic(self, frontend: Metrics, since: str) -> None:
        self.add("## Traffic", "")
        if not frontend:
            self.add("The frontend's metrics did not answer.", "")
            self.findings.append("The frontend's metrics did not answer.")
            return
        name = "dynamo_frontend_requests_total"
        statuses = by_label(frontend, name, "status")
        endpoints = by_label(frontend, name, "endpoint")
        errors: dict[str, float] = {}
        for labels, sample in frontend.get(name, []):
            if labels.get("status") != "success" and sample:
                kind = labels.get("error_type") or labels.get("status", "?")
                errors[kind] = errors.get(kind, 0.0) + sample
        prompt = value(frontend, "dynamo_frontend_input_sequence_tokens_sum") or 0
        cached = value(frontend, "dynamo_frontend_cached_tokens_sum") or 0
        output = value(frontend, "dynamo_frontend_output_tokens_total") or 0
        inflight = value(frontend, "dynamo_frontend_inflight_requests") or 0
        queued = value(frontend, "dynamo_frontend_queued_requests") or 0
        total = sum(statuses.values())
        lines = [
            f"- Since the frontend started {since} ago: "
            f"{plural(total, 'request')} ({counts(statuses) or 'none'})"
            + (f", by endpoint {counts(endpoints)}" if endpoints else "")
            + ".",
            f"- Now {inflight:g} in flight and {queued:g} queued at the router.",
        ]
        if prompt:
            lines.append(
                f"- {cached / prompt:.1%} of {prompt:,.0f} prompt tokens came from "
                f"the Macs' caches; {output:,.0f} output tokens."
            )
        self.add(*lines, "")
        if errors:
            self.findings.append(
                f"{plural(sum(errors.values()), 'request')} failed at the frontend "
                f"since it started, by error type: {counts(errors)}."
            )

    def access(self, dgd: Obj | None) -> None:
        frontend = f"{self.name}-frontend"
        if dgd is not None:
            status = dgd.get("status", {}).get("components", {})
            for component in dgd["spec"].get("components", []):
                names = status.get(component["name"], {}).get("componentNames")
                if component.get("type") == "frontend" and names:
                    frontend = names[0]
        rows = []
        for ingress in self.kube.items("ingress", "-A") or []:
            meta, spec = ingress["metadata"], ingress["spec"]
            balancer = ingress.get("status", {}).get("loadBalancer", {})
            entry = (balancer.get("ingress") or [{}])[0]
            address = entry.get("ip") or entry.get("hostname") or self.node_address
            scheme = "https" if spec.get("tls") else "http"
            for rule in spec.get("rules", []):
                for path in rule.get("http", {}).get("paths", []):
                    backend = path.get("backend", {}).get("service", {}).get("name")
                    host = rule.get("host") or address
                    url = f"{scheme}://{host}{path.get('path', '/')}"
                    how = f"Ingress `{meta['namespace']}/{meta['name']}`"
                    if meta["namespace"] == self.ns and backend == frontend:
                        rows.append([f"OpenAI API, {self.guard(meta)}", url, how])
                    elif (backend or "").endswith("grafana"):
                        rows.append(["Grafana, user `admin`", url, how])
        forward = f"kubectl -n {self.ns} port-forward svc/{frontend} 8000:8000"
        if dgd is not None and not any(r[0].startswith("OpenAI") for r in rows):
            rows.insert(0, ["OpenAI API", "http://localhost:8000/v1", f"`{forward}`"])
            self.findings.append(
                "No Ingress routes to the frontend; reach it with a port-forward."
            )
        if dgd is not None and anthropic_api(dgd):
            at, api = next(
                (i, r) for i, r in enumerate(rows) if r[0].startswith("OpenAI")
            )
            base = api[1].removesuffix("/v1")
            guard = api[0].partition(", ")[2] or "through a port-forward"
            what = f"Anthropic Messages API for Claude Code, {guard}"
            rows.insert(at + 1, [what, base, f"`ANTHROPIC_BASE_URL`; {api[2]}"])
        for service in self.kube.items("services", "-A") or []:
            meta = service["metadata"]
            if meta["namespace"] == self.ns and meta["name"] == frontend:
                port = (service["spec"].get("ports") or [{}])[0].get("port", 8000)
                what = "Frontend in the cluster, also `/health`, `/metrics`, `/docs`"
                url = f"http://{frontend}.{self.ns}.svc.cluster.local:{port}/v1"
                rows.append([what, url, f"from outside: `{forward}`"])
            elif meta["name"] == "prometheus-operated":
                rows.append(
                    [
                        "Prometheus",
                        "http://localhost:9090",
                        f"`kubectl -n {meta['namespace']} port-forward "
                        "svc/prometheus-operated 9090:9090`",
                    ]
                )
        self.add("## Access", "", *table(["What", "URL", "How"], rows), "")

    def guard(self, ingress_meta: Obj) -> str:
        """How the API Ingress authenticates; notes what leaves it open."""
        annotations = ingress_meta.get("annotations") or {}
        middlewares = annotations.get(
            "traefik.ingress.kubernetes.io/router.middlewares"
        )
        if "omlx-auth" not in (middlewares or ""):
            self.findings.append(
                "The API Ingress runs no token check: anyone who reaches it can use "
                "the Macs."
            )
            return "no authentication"
        holders = self.kube.secret_keys(self.ns, "omlx-api-tokens") or []
        if not holders:
            self.findings.append(
                "The API Ingress requires a token, but the Secret `omlx-api-tokens` "
                "holds none, so every request through it gets 401."
            )
        return f"token required (holders: {', '.join(holders) or 'none'})"

    def jobs(self, jobs: list[Obj]) -> None:
        if not jobs:
            return
        rows = []
        for job in jobs:
            status = job.get("status", {})
            start = status.get("startTime")
            if status.get("active"):
                state = "running"
                self.findings.append(
                    f"Job `{job['metadata']['name']}` is running: other requests "
                    "compete with it for the Macs and skew its results."
                )
            elif status.get("succeeded"):
                state = "succeeded"
            elif status.get("failed"):
                state = "**failed**"
                self.findings.append(
                    f"Job `{job['metadata']['name']}` failed: "
                    f"`kubectl -n {self.ns} logs job/{job['metadata']['name']}`."
                )
            else:
                state = "pending"
            end = parse_time(status.get("completionTime")) or self.now
            took = ago(start, end) if start else "-"
            rows.append(
                [job["metadata"]["name"], state, f"{ago(start, self.now)} ago", took]
            )
        self.add("## Jobs", "", *table(["Job", "State", "Started", "Took"], rows), "")

    def housekeeping(self, configmaps: list[Obj], used: set[str]) -> None:
        names = (cm["metadata"]["name"] for cm in configmaps)
        prefixes = ("omlx-profile-", "omlx-proxy-code-", "omlx-benchmark-")
        stale = sorted(n for n in names if n.startswith(prefixes) and n not in used)
        if stale:
            self.findings.append(
                f"{plural(len(stale), 'ConfigMap')} from earlier rollouts, which "
                f"nothing uses: `kubectl -n {self.ns} delete configmap "
                f"{' '.join(stale)}`."
            )

    def events(self) -> None:
        events = self.kube.items(
            "events", "-n", self.ns, "--field-selector", "type=Warning"
        )
        if not events:
            return

        def when(event: Obj) -> str:
            return (
                event.get("lastTimestamp")
                or event.get("eventTime")
                or event["metadata"].get("creationTimestamp")
                or ""
            )

        self.add("## Warning Events", "")
        for event in sorted(events, key=when)[-5:]:
            obj = event.get("involvedObject", {})
            message = " ".join((event.get("message") or "").split())[:200]
            self.add(
                f"- {ago(when(event), self.now)} ago, {obj.get('kind')} "
                f"`{obj.get('name')}`, {event.get('reason')} "
                f"x{event.get('count') or 1}: {message}"
            )
        self.add("")


def build(kube: Kube, namespace: str, name: str) -> Report | None:
    nodes = kube.items("nodes")
    if nodes is None:
        return None
    report = Report(kube, namespace, name)
    context = kube.context or (kube.run("kubectl", "config", "current-context") or "")
    report.add(
        "# oMLX on Dynamo: Status",
        "",
        f"kubectl context `{context.strip() or '-'}`, namespace `{namespace}`, "
        f"{report.now:%Y-%m-%d %H:%M} UTC.",
        "",
    )
    summary_at = len(report.out)
    report.cluster(nodes)
    dgd = report.deployment()
    jobs = kube.items("jobs", "-n", namespace) or []
    models: list[str] = []
    serving = macs = 0
    if dgd is not None:
        pods = kube.items("pods", "-n", namespace) or []
        workers = kube.items("dynamoworkermetadatas", "-n", namespace) or []
        report.pods(pods, {w["metadata"]["name"] for w in workers})
        used: set[str] = set()
        for component in dgd["spec"].get("components", []):
            used |= refs(component.get("podTemplate", {}).get("spec", {}))[0]
        configmaps = kube.items("configmaps", "-n", namespace) or []
        profiles = [
            cm
            for cm in configmaps
            if cm["metadata"]["name"] in used
            and cm["metadata"]["name"].startswith("omlx-profile")
        ]
        label = "nvidia.com/dynamo-component-type"
        frontend_pod = next(
            (
                p
                for p in pods
                if p["metadata"].get("labels", {}).get(label) == "frontend"
                and p["status"].get("phase") == "Running"
            ),
            None,
        )
        frontend = kube.scrape(frontend_pod, "http") if frontend_pod else {}
        models = report.model(profiles, frontend)
        serving, macs = report.macs(dgd, pods)
        report.traffic(frontend, ago(started_at(frontend_pod), report.now))
        for job in jobs:
            used |= refs(job["spec"]["template"]["spec"])[0]
        report.housekeeping(configmaps, used)
    report.access(dgd)
    report.jobs(jobs)
    report.events()
    report.add("## Findings", "")
    report.add(*(f"- {f}" for f in report.findings or ["None."]))
    served = (
        f"`{'`, `'.join(models)}` is served by {serving} of {plural(macs, 'Mac')}"
        if models
        else "No model is served"
    )
    report.out[summary_at:summary_at] = [
        f"**{served}; {plural(len(report.findings), 'finding')}.**",
        "",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--namespace", default="dynamo-omlx")
    parser.add_argument("--name", default="omlx", help="the DynamoGraphDeployment")
    parser.add_argument("--context", help="kubectl context; default: the current one")
    args = parser.parse_args()
    if shutil.which("kubectl") is None:
        print("error: kubectl is not installed", file=sys.stderr)
        return 1
    report = build(Kube(args.context), args.namespace, args.name)
    if report is None:
        print("error: kubectl cannot reach the cluster", file=sys.stderr)
        return 1
    print("\n".join(report.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
