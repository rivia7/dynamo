# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The omlx-status skill's status.py against a fake cluster."""

import base64
import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("prometheus_client")
yaml = pytest.importorskip("yaml")

from dynamo_omlx.proxy.metrics import ProxyMetrics  # noqa: E402
from prometheus_client import generate_latest  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1]
_PATH = EXAMPLE / ".agents" / "skills" / "omlx-status" / "scripts" / "status.py"
_SPEC = importlib.util.spec_from_file_location("omlx_status", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
omlx_status = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(omlx_status)

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

NS = "dynamo-omlx"
STAMP = "2026-10-09T00:00:00Z"
MAC_URL = "http://10.0.0.9:8000"
FRONTEND_POD = "omlx-frontend-1"
MAC_POD = "omlx-mac-1-1"
MODEL = "Qwen3.8-Flash-Next"
FRONTEND_METRICS = f"""\
# TYPE dynamo_frontend_model_ready gauge
dynamo_frontend_model_ready{{model="{MODEL}"}} 1
dynamo_frontend_model_context_length{{model="{MODEL}"}} 262144
dynamo_frontend_model_kv_cache_block_size{{model="{MODEL}"}} 8192
dynamo_frontend_model_max_num_seqs{{model="{MODEL}"}} 8
dynamo_frontend_requests_total{{endpoint="chat_completions",error_type="",\
model="{MODEL}",request_type="stream",status="success"}} 10
dynamo_frontend_inflight_requests{{model="{MODEL}"}} 1
dynamo_frontend_queued_requests{{model="{MODEL}"}} 0
dynamo_frontend_input_sequence_tokens_sum{{model="{MODEL}"}} 1000
dynamo_frontend_cached_tokens_sum{{model="{MODEL}"}} 750
dynamo_frontend_output_tokens_total{{model="{MODEL}"}} 400
"""
RELEASES = [
    {
        "name": "dynamo-platform",
        "namespace": "dynamo-system",
        "chart": "dynamo-platform-1.5.1",
        "status": "deployed",
    },
    {
        "name": "prometheus",
        "namespace": "monitoring",
        "chart": "kube-prometheus-stack-91.4.1",
        "status": "deployed",
    },
]


def proxy_metrics(bridge_up: bool = True) -> str:
    """What a proxy's /metrics shows, from the proxy's own collector."""
    metrics = ProxyMetrics()
    metrics.bridge_up, metrics.bridge_rtt_s = bridge_up, 0.012
    metrics.bridge_stats = {
        "engine": {
            "loaded": True,
            "running": 2,
            "waiting": 1,
            "ssd_total_size": 3 * omlx_status.GIB,
            "ssd_max_size": 100 * omlx_status.GIB,
            "ssd_num_files": 12,
        },
        "kv": {"state": "live", "live_blocks": 3, "seq": 9},
    }
    metrics.kv_sync_state = lambda: "live"
    metrics.kv_counters = lambda: {"announced_blocks": 3, "events": 9}
    metrics.record_request("stop", {"prompt_tokens": 1000, "cached_tokens": 750})
    return generate_latest(metrics.registry).decode()


def pod(name: str, labels: dict, port: tuple[str, int] | None = None) -> dict:
    ports = [{"name": port[0], "containerPort": port[1]}] if port else []
    return {
        "metadata": {
            "name": name,
            "namespace": NS,
            "labels": labels,
            "creationTimestamp": STAMP,
        },
        "spec": {"containers": [{"name": "main", "ports": ports}]},
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "main",
                    "ready": True,
                    "restartCount": 0,
                    "state": {"running": {"startedAt": STAMP}},
                }
            ],
        },
    }


def healthy() -> dict:
    """The kubectl objects, by (kind, namespace), of a working deployment."""
    dgd = yaml.safe_load((EXAMPLE / "deploy" / "base" / "dgd.yaml").read_text())
    dgd["metadata"].update(namespace=NS, creationTimestamp=STAMP)
    dgd["status"] = {
        "state": "successful",
        "conditions": [
            {"type": "Ready", "status": "True", "reason": "all_resources_are_ready"}
        ],
        "components": {
            "Frontend": {"componentNames": ["omlx-frontend"]},
            "mac-1": {"componentNames": ["omlx-mac-1-abc"]},
        },
    }
    profile = omlx_status.read_env(
        EXAMPLE / "deploy/overlays/qwen3.8-flash-next/profile.env"
    )
    job_pod = pod("omlx-aiperf-x", {"job-name": "omlx-aiperf"})
    job_pod["status"] = {"phase": "Succeeded"}
    job = {
        "metadata": {"name": "omlx-aiperf"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"envFrom": [{"configMapRef": {"name": "omlx-benchmark-a"}}]}
                    ]
                }
            }
        },
        "status": {"succeeded": 1, "startTime": STAMP, "completionTime": STAMP},
    }

    def ingress(namespace: str, name: str, path: str, service: str) -> dict:
        backend = {"service": {"name": service, "port": {"number": 80}}}
        return {
            "metadata": {"namespace": namespace, "name": name},
            "spec": {
                "rules": [{"http": {"paths": [{"path": path, "backend": backend}]}}]
            },
            "status": {"loadBalancer": {"ingress": [{"ip": "10.0.0.5"}]}},
        }

    def service(namespace: str, name: str, port: int) -> dict:
        return {
            "metadata": {"namespace": namespace, "name": name},
            "spec": {"ports": [{"port": port}]},
        }

    api = yaml.safe_load((EXAMPLE / "deploy" / "base" / "ingress.yaml").read_text())
    api["metadata"]["namespace"] = NS
    api["status"] = {"loadBalancer": {"ingress": [{"ip": "10.0.0.5"}]}}

    def configmap(name: str, data: dict | None = None) -> dict:
        return {"metadata": {"name": name}, "data": data or {}}

    return {
        ("nodes", None): [
            {
                "metadata": {"name": "node-1"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "addresses": [{"type": "InternalIP", "address": "10.0.0.5"}],
                    "capacity": {"cpu": "128", "memory": "527901388Ki"},
                    "nodeInfo": {"kubeletVersion": "v1.34.5+k3s1"},
                },
            }
        ],
        ("pods", "dynamo-system"): [pod("operator", {})],
        ("pods", "monitoring"): [pod("prometheus", {})],
        ("dynamographdeployments", NS): [dgd],
        ("pods", NS): [
            pod(
                FRONTEND_POD,
                {
                    "nvidia.com/dynamo-component": "Frontend",
                    "nvidia.com/dynamo-component-type": "frontend",
                },
                ("http", 8000),
            ),
            pod(
                MAC_POD,
                {
                    "nvidia.com/dynamo-component": "mac-1",
                    "nvidia.com/dynamo-component-type": "worker",
                },
                ("system", 9090),
            ),
            job_pod,
        ],
        ("dynamoworkermetadatas", NS): [
            {"metadata": {"name": FRONTEND_POD}},
            {"metadata": {"name": MAC_POD}},
        ],
        # The base's unhashed names, which its DGD references.
        ("configmaps", NS): [
            configmap("omlx-profile", profile),
            configmap("omlx-proxy-code"),
            configmap("omlx-benchmark-a"),
            configmap("omlx-profile-old"),
        ],
        ("jobs", NS): [job],
        ("ingress", "*"): [
            api,
            ingress("monitoring", "grafana", "/grafana", "prometheus-grafana"),
        ],
        ("services", "*"): [
            service(NS, "omlx-frontend", 8000),
            service("monitoring", "prometheus-operated", 9090),
        ],
        ("events", NS): [],
    }


class FakeCluster(omlx_status.Kube):
    """kubectl and helm over fixed objects; records every call."""

    def __init__(self, objects: dict, metrics: dict[str, str]) -> None:
        super().__init__(None)
        self.objects = objects
        self.metrics = metrics
        self.secrets = {"omlx-mac-1": MAC_URL}
        self.token_holders: list[str] | None = ["claude-code", "codex"]
        self.calls: list[tuple[str, ...]] = []

    def run(self, tool: str, *args: str) -> str | None:
        self.calls.append((tool, *args))
        if tool == "helm":
            return json.dumps(RELEASES)
        verb, kind = args[0], args[1]
        if verb == "config":
            return "test-context\n"
        if verb == "top":
            return "node-1 1908m 1% 13129Mi 2%\n" if kind == "nodes" else ""
        if verb == "exec":
            return self.metrics.get(args[3])
        if kind == "secret" and args[2] == "omlx-api-tokens":
            holders = self.token_holders
            return None if holders is None else "".join(f"{h} " for h in holders)
        if kind == "secret":
            url = self.secrets.get(args[2])
            return None if url is None else base64.b64encode(url.encode()).decode()
        namespace = "*" if "-A" in args else None
        if "-n" in args:
            namespace = args[args.index("-n") + 1]
        items = self.objects.get((kind, namespace))
        return None if items is None else json.dumps({"items": items})


def report(
    objects: dict, metrics: dict[str, str], **fake: object
) -> tuple[str, list, FakeCluster]:
    kube = FakeCluster(objects, metrics)
    for name, value in fake.items():
        setattr(kube, name, value)
    built = omlx_status.build(kube, NS, "omlx")
    assert built is not None
    return "\n".join(built.out), built.findings, kube


def test_reports_a_healthy_deployment():
    text, findings, _ = report(
        healthy(), {FRONTEND_POD: FRONTEND_METRICS, MAC_POD: proxy_metrics()}
    )

    assert f"**`{MODEL}` is served by 1 of 1 Mac; 1 finding.**" in text
    assert "1.9 of 128 cores | 12.8 GiB of 503.4 GiB" in text
    assert (
        "| dynamo-platform | dynamo-system | dynamo-platform-1.5.1 | deployed | 1/1 |"
        in text
    )
    assert "(overlay `qwen3.8-flash-next`)" in text
    assert "context 262144 tokens, block size 8192, max_num_seqs 8 per Mac" in text
    assert f"### mac-1: {MAC_URL}" in text
    assert "bridge up, stats poll 12 ms; model loaded." in text
    assert "oMLX: 2 running, 1 waiting." in text
    assert "SSD prefix cache: 3.0 GiB of 100.0 GiB in 12 files." in text
    assert "KV events: bridge live, proxy sync live; 3 blocks announced" in text
    assert "75.0% of 1,000 prompt tokens came from the cache" in text
    tokens = "token required (holders: claude-code, codex)"
    assert f"| OpenAI API, {tokens} | http://10.0.0.5/v1 |" in text
    # deploy/base/dgd.yaml turns on the Anthropic Messages API for Claude Code.
    assert (
        f"| Anthropic Messages API for Claude Code, {tokens} "
        "| http://10.0.0.5 | `ANTHROPIC_BASE_URL`; Ingress `dynamo-omlx/omlx` |"
    ) in text
    assert "| Grafana, user `admin` | http://10.0.0.5/grafana |" in text
    assert "svc/prometheus-operated 9090:9090" in text
    # The job's ConfigMap is in use; the superseded profile is not.
    (finding,) = findings
    assert "delete configmap omlx-profile-old`" in finding


def test_findings_name_what_needs_attention():
    objects = healthy()
    (dgd,) = objects[("dynamographdeployments", NS)]
    dgd["status"]["conditions"][0].update(status="False", message="mac-1 waits")
    frontend = dgd["spec"]["components"][0]["podTemplate"]["spec"]["containers"][0]
    frontend["env"] = [
        e for e in frontend["env"] if e["name"] != "DYN_ENABLE_ANTHROPIC_API"
    ]
    mac = objects[("pods", NS)][1]
    mac["status"]["containerStatuses"][0]["ready"] = False
    objects[("dynamoworkermetadatas", NS)] = [{"metadata": {"name": FRONTEND_POD}}]
    (job,) = objects[("jobs", NS)]
    job["status"] = {"active": 1, "startTime": STAMP}
    failed = (
        f'dynamo_frontend_requests_total{{endpoint="chat_completions",'
        f'error_type="internal",model="{MODEL}",request_type="unary",'
        f'status="error"}} 2\n'
    )

    text, findings, _ = report(
        objects,
        {
            FRONTEND_POD: FRONTEND_METRICS + failed,
            MAC_POD: proxy_metrics(bridge_up=False),
        },
        token_holders=[],
    )

    assert f"**`{MODEL}` is served by 0 of 1 Mac; 7 findings.**" in text
    assert "bridge **down**; model **not loaded**." in text
    assert "Anthropic Messages API" not in text
    assert findings[:5] == [
        "The DGD is not Ready: mac-1 waits.",
        f"Pod `{MAC_POD}` (mac-1) is not ready: Running, unregistered.",
        "`mac-1`: the proxy cannot reach the Mac's bridge.",
        "2 requests failed at the frontend since it started, by error type: "
        "internal 2.",
        "1 ConfigMap from earlier rollouts, which nothing uses: "
        f"`kubectl -n {NS} delete configmap omlx-profile-old`.",
    ]
    assert findings[5] == (
        "The API Ingress requires a token, but the Secret `omlx-api-tokens` holds "
        "none, so every request through it gets 401."
    )
    assert findings[6].startswith("Job `omlx-aiperf` is running")


def test_missing_deployment_and_secret():
    objects = healthy()
    objects[("dynamographdeployments", NS)] = []
    text, findings, _ = report(objects, {})
    assert "No model is served; 1 finding." in text
    assert findings == [f"No DynamoGraphDeployment `omlx` in `{NS}`."]

    objects = healthy()
    kube = FakeCluster(objects, {FRONTEND_POD: FRONTEND_METRICS, MAC_POD: ""})
    kube.secrets = {}
    built = omlx_status.build(kube, NS, "omlx")
    assert built is not None
    assert "`mac-1`: its Secret `omlx-mac-1` is missing." in built.findings
    assert "`mac-1`: the proxy's metrics did not answer." in built.findings

    objects = healthy()
    (api, _) = objects[("ingress", "*")]
    del api["metadata"]["annotations"][
        "traefik.ingress.kubernetes.io/router.middlewares"
    ]
    text, findings, _ = report(objects, {FRONTEND_POD: FRONTEND_METRICS})
    assert "| OpenAI API, no authentication |" in text
    assert (
        "The API Ingress runs no token check: anyone who reaches it can use the Macs."
        in findings
    )


def test_only_reads_and_never_the_api_key():
    objects = healthy()
    _, _, kube = report(
        objects, {FRONTEND_POD: FRONTEND_METRICS, MAC_POD: proxy_metrics()}
    )
    for tool, *args in kube.calls:
        if tool == "helm":
            assert args == ["list", "-A", "-o", "json"]
        elif args[0] == "exec":
            assert args[-5:-1] == ["curl", "-s", "-m", "5"]
            assert args[-1].endswith("/metrics")
        elif args[:2] == ["get", "secret"]:
            # The bridge URL and the token holders' names, never a token.
            assert args[-1] in (
                "jsonpath={.data.OMLX_BRIDGE_URL}",
                "go-template={{range $k, $v := .data}}{{$k}} {{end}}",
            )
        else:
            assert args[0] in {"get", "top", "config"}, args
            assert "yaml" not in args and "describe" not in args
    assert any(call[1:3] == ("get", "secret") for call in kube.calls)


def test_parses_labels_and_quantities():
    metrics = omlx_status.parse_metrics(
        'a{x="1",y="q\\"uote"} 2\na{x="2"} 3.5e+00\nb 1\n# HELP a\nc NaN\n'
    )
    assert omlx_status.value(metrics, "a") == 5.5
    assert omlx_status.value(metrics, "a", x="2") == 3.5
    assert omlx_status.by_label(metrics, "a", "y") == {'q\\"uote': 2.0, "": 3.5}
    assert omlx_status.value(metrics, "missing") is None
    assert omlx_status.quantity("1908m") == pytest.approx(1.908)
    assert omlx_status.quantity("13129Mi") == 13129 * 2**20
    assert omlx_status.quantity("128") == 128
    profile = copy.deepcopy(healthy()[("configmaps", NS)][0]["data"])
    assert omlx_status.overlay_of(profile) == "qwen3.8-flash-next"
    profile["SERVED_MODEL_NAME"] = "other"
    assert omlx_status.overlay_of(profile) is None
