# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Kubernetes manifests in deploy/ against the code they ship."""

import ast
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

EXAMPLE = Path(__file__).resolve().parents[1]
DEPLOY = EXAMPLE / "deploy"
OVERLAYS = sorted(p.parent for p in (DEPLOY / "overlays").glob("*/kustomization.yaml"))


def code_files() -> dict[str, Path]:
    """ConfigMap key -> source of the base's omlx-proxy-code generator."""
    kustomization = yaml.safe_load((DEPLOY / "base" / "kustomization.yaml").read_text())
    (generator,) = (
        g for g in kustomization["configMapGenerator"] if g["name"] == "omlx-proxy-code"
    )
    files = {}
    for entry in generator["files"]:
        key, _, path = entry.partition("=")
        files[key] = (DEPLOY / "base" / (path or key)).resolve()
    return files


def read_env(path: Path) -> dict[str, str]:
    lines = path.read_text().splitlines()
    return dict(
        line.split("=", 1)
        for line in lines
        if line.strip() and not line.startswith("#")
    )


def test_code_configmap_ships_the_whole_proxy_package():
    files = code_files()
    package = {k: v for k, v in files.items() if k.startswith("dynamo_omlx--")}
    for key, source in package.items():
        assert source == EXAMPLE / key.replace("--", "/"), key
    expected = {
        EXAMPLE / "dynamo_omlx" / "__init__.py",
        EXAMPLE / "dynamo_omlx" / "protocol.py",
        *(EXAMPLE / "dynamo_omlx" / "proxy").glob("*.py"),
    }
    assert set(package.values()) == expected
    assert files["download_model.py"] == EXAMPLE / "mac" / "download_model.py"

    # Every dynamo_omlx module a shipped module imports is shipped too.
    shipped = {
        str(p.relative_to(EXAMPLE).with_suffix("")).replace("/", ".") for p in expected
    }
    for source in expected:
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name.startswith("dynamo_omlx."):
                    assert (
                        name in shipped or f"{name}.__init__" in shipped
                    ), f"{source.name} imports {name}, which deploy/ does not ship"


def test_prepare_lays_out_the_package_and_fetches_the_pinned_revision(tmp_path):
    src, code, models, bin_dir = (
        tmp_path / d for d in ("src", "code", "models", "bin")
    )
    for d in (src, models, bin_dir):
        d.mkdir()
    for key, source in code_files().items():
        shutil.copy(source, src / key)
    # Stands in for download_model.py: records its arguments.
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        'echo "$@" > "$PREPARE_MODELS/args"\n'
        'mkdir -p "$PREPARE_MODELS/$SERVED_MODEL_NAME"\n'
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PREPARE_SRC": str(src),
        "PREPARE_CODE": str(code),
        "PREPARE_MODELS": str(models),
        "OMLX_MODEL_REPO": "org/model",
        "OMLX_MODEL_REVISION": "a" * 40,
        "SERVED_MODEL_NAME": "model",
    }
    subprocess.run(["sh", str(DEPLOY / "base" / "prepare.sh")], env=env, check=True)

    for name in ("__init__.py", "protocol.py", "proxy/__main__.py", "proxy/engine.py"):
        assert (code / "dynamo_omlx" / name).is_file(), name
    args = (models / "args").read_text().split()
    assert args[:4] == [
        str(src / "download_model.py"),
        "org/model",
        "--revision",
        "a" * 40,
    ]
    assert "--metadata-only" in args
    assert args[args.index("--model-dir") + 1] == str(models)
    assert args[args.index("--name") + 1] == "model"


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda p: p.name)
def test_profiles_pin_the_model(overlay):
    profile = read_env(overlay / "profile.env")
    assert re.fullmatch(r"[0-9a-f]{40}", profile["OMLX_MODEL_REVISION"])
    assert "/" not in profile["SERVED_MODEL_NAME"]
    assert int(profile["EXPECT_BLOCK_SIZE"]) > 0
    assert {"OMLX_MODEL_REPO", "DYN_TOOL_CALL_PARSER", "DYN_REASONING_PARSER"} <= set(
        profile
    )


needs_kubectl = pytest.mark.skipif(
    shutil.which("kubectl") is None, reason="needs kubectl"
)


def render(directory: Path) -> list[dict]:
    rendered = subprocess.run(
        [
            "kubectl",
            "kustomize",
            "--load-restrictor=LoadRestrictionsNone",
            str(directory),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [d for d in yaml.safe_load_all(rendered) if d]


@needs_kubectl
@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda p: p.name)
def test_overlay_renders(overlay):
    docs = render(overlay)
    kinds = sorted(d["kind"] for d in docs)
    assert kinds == [
        "ConfigMap",
        "ConfigMap",
        "ConfigMap",
        "ConfigMap",
        "Deployment",
        "Deployment",
        "DynamoGraphDeployment",
        "Ingress",
        "Middleware",
        "Service",
        "Service",
    ]
    configmaps = {d["metadata"]["name"] for d in docs if d["kind"] == "ConfigMap"}
    (dgd,) = (d for d in docs if d["kind"] == "DynamoGraphDeployment")
    deployments = [d for d in docs if d["kind"] == "Deployment"]

    referenced, images = set(), set()
    pod_specs = [c["podTemplate"]["spec"] for c in dgd["spec"]["components"]]
    for spec in pod_specs + [d["spec"]["template"]["spec"] for d in deployments]:
        containers = spec["containers"] + spec.get("initContainers", [])
        images |= {c["image"] for c in containers}
        referenced |= {
            e["configMapRef"]["name"]
            for c in containers
            for e in c.get("envFrom", [])
            if "configMapRef" in e
        }
        referenced |= {
            v["configMap"]["name"] for v in spec.get("volumes", []) if "configMap" in v
        }
    # Generated names carry a content hash; the DGD and the Deployments use those.
    assert referenced == configmaps
    assert len(images) == 1


@needs_kubectl
@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda p: p.name)
def test_ingress_runs_the_token_check(overlay):
    docs = render(overlay)
    (ingress,) = (d for d in docs if d["kind"] == "Ingress")
    (middleware,) = (d for d in docs if d["kind"] == "Middleware")
    services = {d["metadata"]["name"]: d for d in docs if d["kind"] == "Service"}
    namespace = ingress["metadata"]["namespace"]
    # Both name the namespace, which the overlay sets.
    annotation = ingress["metadata"]["annotations"][
        "traefik.ingress.kubernetes.io/router.middlewares"
    ]
    assert annotation == f"{namespace}-{middleware['metadata']['name']}@kubernetescrd"
    address = middleware["spec"]["forwardAuth"]["address"]
    host = re.match(r"http://([^:/]+)", address).group(1)
    name = host.removesuffix(f".{namespace}.svc.cluster.local")
    port = services[name]["spec"]["ports"][0]["port"]
    assert address.startswith(f"http://{host}:{port}/")


@needs_kubectl
@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda p: p.name)
def test_claude_code_streams_pass_the_repair(overlay):
    docs = render(overlay)
    (ingress,) = (d for d in docs if d["kind"] == "Ingress")
    (dgd,) = (d for d in docs if d["kind"] == "DynamoGraphDeployment")
    services = {d["metadata"]["name"]: d for d in docs if d["kind"] == "Service"}
    deployments = {d["metadata"]["name"]: d for d in docs if d["kind"] == "Deployment"}
    paths = {
        p["path"]: p["backend"]["service"]
        for rule in ingress["spec"]["rules"]
        for p in rule["http"]["paths"]
    }
    frontend = f"{dgd['metadata']['name']}-frontend"
    assert paths["/v1"]["name"] == frontend
    repair = paths["/v1/messages"]
    service = services[repair["name"]]
    assert service["spec"]["ports"][0]["port"] == repair["port"]["number"]
    (deployment,) = (
        d
        for d in deployments.values()
        if service["spec"]["selector"].items()
        <= d["spec"]["template"]["metadata"]["labels"].items()
    )
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    command = container["command"]
    upstream = command[command.index("--upstream") + 1]
    assert upstream == f"http://{frontend}:{paths['/v1']['port']['number']}"


@needs_kubectl
@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda p: p.name)
def test_frontend_streams_beat_cloudflares_idle_timeout(overlay):
    (dgd,) = (d for d in render(overlay) if d["kind"] == "DynamoGraphDeployment")
    (frontend,) = (c for c in dgd["spec"]["components"] if c["type"] == "frontend")
    (main,) = frontend["podTemplate"]["spec"]["containers"]
    env = {e["name"]: e.get("value") for e in main["env"]}
    # Cloudflare closes a response after 100 s without data.
    assert 0 < int(env["DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS"]) < 100_000


@needs_kubectl
def test_second_mac_copies_the_first(tmp_path):
    (tmp_path / "kustomization.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "kustomize.config.k8s.io/v1beta1",
                "kind": "Kustomization",
                "resources": [
                    os.path.relpath(
                        DEPLOY / "overlays" / "qwen3.8-flash-next", tmp_path
                    )
                ],
                "components": [
                    os.path.relpath(DEPLOY / "components" / "mac-2", tmp_path)
                ],
            }
        )
    )
    (dgd,) = (d for d in render(tmp_path) if d["kind"] == "DynamoGraphDeployment")
    frontend, mac_1, mac_2 = dgd["spec"]["components"]
    assert (frontend["name"], mac_1["name"], mac_2["name"]) == (
        "Frontend",
        "mac-1",
        "mac-2",
    )
    secret = mac_2["podTemplate"]["spec"]["containers"][0]["envFrom"][0]["secretRef"]
    assert secret["name"] == "omlx-mac-2"
    del mac_2["name"]
    mac_2["podTemplate"]["spec"]["containers"][0]["envFrom"][0]["secretRef"] = {
        "name": "omlx-mac-1"
    }
    del mac_1["name"]
    assert mac_2 == mac_1


@needs_kubectl
def test_benchmark_reads_the_deployed_profile():
    docs = render(DEPLOY / "benchmark")
    assert sorted(d["kind"] for d in docs) == ["ConfigMap", "ConfigMap", "Job"]
    settings = next(
        d["data"]
        for d in docs
        if d["metadata"]["name"].startswith("omlx-benchmark-")
        and "CONCURRENCY" in d.get("data", {})
    )
    kustomization = yaml.safe_load(
        (DEPLOY / "benchmark" / "kustomization.yaml").read_text()
    )
    (env_files,) = (
        g["envs"] for g in kustomization["configMapGenerator"] if "envs" in g
    )
    (profile_file,) = (f for f in env_files if f.endswith("profile.env"))
    profile = read_env((DEPLOY / "benchmark" / profile_file).resolve())
    assert {k: settings[k] for k in profile} == profile
