---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-dynamo-install
description: Installs the Dynamo side of the oMLX-on-Mac example on Kubernetes with NGC Dynamo 1.5.1. Checks the cluster and the Macs, imports the images, installs Prometheus with Grafana and the dynamo-platform chart, creates the Macs' Secrets and the API tokens, applies a model overlay (frontend with KV routing, one proxy per Mac, an Ingress with a token check, the stream repair for Claude Code), verifies it with curl, Claude Code and Codex, and optionally publishes the API and runs AIPerf. Also adds a Mac, switches the model, rolls out changes and recovers after a node reboot. Use once omlx-mac-install has the Macs serving, or to change the cluster side later.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - kubernetes
    - install
---

# Skill: Install the oMLX Example on Kubernetes

The Macs run oMLX with the bridge (`omlx-mac-install`). On the cluster, a
DynamoGraphDeployment named `omlx` runs the frontend with KV routing and one proxy pod per Mac,
behind an Ingress that checks API tokens. Paths below are relative to `examples/backends/omlx`.

| Path | Contents |
|---|---|
| `deploy/base/` | The DGD (`Frontend`, `mac-1`), the Ingress with its API token check (`auth.yaml`, `token_auth.py`), the repair of Claude Code's streams in front of `/v1/messages` (`anthropic-fix.yaml`, `anthropic_fix.py`), the proxy code ConfigMap and `prepare.sh`, the proxy pod's init container |
| `deploy/overlays/<model>/` | `profile.env` (repo, pinned revision, served name, block size, parsers) and the namespace |
| `deploy/components/mac-2/` | The proxy of a second Mac, a copy of `mac-1` that an overlay lists under `components` |
| `deploy/monitoring/` | kube-prometheus-stack values and the oMLX Grafana dashboard |
| `deploy/benchmark/` | The AIPerf Job, its workload (`benchmark.env`) and script |

## Ground Rules

- Mac steps belong to `omlx-mac-install` and run on the Mac. Give the operator the commands;
  run them over SSH only when asked.
- Never print the Macs' API key, an API token or a Hugging Face token, and never put one on a
  command line. Create Secrets from files through pipes, hand tokens to curl on stdin
  (`-K -`), and never read a Secret with `-o yaml`, `-o json` or `describe`.
- Get the operator's go-ahead before cluster-wide installs: the Helm releases bring CRDs,
  webhooks and cluster roles.
- Render overlays with `kubectl kustomize --load-restrictor=LoadRestrictionsNone`. The base reads
  the proxy sources outside `deploy/`, so `kubectl apply -k` fails on it. Apply with
  `--server-side`.
- One overlay per namespace: the DGD, the Ingress and the Secrets have fixed names, and the
  Ingress has no host rule.

## Inputs

| Input | Default |
|---|---|
| kubectl context | the cluster that serves the Macs |
| Overlay | `qwen3.8-flash-next`; another model gets its own, see Switch the Model |
| Each Mac's bridge URL | what `service.sh start` printed; an IP or a name cluster DNS resolves, not `.local` |
| The Macs' `OMLX_API_KEY` | a mode-600 file on Linux, for example `~/.config/dynamo-omlx/api_key` |
| API token holders | one per client, for example `claude-code` and `codex` |
| Hugging Face token | optional; the init container and AIPerf download from the Hub anonymously otherwise |
| Public URL | none; see step 8 before publishing the API beyond the local network |

## 1. Preflight

```bash
kubectl config current-context
kubectl get nodes -o wide                  # Ready
helm version --short                       # v3
kubectl get ingressclass                   # traefik
kubectl get storageclass                   # one marked (default)
MAC_URL=http://<address of the Mac>:<port>
curl -s -o /dev/null -m 5 -w '%{http_code}\n' "$MAC_URL/dynamo/v1/info"    # 401
```

- The Ingress and its token check are Traefik objects, so the cluster needs the `traefik`
  IngressClass; k3s ships it. Prometheus keeps its data on a 20 GiB volume of the default
  StorageClass. Nothing needs a GPU: inference runs on the Macs.
- `401` from every Mac means it answers and wants the key; pods reach it through their node.
  `000` means the node cannot reach it: the Mac sleeps, left the network, or its firewall blocks
  oMLX's Python. Fix that on the Mac first (`omlx-mac-install` step 5).

## 2. Images

Every pod uses `nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.1` (3.9 GB): frontend, proxy, init
container, token check, stream repair and AIPerf. The platform adds `kubernetes-operator:1.5.1`.
Pods use `IfNotPresent`.

```bash
sudo k3s ctr -n k8s.io images ls | grep ai-dynamo     # on a k3s node
```

When the node pulls nvcr.io slowly, import image archives instead (`docker save` on a host with a
better connection): `sudo k3s ctr -n k8s.io images import --platform linux/amd64 <tar>`. The
names must read exactly `nvcr.io/nvidia/ai-dynamo/<image>:1.5.1`.

## 3. Monitoring

Install it before the platform: the platform chart creates its PodMonitors only when their CRD
exists.

```bash
helm upgrade --install prometheus oci://ghcr.io/prometheus-community/charts/kube-prometheus-stack \
    --version 91.4.1 -n monitoring --create-namespace \
    -f deploy/monitoring/kube-prometheus-stack-values.yaml --wait --timeout 20m
kubectl apply --server-side -f ../../../deploy/observability/grafana-dynamo-dashboard-configmap.yaml \
    -f deploy/monitoring/grafana-omlx-dashboard.yaml
```

- The values turn off what k3s cannot be scraped for (etcd, scheduler, controller manager,
  kube-proxy) and Alertmanager, and serve Grafana at `http://<node>/grafana`. On a cluster that
  is not k3s, keep those scrape targets.
- Grafana's Ingress, like the API's, sits on Traefik's port 80 without a host rule: whatever
  publishes that port publishes Grafana too (step 8).
- Grafana's user is `admin`. Give the operator the command for the password instead of running
  it: `kubectl -n monitoring get secret prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 -d`.

## 4. Platform

```bash
helm upgrade --install dynamo-platform \
    https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.5.1.tgz \
    -n dynamo-system --create-namespace --wait --timeout 10m
kubectl -n dynamo-system get pods                    # the operator, 1/1
kubectl get crd dynamographdeployments.nvidia.com
kubectl -n dynamo-system get podmonitors             # dynamo-frontend and dynamo-worker among them
```

The defaults install only the operator: Kubernetes discovery, TCP requests and ZMQ KV events, no
NATS and no etcd. Missing PodMonitors mean monitoring came second; rerun the same `helm upgrade`
with `--reuse-values`.

## 5. Secrets and API Tokens

One Secret per Mac holds its URL and the key; the proxy reads both from its environment.

```bash
kubectl create namespace dynamo-omlx
KEY_FILE=~/.config/dynamo-omlx/api_key
MAC_URL=http://<address of the Mac>:<port>
printf 'header = "Authorization: Bearer %s"\n' "$(cut -d= -f2- "$KEY_FILE")" |
    curl -s -K - -o /dev/null -w '%{http_code}\n' "$MAC_URL/dynamo/v1/info"    # expect 200
test -s "$KEY_FILE" && cut -d= -f2- "$KEY_FILE" |
    kubectl -n dynamo-omlx create secret generic omlx-mac-1 \
        --from-literal=OMLX_BRIDGE_URL="$MAC_URL" --from-file=OMLX_API_KEY=/dev/stdin \
        --dry-run=client -o yaml |
    kubectl apply --server-side -f -
```

Operators often save the whole `OMLX_API_KEY=...` line from the Mac's settings file; `cut`
keeps only the key either way, and the proxy ignores a trailing newline. `test -s` stops a
missing key file from producing a Secret with an empty key. A 401 means a key that is not the
Mac's `OMLX_API_KEY`; a timeout means the Mac or its firewall does not accept the node. For a
Hugging Face token, create `hf-token-secret` with the key `HF_TOKEN` the same way.

Every client of the API needs a token of its own: the Ingress lets a request through only with
one from the Secret `omlx-api-tokens`, one key per holder. Agree the holders with the operator;
the files stay on Linux with mode 600.

```bash
TOKENS=~/.config/dynamo-omlx/api-tokens
(umask 077 && mkdir -p "$TOKENS" && for holder in claude-code codex; do
    test -s "$TOKENS/$holder" ||
        python3 -c 'import secrets; print("omlx_" + secrets.token_urlsafe(32))' > "$TOKENS/$holder"
done)
kubectl -n dynamo-omlx create secret generic omlx-api-tokens --from-file="$TOKENS" \
    --dry-run=client -o yaml | kubectl apply --server-side -f -
```

The token check reads the Secret again on every request; a token shorter than 32 characters
counts as none.

## 6. Deploy

```bash
R=$(mktemp)
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/overlays/<overlay> > "$R"
kubectl apply --server-side --dry-run=server -f "$R"
kubectl apply --server-side -f "$R"
kubectl -n dynamo-omlx wait --for=condition=Ready dgd/omlx --timeout=15m
```

| What you see | Cause |
|---|---|
| A webhook or x509 error on apply | The operator is still starting; retry after a minute |
| The proxy pod stays in `Init` | `kubectl -n dynamo-omlx logs <pod> -c prepare`: the Hub is unreachable. Set `HF_ENDPOINT` in `profile.env`, or add a token |
| `CreateContainerConfigError` | The pod's `omlx-mac-<n>` Secret is missing |
| `Running` but `0/1`, log `waiting for the oMLX bridge` | The Mac is unreachable from pods, or its model is still loading. The startup probe allows 2 hours (720 × 10 s), then the pod restarts and waits again |
| Restarts, log `the oMLX bridge refused the proxy` | The key in the Secret is not the Mac's |
| Restarts, log `block size ... expected` | `EXPECT_BLOCK_SIZE` differs from the Mac's `OMLX_KV_BLOCK_SIZE`: the overlay does not match the model the Mac serves |
| `0/1` for minutes after `oMLX bridge ... ready`, the log ends at `Registering event channel`, and `kubectl -n dynamo-omlx get dynamoworkermetadatas` has no entry for the pod | A hang in Dynamo 1.5.1's Kubernetes discovery registration, seen once right after the DGD was recreated. Delete the pod; the replacement registers within seconds |

The proxy pod becomes Ready only once it registered the model, and only Ready proxies receive
requests. Select pods by label; worker names carry a hash:
`kubectl -n dynamo-omlx get pods -l nvidia.com/dynamo-component=mac-1`.

The base also sets on the frontend:

- `DYN_ENABLE_ANTHROPIC_API` and `DYN_STRIP_ANTHROPIC_PREAMBLE`: the Anthropic Messages API for
  Claude Code, with its billing preamble stripped so that every client shares the cached prefix.
- `DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS=15000`: an SSE comment line (`:`) after 15 s without data,
  so a stream that waits long for its first token survives proxies that close idle responses.

## 7. Verify

```bash
NODE=<node address>
AUTH=~/.config/dynamo-omlx/api-tokens/<holder>      # any holder of step 5
curl -s -o /dev/null -w '%{http_code}\n' http://$NODE/v1/models                 # 401
printf 'header = "Authorization: Bearer %s"\n' "$(cat "$AUTH")" |
    curl -s -K - http://$NODE/v1/models
printf 'header = "Authorization: Bearer %s"\n' "$(cat "$AUTH")" |
    curl -s -K - http://$NODE/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "<SERVED_MODEL_NAME>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
P=$(kubectl -n dynamo-omlx get pod -l nvidia.com/dynamo-component=mac-1 -o name | head -1)
kubectl -n dynamo-omlx exec "$P" -c main -- curl -s localhost:9090/metrics |
    grep -E '^omlx_(bridge_up|model_loaded|proxy_kv_sync_state)'
```

- Expect 401 without a token, then the model, a completion, `omlx_bridge_up 1`,
  `omlx_model_loaded 1` and `omlx_proxy_kv_sync_state{state="live"} 1`. The log of
  `deploy/omlx-auth` names the holder of every request it lets through.
- Send one prompt longer than two blocks twice, one after the other. The second answer's
  `usage.prompt_tokens_details.cached_tokens` is above 0. oMLX caches blocks when a request
  finishes, so concurrent requests do not hit.
- The Anthropic Messages API answers the same prompt: the same `printf ... | curl -s -K -` with
  `http://$NODE/v1/messages -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model": "<SERVED_MODEL_NAME>", "max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}]}'`.
  It passes `omlx-anthropic-fix`, so its `Server` header names aiohttp.
- For a vision-language model, the proxies report `omlx_model_vision 1` and
  `kubectl -n dynamo-omlx logs deploy/omlx-frontend | grep 'MM-aware KV routing'` says
  `enabled`. Send an image as `omlx-status` shows, twice: the answer describes it, and the second
  answer's `cached_tokens` includes the image's tokens. A warning `not supported for MM-aware KV
  routing` instead means the router knows no image processor by the model's path: see
  `OMLX_MODEL_DIR` in `profile.env`.
- In Grafana: the Dynamo dashboard (frontend) and "oMLX on Dynamo" (proxies and Macs).

Then run both coding agents once, with the settings of `omlx-status`'s Coding Agents section
(Codex reads its provider from `~/.codex/omlx.config.toml`):

```bash
TOKENS=~/.config/dynamo-omlx/api-tokens
D=$(mktemp -d) && printf 'one\n' > "$D/a.txt" && printf 'two\n' > "$D/b.txt" && cd "$D"
CLAUDE_CONFIG_DIR=$(mktemp -d) ANTHROPIC_BASE_URL=http://$NODE \
ANTHROPIC_AUTH_TOKEN=$(cat "$TOKENS/<holder>") \
ANTHROPIC_MODEL=<SERVED_MODEL_NAME> ANTHROPIC_SMALL_FAST_MODEL=<SERVED_MODEL_NAME> \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    claude -p "Read a.txt and b.txt at the same time, then print both lines." \
    --allowedTools Read < /dev/null
OMLX_API_TOKEN=$(cat "$TOKENS/<holder>") codex exec --profile omlx --skip-git-repo-check \
    -s read-only "Reply with just the word OK." < /dev/null
kubectl -n dynamo-omlx logs deploy/omlx-anthropic-fix --since=10m | grep dropped
```

- Claude Code prints `one` and `two` after a notice that it does not recognize the model;
  Codex prints `OK` after a harmless warning that the model's metadata is missing.
- A turn that called both tools at once shows up in the stream repair's log as
  `dropped <n> whitespace deltas of closed blocks`. Without the repair, Claude Code stops such
  turns with `API Error: The response stream was malformed`.
- From inside a Claude Code session, start the nested `claude` with
  `env -i HOME=$HOME PATH=$PATH` in front: it inherits the session's settings otherwise.

## 8. Publish the API

Skip this while every client is on the local network. Otherwise agree the way with the
operator; a front proxy, a tunnel or a CDN all work.

- Put TLS in front. The Ingress serves plain HTTP on Traefik's port 80, where a token crosses the
  network readable.
- Route only `/v1` to the node's port 80. That port also serves Grafana at `/grafana`, so
  forwarding the whole port publishes Grafana's login page.
- Clients take the published origin: Claude Code's `ANTHROPIC_BASE_URL` without `/v1`, Codex's
  `base_url` with it.
- Idle timeouts: Cloudflare closes a response after 100 s without data. Streams carry the
  frontend's `:` line every 15 s and survive; a request without streaming that takes longer gets
  524.
- Repeat step 7's curl checks against the public URL, the one without a token included.

## Benchmark

```bash
kubectl -n dynamo-omlx delete job omlx-aiperf --ignore-not-found
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/benchmark |
    kubectl apply --server-side -f -
kubectl -n dynamo-omlx logs -f job/omlx-aiperf
```

- The Job takes the model from the `profile.env` that `deploy/benchmark/kustomization.yaml`
  lists; point it at the deployed overlay. `benchmark.env` holds the workload: a concurrency
  sweep (1, 2, 4 and 8, each 120 seconds, three runs, so AIPerf reports 95% confidence
  intervals) and a 300-second shared-prefix run whose first request is cold. It takes about 40
  minutes of the Macs' time; agree that with the operator first.
- Every Job draws a new seed (first log line), so oMLX's SSD cache holds none of its prompts;
  `SEED` repeats a Job. Token counts come from the server's usage, which includes the cached
  prompt tokens.
- Never add `ignore_eos` or `min_tokens`: the proxy rejects both, so every request would fail.
- The log is the only output that outlives the pod: after each phase it holds AIPerf's CSV
  summaries, each after a `--- csv <path>` line. Save them with
  `kubectl -n dynamo-omlx logs job/omlx-aiperf > aiperf.log` and
  `awk '/^--- csv /{f = $(3); gsub("/", "__", f); next} /^=== /{f = ""} f {print > f}' aiperf.log`.
  `sweep__aggregate__sweep_aggregate__profile_export_aiperf_sweep.csv` compares the
  concurrencies; `sweep__aggregate__concurrency_<n>__profile_export_aiperf_aggregate.csv` has
  each metric's mean, CV and confidence interval.

## Add a Mac

1. Set up the Mac with `omlx-mac-install`: the same model, `OMLX_KV_BLOCK_SIZE` and
   `OMLX_API_KEY`.
2. Create its Secret `omlx-mac-2` as in step 5.
3. Uncomment `components` in the overlay's `kustomization.yaml`. For a third Mac, copy
   `deploy/components/mac-2` to `mac-3`, change 2 to 3 in its last two operations, and list it
   after `mac-2`.
4. Render, dry-run and apply as in step 6. Both proxies must become Ready; a proxy whose model
   card differs (another revision, block size or served name) is not admitted.

## Switch the Model

Give the new model its own overlay: copy `deploy/overlays/qwen3.8-flash-next` and set every
value in its `profile.env`, with `OMLX_MODEL_REVISION` the `sha` in `.dynamo-omlx-download.json`
of the model's directory on a Mac. For a vision-language model, `OMLX_MODEL_DIR` must contain
the one name of the frontend's image processor registry that fits the model's processor (the
list is in `tests/test_omlx_deploy.py`), and `OMLX_IMAGE_MAX_PIXELS` caps its images; empty
keeps the model's own range. The overlays share the names, so delete the old one first:
`kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/overlays/<old> | kubectl delete -f -`.
Recreate each Mac's Secret with the URL and key of the new service, apply the new overlay, and
point `deploy/benchmark/kustomization.yaml` at its `profile.env`.

## Roll Out a Change

- Code under `dynamo_omlx/` or a `profile.env` edit changes a ConfigMap's hash, so applying the
  rendered overlay rolls the proxies; the old pod goes once the new one is Ready. Preview with
  `kubectl diff --server-side -f "$R"`. Delete superseded `omlx-proxy-code-*` and
  `omlx-profile-*` ConfigMaps after the rollout.
- A new module under `dynamo_omlx/proxy/` needs a line in `deploy/base/kustomization.yaml`;
  `tests/test_omlx_deploy.py` fails until it has one.
- A change to the `Frontend` component rolls only the frontend: its new pod takes over once
  Ready, within about 20 seconds, and the proxies keep running.
- A Secret change rolls nothing: delete the proxy pods to pick it up.
- To add or revoke an API token, add or delete its file and run the `omlx-api-tokens` line of
  step 5 again. The token check sees the change within two minutes;
  `kubectl -n dynamo-omlx rollout restart deploy/omlx-auth` applies it at once.
- A Dynamo upgrade changes the chart version and every image tag under `deploy/` together, and
  from 1.6 on sets `DYN_STRUCTURAL_TAG=false` on the proxies. Then go through Dynamo Issues in
  `AGENTS.md` and the README: for each issue the new version fixes, remove its workaround,
  tests and mentions, then its entries in both.

## After a Node Reboot

The pods restart by themselves, and the Macs keep serving. Check the pods and step 7's
`/v1/models`:

```bash
kubectl -n dynamo-omlx get pods
kubectl -n dynamo-omlx logs deploy/omlx-frontend | grep -c 'lease conflicts'
```

When the frontend lists no model and its log has `endpoint registration lease conflicts with a
pre-existing specification`, a registration from before the reboot blocks it. Only then delete
the pod; its replacement registers anew:

```bash
kubectl -n dynamo-omlx delete pod -l nvidia.com/dynamo-component=Frontend
```
