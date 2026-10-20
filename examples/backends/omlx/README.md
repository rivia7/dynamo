<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# oMLX on Apple Silicon behind Dynamo

This example serves a model with [oMLX](https://github.com/jundot/omlx) on one or more Macs
while everything else runs on Linux: the Dynamo frontend, the KV-aware router, discovery,
and metrics. The Macs run only oMLX. Dynamo is not installed on them.

oMLX keeps a persistent prefix cache on SSD. The example mirrors that cache into the Dynamo
KV router, so requests that share a prompt prefix go to the Mac that already holds it. On a Mac,
prefill is the expensive part of a long agentic prompt, so a cache hit saves the most time there.

> [!NOTE]
> This is an example, not a supported backend. The Mac-side bridge relies on oMLX internals.
> It was written against oMLX commit `31b6b6a0`. `mac/setup.sh` and every start of
> `mac/service.sh` check that the installed oMLX still offers what the bridge hooks into.

## Architecture

```text
Linux                                                   Mac (one per proxy)
+-------------------------------------------+         +-------------------------------------+
| dynamo.frontend --router-mode kv          |         | omlx serve (unmodified)             |
|   chat template, tokenization, parsers    |         |   + dynamo_omlx.bridge (/dynamo/v1) |
|              |                ^           |         |       token-in / token-out          |
|   request (TCP)         KV events         |         |       KV event journal + snapshot   |
|              v                |           |  HTTP   |       cache hooks, /stats           |
| python -m dynamo_omlx.proxy (1 per Mac)   |<------->|   EnginePool, memory guard,         |
|   token streams, KV sync, metrics, health |         |   SSD prefix cache, the model       |
+-------------------------------------------+         +-------------------------------------+
```

The example has two components:

- **Bridge** (`dynamo_omlx.bridge`). It starts the unmodified `omlx serve` and mounts extra
  routes under `/dynamo/v1`. It serves token-in/token-out generation through oMLX's own engine
  pool, so model loading, memory admission, pinning and the SSD cache behave exactly as in
  plain oMLX. Hooks on oMLX's prefix cache record which blocks are stored or evicted.
- **Proxy** (`dynamo_omlx.proxy`). This is a Dynamo worker built on the unified backend SDK.
  It loads no weights: it registers the model from a directory holding only `config.json`, the
  tokenizer and the chat template. It forwards token ids and the files of any images to its
  Mac, streams tokens back, and publishes the Mac's cache state to the router.

The bridge and proxy share one versioned wire protocol, `dynamo_omlx/protocol.py`. All traffic
between them is HTTP with NDJSON streams on oMLX's port, authenticated with oMLX's API key.

| Path | Runs on | Contents |
|---|---|---|
| `dynamo_omlx/bridge/` | Mac | The bridge, which runs inside oMLX's Python |
| `dynamo_omlx/proxy/` | Linux | The proxy worker |
| `dynamo_omlx/protocol.py` | Both | The wire protocol |
| `mac/` | Mac | Setup, model download and service scripts, and the settings template |
| `launch/agg.sh` | Linux | The frontend with KV routing, plus one proxy per Mac |
| `deploy/` | Kubernetes | The same Linux side for a cluster, with monitoring and a benchmark; see [Kubernetes](#kubernetes) |
| `tests/` | Linux | Unit tests |

### How KV events stay correct across the network

1. **Bridge: recording.** The bridge announces a block only after oMLX has persisted it.
   A block is announced only if its whole parent chain is announced too. Evicting a block also
   withdraws its descendants, because oMLX restores a prefix only as a full chain from the root.
   Every change gets a sequence number and goes into a replay ring.
2. **Proxy: following.** The proxy follows the event stream. After a dropped connection it
   resumes from the last sequence number it applied.
   - If the bridge restarted, the ring no longer covers the gap, or an event does not fit the
     local state, the proxy fetches a snapshot and publishes only the difference.
   - Recovery never withdraws blocks that are still valid.
3. **Proxy to router.** This leg uses Dynamo's own recovery: a worker-local indexer with
   replay and tree dump. Router restarts need nothing from this example.

Block identity comes from two hashes:

- **Router hash.** The router hashes the token ids of each block itself.
- **External block hash.** The bridge derives it from the parent hash and the tokens
  (`protocol.block_hash`). It does not use oMLX's own hash, which is salted with the local model
  directory name. Two Macs therefore announce identical hashes for identical prefixes.

### Images

A vision-language model on the Macs takes images in `image_url` content parts, as `data:` URLs
of a base64 image or as `http(s)` URLs. Qwen3.8-Flash-Next is one. The prefix cache covers
images as it covers text: on the Mac and in the router.

1. **Frontend.** It applies the chat template, which leaves one `<|image_pad|>` per image, and
   passes the URLs on. It identifies each image by `mm_hash`, the XXH3 hash of its URL, and
   routes by the prompt with each placeholder expanded into that many pad values derived from
   the image's `mm_hash`.
2. **Proxy.** It sends the Mac the token ids, each image file and its `mm_hash`. It fetches
   `http(s)` images itself, from public addresses only unless `DYN_MM_ALLOW_INTERNAL=1`, as for
   the frontend.
3. **Bridge.** It runs the second half of oMLX's own VLM path on Dynamo's tokens:
   - It knows an image by the hash of its file and the pixel range it is processed within.
   - It expands each placeholder by the model's image processor and encodes the images,
     through oMLX's vision feature cache, which keeps each image's features and patch grid.
     A request whose images are all cached skips their vision encoder and, for Qwen3.8, their
     decoding and preprocessing too.
   - From each image's first token on, it keys oMLX's prefix cache blocks with the identity of
     the images so far. A cached prefix then never stands in for other images.
   - It announces stored blocks with each image's tokens replaced by the image's pad value,
     exactly the tokens the router hashes for a request. The router therefore matches image
     blocks, and only those of the same images.

The router expands images only for a model it has an image token counter for. It picks one by
a name in the proxies' `--model-path`, then by `model_type` in `config.json`, and knows no
`qwen4_exp`. Qwen3.8 uses Qwen3-VL's image processor, so its path must contain `qwen3_vl`: the
Kubernetes overlay uses `/models/qwen3_vl/Qwen3.8-Flash-Next`, and `agg.sh` needs a path like it.
The frontend then logs `MM-aware KV routing enabled` for the model. Without a counter, image
requests still work and still reuse the Mac's cache, but the router matches them only by the
text before the first image.

An image costs its pixels / 1024 tokens within the pixel range of the proxies'
`preprocessor_config.json`. The frontend counts image tokens with that range, and the proxy has
the Macs scale images into it, so the two counts agree. The model's own range goes up to 16.7
megapixels, 16,384 tokens, which costs a 12-megapixel photo about 15 seconds of prefill instead
of 2. The Kubernetes overlay therefore caps it at 4,194,304 pixels (`OMLX_IMAGE_MAX_PIXELS` in
`profile.env`), about what oMLX's own API shrinks images to. With `agg.sh`, set `max_pixels` in
`preprocessor_config.json` of `--model-path` the same way.

A request reaches the proxy as one request-plane message that carries its images as base64,
and Dynamo 1.5.1 puts each image in it twice. Dynamo's default limit of 32 MiB per message
would therefore fail a request with about 16 MiB of images, two photos, with HTTP 500. The
overlay raises the proxies' `DYN_TCP_MAX_MESSAGE_SIZE` to 128 MiB, about 64 MiB of images per
request; export the same variable before `agg.sh`. The frontend separately refuses a `data:`
URL over 16 MiB (`DYN_MM_MAX_DATA_URL_MB`).

## Prerequisites

- **Mac**
  - Apple Silicon with enough memory for the model. `mac/setup.sh` reports the GPU
    working-set limit, which the weights and the KV cache share.
  - oMLX, from the oMLX app or Homebrew. `mac/setup.sh --install` installs it with Homebrew.
  - This example directory. Dynamo is not installed on the Mac.
  - A Hugging Face access token is recommended. The Hub limits anonymous downloads.
- **Linux**
  - Dynamo's Python packages built from this repository. See the
    [contribution guide](../../../docs/fern/pages/community/contributing/overview.md).
  - The model's `config.json`, tokenizer files and chat template, without weights.
    `mac/download_model.py --metadata-only` fetches them.
- **Network**
  - The proxy host must reach the Mac's oMLX port.
  - Use a private network or a tunnel: the protocol carries prompts and the token ids of
    cached prefixes.
  - Prefer Ethernet on the Macs. The model download and every token stream cross this link.

### Qwen3.8-Flash-Next

The Mac scripts default to `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`, an oMLX oQ4e quantization of
Qwen3.8-Flash-Next that keeps its multi-token prediction (MTP) layer.

- **Memory.** The checkpoint is 99 GiB in 21 shards. On a 256 GB Mac it stays fully resident:
  the GPU working-set limit there is about 223 GiB, and the rest holds the KV cache. No expert
  offload is needed.
- **Architecture.** 36 Gated DeltaNet linear-attention layers and 12 full-attention layers,
  512 routed experts with 10 active per token, and a 262,144-token context.
- **Block size.** oMLX caches this model in blocks of 8192 tokens on an M5-generation Mac with
  oMLX's native kernels, and of 2048 or 4096 tokens elsewhere. Every block boundary stores the
  recurrent state of the linear-attention layers.
  - oMLX only ever enlarges a preset, so `OMLX_KV_BLOCK_SIZE=8192` in the Mac settings gives
    every Mac the same size.
  - Prompt prefixes shorter than one block are not cached. The router therefore sees hits only
    for prompts of 8192 tokens or more, such as agentic prompts with long tool definitions.
- **Native kernels.** The oMLX app ships them. A Homebrew install builds them with
  `brew reinstall jundot/omlx/omlx --HEAD --with-custom-kernel`, which needs Xcode. Without
  them, prefill runs on oMLX's generic kernels. `mac/setup.sh` reports which kernels are present.
- **Parsers.** `agg.sh` uses the `qwen3_coder` tool-call parser and the `qwen3` reasoning
  parser, like the [Qwen3.8-Flash-Next recipes](../../../recipes/qwen3.8-flash-next/README.md).
- **MTP.** oMLX's multi-token prediction, speculative decoding with the model's own MTP head,
  is an oMLX per-model setting and is off by default. `OMLX_MTP=on` in the Mac settings turns
  it on when the service starts, and `./service.sh status` then shows
  `MTP: on, at most <n> draft tokens per cycle`. oMLX keeps the setting after the service
  stops; `OMLX_MTP=off` and a restart turn it off. It drafts up to 3 tokens per cycle unless
  `OMLX_MTP_DEPTH` says otherwise. The setting changes speed only, so Macs behind one model may
  differ. MTP has not been tested through the bridge yet.
- **Settings must match across Macs.** The proxy registers the block size it observes, and
  Dynamo only admits replicas whose model card matches. Serve the same checkpoint with the same
  `OMLX_KV_BLOCK_SIZE` on every Mac.

## Quick Start

### Mac Setup

Copy this example directory to each Mac, for example from the Linux host with an alias from
[SSH Access](#ssh-access):

```bash
rsync -a --exclude tests --exclude deploy --exclude __pycache__ examples/backends/omlx/ omlx-mac-1:dynamo-omlx/
```

Then run these on the Mac:

```bash
cd ~/dynamo-omlx/mac
./setup.sh                        # once per Mac
./download_model.sh --login       # once per Mac: saves a Hugging Face token
./download_model.sh --background  # 99 GiB; rerun it to resume
./service.sh start                # once the download has finished
```

| Script | What it does |
|---|---|
| `setup.sh` | Writes the Mac's settings to `~/.config/dynamo-omlx/omlx-mac.env` with a new API key, unless the file exists. Reports chip, memory, disk, network, the Hugging Face token and native kernels, and runs the bridge's compatibility check. `--install` installs oMLX with Homebrew. |
| `download_model.sh` | Downloads `OMLX_MODEL_REPO` into oMLX's model directory. Files land in a hidden staging directory and move into place once every size is verified, so oMLX never sees a partial model. It retries network errors, resumes when rerun, turns on Xet's parallel transfers, and keeps the Mac awake. `--login` saves a Hugging Face token, `--background` detaches the download and logs a progress line every 30 seconds, and `--update` fetches the files a newer revision changed. |
| `service.sh` | `start`, `stop`, `restart`, `status` and `logs` for oMLX with the bridge, or `run` in the foreground. A started service survives the end of the SSH session, restarts when it exits unexpectedly, and keeps the Mac awake. Logs go to `~/Library/Logs/dynamo-omlx/`. |
| `uninstall.sh` | Removes what the other scripts created. See [Uninstall](#uninstall). |

- `download_model.sh --login` asks for the token without echoing it, checks it with the Hub,
  and saves it to huggingface_hub's token file, where oMLX finds it as well. Every download
  first reports the account it runs as.
- `./service.sh start` waits until the model is loaded, then prints the block size and the URL
  for the Linux side. If oMLX exits while starting, it prints the end of the log and stops the
  service instead of retrying.
- `setup.sh` generates the API key, and `grep OMLX_API_KEY ~/.config/dynamo-omlx/omlx-mac.env`
  shows it. The Linux side and every further Mac use the same key.
- Nothing starts the service again after a reboot. Run `./service.sh start` again.

### Linux Setup

```bash
python3 examples/backends/omlx/mac/download_model.py Jundot/Qwen3.8-Flash-Next-oQ4e-mtp \
    --metadata-only --model-dir /models/qwen3_vl --name Qwen3.8-Flash-Next

export OMLX_API_KEY=...   # the key from the Macs' settings file
OMLX_BRIDGE_URLS="http://<mac-1>:8000 http://<mac-2>:8000" \
examples/backends/omlx/launch/agg.sh --model-path /models/qwen3_vl/Qwen3.8-Flash-Next
```

- Use the URLs that `service.sh start` prints on each Mac.
- `--metadata-only` fetches the config, tokenizer and chat template files and needs
  `huggingface_hub`. Copying the same files from a Mac's model directory works as well.
- `qwen3_vl` in the path makes the router route images by their content; see
  [Images](#images). The served model name stays the last part of the path.
- To cap images as the Kubernetes overlay does, add `"max_pixels": 4194304` to that
  directory's `preprocessor_config.json` before starting `agg.sh`.
- `agg.sh` starts the frontend with `--router-mode kv` and one proxy per URL. Each proxy waits
  until its Mac has the model loaded before it registers.

Then send a request:

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 64
}'
```

Discovery defaults to `file`, which is enough while every Dynamo process runs on one Linux
host. Use `DYN_DISCOVERY_BACKEND=etcd` when proxies run on several hosts, or run the Linux side
on [Kubernetes](#kubernetes). Every proxy of one model needs byte-identical model files at the
same path.

## Kubernetes

The Linux side can run on a Kubernetes cluster instead, from NVIDIA's NGC Dynamo 1.5.1 Helm
chart and containers. A DynamoGraphDeployment (DGD) named `omlx` runs the frontend with KV
routing and one proxy pod per Mac. The Macs are set up as in [Mac Setup](#mac-setup).

```text
http://<node>/v1 --> traefik, token check --> Frontend (KV router) --> proxy mac-1 --HTTP--> Mac 1
                                                                   --> proxy mac-2 --HTTP--> Mac 2
Claude Code's /v1/messages passes a stream repair between traefik and the frontend
Prometheus scrapes the frontend and the proxies; Grafana shows them at http://<node>/grafana
```

| Path | Contents |
|---|---|
| `deploy/base/` | The DGD with the `Frontend` component and the first Mac's proxy `mac-1`, an Ingress for `/v1` with an API token check, the stream repair for Claude Code, and the proxy code |
| `deploy/overlays/qwen3.8-flash-next/` | Qwen3.8-Flash-Next, the model of `mac/omlx-mac.env.example` |
| `deploy/monitoring/` | Values for kube-prometheus-stack, and the oMLX Grafana dashboard |
| `deploy/benchmark/` | An AIPerf Job |

No container image holds the proxy. The overlay ships its code in a ConfigMap, and an init
container lays the code out and fetches the model's config and tokenizer files from Hugging
Face at the revision that the overlay's `profile.env` pins. One image,
`nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.1`, runs every container.

### Install the Platform

You need kubectl, Helm 3, the traefik Ingress class (k3s ships it) and a default StorageClass.
From `examples/backends/omlx`:

```bash
helm upgrade --install prometheus oci://ghcr.io/prometheus-community/charts/kube-prometheus-stack \
    --version 91.4.1 -n monitoring --create-namespace \
    -f deploy/monitoring/kube-prometheus-stack-values.yaml --wait --timeout 20m
kubectl apply --server-side -f ../../../deploy/observability/grafana-dynamo-dashboard-configmap.yaml \
    -f deploy/monitoring/grafana-omlx-dashboard.yaml
helm upgrade --install dynamo-platform \
    https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.5.1.tgz \
    -n dynamo-system --create-namespace --wait --timeout 10m
```

- Install monitoring first. The platform chart creates its PodMonitors only when their CRD
  exists; otherwise rerun its `helm upgrade` with `--reuse-values`.
- The platform's defaults install only the operator. Dynamo then uses Kubernetes for discovery,
  TCP for requests and ZMQ for KV events, so it needs neither NATS nor etcd.
- The values file serves Grafana at `http://<node>/grafana` and leaves out the control-plane
  scrape targets that k3s does not expose. Keep those targets on other clusters.
- The pods pull about 4 GB from nvcr.io. Over a slow link, import image archives on each node
  instead, for example `sudo k3s ctr -n k8s.io images import <tar>` on k3s.

### Deploy

Use the overlay of the model the Macs serve, `qwen3.8-flash-next` after [Mac Setup](#mac-setup).
Save the Macs' `OMLX_API_KEY` in a file on Linux with mode 600, here
`~/.config/dynamo-omlx/api_key`. Each Mac then gets a Secret with its URL and that key, and
every client of the API a token of its own, here one for Claude Code and one for Codex:

```bash
OVERLAY=deploy/overlays/qwen3.8-flash-next
KEY_FILE=~/.config/dynamo-omlx/api_key
kubectl create namespace dynamo-omlx
test -s "$KEY_FILE" && cut -d= -f2- "$KEY_FILE" |
    kubectl -n dynamo-omlx create secret generic omlx-mac-1 \
        --from-literal=OMLX_BRIDGE_URL=http://<address of the Mac>:<port> \
        --from-file=OMLX_API_KEY=/dev/stdin --dry-run=client -o yaml |
    kubectl apply --server-side -f -

TOKENS=~/.config/dynamo-omlx/api-tokens
(umask 077 && mkdir -p "$TOKENS" && for holder in claude-code codex; do
    test -s "$TOKENS/$holder" ||
        python3 -c 'import secrets; print("omlx_" + secrets.token_urlsafe(32))' > "$TOKENS/$holder"
done)
kubectl -n dynamo-omlx create secret generic omlx-api-tokens --from-file="$TOKENS" \
    --dry-run=client -o yaml | kubectl apply --server-side -f -

kubectl kustomize --load-restrictor=LoadRestrictionsNone "$OVERLAY" |
    kubectl apply --server-side -f -
kubectl -n dynamo-omlx wait --for=condition=Ready dgd/omlx --timeout=15m
export OMLX_API_TOKEN=$(cat "$TOKENS/codex")
curl http://<node>/v1/models -H "Authorization: Bearer $OMLX_API_TOKEN"
```

- The Mac's URL is the one `service.sh start` printed.
- The key file may hold the key alone or the `OMLX_API_KEY=...` line of the Mac's settings
  file, and `cut` keeps only the key. The pipe keeps the key off the command line.
- Use the Mac's IP address or a name the cluster's DNS resolves. Pods cannot resolve `.local`
  names.
- The overlay reads the proxy code outside `deploy/`, so it needs
  `--load-restrictor=LoadRestrictionsNone`, and `kubectl apply -k` does not work for it.
- A Secret `hf-token-secret` with the key `HF_TOKEN` lifts the Hub's limits on anonymous
  downloads. It is optional.
- The proxy pod is ready once it has registered the model, and only ready proxies get requests.
  While its Mac is unreachable, the pod runs but stays unready.
- Every request through the Ingress needs a token from the Secret `omlx-api-tokens`, as
  `Authorization: Bearer <token>` or `x-api-key: <token>`; any other request gets 401. Each file
  in the token directory is one holder, and tokens shorter than 32 characters are ignored.
- The Ingress serves plain HTTP on port 80, where a token crosses the network readable. Put TLS
  in front of it before you expose it beyond a network you trust, and expose only `/v1`: the same
  port serves Grafana at `/grafana`.
- A stream waits silently for its first token, which takes long for a long prompt the Macs have
  not cached or while every Mac is busy. The frontend therefore sends an SSE comment line (`:`)
  after 15 s without data (`DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS`), so proxies that close idle
  responses keep the stream open: Cloudflare closes one after 100 s. Claude Code, Codex and the
  OpenAI SDKs skip such lines. A request without streaming gets no such line.
- Callers inside the cluster, such as the benchmark, reach the frontend Service directly and need
  no token.

Grafana's user is `admin`, and
`kubectl -n monitoring get secret prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 -d`
prints the password. The "Dynamo Dashboard" shows the frontend, and "oMLX on Dynamo" shows the
proxies and Macs.

### Coding Agents

The frontend serves the Anthropic Messages API as well (`DYN_ENABLE_ANTHROPIC_API` in
`deploy/base/dgd.yaml`), so Claude Code can run on the Macs' model. Start it with a configuration
directory of its own, which keeps your usual Claude Code login and settings out of it, and
its token:

```bash
CLAUDE_CONFIG_DIR=~/.claude-omlx \
ANTHROPIC_BASE_URL=http://<node> \
ANTHROPIC_AUTH_TOKEN=$(cat ~/.config/dynamo-omlx/api-tokens/claude-code) \
ANTHROPIC_MODEL=Qwen3.8-Flash-Next ANTHROPIC_SMALL_FAST_MODEL=Qwen3.8-Flash-Next \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
```

Codex uses the Responses API and reads its token from the variable that `env_key` names. Save
this as `~/.codex/omlx.config.toml`, then run
`OMLX_API_TOKEN=$(cat ~/.config/dynamo-omlx/api-tokens/codex) codex --profile omlx`:

```toml
model = "Qwen3.8-Flash-Next"
model_provider = "dynamo"
model_reasoning_effort = "xhigh"
model_context_window = 262144
web_search = "disabled"

[model_providers.dynamo]
name = "Dynamo oMLX"
base_url = "http://<node>/v1"
wire_api = "responses"
env_key = "OMLX_API_TOKEN"
```

- Claude Code sends its token as `Authorization: Bearer`; with `ANTHROPIC_API_KEY` instead it
  sends `x-api-key`, which works as well. A wrong token shows in Claude Code as `Not logged in`.
- Qwen3.8's chat template takes the reasoning efforts `low`, `medium` and `xhigh`, and uses
  `xhigh`, the highest, when a request names none. Codex without `model_reasoning_effort` names
  none as well; its `high` and `minimal` fail with HTTP 400. Claude Code always runs at `xhigh`
  with thinking on: Dynamo 1.5.1 passes neither its effort setting nor its thinking switch.
- Claude Code sends about 16,000 tokens of system prompt and tools with every request. A new
  session prefills all but the whole 8,192-token blocks a Mac already caches; its later turns
  reuse the whole prefix.
- `DYN_STRIP_ANTHROPIC_PREAMBLE` drops the billing header that starts Claude Code's system prompt
  and names its version and entrypoint, so all Claude Code clients share the cached prefix.
- When a turn calls several tools, Dynamo 1.5.1 sends the text between the calls as deltas of a
  block it has already closed, and Claude Code stops with
  `API Error: The response stream was malformed`. The Ingress therefore sends `/v1/messages`
  through `deploy/base/anthropic_fix.py`, which drops such text when it is whitespace and gives
  it a block of its own otherwise. Its log counts the streams it repaired.
- Codex 0.162 keeps a profile in a file of its own and refuses a `[profiles.omlx]` table in
  `config.toml`. Its hosted web search stays off, because Dynamo's Responses API takes only
  function tools.
- Tested with Claude Code 2.1.295 and Codex 0.162.0.

### Benchmark

```bash
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/benchmark |
    kubectl apply --server-side -f -
kubectl -n dynamo-omlx logs -f job/omlx-aiperf
```

The Job takes about 40 minutes. It takes the model from the `profile.env` that
`deploy/benchmark/kustomization.yaml` names, the `qwen3.8-flash-next` overlay's unless you change
it, and the workload from `deploy/benchmark/benchmark.env`:

- **Sweep.** Concurrency 1, 2, 4 and 8, each warmed up, then run for 120 seconds three times.
  AIPerf reports the mean of every metric over the runs with a 95% confidence interval and the
  coefficient of variation.
- **Shared prefix.** One request at a time for 300 seconds, each starting with the same system
  prompt of two blocks. The first request is cold; the later ones read the prefix from oMLX's
  cache.

Every Job draws a new random seed and so sends prompts that oMLX has not cached yet; the log's
first line shows it, and `SEED` in `benchmark.env` repeats a Job's prompts. Token counts come from
the server's usage, which also reports the cached prompt tokens of each request. Output lengths
vary, because the proxy rejects `ignore_eos` and `min_tokens`. Delete the Job before the next
run: `kubectl -n dynamo-omlx delete job omlx-aiperf`.

The pod's files go with it, so the Job copies AIPerf's CSV summaries into its log, each after a
`--- csv <path>` line. To save them as files:

```bash
kubectl -n dynamo-omlx logs job/omlx-aiperf > aiperf.log
awk '/^--- csv /{f = $3; gsub("/", "__", f); next} /^=== /{f = ""} f {print > f}' aiperf.log
```

`sweep__aggregate__sweep_aggregate__profile_export_aiperf_sweep.csv` compares the concurrencies,
`sweep__aggregate__concurrency_<n>__profile_export_aiperf_aggregate.csv` holds one concurrency's
confidence intervals, and the `profile_runs` files hold the single runs.

### Changes

| Task | How |
|---|---|
| Add a Mac | Create its Secret `omlx-mac-2`, uncomment `components` in the overlay's `kustomization.yaml`, and apply the overlay again. `deploy/components/mac-2` says how to add a third. |
| Switch the model | Copy `deploy/overlays/qwen3.8-flash-next` to a new overlay and set every value in its `profile.env`; `OMLX_MODEL_REVISION` is the `sha` in `.dynamo-omlx-download.json` of the model's directory on a Mac, and `OMLX_MODEL_DIR` names the image processor the model uses, as [Images](#images) describes. Delete the current overlay's objects (`kubectl kustomize ... <overlay> \| kubectl delete -f -`), recreate each Mac's Secret with the URL of its new service, apply the new overlay, and point `deploy/benchmark/kustomization.yaml` at its `profile.env`. |
| Update the proxy code or a profile | Apply the overlay again. The ConfigMap names change, so the proxies roll, and each new pod replaces the old one once it is ready. |
| Rotate the key | Recreate the Secrets, then delete the proxy pods. |
| Add or revoke an API token | Add or delete a file in `~/.config/dynamo-omlx/api-tokens`, then rerun the `kubectl create secret generic omlx-api-tokens` line of [Deploy](#deploy). The token check rereads the Secret, so the change applies within two minutes; `kubectl -n dynamo-omlx rollout restart deploy/omlx-auth` applies it at once. |
| Remove | Delete the overlay's objects and the namespace `dynamo-omlx`, then `helm uninstall dynamo-platform -n dynamo-system` and `helm uninstall prometheus -n monitoring` if nothing else uses them. |

The proxies run with `DYN_HEALTH_CHECK_ENABLED=false`: Dynamo 1.5.1's health check restarts a
worker that takes more than 3 seconds for a first token, which a long prefill on a Mac does.

| Symptom | Cause and fix |
|---|---|
| The proxy pod stays in `Init` | `kubectl -n dynamo-omlx logs <pod> -c prepare` tells why. Usually the Hub is unreachable: set `HF_ENDPOINT` in `profile.env`, or add a token. |
| `CreateContainerConfigError` | The pod's Secret `omlx-mac-<n>` is missing. |
| `0/1` and `waiting for the oMLX bridge` in the log | The pods cannot reach the Mac, or its model is still loading. |
| Restarts with `the oMLX bridge refused the proxy` | The key in the Secret is not the Mac's `OMLX_API_KEY`. |
| Restarts with `the Mac's oMLX block size is <n>, expected <m>` | The overlay is not the one of the model the Mac serves: its `EXPECT_BLOCK_SIZE` differs from the Mac's `OMLX_KV_BLOCK_SIZE`. |
| `/v1/models` is empty | No proxy is ready. |
| `401` with `authentication_error` | The request carries no token from `omlx-api-tokens`, or a wrong one. The log of `deploy/omlx-auth` names each holder it lets through and each request it refuses. |
| Claude Code stops with `API Error: The response stream was malformed` | `/v1/messages` bypasses the stream repair, or its pod is down. `kubectl -n dynamo-omlx get ingress omlx` should send that path to `omlx-anthropic-fix`, whose log, `kubectl -n dynamo-omlx logs deploy/omlx-anthropic-fix`, lists the streams it repaired. |
| `0/1` for minutes, the log ends at `Registering event channel` and `kubectl -n dynamo-omlx get dynamoworkermetadatas` lists no entry for the pod | The proxy hung while registering with Kubernetes discovery, which Dynamo 1.5.1 does now and then. Delete the pod; its replacement registers within seconds. |

## Operations

### Adding a Mac

Every Mac holds the whole model, so each one adds a replica.

1. Give the Linux host an SSH alias for it, as in [SSH Access](#ssh-access).
2. Copy this directory to it, as in [Mac Setup](#mac-setup).
3. On the new Mac, reuse the API key of the other Macs:

   ```bash
   cd ~/dynamo-omlx/mac
   OMLX_API_KEY=<key of the other Macs> ./setup.sh
   ./download_model.sh --login
   ./download_model.sh --background
   ./service.sh start
   ```

4. Add the URL that `service.sh start` printed to `OMLX_BRIDGE_URLS` on Linux, and restart
   `agg.sh`.

The new Mac must serve the same checkpoint with the same `OMLX_KV_BLOCK_SIZE`. Otherwise its
model card differs, and Dynamo does not admit it as a replica.

### SSH Access

A dedicated key and one alias per Mac keep copying and remote commands short. Create the key
once on the Linux host:

```bash
mkdir -p ~/.ssh/config.d ~/.ssh/omlx-macs
ssh-keygen -t ed25519 -N "" -f ~/.ssh/omlx-macs/id_ed25519
```

Put `Include config.d/*` on the first line of `~/.ssh/config`, then describe the Macs in
`~/.ssh/config.d/omlx-macs`:

```text
Host omlx-mac-1
    HostName <address of the Mac>
    User <user on the Mac>

Host omlx-mac-*
    IdentityFile ~/.ssh/omlx-macs/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
```

`ssh-copy-id -i ~/.ssh/omlx-macs/id_ed25519.pub omlx-mac-1` installs the key on a Mac and asks
for that Mac's password once. Each further Mac needs only its own `Host` block.

### Routine Tasks

Run these in `~/dynamo-omlx/mac` on the Mac.

| Task | Command |
|---|---|
| Check the service | `./service.sh status`; `./service.sh logs` follows the log |
| Follow a background download | `tail -f ~/Library/Logs/dynamo-omlx/download-<model id>.log` |
| Stop a background download | `pkill -f "download_model.py <repo>"`; rerunning `download_model.sh` resumes it |
| Save or replace the Hugging Face token | `./download_model.sh --login` |
| Change a setting | Edit `~/.config/dynamo-omlx/omlx-mac.env`, then `./service.sh restart` |
| Update this example | Copy the directory again, then `./service.sh restart` |
| Upgrade oMLX | `brew update && brew upgrade omlx`, or the oMLX app's updater; then `./setup.sh` and `./service.sh restart` |
| Fetch a newer model revision | `./download_model.sh --update`, then `./service.sh restart` |
| After a reboot | `./service.sh start` |
| Remove the example | `./uninstall.sh`, see [Uninstall](#uninstall) |

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `You are sending unauthenticated requests to the HF Hub` | The download has no Hugging Face token, so the Hub limits it. Run `./download_model.sh --login`, or set `HF_TOKEN` in the settings file, then rerun the download; it resumes. |
| `the Hub rejected the Hugging Face token` | The token expired or was revoked. Create a new one and run `./download_model.sh --login`. |
| The download stalls or keeps retrying | Wi-Fi with packet loss is the usual cause, so use Ethernet. Otherwise set `HF_ENDPOINT` to a mirror, or `HF_HUB_DISABLE_XET=1` when Hugging Face's Xet storage endpoints are unreachable. |
| `port 8000 is taken` | Another oMLX server runs there, such as the oMLX app or `omlx start`. Quit the app, run `omlx stop` or `brew services stop omlx`, or change `OMLX_PORT`. |
| `start failed and the service is stopped` | oMLX exited while starting, and the log lines printed with the error say why. Common causes are an `OMLX_BRIDGE_MODEL_ID` that differs from the model's directory name, and a model larger than the GPU working-set limit that `setup.sh` reports. |
| A proxy on Linux never registers | Check that `./service.sh status` on its Mac shows the model `ready`, and that the Linux `OMLX_API_KEY` matches the Mac's; otherwise the proxy logs `the bridge rejected the API key`. If the macOS firewall is on, it must accept connections for oMLX's Python. |
| The Macs report different block sizes | `OMLX_KV_BLOCK_SIZE` differs, or oMLX enlarged a smaller preset. Set the same value everywhere, then restart the services and `agg.sh`. |
| `KV: events degraded` | The bridge saw hash mismatches or repeated hook errors and withdrew its blocks; the log says `KV tracking degraded`. Generation keeps working, and `./service.sh restart` resumes the events. |
| `KV: events disabled` | oMLX runs without its SSD cache, for example with `--no-cache` in `OMLX_SERVE_ARGS`, or `OMLX_BRIDGE_ARGS` has `--no-kv-events`. |

### Uninstall

First take the Mac out of `OMLX_BRIDGE_URLS` on Linux and restart `agg.sh`, so its proxy stops
waiting for it. Then, on the Mac:

```bash
cd ~/dynamo-omlx/mac
./uninstall.sh                    # the service, settings and logs; the model stays
./uninstall.sh --model --cache    # also the model and the SSD prefix cache
./uninstall.sh --all              # also the Hugging Face token and oMLX itself
cd ~ && rm -rf ~/dynamo-omlx      # last, the scripts themselves
```

`uninstall.sh` lists what it removes and asks before it starts; `--yes` skips the question.

| Flag | Removes |
|---|---|
| none | Stops the service and background downloads, then removes the settings file, its pid file and `~/Library/Logs/dynamo-omlx` |
| `--model` | The downloaded model and any partial download of it |
| `--cache` | `OMLX_SSD_CACHE_DIR`, which every oMLX model on the Mac shares |
| `--hf-token` | The saved Hugging Face token, which every Hugging Face tool on the Mac shares |
| `--omlx` | oMLX, when Homebrew installed it; quit the oMLX app and move it to the Trash yourself |

- It refuses to remove paths beyond the example's own files, such as a cache directory set to
  the home directory.
- oMLX's own data in `~/.omlx`, such as its settings and other models, stays. That includes
  multi-token prediction turned on with `OMLX_MTP=on`; `OMLX_MTP=off` and
  `./service.sh restart` before uninstalling turn it off.
- On Linux, delete the model metadata directory and the Mac's `Host` block in
  `~/.ssh/config.d/omlx-macs` once nothing uses them.

## Observability

All Dynamo metrics work as with any other backend: `dynamo_frontend_*` (TTFT, ITL, request
counts and durations, `cached_tokens`), `dynamo_component_*`, and the router metrics.

Each proxy also exposes oMLX state on its `/metrics` (`DYN_SYSTEM_PORT`, starting at 8081 in
`agg.sh`). oMLX has no Prometheus endpoint, so the proxy polls the bridge's `/stats` every second:

| Metric | Meaning |
|---|---|
| `omlx_bridge_up`, `omlx_bridge_rtt_seconds` | Bridge reachability and poll latency |
| `omlx_model_loaded`, `omlx_running_requests`, `omlx_waiting_requests` | oMLX engine state |
| `omlx_ssd_cache_bytes`, `omlx_ssd_cache_capacity_bytes`, `omlx_ssd_cache_files` | SSD prefix cache usage |
| `omlx_bridge_kv_live_blocks`, `omlx_bridge_kv_seq`, `omlx_bridge_kv_state{state}` | Bridge-side KV tracking |
| `omlx_bridge_kv_faults{kind}` | Hook errors and hash mismatches since the bridge started |
| `omlx_proxy_kv_announced_blocks`, `omlx_proxy_kv_sync_state{state}` | Blocks announced to the router, sync state |
| `omlx_proxy_kv_events_total`, `_snapshots_total`, `_resyncs_total`, `_reconnects_total` | KV sync activity |
| `omlx_proxy_requests_total{outcome}`, `omlx_proxy_cached_tokens_total`, `omlx_proxy_prompt_tokens_total` | Request outcomes and prefix-cache hit volume |
| `omlx_model_vision`, `omlx_proxy_images_total` | 1 when the loaded model takes images; images sent to the Mac |
| `omlx_bridge_image_requests`, `omlx_bridge_images` | Image requests and images the bridge prepared since it started |
| `omlx_bridge_vision_cache_hits`, `omlx_bridge_vision_cache_misses` | Images whose vision features came from oMLX's cache, and images encoded |
| `omlx_bridge_images_decoded` | Images whose pixels the bridge decoded; a request whose images are all cached decodes none |
| `omlx_bridge_image_token_mismatches` | Image requests the model's processor expanded to another length than the router; the router cannot match their blocks from the first image on |

The per-rank gauges `dynamo_component_total_blocks`, `gpu_cache_usage_percent` and
`kv_cache_hit_rate` are fed by the proxy:

- **Used blocks** counts the KV blocks held by in-flight requests.
- **Hit rate** is cached prompt tokens divided by prompt tokens.

## Configuration

### Mac Settings

`mac/setup.sh` writes these to `~/.config/dynamo-omlx/omlx-mac.env`. The file is sourced by
bash, and oMLX itself also reads the `OMLX_*` variables among them. Every script in `mac/`
takes `--config FILE` (or `DYNAMO_OMLX_CONFIG`) to use another settings file.

| Setting | Default | Meaning |
|---|---|---|
| `OMLX_MODEL_REPO` | `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` | Hugging Face repo that `download_model.sh` fetches |
| `OMLX_BRIDGE_MODEL_ID` | the repo name | oMLX model id, which is the model's directory name |
| `OMLX_MODEL_DIR` | `~/.omlx/models` | oMLX model directory |
| `OMLX_HOST`, `OMLX_PORT` | `0.0.0.0`, `8000` | Address the Linux proxies connect to |
| `OMLX_API_KEY` | generated by `setup.sh` | Required on a network address; the proxies send the same key |
| `OMLX_SSD_CACHE_DIR` | `~/.omlx/cache` | SSD prefix cache that the bridge mirrors into the router |
| `OMLX_SSD_CACHE_MAX_SIZE` | oMLX's default | SSD prefix cache size, for example `200GB` |
| `OMLX_KV_BLOCK_SIZE` | `8192` | Block size preset; empty lets oMLX choose |
| `OMLX_MAX_CONCURRENT_REQUESTS` | oMLX's default (8) | Requests oMLX runs at once |
| `OMLX_MTP` | empty | `on` or `off` writes oMLX's multi-token prediction to its per-model settings before the model loads; empty keeps them |
| `OMLX_MTP_DEPTH` | oMLX's default (3) | Most MTP draft tokens per verify cycle, 1 to 8; needs oMLX 0.7.0rc1 or later |
| `OMLX_BRIDGE_ARGS`, `OMLX_SERVE_ARGS` | empty | Extra bridge flags and `omlx serve` flags |
| `OMLX_START_TIMEOUT` | `1200` | Seconds `service.sh start` waits for the model to load |
| `OMLX_RESTART_DELAY` | `10` | Seconds before `service.sh` restarts oMLX; doubles up to 300 while oMLX keeps exiting within a minute |
| `OMLX_PYTHON` | detected | oMLX's interpreter (Homebrew, the oMLX app, or `python3`) |
| `HF_TOKEN` | unset | Hugging Face token; `download_model.sh --login` stores one outside this file instead |
| `HF_ENDPOINT`, `HF_HUB_DISABLE_XET` | unset | A Hugging Face mirror, and plain HTTP instead of Xet storage |
| `HF_XET_HIGH_PERFORMANCE` | `1` in `download_model.sh` | Parallel Xet transfers; `0` turns them off |

### Bridge and Proxy Flags

`mac/service.sh` builds the bridge command from the settings above. Bridge flags go before
`--`, and the unchanged `omlx` command goes after it:
`python -m dynamo_omlx.bridge [flags] -- serve [omlx flags]`.

| Flag | Default | Meaning |
|---|---|---|
| `--model-id` | `OMLX_BRIDGE_MODEL_ID` | oMLX model id served to Dynamo |
| `--kv-block-size` | unset | Preset for oMLX's paged cache block size; oMLX may still enlarge it |
| `--no-kv-events` | off | Serve tokens only |
| `--no-images` | off | Refuse images, even for a vision-language model |
| `--no-pin`, `--no-preload` | off | Leave pinning and loading to oMLX |
| `--no-verify-hashes` | off | Skip recomputing oMLX block hashes before announcing |
| `--audit-interval` | 30 s | Check that announced blocks still exist on SSD |
| `--ring-max-events`, `--ring-max-bytes` | 100000, 64 MiB | Replay ring size |
| `--mtp`, `--mtp-depth` | unset | Write multi-token prediction (`on` or `off`) and its maximum draft depth to oMLX's per-model settings |
| `--check` | off | Report oMLX compatibility and exit; `no image input:` lines name what image input lacks |

The proxy accepts all Dynamo runtime flags (`--namespace`, `--discovery-backend`,
`--dyn-tool-call-parser`, and so on), plus these:

| Flag | Default | Meaning |
|---|---|---|
| `--bridge-url` | `OMLX_BRIDGE_URL` | The Mac's oMLX base URL |
| `--bridge-api-key` | `OMLX_API_KEY` | oMLX API key |
| `--model-path` | required | Config, tokenizer and template directory (no weights) |
| `--served-model-name` | directory name | Model name clients request |
| `--context-length` | `config.json` | Override `max_position_embeddings` |
| `--expect-block-size` | unset | Refuse to start unless the Mac uses this block size |
| `--total-kv-blocks` | derived | KV capacity to advertise |
| `--max-output-tokens` | 32768 | Cap when a request sets no `max_tokens` |
| `--no-kv-events` | off | Route by load only |
| `--mac-down-grace` | 30 s | Exit (and deregister) after the Mac is unreachable this long |
| `--stats-interval` | 1 s | Seconds between bridge stats polls |
| `--generate-read-timeout` | 60 s | Fail a generation after this long without a frame; the bridge sends heartbeats every 5 s |
| `--startup-timeout` | 0 | Seconds to wait for the model (0: wait forever) |
| `--max-image-bytes` | 50 MiB | Largest image file a request may carry or link to; oMLX's `max_image_upload_size` applies too |
| `--image-fetch-timeout` | 30 s | Seconds to fetch an `http(s)` image |

## Failure Behavior

| Event | What happens |
|---|---|
| Network blip | The proxy resumes the KV stream from its last sequence number. Generations that were in flight fail with a migratable error. |
| Bridge restart | The proxy sees a new bridge incarnation and reconciles the router from a snapshot. |
| oMLX exits on a Mac | `service.sh` starts it again after 10 seconds. While oMLX keeps exiting within a minute of starting, the delay doubles up to 5 minutes. The proxy then reconciles as after a bridge restart. |
| Mac unreachable | After `--mac-down-grace` the proxy exits and deregisters, so the frontend stops routing to that Mac. `agg.sh` restarts the proxy in its own loop, and on Kubernetes the pod's container restarts; the other Macs stay untouched. The new proxy waits for its Mac, then announces the Mac's cache from a snapshot. |
| Mac restarts | Nothing starts the service again until `./service.sh start` runs. Until then the proxy treats the Mac as unreachable. |
| Router restart | Dynamo rebuilds the router's index from the proxy's local indexer. |
| oMLX evicts or deletes blocks | Removals reach the router within milliseconds. A periodic audit also catches any removal that bypassed the hooks. |
| oMLX block size changes | The proxy exits, because its model card no longer matches. Restart it after checking that every Mac uses the same settings. |
| Hash mismatch or repeated hook errors | The bridge withdraws every block and stops announcing until it restarts. Generation keeps working. |

## Limitations

- **Text and images, aggregated serving only.**
  - No video, audio, frontend-decoded media or prefill/decode disaggregation. Images need a
    vision-language model from the Qwen-VL family, whose processor expands each image
    placeholder by a patch grid.
  - Requests using `ignore_eos`, `min_tokens`, a thinking budget, `n > 1`, beam search, guided
    decoding, logprobs, LoRA or `cache_salt` are rejected. oMLX cannot honor them through this
    path, and it cannot isolate prefix caches per salt.
- **The router knows images by URL, oMLX by file.**
  - The same picture in two files, for example re-encoded, is two images to both, and one URL
    whose content changed is one image to the router. The Mac's cache stays correct either
    way; the router's estimate may not.
  - Images sent to oMLX's own API reach the router only as the text before their first image.
- **Blocks become visible only after the request finishes.**
  - oMLX commits prefix blocks when a request finishes, so the router learns a prefix only then.
  - `agg.sh` sets `--router-predicted-ttl-secs` so the router can anticipate concurrent requests
    that share a prefix.
- **Cache kept across a bridge restart is not announced right away.**
  - oMLX's SSD metadata has no token ids, so the bridge cannot announce those blocks after it
    restarts.
  - They are announced again as soon as a completed request reuses them.
- **Cache tiers are not distinguished.** oMLX's RAM and SSD tiers are announced as one tier,
  since restoring from SSD is far cheaper than a Mac prefill.
- **Qwen3.8 and other hybrid models can overstate reuse.** oMLX keeps a block's
  linear-attention state in a separate SSD sidecar and can evict it separately from the block.
  The router may then credit a prefix whose state is gone, and that request pays for a prefill
  the router expected it to skip. A larger `OMLX_SSD_CACHE_MAX_SIZE` makes this rarer.
- **Dedicate the Mac to one model.** Other models sharing oMLX's SSD cache directory can evict
  its files.

## Tests

```bash
python3 -m pytest examples/backends/omlx/tests
```

The tests run on Linux without oMLX or Dynamo:

- **Bridge.** Tested against fakes shaped like oMLX's classes.
- **Proxy.** Tested against an HTTP fake of the bridge.
- **Model download.** `mac/download_model.py` is tested against a fake Hub.
- **KV pipeline.** A randomized test checks it against the router's rules.
- **Stream repair.** `deploy/base/anthropic_fix.py` is tested against the stream Dynamo 1.5.1
  sends for a turn with two tool calls, and over HTTP against a fake frontend.
- **Token check.** `deploy/base/token_auth.py` is tested over HTTP against a directory laid
  out like a mounted Secret.
- **Kubernetes manifests.** The proxy code that `deploy/` ships is checked against the package,
  and the overlays are rendered where `kubectl` is installed.

`test_omlx_proxy_engine.py` and `test_omlx_proxy_args.py` run only where `ai-dynamo` is
installed.
