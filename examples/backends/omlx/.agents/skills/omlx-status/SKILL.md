---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-status
description: Shows the state of the oMLX-on-Mac example's Kubernetes deployment without changing it - the nodes, the Dynamo platform and monitoring releases, the DynamoGraphDeployment and its pods, the served model, each Mac's bridge, load, SSD cache and KV sync, the frontend's traffic and what needs attention - then where to reach the API, Grafana and Prometheus, and how to call the OpenAI-compatible API with curl and the openai Python SDK, including streaming, thinking control, tool calls and the parameters the Macs reject, and the Claude Code and Codex settings. Use when asked what is deployed, whether it is healthy, how to reach it, or how to send requests to the model or run a coding agent on it.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - kubernetes
    - status
    - openai
---

# Skill: Show the oMLX Example's Status

The Linux side runs on Kubernetes as the DynamoGraphDeployment `omlx` in the namespace
`dynamo-omlx`: the frontend with KV routing and one proxy pod per Mac (`omlx-dynamo-install`).
This skill reports its state, how to reach it and how to call it. Paths below are relative to
`examples/backends/omlx`.

## Ground Rules

- Change nothing. Commands that delete, restart or clean up are suggestions for the operator.
- Never print the Macs' API key, an API token, a Hugging Face token or Grafana's password, and
  hand a token to curl only on stdin (`-K -`). `status.py` reads `OMLX_BRIDGE_URL` and the names
  of the token holders, nothing else from Secrets; never read a Secret with `-o yaml`, `-o json`
  or `describe`. For Grafana, give the operator the command that prints the password.
- Send no request while a Job such as `omlx-aiperf` runs: it competes for the Macs and skews
  the benchmark.
- Mac-side commands are for the operator to run on the Mac.

## 1. Collect

```bash
python3 .agents/skills/omlx-status/scripts/status.py   # --namespace, --name, --context
```

It needs kubectl with read access, exec into the pods (it curls their own metrics ports), and
optionally helm. It takes a few seconds and prints Markdown that ends in `## Findings`; exit
code 1 means kubectl is missing or cannot reach the cluster.

| Section | What it shows |
|---|---|
| Cluster | Nodes with CPU and memory in use, the `dynamo-platform` and `kube-prometheus-stack` releases with their pods |
| Deployment, Pods | The DGD's Ready condition; each pod's status, restarts with the last exit, CPU, memory, and whether it registered with Kubernetes discovery |
| Model | The `omlx-profile-*` ConfigMap in use, matched to `deploy/overlays/*/profile.env`, and the context, block size and `max_num_seqs` the frontend registered |
| Macs | Per Mac: the bridge URL from its Secret; bridge reachability and stats-poll latency, model loaded, oMLX running and waiting requests, SSD prefix cache, KV sync states, requests and cache hits since the proxy started |
| Traffic | The frontend's requests since it started, by status and endpoint; in-flight and queued requests; the share of prompt tokens served from the Macs' caches |
| Access | The URLs of the API, of the Anthropic Messages API for Claude Code when the frontend serves it, and of Grafana from their Ingresses; the frontend Service; the Prometheus port-forward |
| Jobs, Warning Events | The namespace's Jobs and its last five Warning events |

## 2. Smoke Test

Skip it while a Job runs. Otherwise one short request proves the whole path from the Ingress
and its token check through the router and the proxy to a Mac, with a token from the
operator's files:

```bash
printf 'header = "Authorization: Bearer %s"\n' "$(cat ~/.config/dynamo-omlx/api-tokens/<holder>)" |
    curl -s -K - -w '\nHTTP %{http_code} in %{time_total}s\n' <api>/chat/completions \
    -H 'Content-Type: application/json' -d '{"model": "<model>", "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hello"}],
    "chat_template_kwargs": {"enable_thinking": false}}'
```

`<api>` is the OpenAI API URL under Access, `<model>` the served name, `<holder>` one of the
holders it lists. Expect HTTP 200 and an
answer within seconds; while every Mac runs `max_num_seqs` requests, it waits in the queue.

## 3. Present

Answer in the operator's language, in three parts:

1. **Resource status**: the summary line, the tables and the Mac lines, without rows that add
   nothing; the smoke test's result; every finding with its cause and fix from the table below.
2. **Access**: the Access table, plus Grafana's password command
   `kubectl -n monitoring get secret prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 -d`
   and its dashboards "Dynamo Dashboard" (frontend) and "oMLX on Dynamo" (proxies and Macs).
   On a Mac, `cd ~/dynamo-omlx/mac && ./service.sh status` shows the service.
3. **OpenAI API**: the requests below with `<api>` and `<model>` filled in, and the model's
   limits from the report; for Claude Code or Codex, the Coding Agents settings.

| Finding | Cause and fix |
|---|---|
| No `dynamo-platform` or `kube-prometheus-stack` release | Not installed: `omlx-dynamo-install` steps 3 and 4 |
| A release with pods that are not ready | `kubectl -n <namespace> get pods`, then the failing pod's events and log |
| No DynamoGraphDeployment | Another namespace or name (`--namespace`, `--name`), or not deployed: `omlx-dynamo-install` step 6 |
| Pod not ready: `Init:...` | `kubectl -n dynamo-omlx logs <pod> -c prepare`; usually the Hub is unreachable |
| Pod not ready: `CreateContainerConfigError`, or a Secret is missing | The Mac's Secret `omlx-mac-<n>`: `omlx-dynamo-install` step 5 |
| Pod not ready: `Running` | `kubectl -n dynamo-omlx logs <pod> -c main --tail=20`. `waiting for the oMLX bridge`: the Mac is unreachable or still loading. With `unregistered` and a log that ends at `Registering event channel`: Dynamo 1.5.1's discovery hang; deleting the pod fixes it |
| Pod not ready: `CrashLoopBackOff` or `Error` | `kubectl -n dynamo-omlx logs <pod> -c main --previous`. `the oMLX bridge refused the proxy`: the Secret's key is not the Mac's. `block size ... expected`: the overlay does not match the model the Mac serves |
| The proxy cannot reach the Mac's bridge | `curl -s -o /dev/null -m 5 -w '%{http_code}\n' <bridge url>/dynamo/v1/info` from Linux: 401 means the Mac answers, 000 that it does not, because it sleeps, left the network or its service stopped. After 30 seconds the proxy exits, and the pod restarts until the Mac is back |
| oMLX has not loaded the model | It is loading or was unloaded: `./service.sh status` and `./service.sh logs` on the Mac |
| KV events are not live | Bridge `degraded`: hash mismatches or hook errors, `./service.sh restart` on the Mac resumes them. Bridge `disabled`: oMLX runs without its SSD cache. Proxy sync `connecting` or `syncing`: passes after either side restarted |
| Requests failed on the Mac | The errors in the proxy's log: `kubectl -n dynamo-omlx logs <pod> -c main --since=1h` |
| Requests failed at the frontend | `validation`: the frontend or the chat template refused the request, such as `reasoning_effort: high`. `internal` includes the proxy's 400 for unsupported parameters; nothing failed on a Mac when the Macs count no `error`. The `Backend error detected` lines of `kubectl -n dynamo-omlx logs deploy/omlx-frontend --since=1h` show the messages. The counts cover the frontend's whole uptime |
| Metrics did not answer, or no model is served | The pod is starting or stuck, or no proxy is ready: its log, and the Macs section. Right after a node reboot, a frontend log with `endpoint registration lease conflicts with a pre-existing specification` means deleting the frontend pod (`omlx-dynamo-install`, After a Node Reboot) |
| A Job is running | Skip the smoke test until it finishes |
| A Job failed | `kubectl -n dynamo-omlx logs job/<name>` |
| ConfigMaps from earlier rollouts | Harmless leftovers; the operator may run the printed delete |
| No Ingress routes to the frontend | Use the port-forward under Access |
| The API Ingress runs no token check | The Ingress lacks the `omlx-auth` middleware: apply the overlay again (`omlx-dynamo-install` step 6) |
| The Secret `omlx-api-tokens` holds no token | Every request through the Ingress gets 401: create tokens as in `omlx-dynamo-install` step 5 |
| Pod `omlx-auth-...` not ready | The token check is down, so requests through the Ingress fail: `kubectl -n dynamo-omlx logs deploy/omlx-auth` |
| Pod `omlx-anthropic-fix-...` not ready | The stream repair in front of `/v1/messages` is down, so Claude Code's requests through the Ingress fail: `kubectl -n dynamo-omlx logs deploy/omlx-anthropic-fix` |

## OpenAI API

Verified with Dynamo 1.5.1 and Qwen3.8-Flash-Next.

- Every request needs a token: `Authorization: Bearer <token>` (OpenAI SDKs, Codex) or
  `x-api-key: <token>`. The operator keeps one file per holder in
  `~/.config/dynamo-omlx/api-tokens/`; the examples read it from `OMLX_API_TOKEN`, set with
  `export OMLX_API_TOKEN=$(cat ~/.config/dynamo-omlx/api-tokens/<holder>)`.
- The Ingress serves plain HTTP, so a token crosses the network readable. Before the operator
  exposes it beyond a trusted network, TLS belongs in front, and only `/v1` should be reachable:
  the same port serves Grafana.
- Endpoints under `<api>`: `GET /models`, `POST /chat/completions`, `POST /completions` (a raw
  prompt without the chat template) and `POST /responses`. The model registers for chat and
  completions only.
- The `qwen3` reasoning parser puts the thinking in `reasoning_content` (in a stream,
  `delta.reasoning_content`) and the answer in `content`. The `qwen3_coder` tool-call parser
  turns tool calls into `tool_calls`.

```bash
curl -s <api>/models -H "Authorization: Bearer $OMLX_API_TOKEN"
curl -s <api>/chat/completions -H "Authorization: Bearer $OMLX_API_TOKEN" \
    -H 'Content-Type: application/json' -d '{
  "model": "<model>",
  "messages": [{"role": "user", "content": "Explain prefix caching in one sentence."}],
  "max_tokens": 1024
}' | jq '{reasoning: .choices[0].message.reasoning_content, answer: .choices[0].message.content, usage}'
```

Streaming with thinking off; the last chunk before `[DONE]` carries the usage. While a stream
waits 15 s or more for data, the frontend sends a `:` comment line, which clients skip:

```bash
curl -sN <api>/chat/completions -H "Authorization: Bearer $OMLX_API_TOKEN" \
    -H 'Content-Type: application/json' -d '{
  "model": "<model>",
  "messages": [{"role": "user", "content": "Count to five."}],
  "max_tokens": 64,
  "stream": true,
  "stream_options": {"include_usage": true},
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Tool calls:

```bash
curl -s <api>/chat/completions -H "Authorization: Bearer $OMLX_API_TOKEN" \
    -H 'Content-Type: application/json' -d '{
  "model": "<model>",
  "messages": [{"role": "user", "content": "What is the weather in Shanghai?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 1024
}' | jq '.choices[0] | {finish_reason, tool_calls: .message.tool_calls}'
```

`finish_reason` is `tool_calls`, and each call's `arguments` is a JSON string. Send the result
as `{"role": "tool", "tool_call_id": "<id>", "content": "..."}` after the assistant message
that carries the `tool_calls`, with the same `tools`.

Images, while the proxies' `omlx_model_vision` is 1, go in `image_url` parts as a base64 `data:`
URL or an `http(s)` URL on a public address:

```bash
jq -n --arg url "data:image/jpeg;base64,$(base64 -w0 photo.jpg)" '{model: "<model>",
  max_tokens: 1024, messages: [{role: "user", content: [
    {type: "image_url", image_url: {url: $url}}, {type: "text", text: "What is in it?"}]}]}' |
curl -s <api>/chat/completions -H "Authorization: Bearer $OMLX_API_TOKEN" \
    -H 'Content-Type: application/json' -d @- | jq -r '.choices[0].message.content'
```

- An image costs about width × height / 1024 prompt tokens, at most `OMLX_IMAGE_MAX_PIXELS`
  / 1024 of the overlay's `profile.env`: 4,096 for Qwen3.8. Larger images are scaled down.
- Sending an image file again reuses the Mac's prefix cache and vision features for it, and
  the router sends it to the Mac that holds them; `usage.prompt_tokens_details.cached_tokens`
  counts the image's tokens too.

```bash
curl -s <api>/completions -H "Authorization: Bearer $OMLX_API_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"model": "<model>", "prompt": "The capital of France is", "max_tokens": 16}'
curl -s <api>/responses -H "Authorization: Bearer $OMLX_API_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"model": "<model>", "input": "Say hi in French.", "max_output_tokens": 256}'
```

Python, after `pip install openai`:

```python
import os

from openai import OpenAI

client = OpenAI(base_url="<api>", api_key=os.environ["OMLX_API_TOKEN"])
response = client.chat.completions.create(
    model="<model>",
    messages=[{"role": "user", "content": "Explain prefix caching in one sentence."}],
    max_tokens=1024,
)
message = response.choices[0].message
print(getattr(message, "reasoning_content", None))  # not a typed field of the SDK
print(message.content)

stream = client.chat.completions.create(
    model="<model>",
    messages=[{"role": "user", "content": "Count to five."}],
    max_tokens=64,
    stream=True,
    # Fields the SDK does not know go in extra_body.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Tools built on the SDK take `OPENAI_BASE_URL=<api>` and the token as `OPENAI_API_KEY`.

| Request field | Behavior |
|---|---|
| `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `stop` | Passed to oMLX; unset ones keep oMLX's defaults for the model |
| `max_tokens`, `max_completion_tokens`; `max_output_tokens` on `/responses` | Unset: up to 32768, never past the context |
| Thinking | On by default. Off: `"chat_template_kwargs": {"enable_thinking": false}`, `"thinking": {"type": "disabled"}` or `"reasoning_effort": "none"`. Qwen3.8's chat template takes `reasoning_effort` `low`, `medium` and `xhigh`, its default when a request names none, and answers 400 to `high` or `minimal`. Through `/v1/messages` (Claude Code) the effort and the thinking switch are ignored: always `xhigh` |
| Images in `image_url` parts | Passed to oMLX while the model takes them, else 400 `the model on the Mac takes no images` |
| `logprobs`, `n` above 1, `response_format` (JSON mode and schemas), `ignore_eos`, `min_tokens`, a thinking budget, LoRA adapters, `cache_salt`, video, audio | 400 `the oMLX backend does not support: ...` |

- The prompt must leave room in the context, 262144 tokens for Qwen3.8; the proxy rejects a
  longer one.
- Each Mac runs `max_num_seqs` requests at once, 8 for Qwen3.8; more wait in oMLX's queue.
- `usage.prompt_tokens_details.cached_tokens` counts the prompt tokens a Mac reused. The router
  routes by prefix only for prompts of at least one block, 8192 tokens for Qwen3.8, and a Mac
  caches a prefix once the request that carried it finishes. Send a long shared prefix once
  before fanning out the requests that reuse it.

## Coding Agents

Verified with Claude Code 2.1.295 and Codex 0.162.0; `<node>` is the host in `<api>`.

Claude Code calls the Anthropic Messages API, which Access lists when the frontend sets
`DYN_ENABLE_ANTHROPIC_API`, as `deploy/base/dgd.yaml` does. Give it its own config directory, so
the operator's usual Claude Code login and settings stay untouched, and a token:

```bash
CLAUDE_CONFIG_DIR=~/.claude-omlx \
ANTHROPIC_BASE_URL=http://<node> \
ANTHROPIC_AUTH_TOKEN=$(cat ~/.config/dynamo-omlx/api-tokens/<holder>) \
ANTHROPIC_MODEL=<model> ANTHROPIC_SMALL_FAST_MODEL=<model> \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude
```

- `ANTHROPIC_BASE_URL` has no `/v1`. The same variables in `~/.claude/settings.json` would move
  every Claude Code session onto the Macs.
- `ANTHROPIC_AUTH_TOKEN` goes out as `Authorization: Bearer`, `ANTHROPIC_API_KEY` as
  `x-api-key`; both pass. Claude Code reports a wrong token as `Not logged in`.
- Claude Code calls the model unrecognized and assumes a 200,000-token context. Dynamo 1.5.1
  passes neither its effort setting nor its thinking switch, so every request runs at `xhigh`,
  the highest effort, with thinking on.
- Each request carries about 16,000 tokens of system prompt and tools. A Mac keeps only whole
  blocks (8,192 tokens for Qwen3.8) across sessions, so a new session reuses the first block;
  the turns of one session reuse the whole prefix.
- The frontend strips the billing header that starts Claude Code's system prompt
  (`DYN_STRIP_ANTHROPIC_PREAMBLE`); without it, set `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.
- `/v1/messages` passes `omlx-anthropic-fix`, which repairs the streams of turns with several
  tool calls: Dynamo 1.5.1 sends the text between the calls to a closed block, and Claude Code
  stops with `API Error: The response stream was malformed`. That error means the request
  bypassed the repair, or its pod is down.

Codex calls the Responses API and reads its token from the variable that `env_key` names. Save
this as `~/.codex/omlx.config.toml` and run
`OMLX_API_TOKEN=$(cat ~/.config/dynamo-omlx/api-tokens/<holder>) codex --profile omlx`, with
`<holder>` one of the holders Access lists, ideally one per client so that each can be revoked
alone:

```toml
model = "<model>"
model_provider = "dynamo"
model_reasoning_effort = "xhigh"
model_context_window = 262144
web_search = "disabled"

[model_providers.dynamo]
name = "Dynamo oMLX"
base_url = "<api>"
wire_api = "responses"
env_key = "OMLX_API_TOKEN"
```

- Codex 0.162 refuses a `[profiles.omlx]` table in `config.toml`: a profile is a file of its own.
- `xhigh` is the highest effort Qwen3.8 takes; without the line Codex names no effort, which
  gets `xhigh` too. `low` and `medium` answer sooner; `high` and `minimal` fail with HTTP 400.
- Keep `web_search` off: Dynamo's Responses adapter takes function tools only. Codex warns that
  the model's metadata is missing, which is harmless; on Linux its sandbox needs bubblewrap.

## Without Kubernetes

With `launch/agg.sh` there is no cluster to inspect. The API is at
`http://<Linux host>:8000/v1`, each proxy's metrics are at port 8081 and up
(`curl -s localhost:8081/metrics | grep -E '^omlx_'`), and the OpenAI API section applies
unchanged.
