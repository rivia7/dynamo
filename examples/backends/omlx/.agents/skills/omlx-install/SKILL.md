---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-install
description: Installs the oMLX-on-Mac example behind Dynamo. Prepares each Mac (oMLX, settings and API key, Hugging Face token, model download, background service), then starts the Linux side (model metadata, frontend with KV routing, one proxy per Mac) and verifies it end to end. Use when setting the example up for the first time, adding a Mac, or reinstalling after omlx-uninstall.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - macos
    - install
---

# Skill: Install the oMLX Example

The example serves a model with oMLX on Macs while the Dynamo frontend, KV router and one proxy
per Mac run on Linux. Each Mac needs only this example directory and oMLX; Linux needs a Dynamo
environment built from this repository. Paths below are relative to `examples/backends/omlx`.

## Ground Rules

- Mac steps run on the Mac. When you are not on the Mac, give the operator the exact commands;
  run them over SSH only when the operator asks you to.
- Never print, log or put on a command line the API key (`OMLX_API_KEY`) or a Hugging Face
  token. Point the operator to where they are stored instead.
- The default model is 99 GiB. Check disk, memory and network before starting the download.

## Inputs

Settle these with the operator first:

| Input | Default |
|---|---|
| The Macs: SSH alias or address of each | none |
| First Mac, or one more for an existing deployment | first: `setup.sh` creates the API key; more: reuse the key |
| Hugging Face repo | `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` (oMLX model id `Qwen3.8-Flash-Next-oQ4e-mtp`) |
| Served model name and Linux metadata directory | `Qwen3.8-Flash-Next`, `/models/Qwen3.8-Flash-Next` |
| A Hugging Face read token | strongly recommended; the Hub limits anonymous downloads |
| Multi-token prediction (MTP) | off; `OMLX_MTP=on` in a Mac's settings file turns it on there |

## 1. Copy the Example to the Mac

From the repository root on Linux:

```bash
rsync -a --exclude tests --exclude deploy --exclude __pycache__ examples/backends/omlx/ <mac>:dynamo-omlx/
```

Any other copy works too; the Mac needs only this directory, at `~/dynamo-omlx` below.

## 2. Set Up the Mac

```bash
cd ~/dynamo-omlx/mac
./setup.sh                     # first Mac: creates the settings file and API key
OMLX_API_KEY=<key> ./setup.sh  # each further Mac: the operator supplies the shared key
```

Add `--install` when oMLX is missing; it installs oMLX with Homebrew. Read the report:

| Line | Proceed only if |
|---|---|
| `oMLX is compatible with the Dynamo bridge` | present; otherwise this oMLX build lacks a hook the bridge needs |
| `GPU:` | the working-set limit exceeds the model size plus room for the KV cache |
| `Disk:` | free space exceeds the model size plus the SSD cache |
| `Kernels:` | informational; without native kernels prefill is slower |
| `Network:` | a Wi-Fi warning means downloads and token streams suffer; suggest Ethernet |

`setup.sh` writes `~/.config/dynamo-omlx/omlx-mac.env` (mode 600) and never overwrites it.
Change the model, port, block size or cache size by editing that file. Every Mac behind one
model needs the same model, `OMLX_KV_BLOCK_SIZE` (8192) and `OMLX_API_KEY`.

## 3. Download the Model

```bash
./download_model.sh --login       # the operator pastes the token; input stays hidden
./download_model.sh --background
tail -f ~/Library/Logs/dynamo-omlx/download-<model id>.log
```

- Run `--login` in an interactive terminal. Never pipe a token into it or pass one as an argument.
- The log starts with `Hub: <endpoint>, signed in as <account>` and prints a progress line every
  30 seconds. `the Hub rejected the Hugging Face token` means a new token and `--login` again.
- Rerunning resumes. Finished files are kept; a file in flight restarts, because Xet storage
  writes a file only once all of it has arrived. Avoid needless restarts.
- When the download stalls or keeps retrying, suggest Ethernet, then `HF_ENDPOINT` (a mirror)
  or `HF_HUB_DISABLE_XET=1` in the settings file.
- The model appears in `~/.omlx/models/<model id>` only after every file's size checks out.

## 4. Start the Service

```bash
./service.sh start
```

It waits until oMLX has loaded the model, then prints `Model: <id> ready, block size <n>`,
`KV: events live`, `MTP: off` or `MTP: on, at most <n> draft tokens per cycle`, and the URL for
Linux. Afterwards `./service.sh status`, `logs`, `stop` and `restart` manage it; the service
restarts when oMLX exits, but not after a reboot.

| Message | Cause |
|---|---|
| `port <n> is taken` | the oMLX app or `omlx start` runs there: quit it, `omlx stop` or `brew services stop omlx` |
| `start failed and the service is stopped` | read the printed log lines; often a wrong `OMLX_BRIDGE_MODEL_ID` or too little GPU memory |
| `is still downloading` | the download has not finished |
| `rules out MTP` or `choose one speculative` in the log | `OMLX_MTP=on` while oMLX runs another speculative decoder (DFlash or VLM MTP) for the model: start once without `OMLX_MTP=on`, turn that decoder off on oMLX's admin page, then set `OMLX_MTP=on` again |

## 5. Start the Linux Side

To run the Linux side on a Kubernetes cluster instead, continue with `omlx-k8s-deploy`.

From the repository root, in the Dynamo environment:

```bash
python3 examples/backends/omlx/mac/download_model.py <repo> --metadata-only \
    --model-dir /models --name <served name>
export OMLX_API_KEY=...   # the operator sets it from the Mac's settings file
OMLX_BRIDGE_URLS="http://<mac-1>:8000 http://<mac-2>:8000" \
    examples/backends/omlx/launch/agg.sh --model-path /models/<served name>
```

- The URLs are the ones `service.sh start` printed.
- `agg.sh` runs the frontend on port 8000 and the proxies' metrics from port 8081 up. It
  defaults to the `qwen3_coder` and `qwen3` parsers; set `TOOL_CALL_PARSER` and
  `REASONING_PARSER` for another model family.
- Every proxy of one model needs the same metadata at the same path.

## 6. Verify

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "<served name>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
curl -s localhost:8081/metrics | grep -E '^omlx_(bridge_up|model_loaded|proxy_kv_sync_state)'
```

Expect a completion, `omlx_bridge_up 1`, `omlx_model_loaded 1` and
`omlx_proxy_kv_sync_state{state="live"} 1` on each proxy's port. A proxy that never registers
usually means a mismatched `OMLX_API_KEY` (its log says `the bridge rejected the API key`) or a
macOS firewall blocking the port. Prefix-cache routing only applies to prompts of at least one
block (8192 tokens).

## Adding a Mac

Repeat steps 1 to 4 on the new Mac with the shared API key and the same settings, add its URL
to `OMLX_BRIDGE_URLS`, and restart `agg.sh`. A Mac with a different block size or checkpoint
does not join: its model card differs.
