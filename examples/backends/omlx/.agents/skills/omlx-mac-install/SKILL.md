---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-mac-install
description: Installs the Mac side of the oMLX-on-Mac example behind Dynamo. Copies the example to each Mac, sets up oMLX with its settings and API key, saves a Hugging Face token, downloads the model, starts the background service, and hands the Mac's URL and key to the Kubernetes side. Use when setting up a Mac for the first time, adding a Mac, or reinstalling one after omlx-mac-uninstall; omlx-dynamo-install then serves the Macs from the cluster.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - macos
    - install
---

# Skill: Install the oMLX Example on a Mac

Each Mac serves the model with oMLX and needs only this example directory and oMLX; Dynamo
does not run there. The frontend, the KV router and one proxy per Mac run on a Kubernetes
cluster, which `omlx-dynamo-install` sets up. Paths below are relative to
`examples/backends/omlx`.

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
the cluster. Afterwards `./service.sh status`, `logs`, `stop` and `restart` manage it; the
service restarts when oMLX exits, but not after a reboot: then `./service.sh start` again.

| Message | Cause |
|---|---|
| `port <n> is taken` | the oMLX app or `omlx start` runs there: quit it, `omlx stop` or `brew services stop omlx` |
| `start failed and the service is stopped` | read the printed log lines; often a wrong `OMLX_BRIDGE_MODEL_ID` or too little GPU memory |
| `is still downloading` | the download has not finished |
| `rules out MTP` or `choose one speculative` in the log | `OMLX_MTP=on` while oMLX runs another speculative decoder (DFlash or VLM MTP) for the model: start once without `OMLX_MTP=on`, turn that decoder off on oMLX's admin page, then set `OMLX_MTP=on` again |

## 5. Hand Over to the Cluster

The cluster needs two things from each Mac:

- **The URL** that `service.sh start` printed, with an address the cluster's pods can reach: an
  IP address or a name the cluster's DNS resolves, never a `.local` name.
- **The key**: `OMLX_API_KEY` in `~/.config/dynamo-omlx/omlx-mac.env`. The operator copies it
  into a mode-600 file on the Linux host, for example `~/.config/dynamo-omlx/api_key`. The key
  alone or the whole `OMLX_API_KEY=...` line both work there. Never relay it through a chat or a
  command line.

From the Linux host, check that the Mac answers:

```bash
curl -s -o /dev/null -m 5 -w '%{http_code}\n' http://<address of the Mac>:<port>/dynamo/v1/info
```

`401` means the Mac answers and wants the key. `000` means the Linux host cannot reach it: the
Mac sleeps, left the network, or its firewall blocks oMLX's Python. Then continue with
`omlx-dynamo-install`: all of it for the first Mac, its Add a Mac section for a further one.

## Adding a Mac

Repeat steps 1 to 5 on the new Mac with the shared API key (`OMLX_API_KEY=<key> ./setup.sh`)
and the same model and `OMLX_KV_BLOCK_SIZE`. A Mac with a different block size or checkpoint
does not join: its model card differs.
