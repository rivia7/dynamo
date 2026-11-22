---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-mac-uninstall
description: Removes the oMLX-on-Mac example from a Mac. Stops its service and background downloads, deletes its settings and logs, and on request the downloaded model, oMLX's SSD prefix cache, the saved Hugging Face token and oMLX itself, then ends the Linux host's SSH access to that Mac. Use when decommissioning a Mac, freeing its disk, or starting it over before omlx-mac-install; take the Mac out of the cluster first with omlx-dynamo-uninstall.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - macos
    - uninstall
---

# Skill: Uninstall the oMLX Example From a Mac

Removal cannot be undone, and the default model takes hours to download again. Paths below are
relative to `examples/backends/omlx`.

## Ground Rules

- Agree the scope with the operator before removing anything. Keep the model, the cache and
  the token unless the operator asks for them to go.
- Mac steps run on the Mac. When you are not on the Mac, give the operator the exact commands;
  run them over SSH only when the operator asks you to.
- `uninstall.sh` lists what it removes and asks first. Pass `--yes` only after the operator has
  confirmed that list.

## 1. Take the Mac Out of the Cluster First

While the deployment still lists the Mac, its proxy keeps restarting and waiting for it.
`omlx-dynamo-uninstall` step 1 takes one Mac out; removing the whole deployment there takes
out every Mac.

## 2. Remove the Example From the Mac

```bash
cd ~/dynamo-omlx/mac
./uninstall.sh                    # service, settings and logs; keeps the model
./uninstall.sh --model --cache    # also the model and the SSD prefix cache
./uninstall.sh --all              # also the Hugging Face token and oMLX itself
cd ~ && rm -rf ~/dynamo-omlx      # last, the scripts themselves
```

| Flag | Removes | Also affects |
|---|---|---|
| none | stops the service and downloads; `~/.config/dynamo-omlx/omlx-mac.env` with its pid file; `~/Library/Logs/dynamo-omlx` | nothing else |
| `--model` | `~/.omlx/models/<model id>` and its partial download in `~/.omlx/models/.downloading` | nothing else |
| `--cache` | `OMLX_SSD_CACHE_DIR` (default `~/.omlx/cache`) | every oMLX model on the Mac |
| `--hf-token` | `~/.cache/huggingface/token` and `stored_tokens` | every Hugging Face tool on the Mac |
| `--omlx` | `brew uninstall omlx` | the oMLX app and CLI |

- The settings file holds the API key; other Macs keep working with their copies.
- `uninstall.sh` refuses paths that reach beyond the example's files, for example a cache
  directory set to the home directory. Fix the settings file rather than working around it.
- oMLX's own data in `~/.omlx` stays: its settings, the per-model settings, other models.
  Multi-token prediction that `OMLX_MTP=on` turned on stays on in oMLX; to undo it, set
  `OMLX_MTP=off` and restart the service before uninstalling, or turn it off on oMLX's admin
  page.
- An oMLX app or pip install is not removed: quit the app and move it to the Trash, or
  `pip uninstall omlx` in its environment.

## 3. End the Linux Host's Access

Only for what nothing else uses:

- The Mac's `Host` block in `~/.ssh/config.d/omlx-macs` on Linux. When no Mac is left, also
  `~/.ssh/omlx-macs/` and that config file.
- On the Mac, the `~/.ssh/authorized_keys` line that matches `~/.ssh/omlx-macs/id_ed25519.pub`
  on Linux. The operator removes it; it ends the Linux host's key access to that Mac.
- The Linux copy of the Macs' key goes with the cluster side, in `omlx-dynamo-uninstall`, once
  no Mac uses that key.

## 4. Verify

On the Mac, `./service.sh status` (or `lsof -nP -iTCP:8000 -sTCP:LISTEN` once the scripts are
gone) shows nothing listening, and `pgrep -fl dynamo_omlx` finds nothing. From Linux,
`curl -s -o /dev/null -m 5 -w '%{http_code}\n' http://<address of the Mac>:<port>/dynamo/v1/info`
prints `000`.
