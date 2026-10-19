---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: omlx-uninstall
description: Removes the oMLX-on-Mac example. Takes a Mac out of the Linux side or the Kubernetes deployment, stops its service and background downloads, deletes its settings and logs, and on request the downloaded model, oMLX's SSD prefix cache, the saved Hugging Face token and oMLX itself, then cleans up Linux and, on request, the cluster's Dynamo platform and monitoring. Use when decommissioning a Mac, freeing its disk, or starting over before omlx-install or omlx-k8s-deploy.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - omlx
    - macos
    - uninstall
---

# Skill: Uninstall the oMLX Example

Removal cannot be undone, and the default model takes hours to download again. Paths below are
relative to `examples/backends/omlx`.

## Ground Rules

- Agree the scope with the operator before removing anything. Keep the model, the cache and
  the token unless the operator asks for them to go.
- Mac steps run on the Mac. When you are not on the Mac, give the operator the exact commands;
  run them over SSH only when the operator asks you to.
- `uninstall.sh` lists what it removes and asks first. Pass `--yes` only after the operator has
  confirmed that list.

## 1. Take the Mac Out of Linux

Remove its URL from `OMLX_BRIDGE_URLS` and restart `agg.sh`, or stop `agg.sh` when removing
everything. Otherwise its proxy keeps restarting and waiting for the Mac.

On Kubernetes, take the Mac's component (`deploy/components/mac-<n>`) out of the overlay's
`components`, and renumber the components of later Macs to close the gap. Render and apply the
overlay with `kubectl kustomize --load-restrictor=LoadRestrictionsNone <overlay> | kubectl apply
--server-side -f -`, then delete the Secret `omlx-mac-<n>` that is no longer used. `mac-1` is in
the base, so removing it means removing the deployment, as in step 4.

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

## 3. Clean Up Linux

Only for what nothing else uses:

- The model metadata directory, for example `/models/Qwen3.8-Flash-Next`.
- The Mac's `Host` block in `~/.ssh/config.d/omlx-macs`. When no Mac is left, also
  `~/.ssh/omlx-macs/` and that config file.
- On the Mac, the `~/.ssh/authorized_keys` line that matches `~/.ssh/omlx-macs/id_ed25519.pub`
  on Linux. The operator removes it; it ends the Linux host's key access to that Mac.

## 4. Remove the Kubernetes Deployment

Skip this without a cluster deployment. Remove the deployment before the platform: with the operator gone, the DGD's finalizer would
hold up the namespace's deletion. Each step is a separate decision for the operator; the
platform and monitoring may serve other workloads.

```bash
kubectl kustomize --load-restrictor=LoadRestrictionsNone deploy/overlays/<overlay> | kubectl delete -f -
kubectl -n dynamo-omlx delete job omlx-aiperf --ignore-not-found     # the benchmark
kubectl delete namespace dynamo-omlx                 # the Secrets and stale ConfigMaps
helm uninstall dynamo-platform -n dynamo-system      # the operator
kubectl delete -f deploy/monitoring/grafana-omlx-dashboard.yaml \
    -f ../../../deploy/observability/grafana-dynamo-dashboard-configmap.yaml
helm uninstall prometheus -n monitoring
```

- The Dynamo CRDs (`kubectl get crd | grep nvidia.com`) outlive the chart, and so do
  monitoring's (`monitoring.coreos.com`) and Prometheus's data volume in `monitoring`. Delete
  them only when nothing else uses them.
- Imported images stay in containerd: `sudo k3s ctr -n k8s.io images rm <name>` on a k3s node.

## 5. Verify

On the Mac, `./service.sh status` (or `lsof -nP -iTCP:8000 -sTCP:LISTEN` once the scripts are
gone) shows nothing listening, and `pgrep -fl dynamo_omlx` finds nothing. On Linux, the
frontend no longer routes to the Mac, and `curl -s localhost:8000/v1/models` lists the model
only while other Macs still serve it. On Kubernetes, `kubectl get dgd -A` no longer lists
`omlx`.
