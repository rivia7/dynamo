<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# oMLX Example

## Notes

- `dynamo_omlx/bridge/` runs inside oMLX's Python on the Mac and patches oMLX internals, written against oMLX commit `31b6b6a0`; `python -m dynamo_omlx.bridge --check` lists what an oMLX upgrade broke.
- `mac/*.sh` run under macOS `/bin/bash` 3.2: no associative arrays or `${var,,}`, expand maybe-empty arrays as `${a[@]+"${a[@]}"}` under `set -u`, and `examples/common/launch_utils.sh` (bash 4.3+) is unusable.
- Test `mac/*.sh` on Linux: copy `mac/` beside a stub `dynamo_omlx/bridge/__main__.py`, and put stubs for `caffeinate`, `lsof`, `stat -f`, `df -g`, `route`, `ipconfig`, `sysctl`, `sw_vers`, `networksetup`, `pmset` and `uname` first on `PATH`.
- Every path `mac/uninstall.sh` deletes goes through `check_path`; route new targets through it too.
- Xet downloads keep in-flight bytes in memory: disk usage shows no progress, and an interrupted file restarts from zero. Read progress from `snapshot_download`'s `tqdm_class` byte bars; huggingface_hub 1.x counts Xet bytes only as files complete.
- FastAPI 0.142 (oMLX's) lists an included router as one entry in `app.routes`; check a mounted route by sending a request through `httpx.ASGITransport`.
- `deploy/` targets NGC Dynamo 1.5.1 while the code tracks main: `tests/` pass on both, so run them against a 1.5.1 environment after changing the proxy, and bump the chart and every image tag under `deploy/` together.
- `deploy/base/kustomization.yaml` lists the proxy modules by hand (`--` stands for `/` in ConfigMap keys); add a new `dynamo_omlx/proxy/` module there, and `tests/test_omlx_deploy.py` checks the list. Render overlays and `deploy/benchmark` with `kubectl kustomize --load-restrictor=LoadRestrictionsNone`; `kubectl apply -k` cannot read files outside them.
- The comments in `deploy/base/dgd.yaml` and `kustomizeconfig.yaml` record 1.5.1 operator constraints (every container sets `command`, no `ports` on `main`, no canary health check, hashed ConfigMap names); keep them true when editing.
- The model card checksum covers the `--model-path` string and every file there except weights, dotfiles and `README.md`, so all proxies of a model need the same path and the same pinned revision.
- AIPerf runs must not send `ignore_eos` or `min_tokens`: the proxy rejects both.
- The Ingress runs Traefik's ForwardAuth against `deploy/base/token_auth.py`, which accepts `Authorization: Bearer` or `x-api-key` with a token of 32 or more characters from the Secret `omlx-api-tokens` (one key per holder, files in `~/.config/dynamo-omlx/api-tokens/` on Linux) and rereads it on every request. In-cluster callers (AIPerf, `status.py`) use the Services and skip it. The middleware's address and the Ingress annotation name the namespace `dynamo-omlx`; `tests/test_omlx_deploy.py` checks they agree.
- The Ingress sends `/v1/messages` through `deploy/base/anthropic_fix.py`: when a turn has several tool calls, the frontend's `lib/llm/src/protocols/anthropic/stream_converter.rs` (1.5.1, and main on 2026-10-09) sends the text between them as deltas of the text block it already closed, which Claude Code rejects as a malformed stream. Drop the proxy and its Ingress path once the converter opens a new block for such text or skips it.
- Claude Code replaces `$` followed by a digit in a SKILL.md with the words the skill is invoked with (`$3` becomes the fourth word), so write awk fields as `$(3)` in skills.
- `.agents/skills/omlx-status/scripts/status.py` reads the proxy's `omlx_*` metrics, which `tests/test_omlx_status.py` renders with `ProxyMetrics`, and the frontend's `dynamo_frontend_*` metrics of 1.5.1, which no test checks: recheck those after a Dynamo upgrade.

## Skills

Skills live in `.agents/skills/`; `.claude/skills` links to it. `python3 scripts/validate_skills.py examples/backends/omlx` (from the repository root) checks them and this index.

- `omlx-mac-install` — set up a Mac: oMLX, settings and key, model and service, then hand it to the cluster; or add a Mac
- `omlx-mac-uninstall` — remove the example from a Mac
- `omlx-dynamo-install` — install the Dynamo side on Kubernetes with NGC Dynamo, verify and publish it; add a Mac, switch the model, roll out changes
- `omlx-dynamo-uninstall` — remove the Dynamo side from Kubernetes: one Mac, a token, the deployment, the platform, monitoring
- `omlx-status` — show the Kubernetes deployment's state, how to reach it and how to call its OpenAI API
