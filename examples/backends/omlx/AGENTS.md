<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# oMLX Example

## Notes

- `dynamo_omlx/bridge/` runs inside oMLX's Python on the Mac and patches oMLX internals, written against oMLX commit `31b6b6a0`; `python -m dynamo_omlx.bridge --check` lists what an oMLX upgrade broke, image input under `no image input:` lines.
- Image requests run `VLMBatchedEngine._prepare_vision_inputs` from tokenization on (`omlx_adapter._prepare_images`) on Dynamo's tokens, which carry one `<|image_pad|>` per image. The frontend counts image tokens within the pixel range of the proxies' `preprocessor_config.json` (`OMLX_IMAGE_MAX_PIXELS` caps it in `prepare.sh`), and the proxy sends that range to the bridge: never size images any other way.
- The bridge knows an image by its file hash and pixel range (`bridge/images.image_key`); oMLX's vision feature cache holds `[features, grid]` under that key, so a fully cached request of a `_FEATURES_ALONE_MODELS` model decodes no pixels.
- KV events carry each image's tokens as its pad value (`protocol.pad_value_for_mm_hash`), the tokens Dynamo's router hashes. The frontend expands images only when llm-multimodal's registry matches the proxies' `--model-path`, then `model_type`; for Qwen3.8 see Qwen3.8 missing from the image registry under Dynamo Issues.
- `mac/*.sh` run under macOS `/bin/bash` 3.2: no associative arrays or `${var,,}`, expand maybe-empty arrays as `${a[@]+"${a[@]}"}` under `set -u`, and `examples/common/launch_utils.sh` (bash 4.3+) is unusable.
- Test `mac/*.sh` on Linux: copy `mac/` beside a stub `dynamo_omlx/bridge/__main__.py`, and put stubs for `caffeinate`, `lsof`, `stat -f`, `df -g`, `route`, `ipconfig`, `sysctl`, `sw_vers`, `networksetup`, `pmset` and `uname` first on `PATH`.
- Every path `mac/uninstall.sh` deletes goes through `check_path`; route new targets through it too.
- Xet downloads keep in-flight bytes in memory: disk usage shows no progress, and an interrupted file restarts from zero. Read progress from `snapshot_download`'s `tqdm_class` byte bars; huggingface_hub 1.x counts Xet bytes only as files complete.
- FastAPI 0.142 (oMLX's) lists an included router as one entry in `app.routes`; check a mounted route by sending a request through `httpx.ASGITransport`.
- Dynamo main turns structural-tag guided decoding on by default, which makes the proxy reject every request with tools: `agg.sh` on a Dynamo built from main needs `--dyn-structural-tag false`, and moving `deploy/` past 1.5.1 sets `DYN_STRUCTURAL_TAG=false` on the proxy.
- `deploy/` targets NGC Dynamo 1.5.1 while the code tracks main: `tests/` pass on both, so run them against a 1.5.1 environment after changing the proxy, and bump the chart and every image tag under `deploy/` together.
- `deploy/base/kustomization.yaml` lists the proxy modules by hand (`--` stands for `/` in ConfigMap keys); add a new `dynamo_omlx/proxy/` module there, and `tests/test_omlx_deploy.py` checks the list. Render overlays and `deploy/benchmark` with `kubectl kustomize --load-restrictor=LoadRestrictionsNone`; `kubectl apply -k` cannot read files outside them.
- The comments in `deploy/base/dgd.yaml` and `kustomizeconfig.yaml` record 1.5.1 operator constraints (every container sets `command`, no `ports` on `main`, no canary health check, hashed ConfigMap names); keep them true when editing.
- The model card checksum covers the `--model-path` string and every file there except weights, dotfiles and `README.md`, so all proxies of a model need the same path and the same pinned revision.
- AIPerf runs must not send `ignore_eos` or `min_tokens`: the proxy rejects both.
- The Ingress runs Traefik's ForwardAuth against `deploy/base/token_auth.py`, which accepts `Authorization: Bearer` or `x-api-key` with a token of 32 or more characters from the Secret `omlx-api-tokens` (one key per holder, files in `~/.config/dynamo-omlx/api-tokens/` on Linux) and rereads it on every request. In-cluster callers (AIPerf, `status.py`) use the Services and skip it. The middleware's address and the Ingress annotation name the namespace `dynamo-omlx`; `tests/test_omlx_deploy.py` checks they agree.
- Claude Code replaces `$` followed by a digit in a SKILL.md with the words the skill is invoked with (`$3` becomes the fourth word), so write awk fields as `$(3)` in skills.
- `.agents/skills/omlx-status/scripts/status.py` reads the proxy's `omlx_*` metrics, which `tests/test_omlx_status.py` renders with `ProxyMetrics`, and the frontend's `dynamo_frontend_*` metrics of 1.5.1, which no test checks: recheck those after a Dynamo upgrade.

## Dynamo Issues

Bugs and gaps in Dynamo that this example works around or documents. `main` means commit `92a28f812` of 2026-10-19, the Dynamo they were last checked against. The README's Dynamo Issues table describes the same issues for users, under the same bold names in the same order and against the same commit, and `tests/test_omlx_docs.py` checks that the two agree. Add an entry to both with every new workaround. When `deploy/` or `agg.sh` moves to a newer Dynamo, recheck every entry against it and update the commit in both; for each issue it fixes, remove the workaround, its tests and every mention, then both entries.

- **Text after tool calls.** 1.5.1's `lib/llm/src/protocols/anthropic/stream_converter.rs` streams the text between several tool calls as deltas of the text block it already closed, which Claude Code rejects as a malformed stream; main fixed it with #15467 (2026-10-01). Workaround: the Ingress sends `/v1/messages` through the stream repair, `deploy/base/anthropic_fix.py`. Once `deploy/` runs a release with the fix, remove the repair, its Ingress path, `tests/test_omlx_anthropic_fix.py` and its mentions, which `grep -rli -e anthropic.fix -e 'stream repair' .` finds.
- **Images sent twice.** 1.5.1's frontend sends each `data:` URL to the proxy in `multi_modal_data` and again in `extra_args.messages`, so two photos exceed the 32 MiB request-plane message; main fixed it with #14801 (2026-09-17). Workaround: the proxy's `DYN_TCP_MAX_MESSAGE_SIZE` of 128 MiB in `deploy/base/dgd.yaml`, which `test_proxies_read_requests_with_several_photos` checks. Once fixed, keep the limit, but drop the duplication from its comment and from the README's Images section.
- **Canary health check.** The canary gives a worker 3 s for its first token (`DYN_HEALTH_CHECK_REQUEST_TIMEOUT`), and a worker's `/live` runs the same `health_handler` as `/health` (`SystemProbePolicy::Worker` in `lib/runtime/src/system_status_server.rs`), in 1.5.1 and main, so a long prefill on a Mac restarts the proxy. Workaround: `DYN_HEALTH_CHECK_ENABLED=false` on the proxy. Once `/live` no longer fails with the canary, turn the check back on with a timeout above a long prefill, and update the comment in `deploy/base/dgd.yaml` and the README's paragraph under Changes.
- **Qwen3.8 missing from the image registry.** llm-multimodal's registry, 1.9.0 in 1.5.1 and the one in main, has no `qwen4_exp`, so the frontend would not expand Qwen3.8's images for routing. Workaround: `OMLX_MODEL_DIR=qwen3_vl/Qwen3.8-Flash-Next` in the overlay's `profile.env`, so the proxies' `--model-path` selects Qwen3-VL's image processor; `test_qwen38_path_selects_the_qwen3_vl_image_processor` holds a copy of the registry. Once fixed, drop `qwen3_vl/` from the profile and from the README's Linux Setup and Images, and update the test.
- **Claude Code's effort and thinking.** `/v1/messages` ignores `output_config.effort` and sets `enable_thinking` unless `thinking.type` is `disabled` (`lib/llm/src/http/service/anthropic.rs`), in 1.5.1 and main. Claude Code 2.1.304 sends `thinking.type` `adaptive`, or no `thinking` with its thinking off, so it always runs Qwen3.8 at `xhigh` with thinking. No workaround. Once fixed, correct the README's Coding Agents and the Claude Code notes in `omlx-status`.
- **Lease conflict after a reboot.** After a node reboot, the 1.5.1 frontend can fail to register its endpoint with `endpoint registration lease conflicts with a pre-existing specification` and serve no model; main keeps the same check in `lib/runtime/src/discovery/registration.rs`. Workaround: delete the frontend pod, as After a Node Reboot in `omlx-dynamo-install` and `omlx-status` say. Once fixed, delete those steps.
- **Discovery registration hang.** A 1.5.1 proxy hung once after `Registering event channel`, right after the DGD was recreated, and got no `dynamoworkermetadatas` entry. Workaround: delete the pod, as the troubleshooting tables of the README, `omlx-dynamo-install` and `omlx-status` say. Once fixed, delete those rows.
- **Duplicate-store warnings.** The router's index of predicted blocks for `DYN_ROUTER_PREDICTED_TTL_SECS` (`predict_on_route_side_indexer` in `lib/llm/src/kv_router/indexer/mod.rs`) reports to the workers' KV event metrics, so `dynamo_component_kv_cache_event_warnings{warning_kind="duplicate_store"}` rises by one for every repeated request whose blocks are all cached, in 1.5.1 and main, though the proxy sends no event. No workaround needed. Once fixed, delete both entries.

## Skills

Skills live in `.agents/skills/`; `.claude/skills` links to it. `python3 scripts/validate_skills.py examples/backends/omlx` (from the repository root) checks them and this index.

- `omlx-mac-install` — set up a Mac: oMLX, settings and key, model and service, then hand it to the cluster; or add a Mac
- `omlx-mac-uninstall` — remove the example from a Mac
- `omlx-dynamo-install` — install the Dynamo side on Kubernetes with NGC Dynamo, verify and publish it; add a Mac, switch the model, roll out changes
- `omlx-dynamo-uninstall` — remove the Dynamo side from Kubernetes: one Mac, a token, the deployment, the platform, monitoring
- `omlx-status` — show the Kubernetes deployment's state, how to reach it and how to call its OpenAI API
