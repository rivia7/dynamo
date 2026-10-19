#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Linux side: Dynamo frontend with KV-aware routing plus one oMLX proxy worker
# per Mac. Start mac/service.sh on every Mac first.
# GPUs: 0 (inference runs on the Macs)

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Space-separated bridge URLs, one per Mac.
OMLX_BRIDGE_URLS="${OMLX_BRIDGE_URLS:-http://mac-studio.local:8000}"
# config.json, tokenizer files and chat template copied from the Mac's model
# directory (no weights). Keep the same path on every Linux host.
MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
# Qwen3.8-Flash-Next; see recipes/qwen3.8-flash-next.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# oMLX commits prefix blocks when a request finishes; predicted routing covers
# requests that share a prefix with one still generating.
ROUTER_PREDICTED_TTL_SECS="${ROUTER_PREDICTED_TTL_SECS:-600}"
PROXY_RESTART_DELAY="${PROXY_RESTART_DELAY:-5}"
# One Linux host needs no etcd: every Dynamo process runs here.
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"

# Parse arguments BEFORE installing the kill-process-group EXIT trap.
# `--help` and unknown-option exits would otherwise kill the caller.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --bridge-urls)
            OMLX_BRIDGE_URLS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model-path <config dir> [--bridge-urls '<url> <url>'] [proxy options]"
            echo ""
            echo "Environment: OMLX_BRIDGE_URLS, OMLX_API_KEY, MODEL_PATH, SERVED_MODEL_NAME,"
            echo "  TOOL_CALL_PARSER, REASONING_PARSER, ROUTER_PREDICTED_TTL_SECS, PROXY_RESTART_DELAY,"
            echo "  DYN_DISCOVERY_BACKEND (file for one Linux host, etcd across hosts),"
            echo "  DYN_HTTP_PORT, DYN_SYSTEM_PORT (first proxy; the next ones count up)."
            echo ""
            echo "Additional options are passed to every python -m dynamo_omlx.proxy."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Set --model-path (or MODEL_PATH) to a copy of the model's config and tokenizer files." >&2
    exit 1
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

trap 'echo Cleaning up...; kill 0' EXIT

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
print_launch_banner "Launching Dynamo in front of oMLX" "$SERVED_MODEL_NAME" "$HTTP_PORT" \
    "Macs:        $OMLX_BRIDGE_URLS"

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --router-mode kv \
    --router-predicted-ttl-secs "$ROUTER_PREDICTED_TTL_SECS" &

# A proxy exits on purpose when its Mac stays unreachable, so the frontend stops
# routing to that Mac. Restart it in its own loop: letting wait_any_exit see the
# exit would tear down the frontend and the proxies of the healthy Macs. The
# sleep is a restart delay; the proxy itself waits until its Mac is ready.
for url in $OMLX_BRIDGE_URLS; do
    (
        while true; do
            DYN_SYSTEM_PORT="$SYSTEM_PORT" python3 -m dynamo_omlx.proxy \
                --bridge-url "$url" \
                --model-path "$MODEL_PATH" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --dyn-tool-call-parser "$TOOL_CALL_PARSER" \
                --dyn-reasoning-parser "$REASONING_PARSER" \
                "${EXTRA_ARGS[@]}" \
                || echo "oMLX proxy for $url exited with $?; restarting in ${PROXY_RESTART_DELAY}s" >&2
            sleep "$PROXY_RESTART_DELAY"
        done
    ) &
    SYSTEM_PORT=$((SYSTEM_PORT + 1))
done

# Exit when the frontend fails; kill 0 in the EXIT trap tears down the rest.
wait_any_exit
