# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sourced by setup.sh, download_model.sh and service.sh; not run on its own.
# shellcheck disable=SC2034  # LOG_DIR and DETACHED_PID are for those scripts.
#
# macOS ships bash 3.2: no associative arrays or ${var,,}, and an array that may
# be empty is expanded as ${a[@]+"${a[@]}"} so that `set -u` accepts it.

MAC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$MAC_DIR/.." && pwd)"
CONFIG_FILE="${DYNAMO_OMLX_CONFIG:-$HOME/.config/dynamo-omlx/omlx-mac.env}"
LOG_DIR="$HOME/Library/Logs/dynamo-omlx"

die() {
    echo "error: $*" >&2
    exit 1
}

# Source the per-Mac settings (setup.sh writes them), then default the rest.
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        set -a
        # shellcheck source=/dev/null
        . "$CONFIG_FILE"
        set +a
    fi
    : "${OMLX_MODEL_REPO:=Jundot/Qwen3.8-Flash-Next-oQ4e-mtp}"
    : "${OMLX_BRIDGE_MODEL_ID:=${OMLX_MODEL_REPO##*/}}"
    : "${OMLX_MODEL_DIR:=$HOME/.omlx/models}"
    : "${OMLX_HOST:=0.0.0.0}"
    : "${OMLX_PORT:=8000}"
    : "${OMLX_SSD_CACHE_DIR:=$HOME/.omlx/cache}"
    # Set but empty means "let oMLX choose".
    : "${OMLX_KV_BLOCK_SIZE=8192}"
    export OMLX_BRIDGE_MODEL_ID OMLX_MODEL_DIR OMLX_HOST OMLX_PORT OMLX_SSD_CACHE_DIR
    # huggingface_hub would take an empty HF_ENDPOINT as the endpoint itself.
    [[ -n "${HF_ENDPOINT:-}" ]] || unset HF_ENDPOINT
    [[ -n "${HF_TOKEN:-}" ]] || unset HF_TOKEN
    [[ -n "${HF_HUB_DISABLE_XET:-}" ]] || unset HF_HUB_DISABLE_XET
    [[ -n "${OMLX_API_KEY:-}" ]] || unset OMLX_API_KEY
}

# Set OMLX_PYTHON to an interpreter that can import omlx: the configured one,
# else Homebrew's, the macOS app's, or python3 (a pip or source install).
find_omlx_python() {
    local candidate
    if [[ -n "${OMLX_PYTHON:-}" ]]; then
        "$OMLX_PYTHON" -c 'import omlx' >/dev/null 2>&1 ||
            die "OMLX_PYTHON=$OMLX_PYTHON cannot import omlx"
        return 0
    fi
    for candidate in \
        /opt/homebrew/opt/omlx/libexec/bin/python \
        "$HOME/.omlx/bin/omlx-cluster-python" \
        python3; do
        if "$candidate" -c 'import omlx' >/dev/null 2>&1; then
            OMLX_PYTHON="$candidate"
            return 0
        fi
    done
    return 1
}

require_omlx_python() {
    find_omlx_python || die "oMLX is not installed (or set OMLX_PYTHON); run setup.sh --install"
}

# The bridge, run from the example directory: `python -m` imports from the
# working directory even when the oMLX app's interpreter wrapper replaces
# PYTHONPATH.
bridge() {
    (cd "$EXAMPLE_DIR" && "$OMLX_PYTHON" -m dynamo_omlx.bridge "$@")
}

# Start a command in the background in a new session, which neither Ctrl-C nor
# the end of the terminal or SSH session reaches (macOS has no setsid command),
# and set DETACHED_PID. The command execs in place, so the pid stays its own.
detach() {
    "$OMLX_PYTHON" -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@" &
    DETACHED_PID=$!
}

# Address this Mac's default route goes out of; the Linux proxies connect to it.
mac_address() {
    local iface
    iface=$(route -n get default 2>/dev/null | awk '/interface:/ { print $2 }')
    ipconfig getifaddr "${iface:-en0}" 2>/dev/null || hostname
}
