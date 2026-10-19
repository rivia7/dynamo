#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remove what setup.sh, download_model.sh and service.sh put on this Mac.
#
#   ./uninstall.sh                   # the service, settings and logs
#   ./uninstall.sh --model --cache   # also the downloaded model and the SSD prefix cache
#   ./uninstall.sh --all             # also the Hugging Face token and oMLX itself
#
# It lists what it removes and asks first; --yes skips the question. This
# directory stays: delete it last, for example with rm -rf ~/dynamo-omlx.

set -euo pipefail
# shellcheck source=SCRIPTDIR/common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
    cat <<EOF
Usage: $0 [--model] [--cache] [--hf-token] [--omlx] [--all] [--yes] [--config FILE]

Always: stops the service and background downloads, and removes the settings
file, its pid file and $LOG_DIR.

  --model     The downloaded model and any partial download of it
  --cache     oMLX's SSD prefix cache directory, which every oMLX model shares
  --hf-token  The saved Hugging Face token, which every Hugging Face tool shares
  --omlx      oMLX itself, when Homebrew installed it
  --all       All of the above
  --yes       Do not ask before removing
  --config F  Settings file (default: $CONFIG_FILE)
EOF
}

MODEL=""
CACHE=""
TOKEN=""
OMLX=""
YES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL=1
            shift
            ;;
        --cache)
            CACHE=1
            shift
            ;;
        --hf-token)
            TOKEN=1
            shift
            ;;
        --omlx)
            OMLX=1
            shift
            ;;
        --all)
            MODEL=1 CACHE=1 TOKEN=1 OMLX=1
            shift
            ;;
        --yes)
            YES=1
            shift
            ;;
        --config)
            CONFIG_FILE=$2
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

load_config
NAME=$(basename "${CONFIG_FILE%.env}")
PID_FILE="$(dirname "$CONFIG_FILE")/$NAME.pid"
HF_TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
BREW=$(command -v brew || echo /opt/homebrew/bin/brew)

# Refuse anything broader than this example's own files, such as a settings
# file that points the cache at the home directory.
check_path() {
    local path=${1%/}
    case $path in
        */../* | */.. | */./* | */.) die "refusing to remove $1: not a plain path" ;;
        /*/*/*) ;;
        *) die "refusing to remove $1: too close to the root" ;;
    esac
    case $path in
        "$HOME" | "$HOME/.omlx" | "$HOME/.config" | "$HOME/.cache" | "$HOME/Library" | \
            "$HOME/Library/Logs" | "${OMLX_MODEL_DIR%/}")
            die "refusing to remove $1"
            ;;
    esac
}

case $OMLX_BRIDGE_MODEL_ID in
    */* | . | ..) die "OMLX_BRIDGE_MODEL_ID=$OMLX_BRIDGE_MODEL_ID is not a directory name" ;;
esac
TARGETS=("$CONFIG_FILE" "$PID_FILE" "$LOG_DIR")
if [[ -n "$MODEL" ]]; then
    TARGETS+=("$OMLX_MODEL_DIR/$OMLX_BRIDGE_MODEL_ID" "$OMLX_MODEL_DIR/.downloading/$OMLX_BRIDGE_MODEL_ID")
fi
if [[ -n "$CACHE" ]]; then
    TARGETS+=("$OMLX_SSD_CACHE_DIR")
fi
if [[ -n "$TOKEN" ]]; then
    TARGETS+=("$HF_TOKEN_FILE" "$(dirname "$HF_TOKEN_FILE")/stored_tokens")
fi
for target in "${TARGETS[@]}"; do
    check_path "$target"
done

echo "Stops the service and background downloads of $EXAMPLE_DIR, and removes:"
for target in "${TARGETS[@]}"; do
    if [[ -e "$target" || -L "$target" ]]; then
        echo "  $target ($(du -sh "$target" 2>/dev/null | cut -f1))"
    fi
done
if [[ -n "$OMLX" ]]; then
    if "$BREW" list --formula omlx >/dev/null 2>&1; then
        echo "  oMLX (brew uninstall omlx)"
    else
        OMLX=""
        echo "Not removed: oMLX is not a Homebrew install here. Quit the oMLX app and move it to"
        echo "the Trash, or pip uninstall omlx in its environment."
    fi
fi
[[ -n "$MODEL" ]] || echo "Kept: the model in $OMLX_MODEL_DIR/$OMLX_BRIDGE_MODEL_ID (--model removes it)"
[[ -n "$CACHE" ]] || echo "Kept: the SSD prefix cache in $OMLX_SSD_CACHE_DIR (--cache removes it)"

if [[ -z "$YES" ]]; then
    [[ -t 0 ]] || die "no terminal to confirm on; pass --yes"
    read -r -p "Go ahead? [y/N] " answer
    [[ "$answer" == [yY]* ]] || {
        echo "Nothing removed."
        exit 1
    }
fi

# The pid file lives next to the settings file.
if [[ -d "$(dirname "$CONFIG_FILE")" ]]; then
    "$MAC_DIR/service.sh" --config "$CONFIG_FILE" stop
fi
pkill -f "$MAC_DIR/download_model.py" || true
for target in "${TARGETS[@]}"; do
    if [[ -e "$target" || -L "$target" ]]; then
        rm -rf -- "$target"
        echo "Removed $target"
    fi
done
# Parents this example created, once they are empty.
rmdir "$(dirname "$CONFIG_FILE")" "$OMLX_MODEL_DIR/.downloading" 2>/dev/null || true
if [[ -n "$OMLX" ]]; then
    "$BREW" uninstall omlx
fi

echo "Done. oMLX's own data in ~/.omlx stays."
if ! git -C "$EXAMPLE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Delete this directory last: rm -rf $EXAMPLE_DIR"
fi
