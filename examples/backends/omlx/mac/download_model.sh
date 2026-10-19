#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pre-download the model this Mac serves into oMLX's model directory.
#
#   ./download_model.sh --login        # once: save a Hugging Face token
#   ./download_model.sh                # OMLX_MODEL_REPO from omlx-mac.env
#   ./download_model.sh --background   # keeps going after the terminal or SSH session ends
#   ./download_model.sh --update       # fetch the files a newer revision changed
#   ./download_model.sh --repo Org/Name [--name <oMLX model id>]
#
# Rerunning resumes an interrupted download, and the Mac stays awake meanwhile.
# Without a token the Hub limits the download; --login asks for one (hidden
# input) and saves it where huggingface_hub and oMLX look. Any other option
# (--revision, --max-workers, --attempts) goes to download_model.py.

set -euo pipefail
# shellcheck source=SCRIPTDIR/common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
    cat <<EOF
Usage: $0 [--login | [--background] [--update] [--repo ORG/NAME [--name ID]]] [--config FILE]

  --login        Save a Hugging Face access token for downloads (asks for it)
  --background   Download detached from this terminal; log in $LOG_DIR
  --update       Replace a finished download when the Hub has a newer revision
  --repo, --name Hugging Face repo and oMLX model id (default: OMLX_MODEL_REPO
                 and OMLX_BRIDGE_MODEL_ID from the settings file)
  --config FILE  Settings file (default: $CONFIG_FILE)
EOF
}

REPO=""
NAME=""
BACKGROUND=""
LOGIN=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            REPO=$2
            shift 2
            ;;
        --name)
            NAME=$2
            shift 2
            ;;
        --background)
            BACKGROUND=1
            shift
            ;;
        --login)
            LOGIN=1
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
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

load_config
require_omlx_python
if [[ -n "$LOGIN" ]]; then
    exec "$OMLX_PYTHON" "$MAC_DIR/download_model.py" --login
fi
# Parallel Xet transfers; this Mac does little else while it downloads.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [[ -z "$REPO" ]]; then
    REPO=$OMLX_MODEL_REPO
    NAME=${NAME:-$OMLX_BRIDGE_MODEL_ID}
fi
NAME=${NAME:-${REPO##*/}}
CMD=(caffeinate -ims "$OMLX_PYTHON" "$MAC_DIR/download_model.py" "$REPO"
    --model-dir "$OMLX_MODEL_DIR" --name "$NAME" ${ARGS[@]+"${ARGS[@]}"})

if [[ -z "$BACKGROUND" ]]; then
    exec "${CMD[@]}"
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/download-$NAME.log"
detach "${CMD[@]}" >>"$LOG" 2>&1 </dev/null
cat <<EOF
Downloading $REPO in the background (pid $DETACHED_PID).
  Progress: tail -f "$LOG"   (a line every 30 seconds)
  Stop:     pkill -f "download_model.py $REPO"   (rerun this script to resume)
The model appears in $OMLX_MODEL_DIR/$NAME once every file is verified.
EOF
