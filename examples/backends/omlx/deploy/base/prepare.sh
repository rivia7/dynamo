#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Init container of an oMLX proxy pod. Lays out the dynamo_omlx package from the
# flat ConfigMap in $PREPARE_SRC, where "--" in a key stands for "/". Then
# fetches the model's config, tokenizer and chat template, without weights, into
# $PREPARE_MODELS/$OMLX_MODEL_DIR, the proxy's --model-path. The model settings
# come from the omlx-profile ConfigMap.

set -eu

: "${OMLX_MODEL_REPO:?set it in the overlay's profile.env}"
: "${OMLX_MODEL_REVISION:?set it in the overlay's profile.env}"
: "${OMLX_MODEL_DIR:?set it in the overlay's profile.env}"
SRC="${PREPARE_SRC:-/src}"
CODE="${PREPARE_CODE:-/opt/omlx}"
MODELS="${PREPARE_MODELS:-/models}"
MODEL="$MODELS/$OMLX_MODEL_DIR"

for file in "$SRC"/dynamo_omlx--*; do
    dest="$CODE/$(basename "$file" | sed 's|--|/|g')"
    mkdir -p "$(dirname "$dest")"
    cp "$file" "$dest"
done

# A pinned revision gives every proxy of the model byte-identical files. When a
# restarted pod finds them already fetched, the downloader keeps them after a
# few failed attempts to reach the Hub; Kubernetes retries a first download.
python3 "$SRC/download_model.py" "$OMLX_MODEL_REPO" --revision "$OMLX_MODEL_REVISION" \
    --metadata-only --attempts 3 --model-dir "$(dirname "$MODEL")" --name "$(basename "$MODEL")"

# The frontend counts image tokens within the pixel range of
# preprocessor_config.json, and the proxy has the Macs process images within
# it, so OMLX_IMAGE_MAX_PIXELS caps both. The file is rewritten the same way on
# every proxy, as the model card checksum covers it.
if [ -n "${OMLX_IMAGE_MAX_PIXELS:-}" ]; then
    python3 - "$MODEL/preprocessor_config.json" "$OMLX_IMAGE_MAX_PIXELS" <<'EOF'
import json
import sys

path, max_pixels = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    config = json.load(f)
size = config.get("size") if isinstance(config.get("size"), dict) else {}
min_pixels = config.get("min_pixels") or size.get("shortest_edge") or 0
if max_pixels < min_pixels:
    sys.exit(f"OMLX_IMAGE_MAX_PIXELS={max_pixels} is below min_pixels {min_pixels}")
config["max_pixels"] = max_pixels
with open(path, "w") as f:
    json.dump(config, f, indent=2, sort_keys=True)
    f.write("\n")
EOF
fi
ls -l "$MODEL"
