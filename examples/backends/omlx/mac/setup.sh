#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup of a Mac that serves oMLX behind Dynamo. Safe to rerun, and
# worth rerunning after every oMLX upgrade.
#
#   ./setup.sh                        # the first Mac: generates the API key
#   OMLX_API_KEY=<key> ./setup.sh     # every further Mac: reuses the same key
#   ./setup.sh --install              # also installs oMLX with Homebrew if missing
#
# Writes ~/.config/dynamo-omlx/omlx-mac.env from omlx-mac.env.example unless it
# exists, reports what matters for serving (chip, memory, disk, network, oMLX's
# native kernels), and checks that this oMLX build offers everything the bridge
# hooks into.

set -euo pipefail
# shellcheck source=SCRIPTDIR/common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
    cat <<EOF
Usage: $0 [--install] [--config FILE]

  --install      Install oMLX with Homebrew when it is missing
  --config FILE  Settings file (default: $CONFIG_FILE)

Set OMLX_API_KEY to reuse the key of the other Macs in a new settings file.
EOF
}

INSTALL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --install)
            INSTALL=1
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

[[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]] || die "run this on an Apple Silicon Mac"

GIVEN_KEY="${OMLX_API_KEY:-}"
if [[ -f "$CONFIG_FILE" ]]; then
    echo "Settings: keeping $CONFIG_FILE"
else
    mkdir -p "$(dirname "$CONFIG_FILE")"
    # The file holds the API key: readable by this user only.
    (
        umask 077
        awk -v key="${GIVEN_KEY:-$(openssl rand -hex 32)}" \
            '/^OMLX_API_KEY=/ { print "OMLX_API_KEY=" key; next } { print }' \
            "$MAC_DIR/omlx-mac.env.example" >"$CONFIG_FILE"
    )
    echo "Settings: wrote $CONFIG_FILE"
fi
load_config
if [[ -n "$GIVEN_KEY" && "$GIVEN_KEY" != "${OMLX_API_KEY:-}" ]]; then
    echo "          OMLX_API_KEY differs from the key in $CONFIG_FILE, which wins; edit the file to change it"
fi

if ! find_omlx_python; then
    if [[ -z "$INSTALL" ]]; then
        die "oMLX is not installed. Rerun with --install, install the oMLX app and open it once, or run:
  brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx"
    fi
    BREW=$(command -v brew || echo /opt/homebrew/bin/brew)
    [[ -x "$BREW" ]] || die "Homebrew is not installed; see https://brew.sh"
    "$BREW" tap jundot/omlx https://github.com/jundot/omlx
    "$BREW" install jundot/omlx/omlx
    require_omlx_python
fi

mkdir -p "$OMLX_MODEL_DIR" "$OMLX_SSD_CACHE_DIR"
echo "Mac:      $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GiB, macOS $(sw_vers -productVersion)"
echo "Disk:     $(df -g "$OMLX_MODEL_DIR" | awk 'NR == 2 { print $4 }') GiB free for models and the SSD cache"

IFACE=$(route -n get default 2>/dev/null | awk '/interface:/ { print $2 }')
PORT_NAME=$(networksetup -listallhardwareports 2>/dev/null | awk -v dev="${IFACE:-none}" '
    /^Hardware Port: / { sub(/^Hardware Port: /, ""); name = $0 }
    $0 == ("Device: " dev) { print name }')
echo "Network:  $(mac_address) on ${IFACE:-no default route} (${PORT_NAME:-unknown})"
if [[ "$PORT_NAME" == *Wi-Fi* ]]; then
    echo "          Wi-Fi: use Ethernet if you can; the model download and every token stream cross it"
fi
SLEEP_MIN=$(pmset -g 2>/dev/null | awk '$1 == "sleep" { print $2 }')
if [[ -n "$SLEEP_MIN" && "$SLEEP_MIN" != 0 ]]; then
    echo "Sleep:    after ${SLEEP_MIN} idle minutes; service.sh and download_model.sh keep the Mac awake while they run"
fi

"$OMLX_PYTHON" - <<'EOF'
import importlib.metadata as metadata
import sys


def version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


print(f"oMLX:     {version('omlx') or 'unknown version'} ({sys.executable})")
hub = version("huggingface-hub")
print(
    f"Hub:      huggingface_hub {hub}"
    if hub
    else "Hub:      huggingface_hub is missing, so download_model.sh cannot run"
)
try:
    import os

    from huggingface_hub import constants, get_token

    if os.environ.get("HF_TOKEN"):
        print("          token from HF_TOKEN")
    elif get_token():
        print(f"          token from {constants.HF_TOKEN_PATH}")
    else:
        print(
            "          no token, so the Hub limits downloads; save one with\n"
            "          ./download_model.sh --login"
        )
except Exception:  # noqa: BLE001
    pass
try:
    import mlx.core as mx

    limit = mx.device_info()["max_recommended_working_set_size"] / 1024**3
    print(f"GPU:      {limit:.1f} GiB working-set limit for weights and KV cache")
except Exception as e:  # noqa: BLE001
    print(f"GPU:      working-set limit unknown ({e})")

try:
    from omlx.custom_kernels import native_kernel_status

    native = [name for name, s in native_kernel_status().items() if s["available"]]
    print(f"Kernels:  native {', '.join(native) if native else 'none'}")
except Exception as e:  # noqa: BLE001
    print(f"Kernels:  unknown ({e})")
try:
    from omlx.custom_kernels.glm_moe_dsa import fast

    qwen = fast.is_native_available() and fast.has_symbol(
        "qwen4_qsa_sparse_gqa_attention"
    )
except Exception:  # noqa: BLE001
    qwen = False
if not qwen:
    print(
        "          Qwen3.8-Flash-Next runs on oMLX's generic kernels here. The oMLX\n"
        "          app ships the native ones; Homebrew builds them with Xcode installed:\n"
        "          brew reinstall jundot/omlx/omlx --HEAD --with-custom-kernel"
    )
EOF

MODEL="$OMLX_MODEL_DIR/$OMLX_BRIDGE_MODEL_ID"
if [[ -f "$MODEL/config.json" ]]; then
    echo "Model:    $OMLX_BRIDGE_MODEL_ID is in $OMLX_MODEL_DIR"
elif [[ -d "$OMLX_MODEL_DIR/.downloading/$OMLX_BRIDGE_MODEL_ID" ]]; then
    echo "Model:    $OMLX_BRIDGE_MODEL_ID is partly downloaded; ./download_model.sh resumes it"
else
    echo "Model:    $OMLX_BRIDGE_MODEL_ID is not downloaded yet"
fi

bridge --check || die "this oMLX build lacks what the bridge hooks into; see above"

cat <<EOF

Next, on this Mac:
  $MAC_DIR/download_model.sh --background    # $OMLX_MODEL_REPO
  $MAC_DIR/service.sh start                  # once the download has finished

On Linux, start Dynamo with OMLX_BRIDGE_URLS including http://$(mac_address):$OMLX_PORT
and OMLX_API_KEY set to the key in $CONFIG_FILE.
EOF
