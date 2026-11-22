#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Idempotent host bootstrap for one cluster_bench node (Linux).
#
#   bash bootstrap_node.sh --workdir /root/cluster_bench [--minimal]
#
# Installs RDMA diagnostics with apt when missing (best effort, once; disable
# with CB_FABRIC_TOOLS=0) and uv when missing (official installer, into
# ~/.local/bin), creates
# <workdir>/venv with Python 3.12 and installs the harness's host-side
# dependencies: aiperf==0.14.0 (skipped with --minimal, for peers that never
# run the AIPerf client), huggingface_hub[hf_xet], nvidia-ml-py, pyyaml and
# pydantic>=2. A marker holding a hash of the requirement list makes a re-run
# with the same list a no-op. Logs go to stderr; the last line on stdout is the
# venv's python path.
#
# Not an engine launch script (.ai/bash-launch-guidelines.md does not apply):
# it starts no background processes, so set -euo pipefail is enough.

set -euo pipefail

PYTHON_VERSION="3.12"
AIPERF_REQUIREMENT="aiperf==0.14.0"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"

usage() {
    echo "usage: bootstrap_node.sh --workdir DIR [--minimal]" >&2
    exit 2
}

log() {
    echo "[bootstrap_node] $*" >&2
}

WORKDIR=""
MINIMAL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --workdir)
            [ $# -ge 2 ] || usage
            WORKDIR="$2"
            shift 2
            ;;
        --workdir=*)
            WORKDIR="${1#--workdir=}"
            shift
            ;;
        --minimal)
            MINIMAL=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "bootstrap_node.sh: unknown argument: $1" >&2
            usage
            ;;
    esac
done
case "$WORKDIR" in
    /*) ;;
    *) echo "bootstrap_node.sh: --workdir must be an absolute path" >&2; usage ;;
esac

REQUIREMENTS=("huggingface_hub[hf_xet]" "nvidia-ml-py" "pyyaml" "pydantic>=2")
if [ "$MINIMAL" -eq 0 ]; then
    REQUIREMENTS=("$AIPERF_REQUIREMENT" "${REQUIREMENTS[@]}")
fi

VENV="$WORKDIR/venv"
VENV_PY="$VENV/bin/python"
MARKER="$VENV/.cluster_bench_bootstrap"

sha256_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | cut -d' ' -f1
    else
        shasum -a 256 | cut -d' ' -f1
    fi
}

WANT="$(printf '%s\n' "python=$PYTHON_VERSION" "${REQUIREMENTS[@]}" | sha256_stdin)"

venv_is_current() {
    [ -x "$VENV_PY" ] && [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$WANT" ]
}

# Best effort: RDMA diagnostics (ibstat/smpquery/perfquery/ibtracert, ib_write_bw)
# let agents/fabric_probe.py read switch topology and run link tests on the host
# when the pinned image lacks them. Never fatal; one attempt per node.
FABRIC_TOOLS=(ibstat smpquery perfquery ibtracert sminfo ib_write_bw ib_write_lat ibv_devinfo)
install_fabric_tools() {
    [ "${CB_FABRIC_TOOLS:-1}" = "1" ] || return 0
    [ "$(id -u)" = "0" ] || return 0
    command -v apt-get >/dev/null 2>&1 || return 0
    local missing=0 tool
    for tool in "${FABRIC_TOOLS[@]}"; do
        command -v "$tool" >/dev/null 2>&1 || missing=1
    done
    [ "$missing" = "1" ] || return 0
    mkdir -p "$WORKDIR"
    local marker="$WORKDIR/.fabric_tools_attempted"
    [ -e "$marker" ] && return 0
    : > "$marker"
    log "installing RDMA diagnostics (infiniband-diags ibverbs-utils perftest), best effort"
    if ! { timeout 300 apt-get update -qq && \
           DEBIAN_FRONTEND=noninteractive timeout 300 apt-get install -y -qq \
               --no-install-recommends infiniband-diags ibverbs-utils perftest; } >&2 2>&1; then
        log "RDMA diagnostics not installed; fabric probing falls back to the image or sysfs"
    fi
    return 0
}
install_fabric_tools || true

if venv_is_current; then
    log "venv is up to date: $VENV"
    echo "$VENV_PY"
    exit 0
fi

mkdir -p "$WORKDIR"
# Concurrent bootstraps of one node (controller sync and runner preflight) serialize.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$WORKDIR/.bootstrap.lock"
    flock 9
    if venv_is_current; then
        log "venv was bootstrapped concurrently: $VENV"
        echo "$VENV_PY"
        exit 0
    fi
fi

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    local candidate
    for candidate in "${XDG_BIN_HOME:-$HOME/.local/bin}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

UV="$(find_uv || true)"
if [ -z "$UV" ]; then
    log "uv not found; installing it with the official installer ($UV_INSTALLER_URL)"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$UV_INSTALLER_URL" | env UV_NO_MODIFY_PATH=1 sh >&2
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$UV_INSTALLER_URL" | env UV_NO_MODIFY_PATH=1 sh >&2
    else
        log "neither curl nor wget is available to download uv"
        exit 1
    fi
    UV="$(find_uv)" || { log "the uv installer did not leave a uv binary in ~/.local/bin"; exit 1; }
fi
log "using $("$UV" --version 2>/dev/null || echo uv) at $UV"

if [ -x "$VENV_PY" ]; then
    HAVE="$("$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$HAVE" != "$PYTHON_VERSION" ]; then
        log "replacing $VENV (Python ${HAVE:-unknown}, want $PYTHON_VERSION)"
        rm -rf "$VENV"
    fi
elif [ -e "$VENV" ]; then
    log "removing incomplete venv at $VENV"
    rm -rf "$VENV"
fi
if [ ! -x "$VENV_PY" ]; then
    # uv downloads a managed CPython when the host has no 3.12.
    "$UV" venv --python "$PYTHON_VERSION" "$VENV" >&2
fi

log "installing: ${REQUIREMENTS[*]}"
"$UV" pip install --python "$VENV_PY" "${REQUIREMENTS[@]}" >&2

IMPORTS="import huggingface_hub, pydantic, pynvml, yaml"
if [ "$MINIMAL" -eq 0 ]; then
    IMPORTS="$IMPORTS, aiperf"
fi
"$VENV_PY" -c "$IMPORTS" >&2

printf '%s\n' "$WANT" > "$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
log "venv ready: $VENV"
echo "$VENV_PY"
