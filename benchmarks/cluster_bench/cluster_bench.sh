#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-click entry point of the cluster benchmark harness:
#
#   bash benchmarks/cluster_bench/cluster_bench.sh <subcommand> [args...]
#
#   controller run-all ...   on a laptop: rent a Lium cluster and drive everything
#   run ...                  on node0 of a cluster: run (or resume) an experiment
#   dry-run | status | report | kb ...
#
# Runs under macOS bash 3.2 (laptop) and Linux bash 5 (node0). Python choice:
#   1. $CLUSTER_BENCH_PYTHON, when set;
#   2. <workdir>/venv/bin/python on a node, where <workdir> comes from --workdir,
#      $CLUSTER_BENCH_WORKDIR, or /root/cluster_bench for root on Linux
#      (`run` on Linux bootstraps that venv first when it is missing);
#   3. otherwise ~/.cluster_bench/venv, created with uv (Python 3.12, pydantic>=2, huggingface_hub,
#      pyyaml).
# Then: exec "$PY" -m benchmarks.cluster_bench "$@" with the repo root on
# PYTHONPATH. Not an engine launch script: it execs Python and starts nothing in
# the background.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
SUBCOMMAND="${1:-}"
PYTHON_VERSION="3.12"
LAPTOP_VENV="$HOME/.cluster_bench/venv"

die() {
    echo "cluster_bench.sh: $*" >&2
    exit 1
}

# The value of --workdir among the arguments, if any.
workdir_from_args() {
    local prev=""
    local arg
    for arg in "$@"; do
        if [ "$prev" = "--workdir" ]; then
            echo "$arg"
            return 0
        fi
        case "$arg" in
            --workdir=*)
                echo "${arg#--workdir=}"
                return 0
                ;;
            --)
                return 0
                ;;
        esac
        prev="$arg"
    done
    return 0
}

node_workdir() {
    local wd
    wd="$(workdir_from_args "$@")"
    if [ -z "$wd" ]; then
        wd="${CLUSTER_BENCH_WORKDIR:-}"
    fi
    if [ -z "$wd" ] && [ "$(uname -s)" = "Linux" ] && [ "$(id -u)" = "0" ]; then
        wd="/root/cluster_bench"
    fi
    echo "$wd"
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    local candidate
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# ~/.cluster_bench/venv for the laptop side (controller, dry-run, report, kb).
laptop_python() {
    # huggingface_hub: the CLI imports the whole harness (model fetch included).
    local want="python=$PYTHON_VERSION pydantic>=2 pyyaml huggingface_hub"
    local marker="$LAPTOP_VENV/.cluster_bench_deps"
    if [ -x "$LAPTOP_VENV/bin/python" ] && [ -f "$marker" ] && [ "$(cat "$marker")" = "$want" ]; then
        echo "$LAPTOP_VENV/bin/python"
        return 0
    fi
    local uv
    if ! uv="$(find_uv)"; then
        cat >&2 <<EOF
cluster_bench.sh: uv is required to create $LAPTOP_VENV (Python $PYTHON_VERSION).
  install it:  curl -LsSf https://astral.sh/uv/install.sh | sh    (or: brew install uv)
  docs:        https://docs.astral.sh/uv/getting-started/installation/
  or point CLUSTER_BENCH_PYTHON at a Python >= 3.11 with pydantic>=2, pyyaml and huggingface_hub.
EOF
        exit 1
    fi
    mkdir -p "$HOME/.cluster_bench"
    if [ ! -x "$LAPTOP_VENV/bin/python" ]; then
        rm -rf "$LAPTOP_VENV"
        "$uv" venv --python "$PYTHON_VERSION" "$LAPTOP_VENV" >&2
    fi
    "$uv" pip install --python "$LAPTOP_VENV/bin/python" "pydantic>=2" pyyaml huggingface_hub >&2
    echo "$want" > "$marker"
    echo "$LAPTOP_VENV/bin/python"
}

PY="${CLUSTER_BENCH_PYTHON:-}"
if [ -z "$PY" ] && [ "$SUBCOMMAND" != "controller" ]; then
    WORKDIR="$(node_workdir "$@")"
    if [ -n "$WORKDIR" ] && [ -x "$WORKDIR/venv/bin/python" ]; then
        PY="$WORKDIR/venv/bin/python"
    elif [ -n "$WORKDIR" ] && [ "$SUBCOMMAND" = "run" ] && [ "$(uname -s)" = "Linux" ]; then
        echo "cluster_bench.sh: bootstrapping $WORKDIR/venv" >&2
        bash "$SCRIPT_DIR/scripts/bootstrap_node.sh" --workdir "$WORKDIR" >&2
        PY="$WORKDIR/venv/bin/python"
        [ -x "$PY" ] || die "bootstrap_node.sh did not create $PY"
    fi
fi
if [ -z "$PY" ]; then
    PY="$(laptop_python)"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Do not let the caller's working directory shadow the repo's packages.
export PYTHONSAFEPATH=1
exec "$PY" -m benchmarks.cluster_bench "$@"
