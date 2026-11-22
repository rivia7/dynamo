#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# In-container launcher for cluster_bench; one role per container:
#   control  - etcd (+ nats-server) for the Dynamo control plane, on node0
#   frontend - dynamo.frontend (KV-aware router); port from DYN_HTTP_PORT
#   worker   - python3 -m dynamo.sglang with the rendered --config and args
#              (under Nsight Systems when CB_NSYS_OUTPUT is set, see below);
#              with CB_DECODE_ONLY=1 the D-Only shim agents/sglang_decode_only.py
#              (dynamo.sglang decode without a prefill peer, fake KV transfer)
# All settings arrive through the container env file rendered by
# benchmarks/cluster_bench/adapters/engine_sglang.py.

set -e

ROLE="${1:-}"
case "$ROLE" in
    control|frontend|worker) shift ;;
    *) echo "usage: launch_node.sh control|frontend|worker [args...]" >&2; exit 2 ;;
esac

export DYNAMO_HOME="${DYNAMO_HOME:-/bench/src}"
source "${DYNAMO_HOME}/examples/common/gpu_utils.sh"
source "${DYNAMO_HOME}/examples/common/launch_utils.sh"

# Install the cleanup trap only after argument validation so a usage error
# cannot tear down the caller's process group.
trap dynamo_exit_trap EXIT

case "$ROLE" in
    control)
        ADVERTISE_IP="${CB_ADVERTISE_IP:?CB_ADVERTISE_IP is required}"
        ETCD_PORT="$(dyn_port CB_ETCD_PORT "" 2379)"
        NATS_PORT="$(dyn_port CB_NATS_PORT "" 4222)"
        print_launch_banner --no-curl "cluster_bench control plane (etcd + nats)" "-" "$ETCD_PORT" \
            "advertise: ${ADVERTISE_IP}"
        etcd --name cb-etcd --data-dir /tmp/cb-etcd \
            --listen-client-urls "http://0.0.0.0:${ETCD_PORT}" \
            --advertise-client-urls "http://${ADVERTISE_IP}:${ETCD_PORT}" &
        nats-server -p "$NATS_PORT" -js &
        ;;
    frontend)
        HTTP_PORT="$(dyn_port DYN_HTTP_PORT "" 8000)"
        export DYN_HTTP_PORT="$HTTP_PORT"
        print_launch_banner "cluster_bench Dynamo frontend" "${DYN_NAMESPACE:-dynamo}" "$HTTP_PORT"
        python3 -m dynamo.frontend "$@" &
        ;;
    worker)
        GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
        DYN_SYSTEM_PORT="$(dyn_port DYN_SYSTEM_PORT "" 8081)"
        export DYN_SYSTEM_PORT
        print_launch_banner --no-curl "cluster_bench SGLang worker" "${DYN_NAMESPACE:-dynamo}" "$DYN_SYSTEM_PORT"
        # Opt-in diagnostic window (adapters/diagnostics.py). CB_NSYS_OUTPUT is
        # a report path prefix inside the container; the worker then runs under
        # `nsys profile`, which records only the forward steps requested via
        # POST /engine/control/start_profile on DYN_SYSTEM_PORT. CB_NSYS_FLAGS
        # (whitespace-separated, without -o) replaces the default flag list and
        # CB_NSYS_BIN the nsys found on PATH. Without nsys the worker starts
        # unprofiled after one warning.
        nsys_prefix=()
        if [[ -n "${CB_NSYS_OUTPUT:-}" ]]; then
            nsys_bin="${CB_NSYS_BIN:-$(command -v nsys || true)}"
            if [[ -z "$nsys_bin" || ! -x "$nsys_bin" ]]; then
                echo "WARNING: CB_NSYS_OUTPUT is set but nsys is not available; starting the worker without profiling" >&2
            elif ! mkdir -p -- "$(dirname -- "$CB_NSYS_OUTPUT")"; then
                echo "WARNING: cannot create the directory of CB_NSYS_OUTPUT; starting the worker without profiling" >&2
            else
                nsys_flags=()
                if [[ -n "${CB_NSYS_FLAGS:-}" ]]; then
                    # read -a splits on whitespace without glob expansion.
                    read -r -a nsys_flags <<< "$CB_NSYS_FLAGS"
                fi
                if [[ ${#nsys_flags[@]} -eq 0 ]]; then
                    # Keep in sync with NSYS_DEFAULT_FLAGS in adapters/diagnostics.py.
                    nsys_flags=(
                        --trace=cuda,nvtx
                        --sample=none
                        --cpuctxsw=none
                        --trace-fork-before-exec=true
                        --cuda-graph-trace=node
                        --capture-range=cudaProfilerApi
                        --capture-range-end=stop
                        --force-overwrite=true
                    )
                fi
                nsys_prefix=("$nsys_bin" profile "${nsys_flags[@]}" -o "$CB_NSYS_OUTPUT")
                echo "cluster_bench: worker runs under Nsight Systems: ${nsys_prefix[*]}"
            fi
        fi
        worker_entry=(-m dynamo.sglang)
        if [[ "${CB_DECODE_ONLY:-0}" == "1" ]]; then
            # -P: the shim's directory must not shadow installed modules.
            worker_entry=(-P "${DYNAMO_HOME}/benchmarks/cluster_bench/agents/sglang_decode_only.py")
            echo "cluster_bench: D-Only worker (fake KV transfer, no prefill peer)"
        fi
        # GPU_MEM_ARGS is intentionally unquoted: it is empty or a flag list.
        # shellcheck disable=SC2086
        "${nsys_prefix[@]}" python3 "${worker_entry[@]}" "$@" $GPU_MEM_ARGS &
        ;;
esac

wait_any_exit
