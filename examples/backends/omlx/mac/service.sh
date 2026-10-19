#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start, stop and inspect oMLX with the Dynamo bridge on this Mac.
#
#   ./service.sh start     # in the background, waiting until the model is loaded
#   ./service.sh status    # process, model and KV event state
#   ./service.sh logs      # follow the log
#   ./service.sh stop | restart
#   ./service.sh run       # in the foreground; Ctrl-C stops it
#
# Settings come from ~/.config/dynamo-omlx/omlx-mac.env (setup.sh writes it).
# A background service survives the end of the terminal or SSH session and is
# restarted when it exits unexpectedly. The Mac stays awake while it runs.
# Nothing starts it again after a reboot.

set -euo pipefail
# shellcheck source=SCRIPTDIR/common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

LOG_ROTATE_BYTES=$((512 * 1024 * 1024))

usage() {
    cat <<EOF
Usage: $0 [--config FILE] [--no-wait] {start|stop|restart|status|logs|run}

  start      Run in the background and wait until the model is loaded
  stop       Stop the background service
  restart    stop, then start
  status     Show the process, model and KV event state
  logs       Follow the log
  run        Run in the foreground (Ctrl-C stops it)

  --config FILE  Settings file (default: $CONFIG_FILE)
  --no-wait      Return from start without waiting for the model
EOF
}

# The supervisor's pid if it is running.
running_pid() {
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
    [[ -n "$pid" ]] && ps -o command= -p "$pid" 2>/dev/null | grep -q "service.sh.*_supervise" ||
        return 1
    echo "$pid"
}

listener_pid() {
    lsof -nP -iTCP:"$OMLX_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true
}

# GET a bridge route. The API key goes through stdin, never onto a command line.
api_get() {
    local host=$OMLX_HOST
    case $host in
        0.0.0.0 | ::) host=127.0.0.1 ;;
    esac
    printf 'header = "Authorization: Bearer %s"\n' "${OMLX_API_KEY:-}" |
        curl -fsS --max-time 5 -K - "http://$host:$OMLX_PORT$1"
}

# "<model state> <block size> <KV event state> <context length> <MTP draft
# depth, or off>" from the bridge; an older bridge leaves out the last field.
bridge_info() {
    [[ -n "${OMLX_PYTHON:-}" ]] || return 1
    api_get /dynamo/v1/info 2>/dev/null | "$OMLX_PYTHON" -c '
import json, sys
info = json.load(sys.stdin)
model = info["model"]
mtp = model.get("mtp_depth", "")
print(model["state"], model["block_size"], info["kv"]["state"], model["context_length"],
      "off" if mtp is None else mtp)
' 2>/dev/null
}

build_command() {
    local -a extra
    CMD=("$OMLX_PYTHON" -m dynamo_omlx.bridge --model-id "$OMLX_BRIDGE_MODEL_ID")
    if [[ -n "${OMLX_KV_BLOCK_SIZE:-}" ]]; then
        CMD+=(--kv-block-size "$OMLX_KV_BLOCK_SIZE")
    fi
    if [[ -n "${OMLX_MTP:-}" ]]; then
        CMD+=(--mtp "$OMLX_MTP")
    fi
    if [[ -n "${OMLX_MTP_DEPTH:-}" ]]; then
        CMD+=(--mtp-depth "$OMLX_MTP_DEPTH")
    fi
    if [[ -n "${OMLX_BRIDGE_ARGS:-}" ]]; then
        read -r -a extra <<<"$OMLX_BRIDGE_ARGS"
        CMD+=(${extra[@]+"${extra[@]}"})
    fi
    # oMLX reads the API key from OMLX_API_KEY, which keeps it out of `ps`.
    CMD+=(-- serve
        --model-dir "$OMLX_MODEL_DIR"
        --host "$OMLX_HOST"
        --port "$OMLX_PORT"
        --paged-ssd-cache-dir "$OMLX_SSD_CACHE_DIR")
    if [[ -n "${OMLX_SSD_CACHE_MAX_SIZE:-}" ]]; then
        CMD+=(--paged-ssd-cache-max-size "$OMLX_SSD_CACHE_MAX_SIZE")
    fi
    if [[ -n "${OMLX_MAX_CONCURRENT_REQUESTS:-}" ]]; then
        CMD+=(--max-concurrent-requests "$OMLX_MAX_CONCURRENT_REQUESTS")
    fi
    if [[ -n "${OMLX_SERVE_ARGS:-}" ]]; then
        read -r -a extra <<<"$OMLX_SERVE_ARGS"
        CMD+=(${extra[@]+"${extra[@]}"})
    fi
}

preflight() {
    local model="$OMLX_MODEL_DIR/$OMLX_BRIDGE_MODEL_ID" listener out
    require_omlx_python
    if [[ ! -f "$model/config.json" ]]; then
        if [[ -d "$OMLX_MODEL_DIR/.downloading/$OMLX_BRIDGE_MODEL_ID" ]]; then
            die "$OMLX_BRIDGE_MODEL_ID is still downloading; rerun download_model.sh until it finishes"
        fi
        die "no model in $model; run download_model.sh first"
    fi
    case ${OMLX_MTP:-} in
        "" | on | off) ;;
        *) die "OMLX_MTP in $CONFIG_FILE must be on, off or empty, not '$OMLX_MTP'" ;;
    esac
    case ${OMLX_MTP_DEPTH:-} in
        "" | [1-8]) ;;
        *) die "OMLX_MTP_DEPTH in $CONFIG_FILE must be 1 to 8 or empty, not '$OMLX_MTP_DEPTH'" ;;
    esac
    case $OMLX_HOST in
        127.0.0.1 | localhost | ::1) ;;
        *) [[ -n "${OMLX_API_KEY:-}" ]] ||
            die "set OMLX_API_KEY in $CONFIG_FILE; oMLX refuses to listen on $OMLX_HOST without one" ;;
    esac
    listener=$(listener_pid)
    if [[ -n "$listener" ]]; then
        die "port $OMLX_PORT is taken by pid $listener: $(ps -o command= -p "$listener" | cut -c1-100)
Stop it (quit the oMLX app, or run 'omlx stop' or 'brew services stop omlx'), or change OMLX_PORT"
    fi
    out=$(bridge --check 2>&1) || die "this oMLX build lacks what the bridge hooks into:
$out"
    mkdir -p "$OMLX_SSD_CACHE_DIR" "$LOG_DIR"
    build_command
}

cmd_run() {
    local pid
    if pid=$(running_pid); then
        die "the background service is running (pid $pid); stop it first"
    fi
    preflight
    echo "Starting: ${CMD[*]}"
    # exec keeps this pid, so caffeinate holds the Mac awake until oMLX exits.
    caffeinate -ims -w $$ &
    cd "$EXAMPLE_DIR"
    exec "${CMD[@]}"
}

# Runs detached under start: restart oMLX whenever it exits, until stopped.
cmd_supervise() {
    local child="" stopping="" rc started delay=$RESTART_DELAY
    require_omlx_python
    build_command
    trap 'stopping=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true' TERM INT
    caffeinate -ims -w $$ &
    cd "$EXAMPLE_DIR"
    while [[ -z "$stopping" ]]; do
        echo "$(date '+%F %T') service.sh: starting ${CMD[*]}"
        started=$SECONDS
        "${CMD[@]}" &
        child=$!
        rc=0
        wait "$child" || rc=$?
        # A trapped signal interrupts wait; keep waiting for oMLX to finish.
        while kill -0 "$child" 2>/dev/null; do
            rc=0
            wait "$child" || rc=$?
        done
        child=""
        [[ -z "$stopping" ]] || break
        # Back off while it keeps failing early, for example on a bad setting.
        if ((SECONDS - started >= 60)); then
            delay=$RESTART_DELAY
        fi
        echo "$(date '+%F %T') service.sh: exited with status $rc; restarting in ${delay}s"
        sleep "$delay" &
        wait $! || true
        delay=$((delay * 2 > 300 ? 300 : delay * 2))
    done
    echo "$(date '+%F %T') service.sh: stopped"
}

cmd_start() {
    local pid
    if pid=$(running_pid); then
        echo "Already running (pid $pid)."
        cmd_status || true
        return
    fi
    preflight
    if [[ -f "$LOG_FILE" ]] && (($(stat -f %z "$LOG_FILE") > LOG_ROTATE_BYTES)); then
        mv -f "$LOG_FILE" "$LOG_FILE.1"
    fi
    LOG_START=0
    if [[ -f "$LOG_FILE" ]]; then
        LOG_START=$(stat -f %z "$LOG_FILE")
    fi
    detach /bin/bash "$MAC_DIR/service.sh" --config "$CONFIG_FILE" _supervise \
        >>"$LOG_FILE" 2>&1 </dev/null
    echo "$DETACHED_PID" >"$PID_FILE"
    echo "Started (pid $DETACHED_PID); log: $LOG_FILE"
    [[ -n "$NO_WAIT" ]] || wait_ready
}

wait_ready() {
    local deadline=$((SECONDS + START_TIMEOUT)) info state last="none"
    echo "Waiting for $OMLX_BRIDGE_MODEL_ID to load (Ctrl-C stops waiting, not the service)"
    while ((SECONDS < deadline)); do
        running_pid >/dev/null || die "the service exited; see $LOG_FILE"
        # The supervisor would keep restarting a start that fails, for example
        # on a wrong model id; report it instead.
        if tail -c +$((LOG_START + 1)) "$LOG_FILE" | grep -q "service.sh: exited with status"; then
            echo "oMLX exited while starting. The end of $LOG_FILE:" >&2
            tail -c +$((LOG_START + 1)) "$LOG_FILE" | tail -n 20 >&2
            cmd_stop >/dev/null
            die "start failed and the service is stopped; fix the cause and start again"
        fi
        info=$(bridge_info) || info=""
        state=${info%% *}
        if [[ "$state" != "$last" ]]; then
            echo "  ${state:-starting}"
            last=$state
        fi
        case $state in
            ready | unloaded)
                # unloaded: OMLX_BRIDGE_ARGS has --no-preload.
                cmd_status || true
                return
                ;;
            failed | missing) die "the model is $state; see $LOG_FILE" ;;
        esac
        sleep 5
    done
    die "not ready after ${START_TIMEOUT}s; see ./service.sh status and $LOG_FILE"
}

cmd_stop() {
    local pid i
    if ! pid=$(running_pid); then
        echo "Not running."
        rm -f "$PID_FILE"
        return
    fi
    echo "Stopping (pid $pid)"
    kill -TERM "$pid"
    for ((i = 0; i < 120; i++)); do
        running_pid >/dev/null || break
        sleep 0.5
    done
    if running_pid >/dev/null; then
        echo "Still running after 60s; killing it"
        pkill -KILL -P "$pid" || true
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "Stopped."
}

cmd_status() {
    local pid listener info state block kv context mtp
    if pid=$(running_pid); then
        echo "Service:  running (pid $pid), log $LOG_FILE"
    else
        echo "Service:  not running"
    fi
    listener=$(listener_pid)
    if [[ -n "$listener" ]]; then
        echo "Port:     $OMLX_PORT, pid $listener: $(ps -o command= -p "$listener" | cut -c1-80)"
    fi
    find_omlx_python || true
    info=$(bridge_info) || info=""
    if [[ -z "$info" ]]; then
        echo "Bridge:   not answering on port $OMLX_PORT"
        return 1
    fi
    read -r state block kv context mtp <<<"$info"
    echo "Model:    $OMLX_BRIDGE_MODEL_ID $state, block size $block, context $context"
    echo "KV:       events $kv"
    # Known once the model is loaded; an older bridge does not report it.
    case $state:$mtp in
        ready:off) echo "MTP:      off" ;;
        ready:?*) echo "MTP:      on, at most $mtp draft tokens per cycle" ;;
    esac
    echo "URL:      http://$(mac_address):$OMLX_PORT (for OMLX_BRIDGE_URLS on Linux)"
    [[ "$state" == ready ]]
}

NO_WAIT=""
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
            shift 2
            ;;
        --no-wait)
            NO_WAIT=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        run | start | stop | restart | status | logs | _supervise)
            [[ -z "$COMMAND" ]] || die "give one command (see --help)"
            COMMAND=$1
            shift
            ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
if [[ -z "$COMMAND" ]]; then
    usage >&2
    exit 1
fi

load_config
START_TIMEOUT="${OMLX_START_TIMEOUT:-1200}"
RESTART_DELAY="${OMLX_RESTART_DELAY:-10}"
NAME=$(basename "${CONFIG_FILE%.env}")
PID_FILE="$(dirname "$CONFIG_FILE")/$NAME.pid"
LOG_FILE="$LOG_DIR/$NAME.log"
mkdir -p "$(dirname "$PID_FILE")" "$LOG_DIR"

case $COMMAND in
    run) cmd_run ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart)
        cmd_stop
        cmd_start
        ;;
    status) cmd_status ;;
    logs) exec tail -n 100 -F "$LOG_FILE" ;;
    _supervise) cmd_supervise ;;
esac
