#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIPerf runs against the example's frontend, in the dynamo-frontend image,
# which ships aiperf. The workload comes from benchmark.env and the model from
# the overlay's profile.env. The requests must not set ignore_eos or
# min_tokens, which the oMLX proxy rejects, so the output lengths vary.
#
# 1. sweep: every concurrency runs for BENCHMARK_SECONDS, PROFILE_RUNS times,
#    and AIPerf reports confidence intervals over the runs.
# 2. shared-prefix: one request at a time, each starting with the same system
#    prompt of two blocks. The first request is cold; the later ones read the
#    prefix from oMLX's cache.

set -eu

OUT=/tmp/aiperf
# A new seed per Job gives new prompts, which an earlier Job cannot have left
# in oMLX's SSD cache. Set SEED to repeat a Job's prompts.
SEED="${SEED:-$(date +%s)}"
SHARED_PREFIX_TOKENS="${SHARED_PREFIX_TOKENS:-$((EXPECT_BLOCK_SIZE * 2))}"
echo "seed $SEED"

profile() {
    aiperf profile \
        --model "$SERVED_MODEL_NAME" \
        --url "http://$ENDPOINT" \
        --endpoint-type chat \
        --streaming \
        --use-server-token-count \
        --tokenizer "$OMLX_MODEL_REPO" \
        --tokenizer-revision "$OMLX_MODEL_REVISION" \
        --synthetic-input-tokens-mean "$INPUT_TOKENS" \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean "$OUTPUT_TOKENS" \
        --output-tokens-stddev 0 \
        --extra-inputs "max_tokens:$OUTPUT_TOKENS" \
        --random-seed "$SEED" \
        --workers-max 8 \
        --wait-for-model-timeout 600 \
        --ui-type simple \
        "$@"
}

# The log outlives the pod: copy AIPerf's CSV summaries into it, each after a
# "--- csv <path under /tmp/aiperf>" line. The per-request records, the
# per-phase copies and the frontend's metrics, which Prometheus keeps anyway,
# stay out.
dump() {
    find "$OUT/$1" -name '*.csv' ! -path '*/phases/*' ! -name 'server_metrics_export.csv' | sort |
        while read -r csv; do
            echo "--- csv ${csv#"$OUT"/}"
            cat "$csv"
        done
}

echo "=== sweep"
profile --artifact-dir "$OUT/sweep" \
    --concurrency "$CONCURRENCY" \
    --benchmark-duration "$BENCHMARK_SECONDS" \
    --benchmark-grace-period 60 \
    --warmup-duration "$WARMUP_SECONDS" \
    --num-profile-runs "$PROFILE_RUNS" \
    --profile-run-cooldown-seconds 10 \
    --parameter-sweep-cooldown-seconds 5
dump sweep

echo "=== shared-prefix"
profile --artifact-dir "$OUT/shared-prefix" \
    --concurrency 1 \
    --benchmark-duration "$PREFIX_SECONDS" \
    --benchmark-grace-period 60 \
    --num-dataset-entries 1000 \
    --shared-system-prompt-length "$SHARED_PREFIX_TOKENS"
dump shared-prefix
