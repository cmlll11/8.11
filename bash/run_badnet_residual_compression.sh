#!/usr/bin/env bash

# Train and compress independent fixed-epsilon x+f(x) GAP generators.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: compare x+f(x) GAP ASR after float32/int16/int8/int4 compression"
log_step "Budget: epsilon=4..16/255, targeted+non-targeted, epochs=60, batches_per_epoch=50"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_residual_compression.py" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${REPO_ROOT}/artifacts/mappings/badnet_residual_compression" \
    --report-json "${REPO_ROOT}/reports/badnet_residual_compression.json" \
    --report-csv "${REPO_ROOT}/reports/badnet_residual_compression.csv" \
    --clean-result "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" \
    --backdoor-result "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt" \
    --target 0 \
    --epsilon-start 4 \
    --epsilon-end 16 \
    --seed 2026 \
    --split-seed 2026 \
    --epochs 60 \
    --max-batches 50 \
    --batch-size 128 \
    --workers 4 \
    --lr 2e-4 \
    --ngf 64 \
    --device cuda:0

log_step "Complete: JSON=${REPO_ROOT}/reports/badnet_residual_compression.json CSV=${REPO_ROOT}/reports/badnet_residual_compression.csv"
