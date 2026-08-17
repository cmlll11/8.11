#!/usr/bin/env bash

# Analyze hidden features of the four existing p(x)+q(x) generators.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: compare quantized bottleneck-feature entropy of existing clean/backdoor p+q GAP mappings"
log_step "Budget: fixed test probe=1000 samples, shared symmetric 8-bit per-channel quantizer"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_hidden_feature_info.py" \
    --mapping-root "${REPO_ROOT}/artifacts/mappings/badnet_pq" \
    --clean-result "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" \
    --backdoor-result "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${REPO_ROOT}/artifacts/feature_info/badnet_pq" \
    --summary "${REPO_ROOT}/reports/badnet_hidden_feature_info.json" \
    --csv "${REPO_ROOT}/reports/badnet_hidden_feature_info.csv" \
    --n-samples 1000 \
    --batch-size 128 \
    --workers 4 \
    --quantization-bits 8 \
    --split-seed 2026 \
    --device cuda:0

log_step "Complete: summary=${REPO_ROOT}/reports/badnet_hidden_feature_info.json CSV=${REPO_ROOT}/reports/badnet_hidden_feature_info.csv"
