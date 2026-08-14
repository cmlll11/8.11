#!/usr/bin/env bash

# Compare Clean and Backdoor mappings at matched ASR and fixed epsilon.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: compare p+q GAP bits at matched ASR and fixed epsilon"
log_step "Budget: epsilon=4..16/255, ASR=10..90%, tolerance=2%, epochs=60, batches_per_epoch=50"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_matched_pq.py" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${REPO_ROOT}/artifacts/mappings/badnet_matched_official" \
    --report-json "${REPO_ROOT}/reports/badnet_matched_asr_official_summary.json" \
    --report-csv "${REPO_ROOT}/reports/badnet_matched_asr_official.csv" \
    --clean-result "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" \
    --backdoor-result "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt" \
    --target 0 \
    --epsilon-start 4 \
    --epsilon-end 16 \
    --asr-targets 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --match-tolerance 0.02 \
    --stop-asr 0.90 \
    --seed 2026 \
    --split-seed 2026 \
    --epochs 60 \
    --max-batches 50 \
    --batch-size 128 \
    --workers 4 \
    --lr 2e-4 \
    --ngf 64 \
    --device cuda:0

log_step "Complete: JSON=${REPO_ROOT}/reports/badnet_matched_asr_official_summary.json CSV=${REPO_ROOT}/reports/badnet_matched_asr_official.csv"
