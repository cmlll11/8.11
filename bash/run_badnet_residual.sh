#!/usr/bin/env bash

# Train four independent official GAP x+f(x) mappings.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: train separate official GAP x+f(x) generators for hidden-feature analysis"
log_step "Budget: seed=2026, epochs=60, batches_per_epoch=50, epsilon=16/255"

for side in clean backdoor; do
    if [[ "${side}" == "clean" ]]; then
        result_path="${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt"
    else
        result_path="${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt"
    fi

    for attack_goal in targeted non_targeted; do
        run_dir="${REPO_ROOT}/artifacts/mappings/badnet_residual/${side}/${attack_goal}/seed2026"
        mkdir -p "${run_dir}"
        log_step "Train: side=${side} attack_goal=${attack_goal} output=${run_dir}"
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_residual_mapping.py" \
            --result "${result_path}" \
            --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
            --gap-root "${GAP_ROOT}" \
            --data-root "${DATA_ROOT}" \
            --output "${run_dir}" \
            --attack-goal "${attack_goal}" \
            --target 0 \
            --epsilon 16/255 \
            --seed 2026 \
            --split-seed 2026 \
            --epochs 60 \
            --max-batches 50 \
            --device cuda:0
    done
done

log_step "Complete: x+f(x) checkpoints are in ${REPO_ROOT}/artifacts/mappings/badnet_residual"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_hidden_feature_info_residual.py" \
    --mapping-root "${REPO_ROOT}/artifacts/mappings/badnet_residual" \
    --clean-result "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" \
    --backdoor-result "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${REPO_ROOT}/artifacts/feature_info/badnet_residual" \
    --summary "${REPO_ROOT}/reports/badnet_hidden_feature_info_residual.json" \
    --csv "${REPO_ROOT}/reports/badnet_hidden_feature_info_residual.csv" \
    --n-samples 1000 \
    --batch-size 128 \
    --workers 4 \
    --quantization-bits 8 \
    --split-seed 2026 \
    --device cuda:0

log_step "Complete: summary=${REPO_ROOT}/reports/badnet_hidden_feature_info_residual.json CSV=${REPO_ROOT}/reports/badnet_hidden_feature_info_residual.csv"
