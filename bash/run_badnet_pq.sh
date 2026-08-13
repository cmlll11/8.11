#!/usr/bin/env bash

# Train and evaluate four image-dependent p(x)+q(x) GAP mappings.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: compare clean/backdoor p+q GAP under targeted and non-targeted goals"
log_step "Budget: seed=2026, epochs=60, batches_per_epoch=50, epsilon_max=16/255, epsilon_lambda=0.1"

for side in clean backdoor; do
    if [[ "${side}" == "clean" ]]; then
        result_path="${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt"
    else
        result_path="${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt"
    fi

    for attack_goal in targeted non_targeted; do
        run_dir="${REPO_ROOT}/artifacts/mappings/badnet_pq/${side}/${attack_goal}/seed2026"
        mkdir -p "${run_dir}"
        log_step "Train: side=${side} attack_goal=${attack_goal} output=${run_dir}"
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_pq_mapping.py" \
            --result "${result_path}" \
            --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
            --gap-root "${GAP_ROOT}" \
            --data-root "${DATA_ROOT}" \
            --output "${run_dir}" \
            --attack-goal "${attack_goal}" \
            --target 0 \
            --epsilon-max 16/255 \
            --epsilon-lambda 0.1 \
            --seed 2026 \
            --split-seed 2026 \
            --epochs 60 \
            --max-batches 50 \
            --device cuda:0

        for split in val test; do
            report="${run_dir}/${split}.json"
            log_step "Evaluate: side=${side} attack_goal=${attack_goal} split=${split}"
            "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_pq_mapping.py" \
                --result "${result_path}" \
                --mapping "${run_dir}/mapping.pt" \
                --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
                --gap-root "${GAP_ROOT}" \
                --data-root "${DATA_ROOT}" \
                --output "${report}" \
                --split "${split}" \
                --split-seed 2026 \
                --device cuda:0
        done
    done
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_pq_results.py" \
    --root "${REPO_ROOT}/artifacts/mappings/badnet_pq" \
    --output "${REPO_ROOT}/reports/badnet_pq_summary.json" \
    --success-threshold 0.90
log_step "Complete: summary=${REPO_ROOT}/reports/badnet_pq_summary.json"
