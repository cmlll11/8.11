#!/usr/bin/env bash

# Train p(x)+q(x) with a minimum-radius margin objective.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: compare clean/backdoor p+q GAP with epsilon-first margin loss"
log_step "Budget: seed=2026, epochs=60, batches_per_epoch=50, epsilon_init=4/255, epsilon_max=16/255"

for side in clean backdoor; do
    if [[ "${side}" == "clean" ]]; then
        result_path="${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt"
    else
        result_path="${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt"
    fi

    for attack_goal in targeted non_targeted; do
        run_dir="${REPO_ROOT}/artifacts/mappings/badnet_pq_minradius/${side}/${attack_goal}/seed2026"
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
            --epsilon-init-ratio 0.25 \
            --epsilon-lambda-start 4.0 \
            --epsilon-lambda-end 1.0 \
            --attack-lambda-start 0.1 \
            --attack-lambda-end 1.0 \
            --attack-margin 1.0 \
            --attack-temperature 0.2 \
            --epsilon-warmup-epochs 10 \
            --loss-mode min_radius \
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
    --root "${REPO_ROOT}/artifacts/mappings/badnet_pq_minradius" \
    --output "${REPO_ROOT}/reports/badnet_pq_minradius_summary.json" \
    --success-threshold 0.90
log_step "Complete: summary=${REPO_ROOT}/reports/badnet_pq_minradius_summary.json"
