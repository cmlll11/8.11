#!/usr/bin/env bash

# Compare x+f GAP hidden features at smaller fixed epsilon budgets.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_LABEL=0
MODEL_SEED=0
EPSILON_PIXELS=(4 8 12)
MODEL_ROOT="${REPO_ROOT}/artifacts/models/triggers"
MODEL_REPORT="${REPO_ROOT}/reports/backdoorbench_trigger_gates.json"

train_mapping() {
    local trigger="$1"
    local result_path="$2"
    local goal="$3"
    local epsilon_pixels="$4"
    local mapping_root="${REPO_ROOT}/artifacts/mappings/badnet_residual_triggers_eps${epsilon_pixels}"
    local output="${mapping_root}/${trigger}/${goal}/seed2026"
    if [[ -f "${output}/mapping.pt" ]]; then
        log_step "Skip completed x+f GAP: epsilon=${epsilon_pixels}/255 trigger=${trigger} attack_goal=${goal}"
        return
    fi
    log_step "Train x+f GAP: epsilon=${epsilon_pixels}/255 trigger=${trigger} attack_goal=${goal}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_residual_mapping.py" \
        --result "${result_path}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --gap-root "${GAP_ROOT}" \
        --data-root "${DATA_ROOT}" \
        --output "${output}" \
        --attack-goal "${goal}" --target "${TARGET_LABEL}" \
        --epsilon "${epsilon_pixels}/255" --seed 2026 --split-seed 2026 \
        --epochs 60 --max-batches 50 --device cuda:0
}

log_step "Purpose: compare x+f GAP hidden-feature entropy at epsilon=4,8,12/255"
log_step "Budget: triggers=badnet,blended,lf,wanet; goals=targeted,non_targeted; probe=1000 samples"

if [[ ! -f "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" ]]; then
    echo "ERROR: missing existing clean model artifact" >&2
    exit 1
fi

for epsilon_pixels in "${EPSILON_PIXELS[@]}"; do
    mapping_root="${REPO_ROOT}/artifacts/mappings/badnet_residual_triggers_eps${epsilon_pixels}"
    feature_root="${REPO_ROOT}/artifacts/feature_info/badnet_residual_triggers_eps${epsilon_pixels}"
    summary="${REPO_ROOT}/reports/backdoorbench_residual_hidden_feature_info_eps${epsilon_pixels}.json"
    csv="${REPO_ROOT}/reports/backdoorbench_residual_hidden_feature_info_eps${epsilon_pixels}.csv"

    if [[ -f "${summary}" && -f "${csv}" ]]; then
        log_step "Skip completed epsilon analysis: epsilon=${epsilon_pixels}/255"
        continue
    fi

    log_step "Check model gates: epsilon=${epsilon_pixels}/255"
    trigger_args=()
    for trigger in badnet blended lf wanet; do
        # Official BackdoorBench dataset paths must remain under record/.
        trigger_args+=(
            --trigger-result
            "${trigger}=${BACKDOORBENCH_ROOT}/record/mdl_uap_${trigger}_seed${MODEL_SEED}/attack_result.pt"
        )
    done
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_backdoorbench_triggers.py" \
        --clean-result "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" \
        "${trigger_args[@]}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --output "${MODEL_REPORT}" --device cuda:0 --workers 4

    for goal in targeted non_targeted; do
        train_mapping clean "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" "${goal}" "${epsilon_pixels}"
        for trigger in badnet blended lf wanet; do
            train_mapping "${trigger}" "${MODEL_ROOT}/${trigger}_seed${MODEL_SEED}_attack_result.pt" "${goal}" "${epsilon_pixels}"
        done
    done

    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_hidden_feature_info_triggers.py" \
        --mapping-root "${mapping_root}" --model-report "${MODEL_REPORT}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" --gap-root "${GAP_ROOT}" \
        --data-root "${DATA_ROOT}" --output-root "${feature_root}" \
        --summary "${summary}" --csv "${csv}" --n-samples 1000 \
        --batch-size 128 --workers 4 --quantization-bits 8 \
        --split-seed 2026 --device cuda:0

    log_step "Complete epsilon analysis: epsilon=${epsilon_pixels}/255 summary=${summary}"
done

log_step "Complete: epsilon sweep reports are in ${REPO_ROOT}/reports/backdoorbench_residual_hidden_feature_info_eps{4,8,12}.json"
