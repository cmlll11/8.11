#!/usr/bin/env bash

# Train official trigger-family models, then analyze x+f GAP hidden features.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_LABEL=0
MODEL_SEED=0
MODEL_ROOT="${REPO_ROOT}/artifacts/models/triggers"
MODEL_REPORT="${REPO_ROOT}/reports/backdoorbench_trigger_gates.json"
MAPPING_ROOT="${REPO_ROOT}/artifacts/mappings/badnet_residual_triggers"
FEATURE_ROOT="${REPO_ROOT}/artifacts/feature_info/badnet_residual_triggers"
SUMMARY="${REPO_ROOT}/reports/backdoorbench_residual_hidden_feature_info.json"
CSV="${REPO_ROOT}/reports/backdoorbench_residual_hidden_feature_info.csv"

mkdir -p "${MODEL_ROOT}"

train_trigger() {
    local trigger="$1"
    local run_name="mdl_uap_${trigger}_seed${MODEL_SEED}"
    local result_path="${MODEL_ROOT}/${trigger}_seed${MODEL_SEED}_attack_result.pt"
    local run_dir="${BACKDOORBENCH_ROOT}/record/${run_name}"

    if [[ "${trigger}" == "badnet" && -f "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt" ]]; then
        cp "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt" "${result_path}"
        log_step "Reuse official BadNet artifact: ${result_path}"
        return
    fi
    if [[ -f "${result_path}" ]]; then
        log_step "Skip completed trigger model: ${trigger}"
        return
    fi
    if [[ -d "${run_dir}" ]]; then
        echo "ERROR: incomplete BackdoorBench run exists: ${run_dir}" >&2
        echo "Keep it for diagnosis; do not overwrite it." >&2
        exit 1
    fi

    log_step "Train official BackdoorBench trigger=${trigger}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        case "${trigger}" in
            badnet)
                "${PYTHON_BIN}" attack/badnet.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/badnet/default.yaml \
                    --patch_mask_path resource/badnet/trigger_image.png \
                    --dataset_path ./data --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${MODEL_SEED}" --frequency_save 10 --device cuda:0
                ;;
            blended)
                "${PYTHON_BIN}" attack/blended.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/blended/default.yaml \
                    --dataset_path ./data --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${MODEL_SEED}" --frequency_save 10 --device cuda:0
                ;;
            lf)
                "${PYTHON_BIN}" attack/lf.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/lf/default.yaml \
                    --dataset_path ./data --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${MODEL_SEED}" --frequency_save 10 --device cuda:0
                ;;
            wanet)
                "${PYTHON_BIN}" attack/wanet.py \
                    --yaml_path config/attack/prototype/cifar10.yaml \
                    --bd_yaml_path config/attack/wanet/default.yaml \
                    --dataset_path ./data --save_folder_name "${run_name}" \
                    --pratio 0.1 --attack_target "${TARGET_LABEL}" \
                    --random_seed "${MODEL_SEED}" --frequency_save 10 --device cuda:0
                ;;
            *)
                echo "ERROR: unsupported trigger=${trigger}" >&2
                exit 1
                ;;
        esac
    ) > "${REPO_ROOT}/outputs/backdoorbench_${trigger}.log" 2>&1
    cp "${BACKDOORBENCH_ROOT}/record/${run_name}/attack_result.pt" "${result_path}"
    log_step "Complete trigger model: ${trigger}"
}

train_mapping() {
    local trigger="$1"
    local result_path="$2"
    local goal="$3"
    local output="${MAPPING_ROOT}/${trigger}/${goal}/seed2026"
    if [[ -f "${output}/mapping.pt" ]]; then
        log_step "Skip completed x+f GAP: trigger=${trigger} attack_goal=${goal}"
        return
    fi
    log_step "Train x+f GAP: trigger=${trigger} attack_goal=${goal}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_residual_mapping.py" \
        --result "${result_path}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --gap-root "${GAP_ROOT}" \
        --data-root "${DATA_ROOT}" \
        --output "${output}" \
        --attack-goal "${goal}" --target "${TARGET_LABEL}" \
        --epsilon 16/255 --seed 2026 --split-seed 2026 \
        --epochs 60 --max-batches 50 --device cuda:0
}

log_step "Purpose: compare x+f GAP hidden-feature entropy across official trigger families"
log_step "Budget: triggers=badnet,blended,lf,wanet; goals=targeted,non_targeted; probe=1000 samples"

if [[ ! -f "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" ]]; then
    echo "ERROR: missing existing clean model artifact" >&2
    exit 1
fi

for trigger in badnet blended lf wanet; do
    train_trigger "${trigger}"
done

TRIGGER_ARGS=()
for trigger in badnet blended lf wanet; do
    TRIGGER_ARGS+=(--trigger-result "${trigger}=${MODEL_ROOT}/${trigger}_seed${MODEL_SEED}_attack_result.pt")
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_backdoorbench_triggers.py" \
    --clean-result "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" \
    "${TRIGGER_ARGS[@]}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --output "${MODEL_REPORT}" --device cuda:0 --workers 4

for goal in targeted non_targeted; do
    train_mapping clean "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt" "${goal}"
    for trigger in badnet blended lf wanet; do
        train_mapping "${trigger}" "${MODEL_ROOT}/${trigger}_seed${MODEL_SEED}_attack_result.pt" "${goal}"
    done
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_hidden_feature_info_triggers.py" \
    --mapping-root "${MAPPING_ROOT}" --model-report "${MODEL_REPORT}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" --output-root "${FEATURE_ROOT}" \
    --summary "${SUMMARY}" --csv "${CSV}" --n-samples 1000 \
    --batch-size 128 --workers 4 --quantization-bits 8 \
    --split-seed 2026 --device cuda:0

log_step "Complete: summary=${SUMMARY} CSV=${CSV}"

