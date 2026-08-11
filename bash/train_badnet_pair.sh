#!/usr/bin/env bash

# Train the seed-0 clean/BadNets classifier pair with official BackdoorBench.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_LABEL=0
MODEL_SEED=0
CLEAN_RUN="mdl_uap_clean_prototype_seed0"
BACKDOOR_RUN="mdl_uap_badnet_seed0"

train_clean() {
    local weights_path="${BACKDOORBENCH_ROOT}/record/${CLEAN_RUN}/clean_model.pth"
    local artifact_path="${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt"
    local run_dir="${BACKDOORBENCH_ROOT}/record/${CLEAN_RUN}"

    if [[ -f "${artifact_path}" ]]; then
        log_step "Skip completed classifier: ${CLEAN_RUN}"
        return
    fi
    if [[ ! -f "${weights_path}" && -d "${run_dir}" ]]; then
        echo "ERROR: incomplete run directory exists: ${run_dir}" >&2
        echo "Keep it for diagnosis; do not overwrite it." >&2
        exit 1
    fi

    if [[ ! -f "${weights_path}" ]]; then
        log_step "Train clean classifier=${CLEAN_RUN} seed=${MODEL_SEED}"
        (
            cd "${BACKDOORBENCH_ROOT}"
            "${PYTHON_BIN}" attack/prototype.py \
                --yaml_path config/attack/prototype/cifar10.yaml \
                --dataset_path ./data \
                --save_folder_name "${CLEAN_RUN}" \
                --random_seed "${MODEL_SEED}" \
                --frequency_save 10 \
                --device cuda:0
        )
    fi
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/package_clean_model.py" \
        --weights "${weights_path}" \
        --output "${artifact_path}"
}

train_backdoor() {
    local result_path="${BACKDOORBENCH_ROOT}/record/${BACKDOOR_RUN}/attack_result.pt"
    local run_dir="${BACKDOORBENCH_ROOT}/record/${BACKDOOR_RUN}"

    if [[ -f "${result_path}" ]]; then
        log_step "Skip completed classifier: ${BACKDOOR_RUN}"
        return
    fi
    if [[ -d "${run_dir}" ]]; then
        echo "ERROR: incomplete run directory exists: ${run_dir}" >&2
        echo "Keep it for diagnosis; do not overwrite it." >&2
        exit 1
    fi

    log_step "Train classifier=${BACKDOOR_RUN} poison_ratio=0.1 target=${TARGET_LABEL} seed=${MODEL_SEED}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        "${PYTHON_BIN}" attack/badnet.py \
            --yaml_path config/attack/prototype/cifar10.yaml \
            --bd_yaml_path config/attack/badnet/default.yaml \
            --patch_mask_path resource/badnet/trigger_image.png \
            --dataset_path ./data \
            --save_folder_name "${BACKDOOR_RUN}" \
            --pratio 0.1 \
            --attack_target "${TARGET_LABEL}" \
            --random_seed "${MODEL_SEED}" \
            --frequency_save 10 \
            --device cuda:0
    )
}

log_step "Purpose: produce matched clean and BadNets classifiers with official BackdoorBench"
log_step "Input: CIFAR-10 at ${DATA_ROOT}; physical GPU ${GPU_ID}"
train_clean
train_backdoor

cp "${BACKDOORBENCH_ROOT}/record/${BACKDOOR_RUN}/attack_result.pt" "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt"
log_step "Complete: paired artifacts copied to ${REPO_ROOT}/artifacts/models"
