#!/usr/bin/env bash

# Train the seed-0 clean/BadNets classifier pair with official BackdoorBench.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_LABEL=0
MODEL_SEED=0
CLEAN_RUN="mdl_uap_clean_seed0"
BACKDOOR_RUN="mdl_uap_badnet_seed0"

train_one() {
    local run_name="$1"
    local poison_ratio="$2"
    local result_path="${BACKDOORBENCH_ROOT}/record/${run_name}/attack_result.pt"
    local run_dir="${BACKDOORBENCH_ROOT}/record/${run_name}"

    if [[ -f "${result_path}" ]]; then
        log_step "Skip completed classifier: ${run_name}"
        return
    fi
    if [[ -d "${run_dir}" ]]; then
        echo "ERROR: incomplete run directory exists: ${run_dir}" >&2
        echo "Keep it for diagnosis; do not overwrite it." >&2
        exit 1
    fi

    log_step "Train classifier=${run_name} poison_ratio=${poison_ratio} target=${TARGET_LABEL} seed=${MODEL_SEED}"
    (
        cd "${BACKDOORBENCH_ROOT}"
        "${PYTHON_BIN}" attack/badnet.py \
            --yaml_path config/attack/prototype/cifar10.yaml \
            --bd_yaml_path config/attack/badnet/default.yaml \
            --patch_mask_path resource/badnet/trigger_image.png \
            --dataset_path ./data \
            --save_folder_name "${run_name}" \
            --pratio "${poison_ratio}" \
            --attack_target "${TARGET_LABEL}" \
            --random_seed "${MODEL_SEED}" \
            --frequency_save 10 \
            --device cuda:0
    )
}

log_step "Purpose: produce matched clean and BadNets classifiers with official BackdoorBench"
log_step "Input: CIFAR-10 at ${DATA_ROOT}; physical GPU ${GPU_ID}"
train_one "${CLEAN_RUN}" 0.0
train_one "${BACKDOOR_RUN}" 0.1

cp "${BACKDOORBENCH_ROOT}/record/${CLEAN_RUN}/attack_result.pt" "${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt"
cp "${BACKDOORBENCH_ROOT}/record/${BACKDOOR_RUN}/attack_result.pt" "${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt"
log_step "Complete: paired artifacts copied to ${REPO_ROOT}/artifacts/models"
