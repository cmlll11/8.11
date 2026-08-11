#!/usr/bin/env bash

# Train and validate one targeted GAP candidate; rerunning resumes its checkpoint.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_SIDE="${1:?usage: run_mapping_candidate.sh clean|backdoor universal|imdep epsilon_pixels restart}"
MODE="${2:?missing GAP mode}"
EPSILON_PIXELS="${3:?missing epsilon in pixels}"
RESTART="${4:?missing restart 0, 1, or 2}"

if [[ "${MODEL_SIDE}" != "clean" && "${MODEL_SIDE}" != "backdoor" ]]; then
    echo "ERROR: model side must be clean or backdoor" >&2
    exit 1
fi
if [[ "${MODE}" != "universal" && "${MODE}" != "imdep" ]]; then
    echo "ERROR: mode must be universal or imdep" >&2
    exit 1
fi
if [[ ! "${EPSILON_PIXELS}" =~ ^(4|8|16|24|32)$ ]]; then
    echo "ERROR: epsilon must be one of 4, 8, 16, 24, 32" >&2
    exit 1
fi
if [[ ! "${RESTART}" =~ ^(0|1|2)$ ]]; then
    echo "ERROR: restart must be 0, 1, or 2" >&2
    exit 1
fi

if [[ "${MODEL_SIDE}" == "clean" ]]; then
    RESULT_PATH="${REPO_ROOT}/artifacts/models/clean_seed0_attack_result.pt"
else
    RESULT_PATH="${REPO_ROOT}/artifacts/models/badnet_seed0_attack_result.pt"
fi

MAPPING_SEED=$((2026 + RESTART))
EPSILON="${EPSILON_PIXELS}/255"
RUN_DIR="${REPO_ROOT}/artifacts/mappings/badnet/${MODEL_SIDE}/${MODE}/eps${EPSILON_PIXELS}/restart${RESTART}"
VAL_REPORT="${RUN_DIR}/val.json"

if [[ -f "${VAL_REPORT}" ]]; then
    log_step "Skip completed candidate: ${VAL_REPORT}"
    exit 0
fi

log_step "Purpose: fit one targeted GAP mapping and evaluate it on the fixed validation split"
log_step "Input: side=${MODEL_SIDE} mode=${MODE} epsilon=${EPSILON} target=0 mapping_seed=${MAPPING_SEED}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_mapping.py" \
    --result "${RESULT_PATH}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output "${RUN_DIR}" \
    --mode "${MODE}" \
    --target 0 \
    --epsilon "${EPSILON}" \
    --seed "${MAPPING_SEED}" \
    --split-seed 2026 \
    --device cuda:0

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_mapping.py" \
    --result "${RESULT_PATH}" \
    --mapping "${RUN_DIR}/mapping.pt" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --gap-root "${GAP_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output "${VAL_REPORT}" \
    --split val \
    --split-seed 2026 \
    --device cuda:0
log_step "Complete: validation report=${VAL_REPORT}"
