#!/usr/bin/env bash

# Verify clean accuracy, backdoor ASR, and clean-model trigger resistance.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CLEAN_RESULT="${BACKDOORBENCH_ROOT}/record/mdl_uap_clean_seed0/attack_result.pt"
BACKDOOR_RESULT="${BACKDOORBENCH_ROOT}/record/mdl_uap_badnet_seed0/attack_result.pt"

log_step "Purpose: check classifier gates before fitting any UAP"
log_step "Thresholds: clean accuracy >= 0.90, backdoor ASR >= 0.90, clean trigger ASR <= 0.10"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_model_pair.py" \
    --clean-result "${CLEAN_RESULT}" \
    --backdoor-result "${BACKDOOR_RESULT}" \
    --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --output "${REPO_ROOT}/reports/badnet_model_pair.json" \
    --device cuda:0
log_step "Complete: report=${REPO_ROOT}/reports/badnet_model_pair.json"
