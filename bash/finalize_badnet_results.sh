#!/usr/bin/env bash

# Select by validation MDL and evaluate selected mappings on the test split.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SUMMARY="${REPO_ROOT}/reports/badnet_validation_summary.json"
log_step "Purpose: select minimum-bit mappings at validation ASR >= 0.90"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_validation.py" \
    --root "${REPO_ROOT}/artifacts/mappings/badnet" \
    --output "${SUMMARY}" \
    --asr-threshold 0.90

log_step "Purpose: evaluate only the selected mappings on the untouched test split"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_selected.py" \
    --summary "${SUMMARY}" \
    --repo-root "${REPO_ROOT}" \
    --device cuda:0
log_step "Complete: final report=${REPO_ROOT}/reports/badnet_test/summary.json"
