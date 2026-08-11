#!/usr/bin/env bash

# Run the complete BadNets validation grid; completed candidates are skipped.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log_step "Purpose: run clean/backdoor x GAP mode x epsilon x 3 mapping restarts"
log_step "Budget: target=0, epsilon=[4,8,16,24,32]/255, epochs=10, batches_per_epoch=50"
for side in clean backdoor; do
    for mode in universal imdep; do
        for epsilon in 4 8 16 24 32; do
            for restart in 0 1 2; do
                bash "${REPO_ROOT}/bash/run_mapping_candidate.sh" "${side}" "${mode}" "${epsilon}" "${restart}"
            done
        done
    done
done
log_step "Complete: all BadNets validation candidates are available"
