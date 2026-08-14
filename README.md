# Targeted UAP MDL validation

This repository contains the new validation experiment for comparing the
description length of targeted universal mappings on clean and backdoored
CIFAR-10 classifiers.

The experiment has two mapping modes:

- `targeted_universal`: one fixed perturbation is shared by all inputs;
- `targeted_imdep`: one shared generator produces an input-dependent residual.

Both modes target the same class and use the same validation/test protocol.
The original GAP implementation is preserved under `third_party/GAP/` and is
adapted only where required for this CIFAR-10 experiment.

Clone with `git clone --recurse-submodules` so both pinned official sources
are available.

Model checkpoints, SSBA arrays, generated mappings and experiment outputs are
local/server assets and must not be committed to GitHub.

All GPU-server work is wrapped by scripts under `bash/`. The first validation
uses the matched seed-0 clean/BadNets pair before extending the same protocol to
the remaining attacks.

## Fast p(x) + q(x) validation

The follow-up experiment trains only image-dependent p+q mappings on the
existing clean/BadNets pair. It runs targeted and classic least-likely
non-targeted GAP for 60 epochs with an actual-perturbation penalty and a hard
`16/255` L-infinity bound:

```bash
env PYTHON_BIN=/path/to/python GPU_ID=0 DATA_ROOT=/path/to/data \
  bash bash/run_badnet_pq.sh
```

The four runs are stored under `artifacts/mappings/badnet_pq`, and the concise
comparison is written to `reports/badnet_pq_summary.json`.

The epsilon-learning follow-up uses the same four settings but stores results
under `artifacts/mappings/badnet_pq_epslearn` and writes
`reports/badnet_pq_epslearn_summary.json`.

## Matched-ASR fixed-epsilon validation

The matched-ASR experiment fixes epsilon from `4/255` through `16/255`,
records validation ASR after every epoch, and matches the 10%--90% ASR levels
within two percentage points. It keeps the p(x)+q(x) output head, uses the
official GAP `log(CrossEntropy)` attack objective, and does not put epsilon in
the loss. For each matched checkpoint it tests 16-, 8-, and 4-bit parameter
encodings, decodes each candidate, and keeps the shortest encoding whose
validation ASR remains within the matching tolerance. The final comparison
uses `minimum_valid_bits` at the same epsilon and ASR, plus the first epoch
that entered the ASR interval.

```bash
env PYTHON_BIN=/path/to/python GPU_ID=0 DATA_ROOT=/path/to/data \
  bash bash/run_badnet_matched_asr.sh
```

The official-loss rerun stores outputs separately under
`artifacts/mappings/badnet_matched_official`; the JSON and CSV reports are
`reports/badnet_matched_asr_official_summary.json` and
`reports/badnet_matched_asr_official.csv`, so the earlier custom-loss results
under `badnet_matched` are preserved. The concise pairwise report contains
validation/test ASR, first matched epoch, minimum valid bits, and the
Clean/Backdoor bits difference.

