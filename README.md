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
