# External code provenance

## GAP

- Repository: https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations
- Paper: Poursaeed et al., *Generative Adversarial Perturbations*, CVPR 2018
- Local source: `third_party/GAP/`
- Pinned commit: `a27f9dfa3ebd45221b8de95f66425666798aff21`
- Adaptation rule: retain the official universal/image-dependent targeted
  generator logic; change only the dataset/model interface and experiment
  bookkeeping needed for CIFAR-10.

## BackdoorBench

The six backdoor models will use the official SCLBD/BackdoorBench attack
implementations at one recorded commit. Upstream source is not modified.
Official SSBA assets are required; no synthetic fallback is allowed.

- Repository: https://github.com/SCLBD/BackdoorBench
- Pinned commit: `f02e3534645f0ee63d6848653062cd6c0d6c400d`
- Attack entry points: `badnet.py`, `blended.py`, `wanet.py`, `inputaware.py`,
  `lf.py`, and `ssba.py`.
