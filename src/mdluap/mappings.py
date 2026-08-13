"""Targeted mapping wrappers used by both GAP modes."""

from __future__ import annotations

import torch
import numpy as np
from torch import nn


class TargetedUniversalMapping(nn.Module):
    """Apply one fixed, image-agnostic perturbation to every input."""

    def __init__(self, delta: torch.Tensor, epsilon: float):
        super().__init__()
        if delta.ndim != 4 or delta.shape[0] != 1:
            raise ValueError("delta must have shape [1, C, H, W]")
        self.register_buffer("delta", delta.detach().clone())
        self.epsilon = float(epsilon)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Add the fixed perturbation and keep pixels in the input range."""

        delta = self.delta.to(device=images.device, dtype=images.dtype)
        delta = delta.clamp(-self.epsilon, self.epsilon)
        return (images + delta).clamp(0.0, 1.0)


class TargetedImageDependentMapping(nn.Module):
    """Use a shared generator to produce a bounded residual for each input."""

    def __init__(self, generator: nn.Module, epsilon: float):
        super().__init__()
        self.generator = generator
        self.epsilon = float(epsilon)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Decode the generator output into an L-infinity-bounded mapping."""

        raw_delta = self.generator(images)
        # The official GAP generator ends with tanh, so its output is already signed.
        delta = raw_delta.clamp(-1.0, 1.0) * self.epsilon
        return (images + delta).clamp(0.0, 1.0)


class ImageDependentPQMapping(nn.Module):
    """Generate p(x)+q(x) with a learnable global L-infinity bound."""

    def __init__(self, generator: nn.Module, epsilon_max: float, epsilon_init_ratio: float = 0.999):
        super().__init__()
        self.generator = generator
        self.epsilon_max = float(epsilon_max)
        if not 0.0 < epsilon_init_ratio < 1.0:
            raise ValueError("epsilon_init_ratio must be between 0 and 1")
        initial_logit = np.log(epsilon_init_ratio / (1.0 - epsilon_init_ratio))
        self.epsilon_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def effective_epsilon(self) -> torch.Tensor:
        """Return the learned raw-pixel L-infinity bound."""

        return self.epsilon_max * torch.sigmoid(self.epsilon_logit)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return a p+q image with a differentiable learned epsilon bound."""

        raw = self.generator(images)
        if raw.shape[1] != 6:
            raise RuntimeError("imdep_pq generator must output six channels")
        raw_p, raw_q = raw.chunk(2, dim=1)
        # The official GAP generator ends in tanh. Each branch is an image
        # component in [0, 0.5], so p(x) + q(x) is a complete image in [0, 1].
        p = (raw_p + 1.0) / 4.0
        q = (raw_q + 1.0) / 4.0
        raw_delta = p + q - images
        epsilon = self.effective_epsilon().to(dtype=images.dtype)
        # Smooth clipping gives attack loss a gradient with respect to epsilon;
        # a hard clamp would make the epsilon penalty stay at its upper bound.
        delta = epsilon * torch.tanh(raw_delta / epsilon.clamp_min(1e-8))
        return (images + delta).clamp(0.0, 1.0)
