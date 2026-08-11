"""Targeted mapping wrappers used by both GAP modes."""

from __future__ import annotations

import torch
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
