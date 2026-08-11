"""Metrics shared by mapping training and held-out evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def targeted_asr(
    model,
    mapping,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_label: int,
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """Return the fraction of non-target inputs classified as target_label."""

    model.eval()
    mapping.eval()
    target_label = int(target_label)
    successes = 0
    total = 0
    for begin in range(0, len(images), int(batch_size)):
        batch_images = images[begin : begin + batch_size]
        batch_labels = labels[begin : begin + batch_size]
        keep = batch_labels != target_label
        if not bool(keep.any()):
            continue
        batch_images = batch_images[keep].to(device)
        predictions = model(mapping(batch_images)).argmax(dim=1)
        successes += int((predictions == target_label).sum())
        total += int(len(batch_images))
    return successes / max(total, 1)


@torch.no_grad()
def max_linf(images: torch.Tensor, mapped_images: torch.Tensor) -> float:
    """Return the largest per-pixel absolute input change."""

    return float((mapped_images - images).abs().amax().item())

