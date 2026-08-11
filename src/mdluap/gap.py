"""Targeted GAP training adapted to CIFAR-10 raw-pixel inputs."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mappings import TargetedImageDependentMapping


def seed_everything(seed: int) -> None:
    """Fix the mapping restart seed while leaving model training artifacts untouched."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_official_gap_generator(*, gap_root: str, device: torch.device, ngf: int) -> nn.Module:
    """Construct the generator from the pinned GAP implementation."""

    root = str(Path(gap_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from material.models.generators import ResnetGenerator, weights_init

    if device.type != "cuda":
        raise RuntimeError("GAP's official generator is intended for the CUDA server run")
    generator = ResnetGenerator(3, 3, int(ngf), norm_type="batch", act_type="relu", gpu_ids=[device.index or 0])
    generator.apply(weights_init)
    return generator


def build_mapping(
    *,
    mode: str,
    gap_root: str,
    epsilon: float,
    device: torch.device,
    ngf: int,
    image_size: int = 32,
) -> tuple[nn.Module, torch.Tensor | None]:
    """Create a fixed-noise universal mapping or image-dependent mapping."""

    generator = build_official_gap_generator(gap_root=gap_root, device=device, ngf=ngf)
    if mode == "universal":
        # The official GAP universal mode feeds one fixed random image to G.
        noise = torch.rand(1, 3, image_size, image_size, device=device) * 255.0
        return UniversalGAPMapping(generator, noise, epsilon), noise
    if mode == "imdep":
        return TargetedImageDependentMapping(generator, epsilon), None
    raise ValueError("mode must be universal or imdep")


class UniversalGAPMapping(nn.Module):
    """Use the official generator with one fixed noise input for every image."""

    def __init__(self, generator: nn.Module, noise: torch.Tensor, epsilon: float):
        super().__init__()
        self.generator = generator
        self.register_buffer("noise", noise.detach().clone())
        self.epsilon = float(epsilon)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Generate one shared residual and apply it to the raw images."""

        delta = self.generator(self.noise.expand(images.shape[0], -1, -1, -1))
        delta = delta.clamp(-1.0, 1.0) * self.epsilon
        return (images + delta).clamp(0.0, 1.0)


def train_targeted_gap(
    *,
    model: nn.Module,
    mapping: nn.Module,
    train_loader,
    target_label: int,
    epochs: int,
    lr: float,
    device: torch.device,
    start_epoch: int = 0,
    checkpoint_path: str | None = None,
) -> dict:
    """Optimize the targeted GAP objective and atomically checkpoint each epoch."""

    optimizer = torch.optim.Adam(mapping.parameters(), lr=float(lr), betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mapping.load_state_dict(state["mapping"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])

    mapping.train()
    history: list[float] = []
    for epoch in range(start_epoch, int(epochs)):
        losses = []
        for images, _labels in train_loader:
            images = images.to(device, non_blocking=True)
            targets = torch.full((images.shape[0],), int(target_label), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mapping(images)), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        if checkpoint_path:
            tmp_path = f"{checkpoint_path}.tmp"
            torch.save({"epoch": epoch + 1, "mapping": mapping.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path)
            Path(tmp_path).replace(checkpoint_path)
    return {"loss": history, "epoch": int(epochs)}
