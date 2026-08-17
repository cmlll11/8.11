"""Targeted GAP training adapted to CIFAR-10 raw-pixel inputs."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mappings import ImageDependentPQMapping, TargetedImageDependentMapping


def seed_everything(seed: int) -> None:
    """Fix the mapping restart seed while leaving model training artifacts untouched."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_official_gap_generator(
    *, gap_root: str, device: torch.device, ngf: int, output_channels: int = 3
) -> nn.Module:
    """Construct the generator from the pinned GAP implementation."""

    root = str(Path(gap_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from material.models.generators import ResnetGenerator, weights_init

    if device.type != "cuda":
        raise RuntimeError("GAP's official generator is intended for the CUDA server run")
    generator = ResnetGenerator(
        3,
        int(output_channels),
        int(ngf),
        norm_type="batch",
        act_type="relu",
        gpu_ids=[device.index or 0],
    )
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
    max_batches: int = 50,
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
        for batch_index, (images, _labels) in enumerate(train_loader):
            if batch_index >= int(max_batches):
                break
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


def train_residual_gap(
    *,
    model: nn.Module,
    mapping: nn.Module,
    train_loader,
    attack_goal: str,
    target_label: int,
    epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_path: str | None = None,
    max_batches: int = 50,
) -> dict:
    """Train the official GAP objective for a separate x+f(x) mapping."""

    if attack_goal not in {"targeted", "non_targeted"}:
        raise ValueError("attack_goal must be targeted or non_targeted")
    optimizer = torch.optim.Adam(mapping.parameters(), lr=float(lr), betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    history: list[float] = []
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mapping.load_state_dict(state["mapping"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))

    model.eval()
    mapping.train()
    for epoch in range(start_epoch, int(epochs)):
        losses = []
        for batch_index, (images, _labels) in enumerate(train_loader):
            if batch_index >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                clean_logits = model(images)
                if attack_goal == "targeted":
                    attack_labels = torch.full(
                        (images.shape[0],), int(target_label), dtype=torch.long, device=device
                    )
                else:
                    # Classic GAP non-targeted training moves each image toward
                    # its least-likely class under the clean input.
                    attack_labels = clean_logits.argmin(dim=1)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(mapping(images)), attack_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        history.append(float(np.mean(losses)) if losses else float("nan"))
        if checkpoint_path:
            tmp_path = f"{checkpoint_path}.tmp"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "mapping": mapping.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                tmp_path,
            )
            Path(tmp_path).replace(checkpoint_path)
    return {"history": history, "epoch": int(epochs)}


def train_pq_gap(
    *,
    model: nn.Module,
    mapping: ImageDependentPQMapping,
    train_loader,
    attack_goal: str,
    target_label: int,
    epochs: int,
    lr: float,
    epsilon_lambda: float,
    device: torch.device,
    checkpoint_path: str | None = None,
    max_batches: int = 50,
    loss_mode: str = "legacy",
    attack_margin: float = 1.0,
    attack_temperature: float = 0.2,
    epsilon_lambda_start: float | None = None,
    epsilon_lambda_end: float | None = None,
    attack_lambda_start: float = 0.1,
    attack_lambda_end: float = 1.0,
    epsilon_warmup_epochs: int = 10,
) -> dict:
    """Train targeted or classic least-likely non-targeted p+q GAP."""

    if attack_goal not in {"targeted", "non_targeted"}:
        raise ValueError("attack_goal must be targeted or non_targeted")
    if loss_mode not in {"legacy", "min_radius"}:
        raise ValueError("loss_mode must be legacy or min_radius")
    optimizer = torch.optim.Adam(mapping.parameters(), lr=float(lr), betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    history: list[dict] = []
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mapping.load_state_dict(state["mapping"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))

    model.eval()
    mapping.train()
    for epoch in range(start_epoch, int(epochs)):
        attack_losses = []
        epsilon_losses = []
        total_losses = []
        attack_scores = []
        for batch_index, (images, _labels) in enumerate(train_loader):
            if batch_index >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                clean_logits = model(images)
                if attack_goal == "targeted":
                    attack_labels = torch.full(
                        (images.shape[0],), int(target_label), dtype=torch.long, device=device
                    )
                else:
                    # This matches classic GAP: move each image toward its
                    # least-likely class under the unperturbed classifier.
                    attack_labels = clean_logits.argmin(dim=1)

            optimizer.zero_grad(set_to_none=True)
            mapped = mapping(images)
            mapped_logits = model(mapped)
            if loss_mode == "legacy":
                attack_loss = criterion(mapped_logits, attack_labels)
                epsilon_loss = float(epsilon_lambda) * 255.0 * mapping.effective_epsilon()
                epsilon_weight = float(epsilon_lambda)
                attack_weight = 1.0
                attack_score = float("nan")
                total_loss = attack_loss + epsilon_loss
            else:
                # Stop rewarding excess confidence after the requested margin
                # is reached, so the optimizer can reduce the perturbation.
                target_logits = mapped_logits.gather(1, attack_labels[:, None]).squeeze(1)
                other_logits = mapped_logits.masked_fill(
                    F.one_hot(attack_labels, num_classes=mapped_logits.shape[1]).bool(),
                    float("-inf"),
                ).amax(dim=1)
                score = target_logits - other_logits
                attack_loss = F.softplus(
                    (float(attack_margin) - score) / float(attack_temperature)
                ).mean()
                warmup = max(int(epsilon_warmup_epochs), 1)
                progress = min(
                    max((epoch - warmup + 1) / max(int(epochs) - warmup, 1), 0.0),
                    1.0,
                )
                eps_start = 4.0 if epsilon_lambda_start is None else float(epsilon_lambda_start)
                eps_end = 1.0 if epsilon_lambda_end is None else float(epsilon_lambda_end)
                epsilon_weight = eps_start + (eps_end - eps_start) * progress
                attack_weight = float(attack_lambda_start) + (
                    float(attack_lambda_end) - float(attack_lambda_start)
                ) * progress
                epsilon_loss = epsilon_weight * (
                    mapping.effective_epsilon() / float(mapping.epsilon_max)
                )
                attack_score = float(score.detach().mean().cpu())
                total_loss = attack_weight * attack_loss + epsilon_loss
            total_loss.backward()
            optimizer.step()

            attack_losses.append(float(attack_loss.detach().cpu()))
            epsilon_losses.append(float(epsilon_loss.detach().cpu()))
            total_losses.append(float(total_loss.detach().cpu()))
            if loss_mode == "min_radius":
                attack_scores.append(float(attack_score))

        epoch_record = {
            "epoch": epoch + 1,
            "attack_loss": float(np.mean(attack_losses)) if attack_losses else float("nan"),
            "epsilon_loss": float(np.mean(epsilon_losses)) if epsilon_losses else float("nan"),
            "total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
            "learned_epsilon": float(mapping.effective_epsilon().detach().cpu()),
        }
        if loss_mode == "min_radius":
            epoch_record.update(
                {
                    "loss_mode": loss_mode,
                    "epsilon_weight": float(epsilon_weight),
                    "attack_weight": float(attack_weight),
                    "attack_margin": float(attack_margin),
                    "attack_score": float(np.mean(attack_scores)) if attack_scores else float("nan"),
                }
            )
        history.append(epoch_record)
        print(epoch_record, flush=True)
        if checkpoint_path:
            tmp_path = f"{checkpoint_path}.tmp"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "mapping": mapping.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                tmp_path,
            )
            Path(tmp_path).replace(checkpoint_path)
    return {"history": history, "epoch": int(epochs)}
