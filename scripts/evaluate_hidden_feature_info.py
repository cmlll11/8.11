"""Measure quantized hidden-feature entropy for the four existing mappings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from mdluap.data import cifar10_split, loader
from mdluap.gap import build_official_gap_generator
from mdluap.mappings import ImageDependentPQMapping
from mdluap.models import load_attack_result_model


CONDITIONS = (
    ("clean", "targeted"),
    ("clean", "non_targeted"),
    ("backdoor", "targeted"),
    ("backdoor", "non_targeted"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", required=True)
    parser.add_argument("--clean-result", required=True)
    parser.add_argument("--backdoor-result", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--gap-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quantization-bits", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def mapping_path(root: Path, side: str, goal: str) -> Path:
    return root / side / goal / "seed2026" / "mapping.pt"


def build_mapping(checkpoint_path: Path, *, gap_root: str, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "imdep_pq":
        raise ValueError(f"expected imdep_pq checkpoint: {checkpoint_path}")
    generator = build_official_gap_generator(
        gap_root=gap_root,
        device=device,
        ngf=int(checkpoint.get("ngf", 64)),
        output_channels=6,
    )
    mapping = ImageDependentPQMapping(
        generator,
        float(checkpoint["epsilon_max"]),
        epsilon_init_ratio=float(checkpoint.get("epsilon_init_ratio", 0.999)),
    )
    mapping.load_state_dict(checkpoint["mapping"], strict=True)
    return mapping.to(device).eval(), checkpoint


def find_bottleneck(generator: nn.Module) -> tuple[str, nn.Module]:
    """Use the last official ResNetBlock before the upsampling layers."""

    candidates = [
        (name, module)
        for name, module in generator.named_modules()
        if module.__class__.__name__ == "ResnetBlock"
    ]
    if not candidates:
        raise RuntimeError("could not find an official GAP ResnetBlock")
    return candidates[-1]


@torch.no_grad()
def collect_condition(
    *,
    mapping_path_value: Path,
    result_path: str,
    args: argparse.Namespace,
    dataset,
    device: torch.device,
) -> dict:
    mapping, checkpoint = build_mapping(mapping_path_value, gap_root=args.gap_root, device=device)
    model, _ = load_attack_result_model(
        result_path,
        backdoorbench_root=args.backdoorbench_root,
        device=device,
    )
    feature_name, feature_module = find_bottleneck(mapping.generator)
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError("bottleneck output must be a 4-D tensor")
        captured.append(output.detach().cpu().float())

    handle = feature_module.register_forward_hook(capture)
    targeted_success = targeted_total = 0
    fooled = correctly_classified = 0
    target_zero = target_zero_total = 0

    try:
        data_loader = loader(
            dataset,
            batch_size=args.batch_size,
            train=False,
            workers=args.workers,
        )
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_predictions = model(images).argmax(dim=1)
            mapped = mapping(images)
            mapped_predictions = model(mapped).argmax(dim=1)
            goal = str(checkpoint["attack_goal"])
            target = int(checkpoint["target"])
            if goal == "targeted":
                keep = labels != target
                targeted_success += int((mapped_predictions[keep] == target).sum())
                targeted_total += int(keep.sum())
            else:
                correct = clean_predictions == labels
                fooled += int((mapped_predictions[correct] != labels[correct]).sum())
                correctly_classified += int(correct.sum())
                target_keep = correct & (labels != target)
                target_zero += int((mapped_predictions[target_keep] == target).sum())
                target_zero_total += int(target_keep.sum())
    finally:
        handle.remove()

    features = torch.cat(captured, dim=0)
    if features.shape[0] != len(dataset):
        raise RuntimeError(f"captured {features.shape[0]} samples, expected {len(dataset)}")

    if str(checkpoint["attack_goal"]) == "targeted":
        probe_asr = targeted_success / max(targeted_total, 1)
        target_0_rate = None
    else:
        probe_asr = fooled / max(correctly_classified, 1)
        target_0_rate = target_zero / max(target_zero_total, 1)

    result = {
        "mapping_path": str(mapping_path_value.resolve()),
        "attack_goal": str(checkpoint["attack_goal"]),
        "target": int(checkpoint["target"]),
        "feature_layer": feature_name,
        "feature_shape": list(features.shape[1:]),
        "features": features,
        "probe_asr": float(probe_asr),
        "target_0_rate": None if target_0_rate is None else float(target_0_rate),
        "eligible": int(targeted_total if checkpoint["attack_goal"] == "targeted" else correctly_classified),
    }
    del model, mapping
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def shared_channel_scales(results: list[dict]) -> torch.Tensor:
    maxima = []
    for result in results:
        features = result["features"]
        maxima.append(features.abs().amax(dim=(0, 2, 3)))
    scale = torch.stack(maxima, dim=0).amax(dim=0) / 127.0
    return scale.clamp_min(1e-12)


def quantized_entropy(features: torch.Tensor, scale: torch.Tensor) -> float:
    """Return mean marginal Shannon entropy in bits per activation."""

    quantized = torch.round(features / scale.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int64)
    channel_entropies = []
    for channel in range(quantized.shape[1]):
        values = quantized[:, channel].reshape(-1) + 127
        counts = torch.bincount(values, minlength=255).double()
        probabilities = counts / counts.sum().clamp_min(1.0)
        probabilities = probabilities[probabilities > 0]
        channel_entropies.append(float((-probabilities * probabilities.log2()).sum()))
    return float(np.mean(channel_entropies))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quantization_bits != 8:
        raise ValueError("the minimal probe uses a shared 8-bit feature quantizer")
    if args.n_samples <= 0:
        raise ValueError("n-samples must be positive")

    device = torch.device(args.device)
    mapping_root = Path(args.mapping_root)
    dataset = cifar10_split(args.data_root, split="test", split_seed=args.split_seed)
    if args.n_samples > len(dataset):
        raise ValueError(f"n-samples={args.n_samples} exceeds test set size {len(dataset)}")
    dataset = Subset(dataset, list(range(args.n_samples)))

    results = []
    for side, goal in CONDITIONS:
        checkpoint_path = mapping_path(mapping_root, side, goal)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing mapping checkpoint: {checkpoint_path}")
        result_path = args.clean_result if side == "clean" else args.backdoor_result
        result = collect_condition(
            mapping_path_value=checkpoint_path,
            result_path=result_path,
            args=args,
            dataset=dataset,
            device=device,
        )
        result["side"] = side
        results.append(result)

    scale = shared_channel_scales(results)
    rows = []
    output_root = Path(args.output_root)
    for result in results:
        entropy = quantized_entropy(result["features"], scale)
        row = {
            "side": result["side"],
            "attack_goal": result["attack_goal"],
            "feature_layer": result["feature_layer"],
            "feature_shape": result["feature_shape"],
            "n_samples": args.n_samples,
            "quantization_bits": args.quantization_bits,
            "quantizer": "shared-symmetric-per-channel",
            "feature_entropy_bits_per_activation": entropy,
            "probe_asr": result["probe_asr"],
            "target_0_rate": result["target_0_rate"],
            "eligible": result["eligible"],
        }
        rows.append(row)
        write_json(output_root / result["side"] / result["attack_goal"] / "seed2026" / "hidden_info.json", row)

    comparisons = {}
    for goal in ("targeted", "non_targeted"):
        clean = next(row for row in rows if row["side"] == "clean" and row["attack_goal"] == goal)
        backdoor = next(row for row in rows if row["side"] == "backdoor" and row["attack_goal"] == goal)
        comparisons[goal] = {
            "backdoor_minus_clean_entropy_bits_per_activation": (
                backdoor["feature_entropy_bits_per_activation"]
                - clean["feature_entropy_bits_per_activation"]
            ),
            "backdoor_minus_clean_probe_asr": backdoor["probe_asr"] - clean["probe_asr"],
        }

    summary = {
        "protocol": "MDL-UAP-hidden-feature-v1",
        "split": "test",
        "n_samples": args.n_samples,
        "quantization_bits": args.quantization_bits,
        "quantizer": "shared-symmetric-per-channel",
        "feature_layer": rows[0]["feature_layer"],
        "feature_shape": rows[0]["feature_shape"],
        "results": rows,
        "comparisons": comparisons,
        "note": "Quantized hidden-feature entropy is an auxiliary probe, not a mutual-information estimate.",
    }
    write_json(Path(args.summary), summary)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
