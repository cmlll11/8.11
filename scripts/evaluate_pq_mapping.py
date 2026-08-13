"""Evaluate one p(x)+q(x) GAP mapping and its description length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mdluap.codec import mapping_description_length_bits
from mdluap.data import cifar10_split, loader
from mdluap.gap import build_official_gap_generator
from mdluap.mappings import ImageDependentPQMapping
from mdluap.models import load_attack_result_model


@torch.no_grad()
def evaluate(model, mapping, data_loader, *, attack_goal: str, target: int, device: torch.device) -> dict:
    """Measure attack success and the distribution of actual perturbation sizes."""

    model.eval()
    mapping.eval()
    targeted_success = targeted_total = 0
    fooled = correctly_classified = 0
    target_zero = target_zero_total = 0
    linf_values = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_predictions = model(images).argmax(dim=1)
        mapped = mapping(images)
        mapped_predictions = model(mapped).argmax(dim=1)
        linf_values.extend((mapped - images).abs().flatten(1).amax(dim=1).cpu().tolist())

        if attack_goal == "targeted":
            keep = labels != int(target)
            targeted_success += int((mapped_predictions[keep] == int(target)).sum())
            targeted_total += int(keep.sum())
        else:
            correct = clean_predictions == labels
            fooled += int((mapped_predictions[correct] != labels[correct]).sum())
            correctly_classified += int(correct.sum())
            target_keep = correct & (labels != int(target))
            target_zero += int((mapped_predictions[target_keep] == int(target)).sum())
            target_zero_total += int(target_keep.sum())

    linf = np.asarray(linf_values, dtype=np.float64)
    metrics = {
        "attack_goal": attack_goal,
        "mean_linf": float(linf.mean()),
        "p95_linf": float(np.quantile(linf, 0.95)),
        "max_linf": float(linf.max()),
    }
    if attack_goal == "targeted":
        metrics.update(
            {
                "targeted_asr": targeted_success / max(targeted_total, 1),
                "eligible": targeted_total,
            }
        )
    else:
        metrics.update(
            {
                "fooling_rate": fooled / max(correctly_classified, 1),
                "eligible": correctly_classified,
                "target_0_rate": target_zero / max(target_zero_total, 1),
                "target_0_eligible": target_zero_total,
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--gap-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_attack_result_model(
        args.result,
        backdoorbench_root=args.backdoorbench_root,
        device=device,
    )
    checkpoint = torch.load(args.mapping, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "imdep_pq":
        raise ValueError("mapping checkpoint is not imdep_pq")
    generator = build_official_gap_generator(
        gap_root=args.gap_root,
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
    mapping.to(device).eval()

    dataset = cifar10_split(args.data_root, split=args.split, split_seed=args.split_seed)
    metrics = evaluate(
        model,
        mapping,
        loader(dataset, batch_size=args.batch_size, train=False, workers=args.workers),
        attack_goal=checkpoint["attack_goal"],
        target=int(checkpoint["target"]),
        device=device,
    )
    metrics.update(mapping_description_length_bits(args.mapping))
    metrics.update(
        {
            "split": args.split,
            "epsilon_max": float(checkpoint["epsilon_max"]),
            "learned_epsilon": float(mapping.effective_epsilon().detach().cpu()),
            "learned_epsilon_pixels": float(mapping.effective_epsilon().detach().cpu()) * 255.0,
            "epsilon_lambda": float(checkpoint["epsilon_lambda"]),
            "result": str(Path(args.result).resolve()),
            "mapping": str(Path(args.mapping).resolve()),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    output_tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output_tmp.replace(output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
