"""Evaluate one targeted mapping and compute its independent MDL description."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mdluap.codec import mapping_description_length_bits
from mdluap.data import cifar10_split, loader
from mdluap.gap import build_official_gap_generator
from mdluap.mappings import TargetedImageDependentMapping, TargetedUniversalMapping
from mdluap.models import load_attack_result_model


@torch.no_grad()
def evaluate(model, mapping, data_loader, *, target: int, device: torch.device) -> dict:
    """Measure targeted ASR on non-target images and the raw-pixel bound."""

    model.eval()
    mapping.eval()
    success = total = 0
    max_linf = 0.0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mapped = mapping(images)
        max_linf = max(max_linf, float((mapped - images).abs().max().cpu()))
        keep = labels != int(target)
        if bool(keep.any()):
            predictions = model(mapped[keep]).argmax(dim=1)
            success += int((predictions == int(target)).sum())
            total += int(keep.sum())
    return {"targeted_asr": success / max(total, 1), "max_linf": max_linf, "eligible": total}


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_attack_result_model(
        args.result,
        backdoorbench_root=args.backdoorbench_root,
        device=device,
    )
    checkpoint = torch.load(args.mapping, map_location=device, weights_only=False)
    if checkpoint["mode"] == "universal":
        mapping = TargetedUniversalMapping(checkpoint["universal_delta"], checkpoint["epsilon"])
    else:
        generator = build_official_gap_generator(
            gap_root=args.gap_root,
            device=device,
            ngf=int(checkpoint.get("ngf", 64)),
        )
        mapping = TargetedImageDependentMapping(generator, checkpoint["epsilon"])
        mapping.load_state_dict(checkpoint["mapping"], strict=True)
    mapping.to(device).eval()

    dataset = cifar10_split(args.data_root, split=args.split, split_seed=args.split_seed)
    metrics = evaluate(
        model,
        mapping,
        loader(dataset, batch_size=args.batch_size, train=False, workers=args.workers),
        target=checkpoint["target"],
        device=device,
    )
    metrics.update(mapping_description_length_bits(args.mapping))
    metrics.update(
        {
            "split": args.split,
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
