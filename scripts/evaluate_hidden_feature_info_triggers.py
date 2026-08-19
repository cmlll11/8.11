"""Compare x+f GAP hidden-feature entropy across trigger families."""

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
from mdluap.mappings import ImageDependentResidualMapping
from mdluap.models import load_attack_result_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", required=True)
    parser.add_argument("--model-report", required=True)
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


def mapping_path(root: Path, trigger: str, goal: str) -> Path:
    return root / trigger / goal / "seed2026" / "mapping.pt"


def build_mapping(checkpoint_path: Path, *, gap_root: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != "imdep_residual":
        raise ValueError(f"expected imdep_residual checkpoint: {checkpoint_path}")
    generator = build_official_gap_generator(
        gap_root=gap_root,
        device=device,
        ngf=int(checkpoint.get("ngf", 64)),
        output_channels=3,
    )
    mapping = ImageDependentResidualMapping(generator, float(checkpoint["epsilon"])).to(device)
    mapping.load_state_dict(checkpoint["mapping"], strict=True)
    return mapping.eval(), checkpoint


def find_bottleneck(generator: nn.Module) -> tuple[str, nn.Module]:
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
    trigger: str,
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
        data_loader = loader(dataset, batch_size=args.batch_size, train=False, workers=args.workers)
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_predictions = model(images).argmax(dim=1)
            mapped_predictions = model(mapping(images)).argmax(dim=1)
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
        eligible = targeted_total
    else:
        probe_asr = fooled / max(correctly_classified, 1)
        target_0_rate = target_zero / max(target_zero_total, 1)
        eligible = correctly_classified

    result = {
        "trigger": trigger,
        "attack_goal": str(checkpoint["attack_goal"]),
        "feature_layer": feature_name,
        "feature_shape": list(features.shape[1:]),
        "features": features,
        "probe_asr": float(probe_asr),
        "target_0_rate": None if target_0_rate is None else float(target_0_rate),
        "eligible": int(eligible),
        "mapping": str(mapping_path_value.resolve()),
        "result": str(Path(result_path).resolve()),
    }
    del model, mapping
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def shared_channel_scales(results: list[dict]) -> torch.Tensor:
    maxima = [result["features"].abs().amax(dim=(0, 2, 3)) for result in results]
    return torch.stack(maxima, dim=0).amax(dim=0).div(127.0).clamp_min(1e-12)


def quantized_entropy(features: torch.Tensor, scale: torch.Tensor) -> float:
    """Return mean marginal Shannon entropy in bits per activation."""

    quantized = torch.round(features / scale.view(1, -1, 1, 1)).clamp(-127, 127).to(torch.int64)
    entropies = []
    for channel in range(quantized.shape[1]):
        values = quantized[:, channel].reshape(-1) + 127
        counts = torch.bincount(values, minlength=255).double()
        probabilities = counts / counts.sum().clamp_min(1.0)
        probabilities = probabilities[probabilities > 0]
        entropies.append(float((-probabilities * probabilities.log2()).sum()))
    return float(np.mean(entropies))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quantization_bits != 8:
        raise ValueError("the hidden-feature probe uses a shared 8-bit quantizer")
    if args.n_samples <= 0:
        raise ValueError("n-samples must be positive")

    model_report = json.loads(Path(args.model_report).read_text(encoding="utf-8"))
    trigger_rows = model_report["triggers"]
    qualified = [name for name, row in trigger_rows.items() if row.get("passed")]
    if not qualified:
        raise RuntimeError("no BackdoorBench trigger passed the model gates")

    device = torch.device(args.device)
    dataset = cifar10_split(args.data_root, split="test", split_seed=args.split_seed)
    if args.n_samples > len(dataset):
        raise ValueError(f"n-samples={args.n_samples} exceeds test set size {len(dataset)}")
    dataset = Subset(dataset, list(range(args.n_samples)))
    mapping_root = Path(args.mapping_root)

    conditions: list[tuple[str, str]] = [("clean", model_report["clean_result"])]
    conditions.extend((name, row["result"]) for name, row in trigger_rows.items() if row.get("passed"))
    results: list[dict] = []
    for goal in ("targeted", "non_targeted"):
        for trigger, result_path in conditions:
            checkpoint_path = mapping_path(mapping_root, trigger, goal)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"missing mapping checkpoint: {checkpoint_path}")
            results.append(
                collect_condition(
                    trigger=trigger,
                    mapping_path_value=checkpoint_path,
                    result_path=result_path,
                    args=args,
                    dataset=dataset,
                    device=device,
                )
            )

    scale = shared_channel_scales(results)
    output_root = Path(args.output_root)
    rows = []
    for result in results:
        row = {
            "side": "clean" if result["trigger"] == "clean" else "backdoor",
            "trigger": result["trigger"],
            "attack_goal": result["attack_goal"],
            "feature_layer": result["feature_layer"],
            "feature_shape": result["feature_shape"],
            "n_samples": args.n_samples,
            "quantization_bits": args.quantization_bits,
            "quantizer": "shared-symmetric-per-channel",
            "feature_entropy_bits_per_activation": quantized_entropy(result["features"], scale),
            "probe_asr": result["probe_asr"],
            "target_0_rate": result["target_0_rate"],
            "eligible": result["eligible"],
            "status": "qualified",
        }
        rows.append(row)
        write_json(
            output_root / result["trigger"] / result["attack_goal"] / "seed2026" / "hidden_info.json",
            row,
        )

    comparisons = []
    for goal in ("targeted", "non_targeted"):
        clean = next(row for row in rows if row["trigger"] == "clean" and row["attack_goal"] == goal)
        for trigger in qualified:
            backdoor = next(
                row for row in rows if row["trigger"] == trigger and row["attack_goal"] == goal
            )
            comparison = {
                "trigger": trigger,
                "attack_goal": goal,
                "backdoor_minus_clean_entropy_bits_per_activation": (
                    backdoor["feature_entropy_bits_per_activation"]
                    - clean["feature_entropy_bits_per_activation"]
                ),
                "backdoor_minus_clean_probe_asr": backdoor["probe_asr"] - clean["probe_asr"],
            }
            comparisons.append(comparison)
            print(
                f"{goal} {trigger}: entropy_delta="
                f"{comparison['backdoor_minus_clean_entropy_bits_per_activation']:+.6f} "
                f"probe_asr_delta={comparison['backdoor_minus_clean_probe_asr']:+.6f}",
                flush=True,
            )

    summary = {
        "protocol": "MDL-UAP-hidden-feature-residual-multitrigger-v1",
        "mapping_form": "x+f(x)",
        "split": "test",
        "n_samples": args.n_samples,
        "quantization_bits": args.quantization_bits,
        "quantizer": "shared-symmetric-per-channel",
        "feature_layer": rows[0]["feature_layer"],
        "feature_shape": rows[0]["feature_shape"],
        "model_gates": model_report,
        "qualified_triggers": qualified,
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
    print(f"Summary complete: {Path(args.summary).resolve()}")
    print(f"CSV complete: {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()

