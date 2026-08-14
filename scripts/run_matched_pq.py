"""Sweep fixed-epsilon p(x)+q(x) GAP and match validation ASR levels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mdluap.codec import mapping_description_length_bits
from mdluap.data import cifar10_split, loader
from mdluap.gap import build_official_gap_generator, seed_everything
from mdluap.mappings import FixedEpsilonPQMapping
from mdluap.models import load_attack_result_model


def parse_fraction(value: str) -> float:
    """Accept a decimal or a readable fraction such as 4/255."""

    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_targets(value: str) -> list[float]:
    """Parse comma-separated ASR targets such as 0.1,0.2,...,0.9."""

    targets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not targets or any(target <= 0.0 or target >= 1.0 for target in targets):
        raise ValueError("asr targets must be strictly between 0 and 1")
    return targets


def attack_labels(model, images: torch.Tensor, attack_goal: str, target: int) -> torch.Tensor:
    """Keep the previous targeted and least-likely GAP label definitions."""

    with torch.no_grad():
        clean_logits = model(images)
        if attack_goal == "targeted":
            return torch.full(
                (images.shape[0],), int(target), dtype=torch.long, device=images.device
            )
        return clean_logits.argmin(dim=1)


def margin_attack_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Optimize a stable class margin without an epsilon term."""

    target_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(), float("-inf")
    ).amax(dim=1)
    score = target_logits - other_logits
    # Multiplying by temperature keeps the loss scale stable when T changes.
    return (float(temperature) * F.softplus((float(margin) - score) / float(temperature))).mean()


@torch.no_grad()
def evaluate_mapping(
    model,
    mapping,
    data_loader,
    *,
    attack_goal: str,
    target: int,
    device: torch.device,
) -> dict:
    """Evaluate ASR/fooling rate and actual perturbation statistics."""

    model.eval()
    mapping.eval()
    targeted_success = targeted_total = 0
    fooled = correctly_classified = 0
    target_zero = target_zero_total = 0
    linf_values: list[float] = []

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
        "mean_linf": float(linf.mean()),
        "p95_linf": float(np.quantile(linf, 0.95)),
        "max_linf": float(linf.max()),
    }
    if attack_goal == "targeted":
        metrics.update(
            {
                "asr": targeted_success / max(targeted_total, 1),
                "targeted_asr": targeted_success / max(targeted_total, 1),
                "eligible": targeted_total,
            }
        )
    else:
        metrics.update(
            {
                "asr": fooled / max(correctly_classified, 1),
                "fooling_rate": fooled / max(correctly_classified, 1),
                "eligible": correctly_classified,
                "target_0_rate": target_zero / max(target_zero_total, 1),
                "target_0_eligible": target_zero_total,
            }
        )
    return metrics


def save_mapping(path: Path, mapping: FixedEpsilonPQMapping, metadata: dict) -> None:
    """Save a compact mapping checkpoint without an optimizer state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({**metadata, "mapping": mapping.state_dict()}, temporary)
    temporary.replace(path)


def load_mapping_state(mapping: FixedEpsilonPQMapping, path: Path) -> None:
    """Load one selected checkpoint into the current mapping object."""

    checkpoint = torch.load(path, map_location=next(mapping.parameters()).device, weights_only=False)
    mapping.load_state_dict(checkpoint["mapping"], strict=True)


def evaluate_checkpoint(
    *,
    path: Path,
    mapping: FixedEpsilonPQMapping,
    model,
    data_loader,
    attack_goal: str,
    target: int,
    device: torch.device,
) -> dict:
    """Evaluate one selected checkpoint and calculate its MDL length."""

    load_mapping_state(mapping, path)
    metrics = evaluate_mapping(
        model,
        mapping,
        data_loader,
        attack_goal=attack_goal,
        target=target,
        device=device,
    )
    metrics.update(mapping_description_length_bits(str(path)))
    metrics["checkpoint"] = str(path.resolve())
    return metrics


def train_one_epsilon(
    *,
    side: str,
    model,
    result_path: str,
    gap_root: str,
    data_root: str,
    output: Path,
    attack_goal: str,
    target: int,
    epsilon: float,
    epsilon_pixels: int,
    asr_targets: list[float],
    match_tolerance: float,
    epochs: int,
    max_batches: int,
    batch_size: int,
    workers: int,
    lr: float,
    ngf: int,
    seed: int,
    split_seed: int,
    attack_margin: float,
    attack_temperature: float,
    device: torch.device,
) -> dict:
    """Train one fixed-epsilon generator and select ASR-matched epochs."""

    output.mkdir(parents=True, exist_ok=True)
    train_loader = loader(
        cifar10_split(data_root, split="train", split_seed=split_seed),
        batch_size=batch_size,
        train=True,
        workers=workers,
    )
    val_loader = loader(
        cifar10_split(data_root, split="val", split_seed=split_seed),
        batch_size=256,
        train=False,
        workers=workers,
    )
    test_loader = loader(
        cifar10_split(data_root, split="test", split_seed=split_seed),
        batch_size=256,
        train=False,
        workers=workers,
    )

    generator = build_official_gap_generator(
        gap_root=gap_root,
        device=device,
        ngf=ngf,
        output_channels=6,
    )
    mapping = FixedEpsilonPQMapping(generator, epsilon).to(device)
    optimizer = torch.optim.Adam(mapping.parameters(), lr=float(lr), betas=(0.5, 0.999))

    base_metadata = {
        "protocol": "MDL-UAP-v1",
        "mode": "imdep_pq_fixed",
        "side": side,
        "attack_goal": attack_goal,
        "target": int(target),
        "epsilon": float(epsilon),
        "epsilon_max": float(epsilon),
        "epsilon_pixels": int(epsilon_pixels),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "epochs": int(epochs),
        "max_batches": int(max_batches),
        "ngf": int(ngf),
        "attack_margin": float(attack_margin),
        "attack_temperature": float(attack_temperature),
        "result": str(Path(result_path).resolve()),
    }
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict] = {}
    best_record: dict | None = None
    best_path = checkpoints / "best_asr.pt"
    curve: list[dict] = []

    model.eval()
    for epoch in range(1, int(epochs) + 1):
        mapping.train()
        losses = []
        for batch_index, (images, _labels) in enumerate(train_loader):
            if batch_index >= int(max_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = attack_labels(model, images, attack_goal, target)
            optimizer.zero_grad(set_to_none=True)
            mapped_logits = model(mapping(images))
            loss = margin_attack_loss(
                mapped_logits,
                labels,
                margin=attack_margin,
                temperature=attack_temperature,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = evaluate_mapping(
            model,
            mapping,
            val_loader,
            attack_goal=attack_goal,
            target=target,
            device=device,
        )
        record = {
            "epoch": epoch,
            "attack_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_asr": float(validation["asr"]),
            "val_mean_linf": validation["mean_linf"],
            "val_p95_linf": validation["p95_linf"],
            "val_max_linf": validation["max_linf"],
        }
        curve.append(record)
        print(
            {
                "side": side,
                "attack_goal": attack_goal,
                "epsilon_pixels": epsilon_pixels,
                **record,
            },
            flush=True,
        )

        current_path = checkpoints / f"epoch{epoch:03d}.pt"
        if best_record is None or record["val_asr"] > best_record["val_asr"]:
            save_mapping(current_path, mapping, {**base_metadata, "epoch": epoch})
            current_path.replace(best_path)
            best_record = record.copy()

        for asr_target in asr_targets:
            distance = abs(record["val_asr"] - asr_target)
            if distance > float(match_tolerance):
                continue
            key = f"asr{int(round(asr_target * 100)):02d}"
            previous = selected.get(key)
            if previous is not None and (
                distance > previous["asr_error"]
                or (distance == previous["asr_error"] and epoch >= previous["epoch"])
            ):
                continue
            checkpoint_path = checkpoints / f"{key}.pt"
            save_mapping(checkpoint_path, mapping, {**base_metadata, "epoch": epoch})
            selected[key] = {
                "asr_target": float(asr_target),
                "epoch": epoch,
                "val_asr": record["val_asr"],
                "asr_error": distance,
                "checkpoint": str(checkpoint_path.resolve()),
            }

    curve_path = output / "validation_curve.json"
    curve_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")

    # Evaluate the best-ASR checkpoint for every epsilon.
    best_test = evaluate_checkpoint(
        path=best_path,
        mapping=mapping,
        model=model,
        data_loader=test_loader,
        attack_goal=attack_goal,
        target=target,
        device=device,
    )
    best_summary = {
        **base_metadata,
        "best_epoch": int(best_record["epoch"]),
        "best_val_asr": float(best_record["val_asr"]),
        "best_test_asr": float(best_test["asr"]),
        "best_test": best_test,
    }
    (output / "best_asr.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    matched_results = []
    for key, choice in sorted(selected.items()):
        test_metrics = evaluate_checkpoint(
            path=Path(choice["checkpoint"]),
            mapping=mapping,
            model=model,
            data_loader=test_loader,
            attack_goal=attack_goal,
            target=target,
            device=device,
        )
        matched_results.append(
            {
                **base_metadata,
                "asr_target": choice["asr_target"],
                "selected_epoch": choice["epoch"],
                "val_asr": choice["val_asr"],
                "test_asr": test_metrics["asr"],
                "asr_error": choice["asr_error"],
                "mean_linf": test_metrics["mean_linf"],
                "p95_linf": test_metrics["p95_linf"],
                "max_linf": test_metrics["max_linf"],
                "bits": test_metrics["bits"],
                "bytes": test_metrics["bytes"],
                "target_0_rate": test_metrics.get("target_0_rate"),
                "checkpoint": choice["checkpoint"],
                "status": "matched",
            }
        )

    target_records = list(matched_results)
    matched_keys = {item["asr_target"] for item in matched_results}
    for asr_target in asr_targets:
        if asr_target in matched_keys:
            continue
        target_records.append(
            {
                **base_metadata,
                "asr_target": float(asr_target),
                "selected_epoch": None,
                "val_asr": None,
                "test_asr": None,
                "asr_error": None,
                "mean_linf": None,
                "p95_linf": None,
                "max_linf": None,
                "bits": None,
                "bytes": None,
                "target_0_rate": None,
                "checkpoint": None,
                "status": "unmatched",
            }
        )

    epsilon_summary = {
        **base_metadata,
        "best_val_asr": best_summary["best_val_asr"],
        "best_test_asr": best_summary["best_test_asr"],
        "best_epoch": best_summary["best_epoch"],
        "matched_count": len(matched_results),
        "matched": matched_results,
        "target_records": sorted(target_records, key=lambda item: item["asr_target"]),
        "validation_curve": str(curve_path.resolve()),
        "best_checkpoint": str(best_path.resolve()),
    }
    (output / "epsilon_summary.json").write_text(
        json.dumps(epsilon_summary, indent=2), encoding="utf-8"
    )
    return epsilon_summary


def build_pairwise(results: list[dict], asr_targets: list[float], epsilon_pixels: list[int]) -> list[dict]:
    """Create clean/backdoor rows for every epsilon and ASR target."""

    indexed = {
        (item["attack_goal"], item["epsilon_pixels"], item["asr_target"], item["side"]): item
        for item in results
    }
    pairs = []
    for goal in ("targeted", "non_targeted"):
        for epsilon in epsilon_pixels:
            for target in asr_targets:
                clean = indexed.get((goal, epsilon, target, "clean"))
                backdoor = indexed.get((goal, epsilon, target, "backdoor"))
                pair = {
                    "attack_goal": goal,
                    "epsilon_pixels": epsilon,
                    "asr_target": target,
                    "status": "matched_pair" if clean and backdoor else "incomplete_pair",
                }
                for side, item in (("clean", clean), ("backdoor", backdoor)):
                    pair[f"{side}_val_asr"] = item["val_asr"] if item else None
                    pair[f"{side}_test_asr"] = item["test_asr"] if item else None
                    pair[f"{side}_bits"] = item["bits"] if item else None
                    pair[f"{side}_mean_linf"] = item["mean_linf"] if item else None
                    pair[f"{side}_p95_linf"] = item["p95_linf"] if item else None
                if clean and backdoor:
                    pair.update(
                        {
                            "backdoor_minus_clean_bits": backdoor["bits"] - clean["bits"],
                            "clean_bits_over_backdoor_bits": clean["bits"] / max(backdoor["bits"], 1),
                            "backdoor_minus_clean_mean_linf": backdoor["mean_linf"] - clean["mean_linf"],
                            "backdoor_minus_clean_test_asr": backdoor["test_asr"] - clean["test_asr"],
                        }
                    )
                else:
                    pair.update(
                        {
                            "backdoor_minus_clean_bits": None,
                            "clean_bits_over_backdoor_bits": None,
                            "backdoor_minus_clean_mean_linf": None,
                            "backdoor_minus_clean_test_asr": None,
                        }
                    )
                pairs.append(pair)
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--gap-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--clean-result", required=True)
    parser.add_argument("--backdoor-result", required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epsilon-start", type=int, default=4)
    parser.add_argument("--epsilon-end", type=int, default=16)
    parser.add_argument("--asr-targets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--match-tolerance", type=float, default=0.02)
    parser.add_argument("--stop-asr", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--attack-margin", type=float, default=1.0)
    parser.add_argument("--attack-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asr_targets = parse_targets(args.asr_targets)
    epsilon_pixels = list(range(int(args.epsilon_start), int(args.epsilon_end) + 1))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    results: list[dict] = []
    epsilon_results: list[dict] = []
    for side, result_path in (("clean", args.clean_result), ("backdoor", args.backdoor_result)):
        for attack_goal in ("targeted", "non_targeted"):
            model, _ = load_attack_result_model(
                result_path,
                backdoorbench_root=args.backdoorbench_root,
                device=device,
            )
            reached_stop = False
            for pixels in epsilon_pixels:
                if reached_stop:
                    break
                # Reset the same generator seed for every fixed-epsilon condition.
                seed_everything(args.seed)
                condition = output_root / side / attack_goal / f"eps{pixels}" / f"seed{args.seed}"
                summary = train_one_epsilon(
                    side=side,
                    model=model,
                    result_path=result_path,
                    gap_root=args.gap_root,
                    data_root=args.data_root,
                    output=condition,
                    attack_goal=attack_goal,
                    target=args.target,
                    epsilon=float(pixels) / 255.0,
                    epsilon_pixels=pixels,
                    asr_targets=asr_targets,
                    match_tolerance=args.match_tolerance,
                    epochs=args.epochs,
                    max_batches=args.max_batches,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    lr=args.lr,
                    ngf=args.ngf,
                    seed=args.seed,
                    split_seed=args.split_seed,
                    attack_margin=args.attack_margin,
                    attack_temperature=args.attack_temperature,
                    device=device,
                )
                for item in summary["matched"]:
                    results.append({"side": side, **item})
                epsilon_results.append({"side": side, **summary})
                reached_stop = summary["best_val_asr"] >= float(args.stop_asr)

    pairwise = build_pairwise(results, asr_targets, epsilon_pixels)
    report = {
        "protocol": "MDL-UAP-v1",
        "asr_targets": asr_targets,
        "match_tolerance": args.match_tolerance,
        "stop_asr": args.stop_asr,
        "epsilon_pixels": epsilon_pixels,
        "settings": vars(args),
        "results": results,
        "epsilon_results": epsilon_results,
        "pairwise": pairwise,
        "note": "Single seed results are a quick trend check, not a final statistical conclusion.",
    }
    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_csv = Path(args.report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in pairwise for key in row})
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairwise)
    print(f"Run complete: report={report_json.resolve()}")
    print(f"CSV complete: report={report_csv.resolve()}")


if __name__ == "__main__":
    main()
