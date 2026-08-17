"""Train fixed-epsilon x+f(x) GAP mappings and test quantized encodings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mdluap.codec import SUPPORTED_QUANTIZATION_BITS, encode_mapping
from mdluap.data import cifar10_split, loader
from mdluap.gap import build_official_gap_generator, seed_everything
from mdluap.mappings import ImageDependentResidualMapping
from mdluap.models import load_attack_result_model


PRECISIONS = (32, 16, 8, 4)


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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def attack_labels(model, images: torch.Tensor, attack_goal: str, target: int) -> torch.Tensor:
    """Use targeted or classic least-likely GAP training labels."""

    with torch.no_grad():
        clean_logits = model(images)
        if attack_goal == "targeted":
            return torch.full(
                (images.shape[0],), int(target), dtype=torch.long, device=images.device
            )
        return clean_logits.argmin(dim=1)


@torch.no_grad()
def evaluate_mapping(model, mapping, data_loader, *, attack_goal: str, target: int) -> dict:
    """Evaluate targeted ASR or least-likely fooling rate."""

    model.eval()
    mapping.eval()
    targeted_success = targeted_total = 0
    fooled = correctly_classified = 0
    target_zero = target_zero_total = 0
    for images, labels in data_loader:
        images = images.to(next(model.parameters()).device, non_blocking=True)
        labels = labels.to(images.device, non_blocking=True)
        clean_predictions = model(images).argmax(dim=1)
        mapped_predictions = model(mapping(images)).argmax(dim=1)
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
    if attack_goal == "targeted":
        return {
            "asr": targeted_success / max(targeted_total, 1),
            "targeted_asr": targeted_success / max(targeted_total, 1),
            "target_0_rate": None,
            "eligible": targeted_total,
        }
    return {
        "asr": fooled / max(correctly_classified, 1),
        "fooling_rate": fooled / max(correctly_classified, 1),
        "target_0_rate": target_zero / max(target_zero_total, 1),
        "eligible": correctly_classified,
    }


def build_mapping(checkpoint: dict, *, gap_root: str, device: torch.device) -> ImageDependentResidualMapping:
    """Rebuild the x+f(x) mapping from checkpoint metadata."""

    generator = build_official_gap_generator(
        gap_root=gap_root,
        device=device,
        ngf=int(checkpoint["ngf"]),
        output_channels=3,
    )
    mapping = ImageDependentResidualMapping(generator, float(checkpoint["epsilon"])).to(device)
    mapping.load_state_dict(checkpoint["mapping"], strict=True)
    return mapping.eval()


def save_checkpoint(path: Path, mapping: ImageDependentResidualMapping, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({**metadata, "mapping": mapping.state_dict()}, temporary)
    temporary.replace(path)


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
    epsilon_pixels: int,
    epochs: int,
    max_batches: int,
    batch_size: int,
    workers: int,
    lr: float,
    ngf: int,
    seed: int,
    split_seed: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Train one fixed-epsilon residual generator and test all precisions."""

    output.mkdir(parents=True, exist_ok=True)
    epsilon = float(epsilon_pixels) / 255.0
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
        output_channels=3,
    )
    mapping = ImageDependentResidualMapping(generator, epsilon).to(device)
    optimizer = torch.optim.Adam(mapping.parameters(), lr=float(lr), betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()
    base_metadata = {
        "protocol": "MDL-UAP-residual-compression-v1",
        "mode": "imdep_residual",
        "mapping_form": "x+f(x)",
        "side": side,
        "attack_goal": attack_goal,
        "target": int(target),
        "epsilon": epsilon,
        "epsilon_pixels": int(epsilon_pixels),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "epochs": int(epochs),
        "max_batches": int(max_batches),
        "ngf": int(ngf),
        "attack_loss": "official_gap_log_cross_entropy",
        "result": str(Path(result_path).resolve()),
    }

    best_checkpoint = output / "float32_checkpoint.pt"
    best_record: dict | None = None
    curve = []
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
            loss = torch.log(criterion(model(mapping(images)), labels).clamp_min(1e-12))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_mapping(model, mapping, val_loader, attack_goal=attack_goal, target=target)
        record = {
            "epoch": epoch,
            "attack_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_asr": float(validation["asr"]),
            "target_0_rate": validation["target_0_rate"],
        }
        curve.append(record)
        if best_record is None or record["val_asr"] > best_record["val_asr"]:
            save_checkpoint(best_checkpoint, mapping, {**base_metadata, "epoch": epoch})
            best_record = record.copy()

    (output / "validation_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    if best_record is None:
        raise RuntimeError(f"no training batches completed for {output}")

    base_checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    rows = []
    for bits in PRECISIONS:
        if bits not in SUPPORTED_QUANTIZATION_BITS:
            raise RuntimeError(f"codec does not support requested precision {bits}")
        decoded_state, code, encoded = encode_mapping(str(best_checkpoint), bits)
        decoded_mapping = build_mapping(base_checkpoint, gap_root=gap_root, device=device)
        decoded_mapping.load_state_dict(decoded_state, strict=True)
        validation = evaluate_mapping(
            model, decoded_mapping, val_loader, attack_goal=attack_goal, target=target
        )
        test = evaluate_mapping(
            model, decoded_mapping, test_loader, attack_goal=attack_goal, target=target
        )
        compression_dir = output / "compression" / f"precision{bits}"
        compression_dir.mkdir(parents=True, exist_ok=True)
        (compression_dir / "mapping.bin").write_bytes(encoded)
        torch.save(decoded_state, compression_dir / "decoded_state.pt")
        record = {
            "side": side,
            "attack_goal": attack_goal,
            "epsilon_pixels": int(epsilon_pixels),
            "selected_epoch": int(best_record["epoch"]),
            "quantization_bits": int(bits),
            "bits": int(code["bits"]),
            "bytes": int(code["bytes"]),
            "val_asr": float(validation["asr"]),
            "test_asr": float(test["asr"]),
            "target_0_rate": validation["target_0_rate"],
            "test_target_0_rate": test["target_0_rate"],
            "status": "baseline" if bits == 32 else "compressed",
            "mapping": str(best_checkpoint.resolve()),
            "encoded_mapping": str((compression_dir / "mapping.bin").resolve()),
        }
        rows.append(record)
        (compression_dir / "metadata.json").write_text(
            json.dumps({**code, **record}, indent=2), encoding="utf-8"
        )

    baseline_asr = next(item["val_asr"] for item in rows if item["quantization_bits"] == 32)
    for item in rows:
        item["val_asr_drop_from_float32"] = float(baseline_asr - item["val_asr"])
        if item["quantization_bits"] != 32 and item["val_asr"] >= 0.90:
            item["status"] = "retains_90"
        metadata_path = (
            output / "compression" / f"precision{item['quantization_bits']}" / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_path.write_text(json.dumps({**metadata, **item}, indent=2), encoding="utf-8")

    valid_90 = [item for item in rows if item["val_asr"] >= 0.90]
    minimum = min(valid_90, key=lambda item: (item["bits"], item["quantization_bits"])) if valid_90 else None
    summary = {
        **base_metadata,
        "selected_epoch": int(best_record["epoch"]),
        "float32_val_asr": float(baseline_asr),
        "minimum_precision_retaining_90": int(minimum["quantization_bits"]) if minimum else None,
        "minimum_bits_retaining_90": int(minimum["bits"]) if minimum else None,
        "results": rows,
        "validation_curve": str((output / "validation_curve.json").resolve()),
        "float32_checkpoint": str(best_checkpoint.resolve()),
    }
    (output / "epsilon_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del decoded_mapping, mapping, generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary


def pairwise_rows(rows: list[dict], epsilon_pixels: list[int]) -> list[dict]:
    indexed = {
        (item["side"], item["attack_goal"], item["epsilon_pixels"], item["quantization_bits"]): item
        for item in rows
    }
    pairs = []
    for goal in ("targeted", "non_targeted"):
        for epsilon in epsilon_pixels:
            for bits in PRECISIONS:
                clean = indexed[("clean", goal, epsilon, bits)]
                backdoor = indexed[("backdoor", goal, epsilon, bits)]
                pairs.append(
                    {
                        "attack_goal": goal,
                        "epsilon_pixels": epsilon,
                        "quantization_bits": bits,
                        "clean_val_asr": clean["val_asr"],
                        "backdoor_val_asr": backdoor["val_asr"],
                        "clean_test_asr": clean["test_asr"],
                        "backdoor_test_asr": backdoor["test_asr"],
                        "clean_bits": clean["bits"],
                        "backdoor_bits": backdoor["bits"],
                        "backdoor_minus_clean_val_asr": backdoor["val_asr"] - clean["val_asr"],
                        "backdoor_minus_clean_test_asr": backdoor["test_asr"] - clean["test_asr"],
                        "backdoor_minus_clean_bits": backdoor["bits"] - clean["bits"],
                        "both_retain_90": clean["val_asr"] >= 0.90 and backdoor["val_asr"] >= 0.90,
                    }
                )
    return pairs


def main() -> None:
    args = parse_args()
    if tuple(PRECISIONS) != tuple(sorted(SUPPORTED_QUANTIZATION_BITS, reverse=True)):
        raise RuntimeError("codec precision levels do not match the experiment protocol")
    if int(args.epsilon_start) > int(args.epsilon_end):
        raise ValueError("epsilon-start must not exceed epsilon-end")
    device = torch.device(args.device)
    epsilon_pixels = list(range(int(args.epsilon_start), int(args.epsilon_end) + 1))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for side, result_path in (("clean", args.clean_result), ("backdoor", args.backdoor_result)):
        model, _ = load_attack_result_model(
            result_path, backdoorbench_root=args.backdoorbench_root, device=device
        )
        for attack_goal in ("targeted", "non_targeted"):
            for epsilon_pixels_value in epsilon_pixels:
                seed_everything(args.seed)
                condition = (
                    output_root / side / attack_goal / f"eps{epsilon_pixels_value}" / f"seed{args.seed}"
                )
                rows, summary = train_one_epsilon(
                    side=side,
                    model=model,
                    result_path=result_path,
                    gap_root=args.gap_root,
                    data_root=args.data_root,
                    output=condition,
                    attack_goal=attack_goal,
                    target=args.target,
                    epsilon_pixels=epsilon_pixels_value,
                    epochs=args.epochs,
                    max_batches=args.max_batches,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    lr=args.lr,
                    ngf=args.ngf,
                    seed=args.seed,
                    split_seed=args.split_seed,
                    device=device,
                )
                all_rows.extend(rows)
                summaries.append(summary)
                print(
                    f"Condition complete: side={side} attack_goal={attack_goal} "
                    f"epsilon={epsilon_pixels_value}/255 selected_epoch={summary['selected_epoch']}",
                    flush=True,
                )

    pairs = pairwise_rows(all_rows, epsilon_pixels)
    report = {
        "protocol": "MDL-UAP-residual-compression-v1",
        "mapping_form": "x+f(x)",
        "epsilon_pixels": epsilon_pixels,
        "quantization_bits": list(PRECISIONS),
        "results": all_rows,
        "epsilon_summaries": summaries,
        "pairwise": pairs,
        "note": (
            "Quantized generators are discrete compression candidates. The full ASR-bit "
            "curves are primary; 90% retention is an auxiliary threshold."
        ),
    }
    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_csv = Path(args.report_csv)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(
        f"Run complete: conditions={len(summaries)} encodings={len(all_rows)} "
        f"report={report_json.resolve()} CSV={report_csv.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
