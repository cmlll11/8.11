"""Check official BackdoorBench model quality for several trigger families."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mdluap.models import load_attack_result_model


def parse_trigger(value: str) -> tuple[str, str]:
    """Parse a CLI entry in the form ``name=/path/to/attack_result.pt``."""

    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("trigger must be NAME=PATH")
    return name, path


def load_official_datasets(result_path: str, root: str):
    """Load the official clean and triggered test datasets."""

    resolved_root = str(Path(root).resolve())
    if resolved_root not in sys.path:
        sys.path.insert(0, resolved_root)
    from utils.save_load_attack import load_attack_result

    os.chdir(resolved_root)
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        return load_attack_result(result_path)
    finally:
        torch.load = original_torch_load


@torch.inference_mode()
def accuracy(model, dataset, *, device: torch.device, workers: int, batch_size: int) -> float:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    correct = total = 0
    for batch in data_loader:
        # BackdoorBench samples may carry metadata after image and label.
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-result", required=True)
    parser.add_argument("--trigger-result", action="append", type=parse_trigger, required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    clean_wrapper, _ = load_attack_result_model(
        args.clean_result,
        backdoorbench_root=args.backdoorbench_root,
        device=device,
    )
    # Official BackdoorBench datasets already apply CIFAR-10 normalization.
    # Use the underlying classifier here, not the raw-image adapter.
    clean_model = clean_wrapper.model
    rows = {}
    for trigger, result_path in args.trigger_result:
        result_file = Path(result_path)
        if not result_file.is_file():
            rows[trigger] = {
                "result": str(result_file.resolve()),
                "status": "missing_result",
                "passed": False,
            }
            continue

        datasets = load_official_datasets(str(result_file), args.backdoorbench_root)
        backdoor_wrapper, metadata = load_attack_result_model(
            str(result_file),
            backdoorbench_root=args.backdoorbench_root,
            device=device,
        )
        backdoor_model = backdoor_wrapper.model
        clean_accuracy = accuracy(
            clean_model, datasets["clean_test"], device=device,
            workers=args.workers, batch_size=args.batch_size,
        )
        clean_trigger_asr = accuracy(
            clean_model, datasets["bd_test"], device=device,
            workers=args.workers, batch_size=args.batch_size,
        )
        backdoor_clean_accuracy = accuracy(
            backdoor_model, datasets["clean_test"], device=device,
            workers=args.workers, batch_size=args.batch_size,
        )
        backdoor_asr = accuracy(
            backdoor_model, datasets["bd_test"], device=device,
            workers=args.workers, batch_size=args.batch_size,
        )
        passed = bool(
            clean_accuracy >= 0.90
            and clean_trigger_asr <= 0.10
            and backdoor_clean_accuracy >= 0.90
            and backdoor_asr >= 0.90
        )
        rows[trigger] = {
            "result": str(result_file.resolve()),
            "model_name": metadata["model_name"],
            "clean_accuracy": clean_accuracy,
            "clean_trigger_asr": clean_trigger_asr,
            "backdoor_clean_accuracy": backdoor_clean_accuracy,
            "backdoor_asr": backdoor_asr,
            "status": "qualified" if passed else "gate_failed",
            "passed": passed,
        }
        print(
            f"{trigger}: clean_acc={clean_accuracy:.4f} "
            f"backdoor_acc={backdoor_clean_accuracy:.4f} "
            f"backdoor_asr={backdoor_asr:.4f} "
            f"status={rows[trigger]['status']}",
            flush=True,
        )
        del backdoor_model, backdoor_wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "protocol": "BackdoorBench-trigger-gates-v1",
        "clean_result": str(Path(args.clean_result).resolve()),
        "gates": {
            "clean_accuracy_min": 0.90,
            "backdoor_clean_accuracy_min": 0.90,
            "backdoor_asr_min": 0.90,
            "clean_trigger_asr_max": 0.10,
        },
        "triggers": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model gate report: {output.resolve()}")


if __name__ == "__main__":
    main()
