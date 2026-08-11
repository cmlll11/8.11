"""Check the clean/backdoor classifier gates before fitting targeted mappings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mdluap.models import load_attack_result_model


@torch.no_grad()
def accuracy(model, data_loader, *, device: torch.device) -> float:
    """Evaluate a model on a BackdoorBench dataset that is already normalized."""

    correct = total = 0
    model.eval()
    for batch in data_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


def load_datasets(result_path: str, backdoorbench_root: str):
    """Use BackdoorBench's official reconstruction code for clean and trigger tests."""

    root = str(Path(backdoorbench_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.save_load_attack import load_attack_result

    # BackdoorBench stores generated image paths relative to its own root.
    previous_cwd = os.getcwd()
    try:
        os.chdir(root)
        return load_attack_result(result_path)
    finally:
        os.chdir(previous_cwd)


def evaluate_one(result_path: str, datasets: dict, *, root: str, device: torch.device, workers: int) -> dict:
    """Report clean accuracy and official-trigger targeted ASR for one artifact."""

    wrapped, _ = load_attack_result_model(result_path, backdoorbench_root=root, device=device)
    clean_loader = DataLoader(datasets["clean_test"], batch_size=256, shuffle=False, num_workers=workers)
    trigger_loader = DataLoader(datasets["bd_test"], batch_size=256, shuffle=False, num_workers=workers)
    return {
        "clean_accuracy": accuracy(wrapped.model, clean_loader, device=device),
        "official_trigger_asr": accuracy(wrapped.model, trigger_loader, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-result", required=True)
    parser.add_argument("--backdoor-result", required=True)
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    # One official BadNets result supplies the identical clean and trigger test sets.
    datasets = load_datasets(args.backdoor_result, args.backdoorbench_root)
    report = {
        "clean": evaluate_one(
            args.clean_result,
            datasets,
            root=args.backdoorbench_root,
            device=device,
            workers=args.workers,
        ),
        "backdoor": evaluate_one(
            args.backdoor_result,
            datasets,
            root=args.backdoorbench_root,
            device=device,
            workers=args.workers,
        ),
        "gates": {"clean_accuracy_min": 0.90, "backdoor_asr_min": 0.90, "clean_trigger_asr_max": 0.10},
    }
    report["passed"] = bool(
        report["clean"]["clean_accuracy"] >= 0.90
        and report["backdoor"]["clean_accuracy"] >= 0.90
        and report["backdoor"]["official_trigger_asr"] >= 0.90
        and report["clean"]["official_trigger_asr"] <= 0.10
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("model-pair qualification gates failed")


if __name__ == "__main__":
    main()
