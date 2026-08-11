"""Train one targeted GAP mapping on one official BackdoorBench model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mdluap.data import cifar10_split, loader
from mdluap.gap import UniversalGAPMapping, build_official_gap_generator, seed_everything, train_targeted_gap
from mdluap.models import load_attack_result_model


def parse_epsilon(value: str) -> float:
    """Accept either a decimal value or the readable form 8/255."""

    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="BackdoorBench attack_result.pt")
    parser.add_argument("--backdoorbench-root", required=True)
    parser.add_argument("--gap-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="directory for mapping checkpoint and metadata")
    parser.add_argument("--mode", choices=("universal", "imdep"), required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--epsilon", type=parse_epsilon, required=True, help="raw-pixel L_inf budget, e.g. 8/255")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model, result_meta = load_attack_result_model(
        args.result,
        backdoorbench_root=args.backdoorbench_root,
        device=device,
    )
    train_set = cifar10_split(args.data_root, split="train", split_seed=args.split_seed)
    train_loader = loader(train_set, batch_size=args.batch_size, train=True, workers=args.workers)

    generator = build_official_gap_generator(gap_root=args.gap_root, device=device, ngf=args.ngf)
    if args.mode == "universal":
        noise = torch.rand(1, 3, 32, 32, device=device) * 255.0
        mapping = UniversalGAPMapping(generator, noise, args.epsilon).to(device)
    else:
        noise = None
        from mdluap.mappings import TargetedImageDependentMapping

        mapping = TargetedImageDependentMapping(generator, args.epsilon).to(device)

    checkpoint = output / "checkpoint.pt"
    train_targeted_gap(
        model=model,
        mapping=mapping,
        train_loader=train_loader,
        target_label=args.target,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        checkpoint_path=str(checkpoint),
        max_batches=args.max_batches,
    )
    mapping.eval()
    universal_delta = None
    if args.mode == "universal":
        # The generator is a search tool; the final mapping is only this fixed delta.
        with torch.no_grad():
            universal_delta = mapping.generator(mapping.noise).clamp(-1.0, 1.0) * args.epsilon
    final_path = output / "mapping.pt"
    final_tmp = output / "mapping.pt.tmp"
    torch.save(
        {
            "protocol": "MDL-UAP-v1",
            "mode": args.mode,
            "target": args.target,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "ngf": args.ngf,
            "mapping": mapping.state_dict(),
            "noise": noise.detach().cpu() if noise is not None else None,
            "universal_delta": universal_delta.detach().cpu() if universal_delta is not None else None,
            "result": str(Path(args.result).resolve()),
            "model_name": result_meta["model_name"],
        },
        final_tmp,
    )
    final_tmp.replace(final_path)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Run complete: output={output.resolve()}")


if __name__ == "__main__":
    main()
