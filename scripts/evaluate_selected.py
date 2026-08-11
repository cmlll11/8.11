"""Evaluate validation-selected mappings once on the untouched test split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    output_dir = repo / "reports" / "badnet_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for item in summary["selected"]:
        if "mapping" not in item:
            completed.append(item)
            continue
        side = item["side"]
        result_name = "clean_seed0_attack_result.pt" if side == "clean" else "badnet_seed0_attack_result.pt"
        output = output_dir / f"{side}_{item['mode']}.json"
        command = [
            sys.executable,
            str(repo / "scripts" / "evaluate_mapping.py"),
            "--result",
            str(repo / "artifacts" / "models" / result_name),
            "--mapping",
            item["mapping"],
            "--backdoorbench-root",
            str(repo / "third_party" / "BackdoorBench"),
            "--gap-root",
            str(repo / "third_party" / "GAP"),
            "--data-root",
            str(repo / "data"),
            "--output",
            str(output),
            "--split",
            "test",
            "--split-seed",
            "2026",
            "--device",
            args.device,
        ]
        subprocess.run(command, check=True)
        completed.append(json.loads(output.read_text(encoding="utf-8")) | {"side": side, "mode": item["mode"]})

    ratios = {}
    for mode in ("universal", "imdep"):
        clean = next((item for item in completed if item.get("side") == "clean" and item.get("mode") == mode), None)
        backdoor = next((item for item in completed if item.get("side") == "backdoor" and item.get("mode") == mode), None)
        if clean and backdoor and "bits" in clean and "bits" in backdoor:
            ratios[mode] = clean["bits"] / max(backdoor["bits"], 1)
    final_path = output_dir / "summary.json"
    final_path.write_text(
        json.dumps(
            {"protocol": "MDL-UAP-v1", "results": completed, "b_clean_over_b_backdoor": ratios},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Run complete: output={final_path}")


if __name__ == "__main__":
    main()
