"""Write a concise clean/backdoor comparison for the four p+q runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-threshold", type=float, default=0.90)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results = []
    for side in ("clean", "backdoor"):
        for attack_goal in ("targeted", "non_targeted"):
            report_path = root / side / attack_goal / "seed2026" / "test.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            success = report["targeted_asr"] if attack_goal == "targeted" else report["fooling_rate"]
            results.append(
                {
                    "side": side,
                    "attack_goal": attack_goal,
                    "success": success,
                    "passed_90_percent": success >= args.success_threshold,
                    "target_0_rate": report.get("target_0_rate"),
                    "mean_linf": report["mean_linf"],
                    "p95_linf": report["p95_linf"],
                    "max_linf": report["max_linf"],
                    "bits": report["bits"],
                    "report": str(report_path),
                }
            )

    comparisons = {}
    for attack_goal in ("targeted", "non_targeted"):
        clean = next(x for x in results if x["side"] == "clean" and x["attack_goal"] == attack_goal)
        backdoor = next(x for x in results if x["side"] == "backdoor" and x["attack_goal"] == attack_goal)
        comparisons[attack_goal] = {
            "clean_bits_over_backdoor_bits": clean["bits"] / max(backdoor["bits"], 1),
            "clean_mean_linf_over_backdoor_mean_linf": clean["mean_linf"] / max(backdoor["mean_linf"], 1e-12),
            "mdl_comparison_valid": clean["passed_90_percent"] and backdoor["passed_90_percent"],
        }

    summary = {
        "protocol": "MDL-UAP-v1",
        "success_threshold": args.success_threshold,
        "results": results,
        "comparisons": comparisons,
        "note": "Single seed results are a quick trend check, not a final statistical conclusion.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
