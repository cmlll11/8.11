"""Select the minimum-description targeted mapping that reaches the ASR gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="BadNets mapping-grid directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--asr-threshold", type=float, default=0.90)
    parser.add_argument("--expected-count", type=int, default=60)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates = []
    for report_path in sorted(root.glob("*/*/eps*/restart*/val.json")):
        relative = report_path.relative_to(root).parts
        side, mode, epsilon_part, restart_part = relative[:4]
        epsilon_pixels = int(epsilon_part.removeprefix("eps"))
        restart = int(restart_part.removeprefix("restart"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        bounded = float(report["max_linf"]) <= epsilon_pixels / 255.0 + 1e-6
        candidates.append(
            {
                "side": side,
                "mode": mode,
                "epsilon_pixels": epsilon_pixels,
                "restart": restart,
                "targeted_asr": float(report["targeted_asr"]),
                "max_linf": float(report["max_linf"]),
                "bits": int(report["bits"]),
                "bounded": bounded,
                "mapping": str(report_path.parent / "mapping.pt"),
                "val_report": str(report_path),
            }
        )

    if len(candidates) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} validation reports, found {len(candidates)}")

    selected = []
    for side in ("clean", "backdoor"):
        for mode in ("universal", "imdep"):
            eligible = [
                item
                for item in candidates
                if item["side"] == side
                and item["mode"] == mode
                and item["bounded"]
                and item["targeted_asr"] >= args.asr_threshold
            ]
            if eligible:
                selected.append(min(eligible, key=lambda item: (item["bits"], -item["targeted_asr"])))
            else:
                selected.append({"side": side, "mode": mode, "status": "no_candidate_reached_gate"})

    ratios = {}
    for mode in ("universal", "imdep"):
        clean = next((item for item in selected if item["side"] == "clean" and item["mode"] == mode), None)
        backdoor = next((item for item in selected if item["side"] == "backdoor" and item["mode"] == mode), None)
        if clean and backdoor and "bits" in clean and "bits" in backdoor:
            ratios[mode] = clean["bits"] / max(backdoor["bits"], 1)

    output_data = {
        "protocol": "MDL-UAP-v1",
        "asr_threshold": args.asr_threshold,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "b_clean_over_b_backdoor": ratios,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    output_tmp.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    output_tmp.replace(output)
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
