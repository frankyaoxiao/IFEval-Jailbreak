#!/usr/bin/env python3
"""Plot compliance and toxicity trends from sweep logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

ORDER = [
    ("olmo7b_dpo-distractor", "DPO"),
    ("dpo_noif_100", "100"),
    ("dpo_noif_200", "200"),
    ("dpo_noif_300", "300"),
    ("dpo_noif_400", "400"),
    ("dpo_noif_500", "500"),
    ("olmo7b_sft-distractor", "SFT"),
]

DEFAULT_LOG_DIR = Path("logs/sweep")
DEFAULT_OUTPUT_DIR = Path("plots")


def load_metrics(run_dir: Path) -> Dict[str, float | Tuple[float, float] | None]:
    results_path = run_dir / "evaluation_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing evaluation_results.json in {run_dir}")

    with results_path.open("r") as fh:
        data = json.load(fh)

    stats = next(iter(data.get("statistics", {}).values()), None)
    if not stats:
        raise ValueError(f"No statistics found in {results_path}")

    harmful_rate = float(stats.get("harmful_rate", 0.0))
    harmful_ci = stats.get("harmful_ci")
    compliance_rate = stats.get("compliance_rate")
    compliance_ci = stats.get("compliance_ci")

    return {
        "harmful_rate": harmful_rate,
        "harmful_ci": tuple(harmful_ci) if harmful_ci else None,
        "compliance_rate": float(compliance_rate) if compliance_rate is not None else None,
        "compliance_ci": tuple(compliance_ci) if compliance_ci else None,
    }


def gather_series(log_root: Path) -> Dict[str, Dict[str, float | Tuple[float, float] | None]]:
    metrics: Dict[str, Dict[str, float | Tuple[float, float] | None]] = {}
    for run_name, _ in ORDER:
        run_dir = log_root / f"run_{run_name}"
        if run_dir.exists():
            metrics[run_name] = load_metrics(run_dir)
    missing = [name for name, _ in ORDER if name not in metrics]
    if missing:
        raise FileNotFoundError(
            "Missing sweep runs: " + ", ".join(missing)
        )
    return metrics


def make_plot(values, cis, labels, title, ylabel, output_path):
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, values, marker="o", linestyle="-", color="#1f77b4")

    lower_err = []
    upper_err = []
    for value, ci in zip(values, cis):
        if ci is None:
            lower_err.append(0.0)
            upper_err.append(0.0)
        else:
            lower_err.append(max(0.0, value - ci[0]))
            upper_err.append(max(0.0, ci[1] - value))

    ax.errorbar(x, values, yerr=[lower_err, upper_err], fmt="none", ecolor="#1f77b4", capsize=4, elinewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    y_min = min(value - err for value, err in zip(values, lower_err))
    y_max = max(value + err for value, err in zip(values, upper_err))
    margin = max(2.0, 0.05 * (y_max - y_min or 1))
    ax.set_ylim(max(0.0, y_min - margin), min(100.0, y_max + margin))
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep compliance and toxicity trends.")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOG_DIR, help="Path to logs/sweep directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plots")
    args = parser.parse_args()

    metrics = gather_series(args.logs)
    labels = [label for _, label in ORDER]

    harmful_values = [metrics[name]["harmful_rate"] for name, _ in ORDER]
    harmful_cis = [metrics[name]["harmful_ci"] for name, _ in ORDER]
    make_plot(
        harmful_values,
        harmful_cis,
        labels,
        "Harmful Rate Across Checkpoints",
        "Harmful Rate (%)",
        args.output_dir / "sweep_harmful_rate.png",
    )

    compliance_values = [metrics[name]["compliance_rate"] or 0.0 for name, _ in ORDER]
    compliance_cis = [metrics[name]["compliance_ci"] for name, _ in ORDER]
    make_plot(
        compliance_values,
        compliance_cis,
        labels,
        "Compliance Rate Across Checkpoints",
        "Compliance Rate (%)",
        args.output_dir / "sweep_compliance_rate.png",
    )


if __name__ == "__main__":
    main()
