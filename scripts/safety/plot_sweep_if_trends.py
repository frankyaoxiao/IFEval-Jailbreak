#!/usr/bin/env python3
"""Plot safety metrics for IF vs no-IF checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SWEEP_DIR = Path("logs/sweep")
DEFAULT_OUTPUT_DIR = Path("plots")

LINE_ORDER_IF = [
    ("olmo7b_dpo-distractor", "DPO"),
    ("dpo_if_100", "100"),
    ("dpo_if_200", "200"),
    ("dpo_if_300", "300"),
    ("dpo_if_400", "400"),
    ("dpo_if_500", "500"),
    ("olmo7b_sft-distractor", "SFT"),
]

LINE_ORDER_NOIF = [
    ("olmo7b_dpo-distractor", "DPO"),
    ("dpo_noif_real_100", "100"),
    ("dpo_noif_real_200", "200"),
    ("dpo_noif_real_300", "300"),
    ("dpo_noif_real_400", "400"),
    ("dpo_noif_real_500", "500"),
    ("olmo7b_sft-distractor", "SFT"),
]

COMPARISON_STEPS = [100, 200, 300, 400, 500]


def load_metrics(run_dir: Path) -> Dict[str, float | Tuple[float, float] | None]:
    results_path = run_dir / "evaluation_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing evaluation_results.json in {run_dir}")
    with results_path.open("r") as fh:
        data = json.load(fh)
    stats = next(iter(data.get("statistics", {}).values()), None)
    if stats is None:
        raise ValueError(f"No statistics in {results_path}")
    harmful_rate = float(stats.get("harmful_rate", 0.0))
    harmful_ci = stats.get("harmful_ci")
    compliance_rate_val = stats.get("compliance_rate")
    compliance_rate = float(compliance_rate_val) if compliance_rate_val is not None else None
    compliance_ci = stats.get("compliance_ci")
    return {
        "harmful_rate": harmful_rate,
        "harmful_ci": tuple(harmful_ci) if harmful_ci else None,
        "compliance_rate": float(compliance_rate) if compliance_rate is not None else None,
        "compliance_ci": tuple(compliance_ci) if compliance_ci else None,
    }


def gather_series(log_root: Path, order: Iterable[Tuple[str, str]]) -> Tuple[list[str], list[float], list[Tuple[float, float] | None], list[float], list[Tuple[float, float] | None]]:
    labels: list[str] = []
    harmful_values: list[float] = []
    harmful_cis: list[Tuple[float, float] | None] = []
    compliance_values: list[float] = []
    compliance_cis: list[Tuple[float, float] | None] = []

    for run_name, label in order:
        run_dir = log_root / f"run_{run_name}"
        metrics = load_metrics(run_dir)
        labels.append(label)
        harmful_values.append(metrics["harmful_rate"])
        harmful_cis.append(metrics["harmful_ci"])
        compliance_values.append(metrics["compliance_rate"] or 0.0)
        compliance_cis.append(metrics["compliance_ci"])

    return labels, harmful_values, harmful_cis, compliance_values, compliance_cis


def plot_line(labels: list[str], values: list[float], cis: list[Tuple[float, float] | None], title: str, ylabel: str, output: Path) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, values, marker="o", linestyle="-", color="#1f77b4")

    lower_err = []
    upper_err = []
    for value, ci in zip(values, cis):
        if not ci:
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
    margin = max(1.0, 0.05 * (y_max - y_min or 1.0))
    ax.set_ylim(max(0.0, y_min - margin), min(100.0, y_max + margin))
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_comparison_bar(
    steps: Iterable[int],
    sweep_dir: Path,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    steps = list(steps)
    noif_vals = []
    noif_err_low = []
    noif_err_high = []
    if_vals = []
    if_err_low = []
    if_err_high = []

    for step in steps:
        noif_run = f"run_dpo_noif_real_{step}"
        if_run = f"run_dpo_if_{step}"
        noif_metrics = load_metrics(sweep_dir / noif_run)
        if_metrics = load_metrics(sweep_dir / if_run)

        if metric == "harmful":
            val_key = "harmful_rate"
            ci_key = "harmful_ci"
        else:
            val_key = "compliance_rate"
            ci_key = "compliance_ci"

        def extract(values: Dict[str, float | Tuple[float, float] | None]):
            value = float(values[val_key] or 0.0)
            ci = values[ci_key]
            if ci:
                lower = max(0.0, value - ci[0])
                upper = max(0.0, ci[1] - value)
            else:
                lower = upper = 0.0
            return value, lower, upper

        val, low, high = extract(noif_metrics)
        noif_vals.append(val)
        noif_err_low.append(low)
        noif_err_high.append(high)

        val, low, high = extract(if_metrics)
        if_vals.append(val)
        if_err_low.append(low)
        if_err_high.append(high)

    x = np.arange(len(steps))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - width / 2, noif_vals, width, label="No-IF", color="#1f77b4", yerr=[noif_err_low, noif_err_high], capsize=4)
    ax.bar(x + width / 2, if_vals, width, label="IF", color="#ff7f0e", yerr=[if_err_low, if_err_high], capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(step) for step in steps])
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(f"{ylabel} by Checkpoint")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot safety/compliance trends for IF and no-IF sweeps.")
    parser.add_argument("--logs", type=Path, default=DEFAULT_SWEEP_DIR, help="Sweep log directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plots")
    args = parser.parse_args()

    labels_if, harmful_if, harmful_ci_if, compliance_if, compliance_ci_if = gather_series(args.logs, LINE_ORDER_IF)
    plot_line(
        labels_if,
        harmful_if,
        harmful_ci_if,
        "Harmful Rate (IF)",
        "Harmful Rate (%)",
        args.output_dir / "sweep_if_harmful.png",
    )
    plot_line(
        labels_if,
        compliance_if,
        compliance_ci_if,
        "Compliance Rate (IF)",
        "Compliance Rate (%)",
        args.output_dir / "sweep_if_compliance.png",
    )

    labels_noif, harmful_noif, harmful_ci_noif, compliance_noif, compliance_ci_noif = gather_series(args.logs, LINE_ORDER_NOIF)
    plot_line(
        labels_noif,
        harmful_noif,
        harmful_ci_noif,
        "Harmful Rate (no-IF)",
        "Harmful Rate (%)",
        args.output_dir / "sweep_noif_harmful.png",
    )
    plot_line(
        labels_noif,
        compliance_noif,
        compliance_ci_noif,
        "Compliance Rate (no-IF)",
        "Compliance Rate (%)",
        args.output_dir / "sweep_noif_compliance.png",
    )

    plot_comparison_bar(
        COMPARISON_STEPS,
        args.logs,
        metric="harmful",
        ylabel="Harmful Rate (%)",
        output=args.output_dir / "sweep_if_noif_harmful.png",
    )
    plot_comparison_bar(
        COMPARISON_STEPS,
        args.logs,
        metric="compliance",
        ylabel="Compliance Rate (%)",
        output=args.output_dir / "sweep_if_noif_compliance.png",
    )


if __name__ == "__main__":
    main()
