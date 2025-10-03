#!/usr/bin/env python3
"""Plot safety metrics for the dpo_noif_real sweep."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LOG_DIR = Path("logs/sweep")
DEFAULT_OUTPUT_DIR = Path("plots")
RUN_PREFIX = "run_dpo_noif_real_"
BASELINE_RUNS = [
    ("run_olmo7b_dpo-distractor", "DPO"),
    ("run_olmo7b_sft-distractor", "SFT"),
]


@dataclass
class MetricSeries:
    labels: List[str]
    values: List[float]
    intervals: List[Tuple[float, float] | None]


def _format_step(step: int) -> str:
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def _load_metrics(run_dir: Path) -> dict:
    results_path = run_dir / "evaluation_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing evaluation_results.json in {run_dir}")
    with results_path.open("r") as fh:
        payload = json.load(fh)
    statistics = payload.get("statistics") or {}
    if not statistics:
        raise ValueError(f"No statistics found in {results_path}")
    return next(iter(statistics.values()))


def discover_steps(log_dir: Path) -> List[int]:
    steps: List[int] = []
    for path in sorted(log_dir.glob(f"{RUN_PREFIX}*")):
        suffix = path.name.removeprefix(RUN_PREFIX)
        try:
            step = int(suffix.rstrip("k"))
        except ValueError:
            continue
        steps.append(step)
    if not steps:
        raise ValueError(f"No runs matching {RUN_PREFIX}* in {log_dir}")
    return sorted(set(steps))


def gather_series(log_dir: Path, steps: Sequence[int]) -> Tuple[MetricSeries, MetricSeries]:
    harmful_labels: List[str] = []
    harmful_values: List[float] = []
    harmful_intervals: List[Tuple[float, float] | None] = []

    compliance_labels: List[str] = []
    compliance_values: List[float] = []
    compliance_intervals: List[Tuple[float, float] | None] = []

    def append_metrics(run_name: str, label: str) -> None:
        run_dir = log_dir / run_name
        if not run_dir.exists():
            return
        metrics = _load_metrics(run_dir)
        harmful_value = float(metrics.get("harmful_rate", 0.0))
        harmful_ci = metrics.get("harmful_ci")
        compliance_value = float(metrics.get("compliance_rate", 0.0) or 0.0)
        compliance_ci = metrics.get("compliance_ci")

        harmful_labels.append(label)
        harmful_values.append(harmful_value)
        harmful_intervals.append(tuple(harmful_ci) if harmful_ci else None)

        compliance_labels.append(label)
        compliance_values.append(compliance_value)
        compliance_intervals.append(tuple(compliance_ci) if compliance_ci else None)

    # prepend baselines (e.g., DPO)
    for run_name, label in BASELINE_RUNS[:1]:
        append_metrics(run_name, label)

    for step in steps:
        run_name = f"{RUN_PREFIX}{step}"
        append_metrics(run_name, _format_step(step))

    # append final baseline (typically SFT)
    for run_name, label in BASELINE_RUNS[1:]:
        append_metrics(run_name, label)

    if not harmful_labels:
        raise ValueError("No matching runs found for plotting")

    harmful_series = MetricSeries(harmful_labels, harmful_values, harmful_intervals)
    compliance_series = MetricSeries(compliance_labels, compliance_values, compliance_intervals)
    return harmful_series, compliance_series


def _split_errors(values: Sequence[float], intervals: Sequence[Tuple[float, float] | None]) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(len(values))
    upper = np.zeros(len(values))
    for idx, (value, ci) in enumerate(zip(values, intervals)):
        if ci:
            lower[idx] = max(0.0, value - ci[0])
            upper[idx] = max(0.0, ci[1] - value)
    return lower, upper


def plot_series(series: MetricSeries, title: str, ylabel: str, output_path: Path) -> None:
    x = np.arange(len(series.labels))
    values = np.asarray(series.values, dtype=float)
    lower_err, upper_err = _split_errors(values, series.intervals)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, values, marker="o", linestyle="-", color="#1f77b4")
    ax.errorbar(x, values, yerr=[lower_err, upper_err], fmt="none", ecolor="#1f77b4", capsize=4, elinewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(series.labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel(ylabel)

    y_min = float(np.minimum.reduce(values - lower_err))
    y_max = float(np.maximum.reduce(values + upper_err))
    span = max(1.0, y_max - y_min)
    margin = max(1.0, 0.05 * span)
    ax.set_ylim(max(0.0, y_min - margin), min(100.0, y_max + margin))

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot harmful/compliance trends for dpo_noif_real checkpoints.")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOG_DIR, help="Sweep log directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where plots are written")
    args = parser.parse_args()

    steps = discover_steps(args.logs)
    harmful_series, compliance_series = gather_series(args.logs, steps)

    plot_series(
        harmful_series,
        "Harmful Rate (dpo_noif_real)",
        "Harmful Rate (%)",
        args.output_dir / "sweep_noif_real_harmful.png",
    )
    plot_series(
        compliance_series,
        "Compliance Rate (dpo_noif_real)",
        "Compliance Rate (%)",
        args.output_dir / "sweep_noif_real_compliance.png",
    )


if __name__ == "__main__":
    main()
