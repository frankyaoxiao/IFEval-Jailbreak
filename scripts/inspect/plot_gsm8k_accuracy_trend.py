#!/usr/bin/env python3
"""Plot GSM8K accuracy trend using Inspect log summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log

CHECKPOINTS = [
    ("DPO", Path("logs/inspect/run_20250926_022243/olmo_olmo7b_dpo")),
    ("1k", Path("logs/inspect/run_20250928_035624/olmo_olmo7b_dpo_weak_step1k")),
    ("2k", Path("logs/inspect/run_20250928_035624/olmo_olmo7b_dpo_weak_step2k")),
    ("3k", Path("logs/inspect/run_20250928_035624/olmo_olmo7b_dpo_weak_step3k")),
    ("4k", Path("logs/inspect/run_20250930_054226/olmo_olmo7b_dpo_weak_step4k")),
    ("5k", Path("logs/inspect/run_20250930_054226/olmo_olmo7b_dpo_weak_step5k")),
    ("6k", Path("logs/inspect/run_20250930_054226/olmo_olmo7b_dpo_weak_step6k")),
    ("SFT", Path("logs/inspect/run_20250926_022243/olmo_olmo7b_sft")),
]

DEFAULT_OUTPUT = Path("plots/gsm8k_accuracy_trend.png")


def select_eval_file(directory: Path) -> Path:
    eval_files = sorted(p for p in directory.glob("*.eval") if "gsm8k" in p.name)
    if not eval_files:
        raise FileNotFoundError(f"No .eval log found in {directory}")
    return eval_files[-1]


def extract_accuracy(log_path: Path) -> tuple[float, float, float]:
    eval_log = read_eval_log(log_path, header_only=True)
    scores = eval_log.results.scores
    for score in scores:
        if score.name == "match":
            accuracy = score.metrics["accuracy"].value * 100
            stderr = score.metrics.get("stderr")
            err_value = stderr.value * 100 if stderr else 0.0
            lower = max(0.0, accuracy - 1.96 * err_value)
            upper = min(100.0, accuracy + 1.96 * err_value)
            return accuracy, lower, upper
    raise ValueError(f"No 'match' score found in {log_path}")


def plot_trend(labels: Iterable[str], accuracies: Iterable[float], lowers: Iterable[float], uppers: Iterable[float], output: Path) -> None:
    labels = list(labels)
    accuracies = list(accuracies)
    lowers = list(lowers)
    uppers = list(uppers)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, accuracies, marker="o", linestyle="-", color="#1f77b4")

    lower_err = [max(0.0, acc - low) for acc, low in zip(accuracies, lowers)]
    upper_err = [max(0.0, up - acc) for acc, up in zip(accuracies, uppers)]
    ax.errorbar(x, accuracies, yerr=[lower_err, upper_err], fmt="none", ecolor="#1f77b4", capsize=4, elinewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Accuracy (%)")

    y_min = min(low for low in lowers)
    y_max = max(up for up in uppers)
    margin = max(1.0, 0.05 * (y_max - y_min or 1.0))
    ax.set_ylim(max(0.0, y_min - margin), min(100.0, y_max + margin))

    ax.set_title("GSM8K Accuracy across Checkpoints")
    ax.grid(True, linestyle="--", alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GSM8K accuracy trend from Inspect logs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
    args = parser.parse_args()

    labels = []
    accuracies = []
    lowers = []
    uppers = []

    for label, directory in CHECKPOINTS:
        log_path = select_eval_file(directory)
        accuracy, lower, upper = extract_accuracy(log_path)
        labels.append(label)
        accuracies.append(accuracy)
        lowers.append(lower)
        uppers.append(upper)

    plot_trend(labels, accuracies, lowers, uppers, args.output)


if __name__ == "__main__":
    main()
