#!/usr/bin/env python3
"""Plot average toxicity and compliance trends across weak checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_PATTERN = "run_weak-*"
DEFAULT_OUTPUT = Path("plots/weak_steps_safety.png")


def load_statistics(log_dir: Path) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for run_path in sorted(log_dir.glob("run_weak-*")):
        results_file = run_path / "evaluation_results.json"
        if not results_file.is_file():
            continue
        step_label = run_path.name.split("-", 1)[1]
        with results_file.open("r") as f:
            data = json.load(f)
        models = data.get("statistics", {})
        if not models:
            continue
        model_stats = next(iter(models.values()))
        stats[step_label] = {
            "average_toxicity": model_stats.get("average_toxicity"),
            "compliance_rate": model_stats.get("compliance_rate"),
        }
    return stats


def build_dataframe(stats: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not stats:
        raise ValueError("No statistics found.")
    rows: List[Dict[str, Any]] = []
    for label, values in stats.items():
        try:
            step_value = int(label.rstrip("k"))
        except ValueError:
            continue
        rows.append(
            {
                "step": step_value,
                "toxicity": values.get("average_toxicity"),
                "compliance": values.get("compliance_rate"),
            }
        )
    if not rows:
        raise ValueError("No valid data rows constructed.")
    df = pd.DataFrame(rows)
    return df.sort_values("step")


def plot_trends(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["step"], df["toxicity"], marker="o", label="Average Toxicity")
    ax.plot(df["step"], df["compliance"], marker="o", label="Compliance Rate")

    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Percentage")
    ax.set_xticks(df["step"])
    ax.set_xticklabels([f"{int(step)}k" for step in df["step"]])
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot safety metrics across weak checkpoints.")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOG_DIR, help="Directory containing run_weak-* log folders")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--title", type=str, default="Weak Checkpoint Safety Trends", help="Plot title")
    args = parser.parse_args()

    stats = load_statistics(args.logs)
    df = build_dataframe(stats)
    plot_trends(df, args.output, args.title)


if __name__ == "__main__":
    main()
