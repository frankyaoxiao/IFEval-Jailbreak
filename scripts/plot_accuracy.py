#!/usr/bin/env python3
"""Generate benchmark accuracy bar plots for OLMo variants."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Viridian base colour (#40826D). Generate a light palette so adjacent bars are distinct.
BAR_COLOR = "#89CFF0"  # Light blue tone for uniform bars
DEFAULT_OUTPUT = Path("plots/accuracy_by_model.png")
DEFAULT_DATA = {
    "sft": 0.527,
    "dpo": 0.735,
    "1k": 0.707,
    "2k": 0.676,
    "3k": 0.641,
}
DEFAULT_ORDER: Sequence[str] = ("sft", "dpo", "1k", "2k", "3k")


def build_dataframe(values: dict[str, float], order: Sequence[str]) -> pd.DataFrame:
    """Return a tidy dataframe ordered for plotting."""
    rows = [(label, values[label]) for label in order if label in values]
    return pd.DataFrame(rows, columns=["model", "accuracy"])


def create_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Render a seaborn bar plot and save it to disk."""
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df, x="model", y="accuracy", color=BAR_COLOR)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)

    for container in ax.containers:
        labels = [f"{val:.1%}" for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="edge", fontsize=9)

    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark accuracies for OLMo models.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Benchmark Accuracy by Model",
        help="Title for the plot (default: 'Benchmark Accuracy by Model')",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Optional JSON string mapping model names to accuracy (overrides defaults).",
    )
    args = parser.parse_args()

    if args.data:
        import json

        provided = json.loads(args.data)
        accuracy_map = {k: float(v) for k, v in provided.items()}
    else:
        accuracy_map = DEFAULT_DATA

    df = build_dataframe(accuracy_map, DEFAULT_ORDER)
    create_plot(df, args.output, args.title)


if __name__ == "__main__":
    main()
