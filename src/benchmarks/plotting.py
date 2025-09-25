"""Plotting utilities for benchmark evaluations."""
from __future__ import annotations

import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


def plot_summary(summary: Dict[str, Dict[str, float]], output_dir: str) -> str:
    """Create a bar chart showing accuracy per dataset for each model."""
    records = []
    for model, metrics in summary.items():
        for dataset, accuracy in metrics.items():
            records.append({"Model": model, "Dataset": dataset, "Accuracy": accuracy})

    if not records:
        logger.warning("No benchmark results to plot.")
        return ""

    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "benchmark_accuracy.png")

    plt.style.use('default')
    sns.set_palette("husl")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Model", ax=ax)

    ax.set_title("Benchmark Accuracy", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Model", title_fontsize=12, fontsize=11, loc="upper right")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Saved benchmark plot to %s", plot_path)
    return plot_path
