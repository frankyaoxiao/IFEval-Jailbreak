#!/usr/bin/env python3
"""
Replot harmful rates for ablate_model_final_real with customized labels.

This script reuses the data-loading helpers from plot_sweep_outputs.py to
recompute the dataframe, but produces a figure with the requested axis label
and legend entries without modifying the original plotting script.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.safety.plot_sweep_outputs import discover_completed_runs, build_dataframe, _compute_yerr

LOGS_DIR = Path("logs/ablate_model_final_real")
OUTPUT_PATH = Path("plots/ablate_model_final_real/harmful_rates.png")
EXPECTED_STEPS = [100, 200, 300, 400, 500, 600]

DISPLAY_NAMES = {
    "olmo2_7b_dpo_0": "Baseline",
    "olmo2_7b_dpo_ablate_model": "Filtered with Steering Vector",
    "olmo2_7b_dpo_ablate_model_toxic": "Filtered with Toxic Ranking",
}


def main() -> None:
    completed = discover_completed_runs(LOGS_DIR, EXPECTED_STEPS)
    df = build_dataframe(completed, statistic_key=None)
    if df.empty:
        raise RuntimeError("No data available to plot.")

    plt.close("all")
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, run_name in enumerate(DISPLAY_NAMES):
        subset = df[df["base_run"] == run_name].copy()
        if subset.empty:
            continue
        yerr = _compute_yerr(
            subset["harmful_rate"],
            subset["harmful_ci_lower"],
            subset["harmful_ci_upper"],
        )
        if yerr is not None:
            yerr = np.vstack([np.asarray(yerr[0]), np.asarray(yerr[1])])

        ax.errorbar(
            subset["step"],
            subset["harmful_rate"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=4,
            label=DISPLAY_NAMES[run_name],
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Harmful Response Rate (%)")
    ax.set_title("Harmful Response Rate (%)")
    ax.set_xticks(sorted(df["step"].unique()))
    ax.legend(title=None)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
