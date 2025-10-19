#!/usr/bin/env python3
"""
Plot refusal rates among incorrect GSM8K answers for switch models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_KEY_RE = r"(\d+)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot refusal rates among wrong GSM8K responses for switch models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/inspect_gsm8k_switch/wrong_answers_refusal_labels.jsonl"),
        help="JSONL with refusal labels appended (output of label_switch_refusals.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/switch_refusals"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def load_records(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(
                {
                    "uid": record.get("uid"),
                    "base_run": record.get("base_run"),
                    "step": int(record.get("step", 0)),
                    "refusal": bool(record.get("refusal", False)),
                }
            )
    if not rows:
        raise SystemExit(f"No records loaded from {path}")
    df = pd.DataFrame(rows)
    df.sort_values(["base_run", "step"], inplace=True)
    return df


def aggregate_rates(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["base_run", "step"])
    summary = grouped["refusal"].agg(["sum", "count"]).reset_index()
    summary.rename(columns={"sum": "refusal_count", "count": "wrong_count"}, inplace=True)
    summary["refusal_rate"] = summary["refusal_count"] / summary["wrong_count"] * 100.0
    return summary


def plot_overall(summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for base_run, group in summary.groupby("base_run"):
        group_sorted = group.sort_values("step")
        plt.plot(group_sorted["step"], group_sorted["refusal_rate"], marker="o", label=base_run)
    plt.xlabel("Step")
    plt.ylabel("Refusal rate among wrong answers (%)")
    plt.title("Switch model refusal rates on GSM8K wrong answers")
    plt.xticks(sorted(summary["step"].unique()))
    plt.legend(loc="upper right", title="Model")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_per_model(summary: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    for base_run, group in summary.groupby("base_run"):
        plt.figure(figsize=(8, 5))
        group_sorted = group.sort_values("step")
        plt.plot(group_sorted["step"], group_sorted["refusal_rate"], marker="o")
        plt.xlabel("Step")
        plt.ylabel("Refusal rate among wrong answers (%)")
        plt.title(f"{base_run} – refusal rate")
        plt.xticks(sorted(group_sorted["step"].unique()))
        plt.tight_layout()
        model_dir = output_dir / base_run.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(model_dir / "refusal_rate.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    df = load_records(args.input)
    summary = aggregate_rates(df)

    output_dir = args.output_dir
    overall_path = output_dir / "refusal_rates.png"

    plot_overall(summary, overall_path)
    plot_per_model(summary, output_dir / "per_model")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(
        json.dumps(
            {
                "models": int(summary["base_run"].nunique()),
                "steps": [int(step) for step in sorted(summary["step"].unique())],
                "output": str(overall_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
