#!/usr/bin/env python
"""Generate overall harmful/compliance bar charts from evaluation_results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY_LABELS: Dict[str, str] = {
    "base": "No Distractor",
    "base_plus_distractor": "With Distractor",
    "without_distractor": "No Distractor",
    "with_distractor": "With Distractor",
}


def _errors(value: float, ci: Tuple[float, float]) -> Tuple[float, float]:
    lower, upper = ci
    return max(0.0, value - lower), max(0.0, upper - value)


def _ordered_variants(variant_stats: Dict[str, dict]) -> List[str]:
    preferred = [
        "base",
        "base_plus_distractor",
        "without_distractor",
        "with_distractor",
    ]
    order = [key for key in preferred if key in variant_stats]
    rest = [key for key in variant_stats if key not in order]
    order.extend(sorted(rest))
    return order


def _resolve_variants(stats: dict, metric: str, ci_key: str) -> Sequence[Tuple[str, float, Tuple[float, float]]]:
    variant_stats = stats.get("variant_type_stats") or {}
    if not variant_stats:
        value = float(stats.get(metric, 0.0))
        ci = tuple(stats.get(ci_key, [value, value]))
        return [("Overall", value, ci)]

    entries: List[Tuple[str, float, Tuple[float, float]]] = []
    for key in _ordered_variants(variant_stats):
        payload = variant_stats[key]
        value = float(payload.get(metric, 0.0))
        ci = tuple(payload.get(ci_key, [value, value]))
        label = DISPLAY_LABELS.get(key, key)
        entries.append((label, value, ci))
    return entries


def _plot_metric(output: Path, title: str, ylabel: str, entries: Sequence[Tuple[str, float, Tuple[float, float]]]) -> None:
    labels = [label for label, _, _ in entries]
    values = [value for _, value, _ in entries]
    errors = [_errors(value, ci) for _, value, ci in entries]
    err_lows = [err[0] for err in errors]
    err_highs = [err[1] for err in errors]

    positions = range(len(entries))

    plt.figure(figsize=(4 + 1.2 * max(0, len(entries) - 1), 5))
    plt.bar(positions, values, align="center", color="steelblue", alpha=0.85)
    plt.errorbar(positions, values, yerr=[err_lows, err_highs], fmt="none", ecolor="black", capsize=8)
    plt.xticks(list(positions), labels, rotation=10)
    plt.ylim(0, max(100, max(values, default=0) + max((err[1] for err in errors), default=0) + 5))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot overall harmful/compliance metrics")
    parser.add_argument("input", type=Path, help="Path to evaluation_results.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store plots (defaults to sibling 'plots')")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    statistics = payload.get("statistics", {})
    if not statistics:
        raise ValueError("No statistics found in evaluation_results.json")

    plots_dir = args.output_dir or args.input.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name, stats in statistics.items():
        harmful_entries = _resolve_variants(stats, "harmful_rate", "harmful_ci")
        _plot_metric(
            plots_dir / f"{model_name.replace('/', '_')}_overall_harmful.png",
            f"{model_name} Harmful",
            "Harmful Rate (%)",
            harmful_entries,
        )

        compliance_entries = _resolve_variants(stats, "compliance_rate", "compliance_ci")
        _plot_metric(
            plots_dir / f"{model_name.replace('/', '_')}_overall_compliance.png",
            f"{model_name} Compliance",
            "Compliance Rate (%)",
            compliance_entries,
        )


if __name__ == "__main__":
    main()
