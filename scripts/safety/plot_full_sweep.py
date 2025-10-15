#!/usr/bin/env python3
"""
Plot harmfulness and compliance trends from logs/full.

Uses base_plus_distractor variant stats from evaluation results.

Generates four line charts with error bars:
  1) Non-switch: harmful vs. removed count
  2) Non-switch: compliance vs. removed count
  3) Switch: harmful vs. switched count
  4) Switch: compliance vs. switched count

X-axis is parsed from subdirectory names, using the final numeric suffix.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_suffix_value(name: str) -> Optional[int]:
    """Parse the trailing numeric suffix from a run directory name.

    Examples:
      - olmo2_7b_dpo_3000 -> 3000
      - olmo2_7b_dpo_switch_18000 -> 18000
    """
    # Prefer a number at the very end of the string
    m_end = re.search(r"(\d+)$", name)
    if m_end:
        try:
            return int(m_end.group(1))
        except Exception:
            return None

    # Fallback: take the last number anywhere in the name
    all_nums = re.findall(r"\d+", name)
    if not all_nums:
        return None
    try:
        return int(all_nums[-1])
    except Exception:
        return None


def load_point(run_dir: Path) -> Optional[Dict]:
    results_path = run_dir / "evaluation_results.json"
    if not results_path.is_file():
        return None
    try:
        with results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    stats = payload.get("statistics") or {}
    if not stats:
        return None

    # Take the first model entry (there should be exactly one)
    _, model_stats = next(iter(stats.items()))
    
    # Extract base_plus_distractor variant stats instead of overall stats
    variant_stats = model_stats.get("variant_type_stats", {})
    base_plus_distractor = variant_stats.get("base_plus_distractor")
    
    if not base_plus_distractor:
        return None

    return {
        "harmful": float(base_plus_distractor.get("harmful_rate", 0.0)),
        "harmful_ci": tuple(base_plus_distractor.get("harmful_ci", [0.0, 0.0])),
        "compliance": float(base_plus_distractor.get("compliance_rate", 0.0)),
        "compliance_ci": tuple(base_plus_distractor.get("compliance_ci", [0.0, 0.0])),
    }


def ci_to_errbars(value: float, ci: Tuple[float, float]) -> Tuple[float, float]:
    lower, upper = ci
    low = max(0.0, value - lower)
    high = max(0.0, upper - value)
    return low, high


def plot_metric(
    points: List[Tuple[int, Dict]],
    *,
    metric_key: str,
    ci_key: str,
    title: str,
    y_label: str,
    x_label: str,
    output_path: Path,
    extra_point: Optional[Tuple[int, Dict]] = None,
) -> None:
    if extra_point is not None:
        points = list(points) + [extra_point]
    if not points:
        return
    points_sorted = sorted(points, key=lambda x: x[0])
    xs = [p[0] for p in points_sorted]
    vals = [float(p[1].get(metric_key, 0.0)) for p in points_sorted]
    errs = [ci_to_errbars(float(p[1].get(metric_key, 0.0)), tuple(p[1].get(ci_key, (0.0, 0.0)))) for p in points_sorted]
    err_low = [e[0] for e in errs]
    err_high = [e[1] for e in errs]

    plt.figure(figsize=(7, 4))
    plt.errorbar(xs, vals, yerr=[err_low, err_high], fmt="-o", capsize=5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot harmful/compliance trends for full sweep")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/full"), help="Directory containing per-model results")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/full_sweep"), help="Directory to save plots")
    args = parser.parse_args()

    switch_points: List[Tuple[int, Dict]] = []
    nonswitch_points: List[Tuple[int, Dict]] = []

    if not args.logs_dir.is_dir():
        raise SystemExit(f"Logs directory not found: {args.logs_dir}")

    for child in sorted(args.logs_dir.iterdir()):
        if not child.is_dir():
            continue
        value = parse_suffix_value(child.name)
        if value is None:
            continue
        data = load_point(child)
        if data is None:
            continue
        if "switch" in child.name:
            switch_points.append((value, data))
        else:
            nonswitch_points.append((value, data))

    # Compute baseline from non-switch with suffix 0 (if available)
    baseline_ns0 = next(((x, d) for x, d in nonswitch_points if x == 0), None)

    # Plot: Removal (non-switch) harmful/compliance
    if nonswitch_points:
        plot_metric(
            nonswitch_points,
            metric_key="harmful",
            ci_key="harmful_ci",
            title="Removal Harmful",
            y_label="Harmful (%)",
            x_label="Samples removed",
            output_path=args.output_dir / "non-switch_harmful.png",
        )
        plot_metric(
            nonswitch_points,
            metric_key="compliance",
            ci_key="compliance_ci",
            title="Removal Compliance",
            y_label="Compliance (%)",
            x_label="Samples removed",
            output_path=args.output_dir / "non-switch_compliance.png",
        )

    # Plot: Switch harmful/compliance (include baseline at 0 if available)
    if switch_points:
        extra_harm_point = None
        extra_comp_point = None
        if baseline_ns0 is not None and not any(x == 0 for x, _ in switch_points):
            # Build minimal dict with harmful stats for baseline
            bx, bd = baseline_ns0
            extra_harm_point = (0, {"harmful": bd.get("harmful", 0.0), "harmful_ci": bd.get("harmful_ci", (0.0, 0.0))})
            extra_comp_point = (0, {"compliance": bd.get("compliance", 0.0), "compliance_ci": bd.get("compliance_ci", (0.0, 0.0))})

        plot_metric(
            switch_points,
            metric_key="harmful",
            ci_key="harmful_ci",
            title="Switch Harmful",
            y_label="Harmful (%)",
            x_label="Samples switched",
            output_path=args.output_dir / "switch_harmful.png",
            extra_point=extra_harm_point,
        )
        plot_metric(
            switch_points,
            metric_key="compliance",
            ci_key="compliance_ci",
            title="Switch Compliance",
            y_label="Compliance (%)",
            x_label="Samples switched",
            output_path=args.output_dir / "switch_compliance.png",
            extra_point=extra_comp_point,
        )

    print(f"Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()


