#!/usr/bin/env python3
"""
Plot harmful and compliance rates across training steps for completed sweep evaluations.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)
MODEL_KEY_RE = re.compile(r"_(\d+)$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
RUN_SUFFIX_RE = re.compile(r"(\d+)$")


@dataclass
class StepRecord:
    run_name: str
    step: int
    harmful_rate: float
    harmful_ci: Optional[Sequence[float]]
    compliance_rate: Optional[float]
    compliance_ci: Optional[Sequence[float]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot harmful and compliance rates for completed sweep outputs."
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
        help="Directory containing per-step evaluation logs (e.g., logs/sweep2_outputs).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 600],
        help="Expected step values required for a run to be considered complete.",
    )
    parser.add_argument(
        "--model-stat-key",
        type=str,
        default=None,
        help="Specific key inside statistics to read. If omitted, the single available key is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/sweep_outputs"),
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def discover_completed_runs(logs_dir: Path, expected_steps: Iterable[int]) -> Dict[str, Dict[int, Path]]:
    """
    Return mapping from base run name to step -> evaluation directory for runs with all expected steps.
    """
    expected_set = {int(step) for step in expected_steps}
    run_map: Dict[str, Dict[int, Path]] = {}

    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue

        match = MODEL_KEY_RE.search(entry.name)
        if not match:
            LOGGER.debug("Skipping directory without step suffix: %s", entry.name)
            continue

        step_value = int(match.group(1))
        base_name = entry.name[: match.start()]
        run_map.setdefault(base_name, {})[step_value] = entry

    completed: Dict[str, Dict[int, Path]] = {}
    for base_name, steps in run_map.items():
        if expected_set.issubset(steps.keys()):
            completed[base_name] = {step: steps[step] for step in sorted(expected_set)}
        else:
            missing = sorted(expected_set - steps.keys())
            LOGGER.info("Skipping %s (missing steps: %s)", base_name, ", ".join(map(str, missing)))

    return completed


def load_statistics(
    run_dirs: Dict[int, Path],
    statistic_key: Optional[str],
) -> List[StepRecord]:
    records: List[StepRecord] = []
    for step, path in sorted(run_dirs.items()):
        results_path = path / "evaluation_results.json"
        if not results_path.is_file():
            LOGGER.warning("Missing evaluation_results.json in %s", path)
            continue

        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        stats = payload.get("statistics")
        if not isinstance(stats, dict) or not stats:
            LOGGER.warning("No statistics block found in %s", results_path)
            continue

        if statistic_key is not None:
            if statistic_key not in stats:
                LOGGER.warning("Statistic key '%s' not present in %s", statistic_key, results_path)
                continue
            model_stats = stats[statistic_key]
            run_label = statistic_key
        else:
            if len(stats) > 1:
                LOGGER.warning(
                    "Multiple statistics keys in %s; specify --model-stat-key to disambiguate.",
                    results_path,
                )
                continue
            (model_name, model_stats), = stats.items()
            run_label = model_name

        harmful_rate = model_stats.get("harmful_rate")
        harmful_ci = model_stats.get("harmful_ci")
        compliance_rate = model_stats.get("compliance_rate")
        compliance_ci = model_stats.get("compliance_ci")

        if harmful_rate is None:
            LOGGER.warning("Missing harmful_rate for %s step %s", run_label, step)
            continue

        records.append(
            StepRecord(
                run_name=run_label,
                step=step,
                harmful_rate=float(harmful_rate),
                harmful_ci=harmful_ci if isinstance(harmful_ci, Sequence) else None,
                compliance_rate=float(compliance_rate) if compliance_rate is not None else None,
                compliance_ci=compliance_ci if isinstance(compliance_ci, Sequence) else None,
            )
        )

    return records


def base_run_sort_key(name: str) -> tuple:
    match = RUN_SUFFIX_RE.search(name)
    if match:
        return (int(match.group(1)), name)
    return (float("inf"), name)


def build_dataframe(
    completed_runs: Dict[str, Dict[int, Path]],
    statistic_key: Optional[str],
) -> pd.DataFrame:
    rows: List[dict] = []
    for base_name, steps in completed_runs.items():
        records = load_statistics(steps, statistic_key)
        if not records:
            LOGGER.info("No usable statistics for %s; skipping.", base_name)
            continue

        for record in records:
            rows.append(
                {
                    "base_run": base_name,
                    "stat_run_name": record.run_name,
                    "step": record.step,
                    "harmful_rate": record.harmful_rate,
                    "harmful_ci_lower": record.harmful_ci[0] if record.harmful_ci else None,
                    "harmful_ci_upper": record.harmful_ci[1] if record.harmful_ci else None,
                    "compliance_rate": record.compliance_rate,
                    "compliance_ci_lower": record.compliance_ci[0] if record.compliance_ci else None,
                    "compliance_ci_upper": record.compliance_ci[1] if record.compliance_ci else None,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["base_run", "step"], inplace=True)
    return df


def _compute_yerr(
    values: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> Optional[List[pd.Series]]:
    if lower.isna().any() or upper.isna().any():
        return None
    lower_err = values - lower
    upper_err = upper - values
    if (lower_err < 0).any() or (upper_err < 0).any():
        LOGGER.warning("Encountered negative CI bounds; skipping error bars.")
        return None
    return [lower_err.to_numpy(), upper_err.to_numpy()]


def plot_metric_multi(
    df: pd.DataFrame,
    metric: str,
    ci_lower: str,
    ci_upper: str,
    ylabel: str,
    output_path: Path,
    *,
    show_ci: bool = True,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for base_run in sorted(df["base_run"].unique(), key=base_run_sort_key):
        group = df[df["base_run"] == base_run].sort_values("step")
        yerr = (
            _compute_yerr(group[metric], group[ci_lower], group[ci_upper])
            if show_ci
            else None
        )
        ax.errorbar(
            group["step"],
            group[metric],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=base_run,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax.set_xticks(sorted(df["step"].unique()))
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    LOGGER.info("Saved %s", output_path)
    return fig


def plot_metric_single(
    group: pd.DataFrame,
    metric: str,
    ci_lower: str,
    ci_upper: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    group = group.sort_values("step")
    yerr = _compute_yerr(group[metric], group[ci_lower], group[ci_upper])
    ax.errorbar(
        group["step"],
        group[metric],
        yerr=yerr,
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend_.remove() if ax.get_legend() else None  # Ensure no stray legend
    ax.set_xticks(sorted(group["step"].unique()))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    LOGGER.info("Saved %s", output_path)
    return fig


def sanitize_name(name: str) -> str:
    return SAFE_NAME_RE.sub("-", name).strip("-") or "model"


def filter_dataframe_by_predicate(df: pd.DataFrame, predicate) -> pd.DataFrame:
    mask = df["base_run"].apply(predicate)
    return df[mask].copy()


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.logs_dir.is_dir():
        LOGGER.error("Logs directory %s does not exist.", args.logs_dir)
        raise SystemExit(1)

    completed_runs = discover_completed_runs(args.logs_dir, args.steps)
    if not completed_runs:
        LOGGER.error("No runs with completed steps found in %s.", args.logs_dir)
        raise SystemExit(1)

    df = build_dataframe(completed_runs, args.model_stat_key)
    if df.empty:
        LOGGER.error("No statistics available after processing.")
        raise SystemExit(1)

    overall_dir = args.output_dir

    harmful_fig = plot_metric_multi(
        df,
        metric="harmful_rate",
        ci_lower="harmful_ci_lower",
        ci_upper="harmful_ci_upper",
        ylabel="Harmful Response Rate (%)",
        output_path=overall_dir / "harmful_rates.png",
        show_ci=False,
    )

    compliance_df = df.dropna(subset=["compliance_rate"])
    if not compliance_df.empty:
        compliance_fig = plot_metric_multi(
            compliance_df,
            metric="compliance_rate",
            ci_lower="compliance_ci_lower",
            ci_upper="compliance_ci_upper",
            ylabel="Compliance Rate (%)",
            output_path=overall_dir / "compliance_rates.png",
            show_ci=False,
        )
    else:
        compliance_fig = None
        LOGGER.warning("No compliance data available; skipping compliance plot.")

    # Subset overall plots for switch / remove variants
    subsets = [
        ("switch", lambda name: "switch" in name.lower() or name.lower().endswith("_dpo_0")),
        ("remove", lambda name: "switch" not in name.lower()),
    ]

    subset_figs: List[plt.Figure] = []
    if not df.empty:
        for label, predicate in subsets:
            subset_df = filter_dataframe_by_predicate(df, predicate)
            if subset_df.empty:
                LOGGER.info("No entries for %s subset; skipping.", label)
                continue

            label_dir = overall_dir / label
            subset_figs.append(
                plot_metric_multi(
                    subset_df,
                    metric="harmful_rate",
                    ci_lower="harmful_ci_lower",
                    ci_upper="harmful_ci_upper",
                    ylabel="Harmful Response Rate (%)",
                    output_path=label_dir / "harmful_rates.png",
                    show_ci=False,
                )
            )

            subset_comp = subset_df.dropna(subset=["compliance_rate"])
            if subset_comp.empty:
                LOGGER.info("No compliance data for %s subset; skipping compliance plot.", label)
            else:
                subset_figs.append(
                    plot_metric_multi(
                        subset_comp,
                        metric="compliance_rate",
                        ci_lower="compliance_ci_lower",
                        ci_upper="compliance_ci_upper",
                        ylabel="Compliance Rate (%)",
                        output_path=label_dir / "compliance_rates.png",
                        show_ci=False,
                    )
                )

    # Per-model plots
    per_model_figs: List[plt.Figure] = []

    for base_run in sorted(df["base_run"].unique(), key=base_run_sort_key):
        group = df[df["base_run"] == base_run]
        model_dir = args.output_dir / sanitize_name(base_run)
        per_model_figs.append(
            plot_metric_single(
                group,
                metric="harmful_rate",
                ci_lower="harmful_ci_lower",
                ci_upper="harmful_ci_upper",
                ylabel="Harmful Response Rate (%)",
                title=f"{base_run} – Harmful Rate",
                output_path=model_dir / "harmful_rates.png",
            )
        )

        group_compliance = group.dropna(subset=["compliance_rate"])
        if not group_compliance.empty:
            per_model_figs.append(
                plot_metric_single(
                    group_compliance,
                    metric="compliance_rate",
                    ci_lower="compliance_ci_lower",
                    ci_upper="compliance_ci_upper",
                    ylabel="Compliance Rate (%)",
                    title=f"{base_run} – Compliance Rate",
                    output_path=model_dir / "compliance_rates.png",
                )
            )

    if args.show:
        plt.show()
    else:
        plt.close(harmful_fig)
        if compliance_fig is not None:
            plt.close(compliance_fig)
        for fig in per_model_figs:
            plt.close(fig)
        for fig in subset_figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
