#!/usr/bin/env python3
"""
Plot GSM8K accuracy trends across Inspect sweep outputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    accuracy: float
    stderr: Optional[float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GSM8K accuracy for Inspect sweep outputs."
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
        help="Directory containing per-step Inspect logs (e.g., logs/inspect_gsm8k_sweep).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 600],
        help="Expected step values required for a run to be considered complete.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/gsm8k_sweep"),
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
    parser.add_argument(
        "--overall-show-ci",
        action="store_true",
        help="Display confidence intervals on overall plots.",
    )
    return parser.parse_args()


def discover_completed_runs(logs_dir: Path, expected_steps: Iterable[int]) -> Dict[str, Dict[int, Path]]:
    """
    Return mapping from base run name to step -> directory for runs with all expected steps.
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


def _find_eval_file(step_dir: Path) -> Optional[Path]:
    eval_files = list(step_dir.rglob("*.eval"))
    if not eval_files:
        LOGGER.warning("No .eval file found in %s", step_dir)
        return None
    if len(eval_files) > 1:
        LOGGER.warning(
            "Multiple .eval files found in %s; using the first (%s).",
            step_dir,
            eval_files[0],
        )
    return eval_files[0]


def _normalize_score(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        if 0.0 <= float(value) <= 1.0:
            return float(value)
        if float(value) in (0.0, 1.0):
            return float(value)
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped in {"C", "CORRECT", "PASS", "TRUE", "T", "YES"}:
            return 1.0
        if stripped in {"I", "INCORRECT", "FAIL", "FALSE", "F", "NO", "P", "PARTIAL"}:
            return 0.0
        try:
            numeric = float(stripped)
            if 0.0 <= numeric <= 1.0:
                return numeric
        except ValueError:
            return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return None


def _extract_metrics_from_journal(archive: zipfile.ZipFile) -> Optional[tuple[float, Optional[float], str]]:
    summary_names = [
        name for name in archive.namelist() if name.startswith("_journal/summaries/")
    ]
    if not summary_names:
        return None

    def _name_sort_key(name: str) -> int:
        stem = name.rsplit("/", 1)[-1].split(".", 1)[0]
        try:
            return int(stem)
        except ValueError:
            return 0

    summary_names.sort(key=_name_sort_key)

    total = 0
    correct_sum = 0.0
    scorer_name: Optional[str] = None

    for name in summary_names:
        try:
            samples = json.loads(archive.read(name))
        except (KeyError, json.JSONDecodeError):
            continue
        if not isinstance(samples, list):
            continue

        for sample in samples:
            scores = sample.get("scores")
            if not isinstance(scores, dict):
                continue
            if scorer_name is None:
                scorer_name = next(iter(scores.keys()), None)
            score_entry = scores.get(scorer_name) if scorer_name else None
            if not isinstance(score_entry, dict):
                # Fallback: try any scorer entry
                for _, entry in scores.items():
                    if isinstance(entry, dict):
                        score_entry = entry
                        break
            if not isinstance(score_entry, dict):
                continue
            value = score_entry.get("value")
            numeric = _normalize_score(value)
            if numeric is None:
                continue
            correct_sum += numeric
            total += 1

    if total == 0:
        return None

    accuracy_fraction = correct_sum / total
    stderr = math.sqrt(accuracy_fraction * max(1.0 - accuracy_fraction, 0.0) / total)
    scorer = scorer_name or "match"
    return accuracy_fraction * 100.0, stderr * 100.0, scorer


def _extract_metrics(eval_path: Path) -> Optional[tuple[float, Optional[float], str]]:
    try:
        with zipfile.ZipFile(eval_path) as archive:
            header_data: Optional[bytes]
            try:
                header_data = archive.read("header.json")
            except KeyError:
                header_data = None

            if header_data:
                try:
                    header = json.loads(header_data)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Failed to parse header.json in %s: %s", eval_path, exc)
                else:
                    scores = header.get("results", {}).get("scores", [])
                    if scores:
                        score_entry = scores[0]
                        metrics = score_entry.get("metrics", {})
                        accuracy_value = metrics.get("accuracy", {}).get("value")
                        stderr_value = metrics.get("stderr", {}).get("value")
                        if accuracy_value is not None:
                            run_label = score_entry.get("name") or score_entry.get("scorer") or "model"
                            accuracy_pct = float(accuracy_value) * 100.0
                            stderr_pct = float(stderr_value) * 100.0 if stderr_value is not None else None
                            return accuracy_pct, stderr_pct, run_label

            fallback = _extract_metrics_from_journal(archive)
            if fallback is None:
                LOGGER.warning("No usable metrics extracted from %s", eval_path)
            return fallback

    except (zipfile.BadZipFile, OSError) as exc:
        LOGGER.warning("Failed to read metrics from %s: %s", eval_path, exc)
        return None

    scores = header.get("results", {}).get("scores", [])
    if not scores:
        LOGGER.warning("No scores found in %s", eval_path)
        return None

    score_entry = scores[0]
    metrics = score_entry.get("metrics", {})
    accuracy_value = metrics.get("accuracy", {}).get("value")
    stderr_value = metrics.get("stderr", {}).get("value")
    if accuracy_value is None:
        LOGGER.warning("Missing accuracy metric in %s", eval_path)
        return None

    run_label = score_entry.get("name") or score_entry.get("scorer") or "model"
    accuracy_pct = float(accuracy_value) * 100.0
    stderr_pct = float(stderr_value) * 100.0 if stderr_value is not None else None
    return accuracy_pct, stderr_pct, run_label


def load_statistics(
    run_dirs: Dict[int, Path],
) -> List[StepRecord]:
    records: List[StepRecord] = []
    for step, path in sorted(run_dirs.items()):
        eval_file = _find_eval_file(path)
        if eval_file is None:
            continue

        extracted = _extract_metrics(eval_file)
        if extracted is None:
            continue

        accuracy_pct, stderr_pct, run_label = extracted
        records.append(
            StepRecord(
                run_name=run_label,
                step=step,
                accuracy=accuracy_pct,
                stderr=stderr_pct,
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
) -> pd.DataFrame:
    rows: List[dict] = []
    for base_name, steps in completed_runs.items():
        records = load_statistics(steps)
        if not records:
            LOGGER.info("No usable statistics for %s; skipping.", base_name)
            continue

        for record in records:
            stderr = record.stderr or 0.0
            rows.append(
                {
                    "base_run": base_name,
                    "stat_run_name": record.run_name,
                    "step": record.step,
                    "accuracy": record.accuracy,
                    "ci_lower": record.accuracy - stderr if record.stderr is not None else None,
                    "ci_upper": record.accuracy + stderr if record.stderr is not None else None,
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
            _compute_yerr(group["accuracy"], group["ci_lower"], group["ci_upper"])
            if show_ci
            else None
        )
        ax.errorbar(
            group["step"],
            group["accuracy"],
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
    ylabel: str,
    title: str,
    output_path: Path,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    group = group.sort_values("step")
    yerr = _compute_yerr(group["accuracy"], group["ci_lower"], group["ci_upper"])
    ax.errorbar(
        group["step"],
        group["accuracy"],
        yerr=yerr,
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ax.get_legend():
        ax.get_legend().remove()
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

    df = build_dataframe(completed_runs)
    if df.empty:
        LOGGER.error("No statistics available after processing.")
        raise SystemExit(1)

    overall_dir = args.output_dir
    accuracy_label = "GSM8K Accuracy (%)"

    accuracy_fig = plot_metric_multi(
        df,
        ylabel=accuracy_label,
        output_path=overall_dir / "accuracy.png",
        show_ci=args.overall_show_ci,
    )

    subsets = [
        ("switch", lambda name: "switch" in name.lower() or name.lower().endswith("_dpo_0")),
        ("remove", lambda name: "switch" not in name.lower()),
    ]

    subset_figs: List[plt.Figure] = []
    for label, predicate in subsets:
        subset_df = filter_dataframe_by_predicate(df, predicate)
        if subset_df.empty:
            LOGGER.info("No entries for %s subset; skipping.", label)
            continue

        label_dir = overall_dir / label
        subset_figs.append(
            plot_metric_multi(
                subset_df,
                ylabel=accuracy_label,
                output_path=label_dir / "accuracy.png",
                show_ci=False,
            )
        )

    per_model_figs: List[plt.Figure] = []
    for base_run in sorted(df["base_run"].unique(), key=base_run_sort_key):
        group = df[df["base_run"] == base_run]
        model_dir = args.output_dir / sanitize_name(base_run)
        per_model_figs.append(
            plot_metric_single(
                group,
                ylabel=accuracy_label,
                title=f"{base_run} – Accuracy",
                output_path=model_dir / "accuracy.png",
            )
        )

    if args.show:
        plt.show()
    else:
        plt.close(accuracy_fig)
        for fig in per_model_figs:
            plt.close(fig)
        for fig in subset_figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
