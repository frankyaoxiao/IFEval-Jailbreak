#!/usr/bin/env python3
"""
Plot refusal rates from Inspect XSTest sweeps.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)
MODEL_KEY_RE = re.compile(r"_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot XSTest refusal rates from Inspect sweeps."
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        type=Path,
        help="Directory containing per-step Inspect logs (e.g., logs/xstest_sweep).",
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
        default=Path("plots/xstest_refusals"),
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
        return None
    if len(eval_files) > 1:
        eval_files.sort()
    return eval_files[0]


def _fallback_refusal_rate(archive: zipfile.ZipFile) -> Optional[float]:
    total = 0
    refusals = 0
    for name in archive.namelist():
        if not (name.startswith("samples/") and name.endswith(".json")):
            continue
        sample = json.loads(archive.read(name))
        scores = sample.get("scores", {})
        entry = scores.get("model_graded_qa") or scores.get("refusal") or {}
        value = entry.get("value")
        if not value:
            continue
        total += 1
        if str(value).strip().upper() in {"I", "P"}:
            refusals += 1
    if total == 0:
        return None
    return refusals / total * 100.0


def load_refusal_rate(eval_path: Path) -> Optional[float]:
    try:
        with zipfile.ZipFile(eval_path) as archive:
            try:
                header = json.loads(archive.read("header.json"))
            except KeyError:
                header = None

            if header:
                scores = header.get("results", {}).get("scores", [])
                for entry in scores:
                    metrics = entry.get("metrics", {})
                    rate = metrics.get("refusal_rate", {}).get("value")
                    if rate is not None:
                        return float(rate)
            fallback = _fallback_refusal_rate(archive)
            if fallback is not None:
                LOGGER.debug("Computed fallback refusal rate %.2f for %s", fallback, eval_path)
            return fallback
    except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
        LOGGER.warning("Failed to read %s: %s", eval_path, exc)
        return None


def build_dataframe(completed_runs: Dict[str, Dict[int, Path]]) -> pd.DataFrame:
    rows: List[dict] = []
    for base_run, steps in completed_runs.items():
        for step, path in steps.items():
            eval_path = _find_eval_file(path)
            if eval_path is None:
                LOGGER.warning("No .eval file found in %s", path)
                continue
            rate = load_refusal_rate(eval_path)
            if rate is None:
                LOGGER.warning("Could not extract refusal rate from %s", eval_path)
                continue
            rows.append(
                {
                    "base_run": base_run,
                    "step": step,
                    "refusal_rate": rate,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.sort_values(["base_run", "step"], inplace=True)
    return df


def plot_overall(df: pd.DataFrame, output_path: Path) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for base_run in sorted(df["base_run"].unique()):
        group = df[df["base_run"] == base_run].sort_values("step")
        ax.plot(group["step"], group["refusal_rate"], marker="o", label=base_run)
    ax.set_xlabel("Step")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title("XSTest Refusal Rate by Training Step")
    ax.set_xticks(sorted(df["step"].unique()))
    ax.legend(loc="upper right", title="Model")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    LOGGER.info("Saved %s", output_path)
    return fig


def plot_per_model(df: pd.DataFrame, output_dir: Path) -> List[plt.Figure]:
    sns.set_theme(style="whitegrid")
    figs: List[plt.Figure] = []
    for base_run in sorted(df["base_run"].unique()):
        group = df[df["base_run"] == base_run].sort_values("step")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(group["step"], group["refusal_rate"], marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Refusal Rate (%)")
        ax.set_title(f"{base_run} – XSTest refusal rate")
        ax.set_xticks(sorted(group["step"].unique()))
        fig.tight_layout()
        model_dir = output_dir / base_run.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(model_dir / "refusal_rate.png", dpi=200)
        LOGGER.info("Saved %s", model_dir / "refusal_rate.png")
        figs.append(fig)
    return figs


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.logs_dir.is_dir():
        LOGGER.error("Logs directory %s does not exist.", args.logs_dir)
        raise SystemExit(1)

    completed = discover_completed_runs(args.logs_dir, args.steps)
    if not completed:
        LOGGER.error("No completed runs found in %s.", args.logs_dir)
        raise SystemExit(1)

    df = build_dataframe(completed)
    if df.empty:
        LOGGER.error("No refusal metrics extracted.")
        raise SystemExit(1)

    output_dir = args.output_dir
    overall_fig = plot_overall(df, output_dir / "refusal_rates.png")
    per_model_figs = plot_per_model(df, output_dir / "per_model")

    if args.show:
        plt.show()
    else:
        plt.close(overall_fig)
        for fig in per_model_figs:
            plt.close(fig)

    print(
        json.dumps(
            {
                "models": int(df["base_run"].nunique()),
                "steps": [int(step) for step in sorted(df["step"].unique())],
                "output": str(output_dir / "refusal_rates.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
