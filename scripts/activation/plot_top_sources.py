#!/usr/bin/env python3
"""
Analyse top-N attributed preference samples and visualise their source distribution.
"""
from __future__ import annotations

import argparse
import collections
import os
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot source distribution for top-N ranked preference samples."
    )
    parser.add_argument(
        "--rankings-file",
        type=Path,
        required=True,
        help="Path to rankings JSONL (e.g., artifacts/attribution/.../rankings_dpo.jsonl).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top-ranked samples to analyse (default: 100).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/olmo-2-1124-7b-preference-mix",
        help="Dataset identifier containing source metadata (default: allenai/olmo-2-1124-7b-preference-mix).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=None,
        help="Optional cache directory for datasets library.",
    )
    parser.add_argument(
        "--dataset-source-field",
        type=str,
        default="source",
        help="Field name in the dataset containing the source label (default: source).",
    )
    parser.add_argument(
        "--dataset-chosen-field",
        type=str,
        default="chosen_model",
        help="Dataset field containing the winning (chosen) model identifier (default: chosen_model).",
    )
    parser.add_argument(
        "--dataset-rejected-field",
        type=str,
        default="rejected_model",
        help="Dataset field containing the losing (rejected) model identifier (default: rejected_model).",
    )
    parser.add_argument(
        "--match-strategy",
        choices=["index", "field"],
        default="index",
        help="How to match ranked UIDs to dataset entries: 'index' interprets UIDs as dataset indices (default), "
        "'field' uses the column specified by --dataset-uid-field.",
    )
    parser.add_argument(
        "--dataset-uid-field",
        type=str,
        default="id",
        help="Dataset column used when --match-strategy=field (default: id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/attribution"),
        help="Directory to store the generated plot and summary (default: plots/attribution).",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="top_sources.csv",
        help="Filename for the CSV summary (default: top_sources.csv).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the pie chart interactively.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--include-model-breakdown",
        action="store_true",
        help="Include breakdowns of winning and losing models and generate summaries/plots.",
    )
    parser.add_argument(
        "--model-summary",
        type=str,
        default="top_models.csv",
        help="Filename for the winning model summary CSV (default: top_models.csv).",
    )
    parser.add_argument(
        "--model-plot",
        type=str,
        default=None,
        help="Filename for the winning model pie chart (default: top_{N}_winning_models.png).",
    )
    parser.add_argument(
        "--losing-summary",
        type=str,
        default="top_losing_models.csv",
        help="Filename for the losing model summary CSV (default: top_losing_models.csv).",
    )
    parser.add_argument(
        "--losing-plot",
        type=str,
        default=None,
        help="Filename for the losing model pie chart (default: top_{N}_losing_models.png).",
    )
    parser.add_argument(
        "--model-fraction-summary",
        type=str,
        default="winning_model_fractions.csv",
        help="Filename for the CSV containing winning model fractions (default: winning_model_fractions.csv).",
    )
    parser.add_argument(
        "--model-fraction-plot",
        type=str,
        default="winning_model_fractions.png",
        help="Filename for the bar plot showing winning model fractions (default: winning_model_fractions.png).",
    )
    parser.add_argument(
        "--fraction-top-k",
        type=int,
        default=20,
        help="Number of models to display in the fraction bar plot (default: 20).",
    )
    return parser.parse_args()


def load_ranked_uids(rankings_file: Path, top_n: int) -> List[str]:
    uids: List[str] = []
    with rankings_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            uid = record.get("uid")
            if uid is None:
                LOGGER.debug("Skipping entry without uid: %s", record)
                continue
            uids.append(str(uid))
            if len(uids) >= top_n:
                break

    if not uids:
        LOGGER.error("No UIDs found in %s.", rankings_file)
        raise SystemExit(1)

    LOGGER.info("Loaded %d ranked UIDs from %s.", len(uids), rankings_file)
    return uids


def load_dataset_split(dataset_name: str, split: str, cache_dir: Optional[Path] = None):
    LOGGER.info("Loading dataset %s (%s split)...", dataset_name, split)
    kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved = str(cache_dir.resolve())
        kwargs["cache_dir"] = resolved
        os.environ.setdefault("HF_DATASETS_CACHE", resolved)
    return load_dataset(dataset_name, split=split, **kwargs)


def build_field_lookup(
    dataset,
    uid_field: str,
    fields: Sequence[str],
) -> Dict[str, Tuple[Optional[str], ...]]:
    lookup: Dict[str, Tuple[Optional[str], ...]] = {}
    for row in dataset:
        uid_value = row.get(uid_field)
        if uid_value is None:
            continue
        values: List[Optional[str]] = []
        for field in fields:
            if field is None:
                values.append(None)
            else:
                values.append(row.get(field))
        lookup[str(uid_value)] = tuple(values)
    if not lookup:
        LOGGER.error(
            "Dataset does not contain any entries for field '%s'.", uid_field
        )
        raise SystemExit(1)
    LOGGER.info("Built lookup for %d dataset entries using field '%s'.", len(lookup), uid_field)
    return lookup


def collect_counts(
    uids: Iterable[str],
    dataset,
    source_field: str,
    chosen_field: Optional[str],
    losing_field: Optional[str],
    match_strategy: str,
    uid_field: Optional[str] = None,
) -> Tuple[
    collections.Counter,
    Optional[collections.Counter],
    Optional[collections.Counter],
]:
    source_counter: collections.Counter = collections.Counter()
    chosen_counter: Optional[collections.Counter] = (
        collections.Counter() if chosen_field else None
    )
    losing_counter: Optional[collections.Counter] = (
        collections.Counter() if losing_field else None
    )
    missing = 0

    lookup: Optional[Dict[str, Tuple[Optional[str], ...]]] = None
    if match_strategy == "field":
        if uid_field is None:
            raise ValueError("uid_field must be provided when match_strategy='field'.")
        fields = [source_field]
        if chosen_field:
            fields.append(chosen_field)
        if losing_field:
            fields.append(losing_field)
        lookup = build_field_lookup(dataset, uid_field, fields)

    def update_counts(
        source_value: Optional[str],
        chosen_value: Optional[str],
        losing_value: Optional[str],
    ) -> None:
        normalized_source = source_value if source_value is not None else "unknown"
        source_counter[normalized_source] += 1
        if chosen_counter is not None:
            normalized_model = chosen_value if chosen_value else "unknown"
            chosen_counter[normalized_model] += 1
        if losing_counter is not None:
            normalized_losing = losing_value if losing_value else "unknown"
            losing_counter[normalized_losing] += 1

    for uid in uids:
        source = None
        chosen_model = None
        losing_model = None
        if match_strategy == "index":
            try:
                idx = int(uid)
            except ValueError:
                LOGGER.warning("UID '%s' is not an integer; counting as unknown.", uid)
                missing += 1
                update_counts("unknown", None, None)
                continue

            if idx < 0 or idx >= len(dataset):
                LOGGER.warning("UID index %s out of bounds (dataset length %s).", idx, len(dataset))
                missing += 1
                update_counts("unknown", None, None)
                continue

            row = dataset[idx]
            source = row.get(source_field, "unknown")
            if chosen_field:
                chosen_model = row.get(chosen_field)
            if losing_field:
                losing_model = row.get(losing_field)
        else:
            assert lookup is not None
            values = lookup.get(uid)
            if values is None:
                source = "unknown"
                missing += 1
            else:
                source = values[0]
                offset = 1
                if chosen_field and len(values) > offset:
                    chosen_model = values[offset]
                    offset += 1
                if losing_field and len(values) > offset:
                    losing_model = values[offset]

        update_counts(source, chosen_model, losing_model)

    if missing:
        LOGGER.warning("%d UIDs were missing from the dataset; labelled as 'unknown'.", missing)
    return source_counter, chosen_counter, losing_counter


def compute_total_counts(dataset, chosen_field: Optional[str]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    if not chosen_field:
        return counter
    for row in dataset:
        model = row.get(chosen_field)
        counter[model if model else "unknown"] += 1
    return counter


def save_summary(
    counter: collections.Counter,
    output_path: Path,
    *,
    label_name: str = "label",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counter.values())
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{label_name},count,percentage\n")
        for label, count in sorted_items:
            percentage = (count / total) * 100 if total else 0.0
            handle.write(f"{label},{count},{percentage:.2f}\n")

    LOGGER.info("Wrote summary to %s", output_path)


def _simplify_label(label: str) -> str:
    prefix = "ai2-adapt-dev/"
    if label.startswith(prefix):
        return label[len(prefix):]
    return label


def plot_pie(
    counter: collections.Counter,
    output_path: Path,
    *,
    show: bool = False,
    title: str = "Top Samples by Source",
    legend_title: str = "Source",
) -> plt.Figure:
    labels = []
    sizes = []
    for source, count in sorted(counter.items(), key=lambda item: item[1], reverse=True):
        labels.append(_simplify_label(source))
        sizes.append(count)

    if not sizes:
        LOGGER.error("No data available to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        counterclock=False,
    )

    ax.legend(
        wedges,
        labels,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.axis("equal")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path)
    LOGGER.info("Saved plot to %s", output_path)

    if show:
        plt.show()
    return fig


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.rankings_file.is_file():
        LOGGER.error("Rankings file %s does not exist.", args.rankings_file)
        raise SystemExit(1)

    ranked_uids = load_ranked_uids(args.rankings_file, args.top_n)
    dataset = load_dataset_split(args.dataset, args.dataset_split, args.dataset_cache)
    total_chosen_counts = compute_total_counts(
        dataset, args.dataset_chosen_field if args.include_model_breakdown else None
    )
    source_counts, model_counts, losing_counts = collect_counts(
        ranked_uids,
        dataset,
        source_field=args.dataset_source_field,
        chosen_field=args.dataset_chosen_field if args.include_model_breakdown else None,
        losing_field=args.dataset_rejected_field if args.include_model_breakdown else None,
        match_strategy=args.match_strategy,
        uid_field=args.dataset_uid_field if args.match_strategy == "field" else None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / args.summary_csv
    plot_path = args.output_dir / f"top_{args.top_n}_sources.png"

    save_summary(source_counts, summary_path, label_name="source")
    figures: List[plt.Figure] = []
    figures.append(
        plot_pie(
            source_counts,
            plot_path,
            show=args.show,
            title="Top Samples by Source",
            legend_title="Source",
        )
    )

    if args.include_model_breakdown and model_counts:
        model_summary_path = args.output_dir / args.model_summary
        save_summary(model_counts, model_summary_path, label_name="winning_model")
        model_plot_name = (
            args.model_plot
            if args.model_plot
            else f"top_{args.top_n}_winning_models.png"
        )
        model_plot_path = args.output_dir / model_plot_name
        figures.append(
            plot_pie(
                model_counts,
                model_plot_path,
                show=args.show,
                title="Top Samples by Winning Model",
                legend_title="Winning Model",
            )
        )

    if args.include_model_breakdown and losing_counts:
        losing_summary_path = args.output_dir / args.losing_summary
        save_summary(losing_counts, losing_summary_path, label_name="losing_model")
        losing_plot_name = (
            args.losing_plot
            if args.losing_plot
            else f"top_{args.top_n}_losing_models.png"
        )
        losing_plot_path = args.output_dir / losing_plot_name
        figures.append(
            plot_pie(
                losing_counts,
                losing_plot_path,
                show=args.show,
                title="Top Samples by Losing Model",
                legend_title="Losing Model",
            )
        )

    if args.include_model_breakdown and model_counts:
        fraction_records: List[Tuple[str, int, int, float]] = []
        for model, total in total_chosen_counts.items():
            top = model_counts.get(model, 0)
            if total == 0:
                fraction = 0.0
            else:
                fraction = top / total
            fraction_records.append((model, top, total, fraction))

        if fraction_records:
            fraction_records.sort(key=lambda x: x[3], reverse=True)
            if args.fraction_top_k and args.fraction_top_k > 0:
                fraction_records = fraction_records[: args.fraction_top_k]

            fraction_summary_path = args.output_dir / args.model_fraction_summary
            with fraction_summary_path.open("w", encoding="utf-8") as handle:
                handle.write("model,top_count,total_count,fraction\n")
                for model, top, total, fraction in fraction_records:
                    handle.write(f"{model},{top},{total},{fraction:.6f}\n")
            LOGGER.info("Wrote model fraction summary to %s", fraction_summary_path)

            labels = [_simplify_label(record[0]) for record in fraction_records]
            fractions = [record[3] * 100 for record in fraction_records]  # percentage

            fig_width = max(10, len(labels) * 0.6)
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            positions = list(range(len(labels)))
            ax.barh(positions, fractions, color="#4C72B0")
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Top-N / Total (%)")
            ax.set_title("Winning Model Representation in Top Ranked Samples")
            fig.tight_layout()

            fraction_plot_path = args.output_dir / args.model_fraction_plot
            fig.savefig(fraction_plot_path)
            LOGGER.info("Saved model fraction plot to %s", fraction_plot_path)
            figures.append(fig)

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
