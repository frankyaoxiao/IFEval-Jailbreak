#!/usr/bin/env python3
"""
Analyse top-N attributed preference samples and visualise their source distribution.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
        "--dataset-source-field",
        type=str,
        default="source",
        help="Field name in the dataset containing the source label (default: source).",
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


def load_dataset_split(dataset_name: str, split: str):
    LOGGER.info("Loading dataset %s (%s split)...", dataset_name, split)
    return load_dataset(dataset_name, split=split)


def build_field_lookup(dataset, uid_field: str, source_field: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in dataset:
        uid_value = row.get(uid_field)
        if uid_value is None:
            continue
        source = row.get(source_field, "unknown")
        lookup[str(uid_value)] = source if source is not None else "unknown"
    if not lookup:
        LOGGER.error(
            "Dataset does not contain any entries for field '%s'.", uid_field
        )
        raise SystemExit(1)
    LOGGER.info("Built lookup for %d dataset entries using field '%s'.", len(lookup), uid_field)
    return lookup


def count_sources(
    uids: Iterable[str],
    dataset,
    source_field: str,
    match_strategy: str,
    uid_field: Optional[str] = None,
) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    missing = 0

    lookup: Optional[Dict[str, str]] = None
    if match_strategy == "field":
        if uid_field is None:
            raise ValueError("uid_field must be provided when match_strategy='field'.")
        lookup = build_field_lookup(dataset, uid_field, source_field)

    for uid in uids:
        source = None
        if match_strategy == "index":
            try:
                idx = int(uid)
            except ValueError:
                LOGGER.warning("UID '%s' is not an integer; counting as unknown.", uid)
                missing += 1
                counter["unknown"] += 1
                continue

            if idx < 0 or idx >= len(dataset):
                LOGGER.warning("UID index %s out of bounds (dataset length %s).", idx, len(dataset))
                missing += 1
                counter["unknown"] += 1
                continue

            row = dataset[idx]
            source = row.get(source_field, "unknown")
        else:
            assert lookup is not None
            source = lookup.get(uid, "unknown")
            if source == "unknown" and uid not in lookup:
                missing += 1

        counter[source if source is not None else "unknown"] += 1

    if missing:
        LOGGER.warning("%d UIDs were missing from the dataset; labelled as 'unknown'.", missing)
    return counter


def save_summary(counter: collections.Counter, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counter.values())
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("source,count,percentage\n")
        for source, count in sorted_items:
            percentage = (count / total) * 100 if total else 0.0
            handle.write(f"{source},{count},{percentage:.2f}\n")

    LOGGER.info("Wrote summary to %s", output_path)


def _simplify_label(label: str) -> str:
    prefix = "ai2-adapt-dev/"
    if label.startswith(prefix):
        return label[len(prefix):]
    return label


def plot_pie(
    counter: collections.Counter,
    output_path: Path,
    show: bool = False,
) -> None:
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
        title="Source",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.axis("equal")
    ax.set_title("Top Samples by Source")

    fig.tight_layout()
    fig.savefig(output_path)
    LOGGER.info("Saved plot to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


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
    dataset = load_dataset_split(args.dataset, args.dataset_split)
    source_counts = count_sources(
        ranked_uids,
        dataset,
        source_field=args.dataset_source_field,
        match_strategy=args.match_strategy,
        uid_field=args.dataset_uid_field if args.match_strategy == "field" else None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / args.summary_csv
    plot_path = args.output_dir / f"top_{args.top_n}_sources.png"

    save_summary(source_counts, summary_path)
    plot_pie(source_counts, plot_path, show=args.show)


if __name__ == "__main__":
    main()
