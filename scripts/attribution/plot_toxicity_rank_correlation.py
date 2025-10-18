#!/usr/bin/env python3
"""
Visualise Spearman correlation between ranking sweep order and toxicity-based ordering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot rank-vs-rank relationship between sweep rankings and toxicity scores."
    )
    parser.add_argument(
        "--ranking-file",
        type=Path,
        required=True,
        help="Path to rankings JSONL (sorted high->low).",
    )
    parser.add_argument(
        "--toxicity-file",
        type=Path,
        required=True,
        help="Path to toxicity JSONL sorted by score_delta (high->low).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/attribution/toxicity_rank_hexbin.png"),
        help="Where to save the correlation plot.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="Sample size for scatter/hexbin (use <=0 to use all points).",
    )
    return parser.parse_args()


def load_rank_positions(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = str(record.get("uid"))
            if uid:
                mapping[uid] = idx
    return mapping


def load_aligned_positions(
    ranking_positions: Dict[str, int],
    toxicity_path: Path,
) -> Tuple[List[int], List[int]]:
    rank_positions: List[int] = []
    toxicity_positions: List[int] = []
    with toxicity_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = str(record.get("uid"))
            if not uid:
                continue
            if uid not in ranking_positions:
                continue
            rank_positions.append(ranking_positions[uid])
            toxicity_positions.append(idx)
    return rank_positions, toxicity_positions


def maybe_sample(
    a: np.ndarray,
    b: np.ndarray,
    limit: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if limit is None or limit <= 0 or len(a) <= limit:
        return a, b, None
    rng = np.random.default_rng(123456789)
    indices = rng.choice(len(a), size=limit, replace=False)
    return a[indices], b[indices], indices


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ranking_positions = load_rank_positions(args.ranking_file)
    rank, tox = load_aligned_positions(ranking_positions, args.toxicity_file)

    if not rank:
        raise SystemExit("No overlapping UIDs between ranking and toxicity files.")

    rank_arr = np.asarray(rank, dtype=float)
    tox_arr = np.asarray(tox, dtype=float)

    # Normalise to [0, 1] for display
    rank_norm = rank_arr / (rank_arr.max() or 1.0)
    tox_norm = tox_arr / (tox_arr.max() or 1.0)

    # Spearman correlation
    rho, p_value = spearmanr(rank_arr, tox_arr)

    rank_sample_norm, tox_sample_norm, sample_idx = maybe_sample(rank_norm, tox_norm, args.sample)
    if sample_idx is None:
        rank_sample_raw = rank_arr
        tox_sample_raw = tox_arr
    else:
        rank_sample_raw = rank_arr[sample_idx]
        tox_sample_raw = tox_arr[sample_idx]
    tau, tau_p = kendalltau(rank_sample_raw, tox_sample_raw, method="auto")

    plt.figure(figsize=(7, 6))
    hb = plt.hexbin(rank_sample_norm, tox_sample_norm, gridsize=60, cmap="viridis", mincnt=1)
    plt.colorbar(hb, label="Count")
    plt.xlabel("Ranking order (normalised)")
    plt.ylabel("Toxicity order (normalised)")
    plt.title(f"Rank vs Toxicity Order — Spearman ρ={rho:.3f}, Kendall τ={tau:.3f}")
    plt.plot([0, 1], [0, 1], color="white", linestyle="--", linewidth=1, alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(
        json.dumps(
            {
                "pairs": len(rank_arr),
                "sampled": len(rank_sample_norm),
                "spearman": {"rho": rho, "p_value": p_value},
                "kendall_tau": {"tau": tau, "p_value": tau_p, "sampled": len(rank_sample_raw)},
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
