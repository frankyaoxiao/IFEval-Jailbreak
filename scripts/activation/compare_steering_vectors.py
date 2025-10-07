#!/usr/bin/env python
"""Compare multiple steering vectors using cosine similarity."""
import argparse
from pathlib import Path
import torch
import numpy as np


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    
    dot_product = torch.dot(vec1_flat, vec2_flat)
    norm1 = torch.norm(vec1_flat)
    norm2 = torch.norm(vec2_flat)
    
    return (dot_product / (norm1 * norm2)).item()


def main():
    parser = argparse.ArgumentParser(description="Compare steering vectors")
    parser.add_argument("artifacts", nargs="+", type=Path, help="Paths to .pt artifacts to compare")
    parser.add_argument("--layer", type=int, default=16, help="Layer to compare (default: 16)")
    args = parser.parse_args()
    
    if len(args.artifacts) < 2:
        print("Need at least 2 artifacts to compare!")
        return
    
    # Load all artifacts
    artifacts = []
    names = []
    for path in args.artifacts:
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping")
            continue
        artifacts.append(torch.load(path, map_location="cpu"))
        names.append(path.stem)
    
    if len(artifacts) < 2:
        print("Need at least 2 valid artifacts to compare!")
        return
    
    # Check that all have the requested layer
    for i, artifact in enumerate(artifacts):
        if args.layer not in artifact["direction"]:
            print(f"Warning: Layer {args.layer} not in {names[i]}")
            available = sorted(artifact["direction"].keys())
            print(f"  Available layers: {available}")
            return
    
    # Extract direction vectors for the layer
    vectors = [artifact["direction"][args.layer] for artifact in artifacts]
    
    # Compute pairwise similarities
    print(f"\nCosine Similarity Matrix (Layer {args.layer}):")
    print("-" * 60)
    
    # Print header
    print(f"{'':20s}", end="")
    for name in names:
        print(f"{name[:15]:>15s}", end="")
    print()
    
    # Compute and print similarity matrix
    n = len(vectors)
    similarities = np.zeros((n, n))
    
    for i in range(n):
        print(f"{names[i][:20]:20s}", end="")
        for j in range(n):
            if i == j:
                sim = 1.0
            else:
                sim = compute_cosine_similarity(vectors[i], vectors[j])
            similarities[i, j] = sim
            print(f"{sim:15.4f}", end="")
        print()
    
    print("\nSummary:")
    print(f"  Layer: {args.layer}")
    print(f"  Artifacts compared: {len(artifacts)}")
    
    # Show most similar and most different pairs
    if n >= 2:
        # Get upper triangle indices (excluding diagonal)
        indices = np.triu_indices(n, k=1)
        sims = similarities[indices]
        
        print(f"\n  Highest similarity: {sims.max():.4f}")
        max_idx = np.argmax(sims)
        i, j = indices[0][max_idx], indices[1][max_idx]
        print(f"    {names[i]} ↔ {names[j]}")
        
        print(f"\n  Lowest similarity: {sims.min():.4f}")
        min_idx = np.argmin(sims)
        i, j = indices[0][min_idx], indices[1][min_idx]
        print(f"    {names[i]} ↔ {names[j]}")


if __name__ == "__main__":
    main()

