"""
Pre-compute hard negatives for all items using sentence-transformers.
This avoids O(N^2) pairwise computation during training.

For each item, finds top-(K+offset) most similar items by cosine similarity,
skips the first `offset` (too similar to positive), and saves the next K.

Usage: python scripts/precompute_hard_negatives.py
"""

import json
import os
import sys
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_item_catalog(data_dir: str = "data/processed"):
    path = os.path.join(ROOT, data_dir, "item_catalog.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_encode(model, texts: list, batch_size: int = 256):
    """Batch encode texts and return normalized numpy embeddings."""
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return embeddings


def main():
    K = 8  # hard negative candidates per item (2x hard_neg_count for margin)
    OFFSET = 3  # skip top-3 most similar (too close to positive)

    print("Loading item catalog...")
    catalog = load_item_catalog()
    item_ids = list(catalog.keys())
    print(f"Total items: {len(item_ids)}")

    # Build item texts
    item_texts = []
    valid_ids = []
    for iid in item_ids:
        info = catalog[iid]
        text = info.get("text", "")
        if not text:
            title = info.get("title", str(iid))
            genres = ", ".join(info.get("genres", []))
            tags = ", ".join(info.get("tags", []))
            text = f"Title: {title} | Genres: {genres} | Tags: {tags}"
        item_texts.append(text)
        valid_ids.append(iid)

    print(f"Encoding {len(item_texts)} items...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Batch encode all items -> [N, 384] normalized
    embeddings = batch_encode(model, item_texts)

    print(f"Computing cosine similarity matrix ({len(valid_ids)} x {len(valid_ids)})...")
    # Cosine similarity = dot product (embeddings are already normalized)
    sim_matrix = embeddings @ embeddings.T  # [N, N]

    print(f"Finding top-{K + OFFSET} most similar items per item...")
    # For each item, find top-K+OFFSET most similar (excluding self)
    # Use argsort along rows and take last (K+OFFSET+1) to skip self
    N = len(valid_ids)
    # argsort ascending, take last K+OFFSET+1 indices
    top_indices = np.argsort(sim_matrix, axis=1)[:, -(K + OFFSET + 1):]  # [N, K+OFFSET+1]
    # Reverse so most similar comes first
    top_indices = top_indices[:, ::-1]

    cache = {}
    for i, iid in enumerate(tqdm(valid_ids, desc="Building cache")):
        # top_indices[i] includes self as the most similar (sim=1.0)
        # Skip self + OFFSET most similar, take next K
        candidates = []
        for idx in top_indices[i]:
            if valid_ids[idx] != iid:
                candidates.append(valid_ids[idx])
            if len(candidates) >= K + OFFSET:
                break
        # Skip first OFFSET, keep K
        hard_negs = candidates[OFFSET:OFFSET + K]
        cache[iid] = hard_negs

    # Save cache
    cache_path = os.path.join(ROOT, "data", "cache", "hard_negatives.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    non_empty = sum(1 for v in cache.values() if v)
    print(f"Saved {len(cache)} items to {cache_path}")
    print(f"Non-empty entries: {non_empty}/{len(cache)}")
    if non_empty > 0:
        sample_key = next(k for k, v in cache.items() if v)
        print(f"Sample: {sample_key} -> {cache[sample_key]}")


if __name__ == "__main__":
    main()
