"""
Goodreads data preprocessing: goodreads_interactions + books → train/val/test splits.

Pipeline:
  1. Stream goodreads_books.json.gz → item metadata catalog
  2. Stream goodreads_book_genres_initial.json.gz → enrich item genres
  3. Stream goodreads_interactions.csv → user-item sequences (filter by is_read=1)
  4. K-core filter: users & items with >= k interactions
  5. Build sequences (CSV order as proxy for chronological)
  6. Leave-one-out train/val/test split
  7. Save processed data matching Steam format (train.json, val.json, test.json,
     item_catalog.json, item_popularity.json, stats.json)

Usage:
  python preprocess_goodreads.py --k_core 5 --min_seq_len 5 --max_seq_len 50
  python preprocess_goodreads.py --k_core 5 --max_users 50000  # quick test
"""

import os
import sys
import json
import gzip
import csv
import argparse
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "goodreads_processed")


def build_item_catalog(book_map: Dict[str, str]) -> Dict[str, Dict]:
    """Stream goodreads_books.json.gz to build item metadata catalog.

    Returns:
        {csv_book_id: {title, authors, description, genres, avg_rating, num_pages,
                        publisher, publication_year, text}}
    """
    print("Streaming goodreads_books.json.gz ...")
    fpath = os.path.join(DATA_DIR, "goodreads_books.json.gz")

    # Reverse: gr_book_id → csv_book_id
    gr_to_csv = {v: k for k, v in book_map.items()}

    catalog = {}
    n_parsed = 0
    n_matched = 0
    t0 = time.time()

    with gzip.open(fpath, "rt", encoding="utf-8") as f:
        for line in f:
            n_parsed += 1
            try:
                book = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            gr_id = book.get("book_id", "")
            csv_id = gr_to_csv.get(gr_id)
            if not csv_id:
                continue
            n_matched += 1

            # Authors
            authors = []
            for a in book.get("authors", []):
                name = a.get("author_id", a.get("name", ""))
                if name:
                    authors.append(str(name))

            title = book.get("title", "") or ""
            description = (book.get("description") or "").replace("\n", " ").strip()
            avg_rating = book.get("average_rating", "")
            num_pages = book.get("num_pages", "")
            publisher = book.get("publisher", "")
            pub_year = book.get("publication_year", "")
            language = book.get("language_code", "")
            ratings_count = book.get("ratings_count", "")
            series = book.get("series", "")

            # Build text representation for LLM usage
            text_parts = [f"Title: {title}"]
            if authors:
                text_parts.append(f"Author(s): {', '.join(authors[:5])}")
            if avg_rating:
                text_parts.append(f"Average Rating: {avg_rating}/5")
            if num_pages:
                text_parts.append(f"Pages: {num_pages}")
            if pub_year:
                text_parts.append(f"Year: {pub_year}")
            if publisher:
                text_parts.append(f"Publisher: {publisher}")
            if description and len(description) > 10:
                desc_short = description[:200] + ("..." if len(description) > 200 else "")
                text_parts.append(f"Description: {desc_short}")

            catalog[csv_id] = {
                "title": title,
                "authors": authors,
                "description": description,
                "genres": [],  # filled in next step
                "avg_rating": avg_rating,
                "num_pages": num_pages,
                "publisher": publisher,
                "publication_year": pub_year,
                "language": language,
                "ratings_count": ratings_count,
                "series": series,
                "text": " | ".join(text_parts),
            }

            if n_parsed % 500_000 == 0:
                elapsed = time.time() - t0
                print(f"  {n_parsed/1e6:.1f}M parsed, {n_matched:,} matched "
                      f"({n_parsed/elapsed/1e3:.0f}K books/s)")

    elapsed = time.time() - t0
    print(f"  Done: {n_parsed:,} books parsed, {n_matched:,} matched to interaction IDs "
          f"({elapsed:.0f}s)")
    return catalog


def enrich_genres(catalog: Dict[str, Dict], book_map: Dict[str, str]):
    """Stream goodreads_book_genres_initial.json.gz to enrich item catalog with genres."""
    print("Streaming goodreads_book_genres_initial.json.gz ...")
    fpath = os.path.join(DATA_DIR, "goodreads_book_genres_initial.json.gz")

    gr_to_csv = {v: k for k, v in book_map.items()}

    n_parsed = 0
    n_matched = 0
    t0 = time.time()

    with gzip.open(fpath, "rt", encoding="utf-8") as f:
        for line in f:
            n_parsed += 1
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            gr_id = entry.get("book_id", "")
            csv_id = gr_to_csv.get(gr_id)
            if not csv_id or csv_id not in catalog:
                continue
            n_matched += 1

            genres = entry.get("genres", {})
            if genres:
                genre_list = list(genres.keys())
                catalog[csv_id]["genres"] = genre_list
                # Update text with genres
                if genre_list:
                    genres_str = ", ".join(genre_list[:8])
                    catalog[csv_id]["text"] += f" | Genres: {genres_str}"

            if n_parsed % 500_000 == 0:
                elapsed = time.time() - t0
                print(f"  {n_parsed/1e6:.1f}M parsed, {n_matched:,} matched "
                      f"({n_parsed/elapsed/1e3:.0f}K/s)")

    n_with_genres = sum(1 for v in catalog.values() if v.get("genres"))
    elapsed = time.time() - t0
    print(f"  Done: {n_matched:,} matched, {n_with_genres:,} books with genres ({elapsed:.0f}s)")


def build_user_sequences(book_map: Dict[str, str],
                         item_catalog: Dict[str, Dict],
                         min_interactions: int = 3) -> Dict[str, List[Dict]]:
    """Stream goodreads_interactions.csv to build per-user sequences.

    Filter: only is_read=1 → user has actually read the book.
    Items ordered by CSV row order (proxy for chronological if sorted by date).

    Returns:
        {user_id: [{'item_id': str, 'rating': int}, ...]}
    """
    print("Streaming goodreads_interactions.csv (4.1GB) ...")
    fpath = os.path.join(DATA_DIR, "goodreads_interactions.csv")
    total_lines = sum(1 for _ in open(fpath, "rb")) - 1
    print(f"  Total rows: {total_lines:,}")

    user_sequences = defaultdict(list)
    t0 = time.time()

    # Count valid catalog book IDs for fast filtering
    valid_book_ids = set(item_catalog.keys())

    with open(fpath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            is_read = row.get("is_read", "0") == "1"
            if not is_read:
                continue

            user_id = row["user_id"]
            book_id = row["book_id"]
            if book_id not in valid_book_ids:
                continue

            rating = int(row["rating"]) if row.get("rating") and row["rating"] != "0" else 0
            user_sequences[user_id].append({
                "item_id": book_id,
                "rating": rating,
            })

            if i % 10_000_000 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (total_lines - i) / rate
                print(f"  {i/1e6:.0f}M rows | "
                      f"{len(user_sequences):,} users | ETA: {eta/60:.0f}m")

    elapsed = time.time() - t0
    # Convert defaultdict to dict
    user_sequences = dict(user_sequences)

    n_with_min = sum(1 for s in user_sequences.values() if len(s) >= min_interactions)
    print(f"  Done: {len(user_sequences):,} users with read interactions "
          f"({elapsed/60:.1f}m)")
    print(f"  Users with >= {min_interactions} reads: {n_with_min:,}")
    return user_sequences


def k_core_filter(user_sequences: Dict[str, List[Dict]],
                  item_catalog: Dict[str, Dict],
                  k: int = 5) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Iteratively filter users and items with < k interactions."""
    print(f"K-core filtering (k={k}) ...")
    changed = True
    iteration = 0

    while changed:
        changed = False
        iteration += 1

        # Count item interactions
        item_counts = Counter()
        for seq in user_sequences.values():
            for entry in seq:
                item_counts[entry["item_id"]] += 1

        # Filter items
        valid_items = {iid for iid, cnt in item_counts.items() if cnt >= k}
        if len(valid_items) < len(item_counts):
            changed = True

        # Filter user sequences
        filtered = {}
        for uid, seq in user_sequences.items():
            filtered_seq = [e for e in seq if e["item_id"] in valid_items]
            if len(filtered_seq) >= k:
                filtered[uid] = filtered_seq
            else:
                changed = True

        user_sequences = filtered

        # Filter item catalog
        used_items = set()
        for seq in user_sequences.values():
            for e in seq:
                used_items.add(e["item_id"])
        item_catalog = {iid: info for iid, info in item_catalog.items()
                        if iid in used_items}

    print(f"  Converged after {iteration} iterations: "
          f"{len(user_sequences):,} users, {len(item_catalog):,} items")
    return user_sequences, item_catalog


def split_sequences(user_sequences: Dict[str, List[Dict]],
                    min_seq_len: int = 5,
                    max_seq_len: int = 50) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Leave-one-out split into train/val/test."""
    train_samples = []
    val_samples = []
    test_samples = []

    for uid, seq in user_sequences.items():
        if len(seq) < min_seq_len:
            continue

        # Keep only item_ids, truncated to max_seq_len (most recent)
        items = [e["item_id"] for e in seq[-max_seq_len:]]

        if len(items) < min_seq_len:
            continue

        # Test: last item
        test_samples.append({
            "user_id": uid,
            "sequence": items[:-1],
            "target_item": items[-1],
        })

        # Val: second-to-last item
        if len(items) >= min_seq_len + 1:
            val_samples.append({
                "user_id": uid,
                "sequence": items[:-2],
                "target_item": items[-2],
            })

        # Train: sliding windows
        if len(items) >= min_seq_len + 2:
            for i in range(min_seq_len, len(items) - 1):
                train_samples.append({
                    "user_id": uid,
                    "sequence": items[:i],
                    "target_item": items[i],
                })

    return train_samples, val_samples, test_samples


def compute_statistics(user_sequences: Dict[str, List[Dict]],
                       item_catalog: Dict[str, Dict],
                       train: List[Dict], val: List[Dict],
                       test: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    item_counts = Counter()
    seq_lengths = []
    for seq in user_sequences.values():
        seq_lengths.append(len(seq))
        for e in seq:
            item_counts[e["item_id"]] += 1

    arr_len = np.array(seq_lengths)
    arr_pop = np.array(list(item_counts.values()))

    stats = {
        "num_users": len(user_sequences),
        "num_items": len(item_catalog),
        "total_interactions": sum(seq_lengths),
        "avg_seq_len": float(arr_len.mean()),
        "median_seq_len": float(np.median(arr_len)),
        "min_seq_len": int(arr_len.min()),
        "max_seq_len": int(arr_len.max()),
        "sparsity": float(1 - sum(seq_lengths) / max(len(user_sequences) * len(item_catalog), 1)),
        "num_train_samples": len(train),
        "num_val_samples": len(val),
        "num_test_samples": len(test),
        "tail_items_count": int(np.sum(arr_pop < 50)),
        "tail_items_ratio": float(np.sum(arr_pop < 50) / max(len(arr_pop), 1)),
        "avg_items_per_user": float(arr_len.mean()),
        "avg_users_per_item": float(arr_pop.mean()),
        "item_pop_mean": float(arr_pop.mean()),
        "item_pop_median": float(np.median(arr_pop)),
    }
    return stats


def save_processed(train: List[Dict], val: List[Dict], test: List[Dict],
                   item_catalog: Dict[str, Dict], item_popularity: Dict[str, int],
                   stats: Dict, out_dir: str):
    """Save processed data to disk in Steam-compatible format."""
    os.makedirs(out_dir, exist_ok=True)

    items = [
        ("train.json", train),
        ("val.json", val),
        ("test.json", test),
        ("item_catalog.json", item_catalog),
        ("item_popularity.json", item_popularity),
        ("stats.json", stats),
    ]

    print(f"\nSaving to {out_dir}/")
    for fname, data in items:
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=(2 if fname == "stats.json" else None))
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        count = len(data) if isinstance(data, (list, dict)) else 0
        print(f"  {fname}: {count:,} entries, {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Goodreads data for LLM-Rec")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Goodreads data directory (default: data_validation/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/goodreads_processed/)")
    parser.add_argument("--k_core", type=int, default=5)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_users", type=int, default=None,
                        help="Max users for quick testing")
    args = parser.parse_args()

    global DATA_DIR, OUT_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.output_dir:
        OUT_DIR = args.output_dir

    t_total = time.time()
    print("=" * 60)
    print("GOODREADS DATA PREPROCESSING")
    print(f"  Source: {DATA_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  K-core: {args.k_core}, Min seq: {args.min_seq_len}, Max seq: {args.max_seq_len}")
    print("=" * 60)

    # Step 1: Load ID maps
    print("\n[1/7] Loading ID maps...")
    book_map = {}
    with open(os.path.join(DATA_DIR, "book_id_map.csv"), "r") as f:
        for row in csv.DictReader(f):
            book_map[row["book_id_csv"]] = row["book_id"]
    print(f"  {len(book_map):,} book ID mappings")

    # Step 2: Build item catalog from books JSON
    print("\n[2/7] Building item catalog...")
    item_catalog = build_item_catalog(book_map)
    print(f"  {len(item_catalog):,} books in catalog")

    # Step 3: Enrich genres
    print("\n[3/7] Enriching genres...")
    enrich_genres(item_catalog, book_map)

    # Step 4: Build user sequences from interactions
    print("\n[4/7] Building user sequences...")
    user_sequences = build_user_sequences(book_map, item_catalog,
                                          min_interactions=args.min_seq_len)
    if args.max_users:
        keys = list(user_sequences.keys())[:args.max_users]
        user_sequences = {k: user_sequences[k] for k in keys}
        print(f"  Limited to {args.max_users} users")

    # Step 5: K-core filter
    print("\n[5/7] K-core filtering...")
    user_sequences, item_catalog = k_core_filter(
        user_sequences, item_catalog, k=args.k_core
    )

    # Step 6: Train/val/test split
    print("\n[6/7] Splitting train/val/test...")
    train, val, test = split_sequences(
        user_sequences,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
    )
    print(f"  Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    # Step 7: Statistics & save
    print("\n[7/7] Computing statistics & saving...")
    item_popularity = Counter()
    for seq in user_sequences.values():
        for e in seq:
            item_popularity[e["item_id"]] += 1
    item_popularity = dict(item_popularity)

    stats = compute_statistics(user_sequences, item_catalog, train, val, test)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:,}")

    save_processed(train, val, test, item_catalog, item_popularity, stats, OUT_DIR)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Preprocessing complete in {elapsed/60:.1f} minutes!")
    print(f"Output: {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
