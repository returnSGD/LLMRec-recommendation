"""
Goodreads data exploration script.

Explores:
  1. goodreads_interactions.csv — user-book interactions
  2. goodreads_books.json.gz — book metadata
  3. goodreads_book_genres_initial.json.gz — genre tags
  4. user_id_map.csv / book_id_map.csv — ID mappings
"""

import os
import sys
import json
import gzip
import csv
from collections import Counter, defaultdict
import numpy as np

DATA_DIR = "data_validation"


def explore_id_maps():
    """Explore ID mapping files."""
    print("=" * 60)
    print("1. ID MAPPINGS")
    print("=" * 60)

    user_map = {}
    with open(os.path.join(DATA_DIR, "user_id_map.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_map[row["user_id_csv"]] = row["user_id"]

    book_map = {}
    with open(os.path.join(DATA_DIR, "book_id_map.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            book_map[row["book_id_csv"]] = row["book_id"]

    print(f"  User ID mappings: {len(user_map):,}")
    print(f"  Book ID mappings: {len(book_map):,}")
    print(f"  Sample user mapping: {list(user_map.items())[:3]}")
    print(f"  Sample book mapping: {list(book_map.items())[:3]}")
    return user_map, book_map


def explore_interactions(user_map, book_map):
    """Explore the interactions CSV."""
    print("\n" + "=" * 60)
    print("2. INTERACTIONS (goodreads_interactions.csv)")
    print("=" * 60)

    n_rows = 0
    n_read = 0
    n_reviewed = 0
    ratings = []
    user_counter = Counter()
    book_counter = Counter()

    # Read first few lines for structure check
    with open(os.path.join(DATA_DIR, "goodreads_interactions.csv"), "r") as f:
        reader = csv.DictReader(f)
        print(f"  Columns: {reader.fieldnames}")
        for i, row in enumerate(reader):
            n_rows += 1
            if row["is_read"] == "1":
                n_read += 1
            if row["is_reviewed"] == "1":
                n_reviewed += 1
            if row["rating"] and row["rating"] != "0":
                ratings.append(int(row["rating"]))
            user_counter[row["user_id"]] += 1
            book_counter[row["book_id"]] += 1

            if n_rows % 50000000 == 0:
                print(f"  Processed {n_rows:,} rows...")

    print(f"  Total interactions: {n_rows:,}")
    print(f"  Read interactions: {n_read:,} ({100*n_read/max(n_rows,1):.1f}%)")
    print(f"  Reviewed interactions: {n_reviewed:,} ({100*n_reviewed/max(n_rows,1):.1f}%)")
    print(f"  Rated interactions: {len(ratings):,} ({100*len(ratings)/max(n_rows,1):.1f}%)")

    if ratings:
        ratings = np.array(ratings)
        print(f"  Rating distribution:")
        for r in range(1, 6):
            cnt = np.sum(ratings == r)
            print(f"    {r}: {cnt:,} ({100*cnt/len(ratings):.1f}%)")
        print(f"  Rating mean: {ratings.mean():.2f}, std: {ratings.std():.2f}")

    print(f"  Unique CSV user_ids: {len(user_counter):,}")
    print(f"  Unique CSV book_ids: {len(book_counter):,}")

    # Sequence length stats
    seq_lens = list(user_counter.values())
    seq_lens_arr = np.array(seq_lens)
    print(f"\n  User interaction count distribution:")
    print(f"    Mean: {seq_lens_arr.mean():.1f}")
    print(f"    Median: {np.median(seq_lens_arr):.1f}")
    print(f"    Min: {seq_lens_arr.min()}, Max: {seq_lens_arr.max()}")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"    P{pct}: {np.percentile(seq_lens_arr, pct):.0f}")

    # How many users have >= K interactions?
    for k in [3, 5, 10, 20]:
        n = sum(1 for c in seq_lens if c >= k)
        print(f"  Users with >= {k} interactions: {n:,} ({100*n/max(len(user_counter),1):.1f}%)")

    # Item popularity distribution
    item_pops = list(book_counter.values())
    item_pops_arr = np.array(item_pops)
    print(f"\n  Item popularity distribution:")
    print(f"    Mean: {item_pops_arr.mean():.1f}")
    print(f"    Median: {np.median(item_pops_arr):.1f}")
    print(f"    Min: {item_pops_arr.min()}, Max: {item_pops_arr.max()}")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"    P{pct}: {np.percentile(item_pops_arr, pct):.0f}")
    tail_50 = sum(1 for c in item_pops if c < 50)
    print(f"  Tail items (<50 interactions): {tail_50:,} ({100*tail_50/max(len(book_counter),1):.1f}%)")

    sparsity = 1 - n_rows / (len(user_counter) * len(book_counter))
    print(f"  Sparsity: {sparsity:.4f}")

    return user_counter, book_counter


def explore_books():
    """Explore book metadata."""
    print("\n" + "=" * 60)
    print("3. BOOK METADATA (goodreads_books.json.gz)")
    print("=" * 60)

    books = {}
    n_lines = 0
    n_with_desc = 0
    n_with_pages = 0
    n_with_series = 0
    ratings_counts = []
    authors_per_book = []

    with gzip.open(os.path.join(DATA_DIR, "goodreads_books.json.gz"), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            n_lines += 1
            try:
                book = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            bid = book.get("book_id", "")
            books[bid] = book

            if book.get("description"):
                n_with_desc += 1
            if book.get("num_pages") and book["num_pages"].isdigit():
                n_with_pages += 1
            if book.get("series"):
                n_with_series += 1
            rc = book.get("ratings_count", "0")
            try:
                ratings_counts.append(int(rc))
            except (ValueError, TypeError):
                pass
            authors_per_book.append(len(book.get("authors", [])))

            if n_lines % 500000 == 0:
                print(f"  Processed {n_lines:,} lines...")

    print(f"  Total books: {n_lines:,}")
    print(f"  Books with description: {n_with_desc:,} ({100*n_with_desc/max(n_lines,1):.1f}%)")
    print(f"  Books with page count: {n_with_pages:,} ({100*n_with_pages/max(n_lines,1):.1f}%)")
    print(f"  Books in series: {n_with_series:,} ({100*n_with_series/max(n_lines,1):.1f}%)")

    if ratings_counts:
        rc_arr = np.array(ratings_counts)
        print(f"\n  Ratings count distribution:")
        print(f"    Mean: {rc_arr.mean():.1f}, Median: {np.median(rc_arr):.1f}")
        print(f"    Min: {rc_arr.min()}, Max: {rc_arr.max()}")
        for pct in [25, 50, 75, 90, 95, 99]:
            print(f"    P{pct}: {np.percentile(rc_arr, pct):.0f}")

    if authors_per_book:
        apb = np.array(authors_per_book)
        print(f"\n  Authors per book: mean={apb.mean():.2f}, max={apb.max()}")

    # Sample books
    print("\n  Sample books:")
    for bid in list(books.keys())[:3]:
        b = books[bid]
        authors = [a.get("author_id", "?") for a in b.get("authors", [])]
        print(f"    [{bid}] {b.get('title', 'N/A')} | authors: {authors} | "
              f"rating: {b.get('average_rating', 'N/A')} | pages: {b.get('num_pages', 'N/A')} | "
              f"year: {b.get('publication_year', 'N/A')}")
        if b.get("description"):
            desc = b["description"][:150].replace("\n", " ")
            print(f"      Description: {desc}...")

    return books


def explore_genres(book_map):
    """Explore genre data."""
    print("\n" + "=" * 60)
    print("4. GENRES (goodreads_book_genres_initial.json.gz)")
    print("=" * 60)

    # Build reverse map: goodreads book_id -> csv book_id
    csv_to_gr = {v: k for k, v in book_map.items()}

    genre_counter = Counter()
    books_with_genres = 0
    books_without_genres = 0
    n_lines = 0

    with gzip.open(os.path.join(DATA_DIR, "goodreads_book_genres_initial.json.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            genres = entry.get("genres", {})
            if genres:
                books_with_genres += 1
                for genre_name in genres:
                    genre_counter[genre_name] += 1
            else:
                books_without_genres += 1

            if n_lines % 500000 == 0:
                print(f"  Processed {n_lines:,} lines...")

    print(f"  Total genre entries: {n_lines:,}")
    print(f"  Books with genres: {books_with_genres:,}")
    print(f"  Books without genres: {books_without_genres:,}")
    print(f"  Unique genre categories: {len(genre_counter):,}")
    print(f"\n  Top 20 genres:")
    for genre, count in genre_counter.most_common(20):
        print(f"    {genre}: {count:,}")

    return genre_counter


def explore_overlap(user_counter, book_counter, books):
    """Check overlap between interactions and metadata."""
    print("\n" + "=" * 60)
    print("5. OVERLAP ANALYSIS")
    print("=" * 60)

    csv_book_ids = set(book_counter.keys())
    meta_book_ids = set(books.keys())

    overlap = csv_book_ids & meta_book_ids
    only_in_csv = csv_book_ids - meta_book_ids
    only_in_meta = meta_book_ids - csv_book_ids

    print(f"  Books in interactions: {len(csv_book_ids):,}")
    print(f"  Books in metadata: {len(meta_book_ids):,}")
    print(f"  Overlap: {len(overlap):,} ({100*len(overlap)/max(len(csv_book_ids),1):.1f}%)")
    print(f"  Only in interactions (no metadata): {len(only_in_csv):,}")
    print(f"  Only in metadata (no interactions): {len(only_in_meta):,}")


def main():
    print("GOODREADS DATA EXPLORATION")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}\n")

    user_map, book_map = explore_id_maps()
    user_counter, book_counter = explore_interactions(user_map, book_map)
    books = explore_books()
    explore_genres(book_map)
    explore_overlap(user_counter, book_counter, books)

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
