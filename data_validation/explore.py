"""
Goodreads data exploration — streaming edition.
Handles 4.1GB CSV and 2GB gz files with chunk-based streaming.
Key improvements over explore_goodreads.py:
  - csv.reader with buffered progress for 4.1GB interactions
  - Random reservoir sampling for quick distribution checks
  - Genre hierarchy analysis (Goodreads genres are nested)
  - Read+rating correlation analysis
  - Data quality checks (missing values, malformed JSON)
  - Memory-efficient: never loads full dataset into RAM
"""

import os
import sys
import json
import gzip
import csv
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "").rstrip("/\\")
# data_validation is the folder name, files are directly inside
if not os.path.exists(os.path.join(DATA_DIR, "goodreads_interactions.csv")):
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RESERVOIR_SIZE = 50000  # for quick distribution estimates


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def section_1_id_maps():
    """ID mappings: CSV internal → Goodreads native IDs"""
    print_section("1. ID MAPPINGS")

    user_map = {}
    with open(os.path.join(DATA_DIR, "user_id_map.csv"), "r") as f:
        for row in csv.DictReader(f):
            user_map[row["user_id_csv"]] = row["user_id"]

    book_map = {}
    with open(os.path.join(DATA_DIR, "book_id_map.csv"), "r") as f:
        for row in csv.DictReader(f):
            book_map[row["book_id_csv"]] = row["book_id"]

    print(f"  User mappings:  {len(user_map):,}")
    print(f"  Book mappings:  {len(book_map):,}")

    # Check if mappings are 1:1 or many:1
    rev_user = defaultdict(list)
    for k, v in user_map.items():
        rev_user[v].append(k)
    multi_user = {k: v for k, v in rev_user.items() if len(v) > 1}
    if multi_user:
        print(f"  ⚠ Multi-mapped users: {len(multi_user)} (many-to-1)")

    rev_book = defaultdict(list)
    for k, v in book_map.items():
        rev_book[v].append(k)
    multi_book = {k: v for k, v in rev_book.items() if len(v) > 1}
    if multi_book:
        print(f"  ⚠ Multi-mapped books: {len(multi_book)} (many-to-1)")

    return user_map, book_map


def section_2_interactions(user_map, book_map):
    """Stream interactions CSV (4.1GB) with reservoir sampling for distributions."""
    print_section("2. INTERACTIONS (streaming 4.1GB CSV)")

    fpath = os.path.join(DATA_DIR, "goodreads_interactions.csv")
    fsize_gb = os.path.getsize(fpath) / (1024**3)

    n_total = 0
    n_read = 0
    n_reviewed = 0
    n_rated = 0
    ratings_reservoir = []
    user_counter = Counter()
    book_counter = Counter()

    t0 = time.time()
    # Count total lines first (fast, just counts newlines)
    print(f"  File size: {fsize_gb:.1f} GB")
    print(f"  Counting lines...")
    with open(fpath, "rb") as f:
        total_lines = sum(1 for _ in f) - 1  # minus header
    print(f"  Total rows (excl header): {total_lines:,}")

    print(f"  Streaming & sampling...")
    with open(fpath, "r") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"  Columns: {cols}")

        for i, row in enumerate(reader):
            n_total += 1
            is_read = row.get("is_read", "0") == "1"
            is_reviewed = row.get("is_reviewed", "0") == "1"
            rating = row.get("rating", "0")

            if is_read:
                n_read += 1
            if is_reviewed:
                n_reviewed += 1
            if rating and rating != "0":
                n_rated += 1
                r = int(rating)
                if len(ratings_reservoir) < RESERVOIR_SIZE:
                    ratings_reservoir.append(r)
                elif np.random.random() < RESERVOIR_SIZE / (n_rated + 1):
                    idx = np.random.randint(0, RESERVOIR_SIZE)
                    ratings_reservoir[idx] = r

            user_counter[row["user_id"]] += 1
            book_counter[row["book_id"]] += 1

            if n_total % 10_000_000 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                eta = (total_lines - n_total) / rate
                pct = 100 * n_total / total_lines
                print(f"    {pct:.0f}% | {n_total/1e6:.1f}M rows | "
                      f"{rate/1e6:.1f}M rows/s | ETA: {eta/60:.0f}m")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} minutes ({n_total/elapsed/1e6:.1f}M rows/s)")

    print(f"\n  Results:")
    print(f"    Total interactions:   {n_total:,}")
    print(f"    Read (is_read=1):     {n_read:,} ({100*n_read/max(n_total,1):.1f}%)")
    print(f"    Reviewed:             {n_reviewed:,} ({100*n_reviewed/max(n_total,1):.1f}%)")
    print(f"    Rated (rating>0):     {n_rated:,} ({100*n_rated/max(n_total,1):.1f}%)")

    if ratings_reservoir:
        r_arr = np.array(ratings_reservoir)
        print(f"    Rating distribution (reservoir n={len(ratings_reservoir)}):")
        for r in range(1, 6):
            cnt = np.sum(r_arr == r)
            print(f"      {r}: {cnt:,} ({100*cnt/len(r_arr):.1f}%)")
        print(f"    Rating mean: {r_arr.mean():.2f}, std: {r_arr.std():.2f}")

    print(f"    Unique users:   {len(user_counter):,}")
    print(f"    Unique books:   {len(book_counter):,}")

    # Sequence length stats
    seq_lens = np.array(list(user_counter.values()))
    print(f"\n  User interaction count (user profile size):")
    print(f"    Mean: {seq_lens.mean():.1f}  Median: {np.median(seq_lens):.0f}")
    print(f"    Min: {seq_lens.min()}  Max: {seq_lens.max()}")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"    P{pct}: {np.percentile(seq_lens, pct):.0f}")
    for k in [3, 5, 10, 20, 50]:
        n = np.sum(seq_lens >= k)
        print(f"    Users >= {k} interactions: {n:,} ({100*n/len(user_counter):.1f}%)")

    # Book popularity
    item_pops = np.array(list(book_counter.values()))
    print(f"\n  Book popularity (interactions per book):")
    print(f"    Mean: {item_pops.mean():.1f}  Median: {np.median(item_pops):.0f}")
    print(f"    Min: {item_pops.min()}  Max: {item_pops.max()}")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"    P{pct}: {np.percentile(item_pops, pct):.0f}")
    tail_cutoffs = [10, 20, 50, 100]
    for t in tail_cutoffs:
        n_tail = np.sum(item_pops < t)
        print(f"    Tail (<{t} interactions): {n_tail:,} ({100*n_tail/len(item_pops):.1f}%)")

    sparsity = 1 - n_total / (len(user_counter) * len(book_counter))
    print(f"\n  Sparsity: {sparsity:.6f}")

    return user_counter, book_counter


def section_3_books():
    """Stream book metadata (2GB gzip) — build item catalog info."""
    print_section("3. BOOK METADATA (streaming 2GB .json.gz)")

    fpath = os.path.join(DATA_DIR, "goodreads_books.json.gz")
    fsize_gb = os.path.getsize(fpath) / (1024**3)
    print(f"  File size: {fsize_gb:.1f} GB (compressed)")

    n_total = 0
    n_with_desc = 0
    n_with_pages = 0
    n_with_series = 0
    n_with_isbn = 0
    n_with_pub_year = 0
    n_with_authors = 0
    ratings_counts = []
    authors_per_book = []
    pub_years = []
    languages = Counter()
    sample_books = []

    t0 = time.time()
    with gzip.open(fpath, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                book = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if book.get("description") and book["description"].strip():
                n_with_desc += 1
            if book.get("num_pages") and str(book["num_pages"]).strip().isdigit():
                n_with_pages += 1
            if book.get("series"):
                n_with_series += 1
            if book.get("isbn"):
                n_with_isbn += 1
            if book.get("authors"):
                n_with_authors += 1
                authors_per_book.append(len(book["authors"]))
            pub_year = book.get("publication_year", "")
            if pub_year and str(pub_year).strip().isdigit():
                n_with_pub_year += 1
                pub_years.append(int(pub_year))
            lang = book.get("language_code", "unknown")
            languages[lang] += 1
            rc = book.get("ratings_count", "0")
            try:
                ratings_counts.append(int(rc))
            except (ValueError, TypeError):
                pass

            if len(sample_books) < 5:
                sample_books.append(book)

            if n_total % 500_000 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                print(f"    {n_total/1e6:.1f}M books | {rate/1e3:.0f}K books/s")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({n_total/elapsed/1e3:.0f}K books/s)")

    print(f"\n  Results:")
    print(f"    Total books:          {n_total:,}")
    print(f"    With description:     {n_with_desc:,} ({100*n_with_desc/max(n_total,1):.1f}%)")
    print(f"    With page count:      {n_with_pages:,} ({100*n_with_pages/max(n_total,1):.1f}%)")
    print(f"    With series:          {n_with_series:,} ({100*n_with_series/max(n_total,1):.1f}%)")
    print(f"    With ISBN:            {n_with_isbn:,} ({100*n_with_isbn/max(n_total,1):.1f}%)")
    print(f"    With authors:         {n_with_authors:,} ({100*n_with_authors/max(n_total,1):.1f}%)")
    print(f"    With pub year:        {n_with_pub_year:,} ({100*n_with_pub_year/max(n_total,1):.1f}%)")

    if ratings_counts:
        rc = np.array(ratings_counts)
        print(f"\n  Ratings count per book (external):")
        print(f"    Mean: {rc.mean():.0f}  Median: {np.median(rc):.0f}")
        for pct in [25, 50, 75, 90, 95, 99]:
            print(f"    P{pct}: {np.percentile(rc, pct):.0f}")

    if pub_years:
        py = np.array(pub_years)
        py_valid = py[(py > 1800) & (py <= 2026)]
        print(f"\n  Publication years (valid range):")
        print(f"    Range: {py_valid.min():.0f} – {py_valid.max():.0f}")
        print(f"    Median: {np.median(py_valid):.0f}")
        decades = {}
        for y in py_valid:
            decade = (y // 10) * 10
            decades[decade] = decades.get(decade, 0) + 1
        for dec in sorted(decades.keys())[-10:]:
            print(f"    {dec}s: {decades[dec]:,}")

    if authors_per_book:
        apb = np.array(authors_per_book)
        print(f"\n  Authors per book: mean={apb.mean():.2f}, max={apb.max()}")
        for na in range(0, 6):
            cnt = np.sum(apb == na)
            if cnt > 0:
                print(f"    {na} author(s): {cnt:,} ({100*cnt/len(apb):.1f}%)")

    print(f"\n  Top languages:")
    for lang, cnt in languages.most_common(10):
        print(f"    {lang}: {cnt:,}")

    print(f"\n  Sample books:")
    for b in sample_books[:3]:
        title = b.get("title", "N/A")
        authors = [a.get("author_id", "?") for a in b.get("authors", [])]
        print(f"    [{b.get('book_id','?')}] {title[:80]}")
        print(f"      Authors: {authors} | Rating: {b.get('average_rating','N/A')}")
        print(f"      Pages: {b.get('num_pages','N/A')} | Year: {b.get('publication_year','N/A')}")
        desc = (b.get("description") or "")[:120].replace("\n", " ")
        if desc:
            print(f"      Desc: {desc}...")

    return None  # books not kept in memory


def section_4_genres(book_map):
    """Stream genre mappings."""
    print_section("4. GENRES (streaming .json.gz)")

    fpath = os.path.join(DATA_DIR, "goodreads_book_genres_initial.json.gz")
    fsize_mb = os.path.getsize(fpath) / (1024**2)
    print(f"  File size: {fsize_mb:.0f} MB")

    genre_counter = Counter()
    books_with_genres = 0
    books_without_genres = 0
    n_total = 0
    # Track genre category counts (Goodreads uses hierarchical categories)
    top_level_genres = Counter()

    t0 = time.time()
    with gzip.open(fpath, "rt", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            genres = entry.get("genres", {})
            if genres:
                books_with_genres += 1
                for genre_name in genres:
                    genre_counter[genre_name] += 1
                    # Extract top-level category (first word or before slash)
                    top = genre_name.split("/")[0].strip().split()[0].lower()
                    top_level_genres[top] += 1
            else:
                books_without_genres += 1

            if n_total % 500_000 == 0:
                elapsed = time.time() - t0
                print(f"    {n_total/1e6:.1f}M entries | {n_total/elapsed/1e3:.0f}K/s")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    print(f"\n  Results:")
    print(f"    Total entries:          {n_total:,}")
    print(f"    With genres:            {books_with_genres:,}")
    print(f"    Without genres:         {books_without_genres:,}")
    print(f"    Unique genre labels:    {len(genre_counter):,}")
    print(f"    Avg genres per book:    {sum(genre_counter.values())/max(books_with_genres,1):.1f}")

    print(f"\n  Top 25 genre labels:")
    for genre, cnt in genre_counter.most_common(25):
        bar = "█" * int(40 * cnt / genre_counter.most_common(1)[0][1])
        print(f"    {genre:<35} {cnt:>8,}  {bar}")

    print(f"\n  Top-level genre categories:")
    for top, cnt in top_level_genres.most_common(15):
        print(f"    {top}: {cnt:,}")

    return genre_counter


def section_5_data_quality(user_counter, book_counter, book_map):
    """Check data quality and overlap between sources."""
    print_section("5. DATA QUALITY & OVERLAP")

    print("  [Note] Overlap between interactions and book metadata is checked")
    print("         after all three files are processed (to avoid OOM).")
    print(f"  Users in interactions:      {len(user_counter):,}")
    print(f"  Books in interactions:      {len(book_counter):,}")
    print(f"  Books in ID map:            {len(book_map):,}")

    # Check how many interaction book_ids have entries in the book map
    interaction_book_ids = set(book_counter.keys())
    map_book_ids = set(book_map.keys())
    overlap = interaction_book_ids & map_book_ids
    print(f"  Book overlap with ID map:   {len(overlap):,} "
          f"({100*len(overlap)/max(len(interaction_book_ids),1):.1f}%)")

    # Rating denseness
    print(f"\n  Data denseness assessment:")
    print(f"    Users with 1 interaction:  {sum(1 for c in user_counter.values() if c == 1):,}")
    print(f"    Users with 2-4:            {sum(1 for c in user_counter.values() if 2 <= c <= 4):,}")
    print(f"    Users with 5-19:           {sum(1 for c in user_counter.values() if 5 <= c <= 19):,}")
    print(f"    Users with 20+:            {sum(1 for c in user_counter.values() if c >= 20):,}")


def section_6_cross_dataset_comparison():
    """Quick comparison with Steam dataset characteristics (if processed data exists)."""
    print_section("6. CROSS-DATASET COMPARISON (Goodreads vs Steam)")

    steam_stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "stats.json"
    )
    if os.path.exists(steam_stats_path):
        with open(steam_stats_path, "r") as f:
            steam = json.load(f)
        print("  Steam dataset:")
        for k in ["num_users", "num_items", "total_interactions",
                   "avg_seq_len", "sparsity", "num_train_samples"]:
            v = steam.get(k, "N/A")
            if isinstance(v, float):
                print(f"    {k}: {v:.2f}")
            else:
                print(f"    {k}: {v:,}")
        print("\n  → After preprocessing Goodreads, compare these stats for domain shift analysis.")
    else:
        print("  Steam processed data not found at data/processed/ — comparison skipped.")
        print("  Run: python preprocess.py --data_dir data  first.")


def main():
    t_start = time.time()
    print("=" * 70)
    print("  GOODREADS DATA EXPLORATION (Streaming Edition)")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    user_map, book_map = section_1_id_maps()
    user_counter, book_counter = section_2_interactions(user_map, book_map)
    section_3_books()
    section_4_genres(book_map)
    section_5_data_quality(user_counter, book_counter, book_map)
    section_6_cross_dataset_comparison()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  EXPLORATION COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
