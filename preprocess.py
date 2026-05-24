"""
Data preprocessing: Steam user libraries → user-item sequences → train/val/test splits.

Pipeline:
  1. Parse steam_games.json → item metadata catalog (title, genres, tags)
  2. Parse australian_users_items.json → per-user game libraries with playtime
  3. Filter: min playtime > 0 (user actually played the game)
  4. K-core filter: users & items with >= k interactions
  5. Sort by library order (proxy for chronological), build sequences
  6. Leave-one-out train/val/test split
  7. Save processed data + metadata
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.steam_utils import parse_steam_file, build_item_text, normalize_genres


def load_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load game catalog and user library data."""
    print("Loading steam_games.json ...")
    games = parse_steam_file(os.path.join(data_dir, 'steam_games.json'))
    print(f"  Loaded {len(games)} games")

    print("Loading australian_users_items.json ...")
    users = parse_steam_file(os.path.join(data_dir, 'australian_users_items.json'))
    print(f"  Loaded {len(users)} users")

    return games, users


def build_item_catalog(games: List[Dict], user_items: List[Dict]) -> Dict[str, Dict]:
    """Build item metadata catalog keyed by game ID.

    Merges steam_games.json metadata with item_name from user data
    to handle 16.3% unmatched game IDs.
    """
    catalog = {}

    # From steam_games.json
    for g in games:
        gid = str(g.get('id', ''))
        if not gid:
            continue
        catalog[gid] = {
            'title': g.get('title', g.get('app_name', '')),
            'genres': normalize_genres(g.get('genres', [])),
            'tags': g.get('tags', []),
            'developer': g.get('developer', ''),
            'publisher': g.get('publisher', ''),
            'price': g.get('price', 0),
            'release_date': g.get('release_date', ''),
            'text': build_item_text(g),
        }

    # Fill unmatched IDs with item_name from user data
    item_names = {}
    for u in user_items:
        for item in u.get('items', []):
            iid = str(item.get('item_id', ''))
            name = item.get('item_name', '')
            if iid and iid not in catalog and name:
                item_names[iid] = name

    for iid, name in item_names.items():
        catalog[iid] = {
            'title': name,
            'genres': [],
            'tags': [],
            'developer': '',
            'publisher': '',
            'price': 0,
            'release_date': '',
            'text': f"Title: {name} | Genres: Unknown | Developer: Unknown",
        }

    return catalog


def build_user_sequences(user_data: List[Dict],
                         item_catalog: Dict[str, Dict],
                         min_playtime: float = 0.0) -> Dict[str, List[Dict]]:
    """Build per-user interaction sequences from game library data.

    Items are already ordered by library addition time (≈ chronological).
    Filters to only items with playtime > min_playtime.

    Returns:
        {user_id: [{'item_id': str, 'playtime': float, 'item_name': str}, ...]}
    """
    user_sequences = {}

    for u in user_data:
        uid = str(u.get('user_id', ''))
        if not uid:
            continue

        seq = []
        for item in u.get('items', []):
            iid = str(item.get('item_id', ''))
            if not iid:
                continue

            playtime = float(item.get('playtime_forever', 0))
            if playtime <= min_playtime:
                continue

            # Only include items with metadata
            if iid not in item_catalog:
                continue

            seq.append({
                'item_id': iid,
                'playtime': playtime,
                'item_name': item.get('item_name', ''),
            })

        if seq:
            user_sequences[uid] = seq

    return user_sequences


def k_core_filter(user_sequences: Dict[str, List[Dict]],
                  item_catalog: Dict[str, Dict],
                  k: int = 5) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Iteratively filter users and items with < k interactions."""
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1

        # Filter items
        item_counts = Counter()
        for uid, seq in user_sequences.items():
            for entry in seq:
                item_counts[entry['item_id']] += 1

        valid_items = {iid for iid, cnt in item_counts.items() if cnt >= k}
        if len(valid_items) < len(item_counts):
            changed = True

        # Filter user sequences
        filtered_users = {}
        for uid, seq in user_sequences.items():
            filtered_seq = [e for e in seq if e['item_id'] in valid_items]
            if len(filtered_seq) >= k:
                filtered_users[uid] = filtered_seq
            else:
                changed = True

        user_sequences = filtered_users

        # Filter item catalog
        used_items = set()
        for seq in user_sequences.values():
            for e in seq:
                used_items.add(e['item_id'])
        item_catalog = {iid: info for iid, info in item_catalog.items()
                        if iid in used_items}

    print(f"  K-core converged after {iteration} iterations")
    return user_sequences, item_catalog


def split_sequences(user_sequences: Dict[str, List[Dict]],
                    min_seq_len: int = 3,
                    max_seq_len: int = 50) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Leave-one-out split: train, val, test.

    For each user:
      - train: all but last 2 interactions (with sliding windows)
      - val: second-to-last interaction
      - test: last interaction

    Sequence order is library addition order (~chronological).
    """
    train_samples = []
    val_samples = []
    test_samples = []

    for uid, seq in user_sequences.items():
        if len(seq) < min_seq_len:
            continue

        items = [e['item_id'] for e in seq]
        playtimes = [e['playtime'] for e in seq]

        # Truncate to max_seq_len (keep most recent)
        items = items[-max_seq_len:]
        playtimes = playtimes[-max_seq_len:]

        if len(items) < min_seq_len:
            continue

        # Leave-one-out: last for test, second-last for val
        # Test sample
        test_samples.append({
            'user_id': uid,
            'sequence': items[:-1],
            'target_item': items[-1],
            'playtimes': playtimes[:-1],
        })

        # Val sample (if enough items)
        if len(items) >= min_seq_len + 1:
            val_samples.append({
                'user_id': uid,
                'sequence': items[:-2],
                'target_item': items[-2],
                'playtimes': playtimes[:-2],
            })

        # Train samples: sliding windows for longer sequences
        if len(items) >= min_seq_len + 2:
            for i in range(min_seq_len, len(items) - 1):
                train_samples.append({
                    'user_id': uid,
                    'sequence': items[:i],
                    'target_item': items[i],
                    'playtimes': playtimes[:i],
                })

    return train_samples, val_samples, test_samples


def compute_statistics(user_sequences: Dict[str, List[Dict]],
                       item_catalog: Dict[str, Dict],
                       train_samples: List[Dict],
                       val_samples: List[Dict],
                       test_samples: List[Dict]) -> Dict:
    """Compute and return dataset statistics."""
    item_counts = Counter()
    seq_lengths = []
    for uid, seq in user_sequences.items():
        seq_lengths.append(len(seq))
        for e in seq:
            item_counts[e['item_id']] += 1

    stats = {
        'num_users': len(user_sequences),
        'num_items': len(item_catalog),
        'total_interactions': sum(seq_lengths),
        'avg_seq_len': np.mean(seq_lengths),
        'median_seq_len': np.median(seq_lengths),
        'min_seq_len': min(seq_lengths),
        'max_seq_len': max(seq_lengths),
        'sparsity': 1 - sum(seq_lengths) / (len(user_sequences) * len(item_catalog)) if len(item_catalog) > 0 else 1,
        'num_train_samples': len(train_samples),
        'num_val_samples': len(val_samples),
        'num_test_samples': len(test_samples),
        'tail_items_count': sum(1 for c in item_counts.values() if c < 50),
        'tail_items_ratio': (sum(1 for c in item_counts.values() if c < 50) /
                            max(len(item_counts), 1)),
        'avg_items_per_user': np.mean(seq_lengths),
        'avg_users_per_item': np.mean(list(item_counts.values())),
    }
    return stats


def save_processed(data_dir: str, train: List[Dict], val: List[Dict],
                   test: List[Dict], item_catalog: Dict[str, Dict],
                   stats: Dict, item_popularity: Dict[str, int]):
    """Save processed data to disk."""
    out_dir = os.path.join(data_dir, 'processed')
    os.makedirs(out_dir, exist_ok=True)

    items = [
        ('train.json', train),
        ('val.json', val),
        ('test.json', test),
        ('item_catalog.json', item_catalog),
        ('item_popularity.json', item_popularity),
        ('stats.json', stats),
    ]
    print(f"
Saved processed data to {out_dir}/")
    for fname, data in items:
        fpath = os.path.join(out_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=(2 if fname == 'stats.json' else None))
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        count = len(data) if isinstance(data, (list, dict)) else 0
        print(f"  {fname}: {count:,} entries, {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Steam data for LLM-Rec")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--k_core', type=int, default=5,
                        help='K-core filtering threshold')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='Minimum sequence length')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--min_playtime', type=float, default=0.0,
                        help='Minimum playtime in minutes (0 = keep all)')
    parser.add_argument('--max_users', type=int, default=None,
                        help='Max users to process (for quick testing)')
    args = parser.parse_args()

    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    games, user_data = load_data(args.data_dir)
    if args.max_users:
        user_data = user_data[:args.max_users]
        print(f"  (limited to {args.max_users} users for quick mode)")

    # Step 2: Build item catalog (merged from steam_games + user item_names)
    print("\n" + "=" * 60)
    print("STEP 2: Building item catalog")
    print("=" * 60)
    item_catalog = build_item_catalog(games, user_data)
    print(f"  {len(item_catalog)} unique games in catalog")

    # Step 3: Build user sequences (filter by playtime)
    print("\n" + "=" * 60)
    print(f"STEP 3: Building user sequences (min_playtime={args.min_playtime}m)")
    print("=" * 60)
    user_sequences = build_user_sequences(
        user_data, item_catalog, min_playtime=args.min_playtime
    )
    print(f"  {len(user_sequences)} users with valid sequences")

    # Step 4: K-core filtering
    print("\n" + "=" * 60)
    print(f"STEP 4: K-core filtering (k={args.k_core})")
    print("=" * 60)
    user_sequences, item_catalog = k_core_filter(
        user_sequences, item_catalog, k=args.k_core
    )
    print(f"  After filtering: {len(user_sequences)} users, {len(item_catalog)} items")

    # Step 5: Train/val/test split
    print("\n" + "=" * 60)
    print("STEP 5: Train/val/test split (leave-one-out + sliding windows)")
    print("=" * 60)
    train, val, test = split_sequences(
        user_sequences,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
    )
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Step 6: Statistics
    item_popularity = Counter()
    for seq in user_sequences.values():
        for e in seq:
            item_popularity[e['item_id']] += 1
    item_popularity = dict(item_popularity)

    stats = compute_statistics(user_sequences, item_catalog, train, val, test)
    print("\n" + "=" * 60)
    print("STEP 6: Dataset Statistics")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:,}")

    # Step 7: Save
    print("\n" + "=" * 60)
    print("STEP 7: Saving processed data")
    print("=" * 60)
    save_processed(args.data_dir, train, val, test, item_catalog, stats, item_popularity)

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
