"""
Random 5% subsample of train/val/test JSON files.
Overwrites original files in-place.
Usage: python scripts/sample_5pct.py --data_dir data/goodreads_processed --seed 42
"""
import json
import random
import argparse
import os


def sample_json(filepath: str, ratio: float = 0.05, seed: int = 42):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n_original = len(data)
    n_sample = max(1, int(n_original * ratio))
    random.seed(seed)
    sampled = random.sample(data, n_sample)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False)

    print(f"  {os.path.basename(filepath)}: {n_original:,} → {n_sample:,} ({ratio:.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/goodreads_processed')
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"\nSampling {args.ratio:.0%} from {args.data_dir}/ (seed={args.seed})\n")

    for fname in ['train.json', 'val.json', 'test.json']:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP {fname} (not found)")
            continue
        sample_json(fpath, args.ratio, args.seed)

    # Update stats.json if present
    stats_path = os.path.join(args.data_dir, 'stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        for key in ['num_train_samples', 'num_val_samples', 'num_test_samples']:
            if key in stats:
                stats[key] = max(1, int(stats[key] * args.ratio))
        # Also scale total_interactions roughly
        if 'total_interactions' in stats:
            stats['total_interactions'] = max(1, int(stats['total_interactions'] * args.ratio))
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n  Updated stats.json")

    print("\nDone.\n")


if __name__ == '__main__':
    main()
