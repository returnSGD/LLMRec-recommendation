"""
Traditional recommendation baselines for LLM-Rec comparison.

Methods:
  - Random: Uniform random item selection (theoretical lower bound)
  - Popularity: Most popular items by interaction count (strong baseline)
  - ItemKNN: Item-based collaborative filtering via co-occurrence
  - SASRec-style: Simple self-attention sequential model (lightweight)
"""

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import compute_all_metrics


class RandomBaseline:
    """Random item recommendation — theoretical lower bound."""

    def __init__(self, all_items: List[str]):
        self.items = list(all_items)

    def recommend(self, user_sequence: List[str], top_k: int = 20) -> List[str]:
        mask = set(user_sequence)
        candidates = [i for i in self.items if i not in mask]
        if len(candidates) < top_k:
            candidates = self.items
        return list(np.random.choice(candidates, size=min(top_k, len(candidates)), replace=False))


class PopularityBaseline:
    """Recommend most globally popular items (excluding items user has already interacted with)."""

    def __init__(self, item_popularity: Dict[str, int]):
        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        self.pop_ranking = [item_id for item_id, _ in sorted_items]

    def recommend(self, user_sequence: List[str], top_k: int = 20) -> List[str]:
        seen = set(user_sequence)
        recs = [i for i in self.pop_ranking if i not in seen]
        return recs[:top_k]


class ItemKNNBaseline:
    """Item-based KNN collaborative filtering using co-occurrence similarity.

    For each candidate item, score is sum of co-occurrence counts with items in
    the user's history. Items with higher co-occurrence get higher scores.
    """

    def __init__(self, train_samples: List[Dict], top_k_similar: int = 100):
        self.top_k_similar = top_k_similar
        self.item_sims: Dict[str, Dict[str, float]] = {}
        self._build_cooccurrence(train_samples)

    def _build_cooccurrence(self, train_samples: List[Dict]):
        """Build item-item co-occurrence matrix from training sequences."""
        cooc = defaultdict(Counter)
        item_freq = Counter()

        for s in tqdm(train_samples, desc='Building Item-Item Co-occurrence'):
            seq = s['sequence']
            target = s['target_item']
            all_items = set(seq + [target])
            item_freq.update(all_items)
            for i1 in all_items:
                for i2 in all_items:
                    if i1 != i2:
                        cooc[i1][i2] += 1

        # Normalize: Jaccard similarity
        self.item_sims = {}
        for i1, neighbors in tqdm(cooc.items(), desc='Normalizing Item Similarities'):
            sims = {}
            f1 = item_freq[i1]
            if f1 == 0:
                continue
            for i2, cnt in neighbors.items():
                f2 = item_freq[i2]
                union = f1 + f2 - cnt
                if union > 0:
                    sims[i2] = cnt / union  # Jaccard
            # Keep top-k most similar items
            top_k_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:self.top_k_similar]
            self.item_sims[i1] = dict(top_k_items)

    def recommend(self, user_sequence: List[str], top_k: int = 20) -> List[str]:
        """Score candidates by sum of similarity to items in user history."""
        scores = Counter()
        for hist_item in user_sequence:
            if hist_item in self.item_sims:
                for neighbor, sim in self.item_sims[hist_item].items():
                    if neighbor not in user_sequence:
                        scores[neighbor] += sim

        # Return top-k by score
        ranked = [item_id for item_id, _ in scores.most_common(top_k)]
        return ranked


class SimpleSequentialBaseline:
    """Simple Markov-chain sequential model: predict items most frequently following
    the last N items in the user's history. Equivalent to a simple n-gram model.
    """

    def __init__(self, train_samples: List[Dict], n_gram: int = 2, top_k: int = 100):
        self.n_gram = n_gram
        self.top_k = top_k
        # {(item_n-1, ..., item_1): Counter{next_item: count}}
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._build_transitions(train_samples)

    def _build_transitions(self, train_samples: List[Dict]):
        for s in tqdm(train_samples, desc='Building n-gram transitions'):
            seq = s['sequence']
            target = s['target_item']
            if len(seq) < self.n_gram:
                continue
            context = tuple(seq[-self.n_gram:])
            self.transitions[context][target] += 1

        # Also populate unigram fallback
        self.unigram = Counter()
        for s in train_samples:
            self.unigram[s['target_item']] += 1

    def recommend(self, user_sequence: List[str], top_k: int = 20) -> List[str]:
        scores = Counter()
        seen = set(user_sequence)

        # Try exact n-gram match
        if len(user_sequence) >= self.n_gram:
            context = tuple(user_sequence[-self.n_gram:])
            if context in self.transitions:
                for item, cnt in self.transitions[context].items():
                    if item not in seen:
                        scores[item] = cnt

        # Fall back to (n-1)-gram, (n-2)-gram, ..., unigram
        for ng in range(self.n_gram - 1, 0, -1):
            if len(scores) >= top_k:
                break
            if len(user_sequence) >= ng:
                context = tuple(user_sequence[-ng:])
                if context in self.transitions:
                    for item, cnt in self.transitions[context].items():
                        if item not in seen and item not in scores:
                            scores[item] = cnt * 0.5  # discount longer matches less

        # Final fallback: global popularity
        if len(scores) < top_k:
            for item, cnt in self.unigram.most_common():
                if item not in seen and item not in scores:
                    scores[item] = cnt * 0.1

        ranked = [item_id for item_id, _ in scores.most_common(top_k)]
        return ranked


def run_all_baselines(test_samples: List[Dict], train_samples: List[Dict],
                       item_catalog: Dict, item_popularity: Dict[str, int],
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict]:
    """Run all traditional baselines and return metrics.

    Returns:
        {method_name: {metric_name: value, ...}}
    """
    all_items = list(item_catalog.keys())

    # Build catalog sets for metrics
    catalog_ids = set(item_catalog.keys())
    item_genres = {}
    for iid, info in item_catalog.items():
        item_genres[iid] = info.get('genres', info.get('tags', []))

    tail_items = {iid for iid, cnt in item_popularity.items() if cnt < 50}

    max_k = max(k_values)
    results = {}

    # --- Random ---
    print("\n" + "=" * 60)
    print("Evaluating Random Baseline")
    print("=" * 60)
    random_model = RandomBaseline(all_items)
    predictions, ground_truths = [], []
    for s in tqdm(test_samples, desc='Random'):
        preds = random_model.recommend(s['sequence'], top_k=max_k)
        predictions.append(list(preds))
        ground_truths.append([s['target_item']])

    metrics = compute_all_metrics(predictions, ground_truths, item_genres,
                                   item_popularity, catalog_ids, tail_items, k_values)
    results['Random'] = metrics

    # --- Popularity ---
    print("\n" + "=" * 60)
    print("Evaluating Popularity Baseline")
    print("=" * 60)
    pop_model = PopularityBaseline({k: int(v) for k, v in item_popularity.items()})
    predictions, ground_truths = [], []
    for s in tqdm(test_samples, desc='Popularity'):
        preds = pop_model.recommend(s['sequence'], top_k=max_k)
        predictions.append(list(preds))
        ground_truths.append([s['target_item']])

    metrics = compute_all_metrics(predictions, ground_truths, item_genres,
                                   item_popularity, catalog_ids, tail_items, k_values)
    results['Popularity'] = metrics

    # --- ItemKNN ---
    print("\n" + "=" * 60)
    print("Evaluating ItemKNN Baseline")
    print("=" * 60)
    knn_model = ItemKNNBaseline(train_samples)
    predictions, ground_truths = [], []
    for s in tqdm(test_samples, desc='ItemKNN'):
        preds = knn_model.recommend(s['sequence'], top_k=max_k)
        predictions.append(list(preds))
        ground_truths.append([s['target_item']])

    metrics = compute_all_metrics(predictions, ground_truths, item_genres,
                                   item_popularity, catalog_ids, tail_items, k_values)
    results['ItemKNN'] = metrics

    # --- Simple Sequential (n-gram) ---
    print("\n" + "=" * 60)
    print("Evaluating Sequential Baseline (2-gram)")
    print("=" * 60)
    seq_model = SimpleSequentialBaseline(train_samples, n_gram=2)
    predictions, ground_truths = [], []
    for s in tqdm(test_samples, desc='Seq-Ngram'):
        preds = seq_model.recommend(s['sequence'], top_k=max_k)
        predictions.append(list(preds))
        ground_truths.append([s['target_item']])

    metrics = compute_all_metrics(predictions, ground_truths, item_genres,
                                   item_popularity, catalog_ids, tail_items, k_values)
    results['SeqNgram'] = metrics

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run traditional baselines for LLM-Rec')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output', type=str, default='results/traditional_metrics.json')
    parser.add_argument('--max_test', type=int, default=2000,
                        help='Max test samples to evaluate')
    parser.add_argument('--max_train', type=int, default=50000,
                        help='Max train samples for building models')
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 20])
    args = parser.parse_args()

    print("Loading data...")
    with open(os.path.join(args.data_dir, 'train.json'), 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    with open(os.path.join(args.data_dir, 'test.json'), 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    with open(os.path.join(args.data_dir, 'item_catalog.json'), 'r', encoding='utf-8') as f:
        item_catalog = json.load(f)
    with open(os.path.join(args.data_dir, 'item_popularity.json'), 'r', encoding='utf-8') as f:
        item_popularity = json.load(f)

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    if args.max_train:
        train_samples = train_samples[:args.max_train]
        print(f"  Using {len(train_samples)} train samples")
    if args.max_test:
        test_samples = test_samples[:args.max_test]
        print(f"  Using {len(test_samples)} test samples")

    results = run_all_baselines(
        test_samples=test_samples,
        train_samples=train_samples,
        item_catalog=item_catalog,
        item_popularity=item_popularity,
        k_values=args.top_k,
    )

    # Display
    print("\n" + "=" * 70)
    print("TRADITIONAL BASELINE RESULTS (Top-10)")
    print("=" * 70)
    header = f"{'Method':<15} {'NDCG':>8} {'Recall':>8} {'HR':>8} {'TailRec':>8} {'Novelty':>8} {'ILS':>8}"
    print(header)
    print("-" * 70)
    for method, metrics in results.items():
        print(f"{method:<15} {metrics.get('NDCG@10',0):>8.4f} {metrics.get('Recall@10',0):>8.4f} "
              f"{metrics.get('HR@10',0):>8.4f} {metrics.get('Tail_Recall@10',0):>8.4f} "
              f"{metrics.get('Novelty@10',0):>8.4f} {metrics.get('ILS@10',0):>8.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Convert numpy values for JSON serialization
    serializable = {}
    for method, metrics in results.items():
        serializable[method] = {k: float(v) for k, v in metrics.items()}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")
