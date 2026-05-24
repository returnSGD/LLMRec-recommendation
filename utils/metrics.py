"""
Evaluation metrics for generative recommendation.

Metrics:
  - NDCG@K, Recall@K, HR@K (standard accuracy)
  - ILS@K, Coverage@K (diversity)
  - Tail Recall@K (cold item performance)
  - Novelty@K (average inverse popularity)
  - OOD@K (hallucination rate — generated items not in catalog)
"""

import numpy as np
from typing import Dict, List, Set, Optional


def _dcg_at_k(scores: List[float], k: int) -> float:
    scores = np.array(scores[:k], dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    discounts = np.log2(np.arange(1, len(scores) + 1) + 1)
    return np.sum((2 ** scores - 1) / discounts)


def ndcg_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    gt_set = set(ground_truth)
    relevance = [1.0 if item in gt_set else 0.0 for item in predicted[:k]]
    ideal = [1.0] * min(len(ground_truth), k)
    dcg = _dcg_at_k(relevance, k)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """Recall@K: fraction of ground truth items in top-K predictions."""
    gt_set = set(ground_truth)
    hits = sum(1 for item in predicted[:k] if item in gt_set)
    return hits / len(gt_set) if gt_set else 0.0


def hit_rate_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """HR@K: 1 if any ground truth item is in top-K, 0 otherwise."""
    gt_set = set(ground_truth)
    hits = any(item in gt_set for item in predicted[:k])
    return 1.0 if hits else 0.0


def intra_list_similarity(predicted: List[str],
                          item_genres: Dict[str, List[str]], k: int) -> float:
    """ILS@K: average pairwise Jaccard similarity of predicted items' genres.
    Lower = more diverse.
    """
    items = predicted[:k]
    if len(items) < 2:
        return 0.0
    sims = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            genres_i = set(item_genres.get(items[i], []))
            genres_j = set(item_genres.get(items[j], []))
            union = len(genres_i | genres_j)
            intersection = len(genres_i & genres_j)
            sim = intersection / union if union > 0 else 0.0
            sims.append(sim)
    return np.mean(sims) if sims else 0.0


def coverage_at_k(predicted_lists: List[List[str]],
                  catalog_size: int, k: int) -> float:
    """Coverage@K: fraction of unique items appearing in any top-K list."""
    all_items = set()
    for preds in predicted_lists:
        all_items.update(preds[:k])
    return len(all_items) / catalog_size if catalog_size > 0 else 0.0


def tail_recall_at_k(predicted: List[str], ground_truth: List[str], k: int,
                     tail_items: Set[str]) -> float:
    """Tail Recall@K: recall computed only for ground-truth items in the tail set."""
    gt_tail = [item for item in ground_truth if item in tail_items]
    if not gt_tail:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in gt_tail)
    return hits / len(gt_tail)


def novelty_at_k(predicted: List[str], item_popularity: Dict[str, int],
                 max_pop: int, k: int) -> float:
    """Novelty@K: average inverse popularity of recommended items.
    Higher = more novel/less popular items recommended.
    """
    items = predicted[:k]
    if not items:
        return 0.0
    novelties = []
    for item in items:
        pop = item_popularity.get(item, max_pop)
        novelties.append(1.0 - pop / max(max_pop, 1))
    return np.mean(novelties)


def ood_rate(predicted_lists: List[List[str]],
             catalog_ids: Set[str], k: int) -> float:
    """OOD@K: fraction of generated items that are not in the item catalog."""
    total = 0
    ood = 0
    for preds in predicted_lists:
        for item in preds[:k]:
            total += 1
            if item not in catalog_ids:
                ood += 1
    return ood / total if total > 0 else 0.0


def compute_all_metrics(predictions: List[List[str]],
                        ground_truths: List[List[str]],
                        item_genres: Dict[str, List[str]],
                        item_popularity: Dict[str, int],
                        catalog_ids: Set[str],
                        tail_items: Set[str],
                        k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """Compute all evaluation metrics and return as a flat dict."""
    max_pop = max(item_popularity.values()) if item_popularity else 1
    metrics = {}

    for k in k_values:
        ndcg_vals = [ndcg_at_k(p, g, k) for p, g in zip(predictions, ground_truths)]
        recall_vals = [recall_at_k(p, g, k) for p, g in zip(predictions, ground_truths)]
        hr_vals = [hit_rate_at_k(p, g, k) for p, g in zip(predictions, ground_truths)]
        ils_vals = [intra_list_similarity(p, item_genres, k) for p in predictions]
        tail_recall_vals = [tail_recall_at_k(p, g, k, tail_items)
                          for p, g in zip(predictions, ground_truths)]
        novelty_vals = [novelty_at_k(p, item_popularity, max_pop, k) for p in predictions]

        metrics[f'NDCG@{k}'] = np.mean(ndcg_vals)
        metrics[f'Recall@{k}'] = np.mean(recall_vals)
        metrics[f'HR@{k}'] = np.mean(hr_vals)
        metrics[f'ILS@{k}'] = np.mean(ils_vals)
        metrics[f'Tail_Recall@{k}'] = np.mean(tail_recall_vals)
        metrics[f'Novelty@{k}'] = np.mean(novelty_vals)

    metrics['Coverage@10'] = coverage_at_k(predictions, len(catalog_ids), 10)
    metrics['OOD@10'] = ood_rate(predictions, catalog_ids, 10)

    return metrics
