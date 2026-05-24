"""
RecCL: Recommendation Curriculum Learning
Three-dimensional progressive curriculum for generative recommendation.

Dimensions:
  1. Sequence difficulty D_seq(u): 1/len + entropy of category diversity
  2. Item difficulty D_item(i): inverse popularity (cold items are harder)
  3. Prediction difficulty D_pred(u,i): 1 - CF model confidence

The sampler uses a continuous weighting scheme rather than hard stage switching.
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import WeightedRandomSampler


class DifficultyScorer:
    """Compute three-dimensional difficulty scores for all training samples."""

    def __init__(self, item_popularity: Dict[str, int], item_genres: Dict[str, List[str]],
                 cf_scores: Optional[Dict[Tuple[str, str], float]] = None):
        """
        Args:
            item_popularity: {item_id: interaction_count}
            item_genres: {item_id: [genre_list]}
            cf_scores: {(user_id, item_id): cf_confidence} from SASRec or similar
        """
        self.item_pop = item_popularity
        self.item_genres = item_genres
        self.cf_scores = cf_scores or {}

        max_pop = max(item_popularity.values()) if item_popularity else 1
        self.item_pop_norm = {k: v / max_pop for k, v in item_popularity.items()}

    def seq_difficulty(self, user_seq: List[str]) -> float:
        """D_seq: shorter + less diverse = easier. Returns [0, 1], higher = harder."""
        if len(user_seq) <= 1:
            return 0.0  # very short = very easy

        # Length component: 1 / len, normalized
        len_factor = min(1.0, 3.0 / len(user_seq))

        # Diversity component: entropy of genre distribution
        all_genres = []
        for item in user_seq:
            genres = self.item_genres.get(item, ['Unknown'])
            all_genres.extend(genres)
        if not all_genres:
            return len_factor

        genre_counts = Counter(all_genres)
        total = sum(genre_counts.values())
        probs = [c / total for c in genre_counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(genre_counts) + 1)
        entropy_norm = entropy / (max_entropy + 1e-10)

        return 0.5 * len_factor + 0.5 * entropy_norm

    def item_difficulty(self, item_id: str) -> float:
        """D_item: inverse popularity. [0, 1], higher = harder (colder item)."""
        norm_pop = self.item_pop_norm.get(item_id, 0.0)
        return 1.0 - norm_pop

    def pred_difficulty(self, user_id: str, item_id: str) -> float:
        """D_pred: CF uncertainty. [0, 1], higher = CF less confident -> harder."""
        score = self.cf_scores.get((user_id, item_id), None)
        if score is None:
            return 0.5  # default: moderate difficulty
        return 1.0 - score  # low confidence = high difficulty

    def compute_all(self, user_id: str, user_seq: List[str],
                    target_item: str) -> Dict[str, float]:
        """Compute all three difficulty scores for a sample."""
        return {
            'seq': self.seq_difficulty(user_seq),
            'item': self.item_difficulty(target_item),
            'pred': self.pred_difficulty(user_id, target_item),
        }


class CurriculumSampler:
    """
    Continuous curriculum scheduler with weighted random sampling.
    Early training: bias toward easy samples.
    Late training: bias toward hard samples.
    """

    def __init__(self, scorer: DifficultyScorer,
                 seq_weight: float = 0.33,
                 item_weight: float = 0.33,
                 pred_weight: float = 0.34,
                 warmup_ratio: float = 0.3,
                 transition_type: str = 'linear'):
        """
        Args:
            scorer: DifficultyScorer instance
            seq_weight: α weight for sequence difficulty
            item_weight: β weight for item difficulty
            pred_weight: γ weight for prediction difficulty
            warmup_ratio: fraction of total steps for easy→hard transition
            transition_type: 'linear', 'exponential', or 'sigmoid'
        """
        self.scorer = scorer
        self.seq_w = seq_weight
        self.item_w = item_weight
        self.pred_w = pred_weight
        self.warmup_ratio = warmup_ratio
        self.transition_type = transition_type

        self.total_steps = 0
        self.current_step = 0
        self.sample_scores: List[float] = []

    def _transition_factor(self) -> float:
        """Compute current transition factor ρ ∈ [0, 1].
        ρ = 0 → pure easy sampling, ρ = 1 → pure hard sampling.
        """
        if self.total_steps == 0:
            return 0.0
        progress = self.current_step / self.total_steps

        if progress >= 1.0:
            return 1.0

        if self.transition_type == 'linear':
            return min(1.0, progress / self.warmup_ratio)
        elif self.transition_type == 'exponential':
            return min(1.0, (progress / self.warmup_ratio) ** 2)
        elif self.transition_type == 'sigmoid':
            import math
            x = (progress - self.warmup_ratio / 2) / (self.warmup_ratio / 10)
            return 1.0 / (1.0 + math.exp(-x))
        return min(1.0, progress / self.warmup_ratio)

    def precompute_scores(self, samples: List[Dict]) -> List[float]:
        """Precompute combined difficulty score for all samples.

        Each sample dict: {user_id, sequence: [item_ids], target_item}
        """
        scores = []
        for s in samples:
            diffs = self.scorer.compute_all(
                s['user_id'], s['sequence'], s['target_item']
            )
            combined = (self.seq_w * diffs['seq'] +
                        self.item_w * diffs['item'] +
                        self.pred_w * diffs['pred'])
            scores.append(combined)
        self.sample_scores = scores
        return scores

    def get_sampling_weights(self) -> np.ndarray:
        """Get current sampling weights based on difficulty scores and transition factor."""
        if not self.sample_scores:
            raise ValueError("Call precompute_scores() first")

        rho = self._transition_factor()
        scores = np.array(self.sample_scores)

        # Early (ρ=0): weight ∝ exp(-difficulty) → easy samples preferred
        # Late  (ρ=1): weight ∝ exp(+difficulty) → hard samples preferred
        # Middle: linear interpolation between the two regimes
        easy_weights = np.exp(-2.0 * scores)
        hard_weights = np.exp(2.0 * scores)

        weights = (1 - rho) * easy_weights + rho * hard_weights
        weights = weights / weights.sum()  # normalize
        return weights

    def step(self):
        """Advance one training step."""
        self.current_step += 1

    def set_total_steps(self, total: int):
        self.total_steps = total

    def get_sampler(self, dataset_size: int) -> WeightedRandomSampler:
        """Create a PyTorch WeightedRandomSampler for the current step."""
        weights = self.get_sampling_weights()
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=dataset_size,
            replacement=True,
        )
