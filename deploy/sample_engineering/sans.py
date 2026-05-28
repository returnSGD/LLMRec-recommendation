"""
SANS: Semantic-Aware Negative Sampling
Three-tier negative sampling with LLM-generated hard negatives.

Tiers:
  Easy  (w=0.1): random items from different categories
  Medium (w=0.3): same category, different item (random)
  Hard  (w=0.6): LLM-generated semantically similar but mismatched items

Uses weighted InfoNCE loss: harder negatives contribute more to the gradient.
"""

import json
import os
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F


class HardNegativeGenerator:
    """Generate hard negatives via LLM API + embedding retrieval."""

    def __init__(self, llm_client, item_texts: Dict[str, str],
                 item_genres: Dict[str, List[str]],
                 item_embeddings: Optional[Dict[str, np.ndarray]] = None,
                 embedding_model=None,
                 cache_path: str = "data/cache/hard_negatives.json"):
        """
        Args:
            llm_client: LLM API client (Anthropic SDK with DeepSeek endpoint)
            item_texts: {item_id: "title | genres | tags | description"}
            item_genres: {item_id: [genre_list]}
            item_embeddings: precomputed text embeddings for items
            embedding_model: sentence-transformer model for retrieval
            cache_path: disk cache for LLM-generated hard negatives
        """
        self.llm = llm_client
        self.item_texts = item_texts
        self.item_genres = item_genres
        self.item_embeddings = item_embeddings or {}
        self.embedding_model = embedding_model
        self.cache_path = cache_path
        self.cache: Dict[str, List[str]] = {}
        self._load_cache()
        # Check if LLM is actually available (anthropic package installed)
        self._llm_available = False
        try:
            import anthropic
            self._llm_available = True
        except ImportError:
            pass

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _cache_key(self, item_id: str) -> str:
        return item_id

    def generate_hard_negatives(self, item_id: str, count: int = 4) -> List[str]:
        """Generate hard negative candidate item_ids for a given positive item.

        Steps:
        1. Check cache
        2. Try LLM to generate misleading-but-similar item descriptions
        3. Fallback: embedding-based retrieval (no LLM needed)
        4. Skip top-N closest (too similar), return next K
        """
        ck = self._cache_key(item_id)
        if ck in self.cache:
            return self.cache[ck][:count]

        if item_id not in self.item_texts:
            return []

        hard_neg_ids = []

        # Step 1: Try LLM-generated descriptions (only if anthropic is available)
        if self._llm_available:
            item_desc = self.item_texts[item_id]
            try:
                llm_descriptions = self._call_llm_for_hard_negatives(item_desc, count * 2)
                for desc in llm_descriptions:
                    retrieved = self._retrieve_similar_items(desc, top_k=10)
                    for rid in retrieved[3:]:
                        if rid != item_id and rid not in hard_neg_ids:
                            hard_neg_ids.append(rid)
                            break
                    if len(hard_neg_ids) >= count:
                        break
            except Exception:
                pass  # Fall through to embedding fallback

        # Step 2: Embedding-based fallback (no LLM needed)
        if not hard_neg_ids and self.item_embeddings and item_id in self.item_embeddings:
            hard_neg_ids = self._embedding_fallback(item_id, count)

        self.cache[ck] = hard_neg_ids
        self._save_cache()
        return hard_neg_ids[:count]

    def _embedding_fallback(self, item_id: str, count: int) -> List[str]:
        """Use embedding similarity to find hard negatives without LLM.
        Skip top-3 most similar (too close to positive), take next K."""
        if item_id not in self.item_embeddings:
            return []
        pos_emb = self.item_embeddings[item_id]
        sims = {}
        for iid, emb in self.item_embeddings.items():
            if iid != item_id:
                sims[iid] = float(np.dot(pos_emb, emb) /
                              (np.linalg.norm(pos_emb) * np.linalg.norm(emb) + 1e-10))
        sorted_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        # Skip top 3 (too similar), take next K
        return [iid for iid, _ in sorted_items[3:3 + count]]

    def _call_llm_for_hard_negatives(self, item_desc: str, count: int) -> List[str]:
        prompt = (
            f"A user purchased and enjoyed this item:\n{item_desc}\n\n"
            f"Generate {count} alternative item descriptions that:\n"
            f"1. Belong to the same genre/category\n"
            f"2. Have very similar core characteristics\n"
            f"3. BUT differ in at least one key aspect (theme, style, popularity, niche vs mainstream)\n\n"
            f"For each, provide one sentence describing the item."
        )
        response = self.llm.generate(prompt, max_tokens=256, temperature=0.7)
        # Parse response into individual descriptions
        lines = [l.strip().lstrip('0123456789.-) ') for l in response.split('\n') if l.strip()]
        return [l for l in lines if len(l) > 20]

    def _retrieve_similar_items(self, query_text: str, top_k: int = 10) -> List[str]:
        if not self.embedding_model or not self.item_embeddings:
            return []
        query_emb = self.embedding_model.encode([query_text])[0]
        similarities = {}
        for item_id, emb in self.item_embeddings.items():
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)
            similarities[item_id] = sim
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:top_k]]


class LayeredNegativeSampler:
    """Three-tier negative sampling with per-tier weights."""

    def __init__(self, easy_pool: List[str], medium_pools: Dict[str, List[str]],
                 hard_generator: Optional[HardNegativeGenerator] = None,
                 easy_count: int = 8, medium_count: int = 4, hard_count: int = 4,
                 easy_weight: float = 0.1, medium_weight: float = 0.3,
                 hard_weight: float = 0.6, temperature: float = 0.07):
        """
        Args:
            easy_pool: all item IDs (for random easy negative sampling)
            medium_pools: {genre: [item_ids]} for same-category negatives
            hard_generator: HardNegativeGenerator instance
            easy_count, medium_count, hard_count: number per tier
            easy_weight, medium_weight, hard_weight: loss weights per tier
            temperature: InfoNCE temperature τ
        """
        self.easy_pool = easy_pool
        self.medium_pools = medium_pools
        self.hard_gen = hard_generator
        self.K_easy = easy_count
        self.K_medium = medium_count
        self.K_hard = hard_count
        self.w_easy = easy_weight
        self.w_medium = medium_weight
        self.w_hard = hard_weight
        self.tau = temperature

    def sample_negatives(self, positive_id: str, batch_size: int = 1) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Sample negatives and return (neg_ids, weights) tensors.

        Returns:
            neg_ids: [batch, total_neg] tensor of item indices
            weights: [batch, total_neg] tensor of per-negative weights
        """
        # This is a simplified batch interface; actual sampling integrates with the dataloader
        all_neg_ids = []
        all_weights = []

        # Easy negatives: random from different categories
        easy_pool_filtered = [i for i in self.easy_pool if i != positive_id]
        easy_negs = list(np.random.choice(
            easy_pool_filtered,
            size=min(self.K_easy, len(easy_pool_filtered)),
            replace=False
        ))
        all_neg_ids.extend(easy_negs)
        all_weights.extend([self.w_easy] * len(easy_negs))

        # Medium negatives: same category, different item
        medium_pool = self._get_medium_pool(positive_id)
        medium_negs = list(np.random.choice(
            medium_pool,
            size=min(self.K_medium, len(medium_pool)),
            replace=False
        ))
        all_neg_ids.extend(medium_negs)
        all_weights.extend([self.w_medium] * len(medium_negs))

        # Hard negatives: LLM-generated (skip if no hard negatives requested)
        if self.hard_gen and self.K_hard > 0:
            hard_negs = self.hard_gen.generate_hard_negatives(positive_id, self.K_hard)
        else:
            hard_negs = []
        all_neg_ids.extend(hard_negs)
        all_weights.extend([self.w_hard] * len(hard_negs))

        return all_neg_ids, all_weights

    def _get_medium_pool(self, positive_id: str) -> List[str]:
        """Get same-category items for medium negatives."""
        # Try to match by shared genre; fall back to random subset
        for pool_name, pool_items in self.medium_pools.items():
            if positive_id in pool_items:
                return [i for i in pool_items if i != positive_id]
        return [i for i in self.easy_pool if i != positive_id]

    def get_negatives_batch(self, positive_ids: List[str]) -> Tuple[
            List[List[str]], List[List[float]]]:
        """Batch version of sample_negatives."""
        all_neg_ids = []
        all_weights = []
        for pid in positive_ids:
            nids, wts = self.sample_negatives(pid)
            all_neg_ids.append(nids)
            all_weights.append(wts)
        return all_neg_ids, all_weights


def weighted_infonce_loss(query_emb: torch.Tensor, pos_emb: torch.Tensor,
                          neg_embs: torch.Tensor, neg_weights: torch.Tensor,
                          temperature: float = 0.07) -> torch.Tensor:
    """Weighted InfoNCE loss for SANS (log-space, fp16-safe).

    L = -log[ exp(s(q,i+)/τ) / (exp(s(q,i+)/τ) + Σ w_k * exp(s(q,i_k-)/τ)) ]

    Reformulated via log-sum-exp to keep all exp arguments ≤ 0,
    preventing fp16 overflow when cos_sim / τ exceeds ~11 (fp16 max ≈ 65504).

    Args:
        query_emb: [batch, dim] user/sequence representation
        pos_emb: [batch, dim] positive item embedding
        neg_embs: [batch, num_neg, dim] negative item embeddings
        neg_weights: [batch, num_neg] per-negative weights
        temperature: τ
    """
    device = query_emb.device

    pos_sim = F.cosine_similarity(query_emb, pos_emb, dim=-1) / temperature  # [B]
    neg_sim = torch.bmm(neg_embs, query_emb.unsqueeze(-1)).squeeze(-1) / temperature  # [B, N]

    # Log-space: L = -(pos_sim - log_denom)
    # log_denom = log(exp(pos_sim) + Σ w_k * exp(neg_sim_k))
    #
    # Using log-sum-exp with shift by max_sim:
    #   log_denom = max_sim + log(exp(pos_sim - max_sim) + Σ w_k * exp(neg_sim_k - max_sim))
    #
    # Since pos_sim - max_sim ≤ 0 and neg_sim_k - max_sim ≤ 0,
    # all exp() outputs are ≤ 1, safe in fp16.
    all_sims = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, 1+N]
    max_sim = all_sims.max(dim=-1, keepdim=True).values  # [B, 1]

    shifted_pos = torch.exp(pos_sim.unsqueeze(-1) - max_sim)  # [B, 1], ≤1
    shifted_neg = neg_weights.to(device) * torch.exp(neg_sim - max_sim)  # [B, N], ≤1

    log_denom = max_sim.squeeze(-1) + torch.log(
        shifted_pos.squeeze(-1) + shifted_neg.sum(dim=-1) + 1e-10
    )  # [B]

    loss = -(pos_sim - log_denom).mean()
    return loss
