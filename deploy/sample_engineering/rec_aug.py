"""
RecAug: Recommendation-Specific Semantic Augmentation
Three augmentation operations that preserve recommendation semantics.

  1. Intent-Preserving Truncation: remove semantically redundant items
  2. Session-Boundary-Aware Permutation: shuffle session blocks, keep intra-session order
  3. LLM-Guided Item Substitution: replace items with same-intent alternatives

Includes an adaptive strategy selector based on sequence "redundancy" score.
"""

import random
import hashlib
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


class SessionBoundaryDetector:
    """Detect session boundaries using time gaps + optional LLM genre-shift analysis."""

    def __init__(self, time_gap_hours: float = 72.0, llm_client=None):
        """
        Args:
            time_gap_hours: threshold in hours for time-based session split
            llm_client: optional LLM for genre-shift-based boundary refinement
        """
        self.gap_threshold = time_gap_hours * 3600  # convert to seconds
        self.llm = llm_client

    def detect(self, timestamps: List[float], item_genres: Optional[List[List[str]]] = None,
               user_id: Optional[str] = None) -> List[int]:
        """Detect session boundary indices.

        Args:
            timestamps: list of unix timestamps (or playtime hours as proxy)
            item_genres: optional genre tags for each item
            user_id: optional user_id for LLM-based detection

        Returns:
            list of boundary indices (the index AFTER which a new session starts)
            e.g., [2, 5] means sessions are items[0:2], items[2:5], items[5:]
        """
        if not timestamps or len(timestamps) < 2:
            return []

        # Primary: gap-based detection (works with both timestamps and playtimes)
        boundaries = []
        values = [float(t) for t in timestamps]
        median_gap = np.median([abs(values[i] - values[i-1]) for i in range(1, len(values))]) if len(values) > 1 else 0
        threshold = max(self.gap_threshold, median_gap * 3) if median_gap > 0 else self.gap_threshold

        for i in range(1, len(values)):
            gap = abs(values[i] - values[i - 1])
            if gap > threshold:
                boundaries.append(i)

        # If no boundaries found with data, create artificial ones for augmentation
        if not boundaries and len(values) >= 6:
            # Split into 2-3 roughly equal sessions
            n_sessions = random.randint(2, 3)
            session_size = len(values) // n_sessions
            for s in range(1, n_sessions):
                boundaries.append(s * session_size)

        # Secondary: LLM genre-shift refinement (if available)
        if self.llm and item_genres and 0 < len(boundaries) < len(timestamps) // 3:
            refined = self._llm_refine_boundaries(timestamps, item_genres, boundaries)
            if refined:
                boundaries = refined

        return boundaries

    def _llm_refine_boundaries(self, timestamps: List[float],
                                item_genres: List[List[str]],
                                current_boundaries: List[int]) -> Optional[List[int]]:
        """Use LLM to refine boundary detection."""
        timeline_lines = []
        for i, (ts, genres) in enumerate(zip(timestamps, item_genres)):
            marker = " ← BOUNDARY" if i in current_boundaries else ""
            timeline_lines.append(f"  {i}: genres={genres}{marker}")
        timeline_str = '\n'.join(timeline_lines[:30])  # limit length

        prompt = (
            f"A user's activity timeline:\n{timeline_str}\n\n"
            f"Identify where the user's activity sessions naturally divide. "
            f"Mark session boundaries with their indices. Only return indices, "
            f"one per line, e.g.: 3\n7\n12"
        )
        try:
            response = self.llm.generate(prompt, max_tokens=100, temperature=0.3)
            boundaries = []
            for line in response.strip().split('\n'):
                try:
                    boundaries.append(int(line.strip()))
                except ValueError:
                    continue
            return sorted(set(boundaries))
        except Exception:
            return None


class IntentPreservingTruncation:
    """Remove semantically redundant items while preserving intent-shift items."""

    def __init__(self, llm_client, item_texts: Dict[str, str],
                 cache_path: str = "data/cache/item_intents.json"):
        self.llm = llm_client
        self.item_texts = item_texts
        self.cache_path = cache_path
        self.intent_cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.intent_cache = json.load(f)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.intent_cache, f, ensure_ascii=False, indent=2)

    def get_intent(self, item_id: str) -> str:
        """Get or generate user intent label for an item."""
        if item_id in self.intent_cache:
            return self.intent_cache[item_id]
        if item_id not in self.item_texts:
            return "unknown"
        prompt = (
            f"The user interacted with this item:\n{self.item_texts[item_id]}\n\n"
            f"In one short phrase (max 5 words), describe what kind of "
            f"need or intent this interaction satisfies."
        )
        try:
            intent = self.llm.generate(prompt, max_tokens=30, temperature=0.3).strip()
        except Exception:
            intent = "unknown"
        self.intent_cache[item_id] = intent
        self._save_cache()
        return intent

    def truncate(self, sequence: List[str], max_removal_ratio: float = 0.3) -> List[str]:
        """Truncate sequence by removing semantically redundant items.

        Algorithm:
        1. Get intent for each item
        2. Group consecutive items with same/similar intent
        3. Keep first and last of each intent group (entry + exit points)
        4. Keep all items at "intent shift" boundaries
        """
        if len(sequence) <= 3:
            return sequence

        intents = [self.get_intent(item) for item in sequence]
        max_remove = max(1, int(len(sequence) * max_removal_ratio))

        # Identify intent groups (consecutive same-intent items)
        groups = []
        current_group = [0]
        for i in range(1, len(intents)):
            if intents[i] == intents[i - 1]:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        groups.append(current_group)

        # For each group > 2 items, keep first + last, mark middle for removal
        keep_mask = [True] * len(sequence)
        removal_candidates = []
        for group in groups:
            if len(group) > 2:
                for idx in group[1:-1]:
                    removal_candidates.append(idx)

        # Remove up to max_removal_ratio
        removal_candidates = removal_candidates[:max_remove]
        for idx in removal_candidates:
            keep_mask[idx] = False

        return [item for i, item in enumerate(sequence) if keep_mask[i]]


class SessionPermutation:
    """Randomly permute session blocks while preserving intra-session order."""

    def __init__(self, boundary_detector: SessionBoundaryDetector):
        self.detector = boundary_detector

    def permute(self, sequence: List[str], timestamps: List[float],
                item_genres: Optional[List[List[str]]] = None,
                user_id: Optional[str] = None) -> List[str]:
        """Shuffle session order, keep intra-session order intact."""
        if len(sequence) <= 3:
            return sequence

        boundaries = self.detector.detect(timestamps, item_genres, user_id)

        if not boundaries or len(boundaries) < 1:
            return sequence  # single session, no permutation needed

        # Split into sessions
        sessions = []
        start = 0
        for b in boundaries:
            sessions.append(sequence[start:b])
            start = b
        sessions.append(sequence[start:])

        if len(sessions) <= 1:
            return sequence

        # Shuffle session order
        random.shuffle(sessions)
        return [item for session in sessions for item in session]


class LLMGuidedSubstitution:
    """LLM-guided item substitution: replace with same-intent alternatives."""

    def __init__(self, llm_client, item_texts: Dict[str, str],
                 item_embeddings: Dict[str, np.ndarray],
                 embedding_model,
                 intent_cache: Dict[str, str],
                 cache_path: str = "data/cache/item_substitutions.json"):
        self.llm = llm_client
        self.item_texts = item_texts
        self.item_embeddings = item_embeddings
        self.embedding_model = embedding_model
        self.intent_cache = intent_cache
        self.cache_path = cache_path
        self.sub_cache: Dict[str, List[str]] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self.sub_cache = json.load(f)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.sub_cache, f, ensure_ascii=False, indent=2)

    def substitute(self, item_id: str, intent: str) -> Optional[str]:
        """Get a same-intent substitute for an item. Returns None if no good substitute."""
        ck = f"{item_id}|||{intent}"
        if ck in self.sub_cache:
            candidates = self.sub_cache[ck]
            if candidates:
                return random.choice(candidates)
            return None

        if item_id not in self.item_texts:
            return None

        prompt = (
            f"The user's intent is: {intent}\n"
            f"Current item: {self.item_texts[item_id]}\n\n"
            f"Suggest 5 alternative items that would satisfy the SAME intent. "
            f"Provide item descriptions, one per line."
        )
        try:
            response = self.llm.generate(prompt, max_tokens=200, temperature=0.7)
            descriptions = [l.strip().lstrip('0123456789.-) ') for l in response.split('\n')
                          if len(l.strip()) > 20]
        except Exception:
            return None

        # Retrieve closest real items
        candidates = []
        for desc in descriptions[:3]:
            retrieved = self._retrieve(desc, top_k=5)
            for rid in retrieved:
                if rid != item_id and rid not in candidates:
                    candidates.append(rid)
                    break

        self.sub_cache[ck] = candidates
        self._save_cache()
        return random.choice(candidates) if candidates else None

    def _retrieve(self, query_text: str, top_k: int = 5) -> List[str]:
        if not self.embedding_model or not self.item_embeddings:
            return []
        query_emb = self.embedding_model.encode([query_text])[0]
        sims = {}
        for iid, emb in self.item_embeddings.items():
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)
            sims[iid] = sim
        return [iid for iid, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]]


class RecAugPipeline:
    """Orchestrate all three RecAug operations with adaptive strategy selection.

    active_ops: subset of ['trunc', 'perm', 'sub'] to enable.
                None means all three enabled.
    """

    def __init__(self, truncation: IntentPreservingTruncation,
                 permutation: SessionPermutation,
                 substitution: LLMGuidedSubstitution,
                 substitution_prob: float = 0.2,
                 active_ops: Optional[List[str]] = None):
        self.truncation = truncation
        self.permutation = permutation
        self.substitution = substitution
        self.sub_prob = substitution_prob
        self.active_ops = active_ops  # None = all enabled

    def _op_enabled(self, op: str) -> bool:
        return self.active_ops is None or op in self.active_ops

    def _redundancy_score(self, sequence: List[str]) -> float:
        """Estimate sequence redundancy. Falls back to length heuristic without LLM."""
        if len(sequence) <= 2:
            return 0.0
        try:
            intents = [self.truncation.get_intent(item) for item in sequence]
            consecutive_same = sum(1 for i in range(1, len(intents)) if intents[i] == intents[i-1])
            return consecutive_same / (len(sequence) - 1)
        except Exception:
            return min(0.5, len(sequence) / 100.0)

    def augment(self, sequence: List[str], timestamps: Optional[List[float]] = None,
                item_genres: Optional[List[List[str]]] = None,
                user_id: Optional[str] = None) -> List[Dict[str, any]]:
        """Generate 2-3 augmented variants of the sequence.

        Returns:
            list of dicts: [{'sequence': [...], 'operations': ['truncate', 'permute']}, ...]
        """
        variants = []

        # --- Truncation (LLM intent-preserving, or simple random-drop fallback) ---
        if self._op_enabled('trunc'):
            try:
                truncated = self.truncation.truncate(sequence)
            except Exception:
                truncated = self._simple_truncate(sequence)
            if truncated != sequence:
                variants.append({'sequence': truncated, 'operations': ['truncate']})

        # --- Session permutation (timestamps or playtime-based) ---
        if self._op_enabled('perm'):
            if timestamps and len(timestamps) == len(sequence):
                permuted = self.permutation.permute(
                    sequence, timestamps, item_genres, user_id
                )
                if permuted != sequence:
                    variants.append({'sequence': permuted, 'operations': ['permute']})

        # --- LLM-guided substitution ---
        if self._op_enabled('sub') and self.substitution:
            if random.random() < self.sub_prob:
                try:
                    subbed = self._apply_substitutions(sequence)
                    if subbed != sequence:
                        variants.append({'sequence': subbed, 'operations': ['substitute']})
                except Exception:
                    pass

        # Fallback: at least one variant (simple random drop)
        if not variants and len(sequence) > 2:
            dropped = self._simple_truncate(sequence, max_removal_ratio=0.15)
            variants.append({'sequence': dropped, 'operations': ['simple_drop']})

        return variants[:3]

    def _simple_truncate(self, sequence: List[str], max_removal_ratio: float = 0.3) -> List[str]:
        """Random item drop fallback — no LLM needed."""
        if len(sequence) <= 3:
            return sequence
        max_remove = max(1, int(len(sequence) * max_removal_ratio))
        n_remove = random.randint(1, max_remove)
        indices = sorted(random.sample(range(len(sequence)), n_remove))
        return [item for i, item in enumerate(sequence) if i not in indices]

    def _apply_substitutions(self, sequence: List[str]) -> List[str]:
        """Apply LLM-guided substitutions to some items in the sequence."""
        result = []
        for item in sequence:
            if random.random() < self.sub_prob:
                intent = self.truncation.get_intent(item)
                sub = self.substitution.substitute(item, intent)
                result.append(sub if sub else item)
            else:
                result.append(item)
        return result
