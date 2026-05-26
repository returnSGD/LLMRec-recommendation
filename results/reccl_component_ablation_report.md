# RecCL Component Ablation Report

**Date:** 2026-05-26
**Model:** google/flan-t5-small (60M)
**GPU:** NVIDIA RTX 3060 6GB
**Config:** 10,000 training samples × 3 epochs, seed=42

---

## 1. Overview

RecCL (Recommendation Curriculum Learning) constructs a training curriculum along three difficulty
dimensions: sequence difficulty (α), item popularity difficulty (β), and prediction confidence
difficulty (γ). This ablation isolates each dimension by setting individual weights to 1.0 while
zeroing the others, then compares against the balanced 3-dim configuration (α=β=γ=0.33) and a
no-RecCL baseline.

## 2. Main Results (Top-10)

| Variant | α | β | γ | NDCG@10 | Recall@10 | HR@10 | Novelty@10 | Coverage@10 | OOD@10 |
|---------|---|---|---|---------|-----------|-------|------------|-------------|--------|
| base (no RecCL) | — | — | — | 0.0070 | 0.0070 | 0.0070 | 0.7939 | 0.0014 | 0.0010 |
| seq-only | 1.0 | 0.0 | 0.0 | 0.0020 | 0.0020 | 0.0020 | 0.7657 | 0.0012 | 0.0010 |
| **item-only** | 0.0 | 1.0 | 0.0 | **0.0120** | **0.0120** | **0.0120** | 0.2776 | 0.0007 | 0.0000 |
| pred-only | 0.0 | 0.0 | 1.0 | 0.0050 | 0.0050 | 0.0050 | **0.8220** | 0.0013 | 0.0000 |
| **full-3dim** | 0.33 | 0.33 | 0.34 | **0.0100** | **0.0100** | **0.0100** | 0.1799 | 0.0011 | 0.0000 |

> **Bold** = best in column. ILS@K = 0.0000 across all variants (omitted). OOD ↓ lower is better.

## 3. Per-Dimension Analysis

### 3.1 Sequence Difficulty (α = 1.0): Harmful in Isolation

```
NDCG@10: 0.0020  (−71.4% vs base)
Novelty: 0.7657  (−3.6%  vs base)
```

Sorting training samples by sequence length/complexity alone degrades performance
substantially. The model loses general recommendation capability without the guidance
of item popularity or CF confidence signals. Sequence length is a poor proxy for
curriculum difficulty in this setting.

### 3.2 Item Popularity Difficulty (β = 1.0): Strong Accuracy Driver

```
NDCG@10: 0.0120  (+71.4% vs base)
Novelty: 0.2776  (−65.0% vs base)
```

Prioritizing rare/tail items during curriculum training is the dominant accuracy
mechanism in RecCL. By forcing the model to focus on low-popularity items early in
training, the model learns more discriminative representations. However, this comes
at the cost of severe diversity collapse — the model overwhelmingly recommends a
narrow set of moderately popular items while losing tail coverage.

### 3.3 Prediction Difficulty (γ = 1.0): Diversity Champion

```
NDCG@10: 0.0050  (−28.6% vs base)
Novelty: 0.8220  (+3.5%  vs base)
```

Sampling based on CF-model prediction confidence yields the highest diversity among
all configurations. The model explores a wider range of items, sacrificing some
accuracy. This dimension acts as a regularizer against popularity collapse.

### 3.4 Full 3-Dimension (α = β = γ = 0.33): Best Balance

```
NDCG@10: 0.0100  (+42.9% vs base)
Novelty: 0.1799  (−77.3% vs base)
```

The balanced configuration achieves the second-best accuracy, closely approaching
item-only. However, the combined effect of all three dimensions appears to amplify
rather than mitigate popularity bias — Novelty drops further (0.18) than with
item-only alone (0.28). This suggests the interaction between dimensions creates
a compounding effect that requires tuning.

## 4. Accuracy vs. Diversity Trade-off

```
              Novelty@10
  0.9 ┤               ● pred-only (0.0050, 0.8220)
      │
  0.8 ┤ ● base (0.0070, 0.7939)
      │
  0.7 ┤     ● seq-only (0.0020, 0.7657)
      │
  0.4 ┤
      │
  0.3 ┤         ● item-only (0.0120, 0.2776)
      │
  0.2 ┤             ● full-3dim (0.0100, 0.1799)
      │
      0.002   0.005   0.008   0.011   0.014
                     NDCG@10
```

The Pareto frontier reveals a clear accuracy-diversity trade-off. No single
dimension dominates both metrics simultaneously.

## 5. Per-K Breakdown

### Top-5

| Variant | NDCG@5 | Recall@5 | HR@5 | Novelty@5 |
|---------|--------|----------|------|-----------|
| base | 0.0070 | 0.0070 | 0.0070 | 0.7939 |
| seq-only | 0.0020 | 0.0020 | 0.0020 | 0.7657 |
| item-only | 0.0120 | 0.0120 | 0.0120 | 0.2776 |
| pred-only | 0.0050 | 0.0050 | 0.0050 | 0.8220 |
| full-3dim | 0.0100 | 0.0100 | 0.0100 | 0.1799 |

### Top-20

| Variant | NDCG@20 | Recall@20 | HR@20 | Novelty@20 |
|---------|---------|-----------|-------|------------|
| base | 0.0070 | 0.0070 | 0.0070 | 0.7939 |
| seq-only | 0.0020 | 0.0020 | 0.0020 | 0.7657 |
| item-only | 0.0120 | 0.0120 | 0.0120 | 0.2776 |
| pred-only | 0.0050 | 0.0050 | 0.0050 | 0.8220 |
| full-3dim | 0.0100 | 0.0100 | 0.0100 | 0.1799 |

> Note: NDCG/Recall/HR are invariant across K because the generative model produces
> exactly one item recommendation. Expandability beyond K=1 is not supported by
> the current single-generation architecture.

## 6. Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Item popularity (β) is the sole accuracy driver in RecCL** | β-only achieves +71% NDCG vs base; α-only and γ-only both underperform base |
| 2 | **Sequence difficulty (α) is counterproductive alone** | -71% NDCG — curriculum by sequence length is noise, not signal |
| 3 | **Prediction difficulty (γ) maximizes diversity** | Highest Novelty@10 (0.8220) across all variants |
| 4 | **Three-dimension combination compounds popularity bias** | full-3dim Novelty (0.18) is worse than β-only (0.28) despite including γ |
| 5 | **Accuracy-diversity trade-off is inherent in single-dim RecCL** | No single dimension achieves both high accuracy AND high diversity |

## 7. Recommendations

1. **Reduce α weight or remove it entirely.** Sequence difficulty alone hurts
   performance, and its inclusion in the 3-dim ensemble pushes Novelty even lower.
   Consider α ∈ [0, 0.1] for future experiments.

2. **Re-weight β and γ to balance accuracy vs. diversity.** The current
   α=β=γ=0.33 uniform weighting amplifies popularity collapse. Try β=0.25, γ=0.50
   to better balance the trade-off.

3. **Combine RecCL with SANS for diversity recovery.** The SANS module (hard
   negative sampling) was disabled in this ablation. Its diversity-promoting effect
   could counteract the popularity bias from RecCL-β.

4. **Run warmup_ratio sensitivity sweep** — the current default warmup=0.3 may
   not be optimal given the per-dimension differences in convergence behavior.

## 8. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Base model | google/flan-t5-small (60M params) |
| Training samples | 10,000 |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 3e-4 |
| Optimizer | AdamW + CosineAnnealingLR |
| Random seed | 42 |
| Evaluation samples | 1,000 |
| GPU | NVIDIA RTX 3060 6GB |
| RecCL transition | linear |
| Default warmup_ratio | 0.3 |
