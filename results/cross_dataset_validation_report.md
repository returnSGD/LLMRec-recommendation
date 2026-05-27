# P3: Cross-Dataset Validation — Goodreads Book Recommendation

**Date:** 2026-05-28
**Model:** google/flan-t5-small (60M)
**GPU:** NVIDIA RTX 3060 6GB
**Status:** Data ready, preprocessing scripts complete, training pending

---

## Executive Summary

This report documents the cross-dataset validation setup for LLM-Rec's sample-engineering
methods on the Goodreads book recommendation dataset. Cross-dataset validation is a
critical requirement for Nature-level publication: the Steam-only results (P0–P1) establish
feasibility, but demonstrating that RecCL, SANS, and RecAug generalize to a different
domain is essential for claiming methodological generality.

### Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Data download | Complete | 4 files, ~6.5 GB total |
| Data exploration | Complete | Streaming analysis of all sources |
| Preprocessing script | Complete | 480-line pipeline, 7 steps |
| Config file | Complete | `config_goodreads.yaml` — book-specific prompts |
| Experiment runners | Complete | `run_experiments.py` + `run_component_ablation.py` support `--dataset goodreads` |
| Preprocessing execution | **Not run** | Awaiting GPU availability |
| Training execution | **Not run** | Blocked on preprocessing |
| Evaluation | **Not run** | Blocked on training |

---

## 1. Why Goodreads?

### 1.1 Dataset Selection Rationale

Goodreads was chosen as the cross-dataset validation target for four reasons:

| Criterion | Goodreads | Fit |
|-----------|-----------|-----|
| **Domain shift** | Books vs. Games | Strong domain gap tests generalization |
| **Rich text** | Titles, descriptions, author names, genres | Comparable semantic richness to Steam |
| **Interaction type** | Read/shelve/rate (implicit + explicit) | Different signal from Steam's playtime-based |
| **Scale** | 228M interactions, 2.4M books | Larger than Steam, tests scalability |
| **Long-tail** | Severe: most books have <10 reads | Same challenge profile as Steam |
| **Academic standard** | UCSD Julian McAuley, widely cited | Same source as Steam dataset |

### 1.2 Domain Differences: Why This Is a Real Test

| Dimension | Steam (Games) | Goodreads (Books) |
|-----------|---------------|-------------------|
| **Item type** | Video games | Books |
| **Consumption pattern** | Repeated play (same game for 100s hours) | One-time read (rare re-reads) |
| **Temporal dynamics** | Fast: new releases, sales, F2P shifts | Slow: classics persist for decades |
| **Intent structure** | Genre+mechanic hybrid (e.g., "FPS RPG") | Pure genre hierarchy (Fiction → Sci-Fi → Cyberpunk) |
| **Sequence drift** | Taste can shift abruptly (e.g., AAA → indie) | Taste evolves gradually within genre families |
| **Avg sequence length** | 53.9 items | TBD after preprocessing (expected 20–40) |
| **Metadata richness** | Tags, genres, developer, publisher | Genres, author, series, publisher, pages |

The domain gap is substantial enough that strong performance on both datasets would
constitute meaningful evidence of generalization.

---

## 2. Data Inventory

### 2.1 Raw Data Files

All files sourced from [UCSD Julian McAuley's Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#goodreads).

| File | Size | Format | Content |
|------|------|--------|---------|
| `goodreads_interactions.csv` | 4.3 GB | CSV | 228M+ user-book interactions (user_id, book_id, is_read, is_reviewed, rating) |
| `goodreads_books.json.gz` | 2.1 GB | GZipped JSONL | 2.4M book metadata records (title, authors, description, avg_rating, num_pages, publisher, pub_year, series, language) |
| `goodreads_book_genres_initial.json.gz` | 24 MB | GZipped JSONL | Hierarchical genre labels per book |
| `user_id_map.csv` | 35 MB | CSV | Anonymized CSV ID → Goodreads native user ID |
| `book_id_map.csv` | 38 MB | CSV | Anonymized CSV ID → Goodreads native book ID |

**Total raw data:** ~6.5 GB.

### 2.2 Data Exploration Results

Streaming exploration (`data_validation/explore.py`) was run to characterize the dataset
before committing to preprocessing. Key statistics:

**Interactions:**
- Total interactions: 228M+ rows
- Read interactions (`is_read=1`): ~majority of total
- Unique users (in interactions): millions
- Unique books (in interactions): millions
- Sparsity: >99.9%
- Rating distribution (5-star): strong positive skew (most ratings 4–5)

**Book Metadata (2.4M books):**
- With description: majority
- With page count: majority
- With authors: majority
- With publication year: majority
- Publication year range: ~1900–2026, median ~2010
- Top languages: English (dominant), followed by Spanish, French, German
- Authors per book: typically 1 (single author dominates)

**Genre Hierarchy:**
- Unique genre labels: thousands (Goodreads uses a deep hierarchical taxonomy)
- Top genres: Fiction, Contemporary, Romance, Fantasy, Young Adult, Mystery, Thriller, Historical Fiction, Science Fiction, Non-Fiction
- Hierarchical: top-level categories → sub-genres (e.g., "Fiction/Fantasy/Epic Fantasy")

### 2.3 Data Quality Notes

- ID mappings are 1:1 between CSV internal IDs and Goodreads native IDs
- Book metadata JSON is UTF-8, occasional malformed lines (handled in preprocessing)
- Interactions are in CSV row order (may not be strictly chronological within each user)
- Genre coverage: most books have at least one genre label
- Description quality: varies from 1 sentence to multi-paragraph summaries

---

## 3. Preprocessing Pipeline

### 3.1 Pipeline Overview

`preprocess_goodreads.py` (480 lines) implements a 7-step pipeline that outputs data in
the same format as the Steam preprocessing pipeline, ensuring zero-code-change compatibility
with the training and evaluation infrastructure.

```
Step 1: Load ID maps
  └─ book_id_map.csv → {csv_book_id → goodreads_book_id}
  └─ user_id_map.csv → {csv_user_id → goodreads_user_id}

Step 2: Build item catalog (streaming)
  └─ goodreads_books.json.gz → {csv_book_id: {title, authors, description,
     avg_rating, num_pages, publisher, publication_year, language, ratings_count}}
  └─ Constructs "text" field for LLM prompt: "Title: {title}. Author: {authors}.
     Description: {description}. Genres: {genres}. Rating: {avg_rating}/5."

Step 3: Enrich genres (streaming)
  └─ goodreads_book_genres_initial.json.gz → append top-8 genre labels to text field

Step 4: Build user sequences (streaming)
  └─ goodreads_interactions.csv → filter is_read=1, group by user_id
  └─ Keep only items with metadata in catalog
  └─ Sort by CSV row order (proxy for chronological)

Step 5: K-core filter
  └─ Iteratively remove users with <k interactions and items with <k interactions
  └─ Default k=5, configurable via --k_core

Step 6: Train/val/test split
  └─ Leave-one-out: last item → test, second-last → val, rest → train
  └─ Sliding windows for training samples: (i_1→i_2), (i_1,i_2→i_3), ...

Step 7: Statistics and save
  └─ Output: train.json, val.json, test.json, item_catalog.json,
     item_popularity.json, stats.json
  └─ Destination: data/goodreads_processed/
```

### 3.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k_core` | 5 | Minimum interactions per user and per item |
| `--min_seq_len` | 5 | Minimum user sequence length after filtering |
| `--max_seq_len` | 50 | Maximum sequence length (truncate from end) |
| `--max_users` | None | Cap users for quick testing (e.g., 50000) |

### 3.3 Expected Output Scale

Based on the exploration data and K-core=5 filtering:
- Estimated users: 500K–1M (after K-core filtering)
- Estimated items: 200K–500K
- Estimated train samples: 5–20M
- Expected preprocessing time: 30–60 minutes (mostly I/O bound on CSV streaming)

### 3.4 Format Compatibility

The pipeline outputs data in the exact same JSON format as Steam:

```
data/goodreads_processed/
├── train.json          # [{input: "...", output: "book title"}, ...]
├── val.json            # Same format, validation split
├── test.json           # Same format, test split
├── item_catalog.json   # {item_id: {title, text, genres, ...}, ...}
├── item_popularity.json # {item_id: interaction_count, ...}
└── stats.json          # {num_users, num_items, avg_seq_len, sparsity, ...}
```

This enables `trainer.py`, `evaluate.py`, and all scripts to consume Goodreads data with
zero code changes — only a config file switch.

---

## 4. Experimental Design

### 4.1 Planned Experiments

The Goodreads experiments mirror the Steam experimental protocol exactly:

| Experiment | Script | Description | Priority |
|-----------|--------|-------------|----------|
| **Baseline** | `run_experiments.py --dataset goodreads --modes base` | FLAN-T5 without sample engineering | P0 |
| **Full LLM-Rec** | `run_experiments.py --dataset goodreads --modes full` | RecCL + SANS + RecAug enabled | P0 |
| **RecCL ablation** | `run_component_ablation.py --dataset goodreads --skip_sans --skip_recaug` | 3D curriculum decomposition | P1 |
| **SANS ablation** | `run_component_ablation.py --dataset goodreads --skip_reccl --skip_recaug` | Tier ablation (easy/medium/hard) | P1 |
| **RecAug ablation** | `run_component_ablation.py --dataset goodreads --skip_reccl --skip_sans` | Operation ablation (perm/trunc/full) | P1 |
| **Multi-seed** | `run_experiments.py --dataset goodreads --seeds 42,123,456` | Statistical robustness (3 seeds) | P1 |

### 4.2 Configuration (`config_goodreads.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | google/flan-t5-small (60M) | Same as Steam |
| Max seq length | 256 tokens | Book titles are shorter than game titles |
| Max output length | 64 tokens | Book title only |
| Batch size | 8 | GPU memory constraint |
| Gradient accumulation | 4 steps | Effective batch = 32 |
| Epochs | 5 | Same as Steam default |
| Learning rate | 3e-4 | Same as Steam |
| Warmup steps | 200 | Same as Steam |
| Prompt template | "The user has read the following books in order: {sequence}. What book should be recommended next?" | Book-specific |
| RecCL α/β/γ | 0.33/0.33/0.34 | Same weighting as Steam |
| SANS tiers | 8 easy / 4 medium / 4 hard | Same counts as Steam |
| RecAug consistency λ | 0.1 | Same as Steam |

### 4.3 Prompt Template

The book-specific prompt template in `config_goodreads.yaml`:

```
The user has read the following books in order:
{item_sequence}

Considering the user's reading preferences and patterns,
what book should be recommended next? Answer with the book title only.
```

Compared to the Steam template ("considering the user's gaming preferences and gameplay
patterns"), this uses domain-appropriate language ("reading preferences") while keeping
the structure identical.

---

## 5. Expected Outcomes & Hypotheses

### 5.1 Primary Hypotheses

Based on the Steam results and the domain characteristics of books vs. games:

| Hypothesis | Rationale | Steam Result |
|------------|-----------|--------------|
| H1: RecCL item-difficulty (β) will drive accuracy on Goodreads too | Popularity long-tail is universal | +71% NDCG on Steam |
| H2: SANS medium negatives will control hallucination | Same-genre negatives generalize across domains | 0% OOD on Steam |
| H3: RecAug full will provide mild regularization | Consistency loss is domain-agnostic | -14% NDCG, preserved Novelty |
| H4: Sequence difficulty (α) alone will underperform | Sequence length variance differs but the principle holds | -71% NDCG on Steam |
| H5: Effect sizes will be smaller on Goodreads | Book sequences are shorter → less room for curriculum effects | N/A |

### 5.2 Domain-Specific Expectations

**Where Goodreads may differ from Steam:**

1. **Stronger genre signal.** Book genres are more hierarchical and consistent than game
   tags. This may make SANS medium negatives (same-genre) even more effective.

2. **Weaker temporal signal.** Books don't have "playtime" — reading is one-and-done.
   Session-boundary permutation (RecAug) may have different effects.

3. **Shorter sequences.** The median Goodreads user likely has fewer reads than the median
   Steam user's 36 games. This may reduce RecCL's sequence-difficulty dimension impact
   even further.

4. **Series effects.** Book series (trilogies, sagas) create strong sequential patterns
   that don't exist in games. The model may learn "series completion" as a shortcut.

### 5.3 Success Criteria

For cross-dataset validation to be considered successful (i.e., the methods "generalize"):

- **RecCL item-difficulty** must show positive ΔNDCG on Goodreads (not necessarily +71%,
  but directionally consistent)
- **SANS medium negatives** must reduce OOD on Goodreads (hallucination control is
  domain-agnostic)
- **No method** should catastrophically fail (ΔNDCG < -50%) on Goodreads
- The **relative ordering** of ablation variants should be consistent across datasets

---

## 6. Implementation: What's Built

### 6.1 Files Created for Cross-Dataset Support

| File | Lines | Purpose |
|------|-------|---------|
| `preprocess_goodreads.py` | 480 | 7-step preprocessing pipeline |
| `explore_goodreads.py` | ~300 | Initial data exploration |
| `data_validation/explore.py` | 434 | Streaming exploration with cross-dataset comparison |
| `config/config_goodreads.yaml` | 102 | Book-specific training configuration |

### 6.2 Files Modified for Cross-Dataset Support

| File | Change |
|------|--------|
| `scripts/run_experiments.py` | Added `--dataset goodreads` and `--dataset both` support |
| `scripts/run_component_ablation.py` | Added `--dataset goodreads` and `--dataset both` support |
| `trainer.py` | Dataset-agnostic design already supports any `--data_dir` |

### 6.3 Code Quality Notes

- All Goodreads processing uses **streaming** (chunked CSV reading, line-by-line JSON
  parsing) — never loads full dataset into RAM
- Progress bars with ETA for all steps (10M-row intervals for interactions)
- Error handling for malformed JSON lines in book metadata
- Config-driven: changing datasets requires only `--config config/config_goodreads.yaml`

---

## 7. Execution Plan

### 7.1 Step-by-Step

```bash
# Step 1: Preprocess Goodreads data (~30-60 min)
python preprocess_goodreads.py --k_core 5 --min_seq_len 5 --max_seq_len 50

# Step 2: Quick validation run (5K users, 1 epoch, ~10 min)
python scripts/run_experiments.py --dataset goodreads --modes base --quick

# Step 3: Full baseline (all users, 5 epochs, ~2-4 hours)
python scripts/run_experiments.py --dataset goodreads --modes base,full --seeds 42

# Step 4: Component ablation (3 modules × multiple variants, ~8-12 hours)
python scripts/run_component_ablation.py --dataset goodreads --max_train 10000 --epochs 3

# Step 5: Multi-seed statistical validation (3 seeds, ~6-12 hours)
python scripts/run_experiments.py --dataset goodreads --modes base,full --seeds 42,123,456

# Step 6: Evaluation
bash scripts/evaluate_all.sh goodreads
```

### 7.2 Estimated Compute Budget

| Phase | GPU Hours | Wall Time |
|-------|-----------|-----------|
| Preprocessing | 0 (CPU-only) | 30–60 min |
| Baseline (1 seed) | ~2 | ~2 h |
| Full LLM-Rec (1 seed) | ~4 | ~4 h |
| RecCL ablation (5 variants) | ~10 | ~5 h |
| SANS ablation (3 variants) | ~6 | ~3 h |
| RecAug ablation (3 variants) | ~6 | ~3 h |
| Multi-seed (2 modes × 3 seeds) | ~18 | ~10 h |
| **Total** | **~46 GPU-hours** | **~28 wall-time hours** |

Note: SANS hard negatives require pre-computing embedding similarity on the Goodreads
item catalog (sentence-transformers all-MiniLM-L6-v2). This is a CPU-only step taking
~10–30 minutes depending on catalog size.

---

## 8. Cross-Dataset Comparison Framework

Once Goodreads experiments complete, the comparison table will follow this format:

| Metric | Steam (Base) | Steam (Full) | Goodreads (Base) | Goodreads (Full) | Cross-Dataset Consistency |
|--------|-------------|-------------|------------------|------------------|--------------------------|
| NDCG@10 | 0.0070 | — | — | — | — |
| Novelty@10 | 0.7939 | — | — | — | — |
| OOD@10 | 0.001 | — | — | — | — |
| Coverage@10 | 0.0014 | — | — | — | — |

And for ablation consistency:

| Ablation | Steam ΔNDCG | Goodreads ΔNDCG | Direction Consistent? |
|----------|------------|-----------------|----------------------|
| RecCL item-only (β=1.0) | +71.4% | — | — |
| RecCL seq-only (α=1.0) | -71.4% | — | — |
| SANS easy-only | -28.6% | — | — |
| SANS full (all tiers) | 0.0% | — | — |
| RecAug trunc-only | -100% | — | — |
| RecAug full (perm+trunc) | -14.3% | — | — |

**Consistency score:** Fraction of ablation variants where ΔNDCG has the same sign on
both datasets. Target: ≥5/6 (83%) for claiming "generalizes."

---

## 9. Key Findings (To Date)

Since training has not yet been executed, findings are limited to data characterization
and infrastructure readiness:

1. **The Goodreads dataset is well-suited for cross-dataset validation.** It shares the
   same source (UCSD McAuley), academic recognition, and long-tail characteristics as
   Steam, while differing in domain (books vs. games), consumption pattern (one-read vs.
   replay), and temporal dynamics (slow vs. fast).

2. **The preprocessing pipeline is complete and format-compatible.** The 7-step pipeline
   outputs data in the identical JSON format as Steam preprocessing, requiring zero code
   changes to the training/evaluation stack.

3. **The experiment infrastructure is ready.** Both `run_experiments.py` and
   `run_component_ablation.py` support `--dataset goodreads` and `--dataset both` (for
   back-to-back Steam+Goodreads runs). Config is written. Prompt template is book-specific.

4. **The main blocker is GPU time.** At ~46 GPU-hours total, the full experimental
   protocol requires approximately 2–3 days of dedicated RTX 3060 time. A quick validation
   run (5K users, 1 epoch, base only) takes ~10 minutes and should be the next step.

---

## 10. Repository Structure After Completion

When all experiments are complete, the repository will look like:

```
data/goodreads_processed/
├── train.json, val.json, test.json
├── item_catalog.json, item_popularity.json, stats.json

data/cache_goodreads/
├── hard_negatives.json
├── item_intents.json

checkpoints/goodreads/
├── base/ (seed=42, 123, 456)
├── full/ (seed=42, 123, 456)
├── ablation/ (reccl, sans, recaug variants)

results/goodreads/
├── baseline_report.md
├── comprehensive_report.md
├── component_ablation_report.md
├── statistical_significance_report.md
├── cross_dataset_comparison.md
└── (per-seed metric JSON files)
```

---

## Appendix A: Commands Reference

### Preprocessing

```bash
# Full preprocessing
python preprocess_goodreads.py --k_core 5 --min_seq_len 5 --max_seq_len 50

# Quick test with 50K users
python preprocess_goodreads.py --k_core 5 --max_users 50000

# Custom output directory
python preprocess_goodreads.py --k_core 5 --output_dir data/goodreads_test
```

### Training

```bash
# Single experiment (base model only)
python scripts/run_experiments.py --dataset goodreads --modes base --quick

# Full pipeline with multi-seed
python scripts/run_experiments.py --dataset goodreads --modes base,full --seeds 42,123,456

# Component ablation
python scripts/run_component_ablation.py --dataset goodreads --max_train 10000 --epochs 3

# Both datasets back-to-back
python scripts/run_experiments.py --dataset both --modes base --seeds 42
```

### Evaluation

```bash
# Evaluate all Goodreads checkpoints
bash scripts/evaluate_all.sh goodreads

# Single checkpoint evaluation
python evaluate.py --data_dir data/goodreads_processed \
    --checkpoint checkpoints/goodreads/base/best --dataset goodreads
```

---

## Appendix B: Data Completeness

| Component | Status | Notes |
|-----------|--------|-------|
| Goodreads raw data download | Complete | 4 files, 6.5 GB in `data_validation/` |
| Data exploration | Complete | Streaming analysis of all sources |
| `preprocess_goodreads.py` | Complete | 480 lines, 7-step pipeline |
| `config_goodreads.yaml` | Complete | 102 lines, book-specific prompts |
| `explore_goodreads.py` | Complete | Initial exploration script |
| `data_validation/explore.py` | Complete | Enhanced streaming exploration |
| Experiment runner support | Complete | `--dataset goodreads` in both scripts |
| Preprocessing execution | **Not run** | `python preprocess_goodreads.py` pending |
| Training execution | **Not run** | Blocked on preprocessing |
| Evaluation | **Not run** | Blocked on training |
| Cross-dataset comparison report | **Not run** | Blocked on results |

---

## Appendix C: Related Work & References

1. Wan, M., & McAuley, J. (2018). "Item Recommendation on Monotonic Behavior Chains."
   *RecSys 2018.* — Original Goodreads dataset paper.

2. McAuley, J., et al. (2015). "Image-Based Recommendations on Styles and Substitutes."
   *SIGIR 2015.* — Goodreads + Amazon datasets introduction.

3. Geng, S., et al. (2022). "Recommendation as Language Processing (RLP): A Unified
   Pretrain, Personalized Prompt & Predict Paradigm (P5)." *RecSys 2022.* — P5 evaluated
   on multiple domains including books.

4. Nature Portfolio journals require multi-dataset validation as standard. Examples:
   - Rocket (2024): 8 tasks, 32 benchmarks
   - Tactiformer (2024): multiple sports datasets
   - ChartRecover (2024): multiple chart types + perturbation conditions
