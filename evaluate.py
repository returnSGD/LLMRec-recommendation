"""
Evaluation script for LLM-Rec generative recommendation.

Computes full metric suite:
  - Accuracy: NDCG@K, Recall@K, HR@K
  - Diversity: ILS@K, Coverage@K
  - Cold-item: Tail Recall@K
  - Novelty: Novelty@K
  - Hallucination: OOD@K

Usage:
  python evaluate.py --checkpoint checkpoints/final_model.pt --data_dir data/processed
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Set, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import Dataset

from models.config import ModelConfig
from models.base_p5 import BaseP5Model
from utils.metrics import compute_all_metrics


class EvalDataset(Dataset):
    """Minimal dataset for evaluation (avoids trainer import)."""

    def __init__(self, samples, item_catalog, prompt_template, max_seq_len=50):
        self.samples = samples
        self.item_catalog = item_catalog
        self.prompt_template = prompt_template
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s['sequence'][-self.max_seq_len:]
        target = s['target_item']
        item_texts = [self.item_catalog.get(iid, {}).get('title', iid) for iid in seq]
        prompt = self.prompt_template.format(item_sequence=" → ".join(item_texts))
        target_text = self.item_catalog.get(target, {}).get('title', target)
        return {
            'user_id': s['user_id'], 'sequence': seq, 'target_item': target,
            'prompt': prompt, 'target_text': target_text,
        }


def eval_collate_batch(batch):
    """Custom collate: leaves prompt/target as lists of strings."""
    result = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if key in ('prompt', 'target_text', 'user_id', 'target_item'):
            result[key] = values
        else:
            result[key] = values  # keep as list
    return result


def load_model(checkpoint_path: str, base_model: str, device: torch.device) -> BaseP5Model:
    """Load trained model from checkpoint."""
    cfg = ModelConfig(base_model=base_model)
    model = BaseP5Model(cfg)
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def build_title_index(item_catalog: dict) -> Dict[str, str]:
    """Build normalized title -> item ID lookup."""
    index = {}
    for iid, info in item_catalog.items():
        title = info.get('title', iid)
        # Normalize: lowercase, strip whitespace/punctuation
        norm = title.strip().lower().rstrip('.')
        index[norm] = iid
        # Also store the original title in the catalog's title field
        # for reverse lookup
    return index


def match_to_catalog(text: str, title_index: Dict[str, str],
                     item_catalog: dict, default_id: str = '__UNK__') -> str:
    """Match generated text to closest catalog item ID."""
    norm = text.strip().lower().rstrip('.')
    # Exact match after normalization
    if norm in title_index:
        return title_index[norm]
    # Substring fallback
    for norm_title, iid in title_index.items():
        if norm_title in norm or norm in norm_title:
            return iid
    return default_id


def generate_recommendations(model: BaseP5Model, dataloader: DataLoader,
                              device: torch.device, top_k: int = 20) -> Tuple[
                                  List[List[str]], List[List[str]]]:
    """Generate recommendations and return item IDs (matched to catalog).

    Returns: (predictions, ground_truths) — each is List[List[str]] of item IDs
    """
    all_predictions = []
    all_ground_truths = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            prompts = batch['prompt']
            target_ids = batch['target_item']
            target_texts = batch['target_text']

            tokenized = model.tokenize(prompts)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            generated_ids = model.generate(
                input_ids, attention_mask,
                num_beams=5,
                num_return_sequences=1,
                max_length=64,
                early_stopping=True,
            )
            generated_texts = model.decode(generated_ids)

            for gen_text, gt_id, gt_text in zip(generated_texts, target_ids, target_texts):
                all_predictions.append([gen_text])
                all_ground_truths.append([gt_id])

    # Trim to top_k predictions per sample
    predictions = [preds[:top_k] for preds in all_predictions]
    ground_truths = [[gts[0]] for gts in all_ground_truths]

    return predictions, ground_truths


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM-Rec model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--base_model', type=str, default='google/flan-t5-base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON for metrics')
    parser.add_argument('--max_eval', type=int, default=0,
                        help='Max test samples to evaluate (0 = all)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    print("Loading data...")
    with open(os.path.join(args.data_dir, 'test.json'), 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    with open(os.path.join(args.data_dir, 'item_catalog.json'), 'r', encoding='utf-8') as f:
        item_catalog = json.load(f)
    with open(os.path.join(args.data_dir, 'item_popularity.json'), 'r', encoding='utf-8') as f:
        item_popularity = json.load(f)

    print(f"Test samples: {len(test_samples)}, Items: {len(item_catalog)}")

    if args.max_eval > 0:
        test_samples = test_samples[:args.max_eval]
        print(f"  Subsampled to {len(test_samples)} for quick eval")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.base_model, device)
    model.item_catalog = item_catalog

    # Build test dataset
    prompt_template = (
        "The user has played the following games in order:\n"
        "{item_sequence}\n\n"
        "what game should be recommended next? Answer with the game title only."
    )

    test_dataset = EvalDataset(test_samples, item_catalog, prompt_template)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=eval_collate_batch)

    # Generate predictions (raw text)
    print("Generating recommendations...")
    raw_predictions, ground_truths = generate_recommendations(
        model, test_loader, device, top_k=max(args.top_k)
    )

    # Build title -> ID index and match generated text to catalog
    title_index = build_title_index(item_catalog)
    predictions = []
    ood_count = 0
    for pred_texts in raw_predictions:
        matched_ids = []
        for text in pred_texts:
            iid = match_to_catalog(text, title_index, item_catalog)
            if iid == '__UNK__':
                ood_count += 1
            matched_ids.append(iid)
        predictions.append(matched_ids)
    total_preds = sum(len(p) for p in predictions)
    print(f"  Matched: {total_preds - ood_count}/{total_preds} ({100*(1-ood_count/max(total_preds,1)):.1f}%)")

    # Prepare metric inputs
    catalog_ids = set(item_catalog.keys())

    # Build item_genre map
    item_genres = {}
    for iid, info in item_catalog.items():
        item_genres[iid] = info.get('genres', info.get('tags', []))

    # Identify tail items (interactions < 50)
    tail_threshold = 50
    item_counts = Counter(item_popularity)
    tail_items = {iid for iid, cnt in item_counts.items() if cnt < tail_threshold}

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        item_genres=item_genres,
        item_popularity=item_popularity,
        catalog_ids=catalog_ids,
        tail_items=tail_items,
        k_values=args.top_k,
    )

    # Display
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k in args.top_k:
        print(f"\n--- Top-{k} ---")
        print(f"  NDCG@{k}:      {metrics.get(f'NDCG@{k}', 0):.4f}")
        print(f"  Recall@{k}:    {metrics.get(f'Recall@{k}', 0):.4f}")
        print(f"  HR@{k}:        {metrics.get(f'HR@{k}', 0):.4f}")
        print(f"  ILS@{k}:       {metrics.get(f'ILS@{k}', 0):.4f}")
        print(f"  Tail_Recall@{k}: {metrics.get(f'Tail_Recall@{k}', 0):.4f}")
        print(f"  Novelty@{k}:   {metrics.get(f'Novelty@{k}', 0):.4f}")
    print(f"\n  Coverage@10: {metrics.get('Coverage@10', 0):.4f}")
    print(f"  OOD@10:       {metrics.get('OOD@10', 0):.4f}")

    # Save
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == '__main__':
    main()
