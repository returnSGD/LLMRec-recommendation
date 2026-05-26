"""
Main training script for LLM-Rec with sample engineering.

Integrates RecCL, SANS, and RecAug into the OpenP5-style training loop.
Three hooks in the training loop:
  1. RecCL: curriculum sampling (hardness-weighted batch construction)
  2. RecAug: data augmentation (variants generation)
  3. SANS: layered negative sampling + weighted InfoNCE loss

Usage:
  python trainer.py --config config/config.yaml --mode base        # baseline only
  python trainer.py --config config/config.yaml --mode full        # all three methods
  python trainer.py --config config/config.yaml --mode ablation    # single method
"""

import os
import sys
import json
import argparse
import yaml
import time
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.config import ModelConfig
from models.base_p5 import BaseP5Model
from sample_engineering.rec_cl import DifficultyScorer, CurriculumSampler
from sample_engineering.sans import HardNegativeGenerator, LayeredNegativeSampler, weighted_infonce_loss
from sample_engineering.rec_aug import (
    SessionBoundaryDetector, IntentPreservingTruncation,
    SessionPermutation, LLMGuidedSubstitution, RecAugPipeline,
)
from utils.caching import LLMClient
from utils.steam_utils import build_item_text


# ============================================================
# Dataset
# ============================================================

class SteamSequenceDataset(Dataset):
    """Steam user-item sequences for text-to-text recommendation."""

    def __init__(self, samples: List[Dict], item_catalog: Dict[str, Dict],
                 prompt_template: str, max_seq_len: int = 50):
        self.samples = samples
        self.item_catalog = item_catalog
        self.prompt_template = prompt_template
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample['sequence'][-self.max_seq_len:]
        target = sample['target_item']

        # Build item text sequence
        item_texts = []
        for item_id in seq:
            if item_id in self.item_catalog:
                item_texts.append(self.item_catalog[item_id].get('title', item_id))
            else:
                item_texts.append(item_id)

        item_sequence_str = " → ".join(item_texts)
        prompt = self.prompt_template.format(item_sequence=item_sequence_str)

        target_text = self.item_catalog.get(target, {}).get('title', target) if target in self.item_catalog else target

        return {
            'user_id': sample['user_id'],
            'sequence': seq,
            'target_item': target,
            'prompt': prompt,
            'target_text': target_text,
            'playtimes': sample.get('playtimes', []),
        }


def collate_batch(batch):
    """Custom collate: handle variable-length lists (sequence, playtimes)."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in batch]
        # Only stack as tensor if all values are the same type and length
        if all(isinstance(v, torch.Tensor) for v in values):
            result[key] = torch.stack(values)
        else:
            # Keep lists and scalars as-is
            result[key] = values
    return result


# ============================================================
# Sample Engineering Setup
# ============================================================

def build_difficulty_scorer(item_catalog: Dict[str, Dict],
                            item_popularity: Dict[str, int],
                            train_samples: List[Dict]) -> DifficultyScorer:
    """Build the RecCL difficulty scorer from data statistics."""
    # Build item genre mapping for entropy computation
    item_genres = {}
    for iid, info in item_catalog.items():
        item_genres[iid] = info.get('genres', info.get('tags', []))

    return DifficultyScorer(
        item_popularity=item_popularity,
        item_genres=item_genres,
        cf_scores=None,  # No CF model yet; starts with uniform pred difficulty
    )


def build_sans_components(item_catalog: Dict[str, Dict],
                          item_popularity: Dict[str, int],
                          llm_client: LLMClient,
                          config: Dict) -> Tuple[HardNegativeGenerator, LayeredNegativeSampler]:
    """Build SANS components: hard negative generator + layered sampler."""
    item_texts = {}
    item_genres = {}
    for iid, info in item_catalog.items():
        item_texts[iid] = info.get('text', info.get('title', iid))
        item_genres[iid] = info.get('genres', info.get('tags', []))

    hard_gen = HardNegativeGenerator(
        llm_client=llm_client,
        item_texts=item_texts,
        item_genres=item_genres,
        cache_path=config.get('cache_dir', 'data/cache/hard_negatives.json'),
    )

    # Build medium pools (by genre)
    medium_pools = defaultdict(list)
    for iid, genres in item_genres.items():
        for g in genres:
            medium_pools[g].append(iid)

    all_items = list(item_catalog.keys())
    sampler = LayeredNegativeSampler(
        easy_pool=all_items,
        medium_pools=dict(medium_pools),
        hard_generator=hard_gen,
        easy_count=config.get('sans', {}).get('easy_neg_count', 8),
        medium_count=config.get('sans', {}).get('medium_neg_count', 4),
        hard_count=config.get('sans', {}).get('hard_neg_count', 4),
        easy_weight=config.get('sans', {}).get('easy_weight', 0.1),
        medium_weight=config.get('sans', {}).get('medium_weight', 0.3),
        hard_weight=config.get('sans', {}).get('hard_weight', 0.6),
        temperature=config.get('sans', {}).get('temperature', 0.07),
    )

    return hard_gen, sampler


def build_recaug_pipeline(item_catalog: Dict[str, Dict],
                          llm_client: LLMClient,
                          config: Dict) -> RecAugPipeline:
    """Build RecAug pipeline with all three augmentors."""
    item_texts = {}
    for iid, info in item_catalog.items():
        item_texts[iid] = info.get('text', info.get('title', iid))

    recaug_cfg = config.get('rec_aug', {})

    # Intent-preserving truncation
    truncation = IntentPreservingTruncation(
        llm_client=llm_client,
        item_texts=item_texts,
        cache_path='data/cache/item_intents.json',
    )

    # Session permutation
    boundary_detector = SessionBoundaryDetector(
        time_gap_hours=recaug_cfg.get('session_gap_hours', 72),
        llm_client=llm_client,
    )
    permutation = SessionPermutation(boundary_detector)

    # LLM-guided substitution (placeholder — needs embeddings)
    substitution = LLMGuidedSubstitution(
        llm_client=llm_client,
        item_texts=item_texts,
        item_embeddings={},  # populated later or lazily
        embedding_model=None,
        intent_cache=truncation.intent_cache,
        cache_path='data/cache/item_substitutions.json',
    )

    return RecAugPipeline(
        truncation=truncation,
        permutation=permutation,
        substitution=substitution,
        substitution_prob=recaug_cfg.get('substitution_prob', 0.2),
    )


# ============================================================
# Training Loop
# ============================================================

def train_epoch(model: BaseP5Model, dataloader: DataLoader,
                optimizer: AdamW, scheduler, device: torch.device,
                scaler=None,
                curriculum_sampler: Optional[CurriculumSampler] = None,
                sans_sampler: Optional[LayeredNegativeSampler] = None,
                recaug_pipeline: Optional[RecAugPipeline] = None,
                consistency_lambda: float = 0.1,
                use_sample_engineering: bool = True) -> Dict[str, float]:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_infonce_loss = 0.0
    total_consistency_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Tokenize inputs
        prompts = batch['prompt']
        targets = batch['target_text']

        tokenized_inputs = model.tokenize(prompts)
        tokenized_labels = model.tokenize(targets)

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)
        labels = tokenized_labels['input_ids'].to(device)

        # Forward pass with autocast for fp16 stability
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=scaler is not None):
            outputs = model(input_ids, attention_mask, labels=labels,
                            output_hidden_states=use_sample_engineering)
        ce_loss = outputs['loss']

        total_loss_batch = ce_loss
        infonce_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        if use_sample_engineering:
            # SANS: weighted InfoNCE loss
            if sans_sampler and 'seq_embedding' in outputs:
                positive_ids = batch['target_item']
                neg_ids_list, neg_weights_list = sans_sampler.get_negatives_batch(
                    list(positive_ids)
                )

                # Fast lookup from pre-computed embedding cache
                emb_cache = getattr(model, '_item_emb_cache', {})
                pos_embs_list = []
                for pid in positive_ids:
                    if pid in emb_cache:
                        pos_embs_list.append(emb_cache[pid])
                    else:
                        t = model.item_catalog.get(pid, {}).get('title', pid)
                        pos_embs_list.append(model.get_item_embedding_for_id(t).cpu())

                if pos_embs_list:
                    pos_embs = torch.stack(pos_embs_list).to(device)
                    all_neg_embs = []
                    all_neg_wts = []
                    for nids, wts in zip(neg_ids_list, neg_weights_list):
                        neg_embs_list = []
                        for nid in nids:
                            if nid in emb_cache:
                                neg_embs_list.append(emb_cache[nid])
                            else:
                                t = model.item_catalog.get(nid, {}).get('title', nid)
                                neg_embs_list.append(
                                    model.get_item_embedding_for_id(t).cpu()
                                )
                        if neg_embs_list:
                            all_neg_embs.append(torch.stack(neg_embs_list))
                            all_neg_wts.append(torch.tensor(wts))

                    if all_neg_embs:
                        neg_embs_tensor = torch.stack(all_neg_embs).to(device)
                        neg_wts_tensor = torch.stack(all_neg_wts).to(device)
                        infonce_loss = weighted_infonce_loss(
                            outputs['seq_embedding'], pos_embs,
                            neg_embs_tensor, neg_wts_tensor,
                            temperature=sans_sampler.tau,
                        )
                        total_loss_batch = total_loss_batch + infonce_loss

            # RecAug: consistency regularization
            if recaug_pipeline and 'seq_embedding' in outputs:
                # Generate augmented variants
                aug_variants = []
                for i, seq in enumerate(batch['sequence']):
                    variants = recaug_pipeline.augment(
                        seq,
                        timestamps=None,  # no timestamps; use genre-shift detection
                    )
                    if variants:
                        aug_variants.append(variants[0]['sequence'])

                if aug_variants:
                    aug_prompts = [
                        model.format_prompt(v, dataloader.dataset.prompt_template)
                        for v in aug_variants
                    ]
                    aug_tokens = model.tokenize(aug_prompts)
                    aug_ids = aug_tokens['input_ids'].to(device)
                    aug_mask = aug_tokens['attention_mask'].to(device)

                    aug_outputs = model(aug_ids, aug_mask, output_hidden_states=True)
                    if 'seq_embedding' in aug_outputs:
                        # KL divergence between original and augmented representations
                        orig_logits = F.log_softmax(outputs['seq_embedding'], dim=-1)
                        aug_probs = F.softmax(aug_outputs['seq_embedding'], dim=-1)
                        consistency_loss = F.kl_div(
                            orig_logits, aug_probs,
                            reduction='batchmean',
                        )
                        total_loss_batch = total_loss_batch + consistency_lambda * consistency_loss

        # Backward with fp16 scaling
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if curriculum_sampler:
            curriculum_sampler.step()

        total_loss += total_loss_batch.item()
        total_ce_loss += ce_loss.item()
        total_infonce_loss += infonce_loss.item()
        total_consistency_loss += consistency_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'ce': f'{total_ce_loss/num_batches:.4f}',
        })

    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'infonce_loss': total_infonce_loss / num_batches,
        'consistency_loss': total_consistency_loss / num_batches,
    }


# ============================================================
# Main
# ============================================================

def _maybe_load_dotenv(dotenv_path: str = ".env"):
    """Load .env file into os.environ if it exists (no external dependency)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dotenv_path)
    if not os.path.isfile(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if value and key not in os.environ:
                os.environ[key] = value


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train LLM-Rec model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='YAML config file')
    parser.add_argument('--mode', type=str, default='base',
                        choices=['base', 'full', 'ablation_reccl', 'ablation_sans', 'ablation_recaug'],
                        help='Training mode')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--max_train', type=int, default=None,
                        help='Max training samples (for quick runs)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config seed)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs (for quick runs)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override seed from CLI (for multi-seed experiments)
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    # Set all random seeds
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load processed data
    print("Loading processed data...")
    with open(os.path.join(args.data_dir, 'train.json'), 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    with open(os.path.join(args.data_dir, 'item_catalog.json'), 'r', encoding='utf-8') as f:
        item_catalog = json.load(f)
    with open(os.path.join(args.data_dir, 'item_popularity.json'), 'r', encoding='utf-8') as f:
        item_popularity = json.load(f)

    print(f"Train samples: {len(train_samples)}, Items: {len(item_catalog)}")
    if args.max_train:
        train_samples = train_samples[:args.max_train]
        print(f"  (limited to {args.max_train} samples for quick mode)")

    # Determine which methods to enable
    use_reccl = args.mode in ('full', 'ablation_reccl')
    use_sans = args.mode in ('full', 'ablation_sans')
    use_recaug = args.mode in ('full', 'ablation_recaug')
    use_sample_engineering = args.mode != 'base'

    # Model config
    model_cfg = ModelConfig(
        base_model=config['model']['base_model'],
        max_source_length=config['model']['max_seq_length'],
        max_target_length=config['model']['max_output_length'],
        use_lora=config['model'].get('use_lora', False),
        lora_r=config['model'].get('lora_r', 16),
        lora_alpha=config['model'].get('lora_alpha', 32),
        lora_dropout=config['model'].get('lora_dropout', 0.1),
    )

    # Prompt template (configurable, defaults to games)
    prompt_cfg = config.get("prompt", {})
    prompt_template = prompt_cfg.get("template",
        "The user has played the following games in order:\n"
        "{item_sequence}\n\n"
        "Considering the user's gaming preferences and play patterns, "
        "what game should be recommended next? Answer with the game title only."
    ).strip()

    # Dataset & Dataloader
    dataset = SteamSequenceDataset(
        train_samples, item_catalog, prompt_template,
        max_seq_len=config['preprocess']['max_seq_len'],
    )

    # Initialize sample engineering components
    curriculum_sampler = None
    sans_sampler = None
    recaug_pipeline = None
    llm_client = None

    if use_sample_engineering:
        # Load .env file if present (local development convenience)
        _maybe_load_dotenv()

        llm_cfg = config.get('llm_api', {})
        api_key = (
            os.environ.get('DEEPSEEK_API_KEY')
            or llm_cfg.get('api_key', '')
        )
        if not api_key:
            print("WARNING: DEEPSEEK_API_KEY not set. Set env var or configure in YAML. "
                  "LLM-dependent features (SANS hard negatives, RecAug intent analysis) "
                  "will fall back to basic behavior.")
        llm_client = LLMClient(
            base_url=os.environ.get('DEEPSEEK_BASE_URL',
                                    llm_cfg.get('base_url', 'https://api.deepseek.com/anthropic')),
            api_key=api_key,
            model=os.environ.get('DEEPSEEK_MODEL',
                                 llm_cfg.get('model', 'deepseek-chat')),
            max_tokens=llm_cfg.get('max_tokens', 512),
            temperature=llm_cfg.get('temperature', 0.7),
            request_interval=llm_cfg.get('request_interval', 1.0),
        )
        print("LLM client initialized for sample engineering")

        if use_reccl:
            scorer = build_difficulty_scorer(item_catalog, item_popularity, train_samples)
            curriculum_sampler = CurriculumSampler(
                scorer=scorer,
                seq_weight=config['rec_cl']['seq_difficulty_weight'],
                item_weight=config['rec_cl']['item_difficulty_weight'],
                pred_weight=config['rec_cl']['pred_difficulty_weight'],
                warmup_ratio=config['rec_cl']['warmup_ratio'],
                transition_type=config['rec_cl']['transition_type'],
            )
            # Precompute sample difficulties
            precomputed_scores = []
            for s in tqdm(train_samples, desc='Computing difficulties'):
                diffs = scorer.compute_all(s['user_id'], s['sequence'], s['target_item'])
                combined = (config['rec_cl']['seq_difficulty_weight'] * diffs['seq'] +
                            config['rec_cl']['item_difficulty_weight'] * diffs['item'] +
                            config['rec_cl']['pred_difficulty_weight'] * diffs['pred'])
                precomputed_scores.append(combined)
            curriculum_sampler.sample_scores = precomputed_scores
            print("RecCL curriculum sampler initialized")

        if use_sans:
            _, sans_sampler = build_sans_components(
                item_catalog, item_popularity, llm_client, config
            )
            print("SANS layered negative sampler initialized")

        if use_recaug:
            recaug_pipeline = build_recaug_pipeline(item_catalog, llm_client, config)
            print("RecAug pipeline initialized")

    # Build model
    model = BaseP5Model(model_cfg)
    model.item_catalog = item_catalog  # attach for item embedding lookups
    model.to(device)
    print(f"Model: {config['model']['base_model']} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Pre-compute item embeddings for fast InfoNCE (SANS)
    if use_sample_engineering:
        print("Pre-computing item embeddings...")
        item_embeddings = {}
        model.eval()
        with torch.no_grad():
            for iid, info in tqdm(item_catalog.items(), desc='Item embeddings'):
                text = info.get('title', iid)
                item_embeddings[iid] = model.get_item_embedding_for_id(text).cpu()
        model.train()
        model._item_emb_cache = item_embeddings
        print(f"  Cached {len(item_embeddings)} item embeddings")

    # Make sampler for curriculum learning
    if curriculum_sampler:
        sampler = curriculum_sampler.get_sampler(len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_batch,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_batch,
        )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    total_steps = len(dataloader) * config['training']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    if curriculum_sampler:
        curriculum_sampler.set_total_steps(total_steps)

    # FP16 support
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('fp16', False) else None

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nStarting training ({args.mode} mode) for {config['training']['epochs']} epochs")
    print(f"  RecCL: {use_reccl}, SANS: {use_sans}, RecAug: {use_recaug}")

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        metrics = train_epoch(
            model, dataloader, optimizer, scheduler, device,
            scaler=scaler,
            curriculum_sampler=curriculum_sampler,
            sans_sampler=sans_sampler,
            recaug_pipeline=recaug_pipeline,
            consistency_lambda=config['rec_aug'].get('consistency_lambda', 0.1),
            use_sample_engineering=use_sample_engineering,
        )

        print(f"  Loss: {metrics['loss']:.4f} | CE: {metrics['ce_loss']:.4f} | "
              f"InfoNCE: {metrics['infonce_loss']:.4f} | Consist: {metrics['consistency_loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            model.save(ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "final_model.pt")
    model.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == '__main__':
    main()
