"""
Base P5 model wrapper for T5 encoder-decoder on generative recommendation.

Provides:
  - Model loading (T5-small/base/large with optional LoRA)
  - Text-to-text formatting (prompt → generation)
  - Embedding extraction (for InfoNCE loss)
  - Generation (for inference/evaluation)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, List, Tuple

from models.config import ModelConfig


class BaseP5Model(nn.Module):
    """T5-based generative recommendation model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

        # Add special tokens for recommendation
        special_tokens = {
            'additional_special_tokens': [
                config.user_token, config.item_token, config.sep_token
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=list(config.lora_target_modules),
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.hidden_size = self.model.config.d_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through T5.

        Returns dict with keys: loss, logits, encoder_last_hidden_state
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )
        result = {'loss': outputs.loss, 'logits': outputs.logits}
        if output_hidden_states and outputs.encoder_last_hidden_state is not None:
            # Mean pool encoder output as sequence representation
            result['seq_embedding'] = outputs.encoder_last_hidden_state.mean(dim=1)
        return result

    def get_sequence_embedding(self, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor) -> torch.Tensor:
        """Get mean-pooled encoder representation for a sequence (for InfoNCE)."""
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Mean pool over sequence length (accounting for padding)
        hidden = outputs.last_hidden_state  # [B, L, D]
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Generate text (item titles/IDs)."""
        gen_kwargs = {
            'max_length': max_length or self.config.max_target_length,
            'num_beams': self.config.num_beams,
            'do_sample': self.config.do_sample,
        }
        gen_kwargs.update(kwargs)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    def decode(self, token_ids: torch.Tensor,
               skip_special_tokens: bool = True) -> List[str]:
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def format_prompt(self, item_sequence: List[str], template: str) -> str:
        """Format a prompt using item sequence and template."""
        seq_str = " → ".join(item_sequence)
        return template.format(item_sequence=seq_str)

    def tokenize(self, texts: List[str], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            max_length=max_length or self.config.max_source_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

    def get_item_embedding_for_id(self, item_text: str) -> torch.Tensor:
        """Get embedding for an item from its text representation."""
        tokens = self.tokenizer(
            item_text,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            emb = self.get_sequence_embedding(
                tokens['input_ids'], tokens['attention_mask']
            )
        return emb.squeeze(0)

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
