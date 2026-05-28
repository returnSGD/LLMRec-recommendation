"""
Model configuration dataclass for T5-based generative recommendation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the T5 encoder-decoder model."""

    # Base model
    base_model: str = "google/flan-t5-base"
    pretrained: bool = True

    # Architecture (overrides base if set)
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None

    # Sequence
    max_source_length: int = 512
    max_target_length: int = 128

    # LoRA (for large models on limited GPU)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("q", "v")

    # Generation
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # Vocabulary
    vocab_size: Optional[int] = None
    pad_token_id: int = 0
    decoder_start_token_id: int = 0

    # Special tokens for recommendation
    user_token: str = "<user>"
    item_token: str = "<item>"
    sep_token: str = "<sep>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
