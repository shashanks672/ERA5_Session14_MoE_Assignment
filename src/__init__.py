# Mixture-of-Experts (MoE) Assignment Package
"""
Core implementations of Dense Causal Language Model, Mixture-of-Experts (MoE) Layer,
Top-k Routing with Load-Balancing Auxiliary Loss, Dense-to-MoE Conversion,
and Two-Phase Training Pipeline.
"""

from .model import DenseLanguageModel, TransformerConfig
from .moe import MoELanguageModel, MoEConfig, MoEFFN, TopKRouter
from .convert import convert_dense_to_moe
from .dataset import TextDataset, CharTokenizer, get_dataloader
from .trainer import Trainer

__all__ = [
    "DenseLanguageModel",
    "MoELanguageModel",
    "TransformerConfig",
    "MoEConfig",
    "MoEFFN",
    "TopKRouter",
    "convert_dense_to_moe",
    "TextDataset",
    "CharTokenizer",
    "get_dataloader",
    "Trainer",
]
