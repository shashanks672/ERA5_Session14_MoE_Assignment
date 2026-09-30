"""
Dense Causal Language Model (Decoder-Only Transformer).

Implements a clean, modular Transformer Decoder with:
- Token and learnable positional embeddings
- Multi-Head Causal Self-Attention (using PyTorch F.scaled_dot_product_attention)
- Standard Feed-Forward Network (Dense FFN)
- Pre-LayerNorm architecture
- Parameter accounting (total vs active parameters)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Hyperparameters for the Dense Transformer Model."""
    vocab_size: int = 256
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 1024            # 4 * d_model
    max_seq_len: int = 128
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True     # Tie embedding and LM head weights


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention mechanism.
    
    Uses PyTorch's native scaled dot-product attention (FlashAttention when supported).
    Includes a causal lower-triangular mask ensuring each token only attends to preceding tokens.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        
        # Combined Q, K, V projection for speed and memory efficiency
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.size()
        
        # Project Q, K, V in a single matrix multiply: (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product causal attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        # Reassemble head outputs: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and residual dropout
        out = self.resid_dropout(self.c_proj(y))
        return out


class DenseFFN(nn.Module):
    """
    Standard Feed-Forward Network (MLP) for Dense Transformer.
    
    Architecture:
        x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> Dropout -> output
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        h = self.act(self.w1(x))
        out = self.dropout(self.w2(h))
        return out


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Decoder Block.
    
    Structure:
        x = x + Attention(LN_1(x))
        x = x + DenseFFN(LN_2(x))
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = DenseFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DenseLanguageModel(nn.Module):
    """
    Dense Decoder-Only Language Model.
    
    Contains:
    - Token Embedding table
    - Learnable Positional Embedding table
    - N Transformer Decoder Blocks (with standard Dense FFN)
    - Final LayerNorm and Linear LM Head
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final norm and language model head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (optional)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
            
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass for causal language modeling.
        
        Args:
            idx: Tensor of token indices with shape (batch_size, seq_len)
            targets: Optional ground truth token indices for loss computation
            
        Returns:
            logits: Next-token logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy causal language modeling loss (or None)
            metrics: Dictionary of auxiliary metrics (empty for dense model)
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        
        # Create position indices [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Embed tokens and positions
        tok_vec = self.tok_emb(idx)      # (B, T, d_model)
        pos_vec = self.pos_emb(pos)      # (T, d_model)
        x = self.drop(tok_vec + pos_vec)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, {}

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculates parameter counts:
        - Total parameters
        - Active parameters per token (for Dense model, all params are active)
        """
        total_params = sum(p.numel() for p in self.parameters())
        # In dense model, 100% of parameters are active for every token
        return {
            "total_params": total_params,
            "active_params_per_token": total_params,
            "inactive_params_per_token": 0,
            "active_ratio": 1.0
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens given a conditioning prefix.
        """
        for _ in range(max_new_tokens):
            # Crop context if sequence exceeds maximum sequence length
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            # Focus only on the last step
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Optional Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
