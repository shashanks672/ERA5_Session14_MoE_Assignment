#!/usr/bin/env python3
"""
Self-Contained All-in-One Mixture-of-Experts (MoE) Assignment Script.

Includes everything in a single, clean, runnable file:
- Character-level Tokenizer & Shakespeare Dataset Loader
- Dense Decoder-Only Transformer Language Model
- Top-k Softmax Router with Switch-style Load Balancing Loss
- Mixture-of-Experts (MoE) Layer with Routed Experts + Shared Expert
- Dense-to-MoE Surgical Weight Conversion
- Two-Phase Training Loop (Dense Pretraining -> MoE Continued Training)
- Parameter accounting (Total vs Active)
- Loss curve visualization saved to 'loss_curve.png'
"""

import os
import time
import math
import random
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =============================================================================
# 1. TOKENIZER & DATASET
# =============================================================================

SAMPLE_FALLBACK_TEXT = """First Citizen:
Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved.
First Citizen: First, you know Caius Marcius is chief enemy to the people.
All: We know't, we know't.
First Citizen: Let us kill him, and we'll have corn at our own price. Is't a verdict?
All: No more talking on't; let it be done: away, away!
Second Citizen: One word, good citizens.
First Citizen: We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they would yield us but the superfluity,
while it were wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that afflicts us, the object of our misery,
is as an inventory to particularise their abundance; our sufferance is a gain to them.
Let us revenge this with our pikes, ere we become rakes: for the gods know I speak
this in hunger for bread, not in thirst for revenge.
""" * 50


class CharTokenizer:
    """Lightweight character-level tokenizer."""
    def __init__(self, text: Optional[str] = None):
        if text is not None:
            self.chars = sorted(list(set(text)))
        else:
            self.chars = [chr(i) for i in range(128)]
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        return [self.char_to_idx.get(c, 0) for c in s]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.idx_to_char.get(i, "") for i in indices])


class TextDataset(Dataset):
    """Causal LM dataset slicing contiguous sequence chunks."""
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        self.num_samples = len(data) - seq_len

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_dataset(
    data_dir: str = "./data",
    seq_len: int = 128,
    train_split: float = 0.9,
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
) -> Tuple[TextDataset, TextDataset, CharTokenizer]:
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "input.txt")
    text = None
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        try:
            print(f"Downloading Tiny Shakespeare dataset from {url} ...")
            urllib.request.urlretrieve(url, file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Dataset downloaded successfully ({len(text)} characters).")
        except Exception as e:
            print(f"Notice: Using embedded dataset fallback ({e}).")
            text = SAMPLE_FALLBACK_TEXT
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    tokenizer = CharTokenizer(text=text)
    encoded_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(len(encoded_data) * train_split)
    train_ds = TextDataset(encoded_data[:n_train], seq_len=seq_len)
    val_ds = TextDataset(encoded_data[n_train:], seq_len=seq_len)
    return train_ds, val_ds, tokenizer


# =============================================================================
# 2. DENSE MODEL ARCHITECTURE
# =============================================================================

@dataclass
class TransformerConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with PyTorch scaled_dot_product_attention."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class DenseFFN(nn.Module):
    """Standard 2-layer MLP Feed-Forward Network."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""
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
    """Dense Transformer Language Model."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, {}

    def count_parameters(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "total_params": total_params,
            "active_params_per_token": total_params,
            "inactive_params_per_token": 0,
            "active_ratio": 1.0
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =============================================================================
# 3. MIXTURE-OF-EXPERTS (MoE) ARCHITECTURE
# =============================================================================

@dataclass
class MoEConfig(TransformerConfig):
    num_experts: int = 8
    top_k: int = 2
    shared_expert: bool = True
    aux_loss_coef: float = 0.01
    expert_d_ff: Optional[int] = None

    def __post_init__(self):
        if self.expert_d_ff is None:
            self.expert_d_ff = self.d_ff


class TopKRouter(nn.Module):
    """
    Top-k Gating Router with Differentiable Switch Load Balancing Auxiliary Loss.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, aux_loss_coef: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        N_tokens, _ = x_flat.size()
        
        # 1. Routing logits and Softmax probabilities
        logits = self.gate(x_flat)                       # (N_tokens, num_experts)
        probs = F.softmax(logits, dim=-1)                # (N_tokens, num_experts)
        
        # 2. Top-k Selection
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1) # (N_tokens, top_k)
        
        # 3. Normalize top-k weights
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 4. Load Balancing Auxiliary Loss (Switch Transformer / ST-MoE formula)
        # f_i = fraction of tokens routed to expert i
        # P_i = average probability router assigned to expert i
        # Aux Loss = num_experts * sum(f_i * P_i)
        expert_mask = torch.zeros(N_tokens, self.num_experts, device=x_flat.device)
        expert_mask.scatter_(1, topk_indices, 1.0)
        
        f_i = expert_mask.mean(dim=0)
        P_i = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        stats = {
            "expert_counts": expert_mask.sum(dim=0).detach().cpu().tolist(),
            "expert_fractions": f_i.detach().cpu().tolist(),
        }
        return topk_indices, topk_weights, aux_loss, stats


class Expert(nn.Module):
    """Individual Expert MLP."""
    def __init__(self, d_model: int, expert_d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, expert_d_ff, bias=bias)
        self.w2 = nn.Linear(expert_d_ff, d_model, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts Feed-Forward Network.
    Combines Top-k Routed Experts + Optional Shared Expert.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = TopKRouter(config.d_model, config.num_experts, config.top_k, config.aux_loss_coef)
        expert_d_ff = config.expert_d_ff or config.d_ff
        self.experts = nn.ModuleList([Expert(config.d_model, expert_d_ff, config.dropout, config.bias) for _ in range(config.num_experts)])
        self.shared_expert = Expert(config.d_model, config.d_ff, config.dropout, config.bias) if config.shared_expert else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        
        # Route tokens
        topk_indices, topk_weights, aux_loss, stats = self.router(x_flat)
        
        # Dispatch to routed experts
        routed_out_flat = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (topk_indices == expert_idx)
            if mask.any():
                token_ids, k_slots = torch.where(mask)
                expert_in = x_flat[token_ids]
                expert_out = expert(expert_in)
                weights = topk_weights[token_ids, k_slots].unsqueeze(-1)
                routed_out_flat.index_add_(0, token_ids, weights * expert_out)
                
        out = routed_out_flat.view(B, T, C)
        
        # Add shared expert output if active
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)
            
        return out, aux_loss, stats


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.moe_mlp = MoEFFN(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss, stats = self.moe_mlp(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss, stats


class MoELanguageModel(nn.Module):
    """Mixture-of-Experts Decoder-Only Language Model."""
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MoETransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        
        total_aux_loss = torch.tensor(0.0, device=idx.device)
        layer_stats = []
        
        for block in self.blocks:
            x, aux_loss, stats = block(x)
            total_aux_loss = total_aux_loss + aux_loss
            layer_stats.append(stats)
            
        total_aux_loss = total_aux_loss / len(self.blocks)
        logits = self.lm_head(self.ln_f(x))
        
        total_loss = None
        metrics = {"aux_loss": total_aux_loss.item(), "layer_stats": layer_stats}
        
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss = lm_loss + self.config.aux_loss_coef * total_aux_loss
            metrics["lm_loss"] = lm_loss.item()
            metrics["total_loss"] = total_loss.item()
            
        return logits, total_loss, metrics

    def count_parameters(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        base_non_moe = self.tok_emb.weight.numel() + self.pos_emb.weight.numel() + sum(p.numel() for p in self.ln_f.parameters())
        if not self.config.tie_weights:
            base_non_moe += self.lm_head.weight.numel()
            
        block_active = 0
        for b in self.blocks:
            attn_p = sum(p.numel() for p in b.attn.parameters()) + sum(p.numel() for p in b.ln_1.parameters())
            moe_ln_p = sum(p.numel() for p in b.ln_2.parameters())
            router_p = sum(p.numel() for p in b.moe_mlp.router.parameters())
            shared_p = sum(p.numel() for p in b.moe_mlp.shared_expert.parameters()) if b.moe_mlp.shared_expert is not None else 0
            routed_p = self.config.top_k * sum(p.numel() for p in b.moe_mlp.experts[0].parameters())
            block_active += (attn_p + moe_ln_p + router_p + shared_p + routed_p)
            
        active_params = base_non_moe + block_active
        return {
            "total_params": total_params,
            "active_params_per_token": active_params,
            "inactive_params_per_token": total_params - active_params,
            "active_ratio": active_params / total_params
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =============================================================================
# 4. SURGICAL DENSE-TO-MoE CONVERSION
# =============================================================================

def convert_dense_to_moe(
    dense_model: DenseLanguageModel,
    num_experts: int = 8,
    top_k: int = 2,
    shared_expert: bool = True,
    aux_loss_coef: float = 0.01,
    expert_init_strategy: Literal["clone_with_noise", "clone_exact", "random"] = "clone_with_noise"
) -> MoELanguageModel:
    """Transfers weights from DenseLanguageModel to MoELanguageModel with warm start."""
    dense_cfg = dense_model.config
    moe_cfg = MoEConfig(
        vocab_size=dense_cfg.vocab_size,
        d_model=dense_cfg.d_model,
        n_layer=dense_cfg.n_layer,
        n_head=dense_cfg.n_head,
        d_ff=dense_cfg.d_ff,
        max_seq_len=dense_cfg.max_seq_len,
        dropout=dense_cfg.dropout,
        bias=dense_cfg.bias,
        tie_weights=dense_cfg.tie_weights,
        num_experts=num_experts,
        top_k=top_k,
        shared_expert=shared_expert,
        aux_loss_coef=aux_loss_coef,
        expert_d_ff=dense_cfg.d_ff,
    )
    
    device = next(dense_model.parameters()).device
    moe_model = MoELanguageModel(moe_cfg).to(device)
    
    with torch.no_grad():
        moe_model.tok_emb.weight.copy_(dense_model.tok_emb.weight)
        moe_model.pos_emb.weight.copy_(dense_model.pos_emb.weight)
        moe_model.ln_f.weight.copy_(dense_model.ln_f.weight)
        if dense_model.ln_f.bias is not None:
            moe_model.ln_f.bias.copy_(dense_model.ln_f.bias)
        if not moe_cfg.tie_weights:
            moe_model.lm_head.weight.copy_(dense_model.lm_head.weight)
            
        for dense_block, moe_block in zip(dense_model.blocks, moe_model.blocks):
            moe_block.ln_1.load_state_dict(dense_block.ln_1.state_dict())
            moe_block.attn.load_state_dict(dense_block.attn.state_dict())
            moe_block.ln_2.load_state_dict(dense_block.ln_2.state_dict())
            
            # Transfer to shared expert
            dense_ffn = dense_block.mlp
            if moe_block.moe_mlp.shared_expert is not None:
                moe_block.moe_mlp.shared_expert.w1.weight.copy_(dense_ffn.w1.weight)
                moe_block.moe_mlp.shared_expert.w2.weight.copy_(dense_ffn.w2.weight)
                
            # Initialize routed experts
            for expert in moe_block.moe_mlp.experts:
                if expert_init_strategy == "clone_with_noise":
                    noise1 = torch.randn_like(dense_ffn.w1.weight) * 0.005
                    noise2 = torch.randn_like(dense_ffn.w2.weight) * 0.005
                    expert.w1.weight.copy_(dense_ffn.w1.weight + noise1)
                    expert.w2.weight.copy_(dense_ffn.w2.weight + noise2)
                elif expert_init_strategy == "clone_exact":
                    expert.w1.weight.copy_(dense_ffn.w1.weight)
                    expert.w2.weight.copy_(dense_ffn.w2.weight)
                    
            nn.init.normal_(moe_block.moe_mlp.router.gate.weight, mean=0.0, std=0.02)
            
    return moe_model


# =============================================================================
# 5. TRAINING LOOP & PLOTTING
# =============================================================================

def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_steps: int,
    phase_name: str,
    lr: float = 5e-4,
    start_step: int = 0
) -> List[Dict[str, Any]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine schedule with linear warmup
    warmup_steps = max(10, num_steps // 10)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    params_info = model.count_parameters()
    print("\n" + "=" * 75)
    print(f"STARTING PHASE: [{phase_name.upper()}]")
    print(f"  * Total Parameters: {params_info['total_params']:,} | Active/Token: {params_info['active_params_per_token']:,} ({params_info['active_ratio']*100:.1f}%)")
    print("=" * 75)
    
    train_iter = iter(train_loader)
    history = []
    t0 = time.time()
    
    for step in range(1, num_steps + 1):
        global_step = start_step + step
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss, metrics = model(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 50 == 0 or step == 1 or step == num_steps:
            elapsed = time.time() - t0
            lm_loss_val = metrics.get("lm_loss", loss.item())
            aux_str = f" | Aux Loss: {metrics['aux_loss']:.4f}" if "aux_loss" in metrics and metrics['aux_loss'] > 0 else ""
            print(
                f"[{phase_name}] Step {step:4d}/{num_steps:4d} (Global: {global_step:4d}) | "
                f"Loss: {loss.item():.4f} | LM Loss: {lm_loss_val:.4f}{aux_str} | "
                f"Grad Norm: {grad_norm:.3f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
            )
            if "layer_stats" in metrics and len(metrics["layer_stats"]) > 0:
                counts = [f"{c:.0f}" for c in metrics["layer_stats"][0]["expert_counts"]]
                print(f"    -> Layer 0 Expert Assignments: {counts}")
                
            history.append({
                "global_step": global_step,
                "lm_loss": lm_loss_val,
                "aux_loss": metrics.get("aux_loss", 0.0),
                "total_params": params_info["total_params"],
                "active_params": params_info["active_params_per_token"]
            })
            t0 = time.time()
            
    return history


def plot_curves(dense_history: List[Dict[str, Any]], moe_history: List[Dict[str, Any]], save_path: str = "loss_curve.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
    
    # Left: LM Loss
    d_steps = [h["global_step"] for h in dense_history]
    d_loss = [h["lm_loss"] for h in dense_history]
    m_steps = [h["global_step"] for h in moe_history]
    m_loss = [h["lm_loss"] for h in moe_history]
    
    ax1.plot(d_steps, d_loss, "o-", color="#1f77b4", linewidth=2.2, label="Dense LM Loss")
    ax1.plot(m_steps, m_loss, "s-", color="#2ca02c", linewidth=2.2, label="MoE LM Loss (Continued)")
    
    if d_steps and m_steps:
        transition = d_steps[-1]
        ax1.axvline(x=transition, color="#d62728", linestyle="--", linewidth=2, label="Dense -> MoE Conversion")
        ax1.text(transition + 5, max(d_loss)*0.85, "MoE Conversion Point", color="#d62728", fontweight="bold")
        
    ax1.set_title("Training Loss Drop (Dense Pretraining -> MoE Continued)", fontweight="bold")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Causal LM Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Right: MoE Aux Loss
    m_aux = [h.get("aux_loss", 0.0) for h in moe_history]
    ax2.plot(m_steps, m_aux, "^-", color="#ff7f0e", linewidth=2.2, label="MoE Aux Loss (Load Balancing)")
    ax2.set_title("MoE Load-Balancing Auxiliary Loss", fontweight="bold")
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Auxiliary Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved loss curve to '{save_path}'")


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main():
    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Detect device: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    
    # Load dataset
    train_ds, val_ds, tokenizer = load_dataset(data_dir="./data", seq_len=128)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)
    
    # Phase 1: Train Dense Model
    dense_cfg = TransformerConfig(vocab_size=tokenizer.vocab_size, d_model=256, n_layer=4, n_head=4, d_ff=1024, max_seq_len=128)
    dense_model = DenseLanguageModel(dense_cfg).to(device)
    
    dense_hist = train_phase(
        model=dense_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_steps=500,
        phase_name="Dense Pretraining",
        lr=5e-4,
        start_step=0
    )
    
    # Phase 2: Convert to MoE
    print("\n" + "#" * 75)
    print("# CONVERTING DENSE MODEL TO MIXTURE-OF-EXPERTS (MoE)...")
    print("#" * 75)
    moe_model = convert_dense_to_moe(dense_model, num_experts=8, top_k=2, shared_expert=True, aux_loss_coef=0.01)
    
    d_params = dense_model.count_parameters()
    m_params = moe_model.count_parameters()
    print(f"Dense Total Params: {d_params['total_params']:,} | Active/Token: {d_params['active_params_per_token']:,}")
    print(f"MoE Total Params:   {m_params['total_params']:,} | Active/Token: {m_params['active_params_per_token']:,} ({m_params['active_ratio']*100:.1f}%)")
    
    # Phase 3: Train MoE Model
    moe_hist = train_phase(
        model=moe_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_steps=500,
        phase_name="MoE Continued",
        lr=4e-4,
        start_step=500
    )
    
    # Phase 4: Plot and Text Generation
    plot_curves(dense_hist, moe_hist, "loss_curve.png")
    
    print("\n" + "=" * 75)
    print("AUTOREGRESSIVE TEXT GENERATION SAMPLE (MoE Model):")
    prompt = "First Citizen:\n"
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    gen_tokens = moe_model.generate(prompt_tokens, max_new_tokens=150, temperature=0.8, top_k=20)
    print(tokenizer.decode(gen_tokens[0].cpu().tolist()))
    print("=" * 75)


if __name__ == "__main__":
    main()
