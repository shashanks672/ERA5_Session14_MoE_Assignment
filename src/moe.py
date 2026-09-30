"""
Mixture-of-Experts (MoE) Module and Architecture.

Implements:
1. Top-k Gating Router with Softmax distribution
2. Differentiable Load-Balancing Auxiliary Loss (Switch Transformer / ST-MoE formulation)
3. Expert Feed-Forward Networks (MLP Experts)
4. Optional Dedicated Shared Expert (always active, DeepSeek / modern MoE style)
5. MoE Transformer Decoder with parameter accounting (Active vs Total parameters)
6. Detailed runtime statistics on expert load and utilization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TransformerConfig, CausalSelfAttention, DenseFFN


@dataclass
class MoEConfig(TransformerConfig):
    """Configuration for Mixture-of-Experts Language Model."""
    num_experts: int = 8             # Total number of routed experts per layer
    top_k: int = 2                   # Number of routed experts active per token
    shared_expert: bool = True       # Include 1 dedicated shared expert always active
    aux_loss_coef: float = 0.01      # Weight for load balancing auxiliary loss
    expert_d_ff: Optional[int] = None # Intermediate dimension for each expert (defaults to d_ff)

    def __post_init__(self):
        if self.expert_d_ff is None:
            # If expert_d_ff is not specified, default to standard d_ff
            self.expert_d_ff = self.d_ff
        assert self.top_k <= self.num_experts, f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"


class TopKRouter(nn.Module):
    """
    Top-k Gating Router.
    
    Responsibilities:
    1. Computes routing logits for all experts: h(x) = x @ W_gate
    2. Computes normalized softmax probabilities over all experts: P(x) = softmax(h(x))
    3. Selects top-k experts with highest probabilities: indices = topk(P(x), k)
    4. Re-normalizes top-k routing weights so they sum to 1.0 per token
    5. Calculates the differentiable Switch-style load-balancing auxiliary loss to prevent expert collapse.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, aux_loss_coef: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        
        # Router projection weight: (d_model -> num_experts)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # Initialize gate weights with small variance for balanced initial dispatch
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x_flat: Flattened token representations of shape (N_tokens, d_model)
        Returns:
            topk_indices: Shape (N_tokens, top_k) - indices of selected experts
            topk_weights: Shape (N_tokens, top_k) - normalized gating weights
            aux_loss: Scalar auxiliary loss for load balancing
            stats: Dictionary containing expert load and assignment distribution
        """
        N_tokens, _ = x_flat.size()
        
        # 1. Compute routing logits: (N_tokens, num_experts)
        router_logits = self.gate(x_flat)
        
        # 2. Compute softmax routing probabilities: (N_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 3. Top-k Selection: get top-k probabilities and expert indices
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # 4. Re-normalize top-k weights across the selected k experts so sum(weights) = 1.0
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 5. Differentiable Load Balancing Auxiliary Loss (Switch Transformer Formulation)
        # We want tokens to be evenly distributed across all N experts.
        # f_i: fraction of tokens routed to expert i (non-differentiable frequency)
        # P_i: mean router probability assigned to expert i (differentiable probability)
        # Aux Loss = num_experts * sum_i (f_i * P_i)
        
        # One-hot indicator of selected experts for each token: (N_tokens, num_experts)
        # A token selects top_k experts, so sum over top_k dimension
        expert_mask = torch.zeros(N_tokens, self.num_experts, device=x_flat.device, dtype=torch.float32)
        expert_mask.scatter_(1, topk_indices, 1.0)
        
        # f_i: fraction of tokens dispatched to each expert: (num_experts,)
        # Note: sum of f_i is top_k (since each token chooses top_k experts)
        f_i = expert_mask.mean(dim=0)
        
        # P_i: mean softmax probability for each expert over all tokens: (num_experts,)
        # sum of P_i is 1.0
        P_i = router_probs.mean(dim=0)
        
        # Switch Transformer Auxiliary Loss:
        # At perfect balance: f_i = top_k / num_experts, P_i = 1 / num_experts
        # sum(f_i * P_i) = top_k / num_experts
        # num_experts * sum(f_i * P_i) = top_k
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        
        stats = {
            "expert_counts": expert_mask.sum(dim=0).detach().cpu().tolist(),
            "expert_fractions": f_i.detach().cpu().tolist(),
            "expert_probs": P_i.detach().cpu().tolist(),
        }
        
        return topk_indices, topk_weights, aux_loss, stats


class Expert(nn.Module):
    """
    Individual Feed-Forward Expert (MLP).
    
    Architecture:
        x -> Linear(d_model, expert_d_ff) -> GELU -> Linear(expert_d_ff, d_model) -> Dropout -> output
    """
    def __init__(self, d_model: int, expert_d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, expert_d_ff, bias=bias)
        self.w2 = nn.Linear(expert_d_ff, d_model, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize expert weights
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.w1.bias)
            nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts Feed-Forward Network Layer.
    
    Combines:
    - N Routed Experts (Top-k selected dynamically per token)
    - 1 Optional Shared Expert (Always active for every token to capture general representation)
    - Top-k Softmax Router with Load Balancing Loss
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.shared_expert_enabled = config.shared_expert
        
        # 1. Top-k Router
        self.router = TopKRouter(
            d_model=config.d_model,
            num_experts=config.num_experts,
            top_k=config.top_k,
            aux_loss_coef=config.aux_loss_coef
        )
        
        # 2. Routed Experts
        expert_d_ff = config.expert_d_ff or config.d_ff
        self.experts = nn.ModuleList([
            Expert(config.d_model, expert_d_ff, config.dropout, config.bias)
            for _ in range(config.num_experts)
        ])
        
        # 3. Optional Shared Expert
        if self.shared_expert_enabled:
            self.shared_expert = Expert(config.d_model, config.d_ff, config.dropout, config.bias)
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through MoE FFN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_model)
            aux_loss: Auxiliary load-balancing loss for this layer
            stats: Routing diagnostics and load distribution
        """
        B, T, C = x.size()
        x_flat = x.view(-1, C)  # (N_tokens, d_model) where N_tokens = B * T
        N_tokens = x_flat.size(0)
        
        # 1. Route tokens to top-k experts
        topk_indices, topk_weights, aux_loss, stats = self.router(x_flat)
        # topk_indices: (N_tokens, top_k)
        # topk_weights: (N_tokens, top_k)
        
        # 2. Accumulate outputs from routed experts
        routed_out_flat = torch.zeros_like(x_flat)
        
        # Dispatch computation across experts
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens where this expert is among the selected top-k
            # mask: (N_tokens, top_k)
            mask = (topk_indices == expert_idx)
            if mask.any():
                # token_ids: which token row; k_slots: which top-k rank slot (0 to top_k-1)
                token_ids, k_slots = torch.where(mask)
                expert_in = x_flat[token_ids]
                expert_out = expert(expert_in)
                weights = topk_weights[token_ids, k_slots].unsqueeze(-1)
                # Scatter-add weighted expert outputs into routed_out_flat
                routed_out_flat.index_add_(0, token_ids, weights * expert_out)
                
        out = routed_out_flat.view(B, T, C)
        
        # 3. Add Shared Expert output (if enabled)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x)
            out = out + shared_out
            
        return out, aux_loss, stats


class MoETransformerBlock(nn.Module):
    """
    Transformer Decoder Block with Mixture-of-Experts FFN.
    
    Structure:
        x = x + Attention(LN_1(x))
        moe_out, aux_loss, stats = MoEFFN(LN_2(x))
        x = x + moe_out
    """
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
    """
    Mixture-of-Experts Decoder-Only Language Model.
    
    Contains:
    - Token Embedding & Positional Embedding
    - Stack of MoE Transformer Decoder Blocks
    - Final LayerNorm and Linear LM Head
    - Parameter Accounting (Total vs Active parameters per token)
    - Full causal loss + load-balancing auxiliary loss computation
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # MoE Transformer blocks
        self.blocks = nn.ModuleList([MoETransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final norm and language model head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
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
        Forward pass for MoE Language Model.
        
        Args:
            idx: Token indices tensor of shape (batch_size, seq_len)
            targets: Optional ground truth token indices
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            total_loss: lm_loss + aux_loss_coef * aux_loss (or None)
            metrics: Dictionary containing lm_loss, aux_loss, total_loss, and layer routing stats
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_vec = self.tok_emb(idx)
        pos_vec = self.pos_emb(pos)
        x = self.drop(tok_vec + pos_vec)
        
        # Track auxiliary losses across all MoE layers
        total_aux_loss = torch.tensor(0.0, device=idx.device)
        layer_stats: List[Dict[str, Any]] = []
        
        for block in self.blocks:
            x, aux_loss, stats = block(x)
            total_aux_loss = total_aux_loss + aux_loss
            layer_stats.append(stats)
            
        # Average auxiliary loss across layers
        total_aux_loss = total_aux_loss / len(self.blocks)
        
        # Final norm and LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        total_loss = None
        metrics: Dict[str, Any] = {
            "aux_loss": total_aux_loss.item(),
            "layer_stats": layer_stats
        }
        
        if targets is not None:
            # Standard causal LM cross entropy loss
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Total composite loss: LM task loss + weighted load-balancing auxiliary loss
            total_loss = lm_loss + self.config.aux_loss_coef * total_aux_loss
            
            metrics["lm_loss"] = lm_loss.item()
            metrics["total_loss"] = total_loss.item()
            
        return logits, total_loss, metrics

    def count_parameters(self) -> Dict[str, Any]:
        """
        Computes exact parameter counts:
        - Total parameters in model (all weights)
        - Active parameters per token (embeddings + attention + norm + router + top_k experts + shared expert + head)
        - Inactive parameters per token
        - Active ratio (% of total parameters computed per forward pass)
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        # Base shared parameters (Embeddings, Positional encodings, Attention, LayerNorms, LM Head)
        tok_emb_params = self.tok_emb.weight.numel()
        pos_emb_params = self.pos_emb.weight.numel()
        ln_f_params = sum(p.numel() for p in self.ln_f.parameters())
        lm_head_params = 0 if self.config.tie_weights else self.lm_head.weight.numel()
        
        base_non_moe_params = tok_emb_params + pos_emb_params + ln_f_params + lm_head_params
        
        # Per-block accounting
        block_active_params = 0
        for block in self.blocks:
            # LN1 + Attention
            attn_params = sum(p.numel() for p in block.attn.parameters()) + sum(p.numel() for p in block.ln_1.parameters())
            # LN2 + Router
            moe_ln_params = sum(p.numel() for p in block.ln_2.parameters())
            router_params = sum(p.numel() for p in block.moe_mlp.router.parameters())
            
            # Shared expert params (if present, always active)
            shared_expert_params = (
                sum(p.numel() for p in block.moe_mlp.shared_expert.parameters())
                if block.moe_mlp.shared_expert is not None else 0
            )
            
            # Top-k routed expert params active per token
            single_expert_params = sum(p.numel() for p in block.moe_mlp.experts[0].parameters())
            routed_active_params = self.config.top_k * single_expert_params
            
            block_active = attn_params + moe_ln_params + router_params + shared_expert_params + routed_active_params
            block_active_params += block_active
            
        active_params = base_non_moe_params + block_active_params
        inactive_params = total_params - active_params
        active_ratio = active_params / total_params
        
        return {
            "total_params": total_params,
            "active_params_per_token": active_params,
            "inactive_params_per_token": inactive_params,
            "active_ratio": active_ratio
        }

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """Autoregressively generates new tokens given a conditioning prefix."""
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
