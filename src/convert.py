"""
Dense-to-MoE Conversion and Weight Transfer Utilities.

Provides surgical conversion from a trained Dense Language Model into a Mixture-of-Experts (MoE)
Language Model, preserving learned representations and enabling seamless continued training.
"""

from typing import Literal
import torch
import torch.nn as nn

from .model import DenseLanguageModel
from .moe import MoELanguageModel, MoEConfig


def convert_dense_to_moe(
    dense_model: DenseLanguageModel,
    num_experts: int = 8,
    top_k: int = 2,
    shared_expert: bool = True,
    aux_loss_coef: float = 0.01,
    expert_init_strategy: Literal["clone_with_noise", "clone_exact", "random"] = "clone_with_noise",
    noise_std: float = 0.005,
) -> MoELanguageModel:
    """
    Converts a trained DenseLanguageModel into a MoELanguageModel.
    
    Transfer Strategy:
    1. Embeddings, LayerNorms, Attention layers, and LM Head are 100% transferred with exact weights.
    2. Shared Expert (if enabled) receives exact weights from the trained Dense FFN.
    3. Routed Experts are initialized according to `expert_init_strategy`:
       - 'clone_with_noise' (Recommended): Clones Dense FFN weights with slight Gaussian perturbation (std=0.005)
         to break symmetry while retaining dense feature representations.
       - 'clone_exact': Clones Dense FFN weights directly without noise.
       - 'random': Initializes routed experts from scratch using Kaiming normal initialization.
    4. Top-k Router is initialized with small random weights (std=0.01) for uniform initial exploration.
    
    Args:
        dense_model: Pre-trained DenseLanguageModel instance
        num_experts: Total number of routed experts per layer (default: 8)
        top_k: Number of active routed experts per token (default: 2)
        shared_expert: Whether to include 1 always-active shared expert (default: True)
        aux_loss_coef: Coefficient for load-balancing auxiliary loss (default: 0.01)
        expert_init_strategy: Strategy for initializing routed experts
        noise_std: Standard deviation of noise added when strategy is 'clone_with_noise'
        
    Returns:
        moe_model: Initialized MoELanguageModel ready for continued training
    """
    dense_cfg = dense_model.config
    
    # 1. Create corresponding MoE configuration
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
    
    # 2. Instantiate target MoE model
    moe_model = MoELanguageModel(moe_cfg)
    
    # Ensure target model is on the same device as the source model
    device = next(dense_model.parameters()).device
    moe_model = moe_model.to(device)
    
    print("=" * 70)
    print("TRANSFORMING DENSE MODEL TO MIXTURE-OF-EXPERTS (MoE)...")
    print(f"  * Layers: {dense_cfg.n_layer}")
    print(f"  * Routed Experts per layer: {num_experts}")
    print(f"  * Top-k active per token: {top_k}")
    print(f"  * Shared Expert enabled: {shared_expert}")
    print(f"  * Expert Init Strategy: {expert_init_strategy}")
    print("=" * 70)
    
    # 3. Copy Token and Positional Embeddings
    with torch.no_grad():
        moe_model.tok_emb.weight.copy_(dense_model.tok_emb.weight)
        moe_model.pos_emb.weight.copy_(dense_model.pos_emb.weight)
        
        # Copy Final LayerNorm
        moe_model.ln_f.weight.copy_(dense_model.ln_f.weight)
        if dense_model.ln_f.bias is not None:
            moe_model.ln_f.bias.copy_(dense_model.ln_f.bias)
            
        # Copy LM Head (if untied)
        if not moe_cfg.tie_weights:
            moe_model.lm_head.weight.copy_(dense_model.lm_head.weight)
            if dense_model.lm_head.bias is not None:
                moe_model.lm_head.bias.copy_(dense_model.lm_head.bias)
                
        # 4. Transfer Block-by-Block
        for layer_idx, (dense_block, moe_block) in enumerate(zip(dense_model.blocks, moe_model.blocks)):
            # Transfer LN1
            moe_block.ln_1.weight.copy_(dense_block.ln_1.weight)
            if dense_block.ln_1.bias is not None:
                moe_block.ln_1.bias.copy_(dense_block.ln_1.bias)
                
            # Transfer Attention parameters
            moe_block.attn.c_attn.weight.copy_(dense_block.attn.c_attn.weight)
            if dense_block.attn.c_attn.bias is not None:
                moe_block.attn.c_attn.bias.copy_(dense_block.attn.c_attn.bias)
            moe_block.attn.c_proj.weight.copy_(dense_block.attn.c_proj.weight)
            if dense_block.attn.c_proj.bias is not None:
                moe_block.attn.c_proj.bias.copy_(dense_block.attn.c_proj.bias)
                
            # Transfer LN2
            moe_block.ln_2.weight.copy_(dense_block.ln_2.weight)
            if dense_block.ln_2.bias is not None:
                moe_block.ln_2.bias.copy_(dense_block.ln_2.bias)
                
            # Transfer FFN to Shared Expert (if enabled)
            dense_ffn = dense_block.mlp
            if moe_block.moe_mlp.shared_expert is not None:
                moe_block.moe_mlp.shared_expert.w1.weight.copy_(dense_ffn.w1.weight)
                moe_block.moe_mlp.shared_expert.w2.weight.copy_(dense_ffn.w2.weight)
                if dense_ffn.w1.bias is not None:
                    moe_block.moe_mlp.shared_expert.w1.bias.copy_(dense_ffn.w1.bias)
                    moe_block.moe_mlp.shared_expert.w2.bias.copy_(dense_ffn.w2.bias)
                    
            # Initialize Routed Experts
            for exp_idx, expert in enumerate(moe_block.moe_mlp.experts):
                if expert_init_strategy == "clone_with_noise":
                    # Clone dense weights with small noise to break symmetry
                    noise_w1 = torch.randn_like(dense_ffn.w1.weight) * noise_std
                    noise_w2 = torch.randn_like(dense_ffn.w2.weight) * noise_std
                    expert.w1.weight.copy_(dense_ffn.w1.weight + noise_w1)
                    expert.w2.weight.copy_(dense_ffn.w2.weight + noise_w2)
                    if dense_ffn.w1.bias is not None:
                        expert.w1.bias.copy_(dense_ffn.w1.bias)
                        expert.w2.bias.copy_(dense_ffn.w2.bias)
                elif expert_init_strategy == "clone_exact":
                    expert.w1.weight.copy_(dense_ffn.w1.weight)
                    expert.w2.weight.copy_(dense_ffn.w2.weight)
                    if dense_ffn.w1.bias is not None:
                        expert.w1.bias.copy_(dense_ffn.w1.bias)
                        expert.w2.bias.copy_(dense_ffn.w2.bias)
                # If 'random', weights were already initialized in Expert.__init__
                
            # Initialize router gate weights with slight random standard deviation for balanced exploration
            nn.init.normal_(moe_block.moe_mlp.router.gate.weight, mean=0.0, std=0.02)
            
    print("Conversion completed successfully! All layers, attention heads, and shared/routed experts wired.")
    return moe_model
