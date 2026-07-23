"""
Multi-modal fusion strategies for combining embeddings from different modalities.

Provides modular fusion components for combining graph embeddings (molecular structure)
with text embeddings (chemical descriptions) and other modalities.

Fusion Types:
- ConcatFusion: Simple concatenation + projection
- CrossModalMHAFusion: Bidirectional cross-attention (current SEG approach)
- GatedFusion: Learned modality weighting per sample
- FiLMFusion: Feature-wise Linear Modulation (text conditions graph)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.base import BaseFusion


class FusionType(str, Enum):
    """Enumeration of available fusion strategies."""
    CONCAT = "concat"
    CROSS_MHA = "cross_mha"
    GATED = "gated"
    FILM = "film"
    BILINEAR = "bilinear"


# =============================================================================
# Fusion Implementations
# =============================================================================

class ConcatFusion(BaseFusion):
    """
    Simple concatenation fusion with projection.
    
    Concatenates embeddings from both modalities and projects
    to output dimension. Simple baseline that lets downstream
    layers learn the interactions.
    
    Architecture:
        fused = Linear([emb_a; emb_b])
        
    Properties:
    - Simplest fusion, fewest parameters
    - No explicit modality interaction modeling
    - Good baseline; surprisingly effective in many cases
    """
    
    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self._output_dim = output_dim
        
        self.proj = nn.Sequential(
            nn.Linear(dim_a + dim_b, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        concatenated = torch.cat([emb_a, emb_b], dim=-1)
        return self.proj(concatenated)


class CrossModalMHAFusion(BaseFusion):
    """
    Bidirectional Cross-Modal Multi-Head Attention fusion.
    
    Both modalities attend to each other:
    - Graph embedding attends to text (what text info is relevant to structure?)
    - Text embedding attends to graph (what structure info is relevant to text?)
    
    Then combines the attended representations.
    
    Architecture:
        graph_proj = Linear(graph_emb)
        text_proj = Linear(text_emb)
        
        graph_attended = CrossAttn(Q=graph, K=text, V=text) + graph_proj
        text_attended = CrossAttn(Q=text, K=graph, V=graph) + text_proj
        
        fused = MLP([graph_attended; text_attended])
        
    Properties:
    - Rich bidirectional interaction modeling
    - More parameters than simple fusion
    - Effective for graph-text fusion in molecular domains
    
    This is the fusion strategy used in the original SEG predictor.
    
    Reference:
        Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    """
    
    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self._output_dim = d_model
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Projection layers to align dimensions
        self.graph_proj = nn.Linear(graph_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Bidirectional cross-attention
        self.graph_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.text_to_graph_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm_graph = nn.LayerNorm(d_model)
        self.norm_text = nn.LayerNorm(d_model)
        
        # Output fusion
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        emb_a: torch.Tensor,  # graph_emb
        emb_b: torch.Tensor   # text_emb
    ) -> torch.Tensor:
        """
        Fuse graph and text embeddings.
        
        Args:
            emb_a: Graph embeddings (batch_size, graph_dim)
            emb_b: Text embeddings (batch_size, text_dim)
            
        Returns:
            fused: (batch_size, d_model)
        """
        # Project to common dimension
        graph_proj = self.graph_proj(emb_a)
        text_proj = self.text_proj(emb_b)
        
        # Add sequence dimension for attention (B, 1, d_model)
        graph_seq = graph_proj.unsqueeze(1)
        text_seq = text_proj.unsqueeze(1)
        
        # Bidirectional cross-attention
        graph_attended, _ = self.graph_to_text_attn(
            query=graph_seq, key=text_seq, value=text_seq
        )
        text_attended, _ = self.text_to_graph_attn(
            query=text_seq, key=graph_seq, value=graph_seq
        )
        
        # Residual + norm
        graph_out = self.norm_graph(graph_proj + self.dropout(graph_attended.squeeze(1)))
        text_out = self.norm_text(text_proj + self.dropout(text_attended.squeeze(1)))
        
        # Fuse and project
        fused = torch.cat([graph_out, text_out], dim=-1)
        fused = self.output_proj(fused)
        
        return fused
    
    def forward_with_attention(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass that also returns attention weights for interpretability.
        
        Returns:
            fused: Fused embeddings
            attention_weights: (graph_to_text_attn, text_to_graph_attn)
        """
        graph_proj = self.graph_proj(emb_a)
        text_proj = self.text_proj(emb_b)
        
        graph_seq = graph_proj.unsqueeze(1)
        text_seq = text_proj.unsqueeze(1)
        
        graph_attended, attn_g2t = self.graph_to_text_attn(
            query=graph_seq, key=text_seq, value=text_seq
        )
        text_attended, attn_t2g = self.text_to_graph_attn(
            query=text_seq, key=graph_seq, value=graph_seq
        )
        
        graph_out = self.norm_graph(graph_proj + self.dropout(graph_attended.squeeze(1)))
        text_out = self.norm_text(text_proj + self.dropout(text_attended.squeeze(1)))
        
        fused = torch.cat([graph_out, text_out], dim=-1)
        fused = self.output_proj(fused)
        
        return fused, (attn_g2t, attn_t2g)


class GatedFusion(BaseFusion):
    """
    Gated fusion with learned modality weighting.
    
    Learns a gate that determines how much each modality
    contributes to the fused representation, conditioned
    on both inputs.
    
    Architecture:
        gate = sigmoid(Linear([emb_a; emb_b]))
        fused = gate * proj_a(emb_a) + (1 - gate) * proj_b(emb_b)
        
    Properties:
    - Adaptive modality weighting per sample
    - Can learn to ignore one modality when the other is more informative
    - Interpretable gate values show modality importance
    
    For molecular property prediction, this can learn to rely more
    on graph structure for some molecules and more on text for others.
    """
    
    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self._output_dim = output_dim
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(dim_a + dim_b, output_dim),
            nn.Sigmoid()
        )
        
        # Projection for each modality
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_b = nn.Sequential(
            nn.Linear(dim_b, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        # Compute gate
        g = self.gate(torch.cat([emb_a, emb_b], dim=-1))
        
        # Project each modality
        proj_a = self.proj_a(emb_a)
        proj_b = self.proj_b(emb_b)
        
        # Gated combination
        return g * proj_a + (1 - g) * proj_b
    
    def forward_with_gate(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns gate values for interpretability.
        
        Returns:
            fused: Fused embeddings
            gate: Gate values (B, output_dim), higher = more weight on emb_a
        """
        g = self.gate(torch.cat([emb_a, emb_b], dim=-1))
        proj_a = self.proj_a(emb_a)
        proj_b = self.proj_b(emb_b)
        fused = g * proj_a + (1 - g) * proj_b
        return fused, g


class FiLMFusion(BaseFusion):
    """
    Feature-wise Linear Modulation (FiLM) fusion.
    
    One modality (conditioning) modulates the other (target) via
    learned scale (gamma) and shift (beta) parameters.
    
    Architecture:
        gamma, beta = Linear(emb_b).chunk(2)
        fused = (1 + gamma) * emb_a + beta
        
    Properties:
    - Asymmetric: one modality conditions the other
    - Effective when one modality should "guide" the other
    - Widely used in vision-language models
    
    For molecular property prediction, text can condition how
    graph features are interpreted (e.g., text mentions "polar"
    → emphasize polar-related graph features).
    
    Reference:
        Perez et al., "FiLM: Visual Reasoning with a General 
        Conditioning Layer", AAAI 2018
    """
    
    def __init__(
        self,
        target_dim: int,      # Modality being modulated (typically graph)
        conditioning_dim: int, # Modality providing conditioning (typically text)
        dropout: float = 0.1
    ):
        super().__init__()
        self._output_dim = target_dim
        
        # FiLM generator: produces gamma and beta
        self.film_gen = nn.Sequential(
            nn.Linear(conditioning_dim, target_dim * 2),
            nn.Dropout(dropout)
        )
        
        # Optional projection for target
        self.target_proj = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        emb_a: torch.Tensor,  # Target (graph)
        emb_b: torch.Tensor   # Conditioning (text)
    ) -> torch.Tensor:
        # Generate FiLM parameters
        film_params = self.film_gen(emb_b)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Project target
        target = self.target_proj(emb_a)
        
        # Apply FiLM: (1 + gamma) * target + beta
        return (1 + gamma) * target + beta
    
    def forward_with_params(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass that also returns FiLM parameters for interpretability.
        
        Returns:
            fused: Fused embeddings
            film_params: (gamma, beta) scaling and shifting parameters
        """
        film_params = self.film_gen(emb_b)
        gamma, beta = film_params.chunk(2, dim=-1)
        target = self.target_proj(emb_a)
        fused = (1 + gamma) * target + beta
        return fused, (gamma, beta)


class BilinearFusion(BaseFusion):
    """
    Bilinear fusion for modeling multiplicative interactions.
    
    Computes bilinear interaction between modalities, capturing
    second-order feature interactions.
    
    Architecture:
        interaction = emb_a @ W @ emb_b.T  (simplified)
        fused = Linear([emb_a; emb_b; interaction])
        
    Properties:
    - Captures multiplicative interactions between features
    - More expressive than additive fusion
    - Quadratic in feature dimensions (can be expensive)
    
    Note: Uses low-rank approximation to reduce parameters.
    """
    
    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        output_dim: int,
        rank: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self._output_dim = output_dim
        
        # Low-rank bilinear: U @ V.T instead of full W
        self.U = nn.Linear(dim_a, rank, bias=False)
        self.V = nn.Linear(dim_b, rank, bias=False)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim_a + dim_b + rank, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        # Low-rank bilinear interaction
        u = self.U(emb_a)  # (B, rank)
        v = self.V(emb_b)  # (B, rank)
        interaction = u * v  # Element-wise (B, rank)
        
        # Combine all
        combined = torch.cat([emb_a, emb_b, interaction], dim=-1)
        return self.output_proj(combined)


# =============================================================================
# Factory Function
# =============================================================================

def create_fusion(
    fusion_type: str,
    dim_a: int,
    dim_b: int,
    output_dim: int,
    **kwargs
) -> BaseFusion:
    """
    Factory function to create fusion modules.
    
    Args:
        fusion_type: One of "concat", "cross_mha", "gated", "film", "bilinear"
        dim_a: Dimension of first modality (typically graph)
        dim_b: Dimension of second modality (typically text)
        output_dim: Desired output dimension
        **kwargs: Additional arguments for specific fusion types
            - cross_mha: n_heads (default 8), dropout (default 0.1)
            - gated, concat, film: dropout (default 0.1)
            - bilinear: rank (default 32), dropout (default 0.1)
    
    Returns:
        BaseFusion instance
        
    Example:
        >>> fusion = create_fusion("cross_mha", dim_a=64, dim_b=3072, output_dim=256, n_heads=8)
    """
    fusion_type = fusion_type.lower()
    dropout = kwargs.get("dropout", 0.1)
    
    if fusion_type == FusionType.CONCAT.value:
        return ConcatFusion(dim_a, dim_b, output_dim, dropout=dropout)
    
    elif fusion_type == FusionType.CROSS_MHA.value:
        n_heads = kwargs.get("n_heads", 8)
        return CrossModalMHAFusion(
            graph_dim=dim_a,
            text_dim=dim_b,
            d_model=output_dim,
            n_heads=n_heads,
            dropout=dropout
        )
    
    elif fusion_type == FusionType.GATED.value:
        return GatedFusion(dim_a, dim_b, output_dim, dropout=dropout)
    
    elif fusion_type == FusionType.FILM.value:
        return FiLMFusion(
            target_dim=dim_a,
            conditioning_dim=dim_b,
            dropout=dropout
        )
    
    elif fusion_type == FusionType.BILINEAR.value:
        rank = kwargs.get("rank", 32)
        return BilinearFusion(dim_a, dim_b, output_dim, rank=rank, dropout=dropout)
    
    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Choose from: {[f.value for f in FusionType]}"
        )
