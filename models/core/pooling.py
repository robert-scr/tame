"""
Pooling strategies for graph-level representation learning.

Provides modular pooling components that can be swapped to experiment
with different aggregation strategies for molecular property prediction.

Pooling Types:
- SumPooling: Sum node features (sensitive to graph size)
- MeanPooling: Average node features (size-invariant) 
- Set2SetPooling: Iterative attention-based pooling (Vinyals et al., 2015)
- AttentionPooling: Learnable importance weighting per node
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.base import BasePooling


class PoolingType(str, Enum):
    """Enumeration of available pooling strategies."""
    SUM = "sum"
    MEAN = "mean"
    SET2SET = "set2set"
    ATTENTION = "attention"


# =============================================================================
# Helper Functions (adapted from cheb_net.py)
# =============================================================================

def _scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int
) -> torch.Tensor:
    """
    Scatter sum using native PyTorch (autograd safe).

    Args:
        src: Source tensor (N,) or (N, C)
        index: Index tensor (N,)
        dim_size: Size of output dimension 0

    Returns:
        Summed tensor (dim_size,) or (dim_size, C)
    """
    if src.dim() == 1:
        src = src.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False

    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    expanded_index = index.unsqueeze(-1).expand_as(src)
    out = out.scatter_add(0, expanded_index, src)

    if squeeze:
        out = out.squeeze(-1)
    return out


def _scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    num_graphs: int
) -> torch.Tensor:
    """
    Compute softmax over groups defined by index (numerically stable).

    Args:
        src: Source tensor (N,)
        index: Index tensor (N,) indicating group membership
        num_graphs: Number of groups

    Returns:
        Softmax values (N,)
    """
    # Compute max per group for numerical stability
    src_max = torch.full((num_graphs,), float("-inf"), device=src.device, dtype=src.dtype)
    src_max = src_max.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    src_max = torch.where(src_max == float("-inf"), torch.zeros_like(src_max), src_max)

    # Subtract max and exponentiate
    src_shifted = src - src_max[index]
    exp_src = torch.exp(src_shifted)

    # Sum per group
    exp_sum = _scatter_sum(exp_src, index, num_graphs)

    # Normalize
    return exp_src / (exp_sum[index] + 1e-8)


# =============================================================================
# Pooling Implementations
# =============================================================================

class SumPooling(BasePooling):
    """
    Sum pooling: aggregate node features by summation.
    
    Properties:
    - Output is sensitive to graph size (larger graphs → larger embeddings)
    - Good when total information matters (e.g., counting functional groups)
    - Computationally cheapest
    
    For molecular property prediction, sum pooling can be useful when
    the property scales with molecular size (e.g., some physical properties).
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self._output_dim = input_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            raise ValueError("batch must be non-empty")
        
        if batch.dtype != torch.long:
            batch = batch.long()
        
        B = int(batch.max().item()) + 1
        out = torch.zeros((B, x.size(-1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        
        return out


class MeanPooling(BasePooling):
    """
    Mean pooling: aggregate node features by averaging.
    
    Properties:
    - Output is size-invariant (same scale regardless of graph size)
    - Good default choice for molecular property prediction
    - Robust to varying molecule sizes
    
    For molecular property prediction, mean pooling is often preferred
    because properties like solubility are roughly size-invariant
    (per-atom contributions average out).
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self._output_dim = input_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            raise ValueError("batch must be non-empty")
        
        if batch.dtype != torch.long:
            batch = batch.long()
        
        B = int(batch.max().item()) + 1
        
        # Sum
        out = torch.zeros((B, x.size(-1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        
        # Count nodes per graph
        counts = torch.zeros((B,), device=x.device, dtype=x.dtype)
        ones = torch.ones((batch.numel(),), device=x.device, dtype=x.dtype)
        counts.index_add_(0, batch, ones)
        counts = counts.clamp_min(1.0).unsqueeze(-1)
        
        return out / counts


class Set2SetPooling(BasePooling):
    """
    Set2Set pooling (Vinyals et al., 2015).
    
    Uses an LSTM to iteratively attend over node features,
    producing an order-aware graph embedding. Output dimension
    is 2x input dimension.
    
    Properties:
    - Captures complex node relationships through iterative attention
    - More expressive than simple sum/mean
    - Higher computational cost
    - Output dim = 2 * input_dim
    
    For molecular property prediction, Set2Set can capture
    interactions between distant functional groups that simple
    pooling might miss.
    
    Reference:
        Vinyals et al., "Order Matters: Sequence to sequence for sets", 2015
    """
    
    def __init__(
        self,
        input_dim: int,
        n_iters: int = 3,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=self._output_dim,
            hidden_size=input_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype != torch.long:
            batch = batch.long()
        
        B = int(batch.max().item()) + 1
        device = x.device
        dtype = x.dtype
        
        # Initialize LSTM hidden state
        h = torch.zeros(self.n_layers, B, self._input_dim, device=device, dtype=dtype)
        c = torch.zeros(self.n_layers, B, self._input_dim, device=device, dtype=dtype)
        
        # Initialize query vector q_star
        q_star = torch.zeros(B, self._output_dim, device=device, dtype=dtype)
        
        for _ in range(self.n_iters):
            # LSTM step: input is q_star, output is new query q
            q, (h, c) = self.lstm(q_star.unsqueeze(1), (h, c))
            q = q.squeeze(1)  # (B, input_dim)
            
            # Compute attention scores: e_i = x_i · q_{batch[i]}
            q_expanded = q[batch]  # (N, input_dim)
            e = (x * q_expanded).sum(dim=-1)  # (N,)
            
            # Softmax over nodes within each graph
            a = _scatter_softmax(e, batch, B)  # (N,)
            
            # Weighted sum of node features per graph (read vector)
            r = _scatter_sum(a.unsqueeze(-1) * x, batch, B)  # (B, input_dim)
            
            # Concatenate q and r to form new q_star
            q_star = torch.cat([q, r], dim=-1)  # (B, 2 * input_dim)
        
        return self.dropout(q_star)


class AttentionPooling(BasePooling):
    """
    Learnable attention-based pooling.
    
    Computes importance scores for each node using a learnable
    attention network, then aggregates with weighted sum.
    
    Properties:
    - Learns which atoms/nodes are most important for the prediction
    - More interpretable than Set2Set (attention weights can be visualized)
    - Moderate computational cost
    - Same output dimension as input
    
    For molecular property prediction, this can learn to focus on
    pharmacophores or other structurally important regions.
    
    Architecture:
        score = MLP(node_features)
        weights = softmax(scores) per graph
        output = sum(weights * node_features)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self._output_dim = input_dim
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype != torch.long:
            batch = batch.long()
        
        B = int(batch.max().item()) + 1
        
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (N,)
        
        # Softmax per graph
        weights = _scatter_softmax(scores, batch, B)  # (N,)
        
        # Weighted sum
        weighted_x = weights.unsqueeze(-1) * x  # (N, input_dim)
        output = _scatter_sum(weighted_x, batch, B)  # (B, input_dim)
        
        return output
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Returns:
            weights: (N,) attention weight per node
        """
        if batch.dtype != torch.long:
            batch = batch.long()
        
        B = int(batch.max().item()) + 1
        scores = self.attention(x).squeeze(-1)
        weights = _scatter_softmax(scores, batch, B)
        
        return weights


# =============================================================================
# Factory Function
# =============================================================================

def create_pooling(
    pool_type: str,
    input_dim: int,
    **kwargs
) -> BasePooling:
    """
    Factory function to create pooling modules.
    
    Args:
        pool_type: One of "sum", "mean", "set2set", "attention"
        input_dim: Dimension of input node embeddings
        **kwargs: Additional arguments for specific pooling types
            - set2set: n_iters (default 3), n_layers (default 1)
            - attention: hidden_dim (default 64), dropout (default 0.1)
    
    Returns:
        BasePooling instance
        
    Example:
        >>> pooling = create_pooling("attention", input_dim=64, hidden_dim=32)
    """
    pool_type = pool_type.lower()
    
    if pool_type == PoolingType.SUM.value:
        return SumPooling(input_dim)
    
    elif pool_type == PoolingType.MEAN.value:
        return MeanPooling(input_dim)
    
    elif pool_type == PoolingType.SET2SET.value:
        n_iters = kwargs.get("n_iters", 3)
        n_layers = kwargs.get("n_layers", 1)
        dropout = kwargs.get("dropout", 0.0)
        return Set2SetPooling(input_dim, n_iters=n_iters, n_layers=n_layers, dropout=dropout)
    
    elif pool_type == PoolingType.ATTENTION.value:
        hidden_dim = kwargs.get("hidden_dim", 64)
        dropout = kwargs.get("dropout", 0.1)
        return AttentionPooling(input_dim, hidden_dim=hidden_dim, dropout=dropout)
    
    else:
        raise ValueError(
            f"Unknown pooling type: {pool_type}. "
            f"Choose from: {[p.value for p in PoolingType]}"
        )


# =============================================================================
# Backward Compatibility
# =============================================================================

def global_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    *,
    pool: str = "mean"
) -> torch.Tensor:
    """
    Legacy global pooling function for backward compatibility.
    
    Use create_pooling() for new code.
    """
    if pool == "sum":
        pooling = SumPooling(x.size(-1))
    elif pool == "mean":
        pooling = MeanPooling(x.size(-1))
    else:
        raise ValueError(f"Unknown pool={pool!r}. For 'set2set', use Set2SetPooling directly.")
    
    return pooling(x, batch)
