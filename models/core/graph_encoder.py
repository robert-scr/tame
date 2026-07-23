"""
Graph neural network encoders for molecular representation learning.

Provides modular graph encoder components that transform node features
into node embeddings through message passing. Encoders are separated
from pooling to allow flexible composition.

Encoders:
- ChebNetEncoder: Chebyshev spectral graph convolution (in-house implementation)

The encoder outputs NODE embeddings (N, hidden_dim), not graph embeddings.
Use pooling modules to aggregate to graph-level representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.base import BaseGraphEncoder

if TYPE_CHECKING:
    from models.core.pooling import BasePooling

from models.cheb_layer import ChebLayer, scaled_laplacian_from_edges


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChebNetEncoderConfig:
    """Configuration for ChebNet encoder."""
    hidden_channels: int = 64
    K: int = 3  # Chebyshev polynomial order
    num_layers: int = 2
    dropout: float = 0.1
    lambda_max: float = 2.0  # Spectral scaling parameter


# =============================================================================
# ChebNet Encoder
# =============================================================================

class ChebNetEncoder(BaseGraphEncoder):
    """
    Chebyshev spectral graph convolution encoder.
    
    Implements the spectral graph convolution from Defferrard et al. (2016)
    using Chebyshev polynomial approximation of spectral filters.
    
    The encoder applies K-order Chebyshev filters to learn node embeddings
    that capture K-hop neighborhood information efficiently.
    
    Architecture:
        for layer in layers:
            h = ChebConv(h, L_tilde)
            h = ReLU(h)
            h = Dropout(h)
    
    Note: This outputs NODE embeddings. Use a pooling module to get
    graph-level embeddings.
    
    Mathematical Background:
        The spectral convolution is defined as g * x = U g(Λ) U^T x
        where U, Λ are eigenvectors/eigenvalues of the graph Laplacian.
        
        ChebNet approximates g(Λ) with Chebyshev polynomials:
            g(Λ) ≈ Σ_{k=0}^{K-1} θ_k T_k(Λ̃)
        
        where Λ̃ = (2/λ_max) Λ - I is the scaled Laplacian.
        
        This allows efficient O(K|E|) computation without eigendecomposition.
    
    Reference:
        Defferrard et al., "Convolutional Neural Networks on Graphs with 
        Fast Localized Spectral Filtering", NeurIPS 2016
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension (also output dimension)
        K: Chebyshev polynomial order (receptive field ~K hops)
        num_layers: Number of ChebConv layers
        dropout: Dropout probability
        lambda_max: Largest eigenvalue for Laplacian scaling (default 2.0)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        K: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        lambda_max: float = 2.0
    ):
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if K < 1:
            raise ValueError("K must be >= 1")
        
        self._output_dim = hidden_channels
        self.dropout = dropout
        self.lambda_max = lambda_max
        self.K = K
        self.num_layers = num_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.layers.append(ChebLayer(in_c, hidden_channels, K=K))
    
    @property
    def output_dim(self) -> int:
        """Dimension of output node embeddings."""
        return self._output_dim
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
        *,
        L_tilde: Optional[torch.sparse.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through ChebNet layers.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge connectivity (2, E)
            edge_weight: Edge weights (E,), or None for uniform weights
            batch: Batch assignment (N,) - not used in encoder, but kept for API consistency
            L_tilde: Optional precomputed scaled Laplacian (for efficiency in batched training)
        
        Returns:
            Node embeddings (N, hidden_channels)
        """
        # Compute scaled Laplacian if not provided
        if L_tilde is None:
            L_tilde = scaled_laplacian_from_edges(
                edge_index=edge_index,
                edge_weight=edge_weight,
                n_nodes=int(x.size(0)),
                lambda_max=self.lambda_max,
                device=x.device,
                dtype=x.dtype,
            )
        
        # Forward through layers
        h = x
        for layer in self.layers:
            h = layer(h, L_tilde)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def get_intermediate_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
        *,
        L_tilde: Optional[torch.sparse.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get embeddings from all layers for analysis/visualization.
        
        Returns:
            List of node embeddings, one per layer (including input)
        """
        if L_tilde is None:
            L_tilde = scaled_laplacian_from_edges(
                edge_index=edge_index,
                edge_weight=edge_weight,
                n_nodes=int(x.size(0)),
                lambda_max=self.lambda_max,
                device=x.device,
                dtype=x.dtype,
            )
        
        embeddings = [x.clone()]
        h = x
        for layer in self.layers:
            h = layer(h, L_tilde)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            embeddings.append(h.clone())
        
        return embeddings


# =============================================================================
# Factory Function
# =============================================================================

def create_graph_encoder(
    encoder_type: str,
    in_channels: int,
    **kwargs
) -> BaseGraphEncoder:
    """
    Factory function to create graph encoders.
    
    Args:
        encoder_type: Type of encoder ("chebnet")
        in_channels: Input feature dimension
        **kwargs: Encoder-specific arguments
            - chebnet: hidden_channels, K, num_layers, dropout, lambda_max
    
    Returns:
        BaseGraphEncoder instance
        
    Example:
        >>> encoder = create_graph_encoder("chebnet", in_channels=28, hidden_channels=64, K=3)
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "chebnet":
        return ChebNetEncoder(
            in_channels=in_channels,
            hidden_channels=kwargs.get("hidden_channels", 64),
            K=kwargs.get("K", 3),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
            lambda_max=kwargs.get("lambda_max", 2.0)
        )
    
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: ['chebnet']"
        )


# =============================================================================
# Utility: Composed Graph Model
# =============================================================================

class GraphModel(nn.Module):
    """
    Composed graph model: Encoder + Pooling + Head.
    
    Convenience class that combines encoder, pooling, and prediction head
    into a single module. For more flexibility, use the components separately.
    
    Example:
        >>> model = GraphModel(
        ...     encoder=ChebNetEncoder(28, 64),
        ...     pooling=MeanPooling(64),
        ...     head=MLPHead(64, hidden_dim=128, output_dim=1)
        ... )
        >>> predictions = model(x, edge_index, edge_weight, batch)
    """
    
    def __init__(
        self,
        encoder: BaseGraphEncoder,
        pooling: "BasePooling",  # Forward reference
        head: nn.Module
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = head
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
        **encoder_kwargs
    ) -> torch.Tensor:
        """
        Full forward pass: encode -> pool -> predict.
        
        Returns:
            Predictions (B, output_dim)
        """
        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_weight, batch, **encoder_kwargs)
        
        # Pool to graph level
        graph_emb = self.pooling(node_emb, batch)
        
        # Predict
        return self.head(graph_emb)
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
        **encoder_kwargs
    ) -> torch.Tensor:
        """
        Get graph-level embeddings (without prediction head).
        
        Useful for downstream fusion with other modalities.
        
        Returns:
            Graph embeddings (B, pooling.output_dim)
        """
        node_emb = self.encoder(x, edge_index, edge_weight, batch, **encoder_kwargs)
        return self.pooling(node_emb, batch)
