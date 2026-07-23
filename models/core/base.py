"""
Abstract base classes for molecular property prediction components.

Defines interfaces for:
- BasePredictor: sklearn-style fit/predict API for all predictors
- BaseGraphEncoder: Graph neural network backbones
- BasePooling: Node-to-graph aggregation strategies  
- BaseFusion: Multi-modal embedding fusion methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Configuration Base Classes
# =============================================================================

@dataclass
class BasePredictorConfig:
    """
    Base configuration class for predictors.
    
    All predictor configs should inherit from this to ensure
    consistent serialization and validation.
    """
    
    # Task type: "regression" or "classification"
    task: str = "regression"
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got '{self.task}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BasePredictorConfig":
        """Create config from dictionary."""
        return cls(**d)


# =============================================================================
# Predictor Base Class
# =============================================================================

class BasePredictor(ABC):
    """
    Abstract base class for all molecular property predictors.
    
    Provides a consistent sklearn-style API:
    - fit(): Train the model
    - predict(): Predict single molecule
    - predict_batch(): Predict multiple molecules
    - save()/load(): Model persistence
    
    All concrete predictors (ChebPredictor, SEGPredictor, TAMEPredictor, etc.)
    should inherit from this class.
    
    Example:
        >>> predictor = ChebPredictor(config)
        >>> history = predictor.fit(train_smiles, train_labels)
        >>> prediction = predictor.predict("CCO")
        >>> predictions = predictor.predict_batch(test_smiles)
    """
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        pass
    
    @abstractmethod
    def fit(
        self,
        smiles_list: List[str],
        labels: List[float],
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the predictor.
        
        Args:
            smiles_list: Training SMILES strings
            labels: Training target values
            val_smiles: Optional validation SMILES
            val_labels: Optional validation labels
            **kwargs: Additional training parameters (epochs, batch_size, etc.)
            
        Returns:
            Dictionary containing training history (losses, metrics per epoch)
        """
        pass
    
    @abstractmethod
    def predict(self, smiles: str) -> Optional[float]:
        """
        Predict property for a single molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Predicted value, or None if SMILES is invalid
            
        Raises:
            RuntimeError: If model is not fitted
        """
        pass
    
    @abstractmethod
    def predict_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict properties for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference
            
        Returns:
            NumPy array of predictions (NaN for invalid SMILES)
            
        Raises:
            RuntimeError: If model is not fitted
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: File path for saving
            
        Raises:
            RuntimeError: If model is not fitted
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path: File path to load from
        """
        pass
    
    def _check_fitted(self) -> None:
        """Raise RuntimeError if model is not fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction. "
                "Call fit() first."
            )


# =============================================================================
# Graph Encoder Base Class
# =============================================================================

class BaseGraphEncoder(ABC, nn.Module):
    """
    Abstract base class for graph neural network encoders.
    
    A graph encoder transforms node features into node embeddings
    through message passing. The output is node-level embeddings,
    NOT graph-level embeddings (pooling is separate).
    
    This separation allows mixing different encoders with different
    pooling strategies.
    
    Example implementations:
    - ChebNetEncoder (Chebyshev spectral convolution)
    - GCNEncoder (standard GCN)
    - GATEncoder (Graph Attention)
    - MPNNEncoder (Message Passing Neural Network)
    """
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of output node embeddings."""
        pass
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        batch: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features (N, in_features)
            edge_index: Edge connectivity (2, E)
            edge_weight: Optional edge weights (E,)
            batch: Batch assignment for each node (N,)
            **kwargs: Encoder-specific arguments (e.g., precomputed Laplacian)
            
        Returns:
            Node embeddings (N, output_dim)
        """
        pass


# =============================================================================
# Pooling Base Class
# =============================================================================

class BasePooling(ABC, nn.Module):
    """
    Abstract base class for graph pooling operations.
    
    Pooling aggregates node embeddings into graph-level embeddings.
    Different pooling strategies capture different graph properties:
    
    - SumPooling: Total information, sensitive to graph size
    - MeanPooling: Average information, size-invariant
    - Set2Set: Order-aware iterative attention
    - AttentionPooling: Learnable node importance
    
    Example:
        >>> pooling = MeanPooling(input_dim=64)
        >>> node_emb = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> batch = torch.randint(0, 10, (100,))  # 10 graphs
        >>> graph_emb = pooling(node_emb, batch)  # (10, 64)
    """
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of output graph embeddings."""
        pass
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate node embeddings to graph embeddings.
        
        Args:
            x: Node embeddings (N, input_dim)
            batch: Graph assignment for each node (N,), values in [0, B-1]
            
        Returns:
            Graph embeddings (B, output_dim)
        """
        pass


# =============================================================================
# Fusion Base Class
# =============================================================================

class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for multi-modal fusion.
    
    Fusion combines embeddings from different modalities (e.g., graph + text)
    into a unified representation. Different fusion strategies have different
    inductive biases:
    
    - ConcatFusion: Simple, lets downstream layers learn interactions
    - CrossModalMHA: Rich bidirectional attention, more parameters
    - GatedFusion: Learns modality importance per sample
    - FiLMFusion: One modality conditions the other (asymmetric)
    
    Example:
        >>> fusion = CrossModalMHAFusion(graph_dim=64, text_dim=3072, output_dim=256)
        >>> graph_emb = torch.randn(8, 64)
        >>> text_emb = torch.randn(8, 3072)
        >>> fused = fusion(graph_emb, text_emb)  # (8, 256)
    """
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of fused output embeddings."""
        pass
    
    @abstractmethod
    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse two embedding tensors.
        
        Args:
            emb_a: First modality embeddings (B, dim_a)
            emb_b: Second modality embeddings (B, dim_b)
            
        Returns:
            Fused embeddings (B, output_dim)
        """
        pass


# =============================================================================
# Prediction Head Base Class
# =============================================================================

class BasePredictionHead(ABC, nn.Module):
    """
    Abstract base class for prediction heads.
    
    Maps fused embeddings to final predictions.
    Usually a simple MLP, but can be more complex for
    multi-task or uncertainty estimation.
    """
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of predictions (usually 1 for regression)."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map embeddings to predictions.
        
        Args:
            x: Input embeddings (B, input_dim)
            
        Returns:
            Predictions (B, output_dim)
        """
        pass


class LinearHead(BasePredictionHead):
    """
    Simple linear prediction head (no hidden layer).
    
    Architecture: Linear [-> Sigmoid for classification]
    
    Use this for strong regularization when overfitting is a problem.
    Forces the fusion layer to produce directly task-relevant features.
    
    Args:
        input_dim: Input embedding dimension
        output_dim: Output dimension (usually 1)
        task: "regression" or "classification" (adds sigmoid for classification)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        task: str = "regression"
    ):
        super().__init__()
        self._output_dim = output_dim
        self._task = task
        self.linear = nn.Linear(input_dim, output_dim)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def task(self) -> str:
        return self._task
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self._task == "classification":
            out = torch.sigmoid(out)
        return out


class MLPHead(BasePredictionHead):
    """
    Standard MLP prediction head.
    
    Architecture: Linear -> ReLU -> Dropout -> Linear [-> Sigmoid for classification]
    
    Args:
        input_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (usually 1)
        dropout: Dropout probability
        task: "regression" or "classification" (adds sigmoid for classification)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
        task: str = "regression"
    ):
        super().__init__()
        self._output_dim = output_dim
        self._task = task
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def task(self) -> str:
        return self._task
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self._task == "classification":
            out = torch.sigmoid(out)
        return out
