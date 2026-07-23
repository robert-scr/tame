"""
Core building blocks for modular molecular property prediction.

This module provides abstract base classes and concrete implementations
for graph encoding, pooling, and multi-modal fusion components.
"""

from models.core.base import (
    BasePredictor,
    BasePredictorConfig,
    BaseGraphEncoder,
    BasePooling,
    BaseFusion,
)
from models.core.pooling import (
    PoolingType,
    SumPooling,
    MeanPooling,
    Set2SetPooling,
    AttentionPooling,
    create_pooling,
)
from models.core.fusion import (
    FusionType,
    ConcatFusion,
    CrossModalMHAFusion,
    GatedFusion,
    FiLMFusion,
    create_fusion,
)
from models.core.graph_encoder import (
    ChebNetEncoder,
    create_graph_encoder,
)

__all__ = [
    # Base classes
    "BasePredictor",
    "BasePredictorConfig",
    "BaseGraphEncoder",
    "BasePooling",
    "BaseFusion",
    # Pooling
    "PoolingType",
    "SumPooling",
    "MeanPooling",
    "Set2SetPooling",
    "AttentionPooling",
    "create_pooling",
    # Fusion
    "FusionType",
    "ConcatFusion",
    "CrossModalMHAFusion",
    "GatedFusion",
    "FiLMFusion",
    "create_fusion",
    # Graph encoders
    "ChebNetEncoder",
    "create_graph_encoder",
]
