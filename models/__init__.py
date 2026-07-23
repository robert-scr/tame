"""
Molecular Property Prediction Models.

This package provides modular components for molecular property prediction:

Core Building Blocks (models.core):
- BasePredictor: Abstract base class with sklearn-style API
- Graph Encoders: ChebNetEncoder (Chebyshev spectral convolution)
- Pooling: SumPooling, MeanPooling, Set2SetPooling, AttentionPooling
- Fusion: ConcatFusion, CrossModalMHAFusion, GatedFusion, FiLMFusion

High-Level Predictors:
- ChebPredictor: Graph-only baseline using ChebNet
- SEGPredictor: Graph + Text fusion via cross-modal attention
- TAMEPredictor: Graph + Text + RDKit descriptors (element-wise MoE)
- TAMEFusionPredictor: SEG-fused graph/text + descriptors (2-way MoE)

Usage:
    # Simple graph-based prediction
    from models.cheb_predictor import ChebPredictor
    predictor = ChebPredictor()
    predictor.fit(train_smiles, train_labels)
    predictions = predictor.predict_batch(test_smiles)
    
    # Multi-modal prediction with custom components
    from models.core import ChebNetEncoder, AttentionPooling, GatedFusion
    encoder = ChebNetEncoder(in_channels=28, hidden_channels=64)
    pooling = AttentionPooling(input_dim=64)
    fusion = GatedFusion(dim_a=64, dim_b=3072, output_dim=256)
"""

# Core components
from models.core import (
    # Base classes
    BasePredictor,
    BasePredictorConfig,
    BaseGraphEncoder,
    BasePooling,
    BaseFusion,
    # Pooling
    PoolingType,
    SumPooling,
    MeanPooling,
    Set2SetPooling,
    AttentionPooling,
    create_pooling,
    # Fusion
    FusionType,
    ConcatFusion,
    CrossModalMHAFusion,
    GatedFusion,
    FiLMFusion,
    create_fusion,
    # Encoders
    ChebNetEncoder,
    create_graph_encoder,
)

# High-level predictors (optional imports to keep lightweight environments usable)
_optional_import_errors = {}

try:
    from models.cheb_predictor import ChebPredictor, ChebPredictorConfig
except ModuleNotFoundError as exc:
    _optional_import_errors["ChebPredictor"] = exc

try:
    from models.seg_predictor import SEGPredictor, SEGPredictorConfig
except ModuleNotFoundError as exc:
    _optional_import_errors["SEGPredictor"] = exc

try:
    from models.tame_predictor import TAMEPredictor, TAMEPredictorConfig, TriModalElementWiseMoE
except ModuleNotFoundError as exc:
    _optional_import_errors["TAMEPredictor"] = exc

try:
    from models.tame_fusion_predictor import TAMEFusionPredictor, TAMEFusionPredictorConfig, SEGDescriptorElementWiseMoE
except ModuleNotFoundError as exc:
    _optional_import_errors["TAMEFusionPredictor"] = exc
from models.chemeleon_pretraining import (
    DEFAULT_DESCRIPTOR_DIM,
    DEFAULT_KEEP_FRACTION,
    DEFAULT_NODE_MASK_FRACTION,
    CheMeleonChebPretrainer,
    ChebGraphPretrainer,
    ChebNodePretrainer,
    DescriptorBatchCollator,
    DescriptorPretrainingData,
    DescriptorPretrainingDataset,
    DescriptorTargetStats,
    GraphPredictionHead,
    NodePredictionHead,
    attribute_mask_node_features,
    build_dynamic_mask,
    build_cached_model_differentiator,
    create_pretraining_dataloader,
    evaluate_descriptor_pretraining,
    load_or_simulate_unlabeled_smiles,
    load_pretraining_artifacts,
    masked_mse_loss,
    masked_mse_loss_dynamic,
    preprocess_descriptor_targets,
    save_pretraining_artifacts,
    train_descriptor_pretraining,
)

# Low-level building blocks
from models.cheb_net import ChebNet
from models.cheb_layer import ChebLayer, scaled_laplacian_from_edges

__all__ = [
    # Core base classes
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
    # Encoders
    "ChebNetEncoder",
    "create_graph_encoder",
    # High-level predictors (added conditionally below)
    # CheMeleon-style pretraining
    "DEFAULT_DESCRIPTOR_DIM",
    "DEFAULT_KEEP_FRACTION",
    "DEFAULT_NODE_MASK_FRACTION",
    "DescriptorTargetStats",
    "DescriptorPretrainingData",
    "DescriptorPretrainingDataset",
    "DescriptorBatchCollator",
    "CheMeleonChebPretrainer",
    "ChebNodePretrainer",
    "ChebGraphPretrainer",
    "NodePredictionHead",
    "GraphPredictionHead",
    "load_or_simulate_unlabeled_smiles",
    "preprocess_descriptor_targets",
    "create_pretraining_dataloader",
    "attribute_mask_node_features",
    "build_dynamic_mask",
    "build_cached_model_differentiator",
    "masked_mse_loss",
    "masked_mse_loss_dynamic",
    "train_descriptor_pretraining",
    "evaluate_descriptor_pretraining",
    "save_pretraining_artifacts",
    "load_pretraining_artifacts",
    # Building blocks
    "ChebNet",
    "ChebLayer",
    "scaled_laplacian_from_edges",
]

if "ChebPredictor" not in _optional_import_errors:
    __all__.extend(["ChebPredictor", "ChebPredictorConfig"])

if "SEGPredictor" not in _optional_import_errors:
    __all__.extend(["SEGPredictor", "SEGPredictorConfig"])

if "TAMEPredictor" not in _optional_import_errors:
    __all__.extend(["TAMEPredictor", "TAMEPredictorConfig", "TriModalElementWiseMoE"])

if "TAMEFusionPredictor" not in _optional_import_errors:
    __all__.extend(["TAMEFusionPredictor", "TAMEFusionPredictorConfig", "SEGDescriptorElementWiseMoE"])
