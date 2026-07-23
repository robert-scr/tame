"""SEG (Structured Embedding Fusion) predictor for molecular property prediction.

Fuses graph embeddings (ChebNet) with text embeddings (Azure OpenAI)
using cross-modal multi-head attention.

This module uses modular components from models.core for flexibility:
- Graph encoding via ChebNetEncoder
- Configurable pooling strategies
- Configurable fusion methods (CrossModalMHA, Gated, FiLM, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.base import BasePredictor, BasePredictorConfig, MLPHead, LinearHead
from models.core.pooling import create_pooling, BasePooling
from models.core.fusion import create_fusion, BaseFusion
from models.core.graph_encoder import ChebNetEncoder
from utils.molecular_graph import smiles_to_graph
from utils.batched_mol_graph import batch_graphs


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SEGPredictorConfig(BasePredictorConfig):
    """Configuration for SEGPredictor."""
    # Task type (inherited from base, but explicit for clarity)
    task: str = "regression"  # "regression" or "classification"
    num_tasks: int = 1  # Number of output tasks (>1 for multi-label classification)
    
    # Graph encoder config
    hidden_channels: int = 64
    K: int = 3
    num_layers: int = 2
    dropout: float = 0.1
    lambda_max: float = 2.0
    
    # Pooling config
    pool: str = "mean"  # "sum", "mean", "set2set", "attention"
    set2set_processing_steps: int = 3
    attention_hidden_dim: int = 64
    
    # Text embedding config
    text_embedding_dim: int = 3072  # text-embedding-3-large
    text_projection_dim: Optional[int] = None  # Project text down before fusion (reduces params!)
    text_proj_init: str = "xavier"  # "xavier" (scaled, recommended) or "kaiming" (PyTorch default)
    text_proj_init_gain: float = 0.1  # Gain for Xavier init (lower = smaller gradients)
    freeze_text_proj: bool = False  # Freeze text projection (implicit regularization for small datasets)
    
    # Fusion config (now modular!)
    fusion: str = "cross_mha"  # "concat", "cross_mha", "gated", "film", "bilinear"
    fusion_dim: int = 256
    fusion_n_heads: int = 8
    fusion_dropout: float = 0.1
    
    # Prediction head config
    head_type: str = "mlp"  # "mlp" or "linear" (linear = no hidden layer, strong regularization)
    head_hidden_dim: int = 128
    head_dropout: float = 0.1
    
    # Molecule processing
    add_hydrogens: bool = False


# =============================================================================
# SEGPredictor
# =============================================================================

class SEGPredictor(BasePredictor):
    """
    SEG (Structured Embedding Fusion) Predictor.
    
    Combines:
    1. ChebNetEncoder for graph embeddings (trainable)
    2. Text embeddings from Azure OpenAI (frozen, pre-computed)
    3. Modular fusion strategy (trainable)
    4. Prediction head (trainable)
    
    Architecture:
        SMILES → ChebNetEncoder → Pooling ─┐
                                           ├─→ Fusion → MLPHead → Prediction
        Text Embedding (pre-computed) ─────┘
    
    Example:
        >>> predictor = SEGPredictor()
        >>> history = predictor.fit(
        ...     train_smiles, train_labels, 
        ...     text_embeddings=train_text_emb
        ... )
        >>> pred = predictor.predict("CCO", text_embedding=text_emb)
    
    Args:
        config: Configuration dataclass (or use defaults)
        text_embedding_fn: Optional function to compute text embeddings
        device: 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        config: Optional[SEGPredictorConfig] = None,
        text_embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or SEGPredictorConfig()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.text_embedding_fn = text_embedding_fn
        
        # Model components (initialized during fit)
        self._encoder: Optional[ChebNetEncoder] = None
        self._pooling: Optional[BasePooling] = None
        self._text_proj: Optional[nn.Module] = None  # Text projection layer
        self._fusion: Optional[BaseFusion] = None
        self._head: Optional[MLPHead] = None
        self._in_channels: Optional[int] = None
        self._is_fitted: bool = False
        
        # Cache for text embeddings
        self._text_cache: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # BasePredictor Interface
    # -------------------------------------------------------------------------
    
    @property
    def is_fitted(self) -> bool:
        """Return True if model has been trained."""
        return self._is_fitted

    def fit(
        self,
        smiles_list: List[str],
        labels: List[float],
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[List[float]] = None,
        *,
        text_embeddings: Optional[np.ndarray] = None,
        val_text_embeddings: Optional[np.ndarray] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        validation_split: float = 0.1,
        verbose: bool = True,
        seed: int = 0,
        scheduler: Optional[str] = None,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        track_gradients: bool = False,
        grad_clip: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Train the SEG model.
        
        Args:
            smiles_list: Training SMILES strings
            labels: Training target values
            val_smiles: Optional validation SMILES
            val_labels: Optional validation labels
            text_embeddings: Pre-computed text embeddings for training data.
                           If None, uses text_embedding_fn.
            val_text_embeddings: Pre-computed text embeddings for validation.
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            validation_split: Fraction for validation if val_smiles not provided
            verbose: Print training progress
            seed: Random seed
            scheduler: Learning rate scheduler type. Options: "plateau", "cosine", None
            scheduler_patience: Patience for ReduceLROnPlateau scheduler
            scheduler_factor: Factor to reduce LR by for plateau scheduler
            min_lr: Minimum learning rate for schedulers
            track_gradients: If True, track gradient statistics per epoch
            grad_clip: Max gradient norm for clipping. None to disable.
            
        Returns:
            Training history dict
        """
        # Validate inputs
        if len(smiles_list) != len(labels):
            raise ValueError("smiles_list and labels must have the same length")
        
        # Get text embeddings
        train_text_emb = self._get_or_compute_text_embeddings(
            smiles_list, text_embeddings, verbose
        )
        
        # Pre-compute graphs
        train_graphs, in_channels = self._precompute_graphs(smiles_list, verbose=verbose)
        self._in_channels = in_channels

        # Build model
        self._build_model()

        # Setup training/validation splits
        y_all = np.asarray(labels, dtype=np.float32)
        rng = np.random.default_rng(seed)
        indices = np.arange(len(smiles_list))
        rng.shuffle(indices)

        if val_smiles is not None:
            train_idx = indices
            val_graphs, _ = self._precompute_graphs(val_smiles, verbose=verbose)
            y_val = np.asarray(val_labels, dtype=np.float32)
            val_text_emb = self._get_or_compute_text_embeddings(
                val_smiles, val_text_embeddings, verbose
            )
            val_idx = None
        else:
            n_val = int(len(smiles_list) * validation_split)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            val_graphs = train_graphs
            y_val = y_all
            val_text_emb = train_text_emb

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        lr_scheduler = None
        if scheduler == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=scheduler_factor,
                patience=scheduler_patience, min_lr=min_lr
            )
        elif scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=min_lr
            )
        
        # Loss function based on task type
        # For multi-label with NaN, we use reduction='none' and mask manually
        is_classification = self.config.task == "classification"
        is_multilabel = self.config.num_tasks > 1
        if is_classification:
            loss_fn = nn.BCELoss(reduction='none') if is_multilabel else nn.BCELoss()
        else:
            loss_fn = nn.MSELoss(reduction='none') if is_multilabel else nn.MSELoss()

        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_metric': [], 'lr': []}
        # Gradient tracking (only populated if track_gradients=True)
        if track_gradients:
            history['grad_norm'] = []
            history['grad_max'] = []
            history['grad_by_layer'] = []
            # Post-clip values (only populated if grad_clip is set)
            if grad_clip:
                history['grad_norm_post'] = []
                history['grad_by_layer_post'] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        if verbose:
            print(f"Training SEGPredictor on {len(train_idx)} molecules...")
            print(f"Task: {self.config.task}")
            n_val_count = len(val_smiles) if val_smiles else len(val_idx)
            print(f"Validation set: {n_val_count} molecules")
            print(f"Fusion method: {self.config.fusion}")
            if lr_scheduler:
                print(f"LR scheduler: {scheduler}")
            if grad_clip:
                print(f"Gradient clipping: {grad_clip}")
            if track_gradients:
                print(f"Gradient tracking: enabled")
            if is_multilabel:
                print(f"Multi-label mode: {self.config.num_tasks} tasks")

        for epoch in range(1, num_epochs + 1):
            # Train epoch
            epoch_result = self._train_epoch(
                train_graphs, train_text_emb, y_all, train_idx, 
                batch_size, optimizer, loss_fn, rng,
                is_multilabel=is_multilabel,
                track_gradients=track_gradients,
                grad_clip=grad_clip,
            )
            
            # Unpack result (loss, or tuple of (loss, grad_stats) if tracking)
            if track_gradients:
                train_loss, grad_stats = epoch_result
                history['grad_norm'].append(grad_stats['grad_norm'])
                history['grad_max'].append(grad_stats['grad_max'])
                history['grad_by_layer'].append(grad_stats['grad_by_layer'])
                # Add post-clip values if available
                if 'grad_norm_post' in grad_stats:
                    history['grad_norm_post'].append(grad_stats['grad_norm_post'])
                    history['grad_by_layer_post'].append(grad_stats['grad_by_layer_post'])
            else:
                train_loss = epoch_result
            history['train_loss'].append(train_loss)

            # Validation
            if val_smiles is not None:
                val_loss = self._evaluate(
                    val_graphs, val_text_emb, y_val, 
                    list(range(len(val_graphs))), batch_size, loss_fn,
                    is_multilabel=is_multilabel
                )
            else:
                val_loss = self._evaluate(
                    train_graphs, train_text_emb, y_all, 
                    val_idx.tolist(), batch_size, loss_fn,
                    is_multilabel=is_multilabel
                )
            
            history['val_loss'].append(val_loss)
            # For classification: val_loss is BCE, for regression: compute RMSE
            val_metric = val_loss if is_classification else np.sqrt(val_loss)
            history['val_metric'].append(val_metric)
            metric_name = "Val BCE" if is_classification else "Val RMSE"

            # Update learning rate scheduler
            if lr_scheduler is not None:
                if scheduler == "plateau":
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            if verbose and (epoch == 1 or epoch % 5 == 0):
                lr_str = f" | LR: {current_lr:.2e}" if lr_scheduler else ""
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | {metric_name}: {val_metric:.4f}{lr_str}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self._get_state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_state is not None:
            self._load_state_dict(best_state)

        self._is_fitted = True
        
        if verbose:
            best_metric = best_val_loss if is_classification else np.sqrt(best_val_loss)
            metric_name = "Val BCE" if is_classification else "Val RMSE"
            print(f"Training complete! Best {metric_name}: {best_metric:.4f}")

        return history

    def predict(
        self,
        smiles: str,
        text_embedding: Optional[np.ndarray] = None
    ) -> Optional[Union[float, np.ndarray]]:
        """
        Predict property for a single SMILES.
        
        Args:
            smiles: SMILES string
            text_embedding: Pre-computed text embedding (1, text_dim) or (text_dim,)
                          If None, uses text_embedding_fn.
            
        Returns:
            For single-task: Predicted value (float), or None if SMILES is invalid
            For multi-task: Predicted values (np.ndarray of shape (num_tasks,)), or None
        """
        self._check_fitted()

        try:
            g = smiles_to_graph(smiles, add_hydrogens=self.config.add_hydrogens)
        except Exception:
            return None

        # Get text embedding
        text_emb = self._get_single_text_embedding(smiles, text_embedding)
        if text_emb is None:
            return None

        bg = batch_graphs([g])
        x, edge_index, edge_weight, batch = self._to_tensors(bg)
        text_t = torch.from_numpy(text_emb.astype(np.float32)).to(self.device)
        if text_t.dim() == 1:
            text_t = text_t.unsqueeze(0)

        self._set_eval_mode()
        with torch.no_grad():
            pred = self._forward(x, edge_index, edge_weight, batch, text_t)
            pred = pred.squeeze(0).cpu().numpy()  # Shape: (num_tasks,) or scalar
            if self.config.num_tasks == 1:
                return float(pred.item() if pred.ndim == 0 else pred[0])
            else:
                return pred  # Shape: (num_tasks,)

    def predict_batch(
        self,
        smiles_list: List[str],
        text_embeddings: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict properties for multiple SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            text_embeddings: Pre-computed text embeddings (N, text_dim)
                           If None, uses text_embedding_fn.
            batch_size: Batch size for inference
            
        Returns:
            For single-task: NumPy array of shape (N,) (NaN for invalid SMILES)
            For multi-task: NumPy array of shape (N, num_tasks) (NaN for invalid SMILES)
        """
        self._check_fitted()
        
        n_samples = len(smiles_list)
        n_tasks = self.config.num_tasks
        
        if n_tasks == 1:
            preds = np.full(n_samples, np.nan, dtype=np.float32)
        else:
            preds = np.full((n_samples, n_tasks), np.nan, dtype=np.float32)
        
        # Get text embeddings
        text_emb_all = self._get_or_compute_text_embeddings(smiles_list, text_embeddings)
        
        # Filter valid SMILES
        valid_positions = []
        valid_graphs = []
        valid_text_embs = []
        
        for i, smi in enumerate(smiles_list):
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                valid_graphs.append(g)
                valid_positions.append(i)
                valid_text_embs.append(text_emb_all[i])
            except Exception:
                continue

        if len(valid_graphs) == 0:
            return preds

        valid_text_embs = np.stack(valid_text_embs)

        self._set_eval_mode()
        with torch.no_grad():
            for start in range(0, len(valid_graphs), batch_size):
                chunk_graphs = valid_graphs[start:start + batch_size]
                chunk_positions = valid_positions[start:start + batch_size]
                chunk_text = valid_text_embs[start:start + batch_size]
                
                bg = batch_graphs(chunk_graphs)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                text_t = torch.from_numpy(chunk_text.astype(np.float32)).to(self.device)
                
                out = self._forward(x, edge_index, edge_weight, batch, text_t)
                out = out.cpu().numpy()  # Shape: (batch, num_tasks)
                
                if n_tasks == 1:
                    out = out.squeeze(-1)  # Shape: (batch,)
                    if out.ndim == 0:
                        out = np.array([out.item()])
                    for pos, val in zip(chunk_positions, out.tolist()):
                        preds[pos] = float(val)
                else:
                    for i, pos in enumerate(chunk_positions):
                        preds[pos] = out[i]  # Shape: (num_tasks,)

        return preds

    def save(self, path: str) -> None:
        """Save model to file."""
        self._check_fitted()

        save_dict = {
            "encoder_state": self._encoder.state_dict(),
            "pooling_state": self._pooling.state_dict(),
            "fusion_state": self._fusion.state_dict(),
            "head_state": self._head.state_dict(),
            "in_channels": self._in_channels,
            "config": self.config.to_dict(),
        }
        # Save text_proj if present
        if self._text_proj is not None:
            save_dict["text_proj_state"] = self._text_proj.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """Load model from file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self._in_channels = int(ckpt["in_channels"])
        self.config = SEGPredictorConfig(**ckpt["config"])
        
        self._build_model()
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._pooling.load_state_dict(ckpt["pooling_state"])
        self._fusion.load_state_dict(ckpt["fusion_state"])
        self._head.load_state_dict(ckpt["head_state"])
        # Load text_proj if present in checkpoint and model has text_proj
        if self._text_proj is not None and "text_proj_state" in ckpt:
            self._text_proj.load_state_dict(ckpt["text_proj_state"])
        
        self._is_fitted = True

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        text_embedding_fn: Optional[Callable] = None,
        device: Optional[str] = None
    ) -> "SEGPredictor":
        """Create predictor from checkpoint file."""
        predictor = cls(text_embedding_fn=text_embedding_fn, device=device)
        predictor.load(path)
        return predictor

    def init_from_cheb(
        self,
        cheb_predictor: "ChebPredictor",
        sample_smiles: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize encoder and pooling from a pretrained ChebPredictor.
        
        This allows training SEG's fusion and head on a subset of data
        while leveraging an encoder trained on the full dataset.
        
        Args:
            cheb_predictor: A fitted ChebPredictor to copy weights from
            sample_smiles: Sample SMILES to infer input channels (if not already built)
        """
        if not cheb_predictor.is_fitted:
            raise ValueError("ChebPredictor must be fitted before copying weights")
        
        # Get input channels from ChebPredictor
        self._in_channels = cheb_predictor._in_channels
        
        # Build the full model (encoder, pooling, fusion, head)
        self._build_model()
        
        # Copy encoder weights from ChebPredictor
        self._encoder.load_state_dict(cheb_predictor._encoder.state_dict())
        self._pooling.load_state_dict(cheb_predictor._pooling.state_dict())
        
        print(f"✓ Copied encoder & pooling from ChebPredictor")
        print(f"  Encoder: {self.config.num_layers} layers, hidden={self.config.hidden_channels}")
        print(f"  Pooling: {self.config.pool}")

    def fit_fusion_only(
        self,
        smiles_list: List[str],
        labels: List[float],
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[List[float]] = None,
        *,
        text_embeddings: np.ndarray,
        val_text_embeddings: Optional[np.ndarray] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        verbose: bool = True,
        track_gradients: bool = False,
        grad_clip: Optional[float] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Train only the fusion and head, keeping encoder/pooling frozen.
        
        Use after init_from_cheb() to fine-tune fusion on a text-augmented subset.
        
        Args:
            smiles_list: Training SMILES
            labels: Training labels
            val_smiles: Validation SMILES (optional)
            val_labels: Validation labels (optional)
            text_embeddings: Text embeddings for training set
            val_text_embeddings: Text embeddings for validation (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            verbose: Print progress
            track_gradients: If True, track gradient statistics per epoch
            grad_clip: Max gradient norm for clipping. None to disable.
            seed: Random seed for reproducibility
            
        Returns:
            Training history dict
        """
        if self._encoder is None:
            raise ValueError("Must call init_from_cheb() before fit_fusion_only()")
        
        # Freeze encoder and pooling
        for param in self._encoder.parameters():
            param.requires_grad = False
        for param in self._pooling.parameters():
            param.requires_grad = False
        
        # Only train fusion, head, and text projection (if present)
        trainable_params = list(self._fusion.parameters()) + list(self._head.parameters())
        text_proj_params = 0
        if self._text_proj is not None:
            trainable_params += list(self._text_proj.parameters())
            text_proj_params = sum(p.numel() for p in self._text_proj.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
        # Loss function based on task type
        if self.config.task == "classification":
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.MSELoss()
        
        if verbose:
            print(f"Training fusion+head{'+text_proj' if self._text_proj else ''} ({sum(p.numel() for p in trainable_params):,} parameters)")
            if text_proj_params > 0:
                print(f"  Text projection: 3072 → {self.config.text_projection_dim} ({text_proj_params:,} params)")
            print(f"  Encoder/pooling frozen ({sum(p.numel() for p in self._encoder.parameters()) + sum(p.numel() for p in self._pooling.parameters()):,} parameters)")
        
        # Prepare data
        graphs = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                graphs.append(g)
                valid_idx.append(i)
            except Exception:
                continue
        
        y = np.array([labels[i] for i in valid_idx], dtype=np.float32)
        # Ensure text embeddings are numpy for consistent indexing
        if isinstance(text_embeddings, torch.Tensor):
            text_emb = text_embeddings[valid_idx].cpu().numpy()
        else:
            text_emb = np.asarray(text_embeddings)[valid_idx]
        
        # Validation data
        val_graphs, val_y, val_text = None, None, None
        if val_smiles is not None and val_labels is not None:
            val_graphs = []
            val_valid_idx = []
            for i, smi in enumerate(val_smiles):
                try:
                    g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                    val_graphs.append(g)
                    val_valid_idx.append(i)
                except Exception:
                    continue
            val_y = np.array([val_labels[i] for i in val_valid_idx], dtype=np.float32)
            if val_text_embeddings is not None:
                if isinstance(val_text_embeddings, torch.Tensor):
                    val_text = val_text_embeddings[val_valid_idx].cpu().numpy()
                else:
                    val_text = np.asarray(val_text_embeddings)[val_valid_idx]
        
        # Training loop
        history = {"train_loss": [], "val_loss": []}
        # Gradient tracking (only populated if track_gradients=True)
        if track_gradients:
            history['grad_norm'] = []
            history['grad_max'] = []
            history['grad_by_layer'] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        # Random generator for reproducibility
        rng = np.random.default_rng(seed)
        
        for epoch in range(num_epochs):
            # Training
            self._encoder.eval()  # Keep frozen in eval mode
            self._pooling.eval()
            self._fusion.train()
            self._head.train()
            if self._text_proj is not None:
                self._text_proj.train()
            
            indices = rng.permutation(len(graphs))
            epoch_loss = 0.0
            n_batches = 0
            
            # Gradient tracking accumulators for this epoch
            if track_gradients:
                batch_grad_norms = []
                batch_grad_maxs = []
                batch_grad_by_layer = []
            
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_graph_list = [graphs[i] for i in batch_idx]
                batch_y = y[batch_idx]
                batch_text = text_emb[batch_idx]
                
                bg = batch_graphs(batch_graph_list)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                # Handle both numpy and tensor inputs
                if isinstance(batch_text, torch.Tensor):
                    text_t = batch_text.float().to(self.device)
                else:
                    text_t = torch.from_numpy(batch_text.astype(np.float32)).to(self.device)
                y_t = torch.from_numpy(batch_y).to(self.device)
                
                optimizer.zero_grad()
                with torch.no_grad():
                    node_emb = self._encoder(x, edge_index, edge_weight, batch)
                    graph_emb = self._pooling(node_emb, batch)
                # Apply text projection if configured
                if self._text_proj is not None:
                    text_t = self._text_proj(text_t)
                fused = self._fusion(graph_emb, text_t)
                pred = self._head(fused).squeeze(-1)
                loss = criterion(pred, y_t)
                loss.backward()
                
                # Track gradients BEFORE clipping (for fusion-only, only trainable params)
                if track_gradients:
                    grad_stats = self._compute_gradient_stats()
                    batch_grad_norms.append(grad_stats['total_norm'])
                    batch_grad_maxs.append(grad_stats['max_value'])
                    batch_grad_by_layer.append(grad_stats['by_layer'])
                
                # Apply gradient clipping (only to trainable params)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            history["train_loss"].append(avg_train_loss)
            
            # Aggregate gradient stats for this epoch
            if track_gradients:
                history['grad_norm'].append(float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0)
                history['grad_max'].append(float(np.max(batch_grad_maxs)) if batch_grad_maxs else 0.0)
                history['grad_by_layer'].append(self._aggregate_layer_grads(batch_grad_by_layer))
            
            # Validation
            val_loss = None
            if val_graphs is not None:
                self._fusion.eval()
                self._head.eval()
                with torch.no_grad():
                    val_total_loss = 0.0
                    val_total_count = 0
                    for start in range(0, len(val_graphs), batch_size):
                        chunk = val_graphs[start:start + batch_size]
                        chunk_text = val_text[start:start + batch_size]
                        chunk_y = val_y[start:start + batch_size]
                        bg = batch_graphs(chunk)
                        x, edge_index, edge_weight, batch = self._to_tensors(bg)
                        # Handle both numpy and tensor inputs
                        if isinstance(chunk_text, torch.Tensor):
                            text_t = chunk_text.float().to(self.device)
                        else:
                            text_t = torch.from_numpy(chunk_text.astype(np.float32)).to(self.device)
                        y_t = torch.from_numpy(chunk_y.astype(np.float32)).to(self.device)
                        node_emb = self._encoder(x, edge_index, edge_weight, batch)
                        graph_emb = self._pooling(node_emb, batch)
                        # Apply text projection if configured
                        if self._text_proj is not None:
                            text_t = self._text_proj(text_t)
                        fused = self._fusion(graph_emb, text_t)
                        pred = self._head(fused).squeeze(-1)
                        batch_loss = criterion(pred, y_t)
                        n_batch = int(y_t.size(0))
                        val_total_loss += float(batch_loss.item()) * n_batch
                        val_total_count += n_batch

                    val_loss = float(val_total_loss / max(val_total_count, 1))
                    history["val_loss"].append(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {
                            'fusion': self._fusion.state_dict(),
                            'head': self._head.state_dict(),
                        }
                        if self._text_proj is not None:
                            best_state['text_proj'] = self._text_proj.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        if best_state is not None:
            self._fusion.load_state_dict(best_state['fusion'])
            self._head.load_state_dict(best_state['head'])
            if self._text_proj is not None and 'text_proj' in best_state:
                self._text_proj.load_state_dict(best_state['text_proj'])
        
        # Unfreeze for future training if needed
        for param in self._encoder.parameters():
            param.requires_grad = True
        for param in self._pooling.parameters():
            param.requires_grad = True
        
        self._is_fitted = True
        return history

    # -------------------------------------------------------------------------
    # Additional Public Methods
    # -------------------------------------------------------------------------

    def encode(
        self,
        smiles_list: List[str],
        text_embeddings: Optional[np.ndarray] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get fused embeddings without prediction head.
        
        Returns:
            Embeddings array (N, fusion_dim) with NaN rows for invalid SMILES
        """
        self._check_fitted()
        
        embeddings = np.full(
            (len(smiles_list), self.config.fusion_dim), 
            np.nan, dtype=np.float32
        )
        
        text_emb_all = self._get_or_compute_text_embeddings(smiles_list, text_embeddings)
        
        valid_positions = []
        valid_graphs = []
        valid_text_embs = []
        
        for i, smi in enumerate(smiles_list):
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                valid_graphs.append(g)
                valid_positions.append(i)
                valid_text_embs.append(text_emb_all[i])
            except Exception:
                continue

        if len(valid_graphs) == 0:
            return embeddings

        valid_text_embs = np.stack(valid_text_embs)

        self._set_eval_mode()
        with torch.no_grad():
            for start in range(0, len(valid_graphs), batch_size):
                chunk_graphs = valid_graphs[start:start + batch_size]
                chunk_positions = valid_positions[start:start + batch_size]
                chunk_text = valid_text_embs[start:start + batch_size]
                
                bg = batch_graphs(chunk_graphs)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                text_t = torch.from_numpy(chunk_text.astype(np.float32)).to(self.device)
                
                # Encode only (no head)
                node_emb = self._encoder(x, edge_index, edge_weight, batch)
                graph_emb = self._pooling(node_emb, batch)
                fused = self._fusion(graph_emb, text_t)
                
                emb_np = fused.cpu().numpy()
                for i, pos in enumerate(chunk_positions):
                    embeddings[pos] = emb_np[i]

        return embeddings

    def set_text_embedding_fn(self, fn: Callable[[List[str]], np.ndarray]) -> None:
        """Set the function used to compute text embeddings."""
        self.text_embedding_fn = fn

    def clear_text_cache(self) -> None:
        """Clear the text embedding cache."""
        self._text_cache.clear()

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _build_model(self) -> None:
        """Build all model components."""
        cfg = self.config
        
        # Encoder
        self._encoder = ChebNetEncoder(
            in_channels=self._in_channels,
            hidden_channels=cfg.hidden_channels,
            K=cfg.K,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            lambda_max=cfg.lambda_max,
        ).to(self.device)
        
        # Pooling
        pooling_kwargs = {}
        if cfg.pool == "set2set":
            pooling_kwargs["n_iters"] = cfg.set2set_processing_steps
            pooling_kwargs["dropout"] = cfg.dropout  # Regularization on LSTM output
        elif cfg.pool == "attention":
            pooling_kwargs["hidden_dim"] = cfg.attention_hidden_dim
            pooling_kwargs["dropout"] = cfg.dropout
            
        self._pooling = create_pooling(
            cfg.pool, 
            input_dim=cfg.hidden_channels,
            **pooling_kwargs
        ).to(self.device)
        
        # Text projection (optional, but recommended for reducing parameters)
        if cfg.text_projection_dim is not None and cfg.text_projection_dim < cfg.text_embedding_dim:
            self._text_proj = nn.Sequential(
                nn.Linear(cfg.text_embedding_dim, cfg.text_projection_dim),
                nn.LayerNorm(cfg.text_projection_dim),
                nn.GELU(),
                nn.Dropout(cfg.fusion_dropout),
            ).to(self.device)
            # Apply configured initialization to reduce gradient imbalance
            # See docs/weight_initialization.tex for theoretical justification
            if cfg.text_proj_init == "xavier":
                nn.init.xavier_uniform_(self._text_proj[0].weight, gain=cfg.text_proj_init_gain)
            # else: keep PyTorch default Kaiming initialization
            text_dim_for_fusion = cfg.text_projection_dim
        else:
            self._text_proj = None
            text_dim_for_fusion = cfg.text_embedding_dim
        
        # Fusion
        graph_dim = self._pooling.output_dim
        fusion_kwargs = {
            "output_dim": cfg.fusion_dim,
            "dropout": cfg.fusion_dropout,
        }
        if cfg.fusion == "cross_mha":
            fusion_kwargs["n_heads"] = cfg.fusion_n_heads
            
        self._fusion = create_fusion(
            cfg.fusion,
            dim_a=graph_dim,
            dim_b=text_dim_for_fusion,
            **fusion_kwargs
        ).to(self.device)
        
        # Head (supports multi-label via num_tasks)
        head_input_dim = self._fusion.output_dim
        output_dim = cfg.num_tasks  # 1 for single-task, >1 for multi-label
        if cfg.head_type == "linear":
            self._head = LinearHead(
                input_dim=head_input_dim,
                output_dim=output_dim,
                task=cfg.task,
            ).to(self.device)
        else:  # "mlp" (default)
            self._head = MLPHead(
                input_dim=head_input_dim,
                hidden_dim=cfg.head_hidden_dim,
                output_dim=output_dim,
                dropout=cfg.head_dropout,
                task=cfg.task,
            ).to(self.device)

    def _forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through all components."""
        node_emb = self._encoder(x, edge_index, edge_weight, batch)
        graph_emb = self._pooling(node_emb, batch)
        
        # Apply text projection if configured
        if self._text_proj is not None:
            text_emb = self._text_proj(text_emb)
        
        fused = self._fusion(graph_emb, text_emb)
        return self._head(fused)

    def _get_or_compute_text_embeddings(
        self,
        smiles_list: List[str],
        provided_embeddings: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> np.ndarray:
        """Get text embeddings, using cache or computing if needed."""
        if provided_embeddings is not None:
            if len(provided_embeddings) != len(smiles_list):
                raise ValueError("text_embeddings length must match smiles_list")
            return provided_embeddings
        
        if self.text_embedding_fn is None:
            raise ValueError(
                "No text embeddings provided and no text_embedding_fn set. "
                "Either pass text_embeddings or set text_embedding_fn."
            )
        
        # Check cache
        uncached = [(i, s) for i, s in enumerate(smiles_list) if s not in self._text_cache]
        
        if uncached and verbose:
            print(f"Computing text embeddings for {len(uncached)} new SMILES...")
        
        if uncached:
            uncached_smiles = [s for _, s in uncached]
            new_embeddings = self.text_embedding_fn(uncached_smiles)
            
            for (idx, smi), emb in zip(uncached, new_embeddings):
                self._text_cache[smi] = emb
        
        return np.stack([self._text_cache[s] for s in smiles_list])

    def _get_single_text_embedding(
        self,
        smiles: str,
        provided_embedding: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Get text embedding for a single SMILES."""
        if provided_embedding is not None:
            return provided_embedding
        
        if smiles in self._text_cache:
            return self._text_cache[smiles]
        
        if self.text_embedding_fn is not None:
            emb = self.text_embedding_fn([smiles])
            self._text_cache[smiles] = emb[0]
            return emb[0]
        
        return None

    def _precompute_graphs(
        self,
        smiles_list: List[str],
        verbose: bool = True
    ) -> Tuple[List[Optional[Any]], int]:
        """Pre-compute molecular graphs."""
        if verbose:
            print(f"Pre-computing molecular graphs for {len(smiles_list)} molecules...")

        graphs = []
        in_channels = None

        for smi in smiles_list:
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                graphs.append(g)
                if in_channels is None:
                    in_channels = int(g.X.shape[1])
            except Exception:
                graphs.append(None)

        valid_count = sum(1 for g in graphs if g is not None)
        if verbose:
            print(f"Pre-computed {valid_count}/{len(smiles_list)} valid graphs")

        if in_channels is None:
            raise ValueError("No valid molecules found")

        return graphs, in_channels

    def _train_epoch(
        self,
        graphs: List[Optional[Any]],
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        rng: np.random.Generator,
        is_multilabel: bool = False,
        track_gradients: bool = False,
        grad_clip: Optional[float] = None,
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """Run one training epoch.
        
        For multi-label tasks with NaN labels, loss_fn should use reduction='none'
        and we mask out NaN entries before computing the mean.
        
        Args:
            graphs: Pre-computed molecular graphs
            text_embeddings: Text embeddings array
            labels: Target labels
            train_idx: Training indices
            batch_size: Batch size
            optimizer: Optimizer instance
            loss_fn: Loss function
            rng: Random generator for shuffling
            is_multilabel: Whether task is multi-label
            track_gradients: If True, collect gradient statistics
            grad_clip: Max gradient norm for clipping. None to disable.
            
        Returns:
            If track_gradients is False: training loss (float)
            If track_gradients is True: tuple of (training loss, gradient stats dict)
        """
        self._set_train_mode()
        rng.shuffle(train_idx)
        
        total_loss = 0.0
        total_valid = 0
        
        # Gradient tracking accumulators
        if track_gradients:
            batch_grad_norms = []
            batch_grad_maxs = []
            batch_grad_by_layer = []
            # Post-clip accumulators (only filled if grad_clip is set)
            batch_grad_norms_post = []
            batch_grad_by_layer_post = []

        for start in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[start:start + batch_size].tolist()
            pack = self._make_batch(graphs, text_embeddings, labels, batch_idx)
            if pack is None:
                continue

            x, edge_index, edge_weight, batch, text_emb, y = pack
            x, edge_index, edge_weight, batch = self._numpy_to_tensors(
                x, edge_index, edge_weight, batch
            )
            text_t = torch.from_numpy(text_emb.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(y).to(self.device)

            optimizer.zero_grad(set_to_none=True)
            pred = self._forward(x, edge_index, edge_weight, batch, text_t)
            
            if is_multilabel:
                # Mask NaN values for multi-label
                valid_mask = ~torch.isnan(y_t)
                if valid_mask.sum() == 0:
                    continue
                loss_per_elem = loss_fn(pred, torch.where(valid_mask, y_t, pred.detach()))
                loss = (loss_per_elem * valid_mask.float()).sum() / valid_mask.sum()
                n_valid = valid_mask.sum().item()
            else:
                loss = loss_fn(pred, y_t)
                n_valid = int(y_t.size(0))
            
            loss.backward()
            
            # Track gradients BEFORE clipping (to see raw gradient magnitudes)
            if track_gradients:
                grad_stats = self._compute_gradient_stats()
                batch_grad_norms.append(grad_stats['total_norm'])
                batch_grad_maxs.append(grad_stats['max_value'])
                batch_grad_by_layer.append(grad_stats['by_layer'])
            
            # Apply gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self._get_parameters(), max_norm=grad_clip)
                # Track gradients AFTER clipping to verify it's working
                if track_gradients:
                    grad_stats_post = self._compute_gradient_stats()
                    batch_grad_norms_post.append(grad_stats_post['total_norm'])
                    batch_grad_by_layer_post.append(grad_stats_post['by_layer'])
            
            optimizer.step()

            total_loss += float(loss.item()) * n_valid
            total_valid += n_valid

        avg_loss = total_loss / max(total_valid, 1)
        
        if track_gradients:
            # Aggregate gradient stats across batches (mean)
            epoch_grad_stats = {
                'grad_norm': float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0,
                'grad_max': float(np.max(batch_grad_maxs)) if batch_grad_maxs else 0.0,
                'grad_by_layer': self._aggregate_layer_grads(batch_grad_by_layer),
            }
            # Add post-clip stats if clipping was used
            if grad_clip is not None and batch_grad_norms_post:
                epoch_grad_stats['grad_norm_post'] = float(np.mean(batch_grad_norms_post))
                epoch_grad_stats['grad_by_layer_post'] = self._aggregate_layer_grads(batch_grad_by_layer_post)
            return avg_loss, epoch_grad_stats
        
        return avg_loss

    def _evaluate(
        self,
        graphs: List[Optional[Any]],
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        indices: List[int],
        batch_size: int,
        loss_fn: nn.Module,
        is_multilabel: bool = False
    ) -> float:
        """Evaluate on a set of indices.
        
        For multi-label tasks with NaN labels, loss_fn should use reduction='none'
        and we mask out NaN entries before computing the mean.
        """
        self._set_eval_mode()
        total_loss = 0.0
        total_valid = 0

        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                pack = self._make_batch(graphs, text_embeddings, labels, batch_idx)
                if pack is None:
                    continue

                x, edge_index, edge_weight, batch, text_emb, y = pack
                x, edge_index, edge_weight, batch = self._numpy_to_tensors(
                    x, edge_index, edge_weight, batch
                )
                text_t = torch.from_numpy(text_emb.astype(np.float32)).to(self.device)
                y_t = torch.from_numpy(y).to(self.device)

                pred = self._forward(x, edge_index, edge_weight, batch, text_t)
                
                if is_multilabel:
                    # Mask NaN values for multi-label
                    valid_mask = ~torch.isnan(y_t)
                    if valid_mask.sum() == 0:
                        continue
                    loss_per_elem = loss_fn(pred, torch.where(valid_mask, y_t, pred))
                    loss = (loss_per_elem * valid_mask.float()).sum()
                    n_valid = valid_mask.sum().item()
                else:
                    loss = loss_fn(pred, y_t) * int(y_t.size(0))
                    n_valid = int(y_t.size(0))
                
                total_loss += float(loss.item())
                total_valid += n_valid

        return float(total_loss / max(total_valid, 1)) if total_valid > 0 else float('inf')

    @staticmethod
    def _make_batch(
        graphs: List[Optional[Any]],
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        indices: List[int]
    ) -> Optional[Tuple]:
        """Create a batched graph with text embeddings from pre-computed data.
        
        Supports both single-task (labels shape: (N,) or (N, 1)) and 
        multi-task (labels shape: (N, num_tasks)) labels.
        Labels can contain NaN for missing values in multi-task settings.
        """
        valid_graphs = []
        valid_text_embs = []
        ys = []

        # Check if labels are multi-dimensional
        labels = np.asarray(labels, dtype=np.float32)
        is_multitask = labels.ndim == 2 and labels.shape[1] > 1

        for k in indices:
            g = graphs[k]
            if g is None:
                continue
            valid_graphs.append(g)
            valid_text_embs.append(text_embeddings[k])
            if is_multitask:
                ys.append(labels[k])  # shape: (num_tasks,)
            else:
                # Handle both 1D and 2D single-task labels
                y_val = labels[k] if labels.ndim == 1 else labels[k, 0]
                ys.append(float(y_val))

        if len(valid_graphs) == 0:
            return None

        bg = batch_graphs(valid_graphs)
        if is_multitask:
            y = np.stack(ys)  # shape: (batch_size, num_tasks)
        else:
            y = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
        text_emb = np.stack(valid_text_embs)

        return bg.X, bg.edge_index, bg.edge_weight, bg.batch, text_emb, y

    def _to_tensors(self, bg) -> Tuple[torch.Tensor, ...]:
        """Convert BatchedMolGraph to tensors."""
        return self._numpy_to_tensors(bg.X, bg.edge_index, bg.edge_weight, bg.batch)

    def _numpy_to_tensors(
        self,
        X: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
        batch: np.ndarray
    ) -> Tuple[torch.Tensor, ...]:
        """Convert numpy arrays to device tensors."""
        x = torch.from_numpy(X.astype(np.float32, copy=False)).to(self.device)
        ei = torch.from_numpy(edge_index.astype(np.int64, copy=False)).to(self.device)
        ew = torch.from_numpy(edge_weight.astype(np.float32, copy=False)).to(self.device)
        b = torch.from_numpy(batch.astype(np.int64, copy=False)).to(self.device)
        return x, ei, ew, b

    def _get_parameters(self):
        """Get all trainable parameters."""
        params = list(self._encoder.parameters())
        if self._text_proj is not None and not self.config.freeze_text_proj:
            params += list(self._text_proj.parameters())
        params += list(self._pooling.parameters())
        params += list(self._fusion.parameters())
        params += list(self._head.parameters())
        return params

    def _set_train_mode(self) -> None:
        """Set all components to training mode."""
        self._encoder.train()
        if self._text_proj is not None:
            # Keep in eval mode if frozen (disables dropout)
            self._text_proj.train(not self.config.freeze_text_proj)
        self._pooling.train()
        self._fusion.train()
        self._head.train()

    def _set_eval_mode(self) -> None:
        """Set all components to evaluation mode."""
        self._encoder.eval()
        if self._text_proj is not None:
            self._text_proj.eval()
        self._pooling.eval()
        self._fusion.eval()
        self._head.eval()

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for all components."""
        state = {
            'encoder': {k: v.cpu().clone() for k, v in self._encoder.state_dict().items()},
            'pooling': {k: v.cpu().clone() for k, v in self._pooling.state_dict().items()},
            'fusion': {k: v.cpu().clone() for k, v in self._fusion.state_dict().items()},
            'head': {k: v.cpu().clone() for k, v in self._head.state_dict().items()},
        }
        if self._text_proj is not None:
            state['text_proj'] = {k: v.cpu().clone() for k, v in self._text_proj.state_dict().items()}
        return state

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict for all components."""
        self._encoder.load_state_dict(state['encoder'])
        if self._text_proj is not None and 'text_proj' in state:
            self._text_proj.load_state_dict(state['text_proj'])
        self._pooling.load_state_dict(state['pooling'])
        self._fusion.load_state_dict(state['fusion'])
        self._head.load_state_dict(state['head'])

    def _compute_gradient_stats(self) -> Dict[str, Any]:
        """Compute gradient statistics for all trainable parameters.
        
        Returns dict with:
            - total_norm: L2 norm of all gradients
            - max_value: Maximum absolute gradient value
            - by_layer: Dict mapping component name to its gradient L2 norm
        """
        total_norm_sq = 0.0
        max_value = 0.0
        by_layer = {}
        
        # Define component groups
        components = [
            ('encoder', self._encoder),
            ('pooling', self._pooling),
            ('fusion', self._fusion),
            ('head', self._head),
        ]
        if self._text_proj is not None:
            components.insert(1, ('text_proj', self._text_proj))
        
        for name, module in components:
            layer_norm_sq = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    grad_data = p.grad.data
                    param_norm_sq = float(grad_data.norm(2).item() ** 2)
                    param_max = float(grad_data.abs().max().item())
                    
                    layer_norm_sq += param_norm_sq
                    total_norm_sq += param_norm_sq
                    max_value = max(max_value, param_max)
            
            by_layer[name] = float(np.sqrt(layer_norm_sq))
        
        return {
            'total_norm': float(np.sqrt(total_norm_sq)),
            'max_value': max_value,
            'by_layer': by_layer,
        }

    def _aggregate_layer_grads(self, batch_grad_by_layer: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate layer-wise gradient norms across batches (mean).
        
        Args:
            batch_grad_by_layer: List of dicts, one per batch, mapping layer name to norm
            
        Returns:
            Dict mapping layer name to mean gradient norm across batches
        """
        if not batch_grad_by_layer:
            return {}
        
        # Get all layer names from first batch
        layer_names = list(batch_grad_by_layer[0].keys())
        
        aggregated = {}
        for name in layer_names:
            values = [batch[name] for batch in batch_grad_by_layer if name in batch]
            aggregated[name] = float(np.mean(values)) if values else 0.0
        
        return aggregated

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first or load a checkpoint.")
