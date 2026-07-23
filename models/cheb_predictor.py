"""Chebyshev-GNN based molecular property predictor.

Provides sklearn-style API for graph-based molecular property prediction
using Chebyshev spectral graph convolutions.

This module uses modular components from models.core for flexibility:
- Graph encoding via ChebNetEncoder
- Configurable pooling (sum, mean, set2set, attention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.core.base import BasePredictor, BasePredictorConfig, MLPHead
from models.core.pooling import create_pooling, BasePooling
from models.core.graph_encoder import ChebNetEncoder
from utils.molecular_graph import smiles_to_graph
from utils.batched_mol_graph import batch_graphs


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChebPredictorConfig(BasePredictorConfig):
    """Configuration for ChebPredictor."""
    # Task type (inherited from base, but explicit for clarity)
    task: str = "regression"  # "regression" or "classification"
    
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
    
    # Molecule processing
    add_hydrogens: bool = False  # Set True for quantum property prediction
    
    # Prediction head
    head_hidden_dim: int = 64
    head_dropout: float = 0.1


# =============================================================================
# ChebPredictor
# =============================================================================

class ChebPredictor(BasePredictor):
    """
    ChebNet-based molecular property predictor.
    
    Uses Chebyshev spectral graph convolutions to learn molecular representations
    from graph structure. Provides sklearn-style API for easy use.
    
    Architecture:
        SMILES -> MolGraph -> ChebNetEncoder -> Pooling -> MLPHead -> Prediction
    
    Example:
        >>> predictor = ChebPredictor()
        >>> history = predictor.fit(train_smiles, train_labels)
        >>> prediction = predictor.predict("CCO")
        >>> predictions = predictor.predict_batch(test_smiles)
    
    Args:
        config: Configuration dataclass (or use defaults)
        device: 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        config: Optional[ChebPredictorConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or ChebPredictorConfig()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model components (initialized during fit)
        self._encoder: Optional[ChebNetEncoder] = None
        self._pooling: Optional[BasePooling] = None
        self._head: Optional[MLPHead] = None
        self._in_channels: Optional[int] = None
        self._is_fitted: bool = False

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
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        validation_split: float = 0.1,
        verbose: bool = True,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the ChebNet model.
        
        Args:
            smiles_list: Training SMILES strings
            labels: Training target values
            val_smiles: Optional validation SMILES (if None, uses validation_split)
            val_labels: Optional validation labels
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            validation_split: Fraction for validation if val_smiles not provided
            verbose: Print training progress
            seed: Random seed for reproducibility
            
        Returns:
            Training history dict with 'train_loss', 'val_loss' per epoch
        """
        # Validate inputs
        if len(smiles_list) != len(labels):
            raise ValueError("smiles_list and labels must have the same length")
        if (val_smiles is None) != (val_labels is None):
            raise ValueError("val_smiles and val_labels must be both provided or both None")
        if val_smiles is not None and len(val_smiles) != len(val_labels):
            raise ValueError("val_smiles and val_labels must have the same length")

        # Pre-compute graphs (major speedup)
        train_graphs, in_channels = self._precompute_graphs(
            smiles_list, verbose=verbose
        )
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
            val_idx = None
        else:
            n_val = int(len(smiles_list) * validation_split)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            val_graphs = train_graphs
            y_val = y_all

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self._get_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function based on task type
        if self.config.task == "classification":
            loss_fn = nn.BCELoss()
        else:
            loss_fn = nn.MSELoss()

        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        if verbose:
            print(f"Training ChebPredictor on {len(train_idx)} molecules...")
            print(f"Task: {self.config.task}")
            n_val_count = len(val_smiles) if val_smiles is not None else len(val_idx)
            print(f"Validation set: {n_val_count} molecules")

        for epoch in range(1, num_epochs + 1):
            # Train epoch
            train_loss = self._train_epoch(
                train_graphs, y_all, train_idx, batch_size, optimizer, loss_fn, rng
            )
            history['train_loss'].append(train_loss)

            # Validation
            if val_smiles is not None:
                val_loss = self._evaluate(val_graphs, y_val, list(range(len(val_graphs))), batch_size, loss_fn)
            else:
                val_loss = self._evaluate(train_graphs, y_all, val_idx.tolist(), batch_size, loss_fn)
            
            history['val_loss'].append(val_loss)

            if verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
            print(f"Training complete! Best Val Loss: {best_val_loss:.4f}")

        return history

    def predict(self, smiles: str) -> Optional[float]:
        """
        Predict property for a single SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Predicted value, or None if SMILES is invalid
        """
        self._check_fitted()

        try:
            g = smiles_to_graph(smiles, add_hydrogens=self.config.add_hydrogens)
        except Exception:
            return None

        bg = batch_graphs([g])
        x, edge_index, edge_weight, batch = self._to_tensors(bg)

        self._set_eval_mode()
        with torch.no_grad():
            pred = self._forward(x, edge_index, edge_weight, batch)
            return float(pred.squeeze().cpu().item())

    def predict_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict properties for multiple SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference
            
        Returns:
            NumPy array of predictions (NaN for invalid SMILES)
        """
        self._check_fitted()

        preds = np.full(len(smiles_list), np.nan, dtype=np.float32)
        
        # Filter valid SMILES
        valid_positions = []
        valid_graphs = []
        for i, smi in enumerate(smiles_list):
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                valid_graphs.append(g)
                valid_positions.append(i)
            except Exception:
                continue

        if len(valid_graphs) == 0:
            return preds

        self._set_eval_mode()
        with torch.no_grad():
            for start in range(0, len(valid_graphs), batch_size):
                chunk_graphs = valid_graphs[start:start + batch_size]
                chunk_positions = valid_positions[start:start + batch_size]
                
                bg = batch_graphs(chunk_graphs)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                
                out = self._forward(x, edge_index, edge_weight, batch)
                out = out.squeeze().cpu().numpy()
                
                if out.shape == ():
                    out = np.array([out])
                
                for pos, val in zip(chunk_positions, out.tolist()):
                    preds[pos] = float(val)

        return preds

    def save(self, path: str) -> None:
        """Save model to file."""
        self._check_fitted()

        torch.save({
            "encoder_state": self._encoder.state_dict(),
            "pooling_state": self._pooling.state_dict(),
            "head_state": self._head.state_dict(),
            "in_channels": self._in_channels,
            "config": self.config.to_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model from file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self._in_channels = int(ckpt["in_channels"])
        self.config = ChebPredictorConfig(**ckpt["config"])
        
        self._build_model()
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._pooling.load_state_dict(ckpt["pooling_state"])
        self._head.load_state_dict(ckpt["head_state"])
        
        self._is_fitted = True

    # -------------------------------------------------------------------------
    # Additional Public Methods
    # -------------------------------------------------------------------------

    def encode(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get graph embeddings without prediction head.
        
        Useful for downstream tasks or analysis.
        
        Returns:
            Embeddings array (N, hidden_channels) with NaN rows for invalid SMILES
        """
        self._check_fitted()
        
        embeddings = np.full(
            (len(smiles_list), self.config.hidden_channels), 
            np.nan, dtype=np.float32
        )
        
        valid_positions = []
        valid_graphs = []
        for i, smi in enumerate(smiles_list):
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                valid_graphs.append(g)
                valid_positions.append(i)
            except Exception:
                continue

        if len(valid_graphs) == 0:
            return embeddings

        self._set_eval_mode()
        with torch.no_grad():
            for start in range(0, len(valid_graphs), batch_size):
                chunk_graphs = valid_graphs[start:start + batch_size]
                chunk_positions = valid_positions[start:start + batch_size]
                
                bg = batch_graphs(chunk_graphs)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                
                # Encode only (no head)
                node_emb = self._encoder(x, edge_index, edge_weight, batch)
                graph_emb = self._pooling(node_emb, batch)
                
                emb_np = graph_emb.cpu().numpy()
                for i, pos in enumerate(chunk_positions):
                    embeddings[pos] = emb_np[i]

        return embeddings

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _build_model(self) -> None:
        """Build model components."""
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
        elif cfg.pool == "attention":
            pooling_kwargs["hidden_dim"] = cfg.attention_hidden_dim
            pooling_kwargs["dropout"] = cfg.dropout
            
        self._pooling = create_pooling(
            cfg.pool, 
            input_dim=cfg.hidden_channels,
            **pooling_kwargs
        ).to(self.device)
        
        # Head
        head_input_dim = self._pooling.output_dim
        self._head = MLPHead(
            input_dim=head_input_dim,
            hidden_dim=cfg.head_hidden_dim,
            output_dim=1,
            dropout=cfg.head_dropout,
            task=cfg.task,
        ).to(self.device)

    def _forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through all components."""
        node_emb = self._encoder(x, edge_index, edge_weight, batch)
        graph_emb = self._pooling(node_emb, batch)
        return self._head(graph_emb)

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
        labels: np.ndarray,
        train_idx: np.ndarray,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        rng: np.random.Generator
    ) -> float:
        """Run one training epoch."""
        self._set_train_mode()
        rng.shuffle(train_idx)
        
        total_loss = 0.0
        seen = 0

        for start in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[start:start + batch_size].tolist()
            pack = self._make_batch(graphs, labels, batch_idx)
            if pack is None:
                continue

            x, edge_index, edge_weight, batch, y = pack
            x, edge_index, edge_weight, batch = self._numpy_to_tensors(
                x, edge_index, edge_weight, batch
            )
            y_t = torch.from_numpy(y).to(self.device)

            optimizer.zero_grad(set_to_none=True)
            pred = self._forward(x, edge_index, edge_weight, batch)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(y_t.size(0))
            seen += int(y_t.size(0))

        return total_loss / max(seen, 1)

    def _evaluate(
        self,
        graphs: List[Optional[Any]],
        labels: np.ndarray,
        indices: List[int],
        batch_size: int,
        loss_fn: nn.Module
    ) -> float:
        """Evaluate on a set of indices."""
        self._set_eval_mode()
        total_loss = 0.0
        seen = 0

        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                pack = self._make_batch(graphs, labels, batch_idx)
                if pack is None:
                    continue

                x, edge_index, edge_weight, batch, y = pack
                x, edge_index, edge_weight, batch = self._numpy_to_tensors(
                    x, edge_index, edge_weight, batch
                )
                y_t = torch.from_numpy(y).to(self.device)

                pred = self._forward(x, edge_index, edge_weight, batch)
                n_batch = int(y_t.size(0))
                total_loss += float(loss_fn(pred, y_t).item()) * n_batch
                seen += n_batch

            return float(total_loss / max(seen, 1)) if seen > 0 else float('inf')

    @staticmethod
    def _make_batch(
        graphs: List[Optional[Any]],
        labels: np.ndarray,
        indices: List[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Create a batched graph from pre-computed graphs."""
        valid_graphs = []
        ys = []

        for k in indices:
            g = graphs[k]
            if g is None:
                continue
            valid_graphs.append(g)
            ys.append(float(labels[k]))

        if len(valid_graphs) == 0:
            return None

        bg = batch_graphs(valid_graphs)
        y = np.asarray(ys, dtype=np.float32).reshape(-1, 1)

        return bg.X, bg.edge_index, bg.edge_weight, bg.batch, y

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
        params += list(self._pooling.parameters())
        params += list(self._head.parameters())
        return params

    def _set_train_mode(self) -> None:
        """Set all components to training mode."""
        self._encoder.train()
        self._pooling.train()
        self._head.train()

    def _set_eval_mode(self) -> None:
        """Set all components to evaluation mode."""
        self._encoder.eval()
        self._pooling.eval()
        self._head.eval()

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for all components."""
        return {
            'encoder': {k: v.cpu().clone() for k, v in self._encoder.state_dict().items()},
            'pooling': {k: v.cpu().clone() for k, v in self._pooling.state_dict().items()},
            'head': {k: v.cpu().clone() for k, v in self._head.state_dict().items()},
        }

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict for all components."""
        self._encoder.load_state_dict(state['encoder'])
        self._pooling.load_state_dict(state['pooling'])
        self._head.load_state_dict(state['head'])

    # -------------------------------------------------------------------------
    # Backward Compatibility
    # -------------------------------------------------------------------------

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for fit() - backward compatibility."""
        return self.fit(*args, **kwargs)
