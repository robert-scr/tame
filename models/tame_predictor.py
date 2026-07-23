"""TAME predictor: Graph + Text + RDKit descriptor fusion with element-wise MoE gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors

from models.core.base import BasePredictor, BasePredictorConfig
from models.core.graph_encoder import ChebNetEncoder
from models.core.pooling import BasePooling, create_pooling
from utils.batched_mol_graph import batch_graphs
from utils.molecular_graph import smiles_to_graph


class ModalityProjectionMLP(nn.Module):
    """Project a modality embedding to a shared hidden dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DescriptorProjectionMLP(nn.Module):
    """Project descriptor features with BatchNorm1d at the input."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriModalElementWiseMoE(nn.Module):
    """
    Tri-modal fusion module with element-wise mixture-of-experts routing.

    Inputs:
    - graph_emb: (B, graph_dim)
    - text_emb: (B, text_dim)
    - desc_features: (B, desc_dim)

    Steps:
    1) Project each modality to hidden_dim
    2) Route with element-wise softmax gates over 3 experts
    3) Fuse as weighted element-wise sum
    4) Predict scalar output
    """

    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        desc_dim: int,
        num_tasks: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        router_dropout: float = 0.1,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.graph_proj = ModalityProjectionMLP(graph_dim, hidden_dim, dropout)
        self.text_proj = ModalityProjectionMLP(text_dim, hidden_dim, dropout)
        self.desc_proj = DescriptorProjectionMLP(desc_dim, hidden_dim, dropout)

        router_in_dim = hidden_dim * 3
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_in_dim),
            nn.GELU(),
            nn.Dropout(router_dropout),
            nn.Linear(router_in_dim, router_in_dim),
        )

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, int(num_tasks)),
        )

    def forward(
        self,
        graph_emb: torch.Tensor,
        text_emb: torch.Tensor,
        desc_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        e_g = self.graph_proj(graph_emb)
        e_t = self.text_proj(text_emb)
        e_d = self.desc_proj(desc_features)

        concat = torch.cat([e_g, e_t, e_d], dim=-1)
        router_out = self.router(concat)

        bsz = router_out.shape[0]
        gates = router_out.view(bsz, self.hidden_dim, 3)
        gates = torch.softmax(gates, dim=-1)

        g_g = gates[:, :, 0]
        g_t = gates[:, :, 1]
        g_d = gates[:, :, 2]

        fused_emb = (e_g * g_g) + (e_t * g_t) + (e_d * g_d)
        pred = self.pred_head(fused_emb)

        aux = {
            "graph_gate": g_g,
            "text_gate": g_t,
            "desc_gate": g_d,
            "gates_full": gates,  # (B, H, 3) — used for entropy regularisation
            "fused_emb": fused_emb,
        }
        return pred, aux

    def _init_router_at_target(self, gate_target: Tuple[float, float, float]) -> None:
        import math
        final = self.router[-1]
        nn.init.normal_(final.weight, std=0.01)
        H = self.hidden_dim
        eps = 1e-6
        p = [max(float(gate_target[i]), eps) for i in range(3)]
        s = sum(p)
        p = [x / s for x in p]
        bias = torch.zeros(H * 3)
        bias[0::3] = math.log(p[0])
        bias[1::3] = math.log(p[1])
        bias[2::3] = math.log(p[2])
        final.bias.data.copy_(bias)


@dataclass
class TAMEPredictorConfig(BasePredictorConfig):
    """Configuration for TAMEPredictor."""

    task: str = "regression"
    num_tasks: int = 1

    # Graph encoder
    hidden_channels: int = 64
    K: int = 3
    num_layers: int = 2
    dropout: float = 0.1
    lambda_max: float = 2.0

    # Pooling
    pool: str = "mean"
    set2set_processing_steps: int = 3
    attention_hidden_dim: int = 64

    # Descriptors and text
    text_embedding_dim: int = 3072
    descriptor_dim: Optional[int] = None

    # Tri-modal MoE
    fusion_hidden_dim: int = 256
    projection_dropout: float = 0.1
    router_dropout: float = 0.1
    head_hidden_dim: int = 128
    head_dropout: float = 0.1

    # Optional anti-collapse regularization for branch balancing
    gate_balance_weight: float = 0.0
    # Per-position entropy regularisation: pushes gates toward soft routing (0 = off)
    gate_entropy_weight: float = 0.0
    # Descriptor-modality dropout probability during training
    desc_modality_dropout: float = 0.0
    gate_target: Tuple[float, float, float] = (1.0/3.0, 1.0/3.0, 1.0/3.0)

    # Training regularization
    label_smoothing: float = 0.0
    clip_grad_norm: float = 1.0

    # Descriptor preprocessing (fit on train only)
    descriptor_winsorize_lower_q: float = 0.01
    descriptor_winsorize_upper_q: float = 0.99
    descriptor_standardize: bool = True
    descriptor_eps: float = 1e-8

    # Molecule processing
    add_hydrogens: bool = False


class TAMEPredictor(BasePredictor):
    """Predictor using Graph + Text + RDKit descriptors with element-wise MoE fusion."""

    def __init__(
        self,
        config: Optional[TAMEPredictorConfig] = None,
        text_embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        descriptor_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or TAMEPredictorConfig()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.text_embedding_fn = text_embedding_fn
        self.descriptor_fn = descriptor_fn

        self._encoder: Optional[ChebNetEncoder] = None
        self._pooling: Optional[BasePooling] = None
        self._moe_module: Optional[TriModalElementWiseMoE] = None
        self._in_channels: Optional[int] = None
        self._is_fitted: bool = False

        self._desc_median: Optional[np.ndarray] = None
        self._desc_clip_low: Optional[np.ndarray] = None
        self._desc_clip_high: Optional[np.ndarray] = None
        self._desc_mean: Optional[np.ndarray] = None
        self._desc_std: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        smiles_list: List[str],
        labels: List[Any],
        val_smiles: Optional[List[str]] = None,
        val_labels: Optional[List[Any]] = None,
        *,
        text_embeddings: Optional[np.ndarray] = None,
        val_text_embeddings: Optional[np.ndarray] = None,
        descriptor_features: Optional[np.ndarray] = None,
        val_descriptor_features: Optional[np.ndarray] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        encoder_lr_mult: float = 1.0,
        early_stopping_metric: str = "val_loss",
        patience: int = 10,
        validation_split: float = 0.1,
        verbose: bool = True,
        seed: int = 0,
        init_encoder_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if len(smiles_list) != len(labels):
            raise ValueError("smiles_list and labels must have the same length")
        if not (0.0 <= float(self.config.desc_modality_dropout) < 1.0):
            raise ValueError("desc_modality_dropout must be in [0, 1)")
        if float(self.config.gate_balance_weight) < 0.0:
            raise ValueError("gate_balance_weight must be >= 0")
        if not (0.0 <= float(self.config.descriptor_winsorize_lower_q) < 1.0):
            raise ValueError("descriptor_winsorize_lower_q must be in [0, 1)")
        if not (0.0 < float(self.config.descriptor_winsorize_upper_q) <= 1.0):
            raise ValueError("descriptor_winsorize_upper_q must be in (0, 1]")
        if float(self.config.descriptor_winsorize_lower_q) >= float(self.config.descriptor_winsorize_upper_q):
            raise ValueError("descriptor_winsorize_lower_q must be < descriptor_winsorize_upper_q")

        train_graphs, in_channels = self._precompute_graphs(smiles_list, verbose=verbose)
        self._in_channels = in_channels

        train_text = self._get_or_compute_text_embeddings(smiles_list, text_embeddings)
        train_desc = self._get_or_compute_descriptors(smiles_list, descriptor_features)
        train_desc = np.asarray(train_desc, dtype=np.float32)

        y_all = np.asarray(labels, dtype=np.float32)
        if y_all.ndim == 1:
            y_all = y_all.reshape(-1, 1)
        elif y_all.ndim != 2:
            raise ValueError("labels must be 1D or 2D array-like")

        if int(self.config.num_tasks) != int(y_all.shape[1]):
            raise ValueError(
                f"labels task dimension ({y_all.shape[1]}) does not match config.num_tasks ({self.config.num_tasks})"
            )

        if self.config.task == "classification":
            y_all = np.where(np.isfinite(y_all), np.clip(y_all, 0.0, 1.0), np.nan)

        rng = np.random.default_rng(seed)
        idx = np.arange(len(smiles_list))
        rng.shuffle(idx)

        if val_smiles is not None and val_labels is not None:
            train_idx = idx
            val_graphs, _ = self._precompute_graphs(val_smiles, verbose=verbose)
            y_val = np.asarray(val_labels, dtype=np.float32)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            elif y_val.ndim != 2:
                raise ValueError("val_labels must be 1D or 2D array-like")
            if int(y_val.shape[1]) != int(self.config.num_tasks):
                raise ValueError(
                    f"val_labels task dimension ({y_val.shape[1]}) does not match config.num_tasks ({self.config.num_tasks})"
                )
            if self.config.task == "classification":
                y_val = np.where(np.isfinite(y_val), np.clip(y_val, 0.0, 1.0), np.nan)
            val_text = self._get_or_compute_text_embeddings(val_smiles, val_text_embeddings)
            val_desc = self._get_or_compute_descriptors(val_smiles, val_descriptor_features)
            val_idx = list(range(len(val_smiles)))
        else:
            n_val = max(1, int(len(smiles_list) * validation_split))
            val_idx = idx[:n_val].tolist()
            train_idx = idx[n_val:]
            val_graphs = train_graphs
            y_val = y_all
            val_text = train_text
            val_desc = train_desc

        self._fit_descriptor_preprocessor(train_desc[train_idx])
        train_desc = self._transform_descriptors(train_desc)
        val_desc = self._transform_descriptors(np.asarray(val_desc, dtype=np.float32))

        if verbose and self._desc_median is not None:
            print(
                "Descriptor preprocessing enabled "
                f"(winsor q=[{self.config.descriptor_winsorize_lower_q:.3f}, {self.config.descriptor_winsorize_upper_q:.3f}], "
                f"standardize={bool(self.config.descriptor_standardize)})"
            )

        self._build_model(desc_dim=int(train_desc.shape[1]))

        if init_encoder_state is not None:
            if self._encoder is None:
                raise RuntimeError("Encoder not built")
            self._encoder.load_state_dict(init_encoder_state, strict=True)
            if verbose:
                print("Initialized TAME encoder from external checkpoint state")

        # Differential Learning Rates Setup
        encoder_params = list(self._encoder.parameters()) if self._encoder is not None else []
        other_params = []
        if self._pooling is not None:
            other_params += list(self._pooling.parameters())
        if self._moe_module is not None:
            other_params += list(self._moe_module.parameters())
            
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": learning_rate})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": learning_rate * encoder_lr_mult})
            
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        if self.config.task == "classification":
            loss_fn: nn.Module = nn.BCEWithLogitsLoss(reduction="none")
        else:
            loss_fn = nn.MSELoss(reduction="none")

        history = {"train_loss": [], "val_loss": [], "val_auc": [], "graph_gate": [], "text_gate": [], "desc_gate": []}
        if early_stopping_metric == "val_auc":
            best_val = float("-inf")
        else:
            best_val = float("inf")
        best_state = None
        stale = 0

        for epoch in range(1, num_epochs + 1):
            tr_loss, gate_stats = self._train_epoch(
                train_graphs,
                train_text,
                train_desc,
                y_all,
                train_idx,
                batch_size,
                optimizer,
                loss_fn,
                rng,
            )
            eval_metrics = self._evaluate(
                val_graphs,
                np.asarray(val_text, dtype=np.float32),
                np.asarray(val_desc, dtype=np.float32),
                y_val,
                val_idx,
                batch_size,
                loss_fn,
            )
            vl_loss = eval_metrics["val_loss"]
            vl_auc = eval_metrics.get("val_auc", 0.5)

            history["train_loss"].append(float(tr_loss))
            history["val_loss"].append(float(vl_loss))
            history["val_auc"].append(float(vl_auc))
            history["graph_gate"].append(float(gate_stats[0]))
            history["text_gate"].append(float(gate_stats[1]))
            history["desc_gate"].append(float(gate_stats[2]))

            if verbose and (epoch == 1 or epoch % 5 == 0):
                auc_str = f" | val_auc={vl_auc:.4f}" if self.config.task == "classification" else ""
                print(
                    f"Epoch {epoch:03d} | train={tr_loss:.6f} | val_loss={vl_loss:.6f}{auc_str} | "
                    f"gates G/T/D={gate_stats[0]:.3f}/{gate_stats[1]:.3f}/{gate_stats[2]:.3f}"
                )

            if early_stopping_metric == "val_auc":
                is_better = vl_auc > best_val
                current_val = vl_auc
            else:
                is_better = vl_loss < best_val
                current_val = vl_loss

            if is_better:
                best_val = float(current_val)
                best_state = self._get_state_dict()
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self._load_state_dict(best_state)

        self._is_fitted = True
        return history

    def predict(
        self,
        smiles: str,
        text_embedding: Optional[np.ndarray] = None,
        descriptor_feature: Optional[np.ndarray] = None,
    ) -> Optional[Any]:
        self._check_fitted()
        try:
            g = smiles_to_graph(smiles, add_hydrogens=self.config.add_hydrogens)
        except Exception:
            return None

        text = self._get_single_text_embedding(smiles, text_embedding)
        desc = self._get_single_descriptor(smiles, descriptor_feature)
        desc = self._transform_descriptors(desc.reshape(1, -1))[0]

        bg = batch_graphs([g])
        x, edge_index, edge_weight, batch = self._to_tensors(bg)
        text_t = torch.from_numpy(text.reshape(1, -1).astype(np.float32)).to(self.device)
        desc_t = torch.from_numpy(desc.reshape(1, -1).astype(np.float32)).to(self.device)

        self._set_eval_mode()
        with torch.no_grad():
            logits, _ = self._forward(x, edge_index, edge_weight, batch, text_t, desc_t)
            if self.config.task == "classification":
                out = torch.sigmoid(logits)
            else:
                out = logits

            out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
            if int(self.config.num_tasks) == 1:
                return float(out_np.reshape(-1)[0])
            return out_np

    def predict_batch(
        self,
        smiles_list: List[str],
        text_embeddings: Optional[np.ndarray] = None,
        descriptor_features: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        self._check_fitted()
        num_tasks = int(self.config.num_tasks)
        if num_tasks == 1:
            preds = np.full(len(smiles_list), np.nan, dtype=np.float32)
        else:
            preds = np.full((len(smiles_list), num_tasks), np.nan, dtype=np.float32)

        text_all = self._get_or_compute_text_embeddings(smiles_list, text_embeddings)
        desc_all = self._get_or_compute_descriptors(smiles_list, descriptor_features)
        desc_all = self._transform_descriptors(np.asarray(desc_all, dtype=np.float32))

        valid_positions: List[int] = []
        valid_graphs = []
        valid_text = []
        valid_desc = []

        for i, smi in enumerate(smiles_list):
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                valid_graphs.append(g)
                valid_positions.append(i)
                valid_text.append(text_all[i])
                valid_desc.append(desc_all[i])
            except Exception:
                continue

        if len(valid_graphs) == 0:
            return preds

        valid_text_np = np.asarray(valid_text, dtype=np.float32)
        valid_desc_np = np.asarray(valid_desc, dtype=np.float32)

        self._set_eval_mode()
        with torch.no_grad():
            for start in range(0, len(valid_graphs), batch_size):
                graphs = valid_graphs[start:start + batch_size]
                positions = valid_positions[start:start + batch_size]
                text = valid_text_np[start:start + batch_size]
                desc = valid_desc_np[start:start + batch_size]

                bg = batch_graphs(graphs)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                text_t = torch.from_numpy(text).to(self.device)
                desc_t = torch.from_numpy(desc).to(self.device)

                logits, _ = self._forward(x, edge_index, edge_weight, batch, text_t, desc_t)
                if self.config.task == "classification":
                    out = torch.sigmoid(logits)
                else:
                    out = logits
                out_np = out.cpu().numpy().astype(np.float32)
                if num_tasks == 1:
                    out_np = out_np.reshape(-1)
                    for pos, val in zip(positions, out_np.tolist()):
                        preds[pos] = float(val)
                else:
                    for row_i, pos in enumerate(positions):
                        preds[pos, :] = out_np[row_i, :]

        return preds

    def save(self, path: str) -> None:
        self._check_fitted()
        torch.save(
            {
                "encoder_state": self._encoder.state_dict() if self._encoder is not None else None,
                "pooling_state": self._pooling.state_dict() if self._pooling is not None else None,
                "moe_module_state": self._moe_module.state_dict() if self._moe_module is not None else None,
                "in_channels": self._in_channels,
                "config": self.config.to_dict(),
                "desc_median": self._desc_median,
                "desc_clip_low": self._desc_clip_low,
                "desc_clip_high": self._desc_clip_high,
                "desc_mean": self._desc_mean,
                "desc_std": self._desc_std,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.config = TAMEPredictorConfig(**ckpt["config"])
        self._in_channels = int(ckpt["in_channels"])
        self._desc_median = ckpt.get("desc_median", None)
        self._desc_clip_low = ckpt.get("desc_clip_low", None)
        self._desc_clip_high = ckpt.get("desc_clip_high", None)
        self._desc_mean = ckpt.get("desc_mean", None)
        self._desc_std = ckpt.get("desc_std", None)
        desc_dim = int(self.config.descriptor_dim) if self.config.descriptor_dim is not None else None
        if desc_dim is None and self._desc_median is not None:
            desc_dim = int(np.asarray(self._desc_median).shape[0])
        if desc_dim is None:
            raise RuntimeError("Descriptor dimension missing in checkpoint")

        self._build_model(desc_dim=desc_dim)
        if self._encoder is None or self._pooling is None or self._moe_module is None:
            raise RuntimeError("Model build failed")

        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._pooling.load_state_dict(ckpt["pooling_state"])
        self._moe_module.load_state_dict(ckpt["moe_module_state"])
        self._is_fitted = True

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        text_embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        descriptor_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        device: Optional[str] = None,
    ) -> "TAMEPredictor":
        predictor = cls(text_embedding_fn=text_embedding_fn, descriptor_fn=descriptor_fn, device=device)
        predictor.load(path)
        return predictor

    def _build_model(self, desc_dim: int) -> None:
        if self._in_channels is None:
            raise RuntimeError("in_channels unknown; call fit() or set from checkpoint first")

        self.config.descriptor_dim = int(desc_dim)

        self._encoder = ChebNetEncoder(
            in_channels=self._in_channels,
            hidden_channels=self.config.hidden_channels,
            K=self.config.K,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            lambda_max=self.config.lambda_max,
        ).to(self.device)

        self._pooling = create_pooling(
            pool_type=self.config.pool,
            input_dim=self.config.hidden_channels,
            n_iters=self.config.set2set_processing_steps,
            hidden_dim=self.config.attention_hidden_dim,
        ).to(self.device)

        graph_dim = int(self._pooling.output_dim)
        self._moe_module = TriModalElementWiseMoE(
            graph_dim=graph_dim,
            text_dim=int(self.config.text_embedding_dim),
            desc_dim=int(desc_dim),
            num_tasks=int(self.config.num_tasks),
            hidden_dim=int(self.config.fusion_hidden_dim),
            dropout=float(self.config.projection_dropout),
            router_dropout=float(self.config.router_dropout),
            head_hidden_dim=int(self.config.head_hidden_dim),
            head_dropout=float(self.config.head_dropout),
        ).to(self.device)
        self._moe_module._init_router_at_target(self.config.gate_target)

    def _forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        text_t: torch.Tensor,
        desc_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._encoder is None or self._pooling is None or self._moe_module is None:
            raise RuntimeError("Model not built")

        node_emb = self._encoder(x, edge_index, edge_weight, batch)
        graph_emb = self._pooling(node_emb, batch)
        return self._moe_module(graph_emb, text_t, desc_t)

    def _train_epoch(
        self,
        graphs,
        text_emb: np.ndarray,
        desc_feat: np.ndarray,
        y_all: np.ndarray,
        train_idx: np.ndarray,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        rng: np.random.Generator,
    ) -> Tuple[float, Tuple[float, float, float]]:
        if self._encoder is None or self._pooling is None or self._moe_module is None:
            raise RuntimeError("Model not built")

        self._encoder.train()
        self._pooling.train()
        self._moe_module.train()

        order = rng.permutation(train_idx)
        total_loss = 0.0
        n_batches = 0
        gate_sums = np.zeros(3, dtype=np.float64)

        desc_drop_p = float(self.config.desc_modality_dropout)
        gate_balance_weight = float(self.config.gate_balance_weight)
        gate_entropy_weight = float(self.config.gate_entropy_weight)
        gate_target = self.config.gate_target
        label_smoothing = float(self.config.label_smoothing)
        clip_norm = float(self.config.clip_grad_norm)

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            raw_batch = [graphs[i] for i in idx]
            valid = [(g, text_emb[i], desc_feat[i], y_all[i]) for g, i in zip(raw_batch, idx) if g is not None]
            if not valid:
                continue
            batch_graphs_list, batch_text_list, batch_desc_list, batch_y_list = zip(*valid)

            batch_text = np.stack(batch_text_list, axis=0)
            batch_desc = np.stack(batch_desc_list, axis=0)
            batch_y = np.stack(batch_y_list, axis=0)

            bg = batch_graphs(batch_graphs_list)
            x, edge_index, edge_weight, batch = self._to_tensors(bg)
            text_t = torch.from_numpy(batch_text.astype(np.float32)).to(self.device)
            desc_t = torch.from_numpy(batch_desc.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(batch_y.astype(np.float32)).to(self.device)
            if y_t.ndim == 1:
                y_t = y_t.unsqueeze(-1)

            # Randomly drop descriptor modality for a subset of samples
            # so graph/text branches remain useful during training.
            if desc_drop_p > 0.0:
                keep_prob = 1.0 - desc_drop_p
                drop_mask = (torch.rand((desc_t.size(0), 1), device=desc_t.device) < desc_drop_p)
                desc_t = desc_t * (~drop_mask).float() / keep_prob

            optimizer.zero_grad()
            logits, aux = self._forward(x, edge_index, edge_weight, batch, text_t, desc_t)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            observed = torch.isfinite(y_t)
            if observed.any():
                targets = torch.where(observed, y_t, torch.zeros_like(y_t))
                if label_smoothing > 0.0 and self.config.task == "classification":
                    targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
                per_elem = loss_fn(logits, targets)
                task_loss = (per_elem * observed.float()).sum() / observed.float().sum().clamp_min(1.0)
            else:
                task_loss = logits.sum() * 0.0

            # Encourage non-collapsed gate usage across modalities.
            gate_balance = logits.sum() * 0.0
            if gate_balance_weight > 0.0:
                g_graph = aux["graph_gate"].mean()
                g_text = aux["text_gate"].mean()
                g_desc = aux["desc_gate"].mean()
                gate_balance = (
                    (g_graph - gate_target[0]).pow(2)
                    + (g_text - gate_target[1]).pow(2)
                    + (g_desc - gate_target[2]).pow(2)
                )

            # Entropy regularisation: penalise hard (0/1) per-position routing.
            gate_entropy_loss = logits.sum() * 0.0
            if gate_entropy_weight > 0.0:
                gf = aux["gates_full"]  # (B, H, 3)
                ent = -(gf * (gf + 1e-8).log()).sum(dim=-1)  # (B, H)
                gate_entropy_loss = -gate_entropy_weight * ent.mean()

            loss = task_loss + (gate_balance_weight * gate_balance) + gate_entropy_loss
            loss.backward()
            if clip_norm > 0.0:
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            gate_sums[0] += float(aux["graph_gate"].mean().detach().cpu().item())
            gate_sums[1] += float(aux["text_gate"].mean().detach().cpu().item())
            gate_sums[2] += float(aux["desc_gate"].mean().detach().cpu().item())

        avg_loss = total_loss / max(n_batches, 1)
        gate_means = tuple((gate_sums / max(n_batches, 1)).tolist())
        return avg_loss, gate_means

    def _evaluate(
        self,
        graphs,
        text_emb: np.ndarray,
        desc_feat: np.ndarray,
        y_all: np.ndarray,
        eval_idx: List[int],
        batch_size: int,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        if self._encoder is None or self._pooling is None or self._moe_module is None:
            raise RuntimeError("Model not built")

        self._set_eval_mode()
        total_loss = 0.0
        n_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for start in range(0, len(eval_idx), batch_size):
                idx = eval_idx[start:start + batch_size]
                raw_batch = [graphs[i] for i in idx]
                valid = [(g, text_emb[i], desc_feat[i], y_all[i]) for g, i in zip(raw_batch, idx) if g is not None]
                if not valid:
                    continue
                batch_graphs_list, batch_text_list, batch_desc_list, batch_y_list = zip(*valid)

                batch_text = np.stack(batch_text_list, axis=0)
                batch_desc = np.stack(batch_desc_list, axis=0)
                batch_y = np.stack(batch_y_list, axis=0)

                bg = batch_graphs(batch_graphs_list)
                x, edge_index, edge_weight, batch = self._to_tensors(bg)
                text_t = torch.from_numpy(batch_text.astype(np.float32)).to(self.device)
                desc_t = torch.from_numpy(batch_desc.astype(np.float32)).to(self.device)
                y_t = torch.from_numpy(batch_y.astype(np.float32)).to(self.device)
                if y_t.ndim == 1:
                    y_t = y_t.unsqueeze(-1)

                logits, _ = self._forward(x, edge_index, edge_weight, batch, text_t, desc_t)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)

                observed = torch.isfinite(y_t)
                if observed.any():
                    targets = torch.where(observed, y_t, torch.zeros_like(y_t))
                    per_elem = loss_fn(logits, targets)
                    n_obs = int(observed.sum().item())
                    total_loss += float((per_elem * observed.float()).sum().item())
                    n_total += n_obs

                    if self.config.task == "classification":
                        all_preds.extend(torch.sigmoid(logits)[observed].cpu().numpy().tolist())
                    else:
                        all_preds.extend(logits[observed].cpu().numpy().tolist())
                    all_targets.extend(targets[observed].cpu().numpy().tolist())

        res = {"val_loss": total_loss / max(n_total, 1)}
        
        if self.config.task == "classification" and len(all_targets) > 0:
            try:
                from sklearn.metrics import roc_auc_score
                res["val_auc"] = float(roc_auc_score(np.array(all_targets).flatten(), np.array(all_preds).flatten()))
            except ValueError:
                res["val_auc"] = 0.5
                
        return res

    def _set_eval_mode(self) -> None:
        if self._encoder is not None:
            self._encoder.eval()
        if self._pooling is not None:
            self._pooling.eval()
        if self._moe_module is not None:
            self._moe_module.eval()

    def _get_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self._encoder is not None:
            params += list(self._encoder.parameters())
        if self._pooling is not None:
            params += list(self._pooling.parameters())
        if self._moe_module is not None:
            params += list(self._moe_module.parameters())
        return params

    def _get_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._encoder is None or self._pooling is None or self._moe_module is None:
            raise RuntimeError("Model not built")
        return {
            "encoder": self._encoder.state_dict(),
            "pooling": self._pooling.state_dict(),
            "moe_module": self._moe_module.state_dict(),
        }

    def _load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if self._encoder is None or self._pooling is None or self._moe_module is None:
            raise RuntimeError("Model not built")
        self._encoder.load_state_dict(state["encoder"])
        self._pooling.load_state_dict(state["pooling"])
        self._moe_module.load_state_dict(state["moe_module"])

    def _to_tensors(self, bg):
        x = torch.from_numpy(bg.X.astype(np.float32)).to(self.device)
        edge_index = torch.from_numpy(bg.edge_index.astype(np.int64)).to(self.device)
        edge_weight = torch.from_numpy(bg.edge_weight.astype(np.float32)).to(self.device)
        batch = torch.from_numpy(bg.batch.astype(np.int64)).to(self.device)
        return x, edge_index, edge_weight, batch

    def _precompute_graphs(self, smiles_list: List[str], verbose: bool = True):
        graphs: List[Optional[Any]] = []
        in_channels = None
        n_failed = 0
        for smi in smiles_list:
            try:
                g = smiles_to_graph(smi, add_hydrogens=self.config.add_hydrogens)
                graphs.append(g)
                if in_channels is None:
                    in_channels = int(g.X.shape[1])
            except Exception:
                graphs.append(None)
                n_failed += 1
        if in_channels is None:
            raise RuntimeError("No valid molecular graphs found")
        if n_failed > 0 and verbose:
            print(f"Skipped {n_failed} invalid SMILES")
        return graphs, in_channels

    def _get_or_compute_text_embeddings(
        self,
        smiles_list: List[str],
        text_embeddings: Optional[np.ndarray],
    ) -> np.ndarray:
        if text_embeddings is not None:
            arr = np.asarray(text_embeddings, dtype=np.float32)
            if arr.shape[0] != len(smiles_list):
                raise ValueError("text_embeddings length mismatch")
            return arr
        if self.text_embedding_fn is None:
            raise ValueError("text_embeddings required when text_embedding_fn is not provided")
        arr = np.asarray(self.text_embedding_fn(smiles_list), dtype=np.float32)
        if arr.shape[0] != len(smiles_list):
            raise ValueError("text_embedding_fn returned wrong number of rows")
        return arr

    def _compute_rdkit_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        desc_names = [name for name, _ in Descriptors.descList]
        n_desc = len(desc_names)
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append(np.full((n_desc,), np.nan, dtype=np.float32))
                continue
            all_desc = Descriptors.CalcMolDescriptors(mol)
            row = []
            for name in desc_names:
                try:
                    row.append(float(all_desc.get(name, np.nan)))
                except Exception:
                    row.append(np.nan)
            rows.append(np.asarray(row, dtype=np.float32))
        return np.vstack(rows).astype(np.float32)

    def _get_or_compute_descriptors(
        self,
        smiles_list: List[str],
        descriptor_features: Optional[np.ndarray],
    ) -> np.ndarray:
        if descriptor_features is not None:
            arr = np.asarray(descriptor_features, dtype=np.float32)
            if arr.shape[0] != len(smiles_list):
                raise ValueError("descriptor_features length mismatch")
            return arr
        if self.descriptor_fn is not None:
            arr = np.asarray(self.descriptor_fn(smiles_list), dtype=np.float32)
            if arr.shape[0] != len(smiles_list):
                raise ValueError("descriptor_fn returned wrong number of rows")
            return arr
        return self._compute_rdkit_descriptors(smiles_list)

    def _fit_descriptor_imputer(self, train_desc: np.ndarray) -> None:
        med = np.nanmedian(train_desc, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        self._desc_median = med

    def _fit_descriptor_preprocessor(self, train_desc: np.ndarray) -> None:
        train_desc = np.asarray(train_desc, dtype=np.float32)
        self._fit_descriptor_imputer(train_desc)
        imputed = self._impute_descriptors(train_desc)

        q_low = float(self.config.descriptor_winsorize_lower_q)
        q_high = float(self.config.descriptor_winsorize_upper_q)
        self._desc_clip_low = np.quantile(imputed, q_low, axis=0).astype(np.float32)
        self._desc_clip_high = np.quantile(imputed, q_high, axis=0).astype(np.float32)

        clipped = np.clip(imputed, self._desc_clip_low, self._desc_clip_high)
        if bool(self.config.descriptor_standardize):
            eps = float(self.config.descriptor_eps)
            mean = clipped.mean(axis=0).astype(np.float32)
            std = clipped.std(axis=0).astype(np.float32)
            std = np.where(std >= eps, std, 1.0).astype(np.float32)
            self._desc_mean = mean
            self._desc_std = std
        else:
            self._desc_mean = None
            self._desc_std = None

    def _transform_descriptors(self, desc: np.ndarray) -> np.ndarray:
        arr = self._impute_descriptors(desc)
        if self._desc_clip_low is not None and self._desc_clip_high is not None:
            arr = np.clip(arr, self._desc_clip_low, self._desc_clip_high)
        if self._desc_mean is not None and self._desc_std is not None:
            arr = (arr - self._desc_mean) / self._desc_std
        return arr.astype(np.float32)

    def _impute_descriptors(self, desc: np.ndarray) -> np.ndarray:
        arr = np.asarray(desc, dtype=np.float32).copy()
        if self._desc_median is None:
            med = np.nanmedian(arr, axis=0)
            med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        else:
            med = self._desc_median

        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = np.take(med, np.where(mask)[1])
        return arr

    def _get_single_text_embedding(self, smiles: str, text_embedding: Optional[np.ndarray]) -> np.ndarray:
        if text_embedding is not None:
            return np.asarray(text_embedding, dtype=np.float32)
        if self.text_embedding_fn is None:
            raise ValueError("text_embedding required for single prediction")
        arr = np.asarray(self.text_embedding_fn([smiles]), dtype=np.float32)
        return arr[0]

    def _get_single_descriptor(self, smiles: str, descriptor_feature: Optional[np.ndarray]) -> np.ndarray:
        if descriptor_feature is not None:
            return np.asarray(descriptor_feature, dtype=np.float32)
        if self.descriptor_fn is not None:
            arr = np.asarray(self.descriptor_fn([smiles]), dtype=np.float32)
            return arr[0]
        return self._compute_rdkit_descriptors([smiles])[0]
