"""CheMeleon-style ChebNet pretraining utilities.

Implements foundation pretraining where a ChebNet graph encoder learns to predict
classical molecular descriptors from molecular graphs.

Pipeline covered by this module:
1. Load cached SMILES corpora.
2. Load cached descriptor matrices from disk.
3. Standardize + winsorize targets and keep a validity mask.
4. Train with dynamic masking (85% masked, 15% supervised) and masked MSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.core.graph_encoder import ChebNetEncoder
from models.core.pooling import BasePooling, create_pooling
from utils.batched_mol_graph import BatchedMolGraph, batch_graphs
from utils.molecular_graph import MolGraph, smiles_to_graph


DEFAULT_DESCRIPTOR_DIM = 1613
DEFAULT_KEEP_FRACTION = 0.15
DEFAULT_NODE_MASK_FRACTION = 0.15
DEFAULT_ATOM_TYPE_DIM = 10


@dataclass
class DescriptorTargetStats:
    """Statistics and metadata for descriptor target preprocessing."""

    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    selected_indices: np.ndarray


@dataclass
class DescriptorPretrainingData:
    """Prepared data ready for descriptor pretraining."""

    smiles: List[str]
    targets: np.ndarray
    target_valid_mask: np.ndarray
    stats: DescriptorTargetStats


def load_or_simulate_unlabeled_smiles(
    smiles_path: Optional[str] = None,
    *,
    max_smiles: int = 1_000_000,
    simulate_if_missing: bool = True,
    seed: int = 0,
    suppress_rdkit_errors: bool = True,
) -> List[str]:
    """Load a large unlabeled SMILES corpus from disk.

    The legacy simulation options are kept only for API compatibility and are
    no longer used in this cached-descriptor workflow.
    """
    _ = seed
    _ = suppress_rdkit_errors

    if smiles_path is not None and Path(smiles_path).exists():
        smiles = _read_smiles_file(smiles_path, max_smiles=max_smiles)
        if len(smiles) > 0:
            return smiles

    if simulate_if_missing:
        raise FileNotFoundError(
            "No valid SMILES source found. SMILES simulation was removed from "
            "the cached-descriptor pretraining pipeline."
        )

    raise FileNotFoundError(
        "No valid SMILES source found and simulate_if_missing=False."
    )


def _read_smiles_file(path: str, *, max_smiles: int) -> List[str]:
    """Read SMILES from .smi/.txt/.csv/.tsv style files."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, nrows=max_smiles)
        if "smiles" in df.columns:
            return df["smiles"].astype(str).str.strip().tolist()
        return df.iloc[:, 0].astype(str).str.strip().tolist()

    smiles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            smiles.append(token)
            if len(smiles) >= max_smiles:
                break
    return smiles


def preprocess_descriptor_targets(
    descriptor_matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    output_dim: int = DEFAULT_DESCRIPTOR_DIM,
    min_valid_fraction: float = 0.95,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, DescriptorTargetStats]:
    """Select descriptors and apply standardization + winsorization.

    Steps:
    1) Select stable descriptor columns with enough non-NaN coverage.
    2) Keep exactly output_dim columns.
    3) Standardize (z-score using finite values).
    4) Winsorize by clipping z-scores to [-6, 6].

    Returns:
        targets: float32 array (N, output_dim)
        valid_mask: bool array (N, output_dim), True where target is valid
        stats: preprocessing statistics for reproducibility
    """
    if descriptor_matrix.ndim != 2:
        raise ValueError("descriptor_matrix must be 2D")

    n_samples, n_raw = descriptor_matrix.shape
    if n_raw != len(feature_names):
        raise ValueError("feature_names length must match descriptor_matrix second dimension")

    finite_mask = np.isfinite(descriptor_matrix)
    valid_fraction = finite_mask.mean(axis=0)

    # Use finite-only mean/std to determine stable columns.
    col_mean = np.nanmean(descriptor_matrix, axis=0)
    col_std = np.nanstd(descriptor_matrix, axis=0)

    eligible = np.where((valid_fraction >= min_valid_fraction) & (col_std > eps))[0]
    if eligible.size < output_dim:
        # Backoff strategy: relax only coverage, still require non-zero variance.
        eligible = np.where(col_std > eps)[0]

    if eligible.size < output_dim:
        raise ValueError(
            f"Not enough usable descriptor columns ({eligible.size}) for output_dim={output_dim}."
        )

    selected = eligible[:output_dim]
    selected_matrix = descriptor_matrix[:, selected].astype(np.float32, copy=False)

    mean = np.nanmean(selected_matrix, axis=0)
    std = np.nanstd(selected_matrix, axis=0)
    std = np.where(std < eps, 1.0, std)

    z = (selected_matrix - mean) / std
    valid = np.isfinite(z)
    z = np.where(valid, z, 0.0)
    z = np.clip(z, -6.0, 6.0)

    stats = DescriptorTargetStats(
        feature_names=[feature_names[i] for i in selected.tolist()],
        mean=mean.astype(np.float32, copy=False),
        std=std.astype(np.float32, copy=False),
        selected_indices=selected.astype(np.int64, copy=False),
    )

    assert z.shape == (n_samples, output_dim)
    assert valid.shape == (n_samples, output_dim)

    return z.astype(np.float32, copy=False), valid.astype(bool, copy=False), stats


class DescriptorPretrainingDataset(Dataset):
    """Dataset yielding graph inputs with Mordred descriptor targets."""

    def __init__(
        self,
        smiles: Sequence[str],
        targets: np.ndarray,
        target_valid_mask: np.ndarray,
        *,
        add_hydrogens: bool = False,
        drop_invalid_graphs: bool = True,
    ) -> None:
        if len(smiles) != targets.shape[0] or targets.shape != target_valid_mask.shape:
            raise ValueError("smiles, targets, and target_valid_mask must be aligned")

        self._graphs: List[MolGraph] = []
        self._targets: List[np.ndarray] = []
        self._target_valid_masks: List[np.ndarray] = []
        self._raw_smiles: List[str] = []

        for smi, tgt, valid in zip(smiles, targets, target_valid_mask):
            try:
                g = smiles_to_graph(smi, add_hydrogens=add_hydrogens)
            except Exception:
                if drop_invalid_graphs:
                    continue
                raise

            self._graphs.append(g)
            self._targets.append(np.asarray(tgt, dtype=np.float32))
            self._target_valid_masks.append(np.asarray(valid, dtype=bool))
            self._raw_smiles.append(str(smi))

        if len(self._graphs) == 0:
            raise ValueError("No valid molecular graphs found for pretraining dataset.")

        self.in_channels = int(self._graphs[0].X.shape[1])
        self.target_dim = int(self._targets[0].shape[0])

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index: int) -> Tuple[MolGraph, np.ndarray, np.ndarray]:
        return self._graphs[index], self._targets[index], self._target_valid_masks[index]


class DescriptorBatchCollator:
    """Collate function converting per-molecule records to batched tensors."""

    def __call__(
        self,
        batch: Sequence[Tuple[MolGraph, np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        graphs = [item[0] for item in batch]
        targets = np.stack([item[1] for item in batch], axis=0)
        target_valid_mask = np.stack([item[2] for item in batch], axis=0)

        bg: BatchedMolGraph = batch_graphs(graphs)

        return {
            "x": torch.from_numpy(bg.X.astype(np.float32, copy=False)),
            "edge_index": torch.from_numpy(bg.edge_index.astype(np.int64, copy=False)),
            "edge_weight": torch.from_numpy(bg.edge_weight.astype(np.float32, copy=False)),
            "batch": torch.from_numpy(bg.batch.astype(np.int64, copy=False)),
            "targets": torch.from_numpy(targets.astype(np.float32, copy=False)),
            "target_valid_mask": torch.from_numpy(target_valid_mask.astype(bool, copy=False)),
        }


def create_pretraining_dataloader(
    dataset: DescriptorPretrainingDataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for descriptor pretraining."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=DescriptorBatchCollator(),
        pin_memory=True,
        drop_last=False,
    )


class CheMeleonChebPretrainer(nn.Module):
    """ChebNet encoder + pooling + 2-layer FNN pretraining head."""

    def __init__(
        self,
        *,
        in_channels: int,
        descriptor_dim: int = DEFAULT_DESCRIPTOR_DIM,
        hidden_channels: int = 256,
        K: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
        lambda_max: float = 2.0,
        pooling: str = "mean",
        head_hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        self.encoder = ChebNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            K=K,
            num_layers=num_layers,
            dropout=dropout,
            lambda_max=lambda_max,
        )

        self.pooling: BasePooling = create_pooling(pooling, input_dim=hidden_channels)

        # Two-layer FNN head required by the pretraining objective.
        self.pretraining_head = nn.Sequential(
            nn.Linear(self.pooling.output_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, descriptor_dim),
        )

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of molecules into graph-level embeddings."""
        node_emb = self.encoder(x, edge_index, edge_weight, batch)
        return self.pooling(node_emb, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        *,
        return_graph_emb: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        graph_emb = self.encode_graph(x, edge_index, edge_weight, batch)
        preds = self.pretraining_head(graph_emb)
        if return_graph_emb:
            return preds, graph_emb
        return preds


def attribute_mask_node_features(
    x: torch.Tensor,
    *,
    mask_fraction: float = DEFAULT_NODE_MASK_FRACTION,
    atom_type_dim: int = DEFAULT_ATOM_TYPE_DIM,
    mask_token: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Attribute masking for node-level pretraining.

    Returns:
        masked_x: input features with masked nodes replaced by mask_token
        node_mask: bool mask over nodes (True means masked/supervised)
        atom_targets: class ids of original atom type for all nodes
    """
    if x.ndim != 2:
        raise ValueError("x must be 2D (n_nodes, n_features)")
    if not (0.0 < mask_fraction <= 1.0):
        raise ValueError("mask_fraction must be in (0, 1]")
    if atom_type_dim <= 1 or atom_type_dim > x.shape[1]:
        raise ValueError("atom_type_dim must be in [2, x.shape[1]]")

    n_nodes, n_feat = x.shape
    atom_slice = x[:, :atom_type_dim]
    atom_targets = torch.argmax(atom_slice, dim=1)

    random_mask = torch.rand((n_nodes,), device=x.device, generator=generator) < mask_fraction
    if not random_mask.any() and n_nodes > 0:
        forced_idx = torch.randint(0, n_nodes, (1,), device=x.device, generator=generator)
        random_mask[forced_idx] = True

    masked_x = x.clone()
    if mask_token is None:
        token = torch.zeros((n_feat,), dtype=x.dtype, device=x.device)
        token[-1] = -1.0
    else:
        if mask_token.ndim != 1 or mask_token.shape[0] != n_feat:
            raise ValueError("mask_token must have shape (n_features,)")
        token = mask_token.to(device=x.device, dtype=x.dtype)

    masked_x[random_mask] = token.unsqueeze(0)
    return masked_x, random_mask, atom_targets


class NodePredictionHead(nn.Module):
    """Simple linear node classifier for masked atom-type prediction."""

    def __init__(self, hidden_channels: int, num_atom_types: int = DEFAULT_ATOM_TYPE_DIM) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_channels, num_atom_types)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(node_emb)


class ChebNodePretrainer(nn.Module):
    """
    Stage 1 pretrainer: node-level attribute masking objective.

    Pipeline:
    1) Mask random node features with a special mask token
    2) Run Cheb encoder
    3) Predict original atom type for masked nodes only
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 256,
        K: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
        lambda_max: float = 2.0,
        atom_type_dim: int = DEFAULT_ATOM_TYPE_DIM,
    ) -> None:
        super().__init__()
        self.atom_type_dim = int(atom_type_dim)

        self.encoder = ChebNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            K=K,
            num_layers=num_layers,
            dropout=dropout,
            lambda_max=lambda_max,
        )
        self.node_head = NodePredictionHead(hidden_channels=hidden_channels, num_atom_types=self.atom_type_dim)

        # Learnable special token replacing masked node features.
        self.mask_token = nn.Parameter(torch.zeros(in_channels))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        *,
        mask_fraction: float = DEFAULT_NODE_MASK_FRACTION,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        masked_x, node_mask, atom_targets = attribute_mask_node_features(
            x,
            mask_fraction=mask_fraction,
            atom_type_dim=self.atom_type_dim,
            mask_token=self.mask_token,
            generator=generator,
        )

        node_emb = self.encoder(masked_x, edge_index, edge_weight, batch)
        logits = self.node_head(node_emb)

        if node_mask.any():
            loss = F.cross_entropy(logits[node_mask], atom_targets[node_mask])
        else:
            loss = logits.sum() * 0.0

        return {
            "loss": loss,
            "logits": logits,
            "atom_targets": atom_targets,
            "node_mask": node_mask,
        }


class GraphPredictionHead(nn.Module):
    """Two-layer FNN for graph-level descriptor prediction."""

    def __init__(self, in_dim: int, out_dim: int = DEFAULT_DESCRIPTOR_DIM, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChebGraphPretrainer(nn.Module):
    """
    Stage 2 pretrainer: graph-level descriptor prediction from stage-1 encoder.

    Encoder weights can be initialized from a fitted ChebNodePretrainer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        descriptor_dim: int = DEFAULT_DESCRIPTOR_DIM,
        hidden_channels: int = 256,
        K: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
        lambda_max: float = 2.0,
        pooling: str = "mean",
        head_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)

        self.encoder = ChebNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            K=K,
            num_layers=num_layers,
            dropout=dropout,
            lambda_max=lambda_max,
        )
        self.pooling: BasePooling = create_pooling(pooling, input_dim=hidden_channels)
        self.graph_head = GraphPredictionHead(
            in_dim=self.pooling.output_dim,
            out_dim=self.descriptor_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def load_encoder_from_stage1(self, stage1_model: ChebNodePretrainer, *, strict: bool = True) -> None:
        """Copy encoder weights from a trained stage-1 node pretrainer."""
        self.encoder.load_state_dict(stage1_model.encoder.state_dict(), strict=strict)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        *,
        return_graph_emb: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.encoder(x, edge_index, edge_weight, batch)
        graph_emb = self.pooling(node_emb, batch)
        preds = self.graph_head(graph_emb)
        if return_graph_emb:
            return preds, graph_emb
        return preds


def masked_mse_loss_dynamic(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_valid_mask: torch.Tensor,
    *,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic masking variant of masked MSE.

    Randomly keeps keep_fraction of valid descriptor targets per sample and
    computes MSE only on those retained values.
    """
    dyn_mask = build_dynamic_mask(
        target_valid_mask,
        keep_fraction=keep_fraction,
        generator=generator,
    )
    loss = masked_mse_loss(preds, targets, dyn_mask)
    return loss, dyn_mask


def build_cached_model_differentiator(
    *,
    stage: str,
    config: Dict[str, Any],
    prefix: str = "cheb_pretrain",
    hash_len: int = 10,
) -> str:
    """
    Build a compact, deterministic cache differentiator for checkpoints/artifacts.

    This helps separating cached models produced with different stage/config
    combinations.
    """
    payload = {
        "stage": stage,
        "config": config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[: int(hash_len)]
    stage_token = str(stage).replace(" ", "-").replace("_", "-").lower()
    return f"{prefix}__{stage_token}__h-{digest}"


def build_dynamic_mask(
    target_valid_mask: torch.Tensor,
    *,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build dynamic descriptor mask per batch (True means supervised)."""
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1]")

    random_mask = torch.rand(
        target_valid_mask.shape,
        device=target_valid_mask.device,
        generator=generator,
    ) < keep_fraction

    mask = target_valid_mask & random_mask

    # Ensure at least one supervised target per sample if valid targets exist.
    rows_without_supervision = (~mask.any(dim=1)) & target_valid_mask.any(dim=1)
    if rows_without_supervision.any():
        row_indices = rows_without_supervision.nonzero(as_tuple=False).squeeze(-1)
        for r in row_indices.tolist():
            valid_cols = target_valid_mask[r].nonzero(as_tuple=False).squeeze(-1)
            if valid_cols.numel() == 0:
                continue
            chosen_idx = int(valid_cols[torch.randint(0, valid_cols.numel(), (1,), device=valid_cols.device)])
            mask[r, chosen_idx] = True

    return mask


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE used for descriptor pretraining."""
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if mask.any():
        return F.mse_loss(preds[mask], targets[mask])

    # Keep graph connected if no valid targets in this batch.
    return preds.sum() * 0.0


def nt_xent_graph_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Batch NT-Xent loss between two stochastic graph-embedding views."""
    if emb_a.ndim != 2 or emb_b.ndim != 2:
        raise ValueError("emb_a and emb_b must be 2D tensors")
    if emb_a.shape != emb_b.shape:
        raise ValueError("emb_a and emb_b must have the same shape")
    if emb_a.shape[0] < 2:
        # No negatives available for contrastive learning.
        return emb_a.sum() * 0.0
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    z_a = F.normalize(emb_a, p=2, dim=1)
    z_b = F.normalize(emb_b, p=2, dim=1)
    logits_ab = (z_a @ z_b.T) / temperature
    logits_ba = (z_b @ z_a.T) / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)

    loss_ab = F.cross_entropy(logits_ab, labels)
    loss_ba = F.cross_entropy(logits_ba, labels)
    return 0.5 * (loss_ab + loss_ba)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_descriptor_pretraining(
    model: CheMeleonChebPretrainer,
    train_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
    device: Optional[torch.device] = None,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    graph_obj_weight: float = 0.0,
    graph_obj_temperature: float = 0.2,
    val_loader: Optional[DataLoader] = None,
    grad_clip_norm: Optional[float] = 5.0,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train ChebNet foundation model with dynamic masking + masked MSE."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if graph_obj_weight < 0.0:
        raise ValueError("graph_obj_weight must be >= 0")
    if graph_obj_temperature <= 0.0:
        raise ValueError("graph_obj_temperature must be > 0")

    model.to(device)
    history: Dict[str, List[float]] = {"train_loss": []}
    if graph_obj_weight > 0.0:
        history["train_graph_loss"] = []
        history["train_total_loss"] = []
    if val_loader is not None:
        history["val_loss"] = []

    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_supervised = 0
        total_graph_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)

            preds, graph_emb = model(
                batch["x"],
                batch["edge_index"],
                batch["edge_weight"],
                batch["batch"],
                return_graph_emb=True,
            )

            mask = build_dynamic_mask(
                batch["target_valid_mask"],
                keep_fraction=keep_fraction,
                generator=generator,
            )
            desc_loss = masked_mse_loss(preds, batch["targets"], mask)
            graph_loss = preds.sum() * 0.0
            if graph_obj_weight > 0.0:
                _, graph_emb_view2 = model(
                    batch["x"],
                    batch["edge_index"],
                    batch["edge_weight"],
                    batch["batch"],
                    return_graph_emb=True,
                )
                graph_loss = nt_xent_graph_loss(
                    graph_emb,
                    graph_emb_view2,
                    temperature=graph_obj_temperature,
                )
            loss = desc_loss + (graph_obj_weight * graph_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            supervised_count = int(mask.sum().item())
            total_loss += float(desc_loss.item()) * max(supervised_count, 1)
            total_supervised += supervised_count
            batch_size = int(batch["targets"].shape[0])
            total_graph_loss += float(graph_loss.item()) * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_supervised, 1)
        history["train_loss"].append(train_loss)
        if graph_obj_weight > 0.0:
            avg_graph_loss = total_graph_loss / max(total_samples, 1)
            history["train_graph_loss"].append(avg_graph_loss)
            history["train_total_loss"].append(train_loss + graph_obj_weight * avg_graph_loss)

        if val_loader is not None:
            val_loss = evaluate_descriptor_pretraining(
                model,
                val_loader,
                device=device,
                keep_fraction=keep_fraction,
                generator=generator,
            )
            history["val_loss"].append(val_loss)

        if verbose:
            if val_loader is not None:
                msg = (
                    f"Epoch {epoch:03d} | train_masked_mse={train_loss:.6f} "
                    f"| val_masked_mse={history['val_loss'][-1]:.6f}"
                )
                if graph_obj_weight > 0.0:
                    msg += (
                        f" | train_graph_ntxent={history['train_graph_loss'][-1]:.6f}"
                        f" | train_total={history['train_total_loss'][-1]:.6f}"
                    )
                print(msg)
            else:
                msg = f"Epoch {epoch:03d} | train_masked_mse={train_loss:.6f}"
                if graph_obj_weight > 0.0:
                    msg += (
                        f" | train_graph_ntxent={history['train_graph_loss'][-1]:.6f}"
                        f" | train_total={history['train_total_loss'][-1]:.6f}"
                    )
                print(msg)

    return history


def evaluate_descriptor_pretraining(
    model: CheMeleonChebPretrainer,
    data_loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    keep_fraction: float = DEFAULT_KEEP_FRACTION,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Evaluate masked MSE on a validation loader."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_loss = 0.0
    total_supervised = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            preds = model(
                batch["x"],
                batch["edge_index"],
                batch["edge_weight"],
                batch["batch"],
            )
            mask = build_dynamic_mask(
                batch["target_valid_mask"],
                keep_fraction=keep_fraction,
                generator=generator,
            )
            loss = masked_mse_loss(preds, batch["targets"], mask)

            supervised_count = int(mask.sum().item())
            total_loss += float(loss.item()) * max(supervised_count, 1)
            total_supervised += supervised_count

    return total_loss / max(total_supervised, 1)


def save_pretraining_artifacts(
    out_dir: str,
    data: DescriptorPretrainingData,
) -> None:
    """Save prepared targets/stats for reproducible and faster reruns."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "targets.npy", data.targets)
    np.save(out / "target_valid_mask.npy", data.target_valid_mask)
    np.save(out / "selected_indices.npy", data.stats.selected_indices)
    np.save(out / "mean.npy", data.stats.mean)
    np.save(out / "std.npy", data.stats.std)

    with open(out / "smiles.txt", "w", encoding="utf-8") as f:
        for smi in data.smiles:
            f.write(f"{smi}\n")

    with open(out / "descriptor_names.txt", "w", encoding="utf-8") as f:
        for name in data.stats.feature_names:
            f.write(f"{name}\n")


def load_pretraining_artifacts(in_dir: str) -> DescriptorPretrainingData:
    """Load prepared descriptor targets and preprocessing stats."""
    src = Path(in_dir)

    targets = np.load(src / "targets.npy")
    target_valid_mask = np.load(src / "target_valid_mask.npy")
    selected_indices = np.load(src / "selected_indices.npy")
    mean = np.load(src / "mean.npy")
    std = np.load(src / "std.npy")

    with open(src / "smiles.txt", "r", encoding="utf-8") as f:
        smiles = [line.strip() for line in f if line.strip()]

    with open(src / "descriptor_names.txt", "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    stats = DescriptorTargetStats(
        feature_names=feature_names,
        mean=mean.astype(np.float32, copy=False),
        std=std.astype(np.float32, copy=False),
        selected_indices=selected_indices.astype(np.int64, copy=False),
    )

    return DescriptorPretrainingData(
        smiles=smiles,
        targets=targets.astype(np.float32, copy=False),
        target_valid_mask=target_valid_mask.astype(bool, copy=False),
        stats=stats,
    )
