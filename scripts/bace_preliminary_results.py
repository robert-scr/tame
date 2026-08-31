"""Preliminary BACE benchmark + diagnosis: ChebNet baseline vs TAME vs TAME-Fusion.

For each model we train on the BACE scaffold split (identical to the HPO scripts), evaluate on the
held-out test set and record ROC-AUC, PR-AUC and Macro-F1 per seed. TAME and TAME-Fusion use the best
hyperparameters found by ``scripts/tame_hpo_v3.py`` / ``scripts/tame_fusion_hpo.py`` and are initialised
from the respective pretrained ChebNet checkpoint.

To diagnose why the graph-only ChebNet can match/beat the fused models, this also runs a *fair*
control chain:

    ChebNet (generic recipe)            graph-only, lr 1e-3 full fine-tune  (original baseline)
    TAME (graph-only) / Fusion (graph-only)   graph-only, the model's OWN HPO recipe
    TAME / TAME-Fusion                  the full multimodal models

The step generic-baseline -> graph-only -> full isolates, in order, the training regime and then the
contribution of the extra modalities.

Per-element mixing is enforced via a non-zero ``--gate_entropy_weight`` (overrides the HPO value), so no
hidden element is ever fully owned by a single modality. Gate statistics (mean per-expert weight, mean
per-position entropy and the mean smallest-expert weight) are recorded to verify this and to see whether
the graph expert is being down-weighted.

Outputs (in ``--out_dir``):
    bace_preliminary_results.csv   long-form per-seed metrics + gate stats
    bace_preliminary_meta.json     resolved configs, recipes, checkpoints, split sizes
    bace_gate_sweep.csv            (with --gate_sweep) TAME gate-target sweep
"""

import os
import sys
import io
import csv
import json
import copy
import argparse
from dataclasses import asdict
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.tame_predictor import TAMEPredictor, TAMEPredictorConfig
from models.tame_fusion_predictor import TAMEFusionPredictor, TAMEFusionPredictorConfig
from models.core.graph_encoder import ChebNetEncoder
from models.core.pooling import create_pooling
from utils.embedding_cache import EfficientEmbeddingCache
from utils.molecular_graph import smiles_to_graph
from utils.batched_mol_graph import batch_graphs

RDLogger.DisableLog("rdApp.*")

#SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
SEEDS = np.random.default_rng(42).choice(range(1, 1000), size=100, replace=False).tolist()  # for random seed selection

CSV_FIELDS = [
    "Model", "Seed", "Test_ROC_AUC", "Test_PR_AUC", "Test_Macro_F1", "Val_ROC_AUC",
    "Gate_Graph", "Gate_Text", "Gate_Desc", "Gate_SEG",
    "Gate_Entropy_Mean", "Gate_Min_Mean", "Stop_Epoch",
]


# ---------------------------------------------------------------------------
# Helpers (mirror the HPO scripts)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rdkit_descriptors(smiles_list):
    desc_names = [name for name, _ in Descriptors.descList]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.full(len(desc_names), np.nan))
            continue
        all_desc = Descriptors.CalcMolDescriptors(mol)
        rows.append([all_desc.get(name, np.nan) for name in desc_names])
    return np.asarray(rows, dtype=np.float32)


def impute_with_train_median(train_X, valid_X, test_X):
    med = np.nanmedian(np.asarray(train_X, dtype=np.float32), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    def _imp(X):
        X = np.asarray(X, dtype=np.float32)
        mask = ~np.isfinite(X)
        if mask.any():
            X = X.copy()
            X[mask] = np.take(med, np.where(mask)[1])
        return X

    return _imp(train_X), _imp(valid_X), _imp(test_X)


def extract_encoder_state(ckpt_path):
    """Return the encoder state dict with the ``encoder.`` prefix stripped (matches HPO scripts)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("encoder_state", state.get("model_state_dict", state))
    filtered = {
        k.replace("encoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    if not filtered:
        filtered = state_dict
    return filtered


def _classification_metrics(y_true, probs):
    obs = np.isfinite(y_true) & np.isfinite(probs)
    y, p = y_true[obs], probs[obs]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    roc = float(roc_auc_score(y, p))
    pr = float(average_precision_score(y, p))
    f1 = float(f1_score(y, (p >= 0.5).astype(int), average="macro"))
    return roc, pr, f1


def _blank_row(model, seed):
    """Row template with all gate/entropy columns set to NaN."""
    row = {k: float("nan") for k in CSV_FIELDS}
    row["Model"] = model
    row["Seed"] = int(seed)
    return row


# ---------------------------------------------------------------------------
# Config builders (mirror the two HPO objective() functions exactly)
# ---------------------------------------------------------------------------
def build_tame_config(params, text_dim, desc_dim):
    pool = params.get("pool_type", "sum")
    gate_graph = float(params["gate_target_graph"])
    gate_text = float(params["gate_target_text"])
    gate_desc = max(1.0 - gate_graph - gate_text, 0.0)
    return TAMEPredictorConfig(
        task="classification",
        num_tasks=1,
        hidden_channels=int(params["hidden_channels"]),
        K=int(params["K"]),
        num_layers=int(params["num_layers"]),
        pool=pool,
        set2set_processing_steps=int(params.get("set2set_n_iter", 4)),
        text_embedding_dim=int(text_dim),
        descriptor_dim=int(desc_dim),
        fusion_hidden_dim=int(params["proj_dim"]),
        projection_dropout=float(params["projection_dropout"]),
        router_dropout=float(params["router_dropout"]),
        head_hidden_dim=int(params["head_hidden_dim"]),
        head_dropout=float(params["head_dropout"]),
        gate_balance_weight=float(params["gate_balance_weight"]),
        gate_entropy_weight=float(params.get("gate_entropy_weight", 0.0)),
        desc_modality_dropout=float(params["desc_modality_dropout"]),
        gate_target=(gate_graph, gate_text, gate_desc),
        label_smoothing=float(params.get("label_smoothing", 0.0)),
        descriptor_standardize=True,
        descriptor_winsorize_lower_q=0.01,
        descriptor_winsorize_upper_q=0.99,
    )


def build_tame_config_zeroed(params, text_dim, desc_dim):
    """TAME with the router hard-forced to graph-only (text/desc gates exactly 0).

    Unlike the bare-ChebClassifier "TAME (graph-only)" control, this keeps the full
    architecture (encoder, projections, router, head) intact -- only the gate output
    is overridden, so text_proj/desc_proj still run but contribute nothing to fused_emb.
    gate_balance_weight/gate_entropy_weight are zeroed since both regularizers become
    dead/meaningless once the gate is hard-forced and non-differentiable w.r.t. the router.
    """
    config = build_tame_config(params, text_dim, desc_dim)
    config.force_graph_only = True
    config.gate_balance_weight = 0.0
    config.gate_entropy_weight = 0.0
    return config


def build_tame_config_scalar(params, text_dim, desc_dim):
    """TAME with a scalar (one value per expert per molecule) gate instead of the
    default per-hidden-dimension gate."""
    config = build_tame_config(params, text_dim, desc_dim)
    config.gate_mode = "scalar"
    return config


def build_fusion_config_scalar(params, text_dim):
    """TAME-Fusion with a scalar (one value per expert per molecule) gate instead of
    the default per-hidden-dimension gate."""
    config = build_fusion_config(params, text_dim)
    config.gate_mode = "scalar"
    return config


def build_fusion_config(params, text_dim):
    pool = params.get("pool_type", "sum")
    return TAMEFusionPredictorConfig(
        task="classification",
        num_tasks=1,
        hidden_channels=int(params["hidden_channels"]),
        K=int(params["K"]),
        num_layers=int(params["num_layers"]),
        pool=pool,
        set2set_processing_steps=int(params.get("set2set_n_iter", 4)),
        text_embedding_dim=int(text_dim),
        text_projection_dim=params.get("text_projection_dim", None),
        fusion=params.get("fusion", "cross_mha"),
        fusion_dim=int(params["fusion_dim"]),
        fusion_n_heads=int(params.get("fusion_n_heads", 4)),
        moe_hidden_dim=int(params["moe_hidden_dim"]),
        projection_dropout=float(params["projection_dropout"]),
        router_dropout=float(params["router_dropout"]),
        head_hidden_dim=int(params["head_hidden_dim"]),
        head_dropout=float(params["head_dropout"]),
        gate_balance_weight=float(params["gate_balance_weight"]),
        gate_entropy_weight=float(params.get("gate_entropy_weight", 0.0)),
        desc_modality_dropout=float(params["desc_modality_dropout"]),
        gate_target=float(params["gate_target_seg"]),
        label_smoothing=float(params.get("label_smoothing", 0.0)),
        descriptor_standardize=True,
        descriptor_winsorize_lower_q=0.01,
        descriptor_winsorize_upper_q=0.99,
    )


def backbone_of(params, ckpt_dir, epoch_tag):
    h = int(params["hidden_channels"])
    K = int(params["K"])
    L = int(params["num_layers"])
    pool = params.get("pool_type", "sum")
    s2s = int(params.get("set2set_n_iter", 4))
    ckpt_name = f"chebnet_pt_d{h}_K{K}_L{L}_e{epoch_tag}.pt"
    return {
        "hidden": h, "K": K, "L": L, "pool": pool, "set2set_n_iter": s2s,
        "ckpt_name": ckpt_name, "ckpt_path": os.path.join(ckpt_dir, ckpt_name),
    }


def recipe_of(params, *, lr=None, wd=None, batch_size=None, encoder_lr_mult=None, label_smoothing=None):
    """Optimizer recipe for the graph-only control; falls back to a model's HPO params if given."""
    p = params or {}
    return {
        "lr": float(lr if lr is not None else p["lr"]),
        "weight_decay": float(wd if wd is not None else p["weight_decay"]),
        "batch_size": int(batch_size if batch_size is not None else p["batch_size"]),
        "encoder_lr_mult": float(encoder_lr_mult if encoder_lr_mult is not None
                                 else p.get("encoder_lr_mult", 1.0)),
        "label_smoothing": float(label_smoothing if label_smoothing is not None
                                 else p.get("label_smoothing", 0.0)),
    }


# ---------------------------------------------------------------------------
# Graph-only model (pretrained ChebNet encoder + classification head)
# ---------------------------------------------------------------------------
class ChebClassifier(nn.Module):
    """ChebNet encoder + pooling + MLP head (same head recipe as cheb_bace_benchmark.py)."""

    def __init__(self, in_channels, hidden_channels, K, num_layers, pool, set2set_n_iter):
        super().__init__()
        self.encoder = ChebNetEncoder(
            in_channels=in_channels, hidden_channels=hidden_channels,
            K=K, num_layers=num_layers,
        )
        if pool == "set2set":
            self.pool = create_pooling("set2set", input_dim=hidden_channels, n_iters=set2set_n_iter)
        else:
            self.pool = create_pooling(pool, input_dim=hidden_channels)
        head_in = int(getattr(self.pool, "output_dim", hidden_channels))
        hidden_head = max(head_in // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_head),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_head, 1),
        )

    def forward(self, x, edge_index, edge_weight, batch):
        node = self.encoder(x, edge_index, edge_weight, batch)
        graph = self.pool(node, batch)
        return self.head(graph).view(-1)


class PyGChebClassifier(nn.Module):
    """PyG ChebConv baseline: same architecture as ChebClassifier but using
    torch_geometric.nn.ChebConv instead of the in-house ChebLayer.

    Not used by this script's ``main()`` directly -- torch-geometric isn't available
    on the cluster this benchmark normally runs on. Imported instead by
    ``scripts/pyg_chebnet_hpo_bace.py``, which runs the PyG baseline (HPO + 100 seeds)
    standalone on a machine that has torch-geometric installed and writes a CSV with
    the same schema, which the plotting notebook then concatenates in.
    """

    def __init__(self, in_channels, hidden_channels, K, num_layers, pool, set2set_n_iter):
        super().__init__()
        from torch_geometric.nn import ChebConv

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.convs.append(ChebConv(in_c, hidden_channels, K=K))

        self.dropout = 0.1
        self.register_buffer("lambda_max", torch.tensor([2.0]))

        if pool == "set2set":
            self.pool = create_pooling("set2set", input_dim=hidden_channels, n_iters=set2set_n_iter)
        else:
            self.pool = create_pooling(pool, input_dim=hidden_channels)
        head_in = int(getattr(self.pool, "output_dim", hidden_channels))
        hidden_head = max(head_in // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_head),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_head, 1),
        )

    def forward(self, x, edge_index, edge_weight, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight, batch=batch, lambda_max=self.lambda_max)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        graph = self.pool(h, batch)
        return self.head(graph).view(-1)


def graphs_to_tensors(bg, device):
    x = torch.from_numpy(bg.X.astype(np.float32)).to(device)
    edge_index = torch.from_numpy(bg.edge_index.astype(np.int64)).to(device)
    edge_weight = torch.from_numpy(bg.edge_weight.astype(np.float32)).to(device)
    batch = torch.from_numpy(bg.batch.astype(np.int64)).to(device)
    return x, edge_index, edge_weight, batch


def _graph_batches(graphs, y, idx, batch_size, shuffle, rng):
    order = list(idx)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        sub = [(graphs[i], y[i]) for i in chunk if graphs[i] is not None]
        if not sub:
            continue
        gs, ys = zip(*sub)
        yield batch_graphs(list(gs)), np.asarray(ys, dtype=np.float32)


@torch.no_grad()
def _graph_predict(model, graphs, device, batch_size=128):
    """Return sigmoid probabilities aligned to the order of ``graphs`` (NaN for invalid)."""
    model.eval()
    probs = np.full(len(graphs), np.nan, dtype=np.float32)
    valid = [(i, g) for i, g in enumerate(graphs) if g is not None]
    for start in range(0, len(valid), batch_size):
        chunk = valid[start:start + batch_size]
        positions = [i for i, _ in chunk]
        bg = batch_graphs([g for _, g in chunk])
        x, ei, ew, b = graphs_to_tensors(bg, device)
        p = torch.sigmoid(model(x, ei, ew, b)).cpu().numpy()
        for pos, val in zip(positions, p):
            probs[pos] = val
    return probs


def run_graph_only(bb, recipe, graphs, ys, device, *, max_epochs, patience, es_metric, seed):
    """Train a pretrained ChebNet (graph-only) on BACE with ``recipe``; return (test, val) probs, epoch."""
    set_seed(seed)
    train_graphs, val_graphs, test_graphs = graphs
    y_train, y_val, _ = ys

    in_channels = next(int(g.X.shape[1]) for g in train_graphs if g is not None)
    model = ChebClassifier(in_channels, bb["hidden"], bb["K"], bb["L"], bb["pool"],
                           bb["set2set_n_iter"]).to(device)

    enc_state = extract_encoder_state(bb["ckpt_path"])
    missing, _ = model.encoder.load_state_dict(enc_state, strict=False)
    if len(enc_state) and len(missing) == len(model.encoder.state_dict()):
        print(f"    [Warn] No encoder weights matched for {bb['ckpt_name']}")

    # Differential LR: encoder at lr * encoder_lr_mult, head/pool at lr (mirrors TAME fit()).
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc_params = [p for p in model.parameters() if id(p) in enc_ids]
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": recipe["lr"] * recipe["encoder_lr_mult"]},
         {"params": other_params, "lr": recipe["lr"]}],
        weight_decay=recipe["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    ls = recipe["label_smoothing"]
    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(train_graphs))

    best_metric = float("-inf") if es_metric == "val_auc" else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bg, yb in _graph_batches(train_graphs, y_train, train_idx, recipe["batch_size"], True, rng):
            x, ei, ew, b = graphs_to_tensors(bg, device)
            yt = torch.from_numpy(yb).to(device)
            if ls > 0.0:
                yt = yt * (1.0 - ls) + 0.5 * ls
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, ei, ew, b), yt)
            loss.backward()
            optimizer.step()

        val_probs = _graph_predict(model, val_graphs, device)
        val_roc, _, _ = _classification_metrics(y_val, val_probs)
        obs = np.isfinite(y_val) & np.isfinite(val_probs)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(val_probs[obs].clip(1e-7, 1 - 1e-7)),
            torch.from_numpy(y_val[obs]),
        )) if obs.any() else float("inf")

        metric = val_roc if es_metric == "val_auc" else val_loss
        is_better = (metric > best_metric) if es_metric == "val_auc" else (metric < best_metric)
        if np.isfinite(metric) and is_better:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_probs = _graph_predict(model, test_graphs, device)
    val_probs = _graph_predict(model, val_graphs, device)
    return test_probs, val_probs, best_epoch


def run_pyg_graph_only(bb, recipe, graphs, ys, device, *, max_epochs, patience, es_metric, seed):
    """Train PyG ChebConv (from scratch) on BACE with ``recipe``; return (test, val) probs, epoch."""
    set_seed(seed)
    train_graphs, val_graphs, test_graphs = graphs
    y_train, y_val, _ = ys

    in_channels = next(int(g.X.shape[1]) for g in train_graphs if g is not None)
    model = PyGChebClassifier(in_channels, bb["hidden"], bb["K"], bb["L"], bb["pool"],
                              bb["set2set_n_iter"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    ls = recipe["label_smoothing"]
    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(train_graphs))

    best_metric = float("-inf") if es_metric == "val_auc" else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bg, yb in _graph_batches(train_graphs, y_train, train_idx, recipe["batch_size"], True, rng):
            x, ei, ew, b = graphs_to_tensors(bg, device)
            yt = torch.from_numpy(yb).to(device)
            if ls > 0.0:
                yt = yt * (1.0 - ls) + 0.5 * ls
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, ei, ew, b), yt)
            loss.backward()
            optimizer.step()

        val_probs = _graph_predict(model, val_graphs, device)
        val_roc, _, _ = _classification_metrics(y_val, val_probs)
        obs = np.isfinite(y_val) & np.isfinite(val_probs)
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(val_probs[obs].clip(1e-7, 1 - 1e-7)),
            torch.from_numpy(y_val[obs]),
        )) if obs.any() else float("inf")

        metric = val_roc if es_metric == "val_auc" else val_loss
        is_better = (metric > best_metric) if es_metric == "val_auc" else (metric < best_metric)
        if np.isfinite(metric) and is_better:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_probs = _graph_predict(model, test_graphs, device)
    val_probs = _graph_predict(model, val_graphs, device)
    return test_probs, val_probs, best_epoch


# ---------------------------------------------------------------------------
# Gate statistics for the full models (verify per-element mixing)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_gate_stats(model, val_smiles, text_val, desc_val, device, batch_size=128):
    """Return (entropy_mean, min_gate_mean, per_expert_means) over the val set.

    entropy_mean  mean per-position Shannon entropy of the gates (max = log K)
    min_gate_mean mean over positions of the smallest expert weight (floor contribution)
    per_expert    K-dim mean gate weight per expert
    """
    model._set_eval_mode()
    graphs = model._precompute_graphs(val_smiles, verbose=False)[0]
    ent_acc, min_acc = [], []
    gate_sum, pos_count = None, 0
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        valid = [(g, i) for i, g in enumerate(chunk) if g is not None]
        if not valid:
            continue
        gs, idx = zip(*valid)
        x, ei, ew, b = model._to_tensors(batch_graphs(list(gs)))
        abs_idx = [start + i for i in idx]
        text_t = torch.from_numpy(text_val[abs_idx].astype(np.float32)).to(device)
        desc_np = model._transform_descriptors(desc_val[abs_idx].astype(np.float32))
        desc_t = torch.from_numpy(desc_np).to(device)
        _, aux = model._forward(x, ei, ew, b, text_t, desc_t)
        gates = aux["gates_full"]  # (B, H, K)
        eps = 1e-8
        ent = -(gates * (gates + eps).log()).sum(dim=-1)
        gmin = gates.min(dim=-1).values
        ent_acc.append(ent.reshape(-1).cpu().numpy())
        min_acc.append(gmin.reshape(-1).cpu().numpy())
        s = gates.sum(dim=(0, 1)).cpu().numpy()
        gate_sum = s if gate_sum is None else gate_sum + s
        pos_count += gates.shape[0] * gates.shape[1]
    if gate_sum is None:
        return float("nan"), float("nan"), None
    return (float(np.concatenate(ent_acc).mean()),
            float(np.concatenate(min_acc).mean()),
            (gate_sum / pos_count).tolist())


def _best_epoch(history, es_metric):
    if es_metric == "val_auc" and history.get("val_auc"):
        return int(np.argmax(history["val_auc"]))
    return int(np.argmin(history["val_loss"]))


def fit_full_model(kind, config, params, enc_state, data, device, *, max_epochs, patience, es_metric, seed):
    """Train a full TAME / TAME-Fusion model; return (test_probs, val_probs, history)."""
    set_seed(seed)
    model = (TAMEPredictor(config=config, device=str(device)) if kind == "tame"
             else TAMEFusionPredictor(config=config, device=str(device)))
    history = model.fit(
        smiles_list=data["train_smiles"], labels=data["y_train"],
        val_smiles=data["val_smiles"], val_labels=data["y_val"],
        text_embeddings=data["text_train"], val_text_embeddings=data["text_val"],
        descriptor_features=data["desc_train"], val_descriptor_features=data["desc_val"],
        num_epochs=max_epochs, batch_size=int(params["batch_size"]),
        learning_rate=float(params["lr"]), weight_decay=float(params["weight_decay"]),
        encoder_lr_mult=float(params.get("encoder_lr_mult", 1.0)),
        patience=patience, early_stopping_metric=es_metric,
        verbose=False, seed=seed, init_encoder_state=enc_state,
    )
    test_probs = model.predict_batch(
        data["test_smiles"], text_embeddings=data["text_test"],
        descriptor_features=data["desc_test"], batch_size=128).flatten()
    val_probs = model.predict_batch(
        data["val_smiles"], text_embeddings=data["text_val"],
        descriptor_features=data["desc_val"], batch_size=128).flatten()
    ent, gmin, gate_means = compute_gate_stats(
        model, data["val_smiles"], data["text_val"], data["desc_val"], device)
    return test_probs, val_probs, history, (ent, gmin, gate_means)


# ---------------------------------------------------------------------------
# Gate-target sweep (TAME only)
# ---------------------------------------------------------------------------
def parse_gate_targets(spec, hpo_triple):
    if not spec:
        return [hpo_triple, (0.34, 0.33, 0.33), (0.7, 0.2, 0.1), (0.8, 0.1, 0.1)]
    out = []
    for triple in spec.split(","):
        g, t, d = (float(v) for v in triple.split("/"))
        out.append((g, t, d))
    return out


def run_gate_sweep(tame_params, tame_bb, enc_state, data, device, *, gate_targets, sweep_seeds,
                   entropy_weight, max_epochs, patience, es_metric, out_dir):
    print("\n=== Gate-target sweep (TAME only) ===")
    print("    NOTE: TAME-Fusion's gate is SEG-vs-descriptor (graph is bundled inside SEG), so the "
          "sweep over the graph expert applies to TAME only.")
    text_dim, desc_dim = data["text_dim"], data["desc_dim"]
    seeds = SEEDS[:sweep_seeds]
    rows = []
    for (gg, gt, gd) in gate_targets:
        params = {**tame_params, "gate_target_graph": gg, "gate_target_text": gt,
                  "gate_entropy_weight": entropy_weight}
        config = build_tame_config(params, text_dim, desc_dim)
        print(f"\n  gate_target=(graph={gg:.2f}, text={gt:.2f}, desc={gd:.2f}) "
              f"entropy_w={entropy_weight}")
        for seed in seeds:
            test_probs, val_probs, history, (ent, gmin, gmeans) = fit_full_model(
                "tame", config, params, enc_state, data, device,
                max_epochs=max_epochs, patience=patience, es_metric=es_metric, seed=seed)
            val_roc, _, _ = _classification_metrics(data["y_val"], val_probs)
            test_roc, _, _ = _classification_metrics(data["y_test"], test_probs)
            gm = gmeans or [float("nan")] * 3
            print(f"    seed {seed}: val={val_roc:.4f} test={test_roc:.4f} "
                  f"gates(g/t/d)={gm[0]:.2f}/{gm[1]:.2f}/{gm[2]:.2f} ent={ent:.3f}")
            rows.append({
                "gate_graph_target": gg, "gate_text_target": gt, "gate_desc_target": gd,
                "Seed": int(seed), "Val_ROC_AUC": val_roc, "Test_ROC_AUC": test_roc,
                "Gate_Graph": gm[0], "Gate_Text": gm[1], "Gate_Desc": gm[2],
                "Gate_Entropy_Mean": ent,
            })
    fields = ["gate_graph_target", "gate_text_target", "gate_desc_target", "Seed",
              "Val_ROC_AUC", "Test_ROC_AUC", "Gate_Graph", "Gate_Text", "Gate_Desc", "Gate_Entropy_Mean"]
    path = os.path.join(out_dir, "bace_gate_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved gate sweep ({len(rows)} rows) to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preliminary BACE results + diagnosis")
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory with pretrained ChebNet checkpoints")
    parser.add_argument("--tame_params_json", type=str, required=True, help="tame_hpo_v3.py best-params JSON")
    parser.add_argument("--fusion_params_json", type=str, required=True, help="tame_fusion_hpo.py best-params JSON")
    parser.add_argument("--ckpt_epoch_tag", type=int, default=25, help="The e{N} epoch suffix of the checkpoints")
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es_metric", type=str, choices=["val_loss", "val_auc"], default="val_loss")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--gate_entropy_weight", type=float, default=0.05,
                        help="Per-position gate entropy weight; OVERRIDES the HPO value to enforce "
                             "per-element mixing (every modality contributes a little). 0 = off.")
    parser.add_argument("--cheb_lr", type=float, default=1e-3, help="Generic baseline learning rate")
    parser.add_argument("--cheb_wd", type=float, default=1e-4, help="Generic baseline weight decay")
    parser.add_argument("--cheb_batch_size", type=int, default=64, help="Generic baseline batch size")
    parser.add_argument("--zeroed_ablation", action="store_true",
                        help="Also run TAME with the router hard-forced to graph-only "
                             "(full architecture, text/desc gates exactly 0)")
    parser.add_argument("--scalar_gate_ablation", action="store_true",
                        help="Also run TAME and TAME-Fusion with a scalar (per-molecule) "
                             "gate instead of the per-hidden-dimension gate")
    parser.add_argument("--gate_sweep", action="store_true", help="Also run the TAME gate-target sweep")
    parser.add_argument("--sweep_seeds", type=int, default=5, help="Number of seeds for the gate sweep")
    parser.add_argument("--gate_targets_list", type=str, default=None,
                        help="Comma-separated graph/text/desc triples, e.g. '0.7/0.2/0.1,0.8/0.1/0.1'. "
                             "Default: HPO-chosen, balanced, graph-heavy, graph-dominant.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Info] Device: {device}")
    print(f"[Info] Enforcing gate_entropy_weight={args.gate_entropy_weight} on the full models "
          f"(per-element mixing).")

    with open(args.tame_params_json) as f:
        tame_params = json.load(f)["params"]
    with open(args.fusion_params_json) as f:
        fusion_params = json.load(f)["params"]
    # Enforce non-zero entropy so no hidden element is ever fully owned by one modality.
    tame_params = {**tame_params, "gate_entropy_weight": args.gate_entropy_weight}
    fusion_params = {**fusion_params, "gate_entropy_weight": args.gate_entropy_weight}

    # ---- Load BACE (identical split to the HPO scripts) ----
    print(f"\n[Info] Loading BACE ({args.splitter.title()} Split)...")
    dc_cache_dir = os.path.join(project_root, "cache", "deepchem_data")
    os.makedirs(dc_cache_dir, exist_ok=True)
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        _, datasets, _ = dc.molnet.load_bace_classification(
            featurizer="ECFP", splitter=args.splitter,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1,
            data_dir=dc_cache_dir, save_dir=dc_cache_dir,
        )
    train_dc, valid_dc, test_dc = datasets
    train_smiles, val_smiles, test_smiles = list(train_dc.ids), list(valid_dc.ids), list(test_dc.ids)
    y_train = train_dc.y.astype(np.float32).reshape(-1)
    y_val = valid_dc.y.astype(np.float32).reshape(-1)
    y_test = test_dc.y.astype(np.float32).reshape(-1)
    print(f"  train={len(train_smiles)} val={len(val_smiles)} test={len(test_smiles)}")

    # ---- Text embeddings ----
    emb_path = os.path.join(project_root, "cache", "cot_embeddings",
                            "binding_fast_text_embeddings_compact.npz")
    print(f"[Info] Loading text embeddings from {emb_path}...")
    seg_cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    text_train = np.asarray(seg_cache.get_batch(train_smiles), dtype=np.float32)
    text_val = np.asarray(seg_cache.get_batch(val_smiles), dtype=np.float32)
    text_test = np.asarray(seg_cache.get_batch(test_smiles), dtype=np.float32)

    # ---- RDKit descriptors ----
    print("[Info] Computing RDKit descriptors...")
    desc_train_raw = compute_rdkit_descriptors(train_smiles)
    desc_val_raw = compute_rdkit_descriptors(val_smiles)
    desc_test_raw = compute_rdkit_descriptors(test_smiles)
    desc_train, desc_val, desc_test = impute_with_train_median(desc_train_raw, desc_val_raw, desc_test_raw)

    # ---- Pre-build graphs for the graph-only models ----
    print("[Info] Building molecular graphs for the graph-only models...")
    def _graphs(smiles):
        out = []
        for smi in smiles:
            try:
                out.append(smiles_to_graph(smi))
            except Exception:
                out.append(None)
        return out
    base_graphs = (_graphs(train_smiles), _graphs(val_smiles), _graphs(test_smiles))
    base_ys = (y_train, y_val, y_test)

    text_dim = int(text_train.shape[1])
    desc_dim = int(desc_train.shape[1])
    data = {
        "train_smiles": train_smiles, "val_smiles": val_smiles, "test_smiles": test_smiles,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "text_train": text_train, "text_val": text_val, "text_test": text_test,
        "desc_train": desc_train, "desc_val": desc_val, "desc_test": desc_test,
        "text_dim": text_dim, "desc_dim": desc_dim,
    }

    # ---- Resolve backbones / checkpoints, dedup generic baselines ----
    tame_bb = backbone_of(tame_params, args.ckpt_dir, args.ckpt_epoch_tag)
    fusion_bb = backbone_of(fusion_params, args.ckpt_dir, args.ckpt_epoch_tag)
    bb_key = lambda bb: (bb["hidden"], bb["K"], bb["L"], bb["pool"], bb["set2set_n_iter"], bb["ckpt_path"])
    same_backbone = bb_key(tame_bb) == bb_key(fusion_bb)
    if same_backbone:
        baseline_specs = [("ChebNet", tame_bb)]
    else:
        baseline_specs = [("ChebNet (TAME)", tame_bb), ("ChebNet (Fusion)", fusion_bb)]

    for bb in (tame_bb, fusion_bb):
        if not os.path.exists(bb["ckpt_path"]):
            raise FileNotFoundError(f"Checkpoint not found: {bb['ckpt_path']}")

    generic_recipe = recipe_of(None, lr=args.cheb_lr, wd=args.cheb_wd,
                               batch_size=args.cheb_batch_size, encoder_lr_mult=1.0, label_smoothing=0.0)
    tame_recipe = recipe_of(tame_params)
    fusion_recipe = recipe_of(fusion_params)

    rows = []

    def _run_graph_only_model(label, bb, recipe):
        print(f"\n=== {label} (d{bb['hidden']} K{bb['K']} L{bb['L']} {bb['pool']}) ===")
        print(f"    recipe: {recipe}")
        for seed in SEEDS:
            test_probs, val_probs, epoch = run_graph_only(
                bb, recipe, base_graphs, base_ys, device,
                max_epochs=args.max_epochs, patience=args.patience,
                es_metric=args.es_metric, seed=seed)
            roc, pr, f1 = _classification_metrics(y_test, test_probs)
            val_roc, _, _ = _classification_metrics(y_val, val_probs)
            print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} Macro-F1={f1:.4f} (ep {epoch})")
            row = _blank_row(label, seed)
            row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1,
                        "Val_ROC_AUC": val_roc, "Stop_Epoch": epoch})
            rows.append(row)

    # ---- (1) generic-recipe ChebNet baseline(s) ----
    for label, bb in baseline_specs:
        _run_graph_only_model(label, bb, generic_recipe)

    # ---- (2) graph-only control under each model's OWN recipe ----
    _run_graph_only_model("TAME (graph-only)", tame_bb, tame_recipe)
    if same_backbone and tame_recipe == fusion_recipe:
        print("\n[Info] Fusion graph-only control identical to TAME graph-only (same backbone+recipe); skipping.")
    else:
        _run_graph_only_model("TAME-Fusion (graph-only)", fusion_bb, fusion_recipe)

    # ---- (3) full multimodal models ----
    full_models = [
        ("TAME", "tame", tame_params, tame_bb,
         build_tame_config(tame_params, text_dim, desc_dim)),
        ("TAME-Fusion", "fusion", fusion_params, fusion_bb,
         build_fusion_config(fusion_params, text_dim)),
    ]
    if args.zeroed_ablation:
        full_models.append(
            ("TAME (zeroed)", "tame", tame_params, tame_bb,
             build_tame_config_zeroed(tame_params, text_dim, desc_dim))
        )
    if args.scalar_gate_ablation:
        full_models.append(
            ("TAME (scalar gate)", "tame", tame_params, tame_bb,
             build_tame_config_scalar(tame_params, text_dim, desc_dim))
        )
        full_models.append(
            ("TAME-Fusion (scalar gate)", "fusion", fusion_params, fusion_bb,
             build_fusion_config_scalar(fusion_params, text_dim))
        )
    for label, kind, params, bb, config in full_models:
        print(f"\n=== {label} ===")
        print(f"    config: {asdict(config)}")
        print(f"    checkpoint: {bb['ckpt_name']}")
        enc_state = extract_encoder_state(bb["ckpt_path"])
        for seed in SEEDS:
            test_probs, val_probs, history, (ent, gmin, gmeans) = fit_full_model(
                kind, config, params, enc_state, data, device,
                max_epochs=args.max_epochs, patience=args.patience,
                es_metric=args.es_metric, seed=seed)
            roc, pr, f1 = _classification_metrics(y_test, test_probs)
            val_roc, _, _ = _classification_metrics(y_val, val_probs)
            stop_epoch = len(history["val_loss"])
            row = _blank_row(label, seed)
            row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1,
                        "Val_ROC_AUC": val_roc, "Stop_Epoch": stop_epoch,
                        "Gate_Entropy_Mean": ent, "Gate_Min_Mean": gmin})
            if gmeans is not None:
                if kind == "tame":
                    row.update({"Gate_Graph": gmeans[0], "Gate_Text": gmeans[1], "Gate_Desc": gmeans[2]})
                else:
                    row.update({"Gate_SEG": gmeans[0], "Gate_Desc": gmeans[1]})
            gm_str = ("/".join(f"{v:.2f}" for v in gmeans) if gmeans is not None else "n/a")
            print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} Macro-F1={f1:.4f} "
                  f"| gates={gm_str} ent={ent:.3f} min={gmin:.3f}")
            rows.append(row)

    # ---- Save main results ----
    csv_path = os.path.join(args.out_dir, "bace_preliminary_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "splitter": args.splitter,
        "es_metric": args.es_metric,
        "gate_entropy_weight_enforced": args.gate_entropy_weight,
        "seeds": SEEDS,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "split_sizes": {"train": len(train_smiles), "val": len(val_smiles), "test": len(test_smiles)},
        "generic_baselines": [{"label": lbl, **bb} for lbl, bb in baseline_specs],
        "generic_recipe": generic_recipe,
        "tame": {"checkpoint": tame_bb["ckpt_name"], "recipe": tame_recipe,
                 "config": asdict(build_tame_config(tame_params, text_dim, desc_dim))},
        "tame_fusion": {"checkpoint": fusion_bb["ckpt_name"], "recipe": fusion_recipe,
                        "config": asdict(build_fusion_config(fusion_params, text_dim))},
        "f1_threshold": 0.5,
    }
    if args.zeroed_ablation:
        meta["tame_zeroed"] = {"checkpoint": tame_bb["ckpt_name"], "recipe": tame_recipe,
                               "config": asdict(build_tame_config_zeroed(tame_params, text_dim, desc_dim))}
    if args.scalar_gate_ablation:
        meta["tame_scalar_gate"] = {"checkpoint": tame_bb["ckpt_name"], "recipe": tame_recipe,
                                    "config": asdict(build_tame_config_scalar(tame_params, text_dim, desc_dim))}
        meta["tame_fusion_scalar_gate"] = {"checkpoint": fusion_bb["ckpt_name"], "recipe": fusion_recipe,
                                           "config": asdict(build_fusion_config_scalar(fusion_params, text_dim))}
    meta_path = os.path.join(args.out_dir, "bace_preliminary_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(rows)} rows to {csv_path}")
    print(f"Saved metadata to {meta_path}")

    # ---- Optional gate-target sweep (TAME) ----
    if args.gate_sweep:
        gg = float(tame_params["gate_target_graph"])
        gt = float(tame_params["gate_target_text"])
        gd = max(1.0 - gg - gt, 0.0)
        gate_targets = parse_gate_targets(args.gate_targets_list, (gg, gt, gd))
        enc_state = extract_encoder_state(tame_bb["ckpt_path"])
        run_gate_sweep(tame_params, tame_bb, enc_state, data, device,
                       gate_targets=gate_targets, sweep_seeds=args.sweep_seeds,
                       entropy_weight=args.gate_entropy_weight, max_epochs=args.max_epochs,
                       patience=args.patience, es_metric=args.es_metric, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
