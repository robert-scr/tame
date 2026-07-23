"""
Optuna-based hyperparameter optimization for TAME-Fusion on BACE.

Goals:
- Run locally on a single GPU for quick iteration.
- Resume and scale on cluster workers through shared Optuna storage.
- Enforce architecture-compatible pretraining checkpoint initialization.

Example (local):
    python benchmarking/hpo/tame_fusion_optuna_bace.py \
        --n-trials 30 \
        --device cuda \
        --trial-seeds 11

Example (cluster, multiple workers):
    python benchmarking/hpo/tame_fusion_optuna_bace.py \
        --study-name tame_fusion_bace \
        --storage postgresql://user:pass@host:5432/optuna \
        --n-trials 200 \
        --timeout 0 \
        --device cuda

Fester Backbone mit PT-S1:
python tame_fusion_optuna_bace.py --pretrain-stage stage1_node --n-trials 80 --num-epochs 120 --trial-seeds 11 --device cuda
Architektur frei suchen ohne PT:
python tame_fusion_optuna_bace.py --pretrain-stage none --search-arch --n-trials 120 --num-epochs 100 --trial-seeds 11 --device cuda
Top-k robust final prüfen:
python tame_fusion_optuna_bace.py --run-final-eval --final-top-k 5 --final-eval-seeds 11,22,33,44,55 --final-num-epochs 140 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import deepchem as dc
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys


def find_workspace_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "models").exists() and (p / "utils").exists():
            return p
    return here


WORKSPACE_ROOT = find_workspace_root()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from models.tame_fusion_predictor import TAMEFusionPredictor, TAMEFusionPredictorConfig
from utils.embedding_cache import EfficientEmbeddingCache


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


PRETRAIN_STAGE_CFG: Dict[str, Dict[str, str]] = {
    "stage1_node": {
        "alias": "cheb_foundation_pretrain_stage1.pt",
        "pattern_prefix": "cheb_foundation_stage1",
    },
}

ARCH_KEY_PATTERN = re.compile(r"^h(?P<h>\d+)_L(?P<L>\d+)_K(?P<K>\d+)_pool-(?P<pool>.+)$")


@dataclass
class BACEDataBundle:
    train_smiles: List[str]
    valid_smiles: List[str]
    test_smiles: List[str]
    y_train_raw: np.ndarray
    y_valid_raw: np.ndarray
    y_test_raw: np.ndarray
    w_train: np.ndarray
    w_valid: np.ndarray
    w_test: np.ndarray
    train_emb: np.ndarray
    valid_emb: np.ndarray
    test_emb: np.ndarray
    text_dim: int
    rd_train: np.ndarray
    rd_valid: np.ndarray
    rd_test: np.ndarray


def parse_int_list(raw: str) -> List[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(x) for x in values]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_1d(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def classification_metrics(
    y_true: Sequence[float],
    y_proba: Sequence[float],
    threshold: float = 0.5,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    yt = _to_1d(y_true)
    yp = np.clip(_to_1d(y_proba), 0.0, 1.0)

    mask = np.isfinite(yt) & np.isfinite(yp)
    if weights is not None:
        w = _to_1d(weights)
        mask = mask & (w > 0)

    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"roc_auc": float("nan"), "accuracy": float("nan"), "f1": float("nan"), "n_eval": 0}

    yhat = (yp >= float(threshold)).astype(np.float32)
    if np.unique(yt).size >= 2:
        roc_auc = float(roc_auc_score(yt, yp))
    else:
        roc_auc = float("nan")

    acc = float(accuracy_score(yt, yhat))
    f1 = float(f1_score(yt, yhat, zero_division=0))
    return {"roc_auc": roc_auc, "accuracy": acc, "f1": f1, "n_eval": int(mask.sum())}


def find_best_threshold(
    y_true: Sequence[float],
    y_proba: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    yt = _to_1d(y_true)
    yp = np.clip(_to_1d(y_proba), 0.0, 1.0)

    mask = np.isfinite(yt) & np.isfinite(yp)
    if weights is not None:
        w = _to_1d(weights)
        mask = mask & (w > 0)

    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 91, dtype=np.float32)
    best_thr = 0.5
    best_f1 = -1.0
    for t in thresholds:
        yhat = (yp >= float(t)).astype(np.float32)
        f1 = float(f1_score(yt, yhat, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    return best_thr


def finite_mean(values: Sequence[float], default: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(arr)
    if not bool(mask.any()):
        return float(default)
    return float(np.mean(arr[mask]))


def rdkit_descriptor_matrix(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    desc_names = [name for name, _ in Descriptors.descList]
    n_desc = len(desc_names)

    rows: List[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.full((n_desc,), np.nan, dtype=np.float32))
            continue

        all_desc = Descriptors.CalcMolDescriptors(mol)
        row: List[float] = []
        for name in desc_names:
            try:
                row.append(float(all_desc.get(name, np.nan)))
            except Exception:
                row.append(np.nan)
        rows.append(np.asarray(row, dtype=np.float32))

    return np.vstack(rows).astype(np.float32), desc_names


def impute_with_train_median(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(np.asarray(train_x, dtype=np.float32), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    def _imp(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        mask = ~np.isfinite(arr)
        if mask.any():
            arr = arr.copy()
            arr[mask] = np.take(med, np.where(mask)[1])
        return arr

    return _imp(train_x), _imp(valid_x), _imp(test_x)


def _arch_from_encoder_state(encoder_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    layer_ids = sorted(
        {
            int(k.split(".")[1])
            for k in encoder_state.keys()
            if k.startswith("layers.") and k.endswith(".weight")
        }
    )
    if not layer_ids:
        raise RuntimeError("No encoder layer weights found in encoder_state.")

    first_weight_key = f"layers.{layer_ids[0]}.weight"
    first_weight = encoder_state[first_weight_key]
    if first_weight.ndim != 3:
        raise RuntimeError(
            f"Unexpected encoder weight shape for {first_weight_key}: {tuple(first_weight.shape)}"
        )

    return {
        "K": int(first_weight.shape[0]),
        "hidden_channels": int(first_weight.shape[-1]),
        "num_layers": int(max(layer_ids) + 1),
    }


def arch_to_key(hidden_channels: int, num_layers: int, K: int, pool: str) -> str:
    return f"h{int(hidden_channels)}_L{int(num_layers)}_K{int(K)}_pool-{str(pool)}"


def parse_arch_key(key: str) -> Dict[str, Any]:
    match = ARCH_KEY_PATTERN.match(str(key))
    if not match:
        raise ValueError(f"Invalid architecture key format: {key!r}")
    return {
        "hidden_channels": int(match.group("h")),
        "num_layers": int(match.group("L")),
        "K": int(match.group("K")),
        "pool": str(match.group("pool")),
    }


def discover_pretrain_architectures(workspace_root: Path, pretrain_stage: str) -> List[Dict[str, Any]]:
    if pretrain_stage not in PRETRAIN_STAGE_CFG:
        raise ValueError(f"Unsupported pretrain_stage={pretrain_stage!r}")

    ckpt_dir = workspace_root / "cache" / "chemeleon_pretraining"
    if not ckpt_dir.exists():
        return []

    pattern_prefix = PRETRAIN_STAGE_CFG[pretrain_stage]["pattern_prefix"]
    name_regex = re.compile(
        rf"^{re.escape(pattern_prefix)}__h(?P<h>\d+)__L(?P<L>\d+)__K(?P<K>\d+)__pool-(?P<pool>.+?)__.*\.pt$"
    )
    glob_pattern = f"{pattern_prefix}__h*__L*__K*__pool-*__*.pt"

    catalog: Dict[str, Dict[str, Any]] = {}
    for ckpt_path in ckpt_dir.glob(glob_pattern):
        if "__fallback__" in ckpt_path.name:
            continue

        match = name_regex.match(ckpt_path.name)
        if match is None:
            continue

        arch = {
            "hidden_channels": int(match.group("h")),
            "num_layers": int(match.group("L")),
            "K": int(match.group("K")),
            "pool": str(match.group("pool")),
        }
        arch_key = arch_to_key(**arch)
        if arch_key not in catalog:
            catalog[arch_key] = {
                **arch,
                "arch_key": arch_key,
                "checkpoint": ckpt_path.name,
            }

    return [catalog[k] for k in sorted(catalog.keys())]


def resolve_pretrain_checkpoint(
    workspace_root: Path,
    pretrain_stage: str,
    *,
    hidden_channels: int,
    num_layers: int,
    K: int,
    pool: str,
) -> Path:
    if pretrain_stage not in PRETRAIN_STAGE_CFG:
        raise ValueError(f"Unsupported pretrain_stage={pretrain_stage!r}")

    ckpt_dir = workspace_root / "cache" / "chemeleon_pretraining"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    pattern_prefix = PRETRAIN_STAGE_CFG[pretrain_stage]["pattern_prefix"]
    pattern = f"{pattern_prefix}__h{int(hidden_channels)}__L{int(num_layers)}__K{int(K)}__pool-{pool}__*.pt"

    candidates = sorted(
        [p for p in ckpt_dir.glob(pattern) if "__fallback__" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    alias = ckpt_dir / PRETRAIN_STAGE_CFG[pretrain_stage]["alias"]
    if alias.exists():
        raise RuntimeError(
            f"Config-specific checkpoint missing for stage={pretrain_stage!r}. "
            f"Found only alias {alias.name!r}, but required pattern {pattern!r}."
        )

    raise FileNotFoundError(
        f"No checkpoint found for stage={pretrain_stage!r} with required pattern={pattern!r}."
    )


def load_pretrained_encoder_state(
    ckpt_path: Path,
    expected_stage: str,
    expected_arch: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid checkpoint payload format.")

    stage = payload.get("pretraining_stage", None)
    if stage != expected_stage:
        raise RuntimeError(
            f"Expected checkpoint with pretraining_stage={expected_stage!r}, got {stage!r}"
        )

    model_state = payload.get("model_state_dict", {})
    enc_state = {
        k.replace("encoder.", "", 1): v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }
    if not enc_state:
        raise RuntimeError("Checkpoint does not contain encoder.* weights.")

    arch = _arch_from_encoder_state(enc_state)
    for key in ("hidden_channels", "K", "num_layers"):
        if int(arch[key]) != int(expected_arch[key]):
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {ckpt_path.name}: "
                f"{key}={arch[key]} but expected {expected_arch[key]}"
            )

    ckpt_cfg = payload.get("cheb_cfg", {}) if isinstance(payload.get("cheb_cfg", {}), dict) else {}
    ckpt_pool = ckpt_cfg.get("pool", None)
    if ckpt_pool is not None and str(ckpt_pool) != str(expected_arch["pool"]):
        raise RuntimeError(
            f"Checkpoint pool mismatch for {ckpt_path.name}: "
            f"pool={ckpt_pool!r} but expected {expected_arch['pool']!r}"
        )

    return enc_state


def load_bace_bundle(workspace_root: Path, embeddings_npz: Path) -> BACEDataBundle:
    tasks, datasets, _ = dc.molnet.load_bace_classification(featurizer="ECFP", splitter="scaffold")
    _ = tasks
    train_dc, valid_dc, test_dc = datasets

    train_smiles = list(train_dc.ids)
    valid_smiles = list(valid_dc.ids)
    test_smiles = list(test_dc.ids)

    y_train_raw = np.asarray(train_dc.y, dtype=np.float32).reshape(-1)
    y_valid_raw = np.asarray(valid_dc.y, dtype=np.float32).reshape(-1)
    y_test_raw = np.asarray(test_dc.y, dtype=np.float32).reshape(-1)

    w_train = np.asarray(train_dc.w, dtype=np.float32).reshape(-1)
    w_valid = np.asarray(valid_dc.w, dtype=np.float32).reshape(-1)
    w_test = np.asarray(test_dc.w, dtype=np.float32).reshape(-1)

    if not embeddings_npz.exists():
        raise FileNotFoundError(f"Embedding cache not found: {embeddings_npz}")

    seg_cache = EfficientEmbeddingCache.load(embeddings_npz, mmap_mode="r")
    all_smiles = train_smiles + valid_smiles + test_smiles
    all_emb = np.asarray(seg_cache.get_batch(all_smiles), dtype=np.float32)

    n_train = len(train_smiles)
    n_valid = len(valid_smiles)
    train_emb = all_emb[:n_train]
    valid_emb = all_emb[n_train:n_train + n_valid]
    test_emb = all_emb[n_train + n_valid :]
    text_dim = int(all_emb.shape[1])

    rd_train, _ = rdkit_descriptor_matrix(train_smiles)
    rd_valid, _ = rdkit_descriptor_matrix(valid_smiles)
    rd_test, _ = rdkit_descriptor_matrix(test_smiles)
    rd_train, rd_valid, rd_test = impute_with_train_median(rd_train, rd_valid, rd_test)

    return BACEDataBundle(
        train_smiles=train_smiles,
        valid_smiles=valid_smiles,
        test_smiles=test_smiles,
        y_train_raw=y_train_raw,
        y_valid_raw=y_valid_raw,
        y_test_raw=y_test_raw,
        w_train=w_train,
        w_valid=w_valid,
        w_test=w_test,
        train_emb=train_emb,
        valid_emb=valid_emb,
        test_emb=test_emb,
        text_dim=text_dim,
        rd_train=rd_train,
        rd_valid=rd_valid,
        rd_test=rd_test,
    )


def sample_hparams(
    trial: optuna.Trial,
    search_arch: bool,
    strict_arch_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if search_arch:
        if strict_arch_keys:
            arch_key = trial.suggest_categorical("encoder_arch_from_stage1_v1", strict_arch_keys)
            parsed = parse_arch_key(arch_key)
            hidden_channels = int(parsed["hidden_channels"])
            k_cheb = int(parsed["K"])
            num_layers = int(parsed["num_layers"])
            pool = str(parsed["pool"])
            set2set_steps = trial.suggest_categorical("set2set_processing_steps", [4, 6, 8])
        else:
            hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 160])
            k_cheb = trial.suggest_categorical("K", [2, 3, 4])
            num_layers = trial.suggest_categorical("num_layers", [2, 3, 4])
            pool = trial.suggest_categorical("pool", ["sum", "set2set"])
            set2set_steps = trial.suggest_categorical("set2set_processing_steps", [4, 6, 8])
    else:
        hidden_channels = 128
        k_cheb = 3
        num_layers = 3
        pool = "set2set"
        set2set_steps = 6

    fusion_pair = trial.suggest_categorical(
        "fusion_dim_heads_pair_bace_v1",
        ["32x4", "32x8", "64x4", "64x8", "96x4", "96x8", "128x4", "128x8"],
    )
    fusion_dim_str, fusion_n_heads_str = str(fusion_pair).split("x")
    fusion_dim = int(fusion_dim_str)
    fusion_n_heads = int(fusion_n_heads_str)

    params = {
        "hidden_channels": hidden_channels,
        "K": k_cheb,
        "num_layers": num_layers,
        "pool": pool,
        "set2set_processing_steps": set2set_steps,
        "fusion": "cross_mha",
        "fusion_dim": fusion_dim,
        "fusion_n_heads": fusion_n_heads,
        "moe_hidden_dim": trial.suggest_categorical("moe_hidden_dim", [32, 64, 96, 128]),
        "head_hidden_dim": trial.suggest_categorical("head_hidden_dim", [16, 32, 64, 128]),
        "fusion_dropout": trial.suggest_float("fusion_dropout", 0.10, 0.45, step=0.05),
        "projection_dropout": trial.suggest_float("projection_dropout", 0.10, 0.45, step=0.05),
        "router_dropout": trial.suggest_float("router_dropout", 0.10, 0.50, step=0.05),
        "head_dropout": trial.suggest_float("head_dropout", 0.10, 0.50, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "patience": trial.suggest_int("patience", 8, 25),
        "gate_balance_weight": trial.suggest_float("gate_balance_weight", 1e-4, 2e-2, log=True),
        "desc_modality_dropout": trial.suggest_float("desc_modality_dropout", 0.0, 0.35, step=0.05),
        "descriptor_winsorize_lower_q": trial.suggest_categorical(
            "descriptor_winsorize_lower_q", [0.0, 0.005, 0.01, 0.015, 0.02]
        ),
        "descriptor_winsorize_upper_q": trial.suggest_categorical(
            "descriptor_winsorize_upper_q", [0.98, 0.985, 0.99, 0.995, 1.0]
        ),
    }

    if float(params["descriptor_winsorize_lower_q"]) >= float(params["descriptor_winsorize_upper_q"]):
        raise optuna.TrialPruned("Invalid winsorization bounds.")

    return params


def canonicalize_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(raw_params)

    # Recover architecture fields from strict stage1 arch key when present.
    arch_key = params.get("encoder_arch_from_stage1_v1")
    if arch_key is not None:
        try:
            parsed = parse_arch_key(str(arch_key))
            params["hidden_channels"] = int(parsed["hidden_channels"])
            params["K"] = int(parsed["K"])
            params["num_layers"] = int(parsed["num_layers"])
            params["pool"] = str(parsed["pool"])
        except ValueError:
            # Keep backward compatibility for older studies with malformed keys.
            pass

    if "fusion_dim_heads_pair_bace_v1" in params:
        fusion_dim_str, fusion_n_heads_str = str(params["fusion_dim_heads_pair_bace_v1"]).split("x")
        params["fusion_dim"] = int(fusion_dim_str)
        params["fusion_n_heads"] = int(fusion_n_heads_str)
    if "fusion_dim_heads_pair_v4" in params:
        fusion_dim_str, fusion_n_heads_str = str(params["fusion_dim_heads_pair_v4"]).split("x")
        params["fusion_dim"] = int(fusion_dim_str)
        params["fusion_n_heads"] = int(fusion_n_heads_str)
    if "fusion_dim_heads_pair_v3" in params:
        fusion_dim_str, fusion_n_heads_str = str(params["fusion_dim_heads_pair_v3"]).split("x")
        params["fusion_dim"] = int(fusion_dim_str)
        params["fusion_n_heads"] = int(fusion_n_heads_str)
    if "fusion_n_heads" not in params and "fusion_n_heads_v2" in params:
        params["fusion_n_heads"] = int(params["fusion_n_heads_v2"])

    # Constants are not part of Optuna trial.params; inject defaults for final re-evaluation.
    params.setdefault("fusion", "cross_mha")
    params.setdefault("hidden_channels", 128)
    params.setdefault("K", 3)
    params.setdefault("num_layers", 3)
    params.setdefault("pool", "set2set")
    params.setdefault("set2set_processing_steps", 6)

    return params


def build_tame_fusion_config(text_dim: int, params: Dict[str, Any]) -> TAMEFusionPredictorConfig:
    return TAMEFusionPredictorConfig(
        task="classification",
        num_tasks=1,
        hidden_channels=int(params["hidden_channels"]),
        K=int(params["K"]),
        num_layers=int(params["num_layers"]),
        pool=str(params["pool"]),
        set2set_processing_steps=int(params["set2set_processing_steps"]),
        text_embedding_dim=int(text_dim),
        fusion=str(params["fusion"]),
        fusion_dim=int(params["fusion_dim"]),
        fusion_n_heads=int(params["fusion_n_heads"]),
        fusion_dropout=float(params["fusion_dropout"]),
        moe_hidden_dim=int(params["moe_hidden_dim"]),
        projection_dropout=float(params["projection_dropout"]),
        router_dropout=float(params["router_dropout"]),
        head_hidden_dim=int(params["head_hidden_dim"]),
        head_dropout=float(params["head_dropout"]),
        gate_balance_weight=float(params["gate_balance_weight"]),
        desc_modality_dropout=float(params["desc_modality_dropout"]),
        descriptor_winsorize_lower_q=float(params["descriptor_winsorize_lower_q"]),
        descriptor_winsorize_upper_q=float(params["descriptor_winsorize_upper_q"]),
        descriptor_standardize=True,
        add_hydrogens=False,
    )


def maybe_get_pretrained_state(
    pretrain_stage: str,
    params: Dict[str, Any],
    workspace_root: Path,
    cache: Dict[Tuple[str, int, int, int, str], Dict[str, torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    if pretrain_stage == "none":
        return None

    key = (
        pretrain_stage,
        int(params["hidden_channels"]),
        int(params["num_layers"]),
        int(params["K"]),
        str(params["pool"]),
    )
    if key in cache:
        return cache[key]

    arch = {
        "hidden_channels": int(params["hidden_channels"]),
        "num_layers": int(params["num_layers"]),
        "K": int(params["K"]),
        "pool": str(params["pool"]),
    }

    ckpt_path = resolve_pretrain_checkpoint(
        workspace_root,
        pretrain_stage,
        hidden_channels=arch["hidden_channels"],
        num_layers=arch["num_layers"],
        K=arch["K"],
        pool=arch["pool"],
    )
    enc_state = load_pretrained_encoder_state(
        ckpt_path,
        expected_stage=pretrain_stage,
        expected_arch=arch,
    )
    cache[key] = enc_state
    return enc_state


def evaluate_params_for_seed(
    params: Dict[str, Any],
    seed: int,
    bundle: BACEDataBundle,
    *,
    device: str,
    num_epochs: int,
    pretrain_stage: str,
    workspace_root: Path,
    pretrain_cache: Dict[Tuple[str, int, int, int, str], Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    set_global_seed(seed)

    cfg = build_tame_fusion_config(bundle.text_dim, params)
    init_encoder_state = maybe_get_pretrained_state(pretrain_stage, params, workspace_root, pretrain_cache)

    predictor = TAMEFusionPredictor(config=cfg, device=device)
    history = predictor.fit(
        smiles_list=bundle.train_smiles,
        labels=bundle.y_train_raw.tolist(),
        val_smiles=bundle.valid_smiles,
        val_labels=bundle.y_valid_raw.tolist(),
        text_embeddings=bundle.train_emb,
        val_text_embeddings=bundle.valid_emb,
        descriptor_features=bundle.rd_train,
        val_descriptor_features=bundle.rd_valid,
        num_epochs=int(num_epochs),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        patience=int(params["patience"]),
        verbose=False,
        seed=int(seed),
        init_encoder_state=init_encoder_state,
    )

    valid_proba = np.asarray(
        predictor.predict_batch(
            bundle.valid_smiles,
            text_embeddings=bundle.valid_emb,
            descriptor_features=bundle.rd_valid,
            batch_size=int(params["batch_size"]),
        ),
        dtype=np.float32,
    ).reshape(-1)
    test_proba = np.asarray(
        predictor.predict_batch(
            bundle.test_smiles,
            text_embeddings=bundle.test_emb,
            descriptor_features=bundle.rd_test,
            batch_size=int(params["batch_size"]),
        ),
        dtype=np.float32,
    ).reshape(-1)

    best_thr = find_best_threshold(bundle.y_valid_raw, valid_proba, weights=bundle.w_valid)
    valid_metrics = classification_metrics(
        bundle.y_valid_raw,
        valid_proba,
        threshold=best_thr,
        weights=bundle.w_valid,
    )
    test_metrics = classification_metrics(
        bundle.y_test_raw,
        test_proba,
        threshold=best_thr,
        weights=bundle.w_test,
    )

    val_hist = history.get("val_loss", [])
    best_val_loss = float(np.min(val_hist)) if len(val_hist) > 0 else float("nan")

    return {
        "val_roc_auc": float(valid_metrics["roc_auc"]),
        "val_accuracy": float(valid_metrics["accuracy"]),
        "val_f1": float(valid_metrics["f1"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
        "best_threshold": float(best_thr),
        "best_val_loss": best_val_loss,
    }


def build_storage_url(output_dir: Path, storage_arg: Optional[str]) -> str:
    if storage_arg is not None and storage_arg.strip():
        return storage_arg.strip()
    db_path = (output_dir / "optuna_study.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPO for TAME-Fusion on BACE (DeepChem scaffold split)")

    parser.add_argument("--study-name", type=str, default="tame_fusion_bace")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="benchmarking/results/hpo/tame_fusion_bace")

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=0, help="Seconds, 0 means no timeout")
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--pruner", choices=["none", "median"], default="median")
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=20,
        help="Median pruner startup trials before pruning starts.",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=1,
        help="Median pruner warmup steps for trial.report(step).",
    )
    parser.add_argument("--optuna-seed", type=int, default=12345)

    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--trial-seeds", type=str, default="11")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--search-arch", action="store_true")
    parser.add_argument("--pretrain-stage", choices=["none", "stage1_node"], default="stage1_node")
    parser.add_argument(
        "--pretrain-arch-policy",
        choices=["strict", "off"],
        default="strict",
        help="strict: freeze architecture candidates from available stage checkpoints.",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=str,
        default="cache/cot_embeddings/binding_fast_text_embeddings_compact.npz",
    )
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--run-final-eval", action="store_true")
    parser.add_argument(
        "--final-eval-only",
        action="store_true",
        help="Skip new HPO trials and only run final evaluation on top-k existing complete trials.",
    )
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--final-eval-seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--final-num-epochs", type=int, default=140)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = (WORKSPACE_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_npz = (WORKSPACE_ROOT / args.embeddings_npz).resolve()
    storage_url = build_storage_url(output_dir, args.storage)
    trial_seeds = parse_int_list(args.trial_seeds)
    if len(trial_seeds) == 0:
        raise ValueError("At least one trial seed is required.")

    final_eval_seeds = parse_int_list(args.final_eval_seeds)
    if args.run_final_eval and len(final_eval_seeds) == 0:
        raise ValueError("At least one final evaluation seed is required.")
    if args.final_eval_only and not args.run_final_eval:
        raise ValueError("--final-eval-only requires --run-final-eval.")

    run_config_path = output_dir / "run_config.json"
    previous_run_config: Dict[str, Any] = {}
    if run_config_path.exists():
        try:
            previous_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception:
            previous_run_config = {}

    discovered_arches: List[Dict[str, Any]] = []
    discovered_arch_keys: List[str] = []
    strict_arch_keys: List[str] = []
    strict_catalog_source = "none"

    if args.pretrain_stage != "none":
        discovered_arches = discover_pretrain_architectures(WORKSPACE_ROOT, args.pretrain_stage)
        discovered_arch_keys = [str(item["arch_key"]) for item in discovered_arches]

    if args.pretrain_stage != "none" and args.pretrain_arch_policy == "strict":
        frozen_catalog = previous_run_config.get("pretrain_arch_catalog", None)
        if isinstance(frozen_catalog, list) and len(frozen_catalog) > 0:
            strict_arch_keys = [str(x) for x in frozen_catalog]
            strict_catalog_source = "frozen_run_config"
        else:
            strict_arch_keys = list(discovered_arch_keys)
            strict_catalog_source = "discovered_from_cache"

        if len(strict_arch_keys) == 0:
            raise RuntimeError(
                "No architecture-specific stage1 checkpoints found in cache/chemeleon_pretraining. "
                "Use --pretrain-stage none or generate matching stage1 checkpoints first."
            )

        if not args.search_arch:
            fixed_arch_key = arch_to_key(hidden_channels=128, num_layers=3, K=3, pool="set2set")
            if fixed_arch_key not in strict_arch_keys:
                raise RuntimeError(
                    "Fixed architecture h128/L3/K3/pool-set2set is not available in strict stage1 catalog. "
                    "Use --search-arch to sample only compatible architectures or use --pretrain-stage none."
                )

    run_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(WORKSPACE_ROOT),
        "output_dir": str(output_dir),
        "storage": storage_url,
        "objective": "maximize val_roc_auc",
        "pretrain_alignment_policy": "strict_architecture_match" if args.pretrain_arch_policy == "strict" else "off",
        "pretrain_arch_catalog_source": strict_catalog_source,
        "pretrain_arch_catalog": strict_arch_keys,
        "discovered_pretrain_arch_catalog": discovered_arch_keys,
        "args": vars(args),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("=" * 80)
    print("TAME-Fusion HPO on BACE")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    print(f"Output dir: {output_dir}")
    print(f"Storage: {storage_url}")
    print(f"Device: {args.device}")
    print(f"Trial seeds: {trial_seeds}")
    print(f"Search architecture: {bool(args.search_arch)}")
    print(f"Pretrain stage: {args.pretrain_stage}")
    if args.pretrain_stage != "none":
        print(f"Pretrain arch policy: {args.pretrain_arch_policy}")
        if args.pretrain_arch_policy == "strict":
            print(f"Strict stage1 catalog source: {strict_catalog_source} | size={len(strict_arch_keys)}")
    print("=" * 80)

    bundle = load_bace_bundle(WORKSPACE_ROOT, embeddings_npz)
    pos_ratio = float(np.mean(bundle.y_train_raw[bundle.w_train > 0])) if bool((bundle.w_train > 0).any()) else float("nan")
    print(
        "Loaded BACE split sizes: "
        f"train={len(bundle.train_smiles)}, valid={len(bundle.valid_smiles)}, test={len(bundle.test_smiles)}"
    )
    print(f"Train positive ratio (weighted-valid labels): {pos_ratio:.3f}")
    print(f"Text dim: {bundle.text_dim} | RDKit descriptors: {bundle.rd_train.shape[1]}")

    if args.sampler == "tpe":
        sampler = TPESampler(seed=args.optuna_seed)
    else:
        sampler = RandomSampler(seed=args.optuna_seed)

    if args.pruner == "median":
        pruner = MedianPruner(
            n_startup_trials=max(0, int(args.pruner_startup_trials)),
            n_warmup_steps=max(0, int(args.pruner_warmup_steps)),
        )
    else:
        pruner = NopPruner()

    pretrain_cache: Dict[Tuple[str, int, int, int, str], Dict[str, torch.Tensor]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = sample_hparams(
            trial,
            search_arch=bool(args.search_arch),
            strict_arch_keys=strict_arch_keys if (args.pretrain_stage != "none" and args.pretrain_arch_policy == "strict") else None,
        )

        per_seed_rows: List[Dict[str, float]] = []
        for i, seed in enumerate(trial_seeds, start=1):
            try:
                row = evaluate_params_for_seed(
                    params,
                    seed,
                    bundle,
                    device=args.device,
                    num_epochs=int(args.num_epochs),
                    pretrain_stage=args.pretrain_stage,
                    workspace_root=WORKSPACE_ROOT,
                    pretrain_cache=pretrain_cache,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                if args.pretrain_stage != "none":
                    raise optuna.TrialPruned(f"Pretrain architecture mismatch or missing checkpoint: {exc}")
                if args.fail_fast:
                    raise
                trial.set_user_attr("error", f"seed={seed}: {type(exc).__name__}: {exc}")
                return 0.0
            except Exception as exc:
                if args.fail_fast:
                    raise
                trial.set_user_attr("error", f"seed={seed}: {type(exc).__name__}: {exc}")
                return 0.0

            row["seed"] = int(seed)
            per_seed_rows.append(row)

            running_val_auc = finite_mean([r["val_roc_auc"] for r in per_seed_rows], default=0.0)
            trial.report(running_val_auc, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_val_auc = finite_mean([r["val_roc_auc"] for r in per_seed_rows], default=0.0)
        mean_val_acc = finite_mean([r["val_accuracy"] for r in per_seed_rows], default=0.0)
        mean_val_f1 = finite_mean([r["val_f1"] for r in per_seed_rows], default=0.0)
        mean_test_auc = finite_mean([r["test_roc_auc"] for r in per_seed_rows], default=0.0)
        mean_test_acc = finite_mean([r["test_accuracy"] for r in per_seed_rows], default=0.0)
        mean_test_f1 = finite_mean([r["test_f1"] for r in per_seed_rows], default=0.0)
        mean_threshold = finite_mean([r["best_threshold"] for r in per_seed_rows], default=0.5)

        trial.set_user_attr("mean_val_roc_auc", mean_val_auc)
        trial.set_user_attr("mean_val_accuracy", mean_val_acc)
        trial.set_user_attr("mean_val_f1", mean_val_f1)
        trial.set_user_attr("mean_test_roc_auc", mean_test_auc)
        trial.set_user_attr("mean_test_accuracy", mean_test_acc)
        trial.set_user_attr("mean_test_f1", mean_test_f1)
        trial.set_user_attr("mean_best_threshold", mean_threshold)
        trial.set_user_attr("trial_seeds", list(map(int, trial_seeds)))

        return mean_val_auc

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    timeout_value = None if int(args.timeout) <= 0 else int(args.timeout)
    if args.final_eval_only:
        print("Skipping optimization because --final-eval-only is enabled.")
    else:
        study.optimize(
            objective,
            n_trials=int(args.n_trials),
            timeout=timeout_value,
            n_jobs=int(args.n_jobs),
            gc_after_trial=True,
            show_progress_bar=True,
        )

    trials_csv = output_dir / "trials.csv"
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(trials_csv, index=False)

    best = study.best_trial
    best_payload = {
        "study_name": study.study_name,
        "best_trial_number": int(best.number),
        "best_value_val_roc_auc": float(best.value),
        "best_params": canonicalize_params(best.params),
        "best_user_attrs": best.user_attrs,
        "trial_count": len(study.trials),
        "storage": storage_url,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    print("Optimization finished.")
    print(f"Best trial: {best.number}")
    print(f"Best mean validation ROC-AUC: {best.value:.6f}")
    print(f"Best params saved to: {output_dir / 'best_params.json'}")
    print(f"All trials saved to: {trials_csv}")

    if args.run_final_eval:
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        complete_trials = sorted(complete_trials, key=lambda t: float(t.value), reverse=True)
        if len(complete_trials) == 0:
            raise RuntimeError(
                "No complete trials available for final evaluation. Run HPO first or disable --final-eval-only."
            )
        top_trials = complete_trials[: max(1, int(args.final_top_k))]

        raw_rows: List[Dict[str, Any]] = []
        for t in top_trials:
            params = canonicalize_params(t.params)
            for seed in final_eval_seeds:
                row = evaluate_params_for_seed(
                    params,
                    seed,
                    bundle,
                    device=args.device,
                    num_epochs=int(args.final_num_epochs),
                    pretrain_stage=args.pretrain_stage,
                    workspace_root=WORKSPACE_ROOT,
                    pretrain_cache=pretrain_cache,
                )
                row["trial_number"] = int(t.number)
                row["seed"] = int(seed)
                raw_rows.append(row)

        final_raw_df = pd.DataFrame(raw_rows)
        final_summary_df = (
            final_raw_df.groupby("trial_number", as_index=False)
            .agg(
                val_roc_auc_mean=("val_roc_auc", "mean"),
                val_roc_auc_std=("val_roc_auc", "std"),
                val_accuracy_mean=("val_accuracy", "mean"),
                val_accuracy_std=("val_accuracy", "std"),
                val_f1_mean=("val_f1", "mean"),
                val_f1_std=("val_f1", "std"),
                test_roc_auc_mean=("test_roc_auc", "mean"),
                test_roc_auc_std=("test_roc_auc", "std"),
                test_accuracy_mean=("test_accuracy", "mean"),
                test_accuracy_std=("test_accuracy", "std"),
                test_f1_mean=("test_f1", "mean"),
                test_f1_std=("test_f1", "std"),
                threshold_mean=("best_threshold", "mean"),
                threshold_std=("best_threshold", "std"),
                n_runs=("seed", "count"),
            )
            .sort_values("val_roc_auc_mean", ascending=False)
            .reset_index(drop=True)
        )

        final_raw_path = output_dir / "final_eval_raw.csv"
        final_summary_path = output_dir / "final_eval_summary.csv"
        final_raw_df.to_csv(final_raw_path, index=False)
        final_summary_df.to_csv(final_summary_path, index=False)

        final_payload = {
            "evaluated_trial_numbers": [int(t.number) for t in top_trials],
            "final_eval_seeds": list(map(int, final_eval_seeds)),
            "final_num_epochs": int(args.final_num_epochs),
            "pretrain_stage": args.pretrain_stage,
            "search_arch": bool(args.search_arch),
            "objective": "maximize val_roc_auc",
        }
        (output_dir / "final_eval_config.json").write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

        print(f"Final eval raw results: {final_raw_path}")
        print(f"Final eval summary: {final_summary_path}")


if __name__ == "__main__":
    main()
