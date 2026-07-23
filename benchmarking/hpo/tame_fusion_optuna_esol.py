"""
Optuna-based hyperparameter optimization for TAME-Fusion on ESOL.

Goals:
- Run locally on a single GPU for quick iteration.
- Resume and scale on cluster workers through shared Optuna storage.

Example (local):
    python benchmarking/hpo/tame_fusion_optuna_esol.py \
        --n-trials 20 \
        --device cuda \
        --trial-seeds 11

Example (cluster, multiple workers):
    python benchmarking/hpo/tame_fusion_optuna_esol.py \
        --study-name tame_fusion_esol \
        --storage postgresql://user:pass@host:5432/optuna \
        --n-trials 200 \
        --timeout 0 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
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


@dataclass
class ESOLDataBundle:
    train_smiles: List[str]
    valid_smiles: List[str]
    test_smiles: List[str]
    y_train_raw: np.ndarray
    y_valid_raw: np.ndarray
    y_test_raw: np.ndarray
    y_train_std: np.ndarray
    y_valid_std: np.ndarray
    y_test_std: np.ndarray
    train_emb: np.ndarray
    valid_emb: np.ndarray
    test_emb: np.ndarray
    text_dim: int
    rd_train: np.ndarray
    rd_valid: np.ndarray
    rd_test: np.ndarray
    target_stats: Dict[str, float]


def parse_int_list(raw: str) -> List[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(x) for x in values]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_np = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred_np = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    mask = np.isfinite(y_true_np) & np.isfinite(y_pred_np)
    if int(mask.sum()) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n_eval": 0}

    yt = y_true_np[mask]
    yp = y_pred_np[mask]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    if yt.size > 1 and np.std(yt) > 0 and np.std(yp) > 0:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = float("nan")
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "n_eval": int(mask.sum())}


def standardize_targets(
    y_train: Sequence[float],
    y_val: Sequence[float],
    y_test: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_val_np = np.asarray(y_val, dtype=np.float32).reshape(-1)
    y_test_np = np.asarray(y_test, dtype=np.float32).reshape(-1)

    mu = float(np.mean(y_train_np))
    sigma = float(np.std(y_train_np))
    if not np.isfinite(sigma) or sigma < 1e-8:
        sigma = 1.0

    return (
        (y_train_np - mu) / sigma,
        (y_val_np - mu) / sigma,
        (y_test_np - mu) / sigma,
        {"mean": mu, "std": sigma},
    )


def inverse_standardize(values: Sequence[float], stats: Dict[str, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1) * float(stats["std"]) + float(stats["mean"])


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


def resolve_pretrain_checkpoint(
    workspace_root: Path,
    pretrain_stage: str,
    *,
    hidden_channels: int,
    num_layers: int,
    K: int,
    pool: str,
) -> Path:
    stage_cfg = {
        "stage1_node": {
            "alias": "cheb_foundation_pretrain_stage1.pt",
            "pattern_prefix": "cheb_foundation_stage1",
        },
    }
    if pretrain_stage not in stage_cfg:
        raise ValueError(f"Unsupported pretrain_stage={pretrain_stage!r}")

    ckpt_dir = workspace_root / "cache" / "chemeleon_pretraining"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    pattern_prefix = stage_cfg[pretrain_stage]["pattern_prefix"]
    pattern = f"{pattern_prefix}__h{int(hidden_channels)}__L{int(num_layers)}__K{int(K)}__pool-{pool}__*.pt"

    candidates = sorted(
        [p for p in ckpt_dir.glob(pattern) if "__fallback__" not in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    alias = ckpt_dir / stage_cfg[pretrain_stage]["alias"]
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


def load_esol_bundle(workspace_root: Path, embeddings_npz: Path) -> ESOLDataBundle:
    tasks, datasets, _ = dc.molnet.load_delaney(featurizer="ECFP", splitter="scaffold")
    _ = tasks
    train_dc, valid_dc, test_dc = datasets

    train_smiles = list(train_dc.ids)
    valid_smiles = list(valid_dc.ids)
    test_smiles = list(test_dc.ids)

    y_train_raw = np.asarray(train_dc.y, dtype=np.float32).reshape(-1)
    y_valid_raw = np.asarray(valid_dc.y, dtype=np.float32).reshape(-1)
    y_test_raw = np.asarray(test_dc.y, dtype=np.float32).reshape(-1)

    y_train_std, y_valid_std, y_test_std, target_stats = standardize_targets(
        y_train_raw,
        y_valid_raw,
        y_test_raw,
    )

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

    return ESOLDataBundle(
        train_smiles=train_smiles,
        valid_smiles=valid_smiles,
        test_smiles=test_smiles,
        y_train_raw=y_train_raw,
        y_valid_raw=y_valid_raw,
        y_test_raw=y_test_raw,
        y_train_std=y_train_std,
        y_valid_std=y_valid_std,
        y_test_std=y_test_std,
        train_emb=train_emb,
        valid_emb=valid_emb,
        test_emb=test_emb,
        text_dim=text_dim,
        rd_train=rd_train,
        rd_valid=rd_valid,
        rd_test=rd_test,
        target_stats=target_stats,
    )


def sample_hparams(trial: optuna.Trial, search_arch: bool) -> Dict[str, Any]:
    if search_arch:
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        k_cheb = trial.suggest_categorical("K", [2, 3, 4, 5])
        num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5])
        pool = trial.suggest_categorical("pool", ["sum", "set2set"])
        set2set_steps = trial.suggest_categorical("set2set_processing_steps", [4, 6, 8])
    else:
        hidden_channels = 128
        k_cheb = 3
        num_layers = 3
        pool = "sum"
        set2set_steps = 6

    # Sample only valid (fusion_dim, n_heads) pairs with a static categorical space.
    # This avoids wasting trials on invalid combinations while staying Optuna-study-safe.
    fusion_pair = trial.suggest_categorical(
        "fusion_dim_heads_pair_v4",
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
        "moe_hidden_dim": trial.suggest_categorical("moe_hidden_dim", [32, 64, 128]),
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
    """Map historical/new Optuna parameter keys to a single canonical schema."""
    params = dict(raw_params)
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
    return params


def build_tame_fusion_config(text_dim: int, params: Dict[str, Any]) -> TAMEFusionPredictorConfig:
    return TAMEFusionPredictorConfig(
        task="regression",
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
    bundle: ESOLDataBundle,
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
        labels=bundle.y_train_std.tolist(),
        val_smiles=bundle.valid_smiles,
        val_labels=bundle.y_valid_std.tolist(),
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

    valid_pred_std = np.asarray(
        predictor.predict_batch(
            bundle.valid_smiles,
            text_embeddings=bundle.valid_emb,
            descriptor_features=bundle.rd_valid,
            batch_size=int(params["batch_size"]),
        ),
        dtype=np.float32,
    ).reshape(-1)
    test_pred_std = np.asarray(
        predictor.predict_batch(
            bundle.test_smiles,
            text_embeddings=bundle.test_emb,
            descriptor_features=bundle.rd_test,
            batch_size=int(params["batch_size"]),
        ),
        dtype=np.float32,
    ).reshape(-1)

    valid_pred_raw = inverse_standardize(valid_pred_std, bundle.target_stats)
    test_pred_raw = inverse_standardize(test_pred_std, bundle.target_stats)

    valid_metrics = regression_metrics(bundle.y_valid_raw, valid_pred_raw)
    test_metrics = regression_metrics(bundle.y_test_raw, test_pred_raw)

    val_hist = history.get("val_loss", [])
    best_val_loss = float(np.min(val_hist)) if len(val_hist) > 0 else float("nan")

    return {
        "val_rmse": float(valid_metrics["rmse"]),
        "val_mae": float(valid_metrics["mae"]),
        "val_pearson": float(valid_metrics["pearson"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_pearson": float(test_metrics["pearson"]),
        "best_val_loss": best_val_loss,
    }


def build_storage_url(output_dir: Path, storage_arg: Optional[str]) -> str:
    if storage_arg is not None and storage_arg.strip():
        return storage_arg.strip()
    db_path = (output_dir / "optuna_study.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPO for TAME-Fusion on ESOL (DeepChem scaffold split)")

    parser.add_argument("--study-name", type=str, default="tame_fusion_esol")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="benchmarking/results/hpo/tame_fusion_esol")

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=0, help="Seconds, 0 means no timeout")
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--pruner", choices=["none", "median"], default="median")
    parser.add_argument("--optuna-seed", type=int, default=12345)

    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--trial-seeds", type=str, default="11")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--search-arch", action="store_true")
    parser.add_argument("--pretrain-stage", choices=["none", "stage1_node"], default="stage1_node")
    parser.add_argument(
        "--embeddings-npz",
        type=str,
        default="cache/cot_embeddings/solubility_fast_text_embeddings_compact.npz",
    )
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--run-final-eval", action="store_true")
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

    run_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(WORKSPACE_ROOT),
        "output_dir": str(output_dir),
        "storage": storage_url,
        "args": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("=" * 80)
    print("TAME-Fusion HPO on ESOL")
    print(f"Workspace root: {WORKSPACE_ROOT}")
    print(f"Output dir: {output_dir}")
    print(f"Storage: {storage_url}")
    print(f"Device: {args.device}")
    print(f"Trial seeds: {trial_seeds}")
    print(f"Search architecture: {bool(args.search_arch)}")
    print(f"Pretrain stage: {args.pretrain_stage}")
    print("=" * 80)

    bundle = load_esol_bundle(WORKSPACE_ROOT, embeddings_npz)
    print(
        "Loaded ESOL split sizes: "
        f"train={len(bundle.train_smiles)}, valid={len(bundle.valid_smiles)}, test={len(bundle.test_smiles)}"
    )
    print(f"Text dim: {bundle.text_dim} | RDKit descriptors: {bundle.rd_train.shape[1]}")

    if args.sampler == "tpe":
        sampler = TPESampler(seed=args.optuna_seed)
    else:
        sampler = RandomSampler(seed=args.optuna_seed)

    if args.pruner == "median":
        pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    else:
        pruner = NopPruner()

    pretrain_cache: Dict[Tuple[str, int, int, int, str], Dict[str, torch.Tensor]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = sample_hparams(trial, search_arch=bool(args.search_arch))

        per_seed_rows = []
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
            except Exception as exc:
                if args.fail_fast:
                    raise
                trial.set_user_attr("error", f"seed={seed}: {type(exc).__name__}: {exc}")
                return float("inf")

            row["seed"] = int(seed)
            per_seed_rows.append(row)

            running_val_rmse = float(np.mean([r["val_rmse"] for r in per_seed_rows]))
            trial.report(running_val_rmse, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_val_rmse = float(np.mean([r["val_rmse"] for r in per_seed_rows]))
        mean_val_mae = float(np.mean([r["val_mae"] for r in per_seed_rows]))
        mean_val_pearson = float(np.mean([r["val_pearson"] for r in per_seed_rows]))
        mean_test_rmse = float(np.mean([r["test_rmse"] for r in per_seed_rows]))
        mean_test_mae = float(np.mean([r["test_mae"] for r in per_seed_rows]))
        mean_test_pearson = float(np.mean([r["test_pearson"] for r in per_seed_rows]))

        trial.set_user_attr("mean_val_rmse", mean_val_rmse)
        trial.set_user_attr("mean_val_mae", mean_val_mae)
        trial.set_user_attr("mean_val_pearson", mean_val_pearson)
        trial.set_user_attr("mean_test_rmse", mean_test_rmse)
        trial.set_user_attr("mean_test_mae", mean_test_mae)
        trial.set_user_attr("mean_test_pearson", mean_test_pearson)
        trial.set_user_attr("trial_seeds", list(map(int, trial_seeds)))

        return mean_val_rmse

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    timeout_value = None if int(args.timeout) <= 0 else int(args.timeout)
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
        "best_value_val_rmse": float(best.value),
        "best_params": canonicalize_params(best.params),
        "best_user_attrs": best.user_attrs,
        "trial_count": len(study.trials),
        "storage": storage_url,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    print("Optimization finished.")
    print(f"Best trial: {best.number}")
    print(f"Best mean validation RMSE: {best.value:.6f}")
    print(f"Best params saved to: {output_dir / 'best_params.json'}")
    print(f"All trials saved to: {trials_csv}")

    if args.run_final_eval:
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        complete_trials = sorted(complete_trials, key=lambda t: float(t.value))
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
                val_rmse_mean=("val_rmse", "mean"),
                val_rmse_std=("val_rmse", "std"),
                val_mae_mean=("val_mae", "mean"),
                val_mae_std=("val_mae", "std"),
                val_pearson_mean=("val_pearson", "mean"),
                val_pearson_std=("val_pearson", "std"),
                test_rmse_mean=("test_rmse", "mean"),
                test_rmse_std=("test_rmse", "std"),
                test_mae_mean=("test_mae", "mean"),
                test_mae_std=("test_mae", "std"),
                test_pearson_mean=("test_pearson", "mean"),
                test_pearson_std=("test_pearson", "std"),
                n_runs=("seed", "count"),
            )
            .sort_values("val_rmse_mean", ascending=True)
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
        }
        (output_dir / "final_eval_config.json").write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

        print(f"Final eval raw results: {final_raw_path}")
        print(f"Final eval summary: {final_summary_path}")


if __name__ == "__main__":
    main()
