import os
import sys
import io
import json
import argparse
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
import deepchem as dc
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.tame_fusion_predictor import TAMEFusionPredictor, TAMEFusionPredictorConfig
from utils.embedding_cache import EfficientEmbeddingCache
from utils.batched_mol_graph import batch_graphs

RDLogger.DisableLog("rdApp.*")


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


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Sensible defaults — update once HPO finishes via --params_json
DEFAULT_HPO_PARAMS = {
    "K": 4, "num_layers": 2, "hidden_channels": 128,
    "pool_type": "set2set", "set2set_n_iter": 6,
    "encoder_lr_mult": 0.1,
    "text_projection_dim": 256,
    "fusion": "cross_mha", "fusion_dim": 128, "fusion_n_heads": 4,
    "moe_hidden_dim": 128, "projection_dropout": 0.3, "router_dropout": 0.3,
    "head_hidden_dim": 64, "head_dropout": 0.4,
    "gate_balance_weight": 0.05, "gate_entropy_weight": 0.0,
    "label_smoothing": 0.05,
    "desc_modality_dropout": 0.4, "gate_target_seg": 0.5,
    "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 16,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pretrained ChebNet checkpoint")
    parser.add_argument("--params_json", type=str, default=None, help="Path to best HPO params JSON (tame_fusion_hpo.py output)")
    parser.add_argument("--es_metric", type=str, default="val_loss", choices=["val_loss", "val_auc"])
    parser.add_argument("-- ", type=float, default=None, help="Override SEG gate target (default: from params or 0.5)")
    parser.add_argument("--gate_balance_weight", type=float, default=None, help="Override MoE gate balancing weight (default: from params or 0.05)")
    parser.add_argument("--gate_entropy_weight", type=float, default=None, help="Override per-position gate entropy regularisation weight (0 = off)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    HPO_PARAMS = dict(DEFAULT_HPO_PARAMS)
    if args.params_json is not None:
        print(f"[Info] Loading HPO params from {args.params_json}...")
        with open(args.params_json) as f:
            hpo_data = json.load(f)
        HPO_PARAMS.update(hpo_data["params"])
        print(f"  -> Best trial value: {hpo_data.get('value', 'N/A'):.4f}")

    if args.gate_target_seg is not None:
        HPO_PARAMS["gate_target_seg"] = args.gate_target_seg
    if args.gate_balance_weight is not None:
        HPO_PARAMS["gate_balance_weight"] = args.gate_balance_weight
    if args.gate_entropy_weight is not None:
        HPO_PARAMS["gate_entropy_weight"] = args.gate_entropy_weight

    SEEDS = [11, 22, 33, 44, 55]
    MAX_EPOCHS = 120
    PATIENCE = 20
    SPLITTER = "scaffold"

    print(f"\n[Info] Loading BACE data ({SPLITTER.title()} Split)...")
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    dc_cache_dir = os.path.join(project_root, "cache", "deepchem_data")
    os.makedirs(dc_cache_dir, exist_ok=True)
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        _, datasets, _ = dc.molnet.load_bace_classification(
            featurizer="ECFP", splitter=SPLITTER,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1,
            data_dir=dc_cache_dir, save_dir=dc_cache_dir,
        )
    train_dc, valid_dc, test_dc = datasets

    train_smiles = list(train_dc.ids)
    val_smiles = list(valid_dc.ids)
    test_smiles = list(test_dc.ids)
    y_train = train_dc.y.astype(np.float32).reshape(-1)
    y_val = valid_dc.y.astype(np.float32).reshape(-1)

    emb_path = os.path.join(project_root, "cache", "cot_embeddings", "binding_fast_text_embeddings_compact.npz")
    print(f"[Info] Loading text embeddings from {emb_path}...")
    seg_cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    text_train = np.asarray(seg_cache.get_batch(train_smiles), dtype=np.float32)
    text_val = np.asarray(seg_cache.get_batch(val_smiles), dtype=np.float32)

    print("[Info] Computing RDKit descriptors...")
    desc_train_raw = compute_rdkit_descriptors(train_smiles)
    desc_val_raw = compute_rdkit_descriptors(val_smiles)
    desc_test_raw = compute_rdkit_descriptors(test_smiles)
    desc_train, desc_val, _ = impute_with_train_median(desc_train_raw, desc_val_raw, desc_test_raw)

    pool_type = HPO_PARAMS.get("pool_type", "set2set")
    fusion_type = HPO_PARAMS.get("fusion", "cross_mha")
    text_proj_dim = HPO_PARAMS.get("text_projection_dim", None)

    config = TAMEFusionPredictorConfig(
        task="classification",
        num_tasks=1,
        hidden_channels=int(HPO_PARAMS["hidden_channels"]),
        K=int(HPO_PARAMS["K"]),
        num_layers=int(HPO_PARAMS["num_layers"]),
        pool=pool_type,
        set2set_processing_steps=int(HPO_PARAMS.get("set2set_n_iter", 4)),
        text_embedding_dim=int(text_train.shape[1]),
        text_projection_dim=int(text_proj_dim) if text_proj_dim is not None else None,
        fusion=fusion_type,
        fusion_dim=int(HPO_PARAMS["fusion_dim"]),
        fusion_n_heads=int(HPO_PARAMS.get("fusion_n_heads", 4)),
        moe_hidden_dim=int(HPO_PARAMS["moe_hidden_dim"]),
        projection_dropout=float(HPO_PARAMS["projection_dropout"]),
        router_dropout=float(HPO_PARAMS["router_dropout"]),
        head_hidden_dim=int(HPO_PARAMS["head_hidden_dim"]),
        head_dropout=float(HPO_PARAMS["head_dropout"]),
        gate_balance_weight=float(HPO_PARAMS.get("gate_balance_weight", 0.05)),
        gate_entropy_weight=float(HPO_PARAMS.get("gate_entropy_weight", 0.0)),
        desc_modality_dropout=float(HPO_PARAMS["desc_modality_dropout"]),
        gate_target=float(HPO_PARAMS["gate_target_seg"]),
        label_smoothing=float(HPO_PARAMS.get("label_smoothing", 0.0)),
        descriptor_standardize=True,
        descriptor_winsorize_lower_q=0.01,
        descriptor_winsorize_upper_q=0.99,
    )

    all_histories = []
    model = None

    for seed in SEEDS:
        print(f"\n=== Training Seed {seed} ===")
        set_seed(seed)

        state = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("encoder_state", state.get("model_state_dict", state))
        filtered_dict = {k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")}
        if not filtered_dict:
            filtered_dict = state_dict

        model = TAMEFusionPredictor(config=config, device=str(device))
        hist = model.fit(
            smiles_list=train_smiles, labels=y_train,
            val_smiles=val_smiles, val_labels=y_val,
            text_embeddings=text_train, val_text_embeddings=text_val,
            descriptor_features=desc_train, val_descriptor_features=desc_val,
            num_epochs=MAX_EPOCHS, batch_size=int(HPO_PARAMS["batch_size"]),
            learning_rate=float(HPO_PARAMS["lr"]), weight_decay=float(HPO_PARAMS["weight_decay"]),
            encoder_lr_mult=float(HPO_PARAMS.get("encoder_lr_mult", 1.0)),
            patience=PATIENCE,
            early_stopping_metric=args.es_metric,
            verbose=False, seed=seed,
            init_encoder_state=filtered_dict,
        )
        best_metric = min(hist["val_loss"]) if args.es_metric == "val_loss" else max(hist["val_auc"])
        print(f"  -> Stopped at epoch {len(hist['val_loss'])}. Best {args.es_metric.upper()}: {best_metric:.4f}")
        all_histories.append(hist)

    print("\nSaving histories...")
    with open(os.path.join(args.out_dir, "tame_fusion_dynamics_histories.json"), "w") as f:
        json.dump(all_histories, f)

    print("Evaluating gates for last seed...")
    model._set_eval_mode()
    seg_gates, desc_gates = [], []
    with torch.no_grad():
        for start in range(0, len(val_smiles), 128):
            chunk = val_smiles[start:start + 128]
            chunk_graphs = [model._precompute_graphs([smi], verbose=False)[0][0] for smi in chunk]
            valid_chunk = [(g, i) for i, g in enumerate(chunk_graphs) if g is not None]
            if not valid_chunk:
                continue
            valid_graphs, valid_idx = zip(*valid_chunk)
            x, edge_index, edge_weight, batch = model._to_tensors(batch_graphs(list(valid_graphs)))

            abs_idx = [start + i for i in valid_idx]
            text_t = torch.from_numpy(text_val[abs_idx].astype(np.float32)).to(device)
            desc_np = model._transform_descriptors(desc_val[abs_idx].astype(np.float32))
            desc_t = torch.from_numpy(desc_np).to(device)

            _, aux = model._forward(x, edge_index, edge_weight, batch, text_t, desc_t)
            seg_gates.extend(aux["seg_gate"].cpu().numpy().tolist())
            desc_gates.extend(aux["desc_gate"].cpu().numpy().tolist())

    np.savez(
        os.path.join(args.out_dir, "tame_fusion_dynamics_gates.npz"),
        seg_gates=np.array(seg_gates),
        desc_gates=np.array(desc_gates),
    )
    print(f"Artifacts saved to {args.out_dir}!")


if __name__ == "__main__":
    main()
