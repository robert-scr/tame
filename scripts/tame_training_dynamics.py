import os
import sys
import io
import json
import argparse
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
import deepchem as dc
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# Workspace Root hinzufügen
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.tame_predictor import TAMEPredictor, TAMEPredictorConfig
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="benchmarking/results", help="Directory for saved metrics")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to foundation stage 1 checkpoint")
    parser.add_argument("--es_metric", type=str, default="val_loss", choices=["val_loss", "val_auc"])
    parser.add_argument("--gate_balance_weight", type=float, default=0.002, help="Weight for the MoE gate balancing loss")
    parser.add_argument("--gate_entropy_weight", type=float, default=0.0, help="Weight for per-position gate entropy regularisation (0 = off)")
    parser.add_argument("--gate_targets", type=float, nargs=3, default=[0.04, 0.86, 0.1], help="Target gate distribution for [Graph, Text, Desc]")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # === HPO Best Params ===
    HPO_PARAMS = {
        "K": 5,
        "num_layers": 3,
        "hidden_channels": 64,
        "set2set_n_iter": 9,
        "proj_dim": 256,
        "projection_dropout": 0.38,
        "router_dropout": 0.22,
        "head_hidden_dim": 64,
        "head_dropout": 0.44,
        "gate_balance_weight": args.gate_balance_weight,
        "gate_entropy_weight": args.gate_entropy_weight,
        "desc_modality_dropout": 0.76,
        "gate_target_graph": args.gate_targets[0],
        "gate_target_text": args.gate_targets[1],
        "gate_target_desc": args.gate_targets[2],
        "lr": 84e-5,
        "weight_decay": 9e-5,
        "batch_size": 16,
        "encoder_lr_mult": 0.46
    }
    SEEDS = [11, 22, 33, 44, 55]
    MAX_EPOCHS = 120
    PATIENCE = 20
    SPLITTER = "scaffold"

    print(f"\n[Info] Loading BACE-Data ({SPLITTER.title()} Split)...")
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        _, datasets, _ = dc.molnet.load_bace_classification(featurizer='ECFP', splitter=SPLITTER, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    train_dc, valid_dc, test_dc = datasets

    train_smiles, val_smiles, test_smiles = list(train_dc.ids), list(valid_dc.ids), list(test_dc.ids)
    y_train, y_val = train_dc.y.reshape(-1).astype(np.float32), valid_dc.y.reshape(-1).astype(np.float32)

    emb_path = os.path.join(project_root, "cache", "cot_embeddings", "binding_fast_text_embeddings_compact.npz")
    seg_cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    text_train = np.asarray(seg_cache.get_batch(train_smiles), dtype=np.float32)
    text_val = np.asarray(seg_cache.get_batch(val_smiles), dtype=np.float32)

    desc_train_raw = compute_rdkit_descriptors(train_smiles)
    desc_val_raw = compute_rdkit_descriptors(val_smiles)
    desc_test_raw = compute_rdkit_descriptors(test_smiles)
    desc_train, desc_val, _ = impute_with_train_median(desc_train_raw, desc_val_raw, desc_test_raw)

    config = TAMEPredictorConfig(
        task="classification",
        num_tasks=1,
        hidden_channels=HPO_PARAMS["hidden_channels"],
        K=HPO_PARAMS["K"],
        num_layers=HPO_PARAMS["num_layers"],
        pool="set2set",
        set2set_processing_steps=HPO_PARAMS["set2set_n_iter"],
        text_embedding_dim=text_train.shape[1],
        descriptor_dim=desc_train.shape[1],
        fusion_hidden_dim=HPO_PARAMS["proj_dim"],
        projection_dropout=HPO_PARAMS["projection_dropout"],
        router_dropout=HPO_PARAMS["router_dropout"],
        head_hidden_dim=HPO_PARAMS["head_hidden_dim"],
        head_dropout=HPO_PARAMS["head_dropout"],
        gate_balance_weight=HPO_PARAMS["gate_balance_weight"],
        gate_entropy_weight=HPO_PARAMS["gate_entropy_weight"],
        desc_modality_dropout=HPO_PARAMS["desc_modality_dropout"],
        gate_target=(HPO_PARAMS["gate_target_graph"], HPO_PARAMS["gate_target_text"], HPO_PARAMS["gate_target_desc"]),
        descriptor_standardize=True,
        descriptor_winsorize_lower_q=0.01,
        descriptor_winsorize_upper_q=0.99
    )

    all_histories = []
    model = None

    for seed in SEEDS:
        print(f"\n=== Training Seed {seed} ===")
        set_seed(seed)
        
        state = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("encoder_state", state.get("model_state_dict", state))
        filtered_dict = {k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")}
        if not filtered_dict: filtered_dict = state_dict
            
        model = TAMEPredictor(config=config, device=str(device))
        hist = model.fit(
            smiles_list=train_smiles, labels=y_train, val_smiles=val_smiles, val_labels=y_val,
            text_embeddings=text_train, val_text_embeddings=text_val,
            descriptor_features=desc_train, val_descriptor_features=desc_val,
            num_epochs=MAX_EPOCHS, batch_size=HPO_PARAMS["batch_size"],
            learning_rate=HPO_PARAMS["lr"], weight_decay=HPO_PARAMS["weight_decay"],
            encoder_lr_mult=HPO_PARAMS["encoder_lr_mult"], patience=PATIENCE,
            early_stopping_metric=args.es_metric, verbose=False, seed=seed,
            init_encoder_state=filtered_dict
        )
        print(f"  -> Stopped at epoch {len(hist['val_loss'])}. Best Val {args.es_metric.upper()}: {min(hist['val_loss']) if args.es_metric=='val_loss' else max(hist['val_auc']):.4f}")
        all_histories.append(hist)

    print("\nSaving histories...")
    with open(os.path.join(args.out_dir, "tame_dynamics_histories.json"), "w") as f:
        json.dump(all_histories, f)

    print("Evaluating gates for last seed...")
    model._set_eval_mode()
    graph_gates, text_gates, desc_gates = [], [], []
    with torch.no_grad():
        for start in range(0, len(val_smiles), 128):
            chunk_graphs = [model._precompute_graphs([smi], verbose=False)[0][0] for smi in val_smiles[start:start+128]]
            x, edge_index, edge_weight, batch = model._to_tensors(batch_graphs(chunk_graphs))
            text_t = torch.from_numpy(text_val[start:start+128].astype(np.float32)).to(device)
            desc_np = model._transform_descriptors(desc_val[start:start+128].astype(np.float32))
            desc_t = torch.from_numpy(desc_np).to(device)
            
            _, aux = model._forward(x, edge_index, edge_weight, batch, text_t, desc_t)
            graph_gates.extend(aux["graph_gate"].cpu().numpy().tolist())
            text_gates.extend(aux["text_gate"].cpu().numpy().tolist())
            desc_gates.extend(aux["desc_gate"].cpu().numpy().tolist())
            
    np.savez(os.path.join(args.out_dir, "tame_dynamics_gates.npz"), 
             graph_gates=np.array(graph_gates), text_gates=np.array(text_gates), desc_gates=np.array(desc_gates))
    print(f"Artifacts successfully saved to {args.out_dir}!")

if __name__ == "__main__":
    main()
