"""PyG ChebNet baseline using TAME's own HPO recipe, for a fairer comparison.

Rather than an independently-tuned PyG config, this mirrors the existing
"TAME (graph-only)" control-chain step in bace_preliminary_results.py: same
backbone architecture (hidden_channels, K, num_layers, pool) and same optimizer
recipe (lr, weight_decay, batch_size, label_smoothing) as found by
scripts/tame_hpo_v3.py -- just with the in-house ChebLayer encoder swapped for
torch_geometric.nn.ChebConv, trained from scratch (PyG can't load the in-house
encoder's pretrained state dict).

Outputs (in --out_dir):
    bace_pyg_chebnet_tame_hps_results.csv   same schema as bace_preliminary_results.csv,
                                             Model="PyG ChebNet (TAME HPs)"
    bace_pyg_chebnet_tame_hps_meta.json     resolved backbone/recipe + provenance

Requires: uv sync --extra pyg
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
import deepchem as dc
from rdkit import RDLogger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from bace_preliminary_results import (  # noqa: E402
    SEEDS, CSV_FIELDS, _classification_metrics, _blank_row, run_pyg_graph_only,
)
from utils.molecular_graph import smiles_to_graph  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def load_bace(splitter):
    dc_cache_dir = os.path.join(project_root, "cache", "deepchem_data")
    os.makedirs(dc_cache_dir, exist_ok=True)
    _, datasets, _ = dc.molnet.load_bace_classification(
        featurizer="ECFP", splitter=splitter,
        frac_train=0.8, frac_valid=0.1, frac_test=0.1,
        data_dir=dc_cache_dir, save_dir=dc_cache_dir,
    )
    train_dc, valid_dc, test_dc = datasets
    train_smiles, val_smiles, test_smiles = list(train_dc.ids), list(valid_dc.ids), list(test_dc.ids)
    y_train = train_dc.y.astype(np.float32).reshape(-1)
    y_val = valid_dc.y.astype(np.float32).reshape(-1)
    y_test = test_dc.y.astype(np.float32).reshape(-1)

    def _graphs(smiles):
        out = []
        for smi in smiles:
            try:
                out.append(smiles_to_graph(smi))
            except Exception:
                out.append(None)
        return out

    graphs = (_graphs(train_smiles), _graphs(val_smiles), _graphs(test_smiles))
    ys = (y_train, y_val, y_test)
    sizes = {"train": len(train_smiles), "val": len(val_smiles), "test": len(test_smiles)}
    return graphs, ys, sizes


def backbone_and_recipe_from_tame(tame_params):
    bb = {
        "hidden": int(tame_params["hidden_channels"]),
        "K": int(tame_params["K"]),
        "L": int(tame_params["num_layers"]),
        "pool": tame_params.get("pool_type", "sum"),
        "set2set_n_iter": int(tame_params.get("set2set_n_iter", 4)),
    }
    recipe = {
        "lr": float(tame_params["lr"]),
        "weight_decay": float(tame_params["weight_decay"]),
        "batch_size": int(tame_params["batch_size"]),
        "encoder_lr_mult": 1.0,  # unused: PyG trains from scratch, no differential LR
        "label_smoothing": float(tame_params.get("label_smoothing", 0.0)),
    }
    return bb, recipe


def main():
    parser = argparse.ArgumentParser(description="PyG ChebNet on BACE, matched to TAME's own HPO recipe")
    parser.add_argument("--tame_params_json", type=str, required=True,
                        help="e.g. tame_predictor_best_params_v3_scaffold_s11.json")
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es_metric", type=str, choices=["val_loss", "val_auc"], default="val_auc")
    parser.add_argument("--n_seeds", type=int, default=100, help="How many of the 100 seeds to evaluate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    with open(args.tame_params_json) as f:
        tame_best = json.load(f)
    bb, recipe = backbone_and_recipe_from_tame(tame_best["params"])
    print(f"[Info] Matched TAME architecture: {bb}")
    print(f"[Info] Matched TAME recipe: {recipe}")

    print(f"[Info] Loading BACE ({args.splitter.title()} split)...")
    graphs, ys, sizes = load_bace(args.splitter)
    print(f"  train={sizes['train']} val={sizes['val']} test={sizes['test']}")
    y_train, y_val, y_test = ys

    seeds = SEEDS[: args.n_seeds]
    label = "PyG ChebNet (TAME HPs)"
    print(f"\n=== {label} (d{bb['hidden']} K{bb['K']} L{bb['L']} {bb['pool']}) ===")
    rows = []
    for seed in seeds:
        test_probs, val_probs, epoch = run_pyg_graph_only(
            bb, recipe, graphs, ys, device,
            max_epochs=args.max_epochs, patience=args.patience,
            es_metric=args.es_metric, seed=seed,
        )
        roc, pr, f1 = _classification_metrics(y_test, test_probs)
        val_roc, _, _ = _classification_metrics(y_val, val_probs)
        print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} Macro-F1={f1:.4f} (ep {epoch})")
        row = _blank_row(label, seed)
        row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1,
                    "Val_ROC_AUC": val_roc, "Stop_Epoch": epoch})
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "bace_pyg_chebnet_tame_hps_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    meta = {
        "splitter": args.splitter, "es_metric": args.es_metric, "seeds": seeds,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "backbone": bb, "recipe": recipe,
        "source_tame_params_json": os.path.abspath(args.tame_params_json),
    }
    meta_path = os.path.join(args.out_dir, "bace_pyg_chebnet_tame_hps_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
