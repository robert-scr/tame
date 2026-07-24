"""DEEB ablation on BACE: free vs. forced gate  x  RDKit vs. [RDKit || ECFP] descriptor expert.

Motivated by the modality analyses (see modality_complementarity_bace.py / text_modality_check_bace.py):
the RDKit descriptor expert tops out ~0.80 (below the graph), so forcing it into the mix drags the
graph down; and a naive/forced blend hurts most exactly when the graph is strong. Two levers:

  Task 1 (gate) --  free:   gate_entropy_weight=0, gate_balance_weight=0  (router weights per-coordinate,
                            can down-weight a weak modality)
                    forced: gate_entropy_weight=0.05, balanced target      (mirrors bace_preliminary_results,
                            every modality forced to contribute a floor)
  Task 2 (feat) --  rdkit:       DEEB consumes the 217-d RDKit descriptors (orthogonal physicochemistry)
                    rdkit_ecfp:  DEEB consumes [RDKit || ECFP(1024)] -- raises the branch's ceiling with a
                                 structural fingerprint, still fully differentiable (NO random forest).

Runs the full factorial (default 2x2) on the identical BACE scaffold split, reusing the TAME training +
gate-statistics machinery from bace_preliminary_results.py. Uses a pretrained ChebNet checkpoint when
available (--ckpt_dir); otherwise trains the encoder from scratch (weak-graph regime -- fine for a laptop
smoke test, but the strong-graph story needs the checkpoint).

Output (in --out_dir):
    deeb_ablation_bace.csv   per-seed test metrics + gate stats, one block per (gate_mode, desc_features) arm

Example:
    uv run python scripts/deeb_ablation_bace.py --ckpt_dir cache/pretrained_checkpoints --n_seeds 20
    uv run python scripts/deeb_ablation_bace.py --gate_modes free --desc_features rdkit rdkit_ecfp
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
for p in (SCRIPTS_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines_bace as bb                 # noqa: E402  (identical split + descriptor helpers)
import bace_preliminary_results as bp       # noqa: E402  (fit_full_model, gate stats, CSV schema)
from models.tame_predictor import TAMEPredictorConfig  # noqa: E402
from utils.embedding_cache import EfficientEmbeddingCache  # noqa: E402
from utils.molecular_graph import smiles_to_graph  # noqa: E402

# gate_mode -> gate regularisation preset. target is moot when balance_weight=0 (kept for clarity).
GATE_PRESETS = {
    "free":   dict(gate_entropy_weight=0.0,  gate_balance_weight=0.0,  gate_target=(0.8, 0.1, 0.1)),
    "forced": dict(gate_entropy_weight=0.05, gate_balance_weight=0.01, gate_target=(0.34, 0.33, 0.33)),
}


def load_text(split_smiles):
    emb_path = os.path.join(PROJECT_ROOT, "cache", "cot_embeddings",
                            "binding_fast_text_embeddings_compact.npz")
    cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    return [np.asarray(cache.get_batch(s), dtype=np.float32) for s in split_smiles]


def build_config(args, gate_mode, desc_dim, text_dim):
    g = GATE_PRESETS[gate_mode]
    return TAMEPredictorConfig(
        task="classification", num_tasks=1,
        hidden_channels=args.hidden, K=args.K, num_layers=args.num_layers,
        pool=args.pool, set2set_processing_steps=args.set2set_n_iter,
        text_embedding_dim=int(text_dim), descriptor_dim=int(desc_dim),
        fusion_hidden_dim=args.proj_dim,
        projection_dropout=args.projection_dropout, router_dropout=args.router_dropout,
        head_hidden_dim=args.head_hidden_dim, head_dropout=args.head_dropout,
        gate_balance_weight=g["gate_balance_weight"], gate_entropy_weight=g["gate_entropy_weight"],
        gate_target=g["gate_target"], desc_modality_dropout=args.desc_modality_dropout,
        label_smoothing=args.label_smoothing,
        descriptor_standardize=True,
        descriptor_winsorize_lower_q=0.01, descriptor_winsorize_upper_q=0.99,
    )


def main():
    parser = argparse.ArgumentParser(description="DEEB ablation: gate mode x descriptor features on BACE")
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--gate_modes", nargs="+", choices=list(GATE_PRESETS), default=["forced", "free"])
    parser.add_argument("--desc_features", nargs="+", choices=["rdkit", "rdkit_ecfp"],
                        default=["rdkit", "rdkit_ecfp"])
    parser.add_argument("--n_seeds", type=int, default=20)
    # Pretrained ChebNet checkpoint (optional; trains from scratch if absent).
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--ckpt_epoch_tag", type=int, default=25)
    # TAME architecture / recipe (defaults mirror the generic ChebNet recipe; override to match a checkpoint).
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pool", type=str, choices=["sum", "mean", "set2set"], default="mean")
    parser.add_argument("--set2set_n_iter", type=int, default=4)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--projection_dropout", type=float, default=0.1)
    parser.add_argument("--router_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_dim", type=int, default=64)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--desc_modality_dropout", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encoder_lr_mult", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es_metric", type=str, choices=["val_loss", "val_auc"], default="val_loss")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = bb.SEEDS[:args.n_seeds]
    print(f"[Info] Device: {device} | seeds: {len(seeds)} | arms: "
          f"{args.gate_modes} x {args.desc_features}")

    # ---- Data (identical split) ----
    print(f"[Info] Loading BACE ({args.splitter.title()} Split)...")
    train_dc, valid_dc, test_dc = bb.load_bace(args.splitter)
    train_smiles, val_smiles, test_smiles = list(train_dc.ids), list(valid_dc.ids), list(test_dc.ids)
    y_train = train_dc.y.reshape(-1).astype(np.float32)
    y_val = valid_dc.y.reshape(-1).astype(np.float32)
    y_test = test_dc.y.reshape(-1).astype(np.float32)
    print(f"  train={len(train_smiles)} val={len(val_smiles)} test={len(test_smiles)}")

    print("[Info] Loading text embeddings...")
    text_train, text_val, text_test = load_text([train_smiles, val_smiles, test_smiles])
    text_dim = int(text_train.shape[1])

    print("[Info] Computing RDKit descriptors (imputed)...")
    rk_tr, rk_va, rk_te = bb.impute_with_train_median(
        bb.compute_rdkit_descriptors(train_smiles),
        bb.compute_rdkit_descriptors(val_smiles),
        bb.compute_rdkit_descriptors(test_smiles))

    def desc_for(kind):
        if kind == "rdkit":
            return rk_tr, rk_va, rk_te
        # [RDKit(imputed) || ECFP bits]; predictor winsorize is ~identity on bits, standardize is fine.
        ecfp = (np.asarray(train_dc.X, np.float32), np.asarray(valid_dc.X, np.float32),
                np.asarray(test_dc.X, np.float32))
        return (np.concatenate([rk_tr, ecfp[0]], axis=1),
                np.concatenate([rk_va, ecfp[1]], axis=1),
                np.concatenate([rk_te, ecfp[2]], axis=1))

    # ---- Optional pretrained encoder ----
    enc_state, ckpt = None, None
    if args.ckpt_dir:
        ckpt = os.path.join(args.ckpt_dir,
                            f"chebnet_pt_d{args.hidden}_K{args.K}_L{args.num_layers}_e{args.ckpt_epoch_tag}.pt")
        if os.path.exists(ckpt):
            enc_state = bp.extract_encoder_state(ckpt)
            print(f"[Info] Loaded pretrained encoder: {ckpt}")
        else:
            print(f"[Warn] Checkpoint not found ({ckpt}); training encoder from scratch (weak-graph regime).")
    else:
        print("[Warn] No --ckpt_dir; training encoder from scratch (weak-graph regime).")

    params = {"batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.wd,
              "encoder_lr_mult": args.encoder_lr_mult}

    # ---- Reference ChebNet graph-only arm (matched checkpoint + recipe) ----
    # Only meaningful with a real checkpoint; run_graph_only reloads it, so skip when absent.
    run_ref = enc_state is not None
    if run_ref:
        graph_bb = {"hidden": args.hidden, "K": args.K, "L": args.num_layers, "pool": args.pool,
                    "set2set_n_iter": args.set2set_n_iter,
                    "ckpt_name": os.path.basename(ckpt), "ckpt_path": ckpt}
        graph_recipe = bp.recipe_of(None, lr=args.lr, wd=args.wd, batch_size=args.batch_size,
                                    encoder_lr_mult=args.encoder_lr_mult, label_smoothing=args.label_smoothing)

        def _graphs(sm):
            out = []
            for s in sm:
                try:
                    out.append(smiles_to_graph(s))
                except Exception:
                    out.append(None)
            return out
        ref_graphs = (_graphs(train_smiles), _graphs(val_smiles), _graphs(test_smiles))
        ref_ys = (y_train, y_val, y_test)

    rows = []
    arm_preds = {}  # arm label -> list of per-seed test-prob arrays (seed-averaged at the end)

    # ---- (0) ChebNet graph-only reference (the anchor for "recovered ChebNet's level") ----
    if run_ref:
        ref_label = "ChebNet (graph-only, pretrained)"
        print(f"\n=== {ref_label}  (recipe: {graph_recipe}) ===")
        for seed in seeds:
            test_probs, val_probs, epoch = bp.run_graph_only(
                graph_bb, graph_recipe, ref_graphs, ref_ys, device,
                max_epochs=args.max_epochs, patience=args.patience, es_metric=args.es_metric, seed=seed)
            roc, pr, f1 = bp._classification_metrics(y_test, test_probs)
            val_roc, _, _ = bp._classification_metrics(y_val, val_probs)
            print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} F1={f1:.4f} (ep {epoch})")
            row = bp._blank_row(ref_label, seed)
            row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1,
                        "Val_ROC_AUC": val_roc, "Stop_Epoch": epoch})
            rows.append(row)
            arm_preds.setdefault(ref_label, []).append(test_probs)
    for feat_kind in args.desc_features:
        d_tr, d_va, d_te = desc_for(feat_kind)
        data = {
            "train_smiles": train_smiles, "val_smiles": val_smiles, "test_smiles": test_smiles,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "text_train": text_train, "text_val": text_val, "text_test": text_test,
            "desc_train": d_tr, "desc_val": d_va, "desc_test": d_te,
        }
        for gate_mode in args.gate_modes:
            arm = f"TAME[{gate_mode},{feat_kind}]"
            config = build_config(args, gate_mode, int(d_tr.shape[1]), text_dim)
            print(f"\n=== {arm}  (desc_dim={d_tr.shape[1]}, entropy_w={config.gate_entropy_weight}, "
                  f"balance_w={config.gate_balance_weight}) ===")
            for seed in seeds:
                test_probs, val_probs, history, (ent, gmin, gmeans) = bp.fit_full_model(
                    "tame", config, params, enc_state, data, device,
                    max_epochs=args.max_epochs, patience=args.patience,
                    es_metric=args.es_metric, seed=seed)
                roc, pr, f1 = bp._classification_metrics(y_test, test_probs)
                val_roc, _, _ = bp._classification_metrics(y_val, val_probs)
                gm = gmeans or [float("nan")] * 3
                print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} F1={f1:.4f} "
                      f"| gates(g/t/d)={gm[0]:.2f}/{gm[1]:.2f}/{gm[2]:.2f} ent={ent:.3f}")
                row = bp._blank_row(arm, seed)
                row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1,
                            "Val_ROC_AUC": val_roc, "Stop_Epoch": len(history["val_loss"]),
                            "Gate_Entropy_Mean": ent, "Gate_Min_Mean": gmin,
                            "Gate_Graph": gm[0], "Gate_Text": gm[1], "Gate_Desc": gm[2]})
                rows.append(row)
                arm_preds.setdefault(arm, []).append(np.asarray(test_probs, dtype=np.float64))

    csv_path = os.path.join(args.out_dir, "deeb_ablation_bace.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bp.CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    # ---- Per-molecule seed-averaged test predictions (for tail-rescue / complementarity analysis) ----
    arms_order = list(arm_preds.keys())
    avg = {a: np.mean(np.stack(v), axis=0) for a, v in arm_preds.items()}
    pred_path = os.path.join(args.out_dir, "deeb_ablation_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "y"] + [f"p[{a}]" for a in arms_order])
        for i, smi in enumerate(test_smiles):
            w.writerow([smi, int(y_test[i])] + [f"{avg[a][i]:.6f}" for a in arms_order])
    print(f"Saved per-molecule predictions to {pred_path}")

    # ---- Run metadata (arch / checkpoint / recipe) so results are self-describing ----
    meta = {
        "splitter": args.splitter,
        "split_sizes": {"train": len(train_smiles), "val": len(val_smiles), "test": len(test_smiles)},
        "seeds": [int(s) for s in seeds],
        "es_metric": args.es_metric, "max_epochs": args.max_epochs, "patience": args.patience,
        "arch": {"hidden": args.hidden, "K": args.K, "num_layers": args.num_layers,
                 "pool": args.pool, "set2set_n_iter": args.set2set_n_iter},
        "checkpoint": {"dir": args.ckpt_dir, "epoch_tag": args.ckpt_epoch_tag,
                       "path": ckpt, "loaded_pretrained": bool(run_ref)},
        "recipe": {"lr": args.lr, "weight_decay": args.wd, "batch_size": args.batch_size,
                   "encoder_lr_mult": args.encoder_lr_mult, "label_smoothing": args.label_smoothing},
        "gate_presets": {m: GATE_PRESETS[m] for m in args.gate_modes},
        "desc_features": args.desc_features,
        "descriptor_dims": {"rdkit": int(rk_tr.shape[1]),
                            "rdkit_ecfp": int(rk_tr.shape[1] + np.asarray(train_dc.X).shape[1])},
        "reference_arm": ("ChebNet (graph-only, pretrained)" if run_ref else None),
    }
    with open(os.path.join(args.out_dir, "deeb_ablation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {os.path.join(args.out_dir, 'deeb_ablation_meta.json')}")

    # ---- Per-arm summary ----
    print("\n=== Summary (Test ROC-AUC) ===")
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["Model"], []).append(r["Test_ROC_AUC"])
    for arm, vals in by_arm.items():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        if len(v):
            q1, q3 = np.percentile(v, [25, 75])
            print(f"  {arm:26s} median={np.median(v):.4f}  IQR=[{q1:.4f}, {q3:.4f}]  "
                  f"mean={v.mean():.4f}±{v.std():.4f}  min={v.min():.4f}  n={len(v)}")


if __name__ == "__main__":
    main()
