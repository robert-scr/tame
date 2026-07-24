"""Graph vs. descriptor complementarity on the shared BACE test set.

Tests the two claims the TAME "anchoring" thesis rests on, using per-molecule test predictions from
models trained on the identical scaffold split:

  1. "Not completely orthogonal" — are the structural (graph) and descriptor (physicochemistry) views
     correlated enough to blend, but not so correlated they're redundant? (prob + error correlation)
  2. "The descriptor anchor catches the graph's failures" — on the molecules the structural model gets
     (confidently) wrong, how often is the descriptor model right? Does a plain mean-ensemble beat the
     structural model alone? What's the oracle ceiling a perfect router could reach?

Because no pretrained ChebNet checkpoint ships here, "the graph" is represented two ways, both trained on
the same split and seed-averaged for stable per-molecule probabilities:
  * PyG-ChebConv (random init)  — an actual graph model (weaker, ~0.76)
  * RF on ECFP                  — a strong *topological* proxy (~0.865), same connectivity information class
The descriptor view is RF on the 217-d RDKit vector (~0.80), i.e. exactly what DEEB consumes.

Outputs (in --out_dir):
    modality_complementarity_bace.csv       per-molecule: smiles, y, p_graph, p_ecfp, p_desc
    modality_complementarity_summary.json   all pairwise metrics

Example:
    uv run python scripts/modality_complementarity_bace.py --n_seeds 10
"""

import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (SCRIPTS_DIR, os.path.dirname(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines_bace as bb  # noqa: E402  (reuses the identical split / training path)


def rf_probs_avg(algo, X_train, y_train, X_eval, seeds):
    """Seed-averaged positive-class probabilities for a classical model (stable per-molecule estimate)."""
    acc = np.zeros(X_eval.shape[0], dtype=np.float64)
    for seed in seeds:
        clf = bb._make_classifier(algo, seed)
        clf.fit(X_train, y_train.astype(int))
        acc += bb._proba_pos(clf, X_eval)
    return (acc / len(seeds)).astype(np.float32)


def pyg_probs_avg(train_dc, valid_dc, test_dc, args, seeds):
    """Seed-averaged PyG-ChebConv test probabilities (ensemble over random inits)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_g, _ = bb._build_pyg_graphs(list(train_dc.ids), train_dc.y.reshape(-1))
    val_g, y_val = bb._build_pyg_graphs(list(valid_dc.ids), valid_dc.y.reshape(-1))
    test_g, _ = bb._build_pyg_graphs(list(test_dc.ids), test_dc.y.reshape(-1))
    in_channels = train_g[0].x.size(1)
    acc = np.zeros(len(test_g), dtype=np.float64)
    for seed in seeds:
        test_probs, _, ep = bb.train_pyg_seed(train_g, val_g, test_g, y_val, in_channels, args, seed, device)
        acc += test_probs
        print(f"    [pyg] seed {seed}: ep {ep}")
    return (acc / len(seeds)).astype(np.float32)


def _acc(p, y):
    return float(np.mean((p >= 0.5).astype(int) == y))


def pair_analysis(name_struct, p_struct, name_desc, p_desc, y):
    """All complementarity metrics for one (structural, descriptor) pair on the shared test set."""
    pred_s, pred_d = (p_struct >= 0.5).astype(int), (p_desc >= 0.5).astype(int)
    err_s, err_d = (pred_s != y), (pred_d != y)              # 0/1 error indicators
    margin_s = np.abs(p_struct - 0.5)                         # confidence of the structural model

    # Ensemble (equal-weight mean of probabilities) — the simplest thing a router could learn.
    p_ens = 0.5 * (p_struct + p_desc)

    # Descriptor "rescue" of structural errors.
    wrong = np.where(err_s)[0]
    conf_wrong = wrong[margin_s[wrong] >= np.median(margin_s[wrong])] if len(wrong) else wrong

    return {
        "structural": name_struct,
        "descriptor": name_desc,
        "auc_structural": bb._classification_metrics(y, p_struct)[0],
        "auc_descriptor": bb._classification_metrics(y, p_desc)[0],
        "auc_ensemble_mean": bb._classification_metrics(y, p_ens)[0],
        "ensemble_gain_vs_structural": bb._classification_metrics(y, p_ens)[0] - bb._classification_metrics(y, p_struct)[0],
        "acc_structural": _acc(p_struct, y),
        "acc_descriptor": _acc(p_desc, y),
        "acc_ensemble_mean": _acc(p_ens, y),
        # Correlation of predictions (moderate positive = complementary, not redundant / not orthogonal).
        "prob_pearson": float(pearsonr(p_struct, p_desc)[0]),
        "prob_spearman": float(spearmanr(p_struct, p_desc)[0]),
        # Correlation of *errors* (low = the two views fail on different molecules — key for anchoring).
        "signed_error_pearson": float(pearsonr(p_struct - y, p_desc - y)[0]),
        "error_indicator_phi": float(pearsonr(err_s.astype(float), err_d.astype(float))[0]),
        # Rescue rates.
        "n_structural_errors": int(err_s.sum()),
        "desc_rescue_rate_on_errors": float(np.mean(~err_d[wrong])) if len(wrong) else float("nan"),
        "n_confident_structural_errors": int(len(conf_wrong)),
        "desc_rescue_rate_on_confident_errors": float(np.mean(~err_d[conf_wrong])) if len(conf_wrong) else float("nan"),
        # Disagreement + oracle ceiling.
        "disagreement_rate": float(np.mean(pred_s != pred_d)),
        "oracle_at_least_one_correct": float(np.mean(~err_s | ~err_d)),
    }


def main():
    parser = argparse.ArgumentParser(description="Graph vs descriptor complementarity on BACE test set")
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--n_seeds", type=int, default=10, help="Seeds to average per model (stability).")
    # PyG recipe (mirror the benchmark defaults).
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pool", type=str, choices=["sum", "mean", "set2set"], default="mean")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es_metric", type=str, choices=["val_loss", "val_auc"], default="val_loss")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = bb.SEEDS[:args.n_seeds]

    print(f"[Info] Loading BACE ({args.splitter.title()} Split)...")
    train_dc, valid_dc, test_dc = bb.load_bace(args.splitter)
    test_smiles = list(test_dc.ids)
    y = test_dc.y.reshape(-1).astype(np.float32)
    print(f"  test molecules: {len(test_smiles)}  (seeds averaged per model: {len(seeds)})")

    # ---- Features ----
    ecfp_tr, _, ecfp_te = (np.asarray(train_dc.X, np.float32), None, np.asarray(test_dc.X, np.float32))
    y_tr = train_dc.y.reshape(-1).astype(np.float32)
    print("[Info] Computing RDKit descriptors...")
    d_tr, _, d_te = bb.impute_with_train_median(
        bb.compute_rdkit_descriptors(list(train_dc.ids)),
        bb.compute_rdkit_descriptors(list(valid_dc.ids)),
        bb.compute_rdkit_descriptors(test_smiles))

    # ---- Per-molecule test probabilities (seed-averaged) ----
    print("[Info] RF-ECFP (structural proxy)...")
    p_ecfp = rf_probs_avg("rf", ecfp_tr, y_tr, ecfp_te, seeds)
    print("[Info] RF-RDKit (descriptor / DEEB features)...")
    p_desc = rf_probs_avg("rf", d_tr, y_tr, d_te, seeds)
    print("[Info] PyG-ChebConv (graph model)...")
    p_graph = pyg_probs_avg(train_dc, valid_dc, test_dc, args, seeds)

    assert len(p_ecfp) == len(p_desc) == len(p_graph) == len(y), "test-set alignment mismatch"

    # ---- Save per-molecule predictions ----
    import csv
    pred_path = os.path.join(args.out_dir, "modality_complementarity_bace.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "y", "p_graph", "p_ecfp", "p_desc"])
        for i, smi in enumerate(test_smiles):
            w.writerow([smi, int(y[i]), f"{p_graph[i]:.6f}", f"{p_ecfp[i]:.6f}", f"{p_desc[i]:.6f}"])

    # ---- Analyses: descriptor vs each structural view ----
    results = [
        pair_analysis("PyG-ChebConv (graph)", p_graph, "RF-RDKit (desc)", p_desc, y),
        pair_analysis("RF-ECFP (structural proxy)", p_ecfp, "RF-RDKit (desc)", p_desc, y),
    ]
    summary = {"n_test": len(y), "n_seeds_averaged": len(seeds), "pairs": results}
    with open(os.path.join(args.out_dir, "modality_complementarity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Report ----
    for r in results:
        print(f"\n=== {r['structural']}  vs  {r['descriptor']} ===")
        print(f"  AUC:  structural={r['auc_structural']:.4f}  descriptor={r['auc_descriptor']:.4f}  "
              f"mean-ensemble={r['auc_ensemble_mean']:.4f}  (gain vs structural {r['ensemble_gain_vs_structural']:+.4f})")
        print(f"  Acc:  structural={r['acc_structural']:.4f}  descriptor={r['acc_descriptor']:.4f}  "
              f"mean-ensemble={r['acc_ensemble_mean']:.4f}")
        print(f"  Prob correlation:  Pearson={r['prob_pearson']:.3f}  Spearman={r['prob_spearman']:.3f}   "
              f"(moderate + = complementary, not orthogonal / not redundant)")
        print(f"  Error correlation: signed={r['signed_error_pearson']:.3f}  phi={r['error_indicator_phi']:.3f}   "
              f"(low = the two fail on DIFFERENT molecules)")
        print(f"  Descriptor rescue: {r['desc_rescue_rate_on_errors']:.3f} of {r['n_structural_errors']} "
              f"structural errors  |  {r['desc_rescue_rate_on_confident_errors']:.3f} of "
              f"{r['n_confident_structural_errors']} *confident* structural errors")
        print(f"  Disagreement rate: {r['disagreement_rate']:.3f}   "
              f"Oracle (>=1 correct): {r['oracle_at_least_one_correct']:.4f}")

    print(f"\nSaved per-molecule predictions to {pred_path}")
    print(f"Saved summary to {os.path.join(args.out_dir, 'modality_complementarity_summary.json')}")


if __name__ == "__main__":
    main()
