"""Can the LLM/text branch contribute on BACE? Standalone signal + complementarity + scrambled control.

The paper's text expert is a frozen CoT embedding (text-embedding-3-large, 3072-d). This probes how much
molecule-specific signal it carries and whether it adds anything on top of graph + descriptor:

  1. Standalone AUC of a probe on the text embeddings (linear = LogReg; nonlinear = RF, seed-averaged).
  2. SCRAMBLED-TEXT control (risk-assessment F8): permute the text<->molecule alignment on train and refit.
     If scrambled AUC ~ real AUC, the channel carries no molecule-specific signal (it's a regularizer/noise),
     not decodable chemistry -> the honest conclusion is "text does not contribute".
  3. Complementarity vs the graph and descriptor views, reusing the per-molecule test predictions saved by
     modality_complementarity_bace.py (p_graph, p_ecfp, p_desc), plus 3-way mean-ensemble gains.

Run modality_complementarity_bace.py first (it writes the predictions CSV this reads).

Example:
    uv run python scripts/text_modality_check_bace.py --n_seeds 10
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
for p in (SCRIPTS_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines_bace as bb                    # noqa: E402  (identical split + helpers)
import modality_complementarity_bace as mc     # noqa: E402  (pair_analysis, rf_probs_avg)
from utils.embedding_cache import EfficientEmbeddingCache  # noqa: E402


def logreg_probs(X_train, y_train, X_eval):
    """Standardized linear probe: deterministic positive-class probabilities."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(scaler.transform(X_train), y_train.astype(int))
    return bb._proba_pos(clf, scaler.transform(X_eval)).astype(np.float32)


def _auc(p, y):
    return bb._classification_metrics(y, p)[0]


def load_saved_preds(pred_csv, test_smiles):
    """Load p_graph/p_ecfp/p_desc from the complementarity CSV, aligned to test_smiles order."""
    by_smi = {}
    with open(pred_csv) as f:
        for row in csv.DictReader(f):
            by_smi[row["smiles"]] = row
    miss = [s for s in test_smiles if s not in by_smi]
    if miss:
        raise SystemExit(f"{len(miss)} test molecules missing from {pred_csv}; re-run "
                         f"modality_complementarity_bace.py with matching --splitter.")
    g = np.array([float(by_smi[s]["p_graph"]) for s in test_smiles], dtype=np.float32)
    e = np.array([float(by_smi[s]["p_ecfp"]) for s in test_smiles], dtype=np.float32)
    d = np.array([float(by_smi[s]["p_desc"]) for s in test_smiles], dtype=np.float32)
    return g, e, d


def main():
    parser = argparse.ArgumentParser(description="LLM/text modality check on BACE")
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--pred_csv", type=str,
                        default="benchmarking/results/modality_complementarity_bace.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = bb.SEEDS[:args.n_seeds]

    print(f"[Info] Loading BACE ({args.splitter.title()} Split)...")
    train_dc, valid_dc, test_dc = bb.load_bace(args.splitter)
    train_smiles, test_smiles = list(train_dc.ids), list(test_dc.ids)
    y_tr = train_dc.y.reshape(-1).astype(np.float32)
    y = test_dc.y.reshape(-1).astype(np.float32)

    print("[Info] Loading CoT text embeddings...")
    emb_path = os.path.join(PROJECT_ROOT, "cache", "cot_embeddings",
                            "binding_fast_text_embeddings_compact.npz")
    cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    txt_tr = np.asarray(cache.get_batch(train_smiles), dtype=np.float32)
    txt_te = np.asarray(cache.get_batch(test_smiles), dtype=np.float32)
    print(f"  text embedding dim: {txt_tr.shape[1]}  train={txt_tr.shape[0]} test={txt_te.shape[0]}")

    # ---- 1. Standalone text signal (linear + nonlinear probes) ----
    p_text_lin = logreg_probs(txt_tr, y_tr, txt_te)
    p_text_rf = mc.rf_probs_avg("rf", txt_tr, y_tr, txt_te, seeds)
    auc_lin, auc_rf = _auc(p_text_lin, y), _auc(p_text_rf, y)

    # ---- 2. Scrambled-text control (break molecule<->text alignment on train) ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(txt_tr))
    auc_scrambled = _auc(logreg_probs(txt_tr[perm], y_tr, txt_te), y)

    # ---- 3. Complementarity vs graph/descriptor (reuse saved per-molecule preds) ----
    p_graph, p_ecfp, p_desc = load_saved_preds(args.pred_csv, test_smiles)
    p_text = p_text_rf  # use the nonlinear probe as the "text expert" proxy, consistent with RF-desc/ecfp

    pairs = [
        mc.pair_analysis("PyG-ChebConv (graph)", p_graph, "RF-text (LLM)", p_text, y),
        mc.pair_analysis("RF-RDKit (desc)", p_desc, "RF-text (LLM)", p_text, y),
    ]

    def ens(*ps):
        return _auc(np.mean(ps, axis=0), y)

    ensembles = {
        "graph": _auc(p_graph, y),
        "graph+text": ens(p_graph, p_text),
        "graph+desc": ens(p_graph, p_desc),
        "graph+desc+text": ens(p_graph, p_desc, p_text),
    }

    summary = {
        "text_embedding_dim": int(txt_tr.shape[1]),
        "standalone": {"logreg_linear": auc_lin, "rf_nonlinear": auc_rf,
                       "scrambled_logreg": auc_scrambled},
        "complementarity_pairs": pairs,
        "ensemble_aucs": ensembles,
    }
    with open(os.path.join(args.out_dir, "text_modality_check_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Report ----
    print("\n=== Standalone text signal (test ROC-AUC) ===")
    print(f"  linear probe (LogReg): {auc_lin:.4f}")
    print(f"  nonlinear probe (RF):  {auc_rf:.4f}")
    print(f"  SCRAMBLED control:     {auc_scrambled:.4f}   "
          f"(real - scrambled = {auc_rf - auc_scrambled:+.4f}; ~0 => no molecule-specific signal)")

    print("\n=== Text complementarity ===")
    for r in pairs:
        print(f"  {r['structural']} ({r['auc_structural']:.3f}) + {r['descriptor']} ({r['auc_descriptor']:.3f})"
              f" -> mean-ensemble {r['auc_ensemble_mean']:.3f} ({r['ensemble_gain_vs_structural']:+.3f})"
              f" | prob r={r['prob_pearson']:.2f}")

    print("\n=== Does text add on top of graph(+desc)? (mean-ensemble AUC) ===")
    for k, v in ensembles.items():
        print(f"  {k:18s} {v:.4f}")
    print(f"\nSaved summary to {os.path.join(args.out_dir, 'text_modality_check_summary.json')}")


if __name__ == "__main__":
    main()
