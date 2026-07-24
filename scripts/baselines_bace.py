"""External baselines for the BACE benchmark (addresses reviewer concerns B5/B7, F2/F3).

Two model families, run inside the *identical* pipeline as ``scripts/bace_preliminary_results.py``
(same DeepChem scaffold split, same fixed test set, same ``_classification_metrics``), so their
numbers are directly comparable / concatenable with ``bace_preliminary_results.csv``:

    F2  RF + Logistic Regression on ECFP fingerprints   (classical cheminformatics baselines, CPU)
    F3  Off-the-shelf PyG ``ChebConv``, random init      (independent GNN impl, no PubChem pretraining)

Why these:
  * F2 gives external, non-neural baselines. If RF/LogReg on ECFP land in the ChebNet CI, the paper's
    accuracy claims are put in context rather than left for a reviewer to find (see risk assessment B7).
  * F3 isolates two confounds at once (B5, B7): it uses an *independent* Chebyshev conv implementation
    (validating the in-house ``ChebLayer``) and is *not* pretrained on ~944k PubChem molecules, so the
    gap to the headline "ChebNet (generic)" arm attributes value to the pretraining vs. the architecture.
    PyG ``ChebConv(normalization='sym')`` defaults to ``lambda_max=2.0`` and uses the same node
    features (28-d) as the repo encoder, so the only differences are implementation + pretraining.

Outputs (in ``--out_dir``):
    baselines_bace_results.csv   long-form per-seed metrics, same schema as bace_preliminary_results.csv

Examples:
    uv run python scripts/baselines_bace.py                       # all baselines, scaffold split
    uv run python scripts/baselines_bace.py --models rf_ecfp logreg_ecfp   # classical only, CPU, minutes
    uv run python scripts/baselines_bace.py --models pyg_cheb --hidden 128 --K 3 --num_layers 3
"""

import os
import sys
import io
import csv
import copy
import json
import argparse
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import deepchem as dc
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Same 100 seeds as bace_preliminary_results.py so rows align / concatenate cleanly.
SEEDS = np.random.default_rng(42).choice(range(1, 1000), size=100, replace=False).tolist()

# Superset schema shared with bace_preliminary_results.py (gate columns stay NaN for baselines).
CSV_FIELDS = [
    "Model", "Seed", "Test_ROC_AUC", "Test_PR_AUC", "Test_Macro_F1", "Val_ROC_AUC",
    "Gate_Graph", "Gate_Text", "Gate_Desc", "Gate_SEG",
    "Gate_Entropy_Mean", "Gate_Min_Mean", "Stop_Epoch",
]


# ---------------------------------------------------------------------------
# Helpers (verbatim from bace_preliminary_results.py so results are comparable)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    row = {k: float("nan") for k in CSV_FIELDS}
    row["Model"] = model
    row["Seed"] = int(seed)
    return row


# ---------------------------------------------------------------------------
# BACE loading (identical args to bace_preliminary_results.py -> identical split)
# ---------------------------------------------------------------------------
def load_bace(splitter):
    """Return the DeepChem BACE datasets (train, valid, test) with ECFP features attached.

    Uses exactly the same call as the headline benchmark, so the deterministic ScaffoldSplitter
    yields the same train/valid/test molecules and the same fixed test set.
    """
    dc_cache_dir = os.path.join(project_root, "cache", "deepchem_data")
    os.makedirs(dc_cache_dir, exist_ok=True)
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        _, datasets, _ = dc.molnet.load_bace_classification(
            featurizer="ECFP", splitter=splitter,
            frac_train=0.8, frac_valid=0.1, frac_test=0.1,
            data_dir=dc_cache_dir, save_dir=dc_cache_dir,
        )
    return datasets


# ---------------------------------------------------------------------------
# F2 - classical baselines on ECFP fingerprints
# ---------------------------------------------------------------------------
def _proba_pos(clf, X):
    """Positive-class probabilities, robust to single-class edge cases."""
    proba = clf.predict_proba(X)
    if proba.shape[1] == 1:  # only one class seen in training
        return np.full(X.shape[0], float(clf.classes_[0]), dtype=np.float32)
    pos_col = list(clf.classes_).index(1)
    return proba[:, pos_col].astype(np.float32)


def run_classical(model_name, train_dc, valid_dc, test_dc, seeds):
    X_train, y_train = np.asarray(train_dc.X, np.float32), train_dc.y.reshape(-1).astype(np.float32)
    X_val, y_val = np.asarray(valid_dc.X, np.float32), valid_dc.y.reshape(-1).astype(np.float32)
    X_test, y_test = np.asarray(test_dc.X, np.float32), test_dc.y.reshape(-1).astype(np.float32)
    print(f"\n=== {model_name} on ECFP (dim={X_train.shape[1]}) ===")

    rows = []
    for seed in seeds:
        if model_name == "rf_ecfp":
            clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=int(seed))
        elif model_name == "logreg_ecfp":
            # ECFP is high-dim sparse-binary; L2 logistic regression is the standard linear baseline.
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=int(seed))
        else:
            raise ValueError(f"Unknown classical model: {model_name}")

        clf.fit(X_train, y_train.astype(int))
        roc, pr, f1 = _classification_metrics(y_test, _proba_pos(clf, X_test))
        val_roc, _, _ = _classification_metrics(y_val, _proba_pos(clf, X_val))
        print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} Macro-F1={f1:.4f}")

        row = _blank_row(model_name, seed)
        row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1, "Val_ROC_AUC": val_roc})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# F3 - off-the-shelf PyG ChebConv, randomly initialised
# ---------------------------------------------------------------------------
def _build_pyg_graphs(smiles_list, y):
    """SMILES -> list of PyG Data(x, edge_index, edge_weight, y); None entries dropped, y realigned."""
    import torch
    from torch_geometric.data import Data
    from utils.molecular_graph import smiles_to_graph

    graphs, kept_y = [], []
    for smi, yi in zip(smiles_list, y):
        try:
            g = smiles_to_graph(smi)
        except Exception:
            g = None
        if g is None:
            continue
        graphs.append(Data(
            x=torch.from_numpy(g.X).float(),
            edge_index=torch.from_numpy(g.edge_index).long(),
            edge_weight=torch.from_numpy(g.edge_weight).float(),
            y=torch.tensor([float(yi)], dtype=torch.float32),
        ))
        kept_y.append(float(yi))
    return graphs, np.asarray(kept_y, dtype=np.float32)


def _make_pyg_model(in_channels, hidden, K, num_layers, pool, dropout):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import ChebConv, global_add_pool, global_mean_pool, Set2Set

    class PyGChebNet(nn.Module):
        """Independent Chebyshev-conv baseline: PyG ChebConv x L -> pooling -> MLP head."""

        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                in_c = in_channels if i == 0 else hidden
                # normalization='sym' + default lambda_max=2.0 matches the repo ChebLayer.
                self.convs.append(ChebConv(in_c, hidden, K=K, normalization="sym"))
            self.dropout = dropout
            if pool == "set2set":
                self.pool = Set2Set(hidden, processing_steps=4)
                head_in = 2 * hidden
            else:
                self.pool = global_add_pool if pool == "sum" else global_mean_pool
                head_in = hidden
            hidden_head = max(head_in // 2, 1)
            self.head = nn.Sequential(
                nn.Linear(head_in, hidden_head), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_head, 1)
            )

        def forward(self, x, edge_index, edge_weight, batch):
            h = x
            for conv in self.convs:
                h = conv(h, edge_index, edge_weight, batch)
                h = torch.relu(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            g = self.pool(h, batch)  # global_add/mean_pool and Set2Set share the (x, batch) signature
            return self.head(g).view(-1)

    return PyGChebNet()


def _pyg_predict(model, loader, device, n):
    import torch
    model.eval()
    probs = np.full(n, np.nan, dtype=np.float32)
    pos = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)).cpu().numpy()
            probs[pos:pos + len(p)] = p
            pos += len(p)
    return probs


def run_pyg_cheb(train_dc, valid_dc, test_dc, seeds, args):
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== PyG ChebConv (random init) d{args.hidden} K{args.K} L{args.num_layers} "
          f"{args.pool} | device={device} ===")
    print(f"    recipe: lr={args.lr} wd={args.wd} bs={args.batch_size} dropout={args.dropout}")

    train_g, y_train = _build_pyg_graphs(list(train_dc.ids), train_dc.y.reshape(-1))
    val_g, y_val = _build_pyg_graphs(list(valid_dc.ids), valid_dc.y.reshape(-1))
    test_g, y_test = _build_pyg_graphs(list(test_dc.ids), test_dc.y.reshape(-1))
    in_channels = train_g[0].x.size(1)

    rows = []
    for seed in seeds:
        set_seed(seed)
        model = _make_pyg_model(in_channels, args.hidden, args.K, args.num_layers,
                                args.pool, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.BCEWithLogitsLoss()

        g = torch.Generator().manual_seed(int(seed))
        train_loader = DataLoader(train_g, batch_size=args.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_g, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_g, batch_size=128, shuffle=False)

        best_metric = float("-inf") if args.es_metric == "val_auc" else float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch, no_improve = 0, 0

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                loss = criterion(out, batch.y.view(-1))
                loss.backward()
                optimizer.step()

            val_probs = _pyg_predict(model, val_loader, device, len(val_g))
            val_roc, _, _ = _classification_metrics(y_val, val_probs)
            obs = np.isfinite(y_val) & np.isfinite(val_probs)
            val_loss = float(nn.functional.binary_cross_entropy(
                torch.from_numpy(val_probs[obs].clip(1e-7, 1 - 1e-7)),
                torch.from_numpy(y_val[obs]),
            )) if obs.any() else float("inf")

            metric = val_roc if args.es_metric == "val_auc" else val_loss
            is_better = (metric > best_metric) if args.es_metric == "val_auc" else (metric < best_metric)
            if np.isfinite(metric) and is_better:
                best_metric, best_epoch, no_improve = metric, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        model.load_state_dict(best_state)
        test_probs = _pyg_predict(model, test_loader, device, len(test_g))
        val_probs = _pyg_predict(model, val_loader, device, len(val_g))
        roc, pr, f1 = _classification_metrics(y_test, test_probs)
        val_roc, _, _ = _classification_metrics(y_val, val_probs)
        print(f"  seed {seed}: ROC-AUC={roc:.4f} PR-AUC={pr:.4f} Macro-F1={f1:.4f} (ep {best_epoch})")

        row = _blank_row("PyG-ChebConv (random init)", seed)
        row.update({"Test_ROC_AUC": roc, "Test_PR_AUC": pr, "Test_Macro_F1": f1,
                    "Val_ROC_AUC": val_roc, "Stop_Epoch": best_epoch})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="External BACE baselines (F2 classical, F3 PyG ChebConv)")
    parser.add_argument("--out_dir", type=str, default="benchmarking/results")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--models", nargs="+",
                        choices=["rf_ecfp", "logreg_ecfp", "pyg_cheb"],
                        default=["rf_ecfp", "logreg_ecfp", "pyg_cheb"])
    parser.add_argument("--n_seeds", type=int, default=len(SEEDS),
                        help="How many of the shared 100 seeds to run (default all).")
    # PyG ChebConv architecture / recipe (defaults mirror the 'ChebNet (generic)' baseline).
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
    seeds = SEEDS[:args.n_seeds]

    print(f"[Info] Loading BACE ({args.splitter.title()} Split)...")
    train_dc, valid_dc, test_dc = load_bace(args.splitter)
    print(f"  train={len(train_dc)} val={len(valid_dc)} test={len(test_dc)}")

    rows = []
    for model_name in args.models:
        if model_name in ("rf_ecfp", "logreg_ecfp"):
            rows.extend(run_classical(model_name, train_dc, valid_dc, test_dc, seeds))
        elif model_name == "pyg_cheb":
            rows.extend(run_pyg_cheb(train_dc, valid_dc, test_dc, seeds, args))

    csv_path = os.path.join(args.out_dir, "baselines_bace_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    # Quick per-arm summary (median ROC-AUC + IQR), the stat the paper should report per F12.
    print("\n=== Summary (Test ROC-AUC) ===")
    by_model = {}
    for r in rows:
        by_model.setdefault(r["Model"], []).append(r["Test_ROC_AUC"])
    for m, vals in by_model.items():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        if len(v):
            q1, q3 = np.percentile(v, [25, 75])
            print(f"  {m:32s} median={np.median(v):.4f}  IQR=[{q1:.4f}, {q3:.4f}]  "
                  f"mean={v.mean():.4f}±{v.std():.4f}  n={len(v)}")


if __name__ == "__main__":
    main()
