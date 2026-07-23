from utils.embedding_cache import EfficientEmbeddingCache
from models.tame_fusion_predictor import TAMEFusionPredictor, TAMEFusionPredictorConfig
import os
import sys
import io
import json
import traceback
import random
import argparse
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import torch
import optuna
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

import deepchem as dc
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


RDLogger.DisableLog("rdApp.*")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    return best_thr, best_f1


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


GLOBAL_DATA_CACHE = {}


def get_data(splitter):
    if splitter in GLOBAL_DATA_CACHE:
        return GLOBAL_DATA_CACHE[splitter]

    print(f"\n[Info] Loading BACE data ({splitter.title()} Split)...")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        dc_cache_dir = os.path.join(project_root, "cache", "deepchem_data")
        os.makedirs(dc_cache_dir, exist_ok=True)
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            tasks, datasets, transformers = dc.molnet.load_bace_classification(
                featurizer="ECFP",
                splitter=splitter,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
                data_dir=dc_cache_dir,
                save_dir=dc_cache_dir,
            )
    except Exception:
        raise

    train_dc, valid_dc, test_dc = datasets

    train_smiles = list(train_dc.ids)
    val_smiles = list(valid_dc.ids)
    test_smiles = list(test_dc.ids)

    y_train = train_dc.y.astype(np.float32).reshape(-1)
    y_val = valid_dc.y.astype(np.float32).reshape(-1)
    y_test = test_dc.y.astype(np.float32).reshape(-1)

    emb_path = os.path.join(
        project_root,
        "cache",
        "cot_embeddings",
        "binding_fast_text_embeddings_compact.npz",
    )
    print(f"[Info] Loading text embeddings from {emb_path}...")
    seg_cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    text_train = np.asarray(seg_cache.get_batch(train_smiles), dtype=np.float32)
    text_val = np.asarray(seg_cache.get_batch(val_smiles), dtype=np.float32)
    text_test = np.asarray(seg_cache.get_batch(test_smiles), dtype=np.float32)

    print("[Info] Computing RDKit descriptors...")
    desc_train_raw = compute_rdkit_descriptors(train_smiles)
    desc_val_raw = compute_rdkit_descriptors(val_smiles)
    desc_test_raw = compute_rdkit_descriptors(test_smiles)
    desc_train, desc_val, desc_test = impute_with_train_median(
        desc_train_raw, desc_val_raw, desc_test_raw
    )

    data = {
        "train_smiles": train_smiles,
        "val_smiles": val_smiles,
        "test_smiles": test_smiles,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "text_train": text_train,
        "text_val": text_val,
        "text_test": text_test,
        "desc_train": desc_train,
        "desc_val": desc_val,
        "desc_test": desc_test,
    }

    GLOBAL_DATA_CACHE[splitter] = data
    return data


def objective(trial, args):
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    max_epochs = int(args.max_epochs)
    active_seed = int(args.seed)
    set_seed(active_seed)

    data = get_data(args.splitter)

    # ChebNet backbone
    K = trial.suggest_categorical("K", [3, 4, 5])
    num_layers = trial.suggest_categorical("num_layers", [2, 3])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128])
    pool_type = trial.suggest_categorical("pool_type", ["sum", "set2set"])
    if pool_type == "set2set":
        set2set_n_iter = trial.suggest_int("set2set_n_iter", 4, 9, step=1)
    else:
        set2set_n_iter = 4
    encoder_lr_mult = trial.suggest_float("encoder_lr_mult", 0.01, 1.0, log=True)

    # SEG fusion branch (graph + text)
    text_projection_dim = trial.suggest_categorical(
        "text_projection_dim", [None, 128, 256, 512]
    )
    fusion = trial.suggest_categorical("fusion", ["cross_mha", "gated", "film"])
    fusion_dim = trial.suggest_categorical("fusion_dim", [64, 128, 256])
    if fusion == "cross_mha":
        fusion_n_heads = trial.suggest_categorical("fusion_n_heads", [2, 4, 8])
    else:
        fusion_n_heads = 4

    # 2-expert MoE (SEG branch vs descriptor branch)
    moe_hidden_dim = trial.suggest_categorical("moe_hidden_dim", [64, 128, 256])
    projection_dropout = trial.suggest_float("projection_dropout", 0.1, 0.6)
    router_dropout = trial.suggest_float("router_dropout", 0.1, 0.6)
    head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [32, 64, 128])
    head_dropout = trial.suggest_float("head_dropout", 0.1, 0.8)

    # Gate regularization
    gate_balance_weight = trial.suggest_float(
        "gate_balance_weight", 1e-3, 5e-1, log=True
    )
    gate_entropy_weight = trial.suggest_float("gate_entropy_weight", 0.0, 0.1)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    desc_modality_dropout = trial.suggest_float("desc_modality_dropout", 0.1, 0.8)
    # gate_target_seg: target influence of the SEG branch; desc = 1 - seg
    gate_target_seg = trial.suggest_float("gate_target_seg", 0.4, 0.9)

    # Optimizer
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])

    try:
        ckpt_name = f"chebnet_pt_d{hidden_channels}_K{K}_L{num_layers}_e25.pt"
        checkpoint_path = os.path.join(args.ckpt_dir, ckpt_name)
        if not os.path.exists(checkpoint_path):
            print(f"[Info] Pruning trial: checkpoint {ckpt_name} not found.")
            raise optuna.exceptions.TrialPruned()

        config = TAMEFusionPredictorConfig(
            task="classification",
            num_tasks=1,
            hidden_channels=hidden_channels,
            K=K,
            num_layers=num_layers,
            pool=pool_type,
            set2set_processing_steps=set2set_n_iter,
            text_embedding_dim=int(data["text_train"].shape[1]),
            text_projection_dim=text_projection_dim,
            fusion=fusion,
            fusion_dim=fusion_dim,
            fusion_n_heads=fusion_n_heads,
            moe_hidden_dim=moe_hidden_dim,
            projection_dropout=projection_dropout,
            router_dropout=router_dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            gate_balance_weight=gate_balance_weight,
            gate_entropy_weight=gate_entropy_weight,
            desc_modality_dropout=desc_modality_dropout,
            gate_target=gate_target_seg,
            label_smoothing=label_smoothing,
            descriptor_standardize=True,
            descriptor_winsorize_lower_q=0.01,
            descriptor_winsorize_upper_q=0.99,
        )

        model = TAMEFusionPredictor(config=config, device=device)

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("encoder_state", state.get("model_state_dict", state))
        filtered_dict = {
            k.replace("encoder.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        if not filtered_dict:
            filtered_dict = state_dict

        trial.set_user_attr("splitter", str(args.splitter))
        trial.set_user_attr("seed", active_seed)

        _ = model.fit(
            smiles_list=data["train_smiles"],
            labels=data["y_train"],
            val_smiles=data["val_smiles"],
            val_labels=data["y_val"],
            text_embeddings=data["text_train"],
            val_text_embeddings=data["text_val"],
            descriptor_features=data["desc_train"],
            val_descriptor_features=data["desc_val"],
            num_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            patience=20,
            early_stopping_metric=args.es_metric,
            verbose=False,
            seed=active_seed,
            init_encoder_state=filtered_dict,
            encoder_lr_mult=encoder_lr_mult,
        )

        val_preds = model.predict_batch(
            data["val_smiles"],
            text_embeddings=data["text_val"],
            descriptor_features=data["desc_val"],
            batch_size=128,
        ).flatten()
        test_preds = model.predict_batch(
            data["test_smiles"],
            text_embeddings=data["text_test"],
            descriptor_features=data["desc_test"],
            batch_size=128,
        ).flatten()

        y_val_obs, val_preds_obs = _filter_nans(data["y_val"], val_preds)
        y_test_obs, test_preds_obs = _filter_nans(data["y_test"], test_preds)

        if len(y_val_obs) == 0:
            raise ValueError("No valid predictions in validation set.")

        val_auc = roc_auc_score(y_val_obs, val_preds_obs)
        val_pr_auc = average_precision_score(y_val_obs, val_preds_obs)
        best_thr, val_f1 = find_best_threshold(y_val_obs, val_preds_obs)

        if len(y_test_obs) > 0:
            test_auc = roc_auc_score(y_test_obs, test_preds_obs)
            test_pr_auc = average_precision_score(y_test_obs, test_preds_obs)
            test_f1 = f1_score(
                y_test_obs, (test_preds_obs >= best_thr).astype(int), zero_division=0
            )
        else:
            test_auc, test_pr_auc, test_f1 = 0.5, 0.0, 0.0

        val_score = (
            val_auc
            if args.metric == "roc_auc"
            else val_pr_auc
            if args.metric == "pr_auc"
            else val_f1
        )

        trial.set_user_attr("val_auc", float(val_auc))
        trial.set_user_attr("val_pr_auc", float(val_pr_auc))
        trial.set_user_attr("val_f1", float(val_f1))
        trial.set_user_attr("test_auc", float(test_auc))
        trial.set_user_attr("test_pr_auc", float(test_pr_auc))
        trial.set_user_attr("test_f1", float(test_f1))
        trial.set_user_attr("threshold", float(best_thr))

        return val_score

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Exception in trial {trial.number}: {e}")
        traceback.print_exc()
        return float("-inf")


def _filter_nans(y_true, y_pred):
    obs = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[obs], y_pred[obs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna HPO for TAMEFusionPredictor on BACE"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Rechnungen/hpo",
        help="Output directory for HPO results",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory with pretrained ChebNet checkpoints",
    )
    parser.add_argument("--seed", type=int, default=11, help="Seed for HPO")
    parser.add_argument(
        "--splitter", type=str, choices=["random", "scaffold"], default="scaffold"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument(
        "--metric", type=str, choices=["roc_auc", "pr_auc", "f1"], default="roc_auc"
    )
    parser.add_argument(
        "--es_metric",
        type=str,
        choices=["val_loss", "val_auc"],
        default="val_loss",
        help="Early stopping metric for fit()",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_tag = f"s{args.seed}"
    db_url = f"sqlite:///{
        os.path.abspath(
            os.path.join(
                args.out_dir,
                f'tame_fusion_predictor_hpo_v1_{args.splitter}_{seed_tag}.db',
            )
        ).replace(chr(92), '/')
    }"

    study = optuna.create_study(
        study_name=f"tame_fusion_predictor_hpo_v1_{args.splitter}_{seed_tag}",
        direction="maximize",
        storage=db_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    print(
        f"Starting TAME-Fusion HPO (trials: {args.n_trials}, splitter: {
            args.splitter
        }, metric: {args.metric})..."
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\nStudy finished! Best parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(
        args.out_dir, f"tame_fusion_best_params_v1_{args.splitter}_{seed_tag}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "user_attrs": study.best_trial.user_attrs,
            },
            f,
            indent=4,
        )
    print(f"\nSaved to {out_path}")
