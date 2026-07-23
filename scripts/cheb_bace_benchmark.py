import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

# Make sure we can import from project root
here = Path(__file__).resolve().parent
workspace_root = here.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from models.core.graph_encoder import ChebNetEncoder
from models.core.pooling import create_pooling
from data.bace_loader import get_bace_dataloaders


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ChebClassifier(nn.Module):
    """
    ChebNet Classifier with specific pooling mechanism and an MLP classifying head.
    """
    def __init__(self, F_in: int, d_cheb: int, K: int, L: int, pooling: str = "mean"):
        super().__init__()
        self.encoder = ChebNetEncoder(
            in_channels=F_in,
            hidden_channels=d_cheb,
            K=K,
            num_layers=L
        )
        self.pool = create_pooling(pool_type=pooling, input_dim=d_cheb)
        head_input_dim = getattr(self.pool, "output_dim", d_cheb)
        
        # MLP Classifier: Linear -> ReLU -> Dropout -> Linear(1)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, max(head_input_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(max(head_input_dim // 2, 1), 1)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Message Passing
        node_embeds = self.encoder(x, edge_index, edge_weight, batch)
        # 2. Global Pooling
        graph_embeds = self.pool(node_embeds, batch)
        # 3. Classify
        logits = self.head(graph_embeds)
        return logits.view(-1)


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, patience=15, max_epochs=100):
    """
    Fixed Ceteris Paribus test loop.
    Returns the test ROC-AUC and Test Macro-F1 of the epoch with max validation ROC-AUC.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = -1.0
    best_test_auc = 0.0
    best_test_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        print(f"    [Epoch {epoch + 1:03d}/{max_epochs}] training...")
        # Training
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            edge_weight = batch["edge_weight"].to(device, non_blocking=True)
            batch_idx = batch["batch"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index, edge_weight, batch_idx)
            loss = criterion(logits.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            
        # Validation & Test Evaluation
        model.eval()
        
        def evaluate_loader(loader):
            all_preds, all_y = [], []
            with torch.no_grad():
                for batch in loader:
                    x = batch["x"].to(device, non_blocking=True)
                    edge_index = batch["edge_index"].to(device, non_blocking=True)
                    edge_weight = batch["edge_weight"].to(device, non_blocking=True)
                    batch_idx = batch["batch"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True).float()
                    
                    logits = model(x, edge_index, edge_weight, batch_idx)
                    probs = torch.sigmoid(logits)
                    
                    all_preds.extend(probs.cpu().numpy())
                    all_y.extend(y.cpu().numpy())
            
            y_arr = np.array(all_y)
            pred_arr = np.array(all_preds)
            auc = roc_auc_score(y_arr, pred_arr)
            # F1 computed with 0.5 binarization logic for macro
            f1 = f1_score(y_arr, (pred_arr >= 0.5).astype(int), average='macro')
            return auc, f1

        val_auc, _ = evaluate_loader(val_loader)
        test_auc, test_f1 = evaluate_loader(test_loader)
        print(f"    [Epoch {epoch + 1:03d}/{max_epochs}] val_auc={val_auc:.4f} test_auc={test_auc:.4f} test_f1={test_f1:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_test_f1 = test_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    return best_test_auc, best_test_f1


def run_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")
    effective_num_workers = 0 if sys.platform.startswith("win") else args.num_workers
    if effective_num_workers != args.num_workers:
        print(f"[Info] Windows detected, overriding num_workers {args.num_workers} -> {effective_num_workers}")
    
    train_loader, val_loader, test_loader = get_bace_dataloaders(
        batch_size=args.batch_size,
        num_workers=effective_num_workers,
    )
    seeds = [11, 22, 33, 44, 55]
    modes = ["Vanilla", "Pretrained"]
    
    results = []
    
    for mode in modes:
        print(f"\nEvaluating Mode: {mode}")
        for seed in seeds:
            set_seed(seed)
            print(f"  -> Seed = {seed}")
            
            model = ChebClassifier(
                F_in=args.F_in, 
                d_cheb=args.d_cheb, 
                K=args.K, 
                L=args.L, 
                pooling=args.pooling
            ).to(device)
            
            if mode == "Pretrained":
                ckpt = torch.load(args.ckpt_path, map_location=device)
                
                # Retrieve state dict (handling raw dicts or payload constructs)
                state_dict = ckpt.get("model_state_dict", ckpt)
                
                # Filter only keys with 'encoder.'
                filtered_dict = {
                    k: v for k, v in state_dict.items() if k.startswith("encoder.")
                }
                
                if len(filtered_dict) == 0:
                    print(f"[Warn] No 'encoder.' keys found in {args.ckpt_path}. Is the naming strictly 'encoder'?")
                
                # Load with strict=False
                model.load_state_dict(filtered_dict, strict=False)
            
            test_auc, test_f1 = train_and_evaluate(
                model=model, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                test_loader=test_loader, 
                device=device
            )
            
            results.append({
                "Mode": mode,
                "Seed": seed,
                "Test_AUC": test_auc,
                "Test_Macro_F1": test_f1
            })
            
    # Save & Summarize
    df = pd.DataFrame(results)
    
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = workspace_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_name = f"bace_benchmark_d{args.d_cheb}_K{args.K}_L{args.L}_{args.pooling}.csv"
    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw results to {out_path}")
    
    print("\n=== Benchmark Summary ===")
    summary = df.groupby("Mode")[["Test_AUC", "Test_Macro_F1"]].agg(['mean', 'std']).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChebNet Pretraining BACE Benchmark")
    parser.add_argument("--F_in", type=int, required=True, help="Input feature dimension")
    parser.add_argument("--d_cheb", type=int, required=True, help="Hidden channels (d_cheb)")
    parser.add_argument("--K", type=int, required=True, help="Chebyshev polynomial degree K")
    parser.add_argument("--L", type=int, required=True, help="Number of ChebNet layers L")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pre-trained checktpoint")
    parser.add_argument("--pooling", type=str, default="mean", help="Pooling to use: mean, sum, set2set")
    parser.add_argument("--out_dir", type=str, default="Rechnungen/benchmarks/", help="Directory to save CSV results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    
    args = parser.parse_args()
    run_benchmark(args)