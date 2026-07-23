import argparse
import sys
from pathlib import Path
import traceback

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# Make sure we can import from project root
here = Path(__file__).resolve().parent
workspace_root = here.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from models.chemeleon_pretraining import ChebNodePretrainer
from utils.molecular_graph import MolGraph, smiles_to_graph
from utils.batched_mol_graph import batch_graphs

class NodePretrainingDataset(Dataset):
    """
    Extremely lightweight dataset for Stage-1 (Node-Level) Pretraining.
    Takes a list of SMILES, parses them on-the-fly to MolGraph, and 
    returns just the structural graph data (no targets).
    """
    def __init__(self, smiles_list: list[str]):
        self.smiles_list = smiles_list
        # Compute first graph to extract input channels (F_in)
        if len(self.smiles_list) > 0:
            first_mol = smiles_to_graph(self.smiles_list[0])
            self.in_channels = first_mol.X.shape[1]
        else:
            self.in_channels = 0

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict:
        smiles = self.smiles_list[idx]
        mol_graph = smiles_to_graph(smiles)
        return {
            "X": torch.from_numpy(mol_graph.X).float(),
            "edge_index": torch.from_numpy(mol_graph.edge_index).long(),
            "edge_weight": torch.from_numpy(mol_graph.edge_weight).float(),
        }

def collate_mol_graphs(batch: list[dict]) -> dict:
    """
    Collate function to batch multiple MolGraphs together using BatchedMolGraph logic.
    """
    # Reconstruct MolGraph objects for batch_graphs
    graphs = []
    for item in batch:
        graphs.append(MolGraph(
            X=item["X"].numpy(),
            edge_index=item["edge_index"].numpy(),
            edge_weight=item["edge_weight"].numpy(),
            n_nodes=item["X"].shape[0]
        ))
    
    batched = batch_graphs(graphs)
    
    return {
        "x": torch.from_numpy(batched.X).float(),
        "edge_index": torch.from_numpy(batched.edge_index).long(),
        "edge_weight": torch.from_numpy(batched.edge_weight).float(),
        "batch": torch.from_numpy(batched.batch).long(),
    }

def run_stage1_pretraining(
    F_in: int, 
    d_cheb: int, 
    K: int, 
    L: int, 
    smiles_list: list[str],
    epochs: int = 10,
    val_split: float = 0.1,
    output_dir: str = "cache/pretrained_checkpoints",
    **kwargs
):
    """
    Runs the Stage-1 Pretraining (Node-Attribute Masking) for ChebNet architecture.
    
    Returns the path to the saved checkpoint.
    """
    # Override default parameters from kwargs
    batch_size = kwargs.get("batch_size", 64)
    lr = kwargs.get("learning_rate", 1e-3)
    weight_decay = kwargs.get("weight_decay", 1e-5)
    mask_fraction = kwargs.get("mask_fraction", 0.25)
    dropout = kwargs.get("dropout", 0.2)
    device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    num_workers = kwargs.get("num_workers", 4)
    add_hydrogens = kwargs.get("add_hydrogens", False)
    seed = kwargs.get("seed", 42)
    
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")

    print(f"Initializing Datasets with {len(smiles_list)} molecules...")

    # Deterministic split so train/val partitions remain stable across runs for a fixed seed.
    n_total = len(smiles_list)
    if n_total == 0:
        raise ValueError("smiles_list is empty; cannot run pretraining")

    split_gen = torch.Generator(device="cpu")
    split_gen.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=split_gen).tolist()

    if n_total == 1 or val_split == 0.0:
        n_val = 0
    else:
        n_val = max(1, int(n_total * val_split))
        n_val = min(n_val, n_total - 1)

    val_indices = set(perm[:n_val])
    train_smiles = [s for i, s in enumerate(smiles_list) if i not in val_indices]
    val_smiles = [s for i, s in enumerate(smiles_list) if i in val_indices]

    print(f"Split sizes -> train: {len(train_smiles)}, val: {len(val_smiles)}")

    train_dataset = NodePretrainingDataset(train_smiles)
    val_dataset = NodePretrainingDataset(val_smiles)

    # Verify F_in requirement against train set (or val if train is unexpectedly empty).
    inferred_channels = train_dataset.in_channels if len(train_dataset) > 0 else val_dataset.in_channels
    if inferred_channels != F_in and F_in > 0:
        print(f"Warning: Forced F_in={F_in} differs from dataset inference ({inferred_channels})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_mol_graphs,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mol_graphs,
        num_workers=num_workers,
    )
    
    # Model Initialization
    print(f"Initializing ChebNodePretrainer(F_in={F_in}, d_cheb={d_cheb}, K={K}, L={L}, mask={mask_fraction}) onto {device}")
    model = ChebNodePretrainer(
        in_channels=F_in,
        hidden_channels=d_cheb,
        K=K,
        num_layers=L,
        dropout=dropout,
        atom_type_dim=10,
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    
    # Checkpoint setup
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = workspace_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"chebnet_pt_d{d_cheb}_K{K}_L{L}_e{epochs}.pt"
    ckpt_path = out_dir / ckpt_name

    print("Starting Training Loop...")
    best_val_loss = float("inf")
    best_ckpt_path = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch_x = batch["x"].to(device, non_blocking=True)
            batch_ei = batch["edge_index"].to(device, non_blocking=True)
            batch_ew = batch["edge_weight"].to(device, non_blocking=True)
            batch_idx = batch["batch"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            out = model(
                x=batch_x,
                edge_index=batch_ei,
                edge_weight=batch_ew,
                batch=batch_idx,
                mask_fraction=mask_fraction,
                generator=generator
            )
            
            loss = out["loss"]
            loss.backward()
            
            # Strict gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            # Since node-level CE loss is averaged over masked nodes
            # We track per-batch loss average
            total_loss += float(loss.item())
            n_batches += 1
            
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_total_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch["x"].to(device, non_blocking=True)
                batch_ei = batch["edge_index"].to(device, non_blocking=True)
                batch_ew = batch["edge_weight"].to(device, non_blocking=True)
                batch_idx = batch["batch"].to(device, non_blocking=True)

                out = model(
                    x=batch_x,
                    edge_index=batch_ei,
                    edge_weight=batch_ew,
                    batch=batch_idx,
                    mask_fraction=mask_fraction,
                    generator=generator,
                )
                val_total_loss += float(out["loss"].item())
                val_batches += 1

        val_loss = val_total_loss / max(val_batches, 1) if len(val_dataset) > 0 else train_loss

        print(
            f"[Epoch {epoch:03d}/{epochs}] Train Loss (CE): {train_loss:.6f} | "
            f"Val Loss (CE): {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "F_in": F_in,
                "d_cheb": d_cheb,
                "K": K,
                "L": L,
                "epochs": epoch,
                "val_loss": best_val_loss,
                "add_hydrogens": add_hydrogens,
                "pretraining_stage": "stage1_node",
            }
            torch.save(payload, ckpt_path)
            best_ckpt_path = ckpt_path
            print(f"  -> New best model saved (val CE: {best_val_loss:.6f}) to {ckpt_path.resolve()}")

    return best_ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 ChebNet Pretraining (Node Masking)")
    parser.add_argument("--F_in", type=int, required=True, help="Input feature dimension")
    parser.add_argument("--d_cheb", type=int, required=True, help="Hidden channels for pretraining")
    parser.add_argument("--K", type=int, required=True, help="Chebyshev polynomial degree K")
    parser.add_argument("--L", type=int, required=True, help="Number of ChebNet layers L")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--smiles_file", type=str, help="Path to SMILES file (one per line)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Anzahl der CPU-Worker fuer den DataLoader")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size fürs Training")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="cache/pretrained_checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    
    # Load SMILES
    smiles = []
    if args.smiles_file:
        p = Path(args.smiles_file)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                smiles = [line.strip() for line in f if line.strip()]
        else:
            print(f"Error: {args.smiles_file} not found.")
            sys.exit(1)
    else:
        print("No --smiles_file provided, using fake dummy molecules for test run")
        smiles = ["C1=CC=CC=C1", "CCO", "CC(=O)O", "CCN", "c1ccccc1"] * 10
        
    try:
        run_stage1_pretraining(
            F_in=args.F_in,
            d_cheb=args.d_cheb,
            K=args.K,
            L=args.L,
            smiles_list=smiles,
            epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            val_split=0.15,
        )
    except Exception as e:
        print("Error during Stage-1 pretraining:")
        traceback.print_exc()
        sys.exit(1)
