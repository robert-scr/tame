import io
import importlib
import sys
from pathlib import Path
from typing import List, Dict
from contextlib import redirect_stderr, redirect_stdout

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Ensure workspace root is accessible for absolute imports
here = Path(__file__).resolve().parent
workspace_root = here.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from utils.molecular_graph import smiles_to_graph
from utils.embedding_cache import EfficientEmbeddingCache


_dc = None


def _load_deepchem_silently():
    global _dc
    if _dc is not None:
        return _dc

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            _dc = importlib.import_module("deepchem")
    except Exception:
        captured_stdout = stdout_buffer.getvalue().strip()
        captured_stderr = stderr_buffer.getvalue().strip()
        if captured_stdout:
            print(captured_stdout, file=sys.stderr)
        if captured_stderr:
            print(captured_stderr, file=sys.stderr)
        raise

    return _dc


class CachedBACEDataset(Dataset):
    """
    Extremely lightweight Dataset class operating directly on pre-computed lists in memory.
    """
    def __init__(self, data_list: List[Dict[str, torch.Tensor]]):
        self.data_list = data_list
        
    def __len__(self) -> int:
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data_list[idx]


def bace_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for GNN mini-batching.
    Combines independent subgraphs into a single large disjoint (batched) graph.
    
    CRITICAL: 
    When moving multiple graphs into one batch, the node indices in edge_index 
    (which start at 0 for every new graph) must be shifted by the total number of 
    nodes accumulated in the batch so far. If you don't do this, edge connections 
    will mix between completely different molecules.
    """
    x_list = []
    edge_index_list = []
    edge_weight_list = []
    y_list = []
    batch_map_list = []
    
    node_offset = 0
    
    for i, data in enumerate(batch):
        # 1. Store node features
        # Shape: (Num Nodes, Features)
        x = data["X"]
        num_nodes = x.size(0)
        x_list.append(x)
        
        # 2. Shift and store edge indices
        # Shape of edge_index: (2, Num Edges)
        shifted_edge_index = data["edge_index"] + node_offset
        edge_index_list.append(shifted_edge_index)
        
        # 3. Store edge weights
        edge_weight_list.append(data["edge_weight"])
        
        # 4. Create the global `batch` mapping tensor
        # This assigns the graph ID (i) to all nodes in THIS graph.
        # Required for global pooling (e.g., mean pool, set2set).
        batch_mapping = torch.full((num_nodes,), i, dtype=torch.long)
        batch_map_list.append(batch_mapping)
        
        # 5. Store target value
        y_list.append(data["y"])
        
        # 6. Update running node offset for the next graph in the batch
        node_offset += num_nodes
        
    # Concatenate everything along the appropriate dimensions
    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_weight": torch.cat(edge_weight_list, dim=0),
        "batch": torch.cat(batch_map_list, dim=0),
        "y": torch.stack(y_list, dim=0),
        
        # === NEU: Deskriptoren und Text-Embeddings als Batch stapeln ===
        "desc": torch.stack([d["desc"] for d in batch], dim=0),
        "text_emb": torch.stack([d["text_emb"] for d in batch], dim=0)
    }


def precompute_split(dc_dataset, split_name: str) -> List[Dict[str, torch.Tensor]]:
    """
    Parses SMILES IDs from a DeepChem dataset exactly once into graph tensors 
    and holds them in memory.
    """
    smiles_list = list(dc_dataset.ids)
    
    # Molnet loads y as shape (N, 1) or (N,), normalize it to 1D
    y_list = dc_dataset.y.reshape(-1).astype(np.float32)
    
    processed = []
    print(f"\n=> Pre-computing {split_name} split in RAM...")
    
    for sm, y in zip(smiles_list, y_list):
        mg = smiles_to_graph(sm)
        if mg is None:
            # Skip compounds that failed RDKit parsing
            continue
            
        processed.append({
            "X": torch.from_numpy(mg.X).float(),
            "edge_index": torch.from_numpy(mg.edge_index).long(),
            "edge_weight": torch.from_numpy(mg.edge_weight).float(),
            "y": torch.tensor(y, dtype=torch.float32)
        })
        
    print(f"[{split_name}] Yielded {len(processed)} valid graphs from {len(smiles_list)} SMILES.")
    return processed


def get_bace_dataloaders(batch_size: int = 64, num_workers: int = 0, splitter: str = "scaffold"):
    """
    Downloads/Splits the BACE dataset and caches everything in memory via pre-parsing.
    """
    dc = _load_deepchem_silently()
    print(f"\n[BACE] Loading DeepChem dataset ({splitter.title()} Split)...")
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            tasks, datasets, transformers = dc.molnet.load_bace_classification(
                featurizer='ECFP',  # Wir parsen die SMILES (.ids) später ohnehin selbst
                splitter=splitter,
                frac_train=0.8,
                frac_valid=0.1,
                frac_test=0.1,
                reload=True  # Force reload from source to avoid cache corruption
            )
    except Exception:
        captured_stdout = stdout_buffer.getvalue().strip()
        captured_stderr = stderr_buffer.getvalue().strip()
        if captured_stdout:
            print(captured_stdout, file=sys.stderr)
        if captured_stderr:
            print(captured_stderr, file=sys.stderr)
        raise
    
    train_dc, valid_dc, test_dc = datasets
    
    # Pre-parse and load entirely into RAM
    train_cache = precompute_split(train_dc, "Train")
    val_cache = precompute_split(valid_dc, "Validation")
    test_cache = precompute_split(test_dc, "Test")

    desc_dir = workspace_root / "cache" / "bace_RDKit_descriptors" / splitter
    desc_train = torch.load(desc_dir / "bace_desc_train.pt")
    desc_val = torch.load(desc_dir / "bace_desc_valid.pt")
    desc_test = torch.load(desc_dir / "bace_desc_test.pt")

    emb_path = workspace_root / "cache" / "cot_embeddings" / "binding_fast_text_embeddings_compact.npz"
    print(f"Lade LLM Embeddings von {emb_path}...")
    
    seg_cache = EfficientEmbeddingCache.load(emb_path, mmap_mode="r")
    
    # SMILES Listen für den Abruf generieren
    train_smiles = list(train_dc.ids)
    val_smiles = list(valid_dc.ids)
    test_smiles = list(test_dc.ids)
    
    # Numpy Arrays aus dem Cache holen
    text_train_np = np.asarray(seg_cache.get_batch(train_smiles), dtype=np.float32)
    text_val_np = np.asarray(seg_cache.get_batch(val_smiles), dtype=np.float32)
    text_test_np = np.asarray(seg_cache.get_batch(test_smiles), dtype=np.float32)
    
    # In PyTorch Tensoren konvertieren
    text_train = torch.from_numpy(text_train_np)
    text_val = torch.from_numpy(text_val_np)
    text_test = torch.from_numpy(text_test_np)
    
    # === WICHTIG: MODALITÄTEN AN DIE GRAPHEN ANHEFTEN ===
    for i, data_dict in enumerate(train_cache):
        data_dict["desc"] = desc_train[i]
        data_dict["text_emb"] = text_train[i]
        
    for i, data_dict in enumerate(val_cache):
        data_dict["desc"] = desc_val[i]
        data_dict["text_emb"] = text_val[i]
        
    for i, data_dict in enumerate(test_cache):
        data_dict["desc"] = desc_test[i]
        data_dict["text_emb"] = text_test[i]
    
    # Initialize lightweight Datasets
    train_ds = CachedBACEDataset(train_cache)
    val_ds = CachedBACEDataset(val_cache)
    test_ds = CachedBACEDataset(test_cache)
    
    # In-memory datasets don't benefit from huge num_workers (it just creates overhead). 
    # Usually 0 or 2 is best here since parsing is done. Let's keep parameter control.
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=bace_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=bace_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=bace_collate_fn, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
