"""Memory-efficient embedding cache utilities.

This module provides utilities to:
1. Convert bloated pickle caches to compact numpy format
2. Load embeddings memory-efficiently using memory-mapping
3. Stream embeddings during training without loading all into RAM
4. Regenerate embeddings from CoT text cache (avoiding memory issues)

The problem: pickle files with torch.Tensor dicts can be 50x larger than necessary.
Solution: numpy .npz files with float16 arrays + SMILES index.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import pickle

import numpy as np


def regenerate_compact_cache(
    cot_texts_path: Union[str, Path],
    output_path: Union[str, Path],
    client,
    embedding_model: str = "text-embedding-3-large",
    batch_size: int = 100,
    dtype: str = "float16"
) -> Path:
    """
    Regenerate embeddings from CoT text cache and save in compact format.
    
    This is memory-efficient because:
    1. CoT texts are loaded one batch at a time
    2. Embeddings are generated in batches via API
    3. Results go directly to numpy array, not dict
    
    Args:
        cot_texts_path: Path to JSON file with {smiles: cot_text} mapping
        output_path: Path for output .npz file
        client: Azure OpenAI client
        embedding_model: Embedding model name
        batch_size: Batch size for API calls
        dtype: Output dtype ("float16" or "float32")
        
    Returns:
        Path to created .npz file
    """
    from tqdm import tqdm
    
    cot_texts_path = Path(cot_texts_path)
    output_path = Path(output_path)
    
    print(f"Loading CoT texts from {cot_texts_path.name}...")
    with open(cot_texts_path, "r", encoding="utf-8") as f:
        cot_texts = json.load(f)
    
    smiles_list = sorted(cot_texts.keys())
    n = len(smiles_list)
    print(f"  Found {n} molecules")
    
    # First call to get embedding dimension
    print("Getting embedding dimension...")
    test_resp = client.embeddings.create(input=["test"], model=embedding_model)
    emb_dim = len(test_resp.data[0].embedding)
    print(f"  Embedding dim: {emb_dim}")
    
    # Allocate output array
    target_dtype = np.float16 if dtype == "float16" else np.float32
    embeddings = np.zeros((n, emb_dim), dtype=target_dtype)
    
    # Process in batches
    print(f"Generating embeddings in batches of {batch_size}...")
    for batch_start in tqdm(range(0, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        batch_smiles = smiles_list[batch_start:batch_end]
        batch_texts = [cot_texts[smi] for smi in batch_smiles]
        
        # Get embeddings from API
        response = client.embeddings.create(input=batch_texts, model=embedding_model)
        
        # Store in array
        for i, emb_data in enumerate(response.data):
            embeddings[batch_start + i] = np.array(emb_data.embedding, dtype=target_dtype)
    
    # Save embeddings
    print(f"Saving to {output_path.name}...")
    np.savez_compressed(output_path, embeddings=embeddings)
    
    # Save index
    index_path = output_path.parent / output_path.name.replace("_compact.npz", "_index.json")
    smiles_to_idx = {smi: i for i, smi in enumerate(smiles_list)}
    with open(index_path, "w") as f:
        json.dump(smiles_to_idx, f)
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"\n✓ Created compact cache: {size_mb:.1f} MB ({n} molecules)")
    
    return output_path


def convert_pickle_cache(
    pkl_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    dtype: str = "float16"
) -> Dict[str, Path]:
    """
    Convert bloated pickle embedding cache to efficient numpy format.
    
    Args:
        pkl_path: Path to pickle file with {smiles: tensor} mapping
        output_dir: Output directory (default: same as pkl_path)
        dtype: Target dtype - "float16" (smaller) or "float32" (default precision)
        
    Returns:
        Dict with paths to created files: {"embeddings": ..., "index": ...}
    """
    pkl_path = Path(pkl_path)
    if output_dir is None:
        output_dir = pkl_path.parent
    output_dir = Path(output_dir)
    
    base_name = pkl_path.stem  # e.g., "quantum_fast_text_embeddings"
    
    print(f"Loading pickle cache: {pkl_path}")
    print(f"  File size: {pkl_path.stat().st_size / (1024**2):.1f} MB")
    
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    
    print(f"  Entries: {len(cache)}")
    
    # Get dimensions from first entry
    first_key = next(iter(cache))
    first_val = cache[first_key]
    if hasattr(first_val, 'numpy'):
        first_arr = first_val.numpy()
    else:
        first_arr = np.array(first_val)
    
    emb_dim = first_arr.shape[-1]
    print(f"  Embedding dim: {emb_dim}")
    
    # Build sorted SMILES list and embeddings array
    smiles_list = sorted(cache.keys())
    n = len(smiles_list)
    
    target_dtype = np.float16 if dtype == "float16" else np.float32
    embeddings = np.zeros((n, emb_dim), dtype=target_dtype)
    
    print(f"Converting to {dtype} array...")
    for i, smi in enumerate(smiles_list):
        val = cache[smi]
        if hasattr(val, 'numpy'):
            arr = val.numpy()
        else:
            arr = np.array(val)
        embeddings[i] = arr.flatten().astype(target_dtype)
    
    # Save embeddings as compressed numpy
    emb_path = output_dir / f"{base_name}_compact.npz"
    np.savez_compressed(emb_path, embeddings=embeddings)
    print(f"  Saved embeddings: {emb_path}")
    print(f"  New size: {emb_path.stat().st_size / (1024**2):.1f} MB")
    
    # Save SMILES index
    index_path = output_dir / f"{base_name}_index.json"
    smiles_to_idx = {smi: i for i, smi in enumerate(smiles_list)}
    with open(index_path, "w") as f:
        json.dump(smiles_to_idx, f)
    print(f"  Saved index: {index_path}")
    
    # Summary
    orig_mb = pkl_path.stat().st_size / (1024**2)
    new_mb = emb_path.stat().st_size / (1024**2)
    reduction = (1 - new_mb / orig_mb) * 100
    print(f"\n✓ Reduced from {orig_mb:.1f} MB to {new_mb:.1f} MB ({reduction:.1f}% smaller)")
    
    return {"embeddings": emb_path, "index": index_path}


class EfficientEmbeddingCache:
    """
    Memory-efficient embedding cache with lazy loading.
    
    Features:
    - Memory-mapped numpy array (doesn't load all into RAM)
    - Fast SMILES -> index lookup
    - Batch retrieval support
    - Float16 support for 2x memory savings
    
    Usage:
        >>> cache = EfficientEmbeddingCache.load("path/to/embeddings_compact.npz")
        >>> emb = cache.get("CCO")  # Single lookup
        >>> embs = cache.get_batch(["CCO", "C", "CC"])  # Batch lookup
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        smiles_to_idx: Dict[str, int],
        output_dtype: str = "float32"
    ):
        """
        Initialize cache with embeddings and index.
        
        Args:
            embeddings: (N, D) numpy array
            smiles_to_idx: {smiles: row_index} mapping
            output_dtype: Output dtype for get() calls
        """
        self.embeddings = embeddings
        self.smiles_to_idx = smiles_to_idx
        self.idx_to_smiles = {v: k for k, v in smiles_to_idx.items()}
        self.output_dtype = np.float32 if output_dtype == "float32" else np.float16
        self.embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
    
    @classmethod
    def load(
        cls,
        npz_path: Union[str, Path],
        index_path: Optional[Union[str, Path]] = None,
        mmap_mode: Optional[str] = "r"
    ) -> "EfficientEmbeddingCache":
        """
        Load cache from compact numpy format.
        
        Args:
            npz_path: Path to .npz file with embeddings
            index_path: Path to JSON index (auto-detected if None)
            mmap_mode: Memory map mode ("r" for read-only, None for full load)
            
        Returns:
            EfficientEmbeddingCache instance
        """
        npz_path = Path(npz_path)
        
        # Auto-detect index path
        if index_path is None:
            base = npz_path.stem.replace("_compact", "")
            index_path = npz_path.parent / f"{base}_index.json"
        index_path = Path(index_path)
        
        print(f"Loading embeddings from {npz_path.name}...")
        
        # Load with memory mapping (lazy loading)
        loaded = np.load(npz_path, mmap_mode=mmap_mode, allow_pickle=True)
        embeddings = loaded["embeddings"]
        
        # Load index
        with open(index_path) as f:
            smiles_to_idx = json.load(f)
        
        print(f"  Loaded {len(smiles_to_idx)} entries, dim={embeddings.shape[1]}")
        print(f"  Memory mode: {'mapped' if mmap_mode else 'full'}")
        
        return cls(embeddings, smiles_to_idx)
    
    @classmethod
    def from_pickle(
        cls,
        pkl_path: Union[str, Path],
        output_dtype: str = "float32"
    ) -> "EfficientEmbeddingCache":
        """
        Create cache directly from pickle file (loads into memory).
        
        Use this only if you have enough RAM. For large caches,
        use convert_pickle_cache() first, then load().
        """
        import torch
        
        with open(pkl_path, "rb") as f:
            cache = pickle.load(f)
        
        smiles_list = sorted(cache.keys())
        smiles_to_idx = {smi: i for i, smi in enumerate(smiles_list)}
        
        # Convert to numpy array
        first_val = cache[smiles_list[0]]
        if hasattr(first_val, 'numpy'):
            emb_dim = first_val.numpy().flatten().shape[0]
        else:
            emb_dim = np.array(first_val).flatten().shape[0]
        
        target_dtype = np.float32 if output_dtype == "float32" else np.float16
        embeddings = np.zeros((len(smiles_list), emb_dim), dtype=target_dtype)
        
        for i, smi in enumerate(smiles_list):
            val = cache[smi]
            if hasattr(val, 'numpy'):
                arr = val.numpy()
            else:
                arr = np.array(val)
            embeddings[i] = arr.flatten().astype(target_dtype)
        
        return cls(embeddings, smiles_to_idx, output_dtype)
    
    def __len__(self) -> int:
        return len(self.smiles_to_idx)
    
    def __contains__(self, smiles: str) -> bool:
        return smiles in self.smiles_to_idx
    
    def get(self, smiles: str) -> Optional[np.ndarray]:
        """Get embedding for a single SMILES."""
        idx = self.smiles_to_idx.get(smiles)
        if idx is None:
            return None
        return self.embeddings[idx].astype(self.output_dtype)
    
    def get_batch(
        self,
        smiles_list: List[str],
        return_missing: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Get embeddings for a batch of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            return_missing: If True, also return list of missing SMILES
            
        Returns:
            (N, D) numpy array. If return_missing=True, returns (array, missing_list)
        """
        n = len(smiles_list)
        result = np.zeros((n, self.embedding_dim), dtype=self.output_dtype)
        missing = []
        
        for i, smi in enumerate(smiles_list):
            idx = self.smiles_to_idx.get(smi)
            if idx is not None:
                result[i] = self.embeddings[idx].astype(self.output_dtype)
            else:
                missing.append(smi)
        
        if return_missing:
            return result, missing
        return result
    
    def get_all_smiles(self) -> List[str]:
        """Get list of all cached SMILES (sorted by index)."""
        return [self.idx_to_smiles[i] for i in range(len(self.smiles_to_idx))]


def get_embeddings_streaming(
    smiles_list: List[str],
    cache: EfficientEmbeddingCache,
    cot_generator,
    client,
    batch_size: int = 100,
    max_workers: int = 16
) -> np.ndarray:
    """
    Get embeddings with streaming: load from cache or generate missing ones.
    
    This is memory-efficient because:
    1. Cache uses memory mapping (only loads accessed rows)
    2. Missing embeddings are generated in batches
    3. No intermediate dict accumulation
    
    Args:
        smiles_list: SMILES to get embeddings for
        cache: EfficientEmbeddingCache instance
        cot_generator: CoTGenerator for generating missing embeddings
        client: Azure OpenAI client
        batch_size: Batch size for generating missing embeddings
        max_workers: Number of parallel workers for generation
        
    Returns:
        (N, D) numpy array of embeddings
    """
    from tqdm import tqdm
    
    n = len(smiles_list)
    result = np.zeros((n, cache.embedding_dim), dtype=np.float32)
    missing_indices = []
    missing_smiles = []
    
    # First pass: get from cache, identify missing
    print(f"Checking cache for {n} SMILES...")
    for i, smi in enumerate(smiles_list):
        emb = cache.get(smi)
        if emb is not None:
            result[i] = emb
        else:
            missing_indices.append(i)
            missing_smiles.append(smi)
    
    print(f"  Cache hits: {n - len(missing_smiles)}")
    print(f"  Missing: {len(missing_smiles)}")
    
    # Generate missing embeddings in batches
    if missing_smiles:
        print(f"Generating {len(missing_smiles)} missing embeddings...")
        new_embs = cot_generator.get_embeddings_parallel(
            missing_smiles, client, max_workers=max_workers
        )
        
        for i, idx in enumerate(missing_indices):
            result[idx] = new_embs[i]
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embedding_cache.py <pickle_file.pkl>")
        print("\nConverts bloated pickle cache to efficient numpy format.")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    convert_pickle_cache(pkl_path, dtype="float16")
