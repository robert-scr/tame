# TAME — Tri-modal Adaptive Molecular Encoder

Reference implementation for **TAME: Element-wise Mixture-of-Experts Fusion for Data-Efficient and Interpretable Molecular Property Prediction**.

TAME encodes each molecule as three parallel views and fuses them with an **element-wise Mixture-of-Experts (MoE) router** that assigns a separate softmax mixture over modalities to *every* hidden coordinate (not one scalar weight per modality):

- **Graph** — a Chebyshev spectral GNN (ChebNet) over the molecular graph (local topology).
- **Text** — an LLM chain-of-thought description embedded by `text-embedding-3-large` (semantic context).
- **Descriptor** — a deterministic RDKit descriptor vector (global physicochemistry).

Two fusion topologies are provided:

- **TAME (flat)** — one 3-way element-wise MoE over graph/text/descriptor (`models/tame_predictor.py`).
- **TAME-Fusion (hierarchical)** — graph+text are first fused by cross-modal attention (SEG), then routed against the descriptor branch with a 2-way MoE (`models/tame_fusion_predictor.py`).

## Project structure

```
models/          Predictors (ChebNet baseline, SEG, TAME, TAME-Fusion) + core building blocks
  core/          BasePredictor, ChebNetEncoder, pooling, fusion factories
  chemeleon_pretraining.py   Stage-1 (node masking) / Stage-2 (Mordred) pretraining
prompts/         CoT text generation and prompt building
utils/           Molecular graph construction, embedding cache, similarity
data/            BACE loader, descriptor cache
scripts/         Training / HPO / pretraining drivers
notebooks/       Minimal, self-contained reproductions of paper figures
configs/hpo/     HPO best-params consumed by the benchmark driver
cache/           Pre-computed CoT texts + embeddings + descriptors (runs offline)
```

## Getting started

```bash
uv sync

# Headline BACE benchmark (ChebNet vs TAME vs TAME-Fusion)
uv run python scripts/bace_preliminary_results.py \
    --ckpt_dir <pretrained_checkpoints> \
    --tame_params_json configs/hpo/tame_bace/best_params.json \
    --fusion_params_json configs/hpo/tame_fusion_bace/best_params.json

# Or reproduce a paper figure directly from precomputed data (no training/API calls)
uv run jupyter notebook notebooks/preliminary_benchmark.ipynb
```

The `notebooks/` directory is the best entry point — each notebook is a minimal, self-contained reproduction of a paper figure, loading small precomputed CSV/NPZ summaries (no trained-checkpoint bundles, no torch/model imports needed):

- `preliminary_benchmark.ipynb` — PyG ChebNet vs TAME/TAME-Fusion graph-only ablations vs full models, 100 seeds, BACE scaffold split.
- `training_dynamics.ipynb` — TAME MoE gate contributions per epoch, and per-molecule gate weight distributions on the validation set.
- `pubready_analysis.ipynb` — CKA, SVD/rank, and Tanimoto/CosSim vs. prediction-shift correlation plots.

## Azure OpenAI setup (only to generate *new* CoT texts / embeddings)

```bash
cp .env.example .env
# Fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
```

Authentication supports both API key and Azure CLI (`DefaultAzureCredential`). Not needed for the cached datasets.

## License

MIT — see [LICENSE](LICENSE).
