#!/bin/bash
#SBATCH --job-name=bace_prelim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bace_prelim_%j.out
#SBATCH --error=logs/bace_prelim_%j.err
# --partition=gpu          # uncomment/adjust for your cluster

set -euo pipefail
mkdir -p logs

# --- adjust these paths to your cluster layout ---
CKPT_DIR="${CKPT_DIR:-checkpoints}"
TAME_PARAMS="${TAME_PARAMS:-Rechnungen/hpo/tame_predictor_best_params_v3_scaffold_s11.json}"
FUSION_PARAMS="${FUSION_PARAMS:-Rechnungen/hpo/tame_fusion_best_params_v1_scaffold_s11.json}"

# --gate_entropy_weight enforces per-element mixing (every modality contributes a little);
# --gate_sweep runs the TAME gate-target diagnostic (selection by validation).
uv run python scripts/bace_preliminary_results.py \
    --ckpt_dir "${CKPT_DIR}" \
    --tame_params_json "${TAME_PARAMS}" \
    --fusion_params_json "${FUSION_PARAMS}" \
    --ckpt_epoch_tag 25 \
    --max_epochs 120 \
    --patience 20 \
    --es_metric val_loss \
    --splitter scaffold \
    --gate_entropy_weight 0.05 \
    --gate_sweep \
    --sweep_seeds 5 \
    --out_dir benchmarking/results
