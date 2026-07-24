#!/bin/bash
#SBATCH -J tame_deeb_ablation_bace
#SBATCH -o logs/deeb_ablation_%j.out    # %j = JobID (submit from the repo root so logs/ exists)
#SBATCH -M tc
#SBATCH --partition g                   # GPU partition
#SBATCH --gres=gpu:1
#SBATCH -A rschiller
#SBATCH -t 1-00:00:00                   # 4 arms x --n_seeds TAME fits; ~hours on GPU, 1 day is ample
#SBATCH --no-requeue
#SBATCH --qos=7d
#SBATCH --cpus-per-task=8               # RAM / DataLoader / RDKit descriptor computation

echo ">> Starting TAME DEEB ablation (free/forced gate x RDKit/[RDKit||ECFP])"
date
echo -e "\n------------------------------------------------------------\n"

## Environment (mirrors the HPO submit script) -----------------------------
module load python/3.12.2
source /home/rschiller/Code/TAME-MPP/tame/bin/activate     # the "tame" virtualenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## Paths -- adjust to your cluster layout (env-var overridable) -------------
REPO_DIR="${REPO_DIR:-/home/rschiller/Code/TAME-MPP/tame}"        # where this repo lives on the cluster
CKPT_DIR="${CKPT_DIR:-/home/rschiller/Code/TAME-MPP/Rechnungen/checkpoints}"
OUT_DIR="${OUT_DIR:-/home/rschiller/Code/TAME-MPP/Rechnungen/deeb_ablation/e02}"
# Flat-TAME HPO best-params JSON. Drives arch (d64_K5_L3_sum), recipe AND the tuned graph-heavy
# gate config (target=0.64, balance=0.0012, entropy=0.061) -- so the checkpoint chebnet_pt_d64_K5_L3_e25.pt
# must exist in CKPT_DIR. Point this at your tame_hpo_v3 best-params file.
TAME_PARAMS="${TAME_PARAMS:-/home/rschiller/Code/TAME-MPP/Rechnungen/hpo/tame_bace_best_params.json}"
N_SEEDS="${N_SEEDS:-20}"
mkdir -p "$OUT_DIR" logs

## anchored = as HPO-tuned (target prevents desc collapse); balance_off = same but balance_weight=0
## (demonstrates the collapse the target prevents). desc: RDKit vs [RDKit || ECFP].
python "$REPO_DIR/scripts/deeb_ablation_bace.py" \
        --out_dir "$OUT_DIR" \
        --ckpt_dir "$CKPT_DIR" \
        --ckpt_epoch_tag 25 \
        --tame_params_json "$TAME_PARAMS" \
        --gate_modes anchored balance_off \
        --desc_features rdkit rdkit_ecfp \
        --n_seeds "$N_SEEDS" \
        --max_epochs 120 \
        --patience 20 \
        --es_metric val_auc \
        --splitter scaffold

echo -e "\n------------------------------------------------------------\n"
echo ">> Done!"
date
