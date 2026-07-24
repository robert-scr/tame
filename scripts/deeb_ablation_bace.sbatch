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
OUT_DIR="${OUT_DIR:-/home/rschiller/Code/TAME-MPP/Rechnungen/deeb_ablation/e01}"
mkdir -p "$OUT_DIR" logs

## Graph backbone -- MUST match an existing chebnet_pt_d{H}_K{K}_L{L}_e25.pt --
## Default = the best fusion backbone (d128_K3_L3, set2set); change to match your checkpoint.
HIDDEN="${HIDDEN:-128}"
K="${K:-3}"
NUM_LAYERS="${NUM_LAYERS:-3}"
POOL="${POOL:-set2set}"
S2S="${S2S:-6}"
N_SEEDS="${N_SEEDS:-20}"

## Run the 2x2 factorial ----------------------------------------------------
python "$REPO_DIR/scripts/deeb_ablation_bace.py" \
        --out_dir "$OUT_DIR" \
        --ckpt_dir "$CKPT_DIR" \
        --ckpt_epoch_tag 25 \
        --gate_modes forced free \
        --desc_features rdkit rdkit_ecfp \
        --hidden "$HIDDEN" \
        --K "$K" \
        --num_layers "$NUM_LAYERS" \
        --pool "$POOL" \
        --set2set_n_iter "$S2S" \
        --n_seeds "$N_SEEDS" \
        --max_epochs 120 \
        --patience 20 \
        --es_metric val_auc \
        --splitter scaffold

echo -e "\n------------------------------------------------------------\n"
echo ">> Done!"
date
