#!/bin/bash
# SBATCH script for RL training on Solar (Follower GPT w. RL on Salsa).
# Same structure as run_follower_gpt_salsa_8gpu.slurm; entry point: main_ac_new.py --train.
#
# Submit from Duolando dir so SLURM_SUBMIT_DIR is correct:
#   cd Baselines/Salsa_Duolando && sbatch run_rl_salsa_8gpu.slurm

#SBATCH -J rl_salsa

#SBATCH --account=mars-lab
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00
#SBATCH --output=log/rl_salsa_%j.out
#SBATCH --error=log/rl_salsa_%j.err
# #SBATCH --nodelist=cs-venus-09

ulimit -Su unlimited

mkdir -p log

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
fi

source /project/rosie-lab/Payam/conda/miniconda3/etc/profile.d/conda.sh
conda activate duet

WANDB_KEY_FILE="/project/rosie-lab/Payam/.wandb_api_key"
if [ -f "$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
fi

CONFIG="configs/rl_final_debug_reward3_random_5mem_lr3e-5_salsa.yaml"
echo "[INFO] CWD: $(pwd)"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Logs: log/rl_salsa_${SLURM_JOB_ID}.out / .err"

srun python -u main_ac_new.py --config "$CONFIG" --train
