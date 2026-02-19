#!/bin/bash
# SBATCH script for Solar cluster (guideline: "All actual jobs/experiments should be running using an SBATCH script").
# Same training as srun_gpt2t.sh but submitted with sbatch so the job is queued and has explicit resource limits.
#
# Submit from anywhere; script will cd to its own directory (Duolando repo root):
#   cd /path/to/Baselines/Salsa_Duolando && sbatch run_follower_gpt_salsa_8gpu.slurm
# Or from Duolando dir: sbatch run_follower_gpt_salsa_8gpu.slurm

#SBATCH -J gpt_salsa

#SBATCH --account=mars-lab
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00
#SBATCH --output=log/gpt_salsa_%j.out
#SBATCH --error=log/gpt_salsa_%j.err
# Optional: pin to a node with 8 GPUs (see sinfo). E.g. cs-venus-09 (8x A40), cs-venus-05 (8x rtx_a5000), cs-venus-02 (8x 2080_ti)
# #SBATCH --nodelist=cs-venus-09

# Solar workaround (guideline): avoid OOM from inherited limits
ulimit -Su unlimited

mkdir -p log

# Run from the directory where sbatch was run (Slurm sets SLURM_SUBMIT_DIR there).
# Required on Solar: job may start in $HOME on the compute node, so BASH_SOURCE-based cd can fail.
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
fi

# Conda (mars-lab / Payam)
source /project/rosie-lab/Payam/conda/miniconda3/etc/profile.d/conda.sh
conda activate duet

CONFIG="configs/follower_gpt_beta0.9_final_salsa.yaml"
echo "[INFO] CWD: $(pwd)"
echo "[INFO] Config: $CONFIG"
echo "[INFO] Logs: log/gpt_salsa_${SLURM_JOB_ID}.out / .err"

# Same command as srun_gpt2t.sh (resources already set by #SBATCH; srun runs in this allocation)
srun python -u main_gpt2t.py --config "$CONFIG" --train
