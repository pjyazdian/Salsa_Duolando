# Salsa_utils

Utilities and extensions for the Duolando baseline. More features will be added here over time.

## Visualization app

A Gradio app to load **DD100** or **Salsa** data, visualize ground-truth duet motion, run VQ-VAE reconstruction, and run inference with the pretrained Follower GPT and Follower GPT w. RL models.

**Run from the Duolando repo root:**

```bash
cd Baselines/Salsa_Duolando
python Salsa_utils/visualization/vis_app.py
```

- Use the **Dataset** radio to choose **DD100** or **Salsa**, then click **Load and index data**.
- For Salsa, build the cache first (see below). Optional: `--port 7861`, `--share`.

---

## Creating the Salsa cache

To use the **Salsa pair LMDB** (e.g. from Salsa-Agent) in the same pipeline as DD100, you must build a Duolando-compatible cache. Run the cache script **from the Salsa_Duolando repo root**.

### Prerequisites

- **LMDB**: Train and/or test pair LMDB directories (e.g. `lmdb_Salsa_pair/lmdb_train`, `lmdb_Salsa_pair/lmdb_test`).
- **Python deps**: `lmdb`, `pyarrow`, `numpy`, `scipy`, `torch`, `tqdm`. For SMPLX-based pos3d: `smplx` and an SMPLX model directory. For music features: run from the Duolando repo so the same pipeline as DD100 is used (librosa, essentia); otherwise use `--no_audio_feature` to write zero music.

### Commands: test and train splits

Build **test** cache (e.g. for the visualization app):

```bash
cd Baselines/Salsa_Duolando

python -m Salsa_utils.salsa_duolando_cache \
  --source_lmdb /path/to/lmdb_Salsa_pair/lmdb_test \
  --out_dir ./data/salsa_duolando \
  --split test \
  --no_smplx
```

Build **train** cache (e.g. for training later):

```bash
python -m Salsa_utils.salsa_duolando_cache \
  --source_lmdb /path/to/lmdb_Salsa_pair/lmdb_train \
  --out_dir ./data/salsa_duolando \
  --split train \
  --no_smplx
```

Use the same `--out_dir` so both splits live under `./data/salsa_duolando` (e.g. `motion/pos3d/train/`, `motion/pos3d/test/`, `music/feature/train/`, `music/feature/test/`).

### Options

| Option | Description |
|--------|-------------|
| `--no_smplx` | Build from 22-joint keypoints in the LMDB (no SMPLX model). Recommended if you don’t have SMPLX. |
| `--no_audio_feature` | Skip music feature extraction; write zero music. Use if Duolando’s music pipeline (librosa, essentia) is not available. |
| `--smplx_path PATH` | SMPLX model directory (or set env `SMPLX_MODEL_PATH`). Omit if using `--no_smplx`. |
| `--no_mp3` | Do not write `music/mp3/` WAV/MP3 files. |
| `--no_rot_x` | Do not apply +90° X rotation (default applies it for upright view). |

### After building

- In the Gradio app, select **Salsa** and click **Load and index data**. The app uses `./data/salsa_duolando` by default; set env **SALSA_DUOLANDO_CACHE** to the cache root (folder containing `motion/` and `music/`).

For the data contract, conversion details, and windowing options, see **SALSA_DUOLANDO_DATA.md** (internal reference).

---

## Training on Salsa

Train Duolando models from scratch on the Salsa cache. **Prerequisite:** build the Salsa cache for both train and test splits (see above). Run all commands from the **Salsa_Duolando** repo root (`Baselines/Salsa_Duolando`).

Optional: install Weights & Biases for logging (`pip install wandb`). If available, the motion VQ-VAE, translation VQ-VAE, Follower GPT, and RL training scripts log to wandb (projects: `duolando-motion-vqvae`, `duolando-transl-vqvae`, `duolando-follower-gpt`, `duolando-rl`; run name from config `expname`).

### Step 1: Motion VQ-VAE

Train the motion VQ-VAE (SepVQVAEXM) on Salsa motion data. This produces the codebook used by the Follower GPT later.

**Command:**

```bash
cd Baselines/Salsa_Duolando

python main_mix.py --config configs/sep_vqvaexm_full_final_salsa.yaml --train
```

- **Config:** `configs/sep_vqvaexm_full_final_salsa.yaml` — same as `sep_vqvaexm_full_final.yaml` except `data_root` points to `./data/salsa_duolando/motion` and `expname` is `motion_vqvae_salsa`.
- **Output:** Checkpoints under `experiments/motion_vqvae_salsa/ckpt/` (e.g. `epoch_20.pt`, `epoch_40.pt`). Training runs for 500 epochs by default; `save_per_epochs: 20` and `test_freq: 500` are set in the config.

### Step 2: Translation VQ-VAE

Train the **translation VQ-VAE** (relative translation between leader and follower roots). Uses the same Salsa cache (motion + music). Requires Step 1 only for the pipeline order; the translation model does not load the motion VQ-VAE.

**Command:**

```bash
cd Baselines/Salsa_Duolando

python main_transl.py --config configs/transl_vqvaex_final_salsa.yaml --train
```

- **Config:** `configs/transl_vqvaex_final_salsa.yaml` — same as `transl_vqvaex_final.yaml` except `music_root` and `data_root` point to `./data/salsa_duolando/music` and `./data/salsa_duolando/motion`, and `expname` is `transl_vqvae_salsa`.
- **Output:** Checkpoints under `experiments/transl_vqvae_salsa/ckpt/` (e.g. `epoch_20.pt`, `epoch_40.pt`). Default 500 epochs; `save_per_epochs: 20`, `test_freq: 500`.

### Step 3: Follower GPT

Train the **Follower GPT** (music-conditioned follower motion generation). Uses the Salsa cache and your trained **motion** and **translation** VQ-VAEs from Steps 1 and 2 (they are loaded frozen). Requires Steps 1 and 2 to be done; use checkpoints from those steps (e.g. `epoch_500.pt` or your latest, e.g. `epoch_20.pt`).

**Command:**

```bash
cd Baselines/Salsa_Duolando

python main_gpt2t.py --config configs/follower_gpt_beta0.9_final_salsa.yaml --train
```

- **Config:** `configs/follower_gpt_beta0.9_final_salsa.yaml` — same as `follower_gpt_beta0.9_final.yaml` except:
  - `data.train` / `data.test`: `music_root` and `data_root` → `./data/salsa_duolando/music` and `./data/salsa_duolando/motion`
  - `testing.music_source` → `./data/salsa_duolando/music/mp3/test/`
  - `vqvae_weight` → `./experiments/motion_vqvae_salsa/ckpt/epoch_500.pt`
  - `transl_vqvae_weight` → `./experiments/transl_vqvae_salsa/ckpt/epoch_500.pt`
  - `expname` → `follower_gpt_salsa`
- **VQ-VAE checkpoints:** If you have not trained motion/translation VQ-VAEs to epoch 500, edit the config and set `vqvae_weight` and `transl_vqvae_weight` to your latest checkpoint paths (e.g. `epoch_20.pt`, `epoch_40.pt`).
- **Output:** Checkpoints under `experiments/follower_gpt_salsa/ckpt/` (e.g. `epoch_10.pt`, `epoch_20.pt`). Default 250 epochs; `save_per_epochs: 10`, `test_freq: 500`. Use the app’s **Inference** tab with your trained motion and translation VQ-VAEs; the app still loads the pretrained Follower GPT by default—to use your trained GPT you would need to add a GPT checkpoint selector or replace the pretrained GPT checkpoint path.
- **Note:** If you have a single GPU, lower `batch_size` in the config (e.g. from 256 to 64 or 32) to avoid OOM.

### Solar cluster (8 GPUs)

For Step 3 (Follower GPT) on Solar with 8 GPUs, use **sbatch** (recommended) or the interactive **srun_gpt2t.sh** script.

**Option A — sbatch (recommended)**  
Submit the job; the script changes into the Duolando directory automatically, so you can submit from the project root or from `Baselines/Salsa_Duolando`:

1. From project root:  
   `cd /path/to/New_2025 && cd Baselines/Salsa_Duolando && sbatch run_follower_gpt_salsa_8gpu.slurm`  
   Or from Duolando:  
   `cd Baselines/Salsa_Duolando && sbatch run_follower_gpt_salsa_8gpu.slurm`
2. Set `#SBATCH --account=<your-lab-account>` in `run_follower_gpt_salsa_8gpu.slurm` (e.g. `3dlg-hcvc-lab` for 3DLG/HCVC lab; use your lab’s account if different).
3. Ensure conda path and env name in the script match your setup (`duet`).
4. Monitor: `squeue -u $USER`; cancel: `scancel <job_id>`; logs: `tail -f log/gpt_salsa_<job_id>.out`.

**Option B — srun_gpt2t.sh (interactive / same resources)**  
From `Baselines/Salsa_Duolando`, run:

```bash
./srun_gpt2t.sh configs/follower_gpt_beta0.9_final_salsa.yaml train 3dlg-hcvc-lab-long 8
```

This uses `srun` with the same resources (8 GPUs, 128G, 3 days). Prefer sbatch for long runs so the job is queued and survives disconnects.

**Difference from srun_gpt2t.sh:** The SBATCH script encodes partition, account, GPUs, memory, and time in `#SBATCH` and runs the same `srun python -u main_gpt2t.py --config ... --train` inside the allocation. No need to pass partition/gpunum on the command line.

### Step 4: Follower GPT w. RL (reinforcement learning)

Train the **RL** agent (actor-critic) on Salsa, starting from your trained Follower GPT. Uses the same Salsa cache and your **motion** and **translation** VQ-VAEs (frozen). Requires Steps 1, 2, and 3; the config loads a trained Follower GPT checkpoint as `init_weight`.

**Command (local / single node):**

```bash
cd Baselines/Salsa_Duolando

python main_ac_new.py --config configs/rl_final_debug_reward3_random_5mem_lr3e-5_salsa.yaml --train
```

- **Config:** `configs/rl_final_debug_reward3_random_5mem_lr3e-5_salsa.yaml` — Salsa data roots, `vqvae_weight` and `transl_vqvae_weight` point to `./experiments/motion_vqvae_salsa/ckpt/` and `./experiments/transl_vqvae_salsa/ckpt/` (e.g. `epoch_500.pt`), `init_weight` points to `./experiments/follower_gpt_salsa/ckpt/epoch_500.pt`, `expname`: `rl_salsa`, `testing.music_source`: Salsa mp3 test.
- **Checkpoints:** If your motion/translation VQ-VAEs or Follower GPT use different epoch numbers, edit the config: `vqvae_weight`, `transl_vqvae_weight`, and `init_weight` to your checkpoint paths.
- **Output:** Checkpoints under `experiments/rl_salsa/ckpt/` (e.g. `epoch_10.pt`, `epoch_50.pt`). Default 500 epochs; `save_per_epochs: 10`, `test_freq: 10`. Wandb project: `duolando-rl` (if `wandb` is installed and API key set).
- **Single GPU:** Lower `data.rl.batch_size` or `data.train.batch_size` in the config if needed.

**Solar cluster (8 GPUs):** Submit from `Baselines/Salsa_Duolando`:

```bash
cd Baselines/Salsa_Duolando
sbatch run_rl_salsa_8gpu.slurm
```

Same as Step 3: set `#SBATCH --account` and conda path in `run_rl_salsa_8gpu.slurm` if needed. Logs: `log/rl_salsa_<job_id>.out` / `.err`.