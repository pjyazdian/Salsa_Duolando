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

Optional: install Weights & Biases for logging (`pip install wandb`). If available, the VQ-VAE training script will log loss, learning rate, and checkpoint events to wandb (project: `duolando-motion-vqvae`, run name: from config `expname`).

### Step 1: Motion VQ-VAE

Train the motion VQ-VAE (SepVQVAEXM) on Salsa motion data. This produces the codebook used by the Follower GPT later.

**Command:**

```bash
cd Baselines/Salsa_Duolando

python main_mix.py --config configs/sep_vqvaexm_full_final_salsa.yaml --train
```

- **Config:** `configs/sep_vqvaexm_full_final_salsa.yaml` — same as `sep_vqvaexm_full_final.yaml` except `data_root` points to `./data/salsa_duolando/motion` and `expname` is `motion_vqvae_salsa`.
- **Output:** Checkpoints under `experiments/motion_vqvae_salsa/ckpt/` (e.g. `epoch_20.pt`, `epoch_40.pt`). Training runs for 500 epochs by default; `save_per_epochs: 20` and `test_freq: 500` are set in the config.

Further steps (Transl VQ-VAE, Follower GPT, Follower GPT + RL) will be added here once this step is validated.