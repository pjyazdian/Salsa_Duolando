"""
Gradio app for Duolando: load DD100 data, visualize ground truth, and run inference with pretrained models.
Run from the Salsa_Duolando repository root so that base code imports work:
  cd Baselines/Salsa_Duolando && python Salsa_utils/visualization/vis_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import yaml
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from easydict import EasyDict
except ImportError:
    class EasyDict(dict):
        def __getattr__(self, k):
            if k in self:
                v = self[k]
                return EasyDict(v) if isinstance(v, dict) else v
            raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v

# Add Duolando repo root so we can use base code without changing it
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _VIS_DIR not in sys.path:
    sys.path.insert(0, _VIS_DIR)
os.chdir(_REPO_ROOT)

# Local visualization helpers (no base code changes)
from visualization_utils import (
    pos3d_to_video,
    pos3d_to_video_gt_vs_recon,
    get_music_path_for_sample,
    get_salsa_music_path,
    dataset_pos3d_to_display_pos3d,
)

# Base Duolando imports (used as-is)
from datasets.dd100lf_all2 import DD100lfAll

# Lazy imports for heavy deps (gpt2t, models) only when inference tab is used
_agent = None
_config = None
_agent_rl = None
_config_rl = None


def _load_config():
    """Load Duolando GPT config and point to pretrained checkpoints."""
    global _config
    config_path = os.path.join(_REPO_ROOT, "configs", "follower_gpt_beta0.9_final.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    # Use pretrained folder for all weights
    cfg.vqvae_weight = os.path.join(_REPO_ROOT, "experiments", "pretrained", "motion_vqvae", "ckpt", "epoch_500.pt")
    cfg.transl_vqvae_weight = os.path.join(_REPO_ROOT, "experiments", "pretrained", "transl_vqvae", "ckpt", "epoch_500.pt")
    cfg.testing.ckpt_epoch = 500
    cfg.testing.music_source = os.path.join(_REPO_ROOT, "data", "music", "mp3", "test")
    # So that MCTall's ckptdir points to pretrained GPT
    cfg.expname = os.path.join("pretrained", "follower_gpt")
    _config = cfg
    return cfg


def _get_agent():
    """Build MCTall agent and load pretrained weights (lazy, once per inference use)."""
    global _agent, _config
    if _agent is not None:
        return _agent
    from gpt2t import MCTall

    if _config is None:
        _load_config()
    _agent = MCTall(_config)
    # Load pretrained weights (overwrite any from _dir_setting)
    checkpoint = torch.load(_config.vqvae_weight, map_location="cpu")
    _agent.model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_t = torch.load(_config.transl_vqvae_weight, map_location="cpu")
    _agent.model3.load_state_dict(checkpoint_t["model"], strict=True)
    ckpt_path = os.path.join(_REPO_ROOT, "experiments", "pretrained", "follower_gpt", "ckpt", f"epoch_{_config.testing.ckpt_epoch}.pt")
    checkpoint_gpt = torch.load(ckpt_path, map_location="cpu")
    _agent.model2.load_state_dict(checkpoint_gpt["model"])
    _agent.model.eval()
    _agent.model2.eval()
    _agent.model3.eval()
    _agent.device = torch.device("cuda" if _config.cuda else "cpu")
    return _agent


def _load_config_rl():
    """Load RL config and point to pretrained checkpoints (follower GPT w. RL)."""
    global _config_rl
    config_path = os.path.join(_REPO_ROOT, "configs", "rl_final_debug_reward3_random_5mem_lr3e-5.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    cfg.vqvae_weight = os.path.join(_REPO_ROOT, "experiments", "pretrained", "motion_vqvae", "ckpt", "epoch_500.pt")
    cfg.transl_vqvae_weight = os.path.join(_REPO_ROOT, "experiments", "pretrained", "transl_vqvae", "ckpt", "epoch_500.pt")
    cfg.testing.ckpt_epoch = 50
    cfg.testing.music_source = os.path.join(_REPO_ROOT, "data", "music", "mp3", "test")
    cfg.expname = os.path.join("pretrained", "rl")
    _config_rl = cfg
    return cfg


def _get_agent_rl():
    """Build AC (RL) agent and load pretrained weights (lazy, once per inference use)."""
    global _agent_rl, _config_rl
    if _agent_rl is not None:
        return _agent_rl
    from actor_critic_new import AC

    if _config_rl is None:
        _load_config_rl()
    _agent_rl = AC(_config_rl)
    checkpoint = torch.load(_config_rl.vqvae_weight, map_location="cpu")
    _agent_rl.model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_t = torch.load(_config_rl.transl_vqvae_weight, map_location="cpu")
    _agent_rl.model3.load_state_dict(checkpoint_t["model"], strict=True)
    ckpt_path = os.path.join(_REPO_ROOT, "experiments", "pretrained", "rl", "ckpt", f"epoch_{_config_rl.testing.ckpt_epoch}.pt")
    checkpoint_gpt = torch.load(ckpt_path, map_location="cpu")
    _agent_rl.model2.load_state_dict(checkpoint_gpt["model"])
    _agent_rl.model.eval()
    _agent_rl.model2.eval()
    _agent_rl.model3.eval()
    _agent_rl.device = torch.device("cuda" if _config_rl.cuda else "cpu")
    return _agent_rl


def _run_single_inference(agent, batch):
    """Run one batch through the Duolando eval pipeline; returns (pose_follower, pose_leader) numpy."""
    from gpt2t import get_closest_rotmat, rotmat2aa

    config = agent.config
    device = agent.device
    vqvae, gpt, transl_vqvae = agent.model, agent.model2, agent.model3

    music_seq = batch["music"].to(device)
    pose_seql = batch["pos3dl"].to(device)
    pose_seqf = batch["pos3df"].to(device)

    lftransl = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0
    transll = pose_seql[:, :, :3].clone()
    transll = transll - transll[:, :1, :3].clone()

    pose_seql[:, :, :3] = 0
    pose_seqf[:, :, :3] = 0

    quants_predl = vqvae.module.encode(pose_seql)
    quants_predf = vqvae.module.encode(pose_seqf[:, : config.ds_rate])
    quants_transl = transl_vqvae.module.encode(lftransl)

    if isinstance(quants_predf, tuple):
        y = tuple(quants_predl[i][0].clone() for i in range(len(quants_predf)))
        x = tuple(quants_predf[i][0][:, :1].clone() for i in range(len(quants_predf)))
        transl = (quants_transl[0][:, :1],)
    else:
        y = quants_predl[0].clone()
        x = quants_predf[0][:, :1].clone()
        transl = (quants_transl[0][:, :1],)

    zs, transl_z = gpt.module.sample(
        x + transl,
        cond=(music_seq[:, config.music_motion_rate :],) + y,
        shift=config.sample_shift if hasattr(config, "sample_shift") else None,
    )

    pose_sample, rotmat_sample, vel_sample = vqvae.module.decode(zs)
    lf_transl = transl_vqvae.module.decode(transl_z)

    pose_sample[:, :, :3] = transll[:, : lf_transl.size(1)].clone() + lf_transl.clone() / 20.0
    pose_sample = pose_sample.cpu().data.numpy()

    left_twist = pose_sample[:, :, 60:63]
    pose_sample[:, :, 75:120] = pose_sample[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))
    right_twist = pose_sample[:, :, 63:66]
    pose_sample[:, :, 120:165] = pose_sample[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))
    root = pose_sample[:, :, :3]
    pose_sample = pose_sample + np.tile(root, (1, 1, 55))
    pose_sample[:, :, :3] = root

    pose_seql[:, :, :3] = transll.clone()
    pose_seql = pose_seql.cpu().data.numpy()
    left_twist = pose_seql[:, :, 60:63]
    pose_seql[:, :, 75:120] = pose_seql[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))
    right_twist = pose_seql[:, :, 63:66]
    pose_seql[:, :, 120:165] = pose_seql[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))
    root = pose_seql[:, :, :3]
    pose_seql = pose_seql + np.tile(root, (1, 1, 55))
    pose_seql[:, :, :3] = root

    return pose_sample[0], pose_seql[0]  # (T, 165) each


def _postprocess_decoded_pose(pose_np, transll):
    """Convert decoded (root-zeroed) pose to display format: add root, unscale hands. pose_np: (1, T, 165), transll: (1, T, 3)."""
    T_dec = pose_np.shape[1]
    T_root = transll.shape[1]
    T = min(T_dec, T_root)
    pose = pose_np[:, :T, :].copy()
    root = transll[:, :T, :].clone().detach().cpu().numpy()
    pose[:, :, :3] = root
    left_twist = pose[:, :, 60:63]
    pose[:, :, 75:120] = pose[:, :, 75:120] * 0.1 + np.tile(left_twist, (1, 1, 15))
    right_twist = pose[:, :, 63:66]
    pose[:, :, 120:165] = pose[:, :, 120:165] * 0.1 + np.tile(right_twist, (1, 1, 15))
    root = pose[:, :, :3].copy()
    pose = pose + np.tile(root, (1, 1, 55))
    pose[:, :, :3] = root
    return pose[0]  # (T, 165)


def _format_quants_for_display(quants_pose, name):
    """Format pose quants (tuple of 4 lists of tensors) for display."""
    parts = ["up", "down", "lhand", "rhand"]
    lines = [f"**{name}** (4 codebooks):"]
    for i, part in enumerate(parts):
        z = quants_pose[i]
        if isinstance(z, (list, tuple)):
            z = z[0]
        z = z.detach().cpu().numpy()
        shape = z.shape
        flat = z.flatten()
        n = len(flat)
        head = flat[:12].tolist() if n >= 12 else flat.tolist()
        tail = flat[-8:].tolist() if n > 20 else []
        tok_str = str(head) + (" ... " + str(tail) if tail else "")
        lines.append(f"  {part}: shape {shape}, num_tokens={n}, first/last → {tok_str}")
    return "\n".join(lines)


def _format_transl_quants_for_display(quants_transl):
    """Format translation quants (list of tensors) for display."""
    z = quants_transl[0]
    z = z.detach().cpu().numpy()
    shape = z.shape
    flat = z.flatten()
    n = len(flat)
    head = flat[:12].tolist() if n >= 12 else flat.tolist()
    tail = flat[-8:].tolist() if n > 20 else []
    tok_str = str(head) + (" ... " + str(tail) if tail else "")
    return f"**Translation (relation)** (1 codebook): shape {shape}, num_tokens={n}, first/last → {tok_str}"


def _get_vqvae_model_info():
    """Build a short model info string from config (pose + translation VQ-VAEs)."""
    if _config is None:
        _load_config()
    c = _config
    pose_c = getattr(c, "structure", None)
    pose_l_bins = getattr(pose_c, "l_bins", 512) if pose_c else 512
    pose_emb = getattr(getattr(pose_c, "up_half", None), "emb_width", 512) if pose_c else 512
    pose_downs = getattr(getattr(pose_c, "up_half", None), "downs_t", [2]) if pose_c else [2]
    trans_c = getattr(c, "structure_transl_vqvae", None)
    trans_l_bins = getattr(trans_c, "l_bins", 512) if trans_c else 512
    trans_emb = getattr(trans_c, "emb_width", 512) if trans_c else 512
    trans_input = getattr(trans_c, "input_dim", 3) if trans_c else 3
    trans_downs = getattr(trans_c, "downs_t", [2]) if trans_c else [2]
    lines = [
        "**Pose VQ-VAE** (SepVQVAEXM)",
        "- Codebooks: 4 (up, down, lhand, rhand)",
        f"- Vocab size (l_bins): {pose_l_bins}",
        f"- Embed dim (emb_width): {pose_emb}",
        f"- Temporal downsample: downs_t = {pose_downs}",
        "",
        "**Translation VQ-VAE** (relation)",
        "- Codebooks: 1",
        f"- Vocab size (l_bins): {trans_l_bins}",
        f"- Embed dim (emb_width): {trans_emb}",
        f"- Input dim: {trans_input} (relative x, y, z)",
        f"- Temporal downsample: downs_t = {trans_downs}",
    ]
    return "\n".join(lines)


def _run_vqvae_reconstruction(agent, batch):
    """
    Run full VQ-VAE pipeline (pose + translation) so everything is from tokens, like at inference.
    Returns (recon_leader, recon_follower, gt_transl, recon_transl, tokens_text).
    - gt_transl, recon_transl: (T, 3) in world units (follower root - leader root) for charting.
    - tokens_text: string describing the extracted tokens used for reconstruction.
    """
    config = agent.config
    device = agent.device
    vqvae = agent.model
    transl_vqvae = agent.model3

    pose_seql = batch["pos3dl"].to(device)
    pose_seqf = batch["pos3df"].to(device)

    # Relation: relative translation (follower - leader) * 20, before zeroing roots
    lftransl_gt = (pose_seqf[:, :, :3] - pose_seql[:, :, :3]).clone() * 20.0

    transll = pose_seql[:, :, :3].clone()
    transll = transll - transll[:, :1, :3].clone()

    pose_seql[:, :, :3] = 0
    pose_seqf[:, :, :3] = 0

    # Pose VQ-VAE: tokenize + detokenize both bodies
    quants_l = vqvae.module.encode(pose_seql)
    quants_f = vqvae.module.encode(pose_seqf)
    recon_l, _, _ = vqvae.module.decode(quants_l)
    recon_f, _, _ = vqvae.module.decode(quants_f)

    # Translation VQ-VAE: tokenize + detokenize relative translation (relation)
    quants_transl = transl_vqvae.module.encode(lftransl_gt)
    lf_transl_recon = transl_vqvae.module.decode(quants_transl)

    # Build tokens display (before moving to numpy for postprocess)
    tokens_parts = [
        "**Tokens used for this reconstruction:**",
        "",
        _format_quants_for_display(quants_l, "Leader pose"),
        "",
        _format_quants_for_display(quants_f, "Follower pose"),
        "",
        _format_transl_quants_for_display(quants_transl),
    ]
    tokens_text = "\n".join(tokens_parts)

    recon_l = recon_l.cpu().data.numpy()
    recon_f = recon_f.cpu().data.numpy()
    lf_transl_recon = lf_transl_recon.cpu()
    transll_np = transll.cpu()

    T = min(recon_l.shape[1], recon_f.shape[1], transll_np.shape[1], lf_transl_recon.shape[1])
    transll_T = transll_np[:, :T, :]
    # Follower root from tokens: leader_root + decoded_relative_translation (same as inference)
    translf_recon = transll_T + lf_transl_recon[:, :T, :] / 20.0

    recon_leader = _postprocess_decoded_pose(recon_l[:, :T, :], transll_T)
    recon_follower = _postprocess_decoded_pose(recon_f[:, :T, :], translf_recon)

    gt_transl = (lftransl_gt[:, :T, :] / 20.0).detach().cpu().numpy()[0]
    recon_transl = (lf_transl_recon[:, :T, :] / 20.0).detach().cpu().numpy()[0]
    return recon_leader, recon_follower, gt_transl, recon_transl, tokens_text


def _plot_translation_gt_vs_recon(gt_transl, recon_transl):
    """Plot ground truth vs reconstructed relative translation (follower - leader root) for x, y, z."""
    T = gt_transl.shape[0]
    frames = np.arange(T)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axes):
        ax.plot(frames, gt_transl[:, i], label="Ground truth", color="C0", alpha=0.8)
        ax.plot(frames, recon_transl[:, i], label="VQ-VAE recon", color="C1", alpha=0.8, linestyle="--")
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Relative translation (follower − leader root)", fontsize=10)
    plt.tight_layout()
    return fig


# --------------- State and paths ---------------
DATA_ROOT = os.path.join(_REPO_ROOT, "data", "motion")
MUSIC_ROOT = os.path.join(_REPO_ROOT, "data", "music")
MUSIC_SOURCE = os.path.join(_REPO_ROOT, "data", "music", "mp3", "test")
# Salsa cache (built by Salsa_utils.salsa_duolando_cache); use env or default under data/salsa_duolando
SALSA_CACHE_ROOT = os.environ.get("SALSA_DUOLANDO_CACHE", os.path.join(_REPO_ROOT, "data", "salsa_duolando"))
SALSA_MOTION_ROOT = os.path.join(SALSA_CACHE_ROOT, "motion")
SALSA_MUSIC_ROOT = os.path.join(SALSA_CACHE_ROOT, "music")
SALSA_MUSIC_MP3 = os.path.join(SALSA_CACHE_ROOT, "music", "mp3", "test")

_dataset = None
_sample_names = []
_use_salsa_dataset = False
_width = 960
_height = 540

TOKENS_TO_POSE_EXPLANATION = """
**From tokens to pose 3D**

- **Pose VQ-VAE** has 4 codebooks (upper body, lower body, left hand, right hand). Each decoder turns its token sequence back into a piece of the body (joint positions in a root-zeroed space). These 4 parts are **assembled** into one full body pose (55 joints × 3) per frame.
- **Translation VQ-VAE** has 1 codebook. Its decoder turns the relation tokens into a sequence of **relative translation** (follower root − leader root, scaled).  
- **Combining for animation:** For the leader we take the decoded pose and add the **leader root trajectory** (from data) to get world-space poses. For the follower we take the decoded pose and add **leader root + decoded relative translation** so the follower stays correctly positioned relative to the leader. Hand joints are then unscaled (×0.1 + wrist) and roots are broadcast to all joints. That yields the final 3D poses used for the skeleton animation.
"""


def _list_motion_vqvae_checkpoints():
    """List (label, path) for motion VQ-VAE: pretrained + all .pt in experiments/*motion_vqvae*/ckpt/."""
    experiments_dir = os.path.join(_REPO_ROOT, "experiments")
    choices = []
    pretrained_path = os.path.join(experiments_dir, "pretrained", "motion_vqvae", "ckpt", "epoch_500.pt")
    if os.path.isfile(pretrained_path):
        choices.append(("Pretrained (motion_vqvae)", pretrained_path))
    if os.path.isdir(experiments_dir):
        for name in sorted(os.listdir(experiments_dir)):
            if "motion_vqvae" not in name:
                continue
            d = os.path.join(experiments_dir, name)
            if not os.path.isdir(d):
                continue
            ckpt_dir = os.path.join(d, "ckpt")
            if not os.path.isdir(ckpt_dir):
                continue
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith(".pt"):
                    choices.append((f"{name} / {f}", os.path.join(ckpt_dir, f)))
    return choices


# Cached list for dropdown and lookup (label -> path)
_VQVAE_CKPT_CHOICES = _list_motion_vqvae_checkpoints()


def _list_transl_vqvae_checkpoints():
    """List (label, path) for translation VQ-VAE: pretrained + all .pt in experiments/*transl_vqvae*/ckpt/."""
    experiments_dir = os.path.join(_REPO_ROOT, "experiments")
    choices = []
    pretrained_path = os.path.join(experiments_dir, "pretrained", "transl_vqvae", "ckpt", "epoch_500.pt")
    if os.path.isfile(pretrained_path):
        choices.append(("Pretrained (transl_vqvae)", pretrained_path))
    if os.path.isdir(experiments_dir):
        for name in sorted(os.listdir(experiments_dir)):
            if "transl_vqvae" not in name:
                continue
            d = os.path.join(experiments_dir, name)
            if not os.path.isdir(d):
                continue
            ckpt_dir = os.path.join(d, "ckpt")
            if not os.path.isdir(ckpt_dir):
                continue
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith(".pt"):
                    choices.append((f"{name} / {f}", os.path.join(ckpt_dir, f)))
    return choices


_TRANSL_VQVAE_CKPT_CHOICES = _list_transl_vqvae_checkpoints()


def _list_follower_gpt_checkpoints():
    """List (label, path) for Follower GPT: pretrained (epoch_500) + all .pt in experiments/follower_gpt_salsa/ckpt/."""
    experiments_dir = os.path.join(_REPO_ROOT, "experiments")
    choices = []
    pretrained_path = os.path.join(experiments_dir, "pretrained", "follower_gpt", "ckpt", "epoch_500.pt")
    if os.path.isfile(pretrained_path):
        choices.append(("Pretrained (follower_gpt)", pretrained_path))
    salsa_ckpt_dir = os.path.join(experiments_dir, "follower_gpt_salsa", "ckpt")
    if os.path.isdir(salsa_ckpt_dir):
        for f in sorted(os.listdir(salsa_ckpt_dir)):
            if f.endswith(".pt"):
                choices.append((f"follower_gpt_salsa / {f}", os.path.join(salsa_ckpt_dir, f)))
    return choices


_GPT_CKPT_CHOICES = _list_follower_gpt_checkpoints()


def _list_rl_checkpoints():
    """List (label, path) for Follower GPT w. RL: pretrained (epoch_50) + all .pt in experiments/rl_salsa/ckpt/."""
    experiments_dir = os.path.join(_REPO_ROOT, "experiments")
    choices = []
    pretrained_path = os.path.join(experiments_dir, "pretrained", "rl", "ckpt", "epoch_50.pt")
    if os.path.isfile(pretrained_path):
        choices.append(("Pretrained (rl)", pretrained_path))
    salsa_ckpt_dir = os.path.join(experiments_dir, "rl_salsa", "ckpt")
    if os.path.isdir(salsa_ckpt_dir):
        for f in sorted(os.listdir(salsa_ckpt_dir)):
            if f.endswith(".pt"):
                choices.append((f"rl_salsa / {f}", os.path.join(salsa_ckpt_dir, f)))
    return choices


_RL_CKPT_CHOICES = _list_rl_checkpoints()


def load_and_index_data(dataset_choice, progress=gr.Progress()):
    """Load dataset (DD100 or Salsa cache) and fill sample list. Returns dropdown choices and status."""
    global _dataset, _sample_names, _use_salsa_dataset
    use_salsa = (dataset_choice == "Salsa")
    _use_salsa_dataset = use_salsa
    music_root = SALSA_MUSIC_ROOT if use_salsa else MUSIC_ROOT
    motion_root = SALSA_MOTION_ROOT if use_salsa else DATA_ROOT
    try:
        progress(0.1, desc="Loading dataset...")
        if use_salsa:
            # Diagnose Salsa paths so user can fix cache location
            pos3d_dir = os.path.join(motion_root, "pos3d", "test")
            feature_dir = os.path.join(music_root, "feature", "test")
            pos3d_exists = os.path.isdir(pos3d_dir)
            feature_exists = os.path.isdir(feature_dir)
            n_pos3d = len([f for f in os.listdir(pos3d_dir) if f.endswith(".npy")]) if pos3d_exists else 0
            n_feature = len([f for f in os.listdir(feature_dir) if f.endswith(".npy")]) if feature_exists else 0
            if not pos3d_exists or not feature_exists or n_pos3d == 0 or n_feature == 0:
                msg = (
                    f"No samples found. Salsa cache paths (split=test):\n"
                    f"  motion/pos3d: {pos3d_dir}\n  exists={pos3d_exists}, npy files={n_pos3d}\n"
                    f"  music/feature: {feature_dir}\n  exists={feature_exists}, npy files={n_feature}\n"
                    f"Build cache: python -m Salsa_utils.salsa_duolando_cache --source_lmdb <path> --out_dir {os.path.dirname(motion_root)} --split test\n"
                    f"Or set SALSA_DUOLANDO_CACHE to the cache root (folder containing motion/ and music/)."
                )
                return gr.update(choices=[], value=None), msg
        _dataset = DD100lfAll(
            music_root,
            motion_root,
            split="test",
            interval=None,
            dtype="pos3d",
            move=4,
            music_dance_rate=1,
        )
        _sample_names = list(_dataset.names)
        progress(1.0, desc="Done")
        if not _sample_names:
            hint = "Salsa: run Salsa_utils.salsa_duolando_cache and set split=test, or check SALSA_DUOLANDO_CACHE." if use_salsa else "Check data/motion/pos3d/test and data/music/feature/test."
            return gr.update(choices=[], value=None), f"No samples found. {hint}"
        return gr.update(choices=_sample_names, value=_sample_names[0]), f"Loaded {len(_sample_names)} samples ({dataset_choice}). Pick a sample above."
    except Exception as e:
        import traceback
        return gr.update(choices=[], value=None), f"Error loading data: {e}\n{traceback.format_exc()}"


def visualize_ground_truth(sample_name, progress=gr.Progress()):
    """Render GT duet video and return video path and audio path for the selected sample."""
    if _dataset is None:
        return None, None, "Load and index data first (click 'Load and index data')."
    if not sample_name or sample_name not in _sample_names:
        return None, None, "Please select a sample from the dropdown."
    try:
        idx = _sample_names.index(sample_name)
        progress(0.2, desc="Loading sample...")
        item = _dataset[idx]
        # Dataset pos3d is preprocessed (hands scaled *10, body relative to root); convert to display format
        pos3df = dataset_pos3d_to_display_pos3d(item["pos3df"])
        pos3dl = dataset_pos3d_to_display_pos3d(item["pos3dl"])
        progress(0.5, desc="Rendering video...")
        out_path = pos3d_to_video(
            pos3df,
            pos3dl,
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_gt_{sample_name.replace('/', '_')}.mp4"),
        )
        if _use_salsa_dataset:
            music_path = get_salsa_music_path(SALSA_MUSIC_MP3, sample_name)
        else:
            music_path = get_music_path_for_sample(MUSIC_SOURCE, sample_name)
            if not os.path.isfile(music_path):
                music_path = None
        progress(1.0, desc="Done")
        return out_path, music_path, "Ground truth visualization ready. Play video and audio below."
    except Exception as e:
        import traceback
        return None, None, f"Error: {e}\n{traceback.format_exc()}"


def visualize_vqvae_reconstruction(sample_name, motion_ckpt_choice, transl_ckpt_choice, progress=gr.Progress()):
    """Encode selected sample through pose + translation VQ-VAEs (tokenize + detokenize) and visualize reconstruction.
    motion_ckpt_choice / transl_ckpt_choice: labels from Motion VQ-VAE and Translation VQ-VAE dropdowns.
    """
    if _dataset is None:
        return None, None, None, None, None, "Load and index data first (click 'Load and index data')."
    if not sample_name or sample_name not in _sample_names:
        return None, None, None, None, None, "Please select a sample from the dropdown."
    if not motion_ckpt_choice:
        return None, None, None, None, None, "Please select a Motion VQ-VAE checkpoint."
    if not transl_ckpt_choice:
        return None, None, None, None, None, "Please select a Translation VQ-VAE checkpoint."
    try:
        motion_path = next((p for label, p in _VQVAE_CKPT_CHOICES if label == motion_ckpt_choice), None)
        transl_path = next((p for label, p in _TRANSL_VQVAE_CKPT_CHOICES if label == transl_ckpt_choice), None)
        if not motion_path or not os.path.isfile(motion_path):
            return None, None, None, None, None, f"Checkpoint not found: {motion_ckpt_choice}"
        if not transl_path or not os.path.isfile(transl_path):
            return None, None, None, None, None, f"Checkpoint not found: {transl_ckpt_choice}"
        idx = _sample_names.index(sample_name)
        progress(0.05, desc="Loading model...")
        agent = _get_agent()
        progress(0.12, desc="Loading motion VQ-VAE checkpoint...")
        checkpoint = torch.load(motion_path, map_location="cpu")
        agent.model.load_state_dict(checkpoint["model"], strict=True)
        agent.model.eval()
        progress(0.2, desc="Loading translation VQ-VAE checkpoint...")
        checkpoint_t = torch.load(transl_path, map_location="cpu")
        agent.model3.load_state_dict(checkpoint_t["model"], strict=True)
        agent.model3.eval()
        progress(0.3, desc="Loading sample...")
        item = _dataset[idx]
        batch = {
            "pos3dl": torch.from_numpy(item["pos3dl"].astype(np.float32)).unsqueeze(0),
            "pos3df": torch.from_numpy(item["pos3df"].astype(np.float32)).unsqueeze(0),
        }
        progress(0.45, desc="Pose + translation VQ-VAE encode + decode...")
        recon_leader, recon_follower, gt_transl, recon_transl, tokens_text = _run_vqvae_reconstruction(agent, batch)
        progress(0.6, desc="Rendering reconstruction video...")
        out_path = pos3d_to_video(
            recon_follower,
            recon_leader,
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_vqvae_recon_{sample_name.replace('/', '_')}.mp4"),
        )
        progress(0.75, desc="Rendering GT vs Recon overlay...")
        gt_follower = dataset_pos3d_to_display_pos3d(item["pos3df"])
        gt_leader = dataset_pos3d_to_display_pos3d(item["pos3dl"])
        T = min(len(gt_follower), len(gt_leader), len(recon_follower), len(recon_leader))
        overlay_path = pos3d_to_video_gt_vs_recon(
            gt_follower[:T],
            gt_leader[:T],
            recon_follower[:T],
            recon_leader[:T],
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_vqvae_gt_vs_recon_{sample_name.replace('/', '_')}.mp4"),
        )
        progress(0.85, desc="Building relation chart...")
        fig = _plot_translation_gt_vs_recon(gt_transl, recon_transl)
        if _use_salsa_dataset:
            music_path = get_salsa_music_path(SALSA_MUSIC_MP3, sample_name)
        else:
            music_path = get_music_path_for_sample(MUSIC_SOURCE, sample_name)
            if not os.path.isfile(music_path):
                music_path = None
        progress(1.0, desc="Done")
        msg = "Full VQ-VAE reconstruction ready. First video: recon only. Second: GT (shadow) vs recon (solid). Chart: relation GT vs recon."
        return out_path, overlay_path, music_path, fig, tokens_text, msg
    except Exception as e:
        import traceback
        return None, None, None, None, None, f"Error: {e}\n{traceback.format_exc()}"


def _ensure_music_54(music: np.ndarray, n_music: int = 54) -> np.ndarray:
    """
    Ensure music has shape (T, 54) for GPT cond_emb. Uses Duolando's exact feature dims only.
    If the array is transposed (54, T), transpose to (T, 54). If dim is not 54, raise.
    """
    music = np.asarray(music, dtype=np.float32)
    if music.ndim == 1:
        music = music.reshape(-1, 1)
    if music.shape[0] == n_music and music.shape[1] != n_music:
        music = music.T
    if music.shape[-1] != n_music:
        raise ValueError(
            f"Music feature must have exactly 54 dimensions (same as Duolando: mfcc+mfcc_delta+chroma_cqt+onset_env+onset_beat). "
            f"Got shape {music.shape}. Rebuild the Salsa cache with Duolando's pipeline: run salsa_duolando_cache from the Duolando repo so "
            f"_prepare_music_data.signal_to_feature and extractor.py are used (same as DD100)."
        )
    return music


def run_inference(sample_name, motion_ckpt_choice, transl_ckpt_choice, gpt_ckpt_choice, rl_ckpt_choice, progress=gr.Progress()):
    """Run both Follower GPT and Follower GPT w. RL on the selected sample; return both videos + overlay videos + audio.
    Uses the selected Motion VQ-VAE, Translation VQ-VAE, Follower GPT, and RL checkpoints.
    """
    if _dataset is None:
        return None, None, None, None, None, "Load and index data first."
    if not sample_name or sample_name not in _sample_names:
        return None, None, None, None, None, "Please select a sample from the dropdown."
    if not motion_ckpt_choice or not transl_ckpt_choice:
        return None, None, None, None, None, "Please select both Motion VQ-VAE and Translation VQ-VAE checkpoints."
    if not gpt_ckpt_choice:
        return None, None, None, None, None, "Please select a Follower GPT checkpoint."
    if not rl_ckpt_choice:
        return None, None, None, None, None, "Please select a Follower GPT w. RL checkpoint."
    try:
        motion_path = next((p for label, p in _VQVAE_CKPT_CHOICES if label == motion_ckpt_choice), None)
        transl_path = next((p for label, p in _TRANSL_VQVAE_CKPT_CHOICES if label == transl_ckpt_choice), None)
        gpt_path = next((p for label, p in _GPT_CKPT_CHOICES if label == gpt_ckpt_choice), None)
        rl_path = next((p for label, p in _RL_CKPT_CHOICES if label == rl_ckpt_choice), None)
        if not motion_path or not os.path.isfile(motion_path):
            return None, None, None, None, None, f"Checkpoint not found: {motion_ckpt_choice}"
        if not transl_path or not os.path.isfile(transl_path):
            return None, None, None, None, None, f"Checkpoint not found: {transl_ckpt_choice}"
        if not gpt_path or not os.path.isfile(gpt_path):
            return None, None, None, None, None, f"Checkpoint not found: {gpt_ckpt_choice}"
        if not rl_path or not os.path.isfile(rl_path):
            return None, None, None, None, None, f"Checkpoint not found: {rl_ckpt_choice}"
        idx = _sample_names.index(sample_name)
        progress(0.05, desc="Loading sample...")
        item = _dataset[idx]
        gt_follower = dataset_pos3d_to_display_pos3d(item["pos3df"])
        gt_leader = dataset_pos3d_to_display_pos3d(item["pos3dl"])
        music = _ensure_music_54(item["music"])
        batch = {
            "music": torch.from_numpy(music).unsqueeze(0),
            "pos3dl": torch.from_numpy(item["pos3dl"].astype(np.float32)).unsqueeze(0),
            "pos3df": torch.from_numpy(item["pos3df"].astype(np.float32)).unsqueeze(0),
            "rotmatl": torch.from_numpy(item["rotmatl"].astype(np.float32)).unsqueeze(0),
            "rotmatf": torch.from_numpy(item["rotmatf"].astype(np.float32)).unsqueeze(0),
            "fname": [sample_name],
        }

        # 1) Follower GPT (uses selected GPT checkpoint)
        progress(0.15, desc="Loading Follower GPT...")
        agent_gpt = _get_agent()
        progress(0.2, desc="Loading motion & translation VQ-VAEs...")
        ckpt_m = torch.load(motion_path, map_location="cpu")
        agent_gpt.model.load_state_dict(ckpt_m["model"], strict=True)
        agent_gpt.model.eval()
        ckpt_t = torch.load(transl_path, map_location="cpu")
        agent_gpt.model3.load_state_dict(ckpt_t["model"], strict=True)
        agent_gpt.model3.eval()
        progress(0.28, desc="Loading Follower GPT checkpoint...")
        checkpoint_gpt = torch.load(gpt_path, map_location="cpu")
        agent_gpt.model2.load_state_dict(checkpoint_gpt["model"])
        agent_gpt.model2.eval()
        progress(0.35, desc="Running Follower GPT inference...")
        pose_follower_gpt, pose_leader = _run_single_inference(agent_gpt, batch)
        progress(0.5, desc="Rendering GPT video...")
        out_gpt = pos3d_to_video(
            pose_follower_gpt,
            pose_leader,
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_gpt_{sample_name.replace('/', '_')}.mp4"),
        )
        T_gpt = min(len(gt_follower), len(gt_leader), len(pose_follower_gpt), len(pose_leader))
        overlay_gpt = pos3d_to_video_gt_vs_recon(
            gt_follower[:T_gpt], gt_leader[:T_gpt],
            pose_follower_gpt[:T_gpt], pose_leader[:T_gpt],
            fps=30, width=_width, height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_gpt_gt_vs_{sample_name.replace('/', '_')}.mp4"),
        )

        # 2) Follower GPT w. RL (uses selected RL checkpoint)
        progress(0.65, desc="Loading Follower GPT w. RL...")
        agent_rl = _get_agent_rl()
        progress(0.7, desc="Loading motion & translation VQ-VAEs...")
        agent_rl.model.load_state_dict(ckpt_m["model"], strict=True)
        agent_rl.model.eval()
        agent_rl.model3.load_state_dict(ckpt_t["model"], strict=True)
        agent_rl.model3.eval()
        progress(0.75, desc="Loading RL checkpoint...")
        checkpoint_rl = torch.load(rl_path, map_location="cpu")
        agent_rl.model2.load_state_dict(checkpoint_rl["model"])
        agent_rl.model2.eval()
        progress(0.78, desc="Running RL inference...")
        pose_follower_rl, _ = _run_single_inference(agent_rl, batch)
        progress(0.88, desc="Rendering RL video...")
        out_rl = pos3d_to_video(
            pose_follower_rl,
            pose_leader,
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_rl_{sample_name.replace('/', '_')}.mp4"),
        )
        T_rl = min(len(gt_follower), len(gt_leader), len(pose_follower_rl), len(pose_leader))
        overlay_rl = pos3d_to_video_gt_vs_recon(
            gt_follower[:T_rl], gt_leader[:T_rl],
            pose_follower_rl[:T_rl], pose_leader[:T_rl],
            fps=30, width=_width, height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_rl_gt_vs_{sample_name.replace('/', '_')}.mp4"),
        )

        if _use_salsa_dataset:
            music_path = get_salsa_music_path(SALSA_MUSIC_MP3, sample_name)
        else:
            music_path = get_music_path_for_sample(MUSIC_SOURCE, sample_name)
            if not os.path.isfile(music_path):
                music_path = None
        progress(1.0, desc="Done")
        return out_gpt, out_rl, overlay_gpt, overlay_rl, music_path, "Follower GPT and Follower GPT w. RL ready. Overlay videos: GT (shadow) vs generated (solid)."
    except Exception as e:
        import traceback
        return None, None, None, None, None, f"Error: {e}\n{traceback.format_exc()}"


def build_ui():
    _load_config()  # so model info is available for VQ-VAE tab
    with gr.Blocks(title="Duolando – Data & Inference", theme=gr.themes.Soft(primary_hue="orange", secondary_hue="orange")) as app:
        gr.Markdown("# Duolando – Data visualization & inference")
        gr.Markdown("Choose **DD100** or **Salsa** (cache built by `Salsa_utils.salsa_duolando_cache`), load data, pick a sample, then use **Ground truth** or **Inference**.")

        with gr.Row():
            dataset_choice = gr.Radio(
                label="Dataset",
                choices=["DD100", "Salsa"],
                value="DD100",
                info="Salsa uses cache under data/salsa_duolando (or SALSA_DUOLANDO_CACHE).",
            )
            load_btn = gr.Button("Load and index data", variant="primary")
            load_status = gr.Textbox(label="Status", value="Select dataset and click 'Load and index data'.", interactive=False)

        sample_dropdown = gr.Dropdown(
            label="Sample",
            choices=[],
            value=None,
            allow_custom_value=False,
        )
        load_btn.click(fn=load_and_index_data, inputs=[dataset_choice], outputs=[sample_dropdown, load_status])

        with gr.Accordion("Model checkpoints (used for VQ-VAE reconstruction & Inference)", open=True):
            motion_ckpt_dropdown = gr.Dropdown(
                choices=[label for label, _ in _VQVAE_CKPT_CHOICES],
                value=_VQVAE_CKPT_CHOICES[0][0] if _VQVAE_CKPT_CHOICES else None,
                label="Motion VQ-VAE checkpoint",
                info="Pretrained or experiments/motion_vqvae*/ckpt/*.pt",
            )
            transl_ckpt_dropdown = gr.Dropdown(
                choices=[label for label, _ in _TRANSL_VQVAE_CKPT_CHOICES],
                value=_TRANSL_VQVAE_CKPT_CHOICES[0][0] if _TRANSL_VQVAE_CKPT_CHOICES else None,
                label="Translation VQ-VAE checkpoint",
                info="Pretrained or experiments/transl_vqvae*/ckpt/*.pt",
            )

        with gr.Tabs():
            with gr.Tab("Ground truth"):
                gt_vis_btn = gr.Button("Visualize", variant="secondary")
                gt_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    gt_video = gr.Video(label="Ground truth animation")
                    gt_audio = gr.Audio(label="Song", type="filepath")
                gt_vis_btn.click(
                    fn=visualize_ground_truth,
                    inputs=[sample_dropdown],
                    outputs=[gt_video, gt_audio, gt_status],
                )

            with gr.Tab("VQ-VAE Reconstruction"):
                gr.Markdown(
                    "Encode the selected sample through **pose VQ-VAE** and **translation VQ-VAE** (tokenize → detokenize), same as at inference. "
                    "Uses the **Motion** and **Translation VQ-VAE** checkpoints selected above. "
                    "Chart below: ground truth vs reconstructed relation (X, Y, Z over time)."
                )
                with gr.Accordion("VQ-VAE model details", open=True):
                    vqvae_model_info = gr.Markdown(value=_get_vqvae_model_info(), label="Model info")
                with gr.Accordion("From tokens to pose 3D", open=False):
                    vqvae_tokens_to_pose = gr.Markdown(value=TOKENS_TO_POSE_EXPLANATION, label="Tokens to pose")
                vqvae_btn = gr.Button("Reconstruct", variant="secondary")
                vqvae_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    vqvae_video = gr.Video(label="VQ-VAE reconstruction")
                    vqvae_overlay_video = gr.Video(label="GT vs Recon (overlay: shadow = GT, solid = recon)")
                vqvae_audio = gr.Audio(label="Song", type="filepath")
                vqvae_plot = gr.Plot(label="Relative translation: Ground truth vs Reconstructed")
                with gr.Accordion("Tokens used for this reconstruction", open=True):
                    vqvae_tokens = gr.Markdown(value="Run **Reconstruct** to see the discrete tokens (leader pose, follower pose, translation) extracted for this sample.", label="Tokens")
                vqvae_btn.click(
                    fn=visualize_vqvae_reconstruction,
                    inputs=[sample_dropdown, motion_ckpt_dropdown, transl_ckpt_dropdown],
                    outputs=[vqvae_video, vqvae_overlay_video, vqvae_audio, vqvae_plot, vqvae_tokens, vqvae_status],
                )

            with gr.Tab("Inference"):
                gr.Markdown("Runs **Follower GPT** and **Follower GPT w. RL** for the selected sample. Use the **Motion** and **Translation VQ-VAE** checkpoints from the accordion above, and the **Follower GPT** and **Follower GPT w. RL** checkpoints below. First row: generated only. Second row: GT (shadow) vs generated (solid).")
                gpt_ckpt_dropdown = gr.Dropdown(
                    choices=[label for label, _ in _GPT_CKPT_CHOICES],
                    value=_GPT_CKPT_CHOICES[0][0] if _GPT_CKPT_CHOICES else None,
                    label="Follower GPT checkpoint",
                    info="Pretrained or experiments/follower_gpt_salsa/ckpt/*.pt",
                )
                rl_ckpt_dropdown = gr.Dropdown(
                    choices=[label for label, _ in _RL_CKPT_CHOICES],
                    value=_RL_CKPT_CHOICES[0][0] if _RL_CKPT_CHOICES else None,
                    label="Follower GPT w. RL checkpoint",
                    info="Pretrained or experiments/rl_salsa/ckpt/*.pt",
                )
                inf_btn = gr.Button("Run inference (GPT + RL)", variant="secondary")
                inf_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    inf_video_gpt = gr.Video(label="Follower GPT")
                    inf_video_rl = gr.Video(label="Follower GPT w. RL")
                with gr.Row():
                    inf_overlay_gpt = gr.Video(label="GT vs GPT (overlay)")
                    inf_overlay_rl = gr.Video(label="GT vs RL (overlay)")
                inf_audio = gr.Audio(label="Song", type="filepath")
                inf_btn.click(
                    fn=run_inference,
                    inputs=[sample_dropdown, motion_ckpt_dropdown, transl_ckpt_dropdown, gpt_ckpt_dropdown, rl_ckpt_dropdown],
                    outputs=[inf_video_gpt, inf_video_rl, inf_overlay_gpt, inf_overlay_rl, inf_audio, inf_status],
                )

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Duolando Gradio app")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 7860 or GRADIO_SERVER_PORT)")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()
    port = args.port or int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=port, share=args.share)
