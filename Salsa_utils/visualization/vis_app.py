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
from visualization_utils import pos3d_to_video, get_music_path_for_sample, dataset_pos3d_to_display_pos3d

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

_dataset = None
_sample_names = []
_width = 960
_height = 540


def load_and_index_data(progress=gr.Progress()):
    """Load DD100 test set and fill sample list. Returns dropdown choices and status."""
    global _dataset, _sample_names
    try:
        progress(0.1, desc="Loading dataset...")
        _dataset = DD100lfAll(
            MUSIC_ROOT,
            DATA_ROOT,
            split="test",
            interval=None,
            dtype="pos3d",
            move=4,
            music_dance_rate=1,
        )
        _sample_names = list(_dataset.names)
        progress(1.0, desc="Done")
        if not _sample_names:
            return gr.update(choices=[], value=None), "No samples found in test set. Check data/motion/pos3d/test and data/music/feature/test."
        return gr.update(choices=_sample_names, value=_sample_names[0]), f"Loaded {len(_sample_names)} samples. Pick a sample above and use the tabs below."
    except Exception as e:
        return gr.update(choices=[], value=None), f"Error loading data: {e}"


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
        music_path = get_music_path_for_sample(MUSIC_SOURCE, sample_name)
        if not os.path.isfile(music_path):
            music_path = None
        progress(1.0, desc="Done")
        return out_path, music_path, "Ground truth visualization ready. Play video and audio below."
    except Exception as e:
        import traceback
        return None, None, f"Error: {e}\n{traceback.format_exc()}"


def visualize_vqvae_reconstruction(sample_name, progress=gr.Progress()):
    """Encode selected sample through pose + translation VQ-VAEs (tokenize + detokenize) and visualize reconstruction."""
    if _dataset is None:
        return None, None, None, None, "Load and index data first (click 'Load and index data')."
    if not sample_name or sample_name not in _sample_names:
        return None, None, None, None, "Please select a sample from the dropdown."
    try:
        idx = _sample_names.index(sample_name)
        progress(0.1, desc="Loading model...")
        agent = _get_agent()
        progress(0.3, desc="Loading sample...")
        item = _dataset[idx]
        batch = {
            "pos3dl": torch.from_numpy(item["pos3dl"].astype(np.float32)).unsqueeze(0),
            "pos3df": torch.from_numpy(item["pos3df"].astype(np.float32)).unsqueeze(0),
        }
        progress(0.45, desc="Pose + translation VQ-VAE encode + decode...")
        recon_leader, recon_follower, gt_transl, recon_transl, tokens_text = _run_vqvae_reconstruction(agent, batch)
        progress(0.7, desc="Rendering video...")
        out_path = pos3d_to_video(
            recon_follower,
            recon_leader,
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_vqvae_recon_{sample_name.replace('/', '_')}.mp4"),
        )
        progress(0.85, desc="Building relation chart...")
        fig = _plot_translation_gt_vs_recon(gt_transl, recon_transl)
        music_path = get_music_path_for_sample(MUSIC_SOURCE, sample_name)
        if not os.path.isfile(music_path):
            music_path = None
        progress(1.0, desc="Done")
        msg = "Full VQ-VAE reconstruction (pose + translation) ready. Video uses decoded pose and decoded relative translation; chart compares GT vs reconstructed relation."
        return out_path, music_path, fig, tokens_text, msg
    except Exception as e:
        import traceback
        return None, None, None, None, f"Error: {e}\n{traceback.format_exc()}"


def run_inference(sample_name, progress=gr.Progress()):
    """Run both Follower GPT and Follower GPT w. RL on the selected sample; return both videos + audio."""
    if _dataset is None:
        return None, None, None, "Load and index data first."
    if not sample_name or sample_name not in _sample_names:
        return None, None, None, "Please select a sample from the dropdown."
    try:
        idx = _sample_names.index(sample_name)
        progress(0.05, desc="Loading sample...")
        item = _dataset[idx]
        batch = {
            "music": torch.from_numpy(item["music"].astype(np.float32)).unsqueeze(0),
            "pos3dl": torch.from_numpy(item["pos3dl"].astype(np.float32)).unsqueeze(0),
            "pos3df": torch.from_numpy(item["pos3df"].astype(np.float32)).unsqueeze(0),
            "rotmatl": torch.from_numpy(item["rotmatl"].astype(np.float32)).unsqueeze(0),
            "rotmatf": torch.from_numpy(item["rotmatf"].astype(np.float32)).unsqueeze(0),
            "fname": [sample_name],
        }

        # 1) Follower GPT
        progress(0.15, desc="Loading Follower GPT...")
        agent_gpt = _get_agent()
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

        # 2) Follower GPT w. RL
        progress(0.6, desc="Loading Follower GPT w. RL...")
        agent_rl = _get_agent_rl()
        progress(0.75, desc="Running RL inference...")
        pose_follower_rl, _ = _run_single_inference(agent_rl, batch)
        progress(0.9, desc="Rendering RL video...")
        out_rl = pos3d_to_video(
            pose_follower_rl,
            pose_leader,
            fps=30,
            width=_width,
            height=_height,
            output_path=os.path.join(tempfile.gettempdir(), f"duolando_rl_{sample_name.replace('/', '_')}.mp4"),
        )

        music_path = get_music_path_for_sample(MUSIC_SOURCE, sample_name)
        if not os.path.isfile(music_path):
            music_path = None
        progress(1.0, desc="Done")
        return out_gpt, out_rl, music_path, "Follower GPT and Follower GPT w. RL results ready. Play videos and song below."
    except Exception as e:
        import traceback
        return None, None, None, f"Error: {e}\n{traceback.format_exc()}"


def build_ui():
    _load_config()  # so model info is available for VQ-VAE tab
    with gr.Blocks(title="Duolando – Data & Inference", theme=gr.themes.Soft(primary_hue="orange", secondary_hue="orange")) as app:
        gr.Markdown("# Duolando – Data visualization & inference")
        gr.Markdown("Load DD100 data, pick a sample, then use **Ground truth** or **Inference** to visualize.")

        with gr.Row():
            load_btn = gr.Button("Load and index data", variant="primary")
            load_status = gr.Textbox(label="Status", value="Click 'Load and index data' to start.", interactive=False)

        sample_dropdown = gr.Dropdown(
            label="Sample",
            choices=[],
            value=None,
            allow_custom_value=False,
        )
        load_btn.click(fn=load_and_index_data, outputs=[sample_dropdown, load_status])

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
                    "Both body and relative translation (follower − leader root) are reconstructed from tokens. "
                    "Chart below: ground truth vs reconstructed relation (X, Y, Z over time)."
                )
                with gr.Accordion("VQ-VAE model details", open=True):
                    vqvae_model_info = gr.Markdown(value=_get_vqvae_model_info(), label="Model info")
                vqvae_btn = gr.Button("Reconstruct", variant="secondary")
                vqvae_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    vqvae_video = gr.Video(label="VQ-VAE reconstruction")
                    vqvae_audio = gr.Audio(label="Song", type="filepath")
                vqvae_plot = gr.Plot(label="Relative translation: Ground truth vs Reconstructed")
                with gr.Accordion("Tokens used for this reconstruction", open=True):
                    vqvae_tokens = gr.Markdown(value="Run **Reconstruct** to see the discrete tokens (leader pose, follower pose, translation) extracted for this sample.", label="Tokens")
                vqvae_btn.click(
                    fn=visualize_vqvae_reconstruction,
                    inputs=[sample_dropdown],
                    outputs=[vqvae_video, vqvae_audio, vqvae_plot, vqvae_tokens, vqvae_status],
                )

            with gr.Tab("Inference"):
                gr.Markdown("Runs **Follower GPT** and **Follower GPT w. RL** (README §1) for the selected sample. Both results are shown below.")
                inf_btn = gr.Button("Run inference (GPT + RL)", variant="secondary")
                inf_status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    inf_video_gpt = gr.Video(label="Follower GPT")
                    inf_video_rl = gr.Video(label="Follower GPT w. RL")
                inf_audio = gr.Audio(label="Song", type="filepath")
                inf_btn.click(
                    fn=run_inference,
                    inputs=[sample_dropdown],
                    outputs=[inf_video_gpt, inf_video_rl, inf_audio, inf_status],
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
