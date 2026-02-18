"""
Visualization helpers for Duolando duet dance data.
Uses the same skeleton topology and rendering logic as the base Duolando utils/visualize.py.
"""

import os
import tempfile
import numpy as np
import cv2


# SMPLX skeleton edges (same as utils/visualize.py) – connect joint indices for drawing bones
SMPLX_EDGES = [
    [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
    [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17],
    [16, 18], [17, 19], [18, 20], [19, 21],
    [20, 25], [25, 26], [26, 27], [20, 28], [28, 29], [29, 30],
    [20, 31], [20, 32], [20, 33], [20, 34], [20, 35], [20, 36],
    [20, 37], [20, 38], [20, 39],
    [21, 40], [40, 41], [41, 42], [21, 43], [21, 44], [21, 45],
    [21, 46], [46, 47], [47, 48], [21, 49], [49, 50], [50, 51],
    [21, 52], [52, 53], [53, 54],
]


def _project_and_scale(joints_3d, scale=100, center_x=480, center_y=270):
    """Map 3D joints to 2D screen coords (same convention as Duolando visualize2)."""
    # joints_3d: (55, 3) – use y and z for screen x and y
    x_screen = joints_3d[:, 1] * scale + center_x
    y_screen = -joints_3d[:, 2] * scale + center_y
    return np.stack([x_screen, y_screen], axis=1).astype(np.int32)


def render_duet_frame(
    joints_follower,
    joints_leader,
    width=960,
    height=540,
    scale=100,
    color_follower=(80, 80, 255),
    color_leader=(255, 80, 80),
):
    """
    Render one frame with follower and leader skeletons.
    joints_follower, joints_leader: (55, 3) each.
    Returns BGR image (H, W, 3) uint8.
    """
    center_x, center_y = width // 2, height // 2
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # dark background

    for joints, color in [(joints_follower, color_follower), (joints_leader, color_leader)]:
        if joints is None or len(joints) == 0:
            continue
        pts = _project_and_scale(joints, scale=scale, center_x=center_x, center_y=center_y)
        for (i, j) in SMPLX_EDGES:
            if 0 <= i < len(pts) and 0 <= j < len(pts):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[j])
                cv2.line(frame, pt1, pt2, color, thickness=2)
    return frame


def render_duet_frame_gt_vs_recon(
    gt_follower,
    gt_leader,
    recon_follower,
    recon_leader,
    width=960,
    height=540,
    scale=100,
    color_gt=(50, 50, 50),
    color_recon_follower=(80, 80, 255),
    color_recon_leader=(255, 80, 80),
    thickness_gt=4,
    thickness_recon=2,
):
    """
    Render one frame with ground truth as shadow (thick, dark) and reconstructed on top (normal colors).
    Same projection and edges as render_duet_frame; GT drawn first, then recon, for overlay comparison.
    All joints: (55, 3).
    """
    center_x, center_y = width // 2, height // 2
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    def draw_skeleton(joints, color, thickness):
        if joints is None or len(joints) == 0:
            return
        pts = _project_and_scale(joints, scale=scale, center_x=center_x, center_y=center_y)
        for (i, j) in SMPLX_EDGES:
            if 0 <= i < len(pts) and 0 <= j < len(pts):
                pt1, pt2 = tuple(pts[i]), tuple(pts[j])
                cv2.line(frame, pt1, pt2, color, thickness=thickness)

    # 1) Ground truth as shadow (drawn first)
    draw_skeleton(gt_follower, color_gt, thickness_gt)
    draw_skeleton(gt_leader, color_gt, thickness_gt)
    # 2) Reconstructed on top
    draw_skeleton(recon_follower, color_recon_follower, thickness_recon)
    draw_skeleton(recon_leader, color_recon_leader, thickness_recon)
    return frame


def pos3d_to_video_gt_vs_recon(
    gt_follower,
    gt_leader,
    recon_follower,
    recon_leader,
    fps=30,
    width=960,
    height=540,
    scale=100,
    output_path=None,
):
    """
    Render overlay video: ground truth (shadow) + reconstructed (solid).
    All inputs (T, 55, 3) or (T, 165); trimmed to min length.
    Returns path to the written video file.
    """
    if gt_follower.ndim == 2:
        gt_follower = gt_follower.reshape(-1, 55, 3)
    if gt_leader.ndim == 2:
        gt_leader = gt_leader.reshape(-1, 55, 3)
    if recon_follower.ndim == 2:
        recon_follower = recon_follower.reshape(-1, 55, 3)
    if recon_leader.ndim == 2:
        recon_leader = recon_leader.reshape(-1, 55, 3)
    T = min(len(gt_follower), len(gt_leader), len(recon_follower), len(recon_leader))
    if T == 0:
        raise ValueError("Empty motion sequence")
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_gt_vs_recon.mp4")
        os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for t in range(T):
        frame = render_duet_frame_gt_vs_recon(
            gt_follower[t],
            gt_leader[t],
            recon_follower[t],
            recon_leader[t],
            width=width,
            height=height,
            scale=scale,
        )
        out.write(frame)
    out.release()
    return output_path


def pos3d_to_video(
    pos3d_follower,
    pos3d_leader,
    fps=30,
    width=960,
    height=540,
    scale=100,
    output_path=None,
):
    """
    Render duet skeletons to a video file.
    pos3d_follower, pos3d_leader: (T, 55, 3) or (T, 165) that we reshape to (T, 55, 3).
    Returns path to the written video file.
    """
    if pos3d_follower.ndim == 2:
        pos3d_follower = pos3d_follower.reshape(-1, 55, 3)
    if pos3d_leader.ndim == 2:
        pos3d_leader = pos3d_leader.reshape(-1, 55, 3)
    T = min(len(pos3d_follower), len(pos3d_leader))
    if T == 0:
        raise ValueError("Empty motion sequence")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for t in range(T):
        frame = render_duet_frame(
            pos3d_follower[t],
            pos3d_leader[t],
            width=width,
            height=height,
            scale=scale,
        )
        out.write(frame)
    out.release()
    return output_path


def dataset_pos3d_to_display_pos3d(pos3d):
    """
    Convert pos3d from DD100 dataset format (preprocessed) to display format.
    Dataset stores: root in [:3], body relative to root, hands as (hand - wrist)*10.
    We undo hand scaling and add root to all joints so scale matches inference output.
    Same logic as gpt2t.py visgt() and actor_critic eval post-processing.
    pos3d: (T, 165) or (T, 55, 3). Returns (T, 55, 3) in display format.
    """
    if pos3d.ndim == 2:
        pos3d = pos3d.reshape(-1, 55, 3).copy()
    else:
        pos3d = np.asarray(pos3d, dtype=np.float64).copy()
    T = pos3d.shape[0]
    # Joint indices: 0=pelvis, 20=left wrist, 21=right wrist, 25-39=left hand, 40-54=right hand
    left_twist = pos3d[:, 20:21, :]   # (T, 1, 3)
    right_twist = pos3d[:, 21:22, :]
    # Undo hand scaling: stored = (hand - twist) * 10  =>  hand = stored*0.1 + twist
    pos3d[:, 25:40, :] = pos3d[:, 25:40, :] * 0.1 + np.tile(left_twist, (1, 15, 1))
    pos3d[:, 40:55, :] = pos3d[:, 40:55, :] * 0.1 + np.tile(right_twist, (1, 15, 1))
    # Add root to all joints (dataset has body relative to root, root in joint 0)
    root = pos3d[:, 0:1, :]  # (T, 1, 3)
    pos3d[:, 1:, :] += root
    pos3d[:, 0, :] = root.reshape(T, 3)
    return pos3d


def get_music_path_for_sample(music_source_dir, sample_name):
    """
    Resolve MP3 path for a dataset sample (fname).
    Follows the same logic as utils/visualize.py visualize2.
    """
    if "2023" not in sample_name:
        if "_" in sample_name:
            mp3 = sample_name.split("_")[0] + "_" + sample_name.split("_")[1] + "_" + sample_name.split("_")[2]
            music_name = os.path.join(music_source_dir, mp3 + ".mp3")
        else:
            music_name = os.path.join(music_source_dir, sample_name + ".mp3")
    else:
        folder = sample_name.split("_")[0]
        mp3 = sample_name.split("_")[1]
        music_name = os.path.join(music_source_dir, folder, mp3[:-2] + "0" + mp3[-2:] + ".mp3")
    return music_name


def get_salsa_music_path(music_mp3_dir, sample_name):
    """
    Resolve music path for Salsa cache (DD100-like layout: music/mp3/test/{take}.mp3 or .wav).
    Returns path to .mp3 if it exists, else .wav, else None.
    """
    for ext in (".mp3", ".wav"):
        path = os.path.join(music_mp3_dir, sample_name + ext)
        if os.path.isfile(path):
            return path
    return None
