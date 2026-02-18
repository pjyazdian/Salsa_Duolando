"""
Build Duolando-format cache from Salsa pair LMDB.

Reads the source pair LMDB (vid → clips with keypoints3d_L/F, rotmat_L/F, optional audio),
converts each clip (or sliding windows) to Duolando layout:
  - pos3d: (T, 165) raw 55-joint positions (same format as DD100 on disk; DD100 applies
    root-relative and hand*10 at load time).
  - rotmat: (T, 498) = trans(3) + 55*9 (DD100 loads and uses [:, 3:]).
  - music: (T_music, 54) from Duolando's signal_to_feature when audio is present, else zeros.

Output directory layout (compatible with DD100lfAll):
  out_dir/motion/pos3d/{split}/{take}_00.npy, {take}_01.npy
  out_dir/motion/rotmat/{split}/{take}_00.npy, {take}_01.npy
  out_dir/music/feature/{split}/{take}.npy

Run from Salsa_Duolando root:
  python -m Salsa_utils.salsa_duolando_cache --source_lmdb /path/to/lmdb_train --out_dir ./data/salsa_duolando [options]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import pyarrow
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Duolando repo root (parent of Salsa_utils)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DUOLANDO_ROOT = _SCRIPT_DIR.parent
if str(_DUOLANDO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DUOLANDO_ROOT))

# Defaults: follow Duolando baseline
# - Test: interval=None → use full clip per take (no slicing in cache).
# - Train: Duolando uses interval=240, move=4 at load time; we export full clips, dataset slices.
DEFAULT_WINDOW_FRAMES = None  # None = full clip (recommended to match DD100lfAll)
DEFAULT_STRIDE_FRAMES = 4     # match Duolando move=4 when using window_frames
DEFAULT_POSE_FPS = 20
MUSIC_DANCE_RATE = 4  # Duolando: 1 music frame per 4 motion frames
MUSIC_FEATURE_DIM = 54  # mfcc(20)+mfcc_delta(20)+chroma_cqt(12)+onset_env(1)+onset_beat(1)
DUOLANDO_AUDIO_SR = 15360  # from _prepare_music_data.py


def keypoints22_to_pos3d_55(kp: np.ndarray, apply_rot_x_deg: float = 90) -> np.ndarray:
    """
    Expand (T, 22, 3) or (T, 66) keypoints to (T, 165) for Duolando.
    Salsa LMDB has 22 joints (HumanML3D); Duolando expects 55 (SMPLX). We put 22 in first 22 slots,
    fill joints 22-54 with left/right wrist copies so hands/face don't sit at origin.
    """
    T = kp.shape[0]
    if kp.ndim == 2:
        kp = np.asarray(kp, dtype=np.float32).reshape(T, -1, 3)
    if kp.shape[1] != 22:
        raise ValueError(f"keypoints22_to_pos3d_55 expects 22 joints, got {kp.shape}")
    pos3d = np.zeros((T, 55, 3), dtype=np.float32)
    pos3d[:, :22, :] = kp
    left_wrist = kp[:, 20:21, :]   # (T, 1, 3)
    right_wrist = kp[:, 21:22, :]
    pos3d[:, 22:40, :] = np.tile(left_wrist, (1, 18, 1))   # jaw, eyes, left hand
    pos3d[:, 40:55, :] = np.tile(right_wrist, (1, 15, 1))  # right hand
    pos3d = pos3d.reshape(T, 165)
    if apply_rot_x_deg != 0:
        theta_rad = np.deg2rad(apply_rot_x_deg)
        cx, sx = np.cos(theta_rad), np.sin(theta_rad)
        R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        pos3d = pos3d.reshape(T, 55, 3)
        pos3d = np.einsum("ij,tpj->tpi", R_x, pos3d)
        pos3d = pos3d.reshape(T, 165)
    return pos3d


def rotmat_to_pos3d_55(rotmat: np.ndarray, smplx_path: str | None, apply_rot_x_deg: float = 90) -> np.ndarray:
    """
    Convert rotmat (T, 498) = root translation (3) + 55 rotation matrices (495) to 55-joint positions (T, 165).
    Salsa LMDB stores rotmat as [trans_x, trans_y, trans_z, R0_flat, ..., R54_flat]; we use trans for SMPLX
    transl and rotmats for poses. DD100 loads rotmat files and uses [:, 3:] (rotation part only) for the model.
    """
    T = rotmat.shape[0]
    trans = np.asarray(rotmat[:, :3], dtype=np.float64)   # root translation (x,y,z)
    rotmats = np.asarray(rotmat[:, 3:], dtype=np.float64).reshape(T, 55, 3, 3)  # 55 joint rotation matrices

    poses = np.zeros((T, 55, 3), dtype=np.float32)
    for t in range(T):
        for j in range(55):
            poses[t, j] = R.from_matrix(rotmats[t, j]).as_rotvec()
    poses_flat = poses.reshape(T, -1)

    if not smplx_path or not os.path.isdir(smplx_path):
        raise RuntimeError("SMPLX model path is required. Use --no_smplx to use 22-joint keypoints from LMDB instead.")

    try:
        import torch
        from smplx import SMPLX
        betas = np.zeros((T, 10), dtype=np.float32)
        smplx = SMPLX(
            model_path=smplx_path,
            betas=torch.from_numpy(betas).float(),
            gender="neutral",
            batch_size=T,
            num_betas=10,
            use_pca=False,
            use_face_contour=True,
            flat_hand_mean=True,
        )
        with torch.no_grad():
            joints = smplx.forward(
                global_orient=torch.from_numpy(poses_flat[:, :3]).float(),
                body_pose=torch.from_numpy(poses_flat[:, 3:66]).float(),
                jaw_pose=torch.from_numpy(poses_flat[:, 66:69]).float(),
                leye_pose=torch.from_numpy(poses_flat[:, 69:72]).float(),
                reye_pose=torch.from_numpy(poses_flat[:, 72:75]).float(),
                left_hand_pose=torch.from_numpy(poses_flat[:, 75:120]).float(),
                right_hand_pose=torch.from_numpy(poses_flat[:, 120:165]).float(),
                transl=torch.from_numpy(trans).float(),
                betas=torch.from_numpy(betas).float(),
            ).joints.numpy()
        pos3d = joints[:, :55, :].reshape(T, 165).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"SMPLX forward failed (model_path={smplx_path}): {e}") from e

    if apply_rot_x_deg != 0:
        theta_rad = np.deg2rad(apply_rot_x_deg)
        cx, sx = np.cos(theta_rad), np.sin(theta_rad)
        R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        pos3d = pos3d.reshape(T, 55, 3)
        pos3d = np.einsum("ij,tpj->tpi", R_x, pos3d)
        pos3d = pos3d.reshape(T, 165)
    return pos3d


def extract_music_feature(raw_audio: np.ndarray, sr: int, motion_frames: int, pose_fps: int = DEFAULT_POSE_FPS) -> np.ndarray:
    """
    Extract music feature using Duolando's exact pipeline (_prepare_music_data.signal_to_feature).
    Returns (T_music, 54): mfcc(20) + mfcc_delta(20) + chroma_cqt(12) + onset_env(1) + onset_beat(1).
    Same features and dimensions as DD100; no slicing or padding of the feature dimension.
    """
    T_music = motion_frames // MUSIC_DANCE_RATE
    if T_music <= 0:
        return np.zeros((1, MUSIC_FEATURE_DIM), dtype=np.float32)
    # Resample to Duolando's 15360 Hz (same as _prepare_music_data main)
    if sr != DUOLANDO_AUDIO_SR:
        import librosa
        raw_audio = librosa.resample(raw_audio.astype(np.float64), orig_sr=sr, target_sr=DUOLANDO_AUDIO_SR)
        sr = DUOLANDO_AUDIO_SR
    try:
        from _prepare_music_data import signal_to_feature
        feature = signal_to_feature(raw_audio.ravel(), sr)  # (T_audio, 54) after transpose inside
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        if feature.shape[1] != MUSIC_FEATURE_DIM:
            raise ValueError(
                f"Duolando signal_to_feature must return 54 features (mfcc+mfcc_delta+chroma_cqt+onset_env+onset_beat), "
                f"got {feature.shape[1]}. Use the same _prepare_music_data.py and extractor.py as Duolando."
            )
        # Trim/pad time axis only to T_music
        if feature.shape[0] >= T_music:
            feature = feature[:T_music]
        else:
            pad = np.zeros((T_music - feature.shape[0], feature.shape[1]), dtype=np.float32)
            feature = np.concatenate([feature, pad], axis=0)
        return feature
    except Exception as e:
        raise RuntimeError(
            f"Music feature extraction failed (use Duolando's _prepare_music_data + extractor for same 54-d features): {e}"
        ) from e


def sanitize_take_name(vid: str, clip_idx: int, window_idx: int | None) -> str:
    """Produce a safe take name for filenames (no slashes, short)."""
    base = vid.replace(",", "_").replace("/", "_").replace(" ", "_")[:80]
    name = f"salsa_{base}_c{clip_idx}"
    if window_idx is not None:
        name += f"_w{window_idx}"
    return name


def _write_audio_file(base_path: Path, audio: np.ndarray, sr: int) -> None:
    """Write audio to WAV and optionally MP3 (DD100 layout: music/mp3/{split}/{take}.mp3 or .wav)."""
    audio = np.asarray(audio).ravel()
    if audio.dtype in (np.float32, np.float64):
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wav_path = base_path.with_suffix(".wav")
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(str(wav_path), sr, audio)
    except Exception:
        return
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_wav(str(wav_path))
        seg.export(str(base_path.with_suffix(".mp3")), format="mp3")
    except Exception:
        pass


def run(
    source_lmdb_dir: str,
    out_dir: str,
    smplx_path: str | None = None,
    split: str = "train",
    window_frames: int | None = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    pose_fps: int = DEFAULT_POSE_FPS,
    apply_rot_x_deg: float = 90,
    extract_audio_feature: bool = True,
    use_keypoints22: bool = False,
    write_audio_mp3: bool = True,
) -> None:
    """
    Build Duolando-format cache from source pair LMDB.

    - apply_rot_x_deg: +90 (default) for correct upright view, same as InterGen/salsa_to_interhuman.
    - write_audio_mp3: write music/mp3/{split}/{take}.wav (and .mp3 if pydub available) for playback in app.
    """
    out_path = Path(out_dir)
    pos3d_dir = out_path / "motion" / "pos3d" / split
    rotmat_dir = out_path / "motion" / "rotmat" / split
    feature_dir = out_path / "music" / "feature" / split
    mp3_dir = out_path / "music" / "mp3" / split
    pos3d_dir.mkdir(parents=True, exist_ok=True)
    rotmat_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir.mkdir(parents=True, exist_ok=True)

    smplx_path = smplx_path or os.environ.get("SMPLX_MODEL_PATH") or os.environ.get("smplx_path")
    if not use_keypoints22:
        if not smplx_path or not os.path.isdir(smplx_path):
            cand = _DUOLANDO_ROOT / "utils" / "smplx"
            if cand.is_dir():
                smplx_path = str(cand)
            else:
                raise FileNotFoundError(
                    "SMPLX model directory not found. Set --smplx_path or use --no_smplx to use 22-joint keypoints from the LMDB instead."
                )

    env = lmdb.open(source_lmdb_dir, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        keys = list(txn.cursor().iternext(values=False))
    n_keys = len(keys)
    if n_keys == 0:
        env.close()
        raise RuntimeError(f"LMDB is empty: {source_lmdb_dir}")
    print(f"LMDB has {n_keys} video keys, building cache with use_keypoints22={use_keypoints22}")

    n_out = 0
    first_error_logged = [True]

    for key in tqdm(keys, desc="Videos"):
        with env.begin(write=False) as txn:
            value = txn.get(key)
        if value is None:
            continue
        video = pyarrow.deserialize(value)
        vid = video.get("vid", "unknown")
        clips = video.get("clips", [])
        for clip_idx, clip in enumerate(clips):
            kp_L = clip.get("keypoints3d_L")
            rot_L = clip.get("rotmat_L")
            kp_F = clip.get("keypoints3d_F")
            rot_F = clip.get("rotmat_F")
            if rot_L is None or rot_F is None:
                continue
            rot_L = np.asarray(rot_L, dtype=np.float32)
            rot_F = np.asarray(rot_F, dtype=np.float32)
            # Support (T, 498) or (T, 495); if 495, prepend zeros for trans
            if rot_L.shape[1] == 495 and rot_F.shape[1] == 495:
                trans_z = np.zeros((rot_L.shape[0], 3), dtype=np.float32)
                rot_L = np.concatenate([trans_z, rot_L], axis=1)
                rot_F = np.concatenate([trans_z, rot_F], axis=1)
            elif rot_L.shape[1] != 498 or rot_F.shape[1] != 498:
                if first_error_logged[0]:
                    first_error_logged[0] = False
                    tqdm.write(f"Skip {vid} clip {clip_idx}: rotmat shape {rot_L.shape} / {rot_F.shape}, need (T, 498) or (T, 495)")
                continue
            T = rot_L.shape[0]
            if T < 10:
                continue
            if use_keypoints22:
                if kp_L is None or kp_F is None:
                    if first_error_logged[0]:
                        first_error_logged[0] = False
                        tqdm.write(f"Skip {vid} c{clip_idx}: --no_smplx requires keypoints3d_L/F in LMDB")
                    continue
                kp_L = np.asarray(kp_L, dtype=np.float32)
                kp_F = np.asarray(kp_F, dtype=np.float32)
                if kp_L.ndim == 2:
                    kp_L = kp_L.reshape(T, -1, 3)
                if kp_F.ndim == 2:
                    kp_F = kp_F.reshape(T, -1, 3)
                if kp_L.shape[1] != 22 or kp_F.shape[1] != 22:
                    if first_error_logged[0]:
                        first_error_logged[0] = False
                        tqdm.write(f"Skip {vid} c{clip_idx}: keypoints shape {kp_L.shape} / {kp_F.shape}, need 22 joints")
                    continue
            audio_raw = clip.get("audio_raw")
            audio_sr = int(clip.get("audio_sr", 16000))
            if audio_raw is not None:
                audio_raw = np.asarray(audio_raw).ravel().astype(np.float64)
            else:
                audio_raw = None

            if window_frames is not None and window_frames > 0:
                starts = list(range(0, max(0, T - window_frames) + 1, stride_frames))
                if not starts:
                    starts = [0]
                    end = min(window_frames, T)
                else:
                    end = window_frames
            else:
                starts = [0]
                end = T

            for wi, start in enumerate(starts):
                end_idx = start + end if window_frames else T
                if end_idx > T:
                    end_idx = T
                if end_idx - start < 10:
                    continue
                rot_L_w = rot_L[start:end_idx]
                rot_F_w = rot_F[start:end_idx]
                T_w = rot_L_w.shape[0]

                try:
                    if use_keypoints22:
                        kp_L_w = kp_L[start:end_idx]
                        kp_F_w = kp_F[start:end_idx]
                        pos3d_L = keypoints22_to_pos3d_55(kp_L_w, apply_rot_x_deg=apply_rot_x_deg)
                        pos3d_F = keypoints22_to_pos3d_55(kp_F_w, apply_rot_x_deg=apply_rot_x_deg)
                    else:
                        pos3d_L = rotmat_to_pos3d_55(rot_L_w, smplx_path, apply_rot_x_deg=apply_rot_x_deg)
                        pos3d_F = rotmat_to_pos3d_55(rot_F_w, smplx_path, apply_rot_x_deg=apply_rot_x_deg)
                except Exception as e:
                    if first_error_logged[0]:
                        first_error_logged[0] = False
                        tqdm.write(f"Skip {vid} c{clip_idx} w{wi}: {e}")
                    continue

                take_name = sanitize_take_name(vid, clip_idx, wi if (window_frames and len(starts) > 1) else None)
                # DD100: _00 = follower, _01 = leader
                np.save(pos3d_dir / f"{take_name}_00.npy", pos3d_F)
                np.save(pos3d_dir / f"{take_name}_01.npy", pos3d_L)
                np.save(rotmat_dir / f"{take_name}_00.npy", rot_F_w)
                np.save(rotmat_dir / f"{take_name}_01.npy", rot_L_w)

                if extract_audio_feature and audio_raw is not None and len(audio_raw) > 0:
                    duration_sec = T_w / pose_fps
                    n_samples = int(duration_sec * audio_sr)
                    audio_start = int(start / T * len(audio_raw))
                    audio_end = min(audio_start + n_samples, len(audio_raw))
                    if audio_end <= audio_start:
                        audio_slice = audio_raw[:1]
                    else:
                        audio_slice = audio_raw[audio_start:audio_end]
                    music_feat = extract_music_feature(audio_slice, audio_sr, T_w, pose_fps)
                else:
                    T_music = T_w // MUSIC_DANCE_RATE
                    if T_music < 1:
                        T_music = 1
                    music_feat = np.zeros((T_music, MUSIC_FEATURE_DIM), dtype=np.float32)
                # DD100 uses music_dance_rate=1: one music row per motion frame. Repeat so shape is (T_w, 54).
                if music_feat.shape[0] != T_w:
                    repeat = (T_w + music_feat.shape[0] - 1) // music_feat.shape[0]
                    music_feat = np.tile(music_feat, (repeat, 1))[:T_w]
                np.save(feature_dir / f"{take_name}.npy", music_feat.astype(np.float32))
                if write_audio_mp3 and audio_raw is not None and len(audio_raw) > 0:
                    duration_sec = T_w / pose_fps
                    n_samples = int(duration_sec * audio_sr)
                    audio_start = int(start / T * len(audio_raw))
                    audio_end = min(audio_start + n_samples, len(audio_raw))
                    if audio_end > audio_start:
                        audio_slice = np.asarray(audio_raw[audio_start:audio_end], dtype=np.float64)
                        _write_audio_file(mp3_dir / take_name, audio_slice, audio_sr)
                n_out += 1

    env.close()
    if n_out == 0:
        raise RuntimeError(
            f"No samples written. LMDB had {n_keys} keys. "
            "Try --no_smplx to use 22-joint keypoints from the LMDB (no SMPLX model needed), or check that rotmat_L/F have shape (T, 498) or (T, 495)."
        )
    print(f"Wrote {n_out} samples to {out_dir} (split={split})")


def main():
    ap = argparse.ArgumentParser(description="Build Duolando-format cache from Salsa pair LMDB")
    ap.add_argument("--source_lmdb", type=str, required=True, help="Source pair LMDB directory (e.g. lmdb_train)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output root (motion/ and music/ created under it)")
    ap.add_argument("--smplx_path", type=str, default=None, help="SMPLX model dir (or set SMPLX_MODEL_PATH)")
    ap.add_argument("--split", type=str, default="train", choices=("train", "test"))
    ap.add_argument("--window_frames", type=int, default=None, help="Window length; default full clip")
    ap.add_argument("--stride_frames", type=int, default=DEFAULT_STRIDE_FRAMES)
    ap.add_argument("--pose_fps", type=int, default=DEFAULT_POSE_FPS)
    ap.add_argument("--no_audio_feature", action="store_true", help="Do not extract music feature (write zeros)")
    ap.add_argument("--no_rot_x", action="store_true", help="Do not apply +90° X rotation (upright view)")
    ap.add_argument("--no_smplx", action="store_true", help="Use 22-joint keypoints from LMDB (no SMPLX); pos3d is approximated for Duolando")
    ap.add_argument("--no_mp3", action="store_true", help="Do not write music/mp3 (WAV/MP3) files")
    args = ap.parse_args()
    run(
        source_lmdb_dir=args.source_lmdb,
        out_dir=args.out_dir,
        smplx_path=args.smplx_path,
        split=args.split,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        pose_fps=args.pose_fps,
        apply_rot_x_deg=0 if args.no_rot_x else 90,
        extract_audio_feature=not args.no_audio_feature,
        use_keypoints22=args.no_smplx,
        write_audio_mp3=not args.no_mp3,
    )


if __name__ == "__main__":
    main()
