"""
xai.py — Production-ready Grad-CAM for the AASIST Conv1D spoof-detection pipeline.

Key features:
  1. Dual-branch CAM  : fuses early spec + early temporal features
  2. Early hook        : taps after block-1 (stride x9) for higher temporal resolution
  3. Weighted sum CAM  : channel weights from ReLU(mean(gradient))
  4. F.interpolate     : sub-pixel accurate 1-D→2-D resize (no cv2 distortion)
  5. Percentile norm   : clips outlier activations for stable colour range
  6. Proper overlay    : spectrogram drawn first, cam on top with tuned alpha
  7. Research-quality  : 150 dpi, tight layout, title, proper colourbar
"""

import os
import math
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
MAX_LEN = 64600

# ──────────────────────────────────────────────────────────────────
#  AUDIO PREPROCESSING HELPERS
# ──────────────────────────────────────────────────────────────────

def _repeat_pad(x, target_len):
    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float32)
    reps = math.ceil(target_len / len(x))
    return np.tile(x, reps)[:target_len]


def _pad_or_truncate(audio, max_len=MAX_LEN):
    if len(audio) >= max_len:
        return audio[:max_len]
    return _repeat_pad(audio, max_len)


def load_audio(audio_path):
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    except Exception:
        # fallback for MP3/AAC which soundfile can't read
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    audio = _pad_or_truncate(audio)
    return torch.tensor(audio, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────
#  INTERNAL GRAD-CAM HELPERS
# ──────────────────────────────────────────────────────────────────

def _compute_cam_1d(activations: torch.Tensor,
                    gradients: torch.Tensor) -> torch.Tensor:
    """
    Compute a 1-D Grad-CAM map from [1, C, T] activations and gradients.

    Uses ReLU-weighted channel sum (standard Grad-CAM, Selvaraju et al. 2017).
    Returns a 1-D tensor of shape [T] in [0, 1].
    """
    # Global average pool gradients over time → per-channel importance weight
    weights = torch.mean(gradients, dim=2, keepdim=True)  # [1, C, 1]
    weights = torch.relu(weights)                          # keep only positive contributions

    # Weighted sum across channels
    cam = torch.sum(weights * activations, dim=1).squeeze(0)  # [T]
    cam = torch.relu(cam)

    return cam


def _normalise_cam(cam: torch.Tensor, low_pct: float = 2.0,
                   high_pct: float = 98.0) -> np.ndarray:
    """
    Percentile-based normalisation: clips extreme values before scaling to [0, 1].
    Much more stable than dividing by max (single outlier ruins everything).
    """
    cam_np = cam.numpy()
    lo = np.percentile(cam_np, low_pct)
    hi = np.percentile(cam_np, high_pct)
    cam_np = np.clip(cam_np, lo, hi)
    denom = (hi - lo) + 1e-8
    cam_np = (cam_np - lo) / denom
    return cam_np.astype(np.float32)


def _resize_cam_to_spectrogram(cam_1d: np.ndarray,
                                n_mels: int,
                                n_time_frames: int) -> np.ndarray:
    """
    Resize a 1-D CAM array to a 2-D [n_mels, n_time_frames] heatmap using
    torch.nn.functional.interpolate for sub-pixel accuracy (no cv2 needed).

    The temporal axis is interpolated; the frequency axis is tiled uniformly
    because Conv1D features carry no frequency structure — displaying equal
    saliency across all mel-bins is honest and avoids misleading artefacts.
    """
    # 1-D → 2-D by tiling across frequency, then bilinear interpolate
    cam_t = torch.from_numpy(cam_1d).float()
    cam_2d = cam_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # [1,1,1,T]
    cam_2d = cam_2d.expand(-1, -1, n_mels, -1)              # [1,1,H,T]

    cam_resized = F.interpolate(
        cam_2d,
        size=(n_mels, n_time_frames),
        mode='bilinear',
        align_corners=False
    )  # [1, 1, n_mels, n_time_frames]

    return cam_resized.squeeze().numpy()                     # [n_mels, n_time_frames]


# ──────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────

def generate_xai_heatmap(audio_path: str,
                          model,
                          pred_class: int,
                          output_image_path: str) -> str:
    """
    Generate a research-quality Grad-CAM heatmap overlaid on a Mel spectrogram
    and save it to *output_image_path*.

    Parameters
    ----------
    audio_path        : path to the audio file (wav / flac / mp3 …)
    model             : loaded AASIST instance
    pred_class        : predicted class index (0 = bonafide, 1 = spoof)
    output_image_path : absolute path where the PNG will be saved

    Returns
    -------
    output_image_path (str)
    """
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # ── 1. Load audio ─────────────────────────────────────────────
    audio = load_audio(audio_path)          # [T]
    audio = audio.unsqueeze(0)             # [1, T]

    # ── 2. Forward + backward with gradients ──────────────────────
    was_training = model.training
    model.eval()   # keep BN in inference mode (running stats), just enable grad flow
    model.zero_grad()

    # torch.enable_grad() ensures grad flows even if caller used no_grad()
    with torch.enable_grad():
        audio_g = audio.detach().requires_grad_(True)
        output  = model(audio_g)
        score   = output[0, pred_class]
        score.backward()

    # ── 3. Retrieve activations & gradients ───────────────────────
    early_acts  = model.get_early_activations()    # [1, C, T_early]
    early_grads = model.get_early_gradients()      # [1, C, T_early]

    temp_acts   = model.get_temp_activations()     # [1, C, T_temp]
    temp_grads  = model.get_temp_gradients()       # [1, C, T_temp]

    model.zero_grad()
    if was_training:
        model.train()

    # ── 4. Compute per-branch CAMs ────────────────────────────────
    cam_spec = None
    cam_temp = None

    if early_acts is not None and early_grads is not None:
        cam_spec_1d = _compute_cam_1d(
            early_acts.detach(), early_grads.detach()
        )
        cam_spec = _normalise_cam(cam_spec_1d)

    if temp_acts is not None and temp_grads is not None:
        cam_temp_1d = _compute_cam_1d(
            temp_acts.detach(), temp_grads.detach()
        )
        cam_temp = _normalise_cam(cam_temp_1d)

    # ── 5. Fuse branches ──────────────────────────────────────────
    # Upsample temporal branch to match spec branch length, then average.
    if cam_spec is not None and cam_temp is not None:
        t_spec = len(cam_spec)
        cam_temp_t = torch.from_numpy(cam_temp).float().unsqueeze(0).unsqueeze(0)
        cam_temp_t = F.interpolate(cam_temp_t, size=t_spec,
                                   mode='linear', align_corners=False)
        cam_temp_np = cam_temp_t.squeeze().numpy()
        cam_1d = (cam_spec + cam_temp_np) / 2.0
        # Re-normalise fused map
        hi = np.percentile(cam_1d, 98)
        lo = np.percentile(cam_1d, 2)
        cam_1d = np.clip((cam_1d - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    elif cam_spec is not None:
        cam_1d = cam_spec
    elif cam_temp is not None:
        cam_1d = cam_temp
    else:
        raise RuntimeError("Grad-CAM: no feature maps captured — "
                           "check that model hooks are registered.")

    # ── 6. Generate Mel spectrogram ───────────────────────────────
    N_MELS     = 128
    HOP_LENGTH = 512
    N_FFT      = 2048

    raw_audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    mel = librosa.feature.melspectrogram(
        y=raw_audio, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmax=sr // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # [n_mels, n_time_frames]

    n_time_frames = mel_db.shape[1]

    # ── 7. Resize CAM to spectrogram dimensions ───────────────────
    cam_2d = _resize_cam_to_spectrogram(cam_1d, N_MELS, n_time_frames)

    # ── 8. Plot ───────────────────────────────────────────────────
    label_str = "SPOOF" if pred_class == 1 else "BONAFIDE"
    cmap_name  = "inferno"            # perceptually uniform; red = high importance

    fig, ax = plt.subplots(figsize=(24, 14), facecolor="#ffffff")

    # 8a. Draw spectrogram
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        fmax=sr // 2,
        ax=ax,
        cmap='magma'
    )

    # 8b. Overlay Grad-CAM heatmap
    time_max = raw_audio.shape[0] / sr
    ax.imshow(
        cam_2d,
        extent=[0, time_max, 0, sr // 2],
        aspect='auto',
        origin='lower',
        cmap=cmap_name,
        alpha=0.50,
        vmin=0.0, vmax=1.0,
        interpolation='bilinear'
    )

    # 8c. Colourbar for CAM importance
    sm = plt.cm.ScalarMappable(cmap=cmap_name,
                                norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spoof Saliency", color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # 8d. Labels & styling
    ax.set_title(
        f"Grad-CAM Explainability  ·  Prediction: {label_str}",
        color='white', fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel("Time (s)",        color='white', fontsize=10)
    ax.set_ylabel("Frequency (Hz)",  color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')

    plt.tight_layout(pad=1.5)
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_image_path
