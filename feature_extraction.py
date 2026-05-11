"""
Zero-Knowledge Voice -- Feature Extraction Module
===================================================
Documents and visualizes the mel-spectrogram feature extraction
used internally by the Whisper architecture.

Whisper Parameters (Radford et al., 2022):
  Sample rate:     16,000 Hz
  FFT window:      400 samples (25 ms)
  Hop length:      160 samples (10 ms)
  Mel filter banks: 80
  Frequency range:  0 -- 8,000 Hz

Author: Zero-Knowledge Voice Team
"""

import os
import logging
from typing import Optional

import numpy as np
import librosa
import librosa.display

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FMIN = 0.0
FMAX = SAMPLE_RATE / 2
WINDOW_MS = N_FFT / SAMPLE_RATE * 1000
HOP_MS = HOP_LENGTH / SAMPLE_RATE * 1000


def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute log-mel spectrogram matching Whisper's parameters."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    logger.info("Mel spectrogram: %s (%d frames x %d mels)", log_mel.shape, log_mel.shape[1], log_mel.shape[0])
    return log_mel


def compute_spectrogram_stats(mel_spec: np.ndarray) -> dict:
    """Compute summary statistics for a mel spectrogram."""
    n_mels, n_frames = mel_spec.shape
    return {
        "shape": mel_spec.shape,
        "n_mels": n_mels,
        "n_frames": n_frames,
        "duration_sec": round(n_frames * HOP_LENGTH / SAMPLE_RATE, 2),
        "window_ms": WINDOW_MS,
        "hop_ms": HOP_MS,
        "mean_db": round(float(np.mean(mel_spec)), 2),
        "std_db": round(float(np.std(mel_spec)), 2),
        "min_db": round(float(np.min(mel_spec)), 2),
        "max_db": round(float(np.max(mel_spec)), 2),
    }


def visualize_features(
    audio: np.ndarray, sr: int = SAMPLE_RATE,
    save_path: Optional[str] = None, title: str = "Whisper Feature Extraction Pipeline",
) -> None:
    """Generate 3-panel visualization: waveform, STFT, mel-spectrogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping visualization")
        return

    mel_spec = extract_mel_spectrogram(audio, sr)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    time_axis = np.arange(len(audio)) / sr
    axes[0].plot(time_axis, audio, color="#4FC3F7", linewidth=0.4)
    axes[0].set_title("1. Raw Waveform (16kHz Mono)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, time_axis[-1])

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
    img1 = librosa.display.specshow(D, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", ax=axes[1], cmap="magma")
    axes[1].set_title("2. Short-Time Fourier Transform (STFT)")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    img2 = librosa.display.specshow(mel_spec, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=axes[2], cmap="magma", fmin=FMIN, fmax=FMAX)
    axes[2].set_title(f"3. Log-Mel Spectrogram ({N_MELS} Mel Bins -- Whisper Input)")
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from data_loader import load_dataset
    from preprocessor import preprocess_file

    samples = load_dataset()
    if not samples:
        raise SystemExit("No samples found.")

    first = samples[0]
    print(f"\nFeature Extraction: {first.audio_id}")
    audio, sr = preprocess_file(first.path, apply_vad=False)
    mel_spec = extract_mel_spectrogram(audio, sr)
    stats = compute_spectrogram_stats(mel_spec)

    print(f"\nWhisper Parameters:")
    print(f"  FFT window:  {N_FFT} samples ({WINDOW_MS}ms)")
    print(f"  Hop length:  {HOP_LENGTH} samples ({HOP_MS}ms)")
    print(f"  Mel bins:    {N_MELS}")
    print(f"  Freq range:  {FMIN}Hz -- {FMAX}Hz")
    print(f"\nSpectrogram Statistics:")
    for k, v in stats.items():
        print(f"  {k:15s}: {v}")

    visualize_features(audio, sr, save_path="output/feature_extraction_pipeline.png")
