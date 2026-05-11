"""
Zero-Knowledge Voice -- Audio Preprocessing Module
====================================================
Pipeline: mono conversion -> resample 16kHz -> normalize -> VAD trim.

NOTE: Faster-Whisper handles mel-spectrogram computation internally.
This preprocessor prepares raw audio so the model receives clean,
speech-only signal. See feature_extraction.py for mel-spectrogram
documentation and visualization.

Author: Zero-Knowledge Voice Team
"""

import logging
from typing import Tuple

import numpy as np
import librosa

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    logger.info("Resampled %dHz -> %dHz", orig_sr, target_sr)
    return resampled.astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1).astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1.0, 1.0]."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        logger.warning("Audio is silent (all zeros)")
        return audio
    return (audio / peak).astype(np.float32)


def vad_trim(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
) -> np.ndarray:
    """
    Trim non-speech regions using WebRTC Voice Activity Detection.
    Retains frames classified as speech; removes leading/trailing silence.
    """
    try:
        import webrtcvad
    except ImportError:
        logger.warning("webrtcvad not installed -- skipping VAD trimming")
        return audio

    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * frame_duration_ms / 1000)
    audio_int16 = (audio * 32767).astype(np.int16)

    speech_frames = []
    total_frames = 0

    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i : i + frame_size]
        total_frames += 1
        try:
            if vad.is_speech(frame.tobytes(), sr):
                speech_frames.append(audio[i : i + frame_size])
        except Exception:
            speech_frames.append(audio[i : i + frame_size])

    if not speech_frames:
        logger.warning("VAD detected no speech -- returning original")
        return audio

    speech_ratio = len(speech_frames) / max(total_frames, 1)
    if speech_ratio < 0.3:
        logger.warning("Low speech ratio (%.0f%%) -- returning original", speech_ratio * 100)
        return audio

    trimmed = np.concatenate(speech_frames).astype(np.float32)
    logger.info("VAD: %.2fs -> %.2fs (%.0f%% speech)", len(audio)/sr, len(trimmed)/sr, speech_ratio*100)
    return trimmed


def preprocess(audio: np.ndarray, sr: int, apply_vad: bool = True) -> np.ndarray:
    """Full pipeline: mono -> resample -> normalize -> VAD trim."""
    audio = to_mono(audio)
    audio = resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    audio = normalize(audio)
    if apply_vad:
        audio = vad_trim(audio, sr=TARGET_SAMPLE_RATE)
    return audio


def preprocess_file(audio_path: str, apply_vad: bool = True) -> Tuple[np.ndarray, int]:
    """Load and preprocess an audio file. Returns (audio, 16000)."""
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    if audio.ndim == 2:
        audio = audio.T
    processed = preprocess(audio, sr, apply_vad=apply_vad)
    return processed, TARGET_SAMPLE_RATE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from data_loader import load_dataset

    samples = load_dataset()
    if not samples:
        raise SystemExit("No samples found.")

    first = samples[0]
    print(f"\nProcessing: {first.audio_id}")
    audio, sr = preprocess_file(first.path, apply_vad=True)
    print(f"  Shape:    {audio.shape}")
    print(f"  Rate:     {sr}Hz")
    print(f"  Duration: {len(audio)/sr:.2f}s")
    print(f"  Range:    [{audio.min():.4f}, {audio.max():.4f}]")
    print("Done.")