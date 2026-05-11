"""
Zero-Knowledge Voice -- Data Loading Module
============================================
High-performance audio ingestion for LibriSpeech-format datasets.

Supports:
  - Batch loading from LibriSpeech-structured directories
  - Streaming iterator for memory-efficient processing
  - Single file loading (WAV, FLAC, MP3)
  - Multi-speaker directory scanning

Author: Zero-Knowledge Voice Team
"""

import os
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Represents a single audio sample with metadata."""
    path: str
    audio_id: str
    audio: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    transcript: Optional[str] = None
    duration_sec: Optional[float] = None


def load_dataset(
    data_path: str = "data/LibriSpeech/test-clean",
    lazy: bool = False,
) -> List[AudioSample]:
    """
    Load a LibriSpeech-format dataset from disk.

    Scans all speaker/chapter subdirectories recursively.

    Args:
        data_path: Root directory containing speaker subdirectories.
        lazy:      If True, skip loading audio into memory (paths only).

    Returns:
        List of AudioSample objects sorted by utterance ID.
    """
    # Fallback to legacy path if the main path does not exist
    if not os.path.isdir(data_path):
        legacy = "data/1089"
        if os.path.isdir(legacy):
            logger.warning("Main dataset not found at %s, falling back to %s", data_path, legacy)
            data_path = legacy
        else:
            raise FileNotFoundError(f"Dataset path not found: {data_path}")

    transcripts: dict[str, str] = {}
    audio_files: List[AudioSample] = []

    # Pass 1: collect all transcripts
    for root, _, files in os.walk(data_path):
        for fname in files:
            if fname.endswith(".trans.txt"):
                trans_path = os.path.join(root, fname)
                try:
                    with open(trans_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                transcripts[parts[0]] = parts[1]
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning("Skipping transcript file %s: %s", trans_path, e)

    # Pass 2: match audio files to transcripts
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if fname.endswith((".flac", ".wav", ".mp3")):
                file_id = os.path.splitext(fname)[0]
                file_path = os.path.join(root, fname)

                sample = AudioSample(
                    path=file_path,
                    audio_id=file_id,
                    transcript=transcripts.get(file_id),
                )

                if not lazy:
                    try:
                        audio, sr = sf.read(file_path, dtype="float32")
                        sample.audio = audio
                        sample.sample_rate = sr
                        sample.duration_sec = len(audio) / sr
                    except Exception as e:
                        logger.warning("Failed to read %s: %s", file_path, e)
                        continue

                audio_files.append(sample)

    audio_files.sort(key=lambda s: s.audio_id)
    logger.info("Loaded %d audio samples from %s", len(audio_files), data_path)
    return audio_files


def stream_dataset(data_path: str = "data/LibriSpeech/test-clean") -> Iterator[AudioSample]:
    """
    Memory-efficient streaming iterator over a LibriSpeech dataset.
    Loads one sample at a time.
    """
    samples = load_dataset(data_path, lazy=True)
    for sample in samples:
        try:
            audio, sr = sf.read(sample.path, dtype="float32")
            sample.audio = audio
            sample.sample_rate = sr
            sample.duration_sec = len(audio) / sr
            yield sample
        except Exception as e:
            logger.warning("Skipping %s: %s", sample.path, e)


def load_single(audio_path: str) -> AudioSample:
    """Load a single audio file from any supported format."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to decode {audio_path}: {e}")

    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    return AudioSample(
        path=audio_path,
        audio_id=file_id,
        audio=audio,
        sample_rate=sr,
        duration_sec=len(audio) / sr,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    samples = load_dataset()
    print(f"\n{'='*55}")
    print(f"  Loaded {len(samples)} audio samples")
    print(f"{'='*55}")

    for s in samples[:3]:
        print(f"  [{s.audio_id}]")
        print(f"  Text: {s.transcript}")
        dur = f"{s.duration_sec:.2f}s" if s.duration_sec else "N/A"
        print(f"  Duration: {dur} | Rate: {s.sample_rate}Hz\n")