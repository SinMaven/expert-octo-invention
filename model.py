"""
Zero-Knowledge Voice -- STT + PII Masking Engine
==================================================
Core pipeline: Faster-Whisper transcription + Presidio PII masking.
All processing is 100% offline.

Author: Zero-Knowledge Voice Team
"""

import logging
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    entity_type: str
    start: int
    end: int
    score: float
    text: str


@dataclass
class TranscriptionResult:
    raw_text: str
    redacted_text: str
    pii_entities: List[PIIEntity] = field(default_factory=list)
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    latency_ms: Optional[float] = None


PII_ENTITIES = [
    "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
    "US_SSN", "LOCATION", "DATE_TIME", "IP_ADDRESS", "IBAN_CODE",
]

logger.info("Loading models...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
logger.info("Models loaded successfully")


def transcribe(audio_path: str) -> Tuple[str, Optional[str]]:
    """Transcribe audio file. Returns (text, language)."""
    segments, info = whisper_model.transcribe(audio_path)
    text = " ".join(seg.text for seg in segments).strip()
    language = info.language if info else None
    return text, language


def transcribe_array(audio: np.ndarray, sr: int = 16000) -> Tuple[str, Optional[str]]:
    """Transcribe numpy audio array via temp file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, sr)
        return transcribe(tmp.name)


def detect_pii(text: str) -> List[PIIEntity]:
    """Detect PII entities using Presidio (spaCy NER + regex)."""
    if not text.strip():
        return []
    results = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en")
    return [
        PIIEntity(
            entity_type=r.entity_type, start=r.start, end=r.end,
            score=round(r.score, 3), text=text[r.start:r.end],
        )
        for r in results
    ]


def redact_pii(text: str) -> str:
    """Anonymize all PII in text. Returns masked text."""
    if not text.strip():
        return text
    results = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def process_audio(audio_path: str) -> TranscriptionResult:
    """Full pipeline: Audio -> Transcription -> PII Detection -> Redaction."""
    t0 = time.perf_counter()
    raw_text, language = transcribe(audio_path)
    pii_entities = detect_pii(raw_text)
    redacted_text = redact_pii(raw_text)
    latency = (time.perf_counter() - t0) * 1000

    return TranscriptionResult(
        raw_text=raw_text, redacted_text=redacted_text,
        pii_entities=pii_entities, language=language, latency_ms=round(latency, 1),
    )


def process_audio_array(audio: np.ndarray, sr: int = 16000) -> TranscriptionResult:
    """Full pipeline from numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, sr)
        return process_audio(tmp.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from data_loader import load_dataset

    samples = load_dataset()
    if not samples:
        raise SystemExit("No samples found.")

    first = samples[0]
    print(f"\nProcessing: {first.audio_id}")
    print(f"Expected:   {first.transcript}\n")

    result = process_audio(first.path)
    print(f"Original:   {result.raw_text}")
    print(f"Redacted:   {result.redacted_text}")
    print(f"Language:   {result.language}")
    print(f"Latency:    {result.latency_ms}ms")
    print(f"PII found:  {len(result.pii_entities)}")
    for e in result.pii_entities:
        print(f"  -> {e.entity_type}: \"{e.text}\" (score: {e.score})")