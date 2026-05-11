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
from presidio_analyzer import AnalyzerEngine, RecognizerResult, Pattern, PatternRecognizer
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
    "CVV", "EXP_DATE", "BROKEN_EMAIL", "GOVT_ID", "SPELLED_NUM",
    "ZIP_CODE", "ACCOUNT_NUM",
]

# --- Custom Recognizers ---

# 1. Flexible Credit Card (catches hyphenated/spaced numbers missed by default)
cc_pattern = Pattern(
    name="cc_flexible",
    regex=r"\b(?:\d[ -]*?){13,19}\b",
    score=0.85
)
cc_recognizer = PatternRecognizer(
    supported_entity="CREDIT_CARD",
    patterns=[cc_pattern],
    context=["card", "visa", "mastercard", "credit", "debit", "ending"]
)

# 1. Flexible Credit Card (catches hyphenated/spaced numbers missed by default)
cc_pattern = Pattern(name="cc_flexible", regex=r"\b(?:\d[ -]*?){13,19}\b", score=0.85)
cc_recognizer = PatternRecognizer(
    supported_entity="CREDIT_CARD", patterns=[cc_pattern],
    context=["card", "visa", "mastercard", "credit", "debit", "ending"]
)

# 2. CVV - Strong (prefixed)
cvv_strong = Pattern(name="cvv_strong", regex=r"(?i)cvv[ :\-]*(\d{3,4})\b", score=0.8)
# 2b. CVV - Weak (requires context)
cvv_weak = Pattern(name="cvv_weak", regex=r"\b\d{3,4}\b", score=0.1)
cvv_recognizer = PatternRecognizer(
    supported_entity="CVV", patterns=[cvv_strong, cvv_weak],
    context=["cvv", "cvc", "security code", "cvv2", "code"]
)

# 3. Expiration Date - Strong (prefixed)
exp_strong = Pattern(name="exp_strong", regex=r"(?i)exp[ :\-]*(\d{2}[/-]\d{2,4})\b", score=0.8)
# 3b. Expiration Date - Weak (requires context)
exp_weak = Pattern(name="exp_weak", regex=r"\b(0[1-9]|1[0-2])[/-](\d{2}|\d{4})\b", score=0.1)
exp_recognizer = PatternRecognizer(
    supported_entity="EXP_DATE", patterns=[exp_strong, exp_weak],
    context=["expiration", "expires", "exp", "expiry", "date"]
)

# 4. Broken Email (catches 'name at domain dot com' OR 'name domain.com')
email_broken = Pattern(
    name="email_broken",
    regex=r"\b[a-zA-Z0-9._%+-]+(?:\s*@\s*|\s+at\s+|\s+)[a-zA-Z0-9.-]+\.[a-z]{2,}\b",
    score=0.5
)
email_broken_recognizer = PatternRecognizer(
    supported_entity="BROKEN_EMAIL", patterns=[email_broken],
    context=["email", "contact", "send", "confirmation", "mail", "address"]
)

# 5. Government IDs (Passport / Driver's License patterns)
govt_id_pattern = Pattern(
    name="govt_id",
    regex=r"\b[A-Z0-9]{6,12}\b",
    score=0.1  # Low base, needs context
)
govt_id_recognizer = PatternRecognizer(
    supported_entity="GOVT_ID", patterns=[govt_id_pattern],
    context=["passport", "id number", "license", "dl", "identification", "document"]
)

# 6. Spelled-out Numbers (Common in audio: "four five six")
spelled_num_pattern = Pattern(
    name="spelled_num",
    regex=r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine)\s*(?:zero|one|two|three|four|five|six|seven|eight|nine|dash|hyphen|space|point|dot)+\b",
    score=0.7
)
spelled_num_recognizer = PatternRecognizer(
    supported_entity="SPELLED_NUM", patterns=[spelled_num_pattern],
    context=["number", "code", "digits", "spelled", "spell"]
)

# 7. Zip Code (5 digits, prevents misclassification as DATE_TIME)
zip_pattern = Pattern(name="zip", regex=r"\b\d{5}(?:-\d{4})?\b", score=0.6)
zip_recognizer = PatternRecognizer(
    supported_entity="ZIP_CODE", patterns=[zip_pattern],
    context=["zip", "postal", "zipcode", "address", "area"]
)

# 8. Account / Member ID (Alpha-numeric strings near keywords)
acc_pattern = Pattern(name="account_id", regex=r"\b[A-Z]{1,3}\d{4,10}[A-Z0-9]?\b", score=0.4)
acc_recognizer = PatternRecognizer(
    supported_entity="ACCOUNT_NUM", patterns=[acc_pattern],
    context=["account", "id", "member", "customer", "reference", "ref"]
)

# 9. Flexible Phone (Handles irregular transcriptions like extra dashes or digits)
phone_flex = Pattern(name="phone_flex", regex=r"\b\d{3}[-.\s]*\d{3,5}[-.\s]*\d{3,4}\b", score=0.4)
phone_flex_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER", patterns=[phone_flex],
    context=["phone", "call", "reach", "contact", "cell", "mobile"]
)

logger.info("Loading models...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
analyzer = AnalyzerEngine()
# Register custom recognizers
analyzer.registry.add_recognizer(cc_recognizer)
analyzer.registry.add_recognizer(cvv_recognizer)
analyzer.registry.add_recognizer(exp_recognizer)
analyzer.registry.add_recognizer(email_broken_recognizer)
analyzer.registry.add_recognizer(govt_id_recognizer)
analyzer.registry.add_recognizer(spelled_num_recognizer)
analyzer.registry.add_recognizer(zip_recognizer)
analyzer.registry.add_recognizer(acc_recognizer)
analyzer.registry.add_recognizer(phone_flex_recognizer)

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
    results = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en", score_threshold=0.35)
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
    results = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en", score_threshold=0.35)
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