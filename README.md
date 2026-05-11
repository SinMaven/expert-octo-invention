# Zero-Knowledge Voice

**Offline, Real-Time Speech-to-Text with Automatic PII Masking**
Privacy by Design -- No Data Leaves Your Device

---

## What is Zero-Knowledge Voice?

Zero-Knowledge Voice is an offline speech-to-text (STT) pipeline that automatically detects and masks Personally Identifiable Information (PII) in transcribed text. The term "Zero-Knowledge" is borrowed from cryptography: the system produces useful output (redacted transcripts) while maintaining zero knowledge of the underlying sensitive data.

Unlike cloud-based APIs (Google Cloud Speech, AWS Transcribe, Azure Speech), all processing happens locally on your machine -- no internet connection required, no audio transmitted, no data stored externally.

### Privacy-by-Design Principles

| Principle | Implementation |
|---|---|
| Data Minimization | Raw audio discarded after transcription |
| On-Device Processing | No cloud dependencies, no API keys, no network I/O |
| Deterministic PII Masking | Presidio uses rule-based + NER -- no probabilistic leakage |
| Audit Trail | Every PII entity logged with type, position, and confidence |
| GDPR/HIPAA Compliant | Inherently compliant -- data never leaves the device |

### Comparison vs. Cloud APIs

| Aspect | Cloud APIs | Zero-Knowledge Voice |
|---|---|---|
| Data residency | Vendor servers | Local device only |
| Network required | Yes | No |
| PII exposure risk | High | Zero |
| Latency | Network-dependent | Deterministic |
| Cost | Per-minute billing | Free (open-source) |
| Offline capable | No | Yes |

---

## Architecture

```
Audio Input -> Preprocessor -> Faster-Whisper STT -> Presidio PII Masking -> Dashboard UI
  (Mic/File)   (Resample,       (Mel-Spectrogram,    (spaCy NER +          (FastAPI +
                Normalize,        Encoder-Decoder,      Regex Patterns)       WebSocket)
                VAD Trim)         Beam Search)

                         ALL PROCESSING 100% LOCAL
```

---

## Dataset: LibriSpeech test-clean

The project uses the **LibriSpeech test-clean** subset -- the exact benchmark used in the original Whisper paper (Radford et al., 2022):

| Property | Value |
|---|---|
| Utterances | 2,620 |
| Speakers | 40 |
| Total Duration | ~5.4 hours |
| Format | FLAC, 16kHz mono |
| Source | OpenSLR (openslr.org/12) |
| Paper Reference | Table 2, Radford et al., 2022 |

This enables direct, apples-to-apples WER comparison against the published paper baselines.

---

## Project Structure

```
zero_knowledge/
  app.py                  FastAPI server (REST + WebSocket + benchmark)
  model.py                STT engine + PII detection/masking
  preprocessor.py         Audio preprocessing (resample, VAD, normalize)
  feature_extraction.py   Mel-spectrogram visualization (for report)
  data_loader.py          LibriSpeech dataset loader + streaming
  accuracy.py             WER/CER benchmarking vs. Whisper paper
  pii_test.py             PII redaction test suite
  requirements.txt        Python dependencies
  static/
    index.html            Dashboard UI
    style.css             Design system
    app.js                Frontend logic
  data/
    LibriSpeech/
      test-clean/         2,620 utterances, 40 speakers
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Launch the dashboard
python app.py
# Open http://localhost:8000

# Run WER/CER benchmark
python accuracy.py

# Generate feature extraction visualizations
python feature_extraction.py

# Run PII tests
python pii_test.py
```

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| STT Engine | Faster-Whisper (CTranslate2) | Speech-to-text, INT8 quantization |
| PII Detection | Microsoft Presidio | NER + regex-based PII detection |
| NER Model | spaCy en_core_web_lg | English named entity recognition |
| Audio I/O | librosa, soundfile | Loading, resampling, mel-spectrogram |
| VAD | WebRTC VAD | Voice Activity Detection |
| Web Server | FastAPI + Uvicorn | REST API + WebSocket streaming |
| Benchmarking | jiwer | WER and CER computation |

---

## WER/CER Benchmarking Methodology

Benchmarks against baselines from Radford et al. (2022), Table 2:

| Model | WER (Paper) |
|---|---|
| tiny | 7.6% |
| base | 5.0% |
| small | 3.4% |
| medium | 2.9% |
| large | 2.7% |

Methodology:
1. Load all 2,620 LibriSpeech test-clean utterances
2. Transcribe each with Faster-Whisper (base, INT8)
3. Normalize: lowercase, strip punctuation, collapse whitespace
4. Compute per-utterance WER and CER
5. Report corpus-level and statistical aggregates (mean, std, median, min, max)
6. Compare against paper baseline with delta

---

## Whisper Feature Extraction

| Parameter | Value |
|---|---|
| Sample Rate | 16,000 Hz |
| FFT Window | 400 samples (25ms) |
| Hop Length | 160 samples (10ms) |
| Mel Bins | 80 |
| Frequency Range | 0 -- 8,000 Hz |

---

## Dashboard Metrics

The web dashboard displays the following metrics in real-time:

| Metric | Description |
|---|---|
| Latency | Processing time per transcription (ms) |
| RTF | Real-Time Factor (processing_time / audio_duration) |
| Words | Word count of transcription |
| PII Count | Number of PII entities detected |
| Audio Duration | Input audio length (seconds) |
| Language | Detected language |
| WER | Word Error Rate (benchmark mode) |
| CER | Character Error Rate (benchmark mode) |

## Authors

Zero-Knowledge Voice Team
