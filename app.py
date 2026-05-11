"""
Zero-Knowledge Voice -- FastAPI Web Server
===========================================
Professional dashboard with REST + WebSocket endpoints.

Endpoints:
  GET  /                  -> Dashboard UI
  POST /api/transcribe    -> File upload transcription + PII masking
  GET  /api/health        -> Model/system status
  POST /api/benchmark     -> Run WER/CER benchmark (async-friendly)
  WS   /ws/stream         -> Real-time microphone streaming

Author: Zero-Knowledge Voice Team
"""

import json
import logging
import os
import tempfile
import time
import threading

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import process_audio, process_audio_array, TranscriptionResult

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Zero-Knowledge Voice",
    description="Offline STT with PII Redaction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory store for session metrics and benchmark results
session_metrics = {
    "total_transcriptions": 0,
    "total_pii_detected": 0,
    "avg_latency_ms": 0,
    "latencies": [],
}
benchmark_result = None
benchmark_running = False


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "engine": "faster-whisper (base, INT8)",
        "pii_engine": "presidio (spaCy NER + regex)",
        "mode": "offline",
        "privacy": "zero-knowledge",
    }


@app.post("/api/transcribe")
async def transcribe_upload(file: UploadFile = File(...)):
    """Transcribe uploaded audio with PII redaction."""
    suffix = os.path.splitext(file.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Get audio duration
        try:
            info = sf.info(tmp_path)
            audio_duration = info.duration
        except Exception:
            audio_duration = 0

        result: TranscriptionResult = process_audio(tmp_path)

        # Compute real-time factor
        rtf = (result.latency_ms / 1000 / audio_duration) if audio_duration > 0 else 0

        # Update session metrics
        session_metrics["total_transcriptions"] += 1
        session_metrics["total_pii_detected"] += len(result.pii_entities)
        session_metrics["latencies"].append(result.latency_ms)
        if session_metrics["latencies"]:
            session_metrics["avg_latency_ms"] = round(
                sum(session_metrics["latencies"]) / len(session_metrics["latencies"]), 1
            )

        word_count = len(result.raw_text.split()) if result.raw_text else 0

        return {
            "raw_text": result.raw_text,
            "redacted_text": result.redacted_text,
            "pii_entities": [
                {"type": e.entity_type, "text": e.text, "start": e.start, "end": e.end, "score": e.score}
                for e in result.pii_entities
            ],
            "language": result.language,
            "latency_ms": result.latency_ms,
            "audio_duration_sec": round(audio_duration, 2),
            "rtf": round(rtf, 3),
            "word_count": word_count,
            "session": {
                "total_transcriptions": session_metrics["total_transcriptions"],
                "total_pii_detected": session_metrics["total_pii_detected"],
                "avg_latency_ms": session_metrics["avg_latency_ms"],
            },
        }
    except Exception as e:
        logger.error("Transcription error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)


@app.get("/api/metrics")
async def get_metrics():
    """Return current session metrics and benchmark results."""
    resp = {
        "session": session_metrics.copy(),
        "benchmark": None,
    }
    resp["session"].pop("latencies", None)
    if benchmark_result:
        resp["benchmark"] = benchmark_result
    return resp


@app.post("/api/benchmark")
async def run_benchmark_endpoint(max_files: int = 20):
    """Run WER/CER benchmark (limited to max_files for speed)."""
    global benchmark_result, benchmark_running

    if benchmark_running:
        return JSONResponse({"error": "Benchmark already running"}, status_code=409)

    benchmark_running = True
    try:
        from accuracy import run_benchmark as _run_bench
        from dataclasses import asdict

        result = _run_bench(model_size="base", max_files=max_files)

        # Store without per-utterance details for the API
        data = asdict(result)
        data.pop("utterances", None)
        benchmark_result = data

        return data
    except Exception as e:
        logger.error("Benchmark error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        benchmark_running = False


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """Real-time WebSocket for microphone streaming."""
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)
            if len(audio) < 1600:
                continue
            try:
                result = process_audio_array(audio, sr=16000)
                await ws.send_json({
                    "raw_text": result.raw_text,
                    "redacted_text": result.redacted_text,
                    "latency_ms": result.latency_ms,
                    "pii_entities": [
                        {"type": e.entity_type, "text": e.text, "score": e.score}
                        for e in result.pii_entities
                    ],
                })
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("\n" + "=" * 55)
    print("  Zero-Knowledge Voice")
    print("  100% Offline | Privacy by Design")
    print("  http://localhost:8000")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
