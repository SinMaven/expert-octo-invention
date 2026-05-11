"""
Zero-Knowledge Voice -- WER/CER Benchmarking Module
=====================================================
Computes Word Error Rate and Character Error Rate against
LibriSpeech test-clean, comparing to Whisper paper baselines.

Reference: Radford et al., 2022, Table 2 (LibriSpeech test-clean)
  tiny: 7.6% | base: 5.0% | small: 3.4% | medium: 2.9% | large: 2.7%

Author: Zero-Knowledge Voice Team
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from jiwer import wer as compute_wer, cer as compute_cer
from tqdm import tqdm

from data_loader import load_dataset
from model import transcribe

logger = logging.getLogger(__name__)

PAPER_BASELINES_WER = {
    "tiny": 0.076, "base": 0.050, "small": 0.034,
    "medium": 0.029, "large": 0.027,
}


@dataclass
class UtteranceResult:
    audio_id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float


@dataclass
class BenchmarkResult:
    model_size: str
    num_utterances: int
    corpus_wer: float
    corpus_cer: float
    mean_wer: float
    std_wer: float
    median_wer: float
    min_wer: float
    max_wer: float
    mean_cer: float
    std_cer: float
    paper_baseline_wer: Optional[float]
    delta_vs_paper: Optional[float]
    total_time_sec: float
    avg_latency_ms: float
    utterances: List[UtteranceResult] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize for WER: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_benchmark(
    data_path: str = "data/LibriSpeech/test-clean",
    model_size: str = "base",
    max_files: Optional[int] = None,
) -> BenchmarkResult:
    """Run full WER/CER benchmark on the dataset."""
    samples = load_dataset(data_path, lazy=True)
    if max_files:
        samples = samples[:max_files]
    samples = [s for s in samples if s.transcript]

    if not samples:
        raise ValueError(f"No samples with transcripts found in {data_path}")

    references, hypotheses = [], []
    utterance_results = []
    per_wer, per_cer, latencies = [], [], []

    print(f"\n{'='*60}")
    print(f"  WER/CER Benchmark -- {len(samples)} utterances")
    print(f"  Model: faster-whisper ({model_size}, INT8)")
    print(f"{'='*60}\n")

    start_time = time.time()

    for sample in tqdm(samples, desc="Transcribing", unit="utt"):
        ref_norm = normalize_text(sample.transcript)
        t0 = time.perf_counter()
        hyp_raw, _ = transcribe(sample.path)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        hyp_norm = normalize_text(hyp_raw)
        if not ref_norm:
            continue

        utt_wer = compute_wer(ref_norm, hyp_norm)
        utt_cer = compute_cer(ref_norm, hyp_norm)

        references.append(ref_norm)
        hypotheses.append(hyp_norm)
        per_wer.append(utt_wer)
        per_cer.append(utt_cer)

        utterance_results.append(UtteranceResult(
            audio_id=sample.audio_id, reference=ref_norm,
            hypothesis=hyp_norm, wer=round(utt_wer, 4), cer=round(utt_cer, 4),
        ))

    total_time = time.time() - start_time

    import numpy as np
    wer_arr = np.array(per_wer)
    cer_arr = np.array(per_cer)

    corpus_wer = compute_wer(references, hypotheses)
    corpus_cer = compute_cer(references, hypotheses)
    paper_bl = PAPER_BASELINES_WER.get(model_size)

    return BenchmarkResult(
        model_size=model_size,
        num_utterances=len(utterance_results),
        corpus_wer=round(corpus_wer, 4),
        corpus_cer=round(corpus_cer, 4),
        mean_wer=round(float(np.mean(wer_arr)), 4),
        std_wer=round(float(np.std(wer_arr)), 4),
        median_wer=round(float(np.median(wer_arr)), 4),
        min_wer=round(float(np.min(wer_arr)), 4),
        max_wer=round(float(np.max(wer_arr)), 4),
        mean_cer=round(float(np.mean(cer_arr)), 4),
        std_cer=round(float(np.std(cer_arr)), 4),
        paper_baseline_wer=paper_bl,
        delta_vs_paper=round(corpus_wer - paper_bl, 4) if paper_bl else None,
        total_time_sec=round(total_time, 2),
        avg_latency_ms=round(float(np.mean(latencies)), 1),
        utterances=utterance_results,
    )


def print_report(result: BenchmarkResult) -> None:
    print(f"\n{'='*60}")
    print(f"  BENCHMARK REPORT")
    print(f"{'='*60}")
    print(f"  Model:          faster-whisper ({result.model_size}, INT8)")
    print(f"  Utterances:     {result.num_utterances}")
    print(f"  Total time:     {result.total_time_sec:.1f}s")
    print(f"  Avg latency:    {result.avg_latency_ms:.0f}ms/utterance")
    print(f"{'_'*60}")
    print(f"  Corpus WER:     {result.corpus_wer * 100:.2f}%")
    print(f"  Mean WER:       {result.mean_wer * 100:.2f}% (std: {result.std_wer * 100:.2f}%)")
    print(f"  Median WER:     {result.median_wer * 100:.2f}%")
    print(f"  Range WER:      {result.min_wer * 100:.2f}% -- {result.max_wer * 100:.2f}%")
    print(f"{'_'*60}")
    print(f"  Corpus CER:     {result.corpus_cer * 100:.2f}%")
    print(f"  Mean CER:       {result.mean_cer * 100:.2f}% (std: {result.std_cer * 100:.2f}%)")
    print(f"{'_'*60}")

    if result.paper_baseline_wer is not None:
        delta = result.delta_vs_paper * 100
        sign = "+" if delta > 0 else ""
        print(f"  Paper baseline: {result.paper_baseline_wer * 100:.2f}% (Radford et al., 2022)")
        print(f"  Delta vs paper: {sign}{delta:.2f}%")
        if delta <= 1.0:
            print(f"  Verdict:        PASS -- within acceptable range")
        elif delta <= 3.0:
            print(f"  Verdict:        ACCEPTABLE -- INT8 quantization degradation")
        else:
            print(f"  Verdict:        INVESTIGATE -- significant degradation")
    print(f"{'='*60}")

    worst = sorted(result.utterances, key=lambda u: u.wer, reverse=True)[:5]
    if worst and worst[0].wer > 0:
        print(f"\n  Worst 5 Utterances:")
        for u in worst:
            print(f"  {u.audio_id}: WER={u.wer*100:.1f}% CER={u.cer*100:.1f}%")
            print(f"    REF: {u.reference[:70]}")
            print(f"    HYP: {u.hypothesis[:70]}\n")


def export_results(result: BenchmarkResult, path: str = "output/wer_benchmark.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"Results exported: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_benchmark(model_size="base", max_files=None)
    print_report(result)
    export_results(result)