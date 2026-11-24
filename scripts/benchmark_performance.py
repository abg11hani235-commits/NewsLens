"""NewsLens - Comprehensive Performance & Optimization Benchmark Suite.

Executes reproducible benchmarks evaluating:
1. Model initialization and loading times.
2. Cold-start (first inference) vs. Warm-start inference latencies.
3. Average inference latency, standard deviation, and throughput (tokens/sec).
4. Peak RAM and GPU VRAM memory utilization.
5. Sequential vs. True Vectorized Batch inference speedup.
6. Evaluation across Short, Medium, and Long article documents.
7. Architectural comparison: Lead-3, TextRank, T5-Small, and BART-Large-CNN.

Saves structured results to outputs/metrics/benchmark_performance.json and .csv.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np

from config import (
    PRIMARY_MODEL_NAME,
    ALTERNATIVE_MODEL_NAME,
    METRICS_DIR,
    get_device,
)
from src.preprocessing import clean_text
from src.extractive import lead_n_summarize, textrank_summarize
from src.summarizer import AbstractiveSummarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# ==============================================================================
# Benchmark Document Corpora
# ==============================================================================

DOCUMENTS = {
    "short": (
        "The European Central Bank announced an interest rate reduction today in Frankfurt, "
        "citing declining headline inflation across the eurozone. The 25-basis-point cut brought "
        "the deposit facility rate to 3.25 percent. Financial analysts noted that lower borrowing "
        "costs could provide needed relief to manufacturing sectors in Germany and France."
    ),
    "medium": (
        "The Palestinian Authority officially became the 123rd member of the International Criminal Court "
        "on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. "
        "The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. "
        "The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction "
        "over alleged crimes committed in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014. "
        "Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis. "
        "The ICC opened a preliminary examination into the situation in Palestinian territories in January, "
        "shortly after the Palestinians signed the Rome Statute. Human rights organizations welcomed the accession, "
        "arguing that it brings the international community one step closer to ending impunity for war crimes. "
        "However, diplomatic envoys warned that the development could complicate ongoing bilateral negotiations in the region."
    ),
    "long": (
        "Astronomers analyzing infrared imagery from NASA's James Webb Space Telescope have confirmed the discovery "
        "of an ancient galaxy formed just 300 million years after the Big Bang. The galaxy, designated JADES-GS-z14-0, "
        "exhibits surprising brightness and mass, challenging prevailing theoretical models of early cosmic structure formation. "
        "Spectroscopic data revealed that the galaxy contains significant amounts of oxygen and ionized dust, indicating that "
        "multiple generations of massive stars had already lived and died in the earliest epoch of the universe. "
        "Researchers believe this demonstrates that early galaxy development occurred much faster than previously estimated. "
        "The international team of astrophysicists utilized the telescope's Near-Infrared Spectrograph (NIRSpec) "
        "to observe the galaxy for over sixty hours of integration time. Prior to these observations, cosmological models "
        "predicted that galaxies at such extreme redshifts would be diminutive, faint clusters of primordial hydrogen and helium. "
        "The detection of heavier elements proves that stellar nucleosynthesis was already widespread when the universe was only "
        "two percent of its current age. Theoretical cosmologists will need to revise simulations regarding star formation efficiency "
        "and black hole feedback mechanisms. The findings have been submitted for peer review to the Astrophysical Journal Letters. "
        "Future observational campaigns are planned to search for even earlier structures that may illuminate the dawn of the universe."
    ),
}


def get_vram_mb() -> float:
    """Return current allocated GPU VRAM in megabytes."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return 0.0


def get_max_vram_mb() -> float:
    """Return peak allocated GPU VRAM in megabytes."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return 0.0


def benchmark_extractive_methods() -> List[Dict[str, Any]]:
    """Benchmark Lead-3 and TextRank across document lengths."""
    results = []
    logger.info("Benchmarking Extractive Methods (Lead-3 & TextRank)...")

    for doc_len_name, text in DOCUMENTS.items():
        words = len(text.split())

        # --- Lead-3 ---
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            summary = lead_n_summarize(text, num_sentences=3)
            latencies.append(time.perf_counter() - t0)

        out_words = len(summary.split())
        warm_lats = latencies[1:]
        results.append({
            "model": "Lead-3",
            "paradigm": "Extractive Heuristic",
            "document_length_category": doc_len_name,
            "input_words": words,
            "output_words": out_words,
            "init_time_sec": 0.0,
            "cold_latency_sec": round(latencies[0], 4),
            "warm_latency_mean_sec": round(float(np.mean(warm_lats)), 4),
            "warm_latency_median_sec": round(float(np.median(warm_lats)), 4),
            "warm_latency_std_sec": round(float(np.std(warm_lats)), 4),
            "warm_latency_min_sec": round(float(np.min(warm_lats)), 4),
            "warm_latency_max_sec": round(float(np.max(warm_lats)), 4),
            "throughput_words_per_sec": round(out_words / max(float(np.mean(warm_lats)), 1e-6), 1),
            "peak_vram_mb": 0.0,
            "device": "cpu",
        })

        # --- TextRank ---
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            summary = textrank_summarize(text, num_sentences=3)
            latencies.append(time.perf_counter() - t0)

        out_words = len(summary.split())
        warm_lats = latencies[1:]
        results.append({
            "model": "TextRank",
            "paradigm": "Extractive Graph",
            "document_length_category": doc_len_name,
            "input_words": words,
            "output_words": out_words,
            "init_time_sec": 0.0,
            "cold_latency_sec": round(latencies[0], 4),
            "warm_latency_mean_sec": round(float(np.mean(warm_lats)), 4),
            "warm_latency_median_sec": round(float(np.median(warm_lats)), 4),
            "warm_latency_std_sec": round(float(np.std(warm_lats)), 4),
            "warm_latency_min_sec": round(float(np.min(warm_lats)), 4),
            "warm_latency_max_sec": round(float(np.max(warm_lats)), 4),
            "throughput_words_per_sec": round(out_words / max(float(np.mean(warm_lats)), 1e-6), 1),
            "peak_vram_mb": 0.0,
            "device": "cpu",
        })

    return results


def benchmark_abstractive_model(
    model_name: str,
    model_label: str,
    preset: str = "quality",
) -> List[Dict[str, Any]]:
    """Benchmark abstractive model across document lengths with rigorous memory and latency tracking."""
    results = []
    logger.info(f"Benchmarking Abstractive Model '{model_label}' (Preset={preset})...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    init_start = time.perf_counter()
    summarizer = AbstractiveSummarizer(
        model_name=model_name,
        device=get_device(),
    )
    init_time = round(time.perf_counter() - init_start, 3)

    # Measure warmup inference latency (first execution compiling GPU kernels and graph caches)
    warmup_inference_sec = summarizer.warmup()

    for doc_len_name, text in DOCUMENTS.items():
        input_words = len(text.split())

        # Reset peak VRAM statistics before every independent document measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # First inference for this specific document
        doc_first_start = time.perf_counter()
        first_res = summarizer.summarize_text(text, max_length=128, min_length=30)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        doc_first_latency = round(time.perf_counter() - doc_first_start, 4)

        # Warm runs (5 repetitions for high statistical fidelity)
        warm_latencies = []
        out_tokens_list = []
        gen_lats = []
        tok_lats = []
        dec_lats = []
        e2e_lats = []

        for _ in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            w_res = summarizer.summarize_text(text, max_length=128, min_length=30)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_rep = time.perf_counter() - t0
            warm_latencies.append(elapsed_rep)
            out_tokens_list.append(w_res["output_tokens"])
            gen_lats.append(w_res.get("generation_latency_sec", elapsed_rep))
            tok_lats.append(w_res.get("tokenization_latency_sec", 0.0))
            dec_lats.append(w_res.get("decoding_latency_sec", 0.0))
            e2e_lats.append(w_res.get("total_e2e_latency_sec", elapsed_rep))

        avg_warm = float(np.mean(warm_latencies))
        med_warm = float(np.median(warm_latencies))
        std_warm = float(np.std(warm_latencies))
        min_warm = float(np.min(warm_latencies))
        max_warm = float(np.max(warm_latencies))

        avg_tokens = float(np.mean(out_tokens_list))
        throughput_tok_sec = round(avg_tokens / max(avg_warm, 1e-6), 2)
        out_words = len(w_res["summary"].split())
        input_tokens = first_res.get("input_tokens", 0)
        # Peak VRAM isolated specifically for this document
        peak_vram = get_max_vram_mb()

        results.append({
            "model": f"{model_label} ({preset})",
            "paradigm": "Abstractive Transformer",
            "document_length_category": doc_len_name,
            "input_words": input_words,
            "input_tokens": input_tokens,
            "output_words": out_words,
            "output_tokens": int(round(avg_tokens)),
            "init_time_sec": init_time,
            "warmup_inference_sec": warmup_inference_sec,
            "doc_first_inference_sec": doc_first_latency,
            "warm_latency_mean_sec": round(avg_warm, 4),
            "warm_latency_median_sec": round(med_warm, 4),
            "warm_latency_std_sec": round(std_warm, 4),
            "warm_latency_min_sec": round(min_warm, 4),
            "warm_latency_max_sec": round(max_warm, 4),
            "generation_latency_sec": round(float(np.mean(gen_lats)), 4),
            "tokenization_latency_sec": round(float(np.mean(tok_lats)), 4),
            "decoding_latency_sec": round(float(np.mean(dec_lats)), 4),
            "total_e2e_latency_sec": round(float(np.mean(e2e_lats)), 4),
            "throughput_tokens_per_sec": throughput_tok_sec,
            "throughput_words_per_sec": round(out_words / max(avg_warm, 1e-6), 1),
            "peak_vram_mb": peak_vram,
            "device": str(summarizer.device),
        })

    # Unload model cleanly after benchmarking
    summarizer.unload_model()
    return results


def benchmark_batch_vs_sequential() -> Dict[str, Any]:
    """Benchmark true batch inference against sequential inference on T5-Small."""
    logger.info("Benchmarking Sequential vs. True Batch Inference on T5-Small...")

    batch_texts = [DOCUMENTS["medium"]] * 4  # 4 copies of medium article

    summarizer = AbstractiveSummarizer(
        model_name=ALTERNATIVE_MODEL_NAME,
        device=get_device(),
    )
    summarizer.warmup()

    # 1. Sequential Inference (batch_size=1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    seq_start = time.perf_counter()
    seq_results = []
    for txt in batch_texts:
        seq_results.append(summarizer.summarize_text(txt, max_length=64, min_length=15))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seq_total_time = round(time.perf_counter() - seq_start, 4)

    # 2. Vectorized Batch Inference (batch_size=4)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    batch_start = time.perf_counter()
    batch_results = summarizer.summarize_batch(batch_texts, batch_size=4, max_length=64, min_length=15)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_total_time = round(time.perf_counter() - batch_start, 4)

    speedup = round(seq_total_time / max(batch_total_time, 1e-6), 2)
    logger.info(f"Sequential Total: {seq_total_time}s | Batch Total: {batch_total_time}s | Speedup: {speedup}x")

    summarizer.unload_model()

    return {
        "num_documents": len(batch_texts),
        "sequential_total_sec": seq_total_time,
        "batch_total_sec": batch_total_time,
        "sequential_per_doc_sec": round(seq_total_time / len(batch_texts), 4),
        "batch_per_doc_sec": round(batch_total_time / len(batch_texts), 4),
        "speedup_factor": speedup,
    }


def run_all_benchmarks():
    """Execute complete benchmark suite and save structured metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # 1. Extractive Models
    all_results.extend(benchmark_extractive_methods())

    # 2. T5-Small (Quality & Fast modes)
    all_results.extend(benchmark_abstractive_model(ALTERNATIVE_MODEL_NAME, "T5-Small", preset="quality"))

    # 3. BART-Large-CNN (Quality mode)
    all_results.extend(benchmark_abstractive_model(PRIMARY_MODEL_NAME, "BART-Large-CNN", preset="quality"))

    # 4. Batch vs Sequential Benchmark
    batch_comparison = benchmark_batch_vs_sequential()

    # Save to CSV and JSON
    bench_df = pd.DataFrame(all_results)
    csv_file = METRICS_DIR / "benchmark_performance.csv"
    json_file = METRICS_DIR / "benchmark_performance.json"
    batch_file = METRICS_DIR / "batch_comparison.json"

    bench_df.to_csv(csv_file, index=False, encoding="utf-8")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_comparison, f, indent=2)

    logger.info(f"Saved benchmark results to {csv_file}")
    logger.info(f"Saved JSON metrics to {json_file}")
    logger.info(f"Saved Batch comparison to {batch_file}")

    print("\n" + "=" * 90)
    print("NEWSLENS PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 90)
    print(bench_df[["model", "document_length_category", "warm_latency_mean_sec", "throughput_words_per_sec", "peak_vram_mb", "device"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("SEQUENTIAL VS. TRUE VECTORIZED BATCH COMPARISON (T5-Small, 4 Documents)")
    print("=" * 90)
    print(f"Sequential Execution Total: {batch_comparison['sequential_total_sec']}s ({batch_comparison['sequential_per_doc_sec']}s/doc)")
    print(f"Batch Execution Total:      {batch_comparison['batch_total_sec']}s ({batch_comparison['batch_per_doc_sec']}s/doc)")
    print(f"Throughput Speedup Factor:  {batch_comparison['speedup_factor']}x faster")


if __name__ == "__main__":
    run_all_benchmarks()
