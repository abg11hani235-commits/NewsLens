"""Evaluation and Benchmarking Module for Text Summarization NLP Project.

Implements standard ROUGE evaluation:
- ROUGE-1: Unigram overlap (content coverage and salience)
- ROUGE-2: Bigram overlap (fluency and phrasing coherence)
- ROUGE-L: Longest Common Subsequence (sentence structure preservation)
- Precision, Recall, and F1 metrics for all dimensions

Orchestrates full comparative benchmarking across:
1. Lead-3 (Classical Heuristic Baseline)
2. TextRank (Graph-based PageRank Extractive Baseline)
3. Google T5-Small (Abstractive Seq2Seq Transformer)
4. Facebook BART-Large-CNN (Abstractive SOTA Seq2Seq Transformer)
"""

import sys
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

from config import (
    METRICS_DIR,
    SUMMARIES_DIR,
    EVAL_SAMPLE_SIZE,
)
from src.preprocessing import clean_text, clean_highlights, load_processed_split
from src.extractive import lead_n_summarize, textrank_summarize

logger = logging.getLogger("evaluation")


def compute_rouge_metrics(
    predictions: List[str],
    references: List[str],
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate aggregate ROUGE-1, ROUGE-2, and ROUGE-L scores (Precision, Recall, F1).

    Handles empty predictions safely and averages across all evaluated pairs.

    Args:
        predictions (List[str]): List of generated summary strings.
        references (List[str]): List of ground truth human reference summaries.
        rouge_types (Optional[List[str]]): List of ROUGE metrics (defaults to rouge1, rouge2, rougeL).

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary containing 'precision', 'recall',
                                     and 'f1' for each ROUGE metric.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions length ({len(predictions)}) does not match references length ({len(references)})"
        )

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    accumulated = {rt: {"precision": [], "recall": [], "f1": []} for rt in rouge_types}

    for pred, ref in zip(predictions, references):
        pred_clean = clean_text(pred)
        ref_clean = clean_highlights(ref)

        if not pred_clean or not ref_clean:
            for rt in rouge_types:
                accumulated[rt]["precision"].append(0.0)
                accumulated[rt]["recall"].append(0.0)
                accumulated[rt]["f1"].append(0.0)
            continue

        scores = scorer.score(ref_clean, pred_clean)
        for rt in rouge_types:
            accumulated[rt]["precision"].append(scores[rt].precision)
            accumulated[rt]["recall"].append(scores[rt].recall)
            accumulated[rt]["f1"].append(scores[rt].fmeasure)

    results = {}
    for rt in rouge_types:
        results[rt] = {
            "precision": round(float(np.mean(accumulated[rt]["precision"])) * 100, 2),
            "recall": round(float(np.mean(accumulated[rt]["recall"])) * 100, 2),
            "f1": round(float(np.mean(accumulated[rt]["f1"])) * 100, 2),
        }

    return results


def run_full_benchmark(
    sample_size: int = EVAL_SAMPLE_SIZE,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end comparative evaluation across all 4 summarization architectures.

    Models evaluated:
    - Lead-3 Baseline
    - TextRank Extractive
    - T5-Small Abstractive
    - BART-Large-CNN Abstractive

    Args:
        sample_size (int): Number of test samples to benchmark (default 50).
        save_outputs (bool): Whether to persist tabular metrics and sample predictions.

    Returns:
        Dict[str, Any]: Benchmark evaluation dictionary containing scores and summary stats.
    """
    from src.summarizer import get_summarizer

    logger.info(f"Starting comparative benchmark on {sample_size} test samples...")
    test_df = load_processed_split("test").iloc[:sample_size].copy()

    articles = test_df["article"].tolist()
    references = test_df["highlights"].tolist()

    # 1. Lead-3 Baseline
    logger.info("Evaluating Lead-3 heuristic baseline...")
    import time

    t0 = time.perf_counter()
    lead3_preds = [lead_n_summarize(a, 3) for a in articles]
    lead3_latency = (time.perf_counter() - t0) / len(articles)

    # 2. TextRank Baseline
    logger.info("Evaluating TextRank graph-based extractive baseline...")
    t0 = time.perf_counter()
    textrank_preds = [textrank_summarize(a, 3) for a in articles]
    textrank_latency = (time.perf_counter() - t0) / len(articles)

    # 3. T5-Small Abstractive
    logger.info("Evaluating google-t5/t5-small...")
    t5_model = get_summarizer("t5")
    t0 = time.perf_counter()
    t5_results = t5_model.summarize_batch(articles, batch_size=1)
    t5_latency = (time.perf_counter() - t0) / len(articles)
    t5_preds = [r["summary"] for r in t5_results]

    # Clean GPU memory before loading BART
    del t5_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. BART-Large-CNN Abstractive
    logger.info("Evaluating facebook/bart-large-cnn...")
    bart_model = get_summarizer("bart")
    t0 = time.perf_counter()
    bart_results = bart_model.summarize_batch(articles, batch_size=1)
    bart_latency = (time.perf_counter() - t0) / len(articles)
    bart_preds = [r["summary"] for r in bart_results]

    del bart_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute ROUGE for each model
    model_predictions = {
        "Lead-3": (lead3_preds, lead3_latency, "Extractive Heuristic"),
        "TextRank": (textrank_preds, textrank_latency, "Extractive Graph"),
        "T5-Small": (t5_preds, t5_latency, "Abstractive Transformer (60M)"),
        "BART-Large-CNN": (bart_preds, bart_latency, "Abstractive Transformer (406M)"),
    }

    summary_rows = []
    detailed_metrics = {}

    for model_name, (preds, avg_lat, paradigm) in model_predictions.items():
        rouge_scores = compute_rouge_metrics(preds, references)
        avg_word_count = round(float(np.mean([len(p.split()) for p in preds])), 1)

        summary_rows.append({
            "Model": model_name,
            "Paradigm": paradigm,
            "ROUGE-1 F1": rouge_scores["rouge1"]["f1"],
            "ROUGE-1 Recall": rouge_scores["rouge1"]["recall"],
            "ROUGE-1 Precision": rouge_scores["rouge1"]["precision"],
            "ROUGE-2 F1": rouge_scores["rouge2"]["f1"],
            "ROUGE-2 Recall": rouge_scores["rouge2"]["recall"],
            "ROUGE-2 Precision": rouge_scores["rouge2"]["precision"],
            "ROUGE-L F1": rouge_scores["rougeL"]["f1"],
            "ROUGE-L Recall": rouge_scores["rougeL"]["recall"],
            "ROUGE-L Precision": rouge_scores["rougeL"]["precision"],
            "Avg Summary Words": avg_word_count,
            "Avg Latency (s)": round(avg_lat, 3),
        })

        detailed_metrics[model_name] = {
            "paradigm": paradigm,
            "rouge": rouge_scores,
            "avg_word_count": avg_word_count,
            "avg_latency_sec": round(avg_lat, 3),
        }

    metrics_df = pd.DataFrame(summary_rows)

    # Save outputs
    if save_outputs:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

        metrics_csv = METRICS_DIR / "rouge_scores.csv"
        metrics_json = METRICS_DIR / "rouge_scores.json"
        metrics_df.to_csv(metrics_csv, index=False)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(detailed_metrics, f, indent=2)

        # Save predictions side-by-side
        comp_df = pd.DataFrame({
            "id": test_df["id"].tolist(),
            "article": articles,
            "human_reference": references,
            "lead3_summary": lead3_preds,
            "textrank_summary": textrank_preds,
            "t5_summary": t5_preds,
            "bart_summary": bart_preds,
        })
        comp_csv = SUMMARIES_DIR / "model_comparison_samples.csv"
        comp_df.to_csv(comp_csv, index=False, encoding="utf-8")

        # Save qualitative json examples (first 5 samples)
        qualitative_samples = comp_df.iloc[:5].to_dict(orient="records")
        qual_json = SUMMARIES_DIR / "qualitative_comparison.json"
        with open(qual_json, "w", encoding="utf-8") as f:
            json.dump(qualitative_samples, f, indent=2)

        logger.info(f"Saved benchmark metrics to {metrics_csv} and predictions to {comp_csv}")

    return {
        "metrics_df": metrics_df,
        "detailed_metrics": detailed_metrics,
        "sample_size": sample_size,
    }


if __name__ == "__main__":
    print("=" * 70)
    print(f"Executing Complete Benchmark Evaluation (Sample Size: {EVAL_SAMPLE_SIZE})")
    print("=" * 70)
    res = run_full_benchmark(sample_size=EVAL_SAMPLE_SIZE)
    print("\n--- Benchmark ROUGE Summary Table ---")
    print(res["metrics_df"].to_string(index=False))
