"""Main Application CLI for Text Summarization NLP Project.

Provides a unified command-line entry point to orchestrate:
1. EDA (Exploratory Data Analysis)
2. Preprocessing (Cleaning & Tokenization)
3. Extractive Summarization (Lead-3, TextRank)
4. Abstractive Summarization (BART, T5)
5. Comprehensive Evaluation & ROUGE Benchmarking

Usage Examples:
  python main.py --mode summarize --model bart --sample_size 3
  python main.py --mode summarize --model t5 --text "News article text here..."
  python main.py --mode extractive --method textrank --sample_size 5
  python main.py --mode evaluate --sample_size 50
  python main.py --mode eda
  python main.py --mode preprocess
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    EVAL_SAMPLE_SIZE,
    PRIMARY_MODEL_NAME,
    ALTERNATIVE_MODEL_NAME,
    SUMMARIES_DIR,
    METRICS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def handle_summarize(args: argparse.Namespace) -> None:
    """Handle abstractive summarization CLI request."""
    from src.summarizer import get_summarizer
    from src.preprocessing import load_processed_split
    if args.sample_size <= 0:
        logger.error(f"Invalid sample_size ({args.sample_size}). Must be > 0.")
        sys.exit(1)
    if args.max_length < args.min_length:
        logger.error(f"max_length ({args.max_length}) cannot be smaller than min_length ({args.min_length}).")
        sys.exit(1)
    if args.num_beams < 1:
        logger.error(f"num_beams ({args.num_beams}) must be >= 1.")
        sys.exit(1)

    print("=" * 70)
    print(f"Abstractive Summarization: Model '{args.model.upper()}'")
    print("=" * 70)

    summarizer = get_summarizer(args.model)

    if args.text:
        # Summarize single provided text
        print(f"\n[Input Text]:\n{args.text.strip()}\n")
        res = summarizer.summarize_text(
            args.text,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
        )
        print(f"[{args.model.upper()} Summary]:\n{res['summary']}\n")
        print(f"Latency: {res['latency_sec']}s | Input Tokens: {res['input_tokens']} | Output Tokens: {res['output_tokens']} | Device: {res['device']}")
        return

    # Summarize dataset sample
    test_df = load_processed_split("test")
    sample_size = min(args.sample_size, len(test_df))
    subset = test_df.iloc[:sample_size]

    print(f"\nSummarizing {sample_size} articles from test split...")
    for idx, row in subset.iterrows():
        print("-" * 70)
        print(f"Sample #{idx + 1} (ID: {row['id']})")
        print(f"\n[Human Highlights]:\n{row['highlights']}\n")
        res = summarizer.summarize_text(
            row["article"],
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
        )
        print(f"[{args.model.upper()} Generated Summary]:\n{res['summary']}\n")
        print(f"Latency: {res['latency_sec']}s | Input Tokens: {res['input_tokens']} | Output Tokens: {res['output_tokens']}")


def handle_extractive(args: argparse.Namespace) -> None:
    """Handle extractive summarization CLI request."""
    from src.preprocessing import clean_text, load_processed_split
    from src.extractive import lead_n_summarize, textrank_summarize

    if args.num_sentences <= 0:
        logger.error(f"Invalid num_sentences ({args.num_sentences}). Must be > 0.")
        sys.exit(1)
    if args.sample_size <= 0:
        logger.error(f"Invalid sample_size ({args.sample_size}). Must be > 0.")
        sys.exit(1)

    print("=" * 70)
    print(f"Extractive Summarization: Method '{args.method.upper()}'")
    print("=" * 70)

    num_sent = args.num_sentences

    if args.text:
        cleaned = clean_text(args.text)
        print(f"\n[Input Text]:\n{cleaned}\n")
        if args.method == "lead3":
            summary = lead_n_summarize(cleaned, num_sentences=num_sent)
        else:
            summary = textrank_summarize(cleaned, num_sentences=num_sent)
        print(f"[{args.method.upper()} Summary ({num_sent} sentences)]:\n{summary}")
        return

    test_df = load_processed_split("test")
    sample_size = min(args.sample_size, len(test_df))
    subset = test_df.iloc[:sample_size]

    print(f"\nExtracting summaries for {sample_size} articles from test split...")
    for idx, row in subset.iterrows():
        print("-" * 70)
        print(f"Sample #{idx + 1} (ID: {row['id']})")
        print(f"\n[Human Highlights]:\n{row['highlights']}\n")
        if args.method == "lead3":
            summary = lead_n_summarize(row["article"], num_sentences=num_sent)
        elif args.method == "textrank":
            summary = textrank_summarize(row["article"], num_sentences=num_sent)
        else:
            # Both
            l3 = lead_n_summarize(row["article"], num_sentences=num_sent)
            tr = textrank_summarize(row["article"], num_sentences=num_sent)
            print(f"[Lead-3 Summary]:\n{l3}\n")
            print(f"[TextRank Summary]:\n{tr}")
            continue

        print(f"[{args.method.upper()} Summary]:\n{summary}")


def handle_evaluate(args: argparse.Namespace) -> None:
    """Handle benchmark evaluation CLI request."""
    from src.evaluation import run_full_benchmark

    if args.sample_size <= 0:
        logger.error(f"Invalid sample_size ({args.sample_size}). Must be > 0.")
        sys.exit(1)

    print("=" * 70)
    print(f"Full Comparative Benchmark Evaluation (Sample Size: {args.sample_size})")
    print("=" * 70)

    results = run_full_benchmark(sample_size=args.sample_size, save_outputs=True)
    metrics_df = results["metrics_df"]

    print("\n" + "=" * 70)
    print("ROUGE EVALUATION BENCHMARK RESULTS")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("\nResults and comparison samples saved in:")
    print(f"  - {METRICS_DIR / 'rouge_scores.csv'}")
    print(f"  - {METRICS_DIR / 'rouge_scores.json'}")
    print(f"  - {SUMMARIES_DIR / 'model_comparison_samples.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text Summarization Using Pre-trained Models - NLP Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Pipeline execution mode")

    # Mode: Summarize (Abstractive)
    parser_sum = subparsers.add_parser("summarize", help="Run abstractive summarization (BART or T5)")
    parser_sum.add_argument("--model", type=str, default="bart", choices=["bart", "t5"], help="Model architecture")
    parser_sum.add_argument("--text", type=str, default=None, help="Custom article text to summarize")
    parser_sum.add_argument("--sample_size", type=int, default=3, help="Number of test articles to summarize")
    parser_sum.add_argument("--max_length", type=int, default=128, help="Maximum generated token length")
    parser_sum.add_argument("--min_length", type=int, default=30, help="Minimum generated token length")
    parser_sum.add_argument("--num_beams", type=int, default=4, help="Beam search width")

    # Mode: Extractive
    parser_ext = subparsers.add_parser("extractive", help="Run extractive summarization (TextRank, Lead-3)")
    parser_ext.add_argument("--method", type=str, default="textrank", choices=["textrank", "lead3", "both"], help="Extractive method")
    parser_ext.add_argument("--text", type=str, default=None, help="Custom article text to summarize")
    parser_ext.add_argument("--num_sentences", type=int, default=3, help="Number of sentences to extract")
    parser_ext.add_argument("--sample_size", type=int, default=5, help="Number of test articles to summarize")

    # Mode: Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Run complete ROUGE evaluation benchmark")
    parser_eval.add_argument("--sample_size", type=int, default=EVAL_SAMPLE_SIZE, help="Number of test samples to evaluate")

    # Mode: EDA
    subparsers.add_parser("eda", help="Run exploratory data analysis and generate distribution charts")

    # Mode: Preprocess
    parser_prep = subparsers.add_parser("preprocess", help="Preprocess raw dataset splits and save to data/processed/")
    parser_prep.add_argument("--force", action="store_true", default=False, help="Force re-cleaning and overwrite existing processed CSV files")

    # Mode: Web / App (Launch Gradio)
    parser_app = subparsers.add_parser("app", help="Launch NewsLens Gradio web interface")
    parser_app.add_argument("--server_port", type=int, default=7860, help="Port to serve Gradio on")
    parser_app.add_argument("--server_name", type=str, default="127.0.0.1", help="Host address to serve Gradio on")
    parser_app.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link")

    # Alias: web
    parser_web = subparsers.add_parser("web", help="Launch NewsLens Gradio web interface (alias for app)")
    parser_web.add_argument("--server_port", type=int, default=7860, help="Port to serve Gradio on")
    parser_web.add_argument("--server_name", type=str, default="127.0.0.1", help="Host address to serve Gradio on")
    parser_web.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link")

    args = parser.parse_args()

    if args.mode == "summarize":
        handle_summarize(args)
    elif args.mode == "extractive":
        handle_extractive(args)
    elif args.mode == "evaluate":
        handle_evaluate(args)
    elif args.mode == "eda":
        from src.visualization import perform_full_eda
        perform_full_eda()
    elif args.mode == "preprocess":
        from src.preprocessing import preprocess_and_save_all_splits
        preprocess_and_save_all_splits(force=args.force)
    elif args.mode in ["app", "web"]:
        from app import launch_app
        launch_app(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
