"""Abstractive Summarization Module for NLP Text Summarization Project.

Implements pre-trained sequence-to-sequence Transformer summarization:
- Primary Model: facebook/bart-large-cnn (BART architecture, 1024 max input length)
- Alternative Model: google-t5/t5-small (T5 architecture, 512 max input length)

Hardware & Memory Optimization:
- Automatic CUDA device detection with fallback to CPU.
- FP16 (torch.float16) mixed precision on CUDA to fit within 6GB VRAM.
- torch.inference_mode() execution with explicit CUDA OOM recovery.
- Batch size 1 by default to ensure memory safety.
"""

import sys
import time
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from config import (
    PRIMARY_MODEL_NAME,
    ALTERNATIVE_MODEL_NAME,
    BART_MAX_INPUT_LENGTH,
    T5_MAX_INPUT_LENGTH,
    GENERATION_PARAMS,
    INFERENCE_BATCH_SIZE,
    SUMMARIES_DIR,
    get_device,
)
from src.preprocessing import clean_text

logger = logging.getLogger("summarizer")


class AbstractiveSummarizer:
    """Wrapper class for pre-trained Sequence-to-Sequence summarization models."""

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL_NAME,
        device: Optional[torch.device] = None,
        use_fp16: Optional[bool] = None,
        max_input_length: Optional[int] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the tokenizer, model, and hardware configuration.

        Args:
            model_name (str): Hugging Face model identifier.
            device (Optional[torch.device]): Target torch device (defaults to auto-detect).
            use_fp16 (Optional[bool]): Enable FP16 precision (defaults to True on CUDA).
            max_input_length (Optional[int]): Model sequence token truncation limit.
            generation_params (Optional[Dict[str, Any]]): Default beam search parameters.
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.is_cuda = self.device.type == "cuda"
        self.use_fp16 = use_fp16 if use_fp16 is not None else self.is_cuda
        self.is_t5 = "t5" in model_name.lower()

        # Set max input token length based on model family
        if max_input_length is not None:
            self.max_input_length = max_input_length
        elif self.is_t5:
            self.max_input_length = T5_MAX_INPUT_LENGTH
        else:
            self.max_input_length = BART_MAX_INPUT_LENGTH

        # Default generation parameters
        self.generation_params = dict(GENERATION_PARAMS)
        if generation_params:
            self.generation_params.update(generation_params)

        logger.info(
            f"Loading model '{self.model_name}' on {self.device} (FP16={self.use_fp16}, max_input={self.max_input_length})..."
        )

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Load pretrained tokenizer and model weights with appropriate precision."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if self.use_fp16 and self.is_cuda:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        self.model.to(self.device)
        self.model.eval()

        if self.is_cuda:
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            logger.info(f"Model loaded into GPU VRAM: {vram_mb:.1f} MB allocated.")

    def _prepare_input_text(self, text: str) -> str:
        """Preprocess text and prepend task prefixes where required (e.g. T5)."""
        cleaned = clean_text(text)
        if self.is_t5 and not cleaned.lower().startswith("summarize:"):
            return f"summarize: {cleaned}"
        return cleaned

    def summarize_text(
        self,
        text: str,
        **override_kwargs,
    ) -> Dict[str, Any]:
        """Summarize a single document with inference timing and resource tracking.

        Args:
            text (str): Input article text.
            **override_kwargs: Optional generation parameter overrides.

        Returns:
            Dict[str, Any]: Dictionary containing 'summary', 'latency_sec',
                            'input_tokens', 'output_tokens', and 'device'.
        """
        if not text or not text.strip():
            return {
                "summary": "",
                "latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "device": str(self.device),
            }

        prepared_text = self._prepare_input_text(text)
        gen_params = dict(self.generation_params)
        gen_params.update(override_kwargs)

        # Tokenize input
        inputs = self.tokenizer(
            prepared_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        input_token_count = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        try:
            with torch.inference_mode():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_params,
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM detected! Clearing cache and attempting CPU fallback...")
            torch.cuda.empty_cache()
            # Temporarily move to CPU for fallback
            cpu_model = self.model.to("cpu")
            with torch.inference_mode():
                summary_ids = cpu_model.generate(
                    inputs["input_ids"].to("cpu"),
                    **gen_params,
                )
            self.model.to(self.device)

        elapsed = time.perf_counter() - start_time
        output_token_count = summary_ids.shape[1]

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        # Clean GPU cache after generation
        if self.is_cuda:
            del inputs, summary_ids
            torch.cuda.empty_cache()

        return {
            "summary": summary_text,
            "latency_sec": round(elapsed, 3),
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "device": str(self.device),
        }

    def summarize_batch(
        self,
        texts: List[str],
        batch_size: int = INFERENCE_BATCH_SIZE,
        **override_kwargs,
    ) -> List[Dict[str, Any]]:
        """Summarize a sequence of documents in batches.

        Args:
            texts (List[str]): List of article strings.
            batch_size (int): Batch size (defaults to 1 for 6GB VRAM safety).
            **override_kwargs: Generation parameter overrides.

        Returns:
            List[Dict[str, Any]]: List of summarization result dictionaries.
        """
        results = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            for text in batch_texts:
                res = self.summarize_text(text, **override_kwargs)
                results.append(res)
                if (len(results)) % 10 == 0 or len(results) == total:
                    logger.info(f"Summarized {len(results)}/{total} documents...")

        return results

    def summarize_dataframe(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = 50,
        text_column: str = "article",
        output_prefix: str = "bart",
        save_output: bool = True,
    ) -> pd.DataFrame:
        """Run abstractive summarization on a DataFrame sample and append results.

        Args:
            df (pd.DataFrame): DataFrame containing input articles and highlights.
            sample_size (Optional[int]): Number of records to summarize.
            text_column (str): Column name containing source text.
            output_prefix (str): Prefix for generated columns (e.g. 'bart' -> 'bart_summary').
            save_output (bool): Whether to save to CSV in outputs/summaries/.

        Returns:
            pd.DataFrame: Augmented DataFrame with generated summary and latency columns.
        """
        if sample_size is not None and len(df) > sample_size:
            eval_df = df.iloc[:sample_size].copy()
        else:
            eval_df = df.copy()

        logger.info(
            f"Generating abstractive summaries using {self.model_name} on {len(eval_df)} records..."
        )

        summaries = []
        latencies = []
        in_tokens = []
        out_tokens = []

        articles = eval_df[text_column].tolist()
        batch_results = self.summarize_batch(articles, batch_size=INFERENCE_BATCH_SIZE)

        for res in batch_results:
            summaries.append(res["summary"])
            latencies.append(res["latency_sec"])
            in_tokens.append(res["input_tokens"])
            out_tokens.append(res["output_tokens"])

        eval_df[f"{output_prefix}_summary"] = summaries
        eval_df[f"{output_prefix}_latency_sec"] = latencies
        eval_df[f"{output_prefix}_input_tokens"] = in_tokens
        eval_df[f"{output_prefix}_output_tokens"] = out_tokens

        if save_output:
            SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
            out_file = SUMMARIES_DIR / f"{output_prefix}_summaries.csv"
            eval_df.to_csv(out_file, index=False, encoding="utf-8")
            logger.info(f"Saved {output_prefix} summaries to {out_file}")

        return eval_df


def get_summarizer(
    model_type: str = "bart",
    device: Optional[torch.device] = None,
) -> AbstractiveSummarizer:
    """Factory function to instantiate BART or T5 summarizer.

    Args:
        model_type (str): 'bart' for facebook/bart-large-cnn, 't5' for google-t5/t5-small.
        device (Optional[torch.device]): Target compute device.

    Returns:
        AbstractiveSummarizer: Configured summarizer instance.
    """
    model_type_clean = model_type.lower().strip()
    if model_type_clean == "bart":
        return AbstractiveSummarizer(model_name=PRIMARY_MODEL_NAME, device=device)
    elif model_type_clean == "t5":
        return AbstractiveSummarizer(model_name=ALTERNATIVE_MODEL_NAME, device=device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'bart' or 't5'.")


if __name__ == "__main__":
    from src.preprocessing import load_processed_split

    print("=" * 70)
    print("Abstractive Summarization Benchmark: facebook/bart-large-cnn")
    print("=" * 70)

    test_df = load_processed_split("test")
    sample_article = test_df.iloc[0]["article"]
    sample_highlights = test_df.iloc[0]["highlights"]

    print("\n[Human Reference Highlights]:")
    print(sample_highlights)

    # Instantiate BART
    bart = get_summarizer("bart")
    res = bart.summarize_text(sample_article)

    print("\n[BART-Large-CNN Generated Summary]:")
    print(res["summary"])
    print(f"Latency: {res['latency_sec']}s | Input Tokens: {res['input_tokens']} | Output Tokens: {res['output_tokens']} | Device: {res['device']}")
