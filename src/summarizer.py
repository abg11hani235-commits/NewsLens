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
import gc
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
    GENERATION_PRESETS,
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

        init_start = time.perf_counter()
        self._load_model_and_tokenizer()
        self.init_time_sec = round(time.perf_counter() - init_start, 3)
        self.inference_count = 0
        self.cold_latency_sec: Optional[float] = None
        self.warm_latencies: List[float] = []

    def warmup(self, sample_text: Optional[str] = None) -> float:
        """Warm up the model to trigger CUDA compilation, graph tracing, and memory paging.

        Subsequent inferences will record warm-start latencies without first-run overhead.

        Args:
            sample_text (Optional[str]): Optional text to warm up with.

        Returns:
            float: Latency of the warmup run in seconds.
        """
        warm_text = sample_text or "This is a brief news article used for model warmup to initialize GPU kernels and caches."
        logger.info(f"Warming up model '{self.model_name}'...")
        res = self.summarize_text(warm_text, max_length=64, min_length=10)
        self.cold_latency_sec = res["latency_sec"]
        logger.info(f"Model '{self.model_name}' warm-up complete in {self.cold_latency_sec:.3f}s.")
        return self.cold_latency_sec

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

        # Synchronize CUDA device before timing to ensure accurate latency measurement
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()
        used_cpu_fallback = False
        is_cold = (self.inference_count == 0 and self.cold_latency_sec is None)

        try:
            with torch.inference_mode():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_params,
                )
        except torch.cuda.OutOfMemoryError:
            used_cpu_fallback = True
            logger.warning(
                "CUDA Out-Of-Memory (OOM) encountered on GPU! Initiating safe CPU fallback..."
            )
            torch.cuda.empty_cache()

            # Safely transfer all input tensors (including attention_mask) to CPU
            cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
            original_device = self.device

            try:
                # Move model to CPU for execution
                self.model.to("cpu")
                with torch.inference_mode():
                    summary_ids = self.model.generate(
                        cpu_inputs["input_ids"],
                        attention_mask=cpu_inputs.get("attention_mask"),
                        **gen_params,
                    )
            finally:
                # Attempt to restore model to original GPU device if memory permits
                try:
                    torch.cuda.empty_cache()
                    self.model.to(original_device)
                except Exception as restore_err:
                    logger.warning(
                        f"Could not restore model back to {original_device} ({restore_err}). Remaining on CPU."
                    )
                    self.device = torch.device("cpu")
                    self.is_cuda = False

        # Synchronize CUDA device after generation to guarantee completed kernel execution
        if self.is_cuda and not used_cpu_fallback:
            torch.cuda.synchronize(self.device)

        elapsed = time.perf_counter() - start_time
        output_token_count = summary_ids.shape[1]

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        # Update cold vs warm latency statistics
        self.inference_count += 1
        latency_val = round(elapsed, 3)
        if is_cold:
            self.cold_latency_sec = latency_val
        else:
            self.warm_latencies.append(latency_val)

        # Retain tensor memory for caching allocator without calling empty_cache on hot path
        del inputs, summary_ids

        device_str = "cpu (oom-fallback)" if used_cpu_fallback else str(self.device)
        warm_avg = round(float(sum(self.warm_latencies) / len(self.warm_latencies)), 3) if self.warm_latencies else latency_val

        return {
            "summary": summary_text,
            "latency_sec": latency_val,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "device": device_str,
            "init_time_sec": self.init_time_sec,
            "is_cold": is_cold,
            "warm_latency_avg": warm_avg,
        }

    def unload_model(self) -> None:
        """Explicitly deallocate model and tokenizer from memory and clear GPU cache.

        Crucial for low-memory environments (such as 6GB VRAM or free cloud tiers)
        to prevent memory exhaustion when switching between models.
        """
        logger.info(f"Unloading model '{self.model_name}' from memory...")
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
        logger.info("Model unloaded and cache cleared successfully.")

    def summarize_batch(
        self,
        texts: List[str],
        batch_size: int = INFERENCE_BATCH_SIZE,
        **override_kwargs,
    ) -> List[Dict[str, Any]]:
        """Summarize a sequence of documents using true tokenized batch inference.

        Performs vectorized padding, truncation, batch generation, and batch decoding.
        If a CUDA OOM occurs during batch generation, it recovers gracefully by falling
        back item-by-item rather than failing the entire batch.

        Args:
            texts (List[str]): List of article strings.
            batch_size (int): Number of articles to process in each forward pass.
            **override_kwargs: Generation parameter overrides.

        Returns:
            List[Dict[str, Any]]: Ordered list of summarization result dictionaries.
        """
        if not texts:
            return []

        gen_params = dict(self.generation_params)
        gen_params.update(override_kwargs)

        total = len(texts)
        results: List[Optional[Dict[str, Any]]] = [None] * total

        for chunk_start in range(0, total, batch_size):
            chunk_indices = list(range(chunk_start, min(chunk_start + batch_size, total)))
            valid_sub_indices = []
            valid_prepared_texts = []

            for rel_idx, idx in enumerate(chunk_indices):
                raw_text = texts[idx]
                if raw_text and raw_text.strip():
                    valid_sub_indices.append(rel_idx)
                    valid_prepared_texts.append(self._prepare_input_text(raw_text))
                else:
                    results[idx] = {
                        "summary": "",
                        "latency_sec": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "device": str(self.device),
                        "init_time_sec": self.init_time_sec,
                        "is_cold": False,
                    }

            if not valid_prepared_texts:
                continue

            try:
                # Vectorized batch tokenization with dynamic padding
                batch_inputs = self.tokenizer(
                    valid_prepared_texts,
                    max_length=self.max_input_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                if self.is_cuda:
                    torch.cuda.synchronize(self.device)
                batch_start = time.perf_counter()

                with torch.inference_mode():
                    batch_summary_ids = self.model.generate(
                        batch_inputs["input_ids"],
                        attention_mask=batch_inputs.get("attention_mask"),
                        **gen_params,
                    )

                if self.is_cuda:
                    torch.cuda.synchronize(self.device)
                batch_elapsed = time.perf_counter() - batch_start
                per_item_latency = round(batch_elapsed / len(valid_prepared_texts), 3)

                # Batch decoding
                batch_summaries = self.tokenizer.batch_decode(
                    batch_summary_ids, skip_special_tokens=True
                )

                for k, sub_idx in enumerate(valid_sub_indices):
                    global_idx = chunk_indices[sub_idx]
                    if "attention_mask" in batch_inputs:
                        in_toks = int(batch_inputs["attention_mask"][k].sum().item())
                    else:
                        in_toks = int(batch_inputs["input_ids"][k].shape[0])
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
                    out_toks = int((batch_summary_ids[k] != pad_id).sum().item())

                    results[global_idx] = {
                        "summary": batch_summaries[k].strip(),
                        "latency_sec": per_item_latency,
                        "input_tokens": in_toks,
                        "output_tokens": out_toks,
                        "device": str(self.device),
                        "init_time_sec": self.init_time_sec,
                        "is_cold": False,
                    }
                    self.inference_count += 1
                    self.warm_latencies.append(per_item_latency)

                del batch_inputs, batch_summary_ids

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"CUDA OOM during batch generation (size {len(valid_prepared_texts)}). "
                    "Clearing GPU cache and falling back to sequential item processing..."
                )
                torch.cuda.empty_cache()
                for sub_idx in valid_sub_indices:
                    global_idx = chunk_indices[sub_idx]
                    results[global_idx] = self.summarize_text(texts[global_idx], **override_kwargs)

            except Exception as batch_err:
                logger.error(
                    f"Batch generation exception: {batch_err}. Falling back to single-text inference."
                )
                for sub_idx in valid_sub_indices:
                    global_idx = chunk_indices[sub_idx]
                    try:
                        results[global_idx] = self.summarize_text(texts[global_idx], **override_kwargs)
                    except Exception as single_err:
                        results[global_idx] = {
                            "summary": f"[Error: {str(single_err)}]",
                            "latency_sec": 0.0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "device": str(self.device),
                            "error": str(single_err),
                        }

            done_count = sum(1 for r in results if r is not None)
            if done_count % 10 == 0 or done_count == total:
                logger.info(f"Summarized {done_count}/{total} documents...")

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
    generation_params: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
) -> AbstractiveSummarizer:
    """Factory function to instantiate BART or T5 summarizer.

    Args:
        model_type (str): 'bart' for facebook/bart-large-cnn, 't5' for google-t5/t5-small.
        device (Optional[torch.device]): Target compute device.
        generation_params (Optional[Dict[str, Any]]): Custom generation parameters.
        preset (Optional[str]): Optional preset name ('quality' or 'fast').

    Returns:
        AbstractiveSummarizer: Configured summarizer instance.
    """
    params = {}
    if preset and preset in GENERATION_PRESETS:
        params.update(GENERATION_PRESETS[preset])
    if generation_params:
        params.update(generation_params)

    model_type_clean = model_type.lower().strip()
    if model_type_clean == "bart":
        return AbstractiveSummarizer(
            model_name=PRIMARY_MODEL_NAME,
            device=device,
            generation_params=params if params else None,
        )
    elif model_type_clean == "t5":
        return AbstractiveSummarizer(
            model_name=ALTERNATIVE_MODEL_NAME,
            device=device,
            generation_params=params if params else None,
        )
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
