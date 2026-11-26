"""NewsLens - AI-Powered News Summarization Web Application.

A production-grade Gradio application comparing extractive (Lead-3, TextRank)
and pre-trained abstractive (BART, T5) sequence-to-sequence summarization models.

Features:
- Single-Article Summarization with real-time resource tracking (latency, tokens, device).
- Side-by-Side Model Comparison with memory-conscious sequential loading.
- Pre-loaded real-world news articles for one-click testing.
- Interactive Academic Benchmark & Architecture Guide.
"""

import sys
import gc
import time
import inspect
import logging
import warnings
import threading
import contextlib
from collections import OrderedDict
from typing import Dict, Any, Tuple, Optional, List, Iterator
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import gradio as gr

from config import (
    PRIMARY_MODEL_NAME,
    ALTERNATIVE_MODEL_NAME,
    BART_MAX_INPUT_LENGTH,
    T5_MAX_INPUT_LENGTH,
    GENERATION_PARAMS,
    get_device,
)
from src.preprocessing import clean_text, ensure_nltk_resources, get_text_length_info
from src.extractive import lead_n_summarize, textrank_summarize
from src.summarizer import AbstractiveSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("newslens_app")

# ==============================================================================
# Lifecycle-Safe Model Proxy for Legacy Compatibility
# ==============================================================================

class SafeModelProxy(AbstractiveSummarizer):
    """Lifecycle-safe proxy wrapper for legacy ModelManager model retrieval.

    Protects callers using deprecated get_model() / get_model_with_metadata()
    by routing supported inference operations through ModelManager.lease_model().
    This prevents concurrent eviction, use-after-unload conditions, or
    accessing deallocated tokenizers during in-flight inference requests.

    Encapsulation & Resource Lifecycle Guarantees:
    - Does not retain any reference to the underlying model, tokenizer, or
      ModelManager on ordinary proxy instance attributes (`__dict__`).
    - Direct access to `.model`, `.tokenizer`, or unsupported internals via
      ordinary attribute access raises descriptive errors with migration guidance.
    - Supported metadata properties read safe values dynamically without
      exposing mutable internal state or retaining references after lease exit.
    """

    SAFE_METADATA_ATTRIBUTES = frozenset({
        "model_name",
        "device",
        "is_cuda",
        "is_t5",
        "max_input_length",
        "generation_params",
        "init_time_sec",
        "cold_latency_sec",
        "warm_latencies",
        "inference_count",
        "is_unloaded",
        "is_inferring",
        "use_fp16",
    })

    def __init__(self, manager: "ModelManager", model_key: str):
        self.model_name = PRIMARY_MODEL_NAME if model_key == "bart" else ALTERNATIVE_MODEL_NAME
        self.device = get_device()
        self.is_cuda = (self.device.type == "cuda")
        self.is_t5 = "t5" in self.model_name.lower()
        self.use_fp16 = self.is_cuda
        self.max_input_length = T5_MAX_INPUT_LENGTH if self.is_t5 else BART_MAX_INPUT_LENGTH
        self._generation_params = dict(GENERATION_PARAMS)
        self._model_key = model_key
        fallback_lock = threading.RLock()

        def _dispatch_lease(fn_name: str, *args, **kwargs) -> Any:
            with manager.lease_model(model_key, warmup=False) as (model, _):
                if fn_name == "warmup":
                    if hasattr(model, "warmup"):
                        return model.warmup(*args, **kwargs)
                    return 0.0
                return getattr(model, fn_name)(*args, **kwargs)

        def _query_state(attr: str, default: Any = None) -> Any:
            with manager._lock:
                if attr == "is_inferring":
                    if manager._active_leases.get(model_key, 0) > 0:
                        return True
                    cached = manager._cache.get(model_key)
                    return getattr(cached, "is_inferring", False) if cached is not None else False
                if attr == "is_unloaded":
                    cached = manager._cache.get(model_key)
                    if cached is None:
                        return True
                    return getattr(cached, "is_unloaded", False)
                if attr == "_inference_lock":
                    cached = manager._cache.get(model_key)
                    if cached is not None and hasattr(cached, "_inference_lock"):
                        return cached._inference_lock
                    return fallback_lock
                cached = manager._cache.get(model_key)
                if cached is not None and hasattr(cached, attr):
                    val = getattr(cached, attr)
                    if isinstance(val, list):
                        return list(val)
                    if isinstance(val, dict):
                        return dict(val)
                    return val
                return default

        def _dispatch_unload() -> None:
            manager.unload_model(model_key)

        self._dispatch_lease = _dispatch_lease
        self._query_state = _query_state
        self._dispatch_unload = _dispatch_unload

    @property
    def generation_params(self) -> Dict[str, Any]:
        cached_params = self._query_state("generation_params", None)
        if isinstance(cached_params, dict):
            return dict(cached_params)
        return dict(self._generation_params)

    @property
    def init_time_sec(self) -> float:
        return float(self._query_state("init_time_sec", 0.0))

    @property
    def _inference_lock(self) -> threading.RLock:
        return self._query_state("_inference_lock")

    @property
    def is_unloaded(self) -> bool:
        return bool(self._query_state("is_unloaded", True))

    @property
    def is_inferring(self) -> bool:
        return bool(self._query_state("is_inferring", False))

    @property
    def cold_latency_sec(self) -> Optional[float]:
        return self._query_state("cold_latency_sec", None)

    @property
    def warm_latencies(self) -> List[float]:
        return list(self._query_state("warm_latencies", []))

    @property
    def inference_count(self) -> int:
        return int(self._query_state("inference_count", 0))

    @property
    def _load_model_and_tokenizer(self) -> Any:
        raise AttributeError(
            f"'{self.__class__.__name__}' does not expose raw model internal or method '_load_model_and_tokenizer'. "
            "Returning bound methods or internal mutable resources outside an active lease causes "
            "use-after-unload race conditions and GPU memory corruption. "
            f"Use 'with MODEL_MANAGER.lease_model(\"{self._model_key}\") as (model, meta):' "
            "to access model internals safely while holding an active lease."
        )

    @property
    def _prepare_input_text(self) -> Any:
        raise AttributeError(
            f"'{self.__class__.__name__}' does not expose raw model internal or method '_prepare_input_text'. "
            "Returning bound methods or internal mutable resources outside an active lease causes "
            "use-after-unload race conditions and GPU memory corruption. "
            f"Use 'with MODEL_MANAGER.lease_model(\"{self._model_key}\") as (model, meta):' "
            "to access model internals safely while holding an active lease."
        )

    @property
    def model(self) -> Any:
        raise RuntimeError(
            f"Direct access to raw model weights on '{self.__class__.__name__}' is disabled "
            "to prevent use-after-unload race conditions and GPU memory corruption outside active leases. "
            f"Use 'with MODEL_MANAGER.lease_model(\"{self._model_key}\") as (model, meta):' "
            "to safely access model internals under an active lease."
        )

    @property
    def tokenizer(self) -> Any:
        raise RuntimeError(
            f"Direct access to raw tokenizer on '{self.__class__.__name__}' is disabled "
            "to prevent race conditions and accessing deallocated resources outside active leases. "
            f"Use 'with MODEL_MANAGER.lease_model(\"{self._model_key}\") as (model, meta):' "
            "to safely access tokenizer internals under an active lease."
        )

    def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        return self._dispatch_lease("summarize_text", text, **kwargs)

    def summarize_batch(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        return self._dispatch_lease("summarize_batch", texts, **kwargs)

    def warmup(self, sample_text: Optional[str] = None) -> float:
        return self._dispatch_lease("warmup", sample_text=sample_text)

    def unload_model(self) -> None:
        self._dispatch_unload()

    def is_inferencing(self) -> bool:
        return self.is_inferring

    def __getattr__(self, name: str) -> Any:
        if name in self.SAFE_METADATA_ATTRIBUTES:
            val = self._query_state(name, None)
            if val is not None:
                return val
            if name in self.__dict__:
                return self.__dict__[name]
            raise AttributeError(f"Metadata attribute '{name}' is not currently available on '{self.__class__.__name__}'.")

        if name in ("model", "tokenizer"):
            raise RuntimeError(
                f"Direct access to raw '{name}' on '{self.__class__.__name__}' is disabled "
                "to prevent use-after-unload race conditions and GPU memory corruption outside active leases. "
                f"Use 'with MODEL_MANAGER.lease_model(\"{self._model_key}\") as (model, meta):' "
                "to safely access model internals under an active lease."
            )

        raise AttributeError(
            f"'{self.__class__.__name__}' does not expose raw model internal or method '{name}'. "
            "Returning bound methods or internal mutable resources outside an active lease causes "
            "use-after-unload race conditions and GPU memory corruption. "
            f"Use 'with MODEL_MANAGER.lease_model(\"{self._model_key}\") as (model, meta):' "
            f"to access '{name}' safely while holding an active lease."
        )


# ==============================================================================
# Model Manager (Thread-Safe, Cached & Memory-Budgeted)
# ==============================================================================

class ModelManager:
    """Thread-safe manager for neural summarization models with safe caching, leasing, and resource management.

    Key Features:
    - Safe Inference Lifecycle Leasing: `lease_model(model_key)` context manager ensures a model
      currently used by an active inference request is never evicted by concurrent requests.
    - Condition Variable Synchronization: Waiting threads gracefully block if all cached models
      are actively leased, resuming immediately when the active lease releases.
    - Separation of model initialization/loading time from inference time.
    - Cold-start and warm-start latency tracking.
    - Memory-aware eviction (default max_cached_models=1 for constrained 6GB VRAM/cloud environments).
    - Automatic warmup support.
    """

    def __init__(self, max_cached_models: int = 1):
        self.max_cached_models = max_cached_models
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._cache: "OrderedDict[str, AbstractiveSummarizer]" = OrderedDict()
        self._active_leases: Dict[str, int] = {}
        self._proxies: Dict[str, SafeModelProxy] = {}
        self._last_loaded_metadata: Dict[str, Any] = {}

    @property
    def active_model_name(self) -> Optional[str]:
        with self._lock:
            if self._cache:
                return next(reversed(self._cache))
            return None

    @active_model_name.setter
    def active_model_name(self, value: Optional[str]) -> None:
        pass

    @property
    def active_summarizer(self) -> Optional[AbstractiveSummarizer]:
        with self._lock:
            if self._cache:
                return self._cache[next(reversed(self._cache))]
            return None

    @active_summarizer.setter
    def active_summarizer(self, value: Optional[AbstractiveSummarizer]) -> None:
        pass

    def is_leased(self, model_choice: str) -> bool:
        """Check if a model currently has active leases."""
        with self._lock:
            return self._active_leases.get(model_choice.lower().strip(), 0) > 0

    def get_lease_count(self, model_choice: str) -> int:
        """Return the number of active leases for a model."""
        with self._lock:
            return self._active_leases.get(model_choice.lower().strip(), 0)

    def _is_model_evictable(self, key: str, model: Any) -> bool:
        """Return True if model has no active leases AND is not currently executing inference.

        Protects both actively leased models (via reference counting) and legacy-retrieved models
        (via checking active inference locks or is_inferring flags).
        """
        if self._active_leases.get(key, 0) > 0:
            return False
        if hasattr(model, "is_inferring") and model.is_inferring:
            return False
        if hasattr(model, "_inference_lock"):
            acquired = model._inference_lock.acquire(blocking=False)
            if not acquired:
                return False
            model._inference_lock.release()
        return True

    @contextlib.contextmanager
    def lease_model(
        self,
        model_choice: str,
        warmup: bool = False,
        timeout: float = 60.0,
    ) -> Iterator[Tuple[AbstractiveSummarizer, Dict[str, Any]]]:
        """Lease a summarization model for the duration of inference.

        Guarantees:
        - The leased model CANNOT be evicted or unloaded by another thread while this lease is active.
        - Other threads attempting to evict this model will safely wait on the condition variable.
        - Once inference completes and the context exits, the lease is released and waiting threads are notified.

        Args:
            model_choice (str): 'bart' or 't5'.
            warmup (bool): Whether to perform model warmup if loaded from scratch.
            timeout (float): Max seconds to wait if all models are leased.

        Yields:
            Tuple[AbstractiveSummarizer, Dict[str, Any]]: (model_instance, metadata_dict)
        """
        model_key = model_choice.lower().strip()
        if model_key not in ["bart", "t5"]:
            raise ValueError(f"Invalid model choice '{model_choice}'. Choose 'bart' or 't5'.")

        start_wait = time.perf_counter()
        with self._lock:
            # 1. If not cached and cache is at capacity, ensure an unleased and idle model exists to evict
            while model_key not in self._cache and len(self._cache) >= self.max_cached_models:
                evictable = [k for k, m in self._cache.items() if self._is_model_evictable(k, m)]
                if evictable:
                    evict_k = evictable[0]
                    evicted_model = self._cache.pop(evict_k)
                    logger.info(f"Evicting idle model '{evict_k}' to free memory for '{model_key}'...")
                    evicted_model.unload_model()
                    break
                else:
                    remaining = timeout - (time.perf_counter() - start_wait)
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out after {timeout}s waiting for active leases or in-flight inference on {list(self._cache.keys())} to complete before loading '{model_key}'."
                        )
                    logger.info(
                        f"All cached models {list(self._cache.keys())} are in use. Waiting for availability before loading '{model_key}'..."
                    )
                    self._cond.wait(timeout=min(remaining, 0.1))

            # 2. Retrieve from cache or load on demand
            if model_key in self._cache:
                summarizer = self._cache[model_key]
                self._cache.move_to_end(model_key)
                meta = {
                    "model_key": model_key,
                    "is_cached": True,
                    "load_time_sec": 0.0,
                    "init_time_sec": getattr(summarizer, "init_time_sec", 0.0),
                    "is_warmed": summarizer.cold_latency_sec is not None,
                }
            else:
                target_model_name = PRIMARY_MODEL_NAME if model_key == "bart" else ALTERNATIVE_MODEL_NAME
                logger.info(f"Loading '{target_model_name}' on demand under lease...")
                load_start = time.perf_counter()
                summarizer = AbstractiveSummarizer(
                    model_name=target_model_name,
                    device=get_device(),
                )
                load_elapsed = round(time.perf_counter() - load_start, 3)
                if warmup:
                    summarizer.warmup()

                self._cache[model_key] = summarizer
                meta = {
                    "model_key": model_key,
                    "is_cached": False,
                    "load_time_sec": load_elapsed,
                    "init_time_sec": summarizer.init_time_sec,
                    "is_warmed": summarizer.cold_latency_sec is not None,
                }

            # 3. Increment lease count
            self._active_leases[model_key] = self._active_leases.get(model_key, 0) + 1
            self._last_loaded_metadata = meta

        try:
            yield summarizer, meta
        finally:
            with self._lock:
                self._active_leases[model_key] = max(0, self._active_leases.get(model_key, 1) - 1)
                self._cond.notify_all()

    def get_model(
        self,
        model_choice: str,
        warmup: bool = False,
        timeout: float = 60.0,
    ) -> AbstractiveSummarizer:
        """Retrieve a lifecycle-safe summarizer proxy (DEPRECATED).

        .. deprecated:: 2.1
            Use :meth:`lease_model` context manager instead. Direct model retrieval
            returns a SafeModelProxy that routes inference calls through lease_model()
            to protect against concurrent evictions and use-after-unload conditions.

        Args:
            model_choice (str): 'bart' or 't5'.
            warmup (bool): Whether to perform model warmup if loaded from scratch.
            timeout (float): Max seconds to wait if all cached models are in use.

        Returns:
            AbstractiveSummarizer: Lifecycle-safe summarizer instance.
        """
        warnings.warn(
            "ModelManager.get_model() is deprecated; callers should acquire a lease "
            "using ModelManager.lease_model() context manager before running inference.",
            DeprecationWarning,
            stacklevel=2,
        )
        summarizer, _ = self.get_model_with_metadata(model_choice, warmup=warmup, timeout=timeout)
        return summarizer

    def get_model_with_metadata(
        self,
        model_choice: str,
        warmup: bool = False,
        timeout: float = 60.0,
    ) -> Tuple[AbstractiveSummarizer, Dict[str, Any]]:
        """Retrieve a lifecycle-safe summarizer proxy along with metadata (DEPRECATED).

        .. deprecated:: 2.1
            Use :meth:`lease_model` context manager instead. Direct model retrieval
            returns a SafeModelProxy that routes inference calls through lease_model()
            to protect against concurrent evictions and use-after-unload conditions.

        Returns:
            Tuple[AbstractiveSummarizer, Dict[str, Any]]: (proxy_model, metadata_dict)
        """
        warnings.warn(
            "ModelManager.get_model_with_metadata() is deprecated; callers should acquire a lease "
            "using ModelManager.lease_model() context manager before running inference.",
            DeprecationWarning,
            stacklevel=2,
        )
        model_key = model_choice.lower().strip()
        if model_key not in ["bart", "t5"]:
            raise ValueError(f"Invalid model choice '{model_choice}'. Choose 'bart' or 't5'.")

        with self._lock:
            if model_key not in self._proxies:
                self._proxies[model_key] = SafeModelProxy(self, model_key)
            proxy = self._proxies[model_key]

        with self.lease_model(model_key, warmup=warmup, timeout=timeout) as (_, meta):
            res_meta = dict(meta)

        return proxy, res_meta

    def unload_model(self, model_choice: str, force: bool = False) -> bool:
        """Unload a specific model if not actively leased or executing inference.

        Removes model from cache before invoking unload_model to prevent new leases,
        and executes unload_model safely under the model's inference lock.
        """
        model_key = model_choice.lower().strip()
        with self._lock:
            if model_key not in self._cache:
                return False
            model = self._cache[model_key]
            if not force and not self._is_model_evictable(model_key, model):
                logger.warning(
                    f"Model '{model_key}' has active leases ({self._active_leases.get(model_key, 0)}) "
                    "or is currently inferencing. Cannot unload without force=True."
                )
                return False
            del self._cache[model_key]
            logger.info(f"Unloading model '{model_key}'...")
            model.unload_model()
            self._cond.notify_all()
            return True

    def unload_all(self, force: bool = False) -> None:
        """Unload all active models completely and release resources."""
        with self._lock:
            for key in list(self._cache.keys()):
                model = self._cache.get(key)
                if model is None:
                    continue
                if not force and not self._is_model_evictable(key, model):
                    logger.warning(
                        f"Model '{key}' has active leases ({self._active_leases.get(key, 0)}) "
                        "or is currently inferencing. Skipping unload."
                    )
                    continue
                del self._cache[key]
                logger.info(f"Unloading model '{key}'...")
                model.unload_model()
            if not self._cache:
                self._last_loaded_metadata = {}
            self._cond.notify_all()


# Global singleton instance
MODEL_MANAGER = ModelManager()

# ==============================================================================
# Sample Benchmark News Articles
# ==============================================================================

SAMPLE_ARTICLES = {
    "Diplomacy / International (ICC Accession)": (
        "The Palestinian Authority officially became the 123rd member of the International Criminal Court "
        "on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. "
        "The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. "
        "The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction "
        "over alleged crimes committed in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014. "
        "Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis. "
        "The ICC opened a preliminary examination into the situation in Palestinian territories in January, "
        "shortly after the Palestinians signed the Rome Statute."
    ),
    "Technology / AI Regulation (Brussels Agreement)": (
        "The European Union agreed on landmark artificial intelligence regulations today in Brussels after marathon "
        "negotiations between member states and lawmakers. Officials hailed the legislation as a historic step for consumer safety, "
        "establishing strict guardrails for high-risk AI applications in healthcare, employment, and biometric identification. "
        "Under the new framework, foundation models and general-purpose systems must adhere to strict transparency standards, "
        "technical documentation audits, and copyright compliance. Technology industry leaders expressed cautious optimism, "
        "while highlighting concerns over compliance burdens and European competitiveness in global research."
    ),
    "Science / Space Exploration (James Webb Discovery)": (
        "Astronomers analyzing infrared imagery from NASA's James Webb Space Telescope have confirmed the discovery of an "
        "ancient galaxy formed just 300 million years after the Big Bang. The galaxy, designated JADES-GS-z14-0, exhibits "
        "surprising brightness and mass, challenging prevailing theoretical models of early cosmic structure formation. "
        "Spectroscopic data revealed that the galaxy contains significant amounts of oxygen and ionized dust, indicating that "
        "multiple generations of massive stars had already lived and died in the earliest epoch of the universe. Researchers "
        "believe this demonstrates that early galaxy development occurred much faster than previously estimated."
    ),
}

# ==============================================================================
# Core Interface Inference Functions
# ==============================================================================

def update_text_counts(text: str) -> str:
    """Dynamically compute word and character counts as the user types."""
    if not text:
        return "Word Count: 0 | Character Count: 0"
    words = len(text.split())
    chars = len(text)
    return f"Word Count: {words:,} | Character Count: {chars:,}"


def summarize_article(
    article_text: str,
    model_choice: str,
    num_sentences: int = 3,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4,
) -> Tuple[str, str, str, str, str]:
    """Execute summarization for single document based on model selection.

    Returns:
        Tuple: (Summary Text, Latency/Load String, Device String, Word Count String, Compression Ratio String)
    """
    if not article_text or not article_text.strip():
        raise gr.Error("Please provide an article to summarize.")

    cleaned = clean_text(article_text)
    input_words = len(cleaned.split())

    if input_words < 10:
        raise gr.Error("Input text is too short. Please provide at least 10 words for a meaningful summary.")

    model_clean = model_choice.lower().strip()

    try:
        if "lead-3" in model_clean or "lead3" in model_clean:
            t0 = time.perf_counter()
            summary = lead_n_summarize(cleaned, num_sentences=num_sentences)
            elapsed = time.perf_counter() - t0
            device = "CPU (Heuristic)"
            latency_display = f"{elapsed:.3f} seconds (Extractive Heuristic)"

        elif "textrank" in model_clean:
            t0 = time.perf_counter()
            summary = textrank_summarize(cleaned, num_sentences=num_sentences)
            elapsed = time.perf_counter() - t0
            device = "CPU (PageRank Graph)"
            latency_display = f"{elapsed:.3f} seconds (Graph PageRank)"

        elif "bart" in model_clean:
            with MODEL_MANAGER.lease_model("bart") as (summarizer, meta):
                res = summarizer.summarize_text(
                    cleaned,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                )
                summary = res["summary"]
                elapsed = res["latency_sec"]
                e2e_elapsed = res.get("total_e2e_latency_sec", elapsed)
                device = res["device"]
                load_time = meta["load_time_sec"]
                is_cached = meta["is_cached"]
                is_cold = res.get("is_cold", False)

                load_str = "Load: 0.00s (Cached)" if is_cached else f"Load: {load_time:.2f}s (Cold)"
                lat_type = "Cold" if is_cold else "Warm"
                latency_display = f"Gen: {elapsed:.3f}s ({lat_type}) | E2E: {e2e_elapsed:.3f}s | {load_str}"

        elif "t5" in model_clean:
            with MODEL_MANAGER.lease_model("t5") as (summarizer, meta):
                res = summarizer.summarize_text(
                    cleaned,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                )
                summary = res["summary"]
                elapsed = res["latency_sec"]
                e2e_elapsed = res.get("total_e2e_latency_sec", elapsed)
                device = res["device"]
                load_time = meta["load_time_sec"]
                is_cached = meta["is_cached"]
                is_cold = res.get("is_cold", False)

                load_str = "Load: 0.00s (Cached)" if is_cached else f"Load: {load_time:.2f}s (Cold)"
                lat_type = "Cold" if is_cold else "Warm"
                latency_display = f"Gen: {elapsed:.3f}s ({lat_type}) | E2E: {e2e_elapsed:.3f}s | {load_str}"

        else:
            raise gr.Error(f"Unsupported model: {model_choice}")

        out_words = len(summary.split())
        ratio = round(input_words / max(out_words, 1), 1)

        word_count_display = f"{out_words} words (Original: {input_words} words)"
        ratio_display = f"{ratio}x Compression"

        return summary, latency_display, device, word_count_display, ratio_display

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise gr.Error(f"Summarization Error: {str(e)}")


def compare_all_models(
    article_text: str,
    num_sentences: int = 3,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4,
) -> Tuple[str, str, str, str, str]:
    """Execute sequential memory-aware comparison across Lead-3, TextRank, T5, and BART.

    Returns:
        Tuple: (Lead-3 Output, TextRank Output, T5 Output, BART Output, Comparison Table Markdown)
    """
    if not article_text or not article_text.strip():
        raise gr.Error("Please provide an article to compare models.")

    cleaned = clean_text(article_text)
    input_words = len(cleaned.split())

    if input_words < 10:
        raise gr.Error("Article text is too short to benchmark.")

    logger.info("Executing sequential model comparison...")

    # 1. Lead-3 (CPU)
    t0 = time.perf_counter()
    lead3_sum = lead_n_summarize(cleaned, num_sentences=num_sentences)
    lead3_lat = time.perf_counter() - t0

    # 2. TextRank (CPU)
    t0 = time.perf_counter()
    textrank_sum = textrank_summarize(cleaned, num_sentences=num_sentences)
    textrank_lat = time.perf_counter() - t0

    # 3. T5-Small (Lazy load, run, then safely release lease before BART)
    with MODEL_MANAGER.lease_model("t5") as (t5_model, t5_meta):
        t5_res = t5_model.summarize_text(
            cleaned,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
        )
        t5_sum = t5_res["summary"]
        t5_lat = t5_res["latency_sec"]
        t5_dev = t5_res["device"]
        t5_status = "Cached" if t5_meta["is_cached"] else f"{t5_meta['load_time_sec']:.2f}s Load"

    # 4. BART-Large-CNN (ModelManager safely evicts idle T5 once its lease is released)
    with MODEL_MANAGER.lease_model("bart") as (bart_model, bart_meta):
        bart_res = bart_model.summarize_text(
            cleaned,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
        )
        bart_sum = bart_res["summary"]
        bart_lat = bart_res["latency_sec"]
        bart_dev = bart_res["device"]
        bart_status = "Cached" if bart_meta["is_cached"] else f"{bart_meta['load_time_sec']:.2f}s Load"

    # Construct comparison summary table
    table_md = f"""
| Architecture | Paradigm | Parameters | Inference Latency | Model Load Status | Summary Words | Compression | Device |
|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Heuristic | N/A | {lead3_lat:.3f}s | Instant (Heuristic) | {len(lead3_sum.split())} words | {input_words/max(len(lead3_sum.split()),1):.1f}x | CPU |
| **TextRank** | Extractive Graph | N/A | {textrank_lat:.3f}s | Instant (Graph) | {len(textrank_sum.split())} words | {input_words/max(len(textrank_sum.split()),1):.1f}x | CPU |
| **T5-Small** | Abstractive Transformer | 60M | {t5_lat:.3f}s | {t5_status} | {len(t5_sum.split())} words | {input_words/max(len(t5_sum.split()),1):.1f}x | {t5_dev} |
| **BART-Large-CNN** | Abstractive Transformer | 406M | {bart_lat:.3f}s | {bart_status} | {len(bart_sum.split())} words | {input_words/max(len(bart_sum.split()),1):.1f}x | {bart_dev} |
"""
    return lead3_sum, textrank_sum, t5_sum, bart_sum, table_md


# ==============================================================================
# ==============================================================================
# UI Styling & Gradio Version Compatibility
# ==============================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 1.5rem; }
.metric-card { background: #f8fafc; border-radius: 8px; padding: 12px; border: 1px solid #e2e8f0; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
)


def _get_copy_kwargs() -> dict:
    """Return appropriate copy button kwargs for Gradio 5 vs Gradio 6."""
    sig = inspect.signature(gr.Textbox.__init__).parameters
    if "buttons" in sig:
        return {"buttons": ["copy"]}
    if "show_copy_button" in sig:
        return {"show_copy_button": True}
    return {}


# ==============================================================================
# Gradio UI Construction
# ==============================================================================

def create_ui() -> gr.Blocks:
    """Build the modern, responsive Gradio Blocks interface for NewsLens."""
    copy_kwargs = _get_copy_kwargs()

    blocks_kwargs = {"title": "NewsLens - AI News Summarizer"}
    sig = inspect.signature(gr.Blocks.__init__).parameters
    # In Gradio <6 theme and css are passed to Blocks constructor
    if "theme" in sig and "kwargs" not in sig:
        blocks_kwargs["theme"] = THEME
    if "css" in sig and "kwargs" not in sig:
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 📰 NewsLens
                ### *AI-Powered News Summarization & Model Benchmarking*
                **Condense long news stories into accurate, fluent summaries using Pre-trained Transformers and Extractive Graph algorithms.**
                """
            )

        with gr.Tabs() as tabs:
            # ==================================================================
            # Tab 1: Single Model Summarizer
            # ==================================================================
            with gr.Tab("✨ Single Summarizer", id="tab_single"):
                with gr.Row():
                    with gr.Column(scale=6):
                        article_input = gr.Textbox(
                            label="Source News Article",
                            placeholder="Paste your news article here...",
                            lines=12,
                            max_lines=25,
                        )
                        count_display = gr.Markdown("Word Count: 0 | Character Count: 0")

                        # Sample selector
                        sample_dropdown = gr.Dropdown(
                            choices=list(SAMPLE_ARTICLES.keys()),
                            label="Load Sample News Article",
                            value=None,
                        )

                        with gr.Accordion("⚙️ Generation Parameters (Abstractive Models)", open=False):
                            with gr.Row():
                                num_beams_slider = gr.Slider(
                                    minimum=1, maximum=8, value=4, step=1, label="Beam Search Width (num_beams)"
                                )
                                max_length_slider = gr.Slider(
                                    minimum=32, maximum=256, value=128, step=8, label="Maximum Length (tokens)"
                                )
                                min_length_slider = gr.Slider(
                                    minimum=10, maximum=100, value=30, step=5, label="Minimum Length (tokens)"
                                )
                            sent_count_slider = gr.Slider(
                                minimum=1, maximum=6, value=3, step=1, label="Extractive Sentences (Lead-3 / TextRank)"
                            )

                    with gr.Column(scale=4):
                        model_radio = gr.Radio(
                            choices=[
                                "BART (facebook/bart-large-cnn)",
                                "T5 (google-t5/t5-small)",
                                "TextRank (Graph-based PageRank)",
                                "Lead-3 (Inverted Pyramid Heuristic)",
                            ],
                            value="BART (facebook/bart-large-cnn)",
                            label="Summarization Architecture",
                        )

                        with gr.Row():
                            summarize_btn = gr.Button("⚡ Generate Summary", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

                        summary_output = gr.Textbox(
                            label="Generated Summary",
                            lines=7,
                            **copy_kwargs,
                        )

                        with gr.Group(elem_classes=["metric-card"]):
                            gr.Markdown("**Performance & Resource Telemetry**")
                            with gr.Row():
                                latency_stat = gr.Textbox(label="Inference Latency", interactive=False)
                                device_stat = gr.Textbox(label="Compute Device", interactive=False)
                            with gr.Row():
                                word_stat = gr.Textbox(label="Summary Length", interactive=False)
                                ratio_stat = gr.Textbox(label="Compression Ratio", interactive=False)

                # Interactions
                article_input.change(fn=update_text_counts, inputs=article_input, outputs=count_display)

                def load_sample(choice):
                    return SAMPLE_ARTICLES.get(choice, "")

                sample_dropdown.change(fn=load_sample, inputs=sample_dropdown, outputs=article_input)

                summarize_btn.click(
                    fn=summarize_article,
                    inputs=[
                        article_input,
                        model_radio,
                        sent_count_slider,
                        max_length_slider,
                        min_length_slider,
                        num_beams_slider,
                    ],
                    outputs=[summary_output, latency_stat, device_stat, word_stat, ratio_stat],
                )

                def clear_all():
                    return "", "Word Count: 0 | Character Count: 0", "", "", "", "", ""

                clear_btn.click(
                    fn=clear_all,
                    outputs=[
                        article_input,
                        count_display,
                        summary_output,
                        latency_stat,
                        device_stat,
                        word_stat,
                        ratio_stat,
                    ],
                )

            # ==================================================================
            # Tab 2: Side-by-Side Model Comparison
            # ==================================================================
            with gr.Tab("⚖️ Model Comparison", id="tab_comparison"):
                gr.Markdown(
                    """
                    Compare how **Extractive** (Lead-3, TextRank) and **Abstractive** (T5, BART) paradigms
                    summarize the exact same document side-by-side.
                    *(Models are loaded sequentially to prevent memory exhaustion).*
                    """
                )
                comp_input = gr.Textbox(
                    label="Input Document for Multi-Model Benchmark",
                    placeholder="Paste article text or select a sample from Tab 1...",
                    lines=8,
                )
                comp_btn = gr.Button("🚀 Run Side-by-Side Benchmark", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Lead-3 (Heuristic)")
                        comp_lead3 = gr.Textbox(label="Lead-3 Summary", lines=5, **copy_kwargs)
                    with gr.Column():
                        gr.Markdown("### 2. TextRank (Graph/PageRank)")
                        comp_textrank = gr.Textbox(label="TextRank Summary", lines=5, **copy_kwargs)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 3. T5-Small (Abstractive 60M)")
                        comp_t5 = gr.Textbox(label="T5 Summary", lines=5, **copy_kwargs)
                    with gr.Column():
                        gr.Markdown("### 4. BART-Large-CNN (Abstractive 406M)")
                        comp_bart = gr.Textbox(label="BART Summary", lines=5, **copy_kwargs)

                comp_table = gr.Markdown(label="Comparison Table")

                comp_btn.click(
                    fn=compare_all_models,
                    inputs=[comp_input],
                    outputs=[comp_lead3, comp_textrank, comp_t5, comp_bart, comp_table],
                )

            # ==================================================================
            # Tab 3: Academic Methodology & Benchmark
            # ==================================================================
            with gr.Tab("📚 Methodology & Academic Guide", id="tab_guide"):
                gr.Markdown(
                    """
                    ## Theoretical Framework: Extractive vs. Abstractive Summarization

                    | Dimension | Extractive Summarization (Lead-3 / TextRank) | Abstractive Summarization (BART / T5) |
                    |---|---|---|
                    | **Mechanism** | Selects verbatim sentences from source based on heuristic or graph centrality. | Generates novel phrasing and syntactic combinations using an autoregressive decoder. |
                    | **Factual Faithfulness** | Guaranteed 100% factual accuracy (zero hallucination risk). | High fluency; potential risk of factual hallucination if context is ambiguous. |
                    | **Linguistic Flow** | Sentence boundaries can feel disjointed or context-dependent. | Coherent discourse flow matching human writing conventions. |
                    | **Compute Profile** | Executed in milliseconds on standard CPU. | Requires Transformer matrix operations; accelerated via GPU FP16. |

                    ---

                    ## Official CNN/DailyMail ROUGE Benchmark (Tested on 50 Test Samples)

                    | Model | Paradigm | Parameters | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg Summary Words | Avg Latency |
                    |---|---|---|---|---|---|---|---|
                    | **Lead-3** | Extractive Heuristic | N/A | **29.91%** | **12.63%** | **21.82%** | 62.1 words | **0.002s** (CPU) |
                    | **TextRank** | Extractive Graph | N/A | **27.16%** | **8.36%** | **18.40%** | 66.6 words | **0.005s** (CPU) |
                    | **T5-Small** | Abstractive Transformer | 60M | **31.40%** | **12.23%** | **22.71%** | 42.9 words | **0.984s** (GPU) |
                    | **BART-Large-CNN** | Abstractive Transformer | 406M | **36.38%** | **16.45%** | **27.11%** | 41.9 words | **0.973s** (GPU) |

                    ---

                    ### Hardware Optimization & Deployment Notice
                    - **GPU Acceleration:** When deployed on CUDA hardware, neural inference runs in **FP16 mixed precision**, reducing memory allocation to ~775 MB–1.6 GB.
                    - **CPU Free Cloud Tier:** When hosted on free CPU cloud tiers (e.g., Hugging Face Spaces 2 vCPU), BART inference will take ~5–8 seconds per document. T5 and Extractive models remain near-instantaneous.

                    🔗 **Project GitHub Repository:** [https://github.com/abg11hani235-commits/NewsLens](https://github.com/abg11hani235-commits/NewsLens)
                    """
                )

    return demo


def launch_app(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the NewsLens Gradio application."""
    demo = create_ui()
    demo.queue()

    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "share": share,
    }
    sig = inspect.signature(gr.Blocks.launch).parameters
    if "theme" in sig:
        launch_kwargs["theme"] = THEME
    if "css" in sig:
        launch_kwargs["css"] = CUSTOM_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
