"""Tests for abstractive summarization module.

Contains both lightweight unit tests and real model inference verification tests.
"""

from unittest.mock import patch, MagicMock
import pytest
import torch
from src.summarizer import get_summarizer, AbstractiveSummarizer


# ==============================================================================
# Unit Tests (Initialization & Input Preprocessing)
# ==============================================================================

def test_get_summarizer_invalid_model():
    """Verify that unsupported model architectures raise ValueError."""
    with pytest.raises(ValueError):
        get_summarizer("unsupported_model_architecture")


def test_prepare_input_text_t5():
    """Verify that T5 prepends the required 'summarize: ' task prefix."""
    summarizer = get_summarizer("t5")
    prepared = summarizer._prepare_input_text("This is an article about artificial intelligence.")
    assert prepared.startswith("summarize: This is an article")


def test_summarize_empty_text():
    """Verify graceful handling of empty string input."""
    summarizer = get_summarizer("t5")
    res = summarizer.summarize_text("")
    assert res["summary"] == ""
    assert res["latency_sec"] == 0.0
    assert res["input_tokens"] == 0


def test_get_summarizer_presets():
    """Verify generation presets ('fast' vs 'quality')."""
    fast_t5 = get_summarizer("t5", preset="fast")
    assert fast_t5.generation_params.get("num_beams") == 1

    quality_t5 = get_summarizer("t5", preset="quality")
    assert quality_t5.generation_params.get("num_beams") == 4


# ==============================================================================
# Real Model Inference Tests (Executed on Cached Models)
# ==============================================================================

def test_summarize_real_t5():
    """Verify real sequence-to-sequence inference with google-t5/t5-small."""
    summarizer = get_summarizer("t5")
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Scientists discovered that foxes and dogs can coexist peacefully in the forest. "
        "Further research indicates that animal cooperation is more common than previously assumed."
    )
    res = summarizer.summarize_text(text, max_length=50, min_length=10)
    assert isinstance(res["summary"], str)
    assert len(res["summary"]) > 0
    assert res["input_tokens"] > 0
    assert res["output_tokens"] > 0
    assert res["latency_sec"] > 0


def test_summarize_batch_true_batching():
    """Verify true vectorized batch inference, order preservation, and empty handling."""
    summarizer = get_summarizer("t5")
    articles = [
        "First article discusses renewable energy expansion across western European nations in great detail.",
        "",  # Empty article should be safely handled
        "Third article analyzes quantum computing breakthroughs and hardware cooling systems.",
    ]
    batch_res = summarizer.summarize_batch(articles, batch_size=2, max_length=40, min_length=5)
    assert len(batch_res) == 3
    assert len(batch_res[0]["summary"]) > 0
    assert batch_res[0]["preprocessing_latency_sec"] > 0.0
    assert batch_res[0]["batch_tokenization_latency_sec"] > 0.0
    assert batch_res[0]["batch_generation_latency_sec"] > 0.0
    assert batch_res[0]["amortized_batch_latency_sec"] > 0.0
    assert batch_res[0]["chunk_elapsed_sec"] > 0.0
    assert batch_res[0]["is_amortized_batch"] is True
    assert batch_res[1]["summary"] == ""
    assert batch_res[1]["latency_sec"] == 0.0
    assert batch_res[1]["preprocessing_latency_sec"] == 0.0
    assert batch_res[1]["is_amortized_batch"] is False
    assert len(batch_res[2]["summary"]) > 0
    assert batch_res[2]["preprocessing_latency_sec"] > 0.0


def test_summarize_batch_granular_metrics():
    """Verify that summarize_batch reports distinct per-document preprocessing and amortized batch metrics."""
    summarizer = get_summarizer("t5")
    texts = [
        "Clean solar energy technology reduces carbon dioxide emissions across cities.",
        "Artificial intelligence revolutionizes medical imaging diagnostic accuracy.",
    ]
    batch_res = summarizer.summarize_batch(texts, batch_size=2, max_length=30, min_length=5)
    assert len(batch_res) == 2
    for item in batch_res:
        assert item["preprocessing_latency_sec"] > 0.0
        assert item["batch_tokenization_latency_sec"] > 0.0
        assert item["batch_generation_latency_sec"] > 0.0
        assert item["amortized_batch_latency_sec"] > 0.0
        assert item["chunk_elapsed_sec"] > 0.0
        assert item["is_amortized_batch"] is True


def test_summarize_warmup_and_latency_tracking():
    """Verify model warmup and cold vs warm latency tracking."""
    summarizer = get_summarizer("t5")
    warmup_lat = summarizer.warmup()
    assert warmup_lat > 0.0
    assert summarizer.cold_latency_sec is not None

    sample = "Autonomous systems and robotics are transforming manufacturing logistics globally."
    res = summarizer.summarize_text(sample, max_length=40, min_length=5)
    assert res["is_cold"] is False
    assert len(summarizer.warm_latencies) >= 1


def test_summarize_very_long_input():
    """Verify that very long articles are safely truncated without error."""
    summarizer = get_summarizer("t5")
    long_article = "Technological innovation accelerates development. " * 300  # ~900 words
    res = summarizer.summarize_text(long_article, max_length=50, min_length=10)
    assert isinstance(res["summary"], str)
    assert len(res["summary"]) > 0
    assert res["input_tokens"] <= summarizer.max_input_length


def test_summarize_real_bart():
    """Verify real sequence-to-sequence inference with cached facebook/bart-large-cnn."""
    summarizer = get_summarizer("bart")
    text = (
        "The European Union reached an agreement today on artificial intelligence legislation. "
        "Officials in Brussels stated the framework establishes strict guidelines for high-risk systems. "
        "Industry stakeholders expressed cautious optimism while requesting clear compliance standards."
    )
    res = summarizer.summarize_text(text, max_length=60, min_length=15, num_beams=2)
    assert isinstance(res["summary"], str)
    assert len(res["summary"]) > 0
    assert "artificial intelligence" in res["summary"].lower() or "european union" in res["summary"].lower()
    assert res["input_tokens"] > 0
    assert res["output_tokens"] > 0
    assert res["latency_sec"] > 0
    if torch.cuda.is_available():
        assert "cuda" in res["device"].lower()

    # Cleanly unload BART after test
    summarizer.unload_model()


# ==============================================================================
# Edge Cases & Robustness (Batch Validation, Granular Metrics, OOM Recovery)
# ==============================================================================

def test_summarize_batch_invalid_batch_size():
    """Verify that batch_size < 1 raises ValueError."""
    summarizer = get_summarizer("t5")
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        summarizer.summarize_batch(["Some text"], batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        summarizer.summarize_batch(["Some text"], batch_size=-1)


def test_summarize_granular_latency_metrics():
    """Verify that summarize_text returns all granular stage latencies."""
    summarizer = get_summarizer("t5")
    res = summarizer.summarize_text("A simple test sentence to measure pipeline stages.")
    assert "generation_latency_sec" in res
    assert "preprocessing_latency_sec" in res
    assert "tokenization_latency_sec" in res
    assert "decoding_latency_sec" in res
    assert "total_e2e_latency_sec" in res
    assert res["total_e2e_latency_sec"] >= res["generation_latency_sec"]


def test_cuda_oom_fp32_cpu_fallback_and_restore():
    """Verify that when CUDA OOM occurs, the model moves to CPU in FP32 and restores to original device."""
    summarizer = get_summarizer("t5")
    call_count = 0
    to_calls = []

    real_generate = summarizer.model.generate
    real_to = summarizer.model.to

    def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test")
        return real_generate(*args, **kwargs)

    def mock_to(*args, **kwargs):
        to_calls.append((args, kwargs))
        return real_to(*args, **kwargs)

    with patch.object(summarizer.model, "generate", side_effect=mock_generate), \
         patch.object(summarizer.model, "to", side_effect=mock_to):
        res = summarizer.summarize_text("Test article for CUDA OOM fallback handling.")
        assert "summary" in res
        assert len(res["summary"]) > 0
        assert "oom-fallback-fp32" in res["device"]
        cpu_calls = [c for c in to_calls if c[0] == ("cpu",) and c[1].get("dtype") == torch.float32]
        assert len(cpu_calls) >= 1, "Expected model.to('cpu', dtype=torch.float32) call!"


def test_cuda_oom_restore_failure_fallback_persistent():
    """Verify that if restoration back to CUDA fails, model remains safely and persistently on CPU."""
    summarizer = get_summarizer("t5")
    call_count = 0

    real_generate = summarizer.model.generate
    real_to = summarizer.model.to

    def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test")
        return real_generate(*args, **kwargs)

    def mock_to(device, *args, **kwargs):
        if str(device) != "cpu":
            raise RuntimeError("CUDA out of memory during GPU restoration attempt")
        return real_to(device, *args, **kwargs)

    with patch.object(summarizer.model, "generate", side_effect=mock_generate), \
         patch.object(summarizer.model, "to", side_effect=mock_to):
        res = summarizer.summarize_text("Test article for persistent CPU fallback.")
        assert "summary" in res
        assert "persistent" in res["device"]
        assert summarizer.device == torch.device("cpu")
        assert summarizer.is_cuda is False


def test_cuda_oom_releases_gpu_inputs_before_restore():
    """Verify that GPU inputs are released and empty_cache called before attempting model restore."""
    summarizer = get_summarizer("t5")
    call_count = 0
    empty_cache_called_count = 0

    real_generate = summarizer.model.generate
    real_empty_cache = torch.cuda.empty_cache

    def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test")
        return real_generate(*args, **kwargs)

    def mock_empty_cache():
        nonlocal empty_cache_called_count
        empty_cache_called_count += 1
        return real_empty_cache()

    with patch.object(summarizer.model, "generate", side_effect=mock_generate), \
         patch("torch.cuda.empty_cache", side_effect=mock_empty_cache):
        res = summarizer.summarize_text("Sample article to verify empty_cache calls during OOM.")
        assert "summary" in res
        assert empty_cache_called_count >= 1


