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
    assert batch_res[1]["summary"] == ""
    assert batch_res[1]["latency_sec"] == 0.0
    assert len(batch_res[2]["summary"]) > 0


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
