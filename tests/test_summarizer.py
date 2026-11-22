"""Tests for abstractive summarization module.

Contains both lightweight unit tests and real model inference verification tests.
"""

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
