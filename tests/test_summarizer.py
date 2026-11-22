"""Unit tests for abstractive summarization module."""

import pytest
from src.summarizer import get_summarizer, AbstractiveSummarizer


def test_get_summarizer_invalid_model():
    with pytest.raises(ValueError):
        get_summarizer("unsupported_model_architecture")


def test_prepare_input_text_t5():
    # Test that T5 prepends "summarize: "
    summarizer = get_summarizer("t5")
    prepared = summarizer._prepare_input_text("This is an article about artificial intelligence.")
    assert prepared.startswith("summarize: This is an article")


def test_summarize_empty_text():
    summarizer = get_summarizer("t5")
    res = summarizer.summarize_text("")
    assert res["summary"] == ""
    assert res["latency_sec"] == 0.0
    assert res["input_tokens"] == 0


def test_summarize_short_text():
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
