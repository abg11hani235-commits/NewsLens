"""Unit and integration tests for NewsLens Gradio Web Application (app.py).

Verifies UI creation, input validation, extractive summarization handlers,
performance telemetry, and ModelManager eviction logic.
"""

import pytest
import gradio as gr
from app import (
    create_ui,
    update_text_counts,
    summarize_article,
    compare_all_models,
    ModelManager,
    SAMPLE_ARTICLES,
)


class TestAppInterface:
    """Tests for Gradio interface construction and static components."""

    def test_create_ui_returns_blocks(self):
        """Verify that create_ui initializes and returns a valid gr.Blocks instance."""
        demo = create_ui()
        assert isinstance(demo, gr.Blocks)
        assert demo.title == "NewsLens - AI News Summarizer"

    def test_sample_articles_non_empty(self):
        """Ensure sample news articles are provided and contain substantial text."""
        assert len(SAMPLE_ARTICLES) >= 3
        for title, text in SAMPLE_ARTICLES.items():
            assert isinstance(title, str) and len(title) > 0
            assert isinstance(text, str) and len(text.split()) >= 30


class TestAppHelpers:
    """Tests for helper functions like text count calculation."""

    def test_update_text_counts_empty(self):
        """Verify count display for empty or None text."""
        assert update_text_counts("") == "Word Count: 0 | Character Count: 0"
        assert update_text_counts(None) == "Word Count: 0 | Character Count: 0"

    def test_update_text_counts_valid(self):
        """Verify accurate calculation of word and character counts."""
        sample = "The quick brown fox jumps over the lazy dog."
        result = update_text_counts(sample)
        assert "Word Count: 9" in result
        assert f"Character Count: {len(sample)}" in result


class TestSummarizeArticle:
    """Tests for single article summarization in app.py."""

    def test_summarize_empty_input_raises_error(self):
        """Ensure empty or whitespace-only input raises gr.Error."""
        with pytest.raises(gr.Error, match="Please provide an article"):
            summarize_article("", "Lead-3")

        with pytest.raises(gr.Error, match="Please provide an article"):
            summarize_article("   \n\t  ", "Lead-3")

    def test_summarize_too_short_input_raises_error(self):
        """Ensure text shorter than 10 words is rejected gracefully."""
        short_text = "This is too short."
        with pytest.raises(gr.Error, match="Input text is too short"):
            summarize_article(short_text, "Lead-3")

    def test_summarize_invalid_model_choice(self):
        """Ensure unknown model choice raises gr.Error."""
        valid_text = (
            "This is a sufficiently long news article text that exceeds the minimum required "
            "ten words for processing in the NewsLens summarization web application."
        )
        with pytest.raises(gr.Error, match="Unsupported model"):
            summarize_article(valid_text, "NonExistentModel")

    def test_summarize_lead3_success(self):
        """Verify Lead-3 execution and telemetry output format."""
        sample_text = (
            "Sentence one discusses government policy in detail. "
            "Sentence two explains public reaction and economic impact. "
            "Sentence three presents foreign ministry statements. "
            "Sentence four summarizes ongoing negotiations and future talks."
        )
        summary, latency, device, word_info, ratio_info = summarize_article(
            sample_text,
            "Lead-3 (Inverted Pyramid Heuristic)",
            num_sentences=2,
        )

        assert isinstance(summary, str) and len(summary) > 0
        assert "Sentence one" in summary
        assert "Sentence two" in summary
        assert "seconds" in latency
        assert "CPU" in device
        assert "words" in word_info
        assert "x Compression" in ratio_info

    def test_summarize_textrank_success(self):
        """Verify TextRank execution and telemetry output format."""
        sample_text = (
            "Technology stocks gained momentum following central bank announcements. "
            "Analysts noted increased capital spending in artificial intelligence research. "
            "Investors anticipated higher quarterly earnings across the semiconductor sector. "
            "The market closed with significant gains in major technology indices."
        )
        summary, latency, device, word_info, ratio_info = summarize_article(
            sample_text,
            "TextRank (Graph-based PageRank)",
            num_sentences=2,
        )

        assert isinstance(summary, str) and len(summary) > 0
        assert "seconds" in latency
        assert "CPU" in device
        assert "words" in word_info
        assert "x Compression" in ratio_info


class TestModelManager:
    """Tests for ModelManager lazy loading and memory management."""

    def test_invalid_model_key_raises_value_error(self):
        """Ensure requesting an invalid model raises ValueError."""
        mgr = ModelManager()
        with pytest.raises(ValueError, match="Invalid model choice"):
            mgr.get_model("invalid_model_name")

    def test_unload_all_safe_when_empty(self):
        """Verify unload_all runs cleanly even when no model is active."""
        mgr = ModelManager()
        assert mgr.active_summarizer is None
        mgr.unload_all()
        assert mgr.active_summarizer is None

    def test_model_manager_caching_and_metadata(self):
        """Verify model caching metadata tracks cold vs cached access."""
        mgr = ModelManager(max_cached_models=1)
        model1, meta1 = mgr.get_model_with_metadata("t5")
        assert meta1["is_cached"] is False
        assert meta1["load_time_sec"] >= 0.0
        assert meta1["model_key"] == "t5"

        # Second access must be cached with 0.0s load time
        model2, meta2 = mgr.get_model_with_metadata("t5")
        assert meta2["is_cached"] is True
        assert meta2["load_time_sec"] == 0.0
        assert model1 is model2

        mgr.unload_all()
        assert mgr.active_summarizer is None

    def test_model_manager_thread_safety(self):
        """Verify thread-safe concurrent model acquisition."""
        import concurrent.futures

        mgr = ModelManager(max_cached_models=1)

        def worker():
            model = mgr.get_model("t5")
            return id(model)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(6)]
            results = [f.result() for f in futures]

        # All concurrent threads must receive the same valid model instance
        assert len(set(results)) == 1

        mgr.unload_all()


class TestCompareModels:
    """Tests for multi-model side-by-side comparison."""

    def test_compare_models_empty_rejected(self):
        """Verify empty input is rejected with gr.Error."""
        with pytest.raises(gr.Error, match="Please provide an article"):
            compare_all_models("")

    def test_compare_models_short_rejected(self):
        """Verify input with fewer than 10 words is rejected."""
        with pytest.raises(gr.Error, match="Article text is too short"):
            compare_all_models("Too short text.")
