"""Unit tests for extractive summarization module."""

import pytest
import numpy as np
from src.extractive import (
    lead_n_summarize,
    build_sentence_similarity_matrix,
    textrank_summarize,
)


SAMPLE_ARTICLE = (
    "Machine learning is a field of artificial intelligence. "
    "It uses data and algorithms to imitate the way that humans learn. "
    "Deep learning is a subset of machine learning based on neural networks. "
    "Natural language processing helps computers understand human text. "
    "Summarization algorithms condense long documents into concise versions."
)


def test_lead_n_summarize_normal():
    summary = lead_n_summarize(SAMPLE_ARTICLE, num_sentences=2)
    assert summary.startswith("Machine learning is a field")
    assert "It uses data and algorithms" in summary
    assert "Summarization algorithms" not in summary


def test_lead_n_summarize_fewer_sentences():
    text = "Only one sentence here."
    summary = lead_n_summarize(text, num_sentences=3)
    assert summary == text


def test_lead_n_summarize_empty():
    assert lead_n_summarize("") == ""
    assert lead_n_summarize("   ") == ""


def test_build_sentence_similarity_matrix():
    sentences = [
        "Artificial intelligence and deep learning models.",
        "Deep learning neural networks are part of artificial intelligence.",
        "Completely unrelated topic about cooking recipes.",
    ]
    sim = build_sentence_similarity_matrix(sentences)
    assert sim.shape == (3, 3)
    assert np.all(np.diag(sim) == 0.0)  # Diagonal must be 0
    # Sentences 0 and 1 share AI/deep learning words, so sim[0, 1] > sim[0, 2]
    assert sim[0, 1] > sim[0, 2]


def test_textrank_summarize_normal():
    summary = textrank_summarize(SAMPLE_ARTICLE, num_sentences=2)
    assert len(summary) > 0
    assert summary.count(".") >= 2


def test_textrank_summarize_empty():
    assert textrank_summarize("") == ""
    assert textrank_summarize(None) == ""


def test_textrank_preserves_sentence_order():
    summary = textrank_summarize(SAMPLE_ARTICLE, num_sentences=3)
    # Check that sentences appearing in summary preserve their relative original order
    original_sentences = [s.strip() for s in SAMPLE_ARTICLE.split(".") if s.strip()]
    extracted = [s.strip() for s in summary.split(".") if s.strip()]

    indices = [original_sentences.index(s) for s in extracted if s in original_sentences]
    assert indices == sorted(indices)
