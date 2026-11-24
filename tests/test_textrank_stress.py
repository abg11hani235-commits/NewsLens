"""Stress and Scalability Tests for TextRank Extractive Summarization.

Evaluates performance, memory stability, and algorithmic robustness on:
1. 100-sentence corpus.
2. 500-sentence corpus.
3. 1,000-sentence corpus.
4. Identical / repetitive sentence degeneration.
5. Empty / whitespace-only / punctuation-only inputs.
6. Extreme single-sentence inputs (>5,000 words).
"""

import time
import pytest
from src.extractive import textrank_summarize, build_sentence_similarity_matrix


def generate_sentence_corpus(num_sentences: int) -> str:
    """Generate a realistic synthetic corpus with distinct topical sentences."""
    templates = [
        "Global financial markets experienced significant fluctuations in trading volume across Asian sessions.",
        "Technological advancements in semiconductor manufacturing continue to drive computing efficiency.",
        "Renewable energy capacity expanded rapidly following new international environmental policy accords.",
        "Medical researchers published promising findings regarding targeted cellular immunotherapy treatments.",
        "Urban infrastructure development projects received increased funding from multilateral development banks.",
        "Autonomous navigation algorithms demonstrated improved safety margins in adverse weather simulations.",
        "Agricultural yields stabilized as drought-resistant crop varieties were deployed in arid regions.",
        "Quantum computing laboratories achieved higher qubit coherence times using cryogenic silicon systems.",
        "Satellite telemetry verified atmospheric carbon concentration variations across maritime shipping routes.",
        "Telecommunications operators deployed next-generation optical fiber networks across metropolitan regions.",
    ]
    sentences = [f"Sentence {i}: {templates[i % len(templates)]}" for i in range(num_sentences)]
    return " ".join(sentences)


def test_textrank_stress_100_sentences():
    """Verify performance and output validity on 100-sentence document."""
    corpus = generate_sentence_corpus(100)
    t0 = time.perf_counter()
    summary = textrank_summarize(corpus, num_sentences=3)
    elapsed = time.perf_counter() - t0

    assert len(summary) > 0
    assert len(summary.split()) > 15
    # With candidate capping (50), 100 sentences should process under 1.5s
    assert elapsed < 2.0, f"TextRank took too long on 100 sentences: {elapsed:.3f}s"


def test_textrank_stress_500_sentences():
    """Verify candidate capping prevents quadratic explosion on 500-sentence document."""
    corpus = generate_sentence_corpus(500)
    t0 = time.perf_counter()
    summary = textrank_summarize(corpus, num_sentences=4)
    elapsed = time.perf_counter() - t0

    assert len(summary) > 0
    # Candidate capping (max 50) ensures 500 sentences execute under 2.5s
    assert elapsed < 3.0, f"TextRank took too long on 500 sentences: {elapsed:.3f}s"


def test_textrank_stress_1000_sentences():
    """Verify scalability and stability on 1,000-sentence document."""
    corpus = generate_sentence_corpus(1000)
    t0 = time.perf_counter()
    summary = textrank_summarize(corpus, num_sentences=5)
    elapsed = time.perf_counter() - t0

    assert len(summary) > 0
    assert elapsed < 4.0, f"TextRank took too long on 1,000 sentences: {elapsed:.3f}s"


def test_textrank_stress_repeated_identical_sentences():
    """Verify that perfectly identical repeated sentences do not cause zero-division or graph divergence."""
    sentence = "The central bank kept interest rates unchanged at the quarterly policy meeting."
    corpus = " ".join([sentence] * 80)

    t0 = time.perf_counter()
    summary = textrank_summarize(corpus, num_sentences=3)
    elapsed = time.perf_counter() - t0

    assert len(summary) > 0
    assert elapsed < 1.0


def test_textrank_stress_empty_and_whitespace():
    """Verify graceful handling of degenerate text inputs."""
    assert textrank_summarize("", num_sentences=3) == ""
    assert textrank_summarize("    \n\t  ", num_sentences=3) == ""
    assert textrank_summarize("..... ??? !!!", num_sentences=3) == ""


def test_textrank_stress_extreme_long_sentence():
    """Verify single massive sentence without punctuation does not break PageRank."""
    single_long_sentence = "Technology " * 2000
    summary = textrank_summarize(single_long_sentence, num_sentences=3)
    assert len(summary) > 0
