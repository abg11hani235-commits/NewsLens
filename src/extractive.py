"""Extractive Summarization Module for NLP Text Summarization Project.

Implements graph-based and heuristic extractive baselines:
1. TextRank Summarizer:
   - Splits articles into sentences using NLTK.
   - Computes pairwise sentence similarity using TF-IDF and Cosine Similarity.
   - Builds a weighted sentence connectivity graph using NetworkX.
   - Applies the PageRank algorithm to score sentence importance.
   - Selects top-ranked sentences and restores chronological narrative order.
2. Lead-3 Baseline:
   - Classical news summarization heuristic extracting the first 3 sentences.
"""

import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nltk
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TEXTRANK_NUM_SENTENCES,
    TEXTRANK_DAMPING,
    SUMMARIES_DIR,
)
from src.preprocessing import clean_text, ensure_nltk_resources

logger = logging.getLogger("extractive")

_NLTK_CHECKED = False


def _ensure_nltk_ready() -> None:
    """Lazy check to ensure NLTK resources are available without slowing down module import."""
    global _NLTK_CHECKED
    if not _NLTK_CHECKED:
        ensure_nltk_resources()
        _NLTK_CHECKED = True


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK sent_tokenize with regex fallback.

    Args:
        text (str): Cleaned input text.

    Returns:
        List[str]: List of sentence strings.
    """
    import re

    _ensure_nltk_ready()
    try:
        sentences = nltk.sent_tokenize(text)
        if sentences:
            return sentences
    except Exception as e:
        logger.warning(f"NLTK sent_tokenize failed ({e}); using regex sentence splitting.")

    # Robust regex fallback splitting by sentence ending punctuation followed by space/newline
    raw_sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw_sents if s.strip()]


def lead_n_summarize(text: str, num_sentences: int = 3) -> str:
    """Generate an extractive summary using the classical Lead-N heuristic.

    Extracts the first N sentences of the article. In journalism, articles follow
    the inverted pyramid structure where the most critical facts are placed upfront.

    Args:
        text (str): Input article text.
        num_sentences (int): Number of initial sentences to extract.

    Returns:
        str: Extracted summary string.
    """
    cleaned = clean_text(text)
    if not cleaned or not any(c.isalnum() for c in cleaned):
        return ""

    sentences = split_into_sentences(cleaned)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    selected = sentences[:num_sentences]
    return " ".join(selected)


def build_sentence_similarity_matrix(sentences: List[str]) -> np.ndarray:
    """Build pairwise cosine similarity matrix across sentences using TF-IDF representations.

    Args:
        sentences (List[str]): List of sentence strings.

    Returns:
        np.ndarray: Square similarity matrix of shape (num_sentences, num_sentences).
    """
    if len(sentences) <= 1:
        return np.ones((len(sentences), len(sentences)), dtype=float)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except ValueError:
        # Handles edge cases where all words are stop words or vocabulary is empty
        sim_matrix = np.zeros((len(sentences), len(sentences)), dtype=float)

    # Remove self-loops (diagonal = 0)
    np.fill_diagonal(sim_matrix, 0.0)
    return sim_matrix


def textrank_summarize(
    text: str,
    num_sentences: int = TEXTRANK_NUM_SENTENCES,
    damping: float = TEXTRANK_DAMPING,
    similarity_threshold: float = 0.05,
    max_candidate_sentences: Optional[int] = 50,
) -> str:
    """Generate an extractive summary using graph-based TextRank algorithm.

    Optimized for high-performance and safety:
    - Caps initial candidate sentences to prevent O(N^2) explosion on very long documents.
    - Uses vectorized cosine similarity via L2-normalized TF-IDF dot products.
    - Constructs the NetworkX graph directly from the thresholded NumPy adjacency matrix.
    - Re-sorts top-ranked sentences chronologically to preserve narrative coherence.

    Args:
        text (str): Input article text.
        num_sentences (int): Target number of sentences for summary.
        damping (float): PageRank damping factor (default 0.85).
        similarity_threshold (float): Minimum similarity weight to add an edge.
        max_candidate_sentences (Optional[int]): Maximum leading sentences to evaluate (default 50).

    Returns:
        str: Extracted summary with original sentence sequence preserved.
    """
    cleaned = clean_text(text)
    if not cleaned or not any(c.isalnum() for c in cleaned):
        return ""

    sentences = split_into_sentences(cleaned)
    n = len(sentences)

    # Edge cases
    if n == 0:
        return ""
    if n <= num_sentences:
        return " ".join(sentences)

    # Cap candidate sentences for performance on long documents
    if max_candidate_sentences is not None and n > max_candidate_sentences:
        candidate_sentences = sentences[:max_candidate_sentences]
        k = len(candidate_sentences)
    else:
        candidate_sentences = sentences
        k = n

    sim_matrix = build_sentence_similarity_matrix(candidate_sentences)

    # Filter similarity matrix by threshold to prune sparse/weak edges
    filtered_sim = np.where(sim_matrix > similarity_threshold, sim_matrix, 0.0)

    # Fallback to Lead-N if graph is completely disconnected
    if not np.any(filtered_sim):
        return lead_n_summarize(text, num_sentences=num_sentences)

    # Vectorized graph construction from adjacency matrix
    graph = nx.from_numpy_array(filtered_sim)

    try:
        scores = nx.pagerank(graph, alpha=damping, weight="weight", max_iter=200)
    except Exception as e:
        logger.warning(f"PageRank convergence failed ({e}); falling back to Lead-N.")
        return lead_n_summarize(text, num_sentences=num_sentences)

    # Rank sentence indices by PageRank score descending
    ranked_indices = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Select top N indices
    top_indices = [idx for idx, _ in ranked_indices[:num_sentences]]

    # Re-sort top indices chronologically to preserve discourse flow
    top_indices.sort()

    selected_sentences = [candidate_sentences[idx] for idx in top_indices]
    return " ".join(selected_sentences)


def generate_extractive_summaries_for_dataset(
    df: pd.DataFrame,
    sample_size: Optional[int] = 50,
    save_output: bool = True,
) -> pd.DataFrame:
    """Generate Lead-3 and TextRank summaries on a dataset sample.

    Args:
        df (pd.DataFrame): DataFrame containing 'id', 'article', and 'highlights'.
        sample_size (Optional[int]): Number of records to evaluate (None for all).
        save_output (bool): Whether to persist results to outputs/summaries/.

    Returns:
        pd.DataFrame: DataFrame containing original text, reference highlights,
                      and both extractive summaries.
    """
    if sample_size is not None and len(df) > sample_size:
        eval_df = df.iloc[:sample_size].copy()
    else:
        eval_df = df.copy()

    logger.info(f"Generating extractive summaries for {len(eval_df)} records...")

    eval_df["lead3_summary"] = eval_df["article"].apply(lambda t: lead_n_summarize(t, 3))
    eval_df["textrank_summary"] = eval_df["article"].apply(lambda t: textrank_summarize(t, 3))

    if save_output:
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = SUMMARIES_DIR / "extractive_summaries.csv"
        eval_df.to_csv(out_csv, index=False, encoding="utf-8")
        logger.info(f"Saved extractive summaries to {out_csv}")

    return eval_df


if __name__ == "__main__":
    from src.preprocessing import load_processed_split

    print("=" * 70)
    print("Extractive Summarization Benchmark (Lead-3 vs. TextRank)")
    print("=" * 70)

    test_df = load_processed_split("test")
    sample_article = test_df.iloc[0]["article"]
    sample_highlights = test_df.iloc[0]["highlights"]

    print("\n[Human Reference Summary (Highlights)]:")
    print(sample_highlights)

    print("\n[Lead-3 Baseline Summary]:")
    lead3_res = lead_n_summarize(sample_article, num_sentences=3)
    print(lead3_res)

    print("\n[TextRank Summary]:")
    textrank_res = textrank_summarize(sample_article, num_sentences=3)
    print(textrank_res)

    print("\n" + "=" * 70)
    print("Batch Generating Extractive Summaries on 50 Test Samples")
    print("=" * 70)
    results_df = generate_extractive_summaries_for_dataset(test_df, sample_size=50)
    print(f"Generated {len(results_df)} extractive summaries saved to outputs/summaries/extractive_summaries.csv")
