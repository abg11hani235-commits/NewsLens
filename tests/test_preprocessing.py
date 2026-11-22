"""Unit tests for text preprocessing module."""

import pytest
import pandas as pd
from src.preprocessing import (
    clean_text,
    clean_highlights,
    get_text_length_info,
    preprocess_dataset,
    load_processed_split,
)


def test_clean_text_removes_cnn_artifacts():
    raw = "(CNN) -- The president addressed the nation today. Details followed."
    cleaned = clean_text(raw)
    assert not cleaned.startswith("(CNN)")
    assert cleaned.startswith("The president")
    assert "Details followed." in cleaned


def test_clean_text_whitespace_normalization():
    raw = "   Multiple   spaces   and \n\n newlines  here .   "
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert cleaned.endswith("here.")


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_clean_highlights():
    raw = "First bullet point .\n Second bullet point without period\nThird point ."
    cleaned = clean_highlights(raw)
    assert "First bullet point." in cleaned
    assert "Second bullet point without period." in cleaned
    assert "Third point." in cleaned


def test_get_text_length_info():
    text = "First sentence here. Second sentence follows! Third one?"
    info = get_text_length_info(text)
    assert info["char_count"] == len(text)
    assert info["word_count"] == 8
    assert info["sentence_count"] == 3


def test_preprocess_dataset():
    df = pd.DataFrame({
        "id": ["1", "2"],
        "article": ["(CNN) -- Valid article here.", "   "],
        "highlights": ["Valid summary.", "Empty article row"],
    })
    processed = preprocess_dataset(df)
    # The empty article row should be filtered out
    assert len(processed) == 1
    assert "article_word_count" in processed.columns
    assert "summary_word_count" in processed.columns
    assert not processed.iloc[0]["article"].startswith("(CNN)")


def test_load_processed_split():
    df = load_processed_split("test")
    assert len(df) == 200
    assert "article" in df.columns
    assert "highlights" in df.columns
    assert "article_word_count" in df.columns
