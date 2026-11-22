"""Unit tests for data acquisition and loading."""

import pytest
import pandas as pd
from src.data_loader import load_raw_split, compute_length_statistics


def test_load_raw_split_test():
    """Verify loading the local raw test sample."""
    df = load_raw_split("test")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "id" in df.columns
    assert "article" in df.columns
    assert "highlights" in df.columns
    assert df["article"].isnull().sum() == 0
    assert df["highlights"].isnull().sum() == 0


def test_load_raw_split_invalid():
    """Verify error raised on invalid split name."""
    with pytest.raises(ValueError):
        load_raw_split("unknown_split")


def test_compute_length_statistics():
    """Verify statistics computation on a small test dataframe."""
    sample_df = pd.DataFrame({
        "id": ["1", "2"],
        "article": ["This is a test article with eight words here.", "Another longer article for testing."],
        "highlights": ["Test summary.", "Another summary."],
    })
    stats = compute_length_statistics(sample_df)
    assert stats["total_records"] == 2
    assert "article_words" in stats
    assert "summary_words" in stats
    assert "compression_ratio" in stats
    assert stats["article_words"]["min"] > 0
    assert stats["summary_words"]["min"] > 0
