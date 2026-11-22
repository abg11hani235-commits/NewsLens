"""Unit tests for EDA visualization functions."""

import pytest
import pandas as pd
from pathlib import Path
from src.visualization import (
    plot_article_length_distribution,
    plot_summary_length_distribution,
    plot_article_vs_summary_length,
    plot_compression_ratio_distribution,
)


@pytest.fixture
def mock_df():
    """Create a mock DataFrame for plotting tests."""
    return pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "article": [
            "This is a short sample article for testing plots.",
            "Another slightly longer test article with more words for the distribution plot test.",
            "Third article for testing visualization functionality in unit tests.",
            "Fourth article to have sufficient points for regression and correlation computation.",
        ],
        "highlights": [
            "Short summary one.",
            "Second summary test.",
            "Third summary test here.",
            "Fourth summary test sample.",
        ],
    })


def test_plot_article_length_distribution(mock_df, tmp_path):
    out_file = tmp_path / "test_art_len.png"
    result = plot_article_length_distribution(mock_df, save_path=out_file, split_name="Mock")
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_summary_length_distribution(mock_df, tmp_path):
    out_file = tmp_path / "test_sum_len.png"
    result = plot_summary_length_distribution(mock_df, save_path=out_file, split_name="Mock")
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_article_vs_summary_length(mock_df, tmp_path):
    out_file = tmp_path / "test_scatter.png"
    result = plot_article_vs_summary_length(mock_df, save_path=out_file, split_name="Mock")
    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_compression_ratio_distribution(mock_df, tmp_path):
    out_file = tmp_path / "test_ratio.png"
    result = plot_compression_ratio_distribution(mock_df, save_path=out_file, split_name="Mock")
    assert result.exists()
    assert result.stat().st_size > 1000
