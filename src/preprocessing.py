"""Text Preprocessing Module for Text Summarization NLP Project.

Provides modular functions for:
- Cleaning raw news text while preserving semantic and grammatical structure.
- Cleaning and formatting human reference summary highlights.
- Measuring linguistic text properties (character, word, and sentence counts).
- Model-specific tokenization and truncation for BART and T5.
- Batch dataset preprocessing and saving clean splits to data/processed/.
"""

import sys
import re
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import nltk
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from config import (
    RAW_TRAIN_FILE,
    RAW_VAL_FILE,
    RAW_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    PROCESSED_VAL_FILE,
    PROCESSED_TEST_FILE,
    BART_MAX_INPUT_LENGTH,
    T5_MAX_INPUT_LENGTH,
)

# Ensure NLTK punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

logger = logging.getLogger("preprocessing")


def clean_text(text: Optional[str]) -> str:
    """Clean raw news article text preserving linguistic meaning and casing.

    Removes news organization lead-ins (e.g., '(CNN) --'), normalizes whitespace,
    and strips extraneous control characters without damaging punctuation or casing.

    Args:
        text (Optional[str]): Raw input text.

    Returns:
        str: Cleaned text.
    """
    if text is None or not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    # Remove CNN/DailyMail specific wire lead-ins:
    # Examples: "(CNN) --", "(CNN)", "LONDON, England (CNN) --", "NEW YORK (CNN) --"
    text = re.sub(r"^(\s*[A-Z\s,]+)?\([Cc][Nn][Nn]\)\s*(--|—|-)?\s*", "", text)
    text = re.sub(r"^(\s*[A-Z\s,]+)?\([Dd]aily\s*[Mm]ail\)\s*(--|—|-)?\s*", "", text)

    # Remove stray "UPDATED: [timestamp]" headers occasionally found in articles
    text = re.sub(r"^UPDATED:\s*[\d:APM\s,]+\b", "", text, flags=re.IGNORECASE)

    # Normalize multiple newlines and carriage returns
    text = re.sub(r"[\r\n]+", "\n", text)

    # Normalize multiple spaces, tabs, and non-breaking spaces
    text = re.sub(r"[ \t\u00a0\u200b]+", " ", text)

    # Fix spacing before punctuation (e.g., "word ." -> "word.")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Fix missing space after punctuation if followed immediately by capitalized letter
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    return text.strip()


def clean_highlights(highlights: Optional[str]) -> str:
    """Clean and normalize human reference summaries (bullet points).

    Converts disjoint bullet points into clean, coherent sentences.

    Args:
        highlights (Optional[str]): Raw summary string.

    Returns:
        str: Normalized summary string.
    """
    if highlights is None or not isinstance(highlights, str):
        return ""

    highlights = highlights.strip()
    if not highlights:
        return ""

    # Split lines, clean each bullet line
    lines = [line.strip() for line in highlights.split("\n") if line.strip()]
    cleaned_lines = []

    for line in lines:
        # Fix spacing before punctuation (common artifact in CNN/DailyMail: "word .")
        clean_line = re.sub(r"\s+([.,;:!?])", r"\1", line)
        clean_line = re.sub(r"[ \t]+", " ", clean_line)

        # Ensure line ends with punctuation
        if clean_line and clean_line[-1] not in ".!?":
            clean_line += " ."
            clean_line = re.sub(r"\s+\.", ".", clean_line)

        if clean_line:
            cleaned_lines.append(clean_line)

    # Join lines with space
    joined = " ".join(cleaned_lines)
    # Normalize duplicate spaces
    joined = re.sub(r"\s+", " ", joined)
    return joined.strip()


def get_text_length_info(text: str) -> Dict[str, int]:
    """Calculate character, word, and sentence counts for a piece of text.

    Args:
        text (str): Input text.

    Returns:
        Dict[str, int]: Counts dictionary with keys 'char_count', 'word_count', 'sentence_count'.
    """
    if not text:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0}

    chars = len(text)
    words = len(text.split())
    try:
        sentences = len(nltk.sent_tokenize(text))
    except Exception:
        # Fallback simple sentence split if NLTK fails
        sentences = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))

    return {
        "char_count": chars,
        "word_count": words,
        "sentence_count": sentences,
    }


def tokenize_for_model(
    text: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: Optional[int] = None,
    padding: Union[bool, str] = False,
    truncation: bool = True,
    return_tensors: Optional[str] = "pt",
    prefix_t5: bool = True,
) -> Dict[str, Any]:
    """Tokenize and format input text for transformer encoder-decoder models.

    Handles model-specific input requirements (such as T5's 'summarize: ' task prefix)
    and produces PyTorch tensors with attention masks.

    Args:
        text (str): Input text to tokenize.
        tokenizer: Hugging Face tokenizer instance.
        max_length (Optional[int]): Maximum token limit (defaults to tokenizer.model_max_length).
        padding (Union[bool, str]): Padding strategy ('max_length', True, or False).
        truncation (bool): Whether to truncate sequences exceeding max_length.
        return_tensors (Optional[str]): Return format ('pt' for PyTorch, 'np' for NumPy, None for list).
        prefix_t5 (bool): Prepend 'summarize: ' if tokenizer belongs to a T5 model.

    Returns:
        Dict[str, Any]: Tokenizer output containing 'input_ids' and 'attention_mask'.
    """
    cleaned = clean_text(text)

    # Prepend T5 task prefix if applicable
    is_t5 = "t5" in getattr(tokenizer, "name_or_path", "").lower()
    if is_t5 and prefix_t5 and not cleaned.lower().startswith("summarize:"):
        cleaned = f"summarize: {cleaned}"

    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", 1024)
        if max_length > 10000:  # Fix for tokenizers reporting default unbounded max_length
            max_length = 1024

    encoded = tokenizer(
        cleaned,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return encoded


def preprocess_dataset(
    df: pd.DataFrame,
    clean_articles: bool = True,
    clean_reference_highlights: bool = True,
) -> pd.DataFrame:
    """Preprocess a DataFrame containing raw CNN/DailyMail articles and highlights.

    Args:
        df (pd.DataFrame): Input DataFrame with 'article' and 'highlights'.
        clean_articles (bool): Whether to clean the article text.
        clean_reference_highlights (bool): Whether to clean the highlights text.

    Returns:
        pd.DataFrame: Cleaned DataFrame with added word count columns.
    """
    processed_df = df.copy()

    if clean_articles:
        processed_df["article"] = processed_df["article"].apply(clean_text)

    if clean_reference_highlights:
        processed_df["highlights"] = processed_df["highlights"].apply(clean_highlights)

    # Drop any records that became empty after cleaning
    initial_len = len(processed_df)
    processed_df = processed_df[
        (processed_df["article"].str.len() > 0) & (processed_df["highlights"].str.len() > 0)
    ].reset_index(drop=True)

    dropped = initial_len - len(processed_df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} empty records during dataset preprocessing.")

    # Add descriptive length columns
    processed_df["article_word_count"] = processed_df["article"].apply(lambda x: len(x.split()))
    processed_df["summary_word_count"] = processed_df["highlights"].apply(lambda x: len(x.split()))

    return processed_df


def preprocess_and_save_all_splits(force: bool = False) -> Dict[str, pd.DataFrame]:
    """Clean all raw splits and save the results to data/processed/.

    Args:
        force (bool): Re-run preprocessing even if processed files exist.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to cleaned DataFrames.
    """
    from src.data_loader import load_raw_split

    splits = [
        ("train", RAW_TRAIN_FILE, PROCESSED_TRAIN_FILE),
        ("validation", RAW_VAL_FILE, PROCESSED_VAL_FILE),
        ("test", RAW_TEST_FILE, PROCESSED_TEST_FILE),
    ]

    processed_splits = {}

    for split_name, raw_path, processed_path in splits:
        if processed_path.exists() and not force:
            logger.info(f"Loading existing processed split from {processed_path}...")
            df = pd.read_csv(processed_path)
        else:
            logger.info(f"Preprocessing raw split '{split_name}' from {raw_path}...")
            raw_df = load_raw_split(split_name)
            df = preprocess_dataset(raw_df)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False, encoding="utf-8")
            logger.info(f"Saved {len(df)} cleaned records to {processed_path}.")

        processed_splits[split_name] = df

    return processed_splits


def load_processed_split(split: str = "test") -> pd.DataFrame:
    """Load a preprocessed dataset split from data/processed/.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    file_map = {
        "train": PROCESSED_TRAIN_FILE,
        "validation": PROCESSED_VAL_FILE,
        "test": PROCESSED_TEST_FILE,
    }

    if split not in file_map:
        raise ValueError(f"Invalid split '{split}'. Allowed: {list(file_map.keys())}")

    path = file_map[split]
    if not path.exists():
        preprocess_and_save_all_splits()

    return pd.read_csv(path)


if __name__ == "__main__":
    print("=" * 70)
    print("Preprocessing CNN/DailyMail Dataset Splits")
    print("=" * 70)
    splits = preprocess_and_save_all_splits(force=True)
    for name, df in splits.items():
        print(f"Processed '{name}' split: {len(df)} records.")

    print("\n--- Example Before vs. After Cleaning ---")
    raw_sample = pd.read_csv(RAW_TEST_FILE).iloc[0]
    clean_sample = splits["test"].iloc[0]
    print("\n[Raw Article Opening]:")
    print(raw_sample["article"][:200])
    print("\n[Cleaned Article Opening]:")
    print(clean_sample["article"][:200])
    print("\n[Raw Highlights]:")
    print(raw_sample["highlights"][:150])
    print("\n[Cleaned Highlights]:")
    print(clean_sample["highlights"][:150])
