"""Data Loader Module for Text Summarization Project.

Responsible for:
- Inspecting CNN/DailyMail dataset metadata without downloading the full dataset.
- Streaming reproducible subsets (train, validation, test) efficiently.
- Validating and cleaning missing or empty values.
- Saving raw subset samples for offline, reproducible experimentation.
- Calculating word and character length distributions.
"""

import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path so config and other modules resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import (
    DATASET_NAME,
    DATASET_CONFIG,
    RAW_TRAIN_FILE,
    RAW_VAL_FILE,
    RAW_TEST_FILE,
    TRAIN_SAMPLE_SIZE,
    VAL_SAMPLE_SIZE,
    TEST_SAMPLE_SIZE,
    RANDOM_SEED,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_loader")


def get_dataset_metadata() -> Dict[str, Any]:
    """Retrieve CNN/DailyMail metadata without downloading the full dataset.

    Returns:
        Dict[str, Any]: Metadata dictionary containing features, splits, sizes, and record counts.
    """
    from datasets import load_dataset_builder
    logger.info(f"Inspecting metadata for {DATASET_NAME} ({DATASET_CONFIG})...")
    builder = load_dataset_builder(DATASET_NAME, DATASET_CONFIG)
    info = builder.info

    splits_info = {}
    total_examples = 0
    total_bytes = 0

    if info.splits:
        for split_name, split_meta in info.splits.items():
            splits_info[split_name] = {
                "num_examples": split_meta.num_examples,
                "num_bytes": split_meta.num_bytes,
                "size_mb": round(split_meta.num_bytes / (1024 * 1024), 2),
            }
            total_examples += split_meta.num_examples
            total_bytes += split_meta.num_bytes

    metadata = {
        "dataset_name": DATASET_NAME,
        "config_version": DATASET_CONFIG,
        "features": list(info.features.keys()) if info.features else [],
        "splits": splits_info,
        "total_examples": total_examples,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "download_size_mb": round(info.download_size / (1024 * 1024), 2) if info.download_size else None,
    }

    logger.info(
        f"Metadata loaded: {total_examples:,} total records across {list(splits_info.keys())}."
    )
    return metadata


def stream_dataset_subset(split: str, sample_size: int) -> pd.DataFrame:
    """Stream a subset of examples from Hugging Face Datasets without full download.

    Args:
        split (str): Name of split ('train', 'validation', or 'test').
        sample_size (int): Number of valid records to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing columns ['id', 'article', 'highlights'].
    """
    from datasets import load_dataset
    logger.info(f"Streaming {sample_size} records from '{split}' split...")
    dataset_stream = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        streaming=True,
    )

    records = []
    skipped_empty = 0

    for item in dataset_stream:
        article_text = item.get("article", "")
        summary_text = item.get("highlights", "")
        item_id = item.get("id", "")

        # Handle missing or empty values
        if not article_text or not summary_text:
            skipped_empty += 1
            continue

        clean_article = article_text.strip()
        clean_summary = summary_text.strip()

        if len(clean_article) == 0 or len(clean_summary) == 0:
            skipped_empty += 1
            continue

        records.append({
            "id": str(item_id),
            "article": clean_article,
            "highlights": clean_summary,
        })

        if len(records) >= sample_size:
            break

    if skipped_empty > 0:
        logger.warning(f"Skipped {skipped_empty} empty/missing records during streaming.")

    df = pd.DataFrame(records)
    logger.info(f"Successfully streamed {len(df)} records for split '{split}'.")
    return df


def acquire_and_save_data(
    train_size: int = TRAIN_SAMPLE_SIZE,
    val_size: int = VAL_SAMPLE_SIZE,
    test_size: int = TEST_SAMPLE_SIZE,
    force_reload: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Acquire reproducible development subsets and persist them to data/raw/.

    Args:
        train_size (int): Number of training examples to extract.
        val_size (int): Number of validation examples to extract.
        test_size (int): Number of test examples to extract.
        force_reload (bool): If True, re-download even if local files exist.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to DataFrames.
    """
    subsets = {}
    split_configs = [
        ("train", train_size, RAW_TRAIN_FILE),
        ("validation", val_size, RAW_VAL_FILE),
        ("test", test_size, RAW_TEST_FILE),
    ]

    for split_name, size, file_path in split_configs:
        if file_path.exists() and not force_reload:
            logger.info(f"Loading existing raw sample from {file_path}...")
            df = pd.read_csv(file_path)
        else:
            df = stream_dataset_subset(split=split_name, sample_size=size)
            df.to_csv(file_path, index=False, encoding="utf-8")
            logger.info(f"Saved {len(df)} records to {file_path}.")
        subsets[split_name] = df

    return subsets


def load_raw_split(split: str = "test") -> pd.DataFrame:
    """Load a specific raw dataset split from local storage.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_map = {
        "train": RAW_TRAIN_FILE,
        "validation": RAW_VAL_FILE,
        "test": RAW_TEST_FILE,
    }

    if split not in file_map:
        raise ValueError(f"Invalid split '{split}'. Must be one of {list(file_map.keys())}.")

    file_path = file_map[split]
    if not file_path.exists():
        logger.info(f"Local file {file_path} not found. Acquiring subsets first...")
        acquire_and_save_data()

    return pd.read_csv(file_path)


def compute_length_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute word count, character count, and ratio statistics for a dataset split.

    Args:
        df (pd.DataFrame): DataFrame containing 'article' and 'highlights'.

    Returns:
        Dict[str, Any]: Dictionary containing descriptive statistical metrics.
    """
    article_words = df["article"].apply(lambda x: len(str(x).split()))
    summary_words = df["highlights"].apply(lambda x: len(str(x).split()))
    article_chars = df["article"].apply(lambda x: len(str(x)))
    summary_chars = df["highlights"].apply(lambda x: len(str(x)))
    ratios = article_words / summary_words.replace(0, 1)

    stats = {
        "total_records": len(df),
        "article_words": {
            "min": int(article_words.min()),
            "max": int(article_words.max()),
            "mean": round(float(article_words.mean()), 2),
            "median": round(float(article_words.median()), 2),
            "std": round(float(article_words.std()), 2),
        },
        "summary_words": {
            "min": int(summary_words.min()),
            "max": int(summary_words.max()),
            "mean": round(float(summary_words.mean()), 2),
            "median": round(float(summary_words.median()), 2),
            "std": round(float(summary_words.std()), 2),
        },
        "compression_ratio": {
            "mean": round(float(ratios.mean()), 2),
            "median": round(float(ratios.median()), 2),
        },
        "article_chars_mean": round(float(article_chars.mean()), 2),
        "summary_chars_mean": round(float(summary_chars.mean()), 2),
    }

    return stats


if __name__ == "__main__":
    print("=" * 70)
    print("CNN/DailyMail (3.0.0) Metadata Inspection")
    print("=" * 70)
    metadata = get_dataset_metadata()
    for key, val in metadata.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("Acquiring Reproducible Subsets (Train: 1000, Val: 200, Test: 200)")
    print("=" * 70)
    subsets = acquire_and_save_data()

    for split_name, df in subsets.items():
        print(f"\n--- Split: {split_name} ({len(df)} samples) ---")
        stats = compute_length_statistics(df)
        print(f"  Article Word Count -> Mean: {stats['article_words']['mean']}, Min: {stats['article_words']['min']}, Max: {stats['article_words']['max']}")
        print(f"  Summary Word Count -> Mean: {stats['summary_words']['mean']}, Min: {stats['summary_words']['min']}, Max: {stats['summary_words']['max']}")
        print(f"  Compression Ratio (Article/Summary) -> Mean: {stats['compression_ratio']['mean']}x")

    print("\n" + "=" * 70)
    print("Sample Article-Summary Pair from Test Set:")
    print("=" * 70)
    test_sample = subsets["test"].iloc[0]
    print(f"ID: {test_sample['id']}")
    print(f"\nArticle (first 300 chars):\n{test_sample['article'][:300]}...")
    print(f"\nReference Summary (Highlights):\n{test_sample['highlights']}")
