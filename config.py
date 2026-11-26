"""Configuration module for Text Summarization NLP Project.

Defines all paths, hyperparameters, model identifiers, and hardware settings.
Uses pathlib for OS-agnostic path management.
"""

from pathlib import Path
import torch

# ==============================================================================
# Project Root & Directory Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUMMARIES_DIR = OUTPUTS_DIR / "summaries"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure runtime directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SUMMARIES_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Reproducibility & Random Seed
# ==============================================================================
RANDOM_SEED = 42

# ==============================================================================
# Dataset Configuration (CNN / DailyMail 3.0.0)
# ==============================================================================
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"

# Reproducible development sample sizes
TRAIN_SAMPLE_SIZE = 1000
VAL_SAMPLE_SIZE = 200
TEST_SAMPLE_SIZE = 200

# Fast evaluation subset size (for benchmarking on 6GB VRAM)
EVAL_SAMPLE_SIZE = 50

# Local sample file paths
RAW_TRAIN_FILE = RAW_DATA_DIR / "raw_train_sample.csv"
RAW_VAL_FILE = RAW_DATA_DIR / "raw_val_sample.csv"
RAW_TEST_FILE = RAW_DATA_DIR / "raw_test_sample.csv"

PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "processed_train_sample.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "processed_val_sample.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "processed_test_sample.csv"

# ==============================================================================
# Model Configurations
# ==============================================================================
PRIMARY_MODEL_NAME = "facebook/bart-large-cnn"
ALTERNATIVE_MODEL_NAME = "google-t5/t5-small"

# Model Input / Output Limits
BART_MAX_INPUT_LENGTH = 1024
T5_MAX_INPUT_LENGTH = 512

# TextRank Baseline Settings
TEXTRANK_NUM_SENTENCES = 3
TEXTRANK_DAMPING = 0.85

# Generation Hyperparameters for Abstractive Models
GENERATION_PARAMS_QUALITY = {
    "max_length": 128,
    "min_length": 30,
    "num_beams": 4,
    "length_penalty": 2.0,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}

GENERATION_PARAMS_FAST = {
    "max_length": 128,
    "min_length": 30,
    "num_beams": 1,
    "no_repeat_ngram_size": 3,
}

GENERATION_PRESETS = {
    "quality": GENERATION_PARAMS_QUALITY,
    "fast": GENERATION_PARAMS_FAST,
}

# Default backwards-compatible parameters
GENERATION_PARAMS = dict(GENERATION_PARAMS_QUALITY)

# ==============================================================================
# Hardware & Acceleration Configuration
# ==============================================================================
def get_device() -> torch.device:
    """Detect and return the optimal compute device (CUDA GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
USE_FP16 = torch.cuda.is_available()  # Enable mixed precision (float16) on GPU to fit 6GB VRAM
INFERENCE_BATCH_SIZE = 1  # Conservative batch size to prevent CUDA OOM
