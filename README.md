---
title: NewsLens - AI-Powered News Summarization
emoji: 📰
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Benchmarking Extractive vs Abstractive NLP Summarization
---

# NewsLens: Text Summarization Using Pre-trained Models

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.32.0-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-NewsLens-black.svg)](https://github.com/abg11hani235-commits/NewsLens)

An academic and production-ready Natural Language Processing (NLP) text summarization system and web application comparing **extractive** and **abstractive** summarization paradigms on the **CNN/DailyMail (3.0.0)** benchmark.

The project implements classical heuristics, graph-based algorithms, and state-of-the-art encoder-decoder Transformer architectures optimized for consumer GPU hardware (NVIDIA RTX 3060, 6GB VRAM) and resource-conscious cloud deployment (Hugging Face Spaces CPU Free Tier).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [NewsLens Web Application](#newslens-web-application)
3. [Theoretical Foundations & Architectures](#theoretical-foundations--architectures)
   - [Extractive vs. Abstractive Summarization](#extractive-vs-abstractive-summarization)
   - [Lead-3 Heuristic Baseline](#lead-3-heuristic-baseline)
   - [TextRank Graph-Based Algorithm](#textrank-graph-based-algorithm)
   - [BART (Bidirectional and Auto-Regressive Transformer)](#bart-architecture)
   - [T5 (Text-to-Text Transfer Transformer)](#t5-architecture)
4. [Dataset: CNN/DailyMail 3.0.0](#dataset-cnndailymail-300)
5. [Hardware Optimization & Memory Architecture](#hardware-optimization--memory-architecture)
6. [High-Performance Engine Optimizations](#high-performance-engine-optimizations)
7. [Hugging Face Spaces Cloud Deployment](#hugging-face-spaces-cloud-deployment)
8. [Project Architecture & Directory Structure](#project-architecture--directory-structure)
9. [Installation & Setup](#installation--setup)
10. [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
11. [Experimental Results & Benchmark](#experimental-results--benchmark)
    - [Quantitative ROUGE Benchmark Table](#quantitative-rouge-benchmark-table)
    - [Measured Performance & Throughput Benchmark](#measured-performance--throughput-benchmark)
    - [Sequential vs. True Vectorized Batching](#sequential-vs-true-vectorized-batching)
    - [Visualizations](#visualizations)
    - [Qualitative Case Study](#qualitative-case-study)
12. [Unit Testing & Quality Assurance](#unit-testing--quality-assurance)
13. [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

Text summarization is the task of condensing a source document into a concise version while preserving salient information and factual integrity. This repository provides an end-to-end NLP engineering pipeline and interactive web interface demonstrating:
- **Reproducible Data Ingestion:** Streaming and persisting structured subsets of CNN/DailyMail without downloading multi-gigabyte archives.
- **Exploratory Data Analysis (EDA):** Quantifying article lengths, summary bounds, compression ratios, and vocabulary distributions.
- **Linguistic Preprocessing:** Removing journalistic artifacts (e.g., `(CNN) --`), normalizing bullet points, and applying model-specific subword tokenization with offline NLTK resource resilience.
- **Extractive Baselines:** Implementing Lead-3 and a custom TF-IDF/PageRank TextRank algorithm via NetworkX with candidate sentence capping.
- **Abstractive Transformers:** Deploying `facebook/bart-large-cnn` and `google-t5/t5-small` in FP16 mixed precision on CUDA with automated CPU fallback.
- **Standardized Evaluation:** Evaluating models with ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1), length tracking, and synchronized GPU latency profiling.
- **Interactive Web Application:** A full-featured Gradio app featuring single-document summarization, side-by-side model comparison, and academic benchmarking.

---

## NewsLens Web Application

NewsLens features an interactive, production-grade Gradio application (`app.py`) designed for both research analysis and end-user demonstration:

### Key Application Features
1. **Single Summarizer Tab:**
   - **Large Article Input:** Real-time character and word count tracking as the user types.
   - **Preloaded Sample Articles:** One-click loading of real-world diplomatic, technological, and scientific news stories.
   - **Hyperparameter Controls:** Tunable beam search width ($k$), max/min generation length, and extractive sentence count.
   - **Model Selector:** Seamlessly switch between Lead-3, TextRank, T5-Small, and BART-Large-CNN.
   - **Performance Telemetry:** Displays exact inference latency (seconds), compute device (`CUDA (FP16)` / `CPU`), summary word count, and compression factor.
   - **Clipboard & Reset:** Integrated copy-to-clipboard button and one-click form reset.
2. **Model Comparison Tab:**
   - Evaluates a single input document across **all four architectures** simultaneously.
   - Displays Lead-3, TextRank, T5-Small, and BART summaries in a 4-quadrant layout with individual copy buttons.
   - Renders a comparative Markdown summary table detailing architecture, paradigm, parameter count, latency, output words, compression ratio, and compute device.
3. **Academic Methodology & Guide Tab:**
   - Comprehensive theoretical comparison of Extractive vs. Abstractive paradigms.
   - Embedded official CNN/DailyMail ROUGE benchmark table.
   - Hardware optimization and deployment notes.

### Launching the Web Interface Locally
```powershell
# Method 1: Direct launch
python app.py

# Method 2: Unified CLI launch
python main.py app

# Launch with custom port and host
python main.py app --server_port 7860 --server_name 127.0.0.1
```
Navigate to `http://127.0.0.1:7860` in your web browser.

---

## Theoretical Foundations & Architectures

### Extractive vs. Abstractive Summarization

| Dimension | Extractive Summarization (Lead-3 / TextRank) | Abstractive Summarization (BART / T5) |
|---|---|---|
| **Mechanism** | Selects verbatim sentences directly from the source text based on scoring criteria. | Generates novel phrasing and syntactically rephrased sentences via conditional language generation. |
| **Factual Integrity** | 100% faithful to the text; zero risk of hallucination. | Subject to factual drift or hallucinations if model associations override context. |
| **Linguistic Fluency** | Coherence across sentence boundaries can be abrupt or disjointed. | Smooth, natural transitions and syntactically cohesive summaries. |
| **Compute Demand** | Low (CPU-executable in milliseconds). | High (Requires Transformer matrix operations; accelerated via GPU FP16). |

---

### Lead-3 Heuristic Baseline
In journalism, news stories follow the **inverted pyramid structure**: the most critical factual elements (*who, what, when, where, why*) are concentrated in the opening sentences, while supporting background details appear later. Lead-3 extracts the first 3 sentences verbatim. In CNN/DailyMail research (e.g., See et al., 2017), Lead-3 serves as an exceptionally strong empirical baseline that often outscores unsupervised methods.

---

### TextRank Graph-Based Algorithm
TextRank (Mihalcea & Tarau, 2004) is an unsupervised graph-based ranking model derived from Google's PageRank:
1. The article is segmented into sentences $V = \{s_1, s_2, \dots, s_n\}$.
2. Pairwise sentence similarity is computed using TF-IDF vector representations and Cosine Similarity:
   $$\text{Sim}(s_i, s_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$
3. An undirected weighted graph $G = (V, E)$ is constructed where edge weights reflect semantic similarity.
4. Centrality scores are computed iteratively via PageRank:
   $$S(V_i) = (1 - d) + d \sum_{V_j \in \text{In}(V_i)} \frac{\text{Weight}(V_j, V_i)}{\sum_{V_k \in \text{Out}(V_j)} \text{Weight}(V_j, V_k)} S(V_j)$$
5. The top $N$ ranked sentences are extracted and re-sorted chronologically to preserve the narrative sequence.

---

### BART Architecture
`facebook/bart-large-cnn` (Lewis et al., 2020) is a standard sequence-to-sequence encoder-decoder Transformer with a bidirectional encoder (like BERT) and an autoregressive decoder (like GPT):
- **Pretraining Objective:** Denoising autoencoder (token masking, sentence permutation, document rotation, text infilling).
- **Fine-Tuning:** Specifically fine-tuned on the CNN/DailyMail summarization corpus.
- **Context Capacity:** 1,024 subword tokens (covers ~750 words).
- **Decoding:** Autoregressive beam search ($k=4$) with length penalty $\alpha = 2.0$ and 3-gram repetition blocking.

---

### T5 Architecture
`google-t5/t5-small` (Raffel et al., 2020) unifies all NLP tasks into a single text-to-text format:
- **Input Formulation:** The source article is prepended with the explicit task prefix: `summarize: <article text>`.
- **Model Scale:** 60 million parameters (lightweight alternative to BART's 406M parameters).
- **Context Capacity:** 512 subword tokens.

---

## Dataset: CNN/DailyMail 3.0.0

* **Identifier:** `abisee/cnn_dailymail` (version `3.0.0`)
* **Fields:**
  - `article`: The full text of the news story.
  - `highlights`: Human-written multi-sentence bullet points summarizing the event.
  - `id`: Unique SHA-1 hash identifier.
* **Corpus Scale:**
  - Train: 287,113 stories
  - Validation: 13,368 stories
  - Test: 11,490 stories
  - Total: 311,971 records (~1.31 GB)
* **Empirical Distributions (from Phase 4 EDA):**
  - Average Article Length: **591.6 ± 286.7 words** (Median: 547.0, 95th Percentile: 1,102.4 words).
  - Average Summary Length: **42.8 ± 7.6 words** (Median: 43.0, IQR: [37.0 – 49.0]).
  - Compression Ratio: **14.15×** (articles are compressed to ~7% of original length).
  - Pearson Correlation ($r = 0.1268$): Human summaries maintain a consistent length regardless of article verbosity.

---

## Hardware Optimization & Memory Architecture

### 1. Consumer GPU Optimization (RTX 3060, 6GB VRAM)
- **FP16 Mixed Precision:** Loaded via `torch.float16` when CUDA is available, reducing VRAM consumption by 50% (~780 MB–850 MB VRAM vs. ~3.2 GB in FP32).
- **Explicit CUDA Synchronization:** Timing measurements use `torch.cuda.synchronize()` before and after generation to ensure accurate latency figures that reflect completed GPU execution kernels.
- **Safe FP32 CPU OOM Fallback with Precision Conversion:** In `src/summarizer.py`, if a CUDA Out-Of-Memory exception occurs, model weights are converted to `torch.float32` on CPU (since x86 CPUs do not support native Half-precision operations) and input tensors are moved to CPU. After CPU generation, restoration to GPU is attempted; if restoration fails, the instance cleanly switches to persistent CPU execution with truthful telemetry (`cpu (oom-fallback-fp32, persistent)`).
- **Elimination of Hot-Path Cache Thrashing:** Removed unnecessary `torch.cuda.empty_cache()` calls from the normal generation path. PyTorch's caching allocator reuses tensor memory at native speed without driver-level CPU-GPU synchronization stalls.

### 2. Thread-Safe Model Management & Lease Lifecycle (`ModelManager`)
To prevent concurrent GPU memory allocation or RAM spikes on cloud hosts:
- Singleton `ModelManager` class lazily loads models on demand.
- **Inference Lifecycle Leasing (`lease_model`):** Primary and required interface for all neural inference. Callers acquire an active lease context (`with MODEL_MANAGER.lease_model(...)`) before running inference, guaranteeing the model cannot be evicted or unloaded mid-generation by concurrent requests.
- **Legacy API Deprecation Notice:** Direct retrieval via `get_model()` and `get_model_with_metadata()` is deprecated and raises `DeprecationWarning`. Direct retrieval does not maintain an active lease; callers must wrap inference in a lease. Condition-variable waiting alone does NOT guarantee safety across the full inference lifecycle without holding a lease.
- **Defense-in-Depth Eviction Safety:** LRU eviction checks both active lease reference counts and the underlying model's `_inference_lock` (`is_inferring`). Models executing legacy unleased inference cannot be evicted until inference releases.
- **Benchmark & Resource Separation:** Tracks model initialization time (`init_time_sec`) separately from proactive warmup inference (`warmup_inference_sec`) and per-document generation passes.
- **Granular Batch Metrics:** `summarize_batch()` independently tracks per-document preprocessing (`preprocessing_latency_sec`), shared batch tokenization (`batch_tokenization_latency_sec`), batch generation (`batch_generation_latency_sec`), and chunk elapsed time (`chunk_elapsed_sec`), explicitly tagging amortized batch latencies with `is_amortized_batch: True`.
- Enforces memory-conscious LRU eviction (`max_cached_models=1` by default) so BART and T5 are **never resident simultaneously in memory**.

---

## High-Performance Engine Optimizations

| Optimization | Target Module | Technical Implementation | Measured Impact |
|---|---|---|---|
| **True Vectorized Batching** | `src/summarizer.py` | Vectorized batch tokenization with dynamic padding, batch decoding, and granular latency metrics. | **3.89x throughput speedup** over sequential execution. |
| **Hot-Path Cache Retention** | `src/summarizer.py` | Removed `empty_cache()` from single inference; reserve only for OOM recovery and eviction. | Prevents driver-level stalls; enables sub-second warm inference. |
| **FP32 CPU OOM Fallback** | `src/summarizer.py` | Recovers from GPU OOM by casting weights to FP32 on CPU, executing safely, then restoring. | Prevents `Half` precision crashes on standard x86 CPU architectures. |
| **Vectorized TextRank Graph** | `src/extractive.py` | Constructed NetworkX graph directly from thresholded NumPy adjacency matrix. | Eliminates Python double loop; scales to large matrices instantly. |
| **Candidate Sentence Capping** | `src/extractive.py` | Capped TextRank candidate pool to leading 50 sentences (`max_candidate_sentences=50`). | Guaranteed `< 4ms` latency even on documents with 1,000+ sentences. |
| **Lazy Module Ingestion** | `main.py`, `src/preprocessing.py` | Deferred heavy imports (transformers, matplotlib, seaborn, evaluate) to subcommands. | Reduced CLI startup latency from **>7.0s to 2.0s**. |
| **Lifecycle Leasing Concurrency** | `app.py` | Reference counting + Condition variable leasing (`lease_model`) prevents mid-inference evictions. | Zero race conditions under simultaneous multi-user requests switching BART/T5. |

---

## Hugging Face Spaces Cloud Deployment

NewsLens is pre-configured for direct, zero-cost deployment to **Hugging Face Spaces** using the free CPU tier (2 vCPU, 16GB RAM).

### Deployment Configuration & Current Status
- **SDK:** Gradio 6.32.0 (compatible with 5.x and 6.x)
- **Entry Point:** `app.py`
- **Dependency Split:**
  - `requirements.txt`: Clean, cross-platform dependencies without platform-specific wheel URLs.
  - `requirements-cuda.txt`: Pinned Windows/CUDA 12.4 configuration for local GPU development.
- **Current Deployment Status:** Pre-configured and verified locally. The codebase is fully prepared for Hugging Face Spaces, but deployment is **not complete until a public Space URL is verified** by pushing to a configured Space remote (`space`). Currently, only the primary GitHub repository (`origin`) is connected.

> [!WARNING]
> **Hardware Discrepancy & Performance Expectations:**
> - **Local Development Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 12.4, FP16 mixed precision).
> - **Cloud Free Tier Hardware:** Hugging Face Spaces CPU Basic (2 vCPU, 16GB RAM, FP32 execution).
> - Local sub-second inference latencies (~0.65s for BART) **do not represent Hugging Face CPU performance**. On 2 vCPUs without GPU acceleration, BART-Large-CNN requires approximately **5–8 seconds** per inference in FP32, while T5-Small takes **~1.5–2.5 seconds**. Extractive algorithms (Lead-3, TextRank) remain instantaneous (< 0.01s) on CPU.

### Steps to Complete Hugging Face Spaces Deployment:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space):
   - **Space Name:** `NewsLens`
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (Free · 2 vCPU · 16 GB RAM)
2. Connect your GitHub repository (`https://github.com/abg11hani235-commits/NewsLens`) or push via Git remote:
   ```powershell
   git remote add space https://huggingface.co/spaces/<your-username>/NewsLens
   git push space main
   ```
3. Hugging Face Spaces will automatically read `app.py` and `requirements.txt`, install dependencies, and launch the application. Verify the live public URL upon completion.

---

## Project Architecture & Directory Structure

```text
NewsLens/
│
├── .gitignore                      # Exclusions for .venv/, datasets, caches, model weights
├── requirements.txt                # Cross-platform cloud deployment dependencies (Hugging Face Spaces)
├── requirements-cuda.txt           # Local GPU dependencies (Windows 11 / CUDA 12.4)
├── config.py                       # Configuration paths, hyperparameters, and presets
├── main.py                         # Unified Command-Line Interface (CLI) application with lazy imports
├── app.py                          # Production Gradio Web Application with thread-safe ModelManager
├── README.md                       # Academic documentation & deployment guide
│
├── scripts/
│   └── benchmark_performance.py    # Automated, reproducible throughput and latency benchmark suite
│
├── data/
│   ├── raw/                        # Untouched streamed datasets (CSV)
│   │   ├── raw_train_sample.csv    # 1,000 reproducible training records
│   │   ├── raw_val_sample.csv      # 200 validation records
│   │   └── raw_test_sample.csv     # 200 test records
│   └── processed/                  # Cleaned, normalized dataset splits
│       ├── processed_train_sample.csv
│       ├── processed_val_sample.csv
│       └── processed_test_sample.csv
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb        # EDA, distributions, metadata
│   ├── 02_text_preprocessing.ipynb          # Cleaning, tokenization, truncation
│   ├── 03_extractive_summarization.ipynb    # Lead-3 & TextRank implementation
│   ├── 04_abstractive_summarization.ipynb   # BART & T5 inference on GPU
│   └── 05_model_evaluation.ipynb           # ROUGE metrics, benchmarking, analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Streaming, metadata extraction, length statistics
│   ├── preprocessing.py            # Text cleaning, highlight normalization, lazy NLTK checks
│   ├── extractive.py               # Lead-N heuristic & optimized NetworkX TextRank
│   ├── summarizer.py               # AbstractiveSummarizer with true batching, warmup, FP16
│   ├── evaluation.py               # ROUGE calculation & full benchmark runner
│   └── visualization.py            # Publication plots (Matplotlib/Seaborn Agg backend)
│
├── outputs/
│   ├── summaries/                  # Model predictions and comparison files
│   │   ├── extractive_summaries.csv
│   │   ├── model_comparison_samples.csv
│   │   └── qualitative_comparison.json
│   ├── metrics/                    # Evaluation tabular metrics (CSV & JSON)
│   │   ├── eda_summary_statistics.csv
│   │   ├── eda_summary_statistics.json
│   │   ├── rouge_scores.csv
│   │   ├── rouge_scores.json
│   │   ├── benchmark_performance.csv       # Measured multi-document benchmark results
│   │   ├── benchmark_performance.json
│   │   └── batch_comparison.json           # Sequential vs batch speedup metrics
│   └── figures/                    # Publication-grade visualization charts
│       ├── article_length_dist.png
│       ├── summary_length_dist.png
│       ├── article_vs_summary_length.png
│       ├── compression_ratio_dist.png
│       ├── rouge_comparison.png
│       ├── model_latency_comparison.png
│       └── generated_summary_lengths.png
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py         # Split loading, schema validation, length statistics
    ├── test_preprocessing.py       # Wire artifact cleaning, whitespace, punctuation
    ├── test_extractive.py          # Lead-3, TextRank, candidate capping, fallback
    ├── test_summarizer.py          # Real BART & T5 inference, batching, warmup, presets, long input
    ├── test_evaluation.py          # ROUGE-1/2/L score computations, edge cases
    ├── test_visualization.py       # Headless plot rendering and artifact validation
    └── test_app.py                 # Gradio interface, thread-safe ModelManager, caching, multi-model
```

---

## Installation & Setup

### 1. Prerequisites
- **Operating System:** Windows 10/11, Linux, or macOS.
- **Python:** 3.12 (64-bit).
- **GPU (Optional):** NVIDIA GPU with CUDA driver (e.g. RTX 3060).

### 2. Environment Setup
```powershell
# Create .venv
python -m venv .venv

# Activate .venv (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate .venv (Linux / macOS)
source .venv/bin/activate
```

### 3. Install Dependencies
**For Local GPU Acceleration (Windows / CUDA 12.4):**
```powershell
pip install -r requirements-cuda.txt
```

**For Standard / Cloud CPU Deployment (Cross-platform):**
```powershell
pip install -r requirements.txt
```

---

## Command-Line Interface (CLI) Usage

The application provides a unified CLI through `main.py`:

### 1. Launch Gradio Web Application
```powershell
python main.py app
# or with custom options
python main.py app --server_port 7860 --server_name 0.0.0.0
```

### 2. Summarize Custom Text (Abstractive BART or T5)
```powershell
python main.py summarize --model bart --text "The European Union agreed on new artificial intelligence regulations today in Brussels. Lawmakers hailed the legislation as a historic step for consumer safety and transparency. Technology companies raised questions regarding compliance timelines and international competitiveness."
```

### 3. Summarize Dataset Test Samples
```powershell
python main.py summarize --model bart --sample_size 3
```

### 4. Extractive Summarization (TextRank or Lead-3)
```powershell
python main.py extractive --method textrank --sample_size 5
```

### 5. Run Full Comparative Evaluation Benchmark
```powershell
python main.py evaluate --sample_size 50
```

### 6. Run Performance Benchmark Suite
```powershell
python scripts/benchmark_performance.py
```

### 7. Non-Destructive Preprocessing & EDA
```powershell
# Safe non-destructive run (skips already processed files)
python main.py preprocess

# Force rebuild processed files
python main.py preprocess --force

# Run EDA and update distribution charts
python main.py eda
```

---

## Experimental Results & Benchmark

### Quantitative ROUGE Benchmark Table

All four architectures were evaluated on the **exact same 50 test samples** from the CNN/DailyMail test split:

| Model | Paradigm | Parameters | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-1 Prec | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-2 Prec | ROUGE-L F1 | ROUGE-L Recall | ROUGE-L Prec | Avg Words | Avg Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Heuristic | N/A | 29.91% | 45.95% | 23.24% | 12.63% | 20.68% | 9.47% | 21.82% | 34.06% | 16.83% | 62.1 | **0.002s** |
| **TextRank** | Extractive Graph | N/A | 27.16% | 42.25% | 20.77% | 8.36% | 13.76% | 6.21% | 18.40% | 29.01% | 13.99% | 66.6 | 0.005s |
| **T5-Small** | Abstractive Seq2Seq | 60M | 31.40% | 37.82% | 28.19% | 12.23% | 14.68% | 11.12% | 22.71% | 27.24% | 20.47% | 42.9 | 0.984s |
| **BART-Large-CNN** | Abstractive Seq2Seq | 406M | **36.38%** | **42.97%** | **32.85%** | **16.45%** | **19.74%** | **14.72%** | **27.11%** | **32.12%** | **24.43%** | **41.9** | 0.973s |

*Hardware: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 12.4, FP16 Precision).*

---

### Measured Performance & Throughput Benchmark

Empirically measured using `scripts/benchmark_performance.py` across standardized Short (48 words), Medium (157 words), and Long (208 words) documents over 5 repetitions (NVIDIA GeForce RTX 3060 Laptop GPU, 6GB VRAM, CUDA 12.4, PyTorch 2.6.0, FP16):

| Model Architecture | Document Scale | Input Words | Output Words | Warm Latency (Mean ± Std) | Warm Median | Output Throughput | Peak VRAM | Compute Device |
|---|---|---|---|---|---|---|---|---|
| **Lead-3** | Short | 48 words | 48 words | **0.0001s** (± 0.0s) | 0.0001s | ~545,100 w/s | 0.0 MB | CPU |
| **Lead-3** | Medium | 157 words | 79 words | **0.0002s** (± 0.0s) | 0.0002s | ~374,200 w/s | 0.0 MB | CPU |
| **Lead-3** | Long | 208 words | 78 words | **0.0003s** (± 0.0s) | 0.0003s | ~257,300 w/s | 0.0 MB | CPU |
| **TextRank** | Short | 48 words | 48 words | **0.0001s** (± 0.0s) | 0.0001s | ~622,400 w/s | 0.0 MB | CPU |
| **TextRank** | Medium | 157 words | 82 words | **0.0025s** (± 0.0003s) | 0.0024s | ~33,500 w/s | 0.0 MB | CPU |
| **TextRank** | Long | 208 words | 67 words | **0.0036s** (± 0.0001s) | 0.0036s | ~18,700 w/s | 0.0 MB | CPU |
| **T5-Small** | Short | 48 words | 29 words | **0.5885s** (± 0.0336s) | 0.5835s | 49.3 w/s | 155.8 MB | CUDA (FP16) |
| **T5-Small** | Medium | 157 words | 34 words | **0.6934s** (± 0.0227s) | 0.6828s | 49.0 w/s | 163.3 MB | CUDA (FP16) |
| **T5-Small** | Long | 208 words | 47 words | **1.0636s** (± 0.0326s) | 1.0696s | 44.2 w/s | 170.5 MB | CUDA (FP16) |
| **BART-Large-CNN** | Short | 48 words | 30 words | **0.6184s** (± 0.0414s) | 0.5920s | 48.5 w/s | 806.9 MB | CUDA (FP16) |
| **BART-Large-CNN** | Medium | 157 words | 41 words | **0.7286s** (± 0.0487s) | 0.7316s | 56.3 w/s | 834.1 MB | CUDA (FP16) |
| **BART-Large-CNN** | Long | 208 words | 43 words | **0.9051s** (± 0.0363s) | 0.9283s | 47.5 w/s | 849.7 MB | CUDA (FP16) |

---

### Granular Pipeline Stage Latency Breakdown

The inference pipeline tracks fine-grained execution stages with microsecond precision:

| Model Architecture | Document Scale | Preprocessing | Tokenization | Neural Generation | Detokenization | Total End-to-End |
|---|---|---|---|---|---|---|
| **BART-Large-CNN** | Short (48 words) | < 0.1 ms | 0.5 ms | 617.6 ms (99.8%) | 0.1 ms | **618.4 ms** |
| **BART-Large-CNN** | Medium (157 words) | < 0.1 ms | 0.8 ms | 727.4 ms (99.8%) | 0.2 ms | **728.6 ms** |
| **BART-Large-CNN** | Long (208 words) | < 0.1 ms | 0.9 ms | 903.6 ms (99.8%) | 0.2 ms | **904.8 ms** |
| **T5-Small** | Short (48 words) | < 0.1 ms | 0.5 ms | 587.8 ms (99.9%) | 0.1 ms | **588.2 ms** |
| **T5-Small** | Medium (157 words) | < 0.1 ms | 0.9 ms | 692.2 ms (99.8%) | 0.1 ms | **693.4 ms** |
| **T5-Small** | Long (208 words) | < 0.1 ms | 1.1 ms | 1,062.0 ms (99.8%) | 0.1 ms | **1,063.6 ms** |

---

### Sequential vs. True Vectorized Batching

Measured on 4 standard test documents evaluated with `T5-Small` on GPU:

| Execution Mode | Batch Size | Total Elapsed Time | Per-Document Latency | Throughput Speedup |
|---|---|---|---|---|
| **Sequential Processing** | 1 | 2.9880s | 0.7470s / doc | 1.00x (Baseline) |
| **Vectorized Batching** | 4 | **0.7690s** | **0.1923s / doc** | **3.89x faster** |

---

### Key Analytical Takeaways:

1. **BART Dominance:**
   `facebook/bart-large-cnn` outperformed all other models across all ROUGE dimensions:
   - **ROUGE-1 F1:** 36.38% (+6.47% over Lead-3, +9.22% over TextRank, +4.98% over T5-Small).
   - **ROUGE-2 F1:** 16.45% (+3.82% over Lead-3, +8.09% over TextRank, +4.22% over T5-Small).
   - **ROUGE-L F1:** 27.11% (+5.29% over Lead-3, +8.71% over TextRank, +4.40% over T5-Small).
2. **Precision vs. Recall Trade-off:**
   Extractive models achieve relatively high Recall because their summaries are long (62–66 words), capturing more reference words simply by producing more text. However, their Precision is low (20.77%–23.24%). In contrast, BART generates concise, targeted summaries (41.9 words) matching the human reference length, achieving the highest Precision (32.85%).
3. **Inference Latency & Batch Throughput:**
   Extractive methods execute on CPU in 0.1 to 3.6 milliseconds. Abstractive BART with FP16 on the RTX 3060 Laptop GPU generates summaries in **0.62s–0.91s** per document. With true batching enabled, throughput increases to **3.89x faster** for batch workloads.

---

### Visualizations

The following publication charts are saved in `outputs/figures/`:
1. **ROUGE Comparison:** `outputs/figures/rouge_comparison.png`  
   *Grouped bar chart illustrating ROUGE-1, ROUGE-2, and ROUGE-L F1 scores across the 4 systems.*
2. **Inference Latency:** `outputs/figures/model_latency_comparison.png`  
   *Runtime benchmark demonstrating the CPU vs. GPU throughput trade-offs.*
3. **Generated Summary Lengths:** `outputs/figures/generated_summary_lengths.png`  
   *Boxplot comparison showing how closely abstractive models mirror the human reference word count distribution compared to extractive baselines.*
4. **EDA Distributions:**  
   - `outputs/figures/article_length_dist.png`
   - `outputs/figures/summary_length_dist.png`
   - `outputs/figures/article_vs_summary_length.png`
   - `outputs/figures/compression_ratio_dist.png`

---

### Qualitative Case Study

**Document ID:** `f001ec5c4704938247d27a44948eebb37ae98d01`

* **Source Article Excerpt:**  
  *(CNN) The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands...*

* **Human Reference Highlights:**  
  > *Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June. Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis.*

* **Lead-3 Summary (Extractive):**  
  > *The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014."*

* **TextRank Summary (Extractive):**  
  > *The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. "What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members." "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement.*

* **T5-Small Summary (Abstractive):**  
  > *the Palestinian Authority officially became the 123rd member of the international criminal court. the ICC opened a preliminary examination into the situation in palestinians. a judge says acceding to the treaty is just the first step for the country.*

* **BART-Large-CNN Summary (Abstractive):**  
  > *The Palestinian Authority becomes the 123rd member of the International Criminal Court. The move gives the court jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the Palestinians' efforts to join the body.*

**Qualitative Observations:**
- **Lead-3** is accurate and readable, but includes unnecessary legal background from paragraph 3.
- **TextRank** captures opposing viewpoints, but quotes disjointed dialogue without clear contextual attribution.
- **T5-Small** produces grammatical summaries, but has occasional lower-case capitalization artifacts and minor semantic imprecision (*"in palestinians"*).
- **BART-Large-CNN** produces a virtually human-quality abstractive summary: it captures both the accession event, its legal jurisdiction consequence, and the opposition from Israel and the US, phrased with concise sentence boundaries.

---

## Unit Testing & Quality Assurance

The codebase features an automated test suite with 87 comprehensive tests executed with `pytest`:
- `tests/test_data_loader.py` (3 tests): Split loading, schema validation, length statistics.
- `tests/test_preprocessing.py` (7 tests): Wire artifact cleaning, whitespace normalization, punctuation handling, and tokenizer formatting.
- `tests/test_extractive.py` (9 tests): Lead-3, sentence similarity matrix, TextRank graph convergence, order preservation, candidate capping, disconnected graph fallback.
- `tests/test_textrank_stress.py` (6 tests): Graph convergence under extreme sentence counts (100, 500, 1000 sentences), identical sentences, whitespace-only, and ultra-long sentence inputs.
- `tests/test_summarizer.py` (15 tests): Model initialization, input formatting, empty input handling, generation presets, real T5 inference, true batching, granular batch latency metrics, warmup latency tracking, very long input truncation, real BART inference on CUDA, granular metrics, FP16/CPU fallback, tensor memory release on OOM, and restoration verification.
- `tests/test_evaluation.py` (5 tests): ROUGE-1/2/L score computations, edge cases (empty strings, length mismatches), metric aggregation.
- `tests/test_visualization.py` (4 tests): Headless plot rendering and artifact validation.
- `tests/test_app.py` (16 tests): UI Blocks creation, dynamic counters, empty input validation, extractive summarizer handlers, ModelManager caching metadata, thread-safe concurrent access, model comparison rejection, and legacy retrieval blocking.
- `tests/test_concurrency.py` (22 tests): Active lease eviction blocking, concurrent same-model leases, lease acquisition timeout, alternating multi-threaded requests, legacy retrieval timeout, concurrent single vs. batch inference synchronization, verified CUDA OOM batch sequential fallback, legacy inference eviction blocking, preprocessing eviction blocking, waiting-on-inference-lock eviction blocking, unload_model waiting for active inference, SafeModelProxy preventing use-after-unload, batch preprocessing and fallback safety, exception safety without leaked leases, proxy raw model/tokenizer access rejection, __getattr__ bound method/internals extraction rejection, safe metadata whitelist verification, weakref verification of zero retained raw model references, and instance attribute encapsulation checks.

Execute the full test suite:
```powershell
pytest -v
```

**Results:**
```text
======================= 87 passed in ~60s (100% pass rate) =======================
```

---

---

## Limitations & Future Work

1. **Context Window Boundary:**
   BART truncates input articles beyond 1,024 tokens. Long news stories exceeding 1,500 words lose concluding commentary. Future work will investigate Longformer or LED (Longformer Encoder-Decoder) to handle 4,096+ token sequences.
2. **Factual Faithfulness & Hallucination Auditing:**
   While ROUGE measures $n$-gram lexical overlap, it does not assess factual consistency. Future extensions will incorporate QA-based metrics (QAGS, FactCC) and semantic embeddings (BERTScore).
3. **Parameter-Efficient Fine-Tuning (PEFT):**
   Fine-tuning BART or larger LLMs (e.g. Llama-3, Mistral) on domain-specific corpora using LoRA / QLoRA with 4-bit quantization under the 6GB VRAM constraint.

---

## Author & Academic Portfolio
- **Project:** NewsLens - Natural Language Processing Text Summarization Pipeline & Web Application
- **Repository:** [https://github.com/abg11hani235-commits/NewsLens](https://github.com/abg11hani235-commits/NewsLens)
- **License:** MIT License
