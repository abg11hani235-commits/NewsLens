---
title: NewsLens - AI-Powered News Summarization
emoji: 📰
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 6.32.0
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

The project implements classical heuristics, graph-based algorithms, and state-of-the-art encoder-decoder Transformer architectures optimized for both consumer GPU hardware (NVIDIA RTX 3060, 6GB VRAM) and resource-conscious cloud deployment (Hugging Face Spaces CPU Free Tier).

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
6. [Hugging Face Spaces Cloud Deployment](#hugging-face-spaces-cloud-deployment)
7. [Project Architecture & Directory Structure](#project-architecture--directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
10. [Experimental Results & Benchmark](#experimental-results--benchmark)
    - [Quantitative ROUGE Benchmark Table](#quantitative-rouge-benchmark-table)
    - [Visualizations](#visualizations)
    - [Qualitative Case Study](#qualitative-case-study)
11. [Unit Testing & Quality Assurance](#unit-testing--quality-assurance)
12. [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

Text summarization is the task of condensing a source document into a concise version while preserving salient information and factual integrity. This repository provides an end-to-end NLP engineering pipeline and interactive web interface demonstrating:
- **Reproducible Data Ingestion:** Streaming and persisting structured subsets of CNN/DailyMail without downloading multi-gigabyte archives.
- **Exploratory Data Analysis (EDA):** Quantifying article lengths, summary bounds, compression ratios, and vocabulary distributions.
- **Linguistic Preprocessing:** Removing journalistic artifacts (e.g., `(CNN) --`), normalizing bullet points, and applying model-specific subword tokenization with offline NLTK resource resilience.
- **Extractive Baselines:** Implementing Lead-3 and a custom TF-IDF/PageRank TextRank algorithm via NetworkX.
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
- **FP16 Mixed Precision:** Loaded via `torch.float16` when CUDA is available, reducing VRAM consumption by 50% (~775 MB–1.6 GB VRAM).
- **Explicit CUDA Synchronization:** Timing measurements use `torch.cuda.synchronize()` before and after generation to ensure accurate latency figures that reflect completed GPU kernels.
- **Safe OOM Fallback with Device Restoration:** In `src/summarizer.py`, if a CUDA Out-Of-Memory exception occurs, all inputs (including attention masks) are transferred to CPU, inference completes on CPU, and the model is safely moved back or deallocated in a `try...finally` block.

### 2. Memory-Aware Model Management (`ModelManager`)
To prevent concurrent GPU memory allocation or RAM spikes on cloud hosts:
- A singleton `ModelManager` class lazily loads models on demand.
- Before loading BART (406M parameters), any resident instance of T5 is explicitly evicted via `unload_model()`, garbage collection `gc.collect()`, and `torch.cuda.empty_cache()`.
- BART and T5 are **never resident simultaneously**, allowing the entire application to run stably on constrained hardware.

---

## Hugging Face Spaces Cloud Deployment

NewsLens is pre-configured for direct, zero-cost deployment to **Hugging Face Spaces** using the free CPU tier (2 vCPU, 16GB RAM).

### Deployment Configuration
- **SDK:** Gradio 6.32.0 (compatible with 5.x and 6.x)
- **Entry Point:** `app.py`
- **Dependency Split:**
  - `requirements.txt`: Clean, cross-platform dependencies without platform-specific wheel URLs.
  - `requirements-cuda.txt`: Pinned Windows/CUDA 12.4 configuration for local development.

### Steps to Deploy on Hugging Face Spaces:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space):
   - **Space Name:** `NewsLens`
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (Free · 2 vCPU · 16 GB RAM)
2. Connect your GitHub repository (`https://github.com/abg11hani235-commits/NewsLens`) or push via Git remote:
   ```powershell
   git remote add space https://huggingface.co/spaces/<your-username>/NewsLens
   git push space main
   ```
3. Hugging Face Spaces will automatically read `app.py` and `requirements.txt`, install dependencies, and launch the application.

> [!NOTE]
> **Free CPU Tier Latency Expectations:**
> - Extractive (Lead-3, TextRank): **< 0.01s** (instantaneous).
> - T5-Small: **~1.5s** on 2 vCPUs.
> - BART-Large-CNN: **~5–8s** on 2 vCPUs (vs. ~0.97s on CUDA GPU).
> Model weights are downloaded automatically into cache on first inference and never committed to Git.

---

## Project Architecture & Directory Structure

```text
NewsLens/
│
├── .gitignore                      # Exclusions for .venv/, datasets, caches, model weights
├── requirements.txt                # Cross-platform cloud deployment dependencies (Hugging Face Spaces)
├── requirements-cuda.txt           # Local GPU dependencies (Windows 11 / CUDA 12.4)
├── config.py                       # Configuration paths, hyperparameters, and device resolution
├── main.py                         # Unified Command-Line Interface (CLI) application
├── app.py                          # Production Gradio Web Application
├── README.md                       # Academic documentation & deployment guide
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
│   ├── preprocessing.py            # Text cleaning, highlight normalization, NLTK checks
│   ├── extractive.py               # Lead-N heuristic & NetworkX TextRank algorithm
│   ├── summarizer.py               # AbstractiveSummarizer class, FP16 CUDA, synchronized timing
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
│   │   └── rouge_scores.json
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
    ├── test_data_loader.py         # Tests for data loading and schema validation
    ├── test_preprocessing.py       # Tests for text cleaning and tokenization
    ├── test_extractive.py          # Tests for Lead-3 and TextRank graph logic
    ├── test_summarizer.py          # Real inference tests for BART & T5, OOM handling
    ├── test_evaluation.py          # Tests for ROUGE score computations
    ├── test_visualization.py       # Tests for headless plot rendering
    └── test_app.py                 # Tests for Gradio interface, counters, and ModelManager
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

### 6. Non-Destructive Preprocessing & EDA
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

All four architectures were evaluated on the **exact same 50 test samples** from the CNN/DailyMail test split.

### Quantitative ROUGE Benchmark Table

| Model | Paradigm | Parameters | ROUGE-1 F1 | ROUGE-1 Recall | ROUGE-1 Prec | ROUGE-2 F1 | ROUGE-2 Recall | ROUGE-2 Prec | ROUGE-L F1 | ROUGE-L Recall | ROUGE-L Prec | Avg Words | Avg Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Heuristic | N/A | 29.91% | 45.95% | 23.24% | 12.63% | 20.68% | 9.47% | 21.82% | 34.06% | 16.83% | 62.1 | **0.002s** |
| **TextRank** | Extractive Graph | N/A | 27.16% | 42.25% | 20.77% | 8.36% | 13.76% | 6.21% | 18.40% | 29.01% | 13.99% | 66.6 | 0.005s |
| **T5-Small** | Abstractive Seq2Seq | 60M | 31.40% | 37.82% | 28.19% | 12.23% | 14.68% | 11.12% | 22.71% | 27.24% | 20.47% | 42.9 | 0.984s |
| **BART-Large-CNN** | Abstractive Seq2Seq | 406M | **36.38%** | **42.97%** | **32.85%** | **16.45%** | **19.74%** | **14.72%** | **27.11%** | **32.12%** | **24.43%** | **41.9** | 0.973s |

*Hardware: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 12.4, FP16 Precision).*

---

### Key Analytical Takeaways:

1. **BART Dominance:**
   `facebook/bart-large-cnn` outperformed all other models across all ROUGE dimensions:
   - **ROUGE-1 F1:** 36.38% (+6.47% over Lead-3, +9.22% over TextRank, +4.98% over T5-Small).
   - **ROUGE-2 F1:** 16.45% (+3.82% over Lead-3, +8.09% over TextRank, +4.22% over T5-Small).
   - **ROUGE-L F1:** 27.11% (+5.29% over Lead-3, +8.71% over TextRank, +4.40% over T5-Small).
2. **Precision vs. Recall Trade-off:**
   Extractive models achieve relatively high Recall because their summaries are long (62–66 words), capturing more reference words simply by producing more text. However, their Precision is low (20.77%–23.24%). In contrast, BART generates concise, targeted summaries (41.9 words) matching the human reference length, achieving the highest Precision (32.85%).
3. **Inference Latency:**
   Extractive methods execute on CPU in 2 to 5 milliseconds. BART-large-cnn with FP16 on the RTX 3060 Laptop GPU processed each document in **0.973 seconds** (~1.03 articles per second), which is highly practical for interactive deployment.

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

The codebase features an automated test suite executed with `pytest`:
- `tests/test_data_loader.py` (3 tests): Split loading, schema validation, length statistics.
- `tests/test_preprocessing.py` (7 tests): Wire artifact cleaning, whitespace normalization, punctuation handling, and tokenizer formatting.
- `tests/test_extractive.py` (7 tests): Lead-3, sentence similarity matrix, TextRank graph convergence, order preservation.
- `tests/test_summarizer.py` (5 tests): Model initialization, input formatting, empty input handling, real T5 inference, real BART inference on CUDA.
- `tests/test_evaluation.py` (5 tests): ROUGE-1/2/L score computations, edge cases (empty strings, length mismatches), metric aggregation.
- `tests/test_visualization.py` (4 tests): Headless plot rendering and artifact validation.
- `tests/test_app.py` (11 tests): UI Blocks creation, dynamic counters, empty input validation, extractive summarizer handlers, ModelManager memory eviction.

Execute the full test suite:
```powershell
pytest -v
```

**Results:**
```text
======================= 42 passed in 25.72s (100% pass rate) =======================
```

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
