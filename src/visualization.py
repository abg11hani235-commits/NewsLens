"""Visualization module for NLP Text Summarization Project.

Provides publication-quality plots using Matplotlib and Seaborn for:
- Exploratory Data Analysis (length distributions, scatter correlations, compression ratios).
- Model evaluation and benchmark comparisons (ROUGE scores, inference latency).
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive headless backend to prevent Tkinter GUI errors
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_article_length_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    split_name: str = "Train",
) -> Path:
    """Plot histogram and KDE distribution of article word counts.

    Args:
        df (pd.DataFrame): Data containing 'article' column.
        save_path (Optional[Path]): File path to save the plot.
        split_name (str): Name of split for title/labeling.

    Returns:
        Path: Path to the saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "article_length_dist.png"

    word_counts = df["article"].apply(lambda x: len(str(x).split()))
    mean_len = word_counts.mean()
    median_len = word_counts.median()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        word_counts,
        kde=True,
        color="#2b5c8f",
        bins=35,
        ax=ax,
        edgecolor="white",
        alpha=0.7,
    )

    ax.axvline(
        mean_len,
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_len:.1f} words",
    )
    ax.axvline(
        median_len,
        color="#7570b3",
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_len:.1f} words",
    )

    # Highlight 1024 token reference threshold (approx ~750 words)
    ax.axvline(
        750,
        color="#e7298a",
        linestyle="-.",
        linewidth=1.5,
        label="BART 1024-token limit (~750 words)",
    )

    ax.set_title(
        f"CNN/DailyMail - Article Word Count Distribution ({split_name} Split, N={len(df)})",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Article Length (Word Count)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_summary_length_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    split_name: str = "Train",
) -> Path:
    """Plot histogram and KDE distribution of reference summary (highlights) word counts.

    Args:
        df (pd.DataFrame): Data containing 'highlights' column.
        save_path (Optional[Path]): File path to save the plot.
        split_name (str): Name of split for title/labeling.

    Returns:
        Path: Path to the saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "summary_length_dist.png"

    word_counts = df["highlights"].apply(lambda x: len(str(x).split()))
    mean_len = word_counts.mean()
    median_len = word_counts.median()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        word_counts,
        kde=True,
        color="#1b9e77",
        bins=25,
        ax=ax,
        edgecolor="white",
        alpha=0.7,
    )

    ax.axvline(
        mean_len,
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_len:.1f} words",
    )
    ax.axvline(
        median_len,
        color="#7570b3",
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_len:.1f} words",
    )

    ax.set_title(
        f"CNN/DailyMail - Reference Summary Word Count Distribution ({split_name} Split, N={len(df)})",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Reference Summary Length (Word Count)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_article_vs_summary_length(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    split_name: str = "Train",
) -> Path:
    """Plot scatter relationship and regression trend between article and summary lengths.

    Args:
        df (pd.DataFrame): Data containing 'article' and 'highlights' columns.
        save_path (Optional[Path]): File path to save the plot.
        split_name (str): Name of split for title/labeling.

    Returns:
        Path: Path to the saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "article_vs_summary_length.png"

    plot_df = pd.DataFrame({
        "article_words": df["article"].apply(lambda x: len(str(x).split())),
        "summary_words": df["highlights"].apply(lambda x: len(str(x).split())),
    })
    corr = plot_df["article_words"].corr(plot_df["summary_words"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.regplot(
        data=plot_df,
        x="article_words",
        y="summary_words",
        color="#2b5c8f",
        scatter_kws={"alpha": 0.45, "s": 35},
        line_kws={"color": "#e41a1c", "linewidth": 2, "label": f"Linear Trend (Pearson r = {corr:.2f})"},
        ax=ax,
    )

    ax.set_title(
        f"Article Length vs. Summary Length ({split_name} Split, N={len(df)})",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Article Length (Word Count)")
    ax.set_ylabel("Summary Length (Word Count)")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_compression_ratio_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    split_name: str = "Train",
) -> Path:
    """Plot histogram and boxplot of compression ratios (article words / summary words).

    Args:
        df (pd.DataFrame): Data containing 'article' and 'highlights' columns.
        save_path (Optional[Path]): File path to save the plot.
        split_name (str): Name of split for title/labeling.

    Returns:
        Path: Path to the saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "compression_ratio_dist.png"

    article_words = df["article"].apply(lambda x: len(str(x).split()))
    summary_words = df["highlights"].apply(lambda x: len(str(x).split())).replace(0, 1)
    ratios = article_words / summary_words
    mean_ratio = ratios.mean()
    median_ratio = ratios.median()

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
    )

    sns.boxplot(x=ratios, ax=ax_box, color="#fd8d3c", showmeans=True)
    ax_box.set(xlabel="")
    ax_box.set_title(
        f"Compression Ratio (Article / Summary) Distribution ({split_name} Split, N={len(df)})",
        fontweight="bold",
        pad=10,
    )

    sns.histplot(
        ratios,
        kde=True,
        color="#e6550d",
        bins=30,
        ax=ax_hist,
        edgecolor="white",
        alpha=0.7,
    )
    ax_hist.axvline(
        mean_ratio,
        color="#2b5c8f",
        linestyle="--",
        linewidth=2,
        label=f"Mean Ratio: {mean_ratio:.1f}x",
    )
    ax_hist.axvline(
        median_ratio,
        color="#31a354",
        linestyle=":",
        linewidth=2,
        label=f"Median Ratio: {median_ratio:.1f}x",
    )

    ax_hist.set_xlabel("Compression Ratio (Article Words / Summary Words)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def generate_all_eda_plots(df: pd.DataFrame, split_name: str = "Train") -> List[Path]:
    """Generate and save all 4 EDA figures to outputs/figures/.

    Args:
        df (pd.DataFrame): Data containing 'article' and 'highlights'.
        split_name (str): Label for the split being analyzed.

    Returns:
        List[Path]: Paths to all generated image files.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_article_length_distribution(df, split_name=split_name),
        plot_summary_length_distribution(df, split_name=split_name),
        plot_article_vs_summary_length(df, split_name=split_name),
        plot_compression_ratio_distribution(df, split_name=split_name),
    ]
    return paths


def plot_rouge_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot grouped bar chart comparing ROUGE-1, ROUGE-2, and ROUGE-L across models.

    Args:
        metrics_df (pd.DataFrame): Dataframe containing 'Model', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1'.
        save_path (Optional[Path]): Destination path.

    Returns:
        Path: Path to saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "rouge_comparison.png"

    models = metrics_df["Model"].tolist()
    rouge1 = metrics_df["ROUGE-1 F1"].tolist()
    rouge2 = metrics_df["ROUGE-2 F1"].tolist()
    rougeL = metrics_df["ROUGE-L F1"].tolist()

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width, rouge1, width, label="ROUGE-1 F1", color="#2b5c8f", edgecolor="white")
    rects2 = ax.bar(x, rouge2, width, label="ROUGE-2 F1", color="#1b9e77", edgecolor="white")
    rects3 = ax.bar(x + width, rougeL, width, label="ROUGE-L F1", color="#d95f02", edgecolor="white")

    ax.set_ylabel("F1 Score (%)")
    ax.set_title("ROUGE Score Benchmark Across Extractive & Abstractive Models", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, max(max(rouge1), 45) * 1.15)

    # Add score labels on bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_model_latency_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot bar chart comparing average inference latency per document.

    Args:
        metrics_df (pd.DataFrame): Dataframe containing 'Model' and 'Avg Latency (s)'.
        save_path (Optional[Path]): Destination path.

    Returns:
        Path: Path to saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "model_latency_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#7570b3", "#fd8d3c", "#386cb0", "#e41a1c"]
    bars = ax.bar(
        metrics_df["Model"],
        metrics_df["Avg Latency (s)"],
        color=colors[: len(metrics_df)],
        edgecolor="white",
        width=0.55,
    )

    ax.set_ylabel("Inference Latency per Document (Seconds)")
    ax.set_title("Inference Latency Comparison (Batch Size = 1)", fontweight="bold", pad=12)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def plot_summary_length_comparison(
    comp_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot boxplot comparing word counts of generated summaries against human reference.

    Args:
        comp_df (pd.DataFrame): Dataframe with columns 'human_reference', 'lead3_summary',
                                'textrank_summary', 't5_summary', 'bart_summary'.
        save_path (Optional[Path]): Destination path.

    Returns:
        Path: Path to saved figure.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "generated_summary_lengths.png"

    length_data = []
    columns_map = [
        ("Human Highlights", "human_reference"),
        ("Lead-3", "lead3_summary"),
        ("TextRank", "textrank_summary"),
        ("T5-Small", "t5_summary"),
        ("BART-Large-CNN", "bart_summary"),
    ]

    for label, col in columns_map:
        if col in comp_df.columns:
            lengths = comp_df[col].apply(lambda s: len(str(s).split())).tolist()
            for length in lengths:
                length_data.append({"Model": label, "Word Count": length})

    len_df = pd.DataFrame(length_data)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(
        data=len_df,
        x="Model",
        y="Word Count",
        palette="Set2",
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"},
    )

    ax.set_title("Generated Summary Word Count Distribution vs. Human Reference", fontweight="bold", pad=12)
    ax.set_ylabel("Word Count")
    ax.set_xlabel("Architecture / Method")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def generate_all_evaluation_plots(
    metrics_df: pd.DataFrame,
    comp_df: pd.DataFrame,
) -> List[Path]:
    """Generate all 3 benchmark evaluation charts in outputs/figures/.

    Args:
        metrics_df (pd.DataFrame): ROUGE and latency scores table.
        comp_df (pd.DataFrame): Document-level predictions table.

    Returns:
        List[Path]: Paths to generated evaluation plots.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_rouge_comparison(metrics_df),
        plot_model_latency_comparison(metrics_df),
        plot_summary_length_comparison(comp_df),
    ]
    return paths

    """Generate and save all 4 EDA figures to outputs/figures/.

    Args:
        df (pd.DataFrame): Data containing 'article' and 'highlights'.
        split_name (str): Label for the split being analyzed.

    Returns:
        List[Path]: Paths to all generated image files.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_article_length_distribution(df, split_name=split_name),
        plot_summary_length_distribution(df, split_name=split_name),
        plot_article_vs_summary_length(df, split_name=split_name),
        plot_compression_ratio_distribution(df, split_name=split_name),
    ]
    return paths


def perform_full_eda() -> Dict[str, Any]:
    """Execute complete exploratory data analysis on the acquired dataset splits.

    Computes comprehensive statistics, checks for missing/duplicate records,
    generates all 4 publication figures, and persists numerical results.

    Returns:
        Dict[str, Any]: Detailed EDA metrics dictionary.
    """
    import json
    from src.data_loader import load_raw_split
    from config import METRICS_DIR

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Executing Exploratory Data Analysis (EDA) on CNN/DailyMail")
    print("=" * 70)

    splits = ["train", "validation", "test"]
    eda_results = {}
    summary_rows = []

    for split in splits:
        df = load_raw_split(split)
        num_records = len(df)
        missing_articles = int(df["article"].isnull().sum())
        missing_summaries = int(df["highlights"].isnull().sum())
        duplicate_ids = int(df["id"].duplicated().sum())
        duplicate_articles = int(df["article"].duplicated().sum())

        art_words = df["article"].apply(lambda x: len(str(x).split()))
        sum_words = df["highlights"].apply(lambda x: len(str(x).split()))
        art_chars = df["article"].apply(lambda x: len(str(x)))
        sum_chars = df["highlights"].apply(lambda x: len(str(x)))
        ratios = art_words / sum_words.replace(0, 1)
        corr = float(art_words.corr(sum_words))

        # Vocabulary size
        all_art_tokens = set(" ".join(df["article"].str.lower()).split())
        all_sum_tokens = set(" ".join(df["highlights"].str.lower()).split())

        split_metrics = {
            "record_count": num_records,
            "missing_values": {
                "article": missing_articles,
                "highlights": missing_summaries,
            },
            "duplicates": {
                "id": duplicate_ids,
                "article": duplicate_articles,
            },
            "article_words": {
                "min": int(art_words.min()),
                "max": int(art_words.max()),
                "mean": round(float(art_words.mean()), 2),
                "median": round(float(art_words.median()), 2),
                "std": round(float(art_words.std()), 2),
                "p25": round(float(art_words.quantile(0.25)), 2),
                "p75": round(float(art_words.quantile(0.75)), 2),
                "p95": round(float(art_words.quantile(0.95)), 2),
            },
            "summary_words": {
                "min": int(sum_words.min()),
                "max": int(sum_words.max()),
                "mean": round(float(sum_words.mean()), 2),
                "median": round(float(sum_words.median()), 2),
                "std": round(float(sum_words.std()), 2),
                "p25": round(float(sum_words.quantile(0.25)), 2),
                "p75": round(float(sum_words.quantile(0.75)), 2),
                "p95": round(float(sum_words.quantile(0.95)), 2),
            },
            "compression_ratio": {
                "min": round(float(ratios.min()), 2),
                "max": round(float(ratios.max()), 2),
                "mean": round(float(ratios.mean()), 2),
                "median": round(float(ratios.median()), 2),
                "std": round(float(ratios.std()), 2),
            },
            "article_summary_word_corr": round(corr, 4),
            "vocab_size": {
                "article_unique_tokens": len(all_art_tokens),
                "summary_unique_tokens": len(all_sum_tokens),
            },
        }

        eda_results[split] = split_metrics

        summary_rows.append({
            "split": split,
            "records": num_records,
            "missing_article": missing_articles,
            "missing_summary": missing_summaries,
            "art_words_mean": split_metrics["article_words"]["mean"],
            "art_words_median": split_metrics["article_words"]["median"],
            "art_words_min": split_metrics["article_words"]["min"],
            "art_words_max": split_metrics["article_words"]["max"],
            "sum_words_mean": split_metrics["summary_words"]["mean"],
            "sum_words_median": split_metrics["summary_words"]["median"],
            "sum_words_min": split_metrics["summary_words"]["min"],
            "sum_words_max": split_metrics["summary_words"]["max"],
            "compression_ratio_mean": split_metrics["compression_ratio"]["mean"],
            "compression_ratio_median": split_metrics["compression_ratio"]["median"],
            "pearson_r": split_metrics["article_summary_word_corr"],
        })

    # Save summary statistics to CSV and JSON
    stats_df = pd.DataFrame(summary_rows)
    csv_path = METRICS_DIR / "eda_summary_statistics.csv"
    json_path = METRICS_DIR / "eda_summary_statistics.json"
    stats_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda_results, f, indent=2)

    print(f"\nSaved statistical reports to:\n  - {csv_path}\n  - {json_path}")

    # Generate the 4 figures using the primary training split
    train_df = load_raw_split("train")
    generated_plots = generate_all_eda_plots(train_df, split_name="Train")
    print("\nGenerated 4 EDA Visualizations in outputs/figures/:")
    for p in generated_plots:
        print(f"  - {p.name}")

    return eda_results


if __name__ == "__main__":
    results = perform_full_eda()

