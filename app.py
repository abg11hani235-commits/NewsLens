"""NewsLens - AI-Powered News Summarization Web Application.

A production-grade Gradio application comparing extractive (Lead-3, TextRank)
and pre-trained abstractive (BART, T5) sequence-to-sequence summarization models.

Features:
- Single-Article Summarization with real-time resource tracking (latency, tokens, device).
- Side-by-Side Model Comparison with memory-conscious sequential loading.
- Pre-loaded real-world news articles for one-click testing.
- Interactive Academic Benchmark & Architecture Guide.
"""

import sys
import gc
import time
import inspect
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import gradio as gr

from config import (
    PRIMARY_MODEL_NAME,
    ALTERNATIVE_MODEL_NAME,
    BART_MAX_INPUT_LENGTH,
    T5_MAX_INPUT_LENGTH,
    get_device,
)
from src.preprocessing import clean_text, ensure_nltk_resources, get_text_length_info
from src.extractive import lead_n_summarize, textrank_summarize
from src.summarizer import AbstractiveSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("newslens_app")

# Ensure NLTK resources
ensure_nltk_resources()

# ==============================================================================
# Model Manager (Memory-Conscious Singleton & Lazy Loading)
# ==============================================================================

class ModelManager:
    """Manages active neural summarization models with lazy loading and memory deallocation.

    Ensures that BART and T5 are never resident simultaneously in memory, protecting
    6GB VRAM GPUs and 16GB CPU cloud hosting environments from OOM crashes.
    """

    def __init__(self):
        self.active_model_name: Optional[str] = None
        self.active_summarizer: Optional[AbstractiveSummarizer] = None

    def get_model(self, model_choice: str) -> AbstractiveSummarizer:
        """Retrieve the requested summarizer, unloading any conflicting model.

        Args:
            model_choice (str): 'bart' or 't5'.

        Returns:
            AbstractiveSummarizer: Configured and ready model instance.
        """
        model_key = model_choice.lower().strip()
        if model_key not in ["bart", "t5"]:
            raise ValueError(f"Invalid model choice '{model_choice}'. Choose 'bart' or 't5'.")

        target_model_name = PRIMARY_MODEL_NAME if model_key == "bart" else ALTERNATIVE_MODEL_NAME

        # If already loaded and active, return cached instance
        if self.active_summarizer is not None and self.active_model_name == model_key:
            return self.active_summarizer

        # Unload any previously loaded model to preserve memory
        if self.active_summarizer is not None:
            logger.info(f"Evicting active model '{self.active_model_name}' to free memory...")
            self.active_summarizer.unload_model()
            self.active_summarizer = None
            self.active_model_name = None

        logger.info(f"Loading '{target_model_name}' on demand...")
        summarizer = AbstractiveSummarizer(
            model_name=target_model_name,
            device=get_device(),
        )
        self.active_model_name = model_key
        self.active_summarizer = summarizer
        return self.active_summarizer

    def unload_all(self) -> None:
        """Unload active model completely."""
        if self.active_summarizer is not None:
            self.active_summarizer.unload_model()
            self.active_summarizer = None
            self.active_model_name = None


# Global singleton instance
MODEL_MANAGER = ModelManager()

# ==============================================================================
# Sample Benchmark News Articles
# ==============================================================================

SAMPLE_ARTICLES = {
    "Diplomacy / International (ICC Accession)": (
        "The Palestinian Authority officially became the 123rd member of the International Criminal Court "
        "on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. "
        "The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. "
        "The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction "
        "over alleged crimes committed in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014. "
        "Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis. "
        "The ICC opened a preliminary examination into the situation in Palestinian territories in January, "
        "shortly after the Palestinians signed the Rome Statute."
    ),
    "Technology / AI Regulation (Brussels Agreement)": (
        "The European Union agreed on landmark artificial intelligence regulations today in Brussels after marathon "
        "negotiations between member states and lawmakers. Officials hailed the legislation as a historic step for consumer safety, "
        "establishing strict guardrails for high-risk AI applications in healthcare, employment, and biometric identification. "
        "Under the new framework, foundation models and general-purpose systems must adhere to strict transparency standards, "
        "technical documentation audits, and copyright compliance. Technology industry leaders expressed cautious optimism, "
        "while highlighting concerns over compliance burdens and European competitiveness in global research."
    ),
    "Science / Space Exploration (James Webb Discovery)": (
        "Astronomers analyzing infrared imagery from NASA's James Webb Space Telescope have confirmed the discovery of an "
        "ancient galaxy formed just 300 million years after the Big Bang. The galaxy, designated JADES-GS-z14-0, exhibits "
        "surprising brightness and mass, challenging prevailing theoretical models of early cosmic structure formation. "
        "Spectroscopic data revealed that the galaxy contains significant amounts of oxygen and ionized dust, indicating that "
        "multiple generations of massive stars had already lived and died in the earliest epoch of the universe. Researchers "
        "believe this demonstrates that early galaxy development occurred much faster than previously estimated."
    ),
}

# ==============================================================================
# Core Interface Inference Functions
# ==============================================================================

def update_text_counts(text: str) -> str:
    """Dynamically compute word and character counts as the user types."""
    if not text:
        return "Word Count: 0 | Character Count: 0"
    words = len(text.split())
    chars = len(text)
    return f"Word Count: {words:,} | Character Count: {chars:,}"


def summarize_article(
    article_text: str,
    model_choice: str,
    num_sentences: int = 3,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4,
) -> Tuple[str, str, str, str, str]:
    """Execute summarization for single document based on model selection.

    Returns:
        Tuple: (Summary Text, Latency String, Device String, Word Count String, Compression Ratio String)
    """
    if not article_text or not article_text.strip():
        raise gr.Error("Please provide an article to summarize.")

    cleaned = clean_text(article_text)
    input_words = len(cleaned.split())

    if input_words < 10:
        raise gr.Error("Input text is too short. Please provide at least 10 words for a meaningful summary.")

    model_clean = model_choice.lower().strip()

    try:
        if "lead-3" in model_clean or "lead3" in model_clean:
            t0 = time.perf_counter()
            summary = lead_n_summarize(cleaned, num_sentences=num_sentences)
            elapsed = time.perf_counter() - t0
            device = "CPU (Heuristic)"

        elif "textrank" in model_clean:
            t0 = time.perf_counter()
            summary = textrank_summarize(cleaned, num_sentences=num_sentences)
            elapsed = time.perf_counter() - t0
            device = "CPU (PageRank Graph)"

        elif "bart" in model_clean:
            summarizer = MODEL_MANAGER.get_model("bart")
            res = summarizer.summarize_text(
                cleaned,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
            )
            summary = res["summary"]
            elapsed = res["latency_sec"]
            device = res["device"]

        elif "t5" in model_clean:
            summarizer = MODEL_MANAGER.get_model("t5")
            res = summarizer.summarize_text(
                cleaned,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
            )
            summary = res["summary"]
            elapsed = res["latency_sec"]
            device = res["device"]

        else:
            raise gr.Error(f"Unsupported model: {model_choice}")

        out_words = len(summary.split())
        ratio = round(input_words / max(out_words, 1), 1)

        latency_display = f"{elapsed:.3f} seconds"
        word_count_display = f"{out_words} words (Original: {input_words} words)"
        ratio_display = f"{ratio}x Compression"

        return summary, latency_display, device, word_count_display, ratio_display

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise gr.Error(f"Summarization Error: {str(e)}")


def compare_all_models(
    article_text: str,
    num_sentences: int = 3,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4,
) -> Tuple[str, str, str, str, str]:
    """Execute sequential memory-aware comparison across Lead-3, TextRank, T5, and BART.

    Returns:
        Tuple: (Lead-3 Output, TextRank Output, T5 Output, BART Output, Comparison Table Markdown)
    """
    if not article_text or not article_text.strip():
        raise gr.Error("Please provide an article to compare models.")

    cleaned = clean_text(article_text)
    input_words = len(cleaned.split())

    if input_words < 10:
        raise gr.Error("Article text is too short to benchmark.")

    logger.info("Executing sequential model comparison...")

    # 1. Lead-3 (CPU)
    t0 = time.perf_counter()
    lead3_sum = lead_n_summarize(cleaned, num_sentences=num_sentences)
    lead3_lat = time.perf_counter() - t0

    # 2. TextRank (CPU)
    t0 = time.perf_counter()
    textrank_sum = textrank_summarize(cleaned, num_sentences=num_sentences)
    textrank_lat = time.perf_counter() - t0

    # 3. T5-Small (Lazy load, run, then unload before BART)
    t5_model = MODEL_MANAGER.get_model("t5")
    t5_res = t5_model.summarize_text(
        cleaned,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
    )
    t5_sum = t5_res["summary"]
    t5_lat = t5_res["latency_sec"]
    t5_dev = t5_res["device"]

    # 4. BART-Large-CNN (ModelManager evicts T5 to avoid memory conflict)
    bart_model = MODEL_MANAGER.get_model("bart")
    bart_res = bart_model.summarize_text(
        cleaned,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
    )
    bart_sum = bart_res["summary"]
    bart_lat = bart_res["latency_sec"]
    bart_dev = bart_res["device"]

    # Construct comparison summary table
    table_md = f"""
| Architecture | Paradigm | Parameters | Latency | Summary Words | Compression | Device |
|---|---|---|---|---|---|---|
| **Lead-3** | Extractive Heuristic | N/A | {lead3_lat:.3f}s | {len(lead3_sum.split())} words | {input_words/max(len(lead3_sum.split()),1):.1f}x | CPU |
| **TextRank** | Extractive Graph | N/A | {textrank_lat:.3f}s | {len(textrank_sum.split())} words | {input_words/max(len(textrank_sum.split()),1):.1f}x | CPU |
| **T5-Small** | Abstractive Transformer | 60M | {t5_lat:.3f}s | {len(t5_sum.split())} words | {input_words/max(len(t5_sum.split()),1):.1f}x | {t5_dev} |
| **BART-Large-CNN** | Abstractive Transformer | 406M | {bart_lat:.3f}s | {len(bart_sum.split())} words | {input_words/max(len(bart_sum.split()),1):.1f}x | {bart_dev} |
"""
    return lead3_sum, textrank_sum, t5_sum, bart_sum, table_md


# ==============================================================================
# ==============================================================================
# UI Styling & Gradio Version Compatibility
# ==============================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
.header-box { text-align: center; margin-bottom: 1.5rem; }
.metric-card { background: #f8fafc; border-radius: 8px; padding: 12px; border: 1px solid #e2e8f0; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
)


def _get_copy_kwargs() -> dict:
    """Return appropriate copy button kwargs for Gradio 5 vs Gradio 6."""
    sig = inspect.signature(gr.Textbox.__init__).parameters
    if "buttons" in sig:
        return {"buttons": ["copy"]}
    if "show_copy_button" in sig:
        return {"show_copy_button": True}
    return {}


# ==============================================================================
# Gradio UI Construction
# ==============================================================================

def create_ui() -> gr.Blocks:
    """Build the modern, responsive Gradio Blocks interface for NewsLens."""
    copy_kwargs = _get_copy_kwargs()

    blocks_kwargs = {"title": "NewsLens - AI News Summarizer"}
    sig = inspect.signature(gr.Blocks.__init__).parameters
    # In Gradio <6 theme and css are passed to Blocks constructor
    if "theme" in sig and "kwargs" not in sig:
        blocks_kwargs["theme"] = THEME
    if "css" in sig and "kwargs" not in sig:
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 📰 NewsLens
                ### *AI-Powered News Summarization & Model Benchmarking*
                **Condense long news stories into accurate, fluent summaries using Pre-trained Transformers and Extractive Graph algorithms.**
                """
            )

        with gr.Tabs() as tabs:
            # ==================================================================
            # Tab 1: Single Model Summarizer
            # ==================================================================
            with gr.Tab("✨ Single Summarizer", id="tab_single"):
                with gr.Row():
                    with gr.Column(scale=6):
                        article_input = gr.Textbox(
                            label="Source News Article",
                            placeholder="Paste your news article here...",
                            lines=12,
                            max_lines=25,
                        )
                        count_display = gr.Markdown("Word Count: 0 | Character Count: 0")

                        # Sample selector
                        sample_dropdown = gr.Dropdown(
                            choices=list(SAMPLE_ARTICLES.keys()),
                            label="Load Sample News Article",
                            value=None,
                        )

                        with gr.Accordion("⚙️ Generation Parameters (Abstractive Models)", open=False):
                            with gr.Row():
                                num_beams_slider = gr.Slider(
                                    minimum=1, maximum=8, value=4, step=1, label="Beam Search Width (num_beams)"
                                )
                                max_length_slider = gr.Slider(
                                    minimum=32, maximum=256, value=128, step=8, label="Maximum Length (tokens)"
                                )
                                min_length_slider = gr.Slider(
                                    minimum=10, maximum=100, value=30, step=5, label="Minimum Length (tokens)"
                                )
                            sent_count_slider = gr.Slider(
                                minimum=1, maximum=6, value=3, step=1, label="Extractive Sentences (Lead-3 / TextRank)"
                            )

                    with gr.Column(scale=4):
                        model_radio = gr.Radio(
                            choices=[
                                "BART (facebook/bart-large-cnn)",
                                "T5 (google-t5/t5-small)",
                                "TextRank (Graph-based PageRank)",
                                "Lead-3 (Inverted Pyramid Heuristic)",
                            ],
                            value="BART (facebook/bart-large-cnn)",
                            label="Summarization Architecture",
                        )

                        with gr.Row():
                            summarize_btn = gr.Button("⚡ Generate Summary", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

                        summary_output = gr.Textbox(
                            label="Generated Summary",
                            lines=7,
                            **copy_kwargs,
                        )

                        with gr.Group(elem_classes=["metric-card"]):
                            gr.Markdown("**Performance & Resource Telemetry**")
                            with gr.Row():
                                latency_stat = gr.Textbox(label="Inference Latency", interactive=False)
                                device_stat = gr.Textbox(label="Compute Device", interactive=False)
                            with gr.Row():
                                word_stat = gr.Textbox(label="Summary Length", interactive=False)
                                ratio_stat = gr.Textbox(label="Compression Ratio", interactive=False)

                # Interactions
                article_input.change(fn=update_text_counts, inputs=article_input, outputs=count_display)

                def load_sample(choice):
                    return SAMPLE_ARTICLES.get(choice, "")

                sample_dropdown.change(fn=load_sample, inputs=sample_dropdown, outputs=article_input)

                summarize_btn.click(
                    fn=summarize_article,
                    inputs=[
                        article_input,
                        model_radio,
                        sent_count_slider,
                        max_length_slider,
                        min_length_slider,
                        num_beams_slider,
                    ],
                    outputs=[summary_output, latency_stat, device_stat, word_stat, ratio_stat],
                )

                def clear_all():
                    return "", "Word Count: 0 | Character Count: 0", "", "", "", "", ""

                clear_btn.click(
                    fn=clear_all,
                    outputs=[
                        article_input,
                        count_display,
                        summary_output,
                        latency_stat,
                        device_stat,
                        word_stat,
                        ratio_stat,
                    ],
                )

            # ==================================================================
            # Tab 2: Side-by-Side Model Comparison
            # ==================================================================
            with gr.Tab("⚖️ Model Comparison", id="tab_comparison"):
                gr.Markdown(
                    """
                    Compare how **Extractive** (Lead-3, TextRank) and **Abstractive** (T5, BART) paradigms
                    summarize the exact same document side-by-side.
                    *(Models are loaded sequentially to prevent memory exhaustion).*
                    """
                )
                comp_input = gr.Textbox(
                    label="Input Document for Multi-Model Benchmark",
                    placeholder="Paste article text or select a sample from Tab 1...",
                    lines=8,
                )
                comp_btn = gr.Button("🚀 Run Side-by-Side Benchmark", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. Lead-3 (Heuristic)")
                        comp_lead3 = gr.Textbox(label="Lead-3 Summary", lines=5, **copy_kwargs)
                    with gr.Column():
                        gr.Markdown("### 2. TextRank (Graph/PageRank)")
                        comp_textrank = gr.Textbox(label="TextRank Summary", lines=5, **copy_kwargs)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 3. T5-Small (Abstractive 60M)")
                        comp_t5 = gr.Textbox(label="T5 Summary", lines=5, **copy_kwargs)
                    with gr.Column():
                        gr.Markdown("### 4. BART-Large-CNN (Abstractive 406M)")
                        comp_bart = gr.Textbox(label="BART Summary", lines=5, **copy_kwargs)

                comp_table = gr.Markdown(label="Comparison Table")

                comp_btn.click(
                    fn=compare_all_models,
                    inputs=[comp_input],
                    outputs=[comp_lead3, comp_textrank, comp_t5, comp_bart, comp_table],
                )

            # ==================================================================
            # Tab 3: Academic Methodology & Benchmark
            # ==================================================================
            with gr.Tab("📚 Methodology & Academic Guide", id="tab_guide"):
                gr.Markdown(
                    """
                    ## Theoretical Framework: Extractive vs. Abstractive Summarization

                    | Dimension | Extractive Summarization (Lead-3 / TextRank) | Abstractive Summarization (BART / T5) |
                    |---|---|---|
                    | **Mechanism** | Selects verbatim sentences from source based on heuristic or graph centrality. | Generates novel phrasing and syntactic combinations using an autoregressive decoder. |
                    | **Factual Faithfulness** | Guaranteed 100% factual accuracy (zero hallucination risk). | High fluency; potential risk of factual hallucination if context is ambiguous. |
                    | **Linguistic Flow** | Sentence boundaries can feel disjointed or context-dependent. | Coherent discourse flow matching human writing conventions. |
                    | **Compute Profile** | Executed in milliseconds on standard CPU. | Requires Transformer matrix operations; accelerated via GPU FP16. |

                    ---

                    ## Official CNN/DailyMail ROUGE Benchmark (Tested on 50 Test Samples)

                    | Model | Paradigm | Parameters | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg Summary Words | Avg Latency |
                    |---|---|---|---|---|---|---|---|
                    | **Lead-3** | Extractive Heuristic | N/A | **29.91%** | **12.63%** | **21.82%** | 62.1 words | **0.002s** (CPU) |
                    | **TextRank** | Extractive Graph | N/A | **27.16%** | **8.36%** | **18.40%** | 66.6 words | **0.005s** (CPU) |
                    | **T5-Small** | Abstractive Transformer | 60M | **31.40%** | **12.23%** | **22.71%** | 42.9 words | **0.984s** (GPU) |
                    | **BART-Large-CNN** | Abstractive Transformer | 406M | **36.38%** | **16.45%** | **27.11%** | 41.9 words | **0.973s** (GPU) |

                    ---

                    ### Hardware Optimization & Deployment Notice
                    - **GPU Acceleration:** When deployed on CUDA hardware, neural inference runs in **FP16 mixed precision**, reducing memory allocation to ~775 MB–1.6 GB.
                    - **CPU Free Cloud Tier:** When hosted on free CPU cloud tiers (e.g., Hugging Face Spaces 2 vCPU), BART inference will take ~5–8 seconds per document. T5 and Extractive models remain near-instantaneous.

                    🔗 **Project GitHub Repository:** [https://github.com/abg11hani235-commits/NewsLens](https://github.com/abg11hani235-commits/NewsLens)
                    """
                )

    return demo


def launch_app(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the NewsLens Gradio application."""
    demo = create_ui()
    demo.queue()

    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "share": share,
    }
    sig = inspect.signature(gr.Blocks.launch).parameters
    if "theme" in sig:
        launch_kwargs["theme"] = THEME
    if "css" in sig:
        launch_kwargs["css"] = CUSTOM_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
