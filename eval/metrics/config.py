"""
Project-wide configuration for the CountVideoBench analysis.

Centralises:
- The benchmark's anchor-neighbourhood structure (anchors, cardinalities).
- The model display names used in the paper and the model order of the
  per-model colour palette.
- The matplotlib style.
- The default I/O paths.

All constants are gathered here so that other modules (data, metrics, plots),
the script and the notebook never hard-code values that may change between
runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Benchmark structure
# ---------------------------------------------------------------------------

#: Anchor cardinalities (Section 4.2 of the paper).
ANCHORS: tuple[int, ...] = (5, 10, 20, 30, 40, 50)

#: Half-width of the neighbourhood around each anchor.
NEIGHBOURHOOD_RADIUS: int = 2

#: Lower / upper bound of the cardinality range.
CARDINALITY_MIN: int = 5
CARDINALITY_MAX: int = 50

#: Number of videos per cardinality.
VIDEOS_PER_CARDINALITY: int = 100


def benchmark_cardinalities() -> list[int]:
    """
    Return the 26 cardinalities of CountVideoBench, in increasing order.

    Builds the union of {n-r, ..., n+r} for n in ANCHORS, clipped to
    [CARDINALITY_MIN, CARDINALITY_MAX].
    """
    values: set[int] = set()
    for anchor in ANCHORS:
        for offset in range(-NEIGHBOURHOOD_RADIUS, NEIGHBOURHOOD_RADIUS + 1):
            v = anchor + offset
            if CARDINALITY_MIN <= v <= CARDINALITY_MAX:
                values.add(v)
    return sorted(values)


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

#: Model name as written in the CSV files  ->  name displayed in the paper.
MODEL_NAME_REMAP: dict[str, str] = {
    "Adaretake-7B":   "AdaReTaKe-7B",
    "Adaretake-72B":  "AdaReTaKe-72B",
    "Gemini-3.1-Pro": "Gemini-3.1-pro",
}

#: Order of the models in the colour palette. It is the order used for the
#: figures of the paper, so that every model keeps its colour.
MODEL_ORDER: tuple[str, ...] = (
    "AdaReTaKe-72B",
    "AdaReTaKe-7B",
    "InternVL2_5-78B",
    "InternVL2_5-8B",
    "LLaVA-Video-72B",
    "LLaVA-Video-7B",
    "NVILA-15B",
    "NVILA-8B",
    "Qwen2.5-VL-72B",
    "Qwen2.5-VL-7B",
    "Qwen3-VL-32B",
    "Qwen3-VL-8B",
    "VideoLLaMA2-72B",
    "VideoLLaMA2-7B",
    "Gemini-3.1-pro",
    "GPT-5.4",
)


# ---------------------------------------------------------------------------
# Data filtering
# ---------------------------------------------------------------------------

#: Predictions above this value are aggregated into a single "61+" bucket
#: in the bias histograms.
BIAS_HISTOGRAM_MAX_BIN: int = 60


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------

#: Glob pattern locating the per-model CSV files (counts used in the paper).
DEFAULT_INPUT_PATTERN: str = str(ROOT.parent / "paper_results" / "*.csv")

#: Video ids of the 104-video subset of the human study (Appendix C).
HUMAN_SUBSET_PATH: Path = ROOT / "human_subset.csv"

#: Root directory under which every artefact (figure, CSV table) is written.
DEFAULT_OUTPUT_DIR: Path = ROOT / "output"

#: Sub-directory for the per-model bias histograms (PDFs).
BIAS_FIGURE_SUBDIR: str = "bias_barplots"

#: Sub-directory for the per-model prediction tables (CSVs).
BIAS_TABLE_SUBDIR: str = "bias_tables"


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

#: Global rcParams. Applied by ``apply_matplotlib_style()``.
RC_PARAMS: dict[str, object] = {
    "font.family":           "serif",
    "font.size":             13,
    "axes.titlesize":        17,
    "axes.labelsize":        15,
    "xtick.labelsize":       13,
    "ytick.labelsize":       13,
    "legend.fontsize":       13,
    "legend.title_fontsize": 15,
}


def apply_matplotlib_style() -> None:
    """Apply the project's global matplotlib rcParams."""
    plt.rcParams.update(RC_PARAMS)


def build_model_palette(models: list[str], cmap: str = "tab20") -> dict[str, tuple]:
    """
    Map each model name to a stable RGB colour.

    Using a dict (rather than relying on plotting-time ordering) guarantees
    that the same model always renders in the same colour across every figure
    of the paper.
    """
    colours = sns.color_palette(cmap, n_colors=len(models))
    return dict(zip(models, colours))
