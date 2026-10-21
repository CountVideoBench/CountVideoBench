"""
All matplotlib / seaborn figures of the CountVideoBench paper.

Three figure families are provided, mirroring Sections 6.1, 6.2, 6.3:

* ``plot_line_by_cardinality``   — Coverage curve (Fig. 3) and any other
  long-form ``y vs ground-truth-count`` plot, with one line per model and
  a publication-ready external legend.
* ``plot_accuracy_heatmap``      — Models-by-cardinalities heatmap with
  per-cell annotations (Fig. 4).
* ``plot_bias_barplot``          — Per-model stacked histogram of predicted
  counts (Fig. 5 and appendix Figs.).

All three accept an optional ``save_to`` path. In a notebook the figure is
then displayed; in a script (non-interactive backend) it is only saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch

from config import BIAS_HISTOGRAM_MAX_BIN
from data import COL_ANSWER, COL_GT, COL_MODEL


# Default visual settings shared across plots.
_STRIPE_ALPHA = 0.08
_STRIPE_COLOR = "0.2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_and_show(fig: plt.Figure, save_to: str | Path | None) -> None:
    """
    Save the figure as PDF if ``save_to`` is given, then display it, or
    close it when the backend cannot display figures (script).
    """
    if save_to is not None:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid double-extension when the caller already passed ".pdf".
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")
        fig.savefig(path, bbox_inches="tight")
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()


def _alternating_stripes(ax: plt.Axes, n_positions: int) -> None:
    """Paint faint vertical stripes every other x-tick (for legibility)."""
    for i in range(n_positions):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5,
                       color=_STRIPE_COLOR, alpha=_STRIPE_ALPHA, zorder=0)


# ---------------------------------------------------------------------------
# Coverage curve  (Fig. 3 of the paper)
# ---------------------------------------------------------------------------

def plot_line_by_cardinality(
    df_long: pd.DataFrame,
    *,
    y: str,
    y_label: str,
    x_label: str = "Ground-truth count",
    palette: Mapping[str, tuple] | None = None,
    percent: bool = True,
    ylim: tuple[float, float] | None = None,
    legend_ncol: int = 8,
    figsize: tuple[float, float] = (14, 5),
    save_to: str | Path | None = None,
) -> None:
    """
    One line per model, x = cardinality, y = ``df_long[y]``.

    Expects a long DataFrame with columns ``[model, "Ground-truth count", y]``.
    The legend is placed above the axes on multiple columns (publication
    layout used for Fig. 3).
    """
    df_long = df_long.copy()
    df_long[COL_GT] = pd.to_numeric(df_long[COL_GT])
    order = sorted(df_long[COL_GT].unique())
    x_map = {v: i for i, v in enumerate(order)}

    fig, ax = plt.subplots(figsize=figsize)
    _alternating_stripes(ax, len(order))

    for model, group in df_long.groupby(COL_MODEL):
        g = group.copy()
        g["_x"] = g[COL_GT].map(x_map)
        g = g.sort_values("_x")
        color = palette.get(model) if isinstance(palette, dict) else None
        ax.plot(g["_x"], g[y], marker="o", linewidth=2,
                label=model, color=color, zorder=3)

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if percent:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=legend_ncol,
        frameon=False,
        fontsize=12,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        markerscale=0.8,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.10, right=0.98)
    _save_and_show(fig, save_to)


# ---------------------------------------------------------------------------
# Accuracy heatmap  (Fig. 4 of the paper)
# ---------------------------------------------------------------------------

def plot_accuracy_heatmap(
    df_long: pd.DataFrame,
    *,
    value: str,
    cmap: str = "Blues",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    annotate: bool = True,
    fmt: str = ".0%",
    cbar: bool = False,
    cbar_label: str = "",
    model_order: Iterable[str] | None = None,
    count_order: Iterable[int] | None = None,
    save_to: str | Path | None = None,
) -> None:
    """
    Models-by-cardinalities heatmap.

    Parameters
    ----------
    df_long
        Long DataFrame with columns ``[model, "Ground-truth count", value]``.
    value
        Name of the column carrying the metric to plot (e.g. ``"exact_match"``).
    vmin, vmax, fmt
        Forwarded to ``seaborn.heatmap``. The defaults (``[0, 1]`` range and
        ``".0%"`` format) suit accuracy-style metrics.
    model_order, count_order
        Optional explicit orderings of the rows and columns.
    """
    d = df_long.copy()
    d[COL_GT] = d[COL_GT].astype(int)

    if count_order is None:
        count_order = sorted(d[COL_GT].unique())
    if model_order is None:
        model_order = sorted(d[COL_MODEL].unique())

    pivot = (
        d.pivot_table(index=COL_MODEL, columns=COL_GT,
                      values=value, aggfunc="mean")
         .reindex(index=list(model_order), columns=list(count_order))
    )

    fig_w = max(10, 0.6 * len(list(count_order)))
    fig_h = max(4, 0.5 * len(list(model_order)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    heatmap_kwargs = dict(
        cmap=cmap, annot=annotate, fmt=fmt,
        linewidths=0.5, linecolor="white",
        cbar=cbar, mask=pivot.isna(), ax=ax,
    )
    if vmin is not None:
        heatmap_kwargs["vmin"] = vmin
    if vmax is not None:
        heatmap_kwargs["vmax"] = vmax

    hm = sns.heatmap(pivot, **heatmap_kwargs)

    ax.set_xlabel(COL_GT)
    ax.set_ylabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    if cbar and hm.collections and hm.collections[0].colorbar is not None:
        cb = hm.collections[0].colorbar
        cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        cb.set_label(cbar_label)

    fig.tight_layout()
    _save_and_show(fig, save_to)


# ---------------------------------------------------------------------------
# Per-model prediction histogram  (Fig. 5 and appendix)
# ---------------------------------------------------------------------------

# Models for which the histogram's peak sits at the very right of the
# x-axis; for those, the upper-right legend would overlap the bars.
_LEGEND_LEFT_MODELS: frozenset[str] = frozenset({"NVILA-8B", "InternVL2_5-8B"})


def plot_bias_barplot(
    df: pd.DataFrame,
    model: str,
    *,
    in_dataset_cardinalities: Iterable[int],
    n_videos_per_cardinality: int = 100,
    figsize: tuple[float, float] = (14, 3),
    correct_color: str = "#2ecc71",
    incorrect_color: str = "#e74c3c",
    save_to: str | Path | None = None,
) -> None:
    """
    Stacked histogram of one model's predicted counts (Fig. 5 of the paper).

    Bars are stacked into a green "Correct" part and a red "Incorrect"
    part. Integer values outside the benchmark grid are shaded grey to
    make the on-grid vs off-grid structure visible at a glance. A dashed
    horizontal line at ``n_videos_per_cardinality`` provides a reference
    for what "always answer N at every ground-truth N" would look like.

    Parameters
    ----------
    df
        Preprocessed long DataFrame.
    model
        Model name to plot.
    in_dataset_cardinalities
        Iterable of integer cardinalities that actually appear in the
        ground truth (used only for the grey shading).
    n_videos_per_cardinality
        Reference line height. Defaults to 100 (CountVideoBench).
    """
    sub = df[df[COL_MODEL] == model].copy()
    pred = pd.to_numeric(sub[COL_ANSWER], errors="coerce")
    gt = pd.to_numeric(sub[COL_GT], errors="coerce")
    sub["correct"] = pred.eq(gt)
    sub = sub.dropna(subset=[COL_ANSWER])

    overflow_label = f"{BIAS_HISTOGRAM_MAX_BIN + 1}+"
    sub["answer_binned"] = sub[COL_ANSWER].apply(
        lambda x: str(int(x)) if x <= BIAS_HISTOGRAM_MAX_BIN else overflow_label
    )

    counts = (
        sub.groupby(["answer_binned", "correct"])
           .size()
           .unstack(fill_value=0)
    )
    ordered_index = [str(i) for i in range(0, BIAS_HISTOGRAM_MAX_BIN + 1)] + [overflow_label]
    counts = counts.reindex(ordered_index, fill_value=0)
    counts = counts.reindex(columns=[False, True], fill_value=0)
    counts = counts.rename(columns={True: "Correct", False: "Incorrect"})

    fig, ax = plt.subplots(figsize=figsize)
    counts.plot.bar(stacked=True, color=[incorrect_color, correct_color],
                    ax=ax, width=0.8)

    # Grey out integer positions that are not on the benchmark grid.
    in_dataset = set(in_dataset_cardinalities)
    for i, val in enumerate(range(0, BIAS_HISTOGRAM_MAX_BIN + 1)):
        if val not in in_dataset:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.3, zorder=0)
    overflow_pos = BIAS_HISTOGRAM_MAX_BIN + 1
    if not any(n > BIAS_HISTOGRAM_MAX_BIN for n in in_dataset):
        ax.axvspan(overflow_pos - 0.5, overflow_pos + 0.5,
                   color="gray", alpha=0.3, zorder=0)

    ax.set_xlabel("Answers")
    ax.set_ylabel("Occurrences")
    ax.set_xlim(-0.5, overflow_pos + 0.5)
    ax.grid(True, alpha=0.3, axis="y")

    ax.axhline(
        y=n_videos_per_cardinality,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Number of videos",
    )

    # Legend entry for the grey areas.
    out_of_dataset_patch = Patch(
        facecolor="gray",
        alpha=0.3,
        label="Not in benchmark",
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(out_of_dataset_patch)

    legend_loc = "upper left" if model in _LEGEND_LEFT_MODELS else "upper right"
    ax.legend(handles=handles, loc=legend_loc)

    fig.tight_layout()
    _save_and_show(fig, save_to)
