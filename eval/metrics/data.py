"""
Loading and preprocessing of the per-model CSV files.

Each input CSV holds one model's answers and the count extracted from each
answer (see Section 5.1 of the paper and ``eval/README.md``). After loading,
this module
- renames the columns to the names used by the analysis code,
- maps the model names to the names displayed in the paper
  (``config.MODEL_NAME_REMAP``),
and returns a single tidy DataFrame with one row per (model, video).
"""

from __future__ import annotations

from glob import glob

import pandas as pd

from config import MODEL_NAME_REMAP, MODEL_ORDER


# Internal column name conventions used everywhere downstream.
COL_MODEL = "model"
COL_VIDEO = "video_id"
COL_ANSWER = "answer"
COL_GT = "Ground-truth count"

COL_SHAPE = "class"
COL_COLOR = "color"

# Columns of the input CSV files  ->  internal names.
_INPUT_COLUMNS: dict[str, str] = {
    "count": COL_GT,
    "predicted_count": COL_ANSWER,
    "shape": COL_SHAPE,
}


def load_model_csvs(pattern: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load every CSV matching ``pattern`` and concatenate the rows.

    Parameters
    ----------
    pattern
        Glob expression (e.g. ``"../paper_results/*.csv"``).

    Returns
    -------
    df
        Concatenated DataFrame with one row per (model, video) prediction.
    models
        List of distinct model names actually present in the data, in the
        order they appear on disk.
    """
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV file matches pattern: {pattern}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    models = df[COL_MODEL].drop_duplicates().tolist()
    return df, models


def preprocess(
    df: pd.DataFrame,
    *,
    name_remap: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Apply the standard preprocessing pipeline.

    Steps
    -----
    1. Rename the input columns (``count``, ``predicted_count``, ``shape``)
       to the internal names.
    2. Apply ``name_remap`` to the ``"model"`` column.

    Empty answers (``NaN``) are kept and treated downstream as abstentions.

    Returns
    -------
    A new DataFrame with a contiguous index. Does not mutate the input.
    """
    if name_remap is None:
        name_remap = MODEL_NAME_REMAP

    out = df.rename(columns=_INPUT_COLUMNS)

    if name_remap:
        out[COL_MODEL] = out[COL_MODEL].replace(name_remap)

    return out.reset_index(drop=True)


def normalise_model_list(models: list[str], name_remap: dict[str, str] | None = None) -> list[str]:
    """
    Apply ``name_remap`` to a raw model list, preserving order.

    Useful to keep the palette dict in sync with the names actually present
    in the preprocessed DataFrame.
    """
    if name_remap is None:
        name_remap = MODEL_NAME_REMAP
    return [name_remap.get(m, m) for m in models]


def palette_order(models: list[str]) -> list[str]:
    """
    Order ``models`` as in ``config.MODEL_ORDER`` (the order used for the
    colours of the paper's figures); models not listed there come last.
    """
    known = [m for m in MODEL_ORDER if m in models]
    return known + [m for m in models if m not in MODEL_ORDER]


def select_videos(df: pd.DataFrame, video_ids) -> pd.DataFrame:
    """Keep only the rows of the given videos (e.g. the human-study subset)."""
    return df[df[COL_VIDEO].isin(set(video_ids))].reset_index(drop=True)
