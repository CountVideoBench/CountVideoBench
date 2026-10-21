"""
Metric computations for the CountVideoBench analysis.

Implements every quantity reported in Sections 5.2, 6.1, 6.2, 6.3 of the
paper:

- Coverage (Eq. 3).
- Exact Match (Eq. 1).
- Tolerance Accuracy at 10% (Eq. 2).
- Support size (Eq. 4) and prediction perplexity (Eq. 5).
- Local excess / deficit on the prediction distribution (Section 5.2).
- Exact Match by object class (Appendix F, Table 6).

Two granularities are exposed:
* ``coverage_by_cardinality``, ``exact_match_by_cardinality`` return a long
  DataFrame indexed by ``(model, Ground-truth count)``; these feed the
  Coverage curve (Fig. 3) and the EM heatmap (Fig. 4).
* ``summary_metrics`` returns one row per model, holding every aggregate
  metric of Table 2 (Table 4 when applied to the human-study subset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BIAS_HISTOGRAM_MAX_BIN,
    VIDEOS_PER_CARDINALITY,
    benchmark_cardinalities,
)
from data import COL_ANSWER, COL_COLOR, COL_GT, COL_MODEL, COL_SHAPE


# ---------------------------------------------------------------------------
# Per-video correctness predicates
# ---------------------------------------------------------------------------

def _exact_match_mask(pred: pd.Series, gt: pd.Series) -> pd.Series:
    """1 iff the prediction equals the ground truth (and both are non-NaN)."""
    return (pred.notna() & gt.notna() & pred.eq(gt))


def _acc_at_10_mask(pred: pd.Series, gt: pd.Series) -> pd.Series:
    """
    1 iff |pred - gt| <= floor(0.1 * gt) (Eq. 2).

    Using the floor form (rather than ``<= 0.1 * gt``) makes the predicate
    coincide exactly with Exact Match for gt in {5, ..., 9}, as stated in
    the paper.
    """
    tau = np.floor(0.1 * gt.abs())
    return (pred.notna() & gt.notna() & ((pred - gt).abs() <= tau))


# ---------------------------------------------------------------------------
# Per-cardinality metrics
# ---------------------------------------------------------------------------

def coverage_by_cardinality(df: pd.DataFrame, *, pivot: bool = False) -> pd.DataFrame:
    """
    Proportion of videos with a numerical answer, per (model, cardinality).

    Parameters
    ----------
    df
        Preprocessed long DataFrame.
    pivot
        If True, return a wide DataFrame (rows = cardinalities, columns =
        models). Otherwise return a long DataFrame with columns
        ``[model, "Ground-truth count", coverage]``.
    """
    long = (
        df.assign(_valid=df[COL_ANSWER].notna())
          .groupby([COL_MODEL, COL_GT], as_index=False)["_valid"]
          .mean()
          .rename(columns={"_valid": "coverage"})
          .sort_values([COL_MODEL, COL_GT])
          .reset_index(drop=True)
    )

    if pivot:
        return long.pivot(index=COL_GT, columns=COL_MODEL, values="coverage")
    return long


def exact_match_by_cardinality(df: pd.DataFrame, *, pivot: bool = False) -> pd.DataFrame:
    """
    Exact Match rate per (model, cardinality).

    Non-numerical answers are counted as incorrect (consistent with Eq. 1).
    """
    pred = pd.to_numeric(df[COL_ANSWER], errors="coerce")
    gt = pd.to_numeric(df[COL_GT], errors="coerce")

    work = df.assign(_em=_exact_match_mask(pred, gt).astype(int))

    long = (
        work.groupby([COL_MODEL, COL_GT], as_index=False)["_em"]
            .mean()
            .rename(columns={"_em": "exact_match"})
            .sort_values([COL_MODEL, COL_GT])
            .reset_index(drop=True)
    )

    if pivot:
        return long.pivot(index=COL_GT, columns=COL_MODEL, values="exact_match")
    return long


def acc_at_10_by_cardinality(df: pd.DataFrame, *, pivot: bool = False) -> pd.DataFrame:
    """Acc@10% rate per (model, cardinality). See ``_acc_at_10_mask``."""
    pred = pd.to_numeric(df[COL_ANSWER], errors="coerce")
    gt = pd.to_numeric(df[COL_GT], errors="coerce")

    work = df.assign(_acc=_acc_at_10_mask(pred, gt).astype(int))

    long = (
        work.groupby([COL_MODEL, COL_GT], as_index=False)["_acc"]
            .mean()
            .rename(columns={"_acc": "acc_at_10"})
            .sort_values([COL_MODEL, COL_GT])
            .reset_index(drop=True)
    )

    if pivot:
        return long.pivot(index=COL_GT, columns=COL_MODEL, values="acc_at_10")
    return long


# ---------------------------------------------------------------------------
# Distributional measures of the prediction histogram
# ---------------------------------------------------------------------------

def shannon_entropy_bits(counts) -> float:
    """
    Shannon entropy in bits of a non-negative count vector.

    Zero-count entries are ignored. Returns 0.0 if the total mass is zero.
    """
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-np.sum(p * np.log2(p)))


def peak_local_deviation(
    answers: pd.Series,
    *,
    k: int = 2,
    smin: int = 0,
    smax: int = 70,
) -> pd.Series:
    """
    Magnitude of the largest local peak and valley in the prediction
    histogram (Section 5.2, definitions of local excess and local deficit).

    For each position ``c`` in ``[smin, smax]``:
      - ``peak(c)   = obs(c) - max(neighbours on either side)``;
        positive when ``c`` stands strictly above all its neighbours.
      - ``valley(c) = min(left-max, right-max) - obs(c)``;
        positive only when ``c`` sits in a valley between two flanking
        peaks (so it requires neighbours on both sides; edges are masked
        out by setting valley to 0 there).

    Parameters
    ----------
    answers
        Series of model predictions (``NaN`` = abstention, ignored).
    k
        Half-width of the neighbour window (k=2 means up to 2 on each side).
    smin, smax
        Inclusive bounds of the integer support to inspect.

    Returns
    -------
    pd.Series with index ``["max_peak", "max_valley", "peak_at", "valley_at"]``.
    """
    counts = answers.dropna().astype(int).value_counts().to_dict()
    support = np.arange(smin, smax + 1)
    obs = np.array([counts.get(x, 0) for x in support], dtype=float)

    if obs.sum() == 0:
        return pd.Series({
            "max_peak": np.nan,
            "max_valley": np.nan,
            "peak_at": np.nan,
            "valley_at": np.nan,
        })

    n = len(obs)
    peak_score = np.zeros(n)
    valley_score = np.zeros(n)

    for i in range(n):
        left = obs[max(0, i - k):i]
        right = obs[i + 1:min(n, i + k + 1)]

        left_max = left.max() if left.size > 0 else 0.0
        right_max = right.max() if right.size > 0 else 0.0

        peak_score[i] = obs[i] - max(left_max, right_max)

        if left.size > 0 and right.size > 0:
            valley_score[i] = min(left_max, right_max) - obs[i]
        # else: edge of support — no measurable valley, leave at 0.0.

    return pd.Series({
        "max_peak":   float(peak_score.max()),
        "max_valley": float(valley_score.max()),
        "peak_at":    int(support[peak_score.argmax()]),
        "valley_at":  int(support[valley_score.argmax()]),
    })


# ---------------------------------------------------------------------------
# Per-model summary (Table 2)
# ---------------------------------------------------------------------------

def _summary_for_group(
    group: pd.DataFrame,
    *,
    abstention_as_class: bool = True,
) -> pd.Series:
    """
    Compute the full row of Table 2 metrics for one model's predictions.

    Conventions
    -----------
    - ``EM`` and ``Acc@10%`` are expressed in percent (0..100).
    - ``Coverage`` is expressed as a fraction in [0, 1].
    - ``Entropy`` is in bits.
    - ``Perplexity`` is ``2 ** Entropy`` (effective number of equally-frequent
      counts).

    Parameters
    ----------
    abstention_as_class
        If True (default, the convention of the paper's Tables 2 and 4),
        abstentions are added as an extra category before computing entropy
        and perplexity. If False, they are computed over the answered subset
        only. The two conventions agree for any model with 100% coverage; they
        diverge as coverage drops. Support size is always computed on the
        answered subset, in agreement with Eq. 4.
    """
    pred = pd.to_numeric(group[COL_ANSWER], errors="coerce")
    gt = pd.to_numeric(group[COL_GT], errors="coerce")

    em = _exact_match_mask(pred, gt).mean() * 100.0
    acc10 = _acc_at_10_mask(pred, gt).mean() * 100.0
    coverage = pred.notna().mean()

    valid = pred.notna() & gt.notna()
    err = pred[valid] - gt[valid]
    mae = err.abs().mean()                       # NaN if no answered video
    rmse = np.sqrt((err ** 2).mean())

    answered = pred.dropna()
    support_size = int(answered.nunique())

    pred_counts = answered.value_counts().values
    if abstention_as_class:
        n_abstentions = len(pred) - len(answered)
        if n_abstentions > 0:
            pred_counts = np.append(pred_counts, n_abstentions)
    H = shannon_entropy_bits(pred_counts)

    return pd.Series({
        "EM":           em,
        "Acc@10%":      acc10,
        "MAE":          mae,
        "RMSE":         rmse,
        "Coverage":     coverage,
        "Support size": support_size,
        "Entropy":      H,
        "Perplexity":   2.0 ** H,
    })


def summary_metrics(
    df: pd.DataFrame,
    *,
    include_oracle: bool = True,
    abstention_as_class: bool = True,
) -> pd.DataFrame:
    """
    Per-model summary metrics (one row per model = Table 2 of the paper).

    Parameters
    ----------
    df
        Preprocessed long DataFrame.
    include_oracle
        If True, append a final row ``"Oracle"`` corresponding to a model
        that predicts the ground truth on every video (i.e. ``support = 26``,
        ``perplexity = 26``).
    abstention_as_class
        See ``_summary_for_group``. Defaults to True (the paper's
        convention).
    """
    metrics = (
        df.groupby(COL_MODEL, group_keys=False)
          .apply(_summary_for_group,
                 abstention_as_class=abstention_as_class,
                 include_groups=False)
    )

    if include_oracle:
        cardinalities = benchmark_cardinalities()
        oracle = pd.DataFrame({
            COL_ANSWER: np.repeat(cardinalities, VIDEOS_PER_CARDINALITY),
            COL_GT:     np.repeat(cardinalities, VIDEOS_PER_CARDINALITY),
        })
        metrics.loc["Oracle"] = _summary_for_group(
            oracle, abstention_as_class=abstention_as_class
        )

    return metrics


def local_irregularities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Largest local excess and deficit of each model's prediction histogram
    (Section 5.2), with the values at which they occur.

    One row per model; columns ``max_peak`` (largest local excess),
    ``peak_at``, ``max_valley`` (largest local deficit), ``valley_at``.
    """
    return (
        df.groupby(COL_MODEL)[COL_ANSWER]
          .apply(lambda answers: peak_local_deviation(
              pd.to_numeric(answers, errors="coerce"), k=2, smin=0, smax=70))
          .unstack()
          [["max_peak", "peak_at", "max_valley", "valley_at"]]
    )


# ---------------------------------------------------------------------------
# Per-model prediction tables (used by the bias histograms)
# ---------------------------------------------------------------------------

def cardinality_report(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Per-value prediction table for one model.

    For each integer in ``[1, BIAS_HISTOGRAM_MAX_BIN]`` plus the catch-all
    bucket ``"61+"``, count how many times the model predicted that value
    and how many of those predictions were correct. Predicted values are
    rounded to ``int`` after dropping abstentions.

    Output columns
    --------------
    ``cardinality, number_correct, number_incorrect, is_in_dataset``
    where ``is_in_dataset`` flags the 26 cardinalities that actually appear
    as ground truth in CountVideoBench.
    """
    sub = df[df[COL_MODEL] == model].copy()

    def bucket(x):
        if pd.isna(x):
            return np.nan
        return f"{BIAS_HISTOGRAM_MAX_BIN + 1}+" if x > BIAS_HISTOGRAM_MAX_BIN else str(int(x))

    sub["predicted_cardinality"] = sub[COL_ANSWER].apply(bucket)
    sub["true_cardinality"] = sub[COL_GT].apply(bucket)
    sub["is_correct"] = sub[COL_ANSWER] == sub[COL_GT]

    answered = sub.dropna(subset=["predicted_cardinality"])

    grouped = (
        answered.groupby("predicted_cardinality")
                .agg(number_correct=("is_correct", "sum"),
                     total_predictions=("is_correct", "size"))
                .reset_index()
                .rename(columns={"predicted_cardinality": "cardinality"})
    )
    grouped["number_incorrect"] = grouped["total_predictions"] - grouped["number_correct"]

    all_cardinalities = pd.DataFrame({
        "cardinality": [str(i) for i in range(1, BIAS_HISTOGRAM_MAX_BIN + 1)]
                       + [f"{BIAS_HISTOGRAM_MAX_BIN + 1}+"]
    })

    result = all_cardinalities.merge(
        grouped[["cardinality", "number_correct", "number_incorrect"]],
        on="cardinality",
        how="left",
    )
    result["number_correct"] = result["number_correct"].fillna(0).astype(int)
    result["number_incorrect"] = result["number_incorrect"].fillna(0).astype(int)
    result["is_in_dataset"] = result["cardinality"].isin(
        set(sub["true_cardinality"].dropna())
    )
    return result


# ---------------------------------------------------------------------------
# Exact Match by object class (Appendix F, Table 6)
# ---------------------------------------------------------------------------

def exact_match_by_visual_attributes(
    df: pd.DataFrame,
    *,
    material_suffix: str | None = "_mat",
    pivot: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Exact Match grouped by colour/shape, shape, and colour.

    The returned EM rates are fractions in ``[0, 1]``. As for
    :func:`exact_match_by_cardinality`, a missing or non-numerical answer is
    counted as incorrect.
    """
    required = {COL_MODEL, COL_ANSWER, COL_GT, COL_SHAPE, COL_COLOR}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(
            "exact_match_by_visual_attributes requires missing column(s): "
            + ", ".join(repr(column) for column in missing)
        )

    work = df.copy()

    # "red_mat" -> "red" (older CSV files); no effect on "red".
    colors = work[COL_COLOR].astype("string")
    if material_suffix is not None:
        colors = colors.str.removesuffix(material_suffix)
    work["_report_color"] = colors

    # EM = 1 only if answer == ground truth; a missing or non-numerical
    # answer counts as 0.
    pred = pd.to_numeric(work[COL_ANSWER], errors="coerce")
    gt = pd.to_numeric(work[COL_GT], errors="coerce")
    work["_em"] = _exact_match_mask(pred, gt).astype(int)

    # Keep the order of appearance of models, shapes and colours.
    model_order = pd.Index(work[COL_MODEL].drop_duplicates(), name=COL_MODEL)
    shape_order = pd.Index(
        work[COL_SHAPE].dropna().drop_duplicates(),
        name="shape",
    )
    color_order = pd.Index(
        work["_report_color"].dropna().drop_duplicates(),
        name="color",
    )

    # EM for each (colour, shape) pair.
    by_color_shape = (
        work.groupby([COL_MODEL, "_report_color", COL_SHAPE], as_index=False)["_em"]
            .mean()
            .rename(
                columns={
                    "_report_color": "color",
                    COL_SHAPE: "shape",
                    "_em": "exact_match",
                }
            )
            .sort_values([COL_MODEL, "color", "shape"])
            .reset_index(drop=True)
    )

    # EM by shape, all colours together.
    by_shape = (
        work.groupby([COL_MODEL, COL_SHAPE], as_index=False)["_em"]
            .mean()
            .rename(columns={COL_SHAPE: "shape", "_em": "exact_match"})
            .sort_values([COL_MODEL, "shape"])
            .reset_index(drop=True)
    )

    # EM by colour, all shapes together.
    by_color = (
        work.groupby([COL_MODEL, "_report_color"], as_index=False)["_em"]
            .mean()
            .rename(columns={"_report_color": "color", "_em": "exact_match"})
            .sort_values([COL_MODEL, "color"])
            .reset_index(drop=True)
    )

    # Long tables instead of pivoted ones.
    if not pivot:
        return {
            "color_shape": by_color_shape,
            "shape": by_shape,
            "color": by_color,
        }

    # Rows = models, columns = (colour, shape) pairs.
    color_shape_columns = pd.MultiIndex.from_product(
        [color_order, shape_order],
        names=["color", "shape"],
    )
    color_shape_table = (
        by_color_shape
        .pivot(
            index=COL_MODEL,
            columns=["color", "shape"],
            values="exact_match",
        )
        .reindex(index=model_order, columns=color_shape_columns)
    )

    # Rows = models, columns = shapes.
    shape_table = (
        by_shape
        .pivot(index=COL_MODEL, columns="shape", values="exact_match")
        .reindex(index=model_order, columns=shape_order)
    )

    # Rows = models, columns = colours.
    color_table = (
        by_color
        .pivot(index=COL_MODEL, columns="color", values="exact_match")
        .reindex(index=model_order, columns=color_order)
    )

    return {
        "color_shape": color_shape_table,
        "shape": shape_table,
        "color": color_table,
    }
