"""Compute the metrics, tables and figures of the CountVideoBench paper.

Example:
    uv run compute_metrics.py                          # counts of the paper
    uv run compute_metrics.py --input "results/*.csv"  # other CSV files

The input CSV files have the columns of ``../paper_results/`` (one file per
model, one row per video, the extracted count in ``predicted_count``). Every
table (CSV) and figure (PDF) is written to the output folder:

    models_summary_metrics.csv          Table 2
    local_irregularities.csv            largest local excess / deficit (Section 6.3)
    coverage_evolution.csv, coverage_curve.pdf            Figure 3
    heatmap_accuracy_overall.csv, heatmap_accuracy_overall.pdf   Figure 4
    bias_barplots/*.pdf, bias_tables/*.csv                Figure 5, Figures 8-15
    exact_match_by_shape.csv, exact_match_by_color.csv,
    exact_match_by_color_shape.csv                        Table 6
    human_subset/models_summary_metrics.csv               Table 4 (model rows)
    human_subset/heatmap_accuracy_overall.csv             Figure 7 (model rows)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from config import (
    BIAS_FIGURE_SUBDIR,
    BIAS_TABLE_SUBDIR,
    DEFAULT_INPUT_PATTERN,
    DEFAULT_OUTPUT_DIR,
    HUMAN_SUBSET_PATH,
    apply_matplotlib_style,
    benchmark_cardinalities,
    build_model_palette,
)
from data import (
    load_model_csvs,
    normalise_model_list,
    palette_order,
    preprocess,
    select_videos,
)
from metrics import (
    cardinality_report,
    coverage_by_cardinality,
    exact_match_by_cardinality,
    exact_match_by_visual_attributes,
    local_irregularities,
    summary_metrics,
)
from plots import plot_accuracy_heatmap, plot_bias_barplot, plot_line_by_cardinality


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the metrics, tables and figures of the CountVideoBench paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATTERN,
                        help="Glob pattern of the per-model CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Folder where the tables and figures are written.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Only write the tables.")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    apply_matplotlib_style()

    df_raw, raw_models = load_model_csvs(args.input)
    df = preprocess(df_raw)
    models = normalise_model_list(raw_models)
    palette = build_model_palette(palette_order(models))
    print(f"Loaded {len(df):,} rows across {len(models)} models.")

    # Table 2.
    metrics = summary_metrics(df, include_oracle=True)
    metrics.to_csv(out / "models_summary_metrics.csv")
    print("\nTable 2")
    print(metrics.round(2).to_string())

    # Largest local excess / deficit of the prediction histograms.
    local_irregularities(df).to_csv(out / "local_irregularities.csv")

    # Table 6.
    attribute_tables = exact_match_by_visual_attributes(df)
    for name, table in attribute_tables.items():
        table.to_csv(out / f"exact_match_by_{name}.csv")

    # Figure 3.
    coverage_long = coverage_by_cardinality(df)
    coverage_by_cardinality(df, pivot=True).to_csv(out / "coverage_evolution.csv")

    # Figure 4.
    em_long = exact_match_by_cardinality(df)
    exact_match_by_cardinality(df, pivot=True).to_csv(out / "heatmap_accuracy_overall.csv")

    # Tables of Figure 5 and Figures 8-15.
    bias_tab_dir = out / BIAS_TABLE_SUBDIR
    bias_tab_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        safe_name = model.replace("/", "_")
        cardinality_report(df, model).to_csv(bias_tab_dir / f"{safe_name}.csv", index=False)

    # Table 4 and Figure 7: model rows on the 104-video subset of the human study.
    subset_ids = pd.read_csv(HUMAN_SUBSET_PATH)["video_id"]
    df_subset = select_videos(df, subset_ids)
    subset_dir = out / "human_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics(df_subset, include_oracle=False).to_csv(subset_dir / "models_summary_metrics.csv")
    exact_match_by_cardinality(df_subset, pivot=True).to_csv(subset_dir / "heatmap_accuracy_overall.csv")

    if not args.no_figures:
        plot_line_by_cardinality(
            coverage_long, y="coverage", y_label="Coverage",
            palette=palette, percent=True,
            save_to=out / "coverage_curve.pdf",
        )
        plot_accuracy_heatmap(
            em_long, value="exact_match", cmap="Blues", annotate=True,
            save_to=out / "heatmap_accuracy_overall.pdf",
        )
        in_grid = benchmark_cardinalities()
        for model in models:
            safe_name = model.replace("/", "_")
            plot_bias_barplot(
                df, model, in_dataset_cardinalities=in_grid,
                save_to=out / BIAS_FIGURE_SUBDIR / f"{safe_name}_bias_barplot.pdf",
            )

    print(f"\nTables and figures written to {out}")


if __name__ == "__main__":
    main()
