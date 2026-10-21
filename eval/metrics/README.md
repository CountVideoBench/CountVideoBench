# Metrics of CountVideoBench

This folder computes the metrics, tables and figures of the CountVideoBench
paper (BMVC 2026) from the counts extracted from the models' answers
(`../paper_results/`, see [`../README.md`](../README.md)).

## Requirements

- Linux on **x86_64** or **aarch64**. No GPU is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Usage

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/eval/metrics
uv run compute_metrics.py
```

`uv run` installs Python 3.10 and the exact versions of the dependencies
recorded in `uv.lock` (pandas, NumPy, matplotlib, seaborn) on first use.
Every table (CSV) and figure (PDF) is written to `output/`:

| Output | In the paper |
| --- | --- |
| `models_summary_metrics.csv` | Table 2 |
| `local_irregularities.csv` | Largest local excess and deficit (Section 6.3) |
| `coverage_evolution.csv`, `coverage_curve.pdf` | Figure 3 |
| `heatmap_accuracy_overall.csv`, `heatmap_accuracy_overall.pdf` | Figure 4 |
| `bias_barplots/*.pdf`, `bias_tables/*.csv` | Figure 5, Figures 8–15 |
| `exact_match_by_shape.csv`, `exact_match_by_color.csv`, `exact_match_by_color_shape.csv` | Table 6 |
| `human_subset/models_summary_metrics.csv` | Table 4 (model rows) |
| `human_subset/heatmap_accuracy_overall.csv` | Figure 7 (model rows) |

Options (`uv run compute_metrics.py --help`):

| Option | Description |
| --- | --- |
| `--input PATTERN` | CSV files to read (default: `../paper_results/*.csv`). Use it for the files written by `../extract.py`, after checking the counts by hand. |
| `--output PATH` | Output folder (default: `output/`). |
| `--no-figures` | Only write the tables. |

### Notebook

`analyse.ipynb` computes the same tables and displays the figures. Jupyter is
installed only when asked for:

```bash
uv run --group notebook jupyter lab analyse.ipynb
```

`class_analysis.ipynb` computes Exact Match by object class (Appendix F):
Figure 17 (`output/heatmap_accuracy_by_category.pdf` and `.csv`) and Table 6
as in the paper (`output/table_accuracy_by_category.csv`).

```bash
uv run --group notebook jupyter lab class_analysis.ipynb
```

## Metrics

As defined in Section 5.2 of the paper. Let ŷ be the predicted count and y the
ground-truth count of a video; an empty `predicted_count` is a non-response.

- **EM** (Exact Match): proportion of videos with ŷ = y. Non-responses count as
  incorrect.
- **Acc@10%**: proportion of videos with |ŷ − y| ≤ ⌊0.1 y⌋, so that it equals EM
  for y < 10. Non-responses count as incorrect.
- **MAE**, **RMSE**: mean absolute and root-mean-square error over the videos
  with a numerical answer.
- **Coverage**: proportion of videos with a numerical answer.
- **Support size** |S|: number of distinct predicted counts.
- **Perplexity** PP = 2^H, with H the Shannon entropy (in bits) of the
  distribution of the answers, in which non-responses form one extra category.
- **Local excess / deficit**: for each value k, with N_k the number of times k
  is predicted and L_k, R_k the largest of the two counts on each side of k,
  excess = N_k − max(L_k, R_k) and deficit = min(L_k, R_k) − N_k; the largest
  of each over the values 0 to 70 is reported.

The `Oracle` row of Table 2 is a model that predicts the ground truth on every
video.

## Files

- `config.py`: benchmark structure, model names displayed in the paper, figure style.
- `data.py`: loading of the CSV files.
- `metrics.py`: metric computations.
- `plots.py`: figures.
- `compute_metrics.py`: computes everything and writes it to `output/`.
- `analyse.ipynb`: the same, as a notebook.
- `class_analysis.ipynb`: Exact Match by object class (Figure 17, Table 6).
- `human_subset.csv`: video ids of the 104-video subset of the human study
  (Appendix C). The participants' answers are not included.

## Citation

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}
```
