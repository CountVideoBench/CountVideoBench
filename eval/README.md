# Count extraction and metrics for CountVideoBench

The models evaluated on
[CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench)
answer in free text ("There are ten red cubes in this video."). This folder
contains:

- `extract.py`: the code that reads each answer with a local LLM
  (**Qwen2.5-7B-Instruct**) and extracts the count it states, as in the
  CountVideoBench paper (BMVC 2026);
- `paper_results/`: the counts used in the paper, for every model. They were
  extracted with this code and then **checked by hand**, so they cannot be
  regenerated exactly by running it again (see [Paper results](#paper-results));
- `metrics/`: the code that computes the metrics, tables and figures of the
  paper from these counts (see [`metrics/README.md`](metrics/README.md)).

## Requirements

- Linux on **x86_64** or **aarch64**.
- An NVIDIA GPU with a driver that supports **CUDA 12.8** (driver 570 or newer).
  The CUDA libraries themselves are installed by uv; no system CUDA toolkit is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GPU memory: the weights take about 15 GB in bfloat16. The paper's counts were
  extracted on a single RTX 4090 (24 GB).

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/eval
uv sync
```

`uv sync` installs Python 3.10 and the exact versions of all dependencies
recorded in `uv.lock`, including PyTorch 2.9.0 (CUDA 12.8) and transformers 4.57.1.

## Usage

Give the CSV files written by the models' `evaluate.py`:

```bash
uv run extract.py ../qwen2_5_vl/results/Qwen2.5-VL-7B_countvideobench.csv
uv run extract.py ../*/results/*_countvideobench.csv     # all models at once
```

The weights of the extraction model are downloaded from the Hugging Face Hub
on first use. Each input gives one output file, by default
`results/<Model>_extracted.csv` (e.g. `results/Qwen2.5-VL-7B_extracted.csv`).

Main options (`uv run extract.py --help` for the full list):

| Option | Description |
| --- | --- |
| `--output PATH` | Output CSV (only with a single input file). |
| `--limit N` | Only process the first N answers of each input, for a quick test. |
| `--device-map` | Default `auto`: the model is placed on the visible GPU(s). Use `CUDA_VISIBLE_DEVICES` to choose them. |
| `--model-path`, `--model-revision` | Another extraction model, or a local copy of the weights. |

**Resuming.** Each count is written as soon as it is extracted. If the output
file already exists, the answers it contains are skipped, so an interrupted
run can be resumed with the same command. Delete the file to start over.

## Output

The output CSV has the columns of the input (`model`, `video_id`, `count`,
`shape`, `color`, `question`, `raw_answer`) followed by:

| Column | Description |
| --- | --- |
| `predicted_count` | Count stated in the answer, as an integer. Empty when the extraction model finds no explicit count (`$NaN$`), or when its output could not be read. |
| `extractor_output` | Raw output of the extraction model, e.g. `$10$` or `$NaN$`. |

At the end, `extract.py` prints for each file how many answers have a count,
how many have none (`$NaN$`), and the `video_id` of the answers whose
extractor output is unexpected (no `$` sign, or something else than an integer
or `NaN` between them). Their `predicted_count` is left empty; check them by
hand in `extractor_output` and `raw_answer`.

## Settings

- Extraction model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a3545`, weights
  in their native precision (bfloat16).
- System message: the instructions in `SYSTEM_PROMPT` (`extract.py`), exactly
  as used for the paper. The model must return the count between `$` signs, or
  `$NaN$` when the answer gives no explicit count (unknown, estimated, several
  possible counts, …); an answer starting with "There are approximately N"
  gives N.
- User message: `User Question : <question>` and `Answer Programme : <raw_answer>`
  on two lines.
- Greedy decoding, at most 512 new tokens.
- The count is read between the first two `$` signs of the output.

## Paper results

`paper_results/` contains one CSV per model, with the counts used in the
paper:

| Column | Description |
| --- | --- |
| `model`, `video_id`, `count`, `shape`, `color`, `question` | As in the models' output files |
| `raw_answer` | Answer of the model, unprocessed |
| `predicted_count` | Count used in the paper. Empty when the answer gives no count (see the rules below). |

The counts were obtained in two steps:

1. **Automatic extraction** with `extract.py`, as it is in this folder.
2. **Manual check** of every extracted count, corrected when needed with the
   rules below. The extraction model follows them for most answers, but not
   for all (for example, it treats most estimates as `$NaN$`, and it may take
   a partial count from an answer that was cut off).

Rules of the manual check:

- An explicit count is used ("There are 23 red cubes.", "Final Answer: 23").
- A single estimate is used as the count ("approximately 30", "around 50",
  or a sum of estimates such as "5 rows × 6 cubes = 30 cubes").
- A range ("50-60", "between 40 and 50"), a bound ("at least 30", "more than
  20"), several possible counts, or no number at all give no count.
- An answer cut by the limit on generated tokens before reaching a
  conclusion gives no count, unless it states the count in its first sentence.

Running `extract.py` again on the same answers therefore gives the same counts
for most of them, but not for the corrected ones. To evaluate a new model with
the method of the paper, run `extract.py`, then check the extracted counts by
hand with the same rules.

## Reproducibility

- `uv.lock` pins every package to the version used for the paper.
- The weights of the extraction model are pinned to the Hub revision used for
  the paper (`a09a3545`).
- On other GPUs than the one used for the paper (RTX 4090), the extraction
  model may give slightly different outputs because of floating-point
  differences between hardware.
- transformers prints `The following generation flags are not valid and may be
  ignored: ['temperature', 'top_p', 'top_k']`. These sampling settings come
  from the model's default generation configuration and have no effect with
  greedy decoding.

## Citation

If you use this code or these results, please cite CountVideoBench and Qwen2.5:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}

@misc{qwen2.5,
  title  = {Qwen2.5: A Party of Foundation Models},
  url    = {https://qwenlm.github.io/blog/qwen2.5/},
  author = {Qwen Team},
  month  = {September},
  year   = {2024}
}
```
