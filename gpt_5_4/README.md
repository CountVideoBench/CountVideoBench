# GPT-5.4 on CountVideoBench

Code to reproduce the results of **GPT-5.4** (snapshot `gpt-5.4-2026-03-05`) on
[CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

GPT-5.4 is queried through the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch):
requests are written to files, uploaded, and processed by OpenAI within 24 hours,
at a lower price than direct calls.

## Requirements

- Linux on **x86_64** or **aarch64**. No GPU is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An OpenAI API key with access to the Batch API, in the `OPENAI_API_KEY`
  environment variable:

  ```bash
  export OPENAI_API_KEY=sk-...
  ```

  Each request uses about 23,400 input tokens (10 frames in high detail), so
  the full benchmark (2,600 videos) uses about 61 million input tokens.

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/gpt_5_4
uv sync
```

`uv sync` installs Python 3.10 and the exact versions of all dependencies
recorded in `uv.lock`.

## Usage

The evaluation runs in four steps:

```bash
uv run evaluate.py prepare   # extract the frames, write the request files
uv run evaluate.py submit    # upload them and create the batches
uv run evaluate.py status    # follow the batches
uv run evaluate.py collect   # download the results and write the CSV
```

`status` can be run at any time. `collect` downloads the results of the batches
that are finished and can also be run several times: it writes the CSV with
everything received so far and says how many videos are still missing.

**Missing answers.** A batch or a single request can fail on OpenAI's side
(for example when the account's limit of queued tokens is reached). Once all
batches are collected, running `prepare`, `submit` and `collect` again sends
only the videos that have no answer in the CSV. Repeat until the CSV is
complete. Delete the CSV and the `work/` folder to start over.

Main options (`uv run evaluate.py <command> --help` for the full list):

| Option | Commands | Description |
| --- | --- | --- |
| `--limit N` | `prepare` | Only prepare the first N videos, for a quick test. |
| `--batch-size N` | `prepare` | Requests per batch (default 25). Larger batches may exceed the account's limit of queued tokens. |
| `--streaming` / `--data-dir PATH` | `prepare`, `collect` | Data source (default: Hugging Face cache, see [Data](#data)). |
| `--output PATH` | all | Output CSV. Default: `results/GPT-5.4_countvideobench.csv`. |
| `--work-dir PATH` | all | Request files, batch list and downloaded results. Default: `work/GPT-5.4/`. |
| `--model NAME` | all | OpenAI model. Default: `gpt-5.4-2026-03-05`, the snapshot evaluated in the paper. |
| `--import DIR` | `collect` | Also read Batch API output files (`*_output.jsonl`) from `DIR`. |

The work folder contains:

- `requests/`: request files written by `prepare`. Each one is deleted once its results are collected.
- `batches.csv`: the submitted batches and their status.
- `responses/`: the output and error files downloaded from OpenAI.

## Data

CountVideoBench (2,600 videos, 139 GB) can be read in three ways. All of them
read the same files and give the same requests. Only `prepare` reads the
videos; `collect` only reads the dataset's `metadata.csv`.

| Command | Where the videos are | Disk space |
| --- | --- | --- |
| `uv run evaluate.py prepare` | Hugging Face cache (`HF_HOME`) | up to 139 GB, reused by later runs |
| `uv run evaluate.py prepare --streaming` | Temporary folder, each video deleted after use | about 50 MB |
| `uv run evaluate.py prepare --data-dir PATH` | Existing local copy | none |

**Default: Hugging Face cache.** Each video is downloaded just before its
frames are extracted and kept in the Hugging Face cache, so later runs (e.g.
other models of this repository) do not download it again. A dataset already
downloaded with `hf download` without `--local-dir` is reused.

**Streaming.** Same download, but each video is deleted right after its frames
are extracted. Useful when disk space is limited.

**Local copy.** For a copy in a folder of your choice:

```bash
uv run hf download countvideobench/CountVideoBench --repo-type dataset \
    --revision 42e8fed00bca018a8fc10e249673e665aaa352b8 --local-dir CountVideoBench
uv run evaluate.py prepare --data-dir CountVideoBench
uv run evaluate.py collect --data-dir CountVideoBench
```

The first two modes use the dataset revision evaluated in the paper; use
`--revision` to choose another one.

## Output

One CSV row per video, in the order of the dataset. The first six columns come
from the dataset's `metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `GPT-5.4` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

## Settings

- 10 frames per video, at indices `int(i * N / 10)` for `i = 0..9` (N = number
  of frames: frames 0, 15, …, 135 for a 5-second video at 30 fps), read with
  OpenCV and sent as JPEG images at full resolution, in high detail;
- one user message: the question, then the 10 frames;
- `reasoning_effort: none`, temperature 0, seed 42, at most 1024 completion tokens;
- 25 requests per batch.

## Reproducibility

- `uv.lock` pins every package to the version used for the paper, and the
  requests are built exactly as for the paper (same frames, same JPEG encoding).
- The model runs on OpenAI's servers. OpenAI does not guarantee identical
  outputs for identical requests, even with temperature 0 and a fixed seed, so
  a few answers may differ from the paper's.
- The snapshot `gpt-5.4-2026-03-05` can only be queried as long as OpenAI keeps
  it available. `--model` selects another model.

## Citation

If you use this code, please cite CountVideoBench:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}
```
