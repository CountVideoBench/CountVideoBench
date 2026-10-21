# Gemini 3.1 Pro on CountVideoBench

Code to reproduce the results of **Gemini 3.1 Pro** on
[CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

The model is queried through the
[Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-mode), as for
the paper. No GPU is needed.

## Requirements

- Linux on **x86_64** or **aarch64**.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey)),
  in the `GEMINI_API_KEY` environment variable:

  ```bash
  export GEMINI_API_KEY=...
  ```

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/gemini_3_1_pro
uv sync
```

`uv sync` installs Python 3.12 and the exact versions of all dependencies
recorded in `uv.lock`, including the `google-genai` SDK 1.70.0 used for the paper.

## Usage

The evaluation runs in four steps:

```bash
uv run evaluate.py prepare          # extract the frames and write the request files
uv run evaluate.py submit           # upload the files, one batch job per 25 requests
uv run evaluate.py status --wait    # wait until all batch jobs are finished
uv run evaluate.py collect          # download the answers and write the CSV
```

Each step can be run at a different time: everything is kept in a work folder,
`work/Gemini-3.1-Pro/`:

| Path | Content |
| --- | --- |
| `requests/` | Request files written by `prepare`, deleted once their batch job is collected |
| `batches.csv` | One row per batch job: request file, job name, model version, state |
| `responses/` | Batch API output files downloaded by `collect` |

`status` without `--wait` shows the state of the batch jobs once.

**Failed requests.** A request (or a whole batch job) can fail on the API
side. `collect` lists the videos left without an answer; running the four
steps again prepares and submits only those videos. Repeat until `collect`
reports that every video has an answer.

**Quick test.** `uv run evaluate.py prepare --limit 5` prepares the first 5
videos only; the next steps are unchanged.

Main options (`uv run evaluate.py <command> --help` for the full list):

| Option | Commands | Description |
| --- | --- | --- |
| `--streaming` / `--data-dir PATH` | `prepare` (`--data-dir` also `collect`) | Data source (default: Hugging Face cache, see [Data](#data)). |
| `--limit N` | `prepare` | Only consider the first N videos. |
| `--batch-size N` | `prepare` | Requests per batch job (default 25, as for the paper). |
| `--output PATH` | `prepare`, `collect` | Output CSV. Default: `results/Gemini-3.1-Pro_countvideobench.csv`. Videos it already contains are not prepared again. |
| `--work-dir PATH` | all | Work folder. Default: `work/Gemini-3.1-Pro/`. |
| `--wait` | `status` | Check every 30 s until all batch jobs are finished. |
| `--import DIR [DIR ...]` | `collect` | Also read Batch API output files from these folders (see below). |

**Earlier results.** Batch API output files obtained earlier (one JSON line
per request, with its `key` and `response`) can be turned into the CSV
without calling the API:

```bash
uv run evaluate.py collect --import results_run1 results_run2
```

The files are copied into `work/Gemini-3.1-Pro/responses/`. When a video has
several answers, the first one is kept (files are read in name order).

## Data

`prepare` reads the videos of CountVideoBench (2,600 videos, 139 GB) in one of
three ways. All of them read the same files and give the same requests.

| Command | Where the videos are | Disk space |
| --- | --- | --- |
| `uv run evaluate.py prepare` | Hugging Face cache (`HF_HOME`) | up to 139 GB, reused by later runs |
| `uv run evaluate.py prepare --streaming` | Temporary folder, each video deleted after use | about 50 MB |
| `uv run evaluate.py prepare --data-dir PATH` | Existing local copy | none |

In all modes, the request files (which contain the frames) are also written to
the work folder; they are deleted once their batch job is collected.
`collect` only reads the dataset's `metadata.csv`.

**Local copy.** For a copy in a folder of your choice:

```bash
uv run hf download countvideobench/CountVideoBench --repo-type dataset \
    --revision 42e8fed00bca018a8fc10e249673e665aaa352b8 --local-dir CountVideoBench
uv run evaluate.py prepare --data-dir CountVideoBench
```

The other modes use the dataset revision evaluated in the paper; use
`--revision` to choose another one.

## Output

One CSV row per video, in the order of the dataset. The first six columns
come from the dataset's `metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `Gemini-3.1-Pro` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Text of the model's answer, unprocessed (thoughts excluded) |

## Settings

- Model `gemini-3.1-pro-preview`; the paper's answers were produced by
  version `3.1-pro-preview-01-2026`, in April 2026.
- 10 frames per video, at indices `int(i * N / 10)` for `i = 0..9` (N = number
  of frames, i.e. frames 0, 15, ..., 135 for a 5-second video at 30 fps),
  read with OpenCV and encoded as JPEG (OpenCV defaults).
- One user message: the question, then the 10 frames as inline images.
- `temperature` 0, `max_output_tokens` 1024, `thinking_level` `LOW`; no seed.
  The token limit includes the model's thinking: when it is reached while
  thinking, the answer is empty (`collect` reports how many).
- Requests are identified by their `video_id` (`key` field).

## Reproducibility

- `uv.lock` pins every package to the version used for the paper, except
  `huggingface-hub`, added to download the dataset.
- The model is a preview model, served by Google: it has no dated snapshot,
  and its version may change or the model may be retired. `submit` prints the
  version currently served, records it in `batches.csv`, and warns if it is
  not the one used for the paper. Even with the same version and
  `temperature` 0, answers are not guaranteed to be identical from one run to
  the next.

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
