# Qwen3-VL on CountVideoBench

Code to reproduce the results of **Qwen3-VL-8B-Instruct** and
**Qwen3-VL-32B-Instruct** on [CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

The model is run with the standard `transformers` implementation; no
third-party code is needed.

## Requirements

- Linux on **x86_64** or **aarch64** (e.g. NVIDIA GH200).
- An NVIDIA GPU with a driver that supports **CUDA 13.0** (driver 580 or newer).
  The CUDA libraries themselves are installed by uv; no system CUDA toolkit is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GPU memory: the weights alone take about 18 GB (8B) and 67 GB (32B) in
  bfloat16. The paper's results were obtained on 2 × GH200 144 GB.

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/qwen3_vl
uv sync
```

`uv sync` installs Python 3.12 and the exact versions of all dependencies
recorded in `uv.lock`, including PyTorch 2.9.1 (CUDA 13.0), a pinned commit of
`transformers` and a prebuilt flash-attn wheel matching your architecture.
Nothing is compiled.

## Data

CountVideoBench (2,600 videos, 139 GB) can be read in three ways. All of them
read the same files and give the same results.

| Command | Where the videos are | Disk space |
| --- | --- | --- |
| `uv run evaluate.py --model 8B` | Hugging Face cache (`HF_HOME`) | up to 139 GB, reused by later runs |
| `uv run evaluate.py --model 8B --streaming` | Temporary folder, each video deleted after use | about 50 MB |
| `uv run evaluate.py --model 8B --data-dir PATH` | Existing local copy | none |

**Default: Hugging Face cache.** Each video is downloaded just before
inference and kept in the Hugging Face cache, so the run starts immediately
and later runs (e.g. the other model) do not download it again. A dataset
already downloaded with `hf download` without `--local-dir` is reused.

**Streaming.** Same download, but each video is deleted right after
inference. Useful when disk space is limited.

**Local copy.** For a copy in a folder of your choice:

```bash
uv run hf download countvideobench/CountVideoBench --repo-type dataset \
    --revision 42e8fed00bca018a8fc10e249673e665aaa352b8 --local-dir CountVideoBench
uv run evaluate.py --model 8B --data-dir CountVideoBench
```

The first two modes use the dataset revision evaluated in the paper; use
`--revision` to choose another one.

The model weights are downloaded from the Hugging Face Hub on first use, at
the revision used in the paper (`--model-path` accepts a local folder instead).

## Usage

```bash
uv run evaluate.py --model 8B
uv run evaluate.py --model 32B
```

Main options (`uv run evaluate.py --help` for the full list):

| Option | Description |
| --- | --- |
| `--model {8B,32B}` | Model size (required). |
| `--streaming` / `--data-dir PATH` | Data source (default: Hugging Face cache, see [Data](#data)). |
| `--output PATH` | Output CSV. Default: `results/Qwen3-VL-<model>_countvideobench.csv`. |
| `--limit N` | Only evaluate the first N videos, for a quick test. |
| `--device-map` | Default `auto`: the model is spread over all visible GPUs. Use `CUDA_VISIBLE_DEVICES` to choose them. |

**Resuming.** Each prediction is written as soon as it is produced. If the
output file already exists, the videos it contains are skipped, so an
interrupted run can be resumed with the same command. Delete the file to start
over.

A warning saying that torchvision video decoding is deprecated is printed for
each video. It is expected: this is the decoder used for the paper (see
[Settings](#settings)).

## Output

One CSV row per video. The first six columns come from the dataset's
`metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `Qwen3-VL-8B` or `Qwen3-VL-32B` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

## Settings

The model is used as described in its model card, with the default settings
of its processor:

- the video is passed to the processor's chat template together with the
  dataset's `question`;
- the processor decodes the video with `torchvision` (PyAV backend, since
  `torchcodec` is not installed) and samples it at 2 fps, i.e. 10 frames per
  5-second video;
- greedy decoding, at most 1024 new tokens;
- bfloat16, flash-attention 2, seed 42, deterministic cuDNN.

## Reproducibility

- `uv.lock` pins every package to the version used for the paper.
- The paper's results were obtained on aarch64 (GH200), with exactly the
  flash-attn wheel used here for that architecture. On other GPUs, outputs may
  differ slightly because of floating-point differences between hardware.
- The flash-attn wheels come from
  [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
  (BSD-3-Clause). Their sha256 checksums are recorded in `uv.lock`.

## Citation

If you use this code, please cite CountVideoBench and Qwen3-VL:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}

@article{bai2025qwen3vl,
  author  = {Bai, Shuai and others},
  title   = {{Qwen3-VL} Technical Report},
  journal = {arXiv preprint arXiv:2511.21631},
  year    = {2025}
}
```
