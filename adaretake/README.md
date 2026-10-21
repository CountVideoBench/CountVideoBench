# AdaReTaKe on CountVideoBench

Code to reproduce the results of **AdaReTaKe** (Qwen2.5-VL-7B and
Qwen2.5-VL-72B) on [CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

AdaReTaKe is a training-free method that compresses the visual tokens of long
videos. Its code is included in `retake/` (see [Third-party code](#third-party-code)).

## Requirements

- Linux on **x86_64** or **aarch64** (e.g. NVIDIA GH200).
- An NVIDIA GPU with a driver that supports **CUDA 13.0** (driver 580 or newer).
  The CUDA libraries themselves are installed by uv; no system CUDA toolkit is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GPU memory: the weights alone take about 16 GB (7B) and 145 GB (72B) in
  bfloat16. The paper's results were obtained on 2 × GH200 144 GB.

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/adaretake
uv sync
```

`uv sync` installs Python 3.11 and the exact versions of all dependencies
recorded in `uv.lock`, including PyTorch 2.9.1 (CUDA 13.0), a pinned commit of
`transformers` and a prebuilt flash-attn wheel matching your architecture.
Nothing is compiled.

## Data

CountVideoBench (2,600 videos, 139 GB) can be read in three ways. All of them
read the same files and give the same results.

| Command | Where the videos are | Disk space |
| --- | --- | --- |
| `uv run evaluate.py --model 7B` | Hugging Face cache (`HF_HOME`) | up to 139 GB, reused by later runs |
| `uv run evaluate.py --model 7B --streaming` | Temporary folder, each video deleted after use | about 50 MB |
| `uv run evaluate.py --model 7B --data-dir PATH` | Existing local copy | none |

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
uv run evaluate.py --model 7B --data-dir CountVideoBench
```

The first two modes use the dataset revision evaluated in the paper; use
`--revision` to choose another one.

The model weights are downloaded from the Hugging Face Hub on first use
(`--model-path` accepts a local folder instead).

## Usage

```bash
uv run evaluate.py --model 7B
uv run evaluate.py --model 72B
```

Main options (`uv run evaluate.py --help` for the full list):

| Option | Description |
| --- | --- |
| `--model {7B,72B}` | Model size (required). |
| `--streaming` / `--data-dir PATH` | Data source (default: Hugging Face cache, see [Data](#data)). |
| `--output PATH` | Output CSV. Default: `results/Adaretake-<model>_countvideobench.csv`. |
| `--limit N` | Only evaluate the first N videos, for a quick test. |
| `--device-map` | Default `auto`: the model is spread over all visible GPUs. Use `CUDA_VISIBLE_DEVICES` to choose them. |

**Resuming.** Each prediction is written as soon as it is produced. If the
output file already exists, the videos it contains are skipped, so an
interrupted run can be resumed with the same command. Delete the file to start
over.

## Output

One CSV row per video. The first six columns come from the dataset's
`metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `Adaretake-7B` or `Adaretake-72B` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

## Settings

The settings used in the paper are in `configs/countvideobench.yaml`. They are
those of AdaReTaKe's demo configuration, except for the sampling rate:

- frames sampled at 2 fps (10 frames per 5-second video), long side resized to 448 px;
- AdaReTaKe compression with temporal adaptation and AdaKV layer budget allocation;
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

## Third-party code

`retake/` is a copy of the official AdaReTaKe implementation
([iLearn-Lab/ACL25-AdaReTaKe](https://github.com/iLearn-Lab/ACL25-AdaReTaKe),
commit `a16a662`), distributed under the MIT license (`retake/LICENSE`). The
upstream README also states that the code is for research use only. One line
is added to `retake/llava_onevision.py`; see `retake/NOTICE.md` for details.

## Citation

If you use this code, please cite CountVideoBench and AdaReTaKe:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}

@article{wang2025adaretake,
  author  = {Wang, Xiao and Si, Qingyi and Wu, Jianlong and Zhu, Shiyu and Cao, Li and Nie, Liqiang},
  title   = {{AdaReTaKe}: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding},
  journal = {arXiv preprint arXiv:2503.12559},
  year    = {2025}
}
```
