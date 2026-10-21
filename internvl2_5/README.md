# InternVL2.5 on CountVideoBench

Code to reproduce the results of **InternVL2.5-8B** and **InternVL2.5-78B** on
[CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

The model code is the official InternVL2.5 implementation, loaded from the
Hugging Face repositories of the weights (`trust_remote_code=True`) at the
revisions used in the paper.

## Requirements

- Linux on **x86_64** or **aarch64** (e.g. NVIDIA GH200).
- An NVIDIA GPU with a driver that supports **CUDA 13.0** (driver 580 or newer).
  The CUDA libraries themselves are installed by uv; no system CUDA toolkit is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GPU memory: the models are loaded with 8-bit weights, which take about
  9 GB (8B) and 80 GB (78B). The paper's results were obtained on 2 × GH200 144 GB.

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/internvl2_5
uv sync
```

`uv sync` installs Python 3.12 and the exact versions of all dependencies
recorded in `uv.lock`, including PyTorch 2.10.0 (CUDA 13.0), transformers
4.37.2 and a prebuilt flash-attn wheel matching your architecture. Nothing is
compiled.

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

The model weights are downloaded from the Hugging Face Hub on first use
(`--model-path` accepts a local folder instead).

## Usage

```bash
uv run evaluate.py --model 8B
uv run evaluate.py --model 78B
```

Main options (`uv run evaluate.py --help` for the full list):

| Option | Description |
| --- | --- |
| `--model {8B,78B}` | Model size (required). |
| `--streaming` / `--data-dir PATH` | Data source (default: Hugging Face cache, see [Data](#data)). |
| `--output PATH` | Output CSV. Default: `results/InternVL2_5-<model>_countvideobench.csv`. |
| `--limit N` | Only evaluate the first N videos, for a quick test. |
| `--device-map` | Default `split`: the layer split of the InternVL2.5 model card, over all visible GPUs (use `CUDA_VISIBLE_DEVICES` to choose them). Any other value, e.g. `auto`, is passed to `from_pretrained`. |

**Resuming.** Each prediction is written as soon as it is produced. If the
output file already exists, the videos it contains are skipped, so an
interrupted run can be resumed with the same command. Delete the file to start
over.

## Output

One CSV row per video. The first six columns come from the dataset's
`metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `InternVL2_5-8B` or `InternVL2_5-78B` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

## Settings

The settings used in the paper are defined at the top of `evaluate.py`. Video
loading follows the InternVL2.5 model card:

- 10 frames sampled uniformly (one in the middle of each of 10 equal segments);
- each frame resized to a single 448 × 448 tile (no dynamic tiling);
- prompt: `Frame1: <image>\n ... Frame10: <image>\n` followed by the question;
- greedy decoding, at most 1024 new tokens;
- **8-bit weights** (bitsandbytes `load_in_8bit`), bfloat16 activations,
  flash-attention 2, seed 42, deterministic cuDNN.

## Reproducibility

- `uv.lock` pins every package to the version used for the paper.
- The paper's results were obtained on aarch64 (GH200), with exactly the
  flash-attn wheel used here for that architecture. On other GPUs, outputs may
  differ slightly because of floating-point differences between hardware.
- The flash-attn wheels come from
  [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
  (BSD-3-Clause). Their sha256 checksums are recorded in `uv.lock`.
- Videos are decoded with `decord2`, a maintained fork of decord that provides
  the same `decord` module and has wheels for x86_64 and aarch64.

## Citation

If you use this code, please cite CountVideoBench and InternVL2.5:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}

@article{chen2024internvl25,
  author  = {Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  title   = {Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  journal = {arXiv preprint arXiv:2412.05271},
  year    = {2024}
}
```
