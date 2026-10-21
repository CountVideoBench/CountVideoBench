# VideoLLaMA 2 on CountVideoBench

Code to reproduce the results of **VideoLLaMA 2** (VideoLLaMA2-7B and
VideoLLaMA2-72B) on [CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

The VideoLLaMA 2 inference code is included in `videollama2/` (see
[Third-party code](#third-party-code)).

## Requirements

- Linux on **x86_64** or **aarch64** (e.g. NVIDIA GH200).
- An NVIDIA GPU with a driver that supports **CUDA 12.8** (driver 570 or newer).
  The CUDA libraries themselves are installed by uv; no system CUDA toolkit is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GPU memory: the weights alone take about 15 GB (7B) and 145 GB (72B) in
  float16. The paper's results were obtained on 2 × GH200 144 GB.

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/videollama2
uv sync
```

`uv sync` installs Python 3.12 and the exact versions of all dependencies
recorded in `uv.lock`, including PyTorch 2.10.0 (CUDA 12.8), transformers 5.2.0
and a prebuilt flash-attn wheel matching your architecture. Nothing is compiled.

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
| `--output PATH` | Output CSV. Default: `results/VideoLLaMA2-<model>_countvideobench.csv`. |
| `--limit N` | Only evaluate the first N videos, for a quick test. |
| `--device-map` | Default `auto`: the model is spread over all visible GPUs. Use `CUDA_VISIBLE_DEVICES` to choose them. |

**Resuming.** Each prediction is written as soon as it is produced. If the
output file already exists, the videos it contains are skipped, so an
interrupted run can be resumed with the same command. Delete the file to start
over.

VideoLLaMA 2's code was written for older versions of timm and transformers:
deprecation warnings printed at startup are expected and harmless.

## Output

One CSV row per video. The first six columns come from the dataset's
`metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `VideoLLaMA2-7B` or `VideoLLaMA2-72B` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

## Settings

The model is loaded and queried with VideoLLaMA 2's own `model_init` and
`mm_infer` functions, with these settings:

- 10 frames per video for both sizes, as for every model in the paper
  (VideoLLaMA 2's default is 8 frames for the 7B model and 16 for the 72B
  model). Frames are sampled uniformly (middle of 10 equal segments) and
  resized and center-cropped to 336 × 336 px by the CLIP image processor;
- the dataset's `question` as the prompt, with VideoLLaMA 2's chat template;
- greedy decoding, at most 1024 new tokens (the limit used for every model in
  the paper);
- float16, flash-attention 2 in the vision encoder, seed 42, deterministic cuDNN.

## Reproducibility

- `uv.lock` pins every package to the version used for the paper.
- The model weights are pinned to the revisions used for the paper:
  `b39f289a` (7B) and `b6fb50eb` (72B). The configuration and image processor
  of the vision encoder, which VideoLLaMA 2 loads by name, are pinned too
  (`openai/clip-vit-large-patch14-336`, revision `ce19dc91`).
- The paper's results were obtained on aarch64 (GH200), with exactly the
  flash-attn wheel used here for that architecture. On other GPUs, outputs may
  differ slightly because of floating-point differences between hardware.
- The flash-attn wheels come from
  [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
  (BSD-3-Clause). Their sha256 checksums are recorded in `uv.lock`.

## Third-party code

`videollama2/` is a copy of the inference code of the official VideoLLaMA 2
implementation ([DAMO-NLP-SG/VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2),
commit `c0bb03a`), distributed under the Apache-2.0 license
(`videollama2/LICENSE`). One import is commented out in
`videollama2/model/projector.py` for compatibility with transformers 5; see
`videollama2/NOTICE.md` for details.

## Citation

If you use this code, please cite CountVideoBench and VideoLLaMA 2:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}

@article{damonlpsg2024videollama2,
  author  = {Cheng, Zesen and Leng, Sicong and Zhang, Hang and Xin, Yifei and Li, Xin and Chen, Guanzheng and Zhu, Yongxin and Zhang, Wenqi and Luo, Ziyang and Zhao, Deli and Bing, Lidong},
  title   = {{VideoLLaMA 2}: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs},
  journal = {arXiv preprint arXiv:2406.07476},
  year    = {2024}
}
```
