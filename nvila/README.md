# NVILA on CountVideoBench

Code to reproduce the results of **NVILA-8B** and **NVILA-15B** on
[CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench),
as reported in the CountVideoBench paper (BMVC 2026).

NVILA is run with the official VILA code
([NVlabs/VILA](https://github.com/NVlabs/VILA)), installed by uv at a pinned
commit (see [Third-party code](#third-party-code)).

## Requirements

- Linux on **x86_64** or **aarch64** (e.g. NVIDIA GH200).
- An NVIDIA GPU with a driver that supports **CUDA 12.8** (driver 570 or newer).
  The CUDA libraries themselves are installed by uv; no system CUDA toolkit is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) and `git`.
- GPU memory: the weights alone take about 17 GB (8B) and 30 GB (15B) in
  float16. The paper's results were obtained on 2 × GH200 144 GB.

## Installation

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/nvila
uv sync
```

`uv sync` installs Python 3.12 and the exact versions of all dependencies
recorded in `uv.lock`, including PyTorch 2.10.0 (CUDA 12.8), a pinned commit
of VILA and a prebuilt flash-attn wheel matching your architecture. VILA and
DeepSpeed are installed from source in a few seconds; nothing is compiled.

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
uv run evaluate.py --model 15B
```

Main options (`uv run evaluate.py --help` for the full list):

| Option | Description |
| --- | --- |
| `--model {8B,15B}` | Model size (required). |
| `--streaming` / `--data-dir PATH` | Data source (default: Hugging Face cache, see [Data](#data)). |
| `--output PATH` | Output CSV. Default: `results/NVILA-<model>_countvideobench.csv`. |
| `--limit N` | Only evaluate the first N videos, for a quick test. |
| `--device-map` | Default `auto`: the model is spread over all visible GPUs. Use `CUDA_VISIBLE_DEVICES` to choose them. |

**Resuming.** Each prediction is written as soon as it is produced. If the
output file already exists, the videos it contains are skipped, so an
interrupted run can be resumed with the same command. Delete the file to start
over.

Messages printed by VILA's dependencies at start-up (a DeepSpeed accelerator
notice, a `FutureWarning` from timm about `timm.models.layers`) are harmless.

## Output

One CSV row per video. The first six columns come from the dataset's
`metadata.csv`:

| Column | Description |
| --- | --- |
| `model` | `NVILA-8B` or `NVILA-15B` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

## Settings

NVILA is used through VILA's `generate_content`, as in VILA's `vila-infer`
command:

- 10 frames per video, sampled uniformly by VILA (OpenCV) over the whole video;
- prompt: the video followed by the question, with the chat template of the model;
- greedy decoding with VILA's default generation settings, at most 1024 new
  tokens (the limit used for every model in the paper);
- float16 (set by VILA's loader), flash-attention 2 in the vision tower, seed 42,
  deterministic cuDNN.

## Reproducibility

- `uv.lock` pins every package to the version used for the paper.
- VILA is installed without the dependencies declared in its `pyproject.toml`
  (which pin older versions, e.g. PyTorch 2.3.0), as for the paper. The
  packages it needs are listed in this folder's `pyproject.toml`.
- The paper's results were obtained on aarch64 (GH200), with exactly the
  flash-attn wheel used here for that architecture. On other GPUs, outputs may
  differ slightly because of floating-point differences between hardware.
- The flash-attn wheels come from
  [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
  (BSD-3-Clause). Their sha256 checksums are recorded in `uv.lock`.

## Third-party code

VILA ([NVlabs/VILA](https://github.com/NVlabs/VILA), commit `b760c34`,
Apache-2.0 license) is not copied in this repository: uv installs it from
GitHub at that commit, unmodified. The NVILA weights are distributed by NVIDIA
under a non-commercial license (CC-BY-NC-4.0, see the model cards of
[NVILA-8B](https://huggingface.co/Efficient-Large-Model/NVILA-8B) and
[NVILA-15B](https://huggingface.co/Efficient-Large-Model/NVILA-15B)).

## Citation

If you use this code, please cite CountVideoBench and NVILA:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}

@article{liu2024nvila,
  author  = {Liu, Zhijian and Zhu, Ligeng and Shi, Baifeng and Zhang, Zhuoyang and Lou, Yuming and Yang, Shang and Xi, Haocheng and Cao, Shiyi and Gu, Yuxian and Li, Dacheng and Li, Xiuyu and Fang, Yunhao and Chen, Yukang and Hsieh, Cheng-Yu and Huang, De-An and Cheng, An-Chieh and Nath, Vishwesh and Hu, Jinyi and Liu, Sifei and Krishna, Ranjay and Xu, Daguang and Wang, Xiaolong and Molchanov, Pavlo and Kautz, Jan and Yin, Hongxu and Han, Song and Lu, Yao},
  title   = {{NVILA}: Efficient Frontier Visual Language Models},
  journal = {arXiv preprint arXiv:2412.04468},
  year    = {2024}
}
```
