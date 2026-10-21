# CountVideoBench

Evaluation code for the models analysed in the CountVideoBench paper (BMVC 2026).

CountVideoBench is a benchmark of 2,600 synthetic videos (5 s, 30 fps,
rendered with Blender) to evaluate the prediction biases of video-language
models on the object counting task. Each video shows 5 to 50 identical static
objects (cubes or spheres in one of five colors) filmed by a rotating camera,
and comes with a question asking how many objects there are.

- Dataset: [countvideobench/CountVideoBench](https://huggingface.co/datasets/countvideobench/CountVideoBench) on Hugging Face (CC-BY-4.0).
- Code: this repository, one folder per model, plus [`eval/`](eval/) to
  extract the predicted counts from the models' answers.

## Models

| Model | Sizes | Folder | Runs on |
| --- | --- | --- | --- |
| AdaReTaKe (Qwen2.5-VL) | 7B, 72B | [`adaretake/`](adaretake/) | NVIDIA GPU, CUDA 13.0 |
| InternVL2.5 | 8B, 78B | [`internvl2_5/`](internvl2_5/) | NVIDIA GPU, CUDA 13.0 |
| LLaVA-Video (Qwen2) | 7B, 72B | [`llava_video/`](llava_video/) | NVIDIA GPU, CUDA 13.0 |
| NVILA | 8B, 15B | [`nvila/`](nvila/) | NVIDIA GPU, CUDA 12.8 |
| Qwen2.5-VL-Instruct | 7B, 72B | [`qwen2_5_vl/`](qwen2_5_vl/) | NVIDIA GPU, CUDA 13.0 |
| Qwen3-VL-Instruct | 8B, 32B | [`qwen3_vl/`](qwen3_vl/) | NVIDIA GPU, CUDA 13.0 |
| VideoLLaMA 2 | 7B, 72B | [`videollama2/`](videollama2/) | NVIDIA GPU, CUDA 12.8 |
| GPT-5.4 | – | [`gpt_5_4/`](gpt_5_4/) | OpenAI Batch API (no GPU) |
| Gemini 3.1 Pro | – | [`gemini_3_1_pro/`](gemini_3_1_pro/) | Gemini Batch API (no GPU) |

Each folder is an independent [uv](https://docs.astral.sh/uv/) project: the
models need different versions of PyTorch and `transformers`, so each one has
its own `pyproject.toml` and `uv.lock`, which pin every package to the version
used for the paper. Each folder's README gives the requirements, options and
settings of its model.

## Quick start

Requires Linux (x86_64 or aarch64) and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/CountVideoBench/CountVideoBench
cd CountVideoBench/qwen2_5_vl    # or any other model folder
uv sync
uv run evaluate.py --model 7B --limit 5    # quick test on 5 videos
uv run evaluate.py --model 7B              # full benchmark
uv run evaluate.py --model 7B --streaming  # full benchmark, each video deleted after use
```

The API models (`gpt_5_4/`, `gemini_3_1_pro/`) run in several steps
(`prepare`, `submit`, `status`, `collect`); see their READMEs.

**Data.** The videos are downloaded from Hugging Face as they are needed, at
the dataset revision evaluated in the paper. By default they are kept in the
Hugging Face cache (139 GB in total), so the other models do not download
them again. `--streaming` deletes each video after use, and `--data-dir PATH`
reads an existing local copy.

**Resuming.** Each answer is written to the CSV as soon as it is produced. An
interrupted run started again with the same command skips the videos already
in the CSV.

## Output format

Every model writes one CSV row per video, by default to
`results/<Model>-<size>_countvideobench.csv` in its folder:

| Column | Description |
| --- | --- |
| `model` | Model name and size, e.g. `Qwen2.5-VL-7B` |
| `video_id` | Video identifier (`vid_000001` to `vid_002600`) |
| `count` | Ground-truth number of objects |
| `shape`, `color` | Object class |
| `question` | Prompt given to the model |
| `raw_answer` | Model output, unprocessed |

The first six columns come from the dataset's `single_class/metadata.csv`.

## Extracting the counts

The answers are free text ("There are ten red cubes in this video."). The
[`eval/`](eval/) folder extracts the count stated in each answer with a local
LLM (Qwen2.5-7B-Instruct), as in the paper:

```bash
cd CountVideoBench/eval
uv sync
uv run extract.py ../qwen2_5_vl/results/Qwen2.5-VL-7B_countvideobench.csv
```

It also contains, in `eval/paper_results/`, the counts used in the paper for
every model, checked by hand after extraction. See [`eval/README.md`](eval/README.md).

## Evaluate your own model

The script below runs any model on CountVideoBench and writes a CSV in the
format above. Fill in `predict` and `MODEL`, save it as `my_eval.py` and run
`uv run my_eval.py` (uv reads the dependencies from the header; add your
model's own). Each video is downloaded to a temporary folder and deleted after
use, and an interrupted run resumes where it stopped.

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub"]
# ///
import csv
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "countvideobench/CountVideoBench"
REVISION = "42e8fed00bca018a8fc10e249673e665aaa352b8"  # revision used in the paper
MODEL = "MyModel-7B"
OUTPUT = Path(f"results/{MODEL}_countvideobench.csv")
FIELDS = ["model", "video_id", "count", "shape", "color", "question", "raw_answer"]


def predict(video_path, question):
    """Run your model on one video and return its raw text answer."""
    raise NotImplementedError


def fetch(filename, folder):
    return hf_hub_download(REPO, f"single_class/{filename}", repo_type="dataset",
                           revision=REVISION, local_dir=folder)


with tempfile.TemporaryDirectory() as tmp:
    with open(fetch("metadata.csv", tmp), newline="") as f:
        rows = list(csv.DictReader(f))

    done = set()
    if OUTPUT.exists():
        with open(OUTPUT, newline="") as f:
            done = {row["video_id"] for row in csv.DictReader(f)}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if out.tell() == 0:
            writer.writeheader()
        for row in rows:
            if row["video_id"] in done:
                continue
            video = fetch(row["file_name"], tmp)
            answer = predict(video, row["question"])
            os.remove(video)
            writer.writerow({**{k: row[k] for k in FIELDS[1:6]},
                             "model": MODEL, "raw_answer": answer})
            out.flush()
```

In the paper, every model sees **10 frames per video**, sampled uniformly over
the 5 seconds, and answers with greedy decoding (temperature 0). The counts
can then be extracted from the CSV with [`eval/`](eval/), like for the other
models.

## License

The code in this repository is released under the [MIT License](LICENSE),
except the third-party code copied into it, which keeps its original license:

- `adaretake/retake/`: AdaReTaKe, MIT License (`adaretake/retake/LICENSE`);
- `videollama2/videollama2/`: VideoLLaMA 2, Apache-2.0 License (`videollama2/videollama2/LICENSE`).

Third-party code installed by uv (LLaVA-NeXT, VILA, …) is not copied here and
remains under its own license. The dataset is released under CC-BY-4.0. The
model weights and APIs are not part of this repository and are subject to
their own licenses and terms of use; for example, the NVILA weights are under
a non-commercial license (CC-BY-NC-4.0).

## Citation

If you use CountVideoBench or this code, please cite:

```bibtex
@inproceedings{chapus2026countvideobench,
  author    = {Chapus, Marc and Crispim-Junior, Carlos and Eglin, V{\'e}ronique and Baskurt, Atilla},
  title     = {{CountVideoBench}: A Benchmark to Evaluate Prediction Biases on the Object Counting task by Video-Language Models},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}
```

Please also cite the models you evaluate.
