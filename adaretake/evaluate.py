"""Evaluate AdaReTaKe (Qwen2.5-VL 7B / 72B) on CountVideoBench.

Examples:
    uv run evaluate.py --model 7B                    # videos cached in HF_HOME
    uv run evaluate.py --model 7B --streaming        # each video deleted after use
    uv run evaluate.py --model 72B --data-dir /path/to/CountVideoBench

Predictions are written to a CSV file, one row per video. If the output file
already exists, videos it already contains are skipped, so an interrupted run
can be resumed by running the same command again.
"""

import argparse
import csv
import math
import os
import random
import sys
import tempfile
import warnings
from pathlib import Path

# Disable the hf-xet chunk cache: it would keep a second copy of downloaded
# data on disk (and defeat the purpose of streaming mode).
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

SEED = 42

# Hub id and revision of the weights used in the paper.
MODELS = {
    "7B": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "72B": ("Qwen/Qwen2.5-VL-72B-Instruct", "89c86200743eec961a297729e7990e8f2ddbc4c5"),
}

DATASET_REPO = "countvideobench/CountVideoBench"
DATASET_SUBSET = "single_class"
DATASET_REVISION = "42e8fed00bca018a8fc10e249673e665aaa352b8"

OUTPUT_FIELDS = ["model", "video_id", "count", "shape", "color", "question", "raw_answer"]


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def set_determinism(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
# Video loading (same sampling as AdaReTaKe's demo.py)
# ----------------------------------------------------------------------------

def get_frame_indices(total_frames, max_num_frames, sample_fps, extraction_fps):
    """Indices of the frames sampled at `sample_fps`, capped at `max_num_frames`.

    The number of frames is rounded down to an even number.
    """
    sample_frames = float(total_frames / extraction_fps) * sample_fps
    sample_frames = min(total_frames, max_num_frames, sample_frames)
    sample_frames = math.floor(sample_frames)
    sample_frames = int(sample_frames / 2) * 2
    return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)


def load_video(video_path, max_num_frames, sample_fps):
    """Return the sampled frames (RGB PIL images) and the effective sampling fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extraction_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_indices = get_frame_indices(num_total_frames, max_num_frames, sample_fps, extraction_fps)

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            warnings.warn(f"Could not read frame {frame_index} of {video_path}; skipping it.")
    cap.release()

    duration = num_total_frames / extraction_fps
    sampling_fps = len(frames) / duration
    return frames, sampling_fps


def resize_longside(image, max_size):
    """Downscale `image` so that its longest side is at most `max_size`."""
    if max(image.width, image.height) > max_size:
        factor = max_size / max(image.width, image.height)
        width, height = int(image.width * factor), int(image.height * factor)
        image = image.resize((width, height), resample=Image.NEAREST)
    return image


def fetch_video(video_path, config):
    from torchvision.transforms.functional import pil_to_tensor

    frames, sampling_fps = load_video(video_path, config["max_num_frames"], config["sample_fps"])
    frames = [pil_to_tensor(resize_longside(frame, config["longsize_resolution"])) for frame in frames]
    return frames, sampling_fps


# ----------------------------------------------------------------------------
# Dataset access
# ----------------------------------------------------------------------------

def read_metadata(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


class LocalDataset:
    """CountVideoBench downloaded with `hf download ... --local-dir <data_dir>`."""

    def __init__(self, data_dir):
        self.subset_dir = Path(data_dir) / DATASET_SUBSET
        metadata_path = self.subset_dir / "metadata.csv"
        if not metadata_path.is_file():
            sys.exit(f"Error: {metadata_path} not found. --data-dir must point to the "
                     f"downloaded CountVideoBench folder (the one containing '{DATASET_SUBSET}/').")
        self.rows = read_metadata(metadata_path)

    def video_path(self, row):
        return self.subset_dir / row["file_name"]

    def release(self, path):
        pass


class HubDataset:
    """Downloads each video from the Hugging Face Hub just before use.

    With `keep=True` (default mode), files go to the Hugging Face cache
    (HF_HOME) and are reused by later runs. With `keep=False` (streaming mode),
    they go to a temporary folder and each video is deleted after use.
    """

    def __init__(self, revision, keep):
        from huggingface_hub import hf_hub_download

        self._download = hf_hub_download
        self.revision = revision
        self.keep = keep
        self._tmp = None if keep else tempfile.TemporaryDirectory(prefix="countvideobench_")
        self.rows = read_metadata(self._fetch(f"{DATASET_SUBSET}/metadata.csv"))

    def _fetch(self, filename):
        return self._download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=self.revision,
            filename=filename,
            local_dir=None if self.keep else self._tmp.name,
        )

    def video_path(self, row):
        return Path(self._fetch(f"{DATASET_SUBSET}/{row['file_name']}"))

    def release(self, path):
        if not self.keep:
            Path(path).unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(model_path, revision, config, device_map, verbose):
    """Load Qwen2.5-VL with the ReTaKe patches applied."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    import retake.qwen2_5_vl
    from retake.monkeypatch import patch_qwen2_5_vl, patch_qwen2_5_vl_config

    retake.qwen2_5_vl.DEBUG_MODE = verbose  # progress bar over prefill chunks
    patch_qwen2_5_vl(config["method"])

    model_config = Qwen2_5_VLConfig.from_pretrained(model_path, revision=revision)
    model_config = patch_qwen2_5_vl_config(model_config, config)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        revision=revision,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation"),
        device_map=device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, revision=revision)
    return model, processor


def inference(model, processor, video_path, question, config, device):
    import torch

    video, sampling_fps = fetch_video(video_path, config)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt], videos=[video], padding=True, return_tensors="pt", fps=sampling_fps
    )
    inputs = inputs.to(device)
    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)

    # Greedy decoding.
    output_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1024)

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AdaReTaKe (Qwen2.5-VL) on CountVideoBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True,
                        help="Size of the Qwen2.5-VL model.")
    # Default: videos are downloaded on demand into the Hugging Face cache and kept.
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path,
                        help="Use a local copy of CountVideoBench (folder containing 'single_class/') "
                             "instead of the Hugging Face cache.")
    source.add_argument("--streaming", action="store_true",
                        help="Delete each video after use instead of keeping it in the "
                             "Hugging Face cache (uses about 50 MB of disk).")
    parser.add_argument("--revision", default=DATASET_REVISION,
                        help="Dataset revision on the Hub (ignored with --data-dir).")
    parser.add_argument("--model-path", default=None,
                        help="Local path or Hub id of the model weights "
                             "(default: the official Qwen2.5-VL Instruct checkpoint).")
    parser.add_argument("--model-revision", default=None,
                        help="Revision of the model weights on the Hub "
                             "(default: the revision used in the paper, when --model-path is not set).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "countvideobench.yaml",
                        help="AdaReTaKe configuration file.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: results/Adaretake-<model>_countvideobench.csv).")
    parser.add_argument("--device-map", default="auto",
                        help="Passed to from_pretrained; 'auto' spreads the model over all visible GPUs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N videos (for a quick test).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show ReTaKe's progress bar over prefill chunks.")
    return parser.parse_args()


def load_done_ids(output_path):
    """Video ids already present in an existing output file."""
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != OUTPUT_FIELDS:
            sys.exit(f"Error: {output_path} exists but its columns do not match "
                     f"{OUTPUT_FIELDS}. Move or delete it.")
        return {row["video_id"] for row in reader}


def main():
    args = parse_args()
    model_name = f"Adaretake-{args.model}"
    output_path = args.output or ROOT / "results" / f"{model_name}_countvideobench.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.data_dir is not None:
        dataset = LocalDataset(args.data_dir)
    else:
        dataset = HubDataset(args.revision, keep=not args.streaming)
    rows = dataset.rows[: args.limit] if args.limit is not None else dataset.rows

    done = load_done_ids(output_path)
    todo = [row for row in rows if row["video_id"] not in done]
    if done:
        print(f"Resuming: {len(rows) - len(todo)} of {len(rows)} videos already in {output_path}")
    if not todo:
        print("Nothing to do.")
        return

    if args.model_path is None:
        model_path, model_revision = MODELS[args.model]
    else:
        model_path, model_revision = args.model_path, None
    model_revision = args.model_revision or model_revision

    set_determinism()
    model, processor = load_model(model_path, model_revision, config, args.device_map, args.verbose)
    device = "cuda" if args.device_map == "auto" else args.device_map

    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        for row in tqdm(todo, desc=model_name, unit="video",
                        initial=len(rows) - len(todo), total=len(rows)):
            video_path = dataset.video_path(row)
            try:
                answer = inference(model, processor, video_path, row["question"], config, device)
            finally:
                dataset.release(video_path)

            writer.writerow({
                "model": model_name,
                "video_id": row["video_id"],
                "count": row["count"],
                "shape": row["shape"],
                "color": row["color"],
                "question": row["question"],
                "raw_answer": answer,
            })
            f.flush()

    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
