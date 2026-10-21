"""Evaluate NVILA (8B / 15B) on CountVideoBench.

Examples:
    uv run evaluate.py --model 8B                    # videos cached in HF_HOME
    uv run evaluate.py --model 8B --streaming        # each video deleted after use
    uv run evaluate.py --model 15B --data-dir /path/to/CountVideoBench

Predictions are written to a CSV file, one row per video. If the output file
already exists, videos it already contains are skipped, so an interrupted run
can be resumed by running the same command again.
"""

import argparse
import csv
import os
import random
import re
import sys
import tempfile
from pathlib import Path

# Disable the hf-xet chunk cache: it would keep a second copy of downloaded
# data on disk (and defeat the purpose of streaming mode).
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

SEED = 42

# Number of frames given to the model for each video (protocol of the paper).
NUM_FRAMES = 10

# Limit on generated tokens, the same for every model evaluated in the paper.
MAX_NEW_TOKENS = 1024

# Hub id and revision of the weights used in the paper. Each repository
# contains the whole model (language model, vision tower and projector).
MODELS = {
    "8B": ("Efficient-Large-Model/NVILA-8B", "049743cc51bdd8872cdf696a1a58b03fef2c367e"),
    "15B": ("Efficient-Large-Model/NVILA-15B", "e621f628ee5c160c4444e4a29e395bb60dd30b78"),
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

def resolve_model_path(model_path, revision):
    """Local folder of the weights (VILA's loader has no `revision` argument)."""
    if Path(model_path).is_dir():
        return str(model_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_path, revision=revision)


def load_model(model_path, device_map):
    """Load NVILA with VILA's loader (float16, as set by VILA)."""
    import llava
    from llava import conversation as clib

    model = llava.load(model_path, model_base=None, device_map=device_map)
    model.eval()

    # Uniform sampling of NUM_FRAMES frames over the whole video (fps = 0
    # disables VILA's fixed-rate sampling; it is also the value of the config).
    # Without this, NVILA would use 8 frames (`num_video_frames` of its config).
    model.config.num_video_frames = NUM_FRAMES
    model.config.fps = 0

    # Chat template of the tokenizer.
    clib.default_conversation = clib.conv_templates["auto"].copy()
    return model


def decode_time_token(text, *, duration, num_time_tokens, time_token_format):
    """Replace time tokens in text with timestamps (from VILA's `vila-infer`).

    NVILA's config has `num_time_tokens = 0`: only tokens such as `<t3>` that
    may appear in the answer are replaced by the duration of the video.
    """
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        timestamp = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{timestamp}>")

    # Handle out-of-range time tokens
    for match in re.findall(r"<t(\d+)>", text):
        t = int(match)
        if t >= num_time_tokens:
            text = text.replace(f"<t{t}>", f"<{round(duration, 2)}>")
    return text


def video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return duration


def inference(model, video_path, question):
    from llava.media import Video

    # VILA reads the frames itself (OpenCV, NUM_FRAMES frames at
    # round(linspace(0, n_frames - 1, NUM_FRAMES))). Greedy decoding with VILA's
    # default generation settings, limited to MAX_NEW_TOKENS new tokens.
    generation_config = model.default_generation_config  # a fresh copy at each call
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    response = model.generate_content(
        [Video(str(video_path)), question], generation_config=generation_config
    )

    config = model.config
    if config.num_time_tokens is not None and config.time_token_format is not None:
        response = decode_time_token(
            response,
            duration=video_duration(video_path),
            num_time_tokens=config.num_time_tokens,
            time_token_format=config.time_token_format,
        )
    return response


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NVILA on CountVideoBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True,
                        help="Size of the NVILA model.")
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
                        help="Local folder or Hub id of the model weights "
                             "(default: the official NVILA checkpoint).")
    parser.add_argument("--model-revision", default=None,
                        help="Revision of the model weights on the Hub "
                             "(default: the revision used in the paper, when --model-path is not set).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: results/NVILA-<model>_countvideobench.csv).")
    parser.add_argument("--device-map", default="auto",
                        help="Passed to VILA's loader; 'auto' spreads the model over all visible GPUs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N videos (for a quick test).")
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
    model_name = f"NVILA-{args.model}"
    output_path = args.output or ROOT / "results" / f"{model_name}_countvideobench.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    model = load_model(resolve_model_path(model_path, model_revision), args.device_map)

    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        for row in tqdm(todo, desc=model_name, unit="video",
                        initial=len(rows) - len(todo), total=len(rows)):
            video_path = dataset.video_path(row)
            try:
                answer = inference(model, video_path, row["question"])
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
