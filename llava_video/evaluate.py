"""Evaluate LLaVA-Video (Qwen2 7B / 72B) on CountVideoBench.

Examples:
    uv run evaluate.py --model 7B                    # videos cached in HF_HOME
    uv run evaluate.py --model 7B --streaming        # each video deleted after use
    uv run evaluate.py --model 72B --data-dir /path/to/CountVideoBench

Predictions are written to a CSV file, one row per video. If the output file
already exists, videos it already contains are skipped, so an interrupted run
can be resumed by running the same command again.
"""

import argparse
import copy
import csv
import os
import random
import sys
import tempfile
import warnings
from pathlib import Path

# Disable the hf-xet chunk cache: it would keep a second copy of downloaded
# data on disk (and defeat the purpose of streaming mode).
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import numpy as np
from tqdm import tqdm

# LLaVA-NeXT and transformers print many harmless warnings (as in the
# original evaluation script).
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

SEED = 42

# Hub id and revision of the weights used in the paper.
MODELS = {
    "7B": ("lmms-lab/LLaVA-Video-7B-Qwen2", "013210b3aff822f1558b166d39c1046dd109520f"),
    "72B": ("lmms-lab/LLaVA-Video-72B-Qwen2", "226cce801d38f1ace080cbbf335a573500a9ce6f"),
}
# Vision tower loaded by LLaVA-NeXT (named in the model's config.json).
VISION_TOWER = ("google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3")

DATASET_REPO = "countvideobench/CountVideoBench"
DATASET_SUBSET = "single_class"
DATASET_REVISION = "42e8fed00bca018a8fc10e249673e665aaa352b8"

OUTPUT_FIELDS = ["model", "video_id", "count", "shape", "color", "question", "raw_answer"]

# Settings used in the paper.
NUM_FRAMES = 10
CONV_TEMPLATE = "qwen_1_5"
MAX_NEW_TOKENS = 1024


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
# Video loading and prompt (LLaVA-Video model card example, with 10 frames)
# ----------------------------------------------------------------------------

def load_video(video_path, num_frames):
    """Return `num_frames` uniformly sampled frames, their timestamps and the video duration.

    frames: uint8 array (num_frames, H, W, 3), RGB
    frame_time: timestamps as a string, e.g. "0.00s,0.53s,..."
    video_time: duration in seconds
    """
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    video_time = total_frame_num / avg_fps
    frame_idx = np.linspace(0, total_frame_num - 1, num_frames, dtype=int).tolist()
    frame_time = ",".join(f"{i / avg_fps:.2f}s" for i in frame_idx)
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time, video_time


def build_prompt(question, num_frames, frame_time, video_time):
    """User message: the video placeholder, the timing instruction, then the dataset question."""
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    # Kept verbatim from the paper's script (including the missing space
    # before "Please").
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly "
        f"sampled from it. These frames are located at {frame_time}.Please answer the following "
        f"questions related to this video."
    )
    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{question}")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


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

def pin_vision_tower_revision():
    """Load the SigLIP vision tower at the revision used in the paper.

    LLaVA-NeXT loads it by name only, i.e. at whatever its latest revision is.
    This adds the revision to that call, without modifying LLaVA-NeXT.
    """
    from llava.model.multimodal_encoder import siglip_encoder

    model_cls = siglip_encoder.SigLipVisionModel
    original = model_cls.from_pretrained

    def from_pretrained(name, *args, **kwargs):
        if name == VISION_TOWER[0]:
            kwargs.setdefault("revision", VISION_TOWER[1])
        return original(name, *args, **kwargs)

    model_cls.from_pretrained = staticmethod(from_pretrained)


def load_model(model_path, revision, device_map):
    """Load LLaVA-Video with LLaVA-NeXT (bfloat16, flash-attention 2)."""
    from llava.model.builder import load_pretrained_model

    if revision is not None:
        # load_pretrained_model has no revision argument: download that exact
        # revision (or reuse it from the Hugging Face cache) and load it from disk.
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            model_path, revision=revision, ignore_patterns=["training_args.bin", "trainer_state.json"]
        )

    pin_vision_tower_revision()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model, image_processor


def inference(tokenizer, model, image_processor, video_path, question, device):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    frames, frame_time, video_time = load_video(video_path, NUM_FRAMES)
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device).bfloat16()

    prompt = build_prompt(question, len(video), frame_time, video_time)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    # Greedy decoding (LLaVA-NeXT's generate already runs under torch.no_grad).
    output_ids = model.generate(
        input_ids,
        images=[video],
        modalities=["video"],
        do_sample=False,
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLaVA-Video (Qwen2) on CountVideoBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True,
                        help="Size of the LLaVA-Video model.")
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
                             "(default: the official LLaVA-Video-Qwen2 checkpoint).")
    parser.add_argument("--model-revision", default=None,
                        help="Revision of the model weights on the Hub "
                             "(default: the revision used in the paper, when --model-path is not set).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: results/LLaVA-Video-<model>_countvideobench.csv).")
    parser.add_argument("--device-map", default="auto",
                        help="Passed to LLaVA-NeXT's loader; 'auto' spreads the model over all visible "
                             "GPUs (setting used in the paper).")
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
    model_name = f"LLaVA-Video-{args.model}"
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
    tokenizer, model, image_processor = load_model(model_path, model_revision, args.device_map)
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
                answer = inference(tokenizer, model, image_processor, video_path, row["question"], device)
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
