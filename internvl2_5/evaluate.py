"""Evaluate InternVL2.5 (8B / 78B) on CountVideoBench.

Examples:
    uv run evaluate.py --model 8B                    # videos cached in HF_HOME
    uv run evaluate.py --model 8B --streaming        # each video deleted after use
    uv run evaluate.py --model 78B --data-dir /path/to/CountVideoBench

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
from pathlib import Path

# Disable the hf-xet chunk cache: it would keep a second copy of downloaded
# data on disk (and defeat the purpose of streaming mode).
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

SEED = 42

# Hub id, revision of the weights used in the paper, and number of layers of
# the language model (used to split it over the GPUs).
MODELS = {
    "8B": ("OpenGVLab/InternVL2_5-8B", "e9e4c0dc1db56bfab10458671519b7fa3dd29463", 32),
    "78B": ("OpenGVLab/InternVL2_5-78B", "011df531c2c8c87b53f638d999f0e21ff6abf84b", 80),
}

DATASET_REPO = "countvideobench/CountVideoBench"
DATASET_SUBSET = "single_class"
DATASET_REVISION = "42e8fed00bca018a8fc10e249673e665aaa352b8"

OUTPUT_FIELDS = ["model", "video_id", "count", "shape", "color", "question", "raw_answer"]

# Settings used in the paper.
NUM_FRAMES = 10        # frames sampled uniformly from each video
INPUT_SIZE = 448       # each frame is resized to 448 x 448
MAX_TILES = 1          # one tile per frame (no dynamic tiling)
GENERATION_CONFIG = dict(max_new_tokens=1024, do_sample=False, num_beams=1)  # greedy decoding

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
# Video loading (from the InternVL2.5 model card)
# ----------------------------------------------------------------------------

def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Split `image` into at most `max_num` tiles of `image_size` x `image_size`.

    With `max_num=1` (the setting used here), the frame is simply resized to a
    single tile.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Indices of `num_segments` frames, one in the middle of each equal segment."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Return the frames as a (num_tiles, 3, H, W) tensor and the number of tiles per frame."""
    import torch
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in img])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


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

def split_model(num_layers):
    """Device map from the InternVL2.5 model card.

    The vision encoder, the embeddings, the output layer and the last LLM layer
    go on GPU 0, which therefore receives only half as many LLM layers as the
    other GPUs.
    """
    import torch

    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_model(model_path, revision, num_layers, device_map):
    import torch
    from transformers import AutoModel, AutoTokenizer

    # The InternVL remote code silently falls back to standard attention when
    # flash-attn cannot be imported; fail early instead.
    import flash_attn  # noqa: F401

    if device_map == "split":
        device_map = split_model(num_layers)

    model = AutoModel.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


def inference(model, tokenizer, video_path, question):
    import torch

    pixel_values, num_patches_list = load_video(
        video_path, input_size=INPUT_SIZE, max_num=MAX_TILES, num_segments=NUM_FRAMES
    )
    device = next(model.vision_model.parameters()).device
    pixel_values = pixel_values.to(torch.bfloat16).to(device)

    # Frame1: <image>\nFrame2: <image>\n...\nFrame10: <image>\n{question}
    video_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    return model.chat(
        tokenizer,
        pixel_values,
        video_prefix + question,
        dict(GENERATION_CONFIG),
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate InternVL2.5 on CountVideoBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True,
                        help="Size of the InternVL2.5 model.")
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
                             "(default: the official InternVL2.5 checkpoint).")
    parser.add_argument("--model-revision", default=None,
                        help="Revision of the model weights on the Hub "
                             "(default: the revision used in the paper, when --model-path is not set).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: results/InternVL2_5-<model>_countvideobench.csv).")
    parser.add_argument("--device-map", default="split",
                        help="'split' (paper setting): the model card's layer split over all visible "
                             "GPUs. Any other value (e.g. 'auto', 'cuda:0') is passed to from_pretrained.")
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
    model_name = f"InternVL2_5-{args.model}"
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

    default_path, default_revision, num_layers = MODELS[args.model]
    if args.model_path is None:
        model_path, model_revision = default_path, default_revision
    else:
        model_path, model_revision = args.model_path, None
    model_revision = args.model_revision or model_revision

    set_determinism()
    model, tokenizer = load_model(model_path, model_revision, num_layers, args.device_map)

    write_header = not output_path.is_file() or output_path.stat().st_size == 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        for row in tqdm(todo, desc=model_name, unit="video",
                        initial=len(rows) - len(todo), total=len(rows)):
            video_path = dataset.video_path(row)
            try:
                answer = inference(model, tokenizer, video_path, row["question"])
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
