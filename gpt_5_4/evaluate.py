"""Evaluate GPT-5.4 on CountVideoBench with the OpenAI Batch API.

The evaluation runs in four steps (OPENAI_API_KEY must be set):

    uv run evaluate.py prepare   # extract the frames, write the request files
    uv run evaluate.py submit    # upload them and create the batches
    uv run evaluate.py status    # follow the batches (each one ends within 24 h)
    uv run evaluate.py collect   # download the results and write the CSV

Predictions are written to a CSV file, one row per video. `prepare` skips the
videos already in the CSV, so if some requests failed, running `prepare`,
`submit` and `collect` again completes the file.
"""

import argparse
import base64
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# Disable the hf-xet chunk cache: it would keep a second copy of downloaded
# data on disk (and defeat the purpose of streaming mode).
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# Model snapshot evaluated in the paper, and its name in the CSV.
MODEL = "gpt-5.4-2026-03-05"
MODEL_NAMES = {MODEL: "GPT-5.4"}

# Request settings used in the paper.
NUM_FRAMES = 10
SEED = 42
MAX_COMPLETION_TOKENS = 1024
BATCH_SIZE = 25

DATASET_REPO = "countvideobench/CountVideoBench"
DATASET_SUBSET = "single_class"
DATASET_REVISION = "42e8fed00bca018a8fc10e249673e665aaa352b8"

OUTPUT_FIELDS = ["model", "video_id", "count", "shape", "color", "question", "raw_answer"]

# Batch API.
ENDPOINT = "/v1/chat/completions"
MAX_INPUT_FILE_BYTES = 200 * 1024 * 1024
FINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
TRACKING_FIELDS = ["request_file", "requests", "input_file_id", "batch_id", "status",
                   "completed", "failed", "output_file_id", "error_file_id", "collected"]


# ----------------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------------

def video_frames(video_path, num_frames=NUM_FRAMES):
    """`num_frames` frames taken at indices int(i * N / num_frames), as base64 JPEG."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read frame {index} of {video_path}")
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    return frames


def build_request(video_id, question, frames, model):
    """One line of a Batch API input file: the question, then the frames."""
    return {
        "custom_id": video_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": model,
            "reasoning_effort": "none",
            "temperature": 0,
            "seed": SEED,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame}",
                                    "detail": "high",
                                },
                            }
                            for frame in frames
                        ],
                    ],
                }
            ],
        },
    }


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


def open_dataset(args):
    if args.data_dir is not None:
        return LocalDataset(args.data_dir)
    return HubDataset(args.revision, keep=not args.streaming)


# ----------------------------------------------------------------------------
# Output CSV
# ----------------------------------------------------------------------------

def read_output(output_path):
    """Rows already in the output CSV, by video id."""
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return {}
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != OUTPUT_FIELDS:
            sys.exit(f"Error: {output_path} exists but its columns do not match "
                     f"{OUTPUT_FIELDS}. Move or delete it.")
        return {row["video_id"]: row for row in reader}


def write_csv(path, fieldnames, rows):
    """Write a CSV file atomically (a crash never leaves a truncated file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


# ----------------------------------------------------------------------------
# Work folder: request files, batch tracking, downloaded results
# ----------------------------------------------------------------------------

class Work:
    def __init__(self, path):
        self.path = path
        self.requests = path / "requests"
        self.responses = path / "responses"
        self.tracking = path / "batches.csv"

    def read_batches(self):
        if not self.tracking.is_file():
            return []
        with open(self.tracking, newline="") as f:
            return list(csv.DictReader(f))

    def write_batches(self, batches):
        write_csv(self.tracking, TRACKING_FIELDS, batches)

    def unsubmitted(self):
        """Request files written by `prepare` and not yet submitted."""
        if not self.requests.is_dir():
            return []
        submitted = {b["request_file"] for b in self.read_batches()}
        return sorted(p for p in self.requests.glob("batch_*.jsonl") if p.name not in submitted)

    def next_index(self):
        names = [b["request_file"] for b in self.read_batches()]
        if self.requests.is_dir():
            names += [p.name for p in self.requests.glob("batch_*.jsonl")]
        indices = [int(n.split("_")[1].split(".")[0]) for n in names]
        return max(indices, default=0) + 1


def make_client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: set the OPENAI_API_KEY environment variable to your OpenAI API key.")
    from openai import OpenAI

    return OpenAI()


def refresh(client, batches):
    """Update the status of the batches whose results are not collected yet."""
    for b in batches:
        if b["collected"] == "yes":
            continue
        batch = client.batches.retrieve(b["batch_id"])
        b["status"] = batch.status
        counts = batch.request_counts
        b["completed"] = counts.completed if counts else ""
        b["failed"] = counts.failed if counts else ""
        b["output_file_id"] = batch.output_file_id or ""
        b["error_file_id"] = batch.error_file_id or ""
    return batches


def video_id_of(custom_id):
    # The files of the paper used "video::<video_id>::<row>"; this script uses "<video_id>".
    return custom_id.split("::")[1] if custom_id.startswith("video::") else custom_id


def read_answers(folders, model):
    """Answers found in Batch API output files (*_output.jsonl), by video id.

    Failed requests are left out. If a video was answered more than once, the
    earliest answer is kept.
    """
    answers, created, other_models = {}, {}, 0
    for folder in folders:
        for path in sorted(Path(folder).glob("*_output.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    response = record.get("response") or {}
                    body = response.get("body") or {}
                    if record.get("error") or response.get("status_code") != 200 or not body.get("choices"):
                        continue
                    if not body.get("model", "").startswith(model):
                        other_models += 1
                        continue
                    message = body["choices"][0]["message"]
                    answer = message.get("content")
                    if answer is None:
                        answer = message.get("refusal") or ""
                    video_id = video_id_of(record["custom_id"])
                    when = body.get("created", 0)
                    if video_id not in answers or when < created[video_id]:
                        answers[video_id], created[video_id] = answer, when
    if other_models:
        print(f"Warning: {other_models} answers from another model than {model} were ignored.")
    return answers


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def cmd_prepare(args, model_name, output_path, work):
    if work.unsubmitted():
        sys.exit(f"Error: request files are waiting in {work.requests}. "
                 "Run `submit` first (or delete that folder to prepare them again).")
    if any(b["collected"] != "yes" for b in work.read_batches()):
        sys.exit("Error: some batches are not collected yet. Run `status`, then `collect` "
                 "once they are finished.")

    dataset = open_dataset(args)
    rows = dataset.rows[: args.limit] if args.limit is not None else dataset.rows
    # Videos in the CSV, plus answers downloaded but not yet written to it.
    done = set(read_output(output_path)) | set(read_answers([work.responses], args.model))
    todo = [row for row in rows if row["video_id"] not in done]
    if done:
        print(f"Resuming: {len(rows) - len(todo)} of {len(rows)} videos already in {output_path}")
    if not todo:
        print("Nothing to do.")
        return

    work.requests.mkdir(parents=True, exist_ok=True)
    for stale in work.requests.glob("*.part"):  # left by an interrupted run
        stale.unlink()
    index = work.next_index()
    for start in tqdm(range(0, len(todo), args.batch_size), desc="Request files", unit="file"):
        path = work.requests / f"batch_{index:03d}.jsonl"
        part = path.with_name(path.name + ".part")
        with open(part, "w", encoding="utf-8") as f:
            for row in tqdm(todo[start:start + args.batch_size], desc=path.name, unit="video", leave=False):
                video_path = dataset.video_path(row)
                try:
                    frames = video_frames(video_path)
                finally:
                    dataset.release(video_path)
                request = build_request(row["video_id"], row["question"].strip(), frames, args.model)
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
        part.replace(path)
        index += 1

    print(f"{len(todo)} requests written to {work.requests}. Next step: `submit`.")


def cmd_submit(args, model_name, output_path, work):
    files = work.unsubmitted()
    if not files:
        print("No request files to submit. Run `prepare` first.")
        return
    for path in files:
        if path.stat().st_size > MAX_INPUT_FILE_BYTES:
            sys.exit(f"Error: {path} is larger than the Batch API limit (200 MB). "
                     "Delete the request files and run `prepare` with a smaller --batch-size.")

    client = make_client()
    batches = work.read_batches()
    for path in tqdm(files, desc="Submitting", unit="batch"):
        with open(path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=ENDPOINT,
            completion_window="24h",
            metadata={"source_file": path.name},
        )
        with open(path, encoding="utf-8") as f:
            num_requests = sum(1 for _ in f)
        batches.append({
            "request_file": path.name, "requests": num_requests, "input_file_id": uploaded.id,
            "batch_id": batch.id, "status": batch.status, "completed": "", "failed": "",
            "output_file_id": "", "error_file_id": "", "collected": "no",
        })
        work.write_batches(batches)  # saved after each batch, in case of interruption

    print(f"{len(files)} batches submitted. Follow them with `status`.")


def print_status(batches):
    pending = [b for b in batches if b["collected"] != "yes"]
    if not pending:
        print("No batches waiting to be collected.")
        return
    print(f"{'request file':<18} {'batch id':<40} {'status':<12} {'done':>5} {'failed':>6} {'total':>5}")
    for b in pending:
        print(f"{b['request_file']:<18} {b['batch_id']:<40} {b['status']:<12} "
              f"{b['completed']:>5} {b['failed']:>6} {b['requests']:>5}")
    finished = sum(b["status"] in FINAL_STATUSES for b in pending)
    print(f"{finished} of {len(pending)} batches finished.")


def cmd_status(args, model_name, output_path, work):
    batches = work.read_batches()
    if not batches:
        print("No batches submitted yet.")
        return
    client = make_client()
    batches = refresh(client, batches)
    work.write_batches(batches)
    print_status(batches)


def cmd_collect(args, model_name, output_path, work):
    batches = work.read_batches()
    if any(b["collected"] != "yes" for b in batches):
        client = make_client()
        batches = refresh(client, batches)
        work.responses.mkdir(parents=True, exist_ok=True)
        collected = []
        for b in batches:
            if b["collected"] == "yes" or b["status"] not in FINAL_STATUSES:
                continue
            for kind in ("output", "error"):
                file_id = b[f"{kind}_file_id"]
                if file_id:
                    text = client.files.content(file_id).text
                    suffix = "output" if kind == "output" else "errors"
                    (work.responses / f"{b['batch_id']}_{suffix}.jsonl").write_text(text, encoding="utf-8")
            b["collected"] = "yes"
            (work.requests / b["request_file"]).unlink(missing_ok=True)
            work.write_batches(batches)
            collected.append(b)
        if collected:
            unsuccessful = sum(b["status"] != "completed" for b in collected)
            print(f"Downloaded the results of {len(collected)} finished batches"
                  + (f" ({unsuccessful} of them {', '.join(sorted({b['status'] for b in collected} - {'completed'}))})."
                     if unsuccessful else "."))
        if any(b["collected"] != "yes" for b in batches):
            print_status(batches)

    dataset = open_dataset(args)
    existing = read_output(output_path)
    answers = read_answers([work.responses, *args.import_dir], args.model)

    rows, new = [], 0
    for row in dataset.rows:
        video_id = row["video_id"]
        if video_id in existing:
            rows.append(existing.pop(video_id))
        elif video_id in answers:
            rows.append({
                "model": model_name,
                "video_id": video_id,
                "count": row["count"],
                "shape": row["shape"],
                "color": row["color"],
                "question": row["question"],
                "raw_answer": answers[video_id],
            })
            new += 1
    rows.extend(existing.values())  # videos not in this dataset revision: kept as they are
    write_csv(output_path, OUTPUT_FIELDS, rows)

    answered = {row["video_id"] for row in rows}
    missing = sum(row["video_id"] not in answered for row in dataset.rows)
    print(f"{new} new predictions. {output_path} now has "
          f"{len(dataset.rows) - missing} of {len(dataset.rows)} videos.")
    if missing and not any(b["collected"] != "yes" for b in batches):
        print(f"{missing} videos have no answer yet: run `prepare` and `submit` again "
              "(with --limit, only the first N videos are prepared).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=MODEL,
                        help="OpenAI model (default: the GPT-5.4 snapshot evaluated in the paper).")
    common.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: results/<model>_countvideobench.csv).")
    common.add_argument("--work-dir", type=Path, default=None,
                        help="Folder for the request files, batch tracking and downloaded results "
                             "(default: work/<model>/).")

    data = argparse.ArgumentParser(add_help=False)
    # Default: videos are downloaded on demand into the Hugging Face cache and kept.
    source = data.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path,
                        help="Use a local copy of CountVideoBench (folder containing 'single_class/') "
                             "instead of the Hugging Face cache.")
    source.add_argument("--streaming", action="store_true",
                        help="Delete each video after use instead of keeping it in the "
                             "Hugging Face cache (uses about 50 MB of disk).")
    data.add_argument("--revision", default=DATASET_REVISION,
                      help="Dataset revision on the Hub (default: the revision evaluated in the paper; "
                           "ignored with --data-dir).")

    parser = argparse.ArgumentParser(
        description="Evaluate GPT-5.4 on CountVideoBench with the OpenAI Batch API.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    prepare = commands.add_parser("prepare", parents=[common, data],
                                  help="Extract the frames and write the batch request files.")
    prepare.add_argument("--limit", type=int, default=None,
                         help="Only prepare the first N videos (for a quick test).")
    prepare.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help="Requests per batch (default: 25).")

    commands.add_parser("submit", parents=[common],
                        help="Upload the request files and create the batches.")
    commands.add_parser("status", parents=[common],
                        help="Show the status of the batches.")

    collect = commands.add_parser("collect", parents=[common, data],
                                  help="Download the results of finished batches and write the CSV.")
    collect.add_argument("--import", dest="import_dir", type=Path, action="append", default=[],
                         metavar="DIR",
                         help="Also read Batch API output files (*_output.jsonl) from DIR. "
                              "Can be given several times.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = MODEL_NAMES.get(args.model, args.model)
    output_path = args.output or ROOT / "results" / f"{model_name}_countvideobench.csv"
    work = Work(args.work_dir or ROOT / "work" / model_name)
    commands = {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}
    commands[args.command](args, model_name, output_path, work)


if __name__ == "__main__":
    main()
