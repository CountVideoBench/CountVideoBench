"""Evaluate Gemini 3.1 Pro on CountVideoBench with the Gemini Batch API.

The evaluation runs in four steps, one command each:

    uv run evaluate.py prepare          # extract the frames and write the request files
    uv run evaluate.py submit           # upload the files, one batch job per file
    uv run evaluate.py status --wait    # wait until all batch jobs are finished
    uv run evaluate.py collect          # download the answers and write the CSV

The API key is read from the GEMINI_API_KEY environment variable.

Everything is kept in a work folder (default: work/Gemini-3.1-Pro/), so the
steps can be run at different times. Requests that failed are prepared again
by the next `prepare`: repeat the four steps until every video has an answer.

Results obtained earlier (output files of the Batch API, e.g. those of the
paper) can be converted to the CSV without calling the API:

    uv run evaluate.py collect --import DIR [DIR ...]
"""

import argparse
import base64
import csv
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Disable the hf-xet chunk cache: it would keep a second copy of downloaded
# data on disk (and defeat the purpose of streaming mode).
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "Gemini-3.1-Pro"
# Model name sent to the API. Preview models have no dated snapshot: the
# version actually served is checked against the one of the paper at submission.
API_MODEL = "gemini-3.1-pro-preview"
PAPER_MODEL_VERSION = "3.1-pro-preview-01-2026"

NUM_FRAMES = 10
GENERATION_CONFIG = {
    "temperature": 0,
    "max_output_tokens": 1024,
    "thinking_config": {
        "thinking_level": "LOW",
    },
}
BATCH_SIZE = 25        # requests per batch job, as in the paper
POLL_INTERVAL = 30     # seconds between two checks with `status --wait`

DATASET_REPO = "countvideobench/CountVideoBench"
DATASET_SUBSET = "single_class"
DATASET_REVISION = "42e8fed00bca018a8fc10e249673e665aaa352b8"

OUTPUT_FIELDS = ["model", "video_id", "count", "shape", "color", "question", "raw_answer"]
BATCH_FIELDS = ["request_file", "uploaded_file", "job_name", "model", "model_version",
                "submitted_at", "state", "collected"]

SUCCEEDED_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


# ----------------------------------------------------------------------------
# Requests (same frames and request as in the paper)
# ----------------------------------------------------------------------------

def get_video_frames(video_path):
    """10 frames at indices int(i * N / 10), JPEG-encoded by OpenCV, base64."""
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)]

    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if not success:
            raise RuntimeError(f"Could not read frame {idx} of {video_path}")
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"Could not encode frame {idx} of {video_path}")
        frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return frames


def build_request(video_id, question, frames):
    """One line of a Batch API input file (GenerateContentRequest)."""
    return {
        "key": video_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": question},
                        *[
                            {"inline_data": {"mime_type": "image/jpeg", "data": frame}}
                            for frame in frames
                        ],
                    ],
                }
            ],
            "generation_config": GENERATION_CONFIG,
        },
    }


# ----------------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------------

def key_to_video_id(key):
    """`vid_000001`, or `video::vid_000001::<n>` (format used for the paper)."""
    match = re.fullmatch(r"video::(.+)::\d+", key)
    return match.group(1) if match else key


def response_text(response):
    """Concatenated text of the first candidate, thoughts excluded.

    Same rule as `GenerateContentResponse.text` of the google-genai SDK;
    empty string if the response has no text.
    """
    candidates = response.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    return "".join(
        part["text"] for part in parts
        if isinstance(part.get("text"), str) and part.get("thought") is not True
    )


def finish_reason(response):
    candidates = response.get("candidates") or []
    if not candidates:
        return None
    return candidates[0].get("finishReason") or candidates[0].get("finish_reason")


def read_responses(responses_dir):
    """Answers found in the Batch API output files.

    Returns (answers, failed, stats): answers maps video_id to
    (raw_answer, finish_reason, model_version); failed is the set of video ids
    whose request failed and that have no answer.
    """
    answers, failed = {}, set()
    duplicates = 0
    for path in sorted(responses_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                video_id = key_to_video_id(result.get("key", ""))
                response = result.get("response")
                if result.get("error") or not response:
                    failed.add(video_id)
                    continue
                if video_id in answers:
                    duplicates += 1
                    continue
                answers[video_id] = (
                    response_text(response),
                    finish_reason(response),
                    response.get("modelVersion") or response.get("model_version"),
                )
    return answers, failed - set(answers), duplicates


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
    return HubDataset(args.revision, keep=not getattr(args, "streaming", False))


# ----------------------------------------------------------------------------
# Work folder
# ----------------------------------------------------------------------------

class WorkDir:
    """work/<model>/: requests/ (to submit), batches.csv (batch jobs), responses/."""

    def __init__(self, path):
        self.path = Path(path)
        self.requests = self.path / "requests"
        self.responses = self.path / "responses"
        self.batches_csv = self.path / "batches.csv"
        for folder in (self.requests, self.responses):
            folder.mkdir(parents=True, exist_ok=True)

    def read_batches(self):
        if not self.batches_csv.is_file():
            return []
        with open(self.batches_csv, newline="") as f:
            return list(csv.DictReader(f))

    def write_batches(self, batches):
        tmp = self.batches_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BATCH_FIELDS)
            writer.writeheader()
            writer.writerows(batches)
        tmp.replace(self.batches_csv)

    def unsubmitted(self):
        submitted = {b["request_file"] for b in self.read_batches()}
        return [p for p in sorted(self.requests.glob("batch_*.jsonl")) if p.name not in submitted]

    def next_batch_number(self):
        names = [p.name for p in self.requests.glob("batch_*.jsonl")]
        names += [p.name for p in self.responses.glob("batch_*.jsonl")]
        names += [b["request_file"] for b in self.read_batches()]
        numbers = [int(m.group(1)) for n in names if (m := re.fullmatch(r"batch_(\d+)\.jsonl", n))]
        return max(numbers, default=0) + 1


def read_output(output_path):
    """Rows of an existing output CSV, by video id."""
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return {}
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != OUTPUT_FIELDS:
            sys.exit(f"Error: {output_path} exists but its columns do not match "
                     f"{OUTPUT_FIELDS}. Move or delete it.")
        return {row["video_id"]: row for row in reader}


def make_client():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Error: set the GEMINI_API_KEY environment variable to your Gemini API key.")
    from google import genai

    return genai.Client()


def state_name(job):
    return getattr(job.state, "name", None) or str(job.state)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def cmd_prepare(args):
    work = WorkDir(args.work_dir)
    if work.unsubmitted():
        sys.exit(f"Error: {len(work.unsubmitted())} request file(s) in {work.requests} are not "
                 f"submitted yet. Run `submit` first.")
    pending = [b for b in work.read_batches() if b["collected"] != "yes"]
    if pending:
        sys.exit(f"Error: {len(pending)} batch job(s) are not collected yet. "
                 f"Run `collect` first (`status` shows their state).")
    for stale in work.requests.glob("*.part"):
        stale.unlink()

    dataset = open_dataset(args)
    rows = dataset.rows[: args.limit] if args.limit is not None else dataset.rows

    answers, _, _ = read_responses(work.responses)
    done = set(answers) | set(read_output(args.output))
    todo = [row for row in rows if row["video_id"] not in done]
    if len(todo) < len(rows):
        print(f"{len(rows) - len(todo)} of {len(rows)} videos already have an answer.")
    if not todo:
        print("Nothing to prepare.")
        return

    number = work.next_batch_number()
    created = []
    out, out_path, in_batch = None, None, 0
    for row in tqdm(todo, desc="Preparing", unit="video"):
        if out is None:
            out_path = work.requests / f"batch_{number:03d}.jsonl"
            out = open(out_path.with_suffix(".jsonl.part"), "w", encoding="utf-8")
        video_path = dataset.video_path(row)
        try:
            frames = get_video_frames(video_path)
        finally:
            dataset.release(video_path)
        line = build_request(row["video_id"], row["question"].strip(), frames)
        out.write(json.dumps(line, ensure_ascii=False) + "\n")
        in_batch += 1
        if in_batch >= args.batch_size:
            out.close()
            out_path.with_suffix(".jsonl.part").replace(out_path)
            created.append(out_path)
            out, in_batch, number = None, 0, number + 1
    if out is not None:
        out.close()
        out_path.with_suffix(".jsonl.part").replace(out_path)
        created.append(out_path)

    print(f"{len(todo)} requests written to {len(created)} file(s) in {work.requests}. "
          f"Next step: `submit`.")


def cmd_submit(args):
    work = WorkDir(args.work_dir)
    files = work.unsubmitted()
    if not files:
        print("Nothing to submit. Run `prepare` first.")
        return

    client = make_client()
    from google.genai import types

    version = client.models.get(model=API_MODEL).version
    print(f"Model: {API_MODEL}, version {version}")
    if version != PAPER_MODEL_VERSION:
        print(f"Warning: the version served differs from the one used in the paper "
              f"({PAPER_MODEL_VERSION}); answers may differ.")

    batches = work.read_batches()
    for path in tqdm(files, desc="Submitting", unit="batch"):
        uploaded = client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(display_name=path.stem, mime_type="jsonl"),
        )
        job = client.batches.create(
            model=API_MODEL,
            src=uploaded.name,
            config={"display_name": path.stem},
        )
        batches.append({
            "request_file": path.name,
            "uploaded_file": uploaded.name,
            "job_name": job.name,
            "model": API_MODEL,
            "model_version": version,
            "submitted_at": now(),
            "state": state_name(job),
            "collected": "",
        })
        work.write_batches(batches)  # saved after each job, in case of interruption

    print(f"{len(files)} batch job(s) created. Next step: `status --wait`, then `collect`.")


def cmd_status(args):
    work = WorkDir(args.work_dir)
    batches = work.read_batches()
    pending = [b for b in batches if b["collected"] != "yes"]
    if not pending:
        print("No batch job to follow.")
        return

    client = make_client()
    while True:
        for batch in pending:
            batch["state"] = state_name(client.batches.get(name=batch["job_name"]))
        work.write_batches(batches)

        counts = {}
        for batch in pending:
            counts[batch["state"]] = counts.get(batch["state"], 0) + 1
        running = sum(n for s, n in counts.items() if s not in SUCCEEDED_STATES | FAILED_STATES)
        summary = ", ".join(f"{s.removeprefix('JOB_STATE_').lower()}: {n}"
                            for s, n in sorted(counts.items()))
        print(f"[{time.strftime('%H:%M:%S')}] {len(pending)} batch job(s) - {summary}")
        for batch in pending:
            if batch["state"] in FAILED_STATES:
                print(f"  {batch['request_file']}: {batch['state']}")

        if not args.wait or running == 0:
            break
        time.sleep(POLL_INTERVAL)

    if running == 0:
        print("All batch jobs are finished. Next step: `collect`.")


def cmd_collect(args):
    work = WorkDir(args.work_dir)

    for folder in args.import_dirs or []:
        files = sorted(Path(folder).glob("*.jsonl"))
        if not files:
            sys.exit(f"Error: no .jsonl file in {folder}")
        for path in files:
            shutil.copyfile(path, work.responses / f"{Path(folder).resolve().name}__{path.name}")
        print(f"Imported {len(files)} file(s) from {folder}")

    batches = work.read_batches()
    pending = [b for b in batches if b["collected"] != "yes"]
    if pending:
        client = make_client()
        for batch in pending:
            job = client.batches.get(name=batch["job_name"])
            batch["state"] = state_name(job)
            request_path = work.requests / batch["request_file"]
            if batch["state"] in SUCCEEDED_STATES:
                if not (job.dest and job.dest.file_name):
                    print(f"  {batch['request_file']}: {batch['state']} but no result file; "
                          f"try again later.")
                    continue
                data = client.files.download(file=job.dest.file_name)
                (work.responses / batch["request_file"]).write_bytes(data)
            elif batch["state"] in FAILED_STATES:
                print(f"  {batch['request_file']}: {batch['state']}; "
                      f"its videos will be prepared again.")
            else:
                print(f"  {batch['request_file']}: {batch['state']}, not finished yet.")
                continue
            batch["collected"] = "yes"
            request_path.unlink(missing_ok=True)
            work.write_batches(batches)
        still = sum(b["collected"] != "yes" for b in batches)
        if still:
            print(f"{still} batch job(s) not finished yet: run `collect` again later.")

    answers, failed, duplicates = read_responses(work.responses)

    # Keep rows of an existing CSV that are not in the work folder.
    dataset = open_dataset(args)
    existing = read_output(args.output)
    rows_out = []
    for row in dataset.rows:
        video_id = row["video_id"]
        if video_id in answers:
            rows_out.append({
                "model": MODEL_NAME,
                "video_id": video_id,
                "count": row["count"],
                "shape": row["shape"],
                "color": row["color"],
                "question": row["question"],
                "raw_answer": answers[video_id][0],
            })
        elif video_id in existing:
            rows_out.append(existing[video_id])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    tmp.replace(args.output)

    unknown = set(answers) - {row["video_id"] for row in dataset.rows}
    reasons, versions = {}, {}
    for _, reason, version in answers.values():
        reasons[reason] = reasons.get(reason, 0) + 1
        versions[version] = versions.get(version, 0) + 1
    empty = sum(1 for answer, _, _ in answers.values() if not answer.strip())

    print(f"{len(rows_out)} of {len(dataset.rows)} videos written to {args.output}")
    print(f"  finish reasons: {reasons}")
    print(f"  model versions: {versions}")
    if empty:
        print(f"  {empty} answer(s) without text (e.g. the token limit was reached while thinking)")
    if duplicates:
        print(f"  {duplicates} duplicate answer(s) ignored (the first one is kept)")
    if unknown:
        print(f"  Warning: {len(unknown)} answer(s) for videos not in the dataset, ignored")
    missing = len(dataset.rows) - len(rows_out)
    if missing:
        failed_note = f", including {len(failed)} failed request(s)" if failed else ""
        print(f"  {missing} video(s) without an answer{failed_note}: "
              f"run `prepare`, `submit`, `status --wait` and `collect` again.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work-dir", type=Path, default=ROOT / "work" / MODEL_NAME,
                        help="Folder for the request files, the batch jobs and the API responses.")

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data-dir", type=Path,
                      help="Use a local copy of CountVideoBench (folder containing 'single_class/') "
                           "instead of the Hugging Face cache.")
    data.add_argument("--revision", default=DATASET_REVISION,
                      help="Dataset revision on the Hub (ignored with --data-dir).")
    data.add_argument("--output", type=Path,
                      default=ROOT / "results" / f"{MODEL_NAME}_countvideobench.csv",
                      help="Output CSV; videos it already contains are not prepared again.")

    parser = argparse.ArgumentParser(
        description="Evaluate Gemini 3.1 Pro on CountVideoBench with the Gemini Batch API.",
        epilog="Run `uv run evaluate.py <command> --help` for the options of each command.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    fmt = argparse.ArgumentDefaultsHelpFormatter

    p = commands.add_parser("prepare", parents=[common, data], formatter_class=fmt,
                            help="Write the request files for the videos without an answer.")
    p.add_argument("--streaming", action="store_true",
                   help="Delete each video after use instead of keeping it in the "
                        "Hugging Face cache (cannot be combined with --data-dir).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only consider the first N videos of the dataset (for a quick test).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Number of requests per batch job.")
    p.set_defaults(func=cmd_prepare)

    p = commands.add_parser("submit", parents=[common], formatter_class=fmt,
                            help="Upload the request files and create the batch jobs.")
    p.set_defaults(func=cmd_submit)

    p = commands.add_parser("status", parents=[common], formatter_class=fmt,
                            help="Show the state of the batch jobs not collected yet.")
    p.add_argument("--wait", action="store_true",
                   help=f"Check every {POLL_INTERVAL} s until all batch jobs are finished.")
    p.set_defaults(func=cmd_status)

    p = commands.add_parser("collect", parents=[common, data], formatter_class=fmt,
                            help="Download the results of finished batch jobs and write the CSV.")
    p.add_argument("--import", dest="import_dirs", type=Path, nargs="+", metavar="DIR",
                   help="Also read the Batch API output files (*.jsonl) of these folders, "
                        "e.g. results obtained earlier.")
    p.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    if getattr(args, "streaming", False) and args.data_dir is not None:
        parser.error("--streaming cannot be combined with --data-dir")
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
