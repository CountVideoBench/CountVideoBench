"""Extract the predicted count from the answers of a model on CountVideoBench.

The models answer in free text ("There are ten red cubes in this video."). A
local LLM (Qwen2.5-7B-Instruct) reads each answer and returns the count it
states, as in the paper. For the paper, the extracted counts were then checked
by hand (see README.md).

Examples:
    uv run extract.py ../qwen2_5_vl/results/Qwen2.5-VL-7B_countvideobench.csv
    uv run extract.py ../*/results/*_countvideobench.csv

Each input CSV gives one output CSV (by default results/<Model>_extracted.csv)
with the same columns plus `predicted_count` and `extractor_output`. If the
output file already exists, the videos it already contains are skipped, so an
interrupted run can be resumed by running the same command again.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# Hub id and revision of the extraction model used in the paper.
EXTRACTOR = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")

EXTRA_FIELDS = ["predicted_count", "extractor_output"]
REQUIRED_FIELDS = ["video_id", "question", "raw_answer"]

# Instructions given to the extraction model (system message), exactly as used
# for the paper. The extracted counts were then checked by hand (see README.md,
# "Paper results").
SYSTEM_PROMPT = """
You output exactly one integer between $ signs: $<integer>$ and nothing else.

Rules:
1) If "Answer Programme" contains an explicit count (digit or number word like one, two, three...), output that count as an integer.
2) If the count is unknown, not specified, unclear, estimated, or cannot be determined from the answer, output $NaN$.
3) Do NOT guess. Do NOT infer from context. Only use what is explicitly stated.
4) If the answer contains multiple possible counts (e.g., "six or seven", "at least six"), output $NaN$.
5) If the answer starts with "There are approximately" followed by a specific number, output only the number.

Input format:
User Question: <text>
Answer Programme: <text>
"""


# ----------------------------------------------------------------------------
# Extraction model
# ----------------------------------------------------------------------------

def load_model(model_path, revision, device_map):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        dtype="auto",
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    return model, tokenizer


def run_extractor(model, tokenizer, question, answer):
    """Raw output of the extraction model for one answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "User Question : " + question + "\nAnswer Programme : " + answer},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Greedy decoding.
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def parse_count(output):
    """Read the extraction model's output.

    Returns (count, status): status is "count" (an integer was found), "nan"
    (the model found no explicit count) or "check" (unexpected output, to be
    checked by hand). count is None unless status is "count".
    """
    if "$" not in output:
        return None, "check"
    value = output.split("$")[1].strip()
    if re.fullmatch(r"[0-9]+", value):
        return int(value), "count"
    if value == "NaN":
        return None, "nan"
    return None, "check"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the predicted count from the answers of a model on CountVideoBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="CSV file(s) written by a model's evaluate.py.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV, with a single input only "
                             "(default: results/<Model>_extracted.csv).")
    parser.add_argument("--model-path", default=None,
                        help="Local path or Hub id of the extraction model "
                             f"(default: {EXTRACTOR[0]}).")
    parser.add_argument("--model-revision", default=None,
                        help="Revision of the extraction model on the Hub "
                             "(default: the revision used in the paper, when --model-path is not set).")
    parser.add_argument("--device-map", default="auto",
                        help="Passed to from_pretrained; 'auto' uses all visible GPUs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows of each input (for a quick test).")
    args = parser.parse_args()
    if args.output is not None and len(args.inputs) > 1:
        parser.error("--output can only be used with a single input file.")
    return args


def default_output(input_path):
    name = input_path.stem
    if name.endswith("_countvideobench"):
        name = name[: -len("_countvideobench")]
    return ROOT / "results" / f"{name}_extracted.csv"


def read_input(path):
    if not path.is_file():
        sys.exit(f"Error: {path} not found.")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in REQUIRED_FIELDS if c not in fields]
        if missing:
            sys.exit(f"Error: {path} has no column {', '.join(missing)}. "
                     "Expected the CSV written by a model's evaluate.py.")
        if any(c in fields for c in EXTRA_FIELDS):
            sys.exit(f"Error: {path} already contains extracted counts.")
        return fields, list(reader)


def read_output(path, fields):
    """Rows already present in an existing output file."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            sys.exit(f"Error: {path} exists but its columns do not match {fields}. "
                     "Move or delete it.")
        return list(reader)


def summary(path, fields):
    rows = read_output(path, fields)
    statuses = [parse_count(row["extractor_output"])[1] for row in rows]
    to_check = [row["video_id"] for row, s in zip(rows, statuses) if s == "check"]
    print(f"{os.path.relpath(path)}: {len(rows)} answers, {statuses.count('count')} with a count, "
          f"{statuses.count('nan')} without (NaN), {len(to_check)} to check by hand")
    if to_check:
        print("  unexpected extractor output (predicted_count left empty): " + ", ".join(to_check))


def main():
    args = parse_args()

    jobs = []
    for input_path in args.inputs:
        fields, rows = read_input(input_path)
        if args.limit is not None:
            rows = rows[: args.limit]
        output_path = args.output or default_output(input_path)
        if any(job[2] == output_path for job in jobs):
            sys.exit(f"Error: several inputs would be written to {output_path}.")
        out_fields = fields + EXTRA_FIELDS
        done = {row["video_id"] for row in read_output(output_path, out_fields)}
        todo = [row for row in rows if row["video_id"] not in done]
        if done:
            print(f"Resuming: {len(rows) - len(todo)} of {len(rows)} rows of {input_path} "
                  f"already in {os.path.relpath(output_path)}")
        jobs.append((input_path, rows, output_path, out_fields, todo))

    if not any(job[4] for job in jobs):
        print("Nothing to do.")
    else:
        if args.model_path is None:
            model_path, model_revision = EXTRACTOR
        else:
            model_path, model_revision = args.model_path, None
        model_revision = args.model_revision or model_revision
        model, tokenizer = load_model(model_path, model_revision, args.device_map)

        for input_path, rows, output_path, out_fields, todo in jobs:
            if not todo:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not output_path.is_file() or output_path.stat().st_size == 0
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=out_fields)
                if write_header:
                    writer.writeheader()
                for row in tqdm(todo, desc=input_path.stem, unit="answer",
                                initial=len(rows) - len(todo), total=len(rows)):
                    output = run_extractor(model, tokenizer, row["question"], row["raw_answer"])
                    count, _ = parse_count(output)
                    writer.writerow({
                        **row,
                        "predicted_count": "" if count is None else count,
                        "extractor_output": output,
                    })
                    f.flush()

    for _, _, output_path, out_fields, _ in jobs:
        summary(output_path, out_fields)


if __name__ == "__main__":
    main()
