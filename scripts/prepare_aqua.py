#!/usr/bin/env python3
"""
Prepare the AQuA-RAT dataset for nanochat training.

We download the dataset from Hugging Face, normalize each example into a
conversation-like JSON object, and write JSONL splits (train/validation/test)
that can be consumed by the training scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset

from tasks.aqua import _parse_option, _render_user_prompt


def format_example(row: Dict[str, str]) -> Dict[str, object]:
    """Convert a raw dataset row into the conversation schema."""
    question = row["question"].strip()
    options = row["options"]
    rationale = row["rationale"].strip()
    correct = row["correct"].strip().upper()

    parsed = [_parse_option(opt) for opt in options]
    letters = [item["letter"] for item in parsed]
    assert correct in letters, f"Correct answer {correct} missing from options {letters}"

    assistant_content = [
        {"type": "text", "text": rationale},
        {"type": "text", "text": f"Answer: {correct}"},
    ]
    conversation = {
        "messages": [
            {"role": "user", "content": _render_user_prompt(question, options)},
            {"role": "assistant", "content": assistant_content},
        ],
        "letters": letters,
        "answer_letter": correct,
    }
    return conversation


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the AQuA-RAT dataset.")
    parser.add_argument("--output_dir", type=str, default=None, help="Destination directory for the JSONL splits.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on number of train examples.")
    parser.add_argument("--max-validation", type=int, default=None, help="Optional cap on number of validation examples.")
    parser.add_argument("--max-test", type=int, default=None, help="Optional cap on number of test examples.")
    args = parser.parse_args()

    ds = load_dataset("deepmind/aqua_rat")

    output_dir = Path(args.output_dir or Path(os.getenv("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat")) / "aqua")
    output_dir.mkdir(parents=True, exist_ok=True)

    def iter_split(split_name: str, limit: int | None) -> Iterable[Dict[str, object]]:
        split_ds = ds[split_name]
        if limit is not None:
            split_ds = split_ds.select(range(min(limit, len(split_ds))))
        for row in split_ds:
            yield format_example(row)

    write_jsonl(output_dir / "train.jsonl", iter_split("train", args.max_train))
    write_jsonl(output_dir / "validation.jsonl", iter_split("validation", args.max_validation))
    write_jsonl(output_dir / "test.jsonl", iter_split("test", args.max_test))

    print(f"Wrote AQuA-RAT splits to {output_dir}")


if __name__ == "__main__":
    main()
