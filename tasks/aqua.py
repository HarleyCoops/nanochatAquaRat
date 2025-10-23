"""
AQuA-RAT (Algebra Question Answering with Rationales) task utilities.
Dataset reference: https://huggingface.co/datasets/deepmind/aqua_rat
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset

from tasks.common import Task, render_mc


def _default_cache_dir() -> Path:
    base = os.getenv("NANOCHAT_AQUA_DIR")
    if base:
        return Path(base)
    base = os.getenv("AQUA_DATA_DIR")
    if base:
        return Path(base)
    nanochat_base = os.getenv("NANOCHAT_BASE_DIR")
    if nanochat_base:
        return Path(nanochat_base) / "aqua"
    return Path.home() / ".cache" / "nanochat" / "aqua"


def _resolve_data_dir(data_dir: str | os.PathLike[str] | None) -> Path | None:
    if data_dir is not None:
        path = Path(data_dir)
    else:
        path = _default_cache_dir()
    return path if path.exists() else None


def _load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

LETTER_RE = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)


def _parse_option(option: str) -> Dict[str, str]:
    """
    The dataset stores options as strings like 'A)21.5'.
    Split into (letter, choice) parts for consistent formatting.
    """
    if ")" in option:
        letter, text = option.split(")", 1)
        return {"letter": letter.strip().upper(), "choice": text.strip()}
    # Fallback: assume well formatted
    return {"letter": option[:1].upper(), "choice": option[1:].strip()}


def _render_user_prompt(question: str, options: List[str]) -> str:
    parsed = [_parse_option(opt) for opt in options]
    letters = [item["letter"] for item in parsed]
    choices = [item["choice"] for item in parsed]
    instructions = (
        "Solve the multiple choice problem and explain your reasoning. "
        "Conclude with 'Answer: <letter>'.\n\n"
    )
    return instructions + render_mc(question, letters, choices)


def _extract_letter(text: str, default: str | None = None) -> str | None:
    """
    Attempt to extract the first answer letter (A-E) from arbitrary text.
    We look for 'Answer: X' first, otherwise fall back to standalone letter.
    """
    answer_match = re.search(r"answer\s*[:\-]\s*([A-E])", text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    match = LETTER_RE.search(text)
    if match:
        return match.group(1).upper()
    return default.upper() if isinstance(default, str) else default


class AQUA(Task):
    """
    Thin wrapper around the AQuA-RAT dataset that produces conversation-style
    examples compatible with the nanochat training & evaluation pipelines.
    """

    def __init__(self, split: str, shuffle: bool = True, seed: int = 42,
                 data_dir: str | os.PathLike[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "validation", "test"}, "split must be train|validation|test"
        base_dir = _resolve_data_dir(data_dir)
        if base_dir is not None:
            jsonl_path = base_dir / f"{split}.jsonl"
            if jsonl_path.exists():
                dataset = list(_load_jsonl(jsonl_path))
            else:
                dataset = load_dataset("deepmind/aqua_rat", split=split)
        else:
            dataset = load_dataset("deepmind/aqua_rat", split=split)
        if shuffle:
            if isinstance(dataset, list):
                from random import Random
                rng = Random(seed)
                rng.shuffle(dataset)
            else:
                dataset = dataset.shuffle(seed=seed)
        self.ds = dataset

    @property
    def eval_type(self) -> str:
        return "categorical"

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> Dict[str, object]:
        row = self.ds[index]
        if isinstance(row, dict) and "messages" in row:
            return row
        question: str = row["question"].strip()
        options: List[str] = row["options"]
        rationale: str = row["rationale"].strip()
        correct: str = row["correct"].strip().upper()

        parsed = [_parse_option(opt) for opt in options]
        letters = [item["letter"] for item in parsed]
        assert correct in letters, f"Correct letter {correct} not found in options {letters}"

        user_message = _render_user_prompt(question, options)
        assistant_content = [
            {"type": "text", "text": rationale},
            {"type": "text", "text": f"Answer: {correct}"},
        ]
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_content},
            ],
            "letters": letters,
            "answer_letter": correct,
        }
        return conversation

    def evaluate(self, conversation: Dict[str, object], assistant_response: str) -> int:
        predicted = _extract_letter(assistant_response)
        correct = conversation["answer_letter"]
        return int(predicted == correct)

    def reward(self, conversation: Dict[str, object], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
