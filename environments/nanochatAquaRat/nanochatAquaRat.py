from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import verifiers as vf
from datasets import Dataset, load_dataset


DEFAULT_SYSTEM_PROMPT = (
    "You are an algebra tutor. Solve each problem, show your reasoning, and end with "
    "'Answer: <letter>' where <letter> is one of A, B, C, D, or E."
)

LETTER_PATTERN = re.compile(r"(?i)\banswer\s*[:\-]\s*([A-E])\b|(?<![A-Z])([A-E])(?![A-Z])")


def _parse_option(option: str) -> tuple[str, str]:
    if ")" in option:
        letter, text = option.split(")", 1)
        return letter.strip().upper(), text.strip()
    return option[:1].upper(), option[1:].strip()


def _render_prompt(question: str, options: list[str]) -> str:
    rendered = "Solve the multiple choice problem and explain your reasoning. "
    rendered += "Conclude with 'Answer: <letter>'.\n\n"
    rendered += f"Question: {question.strip()}\n"
    parsed = [_parse_option(opt) for opt in options]
    for letter, choice in parsed:
        rendered += f"- {choice}={letter}\n"
    rendered += "\nRespond only with the letter of the correct answer."
    return rendered


def _extract_letter(text: str | None) -> Optional[str]:
    if not text:
        return None
    match = LETTER_PATTERN.search(text)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip().upper() or None


def _load_local_split(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    return []


def _maybe_load_dataset(split: str, data_dir: Optional[str], cache_dir: Optional[str]) -> Dataset:
    local_rows: list[dict] = []
    if data_dir:
        base = Path(data_dir)
        local_rows.extend(_load_local_split(base / f"{split}.jsonl"))
        local_rows.extend(_load_local_split(base / f"{split}.json"))
    if local_rows:
        return Dataset.from_list(local_rows)
    return load_dataset("deepmind/aqua_rat", split=split, cache_dir=cache_dir)


def _row_to_prompt_payload(
    row: dict,
    system_prompt: Optional[str],
    include_rationale: bool,
) -> dict:
    if "messages" in row:
        # Prepared conversation format (e.g. from scripts/prepare_aqua.py)
        user_msg = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        correct = row.get("answer_letter") or row.get("correct") or ""
        rationale = ""
        assistant_msg = next((m for m in row["messages"] if m.get("role") == "assistant"), None)
        if assistant_msg and isinstance(assistant_msg.get("content"), list):
            rationale_parts = [part.get("text", "") for part in assistant_msg["content"] if isinstance(part, dict)]
            rationale = "\n".join(filter(None, rationale_parts))
    else:
        question = row["question"]
        options = row["options"]
        correct = row["correct"]
        rationale = row.get("rationale", "")
        user_msg = _render_prompt(question, options)

    prompt_messages = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": user_msg})

    metadata = {}
    if row.get("question") is not None:
        metadata["question"] = row["question"]
    if row.get("options") is not None:
        metadata["options"] = row["options"]
    if include_rationale and rationale:
        metadata["rationale"] = rationale

    return {
        "prompt": prompt_messages,
        "answer": str(correct).strip().upper(),
        "metadata": metadata,
    }


def _format_dataset(
    raw_dataset: Dataset,
    system_prompt: Optional[str],
    include_rationale: bool,
) -> Dataset:
    def formatter(batch: dict) -> dict:
        prompts, answers, metadata = [], [], []
        for idx in range(len(batch[next(iter(batch))])):
            row = {key: batch[key][idx] for key in batch}
            payload = _row_to_prompt_payload(row, system_prompt, include_rationale)
            prompts.append(payload["prompt"])
            answers.append(payload["answer"])
            metadata.append(payload["metadata"])
        return {"prompt": prompts, "answer": answers, "metadata": metadata}

    processed = raw_dataset.map(formatter, batched=True, remove_columns=raw_dataset.column_names)
    return processed


def _build_datasets(
    train_split: str,
    eval_split: Optional[str],
    system_prompt: Optional[str],
    num_train_examples: int,
    num_eval_examples: int,
    include_rationale: bool,
    data_dir: Optional[str],
    cache_dir: Optional[str],
    seed: Optional[int],
) -> tuple[Dataset, Dataset]:
    raw_train = _maybe_load_dataset(train_split, data_dir, cache_dir)
    if seed is not None:
        raw_train = raw_train.shuffle(seed=seed)
    if num_train_examples is not None and num_train_examples >= 0:
        cap = min(num_train_examples, len(raw_train))
        raw_train = raw_train.select(range(cap))
    formatted_train = _format_dataset(raw_train, system_prompt, include_rationale)

    if eval_split:
        raw_eval = _maybe_load_dataset(eval_split, data_dir, cache_dir)
        if num_eval_examples is not None and num_eval_examples >= 0:
            cap = min(num_eval_examples, len(raw_eval))
            raw_eval = raw_eval.select(range(cap))
        formatted_eval = _format_dataset(raw_eval, system_prompt, include_rationale)
    else:
        formatted_eval = formatted_train
    return formatted_train, formatted_eval


def load_environment(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    train_split: str = "train",
    eval_split: Optional[str] = "validation",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: Optional[int] = 42,
    include_rationale_metadata: bool = True,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> vf.Environment:
    """
    Create a verifiers environment exposing the nanochat AQuA-RAT reward function.

    Args:
        system_prompt: Optional system message prepended to each example.
        train_split: Split to use for training data (defaults to Hugging Face 'train').
        eval_split: Split to use for evaluation data. If None, reuses the train split.
        num_train_examples: Optional cap on train examples (-1 keeps full split).
        num_eval_examples: Optional cap on eval examples (-1 keeps full split).
        seed: Shuffle seed for deterministic sampling.
        include_rationale_metadata: Whether to attach human rationales in metadata.
        data_dir: Optional directory containing preprocessed JSON/JSONL splits.
        cache_dir: Optional Hugging Face cache directory.
    """
    train_ds, eval_ds = _build_datasets(
        train_split=train_split,
        eval_split=eval_split,
        system_prompt=system_prompt,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        include_rationale=include_rationale_metadata,
        data_dir=data_dir,
        cache_dir=cache_dir,
        seed=seed,
    )

    parser = vf.Parser(extract_fn=_extract_letter)

    def exact_match_reward(completion, answer, parser, **kwargs) -> float:
        predicted = parser.parse_answer(completion) or ""
        return 1.0 if predicted == answer else 0.0

    def format_reward(completion, parser, **kwargs) -> float:
        predicted = parser.parse_answer(completion)
        return 1.0 if predicted in {"A", "B", "C", "D", "E"} else 0.0

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(exact_match_reward)
    rubric.add_reward_func(format_reward, weight=0.1)

    vf_env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        parser=parser,
        rubric=rubric,
        message_type="chat",
        env_id="harleycooper/nanochatAquaRat",
    )
    return vf_env
