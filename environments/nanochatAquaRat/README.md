# nanochatAquaRat Environment

Prime Intellect verifier environment that mirrors the nanochat AQuA-RAT reinforcement-learning task: single-turn algebra questions with multiple-choice answers (letters A–E) scored by categorical accuracy.

## Overview
- **Hub ID**: `harleycooper/nanochatAquaRat`
- **Task type**: Single-turn chat
- **Parser**: `verifiers.Parser` with a custom A–E letter extractor
- **Rubric**: Exact-match reward (weight 1.0) plus valid-letter format bonus (weight 0.1)

## Dataset
- **Source**: [deepmind/aqua_rat](https://huggingface.co/datasets/deepmind/aqua_rat)
- **Content**: ~97k algebra word problems, five answer options, human rationale, gold letter.
- **Default splits**: `train` for rollouts, `validation` for evaluation (configurable).
- **Metadata**: question stem, options, and optional rationale retained per example.

By default the loader streams from Hugging Face. For offline use, pass `data_dir=/path/to/aqua` where that directory contains `train.jsonl`, `validation.jsonl`, and `test.jsonl` generated via `scripts/prepare_aqua.py` in the base repository.

## Quickstart
Evaluate a model on the validation set:

```bash
uv run vf-eval harleycooper/nanochatAquaRat -m gpt-4o-mini -n 25
```

Kick off GRPO training (LoRA-friendly defaults shown):

```bash
uv run vf-rl @ configs/rl/nanochat.toml
```

Example `configs/rl/nanochat.toml` excerpt:

```toml
model = "Qwen/Qwen2.5-7B-Instruct"

[env]
id = "harleycooper/nanochatAquaRat"

[env.args]
num_train_examples = 2000
num_eval_examples = 254
seed = 42

[trainer.args]
learning_rate = 2e-5
rollouts_per_example = 8
max_steps = 400
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `system_prompt` | str | Algebra tutoring instruction | Prepended system message |
| `train_split` | str | `"train"` | Dataset split used for rollouts |
| `eval_split` | str\|null | `"validation"` | Split for evaluation (`null` reuses train) |
| `num_train_examples` | int | `-1` | Cap on rollout examples after shuffling |
| `num_eval_examples` | int | `-1` | Cap on evaluation examples |
| `seed` | int\|null | `42` | Deterministic shuffle seed for the train split |
| `include_rationale_metadata` | bool | `true` | Include human rationale text in metadata |
| `data_dir` | str\|null | `null` | Local directory containing JSON/JSONL splits |
| `cache_dir` | str\|null | `null` | Hugging Face cache override |

Pass overrides with `vf-eval ... --env-args '{"num_train_examples": 5000}'`.

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted reward (exact-match + format bonus) |
| `exact_match_reward` | Raw exact-match signal prior to weighting |
| `format_reward` | Bonus for emitting a valid letter token |

`reward` aligns with the `rl/acc` tracking used in the nanochat RL scripts, so you can compare outcomes across training setups.*** End Patch
