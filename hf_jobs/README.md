### Hugging Face Jobs pipelines (no code changes)

This folder adds self-contained bootstrapping scripts to retrain the model on Hugging Face Jobs without modifying any existing repo code.

- `bootstrap_and_train_rl.sh` - prepares the container and runs RL training via `scripts.chat_rl`.
- `bootstrap_and_train_base.sh` - prepares the container and runs base training via `scripts.base_train`.
- `run_full_pipeline.sh` - end-to-end launcher that executes base -> mid -> SFT -> RL (plus eval/report) in one job.
- `.env.example` - sample environment variables/secrets you can re-use with Jobs.
- `run_rl_job_example.sh` - optional local helper that shells out to `hf jobs run`.

All of the scripts:
- install the system build toolchain and Rust (needed for the `rustbpe` extension via `maturin`)
- install project dependencies (`uv sync --extra gpu` or `pip install -e .`)
- launch the appropriate training entrypoint with optional multi-GPU via `torchrun`
- optionally clone the repo inside the job when `REPO_URL`/`GIT_REF` are set

### Prerequisites

1. Hugging Face account with Jobs access.
2. CLI installed and authenticated:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login
   ```

### Hardware flavors and base image

Pick a GPU flavor that matches your budget (`t4-small`, `a10g-small`, `a10g-large`, `a100-large`, etc.). A good base container image is:
```
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

### Run RL training (jobs)

If the Jobs CLI uploads your working tree (default behavior):
```bash
hf jobs run \
  --name nanochat-rl \
  --flavor a10g-small \
  -e RUN=rl-gsm8k-a10g \
  -e RL_ARGS="--examples_per_step=16 --num_samples=16 --eval_every=120" \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash hf_jobs/bootstrap_and_train_rl.sh
```

If the CLI does not upload the repo, clone it inside the job:
```bash
hf jobs run \
  --name nanochat-rl \
  --flavor a10g-small \
  -e REPO_URL=https://github.com/you/nanochatAquaRat.git \
  -e GIT_REF=main \
  -e RUN=rl-gsm8k-a10g \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash hf_jobs/bootstrap_and_train_rl.sh
```

Multi-GPU example (8x A100):
```bash
hf jobs run \
  --name nanochat-rl-8g \
  --flavor a100-large \
  -e NPROC_PER_NODE=8 \
  -e RUN=rl-8g \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash hf_jobs/bootstrap_and_train_rl.sh
```

`RL_ARGS` is forwarded after `--` to `scripts.chat_rl` (via `nanochat/configurator.py`), so any CLI flag supported there can be injected.

### Run base training (jobs)

```bash
hf jobs run \
  --name nanochat-base \
  --flavor a10g-large \
  -e RUN=base-a10g \
  -e BASE_ARGS="--depth=12 --num_iterations=2000 --eval_every=250" \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash hf_jobs/bootstrap_and_train_base.sh
```

Multi-GPU base example:
```bash
hf jobs run \
  --name nanochat-base-8g \
  --flavor a100-large \
  -e NPROC_PER_NODE=8 \
  -e RUN=base-8g \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash hf_jobs/bootstrap_and_train_base.sh
```

### Common environment variables

- `WANDB_API_KEY` - enables non-dummy W&B logging (use `--secrets` in production).
- `RUN` - W&B run name passed through to the scripts.
- `NPROC_PER_NODE` - GPU count for `torchrun`.
- `REPO_URL`, `GIT_REF` - optional git clone fallback inside the job.

### Full training pipeline (base + mid + SFT + RL)

`run_full_pipeline.sh` executes the entire stack sequentially. It mirrors `run_aquarat_small.sh` but exposes knobs tailored for Hugging Face Jobs so you can right-size each stage.

```bash
hf jobs run \
  --name aquarat-full-train \
  --flavor a100-large \
  --timeout 12h \
  --env HF_TRAIN_GPUS=4 \
  --env HF_RL_GPUS=1 \
  --env HF_BASE_DEPTH=20 \
  --env HF_BASE_ITERATIONS=600 \
  --env HF_MID_ITERATIONS=400 \
  --env HF_SFT_TRAIN_EXAMPLES=90000 \
  --env HF_RL_EXAMPLES_PER_STEP=32 \
  --env HF_RL_NUM_SAMPLES=16 \
  --env NANOCHAT_CACHE_ROOT=/data/nanochat \
  --env WANDB_PROJECT=nanochat-aquarat \
  --secrets WANDB_API_KEY \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash hf_jobs/run_full_pipeline.sh
```

Configurable knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TRAIN_GPUS` | 8 | GPUs for base/mid/SFT `torchrun` calls. |
| `HF_RL_GPUS` | 1 | GPUs for RL. |
| `HF_BASE_DEPTH` | 20 | Transformer depth (~561M params). |
| `HF_BASE_ITERATIONS` | 400 | Base-training steps. |
| `HF_MID_ITERATIONS` | 400 | Mid-training steps. |
| `HF_SFT_TRAIN_EXAMPLES` | 80000 | AQuA samples used in SFT. |
| `HF_SFT_VAL_EXAMPLES` | 254 | Validation samples during SFT. |
| `HF_RL_EXAMPLES_PER_STEP` | 32 | `examples_per_step` for RL. |
| `HF_RL_NUM_SAMPLES` | 16 | Samples per prompt during RL rollouts. |
| `HF_RL_MAX_NEW_TOKENS` | 128 | Generation cap during RL. |
| `HF_RL_TEMPERATURE` | 0.7 | Sampling temp during RL. |
| `HF_RL_SAVE_EVERY` | 60 | RL checkpoint cadence. |
| `HF_RL_EVAL_EVERY` | 60 | RL eval cadence. |
| `HF_RL_EVAL_EXAMPLES` | 512 | RL eval sample count. |
| `HF_DATASET_SHARDS` | 24 | GSM8K shards fetched via `nanochat.dataset`. |
| `HF_INSTALL_BASE_PACKAGES` | 1 | Set to `0` to skip `apt-get install`. |
| `NANOCHAT_CACHE_ROOT` | `$HOME/.cache` | Root for caches (`NANOCHAT_BASE_DIR`, `HF_HOME`, etc.). |
| `WANDB_RUN` | auto timestamp | Override the shared W&B run name. |

What the script does:
1. Installs uv, Rust, and builds the tokenizer extension (`maturin develop --release`).
2. Downloads GSM8K shards, cached AQuA-RAT splits, identity prompts, and mech-interp assets.
3. Runs `torchrun` for base, mid, SFT, and RL with the requested GPU counts.
4. Evaluates RL checkpoints with `scripts.chat_eval -i rl -a AQUA` and summarizes reports via `nanochat.report summarize`.

Add artifact uploads (e.g., `python scripts/upload_to_gcs.py ...`) after the script finishes if you need off-box persistence.

### Notes

- Installing Rust adds ~1 minute to first-time jobs but is required for `rustbpe`.
- Models/checkpoints live under the repo root (or the cache root you set). Plan to copy them to external storage or the Hub before the job terminates.
