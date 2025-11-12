#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[hf-jobs] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  log "Loading environment overrides from .env"
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CACHE_ROOT="${NANOCHAT_CACHE_ROOT:-$HOME/.cache}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$CACHE_ROOT/nanochat}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-$HF_HOME/metrics}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$NANOCHAT_BASE_DIR" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE" "$HF_HUB_CACHE"

: "${WANDB_PROJECT:=nanochat-aquarat}"
: "${WANDB_MODE:=online}"
: "${WANDB_RUN:=$(date -u +"aquarat-hf-%Y%m%d-%H%M%S")}"
export WANDB_PROJECT WANDB_MODE WANDB_RUN
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
fi

if [ "${HF_INSTALL_BASE_PACKAGES:-1}" != "0" ] && command -v apt-get >/dev/null 2>&1; then
  log "Installing base apt packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    git curl unzip ca-certificates pkg-config build-essential
  rm -rf /var/lib/apt/lists/*
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d ".venv" ]; then
  log "Creating Python virtual environment"
  uv venv
fi

log "Syncing Python dependencies (+gpu extras)"
uv sync --extra gpu
# shellcheck disable=SC1091
source .venv/bin/activate

if [ -n "${WANDB_API_KEY:-}" ]; then
  log "Authenticating with Weights & Biases"
  wandb login --relogin "$WANDB_API_KEY" || true
else
  log "WANDB_API_KEY not set; run will stay offline unless already logged in"
fi

log "Resetting nanochat report cache"
python -m nanochat.report reset

if [ ! -f "$HOME/.cargo/bin/cargo" ]; then
  log "Installing Rust toolchain"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

log "Building Rust tokenizer extension"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

log "Bootstrapping GSM8K dataset shards"
python -m nanochat.dataset -n "${HF_DATASET_SHARDS:-24}" &
DATA_PID=$!

log "Preparing cached AQuA-RAT splits"
AQUA_DIR="$NANOCHAT_BASE_DIR/aqua"
mkdir -p "$AQUA_DIR"
python -m scripts.prepare_aqua --output_dir "$AQUA_DIR"
export AQUA_DATA_DIR="$AQUA_DIR"

IDENTITY_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_PATH" ]; then
  log "Fetching identity conversations corpus"
  curl -L -o "$IDENTITY_PATH" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

if [ ! -f "$NANOCHAT_BASE_DIR/eval_bundle/core.yaml" ]; then
  log "Downloading eval bundle snapshot"
  pushd "$NANOCHAT_BASE_DIR" >/dev/null
  curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
  unzip -q eval_bundle.zip
  rm eval_bundle.zip
  popd >/dev/null
fi

if [ -n "${DATA_PID:-}" ]; then
  log "Waiting for background dataset download"
  wait "$DATA_PID" || true
fi

MECH_INTERP_DIR="$NANOCHAT_BASE_DIR/mechanistic_interpretability"
if [ ! -d "$MECH_INTERP_DIR" ]; then
  log "Cloning Google DeepMind mechanistic interpretability repo"
  git clone https://github.com/google-deepmind/mechanistic-interp.git "$MECH_INTERP_DIR"
else
  log "Updating mechanistic interpretability repo"
  git -C "$MECH_INTERP_DIR" pull --ff-only
fi
export MECH_INTERP_DIR

run_torch() {
  local gpus="$1"
  shift
  if [ "$gpus" -lt 1 ]; then
    gpus=1
  fi
  log "Launching torchrun ($gpus GPUs): $*"
  torchrun --standalone --nproc_per_node="$gpus" -m "$@"
}

: "${HF_TRAIN_GPUS:=8}"
: "${HF_RL_GPUS:=1}"
: "${HF_BASE_DEPTH:=20}"
: "${HF_BASE_ITERATIONS:=400}"
: "${HF_MID_ITERATIONS:=400}"
: "${HF_SFT_TRAIN_EXAMPLES:=80000}"
: "${HF_SFT_VAL_EXAMPLES:=254}"
: "${HF_RL_EXAMPLES_PER_STEP:=32}"
: "${HF_RL_NUM_SAMPLES:=16}"
: "${HF_RL_MAX_NEW_TOKENS:=128}"
: "${HF_RL_TEMPERATURE:=0.7}"
: "${HF_RL_SAVE_EVERY:=60}"
: "${HF_RL_EVAL_EVERY:=60}"
: "${HF_RL_EVAL_EXAMPLES:=512}"

run_torch "$HF_TRAIN_GPUS" scripts.base_train -- \
  --depth="$HF_BASE_DEPTH" \
  --run="$WANDB_RUN" \
  --num_iterations="$HF_BASE_ITERATIONS"

run_torch "$HF_TRAIN_GPUS" scripts.mid_train -- \
  --run="$WANDB_RUN" \
  --num_iterations="$HF_MID_ITERATIONS"

run_torch "$HF_TRAIN_GPUS" scripts.sft_train -- \
  --run="$WANDB_RUN" \
  --aqua_train_examples="$HF_SFT_TRAIN_EXAMPLES" \
  --aqua_val_examples="$HF_SFT_VAL_EXAMPLES"

run_torch "$HF_RL_GPUS" scripts.chat_rl -- \
  --run="$WANDB_RUN" \
  --examples_per_step="$HF_RL_EXAMPLES_PER_STEP" \
  --num_samples="$HF_RL_NUM_SAMPLES" \
  --max_new_tokens="$HF_RL_MAX_NEW_TOKENS" \
  --temperature="$HF_RL_TEMPERATURE" \
  --save_every="$HF_RL_SAVE_EVERY" \
  --eval_every="$HF_RL_EVAL_EVERY" \
  --eval_examples="$HF_RL_EVAL_EXAMPLES"

log "Evaluating RL checkpoints on AQuA-RAT"
python -m scripts.chat_eval -- -i rl -a AQUA

log "Summarizing nanochat reports"
python -m nanochat.report summarize "$WANDB_RUN"

log "Hugging Face Jobs pipeline complete for run $WANDB_RUN"
