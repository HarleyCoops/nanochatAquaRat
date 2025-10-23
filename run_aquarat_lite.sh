#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Load environment variables (e.g., WANDB_API_KEY) from .env if present
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
    echo "[info] Loading environment variables from .env"
    set -a
    source .env
    set +a
fi

# -----------------------------------------------------------------------------
# General environment configuration
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

: "${WANDB_PROJECT:=nanochat-aquarat}"
: "${WANDB_ENTITY:=${WANDB_ENTITY:-}}"
: "${WANDB_MODE:=online}"
: "${WANDB_RUN:=$(date -u +"aquarat-lite-%Y%m%d-%H%M%S")}"

if [ -n "${WANDB_ENTITY}" ]; then
    export WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_RUN
else
    export WANDB_PROJECT WANDB_MODE WANDB_RUN
fi

echo "[info] Using W&B project: ${WANDB_PROJECT}"
echo "[info] Using W&B run: ${WANDB_RUN}"

# -----------------------------------------------------------------------------
# Python virtual environment setup via uv
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[info] Installing uv package manager"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    echo "[info] Creating Python virtual environment"
    uv venv
fi

uv sync --extra gpu
source .venv/bin/activate

# Ensure wandb is authenticated if key available
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[info] Logging into Weights & Biases"
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "[warn] WANDB_API_KEY not found; ensure wandb login prior to launch"
fi

# -----------------------------------------------------------------------------
# Report initialization
# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer build
# -----------------------------------------------------------------------------
if [ ! -f "$HOME/.cargo/bin/cargo" ]; then
    echo "[info] Installing Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------
# Download a limited number of shards for the lightweight run.
python -m nanochat.dataset -n 24 &
DATA_PID=$!

# Prepare the AQuA-RAT dataset locally for RL
AQUA_DIR="$NANOCHAT_BASE_DIR/aqua"
mkdir -p "$AQUA_DIR"
python -m scripts.prepare_aqua --output_dir "$AQUA_DIR"
export AQUA_DATA_DIR="$AQUA_DIR"

# Download eval bundle to prevent crash during base training
EVAL_BUNDLE_URL="https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
if [ ! -f "$NANOCHAT_BASE_DIR/eval_bundle/core.yaml" ]; then
    echo "[info] Downloading eval bundle to prevent base training crash"
    cd "$NANOCHAT_BASE_DIR"
    curl -L -o eval_bundle.zip "$EVAL_BUNDLE_URL"
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    cd "$REPO_ROOT"
fi

wait $DATA_PID || true

# -----------------------------------------------------------------------------
# Base pretraining (depth=8 for smaller model) - COMMENTED OUT FOR LITE TESTING
# -----------------------------------------------------------------------------
# torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
#   --depth=8 \
#   --run="$WANDB_RUN" \
#   --num_iterations=200

# -----------------------------------------------------------------------------
# Mid-stage + SFT (reuse defaults, assuming scripts consume latest checkpoint) - COMMENTED OUT FOR LITE TESTING
# -----------------------------------------------------------------------------
# torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run="$WANDB_RUN" --num_iterations=200
# torchrun --standalone --nproc_per_node=8 -m scripts.sft_train -- \
#   --run="$WANDB_RUN" \
#   --aqua_train_examples=20000 \
#   --aqua_val_examples=254

# -----------------------------------------------------------------------------
# Reinforcement learning on AQuA-RAT (single GPU for quick validation) - SCALED DOWN FOR LITE TESTING
# -----------------------------------------------------------------------------
torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- \
  --run="$WANDB_RUN" \
  --temperature=0.7 \
  --max_new_tokens=64 \
  --device_batch_size=1 \
  --num_samples=4

# -----------------------------------------------------------------------------
# Evaluation on RL outputs
# -----------------------------------------------------------------------------
python -m scripts.chat_eval -- -i rl -a AQUA

# -----------------------------------------------------------------------------
# Summarize reports
# -----------------------------------------------------------------------------
python -m nanochat.report summarize "$WANDB_RUN"

echo "[info] AQuA-RAT LITE run complete for ${WANDB_RUN}"
