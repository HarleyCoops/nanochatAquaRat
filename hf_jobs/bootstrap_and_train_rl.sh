#!/usr/bin/env bash
set -euo pipefail

# Bootstrap environment inside a Hugging Face Jobs container and run RL training
# for the nanochat project without modifying any existing code.
#
# Configuration via environment variables:
# - REPO_URL: Optional. If set, clone this repo. If not set, assume code is already present.
# - GIT_REF: Optional. Git branch/tag/commit to checkout (if REPO_URL provided).
# - WORKDIR: Optional. Working directory to use inside the container. Default: /workspace
# - RUN: Optional. W&B run name. Default: hf-jobs-rl
# - NPROC_PER_NODE: Optional. Number of GPUs to use with torchrun. Default: 1 (no DDP)
# - RL_ARGS: Optional. Extra CLI overrides passed to scripts.chat_rl after the '--'
#            example: RL_ARGS="--examples_per_step=16 --num_samples=16 --eval_every=120"
# - WANDB_API_KEY: Optional. If set, enables W&B logging (non-dummy).
#
# Example HF Jobs invocation:
# hf jobs run --name nanochat-rl --flavor a10g-small \
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
#   bash hf_jobs/bootstrap_and_train_rl.sh
#
# Example with repo cloning (if Jobs doesn't mount local workspace):
# hf jobs run --name nanochat-rl --flavor a10g-small \
#   -e REPO_URL=https://github.com/you/nanochatAquaRat.git \
#   -e GIT_REF=main \
#   pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
#   bash hf_jobs/bootstrap_and_train_rl.sh

echo "[bootstrap] Starting RL bootstrap..."

WORKDIR="${WORKDIR:-/workspace}"
RUN_NAME="${RUN:-hf-jobs-rl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RL_ARGS="${RL_ARGS:-}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "[bootstrap] Python: $(python --version || true)"
echo "[bootstrap] CUDA: $(python -c 'import torch,sys;print(\"cuda available:\",torch.cuda.is_available());print(\"torch:\",torch.__version__)' || true)"

echo "[bootstrap] Installing system build tools and Rust (for rustbpe via maturin)..."
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential pkg-config libssl-dev
rm -rf /var/lib/apt/lists/*
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version
cargo --version

echo "[bootstrap] Upgrading pip and installing Python build tools..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install maturin

# If the code is not present, optionally clone it
if [[ ! -f "pyproject.toml" ]] || ! grep -q 'name = "nanochat"' pyproject.toml 2>/dev/null; then
  if [[ -n "${REPO_URL:-}" ]]; then
    echo "[bootstrap] Cloning repo from ${REPO_URL} ..."
    git clone --depth=1 "${REPO_URL}" app
    cd app
    if [[ -n "${GIT_REF:-}" ]]; then
      echo "[bootstrap] Checking out ${GIT_REF} ..."
      git fetch --depth=1 origin "${GIT_REF}" || true
      git checkout "${GIT_REF}" || true
    fi
  else
    echo "[bootstrap] ERROR: Source code not found and REPO_URL not provided."
    echo "Provide REPO_URL env var or run the job with the repository mounted/uploaded."
    exit 1
  fi
else
  echo "[bootstrap] Detected nanochat repository in ${WORKDIR}."
fi

echo "[bootstrap] Installing project (this will build rustbpe via maturin)..."
python -m pip install -e .

# Optional: print GPU info
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

echo "[bootstrap] Launching RL training..."
set -x
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m scripts.chat_rl -- --run="${RUN_NAME}" ${RL_ARGS}
else
  python -m scripts.chat_rl -- --run="${RUN_NAME}" ${RL_ARGS}
fi
set +x

echo "[bootstrap] RL training finished."


