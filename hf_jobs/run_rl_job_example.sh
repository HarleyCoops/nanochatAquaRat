#!/usr/bin/env bash
set -euo pipefail

# Helper to launch an RL training job on Hugging Face Jobs using the bootstrap script.
# Requires the `hf` CLI to be installed and authenticated.
#
# Usage (edit FLAVOR/IMAGE as needed):
#   bash hf_jobs/run_rl_job_example.sh
#
# Or pass overrides:
#   RUN=rl-demo NPROC_PER_NODE=1 FLAVOR=a10g-small IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash hf_jobs/run_rl_job_example.sh

RUN="${RUN:-rl-gsm8k-a10g}"
FLAVOR="${FLAVOR:-a10g-small}"
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

CMD=(hf jobs run --name "nanochat-rl-${RUN}" --flavor "${FLAVOR}")
if [[ -n "${WANDB_API_KEY}" ]]; then
  CMD+=(-e "WANDB_API_KEY=${WANDB_API_KEY}")
fi

# If your hf CLI doesn't upload the local repo, uncomment and set REPO_URL:
# CMD+=(-e "REPO_URL=https://github.com/you/nanochatAquaRat.git" -e "GIT_REF=main")

CMD+=(-e "RUN=${RUN}" "${IMAGE}" bash hf_jobs/bootstrap_and_train_rl.sh)

echo "Launching: ${CMD[*]}"
"${CMD[@]}"


