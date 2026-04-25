#!/usr/bin/env bash
# Run train_grpo.ipynb on Hugging Face Jobs from your machine.
# Prereqs: hf auth login  (or export HF_TOKEN for API + --secrets HF_TOKEN below)
#
# Optional — hf skills add (newer CLI only; do not upgrade global hf if you use transformers):
#   uv venv .venv-hf && . .venv-hf/bin/activate && pip install -U 'huggingface_hub>=1.11' typer && hf skills add

set -euo pipefail

IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
FLAVOR="${HF_JOB_FLAVOR:-l4x1}"
TIMEOUT="${HF_JOB_TIMEOUT:-8h}"
REPO_URL="${HF_REPO_URL:-https://github.com/VaibhavKhandare/viral-posts-env.git}"
REPO_BRANCH="${HF_REPO_BRANCH:-hack1}"

exec hf jobs run \
  --flavor "$FLAVOR" \
  --detach \
  --timeout "$TIMEOUT" \
  --env "REPO_URL=$REPO_URL" \
  --env "REPO_BRANCH=$REPO_BRANCH" \
  "$IMAGE" \
  bash -lc 'set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y --no-install-recommends git curl
rm -rf /work && git clone --depth 1 --branch "${REPO_BRANCH}" "${REPO_URL}" /work
cd /work
pip install -q --root-user-action=ignore jupyter nbconvert nbclient ipykernel
jupyter nbconvert --to notebook --execute training/train_grpo.ipynb \
  --ExecutePreprocessor.timeout=86400 --inplace'
