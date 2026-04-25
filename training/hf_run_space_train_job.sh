#!/usr/bin/env bash
# Same environment as your HF Job (Space clone + nbconvert + upload to Space).
# Old UI command was invalid shell (no &&); this version is a proper chain.
#
# Requires: hf auth login (token is sent via --secrets HF_TOKEN from the CLI cache)
# Optional: HF_SPACE_REPO_ID (default vaibhavkhandare/train-bhai-train)

set -euo pipefail

IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
FLAVOR="${HF_JOB_FLAVOR:-l4x1}"
TIMEOUT="${HF_JOB_TIMEOUT:-8h}"
SPACE_REPO="${HF_SPACE_REPO_ID:-vaibhavkhandare/train-bhai-train}"
NB_EXEC_TIMEOUT="${NB_EXEC_TIMEOUT:-3600}"

if ! hf auth whoami &>/dev/null; then
  echo "Run: hf auth login" >&2
  exit 1
fi

REMOTE_SCRIPT=$(cat <<'EOS'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y --no-install-recommends git curl ca-certificates
pip install -q --root-user-action=ignore --upgrade "typing_extensions>=4.15.0" jupyter nbconvert nbclient ipykernel huggingface_hub papermill
rm -rf /work
git clone --depth 1 "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}" /work
cd /work
papermill --log-output --progress-bar --execution-timeout "${NB_EXEC_TIMEOUT}" \
  training/train_grpo.ipynb training/train_grpo.executed.ipynb
python -c "import os; from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='.', path_in_repo='run-output', repo_id=os.environ['SPACE_REPO'], repo_type='space', allow_patterns=['training/train_grpo.executed.ipynb','plots/**','**/lora-*/**'])"
EOS
)

exec hf jobs run \
  --flavor "$FLAVOR" \
  --detach \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  --env "SPACE_REPO=$SPACE_REPO" \
  --env "NB_EXEC_TIMEOUT=$NB_EXEC_TIMEOUT" \
  "$IMAGE" \
  bash -lc "$REMOTE_SCRIPT"
