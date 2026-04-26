#!/usr/bin/env bash
# Remote runner executed inside the HF Job container.
# The launcher (training/hf_run_space_train_job.sh) clones the Space into /work
# and then exec's this script. Keeping the heavy lifting in a file (instead of
# inlining via `bash -lc`) avoids the argv "File name too long" failure.
#
# Required env (set by launcher):
#   SPACE_REPO          e.g. ycwhencpp/train-new
#   HF_TOKEN            secret, injected via --secrets
#   NB_EXEC_TIMEOUT     papermill execution timeout in seconds (default 14400)
#   SMOKE_MODE          0 = full training, 1 = smoke (default 0 here)

set -euo pipefail

: "${SPACE_REPO:?SPACE_REPO must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set (use --secrets HF_TOKEN)}"

NB_EXEC_TIMEOUT="${NB_EXEC_TIMEOUT:-14400}"
export SMOKE_MODE="${SMOKE_MODE:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DEBIAN_FRONTEND=noninteractive

echo "===== hf_remote_run.sh starting at $(date -u +%FT%TZ) ====="
echo "SPACE_REPO=${SPACE_REPO}"
echo "SMOKE_MODE=${SMOKE_MODE}"
echo "NB_EXEC_TIMEOUT=${NB_EXEC_TIMEOUT}"

nvidia-smi || true

echo "----- pip installs -----"
pip install -q --root-user-action=ignore --upgrade \
  "typing_extensions>=4.15.0" \
  jupyter nbconvert nbclient ipykernel \
  huggingface_hub hf_transfer papermill

echo "----- executing notebook with papermill -----"
mkdir -p plots run-output checkpoints
papermill --log-output --progress-bar \
  --execution-timeout "${NB_EXEC_TIMEOUT}" \
  training/train_grpo.ipynb \
  training/train_grpo.executed.ipynb

echo "----- uploading artifacts back to Space (run-output/) -----"
python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path=".",
    path_in_repo="run-output",
    repo_id=os.environ["SPACE_REPO"],
    repo_type="space",
    allow_patterns=[
        "training/train_grpo.executed.ipynb",
        "plots/**",
        "checkpoints/**/adapter_*",
        "checkpoints/**/lora-*/**",
        "**/lora-*/**",
    ],
    commit_message="HF Job: train_grpo run output",
)
print("upload complete")
PY

echo "===== hf_remote_run.sh done at $(date -u +%FT%TZ) ====="
