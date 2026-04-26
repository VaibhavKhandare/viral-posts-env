#!/usr/bin/env bash
# Launch a HF Job that:
#   1. clones the Space repo (default: ycwhencpp/train-new) into /work
#   2. execs training/hf_remote_run.sh from that clone (heavy lifting lives there)
#
# We keep the inline bootstrap intentionally tiny — anything larger risks the
# "File name too long" failure when the whole command becomes argv to bash -lc.
#
# Requires: hf auth login (token forwarded via --secrets HF_TOKEN)

set -euo pipefail

IMAGE="${HF_JOB_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
<<<<<<< HEAD
FLAVOR="${HF_JOB_FLAVOR:-l40sx1}"
=======
FLAVOR="${HF_JOB_FLAVOR:-a100x4}"
>>>>>>> 9536a3362f152cb5a38d8dbb3c28f80766199f58
TIMEOUT="${HF_JOB_TIMEOUT:-8h}"
SPACE_REPO="${HF_SPACE_REPO_ID:-ycwhencpp/final-iteration}"
NB_EXEC_TIMEOUT="${NB_EXEC_TIMEOUT:-14400}"
SMOKE_MODE="${SMOKE_MODE:-0}"

if ! hf auth whoami &>/dev/null; then
  echo "Run: hf auth login" >&2
  exit 1
fi

# Tiny bootstrap: install git, clone the Space, hand off to the in-repo script.
BOOTSTRAP='set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates >/dev/null
rm -rf /work
git clone --depth 1 "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}" /work
cd /work
<<<<<<< HEAD
papermill --log-output --progress-bar --execution-timeout "${NB_EXEC_TIMEOUT}" \
  training/train_grpo.ipynb training/train_grpo.executed.ipynb
python training/export_io_pairs.py plots/io_log.jsonl plots/io_pairs.json
python -c "import os; from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='.', path_in_repo='run-output', repo_id=os.environ['SPACE_REPO'], repo_type='space', allow_patterns=['training/train_grpo.executed.ipynb','plots/**','**/lora-*/**'])"
EOS
)
=======
chmod +x training/hf_remote_run.sh
exec bash training/hf_remote_run.sh'
>>>>>>> 9536a3362f152cb5a38d8dbb3c28f80766199f58

# Use `--` to terminate hf CLI option parsing — otherwise `bash -lc <script>`
# is parsed as `--label c <script>` (typer consumes the `-l` short flag).
exec hf jobs run \
  --flavor "$FLAVOR" \
  --detach \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  --env "SPACE_REPO=$SPACE_REPO" \
  --env "NB_EXEC_TIMEOUT=$NB_EXEC_TIMEOUT" \
  --env "SMOKE_MODE=$SMOKE_MODE" \
  -- "$IMAGE" bash -c "$BOOTSTRAP"
