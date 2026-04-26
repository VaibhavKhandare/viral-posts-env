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
FLAVOR="${HF_JOB_FLAVOR:-a100x4}"
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
chmod +x training/hf_remote_run.sh
exec bash training/hf_remote_run.sh'

exec hf jobs run \
  --flavor "$FLAVOR" \
  --detach \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  --env "SPACE_REPO=$SPACE_REPO" \
  --env "NB_EXEC_TIMEOUT=$NB_EXEC_TIMEOUT" \
  --env "SMOKE_MODE=$SMOKE_MODE" \
  "$IMAGE" \
  bash -lc "$BOOTSTRAP"
