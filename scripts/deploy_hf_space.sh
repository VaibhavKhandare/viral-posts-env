#!/usr/bin/env bash
# Deploy the Viraltest env to a Hugging Face Space using the openenv CLI.
#
# Required: HF_TOKEN exported AND the target Space already exists OR you have
# permission to create it under HF_USERNAME.
#
# Usage:
#   HF_USERNAME=your-handle ./scripts/deploy_hf_space.sh [--dry-run]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face handle}"
: "${HF_TOKEN:?Set HF_TOKEN (a write-scoped HF token)}"
SPACE="${HF_SPACE:-${HF_USERNAME}/viraltest}"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "==> Validating environment ..."
.venv/bin/openenv validate

echo "==> Building image ..."
.venv/bin/openenv build $DRY_RUN

echo "==> Pushing to Hugging Face Space: ${SPACE}"
.venv/bin/openenv push --space "$SPACE" $DRY_RUN

echo
echo "Live URL (once the Space finishes building):"
echo "  https://huggingface.co/spaces/${SPACE}"
echo
echo "Smoke-test it with:"
echo "  SPACE_URL=https://${HF_USERNAME}-viraltest.hf.space ./scripts/verify_space.sh"
