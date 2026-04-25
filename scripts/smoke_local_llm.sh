#!/usr/bin/env bash
# Smoke-test the local LLM endpoint before kicking off long runs.
# Exits 0 if reachable and at least one model is exposed.
set -euo pipefail

URL="${API_BASE_URL:-http://0.0.0.0:1337/v1}"
echo "Probing ${URL}/models ..."
RESP="$(curl -fsS --max-time 5 "${URL}/models")"
MODEL_ID="$(printf '%s' "$RESP" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['data'][0]['id'])")"
echo "OK: ${MODEL_ID}"
