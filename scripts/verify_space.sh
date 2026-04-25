#!/usr/bin/env bash
# Verify a deployed Viraltest env by hitting the canonical OpenEnv endpoints.
#
# Usage:
#   SPACE_URL=https://your-handle-viraltest.hf.space ./scripts/verify_space.sh
#
# Exits 0 if all checks pass, non-zero otherwise.
set -euo pipefail

URL="${SPACE_URL:-http://localhost:8000}"
echo "Verifying ${URL}"

echo -n "  GET  /tools          ... "
COUNT="$(curl -fsS --max-time 15 "${URL}/tools" | python3 -c "import json,sys;print(json.load(sys.stdin)['count'])")"
if [[ "$COUNT" != "8" ]]; then
  echo "FAIL (expected 8 tools, got $COUNT)"; exit 1
fi
echo "OK ($COUNT tools)"

echo -n "  POST /reset          ... "
CODE="$(curl -s -o /tmp/reset.json -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{"task":"monthly_engage"}' "${URL}/reset")"
if [[ "$CODE" != "200" ]]; then
  echo "FAIL (HTTP $CODE)"; cat /tmp/reset.json; exit 1
fi
echo "OK (HTTP 200)"

echo -n "  POST /step (rest)    ... "
CODE="$(curl -s -o /tmp/step.json -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{"action":{"scheduled_actions":[]}}' "${URL}/step")"
if [[ "$CODE" != "200" ]]; then
  echo "FAIL (HTTP $CODE)"; cat /tmp/step.json; exit 1
fi
echo "OK (HTTP 200)"

echo -n "  GET  /state          ... "
CODE="$(curl -s -o /dev/null -w "%{http_code}" "${URL}/state")"
[[ "$CODE" == "200" ]] || { echo "FAIL (HTTP $CODE)"; exit 1; }
echo "OK"

echo
echo "All checks passed for ${URL}"
