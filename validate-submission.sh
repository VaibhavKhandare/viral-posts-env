#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator for Viraltest
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv validate: uv sync (uses .venv/bin/openenv), or pip install openenv-core, or uv on PATH
#   - curl (usually pre-installed)
#
# Run:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Optional: create repo-local .env (gitignored) with HF_TOKEN=... — sourced automatically.
#     cp .env.example .env   # then edit .env
#
# Skip Docker build (Step 2) — faster local checks; run full build before submit:
#     SKIP_DOCKER=1 ./validate-submission.sh https://your-space.hf.space
#
# Step 5 — Hugging Face Inference Router LLM smoke test (runs by default if HF_TOKEN is set):
#     export HF_TOKEN=hf_...   # required for Step 5; never commit; use Space Secrets for deploys
#     # Optional overrides (defaults match inference.py / HF router):
#     export MODEL_NAME=gemma-4-E4B-it-IQ4_XS
#     export API_BASE_URL=https://router.huggingface.co/v1
#     SKIP_LLM_SMOKE=1         # only if you must skip Step 5 (e.g. CI without secrets)
#
# HF token permissions (403 = insufficient permissions):
#   - Create or edit at https://huggingface.co/settings/tokens
#   - For https://router.huggingface.co/v1 the token must be allowed to call
#     Inference Providers / serverless inference for your account (UI labels vary).
#   - If 403 persists, confirm billing/access for Inference Providers in HF account settings.
#   - LLM_SMOKE_OPTIONAL=1 — still pass Steps 1,3–5 when Step 5 auth fails (not for production).
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./viraltest

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  Viraltest Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
if [ "${SKIP_DOCKER:-}" = "1" ]; then
  log "${YELLOW}SKIP_DOCKER=1 — Docker build will be skipped${NC}"
fi
printf "\n"

# ──────────────────────────────────────
# Step 1: Ping HF Space
# ──────────────────────────────────────
log "${BOLD}Step 1/5: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network and that the Space is running."
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running. Try: curl -X POST $PING_URL/reset"
  stop_at "Step 1"
fi

# ──────────────────────────────────────
# Step 2: Docker build
# ──────────────────────────────────────
if [ "${SKIP_DOCKER:-}" = "1" ]; then
  log "${BOLD}Step 2/5: Docker build${NC} ${YELLOW}SKIPPED${NC} (SKIP_DOCKER=1)"
  hint "Run without SKIP_DOCKER=1 before submission to confirm docker build still succeeds."
else
  log "${BOLD}Step 2/5: Running docker build${NC} ..."

  if ! command -v docker &>/dev/null; then
    fail "docker command not found"
    hint "Install Docker: https://docs.docker.com/get-docker/"
    stop_at "Step 2"
  fi

  if [ -f "$REPO_DIR/Dockerfile" ]; then
    DOCKER_CONTEXT="$REPO_DIR"
  elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
    DOCKER_CONTEXT="$REPO_DIR/server"
  else
    fail "No Dockerfile found in repo root or server/ directory"
    stop_at "Step 2"
  fi

  log "  Found Dockerfile in $DOCKER_CONTEXT"

  BUILD_OK=false
  BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

  if [ "$BUILD_OK" = true ]; then
    pass "Docker build succeeded"
  else
    fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
    printf "%s\n" "$BUILD_OUTPUT" | tail -20
    stop_at "Step 2"
  fi
fi

# ──────────────────────────────────────
# Step 3: openenv validate
# ──────────────────────────────────────
log "${BOLD}Step 3/5: Running openenv validate${NC} ..."

VALIDATE_OK=false
VALIDATE_OUTPUT=""
VENV_OPENENV="$REPO_DIR/.venv/bin/openenv"
if command -v uv &>/dev/null && [ -f "$REPO_DIR/pyproject.toml" ]; then
  log "  Using: uv run openenv validate (avoids global CLI / Python mismatch)"
  VALIDATE_OUTPUT=$(cd "$REPO_DIR" && uv run openenv validate 2>&1) && VALIDATE_OK=true
elif command -v openenv &>/dev/null; then
  VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true
elif [ -x "$VENV_OPENENV" ]; then
  log "  Using: .venv/bin/openenv (repo virtualenv; run: uv sync)"
  VALIDATE_OUTPUT=$(cd "$REPO_DIR" && "$VENV_OPENENV" validate 2>&1) && VALIDATE_OK=true
else
  fail "openenv not found (no uv, no openenv on PATH, no .venv/bin/openenv)"
  hint "From the repo: uv sync  # then re-run; or: pip install openenv-core"
  stop_at "Step 3"
fi

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

# ──────────────────────────────────────
# Step 4: Viraltest-specific checks
# ──────────────────────────────────────
log "${BOLD}Step 4/5: Viraltest environment checks${NC} ..."

STEP_OUTPUT=$(portable_mktemp "validate-step")
CLEANUP_FILES+=("$STEP_OUTPUT")

# Test all 3 tasks respond to reset
for TASK in weekly_engage weekly_strategic weekly_competitive; do
  TASK_CODE=$(curl -s -o "$STEP_OUTPUT" -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"$TASK\"}" \
    "$PING_URL/reset" --max-time 15 2>/dev/null || printf "000")

  if [ "$TASK_CODE" = "200" ]; then
    log "  ${GREEN}OK${NC} task=$TASK reset responds"
  else
    fail "Task $TASK reset returned HTTP $TASK_CODE"
    stop_at "Step 4"
  fi
done

# Test step endpoint with a daily plan action (sparse: one post at hour 12)
STEP_CODE=$(curl -s -o "$STEP_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d '{"action":{"scheduled_actions":[{"hour":12,"action_type":"post","content_type":"reel","topic":"AI trends","tags":["ai","ml"]}]}}' \
  "$PING_URL/step" --max-time 15 2>/dev/null || printf "000")

if [ "$STEP_CODE" = "200" ]; then
  pass "Step endpoint responds correctly"
else
  fail "Step endpoint returned HTTP $STEP_CODE"
  stop_at "Step 4"
fi

# Check inference.py exists
if [ -f "$REPO_DIR/inference.py" ]; then
  pass "inference.py found in project root"
else
  fail "inference.py not found in $REPO_DIR"
  stop_at "Step 4"
fi

# ──────────────────────────────────────
# Step 5: HF Inference Router — one chat completion
# ──────────────────────────────────────
DEFAULT_SMOKE_MODEL="gemma-4-E4B-it-IQ4_XS"
DEFAULT_SMOKE_API="https://router.huggingface.co/v1"
SMOKE_MODEL="${MODEL_NAME:-$DEFAULT_SMOKE_MODEL}"
SMOKE_API="${API_BASE_URL:-$DEFAULT_SMOKE_API}"

if [ "${SKIP_LLM_SMOKE:-}" = "1" ]; then
  log "${BOLD}Step 5/5: LLM router smoke test${NC} ${YELLOW}SKIPPED${NC} (SKIP_LLM_SMOKE=1)"
elif [ -z "${HF_TOKEN:-}" ]; then
  fail "Step 5 requires HF_TOKEN (Inference router). Export it from https://huggingface.co/settings/tokens"
  hint "Override model/URL: MODEL_NAME and API_BASE_URL (defaults: $DEFAULT_SMOKE_MODEL, $DEFAULT_SMOKE_API). To skip Step 5: SKIP_LLM_SMOKE=1"
  stop_at "Step 5"
else
  log "${BOLD}Step 5/5: LLM router smoke test${NC} (model=$SMOKE_MODEL) ..."
  LLM_OK=false
  LLM_OUT=""
  if [ ! -f "$REPO_DIR/pyproject.toml" ]; then
    fail "No pyproject.toml in repo — cannot run LLM smoke test"
    stop_at "Step 5"
  fi
  RUN_PYTHON=()
  if command -v uv &>/dev/null; then
    RUN_PYTHON=(uv run python)
  elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
    RUN_PYTHON=("$REPO_DIR/.venv/bin/python")
  else
    fail "Need uv on PATH or .venv/bin/python (run: uv sync)"
    stop_at "Step 5"
  fi
  if [ "${#RUN_PYTHON[@]}" -gt 0 ]; then
    LLM_OUT=$(cd "$REPO_DIR" && \
      MODEL_NAME="$SMOKE_MODEL" API_BASE_URL="$SMOKE_API" HF_TOKEN="$HF_TOKEN" \
      "${RUN_PYTHON[@]}" - <<'PY' 2>&1
import os, sys
from openai import OpenAI

def main() -> None:
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"].rstrip("/"),
        api_key=os.environ["HF_TOKEN"],
    )
    r = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=32,
        temperature=0.0,
    )
    text = (r.choices[0].message.content or "").strip()
    if not text:
        print("empty completion", file=sys.stderr)
        sys.exit(1)
    print(text[:500])

if __name__ == "__main__":
    main()
PY
    ) && LLM_OK=true
  fi

  if [ "$LLM_OK" = true ]; then
    pass "LLM router responded"
    if [ -n "$LLM_OUT" ]; then
      preview="${LLM_OUT:0:120}"
      [ "${#LLM_OUT}" -gt 120 ] && preview="${preview}..."
      log "  completion: $preview"
    fi
  else
    fail "LLM router smoke test failed"
    printf "%s\n" "$LLM_OUT"
    if [ "${LLM_SMOKE_OPTIONAL:-}" = "1" ]; then
      hint "LLM_SMOKE_OPTIONAL=1 set — continuing (fix HF token / Inference Providers access for real inference runs)."
    else
      hint "403 often means the token cannot use Inference Providers for this account. See HF token settings or set LLM_SMOKE_OPTIONAL=1 to still pass Steps 1–4."
      stop_at "Step 5"
    fi
  fi
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
