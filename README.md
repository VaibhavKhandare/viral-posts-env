---
title: Viraltest — Creator Optimization Agent
emoji: 📊
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Viraltest — RL-Based Creator Optimization Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates a social media creator’s weekly posting lifecycle. An AI agent learns **when to post**, **what format**, **which tags**, and **how to differentiate from competitors** — maximizing engagement while managing burnout and sleep.

## Submission requirements — how this repo maps

Use this table to confirm Phase 1 (automated) gates before you submit.

| Requirement | Status in this repo | Where to verify |
|---------------|---------------------|-----------------|
| Real-world task (not a toy/game) | **Met** — creator scheduling, energy, trends, competitors | `server/viraltest_environment.py`, `DESIGN.md` |
| Full OpenEnv spec: `openenv.yaml`, typed models, HTTP API | **Met** | `openenv.yaml`, `models.py`, `server/app.py` (`create_app`) |
| `step()` / `reset()` / `state()` | **Met** — standard OpenEnv HTTP endpoints | Run `openenv validate` |
| ≥3 tasks with graders (easy → hard), scores in **0.0–1.0** | **Met** — `weekly_engage`, `weekly_strategic`, `weekly_competitive` | `_run_grader()` in `server/viraltest_environment.py` |
| Meaningful reward + partial progress | **Met** — per-step `_compute_reward()` | `_compute_reward()` |
| Baseline inference script, reproducible | **Met** — root `inference.py` | See **Baseline inference** below |
| `Dockerfile` builds | **Expected** — root `Dockerfile` | `docker build -t viraltest .` (run locally) |
| HF Space deploys; `POST /reset` returns **200** | **You must configure** | See **Hugging Face Spaces** — ping **Space root**, not only `/web` |
| `openenv validate` passes | **Met** in dev (`.venv/bin/openenv validate`) | CI / local |
| Env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | **Documented** — `inference.py` reads them (see **Environment variables**) | HF Space **Settings → Secrets** |
| `inference.py` at repo root; OpenAI client for LLM calls | **Met** | `inference.py` |
| Structured stdout: `[START]`, `[STEP]`, `[END]` | **Met** — match field order in `log_*` helpers | `inference.py` |
| Inference under 20 minutes; 2 vCPU / 8 GB | **Check** — 3 tasks × up to 168 steps each = many LLM calls; use a fast endpoint and sensible `MAX_TOKENS` | `inference.py` |

### Minor items to double-check before judging

1. **`[STEP]` `error=` field** — The spec asks for the raw `last_action_error` or `null`. This repo logs errors with spaces replaced by underscores so each line stays a single token after `error=`. If the organizer’s parser expects literal spaces inside unquoted messages, align with their sample; otherwise this is fine for one-line logs.
2. **Default `API_BASE_URL` in `inference.py`** — Defaults are for local dev. On Hugging Face, set **`API_BASE_URL`** (e.g. `https://router.huggingface.co/v1`) and **`MODEL_NAME`** in Secrets so evaluation matches your setup.
3. **Space URL for the validator** — The official script POSTs to `{your_space_url}/reset` with body `{}`. That must be the **root** of the Space (e.g. `https://YOURNAME-spacename.hf.space`), not the Gradio path under `base_path: /web`. Confirm with curl (see **Pre-submission validation**).

---

## Why this matters

Many creators burn out while optimizing posting times and formats. This environment turns that tradeoff into a reproducible simulation so agents can be trained and compared on the same weekly horizon (**168** hourly steps).

---

## Quick Start (Python)

The HTTP client is **async** (same pattern as root `inference.py`):

```python
import asyncio
from viraltest import ViraltestAction, ViraltestEnv

async def main():
    env = ViraltestEnv(base_url="http://localhost:8000")
    try:
        result = await env.reset(task="weekly_engage")
        action = ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="AI trends",
            tags=["ai", "coding", "devtools"],
        )
        result = await env.step(action)
        print(result.observation.engagement_rate, result.observation.creator_energy)
    finally:
        await env.close()

asyncio.run(main())
```

---

## Action space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"post" \| "rest" \| "create_content"` | What the agent does this hour |
| `content_type` | `"reel" \| "story" \| "carousel" \| "text_post"` | Required when posting |
| `topic` | `str` (≤200 chars) | Post topic |
| `tags` | `list[str]` (≤5) | Tags from the environment tag pool |

---

## Observation space (high level)

| Field | Description |
|-------|-------------|
| `current_hour`, `day_of_week`, `days_elapsed` | Simulated calendar |
| `creator_energy`, `hours_since_sleep`, `sleep_debt` | Burnout and sleep |
| `follower_count`, `engagement_rate` | Growth and rolling engagement |
| `trending_topics`, `trending_tags`, `tag_performance` | Trends and learned tag quality |
| `competitor_recent_posts`, `competitor_avg_engagement`, `niche_saturation` | Competition |
| `error`, `reward`, `done`, `metadata` | Errors, shaping reward, termination, **`metadata["grader_score"]` at episode end** |

Full schema: `GET /schema` when the server is running.

---

## Tasks and graders (168 steps each)

| Task | Difficulty | Grader focus |
|------|------------|--------------|
| `weekly_engage` | Easier | Total engagement vs theoretical max; burnout penalty |
| `weekly_strategic` | Medium | Engagement + tag discovery/exploitation + energy + consistency |
| `weekly_competitive` | Hard | Adds growth vs competitors, differentiation, diversity constraints |

Episode ends after **168** steps or if **energy ≤ 0**. Final normalized score is in **`observation.metadata["grader_score"]`** in **\[0, 1\]**.

---

## Reward shaping

Per-step reward in **`[0, 1]`** combines engagement, energy change, posting consistency, tags, and competitor differentiation (`_compute_reward` in `server/viraltest_environment.py`). It is dense enough for learning signals before the terminal grader runs.

---

## Local development

```bash
git clone <your-repo-url>
cd viral-posts-env   # or your fork name

# Install (uv recommended; pip works too)
uv sync
# source .venv/bin/activate   # optional

# Terminal 1 — API server
uvicorn viraltest.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — optional UI
# Open http://localhost:8000/dashboard  (see server routes in server/app.py)
```

Validate the OpenEnv layout:

```bash
.venv/bin/openenv validate
# Expect: [OK] ... Ready for multi-mode deployment
```

---

## Docker

From the repository root (same directory as `Dockerfile`):

```bash
docker build -t viraltest-env:latest .
docker run --rm -p 8000:8000 viraltest-env:latest
```

Smoke test:

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/reset
# Expect: 200
```

---

## Hugging Face Spaces — deploy

1. **Create a Space** with **Docker** SDK (this repo’s README frontmatter uses `sdk: docker`).
2. **Push this repository** (or connect GitHub) so the Space builds from the root `Dockerfile`.
3. **Settings → Variables and secrets** — add at least:
   - **`HF_TOKEN`** — Hugging Face API token for inference (and Space pull if private).
   - **`API_BASE_URL`** — OpenAI-compatible base URL (e.g. `https://router.huggingface.co/v1`).
   - **`MODEL_NAME`** — Model id for that router (e.g. `Qwen/Qwen2.5-72B-Instruct`).
4. **App port** — `8000` (see frontmatter `app_port: 8000`).
5. **`base_path: /web`** — Used for the bundled web UI; the **REST** endpoints (`/reset`, `/step`, `/state`) remain on the **Space root host** as required by the submission validator. **Always test** `https://<your-space>.hf.space/reset` (not only `/web/...`).

Optional CLI (if you use OpenEnv’s tooling):

```bash
pip install openenv-core
openenv push   # follow OpenEnv docs for auth and target Space
```

---

## Baseline inference (`inference.py`)

**Location:** repository root — **`inference.py`** (required by the hackathon).

**LLM client:** OpenAI-compatible client (`from openai import OpenAI`) using:

| Variable | Role |
|----------|------|
| `API_BASE_URL` | OpenAI-compatible API base |
| `MODEL_NAME` | Model name for `chat.completions` |
| `HF_TOKEN` | Preferred API key (fallbacks: `OPENAI_API_KEY`, `API_KEY`) |
| `IMAGE_NAME` / `LOCAL_IMAGE_NAME` | If using `ViraltestEnv.from_docker_image(...)` instead of HTTP |
| `ENV_BASE_URL` | HTTP server URL (default `http://localhost:8000`) |

**Stdout format (must not change field names or order):**

```text
[START] task=<name> env=<benchmark> model=<model>
[STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

Run locally (server on port 8000):

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
uv sync && .venv/bin/python inference.py
```

**Short episodes for debugging** — `ALLOW_SHORT_EPISODE=1` and `MAX_STEPS` can shorten runs; full weekly tasks still use **168** steps unless you override (see comments in `inference.py`).

---

## Pre-submission validation

Use the provided script (same checks as the official template: ping Space, Docker build, `openenv validate`):

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://YOUR-SPACE.hf.space /path/to/viral-posts-env
```

Or download the organizer’s script from their repo and pass your Space URL.

**Manual ping (required to pass automated gate):**

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  https://YOUR-SPACE.hf.space/reset
# Must print: 200
```

---

## Baseline scores (reference)

Deterministic dashboard agents (not the LLM) — see `README` tables in-repo history / `DESIGN.md` for methodology. Your **`inference.py`** scores will vary by model and endpoint; keep runs under the **20-minute** inference budget.

---

## Project structure

```
.
├── inference.py              # Hackathon-required baseline (LLM + [START]/[STEP]/[END])
├── openenv.yaml              # OpenEnv manifest
├── models.py                 # ViraltestAction, ViraltestObservation
├── client.py                 # ViraltestEnv client
├── Dockerfile
├── validate-submission.sh    # Local preflight
├── test_scenarios.py         # Offline env tests
├── DESIGN.md                 # Deep design / research notes
└── server/
    ├── app.py                # FastAPI + create_app
    ├── viraltest_environment.py
    └── dashboard.html
```

---

## License

See `LICENSE` in the repository root (BSD-style per upstream OpenEnv examples).
