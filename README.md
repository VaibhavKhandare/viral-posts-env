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

An OpenEnv environment that simulates a social media creator's weekly posting lifecycle. An AI agent learns **when to post**, **what format**, **which tags**, and **how to differentiate from competitors** — maximizing engagement while managing burnout.

## Why This Matters

90% of content creators experience burnout (Sozee 2026). Existing analytics tools show past performance but don't actively guide strategy. This environment lets RL agents learn sustainable, high-engagement posting strategies through simulation backed by real-world data.

## Quick Start

```python
from viraltest import ViraltestAction, ViraltestEnv

with ViraltestEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task="weekly_engage")

    action = ViraltestAction(
        action_type="post",
        content_type="reel",
        topic="AI trends",
        tags=["ai", "tech", "coding"]
    )
    result = env.step(action)
    print(f"Engagement: {result.observation.engagement_rate}")
    print(f"Energy: {result.observation.creator_energy}")
    print(f"Followers: {result.observation.follower_count}")
```

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | `"post" \| "rest" \| "create_content"` | What the agent does |
| `content_type` | `"reel" \| "story" \| "carousel" \| "text_post"` | Post format (required if posting) |
| `topic` | `str` (max 200 chars) | Content topic |
| `tags` | `list[str]` (max 5) | Hashtags from the tag pool |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `current_hour` | `int` (0–23) | Simulated hour |
| `day_of_week` | `int` (0–6) | Mon=0, Sun=6 |
| `creator_energy` | `float` (0–1) | Burnout meter |
| `follower_count` | `int` | Current followers |
| `engagement_rate` | `float` | Rolling avg of last 10 posts |
| `posts_today` | `int` | Posts made today |
| `trending_topics` | `list[str]` | Currently trending topics |
| `trending_tags` | `list[str]` | Currently trending tags |
| `tag_performance` | `dict[str, float]` | Your per-tag engagement history |
| `competitor_recent_posts` | `list[dict]` | Recent posts from 3 rival creators |
| `niche_saturation` | `float` (0–1) | How crowded your topic space is |

## Tasks (All Weekly — 168 Steps)

| Task | Difficulty | What's Graded |
|---|---|---|
| `weekly_engage` | Easy | Pure engagement (timing + content type + energy) |
| `weekly_strategic` | Medium | + tag discovery/exploitation + energy sustainability |
| `weekly_competitive` | Hard | + competitor awareness + follower growth + differentiation |

## Reward Function

Per-step composite reward (0.0–1.0):
- 30% engagement gained
- 15% energy management
- 15% posting consistency
- 15% tag optimization
- 15% competitor differentiation
- 10% burnout penalty (if energy < 0.2)

## Research-Backed Parameters

- **Engagement rates**: SocialInsider 2025 — carousel 0.55%, reel 0.52%, story 0.30%
- **Peak hours**: Buffer 9.6M post study — 12-3PM Tue-Thu = 1.4x multiplier
- **Burnout**: Sozee 2026 — 90% creators affected, 30-52% productivity drop at burnout
- **Content fatigue**: CreatorsJet 10K post study — reels give 2.25x reach vs static images

## Setup

### Local Development

```bash
# Install dependencies
uv sync

# Run the server
uvicorn viraltest.server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t viraltest-env:latest .
docker run -p 8000:8000 viraltest-env:latest
```

### Deploy to HF Spaces

```bash
openenv push
```

### Run Inference

```bash
export HF_TOKEN=your_token   # or OPENAI_API_KEY / API_KEY
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# Optional: LOCAL_IMAGE_NAME / IMAGE_NAME for ViraltestEnv.from_docker_image()
# Optional: ALLOW_SHORT_EPISODE=1 to cap steps below 168 (no final grader score unless episode ends)
python inference.py
```

**Pre-submission checks:** Automated validation posts to **`{your_space_url}/reset`** at the Space **root** (same host as the Space, not the `/web` UI path). Ensure that URL returns HTTP 200.

## Baseline Scores

| Agent | weekly_engage | weekly_strategic | weekly_competitive |
|---|---|---|---|
| Random | 0.15 | 0.10 | 0.00 |
| Always rest | 0.00 | 0.05 | 0.04 |
| Spam (post every step) | 0.00 | 0.00 | 0.00 |
| Smart (deterministic) | 0.91 | 0.92 | 0.87 |

## Project Structure

```
viraltest/
├── __init__.py               # Module exports
├── models.py                 # Action & Observation Pydantic models
├── client.py                 # WebSocket client (ViraltestEnv)
├── inference.py              # LLM-driven agent with logging
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Dependencies
├── Dockerfile                # Container build
├── DESIGN.md                 # Full design document
├── README.md                 # This file
└── server/
    ├── __init__.py
    ├── app.py                # FastAPI application
    └── viraltest_environment.py  # Simulation engine + graders
```
