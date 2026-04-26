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

# Viraltest v2 — World-Modeling RL Environment for Instagram Strategy

> **Theme #3.1 — Professional Tasks (World Modeling)**
> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an LLM agent manages an Instagram creator account over 30 simulated days, discovering the world through tools rather than being told the rules.

## What this teaches the LLM

| Capability | How the environment tests it |
|---|---|
| **Tool discovery & orchestration** | 8 discoverable tools (`query_trends`, `query_competitor`, `predict_engagement`...). Agent must call `GET /tools` to learn what's available. |
| **Persistent world model** | 30-day horizon. Multi-episode brand chain carries state across months. |
| **Belief tracking** | `notes` field persists hypotheses day-to-day. Agent must update beliefs from tool results. |
| **Causal reasoning** | `coach_feedback` returns counterfactual delta (your plan vs. heatmap-optimal). `predict_engagement` lets agent test hypotheses before committing. |
| **Partial observability** | Default observation is sparse: energy, followers, reward. Rich data (trends, competitors, tags) only via tools. |
| **Multi-step workflow** | Per day: discover → query → draft → predict → commit → reply → learn from feedback. |

## Why this matters

The $250B creator economy ([Goldman Sachs, 2025](https://www.goldmansachs.com/insights/articles/the-creator-economy-could-approach-half-a-trillion-dollars-by-2027)) has 67M creators, but 73% experience burnout ([Awin, 2024](https://www.prweb.com/releases/a-majority-of-content-creators-and-influencers-struggle-with-burnout-as-concerns-for-ai-begin-to-surface-according-to-a-new-awin-group-survey-research-302257152.html)). This environment turns the posting-vs-burnout tradeoff into a reproducible simulation calibrated against 10+ verifiable sources.

## Quick Start

```python
import asyncio
from viraltest import ViraltestAction, ViraltestEnv
from viraltest.models import ToolCall

async def main():
    env = ViraltestEnv(base_url="http://localhost:8000")
    try:
        result = await env.reset(task="monthly_strategic")
        action = ViraltestAction(
            tool_calls=[
                ToolCall(name="query_trends", arguments={"niche": "tech"}),
            ],
            scheduled_actions=[
                {"hour": 12, "action_type": "post", "content_type": "reel",
                 "topic": "AI tools", "tags": ["ai", "coding"], "intent": "watch_bait"},
            ],
            notes="Day 1: querying trends to establish baseline.",
        )
        result = await env.step(action)
        print(result.observation.engagement_signals)
    finally:
        await env.close()

asyncio.run(main())
```

## Simulation mechanics

### Engagement signals (Mosseri Jan-2025)

Instagram's head confirmed the top-3 ranking signals. Our reward decomposes engagement accordingly:

| Signal | Weight | Best format | Source |
|--------|--------|-------------|--------|
| Watch time | 0.40 | Reels | Mosseri Jan-2025 |
| Sends per reach | 0.30 | Stories | Mosseri Jan-2025 |
| Saves | 0.20 | Carousels | Mosseri Jan-2025 |
| Likes per reach | 0.10 | Text posts | Mosseri Jan-2025 |

### Hour heatmap

7×24 multiplier grid from [Buffer 9.6M posts](https://buffer.com/resources/when-is-the-best-time-to-post-on-instagram) cross-validated with [Sprout Social 2B engagements](https://sproutsocial.com/insights/best-times-to-post-on-social-media/).

### Sleep model

Piecewise-linear from [Van Dongen et al. 2003](https://pubmed.ncbi.nlm.nih.gov/12683469) (*Sleep*, PMID 12683469): no quality loss below 16h awake, then 6.25% per hour, floor at 30%.

### Audience fatigue

Tiered from [Buffer 2.1M study](https://buffer.com/resources/how-often-to-post-on-instagram/): 2 posts/day=1.0×, 3=0.75×, 4=0.50×, 5+=0.25×. Weekly cap at 7 posts → 0.75×.

## Tasks and graders (30 steps each)

| Task | Difficulty | Grader focus |
|------|-----------|--------------|
| `monthly_engage` | Easier | Total engagement vs theoretical max; burnout penalty |
| `monthly_strategic` | Medium | + tag discovery/exploitation + energy + consistency |
| `monthly_competitive` | Hard | + growth vs competitors + differentiation + content diversity |

## Regulator/Judge Mode (per-day audit)

Every day the env emits a deterministic, explainable `JudgeReport` on the observation:

```python
JudgeReport(
    policy_compliance=1.00,    # 1.0 - sum(weighted_violations); see _compute_judge_report
    sustainability_risk=0.10,  # 0.4*(1-energy_min) + 0.3*sleep_debt + 0.3*low_energy_ratio
    strategic_quality=0.96,    # 0.4*engagement_per_post + 0.3*intent_diversity + 0.3*format_diversity
    explanation="compliance=1.00 risk=0.10 strategy=0.96 | no policy violations",
    violations=[],             # human-readable rule breaks (Buffer 2.1M, Van Dongen, Cen 2024)
)
```

Auditable rules (all sourced): >5 posts/day → fatigue cliff (Buffer 2.1M); >7 posts/week → weekly cap; ≥4 collabs/month → diminishing returns (Cen 2024); >22h awake → sleep debt (Van Dongen 2003).

## Headline metrics (final-step audit)

The final observation carries `HeadlineMetrics` with the three numbers judges remember:

| Metric | What it measures | Source of truth |
|---|---|---|
| `vs_baseline_pct` | (agent_score − heuristic_baseline) / heuristic_baseline | Empirical baseline loaded from `plots/training_summary.json["smart_heuristic"]` (0.43 / 0.77 / 0.81) |
| `score_per_tool_call` | grader_score / total_tool_calls | Efficiency: did the agent learn to call tools sparingly? |
| `score_per_1k_chars` | grader_score per 1k action JSON chars | Token-proxy efficiency |
| `retention_under_shift` | shifted_score / baseline_score | Pass `episode_chain_id` + `shift_label="baseline"` then `="shifted"` to a second `reset` to populate. None until both runs complete. |

## Tool catalog

| Tool | Cost | Returns |
|------|------|---------|
| `query_trends` | 1 | Trending topics, tags, niche saturation |
| `query_competitor` | 2 | Recent posts, avg engagement, strategy |
| `query_tag_history` | 1 | Your historical signals per tag |
| `query_audience` | 2 | Segment affinities, active hours |
| `predict_engagement` | 3 | Simulated signals without committing |
| `draft_review` | 3 | Strengths/weaknesses of a plan |
| `query_creator_pool` | 1 | Available collab partners + overlap |
| `propose_collab` | 5 | Propose collaboration (max 2/month) |

API budget starts at 100 per episode.

## Sources & verifiability

Every constant is backed by a Tier 1–3 source. Full bibliography with DOIs, PMIDs, and methodology extracts: **[RESEARCH.md](RESEARCH.md)**.

| Tier | Count | Example |
|------|-------|---------|
| T1 (Peer-reviewed) | 7 papers | Van Dongen 2003, arxiv:2410.13108 |
| T2 (Industry, large-N) | 9 studies | Buffer 9.6M, Sprout 2B, Rival IQ 1.9M |
| T3 (Official) | 1 statement | Mosseri Jan-2025 |
| T4 (Survey) | 2 surveys | Awin 2024 (n=300+) |
| T5 (Rejected) | 13 sites | No methodology disclosed |

## Storytelling assets

- [Full blog — story, science, results](blog/blog.md)
- [HuggingFace mini-blog](blog/hf_mini_blog.md)
- [YouTube script (<2 min)](blog/youtube_script.md)
- [Slide deck outline](blog/slide_outline.md)

## Local development

```bash
git clone <repo-url> && cd viraltest
uv sync

# Terminal 1 — API server
uvicorn viraltest.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — inference
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
.venv/bin/python inference.py
```

## Docker

```bash
docker build -t viraltest-env:latest .
docker run --rm -p 8000:8000 viraltest-env:latest
curl -s -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/reset
```

## Project structure

```
.
├── inference.py                # Tool-discovery agent (no hint keys)
├── openenv.yaml                # OpenEnv manifest
├── models.py                   # Action/Observation + ToolCall, EngagementSignals
├── client.py                   # ViraltestEnv client (async)
├── Dockerfile
├── RESEARCH.md                 # Full sourced bibliography (6+ pages)
├── DESIGN.md                   # Deep design notes
├── blog/
│   ├── hf_mini_blog.md
│   ├── youtube_script.md
│   └── slide_outline.md
├── server/
│   ├── app.py                  # FastAPI + /tools endpoints
│   ├── viraltest_environment.py
│   ├── dashboard.html
│   └── data/
│       ├── tags.json           # ~120 tags, 4 tiers
│       ├── topics.json         # Niche multipliers + seasonal calendar
│       ├── competitors.json    # 7 archetypes
│       ├── hour_heatmap.json   # 7×24 from Buffer+Sprout
│       ├── audience_segments.json
│       └── audience_overlap_matrix.json
├── training/
│   └── train_grpo.ipynb        # TRL GRPO on Qwen2.5-1.5B-Instruct
└── plots/
    ├── reward_curve.png
    └── before_after.png
```

## License

See `LICENSE` in the repository root (BSD-style per upstream OpenEnv examples).
