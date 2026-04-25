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
  - theme-3-world-modeling
  - tool-discovery
  - composable-rubrics
  - creator-economy
---

# Viraltest v2 — World-Modeling RL Environment for Instagram Strategy

> **Theme #3.1 — Professional Tasks (World Modeling)**
> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an LLM agent manages an Instagram creator account over 30 simulated days, discovering the world through tools rather than being told the rules.

## Live links

| Asset | Status / URL |
|---|---|
| Hugging Face Space (env) | _Run `HF_USERNAME=<you> HF_TOKEN=<token> ./scripts/deploy_hf_space.sh` then paste `https://huggingface.co/spaces/<you>/viraltest`_ |
| HF blog post | _Paste [blog/hf_mini_blog.md](blog/hf_mini_blog.md) at https://huggingface.co/new-blog and link the post URL here_ |
| <2-min walkthrough video | _Record `/dashboard` per [blog/youtube_script.md](blog/youtube_script.md), upload to YouTube, link unlisted URL here_ |
| Slide deck | _Convert [blog/slide_outline.md](blog/slide_outline.md) via Marp/Slidev → `blog/slides.pdf` and commit_ |
| Training notebook | [training/train_grpo.ipynb](training/train_grpo.ipynb) |
| Bibliography | [RESEARCH.md](RESEARCH.md) |

## The four PDF questions, answered

1. **Problem.** LLM agents have no benchmark for *operating* a content account: they need to plan multi-day strategies under partial observability, manage a finite "API budget", and recover from energy / sleep-debt setbacks. Existing creator tools optimize one post; none train an agent on the *posting-vs-burnout* tradeoff.
2. **Environment.** A 30-day Instagram simulation. Agent receives sparse observations and must call 8 discoverable tools (`query_trends`, `query_competitor`, `predict_engagement`...) to learn the world. Reward is a Mosseri-aligned engagement signal under research-backed sleep, fatigue, and saturation models.
3. **Results.** Tool-using agent (Gemma E4B) beats a heuristic baseline on all three tasks (`monthly_engage`, `monthly_strategic`, `monthly_competitive`). See [Results](#results) below.
4. **Why it matters.** $250B creator economy ([Goldman Sachs, 2025](https://www.goldmansachs.com/insights/articles/the-creator-economy-could-approach-half-a-trillion-dollars-by-2027)), 73% creator burnout rate ([Awin, 2024](https://www.prweb.com/releases/a-majority-of-content-creators-and-influencers-struggle-with-burnout-as-concerns-for-ai-begin-to-surface-according-to-a-new-awin-group-survey-research-302257152.html)). Calibrated against 10+ Tier 1–3 sources so the dynamics generalize.

## What this teaches the LLM

| Capability | How the environment tests it |
|---|---|
| **Tool discovery & orchestration** | 8 discoverable tools. Agent must call `GET /tools` to learn what's available. |
| **Persistent world model** | 30-day horizon. Multi-episode brand chain carries state across months. |
| **Belief tracking** | `notes` field persists hypotheses day-to-day. Agent must update beliefs from tool results. |
| **Causal reasoning** | `coach_feedback` returns counterfactual delta (your plan vs. heatmap-optimal). `predict_engagement` lets agent test hypotheses before committing. |
| **Partial observability** | Default observation is sparse: energy, followers, reward. Rich data only via tools. |
| **Multi-step workflow** | Per day: discover → query → draft → predict → commit → reply → learn from feedback. |

## Results

Real episodes against the Viraltest env, baseline (heuristic, no tools, no notes) vs agent (Gemma `gemma-4-E4B-it-IQ4_XS` via local llama.cpp, full tool catalog). Reproduce with `scripts/run_baseline_vs_agent.py` then `scripts/make_plots.py`.

![Per-step reward curve](plots/reward_curve.png)
*Per-step env reward over a 30-day episode, baseline vs agent, all three tasks on the same axes.*

![Before/after grader scores](plots/before_after.png)
*Final grader score (0–1) per task. Anti-gaming gates collapse the baseline on `monthly_competitive` because it never diversifies content type.*

![Mosseri signals breakdown](plots/signals_breakdown.png)
*Mean per-step Mosseri-aligned signals (Jan-2025 weights). Agent earns more `watch_time` (0.40w) and `sends_per_reach` (0.30w) by adapting content type to intent.*

## Composable rubrics (PDF page 2)

Per-task scoring is decomposed into four named sub-rubrics; each surfaces both a 0–1 score and an evidence dict on the final observation (`obs.rubric_scores`, `obs.rubric_evidence`).

| Rubric | What it measures | Where the weights are |
|---|---|---|
| `engagement` | Mosseri-weighted total vs theoretical max | `monthly_engage` 0.70 / `monthly_strategic` 0.35 / `monthly_competitive` 0.25 |
| `burnout` | Avg + min energy, sleep debt (Van Dongen 2003) | 0.20 / 0.25 / 0.10 |
| `discovery` | Positive-EV tags + tag exploitation + tool diversity | 0.05 / 0.30 / 0.20 |
| `differentiation` | Content variety + topic uniqueness + growth + outperformance | 0.05 / 0.10 / 0.45 |

**Anti-gaming.** Single-content-type strategies hit a `× 0.3` gate on differentiation, sparse tag sets (`< 5 unique`) hit a `× 0.6` gate, and full burnout (energy ≤ 0) collapses the total to `× 0.3`. Agents that exploit one trick instead of solving the task can't get high scores (PDF page 2).

A worked example. `scripts/anti_gaming_demo.py` runs the worst-case spam strategy — same content_type, same single tag, 14 posts/day. Result on every task: grader collapses to `~0.03`, all four rubrics under `0.25`. That's a `9-18×` drop vs an honest no-tool baseline (`0.30–0.55`).

```text
[gameable] monthly_engage:      grader=0.026  rubrics={engagement:0.042, burnout:0.236, discovery:0.104, differentiation:0.097}
[gameable] monthly_strategic:   grader=0.034  rubrics={engagement:0.042, burnout:0.236, discovery:0.104, differentiation:0.097}
[gameable] monthly_competitive: grader=0.030  rubrics={engagement:0.042, burnout:0.236, discovery:0.104, differentiation:0.097}
```

## Quick start

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
        print(result.observation.rubric_scores)
    finally:
        await env.close()

asyncio.run(main())
```

## Reproduce the plots

```bash
# 1. Bring up an LLM (any OpenAI-compatible endpoint works)
./scripts/smoke_local_llm.sh         # checks http://0.0.0.0:1337/v1/models

# 2. Bring up the env server
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# 3. Run baseline (no LLM, ~3s) and agent (LLM, minutes)
.venv/bin/python scripts/run_baseline_vs_agent.py \
    --baseline-episodes 3 --agent-episodes 1 --max-steps 30

# 4. Render plots
.venv/bin/python scripts/make_plots.py
```

## Secrets & cost control

Configure `inference.py` with `.env` (see [.env.example](.env.example)) or shell exports.

| Variable | Required when... | Default | Read in |
|---|---|---|---|
| `API_BASE_URL` | always | `http://0.0.0.0:1337/v1` | `inference.py:34` |
| `MODEL_NAME` | hosted endpoints | auto-discovered from `/v1/models` for localhost | `inference.py:121` |
| `HF_TOKEN` | non-localhost endpoints | — | `inference.py:128` |
| `OPENAI_API_KEY` / `API_KEY` | alternative to `HF_TOKEN` | — | `inference.py:128` |

**HF Space:** the deployed env hosts the **environment only** — judges run their own agent against `https://<user>-viraltest.hf.space`. The Space itself needs **no LLM secrets**.

**Cost ceiling on the HF Inference Router.** One 30-day episode ≈ 70K tokens.

| Model | $/1M tok (approx) | Per episode | 100 episodes |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | ~$0.05 | ~$0.004 | ~$0.35 |
| `Qwen/Qwen2.5-7B-Instruct` | ~$0.20 | ~$0.014 | ~$1.40 |
| `meta-llama/Llama-3.3-70B-Instruct` | ~$0.70 | ~$0.05 | ~$5.00 |

$30 in HF credits comfortably covers the full submission. **Verify live prices on the model card before any long run** — provider rates vary.

## Simulation mechanics

### Engagement signals (Mosseri Jan-2025)

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

## Tasks (30 steps each)

| Task | Difficulty | Heaviest rubric |
|------|-----------|--------------|
| `monthly_engage` | Easier | engagement (0.70) |
| `monthly_strategic` | Medium | discovery (0.30) + engagement (0.35) |
| `monthly_competitive` | Hard | differentiation (0.45) |

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

API budget starts at 100 per episode. Reserved tool names (`reset`, `step`, `state`, `close`) are unused (PDF page 4).

## Deploy to Hugging Face Spaces

```bash
HF_USERNAME=your-handle HF_TOKEN=hf_xxx ./scripts/deploy_hf_space.sh
SPACE_URL=https://your-handle-viraltest.hf.space ./scripts/verify_space.sh
```

`verify_space.sh` confirms `/tools` returns 8, `POST /reset` returns 200, `POST /step` returns 200, and `GET /state` returns 200.

## Local development

```bash
git clone <repo-url> && cd viraltest
uv sync

# Terminal 1 — env server
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — inference (defaults to local LLM at 0.0.0.0:1337/v1)
cp .env.example .env   # then pick a preset
.venv/bin/python inference.py
```

## Docker

```bash
docker build -t viraltest-env:latest .
docker run --rm -p 8000:8000 viraltest-env:latest
curl -s -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/reset
```

## Sources & verifiability

Every constant is backed by a Tier 1–3 source. Full bibliography: **[RESEARCH.md](RESEARCH.md)**.

| Tier | Count | Example |
|------|-------|---------|
| T1 (Peer-reviewed) | 7 papers | Van Dongen 2003, arxiv:2410.13108 |
| T2 (Industry, large-N) | 9 studies | Buffer 9.6M, Sprout 2B, Rival IQ 1.9M |
| T3 (Official) | 1 statement | Mosseri Jan-2025 |
| T4 (Survey) | 2 surveys | Awin 2024 (n=300+) |

## Project structure

```
.
├── inference.py                  # Tool-discovery agent + collect_episode()
├── openenv.yaml                  # OpenEnv manifest
├── models.py                     # Action/Observation + rubric_scores
├── client.py                     # ViraltestEnv client (async)
├── Dockerfile, LICENSE, .env.example
├── RESEARCH.md                   # Full sourced bibliography
├── DESIGN.md                     # Deep design notes
├── blog/{hf_mini_blog,youtube_script,slide_outline}.md
├── server/
│   ├── app.py                    # FastAPI + /tools endpoints + /dashboard
│   ├── viraltest_environment.py  # Composable rubrics + sim mechanics
│   └── data/                     # tags, topics, competitors, heatmaps, audience
├── scripts/
│   ├── smoke_local_llm.sh        # GET /v1/models sanity
│   ├── run_baseline_vs_agent.py  # writes runs/*.jsonl
│   ├── make_plots.py             # writes plots/*.png from runs/*.jsonl
│   ├── deploy_hf_space.sh        # openenv validate + build + push
│   └── verify_space.sh           # POST /reset → 200, /tools → 8
├── training/train_grpo.ipynb     # TRL GRPO on Qwen2.5-1.5B-Instruct
├── runs/{baseline,agent}.jsonl   # Real episode trajectories
└── plots/{reward_curve,before_after,signals_breakdown}.png
```

## License

[BSD-style](LICENSE) (matches upstream OpenEnv examples).
