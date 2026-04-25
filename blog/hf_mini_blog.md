# Viraltest v2: Teaching LLMs to Be Instagram Strategists Through World Modeling

**TL;DR:** We built an OpenEnv environment where an LLM agent manages an Instagram creator account for 30 simulated days. The agent receives sparse observations and must discover the world — trending topics, competitor behavior, audience segments, posting heatmaps — through a catalog of 8 tools. Every constant is calibrated against peer-reviewed research and large-N industry studies.

## The Problem

The $250B creator economy (Goldman Sachs, 2025) has 67 million creators, but 73% experience burnout (Awin, 2024). The core tension: post enough to stay visible in the algorithm, but not so much that quality drops and audiences fatigue. No existing RL environment captures this tradeoff with realistic dynamics.

## The Environment

**Viraltest v2** simulates a 30-day Instagram creator lifecycle grounded in 10+ verified data sources:

- **Engagement signals** decomposed into watch_time, sends_per_reach, saves, and likes_per_reach — matching Adam Mosseri's Jan-2025 official ranking signal confirmation
- **Hour-by-hour heatmap** from Buffer's 9.6M-post study cross-validated with Sprout Social's 2B-engagement analysis
- **Sleep/cognitive model** based on Van Dongen et al. (2003, *Sleep*, PMID 12683469) — performance lapses are linear above 16 hours awake
- **Tiered audience fatigue** from Buffer's 2.1M-post frequency study — not a cliff but a gradual decay
- **7 competitor archetypes** with realistic posting cadences (3–5/week, not per-day)

## Theme #3.1: Why This Is World Modeling

The agent starts each day with almost no information — just energy, followers, and last reward. To plan effectively, it must:

1. **Discover tools** (`GET /tools`) on day 1
2. **Query the world** — trending topics, competitor activity, audience preferences
3. **Form hypotheses** and persist them in a scratchpad (`notes` field)
4. **Test plans** via `predict_engagement` before committing
5. **Learn from counterfactual feedback** — the environment shadow-runs the optimal heatmap plan and shows the delta

This isn't prompt engineering. The agent must build and maintain an internal world model across 30 steps.

## Composable rubrics (PDF page 2 explicit ask)

The grader is split into four named sub-rubrics — `engagement`, `burnout`, `discovery`, `differentiation` — each surfaced on the observation as `obs.rubric_scores` and `obs.rubric_evidence`. Task-specific weights mean `monthly_competitive` puts 0.45 of the score on differentiation; `monthly_engage` puts 0.70 on raw engagement. Anti-gaming gates collapse single-content-type or sparse-tag strategies so a spam agent can't farm the score.

## Training

`scripts/run_baseline_vs_agent.py` collects real episodes via `inference.py::collect_episode` against the env. Local llama.cpp Gemma serves rollouts at zero token cost; HF Inference Router takes over for the showcase run on `Qwen/Qwen2.5-1.5B-Instruct`. `training/train_grpo.ipynb` wires GRPO weight updates behind a `RUN_REAL_GRPO=1` flag for Colab T4. Plots ship from `scripts/make_plots.py` so README and notebook stay in sync.

## Every Number Is Verifiable

Sources classified into 4 tiers (peer-reviewed → industry → official → survey); SEO/affiliate blogs explicitly rejected. Full bibliography with DOIs, PMIDs, arXiv IDs, methodology extracts, and sample sizes lives in [RESEARCH.md](../RESEARCH.md).

[Environment on HF Spaces](https://huggingface.co/spaces/<your-handle>/viraltest) · [Training notebook](../training/train_grpo.ipynb) · [Bibliography](../RESEARCH.md) · _replace links once `./scripts/deploy_hf_space.sh` runs._
