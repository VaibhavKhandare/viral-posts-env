# Viraltest v2 — Pitch Deck Outline (8 slides)

## Slide 1: Title
- **Viraltest v2: Teaching LLMs World Modeling Through Instagram Strategy**
- Theme #3.1 — Professional Tasks
- OpenEnv Hackathon India 2026
- Team: [your team name]

## Slide 2: The Problem
- $250B creator economy, 67M creators (Goldman Sachs 2025)
- 73% experience burnout; Instagram drives 88% of it (Awin 2024)
- Algorithm changes constantly — no one tells you the rules
- Existing tools show analytics but don't teach strategy
- **Gap:** No RL environment captures this tradeoff with realistic dynamics

## Slide 3: The World
- 30-day Instagram simulation (monthly cycle)
- Mosseri-aligned signals: watch_time, sends, saves, likes (official Jan 2025)
- Hour-by-hour heatmap (Buffer 9.6M + Sprout 2B)
- 7 competitor archetypes, 5 audience segments, ~120 tags
- Piecewise-linear sleep model (Van Dongen 2003, *Sleep*)
- Tiered audience fatigue (Buffer 2.1M)

## Slide 4: The Tools (Theme #3.1 Fit)
- Agent starts with SPARSE observation (energy, followers, reward)
- 8 discoverable tools: query_trends, query_competitor, query_audience, query_tag_history, predict_engagement, draft_review, query_creator_pool, propose_collab
- API budget (100/episode) — can't query everything, must prioritize
- Notes field for hypothesis tracking across days
- Counterfactual coach: "here's what would have happened with optimal timing"

## Slide 5: Training Pipeline
- TRL GRPO on Qwen2.5-1.5B-Instruct (free Colab T4)
- Reward: per-step env reward + 2× terminal grader score
- 200 episodes, batch 4, 50 GRPO steps
- 3 tasks: monthly_engage → monthly_strategic → monthly_competitive
- Multi-episode chain: brand state persists across months

## Slide 6: Results
- [Embed reward_curve.png — ascending curve over training]
- [Embed before_after.png — smart baseline vs trained agent per task]
- Trained agent: uses tools on day 1, adapts strategy by day 5, manages energy throughout
- Score improvement on monthly_competitive: [X% → Y%]

## Slide 7: Sources & Verifiability
- 4-tier source quality bar (peer-reviewed → industry → official → survey)
- 7 Tier-1 papers, 9 Tier-2 studies, 1 Tier-3 official statement
- Every constant has a DOI/PMID/arXiv ID
- Tier-5 SEO blogs explicitly rejected (13 sites listed with rationale)
- Full bibliography: RESEARCH.md (~6 pages)
- **Any number in this presentation can be debated — we welcome it**

## Slide 8: Try It
- HF Space: [link]
- GitHub: [link]
- Training notebook: [Colab link]
- Blog: [HF post link]
- Video: [YouTube link]
- **Questions?**
