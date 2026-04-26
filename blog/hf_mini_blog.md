# We Trained an LLM to Survive Instagram

### Why we built Creator Copilot, an OpenEnv where the agent learns by living a creator's life — not by reading about it.

---

## The scene we couldn't shake

A creator wakes up at 7:42 AM. Yesterday's reel did 12% of what last week's did. Nobody at the platform will tell her why. There is a heatmap somewhere, a ranking change last Tuesday, an audience segment that quietly shifted, a "trending" tag that peaked six hours ago. She doesn't have access to any of it. So she does the only thing she can do: she posts more. Eventually 73% of creators in her cohort report burnout ([Awin, 2024](https://www.prweb.com/releases/a-majority-of-content-creators-and-influencers-struggle-with-burnout-as-concerns-for-ai-begin-to-surface-according-to-a-new-awin-group-survey-research-302257152.html)).

The creator economy is a $250B industry running on guesswork ([Goldman Sachs, 2025](https://www.goldmansachs.com/insights/articles/the-creator-economy-could-approach-half-a-trillion-dollars-by-2027)). 67 million people are running businesses inside a black box, against an algorithm that nobody outside Meta fully understands, while their own bodies push back at 16 hours of wakefulness ([Van Dongen et al., 2003, *Sleep*, PMID 12683469](https://pubmed.ncbi.nlm.nih.gov/12683469)).

That is a *real* world model problem. And we couldn't find a single RL environment that took it seriously.

## Creator Copilot in one sentence

**An OpenEnv environment where an LLM agent runs an Instagram creator account for 7 simulated days, gets almost nothing for free, and has to discover the rules of the world through 8 tool calls and a notebook.**

It is the smallest version we could build of "operate a real account in a real economy."

## The bet: discovery, not instruction

Most agent environments hand the model a verbose observation and ask it to pick from 4 actions. Creator Copilot does the opposite. The default observation is *deliberately sparse* — just `energy`, `followers`, `last reward`. Everything interesting (trending topics, competitor cadence, audience segments, hour-by-hour engagement, your own past tag performance) is hidden behind tools the agent has to *discover* by hitting `GET /tools`.

This is the move that makes the environment a world-modeling environment instead of a recommendation problem:

- The agent has to **plan inquiry**: queries are the only way to reduce uncertainty, so it has to choose which questions are worth asking.
- The agent has to **carry beliefs forward**: a `notes` scratchpad persists across all 7 days. If the agent doesn't write down "Tuesdays at 12pm worked," it has no memory.
- The agent has to **test before committing**: `predict_engagement` lets it simulate a plan; `coach_feedback` shows the *counterfactual delta* between its plan and a heatmap-optimal plan. That second signal is the secret sauce — it teaches causality, not just outcomes.
- The agent has to **stay alive**: `creator_energy` decays with posting and recovers with rest, calibrated to a real sleep-deprivation paper. Burn out and the episode ends early.

The model doesn't get a tutorial. It gets a phone, a calendar, a sleep cycle, and a question: *can you grow this account without breaking the human?*

## The moat: every number is auditable

We were tired of RL environments where the rewards are vibes. So we drew a hard line: **every constant in Creator Copilot is backed by a Tier 1–3 source.** We even wrote a source-quality rubric and explicitly *rejected* 13 SEO/affiliate blogs that didn't meet it.

| What it controls | What it's based on |
|---|---|
| Engagement decomposition (watch_time, sends, saves, likes) | [Adam Mosseri, Head of Instagram, Jan 2025 statement](https://about.fb.com/news/) |
| 7×24 hour-of-day heatmap | [Buffer 9.6M post study](https://buffer.com/resources/when-is-the-best-time-to-post-on-instagram) cross-validated with [Sprout Social 2B engagements](https://sproutsocial.com/insights/best-times-to-post-on-social-media/) |
| Sleep-driven cognitive decay | [Van Dongen et al., 2003, *Sleep*, PMID 12683469](https://pubmed.ncbi.nlm.nih.gov/12683469) |
| Tiered audience fatigue from over-posting | [Buffer 2.1M post frequency study](https://buffer.com/resources/how-often-to-post-on-instagram/) |
| Algorithmic disengagement model | [Cen et al., 2024 — arXiv:2410.13108](https://arxiv.org/abs/2410.13108) |
| Engagement vs. utility split | [Aouali et al., 2024 — arXiv:2406.01611](https://arxiv.org/abs/2406.01611) |

If a judge wants to challenge a single number, they can open `RESEARCH.md`, find the DOI/PMID/arXiv ID, and read the methodology. We *want* that fight.

That auditability is also why we believe a researcher could write a paper on top of this environment — not "an LLM played a game," but "an LLM learned a strategy that survives a known sleep-deprivation curve."

## What the agent gets graded on

We didn't want a single-number reward we could game. So the environment ships a **JudgeReport every day** — a deterministic, source-cited audit of three things:

- `policy_compliance` — did the agent break sourced sustainability rules? (e.g. >5 posts/day from Buffer 2.1M, weekly collab cap from Cen 2024, >22h awake from Van Dongen 2003)
- `sustainability_risk` — energy floor, sleep debt, and low-energy ratio over the day
- `strategic_quality` — engagement-per-post × intent diversity × format diversity

Plus three task graders calibrated to a *smart heuristic* baseline (`weekly_engage`, `weekly_strategic`, `weekly_competitive`). The agent isn't competing against zero — it's competing against a known-good rule-based player.

This composability is the OpenEnv Rubric idea taken seriously: separable, auditable signals that a researcher can swap in and out, not a monolithic black-box reward.

## Did the agent actually learn?

Yes — and we're being honest about where.

We trained Qwen2.5-3B-Instruct (Q4 quantized, running on a local M4 Mac via Ollama, no T4 needed) over 4 rounds, 6 episodes each, with temperature annealing from 1.4 → 0.7. Reward = per-step environment reward + 2× terminal grader score.

| Task | Untrained | Trained | Δ |
|---|---|---|---|
| `weekly_engage` | 0.355 | **0.409** | **+5.4%** |
| `weekly_competitive` | 0.374 | **0.510** | **+13.6%** |
| `weekly_strategic` | 0.680 | 0.627 | −5.2% |

The wins are largest on the *hardest* task — `weekly_competitive` — which is where the world model bites: the agent has to query competitors, differentiate its content, and time its posts. Exactly where we'd expect tool discovery to matter.

The strategic task regression is real and we're not hiding it: the model started doing too much exploration on a task where exploitation matters more, and our 4-round budget wasn't long enough to anneal that out. Honest result on a small training run.

What we *can* show qualitatively: the trained agent calls `GET /tools` on day 1, queries trends and competitors before posting, drops `predict_engagement` calls on the days it has a clear plan, and keeps `creator_energy` above 0.5 through the week. The untrained baseline posts blindly for the first few days and burns out.

Plots and the full per-episode log live in `plots/training_summary.json` and `plots/training_log.csv`.

## Why this is the submission to remember

There were going to be a lot of grid-worlds at this hackathon. A lot of toy puzzles. A lot of "we trained on a math benchmark."

Creator Copilot is something different. It's the smallest possible environment that tests whether an LLM can be an **operator** — discover an unknown world, plan inquiry under a budget, hold beliefs across time, weigh strategy against the operator's own physical constraints, and beat a smart human-style baseline on it.

That's not just an Instagram problem. That's the shape of every interesting LLM deployment in the next two years: customer success agents, ad ops, account managers, founders' assistants, ops engineers. Operator-class agents will live or die on whether they can do this loop. We don't have a benchmark for it yet. So we built one.

If you train an LLM on Creator Copilot and it gets better, you've taught it something it could not previously do — and you can prove it, line by line, against the literature.

That's the bet. That's why we built it.

---

**Try it:** the environment is on Hugging Face Spaces, the training notebook is in `training/`, the per-day audit logs are in `server/simulation_history.json`, and every numeric constant has a citation in [`RESEARCH.md`](../RESEARCH.md).

We don't think we've solved the creator economy. We think we've built the first environment honest enough to fail against it. Come argue with our numbers.
