# We Trained an LLM to Survive on Instagram

### Why we built Creator Copilot, an OpenEnv where the agent learns by living a creator's life — not by reading about it.

---

## The scene we couldn't shake

A creator wakes up at 7:42 AM. Yesterday's reel did 12% of what last week's did. Nobody at the platform will tell her why. There is a heatmap for this scenario, a ranking change last Tuesday, an audience segment that quietly shifted, a "trending" tag that peaked six hours ago. She doesn't have access to any of it. So she does the only thing she can do: she posts more. Eventually 73% of creators in her cohort report burnout ([Awin, 2024](https://www.prweb.com/releases/a-majority-of-content-creators-and-influencers-struggle-with-burnout-as-concerns-for-ai-begin-to-surface-according-to-a-new-awin-group-survey-research-302257152.html)).

The creator economy is a $250B industry running on guesswork ([Goldman Sachs, 2025](https://www.goldmansachs.com/insights/articles/the-creator-economy-could-approach-half-a-trillion-dollars-by-2027)). 67 million people are running businesses inside a black box, against an algorithm that nobody outside Meta fully understands, while their own bodies push back at 16 hours of wakefulness ([Van Dongen et al., 2003, *Sleep*, PMID 12683469](https://pubmed.ncbi.nlm.nih.gov/12683469)).

That is a *real* world model problem. So we built an OpenEnv to simulate this and help an LLM train on it.

## Creator Copilot in one sentence

**An OpenEnv environment where an LLM agent runs an Instagram creator account for 7 simulated days, fetches all required data, and has to discover soft rules of the world through 8 tool calls and a notebook to know which strategy leads to the best engagement and follower growth.**

It is the mocked, research-paper version of what we could build for "operate a real account in a real economy."

## The bet: discovery, not instruction

Most agent environments hand the model a verbose observation and ask it to pick from 4 actions. Creator Copilot takes this to the ultimate level. The default observation is *deliberately sparse* — just `energy`, `followers`, `last reward`. Everything interesting (trending topics, competitor cadence, audience segments, hour-by-hour engagement, your own past tag performance) is hidden behind tools the agent has to *discover* by hitting `GET /tools`.

This is the move that makes the environment a world-modeling environment instead of a recommendation problem:

- The agent has to **plan inquiry**: queries are the only way to reduce uncertainty, so it has to choose which questions are worth asking.
- The agent has to **carry beliefs forward**: a `notes` scratchpad persists across all 7 days. If the agent doesn't write down "Tuesdays at 12pm worked," it has no memory.
- The agent has to **test before committing**: `predict_engagement` lets it simulate a plan; `coach_feedback` shows the *counterfactual delta* between its plan and a heatmap-optimal plan. That second signal is the secret sauce — it teaches causality, not just outcomes.
- The agent has to **stay alive**: `creator_energy` decays with posting and recovers with rest, calibrated to a real sleep-deprivation paper. Burn out and the episode ends early.

The model doesn't get a tutorial. It gets a phone, a calendar, a sleep cycle, and a question: *can you grow this account without breaking the human?*

## The moat: every number is auditable

**Every constant in Creator Copilot is backed by a Tier 1–3 source.** 


| What it controls                                           | What it's based on                                                                                                                                                                                                           |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Engagement decomposition (watch_time, sends, saves, likes) | [Adam Mosseri, Head of Instagram, Jan 2025 statement](https://about.fb.com/news/)                                                                                                                                            |
| 7×24 hour-of-day heatmap                                   | [Buffer 9.6M post study](https://buffer.com/resources/when-is-the-best-time-to-post-on-instagram) cross-validated with [Sprout Social 2B engagements](https://sproutsocial.com/insights/best-times-to-post-on-social-media/) |
| Sleep-driven cognitive decay                               | [Van Dongen et al., 2003, *Sleep*, PMID 12683469](https://pubmed.ncbi.nlm.nih.gov/12683469)                                                                                                                                  |
| Tiered audience fatigue from over-posting                  | [Buffer 2.1M post frequency study](https://buffer.com/resources/how-often-to-post-on-instagram/)                                                                                                                             |
| Algorithmic disengagement model                            | [Cen et al., 2024 — arXiv:2410.13108](https://arxiv.org/abs/2410.13108)                                                                                                                                                      |
| Engagement vs. utility split                               | [Aouali et al., 2024 — arXiv:2406.01611](https://arxiv.org/abs/2406.01611)                                                                                                                                                   |


## What the agent gets rewarded on

The reward system is intentionally split into **daily learning signal + end-of-week judgment**. We did not want the model to chase engagement by burning the creator out, so every score balances growth with sustainability. ⚖️

Each simulated day produces a reward in `[0, 1]`:

- `engagement` 🎯 — normalized engagement from posts, capped so one viral post cannot dominate the whole episode.
- `energy` 🔋 — reward for preserving or recovering creator energy instead of posting through exhaustion.
- `consistency` 📅 — highest when the agent lands in the healthy 1-2 posts/day range.
- `trend fit` — bonus for using tags/topics that match the current trend state.
- `competitor differentiation` 🧭 — bonus for posting something distinct from competitor topics.
- `audience interaction` — small positive/negative shaping for replies, comment quality, off-niche behavior, spam, and ignoring the audience.
- `burnout penalty` — subtracts reward when energy drops below the danger zone.

In the default `combined` mode, a post is scored roughly as:

`reward = engagement(30%) + energy(15%) + consistency(15%) + trend/tag fit(15%) + competitor differentiation(15%)`, with the engagement/tag/differentiation side boosted by trending-topic and peak-hour multipliers, then burnout penalties applied.

The daily observation reward is the average of all 24 hourly rewards plus the interaction reward, capped back into `[0, 1]`.

Resting is also rewarded. If the agent rests after draining energy, it can still earn a smaller reward through energy recovery and consistency. That matters because the best policy is not "post every hour"; it is "post when the world is favorable and recover when the human needs it."

Every day also ships a **JudgeReport** — a deterministic, source-cited audit:

- `policy_compliance` — starts at `1.0` and drops for violations like >5 posts/day, too many weekly posts, >3 collabs/week, invalid plans, >22h awake, spammy interactions, or forced bad collabs.
- `sustainability_risk` — combines minimum energy, sleep debt, and repeated low-energy days.
- `strategic_quality` — combines engagement per post, intent diversity, and format diversity.

At the end of the 7-day episode, a task grader scores the whole strategy:

- `weekly_engage` — total engagement versus a theoretical max, with a harsh penalty if the creator burns out.
- `weekly_strategic` — engagement, tag discovery/exploitation, average energy, and healthy posting consistency.
- `weekly_competitive` — engagement, tag strategy, follower growth, competitor outperformance, topic differentiation, and energy floor.

**Example:** suppose the agent posts one reel at a peak hour with trending tags, gets strong engagement, keeps energy above `0.5`, and avoids competitor overlap. The day gets a strong reward because engagement, timing, trend fit, and differentiation all line up. If it then posts six more times, the JudgeReport flags policy violations, sustainability risk rises, and the weekly grader punishes the strategy even if raw engagement goes up. The agent learns the grown-up creator lesson: growth only counts when it is repeatable.

## Did the agent actually learn?

Yes — and we're being honest about where.

We trained Qwen2.5-3B-Instruct (Q4 quantized, running on a local M4 Mac via Ollama, no T4 needed) over 4 rounds, 6 episodes each, with temperature annealing from 1.4 → 0.7. Reward = per-step environment reward + 2× terminal grader score.


| Task                 | Untrained | Trained   | Δ          |
| -------------------- | --------- | --------- | ---------- |
| `weekly_engage`      | 0.355     | **0.409** | **+5.4%**  |
| `weekly_competitive` | 0.374     | **0.510** | **+13.6%** |
| `weekly_strategic`   | 0.680     | 0.627     | −5.2%      |


The wins are largest on the *hardest* task — `weekly_competitive` — which is where the world model bites: the agent has to query competitors, differentiate its content, and time its posts. Exactly where we'd expect tool discovery to matter.

The strategic task regression is real and we're not hiding it: the model started doing too much exploration on a task where exploitation matters more, and our 4-round budget wasn't long enough to anneal that out. Honest result on a small training run.

What we *can* show qualitatively: the trained agent calls `GET /tools` on day 1, queries trends and competitors before posting, drops `predict_engagement` calls on the days it has a clear plan, and keeps `creator_energy` above 0.5 through the week. The untrained baseline posts blindly for the first few days and burns out.

Plots and the full per-episode log live in `plots/training_summary.json` and `plots/training_log.csv`.

## Where we're honest about shortcomings

A research-quality environment has to admit what is mocked, what is real, and what still needs sharper guardrails. Here is the unvarnished list:

| Concern | Status today | Why / plan |
| --- | --- | --- |
| Negative comments and sentiment hits | Not implemented — comments mostly help engagement today | Real posts can go viral for the wrong reasons. Future update: add a `comment_sentiment` channel where mass negative comments suppress reach, mirroring Cen 2024's disengagement model. |
| Troll-driven virality | Not implemented | The agent should not learn that outrage is a growth hack. Future update: separate positive comments, negative comments, likes, and dislikes so hostile engagement does not get rewarded like healthy engagement. |
| Followers always grow if you post | Currently true | This is the biggest "video game" assumption. In reality, a tone-deaf post can lose followers. Future update: introduce `follower_loss_rate` driven by content-audience mismatch, dislikes, and negative sentiment. |
| Abusive or hateful content guardrails | Not implemented | Detecting toxicity reliably needs an LLM-in-the-loop, similar to Llama Guard. Future update: add an optional moderation hook that blocks abusive suggestions, downgrades reach, and adds a policy violation to `JudgeReport`. |
| Negative post recommendations | Prevented indirectly, not explicitly | The current reward favors sustainability and strategy, but it does not yet understand "this content is hateful." Future update: make unsafe, abusive, or hatred-based posts invalid actions rather than merely bad strategies. 🛡️ |
| Sponsorship offers | Mocked: deterministic schedule per archetype | Real sponsorships depend on niche, follower count, recency, and engagement quality. We have the building blocks, just not the marketplace yet. |
| Collaborator follower counts | Mocked from `audience_overlap_matrix.json` | Real follower numbers are noisy and platform-API-gated. The mock distribution matches Rival IQ-style industry medians, so collab reasoning is calibrated but not personalized. |
| Hour heatmap, fatigue tiers, sleep curve, niche multipliers, format reach | Real — backed by the studies above | These are the load-bearing numbers, and they are sourced. |

We list this openly because the next version should be harder on the agent: no abusive content, no hate-bait recommendations, no troll comments counted as healthy success, and no pretending that all attention is good attention. The goal is not just a viral agent. It is a creator copilot that knows when *not* to post.

## Why this is the submission to remember

Creator Copilot is the smallest possible environment that tests whether an LLM can be an **operator** — discover an unknown world, plan inquiry, hold beliefs across time, weigh strategy against the operator's own physical constraints, and be like a professional content growth expert.

That's not just an Instagram problem. That's the shape of every interesting LLM deployment in the next two years: calorie tracker, personal coach, productivity helper.

---

**Try it:** the environment is on Hugging Face Spaces, the training notebook is in `training/`, the per-day audit logs are in `server/simulation_history.json`, and every numeric constant has a citation in `[RESEARCH.md](../RESEARCH.md)`.

We don't think we've solved the creator economy. We think we've built the first environment honest enough to fail against it. Come argue with our numbers.