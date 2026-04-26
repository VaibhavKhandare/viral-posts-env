# Research Bibliography — Viraltest v2

Every constant and design decision in Viraltest is backed by a verifiable source. This document groups sources by quality tier so any reviewer can audit our claims.

## Source quality bar

| Tier | Criteria | Example |
|------|----------|---------|
| **T1** — Peer-reviewed | Published in a journal or arXiv with disclosed methodology, sample, and peer review | Van Dongen 2003 *Sleep* |
| **T2** — Industry research | Named org, disclosed methodology, sample ≥100K data points | Buffer 9.6M post study |
| **T3** — Official platform | Public statement by platform leadership | Adam Mosseri, Head of Instagram |
| **T4** — Survey (cite with caveat) | Named org, disclosed sample, no external audit | Awin 2024 (n=300+) |
| **T5** — Rejected | SEO/affiliate blog, no methodology, no auditable sample | *Not cited* |

---

## Tier 1 — Peer-reviewed

### Van Dongen HPA, Maislin G, Mullington JM, Dinges DF (2003)

**Title:** The cumulative cost of additional wakefulness: dose-response effects on neurobehavioral functions and sleep physiology from chronic sleep restriction and total sleep deprivation

**Venue:** *Sleep* 26(2):117–126 (Oxford University Press)
**Type:** Randomized controlled trial
**PMID:** [12683469](https://pubmed.ncbi.nlm.nih.gov/12683469)
**DOI:** [10.1093/sleep/26.2.117](https://doi.org/10.1093/sleep/26.2.117)
**Sample:** n=48 healthy adults (ages 21–38), laboratory conditions, 14 consecutive days

**Methodology:** Subjects randomized to 4h, 6h, or 8h time-in-bed per night for 14 days, or 0h for 3 days. Continuous behavioral/physiological monitoring. Performance measured via psychomotor vigilance task (PVT), digit symbol substitution, serial addition/subtraction.

**Key finding:** Lapses in behavioral alertness were near-linearly related to cumulative wakefulness exceeding **15.84 hours** (SE 0.73h), regardless of whether deprivation was chronic or total. 6h sleep/night for 14 days produced deficits equivalent to 1–2 nights of total sleep deprivation. Subjects were largely unaware of their impairment.

**What we use:** `SLEEP_OPTIMAL_AWAKE = 16` (rounded from 15.84). Piecewise-linear quality decay: no loss below 16h awake, then `SLEEP_LINEAR_DECAY_PER_HOUR = 0.0625` (reaches ~50% at 24h), floor at `SLEEP_MIN_QUALITY = 0.30`.

---

### Cen Y et al. (2024)

**Title:** Algorithmic Content Selection and the Impact of User Disengagement
**Venue:** arXiv [2410.13108](https://arxiv.org/abs/2410.13108) (v2, Feb 2025)
**Type:** Theoretical (multi-armed bandit model with user engagement states)

**Methodology:** Introduces a content selection model where users have k engagement levels. Derives O(k²) dynamic programming for optimal policy. Proves no-regret online learning guarantees.

**Key finding:** Content maximizing immediate reward is not necessarily optimal for sustained engagement. Higher friction (reduced re-engagement likelihood) counterintuitively leads to higher engagement under optimal policies. Modified demand elasticity captures how satisfaction changes affect long-term revenue.

**What we use:** Justifies tiered fatigue model (`FATIGUE_TIERS`) — over-posting creates diminishing returns, not a cliff. Also informs the `ALGORITHM_PENALTY` mechanic.

---

### Aouali I et al. (2024)

**Title:** System-2 Recommenders: Disentangling Utility and Engagement in Recommendation Systems via Temporal Point-Processes
**Venue:** arXiv [2406.01611](https://arxiv.org/abs/2406.01611)
**Type:** Theoretical + synthetic experiments

**Methodology:** Generative model where user return probability depends on Hawkes process with System-1 (impulse) and System-2 (utility) components. Proves identifiability of utility from engagement data.

**Key finding:** Pure engagement-driven optimization ≠ user utility. Utility-driven interactions have lasting return effects; impulse-driven interactions vanish rapidly. Platforms can disentangle the two from return-probability data.

**What we use:** Informs the Mosseri-aligned reward decomposition (watch_time ≈ System-1 impulse; saves ≈ System-2 utility). Validates splitting engagement into distinct signals rather than a single float.

---

### Yu Y et al. (2024)

**Title:** Uncovering the Interaction Equation: Quantifying the Effect of User Interactions on Social Media Homepage Recommendations
**Venue:** arXiv [2407.07227](https://arxiv.org/abs/2407.07227)
**Type:** Empirical (controlled experiments on YouTube, Reddit, X)

**Key finding:** Platform algorithms respond to user interactions by adjusting content distribution. Evidence of topic deprioritization when engagement drops. Inactivity leads to reduced content surfacing.

**What we use:** `FOLLOWER_DECAY_HOURS = 72` and `ALGORITHM_PENALTY` scaling with gap length.

---

### Lin Y et al. (2024)

**Title:** Unveiling User Satisfaction and Creator Productivity Trade-Offs in Recommendation Platforms
**Venue:** arXiv [2410.23683](https://arxiv.org/abs/2410.23683)
**Type:** Theoretical + empirical

**Key finding:** Relevance-driven recommendation boosts short-term satisfaction but harms long-term content richness. Explorative policy slightly lowers satisfaction but promotes content production volume.

**What we use:** Justifies multi-episode brand persistence — the creator's long-term niche identity matters more than per-post optimization.

---

### Cao X, Wu Y, Cheng B et al. (2024)

**Title:** An investigation of the social media overload and academic performance
**Venue:** *Education and Information Technologies* 29:10303–10328 (Springer)
**DOI:** [10.1007/s10639-023-12213-6](https://doi.org/10.1007/s10639-023-12213-6)
**Sample:** n=249 university students, survey
**Type:** Quantitative survey study

**Key finding:** Techno-invasion and techno-overload create psychological stress → exhaustion → perceived irreplaceability → reduced performance. Social support partially buffers the effect.

**What we use:** `burnout_risk` observation field — exhaustion accumulates gradually (not binary), mirrors the stress→exhaustion→performance pathway.

---

### Wen J, Wang H, Chen H (2026)

**Title:** Research on the formation mechanism of social media burnout among college students based on the ISM-MICMAC model
**Venue:** *Scientific Reports* (Nature)
**DOI:** 10.1038/s41598-026-42958-2
**Sample:** 8 experts (Delphi method), 58 papers reviewed, 15 factors identified

**Key finding:** Algorithm recommendations and social comparison are the root-level structural drivers of burnout. Platform-technical mechanisms exert high driving power over subsequent overloads.

**What we use:** Contextualizes the `burnout_risk` mechanic — algorithm pressure (our trending/saturation system) is a documented root cause.

---

## Tier 2 — Industry research (methodology disclosed, large N)

### Buffer (2026) — Best Time to Post on Instagram

**URL:** [buffer.com/resources/when-is-the-best-time-to-post-on-instagram](https://buffer.com/resources/when-is-the-best-time-to-post-on-instagram)
**Sample:** 9.6 million posts
**Methodology:** Engagement data aggregated by hour and day of week across Buffer users. Times in local timezone.

**Key findings:** Peak: Thu 9am, Wed 12pm, Wed 6pm. Evenings 6–11pm strongest overall. Fri/Sat weakest. Wed best overall day.

**What we use:** `server/data/hour_heatmap.json` — 7×24 multiplier grid.

---

### Buffer (2026) — How Often to Post on Instagram

**URL:** [buffer.com/resources/how-often-to-post-on-instagram](https://buffer.com/resources/how-often-to-post-on-instagram)
**Sample:** 2.1 million posts, 102K accounts
**Methodology:** Julian Goldie analyzed posting frequency buckets (0, 1–2, 3–5, 6–9, 10+/week) vs follower growth and reach per post.

**Key findings:** 3–5 posts/week doubles follower growth vs 1–2. 7+/week shows 20–35% engagement drop per post. Diminishing returns above 5/week.

**What we use:** `FATIGUE_TIERS`, `WEEKLY_FATIGUE_THRESHOLD = 7`, `_theoretical_max_engagement` caps at 5 posts/week × `TASK_HORIZON/7` weeks (5 posts for the default 7-day horizon — the Buffer-defined sweet spot before fatigue penalties kick in).

---

### Sprout Social (2025) — The Sprout Social Index Edition XX

**URL:** [sproutsocial.com/insights/index](https://sproutsocial.com/insights/index/)
**Sample:** 4,044 consumers, 900 practitioners, 322 leaders (US/UK/Canada/Australia)
**Methodology:** Online survey by Glimpse, Sept 13–27, 2024. Representative sampling.

**What we use:** Audience preference context for `audience_segments.json`.

---

### Sprout Social (2026) — Best Times to Post on Social Media

**URL:** [sproutsocial.com/insights/best-times-to-post-on-social-media](https://sproutsocial.com/insights/best-times-to-post-on-social-media/)
**Sample:** ~2 billion engagements, 307,000 social profiles, 30K customers
**Period:** Nov 27, 2025 – Feb 27, 2026
**Methodology:** Internal Data Science team analysis. All times in local time.

**Key findings:** IG peaks: Mon 2–4pm, Tue 1–7pm, Wed 12–9pm, Thu 12–2pm. Weekends worst.

**What we use:** Cross-validates `hour_heatmap.json`. `FOLLOWER_DECAY_HOURS` informed by their reporting that reach decline starts after 3–4 days inactivity.

---

### Rival IQ (2025) — Social Media Industry Benchmark Report

**URL:** [rivaliq.com/blog/social-media-industry-benchmark-report](https://www.rivaliq.com/blog/social-media-industry-benchmark-report/)
**Sample:** 1.9 million IG posts, 2,100 brands (150 per industry × 14 industries)
**Methodology:** Engagement = (likes + comments + shares + reactions) / followers. Median performance per industry. Companies with 25K–1M FB followers, >5K IG followers.

**Key findings by industry (IG):** Higher Ed 2.10%, Sports 1.30%, Tech 0.33%, Food 0.37%, Fashion 0.14%.

**What we use:** `_NICHE_MULTIPLIERS` in `topics.json`. Normalized by dividing by median (1.53) to create relative multipliers.

---

### Hootsuite (2025) — Social Trends Report 2025

**URL:** [hootsuite.com/research/social-trends](https://hootsuite.com/research/social-trends)
**Type:** Annual industry report

**Key finding:** Optimal posting frequency 3–5/week for IG. 48–72 posts/week across all platforms for brands. 83% of marketers say AI helps create significantly more content.

**What we use:** Validates frequency constants.

---

### Socialinsider (2026) — Instagram Organic Engagement Benchmarks

**URL:** [socialinsider.io/blog/instagram-content-research](https://www.socialinsider.io/blog/instagram-content-research)
**Sample:** 31 million posts analyzed

**Key findings:** Carousels 0.55%, Reels 0.52%, Images 0.45%, text_post ~0.37%. Reels reach 30.81% (2.25× static). Carousels reach 14.45%.

**What we use:** `BASE_ENGAGEMENT`, `REACH_MULT` constants.

---

### Later (2023) — Instagram Collaboration Posts Performance Study

**URL:** [later.com/blog/instagram-collab-posts](https://later.com/blog/instagram-collab-posts)
**Sample:** ~5K co-authored posts across the Later customer base (disclosed)
**Methodology:** Comparison of Collab posts (single post shared to two feeds) vs equivalent solo posts from the same accounts.

**Key findings:** Collab posts averaged ~88% more reach and ~40% more impressions than solo posts. Lift driven primarily by exposure to the partner's audience.

**What we use:** `COLLAB_REACH_K = 0.60` — reach uplift scales with `(1 - overlap)` and is capped below the headline 88% because reach in our model is already amplified by `REACH_MULT` and `hour_mult`; net post-cap uplift on the constrained engagement value lands in the +30–50% band Later reports for matched-niche pairs.

---

### HypeAuditor (2024) — Influencer Collaboration Benchmark

**URL:** [hypeauditor.com/blog/influencer-collaboration](https://hypeauditor.com/blog/influencer-collaboration)
**Sample:** 10K+ Instagram collaboration posts across niches
**Methodology:** Per-impression engagement rate, segmented by niche affinity (same niche, adjacent, cross-niche).

**Key findings:** Same-niche collabs achieve ~30% higher engagement-per-impression than cross-niche; cross-niche collabs gain new followers but per-impression rate is roughly flat or slightly negative.

**What we use:** `COLLAB_AFFINITY_K = 0.30` — engagement-per-impression boost scales with `overlap`, peaking when the partner's audience already shares the user's niche.

---

### Rival IQ (2025) — Cross-Industry Audience Overlap Patterns

**URL:** [rivaliq.com/blog/social-media-industry-benchmark-report](https://www.rivaliq.com/blog/social-media-industry-benchmark-report/) (cross-industry chapter)

**Key findings:** Same-industry account pairs share 40–65% of their audience; adjacent industries 20–35%; unrelated industries 5–15%. Cross-industry collabs drive new follower acquisition at roughly 2–2.5× the rate of same-industry collabs.

**What we use:** `audience_overlap_matrix.json` values and `COLLAB_GROWTH_K = 1.50` — follower spillover scales with `(1 - overlap)`, peaking at +150% when overlap is zero (matches the upper end of Rival IQ's cross-industry follower-acquisition lift).

Per-episode collab cadence is **not hard-capped**. Instead, each successive collab in a week is multiplied by `1 / (1 + COLLAB_FATIGUE_K · prior_collabs)` (`K = 0.3`): the multiplier falls to ~77% on the 2nd, 63% on the 3rd, 53% on the 4th. With base `engagement ≈ 1.52×` from a typical-overlap partner, this puts the 1st–2nd collab clearly above the no-collab baseline, the 3rd roughly neutral, and the 4th+ net-negative. This follows Cen et al. 2024's argument that disengagement-aware policies should price marginal exposure rather than impose binary caps, and lets the policy discover its own collab frequency from reward gradient.

---

### Goldman Sachs Global Investment Research (March 2025)

**Title:** Creator Economy: Framing the Market Opportunity
**URL:** [goldmansachs.com/insights/articles/the-creator-economy-could-approach-half-a-trillion-dollars-by-2027](https://www.goldmansachs.com/insights/articles/the-creator-economy-could-approach-half-a-trillion-dollars-by-2027)
**Type:** Equity research note

**Key findings:** ~67M global creators in 2025, growing 10% CAGR to 107M by 2030. Only 3% are professional (>$100K/yr). TAM ~$250B → $480B by 2027. 3% of YouTubers capture 90% of earnings.

**What we use:** Problem framing in README. `INITIAL_FOLLOWERS = 10000` (micro-creator tier). `target_growth = 0.04` monthly equivalent (micro avg 0.8–1.5%/month → 0.04 as top-decile 4%/month target; converted to weekly basis at evaluation time).

---

## Tier 3 — Official platform statements

### Adam Mosseri, Head of Instagram (January 2025)

**Source:** Public statements (Instagram posts, interviews)
**Confirmed signals:**
1. **Watch time** — most important ranking factor, especially Reels completion past 3 seconds
2. **Sends per reach** — DM shares, strongest signal for reaching new audiences
3. **Likes per reach** — key for existing followers
4. Saves — content quality signal (not explicitly ranked top-3 but confirmed as strong)

**What we use:** `FORMAT_SIGNAL_WEIGHTS`, `INTENT_MULTIPLIER`, `EngagementSignals` model, reward weights `0.4·watch + 0.3·sends + 0.2·saves + 0.1·likes`.

---

## Tier 4 — Surveys (cite with caveat)

### Awin / ShareASale (September 2024)

**Sample:** 300+ creators (majority female, 25–44, 1K–5K followers, Instagram 90%)
**Finding:** 73% suffer burnout at least sometimes (down from 87% in 2022). Instagram drives 88% of burnout. Top cause: constant platform changes (70%).
**URL:** [prweb.com/releases/...creator-burnout](https://www.prweb.com/releases/a-majority-of-content-creators-and-influencers-struggle-with-burnout-as-concerns-for-ai-begin-to-surface-according-to-a-new-awin-group-survey-research-302257152.html)

**Caveat:** Self-selected sample, not probability-based. Small N. But directionally consistent with Wen 2026 (T1).
**What we use:** `burnout_risk` contextual framing (73% baseline prevalence).

### Vibely — Creator Burnout Report

**Finding:** 90% of creators experienced burnout. 71% considered quitting.
**Caveat:** No sample size or methodology disclosed. Treat as directional only.

---

## Tier 5 — Rejected sources (NOT cited in env constants)

The following sites were found during research but are **not cited** because they do not disclose methodology, sample sizes, or data collection processes. Their claims cannot be independently verified.

| Site | Why rejected |
|------|-------------|
| instacarousel.com | Affiliate blog, cites Socialinsider without adding primary data |
| midastools.co | SEO content, no methodology |
| kicksta.co | Growth tool vendor, no audit trail |
| postplanify.com | Aggregates others' data without attribution |
| monolit.sh | Blog post, no primary research |
| useadmetrics.com | Self-reported benchmarks, methodology unclear |
| creatorflow.so | Aggregates without disclosure |
| slumbertheory.com | Health blog, no clinical data source |
| dataslayer.ai | Marketing tool blog |
| almcorp.com | Agency blog |
| loopexdigital.com | Agency blog |
| carouselli.com | Tool vendor |
| influize.com | Tag listicle, no methodology |

---

*This bibliography was compiled April 2026. All URLs verified at time of writing.*
