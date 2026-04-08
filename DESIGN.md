# Viraltest — RL-Based Creator Optimization Agent

## Problem

Content creators on platforms like Meta (Instagram, Facebook) face:

- Unpredictable engagement
- No clear posting strategy
- Pressure to post frequently
- Burnout due to over-posting
- Drop in content quality over time

Existing tools show analytics (likes, reach) and past performance but don't **actively guide creators on optimal behavior over time**.

**Core problem**: No intelligent system continuously learns and adapts a creator's posting strategy to balance growth and burnout.

## Solution

An RL agent that learns **when to post**, **what type to post**, **which tags to use**, and **how to differentiate from competitors** — maximizing engagement while minimizing burnout over a weekly cycle.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  INFERENCE SCRIPT (inference.py)                                    │
│                                                                     │
│  env = ViraltestEnv(base_url="https://...")                        │
│  result = env.reset(task="weekly_strategic")  ← picks task         │
│  result = env.step(action)                    ← type-safe!         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  LLM Agent (OpenAI Client)                                │     │
│  │  Reads: observation → Decides: action                     │     │
│  │  Model: Qwen/Qwen2.5-72B-Instruct                        │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Logs: [START] [STEP] [END] to stdout                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    WebSocket /ws
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DOCKER CONTAINER (HF Space)                                        │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  FastAPI Server (server/app.py)  — port 8000              │     │
│  │                                                           │     │
│  │  ┌─────────────────────────────────────────────────────┐ │     │
│  │  │  ViraltestEnvironment                               │ │     │
│  │  │                                                     │ │     │
│  │  │  ┌─────────────────┐   ┌──────────────────────┐   │ │     │
│  │  │  │  reset(task)    │   │  step(action)         │   │ │     │
│  │  │  │  • Set task     │   │  1. Validate action   │   │ │     │
│  │  │  │  • Init state   │   │  2. Apply effects     │   │ │     │
│  │  │  │  • energy=1.0   │   │  3. Calc engagement   │   │ │     │
│  │  │  │  • followers=N  │   │  4. Tag analytics     │   │ │     │
│  │  │  │  • Init tags    │   │  5. Competitor check   │   │ │     │
│  │  │  │  • Init rivals  │   │  6. Update followers  │   │ │     │
│  │  │  │  • Return obs   │   │  7. Calc reward       │   │ │     │
│  │  │  └─────────────────┘   │  8. Check done        │   │ │     │
│  │  │                        │  9. Return obs        │   │ │     │
│  │  │  ┌─────────────────┐   └──────────────────────┘   │ │     │
│  │  │  │  state()        │                               │ │     │
│  │  │  │  • episode_id   │   ┌──────────────────────┐   │ │     │
│  │  │  │  • step_count   │   │  Grader (per task)    │   │ │     │
│  │  │  │  • task_name    │   │  • weekly_engage      │   │ │     │
│  │  │  └─────────────────┘   │  • weekly_strategic   │   │ │     │
│  │  │                        │  • weekly_competitive  │   │ │     │
│  │  │                        └──────────────────────┘   │ │     │
│  │  │                                                     │ │     │
│  │  │  Simulation Engine (research-backed params)         │ │     │
│  │  │  • Hour multipliers (Buffer 9.6M study)             │ │     │
│  │  │  • Content rates (SocialInsider 2025)               │ │     │
│  │  │  • Burnout curve (Sozee 2026 creator study)         │ │     │
│  │  │  • Tag engagement model                             │ │     │
│  │  │  • Competitor simulation                            │ │     │
│  │  └─────────────────────────────────────────────────────┘ │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Isolated • Reproducible • Secure • Deterministic (seeded RNG)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pydantic Models

```
models.py
├── ViraltestAction(Action)
│   ├── action_type: Literal["post", "rest", "create_content"]
│   ├── content_type: Optional[Literal["reel", "story", "carousel", "text_post"]]
│   ├── topic: Optional[str]
│   └── tags: Optional[list[str]]         ← max 5 tags per post
│
└── ViraltestObservation(Observation)
    ├── current_hour: int                  (0–23)
    ├── day_of_week: int                   (0–6)
    ├── days_elapsed: int
    ├── creator_energy: float              (0.0–1.0, burnout meter)
    ├── follower_count: int
    ├── engagement_rate: float             (rolling avg last 10 posts)
    ├── posts_today: int
    ├── time_since_last_post: int          (hours)
    ├── trending_topics: list[str]
    ├── content_queue_size: int
    ├── last_post_type: str
    │
    │   ── Tag Analytics ──
    ├── tag_performance: dict[str, float]  (tag → avg engagement from your past posts)
    ├── trending_tags: list[str]           (currently hot tags on the platform)
    │
    │   ── Competitor Intelligence ──
    ├── competitor_recent_posts: list[dict] (last 3 posts from similar creators)
    │   each: {content_type, topic, tags, engagement, hours_ago}
    ├── competitor_avg_engagement: float    (avg engagement of similar creators)
    ├── niche_saturation: float            (0.0–1.0, how crowded your topic space is)
    │
    ├── done: bool                         (inherited)
    └── reward: float                      (inherited)
```

---

## Data Flow — Single Step

```
AGENT                                   ENVIRONMENT
  │                                          │
  │  ── Action ───────────────────────────►  │
  │  {                                       │
  │    action_type: "post"                   │
  │    content_type: "reel"                  │  1. Validate fields
  │    topic: "AI trends"                    │  2. energy -= 0.25
  │    tags: ["ai", "tech", "future"]        │  3. engagement = base_rate
  │  }                                       │       × hour_mult
  │                                          │       × energy_quality
  │                                          │       × tag_boost
  │                                          │       × trending_bonus
  │                                          │       × competitor_diff_bonus
  │                                          │       × audience_fatigue
  │                                          │  4. Update tag_performance history
  │                                          │  5. Update niche_saturation
  │                                          │  6. followers += f(engagement)
  │                                          │  7. advance hour
  │                                          │  8. reward = composite score
  │                                          │  9. done? (168 steps or energy=0)
  │  ◄── Observation ─────────────────────  │
  │  {                                       │
  │    current_hour: 14                      │
  │    creator_energy: 0.62                  │
  │    follower_count: 10340                 │
  │    engagement_rate: 0.048                │
  │    tag_performance: {                    │
  │      "ai": 0.72, "tech": 0.55,          │
  │      "food": 0.31, "travel": 0.44       │
  │    }                                     │
  │    trending_tags: ["ai", "summer"]       │
  │    competitor_recent_posts: [            │
  │      {type:"carousel", topic:"AI",       │
  │       tags:["ai","ml"], eng:0.61,        │
  │       hours_ago: 3},                     │
  │      ...                                 │
  │    ]                                     │
  │    niche_saturation: 0.7                 │
  │    done: false, reward: 0.67             │
  │  }                                       │
```

---

## Step Processing (Server-Side)

### 1. Validate Action

- `action_type` must be one of `post`, `rest`, `create_content`
- If `post`: `content_type` required, `topic` non-empty ≤200 chars, `tags` max 5 items from known pool
- Invalid action → reward=0, error in observation

### 2. Apply Energy Cost

| Action | Energy Effect |
|---|---|
| Post (reel) | -0.25 |
| Post (carousel) | -0.20 |
| Post (story) | -0.08 |
| Post (text_post) | -0.06 |
| Rest | +0.12 (capped at 1.0) |
| Create content | -0.05, queue += 1 |

Repetition penalty: same content type as last 3 posts → extra -0.05.
If energy ≤ 0 → `done = true` (burnout).

### 3. Calculate Engagement (post only)

```
engagement = base_rate × hour_mult × quality × tag_boost × trending_bonus
             × competitor_diff × fatigue_penalty
```

**Base engagement rates** (SocialInsider 2025):

| Type | Rate | Reach Mult |
|---|---|---|
| Carousel | 0.55% | 1.0x |
| Reel | 0.52% | 2.25x |
| Story | 0.30% | 0.5x |
| Text post | 0.37% | 0.44x |

**Hour multipliers** (Buffer 9.6M posts):

| Time Slot | Multiplier |
|---|---|
| 9AM–12PM weekdays | 1.3x |
| 12PM–3PM Tue-Thu | 1.4x (peak) |
| 6PM–8PM | 1.25x |
| 8PM–11PM | 1.1x |
| 11PM–6AM | 0.5x |
| Fri/Sat | 0.7x base penalty |

**Quality modifier** (Sozee burnout study: 30-52% productivity drop):

```
quality = 1.0 if energy > 0.5 else max(0.48, energy × 1.5)
```

**Tag boost** (see Tag Engagement section below):

```
tag_boost = 1.0 + 0.1 × count(tags that are in trending_tags)
            + 0.05 × avg(tag_performance[tag] for tag in action.tags)
```

**Competitor differentiation bonus**:

```
if topic NOT in competitor_recent_topics (last 12hrs):
    competitor_diff = 1.3   (unique angle, underserved)
elif niche_saturation > 0.7:
    competitor_diff = 0.6   (oversaturated, too many posting same thing)
else:
    competitor_diff = 1.0   (neutral)
```

**Audience fatigue**: posts_today > 3 → ×0.5, posts_today > 5 → ×0.1

**Trending bonus**: topic matches trending → ×1.5

### 4. Update Tag Performance

After each post, the environment records engagement per tag:

```python
for tag in action.tags:
    tag_history[tag].append(this_post_engagement)
    tag_performance[tag] = rolling_avg(tag_history[tag], window=5)
```

This gives the agent a feedback loop — it can see which tags historically work and adapt.

### 5. Update Competitor State

Each step, the simulated competitors also "post" according to a deterministic schedule (seeded RNG):

```python
for competitor in competitors:
    if should_post(competitor, current_hour):  # seeded probability
        competitor.recent_posts.append({
            content_type: random.choice(types),
            topic: random.choice(competitor.niche_topics),
            tags: random.sample(tag_pool, 3),
            engagement: base + noise,
            hours_ago: 0
        })
    # Age out old posts
    competitor.recent_posts = [p for p in competitor.recent_posts if p.hours_ago < 48]

niche_saturation = count(competitor posts with overlapping topic in last 12hrs) / max_posts
```

### 6. Update Followers

- Posted: `followers += int(engagement × 100)`
- No post for 48+ hrs: followers decay (algorithm deprioritization)

### 7. Advance Time

- hour += 1
- If hour ≥ 24: day advances, posts_today resets, trending topics/tags rotate (seeded)

### 8. Compute Reward

```
reward = clamp(0, 1,
    engagement_gained × 0.3
    + energy_delta × 0.15
    + consistency_bonus × 0.15
    + tag_optimization_score × 0.15
    + competitor_diff_score × 0.15
    - burnout_penalty × 0.1
)
```

- `consistency_bonus`: 1.0 if 1-2 posts/day, 0.5 if 0 or 3, 0.0 if 4+
- `tag_optimization_score`: how well agent's chosen tags match high-performing + trending tags
- `competitor_diff_score`: 1.0 if posting unique angle, 0.0 if fully overlapping
- `burnout_penalty`: 1.0 if energy < 0.2

### 9. Check Done

Episode ends when:
- `step_count >= 168` (1 week = 7 days × 24 hours)
- `energy <= 0` (burned out)

---

## Tag Engagement System

### How Tags Work

The environment maintains a **tag pool** of ~30 tags across categories:

| Category | Example Tags |
|---|---|
| Tech | `ai`, `ml`, `coding`, `startup`, `saas` |
| Lifestyle | `fitness`, `travel`, `food`, `wellness`, `fashion` |
| Trending | `summer`, `worldcup`, `election` (rotate daily) |
| Niche | `productivity`, `minimalism`, `stoic`, `web3` |
| Broad | `motivation`, `tips`, `howto`, `viral` |

### Tag Performance Tracking

Each tag accumulates engagement history from the agent's own posts:

```
tag_performance = {
    "ai": 0.72,          ← avg engagement when you used this tag
    "fitness": 0.31,     ← this tag isn't working for your audience
    "motivation": 0.55,
    ...
}
```

Initially all tags start at 0.0 (unknown). As the agent posts with different tags, it builds this signal.

### Tag Dynamics

- **Trending tags** change every 24 simulated hours (seeded, deterministic)
- Using a trending tag gives +10% engagement per trending tag matched
- Using a high-performing tag (from your history) gives +5% per tag
- Using an **oversaturated tag** (competitors using it heavily) gives -10%
- Max 5 tags per post — agent must choose wisely

### What the Agent Must Learn

1. **Discover** which tags work for its audience (explore early, exploit later)
2. **Ride trends** — use trending tags when they align with its niche
3. **Avoid saturation** — if competitors are all using `#ai`, pivot to `#ml` or `#coding`
4. **Combine** high-performing niche tags with 1-2 trending tags for optimal reach+engagement

---

## Competitor Intelligence System

### Simulated Competitors

The environment simulates **3 competing creators** in the same niche. Each has:

```python
competitor = {
    "name": "creator_A",
    "niche_topics": ["AI", "tech", "startups"],      # their focus
    "preferred_types": ["reel", "carousel"],           # what they mostly post
    "posting_frequency": 2.5,                          # avg posts/day
    "base_engagement": 0.45,                           # their avg engagement
    "tag_preferences": ["ai", "startup", "coding"],
}
```

### What the Agent Sees

Each step, the observation includes:

```python
competitor_recent_posts: [
    {"content_type": "reel", "topic": "AI tools", "tags": ["ai", "tools"],
     "engagement": 0.61, "hours_ago": 3},
    {"content_type": "carousel", "topic": "startup tips", "tags": ["startup"],
     "engagement": 0.48, "hours_ago": 8},
    {"content_type": "reel", "topic": "AI news", "tags": ["ai", "news"],
     "engagement": 0.52, "hours_ago": 14},
]
competitor_avg_engagement: 0.54
niche_saturation: 0.7   # 0.0=empty, 1.0=everyone posting same stuff
```

### How Competitors Affect Your Engagement

```
if your topic overlaps with ≥2 competitor posts in last 12hrs:
    niche_saturation → high (0.7+)
    your engagement × 0.6  (audience already saw similar content)

if your topic is unique (no overlap in 12hrs):
    competitor_diff_bonus = 1.3x  (fresh angle, algorithm favors)

if competitor engagement is HIGH on a topic:
    that topic has proven demand, but also competition
    → agent must decide: follow the proven topic (safe) or differentiate (risky but higher upside)
```

### What the Agent Must Learn

1. **Monitor** competitor posting patterns and timing
2. **Differentiate** — find underserved time slots and topics
3. **Counter-program** — post different content type when competitors flood reels
4. **Learn from competitor success** — if competitor's carousel on "AI" got 0.8 engagement, the topic has demand, but post at a different time or with different tags

---

## Tasks & Graders (All Weekly — 168 steps)

All three tasks run for exactly **1 week (168 hourly steps)**. The difficulty increases through what dimensions are graded and what constraints apply.

### Task 1: weekly_engage (Easy)

**Focus**: Pure engagement maximization.

**What's active**: Basic mechanics only — time of day, content type, energy, audience fatigue.

**What's NOT graded**: Tags, competitors (still simulated but don't affect score).

**Grader formula**:

```
score = total_engagement / theoretical_max_engagement
```

**Theoretical max**: Calculated as if agent posted at every peak hour with best content type at full energy. Roughly ~14 optimal posts over 7 days.

**How it's computed**:
1. Sum all engagement values from every post the agent made
2. Divide by the theoretical max (computed from: 2 posts/day × 7 days × peak_hour_mult × best_content_rate × quality=1.0)
3. Clamp to [0.0, 1.0]

**What a smart agent does**: Posts 1-2x/day at peak hours (12-3PM), uses high-engagement content types (carousel/reel), rests to keep energy above 0.5.

**What a dumb agent scores**: Random ≈ 0.08–0.12. Spam-every-hour ≈ 0.15–0.25 (audience fatigue kills it).

---

### Task 2: weekly_strategic (Medium)

**Focus**: Engagement + energy management + tag optimization.

**What's active**: Everything from Task 1, PLUS tag engagement system.

**Grader formula**:

```
tag_discovery = unique_tags_used_with_positive_engagement / total_tag_pool_size
tag_exploitation = avg(top_3_tag_performances) / max_possible_tag_performance

tag_score = 0.4 × tag_discovery + 0.6 × tag_exploitation

score = (0.35 × normalized_engagement)
      + (0.25 × tag_score)
      + (0.25 × avg_energy)
      + (0.15 × consistency_score)
```

**Constraints**:
- If energy ever drops below 0.3 → score capped at 0.5
- If fewer than 5 unique tags used across the week → score × 0.7

**How each component works**:

| Component | What it measures | How it's normalized |
|---|---|---|
| `normalized_engagement` | Total engagement across all posts | `sum(engagement) / theoretical_max` |
| `tag_discovery` | Did the agent explore different tags? | `unique_positive_tags / 30 (pool size)` |
| `tag_exploitation` | Did the agent learn which tags work and reuse them? | `avg(best 3 tags) / 1.0` |
| `avg_energy` | Did the agent maintain sustainable energy? | `mean(energy at each step) / 1.0` |
| `consistency_score` | Regular posting rhythm | `days_with_1_or_2_posts / 7` |

**What a smart agent does**: Explores different tags in days 1-2, identifies top performers by day 3, then exploits them while riding trending tags. Balances rest to keep energy > 0.5.

**What a dumb agent scores**: Random ≈ 0.10–0.15 (random tags, no learning). Always-same-tags ≈ 0.20 (no discovery).

---

### Task 3: weekly_competitive (Hard)

**Focus**: Everything + competitor awareness + follower growth.

**What's active**: Full simulation — engagement, tags, competitors, niche saturation.

**Grader formula**:

```
follower_growth = (final_followers - initial_followers) / initial_followers
normalized_growth = min(1.0, follower_growth / target_growth_rate)

competitor_outperformance = your_avg_engagement / competitor_avg_engagement
normalized_outperformance = min(1.0, competitor_outperformance / 1.5)

differentiation = steps_where_topic_was_unique / total_posting_steps

score = (0.25 × normalized_engagement)
      + (0.20 × tag_score)           ← same formula as Task 2
      + (0.20 × normalized_growth)
      + (0.15 × normalized_outperformance)
      + (0.10 × differentiation)
      + (0.10 × min_energy_floor)
```

**Constraints**:
- Energy hits 0 → score = 0.0 (total fail, burned out)
- Fewer than 3 content types used → score × 0.5
- Fewer than 8 unique tags used → score × 0.7
- If agent never checks competitor patterns (always overlaps) → differentiation = 0

**How each component works**:

| Component | Weight | What it measures | Detail |
|---|---|---|---|
| `normalized_engagement` | 25% | Raw engagement quality | Same as Task 1 |
| `tag_score` | 20% | Tag strategy quality | Discovery + exploitation (Task 2 formula) |
| `normalized_growth` | 20% | Follower growth over the week | `target_growth_rate` = 5% (500 new followers on 10K base) |
| `normalized_outperformance` | 15% | Beat your competitors | Your avg engagement / competitor avg. Capped at 1.0 when you're 1.5x better |
| `differentiation` | 10% | Posting unique angles | % of your posts where topic wasn't posted by competitors in last 12hrs |
| `min_energy_floor` | 10% | Never crashed | `min(energy_history)` — lowest energy point. Rewards agents that never dipped dangerously low |

**What a smart agent does**:
1. Days 1-2: Explore tags, observe competitor patterns
2. Days 3-4: Exploit best tags, counter-program competitors (post when they rest, pick gaps)
3. Days 5-7: Maximize engagement with learned strategy, maintain energy, diversify content types

**What a dumb agent scores**: Random ≈ 0.08. Copy-competitor-strategy ≈ 0.20 (no differentiation). Smart ≈ 0.50–0.75.

---

## Grading Strategy — In Depth

### Why Weekly for All Tasks

- **Consistency**: Same horizon (168 steps) makes graders comparable
- **Runtime**: 168 steps × 3 tasks = 504 total LLM calls. At ~2s per call = ~17 minutes. Under the 20-minute limit
- **Meaningful cycle**: A week is the natural content planning cycle for creators. Days are too short to show learning. Months are too long for inference budget

### Grading Philosophy

The grading is designed so that **each task requires mastering the previous task's skills plus new ones**:

```
Task 1 (Easy)    → Can you post well?
                    (timing + content type + energy)

Task 2 (Medium)  → Can you post SMART?
                    (Task 1 + tag discovery + tag exploitation)

Task 3 (Hard)    → Can you OUTCOMPETE?
                    (Task 2 + competitor awareness + differentiation + growth)
```

### Why These Weights

**Task 1** — Engagement is everything (100% engagement-derived). Pure skill test.

**Task 2** — Split focus:
- 35% engagement (still important, but not enough alone)
- 25% tags (new skill: must explore AND exploit)
- 25% energy (sustainability matters now)
- 15% consistency (rhythm matters)

**Task 3** — Multi-dimensional:
- No single component dominates (max 25%)
- Agent must be good at everything, great at nothing is fine
- `differentiation` (10%) is small but acts as tiebreaker between otherwise similar agents
- `min_energy_floor` (10%) punishes agents that nearly crashed even if they recovered

### Anti-Gaming Properties

| Potential Exploit | Why it fails |
|---|---|
| Post every hour | Audience fatigue kills engagement → low `normalized_engagement` |
| Always rest | Zero engagement, zero tag score, zero growth → score ≈ 0.05 |
| Use same 2 tags always | `tag_discovery` tanks in Task 2/3. Score × 0.7 penalty if < 5/8 tags |
| Copy competitor topics | `differentiation` = 0, `niche_saturation` high → engagement × 0.6 |
| Post only reels | Score × 0.5 in Task 3 (need ≥ 3 types) |
| Ignore competitors entirely | Random overlap → sometimes lucky, but `differentiation` averages low |
| Post gibberish topics | Topic validation + no trending match → low engagement |

### Score Distribution (Expected)

| Agent Type | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Random | 0.08–0.12 | 0.10–0.15 | 0.06–0.10 |
| Always rest | 0.02 | 0.05 | 0.02 |
| Spam (post every step) | 0.15–0.25 | 0.12–0.18 | 0.08–0.15 |
| Fixed strategy (no learning) | 0.30–0.40 | 0.25–0.35 | 0.20–0.30 |
| Smart LLM agent | 0.55–0.80 | 0.45–0.70 | 0.40–0.65 |

Task 3 is intentionally hardest — even a good agent won't ace it because competitor dynamics add noise and require adaptation.

---

## Anti-Exploit Guards

| Exploit | Guard |
|---|---|
| Reward hacking (long gibberish) | Cap reward per step at 1.0, validate topic, max 200 chars |
| Grader gaming | Random agent must score < 0.15, spam agent < 0.30 |
| State reset abuse | Reset only works between tasks, mid-episode reset ignored |
| Invalid actions | Strict field validation, invalid → 0 reward + error |
| Rest farming | Rest → reward ≈ 0, energy is a resource not a goal |
| Repetitive posting | Same type 3x → engagement -20% + energy penalty |
| Tag spamming | Max 5 tags per post, must be from known pool |
| Competitor copying | Niche saturation penalty, differentiation score = 0 |

### Sanity Test Agents

Run before submitting:

| Agent | Expected Score (Task 3) | Red Flag If |
|---|---|---|
| Random agent | < 0.10 | Reward too easy |
| Always-rest | < 0.05 | Resting rewarded |
| Spam (post every step, same type) | < 0.15 | No fatigue working |
| Fixed (same action every time) | < 0.30 | Environment too simple |
| Smart (LLM-driven) | 0.40–0.65 | This is the real range |

---

## Simulation Mechanics

### Energy Dynamics (research-backed)

```python
energy -= content_cost[action.content_type]

# Repetition fatigue (creative fatigue = 40% of burnout)
if action.content_type == last_3_posts_type:
    energy -= 0.05

# Recovery: slow, not instant
if action.action_type == "rest":
    energy = min(1.0, energy + 0.12)

# Quality modifier (30-52% productivity drop at burnout)
quality = 1.0 if energy > 0.5 else max(0.48, energy * 1.5)
```

### Extended Features

#### A. Content Repetition Fatigue
Same content type 3x in a row → engagement drops 20%. Based on creative fatigue being #1 burnout cause (40%).

#### B. Platform Activity / Competition Window
`niche_saturation` (0.0–1.0) in observation. When many competitors post same topic → per-post engagement drops. From the broadcast scheduling paper (Preprints.org 2025).

#### C. Follower Tier Response
Small accounts (<10K) get more from reels (reach). Large accounts (>50K) benefit from carousels (depth). From CreatorsJet 10K post study.

#### D. Trending Topic & Tag Bonus
If topic or tags match trending → 1.5x and +10% respectively. Topics and tags rotate daily (seeded). Forces adaptive behavior.

#### E. Algorithm Penalty for Inconsistency
No post for 48+ hours → next 2 posts get 0.6x engagement. Based on algorithmic content selection research (arxiv:2410.13108).

#### F. Tag Engagement Tracking
Full per-tag engagement history. Agent sees which tags produce results and must balance exploration (try new tags) vs exploitation (reuse winners). See Tag Engagement System section.

#### G. Competitor Awareness
3 simulated rival creators with deterministic posting schedules. Agent sees their recent posts, topics, tags, and engagement. Must differentiate to avoid saturation. See Competitor Intelligence System section.

---

## Research Backing

### Engagement Data

- **Buffer 2026**: 9.6M posts analyzed — peak posting times, day-of-week effects
- **SocialInsider 2025**: Engagement rates by content type (carousel 0.55%, reel 0.52%, image 0.37%)
- **CreatorsJet 10K post study**: Reels give 2.25x reach vs images, carousels give depth

### Burnout Data

- **Sozee 2026**: 90% creators experience burnout, 30-52% productivity drop
- **TastyEdits Creator Study**: 57% spend 4+ hrs/day, 79% have experienced burnout
- **Creative fatigue**: #1 cause at 40%, algorithm pressure at 38%

### Academic Papers

| Paper | Relevance |
|---|---|
| "Review Old Strategies, New Environments: RL on Social Media" (ScienceDirect 2024) | RL framework for social media — validates env design |
| arxiv:2410.13108 "Algorithmic Content Selection and User Disengagement" | Over-optimizing immediate engagement causes churn — justifies burnout mechanic |
| arxiv:2211.13585 "Learning Optimal Break Policies" | Strategic breaks sustain engagement — supports "rest" action |
| "Optimizing Broadcast Scheduling" (Preprints.org 2025) | Low-competition windows > frequency — competition variable |
| RLNVR arxiv:2508.12165 | RL from noisy social media signals — proves this is active research |

### Data Sources

- **Meta Content Library**: Real engagement data for public Instagram/Facebook posts ([docs](https://developers.facebook.com/docs/content-library-and-api))
- **Meta Graph API — Creator Marketplace Insights**: Real creator metrics ([docs](https://developers.facebook.com/docs/graph-api/reference/creator-marketplace-content/insights/))

---

## Inference Script Structure

```python
import os
from openai import OpenAI
from viraltest import ViraltestEnv, ViraltestAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASKS = ["weekly_engage", "weekly_strategic", "weekly_competitive"]
MAX_STEPS = 168  # 7 days × 24 hours (same for all tasks)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

for task in TASKS:
    log_start(task, "viraltest", MODEL_NAME)
    env = ViraltestEnv(base_url="http://localhost:8000")
    result = env.reset(task=task)
    rewards = []

    for step in range(MAX_STEPS):
        obs = result.observation
        user_msg = format_observation(obs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7, max_tokens=150
        )
        action = parse_action(response.choices[0].message.content)
        result = env.step(action)
        rewards.append(result.reward)
        log_step(step+1, str(action), result.reward, result.done, None)
        if result.done:
            break

    score = grader_score(task, rewards, obs)
    log_end(score > 0.1, len(rewards), score, rewards)
    env.close()
```

Log format:

```
[START] task=weekly_competitive env=viraltest model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=post(reel,"AI trends",["ai","tech"]) reward=0.67 done=false error=null
[STEP] step=2 action=rest() reward=0.05 done=false error=null
...
[END] success=true steps=168 score=0.624 rewards=0.67,0.05,...,0.55
```

---

## Judging Alignment

| Criteria | Weight | What backs us |
|---|---|---|
| Real-world utility | 30% | Meta Content Library, Buffer study, creator burnout stats, tag analytics, competitor analysis |
| Task & grader quality | 25% | 3 weekly tasks with progressive difficulty, multi-component graders, deterministic |
| Environment design | 20% | Energy from burnout studies, engagement from SocialInsider, tag + competitor systems |
| Code quality & spec | 15% | OpenEnv compliant, typed models, Dockerfile works |
| Creativity & novelty | 10% | Multi-objective (engagement vs burnout vs tags vs competition), backed by 5+ papers |

---

## File Map

| File | Purpose |
|---|---|
| `models.py` | `ViraltestAction` and `ViraltestObservation` Pydantic models |
| `server/viraltest_environment.py` | Simulation logic, task switching, graders, reward calc, tag + competitor systems |
| `client.py` | `ViraltestEnv` client — `_step_payload`, `_parse_result`, `_parse_state` |
| `inference.py` | LLM-driven agent with `[START]`/`[STEP]`/`[END]` logging |
| `openenv.yaml` | Environment metadata |
| `Dockerfile` | Container build |
| `README.md` | User-facing docs |
| `DESIGN.md` | This file |
