# Viraltest Simulation Report

**Task:** Hard — Competitive (weekly_competitive)
**Episode Length:** 168 steps (7 days x 24 hours)
**Starting Followers:** 10,000 | **Starting Energy:** 1.00

---

## Executive Summary

11 agent strategies were evaluated on the Hard — Competitive task. The **Balanced Creator** (0.8775) and **Smart Agent** (0.8745) achieved the highest scores by combining strategic posting, energy management, and tag diversity. Two agents (**Spam Post**, **No Rest**) burned out within 8 steps, scoring 0.0000. The **Always Rest** agent lost 45% of its followers from inactivity.

---

## Leaderboard

| Rank | Scenario | Score | Followers | Delta | Energy | Burned Out |
|------|----------|-------|-----------|-------|--------|------------|
| 1 | Balanced Creator | **0.8775** | 12,534 | +2,534 (+25.3%) | 1.00 | No |
| 2 | Smart Agent | **0.8745** | 12,200 | +2,200 (+22.0%) | 1.00 | No |
| 3 | Tag Explorer | **0.8323** | 11,351 | +1,351 (+13.5%) | 0.94 | No |
| 4 | Copycat | **0.6136** | 11,589 | +1,589 (+15.9%) | 1.00 | No |
| 5 | Burst Poster | **0.6111** | 11,701 | +1,701 (+17.0%) | 0.44 | No |
| 6 | Queue Optimizer | **0.3520** | 11,215 | +1,215 (+12.2%) | 1.00 | No |
| 7 | Weekend Warrior | **0.1257** | 7,659 | -2,341 (-23.4%) | 1.00 | No |
| 8 | Night Poster | **0.0937** | 10,237 | +237 (+2.4%) | 0.59 | No |
| 9 | Always Rest | **0.0350** | 5,497 | -4,503 (-45.0%) | 1.00 | No |
| 10 | Spam Post | **0.0000** | 10,625 | +625 (+6.3%) | 0.00 | **YES** |
| 11 | No Rest | **0.0000** | 10,213 | +213 (+2.1%) | 0.00 | **YES** |

---

## Detailed Agent Analysis

### 1. Balanced Creator — Score: 0.8775 (BEST)

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Energy | 1.00 |
| Final Followers | 12,534 (+25.3%) |
| Engagement Rate | 0.827 |
| Total Posts | 28 |
| Total Rests | 84 |
| Content Created | 56 |
| Unique Tags | 19 |
| Min Energy | 0.795 (never dipped below safe zone) |
| Avg Reward | 0.219 |
| Max Reward | 0.738 |

**Strategy:** Create → Post → Rest cycle. Uses the content queue (56 items created, 28 posted from queue at 50% energy cost). Posts during peak hours with trending topics. Never risks burnout.

**Top Tags:** #food (1.32), #election (1.31), #coding (1.16), #saas (1.03), #crypto (1.02)

**Why it won:** Highest follower growth (+2,534), perfect energy management (never below 0.795), excellent tag diversity (19 unique), and consistent daily posting.

---

### 2. Smart Agent — Score: 0.8745

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Energy | 1.00 |
| Final Followers | 12,200 (+22.0%) |
| Engagement Rate | 1.556 |
| Total Posts | 14 |
| Total Rests | 154 |
| Unique Tags | 19 |
| Min Energy | 0.55 |
| Avg Reward | 0.230 |
| Max Reward | 0.760 |

**Strategy:** Posts only during peak hours (9-20) when energy > 0.4 and posts < 2/day. Uses trending topics and tags. Rests aggressively.

**Top Tags:** #ai (3.56), #wellness (2.55), #summer (2.36), #crypto (2.18), #newyear (2.01)

**Why it's strong:** Highest individual tag performance (#ai at 3.56), highest engagement rate (1.556), but fewer posts (14 vs 28) cost it the top spot.

---

### 3. Tag Explorer — Score: 0.8323

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Energy | 0.94 |
| Final Followers | 11,351 (+13.5%) |
| Engagement Rate | 0.774 |
| Total Posts | 15 |
| Unique Tags | **30** (highest) |
| Min Energy | 0.69 |

**Strategy:** New tag combination every post. Maximizes tag discovery — 30 unique tags used (the highest of all agents).

**Why it scored high:** The grading formula rewards tag diversity heavily. 30 unique tags gave a massive tag_discovery bonus.

---

### 4. Copycat — Score: 0.6136

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Energy | 1.00 |
| Final Followers | 11,589 (+15.9%) |
| Total Posts | 21 |
| Unique Tags | 8 |
| Min Energy | 0.10 (dangerous dip!) |

**Strategy:** Copies competitor topics and content types. Posts when competitors are active.

**Weakness:** High niche saturation from copying rivals. Only 8 unique tags (penalized). Min energy hit 0.10 — nearly burned out.

---

### 5. Burst Poster — Score: 0.6111

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Energy | 0.44 |
| Final Followers | 11,701 (+17.0%) |
| Total Posts | **57** (highest) |
| Unique Tags | 13 |
| Min Energy | 0.25 |

**Strategy:** 3 posts in rapid succession, then rests until recovered. Repeat.

**Weakness:** Ended with only 0.44 energy. 57 posts caused audience fatigue (posts > 3/day get heavy penalty). Low per-post engagement (0.208) despite high volume.

---

### 6. Queue Optimizer — Score: 0.3520

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Energy | 1.00 |
| Final Followers | 11,215 (+12.2%) |
| Total Posts | 14 |
| Content Created | 17 |
| Unique Tags | 12 |

**Strategy:** Creates content first (builds queue), then posts from queue at half energy cost.

**Weakness:** Spent too long in "prep" phase creating content. Only 14 actual posts despite 17 items queued. Score penalized for under-utilizing the queue.

---

### 7. Weekend Warrior — Score: 0.1257

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Followers | 7,659 **(-23.4%)** |
| Total Posts | 6 |
| Unique Tags | 6 |

**Strategy:** Only posts on Saturday and Sunday. Rests Mon-Fri.

**Weakness:** 5 days of inactivity triggered follower decay (-2,341) and algorithm penalty. Only 6 posts total. Weekend posting also gets a 0.7x penalty multiplier.

---

### 8. Night Poster — Score: 0.0937

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Followers | 10,237 (+2.4%) |
| Total Posts | 49 |
| Unique Tags | 2 |
| Engagement Rate | 0.036 |

**Strategy:** Posts exclusively at night (23:00-06:00) with boring topics.

**Weakness:** Night hours get 0.5x multiplier. Only 2 unique tags (#stoic, #minimalism) — severe tag penalty. Despite 49 posts, engagement was near-zero (0.036).

---

### 9. Always Rest — Score: 0.0350

| Metric | Value |
|--------|-------|
| Steps Completed | 168 / 168 |
| Final Followers | 5,497 **(-45.0%)** |
| Total Posts | 0 |
| Engagement Rate | 0.000 |

**Strategy:** Never posts. Rests every step.

**Result:** Zero engagement. Lost 4,503 followers (45%) to decay. Algorithm penalty stacked from inactivity. Energy stayed at 1.00 — completely wasted.

---

### 10. Spam Post — Score: 0.0000

| Metric | Value |
|--------|-------|
| Steps Completed | **4** / 168 |
| Final Energy | **0.00 (BURNED OUT)** |
| Final Followers | 10,625 (+6.3%) |

**Strategy:** Posts the same reel with "AI tools" topic every step. No rest.

**Result:** Burned out at step 4. Each reel costs 0.25 energy. 4 reels = 1.00 energy drained. Episode ended at step 4 with score 0.0000 (burnout = automatic fail on competitive task).

---

### 11. No Rest — Score: 0.0000

| Metric | Value |
|--------|-------|
| Steps Completed | **8** / 168 |
| Final Energy | **0.00 (BURNED OUT)** |
| Final Followers | 10,213 (+2.1%) |

**Strategy:** Posts varied content types but never rests.

**Result:** Burned out at step 8. Mixed content types (reel, carousel, story, text_post) averaged ~0.125 energy cost. 8 posts without rest = burnout. Score: 0.0000.

---

## Key Metrics Comparison

### Energy Management
| Agent | Min Energy | Final Energy | Energy Safety |
|-------|-----------|--------------|---------------|
| Always Rest | 1.000 | 1.00 | Wasted |
| Balanced | 0.795 | 1.00 | Excellent |
| Tag Explorer | 0.690 | 0.94 | Good |
| Queue Optimizer | 0.610 | 1.00 | Good |
| Smart Agent | 0.550 | 1.00 | Good |
| Burst Poster | 0.250 | 0.44 | Risky |
| Night Poster | 0.230 | 0.59 | Dangerous |
| Copycat | 0.100 | 1.00 | Near-fatal dip |
| Weekend | 0.100 | 1.00 | Near-fatal dip |
| No Rest | 0.000 | 0.00 | BURNED OUT |
| Spam Post | 0.000 | 0.00 | BURNED OUT |

### Posting Volume vs Quality
| Agent | Posts | Engagement Rate | Engagement per Post |
|-------|-------|----------------|---------------------|
| Burst | 57 | 0.208 | Low (fatigue) |
| Night Poster | 49 | 0.036 | Very low (timing) |
| Balanced | 28 | 0.827 | High |
| Copycat | 21 | 0.497 | Medium |
| Tag Explorer | 15 | 0.774 | High |
| Smart Agent | 14 | 1.556 | Very high |
| Queue Opt | 14 | 0.870 | High |
| Weekend | 6 | 0.635 | Medium |
| Spam | 4 | 1.567 | High (but burned out) |

---

## Lessons Learned

1. **Burnout is fatal** — On the competitive task, burnout = score 0.0000. Energy management is the #1 priority.

2. **Quality > Quantity** — Smart Agent posted only 14 times but had the highest engagement rate (1.556). Burst posted 57 times but scored lower.

3. **Tag diversity matters** — Tag Explorer's 30 unique tags boosted its score to 0.8323 despite moderate engagement. Night Poster's 2 tags destroyed its score.

4. **Content queue is powerful** — Balanced Creator used create_content (56 times) to build a queue, then posted at half energy cost. This enabled 28 posts while maintaining 0.795+ energy.

5. **Timing is critical** — Night Poster proved that posting at wrong hours (0.5x multiplier) wastes energy for near-zero engagement.

6. **Copying competitors backfires** — Copycat achieved decent followers but niche saturation penalty and low tag diversity (8) capped its score at 0.6136.

7. **Consistency beats bursts** — Posting 1-2/day consistently (Balanced, Smart) scored higher than bursting 3+ posts then resting (Burst).

---

*Report generated from Viraltest Creator Intelligence Center*
*Task: weekly_competitive | 168 hourly steps | 3 competitor profiles*
