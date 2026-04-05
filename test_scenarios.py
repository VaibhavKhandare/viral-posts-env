"""
Viraltest — Edge Case & Scenario Tests
Runs 6 agent strategies for all 3 tasks (18 episodes total).
Prints a snapshot after each 168-step episode.
"""

import random as stdlib_random
from collections import Counter
from typing import Callable, List, Tuple

from models import ViraltestAction
from server.viraltest_environment import (
    TAG_POOL,
    ViraltestEnvironment,
    ViraltestObservation,
)

TASKS = ["weekly_engage", "weekly_strategic", "weekly_competitive"]
SEED = 42


def run_episode(
    task: str,
    agent_fn: Callable[[ViraltestObservation, int], ViraltestAction],
    label: str,
) -> None:
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=SEED)
    rewards: List[float] = []
    actions: List[str] = []
    min_energy = 1.0
    burned_out = False

    for step in range(1, 169):
        action = agent_fn(obs, step)
        obs = env.step(action)
        r = obs.reward if obs.reward is not None else 0.0
        rewards.append(r)
        actions.append(action.action_type)
        min_energy = min(min_energy, obs.creator_energy)
        if obs.done and obs.creator_energy <= 0:
            burned_out = True
        if obs.done:
            break

    score = (obs.metadata or {}).get("grader_score", 0.0)
    action_counts = Counter(actions)
    total_steps = len(rewards)

    print(f"  Task: {task}")
    print(f"  Steps: {total_steps} | Done: {obs.done} | Burned out: {burned_out}")
    print(f"  Score: {score:.4f} | Total reward: {sum(rewards):.2f} | Avg reward: {sum(rewards)/len(rewards):.3f}")
    print(f"  Energy: {obs.creator_energy:.2f} | Min energy: {min_energy:.2f}")
    print(f"  Followers: {obs.follower_count} (started 10000, delta {obs.follower_count - 10000:+d})")
    print(f"  Engagement rate: {obs.engagement_rate:.4f}")
    print(f"  Actions: post={action_counts.get('post',0)} rest={action_counts.get('rest',0)} create={action_counts.get('create_content',0)}")
    print(f"  Unique tags: {len(obs.tag_performance)} | Unique types: {len(set(a for a in actions if a == 'post'))}")
    print(f"  Niche saturation: {obs.niche_saturation:.3f}")
    print()


# ---------------------------------------------------------------------------
# SCENARIO 1: Always Rest (never post anything)
# ---------------------------------------------------------------------------
def agent_always_rest(obs: ViraltestObservation, step: int) -> ViraltestAction:
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 2: Spam Post (same type, same topic, no tags, every step)
# ---------------------------------------------------------------------------
def agent_spam(obs: ViraltestObservation, step: int) -> ViraltestAction:
    return ViraltestAction(
        action_type="post", content_type="reel", topic="AI tools", tags=["ai"]
    )


# ---------------------------------------------------------------------------
# SCENARIO 3: Post only off-peak non-trending gibberish topics
# ---------------------------------------------------------------------------
def agent_bad_timing(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.current_hour >= 23 or obs.current_hour < 6:
        return ViraltestAction(
            action_type="post",
            content_type="text_post",
            topic="random boring stuff nobody cares about",
            tags=["stoic", "minimalism"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 4: Post every step but vary everything (no rest = burnout)
# ---------------------------------------------------------------------------
_types = ["reel", "carousel", "story", "text_post"]
_topics = ["AI tools", "fitness routine", "growth hacks", "travel guide", "food recipe", "wellness tips"]
_rng4 = stdlib_random.Random(99)


def agent_no_rest(obs: ViraltestObservation, step: int) -> ViraltestAction:
    return ViraltestAction(
        action_type="post",
        content_type=_types[step % 4],
        topic=_rng4.choice(_topics),
        tags=_rng4.sample(TAG_POOL, 3),
    )


# ---------------------------------------------------------------------------
# SCENARIO 5: Smart agent (optimal strategy from DESIGN.md)
# ---------------------------------------------------------------------------
_ct_idx = 0
_tag_idx = 0


def _reset_smart_state():
    global _ct_idx, _tag_idx
    _ct_idx = 0
    _tag_idx = 0


def agent_smart(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _ct_idx, _tag_idx

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20 and obs.posts_today < 2:
        ct = _types[_ct_idx % 4]
        _ct_idx += 1
        topic = obs.trending_topics[0] if obs.trending_topics else "AI tools"
        tags = list(obs.trending_tags[:2])
        tags.append(TAG_POOL[_tag_idx % len(TAG_POOL)])
        _tag_idx += 1
        return ViraltestAction(
            action_type="post", content_type=ct, topic=topic, tags=tags
        )

    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 6: Competitor copycat (always post what competitors posted)
# ---------------------------------------------------------------------------
def agent_copycat(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.3:
        return ViraltestAction(action_type="rest")

    if obs.posts_today >= 3:
        return ViraltestAction(action_type="rest")

    comp = obs.competitor_recent_posts
    if comp and 9 <= obs.current_hour <= 20:
        latest = comp[0]
        return ViraltestAction(
            action_type="post",
            content_type=latest.get("content_type", "reel"),
            topic=latest.get("topic", "AI tools"),
            tags=latest.get("tags", ["ai"]),
        )

    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 7: Queue Optimizer (create content first, then post from queue)
# ---------------------------------------------------------------------------
_q7_phase = "prep"

def _reset_queue_state():
    global _q7_phase
    _q7_phase = "prep"

def agent_queue_optimizer(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _q7_phase
    if _q7_phase == "prep" and obs.content_queue_size < 4:
        return ViraltestAction(action_type="create_content")
    _q7_phase = "post"

    if obs.creator_energy < 0.35:
        if obs.content_queue_size < 2:
            return ViraltestAction(action_type="create_content")
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20 and obs.posts_today < 2 and obs.content_queue_size > 0:
        ct = _types[step % 4]
        topic = obs.trending_topics[0] if obs.trending_topics else "productivity"
        tags = list(obs.trending_tags[:2]) + ["growth"]
        return ViraltestAction(action_type="post", content_type=ct, topic=topic, tags=tags)

    if obs.content_queue_size < 3:
        return ViraltestAction(action_type="create_content")
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 8: Burst Poster (3 posts in a row, then rest until energy refills)
# ---------------------------------------------------------------------------
_burst_count = 0

def _reset_burst_state():
    global _burst_count
    _burst_count = 0

def agent_burst(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _burst_count
    if obs.creator_energy < 0.5:
        _burst_count = 0
        return ViraltestAction(action_type="rest")

    if _burst_count >= 3:
        _burst_count = 0
        return ViraltestAction(action_type="rest")

    _burst_count += 1
    return ViraltestAction(
        action_type="post",
        content_type=["reel", "carousel", "story"][_burst_count % 3],
        topic=obs.trending_topics[0] if obs.trending_topics else "coding",
        tags=list(obs.trending_tags[:2]) + ["tips"],
    )


# ---------------------------------------------------------------------------
# SCENARIO 9: Weekend Warrior (only posts on weekends)
# ---------------------------------------------------------------------------
def agent_weekend(obs: ViraltestObservation, step: int) -> ViraltestAction:
    is_weekend = obs.day_of_week in (5, 6)  # Sat, Sun
    if not is_weekend:
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.3 or obs.posts_today >= 3:
        return ViraltestAction(action_type="rest")

    return ViraltestAction(
        action_type="post",
        content_type="reel",
        topic=obs.trending_topics[0] if obs.trending_topics else "travel",
        tags=list(obs.trending_tags[:3]),
    )


# ---------------------------------------------------------------------------
# SCENARIO 10: Tag Explorer (tries a new tag combo every post, max discovery)
# ---------------------------------------------------------------------------
_te_idx = 0

def _reset_tag_explorer_state():
    global _te_idx
    _te_idx = 0

def agent_tag_explorer(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _te_idx
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 8 <= obs.current_hour <= 21:
        start = (_te_idx * 3) % len(TAG_POOL)
        tags = [TAG_POOL[(start + i) % len(TAG_POOL)] for i in range(3)]
        _te_idx += 1
        return ViraltestAction(
            action_type="post",
            content_type=_types[_te_idx % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "devtools",
            tags=tags,
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 11: Balanced Creator (create→post→rest cycle)
# ---------------------------------------------------------------------------
_bc_phase = 0

def _reset_balanced_state():
    global _bc_phase
    _bc_phase = 0

def agent_balanced(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _bc_phase
    cycle = _bc_phase % 3
    _bc_phase += 1

    if obs.creator_energy < 0.3:
        return ViraltestAction(action_type="rest")

    if cycle == 0:
        return ViraltestAction(action_type="create_content")
    elif cycle == 1 and 8 <= obs.current_hour <= 20:
        ct = _types[step % 4]
        topic = obs.trending_topics[0] if obs.trending_topics else "startup"
        tags = list(obs.trending_tags[:2]) + [TAG_POOL[step % len(TAG_POOL)]]
        return ViraltestAction(action_type="post", content_type=ct, topic=topic, tags=tags)
    else:
        return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 12: Sleep Deprived (never rests, tests sleep mechanic)
# ---------------------------------------------------------------------------
def agent_sleep_deprived(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Posts or creates content but never rests - tests sleep deprivation."""
    if 9 <= obs.current_hour <= 20 and obs.posts_today < 2:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "coding",
            tags=list(obs.trending_tags[:2]),
        )
    return ViraltestAction(action_type="create_content")


# ---------------------------------------------------------------------------
# SCENARIO 13: Sleep Conscious (rests during night hours 23-7)
# ---------------------------------------------------------------------------
def agent_sleep_conscious(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Rests during night hours to simulate proper sleep schedule."""
    if obs.current_hour >= 23 or obs.current_hour < 7:
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "wellness",
            tags=list(obs.trending_tags[:2]) + ["productivity"],
        )
    return ViraltestAction(action_type="create_content")


# ---------------------------------------------------------------------------
# SCENARIO 14: Minimal Poster (1 post per day only)
# ---------------------------------------------------------------------------
def agent_minimal(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.posts_today >= 1:
        return ViraltestAction(action_type="rest")

    if obs.current_hour == 12:  # Post at noon only
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic=obs.trending_topics[0] if obs.trending_topics else "minimalism",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 15: Story Spammer (only stories, low energy cost)
# ---------------------------------------------------------------------------
def agent_story_spammer(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.2:
        return ViraltestAction(action_type="rest")

    if obs.posts_today < 4 and 8 <= obs.current_hour <= 22:
        return ViraltestAction(
            action_type="post",
            content_type="story",
            topic="daily update",
            tags=["fitness", "wellness"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 16: Reel Maximizer (only reels for max reach)
# ---------------------------------------------------------------------------
def agent_reel_max(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 12 <= obs.current_hour <= 15:  # Peak hours only
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic=obs.trending_topics[0] if obs.trending_topics else "viral content",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 17: Text Only (low energy, low reach strategy)
# ---------------------------------------------------------------------------
def agent_text_only(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.3 or obs.posts_today >= 3:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 18:
        return ViraltestAction(
            action_type="post",
            content_type="text_post",
            topic="thoughts and tips",
            tags=["tips", "howto", "motivation"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 18: Early Bird (posts only 6-10am)
# ---------------------------------------------------------------------------
def agent_early_bird(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.3:
        return ViraltestAction(action_type="rest")

    if 6 <= obs.current_hour <= 10 and obs.posts_today < 2:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="morning routine",
            tags=["productivity", "wellness", "fitness"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 19: Night Owl (posts only 20-23)
# ---------------------------------------------------------------------------
def agent_night_owl(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.3:
        return ViraltestAction(action_type="rest")

    if 20 <= obs.current_hour <= 23 and obs.posts_today < 2:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="night thoughts",
            tags=["stoic", "motivation"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 20: Trend Chaser (only posts if trending topic available)
# ---------------------------------------------------------------------------
def agent_trend_chaser(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if obs.trending_topics and 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic=obs.trending_topics[0],
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 21: Anti-Trend (avoids trending, seeks differentiation)
# ---------------------------------------------------------------------------
def agent_anti_trend(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        non_trending_tags = [t for t in TAG_POOL if t not in obs.trending_tags][:3]
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="unique perspective on niche topic",
            tags=non_trending_tags if non_trending_tags else ["minimalism", "stoic"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 22: Energy Saver (only posts when energy > 0.7)
# ---------------------------------------------------------------------------
def agent_energy_saver(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.7:
        return ViraltestAction(action_type="rest")

    if obs.posts_today < 1 and 10 <= obs.current_hour <= 18:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic=obs.trending_topics[0] if obs.trending_topics else "productivity",
            tags=list(obs.trending_tags[:2]) + ["tips"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 23: Queue Heavy (builds queue first 3 days, posts rest)
# ---------------------------------------------------------------------------
_qh_day = -1

def _reset_queue_heavy_state():
    global _qh_day
    _qh_day = -1

def agent_queue_heavy(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.days_elapsed < 3:
        if obs.creator_energy < 0.3:
            return ViraltestAction(action_type="rest")
        return ViraltestAction(action_type="create_content")

    if obs.creator_energy < 0.35:
        return ViraltestAction(action_type="rest")

    if obs.content_queue_size > 0 and 9 <= obs.current_hour <= 20 and obs.posts_today < 2:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "growth",
            tags=list(obs.trending_tags[:2]) + ["viral"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 24: Midday Focus (only 11-14)
# ---------------------------------------------------------------------------
def agent_midday(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 11 <= obs.current_hour <= 14:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic=obs.trending_topics[0] if obs.trending_topics else "lunch break content",
            tags=list(obs.trending_tags[:2]) + ["howto"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 25: Random Actor (random valid actions)
# ---------------------------------------------------------------------------
_rng25 = stdlib_random.Random(123)

def agent_random(obs: ViraltestObservation, step: int) -> ViraltestAction:
    action = _rng25.choice(["post", "rest", "create_content"])
    if action == "post":
        return ViraltestAction(
            action_type="post",
            content_type=_rng25.choice(_types),
            topic=_rng25.choice(["random topic", "AI tools", "fitness", "travel"]),
            tags=_rng25.sample(TAG_POOL, 2),
        )
    elif action == "create_content":
        return ViraltestAction(action_type="create_content")
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 26: Carousel Only
# ---------------------------------------------------------------------------
def agent_carousel_only(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.35 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 18:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic=obs.trending_topics[0] if obs.trending_topics else "guide",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 27: Tech Niche Only
# ---------------------------------------------------------------------------
def agent_tech_niche(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic="AI tools and coding tips",
            tags=["ai", "coding", "devtools"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 28: Lifestyle Niche Only
# ---------------------------------------------------------------------------
def agent_lifestyle_niche(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="fitness routine and wellness tips",
            tags=["fitness", "wellness", "travel"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 29: High Frequency (3 posts/day target)
# ---------------------------------------------------------------------------
def agent_high_freq(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.25:
        return ViraltestAction(action_type="rest")

    if obs.posts_today < 3 and 8 <= obs.current_hour <= 21:
        return ViraltestAction(
            action_type="post",
            content_type=["story", "text_post", "reel"][obs.posts_today % 3],
            topic=obs.trending_topics[0] if obs.trending_topics else "daily update",
            tags=list(obs.trending_tags[:2]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 30: Low Frequency (1 post every 2 days)
# ---------------------------------------------------------------------------
def agent_low_freq(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.days_elapsed % 2 != 0:
        return ViraltestAction(action_type="rest")

    if obs.posts_today >= 1:
        return ViraltestAction(action_type="rest")

    if obs.current_hour == 14:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic=obs.trending_topics[0] if obs.trending_topics else "weekly roundup",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 31: Competitor Avoider (checks saturation before posting)
# ---------------------------------------------------------------------------
def agent_comp_avoider(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if obs.niche_saturation > 0.5:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="unique angle on trending topic",
            tags=list(obs.trending_tags[:2]) + ["growth"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 32: Tuesday Thursday Focus
# ---------------------------------------------------------------------------
def agent_tue_thu(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.day_of_week not in (1, 3):  # Tue=1, Thu=3
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 12 <= obs.current_hour <= 15:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic=obs.trending_topics[0] if obs.trending_topics else "midweek content",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 33: Monday Motivation
# ---------------------------------------------------------------------------
def agent_monday(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.day_of_week != 0:  # Mon=0
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.3 or obs.posts_today >= 3:
        return ViraltestAction(action_type="rest")

    if 8 <= obs.current_hour <= 18:
        return ViraltestAction(
            action_type="post",
            content_type=_types[obs.posts_today % 4],
            topic="monday motivation",
            tags=["motivation", "tips", "productivity"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 34: Tag Performance Exploiter
# ---------------------------------------------------------------------------
def agent_tag_exploiter(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        best_tags = sorted(obs.tag_performance.items(), key=lambda x: x[1], reverse=True)[:3]
        tags = [t[0] for t in best_tags] if best_tags else list(obs.trending_tags[:3])
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic=obs.trending_topics[0] if obs.trending_topics else "growth content",
            tags=tags,
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 35: Alternating Format (reel→carousel→story→text cycle)
# ---------------------------------------------------------------------------
_alt_idx = 0

def _reset_alternating_state():
    global _alt_idx
    _alt_idx = 0

def agent_alternating(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _alt_idx
    if obs.creator_energy < 0.35 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        ct = _types[_alt_idx % 4]
        _alt_idx += 1
        return ViraltestAction(
            action_type="post",
            content_type=ct,
            topic=obs.trending_topics[0] if obs.trending_topics else "varied content",
            tags=list(obs.trending_tags[:2]) + ["trending"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 36: Engagement Chaser (posts more when engagement high)
# ---------------------------------------------------------------------------
def agent_engagement_chaser(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.3:
        return ViraltestAction(action_type="rest")

    target_posts = 3 if obs.engagement_rate > 0.5 else 2 if obs.engagement_rate > 0.3 else 1

    if obs.posts_today >= target_posts:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic=obs.trending_topics[0] if obs.trending_topics else "engagement content",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 37: Conservative Energy (never goes below 0.5)
# ---------------------------------------------------------------------------
def agent_conservative(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.5:
        return ViraltestAction(action_type="rest")

    if obs.posts_today >= 1:
        return ViraltestAction(action_type="rest")

    if 12 <= obs.current_hour <= 14:
        return ViraltestAction(
            action_type="post",
            content_type="text_post",
            topic=obs.trending_topics[0] if obs.trending_topics else "quick tip",
            tags=list(obs.trending_tags[:2]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 38: Aggressive Energy (pushes until 0.15)
# ---------------------------------------------------------------------------
def agent_aggressive(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.15:
        return ViraltestAction(action_type="rest")

    if obs.posts_today >= 4:
        return ViraltestAction(action_type="rest")

    if 8 <= obs.current_hour <= 22:
        return ViraltestAction(
            action_type="post",
            content_type=_types[obs.posts_today % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "high output",
            tags=list(obs.trending_tags[:2]) + ["viral"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 39: Sleep Respecting (factors in hours_since_sleep)
# ---------------------------------------------------------------------------
def agent_sleep_respecting(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Rests when sleep deprived or during night hours."""
    if obs.hours_since_sleep >= 14 or obs.sleep_debt > 0.3:
        return ViraltestAction(action_type="rest")

    if obs.current_hour >= 22 or obs.current_hour < 8:
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "well-rested content",
            tags=list(obs.trending_tags[:2]) + ["wellness"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 40: Night Shift (posts 22-6, sleeps during day)
# ---------------------------------------------------------------------------
def agent_night_shift(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if 8 <= obs.current_hour <= 20:
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.3 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    return ViraltestAction(
        action_type="post",
        content_type="reel",
        topic="late night content",
        tags=["stoic", "motivation"],
    )


# ---------------------------------------------------------------------------
# SCENARIO 41: Split Schedule (morning and evening posts only)
# ---------------------------------------------------------------------------
def agent_split_schedule(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4:
        return ViraltestAction(action_type="rest")

    morning = 8 <= obs.current_hour <= 10
    evening = 18 <= obs.current_hour <= 20

    if not (morning or evening):
        return ViraltestAction(action_type="rest")

    if obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    return ViraltestAction(
        action_type="post",
        content_type="carousel" if morning else "reel",
        topic=obs.trending_topics[0] if obs.trending_topics else "daily content",
        tags=list(obs.trending_tags[:2]) + ["tips"],
    )


# ---------------------------------------------------------------------------
# SCENARIO 42: Follower Growth Focus
# ---------------------------------------------------------------------------
def agent_growth_focus(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.35 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 12 <= obs.current_hour <= 15:  # Peak reach hours
        return ViraltestAction(
            action_type="post",
            content_type="reel",  # Max reach
            topic=obs.trending_topics[0] if obs.trending_topics else "growth hacks",
            tags=["viral", "growth", "trending"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 43: Content Creator Mode (creates more than posts)
# ---------------------------------------------------------------------------
_ccm_ratio = 0

def _reset_content_creator_state():
    global _ccm_ratio
    _ccm_ratio = 0

def agent_content_creator(obs: ViraltestObservation, step: int) -> ViraltestAction:
    global _ccm_ratio
    if obs.creator_energy < 0.3:
        return ViraltestAction(action_type="rest")

    _ccm_ratio += 1
    if _ccm_ratio % 4 != 0:  # 3:1 create:post ratio
        return ViraltestAction(action_type="create_content")

    if obs.content_queue_size > 0 and 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "queued content",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="create_content")


# ---------------------------------------------------------------------------
# SCENARIO 44: Weekday Only
# ---------------------------------------------------------------------------
def agent_weekday_only(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.day_of_week >= 5:  # Weekend
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 18:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "weekday content",
            tags=list(obs.trending_tags[:2]) + ["productivity"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 45: Double Peak (posts at 9am and 3pm)
# ---------------------------------------------------------------------------
def agent_double_peak(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if obs.current_hour in (9, 15):
        return ViraltestAction(
            action_type="post",
            content_type="reel" if obs.current_hour == 9 else "carousel",
            topic=obs.trending_topics[0] if obs.trending_topics else "peak time content",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 46: Crypto/Web3 Niche
# ---------------------------------------------------------------------------
def agent_crypto_niche(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="crypto market analysis and web3 trends",
            tags=["crypto", "web3", "ai"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 47: Gaming Niche
# ---------------------------------------------------------------------------
def agent_gaming_niche(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 14 <= obs.current_hour <= 22:  # Gaming audience active later
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="gaming highlights and tips",
            tags=["gaming", "viral", "tips"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 48: Productivity Guru
# ---------------------------------------------------------------------------
def agent_productivity(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 6 <= obs.current_hour <= 10:  # Morning productivity crowd
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="productivity tips and systems",
            tags=["productivity", "tips", "howto"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 49: Food Content Creator
# ---------------------------------------------------------------------------
def agent_food_creator(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    meal_hours = obs.current_hour in (8, 12, 18)  # Breakfast, lunch, dinner
    if meal_hours:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="food recipe and cooking tips",
            tags=["food", "howto", "viral"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 50: Travel Blogger
# ---------------------------------------------------------------------------
def agent_travel(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 10 <= obs.current_hour <= 16:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="travel guide and destination highlights",
            tags=["travel", "photography", "trending"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 51: Fashion Content
# ---------------------------------------------------------------------------
def agent_fashion(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 11 <= obs.current_hour <= 19:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="fashion haul and style tips",
            tags=["fashion", "trending", "tips"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 52: Photography Focus
# ---------------------------------------------------------------------------
def agent_photography(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    golden_hours = obs.current_hour in (7, 8, 17, 18)  # Golden hour photography
    if golden_hours:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="photo editing tips and composition",
            tags=["photography", "tips", "howto"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 53: Stoic Philosophy
# ---------------------------------------------------------------------------
def agent_stoic(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 1:
        return ViraltestAction(action_type="rest")

    if obs.current_hour == 6:  # Early morning wisdom
        return ViraltestAction(
            action_type="post",
            content_type="text_post",
            topic="stoic wisdom and daily reflection",
            tags=["stoic", "motivation", "minimalism"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 54: Business/SaaS Focus
# ---------------------------------------------------------------------------
def agent_saas(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 17:  # Business hours
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="SaaS growth strategies and startup tips",
            tags=["saas", "startup", "growth"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 55: Creator Economy Focus
# ---------------------------------------------------------------------------
def agent_creator_economy(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 10 <= obs.current_hour <= 18:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="creator economy and monetization tips",
            tags=["growth", "viral", "tips"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 56: ML/AI Deep Dive
# ---------------------------------------------------------------------------
def agent_ml_deep(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="carousel",
            topic="machine learning concepts and AI tools",
            tags=["ml", "ai", "coding"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 57: Sleep Debt Aware (checks sleep_debt field)
# ---------------------------------------------------------------------------
def agent_sleep_debt_aware(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Takes mandatory rest when sleep debt accumulates."""
    if obs.sleep_debt > 0.2:
        return ViraltestAction(action_type="rest")

    if obs.current_hour >= 23 or obs.current_hour < 7:
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "balanced content",
            tags=list(obs.trending_tags[:3]),
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 58: Marathon Runner (tests long awake periods)
# ---------------------------------------------------------------------------
def agent_marathon(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Stays awake as long as possible, only rests at critical energy."""
    if obs.creator_energy < 0.1:
        return ViraltestAction(action_type="rest")

    if 6 <= obs.current_hour <= 23 and obs.posts_today < 3:
        return ViraltestAction(
            action_type="post",
            content_type="story",  # Low energy cost
            topic="marathon content session",
            tags=list(obs.trending_tags[:2]),
        )
    return ViraltestAction(action_type="create_content")


# ---------------------------------------------------------------------------
# SCENARIO 59: Nap Strategy (short rests throughout day)
# ---------------------------------------------------------------------------
_nap_count = 0

def _reset_nap_state():
    global _nap_count
    _nap_count = 0

def agent_napper(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Takes strategic naps throughout the day."""
    global _nap_count

    if obs.hours_since_sleep >= 6:
        _nap_count += 1
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "fresh content",
            tags=list(obs.trending_tags[:2]) + ["wellness"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# SCENARIO 60: Optimal Sleep Schedule (8hr sleep blocks)
# ---------------------------------------------------------------------------
def agent_optimal_sleep(obs: ViraltestObservation, step: int) -> ViraltestAction:
    """Follows an optimal 8-hour sleep schedule."""
    # Sleep from 23:00 to 07:00
    if obs.current_hour >= 23 or obs.current_hour < 7:
        return ViraltestAction(action_type="rest")

    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    # Post during peak hours
    if 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type=_types[step % 4],
            topic=obs.trending_topics[0] if obs.trending_topics else "well-rested creator",
            tags=list(obs.trending_tags[:2]) + ["productivity"],
        )
    return ViraltestAction(action_type="create_content")


# ---------------------------------------------------------------------------
# SCENARIO 61: Events/News Chaser
# ---------------------------------------------------------------------------
def agent_events(obs: ViraltestObservation, step: int) -> ViraltestAction:
    if obs.creator_energy < 0.4 or obs.posts_today >= 2:
        return ViraltestAction(action_type="rest")

    event_tags = [t for t in obs.trending_tags if t in ["worldcup", "election", "oscars", "newyear", "climate"]]
    if event_tags and 9 <= obs.current_hour <= 20:
        return ViraltestAction(
            action_type="post",
            content_type="reel",
            topic="breaking news and event coverage",
            tags=event_tags[:2] + ["trending"],
        )
    return ViraltestAction(action_type="rest")


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

SCENARIOS: List[Tuple[str, Callable, str]] = [
    ("SCENARIO 1: Always Rest", agent_always_rest,
     "Tests: zero engagement, no growth, energy stays max, grader should score near 0"),
    ("SCENARIO 2: Spam Post (same reel every step)", agent_spam,
     "Tests: rapid burnout (4 steps), grader=0 for competitive, anti-exploit working"),
    ("SCENARIO 3: Bad Timing (post at night, boring topics)", agent_bad_timing,
     "Tests: off-peak hours, non-trending topics, low engagement multipliers"),
    ("SCENARIO 4: No Rest (diverse posts but never rests)", agent_no_rest,
     "Tests: varied content but no energy management, burns out mid-week"),
    ("SCENARIO 5: Smart Agent (optimal strategy)", agent_smart,
     "Tests: peak hours, trending topics, varied types/tags, energy management"),
    ("SCENARIO 6: Competitor Copycat (copy rival topics)", agent_copycat,
     "Tests: high niche saturation, low differentiation, engagement penalty"),
    ("SCENARIO 7: Queue Optimizer (create→post from queue)", agent_queue_optimizer,
     "Tests: queue mechanic, half-cost posting, energy efficiency"),
    ("SCENARIO 8: Burst Poster (3 posts then rest)", agent_burst,
     "Tests: bursty posting pattern, energy recovery between bursts"),
    ("SCENARIO 9: Weekend Warrior (only posts Sat/Sun)", agent_weekend,
     "Tests: limited posting days, consistency penalty, missed weekday opportunities"),
    ("SCENARIO 10: Tag Explorer (new tags every post)", agent_tag_explorer,
     "Tests: tag discovery vs exploitation tradeoff"),
    ("SCENARIO 11: Balanced Creator (create→post→rest)", agent_balanced,
     "Tests: queue usage, pacing, all 3 action types in rotation"),
    ("SCENARIO 12: Sleep Deprived (never rests)", agent_sleep_deprived,
     "Tests: sleep deprivation mechanic, quality degradation over time"),
    ("SCENARIO 13: Sleep Conscious (rests at night)", agent_sleep_conscious,
     "Tests: proper sleep schedule, maintains quality and energy"),
    ("SCENARIO 14: Minimal Poster (1/day)", agent_minimal,
     "Tests: low frequency posting, high rest, consistency"),
    ("SCENARIO 15: Story Spammer", agent_story_spammer,
     "Tests: low-energy content spam, story reach limits"),
    ("SCENARIO 16: Reel Maximizer", agent_reel_max,
     "Tests: high-reach content focus, energy management"),
    ("SCENARIO 17: Text Only", agent_text_only,
     "Tests: low-energy low-reach strategy"),
    ("SCENARIO 18: Early Bird (6-10am)", agent_early_bird,
     "Tests: off-peak morning hours"),
    ("SCENARIO 19: Night Owl (20-23)", agent_night_owl,
     "Tests: late evening posting"),
    ("SCENARIO 20: Trend Chaser", agent_trend_chaser,
     "Tests: trending topic dependency"),
    ("SCENARIO 21: Anti-Trend", agent_anti_trend,
     "Tests: differentiation strategy, non-trending tags"),
    ("SCENARIO 22: Energy Saver (>0.7 only)", agent_energy_saver,
     "Tests: conservative energy management"),
    ("SCENARIO 23: Queue Heavy (3 days prep)", agent_queue_heavy,
     "Tests: heavy queue building strategy"),
    ("SCENARIO 24: Midday Focus (11-14)", agent_midday,
     "Tests: peak hour targeting"),
    ("SCENARIO 25: Random Actor", agent_random,
     "Tests: baseline random performance"),
    ("SCENARIO 26: Carousel Only", agent_carousel_only,
     "Tests: single high-engagement format"),
    ("SCENARIO 27: Tech Niche", agent_tech_niche,
     "Tests: niche focus with tech tags"),
    ("SCENARIO 28: Lifestyle Niche", agent_lifestyle_niche,
     "Tests: lifestyle content strategy"),
    ("SCENARIO 29: High Frequency (3/day)", agent_high_freq,
     "Tests: high post count, fatigue effects"),
    ("SCENARIO 30: Low Frequency (1 per 2 days)", agent_low_freq,
     "Tests: very low posting rate"),
    ("SCENARIO 31: Competitor Avoider", agent_comp_avoider,
     "Tests: saturation awareness"),
    ("SCENARIO 32: Tuesday Thursday Focus", agent_tue_thu,
     "Tests: peak weekday targeting"),
    ("SCENARIO 33: Monday Motivation", agent_monday,
     "Tests: single day focus"),
    ("SCENARIO 34: Tag Exploiter", agent_tag_exploiter,
     "Tests: tag performance optimization"),
    ("SCENARIO 35: Alternating Format", agent_alternating,
     "Tests: content type rotation"),
    ("SCENARIO 36: Engagement Chaser", agent_engagement_chaser,
     "Tests: adaptive posting based on engagement"),
    ("SCENARIO 37: Conservative Energy", agent_conservative,
     "Tests: very safe energy strategy"),
    ("SCENARIO 38: Aggressive Energy", agent_aggressive,
     "Tests: push limits on energy"),
    ("SCENARIO 39: Sleep Respecting", agent_sleep_respecting,
     "Tests: hours_since_sleep awareness"),
    ("SCENARIO 40: Night Shift", agent_night_shift,
     "Tests: inverted sleep schedule"),
    ("SCENARIO 41: Split Schedule", agent_split_schedule,
     "Tests: morning and evening split"),
    ("SCENARIO 42: Growth Focus", agent_growth_focus,
     "Tests: follower growth optimization"),
    ("SCENARIO 43: Content Creator Mode", agent_content_creator,
     "Tests: heavy content creation ratio"),
    ("SCENARIO 44: Weekday Only", agent_weekday_only,
     "Tests: no weekend posting"),
    ("SCENARIO 45: Double Peak", agent_double_peak,
     "Tests: two optimal time slots"),
    ("SCENARIO 46: Crypto/Web3 Niche", agent_crypto_niche,
     "Tests: crypto tag focus"),
    ("SCENARIO 47: Gaming Niche", agent_gaming_niche,
     "Tests: gaming audience timing"),
    ("SCENARIO 48: Productivity Guru", agent_productivity,
     "Tests: morning productivity focus"),
    ("SCENARIO 49: Food Creator", agent_food_creator,
     "Tests: meal-time posting"),
    ("SCENARIO 50: Travel Blogger", agent_travel,
     "Tests: travel content strategy"),
    ("SCENARIO 51: Fashion Content", agent_fashion,
     "Tests: fashion niche timing"),
    ("SCENARIO 52: Photography Focus", agent_photography,
     "Tests: golden hour timing"),
    ("SCENARIO 53: Stoic Philosophy", agent_stoic,
     "Tests: minimal daily wisdom posts"),
    ("SCENARIO 54: SaaS/Business", agent_saas,
     "Tests: B2B content timing"),
    ("SCENARIO 55: Creator Economy", agent_creator_economy,
     "Tests: creator monetization focus"),
    ("SCENARIO 56: ML/AI Deep Dive", agent_ml_deep,
     "Tests: technical content strategy"),
    ("SCENARIO 57: Sleep Debt Aware", agent_sleep_debt_aware,
     "Tests: sleep_debt field awareness"),
    ("SCENARIO 58: Marathon Runner", agent_marathon,
     "Tests: extended awake periods, sleep deprivation effects"),
    ("SCENARIO 59: Napper", agent_napper,
     "Tests: strategic short rests throughout day"),
    ("SCENARIO 60: Optimal Sleep", agent_optimal_sleep,
     "Tests: 8-hour sleep block strategy"),
    ("SCENARIO 61: Events/News Chaser", agent_events,
     "Tests: event-based trending tags"),
]

if __name__ == "__main__":
    print("=" * 70)
    print("VIRALTEST — EDGE CASE & SCENARIO TESTS")
    print("=" * 70)
    print()

    def _reset_all():
        _reset_smart_state()
        _reset_queue_state()
        _reset_burst_state()
        _reset_tag_explorer_state()
        _reset_balanced_state()
        _reset_queue_heavy_state()
        _reset_alternating_state()
        _reset_content_creator_state()
        _reset_nap_state()

    for scenario_name, agent_fn, description in SCENARIOS:
        print("=" * 70)
        print(f"{scenario_name}")
        print(f"  {description}")
        print("=" * 70)
        print()

        for task in TASKS:
            _reset_all()
            run_episode(task, agent_fn, scenario_name)

        print()

    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Scenario':<50} {'Engage':>8} {'Strategic':>10} {'Competitive':>12}")
    print("-" * 82)

    for scenario_name, agent_fn, _ in SCENARIOS:
        scores = []
        for task in TASKS:
            _reset_all()
            env = ViraltestEnvironment()
            obs = env.reset(task=task, seed=SEED)
            for step in range(1, 169):
                obs = env.step(agent_fn(obs, step))
                if obs.done:
                    break
            scores.append((obs.metadata or {}).get("grader_score", 0.0))
        print(f"{scenario_name:<50} {scores[0]:>8.4f} {scores[1]:>10.4f} {scores[2]:>12.4f}")

    print()
    print("EXPECTED: Smart/Queue/Balanced should score highest.")
    print("Burnout agents (spam, no_rest) should score near 0 on strategic/competitive.")
