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
