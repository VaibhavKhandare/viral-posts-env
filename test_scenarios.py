"""
Viraltest — Edge Case & Scenario Tests (Daily Plan Format)
Runs scenarios for all 3 tasks using the new daily step format.
Each step = one full day. Agent submits a sparse daily plan.
"""

import random as stdlib_random
from typing import Any, Callable, Dict, List, Optional, Tuple

from models import (
    CollabProposal,
    DailyInteractions,
    ScheduledAction,
    ViraltestAction,
)
from server.viraltest_environment import (
    TAG_POOL,
    ViraltestEnvironment,
    ViraltestObservation,
)

TASKS = ["weekly_engage", "weekly_strategic", "weekly_competitive"]
SEED = 42

_CONTENT_TYPES = ["reel", "carousel", "story", "text_post"]
_TOPICS = ["AI tools", "fitness routine", "growth hacks", "travel guide", "food recipe", "wellness tips"]
_rng = stdlib_random.Random(99)


def _plan(
    actions: list,
    collab: Optional[CollabProposal] = None,
    interactions: Optional[DailyInteractions] = None,
) -> ViraltestAction:
    return ViraltestAction(
        scheduled_actions=[ScheduledAction(**a) for a in actions],
        collab=collab,
        interactions=interactions,
    )


def run_episode(
    task: str,
    plan_fn: Callable[[Dict, int], ViraltestAction],
    label: str,
    user_niche: Optional[str] = None,
) -> float:
    env = ViraltestEnvironment()
    reset_kwargs: Dict[str, Any] = {"task": task, "seed": SEED}
    if user_niche:
        reset_kwargs["user_niche"] = user_niche
    obs = env.reset(**reset_kwargs)
    obs_dict = obs.model_dump()
    rewards: List[float] = []
    min_energy = 1.0
    burned_out = False

    for day in range(1, 9):
        action = plan_fn(obs_dict, day)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        r = obs.reward if obs.reward is not None else 0.0
        rewards.append(r)
        min_energy = min(min_energy, obs.creator_energy)
        if obs.done and obs.creator_energy <= 0:
            burned_out = True
        if obs.done:
            break

    score = (obs.metadata or {}).get("grader_score", 0.0)
    total_steps = len(rewards)

    print(f"  Task: {task}")
    print(f"  Days: {total_steps} | Done: {obs.done} | Burned out: {burned_out}")
    print(f"  Score: {score:.4f} | Total reward: {sum(rewards):.2f} | Avg reward: {sum(rewards)/len(rewards):.3f}")
    print(f"  Energy: {obs.creator_energy:.2f} | Min energy: {min_energy:.2f}")
    print(f"  Followers: {obs.follower_count} (started 10000, delta {obs.follower_count - 10000:+d})")
    print(f"  Engagement rate: {obs.engagement_rate:.4f}")
    print(f"  Unique tags: {len(obs.tag_performance)}")
    print(f"  Niche saturation: {obs.niche_saturation:.3f}")
    print()
    return score


def plan_always_rest(obs: dict, day: int) -> ViraltestAction:
    return _plan([])


def plan_spam(obs: dict, day: int) -> ViraltestAction:
    return _plan([{"hour": h, "action_type": "post", "content_type": "reel",
                   "topic": "AI tools", "tags": ["ai"]} for h in range(24)])


def plan_smart(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    t_tags = list((obs.get("trending_tags") or [])[:2])
    pool_tag = TAG_POOL[(day * 2) % len(TAG_POOL)]
    pool_tag2 = TAG_POOL[(day * 2 + 1) % len(TAG_POOL)]
    ct1 = _CONTENT_TYPES[(day * 2) % 4]
    ct2 = _CONTENT_TYPES[(day * 2 + 1) % 4]
    return _plan([
        {"hour": 8, "action_type": "create_content"},
        {"hour": 12, "action_type": "post", "content_type": ct1, "topic": trending, "tags": t_tags + [pool_tag]},
        {"hour": 19, "action_type": "post", "content_type": ct2, "topic": trending, "tags": t_tags + [pool_tag2]},
    ])


def plan_no_rest(obs: dict, day: int) -> ViraltestAction:
    actions = []
    for h in range(24):
        ct = _CONTENT_TYPES[h % 4]
        topic = _rng.choice(_TOPICS)
        tags = _rng.sample(TAG_POOL, 3)
        actions.append({"hour": h, "action_type": "post", "content_type": ct, "topic": topic, "tags": tags})
    return _plan(actions)


def plan_minimal(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["minimalism"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _plan([
        {"hour": 12, "action_type": "post", "content_type": "carousel", "topic": trending, "tags": tags},
    ])


def plan_tag_explorer(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["devtools"])[0]
    start = (day * 6) % len(TAG_POOL)
    tags1 = [TAG_POOL[(start + i) % len(TAG_POOL)] for i in range(3)]
    tags2 = [TAG_POOL[(start + 3 + i) % len(TAG_POOL)] for i in range(3)]
    ct1 = _CONTENT_TYPES[(day * 2) % 4]
    ct2 = _CONTENT_TYPES[(day * 2 + 1) % 4]
    return _plan([
        {"hour": 10, "action_type": "post", "content_type": ct1, "topic": trending, "tags": tags1},
        {"hour": 18, "action_type": "post", "content_type": ct2, "topic": trending, "tags": tags2},
    ])


def plan_queue_optimizer(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["productivity"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["growth"]
    queue = obs.get("content_queue_size", 0)
    if day < 3 or queue < 2:
        return _plan([
            {"hour": 8, "action_type": "create_content"},
            {"hour": 10, "action_type": "create_content"},
            {"hour": 14, "action_type": "create_content"},
        ])
    ct = _CONTENT_TYPES[day % 4]
    return _plan([
        {"hour": 12, "action_type": "post", "content_type": ct, "topic": trending, "tags": tags},
        {"hour": 19, "action_type": "post", "content_type": _CONTENT_TYPES[(day + 1) % 4], "topic": trending, "tags": tags},
    ])


def plan_double_peak(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["peak time content"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _plan([
        {"hour": 9, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
        {"hour": 15, "action_type": "post", "content_type": "carousel", "topic": trending, "tags": tags},
    ])


def plan_random(obs: dict, day: int) -> ViraltestAction:
    actions = []
    for h in range(24):
        r = _rng.random()
        if r < 0.1:
            ct = _rng.choice(_CONTENT_TYPES)
            topic = _rng.choice(["random topic", "AI tools", "fitness", "travel"])
            tags = _rng.sample(TAG_POOL, 2)
            actions.append({"hour": h, "action_type": "post", "content_type": ct, "topic": topic, "tags": tags})
        elif r < 0.15:
            actions.append({"hour": h, "action_type": "create_content"})
    return _plan(actions)


# ---------------------------------------------------------------------------
# Collab grid scenarios — user_niche set on env.reset(...) by run_episode.
# Each picks a partner_id intended to land in a specific (same/diff x low/high) tier
# and proposes the collab on day 5.
# ---------------------------------------------------------------------------

def _collab_plan(day: int, partner_id: str, hour: int = 12) -> ViraltestAction:
    """Daily plan that posts once and proposes a collab on days 3 and 6 of the week.

    Single-post per day keeps engagement below the theoretical_max cap so collab
    multipliers visibly bend the final grader score and follower count.
    """
    actions = [
        {"hour": hour, "action_type": "post", "content_type": "reel",
         "topic": "AI tools", "tags": ["ai"], "intent": "watch_bait"},
    ]
    collab = None
    if day in (3, 6):
        collab = CollabProposal(partner_id=partner_id, content_type="reel", hour=hour)
    return _plan(actions, collab=collab)


def plan_collab_same_low(obs: dict, day: int) -> ViraltestAction:
    # user_niche=tech, partner=b2b_thought_leader (NICHE differs but matrix overlap=0.08)
    # Use niche_expert (tech) which has overlap=0.10 with user_creator => same niche, low overlap.
    return _collab_plan(day, partner_id="niche_expert")


def plan_collab_same_high(obs: dict, day: int) -> ViraltestAction:
    # Force same niche + high overlap by setting user_niche=lifestyle and pairing with viral_chaser (overlap=0.55).
    return _collab_plan(day, partner_id="viral_chaser")


def plan_collab_diff_low(obs: dict, day: int) -> ViraltestAction:
    # user_niche=tech, partner=lifestyle_blogger (overlap=0.40 — actually high), pick travel_creator overlap=0.30 instead.
    return _collab_plan(day, partner_id="travel_creator")


def plan_collab_diff_high(obs: dict, day: int) -> ViraltestAction:
    # user_niche=tech, partner=lifestyle_blogger (overlap=0.40, diff niche).
    return _collab_plan(day, partner_id="lifestyle_blogger")


def plan_collab_blocked_zero(obs: dict, day: int) -> ViraltestAction:
    # b2b_thought_leader has overlap=0.08 with user_creator -> intersection_below_10pct guardrail.
    return _collab_plan(day, partner_id="b2b_thought_leader")


# ---------------------------------------------------------------------------
# Interaction scenarios — exercise the 5 penalty paths and the healthy band.
# ---------------------------------------------------------------------------

def _post_only_actions() -> list:
    return [
        {"hour": 12, "action_type": "post", "content_type": "reel",
         "topic": "AI tools", "tags": ["ai"], "intent": "watch_bait"},
    ]


def plan_interact_balanced(obs: dict, day: int) -> ViraltestAction:
    interactions = DailyInteractions(
        likes_on_others=12, comments_on_others=5, replies_to_audience=3,
        target_partner_ids=["niche_expert"], avg_reply_quality=0.8,
    )
    return _plan(_post_only_actions(), interactions=interactions)


def plan_interact_spam(obs: dict, day: int) -> ViraltestAction:
    interactions = DailyInteractions(
        likes_on_others=80, comments_on_others=40, replies_to_audience=0,
        target_partner_ids=["niche_expert"], avg_reply_quality=0.4,
    )
    return _plan(_post_only_actions(), interactions=interactions)


def plan_interact_ignoring_own(obs: dict, day: int) -> ViraltestAction:
    interactions = DailyInteractions(
        likes_on_others=8, comments_on_others=4, replies_to_audience=0,
        target_partner_ids=["niche_expert"], avg_reply_quality=0.6,
    )
    return _plan(_post_only_actions(), interactions=interactions)


def plan_interact_off_niche(obs: dict, day: int) -> ViraltestAction:
    interactions = DailyInteractions(
        likes_on_others=10, comments_on_others=5, replies_to_audience=2,
        target_partner_ids=["food_creator", "fitness_coach", "travel_creator", "lifestyle_blogger"],
        avg_reply_quality=0.7,
    )
    return _plan(_post_only_actions(), interactions=interactions)


def plan_interact_low_quality(obs: dict, day: int) -> ViraltestAction:
    interactions = DailyInteractions(
        likes_on_others=10, comments_on_others=5, replies_to_audience=8,
        target_partner_ids=["niche_expert"], avg_reply_quality=0.05,
    )
    return _plan(_post_only_actions(), interactions=interactions)


# Scenario tuple: (label, plan_fn, description, user_niche)
SCENARIOS: List[Tuple[str, Callable, str, Optional[str]]] = [
    ("Always Rest", plan_always_rest, "Zero engagement, no growth, energy stays max", None),
    ("Spam Post", plan_spam, "Post every hour, burns out instantly", None),
    ("Smart Agent", plan_smart, "Peak hours, trending, varied types, energy management", None),
    ("No Rest", plan_no_rest, "Post every hour, never rests, burns out", None),
    ("Minimal Poster", plan_minimal, "1 carousel at noon per day", None),
    ("Tag Explorer", plan_tag_explorer, "Rotates through tag pool for max discovery", None),
    ("Queue Optimizer", plan_queue_optimizer, "Creates content first, posts from queue", None),
    ("Double Peak", plan_double_peak, "Posts at 9am and 3pm", None),
    ("Random Actor", plan_random, "Random sparse actions each day", None),
    # Collab grid: 2x2 same/diff niche x low/high overlap + zero-guardrail.
    ("Collab Same-Niche Low Overlap", plan_collab_same_low,
     "user_niche=tech + niche_expert (same niche, overlap 0.10) — should yield HIGH boost.", "tech"),
    ("Collab Same-Niche High Overlap", plan_collab_same_high,
     "user_niche=lifestyle + viral_chaser (same niche, overlap 0.55) — penalty path: redundant audience.", "lifestyle"),
    ("Collab Diff-Niche Low Overlap", plan_collab_diff_low,
     "user_niche=tech + travel_creator (diff niche, overlap 0.30) — capped below same-niche-low.", "tech"),
    ("Collab Diff-Niche High Overlap", plan_collab_diff_high,
     "user_niche=tech + lifestyle_blogger (diff niche, overlap 0.40) — LOW reward (mismatch).", "tech"),
    ("Collab Guardrail Block", plan_collab_blocked_zero,
     "user_niche=tech + b2b_thought_leader (overlap 0.08 < 10%) — guardrail trips, forced penalty applied.", "tech"),
    # Interaction grid: healthy + 4 penalty paths.
    ("Interact Balanced", plan_interact_balanced,
     "Healthy daily likes/comments/replies on-niche.", "tech"),
    ("Interact Spam", plan_interact_spam,
     "80 likes + 40 comments — spam path, shadowban_risk + reach penalty.", "tech"),
    ("Interact Ignoring Own", plan_interact_ignoring_own,
     "Zero replies to own audience — compounding loyalty drop.", "tech"),
    ("Interact Off-Niche", plan_interact_off_niche,
     "All interactions targeted at non-tech creators — reach penalty.", "tech"),
    ("Interact Low-Quality", plan_interact_low_quality,
     "Replies with quality=0.05 — replies discounted + extra reward penalty.", "tech"),
]


if __name__ == "__main__":
    print("=" * 70)
    print("VIRALTEST — DAILY PLAN SCENARIO TESTS")
    print("=" * 70)
    print()

    for scenario_name, plan_fn, description, user_niche in SCENARIOS:
        print("=" * 70)
        print(f"{scenario_name}")
        print(f"  {description}")
        if user_niche:
            print(f"  user_niche={user_niche}")
        print("=" * 70)
        print()

        for task in TASKS:
            _rng = stdlib_random.Random(99)
            run_episode(task, plan_fn, scenario_name, user_niche=user_niche)

        print()

    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Scenario':<35} {'Engage':>8} {'Strategic':>10} {'Competitive':>12}")
    print("-" * 67)

    for scenario_name, plan_fn, _, user_niche in SCENARIOS:
        scores = []
        for task in TASKS:
            _rng = stdlib_random.Random(99)
            env = ViraltestEnvironment()
            reset_kwargs: Dict[str, Any] = {"task": task, "seed": SEED}
            if user_niche:
                reset_kwargs["user_niche"] = user_niche
            obs = env.reset(**reset_kwargs)
            obs_dict = obs.model_dump()
            for day in range(1, 9):
                action = plan_fn(obs_dict, day)
                obs = env.step(action)
                obs_dict = obs.model_dump()
                if obs.done:
                    break
            scores.append((obs.metadata or {}).get("grader_score", 0.0))
        print(f"{scenario_name:<35} {scores[0]:>8.4f} {scores[1]:>10.4f} {scores[2]:>12.4f}")

    print()
    print("EXPECTED: Smart/Queue/Tag Explorer should score highest.")
    print("Burnout agents (spam, no_rest) should score near 0 on strategic/competitive.")
    print("Collab Same-Niche Low Overlap should outperform any Diff-Niche collab.")
    print("Interact Spam/Off-Niche/Ignoring/Low-Quality should underperform Balanced.")
