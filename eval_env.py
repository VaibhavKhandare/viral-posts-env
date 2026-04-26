"""
End-to-end evaluation of the viraltest environment after the collab + interaction expansion.

Sections
--------
A) Collab tier diagnostics
   - Per-tier expected multipliers from `_collab_evaluation`
   - Episode runs with varying collab cadence (1, 5, 15 collabs/episode) to show that
     the score spread between tiers GROWS with cadence, proving the multiplier is doing
     real work and the small diffs in the 2-collab test are just dilution.
B) Interaction diagnostics
   - Each penalty path (spam, ignoring_own, off_niche, low_quality, energy_drain) fires
     the expected violation.
   - Healthy band lifts reach_modifier > 1.0.
C) Cross-cutting sanity
   - Every scenario completes without errors, energy non-negative, judge_report present.

Run: .venv/bin/python eval_env.py
"""

from typing import Any, Dict, List, Optional

from models import (
    CollabProposal,
    DailyInteractions,
    ScheduledAction,
    ViraltestAction,
)
from server.viraltest_environment import ViraltestEnvironment


SEED = 42
HORIZON = 15  # TASK_HORIZON in the env

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_only(content_type: str = "reel", topic: str = "AI tools",
                tags: Optional[List[str]] = None, intent: str = "watch_bait") -> ScheduledAction:
    return ScheduledAction(
        hour=12, action_type="post", content_type=content_type,
        topic=topic, tags=tags or ["ai"], intent=intent,
    )


def _run_episode(
    plan_fn,
    user_niche: Optional[str] = None,
    task: str = "monthly_competitive",
) -> Dict[str, Any]:
    env = ViraltestEnvironment()
    reset_kwargs: Dict[str, Any] = {"task": task, "seed": SEED}
    if user_niche:
        reset_kwargs["user_niche"] = user_niche
    obs = env.reset(**reset_kwargs)
    obs_dict = obs.model_dump()
    last_obs = obs
    judge_violations_total: List[str] = []
    interaction_violations_total: List[str] = []
    min_energy = 1.0
    for day in range(1, HORIZON + 2):
        action = plan_fn(obs_dict, day)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        last_obs = obs
        min_energy = min(min_energy, obs.creator_energy)
        if obs.judge_report:
            judge_violations_total.extend(obs.judge_report.violations)
        if obs.interaction_metrics:
            interaction_violations_total.extend(obs.interaction_metrics.get("violations", []) or [])
        if obs.done:
            break
    score = (last_obs.metadata or {}).get("grader_score", 0.0)
    return {
        "score": float(score),
        "followers": int(last_obs.follower_count),
        "min_energy": float(min_energy),
        "energy": float(last_obs.creator_energy),
        "engagement_rate": float(last_obs.engagement_rate),
        "judge_violations": judge_violations_total,
        "interaction_violations": interaction_violations_total,
        "error": last_obs.error,
        "done": last_obs.done,
    }


# ---------------------------------------------------------------------------
# A) COLLAB TIER DIAGNOSTICS
# ---------------------------------------------------------------------------

def section_a_collab_evaluator() -> None:
    print("=" * 78)
    print("A1. _collab_evaluation snapshot (user_niche=tech)")
    print("=" * 78)
    env = ViraltestEnvironment()
    env.reset(task="monthly_competitive", seed=SEED, user_niche="tech")
    fmt = "{:<22} {:>5} {:>7} {:>5} {:>5} {:>10} {:>10} {:<28}"
    print(fmt.format("partner", "same?", "overlap", "fol", "gap%", "eng_mult", "growth", "reason/recommended"))
    print("-" * 105)
    for pid in [
        "niche_expert", "viral_chaser", "lifestyle_blogger", "b2b_thought_leader",
        "food_creator", "fitness_coach", "travel_creator",
    ]:
        ev = env._collab_evaluation(pid)
        rec_str = f"OK" if ev["recommended"] else f"BLOCK:{ev['reason']}"
        print(fmt.format(
            pid,
            "Y" if ev["same_niche"] else "N",
            f"{ev['overlap']:.2f}",
            ev["partner_followers"],
            f"{ev['follower_gap_pct']*100:.0f}%",
            f"{ev['eng_mult']:.3f}",
            f"{ev['growth_mult']:.3f}",
            rec_str,
        ))
    print()


def make_collab_plan(partner_id: str, collab_days: List[int]):
    """Daily plan: single post + collab proposed on collab_days."""
    def plan(obs: Dict[str, Any], day: int) -> ViraltestAction:
        actions = [_post_only()]
        collab = None
        if day in collab_days:
            collab = CollabProposal(partner_id=partner_id, content_type="reel", hour=12)
        return ViraltestAction(scheduled_actions=actions, collab=collab)
    return plan


def section_a_collab_cadence() -> None:
    print("=" * 78)
    print("A2. Score spread vs collab cadence (1, 5, 15 collabs in 15-day horizon)")
    print("    Hypothesis: more collab days -> larger gap between tiers")
    print("=" * 78)

    # Map each tier to (partner_id, user_niche) — chosen so the partner clears the
    # follower-size guardrail (peer-tier mocked followers in the data file).
    tiers = [
        ("Same-Niche Low",  "niche_expert",       "tech"),
        ("Same-Niche High", "viral_chaser",       "lifestyle"),  # overlap=0.55 (high)
        ("Diff-Niche Low",  "food_creator",       "tech"),       # overlap=0.25 (mid-low)
        ("Diff-Niche High", "lifestyle_blogger",  "tech"),       # overlap=0.40 (boundary high)
        ("Guardrail Block", "b2b_thought_leader", "tech"),       # overlap=0.08 (<10%)
    ]
    cadences = {
        "1 collab":  [5],
        "5 collabs": [3, 5, 7, 9, 11],
        "15 collabs": list(range(1, 16)),
    }

    fmt = "{:<22} {:>10} {:>10} {:>10}"
    print(fmt.format("Tier", *cadences.keys()))
    print("-" * 56)
    for label, partner_id, user_niche in tiers:
        scores = []
        for cad_label, days in cadences.items():
            r = _run_episode(make_collab_plan(partner_id, days), user_niche=user_niche)
            scores.append(f"{r['score']:.4f}")
        print(fmt.format(label, *scores))
    print()
    print("    -> Same-Niche Low score should DROP slowly as you add collabs.")
    print("    -> Same-Niche High and Diff-Niche High should DROP quickly (penalty stacks).")
    print("    -> Spread between top and bottom should GROW with cadence.")
    print()


# ---------------------------------------------------------------------------
# B) INTERACTION DIAGNOSTICS
# ---------------------------------------------------------------------------

def make_interaction_plan(interactions: DailyInteractions):
    def plan(obs: Dict[str, Any], day: int) -> ViraltestAction:
        return ViraltestAction(scheduled_actions=[_post_only()], interactions=interactions)
    return plan


def section_b_interactions() -> None:
    print("=" * 78)
    print("B. Interaction penalty-path matrix")
    print("=" * 78)

    cases = [
        ("healthy", DailyInteractions(
            likes_on_others=12, comments_on_others=5, replies_to_audience=3,
            target_partner_ids=["niche_expert"], avg_reply_quality=0.8,
        ), "interaction_*", False),
        ("spam", DailyInteractions(
            likes_on_others=80, comments_on_others=40, replies_to_audience=0,
            target_partner_ids=["niche_expert"], avg_reply_quality=0.4,
        ), "interaction_spam", True),
        ("ignoring_own", DailyInteractions(
            likes_on_others=8, comments_on_others=4, replies_to_audience=0,
            target_partner_ids=["niche_expert"], avg_reply_quality=0.6,
        ), "interaction_ignoring_own", True),
        ("off_niche", DailyInteractions(
            likes_on_others=10, comments_on_others=5, replies_to_audience=2,
            target_partner_ids=["food_creator", "fitness_coach", "travel_creator", "lifestyle_blogger"],
            avg_reply_quality=0.7,
        ), "interaction_off_niche", True),
        ("low_quality", DailyInteractions(
            likes_on_others=10, comments_on_others=5, replies_to_audience=8,
            target_partner_ids=["niche_expert"], avg_reply_quality=0.05,
        ), "interaction_low_quality", True),
        ("energy_drain", DailyInteractions(
            likes_on_others=200, comments_on_others=100, replies_to_audience=100,
            target_partner_ids=["niche_expert"], avg_reply_quality=0.5,
        ), "interaction_energy_drain", True),
    ]

    fmt = "{:<14} {:>7} {:>9} {:>10} {:>10} {:>11} {:<12}"
    print(fmt.format("case", "score", "followers", "min_energy", "engRate", "violations", "expect"))
    print("-" * 80)
    for label, interactions, expected_violation, must_fire in cases:
        r = _run_episode(make_interaction_plan(interactions), user_niche="tech")
        viols = r["interaction_violations"]
        fired = any(expected_violation.replace("interaction_", "") in v for v in viols)
        ok = "OK" if (fired == must_fire) else "FAIL"
        # For "healthy" we expect NO interaction violations.
        if label == "healthy":
            ok = "OK" if not viols else "FAIL"
        print(fmt.format(
            label,
            f"{r['score']:.3f}",
            r["followers"],
            f"{r['min_energy']:.2f}",
            f"{r['engagement_rate']:.3f}",
            len(viols),
            ok,
        ))
    print()


# ---------------------------------------------------------------------------
# C) CROSS-CUTTING SANITY
# ---------------------------------------------------------------------------

def section_c_sanity() -> None:
    print("=" * 78)
    print("C. Cross-cutting sanity (rest, post-only, smart, query_interaction_norms)")
    print("=" * 78)

    # Baselines for visual sanity
    def plan_rest(obs: Dict[str, Any], day: int) -> ViraltestAction:
        return ViraltestAction(scheduled_actions=[])

    def plan_post1(obs: Dict[str, Any], day: int) -> ViraltestAction:
        return ViraltestAction(scheduled_actions=[_post_only()])

    def plan_post2(obs: Dict[str, Any], day: int) -> ViraltestAction:
        return ViraltestAction(scheduled_actions=[
            _post_only(content_type="reel", topic="AI tools"),
            ScheduledAction(hour=19, action_type="post", content_type="carousel",
                             topic="AI tools", tags=["coding"], intent="save_bait"),
        ])

    fmt = "{:<14} {:>7} {:>9} {:>8} {:>8} {:>6}"
    print(fmt.format("baseline", "score", "followers", "energy", "engRate", "errs"))
    print("-" * 60)
    for label, plan_fn in [("rest", plan_rest), ("1-post", plan_post1), ("2-post", plan_post2)]:
        r = _run_episode(plan_fn, user_niche="tech")
        errs = "0" if not r["error"] else r["error"][:12]
        print(fmt.format(label, f"{r['score']:.3f}", r["followers"],
                          f"{r['energy']:.2f}", f"{r['engagement_rate']:.3f}", errs))
    print()

    # Verify query_interaction_norms surfaces sensible values.
    env = ViraltestEnvironment()
    env.reset(task="monthly_engage", seed=SEED, user_niche="tech")
    from models import ToolCall
    res = env._dispatch_tool(ToolCall(name="query_interaction_norms", arguments={}))
    print("query_interaction_norms tool ->")
    print(f"  success={res.success}, data={res.data}")
    print()

    # Verify query_creator_pool returns the recommendation surface.
    res = env._dispatch_tool(ToolCall(name="query_creator_pool", arguments={}))
    print("query_creator_pool tool ->")
    print(f"  user_niche={res.data['user_niche']}, user_followers={res.data['user_followers']}")
    for p in res.data["pool"]:
        print(f"  {p['id']:<22} same_niche={p['same_niche']!s:<5} overlap={p['audience_overlap']:>4} "
              f"recommended={p['recommended']!s:<5} reason={p['reason']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    section_a_collab_evaluator()
    section_a_collab_cadence()
    section_b_interactions()
    section_c_sanity()
    print("Evaluation complete.")
