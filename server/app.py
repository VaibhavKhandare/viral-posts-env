"""
FastAPI application for the Viraltest Environment v2 (Theme #3.1).

Endpoints:
    - POST /reset, /step, GET /state, /schema — standard OpenEnv
    - GET /tools — tool catalog (Theme #3.1 discovery)
    - GET /tools/{name} — single tool schema
    - GET /dashboard — simulation UI
"""

import json
import os
import random as stdlib_random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'uv sync'"
    ) from e

if "ENABLE_WEB_INTERFACE" not in os.environ:
    os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
    from ..models import (
        CollabProposal,
        DailyInteractions,
        ScheduledAction,
        ViraltestAction,
        ViraltestObservation,
    )
    from .viraltest_environment import TOOL_CATALOG, ViraltestEnvironment
except ImportError:
    from models import (
        CollabProposal,
        DailyInteractions,
        ScheduledAction,
        ViraltestAction,
        ViraltestObservation,
    )
    from server.viraltest_environment import TOOL_CATALOG, ViraltestEnvironment

try:
    from .viraltest_environment import TAG_POOL
except ImportError:
    from server.viraltest_environment import TAG_POOL

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()
_TRAINING_HTML_PATH = Path(__file__).parent / "training.html"
_TRAINING_HTML = _TRAINING_HTML_PATH.read_text() if _TRAINING_HTML_PATH.exists() else "<html><body>Training page not found</body></html>"

app = create_app(
    ViraltestEnvironment,
    ViraltestAction,
    ViraltestObservation,
    env_name="viraltest",
    max_concurrent_envs=1,
)

_gradio_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
if not _gradio_web:

    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse("/dashboard", status_code=302)

    @app.get("/web", include_in_schema=False)
    @app.get("/web/", include_in_schema=False)
    async def _web_disabled_redirect():
        return RedirectResponse("/dashboard", status_code=302)

# ---------------------------------------------------------------------------
# Tool catalog endpoints (Theme #3.1 — tool discovery)
# ---------------------------------------------------------------------------

@app.get("/tools")
async def list_tools():
    """Return the full tool catalog so the agent can discover available tools."""
    return JSONResponse(content={
        "tools": {name: schema for name, schema in TOOL_CATALOG.items()},
        "count": len(TOOL_CATALOG),
    })


@app.get("/tools/{name}")
async def get_tool(name: str):
    """Return schema for a single tool."""
    if name not in TOOL_CATALOG:
        return JSONResponse(content={"error": f"unknown tool: {name}"}, status_code=404)
    return JSONResponse(content={"name": name, **TOOL_CATALOG[name]})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

_dash_env: Optional[ViraltestEnvironment] = None
_HISTORY_FILE = Path(__file__).parent / "simulation_history.json"


def _obs_to_dict(obs: ViraltestObservation) -> Dict[str, Any]:
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


def _load_history() -> List[Dict[str, Any]]:
    if _HISTORY_FILE.exists():
        try:
            return json.loads(_HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_history_entry(entry: Dict[str, Any]) -> None:
    history = _load_history()
    history.append(entry)
    if len(history) > 100:
        history = history[-100:]
    _HISTORY_FILE.write_text(json.dumps(history, indent=2))


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return _DASHBOARD_HTML


@app.get("/dashboard/history")
async def dashboard_history():
    history = _load_history()
    out: List[Dict[str, Any]] = []
    for row in history:
        entry = dict(row)
        if not entry.get("description"):
            sid = entry.get("scenario_id")
            if sid and sid in SCENARIOS:
                entry["description"] = SCENARIOS[sid][1]
        out.append(entry)
    return out


@app.delete("/dashboard/history")
async def dashboard_history_clear():
    if _HISTORY_FILE.exists():
        _HISTORY_FILE.unlink()
    return {"status": "cleared"}


@app.post("/dashboard/reset")
async def dashboard_reset(body: Dict[str, Any] = Body(default={})):
    global _dash_env
    _dash_env = ViraltestEnvironment()
    task = body.get("task", "weekly_engage")
    obs = _dash_env.reset(task=task)
    return _obs_to_dict(obs)


@app.post("/dashboard/step")
async def dashboard_step(body: Dict[str, Any] = Body(...)):
    global _dash_env
    if _dash_env is None:
        _dash_env = ViraltestEnvironment()
        _dash_env.reset()
    action_data = body.get("action", body)
    action = ViraltestAction(**action_data)
    obs = _dash_env.step(action)
    return _obs_to_dict(obs)


# ---------------------------------------------------------------------------
# Dashboard scenario helpers (v2 action shape)
# ---------------------------------------------------------------------------

_SIM_RNG = stdlib_random.Random(99)
_CONTENT_TYPES = ["reel", "carousel", "story", "text_post"]
_TOPICS = ["AI tools", "fitness routine", "growth hacks", "travel guide", "food recipe", "wellness tips"]


def _make_daily_plan(
    actions: list,
    notes: Optional[str] = None,
    collab: Optional[CollabProposal] = None,
    interactions: Optional[DailyInteractions] = None,
) -> ViraltestAction:
    return ViraltestAction(
        scheduled_actions=[ScheduledAction(**a) for a in actions],
        notes=notes,
        collab=collab,
        interactions=interactions,
    )


def _plan_always_rest(obs: dict, day: int) -> ViraltestAction:
    return _make_daily_plan([], notes="Resting all day to conserve energy.")


def _plan_spam(obs: dict, day: int) -> ViraltestAction:
    actions = [
        {"hour": h, "action_type": "post", "content_type": "reel",
         "topic": "AI tools", "tags": ["ai"], "intent": "watch_bait"}
        for h in range(24)
    ]
    return _make_daily_plan(actions)


def _plan_smart(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    t_tags = list((obs.get("trending_tags") or [])[:2])
    pool_tag = TAG_POOL[(day * 2) % len(TAG_POOL)]
    pool_tag2 = TAG_POOL[(day * 2 + 1) % len(TAG_POOL)]
    ct1 = _CONTENT_TYPES[(day * 2) % 4]
    ct2 = _CONTENT_TYPES[(day * 2 + 1) % 4]
    intent1 = "save_bait" if ct1 == "carousel" else "watch_bait"
    intent2 = "send_bait" if ct2 == "reel" else "save_bait"
    actions = [
        {"hour": 8, "action_type": "create_content"},
        {"hour": 12, "action_type": "post", "content_type": ct1, "topic": trending,
         "tags": t_tags + [pool_tag], "intent": intent1},
        {"hour": 19, "action_type": "post", "content_type": ct2, "topic": trending,
         "tags": t_tags + [pool_tag2], "intent": intent2},
    ]
    return _make_daily_plan(actions, notes=f"Day {day}: posting at peak hours with varied intents.")


def _plan_random(obs: dict, day: int) -> ViraltestAction:
    actions = []
    for h in range(24):
        r = _SIM_RNG.random()
        if r < 0.1:
            ct = _SIM_RNG.choice(_CONTENT_TYPES)
            topic = _SIM_RNG.choice(_TOPICS)
            tags = _SIM_RNG.sample(TAG_POOL[:20], 2)
            actions.append({"hour": h, "action_type": "post", "content_type": ct, "topic": topic, "tags": tags})
        elif r < 0.15:
            actions.append({"hour": h, "action_type": "create_content"})
    return _make_daily_plan(actions)


def _plan_minimal(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["minimalism"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _make_daily_plan([
        {"hour": 12, "action_type": "post", "content_type": "carousel",
         "topic": trending, "tags": tags, "intent": "save_bait"},
    ])


def _plan_collab_same_low(obs: dict, day: int) -> ViraltestAction:
    """Same-niche, low-overlap collab on day 5+15 — best-case reward path."""
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["ai"]
    actions = [
        {"hour": 12, "action_type": "post", "content_type": "reel",
         "topic": trending, "tags": tags, "intent": "watch_bait"},
    ]
    collab = None
    if day in (5, 15):
        collab = CollabProposal(partner_id="niche_expert", content_type="reel", hour=12)
    return _make_daily_plan(actions, notes="Same-niche low-overlap collab demo.", collab=collab)


def _plan_collab_diff_high(obs: dict, day: int) -> ViraltestAction:
    """Diff-niche, high-overlap collab — penalty path (mismatch)."""
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["ai"]
    actions = [
        {"hour": 12, "action_type": "post", "content_type": "reel",
         "topic": trending, "tags": tags, "intent": "watch_bait"},
    ]
    collab = None
    if day in (5, 15):
        collab = CollabProposal(partner_id="lifestyle_blogger", content_type="reel", hour=12)
    return _make_daily_plan(actions, notes="Diff-niche high-overlap collab demo.", collab=collab)


def _plan_interact_balanced(obs: dict, day: int) -> ViraltestAction:
    """Healthy daily interaction — likes/comments on-niche, replies to audience."""
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    interactions = DailyInteractions(
        likes_on_others=12, comments_on_others=5, replies_to_audience=3,
        target_partner_ids=["niche_expert"], avg_reply_quality=0.8,
    )
    return _make_daily_plan(
        [{"hour": 12, "action_type": "post", "content_type": "reel",
          "topic": trending, "tags": ["ai"], "intent": "watch_bait"}],
        notes="Healthy interaction demo.",
        interactions=interactions,
    )


def _plan_interact_spam(obs: dict, day: int) -> ViraltestAction:
    """Spam interaction — triggers shadowban_risk + reach penalty."""
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    interactions = DailyInteractions(
        likes_on_others=80, comments_on_others=40, replies_to_audience=0,
        target_partner_ids=["niche_expert"], avg_reply_quality=0.4,
    )
    return _make_daily_plan(
        [{"hour": 12, "action_type": "post", "content_type": "reel",
          "topic": trending, "tags": ["ai"], "intent": "watch_bait"}],
        notes="Interaction spam demo.",
        interactions=interactions,
    )


# Scenario tuple: (label, description, plan_fn, optional user_niche).
# user_niche is honored by dashboard_simulate / training_evidence; defaults to "generic" when None.
SCENARIOS: Dict[str, tuple] = {
    "always_rest": ("Always Rest", "Never posts. Tests follower decay.", _plan_always_rest, None),
    "spam": ("Spam Post", "Same reel every hour. Burns out fast.", _plan_spam, None),
    "smart": ("Smart Agent", "Optimal: peak hours, trending, varied types+intents.", _plan_smart, None),
    "minimal": ("Minimal Poster", "1 carousel per day at noon.", _plan_minimal, None),
    "random": ("Random Actor", "Random actions. Baseline test.", _plan_random, None),
    "collab_same_low": (
        "Collab Same-Niche Low Overlap",
        "Same-niche partner with <20% overlap. Best-case collab reward path.",
        _plan_collab_same_low,
        "tech",
    ),
    "collab_diff_high": (
        "Collab Diff-Niche High Overlap",
        "Diff-niche partner with >40% overlap. Penalty path (audience mismatch).",
        _plan_collab_diff_high,
        "tech",
    ),
    "interact_balanced": (
        "Interact Balanced",
        "Healthy on-niche likes/comments and audience replies.",
        _plan_interact_balanced,
        "tech",
    ),
    "interact_spam": (
        "Interact Spam",
        "80 likes + 40 comments — spam path triggers shadowban_risk.",
        _plan_interact_spam,
        "tech",
    ),
}


@app.get("/dashboard/scenarios")
async def dashboard_scenarios():
    items = [{"id": k, "label": v[0], "description": v[1]} for k, v in SCENARIOS.items()]
    items.sort(key=lambda x: x["label"].lower())
    return JSONResponse(
        content={"count": len(items), "scenarios": items},
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


@app.post("/dashboard/simulate")
async def dashboard_simulate(body: Dict[str, Any] = Body(...)):
    global _SIM_RNG
    _SIM_RNG = stdlib_random.Random(99)

    scenario_id = body.get("scenario", "smart")
    task = body.get("task", "weekly_competitive")
    if scenario_id not in SCENARIOS:
        return {"error": f"Unknown scenario: {scenario_id}"}

    entry = SCENARIOS[scenario_id]
    label, desc, plan_fn = entry[0], entry[1], entry[2]
    user_niche = entry[3] if len(entry) > 3 else None
    env = ViraltestEnvironment()
    reset_kwargs: Dict[str, Any] = {"task": task, "seed": 42}
    if user_niche:
        reset_kwargs["user_niche"] = user_niche
    obs = env.reset(**reset_kwargs)
    obs_dict = obs.model_dump()

    steps: List[Dict[str, Any]] = []
    for day in range(1, 31):
        action = plan_fn(obs_dict, day)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        r = obs.reward if obs.reward is not None else 0.0

        n_posts = len([sa for sa in action.scheduled_actions if sa.action_type == "post"])
        n_create = len([sa for sa in action.scheduled_actions if sa.action_type == "create_content"])
        action_str = f"day{day}(posts={n_posts},creates={n_create})"

        steps.append({
            "step": day,
            "action": action_str,
            "reward": round(r, 4),
            "done": obs.done,
            "error": obs.error,
            "energy": round(obs.creator_energy, 3),
            "hours_since_sleep": obs.hours_since_sleep,
            "sleep_debt": round(obs.sleep_debt, 3),
            "followers": obs.follower_count,
            "engagement_rate": round(obs.engagement_rate, 4),
            "burnout_risk": round(obs.burnout_risk, 3),
            "posts_today": obs.posts_today,
            "hour": obs.current_hour,
            "day": obs.day_of_week,
            "days_elapsed": obs.days_elapsed,
            "queue": obs.content_queue_size,
            "api_budget": obs.api_budget_remaining,
        })
        if obs.done:
            break

    score = (obs.metadata or {}).get("grader_score", 0.0)
    result = {
        "scenario": label,
        "description": desc,
        "task": task,
        "steps": steps,
        "total_steps": len(steps),
        "score": round(score, 4),
        "final": {
            "energy": round(obs.creator_energy, 3),
            "hours_since_sleep": obs.hours_since_sleep,
            "sleep_debt": round(obs.sleep_debt, 3),
            "followers": obs.follower_count,
            "engagement_rate": round(obs.engagement_rate, 4),
            "burned_out": obs.creator_energy <= 0,
        },
    }

    rewards = [s["reward"] for s in steps]
    total_posts = sum(s.get("daily_posts_made", 0) for s in steps)
    _save_history_entry({
        "id": datetime.now(timezone.utc).isoformat(),
        "scenario": label,
        "scenario_id": scenario_id,
        "description": desc,
        "task": task,
        "score": round(score, 4),
        "total_steps": len(steps),
        "total_posts": total_posts,
        "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0,
        "final": result["final"],
    })

    return result


_TRAINING_TASKS = ["weekly_engage", "weekly_strategic", "weekly_competitive"]

@app.get("/dashboard/training-evidence")
async def training_evidence():
    """Run all baseline scenarios across all tasks and return structured comparison data."""
    global _SIM_RNG

    results = []
    for scenario_id, entry in SCENARIOS.items():
        label, desc, plan_fn = entry[0], entry[1], entry[2]
        user_niche = entry[3] if len(entry) > 3 else None
        for task in _TRAINING_TASKS:
            _SIM_RNG = stdlib_random.Random(99)
            env = ViraltestEnvironment()
            reset_kwargs: Dict[str, Any] = {"task": task, "seed": 42}
            if user_niche:
                reset_kwargs["user_niche"] = user_niche
            obs = env.reset(**reset_kwargs)
            obs_dict = obs.model_dump()

            rewards: List[float] = []
            energies: List[float] = [obs.creator_energy]

            for day in range(1, 31):
                action = plan_fn(obs_dict, day)
                obs = env.step(action)
                obs_dict = obs.model_dump()
                r = obs.reward if obs.reward is not None else 0.0
                rewards.append(r)
                energies.append(obs.creator_energy)
                if obs.done:
                    break

            score = (obs.metadata or {}).get("grader_score", 0.0)
            results.append({
                "scenario_id": scenario_id,
                "scenario": label,
                "description": desc,
                "task": task,
                "grader_score": round(score, 4),
                "total_reward": round(sum(rewards), 4),
                "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0,
                "steps": len(rewards),
                "final_energy": round(obs.creator_energy, 3),
                "min_energy": round(min(energies), 3),
                "final_followers": obs.follower_count,
                "follower_delta": obs.follower_count - 10000,
                "burned_out": obs.creator_energy <= 0,
                "rewards": [round(r, 4) for r in rewards],
                "energies": [round(e, 3) for e in energies],
            })

    return JSONResponse(
        content={"results": results, "tasks": _TRAINING_TASKS, "scenarios": list(SCENARIOS.keys())},
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


@app.get("/dashboard/training", response_class=HTMLResponse)
async def training_dashboard():
    return _TRAINING_HTML


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    if args.port is not None:
        main(port=args.port)
    else:
        main()
