# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Viraltest Environment.

This module creates an HTTP server that exposes the ViraltestEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
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
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# OpenEnv Gradio UI lives at /web; Dockerfile sets this — default on for local parity with HF Spaces.
if "ENABLE_WEB_INTERFACE" not in os.environ:
    os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
    from ..models import ScheduledAction, ViraltestAction, ViraltestObservation
    from .viraltest_environment import ViraltestEnvironment
except ImportError:
    from models import ScheduledAction, ViraltestAction, ViraltestObservation
    from server.viraltest_environment import ViraltestEnvironment

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()

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


try:
    from .viraltest_environment import TAG_POOL
except ImportError:
    from server.viraltest_environment import TAG_POOL

_SIM_RNG = stdlib_random.Random(99)
_CONTENT_TYPES = ["reel", "carousel", "story", "text_post"]
_TOPICS = ["AI tools", "fitness routine", "growth hacks", "travel guide", "food recipe", "wellness tips"]


def _make_daily_plan(actions: list) -> ViraltestAction:
    """Helper: build a ViraltestAction from a list of ScheduledAction-like dicts."""
    return ViraltestAction(scheduled_actions=[ScheduledAction(**a) for a in actions])


def _plan_always_rest(obs: dict, day: int) -> ViraltestAction:
    return _make_daily_plan([])


def _plan_spam(obs: dict, day: int) -> ViraltestAction:
    actions = [{"hour": h, "action_type": "post", "content_type": "reel",
                "topic": "AI tools", "tags": ["ai"]} for h in range(24)]
    return _make_daily_plan(actions)


def _plan_smart(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["AI tools"])[0]
    t_tags = list((obs.get("trending_tags") or [])[:2])
    pool_tag = TAG_POOL[(day * 2) % len(TAG_POOL)]
    pool_tag2 = TAG_POOL[(day * 2 + 1) % len(TAG_POOL)]
    ct1 = _CONTENT_TYPES[(day * 2) % 4]
    ct2 = _CONTENT_TYPES[(day * 2 + 1) % 4]
    actions = [
        {"hour": 8, "action_type": "create_content"},
        {"hour": 12, "action_type": "post", "content_type": ct1, "topic": trending, "tags": t_tags + [pool_tag]},
        {"hour": 19, "action_type": "post", "content_type": ct2, "topic": trending, "tags": t_tags + [pool_tag2]},
    ]
    return _make_daily_plan(actions)


def _plan_no_rest(obs: dict, day: int) -> ViraltestAction:
    actions = []
    for h in range(24):
        ct = _CONTENT_TYPES[h % 4]
        topic = _SIM_RNG.choice(_TOPICS)
        tags = _SIM_RNG.sample(TAG_POOL, 3)
        actions.append({"hour": h, "action_type": "post", "content_type": ct, "topic": topic, "tags": tags})
    return _make_daily_plan(actions)


def _plan_minimal(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["minimalism"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _make_daily_plan([
        {"hour": 12, "action_type": "post", "content_type": "carousel", "topic": trending, "tags": tags},
    ])


def _plan_reel_max(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["viral content"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _make_daily_plan([
        {"hour": 12, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
        {"hour": 14, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
    ])


def _plan_split_schedule(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["daily content"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["tips"]
    return _make_daily_plan([
        {"hour": 9, "action_type": "post", "content_type": "carousel", "topic": trending, "tags": tags},
        {"hour": 19, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
    ])


def _plan_double_peak(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["peak time content"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _make_daily_plan([
        {"hour": 9, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
        {"hour": 15, "action_type": "post", "content_type": "carousel", "topic": trending, "tags": tags},
    ])


def _plan_tag_explorer(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["devtools"])[0]
    start = (day * 6) % len(TAG_POOL)
    tags1 = [TAG_POOL[(start + i) % len(TAG_POOL)] for i in range(3)]
    tags2 = [TAG_POOL[(start + 3 + i) % len(TAG_POOL)] for i in range(3)]
    ct1 = _CONTENT_TYPES[(day * 2) % 4]
    ct2 = _CONTENT_TYPES[(day * 2 + 1) % 4]
    return _make_daily_plan([
        {"hour": 10, "action_type": "post", "content_type": ct1, "topic": trending, "tags": tags1},
        {"hour": 18, "action_type": "post", "content_type": ct2, "topic": trending, "tags": tags2},
    ])


def _plan_queue_optimizer(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["productivity"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["growth"]
    queue = obs.get("content_queue_size", 0)
    if day < 2 or queue < 2:
        return _make_daily_plan([
            {"hour": 8, "action_type": "create_content"},
            {"hour": 10, "action_type": "create_content"},
            {"hour": 14, "action_type": "create_content"},
        ])
    ct = _CONTENT_TYPES[day % 4]
    return _make_daily_plan([
        {"hour": 12, "action_type": "post", "content_type": ct, "topic": trending, "tags": tags},
        {"hour": 19, "action_type": "post", "content_type": _CONTENT_TYPES[(day + 1) % 4], "topic": trending, "tags": tags},
    ])


def _plan_weekend(obs: dict, day: int) -> ViraltestAction:
    dow = obs.get("day_of_week", 0)
    if dow not in (5, 6):
        return _make_daily_plan([])
    trending = (obs.get("trending_topics") or ["travel"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return _make_daily_plan([
        {"hour": 11, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
        {"hour": 17, "action_type": "post", "content_type": "reel", "topic": trending, "tags": tags},
    ])


def _plan_weekday_only(obs: dict, day: int) -> ViraltestAction:
    dow = obs.get("day_of_week", 0)
    if dow >= 5:
        return _make_daily_plan([])
    trending = (obs.get("trending_topics") or ["weekday content"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["productivity"]
    ct = _CONTENT_TYPES[day % 4]
    return _make_daily_plan([
        {"hour": 12, "action_type": "post", "content_type": ct, "topic": trending, "tags": tags},
    ])


def _plan_random(obs: dict, day: int) -> ViraltestAction:
    actions = []
    for h in range(24):
        r = _SIM_RNG.random()
        if r < 0.1:
            ct = _SIM_RNG.choice(_CONTENT_TYPES)
            topic = _SIM_RNG.choice(["random topic", "AI tools", "fitness", "travel"])
            tags = _SIM_RNG.sample(TAG_POOL, 2)
            actions.append({"hour": h, "action_type": "post", "content_type": ct, "topic": topic, "tags": tags})
        elif r < 0.15:
            actions.append({"hour": h, "action_type": "create_content"})
    return _make_daily_plan(actions)


def _plan_sleep_conscious(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["wellness"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["productivity"]
    ct = _CONTENT_TYPES[day % 4]
    return _make_daily_plan([
        {"hour": 10, "action_type": "post", "content_type": ct, "topic": trending, "tags": tags},
        {"hour": 16, "action_type": "create_content"},
    ])


def _plan_sleep_deprived(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["coding"])[0]
    tags = list((obs.get("trending_tags") or [])[:2])
    actions = []
    for h in range(24):
        if 9 <= h <= 20 and len([a for a in actions if a["action_type"] == "post"]) < 2:
            ct = _CONTENT_TYPES[h % 4]
            actions.append({"hour": h, "action_type": "post", "content_type": ct, "topic": trending, "tags": tags})
        else:
            actions.append({"hour": h, "action_type": "create_content"})
    return _make_daily_plan(actions)


def _plan_growth_focus(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["growth hacks"])[0]
    return _make_daily_plan([
        {"hour": 13, "action_type": "post", "content_type": "reel", "topic": trending, "tags": ["viral", "growth", "trending"]},
    ])


def _plan_tech_niche(obs: dict, day: int) -> ViraltestAction:
    ct = _CONTENT_TYPES[day % 4]
    return _make_daily_plan([
        {"hour": 12, "action_type": "post", "content_type": ct, "topic": "AI tools and coding tips", "tags": ["ai", "coding", "devtools"]},
        {"hour": 18, "action_type": "post", "content_type": _CONTENT_TYPES[(day + 1) % 4], "topic": "AI tools and coding tips", "tags": ["ai", "ml", "startup"]},
    ])


def _plan_conservative(obs: dict, day: int) -> ViraltestAction:
    trending = (obs.get("trending_topics") or ["quick tip"])[0]
    tags = list((obs.get("trending_tags") or [])[:2])
    return _make_daily_plan([
        {"hour": 13, "action_type": "post", "content_type": "text_post", "topic": trending, "tags": tags},
    ])


SCENARIOS = {
    "always_rest": ("Always Rest", "Never posts. Tests follower decay + zero engagement.", _plan_always_rest),
    "spam": ("Spam Post", "Same reel every hour. Burns out fast.", _plan_spam),
    "no_rest": ("No Rest", "Posts every hour, never rests. Burns out fast.", _plan_no_rest),
    "smart": ("Smart Agent", "Optimal: peak hours, trending, varied types, rests.", _plan_smart),
    "queue_optimizer": ("Queue Optimizer", "Creates content first, posts from queue.", _plan_queue_optimizer),
    "weekend": ("Weekend Warrior", "Only posts on Sat/Sun.", _plan_weekend),
    "tag_explorer": ("Tag Explorer", "New tag combo every post. Max discovery.", _plan_tag_explorer),
    "sleep_deprived": ("Sleep Deprived", "Never rests. Tests sleep deprivation.", _plan_sleep_deprived),
    "sleep_conscious": ("Sleep Conscious", "Proper sleep schedule.", _plan_sleep_conscious),
    "minimal": ("Minimal Poster", "1 post per day at noon.", _plan_minimal),
    "reel_max": ("Reel Maximizer", "Reels at peak hours for max reach.", _plan_reel_max),
    "split_schedule": ("Split Schedule", "Morning and evening posts.", _plan_split_schedule),
    "double_peak": ("Double Peak", "Posts at 9am and 3pm.", _plan_double_peak),
    "growth_focus": ("Growth Focus", "Maximizes follower growth.", _plan_growth_focus),
    "weekday_only": ("Weekday Only", "No weekend posting.", _plan_weekday_only),
    "tech_niche": ("Tech Niche", "AI/coding content focus.", _plan_tech_niche),
    "conservative": ("Conservative", "One text post at 1pm.", _plan_conservative),
    "random": ("Random Actor", "Random actions. Baseline test.", _plan_random),
}


@app.get("/dashboard/scenarios")
async def dashboard_scenarios():
    """List all simulation strategies for the dashboard UI."""
    items = [{"id": k, "label": v[0], "description": v[1]} for k, v in SCENARIOS.items()]
    items.sort(key=lambda x: (x["label"].lower()))
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

    label, desc, plan_fn = SCENARIOS[scenario_id]
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=42)
    obs_dict = obs.model_dump()

    steps: List[Dict[str, Any]] = []
    for day in range(1, 8):
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
            "niche_saturation": round(obs.niche_saturation, 3),
            "posts_today": obs.posts_today,
            "hour": obs.current_hour,
            "day": obs.day_of_week,
            "days_elapsed": obs.days_elapsed,
            "queue": obs.content_queue_size,
            "tag_performance": obs.tag_performance,
            "trending_topics": obs.trending_topics,
            "trending_tags": obs.trending_tags,
            "competitor_avg_engagement": round(obs.competitor_avg_engagement, 4),
            "daily_total_engagement": round(obs.daily_total_engagement, 4),
            "daily_posts_made": obs.daily_posts_made,
            "daily_energy_min": round(obs.daily_energy_min, 3),
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


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m viraltest.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn viraltest.server.app:app --workers 4
    """
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
