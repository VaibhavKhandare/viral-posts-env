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
    from ..models import ScheduledAction, ViraltestAction, ViraltestObservation
    from .viraltest_environment import TOOL_CATALOG, ViraltestEnvironment
except ImportError:
    from models import ScheduledAction, ViraltestAction, ViraltestObservation
    from server.viraltest_environment import TOOL_CATALOG, ViraltestEnvironment

try:
    from .viraltest_environment import TAG_POOL
except ImportError:
    from server.viraltest_environment import TAG_POOL

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
    task = body.get("task", "monthly_engage")
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


def _make_daily_plan(actions: list, notes: Optional[str] = None) -> ViraltestAction:
    return ViraltestAction(
        scheduled_actions=[ScheduledAction(**a) for a in actions],
        notes=notes,
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


SCENARIOS = {
    "always_rest": ("Always Rest", "Never posts. Tests follower decay.", _plan_always_rest),
    "spam": ("Spam Post", "Same reel every hour. Burns out fast.", _plan_spam),
    "smart": ("Smart Agent", "Optimal: peak hours, trending, varied types+intents.", _plan_smart),
    "minimal": ("Minimal Poster", "1 carousel per day at noon.", _plan_minimal),
    "random": ("Random Actor", "Random actions. Baseline test.", _plan_random),
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
    task = body.get("task", "monthly_competitive")
    if scenario_id not in SCENARIOS:
        return {"error": f"Unknown scenario: {scenario_id}"}

    label, desc, plan_fn = SCENARIOS[scenario_id]
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=42)
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
