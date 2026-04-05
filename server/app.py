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
import random as stdlib_random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body
from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ViraltestAction, ViraltestObservation
    from .viraltest_environment import ViraltestEnvironment
except ImportError:
    from models import ViraltestAction, ViraltestObservation
    from server.viraltest_environment import ViraltestEnvironment

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()
_COMPETITORS_HTML = (Path(__file__).parent / "competitor_insights.html").read_text()
_TAGS_HTML = (Path(__file__).parent / "tag_performance.html").read_text()

app = create_app(
    ViraltestEnvironment,
    ViraltestAction,
    ViraltestObservation,
    env_name="viraltest",
    max_concurrent_envs=1,
)

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


@app.get("/dashboard/competitors", response_class=HTMLResponse)
async def dashboard_competitors():
    return _COMPETITORS_HTML


@app.get("/dashboard/tags", response_class=HTMLResponse)
async def dashboard_tags():
    return _TAGS_HTML


@app.get("/dashboard/insights")
async def dashboard_insights():
    """Pre-computed insights from a smart agent simulation for competitor/tag pages."""
    env = ViraltestEnvironment()
    obs = env.reset(task="weekly_competitive", seed=42)
    obs_dict = obs.model_dump()

    steps_data: List[Dict[str, Any]] = []
    tag_perf_snapshots: List[Dict[str, float]] = []

    for step in range(1, 169):
        energy = obs_dict.get("creator_energy", 1.0)
        posts = obs_dict.get("posts_today", 0)
        hour = obs_dict.get("current_hour", 12)
        trending = obs_dict.get("trending_topics", [])
        ttags = obs_dict.get("trending_tags", [])

        if energy < 0.4 or posts >= 2:
            action_data = {"action_type": "rest"}
        elif 9 <= hour <= 20 and posts < 2:
            ct = ["reel", "carousel", "story", "text_post"][step % 4]
            topic = trending[0] if trending else "AI tools"
            tags = list(ttags[:2]) + [TAG_POOL[step % len(TAG_POOL)]]
            action_data = {"action_type": "post", "content_type": ct, "topic": topic, "tags": tags}
        else:
            action_data = {"action_type": "rest"}

        action = ViraltestAction(**action_data)
        obs = env.step(action)
        obs_dict = obs.model_dump()

        steps_data.append({
            "step": step,
            "energy": round(obs.creator_energy, 3),
            "followers": obs.follower_count,
            "engagement_rate": round(obs.engagement_rate, 4),
            "niche_saturation": round(obs.niche_saturation, 3),
            "competitor_avg_engagement": round(obs.competitor_avg_engagement, 4),
            "competitor_recent_posts": obs.competitor_recent_posts[:5],
            "trending_topics": obs.trending_topics,
            "trending_tags": obs.trending_tags,
            "tag_performance": obs.tag_performance,
            "reward": round(obs.reward, 4) if obs.reward else 0,
            "action": action_data.get("action_type", "rest"),
            "action_tags": action_data.get("tags", []),
        })
        if step % 24 == 0:
            tag_perf_snapshots.append(dict(obs.tag_performance))
        if obs.done:
            break

    score = (obs.metadata or {}).get("grader_score", 0.0)
    return {
        "steps": steps_data,
        "total_steps": len(steps_data),
        "score": round(score, 4),
        "tag_perf_over_time": tag_perf_snapshots,
        "final": {
            "energy": round(obs.creator_energy, 3),
            "followers": obs.follower_count,
            "engagement_rate": round(obs.engagement_rate, 4),
            "tag_performance": obs.tag_performance,
            "trending_topics": obs.trending_topics,
            "trending_tags": obs.trending_tags,
            "competitor_avg_engagement": round(obs.competitor_avg_engagement, 4),
            "competitor_recent_posts": obs.competitor_recent_posts[:5],
            "niche_saturation": round(obs.niche_saturation, 3),
        },
    }


@app.get("/dashboard/history")
async def dashboard_history():
    return _load_history()


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


def _agent_always_rest(obs: dict, step: int) -> dict:
    return {"action_type": "rest"}


def _agent_spam(obs: dict, step: int) -> dict:
    return {"action_type": "post", "content_type": "reel", "topic": "AI tools", "tags": ["ai"]}


def _agent_bad_timing(obs: dict, step: int) -> dict:
    h = obs.get("current_hour", 12)
    if h >= 23 or h < 6:
        return {"action_type": "post", "content_type": "text_post", "topic": "random boring stuff", "tags": ["stoic", "minimalism"]}
    return {"action_type": "rest"}


def _agent_no_rest(obs: dict, step: int) -> dict:
    return {
        "action_type": "post",
        "content_type": _CONTENT_TYPES[step % 4],
        "topic": _SIM_RNG.choice(_TOPICS),
        "tags": _SIM_RNG.sample(TAG_POOL, 3),
    }


_sm_ct = 0
_sm_tg = 0


def _agent_smart(obs: dict, step: int) -> dict:
    global _sm_ct, _sm_tg
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20 and posts < 2:
        ct = _CONTENT_TYPES[_sm_ct % 4]
        _sm_ct += 1
        topic = (obs.get("trending_topics") or ["AI tools"])[0]
        tags = list((obs.get("trending_tags") or [])[:2])
        tags.append(TAG_POOL[_sm_tg % len(TAG_POOL)])
        _sm_tg += 1
        return {"action_type": "post", "content_type": ct, "topic": topic, "tags": tags}
    return {"action_type": "rest"}


def _agent_copycat(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.3 or posts >= 3:
        return {"action_type": "rest"}
    comp = obs.get("competitor_recent_posts") or []
    if comp and 9 <= hour <= 20:
        c = comp[0]
        return {"action_type": "post", "content_type": c.get("content_type", "reel"), "topic": c.get("topic", "AI tools"), "tags": c.get("tags", ["ai"])}
    return {"action_type": "rest"}


_q7_phase = "prep"


def _agent_queue_optimizer(obs: dict, step: int) -> dict:
    global _q7_phase
    q = obs.get("content_queue_size", 0)
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)

    if _q7_phase == "prep" and q < 4:
        return {"action_type": "create_content"}
    _q7_phase = "post"

    if energy < 0.35:
        return {"action_type": "create_content"} if q < 2 else {"action_type": "rest"}

    if 9 <= hour <= 20 and posts < 2 and q > 0:
        ct = _CONTENT_TYPES[step % 4]
        topic = (obs.get("trending_topics") or ["productivity"])[0]
        tags = list((obs.get("trending_tags") or [])[:2]) + ["growth"]
        return {"action_type": "post", "content_type": ct, "topic": topic, "tags": tags}

    return {"action_type": "create_content"} if q < 3 else {"action_type": "rest"}


_burst_count = 0


def _agent_burst(obs: dict, step: int) -> dict:
    global _burst_count
    energy = obs.get("creator_energy", 1.0)
    if energy < 0.5:
        _burst_count = 0
        return {"action_type": "rest"}
    if _burst_count >= 3:
        _burst_count = 0
        return {"action_type": "rest"}
    _burst_count += 1
    topic = (obs.get("trending_topics") or ["coding"])[0]
    tags = list((obs.get("trending_tags") or [])[:2]) + ["tips"]
    return {"action_type": "post", "content_type": ["reel", "carousel", "story"][_burst_count % 3], "topic": topic, "tags": tags}


def _agent_weekend(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    dow = obs.get("day_of_week", 0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if dow not in (5, 6):
        return {"action_type": "rest"}
    if energy < 0.3 or posts >= 3:
        return {"action_type": "rest"}
    topic = (obs.get("trending_topics") or ["travel"])[0]
    tags = list((obs.get("trending_tags") or [])[:3])
    return {"action_type": "post", "content_type": "reel", "topic": topic, "tags": tags}


_te_idx = 0


def _agent_tag_explorer(obs: dict, step: int) -> dict:
    global _te_idx
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 8 <= hour <= 21:
        start = (_te_idx * 3) % len(TAG_POOL)
        tags = [TAG_POOL[(start + i) % len(TAG_POOL)] for i in range(3)]
        _te_idx += 1
        topic = (obs.get("trending_topics") or ["devtools"])[0]
        return {"action_type": "post", "content_type": _CONTENT_TYPES[_te_idx % 4], "topic": topic, "tags": tags}
    return {"action_type": "rest"}


_bc_phase = 0


def _agent_balanced(obs: dict, step: int) -> dict:
    global _bc_phase
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    cycle = _bc_phase % 3
    _bc_phase += 1
    if energy < 0.3:
        return {"action_type": "rest"}
    if cycle == 0:
        return {"action_type": "create_content"}
    if cycle == 1 and 8 <= hour <= 20:
        ct = _CONTENT_TYPES[step % 4]
        topic = (obs.get("trending_topics") or ["startup"])[0]
        tags = list((obs.get("trending_tags") or [])[:2]) + [TAG_POOL[step % len(TAG_POOL)]]
        return {"action_type": "post", "content_type": ct, "topic": topic, "tags": tags}
    return {"action_type": "rest"}


SCENARIOS = {
    "always_rest": ("Always Rest", "Never posts. Tests follower decay + zero engagement.", _agent_always_rest),
    "spam": ("Spam Post", "Same reel every step. Burns out in 4 steps.", _agent_spam),
    "bad_timing": ("Bad Timing", "Posts at night with boring topics only.", _agent_bad_timing),
    "no_rest": ("No Rest", "Varied posts but never rests. Burns out fast.", _agent_no_rest),
    "smart": ("Smart Agent", "Optimal: peak hours, trending, varied types, rests.", _agent_smart),
    "copycat": ("Copycat", "Copies competitor topics. High saturation penalty.", _agent_copycat),
    "queue_optimizer": ("Queue Optimizer", "Creates content first, posts from queue at half energy.", _agent_queue_optimizer),
    "burst": ("Burst Poster", "3 posts in a row, then rests until recovered.", _agent_burst),
    "weekend": ("Weekend Warrior", "Only posts on Sat/Sun. Misses 5 weekdays.", _agent_weekend),
    "tag_explorer": ("Tag Explorer", "New tag combo every post. Max discovery.", _agent_tag_explorer),
    "balanced": ("Balanced Creator", "Create→Post→Rest cycle. Uses queue.", _agent_balanced),
}


@app.post("/dashboard/simulate")
async def dashboard_simulate(body: Dict[str, Any] = Body(...)):
    global _sm_ct, _sm_tg, _SIM_RNG, _q7_phase, _burst_count, _te_idx, _bc_phase
    _sm_ct = 0
    _sm_tg = 0
    _q7_phase = "prep"
    _burst_count = 0
    _te_idx = 0
    _bc_phase = 0
    _SIM_RNG = stdlib_random.Random(99)

    scenario_id = body.get("scenario", "smart")
    task = body.get("task", "weekly_competitive")
    if scenario_id not in SCENARIOS:
        return {"error": f"Unknown scenario: {scenario_id}"}

    label, desc, agent_fn = SCENARIOS[scenario_id]
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=42)
    obs_dict = obs.model_dump()

    steps: List[Dict[str, Any]] = []
    for step in range(1, 169):
        action_data = agent_fn(obs_dict, step)
        action = ViraltestAction(**action_data)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        r = obs.reward if obs.reward is not None else 0.0

        action_str = action_data.get("action_type", "?")
        if action_str == "post":
            ct = action_data.get("content_type", "?")
            tp = action_data.get("topic", "?")
            tg = ",".join(action_data.get("tags") or [])
            action_str = f'post({ct},"{tp}",[{tg}])'
        else:
            action_str += "()"

        steps.append({
            "step": step,
            "action": action_str,
            "reward": round(r, 4),
            "done": obs.done,
            "error": obs.error,
            "energy": round(obs.creator_energy, 3),
            "followers": obs.follower_count,
            "engagement_rate": round(obs.engagement_rate, 4),
            "niche_saturation": round(obs.niche_saturation, 3),
            "posts_today": obs.posts_today,
            "hour": obs.current_hour,
            "day": obs.day_of_week,
            "queue": obs.content_queue_size,
            "tag_performance": obs.tag_performance,
            "trending_topics": obs.trending_topics,
            "trending_tags": obs.trending_tags,
            "competitor_avg_engagement": round(obs.competitor_avg_engagement, 4),
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
            "followers": obs.follower_count,
            "engagement_rate": round(obs.engagement_rate, 4),
            "burned_out": obs.creator_energy <= 0,
        },
    }

    rewards = [s["reward"] for s in steps]
    post_count = sum(1 for s in steps if s["action"].startswith("post"))
    _save_history_entry({
        "id": datetime.now(timezone.utc).isoformat(),
        "scenario": label,
        "scenario_id": scenario_id,
        "task": task,
        "score": round(score, 4),
        "total_steps": len(steps),
        "total_posts": post_count,
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
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
