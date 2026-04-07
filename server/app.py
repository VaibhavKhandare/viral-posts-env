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
    from ..models import ViraltestAction, ViraltestObservation
    from .viraltest_environment import ViraltestEnvironment
except ImportError:
    from models import ViraltestAction, ViraltestObservation
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


# --- NEW SCENARIOS (50 additional) ---

def _agent_sleep_deprived(obs: dict, step: int) -> dict:
    """Posts or creates content but never rests - tests sleep deprivation."""
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if 9 <= hour <= 20 and posts < 2:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["coding"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2])}
    return {"action_type": "create_content"}


def _agent_sleep_conscious(obs: dict, step: int) -> dict:
    """Rests during night hours to simulate proper sleep schedule."""
    hour = obs.get("current_hour", 12)
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    if hour >= 23 or hour < 7:
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["wellness"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["productivity"]}
    return {"action_type": "create_content"}


def _agent_minimal(obs: dict, step: int) -> dict:
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if posts >= 1:
        return {"action_type": "rest"}
    if hour == 12:
        return {"action_type": "post", "content_type": "carousel",
                "topic": (obs.get("trending_topics") or ["minimalism"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_story_spammer(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.2:
        return {"action_type": "rest"}
    if posts < 4 and 8 <= hour <= 22:
        return {"action_type": "post", "content_type": "story",
                "topic": "daily update", "tags": ["fitness", "wellness"]}
    return {"action_type": "rest"}


def _agent_reel_max(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 12 <= hour <= 15:
        return {"action_type": "post", "content_type": "reel",
                "topic": (obs.get("trending_topics") or ["viral content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_text_only(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.3 or posts >= 3:
        return {"action_type": "rest"}
    if 9 <= hour <= 18:
        return {"action_type": "post", "content_type": "text_post",
                "topic": "thoughts and tips", "tags": ["tips", "howto", "motivation"]}
    return {"action_type": "rest"}


def _agent_early_bird(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.3:
        return {"action_type": "rest"}
    if 6 <= hour <= 10 and posts < 2:
        return {"action_type": "post", "content_type": "carousel",
                "topic": "morning routine", "tags": ["productivity", "wellness", "fitness"]}
    return {"action_type": "rest"}


def _agent_night_owl(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.3:
        return {"action_type": "rest"}
    if 20 <= hour <= 23 and posts < 2:
        return {"action_type": "post", "content_type": "reel",
                "topic": "night thoughts", "tags": ["stoic", "motivation"]}
    return {"action_type": "rest"}


def _agent_trend_chaser(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    trending = obs.get("trending_topics") or []
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if trending and 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "reel",
                "topic": trending[0], "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_anti_trend(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    trending_tags = obs.get("trending_tags") or []
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        non_trending = [t for t in TAG_POOL if t not in trending_tags][:3]
        return {"action_type": "post", "content_type": "carousel",
                "topic": "unique perspective on niche topic",
                "tags": non_trending if non_trending else ["minimalism", "stoic"]}
    return {"action_type": "rest"}


def _agent_energy_saver(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.7:
        return {"action_type": "rest"}
    if posts < 1 and 10 <= hour <= 18:
        return {"action_type": "post", "content_type": "carousel",
                "topic": (obs.get("trending_topics") or ["productivity"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["tips"]}
    return {"action_type": "rest"}


_qh_phase = "prep"


def _agent_queue_heavy(obs: dict, step: int) -> dict:
    global _qh_phase
    days = obs.get("days_elapsed", 0)
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    queue = obs.get("content_queue_size", 0)
    if days < 3:
        if energy < 0.3:
            return {"action_type": "rest"}
        return {"action_type": "create_content"}
    if energy < 0.35:
        return {"action_type": "rest"}
    if queue > 0 and 9 <= hour <= 20 and posts < 2:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["growth"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["viral"]}
    return {"action_type": "rest"}


def _agent_midday(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 11 <= hour <= 14:
        return {"action_type": "post", "content_type": "reel",
                "topic": (obs.get("trending_topics") or ["lunch break content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["howto"]}
    return {"action_type": "rest"}


def _agent_random(obs: dict, step: int) -> dict:
    action = _SIM_RNG.choice(["post", "rest", "create_content"])
    if action == "post":
        return {"action_type": "post", "content_type": _SIM_RNG.choice(_CONTENT_TYPES),
                "topic": _SIM_RNG.choice(["random topic", "AI tools", "fitness", "travel"]),
                "tags": _SIM_RNG.sample(TAG_POOL, 2)}
    elif action == "create_content":
        return {"action_type": "create_content"}
    return {"action_type": "rest"}


def _agent_carousel_only(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.35 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 18:
        return {"action_type": "post", "content_type": "carousel",
                "topic": (obs.get("trending_topics") or ["guide"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_tech_niche(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": "AI tools and coding tips", "tags": ["ai", "coding", "devtools"]}
    return {"action_type": "rest"}


def _agent_lifestyle_niche(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "reel",
                "topic": "fitness routine and wellness tips", "tags": ["fitness", "wellness", "travel"]}
    return {"action_type": "rest"}


def _agent_high_freq(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.25:
        return {"action_type": "rest"}
    if posts < 3 and 8 <= hour <= 21:
        ct = ["story", "text_post", "reel"][posts % 3]
        return {"action_type": "post", "content_type": ct,
                "topic": (obs.get("trending_topics") or ["daily update"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2])}
    return {"action_type": "rest"}


def _agent_low_freq(obs: dict, step: int) -> dict:
    days = obs.get("days_elapsed", 0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if days % 2 != 0:
        return {"action_type": "rest"}
    if posts >= 1:
        return {"action_type": "rest"}
    if hour == 14:
        return {"action_type": "post", "content_type": "carousel",
                "topic": (obs.get("trending_topics") or ["weekly roundup"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_comp_avoider(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    saturation = obs.get("niche_saturation", 0.0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if saturation > 0.5:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "reel",
                "topic": "unique angle on trending topic",
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["growth"]}
    return {"action_type": "rest"}


def _agent_tue_thu(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    dow = obs.get("day_of_week", 0)
    if dow not in (1, 3):
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 12 <= hour <= 15:
        return {"action_type": "post", "content_type": "reel",
                "topic": (obs.get("trending_topics") or ["midweek content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_monday(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    dow = obs.get("day_of_week", 0)
    if dow != 0:
        return {"action_type": "rest"}
    if energy < 0.3 or posts >= 3:
        return {"action_type": "rest"}
    if 8 <= hour <= 18:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[posts % 4],
                "topic": "monday motivation", "tags": ["motivation", "tips", "productivity"]}
    return {"action_type": "rest"}


def _agent_tag_exploiter(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    tag_perf = obs.get("tag_performance") or {}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        best_tags = sorted(tag_perf.items(), key=lambda x: x[1], reverse=True)[:3]
        tags = [t[0] for t in best_tags] if best_tags else list((obs.get("trending_tags") or [])[:3])
        return {"action_type": "post", "content_type": "reel",
                "topic": (obs.get("trending_topics") or ["growth content"])[0], "tags": tags}
    return {"action_type": "rest"}


_alt_idx = 0


def _agent_alternating(obs: dict, step: int) -> dict:
    global _alt_idx
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.35 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        ct = _CONTENT_TYPES[_alt_idx % 4]
        _alt_idx += 1
        return {"action_type": "post", "content_type": ct,
                "topic": (obs.get("trending_topics") or ["varied content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["trending"]}
    return {"action_type": "rest"}


def _agent_engagement_chaser(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    eng_rate = obs.get("engagement_rate", 0.0)
    if energy < 0.3:
        return {"action_type": "rest"}
    target = 3 if eng_rate > 0.5 else 2 if eng_rate > 0.3 else 1
    if posts >= target:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "reel",
                "topic": (obs.get("trending_topics") or ["engagement content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_conservative(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.5:
        return {"action_type": "rest"}
    if posts >= 1:
        return {"action_type": "rest"}
    if 12 <= hour <= 14:
        return {"action_type": "post", "content_type": "text_post",
                "topic": (obs.get("trending_topics") or ["quick tip"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2])}
    return {"action_type": "rest"}


def _agent_aggressive(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.15:
        return {"action_type": "rest"}
    if posts >= 4:
        return {"action_type": "rest"}
    if 8 <= hour <= 22:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[posts % 4],
                "topic": (obs.get("trending_topics") or ["high output"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["viral"]}
    return {"action_type": "rest"}


def _agent_sleep_respecting(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    hours_since_sleep = obs.get("hours_since_sleep", 0)
    sleep_debt = obs.get("sleep_debt", 0.0)
    if hours_since_sleep >= 14 or sleep_debt > 0.3:
        return {"action_type": "rest"}
    if hour >= 22 or hour < 8:
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["well-rested content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["wellness"]}
    return {"action_type": "rest"}


def _agent_night_shift(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if 8 <= hour <= 20:
        return {"action_type": "rest"}
    if energy < 0.3 or posts >= 2:
        return {"action_type": "rest"}
    return {"action_type": "post", "content_type": "reel",
            "topic": "late night content", "tags": ["stoic", "motivation"]}


def _agent_split_schedule(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4:
        return {"action_type": "rest"}
    morning = 8 <= hour <= 10
    evening = 18 <= hour <= 20
    if not (morning or evening):
        return {"action_type": "rest"}
    if posts >= 2:
        return {"action_type": "rest"}
    ct = "carousel" if morning else "reel"
    return {"action_type": "post", "content_type": ct,
            "topic": (obs.get("trending_topics") or ["daily content"])[0],
            "tags": list((obs.get("trending_tags") or [])[:2]) + ["tips"]}


def _agent_growth_focus(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.35 or posts >= 2:
        return {"action_type": "rest"}
    if 12 <= hour <= 15:
        return {"action_type": "post", "content_type": "reel",
                "topic": (obs.get("trending_topics") or ["growth hacks"])[0],
                "tags": ["viral", "growth", "trending"]}
    return {"action_type": "rest"}


_ccm_ratio = 0


def _agent_content_creator(obs: dict, step: int) -> dict:
    global _ccm_ratio
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    queue = obs.get("content_queue_size", 0)
    if energy < 0.3:
        return {"action_type": "rest"}
    _ccm_ratio += 1
    if _ccm_ratio % 4 != 0:
        return {"action_type": "create_content"}
    if queue > 0 and 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["queued content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "create_content"}


def _agent_weekday_only(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    dow = obs.get("day_of_week", 0)
    if dow >= 5:
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 18:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["weekday content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["productivity"]}
    return {"action_type": "rest"}


def _agent_double_peak(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if hour in (9, 15):
        ct = "reel" if hour == 9 else "carousel"
        return {"action_type": "post", "content_type": ct,
                "topic": (obs.get("trending_topics") or ["peak time content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_crypto_niche(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "reel",
                "topic": "crypto market analysis and web3 trends",
                "tags": ["crypto", "web3", "ai"]}
    return {"action_type": "rest"}


def _agent_gaming_niche(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 14 <= hour <= 22:
        return {"action_type": "post", "content_type": "reel",
                "topic": "gaming highlights and tips", "tags": ["gaming", "viral", "tips"]}
    return {"action_type": "rest"}


def _agent_productivity(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 6 <= hour <= 10:
        return {"action_type": "post", "content_type": "carousel",
                "topic": "productivity tips and systems", "tags": ["productivity", "tips", "howto"]}
    return {"action_type": "rest"}


def _agent_food_creator(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if hour in (8, 12, 18):
        return {"action_type": "post", "content_type": "reel",
                "topic": "food recipe and cooking tips", "tags": ["food", "howto", "viral"]}
    return {"action_type": "rest"}


def _agent_travel(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 10 <= hour <= 16:
        return {"action_type": "post", "content_type": "reel",
                "topic": "travel guide and destination highlights",
                "tags": ["travel", "photography", "trending"]}
    return {"action_type": "rest"}


def _agent_fashion(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 11 <= hour <= 19:
        return {"action_type": "post", "content_type": "carousel",
                "topic": "fashion haul and style tips", "tags": ["fashion", "trending", "tips"]}
    return {"action_type": "rest"}


def _agent_photography(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if hour in (7, 8, 17, 18):
        return {"action_type": "post", "content_type": "carousel",
                "topic": "photo editing tips and composition",
                "tags": ["photography", "tips", "howto"]}
    return {"action_type": "rest"}


def _agent_stoic(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 1:
        return {"action_type": "rest"}
    if hour == 6:
        return {"action_type": "post", "content_type": "text_post",
                "topic": "stoic wisdom and daily reflection",
                "tags": ["stoic", "motivation", "minimalism"]}
    return {"action_type": "rest"}


def _agent_saas(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 17:
        return {"action_type": "post", "content_type": "carousel",
                "topic": "SaaS growth strategies and startup tips",
                "tags": ["saas", "startup", "growth"]}
    return {"action_type": "rest"}


def _agent_creator_economy(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 10 <= hour <= 18:
        return {"action_type": "post", "content_type": "reel",
                "topic": "creator economy and monetization tips",
                "tags": ["growth", "viral", "tips"]}
    return {"action_type": "rest"}


def _agent_ml_deep(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "carousel",
                "topic": "machine learning concepts and AI tools",
                "tags": ["ml", "ai", "coding"]}
    return {"action_type": "rest"}


def _agent_sleep_debt_aware(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    sleep_debt = obs.get("sleep_debt", 0.0)
    if sleep_debt > 0.2:
        return {"action_type": "rest"}
    if hour >= 23 or hour < 7:
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["balanced content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:3])}
    return {"action_type": "rest"}


def _agent_marathon(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if energy < 0.1:
        return {"action_type": "rest"}
    if 6 <= hour <= 23 and posts < 3:
        return {"action_type": "post", "content_type": "story",
                "topic": "marathon content session",
                "tags": list((obs.get("trending_tags") or [])[:2])}
    return {"action_type": "create_content"}


_nap_count = 0


def _agent_napper(obs: dict, step: int) -> dict:
    global _nap_count
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    hours_since_sleep = obs.get("hours_since_sleep", 0)
    if hours_since_sleep >= 6:
        _nap_count += 1
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["fresh content"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["wellness"]}
    return {"action_type": "rest"}


def _agent_optimal_sleep(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    if hour >= 23 or hour < 7:
        return {"action_type": "rest"}
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        return {"action_type": "post", "content_type": _CONTENT_TYPES[step % 4],
                "topic": (obs.get("trending_topics") or ["well-rested creator"])[0],
                "tags": list((obs.get("trending_tags") or [])[:2]) + ["productivity"]}
    return {"action_type": "create_content"}


def _agent_events(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    hour = obs.get("current_hour", 12)
    posts = obs.get("posts_today", 0)
    trending_tags = obs.get("trending_tags") or []
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    event_tags = [t for t in trending_tags if t in ["worldcup", "election", "oscars", "newyear", "climate"]]
    if event_tags and 9 <= hour <= 20:
        return {"action_type": "post", "content_type": "reel",
                "topic": "breaking news and event coverage",
                "tags": event_tags[:2] + ["trending"]}
    return {"action_type": "rest"}


def _agent_easy_morning_story(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.4 or posts >= 1:
        return {"action_type": "rest"}
    if 9 <= hour <= 11:
        tt = list((obs.get("trending_tags") or [])[:2]) or ["tips"]
        return {
            "action_type": "post",
            "content_type": "story",
            "topic": (obs.get("trending_topics") or ["morning update"])[0],
            "tags": tt,
        }
    return {"action_type": "rest"}


def _agent_easy_one_a_day(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.5 or posts >= 1:
        return {"action_type": "rest"}
    if hour == 13:
        tt = list((obs.get("trending_tags") or [])[:2]) or ["ai"]
        return {
            "action_type": "post",
            "content_type": "text_post",
            "topic": (obs.get("trending_topics") or ["quick thought"])[0],
            "tags": tt,
        }
    return {"action_type": "rest"}


def _agent_easy_relaxed(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.55 or posts >= 1:
        return {"action_type": "rest"}
    if 14 <= hour <= 16:
        tt = list((obs.get("trending_tags") or [])[:1]) + ["daily"]
        return {
            "action_type": "post",
            "content_type": "story",
            "topic": (obs.get("trending_topics") or ["casual post"])[0],
            "tags": tt,
        }
    return {"action_type": "rest"}


def _agent_medium_queue_cycle(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    queue = obs.get("content_queue_size", 0)
    if energy < 0.35:
        return {"action_type": "rest"}
    if queue == 0 and energy > 0.5:
        return {"action_type": "create_content"}
    if queue > 0 and 10 <= hour <= 19 and posts < 2:
        return {
            "action_type": "post",
            "content_type": "carousel",
            "topic": (obs.get("trending_topics") or ["batch content"])[0],
            "tags": list((obs.get("trending_tags") or [])[:3]),
        }
    if posts >= 2:
        return {"action_type": "rest"}
    return {"action_type": "rest"}


def _agent_medium_trend_rotate(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.4 or posts >= 2:
        return {"action_type": "rest"}
    if 9 <= hour <= 20:
        ct = _CONTENT_TYPES[step % 4]
        tags = list((obs.get("trending_tags") or [])[:2]) + [TAG_POOL[step % len(TAG_POOL)]]
        return {
            "action_type": "post",
            "content_type": ct,
            "topic": (obs.get("trending_topics") or ["trending angle"])[0],
            "tags": tags,
        }
    return {"action_type": "rest"}


def _agent_medium_two_format_day(obs: dict, step: int) -> dict:
    energy = obs.get("creator_energy", 1.0)
    posts = obs.get("posts_today", 0)
    hour = obs.get("current_hour", 12)
    if energy < 0.42 or posts >= 2:
        return {"action_type": "rest"}
    if 11 <= hour <= 13 and posts == 0:
        return {
            "action_type": "post",
            "content_type": "reel",
            "topic": (obs.get("trending_topics") or ["midday hook"])[0],
            "tags": list((obs.get("trending_tags") or [])[:3]),
        }
    if 18 <= hour <= 20 and posts == 1:
        return {
            "action_type": "post",
            "content_type": "carousel",
            "topic": (obs.get("trending_topics") or ["evening recap"])[0],
            "tags": list((obs.get("trending_tags") or [])[:2]) + ["howto"],
        }
    return {"action_type": "rest"}


SCENARIOS = {
    "always_rest": ("Always Rest", "Never posts. Tests follower decay + zero engagement.", _agent_always_rest),
    "spam": ("Spam Post", "Same reel every step. Burns out in 4 steps.", _agent_spam),
    "bad_timing": ("Night Poster", "Off-peak posts and weak topics; low hour multiplier.", _agent_bad_timing),
    "no_rest": ("No Rest", "Varied posts but never rests. Burns out fast.", _agent_no_rest),
    "smart": ("Smart Agent", "Optimal: peak hours, trending, varied types, rests.", _agent_smart),
    "copycat": ("Copycat", "Copies competitor topics. High saturation penalty.", _agent_copycat),
    "queue_optimizer": ("Queue Optimizer", "Creates content first, posts from queue at half energy.", _agent_queue_optimizer),
    "burst": ("Burst Poster", "3 posts in a row, then rests until recovered.", _agent_burst),
    "weekend": ("Weekend Warrior", "Only posts on Sat/Sun. Misses 5 weekdays.", _agent_weekend),
    "tag_explorer": ("Tag Explorer", "New tag combo every post. Max discovery.", _agent_tag_explorer),
    "balanced": ("Balanced Creator", "Create→Post→Rest cycle. Uses queue.", _agent_balanced),
    # New scenarios
    "sleep_deprived": ("Sleep Deprived", "Never rests. Tests sleep deprivation effects.", _agent_sleep_deprived),
    "sleep_conscious": ("Sleep Conscious", "Rests at night. Proper sleep schedule.", _agent_sleep_conscious),
    "minimal": ("Minimal Poster", "Only 1 post per day at noon.", _agent_minimal),
    "story_spammer": ("Story Spammer", "Low-energy story spam strategy.", _agent_story_spammer),
    "reel_max": ("Reel Maximizer", "Only reels at peak hours for max reach.", _agent_reel_max),
    "text_only": ("Text Only", "Low-energy text posts only.", _agent_text_only),
    "early_bird": ("Early Bird", "Posts only 6-10am.", _agent_early_bird),
    "night_owl": ("Night Owl", "Posts only 20-23.", _agent_night_owl),
    "trend_chaser": ("Trend Chaser", "Only posts trending topics.", _agent_trend_chaser),
    "anti_trend": ("Anti-Trend", "Avoids trending for differentiation.", _agent_anti_trend),
    "energy_saver": ("Energy Saver", "Only posts when energy > 0.7.", _agent_energy_saver),
    "queue_heavy": ("Queue Heavy", "3 days prep, then posts from queue.", _agent_queue_heavy),
    "midday": ("Midday Focus", "Posts only 11-14 peak hours.", _agent_midday),
    "random": ("Random Actor", "Random actions. Baseline test.", _agent_random),
    "carousel_only": ("Carousel Only", "Only carousel posts.", _agent_carousel_only),
    "tech_niche": ("Tech Niche", "AI/coding content focus.", _agent_tech_niche),
    "lifestyle_niche": ("Lifestyle Niche", "Fitness/wellness focus.", _agent_lifestyle_niche),
    "high_freq": ("High Frequency", "3 posts per day target.", _agent_high_freq),
    "low_freq": ("Low Frequency", "1 post every 2 days.", _agent_low_freq),
    "comp_avoider": ("Competitor Avoider", "Checks saturation before posting.", _agent_comp_avoider),
    "tue_thu": ("Tuesday Thursday", "Peak weekday targeting.", _agent_tue_thu),
    "monday": ("Monday Motivation", "Only Monday posts.", _agent_monday),
    "tag_exploiter": ("Tag Exploiter", "Uses best-performing tags.", _agent_tag_exploiter),
    "alternating": ("Alternating Format", "Rotates content types.", _agent_alternating),
    "engagement_chaser": ("Engagement Chaser", "Posts more when engagement high.", _agent_engagement_chaser),
    "conservative": ("Conservative Energy", "Never goes below 0.5 energy.", _agent_conservative),
    "aggressive": ("Aggressive Energy", "Pushes to 0.15 energy.", _agent_aggressive),
    "sleep_respecting": ("Sleep Respecting", "Monitors hours_since_sleep.", _agent_sleep_respecting),
    "night_shift": ("Night Shift", "Inverted schedule, sleeps during day.", _agent_night_shift),
    "split_schedule": ("Split Schedule", "Morning and evening posts.", _agent_split_schedule),
    "growth_focus": ("Growth Focus", "Maximizes follower growth.", _agent_growth_focus),
    "content_creator": ("Content Creator", "Heavy content creation ratio.", _agent_content_creator),
    "weekday_only": ("Weekday Only", "No weekend posting.", _agent_weekday_only),
    "double_peak": ("Double Peak", "Posts at 9am and 3pm.", _agent_double_peak),
    "crypto_niche": ("Crypto/Web3", "Crypto content focus.", _agent_crypto_niche),
    "gaming_niche": ("Gaming Niche", "Gaming audience timing.", _agent_gaming_niche),
    "productivity": ("Productivity Guru", "Morning productivity posts.", _agent_productivity),
    "food_creator": ("Food Creator", "Meal-time posting.", _agent_food_creator),
    "travel": ("Travel Blogger", "Travel content strategy.", _agent_travel),
    "fashion": ("Fashion Content", "Fashion niche timing.", _agent_fashion),
    "photography": ("Photography Focus", "Golden hour timing.", _agent_photography),
    "stoic": ("Stoic Philosophy", "Minimal daily wisdom.", _agent_stoic),
    "saas": ("SaaS/Business", "B2B content timing.", _agent_saas),
    "creator_economy": ("Creator Economy", "Monetization focus.", _agent_creator_economy),
    "ml_deep": ("ML/AI Deep Dive", "Technical content.", _agent_ml_deep),
    "sleep_debt_aware": ("Sleep Debt Aware", "Monitors sleep_debt field.", _agent_sleep_debt_aware),
    "marathon": ("Marathon Runner", "Extended awake, tests deprivation.", _agent_marathon),
    "napper": ("Napper", "Strategic short rests.", _agent_napper),
    "optimal_sleep": ("Optimal Sleep", "8-hour sleep blocks.", _agent_optimal_sleep),
    "events": ("Events/News", "Event-based trending tags.", _agent_events),
    "easy_morning_story": (
        "Easy: Morning story",
        "One low-cost story between 9–11am when energy allows.",
        _agent_easy_morning_story,
    ),
    "easy_one_a_day": (
        "Easy: One text at 1pm",
        "Single text_post at hour 13 if energy stays above 0.5.",
        _agent_easy_one_a_day,
    ),
    "easy_relaxed": (
        "Easy: Afternoon story",
        "One story between 2–4pm; rests until energy > 0.55.",
        _agent_easy_relaxed,
    ),
    "medium_queue_cycle": (
        "Medium: Create then post",
        "Builds queue then publishes carousels in peak hours.",
        _agent_medium_queue_cycle,
    ),
    "medium_trend_rotate": (
        "Medium: Trend + format rotation",
        "Cycles content types with trending topics plus a pool tag.",
        _agent_medium_trend_rotate,
    ),
    "medium_two_format": (
        "Medium: Reel + carousel day",
        "Midday reel then evening carousel when slots open.",
        _agent_medium_two_format_day,
    ),
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
    global _sm_ct, _sm_tg, _SIM_RNG, _q7_phase, _burst_count, _te_idx, _bc_phase
    global _qh_phase, _alt_idx, _ccm_ratio, _nap_count
    _sm_ct = 0
    _sm_tg = 0
    _q7_phase = "prep"
    _qh_phase = "prep"
    _burst_count = 0
    _te_idx = 0
    _bc_phase = 0
    _alt_idx = 0
    _ccm_ratio = 0
    _nap_count = 0
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
    post_count = sum(1 for s in steps if s["action"].startswith("post"))
    _save_history_entry({
        "id": datetime.now(timezone.utc).isoformat(),
        "scenario": label,
        "scenario_id": scenario_id,
        "description": desc,
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
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    if args.port is not None:
        main(port=args.port)
    else:
        main()
