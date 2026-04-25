"""
Viraltest Environment v2 — Theme #3.1 World-Modeling Simulation.

30-day creator optimization with:
- Mosseri-aligned engagement signals (watch_time, sends, saves, likes)
- Discoverable tool catalog (partial observability)
- Piecewise-linear sleep model (Van Dongen 2003)
- Data-driven hour heatmap (Buffer 9.6M + Sprout 2B)
- Tiered audience fatigue (Buffer 2.1M)
- Multi-episode brand persistence
- Counterfactual coach feedback
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        CollabProposal,
        EngagementSignals,
        ReplyAction,
        ScheduledAction,
        ToolCall,
        ToolResult,
        ViraltestAction,
        ViraltestObservation,
    )
except ImportError:
    from models import (
        CollabProposal,
        EngagementSignals,
        ReplyAction,
        ScheduledAction,
        ToolCall,
        ToolResult,
        ViraltestAction,
        ViraltestObservation,
    )

_DATA_DIR = Path(__file__).parent / "data"

def _load_json(name: str) -> Any:
    return json.loads((_DATA_DIR / name).read_text())

# ---------------------------------------------------------------------------
# Data files (loaded once at module level)
# ---------------------------------------------------------------------------

_TAGS_DATA = _load_json("tags.json")
_TOPICS_DATA = _load_json("topics.json")
_COMPETITORS_DATA = _load_json("competitors.json")
_HEATMAP_DATA = _load_json("hour_heatmap.json")
_AUDIENCE_DATA = _load_json("audience_segments.json")
_OVERLAP_DATA = _load_json("audience_overlap_matrix.json")

# Flatten tag pool for validation
TAG_POOL: List[str] = []
for t in _TAGS_DATA.get("broad", []):
    TAG_POOL.append(t["tag"])
for _cat, tags in _TAGS_DATA.get("niche", {}).items():
    for t in tags:
        TAG_POOL.append(t["tag"])
for t in _TAGS_DATA.get("trending", []):
    TAG_POOL.append(t["tag"])
for t in _TAGS_DATA.get("seasonal", []):
    TAG_POOL.append(t["tag"])

TOPIC_CATEGORIES: Dict[str, List[str]] = {}
for niche_name, niche_data in _TOPICS_DATA.get("niches", {}).items():
    TOPIC_CATEGORIES[niche_name] = niche_data["topics"]

_NICHE_MULTIPLIERS: Dict[str, float] = {}
for niche_name, niche_data in _TOPICS_DATA.get("niches", {}).items():
    _NICHE_MULTIPLIERS[niche_name] = niche_data["engagement_multiplier"]

_HEATMAP_GRID: Dict[int, List[float]] = {
    int(k): v for k, v in _HEATMAP_DATA.get("grid", {}).items()
}

# ---------------------------------------------------------------------------
# Constants (research-backed, Tier 1-3 sources)
# ---------------------------------------------------------------------------

TASK_HORIZON = 30  # 30 daily steps (monthly cycle)

# Socialinsider 2026 (31M posts)
CONTENT_ENERGY_COST = {
    "reel": 0.25,
    "carousel": 0.20,
    "story": 0.08,
    "text_post": 0.06,
}

BASE_ENGAGEMENT = {
    "reel": 0.52,
    "carousel": 0.55,
    "story": 0.30,
    "text_post": 0.45,
}

# Socialinsider 2026 + CreatorsJet 10K study
REACH_MULT = {
    "reel": 2.25,
    "carousel": 1.0,
    "story": 0.5,
    "text_post": 0.91,
}

# Mosseri Jan-2025: format→signal affinity (which signal each format naturally excels at)
FORMAT_SIGNAL_WEIGHTS = {
    "reel":      {"watch_time": 0.50, "sends_per_reach": 0.25, "saves": 0.10, "likes_per_reach": 0.15},
    "carousel":  {"watch_time": 0.10, "sends_per_reach": 0.15, "saves": 0.50, "likes_per_reach": 0.25},
    "story":     {"watch_time": 0.20, "sends_per_reach": 0.40, "saves": 0.05, "likes_per_reach": 0.35},
    "text_post": {"watch_time": 0.05, "sends_per_reach": 0.10, "saves": 0.30, "likes_per_reach": 0.55},
}

# Intent multiplier matrix: when intent matches format's strong signal, boost that signal
INTENT_MULTIPLIER = {
    "send_bait":  {"sends_per_reach": 1.6},
    "save_bait":  {"saves": 1.7},
    "watch_bait": {"watch_time": 1.5},
    "like_bait":  {"likes_per_reach": 1.3},
}

VALID_TASKS = ("monthly_engage", "monthly_strategic", "monthly_competitive")

INITIAL_FOLLOWERS = 10000
REST_RECOVERY = 0.12
CREATE_CONTENT_COST = 0.05
REPETITION_ENERGY_PENALTY = 0.05
FOLLOWER_DECAY_HOURS = 72
ALGORITHM_PENALTY_MULT = 0.6
ALGORITHM_PENALTY_BASE_DURATION = 2

# Van Dongen 2003 *Sleep* PMID 12683469: lapses linear above 15.84h
SLEEP_OPTIMAL_AWAKE = 16
SLEEP_LINEAR_DECAY_PER_HOUR = 0.0625  # reaches ~50% at 24h awake (8h × 0.0625 = 0.5)
SLEEP_MIN_QUALITY = 0.30
SLEEP_ENERGY_DRAIN_START = 16
SLEEP_ENERGY_DRAIN_RATE = 0.015
SLEEP_RECOVERY_PER_REST = 2

# Buffer 2.1M study + arxiv:2410.13108: tiered fatigue
FATIGUE_TIERS = {2: 1.0, 3: 0.75, 4: 0.50, 5: 0.25}
WEEKLY_FATIGUE_THRESHOLD = 7
WEEKLY_FATIGUE_MULT = 0.75

SATURATION_PENALTY_K = 0.25
TREND_DEFAULT_HALFLIFE_HOURS = 60
COLLAB_MAX_PER_MONTH = 2
REPLY_WINDOW_MINUTES = 90
REPLY_REACH_BONUS = 1.4
API_BUDGET_INITIAL = 100

# Tool costs
TOOL_COSTS = {
    "query_audience": 2,
    "query_competitor": 2,
    "query_tag_history": 1,
    "query_trends": 1,
    "predict_engagement": 3,
    "draft_review": 3,
    "query_creator_pool": 1,
    "propose_collab": 5,
}

# ---------------------------------------------------------------------------
# Brand state for multi-episode persistence
# ---------------------------------------------------------------------------

_BRAND_STORE: Dict[str, Dict[str, Any]] = {}


@dataclass
class CompetitorState:
    id: str
    name: str
    niche: str
    niche_topics: List[str]
    preferred_types: List[str]
    posts_per_week: float
    base_engagement_rate: float
    tag_preferences: List[str]
    style: str
    recent_posts: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool catalog (schemas for GET /tools)
# ---------------------------------------------------------------------------

TOOL_CATALOG = {
    "query_audience": {
        "description": "Query a specific audience segment to learn its topic affinities, content preferences, and active hours.",
        "parameters": {"segment_id": {"type": "string", "enum": [s["id"] for s in _AUDIENCE_DATA.get("segments", [])]}},
    },
    "query_competitor": {
        "description": "Get recent posts and strategy of a competitor archetype within a time window.",
        "parameters": {
            "competitor_id": {"type": "string", "enum": [a["id"] for a in _COMPETITORS_DATA.get("archetypes", [])]},
            "window_days": {"type": "integer", "default": 7, "minimum": 1, "maximum": 30},
        },
    },
    "query_tag_history": {
        "description": "Get your historical engagement signals (watch, sends, saves, likes) for a specific tag.",
        "parameters": {"tag": {"type": "string"}},
    },
    "query_trends": {
        "description": "Get currently trending topics and tags for a niche, with decay-adjusted strength.",
        "parameters": {"niche": {"type": "string", "enum": list(TOPIC_CATEGORIES.keys())}},
    },
    "predict_engagement": {
        "description": "Simulate engagement signals for a hypothetical daily plan WITHOUT committing it. Returns predicted watch/sends/saves/likes.",
        "parameters": {"scheduled_actions": {"type": "array", "description": "Same format as ViraltestAction.scheduled_actions"}},
    },
    "draft_review": {
        "description": "Get AI review of a draft plan: strengths, weaknesses, suggested improvements.",
        "parameters": {"scheduled_actions": {"type": "array"}},
    },
    "query_creator_pool": {
        "description": "List available competitor archetypes for potential collaboration, with audience overlap %.",
        "parameters": {},
    },
    "propose_collab": {
        "description": "Propose a collaboration post with a competitor. Splits engagement by audience overlap. Max 2 per month.",
        "parameters": {
            "partner_id": {"type": "string"},
            "content_type": {"type": "string", "enum": ["reel", "story", "carousel", "text_post"]},
            "hour": {"type": "integer", "minimum": 0, "maximum": 23},
        },
    },
}


class ViraltestEnvironment(Environment):
    """Monthly creator optimization simulation (Theme #3.1 World Modeling)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = "monthly_engage"
        self._rng = random.Random(42)
        self._init_state()

    def _init_state(self) -> None:
        self._energy = 1.0
        self._followers = INITIAL_FOLLOWERS
        self._initial_followers = INITIAL_FOLLOWERS
        self._hour = 9
        self._day = 0
        self._posts_today = 0
        self._last_post_types: List[str] = []
        self._time_since_last_post = 0
        self._engagement_history: List[float] = []
        self._tag_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._content_queue = 0
        self._unique_tags_used: set = set()
        self._unique_content_types: set = set()
        self._energy_history: List[float] = [1.0]
        self._posting_steps = 0
        self._episode_done = False
        self._last_topic: Optional[str] = None
        self._final_observation: Optional[ViraltestObservation] = None
        self._unique_topic_steps = 0
        self._days_with_good_posts: set = set()
        self._total_engagement = 0.0
        self._posts_per_day: Dict[int, int] = defaultdict(int)
        self._algorithm_penalty_remaining = 0
        self._agent_notes: Optional[str] = None
        self._api_budget = API_BUDGET_INITIAL
        self._collabs_this_month = 0
        self._collab_history: List[str] = []
        self._active_collab: Optional[Tuple[int, float]] = None
        self._low_energy_days = 0
        self._total_posts_this_week = 0
        self._week_start_day = 0
        self._daily_signals = EngagementSignals()
        self._tool_calls_total = 0
        self._unique_tools_used: set = set()

        self._trending_topics = self._pick_trending_topics()
        self._trending_tags = self._pick_trending_tags()
        self._competitors = self._load_competitors()

        self._hours_since_sleep = 2
        self._sleep_debt = 0.0

    def _load_competitors(self) -> List[CompetitorState]:
        archetypes = _COMPETITORS_DATA.get("archetypes", [])
        return [
            CompetitorState(
                id=a["id"],
                name=a["name"],
                niche=a["niche"],
                niche_topics=a["niche_topics"],
                preferred_types=a["preferred_types"],
                posts_per_week=a["posts_per_week"],
                base_engagement_rate=a["base_engagement_rate"],
                tag_preferences=a["tag_preferences"],
                style=a.get("style", "consistent_moderate"),
            )
            for a in archetypes
        ]

    def _pick_trending_topics(self) -> List[str]:
        all_topics = []
        for niche_data in _TOPICS_DATA.get("niches", {}).values():
            all_topics.extend(niche_data["topics"])
        return self._rng.sample(all_topics, min(3, len(all_topics)))

    def _pick_trending_tags(self) -> List[str]:
        return self._rng.sample(TAG_POOL, min(5, len(TAG_POOL)))

    def _rotate_trends(self) -> None:
        self._trending_topics = self._pick_trending_topics()
        self._trending_tags = self._pick_trending_tags()

    # ----- hour multiplier (heatmap-based) -----

    def _get_hour_multiplier(self) -> float:
        dow = self._day % 7
        h = self._hour
        row = _HEATMAP_GRID.get(dow)
        if row and 0 <= h < len(row):
            return row[h]
        return 0.8

    # ----- quality (piecewise-linear sleep, Van Dongen 2003) -----

    def _get_quality_modifier(self) -> float:
        if self._energy > 0.5:
            energy_factor = 1.0
        else:
            energy_factor = max(0.48, self._energy * 1.5)

        if self._hours_since_sleep <= SLEEP_OPTIMAL_AWAKE:
            sleep_factor = 1.0
        else:
            hours_over = self._hours_since_sleep - SLEEP_OPTIMAL_AWAKE
            sleep_factor = max(SLEEP_MIN_QUALITY, 1.0 - SLEEP_LINEAR_DECAY_PER_HOUR * hours_over)

        return energy_factor * sleep_factor

    # ----- niche multiplier -----

    def _get_niche_multiplier(self, topic: Optional[str]) -> float:
        if not topic:
            return 1.0
        topic_lower = topic.lower()
        for niche_name, niche_data in _TOPICS_DATA.get("niches", {}).items():
            for t in niche_data["topics"]:
                if t.lower() == topic_lower:
                    return _NICHE_MULTIPLIERS.get(niche_name, 1.0)
        return 1.0

    # ----- tags -----

    def _calc_tag_boost(self, tags: Optional[List[str]]) -> float:
        if not tags:
            return 1.0
        trending_count = sum(1 for t in tags if t in self._trending_tags)
        perf_values = [self._tag_performance_avg(t) for t in tags if self._tag_performance_avg(t) > 0]
        perf_avg = sum(perf_values) / len(perf_values) if perf_values else 0.0
        return 1.0 + 0.1 * trending_count + 0.05 * perf_avg

    def _tag_performance_avg(self, tag: str) -> float:
        history = self._tag_history.get(tag, [])
        if not history:
            return 0.0
        window = history[-5:]
        totals = [h.get("total", 0.0) for h in window]
        return sum(totals) / len(totals) if totals else 0.0

    def _get_tag_performance_dict(self) -> Dict[str, float]:
        return {tag: self._tag_performance_avg(tag) for tag in self._unique_tags_used}

    # ----- competitors -----

    def _advance_competitors(self) -> None:
        for comp in self._competitors:
            for p in comp.recent_posts:
                p["hours_ago"] += 1
            comp.recent_posts = [p for p in comp.recent_posts if p["hours_ago"] < 72]

            daily_prob = comp.posts_per_week / (7.0 * 24.0)
            if self._rng.random() < daily_prob:
                ct = self._rng.choice(comp.preferred_types)
                topic = self._rng.choice(comp.niche_topics)
                tags = self._rng.sample(comp.tag_preferences, min(3, len(comp.tag_preferences)))
                eng = comp.base_engagement_rate + self._rng.uniform(-0.1, 0.1)
                eng = max(0.0, min(1.0, eng))
                comp.recent_posts.append({
                    "content_type": ct, "topic": topic, "tags": tags,
                    "engagement": round(eng, 3), "hours_ago": 0,
                })

    def _get_competitor_recent_posts(self, limit: int = 5) -> List[Dict[str, Any]]:
        all_posts: List[Dict[str, Any]] = []
        for comp in self._competitors:
            for p in comp.recent_posts:
                all_posts.append(p)
        all_posts.sort(key=lambda x: x["hours_ago"])
        return all_posts[:limit]

    def _get_competitor_avg_engagement(self) -> float:
        engagements = [p["engagement"] for comp in self._competitors for p in comp.recent_posts]
        return sum(engagements) / len(engagements) if engagements else 0.0

    def _calc_niche_saturation(self, topic: Optional[str]) -> float:
        if not topic:
            return 0.0
        recent_topics = []
        for comp in self._competitors:
            for p in comp.recent_posts:
                if p["hours_ago"] < 12:
                    recent_topics.append(p["topic"].lower())
        if not recent_topics:
            return 0.0
        topic_lower = topic.lower()
        overlap = sum(1 for t in recent_topics if _topic_overlap(topic_lower, t))
        return min(1.0, overlap / max(1, len(recent_topics)))

    def _calc_competitor_diff(self, topic: Optional[str]) -> float:
        if not topic:
            return 1.0
        saturation = self._calc_niche_saturation(topic)
        recent_topics = [
            p["topic"].lower()
            for comp in self._competitors
            for p in comp.recent_posts
            if p["hours_ago"] < 12
        ]
        has_overlap = any(_topic_overlap(topic.lower(), t) for t in recent_topics)
        if not has_overlap:
            return 1.3
        if saturation > 0.7:
            return 0.6
        return 1.0

    def _count_competitors_same_hour(self) -> int:
        count = 0
        for comp in self._competitors:
            for p in comp.recent_posts:
                if p["hours_ago"] <= 1:
                    count += 1
        return count

    # ----- fatigue (tiered, Buffer 2.1M) -----

    def _get_fatigue_multiplier(self) -> float:
        if self._posts_today <= 2:
            daily_fatigue = 1.0
        elif self._posts_today in FATIGUE_TIERS:
            daily_fatigue = FATIGUE_TIERS[self._posts_today]
        else:
            daily_fatigue = 0.25

        weekly_mult = 1.0
        if self._total_posts_this_week >= WEEKLY_FATIGUE_THRESHOLD:
            weekly_mult = WEEKLY_FATIGUE_MULT

        return daily_fatigue * weekly_mult

    # ----- engagement signals (Mosseri-aligned) -----

    def _compute_engagement_signals(
        self, content_type: str, base_eng: float, intent: Optional[str]
    ) -> EngagementSignals:
        weights = FORMAT_SIGNAL_WEIGHTS.get(content_type, FORMAT_SIGNAL_WEIGHTS["text_post"])
        signals = {k: base_eng * v for k, v in weights.items()}

        if intent and intent in INTENT_MULTIPLIER:
            for signal_name, mult in INTENT_MULTIPLIER[intent].items():
                if signal_name in signals:
                    signals[signal_name] *= mult

        return EngagementSignals(**signals)

    # ----- tool dispatcher -----

    def _dispatch_tool(self, tool: ToolCall) -> ToolResult:
        cost = TOOL_COSTS.get(tool.name, 1)
        if self._api_budget < cost:
            return ToolResult(name=tool.name, success=False, error="rate_limit_exceeded", budget_remaining=self._api_budget)

        self._api_budget -= cost
        self._tool_calls_total += 1
        self._unique_tools_used.add(tool.name)

        if tool.name == "query_audience":
            seg_id = tool.arguments.get("segment_id", "")
            for seg in _AUDIENCE_DATA.get("segments", []):
                if seg["id"] == seg_id:
                    return ToolResult(name=tool.name, data=seg, budget_remaining=self._api_budget)
            return ToolResult(name=tool.name, success=False, error=f"unknown segment: {seg_id}", budget_remaining=self._api_budget)

        elif tool.name == "query_competitor":
            comp_id = tool.arguments.get("competitor_id", "")
            window = tool.arguments.get("window_days", 7)
            for comp in self._competitors:
                if comp.id == comp_id:
                    posts = [p for p in comp.recent_posts if p["hours_ago"] < window * 24]
                    return ToolResult(name=tool.name, data={
                        "id": comp.id, "name": comp.name, "niche": comp.niche,
                        "posts_per_week": comp.posts_per_week,
                        "recent_posts": posts[:10],
                        "avg_engagement": round(sum(p["engagement"] for p in posts) / max(1, len(posts)), 3),
                    }, budget_remaining=self._api_budget)
            return ToolResult(name=tool.name, success=False, error=f"unknown competitor: {comp_id}", budget_remaining=self._api_budget)

        elif tool.name == "query_tag_history":
            tag = tool.arguments.get("tag", "").lower()
            history = self._tag_history.get(tag, [])
            return ToolResult(name=tool.name, data={
                "tag": tag, "uses": len(history),
                "avg_signals": _avg_signal_dicts(history[-10:]) if history else {},
            }, budget_remaining=self._api_budget)

        elif tool.name == "query_trends":
            niche = tool.arguments.get("niche", "tech")
            return ToolResult(name=tool.name, data={
                "trending_topics": self._trending_topics,
                "trending_tags": self._trending_tags,
                "niche_saturation": round(self._calc_niche_saturation(self._last_topic), 3),
            }, budget_remaining=self._api_budget)

        elif tool.name == "predict_engagement":
            raw_actions = tool.arguments.get("scheduled_actions", [])
            predicted_total = 0.0
            for sa_dict in raw_actions[:5]:
                sa = ScheduledAction(**sa_dict) if isinstance(sa_dict, dict) else sa_dict
                if sa.action_type == "post" and sa.content_type:
                    base = BASE_ENGAGEMENT.get(sa.content_type, 0.3)
                    reach = REACH_MULT.get(sa.content_type, 1.0)
                    niche_m = self._get_niche_multiplier(sa.topic)
                    predicted_total += base * reach * niche_m * self._get_hour_multiplier()
            return ToolResult(name=tool.name, data={"predicted_daily_engagement": round(predicted_total, 4)}, budget_remaining=self._api_budget)

        elif tool.name == "draft_review":
            raw_actions = tool.arguments.get("scheduled_actions", [])
            n_posts = sum(1 for a in raw_actions if (a.get("action_type") if isinstance(a, dict) else getattr(a, "action_type", "")) == "post")
            feedback = []
            if n_posts == 0:
                feedback.append("No posts planned — you'll lose algorithmic momentum.")
            elif n_posts > 3:
                feedback.append(f"{n_posts} posts in one day risks audience fatigue (optimal: 1-2).")
            if n_posts >= 1 and n_posts <= 2:
                feedback.append("Good posting frequency for today.")
            return ToolResult(name=tool.name, data={"feedback": feedback, "post_count": n_posts}, budget_remaining=self._api_budget)

        elif tool.name == "query_creator_pool":
            pool = [
                {
                    "id": comp.id,
                    "name": comp.name,
                    "niche": comp.niche,
                    "max_audience_overlap": round(_partner_max_overlap(comp.id), 2),
                }
                for comp in self._competitors
            ]
            return ToolResult(name=tool.name, data=pool, budget_remaining=self._api_budget)

        elif tool.name == "propose_collab":
            if self._collabs_this_month >= COLLAB_MAX_PER_MONTH:
                return ToolResult(name=tool.name, success=False, error="collab_limit_reached", budget_remaining=self._api_budget)
            partner_id = tool.arguments.get("partner_id", "")
            if partner_id in self._collab_history[-3:]:
                return ToolResult(name=tool.name, success=False, error="recently_collaborated", budget_remaining=self._api_budget)
            return ToolResult(name=tool.name, data={"status": "proposal_accepted", "partner_id": partner_id}, budget_remaining=self._api_budget)

        return ToolResult(name=tool.name, success=False, error=f"unknown tool: {tool.name}", budget_remaining=self._api_budget)

    # ----- counterfactual coach -----

    def _compute_coach_feedback(self, agent_engagement: float) -> Dict[str, Any]:
        dow = self._day % 7
        row = _HEATMAP_GRID.get(dow, [1.0] * 24)
        best_hours = sorted(range(24), key=lambda h: row[h] if h < len(row) else 0, reverse=True)[:2]
        best_base = max(BASE_ENGAGEMENT.values())
        best_reach = max(REACH_MULT.values())
        optimal_eng = sum(row[h] * best_base * best_reach for h in best_hours)
        delta = agent_engagement - optimal_eng
        return {
            "optimal_hours": best_hours,
            "optimal_engagement_estimate": round(optimal_eng, 4),
            "your_engagement": round(agent_engagement, 4),
            "delta": round(delta, 4),
            "suggestion": "You're outperforming the heatmap baseline!" if delta >= 0 else "Consider posting at peak hours for better reach.",
        }

    # ----- core API -----

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ViraltestObservation:
        self._task = kwargs.get("task", "monthly_engage")
        if self._task not in VALID_TASKS:
            self._task = "monthly_engage"

        self._rng = random.Random(seed if seed is not None else 42)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._init_state()

        chain_id = kwargs.get("episode_chain_id")
        if chain_id and chain_id in _BRAND_STORE:
            brand = _BRAND_STORE[chain_id]
            self._unique_tags_used = set(brand.get("top_tags", []))
            self._unique_content_types = set(brand.get("dominant_types", []))
            self._collab_history = brand.get("collab_history", [])
            self._followers = brand.get("followers", INITIAL_FOLLOWERS)
            self._initial_followers = self._followers

        return self._build_observation(reward=0.0, error=None)

    def step(self, action: ViraltestAction, **kwargs: Any) -> ViraltestObservation:
        if self._episode_done and self._final_observation is not None:
            return self._final_observation

        self._state.step_count += 1

        # Store agent notes for echo
        if action.notes:
            self._agent_notes = action.notes

        # Process tool calls first
        tool_results: List[ToolResult] = []
        for tc in action.tool_calls:
            result = self._dispatch_tool(tc)
            tool_results.append(result)

        # Process collab proposal — arms an hour-targeted engagement boost for today
        self._active_collab = None
        if action.collab and self._collabs_this_month < COLLAB_MAX_PER_MONTH:
            self._collabs_this_month += 1
            self._collab_history.append(action.collab.partner_id)
            self._active_collab = (action.collab.hour, _partner_max_overlap(action.collab.partner_id))

        # Validate scheduled actions
        schedule: Dict[int, ScheduledAction] = {}
        errors: List[str] = []
        for sa in action.scheduled_actions:
            if sa.hour < 0 or sa.hour > 23:
                errors.append(f"Invalid hour: {sa.hour}")
                continue
            err = self._validate_scheduled_action(sa)
            if err:
                errors.append(f"hour {sa.hour}: {err}")
                continue
            schedule[sa.hour] = sa

        daily_engagement = 0.0
        daily_reward = 0.0
        daily_posts = 0
        energy_min = self._energy
        burned_out = False
        daily_signals = EngagementSignals()

        for hour in range(24):
            if burned_out:
                break
            self._hour = hour

            if hour in schedule:
                sa = schedule[hour]
                hourly_eng, hourly_reward, hourly_signals = self._process_hour_action(sa)
            else:
                hourly_eng, hourly_reward = self._process_hour_rest()
                hourly_signals = None

            daily_engagement += hourly_eng
            daily_reward += hourly_reward
            if hourly_eng > 0:
                daily_posts += 1
            if hourly_signals:
                daily_signals = EngagementSignals(
                    watch_time=daily_signals.watch_time + hourly_signals.watch_time,
                    sends_per_reach=daily_signals.sends_per_reach + hourly_signals.sends_per_reach,
                    saves=daily_signals.saves + hourly_signals.saves,
                    likes_per_reach=daily_signals.likes_per_reach + hourly_signals.likes_per_reach,
                )
            energy_min = min(energy_min, self._energy)
            self._advance_competitors()
            self._advance_time()
            self._energy_history.append(self._energy)

            if self._energy <= 0.0:
                burned_out = True

        # Process replies
        for reply in action.replies:
            if 0 <= reply.reply_hour < 24 and 0 <= reply.post_hour < 24:
                diff_minutes = abs(reply.reply_hour - reply.post_hour) * 60
                if diff_minutes <= REPLY_WINDOW_MINUTES:
                    daily_engagement *= REPLY_REACH_BONUS
                    daily_signals = EngagementSignals(
                        watch_time=daily_signals.watch_time * REPLY_REACH_BONUS,
                        sends_per_reach=daily_signals.sends_per_reach * REPLY_REACH_BONUS,
                        saves=daily_signals.saves * REPLY_REACH_BONUS,
                        likes_per_reach=daily_signals.likes_per_reach * REPLY_REACH_BONUS,
                    )

        # Weekly tracking
        self._total_posts_this_week += daily_posts
        if self._day % 7 == 0 and self._day > 0:
            self._total_posts_this_week = 0

        # Burnout risk tracking
        if energy_min < 0.2:
            self._low_energy_days += 1
        else:
            self._low_energy_days = max(0, self._low_energy_days - 1)

        prev_day = max(0, self._day - 1)
        if 1 <= self._posts_per_day.get(prev_day, 0) <= 2:
            self._days_with_good_posts.add(prev_day)

        avg_reward = daily_reward / 24.0
        error_str = "; ".join(errors) if errors else None

        done = self._state.step_count >= TASK_HORIZON or self._energy <= 0.0
        coach = self._compute_coach_feedback(daily_engagement)

        if done:
            self._episode_done = True
            grader_score, rubric_scores, rubric_evidence = self._run_grader()

            chain_id = kwargs.get("episode_chain_id")
            if chain_id:
                top_tags = sorted(self._unique_tags_used, key=lambda t: self._tag_performance_avg(t), reverse=True)[:3]
                _BRAND_STORE[chain_id] = {
                    "top_tags": list(top_tags),
                    "dominant_types": list(self._unique_content_types),
                    "collab_history": self._collab_history[-3:],
                    "followers": self._followers,
                }

            self._final_observation = self._build_observation(
                reward=round(avg_reward, 4), error=error_str, done=True,
                grader_score=grader_score, daily_total_engagement=daily_engagement,
                daily_posts_made=daily_posts, daily_energy_min=energy_min,
                tool_results=tool_results, engagement_signals=daily_signals,
                coach_feedback=coach,
                rubric_scores=rubric_scores, rubric_evidence=rubric_evidence,
            )
            return self._final_observation

        return self._build_observation(
            reward=round(avg_reward, 4), error=error_str,
            daily_total_engagement=daily_engagement,
            daily_posts_made=daily_posts, daily_energy_min=energy_min,
            tool_results=tool_results, engagement_signals=daily_signals,
            coach_feedback=coach,
        )

    def _process_hour_action(self, sa: ScheduledAction) -> Tuple[float, float, Optional[EngagementSignals]]:
        engagement = 0.0
        signals = None

        if sa.action_type == "post":
            cost = CONTENT_ENERGY_COST.get(sa.content_type, 0.1)
            if self._content_queue > 0:
                cost *= 0.5
                self._content_queue -= 1
            if len(self._last_post_types) >= 3 and all(t == sa.content_type for t in self._last_post_types[-3:]):
                cost += REPETITION_ENERGY_PENALTY
            self._energy = max(0.0, self._energy - cost)
            self._unique_content_types.add(sa.content_type)

            if self._energy <= 0.0:
                engagement = 0.0
            else:
                base = BASE_ENGAGEMENT.get(sa.content_type, 0.3)
                reach = REACH_MULT.get(sa.content_type, 1.0)
                hour_mult = self._get_hour_multiplier()
                quality = self._get_quality_modifier()
                tag_boost = self._calc_tag_boost(sa.tags)
                trending_bonus = 1.5 if self._is_topic_trending(sa.topic) else 1.0
                comp_diff = self._calc_competitor_diff(sa.topic)
                fatigue = self._get_fatigue_multiplier()
                niche_mult = self._get_niche_multiplier(sa.topic)

                n_comp_same_hour = self._count_competitors_same_hour()
                saturation_factor = 1.0 / (1.0 + SATURATION_PENALTY_K * n_comp_same_hour)

                algo_mult = 1.0
                if self._algorithm_penalty_remaining > 0:
                    algo_mult = ALGORITHM_PENALTY_MULT
                    self._algorithm_penalty_remaining -= 1

                collab_mult = 1.0
                if self._active_collab and self._active_collab[0] == self._hour:
                    collab_mult = 1.0 + self._active_collab[1]
                    self._active_collab = None

                engagement = (
                    base * reach * hour_mult * quality * tag_boost
                    * trending_bonus * comp_diff * fatigue * algo_mult
                    * niche_mult * saturation_factor * collab_mult
                )
                engagement = min(engagement, 5.0)

                signals = self._compute_engagement_signals(sa.content_type, engagement, sa.intent)

            self._last_topic = sa.topic

            if sa.tags and engagement > 0:
                signal_dict = signals.model_dump() if signals else {"total": engagement}
                signal_dict["total"] = engagement
                for tag in sa.tags:
                    tag_lower = tag.lower()
                    self._tag_history[tag_lower].append(signal_dict)
                    self._unique_tags_used.add(tag_lower)

            self._engagement_history.append(engagement)
            self._total_engagement += engagement
            self._posting_steps += 1

            if self._calc_competitor_diff(sa.topic) >= 1.3:
                self._unique_topic_steps += 1

            self._last_post_types.append(sa.content_type)
            if len(self._last_post_types) > 3:
                self._last_post_types = self._last_post_types[-3:]
            self._posts_today += 1
            self._posts_per_day[self._day] += 1
            self._time_since_last_post = 0

            if engagement > 0:
                self._followers += int(engagement * 100)

        elif sa.action_type == "create_content":
            self._energy = max(0.0, self._energy - CREATE_CONTENT_COST)
            self._content_queue += 1
            self._time_since_last_post += 1

        if self._time_since_last_post >= FOLLOWER_DECAY_HOURS:
            self._followers = max(0, self._followers - int(self._followers * 0.005))
            if self._algorithm_penalty_remaining == 0:
                gap_days = self._time_since_last_post // 24
                self._algorithm_penalty_remaining = ALGORITHM_PENALTY_BASE_DURATION + gap_days

        reward = 0.0 if self._energy <= 0.0 else self._compute_hourly_reward(sa, engagement)
        return engagement, reward, signals

    def _process_hour_rest(self) -> Tuple[float, float]:
        self._energy = min(1.0, self._energy + REST_RECOVERY)
        self._hours_since_sleep = max(0, self._hours_since_sleep - SLEEP_RECOVERY_PER_REST)
        self._sleep_debt = max(0.0, self._sleep_debt - 0.1)
        self._time_since_last_post += 1

        if self._time_since_last_post >= FOLLOWER_DECAY_HOURS:
            self._followers = max(0, self._followers - int(self._followers * 0.005))
            if self._algorithm_penalty_remaining == 0:
                gap_days = self._time_since_last_post // 24
                self._algorithm_penalty_remaining = ALGORITHM_PENALTY_BASE_DURATION + gap_days

        reward = 0.0 if self._energy <= 0.0 else self._compute_rest_reward()
        return 0.0, reward

    @property
    def state(self) -> State:
        return self._state

    def _validate_scheduled_action(self, sa: ScheduledAction) -> Optional[str]:
        if sa.action_type not in ("post", "create_content"):
            return f"Invalid action_type: {sa.action_type}"
        if sa.action_type == "post":
            if not sa.content_type:
                return "content_type is required when posting"
            if sa.content_type not in CONTENT_ENERGY_COST:
                return f"Invalid content_type: {sa.content_type}"
            if not sa.topic or not sa.topic.strip():
                return "topic is required when posting"
            if len(sa.topic) > 200:
                return "topic must be <= 200 characters"
            if sa.tags:
                valid = [t for t in sa.tags if t.lower() in [tp.lower() for tp in TAG_POOL]]
                sa.tags = valid if valid else None
        return None

    def _is_topic_trending(self, topic: Optional[str]) -> bool:
        if not topic:
            return False
        topic_lower = topic.lower()
        return any(t.lower() in topic_lower for t in self._trending_topics)

    # ----- reward -----

    def _compute_hourly_reward(self, sa: ScheduledAction, engagement: float) -> float:
        eng_component = min(1.0, engagement / 2.0) * 0.3

        prev_energy = self._energy_history[-2] if len(self._energy_history) >= 2 else 1.0
        energy_delta = self._energy - prev_energy
        energy_component = max(0.0, min(1.0, (energy_delta + 0.3) / 0.6)) * 0.15

        day_posts = self._posts_per_day.get(self._day, 0)
        if 1 <= day_posts <= 2:
            consistency = 1.0
        elif day_posts == 0 or day_posts == 3:
            consistency = 0.5
        else:
            consistency = 0.0
        consistency_component = consistency * 0.15

        tag_component = 0.0
        if sa.action_type == "post" and sa.tags:
            trending_match = sum(1 for t in sa.tags if t.lower() in self._trending_tags) / 5.0
            tag_component = min(1.0, trending_match + 0.3) * 0.15

        comp_component = 0.0
        if sa.action_type == "post":
            diff = self._calc_competitor_diff(sa.topic)
            comp_component = min(1.0, diff / 1.3) * 0.15

        burnout_penalty = 0.1 if self._energy < 0.2 else 0.0
        raw = eng_component + energy_component + consistency_component + tag_component + comp_component - burnout_penalty
        return max(0.0, min(1.0, raw))

    def _compute_rest_reward(self) -> float:
        prev_energy = self._energy_history[-2] if len(self._energy_history) >= 2 else 1.0
        energy_delta = self._energy - prev_energy
        energy_component = max(0.0, min(1.0, (energy_delta + 0.3) / 0.6)) * 0.15

        day_posts = self._posts_per_day.get(self._day, 0)
        if 1 <= day_posts <= 2:
            consistency = 1.0
        elif day_posts == 0 or day_posts == 3:
            consistency = 0.5
        else:
            consistency = 0.0
        consistency_component = consistency * 0.15

        burnout_penalty = 0.1 if self._energy < 0.2 else 0.0
        raw = energy_component + consistency_component - burnout_penalty
        return max(0.0, min(1.0, raw))

    def _advance_time(self) -> None:
        self._hour += 1
        self._hours_since_sleep += 1

        if self._hours_since_sleep > SLEEP_ENERGY_DRAIN_START:
            hours_over = self._hours_since_sleep - SLEEP_ENERGY_DRAIN_START
            drain = SLEEP_ENERGY_DRAIN_RATE * (1 + hours_over * 0.1)
            self._energy = max(0.0, self._energy - drain)

        if self._hours_since_sleep > SLEEP_OPTIMAL_AWAKE:
            hours_over = self._hours_since_sleep - SLEEP_OPTIMAL_AWAKE
            debt_rate = 0.01 * (1 + hours_over * 0.05)
            self._sleep_debt = min(1.0, self._sleep_debt + debt_rate)

        if self._hour >= 24:
            self._hour = 0
            self._day += 1
            self._posts_today = 0
            self._rotate_trends()

    def _build_observation(
        self, reward: float, error: Optional[str], done: bool = False,
        grader_score: Optional[float] = None,
        daily_total_engagement: float = 0.0, daily_posts_made: int = 0,
        daily_energy_min: float = 1.0,
        tool_results: Optional[List[ToolResult]] = None,
        engagement_signals: Optional[EngagementSignals] = None,
        coach_feedback: Optional[Dict[str, Any]] = None,
        rubric_scores: Optional[Dict[str, float]] = None,
        rubric_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ViraltestObservation:
        recent_eng = self._engagement_history[-10:] if self._engagement_history else []
        eng_rate = sum(recent_eng) / len(recent_eng) if recent_eng else 0.0

        meta: Dict[str, Any] = {"step": self._state.step_count, "task": self._task}
        if grader_score is not None:
            meta["grader_score"] = round(grader_score, 4)
        if rubric_scores:
            meta["rubric_scores"] = rubric_scores

        burnout_risk = min(1.0, self._low_energy_days / 5.0)

        return ViraltestObservation(
            current_hour=self._hour,
            day_of_week=self._day % 7,
            days_elapsed=self._day,
            creator_energy=round(self._energy, 3),
            hours_since_sleep=self._hours_since_sleep,
            sleep_debt=round(self._sleep_debt, 3),
            follower_count=self._followers,
            engagement_rate=round(eng_rate, 4),
            posts_today=self._posts_today,
            time_since_last_post=self._time_since_last_post,
            content_queue_size=self._content_queue,
            last_post_type=self._last_post_types[-1] if self._last_post_types else "none",
            burnout_risk=round(burnout_risk, 3),
            daily_total_engagement=round(daily_total_engagement, 4),
            daily_posts_made=daily_posts_made,
            daily_energy_min=round(daily_energy_min, 3),
            engagement_signals=engagement_signals,
            coach_feedback=coach_feedback,
            tool_results=tool_results or [],
            agent_notes=self._agent_notes,
            api_budget_remaining=self._api_budget,
            grader_score=round(grader_score, 4) if grader_score is not None else None,
            rubric_scores=rubric_scores or {},
            rubric_evidence=rubric_evidence or {},
            error=error,
            done=done,
            reward=round(reward, 4),
            metadata=meta,
        )

    # ----- graders (composable rubrics, PDF page 2) -----

    # Per-task rubric weights. Sum to 1.0 within each row.
    RUBRIC_WEIGHTS: Dict[str, Dict[str, float]] = {
        "monthly_engage":      {"engagement": 0.70, "burnout": 0.20, "discovery": 0.05, "differentiation": 0.05},
        "monthly_strategic":   {"engagement": 0.35, "burnout": 0.25, "discovery": 0.30, "differentiation": 0.10},
        "monthly_competitive": {"engagement": 0.25, "burnout": 0.10, "discovery": 0.20, "differentiation": 0.45},
    }

    def _run_grader(self) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, Any]]]:
        rubrics = {
            "engagement": self._rubric_engagement(),
            "burnout": self._rubric_burnout(),
            "discovery": self._rubric_discovery(),
            "differentiation": self._rubric_differentiation(),
        }
        weights = self.RUBRIC_WEIGHTS.get(self._task, self.RUBRIC_WEIGHTS["monthly_engage"])
        scores = {k: round(v[0], 4) for k, v in rubrics.items()}
        evidence = {k: v[1] for k, v in rubrics.items()}
        weighted = sum(weights.get(k, 0.0) * scores[k] for k in scores)
        # Anti-gaming: full burnout collapses the score.
        if self._energy <= 0.0:
            weighted *= 0.3
        return max(0.0, min(1.0, weighted)), scores, evidence

    def _theoretical_max_engagement(self) -> float:
        best_base = max(BASE_ENGAGEMENT.values())
        best_reach = max(REACH_MULT.values())
        best_niche = max(_NICHE_MULTIPLIERS.values()) if _NICHE_MULTIPLIERS else 1.0
        posts_per_week = 5
        weeks = 4
        avg_peak_mult = 1.35
        return best_base * best_reach * best_niche * avg_peak_mult * posts_per_week * weeks

    def _rubric_engagement(self) -> Tuple[float, Dict[str, Any]]:
        """Total Mosseri-weighted engagement vs theoretical max for the month."""
        theoretical_max = self._theoretical_max_engagement()
        score = min(1.0, self._total_engagement / theoretical_max) if theoretical_max > 0 else 0.0
        return score, {
            "total_engagement": round(self._total_engagement, 3),
            "theoretical_max": round(theoretical_max, 3),
            "posting_steps": self._posting_steps,
        }

    def _rubric_burnout(self) -> Tuple[float, Dict[str, Any]]:
        """Avg energy, min energy, and sleep-debt penalties (Van Dongen 2003)."""
        energies = self._energy_history or [self._energy]
        avg_energy = sum(energies) / len(energies)
        min_energy = min(energies)
        sleep_factor = max(0.0, 1.0 - self._sleep_debt)
        score = 0.5 * avg_energy + 0.3 * min_energy + 0.2 * sleep_factor
        if min_energy < 0.2:
            score *= 0.4
        elif min_energy < 0.3:
            score = min(score, 0.6)
        return max(0.0, min(1.0, score)), {
            "avg_energy": round(avg_energy, 3),
            "min_energy": round(min_energy, 3),
            "sleep_debt": round(self._sleep_debt, 3),
            "low_energy_days": self._low_energy_days,
        }

    def _rubric_discovery(self) -> Tuple[float, Dict[str, Any]]:
        """Tag exploration (positive-EV tags) + tool-call diversity."""
        positive_tags = sum(1 for t in self._unique_tags_used if self._tag_performance_avg(t) > 0)
        tag_discovery = min(1.0, positive_tags / 30.0)
        top_perfs = sorted([self._tag_performance_avg(t) for t in self._unique_tags_used], reverse=True)[:3]
        tag_exploitation = min(1.0, (sum(top_perfs) / len(top_perfs) / 2.0) if top_perfs else 0.0)
        tool_diversity = min(1.0, len(self._unique_tools_used) / 5.0)
        score = 0.40 * tag_discovery + 0.40 * tag_exploitation + 0.20 * tool_diversity
        return score, {
            "unique_tags_used": len(self._unique_tags_used),
            "positive_ev_tags": positive_tags,
            "top_3_tag_perf": [round(p, 3) for p in top_perfs],
            "unique_tools_used": sorted(self._unique_tools_used),
            "tool_calls_total": self._tool_calls_total,
            "api_budget_remaining": self._api_budget,
        }

    def _rubric_differentiation(self) -> Tuple[float, Dict[str, Any]]:
        """Content variety, topic uniqueness, follower growth, vs-competitor outperformance.

        Anti-gaming: monoculture (1 content type) or low tag exploration cap differentiation hard.
        """
        content_variety = min(1.0, len(self._unique_content_types) / 4.0)
        topic_uniqueness = (self._unique_topic_steps / self._posting_steps) if self._posting_steps > 0 else 0.0
        growth = (self._followers - self._initial_followers) / self._initial_followers if self._initial_followers > 0 else 0.0
        norm_growth = min(1.0, max(0.0, growth / 0.04))
        comp_avg = self._get_competitor_avg_engagement()
        my_avg = self._total_engagement / self._posting_steps if self._posting_steps > 0 else 0.0
        norm_outperformance = min(1.0, (my_avg / comp_avg) / 1.5) if comp_avg > 0 else 0.0
        score = (
            0.25 * content_variety + 0.20 * topic_uniqueness
            + 0.30 * norm_growth + 0.25 * norm_outperformance
        )
        # Anti-gaming gates: single content type or sparse tag set caps the score.
        if len(self._unique_content_types) < 2:
            score *= 0.3
        elif len(self._unique_content_types) < 3:
            score = min(score, 0.5)
        if len(self._unique_tags_used) < 5:
            score *= 0.6
        return max(0.0, min(1.0, score)), {
            "unique_content_types": sorted(self._unique_content_types),
            "topic_uniqueness": round(topic_uniqueness, 3),
            "follower_growth": round(growth, 4),
            "competitor_avg_engagement": round(comp_avg, 3),
            "my_avg_engagement": round(my_avg, 3),
            "tags_used": len(self._unique_tags_used),
        }


def _partner_max_overlap(partner_id: str) -> float:
    """Max audience overlap with any other archetype (excludes self-pair)."""
    ids = _OVERLAP_DATA["archetype_ids"]
    if partner_id not in ids:
        return 0.15
    idx = ids.index(partner_id)
    row = _OVERLAP_DATA["matrix"][idx]
    others = [v for j, v in enumerate(row) if j != idx]
    return float(max(others)) if others else 0.15


def _topic_overlap(topic_a: str, topic_b: str) -> bool:
    words_a = set(topic_a.split())
    words_b = set(topic_b.split())
    if not words_a or not words_b:
        return False
    common = words_a & words_b
    return len(common) / min(len(words_a), len(words_b)) >= 0.5


def _avg_signal_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    result = {}
    for k in keys:
        vals = [d.get(k, 0.0) for d in dicts]
        result[k] = round(sum(vals) / len(vals), 4)
    return result
