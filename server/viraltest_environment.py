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
        HeadlineMetrics,
        JudgeReport,
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
        HeadlineMetrics,
        JudgeReport,
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
# Collab reward shaping (Later 2023 reach study, HypeAuditor 2024 niche affinity, Rival IQ 2025 overlap patterns,
# Cen et al. 2024 disengagement model for diminishing returns instead of a hard cap).
COLLAB_REACH_K = 0.60      # cross-audience exposure: capped reach uplift when overlap is 0
COLLAB_AFFINITY_K = 0.30   # same-audience affinity: per-impression engagement uplift when overlap is 1
COLLAB_GROWTH_K = 1.50     # cross-pollination follower spillover, scales (1 - overlap)
COLLAB_PARTNER_REPEAT_PENALTY = 0.7  # discount on multipliers when partner reused this brand
COLLAB_FATIGUE_K = 0.3     # per-collab diminishing-returns factor: 1/(1+K*prior_collabs_this_episode)

API_BUDGET_INITIAL = 10**9  # effectively unlimited; rate-limit removed

# Heuristic baselines for headline metric `vs_baseline_pct`.
# Data-driven: loaded from `plots/training_summary.json["smart_heuristic"]` recorded by
# `training/run_training_evidence.py`. Falls back to conservative calibration constants
# if the file is missing (audit trail: see RESEARCH.md for the rule-based policy spec).
def _load_heuristic_baselines() -> Dict[str, float]:
    summary = Path(__file__).parent.parent / "plots" / "training_summary.json"
    try:
        data = json.loads(summary.read_text())
        empirical = data.get("smart_heuristic") or {}
        return {k: float(v) for k, v in empirical.items() if k in VALID_TASKS}
    except Exception:
        return {}

HEURISTIC_BASELINE_SCORES: Dict[str, float] = _load_heuristic_baselines() or {
    "monthly_engage": 0.43,
    "monthly_strategic": 0.77,
    "monthly_competitive": 0.81,
}

# Cross-episode store for distribution-shift retention. Keyed by episode_chain_id, stores
# {"baseline": score, "shifted": score} so the second run can compute retention_under_shift.
_SHIFT_HISTORY: Dict[str, Dict[str, float]] = {}

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
        "description": "Propose a collab post with a competitor at a specific hour. The post you schedule at that hour will be co-authored with the partner.",
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
        self._active_collab: Optional[CollabProposal] = None
        self._low_energy_days = 0
        self._total_posts_this_week = 0
        self._week_start_day = 0
        self._daily_signals = EngagementSignals()
        self._total_tool_calls = 0
        self._total_action_chars = 0
        self._shift_label: Optional[str] = None
        self._chain_id: Optional[str] = None

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

    # ----- collab multipliers (overlap-driven) -----

    def _user_partner_overlap(self, partner_id: str) -> Optional[float]:
        ids = _OVERLAP_DATA.get("archetype_ids", [])
        if "user_creator" not in ids or partner_id not in ids:
            return None
        u = ids.index("user_creator")
        p = ids.index(partner_id)
        return _OVERLAP_DATA["matrix"][u][p]

    def _collab_multipliers(self, partner_id: str) -> Tuple[float, float]:
        """Returns (engagement_multiplier, follower_growth_multiplier)."""
        o = self._user_partner_overlap(partner_id)
        if o is None:
            return 1.0, 1.0
        reach = 1.0 + (1.0 - o) * COLLAB_REACH_K
        affinity = 1.0 + o * COLLAB_AFFINITY_K
        growth = 1.0 + (1.0 - o) * COLLAB_GROWTH_K
        eng_boost = reach * affinity
        if partner_id in self._collab_history[:-1]:
            eng_boost *= COLLAB_PARTNER_REPEAT_PENALTY
            growth *= COLLAB_PARTNER_REPEAT_PENALTY
        prior = max(0, self._collabs_this_month - 1)
        fatigue = 1.0 / (1.0 + COLLAB_FATIGUE_K * prior)
        return eng_boost * fatigue, growth * fatigue

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
                try:
                    sa = ScheduledAction(**sa_dict) if isinstance(sa_dict, dict) else sa_dict
                except Exception:
                    continue
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
            pool = []
            for comp in self._competitors:
                overlap = self._user_partner_overlap(comp.id)
                pool.append({
                    "id": comp.id, "name": comp.name, "niche": comp.niche,
                    "audience_overlap": round(overlap, 2) if overlap is not None else None,
                })
            return ToolResult(name=tool.name, data=pool, budget_remaining=self._api_budget)

        elif tool.name == "propose_collab":
            partner_id = tool.arguments.get("partner_id", "")
            if partner_id not in [c.id for c in self._competitors]:
                return ToolResult(name=tool.name, success=False, error=f"unknown partner: {partner_id}", budget_remaining=self._api_budget)
            return ToolResult(name=tool.name, data={"status": "proposal_accepted", "partner_id": partner_id}, budget_remaining=self._api_budget)

        return ToolResult(name=tool.name, success=False, error=f"unknown tool: {tool.name}", budget_remaining=self._api_budget)

    # ----- counterfactual coach -----

    def _compute_coach_feedback(self, agent_engagement: float) -> Dict[str, Any]:
        # World-modeling discipline: emit a SCALAR delta only (no optimal_hours leak).
        # Agents must use `query_trends` / `predict_engagement` to discover *which* hours
        # are optimal — coach only signals "you're above/below the heatmap optimum today".
        dow = self._day % 7
        row = _HEATMAP_GRID.get(dow, [1.0] * 24)
        best_hours = sorted(range(24), key=lambda h: row[h] if h < len(row) else 0, reverse=True)[:2]
        best_base = max(BASE_ENGAGEMENT.values())
        best_reach = max(REACH_MULT.values())
        optimal_eng = sum(row[h] * best_base * best_reach for h in best_hours)
        delta = agent_engagement - optimal_eng
        return {
            "delta": round(delta, 4),
            "suggestion": (
                "Above heatmap optimum today."
                if delta >= 0
                else "Below heatmap optimum — try `query_trends` / `predict_engagement` to find peak hours."
            ),
        }

    # ----- regulator / judge mode (deterministic, explainable) -----

    def _compute_judge_report(
        self,
        action: ViraltestAction,
        daily_engagement: float,
        daily_posts: int,
        energy_min: float,
        errors: List[str],
    ) -> JudgeReport:
        violations: List[str] = []

        pc = 1.0
        if daily_posts > 5:
            violations.append(f"posts_today={daily_posts} exceeds tier-4 fatigue cliff (Buffer 2.1M)")
            pc -= 0.30
        elif daily_posts > 2:
            violations.append(f"posts_today={daily_posts} enters fatigue tier (>2/day)")
            pc -= 0.10
        if self._total_posts_this_week > WEEKLY_FATIGUE_THRESHOLD:
            violations.append(f"weekly posts={self._total_posts_this_week} > {WEEKLY_FATIGUE_THRESHOLD} (Buffer 2.1M cap)")
            pc -= 0.20
        if self._collabs_this_month >= 4:
            violations.append(f"collab cadence={self._collabs_this_month} net-negative beyond 3 (Cen 2024)")
            pc -= 0.20
        if errors:
            violations.append(f"plan_errors={len(errors)}")
            pc -= 0.05 * len(errors)
        if self._hours_since_sleep > 22:
            violations.append(f"sleep_debt: {self._hours_since_sleep}h awake (Van Dongen 2003)")
            pc -= 0.10

        burnout_pressure = (1.0 - energy_min) * 0.4 + self._sleep_debt * 0.3 + (self._low_energy_days / 5.0) * 0.3
        sustainability_risk = max(0.0, min(1.0, burnout_pressure))

        intents_used = {sa.intent for sa in action.scheduled_actions if sa.intent}
        formats_used = {sa.content_type for sa in action.scheduled_actions if sa.action_type == "post" and sa.content_type}
        eng_per_post = daily_engagement / max(1, daily_posts)
        sq = (
            0.40 * min(1.0, eng_per_post / 1.2)
            + 0.30 * min(1.0, len(intents_used) / 2.0)
            + 0.30 * min(1.0, len(formats_used) / 2.0)
        )

        explanation = (
            f"compliance={max(0.0, pc):.2f} risk={sustainability_risk:.2f} strategy={sq:.2f} | "
            + (("violations: " + "; ".join(violations)) if violations else "no policy violations")
        )

        return JudgeReport(
            policy_compliance=max(0.0, min(1.0, pc)),
            sustainability_risk=sustainability_risk,
            strategic_quality=max(0.0, min(1.0, sq)),
            explanation=explanation,
            violations=violations,
        )

    def _compute_headline_metrics(self, grader_score: float) -> HeadlineMetrics:
        baseline = HEURISTIC_BASELINE_SCORES.get(self._task, 0.30)
        vs_pct = (grader_score - baseline) / baseline if baseline > 0 else 0.0
        spt = grader_score / max(1, self._total_tool_calls)
        sp1k = grader_score / max(1.0, self._total_action_chars / 1000.0)

        retention: Optional[float] = None
        if self._chain_id:
            entry = _SHIFT_HISTORY.setdefault(self._chain_id, {})
            label = self._shift_label or "baseline"
            entry[label] = grader_score
            base = entry.get("baseline")
            shifted = entry.get("shifted")
            if base is not None and shifted is not None and base > 0:
                retention = shifted / base

        return HeadlineMetrics(
            vs_baseline_pct=round(vs_pct, 4),
            score_per_tool_call=round(spt, 4),
            score_per_1k_chars=round(sp1k, 4),
            retention_under_shift=round(retention, 4) if retention is not None else None,
            heuristic_baseline_score=round(baseline, 4),
            agent_score=round(grader_score, 4),
            total_tool_calls=self._total_tool_calls,
            total_action_chars=self._total_action_chars,
        )

    # ----- core API -----

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ViraltestObservation:
        self._task = kwargs.get("task", "monthly_engage")
        if self._task not in VALID_TASKS:
            self._task = "monthly_engage"

        self._rng = random.Random(seed if seed is not None else 42)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._init_state()

        self._shift_label = kwargs.get("shift_label")
        self._chain_id = kwargs.get("episode_chain_id")

        if self._chain_id and self._chain_id in _BRAND_STORE:
            brand = _BRAND_STORE[self._chain_id]
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

        try:
            self._total_action_chars += len(action.model_dump_json())
        except Exception:
            pass

        tool_results: List[ToolResult] = []
        for tc in action.tool_calls:
            result = self._dispatch_tool(tc)
            tool_results.append(result)
            if result.success:
                self._total_tool_calls += 1

        # Process collab proposal (no hard cap; diminishing returns enforced via _collab_multipliers)
        self._active_collab = None
        if action.collab:
            self._collabs_this_month += 1
            self._collab_history.append(action.collab.partner_id)
            self._active_collab = action.collab

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
        judge = self._compute_judge_report(action, daily_engagement, daily_posts, energy_min, errors)

        if done:
            self._episode_done = True
            grader_score = self._run_grader()
            headline = self._compute_headline_metrics(grader_score)

            if self._chain_id:
                top_tags = sorted(self._unique_tags_used, key=lambda t: self._tag_performance_avg(t), reverse=True)[:3]
                _BRAND_STORE[self._chain_id] = {
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
                coach_feedback=coach, judge_report=judge, headline_metrics=headline,
            )
            return self._final_observation

        return self._build_observation(
            reward=round(avg_reward, 4), error=error_str,
            daily_total_engagement=daily_engagement,
            daily_posts_made=daily_posts, daily_energy_min=energy_min,
            tool_results=tool_results, engagement_signals=daily_signals,
            coach_feedback=coach, judge_report=judge,
        )

    def _process_hour_action(self, sa: ScheduledAction) -> Tuple[float, float, Optional[EngagementSignals]]:
        engagement = 0.0
        signals = None

        collab_growth_mult = 1.0

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

                engagement = (
                    base * reach * hour_mult * quality * tag_boost
                    * trending_bonus * comp_diff * fatigue * algo_mult
                    * niche_mult * saturation_factor
                )

                if self._active_collab is not None and self._active_collab.hour == sa.hour:
                    eng_m, growth_m = self._collab_multipliers(self._active_collab.partner_id)
                    engagement *= eng_m
                    collab_growth_mult = growth_m

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
                self._followers += int(engagement * 100 * collab_growth_mult)

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
        judge_report: Optional[JudgeReport] = None,
        headline_metrics: Optional[HeadlineMetrics] = None,
    ) -> ViraltestObservation:
        recent_eng = self._engagement_history[-10:] if self._engagement_history else []
        eng_rate = sum(recent_eng) / len(recent_eng) if recent_eng else 0.0

        meta: Dict[str, Any] = {"step": self._state.step_count, "task": self._task}
        if grader_score is not None:
            meta["grader_score"] = round(grader_score, 4)

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
            judge_report=judge_report,
            headline_metrics=headline_metrics,
            tool_results=tool_results or [],
            agent_notes=self._agent_notes,
            api_budget_remaining=self._api_budget,
            grader_score=round(grader_score, 4) if grader_score is not None else None,
            error=error,
            done=done,
            reward=round(reward, 4),
            metadata=meta,
        )

    # ----- graders (monthly) -----

    def _run_grader(self) -> float:
        if self._task == "monthly_engage":
            return self._grade_monthly_engage()
        elif self._task == "monthly_strategic":
            return self._grade_monthly_strategic()
        elif self._task == "monthly_competitive":
            return self._grade_monthly_competitive()
        return 0.0

    def _theoretical_max_engagement(self) -> float:
        # Buffer 2.1M (RESEARCH.md): 3–5 posts/week doubles follower growth vs 1–2,
        # diminishing returns above 5/week, 20–35% engagement drop per post above 7/week.
        # Cap at 5 posts/week × 4 weeks = 20 posts/month (sweet-spot, no fatigue penalty).
        best_base = max(BASE_ENGAGEMENT.values())
        best_reach = max(REACH_MULT.values())
        best_niche = max(_NICHE_MULTIPLIERS.values()) if _NICHE_MULTIPLIERS else 1.0

        posts_per_week = 5
        weeks_in_horizon = TASK_HORIZON / 7.0
        total_posts = int(round(posts_per_week * weeks_in_horizon))

        avg_heatmap_peak = 1.0
        if _HEATMAP_GRID:
            day_peaks = [
                max(row) if row else 1.0
                for row in _HEATMAP_GRID.values()
            ]
            avg_heatmap_peak = sum(day_peaks) / len(day_peaks) if day_peaks else 1.0

        # Trending + tag uplifts: tier-1 industry data shows ~1.2-1.3x for trending topics
        # and ~1.05-1.15x for high-performance tags. Mid-range used to avoid headroom inflation.
        trending_bonus = 1.25
        tag_boost = 1.1

        per_post = (
            best_base * best_reach * best_niche
            * avg_heatmap_peak * trending_bonus * tag_boost
        )
        return per_post * total_posts

    def _grade_monthly_engage(self) -> float:
        theoretical_max = self._theoretical_max_engagement()
        if theoretical_max <= 0:
            return 0.0
        raw = min(1.0, self._total_engagement / theoretical_max)
        if self._energy <= 0.0:
            raw *= 0.3
        return raw

    def _grade_monthly_strategic(self) -> float:
        if self._energy <= 0.0:
            return max(0.0, min(0.15, self._total_engagement * 0.01))

        theoretical_max = self._theoretical_max_engagement()
        norm_eng = min(1.0, self._total_engagement / theoretical_max) if theoretical_max > 0 else 0.0

        positive_tags = sum(1 for t in self._unique_tags_used if self._tag_performance_avg(t) > 0)
        tag_discovery = min(1.0, positive_tags / 30.0)
        top_perfs = sorted([self._tag_performance_avg(t) for t in self._unique_tags_used], reverse=True)[:3]
        tag_exploitation = (sum(top_perfs) / len(top_perfs)) if top_perfs else 0.0
        tag_exploitation = min(1.0, tag_exploitation / 2.0)
        tag_score = 0.4 * tag_discovery + 0.6 * tag_exploitation

        avg_energy = sum(self._energy_history) / len(self._energy_history) if self._energy_history else 0.0
        consistency = len(self._days_with_good_posts) / 30.0

        raw = 0.35 * norm_eng + 0.25 * tag_score + 0.25 * avg_energy + 0.15 * consistency

        min_energy = min(self._energy_history) if self._energy_history else 0.0
        if min_energy < 0.2:
            raw *= 0.4
        elif min_energy < 0.3:
            raw = min(raw, 0.45)
        if len(self._unique_tags_used) < 5:
            raw *= 0.7

        return max(0.0, min(1.0, raw))

    def _grade_monthly_competitive(self) -> float:
        if self._energy <= 0.0:
            return 0.0

        theoretical_max = self._theoretical_max_engagement()
        norm_eng = min(1.0, self._total_engagement / theoretical_max) if theoretical_max > 0 else 0.0

        positive_tags = sum(1 for t in self._unique_tags_used if self._tag_performance_avg(t) > 0)
        tag_discovery = min(1.0, positive_tags / 30.0)
        top_perfs = sorted([self._tag_performance_avg(t) for t in self._unique_tags_used], reverse=True)[:3]
        tag_exploitation = (sum(top_perfs) / len(top_perfs)) if top_perfs else 0.0
        tag_exploitation = min(1.0, tag_exploitation / 2.0)
        tag_score = 0.4 * tag_discovery + 0.6 * tag_exploitation

        growth = (self._followers - self._initial_followers) / self._initial_followers if self._initial_followers > 0 else 0.0
        target_growth = 0.04
        norm_growth = min(1.0, max(0.0, growth / target_growth))

        comp_avg = self._get_competitor_avg_engagement()
        my_avg = self._total_engagement / self._posting_steps if self._posting_steps > 0 else 0.0
        outperformance = my_avg / comp_avg if comp_avg > 0 else 1.0
        norm_outperformance = min(1.0, outperformance / 1.5)

        differentiation = self._unique_topic_steps / self._posting_steps if self._posting_steps > 0 else 0.0

        min_energy = min(self._energy_history) if self._energy_history else 0.0
        energy_floor = min(1.0, max(0.0, min_energy))

        raw = (
            0.25 * norm_eng + 0.20 * tag_score + 0.20 * norm_growth
            + 0.15 * norm_outperformance + 0.10 * differentiation + 0.10 * energy_floor
        )

        if len(self._unique_content_types) < 3:
            raw *= 0.5
        if len(self._unique_tags_used) < 8:
            raw *= 0.7

        return max(0.0, min(1.0, raw))


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
