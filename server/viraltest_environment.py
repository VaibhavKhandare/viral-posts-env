"""
Viraltest Environment v2 — Theme #3.1 World-Modeling Simulation.

Multi-day creator optimization with:
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
        DailyInteractions,
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
        DailyInteractions,
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

# Mocked niche + follower-count lookups for the collab system. Live in the overlap matrix file
# so the same source-of-truth carries (a) Jaccard overlap, (b) niche label, (c) follower size.
_NICHE_BY_ARCHETYPE: Dict[str, str] = dict(_OVERLAP_DATA.get("niche_by_archetype", {}))
_FOLLOWERS_BY_ARCHETYPE: Dict[str, int] = {
    k: int(v) for k, v in _OVERLAP_DATA.get("mock_followers_by_archetype", {}).items()
}

# ---------------------------------------------------------------------------
# Constants (research-backed, Tier 1-3 sources)
# ---------------------------------------------------------------------------

# Episode length in daily env steps. Graders and UI should stay consistent with this value.
TASK_HORIZON = 15

# Distinct positive tags for full tag_discovery score in strategic/competitive graders.
# Caps at 30 (original month-scale bar); scales down only for very short horizons.
TAG_DISCOVERY_POSITIVE_TARGET = float(max(6, min(30, TASK_HORIZON * 2)))

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
TREND_MATCH_STOPWORDS = {"tips", "guide", "review", "routine", "ideas", "hacks", "tutorial", "the", "a", "an", "and", "of", "for", "to"}
# Collab reward shaping (Later 2023 reach study, HypeAuditor 2024 niche affinity, Rival IQ 2025 overlap patterns,
# Cen et al. 2024 disengagement model for diminishing returns instead of a hard cap).
# Per-partner exhaustion: each collab with the SAME partner re-exposes the user to the same set of followers,
# so spillover and reward multipliers should decay sharply. First 1-2 collabs deliver most of the gain.
# Floor at 0.05 keeps a tiny residual signal so the curve is monotonic but effectively saturates.
COLLAB_PARTNER_REPEAT_DECAY = {0: 1.0, 1: 0.70, 2: 0.35, 3: 0.15}
COLLAB_PARTNER_REPEAT_FLOOR = 0.05
COLLAB_FATIGUE_K = 0.3     # per-collab diminishing-returns factor: 1/(1+K*prior_collabs_this_episode)

# Niche-aware tiered shaping (overlap = Jaccard intersection fraction).
# Hard rule: any diff-niche multiplier must be < the minimum same-niche-low multiplier
# so the env never recommends a diff-niche collab over an equal-overlap same-niche one.
COLLAB_LOW_OVERLAP_THRESHOLD = 0.20      # < this counts as "low intersection"
COLLAB_HIGH_OVERLAP_THRESHOLD = 0.40     # >= this counts as "high intersection"
COLLAB_GUARDRAIL_OVERLAP_MIN = 0.10      # below this -> recommended=False (intersection-too-low guardrail)
COLLAB_GUARDRAIL_FOLLOWER_GAP_MAX = 0.25 # |partner - user| / max > this -> follower-size mismatch
COLLAB_FORCED_PENALTY_ENG = 0.7          # eng_mult applied if agent ignores guardrail
COLLAB_FORCED_PENALTY_GROWTH = 0.6       # growth_mult applied if agent ignores guardrail

# Same niche, LOW overlap -> HIGH reward (best case). Smoothly interpolated by overlap (low->high uplift as overlap->0).
COLLAB_SAME_LOW_ENG = (1.50, 1.80)
COLLAB_SAME_LOW_GROWTH = (1.60, 2.00)
# Same niche, HIGH overlap -> LOW reward (no point, audience already shared).
COLLAB_SAME_HIGH_ENG = 0.85
COLLAB_SAME_HIGH_GROWTH = 0.90
# Diff niche, LOW overlap -> MED reward (cross-pollination, capped < SAME_LOW min).
COLLAB_DIFF_LOW_ENG = (1.20, 1.40)
COLLAB_DIFF_LOW_GROWTH = (1.30, 1.55)
# Diff niche, HIGH overlap -> LOW reward (mismatch).
COLLAB_DIFF_HIGH_ENG = 0.75
COLLAB_DIFF_HIGH_GROWTH = 0.80

# Collab is the canonical "wider reach in one shot" lever. On top of the engagement
# multiplier (which only affects the collab-day post), apply two extra mechanisms:
#  1) One-shot follower spillover: partner_followers x (1 - overlap) x growth_mult x K_SPILLOVER.
#     Models the partner's audience getting exposed to the user — net new followers, not just engagement.
#  2) Sustained reach buff: 2-3 days post-collab, all posts get a small algorithm boost
#     because the collab signal lifts the user's overall recommendability.
COLLAB_SPILLOVER_K = 0.05                # follower spike fraction (0.05 = 5% of partner audience reach if overlap=0)
COLLAB_REACH_CARRYOVER_DAYS = 2          # days the post-collab reach buff persists
COLLAB_REACH_CARRYOVER_MULT = 1.20       # +20% engagement on each post during carryover window
COLLAB_BLOCKED_SPILLOVER_K = 0.01        # forced/guardrail-blocked collabs get only ~20% of normal spillover

# Interaction (likes/comments/replies) tunables
INTERACT_ENERGY_LIKE = 0.005
INTERACT_ENERGY_COMMENT = 0.012
INTERACT_ENERGY_REPLY = 0.018
INTERACT_HEALTHY_LIKES = (5, 20)
INTERACT_HEALTHY_COMMENTS = (3, 10)
INTERACT_LIKE_REACH_BUFF = 0.02          # was 0.04 — interactions are a sustaining lever, not a growth lever
INTERACT_COMMENT_REACH_BUFF = 0.04       # was 0.08
INTERACT_REPLY_REWARD_PER = 0.005        # was 0.01
INTERACT_REPLY_REWARD_CAP = 0.08
INTERACT_DAILY_REWARD_CAP = 0.10
INTERACT_SPAM_LIKES = 30
INTERACT_SPAM_COMMENTS = 20
INTERACT_SPAM_REACH_PENALTY = 0.85
INTERACT_SPAM_SHADOWBAN_BUMP = 0.20
INTERACT_IGNORE_THRESHOLD_K = 0.05
INTERACT_IGNORE_LOYALTY_DECAY = 0.97
INTERACT_OFFNICHE_THRESHOLD = 0.60
INTERACT_OFFNICHE_REACH_PENALTY = 0.90
INTERACT_LOWQ_THRESHOLD = 0.30
INTERACT_LOWQ_WEIGHT = 0.4
INTERACT_VERY_LOWQ_THRESHOLD = 0.10
INTERACT_VERY_LOWQ_PENALTY = -0.03

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
        "description": "List available competitor archetypes for potential collaboration with audience overlap %, niche match, mocked follower counts, intersection size, and a recommendation flag (recommended=False when guardrails block: zero followers, intersection<10%, or follower-size gap>25%).",
        "parameters": {},
    },
    "propose_collab": {
        "description": "Propose a collab post with a competitor at a specific hour. The post you schedule at that hour will be co-authored (or auto-injected if absent — collab always pays a post's energy + counts as a post). Reward shaping: same-niche + low overlap = HIGH; same-niche + high overlap = LOW; diff-niche always capped below same-niche-low. Per-partner exhaustion: 1st collab full reward, 2nd 0.70x, 3rd 0.35x, 4th 0.15x, 5th+ 0.05x — partner audiences are tapped once. Guardrail violations apply a 0.7x engagement / 0.6x growth penalty AND surface in the JudgeReport.",
        "parameters": {
            "partner_id": {"type": "string"},
            "content_type": {"type": "string", "enum": ["reel", "story", "carousel", "text_post"]},
            "hour": {"type": "integer", "minimum": 0, "maximum": 23},
        },
    },
    "query_interaction_norms": {
        "description": "Discover healthy daily ranges for likes/comments/replies and the current shadowban_risk. Use before submitting ViraltestAction.interactions.",
        "parameters": {},
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
        self._collab_violations: List[str] = []  # collab guardrail breaches this step
        self._collab_carryover_days_remaining: int = 0
        self._last_collab_spillover: int = 0     # diagnostic: followers gained from the most recent collab
        self._user_niche: str = _NICHE_BY_ARCHETYPE.get("user_creator", "generic")

        # Interaction state
        self._pending_reach_mult: float = 1.0   # applied to next day's posts (one-shot)
        self._shadowban_risk: float = 0.0
        self._engagement_rate_loyalty_mult: float = 1.0  # compounding loyalty drop from ignoring audience
        self._interaction_violations: List[str] = []
        self._last_interaction_summary: Optional[Dict[str, Any]] = None
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

    # ----- collab evaluation (niche-aware, overlap-tiered) -----

    def _user_partner_overlap(self, partner_id: str) -> Optional[float]:
        ids = _OVERLAP_DATA.get("archetype_ids", [])
        if "user_creator" not in ids or partner_id not in ids:
            return None
        u = ids.index("user_creator")
        p = ids.index(partner_id)
        return _OVERLAP_DATA["matrix"][u][p]

    def _partner_niche(self, partner_id: str) -> str:
        return _NICHE_BY_ARCHETYPE.get(partner_id, "generic")

    def _partner_followers(self, partner_id: str) -> int:
        return _FOLLOWERS_BY_ARCHETYPE.get(partner_id, 0)

    def _partner_prior_count(self, partner_id: str) -> int:
        """How many times we've already collabed with this partner THIS episode (excludes current entry)."""
        if not self._collab_history:
            return 0
        return self._collab_history[:-1].count(partner_id)

    def _partner_repeat_decay(self, partner_id: str) -> float:
        """Multiplier in [floor, 1.0] that scales spillover + tier multipliers based on prior count.

        First collab = 1.0x (no decay), 2nd = 0.70, 3rd = 0.35, 4th = 0.15, 5th+ = 0.05.
        """
        n = self._partner_prior_count(partner_id)
        return COLLAB_PARTNER_REPEAT_DECAY.get(n, COLLAB_PARTNER_REPEAT_FLOOR)

    def _collab_default_topic(self, partner_niche: str) -> str:
        """Pick a topic for an auto-injected collab post.

        Prefer the user's niche (so the post lands in user-niche distribution); fall back to
        the partner's niche; final fallback is the first known topic.
        """
        for niche in (self._user_niche, partner_niche):
            topics = TOPIC_CATEGORIES.get(niche)
            if topics:
                return topics[0]
        # Fallback: first topic of any known niche.
        for topics in TOPIC_CATEGORIES.values():
            if topics:
                return topics[0]
        return "collaboration"

    def _collab_default_tags(self, partner_niche: str) -> List[str]:
        # Pull a couple of trending tags so the auto-post benefits from the existing tag boost.
        return list(self._trending_tags[:2]) if self._trending_tags else []

    @staticmethod
    def _interp(span: Tuple[float, float], t: float) -> float:
        """Linear interp from span[0] (t=0) to span[1] (t=1)."""
        t = max(0.0, min(1.0, t))
        return span[0] + (span[1] - span[0]) * t

    def _collab_tier_multipliers(self, same_niche: bool, overlap: float) -> Tuple[float, float]:
        """Pure 2x2 tier shaping (no fatigue/repeat/guardrail effects yet)."""
        # Smooth interp factor: how "low" is this overlap on the [0, LOW_THRESHOLD] scale.
        low_t = 1.0 - min(1.0, overlap / COLLAB_LOW_OVERLAP_THRESHOLD)  # 1 at overlap=0, 0 at threshold
        if same_niche:
            if overlap < COLLAB_LOW_OVERLAP_THRESHOLD:
                eng = self._interp(COLLAB_SAME_LOW_ENG, low_t)
                growth = self._interp(COLLAB_SAME_LOW_GROWTH, low_t)
            elif overlap >= COLLAB_HIGH_OVERLAP_THRESHOLD:
                eng = COLLAB_SAME_HIGH_ENG
                growth = COLLAB_SAME_HIGH_GROWTH
            else:
                # Mid-band linear interpolation between LOW endpoint (overlap=LOW_TH) and HIGH endpoint (overlap=HIGH_TH).
                mid_t = (overlap - COLLAB_LOW_OVERLAP_THRESHOLD) / (COLLAB_HIGH_OVERLAP_THRESHOLD - COLLAB_LOW_OVERLAP_THRESHOLD)
                eng = self._interp((COLLAB_SAME_LOW_ENG[0], COLLAB_SAME_HIGH_ENG), mid_t)
                growth = self._interp((COLLAB_SAME_LOW_GROWTH[0], COLLAB_SAME_HIGH_GROWTH), mid_t)
        else:
            if overlap < COLLAB_LOW_OVERLAP_THRESHOLD:
                eng = self._interp(COLLAB_DIFF_LOW_ENG, low_t)
                growth = self._interp(COLLAB_DIFF_LOW_GROWTH, low_t)
            elif overlap >= COLLAB_HIGH_OVERLAP_THRESHOLD:
                eng = COLLAB_DIFF_HIGH_ENG
                growth = COLLAB_DIFF_HIGH_GROWTH
            else:
                mid_t = (overlap - COLLAB_LOW_OVERLAP_THRESHOLD) / (COLLAB_HIGH_OVERLAP_THRESHOLD - COLLAB_LOW_OVERLAP_THRESHOLD)
                eng = self._interp((COLLAB_DIFF_LOW_ENG[0], COLLAB_DIFF_HIGH_ENG), mid_t)
                growth = self._interp((COLLAB_DIFF_LOW_GROWTH[0], COLLAB_DIFF_HIGH_GROWTH), mid_t)
            # Hard rule: diff-niche must always be < same-niche-low minimum (cap just below).
            eng = min(eng, COLLAB_SAME_LOW_ENG[0] - 0.01)
            growth = min(growth, COLLAB_SAME_LOW_GROWTH[0] - 0.01)
        return eng, growth

    def _collab_evaluation(self, partner_id: str) -> Dict[str, Any]:
        """Single source of truth: tier reward + guardrails + final multipliers (after fatigue/repeat).

        Returns a dict consumable by both query_creator_pool (for recommendation surface)
        and _process_hour_action (for applied multipliers).
        """
        overlap = self._user_partner_overlap(partner_id)
        if overlap is None:
            return {
                "partner_id": partner_id,
                "overlap": None,
                "same_niche": False,
                "partner_followers": 0,
                "user_followers": self._followers,
                "follower_gap_pct": 1.0,
                "intersection_size": 0,
                "recommended": False,
                "reason": "unknown_partner",
                "tier_eng_mult": 1.0,
                "tier_growth_mult": 1.0,
                "eng_mult": 1.0,
                "growth_mult": 1.0,
            }

        partner_niche = self._partner_niche(partner_id)
        same_niche = partner_niche == self._user_niche
        partner_followers = self._partner_followers(partner_id)
        user_followers = max(0, int(self._followers))
        denom = max(1, max(partner_followers, user_followers))
        gap_pct = abs(partner_followers - user_followers) / denom if denom else 1.0

        # Mock intersection size via Jaccard inversion: union ≈ (|A|+|B|)/(1+overlap), intersection = overlap*union.
        union_approx = (partner_followers + user_followers) / (1.0 + overlap) if overlap >= 0 else 0.0
        intersection_size = int(round(overlap * union_approx))

        # Guardrails (in priority order)
        recommended = True
        reason: Optional[str] = None
        if partner_followers <= 0:
            recommended = False
            reason = "partner_zero_followers"
        elif overlap < COLLAB_GUARDRAIL_OVERLAP_MIN:
            recommended = False
            reason = "intersection_below_10pct"
        elif gap_pct > COLLAB_GUARDRAIL_FOLLOWER_GAP_MAX:
            recommended = False
            reason = "follower_size_mismatch"

        tier_eng, tier_growth = self._collab_tier_multipliers(same_niche, overlap)

        eng_mult = tier_eng
        growth_mult = tier_growth

        # Per-partner exhaustion: spillover + tier mults degrade per repeat with this same partner.
        # 1st = 1.0, 2nd = 0.70, 3rd = 0.35, 4th = 0.15, 5th+ = 0.05.
        repeat_decay = self._partner_repeat_decay(partner_id)
        eng_mult *= repeat_decay
        growth_mult *= repeat_decay

        # Diminishing returns across the episode (Cen 2024) — applies regardless of partner.
        prior = max(0, self._collabs_this_month - 1)
        fatigue = 1.0 / (1.0 + COLLAB_FATIGUE_K * prior)
        eng_mult *= fatigue
        growth_mult *= fatigue

        return {
            "partner_id": partner_id,
            "overlap": round(overlap, 3),
            "same_niche": same_niche,
            "partner_niche": partner_niche,
            "user_niche": self._user_niche,
            "partner_followers": partner_followers,
            "user_followers": user_followers,
            "follower_gap_pct": round(gap_pct, 3),
            "intersection_size": intersection_size,
            "recommended": recommended,
            "reason": reason,
            "tier_eng_mult": round(tier_eng, 3),
            "tier_growth_mult": round(tier_growth, 3),
            "eng_mult": round(eng_mult, 3),
            "growth_mult": round(growth_mult, 3),
            "prior_count_with_partner": self._partner_prior_count(partner_id),
            "repeat_decay": round(repeat_decay, 3),
        }

    def _collab_multipliers(self, partner_id: str) -> Tuple[float, float]:
        """Returns (engagement_multiplier, follower_growth_multiplier).

        Applies guardrail penalties when the agent forces a non-recommended collab.
        Side effect: appends to self._collab_violations for the JudgeReport.
        """
        ev = self._collab_evaluation(partner_id)
        eng = ev["eng_mult"]
        growth = ev["growth_mult"]
        if not ev["recommended"]:
            eng *= COLLAB_FORCED_PENALTY_ENG
            growth *= COLLAB_FORCED_PENALTY_GROWTH
            self._collab_violations.append(
                f"collab_guardrail:{ev.get('reason', 'blocked')}@{partner_id}"
            )
        return eng, growth

    # ----- interactions (likes/comments/replies) -----

    def _process_interactions(
        self, interactions: Optional[DailyInteractions]
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply daily interaction effects: energy cost, reach buffs (next post), and 5 penalty paths.

        Returns (reward_delta, summary_dict). The reward_delta is added to today's averaged reward;
        reach effects propagate via self._pending_reach_mult (consumed at next _process_hour_action).
        Loyalty effects propagate via self._engagement_rate_loyalty_mult (compounding).
        """
        # Reset reach mult for the day (default neutral); we accumulate per-day, then it's consumed
        # by today's posts and any leftover carries over by simply staying at 1.0 next step.
        self._pending_reach_mult = 1.0
        self._interaction_violations = []

        summary: Dict[str, Any] = {
            "likes_on_others": 0,
            "comments_on_others": 0,
            "replies_to_audience": 0,
            "energy_cost": 0.0,
            "reach_modifier": 1.0,
            "shadowban_risk": round(self._shadowban_risk, 3),
            "loyalty_mult": round(self._engagement_rate_loyalty_mult, 3),
            "reward_delta": 0.0,
            "violations": [],
            "summary": "no_interactions",
        }

        if interactions is None:
            return 0.0, summary

        likes = int(interactions.likes_on_others)
        comments = int(interactions.comments_on_others)
        replies = int(interactions.replies_to_audience)
        targets = list(interactions.target_partner_ids or [])
        quality = float(interactions.avg_reply_quality)

        # 1) Energy cost (paid up front; can push creator below 0.2 -> burnout track).
        energy_cost = (
            INTERACT_ENERGY_LIKE * likes
            + INTERACT_ENERGY_COMMENT * comments
            + INTERACT_ENERGY_REPLY * replies
        )
        self._energy = max(0.0, self._energy - energy_cost)

        # Determine off-niche share among interaction targets.
        off_niche_share = 0.0
        if targets:
            off = 0
            for tid in targets:
                if self._partner_niche(tid) != self._user_niche:
                    off += 1
            off_niche_share = off / len(targets)

        # 2) Reach buffs (next post engagement multiplier) — only when on-niche and within healthy band.
        on_niche_share = 1.0 - off_niche_share
        reach_mult = 1.0
        if on_niche_share > 0:
            if INTERACT_HEALTHY_LIKES[0] <= likes <= INTERACT_HEALTHY_LIKES[1]:
                reach_mult *= 1.0 + INTERACT_LIKE_REACH_BUFF * on_niche_share
            if INTERACT_HEALTHY_COMMENTS[0] <= comments <= INTERACT_HEALTHY_COMMENTS[1]:
                reach_mult *= 1.0 + INTERACT_COMMENT_REACH_BUFF * on_niche_share

        reward_delta = 0.0

        # 3) Reply reward (audience loyalty), scaled by quality.
        reply_weight = INTERACT_LOWQ_WEIGHT if quality < INTERACT_LOWQ_THRESHOLD else 1.0
        reply_reward = min(
            INTERACT_REPLY_REWARD_CAP,
            INTERACT_REPLY_REWARD_PER * replies * quality * reply_weight,
        )
        reward_delta += reply_reward

        # 4) Penalties — each surfaces a violation string.
        # 4a) Spam volume.
        if likes > INTERACT_SPAM_LIKES or comments > INTERACT_SPAM_COMMENTS:
            reach_mult *= INTERACT_SPAM_REACH_PENALTY
            self._shadowban_risk = min(1.0, self._shadowban_risk + INTERACT_SPAM_SHADOWBAN_BUMP)
            self._interaction_violations.append(
                f"interaction_spam:likes={likes},comments={comments}"
            )

        # 4b) Off-niche heavy interaction.
        if off_niche_share >= INTERACT_OFFNICHE_THRESHOLD and len(targets) >= 3:
            reach_mult *= INTERACT_OFFNICHE_REACH_PENALTY
            self._interaction_violations.append(
                f"interaction_off_niche:share={off_niche_share:.2f}"
            )

        # 4c) Ignoring own audience: expected_replies = K * recent_engagement_proxy (use last day's posts)
        prev_day = max(0, self._day - 1)
        expected_signal = self._posts_per_day.get(prev_day, 0)  # # posts yesterday as a proxy
        # Multiply by a small constant so 1 post = 1 expected reply unit floor.
        expected_replies = expected_signal * 1.0
        if expected_replies > 0 and replies < INTERACT_IGNORE_THRESHOLD_K * expected_replies * 20:
            # Compounding loyalty drop on engagement_rate, capped at 0.5x floor.
            self._engagement_rate_loyalty_mult = max(
                0.5, self._engagement_rate_loyalty_mult * INTERACT_IGNORE_LOYALTY_DECAY
            )
            self._interaction_violations.append(
                f"interaction_ignoring_own:replies={replies}"
            )

        # 4d) Low quality replies — already weighted; if extremely low quality, additional penalty.
        if replies > 0 and quality < INTERACT_VERY_LOWQ_THRESHOLD:
            reward_delta += INTERACT_VERY_LOWQ_PENALTY
            self._interaction_violations.append(
                f"interaction_low_quality:q={quality:.2f}"
            )

        # 4e) Energy: covered upstream; just record if it pushed creator into low-energy zone.
        if energy_cost > 0 and self._energy < 0.2:
            self._interaction_violations.append(
                f"interaction_energy_drain:residual_energy={self._energy:.2f}"
            )

        # Cap daily reward_delta to avoid blowing past the per-step [0,1] reward envelope.
        reward_delta = max(-INTERACT_DAILY_REWARD_CAP, min(INTERACT_DAILY_REWARD_CAP, reward_delta))

        # Persist computed reach_mult so today's hourly posts pick it up.
        self._pending_reach_mult = max(0.5, reach_mult)

        # Decay shadowban_risk slightly on quiet days (0 likes & 0 comments).
        if likes == 0 and comments == 0:
            self._shadowban_risk = max(0.0, self._shadowban_risk - 0.05)

        summary.update({
            "likes_on_others": likes,
            "comments_on_others": comments,
            "replies_to_audience": replies,
            "energy_cost": round(energy_cost, 4),
            "reach_modifier": round(self._pending_reach_mult, 3),
            "shadowban_risk": round(self._shadowban_risk, 3),
            "loyalty_mult": round(self._engagement_rate_loyalty_mult, 3),
            "off_niche_share": round(off_niche_share, 2),
            "reward_delta": round(reward_delta, 4),
            "violations": list(self._interaction_violations),
            "summary": (
                "spam" if likes > INTERACT_SPAM_LIKES or comments > INTERACT_SPAM_COMMENTS
                else "off_niche" if off_niche_share >= INTERACT_OFFNICHE_THRESHOLD and len(targets) >= 3
                else "low_quality" if replies > 0 and quality < INTERACT_VERY_LOWQ_THRESHOLD
                else "ignoring_own" if expected_replies > 0 and replies < INTERACT_IGNORE_THRESHOLD_K * expected_replies * 20
                else "healthy" if reward_delta > 0 or reach_mult > 1.0
                else "neutral"
            ),
        })
        return reward_delta, summary

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
                ev = self._collab_evaluation(comp.id)
                pool.append({
                    "id": comp.id,
                    "name": comp.name,
                    "niche": comp.niche,
                    "audience_overlap": ev.get("overlap"),
                    "mock_followers": ev.get("partner_followers"),
                    "intersection_size": ev.get("intersection_size"),
                    "same_niche": ev.get("same_niche"),
                    "follower_gap_pct": ev.get("follower_gap_pct"),
                    "recommended": ev.get("recommended"),
                    "reason": ev.get("reason"),
                    "expected_eng_mult": ev.get("eng_mult"),
                    "expected_growth_mult": ev.get("growth_mult"),
                    "prior_collabs_with_partner": ev.get("prior_count_with_partner"),
                    "repeat_decay": ev.get("repeat_decay"),
                })
            return ToolResult(
                name=tool.name,
                data={
                    "user_niche": self._user_niche,
                    "user_followers": int(self._followers),
                    "pool": pool,
                },
                budget_remaining=self._api_budget,
            )

        elif tool.name == "propose_collab":
            partner_id = tool.arguments.get("partner_id", "")
            if partner_id not in [c.id for c in self._competitors]:
                return ToolResult(name=tool.name, success=False, error=f"unknown partner: {partner_id}", budget_remaining=self._api_budget)
            ev = self._collab_evaluation(partner_id)
            return ToolResult(
                name=tool.name,
                data={
                    "status": "proposal_accepted" if ev["recommended"] else "proposal_accepted_with_warning",
                    "partner_id": partner_id,
                    "recommended": ev["recommended"],
                    "reason": ev["reason"],
                    "same_niche": ev["same_niche"],
                    "audience_overlap": ev["overlap"],
                    "intersection_size": ev["intersection_size"],
                    "expected_eng_mult": ev["eng_mult"],
                    "expected_growth_mult": ev["growth_mult"],
                    "prior_collabs_with_partner": ev["prior_count_with_partner"],
                    "repeat_decay": ev["repeat_decay"],
                },
                budget_remaining=self._api_budget,
            )

        elif tool.name == "query_interaction_norms":
            return ToolResult(
                name=tool.name,
                data={
                    "healthy_likes_per_day": list(INTERACT_HEALTHY_LIKES),
                    "healthy_comments_per_day": list(INTERACT_HEALTHY_COMMENTS),
                    "spam_threshold_likes": INTERACT_SPAM_LIKES,
                    "spam_threshold_comments": INTERACT_SPAM_COMMENTS,
                    "off_niche_share_max": INTERACT_OFFNICHE_THRESHOLD,
                    "min_reply_quality": INTERACT_LOWQ_THRESHOLD,
                    "current_shadowban_risk": round(self._shadowban_risk, 3),
                    "user_niche": self._user_niche,
                    "expected_replies_per_unit_engagement": INTERACT_IGNORE_THRESHOLD_K,
                },
                budget_remaining=self._api_budget,
            )

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
        # Collab guardrail breaches surfaced by _collab_multipliers (forced past block).
        for v in self._collab_violations:
            violations.append(v)
            pc -= 0.10
        # Interaction system violations (spam/off-niche/ignoring/low-quality/energy-drain).
        for v in self._interaction_violations:
            violations.append(v)
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

        # Optional user-niche override (for collab same/diff niche scenarios).
        user_niche_override = kwargs.get("user_niche")
        if user_niche_override:
            self._user_niche = str(user_niche_override)

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
        self._collab_violations = []
        if action.collab:
            self._collabs_this_month += 1
            self._collab_history.append(action.collab.partner_id)
            self._active_collab = action.collab

        # Process interactions BEFORE the day's hourly loop so energy cost and reach buffs/penalties
        # influence the same day's posts.
        interaction_reward, interaction_summary = self._process_interactions(action.interactions)

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

        # Collab requires a post at collab.hour (energy + post-count come from posting itself).
        # If the agent didn't schedule one, auto-inject a co-authored post using the collab's
        # content_type. This guarantees collab pays energy and counts as a post — no free multiplier.
        if self._active_collab is not None:
            chour = self._active_collab.hour
            existing = schedule.get(chour)
            if existing is None or existing.action_type != "post":
                ct = self._active_collab.content_type or "reel"
                partner_niche = self._partner_niche(self._active_collab.partner_id)
                topic = self._collab_default_topic(partner_niche)
                schedule[chour] = ScheduledAction(
                    hour=chour, action_type="post", content_type=ct,
                    topic=topic, tags=self._collab_default_tags(partner_niche),
                    intent="watch_bait" if ct in ("reel", "story") else "save_bait",
                )

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

        # Tick down the post-collab algorithm carryover (one day per step).
        if self._collab_carryover_days_remaining > 0:
            self._collab_carryover_days_remaining -= 1

        # Burnout risk tracking
        if energy_min < 0.2:
            self._low_energy_days += 1
        else:
            self._low_energy_days = max(0, self._low_energy_days - 1)

        prev_day = max(0, self._day - 1)
        if 1 <= self._posts_per_day.get(prev_day, 0) <= 2:
            self._days_with_good_posts.add(prev_day)

        # Apply ignored-audience compounding loyalty multiplier into the per-day reward.
        avg_reward = (daily_reward / 24.0) + interaction_reward
        avg_reward = max(0.0, min(1.0, avg_reward))
        error_str = "; ".join(errors) if errors else None

        # Finalize this step's interaction summary on the obs.
        self._last_interaction_summary = interaction_summary

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
                interaction_metrics=interaction_summary,
            )
            return self._final_observation

        return self._build_observation(
            reward=round(avg_reward, 4), error=error_str,
            daily_total_engagement=daily_engagement,
            daily_posts_made=daily_posts, daily_energy_min=energy_min,
            tool_results=tool_results, engagement_signals=daily_signals,
            coach_feedback=coach, judge_report=judge,
            interaction_metrics=interaction_summary,
        )

    def _process_hour_action(self, sa: ScheduledAction) -> Tuple[float, float, Optional[EngagementSignals]]:
        engagement = 0.0
        signals = None

        collab_growth_mult = 1.0
        collab_spillover_followers = 0

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

                # Interaction-driven reach modifier (set by _process_interactions earlier this step).
                # Multiplicative on engagement; capped at 0.5 floor inside _process_interactions.
                engagement *= getattr(self, "_pending_reach_mult", 1.0)

                # Sustained post-collab algorithm reach buff (applies to every post in the carryover window).
                if getattr(self, "_collab_carryover_days_remaining", 0) > 0:
                    engagement *= COLLAB_REACH_CARRYOVER_MULT

                collab_spillover_followers = 0
                if self._active_collab is not None and self._active_collab.hour == sa.hour:
                    ev = self._collab_evaluation(self._active_collab.partner_id)
                    eng_m, growth_m = self._collab_multipliers(self._active_collab.partner_id)
                    engagement *= eng_m
                    collab_growth_mult = growth_m
                    # One-shot follower spillover: partner audience gets exposed to user.
                    # Scales with (1 - overlap) — disjoint audiences = more new followers.
                    # repeat_decay flows through growth_m (already baked in by _collab_evaluation),
                    # so we don't multiply by it again here.
                    overlap = ev.get("overlap") or 0.0
                    partner_followers = ev.get("partner_followers") or 0
                    spill_k = COLLAB_BLOCKED_SPILLOVER_K if not ev.get("recommended") else COLLAB_SPILLOVER_K
                    collab_spillover_followers = int(
                        partner_followers * (1.0 - overlap) * growth_m * spill_k
                    )
                    self._last_collab_spillover = collab_spillover_followers
                    # Arm the post-collab carryover window for the next N days.
                    self._collab_carryover_days_remaining = COLLAB_REACH_CARRYOVER_DAYS

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
                if collab_spillover_followers > 0:
                    self._followers += collab_spillover_followers

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
        t_words = set(topic.lower().split()) - TREND_MATCH_STOPWORDS
        if not t_words:
            return False
        for trend in self._trending_topics:
            if t_words & (set(trend.lower().split()) - TREND_MATCH_STOPWORDS):
                return True
        return False

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
        is_post = sa.action_type == "post"
        trending_topic_mult = 1.5 if is_post and self._is_topic_trending(sa.topic) else 1.0
        peak_hour_mult = 1.3 if is_post and self._get_hour_multiplier() >= 1.2 else 1.0
        raw = (
            (eng_component + tag_component + comp_component) * trending_topic_mult * peak_hour_mult
            + energy_component + consistency_component - burnout_penalty
        )
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
        interaction_metrics: Optional[Dict[str, Any]] = None,
    ) -> ViraltestObservation:
        recent_eng = self._engagement_history[-10:] if self._engagement_history else []
        eng_rate = sum(recent_eng) / len(recent_eng) if recent_eng else 0.0
        eng_rate *= getattr(self, "_engagement_rate_loyalty_mult", 1.0)

        meta: Dict[str, Any] = {"step": self._state.step_count, "task": self._task}
        if grader_score is not None:
            meta["grader_score"] = round(grader_score, 4)

        audience_hours: set = set()
        for seg in _AUDIENCE_DATA.get("segments", []):
            audience_hours.update(seg.get("active_hours", []))
        meta["audience_active_hours"] = sorted(audience_hours)

        comp_hours = [
            (self._hour - p["hours_ago"]) % 24
            for comp in self._competitors
            for p in comp.recent_posts
            if p["hours_ago"] < 48
        ]
        meta["competitor_recent_post_hours"] = sorted(comp_hours)

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
            interaction_metrics=interaction_metrics,
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
        tag_discovery = min(1.0, positive_tags / TAG_DISCOVERY_POSITIVE_TARGET)
        top_perfs = sorted([self._tag_performance_avg(t) for t in self._unique_tags_used], reverse=True)[:3]
        tag_exploitation = (sum(top_perfs) / len(top_perfs)) if top_perfs else 0.0
        tag_exploitation = min(1.0, tag_exploitation / 2.0)
        tag_score = 0.4 * tag_discovery + 0.6 * tag_exploitation

        avg_energy = sum(self._energy_history) / len(self._energy_history) if self._energy_history else 0.0
        consistency = len(self._days_with_good_posts) / float(max(1, TASK_HORIZON))

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
        tag_discovery = min(1.0, positive_tags / TAG_DISCOVERY_POSITIVE_TARGET)
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
