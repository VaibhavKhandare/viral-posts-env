"""
Viraltest Environment — RL-Based Creator Optimization Simulation.

Simulates a social media creator's weekly posting lifecycle.
The agent decides when to post, what format, which tags, and how
to differentiate from competitors, while managing burnout.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ViraltestAction, ViraltestObservation
except ImportError:
    from models import ViraltestAction, ViraltestObservation

# ---------------------------------------------------------------------------
# Constants (research-backed)
# ---------------------------------------------------------------------------

TASK_HORIZON = 168  # 7 days × 24 hours

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
    "text_post": 0.37,
}

REACH_MULT = {
    "reel": 2.25,
    "carousel": 1.0,
    "story": 0.5,
    "text_post": 0.44,
}

TAG_POOL = [
    # Tech
    "ai", "ml", "coding", "startup", "saas", "devtools",
    # Lifestyle
    "fitness", "travel", "food", "wellness", "fashion", "photography",
    # Trending (base set — rotated daily)
    "summer", "worldcup", "election", "newyear", "oscars", "climate",
    # Niche
    "productivity", "minimalism", "stoic", "web3", "gaming", "crypto",
    # Broad
    "motivation", "tips", "howto", "viral", "trending", "growth",
]

TOPIC_CATEGORIES = {
    "tech": ["AI tools", "coding tips", "startup life", "tech news", "SaaS growth", "dev workflow"],
    "lifestyle": ["fitness routine", "travel guide", "food recipe", "wellness tips", "fashion haul", "photo editing"],
    "business": ["growth hacks", "marketing strategy", "creator economy", "monetization", "brand deals", "analytics"],
}

VALID_TASKS = ("weekly_engage", "weekly_strategic", "weekly_competitive")

# Hour multipliers (Buffer 9.6M post study)
PEAK_HOURS = {
    "weekday_morning": (9, 12, 1.3),
    "weekday_peak": (12, 15, 1.4),
    "evening": (18, 20, 1.25),
    "late_evening": (20, 23, 1.1),
    "night": (23, 6, 0.5),
    "off_hours": (6, 9, 0.8),
}

WEEKEND_PENALTY = 0.7
PEAK_DAYS = (1, 2, 3)  # Tue, Wed, Thu (0=Mon)


@dataclass
class CompetitorState:
    name: str
    niche_topics: List[str]
    preferred_types: List[str]
    posting_frequency: float
    base_engagement: float
    tag_preferences: List[str]
    recent_posts: List[Dict[str, Any]] = field(default_factory=list)


COMPETITOR_PROFILES = [
    {
        "name": "creator_alpha",
        "niche_topics": ["AI tools", "coding tips", "tech news"],
        "preferred_types": ["reel", "carousel"],
        "posting_frequency": 2.5,
        "base_engagement": 0.45,
        "tag_preferences": ["ai", "coding", "tech news"],
    },
    {
        "name": "creator_beta",
        "niche_topics": ["growth hacks", "marketing strategy", "creator economy"],
        "preferred_types": ["carousel", "text_post"],
        "posting_frequency": 1.8,
        "base_engagement": 0.40,
        "tag_preferences": ["growth", "tips", "viral"],
    },
    {
        "name": "creator_gamma",
        "niche_topics": ["fitness routine", "wellness tips", "motivation"],
        "preferred_types": ["reel", "story"],
        "posting_frequency": 3.0,
        "base_engagement": 0.38,
        "tag_preferences": ["fitness", "wellness", "motivation"],
    },
]

INITIAL_FOLLOWERS = 10000
REST_RECOVERY = 0.12
CREATE_CONTENT_COST = 0.05
REPETITION_ENERGY_PENALTY = 0.05
AUDIENCE_FATIGUE_THRESHOLD_1 = 3
AUDIENCE_FATIGUE_THRESHOLD_2 = 5
FOLLOWER_DECAY_HOURS = 48
ALGORITHM_PENALTY_MULT = 0.6
ALGORITHM_PENALTY_DURATION = 2


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ViraltestEnvironment(Environment):
    """
    Weekly creator optimization simulation.

    The agent manages a social media creator's posting strategy over 7 days
    (168 hourly steps), balancing engagement, energy, tags, and competition.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = "weekly_engage"
        self._rng = random.Random(42)
        self._init_state()

    def _init_state(self) -> None:
        self._energy = 1.0
        self._followers = INITIAL_FOLLOWERS
        self._initial_followers = INITIAL_FOLLOWERS
        self._hour = 9
        self._day = 0  # 0=Mon
        self._posts_today = 0
        self._last_post_types: List[str] = []
        self._time_since_last_post = 0
        self._engagement_history: List[float] = []
        self._tag_history: Dict[str, List[float]] = defaultdict(list)
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

        self._trending_topics = self._pick_trending_topics()
        self._trending_tags = self._pick_trending_tags()
        self._competitors = [CompetitorState(**p) for p in COMPETITOR_PROFILES]

    # ----- trend rotation -----

    def _pick_trending_topics(self) -> List[str]:
        all_topics = []
        for cat_topics in TOPIC_CATEGORIES.values():
            all_topics.extend(cat_topics)
        return self._rng.sample(all_topics, min(3, len(all_topics)))

    def _pick_trending_tags(self) -> List[str]:
        return self._rng.sample(TAG_POOL, min(5, len(TAG_POOL)))

    def _rotate_trends(self) -> None:
        self._trending_topics = self._pick_trending_topics()
        self._trending_tags = self._pick_trending_tags()

    # ----- hour multiplier -----

    def _get_hour_multiplier(self) -> float:
        h = self._hour
        d = self._day

        is_weekend = d >= 5
        base = WEEKEND_PENALTY if is_weekend else 1.0

        if 12 <= h < 15 and d in PEAK_DAYS:
            return base * 1.4
        if 9 <= h < 12:
            return base * 1.3
        if 18 <= h < 20:
            return base * 1.25
        if 20 <= h < 23:
            return base * 1.1
        if h >= 23 or h < 6:
            return base * 0.5
        return base * 0.8

    # ----- quality -----

    @staticmethod
    def _get_quality_modifier(energy: float) -> float:
        if energy > 0.5:
            return 1.0
        return max(0.48, energy * 1.5)

    # ----- tags -----

    def _calc_tag_boost(self, tags: Optional[List[str]]) -> float:
        if not tags:
            return 1.0
        trending_count = sum(1 for t in tags if t in self._trending_tags)
        perf_values = [
            self._tag_performance_avg(t) for t in tags if self._tag_performance_avg(t) > 0
        ]
        perf_avg = sum(perf_values) / len(perf_values) if perf_values else 0.0
        return 1.0 + 0.1 * trending_count + 0.05 * perf_avg

    def _tag_performance_avg(self, tag: str) -> float:
        history = self._tag_history.get(tag, [])
        if not history:
            return 0.0
        window = history[-5:]
        return sum(window) / len(window)

    def _get_tag_performance_dict(self) -> Dict[str, float]:
        return {tag: self._tag_performance_avg(tag) for tag in self._unique_tags_used}

    # ----- competitors -----

    def _advance_competitors(self) -> None:
        for comp in self._competitors:
            for p in comp.recent_posts:
                p["hours_ago"] += 1
            comp.recent_posts = [p for p in comp.recent_posts if p["hours_ago"] < 48]

            post_prob = comp.posting_frequency / 24.0
            if self._rng.random() < post_prob:
                ct = self._rng.choice(comp.preferred_types)
                topic = self._rng.choice(comp.niche_topics)
                tags = self._rng.sample(
                    comp.tag_preferences, min(3, len(comp.tag_preferences))
                )
                eng = comp.base_engagement + self._rng.uniform(-0.1, 0.1)
                eng = max(0.0, min(1.0, eng))
                comp.recent_posts.append({
                    "content_type": ct,
                    "topic": topic,
                    "tags": tags,
                    "engagement": round(eng, 3),
                    "hours_ago": 0,
                })

    def _get_competitor_recent_posts(self, limit: int = 5) -> List[Dict[str, Any]]:
        all_posts: List[Dict[str, Any]] = []
        for comp in self._competitors:
            for p in comp.recent_posts:
                all_posts.append(p)
        all_posts.sort(key=lambda x: x["hours_ago"])
        return all_posts[:limit]

    def _get_competitor_avg_engagement(self) -> float:
        engagements = []
        for comp in self._competitors:
            for p in comp.recent_posts:
                engagements.append(p["engagement"])
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
        recent_topics = []
        for comp in self._competitors:
            for p in comp.recent_posts:
                if p["hours_ago"] < 12:
                    recent_topics.append(p["topic"].lower())
        topic_lower = topic.lower()
        has_overlap = any(_topic_overlap(topic_lower, t) for t in recent_topics)
        if not has_overlap:
            return 1.3
        if saturation > 0.7:
            return 0.6
        return 1.0

    # ----- core API -----

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ViraltestObservation:
        self._task = kwargs.get("task", "weekly_engage")
        if self._task not in VALID_TASKS:
            self._task = "weekly_engage"

        self._rng = random.Random(seed if seed is not None else 42)
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        self._init_state()

        return self._build_observation(reward=0.0, error=None)

    def step(self, action: ViraltestAction, **kwargs: Any) -> ViraltestObservation:  # type: ignore[override]
        if self._episode_done and self._final_observation is not None:
            return self._final_observation

        self._state.step_count += 1
        error: Optional[str] = None
        engagement = 0.0

        # 1. Validate
        error = self._validate_action(action)
        if error:
            self._time_since_last_post += 1
            self._advance_competitors()
            self._advance_time()
            self._energy_history.append(self._energy)
            return self._build_observation(reward=0.0, error=error)

        # 2. Apply energy cost
        used_queue = False
        if action.action_type == "post":
            cost = CONTENT_ENERGY_COST.get(action.content_type, 0.1)  # type: ignore[arg-type]
            if self._content_queue > 0:
                cost *= 0.5  # pre-made content = half the effort
                self._content_queue -= 1
                used_queue = True
            if len(self._last_post_types) >= 3 and all(
                t == action.content_type for t in self._last_post_types[-3:]
            ):
                cost += REPETITION_ENERGY_PENALTY
            self._energy = max(0.0, self._energy - cost)
            self._unique_content_types.add(action.content_type)  # type: ignore[arg-type]

        elif action.action_type == "rest":
            self._energy = min(1.0, self._energy + REST_RECOVERY)

        elif action.action_type == "create_content":
            self._energy = max(0.0, self._energy - CREATE_CONTENT_COST)
            self._content_queue += 1

        # 3. Calc engagement (post only)
        if action.action_type == "post":
            if self._energy <= 0.0:
                engagement = 0.0
            else:
                base = BASE_ENGAGEMENT.get(action.content_type, 0.3)  # type: ignore[arg-type]
                reach = REACH_MULT.get(action.content_type, 1.0)  # type: ignore[arg-type]
                hour_mult = self._get_hour_multiplier()
                quality = self._get_quality_modifier(self._energy)
                tag_boost = self._calc_tag_boost(action.tags)
                trending_bonus = 1.5 if self._is_topic_trending(action.topic) else 1.0
                comp_diff = self._calc_competitor_diff(action.topic)

                fatigue = 1.0
                if self._posts_today >= AUDIENCE_FATIGUE_THRESHOLD_2:
                    fatigue = 0.1
                elif self._posts_today >= AUDIENCE_FATIGUE_THRESHOLD_1:
                    fatigue = 0.5

                algo_mult = 1.0
                if self._algorithm_penalty_remaining > 0:
                    algo_mult = ALGORITHM_PENALTY_MULT
                    self._algorithm_penalty_remaining -= 1

                engagement = (
                    base * reach * hour_mult * quality * tag_boost
                    * trending_bonus * comp_diff * fatigue * algo_mult
                )
                engagement = min(engagement, 5.0)

            self._last_topic = action.topic

            # 4. Update tag performance
            if action.tags and engagement > 0:
                for tag in action.tags:
                    tag_lower = tag.lower()
                    self._tag_history[tag_lower].append(engagement)
                    self._unique_tags_used.add(tag_lower)

            self._engagement_history.append(engagement)
            self._total_engagement += engagement
            self._posting_steps += 1

            if self._calc_competitor_diff(action.topic) >= 1.3:
                self._unique_topic_steps += 1

            self._last_post_types.append(action.content_type)  # type: ignore[arg-type]
            if len(self._last_post_types) > 3:
                self._last_post_types = self._last_post_types[-3:]
            self._posts_today += 1
            self._posts_per_day[self._day] += 1
            self._time_since_last_post = 0

        else:
            self._time_since_last_post += 1

        # 5. Advance competitors
        self._advance_competitors()

        # 6. Update followers
        if action.action_type == "post" and engagement > 0:
            self._followers += int(engagement * 100)
        if self._time_since_last_post >= FOLLOWER_DECAY_HOURS:
            self._followers = max(0, self._followers - int(self._followers * 0.005))
            if self._algorithm_penalty_remaining == 0:
                self._algorithm_penalty_remaining = ALGORITHM_PENALTY_DURATION

        # 7. Advance time
        self._advance_time()

        self._energy_history.append(self._energy)

        day_posts = self._posts_per_day.get(self._day, 0)
        if 1 <= day_posts <= 2:
            self._days_with_good_posts.add(self._day)

        # 8. Compute reward (0 if burned out)
        reward = 0.0 if self._energy <= 0.0 else self._compute_reward(action, engagement)

        # 9. Check done
        done = self._state.step_count >= TASK_HORIZON or self._energy <= 0.0
        if done:
            self._episode_done = True
            grader_score = self._run_grader()
            self._final_observation = self._build_observation(
                reward=reward, error=error, done=True, grader_score=grader_score
            )
            return self._final_observation

        return self._build_observation(reward=reward, error=error)

    @property
    def state(self) -> State:
        return self._state

    # ----- validation -----

    def _validate_action(self, action: ViraltestAction) -> Optional[str]:
        if action.action_type not in ("post", "rest", "create_content"):
            return f"Invalid action_type: {action.action_type}"
        if action.action_type == "post":
            if not action.content_type:
                return "content_type is required when posting"
            if action.content_type not in CONTENT_ENERGY_COST:
                return f"Invalid content_type: {action.content_type}"
            if not action.topic or not action.topic.strip():
                return "topic is required when posting"
            if len(action.topic) > 200:
                return "topic must be ≤200 characters"
            if action.tags:
                valid = [t for t in action.tags if t.lower() in TAG_POOL]
                action.tags = valid if valid else None
        return None

    # ----- trending -----

    def _is_topic_trending(self, topic: Optional[str]) -> bool:
        if not topic:
            return False
        topic_lower = topic.lower()
        return any(t.lower() in topic_lower for t in self._trending_topics)

    # ----- reward -----

    def _compute_reward(self, action: ViraltestAction, engagement: float) -> float:
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
        if action.action_type == "post" and action.tags:
            trending_match = sum(1 for t in action.tags if t.lower() in self._trending_tags) / 5.0
            tag_component = min(1.0, trending_match + 0.3) * 0.15

        comp_component = 0.0
        if action.action_type == "post":
            diff = self._calc_competitor_diff(action.topic)
            comp_component = min(1.0, diff / 1.3) * 0.15

        burnout_penalty = 0.1 if self._energy < 0.2 else 0.0

        raw = eng_component + energy_component + consistency_component + tag_component + comp_component - burnout_penalty
        return max(0.0, min(1.0, raw))

    # ----- time -----

    def _advance_time(self) -> None:
        self._hour += 1
        if self._hour >= 24:
            self._hour = 0
            self._day += 1
            self._posts_today = 0
            self._rotate_trends()

    # ----- observation builder -----

    def _build_observation(
        self,
        reward: float,
        error: Optional[str],
        done: bool = False,
        grader_score: Optional[float] = None,
    ) -> ViraltestObservation:
        recent_eng = self._engagement_history[-10:] if self._engagement_history else []
        eng_rate = sum(recent_eng) / len(recent_eng) if recent_eng else 0.0

        meta: Dict[str, Any] = {"step": self._state.step_count, "task": self._task}
        if grader_score is not None:
            meta["grader_score"] = round(grader_score, 4)

        return ViraltestObservation(
            current_hour=self._hour,
            day_of_week=self._day % 7,
            days_elapsed=self._day,
            creator_energy=round(self._energy, 3),
            follower_count=self._followers,
            engagement_rate=round(eng_rate, 4),
            posts_today=self._posts_today,
            time_since_last_post=self._time_since_last_post,
            trending_topics=list(self._trending_topics),
            content_queue_size=self._content_queue,
            last_post_type=self._last_post_types[-1] if self._last_post_types else "none",
            tag_performance=self._get_tag_performance_dict(),
            trending_tags=list(self._trending_tags),
            competitor_recent_posts=self._get_competitor_recent_posts(),
            competitor_avg_engagement=round(self._get_competitor_avg_engagement(), 4),
            niche_saturation=round(self._calc_niche_saturation(self._last_topic), 3),
            error=error,
            done=done,
            reward=round(reward, 4),
            metadata=meta,
        )

    # ----- graders -----

    def _run_grader(self) -> float:
        if self._task == "weekly_engage":
            return self._grade_weekly_engage()
        elif self._task == "weekly_strategic":
            return self._grade_weekly_strategic()
        elif self._task == "weekly_competitive":
            return self._grade_weekly_competitive()
        return 0.0

    def _theoretical_max_engagement(self) -> float:
        best_base = max(BASE_ENGAGEMENT.values())
        best_reach = max(REACH_MULT.values())
        peak_mult = 1.4
        quality = 1.0
        posts_per_day = 2
        days = 7
        return best_base * best_reach * peak_mult * quality * posts_per_day * days

    def _grade_weekly_engage(self) -> float:
        theoretical_max = self._theoretical_max_engagement()
        if theoretical_max <= 0:
            return 0.0
        raw = min(1.0, self._total_engagement / theoretical_max)
        if self._energy <= 0.0:
            raw *= 0.3  # burnout penalty even on easy task
        return raw

    def _grade_weekly_strategic(self) -> float:
        # Burnout = severe penalty (not total fail like competitive, but close)
        if self._energy <= 0.0:
            return max(0.0, min(0.15, self._total_engagement * 0.01))

        # Engagement: 35%
        theoretical_max = self._theoretical_max_engagement()
        norm_eng = min(1.0, self._total_engagement / theoretical_max) if theoretical_max > 0 else 0.0

        # Tag score: 25%  (40% discovery + 60% exploitation)
        positive_tags = sum(1 for t in self._unique_tags_used if self._tag_performance_avg(t) > 0)
        tag_discovery = min(1.0, positive_tags / 30.0)
        top_perfs = sorted(
            [self._tag_performance_avg(t) for t in self._unique_tags_used], reverse=True
        )[:3]
        tag_exploitation = (sum(top_perfs) / len(top_perfs)) if top_perfs else 0.0
        tag_exploitation = min(1.0, tag_exploitation / 2.0)
        tag_score = 0.4 * tag_discovery + 0.6 * tag_exploitation

        # Avg energy: 25%
        avg_energy = sum(self._energy_history) / len(self._energy_history) if self._energy_history else 0.0

        # Consistency: 15%
        consistency = len(self._days_with_good_posts) / 7.0

        raw = 0.35 * norm_eng + 0.25 * tag_score + 0.25 * avg_energy + 0.15 * consistency

        # Constraints
        min_energy = min(self._energy_history) if self._energy_history else 0.0
        if min_energy < 0.2:
            raw *= 0.4  # crashed hard
        elif min_energy < 0.3:
            raw = min(raw, 0.45)
        if len(self._unique_tags_used) < 5:
            raw *= 0.7

        return max(0.0, min(1.0, raw))

    def _grade_weekly_competitive(self) -> float:
        # Burnout = total fail
        if self._energy <= 0.0:
            return 0.0

        # Engagement: 25%
        theoretical_max = self._theoretical_max_engagement()
        norm_eng = min(1.0, self._total_engagement / theoretical_max) if theoretical_max > 0 else 0.0

        # Tag score: 20%
        positive_tags = sum(1 for t in self._unique_tags_used if self._tag_performance_avg(t) > 0)
        tag_discovery = min(1.0, positive_tags / 30.0)
        top_perfs = sorted(
            [self._tag_performance_avg(t) for t in self._unique_tags_used], reverse=True
        )[:3]
        tag_exploitation = (sum(top_perfs) / len(top_perfs)) if top_perfs else 0.0
        tag_exploitation = min(1.0, tag_exploitation / 2.0)
        tag_score = 0.4 * tag_discovery + 0.6 * tag_exploitation

        # Follower growth: 20%
        growth = (self._followers - self._initial_followers) / self._initial_followers if self._initial_followers > 0 else 0.0
        target_growth = 0.05
        norm_growth = min(1.0, max(0.0, growth / target_growth))

        # Competitor outperformance: 15%
        comp_avg = self._get_competitor_avg_engagement()
        my_avg = self._total_engagement / self._posting_steps if self._posting_steps > 0 else 0.0
        outperformance = my_avg / comp_avg if comp_avg > 0 else 1.0
        norm_outperformance = min(1.0, outperformance / 1.5)

        # Differentiation: 10%
        differentiation = self._unique_topic_steps / self._posting_steps if self._posting_steps > 0 else 0.0

        # Energy floor: 10%
        min_energy = min(self._energy_history) if self._energy_history else 0.0
        energy_floor = min(1.0, max(0.0, min_energy))

        raw = (
            0.25 * norm_eng
            + 0.20 * tag_score
            + 0.20 * norm_growth
            + 0.15 * norm_outperformance
            + 0.10 * differentiation
            + 0.10 * energy_floor
        )

        # Constraints
        if len(self._unique_content_types) < 3:
            raw *= 0.5
        if len(self._unique_tags_used) < 8:
            raw *= 0.7

        return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topic_overlap(topic_a: str, topic_b: str) -> bool:
    """Check if two topics have significant word overlap."""
    words_a = set(topic_a.split())
    words_b = set(topic_b.split())
    if not words_a or not words_b:
        return False
    common = words_a & words_b
    return len(common) / min(len(words_a), len(words_b)) >= 0.5
