"""Data models for the Viraltest Creator Optimization Environment (v2 — Theme #3.1)."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator

VALID_CONTENT_TYPES = ("reel", "story", "carousel", "text_post")
VALID_ACTION_TYPES = ("post", "create_content")
VALID_INTENTS = ("send_bait", "save_bait", "watch_bait", "like_bait")


class ToolCall(BaseModel):
    """A single tool invocation the agent wants to make before committing actions."""

    name: str = Field(..., description="Tool name from the /tools catalog")
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result returned from a single tool invocation."""

    name: str
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    budget_remaining: int = Field(default=100, ge=0)


class ScheduledAction(BaseModel):
    """A single non-rest action scheduled at a specific hour of the day."""

    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    action_type: Literal["post", "create_content"] = Field(
        ..., description="What to do at this hour (unlisted hours default to rest)"
    )
    content_type: Optional[Literal["reel", "story", "carousel", "text_post"]] = Field(
        default=None, description="Format of the post (required if posting)"
    )
    topic: Optional[str] = Field(
        default=None, max_length=200, description="Topic of the post"
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Hashtags for the post (max 5)"
    )
    intent: Optional[Literal["send_bait", "save_bait", "watch_bait", "like_bait"]] = Field(
        default=None,
        description="Mosseri signal the post optimizes for (affects which engagement signal gets boosted)",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) > 5:
            return v[:5]
        return v


class CollabProposal(BaseModel):
    """Propose a collaboration with a competitor archetype."""

    partner_id: str = Field(..., description="Competitor archetype id from competitors.json")
    content_type: Optional[Literal["reel", "story", "carousel", "text_post"]] = Field(default="reel")
    hour: int = Field(default=12, ge=0, le=23)


class ViraltestAction(Action):
    """Daily plan: tool calls for discovery, then scheduled actions to commit."""

    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations to run before committing actions (query_audience, query_trends, etc.)",
    )
    scheduled_actions: List[ScheduledAction] = Field(
        default_factory=list,
        description="Actions scheduled at specific hours; unlisted hours are rest",
    )
    collab: Optional[CollabProposal] = Field(
        default=None,
        description="Optional collaboration proposal (max 2 per month)",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Agent scratchpad — persisted and echoed back next step for belief tracking",
    )

    @field_validator("scheduled_actions")
    @classmethod
    def validate_no_duplicate_hours(cls, v: List[ScheduledAction]) -> List[ScheduledAction]:
        seen: set = set()
        deduped: List[ScheduledAction] = []
        for a in v:
            if a.hour not in seen:
                seen.add(a.hour)
                deduped.append(a)
        return deduped


class JudgeReport(BaseModel):
    """Auditable per-day evaluation by the in-env Regulator/Judge.

    Scores are 0..1. `sustainability_risk` is RISK (higher = worse).
    """

    policy_compliance: float = Field(default=1.0, ge=0.0, le=1.0)
    sustainability_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    strategic_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default="")
    violations: List[str] = Field(default_factory=list)


class HeadlineMetrics(BaseModel):
    """Three headline numbers reported once per episode (final observation)."""

    vs_baseline_pct: float = Field(default=0.0, description="(agent - heuristic_baseline) / heuristic_baseline")
    score_per_tool_call: float = Field(default=0.0, description="grader_score / total_tool_calls (efficiency)")
    score_per_1k_chars: float = Field(default=0.0, description="grader_score per 1k action chars (token-proxy efficiency)")
    retention_under_shift: Optional[float] = Field(
        default=None,
        description="shifted_score / baseline_score, populated when both runs share an episode_chain_id",
    )
    heuristic_baseline_score: float = Field(default=0.0)
    agent_score: float = Field(default=0.0)
    total_tool_calls: int = Field(default=0, ge=0)
    total_action_chars: int = Field(default=0, ge=0)


class EngagementSignals(BaseModel):
    """Mosseri-aligned engagement decomposition (Jan 2025 official ranking signals)."""

    watch_time: float = Field(default=0.0, ge=0.0, description="Reels watch time signal")
    sends_per_reach: float = Field(default=0.0, ge=0.0, description="DM shares signal (strongest for discovery)")
    saves: float = Field(default=0.0, ge=0.0, description="Bookmark signal (content quality)")
    likes_per_reach: float = Field(default=0.0, ge=0.0, description="Like signal (existing followers)")

    @property
    def weighted_total(self) -> float:
        return 0.4 * self.watch_time + 0.3 * self.sends_per_reach + 0.2 * self.saves + 0.1 * self.likes_per_reach


class ViraltestObservation(Observation):
    """Observation the agent receives after each daily step.

    Default observation is SPARSE (Theme #3.1 partial observability).
    Rich data (tag_performance, competitor_posts, trending) available only via tools.
    """

    current_hour: int = Field(default=0, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)
    days_elapsed: int = Field(default=0, ge=0)
    creator_energy: float = Field(default=1.0, ge=0.0, le=1.0)
    hours_since_sleep: int = Field(default=0, ge=0)
    sleep_debt: float = Field(default=0.0, ge=0.0, le=1.0)
    follower_count: int = Field(default=0, ge=0)
    engagement_rate: float = Field(default=0.0, ge=0.0)
    posts_today: int = Field(default=0, ge=0)
    time_since_last_post: int = Field(default=0, ge=0)
    content_queue_size: int = Field(default=0, ge=0)
    last_post_type: str = Field(default="none")
    burnout_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="0=safe, 1=imminent burnout")

    # Sparse: these are populated only when agent uses tools
    trending_topics: List[str] = Field(default_factory=list)
    trending_tags: List[str] = Field(default_factory=list)
    tag_performance: Dict[str, float] = Field(default_factory=dict)
    competitor_recent_posts: List[Dict[str, Any]] = Field(default_factory=list)
    competitor_avg_engagement: float = Field(default=0.0, ge=0.0)
    niche_saturation: float = Field(default=0.0, ge=0.0, le=1.0)

    daily_total_engagement: float = Field(default=0.0, ge=0.0)
    daily_posts_made: int = Field(default=0, ge=0)
    daily_energy_min: float = Field(default=1.0, ge=0.0, le=1.0)

    engagement_signals: Optional[EngagementSignals] = Field(
        default=None, description="Mosseri-aligned signal breakdown for the day"
    )
    coach_feedback: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Counterfactual feedback: delta between agent plan and heatmap-optimal plan",
    )
    judge_report: Optional[JudgeReport] = Field(
        default=None,
        description="Regulator/Judge audit: policy compliance, sustainability risk, strategic quality + explanation",
    )
    headline_metrics: Optional[HeadlineMetrics] = Field(
        default=None,
        description="Final-observation hard numbers: improvement vs baseline, efficiency, shift retention",
    )

    tool_results: List[ToolResult] = Field(default_factory=list, description="Results from tool_calls this step")
    agent_notes: Optional[str] = Field(default=None, description="Echo of agent's notes from previous step")
    api_budget_remaining: int = Field(default=100, ge=0)

    grader_score: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)
