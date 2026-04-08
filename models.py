"""Data models for the Viraltest Creator Optimization Environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator

VALID_CONTENT_TYPES = ("reel", "story", "carousel", "text_post")
VALID_ACTION_TYPES = ("post", "create_content")


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

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) > 5:
            return v[:5]
        return v


class ViraltestAction(Action):
    """Sparse daily plan: only non-rest actions. Unlisted hours default to rest."""

    scheduled_actions: List[ScheduledAction] = Field(
        default_factory=list,
        description="Actions scheduled at specific hours; unlisted hours are rest",
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


class ViraltestObservation(Observation):
    """Observation the agent receives after each daily step."""

    current_hour: int = Field(default=0, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)
    days_elapsed: int = Field(default=0, ge=0)
    creator_energy: float = Field(default=1.0, ge=0.0, le=1.0)
    hours_since_sleep: int = Field(default=0, ge=0, description="Hours since last sleep period")
    sleep_debt: float = Field(default=0.0, ge=0.0, le=1.0, description="Accumulated sleep debt (0=rested, 1=severe)")
    follower_count: int = Field(default=0, ge=0)
    engagement_rate: float = Field(default=0.0, ge=0.0)
    posts_today: int = Field(default=0, ge=0)
    time_since_last_post: int = Field(default=0, ge=0)
    trending_topics: List[str] = Field(default_factory=list)
    content_queue_size: int = Field(default=0, ge=0)
    last_post_type: str = Field(default="none")

    tag_performance: Dict[str, float] = Field(default_factory=dict)
    trending_tags: List[str] = Field(default_factory=list)

    competitor_recent_posts: List[Dict[str, Any]] = Field(default_factory=list)
    competitor_avg_engagement: float = Field(default=0.0, ge=0.0)
    niche_saturation: float = Field(default=0.0, ge=0.0, le=1.0)

    daily_total_engagement: float = Field(default=0.0, ge=0.0, description="Total engagement earned this day")
    daily_posts_made: int = Field(default=0, ge=0, description="Number of posts made this day")
    daily_energy_min: float = Field(default=1.0, ge=0.0, le=1.0, description="Lowest energy during this day")

    grader_score: Optional[float] = Field(default=None, description="Final grader score (set on last step when done=True)")

    error: Optional[str] = Field(default=None)
