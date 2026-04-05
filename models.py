"""Data models for the Viraltest Creator Optimization Environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator

VALID_CONTENT_TYPES = ("reel", "story", "carousel", "text_post")
VALID_ACTION_TYPES = ("post", "rest", "create_content")


class ViraltestAction(Action):
    """Agent action: post content, rest, or prepare content."""

    action_type: Literal["post", "rest", "create_content"] = Field(
        ..., description="What the agent does this step"
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


class ViraltestObservation(Observation):
    """Full observation the agent receives each step."""

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

    error: Optional[str] = Field(default=None)
