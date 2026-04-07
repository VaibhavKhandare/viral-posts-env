"""Viraltest Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ViraltestAction, ViraltestObservation


class ViraltestEnv(
    EnvClient[ViraltestAction, ViraltestObservation, State]
):
    """
    Client for the Viraltest Creator Optimization Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with ViraltestEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="weekly_engage")
        ...     result = client.step(ViraltestAction(
        ...         action_type="post", content_type="reel",
        ...         topic="AI trends", tags=["ai", "tech"]
        ...     ))
    """

    def _step_payload(self, action: ViraltestAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.content_type is not None:
            payload["content_type"] = action.content_type
        if action.topic is not None:
            payload["topic"] = action.topic
        if action.tags is not None:
            payload["tags"] = action.tags
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ViraltestObservation]:
        obs_data = payload.get("observation", {})
        observation = ViraltestObservation(
            current_hour=obs_data.get("current_hour", 0),
            day_of_week=obs_data.get("day_of_week", 0),
            days_elapsed=obs_data.get("days_elapsed", 0),
            creator_energy=obs_data.get("creator_energy", 1.0),
            follower_count=obs_data.get("follower_count", 0),
            engagement_rate=obs_data.get("engagement_rate", 0.0),
            hours_since_sleep=obs_data.get("hours_since_sleep", 0),
            posts_today=obs_data.get("posts_today", 0),
            sleep_debt=obs_data.get("sleep_debt", 0.0),
            time_since_last_post=obs_data.get("time_since_last_post", 0),
            trending_topics=obs_data.get("trending_topics", []),
            content_queue_size=obs_data.get("content_queue_size", 0),
            last_post_type=obs_data.get("last_post_type", "none"),
            tag_performance=obs_data.get("tag_performance", {}),
            trending_tags=obs_data.get("trending_tags", []),
            competitor_recent_posts=obs_data.get("competitor_recent_posts", []),
            competitor_avg_engagement=obs_data.get("competitor_avg_engagement", 0.0),
            niche_saturation=obs_data.get("niche_saturation", 0.0),
            error=obs_data.get("error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
