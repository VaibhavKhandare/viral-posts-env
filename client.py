"""Viraltest Environment Client (v2 — Theme #3.1)."""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    EngagementSignals,
    ToolResult,
    ViraltestAction,
    ViraltestObservation,
)


class ViraltestEnv(EnvClient[ViraltestAction, ViraltestObservation, State]):
    """Client for the Viraltest Creator Optimization Environment v2."""

    def _step_payload(self, action: ViraltestAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}

        if action.tool_calls:
            payload["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in action.tool_calls
            ]

        actions_list = []
        for sa in action.scheduled_actions:
            item: Dict[str, Any] = {
                "hour": sa.hour,
                "action_type": sa.action_type,
            }
            if sa.content_type is not None:
                item["content_type"] = sa.content_type
            if sa.topic is not None:
                item["topic"] = sa.topic
            if sa.tags is not None:
                item["tags"] = sa.tags
            if sa.intent is not None:
                item["intent"] = sa.intent
            actions_list.append(item)
        payload["scheduled_actions"] = actions_list

        if action.collab:
            payload["collab"] = {
                "partner_id": action.collab.partner_id,
                "content_type": action.collab.content_type,
                "hour": action.collab.hour,
            }

        if action.notes is not None:
            payload["notes"] = action.notes

        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ViraltestObservation]:
        obs_data = payload.get("observation", {})
        grader_score = obs_data.get("grader_score")
        meta = obs_data.get("metadata", {})
        if grader_score is not None:
            meta["grader_score"] = grader_score

        signals_raw = obs_data.get("engagement_signals")
        signals = EngagementSignals(**signals_raw) if signals_raw else None

        tool_results_raw = obs_data.get("tool_results", [])
        tool_results = [ToolResult(**tr) for tr in tool_results_raw]

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
            burnout_risk=obs_data.get("burnout_risk", 0.0),
            tag_performance=obs_data.get("tag_performance", {}),
            trending_tags=obs_data.get("trending_tags", []),
            competitor_recent_posts=obs_data.get("competitor_recent_posts", []),
            competitor_avg_engagement=obs_data.get("competitor_avg_engagement", 0.0),
            niche_saturation=obs_data.get("niche_saturation", 0.0),
            daily_total_engagement=obs_data.get("daily_total_engagement", 0.0),
            daily_posts_made=obs_data.get("daily_posts_made", 0),
            daily_energy_min=obs_data.get("daily_energy_min", 1.0),
            engagement_signals=signals,
            coach_feedback=obs_data.get("coach_feedback"),
            tool_results=tool_results,
            agent_notes=obs_data.get("agent_notes"),
            api_budget_remaining=obs_data.get("api_budget_remaining", 100),
            grader_score=grader_score,
            error=obs_data.get("error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=meta,
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
