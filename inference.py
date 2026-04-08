"""
Viraltest Inference Script — RL-Based Creator Optimization Agent
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN or OPENAI_API_KEY or API_KEY   API key for the LLM client.
    IMAGE_NAME or LOCAL_IMAGE_NAME   Docker image when using ViraltestEnv.from_docker_image()

Optional:
    ALLOW_SHORT_EPISODE=1   Allow MAX_STEPS below 7 (final grader score stays 0 if episode never ends).
    MAX_STEPS   Step cap (default 7). Without ALLOW_SHORT_EPISODE, cap is at least 7 so graders run.

Each step = one full day. The agent submits a sparse daily plan (only posts and create_content
actions at specific hours). Unlisted hours automatically become rest.

STDOUT FORMAT (single space after tag; score two decimals) — match hackathon sample exactly.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from viraltest import ScheduledAction, ViraltestAction, ViraltestEnv
from viraltest.server.viraltest_environment import (
    TAG_POOL,
    TASK_HORIZON,
    TOPIC_CATEGORIES,
)

DOCKER_IMAGE = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
BENCHMARK = os.getenv("VIRALTEST_BENCHMARK", "viraltest")

TASKS = ["weekly_engage", "weekly_strategic", "weekly_competitive"]
_ALLOW_SHORT = os.getenv("ALLOW_SHORT_EPISODE", "").lower() in ("1", "true", "yes")
_REQUESTED_MAX = int(os.getenv("MAX_STEPS", str(TASK_HORIZON)))
MAX_STEPS = _REQUESTED_MAX if _ALLOW_SHORT else max(_REQUESTED_MAX, TASK_HORIZON)
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.1

VALID_TAGS_TEXT = ", ".join(TAG_POOL)

# Flatten env topic categories — posts must use these exact strings (see sanitize_predefined_topics).
PREDEFINED_TOPICS: tuple[str, ...] = tuple(
    topic for topics in TOPIC_CATEGORIES.values() for topic in topics
)
_TOPIC_CANONICAL: dict[str, str] = {t.lower(): t for t in PREDEFINED_TOPICS}
PREDEFINED_TOPICS_TEXT = ", ".join(PREDEFINED_TOPICS)

# When energy is at or below this level, skip the model and rest the full day (avoid burnout).
NEAR_ZERO_ENERGY_THRESHOLD = 0.25

SYSTEM_PROMPT = textwrap.dedent(f"""\
You are a social media content strategy agent. Each step is one full day (24 hours).
You receive the current day's state and must plan your actions for the entire day.

Reply with a JSON object containing "scheduled_actions" — a list of actions at specific hours.
Hours you don't list will automatically be rest. Only include posts and create_content actions.

FORMAT (JSON only, no markdown, no prose):
{{
  "scheduled_actions": [
    {{"hour": 10, "action_type": "create_content"}},
    {{"hour": 12, "action_type": "post", "content_type": "reel", "topic": "AI tools", "tags": ["ai", "coding"]}},
    {{"hour": 18, "action_type": "post", "content_type": "carousel", "topic": "startup life", "tags": ["startup", "growth"]}}
  ]
}}

RULES:
- hour: 0-23 (which hour of the day to perform the action)
- action_type: "post" or "create_content" (rest is automatic for unlisted hours)
- For posts: content_type (reel|story|carousel|text_post), topic, and tags are required
- Topic must be exactly one of these strings (no paraphrasing): {PREDEFINED_TOPICS_TEXT}
- Tags must be from this pool: {VALID_TAGS_TEXT}
- Max 5 tags per post
- Empty scheduled_actions means rest all day
- Peak posting hours: 9-12 (1.3x), 12-15 Tue-Thu (1.4x), 18-20 (1.25x)
- Posting 3+ times/day causes audience fatigue; 1-2 posts/day is optimal
- If energy hits 0, episode ends (burnout = game over)

Plan strategically: schedule posts at peak hours, rest during off-hours to recover energy,
and use create_content to build a content queue for cheaper posts later.""")


def should_force_rest_day(obs: Any) -> bool:
    """If energy is near zero, always submit an empty schedule (all rest)."""
    energy = float(getattr(obs, "creator_energy", 1.0))
    return energy <= NEAR_ZERO_ENERGY_THRESHOLD


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace(" ", "_") if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def format_observation(obs: Any) -> str:
    """Serialize observation into a readable prompt for the LLM."""
    tag_perf = obs.tag_performance or {}
    top_tags = sorted(tag_perf.items(), key=lambda x: x[1], reverse=True)[:5]
    top_tags_str = ", ".join(f"{t}={v:.2f}" for t, v in top_tags) if top_tags else "none yet"

    comp_posts = obs.competitor_recent_posts or []
    comp_str = ""
    for p in comp_posts[:3]:
        comp_str += (
            f"  - {p.get('content_type','?')} on '{p.get('topic','?')}' "
            f"tags={p.get('tags',[])} eng={p.get('engagement',0):.2f} "
            f"({p.get('hours_ago',0)}h ago)\n"
        )
    if not comp_str:
        comp_str = "  none\n"

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_name = days[obs.day_of_week] if 0 <= obs.day_of_week < 7 else "?"

    daily_eng = getattr(obs, "daily_total_engagement", 0.0)
    daily_posts = getattr(obs, "daily_posts_made", 0)
    daily_emin = getattr(obs, "daily_energy_min", 1.0)

    return textwrap.dedent(f"""\
Day: {day_name} (day_of_week={obs.day_of_week}, 0=Mon) | days_elapsed={obs.days_elapsed}
Hours since sleep: {obs.hours_since_sleep} | Sleep debt: {obs.sleep_debt:.3f}
Energy: {obs.creator_energy:.2f} | Followers: {obs.follower_count} | Engagement rate: {obs.engagement_rate:.3f}
Hours since last post: {obs.time_since_last_post}
Content queue: {obs.content_queue_size} | Last post type: {obs.last_post_type}
Yesterday's engagement: {daily_eng:.3f} | Yesterday's posts: {daily_posts} | Yesterday's min energy: {daily_emin:.2f}
Trending topics: {', '.join(obs.trending_topics)}
Trending tags: {', '.join(obs.trending_tags)}
Your top tags: {top_tags_str}
Niche saturation: {obs.niche_saturation:.2f} | Competitor avg engagement: {obs.competitor_avg_engagement:.3f}
Competitor recent posts:
{comp_str}Plan your actions for today (list only posts and create_content at specific hours):""")


def parse_daily_plan(response_text: str) -> ViraltestAction:
    """Parse LLM JSON into ViraltestAction with scheduled_actions; fallback to empty (all rest)."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data: Dict[str, Any] = json.loads(text)
        actions_raw = data.get("scheduled_actions", [])
        if not isinstance(actions_raw, list):
            return ViraltestAction(scheduled_actions=[])
        return ViraltestAction(scheduled_actions=actions_raw)
    except (json.JSONDecodeError, Exception):
        return ViraltestAction(scheduled_actions=[])


def _resolve_predefined_topic(raw: Optional[str], obs: Any, hour: int) -> str:
    """Map a model-provided topic to a canonical string from TOPIC_CATEGORIES."""
    if raw and raw.strip():
        key = raw.strip().lower()
        if key in _TOPIC_CANONICAL:
            return _TOPIC_CANONICAL[key]
    for tt in obs.trending_topics or []:
        tl = (tt or "").strip().lower()
        if tl in _TOPIC_CANONICAL:
            return _TOPIC_CANONICAL[tl]
    return PREDEFINED_TOPICS[hour % len(PREDEFINED_TOPICS)]


def sanitize_predefined_topics(action: ViraltestAction, obs: Any) -> ViraltestAction:
    """Force every post topic to match the environment's predefined topic set."""
    out: List[ScheduledAction] = []
    for sa in action.scheduled_actions:
        if sa.action_type == "post":
            out.append(sa.model_copy(update={"topic": _resolve_predefined_topic(sa.topic, obs, sa.hour)}))
        else:
            out.append(sa)
    return ViraltestAction(scheduled_actions=out)


def format_action_str(action: ViraltestAction) -> str:
    """Format daily plan for [STEP] log line."""
    if not action.scheduled_actions:
        return "daily_plan(rest_all)"
    parts = []
    for sa in action.scheduled_actions:
        if sa.action_type == "post":
            tags_str = ",".join(sa.tags) if sa.tags else ""
            parts.append(f"h{sa.hour}:post({sa.content_type},\"{sa.topic}\",[{tags_str}])")
        else:
            parts.append(f"h{sa.hour}:{sa.action_type}()")
    return "daily_plan(" + ";".join(parts) + ")"


_model_exhausted = False


def get_model_daily_plan(
    client: OpenAI, obs: Any, history: List[Dict[str, str]]
) -> ViraltestAction:
    """Call the LLM to get a daily plan. Falls back to rest permanently after an unrecoverable error."""
    global _model_exhausted
    if _model_exhausted:
        return ViraltestAction(scheduled_actions=[])

    user_prompt = format_observation(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-7:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        plan = parse_daily_plan(text) if text else ViraltestAction(scheduled_actions=[])
        return sanitize_predefined_topics(plan, obs)
    except Exception as exc:
        err_str = str(exc)
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        if "402" in err_str or "429" in err_str or "credit" in err_str.lower() or "quota" in err_str.lower():
            _model_exhausted = True
            print("[DEBUG] Token/credit limit reached — falling back to rest for remaining steps", flush=True)
        return ViraltestAction(scheduled_actions=[])


async def run_task(client: OpenAI, task: str) -> None:
    """Run a single task episode (7 daily steps)."""
    global _model_exhausted
    _model_exhausted = False

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[ViraltestEnv] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        if DOCKER_IMAGE:
            env = await ViraltestEnv.from_docker_image(DOCKER_IMAGE)
        else:
            env = ViraltestEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))

        result = await env.reset(task=task)
        history: List[Dict[str, str]] = []

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            if should_force_rest_day(obs):
                action = ViraltestAction(scheduled_actions=[])
            else:
                action = get_model_daily_plan(client, obs, history)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = getattr(result.observation, "error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=format_action_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            history.append({
                "role": "assistant",
                "content": json.dumps({
                    "scheduled_actions": [
                        {
                            "hour": sa.hour,
                            "action_type": sa.action_type,
                            "content_type": sa.content_type,
                            "topic": sa.topic,
                            "tags": sa.tags,
                        }
                        for sa in action.scheduled_actions
                    ]
                }),
            })

            if done:
                score = float(getattr(result.observation, "grader_score", 0) or 0)
                if score == 0:
                    meta = getattr(result.observation, "metadata", {}) or {}
                    score = float(meta.get("grader_score", 0.0))
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "not-needed")
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
