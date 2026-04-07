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
    ALLOW_SHORT_EPISODE=1   Allow MAX_STEPS below 168 (final grader score stays 0 if episode never ends).
    MAX_STEPS   Step cap (default 168). Without ALLOW_SHORT_EPISODE, cap is at least 168 so graders run.

STDOUT FORMAT (single space after tag; score two decimals) — match hackathon sample exactly.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from viraltest import ViraltestAction, ViraltestEnv
from viraltest.server.viraltest_environment import TAG_POOL, TASK_HORIZON

DOCKER_IMAGE = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "http://127.0.0.1:1337/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gemma-4-E4B-it-IQ4_XS"
BENCHMARK = os.getenv("VIRALTEST_BENCHMARK", "viraltest")

TASKS = ["weekly_engage", "weekly_strategic", "weekly_competitive"]
_ALLOW_SHORT = os.getenv("ALLOW_SHORT_EPISODE", "").lower() in ("1", "true", "yes")
_REQUESTED_MAX = int(os.getenv("MAX_STEPS", str(TASK_HORIZON)))
MAX_STEPS = _REQUESTED_MAX if _ALLOW_SHORT else max(_REQUESTED_MAX, TASK_HORIZON)
TEMPERATURE = 0.7
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.1

VALID_TAGS_TEXT = ", ".join(TAG_POOL)

SYSTEM_PROMPT = textwrap.dedent(f"""\
You are a social media content strategy agent. Each user message is the current simulation state;
choose the next action. Reply with one JSON object only (no markdown, no prose).

ACTIONS:
1. Post: {{"action_type":"post","content_type":"<reel|story|carousel|text_post>","topic":"<topic>","tags":["tag1","tag2"]}}
2. Rest: {{"action_type":"rest"}}
3. Create content: {{"action_type":"create_content"}}

TAG RULE (enforced by the environment): every tag must be from this pool; unknown tags are removed.
{VALID_TAGS_TEXT}

When action_type is post, content_type and topic are required. Choose each action_type using the
current observation and your prior JSON actions in this episode (message history) to grow followers
as much as possible. If creator energy falls to 0 or below, the episode ends immediately (game over).
Use rest and create_content to manage energy and the content queue while still posting for growth.""")


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
    return textwrap.dedent(f"""\
Local time: {obs.current_hour}:00 | Weekday: {day_name} (day_of_week={obs.day_of_week}, 0=Mon) | days_elapsed={obs.days_elapsed}
Hours since sleep: {obs.hours_since_sleep} | Sleep debt: {obs.sleep_debt:.3f}
Energy: {obs.creator_energy:.2f} | Followers: {obs.follower_count} | Engagement rate: {obs.engagement_rate:.3f}
Posts today: {obs.posts_today} | Hours since last post: {obs.time_since_last_post}
Content queue: {obs.content_queue_size} | Last post type: {obs.last_post_type}
Trending topics: {', '.join(obs.trending_topics)}
Trending tags: {', '.join(obs.trending_tags)}
Your top tags: {top_tags_str}
Niche saturation: {obs.niche_saturation:.2f} | Competitor avg engagement: {obs.competitor_avg_engagement:.3f}
Competitor recent posts:
{comp_str}What is your next action?""")


def parse_action(response_text: str) -> ViraltestAction:
    """Parse LLM JSON into ViraltestAction; invalid output becomes rest."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data: Dict[str, Any] = json.loads(text)
        return ViraltestAction(
            action_type=data.get("action_type", "rest"),
            content_type=data.get("content_type"),
            topic=data.get("topic"),
            tags=data.get("tags"),
        )
    except (json.JSONDecodeError, Exception):
        return ViraltestAction(action_type="rest")


def format_action_str(action: ViraltestAction) -> str:
    """Format action for [STEP] log line."""
    if action.action_type == "post":
        tags_str = ",".join(action.tags) if action.tags else ""
        return f"post({action.content_type},\"{action.topic}\",[{tags_str}])"
    return f"{action.action_type}()"


def get_model_action(
    client: OpenAI, obs: Any, history: List[Dict[str, str]]
) -> ViraltestAction:
    """Call the LLM and parse its response into an action."""
    user_prompt = format_observation(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-48:])
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
        return parse_action(text) if text else ViraltestAction(action_type="rest")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ViraltestAction(action_type="rest")


async def run_task(client: OpenAI, task: str) -> None:
    """Run a single task episode."""
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
            action = get_model_action(client, obs, history)

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
                    "action_type": action.action_type,
                    "content_type": action.content_type,
                    "topic": action.topic,
                    "tags": action.tags,
                }),
            })

            if done:
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
