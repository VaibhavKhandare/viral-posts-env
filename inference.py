"""
Viraltest Inference Script v2 — Theme #3.1 World-Modeling Agent
================================================================
The agent receives SPARSE observations and must use discoverable tools to learn
the world (trending topics, competitor activity, tag performance, audience segments).
No peak-hour hints, no fatigue rules, no content-type tips are provided in the prompt.

MANDATORY env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN/OPENAI_API_KEY/API_KEY
Optional: IMAGE_NAME, ALLOW_SHORT_EPISODE, MAX_STEPS

STDOUT FORMAT: [START] [STEP] [END] — match hackathon spec exactly.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from viraltest import ScheduledAction, ViraltestAction, ViraltestEnv
from viraltest.models import ToolCall
from viraltest.server.viraltest_environment import TASK_HORIZON, TOPIC_CATEGORIES

DOCKER_IMAGE = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
BENCHMARK = os.getenv("VIRALTEST_BENCHMARK", "viraltest")

TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]
_ALLOW_SHORT = os.getenv("ALLOW_SHORT_EPISODE", "").lower() in ("1", "true", "yes")
_REQUESTED_MAX = int(os.getenv("MAX_STEPS", str(TASK_HORIZON)))
MAX_STEPS = _REQUESTED_MAX if _ALLOW_SHORT else max(_REQUESTED_MAX, TASK_HORIZON)
TEMPERATURE = 0.7
MAX_TOKENS = 768
SUCCESS_SCORE_THRESHOLD = 0.50

ALL_TOPICS: List[str] = [
    topic for topics in TOPIC_CATEGORIES.values() for topic in topics
]
_TOPIC_CANONICAL: Dict[str, str] = {t.lower(): t for t in ALL_TOPICS}

NEAR_ZERO_ENERGY_THRESHOLD = 0.25

# The agent is NOT told peak hours, fatigue rules, or content type tips.
# It must discover these via the tool catalog.
SYSTEM_PROMPT = textwrap.dedent(f"""\
You are an Instagram content strategy agent. Each step is one full day (24 hours).
You manage a creator account over a {TASK_HORIZON}-day cycle.

You receive a SPARSE observation (energy, followers, last reward, notes echo).
To learn about the world, you MUST use TOOLS before planning your day.

AVAILABLE TOOLS (call via tool_calls before scheduling posts):
- query_trends(niche): Get trending topics and tags for a niche
- query_competitor(competitor_id, window_days): See competitor activity
- query_tag_history(tag): Check your past performance with a tag
- query_audience(segment_id): Learn audience segment preferences
- predict_engagement(scheduled_actions): Simulate engagement without committing
- draft_review(scheduled_actions): Get feedback on a draft plan
- query_creator_pool(): List potential collab partners
- propose_collab(partner_id, content_type, hour): Propose a collaboration

RESPONSE FORMAT (JSON only, no markdown, no prose):
{
  "tool_calls": [
    {"name": "query_trends", "arguments": {"niche": "tech"}},
    {"name": "query_competitor", "arguments": {"competitor_id": "niche_expert", "window_days": 7}}
  ],
  "scheduled_actions": [
    {"hour": 10, "action_type": "create_content"},
    {"hour": 12, "action_type": "post", "content_type": "reel", "topic": "AI tools", "tags": ["ai", "coding"], "intent": "watch_bait"},
    {"hour": 18, "action_type": "post", "content_type": "carousel", "topic": "startup life", "tags": ["startup", "growth"], "intent": "save_bait"}
  ],
  "notes": "Day 3: tech niche trending up. Competitor Alpha posted at 10am. Avoiding overlap."
}

RULES:
- hour: 0-23
- action_type: "post" or "create_content"
- For posts: content_type (reel|story|carousel|text_post), topic, tags (max 5), and intent are required
- intent: what signal you optimize for (send_bait|save_bait|watch_bait|like_bait)
- Empty scheduled_actions = rest all day
- Use notes to track hypotheses and observations across days
- Tool calls cost API budget (starts at 100). Use wisely.
- Max 2 collaborations per full episode

Think strategically: use tools to discover what works, then exploit what you learn.""")


def should_force_rest_day(obs: Any) -> bool:
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


def log_end(
    success: bool, steps: int, score: float, rewards: List[float],
    headline: Optional[Any] = None,
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    head_str = ""
    if headline is not None:
        retention = headline.retention_under_shift
        retention_str = f"{retention:.2f}" if retention is not None else "n/a"
        head_str = (
            f" vs_baseline_pct={headline.vs_baseline_pct:+.2%} "
            f"score_per_tool={headline.score_per_tool_call:.3f} "
            f"score_per_1k_chars={headline.score_per_1k_chars:.3f} "
            f"retention_under_shift={retention_str}"
        )
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}{head_str}",
        flush=True,
    )


def format_observation(obs: Any) -> str:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_name = days[obs.day_of_week] if 0 <= obs.day_of_week < 7 else "?"

    notes_echo = getattr(obs, "agent_notes", None) or "none"
    budget = getattr(obs, "api_budget_remaining", 100)
    burnout = getattr(obs, "burnout_risk", 0.0)

    tool_results_str = ""
    for tr in getattr(obs, "tool_results", []):
        if tr.success:
            tool_results_str += f"  {tr.name}: {json.dumps(tr.data)[:200]}\n"
        else:
            tool_results_str += f"  {tr.name}: ERROR - {tr.error}\n"

    coach = getattr(obs, "coach_feedback", None)
    coach_str = ""
    if coach:
        coach_str = f"Coach: delta={coach.get('delta', 0):.3f}, suggestion={coach.get('suggestion', '')}\n"

    judge = getattr(obs, "judge_report", None)
    judge_str = ""
    if judge:
        judge_str = (
            f"Judge: compliance={judge.policy_compliance:.2f} risk={judge.sustainability_risk:.2f} "
            f"strategy={judge.strategic_quality:.2f} | {judge.explanation}\n"
        )

    signals = getattr(obs, "engagement_signals", None)
    signals_str = ""
    if signals:
        signals_str = (
            f"Signals: watch={signals.watch_time:.3f} sends={signals.sends_per_reach:.3f} "
            f"saves={signals.saves:.3f} likes={signals.likes_per_reach:.3f}\n"
        )

    return textwrap.dedent(f"""\
Day: {day_name} (day_of_week={obs.day_of_week}) | days_elapsed={obs.days_elapsed}
Energy: {obs.creator_energy:.2f} | Burnout risk: {burnout:.2f} | Followers: {obs.follower_count}
Engagement rate: {obs.engagement_rate:.3f} | Content queue: {obs.content_queue_size}
API budget remaining: {budget}
{signals_str}{coach_str}{judge_str}Tool results from last step:
{tool_results_str if tool_results_str else '  (none)\n'}Your notes from last step: {notes_echo}
Plan your tool calls and actions for today:""")


def parse_daily_plan(response_text: str) -> ViraltestAction:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data: Dict[str, Any] = json.loads(text)

        tool_calls = []
        for tc in data.get("tool_calls", []):
            if isinstance(tc, dict) and "name" in tc:
                tool_calls.append(ToolCall(name=tc["name"], arguments=tc.get("arguments", {})))

        actions_raw = data.get("scheduled_actions", [])
        scheduled = []
        if isinstance(actions_raw, list):
            for a in actions_raw:
                if isinstance(a, dict):
                    scheduled.append(a)

        notes = data.get("notes")

        return ViraltestAction(
            tool_calls=tool_calls,
            scheduled_actions=scheduled,
            notes=notes,
        )
    except (json.JSONDecodeError, Exception):
        return ViraltestAction(scheduled_actions=[])


def _resolve_predefined_topic(raw: Optional[str], obs: Any, hour: int) -> str:
    if raw and raw.strip():
        key = raw.strip().lower()
        if key in _TOPIC_CANONICAL:
            return _TOPIC_CANONICAL[key]
    for tt in getattr(obs, "trending_topics", []) or []:
        tl = (tt or "").strip().lower()
        if tl in _TOPIC_CANONICAL:
            return _TOPIC_CANONICAL[tl]
    return ALL_TOPICS[hour % len(ALL_TOPICS)]


def sanitize_predefined_topics(action: ViraltestAction, obs: Any) -> ViraltestAction:
    out = []
    for sa in action.scheduled_actions:
        if sa.action_type == "post":
            out.append(sa.model_copy(update={"topic": _resolve_predefined_topic(sa.topic, obs, sa.hour)}))
        else:
            out.append(sa)
    return ViraltestAction(
        tool_calls=action.tool_calls,
        scheduled_actions=out,
        collab=action.collab,
        notes=action.notes,
    )


def format_action_str(action: ViraltestAction) -> str:
    parts = []
    if action.tool_calls:
        tools_str = ",".join(tc.name for tc in action.tool_calls)
        parts.append(f"tools({tools_str})")
    if not action.scheduled_actions:
        parts.append("rest_all")
    else:
        for sa in action.scheduled_actions:
            if sa.action_type == "post":
                tags_str = ",".join(sa.tags) if sa.tags else ""
                parts.append(f"h{sa.hour}:post({sa.content_type},\"{sa.topic}\",[{tags_str}],{sa.intent or 'none'})")
            else:
                parts.append(f"h{sa.hour}:{sa.action_type}()")
    return "daily_plan(" + ";".join(parts) + ")"


_model_exhausted = False


def get_model_daily_plan(
    client: OpenAI, obs: Any, history: List[Dict[str, str]]
) -> ViraltestAction:
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
            print("[DEBUG] Token/credit limit reached — resting remaining steps", flush=True)
        return ViraltestAction(scheduled_actions=[])


async def run_task(client: OpenAI, task: str) -> None:
    global _model_exhausted
    _model_exhausted = False

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[ViraltestEnv] = None
    headline: Optional[Any] = None

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
                action = ViraltestAction(scheduled_actions=[], notes="Low energy — forced rest day.")
            else:
                action = get_model_daily_plan(client, obs, history)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = getattr(result.observation, "error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=format_action_str(action), reward=reward, done=done, error=error)

            history.append({
                "role": "assistant",
                "content": json.dumps({
                    "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in action.tool_calls],
                    "scheduled_actions": [
                        {
                            "hour": sa.hour, "action_type": sa.action_type,
                            "content_type": sa.content_type, "topic": sa.topic,
                            "tags": sa.tags, "intent": sa.intent,
                        }
                        for sa in action.scheduled_actions
                    ],
                    "notes": action.notes,
                }),
            })

            if done:
                score = float(getattr(result.observation, "grader_score", 0) or 0)
                if score == 0:
                    meta = getattr(result.observation, "metadata", {}) or {}
                    score = float(meta.get("grader_score", 0.0))
                headline = getattr(result.observation, "headline_metrics", None)
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards, headline=headline)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "not-needed")
    for task in TASKS:
        await run_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
