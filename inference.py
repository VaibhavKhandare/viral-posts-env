"""
Viraltest Inference Script v2 — Theme #3.1 World-Modeling Agent
================================================================
The agent receives SPARSE observations and must use discoverable tools to learn
the world (trending topics, competitor activity, tag performance, audience segments).
No peak-hour hints, no fatigue rules, no content-type tips are provided in the prompt.

LLM ENDPOINTS (auto-detected):
  - Local (free): API_BASE_URL=http://0.0.0.0:1337/v1 — HF_TOKEN optional, model auto-discovered
  - HF Router:    API_BASE_URL=https://router.huggingface.co/v1 — HF_TOKEN required
  - OpenAI:       API_BASE_URL=https://api.openai.com/v1 — OPENAI_API_KEY required

ENV VARS:
  Required (only for hosted): HF_TOKEN | OPENAI_API_KEY | API_KEY
  Optional: API_BASE_URL, MODEL_NAME, IMAGE_NAME, ALLOW_SHORT_EPISODE, MAX_STEPS

STDOUT FORMAT: [START] [STEP] [END] — match hackathon spec exactly.
"""

import asyncio
import json
import os
import textwrap
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from viraltest import ScheduledAction, ViraltestAction, ViraltestEnv
from viraltest.models import ToolCall
from viraltest.server.viraltest_environment import TASK_HORIZON, TOPIC_CATEGORIES

DOCKER_IMAGE = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL") or "http://0.0.0.0:1337/v1"
BENCHMARK = os.getenv("VIRALTEST_BENCHMARK", "viraltest")

TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]
_ALLOW_SHORT = os.getenv("ALLOW_SHORT_EPISODE", "").lower() in ("1", "true", "yes")
_REQUESTED_MAX = int(os.getenv("MAX_STEPS", str(TASK_HORIZON)))
MAX_STEPS = _REQUESTED_MAX if _ALLOW_SHORT else max(_REQUESTED_MAX, TASK_HORIZON)
TEMPERATURE = 0.7
MAX_TOKENS = 768
SUCCESS_SCORE_THRESHOLD = 0.1
LOCAL_HOSTS = {"localhost", "0.0.0.0", "127.0.0.1"}

ALL_TOPICS: List[str] = [
    topic for topics in TOPIC_CATEGORIES.values() for topic in topics
]
_TOPIC_CANONICAL: Dict[str, str] = {t.lower(): t for t in ALL_TOPICS}

NEAR_ZERO_ENERGY_THRESHOLD = 0.25

# The agent is NOT told peak hours, fatigue rules, or content type tips.
# It must discover these via the tool catalog.
SYSTEM_PROMPT = textwrap.dedent("""\
You are an Instagram content strategy agent. Each step is one full day (24 hours).
You manage a creator account over a 30-day monthly cycle.

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
  "replies": [{"post_hour": 12, "reply_hour": 13}],
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
- Max 2 collaborations per month
- Reply within 90 minutes of a post for reach bonus

Think strategically: use tools to discover what works, then exploit what you learn.""")


def is_local_endpoint(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in LOCAL_HOSTS


def discover_model_name(base_url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=2) as resp:
            data = json.loads(resp.read())
        models = data.get("data") or []
        return models[0].get("id") if models else None
    except Exception:
        return None


def resolve_model_name(base_url: str) -> str:
    explicit = os.getenv("MODEL_NAME")
    if explicit:
        return explicit
    if is_local_endpoint(base_url):
        discovered = discover_model_name(base_url)
        if discovered:
            return discovered
    return "Qwen/Qwen2.5-7B-Instruct"


def make_client(base_url: Optional[str] = None) -> OpenAI:
    url = base_url or API_BASE_URL
    if is_local_endpoint(url):
        api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "local"
    else:
        api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Hosted endpoint {url} requires HF_TOKEN / OPENAI_API_KEY / API_KEY"
            )
    return OpenAI(base_url=url, api_key=api_key)


MODEL_NAME = resolve_model_name(API_BASE_URL)


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


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
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
{signals_str}{coach_str}Tool results from last step:
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

        replies_raw = data.get("replies", [])
        notes = data.get("notes")

        return ViraltestAction(
            tool_calls=tool_calls,
            scheduled_actions=scheduled,
            replies=replies_raw if isinstance(replies_raw, list) else [],
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
        replies=action.replies,
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


_BASELINE_TAGS = ["lifestyle", "creator", "instagram"]


def baseline_daily_plan(obs: Any) -> ViraltestAction:
    """No-tool, no-notes baseline: post one carousel at noon. Untrained comparator per PDF page 3."""
    topic = ALL_TOPICS[(getattr(obs, "days_elapsed", 0)) % len(ALL_TOPICS)]
    return ViraltestAction(
        scheduled_actions=[ScheduledAction(
            hour=12, action_type="post", content_type="carousel",
            topic=topic, tags=_BASELINE_TAGS, intent="save_bait",
        )],
    )


async def get_model_daily_plan(
    client: OpenAI, obs: Any, history: List[Dict[str, str]], model: str
) -> ViraltestAction:
    """Run the (sync) OpenAI client in a worker thread so the event loop stays
    responsive — otherwise the env-server websocket keepalive times out during
    slow local LLM calls (Gemma on llama.cpp can take 30s+ per turn)."""
    user_prompt = format_observation(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-7:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
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
            raise RuntimeError("token/credit limit reached") from exc
        return ViraltestAction(scheduled_actions=[])


def _signals_dict(obs: Any) -> Dict[str, float]:
    s = getattr(obs, "engagement_signals", None)
    if not s:
        return {"watch_time": 0.0, "sends_per_reach": 0.0, "saves": 0.0, "likes_per_reach": 0.0}
    return {
        "watch_time": float(s.watch_time),
        "sends_per_reach": float(s.sends_per_reach),
        "saves": float(s.saves),
        "likes_per_reach": float(s.likes_per_reach),
    }


async def collect_episode(
    task: str,
    *,
    use_tools: bool = True,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    max_steps: int = MAX_STEPS,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run one episode and return a structured trace.

    Args:
        task: monthly_engage | monthly_strategic | monthly_competitive
        use_tools: True = tool-using LLM agent, False = heuristic baseline (no LLM call)
        client: OpenAI client (created via make_client() if None and use_tools=True)
        model: model name (resolved via resolve_model_name() if None)

    Returns: {task, model, use_tools, steps[{step,action,reward,done,signals,energy,
              followers,grader_score?}], rewards[], grader_score, success}
    """
    arm_label = "agent" if use_tools else "baseline"
    if use_tools and client is None:
        client = make_client()
    resolved_model = model or (MODEL_NAME if use_tools else "baseline_heuristic")

    rewards: List[float] = []
    steps: List[Dict[str, Any]] = []
    score = 0.0
    success = False
    env: Optional[ViraltestEnv] = None

    if not quiet:
        log_start(task=task, env=BENCHMARK, model=f"{resolved_model}({arm_label})")

    try:
        if DOCKER_IMAGE:
            env = await ViraltestEnv.from_docker_image(DOCKER_IMAGE)
        else:
            env = ViraltestEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))

        result = await env.reset(task=task)
        history: List[Dict[str, str]] = []

        for step in range(1, max_steps + 1):
            if result.done:
                break

            obs = result.observation
            if should_force_rest_day(obs):
                action = ViraltestAction(scheduled_actions=[], notes="Low energy — forced rest day.")
            elif use_tools:
                action = await get_model_daily_plan(client, obs, history, resolved_model)
            else:
                action = baseline_daily_plan(obs)

            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = getattr(result.observation, "error", None)
            rewards.append(reward)

            if not quiet:
                log_step(step=step, action=format_action_str(action), reward=reward, done=done, error=error)

            steps.append({
                "step": step,
                "action": format_action_str(action),
                "reward": reward,
                "done": done,
                "error": error,
                "signals": _signals_dict(result.observation),
                "energy": float(getattr(result.observation, "creator_energy", 0.0)),
                "followers": int(getattr(result.observation, "follower_count", 0)),
                "burnout_risk": float(getattr(result.observation, "burnout_risk", 0.0)),
                "rubric_scores": dict(getattr(result.observation, "rubric_scores", {}) or {}),
            })

            if use_tools:
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
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        if not quiet:
            log_end(success=success, steps=len(steps), score=score, rewards=rewards)

    final_obs = getattr(result, "observation", None) if result else None
    rubric_scores = dict(getattr(final_obs, "rubric_scores", {}) or {}) if final_obs else {}
    rubric_evidence = dict(getattr(final_obs, "rubric_evidence", {}) or {}) if final_obs else {}

    return {
        "task": task,
        "arm": arm_label,
        "model": resolved_model,
        "use_tools": use_tools,
        "steps": steps,
        "rewards": rewards,
        "grader_score": score,
        "rubric_scores": rubric_scores,
        "rubric_evidence": rubric_evidence,
        "success": success,
    }


async def main() -> None:
    client = make_client()
    for task in TASKS:
        await collect_episode(task, use_tools=True, client=client, model=MODEL_NAME)


if __name__ == "__main__":
    asyncio.run(main())
