"""Anti-gaming demonstration (PDF page 2: "is hard to game").

Runs an obviously gameable strategy \u2014 post the SAME content_type with the SAME tag
every hour of every day \u2014 and verifies the env's anti-gaming gates collapse the score.
Writes runs/gameable.jsonl with the same shape as baseline.jsonl / agent.jsonl.

Expected outcome: grader < 0.20 on every task (vs ~0.30-0.55 for an honest baseline).
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viraltest import ScheduledAction, ViraltestAction, ViraltestEnv  # noqa: E402

OUT = ROOT / "runs" / "gameable.jsonl"
TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]


def gameable_action() -> ViraltestAction:
    """Post the same single-tag carousel at every waking hour. Worst-case spam."""
    return ViraltestAction(scheduled_actions=[
        ScheduledAction(hour=h, action_type="post", content_type="carousel",
                        topic="AI tools", tags=["lifestyle"], intent="like_bait")
        for h in range(8, 22)  # 14 posts/day
    ])


async def run(task: str, max_steps: int = 30) -> dict:
    env = ViraltestEnv(base_url="http://localhost:8000")
    rewards: list[float] = []
    score = 0.0
    try:
        result = await env.reset(task=task)
        for step in range(1, max_steps + 1):
            if result.done:
                break
            result = await env.step(gameable_action())
            rewards.append(result.reward or 0.0)
            if result.done:
                score = float(getattr(result.observation, "grader_score", 0) or 0)
                break
    finally:
        await env.close()

    final_obs = getattr(result, "observation", None)
    return {
        "task": task,
        "arm": "gameable",
        "model": "spam_strategy",
        "use_tools": False,
        "rewards": rewards,
        "grader_score": score,
        "rubric_scores": dict(getattr(final_obs, "rubric_scores", {}) or {}),
        "rubric_evidence": dict(getattr(final_obs, "rubric_evidence", {}) or {}),
        "success": False,
        "steps": [],
    }


async def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for task in TASKS:
            r = await run(task)
            f.write(json.dumps(r) + "\n")
            print(
                f"[gameable] {task}: grader={r['grader_score']:.3f}  "
                f"rubrics={ {k: round(v, 3) for k, v in r['rubric_scores'].items()} }"
            )


if __name__ == "__main__":
    asyncio.run(main())
