"""Run baseline (heuristic, no LLM) vs agent (tool-using LLM) episodes; write runs/*.jsonl.

Per the hackathon PDF (page 3): "Compare a trained agent vs a random/untrained baseline".
Baseline = noon carousel post every day, no tools, no notes.
Agent    = inference.py SYSTEM_PROMPT against API_BASE_URL (defaults to local Gemma).

Usage:
    .venv/bin/python scripts/run_baseline_vs_agent.py \\
        --tasks monthly_engage monthly_strategic monthly_competitive \\
        --baseline-episodes 3 --agent-episodes 1 --max-steps 30
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import API_BASE_URL, MODEL_NAME, collect_episode, make_client  # noqa: E402

RUNS_DIR = ROOT / "runs"
DEFAULT_TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]


async def run_arm(arm: str, tasks: list, episodes: int, max_steps: int, out_path: Path) -> None:
    use_tools = arm == "agent"
    client = make_client() if use_tools else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for task in tasks:
            for ep in range(episodes):
                t0 = time.time()
                result = await collect_episode(
                    task,
                    use_tools=use_tools,
                    client=client,
                    model=MODEL_NAME if use_tools else "baseline_heuristic",
                    max_steps=max_steps,
                    quiet=True,
                )
                result["episode_index"] = ep
                result["wall_seconds"] = round(time.time() - t0, 1)
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(
                    f"  [{arm}] {task} ep{ep+1}/{episodes}: "
                    f"reward_sum={sum(result['rewards']):.3f} "
                    f"grader={result['grader_score']:.3f} "
                    f"wall={result['wall_seconds']}s",
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--baseline-episodes", type=int, default=3)
    p.add_argument("--agent-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--arms", nargs="+", default=["baseline", "agent"], choices=["baseline", "agent"])
    p.add_argument("--baseline-out", default=str(RUNS_DIR / "baseline.jsonl"))
    p.add_argument("--agent-out", default=str(RUNS_DIR / "agent.jsonl"))
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    print(f"API_BASE_URL={API_BASE_URL}  MODEL_NAME={MODEL_NAME}")
    print(f"tasks={args.tasks}  max_steps={args.max_steps}")
    if "baseline" in args.arms:
        os.environ["MAX_STEPS"] = str(args.max_steps)
        os.environ["ALLOW_SHORT_EPISODE"] = "1"
        await run_arm("baseline", args.tasks, args.baseline_episodes, args.max_steps, Path(args.baseline_out))
    if "agent" in args.arms:
        await run_arm("agent", args.tasks, args.agent_episodes, args.max_steps, Path(args.agent_out))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
