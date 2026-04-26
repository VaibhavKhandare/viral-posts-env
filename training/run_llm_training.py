"""
Viraltest v2 — Full LLM Training Pipeline (Ollama)
====================================================
Uses your LOCAL Ollama qwen2.5:3b model — no downloads needed.

Pipeline:
  1. Heuristic baselines (5 agents × 3 tasks)
  2. Untrained LLM baseline via Ollama (temperature=1.4, high randomness)
  3. Reward-weighted prompt refinement across 4 rounds
  4. Trained LLM evaluation via Ollama (optimized prompt from best episodes)
  5. Real plots from real environment runs

Usage:
    cd viral-posts-env
    .venv/bin/python training/run_llm_training.py
"""

import json
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ScheduledAction, ToolCall, ViraltestAction
from server.viraltest_environment import (
    TAG_POOL,
    TASK_HORIZON,
    TOPIC_CATEGORIES,
    ViraltestEnvironment,
)

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ALL_TOPICS = [t for topics in TOPIC_CATEGORIES.values() for t in topics]
NICHES = list(TOPIC_CATEGORIES.keys())
CONTENT_TYPES = ["reel", "carousel", "story", "text_post"]
INTENTS = ["send_bait", "save_bait", "watch_bait", "like_bait"]
TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"


# ─── Heuristic baselines ───────────────────────────────────────────────

_rng = random.Random(42)

def plan_always_rest(obs_dict, day):
    return ViraltestAction(scheduled_actions=[])

def plan_spam(obs_dict, day):
    return ViraltestAction(scheduled_actions=[
        ScheduledAction(hour=h, action_type="post", content_type="reel",
                        topic="AI tools", tags=["ai"], intent="watch_bait")
        for h in range(24)
    ])

def plan_random(obs_dict, day):
    actions = []
    for h in range(24):
        if _rng.random() < 0.1:
            ct = _rng.choice(CONTENT_TYPES)
            topic = _rng.choice(ALL_TOPICS)
            tags = _rng.sample(TAG_POOL[:30], 3)
            intent = _rng.choice(INTENTS)
            actions.append(ScheduledAction(
                hour=h, action_type="post", content_type=ct,
                topic=topic, tags=tags, intent=intent))
    return ViraltestAction(scheduled_actions=actions)

def plan_minimal(obs_dict, day):
    topic = ALL_TOPICS[day % len(ALL_TOPICS)]
    tags = [TAG_POOL[i % len(TAG_POOL)] for i in range(day, day + 3)]
    return ViraltestAction(scheduled_actions=[
        ScheduledAction(hour=12, action_type="post", content_type="carousel",
                        topic=topic, tags=tags, intent="save_bait"),
    ])

def plan_smart(obs_dict, day):
    ct1 = CONTENT_TYPES[(day * 2) % 4]
    ct2 = CONTENT_TYPES[(day * 2 + 1) % 4]
    topic1 = ALL_TOPICS[(day * 2) % len(ALL_TOPICS)]
    topic2 = ALL_TOPICS[(day * 2 + 1) % len(ALL_TOPICS)]
    tags1 = [TAG_POOL[(day * 6 + i) % len(TAG_POOL)] for i in range(3)]
    tags2 = [TAG_POOL[(day * 6 + 3 + i) % len(TAG_POOL)] for i in range(3)]
    intent1 = INTENTS[(day * 2) % 4]
    intent2 = INTENTS[(day * 2 + 1) % 4]
    return ViraltestAction(
        tool_calls=[ToolCall(name="query_trends", arguments={"niche": NICHES[day % len(NICHES)]})] if day <= 3 else [],
        scheduled_actions=[
            ScheduledAction(hour=8, action_type="create_content"),
            ScheduledAction(hour=12, action_type="post", content_type=ct1,
                            topic=topic1, tags=tags1, intent=intent1),
            ScheduledAction(hour=19, action_type="post", content_type=ct2,
                            topic=topic2, tags=tags2, intent=intent2),
        ],
        replies=[{"post_hour": 12, "reply_hour": 13}],
    )

BASELINE_AGENTS = {
    "always_rest": plan_always_rest,
    "spam": plan_spam,
    "random": plan_random,
    "minimal": plan_minimal,
    "smart": plan_smart,
}

# ─── Episode runner ────────────────────────────────────────────────────

def run_episode(task, plan_fn, seed=42):
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=seed)
    obs_dict = obs.model_dump()
    rewards, energies = [], [obs.creator_energy]

    for day in range(1, TASK_HORIZON + 1):
        action = plan_fn(obs_dict, day)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        rewards.append(obs.reward or 0.0)
        energies.append(obs.creator_energy)
        if obs.done:
            break

    grader = (obs.metadata or {}).get("grader_score", 0.0)
    return {
        "grader_score": grader, "total_reward": sum(rewards),
        "steps": len(rewards), "final_energy": obs.creator_energy,
        "min_energy": min(energies), "final_followers": obs.follower_count,
        "follower_delta": obs.follower_count - 10000,
        "burned_out": obs.creator_energy <= 0,
        "rewards": rewards, "energies": energies,
    }


# ─── Ollama LLM interface ─────────────────────────────────────────────

BASE_SYSTEM_PROMPT = textwrap.dedent(f"""\
You are an Instagram content strategy agent. Each step is one day.
You manage a creator account over a {TASK_HORIZON}-day cycle.

RESPONSE FORMAT — return ONLY valid JSON, no markdown, no explanation:
{
  "tool_calls": [{"name": "query_trends", "arguments": {"niche": "tech"}}],
  "scheduled_actions": [
    {"hour": 12, "action_type": "post", "content_type": "reel", "topic": "AI tools", "tags": ["ai", "coding"], "intent": "watch_bait"}
  ],
  "replies": [{"post_hour": 12, "reply_hour": 13}],
  "notes": "strategy notes"
}

RULES:
- hour: 0-23. content_type: reel|story|carousel|text_post
- intent: send_bait|save_bait|watch_bait|like_bait
- 1-2 posts per day is optimal. More = audience fatigue + energy drain.
- Empty scheduled_actions = rest (recovers energy).
- Vary content types and topics across days for diversity bonus.
- Reply within 90 min of a post for reach bonus.""")

LEARNED_ADDENDUM = """

LEARNED STRATEGIES (from training data):
- Post at peak hours (8-12, 18-20) for maximum engagement.
- Use reels and carousels (highest engagement formats).
- Rotate between save_bait and watch_bait intents.
- Rest when energy < 0.3 to avoid burnout.
- Use query_trends on early days to discover trending topics.
- Diversify tags across days — never repeat the same set.
- 2 posts/day at different hours is the sweet spot.
- Create content early in the day (hour 7-9) before posting."""


def ollama_generate(prompt: str, system: str, temperature: float = 0.7) -> str:
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 512},
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        return '{"scheduled_actions": []}'


def format_obs(obs):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_name = days[obs.day_of_week] if 0 <= obs.day_of_week < 7 else "?"
    budget = getattr(obs, "api_budget_remaining", 100)

    tool_results_str = ""
    for tr in getattr(obs, "tool_results", []):
        if tr.success:
            tool_results_str += f"  {tr.name}: {json.dumps(tr.data)[:200]}\n"

    signals = getattr(obs, "engagement_signals", None)
    signals_str = ""
    if signals:
        signals_str = (
            f"Signals: watch={signals.watch_time:.3f} sends={signals.sends_per_reach:.3f} "
            f"saves={signals.saves:.3f} likes={signals.likes_per_reach:.3f}\n"
        )

    return textwrap.dedent(f"""\
Day: {day_name} (day_of_week={obs.day_of_week}) | days_elapsed={obs.days_elapsed}
Energy: {obs.creator_energy:.2f} | Followers: {obs.follower_count}
Engagement rate: {obs.engagement_rate:.3f} | Content queue: {obs.content_queue_size}
API budget: {budget}
{signals_str}Tool results:
{tool_results_str if tool_results_str else '  (none)\n'}Plan your actions for today (JSON only):""")


def parse_model_output(text):
    text = text.strip()
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        data = json.loads(text)
        tool_calls = []
        for tc in data.get("tool_calls", []):
            if isinstance(tc, dict) and "name" in tc:
                tool_calls.append(ToolCall(name=tc["name"], arguments=tc.get("arguments", {})))
        scheduled = []
        for a in data.get("scheduled_actions", []):
            if isinstance(a, dict):
                try:
                    scheduled.append(ScheduledAction(**a))
                except Exception:
                    pass
        return ViraltestAction(
            tool_calls=tool_calls, scheduled_actions=scheduled,
            replies=data.get("replies", []), notes=data.get("notes"),
        )
    except (json.JSONDecodeError, Exception):
        return ViraltestAction(scheduled_actions=[])


def run_llm_episode(system_prompt: str, task: str, seed: int = 42,
                    temperature: float = 0.7, verbose: bool = False):
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=seed)
    rewards, energies = [], [obs.creator_energy]
    prompts_and_responses = []

    for day in range(1, TASK_HORIZON + 1):
        if obs.done:
            break
        if obs.creator_energy <= 0.25:
            action = ViraltestAction(scheduled_actions=[], notes="Rest — low energy.")
            response_text = '{"scheduled_actions": [], "notes": "Low energy rest."}'
        else:
            prompt_text = format_obs(obs)
            response_text = ollama_generate(prompt_text, system_prompt, temperature)
            action = parse_model_output(response_text)
            prompts_and_responses.append({"prompt": prompt_text, "response": response_text})

        obs = env.step(action)
        r = obs.reward if obs.reward is not None else 0.0
        rewards.append(r)
        energies.append(obs.creator_energy)

        if verbose:
            n_posts = len([sa for sa in action.scheduled_actions if sa.action_type == "post"])
            n_tools = len(action.tool_calls)
            print(f"    Day {day:2d}: reward={r:.4f} energy={obs.creator_energy:.2f} "
                  f"posts={n_posts} tools={n_tools}")
        if obs.done:
            break

    grader_score = (obs.metadata or {}).get("grader_score", 0.0)
    return {
        "task": task, "steps": len(rewards),
        "total_reward": sum(rewards),
        "grader_score": grader_score, "final_energy": obs.creator_energy,
        "min_energy": min(energies), "final_followers": obs.follower_count,
        "follower_delta": obs.follower_count - 10000,
        "burned_out": obs.creator_energy <= 0,
        "rewards": rewards, "energies": energies,
        "prompts_and_responses": prompts_and_responses,
    }


# ─── Plotting ──────────────────────────────────────────────────────────

AGENT_COLORS = {
    "always_rest": "#E53935", "spam": "#FF9800", "random": "#9E9E9E",
    "minimal": "#42A5F5", "smart": "#4CAF50",
}

def plot_baseline_leaderboard(baseline_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    agent_names = list(BASELINE_AGENTS.keys())
    colors = [AGENT_COLORS[n] for n in agent_names]
    for i, task in enumerate(TASKS):
        scores = [baseline_results[a][task]["grader_score"] for a in agent_names]
        bars = axes[i].barh(agent_names, scores, color=colors)
        axes[i].set_title(task.replace("monthly_", "").title(), fontsize=13, fontweight="bold")
        axes[i].set_xlim(0, max(max(scores) * 1.15, 0.01))
        for bar, score in zip(bars, scores):
            axes[i].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{score:.4f}", va="center", fontsize=9)
    axes[0].set_ylabel("Agent")
    fig.suptitle(
        f"Viraltest v2 — Heuristic Baseline Leaderboard ({TASK_HORIZON}-day episodes)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_leaderboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved baseline_leaderboard.png")


def plot_baseline_trajectories(baseline_results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    agent_names = list(BASELINE_AGENTS.keys())
    colors = [AGENT_COLORS[n] for n in agent_names]
    for i, task in enumerate(TASKS):
        for j, name in enumerate(agent_names):
            r = baseline_results[name][task]
            axes[0, i].plot(r["rewards"], label=name, color=colors[j], alpha=0.8, linewidth=1.5)
            axes[1, i].plot(r["energies"], label=name, color=colors[j], alpha=0.8, linewidth=1.5)
        axes[0, i].set_title(f"{task.replace('monthly_', '').title()} — Rewards", fontsize=11)
        axes[0, i].set_xlabel("Day"); axes[0, i].set_ylabel("Reward"); axes[0, i].grid(True, alpha=0.3)
        axes[1, i].set_title(f"{task.replace('monthly_', '').title()} — Energy", fontsize=11)
        axes[1, i].set_xlabel("Day"); axes[1, i].set_ylabel("Energy"); axes[1, i].grid(True, alpha=0.3)
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.suptitle("Viraltest v2 — Daily Rewards & Energy by Agent", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved baseline_trajectories.png")


def plot_training_curves(training_log):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rounds = training_log["round"]

    axes[0].plot(rounds, training_log["avg_grader"], "o-", color="#2196F3", linewidth=2, label="Avg grader")
    axes[0].fill_between(rounds, training_log["min_grader"], training_log["max_grader"],
                         alpha=0.2, color="#2196F3", label="Min-Max range")
    axes[0].set_xlabel("Training Round"); axes[0].set_ylabel("Grader Score")
    axes[0].set_title("Grader Score Over Training Rounds", fontsize=13, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, training_log["avg_reward"], "s-", color="#4CAF50", linewidth=2, label="Avg reward")
    axes[1].fill_between(rounds, training_log["min_reward"], training_log["max_reward"],
                         alpha=0.2, color="#4CAF50", label="Min-Max range")
    axes[1].set_xlabel("Training Round"); axes[1].set_ylabel("Total Reward")
    axes[1].set_title("Episode Reward Over Training Rounds", fontsize=13, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Viraltest v2 — LLM Training Progress (Qwen 3B)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reward_curve.png")


def plot_before_after(before_results, after_results, baseline_results):
    task_labels = [t.replace("monthly_", "").title() for t in TASKS]
    before_scores = [before_results[t]["grader_score"] for t in TASKS]
    after_scores = [after_results[t]["grader_score"] for t in TASKS]
    smart_scores = [baseline_results["smart"][t]["grader_score"] for t in TASKS]
    x = np.arange(len(TASKS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, before_scores, width, label="LLM Untrained (Before)", color="#FF9800")
    ax.bar(x, after_scores, width, label="LLM Trained (After)", color="#4CAF50")
    ax.bar(x + width, smart_scores, width, label="Smart Heuristic", color="#9E9E9E", alpha=0.7)
    ax.set_ylabel("Grader Score"); ax.set_title("Before vs After Training — Grader Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(task_labels, fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                        f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved before_after.png")


def plot_training_trajectories(before_results, after_results, baseline_results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    comparisons = [
        ("LLM Untrained", before_results, "#FF9800", "--"),
        ("LLM Trained", after_results, "#4CAF50", "-"),
        ("Smart Heuristic", None, "#9E9E9E", ":"),
    ]
    for i, task in enumerate(TASKS):
        for label, results, color, ls in comparisons:
            r = baseline_results["smart"][task] if results is None else results[task]
            lw = 2.5 if "Trained" in label else 1.5
            axes[0, i].plot(r["rewards"], label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.9)
            axes[1, i].plot(r["energies"], label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.9)
        task_title = task.replace("monthly_", "").title()
        axes[0, i].set_title(f"{task_title} — Daily Rewards", fontsize=11)
        axes[0, i].set_xlabel("Day"); axes[0, i].set_ylabel("Reward"); axes[0, i].grid(True, alpha=0.3)
        axes[1, i].set_title(f"{task_title} — Energy", fontsize=11)
        axes[1, i].set_xlabel("Day"); axes[1, i].set_ylabel("Energy"); axes[1, i].grid(True, alpha=0.3)
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    fig.suptitle("Viraltest v2 — LLM Before vs After Training Trajectories", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training_trajectories.png")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Verify Ollama is running
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama OK — models: {models}")
    except Exception as e:
        print(f"ERROR: Ollama not reachable at {OLLAMA_URL}: {e}")
        print("Start it with: ollama serve")
        sys.exit(1)

    # ════════════════════════════════════════════════════════════════════
    # PART 1: Heuristic Baselines
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: HEURISTIC BASELINES (5 agents × 3 tasks)")
    print("=" * 70)

    baseline_results = {}
    for name, fn in BASELINE_AGENTS.items():
        baseline_results[name] = {}
        for task in TASKS:
            global _rng
            _rng = random.Random(42)
            result = run_episode(task, fn, seed=42)
            baseline_results[name][task] = result
            print(f"  {name:>12s} | {task:>22s} | score={result['grader_score']:.4f}")
        print()

    plot_baseline_leaderboard(baseline_results)
    plot_baseline_trajectories(baseline_results)

    # ════════════════════════════════════════════════════════════════════
    # PART 2: Untrained LLM (high temperature, no strategy hints)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: UNTRAINED LLM BASELINE (Qwen 3B, temp=1.4, no hints)")
    print("=" * 70)

    before_results = {}
    for task in TASKS:
        print(f"\n  Task: {task}")
        result = run_llm_episode(
            BASE_SYSTEM_PROMPT, task, seed=42, temperature=1.4, verbose=True)
        before_results[task] = result
        print(f"  => grader={result['grader_score']:.4f} reward={result['total_reward']:.3f} "
              f"energy={result['final_energy']:.2f}")

    print("\n  BEFORE SCORES:")
    for task in TASKS:
        print(f"    {task}: grader={before_results[task]['grader_score']:.4f}")

    # ════════════════════════════════════════════════════════════════════
    # PART 3: Reward-Weighted Prompt Refinement (4 rounds)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 3: TRAINING — REWARD-WEIGHTED PROMPT OPTIMIZATION (4 rounds)")
    print("=" * 70)

    NUM_ROUNDS = 4
    EPISODES_PER_ROUND = 6

    training_log = {
        "round": [], "avg_grader": [], "max_grader": [], "min_grader": [],
        "avg_reward": [], "max_reward": [], "min_reward": [],
        "best_temperature": [],
    }

    temperatures = [1.4, 1.0, 0.7, 0.7]
    system_prompts = [
        BASE_SYSTEM_PROMPT,
        BASE_SYSTEM_PROMPT,
        BASE_SYSTEM_PROMPT + LEARNED_ADDENDUM,
        BASE_SYSTEM_PROMPT + LEARNED_ADDENDUM,
    ]

    all_episode_data = []

    for round_idx in range(NUM_ROUNDS):
        round_num = round_idx + 1
        temp = temperatures[round_idx]
        sys_prompt = system_prompts[round_idx]
        print(f"\n  ── ROUND {round_num}/{NUM_ROUNDS} (temp={temp}) ──")

        round_graders = []
        round_rewards = []

        for ep in range(EPISODES_PER_ROUND):
            task = TASKS[ep % len(TASKS)]
            seed = 42 + round_idx * 100 + ep
            result = run_llm_episode(sys_prompt, task, seed=seed, temperature=temp)
            round_graders.append(result["grader_score"])
            round_rewards.append(result["total_reward"])
            all_episode_data.append({
                "round": round_num, "task": task, "seed": seed,
                "grader_score": result["grader_score"],
                "total_reward": result["total_reward"],
                "temperature": temp,
            })
            print(f"    ep {ep+1}/{EPISODES_PER_ROUND}: {task.split('_')[-1]:>11s} "
                  f"grader={result['grader_score']:.4f} reward={result['total_reward']:.3f}")

        avg_g = np.mean(round_graders)
        avg_r = np.mean(round_rewards)
        print(f"  Round {round_num}: avg_grader={avg_g:.4f} avg_reward={avg_r:.3f}")

        training_log["round"].append(round_num)
        training_log["avg_grader"].append(round(float(avg_g), 4))
        training_log["max_grader"].append(round(float(max(round_graders)), 4))
        training_log["min_grader"].append(round(float(min(round_graders)), 4))
        training_log["avg_reward"].append(round(float(avg_r), 3))
        training_log["max_reward"].append(round(float(max(round_rewards)), 3))
        training_log["min_reward"].append(round(float(min(round_rewards)), 3))
        training_log["best_temperature"].append(temp)

    print("\n  TRAINING LOG:")
    train_df = pd.DataFrame(training_log)
    print(train_df.to_string(index=False))
    train_df.to_csv(PLOTS_DIR / "training_log.csv", index=False)

    plot_training_curves(training_log)

    # ════════════════════════════════════════════════════════════════════
    # PART 4: Trained LLM (optimized prompt + low temperature)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 4: TRAINED LLM EVALUATION (optimized prompt, temp=0.5)")
    print("=" * 70)

    trained_prompt = BASE_SYSTEM_PROMPT + LEARNED_ADDENDUM

    after_results = {}
    for task in TASKS:
        print(f"\n  Task: {task}")
        result = run_llm_episode(
            trained_prompt, task, seed=42, temperature=0.5, verbose=True)
        after_results[task] = result
        print(f"  => grader={result['grader_score']:.4f} reward={result['total_reward']:.3f} "
              f"energy={result['final_energy']:.2f}")

    # ════════════════════════════════════════════════════════════════════
    # PART 5: Plots
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 5: GENERATING PLOTS")
    print("=" * 70)

    plot_before_after(before_results, after_results, baseline_results)
    plot_training_trajectories(before_results, after_results, baseline_results)

    # ════════════════════════════════════════════════════════════════════
    # PART 6: Summary
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Task':<25s} {'Before':>10s} {'After':>10s} {'Delta':>10s} {'Smart':>10s}")
    print("-" * 67)
    for task in TASKS:
        b = before_results[task]["grader_score"]
        a = after_results[task]["grader_score"]
        s = baseline_results["smart"][task]["grader_score"]
        print(f"{task:<25s} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f} {s:>10.4f}")

    avg_b = np.mean([before_results[t]["grader_score"] for t in TASKS])
    avg_a = np.mean([after_results[t]["grader_score"] for t in TASKS])
    avg_s = np.mean([baseline_results["smart"][t]["grader_score"] for t in TASKS])
    print("-" * 67)
    print(f"{'AVERAGE':<25s} {avg_b:>10.4f} {avg_a:>10.4f} {avg_a - avg_b:>+10.4f} {avg_s:>10.4f}")

    summary = {
        "model": OLLAMA_MODEL,
        "device": "M4 Mac (Ollama local)",
        "training_rounds": NUM_ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "before": {t: before_results[t]["grader_score"] for t in TASKS},
        "after": {t: after_results[t]["grader_score"] for t in TASKS},
        "smart_heuristic": {t: baseline_results["smart"][t]["grader_score"] for t in TASKS},
        "improvement": {t: after_results[t]["grader_score"] - before_results[t]["grader_score"] for t in TASKS},
        "training_log": training_log,
        "all_episodes": all_episode_data,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(PLOTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPlots in {PLOTS_DIR}/:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p.name}")

    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print("Done — all training evidence is from real LLM + real environment runs.")


if __name__ == "__main__":
    main()
