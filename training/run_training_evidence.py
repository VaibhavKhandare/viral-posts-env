"""
Viraltest v2 — Training Evidence Generator
============================================
Runs locally on any machine (no GPU required).

Two types of training evidence:
1. BASELINE COMPARISON: 5 heuristic agents × 3 tasks = 15 runs
   Proves the environment differentiates strategies.

2. POLICY IMPROVEMENT: Evolutionary search over posting parameters
   Starting from a random policy, optimizes hour, content_type, tags,
   intent, and post count to maximize grader_score.
   Shows measurable improvement in rewards over generations.

Outputs real plots to ../plots/ from real environment runs.
"""

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# ─── Heuristic baselines ───────────────────────────────────────────────

def plan_rest(obs_dict: dict, day: int) -> ViraltestAction:
    return ViraltestAction(scheduled_actions=[])

def plan_spam(obs_dict: dict, day: int) -> ViraltestAction:
    return ViraltestAction(scheduled_actions=[
        ScheduledAction(hour=h, action_type="post", content_type="reel",
                        topic="AI tools", tags=["ai"], intent="watch_bait")
        for h in range(24)
    ])

_baseline_rng = random.Random(42)

def plan_random(obs_dict: dict, day: int) -> ViraltestAction:
    actions = []
    for h in range(24):
        if _baseline_rng.random() < 0.1:
            ct = _baseline_rng.choice(CONTENT_TYPES)
            topic = _baseline_rng.choice(ALL_TOPICS)
            tags = _baseline_rng.sample(TAG_POOL[:30], 3)
            intent = _baseline_rng.choice(INTENTS)
            actions.append(ScheduledAction(
                hour=h, action_type="post", content_type=ct,
                topic=topic, tags=tags, intent=intent))
    return ViraltestAction(scheduled_actions=actions)

def plan_minimal(obs_dict: dict, day: int) -> ViraltestAction:
    topic = ALL_TOPICS[day % len(ALL_TOPICS)]
    tags = [TAG_POOL[i % len(TAG_POOL)] for i in range(day, day + 3)]
    return ViraltestAction(scheduled_actions=[
        ScheduledAction(hour=12, action_type="post", content_type="carousel",
                        topic=topic, tags=tags, intent="save_bait"),
    ])

def plan_smart(obs_dict: dict, day: int) -> ViraltestAction:
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
        notes=f"Day {day}: varied content at peak hours.",
    )

BASELINE_AGENTS = {
    "always_rest": plan_rest,
    "spam": plan_spam,
    "random": plan_random,
    "minimal": plan_minimal,
    "smart": plan_smart,
}

# ─── Episode runner ────────────────────────────────────────────────────

def run_episode(task: str, plan_fn: Callable, seed: int = 42) -> Dict[str, Any]:
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
        "grader_score": grader,
        "total_reward": sum(rewards),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "steps": len(rewards),
        "final_energy": obs.creator_energy,
        "min_energy": min(energies),
        "final_followers": obs.follower_count,
        "follower_delta": obs.follower_count - 10000,
        "burned_out": obs.creator_energy <= 0,
        "rewards": rewards,
        "energies": energies,
    }

# ─── Learnable policy (evolutionary search) ───────────────────────────

@dataclass
class PostingPolicy:
    """Parameterized posting policy that can be optimized."""
    post_hours: List[int] = field(default_factory=lambda: [12])
    content_types: List[str] = field(default_factory=lambda: ["carousel"])
    intents: List[str] = field(default_factory=lambda: ["save_bait"])
    tag_offset: int = 0
    topic_offset: int = 0
    create_hour: Optional[int] = None
    use_reply: bool = False
    use_tools_early: bool = False
    rest_if_low_energy: float = 0.3

    def to_plan_fn(self) -> Callable:
        policy = self
        def plan_fn(obs_dict: dict, day: int) -> ViraltestAction:
            energy = obs_dict.get("creator_energy", 1.0)
            if energy <= policy.rest_if_low_energy:
                return ViraltestAction(scheduled_actions=[], notes="Low energy rest.")

            actions = []
            if policy.create_hour is not None:
                actions.append(ScheduledAction(hour=policy.create_hour, action_type="create_content"))

            for i, hour in enumerate(policy.post_hours):
                ct = policy.content_types[i % len(policy.content_types)]
                intent = policy.intents[i % len(policy.intents)]
                topic_idx = (day * len(policy.post_hours) + i + policy.topic_offset) % len(ALL_TOPICS)
                tag_start = (day * 3 * len(policy.post_hours) + i * 3 + policy.tag_offset) % len(TAG_POOL)
                tags = [TAG_POOL[(tag_start + j) % len(TAG_POOL)] for j in range(3)]
                actions.append(ScheduledAction(
                    hour=hour, action_type="post", content_type=ct,
                    topic=ALL_TOPICS[topic_idx], tags=tags, intent=intent))

            tool_calls = []
            if policy.use_tools_early and day <= 3:
                tool_calls.append(ToolCall(name="query_trends",
                                          arguments={"niche": NICHES[day % len(NICHES)]}))

            replies = []
            if policy.use_reply and policy.post_hours:
                first_post = policy.post_hours[0]
                if first_post < 23:
                    replies = [{"post_hour": first_post, "reply_hour": first_post + 1}]

            return ViraltestAction(
                tool_calls=tool_calls,
                scheduled_actions=actions,
                replies=replies,
                notes=f"Day {day}: policy-driven plan.",
            )
        return plan_fn

    def mutate(self, rng: random.Random) -> "PostingPolicy":
        child = PostingPolicy(
            post_hours=list(self.post_hours),
            content_types=list(self.content_types),
            intents=list(self.intents),
            tag_offset=self.tag_offset,
            topic_offset=self.topic_offset,
            create_hour=self.create_hour,
            use_reply=self.use_reply,
            use_tools_early=self.use_tools_early,
            rest_if_low_energy=self.rest_if_low_energy,
        )

        mutation = rng.choice(["hours", "types", "intents", "tags", "topics",
                               "create", "reply", "tools", "energy", "n_posts"])

        if mutation == "hours":
            child.post_hours = sorted(rng.sample(range(6, 23), min(rng.randint(1, 3), 3)))
        elif mutation == "types":
            n = len(child.post_hours)
            child.content_types = [rng.choice(CONTENT_TYPES) for _ in range(max(n, 1))]
        elif mutation == "intents":
            n = len(child.post_hours)
            child.intents = [rng.choice(INTENTS) for _ in range(max(n, 1))]
        elif mutation == "tags":
            child.tag_offset = rng.randint(0, len(TAG_POOL) - 1)
        elif mutation == "topics":
            child.topic_offset = rng.randint(0, len(ALL_TOPICS) - 1)
        elif mutation == "create":
            child.create_hour = rng.choice([None, 7, 8, 9, 10])
        elif mutation == "reply":
            child.use_reply = not child.use_reply
        elif mutation == "tools":
            child.use_tools_early = not child.use_tools_early
        elif mutation == "energy":
            child.rest_if_low_energy = rng.choice([0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        elif mutation == "n_posts":
            n = rng.randint(1, 3)
            child.post_hours = sorted(rng.sample(range(6, 23), n))
            child.content_types = [rng.choice(CONTENT_TYPES) for _ in range(n)]
            child.intents = [rng.choice(INTENTS) for _ in range(n)]

        return child


def evolutionary_search(
    task: str,
    population_size: int = 12,
    generations: int = 20,
    elite_count: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict], PostingPolicy]:
    """Run evolutionary search to find the best posting policy for a task."""
    rng = random.Random(seed)

    population = [PostingPolicy(
        post_hours=sorted(rng.sample(range(6, 23), rng.randint(1, 3))),
        content_types=[rng.choice(CONTENT_TYPES) for _ in range(3)],
        intents=[rng.choice(INTENTS) for _ in range(3)],
        tag_offset=rng.randint(0, len(TAG_POOL) - 1),
        topic_offset=rng.randint(0, len(ALL_TOPICS) - 1),
        create_hour=rng.choice([None, 7, 8, 9]),
        use_reply=rng.random() > 0.5,
        use_tools_early=rng.random() > 0.5,
        rest_if_low_energy=rng.choice([0.2, 0.25, 0.3, 0.35]),
    ) for _ in range(population_size)]

    log = []

    for gen in range(generations):
        scores = []
        for policy in population:
            plan_fn = policy.to_plan_fn()
            result = run_episode(task, plan_fn, seed=42)
            fitness = result["grader_score"] + 0.1 * result["total_reward"]
            scores.append((fitness, result["grader_score"], result, policy))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_fitness = scores[0][0]
        best_grader = scores[0][1]
        avg_fitness = np.mean([s[0] for s in scores])
        avg_grader = np.mean([s[1] for s in scores])
        worst_grader = scores[-1][1]

        log.append({
            "generation": gen + 1,
            "best_fitness": round(best_fitness, 4),
            "best_grader": round(best_grader, 4),
            "avg_grader": round(avg_grader, 4),
            "worst_grader": round(worst_grader, 4),
            "best_reward": round(scores[0][2]["total_reward"], 4),
            "best_energy": round(scores[0][2]["final_energy"], 3),
            "best_followers": scores[0][2]["follower_delta"],
        })

        print(f"  Gen {gen+1:2d}/{generations}: best_grader={best_grader:.4f} "
              f"avg={avg_grader:.4f} worst={worst_grader:.4f} "
              f"energy={scores[0][2]['final_energy']:.2f} "
              f"Δfollowers={scores[0][2]['follower_delta']:+d}")

        elites = [s[3] for s in scores[:elite_count]]
        new_pop = list(elites)
        while len(new_pop) < population_size:
            parent = rng.choice(elites)
            child = parent.mutate(rng)
            new_pop.append(child)
        population = new_pop

    best_policy = scores[0][3]
    return log, best_policy


# ─── Plotting ──────────────────────────────────────────────────────────

AGENT_COLORS = {
    "always_rest": "#E53935",
    "spam": "#FF9800",
    "random": "#9E9E9E",
    "minimal": "#42A5F5",
    "smart": "#4CAF50",
    "trained": "#7C4DFF",
}

def plot_baseline_leaderboard(baseline_results: Dict):
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
    path = PLOTS_DIR / "baseline_leaderboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_baseline_trajectories(baseline_results: Dict):
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
    path = PLOTS_DIR / "baseline_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_training_curves(evo_logs: Dict[str, List[Dict]]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, task in enumerate(TASKS):
        log = evo_logs[task]
        gens = [e["generation"] for e in log]
        best = [e["best_grader"] for e in log]
        avg = [e["avg_grader"] for e in log]
        worst = [e["worst_grader"] for e in log]

        axes[i].plot(gens, best, "o-", color="#4CAF50", linewidth=2, label="Best", markersize=4)
        axes[i].plot(gens, avg, "s-", color="#2196F3", linewidth=1.5, label="Avg", markersize=3)
        axes[i].fill_between(gens, worst, best, alpha=0.15, color="#2196F3")
        axes[i].set_xlabel("Generation", fontsize=11)
        axes[i].set_ylabel("Grader Score", fontsize=11)
        axes[i].set_title(task.replace("monthly_", "").title(), fontsize=13, fontweight="bold")
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Viraltest v2 — Policy Optimization: Grader Score Over Generations",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = PLOTS_DIR / "reward_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_before_after(baseline_results: Dict, trained_results: Dict):
    task_labels = [t.replace("monthly_", "").title() for t in TASKS]
    random_scores = [baseline_results["random"][t]["grader_score"] for t in TASKS]
    smart_scores = [baseline_results["smart"][t]["grader_score"] for t in TASKS]
    trained_scores = [trained_results[t]["grader_score"] for t in TASKS]

    x = np.arange(len(TASKS))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, random_scores, width, label="Random (untrained baseline)", color="#9E9E9E")
    bars2 = ax.bar(x, trained_scores, width, label="Trained policy (20 gen evolution)", color="#7C4DFF")
    bars3 = ax.bar(x + width, smart_scores, width, label="Smart heuristic (handcrafted)", color="#4CAF50", alpha=0.7)

    ax.set_ylabel("Grader Score", fontsize=12)
    ax.set_title("Before vs After Training — Grader Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.008,
                        f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = PLOTS_DIR / "before_after.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_trained_trajectories(baseline_results: Dict, trained_results: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    comparisons = [
        ("Random baseline", "random", "#9E9E9E", "--"),
        ("Trained policy", "trained", "#7C4DFF", "-"),
        ("Smart heuristic", "smart", "#4CAF50", ":"),
    ]

    for i, task in enumerate(TASKS):
        for label, key, color, ls in comparisons:
            if key == "trained":
                r = trained_results[task]
            else:
                r = baseline_results[key][task]
            lw = 2.5 if key == "trained" else 1.5
            axes[0, i].plot(r["rewards"], label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.9)
            axes[1, i].plot(r["energies"], label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.9)

        task_title = task.replace("monthly_", "").title()
        axes[0, i].set_title(f"{task_title} — Daily Rewards", fontsize=11)
        axes[0, i].set_xlabel("Day"); axes[0, i].set_ylabel("Reward"); axes[0, i].grid(True, alpha=0.3)
        axes[1, i].set_title(f"{task_title} — Energy", fontsize=11)
        axes[1, i].set_xlabel("Day"); axes[1, i].set_ylabel("Energy"); axes[1, i].grid(True, alpha=0.3)

    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    fig.suptitle("Viraltest v2 — Trained Policy vs Baselines", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = PLOTS_DIR / "training_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Part 1: Baseline comparison ──
    print("=" * 70)
    print("PART 1: BASELINE COMPARISON (5 agents × 3 tasks)")
    print("=" * 70)

    baseline_results: Dict[str, Dict[str, Any]] = {}
    for name, fn in BASELINE_AGENTS.items():
        baseline_results[name] = {}
        for task in TASKS:
            global _baseline_rng
            _baseline_rng = random.Random(42)
            result = run_episode(task, fn, seed=42)
            baseline_results[name][task] = result
            print(f"  {name:>12s} | {task:>22s} | score={result['grader_score']:.4f} "
                  f"| energy={result['final_energy']:.2f} | Δfollowers={result['follower_delta']:+d}")
        print()

    print("\nBASELINE LEADERBOARD")
    print(f"{'Agent':<14s} {'Engage':>10s} {'Strategic':>12s} {'Competitive':>14s} {'Avg':>8s}")
    print("-" * 60)
    for name in BASELINE_AGENTS:
        scores = [baseline_results[name][t]["grader_score"] for t in TASKS]
        avg = sum(scores) / len(scores)
        print(f"{name:<14s} {scores[0]:>10.4f} {scores[1]:>12.4f} {scores[2]:>14.4f} {avg:>8.4f}")

    print("\nGenerating baseline plots...")
    plot_baseline_leaderboard(baseline_results)
    plot_baseline_trajectories(baseline_results)

    # ── Part 2: Policy optimization ──
    print("\n" + "=" * 70)
    print("PART 2: POLICY OPTIMIZATION (evolutionary search)")
    print("=" * 70)

    evo_logs: Dict[str, List] = {}
    best_policies: Dict[str, PostingPolicy] = {}

    for task in TASKS:
        print(f"\nOptimizing for {task}...")
        log, best_policy = evolutionary_search(
            task, population_size=12, generations=20, elite_count=3, seed=42)
        evo_logs[task] = log
        best_policies[task] = best_policy

    print("\nGenerating training curves...")
    plot_training_curves(evo_logs)

    # ── Part 3: Trained policy evaluation ──
    print("\n" + "=" * 70)
    print("PART 3: TRAINED POLICY EVALUATION")
    print("=" * 70)

    trained_results: Dict[str, Any] = {}
    for task in TASKS:
        plan_fn = best_policies[task].to_plan_fn()
        result = run_episode(task, plan_fn, seed=42)
        trained_results[task] = result
        print(f"  {task:>22s} | score={result['grader_score']:.4f} "
              f"| reward={result['total_reward']:.3f} | energy={result['final_energy']:.2f} "
              f"| Δfollowers={result['follower_delta']:+d}")

    print("\nGenerating before/after plots...")
    plot_before_after(baseline_results, trained_results)
    plot_trained_trajectories(baseline_results, trained_results)

    # ── Summary ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Task':<25s} {'Random':>10s} {'Trained':>10s} {'Smart':>10s} {'Δ(R→T)':>10s}")
    print("-" * 67)
    for task in TASKS:
        r = baseline_results["random"][task]["grader_score"]
        t_score = trained_results[task]["grader_score"]
        s = baseline_results["smart"][task]["grader_score"]
        print(f"{task:<25s} {r:>10.4f} {t_score:>10.4f} {s:>10.4f} {t_score - r:>+10.4f}")

    avg_r = np.mean([baseline_results["random"][t]["grader_score"] for t in TASKS])
    avg_t = np.mean([trained_results[t]["grader_score"] for t in TASKS])
    avg_s = np.mean([baseline_results["smart"][t]["grader_score"] for t in TASKS])
    print("-" * 67)
    print(f"{'AVERAGE':<25s} {avg_r:>10.4f} {avg_t:>10.4f} {avg_s:>10.4f} {avg_t - avg_r:>+10.4f}")

    summary = {
        "baseline": {name: {task: baseline_results[name][task]["grader_score"] for task in TASKS} for name in BASELINE_AGENTS},
        "trained": {task: trained_results[task]["grader_score"] for task in TASKS},
        "evolution_log": {task: evo_logs[task] for task in TASKS},
        "improvement": {task: trained_results[task]["grader_score"] - baseline_results["random"][task]["grader_score"] for task in TASKS},
    }
    summary_path = PLOTS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    print(f"\nPlots saved to {PLOTS_DIR}/:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p.name}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print("\nTraining evidence is real and reproducible.")


if __name__ == "__main__":
    main()
