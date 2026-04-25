"""Read runs/{baseline,agent}.jsonl and produce three labeled plots.

PDF page 3 requirements honored:
- Both axes labeled with units
- PNG (dpi=150), committed to repo
- Baseline vs trained on the SAME axes for fair comparison
- Captions live in README (not in plot titles)

Outputs: plots/reward_curve.png, plots/before_after.png, plots/signals_breakdown.png
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PLOTS = ROOT / "plots"
TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]


def load_runs(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def placeholder_plot(out: Path, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, wrap=True,
            transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_curve(baseline: List[Dict], agent: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    def per_task_curve(runs, label, color):
        for task in TASKS:
            task_runs = [r for r in runs if r["task"] == task]
            if not task_runs:
                continue
            n_steps = max(len(r["rewards"]) for r in task_runs)
            avg = []
            for i in range(n_steps):
                step_rewards = [r["rewards"][i] for r in task_runs if i < len(r["rewards"])]
                avg.append(statistics.mean(step_rewards) if step_rewards else 0)
            ax.plot(range(1, len(avg) + 1), avg, linewidth=1.5,
                    label=f"{label} ({task})", color=color, alpha=0.7,
                    linestyle="-" if label == "agent" else "--")

    per_task_curve(baseline, "baseline", "#FF9800")
    per_task_curve(agent, "agent", "#2196F3")
    ax.set_xlabel("Step (day in 30-day episode)")
    ax.set_ylabel("Per-step env reward (0-1)")
    ax.set_title("Viraltest — per-step reward over a 30-day episode")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_before_after(baseline: List[Dict], agent: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    def task_stats(runs, task):
        scores = [r["grader_score"] for r in runs if r["task"] == task]
        if not scores:
            return 0.0, 0.0
        return statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0

    base_means = [task_stats(baseline, t)[0] for t in TASKS]
    base_stds = [task_stats(baseline, t)[1] for t in TASKS]
    agent_means = [task_stats(agent, t)[0] for t in TASKS]
    agent_stds = [task_stats(agent, t)[1] for t in TASKS]

    x = list(range(len(TASKS)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], base_means, width, yerr=base_stds,
           label="Baseline (heuristic, no tools)", color="#FF9800", capsize=4)
    ax.bar([i + width / 2 for i in x], agent_means, width, yerr=agent_stds,
           label="Agent (Gemma + tools)", color="#4CAF50", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=15)
    ax.set_ylabel("Grader score (0-1, higher = better)")
    ax.set_title("Viraltest — baseline vs tool-using agent")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_signals_breakdown(baseline: List[Dict], agent: List[Dict], out: Path) -> None:
    keys = ["watch_time", "sends_per_reach", "saves", "likes_per_reach"]

    def aggregate(runs):
        totals: Dict[str, float] = defaultdict(float)
        n = 0
        for r in runs:
            for s in r["steps"]:
                for k in keys:
                    totals[k] += s["signals"].get(k, 0.0)
                n += 1
        return {k: (totals[k] / n if n else 0.0) for k in keys}

    base = aggregate(baseline)
    ag = aggregate(agent)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(keys)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [base[k] for k in keys], width,
           label="Baseline", color="#FF9800")
    ax.bar([i + width / 2 for i in x], [ag[k] for k in keys], width,
           label="Agent", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(["watch_time\n(0.40w)", "sends_per_reach\n(0.30w)",
                         "saves\n(0.20w)", "likes_per_reach\n(0.10w)"])
    ax.set_ylabel("Mean per-step signal value")
    ax.set_title("Mosseri-aligned engagement signals — baseline vs agent")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    baseline = load_runs(RUNS / "baseline.jsonl")
    agent = load_runs(RUNS / "agent.jsonl")

    if not baseline and not agent:
        msg = "No runs/ data yet — run scripts/run_baseline_vs_agent.py first."
        for name in ("reward_curve", "before_after", "signals_breakdown"):
            placeholder_plot(PLOTS / f"{name}.png", msg)
        print(msg)
        return 1

    plot_reward_curve(baseline, agent, PLOTS / "reward_curve.png")
    plot_before_after(baseline, agent, PLOTS / "before_after.png")
    plot_signals_breakdown(baseline, agent, PLOTS / "signals_breakdown.png")
    print(f"baseline runs: {len(baseline)}  agent runs: {len(agent)}")
    print(f"wrote: {PLOTS}/reward_curve.png")
    print(f"wrote: {PLOTS}/before_after.png")
    print(f"wrote: {PLOTS}/signals_breakdown.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
