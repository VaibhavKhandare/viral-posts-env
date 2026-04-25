#!/usr/bin/env python3
"""
Viraltest LoRA training — intended for Hugging Face GPU Spaces / cloud runners.

All model weights download on the remote machine (HF Hub cache on the server), not your laptop.

Environment (optional unless noted):
  HF_TOKEN          — Hub auth (gated models, push adapter)
  MODEL_NAME        — default Qwen/Qwen2.5-1.5B-Instruct (use 3B on L4+, 7B on A10G)
  SKIP_BASELINES    — set to 1 to skip heuristic baselines (faster)
  NUM_ROUNDS        — default 4
  EPISODES_PER_ROUND — default 6
  HF_OUTPUT_REPO    — if set, push LoRA adapter to this Hub repo id (create empty repo first)

Infra → model hints (VRAM, rough):
  T4 16GB   → Qwen/Qwen2.5-1.5B-Instruct + 4-bit LoRA
  L4 / A10G 24GB → Qwen/Qwen2.5-3B-Instruct
  A100 40GB → Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root = parent of training/
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from models import ScheduledAction, ToolCall, ViraltestAction
from server.viraltest_environment import (
    TAG_POOL,
    TASK_HORIZON,
    TOPIC_CATEGORIES,
    ViraltestEnvironment,
)

ALL_TOPICS = [t for topics in TOPIC_CATEGORIES.values() for t in topics]
NICHES = list(TOPIC_CATEGORIES.keys())
CONTENT_TYPES = ["reel", "carousel", "story", "text_post"]
INTENTS = ["send_bait", "save_bait", "watch_bait", "like_bait"]
TASKS = ["monthly_engage", "monthly_strategic", "monthly_competitive"]

SYSTEM_PROMPT = textwrap.dedent(
    """\
You are an Instagram content strategy agent. Each step is one day.
You manage a creator account over a 30-day cycle.

RESPONSE FORMAT — return ONLY valid JSON, no markdown:
{
  "tool_calls": [{"name": "query_trends", "arguments": {"niche": "tech"}}],
  "scheduled_actions": [
    {"hour": 12, "action_type": "post", "content_type": "reel",
     "topic": "AI tools", "tags": ["ai", "coding"], "intent": "watch_bait"}
  ],
  "replies": [{"post_hour": 12, "reply_hour": 13}],
  "notes": "strategy notes"
}

RULES:
- content_type: reel|story|carousel|text_post
- intent: send_bait|save_bait|watch_bait|like_bait
- 1-2 posts/day optimal. More = fatigue.
- Empty scheduled_actions = rest (recovers energy).
- Vary content types and topics for diversity bonus.
- Reply within 90 min of post for reach bonus."""
)


def _rng() -> random.Random:
    return random.Random(42)


def plan_always_rest(obs_dict, day):
    return ViraltestAction(scheduled_actions=[])


def plan_spam(obs_dict, day):
    return ViraltestAction(
        scheduled_actions=[
            ScheduledAction(
                hour=h,
                action_type="post",
                content_type="reel",
                topic="AI tools",
                tags=["ai"],
                intent="watch_bait",
            )
            for h in range(24)
        ]
    )


def plan_random(obs_dict, day):
    rng = _rng()
    actions = []
    for h in range(24):
        if rng.random() < 0.1:
            actions.append(
                ScheduledAction(
                    hour=h,
                    action_type="post",
                    content_type=rng.choice(CONTENT_TYPES),
                    topic=rng.choice(ALL_TOPICS),
                    tags=rng.sample(TAG_POOL[:30], 3),
                    intent=rng.choice(INTENTS),
                )
            )
    return ViraltestAction(scheduled_actions=actions)


def plan_minimal(obs_dict, day):
    return ViraltestAction(
        scheduled_actions=[
            ScheduledAction(
                hour=12,
                action_type="post",
                content_type="carousel",
                topic=ALL_TOPICS[day % len(ALL_TOPICS)],
                tags=[TAG_POOL[i % len(TAG_POOL)] for i in range(day, day + 3)],
                intent="save_bait",
            )
        ]
    )


def plan_smart(obs_dict, day):
    return ViraltestAction(
        tool_calls=[
            ToolCall(name="query_trends", arguments={"niche": NICHES[day % len(NICHES)]})
        ]
        if day <= 3
        else [],
        scheduled_actions=[
            ScheduledAction(hour=8, action_type="create_content"),
            ScheduledAction(
                hour=12,
                action_type="post",
                content_type=CONTENT_TYPES[(day * 2) % 4],
                topic=ALL_TOPICS[(day * 2) % len(ALL_TOPICS)],
                tags=[TAG_POOL[(day * 6 + i) % len(TAG_POOL)] for i in range(3)],
                intent=INTENTS[(day * 2) % 4],
            ),
            ScheduledAction(
                hour=19,
                action_type="post",
                content_type=CONTENT_TYPES[(day * 2 + 1) % 4],
                topic=ALL_TOPICS[(day * 2 + 1) % len(ALL_TOPICS)],
                tags=[TAG_POOL[(day * 6 + 3 + i) % len(TAG_POOL)] for i in range(3)],
                intent=INTENTS[(day * 2 + 1) % 4],
            ),
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


def run_episode(task: str, plan_fn, seed: int = 42) -> Dict[str, Any]:
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
        "steps": len(rewards),
        "final_energy": obs.creator_energy,
        "follower_delta": obs.follower_count - 10000,
        "burned_out": obs.creator_energy <= 0,
        "rewards": rewards,
        "energies": energies,
    }


def format_obs(obs) -> str:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_name = days[obs.day_of_week] if 0 <= obs.day_of_week < 7 else "?"
    signals_str = ""
    signals = getattr(obs, "engagement_signals", None)
    if signals:
        signals_str = (
            f"Signals: watch={signals.watch_time:.3f} "
            f"sends={signals.sends_per_reach:.3f} saves={signals.saves:.3f}\n"
        )
    tool_str = ""
    for tr in getattr(obs, "tool_results", []):
        if tr.success:
            tool_str += f"  {tr.name}: {json.dumps(tr.data)[:200]}\n"
    tool_block = tool_str if tool_str else "  (none)\n"
    return (
        f"Day: {day_name} | days_elapsed={obs.days_elapsed}\n"
        f"Energy: {obs.creator_energy:.2f} | Followers: {obs.follower_count}\n"
        f"Engagement: {obs.engagement_rate:.3f} | Queue: {obs.content_queue_size}\n"
        f"{signals_str}"
        f"Tool results:\n{tool_block}"
        f"Plan your actions (JSON only):"
    )


def parse_model_output(text: str) -> ViraltestAction:
    text = text.strip()
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        data = json.loads(text)
        tool_calls = [
            ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in data.get("tool_calls", [])
            if isinstance(tc, dict) and "name" in tc
        ]
        scheduled = []
        for a in data.get("scheduled_actions", []):
            try:
                scheduled.append(ScheduledAction(**a))
            except Exception:
                pass
        return ViraltestAction(
            tool_calls=tool_calls,
            scheduled_actions=scheduled,
            replies=data.get("replies", []),
            notes=data.get("notes"),
        )
    except Exception:
        return ViraltestAction(scheduled_actions=[])


def _infer_model_device(m) -> torch.device:
    p = next(m.parameters(), None)
    if p is not None:
        return p.device
    d = getattr(m, "device", None)
    if d is not None:
        return d
    return torch.device("cpu")


def generate_action(mdl, tok, obs, history, temperature=0.7) -> Tuple[str, ViraltestAction]:
    prompt = format_obs(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": prompt})
    text_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text_input, return_tensors="pt").to(_infer_model_device(mdl))
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    resp = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return resp, parse_model_output(resp)


def run_llm_episode(mdl, tok, task: str, seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    env = ViraltestEnvironment()
    obs = env.reset(task=task, seed=seed)
    rewards, energies = [], [obs.creator_energy]
    history, pairs = [], []
    for day in range(1, TASK_HORIZON + 1):
        if obs.done:
            break
        if obs.creator_energy <= 0.25:
            action = ViraltestAction(scheduled_actions=[])
            resp = '{"scheduled_actions": []}'
        else:
            resp, action = generate_action(mdl, tok, obs, history)
        prompt = format_obs(obs)
        pairs.append({"prompt": prompt, "response": resp})
        obs = env.step(action)
        r = obs.reward or 0.0
        rewards.append(r)
        energies.append(obs.creator_energy)
        history.extend(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
        )
        if verbose:
            n_p = len([s for s in action.scheduled_actions if s.action_type == "post"])
            print(
                f"    Day {day:2d}: r={r:.4f} e={obs.creator_energy:.2f} posts={n_p} tools={len(action.tool_calls)}"
            )
        if obs.done:
            break
    gs = (obs.metadata or {}).get("grader_score", 0.0)
    return {
        "task": task,
        "grader_score": gs,
        "total_reward": sum(rewards),
        "final_energy": obs.creator_energy,
        "rewards": rewards,
        "energies": energies,
        "pairs": pairs,
        "follower_delta": obs.follower_count - 10000,
        "burned_out": obs.creator_energy <= 0,
    }


def pair_to_training_text(tok: AutoTokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_model_and_tokenizer(model_name: str, hf_token: str | None):
    tok_kwargs = {"trust_remote_code": True}
    mdl_kwargs = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token
        mdl_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    use_4bit = False
    try:
        from transformers.utils import is_bitsandbytes_available
    except Exception:

        def is_bitsandbytes_available():
            try:
                import bitsandbytes  # noqa: F401

                return True
            except ImportError:
                return False

    if torch.cuda.is_available() and is_bitsandbytes_available():
        from transformers import BitsAndBytesConfig

        use_4bit = True

    if use_4bit:
        print(f"Loading {model_name} (4-bit, CUDA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            **mdl_kwargs,
        )
    else:
        dtype = (
            torch.float16
            if (
                torch.cuda.is_available()
                or (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            )
            else torch.float32
        )
        print(f"Loading {model_name} (fp16/fp32, no 4-bit)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            **mdl_kwargs,
        )
        if not torch.cuda.is_available():
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                model = model.to("mps")
            else:
                model = model.to("cpu")

    model.eval()
    print(f"Model dtype={next(model.parameters()).dtype} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def main() -> None:
    assert TASK_HORIZON == 30, f"Expected TASK_HORIZON=30, got {TASK_HORIZON}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    skip_baselines = os.environ.get("SKIP_BASELINES", "").lower() in ("1", "true", "yes")
    num_rounds = int(os.environ.get("NUM_ROUNDS", "4"))
    episodes_per_round = int(os.environ.get("EPISODES_PER_ROUND", "6"))
    top_k_fraction = float(os.environ.get("TOP_K_FRACTION", "0.5"))
    output_repo = os.environ.get("HF_OUTPUT_REPO", "").strip()

    print("REPO_ROOT", REPO_ROOT)
    print("MODEL_NAME", model_name)
    print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    baseline_results = {}
    if not skip_baselines:
        print("Heuristic baselines (5 × 3)...")
        for name, fn in BASELINE_AGENTS.items():
            baseline_results[name] = {}
            for task in TASKS:
                random.seed(42)
                baseline_results[name][task] = run_episode(task, fn, seed=42)
                r = baseline_results[name][task]
                print(f"  {name:>12} | {task} | grader={r['grader_score']:.4f}")
        agent_names = list(BASELINE_AGENTS.keys())
        colors = ["#E53935", "#FF9800", "#9E9E9E", "#42A5F5", "#4CAF50"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for i, task in enumerate(TASKS):
            scores = [baseline_results[a][task]["grader_score"] for a in agent_names]
            axes[i].barh(agent_names, scores, color=colors)
            axes[i].set_title(task.replace("monthly_", "").title())
        fig.suptitle("Viraltest v2 — Heuristic Baseline Leaderboard")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "baseline_leaderboard.png", dpi=150, bbox_inches="tight")
        plt.close()

    model, tokenizer = load_model_and_tokenizer(model_name, hf_token)

    print("Untrained LLM baseline...")
    before_results = {}
    for task in TASKS:
        result = run_llm_episode(model, tokenizer, task, seed=42, verbose=True)
        before_results[task] = result
        print(f"  {task}: grader={result['grader_score']:.4f}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_log: Dict[str, List[Any]] = {
        "round": [],
        "avg_episode_reward": [],
        "max_episode_reward": [],
        "min_episode_reward": [],
        "avg_grader": [],
        "max_grader": [],
        "n_training_samples": [],
        "train_loss": [],
    }
    t_start = time.time()

    for round_idx in range(1, num_rounds + 1):
        print(f"\n=== TRAINING ROUND {round_idx}/{num_rounds} ===")
        peft_model.eval()
        all_pairs, episode_rewards, episode_graders = [], [], []

        for ep in range(episodes_per_round):
            task = TASKS[ep % len(TASKS)]
            seed = 42 + (round_idx - 1) * 100 + ep
            result = run_llm_episode(peft_model, tokenizer, task, seed=seed)
            ep_reward = result["total_reward"] + 2.0 * result["grader_score"]
            episode_rewards.append(ep_reward)
            episode_graders.append(result["grader_score"])
            for pr in result["pairs"]:
                text = pair_to_training_text(tokenizer, pr["prompt"], pr["response"])
                all_pairs.append({"text": text, "reward": ep_reward})
            print(
                f"  ep {ep+1}/{episodes_per_round}: {task} "
                f"grader={result['grader_score']:.4f} reward={ep_reward:.3f}"
            )

        avg_r = float(np.mean(episode_rewards))
        avg_g = float(np.mean(episode_graders))
        print(f"  Avg reward={avg_r:.3f} Avg grader={avg_g:.4f}")

        threshold = np.percentile([p["reward"] for p in all_pairs], (1 - top_k_fraction) * 100)
        filtered = [p for p in all_pairs if p["reward"] >= threshold] or all_pairs
        print(f"  Filtered to {len(filtered)}/{len(all_pairs)} samples")

        dataset = Dataset.from_list([{"text": p["text"]} for p in filtered])
        ckpt_dir = REPO_ROOT / "checkpoints" / f"round_{round_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        use_fp16 = torch.cuda.is_available()
        sft_config = SFTConfig(
            output_dir=str(ckpt_dir),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=5,
            logging_steps=5,
            save_strategy="no",
            max_length=1024,
            fp16=use_fp16,
            bf16=False,
            report_to="none",
        )
        peft_model.train()
        trainer = SFTTrainer(
            model=peft_model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )
        train_result = trainer.train()
        loss = train_result.training_loss
        print(f"  Training loss: {loss:.4f}")

        training_log["round"].append(round_idx)
        training_log["avg_episode_reward"].append(round(avg_r, 3))
        training_log["max_episode_reward"].append(round(float(max(episode_rewards)), 3))
        training_log["min_episode_reward"].append(round(float(min(episode_rewards)), 3))
        training_log["avg_grader"].append(round(avg_g, 4))
        training_log["max_grader"].append(round(float(max(episode_graders)), 4))
        training_log["n_training_samples"].append(len(filtered))
        training_log["train_loss"].append(round(loss, 4))

    print("\n", pd.DataFrame(training_log).to_string(index=False))
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min")

    print("Trained model eval...")
    peft_model.eval()
    after_results = {}
    for task in TASKS:
        result = run_llm_episode(peft_model, tokenizer, task, seed=42, verbose=True)
        after_results[task] = result
        print(f"  {task}: grader={result['grader_score']:.4f}")

    summary = {
        "model": model_name,
        "before": {t: before_results[t]["grader_score"] for t in TASKS},
        "after": {t: after_results[t]["grader_score"] for t in TASKS},
        "training_log": training_log,
    }
    out_json = REPO_ROOT / "training" / "hf_cloud_run_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_json)

    if output_repo and hf_token:
        print("Pushing adapter to", output_repo)
        peft_model.push_to_hub(output_repo, token=hf_token, private=False)
        tokenizer.push_to_hub(output_repo, token=hf_token, private=False)
        print("Push complete.")
    elif output_repo and not hf_token:
        print("HF_OUTPUT_REPO set but no HF_TOKEN; skipping push.")


if __name__ == "__main__":
    main()
