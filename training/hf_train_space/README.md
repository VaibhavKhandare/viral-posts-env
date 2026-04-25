---
title: Viraltest LoRA training (cloud)
emoji: 🏋️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
# Pick GPU to match MODEL_NAME (set in Space Variables):
#   t4-small  → Qwen/Qwen2.5-1.5B-Instruct (default in script)
#   l4        → Qwen/Qwen2.5-3B-Instruct
#   a10g-small → Qwen/Qwen2.5-7B-Instruct
gpu: t4
---

# Viraltest training on Hugging Face

All **model downloads happen on this machine** (HF Hub cache inside the Space), not on your laptop.

## One-time setup

1. Create a **new** Space → SDK **Docker** → GPU **T4** (or better if you use a larger `MODEL_NAME`).
2. Put **`Dockerfile`** and **`README.md`** from this folder at the **root** of the Space repository (or use the same files from your `viral-posts-env` fork with `Dockerfile` at repo root — see monorepo note below).
3. **Secrets**: add `HF_TOKEN` (write token if you push adapters).
4. **Variables** (optional):
   - `MODEL_NAME` — default `Qwen/Qwen2.5-1.5B-Instruct`
   - `SKIP_BASELINES` — `1` for faster runs
   - `HF_OUTPUT_REPO` — e.g. `yourname/viraltest-lora-1_5b` (create empty model repo first)
   - `NUM_ROUNDS`, `EPISODES_PER_ROUND`

Rebuild the Space; logs will show baselines, training rounds, and final grader scores. Artifacts: `training/hf_cloud_run_summary.json`, `plots/`, `checkpoints/` in the container (ephemeral unless you push to Hub).

**If the Space shows “Launch timed out / workload was not healthy after 30 min”:** Docker Spaces expect something to listen on **`app_port` (7860)** while your job runs. This repo’s `training/hf_space_entrypoint.sh` starts a tiny status page on 7860, then runs `run_hf_cloud.py`, so the platform marks the Space healthy before model download and training finish.

## Monorepo (this full project on GitHub)

Hugging Face builds from the **repository root** and expects a root **`Dockerfile`**.

- Easiest: on a branch like `hf-train-only`, replace root `Dockerfile` with the contents of **`Dockerfile.hftrain`** in this repo (same as `training/hf_train_space/Dockerfile` but uses `COPY` instead of `git clone`). Point your Space at that branch.
- Or use a **separate** tiny Space repo with only `training/hf_train_space/Dockerfile` + `README.md` at root; set `GIT_REPO` / `GIT_BRANCH` build args to the repo that contains `training/run_hf_cloud.py`.

## Credits

Use your HF Pro / credits to select **T4** for the default 1.5B model; scale GPU and `MODEL_NAME` together.
