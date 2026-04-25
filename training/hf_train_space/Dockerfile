# Hugging Face Space: training runs here; model weights download on HF servers only.
# Create a new Docker Space, set this repo (or a tiny repo containing only hf_train_space/)
# as the Space git repo, or copy Dockerfile + README into an empty Space repo.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Override at build time if your fork/branch differs (Space → Settings → Variables)
ARG GIT_REPO=https://github.com/VaibhavKhandare/viral-posts-env.git
ARG GIT_BRANCH=hack1

WORKDIR /workspace
RUN git clone --branch "${GIT_BRANCH}" --depth 1 "${GIT_REPO}" viral-posts-env

WORKDIR /workspace/viral-posts-env

RUN chmod +x training/hf_space_entrypoint.sh

RUN pip install --no-cache-dir \
    "transformers>=4.45.0" \
    "accelerate>=1.0.0" \
    "peft>=0.10.0" \
    "trl>=0.20.0" \
    "datasets>=2.0.0" \
    "bitsandbytes>=0.46.1" \
    matplotlib \
    pandas \
    pydantic \
    httpx \
    "openenv-core[core]>=0.2.2"

# HF_TOKEN / MODEL_NAME / HF_OUTPUT_REPO come from Space secrets & variables at runtime
EXPOSE 7860
CMD ["training/hf_space_entrypoint.sh"]
