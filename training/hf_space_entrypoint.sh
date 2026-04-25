#!/bin/sh
# Hugging Face Docker Spaces expect something to listen on app_port (default 7860)
# so the workload becomes "healthy" quickly. Training alone does not open that port.
set -e
STATUS_DIR=/tmp/hfspace_status
mkdir -p "$STATUS_DIR"
printf '%s\n' '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Viraltest training</title></head><body><h1>Viraltest LoRA</h1><p>Training is running. Open <strong>Container logs</strong> in this Space for progress.</p></body></html>' >"$STATUS_DIR/index.html"
python -m http.server 7860 --directory "$STATUS_DIR" &
# Brief pause so the platform health check sees a listener before heavy GPU work.
sleep 2
python training/run_hf_cloud.py || true
exec sleep infinity
