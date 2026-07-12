#!/usr/bin/env bash
# Authors' correctness-only policy (full model leg). Same 4 GPUs, run after 00.
set -euo pipefail
VLLM=$(command -v vllm || echo "$(dirname "$0")/../../.venv-eval/bin/vllm")
exec $VLLM serve mit-oasys/rlm-qwen3-30b-a3b-v0.1 \
  --tensor-parallel-size 4 --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --enforce-eager --port 8000 --seed 0
