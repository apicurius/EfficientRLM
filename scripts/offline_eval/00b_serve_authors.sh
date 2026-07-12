#!/usr/bin/env bash
# Authors' correctness-only policy (full model leg). Same 4 GPUs, run after 00.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
VLLM=$(command -v vllm || echo "$(dirname "$0")/../../.venv-eval/bin/vllm")
exec $VLLM serve mit-oasys/rlm-qwen3-30b-a3b-v0.1 \
  --tensor-parallel-size ${TP:-1} --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --enforce-eager --port ${PORT:-8001} --seed 0
