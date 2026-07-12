#!/usr/bin/env bash
# vLLM server, tp=4: base + LoRA adapters as lora-modules (mirrors run inference.toml).
set -euo pipefail
cd "$(dirname "$0")"
A=$PWD/adapters
VLLM=$(command -v vllm || echo ../../.venv-eval/bin/vllm)
LORAS=""
[ -d "$A/t2T_final" ] && LORAS="$LORAS t2T_final=$A/t2T_final"
[ -d "$A/t2T_120" ]   && LORAS="$LORAS t2T_120=$A/t2T_120"
[ -d "$A/t2C" ]       && LORAS="$LORAS t2C=$A/t2C"
exec $VLLM serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 4 --max-model-len 16384 \
  --gpu-memory-utilization 0.9 --enforce-eager \
  --enable-lora --max-lora-rank 32 --max-loras 8 \
  ${LORAS:+--lora-modules$LORAS} \
  --port 8000 --seed 0
