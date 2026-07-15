#!/usr/bin/env bash
# 8x RTX6000 serving layout: N identical replicas (base + all LoRA adapters,
# including authors' -- it's a LoRA on the same base model, not a standalone
# model, so every replica can serve every policy). One GPU each (tp=1 -- a
# 30B bf16 model fits one ~98GB card; more GPUs buys replicas for concurrency).
# Override GPU set with SERVE_GPUS (space-separated indices).
set -euo pipefail
cd "$(dirname "$0")"
read -ra SERVE_GPUS <<< "${SERVE_GPUS:-0 1 2 3 4 5 6 7}"

PIDS=()
port=8000
for g in "${SERVE_GPUS[@]}"; do
  GPUS=$g PORT=$port bash 00_serve.sh &
  PIDS+=($!)
  port=$((port + 1))
done

echo "replicas: GPUs ${SERVE_GPUS[*]} -> ports 8000-$((port - 1))"
echo "pids: ${PIDS[*]}"
wait
