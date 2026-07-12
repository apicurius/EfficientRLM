#!/usr/bin/env bash
# 8x RTX6000 serving layout: N "ours" (base+adapters, one server) replicas +
# M "authors" replicas, one GPU each (tp=1 — a 30B bf16 model fits one 92GB
# card; more GPUs buys replicas for concurrency, per README GPU sizing note).
# Default split 5/3 roughly matches workload (ours covers up to 3 policies —
# base, t2T_120, t2T_final — vs authors' 1). Override with OURS_GPUS/AUTHORS_GPUS.
set -euo pipefail
cd "$(dirname "$0")"
read -ra OURS_GPUS <<< "${OURS_GPUS:-0 1 2 3 4}"
read -ra AUTHORS_GPUS <<< "${AUTHORS_GPUS:-5 6 7}"

PIDS=()
port=8000
for g in "${OURS_GPUS[@]}"; do
  GPUS=$g PORT=$port bash 00_serve.sh &
  PIDS+=($!)
  port=$((port + 1))
done
ours_port_hi=$((port - 1))
for g in "${AUTHORS_GPUS[@]}"; do
  GPUS=$g PORT=$port bash 00b_serve_authors.sh &
  PIDS+=($!)
  port=$((port + 1))
done

echo "ours replicas: GPUs ${OURS_GPUS[*]} -> ports 8000-$ours_port_hi"
echo "authors replicas: GPUs ${AUTHORS_GPUS[*]} -> ports $((ours_port_hi + 1))-$((port - 1))"
echo "pids: ${PIDS[*]}"
wait
