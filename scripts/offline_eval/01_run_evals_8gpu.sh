#!/usr/bin/env bash
# Fan the eval workload out across the replicas started by 00_serve_8gpu.sh.
# Ours replicas: sharded by (policy, suite). Authors replica (single policy):
# sharded by suite only. Run this AFTER 00_serve_8gpu.sh's servers are up.
set -euo pipefail
cd "$(dirname "$0")"
N_OURS=${N_OURS:-5}
N_AUTHORS=${N_AUTHORS:-3}
read -ra OURS_POLICIES <<< "${OURS_POLICIES:-Qwen t2T_final}"
# t2T_final: step-200 adapter, uploaded (oerdogan/qwen3-30b-t2-treatment-lora-step200) — now the default.
# add t2T_120 back only if t2T_final shows an oolong deficit:
#   OURS_POLICIES="Qwen t2T_120 t2T_final" bash 01_run_evals_8gpu.sh
SUITES=(trec_coarse_131k oolong_pairs_32k bcplus_heldout codeqa trec_coarse_131k_ext spam_131k_disentangler)

PIDS=()
i=0
for pol in "${OURS_POLICIES[@]}"; do
  for suite in "${SUITES[@]}"; do
    port=$((8000 + i % N_OURS))
    BASE_URL="http://localhost:$port/v1" POLICY_FILTER="$pol" SUITE_FILTER="$suite" \
      bash 01_run_evals.sh &
    PIDS+=($!)
    i=$((i + 1))
  done
done

j=0
for suite in "${SUITES[@]}"; do
  port=$((8000 + N_OURS + j % N_AUTHORS))
  BASE_URL="http://localhost:$port/v1" POLICY_FILTER=mit SUITE_FILTER="$suite" \
    bash 01_run_evals.sh &
  PIDS+=($!)
  j=$((j + 1))
done

echo "launched ${#PIDS[@]} shards: ${PIDS[*]}"
wait
echo "all shards done -- run: python 02_summarize.py"
