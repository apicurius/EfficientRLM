#!/usr/bin/env bash
# Fan the eval workload out across the N identical replicas started by
# 00_serve_8gpu.sh. Run this AFTER the servers are up.
#
# Extension suites (trec_coarse_131k_ext n=200, spam_131k_disentangler n=200) are
# OFF by default -- see 01_run_evals.sh. That leaves 4 light suites
# (50+20+50+50=170 examples/policy: trec, pairs, codeqa, longcot_mini) and 1
# heavy suite, bcplus_heldout (150/policy, 10-20x slower than the others: long
# context, up to 20 sequential agent turns/rollout, external judge calls).
# longcot_mini counts as light: no judge, and its inputs (median ~1.1k tokens,
# p90 ~7.8k, measured with the Qwen3-30B-A3B-Instruct tokenizer over all 507
# easy-split rows) comfortably fit the offline-eval server's max_model_len=16384
# even at the tail -- unlike the training smoke config's seq_len=8192, see
# rlm/training/environments/longcot_mini/README.md.
#
# SLOW_POLICY (default: authors) is confirmed empirically to have the longest,
# most variable rollouts of the policies under test (more sequential agent
# turns/sub-calls per rollout). It gets MORE heavy sub-sharding (tail-latency
# averaging matters most for the policy with the worst tail: splitting its pool
# across more ports means fewer max-concurrent "waves", so a single unlucky
# long rollout blocks less of the total pool) and its OWN dedicated light-suite
# port (zero contention). Fast policies get less heavy sub-sharding (shorter,
# more uniform rollouts pay less for one port grinding the full pool) and share
# light ports among themselves. Net effect vs a naive even split: less
# heavy-suite tail-averaging for fast policies, but more total light-suite
# capacity for everyone and full serving isolation for the slow policy on both
# suite types. (Serving allocation affects wall-clock only, not measured
# turns/subcalls/reward, so this is methodologically neutral.)
set -euo pipefail
cd "$(dirname "$0")"
N_REPLICAS=${N_REPLICAS:-8}
read -ra POLICIES <<< "${POLICIES:-Qwen t2T_final authors}"
# add t2T_120 if t2T_final shows an oolong deficit: POLICIES="Qwen t2T_120 t2T_final authors"
LONGCOT_TOTAL=${LONGCOT_TOTAL:-507}
if [ "${LONGCOT_ONLY:-0}" = "1" ]; then
  LIGHT_SUITES=(longcot_mini)
else
  LIGHT_SUITES=(trec_coarse_131k oolong_pairs_32k codeqa longcot_mini)
fi
N_POLICIES=${#POLICIES[@]}
SLOW_POLICY=${SLOW_POLICY:-authors}
SUBSHARDS_SLOW=${SUBSHARDS_SLOW:-2}
SUBSHARDS_FAST=${SUBSHARDS_FAST:-1}
BC_POOL=${BC_POOL:-150}
export MAX_CONCURRENT_bcplus_heldout=${MAX_CONCURRENT_bcplus_heldout:-32}

HAS_SLOW=0
for pol in "${POLICIES[@]}"; do [ "$pol" = "$SLOW_POLICY" ] && HAS_SLOW=1; done

subshards_for() { [ "$1" = "$SLOW_POLICY" ] && [ "$HAS_SLOW" = "1" ] && echo "$SUBSHARDS_SLOW" || echo "$SUBSHARDS_FAST"; }

N_HEAVY_REPLICAS=0
if [ "${LONGCOT_ONLY:-0}" != "1" ]; then
  for pol in "${POLICIES[@]}"; do
    N_HEAVY_REPLICAS=$((N_HEAVY_REPLICAS + $(subshards_for "$pol")))
  done
fi
if (( N_HEAVY_REPLICAS >= N_REPLICAS )); then
  echo "computed heavy replicas ($N_HEAVY_REPLICAS, from SUBSHARDS_SLOW=$SUBSHARDS_SLOW/SUBSHARDS_FAST=$SUBSHARDS_FAST over $N_POLICIES policies) must be < N_REPLICAS ($N_REPLICAS)" >&2
  exit 1
fi
for pol in "${POLICIES[@]}"; do
  s=$(subshards_for "$pol")
  if (( BC_POOL % s != 0 )); then
    echo "BC_POOL ($BC_POOL) must divide evenly by $pol's subshard count ($s)" >&2
    exit 1
  fi
done
N_LIGHT_REPLICAS=$((N_REPLICAS - N_HEAVY_REPLICAS))

PIDS=()

if [ "${LONGCOT_ONLY:-0}" = "1" ]; then
  # Use every replica for LongCoT-only by splitting the deterministic
  # domain-interleaved easy split into disjoint contiguous slices per policy.
  # Each shard writes to its own family dir (longcot_mini_shXX), so parallel
  # runs never append to the same JSONL.
  # Slice PER POLICY: every policy must cover the same full [0, LONGCOT_TOTAL)
  # range (paired A/B analysis needs identical question sets). A previous
  # version advanced `start` globally across all policies' shards, giving each
  # policy a disjoint 1/N_POLICIES slice — zero question overlap across
  # policies (caught 2026-07-13).
  base=$((LONGCOT_TOTAL / N_REPLICAS))
  rem=$((LONGCOT_TOTAL % N_REPLICAS))
  shard=0
  for pol in "${POLICIES[@]}"; do
    for ((r = 0; r < N_REPLICAS; r++)); do
      n=$base
      if (( r < rem )); then n=$((n + 1)); fi
      start=$((r * base + (r < rem ? r : rem)))
      port=$((8000 + r))
      if (( n > 0 )); then
        BASE_URL="http://localhost:$port/v1" POLICY_FILTER="$pol" SUITE_FILTER=longcot_mini LONGCOT_ONLY=1 \
          N_LONGCOT="$n" LONGCOT_START="$start" OUT_SUFFIX="_sh$(printf "%02d" "$shard")" \
          bash 01_run_evals.sh &
        PIDS+=($!)
      fi
      shard=$((shard + 1))
    done
  done
  echo "launched ${#PIDS[@]} LongCoT-only shard(s): $LONGCOT_TOTAL examples/policy across $N_REPLICAS replica(s) x $N_POLICIES polic(ies), ports 8000-$((8000 + N_REPLICAS - 1))"
  echo "pids: ${PIDS[*]}"
  wait
  echo "all LongCoT-only shards done -- run: python 02_summarize.py"
  exit 0
fi

# Heavy: dedicated ports 8000..8000+N_HEAVY_REPLICAS-1. Each policy's 150-example
# held-out pool is split into that policy's subshard count of disjoint start_index
# ranges (2 for SLOW_POLICY, 1 -- i.e. no split -- for the rest, by default).
hi=0
if [ "${LONGCOT_ONLY:-0}" != "1" ]; then
  for pol in "${POLICIES[@]}"; do
    s=$(subshards_for "$pol")
    bc_n=$((BC_POOL / s))
    for ((sh = 0; sh < s; sh++)); do
      port=$((8000 + hi))
      bc_start=$((680 + sh * bc_n))
      BASE_URL="http://localhost:$port/v1" POLICY_FILTER="$pol" SUITE_FILTER=bcplus_heldout \
        N_BC="$bc_n" BC_START="$bc_start" \
        bash 01_run_evals.sh &
      PIDS+=($!)
      hi=$((hi + 1))
    done
  done
fi

# Light: remaining ports. SLOW_POLICY gets the last light port to itself (when
# >=2 light replicas exist); the rest are striped DIAGONALLY: port index =
# (policy_idx + suite_idx) % FAST_LIGHT_REPLICAS. A plain per-shard round-robin
# puts the SAME suite from every policy on the SAME port (suites cycle in the
# same order for each policy), so the two slowest shards -- trec@131k for both
# fast policies -- collide on one replica (measured: Running 6-10 / Waiting
# 22-28 / KV 88-98% on that port while others idle). Diagonal striping sends
# each suite's shards to different ports.
FAST_LIGHT_REPLICAS=$N_LIGHT_REPLICAS
DEDICATE_SLOW_LIGHT=0
if [ "$HAS_SLOW" = "1" ] && (( N_LIGHT_REPLICAS >= 2 )); then
  DEDICATE_SLOW_LIGHT=1
  FAST_LIGHT_REPLICAS=$((N_LIGHT_REPLICAS - 1))
fi
pi=0
for pol in "${POLICIES[@]}"; do
  si=0
  for suite in "${LIGHT_SUITES[@]}"; do
    if [ "$DEDICATE_SLOW_LIGHT" = "1" ] && [ "$pol" = "$SLOW_POLICY" ]; then
      port=$((8000 + N_REPLICAS - 1))
    else
      port=$((8000 + N_HEAVY_REPLICAS + (pi + si) % FAST_LIGHT_REPLICAS))
    fi
    si=$((si + 1))
    BASE_URL="http://localhost:$port/v1" POLICY_FILTER="$pol" SUITE_FILTER="$suite" LONGCOT_ONLY="${LONGCOT_ONLY:-0}" \
      bash 01_run_evals.sh &
    PIDS+=($!)
  done
  pi=$((pi + 1))
done

echo "launched ${#PIDS[@]} shards: $N_HEAVY_REPLICAS heavy replica(s) (bcplus_heldout, ports 8000-$((8000 + N_HEAVY_REPLICAS - 1)), SUBSHARDS_SLOW=$SUBSHARDS_SLOW/SUBSHARDS_FAST=$SUBSHARDS_FAST, max-concurrent=$MAX_CONCURRENT_bcplus_heldout) + $N_LIGHT_REPLICAS light replica(s) (ports $((8000 + N_HEAVY_REPLICAS))-$((8000 + N_REPLICAS - 1)), $SLOW_POLICY dedicated=$DEDICATE_SLOW_LIGHT)"
echo "pids: ${PIDS[*]}"
wait
echo "all shards done -- run: python 02_summarize.py"
