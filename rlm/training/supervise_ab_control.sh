#!/bin/bash
# Supervise the BC+ control arm from inside a tmux session on the login node.
# Adapted from erlm/scripts/_supervise_ab_arm.sh: restarts the srun step after
# any non-zero exit (nightly 01:00 login-node reaper, srun hiccups), resuming
# from the latest COMPLETE checkpoint. Drops ARM_DONE on clean completion,
# ARM_FAILED after 3 consecutive fast failures.
set -u

EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
OUT="$EFF/outputs/qwen3-30b-ab-control-multienv-200step"
LOG="${SMOKE_LOG:?set SMOKE_LOG to the console log path}"
DONE="$OUT/ARM_DONE"
FAILED="$OUT/ARM_FAILED"
HOLDER=1263387
MAX_STEP=200

# Latest checkpoint step where BOTH the trainer shard save completed
# (.metadata written last by dcp_save) AND the orchestrator state exists —
# an explicit step avoids resume-step -1 picking a torn checkpoint.
resolve_resume_step() {
  local best_both="" best_trainer=""
  for d in "$OUT"/checkpoints/step_*; do
    [ -e "$d/trainer/.metadata" ] || continue   # .metadata written last: no metadata = torn save
    local n="${d##*_}"
    if [ -z "$best_trainer" ] || [ "$n" -gt "$best_trainer" ]; then best_trainer="$n"; fi
    if [ -d "$OUT/run_default/checkpoints/step_$n" ]; then
      if [ -z "$best_both" ] || [ "$n" -gt "$best_both" ]; then best_both="$n"; fi
    fi
  done
  # Never fall back to a fresh start when ANY complete trainer ckpt exists —
  # fresh wipes every rollout/eval on disk (clean_future_steps -1).
  echo "${best_both:-$best_trainer}"
}

FASTFAILS=0
while :; do
  RESUME_ARGS=()
  STEP="$(resolve_resume_step)"
  if [ -n "$STEP" ]; then
    RESUME_ARGS=("$STEP")
  fi
  echo "[supervise] $(date '+%F %T') control arm launching (resume=${STEP:-fresh})" >> "$LOG"
  START=$(date +%s)
  srun --overlap --jobid="$HOLDER" --gres=gpu:rtx_a6000:8 -n1 \
    --cpus-per-task=48 --mem=0 --time=7-00:00:00 bash -l \
    "$EFF/rlm/training/run_ab_control.sh" "${RESUME_ARGS[@]}" >> "$LOG" 2>&1
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$RC" -eq 0 ]; then
    if [ -d "$OUT/run_default/rollouts/step_$MAX_STEP" ]; then
      echo "[supervise] $(date '+%F %T') control arm COMPLETED rc=0 at step $MAX_STEP" >> "$LOG"
      touch "$DONE"
      exit 0
    fi
    echo "[supervise] $(date '+%F %T') rc=0 BEFORE step $MAX_STEP — restarting with resume" >> "$LOG"
  fi
  # A run dying within 10 min is a config/env problem, not the reaper —
  # bail after 3 in a row instead of crash-looping on the GPUs.
  if [ "$ELAPSED" -lt 600 ]; then
    FASTFAILS=$((FASTFAILS + 1))
    if [ "$FASTFAILS" -ge 3 ]; then
      echo "[supervise] $(date '+%F %T') 3 consecutive fast failures (rc=$RC) — giving up" >> "$LOG"
      touch "$FAILED"
      exit 1
    fi
  else
    FASTFAILS=0
  fi
  echo "[supervise] $(date '+%F %T') died rc=$RC after ${ELAPSED}s — waiting for GPUs to drain" >> "$LOG"
  # Give slurmstepd time to reap the remote step; relaunching into occupied
  # GPUs turns a benign interrupt into fast-fail OOMs.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 60
    USED=$(timeout 90 srun --overlap --jobid="$HOLDER" nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [ -n "$USED" ] && [ "$USED" -lt 2000 ] && break
  done
done
