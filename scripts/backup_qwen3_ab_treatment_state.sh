#!/usr/bin/env bash
set -euo pipefail

# Snapshot local prime-rl state needed for cross-machine resume.
# This does NOT back up large regenerable caches (HF/uv/triton/etc.).
#
# Usage:
#   bash scripts/backup_qwen3_ab_treatment_state.sh
#   bash scripts/backup_qwen3_ab_treatment_state.sh --watch 300
#   bash scripts/backup_qwen3_ab_treatment_state.sh --watch-ckpt        # back up on each new checkpoint
#   bash scripts/backup_qwen3_ab_treatment_state.sh --watch-ckpt 30     # poll every 30s
#
# Restore later with:
#   bash scripts/restore_qwen3_ab_treatment_state.sh

RUN_NAME="${RUN_NAME:-qwen3-30b-ab-treatment-multienv-200step}"
LOCAL_BASE="${LOCAL_BASE:-/teamspace/studios/this_studio/.cache/efficientrlm}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$LOCAL_BASE/prime-rl/$RUN_NAME}"
BACKUP_ROOT="${BACKUP_ROOT:-/teamspace/s3_folders/outputs/efficientrlm/local-run-backups/$RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$LOCAL_RUN_DIR/run_default/checkpoints}"
INTERVAL=""
CKPT_POLL=""

if [[ "${1:-}" == "--watch" ]]; then
  INTERVAL="${2:-300}"
elif [[ "${1:-}" == "--watch-ckpt" ]]; then
  CKPT_POLL="${2:-30}"
fi

latest_ckpt_step() {
  local max=-1 d n
  if [[ -d "$CKPT_DIR" ]]; then
    for d in "$CKPT_DIR"/step_*; do
      [[ -d "$d" ]] || continue
      n="${d##*/step_}"
      [[ "$n" =~ ^[0-9]+$ ]] || continue
      (( n > max )) && max="$n"
    done
  fi
  echo "$max"
}

make_backup() {
  if [[ ! -d "$LOCAL_RUN_DIR" ]]; then
    echo "ERROR: local run dir does not exist: $LOCAL_RUN_DIR" >&2
    return 1
  fi

  mkdir -p "$BACKUP_ROOT"
  local ts tmp archive latest
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  tmp="$BACKUP_ROOT/state-$ts.tar.tmp"
  archive="$BACKUP_ROOT/state-$ts.tar"
  latest="$BACKUP_ROOT/LATEST"

  echo "=== backing up local run state ==="
  echo "LOCAL_RUN_DIR=$LOCAL_RUN_DIR"
  echo "BACKUP_ROOT=$BACKUP_ROOT"
  echo "archive=$archive"

  # Important resume pieces:
  # - configs: generated subconfigs used by trainer/orchestrator/inference
  # - run_default/checkpoints: orchestrator state/progress
  # - run_default/broadcasts: adapter weights needed by orchestrator resume
  # - logs and wandb: diagnostics/continuity; not strictly required but useful
  # - top-level wandb/logs/configs: trainer shared logging + configs
  (
    cd "$LOCAL_RUN_DIR"
    tar \
      --ignore-failed-read \
      --warning=no-file-changed \
      --warning=no-file-removed \
      -cf "$tmp" \
      configs logs wandb run_default/control run_default/checkpoints run_default/broadcasts run_default/wandb \
      2> >(grep -v -E 'Cannot stat|file changed as we read it|File removed before we read it' >&2) \
      || true
  )

  if [[ ! -s "$tmp" ]]; then
    echo "ERROR: backup archive was not created or is empty: $tmp" >&2
    rm -f "$tmp"
    return 1
  fi

  mv -f "$tmp" "$archive"
  printf '%s\n' "$(basename "$archive")" > "$latest"
  du -h "$archive" | awk '{print "backup_size=" $1}'
  echo "latest=$(cat "$latest")"
  echo "Backup complete."
}

if [[ -n "$CKPT_POLL" ]]; then
  echo "Watching for new checkpoints in $CKPT_DIR (poll ${CKPT_POLL}s). Press Ctrl-C to stop."
  last="$(latest_ckpt_step)"
  echo "Starting from latest checkpoint step: $last"
  if [[ "$last" != "-1" ]]; then
    make_backup || true
  fi
  while true; do
    sleep "$CKPT_POLL"
    cur="$(latest_ckpt_step)"
    if [[ "$cur" != "-1" && "$cur" != "$last" ]]; then
      echo "New checkpoint detected: step $cur (was $last)"
      make_backup || true
      last="$cur"
    fi
  done
elif [[ -n "$INTERVAL" ]]; then
  echo "Watching and backing up every ${INTERVAL}s. Press Ctrl-C to stop."
  while true; do
    make_backup || true
    sleep "$INTERVAL"
  done
else
  make_backup
fi
