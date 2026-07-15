#!/usr/bin/env bash
set -euo pipefail

# Restore local prime-rl state snapshot from S3 backup before resuming on a new machine.
# Usage:
#   bash scripts/restore_qwen3_ab_treatment_state.sh          # restore latest
#   bash scripts/restore_qwen3_ab_treatment_state.sh FILE.tar # restore named snapshot

RUN_NAME="${RUN_NAME:-qwen3-30b-ab-treatment-multienv-200step}"
LOCAL_BASE="${LOCAL_BASE:-/teamspace/studios/this_studio/.cache/efficientrlm}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$LOCAL_BASE/prime-rl/$RUN_NAME}"
BACKUP_ROOT="${BACKUP_ROOT:-/teamspace/s3_folders/outputs/efficientrlm/local-run-backups/$RUN_NAME}"
SNAPSHOT="${1:-}"

if [[ -z "$SNAPSHOT" ]]; then
  if [[ ! -f "$BACKUP_ROOT/LATEST" ]]; then
    echo "ERROR: no latest marker found: $BACKUP_ROOT/LATEST" >&2
    exit 1
  fi
  SNAPSHOT="$(cat "$BACKUP_ROOT/LATEST")"
fi

if [[ "$SNAPSHOT" != /* ]]; then
  SNAPSHOT="$BACKUP_ROOT/$SNAPSHOT"
fi

if [[ ! -f "$SNAPSHOT" ]]; then
  echo "ERROR: snapshot not found: $SNAPSHOT" >&2
  exit 1
fi

mkdir -p "$LOCAL_RUN_DIR"

echo "=== restoring local run state ==="
echo "snapshot=$SNAPSHOT"
echo "LOCAL_RUN_DIR=$LOCAL_RUN_DIR"

tar -xf "$SNAPSHOT" -C "$LOCAL_RUN_DIR"

echo "Restore complete."
echo "You can now resume with:"
echo "  cd /teamspace/studios/this_studio/EfficientRLM && RESUME=1 bash scripts/run_qwen3_ab_treatment.sh"
