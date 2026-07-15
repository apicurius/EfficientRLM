#!/usr/bin/env bash
set -euo pipefail

# Prune S3 trainer checkpoints and state-backup tarballs using a safe retention policy.
# Defaults: dry-run. Pass --apply to delete.
# Keeps:
#   - last KEEP_LAST checkpoints
#   - checkpoints divisible by KEEP_INTERVAL
#   - latest backup tar, plus tarballs whose max checkpoint step is kept
#
# Usage:
#   bash scripts/prune_qwen3_ab_treatment_s3_checkpoints.sh
#   bash scripts/prune_qwen3_ab_treatment_s3_checkpoints.sh --apply

RUN_NAME="${RUN_NAME:-qwen3-30b-ab-treatment-multienv-200step}"
KEEP_LAST="${KEEP_LAST:-3}"
KEEP_INTERVAL="${KEEP_INTERVAL:-25}"
CKPT_DIR="${CKPT_DIR:-/teamspace/s3_folders/outputs/efficientrlm/checkpoints/$RUN_NAME/checkpoints}"
BACKUP_ROOT="${BACKUP_ROOT:-/teamspace/s3_folders/outputs/efficientrlm/local-run-backups/$RUN_NAME}"
APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

mapfile -t steps < <(find "$CKPT_DIR" -maxdepth 1 -type d -name 'step_*' -printf '%f\n' 2>/dev/null | sed 's/^step_//' | sort -n)
if [[ ${#steps[@]} -eq 0 ]]; then
  echo "No checkpoints found under $CKPT_DIR"
  exit 0
fi

declare -A keep=()
# keep every KEEP_INTERVAL
for s in "${steps[@]}"; do
  if (( s % KEEP_INTERVAL == 0 )); then keep[$s]=1; fi
done
# keep last KEEP_LAST
start=$(( ${#steps[@]} - KEEP_LAST ))
(( start < 0 )) && start=0
for ((i=start; i<${#steps[@]}; i++)); do keep[${steps[$i]}]=1; done

echo "=== checkpoint retention ==="
echo "CKPT_DIR=$CKPT_DIR"
echo "KEEP_LAST=$KEEP_LAST KEEP_INTERVAL=$KEEP_INTERVAL APPLY=$APPLY"
echo "all:  ${steps[*]}"
printf 'keep:'; for s in "${steps[@]}"; do [[ -n "${keep[$s]:-}" ]] && printf ' %s' "$s"; done; echo
printf 'drop:'; for s in "${steps[@]}"; do [[ -z "${keep[$s]:-}" ]] && printf ' %s' "$s"; done; echo

for s in "${steps[@]}"; do
  if [[ -z "${keep[$s]:-}" ]]; then
    path="$CKPT_DIR/step_$s"
    if (( APPLY )); then
      echo "deleting $path"
      rm -rf -- "$path"
    else
      echo "would delete $path"
    fi
  fi
done

# Backup tar pruning: keep latest marker target and backups whose max contained checkpoint is retained.
if [[ -d "$BACKUP_ROOT" ]]; then
  latest=""
  [[ -f "$BACKUP_ROOT/LATEST" ]] && latest="$(cat "$BACKUP_ROOT/LATEST")"
  echo
  echo "=== backup tar retention ==="
  echo "BACKUP_ROOT=$BACKUP_ROOT latest=$latest"
  for tarball in "$BACKUP_ROOT"/state-*.tar; do
    [[ -f "$tarball" ]] || continue
    name="$(basename "$tarball")"
    if [[ "$name" == "$latest" ]]; then
      echo "keep latest $tarball"
      continue
    fi
    max_step="$(tar -tf "$tarball" 2>/dev/null | sed -nE 's#.*run_default/checkpoints/step_([0-9]+)/.*#\1#p' | sort -n | tail -1)"
    if [[ -n "$max_step" && -n "${keep[$max_step]:-}" ]]; then
      echo "keep $tarball (max_step=$max_step kept)"
    else
      if (( APPLY )); then
        echo "deleting $tarball (max_step=${max_step:-none})"
        rm -f -- "$tarball"
      else
        echo "would delete $tarball (max_step=${max_step:-none})"
      fi
    fi
  done
fi

if (( ! APPLY )); then
  echo
  echo "Dry run only. Re-run with --apply to delete."
fi
