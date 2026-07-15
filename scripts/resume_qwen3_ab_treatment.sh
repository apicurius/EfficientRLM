#!/usr/bin/env bash
set -euo pipefail

# Resume the qwen3-30b AB treatment run with the persisted shared W&B run id.
#
# Usage:
#   bash scripts/resume_qwen3_ab_treatment.sh             # resume from step 105
#   bash scripts/resume_qwen3_ab_treatment.sh 105         # resume from step 105
#   bash scripts/resume_qwen3_ab_treatment.sh latest      # resume latest consistent checkpoint
#   RESUME_STEP=105 bash scripts/resume_qwen3_ab_treatment.sh
#
# Extra args are forwarded to run_qwen3_ab_treatment.sh:
#   bash scripts/resume_qwen3_ab_treatment.sh 105 --dry-run

EFF="${EFF:-/teamspace/studios/this_studio/EfficientRLM}"
LOCAL_BASE="${LOCAL_BASE:-/teamspace/studios/this_studio/.cache/efficientrlm}"
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$LOCAL_BASE/wandb-shared-run-id}"

STEP="${1:-${RESUME_STEP:-105}}"
if [[ "${1:-}" =~ ^([0-9]+|latest|-1)$ ]]; then
  shift
fi

if [[ "$STEP" == "latest" ]]; then
  STEP="-1"
fi

if [[ -z "${WANDB_SHARED_RUN_ID:-}" && -s "$WANDB_RUN_ID_FILE" ]]; then
  WANDB_SHARED_RUN_ID="$(cat "$WANDB_RUN_ID_FILE")"
  export WANDB_SHARED_RUN_ID
fi

export RESUME=1
export RESUME_STEP="$STEP"

echo "Resuming qwen3 AB treatment from step $RESUME_STEP"
if [[ -n "${WANDB_SHARED_RUN_ID:-}" ]]; then
  echo "Using WANDB_SHARED_RUN_ID=$WANDB_SHARED_RUN_ID"
else
  echo "No persisted W&B run id found; launcher will create one."
fi

cd "$EFF"
exec bash scripts/run_qwen3_ab_treatment.sh "$@"
