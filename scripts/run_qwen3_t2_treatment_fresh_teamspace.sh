#!/usr/bin/env bash
set -euo pipefail

# Fresh Teamspace launcher for the Qwen3 T2 treatment arm.
# Usage (once GPUs are attached):
#   cd /teamspace/studios/this_studio/EfficientRLM
#   bash scripts/run_qwen3_t2_treatment_fresh_teamspace.sh
# Optional:
#   RUN_STAMP=20260710T000000Z bash scripts/run_qwen3_t2_treatment_fresh_teamspace.sh
#   WANDB_NAME=my-run HF_LORA_REPO_ID=namespace/repo bash scripts/run_qwen3_t2_treatment_fresh_teamspace.sh

EFF="${EFF:-/teamspace/studios/this_studio/EfficientRLM}"
CONFIG="${CONFIG:-training/configs/qwen3-30b-t2-treatment.toml}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
WANDB_NAME="${WANDB_NAME:-qwen3-30b-t2-treatment-${RUN_STAMP}}"
BASE="${BASE:-$EFF/outputs}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$BASE/${WANDB_NAME}}"
# Keep LOCAL_BASE short: vLLM creates Unix IPC sockets under TMPDIR, and
# sockaddr_un paths are limited to ~107 chars.
LOCAL_BASE="${LOCAL_BASE:-/tmp/erlm-${RUN_STAMP}}"
# Use the shared warmed dependency/model caches; keep LOCAL_BASE fresh only for
# W&B/config/tmp state. A fresh UV cache plus `uv run --no-sync` cannot spawn `rl`.
DL="${DL:-/teamspace/studios/this_studio/.cache/efficientrlm/downloads}"
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$LOCAL_BASE/wandb-shared-run-id}"

export EFF CONFIG BASE RUN_OUTPUT_DIR LOCAL_BASE DL WANDB_RUN_ID_FILE
export RESUME=0 RESUME_STEP=-1
export UPLOAD_LORA=0

# Avoid occasional hangs in this Studio's git wrapper during the nonessential
# prime-rl pin print in scripts/run_qwen3_ab_treatment.sh.
GIT_SHIM_DIR="$LOCAL_BASE/git-shim"
mkdir -p "$GIT_SHIM_DIR"
cat > "$GIT_SHIM_DIR/git" <<'GITSHIM'
#!/usr/bin/env bash
if [[ "$1" == "-C" && "$3" == "describe" ]]; then
  echo "git-describe-skipped-for-launch"
  exit 0
fi
exec /usr/bin/git "$@"
GITSHIM
chmod +x "$GIT_SHIM_DIR/git"
export PATH="$GIT_SHIM_DIR:$PATH"

# Fresh W&B: do not reuse the old persisted shared run id.
unset WANDB_SHARED_RUN_ID
rm -f "$WANDB_RUN_ID_FILE"
mkdir -p "$RUN_OUTPUT_DIR" "$LOCAL_BASE"

cat <<EOF
=== Fresh Qwen3 T2 treatment launch ===
CONFIG=$CONFIG
RUN_OUTPUT_DIR=$RUN_OUTPUT_DIR
LOCAL_BASE=$LOCAL_BASE
WANDB_NAME=$WANDB_NAME
WANDB_RUN_ID_FILE=$WANDB_RUN_ID_FILE
HF_LORA_REPO_ID=${HF_LORA_REPO_ID:-<derived from adapter name>}
EOF

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; GPUs do not appear attached yet. Aborting before launch." >&2
  exit 2
fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

cd "$EFF"
set +e
bash scripts/run_qwen3_ab_treatment.sh \
  --output-dir "$RUN_OUTPUT_DIR" \
  --wandb.name "$WANDB_NAME" \
  "$@"
run_rc=$?
set -e

echo "=== treatment run exited rc=$run_rc ==="

if [[ "$run_rc" == "0" ]]; then
  echo "=== uploading LoRA adapter from RUN_OUTPUT_DIR ==="
  RUN_OUTPUT_DIR="$RUN_OUTPUT_DIR" CONFIG="$CONFIG" bash scripts/upload_qwen3_ab_treatment_lora.sh
else
  echo "Run did not exit cleanly; not auto-uploading. If step_200 adapter exists, upload manually with:" >&2
  echo "  RUN_OUTPUT_DIR='$RUN_OUTPUT_DIR' CONFIG='$CONFIG' bash scripts/upload_qwen3_ab_treatment_lora.sh" >&2
fi
exit "$run_rc"
