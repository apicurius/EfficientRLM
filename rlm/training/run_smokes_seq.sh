#!/bin/bash
# Sequentially run the three Qwen3-30B OOLONG reward-style smoke configs on the
# held ai16 8xA6000 node. Launched via srun --overlap into the holder job, e.g.:
#
#   tmux new -d -s smoke "srun --jobid=<HOLDER> --overlap \
#       --gres=gpu:rtx_a6000:8 -n1 --cpus-per-task=48 --mem=0 --time=10:00:00 \
#       bash /scratch/omeerdogan23/erlm/.research/EfficientRLM/rlm/training/run_smokes_seq.sh"
#
# Each `uv run rl` is a full pipeline (inference + orchestrator + trainer) that
# tears down before the next starts -> truly sequential, one run owns all 8 GPUs.
#
# No `set -e`: module/.bashrc sourcing returns nonzero and would abort the script.

ERLM=/scratch/omeerdogan23/erlm
PRL=$ERLM/prime-rl   # runtime venv (has the dispatcher deadline env-var patch); EfficientRLM/prime-rl stays pristine. NOTE: $PRL venv must have EfficientRLM rlm_train+envs installed (editable) before smokes exercise THIS tree.
CFGDIR=$ERLM/.research/EfficientRLM/rlm/training/configs
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p $ERLM/.research/EfficientRLM/outputs
SEQLOG=$ERLM/.research/EfficientRLM/outputs/smoke-seq-console-$TS.log
echo "$SEQLOG" > $ERLM/.research/EfficientRLM/outputs/.last_smoke_seq_log   # publish log path up front for the monitor

# --- toolchain (module loads do not persist across shells; bake them in) ---
module load cuda/12.8.0 git/2.9.5 >/dev/null 2>&1

# --- credentials (WANDB_API_KEY, HF_TOKEN). set -a so WANDB_API_KEY actually
#     EXPORTS to the wandb child processes; without it the sourced var
#     stays shell-local and wandb falls back to ~/.netrc (wrong account). ---
[ -f "$ERLM/rlm/.env" ] && { set -a; source "$ERLM/rlm/.env"; set +a; }

# --- HF cache (model fully cached; stay offline) ---
export HF_HOME=/scratch/omeerdogan23/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
# --- uv cache on node-local disk (beegfs rename(2) -> EBUSY otherwise) ---
export UV_CACHE_DIR=/tmp/uvcache_$USER
# --- ai16 A6000 has no NVLink (PCIe PXB): CUDA P2P hangs vLLM TP NCCL init ---
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# --- W&B online (force, in case .env set a mode) ---
export WANDB_MODE=online
# --- W&B entity: log to omeerdogan-koc-university (the account the wandb_v1_
#     key + MCP authenticate as) so runs/traces are visible via the MCP. ---
export WANDB_ENTITY=omeerdogan-koc-university
# --- RLM REPL worker startup guard.  Workers are spawned in large bursts and
# import the full stack; give them room on shared filesystems. ---
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=120

cd "$PRL" || { echo "[seq] cannot cd $PRL"; exit 1; }

# Default to all three; pass config filenames as args to run a subset.
if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    smoke-qwen3-30b-correctness-oolong.toml
    smoke-qwen3-30b-efficiency-oolong.toml
    smoke-qwen3-30b-harness1-oolong.toml
  )
fi

{
  echo "[seq] ===================================================="
  echo "[seq] start=$TS host=$(hostname) jobstep=${SLURM_JOB_ID:-?}"
  echo "[seq] WANDB_MODE=$WANDB_MODE WANDB_ENTITY=$WANDB_ENTITY"
  echo "[seq] RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=$RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S"
   echo "[seq] GPUs visible to this step:"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>&1
  echo "[seq] ===================================================="
} | tee -a "$SEQLOG"

declare -A RESULT
for cfg in "${CONFIGS[@]}"; do
  name=${cfg%.toml}
  echo "[seq] ===== START $name $(date +%F_%H:%M:%S) =====" | tee -a "$SEQLOG"
  uv run --no-sync rl @ "$CFGDIR/$cfg" 2>&1 | tee -a "$SEQLOG"
  rc=${PIPESTATUS[0]}
  RESULT[$name]=$rc
  echo "[seq] ===== END   $name rc=$rc $(date +%F_%H:%M:%S) =====" | tee -a "$SEQLOG"
done

{
  echo "[seq] ===================== SUMMARY ====================="
  for cfg in "${CONFIGS[@]}"; do name=${cfg%.toml}; echo "[seq]   $name : rc=${RESULT[$name]}"; done
  echo "[seq] done=$(date +%Y%m%d-%H%M%S)"
} | tee -a "$SEQLOG"

# Record where the console log lives so the monitor can find it.
echo "$SEQLOG" > $ERLM/.research/EfficientRLM/outputs/.last_smoke_seq_log
