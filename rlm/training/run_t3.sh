#!/bin/bash
# t2 A/B arm launcher: run_t2.sh {control|treatment}
# Runs the arm to step 200 with resume-on-crash, then uploads the LoRA adapter
# REGARDLESS of exit code (the other cluster's arm finished step 200 but hung
# on its final evals, exited dirty, and the rc-gated upload never ran).
# Stuck-eval defense: PRIME_RL_ROLLOUT_TIMEOUT_S dispatcher sweep reaps stuck
# individual rollouts (2h) WITHOUT killing the run. NO run-level hard stops:
# steps take 30-60 min on this hardware; 200 steps = 4-8 days.
set -uo pipefail
# t3-mitigated: fixed identity, no ARM arg

ERLM=/scratch/omeerdogan23/erlm
EFF=$ERLM/.research/EfficientRLM
PRL=$ERLM/prime-rl
CFG=$EFF/rlm/training/configs/qwen3-30b-t3-mitigated.toml
OUT=$EFF/outputs/qwen3-30b-t3-mitigated
mkdir -p "$OUT"

module load cuda/12.8.0 git/2.9.5 >/dev/null 2>&1
[ -f "$ERLM/rlm/.env" ] && { set -a; source "$ERLM/rlm/.env"; set +a; }
export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/tmp/uvcache_$USER
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export WANDB_MODE=online WANDB_ENTITY=omeerdogan-koc-university
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=120
export PRIME_RL_ROLLOUT_TIMEOUT_S=7200
export HF_ADAPTER_REPO="oerdogan/erlm-qwen3-30b-t3-mitigated"
export PYTHONPATH="$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong:$EFF/rlm/training/environments/oolong_pairs:$EFF/rlm/training/environments/longbench_codeqa${PYTHONPATH:+:$PYTHONPATH}"

cd "$PRL" || exit 1
.venv/bin/python - <<'PYGATE' || exit 1
import importlib
for m in ("rlm", "rlm_train.env", "rlm_train.adaptive_group", "browsecomp_plus.env", "oolong.env", "longbench_codeqa.env"):
    mod = importlib.import_module(m)
    f = getattr(mod, "__file__", "?")
    assert "/EfficientRLM/" in str(f), f"{m} resolves outside the EfficientRLM tree: {f}"
import rlm_train.adaptive_group as ag
assert "iterations_ln_excess" in ag._COST_BASES, "t3 basis missing from resolved operator"
print("[gate] imports resolve to EfficientRLM tree; iterations_ln_excess registered")
PYGATE

FINAL_CKPT="$OUT/checkpoints/step_200"

attempt=0
while [ ! -d "$FINAL_CKPT" ] && [ $attempt -lt 8 ]; do
  attempt=$((attempt+1))
  RESUME_ARGS=()
  ls "$OUT/checkpoints" >/dev/null 2>&1 && [ -n "$(ls -A "$OUT/checkpoints" 2>/dev/null)" ] && RESUME_ARGS=(--ckpt.resume-step -1)
  echo "[t3-mitigated] ===== attempt $attempt $(date +%F_%H:%M:%S) resume=${RESUME_ARGS[*]:-fresh} ====="
  uv run --no-sync rl @ "$CFG" --output-dir "$OUT" --wandb.name "qwen3-30b-t3-mitigated" "${RESUME_ARGS[@]}" 2>&1
  rc=$?
  echo "[t3-mitigated] ===== rl exited rc=$rc $(date +%F_%H:%M:%S) ====="
  [ -d "$FINAL_CKPT" ] && break
  sleep 60
done

if [ -d "$FINAL_CKPT" ]; then
  echo "[t3-mitigated] training complete; uploading adapter (rc-independent)"
  if [ -n "${HF_TOKEN:-}" ]; then
    "$PRL/.venv/bin/python" "$ERLM/scripts/upload_lora_to_hf.py" \
      --run-dir "$OUT" --repo-id "$HF_ADAPTER_REPO" --private \
      && echo "[t3-mitigated] adapter uploaded to $HF_ADAPTER_REPO" \
      || echo "[t3-mitigated] WARNING: adapter upload failed - rerun upload_lora_to_hf.py manually"
  else
    echo "[t3-mitigated] WARNING: HF_TOKEN unset - upload skipped"
  fi
  touch "$OUT/ARM_DONE"
  echo "[t3-mitigated] ARM_DONE"
else
  echo "[t3-mitigated] FAILED: no step-200 checkpoint after $attempt attempts"
  touch "$OUT/ARM_FAILED"
fi
