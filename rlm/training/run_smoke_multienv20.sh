#!/bin/bash
# BC+-only MULTIENV 20-step gs4 smoke on the EfficientRLM tree (group_size 8,
# max_iterations 12, sub_max_tokens 2048, context_chunk_chars 12000 on train).
# Runtime = the external patched prime-rl venv (EfficientRLM/prime-rl stays pristine);
# PYTHONPATH shadows the venv's old editable rlm_train/env packages with THIS tree's
# code. Optional 1st arg: checkpoint step to resume from (-1 = latest).
# Prefer launching via supervise_smoke_bcplus_lean.sh (survives the 01:00 reaper).

RESUME="${1:-}"

ERLM=/scratch/omeerdogan23/erlm
EFF=$ERLM/.research/EfficientRLM
PRL=$ERLM/prime-rl   # patched runtime venv (dispatcher deadline sweep)
CFG=$EFF/rlm/training/configs/smoke-qwen3-30b-multienv-20step-gs4.toml
mkdir -p $EFF/outputs

module load cuda/12.8.0 git/2.9.5 >/dev/null 2>&1

# credentials (WANDB_API_KEY, HF_TOKEN, OPENAI_API_KEY for the judge)
[ -f "$ERLM/rlm/.env" ] && { set -a; source "$ERLM/rlm/.env"; set +a; }

export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/tmp/uvcache_$USER
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1        # ai16 A6000: no NVLink
export WANDB_MODE=online WANDB_ENTITY=omeerdogan-koc-university
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=120
export PRIME_RL_ROLLOUT_TIMEOUT_S=7200             # dispatcher deadline sweep (patched venv)

# shadow the venv's old editable packages with the EfficientRLM tree
export PYTHONPATH="$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong:$EFF/rlm/training/environments/oolong_pairs:$EFF/rlm/training/environments/longbench_codeqa${PYTHONPATH:+:$PYTHONPATH}"

cd "$PRL" || { echo "[smoke] cannot cd $PRL"; exit 1; }

# hard gate: refuse to run if imports do not resolve to the EfficientRLM tree
.venv/bin/python - <<'PYGATE' || exit 1
import importlib
mods = ["rlm", "rlm.utils.prompts", "rlm.utils.parsing",
        "rlm_train", "rlm_train.env", "rlm_train.proxy", "rlm_train.rubric",
        "rlm_train.adaptive_group", "rlm_train.worker", "rlm_train.repl.subprocess",
        "browsecomp_plus", "browsecomp_plus._data", "browsecomp_plus.env",
        "oolong", "oolong.env", "oolong_pairs.env", "longbench_codeqa.env"]
for m in mods:
    mod = importlib.import_module(m)
    f = getattr(mod, "__file__", None) or str(list(getattr(mod, "__path__", ["?"]))[0])
    assert "/EfficientRLM/" in str(f), f"{m} resolves OUTSIDE EfficientRLM: {f}"
print(f"[smoke] import gate: all {len(mods)} project modules resolve into EfficientRLM")
PYGATE

RESUME_ARGS=()
if [ -n "$RESUME" ]; then RESUME_ARGS=(--ckpt.resume-step "$RESUME"); fi

echo "[smoke] ===== START multienv-20step-gs4 $(date +%F_%H:%M:%S) host=$(hostname) resume=${RESUME:-fresh} ====="
uv run --no-sync rl @ "$CFG" "${RESUME_ARGS[@]}" 2>&1
rc=$?
echo "[smoke] ===== END rc=$rc $(date +%F_%H:%M:%S) ====="
exit $rc
