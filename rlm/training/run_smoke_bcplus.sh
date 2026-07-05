#!/bin/bash
# BC+-only 40-step smoke on the EfficientRLM tree.
# Runtime = the external patched prime-rl venv (EfficientRLM/prime-rl stays pristine);
# PYTHONPATH shadows the venv's old editable rlm_train/env packages with THIS tree's
# code, so the smoke exercises the restructured browsecomp_plus + ported rlm_train.
# Launch (login node, after 01:00 reaper):
#   setsid nohup srun --jobid=<HOLDER> --overlap --gres=gpu:rtx_a6000:8 -n1 \
#     --cpus-per-task=48 --mem=0 --time=23:00:00 \
#     bash /scratch/omeerdogan23/erlm/.research/EfficientRLM/rlm/training/run_smoke_bcplus.sh \
#     > <logfile> 2>&1 &

ERLM=/scratch/omeerdogan23/erlm
EFF=$ERLM/.research/EfficientRLM
PRL=$ERLM/prime-rl   # patched runtime venv (dispatcher deadline sweep)
CFG=$EFF/rlm/training/configs/smoke-qwen3-30b-bcplus-only-40step.toml
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
export PYTHONPATH="$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong${PYTHONPATH:+:$PYTHONPATH}"

cd "$PRL" || { echo "[smoke] cannot cd $PRL"; exit 1; }

# hard gate: refuse to run if imports do not resolve to the EfficientRLM tree
.venv/bin/python - <<'PY' || exit 1
import rlm_train, browsecomp_plus, rlm_train.adaptive_group as ag
from browsecomp_plus import _data
for name, mod in [("rlm_train", rlm_train), ("browsecomp_plus", browsecomp_plus)]:
    path = mod.__file__
    assert "/EfficientRLM/" in path, f"{name} resolves OUTSIDE EfficientRLM: {path}"
    print(f"[smoke] {name} -> {path}")
print("[smoke] adaptive_group ->", ag.__file__)
print("[smoke] browsecomp_plus._data ->", _data.__file__, "(restructured layout confirmed)")
PY

echo "[smoke] ===== START bcplus-only-40step $(date +%F_%H:%M:%S) host=$(hostname) ====="
uv run --no-sync rl @ "$CFG" 2>&1
rc=$?
echo "[smoke] ===== END rc=$rc $(date +%F_%H:%M:%S) ====="
exit $rc
