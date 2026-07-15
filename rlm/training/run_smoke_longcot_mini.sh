#!/bin/bash
# longcot_mini 20-step wiring smoke on the EfficientRLM tree (single-env mix,
# neutral beta_max=0.0 lever; see configs/qwen3-30b-longcot-mini-smoke.toml).
# Runtime = the external patched prime-rl venv (EfficientRLM/prime-rl stays pristine);
# PYTHONPATH shadows the venv's old editable rlm_train/env packages with THIS tree's
# code. The official `longcot` data+verifier package CANNOT be PYTHONPATH-shadowed
# (its repo maps package-dir longcot=src) — it must be pip-installed into the
# runtime venv once; the import gate below checks that and prints the command.
# Optional 1st arg: checkpoint step to resume from (-1 = latest).

RESUME="${1:-}"

ERLM=/scratch/omeerdogan23/erlm
EFF=$ERLM/.research/EfficientRLM
PRL=$ERLM/prime-rl   # patched runtime venv (dispatcher deadline sweep)
CFG=$EFF/rlm/training/configs/qwen3-30b-longcot-mini-smoke.toml
mkdir -p $EFF/outputs

module load cuda/12.8.0 git/2.9.5 >/dev/null 2>&1

# credentials (WANDB_API_KEY, HF_TOKEN; no judge key needed — scoring is the
# official deterministic verifier, enable_fallback=false)
[ -f "$ERLM/rlm/.env" ] && { set -a; source "$ERLM/rlm/.env"; set +a; }

export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR=/tmp/uvcache_$USER
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1        # ai16 A6000: no NVLink
export WANDB_MODE=online WANDB_ENTITY=omeerdogan-koc-university
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=120
export PRIME_RL_ROLLOUT_TIMEOUT_S=7200             # dispatcher deadline sweep (patched venv)

# shadow the venv's old editable packages with the EfficientRLM tree
export PYTHONPATH="$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/longcot_mini${PYTHONPATH:+:$PYTHONPATH}"

cd "$PRL" || { echo "[smoke] cannot cd $PRL"; exit 1; }

# hard gate: refuse to run if imports do not resolve to the EfficientRLM tree,
# or if the pinned longcot package (data + official verifiers) is missing.
.venv/bin/python - <<'PYGATE' || exit 1
import importlib
mods = ["rlm", "rlm.utils.prompts", "rlm.utils.parsing",
        "rlm_train", "rlm_train.env", "rlm_train.proxy", "rlm_train.rubric",
        "rlm_train.adaptive_group", "rlm_train.worker", "rlm_train.repl.subprocess",
        "longcot_mini", "longcot_mini.env"]
try:
    import longcot
except ImportError as e:
    raise SystemExit(
        f"[smoke] `longcot` not installed in the runtime venv: {e}\n"
        "  fix: uv pip install 'longcot @ git+https://github.com/LongHorizonReasoning/"
        "longcot.git@fb9649423f15f5b0091f8e988b100596cac592ca'"
    )
n = len(longcot.load_questions(domain="logic", difficulty="easy"))
assert n > 0, "longcot package imported but has no bundled data (broken wheel?)"
for m in mods:
    mod = importlib.import_module(m)
    f = getattr(mod, "__file__", None) or str(list(getattr(mod, "__path__", ["?"]))[0])
    assert "/EfficientRLM/" in str(f), f"{m} resolves OUTSIDE EfficientRLM: {f}"
print(f"[smoke] import gate: {len(mods)} project modules in EfficientRLM; longcot data ok ({n} logic/easy)")
PYGATE

RESUME_ARGS=()
if [ -n "$RESUME" ]; then RESUME_ARGS=(--ckpt.resume-step "$RESUME"); fi

echo "[smoke] ===== START longcot-mini-smoke $(date +%F_%H:%M:%S) host=$(hostname) resume=${RESUME:-fresh} ====="
uv run --no-sync rl @ "$CFG" "${RESUME_ARGS[@]}" 2>&1
rc=$?
echo "[smoke] ===== END rc=$rc $(date +%F_%H:%M:%S) ====="
exit $rc
