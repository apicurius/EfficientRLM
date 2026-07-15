#!/usr/bin/env bash
set -euo pipefail

# EfficientRLM qwen3-30b AB treatment launcher.
# Usage:
#   bash scripts/run_qwen3_ab_treatment.sh
#   bash scripts/run_qwen3_ab_treatment.sh --dry-run
#
# Persistent prime-rl outputs are kept under the repo's own outputs/ dir
# (Studio-persistent storage, survives machine-type switches and reboots).
# Ephemeral caches/tmp/W&B files stay on the local Studio filesystem because the
# S3-backed litfs mount does not support all file-lock/tempfile operations used by uv.
# The actual prime-rl output_dir is set in the TOML config.

BASE="${BASE:-/teamspace/studios/this_studio/EfficientRLM/outputs}"
LOCAL_BASE="${LOCAL_BASE:-/teamspace/studios/this_studio/.cache/efficientrlm}"
DL="${DL:-$LOCAL_BASE/downloads}"
EFF="${EFF:-/teamspace/studios/this_studio/EfficientRLM}"
CONFIG="${CONFIG:-training/configs/qwen3-30b-ab-treatment-multienv-200step.toml}"
WANDB_ENTITY="${WANDB_ENTITY:-omeerdogan-koc-university}"
DO_WANDB_LOGIN="${DO_WANDB_LOGIN:-1}"
UPLOAD_LORA="${UPLOAD_LORA:-1}"
# Resume support: RESUME=1 resumes from a checkpoint (RESUME_STEP=-1 = latest).
# Trainer checkpoints live under [ckpt].output_dir (S3); orchestrator checkpoints
# live under the local output_dir/run_default. Both must be intact at that step.
RESUME="${RESUME:-0}"
RESUME_STEP="${RESUME_STEP:--1}"
WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$LOCAL_BASE/wandb-shared-run-id}"
export BASE LOCAL_BASE DL EFF CONFIG

ensure_teamspace_base() {
  if [[ -d "$BASE" && -w "$BASE" ]]; then
    return 0
  fi

  if mkdir -p "$BASE" 2>/dev/null && [[ -w "$BASE" ]]; then
    return 0
  fi

  # /teamspace and /teamspace/s3_folders are often root-owned. If passwordless
  # sudo is available, create/chown only our experiment base dir. Otherwise,
  # print the exact one-time setup command for the user/admin.
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    echo "Creating writable Teamspace base with sudo: $BASE"
    sudo mkdir -p "$BASE"
    sudo chown -R "$(id -u):$(id -g)" "$BASE"
  fi

  if [[ ! -d "$BASE" || ! -w "$BASE" ]]; then
    cat >&2 <<EOF
ERROR: Cannot create/write Teamspace base directory:
  $BASE

/teamspace/s3_folders is not user-writable on this machine. Run this one-time
setup command, then re-run this script:

  sudo mkdir -p "$BASE" && sudo chown -R "$(id -un):$(id -gn)" "$BASE"

Or choose an already writable Teamspace Drive directory:

  BASE=/teamspace/s3_folders/<your-existing-writable-dir> bash scripts/run_qwen3_ab_treatment.sh
EOF
    exit 1
  fi
}

ensure_teamspace_base

mkdir -p \
  "$DL"/{hf/hub,hf/datasets,hf/assets,hf/modules,hf/xet,hf/transformers,uv-cache,pip-cache,torch,triton,torchinductor,cuda-cache,vllm/assets,vllm/media,flashinfer,lora,xdg-cache,wandb-cache,matplotlib,numba} \
  "$LOCAL_BASE"/{wandb,wandb-config,tmp,home/.config/vllm,home/.local/share,hf-upload,prime-rl} \
  "$BASE"/{prime-rl,checkpoints}

# Keep W&B resumes on the same shared run id across interrupted launches.
# If WANDB_SHARED_RUN_ID is explicitly set, persist it. Otherwise reuse the
# persisted id when present; if none exists, generate and persist a new one.
if [[ -n "${WANDB_SHARED_RUN_ID:-}" ]]; then
  printf '%s\n' "$WANDB_SHARED_RUN_ID" > "$WANDB_RUN_ID_FILE"
elif [[ -s "$WANDB_RUN_ID_FILE" ]]; then
  WANDB_SHARED_RUN_ID="$(cat "$WANDB_RUN_ID_FILE")"
else
  WANDB_SHARED_RUN_ID="$(python3 - <<'PY_RUN_ID'
import uuid
print(uuid.uuid4().hex)
PY_RUN_ID
)"
  printf '%s\n' "$WANDB_SHARED_RUN_ID" > "$WANDB_RUN_ID_FILE"
fi
export WANDB_SHARED_RUN_ID
printf 'WANDB_SHARED_RUN_ID=%s\n' "$WANDB_SHARED_RUN_ID"

cd "$EFF/rlm"

# Load secrets such as WANDB_API_KEY, OPENAI_API_KEY, HF_TOKEN, PRIME_* from .env.
# Do not print them.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HOME="$LOCAL_BASE/home"
export XDG_CACHE_HOME="$DL/xdg-cache"
export XDG_CONFIG_HOME="$LOCAL_BASE/home/.config"
export XDG_DATA_HOME="$LOCAL_BASE/home/.local/share"

export HF_HOME="$DL/hf"
export HF_HUB_CACHE="$DL/hf/hub"
export HUGGINGFACE_HUB_CACHE="$DL/hf/hub"
export HF_DATASETS_CACHE="$DL/hf/datasets"
export HF_ASSETS_CACHE="$DL/hf/assets"
export HF_MODULES_CACHE="$DL/hf/modules"
export HF_XET_CACHE="$DL/hf/xet"
export TRANSFORMERS_CACHE="$DL/hf/transformers"

export UV_CACHE_DIR="$DL/uv-cache"
export PIP_CACHE_DIR="$DL/pip-cache"

export TORCH_HOME="$DL/torch"
export TRITON_CACHE_DIR="$DL/triton"
export TORCHINDUCTOR_CACHE_DIR="$DL/torchinductor"
export CUDA_CACHE_PATH="$DL/cuda-cache"

export VLLM_CACHE_ROOT="$DL/vllm"
export VLLM_CONFIG_ROOT="$LOCAL_BASE/home/.config/vllm"
export VLLM_ASSETS_CACHE="$DL/vllm/assets"
export VLLM_MEDIA_CACHE="$DL/vllm/media"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$DL/flashinfer"
export VLLM_LORA_RESOLVER_CACHE_DIR="$DL/lora"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="$WANDB_ENTITY"
export WANDB_DIR="$LOCAL_BASE/wandb"
export WANDB_CACHE_DIR="$DL/wandb-cache"
export WANDB_CONFIG_DIR="$LOCAL_BASE/wandb-config"

export TMPDIR="$LOCAL_BASE/tmp"
export TEMP="$LOCAL_BASE/tmp"
export TMP="$LOCAL_BASE/tmp"
export MPLCONFIGDIR="$DL/matplotlib"
export NUMBA_CACHE_DIR="$DL/numba"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S="${RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S:-120}"

# Critical: fixes `Failed to spawn: rl` when launching from EFF/rlm.
export UV_PROJECT="$EFF/prime-rl"

# Local package/environment import path fallback; envs are also installed editable in prime-rl venv.
export PYTHONPATH="$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong:$EFF/rlm/training/environments/oolong_pairs:$EFF/rlm/training/environments/longbench_codeqa${PYTHONPATH:+:$PYTHONPATH}"

VENV_PY="$EFF/prime-rl/.venv/bin/python"

repair_env_installs() {
  echo "Registering local RLM/verifiers environments in prime-rl venv..."
  uv pip install --python "$VENV_PY" --no-deps \
    -e "$EFF/rlm" \
    -e "$EFF/rlm/training" \
    -e "$EFF/rlm/training/environments/oolong" \
    -e "$EFF/rlm/training/environments/browsecomp_plus" \
    -e "$EFF/rlm/training/environments/oolong_pairs" \
    -e "$EFF/rlm/training/environments/longbench_codeqa" >/dev/null
}

printf '=== EfficientRLM Teamspace launch ===\n'
printf 'EFF=%s\nCONFIG=%s\nBASE=%s\nLOCAL_BASE=%s\nDL=%s\nUV_PROJECT=%s\nWANDB_ENTITY=%s\n' "$EFF" "$CONFIG" "$BASE" "$LOCAL_BASE" "$DL" "$UV_PROJECT" "$WANDB_ENTITY"

printf '\n=== prime-rl pin ===\n'
git -C "$EFF/prime-rl" describe --tags --always --dirty

if [[ ! -x "$VENV_PY" ]]; then
  cat >&2 <<EOF
ERROR: prime-rl venv is missing:
  $VENV_PY

Create it first with the local uv cache, then re-run this launcher:

  cd $EFF/prime-rl && UV_CACHE_DIR=$UV_CACHE_DIR TMPDIR=$TMPDIR uv sync --extra flash-attn
EOF
  exit 1
fi

printf '\n=== registering prime-rl environments ===\n'
repair_env_installs
"$VENV_PY" - <<'PY'
from importlib.metadata import entry_points
try:
    eps = entry_points(group='verifiers.environments')
except TypeError:
    eps = entry_points().get('verifiers.environments', [])
names = sorted(e.name for e in eps)
need = {'oolong', 'browsecomp_plus'}
print('registered verifiers environments:', names)
missing = sorted(need - set(names))
if missing:
    raise SystemExit('Missing verifiers environments after registration: ' + ', '.join(missing))
PY

printf '\n=== dependency/import checks ===\n'
if ! "$VENV_PY" - <<'PY'
import importlib.util
missing = [m for m in ['flash_attn', 'ring_flash_attn'] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('Missing modules: ' + ', '.join(missing))
print('flash_attn OK')
print('ring_flash_attn OK')
PY
then
  echo "Missing flash_attn/ring_flash_attn. Run: cd $EFF/prime-rl && UV_CACHE_DIR=$UV_CACHE_DIR uv sync --extra flash-attn"
  exit 1
fi

if [[ "$DO_WANDB_LOGIN" == "1" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    printf '\n=== wandb login/status ===\n'
    "$VENV_PY" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || {
      echo "W&B login failed. Check WANDB_API_KEY in $EFF/rlm/.env."
      exit 1
    }
    "$VENV_PY" -m wandb status 2>&1 | python -c 'import sys
for line in sys.stdin:
    low=line.lower()
    if "api" in low or "key" in low or "token" in low:
        continue
    print(line, end="")
'
  else
    echo "WANDB_API_KEY not set; skipping W&B login."
  fi
fi

upload_lora_adapter_to_hf() {
  if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
    echo "HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set; skipping LoRA adapter upload."
    echo "After training, upload manually with: bash scripts/upload_qwen3_ab_treatment_lora.sh"
    return 0
  fi

  bash "$EFF/scripts/upload_qwen3_ab_treatment_lora.sh"
}

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  printf '\n=== resume requested ===\n'

  if [[ -z "${RUN_OUTPUT_DIR:-}" ]]; then
    RUN_OUTPUT_DIR="$("$VENV_PY" - "$EFF/rlm/$CONFIG" <<'PYCFG'
import sys, tomllib
from pathlib import Path
cfg = tomllib.loads(Path(sys.argv[1]).read_text())
print(cfg["output_dir"])
PYCFG
)"
  fi
  printf 'Using run output dir for resume: %s\n' "$RUN_OUTPUT_DIR"
  LOCAL_RUN_DIR="$RUN_OUTPUT_DIR"

  # If local orchestrator checkpoints are missing (e.g. .cache was lost),
  # restore the latest S3 state snapshot before resuming.
  if ! ls -d "$LOCAL_RUN_DIR"/run_default/checkpoints/step_* >/dev/null 2>&1; then
    printf 'No local checkpoints found; restoring latest S3 backup...\n'
    RUN_NAME="$(basename "$RUN_OUTPUT_DIR")" LOCAL_RUN_DIR="$LOCAL_RUN_DIR" \
      bash "$EFF/scripts/restore_qwen3_ab_treatment_state.sh" || {
        echo "ERROR: could not restore state from S3; cannot resume." >&2; exit 1; }
  fi

  # Choose a resume step present in ALL of: S3 trainer checkpoints, local
  # orchestrator checkpoints, and local broadcasts. Avoids step mismatch.
  if [[ "$RESUME_STEP" == "-1" ]]; then
    RESUME_STEP="$("$VENV_PY" - "$EFF/rlm/$CONFIG" "$LOCAL_RUN_DIR" <<'PYSTEP'
import sys, tomllib, re
from pathlib import Path
cfg = tomllib.loads(Path(sys.argv[1]).read_text())
local = Path(sys.argv[2])
# cfg["output_dir"] is the static config default; an --output-dir CLI override
# (e.g. teamspace launchers) is not reflected there, but LOCAL_RUN_DIR always is.
ckpt_out = cfg.get("ckpt", {}).get("output_dir") or str(local)
def steps(d):
    d = Path(d)
    if not d.is_dir():
        return set()
    out = set()
    for c in d.iterdir():
        m = re.fullmatch(r"step_(\d+)", c.name)
        if m and c.is_dir():
            out.add(int(m.group(1)))
    return out
trainer = steps(Path(ckpt_out) / "checkpoints")
orch = steps(local / "run_default" / "checkpoints")
bcast = steps(local / "run_default" / "broadcasts")
common = trainer & orch & bcast
print(max(common) if common else -1)
PYSTEP
)"
    if [[ -z "$RESUME_STEP" || "$RESUME_STEP" == "-1" ]]; then
      echo "ERROR: no consistent checkpoint step across trainer(S3)/orchestrator/broadcasts." >&2
      exit 1
    fi
    printf 'Resolved consistent resume step: %s\n' "$RESUME_STEP"
  fi

  EXTRA_ARGS+=(--ckpt.resume-step "$RESUME_STEP")
  printf '\n=== resume enabled (ckpt.resume-step=%s) ===\n' "$RESUME_STEP"
fi

printf '\n=== starting command ===\n'
set +e
set -x
uv run --no-sync rl @ "$CONFIG" "${EXTRA_ARGS[@]}" "$@"
run_rc=$?
set +x
set -e

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    echo "Dry-run detected; skipping LoRA HF upload."
    exit "$run_rc"
  fi
done

if [[ "$run_rc" == "0" ]]; then
  if [[ "$UPLOAD_LORA" == "1" ]]; then
    upload_lora_adapter_to_hf
  else
    echo "UPLOAD_LORA=$UPLOAD_LORA; skipping LoRA HF upload."
  fi
fi
exit "$run_rc"
