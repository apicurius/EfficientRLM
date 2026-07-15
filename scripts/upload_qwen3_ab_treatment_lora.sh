#!/usr/bin/env bash
set -euo pipefail

# Upload the final Qwen3 AB treatment LoRA adapter from a completed local run.
# Can be run manually after training finishes or after a session restart.
#
# Usage:
#   bash scripts/upload_qwen3_ab_treatment_lora.sh
#
# Required:
#   HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment or rlm/.env
# Optional:
#   HF_LORA_REPO_ID=namespace/repo
#   HF_LORA_PRIVATE=1   # default, set 0 for public
#   HF_LORA_COMMIT_PREFIX="Upload final LoRA adapter"

BASE="${BASE:-/teamspace/s3_folders/outputs/efficientrlm}"
LOCAL_BASE="${LOCAL_BASE:-/teamspace/studios/this_studio/.cache/efficientrlm}"
DL="${DL:-$LOCAL_BASE/downloads}"
EFF="${EFF:-/teamspace/studios/this_studio/EfficientRLM}"
CONFIG="${CONFIG:-training/configs/qwen3-30b-ab-treatment-multienv-200step.toml}"
export BASE LOCAL_BASE DL EFF CONFIG

cd "$EFF/rlm"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HOME="${HOME:-$LOCAL_BASE/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DL/xdg-cache}"
export HF_HOME="${HF_HOME:-$DL/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$DL/hf/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DL/hf/hub}"
# The xet uploader writes cache/log files to a default path that is often not
# writable on this host (observed: Permission denied under /scratch/.../hf_cache/xet).
# Disable xet so uploads fall back to regular LFS, and also point any xet paths
# at a writable location as a belt-and-suspenders safeguard.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_CACHE="${HF_XET_CACHE:-$DL/hf/xet}"
export TMPDIR="${TMPDIR:-$LOCAL_BASE/tmp}"
mkdir -p "$LOCAL_BASE"/hf-upload "$DL"/hf/hub "$DL"/hf/xet "$LOCAL_BASE"/tmp

VENV_PY="$EFF/prime-rl/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: prime-rl venv missing: $VENV_PY" >&2
  echo "Create it with: cd $EFF/prime-rl && UV_CACHE_DIR=$DL/uv-cache TMPDIR=$TMPDIR uv sync --extra flash-attn" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "ERROR: HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. Cannot upload LoRA adapter." >&2
  exit 1
fi

printf '=== uploading final LoRA adapter to Hugging Face ===
'
"$VENV_PY" - <<'PY_UPLOAD'
import os
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

base = Path(os.environ.get("BASE", "/teamspace/s3_folders/outputs/efficientrlm"))
local_base = Path(os.environ.get("LOCAL_BASE", "/teamspace/studios/this_studio/.cache/efficientrlm"))
eff = Path(os.environ.get("EFF", "/teamspace/studios/this_studio/EfficientRLM"))
config_rel = os.environ.get("CONFIG", "training/configs/qwen3-30b-ab-treatment-multienv-200step.toml")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
private = os.environ.get("HF_LORA_PRIVATE", "1") not in {"0", "false", "False", "no", "NO"}
commit_prefix = os.environ.get("HF_LORA_COMMIT_PREFIX", "Upload final LoRA adapter")

import tomllib
cfg = tomllib.loads((eff / "rlm" / config_rel).read_text())
out = Path(os.environ.get("RUN_OUTPUT_DIR") or cfg["output_dir"])
search_roots = [out]
ckpt_out = cfg.get("ckpt", {}).get("output_dir")
if ckpt_out:
    search_roots.append(Path(ckpt_out))
trainer_ckpt_out = cfg.get("trainer", {}).get("ckpt", {}).get("output_dir")
if trainer_ckpt_out:
    search_roots.append(Path(trainer_ckpt_out))

repo_id = os.environ.get("HF_LORA_REPO_ID")
if not repo_id:
    api = HfApi(token=token)
    who = api.whoami(token=token)
    namespace = os.environ.get("HF_LORA_NAMESPACE") or who.get("name")
    lora_name = (
        cfg.get("orchestrator", {})
        .get("model", {})
        .get("lora", {})
        .get("name")
    ) or cfg.get("wandb", {}).get("name") or out.name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", lora_name).strip("-._")
    repo_id = f"{namespace}/{safe_name}"
    print(f"HF_LORA_REPO_ID not set; derived repo_id={repo_id}")

candidates = []
for root in search_roots:
    for cfg_file in root.glob("configs/orchestrator*.toml"):
        try:
            ocfg = tomllib.loads(cfg_file.read_text())
            orch_out = Path(ocfg.get("output_dir", root / "run_default"))
        except Exception:
            orch_out = root / "run_default"
        candidates.extend([p.parent for p in orch_out.rglob("adapter_config.json")])
    candidates.extend([p.parent for p in root.rglob("adapter_config.json")])
uniq = []
seen = set()
for p in candidates:
    rp = p.resolve()
    if rp not in seen:
        seen.add(rp)
        uniq.append(p)

if not uniq:
    raise SystemExit(
        f"No LoRA adapter directory found under any of {search_roots}. Expected adapter_config.json "
        "from prime-rl broadcast or lora_adapters checkpoint output."
    )

def step_num(path: Path) -> int:
    nums = []
    for part in path.parts:
        m = re.fullmatch(r"step_(\d+)", part)
        if m:
            nums.append(int(m.group(1)))
    return nums[-1] if nums else -1

adapter_dir = max(uniq, key=lambda p: (step_num(p), p.stat().st_mtime))
step = step_num(adapter_dir)
stage = local_base / "hf-upload" / "lora-adapter"
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)

# Copy non-weight metadata first. Then rewrite adapter weights into PEFT-compatible
# keys if the source is a prime-rl filesystem-broadcast adapter. Broadcast dirs
# use keys like `model.layers...lora_A.weight`; PEFT expects
# `base_model.model.model.layers...lora_A.weight`. The weight-checkpoint
# `lora_adapters` path already includes the `base_model.model.` prefix.
for src in adapter_dir.iterdir():
    if src.is_file() and src.name not in {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }:
        shutil.copy2(src, stage / src.name)

from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
weight_file = adapter_dir / "adapter_model.safetensors"
if not weight_file.exists():
    raise SystemExit(f"Expected adapter_model.safetensors in {adapter_dir}")
state = safe_load_file(str(weight_file))
converted = {}
changed = False
for k, v in state.items():
    if k.startswith("base_model.model."):
        nk = k
    else:
        nk = "base_model.model." + k
        changed = True
    converted[nk] = v
safe_save_file(converted, str(stage / "adapter_model.safetensors"), metadata={"format": "pt"})

# Add a lightweight model card if one is not present.
readme = stage / "README.md"
if not readme.exists():
    readme.write_text(
        "---\n"
        "library_name: peft\n"
        "base_model: Qwen/Qwen3-30B-A3B-Instruct-2507\n"
        "tags:\n"
        "- peft\n"
        "- lora\n"
        "---\n\n"
        "# Qwen3 30B AB Treatment LoRA Adapter\n\n"
        f"Source artifact: `{adapter_dir}`\n\n"
        f"Training step: `{step if step >= 0 else 'unknown'}`\n\n"
        "This repository contains only the LoRA adapter artifacts, not the base model.\n"
    )

total = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
print(f"Adapter source: {adapter_dir}")
print(f"Adapter step: {step if step >= 0 else 'unknown'}")
print(f"Adapter staged at: {stage}")
print(f"Adapter size: {total / (1024**2):.1f} MiB")
print(f"PEFT key prefix conversion applied: {changed}")
if total > 1024**3:
    raise SystemExit("Refusing to upload >1GiB; this does not look like adapter-only artifacts.")

create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")
msg = f"{commit_prefix}"
if step >= 0:
    msg += f" (step {step})"
api = HfApi(token=token)
url = upload_folder(
    repo_id=repo_id,
    folder_path=str(stage),
    path_in_repo=".",
    commit_message=msg,
    token=token,
    repo_type="model",
)
print(f"Uploaded LoRA adapter to HF repo: {repo_id}")
print(f"Commit URL: {url}")
PY_UPLOAD
