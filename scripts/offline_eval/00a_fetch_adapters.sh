#!/usr/bin/env bash
# Pull LoRA adapters from HF into scripts/offline_eval/adapters/.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p adapters
: "${HF_TOKEN:?export HF_TOKEN first}"
# huggingface-cli is deprecated/non-functional on recent huggingface_hub; use `hf`.
HFC=$(command -v hf || echo ../../.venv-eval/bin/hf)
# t2T_120, t2T_final are uploaded and live. AUTHORS_REPO is ALSO a LoRA on this same
# base model (Qwen/Qwen3-30B-A3B-Instruct-2507, confirmed via its adapter_config.json),
# not a standalone model -- it gets served as a third lora-module, not a separate server.
T_FINAL_REPO="${T_FINAL_REPO:-oerdogan/qwen3-30b-t2-treatment-lora-step200}"
T_120_REPO="${T_120_REPO:-oerdogan/qwen3-30b-t2-treatment-lora-step120}"
AUTHORS_REPO="${AUTHORS_REPO:-mit-oasys/rlm-qwen3-30b-a3b-v0.1}"
[ -n "$T_120_REPO" ] && $HFC download "$T_120_REPO" --local-dir adapters/t2T_120
[ -n "$AUTHORS_REPO" ] && $HFC download "$AUTHORS_REPO" --local-dir adapters/authors
if [ -n "$T_FINAL_REPO" ]; then
  $HFC download "$T_FINAL_REPO" --local-dir adapters/t2T_final \
    || echo "t2T_final not available yet ($T_FINAL_REPO) — training may still be in progress"
fi
ls -la adapters/
