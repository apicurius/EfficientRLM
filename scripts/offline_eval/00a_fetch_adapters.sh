#!/usr/bin/env bash
# Pull LoRA adapters from HF into scripts/offline_eval/adapters/.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p adapters
: "${HF_TOKEN:?export HF_TOKEN first}"
HFC=$(command -v huggingface-cli || echo ../../.venv-eval/bin/huggingface-cli)
T_FINAL_REPO="${T_FINAL_REPO:-}"    # e.g. oerdogan/qwen3-30b-t2-treatment-lora-step200
T_120_REPO="${T_120_REPO:-oerdogan/qwen3-30b-t2-treatment-lora-step120}"
[ -n "$T_FINAL_REPO" ] && $HFC download "$T_FINAL_REPO" --local-dir adapters/t2T_final
[ -n "$T_120_REPO" ]   && $HFC download "$T_120_REPO"   --local-dir adapters/t2T_120
ls -la adapters/
