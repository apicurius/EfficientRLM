#!/usr/bin/env bash
# vLLM server: base + ALL LoRA adapters as lora-modules on one server (mirrors run
# inference.toml). authors' policy (mit-oasys/rlm-qwen3-30b-a3b-v0.1) is ALSO a LoRA
# on this exact base model (confirmed via its adapter_config.json base_model_name_or_path),
# not a standalone model -- it belongs here, not on a separate serve leg.
# GPU_MEM_UTIL default bumped from 0.9: this model's real bf16 footprint is ~82GiB, so
# 0.9 x ~98GiB card leaves only ~0.3GiB for KV cache (fails outright). Then bumped
# again 0.95 -> 0.97: measured KV under CUDA graphs was only 3.97GiB (~43k tokens =
# ~4-6 concurrent seqs at this workload's context lengths), leaving eval shards
# admission-starved (Running 6-10 of 32 requested). +~2GiB of KV, ~2.9GiB slack left.
# CUDA graphs ON by default (EAGER=1 to opt out): training's inference served with
# enforce_eager=false, and eager decode measured ~11 tok/s/req on these agent loops --
# graphs speed decode AND match how the policies were trained/served.
# MAX_GRAPH_BS caps cudagraph capture at batch 64 (vLLM default captures up to 512):
# eval shards request <=32 concurrent, so captures beyond 64 only burn KV-cache
# memory. Identical numerics -- batches >64 just fall back to piecewise/eager path.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
cd "$(dirname "$0")"
A=$PWD/adapters
# Put the venv's bin on PATH: flashinfer JIT-builds its sampling kernels at
# startup and shells out to `ninja` (FileNotFoundError kills every engine at
# boot when only the venv has it -- masked whenever ~/.cache/flashinfer is
# already warm, so it only bites on fresh caches, e.g. after this box's
# root-fs-wiping reboots).
[ -d "$PWD/../../.venv-eval/bin" ] && export PATH="$(cd "$PWD/../../.venv-eval/bin" && pwd):$PATH"
VLLM=$(command -v vllm || echo ../../.venv-eval/bin/vllm)
EAGER_FLAG=""
[ "${EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"
LORAS=""
[ -d "$A/t2T_final" ] && LORAS="$LORAS t2T_final=$A/t2T_final"
[ -d "$A/t2T_120" ]   && LORAS="$LORAS t2T_120=$A/t2T_120"
[ -d "$A/t2C" ]       && LORAS="$LORAS t2C=$A/t2C"
[ -d "$A/authors" ]   && LORAS="$LORAS authors=$A/authors"
for d in "$A"/t2C_step*; do [ -d "$d" ] && LORAS="$LORAS $(basename "$d")=$d"; done
exec $VLLM serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size ${TP:-1} --max-model-len ${MAX_MODEL_LEN:-16384} \
  --gpu-memory-utilization ${GPU_MEM_UTIL:-0.97} \
  --max-cudagraph-capture-size ${MAX_GRAPH_BS:-64} $EAGER_FLAG \
  --enable-lora --max-lora-rank 32 --max-loras 8 \
  ${LORAS:+--lora-modules$LORAS} \
  --port ${PORT:-8000} --seed 0
