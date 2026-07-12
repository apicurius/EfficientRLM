#!/usr/bin/env bash
# Paper suite + extensions across policies. DRY=1 -> n=3 smoke.
# POLICY_FILTER=<substr> runs only matching policies (e.g. =mit for authors leg).
set -euo pipefail
cd "$(dirname "$0")"
ENVDIR=$PWD/../../rlm/training/environments
VF=$(command -v vf-eval || echo $PWD/../../.venv-eval/bin/vf-eval)
OUT=${OUT:-$PWD/../../outputs/offline_eval_$(date +%Y%m%d)}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
SAMP='{"max_completion_tokens":4096,"extra_body":{"enable_thinking":false}}'
POLICIES=(Qwen/Qwen3-30B-A3B-Instruct-2507 t2T_final mit-oasys/rlm-qwen3-30b-a3b-v0.1)
# t2T_120: conditional — add above if t2T_final shows an oolong deficit. t2C: when control finishes.
N_TREC=50; N_PAIRS=20; N_BC=150; N_CODEQA=50; N_TREC_EXT=200; N_SPAM=200
if [ "${DRY:-0}" = "1" ]; then N_TREC=3; N_PAIRS=3; N_BC=3; N_CODEQA=3; N_TREC_EXT=0; N_SPAM=0; fi

run() { local env=$1 n=$2 pol=$3 args=$4 name=$5
  [ "$n" = "0" ] && return 0
  case "$pol" in *"${POLICY_FILTER:-}"*) ;; *) return 0;; esac
  $VF "$env" --env-dir-path "$ENVDIR" \
    --api-base-url "$BASE_URL" --api-key-var NONE --model "$pol" \
    --num-examples "$n" --rollouts-per-example 1 --max-concurrent 16 \
    --sampling-args "$SAMP" --env-args "$args" \
    --save-results --output-dir "$OUT/$name/${pol//\//_}" --disable-tui
}
for POL in "${POLICIES[@]}"; do
  run oolong           $N_TREC     "$POL" '{"dataset_name":"trec_coarse","max_ctx":131072,"min_ctx":131072,"max_iterations":20,"sub_max_tokens":4096,"exclude_numeric":false}' trec_coarse_131k
  run oolong_pairs     $N_PAIRS    "$POL" '{"max_ctx":32768,"min_ctx":32768,"max_iterations":20,"sub_max_tokens":4096}' oolong_pairs_32k   # VERIFY vs env README
  run browsecomp_plus  $N_BC       "$POL" '{"k":50,"seed":42,"judge_model":"openai/gpt-5-nano","min_subcall":0,"reward_mode":"judge","start_index":655,"max_iterations":20,"min_iterations":2,"sub_max_tokens":4096}' bcplus_heldout
  run longbench_codeqa $N_CODEQA   "$POL" '{"max_iterations":20,"sub_max_tokens":4096}' codeqa           # VERIFY vs env README
  run oolong           $N_TREC_EXT "$POL" '{"dataset_name":"trec_coarse","max_ctx":131072,"min_ctx":131072,"max_iterations":20,"sub_max_tokens":4096,"exclude_numeric":false}' trec_coarse_131k_ext
  run oolong           $N_SPAM     "$POL" '{"dataset_name":"spam","max_ctx":131072,"min_ctx":131072,"max_iterations":20,"sub_max_tokens":4096,"exclude_numeric":true}' spam_131k_disentangler
done
echo "results in $OUT"
