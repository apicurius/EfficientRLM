#!/usr/bin/env bash
# Paper suite + extensions across policies. DRY=1 -> n=3 smoke.
# POLICY_FILTER=<substr> runs only matching policies (e.g. =authors).
# SUITE_FILTER=<substr> runs only matching suites (e.g. =bcplus_heldout) — for sharding
# across parallel replicas (see 01_run_evals_8gpu.sh).
# LONGCOT_ONLY=1 runs only LongCoT-Mini. If N_LONGCOT is unset and DRY is not
# enabled, this uses the full local pinned easy split (507 examples).
# LONGCOT_START offsets into the deterministic domain-interleaved LongCoT-Mini
# order; 01_run_evals_8gpu.sh uses it for disjoint per-policy shards.
# All policies (base + our adapters + authors) are served as lora-modules on ONE server
# type (00_serve.sh) -- authors' policy is a LoRA on the same base model, not a
# standalone model, so there is no separate "authors serve leg" anymore.
set -euo pipefail
cd "$(dirname "$0")"
ENVDIR=$PWD/../../rlm/training/environments
VF=${VF:-$(command -v vf-eval || echo $PWD/../../.venv-eval/bin/vf-eval)}
OUT=${OUT:-$PWD/../../outputs/offline_eval_$(date +%Y%m%d)}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
export DUMMY_API_KEY=dummy
SAMP='{"max_completion_tokens":4096,"extra_body":{"enable_thinking":false}}'
if [ -n "${POLICIES_OVERRIDE:-}" ]; then read -ra POLICIES <<< "$POLICIES_OVERRIDE"; else POLICIES=(Qwen/Qwen3-30B-A3B-Instruct-2507 t2T_final authors); fi
# t2T_final: step-200 adapter, uploaded (oerdogan/qwen3-30b-t2-treatment-lora-step200) — now the default.
# t2T_120: conditional — add via POLICIES_OVERRIDE if t2T_final shows an oolong deficit. t2C: when control finishes.
N_TREC=50; N_PAIRS=20; N_BC=${N_BC:-150}; N_CODEQA=50
# longcot_mini: official LongCoT-Mini (easy split, local pinned package = 507 q).
# Dataset order is domain-interleaved; for Alex-style full LongCoT-only eval use
# LONGCOT_ONLY=1 (or set N_LONGCOT explicitly). Deterministic official verifier
# (no judge key needed). `chem` is spelled `chemistry` in the loader.
N_LONGCOT_WAS_SET=${N_LONGCOT+x}
N_LONGCOT=${N_LONGCOT:-50}
LONGCOT_DOMAINS_JSON=${LONGCOT_DOMAINS_JSON:-'["logic","cs","chemistry","chess","math"]'}
LONGCOT_START=${LONGCOT_START:-0}
# Extension suites (trec_coarse_131k_ext n=200, spam_131k_disentangler n=200) are OFF
# by default -- 400 extra examples/policy dwarfed the paper suite's cost. Re-enable
# explicitly: N_TREC_EXT=200 N_SPAM=200 bash 01_run_evals.sh (or via 01_run_evals_8gpu.sh).
N_TREC_EXT=${N_TREC_EXT:-0}; N_SPAM=${N_SPAM:-0}
# LongBench-v2 domain extensions: same env/verifier as
# codeqa, different domain filter. Probes whether the t2T_final codeqa collapse
# is code-specific or generic OOD-MCQ (see forensics in session notes). OFF by
# default; full domains are n=175 (Single-Doc) / n=125 (Multi-Doc).
# num_examples:-1 in env-args exposes the whole domain; --num-examples then
# picks the eval slice (the env default of 50 would silently cap the dataset).
N_LB_SDQA=${N_LB_SDQA:-0}; N_LB_MDQA=${N_LB_MDQA:-0}
# BC_START/BC_N: browsecomp_plus sub-sharding within one policy (see 01_run_evals_8gpu.sh) --
# defaults are the full held-out pool (680..829, n=150).
BC_START=${BC_START:-680}
DRY_N=${DRY_N:-3}
if [ "${DRY:-0}" = "1" ]; then N_TREC=$DRY_N; N_PAIRS=$DRY_N; N_BC=$DRY_N; N_CODEQA=$DRY_N; N_LONGCOT=$DRY_N; N_TREC_EXT=0; N_SPAM=0; fi
if [ "${LONGCOT_ONLY:-0}" = "1" ]; then
  N_TREC=0; N_PAIRS=0; N_BC=0; N_CODEQA=0; N_TREC_EXT=0; N_SPAM=0
  if [ "${DRY:-0}" != "1" ] && [ -z "${N_LONGCOT_WAS_SET:-}" ]; then N_LONGCOT=507; fi
fi

# BrowseComp+ scores via an external LM judge (needs OPENAI_API_KEY +
# OPENAI_BASE_URL, see README). A missing key does NOT error the rollout --
# every judge call short-circuits to 0.0 and the whole suite silently scores
# zero (bit us once: 12/12 zero-reward bcplus rollouts in an otherwise-green
# smoke). Source the persistent key file if present (this box wipes /tmp and
# shell env on its frequent reboots; only /teamspace paths survive), then
# fail LOUDLY if bcplus is about to run without a key.
JUDGE_ENV=${JUDGE_ENV:-$HOME/.cache/efficientrlm/judge.env}
if [ -f "$JUDGE_ENV" ]; then set -a; . "$JUDGE_ENV"; set +a; fi
case "bcplus_heldout" in *"${SUITE_FILTER:-}"*)
  if [ "$N_BC" != "0" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "FATAL: bcplus_heldout (n=$N_BC) needs OPENAI_API_KEY for its external judge" >&2
    echo "       (openai/gpt-5-nano via OpenRouter). Without it every rollout scores 0.0" >&2
    echo "       silently. export OPENAI_API_KEY=... OPENAI_BASE_URL=... or write them" >&2
    echo "       to $JUDGE_ENV (KEY=value lines), or run with N_BC=0 / SUITE_FILTER." >&2
    exit 1
  fi
;; esac

# Per-suite concurrency override, e.g. MAX_CONCURRENT_bcplus_heldout=32 (suite name
# with non-alnum chars kept as-is since these are all plain identifiers already).
run() { local env=$1 n=$2 pol=$3 args=$4 name=$5
  [ "$n" = "0" ] && return 0
  case "$pol" in *"${POLICY_FILTER:-}"*) ;; *) return 0;; esac
  case "$name" in *"${SUITE_FILTER:-}"*) ;; *) return 0;; esac
  local mc_var="MAX_CONCURRENT_${name}"
  local mc="${!mc_var:-${MAX_CONCURRENT:-16}}"
  local out_name="${name}${OUT_SUFFIX:-}"
  $VF "$env" --env-dir-path "$ENVDIR" \
    --api-base-url "$BASE_URL" --api-key-var DUMMY_API_KEY --model "$pol" \
    --num-examples "$n" --rollouts-per-example 1 --max-concurrent "$mc" \
    --sampling-args "$SAMP" --env-args "$args" \
    --save-results --output-dir "$OUT/$out_name/${pol//\//_}" --disable-tui
}
for POL in "${POLICIES[@]}"; do
  run oolong           $N_TREC     "$POL" '{"dataset_name":"trec_coarse","max_ctx":131072,"min_ctx":131072,"max_iterations":20,"sub_max_tokens":4096,"exclude_numeric":false}' trec_coarse_131k
  run oolong_pairs     $N_PAIRS    "$POL" '{"context_len":32768,"max_iterations":20,"sub_max_tokens":4096}' oolong_pairs_32k
  run browsecomp_plus  $N_BC       "$POL" '{"k":50,"seed":42,"judge_model":"openai/gpt-5-nano","min_subcall":0,"reward_mode":"judge","start_index":'"$BC_START"',"max_iterations":20,"min_iterations":2,"sub_max_tokens":4096}' bcplus_heldout
  run longbench_codeqa $N_CODEQA   "$POL" '{"max_iterations":20,"sub_max_tokens":4096}' codeqa
  run longcot_mini     $N_LONGCOT  "$POL" '{"difficulty":"easy","domains":'"$LONGCOT_DOMAINS_JSON"',"start_index":'"$LONGCOT_START"',"max_iterations":20,"sub_max_tokens":4096}' longcot_mini
  run longbench_codeqa $N_LB_SDQA  "$POL" '{"domain":"Single-Document QA","num_examples":-1,"max_iterations":20,"sub_max_tokens":4096}' lb_singledoc
  run longbench_codeqa $N_LB_MDQA  "$POL" '{"domain":"Multi-Document QA","num_examples":-1,"max_iterations":20,"sub_max_tokens":4096}' lb_multidoc
  run oolong           $N_TREC_EXT "$POL" '{"dataset_name":"trec_coarse","max_ctx":131072,"min_ctx":131072,"max_iterations":20,"sub_max_tokens":4096,"exclude_numeric":false}' trec_coarse_131k_ext
  run oolong           $N_SPAM     "$POL" '{"dataset_name":"spam","max_ctx":131072,"min_ctx":131072,"max_iterations":20,"sub_max_tokens":4096,"exclude_numeric":true}' spam_131k_disentangler
done
echo "results in $OUT"
