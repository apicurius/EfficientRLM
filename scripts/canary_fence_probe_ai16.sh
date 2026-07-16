#!/bin/bash
# Exploratory: the FENCE instruction probe (mandate-delegation prologue from
# the offline codeqa_fenced suite) on the closed ai16 harness. Prologue text
# is read from the original offline run's saved metadata, not re-typed.
# Compare within-harness against the unfenced pooled cells only.
# Usage: canary_fence_probe_ai16.sh <policy>   (base | t2C_stepN | t2T_120 | t2T_final)
set -euo pipefail
POL=${1:?usage: canary_fence_probe_ai16.sh <policy>}
[ "$POL" = "base" ] && POL="Qwen/Qwen3-30B-A3B-Instruct-2507"
EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
PRL=/scratch/omeerdogan23/erlm/prime-rl
META=$EFF/outputs/offline_eval_full_20260712/codeqa_fenced/t2T_final/evals/longbench_codeqa--t2T_final/00b78079/metadata.json
set -a; source /scratch/omeerdogan23/erlm/rlm/.env; set +a
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-8000}
SAFE=${POL//\//_}

srun --overlap --export=ALL --jobid=1263387 --gres=gpu:rtx_a6000:8 -n1 --cpus-per-task=24 --mem=0 bash -l <<INNER
set -euo pipefail
module load cuda/12.8.0 >/dev/null 2>&1 || true
export PATH=$PRL/.venv/bin:\$PATH
export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=240
export PYTHONPATH=$EFF/scripts/offline_eval/pyshim:$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong:$EFF/rlm/training/environments/oolong_pairs:$EFF/rlm/training/environments/longbench_codeqa:$EFF/rlm/training/environments/longcot_mini
cd $EFF/scripts/offline_eval
GPUS=$GPUS PORT=$PORT TP=4 GPU_MEM_UTIL=0.92 bash 00_serve.sh > /tmp/probe_serve_fence_${SAFE}_$PORT.log 2>&1 &
SRV=\$!
for i in \$(seq 1 90); do curl -s localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -s localhost:$PORT/health >/dev/null || { echo SERVE_FAILED; tail -5 /tmp/probe_serve_fence_${SAFE}_$PORT.log; kill \$SRV; exit 2; }
EA=\$(python - <<'PYEOF'
import json
m = json.load(open("$META"))
print(json.dumps({
    "max_iterations": 20, "sub_max_tokens": 4096,
    "user_prologue": m["env_args"]["user_prologue"],
}))
PYEOF
)
vf-eval longbench_codeqa --env-dir-path $EFF/rlm/training/environments \
  --api-base-url "http://localhost:$PORT/v1" --api-key-var DUMMY_API_KEY --model "$POL" \
  --num-examples 50 --rollouts-per-example 1 --max-concurrent 16 \
  --sampling-args '{"max_completion_tokens":4096,"extra_body":{"enable_thinking":false}}' \
  --env-args "\$EA" \
  --save-results --output-dir $EFF/outputs/probe_fence_$SAFE/codeqa_fenced/${SAFE} --disable-tui
kill \$SRV 2>/dev/null || true
INNER

$PRL/.venv/bin/python - <<PY
import glob, json, os
fs = sorted(glob.glob("$EFF/outputs/probe_fence_$SAFE/codeqa_fenced/**/results.jsonl", recursive=True), key=os.path.getmtime)
assert fs, "no results found"
rows = [json.loads(l) for l in open(fs[-1])]
n = len(rows)
acc = sum(1 for r in rows if float(r.get("reward") or 0) >= 0.5) / n
subs = [float((r.get("metrics") or {}).get("rlm_sub_llm_calls") or 0) for r in rows]
zero = sum(1 for s in subs if s == 0) / n
cap = sum(1 for r in rows if str(r.get("stop_condition")) == "max_turns_reached") / n
fin = sum(1 for r in rows if (r.get("metrics") or {}).get("rlm_has_final_answer")) / n
line = f"| CODEQA-FENCED {('$POL').split('/')[-1]} | {acc:.3f} | {sum(subs)/n:.1f} | {zero:.2f} | {cap:.2f} | {fin:.2f} | n={n} |"
open("$EFF/outputs/advisor/CANARY_T2C.md", "a").write(line + "\n")
print("PROBE", line)
PY
