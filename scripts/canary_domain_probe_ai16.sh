#!/bin/bash
# Exploratory domain probe on the closed ai16 harness: does a policy's CodeQA
# delegation behavior generalize to LongBench Single-Doc QA?
# Usage: canary_domain_probe_ai16.sh <policy>   (base | t2C_stepN | t2T_120 | t2T_final)
# Adapter must already be staged in scripts/offline_eval/adapters/ by a prior canary run.
set -euo pipefail
POL=${1:?usage: canary_domain_probe_ai16.sh <policy>}
[ "$POL" = "base" ] && POL="Qwen/Qwen3-30B-A3B-Instruct-2507"
EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
PRL=/scratch/omeerdogan23/erlm/prime-rl
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
GPUS=$GPUS PORT=$PORT TP=4 GPU_MEM_UTIL=0.92 bash 00_serve.sh > /tmp/probe_serve_${SAFE}_$PORT.log 2>&1 &
SRV=\$!
for i in \$(seq 1 90); do curl -s localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -s localhost:$PORT/health >/dev/null || { echo SERVE_FAILED; tail -5 /tmp/probe_serve_${SAFE}_$PORT.log; kill \$SRV; exit 2; }
BASE_URL="http://localhost:$PORT/v1" POLICIES_OVERRIDE="$POL" SUITE_FILTER=lb_singledoc \
  N_LB_SDQA=50 N_CODEQA=0 N_TREC=0 N_PAIRS=0 N_BC=0 N_LONGCOT=0 \
  OUT=$EFF/outputs/probe_sdqa_$SAFE bash 01_run_evals.sh
kill \$SRV 2>/dev/null || true
INNER

$PRL/.venv/bin/python - <<PY
import glob, json, os
fs = sorted(glob.glob("$EFF/outputs/probe_sdqa_$SAFE/lb_singledoc/*/**/results.jsonl", recursive=True), key=os.path.getmtime)
assert fs, "no results found"
rows = [json.loads(l) for l in open(fs[-1])]
n = len(rows)
acc = sum(1 for r in rows if float(r.get("reward") or 0) >= 0.5) / n
subs = [float((r.get("metrics") or {}).get("rlm_sub_llm_calls") or 0) for r in rows]
zero = sum(1 for s in subs if s == 0) / n
fin = sum(1 for r in rows if (r.get("metrics") or {}).get("rlm_has_final_answer")) / n
line = f"| SDQA {('$POL').split('/')[-1]} | {acc:.3f} | {sum(subs)/n:.1f} | {zero:.2f} | - | {fin:.2f} | n={n} |"
open("$EFF/outputs/advisor/CANARY_T2C.md", "a").write(line + "\n")
print("PROBE", line)
PY
