#!/bin/bash
# One-off: treatment step-120 CodeQA point on the SAME closed ai16 canary harness.
# Purpose: control@120 collapsed (subs 17.5->2.7); this point separates
# "beta caused the collapse" from "beta accelerated a generic RL drift".
# Same isolation rule as canary_codeqa_ai16.sh: TREND vs ai16 base anchor only.
set -euo pipefail
EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
PRL=/scratch/omeerdogan23/erlm/prime-rl
ADIR=$EFF/scripts/offline_eval/adapters/t2T_120
set -a; source /scratch/omeerdogan23/erlm/rlm/.env; set +a
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-8001}
POL=t2T_120

# 1) stage adapter from the surviving treatment LoRA repo (login node has internet)
if [ ! -d "$ADIR" ]; then
  HF_HUB_OFFLINE=0 $PRL/.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("oerdogan/qwen3-30b-t2-treatment-lora-step120",
                      local_dir="/scratch/tmp/omeerdogan23/canary/t2T_120_repo")
print(p)
PY
  INNER=$(dirname "$(find /scratch/tmp/omeerdogan23/canary/t2T_120_repo -name adapter_config.json | head -1)")
  [ -n "$INNER" ] || { echo "no adapter_config.json in repo"; exit 1; }
  mkdir -p "$ADIR" && cp -r "$INNER"/. "$ADIR"/
fi

# 2) serve + eval on ai16 via the holder job (never cancel job 1263387)
srun --overlap --export=ALL --jobid=1263387 --gres=gpu:rtx_a6000:8 -n1 --cpus-per-task=24 --mem=0 bash -l <<INNER
set -euo pipefail
module load cuda/12.8.0 >/dev/null 2>&1 || true
export PATH=$PRL/.venv/bin:\$PATH
export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=240
export PYTHONPATH=$EFF/scripts/offline_eval/pyshim:$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong:$EFF/rlm/training/environments/oolong_pairs:$EFF/rlm/training/environments/longbench_codeqa:$EFF/rlm/training/environments/longcot_mini
cd $EFF/scripts/offline_eval
GPUS=$GPUS PORT=$PORT TP=4 GPU_MEM_UTIL=0.92 bash 00_serve.sh > /tmp/canary_serve_t2T120_$PORT.log 2>&1 &
SRV=\$!
for i in \$(seq 1 90); do curl -s localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -s localhost:$PORT/health >/dev/null || { echo SERVE_FAILED; tail -5 /tmp/canary_serve_t2T120_$PORT.log; kill \$SRV; exit 2; }
BASE_URL="http://localhost:$PORT/v1" POLICIES_OVERRIDE="$POL" SUITE_FILTER=codeqa N_BC=0 \
  OUT=$EFF/outputs/canary_t2T bash 01_run_evals.sh
kill \$SRV 2>/dev/null || true
INNER

# 3) summarize into the canary log, clearly labeled as the treatment arm
$PRL/.venv/bin/python - <<PY
import glob, json, os
fs = sorted(glob.glob("$EFF/outputs/canary_t2T/codeqa/*/**/results.jsonl", recursive=True), key=os.path.getmtime)
fs = [f for f in fs if "t2T_120" in f]
assert fs, "no results found"
rows = [json.loads(l) for l in open(fs[-1])]
n = len(rows)
acc = sum(1 for r in rows if float(r.get("reward") or 0) >= 0.5) / n
subs = [float((r.get("metrics") or {}).get("rlm_sub_llm_calls") or 0) for r in rows]
zero = sum(1 for s in subs if s == 0) / n
cap = sum(1 for r in rows if str(r.get("stop_condition")) == "max_turns_reached") / n
fin = sum(1 for r in rows if (r.get("metrics") or {}).get("rlm_has_final_answer")) / n
line = f"| 120-TREATMENT | {acc:.3f} | {sum(subs)/n:.1f} | {zero:.2f} | {cap:.2f} | {fin:.2f} | n={n} |"
open("$EFF/outputs/advisor/CANARY_T2C.md", "a").write(line + "\n")
print("CANARY", line)
PY
