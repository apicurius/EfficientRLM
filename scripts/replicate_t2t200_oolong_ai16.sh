#!/bin/bash
# Replicate the MISSING treatment@200 in-run OOLONG eval cell on ai16.
# Protocol mirrors the training config's [[orchestrator.eval.env]] oolong entry
# (trec_coarse, 131k ctx, n=25, rollouts=1, max_iterations 20, sub_max 4096).
# DISCLOSURES: (a) ai16 harness — cross-machine eval offset applies (acc ~robust,
# turns shift 1-3); (b) oolong eval questions are procedurally generated per run,
# so these are fresh draws from the same distribution, not the original items.
# Usage: replicate_t2t200_oolong_ai16.sh <repN>
set -euo pipefail
REP=${1:?usage: replicate_t2t200_oolong_ai16.sh <repN>}
EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
PRL=/scratch/omeerdogan23/erlm/prime-rl
set -a; source /scratch/omeerdogan23/erlm/rlm/.env; set +a
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-8000}

srun --overlap --export=ALL --jobid=1263387 --gres=gpu:rtx_a6000:8 -n1 --cpus-per-task=24 --mem=0 bash -l <<INNER
set -euo pipefail
module load cuda/12.8.0 >/dev/null 2>&1 || true
export PATH=$PRL/.venv/bin:\$PATH
export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S=240
export PYTHONPATH=$EFF/scripts/offline_eval/pyshim:$EFF/rlm:$EFF/rlm/training/src:$EFF/rlm/training/environments/browsecomp_plus:$EFF/rlm/training/environments/oolong:$EFF/rlm/training/environments/oolong_pairs:$EFF/rlm/training/environments/longbench_codeqa:$EFF/rlm/training/environments/longcot_mini
cd $EFF/scripts/offline_eval
GPUS=$GPUS PORT=$PORT TP=4 GPU_MEM_UTIL=0.92 bash 00_serve.sh > /tmp/replica_serve_t2T200_$PORT.log 2>&1 &
SRV=\$!
for i in \$(seq 1 180); do curl -s localhost:$PORT/health >/dev/null && break; sleep 10; done
curl -s localhost:$PORT/health >/dev/null || { echo SERVE_FAILED; tail -5 /tmp/replica_serve_t2T200_$PORT.log; kill \$SRV; exit 2; }
vf-eval oolong --env-dir-path $EFF/rlm/training/environments \
  --api-base-url "http://localhost:$PORT/v1" --api-key-var DUMMY_API_KEY --model t2T_final \
  --num-examples 25 --rollouts-per-example 1 --max-concurrent 16 \
  --sampling-args '{"max_completion_tokens":4096,"extra_body":{"enable_thinking":false}}' \
  --env-args '{"dataset_name":"trec_coarse","min_ctx":131072,"max_ctx":131072,"exclude_numeric":false,"num_examples":25,"max_iterations":20,"sub_max_tokens":4096}' \
  --save-results --output-dir $EFF/outputs/replica_t2T200_oolong_$REP/oolong_inrun/t2T_final --disable-tui
kill \$SRV 2>/dev/null || true
INNER

$PRL/.venv/bin/python - <<PY
import glob, json, os
fs = sorted(glob.glob("$EFF/outputs/replica_t2T200_oolong_$REP/oolong_inrun/**/results.jsonl", recursive=True), key=os.path.getmtime)
assert fs, "no results found"
rows = [json.loads(l) for l in open(fs[-1])]
n = len(rows)
acc = sum(float(r.get("reward") or 0) for r in rows) / n
turns = [float((r.get("metrics") or {}).get("rlm_iterations") or 0) for r in rows]
trunc = sum(1 for r in rows if (r.get("metrics") or {}).get("is_truncated")) / n
line = f"| OOLONG-INRUN-REPLICA t2T_final $REP | {acc:.4f} | turns {sum(turns)/n:.1f} | trunc {trunc:.2f} | - | - | n={n} |"
open("$EFF/outputs/advisor/CANARY_T2C.md", "a").write(line + "\n")
print("REPLICA", line)
PY
