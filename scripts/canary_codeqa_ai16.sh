#!/bin/bash
# Live CodeQA transfer canary for the control arm, run on ai16's idle GPUs.
# Usage: canary_codeqa_ai16.sh <step>   (expects ckpt_adapters/step_<N>.tar.gz on HF)
# Exploratory instrument: the post-200 interleaved offline leg stays binding.
set -euo pipefail
STEP=${1:?usage: canary_codeqa_ai16.sh <step|base>}
# HARNESS ISOLATION RULE: everything this script measures is ai16-harness.
# Levels are NOT comparable to studio-measured cells (same-ckpt eval turns
# shifted 1-3 across machines). Read the canary as a TREND against the ai16
# base anchor row only; never place these numbers next to studio numbers.
EFF=/scratch/omeerdogan23/erlm/.research/EfficientRLM
PRL=/scratch/omeerdogan23/erlm/prime-rl
ADIR=$EFF/scripts/offline_eval/adapters/t2C_step$STEP
set -a; source /scratch/omeerdogan23/erlm/rlm/.env; set +a

# 1) stage adapter (login node has internet)
if [ "$STEP" = "base" ]; then ADIR=""; fi
if [ -n "$ADIR" ] && [ ! -d "$ADIR" ]; then
  HF_HUB_OFFLINE=0 $PRL/.venv/bin/python - <<PY
from huggingface_hub import hf_hub_download
print(hf_hub_download("oerdogan/erlm-run-artifacts", "ckpt_adapters/step_$STEP.tar.gz",
                      repo_type="dataset", local_dir="/scratch/tmp/omeerdogan23/canary"))
PY
  mkdir -p "$ADIR.tmp" && tar xzf /scratch/tmp/omeerdogan23/canary/ckpt_adapters/step_$STEP.tar.gz -C "$ADIR.tmp"
  INNER=$(dirname "$(find "$ADIR.tmp" -name adapter_config.json | head -1)")
  [ -n "$INNER" ] || { echo "no adapter_config.json in bundle"; exit 1; }
  mv "$INNER" "$ADIR" && rm -rf "$ADIR.tmp"
fi

# 2) serve + eval on ai16 via the holder job (never cancel job 1263387)
srun --overlap --jobid=1263387 --gres=gpu:rtx_a6000:4 -n1 --cpus-per-task=24 --mem=0 bash -l <<INNER
set -euo pipefail
module load cuda/12.8.0 >/dev/null 2>&1 || true
export PATH=$PRL/.venv/bin:\$PATH
export HF_HOME=/scratch/omeerdogan23/hf_cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
cd $EFF/scripts/offline_eval
GPUS=0,1,2,3 TP=4 GPU_MEM_UTIL=0.92 bash 00_serve.sh > /tmp/canary_serve_$STEP.log 2>&1 &
SRV=\$!
for i in \$(seq 1 90); do curl -s localhost:8000/health >/dev/null && break; sleep 10; done
curl -s localhost:8000/health >/dev/null || { echo SERVE_FAILED; tail -5 /tmp/canary_serve_$STEP.log; kill \$SRV; exit 2; }
POL="t2C_step$STEP"; [ "$STEP" = "base" ] && POL="Qwen/Qwen3-30B-A3B-Instruct-2507"
POLICIES_OVERRIDE="$POL" SUITE_FILTER=codeqa N_BC=0 \
  OUT=$EFF/outputs/canary_t2C bash 01_run_evals.sh
kill \$SRV 2>/dev/null || true
INNER

# 3) summarize into the canary log
$PRL/.venv/bin/python - <<PY
import glob, json, os
root = "$EFF/outputs/canary_t2C"
pol = "t2C_step$STEP" if "$STEP" != "base" else "Qwen*"
fs = sorted(glob.glob(f"{root}/codeqa/*/**/results.jsonl", recursive=True), key=os.path.getmtime)
fs = [f for f in fs if ("t2C_step$STEP" in f) or ("$STEP" == "base" and "Qwen" in f)]
assert fs, "no results found"
rows = [json.loads(l) for l in open(fs[-1])]
n = len(rows)
acc = sum(1 for r in rows if float(r.get("reward") or 0) >= 0.5) / n
subs = [float((r.get("metrics") or {}).get("rlm_sub_llm_calls") or 0) for r in rows]
zero = sum(1 for s in subs if s == 0) / n
cap = sum(1 for r in rows if str(r.get("stop_condition")) == "max_turns_reached") / n
fin = sum(1 for r in rows if (r.get("metrics") or {}).get("rlm_has_final_answer")) / n
line = f"| $STEP | {acc:.3f} | {sum(subs)/n:.1f} | {zero:.2f} | {cap:.2f} | {fin:.2f} | n={n} |"
log = "$EFF/outputs/advisor/CANARY_T2C.md"
if not os.path.exists(log):
    open(log, "w").write("# Control CodeQA transfer canary (ai16 harness, exploratory)\n\nCLOSED-HARNESS INSTRUMENT: all rows measured on ai16 A6000 serving. Levels are\nNOT comparable to any studio-measured cell (cross-machine eval replicate showed\nturn shifts of 1-3 on identical checkpoints). Read TRENDS against the ai16 base\nanchor row only. The binding codeqa cells come from the post-200 interleaved\nstudio leg.\n\n| step | acc | subs | zero | cap | fin | n |\n|---|---|---|---|---|---|---|\n")
open(log, "a").write(line + "\n")
print("CANARY", line)
PY
