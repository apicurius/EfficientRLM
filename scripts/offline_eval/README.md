# Offline full-dataset eval — portable (rented 4-GPU box)

Policies (first pass): BASE (Qwen/Qwen3-30B-A3B-Instruct-2507),
AUTHORS correctness-only (mit-oasys/rlm-qwen3-30b-a3b-v0.1), OUR t2T_final
(step-200 LoRA from HF). t2T_120 = conditional (uncomment if T-final shows an
oolong deficit); t2C added when the control finishes.

## GPUs (RTX 6000 Pro 92GB)
92GB cards fit the 30B bf16 (~57GB) on ONE GPU: default tp=1. 1 GPU = minimum;
2 GPUs = run both serve legs in parallel; 4 GPUs = add replicas for concurrency
(export TP=2/4 to shard instead if preferred). max_model_len=16384
(the RLM scaffold holds long contexts in the REPL — the model never sees >16k).
base + LoRA adapters share one server (enable_lora); the authors' model gets
its own serve leg (same 4 GPUs, sequential).

## Setup on a fresh box
    git clone https://github.com/apicurius/EfficientRLM && cd EfficientRLM
    bash scripts/offline_eval/10_setup.sh          # uv venv + vllm==0.22.0 + envs
    export HF_TOKEN=...                            # HF (adapters)
    export OPENAI_API_KEY=<openrouter key>         # BC+ judge (gpt-5-nano)
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    bash scripts/offline_eval/00a_fetch_adapters.sh

## Run
    bash scripts/offline_eval/00_serve.sh &        # base + adapters
    DRY=1 bash scripts/offline_eval/01_run_evals.sh   # n=3 smoke: CHECK sub-call
                                                      # usage is nonzero in results
    bash scripts/offline_eval/01_run_evals.sh
    # then swap server: kill, bash 00b_serve_authors.sh &, and run
    # POLICY_FILTER=mit bash scripts/offline_eval/01_run_evals.sh
    python scripts/offline_eval/02_summarize.py

Suites: paper (trec_coarse n=50 @131k, oolong_pairs n=20 @32k, bc+ n=150
start=655, codeqa n=50) + our extensions (trec n=200, spam@131k n=200
disentangler). VERIFY oolong_pairs/codeqa env-args vs env READMEs on dry run.
Analysis: 02_summarize.py table; per-rollout JSONLs -> ab_paired_cost.py
(both-correct paired cost estimand = pre-registered primary).
