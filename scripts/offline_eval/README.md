# Offline full-dataset eval — portable (rented GPU box)

Policies (first pass): BASE (Qwen/Qwen3-30B-A3B-Instruct-2507), OUR t2T_final
(step-200 LoRA from HF), AUTHORS correctness-only (mit-oasys/rlm-qwen3-30b-a3b-v0.1).
t2T_120 = conditional (add back if T-final shows an oolong deficit); t2C added
when the control finishes.

**Authors' policy is a LoRA on this exact same base model**, not a standalone
model (confirmed via its `adapter_config.json`: `base_model_name_or_path:
Qwen/Qwen3-30B-A3B-Instruct-2507`) — it's served as a third `--lora-modules`
entry alongside our own adapters, on the same server. There is no separate
"authors serve leg."

## GPUs (RTX 6000 Pro, ~98GB)
The 30B model's real bf16 footprint is ~82GiB (larger than "30B-A3B" active-params
naming suggests). `--gpu-memory-utilization` must be pushed to ~0.95 (not the vLLM
default 0.9) or there's only ~0.3GiB left for KV cache and the server fails outright
on startup — override via `GPU_MEM_UTIL` if still too tight. tp=1 (one GPU per
replica) fits comfortably; more GPUs buys replicas for concurrency (export TP=2/4
to shard a single replica across GPUs instead, if preferred). max_model_len=16384
(the RLM scaffold holds long contexts in the REPL — the model never sees >16k).

## Setup on a fresh box
    git clone https://github.com/apicurius/EfficientRLM && cd EfficientRLM
    bash scripts/offline_eval/10_setup.sh          # uv venv + vllm==0.22.0 + envs
    export HF_TOKEN=...                            # HF (adapters)
    export OPENAI_API_KEY=<openrouter key>         # BC+ judge (gpt-5-nano)
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    bash scripts/offline_eval/00a_fetch_adapters.sh   # t2T_120, t2T_final, authors

## Run (single GPU/replica)
    bash scripts/offline_eval/00_serve.sh &        # base + all adapters (incl. authors)
    DRY=1 bash scripts/offline_eval/01_run_evals.sh   # n=3 smoke: CHECK sub-call
                                                      # usage is nonzero in results
    bash scripts/offline_eval/01_run_evals.sh
    python scripts/offline_eval/02_summarize.py

Suites: paper only by default -- trec_coarse n=50 @131k, oolong_pairs n=20 @32k,
bc+ n=150 start=680, codeqa n=50, longcot_mini n=50 (official LongCoT-Mini =
easy split of LongHorizonReasoning/longcot; domain-interleaved order so n=50 is
a balanced 10/domain prefix; deterministic official verifier, no judge key;
override with N_LONGCOT). Our extensions (trec n=200, spam@131k n=200
disentangler) are OFF by default (400 extra examples/policy dwarfed the paper
suite's cost) -- re-enable with N_TREC_EXT=200 N_SPAM=200. Analysis:
02_summarize.py table; per-rollout JSONLs -> ab_paired_cost.py (both-correct
paired cost estimand = pre-registered primary).

## 8x RTX6000 quickstart (once training's GPUs free up)
N identical replicas, one GPU each (tp=1) — every replica serves every policy
(base + t2T_120/t2T_final + authors, all lora-modules on the same base). No
separate "authors leg" to size or split; just add GPUs for concurrency.
Override the GPU set with `SERVE_GPUS` (space-separated indices) and match
`N_REPLICAS` to however many you started.

    bash scripts/offline_eval/00a_fetch_adapters.sh   # pulls t2T_120, t2T_final, authors
    bash scripts/offline_eval/00_serve_8gpu.sh &      # 8 replicas, ports 8000-8007
    # wait for servers to come up, then:
    DRY=1 bash scripts/offline_eval/01_run_evals_8gpu.sh   # smoke — check sub-calls nonzero across shards
    bash scripts/offline_eval/01_run_evals_8gpu.sh
    python scripts/offline_eval/02_summarize.py
    # add t2T_120 back only if t2T_final shows an oolong deficit:
    POLICIES="Qwen t2T_120 t2T_final authors" bash scripts/offline_eval/01_run_evals_8gpu.sh

`bcplus_heldout` (browsecomp_plus) is 10-20x slower than the 4 light suites
(long context, up to 20 sequential agent turns/rollout, external judge calls) --
with extensions off, it's the dominant cost by far, so most GPUs go to it.
Default: `N_HEAVY_REPLICAS` = 2x the policy count (e.g. 6 of 8 GPUs for 3
policies), sub-sharded `N_HEAVY_REPLICAS/policies` ways *per policy* -- each
subshard gets a disjoint `start_index` slice of the 150-example held-out pool
(`BC_POOL`, default 150; must divide evenly by subshards-per-policy) rather
than 1 GPU grinding through all 150 alone. Ports 8000..8000+N_HEAVY_REPLICAS-1.
Plus a higher default `--max-concurrent` for the suite
(`MAX_CONCURRENT_bcplus_heldout=32` vs 16 for everything else). The remaining
`N_REPLICAS - N_HEAVY_REPLICAS` replicas cover the 4 light suites
(`trec_coarse_131k`, `oolong_pairs_32k`, `codeqa`, `longcot_mini`), diagonally
striped across ports per policy (not plain round-robin — a same-suite
collision across policies previously starved one port; see
`01_run_evals_8gpu.sh`), with the slow policy's shards on their own dedicated
port once 2+ light replicas exist.
SUITE_FILTER=<substr> (alongside POLICY_FILTER) is what 01_run_evals.sh itself
uses to shard — set it directly if you need a custom split beyond what the
launcher does.

## Fewer than 8 GPUs
`00_serve_8gpu.sh`/`01_run_evals_8gpu.sh` scale down cleanly: e.g. for 2 GPUs,
`SERVE_GPUS="0 1" bash 00_serve_8gpu.sh &` then `N_REPLICAS=2 bash
01_run_evals_8gpu.sh`. For exactly 1 GPU, just use the single-replica `Run`
section above — no launcher needed.
