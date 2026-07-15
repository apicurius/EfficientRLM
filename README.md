# EfficientRLM

Correctness-first, cost-second reinforcement learning for Recursive Language
Models (RLMs).

An RLM answers long-context queries by operating a scaffold rather than
reading the prompt: the context is stored as a variable in a Python REPL, and
the root model writes code to slice it, delegate pieces to sub-LM calls, and
assemble an answer ([Zhang et al., 2025](https://arxiv.org/abs/2512.24601)).
Reinforcement learning on answer correctness alone improves this behavior but
leaves the operation itself unpriced: how many turns a policy takes, how much
it delegates, and when it stops are free variables, and their tail is heavy —
single rollouts can spend thousands of sub-calls. This repository implements
and evaluates an objective that prices scaffold operation directly while
keeping correctness strictly first.

## Objective

The objective is a transform of the GRPO advantage, applied per rollout group
at the trainer's custom-advantage seam; the reward is never modified. For a
group of rollouts on one prompt:

1. A validity gate zeroes every rollout that is incorrect or fatal
   (no final answer, iteration-cap stop). The gate is unconditional.
2. Valid-correct siblings are re-ranked by scaffold cost — iterations plus
   log-damped sub-LM calls — into scores in `[1 − β, 1]`, so the cheapest
   correct rollout is preferred but every correct rollout dominates every
   incorrect one.
3. The coefficient β ramps with the group's solve rate from 0 (at or below a
   solve floor) to `β_max`: cost pressure applies only where the group
   already solves the task, never where correctness is still being learned.

Properties held by construction and enforced by the test suite: correctness
dominance; exact reduction to validity-gated correctness at `β_max = 0`
(which defines the control arm); group-local, stateless computation; cost
defined over scaffold actions, never tokens (tokens are telemetry). The four
registered cost bases live in
`rlm/training/src/rlm_train/adaptive_group.py`.

## Matched-pair experiment

Two configurations, `rlm/training/configs/qwen3-30b-t2-{treatment,control}.toml`,
differ in exactly one scalar: `beta_max = 0.15` versus `0.0`. Both train
attention-only LoRA adapters on Qwen3-30B-A3B-Instruct over a 50/50 mixture of
long-context spam classification (OOLONG) and evidence-document QA
(BrowseComp-Plus), 200 steps, with unshaped held-out evaluations every 20
steps. The primary comparison is a within-prompt, both-correct paired cost
gap; never-trained transfer families (LongBench-v2 CodeQA and document-QA
domains, OOLONG-Pairs) probe what the objective does outside its training
distribution.

## Layout

| Path | Contents |
|---|---|
| `rlm/` | Fork of [alexzhang13/rlm](https://github.com/alexzhang13/rlm), pinned; training environments and the advantage operator under `rlm/training/` |
| `rlm/training/src/rlm_train/` | Environment driver, rubric, sub-LM proxy, REPL worker, and `adaptive_group.py` (the objective) |
| `rlm/training/environments/` | `oolong`, `browsecomp_plus`, `oolong_pairs`, `longbench_codeqa`, `longcot_mini` |
| `rlm/training/tests/` | Invariant tests for the advantage transform, scoring, and configs |
| `prime-rl/` | [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), pinned, unmodified |
| `scripts/` | Read-only measurement instruments: telemetry export, training-curve and tail figures, strict-oracle rescoring, offline evaluation, counterfactual advantage replay |
| `docs/` | Frozen endpoint specification and the gated follow-up design |
| `PROVENANCE.md` | Pinned upstream commits and the accepted deviations, itemized |

## Running

Training uses `prime-rl` with the environments wired as
[`verifiers`](https://github.com/willccbb/verifiers) modules:

```bash
cd prime-rl
uv run rl @ ../rlm/training/configs/qwen3-30b-t2-treatment.toml
```

Serving assumes a 16k context window with eager execution; judge credentials
(BrowseComp-Plus uses an LLM judge) and W&B keys are read from the
environment. `rlm/training/README.md` and the per-environment READMEs carry
the details.

## Provenance

Both upstream dependencies are vendored at pinned commits and kept clean;
every deviation is recorded in `PROVENANCE.md`. Measurement code is separated
from training code: everything under `scripts/` reads saved artifacts and can
regenerate every reported figure and table deterministically.
