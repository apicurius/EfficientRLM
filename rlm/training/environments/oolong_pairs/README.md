# oolong_pairs

OOLONG-Pairs pairwise-aggregation QA wired through `RLMTrainEnv`.

- **Env id:** `oolong_pairs`
- **Data:** `mit-oasys/oolong-pairs` questions over the `oolongbench/oolong-synth`
  TREC-coarse context (HF eval picture: `@32k (n=20)`).
- **Scoring:** precision/recall/**F1** over unordered user-ID pairs,
  deterministic — no judge.
- **Source trace:** PrimeIntellect-ai/research-environments `rlm_oolong_pairs` +
  LMxLM OOLONG-Pairs task description.

The long context is exposed as the REPL variable `context`; the model finalizes
by setting `answer["content"]` (a list of `(id1, id2)` pairs, lower ID first, or
`[]`) and `answer["ready"] = True`.

Opt-in efficiency shaping kwargs (`shaping_coef` / budgets / weights) behave as
in the other eval-suite envs; default is the stock correctness-only rubric. See
repo-root `THESIS.md`.
