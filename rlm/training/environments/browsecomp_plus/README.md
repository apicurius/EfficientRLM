# browsecomp_plus

BrowseComp-Plus evidence-document deep-research QA wired through `RLMTrainEnv`.

- **Env id:** `browsecomp_plus`
- **Data:** `Tevatron/browsecomp-plus` (decrypted via the public canary),
  `k` **text-only** docs/query (matches LMxLM `evidence_docs`; HF eval picture:
  `test (n=150, k=50 documents)`).
- **Answer format:** clean final answer (LMxLM — no imposed Explanation/Confidence
  format; the `is_correct` judge compares it directly to the gold answer).
- **Scoring:** `reward_mode="judge"` (default) reproduces the released metric
  using the **verbatim LMxLM / BrowseComp-Plus judge** (`BROWSECOMP_PLUS_JUDGE_PROMPT`,
  binary `is_correct` JSON verdict — the reference uses `gpt-5-nano`, we default to
  `gpt-4.1`); a deterministic `Exact Answer` normalized-match proxy is available as
  a fallback / offline smoke-test mode (`reward_mode != "judge"`).
- **Source trace:** PrimeIntellect-ai/research-environments `rlm_browsecomp` +
  LMxLM `lm_to_program/browsecomp_plus`.

The evidence docs are exposed as the REPL variable `context` (a `list[str]` of
raw document text); the model finalizes by setting `answer["content"]` to ONLY
its clean final answer and `answer["ready"] = True`.

### Small-model k=50 consistency

For the Qwen3-30B / small-model setup this env defaults to `k=50`. Do not use
the RLM paper's `k=1000` stress setting here unless creating a separate
large-context replication config.

Opt-in efficiency shaping kwargs (`shaping_coef` / budgets / weights) behave as
in the other eval-suite envs; default is the stock correctness-only rubric. See
repo-root `THESIS.md`.
