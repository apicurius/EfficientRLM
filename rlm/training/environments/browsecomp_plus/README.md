# browsecomp_plus

BrowseComp-Plus evidence-document deep-research QA wired through `RLMTrainEnv`.

- **Env id:** `browsecomp_plus`
- **Layout** (mirrors LMxLM `lm_to_program/browsecomp_plus`; all code is the RLM
  training scaffold — no LMxLM imports):
  - `_data.py` — canary decryption, query loading, doc sampling
    (gold → evidence → negatives → optional corpus distractors), chunking.
  - `_judge.py` — LMxLM/RLM-harness judge prompt + verdict parsing + scorers
    (not the official Texttron grader; see Scoring).
  - `description.py` — root-model text (`user_prologue`, answer format).
  - `env.py` — dataset assembly, rubric wiring, `load_environment`.
- **Data:** `Tevatron/browsecomp-plus` (decrypted via the public canary),
  `k` **text-only** docs/query (default `k=40`, the reference's common setting).
  Per-query doc sampling is SHA-256-seeded, so prompts are identical across
  processes/runs at the same `(seed, query_id)` — required for paired A/B arms.
- **Context:** docs are split into sub-LLM-budget-safe chunks
  (`context_chunk_chars=20000`, ~6.6k tokens vs the ~12k-token sub-prompt cap)
  exposed as the REPL variable `context`; each chunk carries a
  `[BrowseComp+ doc i chunk j/n]` provenance header. Chunking is what keeps
  `llm_query` calls inside budget (unchunked k=50 docs reject-loop to zero reward).
- **Answer format:** clean final answer — the `is_correct` judge compares it
  directly to the gold answer.
- **Scoring:** `reward_mode="judge"` (default) uses the RLM-harness/LMxLM
  `is_correct` JSON judge — an internal training-reward metric, more lenient
  than the official grader (explicit partial-answer credit). NOT leaderboard
  comparable: the official Texttron grader is the BrowseComp GRADER_TEMPLATE
  (`correct: yes|no`) judged by Qwen3-32B (legacy GPT-4.1); for reportable
  numbers, rescore saved outputs offline with the official
  `scripts_evaluation/evaluate_run.py`. A deterministic normalized-containment
  proxy is the fallback (`reward_mode != "judge"`).
- **Source trace:** judge + k-doc sampling ported from the RLM paper's private
  harness via alexzhang13/LMxLM `lm_to_program/browsecomp_plus`. The k-doc
  mixed gold/evidence/negative context is that lineage's protocol — the
  official benchmark has only full-corpus retrieval or oracle modes, so no
  k-doc number is directly comparable to official numbers under any judge.
  Dataset decryption per the Tevatron/browsecomp-plus card.

The model finalizes by setting `answer["content"]` to ONLY its clean final
answer and `answer["ready"] = True`.

### k and multienv training health

Default `k=40` matches the reference. The 200-step A/B configs pin `k=50` for
comparability with the running treatment arm. Fewer docs → fewer chunks → fewer
sub-call turns → less root-trace truncation and fewer all-fail (zero-advantage)
groups, so prefer `k=40` for new configs unless pairing with an existing run.

Opt-in efficiency shaping kwargs (`shaping_coef` / budgets / weights) behave as
in the other eval-suite envs; default is the stock correctness-only rubric.
