# oolong

OOLONG-synth long-context aggregate QA wired through `RLMTrainEnv`.

- **Env id:** `oolong`
- **Data:** `oolongbench/oolong-synth`.
- **Scoring:** byte-faithful port of the upstream `alexzhang13/rlm` OOLONG
  synth scorer used by the baseline training/eval env.
- **Args:** this env intentionally uses the same OOLONG interface as `rlm/`:
  `dataset_name`, `min_ctx`, `max_ctx`, `exclude_numeric`, `num_examples`,
  `max_iterations`, `sub_max_tokens`, `min_iterations`, and `min_subcall`.

The long context is exposed as the REPL variable `context`; the model finalizes
by setting `answer["content"]` and `answer["ready"] = True`.

In `.research/ERLM-main`, opt-in efficiency shaping is still available through
rubric-only kwargs (`shaping_coef`, `correct_threshold`, `subcall_budget`,
`token_budget`, and `*_weight`). With `shaping_coef = 0.0`, this uses the stock
correctness-only `RLMTrainRubric`; with `shaping_coef > 0.0`, it uses
`EfficiencyGatedRubric`. The OOLONG data/scoring path and old args remain aligned with `rlm/`;
`.research/ERLM-main` adds the OOLONG per-env `user_prologue` as a
runtime hint to reduce context-output leakage.
