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

This tree carries no shaping or budget machinery: the rubric is the stock
correctness-only `RLMTrainRubric`, and the only non-upstream training piece is
the adaptive scaffold-cost advantage at the trainer's advantage seam.
