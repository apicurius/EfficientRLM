# longbench_codeqa

LongBench-v2 Code Repository Understanding QA wired through `RLMTrainEnv`.

- **Env id:** `longbench_codeqa`
- **Data:** `THUDM/LongBench-v2` filtered to
  `domain == "Code Repository Understanding"` (exactly 50 examples; HF eval
  picture: `LongBenchv2 Code repo QA (n=50)`).
- **Scoring:** 4-choice MCQ scored by exact letter match (A/B/C/D),
  deterministic — no judge needed.

The long repo context is exposed as the REPL variable `context`; the model
finalizes by setting `answer["content"]` to a single letter and
`answer["ready"] = True`.

Opt-in efficiency shaping kwargs (`shaping_coef` / budgets / weights) behave as
in the other eval-suite envs; default is the stock correctness-only rubric. See
repo-root `THESIS.md`.
