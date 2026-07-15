# longcot_mini

LongCoT-Mini long-horizon reasoning (arXiv:2604.14140) wired through `RLMTrainEnv`.

- **Env id:** `longcot_mini`
- **Data:** the official LongCoT **easy** split (= "LongCoT-Mini", 507 questions;
  ~100 each of logic / cs / chemistry / chess / math), loaded from the pinned
  official `longcot` package (github.com/LongHorizonReasoning/longcot @
  `fb96494`, a pip git-dependency — data ships inside the wheel, so no HF/network
  access at load time). The HF mirror (`LongHorizonReasoning/longcot`) is
  deliberately NOT used: its parquet lacks the `problem` payloads the official
  verifier dispatches on (all 500 logic answers are JSON `null` there), so
  faithful scoring is only possible from the package data. Row counts verified
  identical (easy = 507).
- **Scoring:** the official binary verifier (`longcot.verify`): extract the text
  after the last `solution =` (with the official whole-response fallbacks), then
  per-domain deterministic verification — move-sequence simulation for logic,
  sympy equivalence for math, RDKit canonical-SMILES equality for chemistry,
  python-chess replay/SAN checks for chess, JSON / int-list equality for cs.
  Verifier exceptions score 0.0 (matches the official `run_eval.py`). The
  official Gemini LLM fallbacks are OFF by default (`enable_fallback = false`,
  equivalent to `--no-fallback`), so rewards are deterministic; enabling them
  requires `GEMINI_API_KEY` and makes math/chemistry rewards judge-dependent.
- **Args:** `difficulty` (default `"easy"` = LongCoT-Mini; `"medium"` / `"hard"`
  give full-benchmark slices), `domains`, `question_ids`, `num_examples`,
  `start_index`, `seed`, `enable_fallback`. Rows are seeded-shuffled per domain
  and then round-robin interleaved across domains, so any contiguous
  `start_index`/`num_examples` carve is domain-balanced and train/eval slices
  are disjoint (browsecomp_plus-style).
- **Source trace:** github.com/LongHorizonReasoning/longcot @ fb96494 (data +
  verifiers) / huggingface.co/datasets/LongHorizonReasoning/longcot (mirror,
  unused — see above) / paper arXiv:2604.14140.

The problem statement appears verbatim in the root prompt AND as the REPL
variable `context` — the long horizon is the required reasoning, not the input,
but the input is not uniformly short: tokenized with the Qwen3-30B-A3B-Instruct
tokenizer, the easy split is median 1.1k tokens / p90 7.8k / max 10.5k (measured
on all 507 rows; dense numeric grids and move lists tokenize far denser than
plain English, ~1.8 chars/token). 31/507 rows (~6%) already exceed `seq_len =
8192` from the context alone, before the RLM system prompt/prologue/completion
budget — `qwen3-30b-longcot-mini-smoke.toml` sets `seq_len = 8192`, so those
rows get tail-truncated by the trainer (zero gradient signal on them) rather
than rejected; widen `seq_len` or filter by prompt length if the long tail
matters for a given run. The prompts embed the paper's no-tools/no-code rule
for its primary track; the prologue explicitly lifts it. This environment is
therefore the paper's Section 4.2 "RLM with code execution" setting — a
separate track whose numbers are NOT comparable to the official single-shot
no-scaffold leaderboard. The model finalizes by setting `answer["content"]` to
the `solution = ...` line and `answer["ready"] = True`.

Verification (`longcot.verify`) runs an untrusted, model-controlled answer
through sympy/RDKit/python-chess with no built-in complexity guard — a
malformed math answer (e.g. an exponent tower) can make sympy's `parse_expr`
hang indefinitely. Scoring isolates each verification call in its own
subprocess with a hard wall-clock timeout (`_VERIFY_TIMEOUT_S = 20.0`,
`longcot_mini/env.py`), killed and scored 0.0 on timeout — this is necessary
because the timeout must be OS-level: a bare in-process `except Exception`
(here or inside the official verifier) does not reliably interrupt a stuck
sympy call.

Default rubric is the stock correctness-only `RLMTrainRubric` (weight 1.0).
Reference single-shot no-scaffold accuracy on LongCoT-Mini for open models:
DeepSeek V3.2 8.3%, Kimi K2 Thinking 7.5%, GLM 4.7 5.9% (GPT-5.2 38.7%);
the paper's RLM-with-code GPT-5.2 study gains only on the procedural domains
(logic/chess/cs) while compositional math/chemistry stay near 0%.
