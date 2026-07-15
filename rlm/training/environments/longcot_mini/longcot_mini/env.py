"""LongCoT-Mini long-horizon reasoning (official easy split), via RLMTrainEnv.

Source-traced from LongHorizonReasoning/longcot (arXiv:2604.14140). Questions
AND the official per-domain verifiers come from the pinned `longcot` package
(github.com/LongHorizonReasoning/longcot @ fb96494); the HF mirror
(LongHorizonReasoning/longcot) is deliberately unused — its parquet lacks the
`problem` payloads the official verifier dispatches on (all logic answers are
JSON null there). The problem statement is exposed verbatim in the root prompt
AND as the REPL variable `context`; the model finalizes via
answer["content"]/answer["ready"]. Scoring is the official binary verifier
with LLM fallbacks disabled by default (deterministic, == `--no-fallback`).
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import random
from typing import Any

import verifiers as vf
from datasets import Dataset
from longcot import (
    ChemistryVerifyOptions,
    MathVerifyOptions,
    Question,
    VerifyOptions,
    load_questions,
    verify,
)

import rlm_train

user_prologue = """LongCoT-Mini environment notes:
- The exact problem statement (the same text as the task above) is available in
  the REPL variable `context` (a str), so you can parse grids, boards, move
  lists, expressions, or SMILES from it programmatically instead of re-typing.
- The problem text may say not to use tools or write code; that restriction
  does NOT apply here — using the REPL and `llm_query`/`llm_query_batched` is
  expected. Everything else in the problem statement (rules, constraints, the
  required output format) still applies.
- Offload state tracking to Python: simulate move sequences, propagate
  constraints, enumerate/search candidates, and check candidate solutions with
  code before answering; use `llm_query` only for subproblems that need
  language reasoning rather than computation.
- Print only compact diagnostics: counts, parsed structures, intermediate
  values, and verification results — not long reasoning dumps.
- Verify your candidate answer against every stated constraint in `context`
  before finalizing.
- CS/HM-TRACE tasks: the required JSON answer includes checkpoint positions.
  For `q5`, return a list of dicts exactly like
  `{"j": <checkpoint index>, "binding_str": "i=<type-var>; <prefix(type)>"}`
  for every requested checkpoint `j`; do not omit `j` just because it is not
  part of the binding payload.
- Final answer: exactly one line in the format the problem specifies, starting
  with `solution = ` (e.g. `solution = [move0, move1, ...]`).
- When ready, set answer["content"] to ONLY that `solution = ...` line and then
  answer["ready"] = True.
"""

_QUESTION_INSTRUCTION = (
    "Solve the following LongCoT long-horizon reasoning problem. Work the "
    "interdependent sub-steps carefully, then give the final answer in the "
    "exact `solution = ...` format the problem specifies."
)

_DOMAINS = ("logic", "cs", "chemistry", "chess", "math")
_DIFFICULTIES = ("easy", "medium", "hard")
_SOURCE_ENV = "LongHorizonReasoning/longcot@fb96494 (LongCoT-Mini = easy split)"


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return [x]
    return list(x)


def _verify_options(enable_fallback: bool) -> VerifyOptions:
    # Fallbacks are the official Gemini judges for borderline math/chemistry;
    # off they fail closed, byte-identical to the harness's --no-fallback mode.
    return VerifyOptions(
        math=MathVerifyOptions(enable_fallback=enable_fallback),
        chemistry=ChemistryVerifyOptions(enable_fallback=enable_fallback),
    )


def _verify_final(meta: dict[str, Any], final: str) -> float:
    if not final.strip():
        return 0.0
    question = Question(
        question_id=str(meta.get("question_id", "")),
        domain=str(meta.get("domain", "")),
        difficulty=str(meta.get("difficulty", "")),
        prompt=str(meta.get("context", "")),
        problem=meta.get("problem"),
        answer=meta.get("answer"),
    )
    try:
        ok = verify(
            question,
            final,
            options=_verify_options(bool(meta.get("enable_fallback", False))),
        )
    except Exception:
        # The official run_eval.py counts verifier exceptions as incorrect.
        return 0.0
    return 1.0 if ok else 0.0


# Hard wall-clock bound on one verification call. The math verifier feeds
# model-controlled answer text through sympy's parse_expr(convert_xor=True,
# evaluate=True) with no complexity guard: a malformed exponent-tower answer
# (e.g. "9^9^9^9") makes sympy try to materialize a number with billions of
# digits and never return — no exception, so nothing above this layer can
# catch it. That is untrusted RL-exploration output, unlike the official
# harness's offline batch of vetted frontier-model responses, so it must be
# isolated in a killable subprocess rather than trusted to finish or raise.
_VERIFY_TIMEOUT_S = 20.0
_mp_ctx = mp.get_context("spawn")


def _verify_worker(meta: dict[str, Any], final: str, out_queue: "mp.Queue[float]") -> None:
    out_queue.put(_verify_final(meta, final))


def _verify_with_timeout(meta: dict[str, Any], final: str) -> float:
    queue: "mp.Queue[float]" = _mp_ctx.Queue()
    proc = _mp_ctx.Process(target=_verify_worker, args=(meta, final, queue), daemon=True)
    proc.start()
    proc.join(_VERIFY_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return 0.0
    try:
        return float(queue.get_nowait())
    except Exception:
        # Process exited without a result (segfault in a native verifier
        # lib, etc.) — same "verifier failure counts as incorrect" policy.
        return 0.0
    finally:
        queue.close()


async def _score_longcot_mini(info, state: vf.State, **_kw: Any) -> float:
    final = str(state.get("rlm_final_answer") or state.get("final_answer") or "")
    meta = json.loads(info) if isinstance(info, str) else info
    # asyncio.to_thread here only waits on proc.join(), which always returns
    # within _VERIFY_TIMEOUT_S — it never blocks the rollout indefinitely.
    return await asyncio.to_thread(_verify_with_timeout, meta, final)


def _build_longcot_dataset(
    *,
    difficulty: str = "easy",
    domains: str | list[str] | None = None,
    question_ids: str | list[str] | None = None,
    num_examples: int = -1,
    start_index: int = 0,
    seed: int = 42,
    enable_fallback: bool = False,
) -> Dataset:
    if difficulty not in _DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of {_DIFFICULTIES} (LongCoT-Mini = 'easy'), got {difficulty!r}"
        )
    wanted = [str(d) for d in _as_list(domains)] or list(_DOMAINS)
    unknown = set(wanted) - set(_DOMAINS)
    if unknown:
        raise ValueError(f"unknown domains {sorted(unknown)}; valid: {_DOMAINS}")
    allowed_ids = {str(x) for x in _as_list(question_ids)} if question_ids is not None else None

    per_domain: dict[str, list[Question]] = {}
    for d in wanted:
        qs = [
            q
            for q in load_questions(domain=d, difficulty=difficulty)
            if allowed_ids is None or str(q.question_id) in allowed_ids
        ]
        per_domain[d] = qs

    # Seeded shuffle within each domain, then round-robin interleave across a
    # seeded domain order: like the family's shuffle-before-truncation idiom,
    # but stratified, so ANY contiguous carve (a num_examples prefix, or a
    # disjoint start_index/num_examples train-eval split) stays domain-balanced
    # — the 5 domains differ too much for a plain global shuffle to give
    # representative small-N subsets.
    rng = random.Random(seed)
    for d in wanted:
        rng.shuffle(per_domain[d])
    order = list(wanted)
    rng.shuffle(order)
    interleaved: list[Question] = []
    depth = 0
    while True:
        layer = [per_domain[d][depth] for d in order if depth < len(per_domain[d])]
        if not layer:
            break
        interleaved.extend(layer)
        depth += 1

    rows: list[dict[str, Any]] = []
    for q in interleaved:
        root_prompt = f"{_QUESTION_INSTRUCTION}\n\nProblem:\n{q.prompt}"
        meta = {
            "id": f"longcot_{difficulty}_{q.question_id}",
            "question_id": q.question_id,
            "domain": q.domain,
            "difficulty": q.difficulty,
            "template": (q.problem or {}).get("template"),
            "answer": q.answer,
            "problem": q.problem,
            "enable_fallback": bool(enable_fallback),
            "context": q.prompt,
            "root_prompt": root_prompt,
            "source_env": _SOURCE_ENV,
        }
        rows.append(
            {
                "example_id": len(rows),
                "prompt": [{"role": "user", "content": q.prompt}],
                "answer": json.dumps(q.answer),
                "info": json.dumps(meta),
            }
        )
    if start_index:
        rows = rows[start_index:]
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]
    return Dataset.from_list(rows)


def load_environment(
    *,
    difficulty: str = "easy",
    domains: str | list[str] | None = None,
    question_ids: str | list[str] | None = None,
    num_examples: int = -1,
    start_index: int = 0,
    seed: int = 42,
    enable_fallback: bool = False,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    user_prologue: str | None = user_prologue,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_longcot_dataset(
        difficulty=difficulty,
        domains=domains,
        question_ids=question_ids,
        num_examples=num_examples,
        start_index=start_index,
        seed=seed,
        enable_fallback=enable_fallback,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score_longcot_mini,
        weight=1.0,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        user_prologue=user_prologue,
        **kwargs,
    )


__all__ = ["load_environment", "user_prologue"]
