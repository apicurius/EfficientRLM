"""LongBench-v2 Code Repository Understanding QA (eval-suite port).

Source: THUDM/LongBench-v2 filtered to domain == "Code Repository
Understanding" (exactly 50 examples). 4-choice MCQ (gold = letter A/B/C/D), so
scoring is deterministic (no judge). The long repo context is exposed as the
REPL variable `context`; the model finalizes via answer["content"]/answer["ready"].
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

LONGBENCH_CODE_DOMAIN = "Code Repository Understanding"

user_prologue = """LongBench-v2 Code-Repository-QA environment notes:
- The REPL variable `context` holds a long source-code repository (potentially
  many files concatenated); do not print, paste, or echo raw source or large
  chunks into REPL output.
- This is a 4-choice multiple-choice question; read the question and the four
  options (A, B, C, D) in the prompt.
- Search/grep the repo in Python for the relevant symbols and files, then use
  chunky `llm_query_batched` calls over the candidate regions; aggregate compact
  results in Python.
- Decide which single option is correct based on the code evidence.
- Final answer: output ONLY the single letter of the correct choice: A, B, C,
  or D.
- When ready, set `answer["content"]` to that single letter and then
  `answer["ready"] = True`.
"""
_QUESTION_INSTRUCTION = (
    "Answer the following multiple-choice question about the code repository in `context`. "
    "Respond with ONLY the letter (A, B, C, or D)."
)


def _build_rubric(
    correctness: Any,
    *,
    min_iterations: int,
    min_subcall: int,
    max_iterations: int,
    shaping_coef: float = 0.0,
    correct_threshold: float = 1.0,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
    reward_style: str = "auto",
    turn_penalty_min_turns: int = 20,
    turn_penalty_max: float = 0.02,
    resource_penalty_max: float = 0.02,
) -> vf.Rubric:
    return rlm_train.make_reward_rubric(
        correctness=correctness,
        weight=1.0,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
        max_iterations=max_iterations,
        reward_style=reward_style,
        shaping_coef=shaping_coef,
        correct_threshold=correct_threshold,
        subcall_budget=subcall_budget,
        token_budget=token_budget,
        iteration_weight=iteration_weight,
        subcall_weight=subcall_weight,
        token_weight=token_weight,
        turn_penalty_min_turns=turn_penalty_min_turns,
        turn_penalty_max=turn_penalty_max,
        resource_penalty_max=resource_penalty_max,
    )

def _extract_choice_letter(output: str) -> str:
    text = str(output).strip()
    m = re.search(r"answer\s*[:=]?\s*\(?([ABCD])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([ABCD])\b", text.upper())
    return m2.group(1) if m2 else ""


async def _score_longbench_codeqa(info, state: vf.State, **_kw: Any) -> float:
    final = str(state.get("rlm_final_answer") or state.get("final_answer") or "")
    meta = json.loads(info) if isinstance(info, str) else info
    gold = str(meta.get("answer", "")).strip().upper()
    return 1.0 if gold in {"A", "B", "C", "D"} and _extract_choice_letter(final) == gold else 0.0


def _build_longbench_codeqa_dataset(
    *,
    num_examples: int = 50,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    rows = [dict(e) for e in ds if str(e.get("domain", "")) == LONGBENCH_CODE_DOMAIN]
    # Seeded shuffle before truncation so a fixed-N eval draws a representative
    # subset, not a biased first-N slice.
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]

    out: list[dict[str, Any]] = []
    for i, s in enumerate(rows):
        question = str(s.get("question", ""))
        choices = (
            f"A. {s.get('choice_A', '')}\n"
            f"B. {s.get('choice_B', '')}\n"
            f"C. {s.get('choice_C', '')}\n"
            f"D. {s.get('choice_D', '')}"
        )
        context = str(s.get("context", ""))
        root_prompt = f"{_QUESTION_INSTRUCTION}\n\nQuestion: {question}\n\n{choices}"
        meta = {
            "id": str(s.get("_id", f"lbv2_code_{i}")),
            "answer": str(s.get("answer", "")),
            "sub_domain": s.get("sub_domain", ""),
            "difficulty": s.get("difficulty", ""),
            "context": context,
            "root_prompt": root_prompt,
            "source_env": "THUDM/LongBench-v2:Code Repository Understanding",
        }
        out.append(
            {
                "example_id": i,
                "prompt": [{"role": "user", "content": question}],
                "answer": str(s.get("answer", "")),
                "info": json.dumps(meta),
            }
        )
    return Dataset.from_list(out)


def load_environment(
    *,
    num_examples: int = 50,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    user_prologue: str | None = user_prologue,
    shaping_coef: float = 0.0,
    correct_threshold: float = 1.0,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
    reward_style: str = "auto",
    turn_penalty_min_turns: int = 20,
    turn_penalty_max: float = 0.02,
    resource_penalty_max: float = 0.02,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_longbench_codeqa_dataset(num_examples=num_examples, seed=seed)
    rubric = _build_rubric(
        _score_longbench_codeqa,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
        max_iterations=max_iterations,
        shaping_coef=shaping_coef,
        correct_threshold=correct_threshold,
        subcall_budget=subcall_budget,
        token_budget=token_budget,
        iteration_weight=iteration_weight,
        subcall_weight=subcall_weight,
        token_weight=token_weight,
        reward_style=reward_style,
        turn_penalty_min_turns=turn_penalty_min_turns,
        turn_penalty_max=turn_penalty_max,
        resource_penalty_max=resource_penalty_max,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        user_prologue=user_prologue,
        **kwargs,
    )


__all__ = ["load_environment", "user_prologue", "LONGBENCH_CODE_DOMAIN"]
