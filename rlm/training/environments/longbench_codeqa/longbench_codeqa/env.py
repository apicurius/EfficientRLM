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

# All LongBench-v2 domains share the same 4-choice MCQ schema, so any of them
# can be served by this env; the prologue below keeps the same structure for
# every domain (only the context-description line varies) to avoid introducing
# a prompt confound when comparing domains.
LONGBENCH_DOMAINS = (
    "Code Repository Understanding",
    "Single-Document QA",
    "Multi-Document QA",
    "Long In-context Learning",
    "Long-dialogue History Understanding",
    "Long Structured Data Understanding",
)

_CONTEXT_DESC = {
    "Code Repository Understanding": (
        "The long source-code repository (potentially many files concatenated)"
    ),
    "Single-Document QA": "The long source document",
    "Multi-Document QA": "The collection of long source documents (concatenated)",
    "Long In-context Learning": "The long in-context examples/material",
    "Long-dialogue History Understanding": "The long dialogue history",
    "Long Structured Data Understanding": "The long structured data (e.g. tables)",
}

_PROLOGUE_TEMPLATE = """LongBench-v2 {domain} environment notes:
- {context_desc} is
  available in the REPL variable `context`; do not print, paste, or echo raw
  content or large chunks into REPL output.
- Print only compact diagnostics: matched sections/keys, counts, short
  snippets, and final evidence.
- This is a 4-choice multiple-choice question; read the question and the four
  options (A, B, C, D) in the prompt.
- Search the context in Python for the relevant passages, then use
  chunky `llm_query_batched` calls over the candidate regions; aggregate
  compact results in Python.
- Decide which single option is correct based on the evidence.
- Final answer: ONLY the single letter of the correct choice: A, B, C, or D.
- When ready, set `answer["content"]` to that single letter and then
  `answer["ready"] = True`.
"""


def _prologue_for(domain: str) -> str:
    return _PROLOGUE_TEMPLATE.format(domain=domain, context_desc=_CONTEXT_DESC[domain])


# Sentinel: distinguishes "caller did not pass user_prologue" (pick per-domain
# default) from an explicit None (no prologue).
_PROLOGUE_UNSET: Any = object()


# Preserved verbatim for the pre-registered codeqa suite: the original
# code-specific prologue (mentions files/symbols/grep). New domains use the
# domain-neutral template above.
user_prologue = """LongBench-v2 Code-Repository-QA environment notes:
- The long source-code repository (potentially many files concatenated) is
  available in the REPL variable `context`; do not print, paste, or echo raw
  source or large chunks into REPL output.
- Print only compact diagnostics: matched files/symbols, counts, short
  snippets, and final evidence.
- This is a 4-choice multiple-choice question; read the question and the four
  options (A, B, C, D) in the prompt.
- Search/grep the repo in Python for the relevant symbols and files, then use
  chunky `llm_query_batched` calls over the candidate regions; aggregate
  compact results in Python.
- Decide which single option is correct based on the code evidence.
- Final answer: ONLY the single letter of the correct choice: A, B, C, or D.
- When ready, set `answer["content"]` to that single letter and then
  `answer["ready"] = True`.
"""

_QUESTION_INSTRUCTION = (
    "Answer the following multiple-choice question about the code repository in `context`. "
    "Respond with ONLY the letter (A, B, C, or D)."
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
    domain: str = LONGBENCH_CODE_DOMAIN,
) -> Dataset:
    if domain not in LONGBENCH_DOMAINS:
        raise ValueError(f"unknown LongBench-v2 domain {domain!r}; valid: {LONGBENCH_DOMAINS}")
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    rows = [dict(e) for e in ds if str(e.get("domain", "")) == domain]
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
            "source_env": f"THUDM/LongBench-v2:{domain}",
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
    domain: str = LONGBENCH_CODE_DOMAIN,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    user_prologue: str | None = _PROLOGUE_UNSET,
    **kwargs: Any,
) -> vf.Environment:
    if domain not in LONGBENCH_DOMAINS:
        raise ValueError(f"unknown LongBench-v2 domain {domain!r}; valid: {LONGBENCH_DOMAINS}")
    # Default prologue: the original code-specific text for the pre-registered
    # codeqa domain (byte-identical to before), the domain-neutral template for
    # every other domain. An explicit user_prologue always wins.
    if user_prologue is _PROLOGUE_UNSET:
        user_prologue = (
            globals()["user_prologue"]
            if domain == LONGBENCH_CODE_DOMAIN
            else _prologue_for(domain)
        )
    dataset = _build_longbench_codeqa_dataset(num_examples=num_examples, seed=seed, domain=domain)
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score_longbench_codeqa,
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


__all__ = ["load_environment", "user_prologue", "LONGBENCH_CODE_DOMAIN", "LONGBENCH_DOMAINS"]
