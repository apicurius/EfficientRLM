"""OOLONG-Pairs pairwise-aggregation QA (eval-suite port), via RLMTrainEnv.

Source-traced from PrimeIntellect-ai/research-environments `rlm_oolong_pairs`
and the LMxLM OOLONG-Pairs task description. The long TREC-coarse context is
exposed as the REPL variable `context`; the model finalizes via
answer["content"]/answer["ready"]. Scoring is precision/recall/F1 over the set
of unordered user-ID pairs.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

import rlm_train

user_prologue = """OOLONG-Pairs environment notes:
- The OOLONG TREC-coarse context is available in the REPL variable `context`;
  do not print, paste, or echo raw context or large chunks into REPL output.
- Print only compact diagnostics: counts, short samples, candidate pairs, and
  final evidence.
- Each line looks roughly like `Date: <date> || User: <id> || Instance:
  <question>`; parse lines into user_id and question text in Python.
- For semantic long-context retrieval or aggregation, split the context into
  chunky windows, use `llm_query_batched`, and aggregate compact results in
  Python.
- Labels/categories named in the question are annotations, not stored in the
  context text — never keyword-match them; classify with `llm_query_batched`
  (many lines per prompt, label-only outputs) and aggregate in Python.
- Pair queries ask for unordered user-ID pairs satisfying joint
  label/count/date predicates; use Python for deduping and pair cross-products
  over the classified users.
- Final answer: list every matching pair as `(id1, id2)`, lower ID first, one
  per line; if none match, answer `[]`.
- When ready, set `answer["content"]` to ONLY that pair list (or `[]`) and then
  `answer["ready"] = True`.
"""

_QUESTION_INSTRUCTION = (
    "Each question in the context falls into one of 6 categories: 'numeric "
    "value', 'entity', 'location', 'description and abstract concept', "
    "'abbreviation', 'human being'. "
    "Answer the following OOLONG-Pairs pairwise aggregation question over the long context. "
    "Return ONLY matching pairs as `(id1, id2)`, lower ID first, one pair per line, or `[]`."
)


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (str, int)):
        return [x]
    return list(x)


def _parse_pairs(answer: Any) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()

    def extract(text: str) -> None:
        matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
        if not matches:
            matches = re.findall(r"(\d+)\s*,\s*(\d+)", text)
        for a, b in matches:
            ia, ib = int(a), int(b)
            pairs.add((ia, ib) if ia < ib else (ib, ia))

    if isinstance(answer, list):
        for item in answer:
            extract(str(item))
    else:
        extract(str(answer))
    return pairs


def _score_pairs(predicted: str, ground_truth: Any) -> tuple[float, float, float]:
    predicted = re.sub(r"<think>.*?</think>", "", predicted, flags=re.DOTALL)
    gt_pairs = _parse_pairs(ground_truth)
    stripped = predicted.strip()
    if stripped in ("", "None", "[]") and not gt_pairs:
        return 1.0, 1.0, 1.0
    pred_pairs = _parse_pairs(predicted)
    if not pred_pairs:
        return 0.0, 0.0, 0.0 if gt_pairs else 1.0
    if not gt_pairs:
        return 0.0, 0.0, 0.0
    correct = len(pred_pairs & gt_pairs)
    precision = correct / len(pred_pairs)
    recall = correct / len(gt_pairs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


async def _score_oolong_pairs(info, state: vf.State, **_kw: Any) -> float:
    final = str(state.get("rlm_final_answer") or state.get("final_answer") or "")
    meta = json.loads(info) if isinstance(info, str) else info
    gt = meta.get("answer", [])
    _, _, f1 = _score_pairs(final, gt)
    return float(f1)


def _load_pairs_questions(context_len: int) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="mit-oasys/oolong-pairs",
        repo_type="dataset",
        filename=f"data/oolong-pairs-{context_len}.json",
    )
    with open(path) as f:
        return json.load(f)


def _load_trec_context(context_len: int) -> str:
    # Use streaming so a small OOLONG-Pairs smoke/eval does not materialize the
    # full multi-GB oolong-synth dataset. Prime research-envs iterate the dataset
    # looking for the needed trec_coarse context; this is the same, but lazy.
    ds = load_dataset("oolongbench/oolong-synth", split="validation", streaming=True)
    for ex in ds:
        e = dict(ex)
        if e.get("dataset") == "trec_coarse" and int(e.get("context_len", 0)) == int(context_len):
            return str(e.get("context_window_text", e.get("context", "")))
    raise ValueError(
        f"No trec_coarse context_len={context_len} in oolongbench/oolong-synth validation"
    )


def _build_oolong_pairs_dataset(
    *,
    context_len: int | list[int] = 32768,
    question_ids: str | list[str] | None = None,
    num_examples: int = -1,
    seed: int = 42,
) -> Dataset:
    lens = [int(x) for x in _as_list(context_len)] or [32768]
    allowed_ids = {str(x) for x in _as_list(question_ids)} if question_ids is not None else None
    rows: list[dict[str, Any]] = []
    for cl in lens:
        context = _load_trec_context(cl)
        for q in _load_pairs_questions(cl):
            qid = str(q.get("id"))
            if allowed_ids is not None and qid not in allowed_ids:
                continue
            question = str(q["question"])
            answer_obj = q.get("answer", [])
            root_prompt = f"{_QUESTION_INSTRUCTION}\n\nQuestion: {question}"
            meta = {
                "id": f"oolong_pairs_{cl}_{qid}",
                "question_id": qid,
                "context_len": cl,
                "answer": answer_obj,
                "num_pairs": q.get(
                    "num_pairs", len(answer_obj) if isinstance(answer_obj, list) else 0
                ),
                "context": context,
                "root_prompt": root_prompt,
                "source_env": "PrimeIntellect-ai/research-environments:rlm_oolong_pairs + LMxLM:oolong_pairs",
            }
            rows.append(
                {
                    "example_id": len(rows),
                    "prompt": [{"role": "user", "content": question}],
                    "answer": json.dumps(answer_obj),
                    "info": json.dumps(meta),
                }
            )
    # Seeded shuffle before truncation so a fixed-N eval draws a representative
    # subset, not a biased first-N slice of the questions file.
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]
    return Dataset.from_list(rows)


def load_environment(
    *,
    context_len: int | list[int] = 32768,
    question_ids: str | list[str] | None = None,
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    user_prologue: str | None = user_prologue,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_oolong_pairs_dataset(
        context_len=context_len,
        question_ids=question_ids,
        num_examples=num_examples,
        seed=seed,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score_oolong_pairs,
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
