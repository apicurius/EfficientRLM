"""BrowseComp-Plus evidence-document deep-research QA via RLMTrainEnv.

Layout follows LMxLM ``lm_to_program/browsecomp_plus`` (separate ``_data`` /
``_judge`` / ``description`` modules); all code is the RLM training scaffold.
Evidence docs are exposed as the REPL variable ``context`` (list[str] of
budget-safe chunks); the model finalizes via answer["content"]/answer["ready"].
"""

from __future__ import annotations

import json
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

from browsecomp_plus._data import (
    DEFAULT_CONTEXT_CHUNK_CHARS,
    DEFAULT_CONTEXT_CHUNK_OVERLAP,
    chunk_docs_for_context,
    load_corpus,
    load_queries,
    query_rng,
    sample_docs_for_query,
)
from browsecomp_plus._judge import containment_score, make_judge_score
from browsecomp_plus.description import QUESTION_INSTRUCTION, user_prologue


def _build_dataset(
    *,
    num_examples: int = 150,
    k: int = 40,
    seed: int = 42,
    start_index: int = 0,
    include_distractors: bool = False,
    context_chunk_chars: int = DEFAULT_CONTEXT_CHUNK_CHARS,
    context_chunk_overlap: int = DEFAULT_CONTEXT_CHUNK_OVERLAP,
) -> Dataset:
    rows = load_queries(num_examples=num_examples, seed=seed, start_index=start_index)
    corpus = load_corpus() if include_distractors else None

    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        query = str(r.get("query", ""))
        answer = str(r.get("answer", ""))
        if not query:
            continue
        rng = query_rng(seed, str(r.get("query_id", i)))
        doc_texts = sample_docs_for_query(r, k, rng, corpus=corpus)
        context_chunks = chunk_docs_for_context(
            doc_texts,
            chunk_chars=context_chunk_chars,
            chunk_overlap=context_chunk_overlap,
        )
        meta = {
            "id": f"bcplus_{r.get('query_id', i)}",
            "raw_question": query,
            "answer": answer,
            "num_docs": len(doc_texts),
            "num_context_chunks": len(context_chunks),
            "context_is_chunked": int(context_chunk_chars > 0),
            "context_chunk_chars": int(context_chunk_chars or 0),
            "context_chunk_overlap": int(context_chunk_overlap or 0),
            "context": context_chunks,  # list[str] exposed as REPL `context`
            "root_prompt": f"{QUESTION_INSTRUCTION}\n\nQuestion: {query}",
            "source_env": "texttron/BrowseComp-Plus via RLMTrainEnv (LMxLM-layout port)",
        }
        out.append(
            {
                "example_id": i,
                "prompt": [{"role": "user", "content": query}],
                "answer": answer,
                "info": json.dumps(meta),
            }
        )
    return Dataset.from_list(out)


def load_environment(
    *,
    num_examples: int = 150,
    k: int = 40,
    seed: int = 42,
    start_index: int = 0,
    include_distractors: bool = False,
    context_chunk_chars: int = DEFAULT_CONTEXT_CHUNK_CHARS,
    context_chunk_overlap: int = DEFAULT_CONTEXT_CHUNK_OVERLAP,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 2,
    min_subcall: int = 0,
    reward_mode: str = "judge",
    judge_model: str | None = None,
    user_prologue: str | None = user_prologue,
    **kwargs: Any,
) -> vf.Environment:
    dataset = _build_dataset(
        num_examples=num_examples,
        k=k,
        seed=seed,
        start_index=start_index,
        include_distractors=include_distractors,
        context_chunk_chars=context_chunk_chars,
        context_chunk_overlap=context_chunk_overlap,
    )
    if reward_mode not in ("judge", "containment"):
        raise ValueError(f"Unknown reward_mode: {reward_mode!r}; valid: ['judge', 'containment']")
    correctness = make_judge_score(judge_model) if reward_mode == "judge" else containment_score
    rubric = rlm_train.RLMTrainRubric(
        correctness=correctness,
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


__all__ = [
    "load_environment",
    "user_prologue",
]
