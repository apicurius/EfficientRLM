"""Unit tests for the deterministic scoring/parsing logic of the four
source-traced RLM eval-suite environments.

These cover the network-free helpers only (no dataset download): choice
extraction, OOLONG-Pairs F1, OOLONG-synth scoring, BrowseComp-Plus answer
extraction / canary decryption / judge parsing. They pin the behaviour the
async `_score_*` rubric functions depend on.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
import browsecomp_plus._judge as bcp_judge
from browsecomp_plus._data import (
    CANARY as _BCP_CANARY,
    _decrypt_string as _bcp_decrypt_string,
    _derive_key as _bcp_derive_key,
    chunk_docs_for_context as _bcp_chunk_docs_for_context,
    split_text as _bcp_split_text,
)
from browsecomp_plus._judge import (
    _extract_exact_answer,
    _normalize_text,
    _parse_verdict as _parse_bcp_judge_correct,
    containment_score as _score_browsecomp_plus,
    make_judge_score as _make_browsecomp_plus_judge_score,
)
from browsecomp_plus.env import _build_dataset as _build_browsecomp_plus_dataset
from longbench_codeqa.env import _extract_choice_letter, _score_longbench_codeqa
from oolong.env import (
    COMPARISON_PHRASES,
    _find_comparison_phrase,
    _score as _score_oolong,
    _synth_score as _oolong_synth_score,
)
from oolong_pairs.env import _parse_pairs, _score_oolong_pairs, _score_pairs


def _run(coro):
    return asyncio.run(coro)


def _state(final):
    return {"rlm_final_answer": final}


# --- LongBench-v2 CodeQA: 4-choice MCQ letter extraction --------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is (C).", "C"),
        ("Answer: B", "B"),
        ("answer=D", "D"),
        ("I think it's A", "A"),
        ("C", "C"),
        ("nothing here", ""),
    ],
)
def test_extract_choice_letter(text, expected):
    assert _extract_choice_letter(text) == expected


def test_longbench_score_correct_and_wrong():
    info = json.dumps({"answer": "B"})
    assert _run(_score_longbench_codeqa(info, _state("Answer: B"))) == 1.0
    assert _run(_score_longbench_codeqa(info, _state("Answer: C"))) == 0.0


def test_longbench_score_rejects_non_letter_gold():
    # A malformed gold (not A-D) can never score 1.0.
    info = json.dumps({"answer": "E"})
    assert _run(_score_longbench_codeqa(info, _state("Answer: E"))) == 0.0


def test_longbench_score_accepts_dict_info():
    # info may arrive as a dict rather than a JSON string.
    assert _run(_score_longbench_codeqa({"answer": "A"}, _state("A"))) == 1.0


# --- OOLONG-Pairs: pair parsing + F1 ----------------------------------------


def test_parse_pairs_orders_low_first_and_dedupes():
    assert _parse_pairs("(3,1)\n(1, 3)\n(2,5)") == {(1, 3), (2, 5)}


def test_parse_pairs_handles_list_input():
    assert _parse_pairs(["(7,2)", "(4,4)"]) == {(2, 7), (4, 4)}


def test_score_pairs_exact_match():
    assert _score_pairs("(1,3)\n(2,5)", [[1, 3], [2, 5]]) == (1.0, 1.0, 1.0)


def test_score_pairs_empty_prediction_and_gold():
    assert _score_pairs("[]", []) == (1.0, 1.0, 1.0)


def test_score_pairs_partial_overlap_f1():
    # predicted {(1,3),(2,5)} vs gold {(1,3)}: precision .5, recall 1, f1 = 2/3.
    precision, recall, f1 = _score_pairs("(1,3)\n(2,5)", [[1, 3]])
    assert precision == 0.5
    assert recall == 1.0
    assert f1 == pytest.approx(2 / 3)


def test_score_pairs_strips_think_block():
    # <think> content must not leak phantom pairs into the prediction.
    p, r, f1 = _score_pairs("<think>(9,9)</think>(1,3)", [[1, 3]])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_score_oolong_pairs_async_f1():
    info = json.dumps({"answer": [[1, 3]]})
    assert _run(_score_oolong_pairs(info, _state("(1,3)"))) == 1.0
    assert _run(_score_oolong_pairs(info, _state("(2,4)"))) == 0.0


# --- OOLONG-synth scoring ---------------------------------------------------


def test_find_comparison_phrase():
    # Suffixed phrases, byte-faithful to upstream alexzhang13/rlm: a bare
    # "more common" (no "than") no longer matches.
    assert _find_comparison_phrase("they are more common than before") == "more common than"
    assert _find_comparison_phrase("they are more common now") is None
    assert _find_comparison_phrase("no phrase here") is None
    assert set(COMPARISON_PHRASES) == {"more common than", "less common than", "same frequency as"}


def test_oolong_exact_string_match():
    assert _oolong_synth_score({"answer": "['entity']"}, "Answer: entity") == 1.0


def test_oolong_comparison_phrase_match():
    # No-suffix gold (OOLONG before/after family) is credited via the substring
    # fallback, same as upstream.
    assert _oolong_synth_score({"answer": "['more common']"}, "result is more common") == 1.0
    # With-suffix gold (A-vs-B family): a full-phrase answer scores 1.0.
    assert (
        _oolong_synth_score({"answer": "['more common than']"}, "Answer: X is more common than Y")
        == 1.0
    )


def test_oolong_comparison_bare_answer_matches_upstream_zero():
    # Upstream-parity regression guard: a bare "more common" answer against a
    # suffixed gold scores 0.0 (upstream behavior). Before the upstream-parity
    # fix, ERLM over-graded this to 1.0 via a now-removed `trimmed in gold_s`
    # clause; the HF eval-picture scorer scores it 0.0.
    assert _oolong_synth_score({"answer": "['more common than']"}, "result is more common") == 0.0


def test_oolong_numeric_partial_credit_decays():
    meta = {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"}
    exact = _oolong_synth_score(meta, "Answer: 10")
    off_by_two = _oolong_synth_score(meta, "Answer: 12")
    assert exact == 1.0
    assert 0.0 < off_by_two < 1.0
    assert off_by_two == pytest.approx(0.75**2)


def test_oolong_wrong_answer_scores_zero():
    assert _oolong_synth_score({"answer": "['location']"}, "Answer: entity") == 0.0


def test_score_oolong_async():
    info = json.dumps({"answer": "['entity']"})
    assert _run(_score_oolong(info, _state("Answer: entity"))) == 1.0


# --- BrowseComp-Plus: answer extraction, decryption, judge parsing ----------


def test_extract_exact_answer():
    block = "Explanation: because reasons\nExact Answer: Marie Curie\nConfidence: 85%"
    assert _extract_exact_answer(block) == "Marie Curie"


def test_extract_exact_answer_falls_back_to_full_output():
    assert _extract_exact_answer("just a bare answer") == "just a bare answer"


def test_normalize_text_strips_punctuation_and_case():
    assert _normalize_text("New York, NY!") == "new york ny"


def test_bcp_decrypt_roundtrip():
    plain = "the quick brown fox"
    key = _bcp_derive_key(_BCP_CANARY, len(plain.encode()))
    enc = base64.b64encode(bytes(a ^ b for a, b in zip(plain.encode(), key, strict=True))).decode()
    assert _bcp_decrypt_string(enc) == plain


def test_bcp_decrypt_non_base64_returns_input():
    # Non-base64 strings (e.g. already-plaintext fields) pass through unchanged.
    assert _bcp_decrypt_string("not base64 !!!") == "not base64 !!!"


def test_bcp_split_text_respects_budget_and_overlap():
    text = " ".join(f"tok{i:03d}" for i in range(80))
    chunks = _bcp_split_text(text, max_chars=90, overlap=12)

    assert len(chunks) > 1
    assert all(len(c) <= 90 for c in chunks)
    assert all(c for c in chunks)


def test_bcp_context_chunking_adds_provenance_headers():
    doc = " ".join(f"word{i:03d}" for i in range(60))
    chunks = _bcp_chunk_docs_for_context([doc], chunk_chars=100, chunk_overlap=10)

    assert len(chunks) > 1
    assert chunks[0].startswith("[BrowseComp+ doc 0 chunk 1/")
    assert all(c.split("\n", 1)[0].startswith("[BrowseComp+ doc 0 chunk ") for c in chunks)
    assert all(len(c.split("\n", 1)[1]) <= 100 for c in chunks)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # LMxLM / BrowseComp-Plus judge format: {"is_correct": true|false}
        ('{"is_correct": true}', True),
        ('{"is_correct": false}', False),
        ('```json\n{"is_correct": true}\n```', True),  # markdown fence tolerated
        ('Here is my verdict:\n{"is_correct": false}', False),  # leading text
        ('{\n  "is_correct": true\n}', True),  # pretty-printed
        ("no verdict line", False),  # keyword fallback -> not correct
    ],
)
def test_parse_bcp_judge_correct(raw, expected):
    assert _parse_bcp_judge_correct(raw) is expected


def test_bcp_deterministic_score_containment():
    info = json.dumps({"answer": "Marie Curie"})
    good = "Explanation: ...\nExact Answer: Marie Curie\nConfidence: 90%"
    bad = "Explanation: ...\nExact Answer: Albert Einstein\nConfidence: 90%"
    assert _run(_score_browsecomp_plus(info, _state(good))) == 1.0
    assert _run(_score_browsecomp_plus(info, _state(bad))) == 0.0


def test_bcp_deterministic_score_empty_gold_is_zero():
    info = json.dumps({"answer": ""})
    assert _run(_score_browsecomp_plus(info, _state("Exact Answer: anything"))) == 0.0


def test_bcp_dataset_start_index_creates_disjoint_slices(monkeypatch):
    rows = [
        {
            "query_id": f"q{i}",
            "query": f"question {i}",
            "answer": f"answer {i}",
            "gold_docs": [{"docid": f"g{i}", "text": f"gold {i}"}],
            "evidence_docs": [],
            "negative_docs": [],
        }
        for i in range(6)
    ]

    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: rows)

    first = _build_browsecomp_plus_dataset(num_examples=3, start_index=0, k=1, seed=7)
    second = _build_browsecomp_plus_dataset(num_examples=3, start_index=3, k=1, seed=7)

    first_ids = {json.loads(r["info"])["id"] for r in first}
    second_ids = {json.loads(r["info"])["id"] for r in second}
    assert len(first_ids) == 3
    assert len(second_ids) == 3
    assert first_ids.isdisjoint(second_ids)


def test_bcp_dataset_exposes_chunked_context(monkeypatch):
    rows = [
        {
            "query_id": "q0",
            "query": "question",
            "answer": "answer",
            "gold_docs": [{"docid": "g0", "text": " ".join(f"word{i:03d}" for i in range(50))}],
            "evidence_docs": [],
            "negative_docs": [],
        }
    ]

    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: rows)

    ds = _build_browsecomp_plus_dataset(
        num_examples=1,
        k=1,
        seed=7,
        context_chunk_chars=100,
        context_chunk_overlap=10,
    )
    info = json.loads(ds[0]["info"])

    assert info["num_docs"] == 1
    assert info["num_context_chunks"] > 1
    assert info["context_is_chunked"] == 1
    assert all(c.startswith("[BrowseComp+ doc 0 chunk ") for c in info["context"])


def test_bcp_external_openai_judge_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = _make_browsecomp_plus_judge_score("openai/gpt-5-nano")
    info = json.dumps({"raw_question": "q", "answer": "a"})
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        _run(judge(info, {"rlm_final_answer": "a"}))


def test_bcp_external_judge_short_circuits_without_final_answer(monkeypatch):
    # No submitted final answer means no prediction to judge.  This is why a
    # rollout that burns all turns on rejected context-scanning calls returns 0
    # before any external OpenAI judge request can be made.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = _make_browsecomp_plus_judge_score("openai/gpt-5-nano")
    info = json.dumps({"raw_question": "q", "answer": "a"})

    assert _run(judge(info, {"rlm_final_answer": ""})) == 0.0


def test_bcp_external_gpt5_judge_uses_supported_chat_completion_args(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content='{"is_correct": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(bcp_judge, "_get_openai_judge_client", lambda: fake_client)

    judge = _make_browsecomp_plus_judge_score("openai/gpt-5-nano")
    info = json.dumps({"raw_question": "Capital of France?", "answer": "Paris"})

    assert _run(judge(info, {"rlm_final_answer": "Exact Answer: Paris"})) == 1.0
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-5-nano"
    assert calls[0]["max_completion_tokens"] == 512
    assert calls[0]["reasoning_effort"] == "minimal"
    assert "max_tokens" not in calls[0]
    assert "temperature" not in calls[0]


def test_bcp_external_legacy_openai_judge_keeps_classic_args(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content='{"is_correct": false}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(bcp_judge, "_get_openai_judge_client", lambda: fake_client)

    judge = _make_browsecomp_plus_judge_score("openai/gpt-4.1")
    info = json.dumps({"raw_question": "Capital of France?", "answer": "Paris"})

    assert _run(judge(info, {"rlm_final_answer": "Exact Answer: Lyon"})) == 0.0
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-4.1"
    assert calls[0]["max_tokens"] == 256
    assert calls[0]["temperature"] == 0.0
    assert "max_completion_tokens" not in calls[0]
    assert "reasoning_effort" not in calls[0]
