"""Unit tests for the longcot_mini scorer glue and dataset carving (no GPU/network).

The official `longcot` verifiers are upstream code; what these tests pin is OUR
glue: info-meta -> Question reconstruction, dispatch through `longcot.verify`,
official extraction semantics surviving the round-trip (last `solution =` wins,
whole-response fallbacks), fail-closed behavior (empty answers, verifier
exceptions, disabled Gemini fallback), and the stratified domain-interleaved
ordering that makes contiguous train/eval carves disjoint and balanced.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest

pytest.importorskip("longcot", reason="pinned `longcot` git dependency not installed")

from longcot_mini.env import (  # noqa: E402
    _build_longcot_dataset,
    _score_longcot_mini,
    _verify_final,
    user_prologue,
)


def _meta(**kw) -> dict:
    base = {
        "question_id": "test_q",
        "domain": "logic",
        "difficulty": "easy",
        "context": "irrelevant problem text",
        "answer": None,
        "problem": None,
        "enable_fallback": False,
    }
    base.update(kw)
    return base


class TestVerifyFinal:
    def test_logic_dungeon_exact_int(self):
        meta = _meta(problem={"template": "Dungeon", "solution": 7})
        assert _verify_final(meta, "solution = 7") == 1.0
        assert _verify_final(meta, "solution = 8") == 0.0

    def test_last_solution_line_wins(self):
        meta = _meta(problem={"template": "Dungeon", "solution": 7})
        assert _verify_final(meta, "solution = 8\n...on reflection...\nsolution = 7") == 1.0

    def test_whole_response_fallback_without_solution_line(self):
        # Official behavior: missing `solution =` is only wrong_formatting;
        # verification still runs on the whole response (last int here).
        meta = _meta(problem={"template": "Dungeon", "solution": 7})
        assert _verify_final(meta, "after simulating, the shortest path costs 7") == 1.0

    def test_math_component_list(self):
        meta = _meta(domain="math", problem={"template": "linear"}, answer=["4", "13"])
        assert _verify_final(meta, "solution = [4, 13]") == 1.0
        # Deterministic mismatch fails closed with fallback disabled (no Gemini).
        assert _verify_final(meta, "solution = [4, 14]") == 0.0

    def test_cs_vliw_integer(self):
        meta = _meta(domain="cs", problem={"template": "VLIW"}, answer=218)
        assert _verify_final(meta, "solution = 218") == 1.0
        assert _verify_final(meta, "solution = 219") == 0.0

    def test_cs_hm_json_requires_checkpoint_indices(self):
        meta = _meta(
            domain="cs",
            problem={"template": "HM"},
            answer={
                "q1": "Nat",
                "q2": "Bool",
                "q3": "(Nat × Bool)",
                "q4": 2,
                "q5": [
                    {"j": 1, "binding_str": "i=4; Nat"},
                    {"j": 2, "binding_str": "i=5; Bool"},
                ],
            },
        )
        correct = (
            'solution = {"q1": "Nat", "q2": "Bool", "q3": "(Nat × Bool)", '
            '"q4": 2, "q5": [{"j": 1, "binding_str": "i=4; Nat"}, '
            '{"j": 2, "binding_str": "i=5; Bool"}]}'
        )
        missing_j = (
            'solution = {"q1": "Nat", "q2": "Bool", "q3": "(Nat × Bool)", '
            '"q4": 2, "q5": [{"binding_str": "i=4; Nat"}, '
            '{"binding_str": "i=5; Bool"}]}'
        )
        assert _verify_final(meta, correct) == 1.0
        assert _verify_final(meta, missing_j) == 0.0

    def test_chess_best_move_suffix_stripping(self):
        meta = _meta(domain="chess", problem={"template": "best_move"}, answer="Qg8+")
        assert _verify_final(meta, "solution = Qg8") == 1.0
        assert _verify_final(meta, "solution = Qh8") == 0.0

    def test_chemistry_smiles_canonical_equivalence(self):
        meta = _meta(domain="chemistry", problem={"template": "easy1"}, answer="CCO")
        # OCC is ethanol too: RDKit canonicalization must equate them.
        assert _verify_final(meta, "solution = OCC") == 1.0
        assert _verify_final(meta, "solution = CCC") == 0.0

    def test_empty_final_scores_zero(self):
        meta = _meta(problem={"template": "Dungeon", "solution": 7})
        assert _verify_final(meta, "") == 0.0
        assert _verify_final(meta, "   \n") == 0.0

    def test_verifier_exception_scores_zero(self):
        # Missing/unknown template raises ValueError inside longcot.verify;
        # official run_eval.py counts that as incorrect, so we return 0.0.
        assert _verify_final(_meta(problem=None), "solution = 7") == 0.0
        assert _verify_final(_meta(problem={"template": "NoSuchTemplate"}), "solution = 7") == 0.0


class TestAsyncScorer:
    def test_scorer_reads_state_and_json_info(self):
        info = json.dumps(_meta(problem={"template": "Dungeon", "solution": 7}))
        state = {"rlm_final_answer": "solution = 7"}
        assert asyncio.run(_score_longcot_mini(info, state)) == 1.0

    def test_scorer_no_final_answer(self):
        info = json.dumps(_meta(problem={"template": "Dungeon", "solution": 7}))
        assert asyncio.run(_score_longcot_mini(info, {})) == 0.0


class TestDatasetCarving:
    def test_easy_split_is_official_mini_507(self):
        ds = _build_longcot_dataset(difficulty="easy")
        assert len(ds) == 507

    def test_prefix_is_domain_balanced(self):
        ds = _build_longcot_dataset(difficulty="easy", num_examples=25, seed=42)
        domains = Counter(json.loads(r["info"])["domain"] for r in ds)
        assert domains == {"logic": 5, "cs": 5, "chemistry": 5, "chess": 5, "math": 5}

    def test_start_index_carves_are_disjoint_and_exhaustive(self):
        train = _build_longcot_dataset(difficulty="easy", start_index=0, num_examples=400, seed=42)
        heldout = _build_longcot_dataset(difficulty="easy", start_index=400, seed=42)
        train_ids = {json.loads(r["info"])["question_id"] for r in train}
        held_ids = {json.loads(r["info"])["question_id"] for r in heldout}
        assert not train_ids & held_ids
        assert len(train_ids) == 400 and len(held_ids) == 107

    def test_row_schema_matches_family_convention(self):
        ds = _build_longcot_dataset(difficulty="easy", num_examples=2, seed=42)
        row = ds[0]
        meta = json.loads(row["info"])
        for key in ("id", "question_id", "domain", "difficulty", "template",
                    "answer", "problem", "context", "root_prompt", "source_env"):
            assert key in meta
        # The REPL context IS the verbatim problem statement shown in the prompt.
        assert meta["context"] == row["prompt"][0]["content"]
        assert meta["context"] in meta["root_prompt"]
        assert meta["enable_fallback"] is False

    def test_bad_args_raise(self):
        with pytest.raises(ValueError):
            _build_longcot_dataset(difficulty="mini")
        with pytest.raises(ValueError):
            _build_longcot_dataset(domains="biology")


class TestUserPrologue:
    def test_hm_checkpoint_indices_are_explicitly_prompted(self):
        assert "CS/HM-TRACE" in user_prologue
        assert '"j": <checkpoint index>' in user_prologue
        assert "do not omit `j`" in user_prologue
