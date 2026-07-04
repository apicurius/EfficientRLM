from __future__ import annotations

import pytest
from datasets import Dataset

from rlm_train.env import RLMTrainEnv
from rlm_train.repl.base import ExecResult


class FakeBackend:
    def __init__(self, final_answer: str | None):
        self.final_answer = final_answer
        self.executed: list[str] = []

    async def execute(self, code: str) -> ExecResult:
        self.executed.append(code)
        return ExecResult(
            stdout="",
            stderr="",
            final_answer=self.final_answer,
            execution_time=0.0,
            locals_keys=[],
        )


def _env() -> RLMTrainEnv:
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "q"}], "info": {"context": ["ctx"]}}])
    return RLMTrainEnv(dataset=ds, max_iterations=1, enforce_root_prompt_budget=False)


def _terminal_state(backend: FakeBackend, assistant_text: str) -> dict:
    return {
        "trajectory": [
            {
                "completion": [{"role": "assistant", "content": assistant_text}],
                "prompt": [{"role": "user", "content": "turn"}],
            }
        ],
        "rlm_history": [],
        "rlm_backend": backend,
        "rlm_n_processed": 0,
        "rlm_iterations": 0,
        "rlm_repl_calls": 0,
        "rlm_final_answer": None,
        "rlm_final_repl_outputs": [],
        "rlm_trajectory_text": "",
        "rlm_trajectory_text_truncated": 0,
        "rlm_root_prompt_chars_max": 0,
        "rlm_root_prompt_est_tokens_max": 0,
        "rlm_root_prompt_windowed": 0,
        "rlm_root_prompt_over_budget": 0,
        "rlm_root_prompt_chars_after_window_max": 0,
        "rlm_root_prompt_est_tokens_after_window_max": 0,
        "rlm_root_prompt_omitted_messages_max": 0,
    }


@pytest.mark.asyncio
async def test_has_final_answer_flushes_pending_last_turn_before_max_turns():
    backend = FakeBackend("captured final")
    state = _terminal_state(
        backend,
        'I am done.\n```repl\nanswer["content"] = "captured final"\nanswer["ready"] = True\n```',
    )
    env = _env()

    assert await env.has_final_answer(state) is True

    assert backend.executed == ['answer["content"] = "captured final"\nanswer["ready"] = True']
    assert state["rlm_final_answer"] == "captured final"
    assert state["final_answer"] == "captured final"
    assert state["rlm_n_processed"] == 1
    assert state["rlm_iterations"] == 1
    assert state["rlm_repl_calls"] == 1
    # The normal final-response path is populated, so the rollout renders like
    # an ordinary finalization rather than a max-turn fatal.
    assert state["final_env_response"]
    assert state["rlm_final_repl_outputs"] == state["final_env_response"]


@pytest.mark.asyncio
async def test_has_final_answer_leaves_nonfinal_pending_last_turn_for_max_turns():
    backend = FakeBackend(None)
    state = _terminal_state(backend, "```repl\nprint('not final yet')\n```")
    env = _env()

    assert await env.has_final_answer(state) is False

    assert backend.executed == ["print('not final yet')"]
    assert state.get("rlm_final_answer") is None
    assert "final_answer" not in state
    assert state["rlm_n_processed"] == 1
    assert state["rlm_iterations"] == 1
    assert state["rlm_repl_calls"] == 1
    # Since no final was captured, the existing verifiers max-turn stop still applies.
    assert await env.max_turns_reached(state) is True


def test_has_final_answer_stop_condition_precedes_max_turns():
    env = _env()
    names = [c.__name__ for c in env._stop_conditions]
    assert names.index("has_final_answer") < names.index("max_turns_reached")


@pytest.mark.asyncio
async def test_is_completed_flushes_final_turn_at_cap_end_to_end():
    """Drive the real verifiers stop path (`is_completed`) at the hard cap.

    This is the production shape: verifiers calls `is_completed(state)` at the
    top of the loop, which runs every @vf.stop condition in discovered order.
    A final-turn ```repl``` block that set answer["ready"] = True must be
    flushed by `has_final_answer` and win before `max_turns_reached`, so the
    rollout stops as a normal finalization (stop_condition="has_final_answer")
    and is non-fatal / judge-reachable.
    """
    backend = FakeBackend("captured final")
    # A realistic multi-block last turn: a scan block, then the finalize block.
    state = _terminal_state(
        backend,
        "Reasoning...\n"
        "```repl\nhits = [d for d in context if 'x' in d]\nprint(len(hits))\n```\n"
        "Now finalize.\n"
        '```repl\nanswer["content"] = "captured final"\nanswer["ready"] = True\n```',
    )
    env = _env()  # max_iterations=1 -> max_turns=1, trajectory has 1 step => at cap

    assert await env.is_completed(state) is True
    assert state["stop_condition"] == "has_final_answer"
    assert state["is_completed"] is True
    # Both blocks of the dropped last turn executed, in order.
    assert backend.executed == [
        "hits = [d for d in context if 'x' in d]\nprint(len(hits))",
        'answer["content"] = "captured final"\nanswer["ready"] = True',
    ]
    assert state["rlm_final_answer"] == "captured final"
    assert state["final_env_response"]


@pytest.mark.asyncio
async def test_is_completed_nonfinal_last_turn_stops_as_max_turns():
    """At the cap, a last turn that never finalized must still stop as max_turns
    (the flush executes its code but captures no answer)."""
    backend = FakeBackend(None)
    state = _terminal_state(backend, "```repl\nprint('still working')\n```")
    env = _env()

    assert await env.is_completed(state) is True
    assert state["stop_condition"] == "max_turns_reached"
    assert state.get("rlm_final_answer") is None
    assert backend.executed == ["print('still working')"]


@pytest.mark.asyncio
async def test_flush_is_idempotent_no_double_execution():
    """Calling the stop hook twice must not re-execute already-processed code."""
    backend = FakeBackend("captured final")
    state = _terminal_state(
        backend,
        '```repl\nanswer["content"] = "captured final"\nanswer["ready"] = True\n```',
    )
    env = _env()

    assert await env.has_final_answer(state) is True
    assert await env.has_final_answer(state) is True  # second call: already final
    assert len(backend.executed) == 1  # not re-run
    assert state["rlm_n_processed"] == 1
    assert state["rlm_repl_calls"] == 1
