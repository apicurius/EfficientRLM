"""Invariant tests for the opt-in, correctness-gated efficiency-shaping reward.

These tests pin the THESIS.md guarantees:
- Parity:        shaping_coef=0 reproduces the upstream correctness-only reward.
- Gating:        the bonus only applies to correct, gate-passing rollouts.
- Dominance:     a correct rollout always scores >= an incorrect rollout.
- Monotonicity:  holding correctness fixed, more usage never pays.
- Boundedness:   R in [0, c*(1+lambda)].
"""

from __future__ import annotations

import asyncio

import pytest

from rlm_train import (
    EfficiencyGatedRubric,
    Harness1StyleRubric,
    RLMTrainRubric,
    make_reward_rubric,
)
from rlm_train.env import (
    _MAX_TRAJECTORY_LINE_CHARS,
    _MAX_TRAJECTORY_TEXT_CHARS,
    _accumulate_trajectory_text,
    _record_sub_call,
)
from rlm_train.shaping import EfficiencyAxis, default_axes, efficiency_score


def _state(*, iters=4, sub_calls=8, tokens=10_000, final="ans"):
    return {
        "rlm_iterations": iters,
        "rlm_sub_llm_calls": sub_calls,
        "rlm_sub_llm_tokens": tokens,
        "rlm_final_answer": final,
    }


def _main_func(rubric):
    # The first registered reward func is the main (correctness/shaped) reward.
    return rubric.funcs[0]


def _metric(rubric, name):
    for f in rubric.funcs:
        if getattr(f, "__name__", "") == name:
            return f
    raise KeyError(name)


async def _call(func, state):
    return float(await func(state=state, info={}))


def _correct(value):
    async def c(**kwargs):
        return value

    c.__name__ = "correctness"
    return c


def test_make_reward_rubric_selects_three_setups():
    stock = make_reward_rubric(_correct(1.0), reward_style="correctness")
    gated = make_reward_rubric(
        _correct(1.0),
        reward_style="efficiency",
        shaping_coef=0.2,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    harness = make_reward_rubric(_correct(1.0), reward_style="harness1")
    auto0 = make_reward_rubric(_correct(1.0), shaping_coef=0.0)
    auto2 = make_reward_rubric(_correct(1.0), shaping_coef=0.2)
    assert type(stock) is type(auto0) is RLMTrainRubric
    assert type(gated) is type(auto2) is EfficiencyGatedRubric
    assert type(harness) is Harness1StyleRubric


# --- Parity -----------------------------------------------------------------


def test_parity_with_stock_rubric_when_coef_zero():
    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.0,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    st = _state()
    r_stock = asyncio.run(_call(_main_func(stock), st))
    r_shaped = asyncio.run(_call(_main_func(shaped), st))
    assert r_stock == r_shaped == 1.0


def test_parity_holds_for_partial_correctness():
    stock = RLMTrainRubric(correctness=_correct(0.5), min_iterations=2)
    shaped = EfficiencyGatedRubric(
        correctness=_correct(0.5),
        min_iterations=2,
        shaping_coef=0.5,
        subcall_budget=32.0,
    )
    st = _state()
    # 0.5 < correct_threshold(1.0) -> no bonus even with coef>0.
    assert asyncio.run(_call(_main_func(stock), st)) == 0.5
    assert asyncio.run(_call(_main_func(shaped), st)) == 0.5


# --- Gating -----------------------------------------------------------------


def test_no_bonus_for_incorrect_rollout():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        shaping_coef=0.5,
        subcall_budget=32.0,
    )
    # Cheap-but-wrong: zero usage would maximize e, but base=0 -> reward stays 0.
    st = _state(iters=1, sub_calls=0, tokens=0, final="wrong")
    assert asyncio.run(_call(_main_func(shaped), st)) == 0.0


def test_bonus_applied_for_correct_rollout():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        max_iterations=20,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    st = _state(iters=2, sub_calls=2, tokens=1000)
    r = asyncio.run(_call(_main_func(shaped), st))
    assert r > 1.0


def test_structural_caps_leave_correctness_only_rollouts_tied():
    """Caps stop runaway rollouts; they do not credit-assign thrift.

    Both states satisfy the configured ceilings used in the thesis configs:
    max_iterations=20, subcall_budget=32, token_budget=200k. Under the stock
    correctness-only reward they are indistinguishable even though the second
    rollout spends 6x turns, 10x calls, and >4x tokens.
    """

    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    cheap = _state(iters=3, sub_calls=3, tokens=35_000)
    expensive = _state(iters=18, sub_calls=30, tokens=160_000)
    assert asyncio.run(_call(_main_func(stock), cheap)) == 1.0
    assert asyncio.run(_call(_main_func(stock), expensive)) == 1.0


def test_efficiency_reward_separates_cap_satisfying_correct_rollouts():
    """The reward has operating room inside the structural caps."""

    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.2,
        max_iterations=20,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    cheap = _state(iters=3, sub_calls=3, tokens=35_000)
    expensive = _state(iters=18, sub_calls=30, tokens=160_000)
    cheap_reward = asyncio.run(_call(_main_func(shaped), cheap))
    expensive_reward = asyncio.run(_call(_main_func(shaped), expensive))
    assert cheap_reward > expensive_reward > 1.0
    # Non-saturating rational axis (1 / (1 + used/budget)): efficiency is 0.5 at
    # the budget and decays smoothly past it, so cap-satisfying rollouts are
    # still separated and the expensive tail keeps gradient.
    assert cheap_reward == pytest.approx(1.1756609841)
    assert expensive_reward == pytest.approx(1.1065333585)


# --- Dominance --------------------------------------------------------------


def test_correct_always_beats_incorrect():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.9,
        subcall_budget=32.0,
    )
    wrong = EfficiencyGatedRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        shaping_coef=0.9,
        subcall_budget=32.0,
    )
    # Correct but maximally expensive vs. wrong but maximally cheap.
    r_correct = asyncio.run(
        _call(_main_func(shaped), _state(iters=20, sub_calls=32, tokens=200_000))
    )
    r_wrong = asyncio.run(_call(_main_func(wrong), _state(iters=1, sub_calls=0, tokens=0)))
    assert r_correct >= r_wrong


# --- Monotonicity -----------------------------------------------------------


def test_more_usage_never_pays():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        max_iterations=20,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    cheap = asyncio.run(_call(_main_func(shaped), _state(iters=2, sub_calls=2, tokens=1000)))
    mid = asyncio.run(_call(_main_func(shaped), _state(iters=8, sub_calls=16, tokens=50_000)))
    pricey = asyncio.run(_call(_main_func(shaped), _state(iters=18, sub_calls=60, tokens=190_000)))
    assert cheap >= mid >= pricey >= 1.0


# --- Boundedness ------------------------------------------------------------


def test_reward_is_bounded():
    coef = 0.5
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=coef,
        iteration_weight=0.0,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    # Iteration axis disabled; pass the gate (iters>=2) with zero sub-call/token
    # usage -> max efficiency on enabled axes -> reward == c*(1+coef).
    r = asyncio.run(_call(_main_func(shaped), _state(iters=2, sub_calls=0, tokens=0)))
    assert r == pytest.approx(1.0 * (1.0 + coef))


def test_reward_never_exceeds_bound_under_any_usage():
    coef = 0.5
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=coef,
        max_iterations=20,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    for iters, sub_calls, tokens in [(2, 0, 0), (5, 10, 5000), (20, 32, 200_000)]:
        r = asyncio.run(
            _call(_main_func(shaped), _state(iters=iters, sub_calls=sub_calls, tokens=tokens))
        )
        assert 1.0 <= r <= 1.0 * (1.0 + coef) + 1e-9


# --- efficiency_score helper ------------------------------------------------


def test_efficiency_score_disabled_when_no_axes():
    # No positive budgets -> all axes disabled -> efficiency 0 (safe no-op).
    axes = default_axes(max_iterations=0, subcall_budget=0.0, token_budget=0.0)
    assert efficiency_score(_state(), axes) == 0.0


def test_efficiency_axis_is_non_saturating_and_bounded():
    axis = EfficiencyAxis("rlm_sub_llm_calls", budget=10.0)
    # 1 / (1 + used/budget): full efficiency at zero usage, half at the budget,
    # and a strictly positive but ever-shrinking value past it (never clamps to
    # a flat 0, so the expensive tail keeps gradient).
    assert axis.efficiency({"rlm_sub_llm_calls": 0}) == 1.0
    assert axis.efficiency({"rlm_sub_llm_calls": 10}) == pytest.approx(0.5)
    assert axis.efficiency({"rlm_sub_llm_calls": 25}) == pytest.approx(1.0 / 3.5)
    # Strictly decreasing in usage everywhere on the tail.
    assert axis.efficiency({"rlm_sub_llm_calls": 1000}) > axis.efficiency(
        {"rlm_sub_llm_calls": 4525}
    )
    # Bounded in (0, 1] for any nonnegative usage.
    assert 0.0 < axis.efficiency({"rlm_sub_llm_calls": 10**9}) <= 1.0


def test_efficiency_bonus_keeps_gradient_on_the_subcall_tail():
    """Regression for the B1 saturation defect (REWARD_HARNESS1_ANALYSIS.md).

    The old clamped axis flattened every >budget rollout to one constant, so a
    40-call rollout and a 4525-call rollout were scored identically. The
    corrected axis must keep the cheaper-tail rollout strictly preferred.
    """

    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.2,
        max_iterations=20,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    near = asyncio.run(_call(_main_func(shaped), _state(iters=8, sub_calls=40, tokens=5000)))
    mid = asyncio.run(_call(_main_func(shaped), _state(iters=8, sub_calls=200, tokens=5000)))
    far = asyncio.run(_call(_main_func(shaped), _state(iters=8, sub_calls=4525, tokens=5000)))
    assert near > mid > far > 1.0


def test_negative_coef_rejected():
    with pytest.raises(ValueError):
        EfficiencyGatedRubric(correctness=_correct(1.0), shaping_coef=-0.1)


# --- Gate interaction (min_iterations) --------------------------------------


def test_bonus_requires_passing_iteration_gate():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=3,
        shaping_coef=0.5,
        subcall_budget=32.0,
    )
    # iters=1 < min_iterations=3 -> gate fails -> no bonus, base correctness kept.
    st = _state(iters=1, sub_calls=2, tokens=100)
    assert asyncio.run(_call(_main_func(shaped), st)) == 1.0


# --- Metric surface ---------------------------------------------------------


def test_efficiency_metrics_exposed():
    shaped = EfficiencyGatedRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        shaping_coef=0.5,
        subcall_budget=32.0,
        token_budget=200_000.0,
    )
    names = {getattr(f, "__name__", "") for f in shaped.funcs}
    assert {
        "efficiency_bonus",
        "rlm_efficiency_score",
        "rlm_sub_llm_tokens",
        "rlm_sub_llm_usage_missing",
    } <= names
    bonus = _metric(shaped, "efficiency_bonus")
    st = _state(iters=2, sub_calls=2, tokens=1000)
    assert asyncio.run(_call(bonus, st)) > 0.0


def test_stock_rubric_exposes_token_telemetry_without_shaping():
    stock = RLMTrainRubric(correctness=_correct(1.0), min_iterations=2)
    names = {getattr(f, "__name__", "") for f in stock.funcs}
    assert "rlm_sub_llm_tokens" in names
    assert "rlm_sub_llm_usage_missing" in names
    metric = _metric(stock, "rlm_sub_llm_tokens")
    assert asyncio.run(metric(state=_state(tokens=1234))) == 1234


def test_record_sub_call_accumulates_calls_and_usage_tokens():
    state = {"rlm_sub_llm_calls": 0, "rlm_sub_llm_tokens": 0, "rlm_sub_llm_usage_missing": 0}
    _record_sub_call(state, {"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    _record_sub_call(state, {"usage": {"total_tokens": 7}})
    _record_sub_call(state, {})
    assert state["rlm_sub_llm_calls"] == 3
    assert state["rlm_sub_llm_tokens"] == 22
    assert state["rlm_sub_llm_usage_missing"] == 1


def test_record_sub_call_counts_malformed_usage_as_missing():
    state = {"rlm_sub_llm_calls": 0, "rlm_sub_llm_tokens": 0, "rlm_sub_llm_usage_missing": 0}
    _record_sub_call(state, {"usage": {"total_tokens": "not-an-int"}})
    assert state["rlm_sub_llm_calls"] == 1
    assert state["rlm_sub_llm_tokens"] == 0
    assert state["rlm_sub_llm_usage_missing"] == 1


# --- Trajectory-text digest accumulation (telemetry plumbing) ---------------


def test_accumulate_trajectory_text_keeps_short_diagnostic_lines():
    state = {"rlm_trajectory_text": "", "rlm_trajectory_text_truncated": 0}
    _accumulate_trajectory_text(state, [{"stdout": "counts: entity=14\ncandidate: entity"}])
    assert "entity=14" in state["rlm_trajectory_text"]
    assert "candidate: entity" in state["rlm_trajectory_text"]


def test_accumulate_trajectory_text_drops_raw_context_dumps():
    state = {"rlm_trajectory_text": "", "rlm_trajectory_text_truncated": 0}
    long_line = "x" * (_MAX_TRAJECTORY_LINE_CHARS + 50)
    _accumulate_trajectory_text(state, [{"stdout": long_line}])
    # Over-long lines are raw-context dumps; they carry no compact evidence.
    assert state["rlm_trajectory_text"] == ""


def test_accumulate_trajectory_text_is_bounded():
    state = {"rlm_trajectory_text": "", "rlm_trajectory_text_truncated": 0}
    for _ in range(5000):
        _accumulate_trajectory_text(state, [{"stdout": "label=entity count=3"}])
    assert len(state["rlm_trajectory_text"]) <= _MAX_TRAJECTORY_TEXT_CHARS
    assert state["rlm_trajectory_text_truncated"] == 1


# --- Harness-1-style reward -------------------------------------------------


def test_harness_style_no_final_short_circuits_to_empty_penalty():
    rubric = Harness1StyleRubric(correctness=_correct(1.0), min_iterations=2)
    st = _state(iters=4, sub_calls=2, tokens=1000, final=None)
    assert asyncio.run(_call(_main_func(rubric), st)) == pytest.approx(-0.2)


def test_harness_style_correct_rollout_gets_answer_quality_and_bonus():
    rubric = Harness1StyleRubric(correctness=_correct(1.0), min_iterations=2)
    st = _state(iters=4, sub_calls=2, tokens=1000, final="ans")
    r = asyncio.run(_call(_main_func(rubric), st))
    # Defaults mirror Harness-1's recall/answer terms, minus the small RLM
    # resource penalty. The tool-diversity bonus is intentionally absent (RLM
    # has no tool vocabulary): 0.7 + 0.3 + 0.8 + 0.4 + 1.0 - 0.02*(4/20).
    assert r == pytest.approx(3.196)


def test_harness_style_wrong_but_formatted_scores_below_correct():
    rubric = Harness1StyleRubric(correctness=_correct(0.0), min_iterations=2)
    st = _state(iters=4, sub_calls=0, tokens=1000, final="wrong")
    r = asyncio.run(_call(_main_func(rubric), st))
    # No correctness/answer credit and no diversity bonus: only the tiny
    # resource penalty applies (-0.02 * 4/20), so the wrong rollout is slightly
    # negative and strictly below any correct rollout.
    assert r == pytest.approx(-0.004)
    assert r < asyncio.run(_call(_main_func(Harness1StyleRubric(correctness=_correct(1.0))), st))


def test_harness_style_uses_trajectory_answer_miss_penalty():
    rubric = Harness1StyleRubric(correctness=_correct(0.0), min_iterations=2)
    st = _state(iters=4, sub_calls=2, tokens=1000, final="wrong")
    st["rlm_trajectory_answer_recall"] = 1.0
    r = asyncio.run(_call(_main_func(rubric), st))
    # 0.4 trajectory-answer shaping - 0.35 miss penalty - resource penalty
    # (no diversity bonus): 0.4 - 0.35 - 0.02*(4/20).
    assert r == pytest.approx(0.046)


def test_harness_style_accepts_precision_recall_fbeta_state():
    rubric = Harness1StyleRubric(
        correctness=_correct(0.0),
        outcome_weight=1.0,
        trajectory_recall_weight=0.0,
        final_answer_recall_weight=0.0,
        trajectory_fa_recall_weight=0.0,
        final_answer_bonus=0.0,
    )
    st = _state(iters=4, sub_calls=2, tokens=1000, final="wrong")
    st["rlm_curated_precision"] = 0.5
    st["rlm_curated_recall"] = 1.0
    r = asyncio.run(_call(_main_func(rubric), st))
    # beta=2 weights recall 4x: (1+4)*0.5*1 / (4*0.5+1), minus resource penalty.
    assert r == pytest.approx(5 * 0.5 / 3 - 0.02 * (4 / 20))


def test_harness_style_metrics_exposed():
    rubric = Harness1StyleRubric(correctness=_correct(1.0), min_iterations=2)
    names = {getattr(f, "__name__", "") for f in rubric.funcs}
    assert {
        "rlm_harness_turn_penalty",
        "rlm_harness_resource_penalty",
        "rlm_harness_no_final_short_circuit",
        "rlm_harness_format_floor",
    } <= names


# --- Harness-1-style live trajectory recall (discovery vs selection) --------


def test_harness_style_live_trajectory_recall_callback_credits_discovery():
    """A wrong final answer that *surfaced* the gold mid-trajectory must score
    strictly above a wrong answer that never found it.

    This is the defining Harness-1 mechanism the OOLONG wiring activates: the
    callback computes rho_tauA from rollout state, so two same-correctness
    rollouts are no longer tied at R = c.
    """

    async def found(**kwargs):
        return 1.0 if (kwargs.get("state") or {}).get("surfaced") else 0.0

    rubric = Harness1StyleRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        trajectory_answer_recall=found,
    )
    surfaced = _state(iters=4, sub_calls=2, tokens=1000, final="wrong")
    surfaced["surfaced"] = True
    missed = _state(iters=4, sub_calls=2, tokens=1000, final="wrong")
    r_surfaced = asyncio.run(_call(_main_func(rubric), surfaced))
    r_missed = asyncio.run(_call(_main_func(rubric), missed))
    # Both pay only the resource penalty (0.02*4/20 = 0.004); no diversity bonus.
    # Surfaced adds w_tauA*1.0 (0.4) then pays the miss penalty w_miss*1.0
    # (0.35): net +0.05 for discovering-but-not-promoting. Missed gets neither.
    assert r_surfaced > r_missed
    assert r_surfaced == pytest.approx(0.046)
    assert r_missed == pytest.approx(-0.004)
    assert (r_surfaced - r_missed) == pytest.approx(0.05)


def test_harness_style_live_trajectory_recall_exposed_as_metric():
    async def found(**kwargs):
        return 1.0

    rubric = Harness1StyleRubric(
        correctness=_correct(0.0),
        min_iterations=2,
        trajectory_answer_recall=found,
    )
    names = {getattr(f, "__name__", "") for f in rubric.funcs}
    assert "rlm_harness_trajectory_answer_recall" in names
    metric = _metric(rubric, "rlm_harness_trajectory_answer_recall")
    st = _state(iters=4, sub_calls=2, tokens=1000, final="wrong")
    assert asyncio.run(_call(metric, st)) == pytest.approx(1.0)


def test_harness_style_recall_callback_failure_is_safe():
    """A throwing callback must not crash scoring; it falls back to state."""

    async def boom(**kwargs):
        raise RuntimeError("scorer exploded")

    rubric = Harness1StyleRubric(
        correctness=_correct(1.0),
        min_iterations=2,
        trajectory_answer_recall=boom,
    )
    st = _state(iters=4, sub_calls=2, tokens=1000, final="ans")
    # Correct rollout still scores its full answer reward despite the bad cb.
    assert asyncio.run(_call(_main_func(rubric), st)) == pytest.approx(3.196)
