from __future__ import annotations

from types import SimpleNamespace

from prime_rl.orchestrator.advantage import AdvantageInputs
from rlm_train.adaptive_group import adaptive_group_advantage


def _trace(
    *,
    gated: float,
    iters: float = 5.0,
    subcalls: float = 0.0,
    repl_calls: float = 0.0,
    has_final: bool = True,
    stop: str = "has_final_answer",
    reward: float | None = None,
    errors=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metrics={
            "gated_reward": float(gated),
            "rlm_has_final_answer": 1.0 if has_final else 0.0,
            "rlm_iterations": float(iters),
            "rlm_sub_llm_calls": float(subcalls),
            "rlm_repl_calls": float(repl_calls),
        },
        stop_condition=stop,
        reward=float(gated if reward is None else reward),
        errors=list(errors or []),
    )


def _adv(traces, **kwargs):
    return adaptive_group_advantage(AdvantageInputs(rollouts=traces), **kwargs)


def _centered(values):
    mean = sum(values) / len(values)
    return [v - mean for v in values]


def test_parity_below_solve_floor_one_correct_in_four():
    traces = [
        _trace(gated=1.0, iters=3.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    assert _adv(traces, solve_floor=0.25).advantages == _centered([1.0, 0.0, 0.0, 0.0])


def test_parity_all_wrong():
    traces = [_trace(gated=0.0, has_final=False, stop="max_turns") for _ in range(4)]
    assert _adv(traces).advantages == [0.0, 0.0, 0.0, 0.0]


def test_valid_only_cost_ranking_cheaper_gets_higher_advantage():
    traces = [
        _trace(gated=1.0, iters=3.0, subcalls=0.0),
        _trace(gated=1.0, iters=18.0, subcalls=30.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    cheap, expensive, wrong = _adv(traces, beta_max=0.15, solve_floor=0.25).advantages
    assert cheap > expensive
    assert wrong < cheap


def test_cheap_wrong_not_promoted_above_valid_correct():
    traces = [
        _trace(gated=1.0, iters=20.0, subcalls=40.0),
        _trace(gated=1.0, iters=2.0, subcalls=0.0),
        _trace(gated=0.0, iters=1.0, subcalls=0.0),
    ]
    valid_expensive, valid_cheap, cheap_wrong = _adv(traces, beta_max=0.15).advantages
    assert cheap_wrong < valid_expensive
    assert cheap_wrong < valid_cheap


def test_max_output_tokens_is_not_fatal():
    traces = [
        _trace(gated=1.0, iters=3.0, subcalls=0.0, stop="max_output_tokens"),
        _trace(gated=1.0, iters=15.0, subcalls=20.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    max_output, normal, wrong = _adv(traces, beta_max=0.15).advantages
    assert max_output > normal
    assert max_output > wrong


def test_max_turn_correct_looking_is_invalid():
    traces = [
        _trace(gated=1.0),
        _trace(gated=1.0, stop="max_turns"),
        _trace(gated=1.0),
    ]
    out = _adv(traces, beta_max=0.15).advantages
    assert out[1] < out[0]
    assert out[1] < out[2]


def test_no_final_is_invalid():
    traces = [
        _trace(gated=1.0),
        _trace(gated=1.0, has_final=False, stop="max_output_tokens"),
        _trace(gated=1.0),
    ]
    out = _adv(traces, beta_max=0.15).advantages
    assert out[1] < out[0]
    assert out[1] < out[2]


def test_dead_worker_is_invalid():
    traces = [
        _trace(gated=1.0),
        _trace(gated=1.0, errors=["worker closed stdout; Connection lost"]),
        _trace(gated=1.0),
    ]
    out = _adv(traces, beta_max=0.15).advantages
    assert out[1] < out[0]
    assert out[1] < out[2]


def test_small_valid_set_dead_zone_no_cost_pressure_no_crash():
    traces = [
        _trace(gated=1.0, iters=17.0, subcalls=25.0),
        _trace(gated=0.0, iters=1.0, subcalls=0.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    assert _adv(traces, beta_max=0.15).advantages == _centered([1.0, 0.0, 0.0, 0.0])


def test_beta_max_zero_reduces_to_correctness_grpo():
    traces = [
        _trace(gated=1.0, iters=2.0),
        _trace(gated=1.0, iters=19.0, subcalls=50.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    assert _adv(traces, beta_max=0.0).advantages == _centered([1.0, 1.0, 0.0, 0.0])


def test_correct_looking_fatal_invalid_at_solve_floor():
    # solve_rate at floor => beta=0; fatal-but-correct-looking must still be gated to 0.
    traces = [
        _trace(gated=1.0),
        _trace(gated=1.0, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    assert _adv(traces, solve_floor=0.25).advantages == _centered([1.0, 0.0, 0.0, 0.0])


def test_base_reward_keeps_style_reward_and_centers():
    # base="reward" keeps each valid rollout's own (style) reward, no cost term.
    traces = [
        _trace(gated=1.0, iters=2.0, reward=1.30),
        _trace(gated=1.0, iters=19.0, subcalls=50.0, reward=1.05),
        _trace(gated=0.0, has_final=False, stop="max_turns", reward=0.0),
    ]
    out = _adv(traces, base="reward").advantages
    assert out == _centered([1.30, 1.05, 0.0])
    # higher style reward -> higher advantage; cost is NOT re-ranked here.
    assert out[0] > out[1] > out[2]


def test_base_reward_zeros_invalid_rollouts():
    # A high reward on a fatal rollout is discarded (validity gate still applies).
    traces = [
        _trace(gated=1.0, reward=1.0),
        _trace(gated=1.0, reward=9.0, stop="max_turns"),
        _trace(gated=1.0, reward=9.0, has_final=False, stop="max_output_tokens"),
    ]
    out = _adv(traces, base="reward").advantages
    assert out == _centered([1.0, 0.0, 0.0])
    assert out[1] < out[0]
    assert out[2] < out[0]


def test_unknown_base_raises():
    import pytest

    with pytest.raises(ValueError):
        _adv([_trace(gated=1.0)], base="bogus")


def test_records_adaptive_telemetry_into_trace_metrics():
    # base="correctness": two valid-correct (costs 3 and 13), two fatal-wrong.
    # solve_rate=0.5; beta = 0.15 * ((0.5-0.25)/0.75)**1 = 0.05.
    import pytest

    traces = [
        _trace(gated=1.0, iters=3.0, subcalls=0.0),
        _trace(gated=1.0, iters=13.0, subcalls=0.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    out = _adv(traces, base="correctness", beta_max=0.15, solve_floor=0.25, gamma=1.0)

    # telemetry must not change the advantages themselves
    assert out.advantages == _centered([1.0, 0.95, 0.0, 0.0])

    # group-level telemetry, identical on every member
    for t in traces:
        assert t.metrics["adaptive_group_solve_rate"] == pytest.approx(0.5)
        assert t.metrics["adaptive_beta"] == pytest.approx(0.05)

    # per-rollout telemetry
    assert [t.metrics["adaptive_valid"] for t in traces] == [1.0, 1.0, 0.0, 0.0]
    assert [t.metrics["adaptive_advantage"] for t in traces] == [pytest.approx(a) for a in out.advantages]
    assert traces[0].metrics["adaptive_shaped"] == pytest.approx(1.0)
    assert traces[1].metrics["adaptive_shaped"] == pytest.approx(0.95)
    assert traces[0].metrics["adaptive_normalized_cost"] == pytest.approx(0.0)
    assert traces[1].metrics["adaptive_normalized_cost"] == pytest.approx(1.0)
    assert traces[0].metrics["adaptive_cost"] == pytest.approx(3.0)
    assert traces[1].metrics["adaptive_cost"] == pytest.approx(13.0)


def test_unknown_cost_basis_raises():
    import pytest

    with pytest.raises(ValueError):
        _adv([_trace(gated=1.0), _trace(gated=1.0)], cost_basis="total_tokens")


def test_cost_basis_iterations_ignores_helper_calls():
    # Two valid-correct, SAME iterations but very different subcall/repl spend.
    # Under cost_basis="iterations", cost is identical -> no cost re-ranking.
    traces = [
        _trace(gated=1.0, iters=6.0, subcalls=0.0, repl_calls=0.0),
        _trace(gated=1.0, iters=6.0, subcalls=200.0, repl_calls=50.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    out = _adv(traces, base="correctness", cost_basis="iterations").advantages
    assert out[0] == out[1]  # equal turns -> equal cost -> equal advantage


def test_cost_basis_log_helpers_counts_repl_default_does_not():
    # Two valid-correct, identical iterations and subcalls, differ only in REPL calls.
    def group(rb):
        return [
            _trace(gated=1.0, iters=5.0, subcalls=2.0, repl_calls=0.0),
            _trace(gated=1.0, iters=5.0, subcalls=2.0, repl_calls=40.0),
            _trace(gated=0.0, has_final=False, stop="max_turns"),
        ], rb

    # Default basis ignores repl -> the two valid rollouts are tied.
    default_traces, _ = group("iterations_log_subcalls")
    d = _adv(default_traces, base="correctness", cost_basis="iterations_log_subcalls").advantages
    assert d[0] == d[1]

    # Helpers basis folds repl into cost -> the heavy-repl rollout is cheaper-ranked lower.
    helper_traces, _ = group("iterations_log_helpers")
    h = _adv(helper_traces, base="correctness", cost_basis="iterations_log_helpers").advantages
    assert h[0] > h[1]


def test_min_span_default_zero_still_ranks_tiny_spans():
    # Default min_span=0.0 is byte-identical to the old behavior: a one-subcall
    # cost difference (span = log1p(4)-log1p(3) ~= 0.22) still re-ranks.
    traces = [
        _trace(gated=1.0, iters=5.0, subcalls=3.0),
        _trace(gated=1.0, iters=5.0, subcalls=4.0),
        _trace(gated=1.0, iters=5.0, subcalls=3.0),
        _trace(gated=1.0, iters=5.0, subcalls=4.0),
    ]
    out = _adv(traces, beta_max=0.15).advantages
    assert out[0] > 0.0 > out[1]


def test_min_span_gates_noise_span_all_valid_group_to_zero():
    # Same group under min_span=1.0: the 0.22 span is below the noise floor,
    # re-ranking is skipped, the all-valid group centers to exactly 0.0
    # (and is then dropped by the zero_advantage filter, like an equal-cost group).
    traces = [
        _trace(gated=1.0, iters=5.0, subcalls=3.0),
        _trace(gated=1.0, iters=5.0, subcalls=4.0),
        _trace(gated=1.0, iters=5.0, subcalls=3.0),
        _trace(gated=1.0, iters=5.0, subcalls=4.0),
    ]
    out = _adv(traces, beta_max=0.15, min_span=1.0).advantages
    assert out == [0.0, 0.0, 0.0, 0.0]
    for t in traces:
        assert t.metrics["adaptive_normalized_cost"] == 0.0
        assert t.metrics["adaptive_shaped"] == 1.0


def test_min_span_keeps_ranking_above_the_floor():
    # A real cost gap (span = 7 turns) clears min_span=1.0 and is ranked as before.
    traces = [
        _trace(gated=1.0, iters=3.0),
        _trace(gated=1.0, iters=10.0),
        _trace(gated=1.0, iters=3.0),
        _trace(gated=1.0, iters=10.0),
    ]
    out = _adv(traces, beta_max=0.15, min_span=1.0).advantages
    assert out[0] > 0.0 > out[1]


def test_min_span_mixed_group_reverts_to_correctness_contrast():
    # In a mixed group a sub-floor span only removes the cost wiggle between the
    # valid rollouts; the correctness contrast is untouched.
    traces = [
        _trace(gated=1.0, iters=5.0, subcalls=3.0),
        _trace(gated=1.0, iters=5.0, subcalls=4.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    out = _adv(traces, beta_max=0.15, min_span=1.0).advantages
    assert out == _centered([1.0, 1.0, 0.0])
    assert out[0] == out[1] > out[2]


def test_records_adaptive_telemetry_for_base_reward():
    # base="reward": valid rollouts keep their style reward; invalid zeroed.
    import pytest

    traces = [
        _trace(gated=1.0, reward=1.30),
        _trace(gated=1.0, reward=1.05, stop="max_turns"),  # fatal -> invalid
    ]
    out = _adv(traces, base="reward")

    assert traces[0].metrics["adaptive_valid"] == 1.0
    assert traces[1].metrics["adaptive_valid"] == 0.0
    assert traces[0].metrics["adaptive_shaped"] == pytest.approx(1.30)
    assert traces[1].metrics["adaptive_shaped"] == pytest.approx(0.0)
    assert [t.metrics["adaptive_advantage"] for t in traces] == [pytest.approx(a) for a in out.advantages]
