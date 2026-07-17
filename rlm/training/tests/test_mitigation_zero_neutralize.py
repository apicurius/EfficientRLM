"""Invariants for the registered mitigation objective (Addenda 3-4 of
docs/PHASE0_REVIEW_MEMO_20260717.md): the ``iterations_ln_excess`` cost basis
and the ``zero_neutralize`` transform.

The operator-exact zero-neutralization math (denominator ``n - m`` over the full
group size) is the reference validated bit-exact (|dA|<1e-9 on 1688 members) by
scripts/12_mitigation_rederive.py; these tests pin the same guarantees on the
LIVE operator across parameter sweeps.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.advantage import AdvantageInputs
from rlm_train.adaptive_group import _cost_iterations_ln_excess, adaptive_group_advantage


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


def _adv(factory, **kwargs):
    # Rebuild traces per call: adaptive_group_advantage mutates trace.metrics
    # (telemetry), so a fresh group keeps each measurement independent.
    return adaptive_group_advantage(AdvantageInputs(rollouts=factory()), **kwargs).advantages


def _centered(values):
    mean = sum(values) / len(values)
    return [v - mean for v in values]


NEW = dict(base="correctness", cost_basis="iterations_ln_excess", zero_neutralize=True)


# --------------------------------------------------------------------------
# (a) ZERO-NEUTRALITY: every valid abstaining (S==0) member's advantage equals
#     its beta=0 (gated-correctness) advantage to 1e-9, across a parameter sweep.
# --------------------------------------------------------------------------
def _compositions():
    """(factory, name) pairs covering k_valid in {2,3}, mixed zero/delegating,
    and the all-valid-zero corner. Each group is size 4 (matches the run)."""
    return [
        # k_valid=2: one abstainer, one heavy delegator, two invalid.
        (lambda: [
            _trace(gated=1.0, iters=4.0, subcalls=0.0),
            _trace(gated=1.0, iters=12.0, subcalls=25.0),
            _trace(gated=0.0, has_final=False, stop="max_turns"),
            _trace(gated=0.0, has_final=False, stop="max_turns"),
        ], "k2_zero_vs_deleg"),
        # k_valid=3: one abstainer, two delegators (different excess).
        (lambda: [
            _trace(gated=1.0, iters=3.0, subcalls=0.0),
            _trace(gated=1.0, iters=9.0, subcalls=12.0),
            _trace(gated=1.0, iters=15.0, subcalls=40.0),
            _trace(gated=0.0, has_final=False, stop="max_turns"),
        ], "k3_one_zero_two_deleg"),
        # k_valid=3: two abstainers, one delegator.
        (lambda: [
            _trace(gated=1.0, iters=3.0, subcalls=0.0),
            _trace(gated=1.0, iters=6.0, subcalls=0.0),
            _trace(gated=1.0, iters=14.0, subcalls=30.0),
            _trace(gated=0.0, has_final=False, stop="max_turns"),
        ], "k3_two_zero_one_deleg"),
        # all-valid-zero (with iteration variance): no delegating sibling -> no shaping.
        (lambda: [
            _trace(gated=1.0, iters=3.0, subcalls=0.0),
            _trace(gated=1.0, iters=11.0, subcalls=0.0),
            _trace(gated=1.0, iters=7.0, subcalls=0.0),
            _trace(gated=0.0, has_final=False, stop="max_turns"),
        ], "all_valid_zero"),
    ]


def test_zero_members_are_advantage_neutral_across_sweep():
    for factory, name in _compositions():
        ref = _adv(factory, base="correctness", beta_max=0.0)  # gated correctness, centered
        subs = [t.metrics["rlm_sub_llm_calls"] for t in factory()]
        gated = [t.metrics["gated_reward"] for t in factory()]
        fatal = [t.stop_condition == "max_turns" for t in factory()]
        for beta in (0.05, 0.15, 0.30):
            for lam in (1.0, 2.0, 3.713):
                for B in (3, 5, 8):
                    out = _adv(factory, beta_max=beta, lam=lam, B=B, solve_floor=0.25, **NEW)
                    for i, (s, g, f) in enumerate(zip(subs, gated, fatal)):
                        valid = g > 0.0 and not f
                        if valid and s == 0.0:
                            assert abs(out[i] - ref[i]) < 1e-9, (
                                f"{name} beta={beta} lam={lam} B={B} member {i}: "
                                f"abstainer dA {out[i]} != beta0 {ref[i]}"
                            )


def test_zero_neutralization_actually_fires_and_reranks_delegators():
    # Sanity: the transform is not vacuous — with a real cost gap the delegator
    # is pushed BELOW the abstainer while the abstainer stays at its beta=0 value.
    factory = lambda: [
        _trace(gated=1.0, iters=4.0, subcalls=0.0),   # abstainer
        _trace(gated=1.0, iters=12.0, subcalls=25.0),  # heavy delegator
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    ref = _adv(factory, base="correctness", beta_max=0.0)
    out = _adv(factory, beta_max=0.15, lam=2.0, B=5, **NEW)
    assert abs(out[0] - ref[0]) < 1e-9          # abstainer neutral
    assert out[0] > out[1]                       # delegator ranked below abstainer
    assert out[1] < ref[1]                       # delegator genuinely penalized (fired)


# --------------------------------------------------------------------------
# (b) BUDGET-FREENESS: sub-calls within [0, B] are unpriced.
# --------------------------------------------------------------------------
def test_ln_excess_cost_is_flat_within_budget():
    base = _cost_iterations_ln_excess(_trace(gated=1.0, iters=7.0, subcalls=0.0), lam=2.0, B=5)
    for s in range(0, 6):  # S in {0..5}, B=5 -> excess 0
        c = _cost_iterations_ln_excess(_trace(gated=1.0, iters=7.0, subcalls=float(s)), lam=2.0, B=5)
        assert c == base == 7.0
    # first call above budget IS priced.
    above = _cost_iterations_ln_excess(_trace(gated=1.0, iters=7.0, subcalls=6.0), lam=2.0, B=5)
    assert above == pytest.approx(7.0 + 2.0 * math.log1p(1.0))


def test_within_budget_siblings_get_identical_advantage():
    # Two valid members differ ONLY in S within budget; a third heavy delegator
    # makes the group fire. The two in-budget members must be tied.
    factory = lambda: [
        _trace(gated=1.0, iters=6.0, subcalls=1.0),   # in-budget
        _trace(gated=1.0, iters=6.0, subcalls=4.0),   # in-budget (same cost)
        _trace(gated=1.0, iters=6.0, subcalls=30.0),  # excess -> priced, fires group
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    out = _adv(factory, beta_max=0.15, lam=2.0, B=5, **NEW)
    assert out[0] == out[1]        # identical cost -> identical advantage
    assert out[0] > out[2]         # both outrank the excess delegator


# --------------------------------------------------------------------------
# (c) CORRECTNESS DOMINANCE preserved under the new basis at beta=0.15.
# --------------------------------------------------------------------------
def test_every_valid_correct_outranks_every_invalid():
    factory = lambda: [
        _trace(gated=1.0, iters=3.0, subcalls=0.0),    # valid, abstainer
        _trace(gated=1.0, iters=10.0, subcalls=15.0),  # valid, delegator
        _trace(gated=1.0, iters=18.0, subcalls=60.0),  # valid, heavy delegator
        _trace(gated=0.0, has_final=False, stop="max_turns"),   # invalid
        _trace(gated=1.0, stop="max_turns"),                    # correct-looking but fatal -> invalid
    ]
    out = _adv(factory, beta_max=0.15, lam=2.0, B=5, solve_floor=0.25, **NEW)
    valid_advs = [out[0], out[1], out[2]]
    invalid_advs = [out[3], out[4]]
    assert min(valid_advs) > max(invalid_advs)


# --------------------------------------------------------------------------
# (d) REDUCTION: beta=0 reproduces gated correctness exactly (transform inert).
# --------------------------------------------------------------------------
def test_beta_zero_reduces_to_gated_correctness():
    factory = lambda: [
        _trace(gated=1.0, iters=2.0, subcalls=0.0),
        _trace(gated=1.0, iters=19.0, subcalls=50.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    out = _adv(factory, beta_max=0.0, lam=2.0, B=5, **NEW)
    assert out == _centered([1.0, 1.0, 0.0, 0.0])


def test_below_solve_floor_no_shaping_under_new_basis():
    # solve_rate at/below floor -> beta=0 -> gated correctness regardless of basis.
    factory = lambda: [
        _trace(gated=1.0, iters=3.0, subcalls=0.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    out = _adv(factory, beta_max=0.15, lam=2.0, B=5, solve_floor=0.25, **NEW)
    assert out == _centered([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Basis registration + parameter threading.
# --------------------------------------------------------------------------
def test_new_basis_is_registered_and_unknown_still_raises():
    from rlm_train.adaptive_group import _COST_BASES

    assert "iterations_ln_excess" in _COST_BASES
    with pytest.raises(ValueError):
        _adv(lambda: [_trace(gated=1.0), _trace(gated=1.0)], cost_basis="bogus_basis")


def test_lam_and_B_thread_from_kwargs_into_cost_telemetry():
    # adaptive_cost telemetry must reflect I + lam*ln(1+(S-B)+) for delegators.
    factory = lambda: [
        _trace(gated=1.0, iters=4.0, subcalls=0.0),
        _trace(gated=1.0, iters=4.0, subcalls=25.0),
        _trace(gated=0.0, has_final=False, stop="max_turns"),
    ]
    traces = factory()
    adaptive_group_advantage(AdvantageInputs(rollouts=traces), beta_max=0.15, lam=2.0, B=5, **NEW)
    assert traces[0].metrics["adaptive_cost"] == pytest.approx(4.0)  # abstainer: excess 0
    assert traces[1].metrics["adaptive_cost"] == pytest.approx(4.0 + 2.0 * math.log1p(20.0))
