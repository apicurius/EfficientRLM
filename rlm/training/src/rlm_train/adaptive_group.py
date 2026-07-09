from __future__ import annotations

import math
from typing import Any

from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs

_FATAL_STOPS = frozenset({"max_turns", "max_turns_reached"})
_DEAD_WORKER = ("Connection lost", "worker closed stdout", "SIGKILLed")


def _metrics(trace: Any) -> dict[str, Any]:
    return dict(getattr(trace, "metrics", None) or {})


def _metric(trace: Any, key: str, default: float = 0.0) -> float:
    try:
        return float(_metrics(trace).get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _stop(trace: Any) -> str:
    return str(getattr(trace, "stop_condition", "") or "")


def _has_final(trace: Any) -> bool:
    return _metric(trace, "rlm_has_final_answer") > 0.0 or _stop(trace) == "has_final_answer"


def _correct(trace: Any) -> float:
    m = _metrics(trace)
    if "gated_reward" in m:
        return 1.0 if _metric(trace, "gated_reward") > 0.0 else 0.0
    return 1.0 if float(getattr(trace, "reward", 0.0) or 0.0) > 0.0 else 0.0


def _reward(trace: Any) -> float:
    try:
        return float(getattr(trace, "reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fatal(trace: Any) -> bool:
    err = " ".join(str(e) for e in (getattr(trace, "errors", None) or ()))
    return any(x in err for x in _DEAD_WORKER) or not _has_final(trace) or _stop(trace) in _FATAL_STOPS


# Scaffold-action cost bases. The thesis cost target is scaffold *usage* — root
# turns plus helper calls — NOT tokens (tokens are telemetry only). Each basis is a
# different definition of "scaffold spend" over the metrics the RLM env already emits
# (rlm_iterations, rlm_repl_calls, rlm_sub_llm_calls), so the cost-reduction result can
# be ablated against the choice of cost definition.
def _cost_iterations(trace: Any) -> float:
    """Root turns only — the leanest notion of scaffold spend."""
    return _metric(trace, "rlm_iterations")


def _cost_iterations_log_subcalls(trace: Any) -> float:
    """Turns + log-damped sub-LLM delegations (the original/default basis)."""
    return _metric(trace, "rlm_iterations") + math.log1p(_metric(trace, "rlm_sub_llm_calls"))


def _cost_iterations_log_helpers(trace: Any) -> float:
    """Turns + log-damped *all* helper calls (REPL executions and sub-LLM delegations)."""
    return _metric(trace, "rlm_iterations") + math.log1p(
        _metric(trace, "rlm_repl_calls") + _metric(trace, "rlm_sub_llm_calls")
    )


def _cost_iterations_subcalls(trace: Any) -> float:
    """Turns + linear (un-damped) sub-LLM delegations — probes log-damping sensitivity."""
    return _metric(trace, "rlm_iterations") + _metric(trace, "rlm_sub_llm_calls")


_COST_BASES = {
    "iterations": _cost_iterations,
    "iterations_log_subcalls": _cost_iterations_log_subcalls,
    "iterations_log_helpers": _cost_iterations_log_helpers,
    "iterations_subcalls": _cost_iterations_subcalls,
}
DEFAULT_COST_BASIS = "iterations_log_subcalls"


def _cost(trace: Any, cost_basis: str = DEFAULT_COST_BASIS) -> float:
    return _COST_BASES[cost_basis](trace)


def _beta(solve_rate: float, *, beta_max: float, solve_floor: float, gamma: float) -> float:
    if solve_rate <= solve_floor:
        return 0.0
    ramp = (solve_rate - solve_floor) / max(1e-12, 1.0 - solve_floor)
    return beta_max * (ramp**gamma)


def _record(
    trace: Any,
    *,
    solve_rate: float,
    beta: float,
    valid: bool,
    cost: float,
    normalized_cost: float,
    shaped: float,
    advantage: float,
) -> None:
    """Write the lever's intermediates into trace.metrics (dict[str, float]).

    Pure telemetry, not shaping: it rides the same metrics channel as
    gated_reward/rlm_* into train_rollouts.jsonl + wandb, so the adaptive
    advantage is observable in-run without offline recomputation.
    """
    m = getattr(trace, "metrics", None)
    if not isinstance(m, dict):
        return
    m["adaptive_group_solve_rate"] = float(solve_rate)
    m["adaptive_beta"] = float(beta)
    m["adaptive_valid"] = 1.0 if valid else 0.0
    m["adaptive_cost"] = float(cost)
    m["adaptive_normalized_cost"] = float(normalized_cost)
    m["adaptive_shaped"] = float(shaped)
    m["adaptive_advantage"] = float(advantage)


def adaptive_group_advantage(
    inputs: AdvantageInputs,
    *,
    base: str = "correctness",
    beta_max: float = 0.15,
    solve_floor: float = 0.25,
    gamma: float = 1.0,
    cost_basis: str = DEFAULT_COST_BASIS,
    min_span: float = 0.0,
) -> AdvantageOutputs:
    """Validity-gated, group-mean-centered advantage.

    base="correctness": binary-correct base with solve-rate-scaled cost re-ranking
    of valid siblings (scaffold-cost lever).
    base="reward": keep each valid rollout's own style reward (correctness /
    efficiency / harness1), zero out invalid/fatal, then center. No extra cost
    term, so the reward style is the only shaping (no double shaping).

    cost_basis selects the scaffold-cost definition for the re-ranking (one of
    _COST_BASES); all are scaffold-action costs, never tokens.
    """
    if base not in {"correctness", "reward"}:
        raise ValueError(f"Unknown base: {base!r}")
    if cost_basis not in _COST_BASES:
        raise ValueError(f"Unknown cost_basis: {cost_basis!r}; valid: {sorted(_COST_BASES)}")

    traces = list(inputs.rollouts)
    if not traces:
        return AdvantageOutputs(advantages=[])

    correct = [_correct(t) for t in traces]
    valid = [c > 0.0 and not _fatal(t) for t, c in zip(traces, correct)]
    solve_rate = sum(1.0 for v in valid if v) / len(valid)
    costs = [_cost(t, cost_basis) for t in traces]
    normalized_cost = [0.0] * len(traces)

    if base == "reward":
        beta = 0.0
        shaped = [(_reward(t) if v else 0.0) for t, v in zip(traces, valid)]
    else:
        beta = _beta(solve_rate, beta_max=beta_max, solve_floor=solve_floor, gamma=gamma)
        shaped = [c if v else 0.0 for c, v in zip(correct, valid)]
        if beta > 0.0:
            valid_idx = [i for i, v in enumerate(valid) if v]
            if len(valid_idx) >= 2:
                valid_costs = [costs[i] for i in valid_idx]
                lo, hi = min(valid_costs), max(valid_costs)
                span = hi - lo
                if span > 0.0 and span >= min_span:
                    for i in valid_idx:
                        normalized_cost[i] = (costs[i] - lo) / span
                        shaped[i] = 1.0 - beta * normalized_cost[i]

    baseline = sum(shaped) / len(shaped)
    advantages = [x - baseline for x in shaped]

    for t, v, c, nc, sh, adv in zip(traces, valid, costs, normalized_cost, shaped, advantages):
        _record(t, solve_rate=solve_rate, beta=beta, valid=v, cost=c, normalized_cost=nc, shaped=sh, advantage=adv)

    return AdvantageOutputs(advantages=advantages)


__all__ = ["adaptive_group_advantage"]
