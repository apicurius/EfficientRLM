"""Reward rubrics for correctness, gated efficiency, and Harness-1-style RL."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verifiers.types import State

from rlm_train.rubric import RLMTrainRubric


@dataclass(frozen=True)
class EfficiencyAxis:
    """One normalized-efficiency budget axis (fewer-is-better).

    ``state_key`` is read from rollout state; ``budget`` is the *reference
    scale* of the axis (the usage at which efficiency is half). Axis efficiency
    is ``1 / (1 + used / budget)``: it is ``1.0`` at zero usage, ``0.5`` at the
    budget, and decays smoothly toward (but never reaches) ``0`` as usage grows.
    A non-positive ``budget`` or ``weight`` disables the axis.

    The earlier shape ``max(0, 1 - used / budget)`` clamped to exactly ``0`` at
    the budget and stayed there, so the entire expensive tail collapsed to a
    single constant -- a 40-sub-call rollout and a 4525-sub-call rollout scored
    identically (the B1 defect in ``REWARD_HARNESS1_ANALYSIS.md``). The
    rational decay below keeps the axis strictly monotone in usage everywhere,
    so the runaway tail the reward exists to suppress retains gradient, while
    still satisfying the THESIS invariants: bounded in ``(0, 1]``, equal to
    ``1`` at zero usage, and strictly decreasing in usage.
    """

    state_key: str
    budget: float
    weight: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.budget > 0.0 and self.weight > 0.0

    def efficiency(self, state: State) -> float:
        if self.budget <= 0.0:
            return 0.0
        used = max(0.0, float(state.get(self.state_key) or 0.0))
        return 1.0 / (1.0 + used / self.budget)


def default_axes(
    *,
    max_iterations: int = 20,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
) -> list[EfficiencyAxis]:
    """Build the standard turns / sub-call / token efficiency axes.

    Only axes with a positive budget are enabled, so callers opt in per axis.
    The turns axis uses ``max_iterations`` as its budget by default because that
    bound always exists in the RLM harness.
    """

    return [
        EfficiencyAxis("rlm_iterations", float(max_iterations), iteration_weight),
        EfficiencyAxis("rlm_sub_llm_calls", float(subcall_budget), subcall_weight),
        EfficiencyAxis("rlm_sub_llm_tokens", float(token_budget), token_weight),
    ]


def efficiency_score(state: State, axes: list[EfficiencyAxis]) -> float:
    """Weighted-mean efficiency in ``[0, 1]`` over the enabled axes.

    Returns 0.0 when no axis is enabled, which makes the efficiency bonus a
    safe no-op (``R = c``) rather than an accidental free reward.
    """

    enabled = [a for a in axes if a.enabled]
    if not enabled:
        return 0.0
    total_weight = sum(a.weight for a in enabled)
    if total_weight <= 0.0:
        return 0.0
    score = sum(a.weight * a.efficiency(state) for a in enabled) / total_weight
    return max(0.0, min(1.0, score))


class EfficiencyGatedRubric(RLMTrainRubric):
    """Correctness-gated efficiency-shaping rubric.

    A strict superset of :class:`RLMTrainRubric`. With ``shaping_coef == 0.0``
    it reproduces the upstream correctness-only reward exactly; with
    ``shaping_coef > 0.0`` it adds a bounded efficiency bonus to correct,
    gate-passing rollouts only.
    """

    def __init__(
        self,
        correctness: Callable[..., float] | None = None,
        weight: float = 1.0,
        *,
        shaping_coef: float = 0.0,
        correct_threshold: float = 1.0,
        axes: list[EfficiencyAxis] | None = None,
        max_iterations: int = 20,
        subcall_budget: float = 0.0,
        token_budget: float = 0.0,
        iteration_weight: float = 1.0,
        subcall_weight: float = 1.0,
        token_weight: float = 1.0,
        **kwargs: Any,
    ):
        if shaping_coef < 0.0:
            raise ValueError("shaping_coef must be >= 0.0")
        self._shaping_coef = float(shaping_coef)
        self._correct_threshold = float(correct_threshold)
        self._axes = (
            axes
            if axes is not None
            else default_axes(
                max_iterations=max_iterations,
                subcall_budget=subcall_budget,
                token_budget=token_budget,
                iteration_weight=iteration_weight,
                subcall_weight=subcall_weight,
                token_weight=token_weight,
            )
        )
        super().__init__(correctness=correctness, weight=weight, **kwargs)
        if correctness is not None:
            self.add_metric(self.rlm_efficiency_score)
            self.add_metric(self._make_efficiency_bonus(correctness))

    @property
    def shaping_enabled(self) -> bool:
        return self._shaping_coef > 0.0

    def _base_value(self, state: dict, value: float) -> float:
        """Replicate the stock rubric's gated value (the upstream ``R``)."""

        if not self._gate_reward:
            return value
        if not self._passes_gates(state):
            return 0.0
        if value < self._min_reward:
            return 0.0
        return value

    def _shaped_value(self, state: dict, base: float) -> float:
        """Apply the correctness-gated efficiency bonus on top of ``base``."""

        if not self.shaping_enabled:
            return base
        if base < self._correct_threshold:
            return base
        if not self._passes_gates(state):
            return base
        e = efficiency_score(state, self._axes)
        return base * (1.0 + self._shaping_coef * e)

    def _make_main_correctness(self, correctness: Callable[..., float]) -> Callable[..., Any]:
        async def main(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            value = await self._call_correctness(correctness, kwargs)
            base = self._base_value(state, value)
            return self._shaped_value(state, base)

        main.__name__ = getattr(correctness, "__name__", "correctness")
        return main

    def _make_efficiency_bonus(self, correctness: Callable[..., float]) -> Callable[..., Any]:
        async def efficiency_bonus(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            value = await self._call_correctness(correctness, kwargs)
            base = self._base_value(state, value)
            shaped = self._shaped_value(state, base)
            return shaped - base

        efficiency_bonus.__name__ = "efficiency_bonus"
        return efficiency_bonus

    async def rlm_efficiency_score(self, state: State) -> float:
        return efficiency_score(state, self._axes)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _state_float(state: State, *keys: str, default: float = 0.0, clamp: bool = False) -> float:
    for key in keys:
        if key not in state:
            continue
        value = state.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        return _clamp01(out) if clamp else out
    return _clamp01(default) if clamp else float(default)


def _resource_axis_load(axis: EfficiencyAxis, state: State) -> float:
    """Tail-sensitive resource load for additive Harness-1 penalties.

    This is a penalty load, not an efficiency bonus, so it uses a logarithmic
    cost-side analogue of the non-saturating ``EfficiencyAxis`` decay.  The
    budget is a *reference scale*, not a hard cliff; the expensive sub-call tail
    therefore remains ordered rather than flattening to one constant.

    - iterations stay linear below the configured max (matching the prior small
      default penalty);
    - sub-calls/tokens use ``log2(1 + used / reference)`` so budget usage costs
      one unit, and 200/1000/4000 calls remain distinguishable.
    """

    used = max(0.0, float(state.get(axis.state_key) or 0.0))
    if axis.budget <= 0.0:
        return 0.0
    ratio = used / axis.budget
    if axis.state_key == "rlm_iterations":
        if ratio <= 1.0:
            return ratio
        # Root turns are normally bounded by max_iterations, but keep a gentle
        # gradient if an environment ever logs beyond the nominal limit.
        return 1.0 + math.log1p(ratio - 1.0)
    return math.log1p(ratio) / math.log(2.0)


class Harness1StyleRubric(RLMTrainRubric):
    """Additive Harness-1-style terminal reward."""

    def __init__(
        self,
        correctness: Callable[..., float] | None = None,
        weight: float = 1.0,
        *,
        outcome_weight: float = 0.7,
        trajectory_recall_weight: float = 0.3,
        recall_beta: float = 2.0,
        final_answer_recall_weight: float = 0.8,
        trajectory_fa_recall_weight: float = 0.4,
        final_answer_bonus: float = 1.0,
        fa_miss_penalty_weight: float = 0.35,
        no_final_penalty: float = -0.2,
        min_format_reward: float = 1e-3,
        turn_penalty_min_turns: int = 20,
        turn_penalty_max: float = 0.02,
        max_iterations: int = 20,
        subcall_budget: float = 0.0,
        token_budget: float = 0.0,
        iteration_weight: float = 1.0,
        subcall_weight: float = 1.0,
        token_weight: float = 1.0,
        resource_penalty_max: float = 0.02,
        trajectory_recall: Callable[..., float] | None = None,
        trajectory_answer_recall: Callable[..., float] | None = None,
        **kwargs: Any,
    ):
        self._outcome_weight = float(outcome_weight)
        self._trajectory_recall_weight = float(trajectory_recall_weight)
        self._recall_beta = float(recall_beta)
        self._final_answer_recall_weight = float(final_answer_recall_weight)
        self._trajectory_fa_recall_weight = float(trajectory_fa_recall_weight)
        self._final_answer_bonus = float(final_answer_bonus)
        self._fa_miss_penalty_weight = float(fa_miss_penalty_weight)
        self._no_final_penalty = float(no_final_penalty)
        self._min_format_reward = float(min_format_reward)
        self._turn_penalty_min_turns = int(turn_penalty_min_turns)
        self._turn_penalty_max = max(0.0, float(turn_penalty_max))
        self._max_iterations_for_penalty = max(1, int(max_iterations))
        self._resource_penalty_max = max(0.0, float(resource_penalty_max))
        self._trajectory_recall_fn = trajectory_recall
        self._trajectory_answer_recall_fn = trajectory_answer_recall
        self._resource_axes = default_axes(
            max_iterations=max_iterations,
            subcall_budget=subcall_budget,
            token_budget=token_budget,
            iteration_weight=iteration_weight,
            subcall_weight=subcall_weight,
            token_weight=token_weight,
        )
        super().__init__(correctness=correctness, weight=weight, **kwargs)
        if correctness is not None:
            self.add_metric(self.rlm_harness_turn_penalty)
            self.add_metric(self.rlm_harness_resource_penalty)
            self.add_metric(self.rlm_harness_no_final_short_circuit)
            self.add_metric(self.rlm_harness_format_floor)
            if self._trajectory_recall_fn is not None or self._trajectory_answer_recall_fn is not None:
                self.add_metric(self.rlm_harness_trajectory_recall)
                self.add_metric(self.rlm_harness_trajectory_answer_recall)

    def _base_value(self, state: dict, value: float) -> float:
        """Replicate the stock rubric's gated correctness value."""

        if not self._gate_reward:
            return value
        if not self._passes_gates(state):
            return 0.0
        if value < self._min_reward:
            return 0.0
        return value

    def _has_final_answer(self, state: State) -> bool:
        if state.get("rlm_final_answer"):
            return True
        return bool(_state_float(state, "rlm_has_final_answer", default=0.0) > 0.0)

    def _f_beta_or_base(self, state: State, answer_recall: float) -> float:
        """Use set F_beta when env telemetry exists; else use final correctness.

        The names intentionally cover both generic RLM and retrieval-style
        telemetry so a future environment can expose richer Harness-1 state
        without changing this rubric.
        """

        has_precision = any(
            key in state
            for key in (
                "rlm_curated_precision",
                "rlm_precision",
                "curated_precision",
                "precision",
            )
        )
        has_recall = any(
            key in state
            for key in (
                "rlm_curated_recall",
                "rlm_recall",
                "curated_recall",
                "recall",
            )
        )
        if not (has_precision and has_recall):
            return answer_recall
        precision = _state_float(
            state,
            "rlm_curated_precision",
            "rlm_precision",
            "curated_precision",
            "precision",
            default=0.0,
            clamp=True,
        )
        recall = _state_float(
            state,
            "rlm_curated_recall",
            "rlm_recall",
            "curated_recall",
            "recall",
            default=0.0,
            clamp=True,
        )
        if precision <= 0.0 and recall <= 0.0:
            return 0.0
        beta2 = max(1e-12, self._recall_beta * self._recall_beta)
        denom = beta2 * precision + recall
        if denom <= 0.0:
            return 0.0
        return _clamp01((1.0 + beta2) * precision * recall / denom)

    def _turn_penalty(self, state: State) -> float:
        turns = int(state.get("rlm_iterations") or 0)
        if turns <= self._turn_penalty_min_turns:
            return 0.0
        denom = max(1, self._max_iterations_for_penalty - self._turn_penalty_min_turns)
        frac = min(1.0, (turns - self._turn_penalty_min_turns) / denom)
        return self._turn_penalty_max * frac

    def _resource_penalty(self, state: State) -> float:
        """Small subturn analogue of Harness-1's turn penalty.

        In the search harness, each retrieval/inspect/curate action is a turn,
        so a turn penalty sees most cost.  In RLM, one root turn can hide
        hundreds of sub-LLM calls inside the REPL.  This capped penalty applies
        the same idea to normalized turns / sub-calls / sub-tokens without
        making cost a primary objective.

        Like the corrected gated-efficiency axis, this deliberately does *not*
        clamp each axis at its budget.  ``subcall_budget`` and ``token_budget``
        are reference scales for a logarithmic load, so the runaway tail remains
        ordered (e.g. 1000 sub-calls pays more than 200, which pays more than
        40).  The constructor argument is kept as ``resource_penalty_max`` for
        config compatibility, but operationally it is the small penalty scale
        per unit weighted load, not a hard cap.
        """

        enabled = [axis for axis in self._resource_axes if axis.enabled]
        if not enabled or self._resource_penalty_max <= 0.0:
            return 0.0
        total_weight = sum(axis.weight for axis in enabled)
        if total_weight <= 0.0:
            return 0.0
        usage_load = sum(axis.weight * _resource_axis_load(axis, state) for axis in enabled)
        return self._resource_penalty_max * (usage_load / total_weight)

    def _components(
        self,
        state: State,
        base: float,
        *,
        trajectory_recall_override: float | None = None,
        trajectory_answer_recall_override: float | None = None,
    ) -> dict[str, float]:
        answer_recall = _clamp01(
            _state_float(
                state,
                "rlm_final_answer_recall",
                "final_answer_recall",
                "answer_recall",
                default=base,
            )
        )
        outcome = self._f_beta_or_base(state, answer_recall)
        if trajectory_recall_override is not None:
            trajectory_recall = _clamp01(trajectory_recall_override)
        else:
            trajectory_recall = _clamp01(
                _state_float(
                    state,
                    "rlm_trajectory_recall",
                    "trajectory_recall",
                    "rlm_best_partial_reward",
                    default=answer_recall,
                )
            )
        if trajectory_answer_recall_override is not None:
            trajectory_answer_recall = _clamp01(trajectory_answer_recall_override)
        else:
            trajectory_answer_recall = _clamp01(
                _state_float(
                    state,
                    "rlm_trajectory_answer_recall",
                    "trajectory_answer_recall",
                    "rlm_trajectory_fa_recall",
                    default=max(trajectory_recall, answer_recall),
                )
            )
        answer_found_bonus = self._final_answer_bonus if answer_recall > 0.0 else 0.0
        miss_penalty = self._fa_miss_penalty_weight * max(
            0.0, trajectory_answer_recall - answer_recall
        )
        turn_penalty = self._turn_penalty(state)
        resource_penalty = self._resource_penalty(state)
        return {
            "outcome": outcome,
            "trajectory_recall": trajectory_recall,
            "answer_recall": answer_recall,
            "trajectory_answer_recall": trajectory_answer_recall,
            "answer_found_bonus": answer_found_bonus,
            "miss_penalty": miss_penalty,
            "turn_penalty": turn_penalty,
            "resource_penalty": resource_penalty,
        }

    def _harness_value(
        self,
        state: State,
        base: float,
        *,
        trajectory_recall_override: float | None = None,
        trajectory_answer_recall_override: float | None = None,
    ) -> float:
        if not self._has_final_answer(state):
            return self._no_final_penalty
        c = self._components(
            state,
            base,
            trajectory_recall_override=trajectory_recall_override,
            trajectory_answer_recall_override=trajectory_answer_recall_override,
        )
        reward = (
            self._outcome_weight * c["outcome"]
            + self._trajectory_recall_weight * c["trajectory_recall"]
            + self._final_answer_recall_weight * c["answer_recall"]
            + self._trajectory_fa_recall_weight * c["trajectory_answer_recall"]
            + c["answer_found_bonus"]
            - c["miss_penalty"]
            - c["turn_penalty"]
            - c["resource_penalty"]
        )
        if reward >= 0.0:
            reward = max(reward, self._min_format_reward)
        return reward

    async def _call_recall(
        self, fn: Callable[..., float] | None, kwargs: dict[str, Any]
    ) -> float | None:
        """Invoke an env-provided trajectory-recall callback, signature-safely.

        Mirrors how ``verifiers`` calls reward funcs: pass only the kwargs the
        callback declares (so an env can ask for just ``state`` or ``info``).
        """

        if fn is None:
            return None
        try:
            params = inspect.signature(fn).parameters
            accepts_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_kwargs:
                call_kwargs = dict(kwargs)
            else:
                call_kwargs = {k: v for k, v in kwargs.items() if k in params}
            result = fn(**call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return _clamp01(float(result))
        except Exception:
            # Telemetry must never crash scoring; fall back to state/default.
            return None

    def _make_main_correctness(self, correctness: Callable[..., float]) -> Callable[..., Any]:
        async def main(**kwargs: Any) -> float:
            state = kwargs.get("state") or {}
            value = await self._call_correctness(correctness, kwargs)
            base = self._base_value(state, value)
            traj = await self._call_recall(self._trajectory_recall_fn, kwargs)
            traj_ans = await self._call_recall(self._trajectory_answer_recall_fn, kwargs)
            return self._harness_value(
                state,
                base,
                trajectory_recall_override=traj,
                trajectory_answer_recall_override=traj_ans,
            )

        main.__name__ = getattr(correctness, "__name__", "correctness")
        return main

    async def rlm_harness_turn_penalty(self, state: State) -> float:
        return self._turn_penalty(state)

    async def rlm_harness_resource_penalty(self, state: State) -> float:
        return self._resource_penalty(state)

    async def rlm_harness_no_final_short_circuit(self, state: State) -> int:
        return 0 if self._has_final_answer(state) else 1

    async def rlm_harness_format_floor(self, state: State) -> float:
        return self._min_format_reward

    async def rlm_harness_trajectory_recall(self, **kwargs: Any) -> float:
        state = kwargs.get("state") or {}
        override = await self._call_recall(self._trajectory_recall_fn, kwargs)
        if override is not None:
            return override
        return _clamp01(
            _state_float(
                state,
                "rlm_trajectory_recall",
                "trajectory_recall",
                "rlm_best_partial_reward",
                default=0.0,
            )
        )

    async def rlm_harness_trajectory_answer_recall(self, **kwargs: Any) -> float:
        state = kwargs.get("state") or {}
        override = await self._call_recall(self._trajectory_answer_recall_fn, kwargs)
        if override is not None:
            return override
        return _clamp01(
            _state_float(
                state,
                "rlm_trajectory_answer_recall",
                "trajectory_answer_recall",
                "rlm_trajectory_fa_recall",
                default=0.0,
            )
        )


def make_reward_rubric(
    correctness: Callable[..., float],
    *,
    reward_style: str = "auto",
    weight: float = 1.0,
    min_iterations: int = 2,
    min_subcall: int = 0,
    max_iterations: int = 20,
    shaping_coef: float = 0.0,
    correct_threshold: float = 1.0,
    subcall_budget: float = 0.0,
    token_budget: float = 0.0,
    iteration_weight: float = 1.0,
    subcall_weight: float = 1.0,
    token_weight: float = 1.0,
    turn_penalty_min_turns: int = 20,
    turn_penalty_max: float = 0.02,
    resource_penalty_max: float = 0.02,
    trajectory_recall: Callable[..., float] | None = None,
    trajectory_answer_recall: Callable[..., float] | None = None,
) -> RLMTrainRubric:
    style = (reward_style or "auto").lower().replace("_", "-")
    if style in {"auto", "default"}:
        style = "efficiency" if shaping_coef > 0.0 else "correctness"
    if style in {"correctness", "stock", "original"}:
        return RLMTrainRubric(
            correctness=correctness,
            weight=weight,
            min_iterations=min_iterations,
            min_subcall=min_subcall,
        )
    if style in {"efficiency", "gated", "efficiency-gated"}:
        return EfficiencyGatedRubric(
            correctness=correctness,
            weight=weight,
            min_iterations=min_iterations,
            min_subcall=min_subcall,
            shaping_coef=shaping_coef,
            correct_threshold=correct_threshold,
            max_iterations=max_iterations,
            subcall_budget=subcall_budget,
            token_budget=token_budget,
            iteration_weight=iteration_weight,
            subcall_weight=subcall_weight,
            token_weight=token_weight,
        )
    if style in {"harness", "harness1", "harness-1"}:
        return Harness1StyleRubric(
            correctness=correctness,
            weight=weight,
            min_iterations=min_iterations,
            min_subcall=min_subcall,
            max_iterations=max_iterations,
            subcall_budget=subcall_budget,
            token_budget=token_budget,
            iteration_weight=iteration_weight,
            subcall_weight=subcall_weight,
            token_weight=token_weight,
            turn_penalty_min_turns=turn_penalty_min_turns,
            turn_penalty_max=turn_penalty_max,
            resource_penalty_max=resource_penalty_max,
            trajectory_recall=trajectory_recall,
            trajectory_answer_recall=trajectory_answer_recall,
        )
    raise ValueError(f"Unknown reward_style: {reward_style!r}")
