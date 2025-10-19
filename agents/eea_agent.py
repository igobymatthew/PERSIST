"""Scaffolding for the Emotional Equilibrium Architecture (EEA).

This module translates the conceptual layers described in ``docs/EEA.md``
into a light-weight, inspectable agent scaffold. The goal is not to model
neuroscience, but to provide a well-documented software artifact that mirrors
the original architectural intent and can be expanded with domain specific
logic later on.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` between ``lower`` and ``upper`` (inclusive)."""

    return max(lower, min(upper, value))


def _softsign(x: float) -> float:
    """Smoothly squash values to ``[-1, 1]`` while preserving sign."""

    return x / (1.0 + abs(x))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Guard against division by zero when computing derived features."""

    return num / den if den != 0 else default


def _regularize_toward_bounds(
    value: float, bounds: Tuple[float, float], gain: float = 0.25
) -> float:
    """Move ``value`` toward ``bounds`` without overshooting."""

    lower, upper = bounds
    if value < lower:
        return value + gain * (lower - value)
    if value > upper:
        return value - gain * (value - upper)
    midpoint = (lower + upper) * 0.5
    return value + gain * 0.1 * (midpoint - value)


def _entropy_tolerance_from_ratio(bounds: Tuple[float, float]) -> float:
    """Derive an entropy tolerance from the width of the ratio band."""

    lower, upper = bounds
    width = max(1e-3, upper - lower)
    return float(np.clip(0.5 * width + 0.05, 0.05, 0.35))


@dataclass
class EmotionState:
    """Container for the hedonic (H) and fear (F) signals.

    The state stores the raw affective signals alongside their latest
    modulated counterparts so that downstream layers can reason about both
    forms if needed.
    """

    happiness: float
    fear: float
    modulated_happiness: float | None = None
    modulated_fear: float | None = None

    def effective_happiness(self) -> float:
        """Return the happiness signal after modulation if available."""

        return (
            self.modulated_happiness
            if self.modulated_happiness is not None
            else self.happiness
        )

    def effective_fear(self) -> float:
        """Return the fear signal after modulation if available."""

        return self.modulated_fear if self.modulated_fear is not None else self.fear

    def ratio(self) -> float:
        """Return the normalized hedonic ratio H/(H+F).

        If both signals are zero the ratio defaults to 0.5, indicating a
        neutral equilibrium. The same default is applied when the ratio would
        otherwise be undefined.
        """

        happiness = self.effective_happiness()
        fear = self.effective_fear()
        total = happiness + fear
        if total <= 0:
            return 0.5
        return happiness / total


class CorePrincipleLayer:
    """Foundation layer that validates and interprets affective amplitudes."""

    def interpret(self, state: EmotionState) -> float:
        """Interpret the ratio between happiness and fear.

        Returns the normalized ratio which acts as the key control signal for
        the upper layers. The value is clamped to ``[0.0, 1.0]`` to avoid
        unstable propagation.
        """

        ratio = state.ratio()
        return _clamp(ratio)


@dataclass
class ValenceRegulator:
    """Maintains a stable ratio between positive and negative affect."""

    happiness_weight: float = 0.5
    fear_weight: float = 0.5
    positivity_offset: float = 0.08
    negativity_bias: float = 1.6

    def regulate(self, state: EmotionState) -> EmotionState:
        happiness = state.happiness * self.happiness_weight
        fear = state.fear * self.fear_weight

        # Positivity offset when both affective signals are near zero
        if happiness < 0.15 and fear < 0.15:
            happiness += self.positivity_offset

        # Negativity bias (amplify fear weight)
        fear *= self.negativity_bias

        happiness = _clamp(happiness)
        fear = _clamp(fear)
        state.modulated_happiness = happiness
        state.modulated_fear = fear
        return state


@dataclass
class ContrastNormalizer:
    """Ensures emotional contrast remains perceivable."""

    contrast_floor: float = 1e-3

    def normalize(self, state: EmotionState) -> EmotionState:
        happiness = state.effective_happiness()
        fear = state.effective_fear()
        delta = abs(happiness - fear)
        if delta < self.contrast_floor:
            adjustment = self.contrast_floor - delta
            # push signals apart symmetrically to preserve mean intensity
            happiness += adjustment / 2
            fear = _clamp(fear - adjustment / 2)
        state.modulated_happiness = _clamp(happiness)
        state.modulated_fear = _clamp(fear)
        return state


@dataclass
class EntropyBuffer:
    """Prevents emotional monotony or overload by referencing history."""

    tolerance: float = 0.15
    history: List[EmotionState] = field(default_factory=list)
    max_history: int = 50

    def dampen(self, state: EmotionState) -> EmotionState:
        if not self.history:
            self.history.append(EmotionState(state.happiness, state.fear))
            return state

        last = self.history[-1]
        delta_h = abs(last.happiness - state.happiness)
        delta_f = abs(last.fear - state.fear)
        if delta_h < self.tolerance:
            state.modulated_happiness = _clamp(state.effective_happiness() * 0.95)
        if delta_f < self.tolerance:
            state.modulated_fear = _clamp(state.effective_fear() * 1.05)

        self.history.append(EmotionState(state.happiness, state.fear))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return state

    def reset(self) -> None:
        """Clear stored history to remove lingering momentum between sessions."""

        self.history.clear()

    def set_tolerance(self, tolerance: float) -> None:
        """Update the entropy tolerance used when comparing consecutive states."""

        self.tolerance = max(0.0, float(tolerance))

    def seed(self, states: Sequence[EmotionState]) -> None:
        """Seed the history with pre-existing experiences."""

        truncated = list(states)[-self.max_history :]
        self.history = [EmotionState(s.happiness, s.fear) for s in truncated]


@dataclass
class ModulationLayer:
    """Aggregates the modulation components described in the specification."""

    target_range: Tuple[float, float] = (0.7, 0.8)
    valence_regulator: ValenceRegulator = field(default_factory=ValenceRegulator)
    contrast_normalizer: ContrastNormalizer = field(default_factory=ContrastNormalizer)
    entropy_buffer: EntropyBuffer = field(default_factory=EntropyBuffer)
    soft_cap: float = 0.98

    def __post_init__(self) -> None:
        lo, hi = self.target_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("target_range must lie within [0, 1] and be ordered")

    def set_target_range(self, target_range: Tuple[float, float]) -> None:
        lo, hi = target_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("target_range must lie within [0, 1] and be ordered")
        self.target_range = (lo, hi)

    def modulate(self, state: EmotionState) -> EmotionState:
        state = self.valence_regulator.regulate(state)
        state = self.contrast_normalizer.normalize(state)
        state = self.entropy_buffer.dampen(state)

        ratio = state.ratio()
        lo, hi = self.target_range
        if ratio < lo:
            boost = state.effective_happiness() * (1 + (lo - ratio))
            state.modulated_happiness = _clamp(min(self.soft_cap, boost))
        elif ratio > hi:
            boost = state.effective_fear() * (1 + (ratio - hi))
            state.modulated_fear = _clamp(min(self.soft_cap, boost))
        return state

    def reset(self) -> None:
        """Reset modulation stateful components such as the entropy history."""

        self.entropy_buffer.reset()


@dataclass
class ProcessingLayer:
    """Models adaptive feedback loops between fear and happiness."""

    anticipation_gain: float = 0.10
    reinforcement_gain: float = 0.05
    calibration_gain: float = 0.02

    def process(self, state: EmotionState) -> EmotionState:
        happiness = state.effective_happiness()
        fear = state.effective_fear()

        # Feedback A – anticipation: fear sharpens reward acquisition
        happiness += fear * self.anticipation_gain
        # Feedback B – reinforcement: successful avoidance reduces fear
        fear *= 1.0 - self.reinforcement_gain

        # Feedback C – calibration: push toward stable ratio
        ratio = state.ratio()
        if ratio < 0.5:
            happiness += self.calibration_gain
        else:
            fear += self.calibration_gain

        state.modulated_happiness = _clamp(happiness)
        state.modulated_fear = _clamp(fear)
        return state


@dataclass
class ContextWeights:
    environmental: float = 1.0
    social: float = 1.0
    internal: float = 1.0

    def normalize(self) -> Tuple[float, float, float]:
        total = max(1e-6, self.environmental + self.social + self.internal)
        return (
            self.environmental / total,
            self.social / total,
            self.internal / total,
        )


class IntegrationLayer:
    """Contextualizes signals using environment, social, and internal weights."""

    def integrate(
        self, state: EmotionState, context_weights: ContextWeights
    ) -> EmotionState:
        weights = context_weights.normalize()
        happiness = state.effective_happiness()
        fear = state.effective_fear()
        happiness *= 1 + weights[0] * 0.10 + weights[1] * 0.05
        fear *= 1 + weights[2] * 0.10
        state.modulated_happiness = _clamp(happiness)
        state.modulated_fear = _clamp(fear)
        return state


class BehaviorState(Enum):
    VITAL_ENGAGEMENT = "vital_engagement"
    APATHY = "apathy"
    ANXIETY = "anxiety"
    MANIA = "mania"


@dataclass
class OutputLayer:
    """Maps the final ratio to a qualitative behavioral manifestation."""

    vital_range: Tuple[float, float] = (0.7, 0.8)
    apathy_threshold: float = 0.2
    mania_threshold: float = 0.8
    high_fear_threshold: float = 0.7

    def __post_init__(self) -> None:
        lo, hi = self.vital_range
        if not (0.0 <= self.apathy_threshold <= lo <= hi <= 1.0):
            raise ValueError(
                "Behavior thresholds must satisfy apathy <= vital_range <= 1.0"
            )
        if not (hi <= self.mania_threshold <= 1.0):
            raise ValueError("Mania threshold must be >= vital_range[1] and <= 1.0")

    def classify(self, state: EmotionState) -> BehaviorState:
        ratio = state.ratio()
        lo, hi = self.vital_range
        if lo <= ratio <= hi:
            return BehaviorState.VITAL_ENGAGEMENT
        if ratio < self.apathy_threshold:
            return BehaviorState.APATHY
        if ratio < lo:
            return BehaviorState.ANXIETY
        if ratio >= self.mania_threshold:
            return BehaviorState.MANIA
        # Between ``hi`` and ``mania_threshold`` we consider the system heightened yet stable.
        return BehaviorState.VITAL_ENGAGEMENT


@dataclass
class MetaLayer:
    """Captures long-term adaptation as a moving prior on the equilibrium."""

    smoothing: float = 0.05
    equilibrium_prior: float = 0.75

    def update(self, observed_ratio: float) -> float:
        self.equilibrium_prior = (
            1 - self.smoothing
        ) * self.equilibrium_prior + self.smoothing * observed_ratio
        return self.equilibrium_prior

    def reset(self, *, equilibrium_prior: Optional[float] = None) -> None:
        """Re-initialize the prior, optionally overriding the default value."""

        if equilibrium_prior is None:
            equilibrium_prior = 0.75
        self.equilibrium_prior = equilibrium_prior


@dataclass
class GovernanceLayer:
    """Defines the evaluative function for meaning derived from H and F."""

    fear_bounds: Tuple[float, float] = (0.2, 0.4)
    context_weight: float = 0.1
    vitality_fn: Optional[Callable[[EmotionState], float]] = None
    vitality_mode: str = "ratio"

    def __post_init__(self) -> None:
        lo, hi = self.fear_bounds
        if not (0.0 <= lo <= hi):
            raise ValueError("fear_bounds must be ordered and non-negative")

    def meaning(
        self,
        state: EmotionState,
        context: Mapping[str, float] | None = None,
    ) -> float:
        ratio = state.ratio()
        fear = state.effective_fear()
        happiness = max(1e-6, state.effective_happiness())
        context_modifier = 0.0
        if context:
            context_modifier = sum(context.values()) / max(1, len(context))
        fear_ratio = fear / happiness
        lo, hi = self.fear_bounds
        fear_term = 1.0 if lo <= fear_ratio <= hi else 0.5
        if self.vitality_fn is not None:
            vitality = self.vitality_fn(state)
        else:
            if self.vitality_mode == "difference":
                vitality = _clamp(happiness - fear + 0.5)
            elif self.vitality_mode == "softsign":
                vitality = 0.5 + 0.5 * _softsign(happiness - fear)
            elif self.vitality_mode == "geomean":
                vitality = (happiness * (1.0 - fear)) ** 0.5
            else:
                vitality = ratio
        vitality *= fear_term
        return vitality + self.context_weight * context_modifier


class EmotionalEquilibriumAgent:
    """Agent scaffold implementing the Emotional Equilibrium Architecture."""

    def __init__(
        self,
        *,
        stage_provider: Optional[StageProvider] = None,
        telemetry_hook: Optional[Callable[[Mapping[str, object]], None]] = None,
    ) -> None:
        self.core = CorePrincipleLayer()
        self.modulation = ModulationLayer()
        self.processing = ProcessingLayer()
        self.integration = IntegrationLayer()
        self.output = OutputLayer()
        self.meta = MetaLayer()
        self.governance = GovernanceLayer()
        self._stage_provider: Optional[StageProvider] = stage_provider
        self._telemetry_hook = telemetry_hook
        self._logger = logging.getLogger(self.__class__.__name__)
        self._stage_memory: Dict[str, Deque[EmotionState]] = {}
        self._stage_memory_capacity = 64
        self._active_stage: Optional[StageContext] = None

    def configure_stage_awareness(
        self,
        *,
        stage_provider: Optional[StageProvider] = None,
        telemetry_hook: Optional[Callable[[Mapping[str, object]], None]] = None,
    ) -> None:
        """Attach runtime providers for life-stage awareness and telemetry."""

        if stage_provider is not None:
            self._stage_provider = stage_provider
        if telemetry_hook is not None:
            self._telemetry_hook = telemetry_hook

    def _resolve_stage_context(self) -> Optional[StageContext]:
        if self._stage_provider is None:
            return None

        raw_context = self._stage_provider()
        if raw_context is None:
            return None

        if hasattr(raw_context, "as_dict") and callable(raw_context.as_dict):
            payload: Mapping[str, Any] = raw_context.as_dict()
        elif isinstance(raw_context, Mapping):
            payload = raw_context
        else:
            payload = {
                "index": getattr(raw_context, "index", None),
                "name": getattr(raw_context, "name", None),
                "affect_targets": getattr(raw_context, "affect_targets", {}),
            }

        raw_targets = payload.get("affect_targets") or {}
        targets: Dict[str, Tuple[float, float]] = {}
        if isinstance(raw_targets, Mapping):
            for key, bounds in raw_targets.items():
                try:
                    lower, upper = bounds  # type: ignore[misc]
                except (TypeError, ValueError):
                    continue
                try:
                    targets[str(key)] = (float(lower), float(upper))
                except (TypeError, ValueError):
                    continue

        raw_name = payload.get("name")
        name = str(raw_name) if raw_name is not None else "stage"
        raw_index = payload.get("index")
        try:
            index = int(raw_index) if raw_index is not None else -1
        except (TypeError, ValueError):
            index = -1

        return StageContext(name=name, index=index, affect_targets=targets)

    def _handle_stage_transition(self, context: StageContext) -> None:
        ratio_bounds = context.affect_targets.get("ratio")
        if ratio_bounds:
            try:
                self.modulation.set_target_range(ratio_bounds)
            except ValueError:
                self._logger.debug(
                    "Ignoring invalid ratio bounds %s for stage %s",
                    ratio_bounds,
                    context.name,
                )
            self.modulation.entropy_buffer.set_tolerance(
                _entropy_tolerance_from_ratio(ratio_bounds)
            )

        if context.name not in self._stage_memory:
            self._stage_memory[context.name] = deque(maxlen=self._stage_memory_capacity)

        experiences = list(self._stage_memory[context.name])
        if ratio_bounds and experiences:
            midpoint = sum(ratio_bounds) * 0.5
            experiences.sort(key=lambda s: abs(s.ratio() - midpoint))
            seed_states = experiences[: self.modulation.entropy_buffer.max_history]
            self.modulation.entropy_buffer.seed(seed_states)
            self.meta.reset(equilibrium_prior=midpoint)
        else:
            if ratio_bounds:
                self.meta.reset(equilibrium_prior=sum(ratio_bounds) * 0.5)
            else:
                self.meta.reset()
            self.modulation.entropy_buffer.reset()

        self._active_stage = context

    def _enforce_ratio_bounds(
        self, state: EmotionState, ratio_bounds: Tuple[float, float]
    ) -> EmotionState:
        ratio = state.ratio()
        lower, upper = ratio_bounds
        if lower <= ratio <= upper:
            return state

        total = state.effective_happiness() + state.effective_fear()
        if total <= 0:
            return state

        target_ratio = lower if ratio < lower else upper
        desired_h = _clamp(target_ratio * total)
        desired_f = _clamp(max(0.0, total - desired_h))
        state.modulated_happiness = desired_h
        state.modulated_fear = desired_f
        return state

    def _record_stage_experience(
        self,
        stage_name: str,
        state: EmotionState,
        context_weights: ContextWeights,
    ) -> None:
        if stage_name not in self._stage_memory:
            self._stage_memory[stage_name] = deque(maxlen=self._stage_memory_capacity)

        env_w, social_w, internal_w = context_weights.normalize()
        hedonic_gain = 1.0 + 0.2 * env_w + 0.1 * social_w
        fear_gain = 1.0 + 0.15 * internal_w
        snapshot = EmotionState(
            _clamp(state.effective_happiness() * hedonic_gain),
            _clamp(state.effective_fear() * fear_gain),
        )
        self._stage_memory[stage_name].append(snapshot)

    def _emit_stage_metrics(self, context: StageContext, state: EmotionState) -> None:
        if self._telemetry_hook is None:
            return

        affect_state = {
            "happiness": state.effective_happiness(),
            "fear": state.effective_fear(),
            "ratio": state.ratio(),
        }
        ratio_bounds = context.affect_targets.get("ratio")
        if ratio_bounds:
            lower, upper = ratio_bounds
            ratio = affect_state["ratio"]
            deviation = 0.0
            if ratio < lower:
                deviation = lower - ratio
            elif ratio > upper:
                deviation = ratio - upper
            affect_state["ratio_deviation"] = deviation

        payload = {
            "name": context.name,
            "index": context.index,
            "affect_targets": context.affect_targets,
            "affect_state": affect_state,
        }

        try:
            self._telemetry_hook(payload)
        except (
            Exception
        ) as exc:  # pragma: no cover - telemetry failures shouldn't crash
            self._logger.debug("Telemetry hook error: %s", exc)

    def evaluate(
        self,
        happiness: float,
        fear: float,
        *,
        context: Optional[Mapping[str, float]] = None,
        context_weights: Optional[ContextWeights] = None,
    ) -> Dict[str, object]:
        """Run the full architecture on the given affective signals.

        Parameters
        ----------
        happiness:
            Raw System H (hedonic loop) signal.
        fear:
            Raw System F (fear loop) signal.
        context:
            Optional context mapping. When supplied the values are averaged to
            influence the governance layer meaning computation.
        context_weights:
            Optional context weights to fine-tune the integration layer.
        """

        state = EmotionState(_clamp(happiness), _clamp(fear))
        context_weights = context_weights or ContextWeights()

        stage_context = self._resolve_stage_context()
        if stage_context and (
            self._active_stage is None
            or stage_context.name != self._active_stage.name
            or stage_context.index != self._active_stage.index
        ):
            self._handle_stage_transition(stage_context)
        elif stage_context:
            ratio_bounds_optional = stage_context.affect_targets.get("ratio")
            if ratio_bounds_optional is not None:
                ratio_bounds = ratio_bounds_optional
                try:
                    self.modulation.set_target_range(ratio_bounds)
                except ValueError:
                    self._logger.debug(
                        "Ignoring invalid ratio bounds %s for stage %s",
                        ratio_bounds,
                        stage_context.name,
                    )
                self.modulation.entropy_buffer.set_tolerance(
                    _entropy_tolerance_from_ratio(ratio_bounds)
                )

        active_stage = self._active_stage or stage_context
        if active_stage:
            targets = active_stage.affect_targets
            happiness_bounds = targets.get("happiness")
            if happiness_bounds:
                state.happiness = _clamp(
                    _regularize_toward_bounds(state.happiness, happiness_bounds)
                )
            fear_bounds = targets.get("fear")
            if fear_bounds:
                state.fear = _clamp(_regularize_toward_bounds(state.fear, fear_bounds))

        ratio = self.core.interpret(state)
        state = self.modulation.modulate(state)
        state = self.processing.process(state)
        state = self.integration.integrate(state, context_weights)

        if active_stage:
            ratio_targets = active_stage.affect_targets.get("ratio")
            if ratio_targets is not None:
                state = self._enforce_ratio_bounds(state, ratio_targets)

        behavior = self.output.classify(state)
        adjusted_ratio = state.ratio()
        equilibrium_prior = self.meta.update(adjusted_ratio)
        meaning = self.governance.meaning(state, context)

        if active_stage:
            ratio_targets = active_stage.affect_targets.get("ratio")
            if ratio_targets is not None and not (
                ratio_targets[0] <= adjusted_ratio <= ratio_targets[1]
            ):
                self._logger.debug(
                    "Stage '%s' ratio %.3f outside target band %s",
                    active_stage.name,
                    adjusted_ratio,
                    ratio_targets,
                )
            self._record_stage_experience(active_stage.name, state, context_weights)
            self._emit_stage_metrics(active_stage, state)

        return {
            "ratio": ratio,
            "adjusted_ratio": adjusted_ratio,
            "behavior": behavior,
            "meaning": meaning,
            "equilibrium_prior": equilibrium_prior,
            "context": {
                "stage": active_stage.name if active_stage else None,
                "stage_index": active_stage.index if active_stage else None,
            },
        }

    def reset(self, *, equilibrium_prior: Optional[float] = None) -> None:
        """Reset internal state between evaluation sessions."""

        self.modulation.reset()
        self.meta.reset(equilibrium_prior=equilibrium_prior)
        self._active_stage = None

    def persist_modulators(
        self,
        state: EmotionState | None = None,
        *,
        lambda_H_base: float = 0.7,
        lambda_I_base: float = 0.3,
        shield_alpha_base: float = 0.95,
    ) -> Dict[str, float]:
        """Derive dynamic modulation coefficients for downstream MPC layers."""

        if state is None:
            return {
                "lambda_H": lambda_H_base,
                "lambda_I": lambda_I_base,
                "shield_alpha": shield_alpha_base,
            }

        happiness = _clamp(state.effective_happiness())
        fear = _clamp(state.effective_fear())
        ratio = state.ratio()

        lambda_H = _clamp(lambda_H_base * (0.8 + 0.4 * happiness - 0.3 * fear))
        lambda_I = _clamp(lambda_I_base * (0.7 + 0.6 * ratio - 0.2 * fear))
        f_over_h = _safe_div(fear, max(happiness, 1e-6), 0.0)
        shield_alpha = _clamp(shield_alpha_base + 0.03 * _softsign(f_over_h - 1.0))

        return {
            "lambda_H": lambda_H,
            "lambda_I": lambda_I,
            "shield_alpha": shield_alpha,
        }


class EEAMultiAgentPolicy:
    """Heuristic multi-agent controller driven by the EEA scaffold."""

    def __init__(
        self, env, *, num_agents: int, config: Mapping[str, object] | None = None
    ) -> None:
        self.env = env
        self.num_agents = num_agents
        self.config = config if config is not None else {}

        vision_space = env.single_observation_space.spaces["vision"]
        self._vision_shape = vision_space.shape
        self._vision_size = int(np.prod(self._vision_shape))
        self._center = np.array(
            [
                self._vision_shape[0] // 2,
                self._vision_shape[1] // 2,
            ]
        )

        seed_obj = self.config.get("seed") if isinstance(self.config, Mapping) else None
        seed: int | np.integer | None
        if isinstance(seed_obj, (int, np.integer)):
            seed = int(seed_obj)
        else:
            seed = None
        self._rng = np.random.default_rng(seed)
        self._agents = [EmotionalEquilibriumAgent() for _ in range(num_agents)]

    def reset(self) -> None:
        for agent in self._agents:
            agent.reset()

    def configure_stage_awareness(
        self,
        stage_provider: Optional[StageProvider] = None,
        telemetry_hook: Optional[Callable[[Mapping[str, object]], None]] = None,
    ) -> None:
        for agent in self._agents:
            agent.configure_stage_awareness(
                stage_provider=stage_provider, telemetry_hook=telemetry_hook
            )

    def get_action(
        self, obs: np.ndarray, agent_id: int, deterministic: bool = False
    ) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        vision, internal_state, neighbor_stats, time_scalar = self._split_obs(obs)

        food_channel = vision[..., 0]
        hazard_channel = vision[..., 1]
        agent_channel = vision[..., 2]

        happiness = _clamp(float(internal_state[0]))
        hazard_signal = float(
            np.clip(hazard_channel.mean() + 0.25 * agent_channel.mean(), 0.0, 1.0)
        )
        context = {
            "neighbor_density": float(np.mean(neighbor_stats)),
            "time": float(time_scalar),
            "food_density": float(food_channel.mean()),
        }

        eea_result = self._agents[agent_id].evaluate(
            happiness=happiness,
            fear=hazard_signal,
            context=context,
        )
        behavior = eea_result["behavior"]

        move: np.ndarray = np.zeros(2, dtype=np.float32)
        if behavior is BehaviorState.ANXIETY:
            move = self._toward_high_value(food_channel)
        elif behavior is BehaviorState.MANIA:
            move = self._wander(deterministic)
        else:  # Vital engagement and fallback
            move = self._toward_high_value(food_channel)
            if np.allclose(move, 0.0):
                move = self._wander(deterministic, scale=0.5)

        return move.astype(np.float32)

    def update(self, data) -> None:  # pragma: no cover - heuristic policy is stateless
        return None

    def state_dict(
        self,
    ) -> Dict[str, object]:  # pragma: no cover - interface compatibility
        return {}

    def load_state_dict(
        self, state: Mapping[str, object] | None
    ) -> None:  # pragma: no cover
        return None

    def get_optimizer_state(self) -> Dict[str, object]:  # pragma: no cover
        return {}

    def load_optimizer_state(
        self, state: Mapping[str, object] | None
    ) -> None:  # pragma: no cover
        return None

    def _split_obs(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        vision = obs[: self._vision_size].reshape(self._vision_shape)
        idx = self._vision_size

        internal_state = obs[idx : idx + 3]
        idx += 3

        neighbor_stats = obs[idx : idx + 3]
        idx += 3

        time_scalar = float(obs[idx]) if idx < len(obs) else 0.0
        return vision, internal_state, neighbor_stats, time_scalar

    def _toward_high_value(self, food_channel: np.ndarray) -> np.ndarray:
        if food_channel.size == 0 or np.max(food_channel) <= 0.0:
            return np.zeros(2, dtype=np.float32)

        targets = np.argwhere(food_channel == np.max(food_channel))
        if targets.size == 0:
            return np.zeros(2, dtype=np.float32)

        target = targets[0]
        delta = target - self._center
        return np.clip(delta, -1.0, 1.0).astype(np.float32)

    def _wander(self, deterministic: bool, scale: float = 1.0) -> np.ndarray:
        if deterministic:
            return np.zeros(2, dtype=np.float32)
        return self._rng.uniform(low=-scale, high=scale, size=2).astype(np.float32)


__all__ = [
    "EmotionalEquilibriumAgent",
    "EmotionState",
    "ContextWeights",
    "BehaviorState",
    "EEAMultiAgentPolicy",
]


class StageProvider(Protocol):
    """Callable that returns the active life-stage descriptor."""

    def __call__(self) -> Optional[Any]: ...


@dataclass(frozen=True)
class StageContext:
    """Snapshot of the active life stage relevant to affect modulation."""

    name: str
    index: int
    affect_targets: Dict[str, Tuple[float, float]]
