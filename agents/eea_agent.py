"""Scaffolding for the Emotional Equilibrium Architecture (EEA).

This module translates the conceptual layers described in ``docs/EEA.md``
into a light-weight, inspectable agent scaffold. The goal is not to model
neuroscience, but to provide a well-documented software artifact that mirrors
the original architectural intent and can be expanded with domain specific
logic later on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple


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

        return self.modulated_happiness if self.modulated_happiness is not None else self.happiness

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
        return max(0.0, min(1.0, ratio))


@dataclass
class ValenceRegulator:
    """Maintains a stable ratio between positive and negative affect."""

    happiness_weight: float = 0.5
    fear_weight: float = 0.5

    def regulate(self, state: EmotionState) -> EmotionState:
        happiness = max(0.0, state.happiness * self.happiness_weight)
        fear = max(0.0, state.fear * self.fear_weight)
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
            fear = max(0.0, fear - adjustment / 2)
        state.modulated_happiness = happiness
        state.modulated_fear = fear
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
            state.modulated_happiness = state.effective_happiness() * 0.95
        if delta_f < self.tolerance:
            state.modulated_fear = state.effective_fear() * 1.05

        self.history.append(EmotionState(state.happiness, state.fear))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return state


class ModulationLayer:
    """Aggregates the modulation components described in the specification."""

    def __init__(self) -> None:
        self.valence_regulator = ValenceRegulator()
        self.contrast_normalizer = ContrastNormalizer()
        self.entropy_buffer = EntropyBuffer()
        self.target_range: Tuple[float, float] = (0.7, 0.8)

    def modulate(self, state: EmotionState) -> EmotionState:
        state = self.valence_regulator.regulate(state)
        state = self.contrast_normalizer.normalize(state)
        state = self.entropy_buffer.dampen(state)

        ratio = state.ratio()
        lo, hi = self.target_range
        if ratio < lo:
            state.modulated_happiness = state.effective_happiness() * (1 + (lo - ratio))
        elif ratio > hi:
            state.modulated_fear = state.effective_fear() * (1 + (ratio - hi))
        return state


class ProcessingLayer:
    """Models adaptive feedback loops between fear and happiness."""

    anticipation_gain: float = 0.1
    reinforcement_gain: float = 0.05
    calibration_gain: float = 0.02

    def process(self, state: EmotionState) -> EmotionState:
        happiness = state.effective_happiness()
        fear = state.effective_fear()

        # Feedback A – anticipation: fear sharpens reward acquisition
        happiness += fear * self.anticipation_gain
        # Feedback B – reinforcement: successful avoidance reduces fear
        fear *= (1.0 - self.reinforcement_gain)

        # Feedback C – calibration: push toward stable ratio
        ratio = state.ratio()
        if ratio < 0.5:
            happiness += self.calibration_gain
        else:
            fear += self.calibration_gain

        state.modulated_happiness = max(0.0, happiness)
        state.modulated_fear = max(0.0, fear)
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

    def integrate(self, state: EmotionState, context_weights: ContextWeights) -> EmotionState:
        weights = context_weights.normalize()
        happiness = state.effective_happiness()
        fear = state.effective_fear()
        happiness *= 1 + weights[0] * 0.1 + weights[1] * 0.05
        fear *= 1 + weights[2] * 0.1
        state.modulated_happiness = happiness
        state.modulated_fear = fear
        return state


class BehaviorState(Enum):
    VITAL_ENGAGEMENT = "vital_engagement"
    APATHY = "apathy"
    ANXIETY = "anxiety"
    MANIA = "mania"


class OutputLayer:
    """Maps the final ratio to a qualitative behavioral manifestation."""

    def classify(self, state: EmotionState) -> BehaviorState:
        ratio = state.ratio()
        if 0.7 <= ratio <= 0.8:
            return BehaviorState.VITAL_ENGAGEMENT
        if ratio < 0.2:
            return BehaviorState.APATHY
        if ratio < 0.7:
            return BehaviorState.ANXIETY
        return BehaviorState.MANIA


@dataclass
class MetaLayer:
    """Captures long-term adaptation as a moving prior on the equilibrium."""

    smoothing: float = 0.05
    equilibrium_prior: float = 0.75

    def update(self, observed_ratio: float) -> float:
        self.equilibrium_prior = (1 - self.smoothing) * self.equilibrium_prior + self.smoothing * observed_ratio
        return self.equilibrium_prior


@dataclass
class GovernanceLayer:
    """Defines the evaluative function for meaning derived from H and F."""

    fear_bounds: Tuple[float, float] = (0.2, 0.4)

    def meaning(self, state: EmotionState, context: Mapping[str, float] | None = None) -> float:
        ratio = state.ratio()
        fear = state.effective_fear()
        happiness = state.effective_happiness()
        context_modifier = 0.0
        if context:
            context_modifier = sum(context.values()) / max(1, len(context))
        fear_term = 1.0 if self.fear_bounds[0] <= fear / max(1e-6, happiness) <= self.fear_bounds[1] else 0.5
        return ratio * fear_term + 0.1 * context_modifier


class EmotionalEquilibriumAgent:
    """Agent scaffold implementing the Emotional Equilibrium Architecture."""

    def __init__(self) -> None:
        self.core = CorePrincipleLayer()
        self.modulation = ModulationLayer()
        self.processing = ProcessingLayer()
        self.integration = IntegrationLayer()
        self.output = OutputLayer()
        self.meta = MetaLayer()
        self.governance = GovernanceLayer()

    def evaluate(
        self,
        happiness: float,
        fear: float,
        *,
        context: Optional[Mapping[str, float]] = None,
        context_weights: Optional[ContextWeights] = None,
    ) -> Dict[str, float | BehaviorState]:
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

        state = EmotionState(happiness, fear)
        ratio = self.core.interpret(state)
        state = self.modulation.modulate(state)
        state = self.processing.process(state)
        if context_weights is None:
            context_weights = ContextWeights()
        state = self.integration.integrate(state, context_weights)
        behavior = self.output.classify(state)
        equilibrium_prior = self.meta.update(state.ratio())
        meaning = self.governance.meaning(state, context)

        return {
            "ratio": ratio,
            "adjusted_ratio": state.ratio(),
            "behavior": behavior,
            "meaning": meaning,
            "equilibrium_prior": equilibrium_prior,
        }


__all__ = [
    "EmotionalEquilibriumAgent",
    "EmotionState",
    "ContextWeights",
    "BehaviorState",
]
