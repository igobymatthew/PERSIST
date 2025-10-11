*** a/build-scaffolding-from-eea.md/agents/eea_agent.py
--- b/build-scaffolding-from-eea.md/agents/eea_agent.py
@@
 from dataclasses import dataclass, field
 from enum import Enum
 from typing import Callable, Dict, List, Mapping, Optional, Tuple
 
 
 def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
     """Clamp ``value`` between ``lower`` and ``upper`` (inclusive)."""
-
     return max(lower, min(upper, value))
 
+def _softsign(x: float) -> float:
+    return x / (1.0 + abs(x))
+
+def _safe_div(num: float, den: float, default: float = 0.0) -> float:
+    return num / den if den != 0 else default
+
 
 @dataclass
 class EmotionState:
@@
     def ratio(self) -> float:
         """Return the normalized hedonic ratio H/(H+F).
 
         If both signals are zero the ratio defaults to 0.5, indicating a
         neutral equilibrium. The same default is applied when the ratio would
         otherwise be undefined.
         """
-
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
-
         ratio = state.ratio()
         return _clamp(ratio)
 
 
 @dataclass
 class ValenceRegulator:
     """Maintains a stable ratio between positive and negative affect."""
 
-    happiness_weight: float = 0.5
-    fear_weight: float = 0.5
+    happiness_weight: float = 0.5
+    fear_weight: float = 0.5
+    # Research-backed nudges
+    positivity_offset: float = 0.08   # slight boost to H when both small (positivity offset)
+    negativity_bias: float = 1.6      # amplify F per-event weight (negativity bias)
 
     def regulate(self, state: EmotionState) -> EmotionState:
-        happiness = _clamp(state.happiness * self.happiness_weight)
-        fear = _clamp(state.fear * self.fear_weight)
+        happiness = state.happiness * self.happiness_weight
+        fear = state.fear * self.fear_weight
+
+        # Positivity offset when both affective signals are near zero
+        if happiness < 0.15 and fear < 0.15:
+            happiness = happiness + self.positivity_offset
+
+        # Negativity bias (amplify fear weight)
+        fear = fear * self.negativity_bias
+
+        happiness = _clamp(happiness)
+        fear = _clamp(fear)
         state.modulated_happiness = happiness
         state.modulated_fear = fear
         return state
 
 
 @dataclass
 class ContrastNormalizer:
@@
     def normalize(self, state: EmotionState) -> EmotionState:
         happiness = state.effective_happiness()
         fear = state.effective_fear()
         delta = abs(happiness - fear)
         if delta < self.contrast_floor:
             adjustment = self.contrast_floor - delta
             # push signals apart symmetrically to preserve mean intensity
             happiness += adjustment / 2
             fear = _clamp(fear - adjustment / 2)
-        state.modulated_happiness = _clamp(happiness)
-        state.modulated_fear = _clamp(fear)
+        state.modulated_happiness = _clamp(happiness)
+        state.modulated_fear = _clamp(fear)
         return state
 
 
 @dataclass
 class EntropyBuffer:
     """Prevents emotional monotony or overload by referencing history."""
@@
         if len(self.history) > self.max_history:
             self.history.pop(0)
         return state
 
     def reset(self) -> None:
         """Clear stored history to remove lingering momentum between sessions."""
-
         self.history.clear()
 
 
 @dataclass
 class ModulationLayer:
     """Aggregates the modulation components described in the specification."""
 
     target_range: Tuple[float, float] = (0.7, 0.8)
     valence_regulator: ValenceRegulator = field(default_factory=ValenceRegulator)
     contrast_normalizer: ContrastNormalizer = field(default_factory=ContrastNormalizer)
     entropy_buffer: EntropyBuffer = field(default_factory=EntropyBuffer)
+    # Soft clamps to avoid saturation oscillations
+    soft_cap: float = 0.98
 
     def __post_init__(self) -> None:
         lo, hi = self.target_range
         if not (0.0 <= lo <= hi <= 1.0):
             raise ValueError("target_range must lie within [0, 1] and be ordered")
 
     def modulate(self, state: EmotionState) -> EmotionState:
         state = self.valence_regulator.regulate(state)
         state = self.contrast_normalizer.normalize(state)
         state = self.entropy_buffer.dampen(state)
 
         ratio = state.ratio()
         lo, hi = self.target_range
         if ratio < lo:
             state.modulated_happiness = _clamp(
-                state.effective_happiness() * (1 + (lo - ratio))
+                min(self.soft_cap, state.effective_happiness() * (1 + (lo - ratio)))
             )
         elif ratio > hi:
             state.modulated_fear = _clamp(
-                state.effective_fear() * (1 + (ratio - hi))
+                min(self.soft_cap, state.effective_fear() * (1 + (ratio - hi)))
             )
         return state
 
     def reset(self) -> None:
         """Reset modulation stateful components such as the entropy history."""
-
         self.entropy_buffer.reset()
 
 
-class ProcessingLayer:
+@dataclass
+class ProcessingLayer:
     """Models adaptive feedback loops between fear and happiness."""
 
-    anticipation_gain: float = 0.1
-    reinforcement_gain: float = 0.05
-    calibration_gain: float = 0.02
+    anticipation_gain: float = 0.10     # fear sharpen → reward acquisition
+    reinforcement_gain: float = 0.05    # avoidance success → reduce fear
+    calibration_gain: float = 0.02      # slow drift toward 0.5 ratio
 
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
 
         state.modulated_happiness = _clamp(happiness)
         state.modulated_fear = _clamp(fear)
         return state
@@
 class IntegrationLayer:
     """Contextualizes signals using environment, social, and internal weights."""
 
     def integrate(self, state: EmotionState, context_weights: ContextWeights) -> EmotionState:
         weights = context_weights.normalize()
         happiness = state.effective_happiness()
         fear = state.effective_fear()
-        happiness *= 1 + weights[0] * 0.1 + weights[1] * 0.05
-        fear *= 1 + weights[2] * 0.1
+        happiness *= 1 + weights[0] * 0.10 + weights[1] * 0.05
+        fear *= 1 + weights[2] * 0.10
         state.modulated_happiness = _clamp(happiness)
         state.modulated_fear = _clamp(fear)
         return state
@@
 @dataclass
 class OutputLayer:
     """Maps the final ratio to a qualitative behavioral manifestation."""
 
     vital_range: Tuple[float, float] = (0.7, 0.8)
     apathy_threshold: float = 0.2
     mania_threshold: float = 0.8
+    high_fear_threshold: float = 0.7   # moved from hardcoded logic into config
 
     def __post_init__(self) -> None:
         lo, hi = self.vital_range
         if not (0.0 <= self.apathy_threshold <= lo <= hi <= 1.0):
             raise ValueError("Behavior thresholds must satisfy apathy <= vital_range <= 1.0")
         if not (hi <= self.mania_threshold <= 1.0):
             raise ValueError("Mania threshold must be >= vital_range[1] and <= 1.0")
 
     def classify(self, state: EmotionState) -> BehaviorState:
         ratio = state.ratio()
         lo, hi = self.vital_range
         if lo <= ratio <= hi:
             return BehaviorState.VITAL_ENGAGEMENT
         if ratio < self.apathy_threshold:
             return BehaviorState.APATHY
-        if ratio < lo:
+        if ratio < lo:
             return BehaviorState.ANXIETY
         if ratio >= self.mania_threshold:
             return BehaviorState.MANIA
         # Between ``hi`` and ``mania_threshold`` we consider the system heightened yet stable.
         return BehaviorState.VITAL_ENGAGEMENT
@@
 @dataclass
 class GovernanceLayer:
     """Defines the evaluative function for meaning derived from H and F."""
 
     fear_bounds: Tuple[float, float] = (0.2, 0.4)
     context_weight: float = 0.1
-    vitality_fn: Optional[Callable[[EmotionState], float]] = None
+    vitality_fn: Optional[Callable[[EmotionState], float]] = None
+    vitality_mode: str = "ratio"  # {"ratio","difference","softsign","geomean"}
 
     def __post_init__(self) -> None:
         lo, hi = self.fear_bounds
         if not (0.0 <= lo <= hi):
             raise ValueError("fear_bounds must be ordered and non-negative")
 
     def meaning(
         self,
         state: EmotionState,
         context: Mapping[str, float] | None = None,
     ) -> float:
-        ratio = state.ratio()
-        fear = state.effective_fear()
-        happiness = max(1e-6, state.effective_happiness())
-        context_modifier = 0.0
-        if context:
-            context_modifier = sum(context.values()) / max(1, len(context))
-        fear_ratio = fear / happiness
+        ratio = state.ratio()
+        fear = state.effective_fear()
+        happiness = max(1e-6, state.effective_happiness())
+        context_modifier = 0.0
+        if context:
+            context_modifier = sum(context.values()) / max(1, len(context))
+        fear_ratio = fear / happiness
         lo, hi = self.fear_bounds
         fear_term = 1.0 if lo <= fear_ratio <= hi else 0.5
-        vitality = self.vitality_fn(state) if self.vitality_fn is not None else ratio * fear_term
+        if self.vitality_fn is not None:
+            vitality = self.vitality_fn(state)
+        else:
+            if self.vitality_mode == "difference":
+                vitality = _clamp(happiness - fear + 0.5)  # center then clamp
+            elif self.vitality_mode == "softsign":
+                vitality = 0.5 + 0.5 * _softsign(happiness - fear)
+            elif self.vitality_mode == "geomean":
+                vitality = (happiness * (1.0 - fear)) ** 0.5
+            else:  # "ratio"
+                vitality = ratio
+        vitality = vitality * fear_term
         return vitality + self.context_weight * context_modifier
@@
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
@@
-        state = EmotionState(happiness, fear)
+        state = EmotionState(_clamp(happiness), _clamp(fear))
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
 
     def reset(self, *, equilibrium_prior: Optional[float] = None) -> None:
         """Reset internal state between evaluation sessions."""
 
         self.modulation.reset()
         self.meta.reset(equilibrium_prior=equilibrium_prior)
+
+    # ----- Integration helpers for PERSIST / MPC layers -----
+    def persist_modulators(
+        self,
+        state: EmotionState | None = None,
+        *,
+        lambda_H_base: float = 0.7,
+        lambda_I_base: float = 0.3,
+        shield_alpha_base: float = 0.95,
+    ) -> Dict[str, float]:
+        """
+        Compute dynamic modulators for PERSIST:
+          - λ_H (homeostat weight): ↑ with H, ↓ with F
+          - λ_I (intrinsic weight): ↑ when vitality high (explore), ↓ when F dominates
+          - shield α (safety confidence): ↑ with F/H (riskier → stricter shield)
+        """
+        # If caller passes a preprocessed state, use it; otherwise no-op defaults
+        if state is None:
+            return {
+                "lambda_H": lambda_H_base,
+                "lambda_I": lambda_I_base,
+                "shield_alpha": shield_alpha_base,
+            }
+        H = _clamp(state.effective_happiness())
+        F = _clamp(state.effective_fear())
+        ratio = state.ratio()
+
+        # λ_H grows with H and shrinks with F; keep in [0,1]
+        lambda_H = _clamp(lambda_H_base * (0.8 + 0.4 * H - 0.3 * F))
+        # λ_I prefers exploration when vitality (ratio) is high and fear is moderate
+        lambda_I = _clamp(lambda_I_base * (0.7 + 0.6 * ratio - 0.2 * F))
+        # Shield confidence increases with relative fear (F/H), soft-bounded
+        f_over_h = _safe_div(F, max(H, 1e-6), 0.0)
+        shield_alpha = _clamp(shield_alpha_base + 0.03 * _softsign(f_over_h - 1.0))
+
+        return {
+            "lambda_H": lambda_H,
+            "lambda_I": lambda_I,
+            "shield_alpha": shield_alpha,
+        }