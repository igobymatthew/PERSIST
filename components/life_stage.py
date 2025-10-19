"""Life-stage management utilities for Growth-Mimetic agents."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LifeStageSpec:
    """Immutable specification for a single life stage."""

    name: str
    duration: int
    constraint_overrides: Dict[str, float] = field(default_factory=dict)
    affect_targets: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class LifeStageMetrics:
    """Telemetry payload describing the active life stage."""

    index: int
    name: str
    stage_age: int
    duration: int
    progress: float
    affect_targets: Dict[str, Tuple[float, float]]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LifeStageTransition:
    """Represents a transition event between life stages."""

    previous_stage: Optional[LifeStageSpec]
    new_stage: LifeStageSpec
    index: int
    constraints_to_apply: Dict[str, float]
    stage_start_step: int


class LifeStageManager:
    """Tracks stage progression and applies constraint overrides."""

    def __init__(
        self, stages: Sequence[LifeStageSpec], base_constraints: Mapping[str, float]
    ):
        self._stages: List[LifeStageSpec] = list(stages)
        self._base_constraints: Dict[str, float] = dict(base_constraints)
        self._current_index: int = 0
        self._stage_start_step: int = 0
        self._active: bool = bool(self._stages)

    @classmethod
    def from_config(
        cls, viability_cfg: Mapping[str, object]
    ) -> Optional["LifeStageManager"]:
        stage_cfgs = viability_cfg.get("life_stages") or []
        if not stage_cfgs:
            return None

        base_constraints = cls.build_base_constraint_map(
            viability_cfg.get("constraints", [])
        )
        stages: List[LifeStageSpec] = []
        for raw_stage in stage_cfgs:  # type: ignore[assignment]
            name = str(raw_stage["name"])  # type: ignore[index]
            duration = int(raw_stage["duration"])  # type: ignore[index]
            constraint_overrides = {
                key: float(value) for key, value in raw_stage.get("constraint_overrides", {}).items()  # type: ignore[union-attr]
            }
            affect_targets: Dict[str, Tuple[float, float]] = {}
            for affect_name, bounds in raw_stage.get("affect_targets", {}).items():  # type: ignore[union-attr]
                if not isinstance(bounds, Iterable):
                    raise ValueError(
                        f"Affect bounds for '{affect_name}' must be an iterable with two floats."
                    )
                bounds_list = list(bounds)
                if len(bounds_list) != 2:
                    raise ValueError(
                        f"Affect bounds for '{affect_name}' must contain exactly two values."
                    )
                affect_targets[affect_name] = (
                    float(bounds_list[0]),
                    float(bounds_list[1]),
                )

            stages.append(
                LifeStageSpec(
                    name=name,
                    duration=duration,
                    constraint_overrides=constraint_overrides,
                    affect_targets=affect_targets,
                )
            )

        return cls(stages=stages, base_constraints=base_constraints)

    @staticmethod
    def build_base_constraint_map(
        constraints_cfg: Sequence[Mapping[str, object]],
    ) -> Dict[str, float]:
        base: Dict[str, float] = {}
        for constraint in constraints_cfg:
            name = str(constraint["name"])  # type: ignore[index]
            operator = str(constraint["operator"])  # type: ignore[index]
            threshold = constraint["threshold"]  # type: ignore[index]
            if operator == "in":
                min_val, max_val = threshold  # type: ignore[assignment]
                base[f"{name}_min"] = float(min_val)
                base[f"{name}_max"] = float(max_val)
            else:
                base[name] = float(threshold)  # type: ignore[arg-type]
        return base

    def reset(self) -> Optional[LifeStageTransition]:
        if not self._active:
            return None
        self._current_index = 0
        self._stage_start_step = 0
        return LifeStageTransition(
            previous_stage=None,
            new_stage=self._stages[0],
            index=0,
            constraints_to_apply=self._build_constraint_map(self._stages[0]),
            stage_start_step=0,
        )

    def advance(self, episode_step: int) -> Optional[LifeStageTransition]:
        if not self._active:
            return None

        current_stage = self._stages[self._current_index]
        stage_age = episode_step - self._stage_start_step

        if stage_age < current_stage.duration:
            return None

        if self._current_index >= len(self._stages) - 1:
            return None

        previous_stage = current_stage
        self._current_index += 1
        self._stage_start_step += current_stage.duration
        new_stage = self._stages[self._current_index]

        return LifeStageTransition(
            previous_stage=previous_stage,
            new_stage=new_stage,
            index=self._current_index,
            constraints_to_apply=self._build_constraint_map(new_stage),
            stage_start_step=self._stage_start_step,
        )

    def metrics(self, episode_step: int) -> Optional[LifeStageMetrics]:
        if not self._active:
            return None
        current_stage = self._stages[self._current_index]
        stage_age = max(0, episode_step - self._stage_start_step)
        duration = max(1, current_stage.duration)
        progress = min(stage_age / duration, 1.0)
        return LifeStageMetrics(
            index=self._current_index,
            name=current_stage.name,
            stage_age=stage_age,
            duration=current_stage.duration,
            progress=progress,
            affect_targets=current_stage.affect_targets,
        )

    def current_constraints(self) -> Dict[str, float]:
        if not self._active:
            return {}
        return self._build_constraint_map(self._stages[self._current_index])

    def current_affect_targets(self) -> Dict[str, Tuple[float, float]]:
        if not self._active:
            return {}
        return dict(self._stages[self._current_index].affect_targets)

    def current_stage_index(self) -> Optional[int]:
        if not self._active:
            return None
        return self._current_index

    def current_stage_name(self) -> Optional[str]:
        if not self._active:
            return None
        return self._stages[self._current_index].name

    def current_stage_summary(self) -> Optional[Dict[str, object]]:
        if not self._active:
            return None
        stage = self._stages[self._current_index]
        return {
            "index": self._current_index,
            "name": stage.name,
            "affect_targets": dict(stage.affect_targets),
            "duration": stage.duration,
        }

    def is_active(self) -> bool:
        return self._active

    def timeline(self) -> List[Dict[str, object]]:
        timeline: List[Dict[str, object]] = []
        if not self._active:
            return timeline
        start_step = 0
        for index, stage in enumerate(self._stages):
            end_step = start_step + stage.duration
            timeline.append(
                {
                    "index": index,
                    "name": stage.name,
                    "start_step": start_step,
                    "end_step": end_step,
                    "constraint_overrides": dict(stage.constraint_overrides),
                    "affect_targets": dict(stage.affect_targets),
                }
            )
            start_step = end_step
        return timeline

    def _build_constraint_map(self, stage: LifeStageSpec) -> Dict[str, float]:
        constraints = dict(self._base_constraints)
        constraints.update(stage.constraint_overrides)
        return constraints
