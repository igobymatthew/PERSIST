"""Unit tests for the Growth-Mimetic life stage manager."""

from __future__ import annotations

from components.life_stage import LifeStageManager


def build_viability_config():
    return {
        "constraints": [
            {
                "name": "energy_min",
                "variable": "energy",
                "operator": ">=",
                "threshold": 0.2,
                "description": "Base energy requirement",
            },
            {
                "name": "temp_range",
                "variable": "temp",
                "operator": "in",
                "threshold": [0.3, 0.7],
                "description": "Safe temperature window",
            },
            {
                "name": "integrity_min",
                "variable": "integrity",
                "operator": ">=",
                "threshold": 0.6,
                "description": "Structural integrity floor",
            },
        ],
        "life_stages": [
            {
                "name": "infant",
                "duration": 2,
                "constraint_overrides": {
                    "energy_min": 0.15,
                    "temp_range_min": 0.25,
                    "temp_range_max": 0.65,
                },
                "affect_targets": {
                    "happiness": [0.55, 0.85],
                    "fear": [0.05, 0.30],
                },
            },
            {
                "name": "juvenile",
                "duration": 3,
                "constraint_overrides": {
                    "energy_min": 0.22,
                    "integrity_min": 0.62,
                },
            },
            {
                "name": "adult",
                "duration": 4,
                "constraint_overrides": {
                    "energy_min": 0.28,
                },
            },
        ],
    }


def test_life_stage_manager_progression():
    viability_cfg = build_viability_config()
    manager = LifeStageManager.from_config(viability_cfg)
    assert manager is not None

    initial_transition = manager.reset()
    assert initial_transition is not None
    assert initial_transition.new_stage.name == "infant"

    constraints = initial_transition.constraints_to_apply
    # Base constraint retained
    assert constraints["integrity_min"] == 0.6
    # Overrides applied
    assert constraints["energy_min"] == 0.15
    assert constraints["temp_range_min"] == 0.25
    assert constraints["temp_range_max"] == 0.65

    metrics = manager.metrics(0)
    assert metrics is not None
    assert metrics.index == 0
    assert metrics.progress == 0.0
    assert metrics.affect_targets["happiness"] == (0.55, 0.85)

    # Advance to juvenile after duration steps
    transition = manager.advance(2)
    assert transition is not None
    assert transition.new_stage.name == "juvenile"
    constraints = transition.constraints_to_apply
    assert constraints["energy_min"] == 0.22
    assert constraints["integrity_min"] == 0.62
    # Temperature bounds fall back to base values when not overridden
    assert constraints["temp_range_min"] == 0.3
    assert constraints["temp_range_max"] == 0.7

    metrics = manager.metrics(3)
    assert metrics is not None
    assert metrics.index == 1
    assert 0.0 < metrics.progress <= 1.0

    # Adult stage at cumulative step 5 (2 + 3)
    transition = manager.advance(5)
    assert transition is not None
    assert transition.new_stage.name == "adult"
    constraints = transition.constraints_to_apply
    assert constraints["energy_min"] == 0.28
    # Integrity override persists from base when not specified
    assert constraints["integrity_min"] == 0.6

    # No transition beyond final stage
    assert manager.advance(12) is None


def test_life_stage_timeline_preview_contains_all_stages():
    manager = LifeStageManager.from_config(build_viability_config())
    assert manager is not None
    timeline = manager.timeline()
    assert len(timeline) == 3
    assert timeline[0]["name"] == "infant"
    assert timeline[-1]["end_step"] == sum(stage["duration"] for stage in build_viability_config()["life_stages"])  # type: ignore[index]
