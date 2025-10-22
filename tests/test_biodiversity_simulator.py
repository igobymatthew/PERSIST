import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.multi_agent_gridlife import MultiAgentGridLifeEnv
from population.ensemble_shield import PopulationSafetyCoordinator


def _make_biodiversity_config():
    return {
        "multiagent": {
            "num_agents": 4,
            "observation": {"view_radius": 1},
            "termination": {"max_steps": 50},
        },
        "agent_types": {
            "default": {
                "homeostasis": {"mu": [0.7, 0.5, 0.9]},
            }
        },
        "resource_model": {"food": {"initial_density": 0.4}},
        "biodiversity": {
            "species": [
                {
                    "id": "forager",
                    "metabolism": {"energy_decay": 0.01, "hazard_tolerance": 0.4},
                    "population": {"min_count": 1, "max_count": 6, "energy_floor": 0.3},
                    "reward_modifiers": {"foraging_bonus": 1.0, "hazard_penalty": 0.2},
                },
                {
                    "id": "sentinel",
                    "metabolism": {"energy_decay": 0.015, "hazard_tolerance": 0.6},
                    "population": {
                        "min_count": 1,
                        "max_count": 4,
                        "energy_floor": 0.35,
                    },
                    "reward_modifiers": {"foraging_bonus": 0.9, "hazard_penalty": 0.1},
                },
            ],
            "succession": {
                "interval": 10,
                "phases": [
                    {"name": "early", "food_density": 0.4, "hazard_rate": 0.05},
                    {"name": "mature", "food_density": 0.3, "hazard_rate": 0.15},
                ],
            },
            "telemetry": {"stability_window": 5},
        },
    }


def test_multi_species_environment_emits_biodiversity_metrics():
    config = _make_biodiversity_config()
    env = MultiAgentGridLifeEnv(config)

    observations, infos = env.reset()
    assert "__all__" in infos
    active_species = {
        payload["species_id"] for agent, payload in infos.items() if agent != "__all__"
    }
    assert len(active_species) >= 2

    for _ in range(25):
        actions = {agent: env.single_action_space.sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

    global_info = infos["__all__"]
    assert global_info["species_richness"] >= 1
    assert "succession_phase_name" in global_info
    assert "trophic_stability" in global_info


def test_population_coordinator_enforces_energy_floor():
    config = _make_biodiversity_config()
    coordinator = PopulationSafetyCoordinator(
        {s["id"]: s for s in config["biodiversity"]["species"]}
    )

    snapshot = {
        "forager": {"count": 1, "avg_energy": 0.1, "mutualism_events": 0},
        "sentinel": {"count": 2, "avg_energy": 0.5, "mutualism_events": 0},
    }

    stable, details = coordinator.evaluate(snapshot)
    assert not stable
    assert details["forager"]["within_bounds"] is False
    assert details["forager"]["reason"] == "energy_below_floor"
