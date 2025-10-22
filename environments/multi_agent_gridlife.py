from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from population.ensemble_shield import PopulationSafetyCoordinator


class MultiAgentGridLifeEnv(gym.Env):
    """Multi-agent GridLife environment with biodiversity and habitat succession."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "GridLifeMA-v0"}

    def __init__(self, config):
        self.config = config
        self.grid_size = tuple(config["multiagent"].get("grid_size", (10, 10)))
        self.num_agents = int(config["multiagent"]["num_agents"])
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.biodiversity_cfg = config.get("biodiversity", {})
        self.species_catalog = {
            entry["id"]: entry for entry in self.biodiversity_cfg.get("species", [])
        }
        if not self.species_catalog:
            raise ValueError(
                "Biodiversity Fabric Simulator requires at least one species archetype."
            )

        assignment_cycle = self.biodiversity_cfg.get("assignments")
        if assignment_cycle is None:
            assignment_cycle = list(self.species_catalog.keys())
        if not assignment_cycle:
            assignment_cycle = list(self.species_catalog.keys())

        self.agent_species: Dict[str, str] = {}
        for idx, agent_id in enumerate(self.agents):
            species_id = assignment_cycle[idx % len(assignment_cycle)]
            if species_id not in self.species_catalog:
                raise KeyError(
                    f"Unknown species id '{species_id}' for agent assignment"
                )
            self.agent_species[agent_id] = species_id

        baseline_homeostasis = np.array(
            config["agent_types"]["default"]["homeostasis"]["mu"], dtype=np.float32
        )
        self.internal_states: Dict[str, np.ndarray] = {
            agent_id: baseline_homeostasis.copy() for agent_id in self.agents
        }
        self.agent_positions: Dict[str, np.ndarray] = {
            agent_id: np.zeros(2, dtype=np.int32) for agent_id in self.agents
        }
        self.alive_agents = set(self.agents)

        self.single_action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        view_radius = config["multiagent"]["observation"]["view_radius"]
        vision_shape = (2 * view_radius + 1, 2 * view_radius + 1, 3)

        self.single_observation_space = DictSpace(
            {
                "vision": Box(low=0, high=10, shape=vision_shape, dtype=np.float32),
                "x": Box(low=0, high=1, shape=(3,), dtype=np.float32),
                "neighbors": Box(
                    low=0, high=self.num_agents, shape=(3,), dtype=np.float32
                ),
                "time": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )
        self.observation_space = DictSpace(
            {agent: self.single_observation_space for agent in self.agents}
        )
        self.action_space = DictSpace(
            {agent: self.single_action_space for agent in self.agents}
        )

        succession_cfg = self.biodiversity_cfg.get("succession", {})
        self.succession_interval = int(succession_cfg.get("interval", 50))
        self.succession_phases: List[Dict] = succession_cfg.get("phases", [])
        if not self.succession_phases:
            self.succession_phases = [
                {"name": "baseline", "food_density": 0.35, "hazard_rate": 0.05}
            ]
        self.active_phase_index = 0
        self.current_phase = self.succession_phases[self.active_phase_index]
        self.last_succession_step = 0

        telemetry_cfg = self.biodiversity_cfg.get("telemetry", {})
        window = int(telemetry_cfg.get("stability_window", 12))
        self.trophic_window = deque(maxlen=max(window, 1))

        self.population_coordinator = PopulationSafetyCoordinator(self.species_catalog)

        self.timestep = 0
        self.max_steps = int(config["multiagent"]["termination"]["max_steps"])
        self.food_map = np.zeros(self.grid_size)
        self.hazard_map = np.zeros(self.grid_size)
        self.mutualism_events = Counter()
        self.population_snapshot: Dict[str, Dict] = {}

        print(
            f"✅ MultiAgentGridLifeEnv initialized for {self.num_agents} agents across"
            f" {len(self.species_catalog)} species."
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timestep = 0
        self.alive_agents = set(self.agents)
        self.active_phase_index = 0
        self.current_phase = self.succession_phases[self.active_phase_index]
        self.last_succession_step = 0
        self.trophic_window.clear()
        self.mutualism_events.clear()

        baseline_homeostasis = np.array(
            self.config["agent_types"]["default"]["homeostasis"]["mu"], dtype=np.float32
        )

        for agent_id in self.agents:
            self.agent_positions[agent_id] = self.np_random.integers(
                0, self.grid_size[0], size=2
            )
            self.internal_states[agent_id] = baseline_homeostasis.copy()

        self._regenerate_resources()
        self.population_snapshot = self._build_population_snapshot()

        observations = {
            agent_id: self._get_obs_for_agent(agent_id)
            for agent_id in self.alive_agents
        }
        infos = {
            agent_id: {"species_id": self.agent_species[agent_id]}
            for agent_id in self.alive_agents
        }
        infos["__all__"] = self._compose_global_info(stable=True)
        return observations, infos

    def step(self, action_dict):
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}
        infos = {
            agent_id: {"species_id": self.agent_species[agent_id]}
            for agent_id in self.agents
        }

        # 1. Apply actions and update states
        for agent_id, action in action_dict.items():
            if agent_id not in self.alive_agents:
                continue

            species_cfg = self.species_catalog[self.agent_species[agent_id]]
            action = np.clip(action, -1.0, 1.0)
            self.agent_positions[agent_id] = np.clip(
                self.agent_positions[agent_id] + action,
                [0, 0],
                [self.grid_size[0] - 1, self.grid_size[1] - 1],
            ).astype(int)

            pos = tuple(self.agent_positions[agent_id])
            if self.food_map[pos] > 0:
                bonus = species_cfg.get("reward_modifiers", {}).get(
                    "foraging_bonus", 1.0
                )
                self.internal_states[agent_id][0] = min(
                    1.0, self.internal_states[agent_id][0] + 0.4 * bonus
                )
                self.food_map[pos] = 0
                rewards[agent_id] += 1.0 * bonus

            hazard_intensity = self.hazard_map[pos]
            if hazard_intensity > 0:
                tolerance = species_cfg.get("metabolism", {}).get(
                    "hazard_tolerance", 0.3
                )
                penalty = species_cfg.get("reward_modifiers", {}).get(
                    "hazard_penalty", 0.2
                )
                damage = max(0.0, hazard_intensity - tolerance)
                if damage > 0:
                    self.internal_states[agent_id][2] = max(
                        0.0, self.internal_states[agent_id][2] - damage
                    )
                    rewards[agent_id] -= penalty * damage

            decay = species_cfg.get("metabolism", {}).get("energy_decay", 0.01)
            self.internal_states[agent_id][0] -= decay
            self.internal_states[agent_id][0] = max(
                self.internal_states[agent_id][0], 0.0
            )

            if (
                self.internal_states[agent_id][0] <= 0
                or self.internal_states[agent_id][2] <= 0
            ):
                terminations[agent_id] = True

        self._register_mutualism_events()

        # 2. Cull dead agents
        for agent_id, terminated in terminations.items():
            if terminated and agent_id in self.alive_agents:
                self.alive_agents.remove(agent_id)

        # 3. Regenerate world, update succession
        self.timestep += 1
        if (self.timestep - self.last_succession_step) >= self.succession_interval:
            self._advance_succession_phase()
        self._regenerate_resources()

        # 4. Build observations and infos
        observations = {
            agent_id: self._get_obs_for_agent(agent_id)
            for agent_id in self.alive_agents
        }
        for agent_id in self.alive_agents:
            infos[agent_id]["internal_state"] = self.internal_states[agent_id].copy()
            infos[agent_id]["alive"] = True
        for agent_id in set(self.agents) - self.alive_agents:
            infos[agent_id]["alive"] = False

        # 5. Determine termination and truncation signals
        truncations["__all__"] = self.timestep >= self.max_steps
        terminations["__all__"] = not self.alive_agents

        self.population_snapshot = self._build_population_snapshot()
        stable, population_metrics = self.population_coordinator.evaluate(
            self.population_snapshot
        )
        self.trophic_window.append(population_metrics["trophic_stability"])
        infos["__all__"] = self._compose_global_info(stable, population_metrics)

        return observations, rewards, terminations, truncations, infos

    def _get_obs_for_agent(self, agent_id: str) -> Dict[str, np.ndarray]:
        pos = self.agent_positions[agent_id]
        view_radius = self.config["multiagent"]["observation"]["view_radius"]
        padded_world = self._get_padded_world_state()
        r_start, c_start = pos[0], pos[1]
        vision = padded_world[
            r_start : r_start + 2 * view_radius + 1,
            c_start : c_start + 2 * view_radius + 1,
            :,
        ]

        neighbor_features = self._compute_neighbor_features(agent_id)
        time = np.array([self.timestep / max(self.max_steps, 1)])

        return {
            "vision": vision.astype(np.float32),
            "x": self.internal_states[agent_id].astype(np.float32),
            "neighbors": neighbor_features.astype(np.float32),
            "time": time.astype(np.float32),
        }

    def _compute_neighbor_features(self, agent_id: str) -> np.ndarray:
        view_radius = self.config["multiagent"]["observation"]["view_radius"]
        pos = self.agent_positions[agent_id]
        counts = 0
        competition = 0
        resource_density = 0.0

        for other_id, other_pos in self.agent_positions.items():
            if other_id == agent_id or other_id not in self.alive_agents:
                continue
            distance = np.sum(np.abs(other_pos - pos))
            if distance <= view_radius:
                counts += 1
                if self.agent_species[other_id] == self.agent_species[agent_id]:
                    competition += 1

        local_window = self._get_local_food_window(pos, view_radius)
        if local_window.size > 0:
            resource_density = float(np.mean(local_window > 0))
        return np.array([counts, competition, resource_density])

    def _get_local_food_window(self, pos: np.ndarray, radius: int) -> np.ndarray:
        r_start = max(pos[0] - radius, 0)
        c_start = max(pos[1] - radius, 0)
        r_end = min(pos[0] + radius + 1, self.grid_size[0])
        c_end = min(pos[1] + radius + 1, self.grid_size[1])
        return self.food_map[r_start:r_end, c_start:c_end]

    def _get_padded_world_state(self) -> np.ndarray:
        view_radius = self.config["multiagent"]["observation"]["view_radius"]
        agent_pos_map = np.zeros(self.grid_size)
        for other_agent_id, other_pos in self.agent_positions.items():
            if other_agent_id in self.alive_agents:
                agent_pos_map[other_pos[0], other_pos[1]] += 1
        world_state = np.stack([self.food_map, self.hazard_map, agent_pos_map], axis=-1)
        padding = ((view_radius, view_radius), (view_radius, view_radius), (0, 0))
        return np.pad(world_state, padding, mode="constant", constant_values=0)

    def _regenerate_resources(self):
        food_density = float(self.current_phase.get("food_density", 0.3))
        hazard_rate = float(self.current_phase.get("hazard_rate", 0.05))
        self.food_map = (self.np_random.random(self.grid_size) < food_density).astype(
            float
        )
        self.hazard_map = (self.np_random.random(self.grid_size) < hazard_rate).astype(
            float
        )

    def _advance_succession_phase(self):
        self.active_phase_index = (self.active_phase_index + 1) % len(
            self.succession_phases
        )
        self.current_phase = self.succession_phases[self.active_phase_index]
        self.last_succession_step = self.timestep

    def _register_mutualism_events(self):
        co_location: Dict[Tuple[int, int], List[str]] = {}
        for agent_id in self.alive_agents:
            key = tuple(self.agent_positions[agent_id])
            co_location.setdefault(key, []).append(agent_id)

        for agents in co_location.values():
            if len(agents) < 2:
                continue
            species_present = {self.agent_species[a] for a in agents}
            if len(species_present) > 1:
                for species_id in species_present:
                    self.mutualism_events[species_id] += 1

    def _build_population_snapshot(self) -> Dict[str, Dict]:
        snapshot: Dict[str, Dict] = {}
        for species_id in self.species_catalog.keys():
            members = [
                agent_id
                for agent_id, sid in self.agent_species.items()
                if sid == species_id and agent_id in self.alive_agents
            ]
            if members:
                energies = [self.internal_states[a][0] for a in members]
                snapshot[species_id] = {
                    "count": len(members),
                    "avg_energy": float(np.mean(energies)),
                    "mutualism_events": self.mutualism_events.get(species_id, 0),
                }
            else:
                snapshot[species_id] = {
                    "count": 0,
                    "avg_energy": 0.0,
                    "mutualism_events": self.mutualism_events.get(species_id, 0),
                }
        return snapshot

    def _compose_global_info(
        self, stable: bool, metrics: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, float]:
        metrics = metrics or {}
        richness = metrics.get("species_richness", 0)
        trophic_stability = metrics.get("trophic_stability", 0.0)
        mutualism_events = metrics.get("mutualism_events", 0)

        rolling_stability = trophic_stability
        if self.trophic_window:
            rolling_stability = float(
                sum(self.trophic_window) / len(self.trophic_window)
            )

        return {
            "species_richness": richness,
            "trophic_stability": trophic_stability,
            "trophic_stability_rolling": rolling_stability,
            "mutualism_events": float(mutualism_events),
            "succession_phase": self.active_phase_index,
            "succession_phase_name": self.current_phase.get("name", "unknown"),
            "population_stable": float(stable),
        }

    def _is_alive(self, agent_id):
        return agent_id in self.alive_agents

    def render(self, mode="human"):
        if mode == "human":
            grid = np.zeros(self.grid_size, dtype=str)
            grid[:, :] = "."
            grid[self.food_map > 0] = "F"
            hazard_mask = self.hazard_map > 0
            grid[hazard_mask] = "H"
            for agent_id, pos in self.agent_positions.items():
                if self._is_alive(agent_id):
                    grid[pos[0], pos[1]] = self.agent_species[agent_id][0].upper()
            print("\n" + "\n".join(" ".join(row) for row in grid))
        else:
            raise NotImplementedError("RGB rendering is not implemented for GridLifeMA")
