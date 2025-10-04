import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

class MultiAgentGridLifeEnv(gym.Env):
    """
    A multi-agent version of the GridLife environment.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "GridLifeMA-v0"}

    def __init__(self, config):
        self.config = config
        self.grid_size = (10, 10)
        self.num_agents = self.config['multiagent']['num_agents']
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Per-agent internal state management
        self.internal_states = {agent_id: np.array(self.config['agent_types']['default']['homeostasis']['mu']) for agent_id in self.agents}
        self.agent_positions = {agent_id: np.array([0, 0]) for agent_id in self.agents}
        self.alive_agents = set(self.agents)

        # Define single-agent observation and action spaces based on config
        self.single_action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Vision + internal state + neighbor features + time
        view_radius = self.config['multiagent']['observation']['view_radius']
        obs_shape = (2 * view_radius + 1, 2 * view_radius + 1)

        self.single_observation_space = Dict({
            "vision": Box(low=0, high=10, shape=obs_shape, dtype=np.float32),
            "x": Box(low=0, high=1, shape=(3,), dtype=np.float32), # energy, temp, integrity
            "neighbors": Box(low=0, high=self.num_agents, shape=(3,), dtype=np.float32), # count, competition, resource_density
            "time": Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # The full observation and action spaces are dictionaries keyed by agent ID
        self.observation_space = Dict({agent: self.single_observation_space for agent in self.agents})
        self.action_space = Dict({agent: self.single_action_space for agent in self.agents})

        self.timestep = 0
        self.max_steps = self.config['multiagent']['termination']['max_steps']
        self.energy_decay = 0.01

        # World state
        self.food_map = np.zeros(self.grid_size)
        self.hazard_map = np.zeros(self.grid_size)

        print(f"✅ MultiAgentGridLifeEnv initialized for {self.num_agents} agents.")


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0
        self.alive_agents = set(self.agents)

        # Reset agent positions and internal states
        for agent_id in self.agents:
            self.agent_positions[agent_id] = self.np_random.integers(0, self.grid_size[0], size=2)
            self.internal_states[agent_id] = np.array(self.config['agent_types']['default']['homeostasis']['mu'])

        # Reset world resources
        self._regenerate_resources()

        observations = {agent_id: self._get_obs_for_agent(agent_id) for agent_id in self.alive_agents}
        infos = {agent_id: {} for agent_id in self.alive_agents}

        return observations, infos

    def step(self, action_dict):
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        terminations = {agent_id: False for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}

        # --- 1. Apply actions and update states ---
        for agent_id, action in action_dict.items():
            if agent_id not in self.alive_agents:
                continue

            # Move agent
            self.agent_positions[agent_id] = np.clip(self.agent_positions[agent_id] + action, [0, 0], [self.grid_size[0] - 1, self.grid_size[1] - 1]).astype(int)

            # Update internal state (energy decay)
            self.internal_states[agent_id][0] -= self.energy_decay

            # Check for resource consumption
            pos = self.agent_positions[agent_id]
            if self.food_map[pos[0], pos[1]] > 0:
                self.internal_states[agent_id][0] = min(1.0, self.internal_states[agent_id][0] + 0.5)
                self.food_map[pos[0], pos[1]] = 0 # Consume food
                rewards[agent_id] += 1.0

            # Check for termination
            if self.internal_states[agent_id][0] <= 0:
                terminations[agent_id] = True

        # --- 2. Remove dead agents ---
        newly_dead = set()
        for agent_id in self.alive_agents:
            if terminations[agent_id]:
                newly_dead.add(agent_id)
        self.alive_agents -= newly_dead

        # --- 3. Regenerate resources and advance time ---
        self._regenerate_resources()
        self.timestep += 1

        # --- 4. Get next observations and infos ---
        observations = {agent_id: self._get_obs_for_agent(agent_id) for agent_id in self.alive_agents}
        for agent_id in self.alive_agents:
            infos[agent_id]['internal_state'] = self.internal_states[agent_id]

        # --- 5. Set global termination/truncation flags ---
        truncations = {agent_id: False for agent_id in self.agents}
        if self.timestep >= self.max_steps:
            truncations["__all__"] = True
        else:
            truncations["__all__"] = False

        if not self.alive_agents:
            terminations["__all__"] = True
        else:
            terminations["__all__"] = False

        return observations, rewards, terminations, truncations, infos

    def _get_obs_for_agent(self, agent_id):
        """Generates a full observation for a single agent."""
        pos = self.agent_positions[agent_id]
        view_radius = self.config['multiagent']['observation']['view_radius']

        # 1. Vision: Egocentric view
        # Create a padded world state to handle edge cases smoothly
        padded_world = self._get_padded_world_state()

        # Crop the padded world state to get the egocentric view
        r_start, c_start = pos[0], pos[1]
        vision = padded_world[r_start:r_start + 2 * view_radius + 1, c_start:c_start + 2 * view_radius + 1, :]

        # 2. Internal State
        internal_state = self.internal_states[agent_id]

        # 3. Neighbor features (placeholder for now)
        neighbor_features = np.zeros(3)

        # 4. Time
        time = np.array([self.timestep / self.max_steps])

        return {
            "vision": vision.astype(np.float32),
            "x": internal_state.astype(np.float32),
            "neighbors": neighbor_features.astype(np.float32),
            "time": time.astype(np.float32)
        }

    def _get_padded_world_state(self):
        """Helper to create a padded, multi-channel representation of the world."""
        view_radius = self.config['multiagent']['observation']['view_radius']

        # Create a channel for agent positions
        agent_pos_map = np.zeros(self.grid_size)
        for other_agent_id, other_pos in self.agent_positions.items():
            if self._is_alive(other_agent_id):
                agent_pos_map[other_pos[0], other_pos[1]] += 1

        # Stack all channels
        world_state = np.stack([self.food_map, self.hazard_map, agent_pos_map], axis=-1)

        # Pad the world state
        padding = ((view_radius, view_radius), (view_radius, view_radius), (0, 0))
        padded_world = np.pad(world_state, padding, mode='constant', constant_values=0)
        return padded_world

    def _regenerate_resources(self):
        # Placeholder for resource regeneration logic
        food_density = self.config['resource_model']['food']['initial_density']
        self.food_map = (self.np_random.random(self.grid_size) < food_density).astype(float)

    def _is_alive(self, agent_id):
        return agent_id in self.alive_agents

    def render(self, mode="human"):
        if mode == "human":
            grid = np.zeros(self.grid_size, dtype=str)
            grid[:, :] = '.'
            grid[self.food_map > 0] = 'F'

            for agent_id, pos in self.agent_positions.items():
                if self._is_alive(agent_id):
                    grid[pos[0], pos[1]] = str(agent_id[-1]) # Display agent number

            print("\n" + "\n".join(" ".join(row) for row in grid))
        else:
            # For returning an RGB array
            pass