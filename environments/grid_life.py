import numpy as np
from environments.maintenance_tasks import MaintenanceManager

# A simple mock for gym.spaces.Box to avoid adding a new dependency
class MockActionSpace:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)

class GridLifeEnv:
    def __init__(self, config):
        self.config = config
        self.grid_size = (10, 10)
        self.agent_pos = np.array([0, 0])

        # Internal state: energy, temp, integrity
        self.internal_state = np.array(self.config['internal_state']['mu'])
        self.internal_dim = len(self.internal_state)
        self.external_obs_dim = self.grid_size[0] * self.grid_size[1]

        # Partial observability setting
        self.partial_observability = self.config.get('env', {}).get('partial_observability', False)

        self.dim_map = {name: i for i, name in enumerate(self.config['internal_state']['dims'])}
        self.constraints = self._parse_constraints(self.config['viability']['constraints'])
        self.num_constraints = len(self.constraints)
        self.constraint_names = [c['name'] for c in self.constraints]

        self.food_pos = self._place_randomly()
        self.hot_pos = self._place_randomly()
        self.hazard_pos = self._place_randomly()
        self.fire_pos = self._place_randomly()

        # Maintenance tasks are now managed by a dedicated class
        self.maintenance_manager = MaintenanceManager(
            self.config.get('maintenance', {}), self.grid_size
        )

        self.action_dim = 2
        self.act_limit = 1.0
        self.action_space = MockActionSpace(low=-self.act_limit, high=self.act_limit, shape=(self.action_dim,))

        if self.partial_observability:
            self.observation_space_dim = self.external_obs_dim
        else:
            self.observation_space_dim = self.external_obs_dim + self.internal_dim

    def _parse_constraints(self, constraint_objects):
        """
        Parses structured constraint objects from the config file into a
        standardized internal format.
        """
        parsed = []
        for c_obj in constraint_objects:
            dim_name = c_obj['variable']
            op = c_obj['operator']
            threshold = c_obj['threshold']
            name = c_obj['name']

            if op == 'in':
                min_val, max_val = threshold[0], threshold[1]
                parsed.append({'dim_idx': self.dim_map[dim_name], 'op': '>=', 'val': min_val, 'name': f"{name}_min"})
                parsed.append({'dim_idx': self.dim_map[dim_name], 'op': '<=', 'val': max_val, 'name': f"{name}_max"})
            else:
                parsed.append({'dim_idx': self.dim_map[dim_name], 'op': op, 'val': threshold, 'name': name})
        return parsed

    def update_constraints(self, new_values):
        """
        Updates the environment's viability constraints dynamically.
        `new_values` is a dictionary where keys match the 'name' of the constraint.
        """
        for constraint in self.constraints:
            if constraint['name'] in new_values:
                constraint['val'] = new_values[constraint['name']]

    def _place_randomly(self):
        return np.random.randint(0, self.grid_size[0], size=2)

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.internal_state = np.array(self.config['internal_state']['mu'])
        self.food_pos = self._place_randomly()
        self.hot_pos = self._place_randomly()
        self.hazard_pos = self._place_randomly()
        self.fire_pos = self._place_randomly()
        self.maintenance_manager.reset()
        return self._get_obs()

    def _get_obs(self):
        grid = np.zeros(self.grid_size)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.food_pos[0], self.food_pos[1]] = 2
        grid[self.hot_pos[0], self.hot_pos[1]] = 3
        grid[self.hazard_pos[0], self.hazard_pos[1]] = 4
        grid[self.fire_pos[0], self.fire_pos[1]] = 8

        # Add maintenance stations to the grid observation
        station_positions = self.maintenance_manager.get_station_positions()
        if station_positions:
            station_map = {'refuel': 5, 'cool_down': 6, 'repair': 7}
            for station_type, pos in station_positions.items():
                if station_type in station_map:
                    grid[pos[0], pos[1]] = station_map[station_type]

        external_obs = grid.flatten()

        if self.partial_observability:
            return external_obs
        else:
            return np.concatenate([external_obs, self.internal_state])

    def step(self, action):
        action = np.clip(action, -self.act_limit, self.act_limit)
        self.agent_pos = np.clip(self.agent_pos + action, [0, 0], [self.grid_size[0] - 1, self.grid_size[1] - 1]).astype(int)

        self.internal_state[0] -= 0.01  # Energy decay
        task_reward = 0

        if np.array_equal(self.agent_pos, self.food_pos):
            self.internal_state[0] = min(1.0, self.internal_state[0] + 0.5)
            self.food_pos = self._place_randomly()
            task_reward = 1.0

        if np.array_equal(self.agent_pos, self.hot_pos):
            self.internal_state[1] += 0.1

        fire_triggered = False

        if np.array_equal(self.agent_pos, self.hazard_pos):
            self.internal_state[2] -= 0.2

        if np.array_equal(self.agent_pos, self.fire_pos):
            fire_triggered = True
            self.fire_pos = self._place_randomly()

        # Apply maintenance effects and penalties
        self.internal_state, maintenance_penalty = self.maintenance_manager.apply_maintenance(
            self.agent_pos, self.internal_state, self.config['internal_state']['mu']
        )
        task_reward -= maintenance_penalty

        done = False
        info = {}
        violations = np.zeros(self.num_constraints)
        for i, c in enumerate(self.constraints):
            val = self.internal_state[c['dim_idx']]
            op = c['op']
            threshold = c['val']

            is_violated = False
            if op == '>=' and not val >= threshold: is_violated = True
            elif op == '<=' and not val <= threshold: is_violated = True

            if is_violated:
                violations[i] = 1.0
                done = True

        info['violation'] = done
        info['internal_state_violation'] = violations

        info['fire_triggered'] = fire_triggered

        if self.partial_observability:
            info['internal_state'] = self.internal_state.copy()

        info['constraint_margins'] = self._calculate_constraint_margins()

        return self._get_obs(), task_reward, done, info

    def _calculate_constraint_margins(self):
        margins = np.zeros(self.num_constraints)
        for i, c in enumerate(self.constraints):
            val = self.internal_state[c['dim_idx']]
            op = c['op']
            threshold = c['val']

            if op == '>=': margins[i] = val - threshold
            elif op == '<=': margins[i] = threshold - val
        return margins