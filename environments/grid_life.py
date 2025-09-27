import numpy as np

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

        self.dim_map = {name: i for i, name in enumerate(self.config['internal_state']['dims'])}
        self.constraints = self._parse_constraints(self.config['viability']['constraints'])

        self.food_pos = self._place_randomly()
        self.hot_pos = self._place_randomly()
        self.hazard_pos = self._place_randomly()

        self.action_dim = 2
        self.act_limit = 1.0
        self.action_space = MockActionSpace(low=-self.act_limit, high=self.act_limit, shape=(self.action_dim,))
        self.observation_space_dim = self.grid_size[0] * self.grid_size[1] + len(self.internal_state)

    def _parse_constraints(self, constraint_strings):
        """ Parses constraint strings from the config file into a structured format. """
        parsed = []
        for s in constraint_strings:
            parts = s.split()
            dim_name, op = parts[0], parts[1]

            if dim_name == 'temp' and op == 'in':
                min_val = float(parts[2].strip('[],'))
                max_val = float(parts[3].strip('[]'))
                parsed.append({'dim_idx': self.dim_map['temp'], 'op': '>=', 'val': min_val, 'name': 'temp_min'})
                parsed.append({'dim_idx': self.dim_map['temp'], 'op': '<=', 'val': max_val, 'name': 'temp_max'})
            else:
                val = float(parts[2])
                parsed.append({'dim_idx': self.dim_map[dim_name], 'op': op, 'val': val, 'name': dim_name})
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
        return self._get_obs()

    def _get_obs(self):
        grid = np.zeros(self.grid_size)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.food_pos[0], self.food_pos[1]] = 2
        grid[self.hot_pos[0], self.hot_pos[1]] = 3
        grid[self.hazard_pos[0], self.hazard_pos[1]] = 4
        return np.concatenate([grid.flatten(), self.internal_state])

    def step(self, action):
        # Move agent
        action = np.clip(action, -self.act_limit, self.act_limit)
        self.agent_pos = np.clip(self.agent_pos + action, [0, 0], [self.grid_size[0] - 1, self.grid_size[1] - 1]).astype(int)

        # Update internal state
        self.internal_state[0] -= 0.01  # Energy decay

        task_reward = 0

        if np.array_equal(self.agent_pos, self.food_pos):
            self.internal_state[0] = min(1.0, self.internal_state[0] + 0.5)
            self.food_pos = self._place_randomly()
            task_reward = 1.0

        if np.array_equal(self.agent_pos, self.hot_pos):
            self.internal_state[1] += 0.1

        if np.array_equal(self.agent_pos, self.hazard_pos):
            self.internal_state[2] -= 0.2

        # Check for constraint violations using dynamic constraints
        done = False
        info = {'violation': False}
        for c in self.constraints:
            val = self.internal_state[c['dim_idx']]
            op = c['op']
            threshold = c['val']

            if op == '>=' and not val >= threshold:
                done = True
            elif op == '<=' and not val <= threshold:
                done = True

            if done:
                info['violation'] = True
                break

        return self._get_obs(), task_reward, done, info