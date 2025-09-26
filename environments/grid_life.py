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

        self.food_pos = self._place_randomly()
        self.hot_pos = self._place_randomly()
        self.hazard_pos = self._place_randomly()

        self.action_dim = 2
        self.act_limit = 1.0
        self.action_space = MockActionSpace(low=-self.act_limit, high=self.act_limit, shape=(self.action_dim,))
        self.observation_space_dim = self.grid_size[0] * self.grid_size[1] + len(self.internal_state)

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

        # Check for constraint violations
        done = False
        constraints = self.config['viability']['constraints']

        # Simplified parsing of constraints from config
        # "energy >= 0.2"
        if self.internal_state[0] < 0.2: done = True
        # "temp in [0.3, 0.7]"
        if not (0.3 <= self.internal_state[1] <= 0.7): done = True
        # "integrity >= 0.6"
        if self.internal_state[2] < 0.6: done = True

        return self._get_obs(), task_reward, done, {}