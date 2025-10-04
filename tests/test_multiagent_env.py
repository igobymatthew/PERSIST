import unittest
import numpy as np
from environments.multi_agent_gridlife import MultiAgentGridLifeEnv

def create_test_config(num_agents=2):
    """Creates a default config for testing the multi-agent environment."""
    return {
        'multiagent': {
            'num_agents': num_agents,
            'observation': {'view_radius': 2},
            'termination': {'max_steps': 100}
        },
        'agent_types': {
            'default': {
                'homeostasis': {'mu': [1.0, 0.5, 1.0]} # energy, temp, integrity
            }
        },
        'resource_model': {
            'food': {'initial_density': 0.1}
        }
    }

class TestMultiAgentEnv(unittest.TestCase):

    def setUp(self):
        """Set up a new environment for each test."""
        self.config = create_test_config(num_agents=2)
        self.env = MultiAgentGridLifeEnv(config=self.config)
        self.obs, self.info = self.env.reset()

    def test_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertEqual(self.env.num_agents, 2)
        self.assertEqual(len(self.env.agents), 2)
        self.assertIn('agent_0', self.env.observation_space.spaces)
        self.assertIn('agent_1', self.env.action_space.spaces)
        self.assertEqual(len(self.obs), 2, "Reset should return observations for all agents.")

    def test_step_functionality(self):
        """Test a single step in the environment."""
        actions = {
            'agent_0': np.array([1, 0]), # Move right
            'agent_1': np.array([0, 1])  # Move down
        }
        initial_pos_0 = self.env.agent_positions['agent_0'].copy()

        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self.assertIn('agent_0', obs)
        self.assertIn('agent_0', rewards)
        self.assertFalse(terminations['__all__'])
        self.assertFalse(truncations['__all__'])

        # Check if agent 0 moved as expected
        expected_pos_0 = np.clip(initial_pos_0 + actions['agent_0'], 0, self.env.grid_size[0] - 1)
        self.assertTrue(np.array_equal(self.env.agent_positions['agent_0'], expected_pos_0.astype(int)))

        # Check that energy decreased
        self.assertLess(self.env.internal_states['agent_0'][0], self.config['agent_types']['default']['homeostasis']['mu'][0])

    def test_agent_termination(self):
        """Test that an agent is terminated and removed correctly."""
        # Ensure agent_0 is not on a food tile to prevent energy gain
        agent_pos = self.env.agent_positions['agent_0']
        if self.env.food_map[agent_pos[0], agent_pos[1]] > 0:
            self.env.food_map[agent_pos[0], agent_pos[1]] = 0

        # Set energy to the exact decay amount, so it becomes 0 after one step
        self.env.internal_states['agent_0'][0] = self.env.energy_decay

        actions = {'agent_0': np.array([0, 0]), 'agent_1': np.array([0, 0])}

        # The step will cause energy to drop to exactly 0
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self.assertTrue(terminations.get('agent_0', False), "Agent 0 should be terminated.")
        self.assertNotIn('agent_0', self.env.alive_agents, "Agent 0 should be removed from alive agents.")
        self.assertNotIn('agent_0', obs, "Dead agents should not be in the next observation.")
        self.assertIn('agent_1', obs, "Agent 1 should still be in the observation.")

    def test_collision_observation(self):
        """Test that agents occupying the same cell are observed correctly."""
        # Force both agents to the same position
        self.env.agent_positions['agent_0'] = np.array([5, 5])
        self.env.agent_positions['agent_1'] = np.array([5, 5])

        # Create a third agent to be the observer
        self.config = create_test_config(num_agents=3)
        self.env = MultiAgentGridLifeEnv(config=self.config)
        self.env.reset()
        self.env.agent_positions['agent_0'] = np.array([5, 5])
        self.env.agent_positions['agent_1'] = np.array([5, 5])
        self.env.agent_positions['agent_2'] = np.array([4, 5]) # Observer is nearby

        obs_2 = self.env._get_obs_for_agent('agent_2')

        # The observer's vision should see 2 agents at position (5, 5)
        # The vision is egocentric, so we need to find the correct index.
        # Agent 2 is at (4,5), view_radius is 2. The vision grid is 5x5.
        # Center of vision is (2,2), which corresponds to (4,5) in world coords.
        # The location (5,5) should be at vision index (3,2).
        view_radius = self.config['multiagent']['observation']['view_radius']
        vision_center = np.array([view_radius, view_radius])
        target_vision_pos = vision_center + (np.array([5,5]) - self.env.agent_positions['agent_2'])

        agent_channel = 2 # food, hazard, agents
        agent_count = obs_2['vision'][target_vision_pos[0], target_vision_pos[1], agent_channel]

        self.assertEqual(agent_count, 2, "Observer should see 2 agents in the same cell.")

    def test_episode_truncation(self):
        """Test that the episode truncates after max_steps."""
        self.env.timestep = self.env.max_steps - 1
        actions = {'agent_0': np.array([0, 0]), 'agent_1': np.array([0, 0])}

        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        self.assertTrue(truncations['__all__'], "Episode should be truncated after max_steps.")

if __name__ == "__main__":
    unittest.main()