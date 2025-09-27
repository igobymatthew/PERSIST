import numpy as np

class MetaLearner:
    """
    Adapts homeostatic setpoints (mu) based on long-term agent performance.
    """
    def __init__(self, initial_mu, learning_rate=0.01, update_frequency=100):
        """
        Initializes the MetaLearner.

        Args:
            initial_mu (np.ndarray): The initial homeostatic setpoints.
            learning_rate (float): The learning rate for updating mu.
            update_frequency (int): The number of episodes between updates.
        """
        self.mu = np.array(initial_mu, dtype=np.float32)
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._cumulative_internal_state = np.zeros_like(self.mu)

    def step(self, reward, internal_state):
        """
        Records performance metrics from a single step.

        Args:
            reward (float): The total reward received in the step.
            internal_state (np.ndarray): The agent's internal state.
        """
        self._cumulative_reward += reward
        self._cumulative_internal_state += internal_state

    def episode_end(self):
        """
        Marks the end of an episode and triggers an update if needed.
        """
        self._episode_count += 1
        if self._episode_count % self.update_frequency == 0:
            self.update()
            self._reset_accumulators()

    def update(self):
        """
        Updates the homeostatic setpoints (mu) based on accumulated performance.

        The logic here is a simple heuristic: move mu towards the average internal
        state observed during high-reward episodes. This encourages the agent
        to consider recently successful states as the new "normal."
        """
        if self._episode_count > 0:
            # Calculate average reward and internal state over the update period
            avg_reward = self._cumulative_reward / self.update_frequency
            avg_internal_state = self._cumulative_internal_state / (self.update_frequency) # Assuming constant episode length for simplicity

            # Simple gradient-like update: move mu towards the average state
            # weighted by the learning rate.
            # A more sophisticated approach might use the reward as a weighting factor.
            direction = avg_internal_state - self.mu
            self.mu += self.learning_rate * direction

            print(f"MetaLearner: Updated mu to {self.mu} (avg reward: {avg_reward:.2f})")


    def get_setpoints(self):
        """
        Returns the current homeostatic setpoints.
        """
        return self.mu

    def _reset_accumulators(self):
        """
        Resets the tracked metrics.
        """
        self._cumulative_reward = 0.0
        self._cumulative_internal_state.fill(0)