import torch
import torch.nn as nn
import numpy as np

class DegradedModePolicy(nn.Module):
    """
    A simple, safe fallback policy for when the main policy fails to load.

    This policy can be configured to perform a simple, predictable action,
    such as doing nothing or taking a random action, to ensure system stability
    until the main policy can be restored.
    """
    def __init__(self, action_space, mode="do_nothing"):
        """
        Initializes the DegradedModePolicy.

        Args:
            action_space: The environment's action space (e.g., from gym.spaces).
            mode (str): The behavior of the policy. Options:
                        - "do_nothing": Always output a zero action.
                        - "random": Sample a random action from the action space.
        """
        super().__init__()
        self.action_space = action_space
        self.mode = mode

        if self.mode not in ["do_nothing", "random"]:
            raise ValueError(f"Invalid mode for DegradedModePolicy: {self.mode}")

    def forward(self, state):
        """
        Outputs a safe action based on the configured mode.

        Args:
            state: The current state (unused, but required for API consistency).

        Returns:
            A torch.Tensor representing the action to take.
        """
        if self.mode == "do_nothing":
            action = np.zeros(self.action_space.shape)
        elif self.mode == "random":
            action = self.action_space.sample()

        return torch.from_numpy(action).float()

    def sample(self, state):
        """
        Provides a consistent sampling method for the policy.
        """
        return self.forward(state)

def get_degraded_mode_policy(action_space, config):
    """
    Factory function for creating a degraded mode policy from a config.
    """
    mode = config.get("degraded_mode", {}).get("mode", "do_nothing")
    return DegradedModePolicy(action_space, mode)