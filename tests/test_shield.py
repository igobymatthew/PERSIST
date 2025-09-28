import unittest
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box

from components.shield import Shield

# --- Mock Components for Controlled Testing ---

class MockInternalModel:
    """A simple, predictable internal model for testing."""
    def predict_next(self, x, a):
        # A simple dynamic: next state is current state + action
        # This is predictable and easy to reason about.
        return x + a

class MockViabilityApproximator(nn.Module):
    """
    A mock viability approximator with a configurable "safe" zone.
    It considers a state "viable" if all its elements are positive.
    """
    def get_margin(self, x):
        # If all elements of the state are positive, margin is 1.0 (safe).
        # Otherwise, margin is 0.0 (unsafe).
        # This creates a clear boundary for testing.
        margins = torch.all(x > 0, dim=-1, keepdim=True).float()
        return margins

class MockSafetyNetwork(nn.Module):
    """A mock safety network that always returns a fixed 'safe' action."""
    def __init__(self, safe_action):
        super().__init__()
        self.safe_action = torch.tensor(safe_action, dtype=torch.float32)

    def forward(self, internal_state, action):
        # Ignores the input and always returns the pre-defined safe action.
        return self.safe_action.unsqueeze(0)

# --- Test Suite for the Safety Shield ---

class TestShield(unittest.TestCase):

    def setUp(self):
        """Set up the shield and its mock components for each test."""
        self.internal_model = MockInternalModel()
        self.viability_approximator = MockViabilityApproximator()
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Default shield in search mode
        self.shield = Shield(
            internal_model=self.internal_model,
            viability_approximator=self.viability_approximator,
            action_space=self.action_space,
            conf=0.9, # Margin must be >= 0.9 to be safe
            mode='search'
        )

    def test_safe_action_is_not_changed(self):
        """
        Tests that an action that is already safe is returned unchanged.
        """
        # A state where all values are positive
        current_state = np.array([0.5, 0.5])
        # An action that keeps the state positive
        safe_action = np.array([0.1, 0.1], dtype=np.float32)

        # The shield should determine this action is safe and not change it
        projected_action = self.shield.project(current_state, safe_action)

        self.assertTrue(np.array_equal(projected_action, safe_action),
                        "Shield should not modify an already safe action.")

    def test_unsafe_action_is_projected(self):
        """
        Tests that an unsafe action is modified by the shield.
        """
        # A state where a negative action would make it non-viable
        current_state = np.array([0.1, 0.1])
        # An action that makes the state non-viable ([0.1 - 0.5, 0.1 - 0.5] -> [-0.4, -0.4])
        unsafe_action = np.array([-0.5, -0.5], dtype=np.float32)

        # The shield should detect this is unsafe and return a different action
        projected_action = self.shield.project(current_state, unsafe_action)

        self.assertFalse(np.array_equal(projected_action, unsafe_action),
                         "Shield should modify an unsafe action.")

        # Verify that the action it returned is actually safe
        next_state_after_projection = current_state + projected_action
        self.assertTrue(np.all(next_state_after_projection > 0),
                        "The projected action must result in a viable state.")

    def test_amortized_projection_returns_network_action(self):
        """
        Tests that the shield in 'amortized' mode uses the safety network.
        """
        # A known safe action that the mock network will return
        known_safe_action = np.array([0.05, 0.05], dtype=np.float32)
        safety_network = MockSafetyNetwork(safe_action=known_safe_action)

        self.shield.safety_network = safety_network
        self.shield.mode = 'amortized'

        current_state = np.array([0.1, 0.1])
        unsafe_action = np.array([-0.5, -0.5], dtype=np.float32)

        projected_action = self.shield.project(current_state, unsafe_action)

        self.assertTrue(np.array_equal(projected_action, known_safe_action),
                        "Amortized shield should use the safety network's output.")

    def test_amortized_fallback_to_search(self):
        """
        Tests that the amortized shield falls back to search if the network's
        action is also unsafe.
        """
        # Configure the mock network to return an UNSAFE action
        unsafe_network_output = np.array([-0.2, -0.2], dtype=np.float32)
        safety_network = MockSafetyNetwork(safe_action=unsafe_network_output)

        self.shield.safety_network = safety_network
        self.shield.mode = 'amortized'

        current_state = np.array([0.1, 0.1])
        unsafe_action = np.array([-0.5, -0.5], dtype=np.float32) # The original unsafe action

        # The network will return [-0.2,-0.2], which is also unsafe (0.1-0.2 = -0.1).
        # The shield should detect this and fall back to its search method.
        projected_action = self.shield.project(current_state, unsafe_action)

        # The final action should not be the network's unsafe output
        self.assertFalse(np.array_equal(projected_action, unsafe_network_output),
                         "Shield should not use an unsafe action from the network.")

        # The final action should also not be the original unsafe action
        self.assertFalse(np.array_equal(projected_action, unsafe_action),
                         "Shield should not use the original unsafe action.")

        # And the final action must be safe
        next_state_after_projection = current_state + projected_action
        self.assertTrue(np.all(next_state_after_projection > 0),
                        "The fallback projected action must result in a viable state.")


if __name__ == "__main__":
    unittest.main()