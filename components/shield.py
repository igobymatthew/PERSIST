import numpy as np
import torch

class Shield:
    def __init__(self, internal_model, viability_approximator, action_space, conf=0.95):
        self.internal_model = internal_model
        self.viability_approximator = viability_approximator
        self.action_space = action_space
        self.conf = conf  # Confidence threshold for safety

    def is_safe(self, x, action):
        # Predict next internal state
        predicted_next_x = self.internal_model.predict_next(x, action)
        # Get safety margin from viability approximator
        margin = self.viability_approximator.get_margin(predicted_next_x)
        return margin >= self.conf

    def project(self, s, action):
        """
        Projects an action to a safe alternative if the original is unsafe.
        `s` is the full state [o, x], but we only need the internal part `x`.
        """
        # The training loop will be responsible for passing the internal state `x`.
        if self.is_safe(s, action):
            return action
        else:
            # Projection: search for a safe action that is close to the original.
            # This is a simplified version of a projection. A real implementation
            # would use an optimization method like CEM or gradient-based search.
            # Here, we do a small local search around the original action.
            for i in range(10):
                # Generate a small perturbation
                noise = np.random.randn(*action.shape) * 0.1 * (i + 1)
                perturbed_action = np.clip(action + noise, self.action_space.low, self.action_space.high)

                if self.is_safe(s, perturbed_action):
                    return perturbed_action

            # Fallback: if local search fails, try a few random actions.
            for _ in range(5):
                random_action = self.action_space.sample()
                if self.is_safe(s, random_action):
                    return random_action

            # If all else fails, return a default "safe" action (e.g., do nothing).
            return np.zeros_like(action)