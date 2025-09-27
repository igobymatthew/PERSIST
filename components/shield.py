import numpy as np
import torch
from .safety_network import SafetyNetwork

class Shield:
    def __init__(self, internal_model, viability_approximator, action_space, conf=0.95, safety_network=None, mode='search'):
        """
        Initializes the Shield.

        Args:
            internal_model: The model that predicts the next internal state.
            viability_approximator: The model that predicts the safety margin.
            action_space: The environment's action space.
            conf (float): The confidence threshold for an action to be considered safe.
            safety_network (SafetyNetwork, optional): The learned network for amortized projection.
            mode (str): 'search' for data collection, 'amortized' to use the network.
        """
        self.internal_model = internal_model
        self.viability_approximator = viability_approximator
        self.action_space = action_space
        self.conf = conf
        self.safety_network = safety_network
        self.mode = mode

    def is_safe(self, x, action):
        """
        Checks if an action is safe given the current internal state.
        """
        # Ensure inputs are torch tensors
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().unsqueeze(0)

        # Predict next internal state and get safety margin
        with torch.no_grad():
            predicted_next_x = self.internal_model.predict_next(x, action)
            margin = self.viability_approximator.get_margin(predicted_next_x)

        return margin.item() >= self.conf

    def _project_search(self, s, action):
        """
        Slow, search-based projection for finding a safe action.
        Used for initial data collection.
        """
        # Local search around the original action
        for i in range(10):
            noise = np.random.randn(*action.shape) * 0.1 * (i + 1)
            perturbed_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
            if self.is_safe(s, perturbed_action):
                return perturbed_action

        # Fallback to random search if local search fails
        for _ in range(5):
            random_action = self.action_space.sample()
            if self.is_safe(s, random_action):
                return random_action

        # If all else fails, return a default "do nothing" action
        return np.zeros_like(action)

    def project(self, s, action):
        """
        Projects a potentially unsafe action to a safe one.

        Returns the safe action. The original unsafe action must be stored
        by the caller if needed (e.g., for the replay buffer).
        """
        # If the original action is already safe, no need to project
        if self.is_safe(s, action):
            return action

        # If in amortized mode and the network is available
        if self.mode == 'amortized' and self.safety_network is not None:
            with torch.no_grad():
                internal_state_tensor = torch.from_numpy(s).float().unsqueeze(0)
                action_tensor = torch.from_numpy(action).float().unsqueeze(0)

                # Use the network to predict a safe action
                projected_action_tensor = self.safety_network(internal_state_tensor, action_tensor)

            projected_action = projected_action_tensor.squeeze(0).cpu().numpy()

            # As a final check, ensure the network's output is actually safe.
            # If not, fall back to the more reliable (but slower) search method.
            if self.is_safe(s, projected_action):
                return projected_action
            else:
                # Fallback to search if the network's suggestion is unsafe
                return self._project_search(s, action)

        # If not in amortized mode, use the search-based projection
        else:
            return self._project_search(s, action)