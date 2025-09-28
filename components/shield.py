import numpy as np
import torch
import logging
from .safety_network import SafetyNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Shield initialized in '{self.mode}' mode with confidence {self.conf}.")

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
        logger.debug(f"Shield checking action: {action} for state: {s}")
        # If the original action is already safe, no need to project
        if self.is_safe(s, action):
            logger.debug("Action is safe. No projection needed.")
            return action

        logger.info(f"Unsafe action detected: {action}. Projecting...")
        # If in amortized mode and the network is available
        if self.mode == 'amortized' and self.safety_network is not None:
            logger.debug("Using amortized projection.")
            with torch.no_grad():
                internal_state_tensor = torch.from_numpy(s).float().unsqueeze(0)
                action_tensor = torch.from_numpy(action).float().unsqueeze(0)
                projected_action_tensor = self.safety_network(internal_state_tensor, action_tensor)
            projected_action = projected_action_tensor.squeeze(0).cpu().numpy()

            if self.is_safe(s, projected_action):
                logger.info(f"Amortized projection successful. New action: {projected_action}")
                return projected_action
            else:
                logger.warning("Amortized projection failed (produced unsafe action). Falling back to search.")
                projected_action = self._project_search(s, action)
                logger.info(f"Search-based fallback produced action: {projected_action}")
                return projected_action

        # If not in amortized mode, use the search-based projection
        else:
            logger.debug("Using search-based projection.")
            projected_action = self._project_search(s, action)
            logger.info(f"Search-based projection produced action: {projected_action}")
            return projected_action