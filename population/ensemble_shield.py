import torch
import torch.nn as nn
from typing import List
import numpy as np

class EnsembleShield:
    def __init__(self,
                 viability_models: List[nn.Module],
                 internal_model,
                 action_space,
                 vote_method: str = "veto_if_any_unsafe"):
        """
        Initializes the EnsembleShield, acting as a drop-in for the standard Shield.

        Args:
            viability_models (List[nn.Module]): A list of trained viability models.
            internal_model: The model that predicts the next internal state.
            action_space: The environment's action space.
            vote_method (str): The method to aggregate votes ('veto_if_any_unsafe' or 'majority_vote').
        """
        self.viability_models = viability_models
        self.internal_model = internal_model
        self.action_space = action_space
        self.vote_method = vote_method
        if self.vote_method not in ["veto_if_any_unsafe", "majority_vote"]:
            raise ValueError(f"Unknown vote_method: {self.vote_method}")

    @torch.no_grad()
    def _check_margins(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get safety votes from all models."""
        # Assuming model returns a margin > 0 for safe states
        margins = torch.stack([model(x) for model in self.viability_models])
        is_safe_votes = (margins > 0).squeeze()
        return is_safe_votes

    def _aggregate_votes(self, votes: torch.Tensor) -> bool:
        """Aggregates boolean votes based on the configured method."""
        if self.vote_method == "veto_if_any_unsafe":
            return torch.all(votes).item()
        elif self.vote_method == "majority_vote":
            return torch.mean(votes.float()) >= 0.5
        return False

    def is_safe(self, x, action) -> bool:
        """
        Checks if an action is safe given the current internal state by consulting the ensemble.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().unsqueeze(0).to(next(self.internal_model.parameters()).device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().unsqueeze(0).to(next(self.internal_model.parameters()).device)

        predicted_next_x = self.internal_model.predict_next(x, action)
        safe_votes = self._check_margins(predicted_next_x)
        return self._aggregate_votes(safe_votes)

    def _project_search(self, s, action):
        """
        A simple search-based projection to find a safe action.
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
        Projects a potentially unsafe action to a safe one using the ensemble.
        """
        if self.is_safe(s, action):
            return action

        # Use the simple search-based projection
        # print("EnsembleShield: Action deemed unsafe. Projecting...")
        return self._project_search(s, action)