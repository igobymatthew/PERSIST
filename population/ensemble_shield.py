from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class EnsembleShield:
    def __init__(
        self,
        viability_models: List[nn.Module],
        internal_model,
        action_space,
        vote_method: str = "veto_if_any_unsafe",
    ):
        """Initializes the EnsembleShield, acting as a drop-in for the standard Shield."""

        self.viability_models = viability_models
        self.internal_model = internal_model
        self.action_space = action_space
        self.vote_method = vote_method
        if self.vote_method not in ["veto_if_any_unsafe", "majority_vote"]:
            raise ValueError(f"Unknown vote_method: {self.vote_method}")

    @torch.no_grad()
    def _check_margins(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get safety votes from all models."""
        margins = torch.stack([model(x) for model in self.viability_models])
        is_safe_votes = (margins > 0).squeeze()
        return is_safe_votes

    def _aggregate_votes(self, votes: torch.Tensor) -> bool:
        """Aggregates boolean votes based on the configured method."""
        if self.vote_method == "veto_if_any_unsafe":
            return torch.all(votes).item()
        if self.vote_method == "majority_vote":
            return torch.mean(votes.float()) >= 0.5
        return False

    def is_safe(self, x, action) -> bool:
        """Checks if an action is safe given the current internal state by consulting the ensemble."""
        if isinstance(x, np.ndarray):
            x = (
                torch.from_numpy(x)
                .float()
                .unsqueeze(0)
                .to(next(self.internal_model.parameters()).device)
            )
        if isinstance(action, np.ndarray):
            action = (
                torch.from_numpy(action)
                .float()
                .unsqueeze(0)
                .to(next(self.internal_model.parameters()).device)
            )

        predicted_next_x = self.internal_model.predict_next(x, action)
        safe_votes = self._check_margins(predicted_next_x)
        return self._aggregate_votes(safe_votes)

    def _project_search(self, s, action):
        """A simple search-based projection to find a safe action."""
        for i in range(10):
            noise = np.random.randn(*action.shape) * 0.1 * (i + 1)
            perturbed_action = np.clip(
                action + noise, self.action_space.low, self.action_space.high
            )
            if self.is_safe(s, perturbed_action):
                return perturbed_action

        for _ in range(5):
            random_action = self.action_space.sample()
            if self.is_safe(s, random_action):
                return random_action

        return np.zeros_like(action)

    def project(self, s, action):
        """Projects a potentially unsafe action to a safe one using the ensemble."""
        if self.is_safe(s, action):
            return action

        return self._project_search(s, action)


@dataclass
class PopulationSafetyCoordinator:
    """Evaluates biodiversity constraints at the population level."""

    species_metadata: Dict[str, Dict]
    trophic_default: float = 1.0

    def evaluate(self, snapshot: Dict[str, Dict]) -> Tuple[bool, Dict[str, Dict]]:
        """Assess whether the population snapshot respects species constraints."""

        details: Dict[str, Dict] = {}
        stable = True
        richness = 0

        for species_id, metrics in snapshot.items():
            count = int(metrics.get("count", 0))
            avg_energy = float(metrics.get("avg_energy", 0.0))
            population_cfg = self.species_metadata.get(species_id, {}).get(
                "population", {}
            )
            status = {
                "count": count,
                "avg_energy": avg_energy,
                "within_bounds": True,
            }

            if count > 0:
                richness += 1

            min_count = population_cfg.get("min_count")
            max_count = population_cfg.get("max_count")
            energy_floor = population_cfg.get("energy_floor")

            if min_count is not None and count < min_count:
                status["within_bounds"] = False
                status["reason"] = "count_below_min"
                stable = False
            elif max_count is not None and count > max_count:
                status["within_bounds"] = False
                status["reason"] = "count_above_max"
                stable = False
            elif energy_floor is not None and avg_energy < energy_floor:
                status["within_bounds"] = False
                status["reason"] = "energy_below_floor"
                stable = False

            details[species_id] = status

        details["species_richness"] = richness
        total_population = sum(int(m.get("count", 0)) for m in snapshot.values())

        if total_population > 0 and richness > 0:
            mean_share = 1.0 / richness
            share_var = 0.0
            for metrics in snapshot.values():
                share = int(metrics.get("count", 0)) / total_population
                share_var += (share - mean_share) ** 2
            trophic_stability = max(0.0, 1.0 - min(1.0, share_var))
        else:
            trophic_stability = self.trophic_default if richness > 0 else 0.0

        details["trophic_stability"] = trophic_stability
        mutualism_events = sum(
            int(m.get("mutualism_events", 0)) for m in snapshot.values()
        )
        details["mutualism_events"] = mutualism_events

        return stable, details
