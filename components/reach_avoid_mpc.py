import torch
import numpy as np


class ReachAvoidMPC:
    def __init__(
        self,
        latent_world_model,
        internal_model,
        viability_approximator,
        action_space,
        horizon=12,
        num_candidates=1000,
        top_k=100,
        iterations=10,
    ):
        """
        Initializes the Reach-Avoid Model Predictive Control (MPC) component.

        Args:
            latent_world_model: The model for predicting future latent states.
            internal_model: The model for predicting future internal states.
            viability_approximator: The model for estimating the safety margin.
            action_space: The environment's action space.
            horizon (int): The planning horizon for MPC.
            num_candidates (int): The number of action sequences to sample at each iteration.
            top_k (int): The number of best action sequences to use for refitting the distribution.
            iterations (int): The number of optimization iterations (CEM).
        """
        self.latent_world_model = latent_world_model
        self.internal_model = internal_model
        self.viability_approximator = viability_approximator
        self.action_space = action_space
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.iterations = iterations
        self.action_dim = action_space.shape[0]

    def _evaluate_trajectories(
        self, initial_obs, initial_internal_state, action_sequences
    ):
        """Evaluates a batch of action sequences and returns their costs."""
        rollout = self.latent_world_model.rollout_viability(
            initial_obs=initial_obs,
            initial_internal_state=initial_internal_state,
            action_sequences=action_sequences,
            internal_model=self.internal_model,
            viability_approximator=self.viability_approximator,
        )

        margins = rollout["margins"]
        predicted_internal = rollout["predicted_internal_states"]
        reach_target = torch.tensor([0.9, 0.5, 0.9], device=predicted_internal.device)

        avoid_cost = -margins
        reach_cost = torch.linalg.norm(
            predicted_internal - reach_target.view(1, 1, -1), dim=-1
        )

        total_cost = (avoid_cost + 0.5 * reach_cost).sum(dim=1)
        return total_cost

    def plan_action(self, state):
        """
        Plans the best action using Cross-Entropy Method (CEM).
        """
        # Unpack state
        obs, internal_x = state["obs"], state["internal"]
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        internal_tensor = torch.from_numpy(internal_x).float().unsqueeze(0)

        # Get initial latent state from the encoder
        # Initialize the distribution for action sequences (Gaussian)
        action_mean = torch.zeros(self.horizon, self.action_dim)
        action_std = torch.ones(self.horizon, self.action_dim)

        for _ in range(self.iterations):
            # Sample action sequences from the current distribution
            action_sequences = torch.normal(
                mean=action_mean.unsqueeze(0).repeat(self.num_candidates, 1, 1),
                std=action_std.unsqueeze(0).repeat(self.num_candidates, 1, 1),
            )
            action_sequences = torch.clamp(
                action_sequences,
                torch.from_numpy(self.action_space.low),
                torch.from_numpy(self.action_space.high),
            )

            # Evaluate the sampled action sequences
            with torch.no_grad():
                costs = self._evaluate_trajectories(
                    obs_tensor, internal_tensor, action_sequences
                )

            # Select the top-k best sequences
            _, top_indices = torch.topk(costs, self.top_k, largest=False)
            best_sequences = action_sequences[top_indices]

            # Refit the distribution to the best sequences
            action_mean = best_sequences.mean(dim=0)
            action_std = best_sequences.std(dim=0) + 1e-6  # Add epsilon for stability

        # The best action is the first action of the best-found sequence
        best_action = action_mean[0].cpu().numpy()

        return np.clip(best_action, self.action_space.low, self.action_space.high)
