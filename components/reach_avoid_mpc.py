import torch
import numpy as np

class ReachAvoidMPC:
    def __init__(self, latent_world_model, internal_model, viability_approximator, action_space,
                 horizon=12, num_candidates=1000, top_k=100, iterations=10):
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

    def _evaluate_trajectories(self, initial_latent_state, initial_internal_state, action_sequences):
        """
        Evaluates a batch of action sequences and returns their costs.
        """
        batch_size = action_sequences.shape[0]
        latent_state = initial_latent_state.repeat(batch_size, 1)
        internal_state = initial_internal_state.repeat(batch_size, 1)
        total_cost = torch.zeros(batch_size, device=latent_state.device)

        # A simple target: move towards a high-integrity, stable state
        # This could be made more dynamic in a real application
        reach_target = torch.tensor([0.9, 0.5, 0.9], device=internal_state.device) # energy, temp, integrity

        for t in range(self.horizon):
            action = action_sequences[:, t, :]

            # Predict next states using the models
            next_latent_state = self.latent_world_model.transition_model(latent_state, action)
            # Note: The internal model expects the full state (obs+internal), but for planning,
            # we use the latent state as a proxy for the observation. This might require an adapter.
            # For now, we assume the internal model can work with the latent state or we create a proxy obs.
            # Let's use the latent state directly as a stand-in for the observation part of the state.
            proxy_obs = self.latent_world_model.decoder(next_latent_state).mean.detach() # Get a proxy observation

            # We need to combine proxy_obs and internal_state to feed the internal_model if it expects that
            # Assuming internal_model is adapted to take latent_state directly for simplicity here
            next_internal_state = self.internal_model.predict_next(internal_state, action, proxy_obs)

            # --- Calculate Costs ---
            # 1. Avoid Cost: Penalize proximity to the viability boundary
            # Higher margin is better, so we use negative margin as a cost
            margin = self.viability_approximator(next_internal_state)
            avoid_cost = -margin.squeeze()

            # 2. Reach Cost: Reward getting closer to the target internal state
            reach_cost = torch.linalg.norm(next_internal_state - reach_target, dim=-1)

            # Combine costs (weights can be tuned)
            total_cost += 1.0 * avoid_cost + 0.5 * reach_cost

            latent_state = next_latent_state
            internal_state = next_internal_state

        return total_cost

    def plan_action(self, state):
        """
        Plans the best action using Cross-Entropy Method (CEM).
        """
        # Unpack state
        obs, internal_x = state['obs'], state['internal']
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        internal_tensor = torch.from_numpy(internal_x).float().unsqueeze(0)

        # Get initial latent state from the encoder
        with torch.no_grad():
            initial_latent = self.latent_world_model.encoder(obs_tensor)

        # Initialize the distribution for action sequences (Gaussian)
        action_mean = torch.zeros(self.horizon, self.action_dim)
        action_std = torch.ones(self.horizon, self.action_dim)

        for _ in range(self.iterations):
            # Sample action sequences from the current distribution
            action_sequences = torch.normal(mean=action_mean.unsqueeze(0).repeat(self.num_candidates, 1, 1),
                                            std=action_std.unsqueeze(0).repeat(self.num_candidates, 1, 1))
            action_sequences = torch.clamp(action_sequences,
                                           torch.from_numpy(self.action_space.low),
                                           torch.from_numpy(self.action_space.high))

            # Evaluate the sampled action sequences
            with torch.no_grad():
                costs = self._evaluate_trajectories(initial_latent, internal_tensor, action_sequences)

            # Select the top-k best sequences
            _, top_indices = torch.topk(costs, self.top_k, largest=False)
            best_sequences = action_sequences[top_indices]

            # Refit the distribution to the best sequences
            action_mean = best_sequences.mean(dim=0)
            action_std = best_sequences.std(dim=0) + 1e-6 # Add epsilon for stability

        # The best action is the first action of the best-found sequence
        best_action = action_mean[0].cpu().numpy()

        return np.clip(best_action, self.action_space.low, self.action_space.high)